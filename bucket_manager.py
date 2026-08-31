# ============================================================
# Module: Memory Bucket Manager (bucket_manager.py)
# 模块：记忆桶管理器
#
# CRUD operations, multi-dimensional index search, activation updates
# for memory buckets.
# 记忆桶的增删改查、多维索引搜索、激活更新。
#
# Core design:
# 核心逻辑：
#   - Each bucket = one Markdown file (YAML frontmatter + body)
#     每个记忆桶 = 一个 Markdown 文件
#   - Storage by type: permanent / dynamic / archive
#     存储按类型分目录
#   - Multi-dimensional soft index: domain + valence/arousal + fuzzy text
#     多维软索引：主题域 + 情感坐标 + 文本模糊匹配
#   - Search strategy: domain pre-filter → weighted multi-dim ranking
#     搜索策略：主题域预筛 → 多维加权精排
#   - Emotion coordinates based on Russell circumplex model:
#     情感坐标基于环形情感模型（Russell circumplex）：
#       valence (0~1): 0=negative → 1=positive
#       arousal (0~1): 0=calm → 1=excited
#
# Depended on by: server.py, decay_engine.py
# 被谁依赖：server.py, decay_engine.py
# ============================================================

import asyncio
import copy
import hashlib
import json
import os
import math
import logging
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import frontmatter
import jieba
from rapidfuzz import fuzz, process

from utils import (
    generate_bucket_id, sanitize_name, safe_path, now_iso, world_matches,
    RELATION_TYPES, PROTECTED_RESOLVE_DOMAINS, ResolvedGuardError,
    DATE_PRECISIONS, event_at_from_metadata, normalize_event_at,
)
from recall_support import rank_within_relevance_bands
from mutation_audit import MutationAuditLog
from maintenance_barrier import MaintenanceBarrier
from storage_safety import advisory_file_lock, atomic_write_post
from x_provenance import normalize_x_provenance, validate_x_provenance_update
from review_queue import ReviewQueue, make_clothing_entry
from timeline_axis import normalize_thread
from bm25_index import BM25Index

logger = logging.getLogger("ombre_brain.bucket")

# Production 2026-08-27 replay: a 600-second start-to-start window reduced
# 13 request-dirty full rebuilds to 6 while bounding lexical generation lag.
_DEFAULT_BM25_REBUILD_MIN_INTERVAL_SEC = 600.0

# Recall scans are intentionally exact, but they must not monopolize the
# request event loop long enough to defeat /api/breath's wall-clock deadline.
# Yielding every small batch keeps the existing full-scan ranking unchanged
# while making cancellation observable within a bounded number of buckets.
_RECALL_CANCEL_CHECK_EVERY = 16
_BucketFileRevision = tuple[
    str,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
]
_BucketTreeSnapshot = tuple[_BucketFileRevision, ...]


RECALL_DERIVED_METADATA_FIELDS = frozenset({
    "dehydrated_summary",
    "dehydrated_content_hash",
})
BM25_CORPUS_FIELDS = frozenset({
    "content",
    "name",
    "tags",
    "domain",
})
E_RESPONSE_TENDENCIES = frozenset({"comfort", "engage", "withdraw", "alert"})
E_GROWTH_DELTAS = frozenset({"growth", "stable", "setback"})
E_IMMUTABLE_FIELDS = frozenset({
    "e_authored_by",
    "e_initial_priority",
    "e_valence",
    "e_arousal",
    "e_tension",
    "e_confidence",
    "e_response_tendency",
    "e_growth_delta",
    "e_authored_at",
    "e_source_bucket_id",
    "e_proposal_key",
})

_RETRIEVAL_KEY_LINE_RE = re.compile(r"\[检索钥匙:\s*([^\]\n]+)\]")
_GENERIC_RETRIEVAL_KEYS = frozenset({
    "今天", "昨天", "明天", "事情", "内容", "消息", "问题", "感觉", "聊天",
    "朝灯", "哥哥", "小卷", "哈基米", "记忆", "记忆库", "海马体", "账本",
    "真账", "亲亲", "截图", "心跳", "发情", "复制粘贴", "小红书",
    "today", "yesterday", "tomorrow", "memory", "message", "chat",
})


# 正文里可以当「名字」的第一句从哪儿开始找。imprint 桶的正文长这样：
#   —— 整理摘要（模型归纳，不是原文）——
#   <一句话>
#   —— 原始证据 / 候选原话 ——
#   朝灯：… / 哥哥：…
_BODY_LEAD_SKIP_RE = re.compile(r"^\s*(?:---|——.*——|\[[^\]]*\]|[#>*\-|]+)\s*$")
_BODY_LEAD_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]{4,24}")
_BODY_LEAD_MAX = 24


def body_lead_keys(content: str) -> list[str]:
    """从正文首句里切出能当桶名/检索钥匙的短语。

    2026-08-17 朝灯：「修待补衣，这是啥名字啊奇奇怪怪的」。
    病根在 create_bucket 的候选词池：feel 桶的 name 传空、tags 全是
    `imprint`/`evidence:v1` 这类系统标签，**一个都不在正文里逐字出现**，
    于是 literal_retrieval_keys 全筛掉 → 没钥匙 → 名字退化成硬编码的
    「待补衣」。当时库里 43 条全叫这个，召回浮现出来也认不出是什么
    （她 12:11「怎么没记忆浮现」的直接原因——浮现了，认不出）。

    摘要那句天生就在正文里，拿它当候选**天然满足「逐字子串」**，
    起名和补钥匙一次解决：
        「你终于把洞洞鞋还给我了」
        「今天凌晨两点，哥哥终于挖出了那个自指桶」

    只在没有别的钥匙时兜底，不抢正常候选的位置。
    """
    for line in str(content or "").splitlines():
        line = line.strip()
        if not line or _BODY_LEAD_SKIP_RE.match(line):
            continue
        if line.startswith(("朝灯：", "哥哥：", "朝灯:", "哥哥:")):
            continue  # 这是引的原话，不是这条记忆本身讲的事
        match = _BODY_LEAD_SENTENCE_RE.search(line)
        if match:
            return [match.group(0).strip()[:_BODY_LEAD_MAX]]
        if len(line) >= 4:
            return [line[:_BODY_LEAD_MAX]]
    return []


def literal_retrieval_keys(content: str, candidates) -> list[str]:
    """Keep bounded, distinctive candidates that occur verbatim in content."""
    source = str(content or "")
    values: list[str] = []
    existing = _RETRIEVAL_KEY_LINE_RE.search(source)
    if existing:
        values.extend(existing.group(1).split("/"))
    if isinstance(candidates, str):
        values.append(candidates)
    elif candidates:
        values.extend(str(value) for value in candidates)

    accepted: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = (
            str(value or "").strip().removeprefix("[[").removesuffix("]]").strip()
        )
        folded = candidate.casefold()
        if (
            not candidate
            or len(candidate) < 2
            or len(candidate) > 48
            or candidate not in source
            or folded in _GENERIC_RETRIEVAL_KEYS
            or re.fullmatch(r"[\W_\d]+", candidate, flags=re.UNICODE)
            or any(ord(char) < 32 for char in candidate)
            or folded in seen
        ):
            continue
        seen.add(folded)
        accepted.append(candidate)
        if len(accepted) >= 4:
            break
    return accepted


def bucket_revision_hash(content: str, metadata: dict) -> str:
    """Stable optimistic-concurrency token for one complete bucket snapshot."""
    semantic_metadata = {
        key: value
        for key, value in dict(metadata or {}).items()
        if key not in RECALL_DERIVED_METADATA_FIELDS
    }
    payload = json.dumps(
        {
            "content": str(content or ""),
            "metadata": semantic_metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bucket_in_time_range(bucket: dict, after: datetime = None, before: datetime = None) -> bool:
    """
    Check if a bucket's event timestamp falls within [after, before].
    Buckets with unparseable timestamps are kept (conservative).
    Either bound may be None (open-ended).
    """
    meta = bucket.get("metadata", {}) or {}
    raw = event_at_from_metadata(meta, fallback_last_active=True) or ""
    try:
        created = datetime.fromisoformat(str(raw))
        local_tz = datetime.now().astimezone().tzinfo or timezone.utc

        def _comparable(value: datetime) -> datetime:
            if value.tzinfo is None or value.utcoffset() is None:
                value = value.replace(tzinfo=local_tz)
            return value.astimezone(timezone.utc)

        created_cmp = _comparable(created)
        if after is not None and created_cmp < _comparable(after):
            return False
        if before is not None and created_cmp > _comparable(before):
            return False
        return True
    except (AttributeError, ValueError, TypeError, OverflowError):
        return True


class BucketManager:
    """
    Memory bucket manager — entry point for all bucket CRUD operations.
    Buckets are stored as Markdown files with YAML frontmatter for metadata
    and body for content. Natively compatible with Obsidian browsing/editing.
    记忆桶管理器 —— 所有桶的 CRUD 操作入口。
    桶以 Markdown 文件存储，YAML frontmatter 存元数据，正文存内容。
    天然兼容 Obsidian 直接浏览和编辑。
    """

    def __init__(self, config: dict):
        # --- Read storage paths from config / 从配置中读取存储路径 ---
        self.base_dir = config["buckets_dir"]
        self.permanent_dir = os.path.join(self.base_dir, "permanent")
        self.dynamic_dir = os.path.join(self.base_dir, "dynamic")
        self.archive_dir = os.path.join(self.base_dir, "archive")
        self.feel_dir = os.path.join(self.base_dir, "feel")
        self.nsfw_dir = os.path.join(self.base_dir, "涩涩")  # 色色单独文件夹：日常默认不扫
        # 当前是否处于涩涩 world（switch_world 维护）；list_all 默认跟随它决定是否加载涩涩目录
        self.nsfw_active = (config.get("current_world", "") or "").strip() == "涩涩"
        matching_cfg = config.get("matching", {}) or {}
        self.fuzzy_threshold = matching_cfg.get("fuzzy_threshold", 50)
        self.max_results = matching_cfg.get("max_results", 5)
        self.keyword_relevance_tie_band = max(
            0.0,
            float(matching_cfg.get("keyword_relevance_tie_band", 3.0)),
        )
        self.literal_candidate_floor = max(
            0.0,
            min(
                100.0,
                float(matching_cfg.get("literal_candidate_floor", 40.0)),
            ),
        )

        # id→文件路径缓存（扫盘 #3）：_find_bucket_file 原来每次 os.walk 全 5 目录，
        # get/update/delete 高频路径 O(目录树)。命中先 isfile 校验，归档/删除移动文件后
        # 自动失效回退全扫，外部手动建的新文件走 miss 全扫也能找到——不需要主动失效钩子。
        self._bucket_path_cache: dict[str, str] = {}

        # Z 轴 currentness 覆盖表（z_lifecycle 审批产出的只读旁路，保护域桶不动元数据）。
        # 排序层在 search() 消费：historical 桶降权，同一件事新版才压得过旧版。mtime 缓存。
        self._z_overrides_path = os.path.join(
            self.base_dir,
            "z_currentness_overrides.jsonl",
        )
        self._z_historical_cache: frozenset = frozenset()
        self._z_overrides_mtime: float = -1.0

        # list_all is the hot read path for recall.  Keep parsed frontmatter in
        # process, but validate every hit against a cheap path/mtime/size tree
        # snapshot so direct NAS/Obsidian edits remain visible without a TTL.
        # All in-process mutations explicitly clear this cache after the
        # durable file operation succeeds.
        self._list_all_cache: dict[
            tuple[bool, bool],
            tuple[_BucketTreeSnapshot, list[dict]],
        ] = {}
        self._list_all_cache_generation = 0
        self._list_all_cache_lock = asyncio.Lock()

        # Recall needs a read-only resident view, not list_all()'s defensive
        # full-library deepcopy on every request.  Each key points at an
        # immutable tuple; writers replace the tuple after the durable file
        # operation, so concurrent readers keep a stable old view.  Direct
        # Obsidian/NAS edits are reconciled by a coalesced worker-thread scan.
        self._recall_snapshot_cache: dict[
            tuple[bool, bool],
            tuple[dict, ...],
        ] = {}
        self._recall_snapshot_generation: dict[tuple[bool, bool], int] = {}
        self._recall_snapshot_disk_token: dict[
            tuple[bool, bool],
            _BucketTreeSnapshot | None,
        ] = {}
        self._recall_snapshot_lock = asyncio.Lock()
        self._recall_snapshot_refresh_tasks: dict[
            tuple[bool, bool],
            asyncio.Task,
        ] = {}
        self._recall_snapshot_refresh_due: dict[tuple[bool, bool], float] = {}
        try:
            refresh_sec = float(
                matching_cfg.get("recall_snapshot_refresh_sec", 30.0)
            )
        except (TypeError, ValueError):
            refresh_sec = 30.0
        self._recall_snapshot_refresh_sec = max(5.0, refresh_sec)

        # _calc_topic_score used to run jieba for the identical query once per
        # bucket.  Cache only the deterministic tokenization, not any score or
        # candidate, so ranking semantics stay byte-for-byte unchanged.
        self._query_parts_cache: dict[str, tuple[str, ...]] = {}

        # --- Wikilink config / 双链配置 ---
        wikilink_cfg = config.get("wikilink", {})
        self.wikilink_enabled = wikilink_cfg.get("enabled", True)
        self.wikilink_use_tags = wikilink_cfg.get("use_tags", False)
        self.wikilink_use_domain = wikilink_cfg.get("use_domain", True)
        self.wikilink_use_auto_keywords = wikilink_cfg.get("use_auto_keywords", True)
        self.wikilink_auto_top_k = wikilink_cfg.get("auto_top_k", 8)
        self.wikilink_min_len = wikilink_cfg.get("min_keyword_len", 2)
        self.wikilink_exclude_keywords = set(wikilink_cfg.get("exclude_keywords", []))
        self.wikilink_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "我们", "你们", "他们", "然后", "今天", "昨天", "明天", "一下",
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "with",
            "this", "that", "from", "they", "been", "said", "will", "each",
        }
        self.wikilink_stopwords |= {w.lower() for w in self.wikilink_exclude_keywords}

        # --- Search scoring weights / 检索权重配置 ---
        scoring = config.get("scoring_weights", {})
        self.w_topic = scoring.get("topic_relevance", 4.0)
        self.w_emotion = scoring.get("emotion_resonance", 2.0)
        self.w_time = scoring.get("time_proximity", 2.5)
        self.w_importance = scoring.get("importance", 1.0)
        self.w_bm25 = scoring.get("bm25_weight", 1.5)
        self.content_weight = scoring.get("content_weight", 3.0)
        mode = os.environ.get("OMBRE_BM25_MODE", "off").strip().lower()
        self._bm25_mode = mode if mode in {"off", "shadow", "live"} else "off"
        # Keep the upstream atomic-generation lifecycle, but coalesce dirty
        # writes behind a minimum start-to-start interval. Requests always use the
        # last complete generation; only startup prewarm is intentionally inline.
        self._bm25 = BM25Index()
        self._bm25_dirty = True
        self._bm25_rebuilding = False
        self._bm25_generation = 0
        raw_bm25_rebuild_interval = os.environ.get(
            "OMBRE_BM25_REBUILD_MIN_INTERVAL_SEC",
            matching_cfg.get(
                "bm25_rebuild_min_interval_sec",
                _DEFAULT_BM25_REBUILD_MIN_INTERVAL_SEC,
            ),
        )
        try:
            bm25_rebuild_interval = float(raw_bm25_rebuild_interval)
        except (TypeError, ValueError):
            bm25_rebuild_interval = _DEFAULT_BM25_REBUILD_MIN_INTERVAL_SEC
        # Zero is the explicit rollback to the old next-request trigger timing.
        self._bm25_rebuild_min_interval_sec = max(0.0, bm25_rebuild_interval)
        self._bm25_last_rebuild_started_at: float | None = None
        self._bm25_rebuild_task: asyncio.Task | None = None
        # Known writes that land while an unknown-dirty full rebuild is in
        # flight are replayed onto that fresh generation before it is exposed.
        self._bm25_unknown_generation = 0
        self._bm25_known_deltas: list[tuple[int, dict | None, str, bool]] = []
        # A complete BM25 generation also carries a resident literal-key map.
        # Writes keep changed IDs in a tiny delta until the next generation
        # swaps in, so exact navigation never waits for a full rebuild.
        self._bm25_dirty_bucket_ids: set[str] = set()
        # External edits do not identify which bucket changed. Until their
        # replacement generation is ready, keep the exact legacy scorer rather
        # than reading an unverifiable resident row.
        self._bm25_unknown_dirty = True
        self._locks_dir = os.path.join(self.base_dir, ".locks")
        self._maintenance_barrier = MaintenanceBarrier(self.base_dir)
        self._clothing_review_queue = ReviewQueue(
            os.path.join(self.base_dir, "review_queue.jsonl"),
            maintenance_root=self.base_dir,
        )
        with self._maintenance_barrier.shared():
            self.audit_log = MutationAuditLog(
                self.base_dir,
                config.get("audit", {}),
            )
        self._bucket_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, bucket_id: str) -> asyncio.Lock:
        lock = self._bucket_locks.get(bucket_id)
        if lock is None:
            lock = asyncio.Lock()
            self._bucket_locks[bucket_id] = lock
        return lock

    @asynccontextmanager
    async def _write_guard(self, bucket_id: str):
        """Serialize writes in this process and across maintenance processes."""
        async with self._maintenance_barrier.shared_async():
            async with self._lock_for(bucket_id):
                lock_name = re.sub(r"[^A-Za-z0-9_.-]", "_", bucket_id)
                lock_path = os.path.join(self._locks_dir, f"{lock_name}.lock")
                with advisory_file_lock(lock_path):
                    yield

    @staticmethod
    def _post_snapshot(post, file_path: str = "") -> dict:
        snapshot = {
            "metadata": dict(post.metadata),
            "content": post.content,
        }
        if file_path:
            snapshot["path"] = os.path.abspath(file_path)
        return snapshot

    def _atomic_write_post(
        self,
        file_path: str,
        post,
        *,
        bm25_content_changed: bool = True,
    ) -> None:
        atomic_write_post(file_path, post)
        incremental_applied = False
        bucket_id = str(post.get("id", ""))
        if bm25_content_changed and bucket_id:
            bucket = {
                "id": bucket_id,
                "metadata": dict(post.metadata),
                "content": post.content,
                "path": os.path.abspath(file_path),
            }
            incremental_applied = self._apply_bm25_incremental(
                bucket=bucket,
                visible=self._recall_path_visible(
                    file_path,
                    self._recall_cache_key(False),
                ),
            )
            if not incremental_applied:
                self._bm25_dirty_bucket_ids.add(bucket_id)
        self._refresh_recall_snapshot_entry(
            file_path,
            bm25_content_changed=(
                bm25_content_changed and not incremental_applied
            ),
        )
        self.invalidate_list_all_cache(
            bm25_content_changed=(
                bm25_content_changed and not incremental_applied
            ),
            bm25_change_is_known=True,
        )

    def invalidate_list_all_cache(
        self,
        *,
        bm25_content_changed: bool = True,
        bm25_change_is_known: bool = False,
    ) -> None:
        """Drop parsed bucket snapshots after a durable in-process mutation."""
        self._list_all_cache_generation += 1
        self._list_all_cache.clear()
        if bm25_content_changed:
            self._mark_bm25_dirty(known_change=bm25_change_is_known)

    def _recall_cache_key(
        self,
        include_archive: bool = False,
        include_nsfw: bool | None = None,
    ) -> tuple[bool, bool]:
        if include_nsfw is None:
            include_nsfw = getattr(self, "nsfw_active", False)
        return (bool(include_archive), bool(include_nsfw))

    def _recall_dirs(self, cache_key: tuple[bool, bool]) -> list[str]:
        include_archive, include_nsfw = cache_key
        dirs = [self.permanent_dir, self.dynamic_dir, self.feel_dir]
        if include_archive:
            dirs.append(self.archive_dir)
        if include_nsfw:
            dirs.append(self.nsfw_dir)
        return dirs

    @staticmethod
    def _path_is_under(file_path: str, directory: str) -> bool:
        try:
            return os.path.commonpath(
                (os.path.abspath(file_path), os.path.abspath(directory))
            ) == os.path.abspath(directory)
        except (OSError, ValueError):
            return False

    def _recall_path_visible(
        self,
        file_path: str,
        cache_key: tuple[bool, bool],
    ) -> bool:
        return any(
            self._path_is_under(file_path, directory)
            for directory in self._recall_dirs(cache_key)
        )

    def _replace_recall_snapshot(
        self,
        cache_key: tuple[bool, bool],
        buckets: tuple[dict, ...],
        *,
        disk_token: _BucketTreeSnapshot | None = None,
    ) -> None:
        self._recall_snapshot_cache[cache_key] = buckets
        self._recall_snapshot_generation[cache_key] = (
            self._recall_snapshot_generation.get(cache_key, 0) + 1
        )
        self._recall_snapshot_disk_token[cache_key] = disk_token

    def _refresh_recall_snapshot_entry(
        self,
        file_path: str,
        *,
        previous_path: str = "",
        bm25_content_changed: bool = True,
    ) -> None:
        """Write one durable bucket mutation through to resident recall views."""
        if not self._recall_snapshot_cache:
            return
        bucket = self._load_bucket(file_path) if os.path.isfile(file_path) else None
        bucket_id = str(bucket.get("id", "")) if bucket else ""
        if bm25_content_changed and bucket_id:
            self._bm25_dirty_bucket_ids.add(bucket_id)
        normalized_path = os.path.normcase(os.path.abspath(file_path))
        normalized_previous = (
            os.path.normcase(os.path.abspath(previous_path))
            if previous_path
            else ""
        )
        for cache_key, current in list(self._recall_snapshot_cache.items()):
            updated = list(current)
            found = None
            for index, existing in enumerate(updated):
                existing_path = os.path.normcase(
                    os.path.abspath(str(existing.get("path", "")))
                )
                if (
                    (bucket_id and str(existing.get("id", "")) == bucket_id)
                    or existing_path == normalized_path
                    or (normalized_previous and existing_path == normalized_previous)
                ):
                    found = index
                    break
            visible = bool(bucket) and self._recall_path_visible(file_path, cache_key)
            if visible and found is not None:
                updated[found] = bucket
            elif visible:
                updated.append(bucket)
            elif found is not None:
                updated.pop(found)
            else:
                continue
            self._replace_recall_snapshot(cache_key, tuple(updated))

    def _remove_recall_snapshot_entry(
        self,
        bucket_id: str,
        file_path: str,
    ) -> None:
        if not self._recall_snapshot_cache:
            return
        normalized_path = os.path.normcase(os.path.abspath(file_path))
        for cache_key, current in list(self._recall_snapshot_cache.items()):
            updated = tuple(
                bucket
                for bucket in current
                if str(bucket.get("id", "")) != str(bucket_id)
                and os.path.normcase(
                    os.path.abspath(str(bucket.get("path", "")))
                ) != normalized_path
            )
            if len(updated) != len(current):
                self._replace_recall_snapshot(cache_key, updated)

    def _scan_recall_snapshot_sync(
        self,
        dirs: list[str],
    ) -> tuple[list[str], _BucketTreeSnapshot]:
        paths: list[str] = []
        snapshot: list[_BucketFileRevision] = []
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    file_path = os.path.join(root, filename)
                    paths.append(file_path)
                    try:
                        info = os.stat(file_path)
                        snapshot.append(
                            (
                                file_path,
                                int(info.st_dev),
                                int(info.st_ino),
                                int(info.st_size),
                                int(info.st_mtime_ns),
                                int(info.st_ctime_ns),
                            )
                        )
                    except OSError:
                        snapshot.append(
                            (file_path, None, None, None, None, None)
                        )
        return paths, tuple(snapshot)

    def _load_recall_snapshot_sync(self, paths: list[str]) -> tuple[dict, ...]:
        return tuple(
            bucket
            for file_path in paths
            if (bucket := self._load_bucket(file_path)) is not None
        )

    async def _refresh_recall_snapshot_background(
        self,
        cache_key: tuple[bool, bool],
    ) -> None:
        """Reconcile external edits off the request thread and swap atomically."""
        try:
            generation = self._recall_snapshot_generation.get(cache_key, 0)
            dirs = self._recall_dirs(cache_key)
            paths, disk_token = await asyncio.to_thread(
                self._scan_recall_snapshot_sync,
                dirs,
            )
            if self._recall_snapshot_disk_token.get(cache_key) == disk_token:
                return
            buckets = await asyncio.to_thread(
                self._load_recall_snapshot_sync,
                paths,
            )
            after_paths, after_token = await asyncio.to_thread(
                self._scan_recall_snapshot_sync,
                dirs,
            )
            if after_paths != paths or after_token != disk_token:
                return
            async with self._recall_snapshot_lock:
                if generation != self._recall_snapshot_generation.get(cache_key, 0):
                    return
                self._replace_recall_snapshot(
                    cache_key,
                    buckets,
                    disk_token=disk_token,
                )
                self._mark_bm25_dirty()
        except Exception as exc:
            logger.warning(
                "Recall snapshot background refresh failed: %s",
                type(exc).__name__,
            )

    def _schedule_recall_snapshot_refresh(
        self,
        cache_key: tuple[bool, bool],
    ) -> None:
        now = time.monotonic()
        current = self._recall_snapshot_refresh_tasks.get(cache_key)
        if current is not None and not current.done():
            return
        if now < self._recall_snapshot_refresh_due.get(cache_key, 0.0):
            return
        self._recall_snapshot_refresh_due[cache_key] = (
            now + self._recall_snapshot_refresh_sec
        )
        task = asyncio.create_task(
            self._refresh_recall_snapshot_background(cache_key)
        )
        self._recall_snapshot_refresh_tasks[cache_key] = task

        def _clear(done: asyncio.Task) -> None:
            if self._recall_snapshot_refresh_tasks.get(cache_key) is done:
                self._recall_snapshot_refresh_tasks.pop(cache_key, None)

        task.add_done_callback(_clear)

    async def prewarm_recall_snapshot(
        self,
        include_archive: bool = False,
        include_nsfw: bool | None = None,
    ) -> bool:
        cache_key = self._recall_cache_key(include_archive, include_nsfw)
        if cache_key in self._recall_snapshot_cache:
            return True
        async with self._recall_snapshot_lock:
            if cache_key in self._recall_snapshot_cache:
                return True
            while True:
                generation = self._list_all_cache_generation
                buckets = await self.list_all(
                    include_archive=cache_key[0],
                    include_nsfw=cache_key[1],
                )
                if generation == self._list_all_cache_generation:
                    break
                await asyncio.sleep(0)
            cached = self._list_all_cache.get(cache_key)
            disk_token = cached[0] if cached is not None else None
            self._replace_recall_snapshot(
                cache_key,
                tuple(buckets),
                disk_token=disk_token,
            )
            self._recall_snapshot_refresh_due[cache_key] = (
                time.monotonic() + self._recall_snapshot_refresh_sec
            )
        return True

    async def borrow_recall_snapshot(
        self,
        include_archive: bool = False,
        include_nsfw: bool | None = None,
    ) -> tuple[dict, ...]:
        """Return the resident immutable recall tuple without scanning/copying."""
        cache_key = self._recall_cache_key(include_archive, include_nsfw)
        if cache_key not in self._recall_snapshot_cache:
            await self.prewarm_recall_snapshot(
                include_archive=cache_key[0],
                include_nsfw=cache_key[1],
            )
        snapshot = self._recall_snapshot_cache.get(cache_key, ())
        self._schedule_recall_snapshot_refresh(cache_key)
        return snapshot

    async def recall_snapshot_token(
        self,
        include_archive: bool = False,
        include_nsfw: bool | None = None,
    ) -> tuple | None:
        """O(1) token for views derived from the resident recall snapshot."""
        cache_key = self._recall_cache_key(include_archive, include_nsfw)
        if cache_key not in self._recall_snapshot_cache:
            return None
        return (cache_key, self._recall_snapshot_generation.get(cache_key, 0))

    def _bm25_with_delta(
        self,
        current,
        *,
        bucket: dict | None = None,
        bucket_id: str = "",
        visible: bool,
    ):
        """Return one complete COW generation with resident sidecars."""
        target_id = str(bucket_id or (bucket or {}).get("id", ""))
        if not target_id:
            raise ValueError("incremental BM25 delta requires bucket id")
        fresh = (
            current.with_upsert(bucket)
            if visible and bucket is not None
            else current.with_delete(target_id)
        )
        keyword_rows = dict(
            getattr(current, "_keyword_score_rows", {})
        )
        literal_rows = list(
            getattr(current, "_literal_retrieval_rows", ())
        )
        replacement = None
        if visible and bucket is not None:
            row = self._build_keyword_score_row(bucket)
            keyword_rows[target_id] = row
            if row["retrieval_keys"]:
                replacement = (target_id, row["retrieval_keys"])
        else:
            keyword_rows.pop(target_id, None)

        found = False
        updated_literal_rows = []
        for row in literal_rows:
            if row[0] != target_id:
                updated_literal_rows.append(row)
                continue
            found = True
            if replacement is not None:
                updated_literal_rows.append(replacement)
        if not found and replacement is not None:
            updated_literal_rows.append(replacement)

        fresh._keyword_score_rows = keyword_rows
        fresh._literal_retrieval_rows = tuple(updated_literal_rows)
        return fresh

    def _apply_bm25_incremental(
        self,
        *,
        bucket: dict | None = None,
        bucket_id: str = "",
        visible: bool,
    ) -> bool:
        """Atomically install one copy-on-write BM25 generation."""
        current = self._bm25
        if (
            self._bm25_mode == "off"
            or current is None
            or getattr(current, "_index", None) is None
        ):
            return False
        target_id = str(bucket_id or (bucket or {}).get("id", ""))
        if not target_id:
            return False

        started_at = time.perf_counter()
        try:
            delta_bucket = None
            if visible and bucket is not None:
                delta_bucket = {
                    "id": target_id,
                    "metadata": dict(bucket.get("metadata", {}) or {}),
                    "content": str(bucket.get("content", "")),
                    "path": str(bucket.get("path", "")),
                }
            fresh = self._bm25_with_delta(
                current,
                bucket=delta_bucket,
                bucket_id=target_id,
                visible=visible,
            )
            self._bm25_generation += 1
            self._bm25 = fresh
            if self._bm25_rebuilding:
                self._bm25_known_deltas.append(
                    (
                        self._bm25_generation,
                        delta_bucket,
                        target_id,
                        visible,
                    )
                )
            self._bm25_dirty_bucket_ids.discard(target_id)
            if not self._bm25_unknown_dirty and not self._bm25_dirty_bucket_ids:
                self._bm25_dirty = False
            logger.info(
                "bm25_incremental=%s",
                json.dumps(
                    {
                        "bucket_id": target_id,
                        "elapsed_ms": round(
                            (time.perf_counter() - started_at) * 1000.0,
                            3,
                        ),
                        "generation": self._bm25_generation,
                        "operation": "upsert" if visible else "delete",
                        "rows": len(getattr(fresh, "_ids", ())),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
            return True
        except Exception as exc:
            logger.warning(
                "[bm25] incremental update failed; keeping complete old index: %s",
                type(exc).__name__,
            )
            return False

    def _sync_bm25_path_transition(
        self,
        bucket_id: str,
        previous_path: str,
        current_path: str,
    ) -> bool:
        cache_key = self._recall_cache_key(False)
        was_visible = self._recall_path_visible(previous_path, cache_key)
        is_visible = self._recall_path_visible(current_path, cache_key)
        if was_visible == is_visible:
            return True
        bucket = self._load_bucket(current_path) if is_visible else None
        applied = self._apply_bm25_incremental(
            bucket=bucket,
            bucket_id=bucket_id,
            visible=is_visible,
        )
        if not applied:
            self._bm25_dirty_bucket_ids.add(str(bucket_id))
            self._mark_bm25_dirty(known_change=True)
        return applied

    def _mark_bm25_dirty(self, *, known_change: bool = False) -> None:
        self._bm25_generation += 1
        self._bm25_dirty = True
        if not known_change:
            self._bm25_unknown_dirty = True
            self._bm25_unknown_generation += 1
        # In-process writes happen on the service loop. Schedule one delayed
        # refresh immediately; sync maintenance callers safely fall back to the
        # same scheduler on their next search.
        self._schedule_bm25_rebuild()

    def _bm25_rebuild_delay(self) -> float:
        if self._bm25_last_rebuild_started_at is None:
            return 0.0
        return max(
            0.0,
            self._bm25_last_rebuild_started_at
            + self._bm25_rebuild_min_interval_sec
            - time.monotonic(),
        )

    def _schedule_bm25_rebuild(self) -> bool:
        """Coalesce dirty generations into one delayed background rebuild."""
        if (
            self._bm25_mode == "off"
            or not self._bm25_dirty
            or self._bm25_rebuilding
        ):
            return False
        current = self._bm25_rebuild_task
        if current is not None and not current.done():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False

        delay = self._bm25_rebuild_delay()
        task = loop.create_task(self._rebuild_bm25_when_due())
        self._bm25_rebuild_task = task
        logger.info(
            "bm25_rebuild_scheduled=%s",
            json.dumps(
                {
                    "delay_ms": round(delay * 1000.0, 3),
                    "generation": self._bm25_generation,
                    "min_interval_sec": self._bm25_rebuild_min_interval_sec,
                    "mode": self._bm25_mode,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

        def _clear(done: asyncio.Task) -> None:
            if self._bm25_rebuild_task is done:
                self._bm25_rebuild_task = None
                if not done.cancelled() and self._bm25_dirty:
                    self._schedule_bm25_rebuild()

        task.add_done_callback(_clear)
        return True

    async def _rebuild_bm25_when_due(self) -> bool:
        try:
            delay = self._bm25_rebuild_delay()
            if delay > 0:
                await asyncio.sleep(delay)
            if not self._bm25_dirty or self._bm25_rebuilding:
                return False
            # A resident-snapshot preparation failure must be rate-limited too;
            # otherwise the done callback would immediately reschedule forever.
            self._bm25_last_rebuild_started_at = time.monotonic()
            snapshot_fn = getattr(self, "borrow_recall_snapshot", None)
            buckets = (
                await snapshot_fn(include_archive=False)
                if callable(snapshot_fn)
                else await self.list_all(include_archive=False)
            )
            self._bm25_rebuilding = True
            generation = self._bm25_generation
            return await self._rebuild_bm25_async(
                list(buckets),
                generation,
                reason="request_dirty",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._bm25_rebuilding = False
            logger.warning(
                "[bm25] coalesced rebuild preparation failed; keeping old index: %s",
                type(exc).__name__,
            )
            return False

    @staticmethod
    def _build_keyword_score_row(bucket: dict) -> dict:
        """Freeze the exact inputs used by the legacy topic scorer."""
        metadata = bucket.get("metadata", {}) or {}
        name = metadata.get("name", "")
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        domain = metadata.get("domain", [])
        if isinstance(domain, str):
            domain = [domain]
        full_content = str(bucket.get("content", ""))
        return {
            "retrieval_keys": tuple(
                key.casefold()
                for key in literal_retrieval_keys(
                    full_content,
                    metadata.get("retrieval_keys", []),
                )
            ),
            "name": str(name),
            "name_present": bool(name),
            "name_lower": str(name).lower(),
            "tags": tuple(str(tag) for tag in tags),
            "tags_lower": tuple(str(tag).lower() for tag in tags),
            "domain": tuple(str(value) for value in domain),
            "content": full_content[:1000],
        }

    @staticmethod
    def _build_bm25_index(buckets: list[dict]) -> BM25Index:
        index = BM25Index()
        index.build(buckets)
        # Validate literal navigation keys beside the same complete BM25
        # generation. Requests then scan a small resident key table instead of
        # reparsing every full bucket body.
        literal_rows: list[tuple[str, tuple[str, ...]]] = []
        keyword_score_rows: dict[str, dict] = {}
        for bucket in buckets:
            bucket_id = str(bucket.get("id", ""))
            if not bucket_id:
                continue
            row = BucketManager._build_keyword_score_row(bucket)
            keyword_score_rows[bucket_id] = row
            keys = row["retrieval_keys"]
            if keys:
                literal_rows.append((bucket_id, keys))
        index._literal_retrieval_rows = tuple(literal_rows)
        index._keyword_score_rows = keyword_score_rows
        return index

    def _keyword_score_row_for_bucket(self, bucket: dict) -> dict:
        bucket_id = str(bucket.get("id", ""))
        if bucket_id and bucket_id not in self._bm25_dirty_bucket_ids:
            row = getattr(self._bm25, "_keyword_score_rows", {}).get(bucket_id)
            if row is not None:
                return row
        return self._build_keyword_score_row(bucket)

    def _cheap_topic_score_from_row(
        self,
        query: str,
        query_lower: str,
        query_parts: tuple[str, ...],
        row: dict,
    ) -> float:
        """Return the legacy topic score without the fuzzy body component."""
        if any(key in query_lower for key in row["retrieval_keys"]):
            return 1.0

        name_lower = row["name_lower"]
        tags_lower = row["tags_lower"]
        if query_lower in name_lower or any(
            query_lower in tag for tag in tags_lower
        ):
            return 1.0

        hit_count = 0
        for part in query_parts:
            if name_lower and part in name_lower:
                hit_count += 1
                continue
            if any(tag and part in tag for tag in tags_lower):
                hit_count += 1
        partial_hit_score = hit_count / len(query_parts) if query_parts else 0.0
        name_score = (
            fuzz.ratio(query, row["name"]) / 100.0
            if row["name_present"]
            else 0.0
        )
        domain_score = (
            max([fuzz.ratio(query, value) for value in row["domain"]] + [0])
            / 100.0
            if row["domain"]
            else 0.0
        )
        tag_score = (
            max([fuzz.ratio(query, tag) for tag in row["tags"]] + [0])
            / 100.0
            if row["tags"]
            else 0.0
        )
        return max(
            partial_hit_score,
            name_score,
            tag_score,
            domain_score * 0.9,
        )

    def _bounded_topic_scores(
        self,
        query: str,
        candidates: list[dict],
        *,
        limit: int,
    ) -> dict[str, float]:
        """Compute the exact candidate superset needed by relevance top-k.

        The kth best non-body score is a proven lower bound for the kth final
        topic score. The existing relevance tie band extends that bound. Only
        rows whose non-body score or possible body score reaches the extended
        bound can enter the returned top-k, so this skips work without a new
        relevance threshold or a ranking change.
        """
        query_lower = str(query or "").lower()
        query_parts = self._query_parts_for_search(query_lower)
        rows: list[tuple[str, dict]] = []
        cheap_scores: list[float] = []
        for bucket in candidates:
            bucket_id = str(bucket.get("id", ""))
            if not bucket_id:
                continue
            row = self._keyword_score_row_for_bucket(bucket)
            rows.append((bucket_id, row))
            cheap_scores.append(
                self._cheap_topic_score_from_row(
                    query,
                    query_lower,
                    query_parts,
                    row,
                )
            )

        if not rows:
            return {}
        kth_index = min(max(1, int(limit)), len(rows)) - 1
        kth_lower_bound = sorted(cheap_scores, reverse=True)[kth_index]
        tie_margin = self.keyword_relevance_tie_band / 100.0
        # One extra 1e-4 absorbs the later percentage rounding at a boundary.
        candidate_floor = max(0.0, kth_lower_bound - tie_margin - 0.0001)
        body_cutoff = min(100.0, candidate_floor / 0.8 * 100.0)
        body_matches = process.extract(
            query,
            [row["content"] for _bucket_id, row in rows],
            scorer=fuzz.partial_ratio,
            score_cutoff=body_cutoff,
            limit=None,
        )
        body_scores = {
            int(index): float(score) / 100.0 * 0.8
            for _choice, score, index in body_matches
        }

        exact_scores: dict[str, float] = {}
        for index, ((bucket_id, _row), cheap_score) in enumerate(
            zip(rows, cheap_scores)
        ):
            body_score = body_scores.get(index, 0.0)
            if cheap_score >= candidate_floor or index in body_scores:
                exact_scores[bucket_id] = max(cheap_score, body_score)
        return exact_scores

    async def _rebuild_bm25_async(
        self,
        buckets: list[dict],
        generation: int,
        *,
        reason: str = "request_dirty",
        offload: bool = True,
    ) -> bool:
        started_at = time.perf_counter()
        self._bm25_last_rebuild_started_at = time.monotonic()
        applied = False
        replayed_deltas = 0
        unknown_generation = self._bm25_unknown_generation
        try:
            # Startup runs before requests exist, so an inline build avoids
            # jieba's extremely slow first-use path in a fresh worker thread.
            # Dirty live rebuilds retain the upstream worker-thread behavior.
            fresh = (
                await asyncio.to_thread(self._build_bm25_index, buckets)
                if offload
                else self._build_bm25_index(buckets)
            )
            # A write may invalidate the snapshot while jieba is building it.
            # Never let that stale build clear the newer dirty generation.
            deltas = [
                delta
                for delta in self._bm25_known_deltas
                if delta[0] > generation
            ]
            replay_generations = [delta[0] for delta in deltas]
            expected_generations = list(
                range(generation + 1, self._bm25_generation + 1)
            )
            if (
                unknown_generation == self._bm25_unknown_generation
                and replay_generations == expected_generations
            ):
                for _delta_generation, bucket, bucket_id, visible in deltas:
                    fresh = self._bm25_with_delta(
                        fresh,
                        bucket=bucket,
                        bucket_id=bucket_id,
                        visible=visible,
                    )
                    replayed_deltas += 1
                self._bm25 = fresh
                self._bm25_dirty = False
                self._bm25_dirty_bucket_ids.clear()
                self._bm25_unknown_dirty = False
                applied = True
        except Exception as exc:
            logger.warning("[bm25] background rebuild failed; keeping old index: %s", exc)
        finally:
            self._bm25_rebuilding = False
            self._bm25_known_deltas.clear()
            logger.info(
                "bm25_rebuild=%s",
                json.dumps(
                    {
                        "applied": applied,
                        "elapsed_ms": round(
                            (time.perf_counter() - started_at) * 1000.0,
                            3,
                        ),
                        "generation": generation,
                        "execution": "worker_thread" if offload else "startup_inline",
                        "mode": self._bm25_mode,
                        "reason": reason,
                        "replayed_deltas": replayed_deltas,
                        "rows": len(buckets),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        return applied

    async def prewarm_bm25(self) -> bool:
        """Build the optional BM25 index once before requests are accepted."""
        if self._bm25_mode == "off":
            return False
        if not self._bm25_dirty and getattr(self._bm25, "_index", None) is not None:
            return True
        if self._bm25_rebuilding:
            return False

        started_at = time.perf_counter()
        try:
            snapshot_fn = getattr(self, "borrow_recall_snapshot", None)
            buckets = (
                await snapshot_fn(include_archive=False)
                if callable(snapshot_fn)
                and hasattr(self, "_recall_snapshot_cache")
                else await self.list_all(include_archive=False)
            )
            loaded_at = time.perf_counter()
            self._bm25_rebuilding = True
            generation = self._bm25_generation
            applied = await self._rebuild_bm25_async(
                list(buckets),
                generation,
                reason="startup_prewarm",
                offload=False,
            )
            logger.info(
                "bm25_prewarm=%s",
                json.dumps(
                    {
                        "applied": applied,
                        "build_ms": round(
                            (time.perf_counter() - loaded_at) * 1000.0,
                            3,
                        ),
                        "generation": generation,
                        "load_ms": round(
                            (loaded_at - started_at) * 1000.0,
                            3,
                        ),
                        "mode": self._bm25_mode,
                        "rows": len(buckets),
                        "total_ms": round(
                            (time.perf_counter() - started_at) * 1000.0,
                            3,
                        ),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
            return applied
        except Exception as exc:
            self._bm25_rebuilding = False
            logger.warning(
                "[bm25] startup prewarm failed; keeping existing index: %s",
                type(exc).__name__,
            )
            return False

    # ---------------------------------------------------------
    # Create a new bucket
    # 创建新桶
    # ---------------------------------------------------------
    async def create(
        self,
        content: str,
        tags: list[str] = None,
        importance: int = 5,
        domain: list[str] = None,
        valence: float = 0.5,
        arousal: float = 0.3,
        bucket_type: str = "dynamic",
        name: str = None,
        retrieval_keys: list[str] = None,
        pinned: bool = False,
        protected: bool = False,
        world: str = "",
        thread: str = "other",
        chord_tag: str = None,
        tier: int = None,
        sense: list[str] = None,
        event_at=None,
        date_precision: str = None,
        date_source: str = None,
        date_confidence: float = None,
        actor: str = "system",
        x_provenance: dict = None,
        curated_write_key: str = None,
        curated_payload_sha256: str = None,
        vector_policy: str = None,
        lmc5_recall_state: str = None,
        e_authored_by: str = "",
        e_initial_priority: int | None = None,
        e_valence: float | None = None,
        e_arousal: float | None = None,
        e_tension: float | None = None,
        e_confidence: float | None = None,
        e_response_tendency: str = "",
        e_growth_delta: str = "",
        e_source_bucket_id: str = "",
        e_proposal_key: str = "",
    ) -> str:
        bucket_id = generate_bucket_id()
        domain = domain or ["未分类"]
        tags = list(tags) if tags else []
        original_content = str(content or "")
        retrieval_candidates = list(retrieval_keys or []) + [name or ""] + tags
        selected_retrieval_keys = literal_retrieval_keys(
            original_content,
            retrieval_candidates,
        )
        if not selected_retrieval_keys:
            # 候选池全军覆没（feel 桶的常态：name 空 + 全是系统标签）→ 退回正文首句。
            # 见 body_lead_keys：那句天生在正文里，能同时当名字和检索钥匙。
            selected_retrieval_keys = literal_retrieval_keys(
                original_content,
                body_lead_keys(original_content),
            )
        needs_clothing = not selected_retrieval_keys
        if not str(name or "").strip():
            name = selected_retrieval_keys[0] if selected_retrieval_keys else "待补衣"
        recorded_at = now_iso()
        if event_at is None:
            normalized_event_at = recorded_at
            inferred_precision = "second"
            effective_date_source = date_source or "recorded_at_default"
            effective_confidence = 0.5 if date_confidence is None else float(date_confidence)
        else:
            normalized_event_at, inferred_precision = normalize_event_at(event_at)
            effective_date_source = date_source or "explicit"
            effective_confidence = 1.0 if date_confidence is None else float(date_confidence)
        effective_precision = date_precision or inferred_precision
        if effective_precision not in DATE_PRECISIONS:
            raise ValueError(
                f"date_precision must be one of {sorted(DATE_PRECISIONS)}"
            )
        effective_confidence = max(0.0, min(1.0, effective_confidence))

        # --- Stamp creation date into name and tags (skip feel buckets) ---
        # --- 时间戳进桶名+标签（feel 桶铁律：不动）---
        if bucket_type != "feel":
            today = datetime.now().strftime("%Y-%m-%d")
            date_re = re.compile(r"\d{4}-\d{2}-\d{2}")
            if name and not date_re.search(name):
                name = f"{name}_{today}"
            if not any(date_re.fullmatch(str(t)) for t in tags):
                tags.append(today)

        bucket_name = sanitize_name(name) if name else bucket_id
        # 2026-08-18：钥匙**只进 metadata.retrieval_keys，不再追加进正文**。
        # 8/17 曾在正文尾部补一行「[检索钥匙: …]」，但召回侧读的是 metadata（见
        # server._…retrieval_keys），正文里那行没人用；反而把「正文即原文」的契约
        # 全撕了：E 轴一字不改、curated_writer 的 receipt sha256、recall-before-write
        # 的乐观锁 hash、feel 桶按正文解析数值——13 条测试因此红。钥匙本来就是
        # 正文的逐字子串，不写回正文一样搜得到。已带钥匙行的旧桶原样保留，
        # literal_retrieval_keys 仍会从正文里读它。
        linked_content = original_content

        if pinned or protected:
            importance = 10

        metadata = {
            "id": bucket_id,
            "name": bucket_name,
            "tags": tags,
            "domain": domain,
            "valence": max(0.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "importance": max(1, min(10, importance)),
            "type": bucket_type,
            "event_at": normalized_event_at,
            "recorded_at": recorded_at,
            "date_precision": effective_precision,
            "date_source": effective_date_source,
            "date_confidence": effective_confidence,
            # Transitional read compatibility for older clients and vault views.
            "created": normalized_event_at,
            "last_active": recorded_at,
            "activation_count": 1,
            "thread": normalize_thread(thread),
        }
        if selected_retrieval_keys:
            metadata["retrieval_keys"] = selected_retrieval_keys
        else:
            metadata["needs_clothing"] = True
            metadata["clothing_reason"] = "no_literal_retrieval_key"
        e_fields = (
            e_authored_by,
            e_initial_priority,
            e_valence,
            e_arousal,
            e_tension,
            e_confidence,
            e_response_tendency,
            e_growth_delta,
            e_source_bucket_id,
            e_proposal_key,
        )
        if any(value not in (None, "") for value in e_fields):
            author = str(e_authored_by or "").strip()
            if not author or len(author) > 120 or "\n" in author:
                raise ValueError("E-axis content requires a bounded e_authored_by")
            if type(e_initial_priority) is not int or not 1 <= e_initial_priority <= 100:
                raise ValueError("e_initial_priority must be a plain integer in 1..100")

            def e_number(value, name: str, low: float, high: float) -> float:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be in [{low}, {high}]")
                number = float(value)
                if not math.isfinite(number) or not low <= number <= high:
                    raise ValueError(f"{name} must be in [{low}, {high}]")
                return number

            tendency = str(e_response_tendency or "").strip()
            growth = str(e_growth_delta or "").strip()
            if tendency not in E_RESPONSE_TENDENCIES:
                raise ValueError("invalid e_response_tendency")
            if growth not in E_GROWTH_DELTAS:
                raise ValueError("invalid e_growth_delta")
            source_bucket_id = str(e_source_bucket_id or "").strip()
            if len(source_bucket_id) > 128 or "\n" in source_bucket_id:
                raise ValueError("invalid e_source_bucket_id")
            proposal_key = str(e_proposal_key or "").strip()
            if len(proposal_key) > 160 or "\n" in proposal_key:
                raise ValueError("invalid e_proposal_key")
            metadata.update({
                "e_authored_by": author,
                "e_initial_priority": e_initial_priority,
                "e_valence": e_number(e_valence, "e_valence", -1.0, 1.0),
                "e_arousal": e_number(e_arousal, "e_arousal", 0.0, 1.0),
                "e_tension": e_number(e_tension, "e_tension", 0.0, 1.0),
                "e_confidence": e_number(e_confidence, "e_confidence", 0.0, 1.0),
                "e_response_tendency": tendency,
                "e_growth_delta": growth,
                "e_authored_at": recorded_at,
            })
            if source_bucket_id:
                metadata["e_source_bucket_id"] = source_bucket_id
            if proposal_key:
                metadata["e_proposal_key"] = proposal_key
        if x_provenance is not None:
            # X provenance is part of the bucket's first durable write.  It
            # must never be patched in after create, because a failed second
            # write would leave a derived memory with no evidence chain.
            metadata.update(normalize_x_provenance(x_provenance))
        curated_fields = (
            curated_write_key,
            curated_payload_sha256,
            vector_policy,
            lmc5_recall_state,
        )
        if any(value is not None for value in curated_fields):
            if not all(isinstance(value, str) and value for value in curated_fields):
                raise ValueError(
                    "curated write metadata must be supplied as one complete set"
                )
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}",
                curated_write_key,
            ):
                raise ValueError("invalid curated_write_key")
            if not re.fullmatch(r"[0-9a-f]{64}", curated_payload_sha256):
                raise ValueError("invalid curated_payload_sha256")
            if vector_policy not in {"required", "fts_only"}:
                raise ValueError("invalid vector_policy")
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", lmc5_recall_state):
                raise ValueError("invalid lmc5_recall_state")
            metadata.update({
                "curated_write_key": curated_write_key,
                "curated_payload_sha256": curated_payload_sha256,
                "vector_policy": vector_policy,
                "lmc5_recall_state": lmc5_recall_state,
            })
        if pinned:
            metadata["pinned"] = True
        if protected:
            metadata["protected"] = True
        # world 字段仅在非空时写入（保持日常桶 frontmatter 简洁）
        if world and world.strip():
            metadata["world"] = world.strip()
        # chord_tag 字段：和弦记号串作为情绪坐标索引,仅在非空时写入
        # chord_tag: chord-notation string as emotion-coordinate index; written only if non-empty
        # 不参与表达,仅作跨窗口标记。格式示例: "Em(maj7) → A13#11 · 92bpm · f"
        if chord_tag and chord_tag.strip():
            metadata["chord_tag"] = chord_tag.strip()

        # tier 字段（2026-05-30 #4 核心画像分离）：
        # 仿生五层记忆分层 0=核心画像/铁律 / 1=档案 / 2=日记 / 3=趋势 / 4=时段摘要
        # 仅在显式传入时写入；旧桶 tier 缺失，briefing 按 None 处理（不进 tier0 原文 slots）。
        # tier=0 的桶在 briefing(format=json) 时单独原文输出、不进 LLM 压缩链路。
        if tier is not None:
            metadata["tier"] = int(tier)

        # sense 字段（五感入口层 v1）：嗅觉/味觉/触觉/听觉 标签，仅在非空时写入。
        # 让 breath 能按感官线索把"带味道/触感的记忆"连情绪一起浮回（普鲁斯特钩子）。
        if sense:
            clean_sense = [s for s in sense if isinstance(s, str) and s.strip()]
            if clean_sense:
                metadata["sense"] = clean_sense

        # Defensive: ensure no 'content' key sneaks into metadata kwargs
        # 防御性：确保 metadata 里没有 content 键，否则会和 body 撞 Post() 参数
        metadata.pop("content", None)

        if bucket_type == "permanent" or pinned:
            type_dir = self.permanent_dir
            if pinned and bucket_type != "permanent":
                metadata["type"] = "permanent"
        elif bucket_type == "feel":
            type_dir = self.feel_dir
        elif bucket_type == "archived":
            # A strict writer may stage a body in the cold store while an
            # external index is being prepared.  Keeping both the metadata
            # type and the physical path archived makes that intermediate
            # body invisible to ordinary ``list_all``/``search`` callers.
            type_dir = self.archive_dir
        else:
            type_dir = self.dynamic_dir
            
        if bucket_type == "feel":
            primary_domain = "沉淀物"
        else:
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
            
        target_dir = os.path.join(type_dir, primary_domain)

        if bucket_name and bucket_name != bucket_id:
            filename = f"{bucket_name}_{bucket_id}.md"
        else:
            filename = f"{bucket_id}.md"
        file_path = safe_path(target_dir, filename)

        post = frontmatter.Post(linked_content, **metadata)
        async with self._write_guard(bucket_id):
            os.makedirs(target_dir, exist_ok=True)
            event_id = self.audit_log.begin(
                actor=actor,
                action="create",
                bucket_id=bucket_id,
                before=None,
                after=self._post_snapshot(post, file_path),
                details={"path": os.path.abspath(file_path)},
            )
            try:
                self._atomic_write_post(file_path, post)
                self._bucket_path_cache[bucket_id] = file_path
                self.audit_log.commit(event_id)
            except Exception as e:
                self.audit_log.fail(event_id, e)
                logger.error(f"Failed to write bucket file / 写入桶文件失败: {file_path}: {e}")
                raise

        logger.info(
            f"Created bucket / 创建记忆桶: {bucket_id} ({bucket_name}) → {primary_domain}/"
            + (" [PINNED]" if pinned else "") + (" [PROTECTED]" if protected else "")
        )
        if needs_clothing:
            try:
                self._clothing_review_queue.enqueue(
                    make_clothing_entry(
                        bucket_id,
                        bucket_name,
                        content_sha256=hashlib.sha256(
                            original_content.encode("utf-8")
                        ).hexdigest(),
                        source=actor,
                    )
                )
            except Exception as exc:
                # The durable memory is authoritative and must survive a
                # sidecar outage.  Frontmatter still makes the gap observable.
                logger.warning(
                    "Bucket clothing queue unavailable for %s: %s",
                    bucket_id,
                    type(exc).__name__,
                )
        return bucket_id

    # ---------------------------------------------------------
    # Read bucket content
    # ---------------------------------------------------------
    async def get(self, bucket_id: str) -> Optional[dict]:
        if not bucket_id or not isinstance(bucket_id, str):
            return None
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return None
        return self._load_bucket(file_path)

    async def cache_recall_dehydration(
        self,
        bucket_id: str,
        *,
        expected_content_hash: str,
        summary: str,
        actor: str = "system:recall-dehydration-cache",
    ) -> bool:
        """Persist a derived recall summary without changing memory activity.

        The cache is valid only for the exact Markdown body hash supplied by
        the caller.  Unlike ``update()``, this narrow derived-metadata write
        deliberately does not touch ``last_active`` or move the bucket.
        """
        expected_content_hash = str(expected_content_hash or "").strip().lower()
        summary = str(summary or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_content_hash):
            raise ValueError("expected_content_hash must be sha256 hex")
        if len(summary) < 10:
            raise ValueError("dehydrated summary is empty or too short")

        async with self._write_guard(bucket_id):
            file_path = self._find_bucket_file(bucket_id)
            if not file_path:
                return False

            event_id = None
            try:
                post = self._safe_load_post(file_path)
                actual_content_hash = hashlib.sha256(
                    str(post.content or "").encode("utf-8")
                ).hexdigest()
                if actual_content_hash != expected_content_hash:
                    logger.info(
                        "Recall dehydration cache skipped after body change: %s",
                        bucket_id,
                    )
                    return False
                if (
                    post.get("dehydrated_content_hash") == expected_content_hash
                    and post.get("dehydrated_summary") == summary
                ):
                    return True

                before = self._post_snapshot(post, file_path)
                post["dehydrated_summary"] = summary
                post["dehydrated_content_hash"] = expected_content_hash
                event_id = self.audit_log.begin(
                    actor=actor,
                    action="cache_recall_dehydration",
                    bucket_id=bucket_id,
                    before=before,
                    after=self._post_snapshot(post, file_path),
                    details={
                        "changed_fields": [
                            "dehydrated_content_hash",
                            "dehydrated_summary",
                        ],
                    },
                )
                self._atomic_write_post(
                    file_path,
                    post,
                    bm25_content_changed=False,
                )
                self.audit_log.commit(event_id)
                return True
            except Exception as exc:
                self.audit_log.fail(event_id, exc)
                logger.warning(
                    "Failed to persist recall dehydration metadata for %s: %s",
                    bucket_id,
                    type(exc).__name__,
                )
                return False

    async def set_thread(
        self,
        bucket_id: str,
        thread: str,
        *,
        actor: str = "system:timeline",
        expected_revision_hash: str = "",
    ) -> bool:
        """Persist only the X-axis thread without touching activity metadata."""

        normalized = normalize_thread(thread)
        async with self._write_guard(bucket_id):
            file_path = self._find_bucket_file(bucket_id)
            if not file_path:
                return False
            event_id = None
            try:
                post = self._safe_load_post(file_path)
                if expected_revision_hash:
                    actual_revision_hash = bucket_revision_hash(
                        post.content,
                        post.metadata,
                    )
                    if actual_revision_hash != expected_revision_hash:
                        logger.warning(
                            "Bucket revision changed before thread assignment: %s",
                            bucket_id,
                        )
                        return False
                if normalize_thread(post.get("thread")) == normalized and (
                    post.get("thread") == normalized
                ):
                    return True
                before = self._post_snapshot(post, file_path)
                post["thread"] = normalized
                event_id = self.audit_log.begin(
                    actor=actor,
                    action="set_thread",
                    bucket_id=bucket_id,
                    before=before,
                    after=self._post_snapshot(post, file_path),
                    details={"changed_fields": ["thread"]},
                )
                self._atomic_write_post(
                    file_path,
                    post,
                    bm25_content_changed=False,
                )
                self.audit_log.commit(event_id)
                return True
            except Exception as exc:
                self.audit_log.fail(event_id, exc)
                logger.error(
                    "Failed to assign timeline thread for %s: %s",
                    bucket_id,
                    type(exc).__name__,
                )
                return False

    def _move_bucket(self, file_path: str, target_type_dir: str, domain: list[str] = None) -> str:
        primary_domain = sanitize_name(domain[0]) if domain else "未分类"
        target_dir = os.path.join(target_type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        new_path = safe_path(target_dir, filename)
        if os.path.normpath(file_path) != os.path.normpath(new_path):
            os.replace(file_path, new_path)
            self._refresh_recall_snapshot_entry(
                str(new_path),
                previous_path=file_path,
                bm25_content_changed=False,
            )
            moved_bucket = self._load_bucket(str(new_path))
            bucket_id = str((moved_bucket or {}).get("id", ""))
            if bucket_id:
                self._sync_bm25_path_transition(
                    bucket_id,
                    file_path,
                    str(new_path),
                )
            logger.info(f"Moved bucket / 移动记忆桶: {filename} → {target_dir}/")
        return new_path

    # ---------------------------------------------------------
    # Update bucket
    # ---------------------------------------------------------
    async def update(
        self,
        bucket_id: str,
        actor: str = "system",
        expected_content_hash: str = "",
        expected_revision_hash: str = "",
        **kwargs,
    ) -> bool:
        async with self._write_guard(bucket_id):
            file_path = self._find_bucket_file(bucket_id)
            if not file_path:
                return False

            event_id = None
            try:
                post = self._safe_load_post(file_path)
                if post.get("e_authored_by"):
                    for field in E_IMMUTABLE_FIELDS:
                        if field in kwargs and kwargs[field] != post.get(field):
                            raise ValueError(
                                f"{field} is immutable; write a successor E record"
                            )
                    if "content" in kwargs and kwargs["content"] != post.content:
                        raise ValueError(
                            "primary-authored E content is immutable; write a successor"
                        )
                if expected_revision_hash:
                    actual_revision_hash = bucket_revision_hash(
                        post.content,
                        post.metadata,
                    )
                    if actual_revision_hash != expected_revision_hash:
                        logger.warning(
                            "Bucket revision changed before update / 写入前桶版本已变更: %s",
                            bucket_id,
                        )
                        return False
                if expected_content_hash:
                    actual_content_hash = hashlib.sha256(
                        str(post.content or "").encode("utf-8")
                    ).hexdigest()
                    if actual_content_hash != expected_content_hash:
                        logger.warning(
                            "Bucket revision changed before update / 写入前桶版本已变更: %s",
                            bucket_id,
                        )
                        return False
                before = self._post_snapshot(post, file_path)
                old_domain = post.get("domain", ["未分类"])
                old_type = post.get("type", "dynamic")
                old_pinned = post.get("pinned", False)

                # Provenance rewrites (and invalid saga list edits) must fail
                # before last_active or any other frontmatter can change.
                validate_x_provenance_update(post.metadata, kwargs)

                # Guard: protected-domain (or feel-type) buckets can never be resolved.
                # 守卫：保护域桶（或 feel 类型）禁止 resolved=1（5.10 黑洞事件根治）
                if kwargs.get("resolved") is True:
                    cur_domain = old_domain if isinstance(old_domain, list) else [old_domain]
                    hit = [d for d in cur_domain if d in PROTECTED_RESOLVE_DOMAINS]
                    if hit or old_type == "feel":
                        label = ",".join(hit) if hit else f"type={old_type}"
                        logger.warning(
                            f"[ResolvedGuard] refused resolved=True on {bucket_id} (protected: {label})"
                        )
                        raise ResolvedGuardError(
                            f"桶 {bucket_id} 属于保护域 [{label}]，禁止 resolved=1"
                        )

                if "recorded_at" in kwargs:
                    raise ValueError("recorded_at is immutable after bucket creation")

                time_value = kwargs.get("event_at")
                legacy_time_value = kwargs.get("created")
                if time_value is not None or legacy_time_value is not None:
                    normalized, inferred_precision = normalize_event_at(
                        time_value if time_value is not None else legacy_time_value
                    )
                    kwargs["event_at"] = normalized
                    kwargs["created"] = normalized
                    kwargs.setdefault("date_precision", inferred_precision)
                    kwargs.setdefault(
                        "date_source",
                        "explicit_update" if time_value is not None else "legacy_created_update",
                    )
                    kwargs.setdefault("date_confidence", 1.0 if time_value is not None else 0.5)

                if "date_precision" in kwargs and kwargs["date_precision"] not in DATE_PRECISIONS:
                    raise ValueError(
                        f"date_precision must be one of {sorted(DATE_PRECISIONS)}"
                    )
                if kwargs.get("date_confidence") is not None:
                    kwargs["date_confidence"] = max(
                        0.0, min(1.0, float(kwargs["date_confidence"]))
                    )

                changed_fields = []
                for key, value in kwargs.items():
                    if value is not None:
                        if key == "content":
                            if post.content != value:
                                changed_fields.append(key)
                            post.content = value
                        else:
                            if post.get(key) != value:
                                changed_fields.append(key)
                            post[key] = value

                post["last_active"] = now_iso()
                if "last_active" not in changed_fields:
                    changed_fields.append("last_active")

                new_pinned = post.get("pinned", False)
                new_type = post.get("type", "dynamic")
                new_domain = post.get("domain", ["未分类"])

                need_move = False
                target_type_dir = None

                if new_pinned and not old_pinned:
                    target_type_dir = self.permanent_dir
                    need_move = True
                elif not new_pinned and old_pinned:
                    if new_type == "feel":
                        target_type_dir = self.feel_dir
                    elif new_type == "archived":
                        target_type_dir = self.archive_dir
                    else:
                        target_type_dir = self.dynamic_dir
                    need_move = True
                elif new_type != old_type:
                    if new_type == "permanent":
                        target_type_dir = self.permanent_dir
                    elif new_type == "feel":
                        target_type_dir = self.feel_dir
                    elif new_type == "archived":
                        target_type_dir = self.archive_dir
                    else:
                        target_type_dir = self.dynamic_dir
                    need_move = True
                elif new_domain != old_domain:
                    if new_pinned or new_type == "permanent":
                        target_type_dir = self.permanent_dir
                    elif new_type == "feel":
                        target_type_dir = self.feel_dir
                    elif new_type == "archived":
                        target_type_dir = self.archive_dir
                    else:
                        target_type_dir = self.dynamic_dir
                    need_move = True

                action = "update"
                if "resolved" in kwargs and kwargs["resolved"] != before["metadata"].get("resolved"):
                    action = "resolve" if kwargs["resolved"] else "unresolve"
                event_id = self.audit_log.begin(
                    actor=actor,
                    action=action,
                    bucket_id=bucket_id,
                    before=before,
                    after=self._post_snapshot(post, file_path),
                    details={
                        "changed_fields": sorted(set(changed_fields)),
                        "move_requested": bool(need_move),
                    },
                )

                self._atomic_write_post(
                    file_path,
                    post,
                    bm25_content_changed=bool(
                        BM25_CORPUS_FIELDS.intersection(changed_fields)
                        or "retrieval_keys" in changed_fields
                    ),
                )

                if need_move and target_type_dir:
                    new_path = self._move_bucket(file_path, target_type_dir, new_domain)
                    self._bucket_path_cache[bucket_id] = new_path

                self.audit_log.commit(event_id)

            except ResolvedGuardError:
                raise
            except Exception as e:
                self.audit_log.fail(event_id, e)
                logger.error(f"Failed to update bucket / 更新桶失败: {bucket_id}: {e}")
                return False

        logger.info(f"Updated bucket / 更新记忆桶: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Relation edges (6 类关系边：只在 source 桶记出边，反向遍历得入边)
    # ---------------------------------------------------------
    async def add_relation(
        self, source_id: str, target_id: str, rel_type: str, note: str = "",
        actor: str = "system",
        *,
        strength: float | None = None,
    ) -> bool:
        if rel_type not in RELATION_TYPES:
            logger.warning(f"Unknown relation type / 未知关系类型: {rel_type}")
            return False
        if source_id == target_id:
            return False
        if strength is not None:
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                return False
            strength = float(strength)
            if not 0.0 <= strength <= 1.0:
                return False
        if not await self.get(target_id):
            logger.warning(f"Relation target not found / 关系目标桶不存在: {target_id}")
            return False
        async with self._write_guard(source_id):
            file_path = self._find_bucket_file(source_id)
            if not file_path:
                return False
            event_id = None
            try:
                post = self._safe_load_post(file_path)
                relations = list(post.get("relations") or [])
                for r in relations:
                    if isinstance(r, dict) and r.get("type") == rel_type and r.get("target") == target_id:
                        return True  # 幂等：已有同 type+target 的边就不重复
                before = self._post_snapshot(post, file_path)
                edge = {"type": rel_type, "target": target_id}
                if note and note.strip():
                    edge["note"] = note.strip()
                if strength is not None:
                    edge["strength"] = strength
                relations.append(edge)
                post["relations"] = relations
                post["last_active"] = now_iso()
                event_id = self.audit_log.begin(
                    actor=actor,
                    action="add_relation",
                    bucket_id=source_id,
                    before=before,
                    after=self._post_snapshot(post, file_path),
                    details={"edge": edge},
                )
                self._atomic_write_post(
                    file_path,
                    post,
                    bm25_content_changed=False,
                )
                self.audit_log.commit(event_id)
                logger.info(f"Added relation / 加边: {source_id} -[{rel_type}]-> {target_id}")
                return True
            except Exception as e:
                self.audit_log.fail(event_id, e)
                logger.error(f"Failed to add relation / 加边失败 {source_id}->{target_id}: {e}")
                return False

    async def remove_relation(
        self, source_id: str, target_id: str, rel_type: str = "",
        actor: str = "system",
    ) -> int:
        async with self._write_guard(source_id):
            file_path = self._find_bucket_file(source_id)
            if not file_path:
                return 0
            event_id = None
            try:
                post = self._safe_load_post(file_path)
                relations = list(post.get("relations") or [])
                if not relations:
                    return 0
                kept = []
                removed_edges = []
                for r in relations:
                    if not isinstance(r, dict):
                        kept.append(r)
                        continue
                    if r.get("target") != target_id:
                        kept.append(r)
                        continue
                    if rel_type and r.get("type") != rel_type:
                        kept.append(r)
                        continue
                    removed_edges.append(r)
                if not removed_edges:
                    return 0
                before = self._post_snapshot(post, file_path)
                if kept:
                    post["relations"] = kept
                else:
                    post.metadata.pop("relations", None)
                post["last_active"] = now_iso()
                event_id = self.audit_log.begin(
                    actor=actor,
                    action="remove_relation",
                    bucket_id=source_id,
                    before=before,
                    after=self._post_snapshot(post, file_path),
                    details={"removed_edges": removed_edges},
                )
                self._atomic_write_post(
                    file_path,
                    post,
                    bm25_content_changed=False,
                )
                self.audit_log.commit(event_id)
                removed = len(removed_edges)
                logger.info(f"Removed {removed} relation(s) / 删边: {source_id}->{target_id}")
                return removed
            except Exception as e:
                self.audit_log.fail(event_id, e)
                logger.error(f"Failed to remove relation / 删边失败 {source_id}->{target_id}: {e}")
                return 0

    # ---------------------------------------------------------
    # Delete bucket
    # ---------------------------------------------------------
    async def delete(self, bucket_id: str, actor: str = "system") -> bool:
        async with self._write_guard(bucket_id):
            file_path = self._find_bucket_file(bucket_id)
            if not file_path:
                return False

            event_id = None
            try:
                post = self._safe_load_post(file_path)
                if post.get("protected", False):
                    logger.warning(f"Cannot delete protected bucket / 受保护的桶不可删除: {bucket_id}")
                    return False
                before = self._post_snapshot(post, file_path)
                event_id = self.audit_log.begin(
                    actor=actor,
                    action="delete",
                    bucket_id=bucket_id,
                    before=before,
                    after=None,
                    details={"path": os.path.abspath(file_path)},
                )
                os.remove(file_path)
                self._bucket_path_cache.pop(bucket_id, None)
                self._remove_recall_snapshot_entry(bucket_id, file_path)
                incremental_applied = self._apply_bm25_incremental(
                    bucket_id=bucket_id,
                    visible=False,
                )
                if not incremental_applied:
                    self._bm25_dirty_bucket_ids.add(bucket_id)
                self.invalidate_list_all_cache(
                    bm25_content_changed=not incremental_applied,
                    bm25_change_is_known=True,
                )
                self.audit_log.commit(event_id)
            except Exception as e:
                self.audit_log.fail(event_id, e)
                logger.error(f"Failed to delete bucket file / 删除桶文件失败: {bucket_id}: {e}")
                return False

        logger.info(f"Deleted bucket / 删除记忆桶: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Touch bucket
    # ---------------------------------------------------------
    async def touch(
        self,
        bucket_id: str,
        actor: str = "system:touch",
        *,
        ripple: bool = True,
        raise_on_error: bool = False,
    ) -> bool:
        # One touch includes its bounded time ripple.  Keep the maintenance
        # lease across both phases so an exclusive night snapshot cannot land
        # between the primary mutation and its derived mutations.
        async with self._maintenance_barrier.shared_async():
            current_time = None
            async with self._write_guard(bucket_id):
                file_path = self._find_bucket_file(bucket_id)
                if not file_path:
                    if raise_on_error:
                        raise FileNotFoundError(bucket_id)
                    return False

                event_id = None
                try:
                    post = self._safe_load_post(file_path)
                    before = self._post_snapshot(post, file_path)
                    post["last_active"] = now_iso()
                    post["activation_count"] = post.get("activation_count", 0) + 1
                    event_id = self.audit_log.begin(
                        actor=actor,
                        action="touch",
                        bucket_id=bucket_id,
                        before=before,
                        after=self._post_snapshot(post, file_path),
                        details={"activation_increment": 1},
                    )
                    self._atomic_write_post(
                        file_path,
                        post,
                        bm25_content_changed=False,
                    )
                    self.audit_log.commit(event_id)
                    current_time = datetime.fromisoformat(
                        str(
                            event_at_from_metadata(
                                post.metadata,
                                fallback_last_active=True,
                            )
                        )
                    )
                except Exception as e:
                    self.audit_log.fail(event_id, e)
                    logger.warning(
                        f"Failed to touch bucket / 触碰桶失败: {bucket_id}: {e}"
                    )
                    if raise_on_error:
                        raise
                    return False
            if ripple and current_time is not None:
                await self._time_ripple(bucket_id, current_time)
            return current_time is not None

    async def _time_ripple(self, source_id: str, reference_time: datetime, hours: float = 48.0) -> None:
        try:
            all_buckets = await self.list_all(include_archive=False)
        except Exception:
            return

        rippled = 0
        max_ripple = 5
        for bucket in all_buckets:
            if rippled >= max_ripple:
                break
            if bucket["id"] == source_id:
                continue
            meta = bucket.get("metadata", {})
            if meta.get("pinned") or meta.get("protected") or meta.get("type") in ("permanent", "feel"):
                continue

            created_str = event_at_from_metadata(meta, fallback_last_active=True) or ""
            try:
                created = datetime.fromisoformat(str(created_str))
                delta_hours = abs((reference_time - created).total_seconds()) / 3600
            except (ValueError, TypeError):
                continue

            if delta_hours <= hours:
                async with self._write_guard(bucket["id"]):
                    event_id = None
                    try:
                        file_path = self._find_bucket_file(bucket["id"])
                        if not file_path:
                            continue
                        post = self._safe_load_post(file_path)
                        before = self._post_snapshot(post, file_path)
                        current_count = post.get("activation_count", 1)
                        post["activation_count"] = round(current_count + 0.3, 1)
                        event_id = self.audit_log.begin(
                            actor="system:time_ripple",
                            action="time_ripple",
                            bucket_id=bucket["id"],
                            before=before,
                            after=self._post_snapshot(post, file_path),
                            details={"source_id": source_id, "activation_increment": 0.3},
                        )
                        self._atomic_write_post(
                            file_path,
                            post,
                            bm25_content_changed=False,
                        )
                        self.audit_log.commit(event_id)
                        rippled += 1
                    except Exception as e:
                        self.audit_log.fail(event_id, e)
                        continue

    # ---------------------------------------------------------
    # Multi-dimensional search (core feature)
    # ---------------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = None,
        domain_filter: list[str] = None,
        world_filter: list[str] = None,
        query_valence: float = None,
        query_arousal: float = None,
        created_after: datetime = None,
        created_before: datetime = None,
        relevance_first: bool = False,
        relevance_candidate_floor: float = None,
        preloaded_buckets: list[dict] | None = None,
    ) -> list[dict]:
        if not query or not query.strip():
            return []

        limit = limit or self.max_results
        all_buckets = (
            preloaded_buckets
            if preloaded_buckets is not None
            else await self.list_all(include_archive=False)
        )

        if not all_buckets:
            return []

        # --- 修复域过滤的脆弱迭代 ---
        if domain_filter:
            filter_set = {str(d).lower() for d in domain_filter}
            candidates = []
            for b in all_buckets:
                b_domain = b["metadata"].get("domain", [])
                if isinstance(b_domain, str):
                    b_domain = [b_domain]
                elif not isinstance(b_domain, list):
                    b_domain = []
                if {str(d).lower() for d in b_domain} & filter_set:
                    candidates.append(b)
            if not candidates:
                # domain_filter 没有匹配到任何桶时，严格返回空，
                # 而不是退化成搜全部（避免用户以为过滤生效但实际没过滤）
                logger.info(
                    f"domain_filter {domain_filter} matched no buckets, returning empty"
                )
                return []
        else:
            candidates = all_buckets

        # --- World 过滤：world_filter=None 跳过；否则按 world 字段过滤 ---
        # 桶 world="通用" 在任何 world_filter 下都通过；world_filter 为空列表
        # 表示"日常模式"，只让 world 字段为空的桶 + world="通用" 通过。
        if world_filter is not None:
            wf_set = {str(w).strip() for w in world_filter}
            candidates = [
                b for b in candidates
                if world_matches(b["metadata"].get("world", ""), wf_set)
            ]

        # --- Created time range filter ---
        # --- 创建时间范围过滤：用 frontmatter 的 created 字段，无法解析的桶不过滤掉 ---
        if created_after is not None or created_before is not None:
            candidates = [
                b for b in candidates
                if _bucket_in_time_range(b, created_after, created_before)
            ]

        # Exact upstream behavior: never await a full-vault rebuild in the
        # request.  The current query uses the last complete index while a
        # fresh index is built and swapped atomically in a background thread.
        bm25_scores: dict[str, float] = {}
        bm25_shadow_ready = False
        if self._bm25_mode != "off" and self._bm25 is not None:
            if self._bm25_dirty and not self._bm25_rebuilding:
                # Never build 13k rows on every dirty request. One timer owns the
                # next generation while this request uses the old complete one.
                self._schedule_bm25_rebuild()
            try:
                # rank_bm25 scores the whole corpus. Keep that CPU loop out of
                # the async request thread just like the upstream rebuild.
                bm25_shadow_ready = bool(
                    getattr(self._bm25, "_index", None) is not None
                )
                bm25_scores = await asyncio.to_thread(self._bm25.score, query)
            except Exception as exc:
                logger.warning("[bm25] score failed; skipping this dimension: %s", exc)

        keyword_rows_ready = bool(
            bm25_shadow_ready
            and getattr(self._bm25, "_keyword_score_rows", None) is not None
            and (
                not self._bm25_unknown_dirty
                or self._bm25_rebuild_min_interval_sec > 0
            )
        )

        # The old path paid rapidfuzz.partial_ratio for every body on the event
        # loop. Reuse the resident score rows and the existing output limit +
        # relevance tie band to derive an exact candidate superset in a worker.
        # BM25 live mode keeps the full population because BM25 then changes
        # the live relevance score.
        bounded_topic_scores: dict[str, float] | None = None
        if (
            relevance_first
            and keyword_rows_ready
            and self._bm25_mode != "live"
        ):
            try:
                bounded_topic_scores = await asyncio.to_thread(
                    self._bounded_topic_scores,
                    query,
                    list(candidates),
                    limit=limit,
                )
                candidates = [
                    bucket
                    for bucket in candidates
                    if str(bucket.get("id", "")) in bounded_topic_scores
                ]
            except Exception as exc:
                bounded_topic_scores = None
                logger.warning(
                    "[keyword] bounded topic scoring failed; using legacy scan: %s",
                    exc,
                )

        scored = []
        query_casefold = query.casefold()
        z_historical = self._z_historical_ids()
        for candidate_index, bucket in enumerate(candidates, start=1):
            if candidate_index % _RECALL_CANCEL_CHECK_EVERY == 0:
                await asyncio.sleep(0)
            meta = bucket.get("metadata", {})

            try:
                bucket_id = str(bucket.get("id", ""))
                row_keys: tuple = ()
                if bounded_topic_scores is not None:
                    topic_score = bounded_topic_scores[bucket_id]
                elif keyword_rows_ready:
                    _score_row = self._keyword_score_row_for_bucket(bucket)
                    row_keys = _score_row.get("retrieval_keys", ()) or ()
                    topic_score = self._calc_topic_score_from_row(
                        query,
                        _score_row,
                    )
                else:
                    topic_score = self._calc_topic_score(query, bucket)
                    row_keys = tuple(
                        key.casefold()
                        for key in literal_retrieval_keys(
                            str(bucket.get("content", "")),
                            (bucket.get("metadata", {}) or {}).get(
                                "retrieval_keys", []
                            ),
                        )
                    )
                bm25_score = bm25_scores.get(bucket_id, 0.0)
                emotion_score = self._calc_emotion_score(query_valence, query_arousal, meta)
                time_score = self._calc_time_score(meta)
                importance_score = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                if relevance_first:
                    # Retrieval must be relevance-led.  Emotion, recency and
                    # importance are retained only as a close-score tie-break.
                    if self._bm25_mode == "live" and bm25_scores:
                        # Max-win 融合(2026-08-28):加权平均会让 fuzz 的口语碎词
                        # 高分稀释 BM25 的 IDF 强信号——账本 30 题实测:
                        # 加权 hit@1=3/hit@10=9 → max-win hit@1=6/hit@10=11。
                        # BM25 强分直接做主,fuzz 路整体降权 0.6 只作兜底。
                        normalized = max(
                            bm25_score,
                            topic_score * 0.6,
                        ) * 100.0
                        # 钥匙逐字命中=确定性导航,不吃 fuzz 的 0.6 连坐
                        # (2026-08-29 t2658 案:长 query 情绪词稀释下钥匙桶
                        # 被 max-win 挤出 top40,补键白补)。
                        if row_keys and any(
                            key in query_casefold for key in row_keys
                        ):
                            normalized = 100.0
                    else:
                        normalized = topic_score * 100.0
                    secondary_weight = self.w_emotion + self.w_time + self.w_importance
                    secondary_total = (
                        emotion_score * self.w_emotion
                        + time_score * self.w_time
                        + importance_score * self.w_importance
                    )
                    tie_break_score = (
                        secondary_total / secondary_weight * 100.0
                        if secondary_weight > 0 else 0.0
                    )
                else:
                    total = (
                        topic_score * self.w_topic
                        + emotion_score * self.w_emotion
                        + time_score * self.w_time
                        + importance_score * self.w_importance
                    )
                    weight_sum = (
                        self.w_topic + self.w_emotion + self.w_time + self.w_importance
                    )
                    if self._bm25_mode == "live" and bm25_scores:
                        total += bm25_score * self.w_bm25
                        weight_sum += self.w_bm25
                    normalized = (total / weight_sum) * 100 if weight_sum > 0 else 0
                    tie_break_score = normalized

                # Threshold check uses raw (pre-penalty) score so resolved buckets
                # remain reachable by keyword (penalty applied only to ranking).
                # 阈值用原始分数判定，确保 resolved 桶在关键词命中时仍可被搜出
                candidate_floor = (
                    (
                        self.literal_candidate_floor
                        if relevance_candidate_floor is None
                        else max(
                            0.0,
                            min(100.0, float(relevance_candidate_floor)),
                        )
                    )
                    if relevance_first else self.fuzzy_threshold
                )
                if normalized >= candidate_floor:
                    # Stale buckets get ranking penalty but stay reachable:
                    # candidate_floor above still uses their pre-penalty score.
                    # Both resolved todos and Z-axis historical facts lose on
                    # the main score so a current fact can actually outrank them.
                    is_stale = meta.get("resolved", False) or (
                        str(meta.get("id", "")).strip() in z_historical
                    )
                    if is_stale:
                        normalized *= 0.3
                        tie_break_score *= 0.3
                    scored_bucket = dict(bucket)
                    scored_bucket["score"] = round(normalized, 2)
                    if relevance_first:
                        scored_bucket["_keyword_tie_break_score"] = round(
                            tie_break_score,
                            4,
                        )
                        scored_bucket["_bm25_relevance_score"] = round(
                            bm25_score * 100.0
                            if self._bm25_mode == "live" else 0.0,
                            4,
                        )
                        if self._bm25_mode == "shadow":
                            scored_bucket["_bm25_shadow_ready"] = bm25_shadow_ready
                            scored_bucket["_bm25_shadow_generation"] = (
                                self._bm25_generation
                            )
                            if bm25_shadow_ready and bm25_scores:
                                scored_bucket["_bm25_shadow_score"] = round(
                                    bm25_score * 100.0,
                                    4,
                                )
                                shadow_relevance_weight = self.w_topic + self.w_bm25
                                scored_bucket["_bm25_shadow_relevance_score"] = round(
                                    (
                                        topic_score * self.w_topic
                                        + bm25_score * self.w_bm25
                                    )
                                    / shadow_relevance_weight
                                    * 100.0
                                    if shadow_relevance_weight > 0
                                    else 0.0,
                                    4,
                                )
                    scored.append(scored_bucket)
            except Exception as e:
                logger.warning(
                    f"Scoring failed for bucket {bucket.get('id', '?')} / "
                    f"桶评分失败: {e}"
                )
                continue

        if relevance_first:
            scored = rank_within_relevance_bands(
                scored,
                relevance_score=lambda item: item.get("score", 0),
                tie_break_score=lambda item: item.get(
                    "_keyword_tie_break_score",
                    0,
                ),
                band_width=self.keyword_relevance_tie_band,
            )
        else:
            scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ---------------------------------------------------------
    # Topic relevance sub-score (REWRITTEN: MAX-WIN + JIEBA SEGMENTATION + TYPE SAFETY)
    # 彻底重写的文本相关性算法：双向切词短路 + 最高分原则 + 极致类型安全
    # ---------------------------------------------------------
    def _calc_topic_score(self, query: str, bucket: dict) -> float:
        """
        Calculate text dimension relevance score (0~1).
        计算文本维度的相关性得分（强化版安全机制 + Jieba 切词）。
        """
        return self._calc_topic_score_from_row(
            query,
            self._build_keyword_score_row(bucket),
        )

    def _calc_topic_score_from_row(self, query: str, row: dict) -> float:
        """Run the unchanged topic formula against an immutable resident row."""
        query_lower = query.lower()
        query_parts = self._query_parts_for_search(query_lower)
        cheap_score = self._cheap_topic_score_from_row(
            query,
            query_lower,
            query_parts,
            row,
        )
        if cheap_score >= 1.0:
            return 1.0
        content_score = (
            fuzz.partial_ratio(query, row["content"]) / 100.0
            if row["content"]
            else 0.0
        )
        return max(cheap_score, content_score * 0.8)

    def _query_parts_for_search(self, query_lower: str) -> tuple[str, ...]:
        cached = self._query_parts_cache.get(query_lower)
        if cached is not None:
            return cached
        try:
            words = list(jieba.cut(query_lower))
        except Exception:
            words = query_lower.split()
        query_parts = tuple(
            p.strip()
            for p in words
            if p.strip() and p.strip() not in self.wikilink_stopwords
        ) or (query_lower,)
        if len(self._query_parts_cache) >= 128:
            self._query_parts_cache.pop(next(iter(self._query_parts_cache)))
        self._query_parts_cache[query_lower] = query_parts
        return query_parts

    # ---------------------------------------------------------
    # Emotion resonance sub-score
    # ---------------------------------------------------------
    def _calc_emotion_score(
        self, q_valence: float, q_arousal: float, meta: dict
    ) -> float:
        if q_valence is None or q_arousal is None:
            return 0.5

        try:
            b_valence = float(meta.get("valence", 0.5))
            b_arousal = float(meta.get("arousal", 0.3))
        except (ValueError, TypeError):
            return 0.5

        dist = math.sqrt((q_valence - b_valence) ** 2 + (q_arousal - b_arousal) ** 2)
        return max(0.0, 1.0 - dist / 1.414)

    # ---------------------------------------------------------
    # Time proximity sub-score
    # ---------------------------------------------------------
    def _calc_time_score(self, meta: dict) -> float:
        last_active_str = meta.get("last_active", meta.get("created", ""))
        try:
            last_active = datetime.fromisoformat(str(last_active_str))
            days = max(0.0, (datetime.now() - last_active).total_seconds() / 86400)
        except (ValueError, TypeError):
            days = 30
        return math.exp(-0.1 * days)

    # ---------------------------------------------------------
    # Z-axis currentness overlay (superseded-fact ranking penalty)
    # ---------------------------------------------------------
    def _z_historical_ids(self) -> frozenset:
        """Return active historical IDs from the fail-open Z overlay."""
        try:
            mtime = os.path.getmtime(self._z_overrides_path)
        except OSError:
            return frozenset()
        if mtime == self._z_overrides_mtime:
            return self._z_historical_cache
        ids = set()
        try:
            with open(self._z_overrides_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if str(row.get("status", "")).strip() != "active":
                        continue
                    historical_id = str(
                        row.get("historical_bucket_id", "")
                    ).strip()
                    if historical_id:
                        ids.add(historical_id)
        except OSError:
            return frozenset()
        self._z_historical_cache = frozenset(ids)
        self._z_overrides_mtime = mtime
        return self._z_historical_cache

    # ---------------------------------------------------------
    # List all buckets
    # ---------------------------------------------------------
    async def list_all_snapshot_token(
        self,
        include_archive: bool = False,
        include_nsfw: bool | None = None,
    ) -> tuple:
        """Cheap freshness token for consumers caching views derived from list_all().

        Runs the same directory snapshot list_all() uses for invalidation without
        loading or copying any bucket bodies.  Equal tokens (same key + snapshot)
        guarantee list_all() content is unchanged.
        """
        if include_nsfw is None:
            include_nsfw = getattr(self, "nsfw_active", False)
        dirs = [self.permanent_dir, self.dynamic_dir, self.feel_dir]
        if include_archive:
            dirs.append(self.archive_dir)
        if include_nsfw:
            dirs.append(self.nsfw_dir)
        _, snapshot = await self._bucket_tree_snapshot(dirs)
        return ((bool(include_archive), bool(include_nsfw)), snapshot)

    async def list_all(self, include_archive: bool = False, include_nsfw: bool | None = None) -> list[dict]:
        # include_nsfw=None → 跟随当前世界状态 self.nsfw_active（switch_world 维护）；
        # 显式 True/False 可覆盖（如 dashboard 管理界面传 True 看全部）。
        if include_nsfw is None:
            include_nsfw = getattr(self, "nsfw_active", False)
        dirs = [self.permanent_dir, self.dynamic_dir, self.feel_dir]
        if include_archive:
            dirs.append(self.archive_dir)
        if include_nsfw:
            dirs.append(self.nsfw_dir)  # 涩涩独立目录：日常默认不扫，涩涩 world 或显式 True 才加载

        cache_key = (bool(include_archive), bool(include_nsfw))
        paths, snapshot = await self._bucket_tree_snapshot(dirs)
        cached = self._list_all_cache.get(cache_key)
        if cached is not None and cached[0] == snapshot:
            return copy.deepcopy(cached[1])
        if cached is not None:
            self._mark_bm25_dirty()

        async with self._list_all_cache_lock:
            # Another request may have populated the same snapshot while this
            # one waited.  Reuse it instead of parsing thousands of files twice.
            cached = self._list_all_cache.get(cache_key)
            if cached is not None and cached[0] == snapshot:
                return copy.deepcopy(cached[1])

            generation = self._list_all_cache_generation
            buckets = []
            for loaded, file_path in enumerate(paths, start=1):
                if loaded % _RECALL_CANCEL_CHECK_EVERY == 0:
                    await asyncio.sleep(0)
                bucket = self._load_bucket(file_path)
                if bucket:
                    buckets.append(bucket)

            after_paths, after_snapshot = await self._bucket_tree_snapshot(dirs)
            if (
                generation == self._list_all_cache_generation
                and after_paths == paths
                and after_snapshot == snapshot
            ):
                self._list_all_cache[cache_key] = (snapshot, buckets)
            return copy.deepcopy(buckets)

    async def _bucket_tree_snapshot(
        self,
        dirs: list[str],
    ) -> tuple[list[str], _BucketTreeSnapshot]:
        """Enumerate visible Markdown files and their cheap change tokens."""
        paths: list[str] = []
        snapshot: list[_BucketFileRevision] = []
        scanned = 0
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    scanned += 1
                    if scanned % _RECALL_CANCEL_CHECK_EVERY == 0:
                        await asyncio.sleep(0)
                    file_path = os.path.join(root, filename)
                    paths.append(file_path)
                    try:
                        info = os.stat(file_path)
                        snapshot.append(
                            (
                                file_path,
                                int(info.st_dev),
                                int(info.st_ino),
                                int(info.st_size),
                                int(info.st_mtime_ns),
                                int(info.st_ctime_ns),
                            )
                        )
                    except OSError:
                        # Preserve list_all's historical best-effort load.  A
                        # missing stat token also prevents a false cache match
                        # once the file becomes readable again.
                        snapshot.append(
                            (file_path, None, None, None, None, None)
                        )
        return paths, tuple(snapshot)

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    async def get_stats(self) -> dict:
        stats = {
            "permanent_count": 0,
            "dynamic_count": 0,
            "archive_count": 0,
            "feel_count": 0,
            "total_size_kb": 0.0,
            "domains": {},
        }

        for subdir, key in [
            (self.permanent_dir, "permanent_count"),
            (self.dynamic_dir, "dynamic_count"),
            (self.archive_dir, "archive_count"),
            (self.feel_dir, "feel_count"),
        ]:
            if not os.path.exists(subdir):
                continue
            for root, _, files in os.walk(subdir):
                for f in files:
                    if f.endswith(".md"):
                        stats[key] += 1
                        fpath = os.path.join(root, f)
                        try:
                            stats["total_size_kb"] += os.path.getsize(fpath) / 1024
                        except OSError:
                            pass
                        domain_name = os.path.basename(root)
                        if domain_name != os.path.basename(subdir):
                            stats["domains"][domain_name] = stats["domains"].get(domain_name, 0) + 1

        return stats

    # ---------------------------------------------------------
    # Archive bucket
    # ---------------------------------------------------------
    async def archive(self, bucket_id: str, actor: str = "system") -> bool:
        async with self._write_guard(bucket_id):
            file_path = self._find_bucket_file(bucket_id)
            if not file_path:
                return False

            event_id = None
            try:
                post = self._safe_load_post(file_path)
                before = self._post_snapshot(post, file_path)
                domain = post.get("domain", ["未分类"])
                primary_domain = sanitize_name(domain[0]) if domain else "未分类"
                archive_subdir = os.path.join(self.archive_dir, primary_domain)
                os.makedirs(archive_subdir, exist_ok=True)

                dest = safe_path(archive_subdir, os.path.basename(file_path))

                post["type"] = "archived"
                event_id = self.audit_log.begin(
                    actor=actor,
                    action="archive",
                    bucket_id=bucket_id,
                    before=before,
                    after=self._post_snapshot(post, str(dest)),
                    details={
                        "source_path": os.path.abspath(file_path),
                        "destination_path": os.path.abspath(str(dest)),
                    },
                )
                self._atomic_write_post(
                    file_path,
                    post,
                    bm25_content_changed=False,
                )
                os.replace(file_path, str(dest))
                self._refresh_recall_snapshot_entry(
                    str(dest),
                    previous_path=file_path,
                    bm25_content_changed=False,
                )
                self._sync_bm25_path_transition(
                    bucket_id,
                    file_path,
                    str(dest),
                )
                self._bucket_path_cache[bucket_id] = str(dest)
                self.audit_log.commit(event_id)
            except Exception as e:
                self.audit_log.fail(event_id, e)
                logger.error(
                    f"Failed to archive bucket / 归档桶失败: {bucket_id}: {e}"
                )
                return False

        logger.info(f"Archived bucket / 归档记忆桶: {bucket_id} → archive/{primary_domain}/")
        return True

    async def relocate(
        self,
        bucket_id: str,
        *,
        actor: str = "system:relocate",
        rename_from_metadata: bool = False,
    ) -> bool:
        """Move a bucket to the directory implied by its metadata, with audit."""
        async with self._write_guard(bucket_id):
            file_path = self._find_bucket_file(bucket_id)
            if not file_path:
                return False

            event_id = None
            try:
                post = self._safe_load_post(file_path)
                domain = post.get("domain", ["未分类"])
                bucket_type = post.get("type", "dynamic")
                if post.get("pinned") or bucket_type == "permanent":
                    target_type_dir = self.permanent_dir
                elif bucket_type == "feel":
                    target_type_dir = self.feel_dir
                elif bucket_type == "archived":
                    target_type_dir = self.archive_dir
                else:
                    target_type_dir = self.dynamic_dir

                primary_domain = (
                    "沉淀物"
                    if bucket_type == "feel"
                    else sanitize_name(domain[0]) if domain else "未分类"
                )
                target_dir = os.path.join(target_type_dir, primary_domain)
                os.makedirs(target_dir, exist_ok=True)
                filename = os.path.basename(file_path)
                if rename_from_metadata:
                    bucket_name = sanitize_name(post.get("name", "")) or bucket_id
                    filename = (
                        f"{bucket_name}_{bucket_id}.md"
                        if bucket_name != bucket_id
                        else f"{bucket_id}.md"
                    )
                destination = safe_path(target_dir, filename)
                if os.path.normcase(os.path.abspath(file_path)) == os.path.normcase(
                    os.path.abspath(str(destination))
                ):
                    return True

                before = self._post_snapshot(post, file_path)
                event_id = self.audit_log.begin(
                    actor=actor,
                    action="relocate",
                    bucket_id=bucket_id,
                    before=before,
                    after=self._post_snapshot(post, str(destination)),
                    details={
                        "source_path": os.path.abspath(file_path),
                        "destination_path": os.path.abspath(str(destination)),
                    },
                )
                os.replace(file_path, str(destination))
                self._refresh_recall_snapshot_entry(
                    str(destination),
                    previous_path=file_path,
                    bm25_content_changed=False,
                )
                incremental_applied = self._sync_bm25_path_transition(
                    bucket_id,
                    file_path,
                    str(destination),
                )
                self._bucket_path_cache[bucket_id] = str(destination)
                self.invalidate_list_all_cache(
                    bm25_content_changed=not incremental_applied,
                    bm25_change_is_known=True,
                )
                self.audit_log.commit(event_id)
                return True
            except Exception as e:
                self.audit_log.fail(event_id, e)
                logger.error(f"Failed to relocate bucket / 移动桶失败: {bucket_id}: {e}")
                return False

    # ---------------------------------------------------------
    # Internal: find bucket file
    # ---------------------------------------------------------
    def _find_bucket_file(self, bucket_id: str) -> Optional[str]:
        if not bucket_id:
            return None
        cached = self._bucket_path_cache.get(bucket_id)
        if cached is not None:
            if os.path.isfile(cached):
                return cached
            self._bucket_path_cache.pop(bucket_id, None)
        # 含 nsfw_dir：按 id 精确取/改/建边永远找得到（隔离只在 list_all 召回层，不在精确取层）
        for dir_path in [self.permanent_dir, self.dynamic_dir, self.archive_dir, self.feel_dir, self.nsfw_dir]:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    name_part = fname[:-3]
                    if name_part == bucket_id or name_part.endswith(f"_{bucket_id}"):
                        path = os.path.join(root, fname)
                        self._bucket_path_cache[bucket_id] = path
                        return path
        return None

    # ---------------------------------------------------------
    # Internal: load bucket data
    # ---------------------------------------------------------
    def _safe_load_post(self, file_path: str):
        """
        Wrap frontmatter.load to tolerate dirty YAML headers that contain
        a 'content' key (would collide with the body positional arg).
        Strategy: try native load first; on collision, manually strip the
        offending key from YAML and rebuild the Post object.
        包装 frontmatter.load 以容忍 YAML 头里混入 'content' 键的脏数据
        （会和 body 位置参数撞键）。策略：先尝试原生 load；如果撞键，
        手动从 YAML 里剥掉冲突字段后重建 Post 对象。
        """
        try:
            return frontmatter.load(file_path)
        except TypeError as e:
            if "content" not in str(e):
                raise
            # Manual repair: split YAML header, drop 'content' key, rebuild
            # 手动修复：拆开 YAML 头，丢掉 content 键，重组
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            if not text.startswith("---\n"):
                raise
            end = text.find("\n---\n", 4)
            if end < 0:
                raise
            yaml_part = text[4:end]
            body = text[end + 5:]
            # Remove 'content:' line and any continuation lines until next top-level key
            # 删 content: 行及其续行，直到下一个顶级 YAML 字段
            cleaned_lines = []
            skip = False
            for line in yaml_part.splitlines(keepends=True):
                if skip:
                    # continuation if line starts with whitespace; otherwise stop skipping
                    if line and line[0] in " \t":
                        continue
                    skip = False
                if line.startswith("content:"):
                    skip = True
                    continue
                cleaned_lines.append(line)
            cleaned_yaml = "".join(cleaned_lines)
            # closing --- 必须独占一行：保留的末行若无尾换行（yaml_part 不带尾
            # 换行），重组会把它和 --- 黏成一行 → YAML 头解析失败、metadata 整段
            # 被吞进 body=静默丢字段。补一个尾换行守住。
            if cleaned_yaml and not cleaned_yaml.endswith("\n"):
                cleaned_yaml += "\n"
            cleaned_text = "---\n" + cleaned_yaml + "---\n" + body
            logger.warning(
                f"Auto-cleaned 'content' from YAML header / "
                f"自动清理YAML头里的content键: {file_path}"
            )
            return frontmatter.loads(cleaned_text)

    def _load_bucket(self, file_path: str) -> Optional[dict]:
        try:
            post = self._safe_load_post(file_path)
            return {
                "id": post.get("id", Path(file_path).stem),
                "metadata": dict(post.metadata),
                "content": post.content,
                "path": file_path,
            }
        except Exception as e:
            logger.warning(
                f"Failed to load bucket file / 加载桶文件失败: {file_path}: {e}"
            )
            return None 
