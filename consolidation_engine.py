# ============================================================
# Module: Memory Consolidation Engine (consolidation_engine.py)
# 模块：记忆整理引擎（夜班）
#
# Server-side nightly tidy: finds near-duplicate pairs and stale buckets.
# REPORT-FIRST and SAFE-BY-DEFAULT:
#   - NEVER deletes, archives, or merges buckets.
#   - report_only never writes a report into the recall corpus.
#   - Legacy digest/report writes require metabolism.mode=apply.
# 服务端每晚整理：找出疑似重复对 + 疑似过期桶。
# 默认 report_only：不删除/归档/合并/digest，也不把巡检报告写回召回语料。
# 旧写入行为必须显式设置 metabolism.mode=apply。
#
# Rationale: a server-side loop runs unattended — it has no human judgment,
# so it must not make destructive or fuzzy-merge decisions. The judgment-heavy
# merges/deletes are left to a human-in-the-loop session (哥哥 reviews the
# report and acts with trace()). Mirrors the discipline baked into
# cron/nightly-consolidation.md.
# 理由：无人值守的循环没有判断力，绝不能做破坏性/模糊合并决策。拿捏的合并/删除
# 留给在线的有判断的那一步（哥哥读报告再用 trace 动手）。
#
# Depends on: bucket_manager, embedding_engine, utils. Used by: server.py
# ============================================================

import asyncio
import contextvars
import hashlib
import json
import logging
import math
import threading
from array import array
from datetime import datetime

from utils import PROTECTED_RESOLVE_DOMAINS

try:  # Optional accelerator; without numpy the exact pure-Python scorer runs.
    import numpy as _np
except Exception:  # pragma: no cover - exercised by patching _np = None
    _np = None

logger = logging.getLogger("ombre_brain.consolidation")

# Bucket types that are never touched by consolidation.
# 整理永不触碰的桶类型。
# episode/saga are derived narrative layers (kernel 3) — consolidation must
# leave them alone (they legitimately summarize many buckets, not duplicates).
# episode/saga 是内核 3 的派生叙事层，整理引擎不碰（它们本就是多桶摘要、非重复）。
_EXEMPT_TYPES = ("permanent", "feel", "archived", "episode", "saga")
_PAIRWISE_YIELD_EVERY = 64
_VECTOR_LOAD_YIELD_EVERY = 16

# ---------------------------------------------------------------------------
# Vectorized duplicate screening (2026-09-14 night-run lock incident: the
# pure-Python full scan of ~11.6k buckets, ~68M pairs, held the night run's
# exclusive maintenance lease for hours).
#
# float64 block matrix products only SCREEN pairs.  A pair whose screened
# cosine lies inside +/-_VECTOR_SCREEN_MARGIN of the threshold, or whose
# round(sim, 4) is not stable inside that margin, is recomputed with the
# original EmbeddingEngine._max_prepared_similarity kernel.  For records whose
# segment norms are 0 or inside [_VECTOR_SAFE_NORM_MIN, _VECTOR_SAFE_NORM_MAX]
# the legacy kernel can neither raise nor overflow, and the float64 screening
# error is bounded by a few * dimension * 2.2e-16 (~1e-12 at 1024 dims), far
# below the margin.  Every other record (prepare error, unsafe norm) and every
# cross-dimension pair goes through the legacy per-pair evaluation, so the
# pairs, the rounded similarities, the error strings and their order are
# identical to the legacy scorer.
# ---------------------------------------------------------------------------
_VECTOR_SCREEN_MARGIN = 1e-6
_VECTOR_SAFE_NORM_MIN = 1e-100
_VECTOR_SAFE_NORM_MAX = 1e100
_VECTOR_MAX_DIMENSION = 1 << 20
_VECTOR_BLOCK_BYTES = 48 * 1024 * 1024
_VECTOR_ASSEMBLE_YIELD_EVERY = 256
_EVENT_PAIR = 0
_EVENT_ERROR = 1


def _stock_vector_kernel(engine) -> bool:
    """True only for the unmodified EmbeddingEngine similarity kernel.

    The matrix screen mirrors that exact kernel.  Subclass or instance
    overrides (including test doubles) keep the legacy per-pair scorer.
    """

    try:
        from embedding_engine import EmbeddingEngine
    except Exception:
        return False
    if not isinstance(engine, EmbeddingEngine):
        return False
    instance_attrs = getattr(engine, "__dict__", {})
    for name in (
        "_prepare_embedding_record",
        "_max_prepared_similarity",
        "_embedding_segments",
    ):
        if name in instance_attrs:
            return False
        if getattr(type(engine), name, None) is not getattr(
            EmbeddingEngine, name, None
        ):
            return False
    return True


def _vector_threshold_supported(threshold) -> bool:
    return (
        not isinstance(threshold, bool)
        and isinstance(threshold, (int, float))
        and math.isfinite(float(threshold))
    )


def _vector_record_is_safe(record) -> bool:
    """Whether a prepared record can be screened by float64 matrices."""

    try:
        dimension = len(record[0][0])
    except Exception:
        return False
    if not 0 < dimension <= _VECTOR_MAX_DIMENSION:
        return False
    for segment in record:
        try:
            values, norm = segment
        except Exception:
            return False
        if (
            not isinstance(values, array)
            or values.typecode != "d"
            or len(values) != dimension
            or type(norm) is not float
        ):
            return False
        if norm == 0.0:
            continue
        if not _VECTOR_SAFE_NORM_MIN <= norm <= _VECTOR_SAFE_NORM_MAX:
            return False
    return True


async def _run_vector_block(function):
    """Run one bounded CPU block on a daemon worker thread.

    The event loop stays free while the block runs and regains control
    between blocks.  A daemon thread (not the default executor) never holds
    interpreter shutdown; the context is copied like the night run's leaves.
    """

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    context = contextvars.copy_context()

    def deliver(value, error) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(value)

    def run() -> None:
        value = None
        error = None
        try:
            value = context.run(function)
        except BaseException as exc:  # transported to the awaiting task
            error = exc
        try:
            loop.call_soon_threadsafe(deliver, value, error)
        except RuntimeError:
            pass  # loop already closed after cancellation

    threading.Thread(
        target=run,
        name="consolidation-vector-block",
        daemon=True,
    ).start()
    return await future


class _VectorScan:
    """One vectorized duplicate scan over a stable ``embs`` snapshot."""

    def __init__(self, engine, ids, embs, threshold, new_mask):
        self.engine = engine
        self.ids = ids
        self.embs = embs
        self.threshold = threshold
        self.new_mask = new_mask
        self.prepared: dict[int, tuple] = {}
        self.prepare_errors: dict[int, str] = {}
        self.groups: dict[int, list[int]] = {}
        self.group_of: dict[int, int] = {}
        self.slow: list[int] = []

    # -- preparation --------------------------------------------------------
    def prepare(self) -> None:
        prepare = self.engine._prepare_embedding_record
        for index, bucket_id in enumerate(self.ids):
            try:
                record = prepare(self.embs[bucket_id])
            except Exception as exc:
                self.prepare_errors[index] = type(exc).__name__
                self.slow.append(index)
                continue
            self.prepared[index] = record
            if _vector_record_is_safe(record):
                dimension = len(record[0][0])
                self.groups.setdefault(dimension, []).append(index)
                self.group_of[index] = dimension
            else:
                self.slow.append(index)

    def _in_scope(self, left: int, right: int) -> bool:
        return self.new_mask is None or self.new_mask[left] or self.new_mask[right]

    # -- legacy per-pair evaluation (bad/unsafe/cross-dimension pairs) ------
    def _legacy_event(self, left: int, right: int):
        if left in self.prepare_errors:
            return (left, right, _EVENT_ERROR, self.prepare_errors[left])
        if right in self.prepare_errors:
            return (left, right, _EVENT_ERROR, self.prepare_errors[right])
        try:
            sim = self.engine._max_prepared_similarity(
                self.prepared[left],
                self.prepared[right],
            )
        except Exception as exc:
            return (left, right, _EVENT_ERROR, type(exc).__name__)
        if sim < self.threshold:
            return None
        return (left, right, _EVENT_PAIR, round(sim, 4))

    def slow_events(self) -> list[tuple]:
        events: list[tuple] = []
        total = len(self.ids)
        slow_set = set(self.slow)
        for slow_index in self.slow:
            for other in range(total):
                if other == slow_index or (other in slow_set and other < slow_index):
                    continue
                left, right = (
                    (slow_index, other) if slow_index < other else (other, slow_index)
                )
                if not self._in_scope(left, right):
                    continue
                event = self._legacy_event(left, right)
                if event is not None:
                    events.append(event)
        if len(self.groups) > 1:
            fast = sorted(self.group_of)
            for position, left in enumerate(fast):
                left_group = self.group_of[left]
                for right in fast[position + 1:]:
                    if self.group_of[right] == left_group:
                        continue
                    if not self._in_scope(left, right):
                        continue
                    event = self._legacy_event(left, right)
                    if event is not None:
                        events.append(event)
        return events

    # -- matrix screening -----------------------------------------------------
    def build_group(self, members: list[int]):
        counts = _np.fromiter(
            (len(self.prepared[index]) for index in members),
            dtype=_np.int64,
            count=len(members),
        )
        starts = _np.zeros(len(members), dtype=_np.int64)
        if len(members) > 1:
            _np.cumsum(counts[:-1], out=starts[1:])
        total_rows = int(counts.sum())
        dimension = len(self.prepared[members[0]][0][0])
        matrix = _np.empty((total_rows, dimension), dtype=_np.float64)
        row = 0
        for index in members:
            for values, norm in self.prepared[index]:
                if norm == 0.0:
                    matrix[row].fill(0.0)
                else:
                    matrix[row] = _np.frombuffer(values, dtype=_np.float64)
                    matrix[row] /= norm
                row += 1
        return matrix, starts, counts

    def _classify(self, members, left_positions, right_positions, values):
        """Turn screened candidates into pair events (exact where needed)."""

        events: list[tuple] = []
        threshold = self.threshold
        upper = threshold + _VECTOR_SCREEN_MARGIN
        kernel = self.engine._max_prepared_similarity
        for left_position, right_position, value in zip(
            left_positions.tolist(),
            right_positions.tolist(),
            values.tolist(),
        ):
            left = members[left_position]
            right = members[right_position]
            if value >= upper:
                low = round(value - _VECTOR_SCREEN_MARGIN, 4)
                if low == round(value + _VECTOR_SCREEN_MARGIN, 4):
                    events.append((left, right, _EVENT_PAIR, low))
                    continue
            try:
                sim = kernel(self.prepared[left], self.prepared[right])
            except Exception as exc:  # unreachable for safe records
                events.append((left, right, _EVENT_ERROR, type(exc).__name__))
                continue
            if sim < threshold:
                continue
            events.append((left, right, _EVENT_PAIR, round(sim, 4)))
        return events

    def row_budget(self, total_rows: int) -> int:
        return max(1, _VECTOR_BLOCK_BYTES // (8 * max(1, total_rows)))

    def full_blocks(self, starts, counts, total_rows):
        budget = self.row_budget(total_rows)
        blocks = []
        begin = 0
        count = len(counts)
        while begin < count:
            end = begin
            rows = 0
            while end < count and (end == begin or rows + int(counts[end]) <= budget):
                rows += int(counts[end])
                end += 1
            blocks.append((begin, end))
            begin = end
        return blocks

    def score_full_block(self, members, matrix, starts, counts, begin, end):
        floor = self.threshold - _VECTOR_SCREEN_MARGIN
        total_rows = matrix.shape[0]
        row_start = int(starts[begin])
        row_end = int(starts[end]) if end < len(members) else total_rows
        products = matrix[row_start:row_end] @ matrix[row_start:].T
        if int(counts[begin:end].max()) > 1:
            products = _np.maximum.reduceat(
                products, starts[begin:end] - row_start, axis=0
            )
        if int(counts[begin:].max()) > 1:
            products = _np.maximum.reduceat(
                products, starts[begin:] - row_start, axis=1
            )
        _np.maximum(products, 0.0, out=products)
        mask = _np.triu(products >= floor, k=1)
        rows, cols = _np.nonzero(mask)
        values = products[rows, cols]
        return self._classify(members, rows + begin, cols + begin, values)

    def new_blocks(self, new_positions, counts, total_rows):
        budget = self.row_budget(total_rows)
        blocks = []
        begin = 0
        count = len(new_positions)
        while begin < count:
            end = begin
            rows = 0
            while end < count and (
                end == begin or rows + int(counts[new_positions[end]]) <= budget
            ):
                rows += int(counts[new_positions[end]])
                end += 1
            blocks.append(new_positions[begin:end])
            begin = end
        return blocks

    def score_new_block(self, members, matrix, starts, counts, is_new, block):
        floor = self.threshold - _VECTOR_SCREEN_MARGIN
        block = _np.asarray(block, dtype=_np.int64)
        row_indices = _np.concatenate(
            [
                _np.arange(int(starts[p]), int(starts[p]) + int(counts[p]))
                for p in block
            ]
        )
        products = matrix[row_indices] @ matrix.T
        if int(counts[block].max()) > 1:
            offsets = _np.zeros(len(block), dtype=_np.int64)
            if len(block) > 1:
                _np.cumsum(counts[block][:-1], out=offsets[1:])
            products = _np.maximum.reduceat(products, offsets, axis=0)
        if int(counts.max()) > 1:
            products = _np.maximum.reduceat(products, starts, axis=1)
        _np.maximum(products, 0.0, out=products)
        columns = _np.arange(len(members), dtype=_np.int64)
        # A new/new pair is produced once, from its smaller position.
        excluded = is_new[None, :] & (columns[None, :] <= block[:, None])
        mask = (products >= floor) & ~excluded
        rows, cols = _np.nonzero(mask)
        values = products[rows, cols]
        own = block[rows]
        left_positions = _np.minimum(own, cols)
        right_positions = _np.maximum(own, cols)
        return self._classify(members, left_positions, right_positions, values)


class ConsolidationEngine:
    """
    Nightly memory consolidation — find duplicates + stale, write a report.
    每晚记忆整理 —— 找重复 + 找过期，写报告。永不删除。
    """

    def __init__(self, config: dict, bucket_mgr, embedding_engine):
        metabolism_cfg = config.get("metabolism", {}) or {}
        self.metabolism_mode = str(
            metabolism_cfg.get("mode", "report_only")
        ).strip()
        if self.metabolism_mode not in {"report_only", "apply"}:
            raise ValueError("metabolism.mode must be exactly 'report_only' or 'apply'")

        cfg = config.get("consolidation", {})
        self.enabled = cfg.get("enabled", True)
        self.interval_hours = cfg.get("interval_hours", 24)
        # Pair similarity at/above this is a duplicate candidate (reported).
        # 相似度达到此值即视为重复候选（仅报告）。
        self.dup_threshold = cfg.get("dup_threshold", 0.85)
        # Near-identical: only these may be auto-digested, and only if enabled.
        # 近乎相同：只有这些可被 auto-digest，且需显式开启。
        self.near_identical_threshold = cfg.get("near_identical_threshold", 0.97)
        self.auto_digest_near_identical = cfg.get("auto_digest_near_identical", False)
        # Buckets idle longer than this many days are stale candidates.
        # 闲置超过这么多天即视为过期候选。
        self.stale_days = cfg.get("stale_days", 14)
        # Cap report size so a noisy night can't blow up a bucket.
        self.max_report_pairs = cfg.get("max_report_pairs", 50)

        self.bucket_mgr = bucket_mgr
        self.embedding_engine = embedding_engine

        self._task: asyncio.Task | None = None
        self._running = False
        self._read_errors: list[str] = []
        self._duplicate_cache_vector_digests: dict[str, str] | None = None
        self._duplicate_cache_vector_order: tuple[str, ...] | None = None
        self._duplicate_cache_fingerprints: dict[str, str] | None = None
        self._duplicate_cache_pairs: list[dict] | None = None
        self._duplicate_cache_threshold: float | None = None
        self._last_duplicate_scan_mode = "full"
        self._last_duplicate_pairs_scored = 0

    @property
    def is_running(self) -> bool:
        return self._running

    # ---------------------------------------------------------
    # Exemption: which buckets consolidation must leave alone.
    # 豁免：整理必须放过的桶（个人/情感/重要 + 固化/feel/钉选/保护）。
    # ---------------------------------------------------------
    @staticmethod
    def _is_exempt(meta: dict) -> bool:
        if meta.get("pinned") or meta.get("protected"):
            return True
        if meta.get("type") in _EXEMPT_TYPES:
            return True
        domain = meta.get("domain", [])
        if isinstance(domain, str):
            domain = [domain]
        if any(d in PROTECTED_RESOLVE_DOMAINS for d in (domain or [])):
            return True
        return False

    @staticmethod
    def _days_inactive(meta: dict) -> float:
        raw = meta.get("last_active", meta.get("created", ""))
        try:
            last = datetime.fromisoformat(str(raw))
            return max(0.0, (datetime.now() - last).total_seconds() / 86400)
        except (ValueError, TypeError):
            return 0.0  # unparseable → treat as fresh (conservative, don't flag)

    # ---------------------------------------------------------
    # find_duplicates — pairwise cosine over non-exempt buckets.
    # Read-only. Returns pairs sorted by similarity desc.
    # 找重复 —— 非豁免桶两两 cosine。只读，按相似度降序返回。
    # ---------------------------------------------------------
    async def find_duplicates(self, threshold: float = None) -> list[dict]:
        threshold = self.dup_threshold if threshold is None else threshold
        if not (self.embedding_engine and self.embedding_engine.enabled):
            return []
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"find_duplicates list failed / 列桶失败: {e}")
            self._read_errors.append(f"find_duplicates.list_all:{type(e).__name__}")
            return []

        candidates = [b for b in buckets if not self._is_exempt(b.get("metadata", {}))]

        # Load embeddings once.
        embs: dict[str, list] = {}
        for index, b in enumerate(candidates, start=1):
            try:
                emb = await self.embedding_engine.get_embedding(b["id"])
            except Exception as exc:
                self._read_errors.append(
                    f"find_duplicates.embedding:{b['id']}:{type(exc).__name__}"
                )
                emb = None
            if emb is not None:
                embs[b["id"]] = emb
            if index % _VECTOR_LOAD_YIELD_EVERY == 0:
                await asyncio.sleep(0)

        cacheable = True
        vector_digests: dict[str, str] = {}
        fingerprints: dict[str, str] = {}
        try:
            vector_digests = {
                bucket_id: self._stable_digest(value)
                for bucket_id, value in embs.items()
            }
            fingerprints = {
                bucket["id"]: self._candidate_fingerprint(bucket)
                for bucket in candidates
            }
        except (TypeError, ValueError, OverflowError):
            # Cache eligibility must never turn malformed source data into a
            # green report. The normal scorer below remains authoritative.
            cacheable = False

        can_increment = bool(
            cacheable
            and self._duplicate_cache_vector_digests is not None
            and self._duplicate_cache_vector_order is not None
            and self._duplicate_cache_fingerprints is not None
            and self._duplicate_cache_pairs is not None
            and self._duplicate_cache_threshold == float(threshold)
            and self._duplicate_cache_vector_digests.items()
            <= vector_digests.items()
            and self._duplicate_cache_fingerprints.items()
            <= fingerprints.items()
            and tuple(
                bucket_id
                for bucket_id in embs
                if bucket_id in self._duplicate_cache_vector_digests
            )
            == self._duplicate_cache_vector_order
        )
        vectorized = self._vectorized_scoring_available(threshold)
        if can_increment:
            old_ids = set(self._duplicate_cache_vector_digests or {})
            current_ids = list(embs)
            new_ids = [bucket_id for bucket_id in current_ids if bucket_id not in old_ids]
            new_set = set(new_ids)
            scored = None
            if vectorized:
                scored = await self._try_score_pairs_vectorized(
                    candidates,
                    embs,
                    threshold,
                    new_ids=new_set,
                )
            if scored is not None:
                added_pairs, pair_errors = scored
                old_count = len(current_ids) - len(new_ids)
                pairs_scored = (
                    len(current_ids) * (len(current_ids) - 1) // 2
                    - old_count * (old_count - 1) // 2
                )
            else:
                new_positions = [
                    index
                    for index, bucket_id in enumerate(current_ids)
                    if bucket_id in new_set
                ]
                pair_ids = []
                for index, left in enumerate(current_ids):
                    if left in new_set:
                        pair_ids.extend(
                            (left, right) for right in current_ids[index + 1:]
                        )
                        continue
                    pair_ids.extend(
                        (left, current_ids[position])
                        for position in new_positions
                        if position > index
                    )
                added_pairs, pair_errors = await self._score_pair_ids(
                    candidates,
                    embs,
                    pair_ids,
                    threshold,
                )
                pairs_scored = len(pair_ids)
            pairs = self._top_pairs(
                [*(self._duplicate_cache_pairs or []), *added_pairs],
                current_ids,
            )
            self._last_duplicate_scan_mode = "incremental_append"
            self._last_duplicate_pairs_scored = pairs_scored
        else:
            scored = None
            if vectorized:
                scored = await self._try_score_pairs_vectorized(
                    candidates,
                    embs,
                    threshold,
                )
            if scored is not None:
                pairs, pair_errors = scored
            else:
                pairs, pair_errors = await self._score_duplicate_pairs(
                    candidates,
                    embs,
                    threshold,
                )
            self._last_duplicate_scan_mode = "full"
            self._last_duplicate_pairs_scored = len(embs) * (len(embs) - 1) // 2
        self._read_errors.extend(pair_errors)
        if not self._read_errors and cacheable:
            self._duplicate_cache_vector_digests = dict(vector_digests)
            self._duplicate_cache_vector_order = tuple(embs)
            self._duplicate_cache_fingerprints = dict(fingerprints)
            self._duplicate_cache_pairs = [dict(pair) for pair in pairs]
            self._duplicate_cache_threshold = float(threshold)
        else:
            self._clear_duplicate_cache()
        return pairs

    @staticmethod
    def _stable_digest(value) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _candidate_fingerprint(cls, bucket: dict) -> str:
        meta = bucket.get("metadata", {})
        domain = meta.get("domain", [])
        if isinstance(domain, str):
            domain = [domain]
        return cls._stable_digest({
            "content_length": len(bucket.get("content", "") or ""),
            "domain": domain or [],
            "name": meta.get("name", bucket.get("id")),
            "pinned": bool(meta.get("pinned")),
            "protected": bool(meta.get("protected")),
            "type": meta.get("type"),
        })

    def _clear_duplicate_cache(self) -> None:
        self._duplicate_cache_vector_digests = None
        self._duplicate_cache_vector_order = None
        self._duplicate_cache_fingerprints = None
        self._duplicate_cache_pairs = None
        self._duplicate_cache_threshold = None

    async def _score_duplicate_pairs(
        self,
        candidates: list[dict],
        embs: dict[str, list],
        threshold: float,
    ) -> tuple[list[dict], list[str]]:
        """Score one stable vector snapshot without blocking the event loop."""

        ids = list(embs)
        pair_ids = (
            (left, right)
            for index, left in enumerate(ids)
            for right in ids[index + 1:]
        )
        return await self._score_pair_ids(
            candidates,
            embs,
            pair_ids,
            threshold,
        )

    async def _score_pair_ids(
        self,
        candidates: list[dict],
        embs: dict[str, list],
        pair_ids,
        threshold: float,
    ) -> tuple[list[dict], list[str]]:
        by_id = {b["id"]: b for b in candidates}
        prepared: dict[str, object] = {}
        pairs: list[dict] = []
        errors: list[str] = []
        scored = 0

        def prepared_record(bucket_id: str):
            if bucket_id not in prepared:
                prepared[bucket_id] = (
                    self.embedding_engine._prepare_embedding_record(
                        embs[bucket_id]
                    )
                )
            return prepared[bucket_id]

        for a, b in pair_ids:
            scored += 1
            sim = None
            try:
                sim = self.embedding_engine._max_prepared_similarity(
                    prepared_record(a),
                    prepared_record(b),
                )
            except Exception as exc:
                errors.append(
                    f"find_duplicates.cosine:{a}:{b}:{type(exc).__name__}"
                )
            else:
                if sim < threshold:
                    sim = None
            if sim is not None:
                ma = by_id[a]["metadata"]
                mb = by_id[b]["metadata"]
                pairs.append({
                    "a_id": a,
                    "a_name": ma.get("name", a),
                    "a_len": len(by_id[a].get("content", "") or ""),
                    "b_id": b,
                    "b_name": mb.get("name", b),
                    "b_len": len(by_id[b].get("content", "") or ""),
                    "similarity": round(sim, 4),
                })
            if scored % _PAIRWISE_YIELD_EVERY == 0:
                await asyncio.sleep(0)
        return self._top_pairs(pairs, list(embs)), errors

    def _vectorized_scoring_available(self, threshold) -> bool:
        return bool(
            _np is not None
            and _vector_threshold_supported(threshold)
            and _stock_vector_kernel(self.embedding_engine)
        )

    async def _try_score_pairs_vectorized(
        self,
        candidates: list[dict],
        embs: dict[str, list],
        threshold: float,
        *,
        new_ids: set[str] | None = None,
    ) -> tuple[list[dict], list[str]] | None:
        """Vectorized scoring; ``None`` means "use the legacy scorer"."""

        try:
            return await self._score_pairs_vectorized(
                candidates,
                embs,
                threshold,
                new_ids=new_ids,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The legacy scorer is authoritative: it reproduces any genuine
            # data error (for example a malformed candidate) exactly.
            logger.warning(
                "Vectorized duplicate scan failed, using exact legacy scan / "
                "向量化查重失败，退回逐对精算: %s",
                type(exc).__name__,
            )
            return None

    async def _score_pairs_vectorized(
        self,
        candidates: list[dict],
        embs: dict[str, list],
        threshold: float,
        *,
        new_ids: set[str] | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Exact equivalent of ``_score_pair_ids`` over the full scan
        (``new_ids is None``) or the incremental append scan (pairs with at
        least one id in ``new_ids``), with block matrix screening."""

        ids = list(embs)
        if len(ids) < 2 or (new_ids is not None and not new_ids):
            return self._top_pairs([], ids), []
        new_mask = (
            None if new_ids is None else [bucket_id in new_ids for bucket_id in ids]
        )
        scan = _VectorScan(self.embedding_engine, ids, embs, threshold, new_mask)
        await _run_vector_block(scan.prepare)

        events: list[tuple] = []
        for dimension in sorted(scan.groups):
            members = scan.groups[dimension]
            if len(members) < 2:
                continue
            matrix, starts, counts = await _run_vector_block(
                lambda members=members: scan.build_group(members)
            )
            total_rows = matrix.shape[0]
            if new_mask is None:
                for begin, end in scan.full_blocks(starts, counts, total_rows):
                    events.extend(
                        await _run_vector_block(
                            lambda begin=begin, end=end: scan.score_full_block(
                                members, matrix, starts, counts, begin, end
                            )
                        )
                    )
            else:
                is_new = _np.fromiter(
                    (new_mask[index] for index in members),
                    dtype=bool,
                    count=len(members),
                )
                new_positions = [
                    position
                    for position, index in enumerate(members)
                    if new_mask[index]
                ]
                for block in scan.new_blocks(new_positions, counts, total_rows):
                    events.extend(
                        await _run_vector_block(
                            lambda block=block: scan.score_new_block(
                                members, matrix, starts, counts, is_new, block
                            )
                        )
                    )
            del matrix
        if scan.slow or len(scan.groups) > 1:
            events.extend(await _run_vector_block(scan.slow_events))

        events.sort(key=lambda event: (event[0], event[1]))
        by_id = {b["id"]: b for b in candidates}
        pairs: list[dict] = []
        errors: list[str] = []
        for position, (left, right, kind, payload) in enumerate(events, start=1):
            a = ids[left]
            b = ids[right]
            if kind == _EVENT_ERROR:
                errors.append(f"find_duplicates.cosine:{a}:{b}:{payload}")
            else:
                ma = by_id[a]["metadata"]
                mb = by_id[b]["metadata"]
                pairs.append({
                    "a_id": a,
                    "a_name": ma.get("name", a),
                    "a_len": len(by_id[a].get("content", "") or ""),
                    "b_id": b,
                    "b_name": mb.get("name", b),
                    "b_len": len(by_id[b].get("content", "") or ""),
                    "similarity": payload,
                })
            if position % _VECTOR_ASSEMBLE_YIELD_EVERY == 0:
                await asyncio.sleep(0)
        return self._top_pairs(pairs, ids), errors

    def _top_pairs(
        self,
        pairs: list[dict],
        vector_order: list[str],
    ) -> list[dict]:
        order = {
            bucket_id: index
            for index, bucket_id in enumerate(vector_order)
        }
        pairs.sort(
            key=lambda pair: (
                -float(pair["similarity"]),
                order[str(pair["a_id"])],
                order[str(pair["b_id"])],
            )
        )
        return pairs[: self.max_report_pairs]

    # ---------------------------------------------------------
    # find_stale — non-exempt, unresolved buckets idle > days.
    # Read-only. Personal/emotional/important domains are exempt.
    # 找过期 —— 非豁免、未解决、闲置超 days 的桶。只读，个人/情感/重要域豁免。
    # ---------------------------------------------------------
    async def find_stale(self, days: int = None) -> list[dict]:
        days = self.stale_days if days is None else days
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"find_stale list failed / 列桶失败: {e}")
            self._read_errors.append(f"find_stale.list_all:{type(e).__name__}")
            return []

        stale = []
        for b in buckets:
            meta = b.get("metadata", {})
            if self._is_exempt(meta):
                continue
            if meta.get("resolved", False):
                continue  # already settled, not stale-actionable
            idle = self._days_inactive(meta)
            if idle > days:
                domain = meta.get("domain", [])
                if isinstance(domain, str):
                    domain = [domain]
                stale.append({
                    "id": b["id"],
                    "name": meta.get("name", b["id"]),
                    "days_inactive": round(idle, 1),
                    "importance": int(meta.get("importance", 5)),
                    "domain": domain,
                })
        stale.sort(key=lambda s: s["days_inactive"], reverse=True)
        return stale

    # ---------------------------------------------------------
    # One consolidation cycle. REPORT-FIRST, NEVER deletes.
    # 一轮整理。报告优先，绝不删除。
    # ---------------------------------------------------------
    async def run_consolidation_cycle(self) -> dict:
        self._read_errors = []
        dups = await self.find_duplicates()
        stale = await self.find_stale()
        if self._read_errors:
            result = {
                "ok": False,
                "mode": self.metabolism_mode,
                "dup_pairs": 0,
                "stale_count": 0,
                "auto_digested": 0,
                "would_digest": [],
                "would_create_report": False,
                "report_bucket_id": None,
                "errors": list(self._read_errors),
            }
            logger.error(
                "Consolidation read failed / 整理读取失败: errors=%d",
                len(result["errors"]),
            )
            return result

        # --- Optional, OFF by default: digest (hide, reversible) the shorter of
        #     a near-identical pair. Never delete; never touch exempt buckets. ---
        # --- 可选，默认关：把近乎相同的对里较短的那条 digest（隐藏，可逆）。永不删。 ---
        auto_digested = 0
        operation_errors: list[str] = []
        would_digest: list[str] = []
        if self.auto_digest_near_identical:
            digested_ids: set[str] = set()
            for p in dups:
                if p["similarity"] < self.near_identical_threshold:
                    continue
                # hide the shorter one (less complete); skip if already hidden this run
                loser = p["a_id"] if p["a_len"] <= p["b_len"] else p["b_id"]
                if loser in digested_ids:
                    continue
                would_digest.append(str(loser))
                digested_ids.add(loser)
                if self.metabolism_mode == "apply":
                    try:
                        ok = await self.bucket_mgr.update(loser, digested=True)
                        if ok:
                            auto_digested += 1
                            logger.info(f"Auto-digested near-identical / 近重自动隐藏: {loser}")
                    except Exception as e:
                        logger.warning(f"Auto-digest failed / 自动隐藏失败 {loser}: {e}")
                        operation_errors.append(
                            f"auto_digest:{loser}:{type(e).__name__}"
                        )

        # --- Write ONE review report bucket if there is anything to review. ---
        # --- 有东西要复盘才写一条报告桶（避免空夜刷桶）。 ---
        report_id = None
        if self.metabolism_mode == "apply" and (dups or stale):
            report_id = await self._write_report(dups, stale, auto_digested)

        result = {
            "ok": not operation_errors,
            "mode": self.metabolism_mode,
            "dup_pairs": len(dups),
            "stale_count": len(stale),
            "auto_digested": auto_digested,
            "would_digest": would_digest,
            "would_create_report": bool(dups or stale),
            "duplicate_candidates": dups,
            "stale_candidates": stale,
            "report_bucket_id": report_id,
            "errors": operation_errors,
            "duplicate_scan_mode": self._last_duplicate_scan_mode,
            "duplicate_pairs_scored": self._last_duplicate_pairs_scored,
        }
        logger.info(
            "Consolidation cycle complete / 整理周期完成: "
            "ok=%s mode=%s dup_pairs=%d stale_count=%d "
            "would_digest=%d errors=%d scan=%s scored=%d",
            result["ok"],
            result["mode"],
            result["dup_pairs"],
            result["stale_count"],
            len(result["would_digest"]),
            len(result["errors"]),
            result["duplicate_scan_mode"],
            result["duplicate_pairs_scored"],
        )
        return result

    async def _write_report(self, dups: list[dict], stale: list[dict], auto_digested: int) -> str | None:
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"# 夜班整理报告 · {today}", ""]
        lines.append(
            "本报告只是**复盘提示，没有动你的记忆**（除显式开启的近重隐藏外）。"
            "拿捏的合并/删除请在线时用 trace 自己定。个人/情感/重要域已豁免，不在下表。"
        )
        if auto_digested:
            lines.append(f"\n本轮自动隐藏（digest，可逆）近乎相同桶：{auto_digested} 条。")

        if dups:
            lines.append(f"\n## 疑似重复 {len(dups)} 对（相似度≥{self.dup_threshold}）")
            for p in dups:
                tier = "建议删旧留全" if p["similarity"] >= self.near_identical_threshold else "看是否同一件事再合"
                lines.append(
                    f"- {p['similarity']:.2f} · [{p['a_name']}]({p['a_id']}, {p['a_len']}字) "
                    f"↔ [{p['b_name']}]({p['b_id']}, {p['b_len']}字) — {tier}"
                )
        if stale:
            lines.append(f"\n## 疑似过期 {len(stale)} 条（闲置>{self.stale_days}天，未解决，非豁免）")
            for s in stale:
                lines.append(
                    f"- 闲置{s['days_inactive']:.0f}天 · imp{s['importance']} · "
                    f"[{s['name']}]({s['id']}) · {'/'.join(s['domain'])} "
                    f"— 做完了就 resolve（销账前先查证），拿不准就留"
                )
        content = "\n".join(lines)

        try:
            return await self.bucket_mgr.create(
                content=content,
                name="夜班整理报告",
                tags=["夜班", "记忆整理", "报告"],
                domain=["记忆整理"],
                importance=3,
                valence=0.5,
                arousal=0.2,
                bucket_type="dynamic",
                actor="night:consolidation",
            )
        except Exception as e:
            logger.error(f"Failed to write consolidation report / 写报告失败: {e}")
            return None

    # ---------------------------------------------------------
    # Background task management (mirrors DecayEngine).
    # 后台任务管理（仿衰减引擎）。
    # ---------------------------------------------------------
    async def ensure_started(self) -> None:
        if self.enabled and not self._running:
            await self.start()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._background_loop())
        logger.info(
            f"Consolidation engine started, interval: {self.interval_hours}h / "
            f"整理引擎已启动，间隔: {self.interval_hours} 小时"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Consolidation engine stopped / 整理引擎已停止")

    async def _background_loop(self) -> None:
        while self._running:
            try:
                await self.run_consolidation_cycle()
            except Exception as e:
                logger.error(f"Consolidation cycle error / 整理周期出错: {e}")
            try:
                await asyncio.sleep(self.interval_hours * 3600)
            except asyncio.CancelledError:
                break
