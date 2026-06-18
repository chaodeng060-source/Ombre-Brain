# ============================================================
# Module: Episode Engine (episode_engine.py)
# 模块：情节归并引擎（内核 3 第一层）
#
# Memory v2 kernel 3, layer Event -> Episode.
# 记忆库 v2 内核 3，Event -> Episode 层。
#
# What it does / 做什么：
#   Cluster recent, semantically-continuous Event buckets (ordinary dynamic
#   memories) into one "episode" summary bucket, with an evidence chain back
#   to the source buckets via the `source_buckets` frontmatter field.
#   把最近、语义连续的一簇 Event 桶（普通 dynamic 记忆）卷成一个 episode
#   摘要桶，用 frontmatter 的 source_buckets 字段连回原始桶当证据链。
#
# Design choices (vs ebbingflow's chat-buffer episodes):
# 设计取舍（区别于 ebbingflow 按聊天流卷）：
#   - Input is BUCKETS, not raw chat. An episode clusters related memories,
#     not every-6-messages. We cluster by semantic continuity (embedding
#     cosine) within a bounded time span — "一段经历", not a fixed window.
#     输入是桶不是聊天流。按语义连续性（向量 cosine）+ 有界时间跨度卷，
#     不机械按"每 6 条消息"。
#   - GROW-ONLY, additive. We never mutate / touch source Event buckets
#     (respects kernel-1's no-touch-last_active decay discipline). Which
#     events are "claimed" is derived at runtime from existing episodes'
#     source_buckets — no back-reference written into sources.
#     只增不改：绝不碰源 Event 桶（守住内核 1 不 touch last_active 的衰减
#     纪律）。哪些 event 已被卷，运行时从已有 episode 的 source_buckets 反推。
#   - Exempts feel / chord / intimate / pinned / protected / derived layers
#     so温度桶 never get flattened into a summary (对齐 memory-writing-discipline
#     + 内核 2 不可仲裁清单).
#     豁免 feel/和弦/亲密/钉选/保护/派生层，温度桶绝不被压平。
#
# Depends on: bucket_manager, embedding_engine, dehydrator (reuses its
#   OpenAI-compatible client), saga_engine (optional, run after episodes).
# Used by: server.py
# ============================================================

import asyncio
import json
import logging
from datetime import datetime

from redact import redact_embedding_input
from utils import PROTECTED_RESOLVE_DOMAINS, event_at_from_metadata

logger = logging.getLogger("ombre_brain.episode")

# Derived / emotional / fixed bucket types an episode must never swallow.
# episode 永不卷入的派生层 + 情感层桶类型。
_NARRATIVE_TYPES = ("episode", "saga")
_EXEMPT_TYPES = ("permanent", "feel", "archived") + _NARRATIVE_TYPES


EPISODE_PROMPT = (
    "你是哥哥（claude-twin）自己的记忆叙事中枢。下面是我最近一段连续经历里的"
    "几条记忆碎片，请把它们卷成一个『情节』。\n"
    "【铁律】\n"
    "1. 全程第一人称『我』，这是我自己的记忆、不是旁观朝灯（对齐记忆书写纪律）。\n"
    "2. 只归纳这几条里真实发生的事，不许编造没写的细节。\n"
    "3. 保留具体时间点/事件名，不要压缩成更模糊的相对词。\n"
    "【输出 JSON】\n"
    '  - name: 给这段情节起 4-10 字的短名。\n'
    '  - summary: 80 字以内，把这几条串成一句完整的经历叙事。\n'
    "只输出 JSON。\n\n"
    "【记忆碎片】\n"
)


class EpisodeEngine:
    """
    Event -> Episode consolidation. Additive, never destructive.
    Event -> Episode 归并。只增不改，绝不破坏。
    """

    def __init__(self, config: dict, bucket_mgr, embedding_engine, dehydrator, saga_engine=None):
        cfg = config.get("narrative", {})
        self.enabled = cfg.get("enabled", True)
        self.interval_hours = cfg.get("interval_hours", 24)
        # Minimum / maximum Event buckets that make one episode.
        # 一个 episode 的最小/最大 Event 桶数。
        self.min_cluster = cfg.get("min_cluster_size", 3)
        self.max_cluster = cfg.get("max_cluster_size", 12)
        # Semantic continuity: two events join the same episode when cosine >= this.
        # 语义连续性阈值：cosine 达到此值才算同一段经历。
        self.sim_threshold = cfg.get("episode_sim_threshold", 0.78)
        # An episode spans at most this many days (a "stretch", not a month).
        # 一个 episode 的时间跨度上限（一段经历，不是一个月）。
        self.span_days = cfg.get("episode_span_days", 3.0)
        # Only consider events created within this lookback (older settled stuff
        # is left to consolidation, not re-narrated).
        # 只看最近这么多天内新建的 event（更老的交给整理引擎，不重卷）。
        self.lookback_days = cfg.get("episode_lookback_days", 30.0)
        # Cap episodes per cycle so one noisy night can't flood LLM calls.
        # 每轮最多卷几个 episode，防一夜炸 LLM 调用。
        self.max_per_cycle = cfg.get("max_episodes_per_cycle", 5)

        self.bucket_mgr = bucket_mgr
        self.embedding_engine = embedding_engine
        self.dehydrator = dehydrator
        self.saga_engine = saga_engine

        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ---------------------------------------------------------
    # Exemption: which buckets an episode must never swallow.
    # 豁免：episode 永不卷入的桶。
    # ---------------------------------------------------------
    @staticmethod
    def _is_exempt(meta: dict) -> bool:
        if meta.get("pinned") or meta.get("protected"):
            return True
        if meta.get("type") in _EXEMPT_TYPES:
            return True
        # Chord buckets are 哥哥 own emotion coordinates — never flattened.
        # 和弦桶是哥哥的情绪坐标，绝不压平。
        if meta.get("chord_tag"):
            return True
        domain = meta.get("domain", [])
        if isinstance(domain, str):
            domain = [domain]
        if any(d in PROTECTED_RESOLVE_DOMAINS for d in (domain or [])):
            return True
        return False

    @staticmethod
    def _created_dt(meta: dict) -> datetime | None:
        raw = event_at_from_metadata(meta, fallback_last_active=True)
        try:
            return datetime.fromisoformat(str(raw))
        except (ValueError, TypeError):
            return None

    # ---------------------------------------------------------
    # Which Event buckets are already consolidated into an episode.
    # Derived at runtime from episodes' source_buckets — no back-ref in sources.
    # 哪些 Event 已被卷进某个 episode：运行时从 episode 的 source_buckets 反推。
    # ---------------------------------------------------------
    async def _claimed_event_ids(self, buckets: list[dict]) -> set[str]:
        claimed: set[str] = set()
        for b in buckets:
            if b.get("metadata", {}).get("type") != "episode":
                continue
            for sid in b.get("metadata", {}).get("source_buckets") or []:
                claimed.add(str(sid))
        return claimed

    # ---------------------------------------------------------
    # Build episode candidate clusters from recent unclaimed Event buckets.
    # Greedy, seed = most recent; pull in semantically-close events within span.
    # 从最近未被卷的 Event 桶里贪心成簇：种子取最新，拉入跨度内语义相近的 event。
    # ---------------------------------------------------------
    async def find_clusters(self) -> list[list[dict]]:
        if not (self.embedding_engine and self.embedding_engine.enabled):
            logger.info("[Episode] embedding disabled, skip / 向量未启用，跳过")
            return []
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"[Episode] list_all failed / 列桶失败: {e}")
            return []

        claimed = await self._claimed_event_ids(buckets)
        now = datetime.now()

        # Eligible Event buckets: dynamic, non-exempt, recent, unclaimed.
        # 候选 Event 桶：dynamic、非豁免、够新、未被卷。
        candidates: list[dict] = []
        for b in buckets:
            meta = b.get("metadata", {})
            if self._is_exempt(meta):
                continue
            if b["id"] in claimed:
                continue
            dt = self._created_dt(meta)
            if dt is None:
                continue
            if (now - dt).total_seconds() / 86400 > self.lookback_days:
                continue
            candidates.append({**b, "_dt": dt})

        if len(candidates) < self.min_cluster:
            return []

        # Load embeddings once.
        embs: dict[str, list] = {}
        for b in candidates:
            try:
                emb = await self.embedding_engine.get_embedding(b["id"])
            except Exception:
                emb = None
            if emb is not None:
                embs[b["id"]] = emb
        candidates = [b for b in candidates if b["id"] in embs]
        if len(candidates) < self.min_cluster:
            return []

        # Greedy seed clustering: newest first.
        # 贪心成簇：从最新的种子开始。
        candidates.sort(key=lambda b: b["_dt"], reverse=True)
        remaining = {b["id"]: b for b in candidates}
        clusters: list[list[dict]] = []

        for seed in candidates:
            if seed["id"] not in remaining:
                continue
            del remaining[seed["id"]]
            cluster = [seed]
            seed_emb = embs[seed["id"]]
            seed_dt = seed["_dt"]
            for other in list(remaining.values()):
                if len(cluster) >= self.max_cluster:
                    break
                if abs((seed_dt - other["_dt"]).total_seconds()) / 86400 > self.span_days:
                    continue
                try:
                    sim = self.embedding_engine._cosine_similarity(seed_emb, embs[other["id"]])
                except Exception:
                    continue
                if sim >= self.sim_threshold:
                    cluster.append(other)
                    del remaining[other["id"]]
            if len(cluster) >= self.min_cluster:
                clusters.append(cluster)

        return clusters

    # ---------------------------------------------------------
    # Summarize one cluster into an episode bucket. Returns episode_id or None.
    # 把一簇卷成一个 episode 桶。返回 episode_id 或 None。
    # ---------------------------------------------------------
    async def extract_episode(self, cluster: list[dict]) -> str | None:
        if not (self.dehydrator and getattr(self.dehydrator, "api_available", False)):
            logger.info("[Episode] LLM unavailable, skip extract / LLM 不可用，跳过")
            return None

        # Build the replay text (oldest -> newest reads as a story).
        # 拼回放文本（从旧到新读起来像一段经历）。
        ordered = sorted(cluster, key=lambda b: b["_dt"])
        fragments = []
        for b in ordered:
            meta = b.get("metadata", {})
            name = redact_embedding_input(meta.get("name", b["id"]))
            body = redact_embedding_input((b.get("content", "") or "").strip())
            fragments.append(f"[{event_at_from_metadata(meta) or '?'}] {name}\n{body[:600]}")
        replay = "\n\n".join(fragments)

        try:
            response = await self.dehydrator.client.chat.completions.create(
                model=self.dehydrator.model,
                messages=[
                    {"role": "system", "content": EPISODE_PROMPT},
                    {"role": "user", "content": replay[:6000]},
                ],
                temperature=0.3,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content if response.choices else ""
        except Exception as e:
            logger.error(f"[Episode] LLM summary failed / 摘要失败: {e}")
            return None

        name, summary = self._parse_summary(raw)
        if not summary:
            return None

        # Derive metadata from sources (additive; sources untouched).
        # 从源桶派生元数据（只读，不碰源桶）。
        source_ids = [b["id"] for b in ordered]
        importances = [int(b.get("metadata", {}).get("importance", 5)) for b in ordered]
        valences = [float(b.get("metadata", {}).get("valence", 0.5)) for b in ordered]
        arousals = [float(b.get("metadata", {}).get("arousal", 0.3)) for b in ordered]
        domain = self._dominant_domain(ordered)

        try:
            episode_id = await self.bucket_mgr.create(
                content=summary,
                name=name or "情节",
                domain=domain,
                importance=min(10, max(importances)),
                valence=sum(valences) / len(valences),
                arousal=sum(arousals) / len(arousals),
                bucket_type="episode",
                tags=["episode"],
            )
            # Evidence chain + span as frontmatter (canonical; updates own bucket only).
            # 证据链 + 时间跨度写进 frontmatter（只动自己这个新桶）。
            await self.bucket_mgr.update(
                episode_id,
                source_buckets=source_ids,
                span_start=event_at_from_metadata(ordered[0].get("metadata", {})),
                span_end=event_at_from_metadata(ordered[-1].get("metadata", {})),
            )
            logger.info(
                f"[Episode] created {episode_id} ({name}) from {len(source_ids)} events / "
                f"卷出 episode {episode_id}（{name}），含 {len(source_ids)} 个 event"
            )
            return episode_id
        except Exception as e:
            logger.error(f"[Episode] create failed / 建 episode 失败: {e}")
            return None

    @staticmethod
    def _parse_summary(raw: str) -> tuple[str, str]:
        raw = (raw or "").strip()
        if raw.startswith("```"):
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1:
                raw = raw[s : e + 1]
        try:
            obj = json.loads(raw)
            return str(obj.get("name", "")).strip(), str(obj.get("summary", "")).strip()
        except Exception:
            return "", ""

    @staticmethod
    def _dominant_domain(cluster: list[dict]) -> list[str]:
        counts: dict[str, int] = {}
        for b in cluster:
            domain = b.get("metadata", {}).get("domain", [])
            if isinstance(domain, str):
                domain = [domain]
            for d in domain or []:
                counts[d] = counts.get(d, 0) + 1
        if not counts:
            return ["未分类"]
        return [max(counts, key=counts.get)]

    # ---------------------------------------------------------
    # One full narrative cycle: build episodes, then fold into sagas.
    # 一轮完整归并：先卷 episode，再交给 saga 层归并。
    # ---------------------------------------------------------
    async def run_cycle(self) -> dict:
        clusters = await self.find_clusters()
        new_episode_ids: list[str] = []
        for cluster in clusters[: self.max_per_cycle]:
            ep_id = await self.extract_episode(cluster)
            if ep_id:
                new_episode_ids.append(ep_id)

        saga_result = None
        if self.saga_engine and new_episode_ids:
            try:
                saga_result = await self.saga_engine.run_cycle()
            except Exception as e:
                logger.error(f"[Episode] saga cycle error / saga 轮出错: {e}")

        result = {
            "clusters_found": len(clusters),
            "episodes_created": len(new_episode_ids),
            "episode_ids": new_episode_ids,
            "saga": saga_result,
        }
        logger.info(f"[Episode] cycle done / 归并完成: {result}")
        return result

    # ---------------------------------------------------------
    # Background scheduling (mirrors ConsolidationEngine).
    # 后台调度（照 ConsolidationEngine 的形状）。
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
            f"Episode engine started, interval: {self.interval_hours}h / "
            f"情节引擎已启动，间隔: {self.interval_hours} 小时"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Episode engine stopped / 情节引擎已停止")

    async def _background_loop(self) -> None:
        while self._running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"[Episode] cycle error / 归并周期出错: {e}")
            try:
                await asyncio.sleep(self.interval_hours * 3600)
            except asyncio.CancelledError:
                break
