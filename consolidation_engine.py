# ============================================================
# Module: Memory Consolidation Engine (consolidation_engine.py)
# 模块：记忆整理引擎（夜班）
#
# Server-side nightly tidy: finds near-duplicate pairs and stale buckets,
# writes a review report. REPORT-FIRST and SAFE-BY-DEFAULT:
#   - NEVER deletes, archives, or merges buckets.
#   - Only optionally digests (=hide, reversible) near-identical pairs,
#     and that is OFF by default.
# 服务端每晚整理：找出疑似重复对 + 疑似过期桶，写一条复盘报告。
# 报告优先、默认安全：绝不删除/归档/合并；只在显式开启时 digest（隐藏，可逆）
# 近乎相同的对，且该行为默认关闭。
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
import logging
from datetime import datetime

from utils import PROTECTED_RESOLVE_DOMAINS

logger = logging.getLogger("ombre_brain.consolidation")

# Bucket types that are never touched by consolidation.
# 整理永不触碰的桶类型。
_EXEMPT_TYPES = ("permanent", "feel", "archived")


class ConsolidationEngine:
    """
    Nightly memory consolidation — find duplicates + stale, write a report.
    每晚记忆整理 —— 找重复 + 找过期，写报告。永不删除。
    """

    def __init__(self, config: dict, bucket_mgr, embedding_engine):
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
            return []

        candidates = [b for b in buckets if not self._is_exempt(b.get("metadata", {}))]

        # Load embeddings once.
        embs: dict[str, list] = {}
        for b in candidates:
            try:
                emb = await self.embedding_engine.get_embedding(b["id"])
            except Exception:
                emb = None
            if emb is not None:
                embs[b["id"]] = emb

        by_id = {b["id"]: b for b in candidates}
        ids = list(embs.keys())
        pairs = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                try:
                    sim = self.embedding_engine._cosine_similarity(embs[a], embs[b])
                except Exception:
                    continue
                if sim >= threshold:
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
        pairs.sort(key=lambda p: p["similarity"], reverse=True)
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
        dups = await self.find_duplicates()
        stale = await self.find_stale()

        # --- Optional, OFF by default: digest (hide, reversible) the shorter of
        #     a near-identical pair. Never delete; never touch exempt buckets. ---
        # --- 可选，默认关：把近乎相同的对里较短的那条 digest（隐藏，可逆）。永不删。 ---
        auto_digested = 0
        if self.auto_digest_near_identical:
            digested_ids: set[str] = set()
            for p in dups:
                if p["similarity"] < self.near_identical_threshold:
                    continue
                # hide the shorter one (less complete); skip if already hidden this run
                loser = p["a_id"] if p["a_len"] <= p["b_len"] else p["b_id"]
                if loser in digested_ids:
                    continue
                try:
                    ok = await self.bucket_mgr.update(loser, digested=True)
                    if ok:
                        auto_digested += 1
                        digested_ids.add(loser)
                        logger.info(f"Auto-digested near-identical / 近重自动隐藏: {loser}")
                except Exception as e:
                    logger.warning(f"Auto-digest failed / 自动隐藏失败 {loser}: {e}")

        # --- Write ONE review report bucket if there is anything to review. ---
        # --- 有东西要复盘才写一条报告桶（避免空夜刷桶）。 ---
        report_id = None
        if dups or stale:
            report_id = await self._write_report(dups, stale, auto_digested)

        result = {
            "dup_pairs": len(dups),
            "stale_count": len(stale),
            "auto_digested": auto_digested,
            "report_bucket_id": report_id,
        }
        logger.info(f"Consolidation cycle complete / 整理周期完成: {result}")
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
