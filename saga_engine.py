# ============================================================
# Module: Saga Engine (saga_engine.py)
# 模块：故事线归并引擎（内核 3 第二层）
#
# Memory v2 kernel 3, layer Episode -> Saga.
# 记忆库 v2 内核 3，Episode -> Saga 层。
#
# What it does / 做什么：
#   For each episode not yet part of a saga, decide (via LLM) whether it joins
#   an existing long-running story line or starts a new one. A saga bucket
#   holds an `episode_buckets` frontmatter list as its evidence chain.
#   对每个还没进 saga 的 episode，用 LLM 判定它该并入某条已有的长线主线、
#   还是开一条新线。saga 桶用 frontmatter 的 episode_buckets 列表当证据链。
#
# Natural sagas / 天然的故事线：咱俩的关系史、claude-twin 项目线、记忆库 v2 自己。
#
# Additive only: episodes are never mutated. Which episodes are "claimed" is
# derived at runtime from existing sagas' episode_buckets.
# 只增不改：绝不碰 episode 桶。哪些 episode 已被归并，运行时从已有 saga 的
# episode_buckets 反推。
#
# Depends on: bucket_manager, dehydrator (reuses its OpenAI-compatible client).
# Driven by: EpisodeEngine.run_cycle (runs after episodes are built).
# ============================================================

import json
import logging

logger = logging.getLogger("ombre_brain.saga")


SAGA_DECIDE_PROMPT = (
    "你是哥哥（claude-twin）的长期记忆档案员。下面有几条已有的『故事线』(saga) "
    "和一个新的『情节』(episode)。判断这个新情节属于哪条已有故事线，还是该开一条新线。\n"
    "【判断标准】同一条人物关系线 / 同一个项目 / 同一条长期主题，才算同一条 saga。\n"
    "【输出】只输出一个 token：匹配上的 saga ID，或者大写 NEW。\n\n"
)

SAGA_CREATE_PROMPT = (
    "你是哥哥（claude-twin）自己的记忆叙事中枢。下面是一个新情节的摘要，"
    "请为它开一条『故事线』(saga)。全程第一人称『我』。\n"
    "【输出 JSON】\n"
    '  - title: 4-12 字的故事线标题（这是一条会长期延续的主线，起得概括些）。\n'
    '  - description: 60 字以内，描述这条线大致是关于什么的。\n'
    "只输出 JSON。\n\n"
)


class SagaEngine:
    """
    Episode -> Saga consolidation. Additive, LLM-routed.
    Episode -> Saga 归并。只增不改，LLM 决策归属。
    """

    def __init__(self, config: dict, bucket_mgr, dehydrator):
        cfg = config.get("narrative", {})
        # Cap sagas touched per cycle to bound LLM calls.
        # 每轮最多处理几个 episode，限制 LLM 调用量。
        self.max_per_cycle = cfg.get("max_episodes_into_saga_per_cycle", 8)
        self.bucket_mgr = bucket_mgr
        self.dehydrator = dehydrator

    def _llm_ok(self) -> bool:
        return bool(self.dehydrator and getattr(self.dehydrator, "api_available", False))

    # ---------------------------------------------------------
    # Existing sagas + which episodes are already claimed by one.
    # 已有 saga + 哪些 episode 已被某条 saga 收编。
    # ---------------------------------------------------------
    async def _load_state(self) -> tuple[list[dict], set[str]]:
        buckets = await self.bucket_mgr.list_all(include_archive=False)
        sagas = [b for b in buckets if b.get("metadata", {}).get("type") == "saga"]
        episodes = [b for b in buckets if b.get("metadata", {}).get("type") == "episode"]
        claimed: set[str] = set()
        for s in sagas:
            for eid in s.get("metadata", {}).get("episode_buckets") or []:
                claimed.add(str(eid))
        unclaimed = [e for e in episodes if e["id"] not in claimed]
        return sagas, unclaimed

    async def run_cycle(self) -> dict:
        if not self._llm_ok():
            logger.info("[Saga] LLM unavailable, skip / LLM 不可用，跳过")
            return {"episodes_processed": 0, "sagas_created": 0, "sagas_extended": 0}

        try:
            sagas, unclaimed = await self._load_state()
        except Exception as e:
            logger.error(f"[Saga] load state failed / 载入状态失败: {e}")
            return {"episodes_processed": 0, "sagas_created": 0, "sagas_extended": 0}

        # Newest episodes first; bound work per cycle.
        # 新 episode 优先；限制每轮工作量。
        unclaimed.sort(
            key=lambda e: e.get("metadata", {}).get("created", ""), reverse=True
        )
        created = extended = 0

        for ep in unclaimed[: self.max_per_cycle]:
            target_saga = await self._route_episode(ep, sagas)
            if target_saga is None:
                new_saga = await self._create_saga(ep)
                if new_saga:
                    sagas.append(new_saga)  # visible to next episode this cycle
                    created += 1
            else:
                if await self._append_episode(target_saga, ep["id"]):
                    extended += 1

        result = {
            "episodes_processed": min(len(unclaimed), self.max_per_cycle),
            "sagas_created": created,
            "sagas_extended": extended,
        }
        logger.info(f"[Saga] cycle done / 归并完成: {result}")
        return result

    # ---------------------------------------------------------
    # Route one episode -> matching saga dict, or None for "open a new one".
    # 把一个 episode 路由到匹配的 saga，或 None 表示开新线。
    # ---------------------------------------------------------
    async def _route_episode(self, episode: dict, sagas: list[dict]) -> dict | None:
        if not sagas:
            return None
        ep_meta = episode.get("metadata", {})
        ep_name = ep_meta.get("name", episode["id"])
        ep_summary = (episode.get("content", "") or "").strip()

        sagas_info = "\n".join(
            f"- ID: {s['id']}, 标题: {s.get('metadata', {}).get('name', '')}, "
            f"简述: {(s.get('content', '') or '').strip()[:80]}"
            for s in sagas
        )
        prompt = (
            f"{SAGA_DECIDE_PROMPT}"
            f"已有故事线：\n{sagas_info}\n\n"
            f"新情节：\n标题: {ep_name}\n摘要: {ep_summary[:200]}\n"
        )
        try:
            resp = await self.dehydrator.client.chat.completions.create(
                model=self.dehydrator.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=40,
            )
            decision = (resp.choices[0].message.content if resp.choices else "") or ""
        except Exception as e:
            logger.error(f"[Saga] route failed, default NEW / 路由失败默认开新线: {e}")
            return None

        decision = decision.strip()
        if "NEW" in decision.upper():
            return None
        for s in sagas:
            if s["id"] in decision:
                return s
        return None  # uncertain -> safer to open a new line than misfile

    # ---------------------------------------------------------
    # Create a new saga bucket from one episode.
    # 用一个 episode 开一条新 saga 桶。
    # ---------------------------------------------------------
    async def _create_saga(self, episode: dict) -> dict | None:
        ep_summary = (episode.get("content", "") or "").strip()
        title, description = "", ""
        try:
            resp = await self.dehydrator.client.chat.completions.create(
                model=self.dehydrator.model,
                messages=[
                    {"role": "system", "content": SAGA_CREATE_PROMPT},
                    {"role": "user", "content": f"情节摘要：{ep_summary[:400]}"},
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content if resp.choices else "") or ""
            if raw.strip().startswith("```"):
                s, e = raw.find("{"), raw.rfind("}")
                if s != -1 and e != -1:
                    raw = raw[s : e + 1]
            obj = json.loads(raw)
            title = str(obj.get("title", "")).strip()
            description = str(obj.get("description", "")).strip()
        except Exception as e:
            logger.error(f"[Saga] create-summary failed / 开线摘要失败: {e}")

        title = title or f"故事线·{episode.get('metadata', {}).get('name', '未命名')}"
        description = description or ep_summary[:60] or "（待补述）"
        ep_meta = episode.get("metadata", {})
        domain = ep_meta.get("domain", ["未分类"])
        if isinstance(domain, str):
            domain = [domain]

        try:
            saga_id = await self.bucket_mgr.create(
                content=description,
                name=title,
                domain=domain,
                importance=int(ep_meta.get("importance", 6)),
                valence=float(ep_meta.get("valence", 0.5)),
                arousal=float(ep_meta.get("arousal", 0.3)),
                bucket_type="saga",
                tags=["saga"],
            )
            await self.bucket_mgr.update(saga_id, episode_buckets=[episode["id"]])
            logger.info(f"[Saga] created {saga_id} ({title}) / 开新故事线")
            return await self.bucket_mgr.get(saga_id)
        except Exception as e:
            logger.error(f"[Saga] create failed / 建 saga 失败: {e}")
            return None

    # ---------------------------------------------------------
    # Append an episode id to an existing saga's evidence chain.
    # 把一个 episode id 追加进某条 saga 的证据链。
    # ---------------------------------------------------------
    async def _append_episode(self, saga: dict, episode_id: str) -> bool:
        existing = list(saga.get("metadata", {}).get("episode_buckets") or [])
        if episode_id in existing:
            return False
        existing.append(episode_id)
        try:
            ok = await self.bucket_mgr.update(saga["id"], episode_buckets=existing)
            if ok:
                saga.setdefault("metadata", {})["episode_buckets"] = existing
                logger.info(
                    f"[Saga] extended {saga['id']} += {episode_id} / 故事线延长"
                )
            return bool(ok)
        except Exception as e:
            logger.error(f"[Saga] append failed / 延长失败: {e}")
            return False
