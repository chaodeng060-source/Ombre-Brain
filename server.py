# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模块：MCP 服务器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 启动 Ombre Brain MCP 服务，注册记忆操作工具供 Claude 调用。
#
# Core responsibilities:
# 核心职责：
#   - Initialize config, bucket manager, dehydrator, decay engine
#     初始化配置、记忆桶管理器、脱水器、衰减引擎
#   - Expose 5 MCP tools:
#     暴露 5 个 MCP 工具：
#       breath — Surface unresolved memories or search by keyword
#                浮现未解决记忆 或 按关键词检索
#       hold   — Store a single memory
#                存储单条记忆
#       grow   — Diary digest, auto-split into multiple buckets
#                日记归档，自动拆分多桶
#       trace  — Modify metadata / resolved / delete
#                修改元数据 / resolved 标记 / 删除
#       pulse  — System status + bucket listing
#                系统状态 + 所有桶列表
#
# Startup:
# 启动方式：
#   Local:  python server.py
#   Remote: OMBRE_TRANSPORT=streamable-http python server.py
#   Docker: docker-compose up
# ============================================================

import os
import sys
import json
import random
import logging
import asyncio
import httpx
import jieba
from uuid import uuid4
from datetime import datetime

# --- jieba 预热：避免首次 search 卡顿 / Pre-load jieba dict to avoid first-call lag ---
jieba.initialize()

# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from embedding_engine import EmbeddingEngine
from import_memory import ImportEngine
from r2_storage import r2_storage
from utils import (
    load_config, setup_logging, strip_wikilinks, count_tokens_approx,
    world_matches, save_current_world, UNIVERSAL_WORLD,
)

# --- Load config & init logging / 加载配置 & 初始化日志 ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain")

# --- Initialize core components / 初始化核心组件 ---
bucket_mgr = BucketManager(config)                  # Bucket manager / 记忆桶管理器
dehydrator = Dehydrator(config)                      # Dehydrator / 脱水器
decay_engine = DecayEngine(config, bucket_mgr)       # Decay engine / 衰减引擎
embedding_engine = EmbeddingEngine(config)            # Embedding engine / 向量化引擎
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 导入引擎

# --- Create MCP server instance / 创建 MCP 服务器实例 ---
# host="0.0.0.0" so Docker container's SSE is externally reachable
# stdio mode ignores host (no network)
mcp = FastMCP(
    "Ombre Brain",
    host="0.0.0.0",
    port=8000,
)


# =============================================================
# /health endpoint: lightweight keepalive
# 轻量保活接口
# For Cloudflare Tunnel or reverse proxy to ping, preventing idle timeout
# 供 Cloudflare Tunnel 或反代定期 ping，防止空闲超时断连
# =============================================================
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "status": "ok",
            "buckets": stats["permanent_count"] + stats["dynamic_count"],
            "decay_engine": "running" if decay_engine.is_running else "stopped",
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# =============================================================
# /breath-hook endpoint: Dedicated hook for SessionStart
# 会话启动专用挂载点
# =============================================================
@mcp.custom_route("/breath-hook", methods=["GET"])
async def breath_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # pinned
        pinned = [b for b in all_buckets if b["metadata"].get("pinned") or b["metadata"].get("protected")]
        # top 2 unresolved by score
        unresolved = [b for b in all_buckets
                      if not b["metadata"].get("resolved", False)
                      and b["metadata"].get("type") not in ("permanent", "feel")
                      and not b["metadata"].get("pinned")
                      and not b["metadata"].get("protected")]
        scored = sorted(unresolved, key=lambda b: decay_engine.calculate_score(b["metadata"]), reverse=True)

        parts = []
        token_budget = 10000
        for b in pinned:
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            parts.append(f"📌 [核心准则] {summary}")
            token_budget -= count_tokens_approx(summary)

        # Diversity: top-1 fixed + shuffle rest from top-20
        candidates = list(scored)
        if len(candidates) > 1:
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: max 20 surfacing buckets in hook
        candidates = candidates[:20]

        for b in candidates:
            if token_budget <= 0:
                break
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens > token_budget:
                break
            parts.append(summary)
            token_budget -= summary_tokens

        if not parts:
            return PlainTextResponse("")
        return PlainTextResponse("[Ombre Brain - 记忆浮现]\n" + "\n---\n".join(parts))
    except Exception as e:
        logger.warning(f"Breath hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# /dream-hook endpoint: Dedicated hook for Dreaming
# Dreaming 专用挂载点
# =============================================================
@mcp.custom_route("/dream-hook", methods=["GET"])
async def dream_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        candidates = [
            b for b in all_buckets
            if b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]
        candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        recent = candidates[:10]

        if not recent:
            return PlainTextResponse("")

        parts = []
        for b in recent:
            meta = b["metadata"]
            resolved_tag = "[已解决]" if meta.get("resolved", False) else "[未解决]"
            parts.append(
                f"{meta.get('name', b['id'])} {resolved_tag} "
                f"V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}\n"
                f"{strip_wikilinks(b['content'][:200])}"
            )

        return PlainTextResponse("[Ombre Brain - Dreaming]\n" + "\n---\n".join(parts))
    except Exception as e:
        logger.warning(f"Dream hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# Internal helper: resolve world filter for breath
# 内部辅助：根据 world 显式参数 + 全局 current_world 决定过滤集合
# 返回 None 表示不过滤（"all" 模式），否则返回 filter list。
# =============================================================
def _resolve_world_filter(world_param: str, current_world: str):
    wp = (world_param or "").strip()
    if wp.lower() == "all":
        return None
    if wp:
        return [x.strip() for x in wp.split(",") if x.strip()]
    return [(current_world or "").strip()]


# =============================================================
# Internal helper: merge-or-create
# 内部辅助：检查是否可合并，可以则合并，否则新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重复逻辑
# =============================================================
async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    world: str = "",
) -> tuple[str, str, bool]:
    """
    Check if a similar bucket exists for merging; merge if so, create if not.
    Returns (bucket_id, display_name, is_merged).
    检查是否有相似桶可合并，有则合并，无则新建。
    返回 (桶ID, 显示名, 是否合并)。
    """
    # 合并候选必须在同一个 world 内（避免日常桶被角色记忆合并污染或反过来）。
    # world="" 即日常桶，只在日常桶之间合并；通用桶单独按通用合并。
    world_filter = [(world or "").strip()]
    try:
        existing = await bucket_mgr.search(
            content, limit=5,
            domain_filter=domain or None,
            world_filter=world_filter,
        )
        # 主 domain 严格匹配护栏：新内容的 primary domain 必须出现在候选桶的 domain 列表里
        if domain and existing:
            primary = domain[0]
            existing = [b for b in existing if primary in b["metadata"].get("domain", [])]
    except Exception as e:
        logger.warning(f"Search for merge failed, creating new / 合并搜索失败，新建: {e}")
        existing = []

    if existing and existing[0].get("score", 0) > config.get("merge_threshold", 85):
        bucket = existing[0]
        # --- Never merge into pinned/protected/permanent buckets ---
        # --- 不合并到钉选/保护/固化桶（这些桶分数恒定 999，标签网常常很宽，
        # ---  允许吸入会让它们变成"吸尘器"把所有相关 hold 都揽进去）---
        bmeta = bucket["metadata"]
        if not (bmeta.get("pinned") or bmeta.get("protected") or bmeta.get("type") == "permanent"):
            try:
                merged = await dehydrator.merge(bucket["content"], content)
                old_v = bucket["metadata"].get("valence", 0.5)
                old_a = bucket["metadata"].get("arousal", 0.3)
                merged_valence = round((old_v + valence) / 2, 2)
                merged_arousal = round((old_a + arousal) / 2, 2)
                await bucket_mgr.update(
                    bucket["id"],
                    content=merged,
                    tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                    importance=max(bucket["metadata"].get("importance", 5), importance),
                    domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                    valence=merged_valence,
                    arousal=merged_arousal,
                )
                # --- Update embedding after merge ---
                try:
                    await embedding_engine.generate_and_store(bucket["id"], merged)
                except Exception:
                    pass
                return bucket["id"], bucket["metadata"].get("name", bucket["id"]), True
            except Exception as e:
                logger.warning(f"Merge failed, creating new / 合并失败，新建: {e}")

    bucket_id = await bucket_mgr.create(
        content=content,
        tags=tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=name or None,
        world=world,
    )
    # --- Generate embedding for new bucket ---
    try:
        await embedding_engine.generate_and_store(bucket_id, content)
    except Exception:
        pass
    display = name if name else bucket_id
    return bucket_id, display, False


# =============================================================
# Background backfill: hydrate relations for legacy buckets without edges
# 后台 backfill：给老桶补建关系网
# Lazy-started on first hold/breath call. Idempotent — only touches buckets
# whose `relations` field is empty/missing, so it can run safely on every
# server restart without redoing work.
# =============================================================
_backfill_started = False


async def _startup_backfill_loop() -> None:
    """Walk eligible buckets without relations and run _auto_infer_edges on each.
    Rate-limited to ~1 bucket per 2s so the LLM API isn't hammered."""
    global _backfill_started
    try:
        await asyncio.sleep(30)  # let the server fully come up first
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.warning(f"Backfill list_all failed / 列桶失败: {e}")
            return

        candidates = [
            b for b in all_buckets
            if not b["metadata"].get("pinned")
            and not b["metadata"].get("protected")
            and b["metadata"].get("type") not in ("feel", "permanent")
            and not b["metadata"].get("resolved", False)
            and not b["metadata"].get("relations")
        ]
        candidates.sort(key=lambda b: b["id"])

        if not candidates:
            logger.info("Backfill: no eligible buckets / 没有需 backfill 的桶")
            return

        logger.info(
            f"Backfill starting: {len(candidates)} eligible buckets / "
            f"开始 backfill {len(candidates)} 个桶"
        )

        for i, b in enumerate(candidates):
            try:
                n = await _auto_infer_edges(
                    source_id=b["id"],
                    content=b["content"],
                    world=b["metadata"].get("world", ""),
                )
                if i % 5 == 0:
                    logger.info(
                        f"Backfill {i + 1}/{len(candidates)} | "
                        f"{b['id'][:6]} +{n}边"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Backfill bucket {b['id']} failed: {e}")
            await asyncio.sleep(2)  # rate-limit LLM calls

        logger.info(
            f"Backfill complete: {len(candidates)} buckets / "
            f"backfill 完成 {len(candidates)} 桶"
        )
    finally:
        # Don't reset flag on success — we don't want repeat passes within same
        # process. A server restart will re-trigger via _maybe_start_backfill.
        pass


def _maybe_start_backfill() -> None:
    """Lazy start of the backfill loop on first MCP tool call."""
    global _backfill_started
    if _backfill_started:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no event loop yet, will retry on next call
    _backfill_started = True
    loop.create_task(_startup_backfill_loop())
    logger.info("Backfill task scheduled (T-30s) / backfill 已排程")


# =============================================================
# Helper: auto-edge inference for newly created bucket
# 工具：新桶自动建边
# Wraps embedding-similar + keyword-search candidate gathering, LLM relation
# inference via dehydrator, and add_relation calls. All failures swallowed —
# never blocks hold.
# 包装 embedding 邻居 + 关键词搜索拿候选，dehydrator LLM 判边，bucket_mgr 加边。
# 所有失败吞掉——绝不阻塞 hold 主流程。
# =============================================================
async def _auto_infer_edges(
    source_id: str, content: str, world: str = ""
) -> list[dict]:
    """Returns list of edge dicts actually added: [{type, target, target_name, note}]."""
    if not content or not content.strip():
        return []

    # --- Gather candidates: vector neighbors + keyword search, dedup, exclude self ---
    candidate_ids: list[str] = []
    seen = {source_id}

    try:
        vec_hits = await embedding_engine.search_similar(content, top_k=8)
        for bid, _score in vec_hits:
            if bid not in seen:
                candidate_ids.append(bid)
                seen.add(bid)
    except Exception as e:
        logger.warning(f"Vector candidate fetch failed / 向量候选失败: {e}")

    try:
        kw_hits = await bucket_mgr.search(content, limit=5)
        for b in kw_hits:
            bid = b.get("id")
            if bid and bid not in seen:
                candidate_ids.append(bid)
                seen.add(bid)
    except Exception as e:
        logger.warning(f"Keyword candidate fetch failed / 关键词候选失败: {e}")

    if not candidate_ids:
        return []

    # --- Build candidate list with dehydrated summaries (cap 8 to bound LLM cost) ---
    candidates: list[dict] = []
    wf_set = {(world or "").strip()}
    for bid in candidate_ids[:8]:
        b = await bucket_mgr.get(bid)
        if not b:
            continue
        # Skip cross-world candidates: don't link daily↔roleplay buckets
        # 跨世界候选跳过：避免日常↔角色扮演桶被自动连边
        b_world = b["metadata"].get("world", "")
        if not world_matches(b_world, wf_set):
            continue
        try:
            summary = await dehydrator.dehydrate(
                strip_wikilinks(b["content"]),
                {k: v for k, v in b["metadata"].items() if k != "tags"},
            )
        except Exception:
            summary = (b["content"] or "")[:200]
        candidates.append({
            "id": bid,
            "name": b["metadata"].get("name", bid),
            "summary": summary,
        })

    if not candidates:
        return []

    edges = await dehydrator.infer_relations(content, candidates)
    if not edges:
        return []

    cand_name_by_id = {c["id"]: c["name"] for c in candidates}
    added: list[dict] = []
    for edge in edges:
        ok = await bucket_mgr.add_relation(
            source_id, edge["target"], edge["type"], edge.get("note", "")
        )
        if ok:
            added.append({
                "type": edge["type"],
                "target": edge["target"],
                "target_name": cand_name_by_id.get(edge["target"], edge["target"]),
                "note": edge.get("note", ""),
            })
    if added:
        logger.info(
            f"Auto-edge inference added {len(added)} edge(s) / 自动建边 {len(added)} 条 "
            f"from {source_id}"
        )
    return added


# =============================================================
# Tool 1: breath — Breathe
# 工具 1：breath — 呼吸
#
# No args: surface highest-weight unresolved memories (active push)
# 无参数：浮现权重最高的未解决记忆
# With args: search by keyword + emotion coordinates
# 有参数：按关键词+情感坐标检索记忆
# =============================================================
@mcp.tool()
async def breath(
    query: str = "",
    max_tokens: int = 10000,
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    max_results: int = 20,
    world: str = "",
    relation_depth: int = 1,
) -> str:
    """检索/浮现记忆。不传query或传空=自动浮现,有query=关键词检索。max_tokens控制返回总token上限(默认10000)。domain逗号分隔,valence/arousal 0~1(-1忽略)。max_results控制返回数量上限(默认20,最大50)。world=过滤世界:留空走全局current_world(日常时只出日常+通用、角色扮演时只出该世界+通用),"all"跳过过滤,"旧世界"/"当前世界"等显式指定。world="通用"的桶永远跟着出。relation_depth=沿关系边召回邻居的跳数(默认1,0=不走关系边),目前 MVP 只走 1 跳出边,最多附加 5 条。"""
    await decay_engine.ensure_started()
    _maybe_start_backfill()
    max_results = min(max_results, 50)
    max_tokens = min(max_tokens, 20000)

    # --- Resolve world filter once (used by all modes) ---
    # --- 解析 world filter：显式参数 > current_world ---
    world_filter = _resolve_world_filter(world, config.get("current_world", ""))
    wf_set = {str(w).strip() for w in world_filter} if world_filter is not None else None

    # --- No args or empty query: surfacing mode (weight pool active push) ---
    # --- 无参数或空query：浮现模式（权重池主动推送）---
    if not query or not query.strip():
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for surfacing / 浮现列桶失败: {e}")
            return "记忆系统暂时无法访问。"

        # --- Pinned/protected buckets: always surface as core principles ---
        # --- 钉选桶：作为核心准则，始终浮现（不受 world 过滤影响）---
        pinned_buckets = [
            b for b in all_buckets
            if b["metadata"].get("pinned") or b["metadata"].get("protected")
        ]
        pinned_results = []
        for b in pinned_buckets:
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                pinned_results.append(f"📌 [核心准则] [bucket_id:{b['id']}] {summary}")
            except Exception as e:
                logger.warning(f"Failed to dehydrate pinned bucket / 钉选桶脱水失败: {e}")
                continue

        # --- Unresolved buckets: surface top N by weight ---
        # --- 未解决桶：按权重浮现前 N 条 ---
        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved", False)
            and b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]

        # --- World filter on surfacing pool ---
        # --- 浮现池按 world 过滤：日常/角色扮演时不串场 ---
        if wf_set is not None:
            unresolved = [
                b for b in unresolved
                if world_matches(b["metadata"].get("world", ""), wf_set)
            ]

        logger.info(
            f"Breath surfacing: {len(all_buckets)} total, "
            f"{len(pinned_buckets)} pinned, {len(unresolved)} unresolved "
            f"(world_filter={wf_set if wf_set is not None else 'all'})"
        )

        scored = sorted(
            unresolved,
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )

        if scored:
            top_scores = [(b["metadata"].get("name", b["id"]), decay_engine.calculate_score(b["metadata"])) for b in scored[:5]]
            logger.info(f"Top unresolved scores: {top_scores}")

        # --- Token-budgeted surfacing with diversity + hard cap ---
        # --- 按 token 预算浮现，带多样性 + 硬上限 ---
        # Top-1 always surfaces; rest sampled from top-20 for diversity
        token_budget = max_tokens
        for r in pinned_results:
            token_budget -= count_tokens_approx(r)

        candidates = list(scored)
        if len(candidates) > 1:
            # Ensure highest-score bucket is first, shuffle rest from top-20
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: never surface more than max_results buckets
        candidates = candidates[:max_results]

        dynamic_results = []
        for b in candidates:
            if token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                summary_tokens = count_tokens_approx(summary)
                if summary_tokens > token_budget:
                    break
                # NOTE: no touch() here — surfacing should NOT reset decay timer
                score = decay_engine.calculate_score(b["metadata"])
                dynamic_results.append(f"[权重:{score:.2f}] [bucket_id:{b['id']}] {summary}")
                token_budget -= summary_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮现脱水失败: {e}")
                continue

        if not pinned_results and not dynamic_results:
            return "权重池平静，没有需要处理的记忆。"

        parts = []
        if pinned_results:
            parts.append("=== 核心准则 ===\n" + "\n---\n".join(pinned_results))
        if dynamic_results:
            parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
        return "\n\n".join(parts)

    # --- Feel retrieval: domain="feel" is a special channel ---
    # --- Feel 检索：domain="feel" 是独立入口 ---
    if domain.strip().lower() == "feel":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not feels:
                return "没有留下过 feel。"
            results = []
            for f in feels:
                created = f["metadata"].get("created", "")
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                results.append(entry)
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            return "=== 你留下的 feel ===\n" + "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            return "读取 feel 失败。"

    # --- With args: search mode (keyword + vector dual channel) ---
    # --- 有参数：检索模式（关键词 + 向量双通道）---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None

    try:
        matches = await bucket_mgr.search(
            query,
            limit=max(max_results, 20),
            domain_filter=domain_filter,
            world_filter=world_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    # --- Vector similarity channel: find semantically related buckets ---
    # --- 向量相似度通道：找到语义相关的桶 ---
    matched_ids = {b["id"] for b in matches}
    try:
        vector_results = await embedding_engine.search_similar(query, top_k=max(max_results, 20))
        vector_added = 0  # 限流计数器：向量通道最多补 3 条，防止污染精准检索
        for bucket_id, sim_score in vector_results:
            if vector_added >= 3:
                break
            if bucket_id not in matched_ids and sim_score > 0.65:  # 阈值 0.5 → 0.65，提高语义相关性门槛
                bucket = await bucket_mgr.get(bucket_id)
                if not bucket:
                    continue
                if bucket["metadata"].get("pinned") or bucket["metadata"].get("protected"):
                    continue
                # 向量通道也走 world filter，避免跨世界语义召回污染
                if wf_set is not None and not world_matches(
                    bucket["metadata"].get("world", ""), wf_set
                ):
                    continue
                bucket["score"] = round(sim_score * 100, 2)
                bucket["vector_match"] = True
                matches.append(bucket)
                matched_ids.add(bucket_id)
                vector_added += 1
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")

    results = []
    token_used = 0
    for bucket in matches:
        if token_used >= max_tokens:
            break
        try:
            clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
            # --- Memory reconstruction: shift displayed valence by current mood ---
            # --- 记忆重构：根据当前情绪微调展示层 valence（±0.1）---
            if q_valence is not None and "valence" in clean_meta:
                original_v = float(clean_meta.get("valence", 0.5))
                shift = (q_valence - 0.5) * 0.2  # ±0.1 max shift
                clean_meta["valence"] = max(0.0, min(1.0, original_v + shift))
            summary = await dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
            summary_tokens = count_tokens_approx(summary)
            if token_used + summary_tokens > max_tokens:
                break
            await bucket_mgr.touch(bucket["id"])
            if bucket.get("vector_match"):
                summary = f"[语义关联] [bucket_id:{bucket['id']}] {summary}"
            else:
                summary = f"[bucket_id:{bucket['id']}] {summary}"
            results.append(summary)
            token_used += summary_tokens
        except Exception as e:
            logger.warning(f"Failed to dehydrate search result / 检索结果脱水失败: {e}")
            continue

    # --- Relation expansion: 1-hop out-edges of matched buckets ---
    # --- 关系网召回：沿主结果桶的出边带 1 跳邻居（不进主排序，单独列在末尾）---
    if relation_depth >= 1 and matches and token_used < max_tokens:
        seen_neighbors = set()
        neighbor_msgs = []
        for bucket in matches:
            if len(neighbor_msgs) >= 5 or token_used >= max_tokens:
                break
            relations = bucket["metadata"].get("relations") or []
            if not isinstance(relations, list):
                continue
            for r in relations:
                if len(neighbor_msgs) >= 5 or token_used >= max_tokens:
                    break
                if not isinstance(r, dict):
                    continue
                t_id = r.get("target")
                if not t_id or t_id in matched_ids or t_id in seen_neighbors:
                    continue
                try:
                    neighbor = await bucket_mgr.get(t_id)
                except Exception:
                    continue
                if not neighbor:
                    continue
                if wf_set is not None and not world_matches(
                    neighbor["metadata"].get("world", ""), wf_set
                ):
                    continue
                try:
                    clean_meta = {k: v for k, v in neighbor["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(
                        strip_wikilinks(neighbor["content"]), clean_meta
                    )
                    summary_tokens = count_tokens_approx(summary)
                    if token_used + summary_tokens > max_tokens:
                        break
                    rel_type = r.get("type", "?")
                    neighbor_msgs.append(
                        f"[关系:{rel_type}←{bucket['id']}] [bucket_id:{t_id}] {summary}"
                    )
                    token_used += summary_tokens
                    seen_neighbors.add(t_id)
                except Exception as e:
                    logger.warning(f"Failed to dehydrate neighbor / 邻居脱水失败: {e}")
                    continue
        if neighbor_msgs:
            results.append("--- 关系网邻居 ---\n" + "\n---\n".join(neighbor_msgs))

    # --- Random surfacing: when search returns < 3, 40% chance to float old memories ---
    # --- 随机浮现：检索结果不足 3 条时，40% 概率从低权重旧桶里漂上来 ---
    if len(matches) < 3 and random.random() < 0.4:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in matches}
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and decay_engine.calculate_score(b["metadata"]) < 2.0
                and (wf_set is None or world_matches(b["metadata"].get("world", ""), wf_set))
            ]
            if low_weight:
                drifted = random.sample(low_weight, min(random.randint(1, 3), len(low_weight)))
                drift_results = []
                for b in drifted:
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                    drift_results.append(f"[surface_type: random]\n{summary}")
                results.append("--- 忽然想起来 ---\n" + "\n---\n".join(drift_results))
        except Exception as e:
            logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        return "未找到相关记忆。"

    return "\n---\n".join(results)


# =============================================================
# Tool 2: hold — Hold on to this
# 工具 2：hold — 握住，留下来
# =============================================================
@mcp.tool()
async def hold(
    content: str,
    tags: str = "",
    importance: int = 5,
    pinned: bool = False,
    feel: bool = False,
    source_bucket: str = "",
    valence: float = -1,
    arousal: float = -1,
    image_base64: str = "",
    image_filename: str = "image",
    world: str = "",
) -> str:
    """存储单条记忆,自动打标+合并。tags逗号分隔,importance 1-10。pinned=True创建永久钉选桶。feel=True存储你的第一人称感受(不参与普通浮现)。source_bucket=被消化的记忆桶ID(feel模式下,标记源记忆为已消化)。image_base64=可选,base64编码的图片数据,会上传到R2并把URL插入正文(允许此条记忆带图)。image_filename=图片名称提示(默认image)。world=显式指定世界归属,留空时走全局current_world(日常聊天=空,角色扮演=具体世界名),"通用"表示跨世界设定。feel桶不归属世界。"""
    await decay_engine.ensure_started()
    _maybe_start_backfill()

    # --- Input validation / 输入校验 ---
    if not content or not content.strip():
        return "内容为空，无法存储。"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # --- Resolve effective world / 解析当前桶的 world 归属 ---
    # 显式传 world > 全局 current_world。feel 桶在下面单独处理（feel 跨世界）。
    effective_world = (world or "").strip() or (config.get("current_world", "") or "").strip()

    # --- Optional image upload to R2 / 可选：上传图片到 R2 ---
    # If image_base64 provided and R2 configured, upload and prepend URL
    # markdown to content so the image is rendered in Obsidian and
    # surfaced when this bucket is read later.
    # 若提供了 image_base64 且 R2 已配置，上传并在正文前部插入图片 URL，
    # 这样 Obsidian 能直接渲染，桶被读取时图片 URL 也会跟着 content 出来。
    image_url: str = ""
    if image_base64 and image_base64.strip():
        try:
            image_url = r2_storage.upload_base64(image_base64, image_filename) or ""
        except Exception as e:
            logger.warning(f"R2 image upload raised / R2 上传抛错: {e}")
            image_url = ""
        if image_url:
            # Prepend image markdown so dehydrator/Obsidian both see it
            # 在正文前插入图片 markdown，dehydrator 和 Obsidian 都能识别
            content = f"![{image_filename}]({image_url})\n\n{content}"
            logger.info(f"Hold attached image / 附加图片: {image_url}")
        else:
            logger.warning(
                "Image was provided but R2 upload returned no URL "
                "(R2 disabled or upload failed) / "
                "提供了图片但 R2 上传未返回 URL（R2 未启用或上传失败）"
            )

    # --- Feel mode: store as feel type, minimal metadata ---
    # --- Feel 模式：存为 feel 类型，最少元数据 ---
    if feel:
        # Feel valence/arousal = model's own perspective
        feel_valence = valence if 0 <= valence <= 1 else 0.5
        feel_arousal = arousal if 0 <= arousal <= 1 else 0.3
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=[],
            importance=5,
            domain=[],
            valence=feel_valence,
            arousal=feel_arousal,
            name=None,
            bucket_type="feel",
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        # --- Mark source memory as digested + store model's valence perspective ---
        # --- 标记源记忆为已消化 + 存储模型视角的 valence ---
        if source_bucket and source_bucket.strip():
            try:
                update_kwargs = {"digested": True}
                if 0 <= valence <= 1:
                    update_kwargs["model_valence"] = feel_valence
                await bucket_mgr.update(source_bucket.strip(), **update_kwargs)
            except Exception as e:
                logger.warning(f"Failed to mark source as digested / 标记已消化失败: {e}")
        return f"🫧feel→{bucket_id}"

    # --- Step 1: auto-tagging / 自动打标 ---
    try:
        analysis = await dehydrator.analyze(content)
    except Exception as e:
        logger.warning(f"Auto-tagging failed, using defaults / 自动打标失败: {e}")
        analysis = {
            "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
            "tags": [], "suggested_name": "",
        }

    domain = analysis["domain"]
    valence = analysis["valence"]
    arousal = analysis["arousal"]
    auto_tags = analysis["tags"]
    suggested_name = analysis.get("suggested_name", "")

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    # --- Pinned buckets bypass merge and are created directly in permanent dir ---
    # --- 钉选桶跳过合并，直接新建到 permanent 目录 ---
    if pinned:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=all_tags,
            importance=10,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=suggested_name or None,
            bucket_type="permanent",
            pinned=True,
            world=effective_world,
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        return f"📌钉选→{bucket_id} {','.join(domain)}"

    # --- Step 2: merge or create / 合并或新建 ---
    bucket_id, result_name, is_merged = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=suggested_name,
        world=effective_world,
    )

    # --- Step 3: auto-edge inference (only on new buckets, never on merges) ---
    # --- 自动建边：仅对新建桶，合并桶已经融进了相关性，不再加边避免冗余 ---
    added_edges: list[dict] = []
    if not is_merged:
        try:
            added_edges = await _auto_infer_edges(
                source_id=bucket_id, content=content, world=effective_world
            )
        except Exception as e:
            logger.warning(f"Auto-edge inference failed / 自动建边失败: {e}")

    action = "合并→" if is_merged else "新建→"
    base = f"{action}{result_name} {','.join(domain)}"
    if not added_edges:
        return base
    # 写=读：hold 同时返回相关桶（让用户感知这条记忆和什么连着）
    related_lines = [
        f"  • [{e['type']}] {e['target_name']} ({e['target']})"
        + (f" — {e['note']}" if e.get("note") else "")
        for e in added_edges
    ]
    return f"{base} +{len(added_edges)}边\n关联：\n" + "\n".join(related_lines)


# =============================================================
# Tool 3: grow — Grow, fragments become memories
# 工具 3：grow — 生长，一天的碎片长成记忆
# =============================================================
@mcp.tool()
async def grow(content: str, world: str = "") -> str:
    """日记归档,自动拆分为多桶。短内容(<30字)走快速路径。world留空走全局current_world。"""
    await decay_engine.ensure_started()

    if not content or not content.strip():
        return "内容为空，无法整理。"

    # --- Resolve effective world / 解析当前批次的 world 归属 ---
    effective_world = (world or "").strip() or (config.get("current_world", "") or "").strip()

    # --- Short content fast path: skip digest, use hold logic directly ---
    # --- 短内容快速路径：跳过 digest 拆分，直接走 hold 逻辑省一次 API ---
    # For very short inputs (like "1"), calling digest is wasteful:
    # it sends the full DIGEST_PROMPT (~800 tokens) to DeepSeek for nothing.
    # Instead, run analyze + create directly.
    if len(content.strip()) < 30:
        logger.info(f"grow short-content fast path: {len(content.strip())} chars")
        try:
            analysis = await dehydrator.analyze(content)
        except Exception as e:
            logger.warning(f"Fast-path analyze failed / 快速路径打标失败: {e}")
            analysis = {
                "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
                "tags": [], "suggested_name": "",
            }
        _bid, result_name, is_merged = await _merge_or_create(
            content=content.strip(),
            tags=analysis.get("tags", []),
            importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
            domain=analysis.get("domain", ["未分类"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
            world=effective_world,
        )
        action = "合并" if is_merged else "新建"
        return f"{action} → {result_name} | {','.join(analysis.get('domain', []))} V{analysis.get('valence', 0.5):.1f}/A{analysis.get('arousal', 0.3):.1f}"

    # --- Step 1: let API split and organize / 让 API 拆分整理 ---
    try:
        items = await dehydrator.digest(content)
    except Exception as e:
        logger.error(f"Diary digest failed / 日记整理失败: {e}")
        return f"日记整理失败: {e}"

    if not items:
        return "内容为空或整理失败。"

    results = []
    created = 0
    merged = 0

    # --- Step 2: merge or create each item (with per-item error handling) ---
    # --- 逐条合并或新建（单条失败不影响其他）---
    for item in items:
        try:
            _bid, result_name, is_merged = await _merge_or_create(
                content=item["content"],
                tags=item.get("tags", []),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分类"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name", ""),
                world=effective_world,
            )

            if is_merged:
                results.append(f"📎{result_name}")
                merged += 1
            else:
                results.append(f"📝{item.get('name', result_name)}")
                created += 1
        except Exception as e:
            logger.warning(
                f"Failed to process diary item / 日记条目处理失败: "
                f"{item.get('name', '?')}: {e}"
            )
            results.append(f"⚠️{item.get('name', '?')}")

    return f"{len(items)}条|新{created}合{merged}\n" + "\n".join(results)


# =============================================================
# Tool 4: trace — Trace, redraw the outline of a memory
# 工具 4：trace — 描摹，重新勾勒记忆的轮廓
# Also handles deletion (delete=True)
# 同时承接删除功能
# =============================================================
@mcp.tool()
async def trace(
    bucket_id: str,
    name: str = "",
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    importance: int = -1,
    tags: str = "",
    resolved: int = -1,
    pinned: int = -1,
    digested: int = -1,
    content: str = "",
    world: str = "",
    delete: bool = False,
    add_relation: str = "",
    remove_relation: str = "",
) -> str:
    """修改记忆元数据或内容。resolved=1沉底/0激活,pinned=1钉选/0取消,digested=1隐藏(保留但不浮现)/0取消隐藏,content=替换桶正文,delete=True删除。world=改世界归属(传"(none)"清空回日常),只传需改的,-1或空=不改。add_relation格式"type:target_id"或"type:target_id:note",6类:causes/contributes/improves/explains/updates/kin。remove_relation格式"target_id"或"type:target_id"。"""

    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    # --- Delete mode / 删除模式 ---
    if delete:
        success = await bucket_mgr.delete(bucket_id)
        if success:
            embedding_engine.delete_embedding(bucket_id)
        return f"已遗忘记忆桶: {bucket_id}" if success else f"未找到记忆桶: {bucket_id}"

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"

    # --- Collect only fields actually passed / 只收集用户实际传入的字段 ---
    updates = {}
    if name:
        updates["name"] = name
    if domain:
        updates["domain"] = [d.strip() for d in domain.split(",") if d.strip()]
    if 0 <= valence <= 1:
        updates["valence"] = valence
    if 0 <= arousal <= 1:
        updates["arousal"] = arousal
    if 1 <= importance <= 10:
        updates["importance"] = importance
    if tags:
        updates["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if resolved in (0, 1):
        updates["resolved"] = bool(resolved)
    if pinned in (0, 1):
        updates["pinned"] = bool(pinned)
        if pinned == 1:
            updates["importance"] = 10  # pinned → lock importance
    if digested in (0, 1):
        updates["digested"] = bool(digested)
    if world:
        w = world.strip()
        # sentinel "(none)" → 清空 world 字段（挪回日常）
        updates["world"] = "" if w == "(none)" else w
    if content:
        updates["content"] = content

    # --- Relation edits / 关系边操作（独立于 metadata update，避免和 update 字段竞争）---
    relation_msgs = []
    if add_relation and add_relation.strip():
        parts = [p.strip() for p in add_relation.split(":", 2)]
        if len(parts) < 2:
            return "add_relation 格式错误，需 'type:target_id' 或 'type:target_id:note'。"
        rel_type, target_id = parts[0], parts[1]
        note = parts[2] if len(parts) >= 3 else ""
        ok = await bucket_mgr.add_relation(bucket_id, target_id, rel_type, note)
        relation_msgs.append(f"+边 [{rel_type}→{target_id}]" if ok else f"加边失败 [{rel_type}→{target_id}]")
    if remove_relation and remove_relation.strip():
        parts = [p.strip() for p in remove_relation.split(":", 1)]
        if len(parts) == 1:
            rel_type, target_id = "", parts[0]
        else:
            rel_type, target_id = parts[0], parts[1]
        n = await bucket_mgr.remove_relation(bucket_id, target_id, rel_type)
        relation_msgs.append(f"-边 ×{n} [{target_id}]" if n else f"删边未命中 [{target_id}]")

    if not updates and not relation_msgs:
        return "没有任何字段需要修改。"

    if updates:
        success = await bucket_mgr.update(bucket_id, **updates)
        if not success:
            return f"修改失败: {bucket_id}"
    else:
        success = True

    # Re-generate embedding if content changed
    if "content" in updates:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    changed = ", ".join(f"{k}={v}" for k, v in updates.items() if k != "content")
    if "content" in updates:
        changed += (", content=已替换" if changed else "content=已替换")
    # Explicit hint about resolved state change semantics
    # 特别提示 resolved 状态变化的语义
    if "resolved" in updates:
        if updates["resolved"]:
            changed += " → 已沉底，只在关键词触发时重新浮现"
        else:
            changed += " → 已重新激活，将参与浮现排序"
    if "digested" in updates:
        if updates["digested"]:
            changed += " → 已隐藏，保留但不再浮现"
        else:
            changed += " → 已取消隐藏，重新参与浮现"
    if relation_msgs:
        changed = (changed + "; " if changed else "") + "; ".join(relation_msgs)
    return f"已修改记忆桶 {bucket_id}: {changed}"


# =============================================================
# Tool: backfill_relations — run auto-edge inference on existing buckets
# 工具：backfill_relations — 给老桶批量自动建边
# Hold-time auto-edge only fires on new buckets; this tool fills in the
# graph for memories that existed before the feature shipped. Batched to
# avoid MCP timeout and to let the caller resume between calls.
# =============================================================
@mcp.tool()
async def backfill_relations(
    bucket_id: str = "",
    limit: int = 5,
    offset: int = 0,
) -> str:
    """对已有桶批量跑自动建边。
    bucket_id=指定单桶处理（最快验证用）。
    bucket_id 为空时按 limit/offset 批量遍历 dynamic 桶（跳过 pinned/permanent/feel/resolved），每次最多 10 个，多次调用滚动跑完。
    返回每桶加了几条边和下一批 offset。"""
    if bucket_id and bucket_id.strip():
        bucket = await bucket_mgr.get(bucket_id.strip())
        if not bucket:
            return f"未找到桶: {bucket_id}"
        try:
            edges = await _auto_infer_edges(
                source_id=bucket["id"],
                content=bucket["content"],
                world=bucket["metadata"].get("world", ""),
            )
            n = len(edges)
            return f"{bucket['id']}: +{n}边"
        except Exception as e:
            logger.warning(f"backfill single failed {bucket_id}: {e}")
            return f"{bucket_id}: 失败 {e}"

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        return f"列桶失败: {e}"

    # 跳过钉选/保护/permanent/feel/resolved——这些桶要么不需要长边、要么是潜意识素材库
    eligible = [
        b for b in all_buckets
        if not b["metadata"].get("pinned")
        and not b["metadata"].get("protected")
        and b["metadata"].get("type") not in ("feel", "permanent")
        and not b["metadata"].get("resolved", False)
    ]
    eligible.sort(key=lambda b: b["id"])

    limit = max(1, min(int(limit), 10))
    offset = max(0, int(offset))
    batch = eligible[offset:offset + limit]

    if not batch:
        return f"无桶可处理 (eligible={len(eligible)}, offset={offset})"

    results = []
    total = 0
    for b in batch:
        try:
            edges = await _auto_infer_edges(
                source_id=b["id"],
                content=b["content"],
                world=b["metadata"].get("world", ""),
            )
            n = len(edges)
            results.append(f"{b['id'][:6]}+{n}")
            total += n
        except Exception as e:
            logger.warning(f"backfill bucket {b['id']} failed: {e}")
            results.append(f"{b['id'][:6]}!err")

    next_offset = offset + len(batch)
    remaining = len(eligible) - next_offset
    return (
        f"批 {offset}-{next_offset - 1}/{len(eligible)} | "
        f"+{total}边 | {' '.join(results)} | "
        f"剩 {remaining}, next offset={next_offset}"
    )


# =============================================================
# Tool: switch_world — change global current_world pointer at runtime
# 工具：switch_world — 切换全局当前世界指针
# =============================================================
@mcp.tool()
async def switch_world(world: str = "") -> str:
    """切换全局当前世界。空字符串=日常模式(只浮现日常+通用桶),具体世界名=角色扮演模式。
    生效范围：之后所有 hold(不传 world 时)写到该世界,breath(不传 world 时)只浮该世界+通用。
    持久化到 {buckets_dir}/.ombre_runtime.yaml,重启不丢。pulse 可看当前指针。"""
    target = (world or "").strip()
    valid_worlds = config.get("worlds", []) or []
    if target and target not in valid_worlds:
        return (
            f"未知世界: {target!r}。已知 worlds: {valid_worlds}\n"
            f"如要新增世界,先在 config.yaml 的 worlds: 列表里加上,再切换。"
        )
    try:
        save_current_world(config["buckets_dir"], target)
    except OSError as e:
        return f"持久化失败: {e}"
    config["current_world"] = target
    label = target if target else "日常模式 (空)"
    logger.info(f"current_world switched → {label}")
    return f"已切换到 → {label}"


# =============================================================
# Tool 5: pulse — Heartbeat, system status + memory listing
# 工具 5：pulse — 脉搏，系统状态 + 记忆列表
# =============================================================
@mcp.tool()
async def pulse(include_archive: bool = False) -> str:
    """系统状态+记忆桶列表。include_archive=True含归档。"""
    try:
        stats = await bucket_mgr.get_stats()
    except Exception as e:
        return f"获取系统状态失败: {e}"

    cw = (config.get("current_world") or "").strip() or "日常模式 (空)"
    status = (
        f"=== Ombre Brain 记忆系统 ===\n"
        f"当前世界: {cw}\n"
        f"固化记忆桶: {stats['permanent_count']} 个\n"
        f"动态记忆桶: {stats['dynamic_count']} 个\n"
        f"归档记忆桶: {stats['archive_count']} 个\n"
        f"总存储大小: {stats['total_size_kb']:.1f} KB\n"
        f"衰减引擎: {'运行中' if decay_engine.is_running else '已停止'}\n"
    )

    # --- List all bucket summaries / 列出所有桶摘要 ---
    try:
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
    except Exception as e:
        return status + f"\n列出记忆桶失败: {e}"

    if not buckets:
        return status + "\n记忆库为空。"

    lines = []
    for b in buckets:
        meta = b.get("metadata", {})
        if meta.get("pinned") or meta.get("protected"):
            icon = "📌"
        elif meta.get("type") == "permanent":
            icon = "📦"
        elif meta.get("type") == "feel":
            icon = "🫧"
        elif meta.get("type") == "archived":
            icon = "🗄️"
        elif meta.get("resolved", False):
            icon = "✅"
        else:
            icon = "💭"
        try:
            score = decay_engine.calculate_score(meta)
        except Exception:
            score = 0.0
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        resolved_tag = " [已解决]" if meta.get("resolved", False) else ""
        lines.append(
            f"{icon} [{meta.get('name', b['id'])}]{resolved_tag} "
            f"bucket_id:{b['id']} "
            f"主题:{domains} "
            f"情感:V{val:.1f}/A{aro:.1f} "
            f"重要:{meta.get('importance', '?')} "
            f"权重:{score:.2f} "
            f"标签:{','.join(meta.get('tags', []))}"
        )

    return status + "\n=== 记忆列表 ===\n" + "\n".join(lines)


# =============================================================
# Tool 6: dream — Dreaming, digest recent memories
# 工具 6：dream — 做梦，消化最近的记忆
#
# Reads recent surface-level buckets (≤10), returns them for
# Claude to introspect under prompt guidance.
# 读取最近新增的表层桶（≤10个），返回给 Claude 在提示词引导下自主思考。
# Claude then decides: resolve some, write feels, or do nothing.
# =============================================================
@mcp.tool()
async def dream() -> str:
    """做梦——读取最近新增的记忆桶,供你自省。读完后可以trace(resolved=1)放下,或hold(feel=True)写感受。"""
    await decay_engine.ensure_started()

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Dream failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Filter: recent surface-level dynamic buckets (not permanent/pinned/feel) ---
    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]

    # --- Sort by creation time desc, take top 10 ---
    candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    recent = candidates[:10]

    if not recent:
        return "没有需要消化的新记忆。"

    parts = []
    for b in recent:
        meta = b["metadata"]
        resolved_tag = " [已解决]" if meta.get("resolved", False) else " [未解决]"
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        created = meta.get("created", "")
        parts.append(
            f"[{meta.get('name', b['id'])}]{resolved_tag} "
            f"主题:{domains} V{val:.1f}/A{aro:.1f} "
            f"创建:{created}\n"
            f"ID: {b['id']}\n"
            f"{strip_wikilinks(b['content'][:500])}"
        )

    header = (
        "=== Dreaming ===\n"
        "以下是你最近的记忆。用第一人称想：\n"
        "- 这些东西里有什么在你这里留下了重量？\n"
        "- 有什么还没想清楚？\n"
        "- 有什么可以放下了？\n"
        "想完之后：值得放下的用 trace(bucket_id, resolved=1)；\n"
        "有沉淀的用 hold(content=\"...\", feel=True, source_bucket=\"bucket_id\", valence=你的感受) 写下来。\n"
        "valence 是你对这段记忆的感受，不是事件本身的情绪。\n"
        "没有沉淀就不写，不强迫产出。\n"
    )

    # --- Connection hint: find most similar pair via embeddings ---
    connection_hint = ""
    if embedding_engine and embedding_engine.enabled and len(recent) >= 2:
        try:
            best_pair = None
            best_sim = 0.0
            ids = [b["id"] for b in recent]
            names = {b["id"]: b["metadata"].get("name", b["id"]) for b in recent}
            embeddings = {}
            for bid in ids:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb
            for i, id_a in enumerate(ids):
                for id_b in ids[i+1:]:
                    if id_a in embeddings and id_b in embeddings:
                        sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (id_a, id_b)
            if best_pair and best_sim > 0.5:
                connection_hint = (
                    f"\n💭 [{names[best_pair[0]]}] 和 [{names[best_pair[1]]}] "
                    f"似乎有关联 (相似度:{best_sim:.2f})——不替你下结论，你自己想。\n"
                )
        except Exception as e:
            logger.warning(f"Dream connection hint failed: {e}")

    # --- Feel crystallization hint: detect repeated feel themes ---
    crystal_hint = ""
    if embedding_engine and embedding_engine.enabled:
        try:
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if len(feels) >= 3:
                feel_embeddings = {}
                for f in feels:
                    emb = await embedding_engine.get_embedding(f["id"])
                    if emb is not None:
                        feel_embeddings[f["id"]] = emb
                # Find clusters: feels with similarity > 0.7 to at least 2 others
                for fid, femb in feel_embeddings.items():
                    similar_feels = []
                    for oid, oemb in feel_embeddings.items():
                        if oid != fid:
                            sim = embedding_engine._cosine_similarity(femb, oemb)
                            if sim > 0.7:
                                similar_feels.append(oid)
                    if len(similar_feels) >= 2:
                        feel_bucket = next((f for f in feels if f["id"] == fid), None)
                        if feel_bucket and not feel_bucket["metadata"].get("pinned"):
                            content_preview = strip_wikilinks(feel_bucket["content"][:80])
                            crystal_hint = (
                                f"\n🔮 你已经写过 {len(similar_feels)+1} 条相似的 feel "
                                f"（围绕「{content_preview}…」）。"
                                f"如果这已经是确信而不只是感受了，"
                                f"你可以用 hold(content=\"...\", pinned=True) 升级它。"
                                f"不急，你自己决定。\n"
                            )
                            break
        except Exception as e:
            logger.warning(f"Dream crystallization hint failed: {e}")

    return header + "\n---\n".join(parts) + connection_hint + crystal_hint


# =============================================================
# Tool 7: briefing — Open-window handoff briefing
# 工具 7：briefing — 开窗交接简报
#
# Aggregates pinned + top-weighted unresolved + recently-active
# buckets, compresses via LLM into a ≤1500-char briefing.
# 聚合钉选 + 高权重未解决 + 最近活跃桶，LLM 压缩为 ≤1500 字简报。
# Designed to replace the 18000-token full-breath open-window cost
# with a 3000-token briefing (~80% savings).
# 用于替代开窗时 18000 token 的完整 breath 浮现，压到约 3000 token。
# =============================================================
@mcp.tool()
async def briefing(
    max_chars: int = 1000,
    domain: str = "",
    pinned_only: bool = False,
) -> str:
    """开窗简报。聚合钉选+高权重未解决+最近活跃桶,LLM压缩为≤max_chars字简报,默认1000字。输出顺序:朝灯当前氛围/走向→最近因果链故事→活着的欠账→工程线→铁律。domain逗号分隔可过滤主题域。pinned_only=True只用钉选桶。开窗调一次,省80%token。"""
    await decay_engine.ensure_started()
    max_chars = max(300, min(max_chars, 4000))

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Briefing failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Domain filter ---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()]
    if domain_filter:
        def _domain_hit(b):
            doms = b["metadata"].get("domain", []) or []
            return any(d in doms for d in domain_filter)
        all_buckets = [b for b in all_buckets if _domain_hit(b)]

    # --- Pinned/protected: always included as core principles ---
    # --- 钉选/protected:必入,作为核心准则 ---
    pinned = [
        b for b in all_buckets
        if b["metadata"].get("pinned") or b["metadata"].get("protected")
    ]

    # --- Unresolved by weight (top 10), excluding pinned/feel ---
    # --- 未解决按权重 top 10,排除 pinned/feel ---
    unresolved_pool = [
        b for b in all_buckets
        if not b["metadata"].get("resolved", False)
        and b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]
    unresolved_pool.sort(
        key=lambda b: decay_engine.calculate_score(b["metadata"]),
        reverse=True,
    )
    top_unresolved = unresolved_pool[:10] if not pinned_only else []

    # --- Recently active (top 5 by last_active), de-duped against above ---
    # --- 最近活跃 top 5(按 last_active),与上面去重 ---
    seen_ids = {b["id"] for b in pinned} | {b["id"] for b in top_unresolved}
    recent_pool = [
        b for b in all_buckets
        if b["id"] not in seen_ids
        and b["metadata"].get("type") not in ("feel",)
    ]
    recent_pool.sort(
        key=lambda b: b["metadata"].get("last_active", ""),
        reverse=True,
    )
    recent_active = recent_pool[:5] if not pinned_only else []

    if not pinned and not top_unresolved and not recent_active:
        return "记忆库当前空闲，没有可简报的素材。"

    # --- Build raw material: name + meta + truncated content per bucket ---
    # --- 拼接原始素材:每桶 name + meta + 截断 content ---
    def _format_bucket(b, section_tag: str) -> str:
        meta = b["metadata"]
        name = meta.get("name", b["id"])
        doms = ",".join(meta.get("domain", []) or [])
        tags = ",".join((meta.get("tags", []) or [])[:10])
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        imp = meta.get("importance", 5)
        last_active = meta.get("last_active", "")
        body = strip_wikilinks(b.get("content", ""))[:400]
        return (
            f"[{section_tag}] {name}\n"
            f"  domain:{doms} | tags:{tags}\n"
            f"  V{val:.2f}/A{aro:.2f} 重要:{imp} last_active:{last_active}\n"
            f"  {body}"
        )

    sections = []
    if pinned:
        sections.append(
            "=== 核心准则 (pinned) ===\n"
            + "\n\n".join(_format_bucket(b, "pinned") for b in pinned)
        )
    if top_unresolved:
        sections.append(
            "=== 高权重未解决 ===\n"
            + "\n\n".join(_format_bucket(b, "unresolved") for b in top_unresolved)
        )
    if recent_active:
        sections.append(
            "=== 最近活跃 ===\n"
            + "\n\n".join(_format_bucket(b, "recent") for b in recent_active)
        )

    raw_material = "\n\n".join(sections)

    # --- Compress via LLM ---
    try:
        result = await dehydrator.briefing(raw_material, max_chars=max_chars)
    except Exception as e:
        logger.error(f"Briefing compression failed: {e}")
        return f"简报生成失败：{e}"

    if not result:
        return "简报生成为空，请稍后重试。"

    # --- Stats footer for visibility ---
    stats = (
        f"\n\n---\n"
        f"_素材:{len(pinned)}钉选 / {len(top_unresolved)}未解决 / {len(recent_active)}最近活跃 "
        f"→ 简报{len(result)}字 (~{count_tokens_approx(result)}token)_"
    )

    return result + stats


# =============================================================
# Dashboard API endpoints (for lightweight Web UI)
# 仪表板 API（轻量 Web UI 用）
# =============================================================
@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_buckets(request):
    """List all buckets with metadata (no content for efficiency)."""
    from starlette.responses import JSONResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        result = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "model_valence": meta.get("model_valence"),
                "importance": meta.get("importance", 5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 1),
                "score": decay_engine.calculate_score(meta),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["GET"])
async def api_bucket_detail(request):
    """Get full bucket content by ID."""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = bucket.get("metadata", {})
    return JSONResponse({
        "id": bucket["id"],
        "metadata": meta,
        "content": strip_wikilinks(bucket.get("content", "")),
        "score": decay_engine.calculate_score(meta),
    })

@mcp.custom_route("/api/bucket/{bucket_id}", methods=["POST"])
async def api_bucket_update(request):
    """Update bucket via dashboard. Accepts JSON body with optional fields:
    name, domain (list or csv), tags (list or csv), valence, arousal,
    importance, resolved, pinned, digested, content, image_base64, image_filename.
    Mirrors trace() tool but exposed as HTTP for dashboard editing.
    """
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)

    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid json: {e}"}, status_code=400)

    updates = {}

    if "name" in data and data["name"]:
        updates["name"] = str(data["name"]).strip()

    if "domain" in data and data["domain"]:
        d = data["domain"]
        if isinstance(d, list):
            updates["domain"] = [str(x).strip() for x in d if str(x).strip()]
        else:
            updates["domain"] = [x.strip() for x in str(d).split(",") if x.strip()]

    if "tags" in data and data["tags"]:
        t = data["tags"]
        if isinstance(t, list):
            updates["tags"] = [str(x).strip() for x in t if str(x).strip()]
        else:
            updates["tags"] = [x.strip() for x in str(t).split(",") if x.strip()]

    if "valence" in data:
        try:
            v = float(data["valence"])
            if 0 <= v <= 1:
                updates["valence"] = v
        except (TypeError, ValueError):
            pass

    if "arousal" in data:
        try:
            a = float(data["arousal"])
            if 0 <= a <= 1:
                updates["arousal"] = a
        except (TypeError, ValueError):
            pass

    if "importance" in data:
        try:
            imp = int(data["importance"])
            if 1 <= imp <= 10:
                updates["importance"] = imp
        except (TypeError, ValueError):
            pass

    if "resolved" in data and data["resolved"] is not None:
        updates["resolved"] = bool(data["resolved"])

    if "pinned" in data and data["pinned"] is not None:
        updates["pinned"] = bool(data["pinned"])
        if updates["pinned"]:
            updates["importance"] = 10

    if "digested" in data and data["digested"] is not None:
        updates["digested"] = bool(data["digested"])

    # Content + optional image upload
    new_content = data.get("content", "")
    image_b64 = data.get("image_base64", "")
    image_filename = data.get("image_filename", "image")

    if image_b64 and image_b64.strip():
        try:
            image_url = r2_storage.upload_base64(image_b64, image_filename) or ""
        except Exception as e:
            logger.warning(f"R2 image upload raised in dashboard edit: {e}")
            image_url = ""
        if image_url:
            prefix = f"![{image_filename}]({image_url})\n\n"
            new_content = prefix + (new_content or bucket.get("content", ""))

    if new_content:
        updates["content"] = new_content

    if not updates:
        return JSONResponse({"error": "no fields to update"}, status_code=400)

    success = await bucket_mgr.update(bucket_id, **updates)
    if not success:
        return JSONResponse({"error": "update failed"}, status_code=500)

    if "content" in updates:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    return JSONResponse({"ok": True, "updated": list(updates.keys())})
@mcp.custom_route("/api/search", methods=["GET"])
async def api_search(request):
    """Search buckets by query."""
    from starlette.responses import JSONResponse
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    try:
        matches = await bucket_mgr.search(query, limit=10)
        result = []
        for b in matches:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "score": b.get("score", 0),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/network", methods=["GET"])
async def api_network(request):
    """Get memory network for visualization.
    Edges have two flavors:
      - kind=relation: explicit 6-type semantic edges from frontmatter `relations`
      - kind=similarity: embedding cosine similarity > 0.5 (background layer)
    """
    from starlette.responses import JSONResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        nodes = []
        edges = []
        embeddings = {}
        bucket_ids = set()

        for b in all_buckets:
            meta = b.get("metadata", {})
            bid = b["id"]
            bucket_ids.add(bid)
            nodes.append({
                "id": bid,
                "name": meta.get("name", bid),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "score": decay_engine.calculate_score(meta),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
            })
            if embedding_engine and embedding_engine.enabled:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb

        # Explicit semantic relations (6 types: causes/contributes/improves/explains/updates/kin)
        for b in all_buckets:
            src = b["id"]
            for r in (b.get("metadata", {}).get("relations") or []):
                if not isinstance(r, dict):
                    continue
                tgt = r.get("target")
                rtype = r.get("type")
                if not tgt or not rtype or tgt not in bucket_ids:
                    continue
                edges.append({
                    "source": src,
                    "target": tgt,
                    "kind": "relation",
                    "type": rtype,
                    "note": r.get("note", ""),
                })

        # Embedding similarity edges (background layer, undirected)
        ids = list(embeddings.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                if sim > 0.5:
                    edges.append({
                        "source": id_a,
                        "target": id_b,
                        "kind": "similarity",
                        "similarity": round(sim, 3),
                    })

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/breath-debug", methods=["GET"])
async def api_breath_debug(request):
    """Debug endpoint: simulate breath scoring and return per-bucket breakdown."""
    from starlette.responses import JSONResponse
    query = request.query_params.get("q", "")
    q_valence = request.query_params.get("valence")
    q_arousal = request.query_params.get("arousal")
    q_valence = float(q_valence) if q_valence else None
    q_arousal = float(q_arousal) if q_arousal else None

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        results = []
        w = {
            "topic": bucket_mgr.w_topic,
            "emotion": bucket_mgr.w_emotion,
            "time": bucket_mgr.w_time,
            "importance": bucket_mgr.w_importance,
        }
        w_sum = sum(w.values())

        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            bid = bucket["id"]
            try:
                topic = bucket_mgr._calc_topic_score(query, bucket) if query else 0.0
                emotion = bucket_mgr._calc_emotion_score(q_valence, q_arousal, meta)
                time_s = bucket_mgr._calc_time_score(meta)
                imp = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                raw_total = (
                    topic * w["topic"]
                    + emotion * w["emotion"]
                    + time_s * w["time"]
                    + imp * w["importance"]
                )
                normalized = (raw_total / w_sum) * 100 if w_sum > 0 else 0
                resolved = meta.get("resolved", False)
                if resolved:
                    normalized *= 0.3

                results.append({
                    "id": bid,
                    "name": meta.get("name", bid),
                    "domain": meta.get("domain", []),
                    "type": meta.get("type", "dynamic"),
                    "resolved": resolved,
                    "pinned": meta.get("pinned", False),
                    "scores": {
                        "topic": round(topic, 4),
                        "emotion": round(emotion, 4),
                        "time": round(time_s, 4),
                        "importance": round(imp, 4),
                    },
                    "weights": w,
                    "raw_total": round(raw_total, 4),
                    "normalized": round(normalized, 2),
                    "passed_threshold": normalized >= bucket_mgr.fuzzy_threshold,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["normalized"], reverse=True)
        passed = [r for r in results if r["passed_threshold"]]
        return JSONResponse({
            "query": query,
            "valence": q_valence,
            "arousal": q_arousal,
            "weights": w,
            "threshold": bucket_mgr.fuzzy_threshold,
            "total_candidates": len(results),
            "passed_count": len(passed),
            "results": results[:50],  # top 50 for debug
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard(request):
    """Serve the dashboard HTML page."""
    from starlette.responses import HTMLResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


@mcp.custom_route("/api/config", methods=["GET"])
async def api_config_get(request):
    """Get current runtime config (safe fields only, API key masked)."""
    from starlette.responses import JSONResponse
    dehy = config.get("dehydration", {})
    emb = config.get("embedding", {})
    api_key = dehy.get("api_key", "")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else ("***" if api_key else "")
    return JSONResponse({
        "dehydration": {
            "model": dehy.get("model", ""),
            "base_url": dehy.get("base_url", ""),
            "api_key_masked": masked_key,
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        },
        "embedding": {
            "enabled": emb.get("enabled", False),
            "model": emb.get("model", ""),
        },
        "merge_threshold": config.get("merge_threshold", 75),
        "transport": config.get("transport", "stdio"),
        "buckets_dir": config.get("buckets_dir", ""),
    })


@mcp.custom_route("/api/config", methods=["POST"])
async def api_config_update(request):
    """Hot-update runtime config. Optionally persist to config.yaml."""
    from starlette.responses import JSONResponse
    import yaml
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    updated = []

    # --- Dehydration config ---
    if "dehydration" in body:
        d = body["dehydration"]
        dehy = config.setdefault("dehydration", {})
        for key in ("model", "base_url", "max_tokens", "temperature"):
            if key in d:
                dehy[key] = d[key]
                updated.append(f"dehydration.{key}")
        if "api_key" in d and d["api_key"]:
            dehy["api_key"] = d["api_key"]
            updated.append("dehydration.api_key")
        # Hot-reload dehydrator
        dehydrator.model = dehy.get("model", "deepseek-chat")
        dehydrator.base_url = dehy.get("base_url", "")
        dehydrator.api_key = dehy.get("api_key", "")
        if hasattr(dehydrator, "client") and dehydrator.api_key:
            from openai import AsyncOpenAI
            dehydrator.client = AsyncOpenAI(
                api_key=dehydrator.api_key,
                base_url=dehydrator.base_url,
            )

    # --- Embedding config ---
    if "embedding" in body:
        e = body["embedding"]
        emb = config.setdefault("embedding", {})
        if "enabled" in e:
            emb["enabled"] = bool(e["enabled"])
            embedding_engine.enabled = emb["enabled"]
            updated.append("embedding.enabled")
        if "model" in e:
            emb["model"] = e["model"]
            embedding_engine.model = emb["model"]
            updated.append("embedding.model")

    # --- Merge threshold ---
    if "merge_threshold" in body:
        config["merge_threshold"] = int(body["merge_threshold"])
        updated.append("merge_threshold")

    # --- Persist to config.yaml if requested ---
    if body.get("persist", False):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        try:
            save_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    save_config = yaml.safe_load(f) or {}

            if "dehydration" in body:
                sc_dehy = save_config.setdefault("dehydration", {})
                for key in ("model", "base_url", "max_tokens", "temperature"):
                    if key in body["dehydration"]:
                        sc_dehy[key] = body["dehydration"][key]
                # Never persist api_key to yaml (use env var)

            if "embedding" in body:
                sc_emb = save_config.setdefault("embedding", {})
                for key in ("enabled", "model"):
                    if key in body["embedding"]:
                        sc_emb[key] = body["embedding"][key]

            if "merge_threshold" in body:
                save_config["merge_threshold"] = int(body["merge_threshold"])

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True)
            updated.append("persisted_to_yaml")
        except Exception as e:
            return JSONResponse({"error": f"persist failed: {e}", "updated": updated}, status_code=500)

    return JSONResponse({"updated": updated, "ok": True})


# =============================================================
# Import API — conversation history import
# 导入 API — 对话历史导入
# =============================================================

@mcp.custom_route("/api/import/upload", methods=["POST"])
async def api_import_upload(request):
    """Upload a conversation file and start import."""
    from starlette.responses import JSONResponse

    if import_engine.is_running:
        return JSONResponse({"error": "Import already running"}, status_code=409)

    content_type = request.headers.get("content-type", "")
    filename = ""

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file")
            if not file_field:
                return JSONResponse({"error": "No file field"}, status_code=400)
            raw_bytes = await file_field.read()
            filename = getattr(file_field, "filename", "upload")
            raw_content = raw_bytes.decode("utf-8", errors="replace")
        else:
            body = await request.body()
            raw_content = body.decode("utf-8", errors="replace")
            # Try to get filename from query params
            filename = request.query_params.get("filename", "upload")

        if not raw_content.strip():
            return JSONResponse({"error": "Empty file"}, status_code=400)

        preserve_raw = request.query_params.get("preserve_raw", "").lower() in ("1", "true")
        resume = request.query_params.get("resume", "").lower() in ("1", "true")

    except Exception as e:
        return JSONResponse({"error": f"Failed to read upload: {e}"}, status_code=400)

    # Start import in background
    async def _run_import():
        try:
            await import_engine.start(raw_content, filename, preserve_raw, resume)
        except Exception as e:
            logger.error(f"Import failed: {e}")

    asyncio.create_task(_run_import())

    return JSONResponse({
        "status": "started",
        "filename": filename,
        "size_bytes": len(raw_content.encode()),
    })


@mcp.custom_route("/api/import/status", methods=["GET"])
async def api_import_status(request):
    """Get current import progress."""
    from starlette.responses import JSONResponse
    return JSONResponse(import_engine.get_status())


@mcp.custom_route("/api/import/pause", methods=["POST"])
async def api_import_pause(request):
    """Pause the running import."""
    from starlette.responses import JSONResponse
    if not import_engine.is_running:
        return JSONResponse({"error": "No import running"}, status_code=400)
    import_engine.pause()
    return JSONResponse({"status": "pause_requested"})


@mcp.custom_route("/api/import/patterns", methods=["GET"])
async def api_import_patterns(request):
    """Detect high-frequency patterns after import."""
    from starlette.responses import JSONResponse
    try:
        patterns = await import_engine.detect_patterns()
        return JSONResponse({"patterns": patterns})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/results", methods=["GET"])
async def api_import_results(request):
    """List recently imported/created buckets for review."""
    from starlette.responses import JSONResponse
    try:
        limit = int(request.query_params.get("limit", "50"))
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # Sort by created time, newest first
        all_buckets.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        results = []
        for b in all_buckets[:limit]:
            results.append({
                "id": b["id"],
                "name": b["metadata"].get("name", ""),
                "content": b["content"][:300],
                "type": b["metadata"].get("type", ""),
                "domain": b["metadata"].get("domain", []),
                "tags": b["metadata"].get("tags", []),
                "importance": b["metadata"].get("importance", 5),
                "created": b["metadata"].get("created", ""),
            })
        return JSONResponse({"buckets": results, "total": len(all_buckets)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/review", methods=["POST"])
async def api_import_review(request):
    """Apply review decisions: mark buckets as important/noise/pinned."""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    decisions = body.get("decisions", [])
    if not decisions:
        return JSONResponse({"error": "No decisions provided"}, status_code=400)

    applied = 0
    errors = 0
    for d in decisions:
        bid = d.get("bucket_id", "")
        action = d.get("action", "")
        if not bid or not action:
            continue
        try:
            if action == "important":
                await bucket_mgr.update(bid, importance=9)
            elif action == "pin":
                await bucket_mgr.update(bid, pinned=True)
            elif action == "noise":
                await bucket_mgr.update(bid, resolved=True, importance=1)
            elif action == "delete":
                file_path = bucket_mgr._find_bucket_file(bid)
                if file_path:
                    os.remove(file_path)
            applied += 1
        except Exception as e:
            logger.warning(f"Review action failed for {bid}: {e}")
            errors += 1

    return JSONResponse({"applied": applied, "errors": errors})


# =============================================================
# Twin REST endpoints — bridge for Telegram bot (and other thin frontends)
# Twin REST 接口 —— 给 Telegram bot（及其他薄前端）用的桥接
# =============================================================
@mcp.custom_route("/api/hold", methods=["POST"])
async def api_hold(request):
    """HTTP bridge to hold tool. Body: {content, tags?, importance?, pinned?, source?}.
    HTTP 桥接 hold 工具。source 会作为额外标签合入 tags。"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    content = str(body.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)

    raw_tags = body.get("tags") or []
    if isinstance(raw_tags, list):
        tag_parts = [str(t).strip() for t in raw_tags if str(t).strip()]
    else:
        tag_parts = [t.strip() for t in str(raw_tags).split(",") if t.strip()]
    source = str(body.get("source") or "").strip()
    if source and source not in tag_parts:
        tag_parts.append(source)
    tags_csv = ",".join(tag_parts)

    try:
        importance = int(body.get("importance") or 5)
    except (TypeError, ValueError):
        importance = 5
    pinned = bool(body.get("pinned"))

    try:
        result = await hold(
            content=content,
            tags=tags_csv,
            importance=importance,
            pinned=pinned,
        )
        return JSONResponse({"result": result})
    except Exception as e:
        logger.error(f"/api/hold failed / 失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/briefing", methods=["GET"])
async def api_briefing(request):
    """HTTP bridge to briefing tool. Query: ?max_chars=&domain=&pinned_only=
    HTTP 桥接 briefing 工具，返回纯文本简报。"""
    from starlette.responses import PlainTextResponse, JSONResponse
    try:
        try:
            max_chars = int(request.query_params.get("max_chars", 1500))
        except ValueError:
            max_chars = 1500
        domain = request.query_params.get("domain", "")
        pinned_only = request.query_params.get("pinned_only", "").lower() in ("1", "true", "yes")
        text = await briefing(
            max_chars=max_chars, domain=domain, pinned_only=pinned_only
        )
        return PlainTextResponse(text)
    except Exception as e:
        logger.error(f"/api/briefing failed / 失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================
# Twin queue: bot ↔ CC asynchronous bridge (PoC: jsonl file queue)
# Twin 队列：bot ↔ CC 异步桥接（PoC：jsonl 文件队列）
#
# 朝灯发消息进 Telegram → bot → POST /api/inbox → inbox.jsonl
# CC 调 twin_pull MCP → 读未读 → 我看 → 我回 → 调 twin_send → outbox.jsonl
# bot 后台轮询 GET /api/outbox?after=<id> → 推送回 Telegram
#
# 文件位置：{buckets_dir}/twin/inbox.jsonl, outbox.jsonl
# 每行一条 JSON：{id, ts, source, text, user_id?, read?}
# =============================================================
_TWIN_DIR = os.path.join(config["buckets_dir"], "twin")
os.makedirs(_TWIN_DIR, exist_ok=True)
_TWIN_INBOX = os.path.join(_TWIN_DIR, "inbox.jsonl")
_TWIN_OUTBOX = os.path.join(_TWIN_DIR, "outbox.jsonl")
_twin_lock = asyncio.Lock()


def _twin_append_sync(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _twin_read_all_sync(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _twin_rewrite_sync(path: str, records: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# =============================================================
# Tool: twin_pull — 我（CC）拉 inbox 看朝灯发了什么
# =============================================================
@mcp.tool()
async def twin_pull(unread_only: bool = True, mark_read: bool = True, limit: int = 20) -> str:
    """从 Telegram 端拉取消息。unread_only=True 只看未读,mark_read=True 拉取后标记已读。返回最近的消息。
    朝灯通过 Telegram bot 发的话进 inbox,CC 开窗调这个工具看她说了什么。"""
    async with _twin_lock:
        records = _twin_read_all_sync(_TWIN_INBOX)
        if unread_only:
            picked = [r for r in records if not r.get("read")]
        else:
            picked = list(records)
        picked = picked[-max(1, limit):]
        if mark_read and unread_only and picked:
            picked_ids = {r["id"] for r in picked}
            for r in records:
                if r.get("id") in picked_ids:
                    r["read"] = True
            _twin_rewrite_sync(_TWIN_INBOX, records)

    if not picked:
        return "(inbox 空 / 无未读)"
    lines = []
    for r in picked:
        ts = r.get("ts", "")
        src = r.get("source", "?")
        txt = r.get("text", "")
        lines.append(f"[{ts}] {src}: {txt}")
    return "\n".join(lines)


# =============================================================
# Tool: twin_send — 我（CC）回话写到 outbox,bot 轮询拉走推到 Telegram
# =============================================================
@mcp.tool()
async def twin_send(text: str, to: str = "telegram") -> str:
    """回复朝灯,消息写到 outbox。bot 后台轮询会推到 Telegram。to 默认 telegram。"""
    text = (text or "").strip()
    if not text:
        return "空消息,未发送。"
    rec = {
        "id": uuid4().hex[:12],
        "ts": datetime.now().isoformat(timespec="seconds"),
        "to": to,
        "text": text,
    }
    async with _twin_lock:
        _twin_append_sync(_TWIN_OUTBOX, rec)
    return f"✉️ → {to} ({rec['id']})"


# =============================================================
# REST: bot 写入 inbox（朝灯发的消息）
# =============================================================
@mcp.custom_route("/api/inbox", methods=["POST"])
async def api_inbox_post(request):
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    text = str(body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    rec = {
        "id": uuid4().hex[:12],
        "ts": datetime.now().isoformat(timespec="seconds"),
        "source": str(body.get("source") or "telegram"),
        "user_id": str(body.get("user_id") or ""),
        "text": text,
        "read": False,
    }
    async with _twin_lock:
        _twin_append_sync(_TWIN_INBOX, rec)
    return JSONResponse({"id": rec["id"], "ts": rec["ts"]})


# =============================================================
# REST: bot 轮询 outbox（CC 写的回复）
# =============================================================
@mcp.custom_route("/api/outbox", methods=["GET"])
async def api_outbox_get(request):
    """Query: ?after=<id> → 返回 id 之后的所有 outbox 消息。
    after 为空时返回全部（首次连接用）。"""
    from starlette.responses import JSONResponse
    after = request.query_params.get("after", "")
    async with _twin_lock:
        records = _twin_read_all_sync(_TWIN_OUTBOX)
    if after:
        cut = -1
        for i, r in enumerate(records):
            if r.get("id") == after:
                cut = i
                break
        if cut >= 0:
            records = records[cut + 1:]
    return JSONResponse({"messages": records})


# --- Entry point / 启动入口 ---
if __name__ == "__main__":
    transport = config.get("transport", "stdio")
    logger.info(f"Ombre Brain starting | transport: {transport}")

    if transport in ("sse", "streamable-http"):
        import threading
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        # --- Application-level keepalive: ping /health every 60s ---
        # --- 应用层保活：每 60 秒 ping 一次 /health，防止 Cloudflare Tunnel 空闲断连 ---
        async def _keepalive_loop():
            await asyncio.sleep(10)  # Wait for server to fully start
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        await client.get("http://localhost:8000/health", timeout=5)
                        logger.debug("Keepalive ping OK / 保活 ping 成功")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed / 保活 ping 失败: {e}")
                    await asyncio.sleep(60)

        def _start_keepalive():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_keepalive_loop())

        t = threading.Thread(target=_start_keepalive, daemon=True)
        t.start()

        # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
        # --- 添加 CORS 中间件，让远程客户端（Cloudflare Tunnel / ngrok）能正常连接 ---
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        logger.info("CORS middleware enabled for remote transport / 已启用 CORS 中间件")
        uvicorn.run(_app, host="0.0.0.0", port=8000)
    else:
        mcp.run(transport=transport)
