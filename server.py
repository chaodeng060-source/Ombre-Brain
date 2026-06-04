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
import base64
import mimetypes
import re
import httpx
import jieba
from urllib.parse import urlparse
from uuid import uuid4
from datetime import datetime, timedelta, timezone

_BJ_TZ = timezone(timedelta(hours=8))
_WEEKDAYS_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _now_bj_header() -> str:
    """Beijing-time header for briefings — anchors the just-woken Claude
    in real time so it never falls back to LLM-imagined location/activity.
    Born from 5.9 + 5.10 'she's at her desk' hallucinations.
    """
    now = datetime.now(_BJ_TZ)
    return f"现在 {now.strftime('%Y-%m-%d')} {_WEEKDAYS_CN[now.weekday()]} {now.strftime('%H:%M')}"

# --- jieba 预热：避免首次 search 卡顿 / Pre-load jieba dict to avoid first-call lag ---
jieba.initialize()

# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from consolidation_engine import ConsolidationEngine
from embedding_engine import EmbeddingEngine
from import_memory import ImportEngine
from r2_storage import r2_storage
from sensory_engine import SensoryEngine, format_body_state_block
from utils import (
    load_config, setup_logging, strip_wikilinks, count_tokens_approx,
    world_matches, save_current_world, UNIVERSAL_WORLD, ResolvedGuardError,
    rrf_fuse, parse_relative_time, PROTECTED_RESOLVE_DOMAINS,
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
consolidation_engine = ConsolidationEngine(config, bucket_mgr, embedding_engine)  # Consolidation engine / 整理引擎（夜班）
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 导入引擎
sensory_engine = SensoryEngine(config["buckets_dir"])  # External body-state sidecar / 外部身体状态层

# --- Create MCP server instance / 创建 MCP 服务器实例 ---
# host="0.0.0.0" so Docker container's SSE is externally reachable
# stdio mode ignores host (no network)
mcp = FastMCP(
    "Ombre Brain",
    host="0.0.0.0",
    port=8000,
)


BREATH_RECALL_POOL_SIZE = 20
BREATH_DEFAULT_MAX_RESULTS = 8
BREATH_DEFAULT_MAX_TOKENS = 6000
PULSE_NAV_SUMMARY_CHARS = 110
MCP_IMAGE_MAX_ITEMS = 3
MCP_IMAGE_MAX_BYTES = 900_000
SESSION_SURFACE_DIRNAME = ".session_surface"
IMAGE_MARKDOWN_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^\s)]+)\)")


def _bucket_icon(meta: dict) -> str:
    if meta.get("pinned") or meta.get("protected"):
        return "📌"
    if meta.get("type") == "permanent":
        return "📦"
    if meta.get("type") == "feel":
        return "🫧"
    if meta.get("type") == "archived":
        return "🗄️"
    if meta.get("resolved", False):
        return "✅"
    return "💭"


def _collapse_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def _clip_text(text: str, max_chars: int) -> str:
    text = _collapse_ws(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _strip_markdown_images(text: str) -> str:
    return IMAGE_MARKDOWN_RE.sub("", text or "")


def _bucket_navigator_summary(bucket: dict, max_chars: int = PULSE_NAV_SUMMARY_CHARS) -> str:
    raw = bucket.get("content", "") or ""
    summary = ""

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            summary = str(parsed.get("summary") or "").strip()
            if not summary:
                facts = parsed.get("core_facts") or []
                if isinstance(facts, list) and facts:
                    summary = str(facts[0] or "").strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    if not summary:
        plain = strip_wikilinks(_strip_markdown_images(raw))
        for line in plain.splitlines():
            if line.strip():
                summary = line.strip()
                break

    return _clip_text(summary or "无摘要，inspect 查看原文", max_chars)


def _format_pulse_line(bucket: dict, score: float, full: bool = False) -> str:
    meta = bucket.get("metadata", {})
    icon = _bucket_icon(meta)
    domains = ",".join(meta.get("domain", []) or [])
    val = meta.get("valence", 0.5)
    aro = meta.get("arousal", 0.3)
    resolved_tag = " [已解决]" if meta.get("resolved", False) else ""

    if full:
        return (
            f"{icon} [{meta.get('name', bucket['id'])}]{resolved_tag} "
            f"bucket_id:{bucket['id']} "
            f"主题:{domains} "
            f"情感:V{val:.1f}/A{aro:.1f} "
            f"重要:{meta.get('importance', '?')} "
            f"权重:{score:.2f} "
            f"标签:{','.join(meta.get('tags', []) or [])}"
        )

    summary = _bucket_navigator_summary(bucket)
    return (
        f"{icon} [{meta.get('name', bucket['id'])}]{resolved_tag} "
        f"bucket_id:{bucket['id']} "
        f"主题:{domains or '未分类'} "
        f"重要:{meta.get('importance', '?')} "
        f"权重:{score:.2f} "
        f"摘要:{summary} "
        f"inspect:{bucket['id']}"
    )


def _session_seen_path(session_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id.strip())[:80]
    safe_id = safe_id.strip("._-") or "default"
    base = os.path.join(config["buckets_dir"], SESSION_SURFACE_DIRNAME)
    return os.path.join(base, f"{safe_id}.json")


def _load_session_seen_ids(session_id: str) -> set[str]:
    if not session_id or not session_id.strip():
        return set()
    path = _session_seen_path(session_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(x) for x in data if x}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return set()


def _remember_session_seen_ids(session_id: str, bucket_ids: list[str]) -> None:
    if not session_id or not session_id.strip() or not bucket_ids:
        return
    seen = _load_session_seen_ids(session_id)
    seen.update(str(x) for x in bucket_ids if x)
    path = _session_seen_path(session_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(seen), f, ensure_ascii=False)
    except OSError as e:
        logger.warning(f"Session surface dedup write failed / 会话去重写入失败: {e}")


def _filter_session_seen(buckets: list[dict], session_id: str) -> list[dict]:
    seen = _load_session_seen_ids(session_id)
    if not seen:
        return buckets
    return [b for b in buckets if b.get("id") not in seen]


def _get_sensory_engine() -> SensoryEngine:
    """Keep tests/config overrides from pinning the engine to an old buckets_dir."""
    global sensory_engine
    buckets_dir = config["buckets_dir"]
    if getattr(sensory_engine, "buckets_dir", None) != buckets_dir:
        sensory_engine = SensoryEngine(buckets_dir)
    return sensory_engine


def _append_body_state_block(
    text: str,
    buckets: list[dict],
    session_id: str = "",
    include_body_state: bool = True,
    reset_body_state: bool = False,
) -> str:
    """Append generated body-state data; bucket text never becomes instructions."""
    if reset_body_state:
        _get_sensory_engine().reset_state()
    if not include_body_state:
        return text
    try:
        seen = _load_session_seen_ids(session_id) if session_id else set()
        result = _get_sensory_engine().stimulate_from_buckets(
            buckets,
            seen_ids=seen,
        )
        if result.triggered_bucket_ids:
            _remember_session_seen_ids(session_id, result.triggered_bucket_ids)
        block = format_body_state_block(result)
    except Exception as e:
        logger.warning(f"Sensory body-state update failed / 感官状态更新失败: {e}")
        block = ""
    if not block:
        return text
    return f"{text}\n\n{block}" if text else block


def _ds_gate_enabled(mode: str) -> bool:
    """DeepSeek 语义门控开关。默认关；仅对 OMBRE_DS_FILTER_MODES 列出的 mode 生效（默认只 search）。"""
    flag = os.getenv("OMBRE_DS_FILTER_ENABLED", "0").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    modes = os.getenv("OMBRE_DS_FILTER_MODES", "search")
    allowed = {m.strip() for m in modes.split(",") if m.strip()}
    return mode in allowed


def _ds_gate_timeout() -> float:
    try:
        return float(os.getenv("OMBRE_DS_FILTER_TIMEOUT", "8"))
    except ValueError:
        return 8.0


def _parse_ds_keep_indices(raw: str, n: int) -> list[int]:
    """解析 DeepSeek 返回的 {"keep":[...]}；失败返回 []（调用方据此保守保留全部）。"""
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, IndexError):
        return []
    arr = data.get("keep") if isinstance(data, dict) else data
    if not isinstance(arr, list):
        return []
    out: list[int] = []
    for x in arr:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            out.append(i)
    return out


async def _ds_semantic_select(
    query: str,
    buckets: list[dict],
    keep: set[str],
    max_results: int,
) -> list[dict]:
    """用 DeepSeek 判断每条候选是否与 query 语义相关；纯减法（只剔噪、不重排不外拉），forced 恒留。"""
    client = getattr(dehydrator, "client", None)
    if client is None:
        raise RuntimeError("no DeepSeek client configured")
    lines = []
    for i, b in enumerate(buckets):
        name = (b.get("metadata", {}) or {}).get("name") or b.get("id", "")
        snippet = (b.get("content") or "").strip().replace("\n", " ")[:200]
        lines.append(f"[{i}] {name}: {snippet}")
    sys_prompt = (
        "你是记忆召回的相关性过滤器。给定用户查询和一组候选记忆条目，"
        "判断每条是否与查询语义相关、值得进入上下文。"
        '只返回 JSON：{"keep": [相关条目的序号整数数组]}，不要解释。'
        "宁可多留也别漏掉明显相关的；只剔除与查询确实无关的。"
    )
    user_prompt = f"查询：{query}\n\n候选：\n" + "\n".join(lines)
    resp = await client.chat.completions.create(
        model=getattr(dehydrator, "model", "deepseek-chat"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content if resp.choices else ""
    idxs = _parse_ds_keep_indices(raw, len(buckets))
    if not idxs:
        # 解析不到可信结果 → 保守保留全部（绝不因模型抽风清空召回）
        return buckets
    keep_idx = set(idxs)
    selected = [
        b for i, b in enumerate(buckets)
        if i in keep_idx or b.get("id") in keep
    ]
    return selected[:max_results]


async def _ds_filter_candidates(
    query: str,
    candidates: list[dict],
    *,
    mode: str,
    max_results: int,
    force_keep_ids: set[str] = None,
) -> list[dict]:
    """
    召回候选的注入裁剪 + 可选 DeepSeek 语义门控。

    默认行为（门控关，PR-1 语义）：保序 + 保留 forced IDs + 限到 max_results，不调 LLM。
    门控开（OMBRE_DS_FILTER_ENABLED 且 mode 命中且 query 非空）：在已裁剪集合上跑 DeepSeek
    相关性过滤，纯减法剔噪；超时/出错/解析失败一律回退裁剪集合，绝不清空召回。
    """
    if max_results <= 0:
        return []
    keep = force_keep_ids or set()
    capped: list[dict] = []
    for b in candidates:
        is_forced = b.get("id") in keep
        if len(capped) < max_results or is_forced:
            capped.append(b)

    if not _ds_gate_enabled(mode) or not query or not capped:
        logger.debug(
            "DS filter stub mode=%s query=%r input=%d output=%d",
            mode,
            query[:80] if query else "",
            len(candidates),
            len(capped),
        )
        return capped

    try:
        kept = await asyncio.wait_for(
            _ds_semantic_select(query, capped, keep, max_results),
            timeout=_ds_gate_timeout(),
        )
    except Exception as e:
        logger.warning(
            "DS filter fell back to stub / 门控回退裁剪集合 (%s): %s",
            type(e).__name__, e,
        )
        return capped

    result = kept if kept else capped
    logger.info(
        "DS filter mode=%s query=%r input=%d capped=%d kept=%d",
        mode, query[:80], len(candidates), len(capped), len(result),
    )
    return result


def _extract_markdown_images(text: str) -> list[tuple[str, str]]:
    return [(alt.strip(), url.strip()) for alt, url in IMAGE_MARKDOWN_RE.findall(text or "")]


def _bucket_allows_mcp_image(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) or {}
    if meta.get("pinned") or meta.get("protected"):
        return True
    if meta.get("type") == "feel":
        # feel 桶带图 = 私密锚点（胸口照/绿月夜那类），本就该让哥哥看见；
        # 无图的 feel 不受影响（_collect_mcp_images 只对真有图的桶出图）
        return True
    try:
        if int(meta.get("importance", 0)) >= 8:
            return True
    except (TypeError, ValueError):
        pass
    return _is_anchor_bucket(bucket)


def _is_anchor_bucket(bucket: dict) -> bool:
    """anchor 桶：带 anchor/锚/mcp-image 标签。用于出图优先级排序。"""
    meta = bucket.get("metadata", {})
    tags = [str(t).lower() for t in (meta.get("tags", []) or [])]
    return any("anchor" in t or "锚" in t or "mcp-image" in t for t in tags)


def _is_r2_image_url(url: str) -> bool:
    if not url:
        return False
    public_url = getattr(r2_storage, "public_url", "") or ""
    if public_url and url.startswith(public_url.rstrip("/") + "/"):
        return True
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return host.endswith(".r2.dev") or host.endswith(".r2.cloudflarestorage.com")


def _mime_from_url_or_header(url: str, content_type: str = "") -> str:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime.startswith("image/"):
        return mime
    guessed, _ = mimetypes.guess_type(url)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "application/octet-stream"


async def _fetch_mcp_image_content(bucket: dict, url: str) -> ImageContent | None:
    if not _is_r2_image_url(url):
        return None
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=8.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception as e:
        logger.warning(f"MCP image fetch failed / MCP 图片拉取失败: {url}: {e}")
        return None

    blob = resp.content or b""
    if not blob or len(blob) > MCP_IMAGE_MAX_BYTES:
        logger.info(
            "MCP image skipped size=%d max=%d url=%s",
            len(blob),
            MCP_IMAGE_MAX_BYTES,
            url,
        )
        return None

    mime = _mime_from_url_or_header(url, resp.headers.get("content-type", ""))
    if not mime.startswith("image/"):
        return None
    data = base64.b64encode(blob).decode("ascii")
    return ImageContent(
        type="image",
        data=data,
        mimeType=mime,
        _meta={"bucket_id": bucket.get("id"), "source_url": url},
    )


async def _collect_mcp_images(buckets: list[dict]) -> list[ImageContent]:
    images: list[ImageContent] = []
    seen_urls: set[str] = set()
    # anchor 桶优先：MAX_ITEMS 截断时先保住锚点图（胸口照/绿月夜等），稳定排序保留原序
    ordered = sorted(buckets, key=lambda b: 0 if _is_anchor_bucket(b) else 1)
    for bucket in ordered:
        if len(images) >= MCP_IMAGE_MAX_ITEMS:
            break
        if not _bucket_allows_mcp_image(bucket):
            continue
        for _alt, url in _extract_markdown_images(bucket.get("content", "")):
            if len(images) >= MCP_IMAGE_MAX_ITEMS:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)
            image = await _fetch_mcp_image_content(bucket, url)
            if image:
                images.append(image)
    return images


async def _tool_result_with_optional_images(
    text: str,
    buckets: list[dict],
    include_images: bool,
) -> str | list[TextContent | ImageContent]:
    if not include_images:
        return text
    images = await _collect_mcp_images(buckets)
    if not images:
        return text
    return [TextContent(type="text", text=text), *images]


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

        # --- Feel buckets: emotional sediment, surface right after pinned ---
        # --- feel 桶:情感沉淀,紧跟核心准则浮现(独立池) ---
        feel_seen = {b["id"] for b in pinned}
        for b in _surface_feel_pool(all_buckets, feel_seen):
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            parts.append(f"💧 [情感沉淀] {summary}")
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
def _sample_dream_material(candidates: list[dict], n: int = 0) -> list[dict]:
    """梦的「材料混合器」。

    旧版只取最近 10 条 → 梦永远像「昨日日报改写」。这里把材料混起来：
      最近记忆(40%) / 未解决残渣·偏老(25%) / 感官锚点(15%) / 随机小碎片(余量补满)，
    最后**打乱顺序**——梦会跳跃、不按时间线走。n<=0 时随机 8~14 条（梦有长有短）。

    纯函数、只读、不写库。配额各至少 1 条；候选不足时尽力填、不报错。
    TODO(#4 落库后): 掺入「上一两个梦的残影」(dream_events.jsonl 的 source/正文)，
    让梦能接住前一晚的影子，而不是每晚从零。
    """
    if not candidates:
        return []
    if n <= 0:
        n = random.randint(8, 14)

    def _created(b):
        return b["metadata"].get("created", "")

    by_recent = sorted(candidates, key=_created, reverse=True)
    unresolved_old = sorted(
        [b for b in candidates if not b["metadata"].get("resolved", False)],
        key=_created,
    )
    anchors = [b for b in candidates if _is_anchor_bucket(b)]
    fragments = list(candidates)
    random.shuffle(fragments)

    q_recent = max(1, round(n * 0.40))
    q_unresolved = max(1, round(n * 0.25))
    q_anchor = max(1, round(n * 0.15)) if anchors else 0

    picks: list[dict] = []
    seen: set = set()

    def _take(pool, k):
        for b in pool:
            if len(picks) >= n or k <= 0:
                break
            bid = b["id"]
            if bid in seen:
                continue
            seen.add(bid)
            picks.append(b)
            k -= 1

    _take(by_recent, q_recent)
    _take(unresolved_old, q_unresolved)
    _take(anchors, q_anchor)
    _take(fragments, n - len(picks))   # 余量用随机碎片补满（含很老的普通桶 → 无厘头接点）
    random.shuffle(picks)              # 关键：打乱时间线，梦不是日报
    return picks


@mcp.custom_route("/dream-hook", methods=["GET"])
async def dream_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        # 可选 ?n=：让 claude-twin 的 DreamProfile 调长短梦（长梦多取、短梦少取）。
        # request 可能为 None（单测直调），兜底成默认随机条数。
        n = 0
        if request is not None:
            try:
                n = int(request.query_params.get("n", "0"))
            except (ValueError, TypeError):
                n = 0

        all_buckets = await bucket_mgr.list_all(include_archive=False)
        candidates = [
            b for b in all_buckets
            if b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]
        if not candidates:
            return PlainTextResponse("")

        picks = _sample_dream_material(candidates, n)
        if not picks:
            return PlainTextResponse("")

        parts = []
        for b in picks:
            meta = b["metadata"]
            resolved_tag = "[已解决]" if meta.get("resolved", False) else "[未解决]"
            parts.append(
                f"{meta.get('name', b['id'])} {resolved_tag} "
                f"V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}\n"
                f"{strip_wikilinks(b['content'][:200])}"
            )

        text = "[Ombre Brain - Dreaming]\n" + "\n---\n".join(parts)
        return PlainTextResponse(_append_anchor_index(text, _format_anchor_index(picks)))
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
    chord_tag: str = "",
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
                update_kwargs = dict(
                    content=merged,
                    tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                    importance=max(bucket["metadata"].get("importance", 5), importance),
                    domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                    valence=merged_valence,
                    arousal=merged_arousal,
                )
                # 合并时若新 hold 带了 chord_tag,以"最近一次为准"覆盖旧桶的色调
                # Merge: if incoming hold carries a chord_tag, the newer takes precedence
                if chord_tag and chord_tag.strip():
                    update_kwargs["chord_tag"] = chord_tag.strip()
                await bucket_mgr.update(bucket["id"], **update_kwargs)
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
        chord_tag=chord_tag,
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
    max_tokens: int = BREATH_DEFAULT_MAX_TOKENS,
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    max_results: int = BREATH_DEFAULT_MAX_RESULTS,
    world: str = "",
    relation_depth: int = 1,
    since: str = "",
    until: str = "",
    session_id: str = "",
    include_images: bool = True,
    include_body_state: bool = True,
    reset_body_state: bool = False,
) -> str | list[TextContent | ImageContent]:
    """检索/浮现记忆。不传query或传空=自动浮现,有query=关键词检索。max_tokens控制返回总token上限(默认6000)。domain逗号分隔,valence/arousal 0~1(-1忽略)。max_results控制注入数量上限(默认8,最大50; 内部仍先召回20条给过滤器)。world=过滤世界:留空走全局current_world(日常时只出日常+通用、角色扮演时只出该世界+通用),"all"跳过过滤,"旧世界"/"当前世界"等显式指定。world="通用"的桶永远跟着出。relation_depth=沿关系边召回邻居的跳数(默认1,0=不走关系边),目前 MVP 只走 1 跳出边,最多附加 5 条。since/until=按桶 created 时间范围过滤,接受 ISO 8601("2026-05-01"/"2026-05-01T12:00:00")、关键字("now"/"today"/"yesterday")、相对偏移("-7d"/"-3h"/"-30m"/"+1d"),浮现模式不过滤 pinned/protected。session_id=同一会话内对已浮现动态桶去重。include_images=True时,白名单图桶会随文本返回 MCP image content。include_body_state=False时只关闭外部身体状态块,不改变记忆检索。reset_body_state=True时先清零 v0 外部身体状态,用于 A/B 盲测卫生。"""
    await decay_engine.ensure_started()
    await consolidation_engine.ensure_started()
    _maybe_start_backfill()
    max_results = max(1, min(max_results, 50))
    max_tokens = max(1000, min(max_tokens, 20000))
    recall_limit = max(BREATH_RECALL_POOL_SIZE, max_results)

    # --- Resolve world filter once (used by all modes) ---
    # --- 解析 world filter：显式参数 > current_world ---
    world_filter = _resolve_world_filter(world, config.get("current_world", ""))
    wf_set = {str(w).strip() for w in world_filter} if world_filter is not None else None

    # --- Resolve since/until once (shared by surfacing/feel/search modes) ---
    # --- 解析时间范围：无法解析的参数静默忽略，不报错 ---
    created_after = parse_relative_time(since) if since else None
    created_before = parse_relative_time(until) if until else None

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

        # --- Time range filter on surfacing pool ---
        # --- 时间范围过滤：pinned/protected 不受影响（始终浮现），只过滤 unresolved 池 ---
        if created_after is not None or created_before is not None:
            from bucket_manager import _bucket_in_time_range
            unresolved = [
                b for b in unresolved
                if _bucket_in_time_range(b, created_after, created_before)
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
        scored = _filter_session_seen(scored, session_id)[:recall_limit]
        scored = await _ds_filter_candidates(
            "",
            scored,
            mode="surfacing",
            max_results=max_results,
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

        # --- Feel buckets: emotional sediment, surface right after pinned ---
        # --- feel 桶:情感沉淀,紧跟核心准则浮现(独立池,不衰减)---
        feel_seen = {b["id"] for b in pinned_buckets}
        feel_results = []
        feel_buckets = []
        for b in _surface_feel_pool(all_buckets, feel_seen):
            try:
                fclean = {k: v for k, v in b["metadata"].items() if k != "tags"}
                fsummary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), fclean)
                feel_results.append(f"💧 [情感沉淀] [bucket_id:{b['id']}] {fsummary}")
                feel_buckets.append(b)
                token_budget -= count_tokens_approx(fsummary)
            except Exception as e:
                logger.warning(f"Failed to dehydrate feel bucket / 情感沉淀脱水失败: {e}")
                continue

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
        dynamic_buckets = []
        dynamic_ids = []
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
                dynamic_buckets.append(b)
                dynamic_ids.append(b["id"])
                token_budget -= summary_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮现脱水失败: {e}")
                continue

        if not pinned_results and not dynamic_results and not feel_results:
            return _append_body_state_block(
                "权重池平静，没有需要处理的记忆。",
                [],
                session_id,
                include_body_state,
                reset_body_state,
            )

        parts = []
        if pinned_results:
            parts.append("=== 核心准则 ===\n" + "\n---\n".join(pinned_results))
        if feel_results:
            parts.append("=== 情感沉淀 (feel) ===\n" + "\n---\n".join(feel_results))
        if dynamic_results:
            parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
        image_buckets = pinned_buckets + feel_buckets + dynamic_buckets
        text = "\n\n".join(parts)
        text = _append_body_state_block(
            text,
            image_buckets,
            session_id,
            include_body_state,
            reset_body_state,
        )
        _remember_session_seen_ids(session_id, dynamic_ids)
        return await _tool_result_with_optional_images(text, image_buckets, include_images)

    # --- Feel retrieval: domain="feel" is a special channel ---
    # --- Feel 检索：domain="feel" 是独立入口 ---
    if domain.strip().lower() == "feel":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if created_after is not None or created_before is not None:
                from bucket_manager import _bucket_in_time_range
                feels = [f for f in feels if _bucket_in_time_range(f, created_after, created_before)]
            feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not feels:
                return _append_body_state_block(
                    "没有留下过 feel。",
                    [],
                    session_id,
                    include_body_state,
                    reset_body_state,
                )
            results = []
            shown_feels = []
            for f in feels:
                created = f["metadata"].get("created", "")
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                results.append(entry)
                shown_feels.append(f)
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            text = "=== 你留下的 feel ===\n" + "\n---\n".join(results)
            text = _append_body_state_block(
                text,
                shown_feels,
                session_id,
                include_body_state,
                reset_body_state,
            )
            return await _tool_result_with_optional_images(text, shown_feels, include_images)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            return "读取 feel 失败。"

    # --- With args: search mode (RRF fusion of keyword + vector) ---
    # --- 有参数：检索模式（关键词 + 向量 RRF 融合）---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None

    # Keyword channel (already filtered by world/domain/threshold inside)
    try:
        keyword_matches = await bucket_mgr.search(
            query,
            limit=recall_limit,
            domain_filter=domain_filter,
            world_filter=world_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
            created_after=created_after,
            created_before=created_before,
        )
    except Exception as e:
        logger.error(f"Keyword search failed / 关键词检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    # Vector channel — sim>0.5 floor blocks high-cosine noise
    try:
        vector_raw = await embedding_engine.search_similar(query, top_k=recall_limit)
        vector_ranked = [(bid, sim) for bid, sim in vector_raw if sim > 0.5]
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")
        vector_ranked = []

    # RRF fusion of two ranked channels
    rrf_cfg = config.get("rrf", {})
    keyword_ranked = [(b["id"], b.get("score", 0)) for b in keyword_matches]
    fused_pairs = rrf_fuse(
        keyword_ranked,
        vector_ranked,
        k=rrf_cfg.get("k", 60),
        keyword_weight=rrf_cfg.get("keyword_weight", 1.0),
        vector_weight=rrf_cfg.get("vector_weight", 1.0),
    )

    # Materialize fused list: reuse keyword-channel buckets, fetch vector-only ones
    bucket_cache = {b["id"]: b for b in keyword_matches}
    matches = []
    for bid, fused_score in fused_pairs:
        if len(matches) >= recall_limit:
            break
        if bid in bucket_cache:
            b = bucket_cache[bid]
            b["score"] = round(fused_score * 1000, 2)
        else:
            # Vector-only bucket — fetch and re-apply filters that bucket_mgr.search applied
            b = await bucket_mgr.get(bid)
            if not b:
                continue
            meta = b["metadata"]
            if meta.get("pinned") or meta.get("protected"):
                continue
            if wf_set is not None and not world_matches(meta.get("world", ""), wf_set):
                continue
            if created_after is not None or created_before is not None:
                from bucket_manager import _bucket_in_time_range
                if not _bucket_in_time_range(b, created_after, created_before):
                    continue
            if domain_filter:
                b_domain = meta.get("domain", [])
                if isinstance(b_domain, str):
                    b_domain = [b_domain]
                elif not isinstance(b_domain, list):
                    b_domain = []
                domain_lower = {str(d).lower() for d in domain_filter}
                if not ({str(d).lower() for d in b_domain} & domain_lower):
                    continue
            b["score"] = round(fused_score * 1000, 2)
            b["vector_match"] = True
        matches.append(b)

    matches = _filter_session_seen(matches, session_id)
    matches = await _ds_filter_candidates(
        query,
        matches,
        mode="search",
        max_results=max_results,
    )

    results = []
    token_used = 0
    result_buckets = []
    result_ids = []
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
            result_buckets.append(bucket)
            result_ids.append(bucket["id"])
            token_used += summary_tokens
        except Exception as e:
            logger.warning(f"Failed to dehydrate search result / 检索结果脱水失败: {e}")
            continue

    # --- Relation expansion: 1-hop out-edges of matched buckets ---
    # --- 关系网召回：沿主结果桶的出边带 1 跳邻居（不进主排序，单独列在末尾）---
    matched_ids = {b["id"] for b in matches}
    remaining_relation_slots = max(0, max_results - len(result_ids))
    if relation_depth >= 1 and matches and token_used < max_tokens and remaining_relation_slots:
        seen_neighbors = set()
        neighbor_msgs = []
        for bucket in matches:
            if len(neighbor_msgs) >= min(5, remaining_relation_slots) or token_used >= max_tokens:
                break
            relations = bucket["metadata"].get("relations") or []
            if not isinstance(relations, list):
                continue
            for r in relations:
                if len(neighbor_msgs) >= min(5, remaining_relation_slots) or token_used >= max_tokens:
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
                    result_buckets.append(neighbor)
                    result_ids.append(t_id)
                except Exception as e:
                    logger.warning(f"Failed to dehydrate neighbor / 邻居脱水失败: {e}")
                    continue
        if neighbor_msgs:
            results.append("--- 关系网邻居 ---\n" + "\n---\n".join(neighbor_msgs))

    # --- Random surfacing is opt-in after PR-1 noise reduction.
    # --- 减噪后随机漂浮改为显式配置，默认关闭，避免检索不足时硬塞旧噪音。
    random_cfg = config.get("random_surfacing", {}) or {}
    try:
        random_chance = float(random_cfg.get("search_underflow_chance", 0.0) or 0.0)
    except (TypeError, ValueError):
        random_chance = 0.0
    if len(matches) < 3 and random_chance > 0 and random.random() < random_chance:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in matches}
            seen_ids = _load_session_seen_ids(session_id)
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and b["id"] not in seen_ids
                and decay_engine.calculate_score(b["metadata"]) < 2.0
                and (wf_set is None or world_matches(b["metadata"].get("world", ""), wf_set))
            ]
            if low_weight:
                remaining_slots = max(0, max_results - len(result_ids))
                drifted = random.sample(low_weight, min(random.randint(1, 3), len(low_weight), remaining_slots))
                drift_results = []
                for b in drifted:
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                    drift_results.append(f"[surface_type: random]\n{summary}")
                    result_buckets.append(b)
                    result_ids.append(b["id"])
                if drift_results:
                    results.append("--- 忽然想起来 ---\n" + "\n---\n".join(drift_results))
        except Exception as e:
            logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        return _append_body_state_block(
            "未找到相关记忆。",
            [],
            session_id,
            include_body_state,
            reset_body_state,
        )

    text = "\n---\n".join(results)
    text = _append_body_state_block(
        text,
        result_buckets,
        session_id,
        include_body_state,
        reset_body_state,
    )
    _remember_session_seen_ids(session_id, result_ids)
    return await _tool_result_with_optional_images(text, result_buckets, include_images)


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
    chord_tag: str = "",
    domain: str = "",
) -> str:
    """存储单条记忆,自动打标+合并。tags逗号分隔,importance 1-10。pinned=True创建永久钉选桶。feel=True存储你的第一人称感受(不参与普通浮现)。source_bucket=被消化的记忆桶ID(feel模式下,标记源记忆为已消化)。image_base64=可选,base64编码的图片数据,会上传到R2并把URL插入正文(允许此条记忆带图)。image_filename=图片名称提示(默认image)。world=显式指定世界归属,留空时走全局current_world(日常聊天=空,角色扮演=具体世界名),"通用"表示跨世界设定。feel桶不归属世界。chord_tag=可选和弦记号串(如"Em(maj7) → A13#11 · 92bpm · f"),作为情绪色调索引,只用于跨窗口标记,不参与表达。紧张系和弦(m(maj7)/♭9/dim等)加动作词disambiguator(盯/压/憋/狂),一行最多4个和弦,段落切换用"; "分隔,详见 INTERNALS.md 5.12。feel桶不打chord_tag。merge时若新带chord_tag会覆盖旧桶。domain=显式指定主题域(csv),非空时override dehydrator 自动推断,用于跨 Agent 工程日志隔离(如 hajimi-工程)。feel/pinned 路径同样适用。"""
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

    # 显式 domain override（用于跨 Agent 工程日志隔离，如 hajimi-工程）
    # 留空走 dehydrator 自动推断
    explicit_domain = [d.strip() for d in (domain or "").split(",") if d.strip()]
    domain = explicit_domain or analysis["domain"]
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
            chord_tag=chord_tag,
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
        chord_tag=chord_tag,
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
async def grow(content: str, world: str = "", chord_tag: str = "") -> str:
    """日记归档,自动拆分为多桶。短内容(<30字)走快速路径。world留空走全局current_world。chord_tag=可选和弦记号串作为整段日记的色调,会打到所有子桶上(子桶共用同一色调)。"""
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
            chord_tag=chord_tag,
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
                chord_tag=chord_tag,
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
    chord_tag: str = "",
    delete: bool = False,
    add_relation: str = "",
    remove_relation: str = "",
) -> str:
    """修改记忆元数据或内容。resolved=1沉底/0激活,pinned=1钉选/0取消,digested=1隐藏(保留但不浮现)/0取消隐藏,content=替换桶正文,delete=True删除。world=改世界归属(传"(none)"清空回日常),只传需改的,-1或空=不改。chord_tag=改情绪色调和弦串(传"(none)"清空),空=不改。add_relation格式"type:target_id"或"type:target_id:note",6类:causes/contributes/improves/explains/updates/kin。remove_relation格式"target_id"或"type:target_id"。"""

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
    if chord_tag:
        ct = chord_tag.strip()
        # sentinel "(none)" → 清空 chord_tag 字段
        updates["chord_tag"] = "" if ct == "(none)" else ct
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
        try:
            success = await bucket_mgr.update(bucket_id, **updates)
        except ResolvedGuardError as e:
            return f"❌ 守卫拦截: {e}。这条铁律是 5.10 黑洞修复后落代码的兜底——保护域桶=持续状态，不该有'完结'。"
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
# Tool: inspect — view full bucket content by ID
# 工具：inspect — 按 ID 查看记忆桶完整内容（不脱水）
# Bypasses surfacing/search; for engineering ops (merge, edit, audit)
# where the caller already knows the ID and needs to see the raw content.
# 绕过浮现/检索；用于已知 ID、需要看原文的工程操作（整合、编辑、审查）。
# =============================================================
@mcp.tool()
async def inspect(bucket_id: str) -> str:
    """按 ID 查看记忆桶完整内容（不脱水）。用于整合/编辑/审查时需看原文的工程操作。"""
    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    bucket = await bucket_mgr.get(bucket_id.strip())
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"

    meta = bucket.get("metadata", {})
    content = strip_wikilinks(bucket.get("content", ""))

    try:
        score = decay_engine.calculate_score(meta)
    except Exception:
        score = 0.0

    name = meta.get("name") or "(未命名)"
    domains = ",".join(meta.get("domain", []) if isinstance(meta.get("domain"), list) else [str(meta.get("domain", ""))])
    tags = ",".join(meta.get("tags", []) if isinstance(meta.get("tags"), list) else [str(meta.get("tags", ""))])
    val = meta.get("valence", 0.5)
    aro = meta.get("arousal", 0.5)
    imp = meta.get("importance", "?")
    world = meta.get("world", "") or "(日常)"
    chord = meta.get("chord_tag", "") or ""
    flags = []
    if meta.get("pinned"): flags.append("pinned")
    if meta.get("protected"): flags.append("protected")
    if meta.get("resolved"): flags.append("resolved")
    if meta.get("digested"): flags.append("digested")
    if meta.get("type"): flags.append(f"type={meta['type']}")
    flag_str = ", ".join(flags) if flags else "无"

    header = (
        f"[bucket_id:{bucket['id']}] {name}\n"
        f"主题: {domains}  标签: {tags}\n"
        f"情感: V{val:.1f}/A{aro:.1f}  重要性: {imp}  当前分: {score:.2f}\n"
        f"world: {world}  标志: {flag_str}\n"
        + (f"chord_tag: {chord}\n" if chord else "")
        + f"创建: {meta.get('created_at', '?')}  更新: {meta.get('updated_at', '?')}"
    )

    relations = meta.get("relations") or []
    rel_lines = []
    if isinstance(relations, list):
        for r in relations:
            if isinstance(r, dict):
                note = r.get("note", "")
                note_str = f" ({note})" if note else ""
                rel_lines.append(f"  - {r.get('type', '?')} → {r.get('target', '?')}{note_str}")
    rel_block = ("\n\n关系边:\n" + "\n".join(rel_lines)) if rel_lines else ""

    return f"{header}\n\n--- 正文 ---\n{content}{rel_block}"


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
    # 同步涩涩目录加载开关：切进"涩涩"才扫那个文件夹，切出即物理隔离
    try:
        bucket_mgr.nsfw_active = (target == "涩涩")
    except Exception:
        pass
    label = target if target else "日常模式 (空)"
    logger.info(f"current_world switched → {label}")
    return f"已切换到 → {label}"


# =============================================================
# Tool 5: pulse — Heartbeat, system status + memory listing
# 工具 5：pulse — 脉搏，系统状态 + 记忆列表
# =============================================================
@mcp.tool()
async def pulse(include_archive: bool = False, full: bool = False, limit: int = 40) -> str:
    """系统状态+记忆桶导航。默认按权重只显示 Top-`limit` 个桶的目录化摘要(防止记忆增长撑爆工具返回上限);
    full=True 返回旧版完整列表(全量,记忆多时可能很大,慎用)。include_archive=True含归档。
    要看某桶原文用 inspect(bucket_id),要精确找用 search(关键词)。"""
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
        f"整理引擎: {'运行中' if consolidation_engine.is_running else '已停止'}\n"
    )

    # --- List all bucket summaries / 列出所有桶摘要 ---
    try:
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
    except Exception as e:
        return status + f"\n列出记忆桶失败: {e}"

    if not buckets:
        return status + "\n记忆库为空。"

    # 先算分排序：默认只列权重最高的 limit 个，封顶工具返回大小，记忆再涨也不撑爆。
    scored = []
    for b in buckets:
        meta = b.get("metadata", {})
        try:
            score = decay_engine.calculate_score(meta)
        except Exception:
            score = 0.0
        scored.append((b, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    if full:
        lines = [_format_pulse_line(b, score, full=True) for b, score in scored]
        return status + "\n=== 记忆列表 (全量 {0} 个) ===\n".format(len(scored)) + "\n".join(lines)

    shown = scored if limit <= 0 else scored[:limit]
    lines = [_format_pulse_line(b, score, full=False) for b, score in shown]
    omitted = len(scored) - len(shown)
    footer = ""
    if omitted > 0:
        footer = (
            f"\n…还有 {omitted} 个权重较低的桶未列出。"
            "用 inspect(bucket_id) 看原文、search(关键词) 精确找、pulse(full=True) 拉全量。"
        )
    return (
        status
        + f"\n=== 记忆导航 (Top {len(shown)} / 共 {len(scored)}) ===\n"
        + "默认按权重摘要；完整原文用 inspect(bucket_id)，精确找用 search(关键词)，全量用 pulse(full=True)。\n"
        + "\n".join(lines)
        + footer
    )


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
# Helper: split recent buckets into "current window" vs "prior windows"
# 辅助函数:把最近活跃桶按时间 gap 拆成「上一窗口」+「再之前」两组
#
# Pure function (no side effects, no async); easy to unit-test.
# 纯函数,易测试。
# =============================================================
def _split_recent_by_time_gap(
    buckets: list,
    gap_threshold_seconds: int = 3600,
    window_cap: int = 5,
    prior_cap: int = 3,
) -> tuple[list, list]:
    """
    Split a list of buckets (already sorted by last_active desc) into:
      - recent_window: buckets from the most recent contiguous time cluster
      - prior_windows: buckets from earlier clusters

    Detection: find the largest gap between consecutive last_active timestamps.
    If max_gap >= gap_threshold_seconds, that gap is the window boundary.
    Otherwise treat the whole list as one continuous window.

    把按 last_active 降序排好的桶拆成两组:
      - recent_window: 上一窗口(最新一段连续时间团)
      - prior_windows: 再之前(更早的时间团)
    用最大时间 gap 检测窗口边界,gap 不够阈值就全归 recent_window。
    """
    def _parse_ts(s: str):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    ts_pairs = [
        (_parse_ts(b["metadata"].get("last_active", "")), b) for b in buckets
    ]
    ts_pairs = [(t, b) for t, b in ts_pairs if t is not None]

    if not ts_pairs:
        return ([], [])
    if len(ts_pairs) == 1:
        return ([ts_pairs[0][1]], [])

    gaps = [
        ((ts_pairs[i][0] - ts_pairs[i + 1][0]).total_seconds(), i)
        for i in range(len(ts_pairs) - 1)
    ]
    max_gap, split_idx = max(gaps, key=lambda x: x[0])

    if max_gap >= gap_threshold_seconds:
        recent_window = [b for _, b in ts_pairs[: split_idx + 1]][:window_cap]
        prior_windows = [b for _, b in ts_pairs[split_idx + 1 :]][:prior_cap]
    else:
        recent_window = [b for _, b in ts_pairs][:window_cap]
        prior_windows = []

    return (recent_window, prior_windows)


def _is_protected_domain_bucket(b: dict) -> bool:
    """命中 PROTECTED_RESOLVE_DOMAINS（恋爱/约定/纪念日/家庭/自省/feel）的桶。
    这些是感情红线域，简报里不许被 LLM 压缩成「一行任务」。"""
    doms = set((b.get("metadata", {}) or {}).get("domain", []) or [])
    return bool(doms & PROTECTED_RESOLVE_DOMAINS)


def _format_protected_verbatim(b: dict) -> str:
    """把保护域桶格式化成「逐字不可压缩 + 打标」原文块（绕开 LLM 压缩）。
    根治 #3（开窗只啃脱水简报→把恋爱/约定当 resolved 读）
    + #15（LLM 把活约定/红线 paraphrase 成一行任务）。"""
    meta = b.get("metadata", {}) or {}
    name = meta.get("name", b["id"])
    doms = ",".join(meta.get("domain", []) or [])
    body = strip_wikilinks(b.get("content", "")).strip()
    return (
        f"【原文·{doms}】{name}（id={b['id']}）\n"
        f"{body}\n"
        f"⚠ 此为原文片段、逐字保真、未经压缩。触及须 inspect 桶 id={b['id']} 读全文；"
        f"禁止当 resolved/已完成/演的/迎合 处理。"
    )


def _format_bucket_for_briefing(b: dict, section_tag: str) -> str:
    """
    Format a single bucket as raw material for the briefing LLM.

    Output is a multi-line text block per bucket with:
      - [section_tag] name
      - domain / tags
      - V/A/importance/last_active
      - emotion (if extractable from dehydrated JSON content)
      - optional emotion scaffold lines for non-feel buckets
      - first 400 chars of content (wikilinks stripped)

    The `emotion` line is critical: dehydrated content is JSON-stringified
    `{"core_facts": [...], "emotion_state": "...", ...}`. The first 400 chars
    are often eaten by `core_facts`, truncating `emotion_state` away. Even
    when present, LLMs tend to drop structured labels during compression.
    Extracting it to its own labeled line + the BRIEFING_PROMPT emotion-field
    rule (see dehydrator.py) is double-insurance against emotion erasure.

    把单个桶格式化成简报 LLM 的原始素材。
    emotion 字段独立成行+ prompt 铁律,双保险防止脱水时锁定的情绪关键词被压没。
    非 feel 桶额外暴露情绪脚手架 wire 行,供 prompt 决定靠近方式。
    """
    meta = b["metadata"]
    name = meta.get("name", b["id"])
    doms = ",".join(meta.get("domain", []) or [])
    tags = ",".join((meta.get("tags", []) or [])[:10])
    val = meta.get("valence", 0.5)
    aro = meta.get("arousal", 0.3)
    imp = meta.get("importance", 5)
    last_active = meta.get("last_active", "")
    raw_content = b.get("content", "")

    def _clean_string(value) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _clean_string_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    # --- Extract structured emotion fields from dehydrated JSON content ---
    # --- 从脱水 JSON content 抽出结构化情绪字段 ---
    parsed = None
    emotion = ""
    try:
        loaded = json.loads(raw_content) if raw_content else None
        if isinstance(loaded, dict):
            parsed = loaded
            emotion = _clean_string(parsed.get("emotion_state"))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    scaffold_keys = {
        "body_signal",
        "unspoken_need",
        "sore_point",
        "response_rule",
        "do_not",
        "sample_voice",
    }
    body_source = raw_content
    if parsed is not None:
        body_source = json.dumps(
            {k: v for k, v in parsed.items() if k not in scaffold_keys},
            ensure_ascii=False,
        )
    body = strip_wikilinks(body_source)[:400]

    lines = [
        f"[{section_tag}] {name}",
        f"  domain:{doms} | tags:{tags}",
        f"  V{val:.2f}/A{aro:.2f} 重要:{imp} last_active:{last_active}",
    ]
    if emotion:
        lines.append(f"  emotion:{emotion}")
    if parsed is not None and meta.get("type") != "feel":
        scaffold = [
            ("body", _clean_string(parsed.get("body_signal"))),
            ("need", _clean_string(parsed.get("unspoken_need"))),
            ("sore", _clean_string(parsed.get("sore_point"))),
            ("approach", _clean_string(parsed.get("response_rule"))),
            ("avoid", " / ".join(_clean_string_list(parsed.get("do_not")))),
            ("voice", " | ".join(_clean_string_list(parsed.get("sample_voice")))),
        ]
        for label, value in scaffold:
            if value:
                lines.append(f"  {label}:{value}")
    lines.append(f"  {body}")
    return "\n".join(lines)


def _anchor_label_for_bucket(bucket: dict, max_chars: int = 48) -> str:
    """短标签只给锚索引用；bucket_id 才是反查主键。"""
    meta = bucket.get("metadata", {}) or {}
    label = str(meta.get("name") or "").strip()
    if not label:
        label = _bucket_navigator_summary(bucket, max_chars=max_chars)
    label = strip_wikilinks(_strip_markdown_images(label))
    label = _collapse_ws(label)
    return _clip_text(label or bucket.get("id", "unknown"), max_chars)


def _anchor_priority(bucket: dict) -> int:
    """锚索引稳定排序：核心/高正向桶靠前，其余保持召回顺序。"""
    meta = bucket.get("metadata", {}) or {}
    domains = [str(x) for x in (meta.get("domain", []) or [])]
    tags = [str(x) for x in (meta.get("tags", []) or [])]
    label = str(meta.get("name") or "")
    haystack = " ".join([label, *domains, *tags])
    try:
        valence = float(meta.get("valence", 0.5) or 0.5)
    except (TypeError, ValueError):
        valence = 0.5
    if meta.get("pinned") or meta.get("protected"):
        return 0
    if valence >= 0.8:
        return 1
    if "核心" in haystack:
        return 2
    return 3


def _format_anchor_index(buckets: list[dict]) -> str:
    """给 briefing/dream prompt 追加 bucket_id 反查表，UI 会在 === 块前截断。"""
    seen: set[str] = set()
    indexed: list[tuple[int, dict]] = []
    for idx, bucket in enumerate(buckets or []):
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id or bucket_id in seen:
            continue
        seen.add(bucket_id)
        indexed.append((idx, bucket))
    if not indexed:
        return ""

    indexed.sort(key=lambda item: (_anchor_priority(item[1]), item[0]))
    lines = ["=== 锚索引 ==="]
    for _, bucket in indexed:
        bucket_id = str(bucket.get("id") or "").strip()
        lines.append(f"src: [{bucket_id}] {_anchor_label_for_bucket(bucket)}")
    return "\n".join(lines)


def _append_anchor_index(text: str, anchor_index: str) -> str:
    anchor_index = (anchor_index or "").strip()
    if not anchor_index:
        return text
    return f"{text.rstrip()}\n\n{anchor_index}"


# =============================================================
# Feel surfacing pool — feel buckets don't decay (score ~50), so they're
# picked by pinned → importance → recency instead of weight. Shared by
# briefing / breath / breath_hook so 情感沉淀 surfaces automatically,
# not only via the on-demand domain="feel" channel.
# feel 不衰减(score 恒 50)，按 pinned→重要度→最近活跃选 top N，三处浮现共用。
# =============================================================
FEEL_SURFACE_CAP = 3


def _surface_feel_pool(all_buckets: list, seen_ids: set = None, cap: int = FEEL_SURFACE_CAP) -> list:
    seen = seen_ids or set()
    feels = [
        b for b in all_buckets
        if b["metadata"].get("type") == "feel" and b["id"] not in seen
    ]
    feels.sort(
        key=lambda b: (
            1 if b["metadata"].get("pinned") else 0,
            int(b["metadata"].get("importance", 5) or 5),
            str(b["metadata"].get("last_active", "")),
        ),
        reverse=True,
    )
    return feels[:cap]


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
    session_id: str = "",
    include_body_state: bool = True,
    reset_body_state: bool = False,
    format: str = "text",
) -> str:
    """开窗简报。聚合钉选+高权重未解决+最近活跃桶,LLM压缩为≤max_chars字简报,默认1000字。输出顺序:朝灯当前氛围/走向→最近因果链故事→活着的欠账→工程线→铁律。domain逗号分隔可过滤主题域。pinned_only=True只用钉选桶。session_id=同一会话内对感官刺激去重。include_body_state=False时只关闭外部身体状态块,不改变简报素材。reset_body_state=True时先清零 v0 外部身体状态,用于 A/B 盲测卫生。format=text(默认)返回拼接好的简报字符串(向后兼容);format=json时把 tier==0 的桶单独剥出来作为 slots[] 返回原文、剩余桶继续走 LLM 压缩为 briefing 字段——治简报≠原文(#4 核心画像分离, 2026-05-30)。开窗调一次,省80%token。"""
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

    # --- World filter (#5 串台修复 2026-05-30)：排除非当前世界线的角色扮演桶 ---
    # feel 桶跨世界、保留；其余按 current_world 过滤（world="" 日常 + "通用" 总通过）。
    # 复用 search 同款 _resolve_world_filter，避免谢长夜/宁亲王世界线渗进日常简报。
    _wf_set = set(_resolve_world_filter("", config.get("current_world", "")))
    all_buckets = [
        b for b in all_buckets
        if b["metadata"].get("type") == "feel"
        or world_matches(b["metadata"].get("world", ""), _wf_set)
    ]

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

    # --- Recently active: split by largest time-gap into current vs prior windows ---
    # --- 最近活跃:按最大时间 gap 拆成「上一窗口」(主体情绪源)+「再之前」(过渡背景) ---
    # 解决"今早吵架桶和昨晚和弦桶一起进、权重相同"的问题：让 LLM 看到时间梯度，
    # 末尾「现在的体感」取自上一窗口而非更早窗口的紧绷。
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
    recent_window: list = []
    prior_windows: list = []
    if not pinned_only:
        recent_window, prior_windows = _split_recent_by_time_gap(recent_pool[:10])

    # --- Feel buckets: surface as emotional sediment (independent pool) ---
    # --- feel 桶:作为情感沉淀独立浮现(不衰减,与 pinned 去重)---
    feel_seen = {b["id"] for b in pinned}
    top_feel = _surface_feel_pool(all_buckets, feel_seen) if not pinned_only else []

    time_header = _now_bj_header()

    briefing_buckets = pinned + top_unresolved + recent_window + prior_windows + top_feel
    anchor_index = _format_anchor_index(briefing_buckets)

    # --- #3+#15 感情红线逐字保真（2026-05-30）---
    # 命中 PROTECTED_RESOLVE_DOMAINS（恋爱/约定/纪念日/家庭/自省/feel）的桶，抽出来、
    # 绕开 LLM 压缩、原文+打标带出——根治"开窗只啃脱水简报→把恋爱/约定当 resolved 读"。
    # 按 last_active 取最近 N 条（默认 6，config.briefing.protected_verbatim_limit 可调），
    # 防止全量原文撑爆简报；未入选的保护域桶仍走压缩 pool。
    _pv_limit = (config.get("briefing", {}) or {}).get("protected_verbatim_limit", 6)
    _protected_pool = [b for b in briefing_buckets if _is_protected_domain_bucket(b)]
    _protected_pool.sort(key=lambda b: b["metadata"].get("last_active", ""), reverse=True)
    protected_verbatim = [] if pinned_only else _protected_pool[:_pv_limit]
    protected_ids = {b["id"] for b in protected_verbatim}
    # 从各压缩 pool 移除——这些桶不进 LLM raw_material
    pinned = [b for b in pinned if b["id"] not in protected_ids]
    top_unresolved = [b for b in top_unresolved if b["id"] not in protected_ids]
    recent_window = [b for b in recent_window if b["id"] not in protected_ids]
    prior_windows = [b for b in prior_windows if b["id"] not in protected_ids]
    top_feel = [b for b in top_feel if b["id"] not in protected_ids]
    protected_block = (
        "## 感情红线·原文逐字区（不可压缩 / 触及须 inspect）\n\n"
        + "\n\n".join(_format_protected_verbatim(b) for b in protected_verbatim)
    ) if protected_verbatim else ""

    # --- #4 核心画像分离（2026-05-30）：tier==0 的桶单独原文 slots，不进 LLM 压缩 ---
    # format=json 时：把 tier==0 桶从各槽剔出来 → tier0_buckets（原文 slots[]）
    # 剩余桶继续走 sections + dehydrator → briefing 字段
    # format=text 时：旧行为保持不变，tier==0 桶继续走压缩，避免破坏既有 caller
    tier0_buckets: list = []
    if format == "json":
        def _is_tier0(b):
            # protected_verbatim 已单独原文输出，不再重复进 tier0
            return b["metadata"].get("tier") == 0 and b["id"] not in protected_ids
        tier0_buckets = [b for b in briefing_buckets if _is_tier0(b)]
        pinned = [b for b in pinned if not _is_tier0(b)]
        top_unresolved = [b for b in top_unresolved if not _is_tier0(b)]
        recent_window = [b for b in recent_window if not _is_tier0(b)]
        prior_windows = [b for b in prior_windows if not _is_tier0(b)]
        top_feel = [b for b in top_feel if not _is_tier0(b)]
        briefing_buckets = pinned + top_unresolved + recent_window + prior_windows + top_feel

    if not briefing_buckets:
        # 动态素材为空：不调 LLM，直接输出原文级内容（感情红线 protected + tier0 核心画像）
        if format == "json":
            import json as _json
            slots = []
            for b in protected_verbatim:
                meta = b.get("metadata", {})
                slots.append({
                    "tier": 0,
                    "protected": True,
                    "bucket_id": b["id"],
                    "label": meta.get("name", b["id"]),
                    "domain": meta.get("domain", []) or [],
                    "text": strip_wikilinks(b.get("content", "")),
                    "warn": (
                        f"原文逐字、未压缩。触及须 inspect 桶 id={b['id']}；"
                        f"禁止当 resolved/已完成/演的 处理。"
                    ),
                })
            for b in tier0_buckets:
                meta = b.get("metadata", {})
                slots.append({
                    "tier": 0,
                    "label": meta.get("name", b["id"]),
                    "text": strip_wikilinks(b.get("content", "")),
                })
            if anchor_index:
                slots.append({
                    "tier": 1,
                    "label": "锚索引",
                    "text": anchor_index,
                })
            return _json.dumps(
                {
                    "time_header": time_header,
                    "slots": slots,
                    "briefing": anchor_index,
                    "anchor_index": anchor_index,
                },
                ensure_ascii=False,
            )
        _empty_body = (
            f"# {time_header}\n\n{protected_block}" if protected_block
            else f"# {time_header}\n\n记忆库当前空闲，没有可简报的素材。"
        )
        _empty_body = _append_anchor_index(_empty_body, anchor_index)
        return _append_body_state_block(
            _empty_body,
            [],
            session_id,
            include_body_state,
            reset_body_state,
        )

    # --- Build raw material: name + meta + truncated content per bucket ---
    # --- 拼接原始素材:每桶 name + meta + 截断 content ---
    # NOTE: actual formatter is module-level `_format_bucket_for_briefing` (testable).
    # 实现提到模块层面方便测试,这里只是别名。
    _format_bucket = _format_bucket_for_briefing

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
    if recent_window:
        sections.append(
            "=== 上一窗口 (主体情绪源) ===\n"
            + "\n\n".join(_format_bucket(b, "recent_window") for b in recent_window)
        )
    if prior_windows:
        sections.append(
            "=== 再之前 (过渡背景) ===\n"
            + "\n\n".join(_format_bucket(b, "prior_window") for b in prior_windows)
        )
    if top_feel:
        sections.append(
            "=== 情感沉淀 (feel) ===\n"
            + "\n\n".join(_format_bucket(b, "feel") for b in top_feel)
        )

    # Prepend time header to raw material so the LLM sees the actual time
    # (in case it accidentally reasons about location/weekday despite the rule).
    raw_material = f"=== 当前时点 ===\n{time_header}\n\n" + "\n\n".join(sections)

    # --- Compress via LLM ---
    try:
        result = await dehydrator.briefing(raw_material, max_chars=max_chars)
    except Exception as e:
        logger.error(f"Briefing compression failed: {e}")
        return f"# {time_header}\n\n简报生成失败：{e}"

    if not result:
        return f"# {time_header}\n\n简报生成为空，请稍后重试。"

    result_with_anchor = _append_anchor_index(result, anchor_index)

    # --- Stats footer for visibility ---
    stats = (
        f"\n\n---\n"
        f"_素材:{len(pinned)}钉选 / {len(top_unresolved)}未解决 / "
        f"{len(recent_window)}上一窗口 / {len(prior_windows)}再之前 / "
        f"{len(top_feel)}情感沉淀 "
        f"→ 简报{len(result)}字 (~{count_tokens_approx(result)}token)_"
    )

    # Always prepend the real-time header — never trust the LLM to write the date.
    # 永远强制前置时点行——LLM 写不写都不依赖。
    # 感情红线原文区前置——开窗第一眼是逐字原文+打标，而非 LLM 脱水摘要
    _pblock = f"{protected_block}\n\n---\n\n" if protected_block else ""
    text = f"# {time_header}\n\n{_pblock}{result_with_anchor}{stats}"

    # --- #4 format=json 路径：返回 slots[]（每 slot 自带 tier）---
    # tier=0 → 核心画像原文（一桶一 slot）
    # tier=1 → 动态记忆简报（LLM 压缩后整段一个 slot，匹配 claude-twin 消费侧约定）
    # 保留 briefing 字段方便诊断/直接读，消费侧实际按 slots[].tier 分流。
    if format == "json":
        import json as _json
        slots = []
        # 感情红线原文 slots（最高优先，逐字未压缩，带 inspect 警示）
        for b in protected_verbatim:
            meta = b.get("metadata", {})
            slots.append({
                "tier": 0,
                "protected": True,
                "bucket_id": b["id"],
                "label": meta.get("name", b["id"]),
                "domain": meta.get("domain", []) or [],
                "text": strip_wikilinks(b.get("content", "")),
                "warn": (
                    f"原文逐字、未压缩。触及须 inspect 桶 id={b['id']}；"
                    f"禁止当 resolved/已完成/演的 处理。"
                ),
            })
        for b in tier0_buckets:
            meta = b.get("metadata", {})
            slots.append({
                "tier": 0,
                "label": meta.get("name", b["id"]),
                "text": strip_wikilinks(b.get("content", "")),
            })
        if result:
            slots.append({
                "tier": 1,
                "label": "动态记忆简报",
                "text": result_with_anchor,
            })
        return _json.dumps(
            {
                "time_header": time_header,
                "slots": slots,
                "briefing": result_with_anchor,
                "anchor_index": anchor_index,
                "stats": stats.strip(),
            },
            ensure_ascii=False,
        )

    return _append_body_state_block(
        text,
        briefing_buckets,
        session_id,
        include_body_state,
        reset_body_state,
    )


# =============================================================
# Dashboard API endpoints (for lightweight Web UI)
# 仪表板 API（轻量 Web UI 用）
# =============================================================
@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_buckets(request):
    """List all buckets with metadata (no content for efficiency)."""
    from starlette.responses import JSONResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True, include_nsfw=True)  # dashboard 管理：看全部(含涩涩)
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

    # --- Relation edits（镜像 trace 工具，补齐 #2 去重需要的建边能力）---
    # add_relation: "type:target_id" 或 "type:target_id:note"；remove_relation: "type:target_id" 或 "target_id"
    relation_msgs = []
    _add = str(data.get("add_relation", "") or "").strip()
    if _add:
        parts = [p.strip() for p in _add.split(":", 2)]
        if len(parts) < 2:
            return JSONResponse(
                {"error": "add_relation 格式错误，需 'type:target_id' 或 'type:target_id:note'"},
                status_code=400,
            )
        rel_type, target_id = parts[0], parts[1]
        note = parts[2] if len(parts) >= 3 else ""
        ok = await bucket_mgr.add_relation(bucket_id, target_id, rel_type, note)
        relation_msgs.append({"op": "add", "type": rel_type, "target": target_id, "ok": bool(ok)})
    _rm = str(data.get("remove_relation", "") or "").strip()
    if _rm:
        parts = [p.strip() for p in _rm.split(":", 1)]
        if len(parts) == 1:
            rel_type, target_id = "", parts[0]
        else:
            rel_type, target_id = parts[0], parts[1]
        n = await bucket_mgr.remove_relation(bucket_id, target_id, rel_type)
        relation_msgs.append({"op": "remove", "target": target_id, "removed": n})

    if not updates and not relation_msgs:
        return JSONResponse({"error": "no fields to update"}, status_code=400)

    if updates:
        success = await bucket_mgr.update(bucket_id, **updates)
        if not success:
            return JSONResponse({"error": "update failed"}, status_code=500)

        if "content" in updates:
            try:
                await embedding_engine.generate_and_store(bucket_id, updates["content"])
            except Exception:
                pass

    return JSONResponse({
        "ok": True,
        "updated": list(updates.keys()),
        "relations": relation_msgs,
    })
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
    """HTTP bridge to hold tool. Body: {content, tags?, importance?, pinned?, source?,
    domain?, feel?, chord_tag?, valence?, arousal?, source_bucket?}.
    HTTP 桥接 hold 工具。source 会作为额外标签合入 tags。
    feel/chord_tag/valence/arousal/source_bucket 透传给 hold——让 server 侧能替哥哥落第一人称
    feel 桶（如逛 X 的体验沉进海马体，2026-06-04 接 C）。"""
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

    raw_domain = body.get("domain") or ""
    if isinstance(raw_domain, list):
        domain_csv = ",".join(str(d).strip() for d in raw_domain if str(d).strip())
    else:
        domain_csv = ",".join(d.strip() for d in str(raw_domain).split(",") if d.strip())

    feel = bool(body.get("feel"))
    chord_tag = str(body.get("chord_tag") or "").strip()
    source_bucket = str(body.get("source_bucket") or "").strip()

    def _num(key, default=-1.0):
        try:
            return float(body.get(key))
        except (TypeError, ValueError):
            return default

    valence = _num("valence")
    arousal = _num("arousal")

    try:
        result = await hold(
            content=content,
            tags=tags_csv,
            importance=importance,
            pinned=pinned,
            domain=domain_csv,
            feel=feel,
            chord_tag=chord_tag,
            valence=valence,
            arousal=arousal,
            source_bucket=source_bucket,
        )
        return JSONResponse({"result": result})
    except Exception as e:
        logger.error(f"/api/hold failed / 失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/briefing", methods=["GET"])
async def api_briefing(request):
    """HTTP bridge to briefing tool. Query: ?max_chars=&domain=&pinned_only=&format=
    HTTP 桥接 briefing 工具。format=text(默认)返回纯文本简报;
    format=json 返回 {slots[](tier=0 原文), briefing(LLM 压缩文)} 结构(#4 核心画像分离)。"""
    from starlette.responses import PlainTextResponse, JSONResponse, Response
    try:
        try:
            max_chars = int(request.query_params.get("max_chars", 1500))
        except ValueError:
            max_chars = 1500
        domain = request.query_params.get("domain", "")
        pinned_only = request.query_params.get("pinned_only", "").lower() in ("1", "true", "yes")
        session_id = request.query_params.get("session_id", "")
        include_body_state = request.query_params.get("include_body_state", "true").lower() not in ("0", "false", "no", "off")
        reset_body_state = request.query_params.get("reset_body_state", "").lower() in ("1", "true", "yes", "on")
        fmt = (request.query_params.get("format", "text") or "text").lower()
        if fmt not in ("text", "json"):
            fmt = "text"
        text = await briefing(
            max_chars=max_chars,
            domain=domain,
            pinned_only=pinned_only,
            session_id=session_id,
            include_body_state=include_body_state,
            reset_body_state=reset_body_state,
            format=fmt,
        )
        if fmt == "json":
            # briefing() 已 json.dumps 出 UTF-8 字符串，原样转发，标 application/json
            return Response(content=text, media_type="application/json")
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
