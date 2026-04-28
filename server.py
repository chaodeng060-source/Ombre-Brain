# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模块：MCP 服务器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 启动 Ombre Brain MCP 服务，注册记忆操作工具供 Claude 调用。
# ============================================================

import os
import sys
import random
import logging
import asyncio
import httpx
import threading
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# --- Ensure same-directory modules can be imported ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from utils import load_config, setup_logging
from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine

# --- Init Config & Logging ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain.server")

# --- Init Core Modules ---
bucket_mgr = BucketManager(config)
dehydrator = Dehydrator(config)
decay_engine = DecayEngine(config, bucket_mgr)

# --- Init MCP Server ---
mcp = FastMCP("Ombre Brain Memory", dependencies=["httpx"])


# ============================================================
# Tool 1: breath (Surface & Search)
# ============================================================
@mcp.tool()
async def breath(
    query: str = None,
    limit: int = 5,
    domain_filter: list[str] = None,
    query_valence: float = None,
    query_arousal: float = None
) -> str:
    """
    [OMBRE] Surface unresolved memories or search by keyword.
    浮现未解决的记忆，或通过多维索引（关键词+情感+主题）精确检索记忆。
    """
    try:
        # 确保衰减引擎已启动
        await decay_engine.ensure_started()

        if query and query.strip():
            # ======= 核心修复点：强制走精准多维检索 (Max-Win) =======
            logger.info(f"Breath triggered search / 触发记忆搜索: query='{query}'")
            results = await bucket_mgr.search(
                query=query,
                limit=limit,
                domain_filter=domain_filter,
                query_valence=query_valence,
                query_arousal=query_arousal
            )
        else:
            # ======= 没有查询词时：走主动浮现模式（自然呼吸） =======
            logger.info("Breath triggered auto-surface / 触发记忆自然浮现")
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            
            scored_buckets = []
            for b in all_buckets:
                meta = b.get("metadata", {})
                # 跳过固化和保护的桶，跳过沉淀物
                if meta.get("type") in ("permanent", "feel") or meta.get("pinned"):
                    continue
                # 只浮现未解决的记忆
                if not meta.get("resolved", False):
                    # 走衰减引擎算分
                    score = decay_engine.calculate_score(meta)
                    b["score"] = score
                    scored_buckets.append(b)
            
            # 按得分降序排序，取前 limit 条
            scored_buckets.sort(key=lambda x: x.get("score", 0), reverse=True)
            results = scored_buckets[:limit]

        if not results:
            return "没有找到匹配的记忆，或者当前没有需要浮现的未解决往事。"

        # 格式化输出并触碰（Touch）唤醒记忆
        output = []
        for b in results:
            meta = b.get("metadata", {})
            bucket_id = b["id"]
            name = meta.get("name", bucket_id)
            domain = meta.get("domain", ["未分类"])[0]
            tags = meta.get("tags", [])
            score = b.get("score", 0)

            # 触碰记忆：触发时间涟漪，增加活跃度
            await bucket_mgr.touch(bucket_id)

            # 组装返回文本
            tags_str = f" #{' #'.join(tags)}" if tags else ""
            header = f"[{domain}] {name}{tags_str} (Score: {score:.2f})"
            
            # 截断过长的正文，防止 Token 爆炸
            content = b.get("content", "")
            if len(content) > 1000:
                content = content[:1000] + "\n...[Content Truncated / 记忆过长已截断]..."

            output.append(f"{header}\nID: {bucket_id}\n---\n{content}")

        return "\n\n======\n\n".join(output)

    except Exception as e:
        logger.error(f"Breath execution failed / 呼吸动作执行失败: {e}")
        return f"记忆检索失败，由于系统异常: {str(e)}"


# ============================================================
# Tool 2: hold (Store Memory)
# ============================================================
@mcp.tool()
async def hold(
    content: str,
    domain: str = "mood",
    tags: list[str] = None,
    importance: int = 5,
    valence: float = 0.5,
    arousal: float = 0.3,
    pinned: bool = False
) -> str:
    """
    [OMBRE] Store a new memory.
    捕捉并存储一条新的记忆。
    """
    try:
        domain_list = [domain] if domain else ["碎碎念"]
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=tags,
            importance=importance,
            domain=domain_list,
            valence=valence,
            arousal=arousal,
            pinned=pinned
        )
        return f"记忆已成功锁定。ID: {bucket_id}"
    except Exception as e:
        logger.error(f"Hold execution failed / 存储动作执行失败: {e}")
        return f"存储失败: {str(e)}"


# ============================================================
# Tool 3: trace (Modify Memory)
# ============================================================
@mcp.tool()
async def trace(
    bucket_id: str,
    resolved: bool = None,
    pinned: bool = None,
    digested: bool = None,
    domain: list[str] = None
) -> str:
    """
    [OMBRE] Modify metadata of an existing memory.
    修改已有记忆的状态。
    """
    try:
        kwargs = {}
        if resolved is not None: kwargs["resolved"] = resolved
        if pinned is not None: kwargs["pinned"] = pinned
        if digested is not None: kwargs["digested"] = digested
        if domain is not None: kwargs["domain"] = domain
        
        success = await bucket_mgr.update(bucket_id, **kwargs)
        if success:
            return f"记忆 {bucket_id} 状态更新成功。"
        return f"找不到指定的记忆: {bucket_id}"
    except Exception as e:
        logger.error(f"Trace execution failed / 修改动作执行失败: {e}")
        return f"状态更新失败: {str(e)}"


# ============================================================
# Tool 4: grow (Process & Digest)
# ============================================================
@mcp.tool()
async def grow(content: str) -> str:
    """
    [OMBRE] Digest a long text or diary entry into multiple compressed memories.
    将长文本或日记消化为多条压缩记忆。
    """
    try:
        results = await dehydrator.process(content)
        if not results:
            return "消化失败：无法提取有效信息。"
        
        output = []
        for res in results:
            b_id = await bucket_mgr.create(
                content=res.get("content"),
                tags=res.get("tags"),
                importance=res.get("importance", 5),
                domain=res.get("domain"),
                valence=res.get("valence", 0.5),
                arousal=res.get("arousal", 0.3)
            )
            output.append(f"- 提取记忆: {res.get('name')} (ID: {b_id})")
        return "长文本消化完成：\n" + "\n".join(output)
    except Exception as e:
        logger.error(f"Grow execution failed / 消化动作执行失败: {e}")
        return f"消化过程异常: {str(e)}"


# ============================================================
# Tool 5: pulse (System Status)
# ============================================================
@mcp.tool()
async def pulse() -> str:
    """
    [OMBRE] Check system status and memory stats.
    检查系统心跳与记忆统计。
    """
    try:
        stats = await bucket_mgr.get_stats()
        return (
            f"Ombre Brain 脉搏正常。\n"
            f"动态记忆: {stats.get('dynamic_count', 0)}\n"
            f"归档记忆: {stats.get('archive_count', 0)}\n"
            f"永久记忆: {stats.get('permanent_count', 0)}\n"
            f"总容量: {stats.get('total_size_kb', 0):.2f} KB"
        )
    except Exception as e:
        logger.error(f"Pulse execution failed / 脉搏检查失败: {e}")
        return f"系统状态读取失败: {str(e)}"


# ============================================================
# Main Entry / Server Start
# ============================================================
def main():
    transport = os.environ.get("OMBRE_TRANSPORT", "sse")
    logger.info(f"Starting Ombre Brain MCP server... Transport: {transport}")

    if transport == "streamable-http":
        _app = mcp.streamable_http_app()
    else:
        _app = mcp.sse_app()

    # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
    # --- 添加 CORS 中间件，让远程客户端（Cloudflare Tunnel / ngrok）能正常连接 ---
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    logger.info("CORS middleware enabled for remote transport / 已启用 CORS 中间件")
    
    # --- Background Keepalive (防止云端实例断连休眠) ---
    async def _keepalive_loop():
        await asyncio.sleep(10)
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

    # 启动 Uvicorn 服务
    uvicorn.run(_app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

if __name__ == "__main__":
    main()
