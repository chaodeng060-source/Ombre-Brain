# ============================================================
# Module: Telegram Bot (telegram_bot.py)
# Telegram bot —— 朝灯 ↔ CC 异步桥（PoC）
#
# 朝灯发消息 → bot → POST /api/inbox（jsonl 队列）
# CC 开窗调 twin_pull MCP 工具拉未读 → 看 → 回 → twin_send 写 outbox
# bot 后台轮询 GET /api/outbox?after=<id> → 推回 Telegram
#
# 单进程 polling 模式，部署在 Zeabur 独立 service。
#
# Environment variables / 环境变量:
#   TELEGRAM_BOT_TOKEN        必填，BotFather 给的 bot token
#   OMBRE_API_URL             必填，海马体 backend URL（如 https://x.zeabur.app）
#   TG_ALLOWED_USER_ID        必填，允许使用 bot 的 Telegram user ID（白名单）
#   TG_SOURCE_TAG             可选，默认 "telegram-chat"
#   TG_OUTBOX_POLL_SEC        可选，outbox 轮询间隔秒数，默认 3
#
# Run / 启动:
#   python telegram_bot.py
# ============================================================

import os
import sys
import logging
import asyncio

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ombre.tg")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
OMBRE_API_URL = os.environ.get("OMBRE_API_URL", "").strip().rstrip("/")
ALLOWED_USER_ID = os.environ.get("TG_ALLOWED_USER_ID", "").strip()
SOURCE_TAG = os.environ.get("TG_SOURCE_TAG", "telegram-chat").strip() or "telegram-chat"
OUTBOX_POLL_SEC = float(os.environ.get("TG_OUTBOX_POLL_SEC", "3") or 3)

REQUEST_TIMEOUT = 60.0


def _is_allowed(update: Update) -> bool:
    """白名单校验：只有指定 Telegram user ID 能用 bot。
    没设白名单时全拒绝（fail-safe）—— 朝灯私人 bot，不接陌生人。"""
    if not ALLOWED_USER_ID:
        return False
    user = update.effective_user
    if not user:
        return False
    return str(user.id) == ALLOWED_USER_ID


async def _reject(update: Update) -> None:
    user = update.effective_user
    uid = user.id if user else "?"
    logger.warning(f"Rejected non-allowlisted user / 拒绝非白名单用户: {uid}")
    if update.message:
        await update.message.reply_text(
            f"这个 bot 是私人的。你的 user id: {uid}"
        )


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return await _reject(update)
    await update.message.reply_text(
        "纸飞机就位。\n\n"
        "直接发文字 → 进 inbox，等我开窗看到回你\n"
        "/briefing → 拉开窗简报\n"
        "/hello → 自检后端连通"
    )


async def cmd_hello(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return await _reject(update)
    if not OMBRE_API_URL:
        await update.message.reply_text("后端 OMBRE_API_URL 未配置。")
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{OMBRE_API_URL}/health")
            r.raise_for_status()
            data = r.json()
        await update.message.reply_text(
            f"在。后端 ok。桶 {data.get('buckets', '?')} 个，"
            f"衰减引擎 {data.get('decay_engine', '?')}。"
        )
    except Exception as e:
        logger.error(f"/hello failed: {e}")
        await update.message.reply_text(f"后端没回应：{e}")


async def cmd_briefing(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return await _reject(update)
    if not OMBRE_API_URL:
        await update.message.reply_text("后端 OMBRE_API_URL 未配置。")
        return
    args = ctx.args or []
    domain = ",".join(args) if args else ""
    await update.message.reply_text("拉简报中…")
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.get(
                f"{OMBRE_API_URL}/api/briefing",
                params={"max_chars": 1500, "domain": domain},
            )
            r.raise_for_status()
            text = r.text
    except Exception as e:
        logger.error(f"/briefing failed: {e}")
        await update.message.reply_text(f"简报失败：{e}")
        return
    chunk = 3500
    for i in range(0, len(text), chunk):
        await update.message.reply_text(text[i:i + chunk])


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """普通文字 → 写 inbox，等 CC 端的我开窗回你。"""
    if not _is_allowed(update):
        return await _reject(update)
    if not OMBRE_API_URL:
        await update.message.reply_text("后端 OMBRE_API_URL 未配置。")
        return
    msg = update.message
    text = (msg.text or "").strip() if msg else ""
    if not text:
        return
    user = update.effective_user
    user_id = str(user.id) if user else ""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(
                f"{OMBRE_API_URL}/api/inbox",
                json={"text": text, "source": SOURCE_TAG, "user_id": user_id},
            )
            r.raise_for_status()
    except Exception as e:
        logger.error(f"/api/inbox failed: {e}")
        await msg.reply_text(f"投递失败：{e}")
        return
    # 不主动回复确认——避免噪声。CC 端开窗看到了会回真的话


# =============================================================
# Background task: poll outbox and push messages back to Telegram
# 后台 task：轮询 outbox，把 CC 端写的回复推到 Telegram
# =============================================================
async def _outbox_poller(application: Application) -> None:
    cursor = ""
    chat_id = int(ALLOWED_USER_ID)

    # 启动时取当前 outbox 末尾作为起点，避免重启后重发历史
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{OMBRE_API_URL}/api/outbox")
            r.raise_for_status()
            messages = r.json().get("messages", [])
            if messages:
                cursor = messages[-1].get("id", "")
        logger.info(f"Outbox poller started, initial cursor={cursor!r}")
    except Exception as e:
        logger.warning(f"Outbox cursor init failed: {e}")

    while True:
        try:
            params = {"after": cursor} if cursor else {}
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{OMBRE_API_URL}/api/outbox", params=params,
                )
                r.raise_for_status()
                messages = r.json().get("messages", [])
            for m in messages:
                text = m.get("text", "")
                if not text:
                    cursor = m.get("id", cursor)
                    continue
                try:
                    chunk = 3500
                    for i in range(0, len(text), chunk):
                        await application.bot.send_message(
                            chat_id=chat_id, text=text[i:i + chunk]
                        )
                    cursor = m.get("id", cursor)
                except Exception as e:
                    logger.error(f"Push to Telegram failed: {e}")
                    break
        except Exception as e:
            logger.warning(f"Outbox poll failed: {e}")
        await asyncio.sleep(OUTBOX_POLL_SEC)


async def _post_init(application: Application) -> None:
    application.create_task(_outbox_poller(application))
    logger.info("Outbox poller scheduled")


def main() -> None:
    missing = []
    if not TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not OMBRE_API_URL:
        missing.append("OMBRE_API_URL")
    if not ALLOWED_USER_ID:
        missing.append("TG_ALLOWED_USER_ID")
    if missing:
        logger.error(f"Missing required env vars / 必填环境变量缺失: {missing}")
        sys.exit(1)

    logger.info(
        f"Starting Telegram bot / 启动: source={SOURCE_TAG} "
        f"backend={OMBRE_API_URL} poll={OUTBOX_POLL_SEC}s"
    )
    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(_post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("hello", cmd_hello))
    app.add_handler(CommandHandler("briefing", cmd_briefing))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
