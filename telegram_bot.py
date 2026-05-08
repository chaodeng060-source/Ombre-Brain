# ============================================================
# Module: Telegram Bot (telegram_bot.py)
# Telegram bot —— 朝灯专属移动入口
#
# Thin frontend：消息进bot → POST /api/hold → 写桶
# 单进程 polling 模式，部署在 Zeabur 独立 service。
# 海马体 backend 通过 OMBRE_API_URL 访问 REST 桥接。
#
# Environment variables / 环境变量:
#   TELEGRAM_BOT_TOKEN        必填，BotFather 给的 bot token
#   OMBRE_API_URL             必填，海马体 backend URL（如 https://x.zeabur.app）
#   TG_ALLOWED_USER_ID        必填，允许使用 bot 的 Telegram user ID（白名单）
#   TG_SOURCE_TAG             可选，默认 "telegram-chat"，会作为标签写入桶
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

REQUEST_TIMEOUT = 60.0


def _is_allowed(update: Update) -> bool:
    """白名单校验：只有指定 Telegram user ID 能用 bot。
    没设白名单时，全拒绝（fail-safe）—— 朝灯私人 bot，不接陌生人。"""
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
            f"这个 bot 是私人的。你的 user id: {uid}（如果你应该有权限，请把这个 id 给主人）"
        )


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return await _reject(update)
    await update.message.reply_text(
        "纸飞机就位。\n\n"
        "直接发文字 → 写桶（自动打标）\n"
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
            f"在。后端ok。桶 {data.get('buckets', '?')} 个，"
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
    # Telegram 单条上限约 4096 字符，分段发
    chunk = 3500
    for i in range(0, len(text), chunk):
        await update.message.reply_text(text[i:i + chunk])


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return await _reject(update)
    if not OMBRE_API_URL:
        await update.message.reply_text("后端 OMBRE_API_URL 未配置。")
        return
    msg = update.message
    text = (msg.text or "").strip() if msg else ""
    if not text:
        return
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(
                f"{OMBRE_API_URL}/api/hold",
                json={"content": text, "source": SOURCE_TAG, "importance": 5},
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.error(f"/api/hold failed: {e}")
        await msg.reply_text(f"写桶失败：{e}")
        return
    await msg.reply_text(data.get("result", "写入成功"))


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
        f"Starting Telegram bot / 启动: source={SOURCE_TAG} backend={OMBRE_API_URL}"
    )
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("hello", cmd_hello))
    app.add_handler(CommandHandler("briefing", cmd_briefing))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
