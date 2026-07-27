#!/usr/bin/env python3
"""
Ombre Brain 手动记忆写入工具
用途：在 Copilot 端直接写入记忆文件，绕过 MCP 和 API 调用
用法：
  python3 write_memory.py --name "记忆名" --content "内容" --domain "情感" --tags "标签1,标签2"
  或交互模式：python3 write_memory.py
"""

import os
import argparse
import asyncio

from bucket_manager import BucketManager
from utils import load_config

VAULT_ROOT = os.path.expanduser("~/Documents/Obsidian Vault/Ombre Brain")


def write_memory(
    name: str,
    content: str,
    domain: list[str],
    tags: list[str],
    importance: int = 7,
    valence: float = 0.5,
    arousal: float = 0.3,
    event_at: str = None,
):
    config = load_config()
    config["buckets_dir"] = os.environ.get("OMBRE_BUCKETS_DIR", VAULT_ROOT)
    manager = BucketManager(config)
    mid = asyncio.run(
        manager.create(
            content=content,
            name=name,
            domain=domain,
            tags=tags,
            importance=importance,
            valence=valence,
            arousal=arousal,
            event_at=event_at,
            actor="cli:write_memory",
        )
    )
    path = manager._find_bucket_file(mid)
    print(f"✓ 已写入: {path}")
    print(f"  ID: {mid} | 名称: {name}")
    return mid


def interactive():
    print("=== Ombre Brain 手动写入 ===")
    name = input("记忆名称: ").strip()
    content = input("内容: ").strip()
    domain = [d.strip() for d in input("主题域(逗号分隔): ").split(",") if d.strip()]
    tags = [t.strip() for t in input("标签(逗号分隔): ").split(",") if t.strip()]
    importance = int(input("重要性(1-10, 默认7): ").strip() or "7")
    valence = float(input("效价(0-1, 默认0.5): ").strip() or "0.5")
    arousal = float(input("唤醒(0-1, 默认0.3): ").strip() or "0.3")
    write_memory(name, content, domain, tags, importance, valence, arousal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手动写入 Ombre Brain 记忆")
    parser.add_argument("--name", help="记忆名称")
    parser.add_argument("--content", help="记忆内容")
    parser.add_argument("--domain", help="主题域,逗号分隔")
    parser.add_argument("--tags", help="标签,逗号分隔")
    parser.add_argument("--importance", type=int, default=7)
    parser.add_argument("--valence", type=float, default=0.5)
    parser.add_argument("--arousal", type=float, default=0.3)
    parser.add_argument("--event-at", help="事件发生时间（ISO 8601）；留空则按写入时刻并标低置信度")
    args = parser.parse_args()

    if args.name and args.content and args.domain:
        write_memory(
            name=args.name,
            content=args.content,
            domain=[d.strip() for d in args.domain.split(",")],
            tags=[t.strip() for t in (args.tags or "").split(",") if t.strip()],
            importance=args.importance,
            valence=args.valence,
            arousal=args.arousal,
            event_at=args.event_at,
        )
    else:
        interactive()
