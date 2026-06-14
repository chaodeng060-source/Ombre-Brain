#!/usr/bin/env python3
"""
Backfill sense tags from structured sensory intensities.
给「有 sensory.* 强度、却没对应 sense 标签」的存量桶补 sense 通道（闭环另一半，小卷 #1）。

辣椒酱桶有 sensory.spicy:0.9 但没 sense:[味觉]：它能被读到点燃身体状态，却不会因
「味觉/辣/入口」类 query 上浮。这里按 senses_from_sensory 把结构化强度映射成 sense 通道补齐，
让「读到会有感觉」和「相关感官线索把它读出来」真正闭环。

幂等：只给「派生出的 sense 通道尚未在 metadata.sense 里」的桶追加；不删已有 sense、不改正文、
不动向量。默认 dry-run（只打印将要怎么改）；真正写入要带 --go。
注意：写入经 bucket_mgr.update，会顺带把这些桶的 last_active bump 到当下（感官桶数量很少，可忽略；
recency 闸现按 created 判龄，不会因此误判「最近活跃」）。

Usage:
    OMBRE_BUCKETS_DIR=/data python backfill_senses.py            # dry-run（默认，只看）
    OMBRE_BUCKETS_DIR=/data python backfill_senses.py --go       # 实际写入
"""

import asyncio
import argparse
import sys

sys.path.insert(0, ".")
from utils import load_config
from bucket_manager import BucketManager
from sensory_engine import senses_from_sensory
from sense_tagger import normalize_sense_field, union_senses


async def backfill(go: bool = False):
    config = load_config()
    bucket_mgr = BucketManager(config)

    all_buckets = await bucket_mgr.list_all(include_archive=True)
    print(f"Total buckets: {len(all_buckets)}")

    todo = []
    for b in all_buckets:
        derived = senses_from_sensory(b)
        if not derived:
            continue
        existing = normalize_sense_field(b["metadata"].get("sense"))
        merged = union_senses(existing, derived)
        if merged != existing:
            todo.append((b, existing, merged))

    print(f"Buckets needing sense backfill: {len(todo)}")
    for b, existing, merged in todo:
        name = b["metadata"].get("name", b["id"])
        added = [s for s in merged if s not in existing]
        print(f"  {b['id'][:12]} ({name[:28]}): {existing or '[]'} -> {merged}  (+{added})")

    if not go:
        print("\n(dry-run) 没有写入。确认无误后加 --go 实际写入。")
        return

    ok = 0
    fail = 0
    for b, _existing, merged in todo:
        try:
            updated = await bucket_mgr.update(b["id"], sense=merged)
            if updated:
                ok += 1
            else:
                fail += 1
                print(f"  FAIL (update returned False): {b['id']}")
        except Exception as e:
            fail += 1
            print(f"  ERROR: {b['id']}: {e}")
    print(f"\n=== Done: {ok} updated, {fail} failed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true", help="actually write (default: dry-run)")
    args = parser.parse_args()
    asyncio.run(backfill(go=args.go))
