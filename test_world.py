"""world 字段端到端测试。不依赖 LLM API。
覆盖：
  - bucket_manager.create 写 world 字段
  - bucket_manager.search world_filter 三种语义（None/空/单值/多值）
  - utils.world_matches 边界
  - server._resolve_world_filter 推断
  - 持久化指针 save/load
"""
import asyncio
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_config, world_matches, UNIVERSAL_WORLD,
    save_current_world, _load_runtime_current_world,
)
from bucket_manager import BucketManager


@pytest.fixture
def buckets_dir(tmp_path):
    root = tmp_path / "buckets"
    for sub in ["permanent", "dynamic", "archive", "feel"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return str(root)


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


def test_world_matches():
    print("=== 1. world_matches 边界 ===")
    # 通用桶在任何 filter 下通过
    assert world_matches("通用", set()) is True
    assert world_matches("通用", {"当前世界"}) is True
    assert world_matches("通用", {""}) is True
    # 日常桶 (world="") 只在 filter 含 "" 时通过
    assert world_matches("", {""}) is True
    assert world_matches("", {"当前世界"}) is False
    assert world_matches("", set()) is False
    # 命名世界精确匹配
    assert world_matches("当前世界", {"当前世界"}) is True
    assert world_matches("当前世界", {"旧世界"}) is False
    assert world_matches("当前世界", {"当前世界", "旧世界"}) is True
    # 空白容错
    assert world_matches("  当前世界  ", {"当前世界"}) is True
    assert world_matches(None, {""}) is True
    print("  [OK]")


def test_resolve_world_filter():
    print("=== 2. _resolve_world_filter 推断 ===")
    from server import _resolve_world_filter as rwf
    # 显式 all → None
    assert rwf("all", "当前世界") is None
    assert rwf("ALL", "当前世界") is None
    # 显式 world → 列表
    assert rwf("旧世界", "当前世界") == ["旧世界"]
    assert rwf("旧世界,通用", "") == ["旧世界", "通用"]
    # 留空 → current_world
    assert rwf("", "当前世界") == ["当前世界"]
    assert rwf("", "") == [""]   # 日常模式
    print("  [OK]")


@pytest.mark.asyncio
async def test_search_world_filter(buckets_dir):
    print("=== 3. bucket_manager.search world_filter ===")
    config = {
        "buckets_dir": buckets_dir,
        "matching": {"fuzzy_threshold": 50, "max_results": 50},
        "scoring_weights": {
            "topic_relevance": 4.0, "emotion_resonance": 2.0,
            "time_proximity": 2.5, "importance": 1.0,
        },
        "wikilink": {"enabled": False},
    }
    bm = BucketManager(config)

    # 建 4 个桶：日常（无 world）、当前世界、旧世界、通用
    bid_daily = await bm.create(
        content="今天 P 喵喵叫了三声", domain=["日常"], tags=["猫"], importance=5,
    )
    bid_now = await bm.create(
        content="谢长夜在城北街角点了一盏灯", domain=["谢长夜"], tags=["谢长夜", "灯"],
        importance=5, world="当前世界",
    )
    bid_old = await bm.create(
        content="谢长夜旧世界里那把碎了的剑", domain=["谢长夜"], tags=["谢长夜", "剑"],
        importance=5, world="旧世界",
    )
    bid_universal = await bm.create(
        content="谢长夜的人设：白发红眼, 沉默寡言", domain=["谢长夜"], tags=["谢长夜", "人设"],
        importance=5, world="通用",
    )

    # 用一个能命中所有桶 name/tag 的 query 跨集合检查 filter
    q = "谢长夜"

    # filter=None → 全部带"谢长夜"标签的命中（不过滤 world）
    r = await bm.search(q, limit=20, world_filter=None)
    ids = {b["id"] for b in r}
    assert bid_now in ids and bid_old in ids and bid_universal in ids, f"None: {ids}"
    print(f"  filter=None → {len(ids)} 命中 (含三个 world 桶) [OK]")

    # filter=[""] → 日常模式：只出 world="" 桶 + 通用桶
    r = await bm.search(q, limit=20, world_filter=[""])
    ids = {b["id"] for b in r}
    assert bid_now not in ids and bid_old not in ids, f"日常: {ids}"
    assert bid_universal in ids, f"通用桶应跟着出: {ids}"
    print(f"  filter=[\"\"] (日常) → 通用桶在,角色世界桶不在 [OK]")

    # filter=["当前世界"] → 当前世界 + 通用
    r = await bm.search(q, limit=20, world_filter=["当前世界"])
    ids = {b["id"] for b in r}
    assert bid_now in ids, f"当前: {ids}"
    assert bid_old not in ids, f"旧世界应被过滤: {ids}"
    assert bid_universal in ids, f"通用应跟着出: {ids}"
    print(f"  filter=[\"当前世界\"] → 当前+通用,旧世界被过滤 [OK]")

    # filter=["旧世界"] → 旧世界 + 通用
    r = await bm.search(q, limit=20, world_filter=["旧世界"])
    ids = {b["id"] for b in r}
    assert bid_old in ids and bid_universal in ids
    assert bid_now not in ids
    print(f"  filter=[\"旧世界\"] → 旧+通用,当前世界被过滤 [OK]")

    # filter=["当前世界","旧世界"] → 多值并集 + 通用
    r = await bm.search(q, limit=20, world_filter=["当前世界", "旧世界"])
    ids = {b["id"] for b in r}
    assert {bid_now, bid_old, bid_universal}.issubset(ids)
    print(f"  filter=[当前,旧] → 三个角色桶都在 [OK]")

    # 清理
    for bid in [bid_daily, bid_now, bid_old, bid_universal]:
        await bm.delete(bid)


def test_runtime_sidecar(tmp_dir):
    print("=== 4. 运行时 sidecar 持久化 ===")
    # 初始没文件
    assert _load_runtime_current_world(tmp_dir) is None
    # 写入
    save_current_world(tmp_dir, "当前世界")
    assert _load_runtime_current_world(tmp_dir) == "当前世界"
    # 切回日常
    save_current_world(tmp_dir, "")
    assert _load_runtime_current_world(tmp_dir) == ""
    # 切到另一个
    save_current_world(tmp_dir, "旧世界")
    assert _load_runtime_current_world(tmp_dir) == "旧世界"
    print("  [OK]")


async def main():
    test_world_matches()
    test_resolve_world_filter()

    tmp_dir = tempfile.mkdtemp(prefix="ombre_world_test_")
    try:
        for sub in ["permanent", "dynamic", "archive", "feel"]:
            os.makedirs(os.path.join(tmp_dir, sub), exist_ok=True)
        await test_search_world_filter(tmp_dir)
        test_runtime_sidecar(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 40)
    print("world 字段测试全部通过 [OK]")


if __name__ == "__main__":
    asyncio.run(main())
