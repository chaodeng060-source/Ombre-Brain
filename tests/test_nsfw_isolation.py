# ============================================================
# 色色单独文件夹：涩涩独立目录 + 日常默认不扫 + 切进涩涩 world 才加载
# ============================================================
import os
import pytest


def _write_bucket(path, bid, name, domain="涩涩", world="涩涩"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"---\nid: {bid}\nname: {name}\ntype: dynamic\n"
            f"domain:\n  - {domain}\nworld: {world}\nimportance: 8\n---\n正文内容_{bid}"
        )


@pytest.mark.asyncio
async def test_nsfw_excluded_by_default(bucket_mgr):
    os.makedirs(bucket_mgr.nsfw_dir, exist_ok=True)
    _write_bucket(os.path.join(bucket_mgr.nsfw_dir, "nsfw1.md"), "nsfw1", "涩涩内容1")

    ids = {b["id"] for b in await bucket_mgr.list_all()}
    assert "nsfw1" not in ids            # 默认(nsfw_active=False)：日常不碰


@pytest.mark.asyncio
async def test_nsfw_visible_when_explicit(bucket_mgr):
    os.makedirs(bucket_mgr.nsfw_dir, exist_ok=True)
    _write_bucket(os.path.join(bucket_mgr.nsfw_dir, "nsfw2.md"), "nsfw2", "涩涩内容2")

    ids = {b["id"] for b in await bucket_mgr.list_all(include_nsfw=True)}
    assert "nsfw2" in ids                # 显式 include_nsfw=True(dashboard 管理)：看得到


@pytest.mark.asyncio
async def test_nsfw_follows_active_flag(bucket_mgr):
    os.makedirs(bucket_mgr.nsfw_dir, exist_ok=True)
    _write_bucket(os.path.join(bucket_mgr.nsfw_dir, "nsfw3.md"), "nsfw3", "涩涩内容3")

    bucket_mgr.nsfw_active = True        # 切进涩涩 world
    assert "nsfw3" in {b["id"] for b in await bucket_mgr.list_all()}

    bucket_mgr.nsfw_active = False       # 切出
    assert "nsfw3" not in {b["id"] for b in await bucket_mgr.list_all()}


@pytest.mark.asyncio
async def test_normal_buckets_unaffected(bucket_mgr):
    # 日常桶照常出现，不受涩涩隔离影响
    os.makedirs(bucket_mgr.dynamic_dir, exist_ok=True)
    sub = os.path.join(bucket_mgr.dynamic_dir, "日常")
    os.makedirs(sub, exist_ok=True)
    _write_bucket(os.path.join(sub, "daily1.md"), "daily1", "日常内容", domain="日常", world="")

    ids = {b["id"] for b in await bucket_mgr.list_all()}
    assert "daily1" in ids


@pytest.mark.asyncio
async def test_nsfw_get_by_id_always_works(bucket_mgr):
    # 隔离的是召回；按 id 精确取（get/inspect/edit/建边）必须永远找得到，即便日常模式
    os.makedirs(bucket_mgr.nsfw_dir, exist_ok=True)
    _write_bucket(os.path.join(bucket_mgr.nsfw_dir, "nsfw_get.md"), "nsfw_get", "涩涩精确取")

    assert bucket_mgr.nsfw_active is False           # 日常模式
    assert "nsfw_get" not in {b["id"] for b in await bucket_mgr.list_all()}  # 召回看不到
    b = await bucket_mgr.get("nsfw_get")
    assert b is not None and b["id"] == "nsfw_get"   # 但精确取得到
