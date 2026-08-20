# ============================================================
# Recall latency caches — E-axis grouping cache + async dehydration backfill
# 召回性能缓存：E 轴分组小抽屉 + 脱水后台补写
#
# Contract under test (2026-08-20, 朝灯裁定「E 轴一根毫毛不动」):
#   1. E-axis projection behaviour is byte-identical; only the data fetch
#      is cached.  Any bucket write invalidates the cache naturally.
#   2. Recall beats never wait on a live LLM dehydration; the summary is
#      computed off-path and lands in the frontmatter cache for next beat.
# ============================================================

import asyncio
import hashlib

import pytest

import server
from e_axis_recall import EAxisRecallConfig, group_primary_authored_buckets


# ---------- list_all_snapshot_token ----------

def test_snapshot_token_stable_until_write(bucket_mgr):
    async def scenario():
        t1 = await bucket_mgr.list_all_snapshot_token()
        t2 = await bucket_mgr.list_all_snapshot_token()
        assert t1 == t2
        await bucket_mgr.create(content="new bucket flips the tree snapshot", tags=["测试"])
        t3 = await bucket_mgr.list_all_snapshot_token()
        assert t3 != t1
    asyncio.run(scenario())


def test_snapshot_token_matches_list_all_key_semantics(bucket_mgr):
    async def scenario():
        plain = await bucket_mgr.list_all_snapshot_token()
        with_archive = await bucket_mgr.list_all_snapshot_token(include_archive=True)
        assert plain[0] != with_archive[0]
    asyncio.run(scenario())


# ---------- _e_axis_rows_cached ----------

class _CountingMgr:
    """list_all counter with a real-enough snapshot token."""

    def __init__(self):
        self.list_all_calls = 0
        self.token = ("key", ("snap", 1))
        self.buckets = []

    async def list_all_snapshot_token(self, include_archive=False, include_nsfw=None):
        return self.token

    async def list_all(self, include_archive=False, include_nsfw=None):
        self.list_all_calls += 1
        return list(self.buckets)


@pytest.fixture
def e_cfg():
    return EAxisRecallConfig(enabled=True, min_confidence=0.3)


def _reset_e_cache():
    server._E_AXIS_ROWS_CACHE["token"] = None
    server._E_AXIS_ROWS_CACHE["cfg"] = None
    server._E_AXIS_ROWS_CACHE["rows"] = {}


def test_e_rows_cache_skips_list_all_when_snapshot_unchanged(monkeypatch, e_cfg):
    mgr = _CountingMgr()
    monkeypatch.setattr(server, "bucket_mgr", mgr)
    _reset_e_cache()

    async def scenario():
        r1 = await server._e_axis_rows_cached(e_cfg)
        r2 = await server._e_axis_rows_cached(e_cfg)
        assert mgr.list_all_calls == 1
        assert r1 == r2 == group_primary_authored_buckets([], e_cfg)
    asyncio.run(scenario())
    _reset_e_cache()


def test_e_rows_cache_invalidates_on_new_snapshot(monkeypatch, e_cfg):
    mgr = _CountingMgr()
    monkeypatch.setattr(server, "bucket_mgr", mgr)
    _reset_e_cache()

    async def scenario():
        await server._e_axis_rows_cached(e_cfg)
        mgr.token = ("key", ("snap", 2))  # a bucket write moved the tree
        await server._e_axis_rows_cached(e_cfg)
        assert mgr.list_all_calls == 2
    asyncio.run(scenario())
    _reset_e_cache()


def test_e_rows_cache_invalidates_on_config_change(monkeypatch, e_cfg):
    mgr = _CountingMgr()
    monkeypatch.setattr(server, "bucket_mgr", mgr)
    _reset_e_cache()

    async def scenario():
        await server._e_axis_rows_cached(e_cfg)
        stricter = EAxisRecallConfig(enabled=True, min_confidence=0.9)
        await server._e_axis_rows_cached(stricter)
        assert mgr.list_all_calls == 2
    asyncio.run(scenario())
    _reset_e_cache()


# ---------- async dehydration fallback ----------

class _SlowDehydrator:
    """Records calls; would be the multi-second LLM path in production."""

    def __init__(self):
        self.calls = 0

    async def dehydrate_with_source(self, content, key, write_cache=False):
        self.calls += 1
        return ("LLM 摘要：" + content[:20], "computed")

    def format_dehydration_summary(self, summary, metadata):
        return summary


class _WriterMgr:
    def __init__(self):
        self.writes = []

    async def cache_recall_dehydration(self, bucket_id, *, expected_content_hash, summary):
        self.writes.append((bucket_id, expected_content_hash, summary))
        return True


def _fresh_bucket(body: str) -> dict:
    return {"id": "b-async-1", "content": body, "metadata": {}}


def test_async_fallback_returns_immediately_and_backfills(monkeypatch):
    dehy = _SlowDehydrator()
    mgr = _WriterMgr()
    monkeypatch.setattr(server, "dehydrator", dehy)
    monkeypatch.setattr(server, "bucket_mgr", mgr)
    monkeypatch.setattr(server, "config", {"dehydration": {}})
    monkeypatch.delenv("OMBRE_RECALL_DEHYDRATE_ASYNC", raising=False)
    server._DEHYDRATE_BACKFILL_PENDING.clear()

    body = "这是一个还没有脱水缓存的长桶正文 " * 30

    async def scenario():
        summary = await server._dehydrate_for_recall(
            body,
            {},
            bucket=_fresh_bucket(body),
            allow_async_fallback=True,
        )
        # Immediate degraded summary, no synchronous LLM call.
        assert dehy.calls == 0
        assert summary.startswith("这是一个还没有脱水缓存的长桶正文")
        assert len(summary) <= 301
        # Background task computes and persists the real summary.
        await asyncio.sleep(0)
        for _ in range(20):
            if mgr.writes:
                break
            await asyncio.sleep(0.01)
        assert dehy.calls == 1
        assert len(mgr.writes) == 1
        bucket_id, content_hash, written = mgr.writes[0]
        assert bucket_id == "b-async-1"
        assert content_hash == hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert written.startswith("LLM 摘要：")
        assert not server._DEHYDRATE_BACKFILL_PENDING
    asyncio.run(scenario())


def test_async_fallback_disabled_keeps_synchronous_path(monkeypatch):
    dehy = _SlowDehydrator()
    mgr = _WriterMgr()
    monkeypatch.setattr(server, "dehydrator", dehy)
    monkeypatch.setattr(server, "bucket_mgr", mgr)
    monkeypatch.setattr(server, "config", {"dehydration": {}})
    monkeypatch.setenv("OMBRE_RECALL_DEHYDRATE_ASYNC", "0")
    server._DEHYDRATE_BACKFILL_PENDING.clear()

    body = "开关关闭时保持原同步行为的桶正文"

    async def scenario():
        summary = await server._dehydrate_for_recall(
            body,
            {},
            bucket=_fresh_bucket(body),
            allow_async_fallback=True,
        )
        assert dehy.calls == 1  # legacy synchronous computation
        assert summary.startswith("LLM 摘要：")
    asyncio.run(scenario())


def test_recall_callers_without_flag_never_degrade(monkeypatch):
    """briefing/breath 路径不传 allow_async_fallback，永远拿完整摘要。"""
    dehy = _SlowDehydrator()
    mgr = _WriterMgr()
    monkeypatch.setattr(server, "dehydrator", dehy)
    monkeypatch.setattr(server, "bucket_mgr", mgr)
    monkeypatch.setattr(server, "config", {"dehydration": {}})
    monkeypatch.delenv("OMBRE_RECALL_DEHYDRATE_ASYNC", raising=False)

    body = "简报路径的桶正文，不允许降级"

    async def scenario():
        summary = await server._dehydrate_for_recall(body, {}, bucket=_fresh_bucket(body))
        assert dehy.calls == 1
        assert summary.startswith("LLM 摘要：")
    asyncio.run(scenario())
