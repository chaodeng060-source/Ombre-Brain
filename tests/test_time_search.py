# ============================================================
# Test: Time-range filter — parse_relative_time + search created range
# 测试：时间范围检索 — parse_relative_time + search 的 created 范围过滤
#
# Pure unit + bucket_manager integration. No LLM/embedding needed.
# 纯单元 + bucket_manager 集成,不需要 LLM/embedding。
# ============================================================

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from utils import parse_relative_time
from bucket_manager import _bucket_in_time_range


# ============================================================
# parse_relative_time — unit tests
# ============================================================
class TestParseRelativeTime:
    def test_none_input(self):
        assert parse_relative_time(None) is None

    def test_empty_string(self):
        assert parse_relative_time("") is None
        assert parse_relative_time("   ") is None

    def test_non_string(self):
        assert parse_relative_time(123) is None

    def test_garbage(self):
        assert parse_relative_time("not a time") is None
        assert parse_relative_time("上周") is None  # NLU intentionally not supported

    def test_now(self):
        ref = datetime(2026, 5, 11, 22, 30, 0)
        assert parse_relative_time("now", reference=ref) == ref
        assert parse_relative_time("NOW", reference=ref) == ref

    def test_today(self):
        ref = datetime(2026, 5, 11, 22, 30, 45)
        result = parse_relative_time("today", reference=ref)
        assert result == datetime(2026, 5, 11, 0, 0, 0, 0)

    def test_yesterday(self):
        ref = datetime(2026, 5, 11, 22, 30, 45)
        result = parse_relative_time("yesterday", reference=ref)
        assert result == datetime(2026, 5, 10, 0, 0, 0, 0)

    def test_relative_negative_days(self):
        ref = datetime(2026, 5, 11, 22, 30, 0)
        result = parse_relative_time("-7d", reference=ref)
        assert result == ref - timedelta(days=7)

    def test_relative_negative_hours(self):
        ref = datetime(2026, 5, 11, 22, 30, 0)
        result = parse_relative_time("-3h", reference=ref)
        assert result == ref - timedelta(hours=3)

    def test_relative_negative_minutes(self):
        ref = datetime(2026, 5, 11, 22, 30, 0)
        result = parse_relative_time("-30m", reference=ref)
        assert result == ref - timedelta(minutes=30)

    def test_relative_positive(self):
        ref = datetime(2026, 5, 11, 22, 30, 0)
        assert parse_relative_time("+1d", reference=ref) == ref + timedelta(days=1)
        assert parse_relative_time("1d", reference=ref) == ref + timedelta(days=1)

    def test_relative_garbage_unit(self):
        # "5y" not supported (only d/h/m); should fall through and fail
        assert parse_relative_time("5y") is None

    def test_iso_date_only(self):
        result = parse_relative_time("2026-05-01")
        assert result == datetime(2026, 5, 1, 0, 0, 0)

    def test_iso_datetime(self):
        result = parse_relative_time("2026-05-01T12:34:56")
        assert result == datetime(2026, 5, 1, 12, 34, 56)

    def test_iso_invalid(self):
        # 13月不存在 / month 13 invalid
        assert parse_relative_time("2026-13-01") is None


# ============================================================
# _bucket_in_time_range — helper unit
# ============================================================
def _make_bucket(created_str: str) -> dict:
    return {"id": "x", "metadata": {"created": created_str}}


class TestBucketInTimeRange:
    def test_no_bounds_keeps_bucket(self):
        b = _make_bucket("2026-05-01T10:00:00")
        assert _bucket_in_time_range(b) is True

    def test_within_range(self):
        b = _make_bucket("2026-05-05T10:00:00")
        after = datetime(2026, 5, 1)
        before = datetime(2026, 5, 10)
        assert _bucket_in_time_range(b, after, before) is True

    def test_before_after_bound(self):
        b = _make_bucket("2026-04-25T10:00:00")
        after = datetime(2026, 5, 1)
        assert _bucket_in_time_range(b, after=after) is False

    def test_after_before_bound(self):
        b = _make_bucket("2026-05-20T10:00:00")
        before = datetime(2026, 5, 10)
        assert _bucket_in_time_range(b, before=before) is False

    def test_open_lower_bound(self):
        b = _make_bucket("2025-01-01T00:00:00")
        before = datetime(2026, 5, 10)
        assert _bucket_in_time_range(b, before=before) is True

    def test_open_upper_bound(self):
        b = _make_bucket("2027-01-01T00:00:00")
        after = datetime(2026, 5, 1)
        assert _bucket_in_time_range(b, after=after) is True

    def test_unparseable_kept_conservatively(self):
        b = _make_bucket("not a timestamp")
        after = datetime(2026, 5, 1)
        before = datetime(2026, 5, 10)
        # Unparseable timestamps are kept rather than dropped — losing visibility
        # on broken metadata is worse than including a noisy bucket.
        # 无法解析的时间戳保守保留——丢一个噪音桶比让损坏元数据消失更好。
        assert _bucket_in_time_range(b, after, before) is True

    def test_falls_back_to_last_active(self):
        b = {"id": "x", "metadata": {"last_active": "2026-05-05T10:00:00"}}
        after = datetime(2026, 5, 1)
        before = datetime(2026, 5, 10)
        assert _bucket_in_time_range(b, after, before) is True


# ============================================================
# bucket_manager.search — integration with created range
# ============================================================
@pytest_asyncio.fixture
async def time_seeded_env(test_config, bucket_mgr):
    """Create three buckets with controlled `created` timestamps."""
    import frontmatter as fm

    seeds = [
        ("old", "2026-04-15T10:00:00", "asyncio Python event loop"),
        ("mid", "2026-05-05T10:00:00", "asyncio Python event loop"),
        ("new", "2026-05-20T10:00:00", "asyncio Python event loop"),
    ]
    ids = {}
    for label, ts, content in seeds:
        bid = await bucket_mgr.create(
            content=content,
            tags=["asyncio", "python"],
            importance=5,
            domain=["编程"],
        )
        fpath = bucket_mgr._find_bucket_file(bid)
        post = fm.load(fpath)
        post["created"] = ts
        post["last_active"] = ts
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(fm.dumps(post))
        ids[label] = bid
    return bucket_mgr, ids


class TestSearchTimeRange:
    @pytest.mark.asyncio
    async def test_no_time_filter_returns_all(self, time_seeded_env):
        bm, ids = time_seeded_env
        results = await bm.search("asyncio", limit=10)
        result_ids = {r["id"] for r in results}
        assert ids["old"] in result_ids
        assert ids["mid"] in result_ids
        assert ids["new"] in result_ids

    @pytest.mark.asyncio
    async def test_created_after_filter(self, time_seeded_env):
        bm, ids = time_seeded_env
        results = await bm.search(
            "asyncio",
            limit=10,
            created_after=datetime(2026, 5, 1),
        )
        result_ids = {r["id"] for r in results}
        assert ids["old"] not in result_ids
        assert ids["mid"] in result_ids
        assert ids["new"] in result_ids

    @pytest.mark.asyncio
    async def test_created_before_filter(self, time_seeded_env):
        bm, ids = time_seeded_env
        results = await bm.search(
            "asyncio",
            limit=10,
            created_before=datetime(2026, 5, 10),
        )
        result_ids = {r["id"] for r in results}
        assert ids["old"] in result_ids
        assert ids["mid"] in result_ids
        assert ids["new"] not in result_ids

    @pytest.mark.asyncio
    async def test_both_bounds_inclusive_window(self, time_seeded_env):
        bm, ids = time_seeded_env
        results = await bm.search(
            "asyncio",
            limit=10,
            created_after=datetime(2026, 5, 1),
            created_before=datetime(2026, 5, 10),
        )
        result_ids = {r["id"] for r in results}
        assert ids["old"] not in result_ids
        assert ids["mid"] in result_ids
        assert ids["new"] not in result_ids

    @pytest.mark.asyncio
    async def test_empty_window_returns_empty(self, time_seeded_env):
        bm, ids = time_seeded_env
        results = await bm.search(
            "asyncio",
            limit=10,
            created_after=datetime(2026, 6, 1),
            created_before=datetime(2026, 6, 10),
        )
        assert results == []
