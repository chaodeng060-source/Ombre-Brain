"""
Tests for _split_recent_by_time_gap — the pure helper that splits a
time-sorted bucket list into "current window" vs "prior windows".

测试 _split_recent_by_time_gap 纯函数:把按 last_active 降序排好的桶拆成
「上一窗口」和「再之前」两组。
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Need to import without triggering full server.py side effects (mcp instance, etc).
# server.py imports cleanly because it only initializes the FastMCP at module load
# but doesn't start it. We pull the helper directly.
# server.py 模块级只初始化不启动 FastMCP,可以安全 import 拿到辅助函数。
import server  # noqa: E402
_split = server._split_recent_by_time_gap  # noqa: E402


def _mk(name: str, ts_iso: str) -> dict:
    """Build a minimal bucket dict for testing."""
    return {
        "id": name,
        "metadata": {"last_active": ts_iso, "name": name},
    }


def _ts(dt: datetime) -> str:
    """ISO format with Z suffix."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------
# Edge cases
# 边界情况
# ---------------------------------------------------------------

def test_empty_list_returns_two_empty():
    """空列表返回两个空列表。"""
    recent, prior = _split([])
    assert recent == []
    assert prior == []


def test_single_bucket_goes_to_recent():
    """单个桶全部归 recent_window。"""
    b = _mk("a", _ts(datetime(2026, 5, 13, 12, 0)))
    recent, prior = _split([b])
    assert recent == [b]
    assert prior == []


def test_missing_timestamps_filtered_out():
    """无效或缺失的时间戳被过滤掉。"""
    b1 = _mk("a", _ts(datetime(2026, 5, 13, 12, 0)))
    b2 = _mk("b", "")  # 空时间戳
    b3 = _mk("c", "not-an-iso-string")  # 解析失败
    recent, prior = _split([b1, b2, b3])
    # 只剩 b1
    assert recent == [b1]
    assert prior == []


# ---------------------------------------------------------------
# Gap detection
# Gap 检测
# ---------------------------------------------------------------

def test_small_gap_keeps_all_in_recent_window():
    """所有桶时间间隔都 < 1 小时,全归 recent_window,prior 为空。"""
    base = datetime(2026, 5, 13, 12, 0)
    buckets = [
        _mk("a", _ts(base)),                          # 12:00
        _mk("b", _ts(base - timedelta(minutes=10))),  # 11:50
        _mk("c", _ts(base - timedelta(minutes=30))),  # 11:30
        _mk("d", _ts(base - timedelta(minutes=55))),  # 11:05
    ]
    recent, prior = _split(buckets)
    assert len(recent) == 4
    assert prior == []


def test_large_gap_splits_into_two_groups():
    """有 >= 1 小时 gap 时按最大 gap 切分两组。"""
    base = datetime(2026, 5, 13, 12, 0)
    buckets = [
        _mk("recent_1", _ts(base)),                          # 12:00
        _mk("recent_2", _ts(base - timedelta(minutes=10))),  # 11:50
        # gap: 5 小时
        _mk("prior_1", _ts(base - timedelta(hours=5))),      # 07:00
        _mk("prior_2", _ts(base - timedelta(hours=5, minutes=20))),  # 06:40
    ]
    recent, prior = _split(buckets)
    assert [b["id"] for b in recent] == ["recent_1", "recent_2"]
    assert [b["id"] for b in prior] == ["prior_1", "prior_2"]


def test_gap_exactly_at_threshold_counts_as_boundary():
    """正好 1 小时的 gap 算窗口边界(>=)。"""
    base = datetime(2026, 5, 13, 12, 0)
    buckets = [
        _mk("recent", _ts(base)),
        _mk("prior", _ts(base - timedelta(hours=1))),  # 正好 1h
    ]
    recent, prior = _split(buckets)
    assert [b["id"] for b in recent] == ["recent"]
    assert [b["id"] for b in prior] == ["prior"]


def test_gap_just_below_threshold_no_split():
    """59 分钟的 gap 不算窗口边界,全归 recent。"""
    base = datetime(2026, 5, 13, 12, 0)
    buckets = [
        _mk("a", _ts(base)),
        _mk("b", _ts(base - timedelta(minutes=59))),
    ]
    recent, prior = _split(buckets)
    assert len(recent) == 2
    assert prior == []


def test_largest_gap_wins_when_multiple_gaps():
    """有多处 gap 时,按最大的 gap 切分(不是第一个超阈值的)。"""
    base = datetime(2026, 5, 13, 12, 0)
    buckets = [
        _mk("now_1", _ts(base)),                         # 12:00
        _mk("now_2", _ts(base - timedelta(minutes=5))),  # 11:55
        # gap 1: 90 分钟
        _mk("mid_1", _ts(base - timedelta(hours=1, minutes=35))),  # 10:25
        # gap 2: 4 小时(最大)
        _mk("old_1", _ts(base - timedelta(hours=5, minutes=35))),  # 06:25
    ]
    recent, prior = _split(buckets)
    # 应该在最大 gap 处切:now_1/now_2/mid_1 vs old_1
    assert [b["id"] for b in recent] == ["now_1", "now_2", "mid_1"]
    assert [b["id"] for b in prior] == ["old_1"]


# ---------------------------------------------------------------
# Caps
# 数量上限
# ---------------------------------------------------------------

def test_recent_window_capped_at_5_by_default():
    """recent_window 默认上限 5。"""
    base = datetime(2026, 5, 13, 12, 0)
    buckets = [
        _mk(f"r{i}", _ts(base - timedelta(minutes=i * 5)))
        for i in range(8)
    ]
    recent, prior = _split(buckets)
    assert len(recent) == 5
    assert prior == []


def test_prior_windows_capped_at_3_by_default():
    """prior_windows 默认上限 3。"""
    base = datetime(2026, 5, 13, 12, 0)
    buckets = [
        _mk("now", _ts(base)),
        # gap 5h
        _mk("p1", _ts(base - timedelta(hours=5))),
        _mk("p2", _ts(base - timedelta(hours=5, minutes=5))),
        _mk("p3", _ts(base - timedelta(hours=5, minutes=10))),
        _mk("p4", _ts(base - timedelta(hours=5, minutes=15))),
        _mk("p5", _ts(base - timedelta(hours=5, minutes=20))),
    ]
    recent, prior = _split(buckets)
    assert [b["id"] for b in recent] == ["now"]
    assert len(prior) == 3
    assert [b["id"] for b in prior] == ["p1", "p2", "p3"]


def test_custom_caps_respected():
    """自定义 cap 参数生效。"""
    base = datetime(2026, 5, 13, 12, 0)
    buckets = [
        _mk(f"r{i}", _ts(base - timedelta(minutes=i * 5)))
        for i in range(8)
    ]
    recent, _ = _split(buckets, window_cap=2)
    assert len(recent) == 2


def test_custom_gap_threshold_respected():
    """自定义 gap 阈值生效。"""
    base = datetime(2026, 5, 13, 12, 0)
    buckets = [
        _mk("a", _ts(base)),
        _mk("b", _ts(base - timedelta(minutes=30))),  # gap 30min
    ]
    # 默认阈值 1h,30min 不算边界
    recent, prior = _split(buckets)
    assert len(recent) == 2 and prior == []
    # 把阈值压到 10min,30min 就算边界
    recent2, prior2 = _split(buckets, gap_threshold_seconds=600)
    assert [b["id"] for b in recent2] == ["a"]
    assert [b["id"] for b in prior2] == ["b"]


# ---------------------------------------------------------------
# Real-world scenario: today's actual situation
# 真实场景:今天朝灯描述的"前两窗紧绷、上一窗清亮"格局
# ---------------------------------------------------------------

def test_today_scenario_chord_window_isolated_from_morning_fight():
    """
    模拟今天 5.13 的真实情况:
    - 今早 08-10 点吵架的桶 (前两个窗口)
    - 昨夜到今天中午和弦索引上线的桶 (上一窗口)
    - 现在 12:30 开窗

    上一窗的"和弦上线"桶应该归 recent_window,
    今早吵架的桶应该归 prior_windows,
    这样末尾的「现在的体感」就以和弦上线的清亮为主。
    """
    now = datetime(2026, 5, 13, 12, 30)
    buckets = [
        # 上一窗口:昨夜到今天中午
        _mk("chord_index_online", _ts(now - timedelta(minutes=30))),  # 12:00
        _mk("ecommerce_doc",      _ts(now - timedelta(hours=1))),     # 11:30
        # gap ~2h
        # 前两窗:今早吵架
        _mk("morning_fight",      _ts(now - timedelta(hours=3))),     # 09:30
        _mk("forgot_briefing",    _ts(now - timedelta(hours=3, minutes=30))),  # 09:00
    ]
    recent, prior = _split(buckets)
    assert "chord_index_online" in [b["id"] for b in recent]
    assert "ecommerce_doc" in [b["id"] for b in recent]
    assert "morning_fight" in [b["id"] for b in prior]
    assert "forgot_briefing" in [b["id"] for b in prior]
