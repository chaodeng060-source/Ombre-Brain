"""
Tests for _created_within_days — the absolute age gate on the briefing's
"recently active" narrative sections.

测试 _created_within_days：简报「最近活跃」叙事段的绝对年龄闸。
根因：last_active 会被 inspect/backfill_relations/touch/update bump，
一个月前的桶会冒充「最近活跃」被 LLM 写成「前两天」（朝灯 2026-06-08 戳穿卡兜事）。
修复：改用 created（事件真正发生时间）做硬性上限。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402
_within = server._created_within_days  # noqa: E402

NOW = datetime(2026, 6, 8, 7, 25)


def _mk(created=None, last_active=None):
    meta = {}
    if created is not None:
        meta["created"] = created
    if last_active is not None:
        meta["last_active"] = last_active
    return {"id": "x", "metadata": meta}


def _ts(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# --- 核心 bug：一个月前的桶 last_active 被 bump 到现在，仍要被踢出 ---

def test_month_old_event_bumped_last_active_is_rejected():
    """卡兜场景：created 一个月前、last_active 昨夜被 backfill 刷新 → 踢出。"""
    b = _mk(
        created=_ts(NOW - timedelta(days=30)),       # 事件一个月前
        last_active=_ts(NOW - timedelta(hours=8)),   # 维护操作刚 bump
    )
    assert _within(b, 7, now=NOW) is False


def test_recent_event_passes():
    """昨天发生的事 → 进最近活跃叙事。"""
    b = _mk(created=_ts(NOW - timedelta(days=1)))
    assert _within(b, 7, now=NOW) is True


def test_event_just_inside_window_passes():
    """正好 7 天内（边界）→ 通过。"""
    b = _mk(created=_ts(NOW - timedelta(days=7) + timedelta(minutes=1)))
    assert _within(b, 7, now=NOW) is True


def test_event_just_outside_window_rejected():
    """刚过 7 天 → 踢出。"""
    b = _mk(created=_ts(NOW - timedelta(days=7) - timedelta(minutes=1)))
    assert _within(b, 7, now=NOW) is False


# --- 缺失/坏数据：保守踢出 ---

def test_missing_created_rejected():
    """没有 created（哪怕 last_active 很新）→ 保守踢出。"""
    b = _mk(last_active=_ts(NOW))
    assert _within(b, 7, now=NOW) is False


def test_unparseable_created_rejected():
    """created 解析失败 → 踢出。"""
    b = _mk(created="not-a-date")
    assert _within(b, 7, now=NOW) is False


def test_empty_metadata_rejected():
    """空 metadata → 踢出，不抛异常。"""
    assert _within({"id": "x", "metadata": {}}, 7, now=NOW) is False


# --- tz-aware created 也能比 ---

def test_tz_aware_created_handled():
    """带时区的 created 不会因 naive/aware 相减报错。"""
    b = _mk(created=(NOW - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S+08:00"))
    # 不传 now，让函数内部按 ev.tzinfo 取对应 now；只要求不抛异常、返回 bool
    assert isinstance(_within(b, 7), bool)


# --- 配置可调 ---

def test_custom_max_age_days():
    """收紧到 3 天时，5 天前的事被踢出。"""
    b = _mk(created=_ts(NOW - timedelta(days=5)))
    assert _within(b, 7, now=NOW) is True
    assert _within(b, 3, now=NOW) is False
