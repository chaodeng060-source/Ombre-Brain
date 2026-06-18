"""
Tests for _event_age_label — the hard date stamp injected into every
briefing material bucket so the LLM always has an absolute time anchor.

测试 _event_age_label：简报素材里每个浮现桶的硬日期章。
根因：旧桶从「高权重未解决」「感情红线」等池冒出来时，日期没有进入最终消费路径，
LLM 会自行编「前两天」。本章给所有浮现桶强制盖章，没日期就显式禁止近期化。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402

_label = server._event_age_label  # noqa: E402

NOW = datetime(2026, 6, 18, 9, 0)


def _mk(created=None):
    meta = {}
    if created is not None:
        meta["created"] = created
    return {"id": "x", "metadata": meta}


def _ts(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# --- 核心：缺日期的旧桶必须显式禁近期化（卡兜咬人桶 created 为空）---

def test_missing_created_emits_no_date_warning():
    """created 缺失 → 显式「无确切日期」+ 禁词，而非留空让 LLM 编「前两天」。"""
    out = _label(_mk(), now=NOW)
    assert "无确切日期" in out
    assert "前两天" in out  # 禁词清单点名


def test_empty_metadata_emits_no_date_warning():
    """空 metadata 也走禁近期化分支，不抛异常。"""
    out = _label({"id": "x", "metadata": {}}, now=NOW)
    assert "无确切日期" in out


def test_unparseable_created_emits_no_date_warning():
    out = _label(_mk(created="not-a-date"), now=NOW)
    assert "无确切日期" in out


# --- 有日期：盖出绝对日期 + 距今天数 ---

def test_month_old_event_stamped_with_real_age():
    """卡兜场景：一个月前的事必须盖「距今 N 天」，不许相对化。"""
    b = _mk(created=_ts(NOW - timedelta(days=30)))
    out = _label(b, now=NOW)
    assert "2026-05-19" in out
    assert "距今 30 天" in out


def test_today_label():
    out = _label(_mk(created=_ts(NOW - timedelta(hours=2))), now=NOW)
    assert "今天" in out


def test_yesterday_label():
    out = _label(_mk(created=_ts(NOW - timedelta(days=1, hours=1))), now=NOW)
    assert "昨天" in out


def test_cross_midnight_uses_calendar_day_not_elapsed_24_hours():
    b = _mk(created="2026-06-17T23:50:00")
    out = _label(b, now=datetime(2026, 6, 18, 0, 10))
    assert "2026-06-17" in out
    assert "昨天" in out


def test_future_date_is_warning_not_today():
    b = _mk(created="2026-06-19T09:00:00")
    out = _label(b, now=NOW)
    assert "晚于当前日期" in out
    assert "今天" not in out


def test_specific_date_rendered():
    b = _mk(created="2026-05-30T10:00:00")
    out = _label(b, now=NOW)
    assert "发生于 2026-05-30" in out


# --- tz-aware created 不崩 ---

def test_tz_aware_created_handled():
    b = _mk(created=(NOW - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S+08:00"))
    # 不传 now，让函数内部按 ev.tzinfo 取对应 now；只要求不抛异常、返回带日期的串
    out = _label(b)
    assert isinstance(out, str) and "发生于" in out


def test_tz_aware_date_is_rendered_in_beijing_calendar():
    b = _mk(created="2026-06-17T16:30:00+00:00")
    out = _label(b, now=datetime.fromisoformat("2026-06-18T01:00:00+00:00"))
    assert "发生于 2026-06-18（今天）" in out
