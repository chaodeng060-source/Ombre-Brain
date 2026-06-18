"""
Tests for backfill_created.infer_created — pure date-inference for the
全库时间标 backfill. No filesystem / config needed.

测试 created 回填的纯日期推断：优先级、幂等、保守拒绝。
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backfill_created import infer_created, _first_valid_date  # noqa: E402

TODAY = date(2026, 6, 18)


def _meta(name="", tags=None, created=None):
    m = {"name": name, "tags": tags or []}
    if created is not None:
        m["created"] = created
    return m


# --- 幂等：已有合法 created 不覆盖 ---

def test_existing_created_skipped():
    iso, src = infer_created(_meta(name="卡兜_2026-05-30",
                                   created="2026-05-30T00:00:00"), today=TODAY)
    assert src == "already-has-created"
    assert iso is None


def test_malformed_created_rederived_from_name():
    """created 损坏 → 不当它存在，从桶名重新推断。"""
    iso, src = infer_created(_meta(name="卡兜咬人事件_2026-05-30",
                                   created="not-a-date"), today=TODAY)
    assert src == "name"
    assert iso == "2026-05-30T00:00:00"


# --- 优先级：name > tag > content ---

def test_name_date_used():
    iso, src = infer_created(_meta(name="卡兜咬人事件_2026-05-30"), today=TODAY)
    assert src == "name" and iso == "2026-05-30T00:00:00"


def test_tag_date_when_name_has_none():
    iso, src = infer_created(_meta(name="aa4b2ec0d57d", tags=["卡兜", "2026-06-02"]),
                             today=TODAY)
    assert src == "tag" and iso == "2026-06-02T00:00:00"


def test_name_wins_over_tag():
    iso, src = infer_created(_meta(name="事件_2026-05-30", tags=["2026-01-01"]),
                             today=TODAY)
    assert src == "name" and iso == "2026-05-30T00:00:00"


def test_content_only_opt_in():
    meta = _meta(name="无日期桶", tags=[])
    # 默认不扫正文
    iso, src = infer_created(meta, content="2026-06-02 发生的事", today=TODAY)
    assert src == "no-date-found" and iso is None
    # 开了 --scan-content 才用
    iso, src = infer_created(meta, content="2026-06-02 发生的事",
                             today=TODAY, scan_content=True)
    assert src == "content" and iso == "2026-06-02T00:00:00"


# --- 保守拒绝 ---

def test_no_date_anywhere():
    iso, src = infer_created(_meta(name="纯工程笔记", tags=["工程"]), today=TODAY)
    assert src == "no-date-found" and iso is None


def test_future_date_rejected():
    """晚于今天的日期不可能是事件时间 → 不采用。"""
    iso, src = infer_created(_meta(name="计划_2026-12-31"), today=TODAY)
    assert src == "no-date-found" and iso is None


def test_impossible_date_rejected():
    assert _first_valid_date("2026-13-40", TODAY) is None


def test_old_year_rejected():
    """年份 < 2024 当噪声拒绝（避免把版本号/无关数字当日期）。"""
    assert _first_valid_date("1999-01-01", TODAY) is None


# --- 多种分隔符 ---

def test_dot_and_slash_separators():
    assert _first_valid_date("迁移 2026.6.2 完成", TODAY) == date(2026, 6, 2)
    assert _first_valid_date("2026/5/30 咬人", TODAY) == date(2026, 5, 30)
