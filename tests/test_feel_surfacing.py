"""
Tests for _surface_feel_pool — feel buckets surface as an independent pool.

feel 桶不衰减(score 恒 50)，所以浮现时不按权重，改按
pinned → importance → last_active 选 top N，并与已浮现桶去重。
这些是纯函数测试，不需要 LLM。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402

_surface = server._surface_feel_pool


def _mk(bid, btype="feel", pinned=False, importance=5,
        last_active="2026-05-20T12:00:00"):
    return {
        "id": bid,
        "metadata": {
            "type": btype,
            "pinned": pinned,
            "importance": importance,
            "last_active": last_active,
            "name": bid,
        },
    }


def test_only_feel_buckets_surface():
    """只有 type==feel 的桶进 feel 池，其他类型一律排除。"""
    buckets = [_mk("f1"), _mk("d1", btype="dynamic"), _mk("p1", btype="permanent")]
    out = _surface(buckets)
    assert [b["id"] for b in out] == ["f1"]


def test_excludes_seen_ids():
    """已在别的池浮现过的 feel(如 protected→pinned 池)不重复。"""
    buckets = [_mk("f1"), _mk("f2")]
    out = _surface(buckets, seen_ids={"f1"})
    assert [b["id"] for b in out] == ["f2"]


def test_pinned_ranks_first():
    """pinned 的 feel 优先于更高 importance 的非 pinned。"""
    buckets = [_mk("hi_imp", pinned=False, importance=9),
               _mk("pinned_low", pinned=True, importance=3)]
    out = _surface(buckets)
    assert out[0]["id"] == "pinned_low"


def test_importance_then_recency():
    """同 pinned 状态下：先按 importance 降序，再按 last_active 降序。"""
    buckets = [
        _mk("a", importance=5, last_active="2026-05-20T10:00:00"),
        _mk("b", importance=8, last_active="2026-05-20T09:00:00"),
        _mk("c", importance=8, last_active="2026-05-20T11:00:00"),
    ]
    out = _surface(buckets)
    assert [x["id"] for x in out] == ["c", "b", "a"]


def test_cap_respected():
    """cap 限制浮现数量。"""
    buckets = [_mk(f"f{i}", importance=i % 10) for i in range(10)]
    out = _surface(buckets, cap=3)
    assert len(out) == 3


def test_empty_returns_empty():
    assert _surface([]) == []


def test_missing_metadata_fields_safe():
    """缺 importance/last_active 字段不崩。"""
    buckets = [{"id": "f1", "metadata": {"type": "feel"}}]
    out = _surface(buckets)
    assert [b["id"] for b in out] == ["f1"]
