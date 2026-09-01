"""PG 召回快路径的契约。

2026-08-21 00:45 接上：召回的向量检索从「全库逐桶 O(n) 余弦扫描」
换成 PG 的 ivfflat 索引查询（实测 11903 桶 3530ms → 154ms）。

这里锁三条命脉，任何一条破了都会直接伤到朝灯能不能搜到自己的记忆：
  1. 开关默认关 —— 不许因为代码上线就悄悄改变现有召回行为
  2. PG 出任何问题都必须回落扫表 —— 宁可慢，不能让她的记忆凭空消失
  3. 空结果也要回落 —— 空可能是镜像没灌好，不等于「真的没有相关记忆」
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_engine import EmbeddingEngine  # noqa: E402


def _engine() -> EmbeddingEngine:
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    engine.enabled = True
    return engine


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  "])
def test_pg_recall_disabled_by_default_and_for_falsy_values(monkeypatch, value):
    """默认必须是关的。上线代码不等于改变她当下的召回行为。"""
    if value == "":
        monkeypatch.delenv("OMBRE_PG_RECALL_ENABLED", raising=False)
    else:
        monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", value)
    assert _engine()._pg_recall_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_pg_recall_enabled_for_truthy_values(monkeypatch, value):
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", value)
    assert _engine()._pg_recall_enabled() is True


@pytest.mark.asyncio
async def test_pg_failure_falls_back_instead_of_raising(monkeypatch):
    """PG 连不上时返回 None（触发调用方回落扫表），绝不抛异常。

    抛异常会让整次召回失败 —— 对她来说就是「哥哥突然不记得我了」。
    """
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")

    class _Boom:
        @staticmethod
        async def connect(*_a, **_kw):
            raise RuntimeError("connection refused")

    fake = types.SimpleNamespace(AsyncConnection=_Boom)
    monkeypatch.setitem(sys.modules, "psycopg", fake)

    got = await _engine()._search_similar_pg([0.1] * 1024, 10)
    assert got is None, "PG 失败必须返回 None 以回落扫表"


@pytest.mark.asyncio
async def test_missing_driver_falls_back(monkeypatch):
    """驱动没装也要回落，不能让召回直接崩。"""
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")
    monkeypatch.setitem(sys.modules, "psycopg", None)  # import 时报 TypeError
    got = await _engine()._search_similar_pg([0.1] * 1024, 10)
    assert got is None


@pytest.mark.asyncio
async def test_empty_result_falls_back_rather_than_claiming_no_memories(monkeypatch):
    """PG 返回空 → 回落扫表。

    空结果可能是镜像没灌好。若直接把空当答案，她会看到「没有相关记忆」，
    而实际上 md 里明明有 —— 这正是 7 月那次「记了但搜不到」的伤口。
    """
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")

    class _Cur:
        async def execute(self, *_a, **_kw):
            return None

        async def fetchall(self):
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class _Connector:
        @staticmethod
        async def connect(*_a, **_kw):
            return _Conn()

    monkeypatch.setitem(
        sys.modules, "psycopg", types.SimpleNamespace(AsyncConnection=_Connector)
    )
    got = await _engine()._search_similar_pg([0.1] * 1024, 10)
    assert got is None, "空结果必须回落，不能当成『没有相关记忆』"


@pytest.mark.asyncio
async def test_distance_is_converted_to_similarity(monkeypatch):
    """PG 给的是余弦距离，召回全链路用的是相似度：sim = 1 - distance。

    搞反了排序会整个倒过来 —— 最不相关的排最前。
    """
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")
    rows = [("bucket_near", 0.1), ("bucket_far", 0.9)]

    class _Cur:
        async def execute(self, *_a, **_kw):
            return None

        async def fetchall(self):
            return rows

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        types.SimpleNamespace(
            AsyncConnection=types.SimpleNamespace(
                connect=lambda *_a, **_kw: _make(_Conn())
            )
        ),
    )
    got = await _engine()._search_similar_pg([0.1] * 1024, 10)
    assert got == [("bucket_near", pytest.approx(0.9)),
                   ("bucket_far", pytest.approx(0.1))]


@pytest.mark.asyncio
async def test_selected_bucket_scores_share_pg_query_embedding_and_keep_top_k(
    monkeypatch,
):
    """E 锚分数来自同一 PG 镜像，且不能改变正常召回的 top-k。"""
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")
    normal_rows = [("normal_near", 0.1), ("normal_far", 0.4)]
    selected_rows = [("e_comfort", 0.2), ("e_boundary", 0.7)]
    executed = []

    class _Cur:
        def __init__(self):
            self._selected_query_seen = False

        async def execute(self, sql, params=None):
            executed.append((sql, params))
            if "WHERE bucket_id = ANY" in sql:
                self._selected_query_seen = True

        async def fetchall(self):
            return selected_rows if self._selected_query_seen else normal_rows

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        types.SimpleNamespace(
            AsyncConnection=types.SimpleNamespace(
                connect=lambda *_a, **_kw: _make(_Conn())
            )
        ),
    )

    top_k, selected = await _engine()._search_similar_pg_with_selected_scores(
        [0.1] * 1024,
        2,
        frozenset({"e_boundary", "e_comfort"}),
    )

    assert top_k == [
        ("normal_near", pytest.approx(0.9)),
        ("normal_far", pytest.approx(0.6)),
    ]
    assert selected == {
        "e_comfort": pytest.approx(0.8),
        "e_boundary": pytest.approx(0.3),
    }
    selected_calls = [
        (sql, params)
        for sql, params in executed
        if "WHERE bucket_id = ANY" in sql
    ]
    assert len(selected_calls) == 1
    assert selected_calls[0][1][1] == ["e_boundary", "e_comfort"]


@pytest.mark.asyncio
async def test_selected_bucket_query_failure_preserves_normal_pg_recall(monkeypatch):
    """E 侧读证失败只回落 Russell，不能炸掉已经拿到的正常召回。"""
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")
    normal_rows = [("normal_near", 0.1)]

    class _Cur:
        async def execute(self, sql, _params=None):
            if "WHERE bucket_id = ANY" in sql:
                raise RuntimeError("selected read failed")

        async def fetchall(self):
            return normal_rows

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        types.SimpleNamespace(
            AsyncConnection=types.SimpleNamespace(
                connect=lambda *_a, **_kw: _make(_Conn())
            )
        ),
    )

    top_k, selected = await _engine()._search_similar_pg_with_selected_scores(
        [0.1] * 1024,
        1,
        frozenset({"e_comfort"}),
    )

    assert top_k == [("normal_near", pytest.approx(0.9))]
    assert selected == {}


async def _make(value):
    return value
