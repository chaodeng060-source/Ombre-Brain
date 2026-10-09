from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import curated_lexical_recall as lexical


def test_lexical_mode_defaults_off_and_invalid_values_stay_off(monkeypatch) -> None:
    monkeypatch.delenv("OMBRE_PG_LEXICAL_MODE", raising=False)
    assert lexical.pg_lexical_mode() == "off"

    monkeypatch.setenv("OMBRE_PG_LEXICAL_MODE", "unexpected")
    assert lexical.pg_lexical_mode() == "off"

    monkeypatch.setenv("OMBRE_PG_LEXICAL_MODE", "shadow")
    assert lexical.pg_lexical_mode() == "shadow"


def test_explicit_literal_keeps_the_complete_phrase(monkeypatch) -> None:
    monkeypatch.setattr(
        lexical,
        "literal_query_terms",
        lambda _query, max_terms: ["得毕业答辩", "毕业答辩", "答辩"][:max_terms],
    )

    assert lexical.substantive_literal_terms("查一下毕业答辩") == ["毕业答辩"]


def test_merge_prefers_exact_literal_and_bounds_scores() -> None:
    hits = lexical._merge_curated_rows(
        [("shared", 0.4), ("fts", 2.0)],
        [("shared", 0.8), ("literal", -1.0)],
        limit=3,
    )

    by_id = {hit.bucket_id: hit for hit in hits}
    assert by_id["shared"].channel == "literal"
    assert by_id["shared"].original_support == 1.0
    assert by_id["fts"].score == 1.0
    assert by_id["literal"].score == 0.0
    assert len(hits) == 3


def test_search_is_a_no_op_when_mode_is_off(monkeypatch) -> None:
    class UnexpectedConnection:
        @staticmethod
        async def connect(*_args, **_kwargs):
            raise AssertionError("off mode must not connect to PostgreSQL")

    monkeypatch.delenv("OMBRE_PG_LEXICAL_MODE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(AsyncConnection=UnexpectedConnection),
    )

    result = asyncio.run(
        lexical.search_curated_lexical("雾凇雪屋", include_fts=True)
    )
    assert result == []


def test_live_query_is_parameterized_read_only_and_bounded(monkeypatch) -> None:
    calls: list[tuple[str, tuple]] = []
    connect_calls: list[tuple[str, dict]] = []
    rows_by_call = [
        [("literal-hit", 1.0)],
        [("fts-hit", 0.3), ("literal-hit", 0.2)],
    ]

    class FakeCursor:
        def __init__(self) -> None:
            self.rows = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def execute(self, statement: str, parameters: tuple) -> None:
            calls.append((statement, parameters))
            self.rows = rows_by_call[len(calls) - 1]

        async def fetchall(self):
            return self.rows

    class FakeConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        def cursor(self):
            return FakeCursor()

    class FakeAsyncConnection:
        @staticmethod
        async def connect(dsn: str, **kwargs):
            connect_calls.append((dsn, kwargs))
            return FakeConnection()

    monkeypatch.setenv("OMBRE_PG_LEXICAL_MODE", "live")
    monkeypatch.setattr(lexical, "should_run_local_literal", lambda _query: True)
    monkeypatch.setattr(
        lexical,
        "substantive_literal_terms",
        lambda _query: ["雾凇雪屋"],
    )
    monkeypatch.setattr(lexical, "lexical_query", lambda _query: "雾凇 雪屋")
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(AsyncConnection=FakeAsyncConnection),
    )

    hits = asyncio.run(
        lexical.search_curated_lexical(
            "private-query-marker", top_k=99, include_fts=True, dsn="postgresql://test"
        )
    )

    assert connect_calls[0][0] == "postgresql://test"
    assert connect_calls[0][1]["connect_timeout"] == 2
    options = connect_calls[0][1]["options"]
    assert "default_transaction_read_only=on" in options
    assert "statement_timeout=500ms" in options
    assert "lock_timeout=100ms" in options
    assert len(calls) == 2
    assert "private-query-marker" not in " ".join(sql for sql, _ in calls)
    assert calls[0][1][-1] == 3
    assert calls[1][1][-1] == 20
    assert [hit.bucket_id for hit in hits] == ["literal-hit", "fts-hit"]
    assert hits[0].channel == "literal"
    assert hits[0].original_support == 1.0


def test_database_errors_fail_soft_without_logging_the_query(monkeypatch, caplog) -> None:
    class BrokenAsyncConnection:
        @staticmethod
        async def connect(*_args, **_kwargs):
            raise RuntimeError("synthetic connection failure")

    monkeypatch.setenv("OMBRE_PG_LEXICAL_MODE", "shadow")
    monkeypatch.setattr(lexical, "should_run_local_literal", lambda _query: True)
    monkeypatch.setattr(
        lexical,
        "substantive_literal_terms",
        lambda _query: ["雾凇雪屋"],
    )
    monkeypatch.setattr(lexical, "lexical_query", lambda _query: "雾凇 雪屋")
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(AsyncConnection=BrokenAsyncConnection),
    )

    result = asyncio.run(
        lexical.search_curated_lexical(
            "private-query-marker", include_fts=True, dsn="postgresql://test"
        )
    )

    assert result == []
    assert "private-query-marker" not in caplog.text
