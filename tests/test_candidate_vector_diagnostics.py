"""Candidate vector diagnostic seam — numeric-only, read-only contract.

These tests pin the private diagnostic entry point
``EmbeddingEngine.search_similar_diagnostic``.  They use numeric mechanical
fixtures only (no provider, no real database, no private corpus).  The core
must reuse the ordinary ranking path and the single redacted/truncated query
embedding, suppress the background shadow, and expose membership/ranks
without ever reporting a present-but-unscorable row as absent.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import types

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_engine import EmbeddingEngine  # noqa: E402
from redact import redact_embedding_input  # noqa: E402


# --------------------------------------------------------------------------
# Numeric fixtures
# --------------------------------------------------------------------------

QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]


def _ok_response(vector):
    return types.SimpleNamespace(
        data=[types.SimpleNamespace(embedding=list(vector))]
    )


class _RecordingEmbeddings:
    def __init__(self, outcome):
        self.calls: list[dict] = []
        self._outcome = outcome

    async def create(self, *, model, input):
        self.calls.append({"model": model, "input": input})
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _RecordingClient:
    """Only ``embeddings`` is a valid attribute; anything else is a test bug."""

    def __init__(self, outcome):
        self.embeddings = _RecordingEmbeddings(outcome)
        self.options = []

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def __getattr__(self, name):
        raise AssertionError(f"forbidden embedding client access: {name}")


def _make_engine(tmp_path, client, *, enabled=True, db_name="embeddings.db"):
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    engine.enabled = enabled
    engine.client = client
    engine.model = "numeric-test-model"
    engine.timeout = 1.0
    engine.base_url = "http://numeric.invalid"
    engine.api_key = ""
    engine.embed_proxy = ""
    engine._circuit_threshold = 3
    engine._circuit_cooldown = 300
    engine._consec_fail = 0
    engine._circuit_until = 0.0
    engine.db_path = str(tmp_path / db_name)
    return engine


def _write_sqlite_store(db_path, rows):
    """``rows`` is a list of ``(bucket_id, raw_embedding)`` JSON strings."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE embeddings ("
        "bucket_id TEXT PRIMARY KEY, embedding TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    for bucket_id, raw in rows:
        conn.execute(
            "INSERT INTO embeddings (bucket_id, embedding, updated_at) "
            "VALUES (?, ?, ?)",
            (bucket_id, raw, "2026-01-01T00:00:00Z"),
        )
    conn.commit()
    conn.close()


def _forbidden(*_args, **_kwargs):
    raise AssertionError("diagnostic must not reach this seam")


# --------------------------------------------------------------------------
# Fake psycopg connector (protocol-shaped, no real driver/database)
# --------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, ann_rows, selected_rows=()):
        self.ann_rows = list(ann_rows)
        self.selected_rows = list(selected_rows)
        self._next = []
        self.executed: list[tuple] = []

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "WHERE bucket_id = ANY" in sql:
            self._next = self.selected_rows
        elif "FROM ombre_vectors" in sql:
            rows = self.ann_rows
            # The exact rollback query carries the requested LIMIT; honour it so
            # the fake matches the real driver's protocol shape.
            if "GROUP BY bucket_id" in sql and params:
                rows = rows[: int(params[-1])]
            self._next = rows
        else:
            self._next = []

    async def fetchall(self):
        return list(self._next)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FailingSelectedCursor(_FakeCursor):
    async def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "WHERE bucket_id = ANY" in sql:
            raise RuntimeError("selected probe failed")
        if "FROM ombre_vectors" in sql:
            self._next = self.ann_rows
        else:
            self._next = []


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeConnector:
    def __init__(self, cursor):
        self._cursor = cursor
        self.connect_calls: list[dict] = []

    async def connect(self, _dsn, **kwargs):
        self.connect_calls.append(dict(kwargs))
        return _FakeConn(self._cursor)


def _install_pg(monkeypatch, cursor):
    connector = _FakeConnector(cursor)
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        types.SimpleNamespace(AsyncConnection=connector),
    )
    return connector


# --------------------------------------------------------------------------
# Single embedding call, redaction/truncation, no retry
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_diagnostic_makes_exactly_one_redacted_truncated_call(tmp_path):
    secret = "sk-abcdefghij1234567890"
    query = secret + " " + ("y" * 3000)
    engine = _make_engine(tmp_path, _RecordingClient(_ok_response(QUERY_VECTOR)))
    _write_sqlite_store(
        engine.db_path,
        [
            ("b1", json.dumps([1.0, 0.0, 0.0, 0.0])),
            ("b2", json.dumps([0.8, 0.6, 0.0, 0.0])),
        ],
    )

    hits, status, selected, diag = await engine.search_similar_diagnostic(
        query,
        top_k=2,
        target_ids=["b1", "t_absent"],
        score_bucket_ids=["b2"],
    )

    calls = engine.client.embeddings.calls
    assert len(calls) == 1, "diagnostic must reuse a single embedding call"
    assert engine.client.options == [{"max_retries": 0}]
    recorded = calls[0]["input"]
    assert recorded == redact_embedding_input(query)[:2000]
    assert secret not in recorded
    assert "[REDACTED]" in recorded
    assert len(recorded) == 2000
    assert calls[0]["model"] == "numeric-test-model"

    assert status == "ok", diag
    assert diag["backend"] == "sqlite"
    assert diag["targets"]["b1"]["returned_rank"] == 1
    # E-axis id is scored but never reported as a target.
    assert set(diag["targets"]) == {"b1", "t_absent"}
    assert "b2" not in diag["targets"]
    assert selected["b2"] == pytest.approx(0.8)

    blob = json.dumps(diag)
    assert query not in blob
    assert secret not in blob
    assert "y" * 50 not in blob


@pytest.mark.asyncio
async def test_diagnostic_does_not_retry_on_failure(tmp_path):
    engine = _make_engine(
        tmp_path, _RecordingClient(RuntimeError("numeric failure"))
    )
    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q", target_ids=["t1"]
    )
    assert len(engine.client.embeddings.calls) == 1
    assert status == "error"
    assert diag["status"] == "error"
    assert diag["targets"]["t1"]["index_member"] is None
    assert diag["targets"]["t1"]["selected_score"] is None
    assert engine._consec_fail == 0
    assert engine._circuit_until == 0


@pytest.mark.asyncio
async def test_diagnostic_reports_timeout_status_once(tmp_path):
    engine = _make_engine(
        tmp_path, _RecordingClient(httpx.TimeoutException("slow"))
    )
    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q", target_ids=["t1"]
    )
    assert len(engine.client.embeddings.calls) == 1
    assert status == "timeout"
    assert diag["status"] == "timeout"
    assert diag["targets"]["t1"]["index_member"] is None


@pytest.mark.asyncio
async def test_disabled_engine_makes_no_call_and_reports_unknown(tmp_path):
    engine = _make_engine(tmp_path, None, enabled=False)
    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q", target_ids=["t1", "t2"]
    )
    assert (hits, status, selected) == ([], "error", {})
    assert diag["backend"] == "none"
    assert diag["status"] == "error"
    assert all(
        entry["index_member"] is None for entry in diag["targets"].values()
    )


# --------------------------------------------------------------------------
# SQLite read-only path
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sqlite_outside_topk_absent_and_unscorable(tmp_path):
    engine = _make_engine(tmp_path, _RecordingClient(_ok_response(QUERY_VECTOR)))
    _write_sqlite_store(
        engine.db_path,
        [
            ("b_top", json.dumps([1.0, 0.0, 0.0, 0.0])),
            ("b_mid", json.dumps([0.8, 0.6, 0.0, 0.0])),
            ("b_low", json.dumps([0.0, 1.0, 0.0, 0.0])),
            ("b_dim", json.dumps([1.0, 0.0])),
            ("b_bad", "not-json"),
        ],
    )

    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q",
        top_k=1,
        target_ids=["b_mid", "b_low", "b_dim", "b_bad", "t_absent"],
        score_bucket_ids=["b_top"],
    )

    assert status == "ok", diag
    assert hits == [("b_top", pytest.approx(1.0))], "normal_hits must not widen"
    assert diag["backend"] == "sqlite"
    assert diag["rank_scope"] == "sqlite_full_scan"
    assert set(diag["targets"]) == {
        "b_mid",
        "b_low",
        "b_dim",
        "b_bad",
        "t_absent",
    }

    mid = diag["targets"]["b_mid"]
    assert mid == {
        "index_member": True,
        "returned_rank": None,
        "selected_score": pytest.approx(0.8),
        "scan_rank": 2,
        "scorable": True,
    }

    low = diag["targets"]["b_low"]
    assert low["index_member"] is True
    assert low["scan_rank"] == 3
    assert low["selected_score"] == pytest.approx(0.0)
    assert low["scorable"] is True

    # Present but not parseable: a member, never reported absent.
    bad = diag["targets"]["b_bad"]
    assert bad["index_member"] is True
    assert bad["scan_rank"] is None
    assert bad["selected_score"] is None
    assert bad["scorable"] is False

    # Present but dimension-mismatched: a member, unscorable.
    dim = diag["targets"]["b_dim"]
    assert dim["index_member"] is True
    assert dim["scorable"] is False

    absent = diag["targets"]["t_absent"]
    assert absent["index_member"] is False
    assert absent["scan_rank"] is None
    assert absent["selected_score"] is None
    assert absent["scorable"] is False

    assert selected["b_top"] == pytest.approx(1.0)
    assert "b_mid" in selected


@pytest.mark.asyncio
async def test_diagnostic_avoids_init_and_cache_writes(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, _RecordingClient(_ok_response(QUERY_VECTOR)))
    _write_sqlite_store(
        engine.db_path,
        [("b1", json.dumps([1.0, 0.0, 0.0, 0.0]))],
    )
    for name in (
        "_init_db",
        "_store_embedding",
        "_invalidate_vector_cache",
        "_get_cached_vectors",
        "_schedule_pg_vector_shadow",
    ):
        monkeypatch.setattr(engine, name, _forbidden)

    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q", top_k=1, target_ids=["b1"]
    )
    assert status == "ok", diag
    assert hits == [("b1", pytest.approx(1.0))]
    assert diag["targets"]["b1"]["index_member"] is True


@pytest.mark.asyncio
async def test_unavailable_store_reports_error_not_absent(tmp_path):
    engine = _make_engine(
        tmp_path,
        _RecordingClient(_ok_response(QUERY_VECTOR)),
        db_name="missing/embeddings.db",
    )
    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q", target_ids=["t1"]
    )
    assert status == "error"
    assert diag["backend"] == "none"
    assert diag["rank_scope"] == "none"
    assert diag["targets"]["t1"]["index_member"] is None


@pytest.mark.asyncio
async def test_unpreparable_query_reports_error_not_absent(tmp_path):
    engine = _make_engine(
        tmp_path, _RecordingClient(_ok_response("bad-shape"))
    )
    _write_sqlite_store(
        engine.db_path,
        [("b1", json.dumps([1.0, 0.0, 0.0, 0.0]))],
    )
    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q", top_k=1, target_ids=["b1", "t2"]
    )
    assert status == "error"
    assert diag["status"] == "error"
    assert diag["rank_scope"] == "none"
    assert diag["targets"]["b1"]["index_member"] is None


# --------------------------------------------------------------------------
# PG path
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pg_hnsw_outside_ann_topk_vs_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")
    monkeypatch.setenv("OMBRE_PG_RECALL_MODE", "hnsw")
    cursor = _FakeCursor(
        ann_rows=[("b1", 0.1), ("b2", 0.2)],
        selected_rows=[("b2", 0.3)],
    )
    connector = _install_pg(monkeypatch, cursor)
    client = _RecordingClient(_ok_response(QUERY_VECTOR))
    engine = _make_engine(tmp_path, client)
    monkeypatch.setattr(engine, "_schedule_pg_vector_shadow", _forbidden)

    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q",
        top_k=1,
        target_ids=["b2", "t_absent"],
        score_bucket_ids=["b1"],
    )

    assert status == "ok", diag
    assert hits == [("b1", pytest.approx(0.9))], "shadow/selected must not widen"
    assert diag["backend"] == "pg_hnsw"
    assert diag["rank_scope"] == "ann_topk"
    assert set(diag["targets"]) == {"b2", "t_absent"}

    b2 = diag["targets"]["b2"]
    assert b2["index_member"] is True
    assert b2["returned_rank"] is None, "no invented global ANN rank"
    assert b2["selected_score"] == pytest.approx(0.7)
    assert b2["scan_rank"] is None

    absent = diag["targets"]["t_absent"]
    assert absent["index_member"] is False
    assert absent["selected_score"] is None

    assert selected["b2"] == pytest.approx(0.7)
    assert len(client.embeddings.calls) == 1
    assert connector.connect_calls
    assert (
        connector.connect_calls[0].get("options")
        == "-c default_transaction_read_only=on"
    )


@pytest.mark.asyncio
async def test_pg_exact_backend_label(tmp_path, monkeypatch):
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")
    monkeypatch.setenv("OMBRE_PG_RECALL_MODE", "exact")
    cursor = _FakeCursor(
        ann_rows=[("b1", 0.1), ("b2", 0.2)],
        selected_rows=[("b2", 0.3)],
    )
    _install_pg(monkeypatch, cursor)
    engine = _make_engine(tmp_path, _RecordingClient(_ok_response(QUERY_VECTOR)))
    monkeypatch.setattr(engine, "_schedule_pg_vector_shadow", _forbidden)

    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q", top_k=1, target_ids=["b2"], score_bucket_ids=["b1"]
    )
    assert status == "ok"
    assert hits == [("b1", pytest.approx(0.9))]
    assert diag["backend"] == "pg_exact"
    assert diag["rank_scope"] == "pg_exact_topk"
    assert diag["targets"]["b2"]["index_member"] is True
    assert diag["targets"]["b2"]["returned_rank"] is None
    assert diag["targets"]["b2"]["selected_score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_pg_selected_failure_marks_membership_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")
    monkeypatch.setenv("OMBRE_PG_RECALL_MODE", "hnsw")
    cursor = _FailingSelectedCursor(ann_rows=[("b1", 0.1)])
    _install_pg(monkeypatch, cursor)
    engine = _make_engine(tmp_path, _RecordingClient(_ok_response(QUERY_VECTOR)))
    monkeypatch.setattr(engine, "_schedule_pg_vector_shadow", _forbidden)

    hits, status, selected, diag = await engine.search_similar_diagnostic(
        "q", top_k=1, target_ids=["b1", "t_unknown"]
    )

    assert status == "ok", diag
    assert hits == [("b1", pytest.approx(0.9))]
    assert diag["backend"] == "pg_hnsw"
    # In the returned top-k -> known member.
    assert diag["targets"]["b1"]["index_member"] is True
    assert diag["targets"]["b1"]["returned_rank"] == 1
    # Failed membership probe -> unknown, never "absent".
    unknown = diag["targets"]["t_unknown"]
    assert unknown["index_member"] is None
    assert unknown["selected_score"] is None
    assert unknown["scan_rank"] is None


# --------------------------------------------------------------------------
# Ordinary-path compatibility
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ordinary_path_keeps_historical_helper_signatures(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")
    monkeypatch.setenv("OMBRE_PG_RECALL_MODE", "hnsw")
    client = _RecordingClient(_ok_response(QUERY_VECTOR))
    engine = _make_engine(tmp_path, client)
    seen = {}

    async def legacy_selected(query_embedding, top_k, score_bucket_ids):
        seen["selected"] = (list(query_embedding), top_k, set(score_bucket_ids))
        return [("b1", 0.9)], {"b1": 0.9}

    async def legacy_plain(query_embedding, top_k):
        seen["plain"] = (list(query_embedding), top_k)
        return [("b1", 0.9)]

    monkeypatch.setattr(
        engine, "_search_similar_pg_with_selected_scores", legacy_selected
    )
    monkeypatch.setattr(engine, "_search_similar_pg", legacy_plain)
    monkeypatch.setattr(
        engine, "_schedule_pg_vector_shadow", lambda *_a, **_k: None
    )

    hits, status, scores = await engine.search_similar_with_selected_scores(
        "q", top_k=2, score_bucket_ids=["b1"]
    )
    assert seen["selected"] == (QUERY_VECTOR, 2, {"b1"})
    assert hits == [("b1", pytest.approx(0.9))]
    assert scores == {"b1": pytest.approx(0.9)}

    hits2, status2 = await engine.search_similar_with_status("q", top_k=2)
    assert seen["plain"] == (QUERY_VECTOR, 2)
    assert hits2 == [("b1", pytest.approx(0.9))]
    assert status2 == "ok"

    # The ordinary path must not pass read-only options to PG.
    assert len(client.embeddings.calls) == 2
    assert client.options == []


@pytest.mark.asyncio
async def test_real_sdk_diagnostic_disables_transport_retries(tmp_path):
    from openai import AsyncOpenAI
    attempts = []
    def respond(req):
        attempts.append(req)
        return httpx.Response(500, json={"error": {"message": "numeric failure"}})
    transport = httpx.MockTransport(respond)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = AsyncOpenAI(api_key="numeric-fixture", base_url="https://fixture.invalid/v1",
                             max_retries=4, http_client=http_client)
        engine = _make_engine(tmp_path, client)
        engine._consec_fail = 2
        hits, status, _, diag = await engine.search_similar_diagnostic("1", target_ids=["b1"])
        assert status == "error"
        assert len(attempts) == 1
        assert client.max_retries == 4
        assert engine._consec_fail == 2
        assert engine._circuit_until == 0
        assert diag["targets"]["b1"]["index_member"] is None


@pytest.mark.asyncio
async def test_ordinary_shadow_still_scheduled_but_diagnostic_is_not(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "1")
    monkeypatch.setenv("OMBRE_PG_RECALL_MODE", "hnsw")
    monkeypatch.setenv("OMBRE_PG_VECTOR_SHADOW", "1")
    cursor = _FakeCursor(ann_rows=[("b1", 0.1)])
    connector = _install_pg(monkeypatch, cursor)
    engine = _make_engine(tmp_path, _RecordingClient(_ok_response(QUERY_VECTOR)))
    scheduled = []

    def record_shadow(*args, **kwargs):
        scheduled.append((args, kwargs))

    monkeypatch.setattr(engine, "_schedule_pg_vector_shadow", record_shadow)

    await engine.search_similar_with_status("q", top_k=1)
    assert len(scheduled) == 1, "ordinary path keeps calling the shadow"
    assert "options" not in connector.connect_calls[0]

    scheduled.clear()
    connector.connect_calls.clear()
    await engine.search_similar_diagnostic(
        "q", top_k=1, target_ids=["b1"]
    )
    assert scheduled == [], "diagnostic must not schedule the shadow"
    assert (
        connector.connect_calls[0].get("options")
        == "-c default_transaction_read_only=on"
    )
