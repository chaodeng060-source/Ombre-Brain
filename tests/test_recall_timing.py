import asyncio
import json
import logging

import pytest

import server
from embedding_engine import EmbeddingEngine
from recall_timing import (
    begin_recall_timing,
    finish_recall_timing,
    finish_recall_stage,
    record_recall_stage,
    reset_recall_timing,
    set_recall_partial_result,
    start_recall_stage,
)


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def test_timing_receipt_accumulates_calls_without_content():
    token = begin_recall_timing()
    try:
        record_recall_stage("embedding", 0.012)
        record_recall_stage("embedding", 0.003)
        receipt = finish_recall_timing(status="ok", partial=False)
    finally:
        reset_recall_timing(token)

    assert receipt["schema_version"] == 1
    assert receipt["status"] == "ok"
    assert receipt["partial"] is False
    assert receipt["stages"]["embedding"] == {
        "elapsed_ms": 15.0,
        "calls": 2,
    }
    assert "query" not in json.dumps(receipt)


@pytest.mark.asyncio
async def test_embedding_engine_splits_remote_and_local_timing(tmp_path, monkeypatch):
    engine = object.__new__(EmbeddingEngine)
    engine.enabled = True
    engine.db_path = str(tmp_path / "embeddings.db")

    import sqlite3

    with sqlite3.connect(engine.db_path) as conn:
        conn.execute(
            "CREATE TABLE embeddings (bucket_id TEXT PRIMARY KEY, embedding TEXT)"
        )
        conn.execute(
            "INSERT INTO embeddings(bucket_id, embedding) VALUES (?, ?)",
            ("bucket-a", "[1.0, 0.0]"),
        )

    async def fake_embedding(_query):
        await asyncio.sleep(0)
        return [1.0, 0.0], "ok"

    monkeypatch.setattr(engine, "_generate_embedding_with_status", fake_embedding)

    token = begin_recall_timing()
    try:
        results, status = await engine.search_similar_with_status("secret query", top_k=1)
        receipt = finish_recall_timing(status="ok", partial=False)
    finally:
        reset_recall_timing(token)

    assert status == "ok"
    assert results == [("bucket-a", 1.0)]
    assert receipt["stages"]["embedding"]["calls"] == 1
    assert receipt["stages"]["vector_retrieval"]["calls"] == 1
    assert "secret query" not in json.dumps(receipt)


@pytest.mark.asyncio
async def test_api_breath_returns_and_logs_structured_timing(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="ombre_brain")

    async def fake_breath(**_kwargs):
        record_recall_stage("expansion", 0.002)
        return "memory result"

    monkeypatch.setattr(server, "breath", fake_breath)
    response = await server.api_breath(
        _JsonRequest({"query": "private words", "policy": "conversation"})
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["raw"] == "memory result"
    assert payload["policy"] == "conversation"
    assert payload["partial"] is False
    assert payload["timing"]["status"] == "ok"
    assert payload["timing"]["stages"]["expansion"] == {
        "elapsed_ms": 2.0,
        "calls": 1,
    }
    assert "breath_timing=" in caplog.text
    assert "private words" not in caplog.text


@pytest.mark.asyncio
async def test_api_breath_logs_timing_before_propagating_cancellation(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="ombre_brain")
    entered = asyncio.Event()

    async def hanging_breath(**_kwargs):
        record_recall_stage("expansion", 0.003)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "breath", hanging_breath)
    task = asyncio.create_task(
        server.api_breath(_JsonRequest({"query": "private cancellation"}))
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert '"status": "cancelled"' in caplog.text
    assert '"partial": true' in caplog.text
    assert '"elapsed_ms": 3.0' in caplog.text
    assert "private cancellation" not in caplog.text


@pytest.mark.asyncio
async def test_api_breath_deadline_returns_available_partial(monkeypatch):
    cancelled = False

    async def hanging_breath(**_kwargs):
        nonlocal cancelled
        set_recall_partial_result("[bucket_id:abc123] available candidate")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    monkeypatch.setattr(server, "breath", hanging_breath)
    monkeypatch.setattr(server, "_breath_deadline_sec", lambda: 0.01)
    response = await server.api_breath(_JsonRequest({"query": "deadline query"}))

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["raw"] == "[bucket_id:abc123] available candidate"
    assert payload["partial"] is True
    assert payload["timing"]["status"] == "deadline"
    assert payload["timing"]["partial"] is True
    assert cancelled is True


@pytest.mark.asyncio
async def test_api_breath_deadline_without_candidate_returns_empty(monkeypatch):
    async def hanging_breath(**_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "breath", hanging_breath)
    monkeypatch.setattr(server, "_breath_deadline_sec", lambda: 0.01)
    response = await server.api_breath(_JsonRequest({"query": "deadline query"}))

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["raw"] == "未找到相关记忆。"
    assert payload["partial"] is True
    assert payload["timing"]["status"] == "deadline"


def test_breath_deadline_is_bounded(monkeypatch):
    monkeypatch.setenv("OMBRE_BREATH_DEADLINE_SEC", "99")
    assert server._breath_deadline_sec() == 13.0
    monkeypatch.setenv("OMBRE_BREATH_DEADLINE_SEC", "bad")
    assert server._breath_deadline_sec() == 11.0


def test_local_partial_renderer_keeps_order_limit_and_redacts_secrets():
    candidates = [
        {
            "id": "aaa111",
            "content": json.dumps({"summary": "first api_key=sk-secret123456789"}),
            "metadata": {"name": "first", "domain": []},
        },
        {
            "id": "bbb222",
            "content": json.dumps({"summary": "second"}),
            "metadata": {"name": "second", "domain": []},
        },
    ]

    text = server._local_partial_recall_text(
        candidates,
        max_results=1,
        max_tokens=1000,
        state_profile={},
    )

    assert "[bucket_id:aaa111]" in text
    assert "[bucket_id:bbb222]" not in text
    assert "sk-secret123456789" not in text
    assert "[REDACTED]" in text


def test_open_stage_is_included_when_deadline_receipt_finishes(monkeypatch):
    ticks = iter((10.0, 11.0, 13.5, 13.5))
    monkeypatch.setattr("recall_timing.time.perf_counter", lambda: next(ticks))
    token = begin_recall_timing()
    try:
        start_recall_stage("assembly")
        receipt = finish_recall_timing(status="deadline", partial=True)
    finally:
        reset_recall_timing(token)

    assert receipt["total_ms"] == 3500.0
    assert receipt["stages"]["assembly"] == {
        "elapsed_ms": 2500.0,
        "calls": 1,
    }
    assert receipt["unattributed_ms"] == 1000.0


def test_finished_stage_is_not_double_counted(monkeypatch):
    ticks = iter((20.0, 21.0, 22.5, 24.0))
    monkeypatch.setattr("recall_timing.time.perf_counter", lambda: next(ticks))
    token = begin_recall_timing()
    try:
        start_recall_stage("assembly")
        finish_recall_stage("assembly")
        receipt = finish_recall_timing(status="ok", partial=False)
    finally:
        reset_recall_timing(token)

    assert receipt["stages"]["assembly"] == {
        "elapsed_ms": 1500.0,
        "calls": 1,
    }
