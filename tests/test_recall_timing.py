import asyncio
import json
import logging

import pytest

import server
from embedding_engine import EmbeddingEngine
from recall_timing import (
    begin_recall_timing,
    finish_recall_timing,
    record_recall_stage,
    reset_recall_timing,
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
