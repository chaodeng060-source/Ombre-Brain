from __future__ import annotations

import json
import sqlite3

import pytest

from embedding_engine import EmbeddingEngine


def _engine(tmp_path) -> EmbeddingEngine:
    return EmbeddingEngine(
        {
            "buckets_dir": str(tmp_path),
            "embedding": {
                "enabled": True,
                "api_key": "test-only",
                "base_url": "https://example.invalid/v1",
            },
        }
    )


@pytest.mark.asyncio
async def test_multichunk_embedding_is_all_or_nothing(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(engine, "_split_into_chunks", lambda _content: ["one", "two"])

    with sqlite3.connect(engine.db_path) as conn:
        conn.execute(
            "INSERT INTO embeddings(bucket_id, embedding, updated_at) VALUES (?, ?, ?)",
            ("bucket-1", json.dumps([[9.0, 9.0]]), "before"),
        )
        conn.commit()

    answers = iter([[1.0, 1.0], []])

    async def fake_generate(_text):
        return next(answers)

    monkeypatch.setattr(engine, "_generate_embedding", fake_generate)

    assert await engine.generate_and_store("bucket-1", "body") is False
    with sqlite3.connect(engine.db_path) as conn:
        stored = conn.execute(
            "SELECT embedding, updated_at FROM embeddings WHERE bucket_id = ?",
            ("bucket-1",),
        ).fetchone()
    assert stored == (json.dumps([[9.0, 9.0]]), "before")


@pytest.mark.asyncio
async def test_multichunk_embedding_commits_only_after_every_chunk(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(engine, "_split_into_chunks", lambda _content: ["one", "two"])

    answers = iter([[1.0, 0.0], [0.0, 1.0]])

    async def fake_generate(_text):
        return next(answers)

    monkeypatch.setattr(engine, "_generate_embedding", fake_generate)

    assert await engine.generate_and_store("bucket-1", "body") is True
    with sqlite3.connect(engine.db_path) as conn:
        stored = conn.execute(
            "SELECT embedding FROM embeddings WHERE bucket_id = ?",
            ("bucket-1",),
        ).fetchone()
    assert json.loads(stored[0]) == [[1.0, 0.0], [0.0, 1.0]]
