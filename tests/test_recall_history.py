from __future__ import annotations

import json
import os
import time

import server
from recall_history import (
    JsonFileRecallHistory,
    default_content_fingerprint,
    recall_identity,
)


def _bucket(bucket_id: str, content: str) -> dict:
    return {"id": bucket_id, "content": content, "metadata": {"name": bucket_id}}


def test_history_is_session_scoped_hash_only_and_expires(tmp_path):
    history = JsonFileRecallHistory(tmp_path, ttl_seconds=60)
    key = recall_identity("curated", "bucket-42")
    history.mark("private/session", [key])

    assert history.seen("private/session", [key]) == {key}
    assert history.seen("other/session", [key]) == set()

    state = next(tmp_path.glob("session-*.json"))
    raw = state.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["version"] == 1
    assert "private/session" not in raw
    assert "bucket-42" not in raw

    old = time.time() - 61
    os.utime(state, (old, old))
    assert history.seen("private/session", [key]) == set()


def test_history_read_does_not_create_state(tmp_path):
    history = JsonFileRecallHistory(tmp_path / "missing")
    key = recall_identity("curated", "bucket-1")

    assert history.seen("session-a", [key]) == set()
    assert not history.state_dir.exists()


def test_ombre_helpers_filter_previous_winner_and_backfill(tmp_path, monkeypatch):
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    candidates = [
        _bucket("one", "first sufficiently long memory"),
        _bucket("two", "second sufficiently long memory"),
    ]

    server._remember_session_seen_ids("session-a", ["one"])

    assert [bucket["id"] for bucket in server._filter_session_seen(candidates, "session-a")] == ["two"]
    assert [bucket["id"] for bucket in server._filter_session_seen(candidates, "session-b")] == ["one", "two"]


def test_same_turn_content_duplicates_with_different_ids_share_one_slot():
    event = "朝灯说完整稿必须真正参与每轮主动召回"
    buckets = [
        _bucket("one", f"[knowledge_base] {event}"),
        _bucket("two", event),
        _bucket("three", "这是另一条足够长而且不同的候选记忆"),
    ]

    result, suppressed, errors = server._dedupe_recall_content(buckets)

    assert [bucket["id"] for bucket in result] == ["one", "three"]
    assert suppressed == 1
    assert errors == 0
    assert result[0]["metadata"]["content_duplicates_merged"] == 1


def test_content_fingerprint_keeps_digits_and_short_snippets_distinct():
    assert default_content_fingerprint("心率是70到93次") != default_content_fingerprint("心率是71到93次")
    assert default_content_fingerprint("same") is None


def test_history_read_failure_fails_open(monkeypatch):
    class BrokenHistory:
        def seen(self, _session_id, _keys):
            raise OSError("unavailable")

    monkeypatch.setattr(server, "_session_recall_history", lambda: BrokenHistory())
    buckets = [_bucket("one", "first sufficiently long memory")]
    assert server._filter_session_seen(buckets, "session-a") == buckets
