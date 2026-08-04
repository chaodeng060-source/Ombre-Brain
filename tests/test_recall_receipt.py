from __future__ import annotations

import json
import os

import pytest

import server
from recall_receipt import RecallReceiptConflict, RecallReceiptError, RecallReceiptStore


class _Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class _BucketManager:
    def __init__(self, *, fail_once: str = ""):
        self.fail_once = fail_once
        self.touches = []

    async def get(self, bucket_id):
        return {"id": bucket_id, "metadata": {"activation_count": 1}}

    async def touch(self, bucket_id, actor="", *, ripple=True, raise_on_error=False):
        self.touches.append((bucket_id, actor, ripple, raise_on_error))
        if bucket_id == self.fail_once:
            self.fail_once = ""
            raise RuntimeError("transient")


def _store(tmp_path):
    store = RecallReceiptStore(tmp_path / "buckets")
    store.initialize()
    return store


def test_receipt_store_is_idempotent_and_content_free(tmp_path):
    store = _store(tmp_path)
    begun = store.begin("event-1", ["bucket-a", "bucket-a", "bucket-b"], "test")
    assert begun == {
        "duplicate": False,
        "pending": ["bucket-a", "bucket-b"],
        "total": 2,
    }
    store.mark_applied("event-1", "bucket-a")
    store.mark_applied("event-1", "bucket-b")
    assert store.status("event-1") == {"status": "complete", "applied": 2, "pending": 0}
    assert store.begin("event-1", ["bucket-a", "bucket-b"], "retry") == {
        "duplicate": True,
        "pending": [],
        "total": 2,
    }
    assert os.stat(store.directory).st_mode & 0o777 == 0o700
    assert os.stat(store.path).st_mode & 0o777 == 0o600


def test_receipt_store_rejects_event_payload_conflict(tmp_path):
    store = _store(tmp_path)
    store.begin("event-1", ["bucket-a"], "test")
    with pytest.raises(RecallReceiptConflict):
        store.begin("event-1", ["bucket-b"], "test")


def test_receipt_store_rejects_symlinked_sidecar(tmp_path):
    buckets = tmp_path / "buckets"
    buckets.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (buckets / ".recall_receipts").symlink_to(target, target_is_directory=True)
    with pytest.raises(RecallReceiptError, match="unsafe receipt directory"):
        RecallReceiptStore(buckets).initialize()


def test_failed_item_remains_resumable(tmp_path):
    store = _store(tmp_path)
    store.begin("event-1", ["bucket-a", "bucket-b"], "test")
    store.mark_applied("event-1", "bucket-a")
    store.mark_failed("event-1", "bucket-b", RuntimeError("transient"))
    assert store.status("event-1") == {"status": "pending", "applied": 1, "pending": 1}
    retry = store.begin("event-1", ["bucket-a", "bucket-b"], "retry")
    assert retry["duplicate"] is False
    assert retry["pending"] == ["bucket-b"]


@pytest.mark.asyncio
async def test_api_receipt_retries_only_failed_item_without_temporal_ripple(tmp_path, monkeypatch):
    store = _store(tmp_path)
    manager = _BucketManager(fail_once="bucket-b")
    monkeypatch.setattr(server, "_get_recall_receipt_store", lambda: store)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    request = _Request({
        "event_id": "event-1",
        "bucket_ids": ["bucket-a", "bucket-b"],
        "source": "test",
    })

    first = await server.api_recall_receipt(request)
    first_body = json.loads(first.body)
    assert first_body["ok"] is False
    assert first_body["applied"] == 1
    assert first_body["pending"] == 1

    second = await server.api_recall_receipt(request)
    second_body = json.loads(second.body)
    assert second_body["ok"] is True
    assert second_body["status"] == "complete"
    assert [row[0] for row in manager.touches] == ["bucket-a", "bucket-b", "bucket-b"]
    assert all(row[2] is False for row in manager.touches)
    assert all(row[3] is True for row in manager.touches)


@pytest.mark.asyncio
async def test_api_receipt_returns_conflict_without_touching_bucket(tmp_path, monkeypatch):
    store = _store(tmp_path)
    manager = _BucketManager()
    monkeypatch.setattr(server, "_get_recall_receipt_store", lambda: store)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    first = _Request({"event_id": "event-1", "bucket_ids": ["bucket-a"]})
    second = _Request({"event_id": "event-1", "bucket_ids": ["bucket-b"]})
    assert (await server.api_recall_receipt(first)).status_code == 200
    conflict = await server.api_recall_receipt(second)
    assert conflict.status_code == 409
    assert [row[0] for row in manager.touches] == ["bucket-a"]
