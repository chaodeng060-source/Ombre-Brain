import asyncio
import json
import os

import pytest

from backfill_created import infer_time_metadata
from bucket_manager import BucketManager


def _audited_manager(test_config):
    config = dict(test_config)
    config["audit"] = {"enabled": True}
    return BucketManager(config)


@pytest.mark.asyncio
async def test_create_writes_split_time_model_and_committed_audit(test_config):
    manager = _audited_manager(test_config)
    bucket_id = await manager.create(
        content="发生过的事",
        name="时间模型",
        domain=["测试"],
        event_at="2026-05-30",
        date_source="user",
        date_confidence=0.95,
        actor="test:user",
    )

    bucket = await manager.get(bucket_id)
    meta = bucket["metadata"]
    assert meta["event_at"] == "2026-05-30T00:00:00"
    assert meta["created"] == meta["event_at"]
    assert meta["recorded_at"]
    assert meta["date_precision"] == "day"
    assert meta["date_source"] == "user"
    assert meta["date_confidence"] == 0.95

    events = manager.audit_log.list_events(bucket_id)
    assert len(events) == 1
    assert events[0]["actor"] == "test:user"
    assert events[0]["action"] == "create"
    assert events[0]["status"] == "committed"
    after = json.loads(events[0]["after_json"])
    assert after["metadata"]["event_at"] == "2026-05-30T00:00:00"


@pytest.mark.asyncio
async def test_default_event_time_is_marked_as_low_confidence_record_time(test_config):
    manager = _audited_manager(test_config)
    bucket_id = await manager.create(content="即时记录")
    meta = (await manager.get(bucket_id))["metadata"]

    assert meta["event_at"] == meta["recorded_at"]
    assert meta["date_source"] == "recorded_at_default"
    assert meta["date_confidence"] == 0.5


@pytest.mark.asyncio
async def test_update_relation_resolve_and_delete_are_audited(test_config):
    manager = _audited_manager(test_config)
    source = await manager.create(content="source", domain=["测试"])
    target = await manager.create(content="target", domain=["测试"])

    assert await manager.update(
        source,
        actor="test:editor",
        event_at="2026-06-01T12:30",
        importance=8,
    )
    assert await manager.add_relation(
        source, target, "explains", actor="test:editor"
    )
    assert await manager.update(source, actor="test:editor", resolved=True)
    assert await manager.remove_relation(
        source, target, "explains", actor="test:editor"
    ) == 1
    assert await manager.delete(source, actor="test:editor")

    actions = [
        event["action"] for event in manager.audit_log.list_events(source)
    ]
    assert actions == [
        "create",
        "update",
        "add_relation",
        "resolve",
        "remove_relation",
        "delete",
    ]
    assert all(
        event["status"] == "committed"
        for event in manager.audit_log.list_events(source)
    )


@pytest.mark.asyncio
async def test_failed_atomic_update_keeps_original_and_marks_audit_failed(
    test_config, monkeypatch
):
    manager = _audited_manager(test_config)
    bucket_id = await manager.create(content="original", domain=["测试"])

    def fail_write(_path, _post):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(manager, "_atomic_write_post", fail_write)
    assert not await manager.update(
        bucket_id,
        actor="test:failure",
        content="corrupt-me",
    )

    bucket = await manager.get(bucket_id)
    assert bucket["content"] == "original"
    events = manager.audit_log.list_events(bucket_id)
    assert events[-1]["status"] == "failed"
    assert "simulated disk failure" in events[-1]["error"]


@pytest.mark.asyncio
async def test_concurrent_updates_leave_a_complete_bucket(test_config):
    manager = _audited_manager(test_config)
    bucket_id = await manager.create(content="start", domain=["测试"])

    await asyncio.gather(
        *[
            manager.update(
                bucket_id,
                actor=f"test:worker:{index}",
                content=f"value-{index}",
                importance=(index % 10) + 1,
            )
            for index in range(12)
        ]
    )

    bucket = await manager.get(bucket_id)
    assert bucket is not None
    assert bucket["content"].startswith("value-")
    assert 1 <= bucket["metadata"]["importance"] <= 10
    assert not [
        name
        for root, _, files in os.walk(test_config["buckets_dir"])
        for name in files
        if name.endswith(".tmp")
    ]
    committed_updates = [
        event
        for event in manager.audit_log.list_events(bucket_id)
        if event["action"] == "update" and event["status"] == "committed"
    ]
    assert len(committed_updates) == 12


def test_time_migration_marks_legacy_created_as_low_confidence():
    updates = infer_time_metadata(
        {
            "created": "2026-05-20T09:00:00",
            "name": "没有日期证据",
            "tags": [],
        },
        recorded_at="2026-06-18T20:00:00",
    )
    assert updates["event_at"] == "2026-05-20T09:00:00"
    assert updates["recorded_at"] == "2026-06-18T20:00:00"
    assert updates["date_source"] == "legacy_created"
    assert updates["date_confidence"] == 0.3
