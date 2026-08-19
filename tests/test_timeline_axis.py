from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from bucket_manager import bucket_revision_hash
from timeline_axis import (
    OTHER_THREAD,
    normalize_thread,
    normalize_thread_hint,
    plan_timeline_assignments,
    run_timeline_sweep,
    timeline_neighbors,
)


def _bucket(
    bucket_id: str,
    *,
    event_at: str,
    thread: str = "",
    source_session: str = "session-a",
    source_event_ids: tuple[str, ...] = ("event-1",),
    bucket_type: str = "dynamic",
    relations: tuple[dict, ...] = (),
) -> dict:
    metadata = {
        "id": bucket_id,
        "name": bucket_id,
        "event_at": event_at,
        "created": event_at,
        "type": bucket_type,
        "source_session": source_session,
        "source_event_ids": list(source_event_ids),
        "relations": list(relations),
    }
    if thread:
        metadata["thread"] = thread
    return {"id": bucket_id, "metadata": metadata, "content": bucket_id}


def test_thread_contract_keeps_labels_but_rejects_machine_candidates():
    assert normalize_thread(" 工程演进 ") == "工程演进"
    assert normalize_thread("") == OTHER_THREAD
    assert normalize_thread_hint("room:main") == OTHER_THREAD
    assert normalize_thread_hint("task_wr_976a40a17657_01") == OTHER_THREAD
    assert normalize_thread_hint("event:1234") == OTHER_THREAD
    assert normalize_thread_hint("relation:1234") == OTHER_THREAD
    assert normalize_thread_hint("episode:1234") == OTHER_THREAD


def test_unreviewed_events_episodes_kin_and_hints_remain_in_other():
    base = datetime(2026, 8, 12, 10, 0, 0)
    buckets = [
        _bucket(
            "a",
            event_at=base.isoformat(),
            relations=(
                {"type": "kin", "target": "b", "note": "同一事件"},
            ),
        ),
        _bucket("b", event_at=(base + timedelta(hours=1)).isoformat()),
        _bucket(
            "episode",
            event_at=(base + timedelta(hours=2)).isoformat(),
            bucket_type="episode",
        ),
    ]

    plan = plan_timeline_assignments(
        buckets,
        thread_hints_by_bucket={"a": "辩论日", "b": "基础设施"},
    )

    assert {item.thread for item in plan.assignments} == {OTHER_THREAD}
    assert plan.candidate_hint_count == 2
    assert plan.new_line_count == 0
    assert plan.orphan_count == 3


def test_explicit_review_is_the_only_label_seed():
    buckets = [
        _bucket("a", event_at="2026-07-01T09:00:00"),
        _bucket("b", event_at="2026-07-20T09:00:00"),
        _bucket("orphan", event_at="2026-08-12T09:00:00"),
    ]
    plan = plan_timeline_assignments(
        buckets,
        reviewed_threads_by_bucket={"a": "工程演进", "b": "工程演进"},
    )

    assert {item.bucket_id: item.thread for item in plan.assignments} == {
        "a": "工程演进",
        "b": "工程演进",
        "orphan": OTHER_THREAD,
    }
    assert plan.assigned_count == 2
    assert plan.new_line_count == 1
    assert plan.orphan_count == 1


def test_typed_in_thread_propagates_one_named_anchor_only():
    buckets = [
        _bucket(
            "anchor",
            event_at="2026-07-01T09:00:00",
            thread="工程演进",
            relations=({"type": "in_thread", "target": "middle"},),
        ),
        _bucket(
            "middle",
            event_at="2026-07-20T09:00:00",
            relations=({"type": "in_thread", "target": "tail"},),
        ),
        _bucket("tail", event_at="2026-08-01T09:00:00"),
    ]

    plan = plan_timeline_assignments(buckets)

    assert {item.bucket_id: item.thread for item in plan.assignments} == {
        "anchor": "工程演进",
        "middle": "工程演进",
        "tail": "工程演进",
    }
    assert plan.assigned_count == 2


def test_unanchored_or_conflicting_in_thread_components_fail_closed():
    unanchored = [
        _bucket(
            "a",
            event_at="2026-07-01T09:00:00",
            relations=({"type": "in_thread", "target": "b"},),
        ),
        _bucket("b", event_at="2026-07-20T09:00:00"),
    ]
    assert {
        item.thread for item in plan_timeline_assignments(unanchored).assignments
    } == {OTHER_THREAD}

    conflicted = [
        _bucket(
            "a",
            event_at="2026-07-01T09:00:00",
            thread="工程演进",
            relations=({"type": "in_thread", "target": "middle"},),
        ),
        _bucket(
            "middle",
            event_at="2026-07-20T09:00:00",
            relations=({"type": "in_thread", "target": "b"},),
        ),
        _bucket(
            "b",
            event_at="2026-08-01T09:00:00",
            thread="记忆治理",
        ),
    ]
    assigned = {
        item.bucket_id: item.thread
        for item in plan_timeline_assignments(conflicted).assignments
    }
    assert assigned == {
        "a": "工程演进",
        "b": "记忆治理",
        "middle": OTHER_THREAD,
    }


def test_neighbors_are_adjacent_bounded_and_other_never_propagates():
    buckets = [
        _bucket("a", event_at="2026-07-01T09:00:00", thread="工程演进"),
        _bucket("b", event_at="2026-07-20T09:00:00", thread="工程演进"),
        _bucket("c", event_at="2026-08-12T09:00:00", thread="工程演进"),
        _bucket(
            "orphan",
            event_at="2026-08-01T09:00:00",
            thread=OTHER_THREAD,
        ),
    ]

    found = timeline_neighbors(
        buckets,
        ["b"],
        neighbor_window=2,
        max_results=2,
    )
    assert [item.bucket_id for item in found] == ["a", "c"]
    assert [item.direction for item in found] == ["previous", "next"]
    assert timeline_neighbors(
        buckets,
        ["orphan"],
        neighbor_window=2,
        max_results=2,
    ) == []


@pytest.mark.asyncio
async def test_bucket_thread_write_preserves_activity_metadata(bucket_mgr):
    bucket_id = await bucket_mgr.create(
        content="一条记忆",
        name="thread-test",
    )
    before = await bucket_mgr.get(bucket_id)
    assert before["metadata"]["thread"] == OTHER_THREAD

    assert await bucket_mgr.set_thread(
        bucket_id,
        "工程演进",
        actor="test:timeline",
    )
    after = await bucket_mgr.get(bucket_id)

    assert after["metadata"]["thread"] == "工程演进"
    assert after["metadata"]["last_active"] == before["metadata"]["last_active"]
    assert after["metadata"]["activation_count"] == before["metadata"]["activation_count"]


@pytest.mark.asyncio
async def test_sweep_backfills_other_and_reviewed_threads_idempotently(bucket_mgr):
    first_id = await bucket_mgr.create(content="第一段", name="first")
    second_id = await bucket_mgr.create(content="第二段", name="second")

    first = await run_timeline_sweep(
        bucket_mgr,
        reviewed_threads_by_bucket={first_id: "工程演进"},
        revision_hash_provider=bucket_revision_hash,
    )
    second = await run_timeline_sweep(
        bucket_mgr,
        reviewed_threads_by_bucket={first_id: "工程演进"},
        revision_hash_provider=bucket_revision_hash,
    )

    assert first.assigned_count == 1
    assert first.updated_count == 1
    assert first.orphan_count == 1
    assert second.assigned_count == 0
    assert second.updated_count == 0


@pytest.mark.asyncio
async def test_sweep_writes_missing_legacy_thread_as_other():
    bucket = _bucket("legacy", event_at="2026-06-01T00:00:00+08:00")
    writes = []

    class Manager:
        async def list_all(self, include_archive=False):
            return [bucket]

        async def set_thread(self, bucket_id, thread, **_kwargs):
            writes.append((bucket_id, thread))
            bucket["metadata"]["thread"] = thread
            return True

    report = await run_timeline_sweep(Manager())

    assert writes == [("legacy", OTHER_THREAD)]
    assert report.updated_count == 1
    assert report.assigned_count == 0
    assert report.orphan_count == 1


@pytest.mark.asyncio
async def test_thread_write_rejects_stale_revision(bucket_mgr):
    bucket_id = await bucket_mgr.create(
        content="最初正文",
        name="revision-test",
    )
    before = await bucket_mgr.get(bucket_id)
    revision = bucket_revision_hash(before["content"], before["metadata"])
    assert await bucket_mgr.update(bucket_id, content="正文已经变化")

    assert not await bucket_mgr.set_thread(
        bucket_id,
        "不该覆盖",
        actor="test:stale-timeline",
        expected_revision_hash=revision,
    )
    after = await bucket_mgr.get(bucket_id)
    assert after["metadata"]["thread"] == OTHER_THREAD
