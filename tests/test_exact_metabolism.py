"""Synthetic-only contract tests; no server, NAS, models or private vaults."""

from __future__ import annotations

import asyncio
import copy
import json
import multiprocessing
import os
import socket
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import frontmatter
import pytest

from bucket_manager import BucketManager, bucket_revision_hash
from exact_metabolism import (
    DEFAULT_MAX_UPDATES,
    plan_exact_consolidation,
    recovery_records_from_audit,
    restore_exact_consolidation,
    run_exact_consolidation,
)
from utils import PROTECTED_RESOLVE_DOMAINS


BODY = "原始事件：我在白板上画了三条线。\n逐字保留这一行。"


def memory(bucket_id="a", *, content=BODY, **metadata):
    meta = {
        "id": bucket_id, "name": f"name-{bucket_id}", "type": "dynamic",
        "tags": ["白板"], "domain": ["工程"], "world": "现实",
        "created": "2026-10-22T01:02:03+00:00",
        "event_at": "2026-10-22T01:02:03+00:00",
        "recorded_at": "2026-10-22T01:02:04+00:00",
        "last_active": "2026-10-22T01:02:04+00:00", "activation_count": 1,
        "source_session": "session-1", "source_event_ids": ["event-1"],
        "authored_by": "speaker-1", "fact_status": "current", "thread": "other",
        "retrieval_keys": ["三条线"],
    }
    meta.update(metadata)
    return {"id": bucket_id, "content": content, "metadata": meta}


def raw_ref():
    return {"database": "/synthetic/raw.sqlite3", "table": "raw_events",
            "raw_events": [{"session_id": "session-1", "source_event_id": "event-1", "row_id": 1}]}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("isolated exact-metabolism tests must not use network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def exact_config(tmp_path):
    root = tmp_path / "synthetic-vault"
    root.mkdir()
    return {"buckets_dir": str(root), "audit": {"enabled": True},
            "wikilink": {"enabled": False}}


@pytest.fixture
def exact_mgr(exact_config):
    return BucketManager(exact_config)


def seed(manager, *buckets):
    directory = Path(manager.dynamic_dir) / "synthetic"
    directory.mkdir(parents=True, exist_ok=True)
    for bucket in buckets:
        path = directory / f"copy_{bucket['id']}.md"
        post = frontmatter.Post(bucket["content"], **bucket["metadata"])
        path.write_text(frontmatter.dumps(post), encoding="utf-8")
    manager.invalidate_list_all_cache()


def revision(bucket):
    return bucket_revision_hash(bucket["content"], bucket["metadata"])


def test_plan_is_readable_deterministic_and_does_not_copy_bodies():
    rows = [memory("c"), memory("a"), memory("b", recorded_at="later",
            last_active="later", activation_count=9, dehydrated_summary="cache only")]
    before = copy.deepcopy(rows)
    plan = plan_exact_consolidation(rows)
    assert plan == plan_exact_consolidation(reversed(rows))
    assert plan["available"] == plan["planned"] == 2
    assert [(e["winner_id"], e["loser_id"]) for e in plan["entries"]] == [("a", "b"), ("a", "c")]
    assert plan["entries"][0]["source_fields"] == ["source_event_ids"]
    assert BODY not in json.dumps(plan, ensure_ascii=False)
    assert rows == before


@pytest.mark.parametrize("changes", [
    {"source_event_ids": ["event-2"]}, {"source_chunk_ids": ["chunk-2"]},
    {"source_session": "session-2"}, {"source_ref": "raw:another-message"},
    {"world": "虚构世界"}, {"authored_by": "speaker-2"},
    {"fact_status": "historical"}, {"fact_status": "contested"},
    {"tags": ["另一检索词"]}, {"domain": ["工作"]},
    {"retrieval_keys": ["原始事件"]}, {"thread": "另一叙事线"},
    {"relations": [{"target": "other", "type": "explains"}]},
    {"event_at": "2026-10-21T01:02:03+00:00"},
    {"created": "2026-10-21T01:02:03+00:00"},
    {"new_semantic_field": "retain unknown metadata"},
])
def test_all_semantic_metadata_must_match(changes):
    assert plan_exact_consolidation([memory("a"), memory("b", **changes)])["planned"] == 0


@pytest.mark.parametrize("content", [BODY + " ", " " + BODY, BODY + "\n", BODY.replace("\n", "\r\n"),
                                      BODY.replace("三", "四"), BODY.replace("：", ":")])
def test_body_comparison_does_not_normalize_or_rewrite(content):
    assert plan_exact_consolidation([memory("a"), memory("b", content=content)])["planned"] == 0


def test_type_and_list_order_are_not_coerced():
    for left, right in [(True, 1), (1, "1"), (1, 1.0), (date(2026, 10, 22), "2026-10-22"),
                        (["one", "two"], ["two", "one"])]:
        assert plan_exact_consolidation([memory("a", detail=left), memory("b", detail=right)])["planned"] == 0


def test_malformed_metadata_does_not_interrupt_valid_exact_group():
    malformed = memory("bad")
    malformed["metadata"][123] = "not a string key"
    result = plan_exact_consolidation([malformed, memory("a"), memory("b")])
    assert result["excluded"] == {"invalid_metadata": 1}
    assert result["planned"] == 1


@pytest.mark.parametrize("evidence", [
    {}, {"source_event_ids": []}, {"source_event_ids": ["unknown"]},
    {"source_event_ids": "event-1"}, {"source_event_ids": ["event-1", "event-1"]},
    {"source_chunk_ids": ["unknown"]}, {"source_ref": "unknown"},
    {"source_ref": {"raw_events": []}},
    {"source_ref": {"database": "/synthetic/raw.sqlite3", "table": "raw_events",
                    "raw_events": [{"session_id": "unknown", "source_event_id": "event-1"}]}},
])
def test_identical_words_without_explicit_raw_event_evidence_are_not_duplicates(evidence):
    rows = [memory("a"), memory("b")]
    for row in rows:
        row["metadata"].pop("source_event_ids")
        row["metadata"].update(evidence)
    plan = plan_exact_consolidation(rows)
    assert plan["planned"] == 0
    assert plan["excluded"] == {"no_explicit_source": 2}


@pytest.mark.parametrize("evidence", [
    {"source_event_ids": ["event-1"]}, {"source_chunk_ids": ["chunk-1"]},
    {"source_ref": "raw:session-1/event-1"}, {"source_ref": raw_ref()},
])
def test_supported_exact_provenance_forms(evidence):
    rows = [memory("a"), memory("b")]
    for row in rows:
        row["metadata"].pop("source_event_ids")
        row["metadata"].update(copy.deepcopy(evidence))
    assert plan_exact_consolidation(rows)["planned"] == 1


@pytest.mark.parametrize("changes", [
    {"type": "feel"}, {"type": "permanent"}, {"type": "archived"}, {"type": "episode"},
    {"type": "saga"}, {"pinned": True}, {"protected": True}, {"resolved": True},
    {"digested": True}, {"permanent": True}, {"nsfw": True}, {"is_nsfw": True},
    {"e_authored_by": "哥哥"}, {"e_initial_priority": 80}, {"e_axis": None},
    {"axis": "E"}, {"tags": ["experience"]},
    {"world": "涩涩"}, {"world": "NSFW"}, {"protected_world": True},
    {"exact_duplicate_of": "existing"},
] + [{"domain": [d]} for d in sorted(PROTECTED_RESOLVE_DOMAINS)])
def test_exemptions_apply_to_both_winner_and_loser(changes):
    assert plan_exact_consolidation([memory("a", **changes), memory("b", **changes)])["planned"] == 0


def test_additional_protection_budget_and_duplicate_ids():
    assert plan_exact_consolidation([memory("a"), memory("b")], protected_domains=["工程"])["planned"] == 0
    assert plan_exact_consolidation([memory("a"), memory("b")], protected_worlds=["现实"])["planned"] == 0
    rows = [memory(f"{n:03d}") for n in range(DEFAULT_MAX_UPDATES + 6)]
    plan = plan_exact_consolidation(rows)
    assert plan["planned"] == DEFAULT_MAX_UPDATES
    assert plan["deferred"] == 5
    assert plan_exact_consolidation(rows, max_updates=0)["planned"] == 0
    assert plan_exact_consolidation([memory("a"), memory("a"), memory("b")])["planned"] == 0
    with pytest.raises(ValueError, match="max_updates"):
        plan_exact_consolidation(rows, max_updates=-1)


@pytest.mark.asyncio
async def test_only_one_active_entry_body_ids_and_retrieval_metadata_survive(exact_mgr):
    seed(exact_mgr, memory("c"), memory("a"), memory("b", digested=False, exact_duplicate_of=""))
    before = {bid: await exact_mgr.get(bid) for bid in ("a", "b", "c")}
    result = await run_exact_consolidation(exact_mgr)
    assert result["status"] == "completed"
    assert result["applied"] == ["b", "c"]
    assert result.get("consolidated", 0) == 2
    assert result.get("errors") == []
    assert await exact_mgr.get("a") == before["a"]
    for bid in ("b", "c"):
        after = await exact_mgr.get(bid)
        assert after["content"] == before[bid]["content"] == BODY
        assert after["id"] == before[bid]["id"]
        assert after["path"] == before[bid]["path"]
        assert after["metadata"]["digested"] is True
        assert after["metadata"]["exact_duplicate_of"] == "a"
        changed = {k for k in set(before[bid]["metadata"]) | set(after["metadata"])
                   if before[bid]["metadata"].get(k) != after["metadata"].get(k)}
        assert changed == {"digested", "exact_duplicate_of", "last_active"}
    active = [b["id"] for b in await exact_mgr.list_all() if not b["metadata"].get("digested")]
    assert active == ["a"]
    events = exact_mgr.audit_log.list_events()
    assert len(events) == 2
    assert all(e["status"] == "committed" and e["actor"] == "system:exact-metabolism" for e in events)
    assert all(json.loads(e["before_json"])["content"] == BODY for e in events)
    assert BODY not in json.dumps(result["recovery"], ensure_ascii=False)
    again = await run_exact_consolidation(exact_mgr)
    assert again["applied"] == []
    assert again["consolidated"] == 0 and again["errors"] == []
    assert len(exact_mgr.audit_log.list_events()) == 2


@pytest.mark.asyncio
async def test_run_budget_resumes_without_alias_chains(exact_mgr):
    seed(exact_mgr, *(memory(bid) for bid in ("d", "c", "b", "a")))
    for loser in ("b", "c", "d"):
        result = await run_exact_consolidation(exact_mgr, max_updates=1)
        assert result["applied"] == [loser]
        assert (await exact_mgr.get(loser))["metadata"]["exact_duplicate_of"] == "a"
    assert (await run_exact_consolidation(exact_mgr))["applied"] == []


@pytest.mark.asyncio
async def test_later_smaller_id_does_not_replace_active_alias_root(exact_mgr):
    seed(exact_mgr, memory("b"), memory("c"))
    assert (await run_exact_consolidation(exact_mgr))["applied"] == ["c"]
    seed(exact_mgr, memory("a"))
    result = await run_exact_consolidation(exact_mgr)
    assert result["applied"] == ["a"]
    for bid in ("a", "c"):
        assert (await exact_mgr.get(bid))["metadata"]["exact_duplicate_of"] == "b"
    assert not (await exact_mgr.get("b"))["metadata"].get("digested")


@pytest.mark.asyncio
async def test_multiple_prior_alias_roots_reported_without_dead_links(exact_mgr):
    seed(exact_mgr, memory("a"), memory("b"),
         memory("c", digested=True, exact_duplicate_of="a"),
         memory("d", digested=True, exact_duplicate_of="b"))
    result = await run_exact_consolidation(exact_mgr)
    assert result["status"] == "partial"
    assert result["plan"]["blocked_groups"] == [{"ids": ["a", "b"], "reason": "multiple_active_alias_roots"}]
    assert result["consolidated"] == 0
    assert result["errors"] == ["multiple_active_alias_roots:a,b"]
    assert exact_mgr.audit_log.list_events() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_change", ["space", "newline", "crlf"])
async def test_storage_parser_cannot_turn_distinct_raw_bodies_into_exact_duplicates(exact_mgr, raw_change):
    seed(exact_mgr, memory("a"), memory("b"))
    path = Path((await exact_mgr.get("b"))["path"])
    raw = path.read_bytes()
    raw = raw.replace(b"\n", b"\r\n") if raw_change == "crlf" else raw + (b" " if raw_change == "space" else b"\n")
    path.write_bytes(raw)
    assert (await exact_mgr.get("a"))["content"] == (await exact_mgr.get("b"))["content"]
    result = await run_exact_consolidation(exact_mgr)
    assert result["applied"] == []
    assert result["plan"]["excluded"] == {"body_would_normalize": 1}
    assert path.read_bytes() == raw
    assert exact_mgr.audit_log.list_events() == []


@pytest.mark.asyncio
async def test_raw_body_rechecked_under_lease_even_when_parsed_revision_is_equal(exact_mgr, monkeypatch):
    seed(exact_mgr, memory("a"), memory("b"))
    path = Path((await exact_mgr.get("b"))["path"])
    before = revision(await exact_mgr.get("b"))
    original = exact_mgr._maintenance_barrier

    class ExternalEditBeforeLease:
        @asynccontextmanager
        async def exclusive_async(self, **kwargs):
            path.write_bytes(path.read_bytes() + b" ")
            async with original.exclusive_async(**kwargs):
                yield

    monkeypatch.setattr(exact_mgr, "_maintenance_barrier", ExternalEditBeforeLease())
    result = await run_exact_consolidation(exact_mgr)
    assert revision(await exact_mgr.get("b")) == before
    assert result["skipped"] == [{"loser_id": "b", "reason": "body_would_normalize"}]
    assert path.read_bytes().endswith(b" ")
    assert exact_mgr.audit_log.list_events() == []


@pytest.mark.asyncio
async def test_disabled_does_not_even_read_manager():
    result = await run_exact_consolidation(object(), enabled=False)
    assert result["status"] == "disabled"
    assert result["consolidated"] == 0 and result["errors"] == []


@pytest.mark.asyncio
async def test_audit_is_required_for_durable_crash_recovery(exact_mgr):
    seed(exact_mgr, memory("a"), memory("b"))
    exact_mgr.audit_log.enabled = False
    with pytest.raises(ValueError, match="mutation audit"):
        await run_exact_consolidation(exact_mgr)
    assert not (await exact_mgr.get("b"))["metadata"].get("digested")


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_id,changes", [("a", {"content": "winner changed"}),
    ("b", {"content": "loser changed"}), ("a", {"pinned": True}),
    ("b", {"tags": ["new key"]})])
async def test_revalidation_rejects_stale_winner_or_loser(exact_mgr, monkeypatch, changed_id, changes):
    seed(exact_mgr, memory("a"), memory("b"))
    original = exact_mgr._maintenance_barrier

    class ChangeBeforeLease:
        @asynccontextmanager
        async def exclusive_async(self, **kwargs):
            # Simulate an ordinary writer between advisory plan and exclusive acquisition.
            assert await exact_mgr.update(changed_id, actor="ordinary-writer", **changes)
            async with original.exclusive_async(**kwargs):
                yield

    # update itself must still see the real shared barrier.
    changed_barrier = ChangeBeforeLease()
    changed_barrier.shared_async = original.shared_async
    monkeypatch.setattr(exact_mgr, "_maintenance_barrier", changed_barrier)
    result = await run_exact_consolidation(exact_mgr)
    assert result["status"] == "partial"
    assert result["applied"] == []
    assert result["consolidated"] == 0 and result["errors"] == []
    assert result["skipped"] == [{"loser_id": "b", "reason": "changed_since_plan"}]
    assert not (await exact_mgr.get("b"))["metadata"].get("digested")


@pytest.mark.asyncio
async def test_update_gets_cas_and_existing_exclusive_is_reentrant(exact_mgr, monkeypatch):
    seed(exact_mgr, memory("a"), memory("b"))
    expected = revision(await exact_mgr.get("b"))
    original_update = exact_mgr.update
    calls = []

    async def checked_update(bucket_id, **kwargs):
        calls.append(kwargs)
        assert kwargs["expected_revision_hash"] == expected
        assert set(kwargs) == {"actor", "expected_revision_hash", "digested", "exact_duplicate_of"}
        return await original_update(bucket_id, **kwargs)

    monkeypatch.setattr(exact_mgr, "update", checked_update)
    async with exact_mgr._maintenance_barrier.exclusive_async(timeout=1):
        result = await run_exact_consolidation(exact_mgr, actor="test:exact-owner")
    assert result["applied"] == ["b"]
    assert result.get("consolidated", 0) == 1 and not result.get("errors")
    assert len(calls) == 1
    assert exact_mgr.audit_log.list_events()[0]["actor"] == "test:exact-owner"


@pytest.mark.asyncio
async def test_shared_to_exclusive_upgrade_is_reported_without_bypass(exact_mgr):
    seed(exact_mgr, memory("a"), memory("b"))
    async with exact_mgr._maintenance_barrier.shared_async():
        result = await run_exact_consolidation(exact_mgr)
    assert result["status"] == "blocked"
    assert "cannot upgrade shared to exclusive" in result["error"]
    assert result["errors"] == [result["error"]] and result["consolidated"] == 0
    assert exact_mgr.audit_log.list_events() == []


@pytest.mark.asyncio
async def test_competing_task_timeout_is_reported_and_leaves_no_lock(exact_mgr):
    seed(exact_mgr, memory("a"), memory("b"))
    async with exact_mgr._maintenance_barrier.exclusive_async():
        result = await asyncio.create_task(run_exact_consolidation(exact_mgr, lease_timeout=0.05))
    assert result["status"] == "blocked"
    assert "timed out" in result["error"]
    assert result["errors"] == [result["error"]] and result["consolidated"] == 0
    assert (await run_exact_consolidation(exact_mgr))["applied"] == ["b"]


@pytest.mark.asyncio
async def test_cancellation_after_durable_write_recovers_from_audit_and_resumes(exact_mgr, monkeypatch):
    seed(exact_mgr, memory("a"), memory("b"), memory("c"), memory("d"))
    original_update = exact_mgr.update

    async def interrupted_update(*args, **kwargs):
        assert await original_update(*args, **kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(exact_mgr, "update", interrupted_update)
    with pytest.raises(asyncio.CancelledError):
        await run_exact_consolidation(exact_mgr)
    assert (await exact_mgr.get("b"))["metadata"]["digested"] is True
    monkeypatch.setattr(exact_mgr, "update", original_update)
    resumed = await run_exact_consolidation(exact_mgr)
    assert resumed["applied"] == ["c", "d"]
    records = recovery_records_from_audit(exact_mgr.audit_log.list_events())
    assert [r["bucket_id"] for r in records] == ["b", "c", "d"]
    result = await restore_exact_consolidation(exact_mgr, records)
    assert result["restored"] == ["b", "c", "d"]
    for bid in ("b", "c", "d"):
        bucket = await exact_mgr.get(bid)
        assert bucket["content"] == BODY
        assert bucket["metadata"]["digested"] is False
        assert bucket["metadata"]["exact_duplicate_of"] == ""
    assert (await restore_exact_consolidation(exact_mgr, records))["restored"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_journal_commit_interruption_does_not_strand_a_written_copy(exact_mgr, monkeypatch, failure):
    seed(exact_mgr, memory("a"), memory("b"))
    original_commit = exact_mgr.audit_log.commit

    def interrupt(_event_id):
        raise failure("simulated audit commit interruption")

    monkeypatch.setattr(exact_mgr.audit_log, "commit", interrupt)
    if failure is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            await run_exact_consolidation(exact_mgr)
        expected_status = "pending"
    else:
        result = await run_exact_consolidation(exact_mgr)
        assert result["status"] == "partial"
        assert result["applied"] == ["b"]
        assert result["failed"] == [{"loser_id": "b", "reason": "update_failed"}]
        assert result["consolidated"] == 1 and result["errors"] == ["update_failed:b"]
        expected_status = "failed"
    events = exact_mgr.audit_log.list_events()
    assert events[0]["status"] == expected_status
    records = recovery_records_from_audit(events)
    assert len(records) == 1
    assert records[0]["after_revision_hash"] == revision(await exact_mgr.get("b"))
    monkeypatch.setattr(exact_mgr.audit_log, "commit", original_commit)
    assert (await restore_exact_consolidation(exact_mgr, records))["restored"] == ["b"]


@pytest.mark.asyncio
async def test_restore_conflict_preserves_subsequent_edit_and_custom_actor_audit(exact_mgr):
    seed(exact_mgr, memory("a"), memory("b", digested=False, exact_duplicate_of=""))
    result = await run_exact_consolidation(exact_mgr, actor="test:coordinator")
    events = exact_mgr.audit_log.list_events()
    assert recovery_records_from_audit(events) == []
    records = recovery_records_from_audit(events, actor="test:coordinator")
    assert records[0]["original_markers"] == {
        "digested": {"present": True, "value": False},
        "exact_duplicate_of": {"present": True, "value": ""}}
    assert records[0]["after_revision_hash"] == result["recovery"][0]["after_revision_hash"]
    assert await exact_mgr.update("b", actor="later-edit", tags=["edited later"])
    before_restore = await exact_mgr.get("b")
    restored = await restore_exact_consolidation(exact_mgr, records)
    assert restored["restored"] == []
    assert await exact_mgr.get("b") == before_restore


def _child_writer(config, ready, proceed, attempting, finished):
    # Construct before maintenance so the test exercises update's shared
    # lease itself, rather than just its constructor's shared lease.
    manager = BucketManager(config)
    ready.set()
    assert proceed.wait(timeout=5)
    attempting.set()

    async def write():
        assert await manager.update("a", actor="process-writer", content="later winner body")
        assert await manager.update("b", actor="process-writer", tags=["later loser key"])

    asyncio.run(write())
    finished.set()


@pytest.mark.skipif(os.name == "nt", reason="cross-process POSIX maintenance barrier contract")
@pytest.mark.asyncio
async def test_real_process_cannot_change_winner_or_loser_during_recheck_and_apply(exact_mgr, exact_config, monkeypatch):
    seed(exact_mgr, memory("a"), memory("b"))
    context = multiprocessing.get_context("spawn")
    ready, proceed, attempting, finished = (context.Event() for _ in range(4))
    process = context.Process(target=_child_writer, args=(exact_config, ready, proceed, attempting, finished))
    real_get, real_update = exact_mgr.get, exact_mgr.update
    started = False

    async def checked_get(bucket_id):
        nonlocal started
        bucket = await real_get(bucket_id)
        if bucket_id == "a" and not started:
            started = True
            proceed.set()
            assert await asyncio.to_thread(attempting.wait, 5)
            await asyncio.sleep(0.1)
            assert not finished.is_set(), "writer entered during winner revalidation"
        return bucket

    async def checked_update(*args, **kwargs):
        assert not finished.is_set(), "writer entered before loser CAS"
        ok = await real_update(*args, **kwargs)
        await asyncio.sleep(0.05)
        assert not finished.is_set(), "writer entered during loser CAS"
        return ok

    monkeypatch.setattr(exact_mgr, "get", checked_get)
    monkeypatch.setattr(exact_mgr, "update", checked_update)
    try:
        process.start()
        assert await asyncio.to_thread(ready.wait, 5)
        result = await run_exact_consolidation(exact_mgr)
        assert result["applied"] == ["b"]
        assert await asyncio.to_thread(finished.wait, 5)
        await asyncio.to_thread(process.join, 5)
        assert process.exitcode == 0
        assert (await real_get("a"))["content"] == "later winner body"
        assert (await real_get("b"))["metadata"]["tags"] == ["later loser key"]
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
