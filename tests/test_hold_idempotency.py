"""Synthetic write/replay contracts; no provider, live vault or model calls."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import server


@pytest.fixture
def runtime(monkeypatch, bucket_mgr):
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "config", {"current_world": ""})
    monkeypatch.setattr(server, "_ensure_decay_background", AsyncMock())
    monkeypatch.setattr(server, "_maybe_start_backfill", lambda: None)
    monkeypatch.setattr(server, "_synchronize_bucket_entities", AsyncMock())
    monkeypatch.setattr(server, "_auto_infer_edges", AsyncMock(return_value=[]))
    monkeypatch.setattr(server, "_mark_briefing_cache_dirty", lambda *_: None)
    monkeypatch.setattr(bucket_mgr, "auto_link_created_bucket", AsyncMock())
    analyze = AsyncMock(return_value={
        "domain": ["test"], "valence": 0.5, "arousal": 0.3,
        "tags": [], "suggested_name": "synthetic", "entities": [],
    })
    embedding = AsyncMock()
    monkeypatch.setattr(server, "dehydrator", SimpleNamespace(analyze=analyze))
    monkeypatch.setattr(server, "embedding_engine", SimpleNamespace(generate_and_store=embedding))
    # A keyed event is never allowed through heuristic merge, including a
    # different event that happens to have identical words.
    monkeypatch.setattr(server, "_find_merge_candidates", AsyncMock(
        side_effect=AssertionError("keyed event reached heuristic merge")))
    return bucket_mgr, analyze, embedding


class Request:
    def __init__(self, **body):
        self.body = body

    async def json(self):
        return self.body


async def write(**overrides):
    payload = {"content": "synthetic event body", "idempotency_key": "imprint:v1:event-A"}
    payload.update(overrides)
    response = await server.api_hold(Request(**payload))
    assert response.status_code == 200, response.body
    return json.loads(response.body)


def files(manager):
    return sorted(Path(manager.base_dir).rglob("*.md"))


@pytest.mark.asyncio
async def test_keyed_bucket_remains_visible_to_literal_recall_and_patrol(runtime):
    from rg_literal_recall import bucket_id_from_path
    from patrol import _bucket_filename_shape
    manager, *_ = runtime
    result = await write()
    path = files(manager)[0]
    assert bucket_id_from_path(str(path)) == result["bucket_id"]
    assert _bucket_filename_shape({"source_kind": "markdown",
        "frontmatter_id": result["bucket_id"], "source_basename": path.name}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [{}, {"feel": True}, {"pinned": True}])
async def test_first_write_and_replay_do_not_touch_existing_body_or_metadata(runtime, mode):
    manager, analyze, embedding = runtime
    first = await write(**mode)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in files(manager)}
    assert len(before) == 1
    second = await write(**mode)
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in files(manager)}
    assert before == after
    assert first["bucket_id"] == second["bucket_id"]
    assert second["replayed"] is True
    assert embedding.await_count == 1
    assert analyze.await_count == (0 if mode.get("feel") else 1)


@pytest.mark.asyncio
async def test_same_words_different_event_world_mode_and_changed_words_survive(runtime):
    manager, *_ = runtime
    responses = [await write(), await write(idempotency_key="imprint:v1:event-B"),
                 await write(world="fictional-test-world"),
                 await write(content="synthetic event body changed"),
                 await write(feel=True), await write(pinned=True)]
    assert len({r["bucket_id"] for r in responses}) == 6
    assert len(files(manager)) == 6


@pytest.mark.asyncio
async def test_concurrent_same_event_creates_one_bucket(runtime):
    manager, analyze, _ = runtime
    arrived = 0
    ready = asyncio.Event()

    async def all_requests_passed_initial_lookup(_content):
        nonlocal arrived
        arrived += 1
        if arrived == 12:
            ready.set()
        await ready.wait()
        return {"domain": ["test"], "valence": 0.5, "arousal": 0.3,
                "tags": [], "suggested_name": f"worker-{arrived}", "entities": []}

    analyze.side_effect = all_requests_passed_initial_lookup
    responses = await asyncio.gather(*(write() for _ in range(12)))
    assert len({r["bucket_id"] for r in responses}) == 1
    assert len(files(manager)) == 1


@pytest.mark.asyncio
async def test_cancel_after_durable_body_then_restart_and_retry_creates_no_duplicate(runtime, monkeypatch, test_config):
    manager, analyze, _ = runtime
    monkeypatch.setattr(manager, "auto_link_created_bucket", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await write()
    assert len(files(manager)) == 1
    from bucket_manager import BucketManager
    restarted = BucketManager(test_config)
    monkeypatch.setattr(server, "bucket_mgr", restarted)
    result = await write()
    assert result["replayed"] is True
    assert len(files(restarted)) == 1
    assert analyze.await_count == 1


@pytest.mark.asyncio
async def test_replay_refuses_mutated_body_without_overwriting(runtime):
    manager, *_ = runtime
    await write()
    path = files(manager)[0]
    changed = path.read_text().replace("synthetic event body", "body edited by another owner")
    path.write_text(changed)
    response = await server.api_hold(Request(content="synthetic event body", idempotency_key="imprint:v1:event-A"))
    assert response.status_code == 409
    assert path.read_text() == changed
    assert len(files(manager)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["", " ", "../file", "bad\nkey", "k" * 257, 123, None])
async def test_invalid_explicit_key_is_rejected_before_hold(monkeypatch, key):
    hold = AsyncMock()
    monkeypatch.setattr(server, "hold", hold)
    response = await server.api_hold(Request(content="test", idempotency_key=key))
    assert response.status_code == 400
    hold.assert_not_awaited()


@pytest.mark.asyncio
async def test_unkeyed_legacy_request_keeps_original_contract(monkeypatch):
    hold = AsyncMock(return_value="legacy-result")
    monkeypatch.setattr(server, "hold", hold)
    response = await server.api_hold(Request(content="test"))
    assert json.loads(response.body) == {"result": "legacy-result"}
    assert "idempotency_key" not in hold.call_args.kwargs


@pytest.mark.asyncio
async def test_receipt_world_is_captured_before_async_analysis(runtime):
    manager, analyze, _ = runtime
    server.config["current_world"] = "initial-world"
    original = analyze.return_value

    async def change_config(_):
        server.config["current_world"] = "different-world"
        return original

    analyze.side_effect = change_config
    result = await write()
    bucket = await manager.get(result["bucket_id"])
    assert bucket is not None
    assert bucket["metadata"]["world"] == "initial-world"


@pytest.mark.asyncio
async def test_replay_does_not_start_background_mutation(runtime, monkeypatch):
    await write()
    background = AsyncMock(side_effect=AssertionError("replay started decay"))
    monkeypatch.setattr(server, "_ensure_decay_background", background)
    assert (await write())["replayed"] is True
    background.assert_not_awaited()


def test_two_processes_share_durable_identity_and_never_overwrite(test_config):
    script = r'''
import asyncio, json, sys
from bucket_manager import BucketManager
from hold_idempotency import HoldWriteIdentity
manager = BucketManager(json.loads(sys.argv[1]))
async def no_links(*args, **kwargs): pass
manager.auto_link_created_bucket = no_links
identity = HoldWriteIdentity.build("event:process-test", "synthetic process body", "", feel=False, pinned=False)
print("READY", flush=True)
sys.stdin.readline()
bucket_id = asyncio.run(manager.create(content="synthetic process body", name=sys.argv[2], hold_write_identity=identity))
print(json.dumps({"id": bucket_id, "replayed": identity.replayed}), flush=True)
'''
    children = [subprocess.Popen(
        [sys.executable, "-c", script, json.dumps(test_config), f"writer-{i}"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for i in range(2)]
    try:
        for child in children:
            assert child.stdout.readline().strip() == "READY"
        for child in children:
            child.stdin.write("go\n")
            child.stdin.flush()
        results = []
        for child in children:
            stdout, stderr = child.communicate(timeout=20)
            assert child.returncode == 0, stderr
            results.append(json.loads(stdout.strip().splitlines()[-1]))
        assert results[0]["id"] == results[1]["id"]
        assert sorted(r["replayed"] for r in results) == [False, True]
        assert len(list(Path(test_config["buckets_dir"]).rglob("*.md"))) == 1
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate()
