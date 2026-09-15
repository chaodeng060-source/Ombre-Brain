"""Local regressions for the 2026-09-15 RP deadline failure; no providers."""
import os
import socket
import hashlib
import json
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import frontmatter
import pytest

import server
from bucket_manager import BucketManager


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*_args, **_kwargs):
        pytest.fail("this regression must not call a provider")
    monkeypatch.setattr(socket.socket, "connect", denied)


@pytest.fixture
def manager(test_config, monkeypatch):
    monkeypatch.setenv("OMBRE_BM25_MODE", "live")
    return BucketManager(test_config)


async def warm(manager):
    await manager.prewarm_recall_snapshot()
    await manager.prewarm_bm25()
    assert not manager._bm25_dirty


@pytest.mark.asyncio
async def test_known_write_refresh_loads_only_changed_file_without_rebuilding(manager, monkeypatch):
    first = await manager.create("first synthetic scene", world="恨海RP")
    await warm(manager)
    second = await manager.create("second synthetic scene", world="恨海RP")
    generation = manager._bm25_generation
    token = await manager.recall_snapshot_token()
    loader = Mock(wraps=manager._load_bucket)
    monkeypatch.setattr(manager, "_load_bucket", loader)

    await manager._refresh_recall_snapshot_background((False, False))

    assert not manager._bm25_dirty
    assert manager._bm25_generation == generation
    assert await manager.recall_snapshot_token() == token
    assert loader.call_count == 1
    assert {b["id"] for b in manager._recall_snapshot_cache[(False, False)]} == {first, second}


@pytest.mark.asyncio
async def test_external_edit_still_invalidates_index_and_updates_snapshot(manager, monkeypatch):
    bid = await manager.create("old synthetic scene", world="恨海RP")
    await manager.create("unchanged synthetic scene", world="恨海RP")
    await warm(manager)
    row = await manager.get(bid)
    post = frontmatter.load(row["path"])
    post.content = "external replacement scene"
    frontmatter.dump(post, row["path"])
    loader = Mock(wraps=manager._load_bucket)
    monkeypatch.setattr(manager, "_load_bucket", loader)

    await manager._refresh_recall_snapshot_background((False, False))

    assert manager._bm25_dirty and manager._bm25_unknown_dirty
    assert loader.call_count == 1
    cached = {b["id"]: b for b in manager._recall_snapshot_cache[(False, False)]}
    assert cached[bid]["content"] == "external replacement scene"


@pytest.mark.asyncio
async def test_external_delete_and_same_stat_atomic_replacement_are_not_hidden(manager):
    bid = await manager.create("alpha scene", world="恨海RP")
    other = await manager.create("another scene", world="恨海RP")
    await warm(manager)
    row = await manager.get(bid)
    old = os.stat(row["path"])
    post = frontmatter.load(row["path"])
    post.content = "bravo scene"
    replacement = str(row["path"]) + ".replacement"
    frontmatter.dump(post, replacement)
    os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
    os.replace(replacement, row["path"])
    os.unlink((await manager.get(other))["path"])

    await manager._refresh_recall_snapshot_background((False, False))

    assert manager._bm25_unknown_dirty
    cached = {b["id"]: b for b in manager._recall_snapshot_cache[(False, False)]}
    assert other not in cached
    assert cached[bid]["content"] == "bravo scene"


@pytest.mark.asyncio
async def test_concurrent_write_cannot_be_replaced_by_an_older_disk_scan(manager, monkeypatch):
    bid = await manager.create("before external edit", world="恨海RP")
    await warm(manager)
    path = (await manager.get(bid))["path"]
    post = frontmatter.load(path)
    post.content = "external edit while live"
    frontmatter.dump(post, path)
    entered, release = threading.Event(), threading.Event()
    original = manager._load_recall_snapshot_sync

    def pause(paths):
        result = original(paths)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(manager, "_load_recall_snapshot_sync", pause)
    task = asyncio.create_task(manager._refresh_recall_snapshot_background((False, False)))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        new_id = await manager.create("new concurrent scene", world="恨海RP")
        token = await manager.recall_snapshot_token()
    finally:
        release.set()
        await task
    assert await manager.recall_snapshot_token() == token
    assert new_id in {b["id"] for b in manager._recall_snapshot_cache[(False, False)]}
    await manager._refresh_recall_snapshot_background((False, False))
    assert manager._bm25_unknown_dirty
    assert {b["id"]: b["content"] for b in manager._recall_snapshot_cache[(False, False)]}[bid] == post.content


def test_timeline_checks_only_requested_world_paths_and_keeps_shared_neighbors(monkeypatch):
    from tests.test_timeline_breath import _bucket
    seed = _bucket("seed", "synthetic", thread="plot")
    shared = _bucket("shared", "synthetic", thread="plot", event_at="2026-08-02T00:00:00+08:00")
    seed["metadata"]["world"] = "恨海RP"
    shared["metadata"]["world"] = "通用"
    irrelevant = [_bucket(f"daily-{i}", "synthetic", thread="plot") for i in range(16000)]
    checker = Mock(wraps=server._is_main_recall_bucket)
    monkeypatch.setattr(server, "_is_main_recall_bucket", checker)
    monkeypatch.setattr(server, "config", {"timeline_recall": {"enabled": True}})
    result = server._timeline_recall_neighbors(
        [seed, shared, *irrelevant], ["seed"], query="synthetic", intent="default",
        world_filter={"恨海RP"}, domain_filter=None, created_after=None,
        created_before=None, max_results=1,
    )
    assert [n.bucket_id for n in result] == ["shared"]
    assert {call.args[0]["id"] for call in checker.call_args_list} == {"seed", "shared"}


@pytest.mark.asyncio
async def test_hold_status_confirms_exact_body_in_exact_world_without_writes(manager, monkeypatch):
    content = "synthetic complete RP round"
    bid = await manager.create(content, world="恨海RP")
    await manager.prewarm_recall_snapshot()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    before = {str(p): p.stat().st_mtime_ns for p in Path(manager.base_dir).rglob("*") if p.is_file()}
    digest = hashlib.sha256(content.encode()).hexdigest()
    response = await server.api_hold_status(SimpleNamespace(query_params={
        "world": "恨海RP", "content_sha256": digest,
    }))
    assert response.status_code == 200
    assert json.loads(response.body) == {
        "state": "stored", "bucket_id": bid, "world": "恨海RP", "content_sha256": digest,
    }
    assert content not in response.body.decode()
    after = {str(p): p.stat().st_mtime_ns for p in Path(manager.base_dir).rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.asyncio
async def test_hold_status_does_not_confirm_other_world_or_stale_snapshot(manager, monkeypatch):
    content = "synthetic complete RP round"
    bid = await manager.create(content, world="daily")
    await manager.prewarm_recall_snapshot()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    digest = hashlib.sha256(content.encode()).hexdigest()
    response = await server.api_hold_status(SimpleNamespace(query_params={
        "world": "恨海RP", "content_sha256": digest,
    }))
    assert json.loads(response.body)["state"] == "unconfirmed"
    # The resident snapshot still has the old body. Confirm against disk, not
    # just a matching cached hash, after an external replacement.
    row = await manager.get(bid)
    post = frontmatter.load(row["path"])
    post.content = "different body on disk"
    frontmatter.dump(post, row["path"])
    response = await server.api_hold_status(SimpleNamespace(query_params={
        "world": "daily", "content_sha256": digest,
    }))
    assert json.loads(response.body)["state"] == "unconfirmed"


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{}, {"world": "恨海RP", "content_sha256": "bad"}])
async def test_hold_status_rejects_invalid_lookup_without_loading(params, monkeypatch):
    monkeypatch.setattr(server, "bucket_mgr", SimpleNamespace())
    assert (await server.api_hold_status(SimpleNamespace(query_params=params))).status_code == 400


def test_hold_confirmation_route_is_protected_by_existing_api_auth():
    from tests.test_mcp_auth import _exercise, TEST_TOKEN
    from mcp_auth import APIBearerAuthMiddleware
    calls, messages = _exercise("/api/hold-status", method="GET", middleware_class=APIBearerAuthMiddleware)
    assert calls == 0 and messages[0]["status"] == 401
    calls, messages = _exercise("/api/hold-status", method="GET", middleware_class=APIBearerAuthMiddleware,
                               headers=[(b"authorization", f"Bearer {TEST_TOKEN}".encode())])
    assert calls == 1 and messages[0]["status"] == 204
