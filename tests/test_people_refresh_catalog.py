"""Synthetic catalog-change refresh/retry tests; no NAS or production stores."""

from __future__ import annotations

import asyncio
import copy
import json
import socket
import stat
from pathlib import Path

import pytest

import people_memory
from entity_store import ENTITY_TYPES, EntityStore
from maintenance_barrier import MaintenanceBarrier
from people_memory import refresh_people_links


STATE_FILE = "people-refresh-catalog.json"


class Buckets:
    def __init__(self, count=1, content="Alice 和 Bob 是同事。"):
        self.rows = [{"id": f"source-{index:03d}", "content": content,
                      "metadata": {"type": "dynamic", "world": ""}}
                     for index in range(count)]

    async def list_all(self, **kwargs):
        assert kwargs == {"include_archive": True, "include_nsfw": True}
        return copy.deepcopy(self.rows)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("catalog refresh tests must not use network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def store_at(tmp_path, names=("Alice",)):
    return EntityStore({"buckets_dir": str(tmp_path / "synthetic-vault"), "entities": {
        "seeds": [{"canonical_name": name, "type": "person", "aliases": []} for name in names],
    }})


def registry(store):
    return sorted((row.entity_id, row.entity_type, row.canonical_name, row.aliases)
                  for entity_type in ENTITY_TYPES for row in store.catalog(entity_type))


def state_path(store):
    return Path(store.entities_dir) / STATE_FILE


def assert_small_private_state(store):
    path = state_path(store)
    state = json.loads(path.read_text())
    assert set(state) == {"schema_version", "catalog_sha256"}
    assert state["schema_version"] == 1
    assert len(state["catalog_sha256"]) == 64
    assert len(path.read_bytes()) < 160
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert {p.name for p in Path(store.entities_dir).iterdir()} == {"entities.sqlite3", STATE_FILE}


@pytest.mark.asyncio
async def test_new_bob_refreshes_current_body_and_persisted_second_run_is_idle(tmp_path):
    store, buckets = store_at(tmp_path), Buckets()
    before = copy.deepcopy(buckets.rows)
    # Existing current Alice link is the original hash-only skip reproducer.
    store.resolve_and_link("source-000", buckets.rows[0]["content"])
    await refresh_people_links(store, buckets)
    original_signature = state_path(store).read_bytes()
    bob = store.resolve_or_create("Bob", entity_type="person")
    declared = registry(store)
    assert store.linked_bucket_ids(entity_ids=[bob.entity_id]) == []

    result = await refresh_people_links(store, buckets)
    assert result == {"checked": 1, "refreshed": 1, "memory_written": False}
    assert store.linked_bucket_ids(entity_ids=[bob.entity_id]) == ["source-000"]
    assert registry(store) == declared and buckets.rows == before
    assert state_path(store).read_bytes() != original_signature
    assert_small_private_state(store)

    path = state_path(store)
    saved = (path.read_bytes(), path.stat().st_mtime_ns)
    reopened = EntityStore({"buckets_dir": store.buckets_dir}, initialize=False)
    assert (await refresh_people_links(reopened, buckets))["refreshed"] == 0
    assert (path.read_bytes(), path.stat().st_mtime_ns) == saved
    assert registry(reopened) == declared


@pytest.mark.asyncio
async def test_added_alias_refreshes_unchanged_body_without_creating_people(tmp_path):
    store, buckets = store_at(tmp_path, ("Alice", "Bob")), Buckets(content="Alice 和 Bobby 是同事。")
    await refresh_people_links(store, buckets)
    bob = store.resolve_or_create("Bob", aliases=["Bobby"], entity_type="person")
    declared = registry(store)
    before = copy.deepcopy(buckets.rows)
    assert store.linked_bucket_ids(entity_ids=[bob.entity_id]) == []
    assert (await refresh_people_links(store, buckets))["refreshed"] == 1
    assert store.linked_bucket_ids(query="Bobby") == ["source-000"]
    assert registry(store) == declared and buckets.rows == before
    assert (await refresh_people_links(store, buckets))["refreshed"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_failed_later_batch_invalidates_watermark_and_retries_completed_batches(tmp_path, monkeypatch, failure):
    store, buckets = store_at(tmp_path), Buckets(count=201)
    await refresh_people_links(store, buckets)
    bob = store.resolve_or_create("Bob", entity_type="person")
    declared = registry(store)
    original_refresh = store.refresh_declared_links
    batches = []

    def fail_second_batch(records):
        batches.append(len(records))
        if len(batches) == 2:
            raise failure("synthetic second-batch failure")
        return original_refresh(records)

    monkeypatch.setattr(store, "refresh_declared_links", fail_second_batch)
    with pytest.raises(failure, match="second-batch failure"):
        await refresh_people_links(store, buckets)
    assert batches == [100, 100]
    assert len(store.linked_bucket_ids(entity_ids=[bob.entity_id])) == 100
    assert not state_path(store).exists()

    reopened = EntityStore({"buckets_dir": store.buckets_dir}, initialize=False)
    assert await refresh_people_links(reopened, buckets) == {
        "checked": 201, "refreshed": 201, "memory_written": False}
    assert len(reopened.linked_bucket_ids(entity_ids=[bob.entity_id])) == 201
    assert registry(reopened) == declared
    assert (await refresh_people_links(reopened, buckets))["refreshed"] == 0
    assert_small_private_state(reopened)


@pytest.mark.asyncio
async def test_publish_failure_retries_even_after_all_links_landed(tmp_path, monkeypatch):
    store, buckets = store_at(tmp_path), Buckets()
    await refresh_people_links(store, buckets)
    bob = store.resolve_or_create("Bob", entity_type="person")
    publish = people_memory.atomic_write_text

    def fail_publish(*args, **kwargs):
        raise OSError("synthetic watermark failure")

    monkeypatch.setattr(people_memory, "atomic_write_text", fail_publish)
    with pytest.raises(OSError, match="watermark failure"):
        await refresh_people_links(store, buckets)
    assert store.linked_bucket_ids(entity_ids=[bob.entity_id]) == ["source-000"]
    assert not state_path(store).exists()
    monkeypatch.setattr(people_memory, "atomic_write_text", publish)
    assert (await refresh_people_links(store, buckets))["refreshed"] == 1
    assert (await refresh_people_links(store, buckets))["refreshed"] == 0
    assert_small_private_state(store)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["missing", "bad_json", "bad_schema", "bad_hash", "bad_mode"])
async def test_missing_or_invalid_watermark_rebuilds_hash_current_links(tmp_path, invalid):
    store, buckets = store_at(tmp_path, ("Alice", "Bob")), Buckets()
    await refresh_people_links(store, buckets)
    bob = store.resolve_query("Bob").entity_ids[0]
    # Simulate a preexisting incomplete sidecar: Alice's current body hash
    # remains, so the old hash-only refresh would never restore Bob.
    connection = store._connect_write()
    try:
        connection.execute("DELETE FROM bucket_entities WHERE entity_id = ?", (bob,))
    finally:
        connection.close()
    path = state_path(store)
    if invalid == "missing":
        path.unlink()
    elif invalid == "bad_json":
        path.write_text("{interrupted")
    elif invalid == "bad_schema":
        state = json.loads(path.read_text())
        state["schema_version"] = 999
        path.write_text(json.dumps(state))
    elif invalid == "bad_hash":
        path.write_text('{"schema_version":1,"catalog_sha256":false}')
    else:
        path.chmod(0o644)
    assert (await refresh_people_links(store, buckets))["refreshed"] == 1
    assert store.linked_bucket_ids(entity_ids=[bob]) == ["source-000"]
    assert (await refresh_people_links(store, buckets))["refreshed"] == 0
    assert_small_private_state(store)


@pytest.mark.asyncio
async def test_catalog_change_during_scan_does_not_publish_mixed_generation(tmp_path, monkeypatch):
    store, buckets = store_at(tmp_path), Buckets()
    await refresh_people_links(store, buckets)
    store.resolve_or_create("Bob", entity_type="person")
    original = store.refresh_declared_links

    def register_during_refresh(records):
        result = original(records)
        store.resolve_or_create("Cara", entity_type="person")
        return result

    monkeypatch.setattr(store, "refresh_declared_links", register_during_refresh)
    with pytest.raises(RuntimeError, match="catalog changed"):
        await refresh_people_links(store, buckets)
    assert not state_path(store).exists()
    monkeypatch.setattr(store, "refresh_declared_links", original)
    assert (await refresh_people_links(store, buckets))["refreshed"] == 1
    assert (await refresh_people_links(store, buckets))["refreshed"] == 0


@pytest.mark.asyncio
async def test_cross_type_alias_collision_also_invalidates_catalog(tmp_path):
    store, buckets = store_at(tmp_path), Buckets(content="Alice 和 Coach 是同事。")
    person = store.resolve_or_create("Bob", aliases=["Coach"], entity_type="person")
    await refresh_people_links(store, buckets)
    assert store.linked_bucket_ids(entity_ids=[person.entity_id]) == ["source-000"]
    store.resolve_or_create("Coach", entity_type="place")
    declared = registry(store)
    assert (await refresh_people_links(store, buckets))["refreshed"] == 1
    assert store.linked_bucket_ids(entity_ids=[person.entity_id]) == []
    assert registry(store) == declared
    assert (await refresh_people_links(store, buckets))["refreshed"] == 0


@pytest.mark.asyncio
async def test_existing_night_exclusive_lease_is_reentrant(tmp_path):
    store, buckets = store_at(tmp_path), Buckets()
    async with MaintenanceBarrier(store.buckets_dir).exclusive_async(timeout=1):
        assert (await refresh_people_links(store, buckets))["refreshed"] == 1
        assert (await refresh_people_links(store, buckets))["refreshed"] == 0


@pytest.mark.asyncio
async def test_no_people_does_not_initialize_vault_or_watermark(tmp_path):
    root = tmp_path / "absent"
    absent = EntityStore({"buckets_dir": str(root)}, initialize=False)

    class NoScan:
        async def list_all(self, **kwargs):
            raise AssertionError("no people must not scan buckets")

    for store in (None, absent):
        assert await refresh_people_links(store, NoScan()) == {
            "checked": 0, "refreshed": 0, "memory_written": False}
        assert not root.exists()
    empty = store_at(tmp_path, ())
    before = {p.relative_to(empty.buckets_dir) for p in Path(empty.buckets_dir).rglob("*")}
    assert (await refresh_people_links(empty, NoScan()))["checked"] == 0
    assert {p.relative_to(empty.buckets_dir) for p in Path(empty.buckets_dir).rglob("*")} == before
