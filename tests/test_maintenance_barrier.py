from __future__ import annotations

import asyncio
import contextvars
import multiprocessing
import os
import queue
import sqlite3
import threading
from pathlib import Path

import pytest

import maintenance_barrier as barrier_module
from maintenance_barrier import (
    MaintenanceBarrier,
    MaintenanceBarrierError,
    MaintenanceBarrierTimeout,
)
from e_axis_shadow import EAxisShadowStore
from lmc5_ledger import LMC5Ledger
from review_queue import ReviewQueue
from snapshot_manager import SnapshotManager


def _database(path: Path, value: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE state (value TEXT NOT NULL)")
        connection.execute("INSERT INTO state VALUES (?)", (value,))
        connection.commit()


def _value(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return str(connection.execute("SELECT value FROM state").fetchone()[0])


def _child_shared_lease(root: str, events) -> None:
    with MaintenanceBarrier(root).shared():
        events.put("entered")


@pytest.mark.asyncio
async def test_exclusive_async_blocks_other_task_and_allows_nested_calls(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    barrier = MaintenanceBarrier(root)
    entered = asyncio.Event()

    async def ordinary_writer():
        async with barrier.shared_async():
            entered.set()

    async with barrier.exclusive_async():
        async with barrier.shared_async():
            with barrier.shared():
                pass
        task = asyncio.create_task(ordinary_writer())
        await asyncio.sleep(0.05)
        assert not entered.is_set()

    await asyncio.wait_for(task, timeout=1)
    assert entered.is_set()


@pytest.mark.asyncio
async def test_touch_keeps_shared_lease_through_time_ripple(
    bucket_mgr,
    monkeypatch,
):
    bucket_id = await bucket_mgr.create("touch maintenance barrier regression")
    before_touch = await bucket_mgr.get(bucket_id)
    ripple_started = asyncio.Event()
    release_ripple = asyncio.Event()
    exclusive_attempted = asyncio.Event()
    exclusive_entered = asyncio.Event()
    order: list[str] = []

    async def paused_ripple(source_id, reference_time, hours=48.0):
        assert source_id == bucket_id
        assert reference_time is not None
        assert hours == 48.0
        order.append("ripple_started")
        ripple_started.set()
        await release_ripple.wait()
        order.append("ripple_finished")

    async def exclusive_maintenance():
        exclusive_attempted.set()
        async with bucket_mgr._maintenance_barrier.exclusive_async(timeout=1):
            order.append("exclusive_entered")
            exclusive_entered.set()

    monkeypatch.setattr(bucket_mgr, "_time_ripple", paused_ripple)
    touch_task = asyncio.create_task(bucket_mgr.touch(bucket_id))
    await asyncio.wait_for(ripple_started.wait(), timeout=1)

    maintenance_task = asyncio.create_task(exclusive_maintenance())
    await asyncio.wait_for(exclusive_attempted.wait(), timeout=1)
    await asyncio.sleep(0.05)
    assert not exclusive_entered.is_set()

    release_ripple.set()
    await asyncio.wait_for(touch_task, timeout=1)
    await asyncio.wait_for(maintenance_task, timeout=1)

    assert order == [
        "ripple_started",
        "ripple_finished",
        "exclusive_entered",
    ]
    after_touch = await bucket_mgr.get(bucket_id)
    assert after_touch["metadata"]["activation_count"] == (
        before_touch["metadata"]["activation_count"] + 1
    )


@pytest.mark.asyncio
async def test_exclusive_async_to_thread_context_manager_is_reentrant(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    barrier = MaintenanceBarrier(root)
    observed: list[str] = []

    def sync_leaf() -> None:
        with barrier.shared():
            observed.append("entered")

    async def run_in_copied_thread() -> None:
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        completed = loop.create_future()

        def worker() -> None:
            try:
                context.run(sync_leaf)
            except BaseException as exc:
                loop.call_soon_threadsafe(completed.set_exception, exc)
            else:
                loop.call_soon_threadsafe(completed.set_result, None)

        threading.Thread(target=worker, daemon=True).start()
        await completed

    async with barrier.exclusive_async():
        await asyncio.wait_for(run_in_copied_thread(), timeout=1)

    assert observed == ["entered"]


@pytest.mark.asyncio
async def test_async_timeout_and_cancellation_do_not_leak_a_lease(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    barrier = MaintenanceBarrier(root)

    async def competing_writer(timeout=None):
        async with barrier.shared_async(timeout=timeout):
            pass

    async with barrier.exclusive_async():
        with pytest.raises(MaintenanceBarrierTimeout):
            await asyncio.create_task(competing_writer(timeout=0.05))
        waiter = asyncio.create_task(competing_writer())
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    async with barrier.shared_async(timeout=0.2):
        pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX fork/flock capability test")
def test_exclusive_lease_blocks_a_forked_process(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    context = multiprocessing.get_context("fork")
    events = context.Queue()

    with MaintenanceBarrier(root).exclusive():
        process = context.Process(
            target=_child_shared_lease,
            args=(os.fspath(root), events),
        )
        process.start()
        with pytest.raises(queue.Empty):
            events.get(timeout=0.1)

    assert events.get(timeout=2) == "entered"
    process.join(timeout=2)
    assert process.exitcode == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX at-fork lock reset test")
def test_fork_child_does_not_wait_on_an_inherited_thread_guard(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    context = multiprocessing.get_context("fork")
    events = context.Queue()
    guard_held = threading.Event()
    release_guard = threading.Event()

    def hold_parent_guard() -> None:
        with barrier_module._HELD_GUARD:
            guard_held.set()
            release_guard.wait(timeout=2)

    holder = threading.Thread(target=hold_parent_guard, daemon=True)
    holder.start()
    assert guard_held.wait(timeout=1)
    process = context.Process(
        target=_child_shared_lease,
        args=(os.fspath(root), events),
    )
    try:
        process.start()
        assert events.get(timeout=1) == "entered"
    finally:
        release_guard.set()
        holder.join(timeout=1)
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
    assert process.exitcode == 0


def test_shared_lease_cannot_upgrade_to_exclusive(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    barrier = MaintenanceBarrier(root)

    with barrier.shared():
        with pytest.raises(MaintenanceBarrierError, match="cannot upgrade"):
            with barrier.exclusive():
                pass


def test_explicit_vault_root_gives_every_store_the_same_lock_path(tmp_path):
    root = tmp_path / "vault"
    backup = tmp_path / "backups"
    root.mkdir()
    expected = MaintenanceBarrier(root).lock_path

    snapshot = SnapshotManager(root, backup)
    ledger = LMC5Ledger(
        root / ".lmc5" / "pipeline.sqlite3",
        maintenance_root=root,
    )
    review = ReviewQueue(
        root / "review_queue.jsonl",
        maintenance_root=root,
    )
    shadow = EAxisShadowStore(
        root / ".axis" / "e-shadow.jsonl",
        maintenance_root=root,
    )

    assert snapshot.maintenance_barrier.lock_path == expected
    assert ledger._maintenance_barrier.lock_path == expected
    assert review._maintenance_barrier.lock_path == expected
    assert shadow._maintenance_barrier.lock_path == expected


def test_snapshot_exclusive_lease_prevents_mixed_sqlite_timepoints(
    tmp_path, monkeypatch
):
    source = tmp_path / "vault"
    backup = tmp_path / "backups"
    source.mkdir()
    _database(source / "a.db", "old")
    _database(source / "b.db", "old")
    manager = SnapshotManager(source, backup)
    writer_started = threading.Event()
    writer_finished = threading.Event()
    original_backup = manager._backup_sqlite

    def guarded_writer():
        writer_started.wait(timeout=2)
        with MaintenanceBarrier(source).shared():
            with sqlite3.connect(source / "b.db") as connection:
                connection.execute("UPDATE state SET value = 'new'")
                connection.commit()
            with sqlite3.connect(source / "a.db") as connection:
                connection.execute("UPDATE state SET value = 'new'")
                connection.commit()
        writer_finished.set()

    thread = threading.Thread(target=guarded_writer, daemon=True)
    thread.start()

    def pause_after_first_database(entry, target):
        result = original_backup(entry, target)
        if entry.relative.as_posix() == "a.db":
            writer_started.set()
            assert not writer_finished.wait(timeout=0.05)
        return result

    monkeypatch.setattr(manager, "_backup_sqlite", pause_after_first_database)
    snapshot = manager.create_snapshot("coherent")
    thread.join(timeout=2)
    assert writer_finished.is_set()

    verified = manager.verify_snapshot(
        "coherent",
        expected_manifest_sha256=snapshot.manifest_sha256,
    )
    assert verified.file_count == 2
    assert _value(snapshot.snapshot_path / "files" / "a.db") == "old"
    assert _value(snapshot.snapshot_path / "files" / "b.db") == "old"
    assert _value(source / "a.db") == "new"
    assert _value(source / "b.db") == "new"
