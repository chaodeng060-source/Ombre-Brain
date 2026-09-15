"""Bounded shared-lease waits (2026-09-14 night-run lock incident).

A night run held the exclusive maintenance lease for ~5 hours and every
ordinary writer waited on it forever.  Writers now wait at most the configured
shared bound, fail with MaintenanceBarrierTimeout, leave no flock or file
descriptor behind, and the night run's own nested writes never wait.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import maintenance_barrier as barrier_module
from maintenance_barrier import (
    DEFAULT_SHARED_LEASE_TIMEOUT_SECONDS,
    MAINTENANCE_BUSY_CODE,
    MAINTENANCE_BUSY_ERRORS,
    MaintenanceBarrier,
    MaintenanceBarrierTimeout,
    configure_shared_lease_timeout,
    shared_lease_timeout,
    shared_lease_timeout_from_config,
)


@pytest.fixture(autouse=True)
def _restore_shared_timeout():
    # Import server first so its import-time configuration is the value every
    # test here restores (other test modules import it at collection anyway).
    import server  # noqa: F401

    previous = shared_lease_timeout()
    yield
    configure_shared_lease_timeout(previous)


def _fd_count() -> int:
    return len(tuple(Path("/proc/self/fd").iterdir()))


def _assert_lock_is_free(barrier: MaintenanceBarrier) -> None:
    # A stray shared flock from a timed-out waiter would make this probe busy.
    handle = barrier_module._acquire_handle(
        barrier.lock_path, "exclusive", blocking=False
    )
    barrier_module._release_handle(handle)


def test_config_reader_defaults_and_rollback_value():
    assert DEFAULT_SHARED_LEASE_TIMEOUT_SECONDS == 60.0
    assert shared_lease_timeout_from_config({}) == 60.0
    assert shared_lease_timeout_from_config(None) == 60.0
    assert shared_lease_timeout_from_config({"maintenance_barrier": None}) == 60.0
    assert (
        shared_lease_timeout_from_config(
            {"maintenance_barrier": {"shared_lease_timeout_seconds": 5}}
        )
        == 5.0
    )
    for invalid in (-1, "60", True, float("nan"), float("inf"), None):
        assert (
            shared_lease_timeout_from_config(
                {"maintenance_barrier": {"shared_lease_timeout_seconds": invalid}}
            )
            == 60.0
        )
    # explicit 0 = legacy unbounded wait (config-only rollback lever)
    assert (
        shared_lease_timeout_from_config(
            {"maintenance_barrier": {"shared_lease_timeout_seconds": 0}}
        )
        is None
    )
    with pytest.raises(ValueError):
        configure_shared_lease_timeout(0)
    configure_shared_lease_timeout(1.5)
    assert shared_lease_timeout() == 1.5
    assert configure_shared_lease_timeout(None) == 1.5
    assert shared_lease_timeout() is None


@pytest.mark.asyncio
async def test_async_writer_times_out_on_default_bound_without_leaking(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    holder = MaintenanceBarrier(root)
    writer = MaintenanceBarrier(root)
    configure_shared_lease_timeout(0.2)
    before = _fd_count()

    async def ordinary_writer():
        async with writer.shared_async():
            return "entered"

    async with holder.exclusive_async(timeout=1):
        inside = _fd_count()
        for _ in range(3):
            started = time.monotonic()
            with pytest.raises(MaintenanceBarrierTimeout) as raised:
                await asyncio.create_task(ordinary_writer())
            elapsed = time.monotonic() - started
            assert 0.15 <= elapsed < 2.0
            assert str(raised.value).startswith(MAINTENANCE_BUSY_CODE)
            assert raised.value.code == MAINTENANCE_BUSY_CODE
            assert _fd_count() == inside

    # "No leak" = no more descriptors than before.  In the full suite earlier
    # tests' unreachable sqlite connections can be GC-finalized mid-test
    # (2026-09-14: 16 fds from test_maintenance_barrier.py), so the count may drop.
    assert _fd_count() <= before
    _assert_lock_is_free(holder)
    assert await asyncio.wait_for(ordinary_writer(), timeout=1) == "entered"


@pytest.mark.asyncio
async def test_explicit_none_keeps_the_legacy_unbounded_wait(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    holder = MaintenanceBarrier(root)
    configure_shared_lease_timeout(0.05)
    entered = asyncio.Event()

    async def patient_writer():
        async with MaintenanceBarrier(root).shared_async(timeout=None):
            entered.set()

    async with holder.exclusive_async(timeout=1):
        waiter = asyncio.create_task(patient_writer())
        await asyncio.sleep(0.3)
        assert not entered.is_set()
        assert not waiter.done()
    await asyncio.wait_for(waiter, timeout=1)
    assert entered.is_set()


@pytest.mark.asyncio
async def test_unconfigured_process_keeps_legacy_unbounded_default(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    holder = MaintenanceBarrier(root)
    configure_shared_lease_timeout(None)
    entered = asyncio.Event()

    async def writer():
        async with MaintenanceBarrier(root).shared_async():
            entered.set()

    async with holder.exclusive_async(timeout=1):
        task = asyncio.create_task(writer())
        await asyncio.sleep(0.3)
        assert not task.done()
    await asyncio.wait_for(task, timeout=1)
    assert entered.is_set()


def test_thread_writers_time_out_finish_and_release_everything(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    holder = MaintenanceBarrier(root)
    configure_shared_lease_timeout(0.2)
    outcomes: list[str] = []
    outcomes_guard = threading.Lock()

    def leaf() -> None:
        try:
            with MaintenanceBarrier(root).shared():
                result = "entered"
        except MaintenanceBarrierTimeout as exc:
            result = "timeout" if MAINTENANCE_BUSY_CODE in str(exc) else "bad-message"
        with outcomes_guard:
            outcomes.append(result)

    before = _fd_count()
    with holder.exclusive():
        inside = _fd_count()
        threads = [
            threading.Thread(target=leaf, name=f"leaf-{index}", daemon=True)
            for index in range(6)
        ]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        assert time.monotonic() - started < 5
        assert _fd_count() == inside
    assert outcomes == ["timeout"] * 6
    # "No leak" = no more descriptors than before.  In the full suite earlier
    # tests' unreachable sqlite connections can be GC-finalized mid-test
    # (2026-09-14: 16 fds from test_maintenance_barrier.py), so the count may drop.
    assert _fd_count() <= before
    _assert_lock_is_free(holder)

    # Once the exclusive lease is gone the same thread path enters normally.
    leaf_thread = threading.Thread(target=leaf, daemon=True)
    leaf_thread.start()
    leaf_thread.join(timeout=5)
    assert outcomes[-1] == "entered"


def test_event_loop_sync_leaf_still_fails_closed_immediately(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    configure_shared_lease_timeout(5)
    held = threading.Event()
    release = threading.Event()

    def hold_exclusive_elsewhere() -> None:
        with MaintenanceBarrier(root).exclusive():
            held.set()
            release.wait(timeout=10)

    async def scenario():
        started = time.monotonic()
        with pytest.raises(MAINTENANCE_BUSY_ERRORS):
            with MaintenanceBarrier(root).shared():
                pass
        return time.monotonic() - started

    holder = threading.Thread(target=hold_exclusive_elsewhere, daemon=True)
    holder.start()
    try:
        assert held.wait(timeout=5)
        elapsed = asyncio.run(scenario())
    finally:
        release.set()
        holder.join(timeout=5)
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_nested_night_run_leases_never_wait_under_a_tiny_bound(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    night = MaintenanceBarrier(root)
    curated_view = MaintenanceBarrier(root)
    configure_shared_lease_timeout(0.01)
    observed: list[str] = []

    def sync_leaf() -> None:
        with curated_view.shared():
            with night.shared():
                observed.append("thread-leaf")

    async def copied_context_thread() -> None:
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

    async with night.exclusive_async(timeout=1):
        started = time.monotonic()
        for _ in range(25):
            async with curated_view.shared_async():
                with night.shared():
                    observed.append("nested")
        await asyncio.wait_for(copied_context_thread(), timeout=1)
        await asyncio.to_thread(sync_leaf)
        assert time.monotonic() - started < 1.0
    assert observed.count("nested") == 25
    assert observed.count("thread-leaf") == 2


# ---------------------------------------------------------------------------
# Real night coordinator: its own curated/bucket/ledger writes stay nested and
# unaffected while outside writers (async request path and the raw-event
# thread path) get the bounded busy failure.
# ---------------------------------------------------------------------------


class _PausingEmbedding:
    def __init__(self) -> None:
        self.enabled = True
        self.stored: set[str] = set()
        self.paused = asyncio.Event()
        self.resume = asyncio.Event()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        if not self.paused.is_set():
            self.paused.set()
            await self.resume.wait()
        self.stored.add(bucket_id)
        return True

    async def get_embedding(self, bucket_id: str):
        return [1.0] if bucket_id in self.stored else None

    def delete_embedding(self, bucket_id: str) -> None:
        self.stored.discard(bucket_id)

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        return 1.0 if a == b else 0.0


@pytest.mark.asyncio
async def test_night_run_writes_are_unaffected_while_outside_writers_time_out(
    tmp_path, test_config, bucket_mgr
):
    from consolidation_engine import ConsolidationEngine
    from curated_writer import CuratedWriteCoordinator
    from decay_engine import DecayEngine
    from lmc5_ledger import LMC5Ledger
    from lmc5_proposer import StrictOmbreProposer
    from night_run_coordinator import NightRunCoordinator
    from snapshot_manager import SnapshotManager
    from tests.test_night_run_coordinator import _Provider

    await bucket_mgr.create(content="夜班前的记忆", name="seed", bucket_type="dynamic")
    root = Path(test_config["buckets_dir"])
    ledger = LMC5Ledger(root / ".lmc5" / "pipeline.sqlite3", maintenance_root=root)
    embedding = _PausingEmbedding()
    coordinator = NightRunCoordinator(
        ledger=ledger,
        snapshots=SnapshotManager(root, tmp_path / "night-snapshots"),
        proposer=StrictOmbreProposer(
            _Provider(), timeout_seconds=1, model="test-model", provider_name="test"
        ),
        curated=CuratedWriteCoordinator(bucket_mgr, embedding),
        decay_engine=DecayEngine(test_config, bucket_mgr),
        consolidation_engine=ConsolidationEngine(test_config, bucket_mgr, embedding),
        bucket_manager=bucket_mgr,
    )
    ledger.append_raw_event("room-main", "bounded-1", '{"message":"朝灯今晚想看星星"}')
    configure_shared_lease_timeout(0.2)

    run_task = asyncio.create_task(
        coordinator.run(run_id="night-bounded-wait-1", cutoff=datetime.now(timezone.utc))
    )
    await asyncio.wait_for(embedding.paused.wait(), timeout=10)

    # Outside async writer (HTTP/MCP pattern) -> bounded busy failure.
    started = time.monotonic()
    with pytest.raises(MaintenanceBarrierTimeout):
        async with bucket_mgr._maintenance_barrier.shared_async():
            pass
    assert time.monotonic() - started < 3.0

    # Outside raw-event thread (daemon-thread ledger append) -> bounded too.
    started = time.monotonic()
    with pytest.raises(MaintenanceBarrierTimeout):
        await asyncio.to_thread(
            ledger.append_raw_event,
            "room-main",
            "outside-while-night-runs",
            '{"message":"outside"}',
        )
    assert time.monotonic() - started < 3.0
    assert not run_task.done()

    embedding.resume.set()
    outcome = await asyncio.wait_for(run_task, timeout=20)
    assert outcome.run.stage == "complete"
    assert outcome.counts["x_ready"] == 1
    assert outcome.counts["dispatch_attempted"] == 1
    assert outcome.counts["dispatch_deferred_budget"] == 0

    # After the night run the outside writers get in normally.
    async with bucket_mgr._maintenance_barrier.shared_async():
        pass
    ledger.append_raw_event("room-main", "after-night", '{"message":"after"}')


# ---------------------------------------------------------------------------
# HTTP mapping
# ---------------------------------------------------------------------------


class _HookRequest:
    def __init__(self, body):
        self._body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.headers = {"x-ombre-hook-token": "hook-secret"}
        self.client = SimpleNamespace(host="127.0.0.1")

    async def body(self):
        return self._body


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_raw_event_route_maps_lease_timeout_to_existing_busy_code(
    monkeypatch, test_config
):
    import lmc5_ingest_guard
    import server

    lock_path = lmc5_ingest_guard.RAW_INGEST_LOCK_PATH.with_name(
        f"ombre-lmc5-raw-ingest-test-{uuid4().hex}.lock"
    )
    monkeypatch.setattr(lmc5_ingest_guard, "RAW_INGEST_LOCK_PATH", lock_path)
    monkeypatch.setenv("OMBRE_HOOK_TOKEN", "hook-secret")
    monkeypatch.setitem(server.config, "buckets_dir", test_config["buckets_dir"])

    class TimedOutLedger:
        def append_raw_events(self, events):
            raise barrier_module._timeout_error(0.01)

    monkeypatch.setattr(server, "_get_lmc5_ledger", lambda: TimedOutLedger())
    try:
        response = await server.lmc5_raw_events_hook(
            _HookRequest(
                {
                    "schema_version": 1,
                    "session_id": "session-maintenance",
                    "events": [
                        {"source_event_id": "event-1", "payload": '{"uuid":"event-1"}'}
                    ],
                }
            )
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "error": "raw-event ingest is paused",
        "code": "raw_ingest.busy",
    }


@pytest.mark.asyncio
async def test_api_hold_maps_lease_timeout_to_503_busy(monkeypatch):
    import server

    async def busy_hold(**_kwargs):
        raise barrier_module._timeout_error(60.0)

    monkeypatch.setattr(server, "hold", busy_hold)
    response = await server.api_hold(_JsonRequest({"content": "维护时写入"}))
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "error": "memory maintenance is in progress; retry later",
        "code": MAINTENANCE_BUSY_CODE,
    }


def test_server_import_applies_the_configured_bound():
    import server

    # The autouse fixture imports server before capturing/restoring, so the
    # value observed here is the one server.py applied at import time.
    assert shared_lease_timeout() == shared_lease_timeout_from_config(server.config)
    assert shared_lease_timeout() is not None
