from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import pytest

from retrieval_hints import Generation, source_record
from retrieval_hints_batch import NightBatch, TZ, existing_host_lock, night_seconds_left
from retrieval_hints_storage import HintsStore
from tests.test_retrieval_hints import payload
from tests.test_retrieval_hints_storage import Peer, bucket


class Model:
    timeout_s = 15
    def __init__(self, generate=None):
        self.calls = 0
        self.fn = generate

    async def generate(self, body):
        self.calls += 1
        return self.fn() if self.fn else Generation("ok", payload(), {"output_tokens": 20}, outbound_calls=1)


def setup_batch(tmp_path, count=4, **kwargs):
    sources = {f"synthetic-{i}": {**bucket(), "id": f"synthetic-{i}"} for i in range(count)}
    approved = [source_record(b) for b in sources.values()]
    store = HintsStore(tmp_path / "hints")
    model = Model()
    args = dict(enabled=True, backfill_enabled=True, store=store, generator=model,
                peer=Peer(), load_source=sources.get, publish_lock=nullcontext, worker_lock=nullcontext,
                now=lambda: datetime(2026, 9, 18, 1, 0, tzinfo=TZ))
    args.update(kwargs)
    return NightBatch(**args), approved, store, args["generator"]


@pytest.mark.parametrize("hour,minute,expected", [(0,59,False),(1,0,True),(6,29,True),(6,30,False),(12,0,False)])
def test_night_boundaries(hour, minute, expected):
    assert bool(night_seconds_left(datetime(2026, 9, 18, hour, minute, tzinfo=TZ))) is expected


@pytest.mark.asyncio
async def test_day_and_off_are_zero_side_effect(tmp_path):
    for option in [{"enabled": False}, {"now": lambda: datetime(2026,9,18,12,tzinfo=TZ)}]:
        batch, approved, store, model = setup_batch(tmp_path, **option)
        result = await batch.run(approved)
        assert result["outbound_calls"] == 0 and model.calls == 0
        assert not store.path.exists()


@pytest.mark.asyncio
async def test_limit_and_idempotent_resume(tmp_path):
    batch, approved, store, model = setup_batch(tmp_path, limit=2)
    result = await batch.run(approved)
    assert model.calls == 2 and result["published"] == 2
    result = await batch.run(approved)
    assert model.calls == 4 and result["published"] == 4
    result = await batch.run(approved)
    assert model.calls == 4 and result["outbound_calls"] == 4
    assert all(j["attempts"] == 1 for j in store.jobs())


@pytest.mark.asyncio
async def test_three_generation_failures_stop_and_remain_stopped(tmp_path):
    model = Model(lambda: Generation("timeout", outbound_calls=1))
    batch, approved, store, _ = setup_batch(tmp_path, generator=model)
    result = await batch.run(approved)
    assert result["status"] == "stopped_three_failures" and model.calls == 3
    assert result["receipts"][0]["usage"] is None
    await batch.run(approved)
    assert model.calls == 3


@pytest.mark.asyncio
async def test_three_sync_failures_stop_with_durable_payloads(tmp_path):
    peer = Peer()
    peer.fail = True
    batch, approved, store, model = setup_batch(tmp_path, peer=peer)
    result = await batch.run(approved)
    assert result["status"] == "stopped_three_failures"
    assert model.calls == len(store.pending()) == 3


@pytest.mark.asyncio
async def test_network_is_outside_production_lock(tmp_path):
    held = False
    @contextmanager
    def lease():
        nonlocal held
        assert not held
        held = True
        try:
            yield
        finally:
            held = False
    def generate():
        assert not held
        return Generation("ok", payload(), outbound_calls=1)
    batch, approved, store, model = setup_batch(tmp_path, count=1, generator=Model(generate), publish_lock=lease)
    assert (await batch.run(approved))["published"] == 1


@pytest.mark.asyncio
async def test_no_post_window_writes_after_slow_provider(tmp_path):
    current = datetime(2026,9,18,6,29,tzinfo=TZ)
    def generate():
        nonlocal current
        current = datetime(2026,9,18,6,30,tzinfo=TZ)
        return Generation("ok", payload(), outbound_calls=1)
    batch, approved, store, model = setup_batch(tmp_path, count=1, generator=Model(generate), now=lambda: current)
    result = await batch.run(approved)
    assert result["status"] == "deferred_after_generation"
    assert store.pending() == []
    assert store.jobs()[0]["state"] == "inflight"  # Paid/uncertain, not retried.
    current = datetime(2026,9,19,1,tzinfo=TZ)
    await batch.run(approved)
    assert model.calls == 1


@pytest.mark.asyncio
async def test_oversize_manifest_cannot_expand_pilot(tmp_path):
    batch, approved, store, model = setup_batch(tmp_path, count=201)
    with pytest.raises(ValueError, match="200"):
        await batch.run(approved)
    assert not store.path.exists() and model.calls == 0


def test_host_lock_cannot_be_created_or_substituted(tmp_path):
    missing = tmp_path / "missing.lock"
    with pytest.raises(FileNotFoundError):
        with existing_host_lock(missing):
            pass
    assert not missing.exists()
    actual = tmp_path / "actual.lock"
    actual.touch()
    with pytest.raises(ValueError, match="identity"):
        with existing_host_lock(actual, expected_inode=actual.stat().st_ino + 1):
            pass
    with existing_host_lock(actual):
        with pytest.raises(BlockingIOError):
            with existing_host_lock(actual):
                pass
