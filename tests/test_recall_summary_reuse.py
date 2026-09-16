"""Exact-body recall reuse; fake provider only, no vault writes."""
import asyncio
import hashlib
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import dehydrator as module
from dehydrator import Dehydrator

BODY = "Synthetic event evidence with dates and actions. " * 80
SUMMARY = "The synthetic event preserved its dates and actions."


class Provider:
    def __init__(self):
        self.requests = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.result = SUMMARY
        self.cancelled = False

    async def create(self, **request):
        self.requests.append(request)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if isinstance(self.result, Exception):
            raise self.result
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.result))])


def make(config):
    instance = Dehydrator(config)
    provider = Provider()
    instance.api_available = True
    instance.client = SimpleNamespace(chat=SimpleNamespace(completions=provider))
    return instance, provider


@pytest.mark.asyncio
async def test_same_body_reuses_memory_and_persistent_cache_only(test_config, caplog):
    d, p = make(test_config)
    legacy = Path(d.cache_db_path).read_bytes()
    with caplog.at_level(logging.INFO, logger="ombre_brain.dehydrator"):
        assert await d.dehydrate_recall_with_source(BODY) == (SUMMARY, "computed")
        assert await d.dehydrate_recall_with_source(BODY) == (SUMMARY, "memory_hit")
        restarted, other = make(test_config)
        assert await restarted.dehydrate_recall_with_source(BODY) == (SUMMARY, "persistent_hit")
    assert len(p.requests) == 1 and not other.requests
    assert Path(d.cache_db_path).read_bytes() == legacy
    assert "event=memory_hit" in caplog.text and "saved_calls=1" in caplog.text
    assert BODY not in caplog.text


@pytest.mark.asyncio
async def test_singleflight_one_cancel_does_not_cancel_other(test_config):
    d, p = make(test_config)
    p.release.clear()
    first = asyncio.create_task(d.dehydrate_recall_with_source(BODY))
    await p.started.wait()
    second = asyncio.create_task(d.dehydrate_recall_with_source(BODY))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    p.release.set()
    assert await second == (SUMMARY, "coalesced_hit")
    assert len(p.requests) == 1 and not p.cancelled
    assert not d._recall_summary_flights


@pytest.mark.asyncio
async def test_all_waiters_cancel_stops_provider_and_does_not_cache(test_config):
    d, p = make(test_config)
    p.release.clear()
    task = asyncio.create_task(d.dehydrate_recall_with_source(BODY))
    await p.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert p.cancelled and not d._recall_summary_flights
    p.release.set()
    assert await d.dehydrate_recall_with_source(BODY) == (SUMMARY, "computed")
    assert len(p.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("model", "another-model"), ("base_url", "https://another.invalid/v1"),
    ("max_tokens", 777), ("temperature", 0.8),
    ("recall_dehydration_disable_thinking", False),
    ("DEHYDRATE_PROMPT", "Another prompt"),
    ("RECALL_REDACTION_CONTRACT", "redaction/v99"),
    ("RECALL_OUTPUT_CONTRACT", "output/v99"),
])
async def test_output_contract_changes_miss(test_config, monkeypatch, field, value):
    d, p = make(test_config)
    d.recall_dehydration_disable_thinking = True
    await d.dehydrate_recall_with_source(BODY)
    monkeypatch.setattr(module if field.isupper() else d, field, value)
    assert (await d.dehydrate_recall_with_source(BODY))[1] == "computed"
    assert len(p.requests) == 2


@pytest.mark.asyncio
async def test_body_input_and_midflight_parameter_snapshot(test_config):
    d, p = make(test_config)
    old_model = d.model
    p.release.clear()
    task = asyncio.create_task(d.dehydrate_recall_with_source(BODY, raw_body=BODY + "[[raw]]"))
    await p.started.wait()
    d.model = "new-model"
    p.release.set()
    await task
    assert p.requests[0]["model"] == old_model
    await d.dehydrate_recall_with_source(BODY, raw_body=BODY + "[[raw]]")
    d.model = old_model
    assert (await d.dehydrate_recall_with_source(BODY, raw_body=BODY + "[[raw]]"))[1] == "memory_hit"
    await d.dehydrate_recall_with_source(BODY, raw_body=BODY + "[[different]]")
    await d.dehydrate_recall_with_source(BODY + "new input", raw_body=BODY + "[[raw]]")
    assert len(p.requests) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "tiny", RuntimeError("fake failure")])
async def test_failure_is_not_cached(test_config, bad):
    d, p = make(test_config)
    p.result = bad
    with pytest.raises(RuntimeError):
        await d.dehydrate_recall_with_source(BODY)
    p.result = SUMMARY
    assert await d.dehydrate_recall_with_source(BODY) == (SUMMARY, "computed")
    assert len(p.requests) == 2


@pytest.mark.asyncio
async def test_async_fallback_is_unchanged_and_coalesces(test_config):
    d, p = make(test_config)
    p.release.clear()
    fallback = " ".join(BODY.split())[:300] + "…"
    for _ in range(3):
        assert await d.dehydrate_recall_with_source(BODY, allow_async_fallback=True) == (fallback, "passthrough_async")
    await p.started.wait()
    p.release.set()
    await asyncio.gather(*(entry.task for entry in d._recall_summary_flights.values()))
    assert await d.dehydrate_recall_with_source(BODY) == (SUMMARY, "memory_hit")
    assert len(p.requests) == 1


@pytest.mark.asyncio
async def test_server_ignores_unversioned_frontmatter_and_never_writes_bucket(test_config, monkeypatch):
    import server
    d, p = make(test_config)
    monkeypatch.setattr(server, "dehydrator", d)
    monkeypatch.setattr(server, "_recall_dehydrate_async_enabled", lambda: False)
    class ForbiddenWriter:
        async def cache_recall_dehydration(self, *args, **kwargs):
            pytest.fail("recall must not write vault frontmatter")
    monkeypatch.setattr(server, "bucket_mgr", ForbiddenWriter())
    meta = {"dehydrated_summary": "Wrong old parameter summary", "dehydrated_content_hash": hashlib.sha256(BODY.encode()).hexdigest()}
    a = await server._dehydrate_for_recall(BODY, {"name": "First bucket"}, bucket={"id": "first", "content": BODY, "metadata": meta})
    b = await server._dehydrate_for_recall(BODY, {"name": "Second bucket"}, bucket={"id": "second", "content": BODY, "metadata": meta})
    assert SUMMARY in a and SUMMARY in b and "First bucket" not in b
    assert "Second bucket" not in a and len(p.requests) == 1


@pytest.mark.asyncio
async def test_working_hours_defer_background_provider_but_still_serve_cache(test_config, monkeypatch):
    import server
    d, p = make(test_config)
    monkeypatch.setattr(server, "dehydrator", d)
    monkeypatch.setattr(server, "_recall_dehydrate_async_enabled", lambda: True)
    monkeypatch.setattr(server, "_ds_offpeak_now", lambda: False)
    bucket = {"id": "synthetic", "content": BODY}
    result = await server._dehydrate_for_recall(BODY, {}, bucket=bucket, allow_async_fallback=True)
    await asyncio.sleep(0)
    assert result == " ".join(BODY.split())[:300] + "…"
    assert not p.requests and not d._recall_summary_flights
    # Explicit synchronous callers retain the existing real-time contract.
    await d.dehydrate_recall_with_source(BODY)
    assert await server._dehydrate_for_recall(BODY, {}, bucket=bucket, allow_async_fallback=True) == SUMMARY
    assert len(p.requests) == 1


@pytest.mark.asyncio
async def test_background_distinct_keys_keep_two_provider_limit(test_config):
    d, p = make(test_config)
    p.release.clear()
    for i in range(5):
        await d.dehydrate_recall_with_source(BODY + str(i), allow_async_fallback=True)
    await p.started.wait()
    await asyncio.sleep(0)
    assert len(p.requests) == 2
    p.release.set()
    await asyncio.gather(*(f.task for f in d._recall_summary_flights.values()))
    assert len(p.requests) == 5


@pytest.mark.asyncio
async def test_many_identical_cold_requests_have_one_producer(test_config, caplog):
    from tools.recall_summary_window import summarize
    import time
    d, p = make(test_config)
    p.release.clear()
    start = time.time()
    with caplog.at_level(logging.INFO, logger="ombre_brain.dehydrator"):
        callers = [asyncio.create_task(d.dehydrate_recall_with_source(BODY)) for _ in range(8)]
        await p.started.wait()
        p.release.set()
        results = await asyncio.gather(*callers)
    assert len(p.requests) == 1
    assert [source for _, source in results].count("coalesced_hit") == 7
    counters = summarize([r.getMessage() for r in caplog.records], start - 0.01, time.time() + 0.01)
    assert counters["eligible_requests"] == 8
    assert counters["saved_logical_calls"] == counters["cache_hits"] == 7
    assert counters["provider_invocations"] == 1


@pytest.mark.asyncio
async def test_strict_namespace_never_adopts_legacy_or_v2_summary(test_config):
    d, p = make(test_config)
    bad = "Untrusted old summary with an unknown output contract."
    d._set_cached_summary(BODY, bad)
    d._set_recall_cached_summary(BODY, bad)
    d._set_read_only_memory_summary(BODY, bad)
    legacy = Path(d.cache_db_path).read_bytes()
    assert await d.dehydrate_recall_with_source(BODY) == (SUMMARY, "computed")
    assert len(p.requests) == 1 and Path(d.cache_db_path).read_bytes() == legacy


@pytest.mark.asyncio
async def test_strict_sidecar_lock_fails_open_to_memory(test_config):
    d, p = make(test_config)
    lock = sqlite3.connect(d.recall_cache_db_path, timeout=0)
    lock.execute("BEGIN EXCLUSIVE")
    try:
        assert await d.dehydrate_recall_with_source(BODY) == (SUMMARY, "computed")
        assert await d.dehydrate_recall_with_source(BODY) == (SUMMARY, "memory_hit")
        assert len(p.requests) == 1
    finally:
        lock.rollback()
        lock.close()


@pytest.mark.asyncio
async def test_actual_bucket_and_authoritative_files_remain_byte_identical(test_config, monkeypatch):
    import server
    from bucket_manager import BucketManager
    manager = BucketManager(test_config)
    identifier = await manager.create(BODY, name="Synthetic unchanged vault")
    bucket = await manager.get(identifier)
    d, p = make(test_config)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", d)
    root = Path(test_config["buckets_dir"])
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*")
              if path.is_file() and ".recall_cache" not in path.parts}
    original_metadata = dict(bucket["metadata"])
    await server._dehydrate_for_recall(bucket["content"], original_metadata, bucket=bucket)
    after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*")
             if path.is_file() and ".recall_cache" not in path.parts}
    assert before == after
    assert (await manager.get(identifier))["metadata"] == original_metadata
    assert len(p.requests) == 1


@pytest.mark.asyncio
async def test_unavailable_api_preserves_async_fallback(test_config):
    d, p = make(test_config)
    d.api_available = False
    assert (await d.dehydrate_recall_with_source(BODY, allow_async_fallback=True))[1] == "passthrough_async"
    assert not p.requests and not d._recall_summary_flights
    with pytest.raises(RuntimeError):
        await d.dehydrate_recall_with_source(BODY)
