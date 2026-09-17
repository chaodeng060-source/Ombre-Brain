"""Bounded night-window execution. Production locks are injected by the host.

No scheduler, service restart, full-corpus grant, or provider retry is implicit.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from retrieval_hints import PROMPT_VERSION, bound_source_context, canonical_json, content_hash
from retrieval_hints_storage import active

TZ = ZoneInfo("Asia/Shanghai")


def night_seconds_left(now):
    local = now.astimezone(TZ)
    minutes = local.hour * 60 + local.minute
    if not 60 <= minutes < 390:
        return 0.0
    return (local.replace(hour=6, minute=30, second=0, microsecond=0) - local).total_seconds()


@contextmanager
def existing_host_lock(path, *, expected_device=None, expected_inode=None):
    """Never create a look-alike container lock. Validate the host bind identity."""
    with Path(path).open("r+") as stream:
        stat = os.fstat(stream.fileno())
        if ((expected_device is not None and stat.st_dev != expected_device)
                or (expected_inode is not None and stat.st_ino != expected_inode)):
            raise ValueError("host_lock_identity_mismatch")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class NightBatch:
    def __init__(self, *, enabled, backfill_enabled, store, generator, peer, load_source,
                 publish_lock, worker_lock, maintenance_lock=nullcontext,
                 now=lambda: datetime.now(TZ), monotonic=time.monotonic, busy=lambda: False,
                 tokenizer=None, limit=200, max_seconds=1800, load_source_context=lambda bucket: ()):
        if not 1 <= limit <= 200 or not 0 < max_seconds <= 1800:
            raise ValueError("pilot_budget")
        self.enabled, self.backfill_enabled = enabled, backfill_enabled
        self.store, self.generator, self.peer, self.load_source = store, generator, peer, load_source
        self.publish_lock, self.worker_lock, self.maintenance_lock = publish_lock, worker_lock, maintenance_lock
        self.now, self.monotonic, self.busy = now, monotonic, busy
        self.tokenizer, self.limit, self.max_seconds = tokenizer, limit, max_seconds
        self.load_source_context = load_source_context
        self.schema_version = getattr(generator, "schema_version", 1)
        self.prompt_version = getattr(generator, "prompt_version", PROMPT_VERSION)

    def _room(self, deadline, seconds=0):
        return night_seconds_left(self.now()) > seconds and deadline - self.monotonic() > seconds

    @contextmanager
    def _write(self, deadline):
        if not self._room(deadline):
            raise TimeoutError("night_or_batch_deadline")
        with self.publish_lock():
            if not self._room(deadline):
                raise TimeoutError("night_or_batch_deadline")
            yield

    async def run(self, approved_sources):
        if not self.enabled:
            return {"status": "disabled", "outbound_calls": 0}
        if not night_seconds_left(self.now()):
            return {"status": "outside_night_window", "outbound_calls": 0}
        if len(approved_sources) > 200 or len({r["bucket_id"] for r in approved_sources}) != len(approved_sources):
            raise ValueError("pilot_requires_at_most_200_distinct_ids")
        batch_id = content_hash(canonical_json(approved_sources if self.schema_version == 1 else
                                  {"sources": approved_sources, "prompt_version": self.prompt_version}))
        deadline = self.monotonic() + self.max_seconds
        try:
            with self.worker_lock():
                return await self._run(approved_sources, batch_id, deadline)
        except BlockingIOError:
            return {"status": "busy", "outbound_calls": 0, "batch_id": batch_id}

    async def _run(self, approved, batch_id, deadline):
        state = self.store.get_batch_state(batch_id) or {
            "status": "running", "outbound_calls": 0, "generated": 0, "published": 0,
            "failures": {}, "receipts": [], "batch_id": batch_id,
        }
        if state["status"] == "stopped_three_failures":
            return state
        state["status"] = "running"
        allowed = {row["bucket_id"]: row for row in approved}
        new_ids = {j["bucket_id"] for j in self.store.jobs() if j["kind"] == "new_write"}
        approved = sorted(approved, key=lambda row: row["bucket_id"] not in new_ids)
        processed = 0
        for item in approved:
            if self.busy() or not self._room(deadline):
                state["status"] = "deferred"
                break
            bid = item["bucket_id"]
            bucket = self.load_source(bid)
            if (not active(bucket) or content_hash(bucket.get("content") or "") != item["source_content_sha256"]):
                state["receipts"].append({"bucket_id": bid, "status": "source_changed_or_inactive"})
                continue
            # Restore interrupted PG publication before considering any model call.
            pending = [r for r in self.store.pending() if r["bucket_id"] == bid
                       and r["source_content_sha256"] == item["source_content_sha256"]
                       and r["prompt_version"] == self.prompt_version]
            published = self.store.lookup(bucket)
            if published is not None and published["payload"]["schema_version"] >= self.schema_version:
                continue
            if not pending:
                known = [j for j in self.store.jobs() if j["bucket_id"] == bid
                         and json.loads(j["source_json"])["source_content_sha256"] == item["source_content_sha256"]
                         and json.loads(j["source_json"])["prompt_version"] == self.prompt_version]
                if known and known[0]["attempts"]:
                    continue  # timeout/crash may already be billed; never blind retry.
                if not self.backfill_enabled and bid not in new_ids:
                    continue
                spent = sum(j["attempts"] for j in self.store.jobs() if j["bucket_id"] in allowed)
                if spent >= 200 or processed >= self.limit or not self._room(deadline, self.generator.timeout_s + 2):
                    state["status"] = "budget_or_window_deferred"
                    break
                with self._write(deadline):
                    key = self.store.register(bucket, kind="new_write" if bid in new_ids else "backfill",
                                              schema_version=self.schema_version)
                    if not self.store.reserve(key):
                        continue
                # NO production/maintenance lease is held during this await.
                generation_options = ({"source_context": bound_source_context(bucket, self.load_source_context(bucket))}
                                      if self.schema_version == 2 else {})
                result = await self.generator.generate(bucket.get("content") or "", **generation_options)
                processed += 1
                state["outbound_calls"] += result.outbound_calls
                state["receipts"].append({"bucket_id": bid, "status": result.status, "usage": result.usage,
                    "requested_model": result.requested_model, "response_model": result.response_model,
                    "elapsed_ms": result.elapsed_ms, "outbound_calls": result.outbound_calls, "error": result.error})
                if not self._room(deadline):
                    state["status"] = "deferred_after_generation"
                    return state  # reservation remains uncertain, not retried.
                with self._write(deadline):
                    if result.status == "ok":
                        version = self.store.stage(bucket, result, tokens=self.tokenizer,
                            lexical_version=getattr(self.tokenizer, "version", "jieba-search-v1"))
                        pending = [self.store._version(bid, version)]
                        state["generated"] += 1
                    self.store.finish_job(key, result)
                    failures = state["failures"]
                    failures["generation"] = 0 if result.status == "ok" else failures.get("generation", 0) + 1
                    if failures["generation"] >= 3:
                        state["status"] = "stopped_three_failures"
                    self.store.set_batch_state(batch_id, state)
                if state["status"] == "stopped_three_failures":
                    return state
                if result.status != "ok":
                    continue
            try:
                with self._write(deadline), self.maintenance_lock():
                    if not self._room(deadline):
                        raise TimeoutError("night_or_batch_deadline")
                    self.store.publish(bid, pending[0]["version"], self.peer, lambda: self.load_source(bid))
                    state["published"] += 1
                    state["failures"]["sync"] = 0
                    self.store.set_batch_state(batch_id, state)
            except BlockingIOError:
                state["status"] = "busy"
                return state
            except TimeoutError:
                state["status"] = "deferred"
                return state
            except Exception as exc:
                state["failures"]["sync"] = state["failures"].get("sync", 0) + 1
                state["receipts"].append({"bucket_id": bid, "status": "sync_error", "error": type(exc).__name__})
                if state["failures"]["sync"] >= 3:
                    state["status"] = "stopped_three_failures"
                with self._write(deadline):
                    self.store.set_batch_state(batch_id, state)
                if state["status"] == "stopped_three_failures":
                    return state
        if state["status"] == "running":
            state["status"] = "pass_finished"  # Not quality acceptance or full completion.
        if self._room(deadline):
            with self._write(deadline):
                self.store.set_batch_state(batch_id, state)
        return state
