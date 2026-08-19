from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from curated_writer import CuratedWriteCoordinator, CuratedWriteResult
from lmc5_ledger import LMC5Ledger
from lmc5_proposer import (
    ProposerBatch,
    ProposerContractError,
    StrictOmbreProposer,
)
from maintenance_barrier import MaintenanceBarrier
from review_queue import ReviewQueue
from night_run_coordinator import (
    NightRunCoordinator,
    NightRunCoordinatorError,
    NightRunPolicy,
)
from snapshot_manager import SnapshotManager


class _Provider:
    def __init__(
        self,
        *,
        candidate_type: str = "event",
        risk: str = "normal",
        invalid: bool = False,
        empty: bool = False,
    ) -> None:
        self.candidate_type = candidate_type
        self.risk = risk
        self.invalid = invalid
        self.empty = empty
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        if self.invalid:
            return {"choices": []}
        raw_input = prompt.split("INPUT=", 1)[1]
        proposer_input = json.loads(raw_input)
        chunk = proposer_input["chunks"][0]
        candidates: list[dict[str, Any]] = []
        if not self.empty:
            candidates.append(
                {
                    "type": self.candidate_type,
                    "title": "夜间候选",
                    "content": "朝灯今晚想看星星",
                    "importance": 7,
                    "thread_hint": "night",
                    "relation_hints": [],
                    "source_chunk_ids": [chunk["id"]],
                    "evidence": chunk["text"][:12],
                    "risk": self.risk,
                }
            )
        content = json.dumps(
            {"schema_version": 1, "candidates": candidates},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": content},
                }
            ]
        }


class _FakeCurated:
    def __init__(self, barrier: MaintenanceBarrier) -> None:
        self._maintenance_barrier = barrier
        self.calls: list[dict[str, Any]] = []
        self.results: dict[str, CuratedWriteResult] = {}

    async def write(self, **kwargs: Any) -> CuratedWriteResult:
        self.calls.append(kwargs)
        key = kwargs["idempotency_key"]
        if key in self.results:
            return self.results[key]
        result = CuratedWriteResult(
            success=True,
            status="completed",
            bucket_id=f"bucket-{len(self.results) + 1}",
            vector_policy="required",
            recall_state="ready_vector",
        )
        self.results[key] = result
        return result


class _AxisBucketManager:
    def __init__(self, barrier: MaintenanceBarrier, buckets: dict[str, dict]) -> None:
        self._maintenance_barrier = barrier
        self.buckets = buckets
        self.relations: list[dict[str, Any]] = []

    async def get(self, bucket_id: str):
        return self.buckets.get(bucket_id)

    async def list_all(self, include_archive: bool = False):
        return list(self.buckets.values())

    async def set_thread(
        self,
        bucket_id: str,
        thread: str,
        **_kwargs,
    ) -> bool:
        bucket = self.buckets.get(bucket_id)
        if bucket is None:
            return False
        bucket.setdefault("metadata", {})["thread"] = thread
        return True

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        note: str = "",
        strength: float | None = None,
        actor: str = "system",
    ) -> bool:
        self.relations.append({
            "source": source_id,
            "target": target_id,
            "type": rel_type,
            "note": note,
            "strength": strength,
            "actor": actor,
        })
        return True


class _RetryEmbedding:
    def __init__(self) -> None:
        self.enabled = True
        self.calls = 0
        self.stored: set[str] = set()

    async def generate_and_store(
        self,
        bucket_id: str,
        content: str,
    ) -> bool:
        self.calls += 1
        if self.calls == 1:
            return False
        self.stored.add(bucket_id)
        return True

    async def get_embedding(self, bucket_id: str):
        return [1.0] if bucket_id in self.stored else None

    def delete_embedding(self, bucket_id: str) -> None:
        self.stored.discard(bucket_id)

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        return 1.0 if a == b else 0.0


class _ReportEngine:
    def __init__(
        self,
        barrier: MaintenanceBarrier,
        *,
        kind: str,
        unsafe: bool = False,
    ) -> None:
        self.bucket_mgr = SimpleNamespace(_maintenance_barrier=barrier)
        self.metabolism_mode = "report_only"
        self.kind = kind
        self.unsafe = unsafe
        self.calls = 0

    async def run_decay_cycle(self) -> dict[str, Any]:
        assert self.kind == "decay"
        self.calls += 1
        return {
            "ok": True,
            "mode": "report_only",
            "checked": 1,
            "archived": 1 if self.unsafe else 0,
            "auto_resolved": 0,
            "would_archive": ["candidate"],
            "would_auto_resolve": [],
            "lowest_score": 0.2,
            "errors": [],
        }

    async def run_consolidation_cycle(self) -> dict[str, Any]:
        assert self.kind == "consolidation"
        self.calls += 1
        return {
            "ok": True,
            "mode": "report_only",
            "dup_pairs": 1,
            "stale_count": 0,
            "auto_digested": 1 if self.unsafe else 0,
            "would_digest": ["candidate"],
            "would_create_report": True,
            "duplicate_candidates": [],
            "stale_candidates": [],
            "report_bucket_id": None,
            "errors": [],
        }


@dataclass
class _Harness:
    source: Path
    backups: Path
    ledger: LMC5Ledger
    snapshots: SnapshotManager
    provider: _Provider
    curated: _FakeCurated
    decay: _ReportEngine
    consolidation: _ReportEngine
    coordinator: NightRunCoordinator


def _harness(
    tmp_path: Path,
    *,
    candidate_type: str = "event",
    risk: str = "normal",
    invalid_provider: bool = False,
    empty_provider: bool = False,
    unsafe_decay: bool = False,
    policy: NightRunPolicy | None = None,
) -> _Harness:
    source = tmp_path / "vault"
    backups = tmp_path / "snapshots"
    source.mkdir()
    (source / "dynamic").mkdir()
    ledger = LMC5Ledger(
        source / ".lmc5" / "ledger.db",
        maintenance_root=source,
    )
    snapshots = SnapshotManager(source, backups)
    barrier = MaintenanceBarrier(source)
    provider = _Provider(
        candidate_type=candidate_type,
        risk=risk,
        invalid=invalid_provider,
        empty=empty_provider,
    )
    proposer = StrictOmbreProposer(
        provider,
        timeout_seconds=1,
        model="test-model",
        provider_name="test-provider",
    )
    curated = _FakeCurated(barrier)
    decay = _ReportEngine(
        barrier, kind="decay", unsafe=unsafe_decay
    )
    consolidation = _ReportEngine(barrier, kind="consolidation")
    coordinator = NightRunCoordinator(
        ledger=ledger,
        snapshots=snapshots,
        proposer=proposer,
        curated=curated,
        decay_engine=decay,
        consolidation_engine=consolidation,
        policy=policy,
    )
    return _Harness(
        source=source,
        backups=backups,
        ledger=ledger,
        snapshots=snapshots,
        provider=provider,
        curated=curated,
        decay=decay,
        consolidation=consolidation,
        coordinator=coordinator,
    )


@pytest.mark.asyncio
async def test_event_run_completes_snapshot_chunk_x_and_report_only_m(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    harness.ledger.append_raw_event(
        "room-main",
        "message-1",
        json.dumps(
            {
                "message": "朝灯今晚想看星星",
                "api_key": "must-not-leave-local",
            },
            ensure_ascii=False,
        ),
    )
    cutoff = datetime.now(timezone.utc)

    outcome = await harness.coordinator.run(
        run_id="night-success-1",
        cutoff=cutoff,
    )

    assert outcome.run.stage == "complete"
    assert {
        key: outcome.counts[key]
        for key in (
            "raw_events",
            "chunks",
            "proposer_chunks",
            "candidates",
            "x_ready",
            "m_computed",
        )
    } == {
        "raw_events": 1,
        "chunks": 1,
        "proposer_chunks": 1,
        "candidates": 2,
        "x_ready": 1,
        "m_computed": 1,
    }
    assert outcome.counts["proposer_attempted"] == 1
    assert outcome.counts["proposer_succeeded"] == 1
    assert outcome.counts["proposer_pending_after"] == 0
    assert {
        outcome.counts[key]
        for key in (
            "timeline_scanned",
            "timeline_assigned",
            "timeline_named",
            "timeline_updated",
            "timeline_new_lines",
            "timeline_orphans",
        )
    } == {0}
    assert len(harness.curated.calls) == 1
    assert harness.curated.calls[0]["vector_policy"] == "required"
    assert (
        harness.curated.calls[0]["bucket_options"]["x_provenance"][
            "source_event_ids"
        ]
        == ["message-1"]
    )
    assert "must-not-leave-local" not in harness.provider.prompts[0]
    assert "[REDACTED]" in harness.provider.prompts[0]
    assert len(harness.ledger.list_candidates("ready")) == 2
    assert harness.ledger.list_candidates("pending") == ()
    assert harness.ledger.get_night_run("night-success-1").sequence == 7
    verified = harness.snapshots.verify_snapshot(
        "night-success-1",
        expected_manifest_sha256=outcome.snapshot_manifest_sha256,
    )
    assert verified.snapshot_id == "night-success-1"


@pytest.mark.asyncio
async def test_report_only_receipt_flush_yields_during_large_backlog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path)
    harness.ledger.append_raw_event(
        "room-main",
        "m-source",
        '{"message":"report-only receipt source"}',
    )
    harness.ledger.record_event_chunk(
        "m-chunk",
        "report-only receipt source",
        [("room-main", "m-source")],
    )
    for index in range(18):
        harness.ledger.record_candidate(
            f"m-candidate-{index}",
            "M",
            json.dumps({"index": index}),
            ["m-chunk"],
        )

    original_sleep = asyncio.sleep
    yields: list[float] = []

    async def observed_sleep(delay: float) -> None:
        yields.append(delay)
        await original_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", observed_sleep)
    counts: dict[str, int] = {}

    await harness.coordinator._run_metabolism(counts)

    assert counts["m_computed"] == 18
    assert harness.ledger.list_candidates("pending") == ()
    assert yields == [0]


@pytest.mark.asyncio
async def test_cutoff_is_exclusive_and_newer_event_remains_uncovered(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, empty_provider=True)
    harness.ledger.append_raw_event(
        "room-main", "before", '{"message":"before cutoff"}'
    )
    cutoff = datetime.now(timezone.utc)
    harness.ledger.append_raw_event(
        "room-main", "after", '{"message":"after cutoff"}'
    )

    outcome = await harness.coordinator.run(
        run_id="night-cutoff-1",
        cutoff=cutoff,
    )

    assert outcome.run.stage == "complete"
    uncovered = harness.ledger.list_uncovered_raw_events(limit=10)
    assert [row.identity.source_event_id for row in uncovered] == ["after"]
    assert outcome.counts["raw_events"] == 1
    assert outcome.counts["candidates"] == 0
    assert harness.curated.calls == []


@pytest.mark.asyncio
async def test_preference_writes_x_but_defers_unwired_z_and_e(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, candidate_type="preference")
    harness.ledger.append_raw_event(
        "room-main", "preference-1", '{"message":"我更喜欢墨绿色"}'
    )

    outcome = await harness.coordinator.run(
        run_id="night-defer-1",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.run.stage == "complete"
    assert len(harness.curated.calls) == 1
    assert harness.curated.calls[0]["bucket_options"]["tags"] == [
        "lmc5", "night", "preference"
    ]
    deferred = harness.ledger.list_candidates("deferred")
    assert {(row.axis, row.error_code) for row in deferred} == {
        ("Z", "z.storage_unavailable"),
        ("E", "e.proposal_storage_unavailable"),
    }
    ready = harness.ledger.list_candidates("ready")
    assert {(row.axis, row.error_code) for row in ready} == {
        ("M", None),
        ("X", None),
    }


@pytest.mark.asyncio
async def test_night_y_writes_safe_relation_and_queues_dangerous_with_strength(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    barrier = harness.coordinator.maintenance_barrier
    buckets = {
        "bucket-1": {
            "id": "bucket-1",
            "content": "朝灯今晚想看星星",
            "metadata": {"id": "bucket-1", "name": "夜间候选", "type": "dynamic"},
        },
        "safe-target": {
            "id": "safe-target",
            "content": "一起看过月亮",
            "metadata": {"id": "safe-target", "name": "月亮", "type": "dynamic"},
        },
        "danger-target": {
            "id": "danger-target",
            "content": "后来睡得很晚",
            "metadata": {"id": "danger-target", "name": "晚睡", "type": "dynamic"},
        },
    }
    manager = _AxisBucketManager(barrier, buckets)
    queue = ReviewQueue(
        harness.source / "review_queue.jsonl",
        maintenance_root=harness.source,
    )
    harness.coordinator.bucket_manager = manager
    harness.coordinator.review_queue = queue

    async def targets(_text: str) -> frozenset[str]:
        return frozenset({"safe-target", "danger-target"})

    def provider(prompt: str) -> dict[str, Any]:
        proposer_input = json.loads(prompt.split("INPUT=", 1)[1])
        chunk = proposer_input["chunks"][0]
        content = json.dumps({
            "schema_version": 1,
            "candidates": [{
                "type": "event",
                "title": "夜间候选",
                "content": "朝灯今晚想看星星",
                "importance": 7,
                "thread_hint": "night",
                "relation_hints": [
                    {
                        "relation_type": "kin",
                        "target_id": "safe-target",
                        "strength": 0.8,
                        "reason": "同类夜空记忆",
                    },
                    {
                        "relation_type": "causes",
                        "target_id": "danger-target",
                        "strength": 0.6,
                        "reason": "可能导致晚睡",
                    },
                ],
                "source_chunk_ids": [chunk["id"]],
                "evidence": "看星星",
                "risk": "normal",
            }],
        }, ensure_ascii=False)
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    harness.coordinator.proposer = StrictOmbreProposer(provider)
    harness.coordinator.relation_target_provider = targets
    harness.ledger.append_raw_event(
        "room-main", "relation-1", '{"message":"朝灯今晚想看星星"}'
    )

    outcome = await harness.coordinator.run(
        run_id="night-y-ready",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.counts["y_safe_ready"] == 1
    assert outcome.counts["y_review_ready"] == 1
    assert manager.relations == [{
        "source": "bucket-1",
        "target": "safe-target",
        "type": "kin",
        "note": "同类夜空记忆",
        "strength": 0.8,
        "actor": "lmc5:night:y-safe",
    }]
    pending = queue.list_pending("relation")
    assert [(row["rel_type"], row["strength"]) for row in pending] == [
        ("causes", 0.6)
    ]
    assert {row.axis for row in harness.ledger.list_candidates("ready")} == {
        "M", "X", "Y"
    }


@pytest.mark.asyncio
async def test_night_z_queues_registered_fact_pair_without_applying_lifecycle(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, candidate_type="preference")
    barrier = harness.coordinator.maintenance_barrier
    buckets = {
        "bucket-1": {
            "id": "bucket-1",
            "content": "主色: 墨绿",
            "metadata": {
                "id": "bucket-1",
                "name": "新主色",
                "type": "dynamic",
                "recorded_at": "2026-08-11T17:00:00+08:00",
            },
        },
        "old-color": {
            "id": "old-color",
            "content": "主色: 浅绿",
            "metadata": {
                "id": "old-color",
                "name": "旧主色",
                "type": "dynamic",
                "recorded_at": "2026-08-01T12:00:00+08:00",
            },
        },
    }
    manager = _AxisBucketManager(barrier, buckets)
    queue = ReviewQueue(
        harness.source / "review_queue.jsonl",
        maintenance_root=harness.source,
    )
    harness.coordinator.bucket_manager = manager
    harness.coordinator.review_queue = queue
    harness.coordinator.fact_slot_registry = {"preference.ui.primary_color": frozenset({"主色"})}
    harness.coordinator._fact_slot_config = {
        "preference.ui.primary_color": {"aliases": ["主色"]}
    }

    def provider(prompt: str) -> dict[str, Any]:
        proposer_input = json.loads(prompt.split("INPUT=", 1)[1])
        chunk = proposer_input["chunks"][0]
        content = json.dumps({
            "schema_version": 1,
            "candidates": [{
                "type": "preference",
                "title": "新主色",
                "content": "主色: 墨绿",
                "importance": 8,
                "thread_hint": "ui",
                "relation_hints": [],
                "source_chunk_ids": [chunk["id"]],
                "evidence": "墨绿",
                "risk": "normal",
            }],
        }, ensure_ascii=False)
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    harness.coordinator.proposer = StrictOmbreProposer(provider)
    harness.ledger.append_raw_event(
        "room-main", "fact-1", '{"message":"主色改成墨绿"}'
    )

    outcome = await harness.coordinator.run(
        run_id="night-z-ready",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.counts["z_review_ready"] == 1
    pending = queue.list_pending("z_conflict")
    assert len(pending) == 1
    assert pending[0]["fact_key"] == "preference.ui.primary_color"
    assert pending[0]["current_bucket_id"] == "bucket-1"
    assert pending[0]["historical_bucket_id"] == "old-color"
    assert "fact_status" not in buckets["bucket-1"]["metadata"]
    assert "fact_status" not in buckets["old-color"]["metadata"]


@pytest.mark.asyncio
async def test_provider_failure_is_retryable_and_run_is_deferred(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, invalid_provider=True)
    harness.ledger.append_raw_event(
        "room-main", "message-1", '{"message":"provider failure"}'
    )

    outcome = await harness.coordinator.run(
        run_id="night-provider-error",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.run.stage == "deferred"
    run = harness.ledger.get_night_run("night-provider-error")
    assert run.stage == "deferred"
    assert run.errors == ()
    assert run.counts["proposer_attempted"] == 1
    assert run.counts["proposer_retryable"] == 1
    assert run.counts["proposer_pending_after"] == 1
    assert len(harness.ledger.list_pending_proposer_chunks(limit=10)) == 1
    with sqlite3.connect(harness.ledger.path) as connection:
        assert connection.execute(
            "SELECT outcome, error_code FROM chunk_proposer_outcomes"
        ).fetchall() == [("retryable_error", "provider.no_choices")]


@pytest.mark.asyncio
async def test_retryable_head_chunk_does_not_block_later_proposals(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    prompts: list[str] = []

    async def provider(prompt: str) -> dict[str, Any]:
        prompts.append(prompt)
        raw_input = prompt.split("INPUT=", 1)[1]
        proposer_input = json.loads(raw_input)
        chunk = proposer_input["chunks"][0]
        if "poison proposal" in chunk["text"]:
            return {"choices": []}
        content = json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "type": "event",
                        "title": "健康候选",
                        "content": "健康候选仍被提议。",
                        "importance": 6,
                        "thread_hint": "night",
                        "relation_hints": [],
                        "source_chunk_ids": [chunk["id"]],
                        "evidence": "healthy proposal",
                        "risk": "normal",
                    }
                ],
            },
            ensure_ascii=False,
        )
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": content},
                }
            ]
        }

    harness.coordinator.proposer = StrictOmbreProposer(provider)
    harness.ledger.append_raw_event(
        "room-main", "poison", '{"message":"poison proposal"}'
    )
    harness.ledger.append_raw_event(
        "room-main", "healthy", '{"message":"healthy proposal"}'
    )

    outcome = await harness.coordinator.run(
        run_id="night-head-skip",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.run.stage == "deferred"
    run = harness.ledger.get_night_run("night-head-skip")
    assert run.stage == "deferred"
    assert run.counts["proposer_errors"] == 1
    assert run.counts["proposer_chunks"] == 1
    assert len(prompts) == 2
    pending = harness.ledger.list_pending_proposer_chunks(limit=10)
    assert len(pending) == 1
    assert b"poison proposal" in pending[0].content
    with sqlite3.connect(harness.ledger.path) as connection:
        assert connection.execute(
            "SELECT outcome, error_code FROM chunk_proposer_outcomes "
            "ORDER BY id"
        ).fetchall() == [
            ("retryable_error", "provider.no_choices"),
            ("candidates_persisted", None),
        ]
    assert harness.ledger.list_candidates("pending") == ()
    assert len(harness.curated.calls) == 1


@pytest.mark.asyncio
async def test_proposer_cap_defers_then_next_run_drains_without_repeats(
    tmp_path: Path,
) -> None:
    harness = _harness(
        tmp_path,
        empty_provider=True,
        policy=NightRunPolicy(proposer_max_chunks_per_run=2),
    )
    for index in range(3):
        harness.ledger.append_raw_event(
            "room-main",
            f"event-{index}",
            json.dumps({"message": f"bounded-{index}"}),
        )
    cutoff = datetime.now(timezone.utc)

    first = await harness.coordinator.run(
        run_id="night-bounded-1",
        cutoff=cutoff,
    )
    assert first.run.stage == "deferred"
    assert first.counts["proposer_attempted"] == 2
    assert first.counts["proposer_succeeded"] == 2
    assert first.counts["proposer_pending_before"] == 3
    assert first.counts["proposer_pending_after"] == 1

    second = await harness.coordinator.run(
        run_id="night-bounded-2",
        cutoff=cutoff,
    )
    assert second.run.stage == "complete"
    assert second.counts["proposer_attempted"] == 1
    assert second.counts["proposer_pending_before"] == 1
    assert second.counts["proposer_pending_after"] == 0
    assert len(harness.provider.prompts) == 3


@pytest.mark.asyncio
async def test_proposer_concurrency_parallelizes_independent_chunks(
    tmp_path: Path,
) -> None:
    harness = _harness(
        tmp_path,
        policy=NightRunPolicy(
            proposer_max_chunks_per_run=4,
            proposer_concurrency=4,
        ),
    )

    class _ConcurrentEmptyProposer:
        def __init__(self) -> None:
            self.active = 0
            self.peak = 0
            self.release = asyncio.Event()

        async def propose(self, *args: Any, **kwargs: Any) -> ProposerBatch:
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.peak == 4:
                self.release.set()
            try:
                await asyncio.wait_for(self.release.wait(), timeout=1)
                return ProposerBatch(
                    schema_version=1,
                    candidates=(),
                    prompt_digest="a" * 64,
                    output_digest="b" * 64,
                    model="test-model",
                    provider="test-provider",
                )
            finally:
                self.active -= 1

    proposer = _ConcurrentEmptyProposer()
    harness.coordinator.proposer = proposer
    for index in range(4):
        harness.ledger.append_raw_event(
            "room-main",
            f"parallel-{index}",
            json.dumps({"message": f"parallel-{index}"}),
        )

    outcome = await harness.coordinator.run(
        run_id="night-parallel-proposer",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.run.stage == "complete"
    assert outcome.counts["proposer_attempted"] == 4
    assert outcome.counts["proposer_succeeded"] == 4
    assert outcome.counts["proposer_pending_after"] == 0
    assert proposer.peak == 4
    with sqlite3.connect(harness.ledger.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM chunk_proposer_outcomes "
            "WHERE outcome = 'zero_candidates'"
        ).fetchone() == (4,)


@pytest.mark.asyncio
async def test_proposer_pool_refills_without_waiting_for_slow_tail(
    tmp_path: Path,
) -> None:
    harness = _harness(
        tmp_path,
        policy=NightRunPolicy(
            proposer_max_chunks_per_run=4,
            proposer_concurrency=2,
        ),
    )

    class _TailAwareProposer:
        def __init__(self) -> None:
            self.calls = 0
            self.slow_release = asyncio.Event()
            self.third_started = asyncio.Event()

        async def propose(self, *args: Any, **kwargs: Any) -> ProposerBatch:
            self.calls += 1
            call_number = self.calls
            if call_number == 1:
                await self.slow_release.wait()
            if call_number == 3:
                self.third_started.set()
            return ProposerBatch(
                schema_version=1,
                candidates=(),
                prompt_digest="c" * 64,
                output_digest="d" * 64,
                model="test-model",
                provider="test-provider",
            )

    proposer = _TailAwareProposer()
    harness.coordinator.proposer = proposer
    for index in range(4):
        harness.ledger.append_raw_event(
            "room-main",
            f"refill-{index}",
            json.dumps({"message": f"refill-{index}"}),
        )

    run = asyncio.create_task(
        harness.coordinator.run(
            run_id="night-refill-proposer",
            cutoff=datetime.now(timezone.utc),
        )
    )
    await asyncio.wait_for(proposer.third_started.wait(), timeout=1)
    assert not run.done()
    proposer.slow_release.set()
    outcome = await run

    assert outcome.run.stage == "complete"
    assert outcome.counts["proposer_succeeded"] == 4
    assert proposer.calls == 4


@pytest.mark.asyncio
async def test_inflight_success_clears_concurrent_error_streak(
    tmp_path: Path,
) -> None:
    harness = _harness(
        tmp_path,
        policy=NightRunPolicy(
            proposer_max_chunks_per_run=5,
            proposer_concurrency=4,
        ),
    )

    class _RecoveringProposer:
        def __init__(self) -> None:
            self.calls = 0

        async def propose(self, *args: Any, **kwargs: Any) -> ProposerBatch:
            self.calls += 1
            if self.calls <= 3:
                raise ProposerContractError(
                    "provider.no_choices",
                    "test failure",
                )
            return ProposerBatch(
                schema_version=1,
                candidates=(),
                prompt_digest="e" * 64,
                output_digest="f" * 64,
                model="test-model",
                provider="test-provider",
            )

    proposer = _RecoveringProposer()
    harness.coordinator.proposer = proposer
    for index in range(5):
        harness.ledger.append_raw_event(
            "room-main",
            f"recovery-{index}",
            json.dumps({"message": f"recovery-{index}"}),
        )

    outcome = await harness.coordinator.run(
        run_id="night-concurrent-recovery",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.run.stage == "deferred"
    assert outcome.counts["proposer_attempted"] == 5
    assert outcome.counts["proposer_retryable"] == 3
    assert outcome.counts["proposer_succeeded"] == 2
    assert outcome.counts["proposer_circuit_breaker"] == 0


@pytest.mark.parametrize("value", (1, 4, 8))
def test_proposer_policy_accepts_bounded_concurrency(value: int) -> None:
    assert NightRunPolicy(proposer_concurrency=value).proposer_concurrency == value


@pytest.mark.parametrize("value", (0, 9, 16, True, 1.0))
def test_proposer_policy_rejects_invalid_concurrency(value: object) -> None:
    with pytest.raises(ValueError):
        NightRunPolicy(proposer_concurrency=value)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_three_consecutive_proposer_errors_open_circuit_without_faking_terminal(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, invalid_provider=True)
    for index in range(4):
        harness.ledger.append_raw_event(
            "room-main",
            f"poison-{index}",
            json.dumps({"message": f"poison-{index}"}),
        )

    outcome = await harness.coordinator.run(
        run_id="night-circuit-1",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.run.stage == "deferred"
    assert outcome.counts["proposer_attempted"] == 3
    assert outcome.counts["proposer_retryable"] == 3
    assert outcome.counts["proposer_circuit_breaker"] == 1
    assert outcome.counts["proposer_pending_after"] == 4
    assert outcome.counts["proposer_quarantined"] == 0
    assert len(harness.ledger.list_pending_proposer_chunks(limit=10)) == 4


@pytest.mark.asyncio
async def test_repeated_poison_is_visible_as_quarantined_but_remains_pending(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, invalid_provider=True)
    harness.ledger.append_raw_event(
        "room-main", "poison", '{"message":"repeat poison"}'
    )
    cutoff = datetime.now(timezone.utc)

    outcomes = []
    for attempt in range(1, 4):
        outcomes.append(
            await harness.coordinator.run(
                run_id=f"night-quarantine-{attempt}",
                cutoff=cutoff,
            )
        )

    assert [outcome.run.stage for outcome in outcomes] == [
        "deferred",
        "deferred",
        "deferred",
    ]
    assert outcomes[-1].counts["proposer_quarantined"] == 1
    pending = harness.ledger.list_pending_proposer_chunks(limit=10)
    assert len(pending) == 1
    assert pending[0].retry_count == 3


@pytest.mark.asyncio
async def test_proposer_wall_budget_times_out_as_retryable_deferred(
    tmp_path: Path,
) -> None:
    harness = _harness(
        tmp_path,
        policy=NightRunPolicy(proposer_wall_budget_seconds=1),
    )

    class _SlowProposer:
        async def propose(self, *args, **kwargs):
            await asyncio.sleep(10)

    harness.coordinator.proposer = _SlowProposer()
    harness.ledger.append_raw_event(
        "room-main", "slow", '{"message":"slow provider"}'
    )

    outcome = await harness.coordinator.run(
        run_id="night-wall-budget",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.run.stage == "deferred"
    assert outcome.counts["proposer_attempted"] == 1
    assert outcome.counts["proposer_retryable"] == 1
    assert outcome.counts["proposer_wall_budget_exhausted"] == 1
    assert outcome.counts["proposer_pending_after"] == 1
    with sqlite3.connect(harness.ledger.path) as connection:
        assert connection.execute(
            "SELECT outcome, error_code FROM chunk_proposer_outcomes"
        ).fetchall() == [("retryable_error", "provider.run_budget")]


@pytest.mark.asyncio
async def test_invalid_raw_json_stays_uncovered_and_fails_closed(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    harness.ledger.append_raw_event(
        "room-main", "message-1", '{"duplicate":1,"duplicate":2}'
    )

    with pytest.raises(NightRunCoordinatorError) as raised:
        await harness.coordinator.run(
            run_id="night-invalid-raw",
            cutoff=datetime.now(timezone.utc),
        )

    assert raised.value.code == "raw.invalid_json"
    assert harness.ledger.get_night_run("night-invalid-raw").stage == "error"
    assert len(harness.ledger.list_uncovered_raw_events(limit=10)) == 1
    assert harness.ledger.list_pending_proposer_chunks(limit=10) == ()


@pytest.mark.asyncio
async def test_report_only_engine_cannot_claim_success_after_mutation(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, unsafe_decay=True)
    harness.ledger.append_raw_event(
        "room-main", "message-1", '{"message":"unsafe metabolism"}'
    )

    with pytest.raises(NightRunCoordinatorError) as raised:
        await harness.coordinator.run(
            run_id="night-unsafe-m",
            cutoff=datetime.now(timezone.utc),
        )

    assert raised.value.code == "metabolism.decay_unsafe"
    run = harness.ledger.get_night_run("night-unsafe-m")
    assert run.stage == "error"
    assert [row.axis for row in harness.ledger.list_candidates("pending")] == [
        "M"
    ]


@pytest.mark.asyncio
async def test_completed_run_id_is_never_resumed_or_resnapshotted(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, empty_provider=True)
    cutoff = datetime.now(timezone.utc)
    first = await harness.coordinator.run(
        run_id="night-no-resume",
        cutoff=cutoff,
    )

    with pytest.raises(NightRunCoordinatorError) as raised:
        await harness.coordinator.run(
            run_id="night-no-resume",
            cutoff=datetime.now(timezone.utc),
        )

    assert raised.value.code == "run.reused"
    current = harness.ledger.get_night_run("night-no-resume")
    assert current.stage == "complete"
    assert current.sequence == first.run.sequence


def test_components_with_different_maintenance_roots_are_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vault"
    source.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    snapshots = SnapshotManager(source, tmp_path / "snapshots")
    ledger = LMC5Ledger(
        source / ".lmc5" / "ledger.db",
        maintenance_root=source,
    )
    provider = _Provider(empty=True)
    proposer = StrictOmbreProposer(provider)
    wrong_barrier = MaintenanceBarrier(other)
    curated = _FakeCurated(wrong_barrier)
    decay = _ReportEngine(wrong_barrier, kind="decay")
    consolidation = _ReportEngine(wrong_barrier, kind="consolidation")

    with pytest.raises(ValueError, match="share one maintenance barrier"):
        NightRunCoordinator(
            ledger=ledger,
            snapshots=snapshots,
            proposer=proposer,
            curated=curated,
            decay_engine=decay,
            consolidation_engine=consolidation,
        )


@pytest.mark.asyncio
async def test_real_x_write_retries_without_duplicate_bucket(
    tmp_path: Path,
    test_config: dict[str, Any],
    bucket_mgr,
) -> None:
    from consolidation_engine import ConsolidationEngine
    from decay_engine import DecayEngine

    seed_id = await bucket_mgr.create(
        content="夜班前已经存在的记忆",
        name="night-seed",
        bucket_type="dynamic",
    )
    seed_before = await bucket_mgr.get(seed_id)
    assert seed_before is not None

    root = Path(test_config["buckets_dir"])
    ledger = LMC5Ledger(
        root / ".lmc5" / "pipeline.sqlite3",
        maintenance_root=root,
    )
    snapshots = SnapshotManager(root, tmp_path / "night-snapshots")
    provider = _Provider()
    proposer = StrictOmbreProposer(
        provider,
        timeout_seconds=1,
        model="test-model",
        provider_name="test-provider",
    )
    embedding = _RetryEmbedding()
    curated = CuratedWriteCoordinator(bucket_mgr, embedding)
    decay = DecayEngine(test_config, bucket_mgr)
    consolidation = ConsolidationEngine(
        test_config,
        bucket_mgr,
        embedding,
    )
    coordinator = NightRunCoordinator(
        ledger=ledger,
        snapshots=snapshots,
        proposer=proposer,
        curated=curated,
        decay_engine=decay,
        consolidation_engine=consolidation,
        bucket_manager=bucket_mgr,
    )
    ledger.append_raw_event(
        "room-main",
        "night-real-write",
        '{"message":"朝灯今晚想看星星"}',
    )

    first = await coordinator.run(
        run_id="night-real-write-r1",
        cutoff=datetime.now(timezone.utc),
    )

    assert first.run.stage == "deferred"
    assert first.counts["dispatch_attempted"] == 1
    assert first.counts["dispatch_retryable"] == 1
    assert first.counts["dispatch_pending_after"] == 1
    assert first.counts["dispatch_circuit_breaker"] == 0
    assert first.counts["m_computed"] == 1
    assert len(ledger.list_candidates("pending")) == 1

    second = await coordinator.run(
        run_id="night-real-write-r2",
        cutoff=datetime.now(timezone.utc),
    )

    assert second.run.stage == "complete"
    assert second.counts["timeline_scanned"] == 2
    assert second.counts["timeline_assigned"] == 0
    assert second.counts["timeline_named"] == 0
    assert second.counts["timeline_updated"] == 0
    assert second.counts["timeline_new_lines"] == 0
    assert second.counts["timeline_orphans"] == 2
    assert embedding.calls == 2
    assert len(provider.prompts) == 1
    assert ledger.list_candidates("pending") == ()
    visible = await bucket_mgr.list_all(include_archive=False)
    curated_buckets = [
        bucket
        for bucket in visible
        if (bucket.get("metadata") or {}).get("curated_write_key")
    ]
    assert len(curated_buckets) == 1
    assert (
        curated_buckets[0]["metadata"]["lmc5_recall_state"]
        == "ready_vector"
    )
    all_curated_buckets = [
        bucket
        for bucket in await bucket_mgr.list_all(include_archive=True)
        if (bucket.get("metadata") or {}).get("curated_write_key")
    ]
    assert [bucket["id"] for bucket in all_curated_buckets] == [
        curated_buckets[0]["id"]
    ]
    assert curated_buckets[0]["metadata"]["thread"] == "other"
    seed_after = await bucket_mgr.get(seed_id)
    assert seed_after is not None
    assert seed_after["content"] == seed_before["content"]
    assert seed_after["metadata"] == seed_before["metadata"]


@pytest.mark.asyncio
async def test_night_timeline_assigns_after_snapshot_and_reports_counts(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path, empty_provider=True)
    marker = harness.source / "timeline-target.json"
    marker.write_text('{"thread":"other"}', encoding="utf-8")
    manager = _AxisBucketManager(
        harness.coordinator.maintenance_barrier,
        {
            "anchor": {
                "id": "anchor",
                "content": "基础设施起点",
                "metadata": {
                    "id": "anchor",
                    "type": "dynamic",
                    "thread": "基础设施演进",
                    "relations": [
                        {"type": "in_thread", "target": "target"},
                    ],
                },
            },
            "target": {
                "id": "target",
                "content": "基础设施后续",
                "metadata": {
                    "id": "target",
                    "type": "dynamic",
                    "thread": "other",
                    "relations": [],
                },
            },
        },
    )
    original_set_thread = manager.set_thread

    async def persist_thread(bucket_id: str, thread: str, **kwargs) -> bool:
        changed = await original_set_thread(bucket_id, thread, **kwargs)
        if changed and bucket_id == "target":
            marker.write_text(
                json.dumps({"thread": thread}, ensure_ascii=False),
                encoding="utf-8",
            )
        return changed

    manager.set_thread = persist_thread
    harness.coordinator.bucket_manager = manager

    outcome = await harness.coordinator.run(
        run_id="night-timeline-snapshot-order",
        cutoff=datetime.now(timezone.utc),
    )

    assert outcome.counts["timeline_scanned"] == 2
    assert outcome.counts["timeline_assigned"] == 1
    assert outcome.counts["timeline_named"] == 2
    assert outcome.counts["timeline_updated"] == 1
    assert outcome.counts["timeline_new_lines"] == 0
    assert outcome.counts["timeline_orphans"] == 0
    verified = harness.snapshots.verify_snapshot(
        "night-timeline-snapshot-order",
        expected_manifest_sha256=outcome.snapshot_manifest_sha256,
    )
    assert json.loads(
        (verified.snapshot_path / "files" / marker.name).read_text("utf-8")
    ) == {"thread": "other"}
    assert json.loads(marker.read_text("utf-8")) == {
        "thread": "基础设施演进",
    }


@pytest.mark.asyncio
async def test_dispatch_retryable_candidate_does_not_block_later_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path)
    records = (
        SimpleNamespace(candidate_id=1, axis="X"),
        SimpleNamespace(candidate_id=2, axis="E"),
    )
    statuses = {1: "pending", 2: "pending"}
    seen: list[int] = []

    def list_candidates(
        status: str,
        *,
        limit: int,
        after: int | None = None,
    ):
        return tuple(
            record
            for record in records
            if statuses[record.candidate_id] == status
            and (after is None or record.candidate_id > after)
        )[:limit]

    async def dispatch(record, _counts) -> None:
        seen.append(record.candidate_id)
        if record.candidate_id == 1:
            raise NightRunCoordinatorError("x.write_retryable")
        statuses[record.candidate_id] = "ready"

    monkeypatch.setattr(harness.ledger, "list_candidates", list_candidates)
    monkeypatch.setattr(harness.coordinator, "_dispatch_candidate", dispatch)
    counts: dict[str, int] = {}

    await harness.coordinator._dispatch_pending(counts)

    assert seen == [1, 2]
    assert counts["dispatch_attempted"] == 2
    assert counts["dispatch_retryable"] == 1
    assert counts["dispatch_pending_after"] == 1
    assert counts["dispatch_circuit_breaker"] == 0


@pytest.mark.asyncio
async def test_dispatch_three_consecutive_retryables_open_circuit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path)
    records = tuple(
        SimpleNamespace(candidate_id=index, axis="X")
        for index in range(1, 5)
    )
    seen: list[int] = []

    def list_candidates(
        status: str,
        *,
        limit: int,
        after: int | None = None,
    ):
        if status != "pending":
            return ()
        return tuple(
            record
            for record in records
            if after is None or record.candidate_id > after
        )[:limit]

    async def dispatch(record, _counts) -> None:
        seen.append(record.candidate_id)
        raise NightRunCoordinatorError("x.write_retryable")

    monkeypatch.setattr(harness.ledger, "list_candidates", list_candidates)
    monkeypatch.setattr(harness.coordinator, "_dispatch_candidate", dispatch)
    counts: dict[str, int] = {}

    await harness.coordinator._dispatch_pending(counts)

    assert seen == [1, 2, 3]
    assert counts["dispatch_attempted"] == 3
    assert counts["dispatch_retryable"] == 3
    assert counts["dispatch_pending_after"] == 4
    assert counts["dispatch_circuit_breaker"] == 1
