"""Night-run dispatch caps (2026-09-14 night-run lock incident).

One night dispatched a four-week backlog (2363 candidates) inside the
exclusive lease.  Dispatch now stops between candidates after
``dispatch_max_candidates_per_run`` or ``dispatch_wall_budget_seconds``; the
rest stays pending for the next run, which picks it up without repeating any
already dispatched candidate.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from night_run_coordinator import NightRunCoordinatorError, NightRunPolicy
from night_run_runtime import (
    DEFAULT_DISPATCH_MAX_CANDIDATES_PER_RUN,
    DEFAULT_DISPATCH_WALL_BUDGET_SECONDS,
    NightRunRuntimeError,
    _dispatch_max_candidates_per_run,
    _dispatch_wall_budget_seconds,
)
from tests.test_night_run_coordinator import _harness


def _append_events(harness, count: int) -> None:
    for index in range(count):
        harness.ledger.append_raw_event(
            "room-main",
            f"budget-event-{index}",
            json.dumps({"message": f"朝灯今晚第{index}件事"}, ensure_ascii=False),
        )


def test_policy_defaults_and_validation():
    policy = NightRunPolicy()
    assert policy.dispatch_max_candidates_per_run == 500
    assert policy.dispatch_wall_budget_seconds == 1800
    for kwargs in (
        {"dispatch_max_candidates_per_run": 0},
        {"dispatch_max_candidates_per_run": True},
        {"dispatch_max_candidates_per_run": 100_001},
        {"dispatch_wall_budget_seconds": 0},
        {"dispatch_wall_budget_seconds": 3600},
        {"dispatch_wall_budget_seconds": 1.5},
    ):
        with pytest.raises(ValueError):
            NightRunPolicy(**kwargs)


def test_runtime_config_readers_default_and_fail_closed():
    assert DEFAULT_DISPATCH_MAX_CANDIDATES_PER_RUN == 500
    assert DEFAULT_DISPATCH_WALL_BUDGET_SECONDS == 1800
    assert _dispatch_max_candidates_per_run({}) == 500
    assert _dispatch_wall_budget_seconds({}) == 1800
    assert _dispatch_max_candidates_per_run({"lmc5_night": None}) == 500
    assert (
        _dispatch_max_candidates_per_run(
            {"lmc5_night": {"dispatch_max_candidates_per_run": 40}}
        )
        == 40
    )
    assert (
        _dispatch_wall_budget_seconds(
            {"lmc5_night": {"dispatch_wall_budget_seconds": 900}}
        )
        == 900
    )
    for value in (0, -1, True, "500", 100_001):
        with pytest.raises(NightRunRuntimeError) as raised:
            _dispatch_max_candidates_per_run(
                {"lmc5_night": {"dispatch_max_candidates_per_run": value}}
            )
        assert raised.value.code == "dispatch.candidate_cap_invalid"
    for value in (0, 3600, 1.0, False):
        with pytest.raises(NightRunRuntimeError) as raised:
            _dispatch_wall_budget_seconds(
                {"lmc5_night": {"dispatch_wall_budget_seconds": value}}
            )
        assert raised.value.code == "dispatch.wall_budget_invalid"


@pytest.mark.asyncio
async def test_candidate_cap_defers_rest_and_next_runs_drain_without_repeats(
    tmp_path: Path,
) -> None:
    harness = _harness(
        tmp_path,
        policy=NightRunPolicy(dispatch_max_candidates_per_run=2),
    )
    _append_events(harness, 5)
    cutoff = datetime.now(timezone.utc)

    first = await harness.coordinator.run(run_id="night-cap-1", cutoff=cutoff)
    assert first.run.stage == "deferred"
    assert first.counts["proposer_pending_after"] == 0
    assert first.counts["dispatch_attempted"] == 2
    assert first.counts["x_ready"] == 2
    assert first.counts["dispatch_cap_reached"] == 1
    assert first.counts["dispatch_wall_budget_exhausted"] == 0
    assert first.counts["dispatch_deferred_budget"] == 3
    assert first.counts["dispatch_pending_after"] == 3
    assert first.counts["m_computed"] == 5  # later stages still ran
    assert harness.ledger.get_night_run("night-cap-1").stage == "deferred"
    assert len(harness.curated.calls) == 2

    second = await harness.coordinator.run(run_id="night-cap-2", cutoff=cutoff)
    assert second.run.stage == "deferred"
    assert second.counts["dispatch_attempted"] == 2
    assert second.counts["dispatch_deferred_budget"] == 1
    assert second.counts["dispatch_pending_after"] == 1

    third = await harness.coordinator.run(run_id="night-cap-3", cutoff=cutoff)
    assert third.run.stage == "complete"
    assert third.counts["dispatch_attempted"] == 1
    assert third.counts["dispatch_cap_reached"] == 0
    assert third.counts["dispatch_deferred_budget"] == 0
    assert third.counts["dispatch_pending_after"] == 0

    keys = [call["idempotency_key"] for call in harness.curated.calls]
    assert len(keys) == 5
    assert len(set(keys)) == 5  # every candidate dispatched exactly once
    assert harness.ledger.list_candidates("pending") == ()
    ready = harness.ledger.list_candidates("ready", limit=100)
    assert sorted(row.axis for row in ready) == ["M"] * 5 + ["X"] * 5
    assert len(harness.provider.prompts) == 5  # proposer never re-ran


@pytest.mark.asyncio
async def test_exact_cap_is_not_a_deferral(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        policy=NightRunPolicy(dispatch_max_candidates_per_run=3),
    )
    _append_events(harness, 3)
    outcome = await harness.coordinator.run(
        run_id="night-cap-exact",
        cutoff=datetime.now(timezone.utc),
    )
    assert outcome.run.stage == "complete"
    assert outcome.counts["dispatch_attempted"] == 3
    assert outcome.counts["dispatch_cap_reached"] == 0
    assert outcome.counts["dispatch_deferred_budget"] == 0


@pytest.mark.asyncio
async def test_wall_budget_stops_between_candidates(tmp_path, monkeypatch):
    harness = _harness(
        tmp_path,
        policy=NightRunPolicy(dispatch_wall_budget_seconds=1),
    )
    records = tuple(
        SimpleNamespace(candidate_id=index, axis=axis)
        for index, axis in enumerate(["X", "M", "X", "X", "M", "X"], start=1)
    )
    statuses = {record.candidate_id: "pending" for record in records}
    seen: list[int] = []

    def list_candidates(status: str, *, limit: int, after: int | None = None):
        return tuple(
            record
            for record in records
            if statuses[record.candidate_id] == status
            and (after is None or record.candidate_id > after)
        )[:limit]

    async def slow_dispatch(record, _counts) -> None:
        seen.append(record.candidate_id)
        await asyncio.sleep(0.6)
        statuses[record.candidate_id] = "ready"

    monkeypatch.setattr(harness.ledger, "list_candidates", list_candidates)
    monkeypatch.setattr(harness.coordinator, "_dispatch_candidate", slow_dispatch)
    counts: dict[str, int] = {}

    await harness.coordinator._dispatch_pending(counts)

    assert seen == [1, 3]  # the in-flight candidate finished; nothing cut off
    assert counts["dispatch_attempted"] == 2
    assert counts["dispatch_wall_budget_exhausted"] == 1
    assert counts["dispatch_cap_reached"] == 0
    assert counts["dispatch_deferred_budget"] == 2
    assert counts["dispatch_pending_after"] == 2
    assert {cid for cid, status in statuses.items() if status == "pending"} == {2, 4, 5, 6}


@pytest.mark.asyncio
async def test_cap_counts_retryables_and_skips_metabolism_rows(tmp_path, monkeypatch):
    harness = _harness(
        tmp_path,
        policy=NightRunPolicy(dispatch_max_candidates_per_run=2),
    )
    records = tuple(
        SimpleNamespace(candidate_id=index, axis=axis)
        for index, axis in enumerate(["M", "X", "E", "M", "X", "Y"], start=1)
    )
    statuses = {record.candidate_id: "pending" for record in records}
    seen: list[int] = []

    def list_candidates(status: str, *, limit: int, after: int | None = None):
        return tuple(
            record
            for record in records
            if statuses[record.candidate_id] == status
            and (after is None or record.candidate_id > after)
        )[:limit]

    async def dispatch(record, _counts) -> None:
        seen.append(record.candidate_id)
        if record.candidate_id == 2:
            raise NightRunCoordinatorError("x.write_retryable")
        statuses[record.candidate_id] = "ready"

    monkeypatch.setattr(harness.ledger, "list_candidates", list_candidates)
    monkeypatch.setattr(harness.coordinator, "_dispatch_candidate", dispatch)
    counts: dict[str, int] = {}

    await harness.coordinator._dispatch_pending(counts)

    assert seen == [2, 3]
    assert counts["dispatch_attempted"] == 2
    assert counts["dispatch_retryable"] == 1
    assert counts["dispatch_cap_reached"] == 1
    # pending non-M after: retryable 2 + unvisited 5 and 6
    assert counts["dispatch_pending_after"] == 3
    assert counts["dispatch_deferred_budget"] == 2


def test_budget_counts_must_balance_in_validation(tmp_path):
    harness = _harness(tmp_path)
    counts = {
        "proposer_watermark": 0,
        "proposer_pending_before": 0,
        "proposer_attempted": 0,
        "proposer_succeeded": 0,
        "proposer_retryable": 0,
        "proposer_pending_after": 0,
        "proposer_quarantined": 0,
        "proposer_circuit_breaker": 0,
        "proposer_wall_budget_exhausted": 0,
        "dispatch_attempted": 1,
        "dispatch_retryable": 0,
        "dispatch_pending_after": 2,
        "dispatch_circuit_breaker": 0,
        "dispatch_deferred_budget": 2,
        "dispatch_cap_reached": 0,
        "dispatch_wall_budget_exhausted": 0,
    }
    snapshot = SimpleNamespace(manifest_sha256="0" * 64)
    harness.coordinator.snapshots = SimpleNamespace(
        verify_snapshot=lambda *_args, **_kwargs: snapshot,
        maintenance_barrier=harness.coordinator.maintenance_barrier,
    )

    async def validate(values):
        await harness.coordinator._validate(
            run_id="night-validate",
            cutoff_iso=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            snapshot=snapshot,
            counts=values,
        )

    # deferred candidates without a recorded budget stop are rejected
    with pytest.raises(NightRunCoordinatorError) as raised:
        asyncio.run(validate(dict(counts)))
    assert raised.value.code == "validation.dispatch_counts"
    # cap and wall flags cannot both be set
    with pytest.raises(NightRunCoordinatorError) as raised:
        asyncio.run(
            validate({**counts, "dispatch_cap_reached": 1, "dispatch_wall_budget_exhausted": 1})
        )
    assert raised.value.code == "validation.dispatch_counts"
