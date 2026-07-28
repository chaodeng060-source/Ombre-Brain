from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

import lmc5_ledger as ledger_module
from lmc5_ledger import (
    EventIdentity,
    LMC5Ledger,
    LedgerConflictError,
    LedgerCorruptionError,
    LedgerSecurityError,
    LedgerStateError,
    LedgerValidationError,
)


def _ledger(tmp_path) -> LMC5Ledger:
    return LMC5Ledger(tmp_path / "private-ledger" / "lmc5.db")


def _seed_chunk(ledger: LMC5Ledger) -> None:
    ledger.append_raw_event("session-a", "event-1", "first")
    ledger.record_event_chunk(
        "chunk-1", "chunk body", [EventIdentity("session-a", "event-1")]
    )


def test_database_is_wal_full_sync_and_private(tmp_path):
    ledger = _ledger(tmp_path)

    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2

    if os.name != "nt":
        assert stat.S_IMODE(ledger.path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600


def test_symlink_database_is_rejected(tmp_path):
    target = tmp_path / "target.db"
    target.write_bytes(b"")
    link = tmp_path / "ledger.db"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(LedgerSecurityError):
        LMC5Ledger(link)


def test_raw_event_is_append_only_idempotent_and_conflict_safe(tmp_path):
    ledger = _ledger(tmp_path)
    first = ledger.append_raw_event("s", "e", "exact raw body")
    replay = ledger.append_raw_event("s", "e", "exact raw body")

    assert first.created is True
    assert replay.created is False
    assert replay.row_id == first.row_id
    with pytest.raises(LedgerConflictError):
        ledger.append_raw_event("s", "e", "changed body")

    with sqlite3.connect(ledger.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE raw_events SET payload = X'00'")


def test_atomic_raw_batch_rolls_back_on_conflict(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.append_raw_event("s", "existing", "one")

    with pytest.raises(LedgerConflictError):
        ledger.append_raw_events(
            [
                ("s", "new", "new event"),
                ("s", "existing", "different"),
            ]
        )

    report = ledger.coverage_report()
    assert report.total_raw_events == 1
    assert report.uncovered_event_ids == (EventIdentity("s", "existing"),)


def test_chunks_bind_exact_sources_and_coverage_does_not_use_max_watermark(tmp_path):
    ledger = _ledger(tmp_path)
    for event_id in ("event-1", "event-2", "event-3"):
        ledger.append_raw_event("s", event_id, f"body-{event_id}")

    created = ledger.record_event_chunk(
        "chunk-13",
        "derived from one and three",
        [EventIdentity("s", "event-1"), EventIdentity("s", "event-3")],
    )
    replay = ledger.record_event_chunk(
        "chunk-13",
        "derived from one and three",
        [EventIdentity("s", "event-3"), EventIdentity("s", "event-1")],
    )
    assert created.created is True
    assert replay.created is False

    report = ledger.coverage_report()
    assert report.total_raw_events == 3
    assert report.covered_event_ids == (
        EventIdentity("s", "event-1"),
        EventIdentity("s", "event-3"),
    )
    assert report.holes == (EventIdentity("s", "event-2"),)
    assert report.is_fully_covered is False

    with pytest.raises(LedgerConflictError):
        ledger.record_event_chunk(
            "chunk-13",
            "derived from one and three",
            [EventIdentity("s", "event-1")],
        )
    with pytest.raises(LedgerStateError):
        ledger.record_event_chunk(
            "chunk-missing",
            "no source",
            [EventIdentity("s", "missing")],
        )
    with pytest.raises(LedgerValidationError):
        ledger.record_event_chunk(
            "chunk-empty",
            "   ",
            [EventIdentity("s", "event-2")],
        )


def test_uncovered_feed_is_bounded_stable_and_returns_exact_payload(
    tmp_path, monkeypatch
):
    timestamps = iter(
        [
            "2026-07-28T00:00:00.000000+00:00",
            "2026-07-28T00:01:00.000000+00:00",
            "2026-07-28T00:02:00.000000+00:00",
            "2026-07-28T00:03:00.000000+00:00",
        ]
    )
    monkeypatch.setattr(ledger_module, "_utc_now", lambda: next(timestamps))
    ledger = _ledger(tmp_path)
    first = ledger.append_raw_event("s", "event-1", b"raw-one")
    ledger.append_raw_event("s", "event-2", b"raw-two")
    ledger.append_raw_event("s", "event-3", b"raw-three")
    ledger.record_event_chunk(
        "chunk-2", "covered", [EventIdentity("s", "event-2")]
    )

    page_one = ledger.list_uncovered_raw_events(limit=1)
    assert len(page_one) == 1
    assert page_one[0].identity == EventIdentity("s", "event-1")
    assert page_one[0].payload == b"raw-one"
    assert page_one[0].row_id == first.row_id

    page_two = ledger.list_uncovered_raw_events(
        limit=2, after=page_one[0].row_id
    )
    assert [record.identity.source_event_id for record in page_two] == ["event-3"]
    before = ledger.list_uncovered_raw_events(
        limit=10, created_before="2026-07-28T00:01:30Z"
    )
    assert [record.identity.source_event_id for record in before] == ["event-1"]

    for bad_limit in (0, 1001, True):
        with pytest.raises(LedgerValidationError):
            ledger.list_uncovered_raw_events(limit=bad_limit)
    for bad_after in (-1, True):
        with pytest.raises(LedgerValidationError):
            ledger.list_uncovered_raw_events(after=bad_after)
    for bad_time in ("yesterday", "2026-07-28T00:00:00"):
        with pytest.raises(LedgerValidationError):
            ledger.list_uncovered_raw_events(created_before=bad_time)


def test_candidate_idempotency_statuses_and_machine_code_only_errors(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)

    first = ledger.record_candidate("candidate-key", "Z", '{"slot":"home"}', ["chunk-1"])
    replay = ledger.record_candidate(
        "candidate-key", "Z", '{"slot":"home"}', ["chunk-1"]
    )
    assert first.created is True
    assert replay.created is False

    deferred = ledger.transition_candidate(
        "candidate-key",
        "deferred",
        expected_status="pending",
        error_code="quota.exhausted",
    )
    assert deferred.status == "deferred"
    assert deferred.error_code == "quota.exhausted"

    with pytest.raises(LedgerValidationError):
        ledger.transition_candidate(
            "candidate-key",
            "error",
            error_code="model returned raw private conversation text",
        )
    with sqlite3.connect(ledger.path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE candidates SET error_code = ? WHERE idempotency_key = ?",
                ("raw private conversation text", "candidate-key"),
            )
    with pytest.raises(LedgerConflictError):
        ledger.record_candidate("candidate-key", "Z", "changed", ["chunk-1"])
    with pytest.raises(LedgerValidationError):
        ledger.record_candidate("candidate-empty", "Z", b"", ["chunk-1"])

    report = ledger.coverage_report()
    assert report.candidate_status_counts["deferred"] == 1
    assert report.deferred_candidate_keys == ("candidate-key",)
    assert report.pending_candidate_keys == ()
    assert report.error_candidate_keys == ()


def test_candidate_feed_is_bounded_status_scoped_and_payload_safe(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    first = ledger.record_candidate("candidate-1", "X", b"first", ["chunk-1"])
    ledger.record_candidate("candidate-2", "Y", b"second", ["chunk-1"])
    ledger.record_candidate(
        "candidate-review", "Z", b"review", ["chunk-1"], status="review"
    )

    page_one = ledger.list_candidates("pending", limit=1)
    assert [record.idempotency_key for record in page_one] == ["candidate-1"]
    assert page_one[0].payload == b"first"
    assert page_one[0].source_chunk_ids == ("chunk-1",)
    page_two = ledger.list_candidates(
        "pending", limit=10, after=first.candidate_id
    )
    assert [record.idempotency_key for record in page_two] == ["candidate-2"]
    assert [record.idempotency_key for record in ledger.list_candidates("review")] == [
        "candidate-review"
    ]

    with pytest.raises(LedgerValidationError):
        ledger.list_candidates("unknown")
    with pytest.raises(LedgerValidationError):
        ledger.list_candidates("pending", limit=0)
    with pytest.raises(LedgerValidationError):
        ledger.list_candidates("pending", after=-1)


def test_concurrent_candidate_replay_creates_exactly_once(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)

    def write_once(_index: int):
        return ledger.record_candidate("shared-key", "Y", "same", ["chunk-1"])

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_once, range(24)))

    assert sum(result.created for result in results) == 1
    assert len({result.candidate_id for result in results}) == 1
    assert ledger.coverage_report().candidate_status_counts["pending"] == 1


def test_write_receipt_replay_is_exact_and_mismatch_conflicts(tmp_path):
    ledger = _ledger(tmp_path)
    request_hash = hashlib.sha256(b"request").hexdigest()
    result_hash = hashlib.sha256(b"result").hexdigest()

    first = ledger.record_write_receipt(
        "write-1", request_hash, "bucket:abc", result_hash=result_hash
    )
    replay = ledger.record_write_receipt(
        "write-1", request_hash, "bucket:abc", result_hash=result_hash
    )
    assert first.created is True
    assert replay.created is False

    with pytest.raises(LedgerConflictError):
        ledger.record_write_receipt(
            "write-1",
            hashlib.sha256(b"different").hexdigest(),
            "bucket:abc",
            result_hash=result_hash,
        )


def test_night_run_has_snapshot_compare_and_set_and_sanitized_errors(tmp_path):
    ledger = _ledger(tmp_path)
    started = ledger.start_night_run("night-1", "snapshot:sha256:abc", counts={"raw": 3})
    assert started.stage == "started"
    assert started.sequence == 0

    validated = ledger.record_night_stage(
        "night-1",
        "validate",
        expected_stage="started",
        counts={"covered": 2, "raw": 3},
        errors=["coverage.hole"],
    )
    assert validated.sequence == 1
    assert validated.errors == ("coverage.hole",)
    replay = ledger.record_night_stage(
        "night-1",
        "validate",
        counts={"raw": 3, "covered": 2},
        errors=["coverage.hole"],
    )
    assert replay.sequence == 1
    assert replay.created is False

    with pytest.raises(LedgerConflictError):
        ledger.start_night_run("night-1", "other-snapshot")
    with pytest.raises(LedgerStateError):
        ledger.record_night_stage(
            "night-1", "complete", expected_stage="snapshot", counts={}
        )
    with pytest.raises(LedgerValidationError):
        ledger.record_night_stage(
            "night-1", "error", errors=["raw text must never enter this log"]
        )

    ledger.record_night_stage(
        "night-1",
        "complete",
        expected_stage="validate",
        counts={"covered": 3, "raw": 3},
    )
    with pytest.raises(LedgerStateError):
        ledger.record_night_stage("night-1", "post_complete")


def test_composable_transaction_rolls_back_all_pipeline_rows(tmp_path):
    ledger = _ledger(tmp_path)

    with pytest.raises(LedgerStateError):
        with ledger.transaction() as transaction:
            transaction.append_raw_event("s", "e", "raw")
            transaction.record_event_chunk(
                "chunk",
                "chunk",
                [EventIdentity("s", "e")],
            )
            transaction.record_candidate(
                "candidate", "X", "candidate", ["missing-chunk"]
            )

    assert ledger.coverage_report().total_raw_events == 0
    assert ledger.coverage_report().total_chunks == 0


def test_corrupt_status_json_and_non_database_files_fail_closed(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_night_run("night", "snapshot")
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE night_runs SET errors_json = ? WHERE run_id = ?",
            ('["private free form sentence"]', "night"),
        )
        connection.commit()

    with pytest.raises(LedgerCorruptionError):
        ledger.get_night_run("night")

    corrupt_path = tmp_path / "corrupt" / "ledger.db"
    corrupt_path.parent.mkdir()
    corrupt_path.write_bytes(b"this is not sqlite")
    with pytest.raises(LedgerCorruptionError):
        LMC5Ledger(corrupt_path)


def test_deep_integrity_recomputes_immutable_payload_digests(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    ledger.record_candidate("candidate", "M", "body", ["chunk-1"])

    result = ledger.verify_integrity()
    assert result["raw_events"] == 1
    assert result["event_chunks"] == 1
    assert result["candidates"] == 1
