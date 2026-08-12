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


def test_pending_proposer_chunks_require_a_successful_terminal_outcome(tmp_path):
    ledger = _ledger(tmp_path)
    for index in range(1, 4):
        event_id = f"event-{index}"
        chunk_id = f"chunk-{index}"
        ledger.append_raw_event("s", event_id, f"raw-{index}")
        ledger.record_event_chunk(
            chunk_id,
            f"chunk-body-{index}",
            [EventIdentity("s", event_id)],
        )

    page = ledger.list_pending_proposer_chunks(limit=2)
    assert [chunk.chunk_id for chunk in page] == ["chunk-1", "chunk-2"]
    assert page[0].content == b"chunk-body-1"
    assert page[0].source_event_ids == (EventIdentity("s", "event-1"),)
    assert [
        chunk.chunk_id
        for chunk in ledger.list_pending_proposer_chunks(
            after=page[-1].row_id
        )
    ] == ["chunk-3"]

    ledger.record_chunk_proposer_outcome(
        "attempt-error",
        "chunk-1",
        "retryable_error",
        error_code="model.timeout",
    )
    ledger.record_chunk_proposer_outcome(
        "attempt-zero",
        "chunk-2",
        "zero_candidates",
    )
    with ledger.transaction() as transaction:
        transaction.record_candidate(
            "candidate-3", "Y", "candidate", ["chunk-3"]
        )
        transaction.record_chunk_proposer_outcome(
            "attempt-candidate",
            "chunk-3",
            "candidates_persisted",
            candidate_keys=["candidate-3"],
        )

    assert [
        chunk.chunk_id for chunk in ledger.list_pending_proposer_chunks()
    ] == ["chunk-1"]
    pending = ledger.list_pending_proposer_chunks()[0]
    assert pending.retry_count == 1
    assert pending.latest_error_code == "model.timeout"
    ledger.record_chunk_proposer_outcome(
        "attempt-error-newer",
        "chunk-1",
        "retryable_error",
        error_code="schema_relation",
    )
    pending = ledger.list_pending_proposer_chunks()[0]
    assert pending.retry_count == 2
    assert pending.latest_error_code == "schema_relation"
    ledger.record_chunk_proposer_outcome(
        "attempt-retry-success",
        "chunk-1",
        "zero_candidates",
    )
    assert ledger.list_pending_proposer_chunks() == ()

    for bad_limit in (0, 1001, True):
        with pytest.raises(LedgerValidationError):
            ledger.list_pending_proposer_chunks(limit=bad_limit)
    for bad_after in (-1, True):
        with pytest.raises(LedgerValidationError):
            ledger.list_pending_proposer_chunks(after=bad_after)


def test_pending_proposer_watermark_priority_and_backlog_stats(tmp_path):
    ledger = _ledger(tmp_path)
    rows = {}
    for index in range(1, 5):
        event_id = f"event-{index}"
        chunk_id = f"chunk-{index}"
        ledger.append_raw_event("s", event_id, f"raw-{index}")
        ledger.record_event_chunk(
            chunk_id,
            f"chunk-body-{index}",
            [EventIdentity("s", event_id)],
        )
        rows[chunk_id] = ledger.list_pending_proposer_chunks(
            limit=10
        )[-1].row_id

    for attempt in range(3):
        ledger.record_chunk_proposer_outcome(
            f"chunk-1-retry-{attempt}",
            "chunk-1",
            "retryable_error",
            error_code="provider.timeout",
        )
    ledger.record_chunk_proposer_outcome(
        "chunk-2-retry-0",
        "chunk-2",
        "retryable_error",
        error_code="provider.timeout",
    )

    assert ledger.proposer_watermark() == rows["chunk-4"]
    assert [
        row.chunk_id
        for row in ledger.list_pending_proposer_chunks(
            limit=10,
            through=rows["chunk-3"],
            prioritize_retries=True,
        )
    ] == ["chunk-3", "chunk-2", "chunk-1"]
    stats = ledger.proposer_backlog_stats(through=rows["chunk-4"])
    assert (stats.pending, stats.unattempted, stats.quarantined) == (4, 2, 1)

    for bad_through in (-1, True):
        with pytest.raises(LedgerValidationError):
            ledger.list_pending_proposer_chunks(through=bad_through)
        with pytest.raises(LedgerValidationError):
            ledger.proposer_backlog_stats(through=bad_through)
    with pytest.raises(LedgerValidationError):
        ledger.list_pending_proposer_chunks(prioritize_retries=1)
    with pytest.raises(LedgerValidationError):
        ledger.list_pending_proposer_chunks(
            after=0,
            prioritize_retries=True,
        )


def test_proposer_outcome_replay_is_exact_and_success_is_terminal(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    ledger.append_raw_event("session-a", "event-2", "second")
    ledger.record_event_chunk(
        "chunk-2", "second chunk", [EventIdentity("session-a", "event-2")]
    )

    first = ledger.record_chunk_proposer_outcome(
        "attempt-1",
        "chunk-1",
        "retryable_error",
        error_code="provider.busy",
    )
    replay = ledger.record_chunk_proposer_outcome(
        "attempt-1",
        "chunk-1",
        "retryable_error",
        error_code="provider.busy",
    )
    assert first.created is True
    assert replay.created is False
    assert replay.outcome_id == first.outcome_id

    with pytest.raises(LedgerConflictError):
        ledger.record_chunk_proposer_outcome(
            "attempt-1",
            "chunk-1",
            "retryable_error",
            error_code="provider.timeout",
        )
    with pytest.raises(LedgerConflictError):
        ledger.record_chunk_proposer_outcome(
            "attempt-1",
            "chunk-2",
            "retryable_error",
            error_code="provider.busy",
        )
    with pytest.raises(LedgerConflictError):
        ledger.record_chunk_proposer_outcome(
            "attempt-1",
            "chunk-1",
            "zero_candidates",
        )

    success = ledger.record_chunk_proposer_outcome(
        "attempt-2",
        "chunk-1",
        "zero_candidates",
    )
    assert success.outcome == "zero_candidates"
    with pytest.raises(LedgerStateError):
        ledger.record_chunk_proposer_outcome(
            "attempt-after-success",
            "chunk-1",
            "retryable_error",
            error_code="provider.busy",
        )
    assert ledger.record_chunk_proposer_outcome(
        "attempt-2",
        "chunk-1",
        "zero_candidates",
    ).created is False


def test_candidates_persisted_requires_atomic_linked_candidates(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    ledger.append_raw_event("session-a", "event-2", "second")
    ledger.record_event_chunk(
        "chunk-2", "second chunk", [EventIdentity("session-a", "event-2")]
    )

    ledger.record_candidate(
        "candidate-2", "Z", "second candidate", ["chunk-2"]
    )
    with pytest.raises(LedgerStateError):
        ledger.record_chunk_proposer_outcome(
            "wrong-source-attempt",
            "chunk-1",
            "candidates_persisted",
            candidate_keys=["candidate-2"],
        )

    with ledger.transaction() as transaction:
        transaction.record_candidate(
            "candidate-1", "Z", "first candidate", ["chunk-1"]
        )
        outcome = transaction.record_chunk_proposer_outcome(
            "attempt-candidates",
            "chunk-1",
            "candidates_persisted",
            candidate_keys=["candidate-1"],
        )
    assert outcome.candidate_keys == ("candidate-1",)

    with pytest.raises(LedgerConflictError):
        ledger.record_chunk_proposer_outcome(
            "attempt-candidates",
            "chunk-1",
            "candidates_persisted",
            candidate_keys=["candidate-2"],
        )

    with pytest.raises(LedgerStateError):
        with ledger.transaction() as transaction:
            transaction.record_candidate(
                "rolled-back-candidate",
                "X",
                "must roll back",
                ["chunk-1"],
            )
            transaction.record_chunk_proposer_outcome(
                "rolled-back-attempt",
                "chunk-2",
                "candidates_persisted",
                candidate_keys=["rolled-back-candidate"],
            )
    assert [
        record.idempotency_key for record in ledger.list_candidates("pending")
    ] == ["candidate-2", "candidate-1"]


def test_multi_chunk_outcomes_share_one_atomic_transaction(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    ledger.append_raw_event("session-a", "event-2", "second")
    ledger.record_event_chunk(
        "chunk-2", "second chunk", [EventIdentity("session-a", "event-2")]
    )

    with pytest.raises(LedgerStateError):
        with ledger.transaction() as transaction:
            transaction.record_chunk_proposer_outcome(
                "attempt-chunk-1",
                "chunk-1",
                "zero_candidates",
            )
            transaction.record_chunk_proposer_outcome(
                "attempt-missing",
                "missing-chunk",
                "zero_candidates",
            )
    assert [
        chunk.chunk_id for chunk in ledger.list_pending_proposer_chunks()
    ] == ["chunk-1", "chunk-2"]

    with ledger.transaction() as transaction:
        first = transaction.record_chunk_proposer_outcome(
            "attempt-chunk-1",
            "chunk-1",
            "zero_candidates",
        )
        second = transaction.record_chunk_proposer_outcome(
            "attempt-chunk-2",
            "chunk-2",
            "zero_candidates",
        )
    assert first.created is True
    assert second.created is True
    assert ledger.list_pending_proposer_chunks() == ()


def test_concurrent_proposer_replay_creates_exactly_once(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)

    def write_once(_index: int):
        return ledger.record_chunk_proposer_outcome(
            "shared-attempt",
            "chunk-1",
            "zero_candidates",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_once, range(24)))

    assert sum(result.created for result in results) == 1
    assert len({result.outcome_id for result in results}) == 1
    assert ledger.list_pending_proposer_chunks() == ()


def test_different_keys_concurrently_compete_for_one_terminal_outcome(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)

    def write_once(index: int):
        try:
            return ledger.record_chunk_proposer_outcome(
                f"attempt-{index}",
                "chunk-1",
                "zero_candidates",
            )
        except LedgerStateError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_once, range(24)))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0].created is True
    assert ledger.verify_integrity()["chunk_proposer_outcomes"] == 1


@pytest.mark.parametrize(
    ("outcome", "candidate_keys", "error_code"),
    [
        ("zero_candidates", ["candidate"], None),
        ("zero_candidates", [], "unexpected.error"),
        ("candidates_persisted", [], None),
        ("candidates_persisted", ["candidate"], "unexpected.error"),
        ("retryable_error", ["candidate"], "provider.busy"),
        ("retryable_error", [], None),
        ("retryable_error", "candidate", "provider.busy"),
    ],
)
def test_proposer_outcome_shape_is_fail_closed(
    tmp_path, outcome, candidate_keys, error_code
):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    ledger.record_candidate("candidate", "X", "body", ["chunk-1"])

    with pytest.raises(LedgerValidationError):
        ledger.record_chunk_proposer_outcome(
            "attempt",
            "chunk-1",
            outcome,
            candidate_keys=candidate_keys,
            error_code=error_code,
        )


def test_proposer_errors_are_machine_codes_and_history_is_append_only(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)

    with pytest.raises(LedgerValidationError):
        ledger.record_chunk_proposer_outcome(
            "attempt",
            "chunk-1",
            "retryable_error",
            error_code="model returned private conversation text",
        )

    recorded = ledger.record_chunk_proposer_outcome(
        "attempt",
        "chunk-1",
        "retryable_error",
        error_code="model.invalid_json",
    )
    with sqlite3.connect(ledger.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                """
                UPDATE chunk_proposer_outcomes
                SET error_code = 'model.other'
                WHERE id = ?
                """,
                (recorded.outcome_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO chunk_proposer_outcomes(
                    idempotency_key, chunk_id, outcome, error_code,
                    candidate_count, candidate_set_digest, created_at
                ) VALUES (?, ?, 'retryable_error', ?, 0, ?, ?)
                """,
                (
                    "bad-attempt",
                    "chunk-1",
                    "free form error sentence",
                    hashlib.sha256(b"[]").hexdigest(),
                    "2026-07-28T00:00:00+00:00",
                ),
            )


def test_persisted_candidate_set_is_sealed_against_late_insert(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    first = ledger.record_candidate(
        "candidate-1", "X", "first", ["chunk-1"]
    )
    second = ledger.record_candidate(
        "candidate-2", "Y", "second", ["chunk-1"]
    )
    outcome = ledger.record_chunk_proposer_outcome(
        "attempt",
        "chunk-1",
        "candidates_persisted",
        candidate_keys=["candidate-1"],
    )

    with sqlite3.connect(ledger.path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="sealed"):
            connection.execute(
                """
                INSERT INTO chunk_proposer_outcome_candidates(
                    outcome_id, candidate_id
                ) VALUES (?, ?)
                """,
                (outcome.outcome_id, second.candidate_id),
            )
    assert ledger.record_chunk_proposer_outcome(
        "attempt",
        "chunk-1",
        "candidates_persisted",
        candidate_keys=["candidate-1"],
    ).created is False
    assert first.candidate_id != second.candidate_id
    assert ledger.verify_integrity()["chunk_proposer_outcomes"] == 1


def test_terminal_outcome_rejects_late_error_at_database_boundary(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    ledger.record_chunk_proposer_outcome(
        "success",
        "chunk-1",
        "zero_candidates",
    )

    with sqlite3.connect(ledger.path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="already terminal"):
            connection.execute(
                """
                INSERT INTO chunk_proposer_outcomes(
                    idempotency_key, chunk_id, outcome, error_code,
                    candidate_count, candidate_set_digest, created_at
                ) VALUES (?, ?, 'retryable_error', ?, 0, ?, ?)
                """,
                (
                    "late-error",
                    "chunk-1",
                    "provider.busy",
                    hashlib.sha256(b"[]").hexdigest(),
                    "2026-07-28T00:00:00+00:00",
                ),
            )
    assert ledger.verify_integrity()["chunk_proposer_outcomes"] == 1


def test_existing_schema_version_is_extended_in_place(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("DROP TABLE chunk_proposer_outcome_candidates")
        connection.execute("DROP TABLE chunk_proposer_outcomes")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

    migrated = LMC5Ledger(ledger.path)
    with sqlite3.connect(migrated.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    assert [
        chunk.chunk_id for chunk in migrated.list_pending_proposer_chunks()
    ] == ["chunk-1"]
    assert migrated.verify_integrity()["chunk_proposer_outcomes"] == 0
    with sqlite3.connect(migrated.path) as connection:
        assert connection.execute(
            """
            SELECT payload
            FROM raw_events
            WHERE session_id = 'session-a' AND source_event_id = 'event-1'
            """
        ).fetchone()[0] == b"first"


def test_failed_v1_migration_rolls_back_schema_and_version(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("DROP TABLE chunk_proposer_outcome_candidates")
        connection.execute("DROP TABLE chunk_proposer_outcomes")
        connection.execute(
            """
            CREATE TABLE chunk_proposer_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                chunk_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                error_code TEXT,
                candidate_count INTEGER NOT NULL,
                candidate_set_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                unexpected_column TEXT
            )
            """
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

    with pytest.raises(LedgerCorruptionError):
        LMC5Ledger(ledger.path)

    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(chunk_proposer_outcomes)"
            ).fetchall()
        }
        assert "unexpected_column" in columns
        assert (
            connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'chunk_proposer_outcome_candidates'
                """
            ).fetchone()
            is None
        )


def test_exact_column_constraintless_v1_table_is_rejected_and_rolled_back(
    tmp_path,
):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("DROP TABLE chunk_proposer_outcome_candidates")
        connection.execute("DROP TABLE chunk_proposer_outcomes")
        connection.execute(
            """
            CREATE TABLE chunk_proposer_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                error_code TEXT,
                candidate_count INTEGER NOT NULL,
                candidate_set_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

    with pytest.raises(LedgerCorruptionError):
        LMC5Ledger(ledger.path)

    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'chunk_proposer_outcome_candidates'
                """
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'index'
                  AND name = 'idx_chunk_proposer_terminal'
                """
            ).fetchone()
            is None
        )
        assert connection.execute(
            """
            SELECT payload
            FROM raw_events
            WHERE session_id = 'session-a' AND source_event_id = 'event-1'
            """
        ).fetchone()[0] == b"first"


def test_same_name_noop_trigger_is_rejected_and_migration_rolls_back(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_chunk(ledger)
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("DROP TABLE chunk_proposer_outcome_candidates")
        connection.execute("DROP TABLE chunk_proposer_outcomes")
        connection.execute("DROP TRIGGER raw_events_no_update")
        connection.execute(
            """
            CREATE TRIGGER raw_events_no_update
            BEFORE UPDATE ON raw_events
            BEGIN
                SELECT 1;
            END
            """
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

    with pytest.raises(LedgerCorruptionError):
        LMC5Ledger(ledger.path)

    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'chunk_proposer_outcome_candidates'
                """
            ).fetchone()
            is None
        )
        trigger_sql = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'trigger' AND name = 'raw_events_no_update'
            """
        ).fetchone()[0]
        assert "SELECT 1" in trigger_sql


def test_same_name_wrong_partial_terminal_index_is_rejected(tmp_path):
    ledger = _ledger(tmp_path)
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("DROP INDEX idx_chunk_proposer_terminal")
        connection.execute(
            """
            CREATE UNIQUE INDEX idx_chunk_proposer_terminal
            ON chunk_proposer_outcomes(chunk_id)
            WHERE outcome = 'zero_candidates'
            """
        )
        connection.commit()

    with pytest.raises(LedgerCorruptionError):
        LMC5Ledger(ledger.path)

    with sqlite3.connect(ledger.path) as connection:
        index_sql = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_chunk_proposer_terminal'
            """
        ).fetchone()[0]
        assert "outcome = 'zero_candidates'" in index_sql


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


def test_night_run_enforces_forward_lifecycle_and_exact_replay(tmp_path):
    ledger = _ledger(tmp_path)
    started = ledger.start_night_run("night-1", "snapshot:sha256:abc", counts={"raw": 3})
    assert started.stage == "started"
    assert started.sequence == 0

    snapshotted = ledger.record_night_stage(
        "night-1",
        "snapshotted",
        expected_stage="started",
        counts={"raw": 3},
    )
    assert snapshotted.sequence == 1
    replay = ledger.record_night_stage(
        "night-1",
        "snapshotted",
        counts={"raw": 3},
    )
    assert replay.sequence == 1
    assert replay.created is False
    assert ledger.start_night_run(
        "night-1",
        "snapshot:sha256:abc",
        counts={"raw": 3},
    ).stage == "snapshotted"
    with pytest.raises(LedgerConflictError):
        ledger.start_night_run(
            "night-1",
            "snapshot:sha256:abc",
            counts={"raw": 999},
        )

    with pytest.raises(LedgerConflictError):
        ledger.start_night_run("night-1", "other-snapshot")
    with pytest.raises(LedgerStateError):
        ledger.record_night_stage(
            "night-1", "chunked", expected_stage="started", counts={}
        )
    with pytest.raises(LedgerValidationError):
        ledger.record_night_stage(
            "night-1", "error", errors=["raw text must never enter this log"]
        )

    prior = "snapshotted"
    for sequence, stage in enumerate(
        (
            "chunked",
            "proposed",
            "dispatched",
            "metabolism_reported",
            "validated",
            "complete",
        ),
        start=2,
    ):
        completed = ledger.record_night_stage(
            "night-1",
            stage,
            expected_stage=prior,
            counts={"raw": 3, "sequence": sequence},
        )
        assert completed.sequence == sequence
        prior = stage

    exact_terminal_replay = ledger.record_night_stage(
        "night-1",
        "complete",
        counts={"sequence": 7, "raw": 3},
    )
    assert exact_terminal_replay.created is False
    assert exact_terminal_replay.sequence == 7
    with pytest.raises(LedgerConflictError):
        ledger.record_night_stage("night-1", "complete", counts={"raw": 3})
    with pytest.raises(LedgerStateError):
        ledger.record_night_stage("night-1", "error")


def test_night_run_rejects_unknown_backward_skipped_and_new_rollback(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_night_run("night", "snapshot")

    for stage in ("snapshot", "validate", "post_complete"):
        with pytest.raises(LedgerValidationError):
            ledger.record_night_stage("night", stage)
    with pytest.raises(LedgerValidationError):
        ledger.record_night_stage(
            "night", "snapshotted", expected_stage="snapshot"
        )

    for skipped in (
        "chunked",
        "proposed",
        "dispatched",
        "metabolism_reported",
        "validated",
        "complete",
        "rolled_back",
    ):
        with pytest.raises(LedgerStateError):
            ledger.record_night_stage("night", skipped)

    ledger.record_night_stage("night", "snapshotted")
    with pytest.raises(LedgerStateError):
        ledger.record_night_stage("night", "started")
    with pytest.raises(LedgerStateError):
        ledger.record_night_stage("night", "validated")


@pytest.mark.parametrize(
    "last_stage",
    (
        "started",
        "snapshotted",
        "chunked",
        "proposed",
        "dispatched",
        "metabolism_reported",
        "validated",
    ),
)
def test_any_nonterminal_night_stage_can_fail_closed(tmp_path, last_stage):
    ledger = _ledger(tmp_path)
    run_id = f"night-{last_stage}"
    ledger.start_night_run(run_id, "snapshot")
    for stage in (
        "snapshotted",
        "chunked",
        "proposed",
        "dispatched",
        "metabolism_reported",
        "validated",
    ):
        if last_stage == "started":
            break
        ledger.record_night_stage(run_id, stage)
        if stage == last_stage:
            break
    failed = ledger.record_night_stage(
        run_id,
        "error",
        expected_stage=last_stage,
        errors=["night.test_failure"],
    )
    assert failed.stage == "error"
    assert failed.errors == ("night.test_failure",)
    assert ledger.record_night_stage(
        run_id,
        "error",
        errors=["night.test_failure"],
    ).created is False
    with pytest.raises(LedgerStateError):
        ledger.record_night_stage(run_id, "complete")


def test_nonterminal_night_runs_are_read_only_stably_paginated(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_night_run("night-1", "snapshot-1", counts={"raw": 1})
    ledger.start_night_run("night-2", "snapshot-2")
    ledger.record_night_stage("night-2", "snapshotted", counts={"raw": 2})
    ledger.start_night_run("night-3", "snapshot-3")
    ledger.record_night_stage(
        "night-3", "error", errors=["night.synthetic_failure"]
    )
    ledger.start_night_run("night-4", "snapshot-4")
    for stage in (
        "snapshotted",
        "chunked",
        "proposed",
        "dispatched",
        "metabolism_reported",
        "validated",
        "complete",
    ):
        ledger.record_night_stage("night-4", stage)
    ledger.start_night_run("night-5", "snapshot-5")
    ledger.record_night_stage("night-5", "snapshotted")

    with sqlite3.connect(ledger.path) as connection:
        before = tuple(
            connection.execute(
                "SELECT COUNT(*) FROM night_run_stages"
            ).fetchone()
        )

    first_page = ledger.list_nonterminal_night_runs(limit=1)
    assert [run.run_id for run in first_page] == ["night-1"]
    assert first_page[0].cursor == first_page[0].row_id
    assert first_page[0].counts == {"raw": 1}
    second_page = ledger.list_nonterminal_night_runs(
        limit=1, after=first_page[-1].cursor
    )
    assert [run.run_id for run in second_page] == ["night-2"]
    assert second_page[0].stage == "snapshotted"
    third_page = ledger.list_nonterminal_night_runs(
        limit=1, after=second_page[-1].cursor
    )
    assert [run.run_id for run in third_page] == ["night-5"]
    assert ledger.list_nonterminal_night_runs(
        after=third_page[-1].cursor
    ) == ()
    assert ledger.list_interrupted_night_runs() == (
        first_page + second_page + third_page
    )

    with sqlite3.connect(ledger.path) as connection:
        after = tuple(
            connection.execute(
                "SELECT COUNT(*) FROM night_run_stages"
            ).fetchone()
        )
    assert after == before

    for bad_limit in (0, 1001, True):
        with pytest.raises(LedgerValidationError):
            ledger.list_nonterminal_night_runs(limit=bad_limit)
    for bad_after in (-1, True):
        with pytest.raises(LedgerValidationError):
            ledger.list_nonterminal_night_runs(after=bad_after)


def test_legacy_rolled_back_run_is_readable_terminal_but_not_newly_writable(
    tmp_path,
):
    ledger = _ledger(tmp_path)
    ledger.start_night_run("legacy", "snapshot")
    with sqlite3.connect(ledger.path) as connection:
        now = "2026-01-01T00:00:00+00:00"
        connection.execute(
            """
            INSERT INTO night_run_stages(
                run_id, sequence, stage, counts_json, errors_json, recorded_at
            ) VALUES ('legacy', 1, 'validate', '{}', '[]', ?)
            """,
            (now,),
        )
        connection.execute(
            """
            INSERT INTO night_run_stages(
                run_id, sequence, stage, counts_json, errors_json, recorded_at
            ) VALUES ('legacy', 2, 'rolled_back', '{}', '[]', ?)
            """,
            (now,),
        )
        connection.execute(
            """
            UPDATE night_runs
            SET stage = 'rolled_back', sequence = 2,
                counts_json = '{}', errors_json = '[]', updated_at = ?
            WHERE run_id = 'legacy'
            """,
            (now,),
        )
        connection.commit()

    run = ledger.get_night_run("legacy")
    assert run.stage == "rolled_back"
    assert ledger.list_nonterminal_night_runs() == ()
    assert ledger.record_night_stage("legacy", "rolled_back").created is False
    with pytest.raises(LedgerStateError):
        ledger.record_night_stage("legacy", "error")
    assert ledger.verify_integrity()["night_runs"] == 1


def test_integrity_rejects_skipped_or_divergent_night_history(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_night_run("night", "snapshot")
    with sqlite3.connect(ledger.path) as connection:
        now = "2026-01-01T00:00:00+00:00"
        connection.execute(
            """
            INSERT INTO night_run_stages(
                run_id, sequence, stage, counts_json, errors_json, recorded_at
            ) VALUES ('night', 1, 'proposed', '{}', '[]', ?)
            """,
            (now,),
        )
        connection.execute(
            """
            UPDATE night_runs
            SET stage = 'proposed', sequence = 1, updated_at = ?
            WHERE run_id = 'night'
            """,
            (now,),
        )
        connection.commit()

    with pytest.raises(
        LedgerCorruptionError, match="invalid transition"
    ):
        ledger.verify_integrity()
    with pytest.raises(
        LedgerCorruptionError, match="invalid transition"
    ):
        ledger.list_nonterminal_night_runs()


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
