from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest
import lmc5_candidate_reader as candidate_reader_module

from lmc5_candidate_reader import ReadOnlyLMC5CandidateLedger, ReadOnlyLedgerError
from lmc5_ledger import EventIdentity, LMC5Ledger


def _file_state(path):
    stat_result = path.stat()
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        stat_result.st_mtime_ns,
        stat_result.st_size,
    )


def _database_state(path):
    connection = sqlite3.connect(path)
    try:
        schema = tuple(connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ))
        candidates = tuple(connection.execute(
            "SELECT id, status, payload_digest, updated_at "
            "FROM candidates ORDER BY id"
        ))
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        return version, schema, candidates
    finally:
        connection.close()


def _seed(path):
    ledger = LMC5Ledger(path)
    ledger.append_raw_event("session-1", "event-1", b"redacted event")
    ledger.record_event_chunk(
        "chunk-1",
        b"redacted candidate source",
        [EventIdentity("session-1", "event-1")],
    )
    payload = json.dumps({
        "axis": "X",
        "schema": "ombre.lmc5-axis-candidate/v1",
        "base_digest": "a" * 64,
        "draft": {
            "type": "preference",
            "title": "preference",
            "content": "clear answers",
            "relation_hints": [],
        },
        "origin_run_id": "lmc5-night-test",
        "source": {"created_at": "2026-07-31T00:00:00+00:00"},
    }, sort_keys=True).encode()
    ledger.record_candidate("candidate-x-1", "X", payload, ["chunk-1"])


def test_reader_does_not_change_database_file_schema_or_candidate_state(tmp_path):
    database = tmp_path / "pipeline.sqlite3"
    _seed(database)
    before_file = _file_state(database)
    before_database = _database_state(database)

    reader = ReadOnlyLMC5CandidateLedger(database)
    rows = reader.list_candidates("pending", limit=100)

    assert len(rows) == 1
    assert rows[0].axis == "X"
    assert rows[0].status == "pending"
    assert _file_state(database) == before_file
    assert _database_state(database) == before_database


def test_reader_connection_has_two_independent_write_fences(tmp_path):
    database = tmp_path / "pipeline.sqlite3"
    _seed(database)
    reader = ReadOnlyLMC5CandidateLedger(database)
    connection = reader._connect()
    try:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            connection.execute(
                "UPDATE candidates SET status = 'ready' WHERE id = 1"
            )
    finally:
        connection.close()

    assert LMC5Ledger(database).list_candidates("pending")[0].status == "pending"


def test_reader_never_creates_a_missing_database(tmp_path):
    database = tmp_path / "missing.sqlite3"
    with pytest.raises(Exception, match="readonly.ledger_unavailable"):
        ReadOnlyLMC5CandidateLedger(database)
    assert not os.path.lexists(database)


def test_reader_preserves_source_entries_and_all_database_metadata(tmp_path):
    database = tmp_path / "pipeline.sqlite3"
    _seed(database)
    before_entries = tuple(sorted(item.name for item in tmp_path.iterdir()))
    before = database.stat()

    reader = ReadOnlyLMC5CandidateLedger(database)
    assert len(reader.list_candidates("pending", limit=100)) == 1

    after = database.stat()
    after_entries = tuple(sorted(item.name for item in tmp_path.iterdir()))
    assert after_entries == before_entries
    assert (
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_atime_ns,
    ) == (
        before.st_dev,
        before.st_ino,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_atime_ns,
    )
    assert not (tmp_path / "pipeline.sqlite3-wal").exists()
    assert not (tmp_path / "pipeline.sqlite3-shm").exists()


def test_reader_rejects_symlink_in_ancestor_chain(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    database = actual / "pipeline.sqlite3"
    _seed(database)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ReadOnlyLedgerError, match="readonly.parent_unsafe"):
        ReadOnlyLMC5CandidateLedger(alias / "pipeline.sqlite3")


def test_reader_rejects_real_ancestor_swap_during_open(tmp_path, monkeypatch):
    parent = tmp_path / "owned"
    ancestor = parent / "ancestor"
    ancestor.mkdir(parents=True)
    database = ancestor / "pipeline.sqlite3"
    _seed(database)
    replacement = parent / "replacement"
    replacement.mkdir()
    parked = parent / "parked"
    directory_flags = ReadOnlyLMC5CandidateLedger._directory_flags()
    real_open = candidate_reader_module.os.open
    swapped = False

    def swap_after_stat(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "ancestor" and dir_fd is not None and not swapped:
            swapped = True
            ancestor.rename(parked)
            replacement.rename(ancestor)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(candidate_reader_module.os, "open", swap_after_stat)
    monkeypatch.setattr(
        ReadOnlyLMC5CandidateLedger,
        "_directory_flags",
        staticmethod(lambda: directory_flags),
    )

    with pytest.raises(ReadOnlyLedgerError, match="readonly.parent_unsafe"):
        ReadOnlyLMC5CandidateLedger(database)

    assert swapped
    assert list(ancestor.iterdir()) == []


def test_reader_rejects_parent_replacement_after_snapshot(tmp_path, monkeypatch):
    parent = tmp_path / "owned"
    ancestor = parent / "ancestor"
    ancestor.mkdir(parents=True)
    database = ancestor / "pipeline.sqlite3"
    _seed(database)
    replacement = parent / "replacement"
    replacement.mkdir()
    parked = parent / "parked"
    real_assert = ReadOnlyLMC5CandidateLedger._assert_quiescent_sidecars
    calls = 0

    def swap_after_final_source_check(self, parent_descriptor):
        nonlocal calls
        real_assert(self, parent_descriptor)
        calls += 1
        if calls == 2:
            ancestor.rename(parked)
            replacement.rename(ancestor)

    monkeypatch.setattr(
        ReadOnlyLMC5CandidateLedger,
        "_assert_quiescent_sidecars",
        swap_after_final_source_check,
    )

    with pytest.raises(ReadOnlyLedgerError, match="readonly.parent_changed"):
        ReadOnlyLMC5CandidateLedger(database)

    assert calls == 2
    assert list(ancestor.iterdir()) == []


def test_reader_rejects_nonempty_wal_without_touching_it(tmp_path):
    database = tmp_path / "pipeline.sqlite3"
    _seed(database)
    wal = tmp_path / "pipeline.sqlite3-wal"
    wal.write_bytes(b"uncheckpointed")
    before = wal.stat()

    with pytest.raises(ReadOnlyLedgerError, match="readonly.uncheckpointed"):
        ReadOnlyLMC5CandidateLedger(database)

    after = wal.stat()
    assert (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_atime_ns,
    ) == (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_atime_ns,
    )
