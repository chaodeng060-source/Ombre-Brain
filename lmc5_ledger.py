"""Durable, fail-closed bookkeeping for an LMC-5 memory pipeline.

The ledger deliberately stores pipeline facts, not operational prose:

* raw event bodies live only in ``raw_events.payload``;
* status and error fields accept short machine codes only;
* coverage is computed from exact source-event joins, never a MAX watermark.

Every mutating public method owns a SQLite ``BEGIN IMMEDIATE`` transaction.
``transaction()`` is also exposed so future ingest/night hooks can compose
several ledger operations atomically through :class:`LedgerTransaction`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_READ_LIMIT = 1_000
CANDIDATE_STATUSES = frozenset(
    {"pending", "ready", "review", "rejected", "deferred", "error"}
)
TERMINAL_NIGHT_STAGES = frozenset({"complete", "rolled_back", "error"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MACHINE_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_CANDIDATE_TRANSITIONS = {
    "pending": frozenset(
        {"pending", "ready", "review", "rejected", "deferred", "error"}
    ),
    "ready": frozenset({"ready", "review", "rejected", "error"}),
    "review": frozenset({"review", "ready", "rejected", "deferred", "error"}),
    "rejected": frozenset({"rejected"}),
    "deferred": frozenset(
        {"deferred", "pending", "ready", "review", "rejected", "error"}
    ),
    "error": frozenset({"error", "pending", "deferred", "rejected"}),
}


class LedgerError(RuntimeError):
    """Base class for ledger failures."""


class LedgerConflictError(LedgerError):
    """An idempotency identity was reused with different immutable data."""


class LedgerCorruptionError(LedgerError):
    """The database or a persisted contract is inconsistent."""


class LedgerSecurityError(LedgerError):
    """The database path does not meet the local file-safety contract."""


class LedgerValidationError(LedgerError, ValueError):
    """Caller input does not satisfy the ledger contract."""


class LedgerStateError(LedgerError):
    """A state transition failed its compare-and-set contract."""


@dataclass(frozen=True, order=True)
class EventIdentity:
    session_id: str
    source_event_id: str


@dataclass(frozen=True)
class RawEventResult:
    row_id: int
    identity: EventIdentity
    payload_digest: str
    created: bool


@dataclass(frozen=True)
class RawEventRecord:
    row_id: int
    identity: EventIdentity
    payload: bytes
    payload_digest: str
    recorded_at: str


@dataclass(frozen=True)
class ChunkResult:
    chunk_id: str
    content_digest: str
    source_event_ids: tuple[EventIdentity, ...]
    created: bool


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: int
    idempotency_key: str
    payload_digest: str
    axis: str
    status: str
    error_code: str | None
    source_chunk_ids: tuple[str, ...]
    created: bool


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: int
    idempotency_key: str
    axis: str
    payload: bytes
    payload_digest: str
    status: str
    error_code: str | None
    source_chunk_ids: tuple[str, ...]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class WriteReceiptResult:
    idempotency_key: str
    request_hash: str
    result_ref: str
    result_hash: str | None
    created: bool


@dataclass(frozen=True)
class NightRunResult:
    run_id: str
    snapshot_id: str
    stage: str
    counts: Mapping[str, int]
    errors: tuple[str, ...]
    sequence: int
    created: bool


@dataclass(frozen=True)
class CoverageReport:
    total_raw_events: int
    covered_event_ids: tuple[EventIdentity, ...]
    uncovered_event_ids: tuple[EventIdentity, ...]
    total_chunks: int
    orphan_chunk_ids: tuple[str, ...]
    candidate_status_counts: Mapping[str, int]
    pending_candidate_keys: tuple[str, ...]
    deferred_candidate_keys: tuple[str, ...]
    error_candidate_keys: tuple[str, ...]

    @property
    def holes(self) -> tuple[EventIdentity, ...]:
        """Exact raw event identities not represented by any chunk."""

        return self.uncovered_event_ids

    @property
    def covered_count(self) -> int:
        return len(self.covered_event_ids)

    @property
    def uncovered_count(self) -> int:
        return len(self.uncovered_event_ids)

    @property
    def is_fully_covered(self) -> bool:
        return not self.uncovered_event_ids and not self.orphan_chunk_ids


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _payload_bytes(value: str | bytes | bytearray | memoryview, field: str) -> bytes:
    if isinstance(value, str):
        result = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        result = bytes(value)
    else:
        raise LedgerValidationError(f"{field} must be str or bytes")
    if len(result) > MAX_PAYLOAD_BYTES:
        raise LedgerValidationError(f"{field} exceeds {MAX_PAYLOAD_BYTES} bytes")
    return result


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identifier(value: Any, field: str, *, max_length: int = 512) -> str:
    if not isinstance(value, str):
        raise LedgerValidationError(f"{field} must be a string")
    if not value or len(value) > max_length or _CONTROL_RE.search(value):
        raise LedgerValidationError(f"{field} is not a valid identifier")
    return value


def _machine_code(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _MACHINE_CODE_RE.fullmatch(value):
        raise LedgerValidationError(f"{field} must be a short machine code")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise LedgerValidationError(f"{field} must be a SHA-256 hex digest")
    normalized = value.lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise LedgerValidationError(f"{field} must be a SHA-256 hex digest")
    return normalized


def _normalize_counts(counts: Mapping[str, int] | None) -> dict[str, int]:
    if counts is None:
        return {}
    if not isinstance(counts, Mapping):
        raise LedgerValidationError("counts must be a mapping")
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        safe_key = _machine_code(key, "count key")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LedgerValidationError("count values must be non-negative integers")
        normalized[safe_key] = value
    return dict(sorted(normalized.items()))


def _normalize_errors(errors: Iterable[str] | None) -> tuple[str, ...]:
    if errors is None:
        return ()
    if isinstance(errors, (str, bytes)):
        raise LedgerValidationError("errors must be an iterable of machine codes")
    return tuple(_machine_code(value, "error") for value in errors)


def _read_limit(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > MAX_READ_LIMIT
    ):
        raise LedgerValidationError(
            f"limit must be an integer between 1 and {MAX_READ_LIMIT}"
        )
    return value


def _after_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerValidationError("after must be a non-negative integer row id")
    return value


def _created_before(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise LedgerValidationError("created_before must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerValidationError(
            "created_before must be a timezone-aware timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LedgerValidationError(
            "created_before must be a timezone-aware timestamp"
        )
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _json_counts(counts: Mapping[str, int]) -> str:
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def _json_errors(errors: Sequence[str]) -> str:
    return json.dumps(list(errors), separators=(",", ":"))


class LMC5Ledger:
    """SQLite-backed pipeline ledger with exact-source coverage accounting."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        busy_timeout_ms: int = 30_000,
        secure_permissions: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = self.path.resolve()
        if busy_timeout_ms < 1:
            raise LedgerValidationError("busy_timeout_ms must be positive")
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.secure_permissions = bool(secure_permissions)
        self._prepare_path()
        self._initialize()

    def _prepare_path(self) -> None:
        parent = self.path.parent
        try:
            if parent.exists():
                parent_stat = parent.lstat()
                if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(
                    parent_stat.st_mode
                ):
                    raise LedgerSecurityError("ledger parent must be a real directory")
            else:
                parent.mkdir(parents=True, mode=0o700)

            if self.secure_permissions:
                os.chmod(parent, 0o700)

            if self.path.exists() or self.path.is_symlink():
                file_stat = self.path.lstat()
                if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(
                    file_stat.st_mode
                ):
                    raise LedgerSecurityError("ledger must be a regular file")
                if file_stat.st_nlink != 1:
                    raise LedgerSecurityError("ledger hard links are not allowed")
            else:
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(self.path, flags, 0o600)
                os.close(descriptor)

            if self.secure_permissions:
                os.chmod(self.path, 0o600)
        except LedgerError:
            raise
        except OSError as exc:
            raise LedgerSecurityError("unable to secure ledger path") from exc

    def _assert_safe_file(self) -> None:
        try:
            file_stat = self.path.lstat()
        except OSError as exc:
            raise LedgerSecurityError("ledger file is unavailable") from exc
        if (
            stat.S_ISLNK(file_stat.st_mode)
            or not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
        ):
            raise LedgerSecurityError("ledger file identity changed")

    def _connect(self) -> sqlite3.Connection:
        self._assert_safe_file()
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self.busy_timeout_ms / 1000,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA trusted_schema = OFF")
            return connection
        except sqlite3.DatabaseError as exc:
            raise LedgerCorruptionError("unable to open ledger database") from exc

    def _initialize(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise LedgerCorruptionError("ledger could not enable WAL mode")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, SCHEMA_VERSION):
                raise LedgerCorruptionError("unsupported ledger schema version")
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS raw_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    payload_digest TEXT NOT NULL
                        CHECK(length(payload_digest) = 64),
                    recorded_at TEXT NOT NULL,
                    UNIQUE(session_id, source_event_id)
                );
                CREATE TRIGGER IF NOT EXISTS raw_events_no_update
                BEFORE UPDATE ON raw_events
                BEGIN
                    SELECT RAISE(ABORT, 'raw_events are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS raw_events_no_delete
                BEFORE DELETE ON raw_events
                BEGIN
                    SELECT RAISE(ABORT, 'raw_events are append-only');
                END;

                CREATE TABLE IF NOT EXISTS event_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content BLOB NOT NULL,
                    content_digest TEXT NOT NULL
                        CHECK(length(content_digest) = 64),
                    created_at TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS event_chunks_no_update
                BEFORE UPDATE ON event_chunks
                BEGIN
                    SELECT RAISE(ABORT, 'event_chunks are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS event_chunks_no_delete
                BEFORE DELETE ON event_chunks
                BEGIN
                    SELECT RAISE(ABORT, 'event_chunks are append-only');
                END;

                CREATE TABLE IF NOT EXISTS chunk_sources (
                    chunk_id TEXT NOT NULL
                        REFERENCES event_chunks(chunk_id) ON DELETE RESTRICT,
                    raw_event_id INTEGER NOT NULL
                        REFERENCES raw_events(id) ON DELETE RESTRICT,
                    PRIMARY KEY(chunk_id, raw_event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunk_sources_raw
                    ON chunk_sources(raw_event_id);
                CREATE TRIGGER IF NOT EXISTS chunk_sources_no_update
                BEFORE UPDATE ON chunk_sources
                BEGIN
                    SELECT RAISE(ABORT, 'chunk_sources are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS chunk_sources_no_delete
                BEFORE DELETE ON chunk_sources
                BEGIN
                    SELECT RAISE(ABORT, 'chunk_sources are append-only');
                END;

                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    axis TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    payload_digest TEXT NOT NULL
                        CHECK(length(payload_digest) = 64),
                    status TEXT NOT NULL CHECK(
                        status IN (
                            'pending', 'ready', 'review', 'rejected',
                            'deferred', 'error'
                        )
                    ),
                    error_code TEXT CHECK(
                        error_code IS NULL OR (
                            length(error_code) BETWEEN 1 AND 128
                            AND substr(error_code, 1, 1) GLOB '[A-Za-z0-9]'
                            AND error_code NOT GLOB '*[^A-Za-z0-9_.:-]*'
                        )
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_status
                    ON candidates(status);
                CREATE TRIGGER IF NOT EXISTS candidates_immutable_identity
                BEFORE UPDATE ON candidates
                WHEN NEW.id != OLD.id
                     OR NEW.idempotency_key != OLD.idempotency_key
                     OR NEW.axis != OLD.axis
                     OR NEW.payload != OLD.payload
                     OR NEW.payload_digest != OLD.payload_digest
                     OR NEW.created_at != OLD.created_at
                BEGIN
                    SELECT RAISE(ABORT, 'candidate identity is immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS candidates_no_delete
                BEFORE DELETE ON candidates
                BEGIN
                    SELECT RAISE(ABORT, 'candidates cannot be deleted');
                END;
                CREATE TABLE IF NOT EXISTS candidate_chunks (
                    candidate_id INTEGER NOT NULL
                        REFERENCES candidates(id) ON DELETE RESTRICT,
                    chunk_id TEXT NOT NULL
                        REFERENCES event_chunks(chunk_id) ON DELETE RESTRICT,
                    PRIMARY KEY(candidate_id, chunk_id)
                );
                CREATE TRIGGER IF NOT EXISTS candidate_chunks_no_update
                BEFORE UPDATE ON candidate_chunks
                BEGIN
                    SELECT RAISE(ABORT, 'candidate sources are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS candidate_chunks_no_delete
                BEFORE DELETE ON candidate_chunks
                BEGIN
                    SELECT RAISE(ABORT, 'candidate sources are immutable');
                END;

                CREATE TABLE IF NOT EXISTS write_receipts (
                    idempotency_key TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL CHECK(length(request_hash) = 64),
                    result_ref TEXT NOT NULL,
                    result_hash TEXT CHECK(
                        result_hash IS NULL OR length(result_hash) = 64
                    ),
                    created_at TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS write_receipts_no_update
                BEFORE UPDATE ON write_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'write_receipts are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS write_receipts_no_delete
                BEFORE DELETE ON write_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'write_receipts are append-only');
                END;

                CREATE TABLE IF NOT EXISTS night_runs (
                    run_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    counts_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK(sequence >= 0),
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS night_run_stages (
                    run_id TEXT NOT NULL
                        REFERENCES night_runs(run_id) ON DELETE RESTRICT,
                    sequence INTEGER NOT NULL CHECK(sequence >= 0),
                    stage TEXT NOT NULL,
                    counts_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, sequence)
                );
                CREATE TRIGGER IF NOT EXISTS night_runs_immutable_identity
                BEFORE UPDATE ON night_runs
                WHEN NEW.run_id != OLD.run_id
                     OR NEW.snapshot_id != OLD.snapshot_id
                     OR NEW.started_at != OLD.started_at
                BEGIN
                    SELECT RAISE(ABORT, 'night run identity is immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS night_runs_no_delete
                BEFORE DELETE ON night_runs
                BEGIN
                    SELECT RAISE(ABORT, 'night runs cannot be deleted');
                END;
                CREATE TRIGGER IF NOT EXISTS night_run_stages_no_update
                BEFORE UPDATE ON night_run_stages
                BEGIN
                    SELECT RAISE(ABORT, 'night stage history is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS night_run_stages_no_delete
                BEFORE DELETE ON night_run_stages
                BEGIN
                    SELECT RAISE(ABORT, 'night stage history is append-only');
                END;

                PRAGMA user_version = 1;
                COMMIT;
                """
            )
            self._verify_schema(connection)
            self._quick_check(connection)
        except LedgerError:
            raise
        except sqlite3.DatabaseError as exc:
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise LedgerCorruptionError("ledger initialization failed") from exc
        finally:
            if connection is not None:
                connection.close()
            self._secure_sidecars()

    def _secure_sidecars(self) -> None:
        if not self.secure_permissions:
            return
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self.path}{suffix}")
            try:
                if candidate.exists():
                    file_stat = candidate.lstat()
                    if stat.S_ISREG(file_stat.st_mode):
                        os.chmod(candidate, 0o600)
            except OSError as exc:
                raise LedgerSecurityError("unable to secure ledger files") from exc

    @staticmethod
    def _quick_check(connection: sqlite3.Connection, *, deep: bool = False) -> None:
        pragma = "integrity_check" if deep else "quick_check"
        try:
            rows = connection.execute(f"PRAGMA {pragma}").fetchall()
        except sqlite3.DatabaseError as exc:
            raise LedgerCorruptionError("ledger integrity check failed") from exc
        if [row[0] for row in rows] != ["ok"]:
            raise LedgerCorruptionError("ledger integrity check reported corruption")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise LedgerCorruptionError("ledger foreign-key check failed")

    @staticmethod
    def _verify_schema(connection: sqlite3.Connection) -> None:
        expected = {
            "raw_events": {
                "id",
                "session_id",
                "source_event_id",
                "payload",
                "payload_digest",
                "recorded_at",
            },
            "event_chunks": {"chunk_id", "content", "content_digest", "created_at"},
            "chunk_sources": {"chunk_id", "raw_event_id"},
            "candidates": {
                "id",
                "idempotency_key",
                "axis",
                "payload",
                "payload_digest",
                "status",
                "error_code",
                "created_at",
                "updated_at",
            },
            "candidate_chunks": {"candidate_id", "chunk_id"},
            "write_receipts": {
                "idempotency_key",
                "request_hash",
                "result_ref",
                "result_hash",
                "created_at",
            },
            "night_runs": {
                "run_id",
                "snapshot_id",
                "stage",
                "counts_json",
                "errors_json",
                "sequence",
                "started_at",
                "updated_at",
            },
            "night_run_stages": {
                "run_id",
                "sequence",
                "stage",
                "counts_json",
                "errors_json",
                "recorded_at",
            },
        }
        for table, columns in expected.items():
            rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
            if {row["name"] for row in rows} != columns:
                raise LedgerCorruptionError("ledger schema does not match contract")
        required_triggers = {
            "raw_events_no_update",
            "raw_events_no_delete",
            "event_chunks_no_update",
            "event_chunks_no_delete",
            "chunk_sources_no_update",
            "chunk_sources_no_delete",
            "candidate_chunks_no_update",
            "candidate_chunks_no_delete",
            "candidates_immutable_identity",
            "candidates_no_delete",
            "write_receipts_no_update",
            "write_receipts_no_delete",
            "night_runs_immutable_identity",
            "night_runs_no_delete",
            "night_run_stages_no_update",
            "night_run_stages_no_delete",
        }
        trigger_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
        if not required_triggers.issubset({row["name"] for row in trigger_rows}):
            raise LedgerCorruptionError("ledger append-only guards are missing")

    @contextmanager
    def transaction(self) -> Iterator["LedgerTransaction"]:
        """Open a composable, immediate transaction.

        A raised exception rolls back every ledger operation issued through the
        yielded transaction facade.
        """

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield LedgerTransaction(self, connection)
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_sidecars()

    def append_raw_event(
        self,
        session_id: str,
        source_event_id: str,
        payload: str | bytes | bytearray | memoryview,
    ) -> RawEventResult:
        with self.transaction() as transaction:
            return transaction.append_raw_event(session_id, source_event_id, payload)

    def append_raw_events(
        self,
        events: Iterable[
            tuple[str, str, str | bytes | bytearray | memoryview]
            | Mapping[str, Any]
        ],
    ) -> tuple[RawEventResult, ...]:
        """Atomically append a batch; any conflict rolls the full batch back."""

        with self.transaction() as transaction:
            results: list[RawEventResult] = []
            for event in events:
                if isinstance(event, Mapping):
                    results.append(
                        transaction.append_raw_event(
                            event["session_id"],
                            event["source_event_id"],
                            event["payload"],
                        )
                    )
                else:
                    session_id, source_event_id, payload = event
                    results.append(
                        transaction.append_raw_event(
                            session_id, source_event_id, payload
                        )
                    )
            return tuple(results)

    def record_event_chunk(
        self,
        chunk_id: str,
        content: str | bytes | bytearray | memoryview,
        source_event_ids: Iterable[EventIdentity | tuple[str, str]],
    ) -> ChunkResult:
        with self.transaction() as transaction:
            return transaction.record_event_chunk(
                chunk_id, content, source_event_ids
            )

    def record_candidate(
        self,
        idempotency_key: str,
        axis: str,
        payload: str | bytes | bytearray | memoryview,
        source_chunk_ids: Iterable[str],
        *,
        status: str = "pending",
    ) -> CandidateResult:
        with self.transaction() as transaction:
            return transaction.record_candidate(
                idempotency_key,
                axis,
                payload,
                source_chunk_ids,
                status=status,
            )

    def transition_candidate(
        self,
        idempotency_key: str,
        status: str,
        *,
        expected_status: str | None = None,
        error_code: str | None = None,
    ) -> CandidateResult:
        with self.transaction() as transaction:
            return transaction.transition_candidate(
                idempotency_key,
                status,
                expected_status=expected_status,
                error_code=error_code,
            )

    def record_write_receipt(
        self,
        idempotency_key: str,
        request_hash: str,
        result_ref: str,
        *,
        result_hash: str | None = None,
    ) -> WriteReceiptResult:
        with self.transaction() as transaction:
            return transaction.record_write_receipt(
                idempotency_key,
                request_hash,
                result_ref,
                result_hash=result_hash,
            )

    def start_night_run(
        self,
        run_id: str,
        snapshot_id: str,
        *,
        counts: Mapping[str, int] | None = None,
    ) -> NightRunResult:
        with self.transaction() as transaction:
            return transaction.start_night_run(run_id, snapshot_id, counts=counts)

    def record_night_stage(
        self,
        run_id: str,
        stage: str,
        *,
        counts: Mapping[str, int] | None = None,
        errors: Iterable[str] | None = None,
        expected_stage: str | None = None,
    ) -> NightRunResult:
        with self.transaction() as transaction:
            return transaction.record_night_stage(
                run_id,
                stage,
                counts=counts,
                errors=errors,
                expected_stage=expected_stage,
            )

    def get_night_run(self, run_id: str) -> NightRunResult:
        safe_run_id = _identifier(run_id, "run_id")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM night_runs WHERE run_id = ?", (safe_run_id,)
            ).fetchone()
            if row is None:
                raise LedgerStateError("night run does not exist")
            return self._night_run_from_row(row, created=False)
        except sqlite3.DatabaseError as exc:
            raise LedgerCorruptionError("unable to read night run") from exc
        finally:
            connection.close()

    def list_uncovered_raw_events(
        self,
        *,
        limit: int = 100,
        after: int | None = None,
        created_before: str | None = None,
    ) -> tuple[RawEventRecord, ...]:
        """Return a bounded, stable feed of raw events with no chunk source.

        ``after`` is the last internal row id observed by the caller.  The
        returned identity remains the external ``(session_id, source_event_id)``
        pair; the row id is only a monotonic paging cursor. ``created_before``
        is an exclusive, timezone-aware boundary.
        """

        safe_limit = _read_limit(limit)
        safe_after = _after_id(after) if after is not None else 0
        safe_before = (
            _created_before(created_before)
            if created_before is not None
            else None
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            clauses = [
                "re.id > ?",
                """
                NOT EXISTS(
                    SELECT 1 FROM chunk_sources cs
                    WHERE cs.raw_event_id = re.id
                )
                """,
            ]
            params: list[Any] = [safe_after]
            if safe_before is not None:
                clauses.append("re.recorded_at < ?")
                params.append(safe_before)
            params.append(safe_limit)
            rows = connection.execute(
                f"""
                SELECT re.id, re.session_id, re.source_event_id,
                       re.payload, re.payload_digest, re.recorded_at
                FROM raw_events re
                WHERE {" AND ".join(clauses)}
                ORDER BY re.id
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            results: list[RawEventRecord] = []
            for row in rows:
                payload = bytes(row["payload"])
                if _sha256(payload) != row["payload_digest"]:
                    raise LedgerCorruptionError(
                        "persisted raw-event digest does not match"
                    )
                results.append(
                    RawEventRecord(
                        row_id=int(row["id"]),
                        identity=EventIdentity(
                            row["session_id"], row["source_event_id"]
                        ),
                        payload=payload,
                        payload_digest=row["payload_digest"],
                        recorded_at=row["recorded_at"],
                    )
                )
            connection.commit()
            return tuple(results)
        except LedgerError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except sqlite3.DatabaseError as exc:
            if connection.in_transaction:
                connection.rollback()
            raise LedgerCorruptionError(
                "unable to read uncovered raw events"
            ) from exc
        finally:
            connection.close()

    def list_candidates(
        self,
        status: str,
        *,
        limit: int = 100,
        after: int | None = None,
    ) -> tuple[CandidateRecord, ...]:
        """Return a bounded candidate work queue in stable insertion order."""

        safe_status = _machine_code(status, "status").lower()
        if safe_status not in CANDIDATE_STATUSES:
            raise LedgerValidationError("unknown candidate status")
        safe_limit = _read_limit(limit)
        safe_after = _after_id(after) if after is not None else 0
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                """
                SELECT *
                FROM candidates
                WHERE status = ? AND id > ?
                ORDER BY id
                LIMIT ?
                """,
                (safe_status, safe_after, safe_limit),
            ).fetchall()
            results: list[CandidateRecord] = []
            for row in rows:
                payload = bytes(row["payload"])
                if _sha256(payload) != row["payload_digest"]:
                    raise LedgerCorruptionError(
                        "persisted candidate digest does not match"
                    )
                error_code = row["error_code"]
                if error_code is not None:
                    try:
                        _machine_code(error_code, "persisted error_code")
                    except LedgerValidationError as exc:
                        raise LedgerCorruptionError(
                            "invalid candidate machine code persisted"
                        ) from exc
                chunks = tuple(
                    source["chunk_id"]
                    for source in connection.execute(
                        """
                        SELECT chunk_id
                        FROM candidate_chunks
                        WHERE candidate_id = ?
                        ORDER BY chunk_id
                        """,
                        (row["id"],),
                    ).fetchall()
                )
                if not chunks:
                    raise LedgerCorruptionError(
                        "candidate has no persisted source chunks"
                    )
                results.append(
                    CandidateRecord(
                        candidate_id=int(row["id"]),
                        idempotency_key=row["idempotency_key"],
                        axis=row["axis"],
                        payload=payload,
                        payload_digest=row["payload_digest"],
                        status=row["status"],
                        error_code=error_code,
                        source_chunk_ids=chunks,
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )
            connection.commit()
            return tuple(results)
        except LedgerError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except sqlite3.DatabaseError as exc:
            if connection.in_transaction:
                connection.rollback()
            raise LedgerCorruptionError("unable to read candidates") from exc
        finally:
            connection.close()

    def coverage_report(self, *, session_id: str | None = None) -> CoverageReport:
        safe_session_id = (
            _identifier(session_id, "session_id") if session_id is not None else None
        )
        connection = self._connect()
        try:
            where = "WHERE re.session_id = ?" if safe_session_id is not None else ""
            params: tuple[Any, ...] = (
                (safe_session_id,) if safe_session_id is not None else ()
            )
            rows = connection.execute(
                f"""
                SELECT re.session_id, re.source_event_id,
                       EXISTS(
                           SELECT 1 FROM chunk_sources cs
                           WHERE cs.raw_event_id = re.id
                       ) AS covered
                FROM raw_events re
                {where}
                ORDER BY re.id
                """,
                params,
            ).fetchall()
            covered: list[EventIdentity] = []
            uncovered: list[EventIdentity] = []
            for row in rows:
                identity = EventIdentity(row["session_id"], row["source_event_id"])
                (covered if row["covered"] else uncovered).append(identity)

            total_chunks = int(
                connection.execute("SELECT COUNT(*) FROM event_chunks").fetchone()[0]
            )
            orphan_rows = connection.execute(
                """
                SELECT ec.chunk_id
                FROM event_chunks ec
                LEFT JOIN chunk_sources cs ON cs.chunk_id = ec.chunk_id
                WHERE cs.chunk_id IS NULL
                ORDER BY ec.chunk_id
                """
            ).fetchall()

            status_counts = {status: 0 for status in sorted(CANDIDATE_STATUSES)}
            for row in connection.execute(
                "SELECT status, COUNT(*) AS amount FROM candidates GROUP BY status"
            ).fetchall():
                if row["status"] not in CANDIDATE_STATUSES:
                    raise LedgerCorruptionError("unknown candidate status persisted")
                status_counts[row["status"]] = int(row["amount"])

            def keys_for(status: str) -> tuple[str, ...]:
                return tuple(
                    row["idempotency_key"]
                    for row in connection.execute(
                        """
                        SELECT idempotency_key
                        FROM candidates
                        WHERE status = ?
                        ORDER BY id
                        """,
                        (status,),
                    ).fetchall()
                )

            return CoverageReport(
                total_raw_events=len(rows),
                covered_event_ids=tuple(covered),
                uncovered_event_ids=tuple(uncovered),
                total_chunks=total_chunks,
                orphan_chunk_ids=tuple(row["chunk_id"] for row in orphan_rows),
                candidate_status_counts=status_counts,
                pending_candidate_keys=keys_for("pending"),
                deferred_candidate_keys=keys_for("deferred"),
                error_candidate_keys=keys_for("error"),
            )
        except LedgerError:
            raise
        except sqlite3.DatabaseError as exc:
            raise LedgerCorruptionError("unable to build coverage report") from exc
        finally:
            connection.close()

    def verify_integrity(self, *, deep: bool = True) -> Mapping[str, int]:
        """Verify SQLite structure plus application-level immutable digests."""

        connection = self._connect()
        try:
            self._verify_schema(connection)
            self._quick_check(connection, deep=deep)
            counts: dict[str, int] = {}
            for table, body_column, digest_column in (
                ("raw_events", "payload", "payload_digest"),
                ("event_chunks", "content", "content_digest"),
                ("candidates", "payload", "payload_digest"),
            ):
                rows = connection.execute(
                    f"SELECT {body_column}, {digest_column} FROM {table}"
                ).fetchall()
                for row in rows:
                    body = bytes(row[body_column])
                    if _sha256(body) != row[digest_column]:
                        raise LedgerCorruptionError(
                            "persisted payload digest does not match"
                        )
                counts[table] = len(rows)

            candidate_rows = connection.execute(
                "SELECT status, error_code FROM candidates"
            ).fetchall()
            for row in candidate_rows:
                if row["status"] not in CANDIDATE_STATUSES:
                    raise LedgerCorruptionError(
                        "invalid candidate status persisted"
                    )
                if row["error_code"] is not None:
                    try:
                        _machine_code(row["error_code"], "persisted error_code")
                    except LedgerValidationError as exc:
                        raise LedgerCorruptionError(
                            "invalid candidate machine code persisted"
                        ) from exc

            receipt_rows = connection.execute(
                "SELECT request_hash, result_hash FROM write_receipts"
            ).fetchall()
            for row in receipt_rows:
                if not _SHA256_RE.fullmatch(row["request_hash"]):
                    raise LedgerCorruptionError("invalid receipt digest persisted")
                if row["result_hash"] is not None and not _SHA256_RE.fullmatch(
                    row["result_hash"]
                ):
                    raise LedgerCorruptionError("invalid result digest persisted")
            counts["write_receipts"] = len(receipt_rows)

            run_rows = connection.execute("SELECT * FROM night_runs").fetchall()
            for row in run_rows:
                self._night_run_from_row(row, created=False)
            counts["night_runs"] = len(run_rows)
            return counts
        except LedgerError:
            raise
        except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
            raise LedgerCorruptionError("ledger semantic integrity check failed") from exc
        finally:
            connection.close()

    @staticmethod
    def _night_run_from_row(
        row: sqlite3.Row, *, created: bool
    ) -> NightRunResult:
        try:
            raw_counts = json.loads(row["counts_json"])
            raw_errors = json.loads(row["errors_json"])
            counts = _normalize_counts(raw_counts)
            errors = _normalize_errors(raw_errors)
            stage = _machine_code(row["stage"], "persisted stage")
            snapshot_id = _identifier(row["snapshot_id"], "persisted snapshot_id")
            sequence = int(row["sequence"])
            if sequence < 0:
                raise LedgerCorruptionError("negative night-run sequence")
        except LedgerCorruptionError:
            raise
        except LedgerError as exc:
            raise LedgerCorruptionError(
                "invalid night-run contract persisted"
            ) from exc
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise LedgerCorruptionError("invalid night-run contract persisted") from exc
        return NightRunResult(
            run_id=row["run_id"],
            snapshot_id=snapshot_id,
            stage=stage,
            counts=counts,
            errors=errors,
            sequence=sequence,
            created=created,
        )


class LedgerTransaction:
    """Transaction-scoped operations; instances are created by ``transaction``."""

    def __init__(self, ledger: LMC5Ledger, connection: sqlite3.Connection) -> None:
        self._ledger = ledger
        self._connection = connection

    def append_raw_event(
        self,
        session_id: str,
        source_event_id: str,
        payload: str | bytes | bytearray | memoryview,
    ) -> RawEventResult:
        safe_session = _identifier(session_id, "session_id")
        safe_source = _identifier(source_event_id, "source_event_id")
        body = _payload_bytes(payload, "payload")
        digest = _sha256(body)
        existing = self._connection.execute(
            """
            SELECT id, payload, payload_digest
            FROM raw_events
            WHERE session_id = ? AND source_event_id = ?
            """,
            (safe_session, safe_source),
        ).fetchone()
        identity = EventIdentity(safe_session, safe_source)
        if existing is not None:
            stored_body = bytes(existing["payload"])
            if (
                existing["payload_digest"] != _sha256(stored_body)
                or existing["payload_digest"] != digest
                or stored_body != body
            ):
                raise LedgerConflictError(
                    "raw event identity conflicts with persisted payload"
                )
            return RawEventResult(
                row_id=int(existing["id"]),
                identity=identity,
                payload_digest=digest,
                created=False,
            )
        cursor = self._connection.execute(
            """
            INSERT INTO raw_events(
                session_id, source_event_id, payload, payload_digest, recorded_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (safe_session, safe_source, body, digest, _utc_now()),
        )
        return RawEventResult(
            row_id=int(cursor.lastrowid),
            identity=identity,
            payload_digest=digest,
            created=True,
        )

    def record_event_chunk(
        self,
        chunk_id: str,
        content: str | bytes | bytearray | memoryview,
        source_event_ids: Iterable[EventIdentity | tuple[str, str]],
    ) -> ChunkResult:
        safe_chunk_id = _identifier(chunk_id, "chunk_id")
        body = _payload_bytes(content, "content")
        if not body.strip():
            raise LedgerValidationError(
                "event chunk content must be non-empty"
            )
        digest = _sha256(body)
        identities: set[EventIdentity] = set()
        for source in source_event_ids:
            if isinstance(source, EventIdentity):
                identity = EventIdentity(
                    _identifier(source.session_id, "session_id"),
                    _identifier(source.source_event_id, "source_event_id"),
                )
            else:
                try:
                    session_id, source_event_id = source
                except (TypeError, ValueError) as exc:
                    raise LedgerValidationError(
                        "source_event_ids must contain identity pairs"
                    ) from exc
                identity = EventIdentity(
                    _identifier(session_id, "session_id"),
                    _identifier(source_event_id, "source_event_id"),
                )
            identities.add(identity)
        if not identities:
            raise LedgerValidationError("a chunk must name at least one raw event")
        ordered_identities = tuple(sorted(identities))

        raw_ids: list[int] = []
        for identity in ordered_identities:
            row = self._connection.execute(
                """
                SELECT id FROM raw_events
                WHERE session_id = ? AND source_event_id = ?
                """,
                (identity.session_id, identity.source_event_id),
            ).fetchone()
            if row is None:
                raise LedgerStateError("chunk references an unknown raw event")
            raw_ids.append(int(row["id"]))

        existing = self._connection.execute(
            "SELECT content, content_digest FROM event_chunks WHERE chunk_id = ?",
            (safe_chunk_id,),
        ).fetchone()
        if existing is not None:
            stored_body = bytes(existing["content"])
            source_rows = self._connection.execute(
                """
                SELECT re.session_id, re.source_event_id
                FROM chunk_sources cs
                JOIN raw_events re ON re.id = cs.raw_event_id
                WHERE cs.chunk_id = ?
                ORDER BY re.session_id, re.source_event_id
                """,
                (safe_chunk_id,),
            ).fetchall()
            stored_sources = tuple(
                EventIdentity(row["session_id"], row["source_event_id"])
                for row in source_rows
            )
            if (
                existing["content_digest"] != _sha256(stored_body)
                or existing["content_digest"] != digest
                or stored_body != body
                or stored_sources != ordered_identities
            ):
                raise LedgerConflictError(
                    "chunk identity conflicts with persisted content or sources"
                )
            return ChunkResult(
                chunk_id=safe_chunk_id,
                content_digest=digest,
                source_event_ids=ordered_identities,
                created=False,
            )

        self._connection.execute(
            """
            INSERT INTO event_chunks(chunk_id, content, content_digest, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (safe_chunk_id, body, digest, _utc_now()),
        )
        self._connection.executemany(
            "INSERT INTO chunk_sources(chunk_id, raw_event_id) VALUES (?, ?)",
            ((safe_chunk_id, raw_id) for raw_id in raw_ids),
        )
        return ChunkResult(
            chunk_id=safe_chunk_id,
            content_digest=digest,
            source_event_ids=ordered_identities,
            created=True,
        )

    def record_candidate(
        self,
        idempotency_key: str,
        axis: str,
        payload: str | bytes | bytearray | memoryview,
        source_chunk_ids: Iterable[str],
        *,
        status: str = "pending",
    ) -> CandidateResult:
        safe_key = _identifier(idempotency_key, "idempotency_key")
        safe_axis = _machine_code(axis, "axis")
        safe_status = _machine_code(status, "status").lower()
        if safe_status not in CANDIDATE_STATUSES:
            raise LedgerValidationError("unknown candidate status")
        body = _payload_bytes(payload, "payload")
        if not body.strip():
            raise LedgerValidationError(
                "candidate payload must be non-empty"
            )
        digest = _sha256(body)
        chunks = tuple(
            sorted({_identifier(chunk_id, "chunk_id") for chunk_id in source_chunk_ids})
        )
        if not chunks:
            raise LedgerValidationError("a candidate must name at least one chunk")
        for chunk_id in chunks:
            if (
                self._connection.execute(
                    "SELECT 1 FROM event_chunks WHERE chunk_id = ?", (chunk_id,)
                ).fetchone()
                is None
            ):
                raise LedgerStateError("candidate references an unknown chunk")

        existing = self._connection.execute(
            "SELECT * FROM candidates WHERE idempotency_key = ?", (safe_key,)
        ).fetchone()
        if existing is not None:
            stored_body = bytes(existing["payload"])
            source_rows = self._connection.execute(
                """
                SELECT chunk_id FROM candidate_chunks
                WHERE candidate_id = ? ORDER BY chunk_id
                """,
                (existing["id"],),
            ).fetchall()
            stored_chunks = tuple(row["chunk_id"] for row in source_rows)
            if (
                existing["payload_digest"] != _sha256(stored_body)
                or existing["payload_digest"] != digest
                or stored_body != body
                or existing["axis"] != safe_axis
                or stored_chunks != chunks
            ):
                raise LedgerConflictError(
                    "candidate key conflicts with persisted immutable data"
                )
            return CandidateResult(
                candidate_id=int(existing["id"]),
                idempotency_key=safe_key,
                payload_digest=digest,
                axis=existing["axis"],
                status=existing["status"],
                error_code=existing["error_code"],
                source_chunk_ids=chunks,
                created=False,
            )

        now = _utc_now()
        cursor = self._connection.execute(
            """
            INSERT INTO candidates(
                idempotency_key, axis, payload, payload_digest, status,
                error_code, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (safe_key, safe_axis, body, digest, safe_status, now, now),
        )
        candidate_id = int(cursor.lastrowid)
        self._connection.executemany(
            "INSERT INTO candidate_chunks(candidate_id, chunk_id) VALUES (?, ?)",
            ((candidate_id, chunk_id) for chunk_id in chunks),
        )
        return CandidateResult(
            candidate_id=candidate_id,
            idempotency_key=safe_key,
            payload_digest=digest,
            axis=safe_axis,
            status=safe_status,
            error_code=None,
            source_chunk_ids=chunks,
            created=True,
        )

    def transition_candidate(
        self,
        idempotency_key: str,
        status: str,
        *,
        expected_status: str | None = None,
        error_code: str | None = None,
    ) -> CandidateResult:
        safe_key = _identifier(idempotency_key, "idempotency_key")
        safe_status = _machine_code(status, "status").lower()
        if safe_status not in CANDIDATE_STATUSES:
            raise LedgerValidationError("unknown candidate status")
        safe_expected = None
        if expected_status is not None:
            safe_expected = _machine_code(expected_status, "expected_status").lower()
            if safe_expected not in CANDIDATE_STATUSES:
                raise LedgerValidationError("unknown expected candidate status")
        safe_error = (
            _machine_code(error_code, "error_code")
            if error_code is not None
            else None
        )
        if safe_error is not None and safe_status not in {
            "review",
            "rejected",
            "deferred",
            "error",
        }:
            raise LedgerValidationError(
                "error_code is only valid for non-ready candidate states"
            )

        row = self._connection.execute(
            "SELECT * FROM candidates WHERE idempotency_key = ?", (safe_key,)
        ).fetchone()
        if row is None:
            raise LedgerStateError("candidate does not exist")
        current = row["status"]
        if safe_expected is not None and current != safe_expected:
            raise LedgerStateError("candidate compare-and-set failed")
        if safe_status not in _CANDIDATE_TRANSITIONS[current]:
            raise LedgerStateError("candidate transition is not allowed")
        if current == safe_status:
            if row["error_code"] != safe_error:
                raise LedgerStateError(
                    "idempotent status replay changed its machine code"
                )
        else:
            self._connection.execute(
                """
                UPDATE candidates
                SET status = ?, error_code = ?, updated_at = ?
                WHERE id = ?
                """,
                (safe_status, safe_error, _utc_now(), row["id"]),
            )
            row = self._connection.execute(
                "SELECT * FROM candidates WHERE id = ?", (row["id"],)
            ).fetchone()
        chunks = tuple(
            source["chunk_id"]
            for source in self._connection.execute(
                """
                SELECT chunk_id FROM candidate_chunks
                WHERE candidate_id = ? ORDER BY chunk_id
                """,
                (row["id"],),
            ).fetchall()
        )
        return CandidateResult(
            candidate_id=int(row["id"]),
            idempotency_key=safe_key,
            payload_digest=row["payload_digest"],
            axis=row["axis"],
            status=row["status"],
            error_code=row["error_code"],
            source_chunk_ids=chunks,
            created=False,
        )

    def record_write_receipt(
        self,
        idempotency_key: str,
        request_hash: str,
        result_ref: str,
        *,
        result_hash: str | None = None,
    ) -> WriteReceiptResult:
        safe_key = _identifier(idempotency_key, "idempotency_key")
        safe_request_hash = _digest(request_hash, "request_hash")
        safe_result_ref = _identifier(result_ref, "result_ref")
        safe_result_hash = (
            _digest(result_hash, "result_hash") if result_hash is not None else None
        )
        row = self._connection.execute(
            "SELECT * FROM write_receipts WHERE idempotency_key = ?", (safe_key,)
        ).fetchone()
        if row is not None:
            if (
                row["request_hash"] != safe_request_hash
                or row["result_ref"] != safe_result_ref
                or row["result_hash"] != safe_result_hash
            ):
                raise LedgerConflictError(
                    "write receipt key conflicts with persisted receipt"
                )
            return WriteReceiptResult(
                idempotency_key=safe_key,
                request_hash=row["request_hash"],
                result_ref=row["result_ref"],
                result_hash=row["result_hash"],
                created=False,
            )
        self._connection.execute(
            """
            INSERT INTO write_receipts(
                idempotency_key, request_hash, result_ref, result_hash, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                safe_key,
                safe_request_hash,
                safe_result_ref,
                safe_result_hash,
                _utc_now(),
            ),
        )
        return WriteReceiptResult(
            idempotency_key=safe_key,
            request_hash=safe_request_hash,
            result_ref=safe_result_ref,
            result_hash=safe_result_hash,
            created=True,
        )

    def start_night_run(
        self,
        run_id: str,
        snapshot_id: str,
        *,
        counts: Mapping[str, int] | None = None,
    ) -> NightRunResult:
        safe_run_id = _identifier(run_id, "run_id")
        safe_snapshot = _identifier(snapshot_id, "snapshot_id")
        safe_counts = _normalize_counts(counts)
        row = self._connection.execute(
            "SELECT * FROM night_runs WHERE run_id = ?", (safe_run_id,)
        ).fetchone()
        if row is not None:
            result = self._ledger._night_run_from_row(row, created=False)
            if result.snapshot_id != safe_snapshot:
                raise LedgerConflictError(
                    "night run id conflicts with persisted snapshot"
                )
            if (
                result.stage == "started"
                and dict(result.counts) != safe_counts
            ):
                raise LedgerConflictError(
                    "night run replay changed its initial counts"
                )
            return result

        now = _utc_now()
        counts_json = _json_counts(safe_counts)
        errors_json = _json_errors(())
        self._connection.execute(
            """
            INSERT INTO night_runs(
                run_id, snapshot_id, stage, counts_json, errors_json,
                sequence, started_at, updated_at
            ) VALUES (?, ?, 'started', ?, ?, 0, ?, ?)
            """,
            (safe_run_id, safe_snapshot, counts_json, errors_json, now, now),
        )
        self._connection.execute(
            """
            INSERT INTO night_run_stages(
                run_id, sequence, stage, counts_json, errors_json, recorded_at
            ) VALUES (?, 0, 'started', ?, ?, ?)
            """,
            (safe_run_id, counts_json, errors_json, now),
        )
        row = self._connection.execute(
            "SELECT * FROM night_runs WHERE run_id = ?", (safe_run_id,)
        ).fetchone()
        return self._ledger._night_run_from_row(row, created=True)

    def record_night_stage(
        self,
        run_id: str,
        stage: str,
        *,
        counts: Mapping[str, int] | None = None,
        errors: Iterable[str] | None = None,
        expected_stage: str | None = None,
    ) -> NightRunResult:
        safe_run_id = _identifier(run_id, "run_id")
        safe_stage = _machine_code(stage, "stage").lower()
        safe_expected = (
            _machine_code(expected_stage, "expected_stage").lower()
            if expected_stage is not None
            else None
        )
        safe_counts = _normalize_counts(counts)
        safe_errors = _normalize_errors(errors)
        row = self._connection.execute(
            "SELECT * FROM night_runs WHERE run_id = ?", (safe_run_id,)
        ).fetchone()
        if row is None:
            raise LedgerStateError("night run does not exist")
        current = self._ledger._night_run_from_row(row, created=False)
        if safe_expected is not None and current.stage != safe_expected:
            raise LedgerStateError("night run compare-and-set failed")
        counts_json = _json_counts(safe_counts)
        errors_json = _json_errors(safe_errors)
        if safe_stage == current.stage:
            if (
                counts_json != row["counts_json"]
                or errors_json != row["errors_json"]
            ):
                raise LedgerConflictError(
                    "night stage replay changed counts or errors"
                )
            return current
        if current.stage in TERMINAL_NIGHT_STAGES:
            raise LedgerStateError("night run is already terminal")

        sequence = current.sequence + 1
        now = _utc_now()
        self._connection.execute(
            """
            INSERT INTO night_run_stages(
                run_id, sequence, stage, counts_json, errors_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                safe_run_id,
                sequence,
                safe_stage,
                counts_json,
                errors_json,
                now,
            ),
        )
        self._connection.execute(
            """
            UPDATE night_runs
            SET stage = ?, counts_json = ?, errors_json = ?,
                sequence = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (
                safe_stage,
                counts_json,
                errors_json,
                sequence,
                now,
                safe_run_id,
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM night_runs WHERE run_id = ?", (safe_run_id,)
        ).fetchone()
        return self._ledger._night_run_from_row(row, created=True)
