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

from maintenance_barrier import MaintenanceBarrier


SCHEMA_VERSION = 2
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_READ_LIMIT = 1_000
CANDIDATE_STATUSES = frozenset(
    {"pending", "ready", "review", "rejected", "deferred", "error"}
)
PROPOSER_OUTCOMES = frozenset(
    {"zero_candidates", "candidates_persisted", "retryable_error"}
)
SUCCESSFUL_PROPOSER_OUTCOMES = frozenset(
    {"zero_candidates", "candidates_persisted"}
)
TERMINAL_NIGHT_STAGES = frozenset({"complete", "rolled_back", "error"})
NIGHT_RUN_FORWARD_STAGES = (
    "started",
    "snapshotted",
    "chunked",
    "proposed",
    "dispatched",
    "metabolism_reported",
    "validated",
    "complete",
)
NIGHT_RUN_STAGES = frozenset(
    (*NIGHT_RUN_FORWARD_STAGES, "error", "rolled_back")
)
_NIGHT_STAGE_TRANSITIONS = {
    stage: frozenset(
        {
            NIGHT_RUN_FORWARD_STAGES[index + 1],
            "error",
        }
    )
    for index, stage in enumerate(NIGHT_RUN_FORWARD_STAGES[:-1])
}
_NIGHT_STAGE_TRANSITIONS["complete"] = frozenset()
_NIGHT_STAGE_TRANSITIONS["error"] = frozenset()
_NIGHT_STAGE_TRANSITIONS["rolled_back"] = frozenset()

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MACHINE_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SCHEMA_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
    |(?P<line_comment>--[^\r\n]*)
    |(?P<block_comment>/\*.*?\*/)
    |(?P<string>'(?:''|[^'])*')
    |(?P<double_quote>"(?:""|[^"])*")
    |(?P<backtick>`(?:``|[^`])*`)
    |(?P<bracket>\[(?:\]\]|[^\]])*\])
    |(?P<word>[A-Za-z_][A-Za-z0-9_$]*)
    |(?P<number>\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)
    |(?P<operator><=|>=|<>|!=|==|\|\||[-+*/%<>=~(),.;])
    """,
    re.DOTALL | re.VERBOSE,
)

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

_PROPOSER_TABLE_SQL_DIGESTS = {
    "chunk_proposer_outcomes": (
        "5658e6f85ac2a5ed7b69319ef77e633b5cafbd3c1cc3fea235f5d66d2f086e45"
    ),
    "chunk_proposer_outcome_candidates": (
        "da02b5d0cc16e23a479ca99ecc18a42bae706a4a61c300bd2d617b5927d95e3e"
    ),
}
_PROPOSER_TABLE_COLUMNS = {
    "chunk_proposer_outcomes": (
        ("id", "INTEGER", 0, None, 1),
        ("idempotency_key", "TEXT", 1, None, 0),
        ("chunk_id", "TEXT", 1, None, 0),
        ("outcome", "TEXT", 1, None, 0),
        ("error_code", "TEXT", 0, None, 0),
        ("candidate_count", "INTEGER", 1, None, 0),
        ("candidate_set_digest", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "chunk_proposer_outcome_candidates": (
        ("outcome_id", "INTEGER", 1, None, 1),
        ("candidate_id", "INTEGER", 1, None, 2),
    ),
}
_PROPOSER_FOREIGN_KEYS = {
    "chunk_proposer_outcomes": {
        (
            "event_chunks",
            "chunk_id",
            "chunk_id",
            "NO ACTION",
            "RESTRICT",
            "NONE",
        ),
    },
    "chunk_proposer_outcome_candidates": {
        (
            "chunk_proposer_outcomes",
            "outcome_id",
            "id",
            "NO ACTION",
            "RESTRICT",
            "NONE",
        ),
        ("candidates", "candidate_id", "id", "NO ACTION", "RESTRICT", "NONE"),
    },
}
_PROPOSER_INDEXES = {
    "chunk_proposer_outcomes": {
        "idx_chunk_proposer_terminal": (
            1,
            "c",
            1,
            (("chunk_id", 0, "BINARY", 1), (None, 0, "BINARY", 0)),
        ),
        "idx_chunk_proposer_outcomes_chunk": (
            0,
            "c",
            0,
            (
                ("chunk_id", 0, "BINARY", 1),
                ("id", 0, "BINARY", 1),
                (None, 0, "BINARY", 0),
            ),
        ),
        "sqlite_autoindex_chunk_proposer_outcomes_1": (
            1,
            "u",
            0,
            (("idempotency_key", 0, "BINARY", 1), (None, 0, "BINARY", 0)),
        ),
    },
    "chunk_proposer_outcome_candidates": {
        "sqlite_autoindex_chunk_proposer_outcome_candidates_1": (
            1,
            "pk",
            0,
            (
                ("outcome_id", 0, "BINARY", 1),
                ("candidate_id", 0, "BINARY", 1),
                (None, 0, "BINARY", 0),
            ),
        ),
    },
}
_NAMED_INDEX_SQL_DIGESTS = {
    "idx_chunk_proposer_outcomes_chunk": (
        "01dbc2c81b8dbc52eba3656ba9f24fee313d2614ff5b05f329627d6828cc6aac"
    ),
    "idx_chunk_proposer_terminal": (
        "34cf885d1fbc78398a561247893a29e91566d95f6a4882679f438486c6d4f11c"
    ),
}
_TRIGGER_SQL_DIGESTS = {
    "candidate_chunks_no_delete": (
        "2ed387071851009fa8e6f9c023e58f93dc4e7883505718bff67cd2914968eae6"
    ),
    "candidate_chunks_no_update": (
        "f60149d0f57f6493ab8fc00cb94a25afc324f7a5d133a33d2cf77d9d63209c16"
    ),
    "candidates_immutable_identity": (
        "86d0ee8e16ffe1a338dc879a456facffa27bee69105846cbfb33fdd72339ce5f"
    ),
    "candidates_no_delete": (
        "c5ac3e2c341aa655dd93af3bfea49328c7a0f4f64b85fef8a779db634ec8a0f1"
    ),
    "chunk_proposer_outcome_candidates_capacity": (
        "b3ae348c15279e5bd2a57d6cc7bf1f86fa287aa72a19a9f28e2dc4b746bc2d8a"
    ),
    "chunk_proposer_outcome_candidates_cover_chunk": (
        "c24ea16825a7ccdb02cb3bb14cd99741aed42d926dd02c7db6b3e3ee48f2a3e3"
    ),
    "chunk_proposer_outcome_candidates_no_delete": (
        "2e5b5627664ebcf18735d101d68bc1846b943256ecff554902c2de7999e3ce7c"
    ),
    "chunk_proposer_outcome_candidates_no_update": (
        "2e37d0233fa48316577c5e6cd8b35a436fedde4260db48e8b1c477653293e26c"
    ),
    "chunk_proposer_outcomes_no_delete": (
        "61ad4011d6efe82f54b3f80a8243e2d01311ada8aeb8d546d979c2232fed6e4e"
    ),
    "chunk_proposer_outcomes_no_update": (
        "518df9f2044f438407129b99dd5ac38c61bdb723a9fb998bfc2812087e062009"
    ),
    "chunk_proposer_outcomes_terminal_guard": (
        "b3312c974ce946b24cf3360687dfb0dcd3be6a071410749ab62e755c5bfeeff7"
    ),
    "chunk_sources_no_delete": (
        "e05f7a67d3148a5080c68720278efb9dbb433d5c9cef32df57e4b1a7a7489be1"
    ),
    "chunk_sources_no_update": (
        "8b8981adb2423a7ef76af1a3011920bf87624af545bb526d0589bdaeaef61d9e"
    ),
    "event_chunks_no_delete": (
        "e27fdbea4bc95c2377e1d80377acb232dad20261300af9798c6ca2d11bdb13e5"
    ),
    "event_chunks_no_update": (
        "281eeea317fd96c81fb13e856c2a1ccf964941c174d583be5dd364d5b1c62382"
    ),
    "night_run_stages_no_delete": (
        "241631f872f3618b16e0cb96c393ae35165e0464999a48117bfaa543d350f063"
    ),
    "night_run_stages_no_update": (
        "783e564cf62c9ac656ae8c70319282272ee202e6fd3f9bddb9dbc36246870926"
    ),
    "night_runs_immutable_identity": (
        "59f27ee223a4fc92526b53ccc4acd6ff4c1638104beebda9d01a6ccf7467153a"
    ),
    "night_runs_no_delete": (
        "6f28a06dde1d64a1a110ded339500aa57ad7a03c631914762ba8e6cf0f5ff00a"
    ),
    "raw_events_no_delete": (
        "657692ce041a8f24b78e3d285364333896da5d31854ebef4979d282554ed1459"
    ),
    "raw_events_no_update": (
        "9ce02aba2d040c5ec7f49301a0e4019d3cf0e5a1e3fc1b10ceba79e70564b881"
    ),
    "write_receipts_no_delete": (
        "a848ca6202dd419bac7cb24adec585aebd0c314943dc69be8573c8403fd842a7"
    ),
    "write_receipts_no_update": (
        "5b4f1fd6f4f767da75d73b9556d8bc5a2b6f683c9dbb91f02aea20ad6a011dad"
    ),
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
class PendingProposerChunk:
    row_id: int
    chunk_id: str
    content: bytes
    content_digest: str
    source_event_ids: tuple[EventIdentity, ...]
    created_at: str


@dataclass(frozen=True)
class ChunkProposerOutcomeResult:
    outcome_id: int
    idempotency_key: str
    chunk_id: str
    outcome: str
    candidate_keys: tuple[str, ...]
    error_code: str | None
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
class InterruptedNightRun:
    """A nonterminal night run plus its stable pagination cursor."""

    row_id: int
    run_id: str
    snapshot_id: str
    stage: str
    counts: Mapping[str, int]
    errors: tuple[str, ...]
    sequence: int

    @property
    def cursor(self) -> int:
        return self.row_id


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


def _normalize_schema_sql(value: str) -> str:
    """Return a formatting-insensitive token stream for SQLite schema SQL."""

    tokens: list[str] = []
    position = 0
    while position < len(value):
        match = _SCHEMA_TOKEN_RE.match(value, position)
        if match is None:
            raise LedgerCorruptionError("ledger schema SQL cannot be parsed")
        position = match.end()
        kind = match.lastgroup
        token = match.group()
        if kind in {"space", "line_comment", "block_comment"}:
            continue
        if kind == "string":
            tokens.append(token)
        elif kind == "double_quote":
            tokens.append(token[1:-1].replace('""', '"').lower())
        elif kind == "backtick":
            tokens.append(token[1:-1].replace("``", "`").lower())
        elif kind == "bracket":
            tokens.append(token[1:-1].replace("]]", "]").lower())
        elif kind == "word":
            tokens.append(token.lower())
        else:
            tokens.append(token)
    for index in range(len(tokens) - 2):
        if tokens[index : index + 3] == ["if", "not", "exists"]:
            del tokens[index : index + 3]
            break
    return " ".join(tokens)


def _schema_sql_digest(value: str) -> str:
    return _sha256(_normalize_schema_sql(value).encode("utf-8"))


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


def _persisted_night_stage(
    value: Any,
    *,
    allow_legacy: bool = False,
) -> str:
    try:
        stage = _machine_code(value, "persisted stage").lower()
    except LedgerError as exc:
        raise LedgerCorruptionError("invalid night-run stage persisted") from exc
    if stage not in NIGHT_RUN_STAGES and not allow_legacy:
        raise LedgerCorruptionError("unknown night-run stage persisted")
    return stage


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


def _candidate_set_digest(candidate_keys: Sequence[str]) -> str:
    canonical = json.dumps(
        list(candidate_keys),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(canonical)


class LMC5Ledger:
    """SQLite-backed pipeline ledger with exact-source coverage accounting."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        busy_timeout_ms: int = 30_000,
        secure_permissions: bool = True,
        maintenance_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = self.path.resolve()
        if busy_timeout_ms < 1:
            raise LedgerValidationError("busy_timeout_ms must be positive")
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.secure_permissions = bool(secure_permissions)
        if maintenance_root is not None:
            barrier_root = Path(maintenance_root)
        else:
            parent = self.path.parent
            barrier_root = parent.parent if parent.name.startswith(".") else parent
        if not barrier_root.is_absolute():
            barrier_root = barrier_root.resolve()
        if not barrier_root.exists():
            barrier_root.mkdir(parents=True, mode=0o700)
        self._maintenance_barrier = MaintenanceBarrier(barrier_root)
        with self._maintenance_barrier.shared():
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
            if version not in (0, 1, SCHEMA_VERSION):
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

                CREATE TABLE IF NOT EXISTS chunk_proposer_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    chunk_id TEXT NOT NULL
                        REFERENCES event_chunks(chunk_id) ON DELETE RESTRICT,
                    outcome TEXT NOT NULL CHECK(
                        outcome IN (
                            'zero_candidates', 'candidates_persisted',
                            'retryable_error'
                        )
                    ),
                    error_code TEXT CHECK(
                        error_code IS NULL OR (
                            length(error_code) BETWEEN 1 AND 128
                            AND substr(error_code, 1, 1) GLOB '[A-Za-z0-9]'
                            AND error_code NOT GLOB '*[^A-Za-z0-9_.:-]*'
                        )
                    ),
                    candidate_count INTEGER NOT NULL
                        CHECK(candidate_count >= 0),
                    candidate_set_digest TEXT NOT NULL
                        CHECK(length(candidate_set_digest) = 64),
                    created_at TEXT NOT NULL,
                    CHECK(
                        (outcome = 'retryable_error' AND error_code IS NOT NULL)
                        OR
                        (outcome != 'retryable_error' AND error_code IS NULL)
                    )
                );
                CREATE INDEX IF NOT EXISTS idx_chunk_proposer_outcomes_chunk
                    ON chunk_proposer_outcomes(chunk_id, id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_proposer_terminal
                    ON chunk_proposer_outcomes(chunk_id)
                    WHERE outcome IN (
                        'zero_candidates', 'candidates_persisted'
                    );
                CREATE TRIGGER IF NOT EXISTS chunk_proposer_outcomes_no_update
                BEFORE UPDATE ON chunk_proposer_outcomes
                BEGIN
                    SELECT RAISE(ABORT, 'chunk proposer outcomes are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS chunk_proposer_outcomes_no_delete
                BEFORE DELETE ON chunk_proposer_outcomes
                BEGIN
                    SELECT RAISE(ABORT, 'chunk proposer outcomes are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS
                    chunk_proposer_outcomes_terminal_guard
                BEFORE INSERT ON chunk_proposer_outcomes
                WHEN EXISTS(
                    SELECT 1
                    FROM chunk_proposer_outcomes prior
                    WHERE prior.chunk_id = NEW.chunk_id
                      AND prior.outcome IN (
                          'zero_candidates', 'candidates_persisted'
                      )
                )
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'chunk proposer outcome is already terminal'
                    );
                END;

                CREATE TABLE IF NOT EXISTS chunk_proposer_outcome_candidates (
                    outcome_id INTEGER NOT NULL
                        REFERENCES chunk_proposer_outcomes(id) ON DELETE RESTRICT,
                    candidate_id INTEGER NOT NULL
                        REFERENCES candidates(id) ON DELETE RESTRICT,
                    PRIMARY KEY(outcome_id, candidate_id)
                );
                CREATE TRIGGER IF NOT EXISTS
                    chunk_proposer_outcome_candidates_no_update
                BEFORE UPDATE ON chunk_proposer_outcome_candidates
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'chunk proposer outcome candidates are append-only'
                    );
                END;
                CREATE TRIGGER IF NOT EXISTS
                    chunk_proposer_outcome_candidates_capacity
                BEFORE INSERT ON chunk_proposer_outcome_candidates
                WHEN (
                    SELECT COUNT(*)
                    FROM chunk_proposer_outcome_candidates existing
                    WHERE existing.outcome_id = NEW.outcome_id
                ) >= (
                    SELECT candidate_count
                    FROM chunk_proposer_outcomes parent
                    WHERE parent.id = NEW.outcome_id
                )
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'chunk proposer candidate set is sealed'
                    );
                END;
                CREATE TRIGGER IF NOT EXISTS
                    chunk_proposer_outcome_candidates_cover_chunk
                BEFORE INSERT ON chunk_proposer_outcome_candidates
                WHEN NOT EXISTS(
                    SELECT 1
                    FROM chunk_proposer_outcomes parent
                    JOIN candidate_chunks source
                      ON source.candidate_id = NEW.candidate_id
                     AND source.chunk_id = parent.chunk_id
                    WHERE parent.id = NEW.outcome_id
                )
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'proposer outcome candidate does not cover chunk'
                    );
                END;
                CREATE TRIGGER IF NOT EXISTS
                    chunk_proposer_outcome_candidates_no_delete
                BEFORE DELETE ON chunk_proposer_outcome_candidates
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'chunk proposer outcome candidates are append-only'
                    );
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

                """
            )
            self._verify_schema(connection)
            self._quick_check(connection)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except LedgerError:
            if connection is not None and connection.in_transaction:
                connection.rollback()
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
            "chunk_proposer_outcomes": {
                "id",
                "idempotency_key",
                "chunk_id",
                "outcome",
                "error_code",
                "candidate_count",
                "candidate_set_digest",
                "created_at",
            },
            "chunk_proposer_outcome_candidates": {
                "outcome_id",
                "candidate_id",
            },
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

        for table, expected_columns in _PROPOSER_TABLE_COLUMNS.items():
            rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
            actual_columns = tuple(
                (
                    row["name"],
                    str(row["type"]).upper(),
                    int(row["notnull"]),
                    row["dflt_value"],
                    int(row["pk"]),
                )
                for row in rows
            )
            if actual_columns != expected_columns:
                raise LedgerCorruptionError(
                    "ledger proposer table columns do not match contract"
                )

            table_row = connection.execute(
                """
                SELECT tbl_name, sql
                FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (table,),
            ).fetchone()
            if (
                table_row is None
                or table_row["tbl_name"] != table
                or not isinstance(table_row["sql"], str)
                or _schema_sql_digest(table_row["sql"])
                != _PROPOSER_TABLE_SQL_DIGESTS[table]
            ):
                raise LedgerCorruptionError(
                    "ledger proposer table constraints do not match contract"
                )

            foreign_keys = {
                (
                    row["table"],
                    row["from"],
                    row["to"],
                    row["on_update"],
                    row["on_delete"],
                    row["match"],
                )
                for row in connection.execute(
                    f"PRAGMA foreign_key_list({table})"
                ).fetchall()
            }
            if foreign_keys != _PROPOSER_FOREIGN_KEYS[table]:
                raise LedgerCorruptionError(
                    "ledger proposer foreign keys do not match contract"
                )

            index_rows = connection.execute(
                f"PRAGMA index_list({table})"
            ).fetchall()
            expected_indexes = _PROPOSER_INDEXES[table]
            actual_indexes = {
                row["name"]: (
                    int(row["unique"]),
                    row["origin"],
                    int(row["partial"]),
                )
                for row in index_rows
            }
            if set(actual_indexes) != set(expected_indexes):
                raise LedgerCorruptionError(
                    "ledger proposer indexes do not match contract"
                )
            for index_name, expected_index in expected_indexes.items():
                if actual_indexes[index_name] != expected_index[:3]:
                    raise LedgerCorruptionError(
                        "ledger proposer index semantics do not match contract"
                    )
                safe_index_name = index_name.replace('"', '""')
                index_columns = tuple(
                    (
                        row["name"],
                        int(row["desc"]),
                        str(row["coll"]).upper(),
                        int(row["key"]),
                    )
                    for row in connection.execute(
                        f'PRAGMA index_xinfo("{safe_index_name}")'
                    ).fetchall()
                )
                if index_columns != expected_index[3]:
                    raise LedgerCorruptionError(
                        "ledger proposer index columns do not match contract"
                    )

        for index_name, expected_digest in _NAMED_INDEX_SQL_DIGESTS.items():
            index_row = connection.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type = 'index' AND name = ?
                """,
                (index_name,),
            ).fetchone()
            if (
                index_row is None
                or not isinstance(index_row["sql"], str)
                or _schema_sql_digest(index_row["sql"]) != expected_digest
            ):
                raise LedgerCorruptionError(
                    "ledger named index definition does not match contract"
                )

        trigger_rows = {
            row["name"]: (row["tbl_name"], row["sql"])
            for row in connection.execute(
                """
                SELECT name, tbl_name, sql
                FROM sqlite_master
                WHERE type = 'trigger'
                """
            ).fetchall()
        }
        for trigger_name, expected_digest in _TRIGGER_SQL_DIGESTS.items():
            trigger = trigger_rows.get(trigger_name)
            if (
                trigger is None
                or not isinstance(trigger[1], str)
                or _schema_sql_digest(trigger[1]) != expected_digest
            ):
                raise LedgerCorruptionError(
                    "ledger trigger definition does not match contract"
                )

    @contextmanager
    def transaction(self) -> Iterator["LedgerTransaction"]:
        """Open a composable, immediate transaction.

        A raised exception rolls back every ledger operation issued through the
        yielded transaction facade.
        """

        with self._maintenance_barrier.shared():
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

    def record_chunk_proposer_outcome(
        self,
        idempotency_key: str,
        chunk_id: str,
        outcome: str,
        *,
        candidate_keys: Iterable[str] = (),
        error_code: str | None = None,
    ) -> ChunkProposerOutcomeResult:
        with self.transaction() as transaction:
            return transaction.record_chunk_proposer_outcome(
                idempotency_key,
                chunk_id,
                outcome,
                candidate_keys=candidate_keys,
                error_code=error_code,
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
            return self._verify_night_run_history(connection, row)
        except sqlite3.DatabaseError as exc:
            raise LedgerCorruptionError("unable to read night run") from exc
        finally:
            connection.close()

    def list_nonterminal_night_runs(
        self,
        *,
        limit: int = 100,
        after: int | None = None,
    ) -> tuple[InterruptedNightRun, ...]:
        """Return a bounded, stable feed of night runs needing recovery.

        This is deliberately read-only: callers decide whether and how to
        resume a run. ``after`` is the last ``row_id`` (or ``cursor``)
        observed by the caller.
        """

        safe_limit = _read_limit(limit)
        safe_after = _after_id(after) if after is not None else 0
        terminal = tuple(sorted(TERMINAL_NIGHT_STAGES))
        placeholders = ", ".join("?" for _ in terminal)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT rowid AS recovery_row_id, *
                FROM night_runs
                WHERE rowid > ?
                  AND stage NOT IN ({placeholders})
                ORDER BY rowid
                LIMIT ?
                """,
                (safe_after, *terminal, safe_limit),
            ).fetchall()
            records: list[InterruptedNightRun] = []
            for row in rows:
                result = self._verify_night_run_history(connection, row)
                row_id = int(row["recovery_row_id"])
                if row_id <= 0 or result.stage in TERMINAL_NIGHT_STAGES:
                    raise LedgerCorruptionError(
                        "invalid nonterminal night-run cursor persisted"
                    )
                records.append(
                    InterruptedNightRun(
                        row_id=row_id,
                        run_id=result.run_id,
                        snapshot_id=result.snapshot_id,
                        stage=result.stage,
                        counts=result.counts,
                        errors=result.errors,
                        sequence=result.sequence,
                    )
                )
            return tuple(records)
        except LedgerError:
            raise
        except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
            raise LedgerCorruptionError(
                "unable to list nonterminal night runs"
            ) from exc
        finally:
            connection.close()

    def list_interrupted_night_runs(
        self,
        *,
        limit: int = 100,
        after: int | None = None,
    ) -> tuple[InterruptedNightRun, ...]:
        """Compatibility name for :meth:`list_nonterminal_night_runs`."""

        return self.list_nonterminal_night_runs(limit=limit, after=after)

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

    def list_pending_proposer_chunks(
        self,
        *,
        limit: int = 100,
        after: int | None = None,
    ) -> tuple[PendingProposerChunk, ...]:
        """Return chunks without a successful terminal proposer outcome.

        The absence of an outcome and any number of ``retryable_error``
        outcomes are both pending. Once either ``zero_candidates`` or
        ``candidates_persisted`` is durably recorded, the chunk is excluded
        forever. ``after`` is the stable SQLite row id cursor for the
        append-only ``event_chunks`` table.
        """

        safe_limit = _read_limit(limit)
        safe_after = _after_id(after) if after is not None else 0
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._verify_proposer_outcomes(connection)
            rows = connection.execute(
                """
                SELECT ec.rowid AS row_id, ec.chunk_id, ec.content,
                       ec.content_digest, ec.created_at
                FROM event_chunks ec
                WHERE ec.rowid > ?
                  AND NOT EXISTS(
                      SELECT 1
                      FROM chunk_proposer_outcomes cpo
                      WHERE cpo.chunk_id = ec.chunk_id
                        AND cpo.outcome IN (
                            'zero_candidates', 'candidates_persisted'
                        )
                  )
                ORDER BY ec.rowid
                LIMIT ?
                """,
                (safe_after, safe_limit),
            ).fetchall()
            results: list[PendingProposerChunk] = []
            for row in rows:
                content = bytes(row["content"])
                if _sha256(content) != row["content_digest"]:
                    raise LedgerCorruptionError(
                        "persisted event-chunk digest does not match"
                    )
                source_rows = connection.execute(
                    """
                    SELECT re.session_id, re.source_event_id
                    FROM chunk_sources cs
                    JOIN raw_events re ON re.id = cs.raw_event_id
                    WHERE cs.chunk_id = ?
                    ORDER BY re.session_id, re.source_event_id
                    """,
                    (row["chunk_id"],),
                ).fetchall()
                sources = tuple(
                    EventIdentity(source["session_id"], source["source_event_id"])
                    for source in source_rows
                )
                if not sources:
                    raise LedgerCorruptionError(
                        "event chunk has no persisted source events"
                    )
                results.append(
                    PendingProposerChunk(
                        row_id=int(row["row_id"]),
                        chunk_id=row["chunk_id"],
                        content=content,
                        content_digest=row["content_digest"],
                        source_event_ids=sources,
                        created_at=row["created_at"],
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
                "unable to read pending proposer chunks"
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

            counts["chunk_proposer_outcomes"] = (
                self._verify_proposer_outcomes(connection)
            )

            counts["night_runs"] = self._verify_night_runs(connection)
            return counts
        except LedgerError:
            raise
        except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
            raise LedgerCorruptionError("ledger semantic integrity check failed") from exc
        finally:
            connection.close()

    @staticmethod
    def _verify_proposer_outcomes(connection: sqlite3.Connection) -> int:
        rows = connection.execute(
            """
            SELECT *
            FROM chunk_proposer_outcomes
            ORDER BY chunk_id, id
            """
        ).fetchall()
        terminal_chunks: set[str] = set()
        for row in rows:
            chunk_id = row["chunk_id"]
            if chunk_id in terminal_chunks:
                raise LedgerCorruptionError(
                    "chunk proposer history continues after terminal outcome"
                )
            result = LMC5Ledger._chunk_proposer_outcome_from_row(
                connection, row, created=False
            )
            if result.outcome in SUCCESSFUL_PROPOSER_OUTCOMES:
                terminal_chunks.add(result.chunk_id)
        return len(rows)

    @staticmethod
    def _verify_night_runs(connection: sqlite3.Connection) -> int:
        run_rows = connection.execute(
            "SELECT rowid AS recovery_row_id, * FROM night_runs ORDER BY rowid"
        ).fetchall()
        for row in run_rows:
            LMC5Ledger._verify_night_run_history(connection, row)
        return len(run_rows)

    @staticmethod
    def _verify_night_run_history(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> NightRunResult:
        current = LMC5Ledger._night_run_from_row(row, created=False)
        stage_rows = connection.execute(
            """
            SELECT *
            FROM night_run_stages
            WHERE run_id = ?
            ORDER BY sequence
            """,
            (current.run_id,),
        ).fetchall()
        if not stage_rows:
            raise LedgerCorruptionError(
                "night run lacks append-only stage history"
            )

        prior_stage: str | None = None
        final_counts: Mapping[str, int] | None = None
        final_errors: tuple[str, ...] | None = None
        legacy_rollback = current.stage == "rolled_back"
        for expected_sequence, stage_row in enumerate(stage_rows):
            if int(stage_row["sequence"]) != expected_sequence:
                raise LedgerCorruptionError(
                    "night-run stage history is not contiguous"
                )
            stage = _persisted_night_stage(
                stage_row["stage"],
                allow_legacy=legacy_rollback,
            )
            try:
                stage_counts = _normalize_counts(
                    json.loads(stage_row["counts_json"])
                )
                stage_errors = _normalize_errors(
                    json.loads(stage_row["errors_json"])
                )
            except LedgerError as exc:
                raise LedgerCorruptionError(
                    "invalid night-run stage history persisted"
                ) from exc
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise LedgerCorruptionError(
                    "invalid night-run stage history persisted"
                ) from exc

            if expected_sequence == 0:
                if stage != "started" or stage_errors:
                    raise LedgerCorruptionError(
                        "night-run history has an invalid initial stage"
                    )
            else:
                assert prior_stage is not None
                if legacy_rollback:
                    is_final = expected_sequence == len(stage_rows) - 1
                    allowed = (
                        prior_stage not in TERMINAL_NIGHT_STAGES
                        and stage != prior_stage
                        and (
                            (is_final and stage == "rolled_back")
                            or (
                                not is_final
                                and stage not in TERMINAL_NIGHT_STAGES
                            )
                        )
                    )
                else:
                    allowed = stage in _NIGHT_STAGE_TRANSITIONS[prior_stage]
                if not allowed:
                    raise LedgerCorruptionError(
                        "night-run history contains an invalid transition"
                    )
            prior_stage = stage
            final_counts = stage_counts
            final_errors = stage_errors

        if (
            len(stage_rows) - 1 != current.sequence
            or prior_stage != current.stage
            or dict(final_counts or {}) != dict(current.counts)
            or (final_errors or ()) != current.errors
        ):
            raise LedgerCorruptionError(
                "night run disagrees with append-only stage history"
            )
        return current

    @staticmethod
    def _chunk_proposer_outcome_from_row(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        created: bool,
    ) -> ChunkProposerOutcomeResult:
        try:
            outcome = _machine_code(row["outcome"], "persisted proposer outcome")
            if outcome not in PROPOSER_OUTCOMES:
                raise LedgerCorruptionError(
                    "unknown chunk proposer outcome persisted"
                )
            idempotency_key = _identifier(
                row["idempotency_key"], "persisted proposer idempotency key"
            )
            chunk_id = _identifier(row["chunk_id"], "persisted chunk_id")
            error_code = row["error_code"]
            if error_code is not None:
                error_code = _machine_code(
                    error_code, "persisted proposer error_code"
                )
            candidate_rows = connection.execute(
                """
                SELECT c.idempotency_key,
                       EXISTS(
                           SELECT 1
                           FROM candidate_chunks cc
                           WHERE cc.candidate_id = c.id
                             AND cc.chunk_id = ?
                       ) AS covers_chunk
                FROM chunk_proposer_outcome_candidates cpoc
                JOIN candidates c ON c.id = cpoc.candidate_id
                WHERE cpoc.outcome_id = ?
                ORDER BY c.idempotency_key
                """,
                (chunk_id, row["id"]),
            ).fetchall()
            candidate_keys = tuple(
                _identifier(
                    candidate["idempotency_key"],
                    "persisted candidate idempotency key",
                )
                for candidate in candidate_rows
            )
            candidate_count = int(row["candidate_count"])
            if (
                candidate_count < 0
                or candidate_count != len(candidate_keys)
                or _digest(
                    row["candidate_set_digest"],
                    "persisted candidate set digest",
                )
                != _candidate_set_digest(candidate_keys)
            ):
                raise LedgerCorruptionError(
                    "persisted proposer candidate set is not sealed"
                )
            if outcome == "retryable_error":
                if error_code is None or candidate_keys:
                    raise LedgerCorruptionError(
                        "retryable proposer outcome has invalid attachments"
                    )
            elif error_code is not None:
                raise LedgerCorruptionError(
                    "successful proposer outcome persisted an error"
                )
            if outcome == "zero_candidates" and candidate_keys:
                raise LedgerCorruptionError(
                    "zero-candidate proposer outcome has candidate attachments"
                )
            if outcome == "candidates_persisted":
                if not candidate_keys or not all(
                    bool(candidate["covers_chunk"])
                    for candidate in candidate_rows
                ):
                    raise LedgerCorruptionError(
                        "candidate proposer outcome lacks covered candidates"
                    )
        except LedgerCorruptionError:
            raise
        except LedgerError as exc:
            raise LedgerCorruptionError(
                "invalid chunk proposer outcome persisted"
            ) from exc
        except (KeyError, TypeError, ValueError, sqlite3.DatabaseError) as exc:
            raise LedgerCorruptionError(
                "invalid chunk proposer outcome persisted"
            ) from exc
        return ChunkProposerOutcomeResult(
            outcome_id=int(row["id"]),
            idempotency_key=idempotency_key,
            chunk_id=chunk_id,
            outcome=outcome,
            candidate_keys=candidate_keys,
            error_code=error_code,
            created=created,
        )

    @staticmethod
    def _night_run_from_row(
        row: sqlite3.Row, *, created: bool
    ) -> NightRunResult:
        try:
            raw_counts = json.loads(row["counts_json"])
            raw_errors = json.loads(row["errors_json"])
            counts = _normalize_counts(raw_counts)
            errors = _normalize_errors(raw_errors)
            stage = _persisted_night_stage(row["stage"])
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

    def record_chunk_proposer_outcome(
        self,
        idempotency_key: str,
        chunk_id: str,
        outcome: str,
        *,
        candidate_keys: Iterable[str] = (),
        error_code: str | None = None,
    ) -> ChunkProposerOutcomeResult:
        safe_key = _identifier(idempotency_key, "idempotency_key")
        safe_chunk_id = _identifier(chunk_id, "chunk_id")
        safe_outcome = _machine_code(outcome, "outcome").lower()
        if safe_outcome not in PROPOSER_OUTCOMES:
            raise LedgerValidationError("unknown chunk proposer outcome")
        if isinstance(candidate_keys, (str, bytes)):
            raise LedgerValidationError(
                "candidate_keys must be an iterable of identifiers"
            )
        safe_candidates = tuple(
            sorted(
                {
                    _identifier(candidate_key, "candidate idempotency key")
                    for candidate_key in candidate_keys
                }
            )
        )
        safe_error = (
            _machine_code(error_code, "error_code")
            if error_code is not None
            else None
        )
        if safe_outcome == "zero_candidates":
            if safe_candidates or safe_error is not None:
                raise LedgerValidationError(
                    "zero_candidates cannot attach candidates or an error"
                )
        elif safe_outcome == "candidates_persisted":
            if not safe_candidates or safe_error is not None:
                raise LedgerValidationError(
                    "candidates_persisted requires candidates and no error"
                )
        elif safe_error is None or safe_candidates:
            raise LedgerValidationError(
                "retryable_error requires one machine code and no candidates"
            )

        existing = self._connection.execute(
            """
            SELECT *
            FROM chunk_proposer_outcomes
            WHERE idempotency_key = ?
            """,
            (safe_key,),
        ).fetchone()
        if existing is not None:
            persisted = self._ledger._chunk_proposer_outcome_from_row(
                self._connection, existing, created=False
            )
            if (
                persisted.chunk_id != safe_chunk_id
                or persisted.outcome != safe_outcome
                or persisted.candidate_keys != safe_candidates
                or persisted.error_code != safe_error
            ):
                raise LedgerConflictError(
                    "proposer outcome key conflicts with persisted identity"
                )
            return persisted

        chunk_row = self._connection.execute(
            """
            SELECT content, content_digest
            FROM event_chunks
            WHERE chunk_id = ?
            """,
            (safe_chunk_id,),
        ).fetchone()
        if chunk_row is None:
            raise LedgerStateError(
                "proposer outcome references an unknown chunk"
            )
        chunk_content = bytes(chunk_row["content"])
        if _sha256(chunk_content) != chunk_row["content_digest"]:
            raise LedgerCorruptionError(
                "persisted event-chunk digest does not match"
            )
        if (
            self._connection.execute(
                """
                SELECT 1
                FROM chunk_proposer_outcomes
                WHERE chunk_id = ?
                  AND outcome IN (
                      'zero_candidates', 'candidates_persisted'
                  )
                LIMIT 1
                """,
                (safe_chunk_id,),
            ).fetchone()
            is not None
        ):
            raise LedgerStateError(
                "chunk proposer outcome is already terminal"
            )

        candidate_ids: list[int] = []
        for candidate_key in safe_candidates:
            candidate = self._connection.execute(
                """
                SELECT c.id,
                       EXISTS(
                           SELECT 1
                           FROM candidate_chunks cc
                           WHERE cc.candidate_id = c.id
                             AND cc.chunk_id = ?
                       ) AS covers_chunk
                FROM candidates c
                WHERE c.idempotency_key = ?
                """,
                (safe_chunk_id, candidate_key),
            ).fetchone()
            if candidate is None:
                raise LedgerStateError(
                    "proposer outcome references an unknown candidate"
                )
            if not bool(candidate["covers_chunk"]):
                raise LedgerStateError(
                    "proposer outcome candidate does not cover the chunk"
                )
            candidate_ids.append(int(candidate["id"]))

        cursor = self._connection.execute(
            """
            INSERT INTO chunk_proposer_outcomes(
                idempotency_key, chunk_id, outcome, error_code,
                candidate_count, candidate_set_digest, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                safe_key,
                safe_chunk_id,
                safe_outcome,
                safe_error,
                len(safe_candidates),
                _candidate_set_digest(safe_candidates),
                _utc_now(),
            ),
        )
        outcome_id = int(cursor.lastrowid)
        if candidate_ids:
            self._connection.executemany(
                """
                INSERT INTO chunk_proposer_outcome_candidates(
                    outcome_id, candidate_id
                ) VALUES (?, ?)
                """,
                (
                    (outcome_id, candidate_id)
                    for candidate_id in candidate_ids
                ),
            )
        row = self._connection.execute(
            "SELECT * FROM chunk_proposer_outcomes WHERE id = ?",
            (outcome_id,),
        ).fetchone()
        return self._ledger._chunk_proposer_outcome_from_row(
            self._connection, row, created=True
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
            result = self._ledger._verify_night_run_history(
                self._connection, row
            )
            if result.snapshot_id != safe_snapshot:
                raise LedgerConflictError(
                    "night run id conflicts with persisted snapshot"
                )
            initial_row = self._connection.execute(
                """
                SELECT counts_json
                FROM night_run_stages
                WHERE run_id = ? AND sequence = 0
                """,
                (safe_run_id,),
            ).fetchone()
            if initial_row is None:
                raise LedgerCorruptionError(
                    "night run lacks its initial stage"
                )
            try:
                initial_counts = _normalize_counts(
                    json.loads(initial_row["counts_json"])
                )
            except LedgerError as exc:
                raise LedgerCorruptionError(
                    "invalid initial night-run counts persisted"
                ) from exc
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise LedgerCorruptionError(
                    "invalid initial night-run counts persisted"
                ) from exc
            if initial_counts != safe_counts:
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
        if safe_stage not in NIGHT_RUN_STAGES:
            raise LedgerValidationError("unknown night-run stage")
        safe_expected = (
            _machine_code(expected_stage, "expected_stage").lower()
            if expected_stage is not None
            else None
        )
        if safe_expected is not None and safe_expected not in NIGHT_RUN_STAGES:
            raise LedgerValidationError("unknown expected night-run stage")
        safe_counts = _normalize_counts(counts)
        safe_errors = _normalize_errors(errors)
        row = self._connection.execute(
            "SELECT * FROM night_runs WHERE run_id = ?", (safe_run_id,)
        ).fetchone()
        if row is None:
            raise LedgerStateError("night run does not exist")
        current = self._ledger._verify_night_run_history(
            self._connection, row
        )
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
        if safe_stage not in _NIGHT_STAGE_TRANSITIONS[current.stage]:
            raise LedgerStateError("night run transition is not allowed")

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
