"""Narrow, fail-closed, read-only access to LMC-5 candidate rows."""

from __future__ import annotations

import hashlib
import errno
import os
import re
import sqlite3
import stat
from pathlib import Path

from lmc5_ledger import (
    CANDIDATE_STATUSES,
    MAX_READ_LIMIT,
    SCHEMA_VERSION,
    CandidateRecord,
)


_MACHINE_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CANDIDATE_COLUMNS = (
    "id",
    "idempotency_key",
    "axis",
    "payload",
    "payload_digest",
    "status",
    "error_code",
    "created_at",
    "updated_at",
)
_CANDIDATE_CHUNK_COLUMNS = ("candidate_id", "chunk_id")
_MAX_LEDGER_BYTES = 512 * 1024 * 1024


class ReadOnlyLedgerError(RuntimeError):
    """The candidate database could not satisfy the strict read contract."""


class ReadOnlyLMC5CandidateLedger:
    """Expose only ``list_candidates`` over a SQLite read-only connection.

    Unlike :class:`lmc5_ledger.LMC5Ledger`, construction never prepares,
    migrates, chmods, checkpoints, or creates the database. SQLite also has
    ``query_only`` enabled as a second fence behind URI ``mode=ro``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        busy_timeout_ms: int = 30_000,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = self.path.resolve()
        if type(busy_timeout_ms) is not int or busy_timeout_ms < 1:
            raise ReadOnlyLedgerError("readonly.busy_timeout_invalid")
        self.busy_timeout_ms = busy_timeout_ms
        self._snapshot = self._read_snapshot()
        self._validate_schema()

    @staticmethod
    def _directory_flags() -> int:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        if (
            not nofollow
            or not directory
            or os.open not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd
        ):
            raise ReadOnlyLedgerError("readonly.platform_unsupported")
        return (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow
            | directory
        )

    @classmethod
    def _open_directory_chain(cls, directory: Path) -> int:
        absolute = Path(os.path.abspath(directory))
        try:
            descriptor = os.open(
                absolute.anchor or os.sep,
                cls._directory_flags(),
            )
        except OSError as exc:
            raise ReadOnlyLedgerError("readonly.parent_unsafe") from exc
        try:
            for component in absolute.parts[1:]:
                try:
                    expected = os.stat(
                        component,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise ReadOnlyLedgerError(
                        "readonly.parent_unsafe"
                    ) from exc
                if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(
                    expected.st_mode
                ):
                    raise ReadOnlyLedgerError("readonly.parent_unsafe")
                try:
                    child = os.open(
                        component,
                        cls._directory_flags(),
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise ReadOnlyLedgerError("readonly.parent_unsafe") from exc
                opened = os.fstat(child)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (expected.st_dev, expected.st_ino)
                ):
                    os.close(child)
                    raise ReadOnlyLedgerError("readonly.parent_unsafe")
                os.close(descriptor)
                descriptor = child
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _read_snapshot(self) -> bytes:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        noatime = getattr(os, "O_NOATIME", 0)
        if not nofollow or not noatime:
            raise ReadOnlyLedgerError("readonly.platform_unsupported")
        parent = self._open_directory_chain(self.path.parent)
        descriptor = -1
        try:
            try:
                before = os.stat(
                    self.path.name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ReadOnlyLedgerError("readonly.ledger_unavailable") from exc
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or not 0 < before.st_size <= _MAX_LEDGER_BYTES
            ):
                raise ReadOnlyLedgerError("readonly.ledger_unsafe")
            self._assert_quiescent_sidecars(parent)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= nofollow | noatime
            try:
                descriptor = os.open(
                    self.path.name,
                    flags,
                    dir_fd=parent,
                )
            except OSError as exc:
                if exc.errno in {
                    errno.EPERM,
                    errno.EACCES,
                    errno.EINVAL,
                    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                }:
                    raise ReadOnlyLedgerError(
                        "readonly.noatime_unavailable"
                    ) from exc
                raise ReadOnlyLedgerError("readonly.open_failed") from exc
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise ReadOnlyLedgerError("readonly.ledger_changed")
            chunks: list[bytes] = []
            remaining = _MAX_LEDGER_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            snapshot = b"".join(chunks)
            if len(snapshot) > _MAX_LEDGER_BYTES:
                raise ReadOnlyLedgerError("readonly.ledger_too_large")
            after = os.fstat(descriptor)
            if (
                after.st_dev,
                after.st_ino,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_atime_ns,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_atime_ns,
            ):
                raise ReadOnlyLedgerError("readonly.ledger_changed")
            if not snapshot.startswith(b"SQLite format 3\x00"):
                raise ReadOnlyLedgerError("readonly.ledger_invalid")
            try:
                current = os.stat(
                    self.path.name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ReadOnlyLedgerError("readonly.ledger_changed") from exc
            if (
                stat.S_ISLNK(current.st_mode)
                or (
                    current.st_dev,
                    current.st_ino,
                    current.st_nlink,
                    current.st_size,
                    current.st_mtime_ns,
                    current.st_ctime_ns,
                )
                != (
                    before.st_dev,
                    before.st_ino,
                    before.st_nlink,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
            ):
                raise ReadOnlyLedgerError("readonly.ledger_changed")
            self._assert_quiescent_sidecars(parent)
            verification_parent = self._open_directory_chain(self.path.parent)
            try:
                opened_parent = os.fstat(parent)
                current_parent = os.fstat(verification_parent)
                if (opened_parent.st_dev, opened_parent.st_ino) != (
                    current_parent.st_dev,
                    current_parent.st_ino,
                ):
                    raise ReadOnlyLedgerError("readonly.parent_changed")
                reachable = os.stat(
                    self.path.name,
                    dir_fd=verification_parent,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISLNK(reachable.st_mode)
                    or (
                        reachable.st_dev,
                        reachable.st_ino,
                        reachable.st_nlink,
                        reachable.st_size,
                        reachable.st_mtime_ns,
                        reachable.st_ctime_ns,
                    )
                    != (
                        before.st_dev,
                        before.st_ino,
                        before.st_nlink,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                ):
                    raise ReadOnlyLedgerError("readonly.ledger_changed")
            except ReadOnlyLedgerError:
                raise
            except OSError as exc:
                raise ReadOnlyLedgerError("readonly.parent_changed") from exc
            finally:
                os.close(verification_parent)
            return snapshot
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def _assert_quiescent_sidecars(self, parent: int) -> None:
        for suffix in ("-wal", "-journal"):
            sidecar_name = self.path.name + suffix
            try:
                sidecar = os.stat(
                    sidecar_name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ReadOnlyLedgerError(
                    "readonly.sidecar_unavailable"
                ) from exc
            if (
                stat.S_ISLNK(sidecar.st_mode)
                or not stat.S_ISREG(sidecar.st_mode)
                or sidecar.st_nlink != 1
            ):
                raise ReadOnlyLedgerError("readonly.sidecar_unsafe")
            if sidecar.st_size:
                raise ReadOnlyLedgerError("readonly.uncheckpointed")

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                ":memory:",
                timeout=self.busy_timeout_ms / 1000,
                isolation_level=None,
                check_same_thread=False,
            )
            image = bytearray(self._snapshot)
            # A clean main database may retain WAL read/write version bytes
            # even when its WAL is absent or empty.  The immutable in-memory
            # copy has no sidecar VFS, so normalize only the private copy back
            # to rollback-format header bytes.  Source bytes stay untouched;
            # a non-empty WAL was rejected before this snapshot was read.
            image[18] = image[19] = 1
            connection.deserialize(bytes(image))
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.execute("PRAGMA query_only = ON")
            query_only = int(
                connection.execute("PRAGMA query_only").fetchone()[0]
            )
            if query_only != 1:
                raise ReadOnlyLedgerError("readonly.query_fence_failed")
            return connection
        except ReadOnlyLedgerError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ReadOnlyLedgerError("readonly.open_failed") from exc

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
        return tuple(
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})")
        )

    def _validate_schema(self) -> None:
        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != SCHEMA_VERSION:
                raise ReadOnlyLedgerError("readonly.schema_version_invalid")
            if self._columns(connection, "candidates") != _CANDIDATE_COLUMNS:
                raise ReadOnlyLedgerError("readonly.candidate_schema_invalid")
            if self._columns(
                connection,
                "candidate_chunks",
            ) != _CANDIDATE_CHUNK_COLUMNS:
                raise ReadOnlyLedgerError("readonly.chunk_schema_invalid")
        except ReadOnlyLedgerError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ReadOnlyLedgerError("readonly.schema_unavailable") from exc
        finally:
            connection.close()

    def list_candidates(
        self,
        status: str,
        *,
        limit: int = 100,
        after: int | None = None,
    ) -> tuple[CandidateRecord, ...]:
        if type(status) is not str or status.lower() not in CANDIDATE_STATUSES:
            raise ReadOnlyLedgerError("readonly.status_invalid")
        safe_status = status.lower()
        if type(limit) is not int or not 1 <= limit <= MAX_READ_LIMIT:
            raise ReadOnlyLedgerError("readonly.limit_invalid")
        safe_after = 0 if after is None else after
        if type(safe_after) is not int or safe_after < 0:
            raise ReadOnlyLedgerError("readonly.cursor_invalid")

        connection = self._connect()
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                """
                SELECT * FROM candidates
                WHERE status = ? AND id > ?
                ORDER BY id
                LIMIT ?
                """,
                (safe_status, safe_after, limit),
            ).fetchall()
            results: list[CandidateRecord] = []
            for row in rows:
                payload = bytes(row["payload"])
                payload_digest = str(row["payload_digest"])
                if hashlib.sha256(payload).hexdigest() != payload_digest:
                    raise ReadOnlyLedgerError("readonly.payload_digest_mismatch")
                error_code = row["error_code"]
                if error_code is not None and (
                    type(error_code) is not str
                    or _MACHINE_CODE_RE.fullmatch(error_code) is None
                ):
                    raise ReadOnlyLedgerError("readonly.error_code_invalid")
                chunks = tuple(
                    str(chunk["chunk_id"])
                    for chunk in connection.execute(
                        """
                        SELECT chunk_id FROM candidate_chunks
                        WHERE candidate_id = ?
                        ORDER BY chunk_id
                        """,
                        (row["id"],),
                    )
                )
                if not chunks or any(not chunk for chunk in chunks):
                    raise ReadOnlyLedgerError("readonly.source_chunks_invalid")
                values = {
                    key: row[key]
                    for key in (
                        "idempotency_key",
                        "axis",
                        "status",
                        "created_at",
                        "updated_at",
                    )
                }
                if any(type(value) is not str or not value for value in values.values()):
                    raise ReadOnlyLedgerError("readonly.row_invalid")
                results.append(CandidateRecord(
                    candidate_id=int(row["id"]),
                    idempotency_key=values["idempotency_key"],
                    axis=values["axis"],
                    payload=payload,
                    payload_digest=payload_digest,
                    status=values["status"],
                    error_code=error_code,
                    source_chunk_ids=chunks,
                    created_at=values["created_at"],
                    updated_at=values["updated_at"],
                ))
            connection.commit()
            return tuple(results)
        except ReadOnlyLedgerError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except sqlite3.DatabaseError as exc:
            if connection.in_transaction:
                connection.rollback()
            raise ReadOnlyLedgerError("readonly.query_failed") from exc
        finally:
            connection.close()


__all__ = ["ReadOnlyLMC5CandidateLedger", "ReadOnlyLedgerError"]
