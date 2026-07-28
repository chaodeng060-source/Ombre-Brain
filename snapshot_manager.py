"""Bounded, fail-closed snapshots for LMC-5 night runs.

The mutable Ombre store is a mixed filesystem:

* bucket bodies are regular files below the configured ``buckets_dir``;
* vector, audit, and pipeline ledgers are SQLite databases in that tree;
* lock files and SQLite WAL/SHM sidecars are runtime coordination state.

Copying that tree recursively is not a valid snapshot: a WAL-backed database
may have committed pages that are not present in the main database file yet.
This module copies ordinary files through verified file descriptors and copies
every SQLite database through SQLite's online backup API.  A snapshot becomes
visible only after a complete, hashed manifest has been fsynced and the staging
directory is atomically renamed into place.

Restore is deliberately limited to a new, isolated destination.  It refuses
the live source tree, the backup tree, existing destinations, symlinks, hard
links, extra files, manifest traversal, and hash mismatches.  Promoting an
isolated restore into production is intentionally outside this module.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator, Mapping, Sequence

from maintenance_barrier import MaintenanceBarrier


MANIFEST_SCHEMA = "ombre.lmc5.snapshot/v2"
EXCLUSION_POLICY_SCHEMA = "ombre.lmc5.snapshot-exclusions/v1"
MANIFEST_NAME = "manifest.json"
FILES_DIR = "files"
_SQLITE_HEADER = b"SQLite format 3\x00"
_SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_COORDINATION_DIRECTORIES = frozenset({".locks", ".curated-write-locks"})
_COORDINATION_FILE_SUFFIXES = (".lock",)
_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COPY_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_EXCLUDED_ROOT_FILES = (
    ".ombre_runtime.yaml",
    "body_state.json",
    "dehydration_cache.db",
    "dehydration_cache.db-journal",
    "dehydration_cache.db-shm",
    "dehydration_cache.db-wal",
)
_EXCLUDED_ROOT_DIRECTORIES = (
    ".session_surface",
    "twin",
)
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


class SnapshotError(RuntimeError):
    """Base class for snapshot failures."""


class SnapshotSecurityError(SnapshotError):
    """A filesystem boundary is unsafe."""


class SnapshotValidationError(SnapshotError, ValueError):
    """A caller or manifest value is invalid."""


class SnapshotIntegrityError(SnapshotError):
    """Persisted snapshot content is incomplete, corrupt, or inconsistent."""


class SnapshotLimitError(SnapshotError):
    """A configured snapshot bound would be exceeded."""


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    kind: str
    size: int
    sha256: str
    mode: int


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_id: str
    snapshot_path: Path
    manifest_sha256: str
    file_count: int
    total_bytes: int
    files: tuple[SnapshotFile, ...]


@dataclass(frozen=True)
class RestoreResult:
    snapshot_id: str
    destination: Path
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class _SourceEntry:
    relative: PurePosixPath
    source: Path
    initial_stat: os.stat_result
    kind: str


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(path), os.fspath(root))) == os.fspath(
            root
        )
    except ValueError:
        return False


def _overlaps(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    """Persist every directory entry in a newly built private tree."""

    if os.name == "nt":
        return
    directories = [root]
    for current, names, _files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in names:
            child = current_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISDIR(
                child_stat.st_mode
            ):
                raise SnapshotSecurityError(
                    f"unsafe directory appeared while syncing: {child}"
                )
            directories.append(child)
    for directory in sorted(
        set(directories),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _validate_positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SnapshotValidationError(f"{field} must be a positive integer")
    return value


def _validate_snapshot_id(value: str) -> str:
    if not isinstance(value, str) or not _SNAPSHOT_ID_RE.fullmatch(value):
        raise SnapshotValidationError("invalid snapshot_id")
    return value


def _validate_manifest_digest(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SnapshotValidationError("invalid expected_manifest_sha256")
    return value


def _safe_manifest_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SnapshotIntegrityError("manifest contains an invalid file path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise SnapshotIntegrityError("manifest file path escapes the snapshot")
    return relative


def _safe_join(root: Path, relative: PurePosixPath) -> Path:
    candidate = _absolute(root.joinpath(*relative.parts))
    if not _is_within(candidate, root) or candidate == root:
        raise SnapshotSecurityError("path escapes its configured root")
    return candidate


def _assert_no_symlink_components(
    path: Path,
    *,
    allow_missing_leaf: bool = False,
) -> None:
    """Reject symlinks in every existing component of an absolute path."""

    absolute = _absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            item_stat = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf or not is_leaf:
                continue
            raise SnapshotSecurityError(f"required path is missing: {current}")
        if stat.S_ISLNK(item_stat.st_mode):
            raise SnapshotSecurityError(f"symlink path component rejected: {current}")
        if not is_leaf and not stat.S_ISDIR(item_stat.st_mode):
            raise SnapshotSecurityError(
                f"non-directory path component rejected: {current}"
            )


def _assert_real_directory(path: Path) -> None:
    _assert_no_symlink_components(path)
    try:
        item_stat = path.lstat()
    except OSError as exc:
        raise SnapshotSecurityError(f"directory is unavailable: {path}") from exc
    if not stat.S_ISDIR(item_stat.st_mode):
        raise SnapshotSecurityError(f"path is not a real directory: {path}")


def _prepare_private_directory(path: Path) -> None:
    _assert_no_symlink_components(path, allow_missing_leaf=True)
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        _assert_real_directory(path)
        if os.name != "nt":
            os.chmod(path, 0o700)
    except SnapshotError:
        raise
    except OSError as exc:
        raise SnapshotSecurityError(f"unable to prepare directory: {path}") from exc


@contextmanager
def _verified_reader(
    path: Path,
    *,
    expected: os.stat_result | None = None,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Open one regular, single-link file without following the final symlink."""

    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise SnapshotSecurityError(f"unsafe source file rejected: {path}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            os.close(descriptor)
            raise SnapshotSecurityError(f"source file changed while opening: {path}")
        if expected is not None and (
            opened.st_dev != expected.st_dev or opened.st_ino != expected.st_ino
        ):
            os.close(descriptor)
            raise SnapshotIntegrityError(
                f"source file changed after enumeration: {path}"
            )
        handle = os.fdopen(descriptor, "rb")
        try:
            yield handle, opened
        finally:
            handle.close()
    except SnapshotError:
        raise
    except OSError as exc:
        raise SnapshotSecurityError(f"unable to read source file: {path}") from exc


def _hash_open_file(
    handle: BinaryIO,
    *,
    max_bytes: int | None = None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        block = handle.read(_COPY_CHUNK_SIZE)
        if not block:
            break
        digest.update(block)
        size += len(block)
        if max_bytes is not None and size > max_bytes:
            raise SnapshotLimitError("file grew past its verification bound")
    return digest.hexdigest(), size


def _hash_file(
    path: Path,
    *,
    max_bytes: int | None = None,
) -> tuple[str, int]:
    with _verified_reader(path) as (handle, opened):
        if max_bytes is not None and opened.st_size > max_bytes:
            raise SnapshotLimitError(f"file exceeds verification bound: {path}")
        return _hash_open_file(handle, max_bytes=max_bytes)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _manifest_exclusion_policy() -> dict[str, object]:
    """Return the exact, versioned set of runtime-only snapshot exclusions."""

    return {
        "schema": EXCLUSION_POLICY_SCHEMA,
        "root_files": list(_EXCLUDED_ROOT_FILES),
        "root_directories": list(_EXCLUDED_ROOT_DIRECTORIES),
        "directory_names_any_depth": sorted(_COORDINATION_DIRECTORIES),
        "file_suffixes_any_depth": list(_COORDINATION_FILE_SUFFIXES),
        "sqlite_sidecar_suffixes_for_captured_databases": list(
            _SQLITE_SIDECAR_SUFFIXES
        ),
    }


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` without ever replacing ``destination``."""

    if os.name == "nt":
        # Windows os.rename() already refuses to replace an existing target.
        os.rename(source, destination)
        return
    if not sys.platform.startswith("linux"):
        raise SnapshotSecurityError(
            "atomic no-replace publication is unavailable on this platform"
        )

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise SnapshotSecurityError(
            "atomic no-replace publication is unavailable"
        ) from exc

    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return

    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    if error_number in {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }:
        raise SnapshotSecurityError(
            "atomic no-replace publication is unavailable"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        os.fspath(destination),
    )


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    parent = path.parent
    _assert_real_directory(parent)
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short atomic write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(parent)
    except Exception:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class SnapshotManager:
    """Create verified snapshots and restore them only into isolated roots."""

    def __init__(
        self,
        source_root: str | os.PathLike[str],
        backup_root: str | os.PathLike[str],
        *,
        max_files: int = 100_000,
        max_total_bytes: int = 16 * 1024 * 1024 * 1024,
        max_file_bytes: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        self.source_root = _absolute(source_root)
        self.backup_root = _absolute(backup_root)
        self.max_files = _validate_positive_int(max_files, "max_files")
        self.max_total_bytes = _validate_positive_int(
            max_total_bytes, "max_total_bytes"
        )
        self.max_file_bytes = _validate_positive_int(
            max_file_bytes, "max_file_bytes"
        )

        _assert_real_directory(self.source_root)
        if _overlaps(self.source_root, self.backup_root):
            raise SnapshotSecurityError(
                "source_root and backup_root must be disjoint"
            )
        self.maintenance_barrier = MaintenanceBarrier(self.source_root)
        _prepare_private_directory(self.backup_root)

    def create_snapshot(self, snapshot_id: str) -> SnapshotResult:
        with self.maintenance_barrier.exclusive():
            return self._create_snapshot_locked(snapshot_id)

    def _create_snapshot_locked(self, snapshot_id: str) -> SnapshotResult:
        safe_id = _validate_snapshot_id(snapshot_id)
        self._revalidate_roots()
        destination = self.backup_root / safe_id
        if destination.exists() or destination.is_symlink():
            raise SnapshotIntegrityError("snapshot_id already exists")

        staging = self.backup_root / f".{safe_id}.{uuid.uuid4().hex}.tmp"
        os.mkdir(staging, 0o700)
        published_identity: tuple[int, int] | None = None
        try:
            payload_root = staging / FILES_DIR
            os.mkdir(payload_root, 0o700)
            source_entries = self._collect_source_entries()
            files: list[SnapshotFile] = []
            total_bytes = 0

            for entry in source_entries:
                target = _safe_join(payload_root, entry.relative)
                self._prepare_target_parent(payload_root, target.parent)
                if entry.kind == "sqlite":
                    size, digest = self._backup_sqlite(entry, target)
                else:
                    size, digest = self._copy_regular(entry, target)
                if size > self.max_file_bytes:
                    raise SnapshotLimitError(
                        f"snapshot file exceeds max_file_bytes: {entry.relative}"
                    )
                total_bytes += size
                if total_bytes > self.max_total_bytes:
                    raise SnapshotLimitError("snapshot exceeds max_total_bytes")
                files.append(
                    SnapshotFile(
                        path=entry.relative.as_posix(),
                        kind=entry.kind,
                        size=size,
                        sha256=digest,
                        mode=stat.S_IMODE(entry.initial_stat.st_mode),
                    )
                )

            # Ordinary bucket files are not transactional.  Detect additions,
            # removals, replacements, and in-place edits across the snapshot
            # window rather than silently publishing an incomplete tree.
            source_after = self._collect_source_entries()
            if self._source_signature(source_entries) != self._source_signature(
                source_after
            ):
                raise SnapshotIntegrityError("source tree changed during snapshot")

            manifest = {
                "schema": MANIFEST_SCHEMA,
                "snapshot_id": safe_id,
                "created_at": _utc_now(),
                "source_root_sha256": hashlib.sha256(
                    os.fspath(self.source_root).encode("utf-8")
                ).hexdigest(),
                "exclusion_policy": _manifest_exclusion_policy(),
                "file_count": len(files),
                "total_bytes": total_bytes,
                "files": [
                    {
                        "path": item.path,
                        "kind": item.kind,
                        "size": item.size,
                        "sha256": item.sha256,
                        "mode": item.mode,
                    }
                    for item in files
                ],
            }
            manifest_payload = _canonical_json_bytes(manifest)
            if len(manifest_payload) > _MAX_MANIFEST_BYTES:
                raise SnapshotLimitError("snapshot manifest is too large")
            _fsync_tree_directories(payload_root)
            _atomic_write_bytes(staging / MANIFEST_NAME, manifest_payload)
            manifest_digest = hashlib.sha256(manifest_payload).hexdigest()
            _fsync_directory(staging)
            self._verify_snapshot_root(
                staging,
                safe_id,
                expected_manifest_sha256=manifest_digest,
            )

            # Atomic no-replace is the single publication point for a complete
            # snapshot.  A check followed by plain os.rename() is insufficient:
            # POSIX would replace an empty directory created in between.
            staging_stat = staging.lstat()
            try:
                _rename_no_replace(staging, destination)
            except FileExistsError as exc:
                raise SnapshotIntegrityError(
                    "snapshot_id raced with another writer"
                ) from exc
            published_identity = (staging_stat.st_dev, staging_stat.st_ino)
            _fsync_directory(self.backup_root)
            return SnapshotResult(
                snapshot_id=safe_id,
                snapshot_path=destination,
                manifest_sha256=manifest_digest,
                file_count=len(files),
                total_bytes=total_bytes,
                files=tuple(files),
            )
        except Exception:
            if published_identity is not None:
                self._rollback_published_tree(
                    destination,
                    owned_parent=self.backup_root,
                    expected_identity=published_identity,
                )
            self._cleanup_private_tree(
                staging,
                owned_parent=self.backup_root,
                name_prefix=f".{safe_id}.",
            )
            raise

    def verify_snapshot(
        self,
        snapshot_id: str,
        *,
        expected_manifest_sha256: str,
    ) -> SnapshotResult:
        safe_id = _validate_snapshot_id(snapshot_id)
        expected_digest = _validate_manifest_digest(expected_manifest_sha256)
        self._revalidate_roots()
        return self._verify_snapshot_root(
            self.backup_root / safe_id,
            safe_id,
            expected_manifest_sha256=expected_digest,
        )

    def _verify_snapshot_root(
        self,
        snapshot_root: Path,
        snapshot_id: str,
        *,
        expected_manifest_sha256: str,
    ) -> SnapshotResult:
        snapshot_root, files, manifest_digest = self._load_manifest_at(
            snapshot_root,
            snapshot_id,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        actual_paths = self._enumerate_snapshot_payload(snapshot_root / FILES_DIR)
        expected_paths = {PurePosixPath(item.path) for item in files}
        if actual_paths != expected_paths:
            raise SnapshotIntegrityError(
                "snapshot payload does not exactly match its manifest"
            )

        total = 0
        for item in files:
            source = _safe_join(snapshot_root / FILES_DIR, PurePosixPath(item.path))
            digest, size = _hash_file(source, max_bytes=item.size)
            if digest != item.sha256 or size != item.size:
                raise SnapshotIntegrityError(f"snapshot hash mismatch: {item.path}")
            if item.kind == "sqlite":
                self._verify_sqlite(source)
            total += size
        if total != sum(item.size for item in files):
            raise SnapshotIntegrityError("snapshot byte accounting mismatch")
        return SnapshotResult(
            snapshot_id=snapshot_id,
            snapshot_path=snapshot_root,
            manifest_sha256=manifest_digest,
            file_count=len(files),
            total_bytes=total,
            files=files,
        )

    def restore_isolated(
        self,
        snapshot_id: str,
        destination_root: str | os.PathLike[str],
        *,
        expected_manifest_sha256: str,
    ) -> RestoreResult:
        """Restore into a brand-new tree that cannot overlap live or backup data."""

        safe_id = _validate_snapshot_id(snapshot_id)
        destination = _absolute(destination_root)
        self._revalidate_roots()
        if _overlaps(destination, self.source_root) or _overlaps(
            destination, self.backup_root
        ):
            raise SnapshotSecurityError(
                "isolated restore destination overlaps live or backup data"
            )
        _assert_no_symlink_components(destination, allow_missing_leaf=True)
        if destination.exists() or destination.is_symlink():
            raise SnapshotSecurityError(
                "isolated restore destination must not already exist"
            )
        parent = destination.parent
        _assert_real_directory(parent)

        expected_digest = _validate_manifest_digest(expected_manifest_sha256)
        verified = self.verify_snapshot(
            safe_id,
            expected_manifest_sha256=expected_digest,
        )
        staging = parent / f".{destination.name}.restore.{uuid.uuid4().hex}.tmp"
        if _overlaps(staging, self.source_root) or _overlaps(
            staging, self.backup_root
        ):
            raise SnapshotSecurityError("restore staging path overlaps protected data")
        os.mkdir(staging, 0o700)
        published_identity: tuple[int, int] | None = None
        try:
            for item in verified.files:
                relative = PurePosixPath(item.path)
                source = _safe_join(
                    verified.snapshot_path / FILES_DIR,
                    relative,
                )
                target = _safe_join(staging, relative)
                self._prepare_target_parent(staging, target.parent)
                if item.kind == "sqlite":
                    digest_before, size_before = _hash_file(
                        source,
                        max_bytes=item.size,
                    )
                    if (
                        digest_before != item.sha256
                        or size_before != item.size
                    ):
                        raise SnapshotIntegrityError(
                            f"snapshot changed before restore: {item.path}"
                        )
                    source_entry = _SourceEntry(
                        relative=relative,
                        source=source,
                        initial_stat=source.lstat(),
                        kind="sqlite",
                    )
                    self._backup_sqlite(source_entry, target)
                    digest_after, size_after = _hash_file(
                        source,
                        max_bytes=item.size,
                    )
                    if (
                        digest_after != item.sha256
                        or size_after != item.size
                    ):
                        raise SnapshotIntegrityError(
                            f"snapshot changed during restore: {item.path}"
                        )
                else:
                    source_entry = _SourceEntry(
                        relative=relative,
                        source=source,
                        initial_stat=source.lstat(),
                        kind="file",
                    )
                    size, digest = self._copy_regular(source_entry, target)
                    if size != item.size or digest != item.sha256:
                        raise SnapshotIntegrityError(
                            f"snapshot changed during restore: {item.path}"
                        )
                if os.name != "nt":
                    os.chmod(target, item.mode)
            _fsync_tree_directories(staging)
            staging_stat = staging.lstat()
            try:
                _rename_no_replace(staging, destination)
            except FileExistsError as exc:
                raise SnapshotSecurityError(
                    "restore destination appeared during copy"
                ) from exc
            published_identity = (staging_stat.st_dev, staging_stat.st_ino)
            _fsync_directory(parent)
            return RestoreResult(
                snapshot_id=safe_id,
                destination=destination,
                file_count=verified.file_count,
                total_bytes=verified.total_bytes,
            )
        except Exception:
            if published_identity is not None:
                self._rollback_published_tree(
                    destination,
                    owned_parent=parent,
                    expected_identity=published_identity,
                )
            self._cleanup_private_tree(
                staging,
                owned_parent=parent,
                name_prefix=f".{destination.name}.restore.",
            )
            raise

    def _revalidate_roots(self) -> None:
        _assert_real_directory(self.source_root)
        _assert_real_directory(self.backup_root)
        if _overlaps(self.source_root, self.backup_root):
            raise SnapshotSecurityError(
                "source_root and backup_root must remain disjoint"
            )

    def _collect_source_entries(self) -> list[_SourceEntry]:
        raw_entries: list[tuple[PurePosixPath, Path, os.stat_result]] = []
        stack: list[tuple[Path, tuple[str, ...]]] = [(self.source_root, ())]
        scanned_files = 0
        directories_seen = 1

        while stack:
            directory, relative_parts = stack.pop()
            _assert_real_directory(directory)
            try:
                with os.scandir(directory) as iterator:
                    children = sorted(iterator, key=lambda item: item.name)
            except OSError as exc:
                raise SnapshotSecurityError(
                    f"unable to enumerate source directory: {directory}"
                ) from exc
            for child in children:
                relative_tuple = (*relative_parts, child.name)
                relative = PurePosixPath(*relative_tuple)
                _safe_manifest_path(relative.as_posix())
                try:
                    child_stat = child.stat(follow_symlinks=False)
                except OSError as exc:
                    raise SnapshotSecurityError(
                        f"unable to inspect source entry: {relative}"
                    ) from exc
                if stat.S_ISLNK(child_stat.st_mode):
                    raise SnapshotSecurityError(
                        f"source symlink rejected: {relative}"
                    )
                if stat.S_ISDIR(child_stat.st_mode):
                    # Lock bytes have no recovery meaning.  The directory
                    # entry itself still had to be a real directory.
                    if (
                        self._is_coordination_directory(relative)
                        or self._is_runtime_excluded_directory(relative)
                    ):
                        continue
                    directories_seen += 1
                    if directories_seen > self.max_files:
                        raise SnapshotLimitError(
                            "source exceeds directory traversal bound"
                        )
                    stack.append((Path(child.path), relative_tuple))
                    continue
                if not stat.S_ISREG(child_stat.st_mode) or child_stat.st_nlink != 1:
                    raise SnapshotSecurityError(
                        f"non-regular or hard-linked source rejected: {relative}"
                    )
                if stat.S_IMODE(child_stat.st_mode) > 0o777:
                    raise SnapshotSecurityError(
                        f"source file has special permission bits: {relative}"
                    )
                if (
                    self._is_coordination_file(relative)
                    or self._is_runtime_excluded_file(relative)
                ):
                    continue
                scanned_files += 1
                if scanned_files > self.max_files * 4 + 16:
                    raise SnapshotLimitError(
                        "source exceeds bounded raw-entry scan"
                    )
                raw_entries.append(
                    (relative, Path(child.path), child_stat)
                )

        sqlite_paths: set[PurePosixPath] = set()
        for relative, source, source_stat in raw_entries:
            if self._looks_like_sidecar(relative):
                continue
            with _verified_reader(source, expected=source_stat) as (handle, _opened):
                header = handle.read(len(_SQLITE_HEADER))
            suffix_declares_sqlite = source.suffix.lower() in _SQLITE_SUFFIXES
            header_declares_sqlite = header == _SQLITE_HEADER
            if suffix_declares_sqlite and not header_declares_sqlite:
                raise SnapshotIntegrityError(
                    f"database-like file is not a valid SQLite database: {relative}"
                )
            if header_declares_sqlite:
                sqlite_paths.add(relative)

        entries: list[_SourceEntry] = []
        total_source_bytes = 0
        for relative, source, source_stat in raw_entries:
            if self._is_sqlite_sidecar_for(relative, sqlite_paths):
                continue
            if len(entries) >= self.max_files:
                raise SnapshotLimitError("source exceeds max_files")
            if source_stat.st_size > self.max_file_bytes:
                raise SnapshotLimitError(
                    f"source file exceeds max_file_bytes: {relative}"
                )
            total_source_bytes += source_stat.st_size
            if total_source_bytes > self.max_total_bytes:
                raise SnapshotLimitError("source exceeds max_total_bytes")
            entries.append(
                _SourceEntry(
                    relative=relative,
                    source=source,
                    initial_stat=source_stat,
                    kind="sqlite" if relative in sqlite_paths else "file",
                )
            )
        entries.sort(key=lambda item: item.relative.as_posix())
        return entries

    @staticmethod
    def _source_signature(entries: Sequence[_SourceEntry]) -> tuple[tuple, ...]:
        signature = []
        for entry in entries:
            source_stat = entry.initial_stat
            base = (
                entry.relative.as_posix(),
                entry.kind,
                source_stat.st_dev,
                source_stat.st_ino,
            )
            if entry.kind == "sqlite":
                # SQLite's online backup intentionally tolerates in-place WAL
                # commits. Replacement of the database inode remains fatal.
                signature.append(base)
            else:
                signature.append(
                    base
                    + (
                        source_stat.st_size,
                        source_stat.st_mtime_ns,
                        source_stat.st_ctime_ns,
                    )
                )
        return tuple(signature)

    @staticmethod
    def _looks_like_sidecar(relative: PurePosixPath) -> bool:
        return relative.name.endswith(_SQLITE_SIDECAR_SUFFIXES)

    @staticmethod
    def _is_coordination_directory(relative: PurePosixPath) -> bool:
        return relative.name in _COORDINATION_DIRECTORIES

    @staticmethod
    def _is_coordination_file(relative: PurePosixPath) -> bool:
        return relative.name.endswith(_COORDINATION_FILE_SUFFIXES)

    @staticmethod
    def _is_runtime_excluded_directory(relative: PurePosixPath) -> bool:
        return (
            len(relative.parts) == 1
            and relative.name in _EXCLUDED_ROOT_DIRECTORIES
        )

    @staticmethod
    def _is_runtime_excluded_file(relative: PurePosixPath) -> bool:
        return len(relative.parts) == 1 and relative.name in _EXCLUDED_ROOT_FILES

    @staticmethod
    def _is_sqlite_sidecar_for(
        relative: PurePosixPath,
        sqlite_paths: set[PurePosixPath],
    ) -> bool:
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            if relative.name.endswith(suffix):
                base = relative.with_name(relative.name[: -len(suffix)])
                return base in sqlite_paths
        return False

    def _copy_regular(
        self,
        entry: _SourceEntry,
        target: Path,
    ) -> tuple[int, str]:
        if target.exists() or target.is_symlink():
            raise SnapshotSecurityError(f"copy target already exists: {target}")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        digest = hashlib.sha256()
        size = 0
        try:
            with _verified_reader(
                entry.source,
                expected=entry.initial_stat,
            ) as (source, opened):
                descriptor = os.open(
                    target,
                    flags,
                    stat.S_IMODE(opened.st_mode) or 0o600,
                )
                while True:
                    block = source.read(_COPY_CHUNK_SIZE)
                    if not block:
                        break
                    size += len(block)
                    if size > self.max_file_bytes:
                        raise SnapshotLimitError(
                            f"file grew past max_file_bytes: {entry.relative}"
                        )
                    digest.update(block)
                    offset = 0
                    while offset < len(block):
                        written = os.write(descriptor, block[offset:])
                        if written <= 0:
                            raise OSError("short snapshot write")
                        offset += written
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = -1

                after = entry.source.lstat()
                if (
                    after.st_dev != opened.st_dev
                    or after.st_ino != opened.st_ino
                    or after.st_size != opened.st_size
                    or after.st_mtime_ns != opened.st_mtime_ns
                    or after.st_ctime_ns != opened.st_ctime_ns
                ):
                    raise SnapshotIntegrityError(
                        f"ordinary source changed during snapshot: {entry.relative}"
                    )
            return size, digest.hexdigest()
        except Exception:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            target.unlink(missing_ok=True)
            raise

    def _backup_sqlite(
        self,
        entry: _SourceEntry,
        target: Path,
    ) -> tuple[int, str]:
        if target.exists() or target.is_symlink():
            raise SnapshotSecurityError(f"SQLite target already exists: {target}")
        before = entry.source.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_dev != entry.initial_stat.st_dev
            or before.st_ino != entry.initial_stat.st_ino
        ):
            raise SnapshotSecurityError(
                f"SQLite source changed or became unsafe: {entry.relative}"
            )

        source_uri = entry.source.as_uri() + "?mode=ro"
        try:
            with sqlite3.connect(source_uri, uri=True) as source_db:
                source_db.execute("PRAGMA query_only = ON")
                with sqlite3.connect(target) as target_db:
                    source_db.backup(target_db)
                    # ``backup`` copies the source database header, including
                    # WAL journal mode.  A snapshot must be a self-contained
                    # single database file rather than a main file that causes
                    # unmanifested ``-wal``/``-shm`` sidecars to appear when it
                    # is verified or restored.
                    target_db.commit()
                    journal_mode = target_db.execute(
                        "PRAGMA journal_mode = DELETE"
                    ).fetchone()
                    if journal_mode is None or journal_mode[0].lower() != "delete":
                        raise SnapshotIntegrityError(
                            f"SQLite backup could not become self-contained: "
                            f"{entry.relative}"
                        )
                    check = target_db.execute("PRAGMA quick_check").fetchone()
                    if check is None or check[0] != "ok":
                        raise SnapshotIntegrityError(
                            f"SQLite backup failed quick_check: {entry.relative}"
                        )
                    target_db.commit()
            after = entry.source.lstat()
            if (
                stat.S_ISLNK(after.st_mode)
                or not stat.S_ISREG(after.st_mode)
                or after.st_nlink != 1
                or after.st_dev != before.st_dev
                or after.st_ino != before.st_ino
            ):
                raise SnapshotIntegrityError(
                    f"SQLite source was replaced during backup: {entry.relative}"
                )
            with open(target, "rb") as handle:
                os.fsync(handle.fileno())
            if os.name != "nt":
                os.chmod(target, stat.S_IMODE(before.st_mode))
            digest, size = _hash_file(target)
            if size > self.max_file_bytes:
                raise SnapshotLimitError(
                    f"SQLite backup exceeds max_file_bytes: {entry.relative}"
                )
            return size, digest
        except SnapshotError:
            target.unlink(missing_ok=True)
            raise
        except (OSError, sqlite3.Error) as exc:
            target.unlink(missing_ok=True)
            raise SnapshotIntegrityError(
                f"unable to create SQLite backup: {entry.relative}"
            ) from exc

    @staticmethod
    def _verify_sqlite(path: Path) -> None:
        try:
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as database:
                result = database.execute("PRAGMA quick_check").fetchone()
            if result is None or result[0] != "ok":
                raise SnapshotIntegrityError(f"SQLite quick_check failed: {path}")
        except SnapshotError:
            raise
        except sqlite3.Error as exc:
            raise SnapshotIntegrityError(f"invalid SQLite snapshot: {path}") from exc

    def _load_manifest_at(
        self,
        snapshot_root: Path,
        snapshot_id: str,
        *,
        expected_manifest_sha256: str,
    ) -> tuple[Path, tuple[SnapshotFile, ...], str]:
        _assert_real_directory(snapshot_root)
        self._assert_snapshot_root_layout(snapshot_root)
        manifest_path = snapshot_root / MANIFEST_NAME
        with _verified_reader(manifest_path) as (handle, manifest_stat):
            if manifest_stat.st_size > _MAX_MANIFEST_BYTES:
                raise SnapshotLimitError("snapshot manifest is too large")
            payload = handle.read(_MAX_MANIFEST_BYTES + 1)
        manifest_digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(manifest_digest, expected_manifest_sha256):
            raise SnapshotIntegrityError("snapshot manifest digest mismatch")
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError("snapshot manifest is invalid JSON") from exc
        if not isinstance(manifest, dict):
            raise SnapshotIntegrityError("snapshot manifest must be an object")
        required_keys = {
            "schema",
            "snapshot_id",
            "created_at",
            "source_root_sha256",
            "exclusion_policy",
            "file_count",
            "total_bytes",
            "files",
        }
        if set(manifest) != required_keys:
            raise SnapshotIntegrityError("snapshot manifest fields are invalid")
        if (
            manifest["schema"] != MANIFEST_SCHEMA
            or manifest["snapshot_id"] != snapshot_id
        ):
            raise SnapshotIntegrityError("snapshot manifest identity mismatch")
        if (
            not isinstance(manifest["source_root_sha256"], str)
            or not _SHA256_RE.fullmatch(manifest["source_root_sha256"])
        ):
            raise SnapshotIntegrityError("snapshot source identity is invalid")
        expected_source_digest = hashlib.sha256(
            os.fspath(self.source_root).encode("utf-8")
        ).hexdigest()
        if manifest["source_root_sha256"] != expected_source_digest:
            raise SnapshotIntegrityError("snapshot belongs to a different source root")
        if manifest["exclusion_policy"] != _manifest_exclusion_policy():
            raise SnapshotIntegrityError(
                "snapshot exclusion policy is invalid or unsupported"
            )
        if not isinstance(manifest["created_at"], str) or not manifest["created_at"]:
            raise SnapshotIntegrityError("snapshot created_at is invalid")
        try:
            parsed_created = datetime.fromisoformat(
                manifest["created_at"].replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise SnapshotIntegrityError("snapshot created_at is invalid") from exc
        if parsed_created.tzinfo is None or parsed_created.utcoffset() is None:
            raise SnapshotIntegrityError("snapshot created_at must include a timezone")
        raw_files = manifest["files"]
        declared_count = manifest["file_count"]
        declared_bytes = manifest["total_bytes"]
        if (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count < 0
            or declared_count > self.max_files
        ):
            raise SnapshotIntegrityError("snapshot file_count is invalid")
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes < 0
            or declared_bytes > self.max_total_bytes
        ):
            raise SnapshotIntegrityError("snapshot total_bytes is invalid")
        if not isinstance(raw_files, list) or len(raw_files) > self.max_files:
            raise SnapshotLimitError("snapshot manifest exceeds max_files")

        files: list[SnapshotFile] = []
        seen: set[PurePosixPath] = set()
        total = 0
        for raw in raw_files:
            if not isinstance(raw, dict) or set(raw) != {
                "path",
                "kind",
                "size",
                "sha256",
                "mode",
            }:
                raise SnapshotIntegrityError("snapshot file entry is invalid")
            relative = _safe_manifest_path(raw["path"])
            if relative in seen:
                raise SnapshotIntegrityError("snapshot manifest has duplicate paths")
            seen.add(relative)
            kind = raw["kind"]
            if kind not in {"file", "sqlite"}:
                raise SnapshotIntegrityError("snapshot file kind is invalid")
            size = raw["size"]
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or size > self.max_file_bytes
            ):
                raise SnapshotLimitError("snapshot file size is invalid")
            digest = raw["sha256"]
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise SnapshotIntegrityError("snapshot file digest is invalid")
            mode = raw["mode"]
            if (
                isinstance(mode, bool)
                or not isinstance(mode, int)
                or mode < 0
                or mode > 0o777
            ):
                raise SnapshotIntegrityError("snapshot file mode is invalid")
            total += size
            if total > self.max_total_bytes:
                raise SnapshotLimitError("snapshot manifest exceeds max_total_bytes")
            files.append(
                SnapshotFile(
                    path=relative.as_posix(),
                    kind=kind,
                    size=size,
                    sha256=digest,
                    mode=mode,
                )
            )
        if (
            declared_count != len(files)
            or declared_bytes != total
        ):
            raise SnapshotIntegrityError("snapshot manifest accounting mismatch")
        return snapshot_root, tuple(files), manifest_digest

    @staticmethod
    def _assert_snapshot_root_layout(snapshot_root: Path) -> None:
        expected = {
            MANIFEST_NAME: "file",
            FILES_DIR: "directory",
        }
        actual: dict[str, str] = {}
        try:
            with os.scandir(snapshot_root) as iterator:
                children = list(iterator)
        except OSError as exc:
            raise SnapshotSecurityError(
                "unable to enumerate snapshot root"
            ) from exc
        for child in children:
            try:
                child_stat = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise SnapshotSecurityError(
                    "unable to inspect snapshot root entry"
                ) from exc
            if stat.S_ISLNK(child_stat.st_mode):
                raise SnapshotSecurityError(
                    f"snapshot root symlink rejected: {child.name}"
                )
            if stat.S_ISREG(child_stat.st_mode) and child_stat.st_nlink == 1:
                kind = "file"
            elif stat.S_ISDIR(child_stat.st_mode):
                kind = "directory"
            else:
                raise SnapshotSecurityError(
                    f"unsafe snapshot root entry: {child.name}"
                )
            actual[child.name] = kind
        if actual != expected:
            raise SnapshotIntegrityError(
                "snapshot root must contain only manifest.json and files"
            )

    def _enumerate_snapshot_payload(
        self,
        payload_root: Path,
    ) -> set[PurePosixPath]:
        _assert_real_directory(payload_root)
        found: set[PurePosixPath] = set()
        stack: list[tuple[Path, tuple[str, ...]]] = [(payload_root, ())]
        directories_seen = 1
        while stack:
            directory, relative_parts = stack.pop()
            _assert_real_directory(directory)
            for child in os.scandir(directory):
                child_stat = child.stat(follow_symlinks=False)
                relative_tuple = (*relative_parts, child.name)
                relative = PurePosixPath(*relative_tuple)
                if stat.S_ISLNK(child_stat.st_mode):
                    raise SnapshotSecurityError(
                        f"snapshot payload symlink rejected: {relative}"
                    )
                if stat.S_ISDIR(child_stat.st_mode):
                    directories_seen += 1
                    if directories_seen > self.max_files:
                        raise SnapshotLimitError(
                            "snapshot payload exceeds directory traversal bound"
                        )
                    stack.append((Path(child.path), relative_tuple))
                    continue
                if not stat.S_ISREG(child_stat.st_mode) or child_stat.st_nlink != 1:
                    raise SnapshotSecurityError(
                        f"unsafe snapshot payload entry: {relative}"
                    )
                found.add(relative)
                if len(found) > self.max_files:
                    raise SnapshotLimitError("snapshot payload exceeds max_files")
        return found

    @staticmethod
    def _prepare_target_parent(root: Path, parent: Path) -> None:
        if not _is_within(parent, root):
            raise SnapshotSecurityError("target parent escapes staging root")
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        _assert_real_directory(parent)
        if os.name != "nt":
            os.chmod(parent, 0o700)

    def _cleanup_private_tree(
        self,
        path: Path,
        *,
        owned_parent: Path,
        name_prefix: str,
    ) -> None:
        path = _absolute(path)
        owned_parent = _absolute(owned_parent)
        if (
            path.parent != owned_parent
            or not path.name.startswith(name_prefix)
            or not path.name.endswith(".tmp")
        ):
            raise SnapshotSecurityError("refusing to clean an unowned staging path")
        if not path.exists() and not path.is_symlink():
            return
        _assert_real_directory(owned_parent)
        if path.is_symlink():
            path.unlink()
            return
        _assert_real_directory(path)
        shutil.rmtree(path)

    @staticmethod
    def _rollback_published_tree(
        path: Path,
        *,
        owned_parent: Path,
        expected_identity: tuple[int, int],
    ) -> None:
        """Remove only the directory instance published by this operation."""

        path = _absolute(path)
        owned_parent = _absolute(owned_parent)
        if path.parent != owned_parent:
            raise SnapshotSecurityError(
                "refusing to roll back an unowned published path"
            )
        _assert_real_directory(owned_parent)
        try:
            published_stat = path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise SnapshotSecurityError(
                "unable to inspect published path during rollback"
            ) from exc
        if (
            stat.S_ISLNK(published_stat.st_mode)
            or not stat.S_ISDIR(published_stat.st_mode)
            or (published_stat.st_dev, published_stat.st_ino)
            != expected_identity
        ):
            raise SnapshotSecurityError(
                "published path identity changed before rollback"
            )
        try:
            shutil.rmtree(path)
            _fsync_directory(owned_parent)
        except OSError as exc:
            raise SnapshotSecurityError(
                "unable to roll back failed publication"
            ) from exc
