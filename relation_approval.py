"""Crash-recoverable named approval for queued relation candidates.

Machine inference may only enqueue relation proposals.  This module is the
single path that turns a dangerous pending proposal into a graph edge after a
named reviewer approves it.  The bucket and transaction journal are accessed
through pinned directory descriptors: no symlinked ancestor is followed, and
every bucket replacement is conditional on the inode/content revision that was
prepared.  That keeps retries idempotent without letting an interrupted or
concurrent writer escape the configured vault.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path, PurePath

import frontmatter

from maintenance_barrier import MaintenanceBarrier
from review_queue import (
    KIND_RELATION,
    STATUS_APPLIED,
    STATUS_PENDING,
    make_relation_entry,
)
from utils import REVIEW_RELATION_TYPES, now_iso


class RelationApprovalError(RuntimeError):
    """Base class for a failed relation approval."""


class RelationApprovalNotFound(RelationApprovalError):
    """The queue row or one of its endpoint buckets no longer exists."""


class RelationApprovalStateError(RelationApprovalError):
    """The proposed relation is not in an approvable state."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reviewer(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("reviewer is required")
    return normalized[:120]


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if (
        not nofollow
        or not directory
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
    ):
        raise RelationApprovalStateError(
            "secure relation directory access is unsupported"
        )
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow | directory


def _open_absolute_directory(path: Path) -> int:
    """Open every absolute component without following a symlink."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = _directory_flags()
    try:
        descriptor = os.open(absolute.anchor or os.sep, flags)
    except OSError as exc:
        raise RelationApprovalStateError(
            "relation directory anchor is unavailable"
        ) from exc
    try:
        for component in absolute.parts[1:]:
            try:
                expected = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise RelationApprovalStateError(
                    "relation directory ancestor is unavailable"
                ) from exc
            if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(
                expected.st_mode
            ):
                raise RelationApprovalStateError(
                    "relation directory ancestor is unsafe"
                )
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise RelationApprovalStateError(
                    "relation directory ancestor is unsafe"
                ) from exc
            try:
                opened = os.fstat(child)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (expected.st_dev, expected.st_ino)
                ):
                    raise RelationApprovalStateError(
                        "relation directory ancestor identity changed"
                    )
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_child_directory(
    parent_fd: int,
    name: str,
    *,
    create: bool,
    private: bool = False,
) -> int:
    if not name or name in {".", ".."} or os.sep in name:
        raise RelationApprovalStateError("relation directory name is invalid")
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise RelationApprovalStateError(
                "unable to create relation transaction directory"
            ) from exc
    try:
        expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise RelationApprovalStateError(
            "relation transaction directory is unavailable"
        ) from exc
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(expected.st_mode):
        raise RelationApprovalStateError(
            "relation transaction directory must be real"
        )
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise RelationApprovalStateError(
            "relation transaction directory must be real"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (expected.st_dev, expected.st_ino)
        ):
            raise RelationApprovalStateError(
                "relation transaction directory identity changed"
            )
        if private and os.name != "nt":
            os.fchmod(descriptor, 0o700)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _relative_parts(relative: str) -> tuple[str, ...]:
    path = PurePath(relative)
    parts = path.parts
    if (
        not relative
        or path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RelationApprovalStateError(
            "relation transaction contains an invalid bucket path"
        )
    return tuple(parts)


def _open_relative_parent(root_fd: int, relative: str) -> tuple[int, str]:
    parts = _relative_parts(relative)
    descriptor = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            child = _open_child_directory(
                descriptor,
                component,
                create=False,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except Exception:
        os.close(descriptor)
        raise


def _revision(info: os.stat_result, payload: bytes) -> dict:
    return {
        "dev": int(info.st_dev),
        "ino": int(info.st_ino),
        "nlink": int(info.st_nlink),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
        "mode": int(stat.S_IMODE(info.st_mode)),
        "sha256": _sha256_bytes(payload),
    }


def _same_revision(left: dict, right: dict) -> bool:
    fields = (
        "dev",
        "ino",
        "nlink",
        "size",
        "mtime_ns",
        "ctime_ns",
        "sha256",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def _read_regular_at(directory_fd: int, name: str) -> tuple[bytes, dict]:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise RelationApprovalNotFound("relation bucket is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise RelationApprovalStateError(
            "relation bucket path must be a singly-linked regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RelationApprovalStateError("relation bucket open failed") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise RelationApprovalStateError(
                "relation bucket identity changed during open"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        initial = _revision(opened, payload)
        final = _revision(after, payload)
        if not _same_revision(initial, final):
            raise RelationApprovalStateError(
                "relation bucket changed during read"
            )
        return payload, final
    finally:
        os.close(descriptor)


def _read_private_file(directory_fd: int, name: str) -> bytes:
    try:
        payload, _ = _read_regular_at(directory_fd, name)
        return payload
    except RelationApprovalNotFound as exc:
        raise RelationApprovalStateError(
            "relation transaction payload is unavailable"
        ) from exc


def _atomic_replace_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    expected_revision: dict,
) -> dict:
    current_payload, current = _read_regular_at(directory_fd, name)
    if _sha256_bytes(current_payload) == _sha256_bytes(payload):
        return current
    if not _same_revision(current, expected_revision):
        raise RelationApprovalStateError(
            "relation bucket changed after approval was prepared"
        )

    temporary = f".{name}.relation-{secrets.token_hex(8)}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            flags,
            int(expected_revision.get("mode", 0o600)),
            dir_fd=directory_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short relation bucket write")
            view = view[written:]
        os.fsync(descriptor)
        if os.name != "nt":
            os.fchmod(descriptor, int(expected_revision.get("mode", 0o600)))
        os.close(descriptor)
        descriptor = -1

        # A cooperative writer cannot enter while the exclusive maintenance
        # lease is held.  Rechecking immediately before replace also rejects a
        # non-cooperative change instead of silently clobbering it.
        _, before_replace = _read_regular_at(directory_fd, name)
        if not _same_revision(before_replace, expected_revision):
            raise RelationApprovalStateError(
                "relation bucket changed before atomic replace"
            )
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        _, replaced = _read_regular_at(directory_fd, name)
        if replaced["sha256"] != _sha256_bytes(payload):
            raise RelationApprovalStateError(
                "relation bucket replacement verification failed"
            )
        return replaced
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _write_private_file(directory_fd: int, name: str, payload: bytes) -> None:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        info = None
    except OSError as exc:
        raise RelationApprovalStateError(
            "relation transaction file is unavailable"
        ) from exc
    if info is not None and (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise RelationApprovalStateError(
            "relation transaction file must be a singly-linked regular file"
        )
    temporary = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short relation journal write")
            view = view[written:]
        os.fsync(descriptor)
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except OSError as exc:
        raise RelationApprovalStateError(
            "unable to persist relation transaction"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


class RelationApprovalTransaction:
    """Atomically apply one dangerous queued edge with a durable verdict."""

    JOURNAL_DIR = ".relation-approval-transactions"

    def __init__(self, buckets_dir, bucket_manager, review_queue):
        self.root = Path(os.path.abspath(os.fspath(buckets_dir)))
        self.bucket_manager = bucket_manager
        self.review_queue = review_queue
        self.journal_root = self.root / self.JOURNAL_DIR
        self._barrier = MaintenanceBarrier(self.root)

    @staticmethod
    def _entry(rows: list[dict], key: str) -> dict | None:
        return next((row for row in rows if row.get("key") == key), None)

    @staticmethod
    def _transaction_name(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

    def _open_root(self) -> int:
        return _open_absolute_directory(self.root)

    def _open_journal_root(self, root_fd: int, *, create: bool) -> int:
        return _open_child_directory(
            root_fd,
            self.JOURNAL_DIR,
            create=create,
            private=True,
        )

    def _open_transaction(
        self,
        journal_fd: int,
        key: str,
        *,
        create: bool,
    ) -> int:
        return _open_child_directory(
            journal_fd,
            self._transaction_name(key),
            create=create,
            private=True,
        )

    @staticmethod
    def _write_manifest(transaction_fd: int, manifest: dict) -> None:
        _write_private_file(
            transaction_fd,
            "manifest.json",
            (
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )

    @staticmethod
    def _load_manifest(transaction_fd: int) -> dict:
        try:
            value = json.loads(
                _read_private_file(transaction_fd, "manifest.json").decode(
                    "utf-8"
                )
            )
        except Exception as exc:
            raise RelationApprovalStateError(
                "relation transaction journal is corrupt"
            ) from exc
        if not isinstance(value, dict) or not value.get("queue_key"):
            raise RelationApprovalStateError(
                "relation transaction manifest is invalid"
            )
        return value

    def _read_bucket(self, root_fd: int, relative: str) -> tuple[bytes, dict]:
        directory_fd, name = _open_relative_parent(root_fd, relative)
        try:
            return _read_regular_at(directory_fd, name)
        finally:
            os.close(directory_fd)

    def _replace_bucket(
        self,
        root_fd: int,
        relative: str,
        payload: bytes,
        expected_revision: dict,
    ) -> dict:
        directory_fd, name = _open_relative_parent(root_fd, relative)
        try:
            return _atomic_replace_at(
                directory_fd,
                name,
                payload,
                expected_revision=expected_revision,
            )
        finally:
            os.close(directory_fd)

    @staticmethod
    def _journal_payload(transaction_fd: int, name: str, digest: str) -> bytes:
        payload = _read_private_file(transaction_fd, name)
        if _sha256_bytes(payload) != digest:
            raise RelationApprovalStateError(
                "relation transaction payload hash mismatch"
            )
        return payload

    def _write_target(
        self,
        root_fd: int,
        transaction_fd: int,
        manifest: dict,
    ) -> dict:
        payload = self._journal_payload(
            transaction_fd,
            "source.target.md",
            manifest["target_sha256"],
        )
        current, current_revision = self._read_bucket(
            root_fd,
            manifest["source_path"],
        )
        if _sha256_bytes(current) == manifest["target_sha256"]:
            return current_revision
        expected = manifest.get("original_revision")
        if not isinstance(expected, dict) or not _same_revision(
            current_revision,
            expected,
        ):
            raise RelationApprovalStateError(
                "relation bucket revision no longer matches approval"
            )
        return self._replace_bucket(
            root_fd,
            manifest["source_path"],
            payload,
            expected,
        )

    def _restore_original(
        self,
        root_fd: int,
        transaction_fd: int,
        manifest: dict,
    ) -> None:
        payload = self._journal_payload(
            transaction_fd,
            "source.original.md",
            manifest["original_sha256"],
        )
        current, revision = self._read_bucket(
            root_fd,
            manifest["source_path"],
        )
        digest = _sha256_bytes(current)
        if digest != manifest["original_sha256"]:
            if digest != manifest["target_sha256"]:
                raise RelationApprovalStateError(
                    "relation rollback refused a concurrent bucket revision"
                )
            self._replace_bucket(
                root_fd,
                manifest["source_path"],
                payload,
                revision,
            )
        manifest["state"] = "rolled_back"
        self._write_manifest(transaction_fd, manifest)

    @staticmethod
    def _validate_entry(entry: dict) -> tuple[str, str, str, str]:
        if entry.get("kind") != KIND_RELATION:
            raise RelationApprovalStateError(
                "review item is not a relation candidate"
            )
        source_id = str(entry.get("source_id") or "").strip()
        target_id = str(entry.get("target_id") or "").strip()
        rel_type = str(entry.get("rel_type") or "").strip()
        note = str(entry.get("note") or "").strip()[:500]
        if not source_id or not target_id or source_id == target_id:
            raise RelationApprovalStateError("relation endpoints are invalid")
        if rel_type not in REVIEW_RELATION_TYPES:
            raise RelationApprovalStateError(
                "only dangerous relation types use named approval"
            )
        expected_key = make_relation_entry(source_id, target_id, rel_type)["key"]
        if str(entry.get("key") or "") != expected_key:
            raise RelationApprovalStateError(
                "relation candidate key is inconsistent"
            )
        return source_id, target_id, rel_type, note

    @classmethod
    def _validate_manifest_entry(cls, manifest: dict, entry: dict) -> None:
        source_id, target_id, rel_type, _ = cls._validate_entry(entry)
        if any(
            str(manifest.get(field) or "") != expected
            for field, expected in (
                ("queue_key", str(entry.get("key") or "")),
                ("source_id", source_id),
                ("target_id", target_id),
                ("rel_type", rel_type),
            )
        ):
            raise RelationApprovalStateError(
                "relation journal does not match the queue row"
            )

    def _recover_locked(self) -> list[str]:
        root_fd = self._open_root()
        try:
            try:
                journal_fd = self._open_journal_root(root_fd, create=False)
            except RelationApprovalStateError:
                try:
                    info = os.stat(
                        self.JOURNAL_DIR,
                        dir_fd=root_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    return []
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise
                raise
            try:
                rows = self.review_queue.all()
                recovered: list[str] = []
                for name in sorted(os.listdir(journal_fd)):
                    try:
                        info = os.stat(
                            name,
                            dir_fd=journal_fd,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise RelationApprovalStateError(
                            "relation transaction entry is unavailable"
                        ) from exc
                    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(
                        info.st_mode
                    ):
                        raise RelationApprovalStateError(
                            "relation transaction entry is unsafe"
                        )
                    transaction_fd = _open_child_directory(
                        journal_fd,
                        name,
                        create=False,
                        private=True,
                    )
                    try:
                        manifest = self._load_manifest(transaction_fd)
                        if manifest.get("state") in {
                            "committed",
                            "rolled_back",
                        }:
                            continue
                        entry = self._entry(rows, manifest["queue_key"])
                        if entry is not None:
                            self._validate_manifest_entry(manifest, entry)
                        if entry and entry.get("status") == STATUS_APPLIED:
                            revision = self._write_target(
                                root_fd,
                                transaction_fd,
                                manifest,
                            )
                            manifest["target_revision"] = revision
                            manifest["state"] = "committed"
                            self._write_manifest(transaction_fd, manifest)
                        else:
                            self._restore_original(
                                root_fd,
                                transaction_fd,
                                manifest,
                            )
                        recovered.append(manifest["queue_key"])
                    finally:
                        os.close(transaction_fd)
                return recovered
            finally:
                os.close(journal_fd)
        finally:
            os.close(root_fd)

    def recover(self) -> list[str]:
        with self._barrier.exclusive():
            return self._recover_locked()

    def apply(
        self,
        key: str,
        *,
        reviewer: str,
        verdict_note: str = "",
    ) -> dict:
        key = str(key or "").strip()
        reviewer = _reviewer(reviewer)
        if not key:
            raise ValueError("key is required")

        with self._barrier.exclusive():
            self._recover_locked()
            entry = self._entry(self.review_queue.all(), key)
            if entry is None:
                raise RelationApprovalNotFound("pending review item not found")
            source_id, target_id, rel_type, note = self._validate_entry(entry)

            root_fd = self._open_root()
            journal_fd = -1
            transaction_fd = -1
            try:
                journal_fd = self._open_journal_root(root_fd, create=True)
                if entry.get("status") == STATUS_APPLIED:
                    transaction_fd = self._open_transaction(
                        journal_fd,
                        key,
                        create=False,
                    )
                    manifest = self._load_manifest(transaction_fd)
                    self._validate_manifest_entry(manifest, entry)
                    if manifest.get("state") != "committed":
                        raise RelationApprovalStateError(
                            "applied relation transaction is not committed"
                        )
                    return {
                        "key": key,
                        "status": STATUS_APPLIED,
                        "changed": False,
                        "queue_changed": False,
                        "source_id": source_id,
                        "target_id": target_id,
                        "rel_type": rel_type,
                    }
                if entry.get("status") != STATUS_PENDING:
                    raise RelationApprovalStateError(
                        f"review item is already {entry.get('status') or 'resolved'}"
                    )

                source_raw = self.bucket_manager._find_bucket_file(source_id)
                target_raw = self.bucket_manager._find_bucket_file(target_id)
                if not source_raw or not target_raw:
                    raise RelationApprovalNotFound(
                        "both relation endpoint buckets must still exist"
                    )
                try:
                    source_relative = os.path.relpath(source_raw, self.root)
                    target_relative = os.path.relpath(target_raw, self.root)
                except ValueError as exc:
                    raise RelationApprovalStateError(
                        "relation bucket path escapes the configured vault"
                    ) from exc
                _relative_parts(source_relative)
                _relative_parts(target_relative)

                original_bytes, original_revision = self._read_bucket(
                    root_fd,
                    source_relative,
                )
                self._read_bucket(root_fd, target_relative)
                try:
                    original = original_bytes.decode("utf-8", errors="strict")
                    source_post = frontmatter.loads(original)
                except Exception as exc:
                    raise RelationApprovalStateError(
                        "relation source bucket is invalid"
                    ) from exc
                relations = list(source_post.get("relations") or [])
                exists = any(
                    isinstance(relation, dict)
                    and relation.get("type") == rel_type
                    and relation.get("target") == target_id
                    for relation in relations
                )
                if exists:
                    target_bytes = original_bytes
                else:
                    target_post = copy.deepcopy(source_post)
                    edge = {"type": rel_type, "target": target_id}
                    if note:
                        edge["note"] = note
                    relations.append(edge)
                    target_post["relations"] = relations
                    target_post["last_active"] = now_iso()
                    target_bytes = frontmatter.dumps(target_post).encode("utf-8")

                transaction_fd = self._open_transaction(
                    journal_fd,
                    key,
                    create=True,
                )
                _write_private_file(
                    transaction_fd,
                    "source.original.md",
                    original_bytes,
                )
                _write_private_file(
                    transaction_fd,
                    "source.target.md",
                    target_bytes,
                )
                manifest = {
                    "version": 2,
                    "queue_key": key,
                    "state": "prepared",
                    "reviewer": reviewer,
                    "verdict_note": str(verdict_note or "").strip()[:500],
                    "source_id": source_id,
                    "target_id": target_id,
                    "rel_type": rel_type,
                    "source_path": source_relative,
                    "original_sha256": _sha256_bytes(original_bytes),
                    "target_sha256": _sha256_bytes(target_bytes),
                    "original_revision": original_revision,
                    "memory_changed": not exists,
                }
                self._write_manifest(transaction_fd, manifest)

                try:
                    target_revision = self._write_target(
                        root_fd,
                        transaction_fd,
                        manifest,
                    )
                    manifest["target_revision"] = target_revision
                    manifest["state"] = "bucket_written"
                    self._write_manifest(transaction_fd, manifest)
                    changed = self.review_queue.apply_relation(
                        key,
                        reviewer=reviewer,
                        verdict_note=manifest["verdict_note"],
                    )
                    if not changed:
                        raise RelationApprovalStateError(
                            "pending review item changed during approval"
                        )
                except Exception:
                    durable = self.review_queue.get(key)
                    if durable and durable.get("status") == STATUS_APPLIED:
                        revision = self._write_target(
                            root_fd,
                            transaction_fd,
                            manifest,
                        )
                        manifest["target_revision"] = revision
                        manifest["state"] = "committed"
                        self._write_manifest(transaction_fd, manifest)
                    else:
                        self._restore_original(
                            root_fd,
                            transaction_fd,
                            manifest,
                        )
                    raise

                manifest["state"] = "committed"
                self._write_manifest(transaction_fd, manifest)
                return {
                    "key": key,
                    "status": STATUS_APPLIED,
                    "changed": not exists,
                    "queue_changed": True,
                    "source_id": source_id,
                    "target_id": target_id,
                    "rel_type": rel_type,
                }
            finally:
                if transaction_fd >= 0:
                    os.close(transaction_fd)
                if journal_fd >= 0:
                    os.close(journal_fd)
                os.close(root_fd)
