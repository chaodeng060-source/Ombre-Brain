"""Crash-recoverable named approval for queued relation candidates.

Machine inference may only enqueue relation proposals.  This module is the
single path that turns a dangerous pending proposal into a graph edge after a
named reviewer approves it.  A small filesystem journal bridges the Markdown
bucket and JSONL review ledger so retries are idempotent and interrupted writes
can be deterministically completed or rolled back.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import frontmatter

from maintenance_barrier import MaintenanceBarrier
from review_queue import (
    KIND_RELATION,
    STATUS_APPLIED,
    STATUS_PENDING,
    make_relation_entry,
)
from storage_safety import atomic_write_text
from utils import REVIEW_RELATION_TYPES, now_iso


class RelationApprovalError(RuntimeError):
    """Base class for a failed relation approval."""


class RelationApprovalNotFound(RelationApprovalError):
    """The queue row or one of its endpoint buckets no longer exists."""


class RelationApprovalStateError(RelationApprovalError):
    """The proposed relation is not in an approvable state."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reviewer(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("reviewer is required")
    return normalized[:120]


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

    def _transaction_dir(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.journal_root / digest

    def _manifest_path(self, transaction_dir: Path) -> Path:
        return transaction_dir / "manifest.json"

    def _write_manifest(self, transaction_dir: Path, manifest: dict) -> None:
        atomic_write_text(
            self._manifest_path(transaction_dir),
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
        )

    def _load_manifest(self, transaction_dir: Path) -> dict:
        try:
            value = json.loads(
                self._manifest_path(transaction_dir).read_text(encoding="utf-8")
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

    def _safe_bucket_path(self, path: str | os.PathLike) -> Path:
        target = Path(os.path.abspath(os.fspath(path)))
        try:
            if os.path.commonpath([self.root, target]) != os.fspath(self.root):
                raise RelationApprovalStateError(
                    "relation bucket path escapes the configured vault"
                )
            info = target.lstat()
        except RelationApprovalStateError:
            raise
        except OSError as exc:
            raise RelationApprovalNotFound("relation bucket is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RelationApprovalStateError(
                "relation bucket path must be a regular file"
            )
        return target

    def _path_from_manifest(self, relative: str) -> Path:
        if not relative or os.path.isabs(relative):
            raise RelationApprovalStateError(
                "relation transaction contains an invalid bucket path"
            )
        return self._safe_bucket_path(self.root / relative)

    @staticmethod
    def _journal_text(transaction_dir: Path, name: str) -> str:
        try:
            return (transaction_dir / name).read_text(encoding="utf-8")
        except OSError as exc:
            raise RelationApprovalStateError(
                "relation transaction payload is unavailable"
            ) from exc

    def _write_target(self, transaction_dir: Path, manifest: dict) -> None:
        target = self._path_from_manifest(manifest["source_path"])
        payload = self._journal_text(transaction_dir, "source.target.md")
        if _sha256(payload) != manifest["target_sha256"]:
            raise RelationApprovalStateError(
                "relation transaction target payload hash mismatch"
            )
        atomic_write_text(target, payload)

    def _restore_original(self, transaction_dir: Path, manifest: dict) -> None:
        target = self._path_from_manifest(manifest["source_path"])
        payload = self._journal_text(transaction_dir, "source.original.md")
        if _sha256(payload) != manifest["original_sha256"]:
            raise RelationApprovalStateError(
                "relation transaction original payload hash mismatch"
            )
        atomic_write_text(target, payload)
        manifest["state"] = "rolled_back"
        self._write_manifest(transaction_dir, manifest)

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
            raise RelationApprovalStateError("relation candidate key is inconsistent")
        return source_id, target_id, rel_type, note

    def _recover_locked(self) -> list[str]:
        if not self.journal_root.exists():
            return []
        rows = self.review_queue.all()
        recovered: list[str] = []
        for transaction_dir in sorted(self.journal_root.iterdir()):
            if not transaction_dir.is_dir():
                continue
            manifest = self._load_manifest(transaction_dir)
            if manifest.get("state") in {"committed", "rolled_back"}:
                continue
            entry = self._entry(rows, manifest["queue_key"])
            if entry and entry.get("status") == STATUS_APPLIED:
                self._write_target(transaction_dir, manifest)
                manifest["state"] = "committed"
                self._write_manifest(transaction_dir, manifest)
            else:
                self._restore_original(transaction_dir, manifest)
            recovered.append(manifest["queue_key"])
        return recovered

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
            transaction_dir = self._transaction_dir(key)

            if entry.get("status") == STATUS_APPLIED:
                if not self._manifest_path(transaction_dir).exists():
                    raise RelationApprovalStateError(
                        "applied relation has no durable transaction journal"
                    )
                manifest = self._load_manifest(transaction_dir)
                if manifest.get("state") != "committed":
                    raise RelationApprovalStateError(
                        "applied relation transaction is not committed"
                    )
                if any(
                    str(manifest.get(field) or "") != expected
                    for field, expected in (
                        ("queue_key", key),
                        ("source_id", source_id),
                        ("target_id", target_id),
                        ("rel_type", rel_type),
                    )
                ):
                    raise RelationApprovalStateError(
                        "applied relation journal does not match the queue row"
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
            source_path = self._safe_bucket_path(source_raw)
            self._safe_bucket_path(target_raw)
            source_post = self.bucket_manager._safe_load_post(os.fspath(source_path))
            relations = list(source_post.get("relations") or [])
            exists = any(
                isinstance(relation, dict)
                and relation.get("type") == rel_type
                and relation.get("target") == target_id
                for relation in relations
            )
            target_post = copy.deepcopy(source_post)
            if not exists:
                edge = {"type": rel_type, "target": target_id}
                if note:
                    edge["note"] = note
                relations.append(edge)
                target_post["relations"] = relations
                target_post["last_active"] = now_iso()

            original = source_path.read_text(encoding="utf-8")
            target = frontmatter.dumps(target_post)
            transaction_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            if os.name != "nt":
                os.chmod(transaction_dir, 0o700)
            atomic_write_text(transaction_dir / "source.original.md", original)
            atomic_write_text(transaction_dir / "source.target.md", target)
            manifest = {
                "version": 1,
                "queue_key": key,
                "state": "prepared",
                "reviewer": reviewer,
                "verdict_note": str(verdict_note or "").strip()[:500],
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": rel_type,
                "source_path": os.path.relpath(source_path, self.root),
                "original_sha256": _sha256(original),
                "target_sha256": _sha256(target),
                "memory_changed": not exists,
            }
            self._write_manifest(transaction_dir, manifest)

            try:
                self._write_target(transaction_dir, manifest)
                manifest["state"] = "bucket_written"
                self._write_manifest(transaction_dir, manifest)
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
                    self._write_target(transaction_dir, manifest)
                    manifest["state"] = "committed"
                    self._write_manifest(transaction_dir, manifest)
                else:
                    self._restore_original(transaction_dir, manifest)
                raise

            manifest["state"] = "committed"
            self._write_manifest(transaction_dir, manifest)
            return {
                "key": key,
                "status": STATUS_APPLIED,
                "changed": not exists,
                "queue_changed": True,
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": rel_type,
            }
