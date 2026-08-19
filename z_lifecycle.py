"""Crash-recoverable approval transaction for Z-axis fact lifecycle changes.

Discovery and ``apply`` only create review candidates.  This module is the
single path that may turn an explicitly approved pair into
``current``/``historical`` bucket metadata.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Callable, Optional

import frontmatter

from maintenance_barrier import MaintenanceBarrier
from review_queue import STATUS_APPLIED, STATUS_PENDING, lifecycle_updates
from storage_safety import atomic_write_text


class ZLifecycleError(RuntimeError):
    """Base class for lifecycle transaction failures."""


class ZLifecycleNotFound(ZLifecycleError):
    """The requested queue entry or one of its buckets no longer exists."""


class ZLifecycleStateError(ZLifecycleError):
    """The queue entry or durable transaction is not in an applicable state."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_reviewer(value: str) -> str:
    reviewer = str(value or "").strip()
    if not reviewer:
        raise ValueError("reviewer is required")
    return reviewer[:120]


class ZLifecycleTransaction:
    """Apply one approved Z pair with a durable rollback/recovery journal.

    Markdown cannot atomically replace two bucket files and the JSONL review
    ledger in one syscall.  The journal therefore stores both original and
    target bytes before the first replacement.  Recovery rolls incomplete
    pending decisions back, or finishes the target state when the queue row was
    already durably marked ``applied``.
    """

    JOURNAL_DIR = ".z-lifecycle-transactions"

    def __init__(self, buckets_dir, bucket_manager, review_queue):
        self.root = Path(os.path.abspath(os.fspath(buckets_dir)))
        self.bucket_manager = bucket_manager
        self.review_queue = review_queue
        self.journal_root = self.root / self.JOURNAL_DIR
        self._barrier = MaintenanceBarrier(self.root)

    @staticmethod
    def _entry(rows: list[dict], key: str) -> Optional[dict]:
        return next((row for row in rows if row.get("key") == key), None)

    def _transaction_dir(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.journal_root / digest

    def _safe_bucket_path(self, path: str | os.PathLike) -> Path:
        target = Path(os.path.abspath(os.fspath(path)))
        try:
            if os.path.commonpath([self.root, target]) != os.fspath(self.root):
                raise ZLifecycleStateError("bucket path escapes the configured vault")
            info = target.lstat()
        except ZLifecycleStateError:
            raise
        except OSError as exc:
            raise ZLifecycleNotFound("bucket file is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ZLifecycleStateError("bucket path must be a regular file")
        return target

    def _manifest_path(self, transaction_dir: Path) -> Path:
        return transaction_dir / "manifest.json"

    def _write_manifest(self, transaction_dir: Path, manifest: dict) -> None:
        atomic_write_text(
            self._manifest_path(transaction_dir),
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

    def _load_manifest(self, transaction_dir: Path) -> dict:
        try:
            value = json.loads(
                self._manifest_path(transaction_dir).read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ZLifecycleStateError("Z transaction journal is corrupt") from exc
        if not isinstance(value, dict) or not value.get("queue_key"):
            raise ZLifecycleStateError("Z transaction manifest is invalid")
        return value

    def _journal_text(self, transaction_dir: Path, name: str) -> str:
        try:
            return (transaction_dir / name).read_text(encoding="utf-8")
        except OSError as exc:
            raise ZLifecycleStateError("Z transaction payload is unavailable") from exc

    def _path_from_manifest(self, relative: str) -> Path:
        if not relative or os.path.isabs(relative):
            raise ZLifecycleStateError("Z transaction contains an invalid bucket path")
        target = Path(os.path.abspath(self.root / relative))
        if os.path.commonpath([self.root, target]) != os.fspath(self.root):
            raise ZLifecycleStateError("Z transaction path escapes the configured vault")
        return target

    def _write_bucket(self, path: Path, text: str) -> None:
        atomic_write_text(path, text)

    def _restore_originals(self, transaction_dir: Path, manifest: dict) -> None:
        for side in ("current", "historical"):
            target = self._path_from_manifest(manifest[f"{side}_path"])
            original = self._journal_text(transaction_dir, f"{side}.original.md")
            if _sha256(original) != manifest[f"{side}_original_sha256"]:
                raise ZLifecycleStateError("Z transaction original payload hash mismatch")
            # Rollback must not reuse an injected/failed target writer.
            atomic_write_text(target, original)
        manifest["state"] = "rolled_back"
        self._write_manifest(transaction_dir, manifest)

    def _write_targets(self, transaction_dir: Path, manifest: dict) -> None:
        for side in ("current", "historical"):
            target = self._path_from_manifest(manifest[f"{side}_path"])
            payload = self._journal_text(transaction_dir, f"{side}.target.md")
            if _sha256(payload) != manifest[f"{side}_target_sha256"]:
                raise ZLifecycleStateError("Z transaction target payload hash mismatch")
            self._write_bucket(target, payload)

    def _recover_locked(self) -> list[str]:
        if not self.journal_root.exists():
            return []
        recovered: list[str] = []
        rows = self.review_queue.all()
        for transaction_dir in sorted(self.journal_root.iterdir()):
            if not transaction_dir.is_dir():
                continue
            manifest = self._load_manifest(transaction_dir)
            state = str(manifest.get("state") or "")
            if state in {"committed", "rolled_back"}:
                continue
            entry = self._entry(rows, manifest["queue_key"])
            if entry and entry.get("status") == STATUS_APPLIED:
                self._write_targets(transaction_dir, manifest)
                manifest["state"] = "committed"
                self._write_manifest(transaction_dir, manifest)
            else:
                self._restore_originals(transaction_dir, manifest)
            recovered.append(manifest["queue_key"])
        return recovered

    def recover(self) -> list[str]:
        """Repair every interrupted transaction before serving writes."""
        with self._barrier.exclusive():
            return self._recover_locked()

    def apply(
        self,
        key: str,
        *,
        reviewer: str,
        verdict_note: str = "",
        validate_pair: Optional[Callable[[dict, dict, str], str]] = None,
    ) -> dict:
        """Apply one pending pair, or return an exact committed replay."""
        key = str(key or "").strip()
        reviewer = _safe_reviewer(reviewer)
        if not key:
            raise ValueError("key is required")

        with self._barrier.exclusive():
            self._recover_locked()
            rows = self.review_queue.all()
            entry = self._entry(rows, key)
            if entry is None:
                raise ZLifecycleNotFound("pending review item not found")
            if entry.get("candidate_type") != "cross_bucket_lifecycle":
                raise ZLifecycleStateError("review item is not a Z lifecycle pair")

            transaction_dir = self._transaction_dir(key)
            if entry.get("status") == STATUS_APPLIED:
                if not self._manifest_path(transaction_dir).exists():
                    raise ZLifecycleStateError(
                        "applied review item has no durable transaction journal"
                    )
                manifest = self._load_manifest(transaction_dir)
                if manifest.get("state") != "committed":
                    raise ZLifecycleStateError("applied Z transaction is not committed")
                return {
                    "key": key,
                    "status": STATUS_APPLIED,
                    "changed": False,
                    "current_bucket_id": entry["current_bucket_id"],
                    "historical_bucket_id": entry["historical_bucket_id"],
                }
            if entry.get("status") != STATUS_PENDING:
                raise ZLifecycleStateError(
                    f"review item is already {entry.get('status') or 'resolved'}"
                )

            current_id = str(entry.get("current_bucket_id") or "").strip()
            historical_id = str(entry.get("historical_bucket_id") or "").strip()
            fact_key = str(entry.get("fact_key") or "").strip().lower()
            current_path = self.bucket_manager._find_bucket_file(current_id)
            historical_path = self.bucket_manager._find_bucket_file(historical_id)
            if not current_path or not historical_path:
                raise ZLifecycleNotFound("both lifecycle buckets must still exist")
            current_path = self._safe_bucket_path(current_path)
            historical_path = self._safe_bucket_path(historical_path)
            if current_path == historical_path:
                raise ZLifecycleStateError("lifecycle endpoints must be distinct files")

            current_post = self.bucket_manager._safe_load_post(os.fspath(current_path))
            historical_post = self.bucket_manager._safe_load_post(
                os.fspath(historical_path)
            )
            current = {
                "id": current_id,
                "metadata": dict(current_post.metadata),
                "content": current_post.content,
                "path": os.fspath(current_path),
            }
            historical = {
                "id": historical_id,
                "metadata": dict(historical_post.metadata),
                "content": historical_post.content,
                "path": os.fspath(historical_path),
            }
            if validate_pair:
                validation_error = validate_pair(current, historical, fact_key)
                if validation_error:
                    raise ZLifecycleStateError(validation_error)
            current_status = str(
                current["metadata"].get("fact_status") or "current"
            ).strip().lower()
            historical_status = str(
                historical["metadata"].get("fact_status") or "current"
            ).strip().lower()
            if current_status == "historical":
                raise ZLifecycleStateError(
                    "chosen current bucket is already historical"
                )
            if historical_status == "historical":
                prior_current = str(
                    historical["metadata"].get("superseded_by_bucket_id") or ""
                ).strip()
                if prior_current != current_id:
                    raise ZLifecycleStateError(
                        "chosen historical bucket was already superseded elsewhere"
                    )

            current_update, historical_update = lifecycle_updates(entry)
            prior_superseded = current["metadata"].get("supersedes_bucket_ids")
            if not isinstance(prior_superseded, list):
                prior_superseded = []
            current_update["supersedes_bucket_ids"] = list(dict.fromkeys(
                [
                    *(
                        str(value).strip()
                        for value in prior_superseded
                        if str(value).strip()
                    ),
                    historical_id,
                ]
            ))
            current_target = copy.deepcopy(current_post)
            historical_target = copy.deepcopy(historical_post)
            for field, value in current_update.items():
                current_target[field] = value
            for field, value in historical_update.items():
                historical_target[field] = value

            originals = {
                "current": current_path.read_text(encoding="utf-8"),
                "historical": historical_path.read_text(encoding="utf-8"),
            }
            targets = {
                "current": frontmatter.dumps(current_target),
                "historical": frontmatter.dumps(historical_target),
            }
            transaction_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            if os.name != "nt":
                os.chmod(transaction_dir, 0o700)
            for side in ("current", "historical"):
                atomic_write_text(
                    transaction_dir / f"{side}.original.md",
                    originals[side],
                )
                atomic_write_text(
                    transaction_dir / f"{side}.target.md",
                    targets[side],
                )

            manifest = {
                "version": 1,
                "queue_key": key,
                "state": "prepared",
                "reviewer": reviewer,
                "verdict_note": str(verdict_note or "").strip()[:500],
                "current_bucket_id": current_id,
                "historical_bucket_id": historical_id,
                "current_path": os.path.relpath(current_path, self.root),
                "historical_path": os.path.relpath(historical_path, self.root),
                "current_original_sha256": _sha256(originals["current"]),
                "historical_original_sha256": _sha256(originals["historical"]),
                "current_target_sha256": _sha256(targets["current"]),
                "historical_target_sha256": _sha256(targets["historical"]),
            }
            self._write_manifest(transaction_dir, manifest)

            try:
                self._write_bucket(current_path, targets["current"])
                manifest["state"] = "current_written"
                self._write_manifest(transaction_dir, manifest)
                self._write_bucket(historical_path, targets["historical"])
                manifest["state"] = "buckets_written"
                self._write_manifest(transaction_dir, manifest)
                changed = self.review_queue.apply_lifecycle(
                    key,
                    verdict_note=manifest["verdict_note"],
                    reviewer=reviewer,
                )
                if not changed:
                    raise ZLifecycleStateError(
                        "pending review item changed during approval"
                    )
            except Exception as exc:
                durable = self.review_queue.get(key)
                if durable and durable.get("status") == STATUS_APPLIED:
                    # The queue rewrite may have committed before a trailing
                    # filesystem error surfaced.  Once the durable verdict is
                    # applied, recovery must finish the targets, never roll the
                    # approved decision back underneath the ledger — and the
                    # caller must be told the truth: the approval *did* commit
                    # (2026-08-19 review P2: re-raising here made the API
                    # answer 503 / memory_mutated=false for a durable commit).
                    self._write_targets(transaction_dir, manifest)
                    manifest["state"] = "committed"
                    manifest["recovered_after_error"] = (
                        f"{type(exc).__name__}: {exc}"[:300]
                    )
                    self._write_manifest(transaction_dir, manifest)
                    return {
                        "key": key,
                        "status": STATUS_APPLIED,
                        "changed": True,
                        "current_bucket_id": current_id,
                        "historical_bucket_id": historical_id,
                        "recovered_after_error": manifest["recovered_after_error"],
                    }
                self._restore_originals(transaction_dir, manifest)
                raise

            manifest["state"] = "committed"
            self._write_manifest(transaction_dir, manifest)
            return {
                "key": key,
                "status": STATUS_APPLIED,
                "changed": True,
                "current_bucket_id": current_id,
                "historical_bucket_id": historical_id,
            }
