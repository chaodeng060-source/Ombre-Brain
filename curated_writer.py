"""Fail-closed coordinator for curated memory body + index writes.

The bucket vault and the vector database are separate durable stores.  This
module makes that split explicit:

* ``required`` writes stage their body in the archive, build the complete
  vector record, and only then promote the bucket into main recall.
* ``fts_only`` writes are allowed only when the caller explicitly selects that
  policy; the promoted bucket is marked accordingly.
* every write is bound to a durable idempotency key and canonical payload
  digest.  Replaying the same request returns the same terminal result, while
  reusing the key for different content fails closed.

The coordinator deliberately does not hide exceptions behind a boolean
"success".  Transient failures remain explicitly retryable, and partially
committed bodies stay in the cold store with a quarantine marker.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
from contextlib import asynccontextmanager, closing
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from maintenance_barrier import MaintenanceBarrier
from storage_safety import advisory_file_lock
from utils import now_iso


VECTOR_POLICIES = frozenset({"required", "fts_only"})
VISIBLE_BUCKET_TYPES = frozenset({"dynamic", "permanent"})
_TERMINAL_STATUSES = frozenset({"completed"})
_INCOMPLETE_STATUSES = frozenset(
    {"preparing", "body_pending", "retryable"}
)
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


class CuratedWriteError(RuntimeError):
    """Base error for strict curated writes."""


class IdempotencyConflictError(CuratedWriteError):
    """The same idempotency key was reused for a different payload."""


class CuratedWriteIntegrityError(CuratedWriteError):
    """A durable receipt no longer matches its referenced bucket."""


@dataclass(frozen=True)
class CuratedWriteResult:
    success: bool
    status: str
    bucket_id: str | None
    vector_policy: str
    recall_state: str
    error_code: str | None = None


_PROCESS_LOCKS: dict[str, asyncio.Lock] = {}


def _process_lock(name: str) -> asyncio.Lock:
    lock = _PROCESS_LOCKS.get(name)
    if lock is None:
        lock = asyncio.Lock()
        _PROCESS_LOCKS[name] = lock
    return lock


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("curated write payload contains a non-finite number")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("curated write payload keys must be non-empty strings")
            normalized[key] = _canonical_value(item)
        return normalized
    raise ValueError(
        f"curated write payload contains unsupported value: {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_digest(
    *,
    content: str,
    vector_policy: str,
    bucket_options: dict[str, Any],
) -> str:
    canonical = _canonical_json(
        {
            "schema": "ombre.curated-write/v1",
            "content": content,
            "vector_policy": vector_policy,
            "bucket_options": bucket_options,
        }
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CuratedWriteCoordinator:
    """Coordinate one strict, idempotent curated write."""

    _ALLOWED_BUCKET_OPTIONS = frozenset(
        {
            "tags",
            "importance",
            "domain",
            "valence",
            "arousal",
            "bucket_type",
            "name",
            "pinned",
            "protected",
            "world",
            "chord_tag",
            "tier",
            "sense",
            "event_at",
            "date_precision",
            "date_source",
            "date_confidence",
            "x_provenance",
        }
    )

    def __init__(
        self,
        bucket_manager,
        embedding_engine,
        *,
        ledger_path: str | os.PathLike[str] | None = None,
    ):
        self.bucket_manager = bucket_manager
        self.embedding_engine = embedding_engine
        base_dir = os.path.abspath(os.fspath(bucket_manager.base_dir))
        self._maintenance_barrier = getattr(
            bucket_manager,
            "_maintenance_barrier",
            None,
        ) or MaintenanceBarrier(base_dir)
        self.ledger_path = os.path.abspath(
            os.fspath(ledger_path or os.path.join(base_dir, ".curated_writes.db"))
        )
        self.locks_dir = os.path.join(base_dir, ".curated-write-locks")
        with self._maintenance_barrier.shared():
            self._init_ledger()

    def _init_ledger(self) -> None:
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        with closing(sqlite3.connect(self.ledger_path, timeout=30)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS curated_writes (
                    idempotency_key TEXT PRIMARY KEY,
                    payload_sha256 TEXT NOT NULL,
                    vector_policy TEXT NOT NULL,
                    status TEXT NOT NULL,
                    bucket_id TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        try:
            os.chmod(self.ledger_path, 0o600)
        except OSError:
            pass

    @asynccontextmanager
    async def _key_guard(self, idempotency_key: str):
        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        lock_path = os.path.join(self.locks_dir, f"{digest}.lock")
        process_key = f"{self.ledger_path}:{digest}"
        async with _process_lock(process_key):
            # The process lock prevents a second coroutine in this process
            # from blocking the event loop on the advisory OS lock.
            with advisory_file_lock(lock_path):
                yield

    def _read_row(self, idempotency_key: str) -> sqlite3.Row | None:
        with closing(sqlite3.connect(self.ledger_path, timeout=30)) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT * FROM curated_writes WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()

    def _claim(
        self,
        *,
        idempotency_key: str,
        payload_sha256: str,
        vector_policy: str,
    ) -> sqlite3.Row:
        stamp = now_iso()
        with closing(sqlite3.connect(self.ledger_path, timeout=30)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM curated_writes WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO curated_writes(
                        idempotency_key, payload_sha256, vector_policy, status,
                        bucket_id, result_json, created_at, updated_at
                    ) VALUES (?, ?, ?, 'preparing', NULL, NULL, ?, ?)
                    """,
                    (
                        idempotency_key,
                        payload_sha256,
                        vector_policy,
                        stamp,
                        stamp,
                    ),
                )
                conn.commit()
            else:
                conn.commit()
                if (
                    row["payload_sha256"] != payload_sha256
                    or row["vector_policy"] != vector_policy
                ):
                    raise IdempotencyConflictError(
                        "idempotency key is already bound to a different payload"
                    )
            return self._read_row(idempotency_key)

    def _set_bucket(self, idempotency_key: str, bucket_id: str) -> None:
        with closing(sqlite3.connect(self.ledger_path, timeout=30)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE curated_writes
                SET bucket_id = ?, status = 'body_pending', updated_at = ?
                WHERE idempotency_key = ?
                  AND status IN ('preparing', 'body_pending', 'retryable')
                """,
                (bucket_id, now_iso(), idempotency_key),
            )
            conn.commit()

    def _finish(
        self,
        idempotency_key: str,
        result: CuratedWriteResult,
    ) -> CuratedWriteResult:
        if result.status not in _TERMINAL_STATUSES:
            raise ValueError("curated write result is not terminal")
        payload = _canonical_json(asdict(result))
        with closing(sqlite3.connect(self.ledger_path, timeout=30)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE curated_writes
                SET status = ?, bucket_id = ?, result_json = ?, updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    result.status,
                    result.bucket_id,
                    payload,
                    now_iso(),
                    idempotency_key,
                ),
            )
            conn.commit()
        return result

    def _record_retryable(
        self,
        idempotency_key: str,
        result: CuratedWriteResult,
    ) -> CuratedWriteResult:
        if result.success or result.status != "retryable":
            raise ValueError("retryable result contract is invalid")
        payload = _canonical_json(asdict(result))
        with closing(sqlite3.connect(self.ledger_path, timeout=30)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE curated_writes
                SET status = 'retryable', bucket_id = ?, result_json = ?,
                    updated_at = ?
                WHERE idempotency_key = ?
                  AND status IN ('preparing', 'body_pending', 'retryable')
                """,
                (
                    result.bucket_id,
                    payload,
                    now_iso(),
                    idempotency_key,
                ),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise CuratedWriteIntegrityError(
                    "curated-write retry state could not be persisted"
                )
            conn.commit()
        return result

    @staticmethod
    def _result_from_row(row: sqlite3.Row) -> CuratedWriteResult:
        try:
            raw = json.loads(row["result_json"])
            return CuratedWriteResult(**raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CuratedWriteIntegrityError(
                "terminal curated-write receipt is malformed"
            ) from exc

    async def _verify_terminal(
        self,
        *,
        row: sqlite3.Row,
        result: CuratedWriteResult,
        idempotency_key: str,
        payload_sha256: str,
        content: str,
    ) -> None:
        if not result.bucket_id or result.bucket_id != row["bucket_id"]:
            raise CuratedWriteIntegrityError(
                "curated-write receipt bucket does not match its ledger row"
            )
        bucket = await self.bucket_manager.get(result.bucket_id)
        if not bucket:
            raise CuratedWriteIntegrityError(
                "curated-write receipt references a missing bucket"
            )
        meta = bucket.get("metadata", {}) or {}
        if bucket.get("content") != content:
            raise CuratedWriteIntegrityError(
                "curated-write bucket body no longer matches its receipt"
            )
        expected = {
            "curated_write_key": idempotency_key,
            "curated_payload_sha256": payload_sha256,
            "vector_policy": result.vector_policy,
            "lmc5_recall_state": result.recall_state,
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            raise CuratedWriteIntegrityError(
                "curated-write bucket metadata no longer matches its receipt"
            )
        if str(meta.get("type") or "") == "archived":
            raise CuratedWriteIntegrityError(
                "completed curated-write bucket is still archived"
            )
        if (
            result.vector_policy == "required"
            and not await self._has_required_vector(result.bucket_id)
        ):
            raise CuratedWriteIntegrityError(
                "completed curated-write bucket lost its required vector"
            )

    async def _find_bucket_by_identity(
        self,
        *,
        idempotency_key: str,
        payload_sha256: str,
    ) -> dict | None:
        """Recover a body created just before a worker crash.

        The identity is written in the bucket's first atomic create, so a
        missing ledger ``bucket_id`` never forces a second body write.
        """
        buckets = await self.bucket_manager.list_all(include_archive=True)
        exact: list[dict] = []
        conflicts: list[dict] = []
        for bucket in buckets:
            meta = bucket.get("metadata", {}) or {}
            if meta.get("curated_write_key") != idempotency_key:
                continue
            if meta.get("curated_payload_sha256") == payload_sha256:
                exact.append(bucket)
            else:
                conflicts.append(bucket)
        if conflicts:
            raise CuratedWriteIntegrityError(
                "curated-write key exists with a different payload marker"
            )
        if len(exact) > 1:
            raise CuratedWriteIntegrityError(
                "multiple buckets share one curated-write identity"
            )
        return exact[0] if exact else None

    async def _has_required_vector(self, bucket_id: str) -> bool:
        getter = getattr(self.embedding_engine, "get_embedding", None)
        if not callable(getter):
            return False
        try:
            return bool(await getter(bucket_id))
        except Exception:
            return False

    async def _recover_promoted_result(
        self,
        *,
        bucket: dict,
        idempotency_key: str,
        payload_sha256: str,
        vector_policy: str,
    ) -> CuratedWriteResult | None:
        """Recognize a promotion committed just before its receipt."""
        meta = bucket.get("metadata", {}) or {}
        ready_state = "ready_vector" if vector_policy == "required" else "ready_fts"
        expected = {
            "curated_write_key": idempotency_key,
            "curated_payload_sha256": payload_sha256,
            "vector_policy": vector_policy,
            "lmc5_recall_state": ready_state,
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            return None
        if str(meta.get("type") or "") == "archived":
            return None
        if vector_policy == "required" and not await self._has_required_vector(
            bucket["id"]
        ):
            raise CuratedWriteIntegrityError(
                "promoted required-vector bucket has no durable vector"
            )
        return CuratedWriteResult(
            success=True,
            status="completed",
            bucket_id=bucket["id"],
            vector_policy=vector_policy,
            recall_state=ready_state,
        )

    @staticmethod
    def _normalize_options(bucket_options: dict[str, Any] | None) -> dict[str, Any]:
        options = dict(bucket_options or {})
        unknown = set(options) - CuratedWriteCoordinator._ALLOWED_BUCKET_OPTIONS
        if unknown:
            raise ValueError(
                f"unsupported curated bucket options: {sorted(unknown)}"
            )
        normalized = _canonical_value(options)
        desired_type = str(normalized.get("bucket_type", "dynamic")).strip()
        if desired_type not in VISIBLE_BUCKET_TYPES:
            raise ValueError(
                "curated bucket_type must be 'dynamic' or 'permanent'"
            )
        normalized["bucket_type"] = desired_type
        return normalized

    @staticmethod
    def _staging_options(options: dict[str, Any]) -> dict[str, Any]:
        staged = dict(options)
        staged["bucket_type"] = "archived"
        # Never let pinning route an intermediate body to permanent storage,
        # and keep protected=False so a failed stage remains removable.
        staged["pinned"] = False
        staged["protected"] = False
        return staged

    @staticmethod
    def _promotion_updates(
        options: dict[str, Any],
        *,
        idempotency_key: str,
        payload_sha256: str,
        vector_policy: str,
        recall_state: str,
    ) -> dict[str, Any]:
        desired_type = options["bucket_type"]
        pinned = bool(options.get("pinned", False))
        if pinned:
            desired_type = "permanent"
        updates = {
            "type": desired_type,
            "pinned": pinned,
            "protected": bool(options.get("protected", False)),
            "curated_write_key": idempotency_key,
            "curated_payload_sha256": payload_sha256,
            "vector_policy": vector_policy,
            "lmc5_recall_state": recall_state,
        }
        if pinned or updates["protected"]:
            updates["importance"] = 10
        return updates

    async def write(
        self,
        *,
        idempotency_key: str,
        content: str,
        vector_policy: str,
        bucket_options: dict[str, Any] | None = None,
        actor: str = "lmc5:curated",
    ) -> CuratedWriteResult:
        async with self._maintenance_barrier.shared_async():
            return await self._write_locked(
                idempotency_key=idempotency_key,
                content=content,
                vector_policy=vector_policy,
                bucket_options=bucket_options,
                actor=actor,
            )

    async def _write_locked(
        self,
        *,
        idempotency_key: str,
        content: str,
        vector_policy: str,
        bucket_options: dict[str, Any] | None = None,
        actor: str = "lmc5:curated",
    ) -> CuratedWriteResult:
        if not isinstance(idempotency_key, str) or not _KEY_RE.fullmatch(
            idempotency_key
        ):
            raise ValueError("invalid curated-write idempotency key")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("curated-write content must be non-empty")
        if vector_policy not in VECTOR_POLICIES:
            raise ValueError("vector_policy must be 'required' or 'fts_only'")
        options = self._normalize_options(bucket_options)
        payload_sha256 = _payload_digest(
            content=content,
            vector_policy=vector_policy,
            bucket_options=options,
        )

        async with self._key_guard(idempotency_key):
            row = self._claim(
                idempotency_key=idempotency_key,
                payload_sha256=payload_sha256,
                vector_policy=vector_policy,
            )
            if row["status"] in _TERMINAL_STATUSES:
                result = self._result_from_row(row)
                await self._verify_terminal(
                    row=row,
                    result=result,
                    idempotency_key=idempotency_key,
                    payload_sha256=payload_sha256,
                    content=content,
                )
                return result
            if row["status"] not in _INCOMPLETE_STATUSES:
                raise CuratedWriteIntegrityError(
                    "curated-write ledger has an unknown nonterminal status"
                )

            bucket_id = row["bucket_id"]
            staged_bucket = None
            if bucket_id:
                staged_bucket = await self.bucket_manager.get(bucket_id)
                if not staged_bucket or staged_bucket.get("content") != content:
                    raise CuratedWriteIntegrityError(
                        "incomplete curated-write stage is missing or has changed"
                    )
                recovered = await self._recover_promoted_result(
                    bucket=staged_bucket,
                    idempotency_key=idempotency_key,
                    payload_sha256=payload_sha256,
                    vector_policy=vector_policy,
                )
                if recovered is not None:
                    return self._finish(idempotency_key, recovered)
                if str(
                    (staged_bucket.get("metadata", {}) or {}).get("type") or ""
                ) != "archived":
                    raise CuratedWriteIntegrityError(
                        "incomplete curated-write body became recall-visible"
                    )
            else:
                staged_bucket = await self._find_bucket_by_identity(
                    idempotency_key=idempotency_key,
                    payload_sha256=payload_sha256,
                )
                if staged_bucket is not None:
                    bucket_id = staged_bucket["id"]
                    recovered = await self._recover_promoted_result(
                        bucket=staged_bucket,
                        idempotency_key=idempotency_key,
                        payload_sha256=payload_sha256,
                        vector_policy=vector_policy,
                    )
                    self._set_bucket(idempotency_key, bucket_id)
                    if recovered is not None:
                        return self._finish(idempotency_key, recovered)
                    if str(
                        (staged_bucket.get("metadata", {}) or {}).get("type") or ""
                    ) != "archived":
                        raise CuratedWriteIntegrityError(
                            "recovered curated-write body is visible but not complete"
                        )
                try:
                    if staged_bucket is None:
                        bucket_id = await self.bucket_manager.create(
                            content=content,
                            actor=actor,
                            curated_write_key=idempotency_key,
                            curated_payload_sha256=payload_sha256,
                            vector_policy=vector_policy,
                            lmc5_recall_state=(
                                "pending_vector"
                                if vector_policy == "required"
                                else "pending_fts"
                            ),
                            **self._staging_options(options),
                        )
                except Exception:
                    # The create may have reached disk before an outer adapter
                    # raised.  Do not claim terminal failure or create again
                    # blindly; the next replay reconciles the atomic marker.
                    return self._record_retryable(
                        idempotency_key,
                        CuratedWriteResult(
                            success=False,
                            status="retryable",
                            bucket_id=None,
                            vector_policy=vector_policy,
                            recall_state="absent",
                            error_code="body_write_failed",
                        ),
                    )
                if not bucket_id:
                    return self._record_retryable(
                        idempotency_key,
                        CuratedWriteResult(
                            success=False,
                            status="retryable",
                            bucket_id=None,
                            vector_policy=vector_policy,
                            recall_state="absent",
                            error_code="body_write_failed",
                        ),
                    )
                if staged_bucket is None:
                    self._set_bucket(idempotency_key, bucket_id)

            annotations = {
                "curated_write_key": idempotency_key,
                "curated_payload_sha256": payload_sha256,
                "vector_policy": vector_policy,
                "lmc5_recall_state": "pending_vector"
                if vector_policy == "required"
                else "pending_fts",
            }
            if not await self.bucket_manager.update(
                bucket_id,
                actor=actor,
                **annotations,
            ):
                return self._record_retryable(
                    idempotency_key,
                    CuratedWriteResult(
                        success=False,
                        status="retryable",
                        bucket_id=bucket_id,
                        vector_policy=vector_policy,
                        recall_state="quarantined_metadata",
                        error_code="staging_metadata_failed",
                    ),
                )

            if vector_policy == "required":
                vector_ok = False
                try:
                    vector_ok = bool(
                        await self.embedding_engine.generate_and_store(
                            bucket_id, content
                        )
                    )
                except Exception:
                    vector_ok = False
                if not vector_ok:
                    quarantine = {
                        **annotations,
                        "type": "archived",
                        "lmc5_recall_state": "quarantined_vector",
                    }
                    await self.bucket_manager.update(
                        bucket_id,
                        actor=actor,
                        **quarantine,
                    )
                    return self._record_retryable(
                        idempotency_key,
                        CuratedWriteResult(
                            success=False,
                            status="retryable",
                            bucket_id=bucket_id,
                            vector_policy=vector_policy,
                            recall_state="quarantined_vector",
                            error_code="vector_required_failed",
                        ),
                    )
                ready_state = "ready_vector"
            else:
                ready_state = "ready_fts"

            promoted = await self.bucket_manager.update(
                bucket_id,
                actor=actor,
                **self._promotion_updates(
                    options,
                    idempotency_key=idempotency_key,
                    payload_sha256=payload_sha256,
                    vector_policy=vector_policy,
                    recall_state=ready_state,
                ),
            )
            if not promoted:
                if vector_policy == "required":
                    try:
                        self.embedding_engine.delete_embedding(bucket_id)
                    except Exception:
                        pass
                await self.bucket_manager.update(
                    bucket_id,
                    actor=actor,
                    type="archived",
                    lmc5_recall_state="quarantined_promotion",
                )
                return self._record_retryable(
                    idempotency_key,
                    CuratedWriteResult(
                        success=False,
                        status="retryable",
                        bucket_id=bucket_id,
                        vector_policy=vector_policy,
                        recall_state="quarantined_promotion",
                        error_code="promotion_failed",
                    ),
                )

            result = CuratedWriteResult(
                success=True,
                status="completed",
                bucket_id=bucket_id,
                vector_policy=vector_policy,
                recall_state=ready_state,
            )
            return self._finish(idempotency_key, result)
