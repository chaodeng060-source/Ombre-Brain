"""Durable, fail-soft SQLite -> PostgreSQL vector mirror queue.

Markdown and the local SQLite embedding store remain authoritative.  Bucket
writes only enqueue a bucket id; a background worker mirrors the latest SQLite
record to PostgreSQL.  PostgreSQL/psql failures therefore leave a local retry
row and never participate in the memory write or recall path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import subprocess
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger("ombre_brain.pg_mirror")


class PgMirrorItemError(ValueError):
    """One bucket cannot be mirrored until its source data changes."""


class PgMirrorUnavailable(RuntimeError):
    """The PostgreSQL transport or mirror schema is currently unavailable."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _enabled(config_value) -> bool:
    override = os.environ.get("OMBRE_PG_MIRROR_ENABLED")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return config_value is True


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@dataclass(frozen=True)
class PendingMutation:
    bucket_id: str
    action: str
    source: str
    revision: int
    enqueued_at: str
    attempts: int
    last_error: str
    next_retry_at: float


class PgMirrorQueue:
    """Coalescing on-disk queue plus one-batch PostgreSQL drain."""

    def __init__(self, config: dict):
        mirror = config.get("pg_mirror", {}) or {}
        self.enabled = _enabled(mirror.get("enabled", False))
        buckets_dir = Path(config["buckets_dir"])
        queue_path = str(mirror.get("queue_path") or "").strip()
        self.path = (
            Path(queue_path).expanduser()
            if queue_path
            else buckets_dir / ".pg_mirror" / "queue.sqlite3"
        )
        self.embedding_db = buckets_dir / "embeddings.db"
        self.database = str(mirror.get("database") or "ombre_mirror")
        self.psql_path = str(mirror.get("psql_path") or "psql")
        self.dimension = int(mirror.get("dimension", 1024))
        self.batch_size = max(1, min(1000, int(mirror.get("batch_size", 100))))
        self.retry_seconds = max(0.01, float(mirror.get("retry_seconds", 30.0)))
        if self.dimension < 1:
            raise ValueError("pg_mirror.dimension must be positive")
        if self.enabled:
            self._init_db()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pending (
                    bucket_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    source TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    enqueued_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    next_retry_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(pending)")
            }
            if "next_retry_at" not in columns:
                connection.execute(
                    "ALTER TABLE pending "
                    "ADD COLUMN next_retry_at REAL NOT NULL DEFAULT 0"
                )
            connection.commit()

    def enqueue(
        self,
        bucket_id: str,
        *,
        action: str = "dirty",
        source: str = "bucket_manager",
    ) -> bool:
        """Persist the latest intent for one bucket without raising upstream."""

        if not self.enabled:
            return False
        bucket_id = str(bucket_id or "").strip()
        source = str(source or "bucket_manager").strip()[:160]
        if not bucket_id or len(bucket_id) > 256 or action not in {"dirty", "upsert", "delete"}:
            logger.warning("Rejected invalid PG mirror queue item")
            return False
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO pending (
                        bucket_id, action, source, revision, enqueued_at,
                        attempts, last_error, next_retry_at
                    ) VALUES (?, ?, ?, 1, ?, 0, '', 0)
                    ON CONFLICT(bucket_id) DO UPDATE SET
                        action = excluded.action,
                        source = excluded.source,
                        revision = pending.revision + 1,
                        enqueued_at = excluded.enqueued_at,
                        attempts = 0,
                        last_error = '',
                        next_retry_at = 0
                    """,
                    (bucket_id, action, source, _now_iso()),
                )
                connection.commit()
            return True
        except Exception as exc:
            logger.warning(
                "PG mirror queue unavailable for %s: %s",
                bucket_id,
                type(exc).__name__,
            )
            return False

    def pending(self, *, limit: int | None = None) -> list[PendingMutation]:
        if not self.enabled:
            return []
        limit = self.batch_size if limit is None else max(1, int(limit))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT bucket_id, action, source, revision, enqueued_at,
                       attempts, last_error, next_retry_at
                  FROM pending
                 ORDER BY enqueued_at, bucket_id
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [PendingMutation(**dict(row)) for row in rows]

    def _ready(self, *, limit: int) -> list[PendingMutation]:
        """Return due rows while leaving deferred poison rows observable."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT bucket_id, action, source, revision, enqueued_at,
                       attempts, last_error, next_retry_at
                  FROM pending
                 WHERE next_retry_at <= ?
                 ORDER BY enqueued_at, bucket_id
                 LIMIT ?
                """,
                (time.time(), max(1, int(limit))),
            ).fetchall()
        return [PendingMutation(**dict(row)) for row in rows]

    def pending_count(self) -> int:
        if not self.enabled:
            return 0
        with closing(self._connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM pending").fetchone()[0])

    def _ack(self, item: PendingMutation) -> None:
        # A newer enqueue increments revision.  Never delete that newer write
        # merely because an older in-flight sync completed afterwards.
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM pending WHERE bucket_id = ? AND revision = ?",
                (item.bucket_id, item.revision),
            )
            connection.commit()

    def _record_failure(
        self,
        item: PendingMutation,
        exc: Exception,
        *,
        defer_item: bool,
    ) -> None:
        message = f"{type(exc).__name__}: {exc}"[:500]
        next_retry_at = 0.0
        if defer_item:
            delay = min(
                max(1.0, self.retry_seconds) * (2 ** min(item.attempts, 8)),
                3600.0,
            )
            next_retry_at = time.time() + delay
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE pending
                   SET attempts = attempts + 1,
                       last_error = ?,
                       next_retry_at = ?
                 WHERE bucket_id = ? AND revision = ?
                """,
                (
                    message,
                    next_retry_at,
                    item.bucket_id,
                    item.revision,
                ),
            )
            connection.commit()

    def _load_source(self, bucket_id: str) -> tuple[str, str] | None:
        if not self.embedding_db.exists():
            return None
        uri = f"file:{self.embedding_db}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
            row = connection.execute(
                "SELECT embedding, updated_at FROM embeddings WHERE bucket_id = ?",
                (bucket_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0]), str(row[1] or "")

    def _segments(self, embedding_json: str) -> list[str]:
        try:
            value = json.loads(embedding_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PgMirrorItemError("embedding is not valid JSON") from exc
        if not isinstance(value, list) or not value:
            raise PgMirrorItemError("embedding is empty")
        segments = value if isinstance(value[0], list) else [value]
        encoded: list[str] = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, list) or len(segment) != self.dimension:
                size = len(segment) if isinstance(segment, list) else -1
                raise PgMirrorItemError(
                    f"segment {index} dim {size} != {self.dimension}"
                )
            numbers: list[str] = []
            for raw in segment:
                try:
                    number = float(raw)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise PgMirrorItemError(
                        f"segment {index} contains a non-number"
                    ) from exc
                if not math.isfinite(number):
                    raise PgMirrorItemError(
                        "embedding contains a non-finite value"
                    )
                numbers.append(repr(number))
            encoded.append("[" + ",".join(numbers) + "]")
        return encoded

    def _run_psql(self, script: str) -> None:
        try:
            result = subprocess.run(
                [
                    self.psql_path,
                    "-d",
                    self.database,
                    "-X",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-q",
                ],
                input=script,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise PgMirrorUnavailable(type(exc).__name__) from exc
        if result.returncode != 0:
            raise PgMirrorUnavailable(result.stderr.strip() or "psql failed")

    def _delete_pg(self, bucket_id: str) -> None:
        self._run_psql(
            "BEGIN;\n"
            f"DELETE FROM ombre_vectors WHERE bucket_id = {_sql_literal(bucket_id)};\n"
            "COMMIT;\n"
        )

    def _upsert_pg(self, bucket_id: str, embedding_json: str, updated_at: str) -> None:
        segments = self._segments(embedding_json)
        rows = ",\n".join(
            "(" + ", ".join((
                _sql_literal(bucket_id),
                str(index),
                _sql_literal(vector) + f"::halfvec({self.dimension})",
                _sql_literal(updated_at),
            )) + ")"
            for index, vector in enumerate(segments)
        )
        self._run_psql(
            "BEGIN;\n"
            f"DELETE FROM ombre_vectors WHERE bucket_id = {_sql_literal(bucket_id)};\n"
            "INSERT INTO ombre_vectors "
            "(bucket_id, segment_idx, embedding, source_updated_at) VALUES\n"
            f"{rows};\n"
            "COMMIT;\n"
        )

    def _apply(self, item: PendingMutation) -> None:
        if item.action == "delete":
            self._delete_pg(item.bucket_id)
            return
        source = self._load_source(item.bucket_id)
        if source is None:
            # A Markdown create is queued before its embedding API call.  The
            # later SQLite commit enqueues an explicit upsert.  Metadata-only
            # buckets with no vector need no PostgreSQL row.
            if item.action == "dirty":
                return
            raise PgMirrorItemError("source embedding is missing")
        self._upsert_pg(item.bucket_id, source[0], source[1])

    def drain_once(self) -> dict[str, int | str]:
        if not self.enabled:
            return {"status": "disabled", "processed": 0, "failed": 0, "remaining": 0}
        processed = 0
        failed = 0
        for item in self._ready(limit=self.batch_size):
            try:
                self._apply(item)
                self._ack(item)
                processed += 1
            except Exception as exc:
                item_error = isinstance(exc, PgMirrorItemError)
                self._record_failure(item, exc, defer_item=item_error)
                failed += 1
                logger.warning(
                    "PG mirror sync deferred for %s: %s",
                    item.bucket_id,
                    type(exc).__name__,
                )
                if not item_error:
                    # One unavailable local PG would fail the whole batch.
                    # Leave this row due so the next cycle probes PG once,
                    # rather than walking and failing every queued bucket.
                    break
        return {
            "status": "deferred" if failed else "ok",
            "processed": processed,
            "failed": failed,
            "remaining": self.pending_count(),
        }


class PgMirrorWorker:
    """One process-local poller; queue durability covers restarts."""

    def __init__(self, queue: PgMirrorQueue):
        self.queue = queue
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if not self.queue.enabled:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="pg-mirror-sync")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(
                "PG mirror worker stopped after prior failure: %s",
                type(exc).__name__,
            )

    async def _run(self) -> None:
        logger.info("PG mirror worker started")
        while True:
            try:
                await asyncio.to_thread(self.queue.drain_once)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("PG mirror worker cycle failed; retrying")
            await asyncio.sleep(self.queue.retry_seconds)
