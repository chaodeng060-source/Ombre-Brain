"""Durable mutation journal for memory bucket changes."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from typing import Any

from utils import now_iso


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class MutationAuditLog:
    """SQLite-backed two-phase journal.

    An event is inserted as pending before the Markdown mutation. Successful
    operations mark it committed. A crash in between leaves a pending event
    containing both the before and intended after state for recovery.
    """

    def __init__(self, buckets_dir: str, config: dict | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        configured_path = cfg.get("path")
        self.path = os.path.abspath(
            configured_path
            or os.path.join(buckets_dir, ".audit", "mutations.sqlite3")
        )
        if self.enabled:
            self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.execute("PRAGMA busy_timeout = 15000")
        return conn

    def _initialize(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS mutation_audit (
                    event_id TEXT PRIMARY KEY,
                    occurred_at TEXT NOT NULL,
                    committed_at TEXT,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    bucket_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'committed', 'failed')),
                    before_json TEXT,
                    after_json TEXT,
                    details_json TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_mutation_audit_bucket_time
                    ON mutation_audit(bucket_id, occurred_at DESC);
                CREATE INDEX IF NOT EXISTS idx_mutation_audit_status_time
                    ON mutation_audit(status, occurred_at DESC);
                """
            )

    def begin(
        self,
        *,
        actor: str,
        action: str,
        bucket_id: str,
        before: Any,
        after: Any,
        details: Any = None,
    ) -> str | None:
        if not self.enabled:
            return None
        event_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mutation_audit (
                    event_id, occurred_at, actor, action, bucket_id, status,
                    before_json, after_json, details_json
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    event_id,
                    now_iso(),
                    (actor or "system").strip() or "system",
                    action,
                    bucket_id,
                    _json(before),
                    _json(after),
                    _json(details),
                ),
            )
        return event_id

    def commit(self, event_id: str | None) -> None:
        if not self.enabled or not event_id:
            return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE mutation_audit
                   SET status = 'committed', committed_at = ?, error = NULL
                 WHERE event_id = ?
                """,
                (now_iso(), event_id),
            )

    def fail(self, event_id: str | None, error: Exception | str) -> None:
        if not self.enabled or not event_id:
            return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE mutation_audit
                   SET status = 'failed', committed_at = ?, error = ?
                 WHERE event_id = ?
                """,
                (now_iso(), str(error), event_id),
            )

    def list_events(self, bucket_id: str | None = None) -> list[dict]:
        """Read audit rows for diagnostics and tests."""
        if not self.enabled:
            return []
        sql = "SELECT * FROM mutation_audit"
        params: tuple = ()
        if bucket_id:
            sql += " WHERE bucket_id = ?"
            params = (bucket_id,)
        # rowid preserves insertion order for multiple mutations in one second.
        sql += " ORDER BY rowid"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
