"""Idempotent receipts for memories that reached the final model prompt.

Candidate retrieval stays read-only.  The caller records a receipt only after
it has committed the rendered recall block to the prompt.  SQLite owns replay
and payload-conflict detection; bucket updates remain resumable item by item.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


_SAFE_BUCKET_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MAX_EVENT_ID_CHARS = 512


class RecallReceiptError(RuntimeError):
    """Base receipt error."""


class RecallReceiptConflict(RecallReceiptError):
    """The same event id was reused for a different set of memories."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_bucket_ids(values: Iterable[object]) -> tuple[str, ...]:
    bucket_ids: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        bucket_id = str(value or "").strip()
        if not bucket_id or bucket_id in seen:
            continue
        if not _SAFE_BUCKET_ID.fullmatch(bucket_id):
            raise ValueError("invalid bucket_id")
        seen.add(bucket_id)
        bucket_ids.append(bucket_id)
        if len(bucket_ids) > 32:
            raise ValueError("too many bucket_ids")
    return tuple(bucket_ids)


class RecallReceiptStore:
    """Small SQLite ledger; no recalled text or query is persisted."""

    def __init__(self, buckets_dir: str | os.PathLike[str]):
        base = Path(buckets_dir).resolve()
        self.directory = base / ".recall_receipts"
        self.path = self.directory / "receipts.sqlite3"

    def initialize(self) -> None:
        if self.directory.exists() or self.directory.is_symlink():
            info = os.lstat(self.directory)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise RecallReceiptError("unsafe receipt directory")
            if info.st_mode & 0o077:
                raise RecallReceiptError("unsafe receipt directory mode")
        else:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(self.directory, 0o700)
        if not self.path.exists() and not self.path.is_symlink():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(self.path, flags, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        self._validate_database_if_present()
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS receipt_events (
                    event_id TEXT PRIMARY KEY,
                    payload_sha256 TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending','complete')),
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS receipt_items (
                    event_id TEXT NOT NULL REFERENCES receipt_events(event_id)
                        ON DELETE CASCADE,
                    bucket_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending','applied')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (event_id, bucket_id)
                );
                CREATE INDEX IF NOT EXISTS idx_receipt_items_status
                    ON receipt_items(status, updated_at);
                """
            )
        os.chmod(self.path, 0o600)

    def _validate_database_if_present(self) -> None:
        if not self.path.exists() and not self.path.is_symlink():
            return
        info = os.lstat(self.path)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1
            or info.st_mode & 0o077
        ):
            raise RecallReceiptError("unsafe receipt database")

    def _connect(self) -> sqlite3.Connection:
        self._validate_database_if_present()
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @staticmethod
    def _payload_sha256(bucket_ids: tuple[str, ...]) -> str:
        encoded = json.dumps(
            bucket_ids, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def begin(self, event_id: str, bucket_ids: Iterable[object], source: str) -> dict:
        event_id = str(event_id or "").strip()
        if not event_id or len(event_id) > _MAX_EVENT_ID_CHARS:
            raise ValueError("invalid event_id")
        normalized = normalize_bucket_ids(bucket_ids)
        if not normalized:
            raise ValueError("bucket_ids required")
        source = str(source or "recall")[:80]
        payload_sha256 = self._payload_sha256(normalized)
        now = _now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT payload_sha256,status FROM receipt_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_sha256"] != payload_sha256:
                    conn.rollback()
                    raise RecallReceiptConflict("event_id payload conflict")
                pending = [
                    row[0]
                    for row in conn.execute(
                        "SELECT bucket_id FROM receipt_items "
                        "WHERE event_id=? AND status='pending' ORDER BY bucket_id",
                        (event_id,),
                    )
                ]
                conn.commit()
                return {
                    "duplicate": existing["status"] == "complete",
                    "pending": pending,
                    "total": len(normalized),
                }
            conn.execute(
                "INSERT INTO receipt_events"
                "(event_id,payload_sha256,source,status,created_at) VALUES(?,?,?,?,?)",
                (event_id, payload_sha256, source, "pending", now),
            )
            conn.executemany(
                "INSERT INTO receipt_items"
                "(event_id,bucket_id,status,updated_at) VALUES(?,?,'pending',?)",
                [(event_id, bucket_id, now) for bucket_id in normalized],
            )
            conn.commit()
        return {"duplicate": False, "pending": list(normalized), "total": len(normalized)}

    def mark_applied(self, event_id: str, bucket_id: str) -> None:
        now = _now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE receipt_items SET status='applied',attempts=attempts+1,"
                "last_error='',updated_at=? WHERE event_id=? AND bucket_id=?",
                (now, event_id, bucket_id),
            )
            remaining = conn.execute(
                "SELECT COUNT(*) FROM receipt_items "
                "WHERE event_id=? AND status!='applied'",
                (event_id,),
            ).fetchone()[0]
            if remaining == 0:
                conn.execute(
                    "UPDATE receipt_events SET status='complete',completed_at=? "
                    "WHERE event_id=?",
                    (now, event_id),
                )
            conn.commit()

    def mark_failed(self, event_id: str, bucket_id: str, error: object) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE receipt_items SET attempts=attempts+1,last_error=?,updated_at=? "
                "WHERE event_id=? AND bucket_id=? AND status='pending'",
                (type(error).__name__[:80], _now_iso(), event_id, bucket_id),
            )

    def status(self, event_id: str) -> dict:
        with self._connect() as conn:
            event = conn.execute(
                "SELECT status FROM receipt_events WHERE event_id=?", (event_id,)
            ).fetchone()
            if event is None:
                return {"status": "missing", "applied": 0, "pending": 0}
            rows = conn.execute(
                "SELECT status,COUNT(*) AS n FROM receipt_items "
                "WHERE event_id=? GROUP BY status",
                (event_id,),
            ).fetchall()
        counts = {row["status"]: int(row["n"]) for row in rows}
        return {
            "status": event["status"],
            "applied": counts.get("applied", 0),
            "pending": counts.get("pending", 0),
        }


__all__ = [
    "RecallReceiptConflict",
    "RecallReceiptError",
    "RecallReceiptStore",
    "normalize_bucket_ids",
]
