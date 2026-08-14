"""Validity markers for mutable operational-status memories.

This is an additive sidecar over Ombre's Markdown vault.  It deliberately
handles only deployment, completion, and progress facts; narrative memories
and the existing registered Z-axis fact slots keep their own semantics.

The temporal model follows Graphiti's edge invalidation distinction:
``valid_at`` / ``invalid_at`` describe event time, while ``expired_at`` is the
processing time at which Ombre learned that a status was no longer current.
Old buckets are retained verbatim and remain available to historical queries.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Mapping


VALIDITY_KIND = "operational_status"
STATE_CURRENT = "current"
STATE_HISTORICAL = "historical"
STATE_CONTESTED = "contested"
STATES = frozenset({STATE_CURRENT, STATE_HISTORICAL, STATE_CONTESTED})

VIEW_CURRENT = "current"
VIEW_HISTORICAL = "historical"
VIEW_NEUTRAL = "neutral"

_PROTECTED_DOMAINS = frozenset({"恋爱", "纪念日", "约定", "家庭", "自省", "feel"})
_NARRATIVE_TYPES = frozenset({"feel", "episode", "saga", "permanent"})
_ENGINEERING_DOMAIN_HINTS = (
    "工程", "编程", "技术", "开发", "运维", "工作进展", "里程碑",
    "记忆恢复", "记忆库", "备份", "nas",
    "software", "engineering", "programming", "devops",
)
_TECHNICAL_CONTENT_HINTS = (
    "任务", "验收", "测试", "代码", "提交", "commit", "push", "分支",
    "branch", "部署", "上线", "发布", "容器", "container", "服务", "server",
    "缓存", "cache", "assembly", "filter", "bug", "接口", "api", "mcp",
    "nas", "vps", "rsync", "manifest", "cron",
)
_STATUS_RE = re.compile(
    r"(?:已|未|没|尚未)?(?:上线|部署|发布|落地|合入|完成|做完|跑完|收口|通过|同步|恢复|修复|重启|备份)"
    r"|(?:进行中|进度|跑了|回滚|撤回|结案|暂停|停滞|失败|全绿|验收|挂起|待办|计划)",
    re.IGNORECASE,
)
_PROGRESS_SNAPSHOT_RE = re.compile(
    r"\b\d+\s*/\s*\d+\b"
    r"|(?:剩|余)\s*\d+\s*(?:条|项|个|步)"
    r"|(?:预计|约)\s*\d+(?:\.\d+)?\s*(?:分钟|小时|天)",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"(?:吗|么|没|是否|怎样|怎么样|如何|到哪|进度|状态|现在|当前|目前|最新|[?？])"
    r"|\b(?:done|deployed|released|complete|completed|progress|status|current|latest)\b",
    re.IGNORECASE,
)
_HISTORICAL_QUERY_HINTS = (
    "以前", "过去", "上次", "历史", "当时", "之前", "曾经", "原来", "最初",
    "old", "previous", "historical", "before", "back then", "used to",
)
_MARKER_FIELDS = (
    "validity_kind",
    "validity_state",
    "status_key",
    "validity_valid_at",
    "validity_invalid_at",
    "validity_expired_at",
    "validity_superseded_by_bucket_id",
    "validity_supersedes_bucket_ids",
    "validity_source_ref",
)


def _metadata_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value]
    return []


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_time(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {raw}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _timestamp_key(value: str | None) -> float:
    normalized = _normalize_time(value)
    if not normalized:
        return float("-inf")
    return datetime.fromisoformat(normalized).timestamp()


def operational_status_query_view(query: str) -> str:
    """Return the narrow status view requested by a natural-language query."""
    normalized = " ".join(str(query or "").strip().lower().split())
    if not normalized or not _STATUS_RE.search(normalized):
        return VIEW_NEUTRAL
    if any(hint in normalized for hint in _HISTORICAL_QUERY_HINTS):
        return VIEW_HISTORICAL
    return VIEW_CURRENT if _QUESTION_RE.search(normalized) else VIEW_NEUTRAL


def is_operational_status_fact(
    content: str,
    domains: Iterable[str] | str | None = None,
    *,
    bucket_type: str = "dynamic",
    pinned: bool = False,
    protected: bool = False,
) -> bool:
    """Conservatively recognize deployment/completion/progress snapshots."""
    if pinned or protected or str(bucket_type or "").strip().lower() in _NARRATIVE_TYPES:
        return False
    domain_values = {item.strip().lower() for item in _metadata_list(domains)}
    if domain_values.intersection({item.lower() for item in _PROTECTED_DOMAINS}):
        return False
    text = str(content or "")
    if not _STATUS_RE.search(text):
        return False
    engineering_domain = any(
        hint in domain
        for domain in domain_values
        for hint in _ENGINEERING_DOMAIN_HINTS
    )
    technical_text = any(hint.lower() in text.lower() for hint in _TECHNICAL_CONTENT_HINTS)
    return engineering_domain or technical_text


def bucket_is_operational_status(bucket: Mapping | None) -> bool:
    """Recognize status evidence for rendering, including protected history.

    Narrative/protected buckets remain ineligible for automatic supersession,
    but a current-status query must still label their embedded engineering
    snapshot as unknown rather than silently presenting it as current.
    """
    if not isinstance(bucket, Mapping):
        return False
    metadata = bucket.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if metadata.get("validity_kind") == VALIDITY_KIND:
        return True
    text = str(bucket.get("content") or "")
    domains = {item.strip().lower() for item in _metadata_list(metadata.get("domain"))}
    return bool(
        _STATUS_RE.search(text)
        and (
            any(
                hint in domain
                for domain in domains
                for hint in _ENGINEERING_DOMAIN_HINTS
            )
            or any(hint.lower() in text.lower() for hint in _TECHNICAL_CONTENT_HINTS)
            or bool(_PROGRESS_SNAPSHOT_RE.search(text))
        )
    )


def validity_label(bucket: Mapping | None, *, view: str) -> dict[str, str | list[str]]:
    """Return a rendering label for one status candidate in a status query."""
    if view == VIEW_NEUTRAL or not bucket_is_operational_status(bucket):
        return {}
    metadata = bucket.get("metadata", {}) if isinstance(bucket, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if metadata.get("validity_kind") != VALIDITY_KIND:
        return {"state": "unknown"}
    state = str(metadata.get("validity_state") or "").strip().lower()
    if state not in STATES:
        return {"state": "unknown"}
    result: dict[str, str | list[str]] = {"state": state}
    for key in ("valid_at", "invalid_at", "expired_at"):
        value = metadata.get(f"validity_{key}")
        if value:
            result[key] = value
    for key in ("superseded_by_bucket_id", "supersedes_bucket_ids"):
        value = metadata.get(f"validity_{key}")
        if value:
            result[key] = value
    return result


class OperationalStatusValidityStore:
    """SQLite marker layer; reads never create or mutate the sidecar."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)

    def _connect(self, *, create: bool) -> sqlite3.Connection | None:
        if not create and not os.path.isfile(self.path):
            return None
        if create:
            os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        if create:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS operational_status_validity (
                    bucket_id TEXT PRIMARY KEY,
                    status_key TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('current','historical','contested')),
                    valid_at TEXT,
                    invalid_at TEXT,
                    expired_at TEXT,
                    superseded_by_bucket_id TEXT,
                    supersedes_bucket_ids TEXT NOT NULL DEFAULT '[]',
                    source_ref TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_status_validity_key_state
                    ON operational_status_validity(status_key, state);
                """
            )
            try:
                os.chmod(os.path.dirname(self.path), 0o700)
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        return connection

    @staticmethod
    def _row_to_marker(row: sqlite3.Row) -> dict:
        try:
            supersedes = json.loads(row["supersedes_bucket_ids"] or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            supersedes = []
        return {
            "validity_kind": VALIDITY_KIND,
            "validity_state": row["state"],
            "status_key": row["status_key"],
            "validity_valid_at": row["valid_at"] or "",
            "validity_invalid_at": row["invalid_at"] or "",
            "validity_expired_at": row["expired_at"] or "",
            "validity_superseded_by_bucket_id": row["superseded_by_bucket_id"] or "",
            "validity_supersedes_bucket_ids": supersedes if isinstance(supersedes, list) else [],
            "validity_source_ref": row["source_ref"],
        }

    def lookup_many(self, bucket_ids: Iterable[str]) -> dict[str, dict]:
        ids = list(dict.fromkeys(
            str(bucket_id).strip()
            for bucket_id in bucket_ids
            if str(bucket_id).strip()
        ))
        if not ids:
            return {}
        connection = self._connect(create=False)
        if connection is None:
            return {}
        try:
            markers: dict[str, dict] = {}
            for offset in range(0, len(ids), 500):
                chunk = ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT * FROM operational_status_validity "
                    f"WHERE bucket_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                markers.update({row["bucket_id"]: self._row_to_marker(row) for row in rows})
            return markers
        finally:
            connection.close()

    def attach(self, buckets: Iterable[dict]) -> list[dict]:
        """Refresh marker fields on in-memory candidates without touching Markdown."""
        candidates = list(buckets)
        markers = self.lookup_many(
            str(bucket.get("id") or "")
            for bucket in candidates
            if isinstance(bucket, Mapping)
        )
        for bucket in candidates:
            if not isinstance(bucket, dict):
                continue
            metadata = bucket.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                bucket["metadata"] = metadata
            for field in _MARKER_FIELDS:
                metadata.pop(field, None)
            marker = markers.get(str(bucket.get("id") or ""))
            if marker:
                metadata.update(marker)
        return candidates

    @staticmethod
    def _upsert(
        connection: sqlite3.Connection,
        *,
        bucket_id: str,
        status_key: str,
        state: str,
        valid_at: str = "",
        invalid_at: str = "",
        expired_at: str = "",
        superseded_by_bucket_id: str = "",
        supersedes_bucket_ids: Iterable[str] = (),
        source_ref: str,
        updated_at: str,
    ) -> None:
        if state not in STATES:
            raise ValueError(f"invalid validity state: {state}")
        connection.execute(
            """
            INSERT INTO operational_status_validity(
                bucket_id, status_key, state, valid_at, invalid_at, expired_at,
                superseded_by_bucket_id, supersedes_bucket_ids, source_ref, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(bucket_id) DO UPDATE SET
                status_key=excluded.status_key,
                state=excluded.state,
                valid_at=excluded.valid_at,
                invalid_at=excluded.invalid_at,
                expired_at=excluded.expired_at,
                superseded_by_bucket_id=excluded.superseded_by_bucket_id,
                supersedes_bucket_ids=excluded.supersedes_bucket_ids,
                source_ref=excluded.source_ref,
                updated_at=excluded.updated_at
            """,
            (
                bucket_id,
                status_key,
                state,
                valid_at or None,
                invalid_at or None,
                expired_at or None,
                superseded_by_bucket_id or None,
                json.dumps(list(dict.fromkeys(supersedes_bucket_ids))),
                source_ref,
                updated_at,
            ),
        )

    def mark_current(
        self,
        bucket_id: str,
        *,
        status_key: str,
        valid_at: str,
        source_ref: str,
    ) -> None:
        """Register an audited current status without altering the bucket file."""
        normalized_valid_at = _normalize_time(valid_at)
        now = _iso_now()
        connection = self._connect(create=True)
        assert connection is not None
        try:
            with connection:
                self._upsert(
                    connection,
                    bucket_id=str(bucket_id),
                    status_key=str(status_key),
                    state=STATE_CURRENT,
                    valid_at=normalized_valid_at,
                    source_ref=str(source_ref),
                    updated_at=now,
                )
        finally:
            connection.close()

    def mark_historical(
        self,
        bucket_id: str,
        *,
        status_key: str,
        valid_at: str,
        invalid_at: str,
        source_ref: str,
        superseded_by_bucket_id: str = "",
    ) -> None:
        """Invalidate one audited legacy status snapshot."""
        normalized_valid_at = _normalize_time(valid_at)
        normalized_invalid_at = _normalize_time(invalid_at)
        now = _iso_now()
        connection = self._connect(create=True)
        assert connection is not None
        try:
            with connection:
                self._upsert(
                    connection,
                    bucket_id=str(bucket_id),
                    status_key=str(status_key),
                    state=STATE_HISTORICAL,
                    valid_at=normalized_valid_at,
                    invalid_at=normalized_invalid_at,
                    expired_at=now,
                    superseded_by_bucket_id=str(superseded_by_bucket_id or ""),
                    source_ref=str(source_ref),
                    updated_at=now,
                )
        finally:
            connection.close()

    def mark_supersession(
        self,
        *,
        old_bucket_id: str,
        new_bucket_id: str,
        old_valid_at: str,
        new_valid_at: str,
        source_ref: str,
        status_key: str = "",
    ) -> dict[str, str]:
        """Atomically apply one Graphiti-style temporal supersession.

        A backfilled event older than the already-known current status is
        recorded as historical instead of replacing the newer truth.
        """
        old_id = str(old_bucket_id).strip()
        new_id = str(new_bucket_id).strip()
        if not old_id or not new_id or old_id == new_id:
            raise ValueError("supersession needs two distinct bucket ids")
        old_time = _normalize_time(old_valid_at)
        new_time = _normalize_time(new_valid_at)
        now = _iso_now()
        connection = self._connect(create=True)
        assert connection is not None
        try:
            with connection:
                old_row = connection.execute(
                    "SELECT * FROM operational_status_validity WHERE bucket_id=?",
                    (old_id,),
                ).fetchone()
                key = str(status_key or (old_row["status_key"] if old_row else "") or f"status.{old_id}")
                current_rows = connection.execute(
                    "SELECT * FROM operational_status_validity "
                    "WHERE status_key=? AND state='current'",
                    (key,),
                ).fetchall()
                newer_current = next(
                    (
                        row for row in current_rows
                        if row["bucket_id"] not in {old_id, new_id}
                        and _timestamp_key(row["valid_at"]) > _timestamp_key(new_time)
                    ),
                    None,
                )
                if newer_current is not None:
                    self._upsert(
                        connection,
                        bucket_id=new_id,
                        status_key=key,
                        state=STATE_HISTORICAL,
                        valid_at=new_time,
                        invalid_at=newer_current["valid_at"] or "",
                        expired_at=now,
                        superseded_by_bucket_id=newer_current["bucket_id"],
                        source_ref=source_ref,
                        updated_at=now,
                    )
                    return {"status_key": key, "current_bucket_id": newer_current["bucket_id"]}

                superseded_ids = [old_id]
                for row in current_rows:
                    current_id = row["bucket_id"]
                    if current_id == new_id:
                        continue
                    if current_id not in superseded_ids:
                        superseded_ids.append(current_id)
                    self._upsert(
                        connection,
                        bucket_id=current_id,
                        status_key=key,
                        state=STATE_HISTORICAL,
                        valid_at=row["valid_at"] or "",
                        invalid_at=new_time,
                        expired_at=now,
                        superseded_by_bucket_id=new_id,
                        supersedes_bucket_ids=json.loads(row["supersedes_bucket_ids"] or "[]"),
                        source_ref=source_ref,
                        updated_at=now,
                    )
                self._upsert(
                    connection,
                    bucket_id=old_id,
                    status_key=key,
                    state=STATE_HISTORICAL,
                    valid_at=(old_row["valid_at"] if old_row else old_time) or old_time,
                    invalid_at=new_time,
                    expired_at=now,
                    superseded_by_bucket_id=new_id,
                    supersedes_bucket_ids=(
                        json.loads(old_row["supersedes_bucket_ids"] or "[]")
                        if old_row else ()
                    ),
                    source_ref=source_ref,
                    updated_at=now,
                )
                self._upsert(
                    connection,
                    bucket_id=new_id,
                    status_key=key,
                    state=STATE_CURRENT,
                    valid_at=new_time,
                    supersedes_bucket_ids=superseded_ids,
                    source_ref=source_ref,
                    updated_at=now,
                )
                return {"status_key": key, "current_bucket_id": new_id}
        finally:
            connection.close()
