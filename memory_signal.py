"""Compact, immutable pages for opt-in two-stage memory reads.

The store deliberately keeps signal-page state in process memory.  A cursor
continues one frozen result set; it never reruns retrieval or ranking.  Full
bucket text remains authoritative in ``inspect`` and is not copied into read
receipts.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from redact import redact_text
from utils import event_at_from_metadata, strip_wikilinks


SIGNAL_LINE_MAX_CHARS = 120
SIGNAL_DEFAULT_PAGE_SIZE = 5
SIGNAL_MAX_PAGE_SIZE = 20


class MemorySignalCursorError(ValueError):
    """Raised when a signal cursor is malformed, expired, or exhausted."""


@dataclass(frozen=True)
class SignalEntry:
    bucket_id: str
    line: str
    partial: bool


@dataclass
class _Snapshot:
    snapshot_id: str
    created_at: float
    page_size: int
    entries: tuple[SignalEntry, ...]
    expanded: dict[str, None] = field(default_factory=dict)


def _one_line(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _field(value: object, max_chars: int) -> str:
    text = _one_line(value).replace("[", "(").replace("]", ")")
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "…"


def _source_label(metadata: dict) -> str:
    explicit = metadata.get("source") or metadata.get("source_agent")
    if explicit:
        return _field(explicit, 18)
    domains = metadata.get("domain") or []
    if isinstance(domains, str):
        domains = [domains]
    if isinstance(domains, list) and domains:
        return _field(domains[0], 18)
    return _field(metadata.get("type") or "bucket", 18)


def _time_status(metadata: dict) -> str:
    state = (
        metadata.get("validity_state")
        or metadata.get("fact_status")
        or "unknown"
    )
    event_at = _one_line(event_at_from_metadata(metadata) or "")
    date = event_at[:10] if event_at else "undated"
    return _field(f"{state}@{date}", 28)


def _first_sentence(content: str) -> str:
    text = _one_line(redact_text(strip_wikilinks(content or "")))
    if not text:
        return "(empty)"
    match = re.match(r".*?[。！？!?](?:[”’」』】）)]?)(?=.|$)", text)
    return match.group(0) if match else text


def build_signal_entry(
    bucket: dict,
    *,
    reason: str,
    max_chars: int = SIGNAL_LINE_MAX_CHARS,
) -> SignalEntry:
    """Build one source-derived signal line without summarising or rewriting."""
    bucket_id = _one_line(bucket.get("id"))
    if not bucket_id:
        raise ValueError("signal entry requires bucket id")
    metadata = bucket.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    prefix = (
        f"[id:{bucket_id}]"
        f"[src:{_source_label(metadata)}]"
        f"[time:{_time_status(metadata)}]"
        f"[why:{_field(reason or 'ranked', 16)}]"
    )
    sentence = _first_sentence(str(bucket.get("content") or ""))
    full_content = _one_line(
        redact_text(strip_wikilinks(str(bucket.get("content") or "")))
    )
    if len(prefix) + len(sentence) + 2 <= max_chars and sentence == full_content:
        return SignalEntry(
            bucket_id=bucket_id,
            line=f"{prefix}「{sentence}」",
            partial=False,
        )

    marker = "[partial]"
    available = max_chars - len(prefix) - len(marker) - 2
    if available < 1:
        raise ValueError("bucket id and signal metadata exceed line budget")
    snippet = sentence[:available]
    line = f"{prefix}「{snippet}」{marker}"
    return SignalEntry(bucket_id=bucket_id, line=line, partial=True)


class MemorySignalStore:
    """Bounded in-memory store for immutable signal pages and read receipts."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 15 * 60,
        max_snapshots: int = 128,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.max_snapshots = max(1, int(max_snapshots))
        self._clock = clock
        self._snapshots: dict[str, _Snapshot] = {}
        self._lock = threading.RLock()

    def _prune_locked(self, now: float) -> None:
        expired = [
            snapshot_id
            for snapshot_id, snapshot in self._snapshots.items()
            if now - snapshot.created_at > self.ttl_seconds
        ]
        for snapshot_id in expired:
            self._snapshots.pop(snapshot_id, None)
        overflow = len(self._snapshots) - self.max_snapshots
        if overflow > 0:
            oldest = sorted(
                self._snapshots.values(),
                key=lambda snapshot: snapshot.created_at,
            )[:overflow]
            for snapshot in oldest:
                self._snapshots.pop(snapshot.snapshot_id, None)

    def create(
        self,
        entries: Iterable[SignalEntry],
        *,
        page_size: int = SIGNAL_DEFAULT_PAGE_SIZE,
    ) -> dict:
        size = max(1, min(int(page_size), SIGNAL_MAX_PAGE_SIZE))
        deduped: list[SignalEntry] = []
        seen: set[str] = set()
        for entry in entries:
            if entry.bucket_id in seen:
                continue
            seen.add(entry.bucket_id)
            deduped.append(entry)
        now = self._clock()
        snapshot_id = secrets.token_urlsafe(12)
        snapshot = _Snapshot(
            snapshot_id=snapshot_id,
            created_at=now,
            page_size=size,
            entries=tuple(deduped),
        )
        with self._lock:
            self._prune_locked(now)
            self._snapshots[snapshot_id] = snapshot
            self._prune_locked(now)
            return self._render_locked(snapshot, offset=0)

    def _snapshot_locked(self, snapshot_id: str) -> _Snapshot:
        now = self._clock()
        self._prune_locked(now)
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None:
            raise MemorySignalCursorError("snapshot_expired_or_unknown")
        return snapshot

    @staticmethod
    def _cursor(snapshot_id: str, offset: int) -> str:
        return f"{snapshot_id}:{offset}"

    def _render_locked(self, snapshot: _Snapshot, *, offset: int) -> dict:
        if offset < 0 or (snapshot.entries and offset >= len(snapshot.entries)):
            raise MemorySignalCursorError("cursor_offset_out_of_range")
        stop = min(len(snapshot.entries), offset + snapshot.page_size)
        page_entries = snapshot.entries[offset:stop]
        has_more = stop < len(snapshot.entries)
        return {
            "mode": "signal",
            "snapshot_id": snapshot.snapshot_id,
            "entries": [entry.line for entry in page_entries],
            "partial": has_more or any(entry.partial for entry in page_entries),
            "has_more": has_more,
            "next_cursor": self._cursor(snapshot.snapshot_id, stop) if has_more else "",
        }

    def page(self, cursor: str) -> dict:
        try:
            snapshot_id, raw_offset = str(cursor or "").rsplit(":", 1)
            offset = int(raw_offset)
        except (TypeError, ValueError) as exc:
            raise MemorySignalCursorError("invalid_cursor") from exc
        if not snapshot_id:
            raise MemorySignalCursorError("invalid_cursor")
        with self._lock:
            snapshot = self._snapshot_locked(snapshot_id)
            return self._render_locked(snapshot, offset=offset)

    def mark_expanded(self, snapshot_id: str, bucket_id: str) -> bool:
        with self._lock:
            try:
                snapshot = self._snapshot_locked(str(snapshot_id or ""))
            except MemorySignalCursorError:
                return False
            member_ids = {entry.bucket_id for entry in snapshot.entries}
            normalized = str(bucket_id or "").strip()
            if normalized not in member_ids:
                return False
            snapshot.expanded.setdefault(normalized, None)
            return True

    def expanded_ids(self, snapshot_id: str) -> tuple[str, ...]:
        with self._lock:
            try:
                snapshot = self._snapshot_locked(str(snapshot_id or ""))
            except MemorySignalCursorError:
                return ()
            return tuple(snapshot.expanded)
