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
    session_id: str = ""
    base_partial: bool = False
    error: str = ""
    skipped_count: int = 0
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
    return "unknown"


def _time_status(metadata: dict) -> str:
    state = (
        metadata.get("validity_state")
        or metadata.get("fact_status")
        or "unknown"
    )
    event_at = _one_line(event_at_from_metadata(metadata) or "")
    date = event_at[:10] if event_at else "undated"
    return _field(f"{state}@{date}", 28)


def _sentences(content: str) -> list[str]:
    text = _one_line(redact_text(strip_wikilinks(content or "")))
    if not text:
        return ["(empty)"]
    sentences = re.findall(
        r"[^。！？!?]+[。！？!?](?:[”’」』】）)]?)?|[^。！？!?]+$",
        text,
    )
    return [sentence.strip() for sentence in sentences if sentence.strip()] or [text]


def _search_terms(value: object) -> tuple[str, ...]:
    text = _one_line(value).casefold()
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_.-]{1,}|[\u3400-\u9fff]+", text):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            if len(token) < 2:
                continue
            terms.add(token)
            for width in range(min(4, len(token)), 1, -1):
                terms.update(
                    token[index:index + width]
                    for index in range(0, len(token) - width + 1)
                )
        else:
            terms.add(token)
    return tuple(sorted(terms, key=lambda term: (-len(term), term)))


def _best_matching_sentence(sentences: list[str], terms: tuple[str, ...]) -> str | None:
    best_sentence = None
    best_score = 0
    for sentence in sentences:
        folded = sentence.casefold()
        score = sum(len(term) * len(term) for term in terms if term in folded)
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence


def _relevant_sentence(content: str, *, query: str = "", match_text: str = "") -> str:
    sentences = _sentences(content)
    query_match = _best_matching_sentence(sentences, _search_terms(query))
    if query_match is not None:
        return query_match
    summary_match = _best_matching_sentence(sentences, _search_terms(match_text))
    return summary_match if summary_match is not None else sentences[0]


def build_signal_entry(
    bucket: dict,
    *,
    reason: str,
    query: str = "",
    match_text: str = "",
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
    sentence = _relevant_sentence(
        str(bucket.get("content") or ""),
        query=query,
        match_text=match_text,
    )
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
        session_id: str = "",
        partial: bool = False,
        error: str = "",
        skipped_count: int = 0,
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
            session_id=str(session_id or "").strip(),
            base_partial=bool(partial),
            error=str(error or "").strip(),
            skipped_count=max(0, int(skipped_count)),
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
        result = {
            "mode": "signal",
            "snapshot_id": snapshot.snapshot_id,
            "entries": [entry.line for entry in page_entries],
            "partial": (
                snapshot.base_partial
                or has_more
                or any(entry.partial for entry in page_entries)
            ),
            "has_more": has_more,
            "next_cursor": self._cursor(snapshot.snapshot_id, stop) if has_more else "",
        }
        if snapshot.error:
            result["error"] = snapshot.error
        if snapshot.skipped_count:
            result["skipped_count"] = snapshot.skipped_count
        return result

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

    def mark_expanded_with_session(
        self,
        snapshot_id: str,
        bucket_id: str,
    ) -> tuple[bool, str]:
        with self._lock:
            try:
                snapshot = self._snapshot_locked(str(snapshot_id or ""))
            except MemorySignalCursorError:
                return False, ""
            member_ids = {entry.bucket_id for entry in snapshot.entries}
            normalized = str(bucket_id or "").strip()
            if normalized not in member_ids:
                return False, ""
            snapshot.expanded.setdefault(normalized, None)
            return True, snapshot.session_id

    def mark_expanded(self, snapshot_id: str, bucket_id: str) -> bool:
        tracked, _session_id = self.mark_expanded_with_session(
            snapshot_id,
            bucket_id,
        )
        return tracked

    def expanded_ids(self, snapshot_id: str) -> tuple[str, ...]:
        with self._lock:
            try:
                snapshot = self._snapshot_locked(str(snapshot_id or ""))
            except MemorySignalCursorError:
                return ()
            return tuple(snapshot.expanded)
