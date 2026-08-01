"""Deterministic E-axis trigger and LMC-5 candidate source adapter.

The producer reads only the already-redacted LMC-5 candidate ledger.  It never
opens raw conversation events or ordinary/private bucket bodies.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from e_axis_shadow import strict_json_loads


ALWAYS_TRIGGER_TYPES = frozenset({
    "relationship_moment",
    "risk_boundary",
    "preference",
})
NEVER_TRIGGER_TYPES = frozenset({
    "fact",
    "engineering_decision",
})
EMOTION_TRIGGER_KEYWORDS = (
    "崩溃", "气死", "委屈", "心痛", "感动", "爱你", "想你",
    "兴奋", "焦虑", "害怕", "纠结", "绝望", "好难", "受不了",
    "心如灰死", "算了", "放弃", "不想",
    "我们", "答应", "承诺", "再也不", "永远", "永不",
    "矛盾", "冲突", "为难", "撕扯",
    "亲", "抱", "吻", "靠着", "蹭", "哭",
    "love you", "miss you", "panic", "anxious", "furious",
    "exhausted", "betrayed", "heartbreak", "devastated",
    "promise", "swear", "never again", "always",
    "we agreed", "you promised", "between us",
    "conflict", "argued", "fight",
    "kiss", "hug", "hold me", "cry",
)
_CANDIDATE_SCHEMA = "ombre.lmc5-axis-candidate/v1"
_READ_STATUSES = ("pending", "deferred", "ready", "review")
_MACHINE_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")


class EAxisSourceError(ValueError):
    """A persisted candidate violated the source adapter contract."""


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    included: bool
    reason: str


@dataclass(frozen=True, slots=True)
class EAxisSubject:
    source_id: str
    source_kind: str
    source_digest: str
    source_run_id: str
    memory_type: str
    title: str
    content: str
    relation_hints: tuple[object, ...]
    created_at: str
    trigger_reason: str


@dataclass(frozen=True, slots=True)
class EAxisSourceScan:
    subjects: tuple[EAxisSubject, ...]
    scanned: int
    skipped: int
    skip_reasons: tuple[tuple[str, int], ...]

    def skip_reason_counts(self) -> dict[str, int]:
        return dict(self.skip_reasons)


def _has_emotional_link(hints: Iterable[object]) -> bool:
    for hint in hints:
        if hint == "emotional_link":
            return True
        if (
            isinstance(hint, Mapping)
            and hint.get("relation_type") == "emotional_link"
        ):
            return True
    return False


def decide_e_axis_trigger(
    *,
    memory_type: object,
    title: object,
    content: object,
    relation_hints: object = (),
) -> TriggerDecision:
    """Apply the official LMC-5 E trigger with an explicit skip reason."""

    normalized_type = str(memory_type or "").strip()
    if normalized_type in ALWAYS_TRIGGER_TYPES:
        return TriggerDecision(True, f"type.{normalized_type}")

    text = f"{str(title or '')}\n{str(content or '')}".lower()
    keyword = next(
        (
            item
            for item in EMOTION_TRIGGER_KEYWORDS
            if item.lower() in text
        ),
        "",
    )
    if normalized_type in NEVER_TRIGGER_TYPES and not keyword:
        return TriggerDecision(False, f"type.{normalized_type}.no_emotion")
    if keyword:
        return TriggerDecision(True, "keyword.emotion")

    hints = relation_hints if isinstance(relation_hints, (list, tuple)) else ()
    if _has_emotional_link(hints):
        return TriggerDecision(True, "relation.emotional_link")
    return TriggerDecision(False, "gate.no_signal")


def _records(ledger: Any, status: str):
    after = None
    while True:
        try:
            page = ledger.list_candidates(status, limit=1_000, after=after)
        except EAxisSourceError:
            raise
        except Exception as exc:
            raise EAxisSourceError("candidate.source_unavailable") from exc
        if not page:
            return
        yield from page
        after = page[-1].candidate_id


def _source_run_id(value: object) -> str:
    if type(value) is not str or _MACHINE_TEXT_RE.fullmatch(value) is None:
        raise EAxisSourceError("candidate.invalid_source_run")
    return value


def _created_at(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise EAxisSourceError("candidate.invalid_created_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EAxisSourceError("candidate.invalid_created_at") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EAxisSourceError("candidate.invalid_created_at")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def iter_candidate_subjects(
    ledger: Any,
    *,
    statuses: tuple[str, ...] = _READ_STATUSES,
) -> EAxisSourceScan:
    """Return deduplicated official-gate subjects plus scanned/skipped counts.

    Every proposer draft has an X candidate, so consuming only X rows avoids
    duplicate E/M/Z copies while still seeing emotional fact/engineering
    candidates that do not receive a dedicated E route.
    """

    by_base_digest: dict[str, EAxisSubject] = {}
    scanned = 0
    skipped = 0
    skip_reasons: Counter[str] = Counter()
    for status in statuses:
        for record in _records(ledger, status):
            if record.axis != "X":
                continue
            scanned += 1
            try:
                payload = strict_json_loads(record.payload)
            except Exception as exc:
                raise EAxisSourceError("candidate.invalid_json") from exc
            if (
                type(payload) is not dict
                or payload.get("schema") != _CANDIDATE_SCHEMA
                or payload.get("axis") != "X"
                or type(payload.get("draft")) is not dict
                or type(payload.get("source")) is not dict
            ):
                raise EAxisSourceError("candidate.invalid_schema")
            draft = payload["draft"]
            source = payload["source"]
            base_digest = payload.get("base_digest")
            if (
                type(base_digest) is not str
                or len(base_digest) != 64
                or any(char not in "0123456789abcdef" for char in base_digest)
            ):
                raise EAxisSourceError("candidate.invalid_digest")
            memory_type = draft.get("type")
            title = draft.get("title")
            content = draft.get("content")
            hints = draft.get("relation_hints")
            if (
                type(memory_type) is not str
                or type(title) is not str
                or type(content) is not str
                or not content.strip()
                or type(hints) is not list
            ):
                raise EAxisSourceError("candidate.invalid_draft")
            decision = decide_e_axis_trigger(
                memory_type=memory_type,
                title=title,
                content=content,
                relation_hints=hints,
            )
            if not decision.included:
                skipped += 1
                skip_reasons[decision.reason] += 1
                continue
            source_run_id = _source_run_id(payload.get("origin_run_id"))
            created_at = _created_at(source.get("created_at"))
            subject = EAxisSubject(
                source_id="candidate:" + base_digest,
                source_kind="lmc5_candidate",
                source_digest=hashlib.sha256(
                    record.payload
                ).hexdigest(),
                source_run_id=source_run_id,
                memory_type=memory_type,
                title=title,
                content=content,
                relation_hints=tuple(hints),
                created_at=created_at,
                trigger_reason=decision.reason,
            )
            existing = by_base_digest.get(base_digest)
            if existing is not None and existing != subject:
                raise EAxisSourceError("candidate.duplicate_conflict")
            by_base_digest[base_digest] = subject

    return EAxisSourceScan(
        subjects=tuple(
            sorted(
                by_base_digest.values(),
                key=lambda item: (item.created_at, item.source_id),
            )
        ),
        scanned=scanned,
        skipped=skipped,
        skip_reasons=tuple(sorted(skip_reasons.items())),
    )


__all__ = [
    "ALWAYS_TRIGGER_TYPES",
    "EMOTION_TRIGGER_KEYWORDS",
    "EAxisSourceError",
    "EAxisSourceScan",
    "EAxisSubject",
    "NEVER_TRIGGER_TYPES",
    "TriggerDecision",
    "decide_e_axis_trigger",
    "iter_candidate_subjects",
]
