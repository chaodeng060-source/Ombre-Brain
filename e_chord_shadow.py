"""Text-free E-chord proposal and private shadow/final receipt ledgers.

The chord is never a retrieval channel.  It may only propose an adjacent swap
between already-retained, primary-authored E memories that share either a
strong persisted event lock or, when explicitly enabled, a source-verifiable
derived lock, and are already a near tie under the existing E rank.  Callers
keep serving arm B unless a separately gated, content-bearing delivery is
validated downstream. This module itself remains text-free and zero-call.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from e_axis_storage import open_secure_e_axis_jsonl, secure_e_axis_lock


LIVE_SCHEMA = "live_chord.v1"
LEGACY_RECEIPT_SCHEMA = "e_chord_shadow_receipt.v1"
RECEIPT_SCHEMA = "e_chord_shadow_receipt.v2"
BYPASS_RECEIPT_SCHEMA = "e_chord_shadow_receipt.v3"
LEGACY_FINAL_SELECTION_SCHEMA = "e_chord_final_selection.v1"
FINAL_SELECTION_SCHEMA = "e_chord_final_selection.v2"
BYPASS_FINAL_SELECTION_SCHEMA = "e_chord_final_selection.v3"
LIVE_BYPASS_FINAL_SELECTION_SCHEMA = "e_chord_final_selection.v4"
MAX_FACETS = 2
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_KEYS = frozenset({
    "schema",
    "turn_id",
    "agent_id",
    "e_authored_by",
    "session_scope",
    "source_turn_digest",
    "captured_at_ms",
    "facets",
})

# This is the reviewed primary-author identity contract used by the current E
# write path.  It must not be guessed from the transport id: Claude routes as
# ``claude`` while its authoritative E records are authored as ``哥哥``.
_E_AUTHOR_BY_AGENT: Mapping[str, str] = {
    "claude": "哥哥",
}
_FACET_KEYS = frozenset({
    "motivation",
    "drive_key",
    "tendency",
    "hunger",
    "salience",
})

# This is the receiver-side copy of Twin's versioned projection contract.  A
# mismatch rejects the projection; Ombre never guesses or remaps live state.
MOTIVATION_TENDENCY: Mapping[str, str] = {
    "explore": "engage",
    "settle": "withdraw",
    "attach": "comfort",
    "social": "engage",
    "imprint": "engage",
    "anchor": "comfort",
    "reminisce": "comfort",
    "selfcheck": "alert",
    "innovate": "engage",
    "collab": "engage",
    "rest": "withdraw",
}
MOTIVATION_DRIVES: Mapping[str, frozenset[str]] = {
    "explore": frozenset({"curiosity"}),
    "settle": frozenset({"reflection"}),
    "attach": frozenset({
        "attachment",
        "libido",
        "stress",
        "possessiveness",
        "protectiveness",
        "fear_separation",
    }),
    "social": frozenset({"social"}),
    "imprint": frozenset({"reflection", "duty"}),
    "anchor": frozenset({"attachment", "reflection", "jealousy"}),
    "reminisce": frozenset({"attachment", "reflection"}),
    "selfcheck": frozenset({"curiosity", "duty"}),
    "innovate": frozenset({"curiosity", "reflection"}),
    "collab": frozenset({"duty", "curiosity"}),
    "rest": frozenset({"fatigue"}),
}
_FACT_MARKERS = frozenset({
    "fact_key",
    "fact_status",
    "fact_value",
    "validity_kind",
    "validity_state",
    "superseded_by_bucket_id",
    "z_fact_key",
})
_FROZEN_BAND_KEY = "_e_chord_relevance_band_id"
_E_ADMISSIBILITY_FLOOR = 0.55
_E_POLARITY_NEUTRAL_BAND = 0.25
_E_RESONANCE_MAX_REGRESSION_MILLI = 30
_E_ADMISSIBILITY_VALUES = frozenset({
    "missing_annotation",
    "below_resonance",
    "opposite_affect",
    "admissible",
})
_DS_DECISION_SOURCES = frozenset({
    "disabled",
    "deterministic_noop",
    "model",
    "fallback",
    "unobserved",
})


def _finite(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit(value: object) -> float | None:
    number = _finite(value)
    if number is None or not 0.0 <= number <= 1.0:
        return None
    return number


def _safe_id(value: object) -> str | None:
    if type(value) is not str or value != value.strip():
        return None
    return value if _SAFE_ID_RE.fullmatch(value) else None


def session_scope_digest(session_id: object) -> str:
    """Return Twin's receiver-verifiable opaque scope for one session."""

    canonical = str(session_id or "").strip()[:512]
    material = "e-chord-session-v1\0" + canonical
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _safe_author(value: object) -> str | None:
    if type(value) is not str or value != value.strip():
        return None
    if not 1 <= len(value) <= 80:
        return None
    if any(character.isspace() or ord(character) < 32 for character in value):
        return None
    return value


@dataclass(frozen=True, slots=True)
class LiveChordFacet:
    motivation: str
    drive_key: str
    tendency: str
    hunger: float
    salience: float


@dataclass(frozen=True, slots=True)
class LiveChord:
    turn_id: str
    agent_id: str
    e_authored_by: str
    session_scope: str
    source_turn_digest: str
    captured_at_ms: int
    facets: tuple[LiveChordFacet, ...]


@dataclass(frozen=True, slots=True)
class EChordShadowConfig:
    enabled: bool = False
    near_tie_epsilon: float = 0.03
    max_age_ms: int = 30_000
    derived_lock_enabled: bool = False
    bypass_enabled: bool = False
    bypass_limit: int = 4


def load_e_chord_shadow_config(root: Mapping[str, object]) -> EChordShadowConfig:
    raw = root.get("e_chord_shadow", {}) if isinstance(root, Mapping) else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("e_chord_shadow must be a mapping")
    if raw.get("enabled") is not True or raw.get("mode") != "shadow":
        return EChordShadowConfig()
    epsilon = _finite(raw.get("near_tie_epsilon", 0.03))
    if epsilon is None or not 0.0 <= epsilon <= 0.1:
        raise ValueError("e_chord_shadow.near_tie_epsilon must be in [0, 0.1]")
    max_age_ms = raw.get("max_age_ms", 30_000)
    if type(max_age_ms) is not int or not 1_000 <= max_age_ms <= 120_000:
        raise ValueError("e_chord_shadow.max_age_ms must be in [1000, 120000]")
    bypass_limit = raw.get("bypass_limit", 4)
    if (
        type(bypass_limit) is not int
        or isinstance(bypass_limit, bool)
        or not 1 <= bypass_limit <= 8
    ):
        raise ValueError("e_chord_shadow.bypass_limit must be in [1, 8]")
    return EChordShadowConfig(
        enabled=True,
        near_tie_epsilon=epsilon,
        max_age_ms=max_age_ms,
        derived_lock_enabled=os.environ.get(
            "OMBRE_E_CHORD_DERIVED_LOCK", "0"
        ).strip().lower() in {"1", "true", "yes", "on"},
        bypass_enabled=os.environ.get(
            "OMBRE_E_CHORD_BYPASS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"},
        bypass_limit=bypass_limit,
    )


def parse_live_chord(
    raw: object,
    *,
    expected_turn_id: str,
    expected_agent_id: str,
    expected_session_id: str,
    now_ms: int,
    max_age_ms: int,
    max_future_ms: int = 5_000,
) -> tuple[LiveChord | None, str]:
    """Strictly bind one projection to this request and current time window."""

    if type(raw) is not dict or set(raw) != _TOP_KEYS:
        return None, "schema.keys"
    if raw.get("schema") != LIVE_SCHEMA:
        return None, "schema.version"
    turn_id = _safe_id(raw.get("turn_id"))
    agent_id = _safe_id(raw.get("agent_id"))
    if turn_id is None:
        return None, "schema.turn_id"
    if agent_id is None:
        return None, "schema.agent_id"
    if turn_id != expected_turn_id:
        return None, "scope.turn"
    if agent_id != expected_agent_id:
        return None, "scope.agent"
    author = _safe_author(raw.get("e_authored_by"))
    if author is None:
        return None, "schema.e_authored_by"
    if author != _E_AUTHOR_BY_AGENT.get(agent_id):
        return None, "scope.author"
    session_scope = raw.get("session_scope")
    if type(session_scope) is not str or _HEX32_RE.fullmatch(session_scope) is None:
        return None, "schema.session_scope"
    if session_scope != session_scope_digest(expected_session_id):
        return None, "scope.session"
    source_turn_digest = raw.get("source_turn_digest")
    if (
        type(source_turn_digest) is not str
        or _HEX64_RE.fullmatch(source_turn_digest) is None
    ):
        return None, "schema.source_turn_digest"
    captured = raw.get("captured_at_ms")
    if type(captured) is not int or captured < 0:
        return None, "schema.captured_at_ms"
    if type(now_ms) is not int or now_ms < 0:
        return None, "clock.invalid"
    if captured > now_ms + max_future_ms:
        return None, "schema.future"
    if captured < now_ms - max_age_ms:
        return None, "schema.stale"

    raw_facets = raw.get("facets")
    if type(raw_facets) is not list or len(raw_facets) > MAX_FACETS:
        return None, "schema.facets"
    facets: list[LiveChordFacet] = []
    seen_motivations: set[str] = set()
    previous_salience = float("inf")
    for value in raw_facets:
        if type(value) is not dict or set(value) != _FACET_KEYS:
            return None, "schema.facet_keys"
        motivation = _safe_id(value.get("motivation"))
        drive_key = _safe_id(value.get("drive_key"))
        tendency = _safe_id(value.get("tendency"))
        hunger = _unit(value.get("hunger"))
        salience = _unit(value.get("salience"))
        if motivation not in MOTIVATION_TENDENCY:
            return None, "schema.motivation"
        if motivation in seen_motivations:
            return None, "schema.duplicate_motivation"
        if drive_key not in MOTIVATION_DRIVES[motivation]:
            return None, "schema.drive_key"
        if tendency != MOTIVATION_TENDENCY[motivation]:
            return None, "schema.mapping"
        if hunger is None:
            return None, "schema.hunger"
        if salience is None:
            return None, "schema.salience"
        if salience > previous_salience:
            return None, "schema.order"
        previous_salience = salience
        seen_motivations.add(motivation)
        facets.append(LiveChordFacet(
            motivation=motivation,
            drive_key=drive_key,
            tendency=tendency,
            hunger=hunger,
            salience=salience,
        ))
    return LiveChord(
        turn_id=turn_id,
        agent_id=agent_id,
        e_authored_by=author,
        session_scope=session_scope,
        source_turn_digest=source_turn_digest,
        captured_at_ms=captured,
        facets=tuple(facets),
    ), "accepted"


@dataclass(frozen=True, slots=True)
class ShadowSwap:
    promoted_id: str
    demoted_id: str
    from_index: int
    to_index: int
    event_lock_digest: str
    lock_kind: str = "strong"
    source_bucket_ids: tuple[str, ...] = ()
    derived_lock_basis: str = ""
    relation_type: str = ""
    relation_from_id: str = ""
    relation_to_id: str = ""
    recorded_day: str = ""
    domain_digest: str = ""


@dataclass(frozen=True, slots=True)
class ShadowProposal:
    b_ids: tuple[str, ...]
    c_ids: tuple[str, ...]
    swaps: tuple[ShadowSwap, ...]
    skipped_reasons: tuple[str, ...]
    violations: tuple[str, ...]


def _candidate_id(candidate: Mapping[str, object]) -> str:
    value = candidate.get("id")
    return value if type(value) is str else ""


def _metadata(candidate: Mapping[str, object]) -> Mapping[str, object]:
    value = candidate.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _event_locks(candidate: Mapping[str, object]) -> frozenset[str]:
    metadata = _metadata(candidate)
    locks: set[str] = set()
    for key in ("event_id", "episode_id", "e_source_bucket_id"):
        value = _safe_id(metadata.get(key))
        if value is not None:
            locks.add(f"{key}:{value}")
    return frozenset(locks)


def _event_lock_digests(candidate: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(sorted(
        hashlib.sha256(lock.encode("utf-8")).hexdigest()[:16]
        for lock in _event_locks(candidate)
    ))


@dataclass(frozen=True, slots=True)
class _SelectedEventLock:
    canonical: str
    kind: str
    source_bucket_ids: tuple[str, ...] = ()
    derived_lock_basis: str = ""
    relation_type: str = ""
    relation_from_id: str = ""
    relation_to_id: str = ""
    recorded_day: str = ""
    domain_digest: str = ""


def _candidate_source_bucket_id(candidate: Mapping[str, object]) -> str:
    return _safe_id(_metadata(candidate).get("e_source_bucket_id")) or ""


def _metadata_domains(bucket: Mapping[str, object]) -> frozenset[str]:
    raw = _metadata(bucket).get("domain", ())
    values: Sequence[object]
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = tuple(raw)
    else:
        values = ()
    return frozenset(
        normalized
        for value in values
        if (normalized := str(value or "").strip().lower())
    )


def _recorded_day(bucket: Mapping[str, object]) -> str:
    value = _metadata(bucket).get("recorded_at")
    if not isinstance(value, str) or len(value) < 10:
        return ""
    day = value[:10]
    try:
        # Keep the receipt content-free while rejecting malformed calendar days.
        date.fromisoformat(day)
    except ValueError:
        return ""
    return day


def _direct_deterministic_relation(
    left_source: Mapping[str, object],
    right_source: Mapping[str, object],
    *,
    allowed_relation_types: frozenset[str],
    min_strength: float,
) -> tuple[str, str, str] | None:
    left_id = _safe_id(left_source.get("id"))
    right_id = _safe_id(right_source.get("id"))
    if left_id is None or right_id is None or left_id == right_id:
        return None
    matches: list[tuple[str, str, str]] = []
    for source_id, target_id, bucket in (
        (left_id, right_id, left_source),
        (right_id, left_id, right_source),
    ):
        raw_relations = _metadata(bucket).get("relations", ())
        if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, str):
            continue
        for edge in raw_relations:
            if not isinstance(edge, Mapping):
                continue
            relation_type = str(edge.get("type") or "").strip()
            strength = _finite(edge.get("strength"))
            generation_method = str(edge.get("generation_method") or "").strip()
            if (
                _safe_id(edge.get("target")) == target_id
                and relation_type in allowed_relation_types
                and strength is not None
                and strength >= min_strength
                and generation_method.startswith("deterministic:")
            ):
                matches.append((relation_type, source_id, target_id))
    return min(matches) if matches else None


def _derived_event_lock(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    source_buckets_by_id: Mapping[str, Mapping[str, object]],
    allowed_relation_types: frozenset[str],
    relation_min_strength: float,
) -> _SelectedEventLock | None:
    left_source_id = _candidate_source_bucket_id(left)
    right_source_id = _candidate_source_bucket_id(right)
    if (
        not left_source_id
        or not right_source_id
        or left_source_id == right_source_id
    ):
        return None
    left_source = source_buckets_by_id.get(left_source_id)
    right_source = source_buckets_by_id.get(right_source_id)
    if not isinstance(left_source, Mapping) or not isinstance(right_source, Mapping):
        return None
    source_ids = tuple(sorted((left_source_id, right_source_id)))
    relation = _direct_deterministic_relation(
        left_source,
        right_source,
        allowed_relation_types=allowed_relation_types,
        min_strength=relation_min_strength,
    )
    if relation is not None:
        relation_type, relation_from_id, relation_to_id = relation
        canonical = (
            f"derived:relation:{relation_type}:{relation_from_id}:{relation_to_id}"
        )
        return _SelectedEventLock(
            canonical=canonical,
            kind="derived",
            source_bucket_ids=source_ids,
            derived_lock_basis="relation",
            relation_type=relation_type,
            relation_from_id=relation_from_id,
            relation_to_id=relation_to_id,
        )

    left_day = _recorded_day(left_source)
    right_day = _recorded_day(right_source)
    shared_domains = sorted(
        _metadata_domains(left_source) & _metadata_domains(right_source)
    )
    if not left_day or left_day != right_day or not shared_domains:
        return None
    domain_digest = hashlib.sha256(
        shared_domains[0].encode("utf-8")
    ).hexdigest()[:16]
    canonical = f"derived:same-day-domain:{left_day}:{domain_digest}"
    return _SelectedEventLock(
        canonical=canonical,
        kind="derived",
        source_bucket_ids=source_ids,
        derived_lock_basis="same_day_domain",
        recorded_day=left_day,
        domain_digest=domain_digest,
    )


def _select_event_lock(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    derived_lock_enabled: bool,
    source_buckets_by_id: Mapping[str, Mapping[str, object]],
    allowed_relation_types: frozenset[str],
    relation_min_strength: float,
) -> _SelectedEventLock | None:
    shared_strong_locks = sorted(_event_locks(left) & _event_locks(right))
    if shared_strong_locks:
        return _SelectedEventLock(
            canonical=shared_strong_locks[0],
            kind="strong",
        )
    if not derived_lock_enabled:
        return None
    return _derived_event_lock(
        left,
        right,
        source_buckets_by_id=source_buckets_by_id,
        allowed_relation_types=allowed_relation_types,
        relation_min_strength=relation_min_strength,
    )


def _derived_lock_canonical(
    *,
    source_bucket_ids: object,
    derived_lock_basis: object,
    relation_type: object,
    relation_from_id: object,
    relation_to_id: object,
    recorded_day: object,
    domain_digest: object,
) -> str | None:
    if (
        type(source_bucket_ids) not in (list, tuple)
        or len(source_bucket_ids) != 2
        or any(_safe_id(value) is None for value in source_bucket_ids)
        or list(source_bucket_ids) != sorted(set(source_bucket_ids))
    ):
        return None
    source_set = set(source_bucket_ids)
    if derived_lock_basis == "relation":
        if (
            _safe_id(relation_type) is None
            or _safe_id(relation_from_id) is None
            or _safe_id(relation_to_id) is None
            or {relation_from_id, relation_to_id} != source_set
            or recorded_day != ""
            or domain_digest != ""
        ):
            return None
        return (
            f"derived:relation:{relation_type}:"
            f"{relation_from_id}:{relation_to_id}"
        )
    if derived_lock_basis == "same_day_domain":
        if (
            relation_type != ""
            or relation_from_id != ""
            or relation_to_id != ""
            or type(recorded_day) is not str
            or type(domain_digest) is not str
            or _HEX16_RE.fullmatch(domain_digest) is None
        ):
            return None
        try:
            date.fromisoformat(recorded_day)
        except ValueError:
            return None
        return f"derived:same-day-domain:{recorded_day}:{domain_digest}"
    return None


def _is_factual(candidate: Mapping[str, object]) -> bool:
    metadata = _metadata(candidate)
    if any(marker in metadata for marker in _FACT_MARKERS):
        return True
    memory_type = str(metadata.get("type") or "").strip().lower()
    return memory_type in {"fact", "profile", "preference", "status_fact"}


def _annotation_value(candidate: Mapping[str, object], name: str) -> object:
    annotation = candidate.get("_e_axis_annotation")
    if isinstance(annotation, Mapping):
        if name in annotation:
            return annotation.get(name)
        if name == "authored_by":
            return annotation.get("e_authored_by")
        return None
    return getattr(annotation, name, None)


def _affinity(candidate: Mapping[str, object], chord: LiveChord) -> float:
    tendency = _annotation_value(candidate, "response_tendency")
    confidence = _unit(_annotation_value(candidate, "confidence"))
    if type(tendency) is not str or confidence is None:
        return 0.0
    return max(
        (
            facet.salience * confidence
            for facet in chord.facets
            if facet.tendency == tendency
        ),
        default=0.0,
    )


def _base_score(candidate: Mapping[str, object]) -> float | None:
    return _finite(candidate.get("_non_relevance_tie_break_score"))


def _polarity(value: object) -> int:
    number = _finite(value)
    if number is None or not -1.0 <= number <= 1.0:
        return 0
    if abs(number) < _E_POLARITY_NEUTRAL_BAND:
        return 0
    return 1 if number > 0 else -1


def _e_admissibility_status(
    *,
    has_e_annotation: bool,
    e_resonance_milli: int,
    e_resonance_floor_milli: int,
    input_polarity: int,
    experience_polarity: int,
) -> str:
    if not has_e_annotation:
        return "missing_annotation"
    if e_resonance_milli < e_resonance_floor_milli:
        return "below_resonance"
    if (
        input_polarity != 0
        and experience_polarity != 0
        and input_polarity != experience_polarity
    ):
        return "opposite_affect"
    return "admissible"


def _candidate_e_guard(candidate: Mapping[str, object]) -> dict[str, object]:
    """Return the text-free, independently recomputable existing-E guard."""

    annotation_valence = _finite(_annotation_value(candidate, "valence"))
    query_valence = _finite(candidate.get("_e_axis_query_valence"))
    resonance = _finite(candidate.get("_e_axis_resonance"))
    floor = _finite(candidate.get("_e_axis_admissibility_floor"))
    usable = (
        annotation_valence is not None
        and -1.0 <= annotation_valence <= 1.0
        and query_valence is not None
        and -1.0 <= query_valence <= 1.0
        and resonance is not None
        and 0.0 <= resonance <= 1.0
        and floor is not None
        and _E_ADMISSIBILITY_FLOOR <= floor <= 1.0
    )
    resonance_milli = round((resonance if resonance is not None else 0.0) * 1000)
    floor_milli = round(
        max(_E_ADMISSIBILITY_FLOOR, floor if floor is not None else 0.0) * 1000
    )
    resonance_milli = max(0, min(1000, resonance_milli))
    floor_milli = max(round(_E_ADMISSIBILITY_FLOOR * 1000), min(1000, floor_milli))
    input_polarity = _polarity(query_valence)
    experience_polarity = _polarity(annotation_valence)
    return {
        "has_e_annotation": usable,
        "e_resonance_milli": resonance_milli,
        "e_resonance_floor_milli": floor_milli,
        "input_polarity": input_polarity,
        "experience_polarity": experience_polarity,
        "e_admissibility": _e_admissibility_status(
            has_e_annotation=usable,
            e_resonance_milli=resonance_milli,
            e_resonance_floor_milli=floor_milli,
            input_polarity=input_polarity,
            experience_polarity=experience_polarity,
        ),
    }


def _recorded_at_epoch(candidate: Mapping[str, object]) -> float:
    value = _metadata(candidate).get("recorded_at")
    if not isinstance(value, str) or not value.strip():
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def select_bypass_candidates(
    e_candidates: Sequence[Mapping[str, object]],
    natural_b_candidates: Sequence[Mapping[str, object]],
    *,
    source_buckets_by_id: Mapping[str, Mapping[str, object]] | None = None,
    limit: int = 4,
) -> list[Mapping[str, object]]:
    """Select a bounded, source-bound E suffix for shadow and gated delivery."""

    if type(limit) is not int or isinstance(limit, bool) or not 1 <= limit <= 8:
        raise ValueError("bypass limit must be in [1, 8]")
    natural_ids = {
        bucket_id
        for candidate in natural_b_candidates
        if (bucket_id := _candidate_id(candidate))
    }
    known_source_ids = natural_ids | set(source_buckets_by_id or {})
    eligible: list[tuple[int, float, str, Mapping[str, object]]] = []
    for candidate in e_candidates:
        bucket_id = _candidate_id(candidate)
        source_bucket_id = _candidate_source_bucket_id(candidate)
        priority = _annotation_value(candidate, "initial_priority")
        if (
            not bucket_id
            or bucket_id in natural_ids
            or not source_bucket_id
            or source_bucket_id not in known_source_ids
            or type(priority) is not int
            or isinstance(priority, bool)
            or not 1 <= priority <= 100
            or _candidate_e_guard(candidate)["e_admissibility"] != "admissible"
        ):
            continue
        eligible.append((
            priority,
            _recorded_at_epoch(candidate),
            bucket_id,
            candidate,
        ))
    eligible.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in eligible[:limit]]


def frozen_relevance_band_ids(
    candidates: Sequence[Mapping[str, object]],
    band_width: float,
) -> dict[str, int]:
    """Return leader-anchored band IDs before any subtractive downstream gate.

    Callers freeze these IDs on the original relevance-ranked pool.  A later
    DS/session filter may remove the band leader, so recomputing bands on the
    retained subset would be unsafe: two originally distinct bands could then
    collapse into one and permit a cross-relevance chord move.
    """

    width = _finite(band_width)
    if width is None or not 0.0 <= width <= 1.0:
        raise ValueError("band_width must be in [0, 1]")

    decorated = [
        (candidate, _finite(candidate.get("_fused_relevance_score")) or 0.0, index)
        for index, candidate in enumerate(candidates)
    ]
    decorated.sort(key=lambda row: (-row[1], row[2]))
    bands: dict[str, int] = {}
    cursor = 0
    band_id = 0
    while cursor < len(decorated):
        leader_score = decorated[cursor][1]
        end = cursor + 1
        while end < len(decorated) and leader_score - decorated[end][1] <= width:
            end += 1
        for candidate, _score, _index in decorated[cursor:end]:
            bands[_candidate_id(candidate)] = band_id
        cursor = end
        band_id += 1
    return bands


def _frozen_band(candidate: Mapping[str, object]) -> int | None:
    value = candidate.get(_FROZEN_BAND_KEY)
    if type(value) is not int or value < 0:
        return None
    return value


def rank_within_frozen_relevance_bands(
    candidates: Sequence[Mapping[str, object]],
    *,
    score_key: str,
) -> list[Mapping[str, object]]:
    """Rerank contiguous members without ever crossing their frozen band."""

    rows = list(candidates)
    result: list[Mapping[str, object]] = []
    cursor = 0
    while cursor < len(rows):
        band = _frozen_band(rows[cursor])
        end = cursor + 1
        while end < len(rows) and _frozen_band(rows[end]) == band:
            end += 1
        segment = rows[cursor:end]
        if band is not None:
            segment = sorted(
                segment,
                key=lambda candidate: _finite(candidate.get(score_key)) or 0.0,
                reverse=True,
            )
        result.extend(segment)
        cursor = end
    return result


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def propose_chord_reorder(
    candidates: Sequence[Mapping[str, object]],
    chord: LiveChord,
    *,
    near_tie_epsilon: float = 0.03,
    bypass_ids: Sequence[str] = (),
    derived_lock_enabled: bool = False,
    source_buckets_by_id: Mapping[str, Mapping[str, object]] | None = None,
    allowed_relation_types: Sequence[str] = ("explains",),
    relation_min_strength: float = 0.4,
) -> ShadowProposal:
    """Propose bounded C ordering without mutating B or serving the result."""

    epsilon = _finite(near_tie_epsilon)
    if epsilon is None or not 0.0 <= epsilon <= 0.1:
        raise ValueError("near_tie_epsilon must be in [0, 0.1]")
    relation_floor = _finite(relation_min_strength)
    if relation_floor is None or not 0.0 <= relation_floor <= 1.0:
        raise ValueError("relation_min_strength must be in [0, 1]")
    relation_types = frozenset(
        relation_type
        for value in allowed_relation_types
        if (relation_type := _safe_id(value)) is not None
    )
    source_lookup = source_buckets_by_id or {}
    original = list(candidates)
    b_ids = tuple(_candidate_id(candidate) for candidate in original)
    bypass_tuple = tuple(bypass_ids)
    bypass_set = set(bypass_tuple)
    violations: list[str] = []
    if any(not bucket_id for bucket_id in b_ids):
        violations.append("invalid_candidate_id")
    if len(set(b_ids)) != len(b_ids):
        violations.append("duplicate_candidate_id")
    if (
        any(_safe_id(bucket_id) is None for bucket_id in bypass_tuple)
        or len(bypass_set) != len(bypass_tuple)
        or len(bypass_tuple) > len(b_ids)
        or (bypass_tuple and tuple(b_ids[-len(bypass_tuple):]) != bypass_tuple)
    ):
        violations.append("bypass_boundary")
    if violations or len(original) < 2 or not chord.facets:
        return ShadowProposal(
            b_ids=b_ids,
            c_ids=b_ids,
            swaps=(),
            skipped_reasons=("no_facets",) if not chord.facets else (),
            violations=tuple(violations),
        )

    proposed = list(original)
    used_locks: set[str] = set()
    moved_ids: set[str] = set()
    swaps: list[ShadowSwap] = []
    skipped: list[str] = []
    index = 0
    while index + 1 < len(proposed):
        left = proposed[index]
        right = proposed[index + 1]
        left_id = _candidate_id(left)
        right_id = _candidate_id(right)
        if (left_id in bypass_set) != (right_id in bypass_set):
            _append_reason(skipped, "bypass_boundary")
            index += 1
            continue
        selected_lock = _select_event_lock(
            left,
            right,
            derived_lock_enabled=derived_lock_enabled,
            source_buckets_by_id=source_lookup,
            allowed_relation_types=relation_types,
            relation_min_strength=relation_floor,
        )
        if selected_lock is None:
            _append_reason(skipped, "event_lock")
            index += 1
            continue
        lock = selected_lock.canonical
        left_band = _frozen_band(left)
        right_band = _frozen_band(right)
        if left_band is None or right_band is None or left_band != right_band:
            _append_reason(skipped, "relevance_band")
            index += 1
            continue
        if lock in used_locks or left_id in moved_ids or right_id in moved_ids:
            _append_reason(skipped, "swap_budget")
            index += 1
            continue
        left_author = _annotation_value(left, "authored_by")
        right_author = _annotation_value(right, "authored_by")
        if (
            left_author != chord.e_authored_by
            or right_author != chord.e_authored_by
        ):
            _append_reason(skipped, "author")
            index += 1
            continue
        if _is_factual(left) or _is_factual(right):
            _append_reason(skipped, "factual")
            index += 1
            continue
        left_e_guard = _candidate_e_guard(left)
        right_e_guard = _candidate_e_guard(right)
        if (
            left_e_guard["e_admissibility"] != "admissible"
            or right_e_guard["e_admissibility"] != "admissible"
        ):
            _append_reason(skipped, "e_admissibility")
            index += 1
            continue
        if (
            int(right_e_guard["e_resonance_milli"])
            + _E_RESONANCE_MAX_REGRESSION_MILLI
            < int(left_e_guard["e_resonance_milli"])
        ):
            _append_reason(skipped, "e_resonance")
            index += 1
            continue
        left_score = _base_score(left)
        right_score = _base_score(right)
        if (
            left_score is None
            or right_score is None
            or abs(left_score - right_score) > epsilon
        ):
            _append_reason(skipped, "near_tie")
            index += 1
            continue
        left_affinity = _affinity(left, chord)
        right_affinity = _affinity(right, chord)
        if right_affinity <= 0.0 or right_affinity <= left_affinity:
            _append_reason(skipped, "no_positive_gain")
            index += 1
            continue

        proposed[index], proposed[index + 1] = right, left
        used_locks.add(lock)
        moved_ids.update((left_id, right_id))
        swaps.append(ShadowSwap(
            promoted_id=right_id,
            demoted_id=left_id,
            from_index=index + 1,
            to_index=index,
            event_lock_digest=hashlib.sha256(lock.encode("utf-8")).hexdigest()[:16],
            lock_kind=selected_lock.kind,
            source_bucket_ids=selected_lock.source_bucket_ids,
            derived_lock_basis=selected_lock.derived_lock_basis,
            relation_type=selected_lock.relation_type,
            relation_from_id=selected_lock.relation_from_id,
            relation_to_id=selected_lock.relation_to_id,
            recorded_day=selected_lock.recorded_day,
            domain_digest=selected_lock.domain_digest,
        ))
        index += 2

    c_ids = tuple(_candidate_id(candidate) for candidate in proposed)
    if set(c_ids) != set(b_ids) or len(c_ids) != len(b_ids):
        violations.append("candidate_set_drift")
    displacement = max(
        (abs(c_ids.index(bucket_id) - position) for position, bucket_id in enumerate(b_ids)),
        default=0,
    )
    if displacement > 1:
        violations.append("max_displacement")
    for swap in swaps:
        if (
            (swap.promoted_id in bypass_set)
            != (swap.demoted_id in bypass_set)
        ):
            violations.append("bypass_boundary")
        before_left = original[swap.to_index]
        before_right = original[swap.from_index]
        validated_lock = _select_event_lock(
            before_left,
            before_right,
            derived_lock_enabled=derived_lock_enabled,
            source_buckets_by_id=source_lookup,
            allowed_relation_types=relation_types,
            relation_min_strength=relation_floor,
        )
        if (
            validated_lock is None
            or validated_lock.kind != swap.lock_kind
            or hashlib.sha256(
                validated_lock.canonical.encode("utf-8")
            ).hexdigest()[:16] != swap.event_lock_digest
        ):
            violations.append("cross_event_move")
        if _frozen_band(before_left) != _frozen_band(before_right):
            violations.append("cross_relevance_move")
        if _is_factual(before_left) or _is_factual(before_right):
            violations.append("fact_move")
    return ShadowProposal(
        b_ids=b_ids,
        c_ids=c_ids if not violations else b_ids,
        swaps=tuple(swaps) if not violations else (),
        skipped_reasons=tuple(skipped),
        violations=tuple(dict.fromkeys(violations)),
    )


def _ids(candidates: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(_candidate_id(candidate) for candidate in candidates)


def _a_cohort_status(
    pre_e_cohort_ids: Sequence[str],
    post_e_cohort_ids: Sequence[str],
    b_ids: Sequence[str],
    ds_decision_source: str,
) -> str:
    if ds_decision_source not in {"disabled", "deterministic_noop"}:
        return f"unscorable_ds_{ds_decision_source}"
    if (
        len(pre_e_cohort_ids) != len(post_e_cohort_ids)
        or len(post_e_cohort_ids) != len(b_ids)
        or set(pre_e_cohort_ids) != set(post_e_cohort_ids)
        or set(post_e_cohort_ids) != set(b_ids)
    ):
        return "unscorable_cohort_drift"
    if tuple(post_e_cohort_ids) != tuple(b_ids):
        return "unscorable_downstream_order"
    return "pure_semantic"


def _projection_digest(chord: LiveChord) -> str:
    body = {
        "schema": LIVE_SCHEMA,
        "turn_id": chord.turn_id,
        "agent_id": chord.agent_id,
        "e_authored_by": chord.e_authored_by,
        "session_scope": chord.session_scope,
        "source_turn_digest": chord.source_turn_digest,
        "captured_at_ms": chord.captured_at_ms,
        "facets": [
            {
                "motivation": facet.motivation,
                "drive_key": facet.drive_key,
                "tendency": facet.tendency,
                "hunger": facet.hunger,
                "salience": facet.salience,
            }
            for facet in chord.facets
        ],
    }
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_shadow_receipt(
    *,
    chord: LiveChord | None,
    payload_status: str,
    pre_e_cohort_ids: Sequence[str],
    post_e_cohort_ids: Sequence[str],
    ds_decision_source: str,
    a_candidates: Sequence[Mapping[str, object]],
    b_candidates: Sequence[Mapping[str, object]],
    proposal: ShadowProposal,
    attempt_index: int,
    first_screen_limit: int,
    request_path_delta_ms: float,
    recorded_at_ms: int,
    bypass_enabled: bool = False,
    bypass_ids: Sequence[str] = (),
    bypass_limit: int = 4,
) -> dict[str, Any]:
    """Build a content-free A/B/C receipt over one frozen retained pool."""

    if type(attempt_index) is not int or not 0 <= attempt_index <= 3:
        raise ValueError("attempt_index must be in [0, 3]")
    if type(first_screen_limit) is not int or not 0 <= first_screen_limit <= 50:
        raise ValueError("first_screen_limit must be in [0, 50]")
    elapsed = _finite(request_path_delta_ms)
    if elapsed is None or elapsed < 0:
        raise ValueError("request_path_delta_ms must be a finite non-negative number")
    if type(recorded_at_ms) is not int or recorded_at_ms < 0:
        raise ValueError("recorded_at_ms must be a non-negative integer")
    if type(payload_status) is not str or not payload_status or len(payload_status) > 80:
        raise ValueError("payload_status is invalid")
    if ds_decision_source not in _DS_DECISION_SOURCES:
        raise ValueError("ds_decision_source is invalid")
    if type(bypass_enabled) is not bool:
        raise ValueError("bypass_enabled must be a boolean")
    bypass_tuple = tuple(bypass_ids)
    if bypass_enabled:
        if (
            type(bypass_limit) is not int
            or isinstance(bypass_limit, bool)
            or not 1 <= bypass_limit <= 8
            or len(bypass_tuple) > bypass_limit
        ):
            raise ValueError("bypass_limit must be in [1, 8]")
    elif bypass_tuple:
        raise ValueError("bypass_ids require bypass_enabled")

    declared_a_ids = _ids(a_candidates)
    b_ids = _ids(b_candidates)
    bypass_set = set(bypass_tuple)
    bypass_boundary = int(
        len(bypass_set) != len(bypass_tuple)
        or any(_safe_id(bucket_id) is None for bucket_id in bypass_tuple)
        or len(bypass_tuple) > len(b_ids)
        or (
            bool(bypass_tuple)
            and tuple(b_ids[-len(bypass_tuple):]) != bypass_tuple
        )
    )
    if "bypass_boundary" in proposal.violations:
        bypass_boundary = 1
    natural_b_ids = (
        b_ids[:-len(bypass_tuple)] if bypass_tuple else b_ids
    )
    pre_e_ids = tuple(pre_e_cohort_ids)
    post_e_ids = tuple(post_e_cohort_ids)
    cohort_status = _a_cohort_status(
        pre_e_ids,
        post_e_ids,
        natural_b_ids if bypass_enabled else b_ids,
        ds_decision_source,
    )
    if cohort_status == "pure_semantic":
        if declared_a_ids != pre_e_ids:
            raise ValueError("pure semantic A does not match frozen pre-E cohort")
        a_ids = pre_e_ids + (bypass_tuple if bypass_enabled else ())
    else:
        # Never serialize a post-hoc ablation as a semantic baseline.  The
        # evaluator excludes this turn; B is a neutral placeholder for A.
        a_ids = b_ids
    relevance_band_ids = [_frozen_band(candidate) for candidate in b_candidates]
    candidate_guards = [
        {
            "id": _candidate_id(candidate),
            "event_lock_digests": list(_event_lock_digests(candidate)),
            "e_source_bucket_id": _candidate_source_bucket_id(candidate),
            "is_factual": _is_factual(candidate),
            "author_match": (
                _annotation_value(candidate, "authored_by") == chord.e_authored_by
                if chord is not None else False
            ),
            **(
                {"origin": "bypass" if _candidate_id(candidate) in bypass_set else "natural"}
                if bypass_enabled else {}
            ),
            **_candidate_e_guard(candidate),
        }
        for candidate in b_candidates
    ]
    c_ids = proposal.c_ids
    effective_swaps = proposal.swaps
    if bypass_enabled:
        for swap in effective_swaps:
            if (
                (swap.promoted_id in bypass_set)
                != (swap.demoted_id in bypass_set)
            ):
                bypass_boundary = 1
                break
        if bypass_boundary:
            c_ids = b_ids
            effective_swaps = ()
    pool_set = set(b_ids)
    same_pool = (
        len(a_ids) == len(b_ids) == len(c_ids)
        and set(a_ids) == pool_set
        and set(c_ids) == pool_set
    )
    max_displacement = max(
        (abs(c_ids.index(bucket_id) - index) for index, bucket_id in enumerate(b_ids))
        if same_pool else (len(b_ids),),
        default=0,
    )
    hard_violations = set(proposal.violations)
    if bypass_boundary:
        hard_violations.add("bypass_boundary")
    if not same_pool:
        hard_violations.add("candidate_set_drift")
    if max_displacement > 1:
        hard_violations.add("max_displacement")
    if not b_ids and c_ids:
        hard_violations.add("zero_to_nonzero")
    guard_by_id = {guard["id"]: guard for guard in candidate_guards}
    for swap in effective_swaps:
        promoted_guard = guard_by_id.get(swap.promoted_id, {})
        demoted_guard = guard_by_id.get(swap.demoted_id, {})
        shared_locks = set(promoted_guard.get("event_lock_digests", ())) & set(
            demoted_guard.get("event_lock_digests", ())
        )
        if swap.lock_kind == "strong":
            lock_is_bound = swap.event_lock_digest in shared_locks
        else:
            expected_sources = sorted({
                str(promoted_guard.get("e_source_bucket_id") or ""),
                str(demoted_guard.get("e_source_bucket_id") or ""),
            })
            canonical = _derived_lock_canonical(
                source_bucket_ids=swap.source_bucket_ids,
                derived_lock_basis=swap.derived_lock_basis,
                relation_type=swap.relation_type,
                relation_from_id=swap.relation_from_id,
                relation_to_id=swap.relation_to_id,
                recorded_day=swap.recorded_day,
                domain_digest=swap.domain_digest,
            )
            lock_is_bound = (
                swap.lock_kind == "derived"
                and not shared_locks
                and expected_sources == list(swap.source_bucket_ids)
                and "" not in expected_sources
                and canonical is not None
                and hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
                == swap.event_lock_digest
            )
        if not lock_is_bound:
            hard_violations.add("cross_event_move")
        if promoted_guard.get("is_factual") or demoted_guard.get("is_factual"):
            hard_violations.add("fact_move")
        if not promoted_guard.get("author_match") or not demoted_guard.get("author_match"):
            hard_violations.add("cross_author_move")
        if (
            promoted_guard.get("e_admissibility") != "admissible"
            or demoted_guard.get("e_admissibility") != "admissible"
            or int(promoted_guard.get("e_resonance_milli", 0))
            + _E_RESONANCE_MAX_REGRESSION_MILLI
            < int(demoted_guard.get("e_resonance_milli", 0))
        ):
            hard_violations.add("e_admissibility_move")

    receipt = {
        "schema": BYPASS_RECEIPT_SCHEMA if bypass_enabled else RECEIPT_SCHEMA,
        "shadow_only": True,
        "affects_ranking": False,
        "payload_status": payload_status,
        "recorded_at_ms": recorded_at_ms,
        "agent_id": chord.agent_id if chord is not None else "",
        "e_authored_by": chord.e_authored_by if chord is not None else "",
        "source_turn_digest": chord.source_turn_digest if chord is not None else "",
        "projection_digest": _projection_digest(chord) if chord is not None else "",
        "facet_count": len(chord.facets) if chord is not None else 0,
        "attempt_index": attempt_index,
        "first_screen_limit": first_screen_limit,
        "pre_e_cohort_ids": list(pre_e_ids),
        "post_e_cohort_ids": list(post_e_ids),
        "a_cohort_status": cohort_status,
        "ds_decision_source": ds_decision_source,
        "pool_ids": list(b_ids),
        "relevance_band_ids": relevance_band_ids,
        "candidate_guards": candidate_guards,
        "arms": {
            "a": list(a_ids),
            "b": list(b_ids),
            "c": list(c_ids),
        },
        "first_screen": {
            "a": list(a_ids[:first_screen_limit]),
            "b": list(b_ids[:first_screen_limit]),
            "c": list(c_ids[:first_screen_limit]),
        },
        "swaps": [
            {
                "promoted_id": swap.promoted_id,
                "demoted_id": swap.demoted_id,
                "from_index": swap.from_index,
                "to_index": swap.to_index,
                "event_lock_digest": swap.event_lock_digest,
                "lock_kind": swap.lock_kind,
                "source_bucket_ids": list(swap.source_bucket_ids),
                "derived_lock_basis": swap.derived_lock_basis,
                "relation_type": swap.relation_type,
                "relation_from_id": swap.relation_from_id,
                "relation_to_id": swap.relation_to_id,
                "recorded_day": swap.recorded_day,
                "domain_digest": swap.domain_digest,
            }
            for swap in effective_swaps
        ],
        "skipped_reasons": list(proposal.skipped_reasons),
        "diagnostics": {
            "same_candidate_pool": same_pool,
            "candidate_set_drift": int("candidate_set_drift" in hard_violations),
            "max_displacement": max_displacement,
            "cross_event_moves": int("cross_event_move" in hard_violations),
            "cross_relevance_moves": int(
                "cross_relevance_move" in hard_violations
            ),
            "fact_moves": int("fact_move" in hard_violations),
            "cross_author_moves": int("cross_author_move" in hard_violations),
            "e_admissibility_moves": int(
                "e_admissibility_move" in hard_violations
            ),
            "zero_to_nonzero": int("zero_to_nonzero" in hard_violations),
            **({"bypass_boundary": bypass_boundary} if bypass_enabled else {}),
            "external_api_delta": 0,
            "hard_violation_count": len(hard_violations),
        },
        "request_path_delta_ms": round(elapsed, 6),
    }
    if bypass_enabled:
        guard_by_id = {guard["id"]: guard for guard in candidate_guards}
        receipt.update({
            "bypass_ids": list(bypass_tuple),
            "bypass_source_ids": [
                str(guard_by_id[bucket_id]["e_source_bucket_id"])
                for bucket_id in bypass_tuple
            ],
            "bypass_limit": bypass_limit,
        })
    return receipt


_RECEIPT_KEYS = frozenset({
    "schema",
    "shadow_only",
    "affects_ranking",
    "payload_status",
    "recorded_at_ms",
    "agent_id",
    "e_authored_by",
    "source_turn_digest",
    "projection_digest",
    "facet_count",
    "attempt_index",
    "first_screen_limit",
    "pre_e_cohort_ids",
    "post_e_cohort_ids",
    "a_cohort_status",
    "ds_decision_source",
    "pool_ids",
    "relevance_band_ids",
    "candidate_guards",
    "arms",
    "first_screen",
    "swaps",
    "skipped_reasons",
    "diagnostics",
    "request_path_delta_ms",
})
_BYPASS_RECEIPT_KEYS = _RECEIPT_KEYS | frozenset({
    "bypass_ids",
    "bypass_source_ids",
    "bypass_limit",
})
_DIAGNOSTIC_KEYS = frozenset({
    "same_candidate_pool",
    "candidate_set_drift",
    "max_displacement",
    "cross_event_moves",
    "fact_moves",
    "cross_author_moves",
    "e_admissibility_moves",
    "cross_relevance_moves",
    "zero_to_nonzero",
    "external_api_delta",
    "hard_violation_count",
})
_BYPASS_DIAGNOSTIC_KEYS = _DIAGNOSTIC_KEYS | frozenset({
    "bypass_boundary",
})
_LEGACY_SWAP_KEYS = frozenset({
    "promoted_id",
    "demoted_id",
    "from_index",
    "to_index",
    "event_lock_digest",
})
_SWAP_KEYS = _LEGACY_SWAP_KEYS | frozenset({
    "lock_kind",
    "source_bucket_ids",
    "derived_lock_basis",
    "relation_type",
    "relation_from_id",
    "relation_to_id",
    "recorded_day",
    "domain_digest",
})
_LEGACY_CANDIDATE_GUARD_KEYS = frozenset({
    "id",
    "event_lock_digests",
    "is_factual",
    "author_match",
    "has_e_annotation",
    "e_resonance_milli",
    "e_resonance_floor_milli",
    "input_polarity",
    "experience_polarity",
    "e_admissibility",
})
_CANDIDATE_GUARD_KEYS = _LEGACY_CANDIDATE_GUARD_KEYS | frozenset({
    "e_source_bucket_id",
})
_BYPASS_CANDIDATE_GUARD_KEYS = _CANDIDATE_GUARD_KEYS | frozenset({
    "origin",
})
_MACHINE_REASON_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,79}$")


def _validated_id_list(value: object, *, maximum: int = 50) -> list[str]:
    if type(value) is not list or len(value) > maximum:
        raise ValueError("invalid E chord receipt contract")
    if any(_safe_id(item) is None for item in value):
        raise ValueError("invalid E chord receipt contract")
    if len(set(value)) != len(value):
        raise ValueError("invalid E chord receipt contract")
    return value


def validate_shadow_receipt(row: object) -> dict[str, Any]:
    if type(row) is not dict:
        raise ValueError("invalid E chord receipt contract")
    schema = row.get("schema")
    if schema not in {
        LEGACY_RECEIPT_SCHEMA,
        RECEIPT_SCHEMA,
        BYPASS_RECEIPT_SCHEMA,
    }:
        raise ValueError("invalid E chord receipt contract")
    is_v3 = schema == BYPASS_RECEIPT_SCHEMA
    if set(row) != (_BYPASS_RECEIPT_KEYS if is_v3 else _RECEIPT_KEYS):
        raise ValueError("invalid E chord receipt contract")
    is_v2 = schema in {RECEIPT_SCHEMA, BYPASS_RECEIPT_SCHEMA}
    if (
        row.get("shadow_only") is not True
        or row.get("affects_ranking") is not False
    ):
        raise ValueError("invalid E chord receipt contract")
    if (
        type(row.get("payload_status")) is not str
        or _MACHINE_REASON_RE.fullmatch(row["payload_status"]) is None
    ):
        raise ValueError("invalid E chord receipt contract")
    if type(row.get("recorded_at_ms")) is not int or row["recorded_at_ms"] < 0:
        raise ValueError("invalid E chord receipt contract")
    if _safe_id(row.get("agent_id")) is None:
        raise ValueError("invalid E chord receipt contract")
    if _safe_author(row.get("e_authored_by")) is None:
        raise ValueError("invalid E chord receipt contract")
    if row.get("e_authored_by") != _E_AUTHOR_BY_AGENT.get(row.get("agent_id")):
        raise ValueError("invalid E chord receipt contract")
    source_turn_digest = row.get("source_turn_digest")
    if (
        type(source_turn_digest) is not str
        or _HEX64_RE.fullmatch(source_turn_digest) is None
    ):
        raise ValueError("invalid E chord receipt contract")
    digest = row.get("projection_digest")
    if type(digest) is not str or _HEX64_RE.fullmatch(digest) is None:
        raise ValueError("invalid E chord receipt contract")
    if type(row.get("facet_count")) is not int or not 0 <= row["facet_count"] <= 2:
        raise ValueError("invalid E chord receipt contract")
    if (
        type(row.get("attempt_index")) is not int
        or not 0 <= row["attempt_index"] <= 3
    ):
        raise ValueError("invalid E chord receipt contract")
    if (
        type(row.get("first_screen_limit")) is not int
        or not 0 <= row["first_screen_limit"] <= 50
    ):
        raise ValueError("invalid E chord receipt contract")
    pre_e_cohort_ids = _validated_id_list(row.get("pre_e_cohort_ids"))
    post_e_cohort_ids = _validated_id_list(row.get("post_e_cohort_ids"))
    ds_decision_source = row.get("ds_decision_source")
    if ds_decision_source not in _DS_DECISION_SOURCES:
        raise ValueError("invalid E chord receipt contract")
    pool_ids = _validated_id_list(
        row.get("pool_ids"),
        maximum=58 if is_v3 else 50,
    )
    bypass_ids: list[str] = []
    bypass_source_ids: list[str] = []
    bypass_limit = 0
    if is_v3:
        bypass_ids = _validated_id_list(row.get("bypass_ids"), maximum=8)
        raw_source_ids = row.get("bypass_source_ids")
        bypass_limit = row.get("bypass_limit")
        if (
            type(raw_source_ids) is not list
            or len(raw_source_ids) != len(bypass_ids)
            or any(_safe_id(item) is None for item in raw_source_ids)
            or type(bypass_limit) is not int
            or isinstance(bypass_limit, bool)
            or not 1 <= bypass_limit <= 8
            or len(bypass_ids) > bypass_limit
            or (
                bool(bypass_ids)
                and pool_ids[-len(bypass_ids):] != bypass_ids
            )
            or any(bucket_id in pre_e_cohort_ids for bucket_id in bypass_ids)
            or any(bucket_id in post_e_cohort_ids for bucket_id in bypass_ids)
        ):
            raise ValueError("invalid E chord receipt contract")
        bypass_source_ids = list(raw_source_ids)
    natural_pool_ids = (
        pool_ids[:-len(bypass_ids)] if bypass_ids else pool_ids
    )
    expected_cohort_status = _a_cohort_status(
        pre_e_cohort_ids,
        post_e_cohort_ids,
        natural_pool_ids if is_v3 else pool_ids,
        ds_decision_source,
    )
    if row.get("a_cohort_status") != expected_cohort_status:
        raise ValueError("invalid E chord receipt contract")
    relevance_band_ids = row.get("relevance_band_ids")
    if (
        type(relevance_band_ids) is not list
        or len(relevance_band_ids) != len(pool_ids)
        or any(type(value) is not int or value < 0 for value in relevance_band_ids)
    ):
        raise ValueError("invalid E chord receipt contract")
    band_by_id = dict(zip(pool_ids, relevance_band_ids, strict=True))
    candidate_guards = row.get("candidate_guards")
    if type(candidate_guards) is not list or len(candidate_guards) != len(pool_ids):
        raise ValueError("invalid E chord receipt contract")
    guard_by_id: dict[str, dict[str, Any]] = {}
    expected_guard_keys = (
        _BYPASS_CANDIDATE_GUARD_KEYS
        if is_v3
        else _CANDIDATE_GUARD_KEYS
        if is_v2
        else _LEGACY_CANDIDATE_GUARD_KEYS
    )
    for index, guard in enumerate(candidate_guards):
        if type(guard) is not dict or set(guard) != expected_guard_keys:
            raise ValueError("invalid E chord receipt contract")
        bucket_id = guard.get("id")
        lock_digests = guard.get("event_lock_digests")
        source_bucket_id = guard.get("e_source_bucket_id", "")
        expected_origin = "bypass" if bucket_id in set(bypass_ids) else "natural"
        if (
            bucket_id != pool_ids[index]
            or type(lock_digests) is not list
            or len(lock_digests) > (3 if is_v2 else 2)
            or lock_digests != sorted(set(lock_digests))
            or any(
                type(value) is not str or _HEX16_RE.fullmatch(value) is None
                for value in lock_digests
            )
            or (
                is_v2
                and source_bucket_id != ""
                and _safe_id(source_bucket_id) is None
            )
            or (
                is_v3
                and guard.get("origin") != expected_origin
            )
            or type(guard.get("is_factual")) is not bool
            or type(guard.get("author_match")) is not bool
            or type(guard.get("has_e_annotation")) is not bool
            or type(guard.get("e_resonance_milli")) is not int
            or not 0 <= guard["e_resonance_milli"] <= 1000
            or type(guard.get("e_resonance_floor_milli")) is not int
            or not round(_E_ADMISSIBILITY_FLOOR * 1000)
            <= guard["e_resonance_floor_milli"] <= 1000
            or type(guard.get("input_polarity")) is not int
            or guard["input_polarity"] not in {-1, 0, 1}
            or type(guard.get("experience_polarity")) is not int
            or guard["experience_polarity"] not in {-1, 0, 1}
            or guard.get("e_admissibility") not in _E_ADMISSIBILITY_VALUES
            or guard["e_admissibility"] != _e_admissibility_status(
                has_e_annotation=guard["has_e_annotation"],
                e_resonance_milli=guard["e_resonance_milli"],
                e_resonance_floor_milli=guard["e_resonance_floor_milli"],
                input_polarity=guard["input_polarity"],
                experience_polarity=guard["experience_polarity"],
            )
        ):
            raise ValueError("invalid E chord receipt contract")
        guard_by_id[bucket_id] = guard
    if is_v3:
        for bucket_id, source_bucket_id in zip(
            bypass_ids,
            bypass_source_ids,
            strict=True,
        ):
            guard = guard_by_id[bucket_id]
            if (
                guard.get("e_source_bucket_id") != source_bucket_id
                or guard.get("e_admissibility") != "admissible"
            ):
                raise ValueError("invalid E chord receipt contract")
    if candidate_guards and (
        len({guard["input_polarity"] for guard in candidate_guards}) != 1
        or len({guard["e_resonance_floor_milli"] for guard in candidate_guards}) != 1
    ):
        raise ValueError("invalid E chord receipt contract")
    if type(row.get("swaps")) is not list or len(row["swaps"]) > len(pool_ids) // 2:
        raise ValueError("invalid E chord receipt contract")
    reconstructed_c = list(pool_ids)
    used_event_locks: set[str] = set()
    moved_ids: set[str] = set()
    recomputed_cross_event_moves = 0
    recomputed_fact_moves = 0
    recomputed_cross_author_moves = 0
    recomputed_e_admissibility_moves = 0
    expected_swap_keys = _SWAP_KEYS if is_v2 else _LEGACY_SWAP_KEYS
    for swap in row["swaps"]:
        if type(swap) is not dict or set(swap) != expected_swap_keys:
            raise ValueError("invalid E chord receipt contract")
        if (
            _safe_id(swap.get("promoted_id")) is None
            or _safe_id(swap.get("demoted_id")) is None
            or swap["promoted_id"] not in pool_ids
            or swap["demoted_id"] not in pool_ids
            or type(swap.get("from_index")) is not int
            or type(swap.get("to_index")) is not int
            or swap["from_index"] != swap["to_index"] + 1
            or not 0 <= swap["to_index"] < swap["from_index"] < len(pool_ids)
            or type(swap.get("event_lock_digest")) is not str
            or _HEX16_RE.fullmatch(swap["event_lock_digest"]) is None
            or band_by_id[swap["promoted_id"]] != band_by_id[swap["demoted_id"]]
            or (
                is_v3
                and guard_by_id[swap["promoted_id"]]["origin"]
                != guard_by_id[swap["demoted_id"]]["origin"]
            )
        ):
            raise ValueError("invalid E chord receipt contract")
        promoted_guard = guard_by_id[swap["promoted_id"]]
        demoted_guard = guard_by_id[swap["demoted_id"]]
        shared_locks = set(promoted_guard["event_lock_digests"]) & set(
            demoted_guard["event_lock_digests"]
        )
        if not is_v2:
            lock_is_bound = swap["event_lock_digest"] in shared_locks
        elif swap.get("lock_kind") == "strong":
            lock_is_bound = (
                swap["event_lock_digest"] in shared_locks
                and swap.get("source_bucket_ids") == []
                and swap.get("derived_lock_basis") == ""
                and swap.get("relation_type") == ""
                and swap.get("relation_from_id") == ""
                and swap.get("relation_to_id") == ""
                and swap.get("recorded_day") == ""
                and swap.get("domain_digest") == ""
            )
        elif swap.get("lock_kind") == "derived":
            expected_sources = sorted({
                promoted_guard.get("e_source_bucket_id", ""),
                demoted_guard.get("e_source_bucket_id", ""),
            })
            canonical = _derived_lock_canonical(
                source_bucket_ids=swap.get("source_bucket_ids"),
                derived_lock_basis=swap.get("derived_lock_basis"),
                relation_type=swap.get("relation_type"),
                relation_from_id=swap.get("relation_from_id"),
                relation_to_id=swap.get("relation_to_id"),
                recorded_day=swap.get("recorded_day"),
                domain_digest=swap.get("domain_digest"),
            )
            lock_is_bound = (
                not shared_locks
                and "" not in expected_sources
                and swap.get("source_bucket_ids") == expected_sources
                and canonical is not None
                and hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
                == swap["event_lock_digest"]
            )
        else:
            lock_is_bound = False
        if not lock_is_bound:
            recomputed_cross_event_moves = 1
        if promoted_guard["is_factual"] or demoted_guard["is_factual"]:
            recomputed_fact_moves = 1
        if not promoted_guard["author_match"] or not demoted_guard["author_match"]:
            recomputed_cross_author_moves = 1
        if (
            promoted_guard["e_admissibility"] != "admissible"
            or demoted_guard["e_admissibility"] != "admissible"
            or promoted_guard["e_resonance_milli"]
            + _E_RESONANCE_MAX_REGRESSION_MILLI
            < demoted_guard["e_resonance_milli"]
        ):
            recomputed_e_admissibility_moves = 1
        if (
            reconstructed_c[swap["to_index"]] != swap["demoted_id"]
            or reconstructed_c[swap["from_index"]] != swap["promoted_id"]
            or swap["event_lock_digest"] in used_event_locks
            or swap["promoted_id"] in moved_ids
            or swap["demoted_id"] in moved_ids
        ):
            raise ValueError("invalid E chord receipt contract")
        reconstructed_c[swap["to_index"]], reconstructed_c[swap["from_index"]] = (
            reconstructed_c[swap["from_index"]],
            reconstructed_c[swap["to_index"]],
        )
        used_event_locks.add(swap["event_lock_digest"])
        moved_ids.update((swap["promoted_id"], swap["demoted_id"]))
    skipped = row.get("skipped_reasons")
    if (
        type(skipped) is not list
        or len(skipped) > 16
        or len(set(skipped)) != len(skipped)
        or any(
            type(reason) is not str
            or _MACHINE_REASON_RE.fullmatch(reason) is None
            for reason in skipped
        )
    ):
        raise ValueError("invalid E chord receipt contract")
    for name in ("arms", "first_screen"):
        value = row.get(name)
        if type(value) is not dict or set(value) != {"a", "b", "c"}:
            raise ValueError("invalid E chord receipt contract")
        for arm in ("a", "b", "c"):
            _validated_id_list(
                value[arm],
                maximum=(58 if is_v3 and name == "arms" else 50),
            )
    if row["arms"]["b"] != pool_ids:
        raise ValueError("invalid E chord receipt contract")
    if row["arms"]["c"] != reconstructed_c:
        raise ValueError("invalid E chord receipt contract")
    if row["a_cohort_status"] == "pure_semantic":
        expected_a = pre_e_cohort_ids + (bypass_ids if is_v3 else [])
        if row["arms"]["a"] != expected_a:
            raise ValueError("invalid E chord receipt contract")
    elif row["arms"]["a"] != row["arms"]["b"]:
        # An unscorable turn gets a neutral B placeholder, never a fabricated
        # "pure semantic" arm.
        raise ValueError("invalid E chord receipt contract")
    same_pool = all(
        len(row["arms"][arm]) == len(pool_ids)
        and set(row["arms"][arm]) == set(pool_ids)
        for arm in ("a", "b", "c")
    )
    if not same_pool:
        raise ValueError("invalid E chord receipt contract")
    for arm in ("a", "b", "c"):
        if row["first_screen"][arm] != row["arms"][arm][:row["first_screen_limit"]]:
            raise ValueError("invalid E chord receipt contract")
    diagnostics = row.get("diagnostics")
    if (
        type(diagnostics) is not dict
        or set(diagnostics)
        != (_BYPASS_DIAGNOSTIC_KEYS if is_v3 else _DIAGNOSTIC_KEYS)
    ):
        raise ValueError("invalid E chord receipt contract")
    if type(diagnostics.get("same_candidate_pool")) is not bool:
        raise ValueError("invalid E chord receipt contract")
    binary_diagnostics = {
        "candidate_set_drift",
        "cross_event_moves",
        "cross_relevance_moves",
        "fact_moves",
        "cross_author_moves",
        "e_admissibility_moves",
        "zero_to_nonzero",
        "external_api_delta",
    }
    if is_v3:
        binary_diagnostics.add("bypass_boundary")
    for name in binary_diagnostics:
        if type(diagnostics.get(name)) is not int or diagnostics[name] < 0:
            raise ValueError("invalid E chord receipt contract")
        if diagnostics[name] not in (0, 1):
            raise ValueError("invalid E chord receipt contract")
    if (
        type(diagnostics.get("hard_violation_count")) is not int
        or diagnostics["hard_violation_count"] < 0
        or type(diagnostics.get("max_displacement")) is not int
        or diagnostics["max_displacement"] < 0
    ):
        raise ValueError("invalid E chord receipt contract")
    if (
        diagnostics["same_candidate_pool"] is not same_pool
        or diagnostics["candidate_set_drift"] != int(not same_pool)
        or diagnostics["cross_event_moves"] != recomputed_cross_event_moves
        or diagnostics["fact_moves"] != recomputed_fact_moves
        or diagnostics["cross_author_moves"] != recomputed_cross_author_moves
        or diagnostics["e_admissibility_moves"]
        != recomputed_e_admissibility_moves
    ):
        raise ValueError("invalid E chord receipt contract")
    if (
        is_v3
        and diagnostics["bypass_boundary"]
        and (row["swaps"] or row["arms"]["c"] != row["arms"]["b"])
    ):
        raise ValueError("invalid E chord receipt contract")
    recomputed_displacement = max(
        (
            abs(row["arms"]["c"].index(bucket_id) - index)
            for index, bucket_id in enumerate(pool_ids)
        ),
        default=0,
    )
    if diagnostics.get("max_displacement") != recomputed_displacement:
        raise ValueError("invalid E chord receipt contract")
    expected_hard_count = (
        diagnostics["candidate_set_drift"]
        + int(recomputed_displacement > 1)
        + diagnostics["cross_event_moves"]
        + diagnostics["cross_relevance_moves"]
        + diagnostics["fact_moves"]
        + diagnostics["cross_author_moves"]
        + diagnostics["e_admissibility_moves"]
        + diagnostics["zero_to_nonzero"]
        + (diagnostics["bypass_boundary"] if is_v3 else 0)
    )
    if diagnostics["hard_violation_count"] != expected_hard_count:
        raise ValueError("invalid E chord receipt contract")
    if diagnostics.get("external_api_delta") != 0:
        raise ValueError("invalid E chord receipt contract")
    elapsed = _finite(row.get("request_path_delta_ms"))
    if elapsed is None or elapsed < 0:
        raise ValueError("invalid E chord receipt contract")
    # Canonical serialization is also the final non-finite guard.
    try:
        json.dumps(row, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid E chord receipt contract") from exc
    return row


class EChordShadowLedger:
    """Append-only private JSONL, separate from annotations and recall data."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(os.path.abspath(os.fspath(path)))
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def append(self, receipt: object) -> None:
        row = validate_shadow_receipt(receipt)
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        # This write runs off the recall response path.  Block in the worker
        # thread so concurrent accepted turns serialize instead of silently
        # losing a shadow sample to a non-blocking lock collision.
        with secure_e_axis_lock(self.lock_path, blocking=True):
            with open_secure_e_axis_jsonl(self.path) as handle:
                handle.seek(0, os.SEEK_END)
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def load(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with secure_e_axis_lock(self.lock_path, blocking=False):
            with open_secure_e_axis_jsonl(self.path) as handle:
                handle.seek(0)
                for line_number, raw in enumerate(handle, 1):
                    if not raw.strip():
                        continue
                    try:
                        value = json.loads(
                            raw,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                ValueError(f"non-finite number: {value}")
                            ),
                        )
                        rows.append(validate_shadow_receipt(value))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"corrupt E chord shadow ledger at line {line_number}"
                        ) from exc
        return rows


_FINAL_SELECTION_KEYS = frozenset({
    "schema",
    "recorded_at_ms",
    "agent_id",
    "source_turn_digest",
    "projection_digest",
    "attempt_index",
    "pool_ids",
    "final_input_cohort_ids",
    "final_input_cohort_status",
    "final_injected_ids",
    "outside_pool_ids",
    "arms",
    "applied_swaps",
    "request_path_delta_ms",
})
_BYPASS_FINAL_SELECTION_KEYS = _FINAL_SELECTION_KEYS | frozenset({
    "bypass_ids",
    "bypass_source_ids",
    "bypass_limit",
})
_LIVE_BYPASS_FINAL_SELECTION_KEYS = _BYPASS_FINAL_SELECTION_KEYS | frozenset({
    "delivered_bypass_ids",
    "displaced_natural_ids",
    "mode",
    "served_arm",
    "live_applied",
    "fallback_reason",
})


def _validate_live_bypass_final_selection(value: dict[str, Any]) -> dict[str, Any]:
    """Validate a final prompt whose membership changed via live delivery."""

    if set(value) != _LIVE_BYPASS_FINAL_SELECTION_KEYS:
        raise ValueError("invalid E chord final selection contract")
    if type(value.get("recorded_at_ms")) is not int or value["recorded_at_ms"] < 0:
        raise ValueError("invalid E chord final selection contract")
    if _safe_id(value.get("agent_id")) is None:
        raise ValueError("invalid E chord final selection contract")
    for name in ("source_turn_digest", "projection_digest"):
        digest = value.get(name)
        if type(digest) is not str or _HEX64_RE.fullmatch(digest) is None:
            raise ValueError("invalid E chord final selection contract")
    if (
        type(value.get("attempt_index")) is not int
        or not 0 <= value["attempt_index"] <= 3
    ):
        raise ValueError("invalid E chord final selection contract")

    pool_ids = _validated_id_list(value.get("pool_ids"), maximum=58)
    bypass_ids = _validated_id_list(value.get("bypass_ids"), maximum=8)
    bypass_source_ids = value.get("bypass_source_ids")
    bypass_limit = value.get("bypass_limit")
    delivered_ids = _validated_id_list(
        value.get("delivered_bypass_ids"),
        maximum=1,
    )
    displaced_ids = _validated_id_list(
        value.get("displaced_natural_ids"),
        maximum=1,
    )
    if (
        not delivered_ids
        or type(bypass_source_ids) is not list
        or len(bypass_source_ids) != len(bypass_ids)
        or any(_safe_id(item) is None for item in bypass_source_ids)
        or type(bypass_limit) is not int
        or isinstance(bypass_limit, bool)
        or not 1 <= bypass_limit <= 8
        or len(bypass_ids) > bypass_limit
        or not set(delivered_ids) <= set(bypass_ids)
        or set(displaced_ids) & set(bypass_ids)
        or (bool(bypass_ids) and pool_ids[-len(bypass_ids):] != bypass_ids)
    ):
        raise ValueError("invalid E chord final selection contract")

    final_input_ids = _validated_id_list(value.get("final_input_cohort_ids"))
    final_ids = _validated_id_list(value.get("final_injected_ids"))
    outside_ids = _validated_id_list(value.get("outside_pool_ids"))
    pool_set = set(pool_ids)
    if (
        len(final_input_ids) > 58
        or len(final_ids) > 32
        or len(outside_ids) > 32
        or not set(final_input_ids) <= pool_set
        or final_input_ids
        != [bucket_id for bucket_id in pool_ids if bucket_id in set(final_input_ids)]
        or value.get("final_input_cohort_status") != "live_bypass_delivery"
        or [bucket_id for bucket_id in final_ids if bucket_id not in pool_set]
        != outside_ids
        or any(bucket_id not in final_ids for bucket_id in delivered_ids)
        or any(bucket_id not in final_input_ids for bucket_id in delivered_ids)
        or set(final_ids) & set(bypass_ids) != set(delivered_ids)
    ):
        raise ValueError("invalid E chord final selection contract")

    arms = value.get("arms")
    if type(arms) is not dict or set(arms) != {"a", "b", "c"}:
        raise ValueError("invalid E chord final selection contract")
    normalized_arms = {
        name: _validated_id_list(arms[name])
        for name in ("a", "b", "c")
    }
    final_pool_ids = [bucket_id for bucket_id in final_ids if bucket_id in pool_set]
    natural_pool_set = pool_set - set(bypass_ids)
    arm_a = normalized_arms["a"]
    arm_b = normalized_arms["b"]
    arm_c = normalized_arms["c"]
    if (
        any(len(ids) > 32 or not set(ids) <= pool_set for ids in normalized_arms.values())
        or arm_a != arm_b
        or not set(arm_b) <= natural_pool_set
        or arm_c != final_pool_ids
        or set(arm_c) - set(arm_b) != set(delivered_ids)
        or set(arm_b) - set(arm_c) != set(displaced_ids)
        or not set(arm_b) <= set(final_input_ids)
        or value.get("applied_swaps") != []
        or value.get("mode") != "live"
        or value.get("served_arm") != "c"
        or value.get("live_applied") is not True
        or value.get("fallback_reason") != ""
    ):
        raise ValueError("invalid E chord final selection contract")
    elapsed = _finite(value.get("request_path_delta_ms"))
    if elapsed is None or elapsed < 0:
        raise ValueError("invalid E chord final selection contract")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid E chord final selection contract") from exc
    return value


def validate_final_selection(value: object) -> dict[str, Any]:
    """Validate Twin's text-free post-filter A/B/C final selection receipt."""

    if type(value) is not dict:
        raise ValueError("invalid E chord final selection contract")
    schema = value.get("schema")
    if schema == LIVE_BYPASS_FINAL_SELECTION_SCHEMA:
        return _validate_live_bypass_final_selection(value)
    if schema not in {
        LEGACY_FINAL_SELECTION_SCHEMA,
        FINAL_SELECTION_SCHEMA,
        BYPASS_FINAL_SELECTION_SCHEMA,
    }:
        raise ValueError("invalid E chord final selection contract")
    is_v3 = schema == BYPASS_FINAL_SELECTION_SCHEMA
    if set(value) != (
        _BYPASS_FINAL_SELECTION_KEYS if is_v3 else _FINAL_SELECTION_KEYS
    ):
        raise ValueError("invalid E chord final selection contract")
    is_v2 = schema in {FINAL_SELECTION_SCHEMA, BYPASS_FINAL_SELECTION_SCHEMA}
    if type(value.get("recorded_at_ms")) is not int or value["recorded_at_ms"] < 0:
        raise ValueError("invalid E chord final selection contract")
    if _safe_id(value.get("agent_id")) is None:
        raise ValueError("invalid E chord final selection contract")
    for name in ("source_turn_digest", "projection_digest"):
        digest = value.get(name)
        if type(digest) is not str or _HEX64_RE.fullmatch(digest) is None:
            raise ValueError("invalid E chord final selection contract")
    if (
        type(value.get("attempt_index")) is not int
        or not 0 <= value["attempt_index"] <= 3
    ):
        raise ValueError("invalid E chord final selection contract")
    pool_ids = _validated_id_list(
        value.get("pool_ids"),
        maximum=58 if is_v3 else 50,
    )
    bypass_ids: list[str] = []
    bypass_source_ids: list[str] = []
    if is_v3:
        bypass_ids = _validated_id_list(value.get("bypass_ids"), maximum=8)
        raw_source_ids = value.get("bypass_source_ids")
        bypass_limit = value.get("bypass_limit")
        if (
            type(raw_source_ids) is not list
            or len(raw_source_ids) != len(bypass_ids)
            or any(_safe_id(item) is None for item in raw_source_ids)
            or type(bypass_limit) is not int
            or isinstance(bypass_limit, bool)
            or not 1 <= bypass_limit <= 8
            or len(bypass_ids) > bypass_limit
            or (
                bool(bypass_ids)
                and pool_ids[-len(bypass_ids):] != bypass_ids
            )
        ):
            raise ValueError("invalid E chord final selection contract")
        bypass_source_ids = list(raw_source_ids)
    natural_pool_ids = (
        pool_ids[:-len(bypass_ids)] if bypass_ids else pool_ids
    )
    final_input_ids = _validated_id_list(value.get("final_input_cohort_ids"))
    if (
        len(final_input_ids) > 50
        or not set(final_input_ids) <= set(natural_pool_ids)
    ):
        raise ValueError("invalid E chord final selection contract")
    if final_input_ids != [
        bucket_id
        for bucket_id in natural_pool_ids
        if bucket_id in set(final_input_ids)
    ]:
        raise ValueError("invalid E chord final selection contract")
    expected_final_cohort_status = (
        "pure_same_cohort"
        if final_input_ids == natural_pool_ids
        else "unscorable_final_cohort_drift"
    )
    if value.get("final_input_cohort_status") != expected_final_cohort_status:
        raise ValueError("invalid E chord final selection contract")
    final_ids = _validated_id_list(value.get("final_injected_ids"))
    outside_ids = _validated_id_list(value.get("outside_pool_ids"))
    if len(final_ids) > 32 or len(outside_ids) > 32:
        raise ValueError("invalid E chord final selection contract")
    pool = set(pool_ids)
    if is_v3 and set(final_ids) & set(bypass_ids):
        raise ValueError("invalid E chord final selection contract")
    if [bucket_id for bucket_id in final_ids if bucket_id not in pool] != outside_ids:
        raise ValueError("invalid E chord final selection contract")
    arms = value.get("arms")
    if type(arms) is not dict or set(arms) != {"a", "b", "c"}:
        raise ValueError("invalid E chord final selection contract")
    for arm in ("a", "b", "c"):
        arm_ids = _validated_id_list(arms[arm])
        if len(arm_ids) > 32 or not set(arm_ids) <= set(natural_pool_ids):
            raise ValueError("invalid E chord final selection contract")
    if arms["b"] != [bucket_id for bucket_id in final_ids if bucket_id in pool]:
        raise ValueError("invalid E chord final selection contract")
    if (
        expected_final_cohort_status != "pure_same_cohort"
        and arms["a"] != arms["b"]
    ):
        raise ValueError("invalid E chord final selection contract")
    if len(arms["c"]) != len(arms["b"]) or set(arms["c"]) != set(arms["b"]):
        raise ValueError("invalid E chord final selection contract")
    applied_swaps = value.get("applied_swaps")
    if type(applied_swaps) is not list or len(applied_swaps) > len(arms["b"]) // 2:
        raise ValueError("invalid E chord final selection contract")
    reconstructed_c = list(arms["b"])
    used_event_locks: set[str] = set()
    moved_ids: set[str] = set()
    expected_swap_keys = _SWAP_KEYS if is_v2 else _LEGACY_SWAP_KEYS
    for swap in applied_swaps:
        if type(swap) is not dict or set(swap) != expected_swap_keys:
            raise ValueError("invalid E chord final selection contract")
        promoted_id = swap.get("promoted_id")
        demoted_id = swap.get("demoted_id")
        from_index = swap.get("from_index")
        to_index = swap.get("to_index")
        event_lock_digest = swap.get("event_lock_digest")
        lock_contract_valid = True
        if is_v2 and swap.get("lock_kind") == "strong":
            lock_contract_valid = (
                swap.get("source_bucket_ids") == []
                and swap.get("derived_lock_basis") == ""
                and swap.get("relation_type") == ""
                and swap.get("relation_from_id") == ""
                and swap.get("relation_to_id") == ""
                and swap.get("recorded_day") == ""
                and swap.get("domain_digest") == ""
            )
        elif is_v2 and swap.get("lock_kind") == "derived":
            canonical = _derived_lock_canonical(
                source_bucket_ids=swap.get("source_bucket_ids"),
                derived_lock_basis=swap.get("derived_lock_basis"),
                relation_type=swap.get("relation_type"),
                relation_from_id=swap.get("relation_from_id"),
                relation_to_id=swap.get("relation_to_id"),
                recorded_day=swap.get("recorded_day"),
                domain_digest=swap.get("domain_digest"),
            )
            lock_contract_valid = (
                canonical is not None
                and hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
                == event_lock_digest
            )
        elif is_v2:
            lock_contract_valid = False
        if (
            _safe_id(promoted_id) is None
            or _safe_id(demoted_id) is None
            or promoted_id not in reconstructed_c
            or demoted_id not in reconstructed_c
            or type(from_index) is not int
            or type(to_index) is not int
            or from_index != to_index + 1
            or not 0 <= to_index < from_index < len(reconstructed_c)
            or type(event_lock_digest) is not str
            or _HEX16_RE.fullmatch(event_lock_digest) is None
            or reconstructed_c[to_index] != demoted_id
            or reconstructed_c[from_index] != promoted_id
            or event_lock_digest in used_event_locks
            or promoted_id in moved_ids
            or demoted_id in moved_ids
            or not lock_contract_valid
        ):
            raise ValueError("invalid E chord final selection contract")
        reconstructed_c[to_index], reconstructed_c[from_index] = (
            reconstructed_c[from_index],
            reconstructed_c[to_index],
        )
        used_event_locks.add(event_lock_digest)
        moved_ids.update((promoted_id, demoted_id))
    if arms["c"] != reconstructed_c:
        raise ValueError("invalid E chord final selection contract")
    elapsed = _finite(value.get("request_path_delta_ms"))
    if elapsed is None or elapsed < 0:
        raise ValueError("invalid E chord final selection contract")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid E chord final selection contract") from exc
    return value


class EChordFinalSelectionLedger:
    """Private append-only ledger for Twin's actual post-filter selections."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(os.path.abspath(os.fspath(path)))
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def append(self, selection: object) -> bool:
        row = validate_final_selection(selection)
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with secure_e_axis_lock(self.lock_path, blocking=True):
            with open_secure_e_axis_jsonl(self.path) as handle:
                handle.seek(0)
                for line_number, raw in enumerate(handle, 1):
                    if not raw.strip():
                        continue
                    try:
                        existing = validate_final_selection(json.loads(raw))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "corrupt E chord final selection ledger "
                            f"at line {line_number}"
                        ) from exc
                    if existing["projection_digest"] != row["projection_digest"]:
                        continue
                    if existing == row:
                        return False
                    raise ValueError("conflicting E chord final selection")
                handle.seek(0, os.SEEK_END)
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return True

    def load(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with secure_e_axis_lock(self.lock_path, blocking=False):
            with open_secure_e_axis_jsonl(self.path) as handle:
                handle.seek(0)
                for line_number, raw in enumerate(handle, 1):
                    if not raw.strip():
                        continue
                    try:
                        rows.append(validate_final_selection(json.loads(raw)))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"corrupt E chord final selection ledger at line {line_number}"
                        ) from exc
        return rows


__all__ = [
    "EChordShadowConfig",
    "EChordShadowLedger",
    "EChordFinalSelectionLedger",
    "BYPASS_FINAL_SELECTION_SCHEMA",
    "BYPASS_RECEIPT_SCHEMA",
    "FINAL_SELECTION_SCHEMA",
    "LIVE_BYPASS_FINAL_SELECTION_SCHEMA",
    "LiveChord",
    "LiveChordFacet",
    "ShadowProposal",
    "ShadowSwap",
    "build_shadow_receipt",
    "frozen_relevance_band_ids",
    "load_e_chord_shadow_config",
    "parse_live_chord",
    "propose_chord_reorder",
    "rank_within_frozen_relevance_bands",
    "select_bypass_candidates",
    "validate_shadow_receipt",
    "validate_final_selection",
]
