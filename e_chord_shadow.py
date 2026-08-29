"""Text-free, zero-call E-chord shadow proposal and private receipt ledger.

The chord is never a retrieval channel.  It may only propose an adjacent swap
between already-retained, primary-authored E memories that share an explicit
event lock and are already a near tie under the existing E rank.  Callers keep
serving arm B; arm C exists solely in the append-only shadow receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e_axis_storage import open_secure_e_axis_jsonl, secure_e_axis_lock


LIVE_SCHEMA = "live_chord.v1"
RECEIPT_SCHEMA = "e_chord_shadow_receipt.v1"
FINAL_SELECTION_SCHEMA = "e_chord_final_selection.v1"
MAX_FACETS = 2
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
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
    return EChordShadowConfig(
        enabled=True,
        near_tie_epsilon=epsilon,
        max_age_ms=max_age_ms,
    )


def parse_live_chord(
    raw: object,
    *,
    expected_turn_id: str,
    expected_agent_id: str,
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
    for key in ("event_id", "episode_id"):
        value = _safe_id(metadata.get(key))
        if value is not None:
            locks.add(f"{key}:{value}")
    return frozenset(locks)


def _event_lock_digests(candidate: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(sorted(
        hashlib.sha256(lock.encode("utf-8")).hexdigest()[:16]
        for lock in _event_locks(candidate)
    ))


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
) -> ShadowProposal:
    """Propose bounded C ordering without mutating B or serving the result."""

    epsilon = _finite(near_tie_epsilon)
    if epsilon is None or not 0.0 <= epsilon <= 0.1:
        raise ValueError("near_tie_epsilon must be in [0, 0.1]")
    original = list(candidates)
    b_ids = tuple(_candidate_id(candidate) for candidate in original)
    violations: list[str] = []
    if any(not bucket_id for bucket_id in b_ids):
        violations.append("invalid_candidate_id")
    if len(set(b_ids)) != len(b_ids):
        violations.append("duplicate_candidate_id")
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
        shared_locks = sorted(_event_locks(left) & _event_locks(right))
        if not shared_locks:
            _append_reason(skipped, "event_lock")
            index += 1
            continue
        lock = shared_locks[0]
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
        before_left = original[swap.to_index]
        before_right = original[swap.from_index]
        if not (_event_locks(before_left) & _event_locks(before_right)):
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
    a_candidates: Sequence[Mapping[str, object]],
    b_candidates: Sequence[Mapping[str, object]],
    proposal: ShadowProposal,
    attempt_index: int,
    first_screen_limit: int,
    request_path_delta_ms: float,
    recorded_at_ms: int,
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

    a_ids = _ids(a_candidates)
    b_ids = _ids(b_candidates)
    relevance_band_ids = [_frozen_band(candidate) for candidate in b_candidates]
    candidate_guards = [
        {
            "id": _candidate_id(candidate),
            "event_lock_digests": list(_event_lock_digests(candidate)),
            "is_factual": _is_factual(candidate),
            "author_match": (
                _annotation_value(candidate, "authored_by") == chord.e_authored_by
                if chord is not None else False
            ),
        }
        for candidate in b_candidates
    ]
    c_ids = proposal.c_ids
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
    if not same_pool:
        hard_violations.add("candidate_set_drift")
    if max_displacement > 1:
        hard_violations.add("max_displacement")
    if not b_ids and c_ids:
        hard_violations.add("zero_to_nonzero")
    guard_by_id = {guard["id"]: guard for guard in candidate_guards}
    for swap in proposal.swaps:
        promoted_guard = guard_by_id.get(swap.promoted_id, {})
        demoted_guard = guard_by_id.get(swap.demoted_id, {})
        shared_locks = set(promoted_guard.get("event_lock_digests", ())) & set(
            demoted_guard.get("event_lock_digests", ())
        )
        if swap.event_lock_digest not in shared_locks:
            hard_violations.add("cross_event_move")
        if promoted_guard.get("is_factual") or demoted_guard.get("is_factual"):
            hard_violations.add("fact_move")
        if not promoted_guard.get("author_match") or not demoted_guard.get("author_match"):
            hard_violations.add("cross_author_move")

    return {
        "schema": RECEIPT_SCHEMA,
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
            }
            for swap in proposal.swaps
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
            "zero_to_nonzero": int("zero_to_nonzero" in hard_violations),
            "external_api_delta": 0,
            "hard_violation_count": len(hard_violations),
        },
        "request_path_delta_ms": round(elapsed, 6),
    }


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
_DIAGNOSTIC_KEYS = frozenset({
    "same_candidate_pool",
    "candidate_set_drift",
    "max_displacement",
    "cross_event_moves",
    "fact_moves",
    "cross_author_moves",
    "cross_relevance_moves",
    "zero_to_nonzero",
    "external_api_delta",
    "hard_violation_count",
})
_SWAP_KEYS = frozenset({
    "promoted_id",
    "demoted_id",
    "from_index",
    "to_index",
    "event_lock_digest",
})
_CANDIDATE_GUARD_KEYS = frozenset({
    "id",
    "event_lock_digests",
    "is_factual",
    "author_match",
})
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
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
    if type(row) is not dict or set(row) != _RECEIPT_KEYS:
        raise ValueError("invalid E chord receipt contract")
    if (
        row.get("schema") != RECEIPT_SCHEMA
        or row.get("shadow_only") is not True
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
    pool_ids = _validated_id_list(row.get("pool_ids"))
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
    for index, guard in enumerate(candidate_guards):
        if type(guard) is not dict or set(guard) != _CANDIDATE_GUARD_KEYS:
            raise ValueError("invalid E chord receipt contract")
        bucket_id = guard.get("id")
        lock_digests = guard.get("event_lock_digests")
        if (
            bucket_id != pool_ids[index]
            or type(lock_digests) is not list
            or len(lock_digests) > 2
            or lock_digests != sorted(set(lock_digests))
            or any(
                type(value) is not str or _HEX16_RE.fullmatch(value) is None
                for value in lock_digests
            )
            or type(guard.get("is_factual")) is not bool
            or type(guard.get("author_match")) is not bool
        ):
            raise ValueError("invalid E chord receipt contract")
        guard_by_id[bucket_id] = guard
    if type(row.get("swaps")) is not list or len(row["swaps"]) > len(pool_ids) // 2:
        raise ValueError("invalid E chord receipt contract")
    reconstructed_c = list(pool_ids)
    used_event_locks: set[str] = set()
    moved_ids: set[str] = set()
    recomputed_cross_event_moves = 0
    recomputed_fact_moves = 0
    recomputed_cross_author_moves = 0
    for swap in row["swaps"]:
        if type(swap) is not dict or set(swap) != _SWAP_KEYS:
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
        ):
            raise ValueError("invalid E chord receipt contract")
        promoted_guard = guard_by_id[swap["promoted_id"]]
        demoted_guard = guard_by_id[swap["demoted_id"]]
        shared_locks = set(promoted_guard["event_lock_digests"]) & set(
            demoted_guard["event_lock_digests"]
        )
        if swap["event_lock_digest"] not in shared_locks:
            recomputed_cross_event_moves = 1
        if promoted_guard["is_factual"] or demoted_guard["is_factual"]:
            recomputed_fact_moves = 1
        if not promoted_guard["author_match"] or not demoted_guard["author_match"]:
            recomputed_cross_author_moves = 1
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
            _validated_id_list(value[arm])
    if row["arms"]["b"] != pool_ids:
        raise ValueError("invalid E chord receipt contract")
    if row["arms"]["c"] != reconstructed_c:
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
    if type(diagnostics) is not dict or set(diagnostics) != _DIAGNOSTIC_KEYS:
        raise ValueError("invalid E chord receipt contract")
    if type(diagnostics.get("same_candidate_pool")) is not bool:
        raise ValueError("invalid E chord receipt contract")
    binary_diagnostics = {
        "candidate_set_drift",
        "cross_event_moves",
        "cross_relevance_moves",
        "fact_moves",
        "cross_author_moves",
        "zero_to_nonzero",
        "external_api_delta",
    }
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
        + diagnostics["zero_to_nonzero"]
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
    "final_injected_ids",
    "outside_pool_ids",
    "arms",
    "request_path_delta_ms",
})


def validate_final_selection(value: object) -> dict[str, Any]:
    """Validate Twin's text-free post-filter A/B/C final selection receipt."""

    if type(value) is not dict or set(value) != _FINAL_SELECTION_KEYS:
        raise ValueError("invalid E chord final selection contract")
    if value.get("schema") != FINAL_SELECTION_SCHEMA:
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
    pool_ids = _validated_id_list(value.get("pool_ids"))
    final_ids = _validated_id_list(value.get("final_injected_ids"))
    outside_ids = _validated_id_list(value.get("outside_pool_ids"))
    if len(final_ids) > 32 or len(outside_ids) > 32:
        raise ValueError("invalid E chord final selection contract")
    pool = set(pool_ids)
    if [bucket_id for bucket_id in final_ids if bucket_id not in pool] != outside_ids:
        raise ValueError("invalid E chord final selection contract")
    arms = value.get("arms")
    if type(arms) is not dict or set(arms) != {"a", "b", "c"}:
        raise ValueError("invalid E chord final selection contract")
    for arm in ("a", "b", "c"):
        arm_ids = _validated_id_list(arms[arm])
        if len(arm_ids) > 32 or not set(arm_ids) <= pool:
            raise ValueError("invalid E chord final selection contract")
    if arms["b"] != [bucket_id for bucket_id in final_ids if bucket_id in pool]:
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
    "FINAL_SELECTION_SCHEMA",
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
    "validate_shadow_receipt",
    "validate_final_selection",
]
