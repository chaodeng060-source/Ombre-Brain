"""Audited E-axis live projection for recall and response posture.

Model-produced E0 rows stay immutable proposals and permanently declare
``shadow_only=true`` and ``affects_ranking=false``.  They never feed the live
projection.  Live E rows come only from immutable bucket metadata written by a
named primary user-facing agent with its own initial priority.  This module
then proves that the row still binds to the current curated Markdown source.

The projection has three deliberately narrow effects:

* emotional resonance may break ties inside the existing relevance bands;
* at most a configured number of E-only memories may appear as labelled
  supporting experience, never as factual authority;
* recalled experience may produce a compact response-posture instruction.

Turning ``e_axis_recall.enabled`` off restores the legacy recall path without
changing buckets, proposals, relation edges, or fact lifecycle state.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from e_axis_curated_reader import bind_loaded_curated_source
from e_axis_shadow import validate_shadow_score


_ACTIVATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_NEGATIVE_HIGH = (
    "气死", "生气", "愤怒", "崩溃", "焦虑", "害怕", "恐慌", "受不了",
    "烦死", "委屈", "心痛", "绝望", "betrayed", "furious", "panic",
    "anxious", "devastated",
)
_NEGATIVE_LOW = (
    "不开心", "难过", "伤心", "失望", "孤独", "疲惫", "好累", "想哭",
    "没力气", "心灰", "sad", "lonely", "exhausted", "heartbreak",
)
_POSITIVE_HIGH = (
    "开心", "高兴", "兴奋", "激动", "好棒", "厉害", "爱你", "亲亲",
    "太好了", "happy", "excited", "love you", "amazing",
)
_POSITIVE_LOW = (
    "安心", "放心", "平静", "舒服", "温柔", "幸福", "放松", "踏实",
    "calm", "relaxed", "safe", "content",
)


def _plain_finite(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bounded_number(
    raw: object,
    *,
    name: str,
    low: float,
    high: float,
) -> float:
    value = _plain_finite(raw)
    if value is None or not low <= value <= high:
        raise ValueError(f"e_axis_recall.{name} must be in [{low}, {high}]")
    return value


def _bounded_int(
    raw: object,
    *,
    name: str,
    low: int,
    high: int,
) -> int:
    if type(raw) is not int or not low <= raw <= high:
        raise ValueError(f"e_axis_recall.{name} must be in [{low}, {high}]")
    return raw


@dataclass(frozen=True, slots=True)
class EAxisRecallConfig:
    enabled: bool = False
    activation_id: str = ""
    min_confidence: float = 0.5
    tie_break_weight: float = 0.2
    side_channel_limit: int = 1
    side_channel_scan_limit: int = 128
    side_channel_min_resonance: float = 0.55
    allowed_rubric_versions: tuple[str, ...] = ()


def load_e_axis_recall_config(root: Mapping[str, object]) -> EAxisRecallConfig:
    """Parse the live projection contract; malformed active config fails shut."""

    raw = root.get("e_axis_recall", {}) if isinstance(root, Mapping) else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("e_axis_recall must be a mapping")
    if raw.get("enabled") is not True or raw.get("mode") != "active":
        return EAxisRecallConfig()

    activation_id = raw.get("activation_id")
    if type(activation_id) is not str or _ACTIVATION_ID_RE.fullmatch(
        activation_id
    ) is None:
        raise ValueError("e_axis_recall.activation_id must be machine text")

    rubrics = raw.get("allowed_rubric_versions")
    if rubrics is None:
        shadow = root.get("e_axis_shadow", {})
        default_rubric = (
            shadow.get("rubric_version") if isinstance(shadow, Mapping) else None
        )
        rubrics = [default_rubric] if default_rubric else []
    if type(rubrics) is not list or not rubrics:
        raise ValueError("e_axis_recall.allowed_rubric_versions is required")
    normalized_rubrics: list[str] = []
    for value in rubrics:
        if type(value) is not str or not value.strip() or len(value) > 160:
            raise ValueError("invalid E live rubric version")
        if value not in normalized_rubrics:
            normalized_rubrics.append(value)

    return EAxisRecallConfig(
        enabled=True,
        activation_id=activation_id,
        min_confidence=_bounded_number(
            raw.get("min_confidence", 0.5),
            name="min_confidence",
            low=0.0,
            high=1.0,
        ),
        tie_break_weight=_bounded_number(
            raw.get("tie_break_weight", 0.2),
            name="tie_break_weight",
            low=0.0,
            high=0.25,
        ),
        side_channel_limit=_bounded_int(
            raw.get("side_channel_limit", 1),
            name="side_channel_limit",
            low=0,
            high=2,
        ),
        side_channel_scan_limit=_bounded_int(
            raw.get("side_channel_scan_limit", 128),
            name="side_channel_scan_limit",
            low=1,
            high=512,
        ),
        side_channel_min_resonance=_bounded_number(
            raw.get("side_channel_min_resonance", 0.55),
            name="side_channel_min_resonance",
            low=0.0,
            high=1.0,
        ),
        allowed_rubric_versions=tuple(normalized_rubrics),
    )


@dataclass(frozen=True, slots=True)
class QueryEmotion:
    valence: float
    arousal: float
    tension: float
    explicit: bool
    source: str


def infer_query_emotion(
    query: str,
    *,
    valence_01: float | None = None,
    arousal: float | None = None,
) -> QueryEmotion:
    """Infer a deterministic Russell coordinate without an extra model call.

    Explicit API coordinates win.  Otherwise a small audited cue table handles
    ordinary Chinese/English emotional wording.  Neutral queries still get a
    low-arousal prior so E can shape close topical ties, but they do not unlock
    the independent emotional side channel.
    """

    text = str(query or "").lower()
    if any(cue in text for cue in _NEGATIVE_HIGH):
        guessed = (-0.85, 0.85, 0.8, True, "lexicon.negative_high")
    elif any(cue in text for cue in _NEGATIVE_LOW):
        guessed = (-0.7, 0.35, 0.55, True, "lexicon.negative_low")
    elif any(cue in text for cue in _POSITIVE_HIGH):
        guessed = (0.8, 0.8, 0.2, True, "lexicon.positive_high")
    elif any(cue in text for cue in _POSITIVE_LOW):
        guessed = (0.65, 0.3, 0.1, True, "lexicon.positive_low")
    else:
        guessed = (0.0, 0.35, 0.2, False, "neutral_prior")

    explicit = False
    q_valence = _plain_finite(valence_01)
    if q_valence is not None and 0.0 <= q_valence <= 1.0:
        mapped_valence = q_valence * 2.0 - 1.0
        explicit = True
    else:
        mapped_valence = guessed[0]
    q_arousal = _plain_finite(arousal)
    if q_arousal is not None and 0.0 <= q_arousal <= 1.0:
        mapped_arousal = q_arousal
        explicit = True
    else:
        mapped_arousal = guessed[1]

    return QueryEmotion(
        valence=mapped_valence,
        arousal=mapped_arousal,
        tension=guessed[2],
        explicit=explicit or guessed[3],
        source="explicit" if explicit else guessed[4],
    )


@dataclass(frozen=True, slots=True)
class ActiveEAnnotation:
    bucket_id: str
    source_digest: str
    valence: float
    arousal: float
    tension: float
    confidence: float
    response_tendency: str
    growth_delta: str
    rubric_version: str
    scored_at: str
    authored_by: str = ""
    initial_priority: int = 0


def _row_timestamp(row: Mapping[str, object]) -> float:
    try:
        parsed = datetime.fromisoformat(str(row.get("scored_at") or ""))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return float("-inf")
        return parsed.timestamp()
    except (OverflowError, TypeError, ValueError):
        return float("-inf")


def group_candidate_rows(
    rows: Iterable[object],
    config: EAxisRecallConfig,
) -> dict[str, tuple[dict, ...]]:
    """Group valid success rows by current Ombre bucket id.

    Candidate-ledger and manual rows are intentionally excluded: only the
    official curated-memory cohort can become live recall evidence.
    """

    if not config.enabled:
        return {}
    grouped: dict[str, list[dict]] = {}
    for value in rows:
        if not isinstance(value, dict):
            continue
        if value.get("status") != "success" \
                or value.get("source_kind") != "curated_memory":
            continue
        raw_id = value.get("bucket_id")
        if type(raw_id) is not str or not raw_id.startswith("bucket:"):
            continue
        bucket_id = raw_id[len("bucket:"):]
        if not bucket_id:
            continue
        rubric = value.get("rubric_version")
        if rubric not in config.allowed_rubric_versions:
            continue
        score, error = validate_shadow_score(
            value.get("score"),
            min_confidence=config.min_confidence,
        )
        if error or score is None:
            continue
        normalized = dict(value)
        normalized["score"] = score
        grouped.setdefault(bucket_id, []).append(normalized)
    return {
        bucket_id: tuple(sorted(
            candidates,
            key=lambda row: (_row_timestamp(row), str(row.get("annotation_key"))),
            reverse=True,
        ))
        for bucket_id, candidates in grouped.items()
    }


def group_primary_authored_buckets(
    buckets: Iterable[object],
    config: EAxisRecallConfig,
    *,
    authored_by: str = "claude",
) -> dict[str, tuple[dict, ...]]:
    """Build live E rows for exactly one named primary agent.

    E records are first-person experience.  Mixing authors would let one
    agent's feelings steer another agent's response posture, so a missing
    author scope fails closed instead of widening to every authored bucket.
    """

    if not config.enabled:
        return {}
    expected_author = str(authored_by or "").strip()
    if not expected_author:
        return {}
    grouped: dict[str, list[dict]] = {}
    for bucket in buckets:
        if not isinstance(bucket, Mapping):
            continue
        metadata = bucket.get("metadata")
        content = bucket.get("content")
        if not isinstance(metadata, Mapping) or type(content) is not str:
            continue
        author = metadata.get("e_authored_by")
        priority = metadata.get("e_initial_priority")
        if (
            type(author) is not str
            or not author.strip()
            or author != author.strip()
            or author != expected_author
            or type(priority) is not int
            or isinstance(priority, bool)
            or not 1 <= priority <= 100
        ):
            continue
        binding = bind_loaded_curated_source(metadata, content)
        if binding is None or binding.bucket_id != str(bucket.get("id") or ""):
            continue
        score = {
            "valence": metadata.get("e_valence"),
            "arousal": metadata.get("e_arousal"),
            "tension": metadata.get("e_tension"),
            "confidence": metadata.get("e_confidence"),
            "response_tendency": metadata.get("e_response_tendency"),
            "growth_delta": metadata.get("e_growth_delta"),
        }
        normalized_score, error = validate_shadow_score(
            score,
            min_confidence=config.min_confidence,
        )
        if error or normalized_score is None:
            continue
        row = {
            "status": "success",
            "authority": "primary_agent",
            "source_kind": "primary_authored",
            "bucket_id": "bucket:" + binding.bucket_id,
            "source_digest": binding.source_digest,
            "score": normalized_score,
            "rubric_version": "primary-authored/v1",
            "scored_at": str(metadata.get("e_authored_at") or ""),
            "e_authored_by": author,
            "e_initial_priority": priority,
        }
        grouped.setdefault(binding.bucket_id, []).append(row)
    return {
        bucket_id: tuple(sorted(
            rows,
            key=lambda row: (
                int(row["e_initial_priority"]),
                _row_timestamp(row),
            ),
            reverse=True,
        ))
        for bucket_id, rows in grouped.items()
    }


def select_current_annotation(
    rows: Sequence[Mapping[str, object]],
    bucket: Mapping[str, object],
    config: EAxisRecallConfig,
) -> ActiveEAnnotation | None:
    """Select the newest row whose digest still binds to this exact bucket."""

    if not config.enabled or not rows:
        return None
    metadata = bucket.get("metadata")
    content = bucket.get("content")
    if not isinstance(metadata, Mapping) or type(content) is not str:
        return None
    binding = bind_loaded_curated_source(metadata, content)
    if binding is None or str(bucket.get("id") or "") != binding.bucket_id:
        return None
    expected_id = "bucket:" + binding.bucket_id
    for row in rows:
        if row.get("bucket_id") != expected_id \
                or row.get("source_digest") != binding.source_digest:
            continue
        score, error = validate_shadow_score(
            row.get("score"),
            min_confidence=config.min_confidence,
        )
        if error or score is None:
            continue
        return ActiveEAnnotation(
            bucket_id=binding.bucket_id,
            source_digest=binding.source_digest,
            valence=score["valence"],
            arousal=score["arousal"],
            tension=score["tension"],
            confidence=score["confidence"],
            response_tendency=score["response_tendency"],
            growth_delta=score["growth_delta"],
            rubric_version=str(row.get("rubric_version") or ""),
            scored_at=str(row.get("scored_at") or ""),
            authored_by=str(row.get("e_authored_by") or ""),
            initial_priority=int(row.get("e_initial_priority") or 0),
        )
    return None


def resonance_score(
    query: QueryEmotion,
    annotation: ActiveEAnnotation | Mapping[str, object],
) -> float:
    """Return bounded Russell-space resonance multiplied by confidence."""

    if isinstance(annotation, ActiveEAnnotation):
        score: Mapping[str, object] = {
            "valence": annotation.valence,
            "arousal": annotation.arousal,
            "tension": annotation.tension,
            "confidence": annotation.confidence,
        }
    else:
        nested = annotation.get("score")
        score = nested if isinstance(nested, Mapping) else annotation
    values = {
        name: _plain_finite(score.get(name))
        for name in ("valence", "arousal", "tension", "confidence")
    }
    if any(value is None for value in values.values()):
        return 0.0
    valence = max(-1.0, min(1.0, values["valence"]))
    arousal = max(0.0, min(1.0, values["arousal"]))
    tension = max(0.0, min(1.0, values["tension"]))
    confidence = max(0.0, min(1.0, values["confidence"]))
    distance = math.sqrt(
        ((query.valence - valence) / 2.0) ** 2
        + (query.arousal - arousal) ** 2
    ) / math.sqrt(2.0)
    russell = max(0.0, 1.0 - distance)
    tension_similarity = max(0.0, 1.0 - abs(query.tension - tension))
    return round((0.8 * russell + 0.2 * tension_similarity) * confidence, 6)


def apply_resonance_tie_break(
    base_score: float,
    resonance: float,
    *,
    weight: float,
) -> float:
    """Bound E's effect; relevance-band membership is enforced by the caller."""

    base = _plain_finite(base_score)
    signal = _plain_finite(resonance)
    bounded_weight = _plain_finite(weight)
    if base is None or signal is None or bounded_weight is None:
        return float(base or 0.0)
    signal = max(0.0, min(1.0, signal))
    bounded_weight = max(0.0, min(0.25, bounded_weight))
    return round(base * (1.0 + bounded_weight * signal), 6)


@dataclass(frozen=True, slots=True)
class ResponsePosture:
    tendency: str
    growth_delta: str
    valence: float
    arousal: float
    tension: float
    confidence: float
    evidence_count: int


def derive_response_posture(
    evidence: Iterable[tuple[ActiveEAnnotation, float]],
) -> ResponsePosture | None:
    """Aggregate selected live E evidence into one non-factual posture."""

    rows = [
        (annotation, max(0.0, min(1.0, float(resonance))))
        for annotation, resonance in evidence
        if isinstance(annotation, ActiveEAnnotation)
    ]
    if not rows:
        return None
    weighted = [
        (annotation, max(0.000001, resonance * annotation.confidence))
        for annotation, resonance in rows
    ]
    total = sum(weight for _, weight in weighted)
    tendency_votes: dict[str, float] = {}
    growth_votes: dict[str, float] = {}
    for annotation, weight in weighted:
        tendency_votes[annotation.response_tendency] = (
            tendency_votes.get(annotation.response_tendency, 0.0) + weight
        )
        growth_votes[annotation.growth_delta] = (
            growth_votes.get(annotation.growth_delta, 0.0) + weight
        )
    tendency = max(tendency_votes, key=lambda key: (tendency_votes[key], key))
    growth = max(growth_votes, key=lambda key: (growth_votes[key], key))

    def average(field: str) -> float:
        return round(
            sum(getattr(annotation, field) * weight for annotation, weight in weighted)
            / total,
            3,
        )

    return ResponsePosture(
        tendency=tendency,
        growth_delta=growth,
        valence=average("valence"),
        arousal=average("arousal"),
        tension=average("tension"),
        confidence=average("confidence"),
        evidence_count=len(weighted),
    )


_POSTURE_TEXT = {
    "comfort": "先承接情绪并给稳定感，再处理事实或任务。",
    "engage": "主动回应并延续共同语境，保持具体而有参与感。",
    "withdraw": "降低刺激和压迫感，留出空间，不强推结论。",
    "alert": "先确认边界与风险，语气清楚稳定，再采取行动。",
}
_GROWTH_TEXT = {
    "growth": "延续已形成的正向习惯。",
    "stable": "保持既有相处方式，不擅自升级含义。",
    "setback": "避免重复曾造成挫败的回应模式。",
}


def format_response_posture(
    posture: ResponsePosture,
    *,
    activation_id: str,
) -> str:
    """Render a compact prompt block whose evidence role is explicit."""

    return (
        "=== E轴回应姿态（experience only，不可改写事实） ===\n"
        f"[e_activation:{activation_id}] "
        f"[tendency:{posture.tendency}] "
        f"[growth:{posture.growth_delta}] "
        f"[confidence:{posture.confidence:.3f}] "
        f"[evidence:{posture.evidence_count}]\n"
        f"{_POSTURE_TEXT[posture.tendency]} "
        f"{_GROWTH_TEXT[posture.growth_delta]} "
        "此块只约束回应姿态；人物、时间、事实与安全边界仍以主召回和 Z 轴为准。"
    )


def rank_annotation_bucket_ids(
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
    query: QueryEmotion,
    *,
    limit: int,
) -> list[str]:
    """Cheap pre-rank before current-source hydration and digest validation."""

    scored: list[tuple[str, float, int, float]] = []
    for bucket_id, rows in grouped.items():
        if not rows:
            continue
        row = rows[0]
        priority = row.get("e_initial_priority")
        safe_priority = priority if type(priority) is int else 0
        scored.append((
            bucket_id,
            resonance_score(query, row),
            safe_priority,
            _row_timestamp(row),
        ))
    scored.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
    return [bucket_id for bucket_id, _, _, _ in scored[:max(0, limit)]]


__all__ = [
    "ActiveEAnnotation",
    "EAxisRecallConfig",
    "QueryEmotion",
    "ResponsePosture",
    "apply_resonance_tie_break",
    "derive_response_posture",
    "format_response_posture",
    "group_candidate_rows",
    "group_primary_authored_buckets",
    "infer_query_emotion",
    "load_e_axis_recall_config",
    "rank_annotation_bucket_ids",
    "resonance_score",
    "select_current_annotation",
]
