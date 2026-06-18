"""Intent-aware recall policy helpers.

This module is deliberately pure: no bucket I/O, no embedding calls, no LLM.
The server asks it one question per search: what recall budget should each
channel get for this query?
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

INTENT_DEFAULT = "default"
INTENT_FACT = "fact"
INTENT_RECALL = "recall"
INTENT_RELATION = "relation"
INTENT_TEMPORAL = "temporal"


DEFAULT_INTENT_RECALL_CONFIG: dict[str, Any] = {
    "enabled": True,
    "min_confidence": 0.55,
    "policies": {
        # Mirrors the pre-kernel-4 behavior: same pool sizes, same RRF weights,
        # relation expansion still capped at 5.
        INTENT_DEFAULT: {
            "keyword_top_k_multiplier": 1.0,
            "vector_top_k_multiplier": 1.0,
            "keyword_weight_multiplier": 1.0,
            "vector_weight_multiplier": 1.0,
            "relation_neighbor_limit": 5,
            "relation_score_boost": 1.0,
            "emotion_domain_boost": 1.0,
            "temporal_recent_boost": 1.0,
            "temporal_recent_days": 14,
        },
        INTENT_FACT: {
            "keyword_top_k_multiplier": 1.35,
            "vector_top_k_multiplier": 1.15,
            "keyword_weight_multiplier": 1.35,
            "vector_weight_multiplier": 1.10,
            "relation_neighbor_limit": 1,
            "relation_score_boost": 1.0,
            "emotion_domain_boost": 1.0,
            "temporal_recent_boost": 1.0,
            "temporal_recent_days": 14,
        },
        INTENT_RECALL: {
            "keyword_top_k_multiplier": 1.0,
            "vector_top_k_multiplier": 1.25,
            "keyword_weight_multiplier": 1.0,
            "vector_weight_multiplier": 1.15,
            "relation_neighbor_limit": 4,
            "relation_score_boost": 1.0,
            "emotion_domain_boost": 1.0,
            "temporal_recent_boost": 1.0,
            "temporal_recent_days": 14,
        },
        INTENT_RELATION: {
            "keyword_top_k_multiplier": 1.0,
            "vector_top_k_multiplier": 1.35,
            "keyword_weight_multiplier": 0.90,
            "vector_weight_multiplier": 1.20,
            "relation_neighbor_limit": 8,
            # 2026-06-14 实测调参：原 1.20/1.15(=×1.38)用力过猛，会把"有关系边"的
            # 桶硬顶到真答案前面（q_rel_trust 1→2 回归）。降成轻量打破平手的级别。
            "relation_score_boost": 1.05,
            "emotion_domain_boost": 1.05,
            "temporal_recent_boost": 1.0,
            "temporal_recent_days": 14,
        },
        INTENT_TEMPORAL: {
            "keyword_top_k_multiplier": 1.30,
            "vector_top_k_multiplier": 0.90,
            "keyword_weight_multiplier": 1.25,
            "vector_weight_multiplier": 0.85,
            "relation_neighbor_limit": 2,
            "relation_score_boost": 1.0,
            "emotion_domain_boost": 1.0,
            "temporal_recent_boost": 1.15,
            "temporal_recent_days": 14,
        },
    },
}


_FACT_TERMS = (
    "哪天", "哪日", "几号", "日期", "什么时候", "具体", "准确", "精确",
    "生日", "生理期", "经期", "电话", "地址", "名字", "是谁", "多少",
    "deadline", "due date", "exact", "when was",
)
_RECALL_TERMS = (
    "回顾", "总结", "概括", "复盘", "想起来", "记得", "之前聊过",
    "讲过什么", "发生过什么", "哪些事", "recap", "summary",
)
_RELATION_TERMS = (
    "我俩", "我们俩", "我们之间", "关系", "相处", "亲密", "恋爱", "感情",
    "她对我", "他对我", "在乎", "信任", "靠近", "疏远",
    "情绪", "感觉", "最近怎么样", "relationship", "between us",
)
_TEMPORAL_TERMS = (
    "最近", "这几天", "这段时间", "这周", "上周", "昨天", "前天", "今天",
    "刚才", "刚刚", "上次", "之前", "后来", "之后", "时间线", "timeline",
    "变化", "趋势", "顺序", "先后",
)

_TIE_PRIORITY = {
    INTENT_FACT: 4,
    INTENT_RELATION: 3,
    INTENT_TEMPORAL: 2,
    INTENT_RECALL: 1,
}

_RELATION_META_TERMS = {
    "恋爱", "关系", "感情", "亲密", "家庭", "约定", "自省", "信任",
    "社交", "心理", "feel", "沉淀物",
}


def classify_query_intent(query: str) -> dict[str, Any]:
    """Classify a query into the small intent set used by recall policies."""
    q = str(query or "").strip().lower()
    if not q:
        return {
            "intent": INTENT_DEFAULT,
            "confidence": 0.0,
            "scores": {},
            "matched_terms": [],
        }

    term_groups = {
        INTENT_FACT: _FACT_TERMS,
        INTENT_RECALL: _RECALL_TERMS,
        INTENT_RELATION: _RELATION_TERMS,
        INTENT_TEMPORAL: _TEMPORAL_TERMS,
    }
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for intent, terms in term_groups.items():
        hits = [term for term in terms if term.lower() in q]
        if hits:
            matched[intent] = hits
            scores[intent] = float(len(hits))

    # A few compound signals are stronger than isolated words.
    if ("我俩" in q or "我们俩" in q or "我们之间" in q) and ("怎么样" in q or "最近" in q):
        scores[INTENT_RELATION] = scores.get(INTENT_RELATION, 0.0) + 2.0
        matched.setdefault(INTENT_RELATION, []).append("relationship_compound")
    if ("哪天" in q or "几号" in q) and ("生理期" in q or "经期" in q):
        scores[INTENT_FACT] = scores.get(INTENT_FACT, 0.0) + 2.0
        matched.setdefault(INTENT_FACT, []).append("period_date_compound")
    if "时间线" in q or "timeline" in q:
        scores[INTENT_TEMPORAL] = scores.get(INTENT_TEMPORAL, 0.0) + 1.5

    if not scores:
        return {
            "intent": INTENT_DEFAULT,
            "confidence": 0.0,
            "scores": {},
            "matched_terms": [],
        }

    ranked = sorted(
        scores.items(),
        key=lambda item: (item[1], _TIE_PRIORITY.get(item[0], 0)),
        reverse=True,
    )
    intent, top_score = ranked[0]
    next_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = max(0.0, top_score - next_score)
    confidence = min(0.95, 0.45 + top_score * 0.10 + margin * 0.05)

    return {
        "intent": intent,
        "confidence": round(confidence, 3),
        "scores": scores,
        "matched_terms": matched.get(intent, []),
    }


def resolve_intent_recall_policy(
    query: str,
    config: dict[str, Any] | None,
    base_recall_limit: int,
    requested_relation_depth: int,
) -> dict[str, Any]:
    """Return the effective recall policy for a query."""
    config = config or {}
    recall_cfg = _deep_merge(
        DEFAULT_INTENT_RECALL_CONFIG,
        config.get("intent_recall", {}) or {},
    )
    rrf_cfg = config.get("rrf", {}) or {}
    classification = classify_query_intent(query)

    policies = recall_cfg.get("policies", {}) or {}
    default_policy = dict(DEFAULT_INTENT_RECALL_CONFIG["policies"][INTENT_DEFAULT])
    default_policy = _deep_merge(default_policy, policies.get(INTENT_DEFAULT, {}) or {})

    intent = classification["intent"]
    effective_intent = INTENT_DEFAULT
    if (
        recall_cfg.get("enabled", True)
        and intent in policies
        and classification["confidence"] >= float(recall_cfg.get("min_confidence", 0.55))
    ):
        effective_intent = intent

    policy = _deep_merge(default_policy, policies.get(effective_intent, {}) or {})
    keyword_base = float(rrf_cfg.get("keyword_weight", 1.0))
    vector_base = float(rrf_cfg.get("vector_weight", 1.0))

    keyword_weight = policy.get("keyword_weight")
    if keyword_weight is None:
        keyword_weight = keyword_base * float(policy.get("keyword_weight_multiplier", 1.0))
    vector_weight = policy.get("vector_weight")
    if vector_weight is None:
        vector_weight = vector_base * float(policy.get("vector_weight_multiplier", 1.0))

    relation_depth = max(0, int(requested_relation_depth or 0))
    if relation_depth > 0:
        relation_depth = max(
            relation_depth,
            int(policy.get("relation_depth_min", 0) or 0),
        )

    return {
        **policy,
        "intent": effective_intent,
        "classified_intent": intent,
        "confidence": classification["confidence"],
        "matched_terms": classification["matched_terms"],
        "keyword_top_k": _scaled_limit(
            base_recall_limit,
            policy.get("keyword_top_k"),
            policy.get("keyword_top_k_multiplier", 1.0),
        ),
        "vector_top_k": _scaled_limit(
            base_recall_limit,
            policy.get("vector_top_k"),
            policy.get("vector_top_k_multiplier", 1.0),
        ),
        "keyword_weight": float(keyword_weight),
        "vector_weight": float(vector_weight),
        "relation_depth": relation_depth,
        "relation_neighbor_limit": max(0, int(policy.get("relation_neighbor_limit", 5))),
    }


def bucket_intent_score_multiplier(
    bucket_or_meta: dict[str, Any],
    policy: dict[str, Any],
    now: datetime | None = None,
) -> float:
    """Return a post-RRF score multiplier for intent-specific channels."""
    intent = (policy or {}).get("intent", INTENT_DEFAULT)
    if intent == INTENT_DEFAULT:
        return 1.0

    meta = bucket_or_meta.get("metadata", bucket_or_meta) if isinstance(bucket_or_meta, dict) else {}
    if not isinstance(meta, dict):
        return 1.0

    multiplier = 1.0
    if intent == INTENT_RELATION:
        relations = meta.get("relations") or []
        if isinstance(relations, list) and relations:
            multiplier *= float(policy.get("relation_score_boost", 1.0))
        if _metadata_terms(meta) & _RELATION_META_TERMS:
            multiplier *= float(policy.get("emotion_domain_boost", 1.0))

    if intent == INTENT_TEMPORAL:
        recent_days = float(policy.get("temporal_recent_days", 14) or 14)
        if _is_recent(meta, recent_days, now=now):
            multiplier *= float(policy.get("temporal_recent_boost", 1.0))

    return max(0.0, multiplier)


def _scaled_limit(base: int, explicit: Any, multiplier: Any) -> int:
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            pass
    try:
        factor = float(multiplier)
    except (TypeError, ValueError):
        factor = 1.0
    return max(1, int(math.ceil(max(1, int(base)) * factor)))


def _metadata_terms(meta: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for key in ("domain", "tags", "sense"):
        value = meta.get(key, [])
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            values = []
        terms.update(str(v).strip() for v in values if str(v).strip())
    bucket_type = str(meta.get("type", "")).strip()
    if bucket_type:
        terms.add(bucket_type)
    return terms


def _is_recent(meta: dict[str, Any], days: float, now: datetime | None = None) -> bool:
    raw = meta.get("event_at") or meta.get("created") or meta.get("last_active") or ""
    if not raw:
        return False
    try:
        timestamp = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return False
    now = now or datetime.now(timestamp.tzinfo)
    return (now - timestamp).total_seconds() <= days * 86400


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
