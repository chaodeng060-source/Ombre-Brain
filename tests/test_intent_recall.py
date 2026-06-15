from datetime import datetime, timedelta

import pytest

from intent_recall import (
    INTENT_DEFAULT,
    INTENT_FACT,
    INTENT_RELATION,
    INTENT_TEMPORAL,
    bucket_intent_score_multiplier,
    classify_query_intent,
    resolve_intent_recall_policy,
)


def test_period_date_query_is_fact_intent():
    classified = classify_query_intent("她生理期哪天")
    assert classified["intent"] == INTENT_FACT
    assert classified["confidence"] >= 0.55


def test_relationship_state_query_is_relation_intent():
    classified = classify_query_intent("我俩最近怎么样")
    assert classified["intent"] == INTENT_RELATION
    assert classified["confidence"] >= 0.55


def test_unknown_query_falls_back_to_default_policy():
    policy = resolve_intent_recall_policy(
        "工程",
        {"rrf": {"keyword_weight": 1.7, "vector_weight": 0.8}},
        base_recall_limit=20,
        requested_relation_depth=1,
    )

    assert policy["intent"] == INTENT_DEFAULT
    assert policy["keyword_top_k"] == 20
    assert policy["vector_top_k"] == 20
    assert policy["keyword_weight"] == pytest.approx(1.7)
    assert policy["vector_weight"] == pytest.approx(0.8)
    assert policy["relation_neighbor_limit"] == 5


def test_fact_policy_expands_precise_channels_and_limits_relations():
    policy = resolve_intent_recall_policy(
        "她生理期哪天",
        {"rrf": {"keyword_weight": 1.0, "vector_weight": 1.0}},
        base_recall_limit=20,
        requested_relation_depth=1,
    )

    assert policy["intent"] == INTENT_FACT
    assert policy["keyword_top_k"] == 27
    assert policy["vector_top_k"] == 23
    assert policy["keyword_weight"] > policy["vector_weight"]
    assert policy["relation_neighbor_limit"] == 1


def test_disabled_intent_recall_keeps_default_behavior():
    policy = resolve_intent_recall_policy(
        "我俩最近怎么样",
        {
            "rrf": {"keyword_weight": 1.0, "vector_weight": 1.0},
            "intent_recall": {"enabled": False},
        },
        base_recall_limit=20,
        requested_relation_depth=1,
    )

    assert policy["intent"] == INTENT_DEFAULT
    assert policy["keyword_top_k"] == 20
    assert policy["vector_top_k"] == 20
    assert policy["keyword_weight"] == pytest.approx(1.0)
    assert policy["vector_weight"] == pytest.approx(1.0)


def test_relation_policy_boosts_related_domains_and_edges():
    policy = resolve_intent_recall_policy(
        "我俩最近怎么样",
        {"rrf": {"keyword_weight": 1.0, "vector_weight": 1.0}},
        base_recall_limit=20,
        requested_relation_depth=1,
    )

    meta = {
        "domain": ["恋爱"],
        "relations": [{"type": "kin", "target": "other"}],
    }
    assert policy["intent"] == INTENT_RELATION
    assert policy["relation_neighbor_limit"] == 8
    # 2026-06-14 调参同步：relation 1.20→1.05、emotion 1.15→1.05（×1.38→×1.1025），
    # 原 1.38 用力过猛会把"有关系边"的桶硬顶到真答案前面。
    assert bucket_intent_score_multiplier(meta, policy) == pytest.approx(1.1025)


def test_temporal_policy_boosts_recent_bucket_only():
    policy = resolve_intent_recall_policy(
        "最近这段时间的变化",
        {"rrf": {"keyword_weight": 1.0, "vector_weight": 1.0}},
        base_recall_limit=20,
        requested_relation_depth=1,
    )
    now = datetime(2026, 6, 14, 12, 0, 0)
    recent = {"created": (now - timedelta(days=2)).isoformat()}
    old = {"created": (now - timedelta(days=40)).isoformat()}

    assert policy["intent"] == INTENT_TEMPORAL
    assert bucket_intent_score_multiplier(recent, policy, now=now) > 1.0
    assert bucket_intent_score_multiplier(old, policy, now=now) == 1.0
