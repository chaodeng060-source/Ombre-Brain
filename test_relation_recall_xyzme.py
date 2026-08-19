"""Y-axis wiring tests for classified, bounded relation recall."""

import server


def _bucket(
    bucket_id,
    *,
    relations=None,
    world="daily",
    fact_key="",
    fact_status="",
):
    metadata = {
        "name": bucket_id,
        "world": world,
        "domain": ["生活"],
        "relations": relations or [],
    }
    if fact_key:
        metadata["fact_key"] = fact_key
    if fact_status:
        metadata["fact_status"] = fact_status
    return {"id": bucket_id, "metadata": metadata, "content": bucket_id}


def _configure(
    monkeypatch,
    propagation_types,
    *,
    propagation_only=True,
    allowed_types=("kin", "explains"),
):
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "relation_recall": {
                "propagation_only": propagation_only,
                "allowed_types": allowed_types,
                "propagation_types": propagation_types,
                "hop1_min_strength": 0.4,
                "hop2_min_strength": 0.7,
            },
            "fact_slots": {
                "enabled": True,
                "registry": {"profile.city": {"aliases": ["城市"]}},
            },
        },
    )


def _neighbors(
    buckets,
    *,
    query="我们之间的关系",
    intent="relation",
    excluded=(),
    max_depth=2,
    max_results=8,
):
    return server._relation_recall_neighbors(
        buckets,
        ["a"],
        query=query,
        intent=intent,
        world_filter=["daily"],
        domain_filter=None,
        created_after=None,
        created_before=None,
        max_depth=max_depth,
        max_results=max_results,
        excluded_ids=excluded,
    )


def test_y_axis_walks_propagation_edges_bidirectionally_and_two_hops(monkeypatch):
    _configure(
        monkeypatch,
        ["kin", "explains", "causes", "contributes", "improves", "updates"],
    )
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "semantic-only", "strength": 0.9},
            {"type": "explains", "target": "b", "strength": 1.0},
            {"type": "causes", "target": "review-causes", "strength": 1.0},
            {"type": "contributes", "target": "review-contributes", "strength": 1.0},
            {"type": "improves", "target": "review-improves", "strength": 1.0},
            {"type": "updates", "target": "storage-update", "strength": 1.0},
        ]),
        _bucket("b", relations=[
            {"type": "explains", "target": "c", "strength": 0.8},
        ]),
        _bucket("c"),
        _bucket("semantic-only"),
        _bucket("incoming", relations=[
            {"type": "explains", "target": "a", "strength": 0.7},
        ]),
        _bucket("review-causes"),
        _bucket("review-contributes"),
        _bucket("review-improves"),
        _bucket("storage-update"),
        _bucket("other-world", world="role", relations=[
            {"type": "explains", "target": "a", "strength": 1.0},
        ]),
    ]

    found = _neighbors(buckets)
    by_id = {item.bucket_id: item for item in found}

    assert set(by_id) == {"b", "c", "incoming"}
    assert by_id["b"].depth == 1 and by_id["b"].direction == "out"
    assert by_id["incoming"].depth == 1 and by_id["incoming"].direction == "in"
    assert by_id["c"].depth == 2 and by_id["c"].via_id == "b"
    assert "semantic-only" not in by_id and "other-world" not in by_id


def test_y_axis_cannot_reintroduce_filtered_historical_fact(monkeypatch):
    _configure(monkeypatch, ["explains"])
    buckets = [
        _bucket("a", relations=[
            {"type": "explains", "target": "old", "strength": 1.0},
            {"type": "explains", "target": "current", "strength": 1.0},
        ]),
        _bucket(
            "old",
            fact_key="profile.city",
            fact_status="historical",
        ),
        _bucket(
            "current",
            fact_key="profile.city",
            fact_status="current",
        ),
    ]

    # 明确命中「城市」槽：Y 轴扩展不能把被 Z 闸压掉的 historical 再带回来。
    found = _neighbors(
        buckets,
        query="现在城市是哪个",
        intent="fact",
    )
    assert [item.bucket_id for item in found] == ["current"]

    # 2026-08-19 复核 P1-3：没命中任何已登记槽的 fact 查询，Z 闸不启用（fail-open），
    # Y 轴扩展照常带回两端——旧断言把空 fact_keys 当「所有槽」过滤是 bug。
    found_neutral = _neighbors(
        buckets,
        query="具体地址是多少",
        intent="fact",
    )
    assert sorted(item.bucket_id for item in found_neutral) == ["current", "old"]


def test_y_axis_honors_session_dedup_and_scalar_config(monkeypatch):
    _configure(monkeypatch, "explains")
    buckets = [
        _bucket("a", relations=[
            {"type": "explains", "target": "seen", "strength": 1.0},
            {"type": "explains", "target": "fresh", "strength": 0.8},
        ]),
        _bucket("seen"),
        _bucket("fresh"),
    ]

    found = _neighbors(buckets, excluded={"seen"})
    assert [item.bucket_id for item in found] == ["fresh"]


def test_storage_only_review_and_unknown_edges_never_expand_even_when_configured(
    monkeypatch,
):
    _configure(
        monkeypatch,
        ["kin", "updates", "causes", "contributes", "improves", "future_type"],
    )
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "same-topic", "strength": 1.0},
            {"type": "updates", "target": "old-state", "strength": 1.0},
            {"type": "causes", "target": "cause", "strength": 1.0},
            {"type": "contributes", "target": "support", "strength": 1.0},
            {"type": "improves", "target": "improvement", "strength": 1.0},
            {"type": "future_type", "target": "unknown", "strength": 1.0},
        ]),
        _bucket("same-topic"),
        _bucket("old-state"),
        _bucket("cause"),
        _bucket("support"),
        _bucket("improvement"),
        _bucket("unknown"),
    ]

    assert _neighbors(buckets) == []


def test_switch_off_restores_exact_legacy_safe_edge_behavior(monkeypatch):
    _configure(
        monkeypatch,
        ["causes", "contributes", "improves", "explains"],
        propagation_only=False,
        allowed_types=["kin", "explains", "causes"],
    )
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "same-topic", "strength": 1.0},
            {"type": "explains", "target": "explanation", "strength": 1.0},
            {"type": "causes", "target": "hard-edge", "strength": 1.0},
        ]),
        _bucket("same-topic"),
        _bucket("explanation"),
        _bucket("hard-edge"),
    ]

    assert {item.bucket_id for item in _neighbors(buckets)} == {
        "same-topic",
        "explanation",
    }


def test_switch_on_ignores_stale_legacy_allowed_types(monkeypatch):
    _configure(
        monkeypatch,
        ["causes", "explains"],
        propagation_only=True,
        allowed_types=["kin", "explains"],
    )
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "same-topic", "strength": 1.0},
            {"type": "causes", "target": "hard-edge", "strength": 1.0},
            {"type": "explains", "target": "explanation", "strength": 1.0},
        ]),
        _bucket("same-topic"),
        _bucket("hard-edge"),
        _bucket("explanation"),
    ]

    assert [item.bucket_id for item in _neighbors(buckets)] == ["explanation"]


def test_missing_switch_defaults_to_propagation_policy(monkeypatch):
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "relation_recall": {
                "allowed_types": ["kin", "explains"],
                "propagation_types": [
                    "kin",
                    "explains",
                    "causes",
                    "contributes",
                    "improves",
                    "updates",
                ],
                "hop1_min_strength": 0.4,
                "hop2_min_strength": 0.7,
            },
        },
    )
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "same-topic", "strength": 1.0},
            {"type": "explains", "target": "explanation", "strength": 1.0},
            {"type": "causes", "target": "hard-edge", "strength": 1.0},
        ]),
        _bucket("same-topic"),
        _bucket("explanation"),
        _bucket("hard-edge"),
    ]

    assert [item.bucket_id for item in _neighbors(buckets)] == ["explanation"]


def test_switch_true_false_true_is_exact_and_reversible(monkeypatch):
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "same-topic", "strength": 1.0},
            {"type": "explains", "target": "explanation", "strength": 1.0},
            {"type": "causes", "target": "hard-edge", "strength": 1.0},
        ]),
        _bucket("same-topic"),
        _bucket("explanation"),
        _bucket("hard-edge"),
    ]

    _configure(
        monkeypatch,
        ["kin", "explains", "causes"],
        propagation_only=True,
    )
    assert [item.bucket_id for item in _neighbors(buckets)] == ["explanation"]

    _configure(
        monkeypatch,
        ["kin", "explains", "causes"],
        propagation_only=False,
        allowed_types=["kin", "explains", "causes"],
    )
    assert {item.bucket_id for item in _neighbors(buckets)} == {
        "same-topic",
        "explanation",
    }

    _configure(
        monkeypatch,
        ["kin", "explains", "causes"],
        propagation_only=True,
    )
    assert [item.bucket_id for item in _neighbors(buckets)] == ["explanation"]


def test_relation_depth_zero_and_max_results_remain_bounded(monkeypatch):
    _configure(monkeypatch, ["explains"])
    buckets = [
        _bucket("a", relations=[
            {"type": "explains", "target": "b", "strength": 1.0},
            {"type": "explains", "target": "c", "strength": 0.9},
            {"type": "explains", "target": "d", "strength": 0.8},
        ]),
        _bucket("b"),
        _bucket("c"),
        _bucket("d"),
    ]

    assert _neighbors(buckets, max_depth=0) == []
    assert len(_neighbors(buckets, max_results=2)) == 2
