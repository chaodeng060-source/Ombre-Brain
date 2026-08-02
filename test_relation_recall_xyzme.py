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


def _neighbors(buckets, *, query="我们之间的关系", intent="relation", excluded=()):
    return server._relation_recall_neighbors(
        buckets,
        ["a"],
        query=query,
        intent=intent,
        world_filter=["daily"],
        domain_filter=None,
        created_after=None,
        created_before=None,
        max_depth=2,
        max_results=8,
        excluded_ids=excluded,
    )


def test_y_axis_walks_propagation_edges_bidirectionally_and_two_hops(monkeypatch):
    _configure(monkeypatch, ["kin", "explains", "causes"])
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "semantic-only", "strength": 0.9},
            {"type": "causes", "target": "b", "strength": 1.0},
        ]),
        _bucket("b", relations=[
            {"type": "explains", "target": "c", "strength": 0.8},
        ]),
        _bucket("c"),
        _bucket("semantic-only"),
        _bucket("incoming", relations=[
            {"type": "causes", "target": "a", "strength": 0.7},
        ]),
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

    found = _neighbors(
        buckets,
        query="具体地址是多少",
        intent="fact",
    )
    assert [item.bucket_id for item in found] == ["current"]


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


def test_semantic_edges_never_expand_even_when_configured(monkeypatch):
    _configure(monkeypatch, ["kin", "updates"])
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "same-topic", "strength": 1.0},
            {"type": "updates", "target": "old-state", "strength": 1.0},
        ]),
        _bucket("same-topic"),
        _bucket("old-state"),
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
        ]),
        _bucket("same-topic"),
        _bucket("hard-edge"),
    ]

    assert [item.bucket_id for item in _neighbors(buckets)] == ["hard-edge"]
