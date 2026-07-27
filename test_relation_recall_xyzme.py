"""Y-axis wiring tests for safe, bounded, two-hop relation recall."""

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


def _configure(monkeypatch, allowed_types):
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "relation_recall": {
                "allowed_types": allowed_types,
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


def test_y_axis_walks_safe_edges_bidirectionally_and_two_hops(monkeypatch):
    _configure(monkeypatch, ["kin", "explains", "causes"])
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "b", "strength": 0.9},
            {"type": "causes", "target": "unsafe", "strength": 1.0},
        ]),
        _bucket("b", relations=[
            {"type": "explains", "target": "c", "strength": 0.8},
        ]),
        _bucket("c"),
        _bucket("unsafe"),
        _bucket("incoming", relations=[
            {"type": "kin", "target": "a", "strength": 0.7},
        ]),
        _bucket("other-world", world="role", relations=[
            {"type": "kin", "target": "a", "strength": 1.0},
        ]),
    ]

    found = _neighbors(buckets)
    by_id = {item.bucket_id: item for item in found}

    assert set(by_id) == {"b", "c", "incoming"}
    assert by_id["b"].depth == 1 and by_id["b"].direction == "out"
    assert by_id["incoming"].depth == 1 and by_id["incoming"].direction == "in"
    assert by_id["c"].depth == 2 and by_id["c"].via_id == "b"
    assert "unsafe" not in by_id and "other-world" not in by_id


def test_y_axis_cannot_reintroduce_filtered_historical_fact(monkeypatch):
    _configure(monkeypatch, ["kin"])
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "old", "strength": 1.0},
            {"type": "kin", "target": "current", "strength": 1.0},
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
    _configure(monkeypatch, "kin")
    buckets = [
        _bucket("a", relations=[
            {"type": "kin", "target": "seen", "strength": 1.0},
            {"type": "kin", "target": "fresh", "strength": 0.8},
        ]),
        _bucket("seen"),
        _bucket("fresh"),
    ]

    found = _neighbors(buckets, excluded={"seen"})
    assert [item.bucket_id for item in found] == ["fresh"]
