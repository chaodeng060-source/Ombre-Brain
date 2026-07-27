"""Z-axis integration tests for canonical fact slots and recall gating."""

import server


def _bucket(bucket_id, *, key="", status="", protected=False):
    metadata = {"name": bucket_id}
    if key:
        metadata["fact_key"] = key
    if status:
        metadata["fact_status"] = status
    if protected:
        metadata["protected"] = True
    return {"id": bucket_id, "metadata": metadata, "content": bucket_id}


def _configure(monkeypatch):
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "fact_slots": {
                "enabled": True,
                "registry": {
                    "profile.city": {"aliases": ["城市"]},
                },
            },
        },
    )


def test_exact_fact_recall_hides_only_registered_historical(monkeypatch):
    _configure(monkeypatch)
    buckets = [
        _bucket("current", key="profile.city", status="current"),
        _bucket("historical", key="profile.city", status="historical"),
        _bucket("contested", key="profile.city", status="contested"),
        _bucket("unknown", key="unregistered.city", status="historical"),
        _bucket("protected", key="profile.city", status="historical", protected=True),
    ]

    kept = server._filter_z_fact_candidates(
        buckets,
        query="具体地址是多少",
        intent="fact",
    )

    assert [bucket["id"] for bucket in kept] == [
        "current",
        "contested",
        "unknown",
        "protected",
    ]


def test_history_and_non_fact_queries_keep_historical(monkeypatch):
    _configure(monkeypatch)
    buckets = [
        _bucket("current", key="profile.city", status="current"),
        _bucket("historical", key="profile.city", status="historical"),
    ]

    history = server._filter_z_fact_candidates(
        buckets,
        query="以前的具体地址是什么",
        intent="fact",
    )
    relation = server._filter_z_fact_candidates(
        buckets,
        query="我们之间发生了什么",
        intent="relation",
    )

    assert [bucket["id"] for bucket in history] == ["current", "historical"]
    assert [bucket["id"] for bucket in relation] == ["current", "historical"]


def test_disabled_or_empty_registry_is_fail_open(monkeypatch):
    buckets = [_bucket("historical", key="profile.city", status="historical")]
    monkeypatch.setattr(
        server,
        "config",
        {**server.config, "fact_slots": {"enabled": False, "registry": {}}},
    )

    assert server._filter_z_fact_candidates(
        buckets,
        query="具体地址是多少",
        intent="fact",
    ) == buckets


def test_historical_wrong_registry_context_is_fail_open(monkeypatch):
    candidate = _bucket("historical", key="profile.city", status="historical")
    candidate["metadata"]["domain"] = ["工作"]
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "fact_slots": {
                "enabled": True,
                "registry": {
                    "profile.city": {
                        "aliases": ["城市"],
                        "domains": ["生活"],
                    },
                },
            },
        },
    )

    assert server._filter_z_fact_candidates(
        [candidate],
        query="具体地址是多少",
        intent="fact",
    ) == [candidate]


def test_pair_validation_rechecks_registry_context_and_existing_slot(monkeypatch):
    _configure(monkeypatch)
    current = _bucket("current")
    historical = _bucket("historical")
    assert server._z_pair_validation_error(
        current,
        historical,
        "profile.city",
    ) == ""

    protected = _bucket("protected", protected=True)
    assert "outside" in server._z_pair_validation_error(
        current,
        protected,
        "profile.city",
    )

    wrong_slot = _bucket("wrong", key="profile.job")
    assert "different fact_key" in server._z_pair_validation_error(
        current,
        wrong_slot,
        "profile.city",
    )

    assert "no longer registered" in server._z_pair_validation_error(
        current,
        historical,
        "profile.job",
    )
