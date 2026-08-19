"""Z-axis integration tests for canonical fact slots and recall gating."""

import hashlib
import json
from pathlib import Path

import pytest

from bucket_manager import BucketManager
import server

_ORIGINAL_GET_REVIEW_QUEUE = server._get_review_queue


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

    # 明确命中已登记槽（「城市」）：只压该槽的 historical；contested / 未注册 / protected 都留。
    kept = server._filter_z_fact_candidates(
        buckets,
        query="现在城市是哪个",
        intent="fact",
    )

    assert [bucket["id"] for bucket in kept] == [
        "current",
        "contested",
        "unknown",
        "protected",
    ]

    # 2026-08-19 复核 P1-3：没有命中任何已登记槽的 fact 查询（「具体地址是多少」对只有
    # 「城市」的注册表是 neutral / fact_keys=()）必须 fail-open——一个 historical 都不删。
    # 旧断言把空 fact_keys 当「所有槽」过滤，违反 docs/Z_AXIS_FACT_SLOTS.md「只有明确命中
    # fact_key 才启用」与「非 Z 查询 top5 不变」。
    untouched = server._filter_z_fact_candidates(
        buckets,
        query="具体地址是多少",
        intent="fact",
    )
    assert [bucket["id"] for bucket in untouched] == [bucket["id"] for bucket in buckets]


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


def test_natural_registered_slot_question_activates_currentness(monkeypatch):
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "fact_slots": {
                "enabled": True,
                "registry": {
                    "preference.ui.primary_color": {"aliases": ["主色"]},
                    "preference.ui.font_style": {"aliases": ["字体倾向"]},
                },
            },
        },
    )
    buckets = [
        _bucket(
            "color-current",
            key="preference.ui.primary_color",
            status="current",
        ),
        _bucket(
            "color-old",
            key="preference.ui.primary_color",
            status="historical",
        ),
        _bucket(
            "font-old",
            key="preference.ui.font_style",
            status="historical",
        ),
        _bucket(
            "protected-old",
            key="preference.ui.primary_color",
            status="historical",
            protected=True,
        ),
    ]

    policy = server._resolve_recall_policy(
        "现在主色是什么",
        base_recall_limit=20,
        requested_relation_depth=1,
    )
    kept = server._filter_z_fact_candidates(
        buckets,
        query="现在主色是什么",
        intent=policy["intent"],
    )

    assert policy["intent"] == "fact"
    assert [bucket["id"] for bucket in kept] == [
        "color-current",
        "font-old",
        "protected-old",
    ]


def test_registered_slot_history_question_keeps_history(monkeypatch):
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "fact_slots": {
                "enabled": True,
                "registry": {
                    "preference.ui.primary_color": {"aliases": ["主色"]},
                },
            },
        },
    )
    buckets = [
        _bucket(
            "color-current",
            key="preference.ui.primary_color",
            status="current",
        ),
        _bucket(
            "color-old",
            key="preference.ui.primary_color",
            status="historical",
        ),
    ]
    policy = server._resolve_recall_policy(
        "以前的主色是什么",
        base_recall_limit=20,
        requested_relation_depth=1,
    )

    assert policy["intent"] == "fact"
    assert server._filter_z_fact_candidates(
        buckets,
        query="以前的主色是什么",
        intent=policy["intent"],
    ) == buckets


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


def _response_json(response):
    return json.loads(response.body.decode("utf-8"))


def _bucket_hash(manager, bucket_id):
    path = Path(manager._find_bucket_file(bucket_id))
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _configure_real_z_pair(monkeypatch, tmp_path):
    buckets_dir = str(tmp_path / "buckets")
    for directory in ("permanent", "dynamic", "archive", "feel", "涩涩"):
        Path(buckets_dir, directory).mkdir(parents=True, exist_ok=True)
    test_config = {
        "buckets_dir": buckets_dir,
        "audit": {"enabled": False},
        "matching": {"fuzzy_threshold": 50, "max_results": 10},
        "wikilink": {"enabled": False},
        "scoring_weights": {},
    }
    config = {
        **test_config,
        "fact_slots": {
            "enabled": True,
            "registry": {
                "profile.city": {"aliases": ["城市"]},
            },
        },
    }
    manager = BucketManager(config)
    historical_id = await manager.create(
        content="城市: 北京",
        domain=["生活"],
        name="Z验收旧城市",
    )
    current_id = await manager.create(
        content="城市: 杭州",
        domain=["生活"],
        name="Z验收新城市",
    )
    monkeypatch.setattr(server, "config", {**server.config, **config})
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "_review_queue", None)
    monkeypatch.setattr(server, "_get_review_queue", _ORIGINAL_GET_REVIEW_QUEUE)
    monkeypatch.setattr(server, "_z_lifecycle_transaction", None)
    body = {
        "current_bucket_id": current_id,
        "historical_bucket_id": historical_id,
        "fact_key": "profile.city",
        "reason": "integration_acceptance",
        "source": "test",
    }
    return manager, current_id, historical_id, body, test_config


@pytest.mark.asyncio
async def test_z_candidate_defaults_to_dry_run_without_any_write(
    monkeypatch,
    tmp_path,
):
    manager, current_id, historical_id, body, test_config = await _configure_real_z_pair(
        monkeypatch,
        tmp_path,
    )
    before = {
        current_id: _bucket_hash(manager, current_id),
        historical_id: _bucket_hash(manager, historical_id),
    }

    response = await server._submit_review_queue_candidate(body)
    payload = _response_json(response)

    assert response.status_code == 200
    assert payload["mode"] == "dry-run"
    assert payload["status"] == "preview"
    assert payload["queue_mutated"] is False
    assert payload["memory_mutated"] is False
    assert not Path(test_config["buckets_dir"], "review_queue.jsonl").exists()
    assert _bucket_hash(manager, current_id) == before[current_id]
    assert _bucket_hash(manager, historical_id) == before[historical_id]


@pytest.mark.asyncio
async def test_z_apply_only_enqueues_pending_and_is_idempotent(
    monkeypatch,
    tmp_path,
):
    manager, current_id, historical_id, body, _ = await _configure_real_z_pair(
        monkeypatch,
        tmp_path,
    )
    before = {
        current_id: _bucket_hash(manager, current_id),
        historical_id: _bucket_hash(manager, historical_id),
    }

    first = _response_json(
        await server._submit_review_queue_candidate({**body, "mode": "apply"})
    )
    second = _response_json(
        await server._submit_review_queue_candidate({**body, "mode": "apply"})
    )
    current = await manager.get(current_id)
    historical = await manager.get(historical_id)

    assert first["status"] == "pending" and first["added"] is True
    assert second["status"] == "pending" and second["added"] is False
    assert first["memory_mutated"] is False
    assert len(server._get_review_queue().list_pending("z_conflict")) == 1
    assert "fact_status" not in current["metadata"]
    assert "fact_status" not in historical["metadata"]
    assert _bucket_hash(manager, current_id) == before[current_id]
    assert _bucket_hash(manager, historical_id) == before[historical_id]


@pytest.mark.asyncio
async def test_z_human_approval_atomically_marks_old_fact_historical(
    monkeypatch,
    tmp_path,
):
    manager, current_id, historical_id, body, _ = await _configure_real_z_pair(
        monkeypatch,
        tmp_path,
    )
    queued = _response_json(
        await server._submit_review_queue_candidate({**body, "mode": "apply"})
    )

    response = await server._apply_review_queue_lifecycle({
        "key": queued["key"],
        "reviewer": "human-reviewer",
        "verdict_note": "确认杭州为当前事实",
    })
    payload = _response_json(response)
    current = await manager.get(current_id)
    historical = await manager.get(historical_id)
    durable = server._get_review_queue().get(queued["key"])

    assert response.status_code == 200
    assert payload["status"] == "applied"
    assert payload["memory_mutated"] is True
    assert current["metadata"]["fact_key"] == "profile.city"
    assert current["metadata"]["fact_status"] == "current"
    assert current["metadata"]["supersedes_bucket_ids"] == [historical_id]
    assert historical["metadata"]["fact_key"] == "profile.city"
    assert historical["metadata"]["fact_status"] == "historical"
    assert historical["metadata"]["superseded_by_bucket_id"] == current_id
    assert durable["status"] == "applied"
    assert durable["reviewer"] == "human-reviewer"
    assert durable["verdict_note"] == "确认杭州为当前事实"

    replay = _response_json(await server._apply_review_queue_lifecycle({
        "key": queued["key"],
        "reviewer": "human-reviewer",
        "verdict_note": "确认杭州为当前事实",
    }))
    assert replay["changed"] is False


@pytest.mark.asyncio
async def test_z_approval_failure_restores_both_buckets_and_keeps_pending(
    monkeypatch,
    tmp_path,
):
    manager, current_id, historical_id, body, _ = await _configure_real_z_pair(
        monkeypatch,
        tmp_path,
    )
    queued = _response_json(
        await server._submit_review_queue_candidate({**body, "mode": "apply"})
    )
    before = {
        current_id: _bucket_hash(manager, current_id),
        historical_id: _bucket_hash(manager, historical_id),
    }
    transaction = server._get_z_lifecycle_transaction()
    real_write = transaction._write_bucket
    calls = 0

    def fail_second_bucket(path, text):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second bucket failure")
        real_write(path, text)

    monkeypatch.setattr(transaction, "_write_bucket", fail_second_bucket)
    response = await server._apply_review_queue_lifecycle({
        "key": queued["key"],
        "reviewer": "human-reviewer",
        "verdict_note": "failure injection",
    })

    assert response.status_code == 503
    assert _bucket_hash(manager, current_id) == before[current_id]
    assert _bucket_hash(manager, historical_id) == before[historical_id]
    assert server._get_review_queue().get(queued["key"])["status"] == "pending"


@pytest.mark.asyncio
async def test_z_startup_recovery_rolls_back_an_interrupted_pair(
    monkeypatch,
    tmp_path,
):
    manager, current_id, historical_id, body, _ = await _configure_real_z_pair(
        monkeypatch,
        tmp_path,
    )
    queued = _response_json(
        await server._submit_review_queue_candidate({**body, "mode": "apply"})
    )
    before = {
        current_id: _bucket_hash(manager, current_id),
        historical_id: _bucket_hash(manager, historical_id),
    }
    transaction = server._get_z_lifecycle_transaction()
    real_write = transaction._write_bucket
    calls = 0

    def crash_on_second_bucket(path, text):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated process death")
        real_write(path, text)

    with monkeypatch.context() as crash:
        crash.setattr(transaction, "_write_bucket", crash_on_second_bucket)
        with pytest.raises(KeyboardInterrupt):
            transaction.apply(
                queued["key"],
                reviewer="human-reviewer",
                verdict_note="crash recovery injection",
                validate_pair=server._z_pair_validation_error,
            )

    assert _bucket_hash(manager, current_id) != before[current_id]
    assert transaction.recover() == [queued["key"]]
    assert _bucket_hash(manager, current_id) == before[current_id]
    assert _bucket_hash(manager, historical_id) == before[historical_id]
    assert server._get_review_queue().get(queued["key"])["status"] == "pending"
