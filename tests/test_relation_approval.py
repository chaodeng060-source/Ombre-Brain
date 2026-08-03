from __future__ import annotations

from pathlib import Path

import pytest

from bucket_manager import BucketManager
from relation_approval import (
    RelationApprovalStateError,
    RelationApprovalTransaction,
)
from review_queue import ReviewQueue, make_relation_entry


def _manager(tmp_path: Path) -> BucketManager:
    buckets_dir = tmp_path / "buckets"
    for directory in ("permanent", "dynamic", "archive", "feel", "涩涩"):
        (buckets_dir / directory).mkdir(parents=True, exist_ok=True)
    return BucketManager({
        "buckets_dir": str(buckets_dir),
        "audit": {"enabled": False},
        "matching": {"fuzzy_threshold": 50, "max_results": 10},
        "wikilink": {"enabled": False},
        "scoring_weights": {},
    })


async def _candidate(tmp_path: Path, rel_type: str = "causes"):
    manager = _manager(tmp_path)
    source_id = await manager.create(
        content="源事件",
        domain=["工作"],
        name="关系源",
    )
    target_id = await manager.create(
        content="目标事件",
        domain=["工作"],
        name="关系目标",
    )
    queue = ReviewQueue(
        Path(manager.base_dir) / "review_queue.jsonl",
        maintenance_root=manager.base_dir,
    )
    entry = make_relation_entry(
        source_id,
        target_id,
        rel_type,
        "具名审批测试",
        source_name="关系源",
        target_name="关系目标",
    )
    assert queue.enqueue(entry) is True
    transaction = RelationApprovalTransaction(
        manager.base_dir,
        manager,
        queue,
    )
    return manager, queue, transaction, entry


@pytest.mark.asyncio
async def test_named_dangerous_relation_approval_is_atomic_and_idempotent(
    tmp_path,
):
    manager, queue, transaction, entry = await _candidate(tmp_path)

    first = transaction.apply(
        entry["key"],
        reviewer="朝灯",
        verdict_note="确认这条因果边",
    )
    source = await manager.get(entry["source_id"])
    durable = queue.get(entry["key"])

    assert first["changed"] is True
    assert first["queue_changed"] is True
    assert source["metadata"]["relations"] == [{
        "type": "causes",
        "target": entry["target_id"],
        "note": "具名审批测试",
    }]
    assert durable["status"] == "applied"
    assert durable["reviewer"] == "朝灯"
    assert durable["verdict_note"] == "确认这条因果边"

    replay = transaction.apply(
        entry["key"],
        reviewer="朝灯",
        verdict_note="重复请求",
    )
    source = await manager.get(entry["source_id"])
    assert replay["changed"] is False
    assert replay["queue_changed"] is False
    assert len(source["metadata"]["relations"]) == 1


@pytest.mark.asyncio
async def test_relation_approval_failure_rolls_back_and_keeps_pending(
    tmp_path,
    monkeypatch,
):
    manager, queue, transaction, entry = await _candidate(tmp_path)

    def fail_queue(*_args, **_kwargs):
        raise OSError("injected queue failure")

    monkeypatch.setattr(queue, "apply_relation", fail_queue)
    with pytest.raises(OSError, match="injected queue failure"):
        transaction.apply(entry["key"], reviewer="朝灯")

    source = await manager.get(entry["source_id"])
    assert source["metadata"].get("relations", []) == []
    assert queue.get(entry["key"])["status"] == "pending"


@pytest.mark.asyncio
async def test_relation_recovery_rolls_back_interrupted_pending_write(
    tmp_path,
    monkeypatch,
):
    manager, queue, transaction, entry = await _candidate(tmp_path)

    class SimulatedCrash(BaseException):
        pass

    def crash_after_bucket_write(*_args, **_kwargs):
        raise SimulatedCrash()

    monkeypatch.setattr(queue, "apply_relation", crash_after_bucket_write)
    with pytest.raises(SimulatedCrash):
        transaction.apply(entry["key"], reviewer="朝灯")
    landed = await manager.get(entry["source_id"])
    assert len(landed["metadata"].get("relations", [])) == 1
    assert queue.get(entry["key"])["status"] == "pending"

    assert transaction.recover() == [entry["key"]]
    recovered = await manager.get(entry["source_id"])
    assert recovered["metadata"].get("relations", []) == []
    assert queue.get(entry["key"])["status"] == "pending"


@pytest.mark.asyncio
async def test_existing_exact_edge_is_a_queue_only_idempotent_commit(tmp_path):
    manager, queue, transaction, entry = await _candidate(tmp_path)
    assert await manager.add_relation(
        entry["source_id"],
        entry["target_id"],
        entry["rel_type"],
        entry["note"],
    ) is True

    result = transaction.apply(entry["key"], reviewer="朝灯")
    source = await manager.get(entry["source_id"])

    assert result["changed"] is False
    assert result["queue_changed"] is True
    assert queue.get(entry["key"])["status"] == "applied"
    assert len(source["metadata"]["relations"]) == 1


@pytest.mark.asyncio
async def test_safe_relation_cannot_use_dangerous_approval_path(tmp_path):
    _, queue, transaction, entry = await _candidate(tmp_path, rel_type="kin")

    with pytest.raises(
        RelationApprovalStateError,
        match="only dangerous relation types",
    ):
        transaction.apply(entry["key"], reviewer="朝灯")

    assert queue.get(entry["key"])["status"] == "pending"


@pytest.mark.asyncio
async def test_relation_apply_api_requires_named_reviewer():
    import server

    response = await server._apply_review_queue_relation({"key": "rel|candidate"})

    assert response.status_code == 400
    assert b"reviewer required" in response.body
