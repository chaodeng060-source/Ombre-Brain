from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from curated_writer import (
    CuratedWriteCoordinator,
    CuratedWriteIntegrityError,
    IdempotencyConflictError,
)


class FakeEmbedding:
    def __init__(
        self,
        *,
        succeeds: bool = True,
        outcomes: list[bool] | None = None,
        delay: float = 0,
    ):
        self.outcomes = list(outcomes) if outcomes is not None else None
        self.succeeds = succeeds
        self.delay = delay
        self.calls: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.stored: set[str] = set()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        self.calls.append((bucket_id, content))
        if self.delay:
            await asyncio.sleep(self.delay)
        outcome = self.outcomes.pop(0) if self.outcomes is not None else self.succeeds
        if outcome:
            self.stored.add(bucket_id)
        return outcome

    def delete_embedding(self, bucket_id: str) -> None:
        self.deleted.append(bucket_id)
        self.stored.discard(bucket_id)

    async def get_embedding(self, bucket_id: str):
        return [1.0] if bucket_id in self.stored else None


@pytest.mark.asyncio
async def test_required_vector_failure_stays_archived_and_replays(
    test_config, bucket_mgr
):
    embedding = FakeEmbedding(outcomes=[False, True])
    writer = CuratedWriteCoordinator(bucket_mgr, embedding)

    first = await writer.write(
        idempotency_key="session:one/chunk:1",
        content="必须完整向量化的记忆",
        vector_policy="required",
        bucket_options={"domain": ["测试"], "name": "严格写入"},
    )
    second = await writer.write(
        idempotency_key="session:one/chunk:1",
        content="必须完整向量化的记忆",
        vector_policy="required",
        bucket_options={"domain": ["测试"], "name": "严格写入"},
    )

    assert first.success is False
    assert first.status == "retryable"
    assert first.recall_state == "quarantined_vector"
    assert second.success is True
    assert second.status == "completed"
    assert second.bucket_id == first.bucket_id
    assert len(embedding.calls) == 2
    assert [b["id"] for b in await bucket_mgr.list_all(include_archive=False)] == [
        second.bucket_id
    ]


@pytest.mark.asyncio
async def test_fts_only_is_explicit_and_does_not_call_embedding(
    test_config, bucket_mgr
):
    embedding = FakeEmbedding(succeeds=False)
    writer = CuratedWriteCoordinator(bucket_mgr, embedding)

    result = await writer.write(
        idempotency_key="fts:one",
        content="只允许全文检索的记忆",
        vector_policy="fts_only",
        bucket_options={"domain": ["测试"]},
    )

    assert result.success is True
    assert result.recall_state == "ready_fts"
    assert embedding.calls == []
    visible = await bucket_mgr.list_all(include_archive=False)
    assert [bucket["id"] for bucket in visible] == [result.bucket_id]
    meta = visible[0]["metadata"]
    assert meta["vector_policy"] == "fts_only"
    assert meta["lmc5_recall_state"] == "ready_fts"


@pytest.mark.asyncio
async def test_required_success_promotes_only_after_vector(
    test_config, bucket_mgr
):
    embedding = FakeEmbedding(succeeds=True)
    writer = CuratedWriteCoordinator(bucket_mgr, embedding)

    result = await writer.write(
        idempotency_key="vector:one",
        content="向量与正文都完整",
        vector_policy="required",
        bucket_options={"domain": ["测试"], "bucket_type": "permanent"},
    )

    assert result.success is True
    assert result.recall_state == "ready_vector"
    bucket = await bucket_mgr.get(result.bucket_id)
    assert bucket["metadata"]["type"] == "permanent"
    assert bucket["metadata"]["lmc5_recall_state"] == "ready_vector"
    assert not Path(bucket["path"]).is_relative_to(Path(bucket_mgr.archive_dir))


@pytest.mark.asyncio
async def test_body_failure_is_retryable_and_replay_safe(test_config, bucket_mgr):
    embedding = FakeEmbedding()
    writer = CuratedWriteCoordinator(bucket_mgr, embedding)
    calls = 0
    original_create = bucket_mgr.create

    async def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected body failure")
        return await original_create(*args, **kwargs)

    bucket_mgr.create = fail_once
    first = await writer.write(
        idempotency_key="body:failure",
        content="写不进去",
        vector_policy="required",
    )
    second = await writer.write(
        idempotency_key="body:failure",
        content="写不进去",
        vector_policy="required",
    )

    assert first.status == "retryable"
    assert first.error_code == "body_write_failed"
    assert first.bucket_id is None
    assert second.success is True
    assert calls == 2
    assert len(embedding.calls) == 1


@pytest.mark.asyncio
async def test_crash_after_body_create_recovers_marker_without_duplicate(
    test_config, bucket_mgr
):
    embedding = FakeEmbedding()
    writer = CuratedWriteCoordinator(bucket_mgr, embedding)
    original_create = bucket_mgr.create
    calls = 0

    async def create_then_crash(*args, **kwargs):
        nonlocal calls
        calls += 1
        bucket_id = await original_create(*args, **kwargs)
        if calls == 1:
            raise OSError("injected crash after durable body")
        return bucket_id

    bucket_mgr.create = create_then_crash
    first = await writer.write(
        idempotency_key="crash:body",
        content="正文已落盘但回执还没来得及写",
        vector_policy="required",
    )
    second = await writer.write(
        idempotency_key="crash:body",
        content="正文已落盘但回执还没来得及写",
        vector_policy="required",
    )

    assert first.status == "retryable"
    assert second.success is True
    assert calls == 1
    assert len(await bucket_mgr.list_all(include_archive=True)) == 1


@pytest.mark.asyncio
async def test_crash_after_promotion_recovers_completed_receipt(
    test_config, bucket_mgr
):
    embedding = FakeEmbedding()
    writer = CuratedWriteCoordinator(bucket_mgr, embedding)
    original_finish = writer._finish
    crashed = False

    def finish_then_crash(key, result):
        nonlocal crashed
        if result.status == "completed" and not crashed:
            crashed = True
            raise OSError("injected crash before terminal receipt")
        return original_finish(key, result)

    writer._finish = finish_then_crash
    with pytest.raises(OSError, match="terminal receipt"):
        await writer.write(
            idempotency_key="crash:promotion",
            content="已经提升到主召回但回执尚未完成",
            vector_policy="required",
        )

    replay = await writer.write(
        idempotency_key="crash:promotion",
        content="已经提升到主召回但回执尚未完成",
        vector_policy="required",
    )
    assert replay.success is True
    assert len(embedding.calls) == 1
    assert len(await bucket_mgr.list_all(include_archive=False)) == 1


@pytest.mark.asyncio
async def test_same_key_different_payload_fails_closed(test_config, bucket_mgr):
    writer = CuratedWriteCoordinator(bucket_mgr, FakeEmbedding())
    await writer.write(
        idempotency_key="stable:key",
        content="第一份正文",
        vector_policy="fts_only",
    )

    with pytest.raises(IdempotencyConflictError):
        await writer.write(
            idempotency_key="stable:key",
            content="被偷换的正文",
            vector_policy="fts_only",
        )


@pytest.mark.asyncio
async def test_completed_required_receipt_fails_closed_if_vector_disappears(
    test_config, bucket_mgr
):
    embedding = FakeEmbedding()
    writer = CuratedWriteCoordinator(bucket_mgr, embedding)
    result = await writer.write(
        idempotency_key="vector:lost",
        content="完整回执不能掩盖后来消失的向量",
        vector_policy="required",
    )
    embedding.delete_embedding(result.bucket_id)

    with pytest.raises(CuratedWriteIntegrityError, match="lost its required vector"):
        await writer.write(
            idempotency_key="vector:lost",
            content="完整回执不能掩盖后来消失的向量",
            vector_policy="required",
        )


@pytest.mark.asyncio
async def test_concurrent_same_key_creates_one_bucket_and_one_vector(
    test_config, bucket_mgr
):
    embedding = FakeEmbedding(succeeds=True, delay=0.02)
    writer = CuratedWriteCoordinator(bucket_mgr, embedding)

    async def write_once():
        return await writer.write(
            idempotency_key="concurrent:key",
            content="并发重放仍只提交一次",
            vector_policy="required",
            bucket_options={"domain": ["测试"]},
        )

    first, second = await asyncio.gather(write_once(), write_once())

    assert first == second
    assert first.success is True
    assert len(embedding.calls) == 1
    visible = await bucket_mgr.list_all(include_archive=False)
    assert [bucket["id"] for bucket in visible] == [first.bucket_id]
