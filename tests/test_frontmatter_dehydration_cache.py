import hashlib

import pytest

import server
from bucket_manager import BucketManager, bucket_revision_hash
from dehydrator import (
    anchor_memory_relative_time_terms,
    recall_frontmatter_time_contract,
    sanitize_dehydration_sample_voice,
)
from recall_timing import (
    begin_recall_timing,
    finish_recall_timing,
    reset_recall_timing,
)


@pytest.mark.asyncio
async def test_derived_summary_write_preserves_body_activity_and_semantic_revision(
    test_config,
):
    manager = BucketManager(test_config)
    bucket_id = await manager.create(
        "朝灯的长记忆正文不会被派生摘要改写。" * 80,
        name="派生摘要",
        domain=["测试"],
    )
    before = await manager.get(bucket_id)
    before_revision = bucket_revision_hash(
        before["content"],
        before["metadata"],
    )
    body_hash = hashlib.sha256(before["content"].encode("utf-8")).hexdigest()

    assert await manager.cache_recall_dehydration(
        bucket_id,
        expected_content_hash=body_hash,
        summary='{"summary":"桶内持久化摘要"}',
        contract="ombre.recall-frontmatter/event-time/v1:2026-08-14",
    )

    after = await manager.get(bucket_id)
    assert after["content"] == before["content"]
    assert after["metadata"]["last_active"] == before["metadata"]["last_active"]
    assert after["metadata"]["dehydrated_content_hash"] == body_hash
    assert after["metadata"]["dehydrated_summary"] == '{"summary":"桶内持久化摘要"}'
    assert after["metadata"]["dehydrated_summary_contract"].endswith(":2026-08-14")
    assert bucket_revision_hash(after["content"], after["metadata"]) == before_revision


@pytest.mark.asyncio
async def test_derived_summary_write_rejects_changed_body(test_config):
    manager = BucketManager(test_config)
    bucket_id = await manager.create("真实正文", domain=["测试"])

    assert not await manager.cache_recall_dehydration(
        bucket_id,
        expected_content_hash=hashlib.sha256(b"stale body").hexdigest(),
        summary='{"summary":"不能写入"}',
    )
    current = await manager.get(bucket_id)
    assert "dehydrated_summary" not in current["metadata"]
    assert "dehydrated_content_hash" not in current["metadata"]


class _RecordingManager:
    def __init__(self):
        self.writes = []

    async def cache_recall_dehydration(self, bucket_id, **kwargs):
        self.writes.append((bucket_id, kwargs))
        return True


class _SourceAwareDehydrator:
    def __init__(self):
        self.calls = 0

    async def dehydrate_with_source(self, content, metadata, *, write_cache):
        self.calls += 1
        assert metadata is None
        assert write_cache is False
        return '{"summary":"新算出的原始摘要"}', "computed"

    @staticmethod
    def format_dehydration_summary(summary, metadata):
        return f"{metadata['name']}\n{summary}"


@pytest.mark.asyncio
async def test_frontmatter_hit_skips_dehydrator_and_trace_counts_sources(monkeypatch):
    manager = _RecordingManager()
    source = _SourceAwareDehydrator()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", source)
    monkeypatch.setattr(
        server,
        "config",
        {"dehydration": {"recall_frontmatter_cache_enabled": True}},
    )
    body = "需要模型脱水的长正文" * 100
    bucket = {
        "id": "bucket-a",
        "content": body,
        "metadata": {"id": "bucket-a", "name": "第一次"},
    }

    token = begin_recall_timing()
    try:
        first = await server._dehydrate_for_recall(
            body,
            {"name": "第一次"},
            bucket=bucket,
        )
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        bucket["metadata"].update(
            dehydrated_summary='{"summary":"新算出的原始摘要"}',
            dehydrated_content_hash=body_hash,
            name="第二次",
        )
        second = await server._dehydrate_for_recall(
            body,
            {"name": "第二次"},
            bucket=bucket,
        )
        receipt = finish_recall_timing(status="ok", partial=False)
    finally:
        reset_recall_timing(token)

    assert first.startswith("第一次\n")
    assert second.startswith("第二次\n")
    assert source.calls == 1
    assert manager.writes[0][0] == "bucket-a"
    assert manager.writes[0][1]["expected_content_hash"] == body_hash
    assert receipt["dehydration"] == {
        "computed": 1,
        "frontmatter_hits": 1,
    }


@pytest.mark.asyncio
async def test_content_change_invalidates_frontmatter_summary(monkeypatch):
    manager = _RecordingManager()
    source = _SourceAwareDehydrator()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", source)
    monkeypatch.setattr(
        server,
        "config",
        {"dehydration": {"recall_frontmatter_cache_enabled": True}},
    )
    old_body = "旧正文" * 100
    new_body = "新正文" * 100
    bucket = {
        "id": "bucket-b",
        "content": new_body,
        "metadata": {
            "id": "bucket-b",
            "name": "内容变化",
            "dehydrated_summary": "旧摘要不应复用",
            "dehydrated_content_hash": hashlib.sha256(
                old_body.encode("utf-8")
            ).hexdigest(),
        },
    }

    result = await server._dehydrate_for_recall(
        new_body,
        {"name": "内容变化"},
        bucket=bucket,
    )

    assert "新算出的原始摘要" in result
    assert source.calls == 1
    assert manager.writes[0][1]["expected_content_hash"] == hashlib.sha256(
        new_body.encode("utf-8")
    ).hexdigest()


def test_memory_relative_time_uses_event_day_and_preserves_quotes():
    content = '今天上午完成了核验，朝灯说「今天下午再看」，昨晚先做了备份。'
    metadata = {
        "event_at": "2026-08-14T23:30:00+08:00",
        "recorded_at": "2026-09-01T09:00:00+08:00",
    }

    anchored = anchor_memory_relative_time_terms(content, metadata)

    assert anchored.startswith("【记忆发生日：2026-08-14】")
    assert "2026-08-14 上午完成了核验" in anchored
    assert "2026-08-13 晚上先做了备份" in anchored
    assert "「今天下午再看」" in anchored
    assert recall_frontmatter_time_contract(content, metadata).endswith(":2026-08-14")


@pytest.mark.asyncio
async def test_relative_time_frontmatter_requires_dated_contract(monkeypatch):
    manager = _RecordingManager()
    source = _SourceAwareDehydrator()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", source)
    monkeypatch.setattr(
        server,
        "config",
        {"dehydration": {"recall_frontmatter_cache_enabled": True}},
    )
    body = "今天上午我核对了部署证据。" * 60
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    bucket = {
        "id": "bucket-relative",
        "content": body,
        "metadata": {
            "id": "bucket-relative",
            "name": "相对时间",
            "event_at": "2026-08-14T11:04:27",
            "dehydrated_summary": '{"summary":"5.21上午的错误摘要"}',
            "dehydrated_content_hash": body_hash,
        },
    }

    result = await server._dehydrate_for_recall(
        body,
        {"name": "相对时间"},
        bucket=bucket,
    )

    assert "新算出的原始摘要" in result
    assert source.calls == 1
    assert manager.writes[0][1]["contract"].endswith(":2026-08-14")


def test_e_authored_sample_voice_keeps_only_explicit_chaodeng_quotes():
    content = (
        '朝灯问「没推到GitHub吗居然」，'
        '我又反手说「GitHub 上没有副本」。'
        '她今天没骂我，只说了「居然」。'
    )
    summary = (
        '{"summary":"归属验证",'
        '"sample_voice":["没推到GitHub吗居然","GitHub 上没有副本","居然"]}'
    )

    cleaned = sanitize_dehydration_sample_voice(
        summary,
        content,
        {"e_authored_by": "claude"},
    )

    assert '"没推到GitHub吗居然"' in cleaned
    assert '"居然"' in cleaned
    assert '"GitHub 上没有副本"' not in cleaned
