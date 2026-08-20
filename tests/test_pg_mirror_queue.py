from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

from bucket_manager import BucketManager, bucket_revision_hash
from consolidation_engine import ConsolidationEngine
from embedding_engine import EmbeddingEngine
import pg_mirror_queue as mirror_module
from pg_mirror_queue import PgMirrorQueue
from timeline_axis import run_timeline_sweep


def _mirror_config(test_config: dict) -> dict:
    config = copy.deepcopy(test_config)
    config["pg_mirror"] = {
        "enabled": True,
        "database": "ombre_mirror_test",
        "psql_path": "/definitely/missing/psql",
        "retry_seconds": 0.01,
        "batch_size": 10,
        "dimension": 1024,
    }
    return config


def _pending(manager: BucketManager):
    return manager.pg_mirror_queue.pending(limit=20)


@pytest.mark.asyncio
async def test_chat_create_enters_the_unified_after_write_queue(test_config):
    manager = BucketManager(_mirror_config(test_config))

    bucket_id = await manager.create(
        "朝灯刚发来的聊天记忆",
        name="聊天写点",
        actor="chat:hold",
    )

    rows = _pending(manager)
    assert [(row.bucket_id, row.action, row.source) for row in rows] == [
        (bucket_id, "dirty", "bucket_manager:after-write")
    ]


@pytest.mark.asyncio
async def test_chat_hold_route_reaches_the_unified_after_write_queue(
    test_config,
    monkeypatch,
):
    import server

    manager = BucketManager(_mirror_config(test_config))

    async def no_background():
        return None

    async def no_embedding(*_args, **_kwargs):
        return None

    async def analyze(_content):
        return {
            "domain": ["测试"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
            "suggested_name": "聊天同步写桶",
            "entities": [],
        }

    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server.decay_engine, "ensure_started", no_background)
    monkeypatch.setattr(server, "_maybe_start_backfill", lambda: None)
    monkeypatch.setattr(server, "dehydrator", SimpleNamespace(analyze=analyze))
    monkeypatch.setattr(
        server,
        "embedding_engine",
        SimpleNamespace(generate_and_store=no_embedding),
    )

    result = await server.hold(
        "聊天同步写桶必须经过统一写入口。",
        pinned=True,
        domain="测试",
    )

    bucket_id = result.split("→", 1)[1].split(" ", 1)[0]
    assert [row.bucket_id for row in _pending(manager)] == [bucket_id]


@pytest.mark.asyncio
async def test_dehydration_frontmatter_write_uses_the_same_queue(test_config):
    manager = BucketManager(_mirror_config(test_config))
    content = "一段足够稳定、可生成脱水摘要的正文"
    bucket_id = await manager.create(content, name="脱水写点", actor="chat:hold")
    before = _pending(manager)[0]

    assert await manager.cache_recall_dehydration(
        bucket_id,
        expected_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        summary="这是写入 frontmatter 的稳定脱水摘要。",
    )

    after = _pending(manager)[0]
    assert after.bucket_id == bucket_id
    assert after.source == "bucket_manager:after-write"
    assert after.revision == before.revision + 1


@pytest.mark.asyncio
async def test_night_consolidation_report_uses_the_same_queue(test_config):
    config = _mirror_config(test_config)
    config["metabolism"] = {"mode": "apply"}
    manager = BucketManager(config)
    engine = ConsolidationEngine(config, manager, embedding_engine=None)

    report_id = await engine._write_report(
        dups=[],
        stale=[{
            "id": "old-bucket",
            "name": "旧桶",
            "days_inactive": 30.0,
            "importance": 2,
            "domain": ["工程"],
        }],
        auto_digested=0,
    )

    assert report_id
    rows = _pending(manager)
    assert [(row.bucket_id, row.source) for row in rows] == [
        (report_id, "bucket_manager:after-write")
    ]


@pytest.mark.asyncio
async def test_manual_timeline_sweep_uses_the_same_queue(test_config):
    manager = BucketManager(_mirror_config(test_config))
    bucket_id = await manager.create(
        "工程第一阶段",
        name="手工时间线",
        actor="chat:hold",
    )
    before = _pending(manager)[0]

    report = await run_timeline_sweep(
        manager,
        reviewed_threads_by_bucket={bucket_id: "工程演进"},
        apply=True,
        actor="operator:timeline-sweep",
        revision_hash_provider=bucket_revision_hash,
    )

    assert report.updated_count == 1
    after = _pending(manager)[0]
    assert after.source == "bucket_manager:after-write"
    assert after.revision == before.revision + 1


def test_manual_write_memory_script_uses_the_same_queue(test_config, monkeypatch):
    import write_memory

    config = _mirror_config(test_config)
    monkeypatch.setattr(write_memory, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", config["buckets_dir"])

    bucket_id = write_memory.write_memory(
        "手工维护写点",
        "手工脚本也必须经过统一写入口。",
        ["测试"],
        ["manual"],
    )

    reopened = PgMirrorQueue(config)
    assert [row.bucket_id for row in reopened.pending(limit=20)] == [bucket_id]


def test_successful_incremental_drain_writes_all_segments_and_acks(
    test_config,
    monkeypatch,
):
    config = _mirror_config(test_config)
    engine = EmbeddingEngine(config)
    engine._store_embedding(
        "two-segment-bucket",
        [[0.25] * 1024, [0.5] * 1024],
    )
    scripts: list[str] = []

    def fake_psql(_command, **kwargs):
        scripts.append(kwargs["input"])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mirror_module.subprocess, "run", fake_psql)

    result = engine.pg_mirror_queue.drain_once()

    assert result == {
        "status": "ok",
        "processed": 1,
        "failed": 0,
        "remaining": 0,
    }
    assert len(scripts) == 1
    assert "DELETE FROM ombre_vectors" in scripts[0]
    assert "INSERT INTO ombre_vectors" in scripts[0]
    assert scripts[0].count("::halfvec(1024)") == 2


@pytest.mark.asyncio
async def test_pg_unavailable_retains_queue_and_sqlite_vector_recall(test_config):
    config = _mirror_config(test_config)
    engine = EmbeddingEngine(config)
    engine.enabled = True
    vector = [0.25] * 1024
    engine._store_embedding("sqlite-stays-live", [vector])

    result = engine.pg_mirror_queue.drain_once()

    assert result["failed"] == 1
    assert result["remaining"] == 1
    assert engine.pg_mirror_queue.pending(limit=1)[0].action == "upsert"

    async def fake_query_embedding(_query):
        return vector, "ok"

    engine._generate_embedding_with_status = fake_query_embedding
    hits, status = await engine.search_similar_with_status("仍走 SQLite", top_k=1)
    assert status == "ok"
    assert hits[0][0] == "sqlite-stays-live"
    assert hits[0][1] == pytest.approx(1.0)


def test_queue_is_durable_and_coalesces_newer_writes(test_config):
    config = _mirror_config(test_config)
    first = PgMirrorQueue(config)
    assert first.enqueue("same-bucket", action="dirty", source="chat:hold")
    assert first.enqueue("same-bucket", action="upsert", source="embedding:store")

    reopened = PgMirrorQueue(config)
    rows = reopened.pending(limit=10)
    assert len(rows) == 1
    assert rows[0].bucket_id == "same-bucket"
    assert rows[0].action == "upsert"
    assert rows[0].source == "embedding:store"
    assert rows[0].revision == 2
