from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sqlite3
from pathlib import Path
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


def _write_bucket(test_config: dict, rel_path: str, bucket_id: str, body: str):
    path = Path(test_config["buckets_dir"]) / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: {bucket_id}\nname: contract fixture\n---\n{body}",
        encoding="utf-8",
    )
    return path


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
    body = "正文第一行\n含 ' 单引号的第二行"
    body_path = _write_bucket(
        config,
        "dynamic/测试/not-derived-from-id.md",
        "two-segment-bucket",
        body,
    )
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
    assert scripts[0].count("BEGIN;") == 1
    assert scripts[0].count("COMMIT;") == 1
    assert "DELETE FROM ombre_vectors" in scripts[0]
    assert "DELETE FROM ombre_bodies" in scripts[0]
    assert "INSERT INTO ombre_vectors" in scripts[0]
    assert "INSERT INTO ombre_bodies" in scripts[0]
    assert scripts[0].count("::halfvec(1024)") == 2
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() in scripts[0]
    assert str(body_path.relative_to(Path(config["buckets_dir"]))) in scripts[0]
    assert "正文第一行\n含 '' 单引号的第二行" in scripts[0]


def test_body_insert_failure_keeps_both_tables_in_one_failed_transaction(
    test_config,
    monkeypatch,
):
    config = _mirror_config(test_config)
    _write_bucket(config, "dynamic/测试/atomic.md", "atomic-bucket", "原子正文")
    engine = EmbeddingEngine(config)
    engine._store_embedding("atomic-bucket", [[0.25] * 1024])
    commands: list[list[str]] = []
    scripts: list[str] = []

    def fail_body_insert(command, **kwargs):
        commands.append(command)
        scripts.append(kwargs["input"])
        return SimpleNamespace(returncode=1, stderr="body constraint failed")

    monkeypatch.setattr(mirror_module.subprocess, "run", fail_body_insert)

    result = engine.pg_mirror_queue.drain_once()

    assert result == {
        "status": "deferred",
        "processed": 0,
        "failed": 1,
        "remaining": 1,
    }
    assert len(scripts) == 1
    assert scripts[0].count("BEGIN;") == 1
    assert scripts[0].count("COMMIT;") == 1
    assert "INSERT INTO ombre_vectors" in scripts[0]
    assert "INSERT INTO ombre_bodies" in scripts[0]
    assert "ON_ERROR_STOP=1" in commands[0]
    assert engine.pg_mirror_queue.pending(limit=1)[0].bucket_id == "atomic-bucket"


def test_dirty_bucket_without_embedding_still_mirrors_its_body(
    test_config,
    monkeypatch,
):
    config = _mirror_config(test_config)
    _write_bucket(
        config,
        "dynamic/测试/body-only.md",
        "body-only",
        "没有向量的桶也属于正文全量合同。",
    )
    queue = PgMirrorQueue(config)
    assert queue.enqueue("body-only", action="dirty", source="test")
    scripts: list[str] = []

    def fake_psql(_command, **kwargs):
        scripts.append(kwargs["input"])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mirror_module.subprocess, "run", fake_psql)

    result = queue.drain_once()

    assert result["processed"] == 1
    assert result["remaining"] == 0
    assert len(scripts) == 1
    assert "DELETE FROM ombre_vectors" in scripts[0]
    assert "INSERT INTO ombre_vectors" not in scripts[0]
    assert "INSERT INTO ombre_bodies" in scripts[0]


def test_delete_removes_vector_and_body_in_the_same_transaction(
    test_config,
    monkeypatch,
):
    queue = PgMirrorQueue(_mirror_config(test_config))
    assert queue.enqueue("deleted-bucket", action="delete", source="test")
    scripts: list[str] = []

    def fake_psql(_command, **kwargs):
        scripts.append(kwargs["input"])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mirror_module.subprocess, "run", fake_psql)

    result = queue.drain_once()

    assert result["processed"] == 1
    assert result["remaining"] == 0
    assert len(scripts) == 1
    assert scripts[0].count("BEGIN;") == 1
    assert scripts[0].count("COMMIT;") == 1
    assert "DELETE FROM ombre_vectors" in scripts[0]
    assert "DELETE FROM ombre_bodies" in scripts[0]
    assert "INSERT INTO" not in scripts[0]


def test_frontmatter_id_is_the_only_bucket_identity(
    test_config,
    monkeypatch,
):
    config = _mirror_config(test_config)
    _write_bucket(
        config,
        "dynamic/测试/arbitrary-name.md",
        "frontmatter-real",
        "只有 frontmatter id 能认出我。",
    )
    patrol = Path(config["buckets_dir"]) / ".lmc5" / "巡检_fake-report.md"
    patrol.parent.mkdir(parents=True)
    patrol.write_text("# 巡检报告，不是桶\n", encoding="utf-8")
    snapshot = (
        Path(config["buckets_dir"])
        / "dynamic"
        / "测试"
        / "current.original_snapshot-fake.md"
    )
    snapshot.write_text("---\nkind: transaction-snapshot\n---\n不是桶\n", encoding="utf-8")

    engine = EmbeddingEngine(config)
    for bucket_id in ("frontmatter-real", "fake-report", "snapshot-fake"):
        engine._store_embedding(bucket_id, [[0.25] * 1024])
    scripts: list[str] = []

    def fake_psql(_command, **kwargs):
        scripts.append(kwargs["input"])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mirror_module.subprocess, "run", fake_psql)

    result = engine.pg_mirror_queue.drain_once()

    assert result["processed"] == 1
    assert result["failed"] == 2
    assert result["remaining"] == 2
    assert len(scripts) == 1
    assert "frontmatter-real" in scripts[0]
    assert "fake-report" not in scripts[0]
    assert "snapshot-fake" not in scripts[0]


@pytest.mark.asyncio
async def test_pg_unavailable_retains_queue_and_sqlite_vector_recall(test_config):
    config = _mirror_config(test_config)
    _write_bucket(
        config,
        "dynamic/测试/sqlite-stays-live.md",
        "sqlite-stays-live",
        "PG 不可用时正文真源和 SQLite 召回都保持可用。",
    )
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


def test_poison_item_backs_off_without_starving_later_writes(
    test_config,
    monkeypatch,
):
    config = _mirror_config(test_config)
    config["pg_mirror"]["batch_size"] = 1
    for bucket_id in ("aaa-poison", "zzz-good-a", "zzz-good-b"):
        _write_bucket(
            config,
            f"dynamic/测试/{bucket_id}.md",
            bucket_id,
            f"{bucket_id} 正文",
        )
    queue = PgMirrorQueue(config)
    with sqlite3.connect(queue.embedding_db) as connection:
        connection.execute(
            "CREATE TABLE embeddings "
            "(bucket_id TEXT PRIMARY KEY, embedding TEXT, updated_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO embeddings VALUES (?, ?, ?)",
            [
                ("aaa-poison", json.dumps([0.1] * 7), "2026-08-20T22:00:00Z"),
                ("zzz-good-a", json.dumps([0.1] * 1024), "2026-08-20T22:00:00Z"),
                ("zzz-good-b", json.dumps([0.2] * 1024), "2026-08-20T22:00:00Z"),
            ],
        )
        connection.commit()
    for bucket_id in ("aaa-poison", "zzz-good-a", "zzz-good-b"):
        assert queue.enqueue(bucket_id, action="upsert", source="test")

    monkeypatch.setattr(
        mirror_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    first = queue.drain_once()
    second = queue.drain_once()
    third = queue.drain_once()

    assert first["failed"] == 1
    assert second["processed"] == 1
    assert third["processed"] == 1
    rows = queue.pending(limit=10)
    assert [(row.bucket_id, row.attempts) for row in rows] == [
        ("aaa-poison", 1)
    ]
    assert rows[0].next_retry_at > 0


@pytest.mark.asyncio
async def test_worker_recovers_after_one_queue_exception(test_config):
    queue = PgMirrorQueue(_mirror_config(test_config))
    queue.retry_seconds = 0.01
    calls = 0

    def flaky_drain():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("database is locked")
        return {"status": "ok", "processed": 0, "failed": 0, "remaining": 0}

    queue.drain_once = flaky_drain
    worker = mirror_module.PgMirrorWorker(queue)
    await worker.start()
    await asyncio.sleep(0.08)

    assert calls >= 2
    assert worker._task is not None
    assert not worker._task.done()
    await worker.stop()


@pytest.mark.asyncio
async def test_worker_stop_swallows_an_already_failed_task(test_config):
    queue = PgMirrorQueue(_mirror_config(test_config))
    worker = mirror_module.PgMirrorWorker(queue)

    async def failed_task():
        raise sqlite3.OperationalError("old worker failure")

    worker._task = asyncio.create_task(failed_task())
    await asyncio.sleep(0)

    await worker.stop()
    assert worker._task is None
