from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

import server
from entity_store import EntityStore


def _seed() -> dict:
    return {
        "canonical_name": "朝灯",
        "type": "person",
        "aliases": ["老婆", "Rosita", "Vae"],
    }


def _bucket(
    bucket_id: str,
    content: str,
    *,
    world: str = "daily",
    domain: list[str] | None = None,
    created: str = "2026-08-02T00:00:00",
    bucket_type: str = "dynamic",
    pinned: bool = False,
    path: str = "",
) -> dict:
    return {
        "id": bucket_id,
        "content": content,
        "path": path,
        "metadata": {
            "name": bucket_id,
            "type": bucket_type,
            "world": world,
            "domain": domain or ["工程"],
            "created": created,
            "importance": 5,
            "valence": 0.5,
            "arousal": 0.3,
            "pinned": pinned,
            "tags": [],
        },
    }


class _NoopLoop:
    async def ensure_started(self):
        return None


class _Decay(_NoopLoop):
    @staticmethod
    def apply_retrieval_decay(score, metadata):
        return score


class _Dehydrator:
    async def dehydrate(self, content, metadata):
        return content


class _Embedding:
    async def search_similar(self, query, top_k=20):
        return []

    async def search_similar_with_status(self, query, top_k=20):
        return [], "ok"


class _Manager:
    def __init__(self, buckets, archive_dir: Path, *, canonical_keyword=False):
        self.buckets = {bucket["id"]: bucket for bucket in buckets}
        self.archive_dir = str(archive_dir)
        self.canonical_keyword = canonical_keyword
        self.queries: list[str] = []

    async def search(self, query, limit=20, **kwargs):
        self.queries.append(query)
        if self.canonical_keyword and query == "朝灯":
            return list(self.buckets.values())[:limit]
        return []

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)

    async def list_all(self, include_archive=False, **kwargs):
        return list(self.buckets.values())


def _configure_server(tmp_path, monkeypatch, buckets, *, canonical_keyword=False):
    buckets_dir = tmp_path / "vault"
    buckets_dir.mkdir()
    cfg = {
        **server.config,
        "buckets_dir": str(buckets_dir),
        "current_world": "daily",
        "entities": {
            "enabled": True,
            "rrf_weight": 1.0,
            "top_k": 20,
            "seeds": [_seed()],
        },
        "query_expansion": {"enabled": False},
        "random_surfacing": {},
    }
    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "_entity_store", None)
    monkeypatch.setattr(server, "_entity_store_key", None)
    monkeypatch.setattr(server, "_entity_store_initialized", False)
    monkeypatch.setattr(server, "_entity_sync_locks", {})
    manager = _Manager(
        buckets,
        tmp_path / "archive",
        canonical_keyword=canonical_keyword,
    )
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", _Embedding())
    monkeypatch.setattr(server, "dehydrator", _Dehydrator())
    monkeypatch.setattr(server, "decay_engine", _Decay())
    monkeypatch.setattr(server, "consolidation_engine", _NoopLoop())
    monkeypatch.setattr(server, "episode_engine", _NoopLoop())
    monkeypatch.setattr(server, "_backfill_started", True)
    return cfg, manager


@pytest.mark.asyncio
async def test_four_alias_queries_return_same_core_bucket(tmp_path, monkeypatch):
    core = _bucket("core", "朝灯的核心记忆")
    cfg, manager = _configure_server(
        tmp_path, monkeypatch, [core], canonical_keyword=True
    )
    EntityStore(cfg).resolve_and_link("core", core["content"])

    outputs = []
    for query in ("Vae", "Rosita", "朝灯", "老婆"):
        outputs.append(
            await server.breath(
                query=query,
                world="daily",
                relation_depth=0,
                include_images=False,
                include_body_state=False,
            )
        )

    assert all("[bucket_id:core]" in output for output in outputs)
    assert manager.queries == ["朝灯"] * 4
    assert server._resolve_entity_recall("老婆饼")[0] == "老婆饼"
    assert server._resolve_entity_recall("Vae2")[0] == "Vae2"


@pytest.mark.asyncio
async def test_entity_only_channel_reuses_all_authority_filters(tmp_path, monkeypatch):
    archive_dir = tmp_path / "archive"
    buckets = [
        _bucket("allowed", "老婆的允许证据"),
        _bucket("wrong-world", "老婆的异世界证据", world="role"),
        _bucket("wrong-domain", "老婆的生活证据", domain=["生活"]),
        _bucket("too-old", "老婆的旧证据", created="2025-01-01T00:00:00"),
        _bucket("pinned", "老婆的钉选证据", pinned=True),
        _bucket("z-bad", "老婆的冲突事实"),
        _bucket("nsfw-bad", "老婆的过滤证据"),
        _bucket(
            "archived",
            "老婆的冷库证据",
            bucket_type="archived",
            path=str(archive_dir / "archived.md"),
        ),
    ]
    cfg, _manager = _configure_server(tmp_path, monkeypatch, buckets)
    store = EntityStore(cfg)
    for bucket in buckets:
        store.resolve_and_link(bucket["id"], bucket["content"])

    original_z = server._filter_z_fact_candidates

    def z_filter(values, *, query, intent):
        return [bucket for bucket in values if bucket["id"] != "z-bad"]

    async def ds_filter(query, values, *, mode, max_results):
        return [bucket for bucket in values if bucket["id"] != "nsfw-bad"][:max_results]

    monkeypatch.setattr(server, "_filter_z_fact_candidates", z_filter)
    monkeypatch.setattr(server, "_ds_filter_candidates", ds_filter)
    try:
        result = await server.breath(
            query="老婆",
            domain="工程",
            world="daily",
            since="2026-07-01",
            relation_depth=0,
            max_results=20,
            include_images=False,
            include_body_state=False,
        )
    finally:
        monkeypatch.setattr(server, "_filter_z_fact_candidates", original_z)

    assert "[bucket_id:allowed]" in result
    for bucket_id in (
        "wrong-world",
        "wrong-domain",
        "too-old",
        "pinned",
        "z-bad",
        "nsfw-bad",
        "archived",
    ):
        assert f"[bucket_id:{bucket_id}]" not in result


@pytest.mark.asyncio
async def test_missing_entity_sidecar_is_readonly_and_fails_open(tmp_path, monkeypatch):
    legacy = _bucket("legacy", "legacy exact query")
    cfg, manager = _configure_server(
        tmp_path, monkeypatch, [legacy], canonical_keyword=False
    )

    async def legacy_search(query, limit=20, **kwargs):
        manager.queries.append(query)
        return [legacy] if query == "legacy exact query" else []

    manager.search = legacy_search
    entity_dir = Path(cfg["buckets_dir"]) / ".entities"

    result = await server.breath(
        query="legacy exact query",
        world="daily",
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )

    assert "[bucket_id:legacy]" in result
    assert not entity_dir.exists()


class _Barrier:
    @asynccontextmanager
    async def shared_async(self):
        yield


class _WriteManager:
    _maintenance_barrier = _Barrier()

    async def create(self, **kwargs):
        return "new-bucket"


class _WriteEmbedding:
    async def generate_and_store(self, bucket_id, content):
        return None


class _MergeManager:
    _maintenance_barrier = _Barrier()

    def __init__(self, bucket):
        self.bucket = bucket

    async def update(self, bucket_id, **kwargs):
        if "content" in kwargs:
            self.bucket["content"] = kwargs["content"]
        self.bucket["metadata"].update(
            {key: value for key, value in kwargs.items() if key != "content"}
        )
        return True

    async def get(self, bucket_id):
        return self.bucket if bucket_id == self.bucket["id"] else None


class _MergeDehydrator:
    async def merge(self, old_content, new_content):
        return "朝灯合并后的最终正文"


class _CompoundMergeDehydrator:
    async def merge(self, old_content, new_content):
        return "老婆饼和朝灯"


@pytest.mark.asyncio
async def test_merge_or_create_passes_validated_entities_to_sidecar(monkeypatch):
    linked = []

    async def no_candidates(**kwargs):
        return []

    monkeypatch.setattr(server, "_find_merge_candidates", no_candidates)
    monkeypatch.setattr(server, "bucket_mgr", _WriteManager())
    monkeypatch.setattr(server, "embedding_engine", _WriteEmbedding())
    monkeypatch.setattr(
        server,
        "_link_bucket_entities",
        lambda bucket_id, content, candidates=None: linked.append(
            (bucket_id, content, candidates)
        ),
    )
    entities = [{"mention": "朝灯", "type": "person"}]

    result = await server._merge_or_create(
        content="朝灯完成了测试",
        tags=[],
        importance=5,
        domain=["工程"],
        valence=0.5,
        arousal=0.3,
        entities=entities,
    )

    assert result == ("new-bucket", "new-bucket", False)
    assert linked == [("new-bucket", "朝灯完成了测试", entities)]


@pytest.mark.asyncio
async def test_merge_revalidates_candidates_against_final_body(tmp_path, monkeypatch):
    existing = _bucket("core", "朝灯原有正文")
    cfg, _manager = _configure_server(tmp_path, monkeypatch, [existing])
    manager = _MergeManager(existing)

    async def candidates(**kwargs):
        return [existing]

    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", _MergeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", _WriteEmbedding())
    monkeypatch.setattr(server, "_find_merge_candidates", candidates)
    monkeypatch.setattr(server, "_merge_candidate_passes_threshold", lambda bucket: True)
    monkeypatch.setattr(server, "_is_merge_protected_bucket", lambda *args: False)
    monkeypatch.setattr(server, "build_supersedes_audit", lambda *args: [])

    result = await server._merge_or_create(
        content="老婆带来的新正文",
        tags=[],
        importance=5,
        domain=["工程"],
        valence=0.5,
        arousal=0.3,
        entities=[{"mention": "老婆", "type": "person"}],
    )

    assert result == ("core", "core", True)
    store = server._get_entity_store(initialize=False)
    assert store is not None
    for query in ("Vae", "Rosita", "朝灯", "老婆"):
        assert store.linked_bucket_ids(query) == ["core"]
    assert store.link_is_current("core", "朝灯合并后的最终正文")


@pytest.mark.asyncio
async def test_merge_compound_candidate_cannot_abort_seed_hash_refresh(
    tmp_path, monkeypatch
):
    existing = _bucket("core", "老婆的旧正文")
    cfg, _manager = _configure_server(tmp_path, monkeypatch, [existing])
    manager = _MergeManager(existing)
    EntityStore(cfg).resolve_and_link("core", existing["content"])

    async def candidates(**kwargs):
        return [existing]

    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", _CompoundMergeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", _WriteEmbedding())
    monkeypatch.setattr(server, "_find_merge_candidates", candidates)
    monkeypatch.setattr(server, "_merge_candidate_passes_threshold", lambda bucket: True)
    monkeypatch.setattr(server, "_is_merge_protected_bucket", lambda *args: False)
    monkeypatch.setattr(server, "build_supersedes_audit", lambda *args: [])

    result = await server._merge_or_create(
        content="老婆带来的新正文",
        tags=[],
        importance=5,
        domain=["工程"],
        valence=0.5,
        arousal=0.3,
        entities=[{"mention": "老婆", "type": "person"}],
    )

    assert result == ("core", "core", True)
    store = server._get_entity_store(initialize=False)
    assert store is not None
    assert store.linked_bucket_ids("老婆") == ["core"]
    assert store.link_is_current("core", "老婆饼和朝灯")


@pytest.mark.asyncio
async def test_entity_sync_relinks_newer_body_after_post_write_race(monkeypatch):
    calls = []

    class RacingManager:
        async def get(self, bucket_id):
            return {"id": bucket_id, "content": "newer body", "metadata": {}}

    monkeypatch.setattr(server, "bucket_mgr", RacingManager())
    monkeypatch.setattr(
        server,
        "_link_bucket_entities",
        lambda bucket_id, content, candidates=None: calls.append(
            (bucket_id, content, candidates)
        ),
    )

    await server._synchronize_bucket_entities(
        "bucket",
        "older body",
        [{"mention": "older", "type": "project"}],
    )

    assert calls == [
        ("bucket", "newer body", None),
    ]


@pytest.mark.asyncio
async def test_entity_sync_serializes_late_old_writers_on_latest_body(monkeypatch):
    calls = []

    class LatestManager:
        async def get(self, bucket_id):
            await asyncio.sleep(0)
            return {"id": bucket_id, "content": "final body", "metadata": {}}

    monkeypatch.setattr(server, "bucket_mgr", LatestManager())
    monkeypatch.setattr(server, "_entity_sync_locks", {})
    monkeypatch.setattr(
        server,
        "_link_bucket_entities",
        lambda bucket_id, content, candidates=None: calls.append(
            (bucket_id, content, candidates)
        ),
    )

    await asyncio.gather(
        server._synchronize_bucket_entities("bucket", "old A", []),
        server._synchronize_bucket_entities("bucket", "old B", []),
        server._synchronize_bucket_entities("bucket", "final body", []),
    )

    assert calls == [
        ("bucket", "final body", None),
        ("bucket", "final body", None),
        ("bucket", "final body", []),
    ]


def test_entity_sync_lock_is_safe_across_worker_event_loops(monkeypatch):
    calls = []

    class CrossLoopManager:
        async def get(self, bucket_id):
            # Keep the process lock contended long enough to bind the old
            # asyncio.Lock implementation to a worker loop.
            await asyncio.sleep(0.02)
            return {"id": bucket_id, "content": "final body", "metadata": {}}

    monkeypatch.setattr(server, "bucket_mgr", CrossLoopManager())
    monkeypatch.setattr(server, "_entity_sync_locks", {})
    monkeypatch.setattr(server, "_entity_sync_locks_guard", threading.Lock())
    monkeypatch.setattr(
        server,
        "_link_bucket_entities",
        lambda bucket_id, content, candidates=None: calls.append(content),
    )

    def run_once(content, barrier):
        barrier.wait()
        asyncio.run(server._synchronize_bucket_entities("shared", content, []))

    # The first contended round binds an asyncio.Lock; the second pair of fresh
    # loops reproduced RuntimeError before the loop-neutral lock fix.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for round_number in range(2):
            barrier = threading.Barrier(2)
            futures = [
                pool.submit(run_once, f"old-{round_number}-{index}", barrier)
                for index in range(2)
            ]
            for future in futures:
                future.result(timeout=5)

    assert calls == ["final body"] * 4
