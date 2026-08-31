import hashlib

import pytest

from relation_graph import PROVENANCE_GENERATION_METHOD


class _GrowDehydrator:
    async def ensure_self_contained(self, content, source_context=""):
        assert source_context == content
        return content

    async def analyze(self, _content):
        return {
            "domain": ["工程"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": ["回执"],
            "suggested_name": "短记忆",
            "entities": [],
        }

    async def digest(self, _content, **_kwargs):
        return [{
            "name": "长记忆",
            "content": "[[朝灯]]完成了一条足够长且可独立理解的 grow 回执测试记忆。",
            "domain": ["工程"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": ["回执"],
            "importance": 5,
            "entities": [],
        }]


class _TwoItemGrowDehydrator(_GrowDehydrator):
    async def digest(self, content, **kwargs):
        first = (await super().digest(content, **kwargs))[0]
        second = {
            **first,
            "name": "另一条长记忆",
            "content": "[[朝灯]]同一次 grow 还拆出了另一条可独立理解的测试记忆。",
        }
        return [first, second]


class _NoopEmbedding:
    async def generate_and_store(self, _bucket_id, _content):
        return True


async def _no_decay_background():
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_merged", "action"),
    [(False, "新建"), (True, "合并")],
)
async def test_short_grow_returns_bucket_id(monkeypatch, is_merged, action):
    import server

    writes = []

    async def _write(**kwargs):
        writes.append(kwargs)
        return "short-bucket-id", "短记忆", is_merged

    monkeypatch.setattr(server, "dehydrator", _GrowDehydrator())
    monkeypatch.setattr(server, "_ensure_decay_background", _no_decay_background)
    monkeypatch.setattr(server, "_merge_or_create", _write)
    monkeypatch.setattr(server, "_mark_briefing_cache_dirty", lambda _reason: None)

    source = "一条短记忆"
    result = await server.grow(source)

    assert result == f"{action} → 短记忆 | 工程 V0.5/A0.3 [short-bucket-id]"
    assert writes[0]["x_provenance"] == {
        "source_kind": "external",
        "source_digest": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_merged", "receipt", "summary"),
    [
        (False, "📝长记忆 [long-bucket-id]", "1条|新1合0"),
        (True, "📎长记忆 [long-bucket-id]", "1条|新0合1"),
    ],
)
async def test_long_grow_returns_bucket_id(
    monkeypatch,
    is_merged,
    receipt,
    summary,
):
    import server

    writes = []

    async def _write(**kwargs):
        writes.append(kwargs)
        return "long-bucket-id", "长记忆", is_merged

    monkeypatch.setattr(server, "dehydrator", _GrowDehydrator())
    monkeypatch.setattr(server, "_ensure_decay_background", _no_decay_background)
    monkeypatch.setattr(server, "_merge_or_create", _write)
    monkeypatch.setattr(server, "_mark_briefing_cache_dirty", lambda _reason: None)

    source = "甲" * 30
    result = await server.grow(source)

    assert result == f"{summary}\n{receipt}"
    assert writes[0]["x_provenance"] == {
        "source_kind": "external",
        "source_digest": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


@pytest.mark.asyncio
async def test_long_grow_auto_links_siblings_from_the_same_source(
    monkeypatch,
    bucket_mgr,
):
    import server

    async def _always_new(*_args, **_kwargs):
        return "new"

    async def _no_entity_sync(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "dehydrator", _TwoItemGrowDehydrator())
    monkeypatch.setattr(server, "embedding_engine", _NoopEmbedding())
    monkeypatch.setattr(server, "_ensure_decay_background", _no_decay_background)
    monkeypatch.setattr(server, "_recall_before_write_decision", _always_new)
    monkeypatch.setattr(server, "_synchronize_bucket_entities", _no_entity_sync)
    monkeypatch.setattr(server, "_mark_briefing_cache_dirty", lambda _reason: None)

    source = "同一份 grow 原料会被拆成两条独立记忆，并保留可审计的同源关系。" * 2
    expected_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    result = await server.grow(source)

    buckets = await bucket_mgr.list_all(include_archive=True)
    siblings = [
        bucket
        for bucket in buckets
        if bucket["metadata"].get("source_digest") == expected_digest
    ]
    assert result.startswith("2条|新2合0")
    assert len(siblings) == 2
    assert all(
        bucket["metadata"].get("source_kind") == "external"
        for bucket in siblings
    )

    relations = [
        (bucket["id"], relation)
        for bucket in siblings
        for relation in bucket["metadata"].get("relations", [])
    ]
    assert len(relations) == 1
    source_id, relation = relations[0]
    assert {source_id, relation["target"]} == {
        bucket["id"] for bucket in siblings
    }
    assert relation["type"] == "kin"
    assert relation["generation_method"] == PROVENANCE_GENERATION_METHOD
    assert relation["evidence"]["bases"] == [{
        "kind": "shared_provenance",
        "field": "source_digest",
        "value": expected_digest,
    }]

    replay = await bucket_mgr.auto_link_created_bucket(siblings[-1]["id"])
    assert replay["created"] == 0
    assert sum(
        len(bucket["metadata"].get("relations", []))
        for bucket in await bucket_mgr.list_all(include_archive=True)
    ) == 1
