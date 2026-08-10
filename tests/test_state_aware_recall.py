import pytest

import server
from fact_slots import (
    STATE_VIEW_CURRENT,
    STATE_VIEW_HISTORICAL,
    STATE_VIEW_NEUTRAL,
    STATE_VIEW_TRANSITION,
    align_fact_state_candidates,
    fact_state_label,
    profile_fact_state_query,
)


REGISTRY = {
    "ui.primary_color": {
        "aliases": ["主色", "primary color"],
        "domains": ["ui"],
    },
}


def _bucket(bucket_id, status, *, fact_key="ui.primary_color", **metadata):
    return {
        "id": bucket_id,
        "content": f"{bucket_id} content",
        "metadata": {
            "name": bucket_id,
            "domain": ["ui"],
            "fact_key": fact_key,
            "fact_status": status,
            **metadata,
        },
    }


@pytest.mark.parametrize(
    ("query", "view"),
    [
        ("主色现在是什么？", STATE_VIEW_CURRENT),
        ("主色以前是什么？", STATE_VIEW_HISTORICAL),
        ("主色从以前到现在怎么变了？", STATE_VIEW_TRANSITION),
        ("这段 UI 故事提到了主色", STATE_VIEW_NEUTRAL),
    ],
)
def test_profile_fact_state_query_views(query, view):
    profile = profile_fact_state_query(query, REGISTRY)

    assert profile["view"] == view
    assert profile["fact_keys"] == ("ui.primary_color",)


def test_profile_fact_state_query_ignores_ambiguous_alias():
    registry = {
        "ui.primary_color": {"aliases": ["颜色"]},
        "ui.accent_color": {"aliases": ["颜色"]},
    }

    profile = profile_fact_state_query("以前的颜色是什么？", registry)

    assert profile["view"] == STATE_VIEW_NEUTRAL
    assert profile["fact_keys"] == ()


def test_align_fact_state_candidates_only_reorders_requested_slot():
    current = _bucket("current", "current")
    historical = _bucket("historical", "historical")
    unrelated = _bucket("other", "current", fact_key="ui.unknown")
    profile = profile_fact_state_query("主色以前是什么？", REGISTRY)

    aligned = align_fact_state_candidates(
        [current, unrelated, historical],
        profile=profile,
        registry=REGISTRY,
    )

    assert [bucket["id"] for bucket in aligned] == ["historical", "current", "other"]


def test_fact_state_label_requires_explicit_registered_state():
    assert fact_state_label(_bucket("current", "current"), REGISTRY) == "current"
    assert fact_state_label(_bucket("implicit", ""), REGISTRY) == ""
    assert fact_state_label(_bucket("unknown", "current", fact_key="other.key"), REGISTRY) == ""


class _BucketManager:
    def __init__(self, buckets):
        self.buckets = {bucket["id"]: bucket for bucket in buckets}

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)


class _SearchBucketManager(_BucketManager):
    def __init__(self, buckets, search_ids=None):
        super().__init__(buckets)
        self.search_ids = search_ids

    async def search(self, query, limit=20, **kwargs):
        selected = self.buckets.values()
        if self.search_ids is not None:
            selected = (
                self.buckets[bucket_id]
                for bucket_id in self.search_ids
                if bucket_id in self.buckets
            )
        return list(selected)[:limit]

    async def list_all(self, include_archive=False):
        return list(self.buckets.values())


class _Decay:
    is_running = True

    async def ensure_started(self):
        return None

    def apply_retrieval_decay(self, score, metadata):
        return score

    def calculate_score(self, metadata):
        return float(metadata.get("importance", 5))


class _Dehydrator:
    async def dehydrate(self, content, metadata, *, write_cache=True):
        assert write_cache is False
        return f"SUMMARY:{content}"


class _Embedding:
    async def search_similar(self, query, top_k=20):
        return []


def _enable_state_overlay(monkeypatch):
    monkeypatch.setitem(
        server.config,
        "fact_slots",
        {"enabled": True, "registry": REGISTRY},
    )
    monkeypatch.setitem(
        server.config,
        "state_aware_recall",
        {"enabled": True, "evidence_labels": True, "state_link_limit": 2},
    )


def _wire_breath(monkeypatch, tmp_path, manager):
    _enable_state_overlay(monkeypatch)
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "decay_engine", _Decay())
    monkeypatch.setattr(server, "dehydrator", _Dehydrator())
    monkeypatch.setattr(server, "embedding_engine", _Embedding())
    monkeypatch.setattr(server, "_backfill_started", True)


@pytest.mark.asyncio
async def test_state_link_current_view_follows_reviewed_reciprocal_pair(monkeypatch):
    _enable_state_overlay(monkeypatch)
    current = _bucket("new", "current", supersedes_bucket_ids=["old"])
    historical = _bucket("old", "historical", superseded_by_bucket_id="new")
    monkeypatch.setattr(server, "bucket_mgr", _BucketManager([current, historical]))

    results = await server._state_link_recall_candidates(
        [historical],
        profile=server._state_recall_profile("主色现在是什么？"),
        world_filter_set=None,
        domain_filter=None,
        created_after=None,
        created_before=None,
        excluded_ids=set(),
        limit=2,
    )

    assert [bucket["id"] for bucket in results] == ["new"]
    assert results[0]["_z_state_relation"] == "supersedes:old"


@pytest.mark.asyncio
async def test_state_link_historical_view_follows_reviewed_reciprocal_pair(monkeypatch):
    _enable_state_overlay(monkeypatch)
    current = _bucket("new", "current", supersedes_bucket_ids=["old"])
    historical = _bucket("old", "historical", superseded_by_bucket_id="new")
    monkeypatch.setattr(server, "bucket_mgr", _BucketManager([current, historical]))

    results = await server._state_link_recall_candidates(
        [current],
        profile=server._state_recall_profile("主色以前是什么？"),
        world_filter_set=None,
        domain_filter=None,
        created_after=None,
        created_before=None,
        excluded_ids=set(),
        limit=2,
    )

    assert [bucket["id"] for bucket in results] == ["old"]
    assert results[0]["_z_state_relation"] == "superseded_by:new"


@pytest.mark.asyncio
async def test_state_link_rejects_one_way_or_cross_slot_metadata(monkeypatch):
    _enable_state_overlay(monkeypatch)
    one_way = _bucket("one-way", "current")
    wrong_slot = _bucket(
        "wrong-slot",
        "current",
        fact_key="ui.unknown",
        supersedes_bucket_ids=["old"],
    )
    historical = _bucket("old", "historical", superseded_by_bucket_id="one-way")
    monkeypatch.setattr(
        server,
        "bucket_mgr",
        _BucketManager([one_way, wrong_slot, historical]),
    )

    results = await server._state_link_recall_candidates(
        [historical],
        profile=server._state_recall_profile("主色现在是什么？"),
        world_filter_set=None,
        domain_filter=None,
        created_after=None,
        created_before=None,
        excluded_ids=set(),
        limit=2,
    )

    assert results == []


def test_recall_prefix_exposes_state_without_downgrading_state_evidence(monkeypatch):
    _enable_state_overlay(monkeypatch)
    monkeypatch.setitem(server.config, "recall_evidence_roles", {"enabled": False})
    bucket = _bucket("old", "historical")
    profile = server._state_recall_profile("主色以前是什么？")

    main_prefix = server._recall_prefix(
        "old",
        "main",
        "curated_rrf",
        bucket=bucket,
        state_profile=profile,
    )
    state_prefix = server._recall_prefix(
        "old",
        "state",
        "z_lifecycle",
        bucket=bucket,
        state_profile=profile,
    )

    assert "[memory_state:historical]" in main_prefix
    assert "[query_state_view:historical]" in main_prefix
    assert "[authority:state_evidence]" in state_prefix
    assert "supporting_only" not in state_prefix


def test_recall_prefix_is_unchanged_for_neutral_query(monkeypatch):
    _enable_state_overlay(monkeypatch)
    monkeypatch.setitem(server.config, "recall_evidence_roles", {"enabled": False})
    bucket = _bucket("new", "current")
    profile = server._state_recall_profile("聊聊 UI 的故事")

    prefix = server._recall_prefix(
        "new",
        "main",
        "curated_rrf",
        bucket=bucket,
        state_profile=profile,
    )

    assert prefix == "[bucket_id:new]"


def test_state_overlay_disable_restores_pre_overlay_transition_filter(monkeypatch):
    _enable_state_overlay(monkeypatch)
    historical = _bucket("old", "historical")

    enabled = server._filter_z_fact_candidates(
        [historical],
        query="主色怎么变化的？",
        intent="fact",
    )
    monkeypatch.setitem(
        server.config,
        "state_aware_recall",
        {"enabled": False},
    )
    disabled = server._filter_z_fact_candidates(
        [historical],
        query="主色怎么变化的？",
        intent="fact",
    )

    assert enabled == [historical]
    assert disabled == []


@pytest.mark.asyncio
async def test_breath_foregrounds_current_historical_and_transition_views(
    tmp_path,
    monkeypatch,
):
    current = _bucket(
        "new",
        "current",
        supersedes_bucket_ids=["old"],
        importance=7,
    )
    historical = _bucket(
        "old",
        "historical",
        superseded_by_bucket_id="new",
        importance=7,
    )
    manager = _SearchBucketManager([current, historical])
    _wire_breath(monkeypatch, tmp_path, manager)

    current_result = await server.breath(
        query="主色现在是什么？",
        max_results=1,
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )
    historical_result = await server.breath(
        query="主色以前是什么？",
        max_results=1,
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )
    transition_result = await server.breath(
        query="主色从以前到现在怎么变了？",
        max_results=2,
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )

    assert "[bucket_id:new]" in current_result
    assert "[memory_state:current]" in current_result
    assert "[bucket_id:old]" not in current_result
    assert "[bucket_id:old]" in historical_result
    assert "[memory_state:historical]" in historical_result
    assert transition_result.index("[bucket_id:new]") < transition_result.index(
        "[bucket_id:old]"
    )
    assert "[query_state_view:transition]" in transition_result


@pytest.mark.asyncio
async def test_breath_recovers_current_through_reviewed_state_link(
    tmp_path,
    monkeypatch,
):
    current = _bucket("new", "current", supersedes_bucket_ids=["old"])
    historical = _bucket("old", "historical", superseded_by_bucket_id="new")
    manager = _SearchBucketManager(
        [current, historical],
        search_ids=["old"],
    )
    _wire_breath(monkeypatch, tmp_path, manager)

    result = await server.breath(
        query="主色现在是什么？",
        max_results=1,
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )

    assert "[bucket_id:new]" in result
    assert "[authority:state_evidence]" in result
    assert "[relation:supersedes:old]" in result
    assert "[bucket_id:old]" not in result
