"""Opt-in guard for high-DF literal collisions (2026-09-05 noise ledger)."""

import types

import pytest

import bm25_index
import bucket_manager
import server


CONVERSATION_THRESHOLD = 0.25


def _common_collision(**overrides) -> dict:
    bucket = {
        "id": "engineering-task",
        "_literal_relevance_score": 100.0,
        "_original_vector_relevance_score": 0.65,
        "_keyword_channel_match": True,
        "_literal_term_dfs": [{"term": "验收", "df": 656}],
    }
    bucket.update(overrides)
    return bucket


def test_guard_is_default_off_and_preserves_existing_score(monkeypatch):
    monkeypatch.delenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", raising=False)
    bucket = _common_collision()

    assert server._anchor_adapted_relevance_score(bucket) == pytest.approx(0.45)
    assert "_literal_collision_guard" not in bucket


def test_common_literal_with_noise_floor_vector_falls_below_anchor(monkeypatch):
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", "1")
    bucket = _common_collision()

    assert server._anchor_adapted_relevance_score(bucket) == pytest.approx(0.2475)
    assert bucket["_literal_collision_guard"] == "common_literal_weak_vector"


@pytest.mark.parametrize(
    "overrides",
    [
        {"_original_vector_relevance_score": 0.71},
        {"_rare_literal_terms": ["蚊子"]},
        {"_literal_term_dfs": [{"term": "蚊子", "df": 5}]},
        {"entity_match": True},
        {"_keyword_channel_match": False},
    ],
)
def test_strong_or_specific_evidence_is_not_guarded(monkeypatch, overrides):
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", "1")
    bucket = _common_collision(**overrides)

    assert server._anchor_adapted_relevance_score(bucket) == pytest.approx(0.45)
    assert "_literal_collision_guard" not in bucket


def test_pure_semantic_candidate_is_untouched(monkeypatch):
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", "1")
    bucket = {
        "_literal_relevance_score": 0.0,
        "_original_vector_relevance_score": 0.65,
    }

    score = server._anchor_adapted_relevance_score(bucket)
    assert score == pytest.approx(0.2925)
    assert score >= CONVERSATION_THRESHOLD


@pytest.mark.parametrize('evidence', [[], [{"term": "验收", "df": 0}]])
def test_missing_positive_df_fails_open(monkeypatch, evidence):
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", "1")
    bucket = _common_collision(_literal_term_dfs=evidence)
    assert server._anchor_adapted_relevance_score(bucket) == pytest.approx(0.45)


def test_single_character_literal_is_not_a_rare_exemption(monkeypatch):
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", "1")
    bucket = _common_collision(_literal_term_dfs=[
        {"term": "慢", "df": 5}, {"term": "验收", "df": 659},
    ])
    assert server._anchor_adapted_relevance_score(bucket) == pytest.approx(0.2475)


def test_floor_is_scoped_and_configurable(monkeypatch):
    monkeypatch.delenv("OMBRE_LITERAL_COLLISION_VECTOR_FLOOR", raising=False)
    assert server._literal_collision_vector_floor() == pytest.approx(0.71)
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_VECTOR_FLOOR", "0.64")
    assert server._literal_collision_vector_floor() == pytest.approx(0.64)
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_VECTOR_FLOOR", "2")
    assert server._literal_collision_vector_floor() == pytest.approx(1.0)
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_VECTOR_FLOOR", "bad")
    assert server._literal_collision_vector_floor() == pytest.approx(0.71)


def test_anchor_gate_drops_collision_but_keeps_semantic(monkeypatch):
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", "1")
    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_ENABLED", "1")
    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_POLICIES", "conversation")
    collision = _common_collision()
    semantic = {
        "id": "same-event-semantic",
        "_literal_relevance_score": 0.0,
        "_original_vector_relevance_score": 0.65,
    }

    kept = server._filter_anchor_policy_candidates(
        [collision, semantic],
        "conversation",
    )

    assert [bucket["id"] for bucket in kept] == ["same-event-semantic"]


def _build_index():
    pytest.importorskip("rank_bm25")
    index = bm25_index.BM25Index()
    buckets = [
        {"id": "mosquito", "metadata": {"name": "蚊子夜"}, "content": "蚊子 睡觉"},
        {"id": "order", "metadata": {"name": "工作台单子"}, "content": "老板 单子 任务"},
    ]
    for i in range(10):
        buckets.append(
            {
                "id": f"task-{i}",
                "metadata": {"name": f"任务{i}"},
                "content": "任务 验收 单子",
            }
        )
    index.build(buckets)
    return index


def test_literal_df_lookup_reports_only_requested_candidates():
    index = _build_index()

    hits = index.literal_term_df_hits(
        "我在给老板打单子，也问蚊子",
        bucket_ids={"mosquito", "order", "missing"},
    )

    assert hits["mosquito"] == (("蚊子", 1),)
    order = dict(hits["order"])
    assert order["老板"] == 1
    assert order["单子"] == 11
    assert "task-0" not in hits


def test_df_lookup_includes_single_character_terms():
    index = _build_index()
    index.build([{"id": "slow", "content": "慢 修 单"}])
    assert dict(index.literal_term_df_hits("慢 修 单", bucket_ids={"slow"})["slow"]) == {
        "慢": 1, "修": 1, "单": 1,
    }


def test_bucket_manager_df_lookup_fails_open():
    class _Manager:
        _bm25_mode = "live"
        _bm25 = None

    lookup = bucket_manager.BucketManager.literal_term_df_hits
    manager = _Manager()
    assert lookup(manager, "验收", bucket_ids={"a"}) == {}

    class _Boom:
        def literal_term_df_hits(self, *args, **kwargs):
            raise RuntimeError("boom")

    manager._bm25 = _Boom()
    assert lookup(manager, "验收", bucket_ids={"a"}) == {}
    manager._bm25 = _build_index()
    assert "order" in lookup(manager, "单子", bucket_ids={"order"})
    manager._bm25_mode = "off"
    assert lookup(manager, "单子", bucket_ids={"order"}) == {}


class _PromptClient:
    def __init__(self):
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content='{"keep": []}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


@pytest.mark.asyncio
async def test_ds_prompt_adds_domain_mismatch_rule_only_when_enabled(monkeypatch):
    client = _PromptClient()
    monkeypatch.setattr(
        server,
        "_ds_filter_provider",
        lambda: ("apiroute-gemini", "gemini-3.7-flash", client, {}),
    )
    monkeypatch.setenv("OMBRE_DS_FILTER_CACHE_TTL", "0")
    bucket = {"id": "engineering", "metadata": {}, "content": "工程任务验收"}

    monkeypatch.delenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", raising=False)
    await server._ds_semantic_select("想睡觉", [bucket], set(), 5)
    prompt_off = client.calls[-1]["messages"][0]["content"]
    assert "只共享「单子、任务、验收、慢、修、提速」等泛词" not in prompt_off

    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", "1")
    await server._ds_semantic_select("想睡觉", [bucket], set(), 5)
    prompt_on = client.calls[-1]["messages"][0]["content"]
    assert "只共享「单子、任务、验收、慢、修、提速」等泛词" in prompt_on
    assert "生活、亲密、情话或身体语境" in prompt_on
    assert "以下规则优先于上面的宽松保留规则" in prompt_on
    assert "仅出现同一人物名字不算相关" in prompt_on
    assert len(client.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_breath_wires_df_and_keyword_evidence(tmp_path, monkeypatch, enabled):
    from tests.test_pr1_noise_tools import (
        FakeBucketMgr, FakeDecay, FakeDehydrator, FakeEmbedding, _bucket,
    )

    manager = FakeBucketMgr([_bucket("collision", "任务验收")])
    lookups = []

    def lookup(query, *, bucket_ids):
        lookups.append((query, bucket_ids))
        return {"collision": (("验收", 659),)}

    manager.literal_term_df_hits = lookup
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", str(int(enabled)))
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "0")
    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_ENABLED", "1")
    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_POLICIES", "conversation")
    monkeypatch.setattr(server, "config", {
        **server.config, "buckets_dir": str(tmp_path),
        "entities": {"enabled": False}, "query_expansion": {"enabled": False},
        "random_surfacing": {},
    })
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding([("collision", 0.65)]))
    monkeypatch.setattr(server, "_backfill_started", True)
    result = await server.breath(
        query="验收", policy="conversation", max_results=1,
        relation_depth=0, include_images=False, include_body_state=False,
    )
    assert ("[bucket_id:collision]" in result) is not enabled
    assert lookups == ([("验收", {"collision"})] if enabled else [])
