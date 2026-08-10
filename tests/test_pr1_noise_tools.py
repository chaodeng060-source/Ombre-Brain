import json
import types

import pytest
from mcp.types import ImageContent, TextContent

import server
from dehydrator import DEHYDRATE_PROMPT


def _bucket(
    bucket_id: str,
    content: str,
    *,
    name: str = None,
    importance: int = 5,
    tags: list[str] = None,
    domain: list[str] = None,
    bucket_type: str = "dynamic",
    score: float = 10.0,
) -> dict:
    return {
        "id": bucket_id,
        "content": content,
        "score": score,
        "metadata": {
            "id": bucket_id,
            "name": name or bucket_id,
            "importance": importance,
            "tags": tags or [],
            "domain": domain or ["工程"],
            "type": bucket_type,
            "valence": 0.5,
            "arousal": 0.3,
            "last_active": "2026-05-22T12:00:00",
        },
    }


class FakeDecay:
    is_running = True

    async def ensure_started(self):
        return None

    def calculate_score(self, meta):
        return float(meta.get("importance", 5))
    def apply_retrieval_decay(self, score, meta):
        return score


class FakeDehydrator:
    async def dehydrate(self, content, meta, *, write_cache=True):
        assert write_cache is False
        return f"SUMMARY:{content}"


class FakeEmbedding:
    def __init__(self, hits=None):
        self.hits = list(hits or [])
        self.top_k_calls = []

    async def search_similar(self, query, top_k=20):
        self.top_k_calls.append(top_k)
        return self.hits[:top_k]


class FakeBucketMgr:
    def __init__(self, buckets):
        self.buckets = list(buckets)
        self.search_limits = []
        self.search_queries = []
        self.preloaded_candidates = []
        self.touched = []
        self.list_all_calls = 0

    async def get_stats(self):
        return {
            "permanent_count": 0,
            "dynamic_count": len(self.buckets),
            "archive_count": 0,
            "total_size_kb": 1.0,
        }

    async def list_all(self, include_archive=False):
        self.list_all_calls += 1
        return list(self.buckets)

    async def search(self, query, limit=20, **kwargs):
        self.search_limits.append(limit)
        self.search_queries.append(query)
        self.preloaded_candidates.append(kwargs.get("preloaded_buckets"))
        return list(self.buckets)[:limit]

    async def get(self, bucket_id):
        return next((b for b in self.buckets if b["id"] == bucket_id), None)

    async def touch(self, bucket_id):
        self.touched.append(bucket_id)


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_pulse_defaults_to_navigator_and_full_preserves_old_listing(monkeypatch):
    content = json.dumps(
        {
            "core_facts": ["5.21朝灯定下海马体减噪三层方案"],
            "summary": "5.21朝灯定下海马体减噪三层方案",
        },
        ensure_ascii=False,
    )
    fake_mgr = FakeBucketMgr([_bucket("b1", content, name="减噪方案", importance=8)])
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())

    nav = await server.pulse()
    assert "=== 记忆导航 (Top 1 / 共 1) ===" in nav
    assert "摘要:5.21朝灯定下海马体减噪三层方案" in nav
    assert "inspect:b1" in nav
    assert "标签:" not in nav

    full = await server.pulse(full=True)
    assert "=== 记忆列表 (全量 1 个) ===" in full
    assert "标签:" in full
    assert "情感:V0.5/A0.3" in full


def test_session_seen_helpers_filter_by_session(tmp_path, monkeypatch):
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    buckets = [_bucket("a", "A"), _bucket("b", "B")]

    server._remember_session_seen_ids("session/one", ["a"])

    remaining = server._filter_session_seen(buckets, "session/one")
    assert [b["id"] for b in remaining] == ["b"]
    assert [b["id"] for b in server._filter_session_seen(buckets, "session/two")] == ["a", "b"]


@pytest.mark.asyncio
async def test_breath_uses_recall_pool_caps_output_and_dedups_session(tmp_path, monkeypatch):
    buckets = [
        _bucket("a", "alpha"),
        _bucket("b", "beta"),
        _bucket("c", "gamma"),
    ]
    fake_mgr = FakeBucketMgr(buckets)
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    first = await server.breath(
        query="工程",
        max_results=2,
        relation_depth=0,
        session_id="s1",
        include_images=False,
    )
    assert "[bucket_id:a]" in first
    assert "[bucket_id:b]" in first
    assert "[bucket_id:c]" not in first
    assert fake_mgr.search_limits[-1] == server.BREATH_RECALL_POOL_SIZE
    assert fake_mgr.list_all_calls == 1

    second = await server.breath(
        query="工程",
        max_results=2,
        relation_depth=0,
        session_id="s1",
        include_images=False,
    )
    assert "[bucket_id:a]" not in second
    assert "[bucket_id:b]" not in second
    assert "[bucket_id:c]" in second
    assert fake_mgr.touched == []


@pytest.mark.asyncio
async def test_breath_reuses_one_bucket_snapshot_across_query_angles(
    tmp_path,
    monkeypatch,
):
    fake_mgr = FakeBucketMgr([_bucket("a", "alpha")])
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setitem(
        server.config,
        "query_expansion",
        {"enabled": True, "allowed_intents": ["recall"]},
    )
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    async def fake_expand(query, *_args, **_kwargs):
        return [query, "第二角度", "第三角度"]

    monkeypatch.setattr(server, "expand_query", fake_expand)

    await server.breath(
        query="你还记得吗",
        max_results=1,
        relation_depth=0,
        include_images=False,
    )

    assert fake_mgr.list_all_calls == 1
    assert fake_mgr.search_queries == ["你还记得吗", "第二角度", "第三角度"]
    assert len({id(rows) for rows in fake_mgr.preloaded_candidates}) == 1


@pytest.mark.asyncio
async def test_api_breath_search_does_not_touch_returned_buckets(tmp_path, monkeypatch):
    fake_mgr = FakeBucketMgr([_bucket("a", "alpha")])
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    response = await server.api_breath(
        _JsonRequest(
            {
                "query": "工程",
                "max_results": 1,
                "relation_depth": 0,
            }
        )
    )

    assert response.status_code == 200
    assert "[bucket_id:a]" in json.loads(response.body)["raw"]
    assert fake_mgr.touched == []


def test_ds_keep_parser_distinguishes_valid_empty_from_bad_payload():
    assert server._parse_ds_keep_indices('{"keep": []}', 3) == []
    assert server._parse_ds_keep_indices('{"keep": [0, 2]}', 3) == [0, 2]
    assert server._parse_ds_keep_indices('not-json', 3) is None
    assert server._parse_ds_keep_indices('{"keep": [99]}', 3) is None
    assert server._normalize_anchor_recall_policy("conversation") == "conversation"
    assert server._normalize_anchor_recall_policy("made-up") == "search"


def test_anchor_quality_gate_uses_adapted_absolute_score(monkeypatch):
    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_ENABLED", "1")
    monkeypatch.setenv(
        "OMBRE_ANCHOR_QUALITY_GATE_POLICIES",
        "conversation,reflex",
    )
    high_vector = _bucket("high-vector", "A")
    high_vector["_original_vector_relevance_score"] = 0.60
    low_vector = _bucket("low-vector", "B")
    low_vector["_original_vector_relevance_score"] = 0.54
    high_literal = _bucket("high-literal", "C")
    high_literal["_literal_relevance_score"] = 60.0
    low_literal = _bucket("low-literal", "D")
    low_literal["_literal_relevance_score"] = 54.0
    anchored_entity = _bucket("entity", "E")
    anchored_entity["entity_match"] = True
    unscored = _bucket("unscored", "F")
    candidates = [
        high_vector,
        low_vector,
        high_literal,
        low_literal,
        anchored_entity,
        unscored,
    ]

    kept = server._filter_anchor_policy_candidates(candidates, "conversation")

    assert [bucket["id"] for bucket in kept] == [
        "high-vector",
        "high-literal",
        "entity",
        "unscored",
    ]
    assert high_vector["_anchor_adapted_relevance_score"] == 0.27
    assert low_vector["_anchor_adapted_relevance_score"] == 0.243
    # Search/manual recall keeps its wider existing contract.
    assert server._filter_anchor_policy_candidates(candidates, "search") == candidates


def test_anchor_quality_gate_is_configurable_and_fail_open(monkeypatch):
    candidate = _bucket("low", "A")
    candidate["_original_vector_relevance_score"] = 0.01

    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_ENABLED", "0")
    assert server._filter_anchor_policy_candidates(
        [candidate], "conversation"
    ) == [candidate]

    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_ENABLED", "1")
    assert server._filter_anchor_policy_candidates(
        [_bucket("unscored", "B")], "conversation"
    )[0]["id"] == "unscored"


@pytest.mark.asyncio
async def test_anchor_conversation_policy_allows_valid_empty(monkeypatch):
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")

    async def _create(**_kw):
        msg = types.SimpleNamespace(content='{"keep": []}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator_with_response(_create))
    buckets = [_bucket("a", "A"), _bucket("b", "B")]

    assert await server._ds_filter_candidates(
        "无关查询", buckets, mode="search", max_results=2, allow_empty=True
    ) == []
    assert [b["id"] for b in await server._ds_filter_candidates(
        "主动搜索", buckets, mode="search", max_results=2, allow_empty=False
    )] == ["a", "b"]


@pytest.mark.asyncio
async def test_breath_conversation_gate_can_stay_silent_while_search_remains_wide(
    tmp_path,
    monkeypatch,
):
    bucket = _bucket("a", "alpha")
    fake_mgr = FakeBucketMgr([bucket])
    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_ENABLED", "1")
    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_POLICIES", "conversation,reflex")
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "0")
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    conversation = await server.breath(
        query="O.o",
        policy="conversation",
        max_results=1,
        relation_depth=0,
        include_images=False,
    )
    search = await server.breath(
        query="O.o",
        policy="search",
        max_results=1,
        relation_depth=0,
        include_images=False,
    )

    assert conversation == "未找到相关记忆。"
    assert "[bucket_id:a]" in search


@pytest.mark.asyncio
async def test_api_breath_forwards_anchor_policy(monkeypatch):
    observed = {}

    async def fake_breath(**kwargs):
        observed.update(kwargs)
        return "未找到相关记忆。"

    monkeypatch.setattr(server, "breath", fake_breath)
    response = await server.api_breath(
        _JsonRequest({"query": "随便问问", "policy": "conversation"})
    )

    payload = json.loads(response.body)
    assert observed["policy"] == "conversation"
    assert payload["policy"] == "conversation"


@pytest.mark.asyncio
async def test_breath_applies_intent_recall_channel_budgets(tmp_path, monkeypatch):
    buckets = [
        _bucket("a", "alpha", domain=["恋爱"]),
        _bucket("b", "beta", domain=["工程"]),
        _bucket("c", "gamma", domain=["工程"]),
    ]
    fake_mgr = FakeBucketMgr(buckets)
    fake_embedding = FakeEmbedding()
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setitem(server.config, "rrf", {"k": 60, "keyword_weight": 1.0, "vector_weight": 1.0})
    monkeypatch.setitem(
        server.config,
        "intent_recall",
        {
            "enabled": True,
            "min_confidence": 0.55,
            "policies": {
                "relation": {
                    "keyword_top_k_multiplier": 2.0,
                    "vector_top_k_multiplier": 3.0,
                    "keyword_weight_multiplier": 0.5,
                    "vector_weight_multiplier": 2.0,
                    "relation_neighbor_limit": 9,
                }
            },
        },
    )
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", fake_embedding)
    monkeypatch.setattr(server, "_backfill_started", True)

    result = await server.breath(
        query="我俩最近怎么样",
        max_results=1,
        relation_depth=0,
        include_images=False,
    )

    assert "[bucket_id:a]" in result
    assert fake_mgr.search_limits[-1] == server.BREATH_RECALL_POOL_SIZE * 2
    assert fake_embedding.top_k_calls[-1] == server.BREATH_RECALL_POOL_SIZE * 3


@pytest.mark.asyncio
async def test_breath_relation_neighbors_do_not_exceed_max_results(tmp_path, monkeypatch):
    buckets = [
        _bucket("a", "alpha"),
        _bucket("b", "beta"),
        _bucket("c", "gamma"),
    ]
    buckets[0]["metadata"]["relations"] = [{"type": "kin", "target": "c"}]
    fake_mgr = FakeBucketMgr(buckets)
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    result = await server.breath(
        query="工程",
        max_results=2,
        relation_depth=1,
        include_images=False,
    )

    assert "[bucket_id:a]" in result
    assert "[bucket_id:b]" in result
    assert "[bucket_id:c]" not in result


@pytest.mark.asyncio
async def test_ds_filter_stub_preserves_order_and_caps():
    buckets = [_bucket("a", "A"), _bucket("b", "B"), _bucket("c", "C")]
    selected = await server._ds_filter_candidates(
        "query",
        buckets,
        mode="search",
        max_results=2,
    )
    assert [b["id"] for b in selected] == ["a", "b"]


def test_mcp_image_whitelist_and_markdown_extraction():
    md = "before\n![辣椒酱](https://pub-test.r2.dev/a.png)\nafter"
    assert server._extract_markdown_images(md) == [
        ("辣椒酱", "https://pub-test.r2.dev/a.png")
    ]
    assert server._bucket_allows_mcp_image(_bucket("hi", md, importance=8))
    assert server._bucket_allows_mcp_image(_bucket("tag", md, tags=["多模态anchor"]))
    assert not server._bucket_allows_mcp_image(_bucket("low", md, importance=7))
    assert server._is_r2_image_url("https://pub-test.r2.dev/a.png")
    assert not server._is_r2_image_url("https://example.com/a.png")


@pytest.mark.asyncio
async def test_tool_result_returns_mcp_content_only_when_images_exist(monkeypatch):
    async def fake_collect(_buckets):
        return [ImageContent(type="image", data="YWJj", mimeType="image/png")]

    monkeypatch.setattr(server, "_collect_mcp_images", fake_collect)
    result = await server._tool_result_with_optional_images("hello", [_bucket("a", "A")], True)

    assert isinstance(result, list)
    assert isinstance(result[0], TextContent)
    assert isinstance(result[1], ImageContent)
    assert result[0].text == "hello"

    text_only = await server._tool_result_with_optional_images("hello", [], False)
    assert text_only == "hello"


def test_dehydrate_prompt_locks_concrete_noise_reduction_rules():
    assert "时间 + 主体 + 事件/动作 + 对象 + 影响" in DEHYDRATE_PROMPT
    assert "不要编日期" in DEHYDRATE_PROMPT
    assert "优于“讨论记忆库优化”" in DEHYDRATE_PROMPT


def _fake_dehydrator_with_response(create_fn):
    return types.SimpleNamespace(
        model="deepseek-chat",
        client=types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create_fn)
            )
        ),
    )


def _fake_dehydrator_with_decision_cache(create_fn):
    cache = {}
    fake = _fake_dehydrator_with_response(create_fn)
    fake.base_url = "https://api.deepseek.com/v1"
    fake.get_recall_ds_filter_decision = cache.get

    def _set(cache_key, selected_indices):
        cache[cache_key] = list(selected_indices)
        return True

    fake.set_recall_ds_filter_decision = _set
    return fake, cache


@pytest.mark.asyncio
async def test_ds_gate_off_by_default_is_pure_stub(monkeypatch):
    monkeypatch.delenv("OMBRE_DS_FILTER_ENABLED", raising=False)
    buckets = [_bucket("a", "A"), _bucket("b", "B"), _bucket("c", "C")]
    selected = await server._ds_filter_candidates(
        "query", buckets, mode="search", max_results=2
    )
    assert [b["id"] for b in selected] == ["a", "b"]


@pytest.mark.asyncio
async def test_ds_gate_subtractive_when_enabled(monkeypatch):
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")

    async def _create(**_kw):
        msg = types.SimpleNamespace(content='{"keep": [0, 2]}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator_with_response(_create))
    buckets = [_bucket("a", "A"), _bucket("b", "B"), _bucket("c", "C")]
    selected = await server._ds_filter_candidates(
        "工程", buckets, mode="search", max_results=3
    )
    # capped=[a,b,c]; DeepSeek keeps idx 0,2 -> subtractive -> a,c
    assert [b["id"] for b in selected] == ["a", "c"]


@pytest.mark.asyncio
async def test_ds_gate_exact_success_cache_reuses_only_same_contract(monkeypatch):
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")
    calls = 0

    async def _create(**_kw):
        nonlocal calls
        calls += 1
        msg = types.SimpleNamespace(content='{"keep": [0, 2]}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    fake, cache = _fake_dehydrator_with_decision_cache(_create)
    monkeypatch.setattr(server, "dehydrator", fake)
    long_prefix = "前" * 200
    buckets = [
        _bucket("a", "A", name="first"),
        _bucket("b", long_prefix + "原始结尾", name="second"),
        _bucket("c", "C", name="third"),
    ]

    first = await server._ds_filter_candidates(
        "工程", buckets, mode="search", max_results=3
    )
    second = await server._ds_filter_candidates(
        "工程", buckets, mode="search", max_results=3
    )
    assert [b["id"] for b in first] == ["a", "c"]
    assert [b["id"] for b in second] == ["a", "c"]
    assert calls == 1
    assert len(cache) == 1

    changed = [_bucket("a", "A"), _bucket("b", "B changed"), _bucket("c", "C")]
    await server._ds_filter_candidates(
        "工程", changed, mode="search", max_results=3
    )
    await server._ds_filter_candidates(
        "工程", buckets, mode="search", max_results=3, allow_empty=True
    )
    await server._ds_filter_candidates(
        "工程", buckets, mode="search", max_results=3, force_keep_ids={"b"}
    )
    await server._ds_filter_candidates(
        "工程", buckets, mode="search", max_results=2
    )
    changed_after_snippet = [
        buckets[0],
        _bucket("b", long_prefix + "变更结尾", name="second"),
        buckets[2],
    ]
    await server._ds_filter_candidates(
        "工程", changed_after_snippet, mode="search", max_results=3
    )
    changed_id_only = [
        _bucket("renamed-a", "A", name="first"),
        buckets[1],
        buckets[2],
    ]
    await server._ds_filter_candidates(
        "工程", changed_id_only, mode="search", max_results=3
    )
    assert calls == 7
    assert len(cache) == 7


@pytest.mark.asyncio
async def test_ds_gate_falls_back_to_capped_on_error(monkeypatch):
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")

    async def _create(**_kw):
        raise RuntimeError("api boom")

    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator_with_response(_create))
    buckets = [_bucket("a", "A"), _bucket("b", "B"), _bucket("c", "C")]
    selected = await server._ds_filter_candidates(
        "工程", buckets, mode="search", max_results=2
    )
    assert [b["id"] for b in selected] == ["a", "b"]


@pytest.mark.asyncio
async def test_ds_gate_failure_is_never_cached(monkeypatch):
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")
    calls = 0

    async def _create(**_kw):
        nonlocal calls
        calls += 1
        raise RuntimeError("api boom")

    fake, cache = _fake_dehydrator_with_decision_cache(_create)
    monkeypatch.setattr(server, "dehydrator", fake)
    buckets = [_bucket("a", "A"), _bucket("b", "B")]

    for _ in range(2):
        selected = await server._ds_filter_candidates(
            "工程", buckets, mode="search", max_results=2
        )
        assert [b["id"] for b in selected] == ["a", "b"]
    assert calls == 2
    assert cache == {}


@pytest.mark.asyncio
async def test_ds_gate_skips_surfacing_without_query(monkeypatch):
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search,surfacing")

    async def _create(**_kw):
        raise AssertionError("LLM must not be called for empty-query surfacing")

    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator_with_response(_create))
    buckets = [_bucket("a", "A"), _bucket("b", "B")]
    selected = await server._ds_filter_candidates(
        "", buckets, mode="surfacing", max_results=2
    )
    assert [b["id"] for b in selected] == ["a", "b"]


# --- 多模态 anchor 补全：anchor 优先级 / feel 出图 / breath e2e ---

def test_feel_bucket_is_mcp_image_eligible():
    md = "![绿月夜](https://pub-test.r2.dev/feel.png)"
    # feel 桶带图 = 私密锚点，应可出图
    assert server._bucket_allows_mcp_image(_bucket("f", md, bucket_type="feel"))
    # 回归：普通低权重非 feel 非 anchor 仍不出图
    assert not server._bucket_allows_mcp_image(_bucket("low", md, importance=7))
    assert server._is_anchor_bucket(_bucket("a", md, tags=["多模态anchor"]))
    assert not server._is_anchor_bucket(_bucket("p", md, importance=9))


@pytest.mark.asyncio
async def test_collect_mcp_images_prioritizes_anchor_buckets(monkeypatch):
    # MAX_ITEMS 截断时，anchor 桶应优先占满，非 anchor（即便 eligible）被挤掉
    monkeypatch.setattr(server, "MCP_IMAGE_MAX_ITEMS", 2)
    fetched = []

    async def fake_fetch(bucket, url):
        fetched.append(bucket["id"])
        return ImageContent(type="image", data="YWJj", mimeType="image/png")

    monkeypatch.setattr(server, "_fetch_mcp_image_content", fake_fetch)
    md = lambda n: f"![x](https://pub-test.r2.dev/{n}.png)"
    plain = _bucket("plain", md("p"), importance=9)            # eligible 但非 anchor
    anchor1 = _bucket("anchor1", md("a1"), tags=["多模态anchor"])
    anchor2 = _bucket("anchor2", md("a2"), tags=["锚"])

    images = await server._collect_mcp_images([plain, anchor1, anchor2])
    assert len(images) == 2
    assert fetched == ["anchor1", "anchor2"]  # anchor 优先，plain 在 cap 处被挤掉


@pytest.mark.asyncio
async def test_breath_search_emits_image_content_end_to_end(tmp_path, monkeypatch):
    md = "锚点\n![胸口](https://pub-test.r2.dev/anchor.png)\n尾巴"
    buckets = [_bucket("img", md, importance=9, tags=["多模态anchor"])]
    fake_mgr = FakeBucketMgr(buckets)
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    captured = []

    async def fake_fetch(bucket, url):
        captured.append(url)
        return ImageContent(type="image", data="YWJj", mimeType="image/png")

    monkeypatch.setattr(server, "_fetch_mcp_image_content", fake_fetch)

    result = await server.breath(query="工程", relation_depth=0, include_images=True)
    assert isinstance(result, list)
    assert any(isinstance(c, ImageContent) for c in result)
    assert isinstance(result[0], TextContent)
    assert "https://pub-test.r2.dev/anchor.png" in captured
