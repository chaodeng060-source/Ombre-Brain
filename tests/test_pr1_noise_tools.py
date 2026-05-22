import json

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


class FakeDehydrator:
    async def dehydrate(self, content, meta):
        return f"SUMMARY:{content}"


class FakeEmbedding:
    async def search_similar(self, query, top_k=20):
        return []


class FakeBucketMgr:
    def __init__(self, buckets):
        self.buckets = list(buckets)
        self.search_limits = []
        self.touched = []

    async def get_stats(self):
        return {
            "permanent_count": 0,
            "dynamic_count": len(self.buckets),
            "archive_count": 0,
            "total_size_kb": 1.0,
        }

    async def list_all(self, include_archive=False):
        return list(self.buckets)

    async def search(self, query, limit=20, **kwargs):
        self.search_limits.append(limit)
        return list(self.buckets)[:limit]

    async def get(self, bucket_id):
        return next((b for b in self.buckets if b["id"] == bucket_id), None)

    async def touch(self, bucket_id):
        self.touched.append(bucket_id)


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
    assert "=== 记忆导航 ===" in nav
    assert "摘要:5.21朝灯定下海马体减噪三层方案" in nav
    assert "inspect:b1" in nav
    assert "标签:" not in nav

    full = await server.pulse(full=True)
    assert "=== 记忆列表 ===" in full
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
