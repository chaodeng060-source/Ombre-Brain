# ============================================================
# DS 精排 memo 缓存（2026-08-21 削 ds_filter 耗时）
#
# Contract under test:
#   1. 同一段 prompt 字节（query + 候选 name/snippet）= 同一个判断请求，
#      temperature=0 下 memo 命中等价于重放同一次判断——LLM 只打一次。
#   2. query / 候选内容任何一变即 miss，绝不串味。
#   3. 解析失败不进缓存（坏协议不能冒充好判断）。
#   4. keep / max_results 不进模型，只在本地并集/截断生效——同一份 memo
#      可以服务不同的 forced 集合，结果仍逐字节正确。
# ============================================================

import types

import pytest

import server


def _bucket(id: str, content: str) -> dict:
    return {"id": id, "content": content, "metadata": {}}


class _CountingClient:
    """按序吐 payload；列表只剩最后一个时无限重复。记录每次收到的 user prompt。"""

    def __init__(self, payloads):
        self.calls = 0
        self.payloads = list(payloads)
        self.user_prompts = []

    async def create(self, **kw):
        self.calls += 1
        self.user_prompts.append(kw["messages"][1]["content"])
        payload = (
            self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        )
        msg = types.SimpleNamespace(content=payload)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _fake_dehydrator(client):
    return types.SimpleNamespace(
        model="gemini-3.7-flash",
        client=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=client.create))
        ),
    )


@pytest.fixture(autouse=True)
def _clean_ds_cache():
    # getattr 容忍：反向验证跑在没打补丁的 server.py 上时没有这个符号，
    # 让测试照常跑、红在断言上（client 被调了 2 次），而不是红在 setup。
    getattr(server, "_DS_SELECT_CACHE", {}).clear()
    yield
    getattr(server, "_DS_SELECT_CACHE", {}).clear()


BUCKETS = None  # 每个测试自建，防串改


@pytest.mark.asyncio
async def test_identical_prompt_calls_llm_once(monkeypatch):
    client = _CountingClient(['{"keep": [0]}'])
    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator(client))
    buckets = [_bucket("a", "alpha"), _bucket("b", "beta")]

    r1 = await server._ds_semantic_select("查询甲", buckets, set(), 5)
    r2 = await server._ds_semantic_select("查询甲", buckets, set(), 5)

    assert client.calls == 1  # 第二次是 memo 重放，不是第二次判断
    assert [b["id"] for b in r1] == ["a"]
    assert [b["id"] for b in r2] == ["a"]


@pytest.mark.asyncio
async def test_different_query_misses_cache(monkeypatch):
    client = _CountingClient(['{"keep": [0, 1]}'])
    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator(client))
    buckets = [_bucket("a", "alpha"), _bucket("b", "beta")]

    await server._ds_semantic_select("查询甲", buckets, set(), 5)
    await server._ds_semantic_select("查询乙", buckets, set(), 5)

    assert client.calls == 2  # query 变了必须重判


@pytest.mark.asyncio
async def test_changed_bucket_content_misses_cache(monkeypatch):
    client = _CountingClient(['{"keep": [0, 1]}'])
    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator(client))
    buckets = [_bucket("a", "alpha"), _bucket("b", "beta")]

    await server._ds_semantic_select("查询甲", buckets, set(), 5)
    buckets[1]["content"] = "beta 内容被改写过"
    await server._ds_semantic_select("查询甲", buckets, set(), 5)

    assert client.calls == 2  # 候选内容变了必须重判


@pytest.mark.asyncio
async def test_invalid_payload_not_cached(monkeypatch):
    client = _CountingClient(["这不是json", '{"keep": [1]}'])
    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator(client))
    buckets = [_bucket("a", "alpha"), _bucket("b", "beta")]

    with pytest.raises(ValueError):
        await server._ds_semantic_select("查询甲", buckets, set(), 5)
    assert client.calls == 1

    result = await server._ds_semantic_select("查询甲", buckets, set(), 5)
    assert client.calls == 2  # 坏协议不许占坑，第二次真打 API
    assert [b["id"] for b in result] == ["b"]


@pytest.mark.asyncio
async def test_ttl_zero_disables_cache(monkeypatch):
    monkeypatch.setenv("OMBRE_DS_FILTER_CACHE_TTL", "0")
    client = _CountingClient(['{"keep": [0]}'])
    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator(client))
    buckets = [_bucket("a", "alpha"), _bucket("b", "beta")]

    await server._ds_semantic_select("查询甲", buckets, set(), 5)
    await server._ds_semantic_select("查询甲", buckets, set(), 5)

    assert client.calls == 2  # 开关关掉 = 回到现状，每打必真


@pytest.mark.asyncio
async def test_keep_and_max_results_apply_locally_over_memo(monkeypatch):
    client = _CountingClient(['{"keep": [0]}'])
    monkeypatch.setattr(server, "dehydrator", _fake_dehydrator(client))
    buckets = [_bucket("a", "alpha"), _bucket("b", "beta")]

    r1 = await server._ds_semantic_select("查询甲", buckets, set(), 5)
    # 同一 prompt、不同 forced 集合：memo 共享判断，本地并集各算各的
    r2 = await server._ds_semantic_select("查询甲", buckets, {"b"}, 5)

    assert client.calls == 1
    assert [b["id"] for b in r1] == ["a"]
    assert [b["id"] for b in r2] == ["a", "b"]  # forced 恒留，且顺序不变
