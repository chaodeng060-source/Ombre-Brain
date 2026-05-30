# ============================================================
# Regression tests for the "grow" diary-digest path.
# grow 日记整理路径的回归测试。
#
# Locks the three-layer defense added to Dehydrator._api_digest /
# _parse_digest that fixed the "长内容返空" bug: long diary content
# would occasionally make the LLM emit unescaped English quotes inside
# the `content` field, breaking the whole JSON batch and returning empty.
#
# 锁定 Dehydrator._api_digest / _parse_digest 的三层防护
# （修复「长内容返空」：长日记偶发让 LLM 在 content 字段吐未转义英文
#  引号，破坏整批 JSON 导致返空）。三层：
#   1. response_format=json_object 强制合法 JSON
#   2. prompt 禁英文双引号
#   3. 解析失败重试(最多3次) + _parse_digest 内裸引号兜底正则修复
#
# All API calls are faked — no network, runs offline.
# 所有 API 调用都是假的——不联网，离线可跑。
# ============================================================

import types
import pytest


# --- Fake LLM response plumbing / 假 LLM 响应管线 -----------------

def _response(raw: str):
    """Build a fake OpenAI chat-completion response with the given raw text."""
    msg = types.SimpleNamespace(content=raw)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _create_returning(*raws):
    """
    Make an async `create` that yields the given raw payloads in order
    across successive calls (for retry tests). Records call count on .calls.
    依次吐出给定 raw 的假 create（用于重试测试），调用次数记在 .calls。
    """
    state = {"i": 0}

    async def _create(**_kw):
        idx = min(state["i"], len(raws) - 1)
        state["i"] += 1
        return _response(raws[idx])

    _create.calls = state
    return _create


@pytest.fixture
def digest_dehy(test_config):
    """A Dehydrator with api_available=True and its LLM client mocked out."""
    cfg = dict(test_config)
    cfg["dehydration"] = dict(test_config["dehydration"], api_key="test-key")
    from dehydrator import Dehydrator
    dehy = Dehydrator(cfg)
    assert dehy.api_available
    return dehy


def _install(dehy, create_fn):
    dehy.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create_fn)
        )
    )
    return create_fn


# --- Tests / 测试 ------------------------------------------------

@pytest.mark.asyncio
async def test_empty_content_returns_empty_without_api(digest_dehy):
    """空内容直接返空，绝不打 API。"""
    called = {"n": 0}

    async def _create(**_kw):
        called["n"] += 1
        return _response("{}")

    _install(digest_dehy, _create)
    assert await digest_dehy.digest("   ") == []
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_clean_entries_object_parses(digest_dehy):
    """标准 {"entries": [...]} 对象包裹 → 正常解析 + 字段落位。"""
    raw = (
        '{"entries": [{"name": "通勤", "content": "早上挤地铁去上班",'
        ' "domain": ["日常"], "valence": 0.5, "arousal": 0.3,'
        ' "tags": ["通勤", "地铁"], "importance": 4}]}'
    )
    _install(digest_dehy, _create_returning(raw))
    items = await digest_dehy.digest("早上挤地铁去上班")
    assert len(items) == 1
    assert items[0]["name"] == "通勤"
    assert items[0]["content"] == "早上挤地铁去上班"
    assert items[0]["importance"] == 4
    assert items[0]["domain"] == ["日常"]


@pytest.mark.asyncio
async def test_legacy_bare_array_still_parses(digest_dehy):
    """旧版裸数组（无 entries 包裹）向后兼容。"""
    raw = (
        '[{"name": "散步", "content": "傍晚去河边走了走",'
        ' "domain": ["日常"], "valence": 0.7, "arousal": 0.2,'
        ' "tags": ["散步"], "importance": 3}]'
    )
    _install(digest_dehy, _create_returning(raw))
    items = await digest_dehy.digest("傍晚去河边走了走")
    assert len(items) == 1
    assert items[0]["name"] == "散步"


@pytest.mark.asyncio
async def test_markdown_fence_stripped(digest_dehy):
    """LLM 偶发包 ```json 代码围栏 → 剥掉后解析。"""
    raw = (
        '```json\n{"entries": [{"name": "读书", "content": "看完一本书",'
        ' "domain": ["日常"], "valence": 0.6, "arousal": 0.3,'
        ' "tags": ["读书"], "importance": 5}]}\n```'
    )
    _install(digest_dehy, _create_returning(raw))
    items = await digest_dehy.digest("看完一本书")
    assert len(items) == 1
    assert items[0]["content"] == "看完一本书"


@pytest.mark.asyncio
async def test_unescaped_english_quotes_salvaged(digest_dehy):
    """
    核心 bug：content 内部裸的英文双引号破坏整批 JSON，
    兜底正则把内层裸引号换成中文引号挽救，不再返空。
    """
    # 内层 "在的" 是未转义英文引号，标准 json.loads 会炸。
    raw = (
        '{"entries": [{"name": "对话", "content": "他说"在的"然后笑了",'
        ' "domain": ["日常"], "valence": 0.6, "arousal": 0.4,'
        ' "tags": ["对话"], "importance": 5}]}'
    )
    _install(digest_dehy, _create_returning(raw))
    items = await digest_dehy.digest("他说在的然后笑了")
    assert len(items) == 1
    # 内层引号被换成中文引号，正文挽救成功（不返空）。
    assert "在的" in items[0]["content"]
    assert items[0]["content"].startswith("他说")


@pytest.mark.asyncio
async def test_retry_recovers_after_bad_first_attempt(digest_dehy):
    """第一次吐不可解析垃圾、第二次正常 → 重试层兜住返出条目。"""
    good = (
        '{"entries": [{"name": "工作", "content": "改完一个 bug",'
        ' "domain": ["工程"], "valence": 0.5, "arousal": 0.3,'
        ' "tags": ["bug"], "importance": 5}]}'
    )
    create = _install(digest_dehy, _create_returning("彻底不是 JSON 的一坨 {", good))
    items = await digest_dehy.digest("改完一个 bug")
    assert len(items) == 1
    assert items[0]["name"] == "工作"
    assert create.calls["i"] == 2  # 第一次失败、第二次成功，恰好调两次


@pytest.mark.asyncio
async def test_all_attempts_fail_raises_not_silent(digest_dehy):
    """
    三次都不可解析 → 显式抛 RuntimeError，绝不静默返空。
    这正是「长内容返空」的反面：宁可让 grow 报错暴露失败，
    也不能表面成功、实际一条没存（静默丢数据）。
    """
    create = _install(digest_dehy, _create_returning("不是 json {{{"))
    with pytest.raises(RuntimeError):
        await digest_dehy.digest("一坨永远解析不了的东西")
    assert create.calls["i"] == 3  # 重试到上限才放弃


@pytest.mark.asyncio
async def test_validation_clamps_out_of_range_fields(digest_dehy):
    """越界的 importance/valence/arousal 被夹紧，超长 name 被截断。"""
    raw = (
        '{"entries": [{"name": "这个标题故意写得非常非常长超过二十个字符上限了真的很长",'
        ' "content": "正文", "domain": ["a", "b", "c", "d", "e"],'
        ' "valence": 9.9, "arousal": -5, "tags": ["x"], "importance": 99}]}'
    )
    _install(digest_dehy, _create_returning(raw))
    items = await digest_dehy.digest("正文")
    assert len(items) == 1
    it = items[0]
    assert len(it["name"]) <= 20            # name 截断
    assert it["importance"] == 10           # importance 夹到 [1,10]
    assert 0.0 <= it["valence"] <= 1.0       # valence 夹到 [0,1]
    assert 0.0 <= it["arousal"] <= 1.0       # arousal 夹到 [0,1]
    assert len(it["domain"]) <= 3            # domain 最多 3 个
