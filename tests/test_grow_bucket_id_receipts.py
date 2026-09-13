import pytest


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


async def _no_decay_background():
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_merged", "action"),
    [(False, "新建"), (True, "合并")],
)
async def test_short_grow_returns_bucket_id(monkeypatch, is_merged, action):
    import server

    async def _write(**_kwargs):
        return "short-bucket-id", "短记忆", is_merged

    monkeypatch.setattr(server, "dehydrator", _GrowDehydrator())
    monkeypatch.setattr(server, "_ensure_decay_background", _no_decay_background)
    monkeypatch.setattr(server, "_merge_or_create", _write)
    monkeypatch.setattr(server, "_mark_briefing_cache_dirty", lambda _reason: None)

    result = await server.grow("一条短记忆")

    assert result == f"{action} → 短记忆 | 工程 V0.5/A0.3 [short-bucket-id]"


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

    async def _write(**_kwargs):
        return "long-bucket-id", "长记忆", is_merged

    monkeypatch.setattr(server, "dehydrator", _GrowDehydrator())
    monkeypatch.setattr(server, "_ensure_decay_background", _no_decay_background)
    monkeypatch.setattr(server, "_merge_or_create", _write)
    monkeypatch.setattr(server, "_mark_briefing_cache_dirty", lambda _reason: None)

    result = await server.grow("甲" * 30)

    assert result == f"{summary}\n{receipt}"
