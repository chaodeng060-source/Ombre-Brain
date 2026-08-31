import json
import types
from pathlib import Path

import pytest

import bm25_index
import bucket_manager
import server


async def _inline_to_thread(function, *args, **kwargs):
    return function(*args, **kwargs)


@pytest.fixture(autouse=True)
def _run_worker_calls_inline(monkeypatch):
    monkeypatch.setattr(bucket_manager.asyncio, "to_thread", _inline_to_thread)


class _FakeBM25:
    def __init__(self, rows: dict[str, dict], scores: dict[str, float]):
        self._index = object()
        self._keyword_score_rows = rows
        self._scores = scores

    def score(self, _query: str) -> dict[str, float]:
        return dict(self._scores)


def _bucket(bucket_id: str, content: str, *, name: str, keys=()) -> dict:
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "name": name,
            "tags": [],
            "domain": ["工程"],
            "retrieval_keys": list(keys),
            "importance": 5,
        },
    }


def _install_live_bm25(manager, buckets: list[dict], scores: dict[str, float]):
    rows = {
        bucket["id"]: manager._build_keyword_score_row(bucket)
        for bucket in buckets
    }
    manager._bm25_mode = "live"
    manager._bm25 = _FakeBM25(rows, scores)
    manager._bm25_dirty = False
    manager._bm25_rebuilding = False
    manager._bm25_unknown_dirty = False
    manager._bm25_dirty_bucket_ids.clear()


@pytest.mark.asyncio
async def test_live_bm25_relevance_uses_max_win(bucket_mgr, monkeypatch):
    bucket = _bucket("max-win", "普通正文", name="普通标题")
    _install_live_bm25(bucket_mgr, [bucket], {"max-win": 0.8})
    monkeypatch.setattr(
        bucket_mgr,
        "_calc_topic_score_from_row",
        lambda *_args, **_kwargs: 0.4,
    )

    result = await bucket_mgr.search(
        "罕见锚点",
        limit=1,
        relevance_first=True,
        relevance_candidate_floor=0,
        preloaded_buckets=[bucket],
    )

    assert result[0]["score"] == 80.0


@pytest.mark.asyncio
async def test_literal_retrieval_key_force_wins_over_bm25(bucket_mgr):
    key_bucket = _bucket(
        "literal-key",
        "远程海马体迁移记录",
        name="迁移记录",
        keys=("远程海马体",),
    )
    bm25_bucket = _bucket("bm25-winner", "普通检索记录", name="普通记录")
    buckets = [key_bucket, bm25_bucket]
    _install_live_bm25(
        bucket_mgr,
        buckets,
        {"literal-key": 0.1, "bm25-winner": 0.8},
    )

    result = await bucket_mgr.search(
        "后来远程海马体怎么样了",
        limit=2,
        relevance_first=True,
        relevance_candidate_floor=0,
        preloaded_buckets=buckets,
    )

    assert result[0]["id"] == "literal-key"
    assert result[0]["score"] == 100.0
    assert result[1]["score"] == 80.0


@pytest.mark.asyncio
async def test_z_historical_overlay_penalizes_main_score_but_keeps_candidate(
    bucket_mgr,
    monkeypatch,
):
    historical = _bucket("historical", "旧事实", name="旧事实")
    current = _bucket("current", "新事实", name="新事实")
    Path(bucket_mgr._z_overrides_path).write_text(
        json.dumps({
            "status": "active",
            "historical_bucket_id": "historical",
        }, ensure_ascii=False) + "\nnot-json\n",
        encoding="utf-8",
    )
    bucket_mgr._bm25_mode = "off"
    monkeypatch.setattr(bucket_mgr, "_calc_topic_score", lambda *_args: 1.0)

    result = await bucket_mgr.search(
        "同一件事",
        limit=2,
        relevance_first=True,
        relevance_candidate_floor=0,
        preloaded_buckets=[historical, current],
    )

    assert [bucket["id"] for bucket in result] == ["current", "historical"]
    assert result[0]["score"] == 100.0
    assert result[1]["score"] == 30.0


def test_bm25_tokenizer_drops_punctuation_fillers_and_single_latin():
    tokens = bm25_index._tokenize("O.o 哈哈哈 还有 海马体迁移")

    assert not {"o", ".", "哈哈", "哈哈哈", "还有"} & set(tokens)
    assert any(token in tokens for token in ("海马", "海马体", "迁移"))


@pytest.mark.asyncio
async def test_apiroute_filter_prompt_explicitly_allows_empty_for_fragments(
    monkeypatch,
):
    captured = {}

    async def _create(**kwargs):
        captured.update(kwargs)
        message = types.SimpleNamespace(content=json.dumps({"keep": []}))
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)
        )
    )
    monkeypatch.setattr(
        server,
        "_ds_filter_provider",
        lambda: ("apiroute-gemini", "fake-model", client, {}),
    )
    server._DS_SELECT_CACHE.clear()

    result = await server._ds_semantic_select(
        "还有…",
        [_bucket("candidate", "旧记忆正文", name="旧记忆")],
        set(),
        5,
    )

    prompt = captured["messages"][0]["content"]
    assert result == []
    assert '返回 {"keep": []}' in prompt
    assert "查询本身没有实质检索意图" in prompt


@pytest.mark.asyncio
async def test_invalid_filter_payload_logs_bounded_raw_head(
    monkeypatch,
    caplog,
):
    raw = "provider-refusal:" + "x" * 700

    async def _create(**_kwargs):
        message = types.SimpleNamespace(content=raw)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)
        )
    )
    monkeypatch.setattr(
        server,
        "_ds_filter_provider",
        lambda: ("apiroute-gemini", "fake-model", client, {}),
    )
    server._DS_SELECT_CACHE.clear()

    with caplog.at_level("ERROR", logger="ombre_brain"):
        with pytest.raises(ValueError):
            await server._ds_semantic_select(
                "具体查询",
                [_bucket("candidate", "旧记忆正文", name="旧记忆")],
                set(),
                5,
            )

    assert "raw_head='provider-refusal:" in caplog.text
    assert "x" * 501 not in caplog.text
