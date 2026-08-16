from pathlib import Path

import pytest

from review_queue import KIND_CLOTHING, ReviewQueue


@pytest.mark.asyncio
async def test_create_appends_literal_retrieval_keys_without_queueing(bucket_mgr):
    original = "朝灯在杭州吃了潮汕菜，记住了这家店。"

    bucket_id = await bucket_mgr.create(
        original,
        name="杭州潮汕菜",
        domain=["生活"],
        retrieval_keys=["杭州", "潮汕菜", "不存在的词", "朝灯"],
    )

    bucket = await bucket_mgr.get(bucket_id)
    assert bucket is not None
    assert bucket["metadata"]["name"] != bucket_id
    assert bucket["metadata"].get("needs_clothing") is not True
    assert bucket["content"] == original + "\n\n[检索钥匙: 杭州 / 潮汕菜]"

    queue = ReviewQueue(Path(bucket_mgr.base_dir) / "review_queue.jsonl")
    assert queue.list_pending(KIND_CLOTHING) == []


@pytest.mark.asyncio
async def test_create_without_literal_key_preserves_body_and_enters_queue(bucket_mgr):
    original = "今天挺开心的。"

    bucket_id = await bucket_mgr.create(
        original,
        domain=["生活"],
        retrieval_keys=["模型凭空生成的词", "今天"],
        actor="test:missing-clothing",
    )

    bucket = await bucket_mgr.get(bucket_id)
    assert bucket is not None
    metadata = bucket["metadata"]
    assert metadata["name"] != bucket_id
    assert metadata["name"].startswith("待补衣")
    assert metadata["needs_clothing"] is True
    assert metadata["clothing_reason"] == "no_literal_retrieval_key"
    assert bucket["content"] == original

    queue = ReviewQueue(Path(bucket_mgr.base_dir) / "review_queue.jsonl")
    pending = queue.list_pending(KIND_CLOTHING)
    assert len(pending) == 1
    assert pending[0]["bucket_id"] == bucket_id
    assert pending[0]["bucket_name"] == metadata["name"]
    assert pending[0]["source"] == "test:missing-clothing"
    assert "content" not in pending[0]


@pytest.mark.asyncio
async def test_every_create_without_name_avoids_name_equal_id(bucket_mgr):
    bucket_id = await bucket_mgr.create("没有可靠实体词", name=None)

    bucket = await bucket_mgr.get(bucket_id)
    assert bucket is not None
    assert bucket["metadata"]["name"] != bucket_id
    assert Path(bucket["path"]).name != f"{bucket_id}.md"


@pytest.mark.asyncio
async def test_queue_failure_never_rejects_the_memory(bucket_mgr, monkeypatch):
    original = "没有可靠实体，也必须完整留下。"

    def fail_enqueue(_entry):
        raise OSError("queue unavailable")

    monkeypatch.setattr(bucket_mgr._clothing_review_queue, "enqueue", fail_enqueue)
    bucket_id = await bucket_mgr.create(original, actor="test:queue-failure")

    bucket = await bucket_mgr.get(bucket_id)
    assert bucket is not None
    assert bucket["content"] == original
    assert bucket["metadata"]["needs_clothing"] is True


@pytest.mark.asyncio
async def test_merge_or_create_returns_visible_placeholder_not_bucket_id(
    bucket_mgr,
    monkeypatch,
):
    import server

    class _Embedding:
        async def generate_and_store(self, _bucket_id, _content):
            return True

    async def _no_candidates(**_kwargs):
        return []

    async def _no_entity_sync(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", _Embedding())
    monkeypatch.setattr(server, "_find_merge_candidates", _no_candidates)
    monkeypatch.setattr(server, "_synchronize_bucket_entities", _no_entity_sync)

    bucket_id, display_name, is_merged = await server._merge_or_create(
        content="今天挺开心的。",
        tags=[],
        importance=5,
        domain=["生活"],
        valence=0.8,
        arousal=0.4,
    )

    assert is_merged is False
    assert display_name != bucket_id
    assert display_name.startswith("待补衣")
