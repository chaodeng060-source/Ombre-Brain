"""裸桶闸 + 钥匙落点。

2026-08-17 上午立的闸：候选钥匙全不在正文里 → 名字「待补衣」、needs_clothing、进审队列、正文原样。
2026-08-17 下午加的兜底：候选全没有时退回正文首句当钥匙/名字（她「这是啥名字啊奇奇怪怪的」）。
2026-08-18 定型：钥匙**只进 metadata.retrieval_keys，不写回正文**——正文即原文，别的模块
（E 轴一字不改、curated receipt、乐观锁 hash、feel 解析）都靠这条契约。
"""
from pathlib import Path

import pytest

from review_queue import KIND_CLOTHING, ReviewQueue

# 首句也切不出钥匙的正文：全是标点/数字会被 literal_retrieval_keys 的 fullmatch 拒掉。
_UNKEYABLE = "……。。。2026-08-17"


@pytest.mark.asyncio
async def test_create_keeps_literal_retrieval_keys_in_metadata_only(bucket_mgr):
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
    # 「不存在的词」不在正文里被筛掉；「朝灯」是通用词被筛掉。
    assert bucket["metadata"]["retrieval_keys"] == ["杭州", "潮汕菜"]
    # 正文一字不动，钥匙不写回正文。
    assert bucket["content"] == original

    queue = ReviewQueue(Path(bucket_mgr.base_dir) / "review_queue.jsonl")
    assert queue.list_pending(KIND_CLOTHING) == []


@pytest.mark.asyncio
async def test_create_falls_back_to_body_lead_sentence_before_placeholder(bucket_mgr):
    """候选钥匙全废时，先拿正文首句当名字/钥匙，不再直接叫「待补衣」。"""
    original = "今天挺开心的。"

    bucket_id = await bucket_mgr.create(
        original,
        domain=["生活"],
        retrieval_keys=["模型凭空生成的词", "今天"],
        actor="test:body-lead",
    )

    bucket = await bucket_mgr.get(bucket_id)
    assert bucket is not None
    metadata = bucket["metadata"]
    assert metadata["name"] != bucket_id
    assert metadata["name"].startswith("今天挺开心的")  # 后面会带当天日期后缀
    assert metadata["retrieval_keys"] == ["今天挺开心的"]
    assert metadata.get("needs_clothing") is not True
    assert bucket["content"] == original

    queue = ReviewQueue(Path(bucket_mgr.base_dir) / "review_queue.jsonl")
    assert queue.list_pending(KIND_CLOTHING) == []


@pytest.mark.asyncio
async def test_create_without_any_key_preserves_body_and_enters_queue(bucket_mgr):
    original = _UNKEYABLE

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
    assert "retrieval_keys" not in metadata
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
    original = _UNKEYABLE

    def fail_enqueue(_entry):
        raise OSError("queue unavailable")

    monkeypatch.setattr(bucket_mgr._clothing_review_queue, "enqueue", fail_enqueue)
    bucket_id = await bucket_mgr.create(original, actor="test:queue-failure")

    bucket = await bucket_mgr.get(bucket_id)
    assert bucket is not None
    assert bucket["content"] == original
    assert bucket["metadata"]["needs_clothing"] is True


@pytest.mark.asyncio
async def test_merge_or_create_returns_visible_name_not_bucket_id(
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
    # 正文首句兜底：她看到的名字是这句话本身，不是 id 也不是「待补衣」。
    assert display_name.startswith("今天挺开心的")

    bucket_id2, display_name2, _ = await server._merge_or_create(
        content=_UNKEYABLE,
        tags=[],
        importance=5,
        domain=["生活"],
        valence=0.8,
        arousal=0.4,
    )
    assert display_name2 != bucket_id2
    assert display_name2.startswith("待补衣")
