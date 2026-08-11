from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_primary_authored_e_fields_are_complete_and_immutable(bucket_mgr):
    bucket_id = await bucket_mgr.create(
        content="我记住：朝灯难过时，先安静接住她，再谈事情。",
        name="主AI体验",
        tags=["relationship_moment"],
        domain=["关系"],
        e_authored_by="claude",
        e_initial_priority=92,
        e_valence=-0.7,
        e_arousal=0.35,
        e_tension=0.55,
        e_confidence=1.0,
        e_response_tendency="comfort",
        e_growth_delta="stable",
        e_source_bucket_id="source-1",
        e_proposal_key="e|proposal-1",
    )

    bucket = await bucket_mgr.get(bucket_id)
    metadata = bucket["metadata"]
    assert metadata["e_authored_by"] == "claude"
    assert metadata["e_initial_priority"] == 92
    assert metadata["e_response_tendency"] == "comfort"
    assert metadata["e_source_bucket_id"] == "source-1"
    assert metadata["e_proposal_key"] == "e|proposal-1"
    assert metadata["e_authored_at"]

    assert await bucket_mgr.update(bucket_id, e_initial_priority=5) is False
    assert await bucket_mgr.update(
        bucket_id,
        content="模型替换掉主 AI 的原话",
    ) is False

    unchanged = await bucket_mgr.get(bucket_id)
    assert unchanged["content"] == bucket["content"]
    assert unchanged["metadata"]["e_initial_priority"] == 92


@pytest.mark.asyncio
async def test_e_fields_reject_missing_primary_authorship(bucket_mgr):
    with pytest.raises(ValueError, match="e_authored_by"):
        await bucket_mgr.create(
            content="没有主 AI 作者的 E 不得落权威记录。",
            tags=["relationship_moment"],
            e_initial_priority=80,
            e_valence=0.1,
            e_arousal=0.4,
            e_tension=0.2,
            e_confidence=1.0,
            e_response_tendency="engage",
            e_growth_delta="stable",
        )


@pytest.mark.asyncio
async def test_experience_tool_writes_exact_primary_authored_record(
    bucket_mgr,
    monkeypatch,
):
    import server

    class _Embedding:
        def __init__(self):
            self.calls = []

        async def generate_and_store(self, bucket_id, content):
            self.calls.append((bucket_id, content))
            return True

    embedding = _Embedding()
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", embedding)
    monkeypatch.setattr(server, "config", {"current_world": ""})
    monkeypatch.setattr(server, "_mark_briefing_cache_dirty", lambda _reason: None)

    content = "我亲自判断：朝灯在质疑结论时，需要我先回到证据，不沿用口头账。"
    result = await server.experience(
        content=content,
        e_authored_by="xiaojuan",
        e_initial_priority=88,
        e_valence=-0.25,
        e_arousal=0.45,
        e_tension=0.5,
        e_response_tendency="engage",
        e_growth_delta="growth",
    )

    assert result.startswith("E→")
    bucket_id = result.split("→", 1)[1].split(" ", 1)[0]
    bucket = await bucket_mgr.get(bucket_id)
    assert bucket["content"] == content
    assert bucket["metadata"]["e_authored_by"] == "xiaojuan"
    assert bucket["metadata"]["e_initial_priority"] == 88
    assert embedding.calls == [(bucket_id, content)]


@pytest.mark.asyncio
async def test_experience_tool_replays_proposal_idempotently(
    bucket_mgr,
    test_config,
    monkeypatch,
):
    import server
    from review_queue import ReviewQueue, make_e_proposal_entry

    class _Embedding:
        async def generate_and_store(self, _bucket_id, _content):
            return True

    source_id = await bucket_mgr.create(
        content="朝灯要求结论必须回到生产证据。",
        tags=["relationship_moment"],
        name="来源",
    )
    queue = ReviewQueue(
        test_config["buckets_dir"] + "/review_queue.jsonl",
        maintenance_root=test_config["buckets_dir"],
    )
    proposal = make_e_proposal_entry(
        source_id,
        "relationship_moment",
        "证据优先",
        "一次机器建议，只能供主 AI 自己判断。",
        suggested_priority=80,
    )
    assert queue.enqueue(proposal)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(server, "embedding_engine", _Embedding())
    monkeypatch.setattr(server, "config", test_config)
    monkeypatch.setattr(server, "_review_queue", queue)
    monkeypatch.setattr(server, "_mark_briefing_cache_dirty", lambda _reason: None)
    kwargs = {
        "content": "我会在结论动摇时重新查生产证据，不拿旧口头账顶替。",
        "e_authored_by": "xiaojuan",
        "e_initial_priority": 90,
        "e_valence": -0.1,
        "e_arousal": 0.4,
        "e_tension": 0.35,
        "e_response_tendency": "engage",
        "e_growth_delta": "growth",
        "proposal_key": proposal["key"],
    }

    first = await server.experience(**kwargs)
    second = await server.experience(**kwargs)

    assert "idempotent:existing" in second
    first_id = first.split("→", 1)[1].split(" ", 1)[0]
    second_id = second.split("→", 1)[1].split(" ", 1)[0]
    assert first_id == second_id
    authored = [
        bucket
        for bucket in await bucket_mgr.list_all(include_archive=True)
        if (bucket.get("metadata") or {}).get("e_proposal_key") == proposal["key"]
    ]
    assert len(authored) == 1
    assert queue.get(proposal["key"])["status"] == "reviewed"
