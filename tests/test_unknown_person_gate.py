from pathlib import Path

import pytest

import unknown_person_gate
from review_queue import KIND_UNKNOWN_PERSON, ReviewQueue


def test_roster_from_config_collects_seed_names_aliases_and_known_links():
    roster = unknown_person_gate.roster_from_config({
        "seeds": [
            {"canonical_name": "朝灯", "aliases": ["灯灯", "朝朝"]},
            {"canonical": "哥哥", "aliases": "Claude"},
        ],
        "known_links": ["小卷", " 哈基米 "],
    })

    assert roster == frozenset({"朝灯", "灯灯", "朝朝", "哥哥", "Claude", "小卷", "哈基米"})


def test_unknown_person_mentions_only_returns_new_short_chinese_wikilinks():
    content = (
        "[[朝灯]]提到[[婷易]]，又写了[[婷易|同一个人]]、[[她]]、"
        "[[mcp_tools]]和[[记忆系统大修收官]]。"
    )

    assert unknown_person_gate.unknown_person_mentions(content, {"朝灯"}) == ("婷易",)


def test_empty_roster_keeps_gate_disabled(monkeypatch):
    monkeypatch.setattr(unknown_person_gate, "_ROSTER_CACHE", frozenset())

    assert unknown_person_gate.mentions_needing_review("提到[[婷易]]。") == ()


@pytest.mark.asyncio
async def test_bucket_create_keeps_body_and_enqueues_content_free_unknown_person_review(
    bucket_mgr, monkeypatch
):
    monkeypatch.setattr(unknown_person_gate, "_ROSTER_CACHE", frozenset({"朝灯"}))
    content = "朝灯在展会上遇到[[婷易]]。"

    bucket_id = await bucket_mgr.create(
        content,
        name="展会",
        domain=["生活"],
        actor="test:unknown-person",
    )

    bucket = await bucket_mgr.get(bucket_id)
    assert bucket is not None
    assert bucket["content"] == content

    queue = ReviewQueue(Path(bucket_mgr.base_dir) / "review_queue.jsonl")
    pending = queue.list_pending(KIND_UNKNOWN_PERSON)
    assert len(pending) == 1
    assert pending[0]["bucket_id"] == bucket_id
    assert pending[0]["mentions"] == ["婷易"]
    assert pending[0]["source"] == "test:unknown-person"
    assert "content" not in pending[0]
    assert content not in str(pending[0])


@pytest.mark.asyncio
async def test_unknown_person_queue_failure_does_not_lose_bucket(bucket_mgr, monkeypatch):
    monkeypatch.setattr(unknown_person_gate, "_ROSTER_CACHE", frozenset({"朝灯"}))

    def fail_enqueue(_entry):
        raise OSError("queue unavailable")

    monkeypatch.setattr(bucket_mgr._clothing_review_queue, "enqueue", fail_enqueue)
    content = "朝灯在展会上遇到[[婷易]]。"

    bucket_id = await bucket_mgr.create(
        content,
        name="展会",
        domain=["生活"],
        actor="test:unknown-person-queue-failure",
    )

    bucket = await bucket_mgr.get(bucket_id)
    assert bucket is not None
    assert bucket["content"] == content
