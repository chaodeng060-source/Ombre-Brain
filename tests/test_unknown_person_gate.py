from pathlib import Path

import pytest

import unknown_person_gate
from review_queue import (
    KIND_UNKNOWN_PERSON,
    ReviewQueue,
    make_unknown_person_entry,
)


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


def test_unknown_person_entry_is_bounded_idempotent_and_contains_no_body():
    content_hash = "a" * 64
    entry = make_unknown_person_entry(
        "bucket-1",
        "n" * 170,
        ["婷易", "婷易", *[f"人物{i}" for i in range(13)]],
        content_sha256=content_hash,
        source="worker:" + "x" * 130,
    )

    assert entry["key"] == "unknown_person|bucket-1|" + content_hash[:12]
    assert entry["kind"] == KIND_UNKNOWN_PERSON
    assert entry["status"] == "pending"
    assert len(entry["bucket_name"]) == 160
    assert len(entry["source"]) == 120
    assert len(entry["mentions"]) == 12
    assert entry["mentions"][0] == "婷易"
    assert len(set(entry["mentions"])) == len(entry["mentions"])
    assert "content" not in entry


def test_unknown_person_entry_rejects_missing_mentions_or_invalid_hash():
    with pytest.raises(ValueError, match="at least one mention"):
        make_unknown_person_entry(
            "bucket-1", "name", [], content_sha256="a" * 64
        )

    with pytest.raises(ValueError, match="sha256"):
        make_unknown_person_entry(
            "bucket-1", "name", ["婷易"], content_sha256="not-a-hash"
        )


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
