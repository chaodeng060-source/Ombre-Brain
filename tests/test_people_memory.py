import asyncio
import copy

import pytest

from entity_store import EntityStore
from people_memory import build_people_cards, evidence_ranked_ids, wants_people_profile


class Buckets:
    def __init__(self):
        self.rows = {}

    async def get(self, bucket_id):
        return copy.deepcopy(self.rows.get(bucket_id))


@pytest.fixture
def people(tmp_path):
    store = EntityStore({"buckets_dir": str(tmp_path / "vault"), "entities": {"seeds": [
        {"canonical_name": "小墨", "type": "person", "aliases": ["墨总"]},
        {"canonical_name": "朝灯", "type": "person", "aliases": []},
        {"canonical_name": "Silas", "type": "person", "aliases": []},
    ]}})
    buckets = Buckets()

    def add(key, content, **metadata):
        buckets.rows[key] = {"id": key, "content": content, "metadata": {
            "type": "dynamic", "world": "", "event_at": "2026-10-22T00:00:00+00:00", **metadata,
        }}
        store.resolve_and_link(key, content)
    return store, buckets, add


def run(view):
    return asyncio.run(view)


def test_identity_alias_and_explicit_relation_keep_source_pointers(people):
    store, buckets, add = people
    add("a", "小墨是朝灯的朋友。", source_ref="raw_events:event:7")
    result = run(build_people_cards(store, buckets, query="墨总是谁"))
    assert result["memory_written"] is False
    assert result["people"][0]["name"] == "小墨"
    row = result["people"][0]["evidence"][0]
    assert row["quote"] == "小墨是朝灯的朋友。"
    assert row["source_kind"] == "memory_body_excerpt"
    assert row["source_refs"] == {"source_ref": "raw_events:event:7"}
    assert result["relations"][0]["object_name"] == "朝灯"
    assert result["relations"][0]["relation"] == "朋友"
    assert evidence_ranked_ids(result) == [("a", 1.0)]


@pytest.mark.parametrize("content", [
    "小墨和朝灯一起散步。", "小墨不是朝灯的朋友。", "如果小墨是朝灯的朋友。",
    "小墨以前是朝灯的朋友。", "小墨是朝灯的朋友？",
    "小墨是朝灯的朋友吗", "小墨是朝灯的朋友的同事。",
])
def test_co_mention_negation_hypothesis_history_and_questions_are_not_edges(people, content):
    store, buckets, add = people
    add("a", content)
    result = run(build_people_cards(store, buckets, query="小墨是什么关系"))
    assert result["relations"] == []


def test_current_profile_excludes_other_world_superseded_and_digested(people):
    store, buckets, add = people
    add("a", "小墨是朝灯的朋友。", world="故事")
    add("b", "小墨是朝灯的同事。", version_status="historical")
    add("c", "小墨是朝灯的同学。", digested=True)
    result = run(build_people_cards(store, buckets, query="小墨是谁"))
    assert result["people"] == []
    historic = run(build_people_cards(store, buckets, query="小墨是谁", include_history=True))
    assert [row["bucket_id"] for row in historic["people"][0]["evidence"]] == ["b"]
    story = run(build_people_cards(store, buckets, query="小墨是谁", world_filter=["故事"]))
    assert story["relations"][0]["world"] == "故事"


def test_changed_body_cannot_keep_stale_identity_links(people):
    store, buckets, add = people
    add("a", "小墨是朝灯的朋友。")
    buckets.rows["a"]["content"] = "Silas是朝灯的同学。"
    result = run(build_people_cards(store, buckets, query="小墨是谁"))
    assert result["people"] == []
    assert result["stale_links_skipped"] == 1


def test_identity_evidence_wins_over_newer_generic_mention(people):
    store, buckets, add = people
    add("a", "小墨是朝灯的朋友。", event_at="2026-01-01T00:00:00+00:00")
    add("b", "我和小墨今天又聊了几句。", event_at="2026-10-22T00:00:00+00:00")
    result = run(build_people_cards(store, buckets, query="小墨是谁", evidence_limit=1))
    assert evidence_ranked_ids(result) == [("a", 1.0)]
    assert result["people"][0]["source_bucket_count"] == 2


def test_alias_collision_does_not_choose_someone(people):
    store, buckets, add = people
    store.resolve_or_create("墨白", aliases=["墨总"], entity_type="person")
    add("a", "小墨是朝灯的朋友。")
    result = run(build_people_cards(store, buckets, query="墨总是谁"))
    assert result["people"] == []
    assert result["ambiguous_terms"] == ["墨总"]


def test_read_view_does_not_initialize_absent_store(tmp_path):
    root = tmp_path / "absent"
    store = EntityStore({"buckets_dir": str(root)}, initialize=False)
    result = run(build_people_cards(store, Buckets(), query="小墨是谁"))
    assert result["people"] == []
    assert not root.exists()


def test_explicit_empty_entity_selection_never_reads_everyones_sources(people):
    store, buckets, add = people
    add("a", "小墨是朝灯的朋友。")
    assert store.linked_sources([]) == {}
    assert store.linked_sources() == {"a": next(iter(store.linked_sources().values()))}


def test_profile_intent_does_not_run_for_casual_mentions():
    assert wants_people_profile("小墨现在是谁")
    assert wants_people_profile("小墨和朝灯是什么关系")
    assert not wants_people_profile("今天和小墨一起吃饭")
