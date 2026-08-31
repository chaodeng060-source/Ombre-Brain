import asyncio
from argparse import Namespace
from pathlib import Path

import pytest

import relation_backfill
from bucket_manager import BucketManager
from curated_writer import CuratedWriteCoordinator
from relation_backfill import build_cleanup_plan, build_relation_patches
from relation_graph import (
    EXPLICIT_GENERATION_METHOD,
    LEGACY_GENERATION_METHOD,
    PROVENANCE_GENERATION_METHOD,
    TIMELINE_GENERATION_METHOD,
    plan_relation_graph,
)


def _bucket(bucket_id, **metadata):
    return {
        "id": bucket_id,
        "metadata": {
            "id": bucket_id,
            "event_at": metadata.pop("event_at", f"2026-08-01T00:00:0{bucket_id[-1]}"),
            "relations": metadata.pop("relations", []),
            **metadata,
        },
        "content": f"body-{bucket_id}",
        "path": f"/vault/{bucket_id}.md",
    }


def _all_relations(manager, bucket_ids):
    async def load():
        rows = [await manager.get(bucket_id) for bucket_id in bucket_ids]
        return [
            (row["id"], relation)
            for row in rows
            for relation in row["metadata"].get("relations") or []
        ]

    return load()


async def _close_manager_tasks(manager):
    tasks = []
    rebuild = getattr(manager, "_bm25_rebuild_task", None)
    if rebuild is not None and not rebuild.done():
        tasks.append(rebuild)
    tasks.extend(
        task
        for task in getattr(manager, "_recall_snapshot_refresh_tasks", {}).values()
        if not task.done()
    )
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


class _ToggleEmbedding:
    def __init__(self):
        self.succeeds = False
        self.calls = []
        self.stored = set()

    async def generate_and_store(self, bucket_id, content):
        self.calls.append((bucket_id, content))
        if self.succeeds:
            self.stored.add(bucket_id)
            return True
        return False

    async def get_embedding(self, bucket_id):
        return [1.0] if bucket_id in self.stored else None

    def delete_embedding(self, bucket_id):
        self.stored.discard(bucket_id)


def test_full_plan_uses_only_explicit_and_structural_evidence():
    buckets = [
        _bucket(
            "source1",
            source_session="session-a",
            source_event_ids=["event-a"],
            source_digest="a" * 64,
        ),
        _bucket(
            "derived1",
            source_session="session-a",
            source_event_ids=["event-a"],
            source_digest="a" * 64,
        ),
        _bucket("episode1", source_buckets=["source1"]),
        _bucket("emotion1", e_source_bucket_id="source1"),
        _bucket("thread01", thread="reviewed-line"),
        _bucket("thread02", thread="reviewed-line"),
        _bucket("unrelated", thread="other"),
    ]

    plan = plan_relation_graph(buckets)
    identities = {
        (item.source_id, item.relation_type, item.target_id)
        for item in plan.relations
    }

    assert plan.input_count == 7
    assert plan.eligible_count == 6
    assert plan.unsupported_count == 1
    assert plan.skipped_by_reason == {"no_deterministic_relation_evidence": 1}
    assert ("emotion1", "explains", "source1") in identities
    assert ("episode1", "explains", "source1") in identities
    assert plan.relation_type_counts == {"explains": 2, "kin": 2}
    assert plan.generation_method_counts == {
        EXPLICIT_GENERATION_METHOD: 2,
        PROVENANCE_GENERATION_METHOD: 1,
        TIMELINE_GENERATION_METHOD: 1,
    }
    assert all(item.evidence["bases"] for item in plan.relations)
    assert not any("unrelated" in identity for identity in identities)


def test_provenance_graph_is_a_deterministic_sparse_forest():
    buckets = [
        _bucket(
            f"event00{index}",
            event_at=f"2026-08-0{index}T00:00:00",
            source_session="session-a",
            source_event_ids=["event-a"],
            source_digest="b" * 64,
        )
        for index in range(1, 5)
    ]

    first = plan_relation_graph(buckets)
    second = plan_relation_graph(list(reversed(buckets)))

    assert len(first.relations) == 3
    assert first.relations == second.relations
    assert all(item.relation_type == "kin" for item in first.relations)
    assert all(item.generation_method == PROVENANCE_GENERATION_METHOD for item in first.relations)
    assert len({(item.source_id, item.target_id) for item in first.relations}) == 3


def test_missing_reference_is_reported_instead_of_fabricated():
    plan = plan_relation_graph([
        _bucket("orphan01", source_buckets=["missing"]),
        _bucket("plain001"),
    ])

    assert plan.relations == ()
    assert plan.eligible_count == 0
    assert plan.skipped_by_reason == {
        "missing_explicit_target": 1,
        "no_deterministic_relation_evidence": 1,
    }


def test_formal_create_auto_links_once_without_touching_old_activity(bucket_mgr):
    async def scenario():
        provenance = {
            "source_kind": "conversation",
            "source_session": "session-a",
            "source_event_ids": ["event-a"],
            "source_digest": "c" * 64,
        }
        first = await bucket_mgr.create(
            content="first",
            name="first",
            x_provenance=provenance,
        )
        assert await bucket_mgr.update(
            first,
            last_active="2000-01-01T00:00:00",
        )
        first_activity = (await bucket_mgr.get(first))["metadata"]["last_active"]
        second = await bucket_mgr.create(
            content="second",
            name="second",
            x_provenance=provenance,
        )

        relations = await _all_relations(bucket_mgr, [first, second])
        assert len(relations) == 1
        source_id, relation = relations[0]
        assert {source_id, relation["target"]} == {first, second}
        assert relation["type"] == "kin"
        assert relation["generation_method"] == PROVENANCE_GENERATION_METHOD
        assert relation["evidence"]["bases"]
        assert (await bucket_mgr.get(first))["metadata"]["last_active"] == first_activity

        repeated = await bucket_mgr.auto_link_created_bucket(second)
        assert repeated["created"] == 0
        assert len(await _all_relations(bucket_mgr, [first, second])) == 1

        unrelated = await bucket_mgr.create(content="third", name="third")
        assert len(await _all_relations(bucket_mgr, [first, second, unrelated])) == 1
        await _close_manager_tasks(bucket_mgr)

    asyncio.run(scenario())


def test_e_axis_create_links_to_its_explicit_source(bucket_mgr):
    async def scenario():
        source = await bucket_mgr.create(content="source", name="source")
        emotion = await bucket_mgr.create(
            content="emotion",
            name="emotion",
            bucket_type="feel",
            e_authored_by="test",
            e_initial_priority=50,
            e_valence=0.2,
            e_arousal=0.8,
            e_tension=0.6,
            e_confidence=0.9,
            e_response_tendency="engage",
            e_growth_delta="stable",
            e_source_bucket_id=source,
        )

        relation = (await bucket_mgr.get(emotion))["metadata"]["relations"][0]
        assert relation["target"] == source
        assert relation["type"] == "explains"
        assert relation["generation_method"] == EXPLICIT_GENERATION_METHOD
        await _close_manager_tasks(bucket_mgr)

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_curated_stage_links_only_after_successful_promotion(bucket_mgr):
    provenance = {
        "source_kind": "conversation",
        "source_session": "session-a",
        "source_event_ids": ["event-a"],
        "source_digest": "d" * 64,
    }
    source = await bucket_mgr.create(
        content="source",
        name="source",
        x_provenance=provenance,
    )
    embedding = _ToggleEmbedding()
    writer = CuratedWriteCoordinator(bucket_mgr, embedding)
    payload = {
        "idempotency_key": "y-stage-promotion",
        "content": "curated",
        "vector_policy": "required",
        "bucket_options": {
            "name": "curated",
            "x_provenance": provenance,
        },
    }

    failed = await writer.write(**payload)
    assert failed.status == "retryable"
    assert failed.bucket_id
    assert (await bucket_mgr.get(failed.bucket_id))["metadata"]["type"] == "archived"
    assert await _all_relations(bucket_mgr, [source, failed.bucket_id]) == []

    embedding.succeeds = True
    promoted = await writer.write(**payload)
    assert promoted.success is True
    relations = await _all_relations(bucket_mgr, [source, promoted.bucket_id])
    assert len(relations) == 1
    assert relations[0][1]["type"] == "kin"

    replayed = await writer.write(**payload)
    assert replayed == promoted
    assert len(await _all_relations(bucket_mgr, [source, promoted.bucket_id])) == 1
    await _close_manager_tasks(bucket_mgr)


def test_batch_upsert_enriches_legacy_edge_and_is_idempotent(bucket_mgr):
    async def scenario():
        source = await bucket_mgr.create(content="source", name="source")
        target = await bucket_mgr.create(content="target", name="target")
        assert await bucket_mgr.add_relation(source, target, "kin", note="legacy note")
        activity = (await bucket_mgr.get(source))["metadata"]["last_active"]
        edge = {
            "type": "kin",
            "target": target,
            "note": "new note must not replace the old one",
            "strength": 1.0,
            "generation_method": LEGACY_GENERATION_METHOD,
            "evidence": {
                "kind": "legacy_unattributed",
                "reason": "test",
            },
        }

        first = await bucket_mgr.upsert_relations(source, [edge], actor="test")
        second = await bucket_mgr.upsert_relations(source, [edge], actor="test")
        stored = (await bucket_mgr.get(source))["metadata"]["relations"][0]

        assert first["enriched"] == 1
        assert second["unchanged"] == 1
        assert stored["note"] == "legacy note"
        assert stored["strength"] == 1.0
        assert stored["generation_method"] == LEGACY_GENERATION_METHOD
        assert stored["evidence"]["kind"] == "legacy_unattributed"
        assert (await bucket_mgr.get(source))["metadata"]["last_active"] == activity
        await _close_manager_tasks(bucket_mgr)

    asyncio.run(scenario())


def test_batch_upsert_does_not_create_reverse_kin_duplicate(bucket_mgr):
    async def scenario():
        left = await bucket_mgr.create(content="left", name="left")
        right = await bucket_mgr.create(content="right", name="right")
        assert await bucket_mgr.add_relation(right, left, "kin", note="already there")

        result = await bucket_mgr.upsert_relations(left, [{
            "type": "kin",
            "target": right,
            "generation_method": PROVENANCE_GENERATION_METHOD,
            "evidence": {"kind": "shared_provenance", "field": "test"},
        }])

        assert result["created"] == 0
        assert result["unchanged"] == 1
        assert len(await _all_relations(bucket_mgr, [left, right])) == 1
        await _close_manager_tasks(bucket_mgr)

    asyncio.run(scenario())


def test_patch_builder_preserves_legacy_attribution_but_adds_current_evidence():
    buckets = [
        _bucket(
            "source1",
            source_session="session-a",
            source_event_ids=["event-a"],
            relations=[{"type": "kin", "target": "target1", "note": "old"}],
        ),
        _bucket(
            "target1",
            source_session="session-a",
            source_event_ids=["event-a"],
        ),
    ]
    plan = plan_relation_graph(buckets)
    patches, report = build_relation_patches(buckets, plan)
    patched = patches["source1"][0]

    assert report["planned_already_present_count"] == 1
    assert report["planned_new_count"] == 0
    assert patched["generation_method"] == LEGACY_GENERATION_METHOD
    assert patched["evidence"]["verification"] == "deterministic-backfill:v1"
    assert patched["note"] == "old"


def test_cleanup_plan_removes_only_orphans_and_reverse_kin_copy():
    buckets = [
        _bucket(
            "aaaa0001",
            relations=[{"type": "kin", "target": "bbbb0002", "note": "left"}],
        ),
        _bucket(
            "bbbb0002",
            relations=[{"type": "kin", "target": "aaaa0001", "note": "right"}],
        ),
        _bucket(
            "cccc0003",
            relations=[{"type": "explains", "target": "missing", "note": "broken"}],
        ),
    ]

    cleanup, report = build_cleanup_plan(buckets)

    assert cleanup == {
        "bbbb0002": [("kin", "aaaa0001")],
        "cccc0003": [("explains", "missing")],
    }
    assert report["by_reason"] == {
        "missing_target": 1,
        "reverse_kin_duplicate": 1,
    }
    assert report["prune_edge_count"] == 2


def test_prune_relations_is_exact_audited_and_activity_neutral(bucket_mgr):
    async def scenario():
        source = await bucket_mgr.create(content="source", name="source")
        first = await bucket_mgr.create(content="first", name="first")
        second = await bucket_mgr.create(content="second", name="second")
        assert await bucket_mgr.add_relation(source, first, "kin", note="remove")
        assert await bucket_mgr.add_relation(source, second, "explains", note="keep")
        activity_before = (await bucket_mgr.get(source))["metadata"]["last_active"]

        result = await bucket_mgr.prune_relations(
            source,
            [("kin", first)],
            actor="test",
        )
        stored = await bucket_mgr.get(source)

        assert result["removed"] == 1
        assert stored["metadata"]["last_active"] == activity_before
        assert stored["metadata"]["relations"] == [{
            "type": "explains",
            "target": second,
            "note": "keep",
        }]
        await _close_manager_tasks(bucket_mgr)

    asyncio.run(scenario())


def test_apply_backfill_snapshots_and_second_run_is_idempotent(
    test_config,
    monkeypatch,
    tmp_path,
):
    async def scenario():
        manager = BucketManager(test_config)

        async def no_auto_link(_bucket_id):
            return {"planned": 0, "created": 0, "enriched": 0, "failed": 0}

        monkeypatch.setattr(manager, "auto_link_created_bucket", no_auto_link)
        provenance = {
            "source_kind": "conversation",
            "source_session": "session-a",
            "source_event_ids": ["event-a"],
        }
        first = await manager.create(content="first", name="first", x_provenance=provenance)
        second = await manager.create(content="second", name="second", x_provenance=provenance)
        assert await manager.add_relation(first, second, "explains", note="legacy semantic")
        await _close_manager_tasks(manager)

        monkeypatch.setattr(relation_backfill, "load_config", lambda: test_config)

        def args(snapshot_id):
            return Namespace(
                apply=True,
                dry_run=False,
                buckets_dir=Path(test_config["buckets_dir"]),
                snapshot_root=tmp_path / "snapshots",
                snapshot_id=snapshot_id,
                audit_samples=10,
            )

        first_run = await relation_backfill.run_backfill(args("before-first"))
        second_run = await relation_backfill.run_backfill(args("before-second"))

        assert first_run["snapshot"]["manifest_sha256"]
        assert first_run["input"]["bucket_count"] == 2
        assert first_run["apply"]["created"] == 1
        assert first_run["apply"]["enriched"] == 1
        assert first_run["verification"] == {
            "preexisting_valid_edges_preserved": True,
            "preexisting_valid_edges_missing_count": 0,
            "planned_edges_present_count": 1,
            "planned_edges_missing_count": 0,
            "all_valid_edges_have_evidence": True,
            "all_valid_edges_have_generation_method": True,
            "relation_quality_clean": True,
        }
        assert second_run["apply"]["created"] == 0
        assert second_run["apply"]["enriched"] == 0
        assert second_run["apply"]["failed"] == 0
        assert second_run["verification"]["planned_edges_missing_count"] == 0

    asyncio.run(scenario())
