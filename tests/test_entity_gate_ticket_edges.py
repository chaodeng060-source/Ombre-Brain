from __future__ import annotations

from pathlib import Path

import pytest

import bucket_manager
from bm25_index import BM25Index
import server
from tests.test_entity_gate_ticket_integration import (
    ENTITY_HIGH,
    ENTITY_HIGH_BOUNDARY,
    ENTITY_LOW,
    NORMAL_KEYWORD,
    _configure_breath,
)


_REAL_DS_FILTER = server._ds_filter_candidates


def _row(
    bucket_id: str,
    *,
    weak_entity: bool = False,
    content: str | None = None,
) -> dict:
    row = {
        "id": bucket_id,
        "content": content or f"body:{bucket_id}",
        "metadata": {"name": bucket_id},
        "_anchor_adapted_relevance_score": 0.3,
    }
    if weak_entity:
        row["weak_entity"] = True
    return row


def _enable_ds(monkeypatch) -> None:
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")
    monkeypatch.setenv("OMBRE_DS_FAILURE_FALLBACK_ENABLED", "1")


@pytest.mark.asyncio
async def test_breath_guard_off_is_byte_and_call_shape_legacy_exact(
    tmp_path: Path,
    monkeypatch,
):
    manager, _embedding, _captures = _configure_breath(
        tmp_path,
        monkeypatch,
        guard_enabled=False,
    )
    monkeypatch.delenv("OMBRE_ENTITY_SCORE_GUARD_ENABLED", raising=False)
    calls = []

    # Deliberately retain the pre-ticket signature. Passing even an empty new
    # keyword argument while the guard is off must fail this characterization.
    async def legacy_ds(
        query,
        candidates,
        *,
        mode,
        max_results,
        force_keep_ids=None,
        allow_empty=False,
    ):
        calls.append({
            "query": query,
            "mode": mode,
            "max_results": max_results,
            "force_keep_ids": set(force_keep_ids or ()),
            "allow_empty": allow_empty,
            "ids": [row["id"] for row in candidates],
            "new_fields": [
                {
                    key: row[key]
                    for key in ("weak_entity", "_entity_recall_evidence")
                    if key in row
                }
                for row in candidates
            ],
        })
        return list(candidates)[:max_results]

    monkeypatch.setattr(server, "_ds_filter_candidates", legacy_ds)

    result = await server.breath(
        query="rare-name",
        max_results=3,
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )

    assert result == (
        "[bucket_id:normal-keyword] body:normal-keyword\n"
        "---\n"
        "[语义关联] [bucket_id:normal-vector] body:normal-vector\n"
        "---\n"
        "[实体关联] [bucket_id:6cc5995aea84] body:6cc5995aea84"
    )
    assert calls == [{
        "query": "rare-name",
        "mode": "search",
        "max_results": 3,
        "force_keep_ids": set(),
        "allow_empty": True,
        "ids": [
            "normal-keyword",
            "normal-vector",
            ENTITY_LOW,
            ENTITY_HIGH_BOUNDARY,
            ENTITY_HIGH,
        ],
        "new_fields": [{}, {}, {}, {}, {}],
    }]
    assert manager.df_calls == []


@pytest.mark.asyncio
async def test_probe_guard_off_omits_ticket_kwarg_and_trace_field(
    tmp_path: Path,
    monkeypatch,
):
    manager, _embedding, _captures = _configure_breath(
        tmp_path,
        monkeypatch,
        guard_enabled=False,
    )
    monkeypatch.delenv("OMBRE_ENTITY_SCORE_GUARD_ENABLED", raising=False)
    monkeypatch.setattr(server, "_append_recall_status_trace", lambda _row: None)
    calls = []

    async def legacy_ds(
        query,
        candidates,
        *,
        mode,
        max_results,
        force_keep_ids=None,
        allow_empty=False,
    ):
        calls.append({
            "query": query,
            "mode": mode,
            "max_results": max_results,
            "force_keep_ids": set(force_keep_ids or ()),
            "allow_empty": allow_empty,
            "ids": [row["id"] for row in candidates],
        })
        return list(candidates)[:max_results]

    monkeypatch.setattr(server, "_ds_filter_candidates", legacy_ds)

    record = await server._probe_anchor_status("rare-name")

    assert calls == [{
        "query": "rare-name",
        "mode": "search",
        "max_results": 2,
        "force_keep_ids": set(),
        "allow_empty": False,
        "ids": [
            "normal-keyword",
            "normal-vector",
            ENTITY_LOW,
            ENTITY_HIGH_BOUNDARY,
            ENTITY_HIGH,
        ],
    }]
    assert set(record) == {
        "ts",
        "query_len",
        "intent",
        "angle_count",
        "query_expansion_skipped_hot_path",
        "vector_status",
        "keyword_candidate_count",
        "vector_candidate_count",
        "entity_candidate_count",
        "entity_query_canonicalized",
        "final_candidate_count",
        "has_evidence",
        "timing_ms",
    }
    assert manager.df_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("normal_count", [0, 1], ids=["zero-normal", "singleton"])
async def test_ticket_cannot_create_model_call_for_legacy_deterministic_noop(
    monkeypatch,
    normal_count,
):
    _enable_ds(monkeypatch)
    calls = 0

    async def select(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return list(_args[1])

    monkeypatch.setattr(server, "_ds_semantic_select", select)
    normals = [_row(f"normal-{index}") for index in range(normal_count)]

    selected = await server._ds_filter_candidates(
        "rare-name",
        normals,
        mode="search",
        max_results=2,
        allow_empty=False,
        weak_entity_tickets=[_row("ticket", weak_entity=True)],
    )

    assert selected == normals
    assert calls == 0


@pytest.mark.asyncio
async def test_ds_ticket_drops_without_model_for_empty_query_or_zero_budget(
    monkeypatch,
):
    _enable_ds(monkeypatch)
    calls = 0

    async def select(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return list(_args[1])

    monkeypatch.setattr(server, "_ds_semantic_select", select)
    normals = [_row("normal-1"), _row("normal-2")]
    ticket = _row("ticket", weak_entity=True)

    empty_query = await server._ds_filter_candidates(
        "",
        normals,
        mode="search",
        max_results=2,
        weak_entity_tickets=[ticket],
    )
    zero_budget = await server._ds_filter_candidates(
        "rare-name",
        normals,
        mode="search",
        max_results=0,
        weak_entity_tickets=[ticket],
    )

    assert empty_query == normals
    assert zero_budget == []
    assert calls == 0


@pytest.mark.asyncio
async def test_full_forced_normal_pool_short_circuits_ticket(monkeypatch):
    _enable_ds(monkeypatch)
    calls = 0

    async def select(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return list(_args[1])

    monkeypatch.setattr(server, "_ds_semantic_select", select)
    normals = [_row("normal-1"), _row("normal-2")]

    selected = await server._ds_filter_candidates(
        "rare-name",
        normals,
        mode="search",
        max_results=2,
        force_keep_ids={"normal-1", "normal-2", "ticket"},
        allow_empty=True,
        weak_entity_tickets=[_row("ticket", weak_entity=True)],
    )

    assert selected == normals
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dedupe_kind",
    ["state-id", "normal-content", "session-seen"],
)
async def test_breath_ticket_pool_dedupes_state_content_and_session_before_ds(
    tmp_path: Path,
    monkeypatch,
    dedupe_kind,
):
    manager, _embedding, captures = _configure_breath(
        tmp_path,
        monkeypatch,
        guard_enabled=True,
    )

    def two_rare_terms(terms):
        manager.df_calls.append(tuple(terms))
        return {
            "rare-name": (1, 10),
            "common-name": (1, 10),
            "very-common-name": (3, 10),
        }

    monkeypatch.setattr(manager, "entity_term_df_stats", two_rare_terms)
    if dedupe_kind == "state-id":
        async def state_link(*_args, **_kwargs):
            return [manager.buckets[ENTITY_LOW]]

        monkeypatch.setattr(server, "_state_link_recall_candidates", state_link)
    elif dedupe_kind == "normal-content":
        manager.buckets[NORMAL_KEYWORD]["content"] = manager.buckets[
            ENTITY_LOW
        ]["content"]
    else:
        monkeypatch.setattr(
            server,
            "_load_session_seen_ids",
            lambda _session_id: {ENTITY_LOW},
        )

    await server.breath(
        query="rare-name",
        max_results=3,
        relation_depth=0,
        session_id="edge-session",
        include_images=False,
        include_body_state=False,
    )

    assert len(captures["ds_calls"]) == 1
    assert captures["ds_calls"][0]["ticket_ids"] == [ENTITY_HIGH_BOUNDARY]


@pytest.mark.asyncio
async def test_probe_ticket_only_does_not_create_ds_call_or_evidence(
    tmp_path: Path,
    monkeypatch,
):
    manager, embedding, _captures = _configure_breath(
        tmp_path,
        monkeypatch,
        guard_enabled=True,
    )

    async def no_keyword(*_args, **_kwargs):
        return []

    async def no_vector(_query, top_k=20):
        return [], "ok"

    manager.search = no_keyword
    embedding.search_similar_with_status = no_vector
    monkeypatch.setattr(server, "_ds_filter_candidates", _REAL_DS_FILTER)
    monkeypatch.setattr(server, "_append_recall_status_trace", lambda _row: None)
    calls = 0

    async def select(_query, rows, _keep, _max_results):
        nonlocal calls
        calls += 1
        return list(rows)

    monkeypatch.setattr(server, "_ds_semantic_select", select)

    record = await server._probe_anchor_status("rare-name")

    assert record["entity_ticket_evaluation"] == (
        "not_run_in_lightweight_probe"
    )
    assert "weak_entity_ticket_count" not in record
    assert record["final_candidate_count"] == 0
    assert record["has_evidence"] is False
    assert calls == 0


def test_entity_term_df_stats_intersects_multitoken_on_current_generation():
    pytest.importorskip("rank_bm25")
    index = BM25Index()
    index.build([
        {"id": "d1", "content": "alpha beta", "metadata": {}},
        {"id": "d2", "content": "alpha gamma", "metadata": {}},
        {"id": "d3", "content": "beta gamma", "metadata": {}},
    ])

    assert index.entity_term_df_stats(
        ["alpha beta", "alpha", "missing"]
    ) == {
        "alpha beta": (1, 3),
        "alpha": (2, 3),
    }

    fresh = index.with_upsert({
        "id": "d2",
        "content": "alpha beta",
        "metadata": {},
    })
    assert index.entity_term_df_stats(["alpha beta"]) == {
        "alpha beta": (1, 3),
    }
    assert fresh.entity_term_df_stats(["alpha beta"]) == {
        "alpha beta": (2, 3),
    }


def test_bucket_manager_entity_df_wrapper_uses_live_pointer_and_fails_closed():
    pytest.importorskip("rank_bm25")
    old = BM25Index()
    old.build([
        {"id": "d1", "content": "alpha beta", "metadata": {}},
        {"id": "d2", "content": "alpha gamma", "metadata": {}},
    ])
    fresh = old.with_upsert({
        "id": "d2",
        "content": "alpha beta",
        "metadata": {},
    })

    class Manager:
        _bm25_mode = "live"
        _bm25 = fresh

    manager = Manager()
    lookup = bucket_manager.BucketManager.entity_term_df_stats
    assert lookup(manager, ["alpha beta"]) == {"alpha beta": (2, 2)}
    manager._bm25 = old
    assert lookup(manager, ["alpha beta"]) == {"alpha beta": (1, 2)}

    manager._bm25_mode = "off"
    assert lookup(manager, ["alpha beta"]) == {}
    manager._bm25_mode = "live"
    manager._bm25 = None
    assert lookup(manager, ["alpha beta"]) == {}

    class BrokenIndex:
        @staticmethod
        def entity_term_df_stats(_terms):
            raise RuntimeError("boom")

    manager._bm25 = BrokenIndex()
    assert lookup(manager, ["alpha beta"]) == {}
