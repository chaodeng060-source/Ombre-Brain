"""Mechanical candidate-trace contracts, not recall-quality acceptance data."""
import copy
import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

import server
from bucket_manager import BucketManager
from candidate_diagnostics import CandidateDiagnostics, current


def bomb(*args, **kwargs):
    raise AssertionError("forbidden diagnostic side effect")


async def async_bomb(*args, **kwargs):
    bomb()


async def empty(*args, **kwargs):
    return []


class NoEgress:
    def __getattribute__(self, name):
        raise AssertionError("diagnostic touched a non-embedding model client")


def request(payload):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode()}
    return Request({"type": "http", "method": "POST", "path": "/api/breath-candidates",
                    "headers": []}, receive)


def bucket(i, *, world="", domain="100", content=None):
    bid = f"b{i:03}"
    return {"id": bid, "content": content or str(10000 + i), "metadata": {
        "id": bid, "name": str(i), "type": "dynamic", "world": world,
        "domain": [domain], "importance": 5, "tags": [],
    }}


class Vector:
    def __init__(self, hits=()):
        self.hits = list(hits)
        self.calls = []

    async def search_similar_diagnostic(self, query, *, target_ids, **kwargs):
        self.calls.append(query)
        return self.hits, "ok", {}, {"backend": "fixture", "targets": {}}


@pytest.fixture
def core(test_config, monkeypatch):
    # Resident-only buckets and a numeric score vector, exercising REAL search,
    # fusion, support, dedup, seen, anchor, worklog and pre-gate return code.
    manager = BucketManager(test_config)
    rows = [bucket(i) for i in range(40)]
    manager._recall_snapshot_cache[manager._recall_cache_key(False, None)] = tuple(rows)
    manager._bm25_mode = "live"
    manager._bm25_dirty = True
    manager._bm25 = SimpleNamespace(
        _ids=[b["id"] for b in rows], _index=object(), _keyword_score_rows=None,
        score=lambda q: {b["id"]: 1.0 - i / 100 for i, b in enumerate(rows)},
    )
    manager._retrieval_hints_enabled = False
    manager._retrieval_attribution_enabled = False
    monkeypatch.setattr(manager, "_schedule_bm25_rebuild", bomb)
    monkeypatch.setattr(manager, "_schedule_recall_snapshot_refresh", bomb)
    monkeypatch.setattr(manager, "prewarm_recall_snapshot", async_bomb)
    monkeypatch.setattr(manager, "_refresh_retrieval_hint_index", async_bomb)
    monkeypatch.setattr(manager, "_calc_topic_score", lambda query, b: 1.0)
    monkeypatch.setattr(manager, "rare_literal_hits", lambda *a, **k: {})
    monkeypatch.setattr(manager, "literal_term_df_hits", lambda *a, **k: {})
    async def get(bid):
        return copy.deepcopy(next((b for b in rows if b["id"] == bid), None))
    monkeypatch.setattr(manager, "get", get)
    cfg = {**server.config, **test_config, "current_world": "",
           "entities": {"enabled": False}, "fact_slots": {"enabled": False},
           "query_expansion": {"enabled": True, "allowed_intents": ["default"]},
           "e_axis_recall": {"enabled": False}, "sense": {"enabled": False}}
    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    vectors = Vector()
    monkeypatch.setattr(server, "embedding_engine", vectors)
    monkeypatch.setattr(server, "dehydrator", NoEgress())
    monkeypatch.setattr(server, "decay_engine", SimpleNamespace(apply_retrieval_decay=lambda s, m: s))
    monkeypatch.setattr(server, "episode_engine", SimpleNamespace(ensure_started=async_bomb))
    for name in ("_ensure_decay_background", "_ensure_consolidation_background", "expand_query",
                 "_ds_filter_candidates", "_dehydrate_for_recall"):
        monkeypatch.setattr(server, name, async_bomb)
    for name in ("_maybe_start_backfill", "_remember_session_seen_ids", "_local_partial_recall_text",
                 "set_recall_partial_result", "_emit_upstream_fusion_shadow"):
        monkeypatch.setattr(server, name, bomb)
    monkeypatch.setattr(server, "search_rg_literal", empty)
    monkeypatch.setattr(server, "search_curated_lexical", empty)
    monkeypatch.setenv("OMBRE_CANDIDATES_DIAGNOSTICS_ENABLED", "1")
    monkeypatch.setenv("OMBRE_PG_LEXICAL_MODE", "off")
    monkeypatch.setenv("OMBRE_UPSTREAM_FUSION_SHADOW", "1")
    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_ENABLED", "0")
    monkeypatch.setenv("OMBRE_SESSION_SEEN_POLICIES", "conversation,reflex")
    return manager, rows, vectors


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", ["0", "1"])
async def test_real_core_returns_before_all_side_effects(core, monkeypatch, parallel):
    manager, rows, vectors = core
    monkeypatch.setenv("OMBRE_PARALLEL_RETRIEVAL", parallel)
    before = copy.deepcopy(rows)
    response = await server.api_breath_candidates(request({
        "query": "10001", "target_ids": ["b000", "b030", "absent"], "max_results": 10,
    }))
    assert response.status_code == 200, response.body
    result = json.loads(response.body)
    assert result["mode"] == "candidates_only"
    assert result["relation"]["status"] == "not_executed_post_gate"
    assert result["bm25"]["targets"]["b030"]["index_member"] is True
    assert result["bm25"]["targets"]["absent"]["index_member"] is False
    stages = {s["stage"]: s for s in result["stages"]}
    assert stages["keyword_threshold_ranked"]["targets"]["b030"]["rank"] == 31
    assert stages["keyword_top_k"]["targets"]["b030"] is None
    assert {entry["stage"] for entry in result["target_drop_observations"]["b030"]} == {"keyword_top_k"}
    assert stages["session_seen"]["status"] == "skipped_policy"
    assert vectors.calls == ["10001"]
    assert rows == before
    assert current.get() is None
    assert b'"content"' not in response.body and b'"name"' not in response.body
    assert b'"query"' not in response.body and b'10001' not in response.body


@pytest.mark.asyncio
async def test_default_off_does_not_enter_core(monkeypatch):
    monkeypatch.delenv("OMBRE_CANDIDATES_DIAGNOSTICS_ENABLED", raising=False)
    monkeypatch.setattr(server, "breath", async_bomb)
    response = await server.api_breath_candidates(request({"query": "1", "target_ids": ["b001"]}))
    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {}, {"query": "1", "target_ids": []}, {"query": "1", "target_ids": ["bad/id"]},
    {"query": " ", "target_ids": ["b001"]},
    {"query": "1", "target_ids": ["b001"], "domain": "feel"},
])
async def test_invalid_input_never_starts_request(payload, monkeypatch):
    monkeypatch.setenv("OMBRE_CANDIDATES_DIAGNOSTICS_ENABLED", "1")
    monkeypatch.setattr(server, "breath", async_bomb)
    response = await server.api_breath_candidates(request(payload))
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_keyword_world_time_domain_stages_and_dirty_no_refresh(core):
    manager, rows, vectors = core
    rows[1]["metadata"]["world"] = "200"
    rows[2]["metadata"]["domain"] = ["200"]
    trace = CandidateDiagnostics(["b001", "b002"])
    token = current.set(trace)
    try:
        await manager.search("10001", world_filter=[], domain_filter=["100"], relevance_first=True,
                             preloaded_buckets=rows)
    finally:
        current.reset(token)
    stages = {s["stage"]: s for s in trace.stages}
    assert stages["keyword_domain"]["targets"]["b002"] is None
    assert stages["keyword_domain"]["targets"]["b001"] is not None
    assert stages["keyword_world"]["targets"]["b001"] is None


@pytest.mark.asyncio
async def test_cold_snapshot_fails_without_prewarm(core):
    manager, rows, vectors = core
    manager._recall_snapshot_cache.clear()
    response = await server.api_breath_candidates(request({"query": "10001", "target_ids": ["b001"]}))
    assert response.status_code == 503
    assert current.get() is None


@pytest.mark.asyncio
async def test_channel_timeout_not_reported_as_target_absence(core, monkeypatch):
    async def failed(*args, **kwargs):
        return [], "timeout", {}, {"backend": "none", "targets": {}}
    monkeypatch.setattr(core[2], "search_similar_diagnostic", failed)
    response = await server.api_breath_candidates(request({"query": "10001", "target_ids": ["b001"]}))
    assert response.status_code == 503
    assert json.loads(response.body)["completed"] is False


@pytest.mark.asyncio
async def test_persistent_write_guards_remain_zero(core, monkeypatch, tmp_path):
    import builtins
    import io
    import os
    import sqlite3
    originals = {"open": builtins.open, "io_open": io.open, "os_open": os.open}
    writes = []
    def guarded_open(fn):
        def checked(path, mode="r", *a, **k):
            if any(flag in mode for flag in "wax+"):
                writes.append(str(path))
                bomb()
            return fn(path, mode, *a, **k)
        return checked
    def checked_os_open(path, flags, *a, **k):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            writes.append(str(path))
            bomb()
        return originals["os_open"](path, flags, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded_open(originals["open"]))
    monkeypatch.setattr(io, "open", guarded_open(originals["io_open"]))
    monkeypatch.setattr(os, "open", checked_os_open)
    monkeypatch.setattr(sqlite3, "connect", bomb)
    response = await server.api_breath_candidates(request({"query": "10001", "target_ids": ["b001"]}))
    assert response.status_code == 200
    assert writes == []


@pytest.mark.asyncio
async def test_normal_breath_flag_on_off_and_diagnostic_candidate_equivalence(core, monkeypatch):
    manager, rows, vectors = core
    async def noop(*a, **k):
        return None
    monkeypatch.setattr(manager, "_schedule_bm25_rebuild", lambda: None)
    monkeypatch.setattr(manager, "_schedule_recall_snapshot_refresh", lambda *a: None)
    monkeypatch.setattr(manager, "_refresh_retrieval_hint_index", noop)
    for name in ("_ensure_decay_background", "_ensure_consolidation_background"):
        monkeypatch.setattr(server, name, noop)
    monkeypatch.setattr(server, "episode_engine", SimpleNamespace(ensure_started=noop))
    monkeypatch.setattr(server, "_maybe_start_backfill", lambda: None)
    monkeypatch.setattr(server, "_emit_upstream_fusion_shadow", lambda *a: None)
    monkeypatch.setattr(server, "_local_partial_recall_text", lambda *a, **k: "")
    monkeypatch.setattr(server, "set_recall_partial_result", lambda *a: None)
    monkeypatch.setitem(server.config, "query_expansion", {"enabled": False})
    monkeypatch.setattr(vectors, "search_similar", empty, raising=False)
    captured = []
    class ReachedGate(BaseException):
        pass
    async def gate(query, candidates, **kwargs):
        captured.append([(b["id"], b["score"]) for b in candidates])
        raise ReachedGate()
    monkeypatch.setattr(server, "_ds_filter_candidates", gate)
    for flag in ("0", "1"):
        monkeypatch.setenv("OMBRE_CANDIDATES_DIAGNOSTICS_ENABLED", flag)
        with pytest.raises(ReachedGate):
            await server.breath(query="10001", max_results=10, include_images=False, include_body_state=False)
    response = await server.api_breath_candidates(request({"query": "10001", "target_ids": ["b001"],
                                                          "max_results": 10}))
    assert response.status_code == 200
    result = json.loads(response.body)
    assert captured[0] == captured[1] == [(b["id"], b["score"]) for b in result["candidates"]]


@pytest.mark.asyncio
async def test_default_off_keeps_original_snapshot_and_index_refresh(core, monkeypatch):
    manager, rows, _ = core
    calls = []
    monkeypatch.setattr(manager, "_schedule_recall_snapshot_refresh", lambda key: calls.append("snapshot"))
    monkeypatch.setattr(manager, "_schedule_bm25_rebuild", lambda: calls.append("bm25"))
    async def refresh(*args):
        calls.append("hints")
    monkeypatch.setattr(manager, "_refresh_retrieval_hint_index", refresh)
    assert current.get() is None
    await manager.borrow_recall_snapshot()
    await manager.search("10001", preloaded_buckets=rows, relevance_first=True)
    assert calls == ["snapshot", "hints", "bm25"]


@pytest.mark.asyncio
async def test_full_breath_with_real_vector_core_one_embedding_no_store_change(core, monkeypatch, tmp_path):
    import httpx
    from pathlib import Path
    from openai import AsyncOpenAI
    from tests.test_candidate_vector_diagnostics import _make_engine, _write_sqlite_store
    attempts = []
    def respond(req):
        attempts.append(json.loads(req.content))
        return httpx.Response(200, json={"object": "list", "model": "numeric-test-model",
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client = AsyncOpenAI(api_key="numeric-fixture", base_url="https://fixture.invalid/v1",
                             max_retries=4, http_client=http_client)
        engine = _make_engine(tmp_path, client)
        _write_sqlite_store(engine.db_path, [("b001", json.dumps([1.0, 0.0, 0.0, 0.0]))])
        path = Path(engine.db_path)
        before = path.read_bytes(), path.stat().st_mtime_ns
        monkeypatch.setattr(server, "embedding_engine", engine)
        monkeypatch.setenv("OMBRE_PG_RECALL_ENABLED", "0")
        for name in ("_init_db", "_store_embedding", "_invalidate_vector_cache", "_schedule_pg_vector_shadow"):
            monkeypatch.setattr(engine, name, bomb)
        response = await server.api_breath_candidates(request({"query": "10001", "target_ids": ["b001", "absent"]}))
        assert response.status_code == 200, response.body
        result = json.loads(response.body)
        assert len(attempts) == 1
        assert attempts[0]["input"] == "10001"
        assert result["vector"]["targets"]["b001"]["returned_rank"] == 1
        assert result["vector"]["targets"]["absent"]["index_member"] is False
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.asyncio
async def test_api_bearer_protects_new_route(monkeypatch):
    import httpx
    from starlette.applications import Starlette
    from starlette.routing import Route
    from mcp_auth import APIBearerAuthMiddleware
    token = "0123456789" * 4
    app = APIBearerAuthMiddleware(Starlette(routes=[
        Route("/api/breath-candidates", server.api_breath_candidates, methods=["POST"]),
    ]), token=token)
    monkeypatch.delenv("OMBRE_CANDIDATES_DIAGNOSTICS_ENABLED", raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
        denied = await client.post("/api/breath-candidates", json={})
        accepted = await client.post("/api/breath-candidates", json={},
                                     headers={"Authorization": f"Bearer {token}"})
    assert denied.status_code == 401
    assert accepted.status_code == 404
