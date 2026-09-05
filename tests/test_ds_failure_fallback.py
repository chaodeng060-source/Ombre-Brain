import asyncio
import logging
import types

import pytest

import server
from recall_timing import (
    begin_recall_timing,
    finish_recall_timing,
    reset_recall_timing,
)


def _candidate(bucket_id: str, score: float) -> dict:
    return {
        "id": bucket_id,
        "content": bucket_id,
        "metadata": {"name": bucket_id},
        "_anchor_adapted_relevance_score": score,
    }


async def _run_with_timing(*args, **kwargs):
    token = begin_recall_timing()
    try:
        selected = await server._ds_filter_candidates(*args, **kwargs)
        receipt = finish_recall_timing(status="ok", partial=False)
    finally:
        reset_recall_timing(token)
    return selected, receipt


def _enable_failure_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")
    monkeypatch.setenv("OMBRE_DS_FAILURE_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FAILURE_ANCHOR_FLOOR", "0.450001")


def test_ds_failure_fallback_is_default_off_and_preserves_current_cap(monkeypatch):
    monkeypatch.delenv("OMBRE_DS_FAILURE_FALLBACK_ENABLED", raising=False)
    candidates = [
        _candidate("first", 0.20),
        _candidate("second", 0.45),
        _candidate("third", 0.44),
    ]

    selected = server._ds_conservative_failure_candidates(
        candidates,
        force_keep_ids=set(),
        max_results=2,
    )

    assert [row["id"] for row in selected] == ["first", "second"]


def test_ds_failure_fallback_replays_1214_as_top_one_without_engineering_noise(
    monkeypatch,
):
    _enable_failure_fallback(monkeypatch)
    # Recorded 12:14 pre-DS order and available Anchor scores. The fifth
    # candidate has no recorded score; do not substitute another bucket's score.
    # The two entity-only
    # engineering collisions sit at Anchor's 0.45 ceiling. A floor immediately
    # above that ceiling deliberately admits no scored row, then the required
    # deterministic top-one fallback keeps the first-ranked relevant memory.
    candidates = [
        _candidate("dd279da3beee", 0.444237),
        _candidate("0282fffed971", 0.311711),
        _candidate("013da98a75e5", 0.45),
        _candidate("019af40158f7", 0.45),
        {"id": "c0731e844589", "content": "", "metadata": {}},
    ]

    selected = server._ds_conservative_failure_candidates(
        candidates,
        force_keep_ids=set(),
        max_results=5,
    )

    assert [row["id"] for row in selected] == ["dd279da3beee"]
    assert not {"013da98a75e5", "019af40158f7"} & {
        row["id"] for row in selected
    }


def test_ds_failure_fallback_keeps_forced_ids_and_at_least_one(monkeypatch):
    _enable_failure_fallback(monkeypatch)
    candidates = [
        _candidate("top", 0.30),
        _candidate("forced", 0.10),
        _candidate("tail", 0.20),
    ]

    selected = server._ds_conservative_failure_candidates(
        candidates,
        force_keep_ids={"forced"},
        max_results=2,
    )

    assert [row["id"] for row in selected] == ["forced"]


@pytest.mark.asyncio
async def test_ds_invalid_payload_uses_conservative_fallback_and_status(
    monkeypatch,
):
    _enable_failure_fallback(monkeypatch)

    async def invalid(*_args, **_kwargs):
        raise server.DSFilterInvalidPayloadError("no_complete_json")

    monkeypatch.setattr(server, "_ds_semantic_select", invalid)
    candidates = [
        _candidate("top", 0.44),
        _candidate("013da98a75e5", 0.45),
        _candidate("019af40158f7", 0.45),
    ]

    selected, receipt = await _run_with_timing(
        "intimate query",
        candidates,
        mode="search",
        max_results=3,
        allow_empty=True,
    )

    assert [row["id"] for row in selected] == ["top"]
    assert receipt["ds_status"] == "invalid"
    assert receipt["ds_gate_outcome"] == "error"
    assert receipt["ds_gate_in"] == 3
    assert receipt["ds_gate_out"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (asyncio.TimeoutError(), "timeout"),
        (RuntimeError("provider failed"), "error"),
    ],
)
async def test_ds_runtime_failures_share_conservative_fallback_and_status(
    monkeypatch,
    failure,
    expected_status,
):
    _enable_failure_fallback(monkeypatch)

    async def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(server, "_ds_semantic_select", fail)
    candidates = [_candidate("top", 0.44), _candidate("noise", 0.45)]

    selected, receipt = await _run_with_timing(
        "query", candidates, mode="search", max_results=2, allow_empty=True
    )

    assert [row["id"] for row in selected] == ["top"]
    assert receipt["ds_status"] == expected_status
    assert receipt["ds_gate_in"] == 2
    assert receipt["ds_gate_out"] == 1


@pytest.mark.asyncio
async def test_ds_cancellation_records_timeout_fallback_but_propagates_cancel(
    monkeypatch,
):
    _enable_failure_fallback(monkeypatch)
    entered = asyncio.Event()

    async def hang(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "_ds_semantic_select", hang)
    candidates = [_candidate("top", 0.44), _candidate("noise", 0.45)]
    token = begin_recall_timing()
    try:
        task = asyncio.create_task(server._ds_filter_candidates(
            "query",
            candidates,
            mode="search",
            max_results=2,
            allow_empty=True,
        ))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        receipt = finish_recall_timing(status="deadline", partial=True)
    finally:
        reset_recall_timing(token)

    assert receipt["ds_status"] == "timeout"
    assert receipt["ds_gate_outcome"] == "timeout"
    assert receipt["ds_gate_in"] == 2
    assert receipt["ds_gate_out"] == 1


@pytest.mark.asyncio
async def test_ds_ok_path_is_unchanged_and_has_status(monkeypatch):
    _enable_failure_fallback(monkeypatch)

    async def keep_second(_query, candidates, _keep, _max_results):
        return candidates[1:2]

    monkeypatch.setattr(server, "_ds_semantic_select", keep_second)
    candidates = [_candidate("first", 0.44), _candidate("second", 0.30)]

    selected, receipt = await _run_with_timing(
        "query", candidates, mode="search", max_results=2, allow_empty=True
    )

    assert [row["id"] for row in selected] == ["second"]
    assert receipt["ds_status"] == "ok"
    assert receipt["ds_gate_outcome"] == "ok"


@pytest.mark.asyncio
async def test_ds_disabled_receipt_keeps_legacy_fields_without_call_status(
    monkeypatch,
):
    monkeypatch.delenv("OMBRE_DS_FILTER_ENABLED", raising=False)
    candidates = [_candidate("first", 0.44), _candidate("second", 0.30)]

    selected, receipt = await _run_with_timing(
        "query", candidates, mode="search", max_results=2
    )

    assert selected == candidates
    assert receipt["ds_gate_outcome"] == "disabled"
    assert "ds_status" not in receipt


@pytest.mark.asyncio
async def test_ds_non_string_payload_is_classified_without_logging_content(
    monkeypatch,
    caplog,
):
    async def create(**_kwargs):
        message = types.SimpleNamespace(content={"secret": "must-not-log"})
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], model="test-model")

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        )
    )
    monkeypatch.setattr(
        server,
        "_ds_filter_provider",
        lambda: ("shared", "test-model", client, {}),
    )
    monkeypatch.setenv("OMBRE_DS_FILTER_CACHE_TTL", "0")

    with caplog.at_level(logging.ERROR, logger="ombre_brain"):
        with pytest.raises(server.DSFilterInvalidPayloadError) as caught:
            await server._ds_semantic_select(
                "query",
                [_candidate("first", 0.44), _candidate("second", 0.30)],
                set(),
                2,
            )

    assert caught.value.reason == "non_string_content"
    assert "parse_reason=non_string_content" in caplog.text
    assert "must-not-log" not in caplog.text


@pytest.mark.asyncio
async def test_anchor_probe_persists_ds_status_and_counts(monkeypatch):
    candidates = [
        _candidate("first", 0.44),
        _candidate("second", 0.45),
    ]

    class Manager:
        async def search(self, *_args, **_kwargs):
            return list(candidates)

        async def get(self, bucket_id):
            return next(
                (row for row in candidates if row["id"] == bucket_id),
                None,
            )

    class Embedding:
        async def search_similar_with_status(self, *_args, **_kwargs):
            return [], "ok"

    async def gate(_query, rows, **_kwargs):
        server.record_recall_ds_gate("invalid", len(rows), 1)
        return rows[:1]

    captured = []
    monkeypatch.setattr(server, "bucket_mgr", Manager())
    monkeypatch.setattr(server, "embedding_engine", Embedding())
    monkeypatch.setattr(server, "_resolve_entity_recall", lambda query: (query, []))
    monkeypatch.setattr(server, "_get_entity_store", lambda **_kwargs: None)
    monkeypatch.setattr(server, "_is_main_recall_bucket", lambda _row: True)
    monkeypatch.setattr(
        server,
        "_filter_z_fact_candidates",
        lambda rows, **_kwargs: list(rows),
    )
    monkeypatch.setattr(
        server,
        "_passes_nonkeyword_recall_filters",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(server, "_ds_filter_candidates", gate)
    monkeypatch.setattr(server, "_exact_retrieval_key_ids", lambda *_args: set())
    monkeypatch.setattr(server, "_append_recall_status_trace", captured.append)
    monkeypatch.setitem(server.config, "query_expansion", {"enabled": False})

    record = await server._probe_anchor_status("real query")

    assert captured == [record]
    assert record["ds_status"] == "invalid"
    assert record["ds_gate_outcome"] == "error"
    assert record["ds_gate_in"] == 2
    assert record["ds_gate_out"] == 1
