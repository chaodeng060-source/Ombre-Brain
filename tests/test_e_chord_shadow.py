import asyncio
import copy
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from e_chord_shadow import (
    EChordFinalSelectionLedger,
    EChordShadowLedger,
    FINAL_SELECTION_SCHEMA,
    build_shadow_receipt,
    load_e_chord_shadow_config,
    parse_live_chord,
    propose_chord_reorder,
    validate_final_selection,
)


NOW_MS = 1_777_777_777_000


def _payload(**overrides):
    value = {
        "schema": "live_chord.v1",
        "turn_id": "turn-abc123",
        "agent_id": "claude",
        "e_authored_by": "哥哥",
        "session_scope": "a" * 32,
        "source_turn_digest": "b" * 64,
        "captured_at_ms": NOW_MS - 100,
        "facets": [
            {
                "motivation": "attach",
                "drive_key": "attachment",
                "tendency": "comfort",
                "hunger": 0.91,
                "salience": 0.91,
            },
            {
                "motivation": "selfcheck",
                "drive_key": "curiosity",
                "tendency": "alert",
                "hunger": 0.71,
                "salience": 0.71,
            },
        ],
    }
    value.update(overrides)
    return value


def _parse(payload=None):
    chord, reason = parse_live_chord(
        _payload() if payload is None else payload,
        expected_turn_id="turn-abc123",
        expected_agent_id="claude",
        now_ms=NOW_MS,
        max_age_ms=30_000,
    )
    assert reason == "accepted"
    assert chord is not None
    return chord


def _candidate(
    bucket_id,
    *,
    event="event-1",
    tendency="engage",
    author="哥哥",
    base=0.5,
    e_score=0.5,
    relevance=0.8,
    relevance_band=0,
    fact=False,
):
    metadata = {"event_id": event, "type": "experience"}
    if fact:
        metadata.update({"fact_key": "preference.example", "fact_status": "current"})
    return {
        "id": bucket_id,
        "metadata": metadata,
        "_fused_relevance_score": relevance,
        "_e_chord_relevance_band_id": relevance_band,
        "_pre_e_tie_break_score": base,
        "_non_relevance_tie_break_score": e_score,
        "_e_axis_annotation": {
            "response_tendency": tendency,
            "confidence": 0.9,
            "authored_by": author,
        },
    }


def _final_selection(**overrides):
    value = {
        "schema": FINAL_SELECTION_SCHEMA,
        "recorded_at_ms": NOW_MS,
        "agent_id": "claude",
        "source_turn_digest": "b" * 64,
        "projection_digest": "c" * 64,
        "attempt_index": 0,
        "pool_ids": ["one"],
        "final_injected_ids": [],
        "outside_pool_ids": [],
        "arms": {"a": [], "b": [], "c": []},
        "request_path_delta_ms": 0.25,
    }
    value.update(overrides)
    return value


def test_parser_accepts_only_bounded_text_free_contract():
    chord = _parse()

    assert len(chord.facets) == 2
    assert chord.facets[0].tendency == "comfort"
    assert chord.e_authored_by == "哥哥"
    assert chord.source_turn_digest == "b" * 64
    assert not hasattr(chord, "query")


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda row: row.update(query="private text"), "schema.keys"),
        (lambda row: row.update(turn_id="other-turn"), "scope.turn"),
        (lambda row: row.update(agent_id="hajimi"), "scope.agent"),
        (lambda row: row.update(e_authored_by="xiaojuan"), "scope.author"),
        (lambda row: row.update(session_scope="dm:claude"), "schema.session_scope"),
        (lambda row: row.update(source_turn_digest="not-a-digest"), "schema.source_turn_digest"),
        (lambda row: row.update(captured_at_ms=NOW_MS - 30_001), "schema.stale"),
        (lambda row: row.update(captured_at_ms=NOW_MS + 5_001), "schema.future"),
        (lambda row: row.update(facets=row["facets"] * 2), "schema.facets"),
        (lambda row: row["facets"][0].update(salience=float("nan")), "schema.salience"),
        (lambda row: row["facets"][0].update(salience=10 ** 400), "schema.salience"),
        (lambda row: row["facets"][0].update(tendency="engage"), "schema.mapping"),
        (lambda row: row["facets"][0].update(drive_key="curiosity"), "schema.drive_key"),
        (lambda row: row["facets"].reverse(), "schema.order"),
    ],
)
def test_parser_fails_closed_on_scope_privacy_freshness_and_mapping(mutate, reason):
    payload = _payload()
    mutate(payload)
    chord, observed = parse_live_chord(
        payload,
        expected_turn_id="turn-abc123",
        expected_agent_id="claude",
        now_ms=NOW_MS,
        max_age_ms=30_000,
    )

    assert chord is None
    assert observed == reason


def test_config_is_default_off_and_active_contract_is_bounded():
    assert load_e_chord_shadow_config({}).enabled is False
    assert load_e_chord_shadow_config({"e_chord_shadow": {"enabled": True}}).enabled is False
    active = load_e_chord_shadow_config({
        "e_chord_shadow": {
            "enabled": True,
            "mode": "shadow",
            "near_tie_epsilon": 0.03,
            "max_age_ms": 30_000,
        }
    })
    assert active.enabled is True
    assert active.near_tie_epsilon == 0.03
    with pytest.raises(ValueError, match="near_tie_epsilon"):
        load_e_chord_shadow_config({
            "e_chord_shadow": {
                "enabled": True,
                "mode": "shadow",
                "near_tie_epsilon": math.nan,
            }
        })


def test_same_event_near_tie_can_propose_one_adjacent_swap_without_mutation():
    original = [
        _candidate("plain", tendency="engage", e_score=0.52),
        _candidate("comfort", tendency="comfort", e_score=0.50),
    ]
    before = copy.deepcopy(original)

    proposal = propose_chord_reorder(original, _parse(), near_tie_epsilon=0.03)

    assert [row["id"] for row in original] == ["plain", "comfort"]
    assert original == before
    assert proposal.b_ids == ("plain", "comfort")
    assert proposal.c_ids == ("comfort", "plain")
    assert proposal.swaps[0].promoted_id == "comfort"
    assert proposal.violations == ()


@pytest.mark.parametrize(
    "left,right,reason",
    [
        (_candidate("left", event="one"), _candidate("right", event="two", tendency="comfort"), "event_lock"),
        (_candidate("left"), _candidate("right", tendency="comfort", author="小卷"), "author"),
        (_candidate("left", fact=True), _candidate("right", tendency="comfort"), "factual"),
        (_candidate("left", e_score=0.7), _candidate("right", tendency="comfort", e_score=0.5), "near_tie"),
    ],
)
def test_wrong_event_wrong_author_fact_and_far_tie_never_move(left, right, reason):
    proposal = propose_chord_reorder([left, right], _parse(), near_tie_epsilon=0.03)

    assert proposal.c_ids == proposal.b_ids
    assert reason in proposal.skipped_reasons


def test_missing_explicit_event_lock_and_zero_candidates_are_noops():
    unlocked = [_candidate("left"), _candidate("right", tendency="comfort")]
    for row in unlocked:
        row["metadata"].pop("event_id")

    assert propose_chord_reorder([], _parse()).c_ids == ()
    proposal = propose_chord_reorder(unlocked, _parse())
    assert proposal.c_ids == proposal.b_ids
    assert "event_lock" in proposal.skipped_reasons


def test_adjacent_candidates_in_different_relevance_bands_never_move():
    rows = [
        _candidate(
            "leader", tendency="engage", relevance=1.0, e_score=0.52,
            relevance_band=0,
        ),
        _candidate(
            "same-band", tendency="engage", relevance=0.7, e_score=0.51,
            relevance_band=0,
        ),
        _candidate(
            "next-band", tendency="comfort", relevance=0.4, e_score=0.50,
            relevance_band=1,
        ),
    ]

    proposal = propose_chord_reorder(
        rows,
        _parse(),
        near_tie_epsilon=0.03,
    )

    assert proposal.c_ids == proposal.b_ids
    assert "relevance_band" in proposal.skipped_reasons


def test_removed_original_band_leader_cannot_merge_two_frozen_bands():
    from e_chord_shadow import frozen_relevance_band_ids

    original = [
        _candidate("removed-leader", relevance=1.0, relevance_band=99),
        _candidate("original-band-0", relevance=0.7, relevance_band=99),
        _candidate(
            "original-band-1", relevance=0.4, tendency="comfort",
            relevance_band=99,
        ),
    ]
    frozen = frozen_relevance_band_ids(original, 0.35)
    for row in original:
        row["_e_chord_relevance_band_id"] = frozen[row["id"]]

    proposal = propose_chord_reorder(original[1:], _parse())

    assert frozen == {
        "removed-leader": 0,
        "original-band-0": 0,
        "original-band-1": 1,
    }
    assert proposal.c_ids == proposal.b_ids
    assert "relevance_band" in proposal.skipped_reasons


def test_at_most_one_adjacent_swap_per_event_lock_and_one_position_per_candidate():
    rows = [
        _candidate("a", tendency="engage", e_score=0.52),
        _candidate("b", tendency="comfort", e_score=0.51),
        _candidate("c", tendency="comfort", e_score=0.50),
        _candidate("d", event="event-2", tendency="engage", e_score=0.52),
        _candidate("e", event="event-2", tendency="comfort", e_score=0.51),
    ]

    proposal = propose_chord_reorder(rows, _parse(), near_tie_epsilon=0.03)

    assert proposal.c_ids == ("b", "a", "c", "e", "d")
    assert len(proposal.swaps) == 2
    assert proposal.violations == ()
    for index, bucket_id in enumerate(proposal.b_ids):
        assert abs(proposal.c_ids.index(bucket_id) - index) <= 1


def test_receipt_has_same_frozen_pool_and_no_private_text_or_session_identifiers():
    b_rows = [
        _candidate("plain", tendency="engage", e_score=0.52),
        _candidate("comfort", tendency="comfort", e_score=0.50),
    ]
    proposal = propose_chord_reorder(b_rows, _parse(), near_tie_epsilon=0.03)
    receipt = build_shadow_receipt(
        chord=_parse(),
        payload_status="accepted",
        a_candidates=list(reversed(b_rows)),
        b_candidates=b_rows,
        proposal=proposal,
        attempt_index=0,
        first_screen_limit=2,
        request_path_delta_ms=0.25,
        recorded_at_ms=NOW_MS,
    )

    assert receipt["shadow_only"] is True
    assert receipt["affects_ranking"] is False
    assert receipt["diagnostics"]["external_api_delta"] == 0
    assert receipt["diagnostics"]["same_candidate_pool"] is True
    assert receipt["arms"]["b"] == ["plain", "comfort"]
    assert receipt["arms"]["c"] == ["comfort", "plain"]
    raw = json.dumps(receipt, ensure_ascii=False, allow_nan=False)
    for forbidden in (
        "turn-abc123",
        "session_scope",
        "user_text",
        "assistant_text",
        "query",
        "attachment",
        "selfcheck",
    ):
        assert forbidden not in raw


def test_receipt_validator_rejects_tampered_candidate_policy_guards():
    from e_chord_shadow import validate_shadow_receipt

    rows = [
        _candidate("plain", tendency="engage", e_score=0.52),
        _candidate("comfort", tendency="comfort", e_score=0.50),
    ]
    chord = _parse()
    receipt = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        a_candidates=rows,
        b_candidates=rows,
        proposal=propose_chord_reorder(rows, chord, near_tie_epsilon=0.03),
        attempt_index=0,
        first_screen_limit=1,
        request_path_delta_ms=0.1,
        recorded_at_ms=NOW_MS,
    )

    tampered = copy.deepcopy(receipt)
    tampered["candidate_guards"][1]["is_factual"] = True
    with pytest.raises(ValueError, match="receipt contract"):
        validate_shadow_receipt(tampered)


def test_private_ledger_is_append_only_and_mode_hardened(tmp_path):
    chord = _parse()
    rows = [_candidate("one")]
    proposal = propose_chord_reorder(rows, chord)
    receipt = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        a_candidates=rows,
        b_candidates=rows,
        proposal=proposal,
        attempt_index=0,
        first_screen_limit=1,
        request_path_delta_ms=0.1,
        recorded_at_ms=NOW_MS,
    )
    path = tmp_path / ".axis" / "e-chord-shadow.jsonl"
    ledger = EChordShadowLedger(path)

    ledger.append(receipt)

    assert ledger.load() == [receipt]
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(ledger.lock_path).st_mode & 0o777 == 0o600


def test_private_ledger_serializes_concurrent_background_appends(tmp_path):
    chord = _parse()
    rows = [_candidate("one")]
    receipt = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        a_candidates=rows,
        b_candidates=rows,
        proposal=propose_chord_reorder(rows, chord),
        attempt_index=0,
        first_screen_limit=1,
        request_path_delta_ms=0.1,
        recorded_at_ms=NOW_MS,
    )
    ledger = EChordShadowLedger(tmp_path / ".axis" / "e-chord-shadow.jsonl")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(ledger.append, [copy.deepcopy(receipt) for _ in range(24)]))

    assert len(ledger.load()) == 24


def test_ledger_rejects_raw_text_and_nonfinite_receipts(tmp_path):
    ledger = EChordShadowLedger(tmp_path / ".axis" / "e-chord-shadow.jsonl")
    with pytest.raises(ValueError, match="receipt contract"):
        ledger.append({"query": "private"})

    chord = _parse()
    rows = [_candidate("one")]
    valid = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        a_candidates=rows,
        b_candidates=rows,
        proposal=propose_chord_reorder(rows, chord),
        attempt_index=0,
        first_screen_limit=1,
        request_path_delta_ms=0.1,
        recorded_at_ms=NOW_MS,
    )
    nested_leak = copy.deepcopy(valid)
    nested_leak["swaps"] = [{"query": "private"}]
    with pytest.raises(ValueError, match="receipt contract"):
        ledger.append(nested_leak)


def test_final_selection_ledger_is_private_idempotent_and_conflict_safe(tmp_path):
    path = tmp_path / ".axis" / "e-chord-final-selection.jsonl"
    ledger = EChordFinalSelectionLedger(path)
    selection = _final_selection()

    assert ledger.append(selection) is True
    assert ledger.append(copy.deepcopy(selection)) is False
    assert ledger.load() == [selection]
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    assert os.stat(path).st_mode & 0o777 == 0o600

    conflicting = _final_selection(
        final_injected_ids=["one"],
        arms={"a": ["one"], "b": ["one"], "c": ["one"]},
    )
    with pytest.raises(ValueError, match="conflicting E chord final selection"):
        ledger.append(conflicting)


def test_final_selection_validator_binds_actual_injection_and_pool_boundary():
    assert validate_final_selection(_final_selection())["arms"]["b"] == []

    mismatched = _final_selection(final_injected_ids=["one"])
    with pytest.raises(ValueError, match="final selection contract"):
        validate_final_selection(mismatched)

    outside = _final_selection(
        final_injected_ids=["other"],
        outside_pool_ids=["other"],
    )
    assert validate_final_selection(outside)["outside_pool_ids"] == ["other"]


def test_server_shadow_hook_keeps_served_b_and_writes_one_text_free_receipt(
    monkeypatch,
):
    import server

    written = []

    class _Ledger:
        def append(self, row):
            written.append(copy.deepcopy(row))

    async def _inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    rows = [
        _candidate("plain", tendency="engage", e_score=0.52),
        _candidate("comfort", tendency="comfort", e_score=0.50),
    ]
    before = copy.deepcopy(rows)
    payload = _payload(captured_at_ms=int(server.time.time() * 1000))
    monkeypatch.setitem(server.config, "e_chord_shadow", {
        "enabled": True,
        "mode": "shadow",
        "near_tie_epsilon": 0.03,
        "max_age_ms": 30_000,
    })
    monkeypatch.setattr(server, "_get_e_chord_shadow_ledger", lambda: _Ledger())
    monkeypatch.setattr(server.asyncio, "to_thread", _inline_to_thread)

    async def _run():
        receipt = await server._record_e_chord_shadow(
            raw_chord=payload,
            expected_turn_id="turn-abc123",
            expected_agent_id="claude",
            b_candidates=rows,
            chord_config=load_e_chord_shadow_config(server.config),
            prelude_elapsed_ms=0.0,
            attempt_index=0,
            first_screen_limit=2,
        )
        if server._e_chord_shadow_write_tasks:
            await asyncio.gather(*tuple(server._e_chord_shadow_write_tasks))
        return receipt

    receipt = asyncio.run(_run())

    assert rows == before
    assert receipt is not None
    assert receipt["arms"]["b"] == ["plain", "comfort"]
    assert receipt["arms"]["c"] == ["comfort", "plain"]
    assert written == [receipt]
    assert written[0]["diagnostics"]["external_api_delta"] == 0


def test_server_shadow_hook_rejects_cross_turn_projection_without_writing(
    monkeypatch,
):
    import server

    written = []

    class _Ledger:
        def append(self, row):
            written.append(row)

    async def _inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setitem(server.config, "e_chord_shadow", {
        "enabled": True,
        "mode": "shadow",
        "near_tie_epsilon": 0.03,
        "max_age_ms": 30_000,
    })
    monkeypatch.setattr(server, "_get_e_chord_shadow_ledger", lambda: _Ledger())
    monkeypatch.setattr(server.asyncio, "to_thread", _inline_to_thread)

    receipt = asyncio.run(server._record_e_chord_shadow(
        raw_chord=_payload(captured_at_ms=int(server.time.time() * 1000)),
        expected_turn_id="another-turn",
        expected_agent_id="claude",
        b_candidates=[_candidate("one")],
        chord_config=load_e_chord_shadow_config(server.config),
        prelude_elapsed_ms=0.0,
        attempt_index=0,
        first_screen_limit=1,
    ))

    assert receipt is None
    assert written == []


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def test_recall_receipt_records_empty_final_selection_without_activation(
    monkeypatch,
):
    import server

    written = []

    class _Ledger:
        def append(self, row):
            written.append(copy.deepcopy(row))
            return True

    async def _inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    selection = _final_selection()
    monkeypatch.setattr(
        server,
        "_get_e_chord_final_selection_ledger",
        lambda: _Ledger(),
    )
    monkeypatch.setattr(server.asyncio, "to_thread", _inline_to_thread)

    response = asyncio.run(server.api_recall_receipt(_JsonRequest({
        "event_id": "event-empty-selection",
        "bucket_ids": [],
        "source": "twin_prompt_injection",
        "e_chord_selection": selection,
    })))
    payload = json.loads(bytes(response.body))

    assert response.status_code == 200
    assert payload["status"] == "complete"
    assert payload["selection_recorded"] is True
    assert payload["applied"] == 0
    assert written == [selection]


def test_recall_receipt_rejects_selection_not_matching_final_ids(monkeypatch):
    import server

    class _Ledger:
        def append(self, _row):
            raise AssertionError("mismatched selection must not be written")

    monkeypatch.setattr(
        server,
        "_get_e_chord_final_selection_ledger",
        lambda: _Ledger(),
    )
    response = asyncio.run(server.api_recall_receipt(_JsonRequest({
        "event_id": "event-mismatch",
        "bucket_ids": ["one"],
        "e_chord_selection": _final_selection(),
    })))

    assert response.status_code == 400
    assert "does not match" in json.loads(bytes(response.body))["error"]


def test_http_breath_forwards_projection_on_the_existing_call(monkeypatch):
    import server

    observed = {}
    payload = _payload(captured_at_ms=int(server.time.time() * 1000))
    shadow_receipt = {"projection_digest": "d" * 64, "attempt_index": 0}

    async def fake_breath(**kwargs):
        observed.update(kwargs)
        server._e_chord_shadow_response_capture.get()["receipt"] = shadow_receipt
        return "unchanged recall"

    monkeypatch.setattr(server, "breath", fake_breath)
    response = asyncio.run(server.api_breath(_JsonRequest({
        "query": "same natural phrase",
        "live_chord": payload,
        "turn_id": "turn-abc123",
        "agent_id": "claude",
    })))

    assert response.status_code == 200
    assert observed["live_chord"] is payload
    assert observed["turn_id"] == "turn-abc123"
    assert observed["agent_id"] == "claude"
    response_payload = json.loads(bytes(response.body))
    assert response_payload["raw"] == "unchanged recall"
    assert response_payload["e_chord_shadow"] == shadow_receipt
