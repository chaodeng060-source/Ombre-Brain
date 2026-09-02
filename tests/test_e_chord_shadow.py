import asyncio
import copy
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from e_chord_shadow import (
    BYPASS_FINAL_SELECTION_SCHEMA,
    BYPASS_RECEIPT_SCHEMA,
    EChordFinalSelectionLedger,
    EChordShadowLedger,
    FINAL_SELECTION_SCHEMA,
    LEGACY_FINAL_SELECTION_SCHEMA,
    LEGACY_RECEIPT_SCHEMA,
    build_shadow_receipt,
    load_e_chord_shadow_config,
    parse_live_chord,
    propose_chord_reorder,
    select_bypass_candidates,
    session_scope_digest,
    validate_final_selection,
    validate_shadow_receipt,
)


NOW_MS = 1_777_777_777_000
TEST_SESSION_ID = "claude:dm:e-chord-test"


def _payload(**overrides):
    value = {
        "schema": "live_chord.v1",
        "turn_id": "turn-abc123",
        "agent_id": "claude",
        "e_authored_by": "哥哥",
        "session_scope": session_scope_digest(TEST_SESSION_ID),
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
        expected_session_id=TEST_SESSION_ID,
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
    query_valence=0.0,
    experience_valence=0.0,
    e_resonance=0.8,
    source="",
    priority=50,
    recorded_at="2026-09-02T12:00:00+08:00",
):
    metadata = {
        "event_id": event,
        "type": "experience",
        "recorded_at": recorded_at,
    }
    if source:
        metadata["e_source_bucket_id"] = source
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
            "valence": experience_valence,
            "initial_priority": priority,
        },
        "_e_axis_query_valence": query_valence,
        "_e_axis_resonance": e_resonance,
        "_e_axis_admissibility_floor": 0.55,
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
        "final_input_cohort_ids": ["one"],
        "final_input_cohort_status": "pure_same_cohort",
        "final_injected_ids": [],
        "outside_pool_ids": [],
        "arms": {"a": [], "b": [], "c": []},
        "applied_swaps": [],
        "request_path_delta_ms": 0.25,
    }
    value.update(overrides)
    if "final_input_cohort_ids" not in overrides:
        value["final_input_cohort_ids"] = list(value["pool_ids"])
    if "final_input_cohort_status" not in overrides:
        value["final_input_cohort_status"] = "pure_same_cohort"
    return value


def test_parser_accepts_only_bounded_text_free_contract():
    chord = _parse()

    assert len(chord.facets) == 2
    assert chord.facets[0].tendency == "comfort"
    assert chord.e_authored_by == "哥哥"
    assert chord.source_turn_digest == "b" * 64
    assert chord.session_scope == "f5dab9a801938a8f5e7947a5d224f4bd"
    assert not hasattr(chord, "query")


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda row: row.update(query="private text"), "schema.keys"),
        (lambda row: row.update(turn_id="other-turn"), "scope.turn"),
        (lambda row: row.update(agent_id="hajimi"), "scope.agent"),
        (lambda row: row.update(e_authored_by="xiaojuan"), "scope.author"),
        (lambda row: row.update(session_scope="dm:claude"), "schema.session_scope"),
        (lambda row: row.update(session_scope="c" * 32), "scope.session"),
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
        expected_session_id=TEST_SESSION_ID,
        now_ms=NOW_MS,
        max_age_ms=30_000,
    )

    assert chord is None
    assert observed == reason


def test_config_is_default_off_and_active_contract_is_bounded(monkeypatch):
    monkeypatch.delenv("OMBRE_E_CHORD_DERIVED_LOCK", raising=False)
    monkeypatch.delenv("OMBRE_E_CHORD_BYPASS", raising=False)
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
    assert active.derived_lock_enabled is False
    assert active.bypass_enabled is False
    assert active.bypass_limit == 4
    monkeypatch.setenv("OMBRE_E_CHORD_DERIVED_LOCK", "1")
    assert load_e_chord_shadow_config({
        "e_chord_shadow": {"enabled": True, "mode": "shadow"}
    }).derived_lock_enabled is True
    monkeypatch.setenv("OMBRE_E_CHORD_BYPASS", "1")
    bypass = load_e_chord_shadow_config({
        "e_chord_shadow": {
            "enabled": True,
            "mode": "shadow",
            "bypass_limit": 8,
        }
    })
    assert bypass.bypass_enabled is True
    assert bypass.bypass_limit == 8
    with pytest.raises(ValueError, match="near_tie_epsilon"):
        load_e_chord_shadow_config({
            "e_chord_shadow": {
                "enabled": True,
                "mode": "shadow",
                "near_tie_epsilon": math.nan,
            }
        })


def test_bypass_selection_requires_source_admissibility_and_respects_cap():
    natural = [_candidate("natural")]
    sources = {
        "source-a": {"id": "source-a", "metadata": {}},
        "source-b": {"id": "source-b", "metadata": {}},
        "source-c": {"id": "source-c", "metadata": {}},
    }
    candidates = [
        _candidate(
            "older-high",
            source="source-a",
            priority=90,
            recorded_at="2026-09-01T12:00:00+08:00",
        ),
        _candidate(
            "newer-high",
            source="source-b",
            priority=90,
            recorded_at="2026-09-02T12:00:00+08:00",
        ),
        _candidate("lower", source="source-c", priority=80),
        _candidate("missing-source", priority=100),
        _candidate("unknown-source", source="not-loaded", priority=100),
        _candidate(
            "opposite",
            source="source-c",
            priority=100,
            query_valence=-0.9,
            experience_valence=0.9,
        ),
        _candidate("natural", source="source-a", priority=100),
    ]

    selected = select_bypass_candidates(
        candidates,
        natural,
        source_buckets_by_id=sources,
        limit=2,
    )

    assert [candidate["id"] for candidate in selected] == [
        "newer-high",
        "older-high",
    ]


def test_bypass_segment_swaps_internally_but_never_across_boundary():
    rows = [
        _candidate("natural", tendency="engage", e_score=0.52, relevance_band=1),
        _candidate(
            "bypass-left",
            tendency="engage",
            e_score=0.52,
            relevance_band=1,
            source="source-left",
        ),
        _candidate(
            "bypass-right",
            tendency="comfort",
            e_score=0.50,
            relevance_band=1,
            source="source-right",
        ),
    ]

    proposal = propose_chord_reorder(
        rows,
        _parse(),
        bypass_ids=("bypass-left", "bypass-right"),
    )

    assert proposal.c_ids == ("natural", "bypass-right", "bypass-left")
    assert [swap.promoted_id for swap in proposal.swaps] == ["bypass-right"]
    assert "bypass_boundary" in proposal.skipped_reasons
    assert proposal.violations == ()


def test_bypass_boundary_violation_clears_the_whole_proposal():
    rows = [
        _candidate("natural", tendency="engage", e_score=0.52),
        _candidate("bypass-left", tendency="engage", e_score=0.52),
        _candidate("bypass-right", tendency="comfort", e_score=0.50),
    ]

    proposal = propose_chord_reorder(
        rows,
        _parse(),
        bypass_ids=("bypass-left",),
    )

    assert proposal.c_ids == proposal.b_ids
    assert proposal.swaps == ()
    assert proposal.violations == ("bypass_boundary",)


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


def test_same_e_source_bucket_is_an_explicit_event_lock():
    rows = [
        _candidate("plain", tendency="engage", e_score=0.52),
        _candidate("comfort", tendency="comfort", e_score=0.50),
    ]
    for row in rows:
        row["metadata"].pop("event_id")
        row["metadata"]["e_source_bucket_id"] = "source-memory-1"

    proposal = propose_chord_reorder(rows, _parse(), near_tie_epsilon=0.03)

    assert proposal.b_ids == ("plain", "comfort")
    assert proposal.c_ids == ("comfort", "plain")
    assert proposal.swaps[0].promoted_id == "comfort"
    assert proposal.violations == ()


def test_event_and_e_source_bucket_lock_namespaces_do_not_cross_match():
    left = _candidate("left", event="shared-value", e_score=0.52)
    right = _candidate("right", tendency="comfort", e_score=0.50)
    right["metadata"].pop("event_id")
    right["metadata"]["e_source_bucket_id"] = "shared-value"

    proposal = propose_chord_reorder([left, right], _parse(), near_tie_epsilon=0.03)

    assert proposal.c_ids == proposal.b_ids
    assert "event_lock" in proposal.skipped_reasons


def _derived_candidates():
    left = _candidate("left-e", event="left-strong", e_score=0.52)
    right = _candidate(
        "right-e",
        event="right-strong",
        tendency="comfort",
        e_score=0.50,
    )
    left["metadata"]["e_source_bucket_id"] = "source-left"
    right["metadata"]["e_source_bucket_id"] = "source-right"
    return left, right


def test_derived_relation_lock_is_opt_in_deterministic_and_auditable():
    left, right = _derived_candidates()
    sources = {
        "source-left": {
            "id": "source-left",
            "metadata": {
                "recorded_at": "2026-09-01T10:00:00+08:00",
                "domain": ["left-only"],
                "relations": [{
                    "type": "explains",
                    "target": "source-right",
                    "strength": 0.8,
                    "generation_method": "deterministic:explicit-reference:v1",
                }],
            },
        },
        "source-right": {
            "id": "source-right",
            "metadata": {
                "recorded_at": "2026-09-02T10:00:00+08:00",
                "domain": ["right-only"],
            },
        },
    }

    disabled = propose_chord_reorder(
        [left, right],
        _parse(),
        source_buckets_by_id=sources,
    )
    enabled = propose_chord_reorder(
        [left, right],
        _parse(),
        derived_lock_enabled=True,
        source_buckets_by_id=sources,
        allowed_relation_types=("explains",),
        relation_min_strength=0.4,
    )

    assert disabled.c_ids == disabled.b_ids
    assert enabled.c_ids == ("right-e", "left-e")
    swap = enabled.swaps[0]
    assert swap.lock_kind == "derived"
    assert swap.derived_lock_basis == "relation"
    assert swap.source_bucket_ids == ("source-left", "source-right")
    assert swap.relation_type == "explains"
    assert (swap.relation_from_id, swap.relation_to_id) == (
        "source-left", "source-right",
    )
    assert len(swap.event_lock_digest) == 16


def test_derived_lock_disabled_is_full_proposal_equivalent():
    left, right = _derived_candidates()
    sources = {
        "source-left": {
            "id": "source-left",
            "metadata": {
                "relations": [{
                    "type": "explains",
                    "target": "source-right",
                    "strength": 0.9,
                    "generation_method": "deterministic:explicit-reference:v1",
                }],
            },
        },
        "source-right": {"id": "source-right", "metadata": {}},
    }

    legacy = propose_chord_reorder([left, right], _parse())
    disabled = propose_chord_reorder(
        [left, right],
        _parse(),
        derived_lock_enabled=False,
        source_buckets_by_id=sources,
        allowed_relation_types=("explains",),
        relation_min_strength=0.4,
    )

    assert disabled == legacy


def test_derived_same_day_domain_lock_and_strong_lock_precedence():
    left, right = _derived_candidates()
    sources = {
        source_id: {
            "id": source_id,
            "metadata": {
                "recorded_at": timestamp,
                "domain": domains,
            },
        }
        for source_id, timestamp, domains in (
            ("source-left", "2026-09-02T01:00:00+08:00", ["relationship", "x"]),
            ("source-right", "2026-09-02T23:00:00+08:00", ["relationship", "y"]),
        )
    }
    derived = propose_chord_reorder(
        [left, right],
        _parse(),
        derived_lock_enabled=True,
        source_buckets_by_id=sources,
    )
    assert derived.swaps[0].lock_kind == "derived"
    assert derived.swaps[0].derived_lock_basis == "same_day_domain"
    assert derived.swaps[0].recorded_day == "2026-09-02"
    assert len(derived.swaps[0].domain_digest) == 16

    left["metadata"]["event_id"] = "shared-strong"
    right["metadata"]["event_id"] = "shared-strong"
    strong = propose_chord_reorder(
        [left, right],
        _parse(),
        derived_lock_enabled=True,
        source_buckets_by_id=sources,
    )
    assert strong.swaps[0].lock_kind == "strong"
    assert strong.swaps[0].source_bucket_ids == ()


def test_derived_lock_rejects_model_edge_and_unrelated_day_domain():
    left, right = _derived_candidates()
    sources = {
        "source-left": {
            "id": "source-left",
            "metadata": {
                "recorded_at": "2026-09-01T10:00:00+08:00",
                "domain": ["left-only"],
                "relations": [{
                    "type": "explains",
                    "target": "source-right",
                    "strength": 0.9,
                    "generation_method": "model:relation:v1",
                }],
            },
        },
        "source-right": {
            "id": "source-right",
            "metadata": {
                "recorded_at": "2026-09-02T10:00:00+08:00",
                "domain": ["right-only"],
            },
        },
    }
    proposal = propose_chord_reorder(
        [left, right],
        _parse(),
        derived_lock_enabled=True,
        source_buckets_by_id=sources,
    )
    assert proposal.c_ids == proposal.b_ids
    assert "event_lock" in proposal.skipped_reasons


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


def test_same_event_opposite_affect_near_tie_is_not_e_admissible():
    rows = [
        _candidate(
            "correct-affect",
            tendency="engage",
            e_score=0.51,
            query_valence=-0.8,
            experience_valence=-0.7,
            e_resonance=0.9,
        ),
        _candidate(
            "opposite-affect",
            tendency="comfort",
            e_score=0.50,
            query_valence=-0.8,
            experience_valence=0.8,
            e_resonance=0.2,
        ),
    ]

    proposal = propose_chord_reorder(rows, _parse(), near_tie_epsilon=0.03)

    assert proposal.c_ids == proposal.b_ids
    assert "e_admissibility" in proposal.skipped_reasons


def test_chord_cannot_promote_materially_weaker_existing_e_resonance():
    rows = [
        _candidate(
            "stronger-e",
            tendency="engage",
            e_score=0.51,
            query_valence=-0.8,
            experience_valence=-0.7,
            e_resonance=0.90,
        ),
        _candidate(
            "weaker-e",
            tendency="comfort",
            e_score=0.50,
            query_valence=-0.8,
            experience_valence=-0.6,
            e_resonance=0.70,
        ),
    ]

    proposal = propose_chord_reorder(rows, _parse(), near_tie_epsilon=0.03)

    assert proposal.c_ids == proposal.b_ids
    assert "e_resonance" in proposal.skipped_reasons


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
        pre_e_cohort_ids=["comfort", "plain"],
        post_e_cohort_ids=["plain", "comfort"],
        ds_decision_source="disabled",
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
    assert receipt["swaps"][0]["lock_kind"] == "strong"
    assert receipt["swaps"][0]["source_bucket_ids"] == []
    assert all("e_source_bucket_id" in guard for guard in receipt["candidate_guards"])
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


def test_bypass_receipt_v3_binds_suffix_sources_origins_and_stays_text_free():
    natural = _candidate("natural", tendency="engage", e_score=0.52)
    bypass_left = _candidate(
        "bypass-left",
        tendency="engage",
        e_score=0.52,
        relevance_band=1,
        source="source-left",
        priority=90,
    )
    bypass_right = _candidate(
        "bypass-right",
        tendency="comfort",
        e_score=0.50,
        relevance_band=1,
        source="source-right",
        priority=80,
    )
    rows = [natural, bypass_left, bypass_right]
    chord = _parse()
    proposal = propose_chord_reorder(
        rows,
        chord,
        bypass_ids=("bypass-left", "bypass-right"),
    )
    receipt = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        pre_e_cohort_ids=["natural"],
        post_e_cohort_ids=["natural"],
        ds_decision_source="disabled",
        a_candidates=[natural],
        b_candidates=rows,
        proposal=proposal,
        attempt_index=0,
        first_screen_limit=1,
        request_path_delta_ms=0.2,
        recorded_at_ms=NOW_MS,
        bypass_enabled=True,
        bypass_ids=("bypass-left", "bypass-right"),
        bypass_limit=4,
    )

    assert receipt["schema"] == BYPASS_RECEIPT_SCHEMA
    assert receipt["bypass_ids"] == ["bypass-left", "bypass-right"]
    assert receipt["bypass_source_ids"] == ["source-left", "source-right"]
    assert receipt["bypass_limit"] == 4
    assert [guard["origin"] for guard in receipt["candidate_guards"]] == [
        "natural",
        "bypass",
        "bypass",
    ]
    assert receipt["arms"]["a"] == ["natural", "bypass-left", "bypass-right"]
    assert receipt["arms"]["c"] == ["natural", "bypass-right", "bypass-left"]
    assert receipt["diagnostics"]["bypass_boundary"] == 0
    assert receipt["diagnostics"]["external_api_delta"] == 0
    assert validate_shadow_receipt(receipt) is receipt
    encoded = json.dumps(receipt, ensure_ascii=False, allow_nan=False)
    assert "content" not in encoded
    assert "query" not in encoded


def test_bypass_disabled_is_byte_for_byte_v2_equivalent():
    rows = [
        _candidate("plain", tendency="engage", e_score=0.52),
        _candidate("comfort", tendency="comfort", e_score=0.50),
    ]
    chord = _parse()
    proposal = propose_chord_reorder(rows, chord)
    common = {
        "chord": chord,
        "payload_status": "accepted",
        "pre_e_cohort_ids": ["plain", "comfort"],
        "post_e_cohort_ids": ["plain", "comfort"],
        "ds_decision_source": "disabled",
        "a_candidates": rows,
        "b_candidates": rows,
        "proposal": proposal,
        "attempt_index": 0,
        "first_screen_limit": 2,
        "request_path_delta_ms": 0.2,
        "recorded_at_ms": NOW_MS,
    }

    before = build_shadow_receipt(**common)
    disabled = build_shadow_receipt(
        **common,
        bypass_enabled=False,
        bypass_ids=(),
        bypass_limit=4,
    )

    assert before["schema"] == "e_chord_shadow_receipt.v2"
    assert json.dumps(before, sort_keys=True, separators=(",", ":")) == json.dumps(
        disabled,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_bypass_v3_validator_rejects_cross_boundary_swap():
    natural = _candidate("natural", tendency="engage", e_score=0.52)
    bypass = _candidate(
        "bypass",
        tendency="comfort",
        e_score=0.50,
        source="source-bypass",
    )
    rows = [natural, bypass]
    chord = _parse()
    safe_proposal = propose_chord_reorder(
        rows,
        chord,
        bypass_ids=("bypass",),
    )
    receipt = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        pre_e_cohort_ids=["natural"],
        post_e_cohort_ids=["natural"],
        ds_decision_source="disabled",
        a_candidates=[natural],
        b_candidates=rows,
        proposal=safe_proposal,
        attempt_index=0,
        first_screen_limit=1,
        request_path_delta_ms=0.1,
        recorded_at_ms=NOW_MS,
        bypass_enabled=True,
        bypass_ids=("bypass",),
        bypass_limit=4,
    )
    forged = copy.deepcopy(receipt)
    forged["swaps"] = [{
        "promoted_id": "bypass",
        "demoted_id": "natural",
        "from_index": 1,
        "to_index": 0,
        "event_lock_digest": next(iter(
            set(forged["candidate_guards"][0]["event_lock_digests"])
            & set(forged["candidate_guards"][1]["event_lock_digests"])
        )),
        "lock_kind": "strong",
        "source_bucket_ids": [],
        "derived_lock_basis": "",
        "relation_type": "",
        "relation_from_id": "",
        "relation_to_id": "",
        "recorded_day": "",
        "domain_digest": "",
    }]
    forged["arms"]["c"] = ["bypass", "natural"]
    forged["first_screen"]["c"] = ["bypass"]
    forged["diagnostics"]["max_displacement"] = 1

    with pytest.raises(ValueError, match="receipt contract"):
        validate_shadow_receipt(forged)


def test_derived_receipt_binds_candidate_sources_and_legacy_v1_still_loads():
    left, right = _derived_candidates()
    sources = {
        source_id: {
            "id": source_id,
            "metadata": {
                "recorded_at": "2026-09-02T12:00:00+08:00",
                "domain": ["relationship"],
            },
        }
        for source_id in ("source-left", "source-right")
    }
    chord = _parse()
    proposal = propose_chord_reorder(
        [left, right],
        chord,
        derived_lock_enabled=True,
        source_buckets_by_id=sources,
    )
    receipt = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        pre_e_cohort_ids=["left-e", "right-e"],
        post_e_cohort_ids=["left-e", "right-e"],
        ds_decision_source="disabled",
        a_candidates=[left, right],
        b_candidates=[left, right],
        proposal=proposal,
        attempt_index=0,
        first_screen_limit=2,
        request_path_delta_ms=0.1,
        recorded_at_ms=NOW_MS,
    )
    assert validate_shadow_receipt(receipt) is receipt
    assert receipt["swaps"][0]["lock_kind"] == "derived"

    tampered = copy.deepcopy(receipt)
    tampered["swaps"][0]["source_bucket_ids"] = ["source-left", "source-other"]
    with pytest.raises(ValueError, match="receipt contract"):
        validate_shadow_receipt(tampered)

    malformed = copy.deepcopy(receipt)
    malformed["swaps"][0]["relation_from_id"] = []
    with pytest.raises(ValueError, match="receipt contract"):
        validate_shadow_receipt(malformed)

    legacy = copy.deepcopy(receipt)
    legacy["schema"] = LEGACY_RECEIPT_SCHEMA
    for guard in legacy["candidate_guards"]:
        guard.pop("e_source_bucket_id")
    legacy["swaps"] = []
    legacy["arms"]["c"] = list(legacy["arms"]["b"])
    legacy["first_screen"]["c"] = list(legacy["first_screen"]["b"])
    legacy["diagnostics"]["max_displacement"] = 0
    assert validate_shadow_receipt(legacy) is legacy


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
        pre_e_cohort_ids=["plain", "comfort"],
        post_e_cohort_ids=["plain", "comfort"],
        ds_decision_source="disabled",
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

    tampered = copy.deepcopy(receipt)
    tampered["candidate_guards"][1]["e_admissibility"] = "opposite_affect"
    with pytest.raises(ValueError, match="receipt contract"):
        validate_shadow_receipt(tampered)


def test_a_cohort_fails_closed_for_model_ds_and_membership_drift():
    from e_chord_shadow import validate_shadow_receipt

    rows = [
        _candidate("plain", tendency="engage", e_score=0.52),
        _candidate("comfort", tendency="comfort", e_score=0.50),
    ]
    chord = _parse()
    proposal = propose_chord_reorder(rows, chord, near_tie_epsilon=0.03)

    model = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        pre_e_cohort_ids=["comfort", "plain"],
        post_e_cohort_ids=["plain", "comfort"],
        ds_decision_source="model",
        a_candidates=list(reversed(rows)),
        b_candidates=rows,
        proposal=proposal,
        attempt_index=0,
        first_screen_limit=1,
        request_path_delta_ms=0.1,
        recorded_at_ms=NOW_MS,
    )
    assert model["a_cohort_status"] == "unscorable_ds_model"
    assert model["arms"]["a"] == model["arms"]["b"]
    assert validate_shadow_receipt(model) is model

    drift = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        pre_e_cohort_ids=["comfort", "plain", "dropped"],
        post_e_cohort_ids=["plain", "comfort"],
        ds_decision_source="disabled",
        a_candidates=list(reversed(rows)),
        b_candidates=rows,
        proposal=proposal,
        attempt_index=0,
        first_screen_limit=1,
        request_path_delta_ms=0.1,
        recorded_at_ms=NOW_MS,
    )
    assert drift["a_cohort_status"] == "unscorable_cohort_drift"
    assert drift["arms"]["a"] == drift["arms"]["b"]
    assert validate_shadow_receipt(drift) is drift


def test_a_cohort_fails_closed_when_a_downstream_gate_only_reorders_b():
    from e_chord_shadow import _a_cohort_status

    assert _a_cohort_status(
        ["current", "other", "historical"],
        ["current", "other", "historical"],
        ["historical", "current", "other"],
        "disabled",
    ) == "unscorable_downstream_order"


@pytest.mark.asyncio
async def test_existing_ds_call_records_decision_source_without_a_second_call(
    monkeypatch,
):
    import server

    rows = [_candidate("a"), _candidate("b")]
    capture = {}
    token = server._ds_filter_decision_capture.set(capture)
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "0")
    try:
        await server._ds_filter_candidates(
            "same query",
            rows,
            mode="search",
            max_results=2,
        )
    finally:
        server._ds_filter_decision_capture.reset(token)
    assert capture == {"source": "disabled"}

    calls = 0

    async def one_model_decision(_query, buckets, _keep, _max_results):
        nonlocal calls
        calls += 1
        return buckets

    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")
    monkeypatch.setattr(server, "_ds_semantic_select", one_model_decision)
    capture = {}
    token = server._ds_filter_decision_capture.set(capture)
    try:
        await server._ds_filter_candidates(
            "same query",
            rows,
            mode="search",
            max_results=2,
        )
    finally:
        server._ds_filter_decision_capture.reset(token)
    assert calls == 1
    assert capture == {"source": "model"}


def test_private_ledger_is_append_only_and_mode_hardened(tmp_path):
    chord = _parse()
    rows = [_candidate("one")]
    proposal = propose_chord_reorder(rows, chord)
    receipt = build_shadow_receipt(
        chord=chord,
        payload_status="accepted",
        pre_e_cohort_ids=["one"],
        post_e_cohort_ids=["one"],
        ds_decision_source="disabled",
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
        pre_e_cohort_ids=["one"],
        post_e_cohort_ids=["one"],
        ds_decision_source="disabled",
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
        pre_e_cohort_ids=["one"],
        post_e_cohort_ids=["one"],
        ds_decision_source="disabled",
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

    drift = _final_selection(
        pool_ids=["one", "two"],
        final_input_cohort_ids=["one"],
        final_input_cohort_status="unscorable_final_cohort_drift",
        final_injected_ids=["one"],
        arms={"a": ["one"], "b": ["one"], "c": ["one"]},
    )
    assert validate_final_selection(drift)["final_input_cohort_ids"] == ["one"]

    forged_status = copy.deepcopy(drift)
    forged_status["final_input_cohort_status"] = "pure_same_cohort"
    with pytest.raises(ValueError, match="final selection contract"):
        validate_final_selection(forged_status)

    fabricated_a = copy.deepcopy(drift)
    fabricated_a["arms"]["a"] = ["two"]
    with pytest.raises(ValueError, match="final selection contract"):
        validate_final_selection(fabricated_a)

    mismatched = _final_selection(final_injected_ids=["one"])
    with pytest.raises(ValueError, match="final selection contract"):
        validate_final_selection(mismatched)

    outside = _final_selection(
        final_injected_ids=["other"],
        outside_pool_ids=["other"],
    )
    assert validate_final_selection(outside)["outside_pool_ids"] == ["other"]


def test_bypass_final_selection_v3_never_accepts_bypass_as_injected():
    selection = _final_selection(
        schema=BYPASS_FINAL_SELECTION_SCHEMA,
        pool_ids=["natural", "bypass-a", "bypass-b"],
        final_input_cohort_ids=["natural"],
        final_input_cohort_status="pure_same_cohort",
        final_injected_ids=["natural"],
        arms={"a": ["natural"], "b": ["natural"], "c": ["natural"]},
        bypass_ids=["bypass-a", "bypass-b"],
        bypass_source_ids=["source-a", "source-b"],
        bypass_limit=4,
    )

    assert validate_final_selection(selection) is selection

    leaked = copy.deepcopy(selection)
    leaked["final_injected_ids"] = ["natural", "bypass-a"]
    leaked["arms"] = {
        "a": ["natural", "bypass-a"],
        "b": ["natural", "bypass-a"],
        "c": ["natural", "bypass-a"],
    }
    with pytest.raises(ValueError, match="final selection contract"):
        validate_final_selection(leaked)


def test_final_selection_validator_reconstructs_only_declared_boundary_swaps():
    swap = {
        "promoted_id": "b",
        "demoted_id": "a",
        "from_index": 1,
        "to_index": 0,
        "event_lock_digest": "0" * 16,
        "lock_kind": "strong",
        "source_bucket_ids": [],
        "derived_lock_basis": "",
        "relation_type": "",
        "relation_from_id": "",
        "relation_to_id": "",
        "recorded_day": "",
        "domain_digest": "",
    }
    valid = _final_selection(
        pool_ids=["a", "b", "c"],
        final_injected_ids=["a", "b", "c"],
        arms={"a": ["a", "b", "c"], "b": ["a", "b", "c"], "c": ["b", "a", "c"]},
        applied_swaps=[swap],
    )
    assert validate_final_selection(valid)["arms"]["c"] == ["b", "a", "c"]

    legacy = copy.deepcopy(valid)
    legacy["schema"] = LEGACY_FINAL_SELECTION_SCHEMA
    legacy["applied_swaps"][0] = {
        key: legacy["applied_swaps"][0][key]
        for key in (
            "promoted_id", "demoted_id", "from_index", "to_index",
            "event_lock_digest",
        )
    }
    assert validate_final_selection(legacy)["arms"]["c"] == ["b", "a", "c"]

    forged = copy.deepcopy(valid)
    forged["arms"]["c"] = ["b", "c", "a"]
    with pytest.raises(ValueError, match="final selection contract"):
        validate_final_selection(forged)

    zero_to_nonzero = _final_selection(
        pool_ids=["a", "b"],
        arms={"a": [], "b": [], "c": ["b"]},
        applied_swaps=[],
    )
    with pytest.raises(ValueError, match="final selection contract"):
        validate_final_selection(zero_to_nonzero)


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
            expected_session_id=TEST_SESSION_ID,
            a_candidates=list(reversed(rows)),
            post_e_candidates=rows,
            b_candidates=rows,
            ds_decision_source="disabled",
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
    assert receipt["arms"]["a"] == ["comfort", "plain"]
    assert receipt["arms"]["b"] == ["plain", "comfort"]
    assert receipt["arms"]["c"] == ["comfort", "plain"]
    assert written == [receipt]
    assert written[0]["diagnostics"]["external_api_delta"] == 0


def test_server_shadow_hook_uses_derived_source_proof_without_external_call(
    monkeypatch,
):
    import server

    written = []

    class _Ledger:
        def append(self, row):
            written.append(copy.deepcopy(row))

    async def _inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    left, right = _derived_candidates()
    rows = [left, right]
    sources = {
        "source-left": {
            "id": "source-left",
            "metadata": {
                "relations": [{
                    "type": "explains",
                    "target": "source-right",
                    "strength": 0.8,
                    "generation_method": "deterministic:explicit-reference:v1",
                }],
            },
        },
        "source-right": {"id": "source-right", "metadata": {}},
    }
    payload = _payload(captured_at_ms=int(server.time.time() * 1000))
    monkeypatch.setenv("OMBRE_E_CHORD_DERIVED_LOCK", "1")
    monkeypatch.setitem(server.config, "e_chord_shadow", {
        "enabled": True,
        "mode": "shadow",
        "near_tie_epsilon": 0.03,
        "max_age_ms": 30_000,
    })
    monkeypatch.setitem(server.config, "relation_recall", {
        "propagation_only": True,
        "propagation_types": ["explains"],
        "hop1_min_strength": 0.4,
    })
    monkeypatch.setattr(server, "_get_e_chord_shadow_ledger", lambda: _Ledger())
    monkeypatch.setattr(server.asyncio, "to_thread", _inline_to_thread)

    async def _run():
        receipt = await server._record_e_chord_shadow(
            raw_chord=payload,
            expected_turn_id="turn-abc123",
            expected_agent_id="claude",
            expected_session_id=TEST_SESSION_ID,
            a_candidates=rows,
            post_e_candidates=rows,
            b_candidates=rows,
            ds_decision_source="disabled",
            chord_config=load_e_chord_shadow_config(server.config),
            prelude_elapsed_ms=0.0,
            attempt_index=0,
            first_screen_limit=2,
            source_buckets_by_id=sources,
        )
        if server._e_chord_shadow_write_tasks:
            await asyncio.gather(*tuple(server._e_chord_shadow_write_tasks))
        return receipt

    receipt = asyncio.run(_run())

    assert receipt is not None
    assert receipt["schema"] == "e_chord_shadow_receipt.v2"
    assert receipt["arms"]["c"] == ["right-e", "left-e"]
    assert receipt["swaps"][0]["lock_kind"] == "derived"
    assert receipt["swaps"][0]["source_bucket_ids"] == [
        "source-left", "source-right",
    ]
    assert receipt["diagnostics"]["external_api_delta"] == 0
    assert written == [receipt]


def test_server_shadow_hook_appends_bypass_only_to_v3_shadow_pool(monkeypatch):
    import server

    written = []

    class _Ledger:
        def append(self, row):
            written.append(copy.deepcopy(row))

    async def _inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    natural = [_candidate("natural", relevance_band=0)]
    bypass = [
        _candidate(
            "bypass-left",
            tendency="engage",
            e_score=0.52,
            relevance_band=1,
            source="source-left",
        ),
        _candidate(
            "bypass-right",
            tendency="comfort",
            e_score=0.50,
            relevance_band=1,
            source="source-right",
        ),
    ]
    natural_before = copy.deepcopy(natural)
    monkeypatch.setenv("OMBRE_E_CHORD_BYPASS", "1")
    monkeypatch.setitem(server.config, "e_chord_shadow", {
        "enabled": True,
        "mode": "shadow",
        "near_tie_epsilon": 0.03,
        "max_age_ms": 30_000,
        "bypass_limit": 4,
    })
    monkeypatch.setattr(server, "_get_e_chord_shadow_ledger", lambda: _Ledger())
    monkeypatch.setattr(server.asyncio, "to_thread", _inline_to_thread)

    async def _run():
        receipt = await server._record_e_chord_shadow(
            raw_chord=_payload(captured_at_ms=int(server.time.time() * 1000)),
            expected_turn_id="turn-abc123",
            expected_agent_id="claude",
            expected_session_id=TEST_SESSION_ID,
            a_candidates=natural,
            post_e_candidates=natural,
            b_candidates=natural,
            bypass_candidates=bypass,
            ds_decision_source="disabled",
            chord_config=load_e_chord_shadow_config(server.config),
            prelude_elapsed_ms=0.0,
            attempt_index=0,
            first_screen_limit=1,
        )
        if server._e_chord_shadow_write_tasks:
            await asyncio.gather(*tuple(server._e_chord_shadow_write_tasks))
        return receipt

    receipt = asyncio.run(_run())

    assert natural == natural_before
    assert receipt is not None
    assert receipt["schema"] == BYPASS_RECEIPT_SCHEMA
    assert receipt["arms"]["b"] == [
        "natural", "bypass-left", "bypass-right",
    ]
    assert receipt["arms"]["c"] == [
        "natural", "bypass-right", "bypass-left",
    ]
    assert receipt["bypass_ids"] == ["bypass-left", "bypass-right"]
    assert written == [receipt]


def test_server_shadow_hook_marks_post_e_state_reorder_unscorable(monkeypatch):
    import server

    written = []

    class _Ledger:
        def append(self, row):
            written.append(copy.deepcopy(row))

    async def _inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    pre_and_post_e = [
        _candidate("current"),
        _candidate("other"),
        _candidate("historical"),
    ]
    post_state_b = [
        pre_and_post_e[2],
        pre_and_post_e[0],
        pre_and_post_e[1],
    ]
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
            raw_chord=_payload(captured_at_ms=int(server.time.time() * 1000)),
            expected_turn_id="turn-abc123",
            expected_agent_id="claude",
            expected_session_id=TEST_SESSION_ID,
            a_candidates=pre_and_post_e,
            post_e_candidates=pre_and_post_e,
            b_candidates=post_state_b,
            ds_decision_source="disabled",
            chord_config=load_e_chord_shadow_config(server.config),
            prelude_elapsed_ms=0.0,
            attempt_index=0,
            first_screen_limit=3,
        )
        if server._e_chord_shadow_write_tasks:
            await asyncio.gather(*tuple(server._e_chord_shadow_write_tasks))
        return receipt

    receipt = asyncio.run(_run())

    assert receipt is not None
    assert receipt["a_cohort_status"] == "unscorable_downstream_order"
    assert receipt["arms"]["a"] == receipt["arms"]["b"]
    assert receipt["post_e_cohort_ids"] == ["current", "other", "historical"]
    assert written == [receipt]


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
        expected_session_id=TEST_SESSION_ID,
        a_candidates=[_candidate("one")],
        post_e_candidates=[_candidate("one")],
        b_candidates=[_candidate("one")],
        ds_decision_source="disabled",
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
        "session_id": TEST_SESSION_ID,
        "live_chord": payload,
        "turn_id": "turn-abc123",
        "agent_id": "claude",
    })))

    assert response.status_code == 200
    assert observed["live_chord"] is payload
    assert observed["turn_id"] == "turn-abc123"
    assert observed["agent_id"] == "claude"
    assert observed["session_id"] == TEST_SESSION_ID
    response_payload = json.loads(bytes(response.body))
    assert response_payload["raw"] == "unchanged recall"
    assert response_payload["e_chord_shadow"] == shadow_receipt
