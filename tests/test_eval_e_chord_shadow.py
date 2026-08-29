import copy
import hashlib
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "eval_e_chord_shadow.py"
SPEC = importlib.util.spec_from_file_location("eval_e_chord_shadow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def _source_digest(digest):
    return hashlib.sha256(f"source:{digest}".encode("utf-8")).hexdigest()


def _receipt(digest, *, pool, a, b, c, latency=0.2, attempt_index=0):
    def _arm(first):
        return list(first) + [bucket_id for bucket_id in pool if bucket_id not in first]

    a_arm = _arm(a)
    b_arm = _arm(b)
    c_arm = _arm(c)
    swaps = []
    if c_arm != b_arm:
        assert len(pool) == 2 and c_arm == list(reversed(b_arm))
        swaps.append({
            "promoted_id": b_arm[1],
            "demoted_id": b_arm[0],
            "from_index": 1,
            "to_index": 0,
            "event_lock_digest": "0" * 16,
        })
    max_displacement = max(
        (
            abs(c_arm.index(bucket_id) - index)
            for index, bucket_id in enumerate(b_arm)
        ),
        default=0,
    )
    return {
        "schema": "e_chord_shadow_receipt.v1",
        "shadow_only": True,
        "affects_ranking": False,
        "payload_status": "accepted",
        "recorded_at_ms": 1_777_777_777_000,
        "agent_id": "claude",
        "e_authored_by": "哥哥",
        "source_turn_digest": _source_digest(digest),
        "projection_digest": digest,
        "facet_count": 1,
        "attempt_index": attempt_index,
        "first_screen_limit": 1,
        "pool_ids": list(pool),
        "relevance_band_ids": [0 for _bucket_id in pool],
        "candidate_guards": [
            {
                "id": bucket_id,
                "event_lock_digests": ["0" * 16],
                "is_factual": False,
                "author_match": True,
            }
            for bucket_id in pool
        ],
        "arms": {"a": a_arm, "b": b_arm, "c": c_arm},
        "first_screen": {"a": list(a), "b": list(b), "c": list(c)},
        "swaps": swaps,
        "skipped_reasons": [],
        "diagnostics": {
            "same_candidate_pool": True,
            "candidate_set_drift": 0,
            "max_displacement": max_displacement,
            "cross_event_moves": 0,
            "cross_relevance_moves": 0,
            "fact_moves": 0,
            "cross_author_moves": 0,
            "zero_to_nonzero": 0,
            "external_api_delta": 0,
            "hard_violation_count": 0,
        },
        "request_path_delta_ms": latency,
    }


def _selection(
    receipt,
    *,
    a=None,
    b=None,
    c=None,
    latency=0.0,
    outside=None,
):
    outside = list(outside or [])
    arms = {
        "a": list(receipt["first_screen"]["a"] if a is None else a),
        "b": list(receipt["first_screen"]["b"] if b is None else b),
        "c": list(receipt["first_screen"]["c"] if c is None else c),
    }
    return {
        "schema": "e_chord_final_selection.v1",
        "recorded_at_ms": receipt["recorded_at_ms"] + 10,
        "agent_id": receipt["agent_id"],
        "source_turn_digest": receipt["source_turn_digest"],
        "projection_digest": receipt["projection_digest"],
        "attempt_index": receipt["attempt_index"],
        "pool_ids": list(receipt["pool_ids"]),
        "final_injected_ids": arms["b"] + outside,
        "outside_pool_ids": outside,
        "arms": arms,
        "request_path_delta_ms": latency,
    }


def _gold(digest, *, case_id, expected, acceptable, noise, zero=False):
    natural_turn_digest = format((int(digest[0], 16) + 1) % 16, "x") * 64
    return {
        "case_id": case_id,
        "projection_digest": digest,
        "agent_id": "claude",
        "source_turn_digest": _source_digest(digest),
        "expected_ids": list(expected),
        "acceptable_ids": list(acceptable),
        "noise_ids": list(noise),
        "expected_zero": zero,
        "source_kind": "natural_conversation",
        "natural_turn_digest": natural_turn_digest,
        "annotation_method": "human",
        "source_evidence": {
            "path": "messages/natural.jsonl",
            "line": 1,
            "line_sha256": "9" * 64,
            "query_sha256": natural_turn_digest,
        },
        "annotated_by": "named-reviewer",
        "annotated_at": "2026-08-29T16:00:00+08:00",
    }


def test_evaluator_scores_same_pool_and_requires_strict_completeness_gain():
    first_digest = "a" * 64
    zero_digest = "b" * 64
    receipts = [
        _receipt(
            first_digest,
            pool=["noise", "expected"],
            a=["noise"],
            b=["noise"],
            c=["expected"],
            latency=0.4,
        ),
        _receipt(
            zero_digest,
            pool=[],
            a=[],
            b=[],
            c=[],
            latency=0.2,
        ),
    ]
    gold = [
        _gold(
            first_digest,
            case_id="natural-1",
            expected=["expected"],
            acceptable=["expected"],
            noise=["noise"],
        ),
        _gold(
            zero_digest,
            case_id="natural-zero",
            expected=[],
            acceptable=[],
            noise=[],
            zero=True,
        ),
    ]

    report = EVAL.evaluate(
        receipts,
        [_selection(receipts[0], latency=0.1), _selection(receipts[1], latency=0.1)],
        gold,
        min_cases=2,
        p95_budget_ms=5.0,
        verified_source_turn_digests={row["source_turn_digest"] for row in gold},
    )

    assert report["status"] == "candidate_for_named_review"
    assert report["eligible_for_live"] is False
    assert report["mechanical_candidate_pass"] is True
    assert report["named_review_required"] is True
    assert report["metrics"]["b"]["completeness"] == 0.0
    assert report["metrics"]["c"]["completeness"] == 1.0
    assert report["metrics"]["b"]["noise_rate"] == 1.0
    assert report["metrics"]["c"]["noise_rate"] == 0.0
    assert report["metrics"]["c"]["correct_zero_rate"] == 1.0
    assert report["metrics"]["c"]["predicted_zero_precision"] == 1.0
    assert report["p95_request_path_delta_ms"] == 0.5


def test_window_retry_scores_final_attempt_and_sums_all_attempt_latency():
    digest = "7" * 64
    first = _receipt(
        digest,
        pool=["window-noise"],
        a=["window-noise"],
        b=["window-noise"],
        c=["window-noise"],
        latency=0.3,
        attempt_index=0,
    )
    final = _receipt(
        digest,
        pool=["noise", "expected"],
        a=["noise"],
        b=["noise"],
        c=["expected"],
        latency=0.7,
        attempt_index=1,
    )
    final["recorded_at_ms"] += 1
    gold = _gold(
        digest,
        case_id="natural-window-retry",
        expected=["expected"],
        acceptable=["expected"],
        noise=["noise"],
    )

    report = EVAL.evaluate(
        [final, first],
        [_selection(final, latency=0.2)],
        [gold],
        min_cases=1,
        p95_budget_ms=2.0,
        verified_source_turn_digests={gold["source_turn_digest"]},
    )

    assert report["status"] == "failed"
    assert report["scorable_cases"] == 1
    assert report["receipt_count"] == 2
    assert report["evaluated_turn_count"] == 1
    assert report["retry_receipt_count"] == 1
    assert report["p95_request_path_delta_ms"] == 1.2
    assert report["metrics"]["c"]["completeness"] == 1.0


def test_equal_shadow_arms_cannot_claim_different_final_c_gain():
    digest = "1" * 64
    receipt = _receipt(
        digest,
        pool=["noise", "expected"],
        a=["noise"],
        b=["noise"],
        c=["noise"],
    )
    selection = _selection(receipt, c=["expected"])
    gold = _gold(
        digest,
        case_id="natural-forged-final-c",
        expected=["expected"],
        acceptable=["expected"],
        noise=["noise"],
    )

    try:
        EVAL.evaluate(
            [receipt],
            [selection],
            [gold],
            min_cases=1,
            verified_source_turn_digests={gold["source_turn_digest"]},
        )
    except ValueError as exc:
        assert "equal shadow arms" in str(exc)
    else:
        raise AssertionError("an unchanged C arm cannot claim a different final result")


def test_final_selection_must_follow_receipt_time_and_visible_cutoff():
    digest = "2" * 64
    receipt = _receipt(
        digest,
        pool=["first", "second"],
        a=["first"],
        b=["first"],
        c=["first"],
    )
    gold = _gold(
        digest,
        case_id="natural-final-selection-boundary",
        expected=["first"],
        acceptable=["first", "second"],
        noise=[],
    )

    predating = _selection(receipt)
    predating["recorded_at_ms"] = receipt["recorded_at_ms"] - 1
    try:
        EVAL.evaluate([receipt], [predating], [gold], min_cases=1)
    except ValueError as exc:
        assert "predates" in str(exc)
    else:
        raise AssertionError("a final selection cannot predate its receipt")

    oversized = _selection(
        receipt,
        a=["first", "second"],
        b=["first", "second"],
        c=["first", "second"],
    )
    try:
        EVAL.evaluate([receipt], [oversized], [gold], min_cases=1)
    except ValueError as exc:
        assert "visible cutoff" in str(exc)
    else:
        raise AssertionError("a final selection cannot exceed its visible cutoff")


def test_window_retry_keeps_first_attempt_policy_violation_in_gate():
    digest = "8" * 64
    first = _receipt(
        digest,
        pool=["noise", "expected"],
        a=["noise"],
        b=["noise"],
        c=["expected"],
        attempt_index=0,
    )
    first["candidate_guards"][1]["is_factual"] = True
    first["diagnostics"]["fact_moves"] = 1
    first["diagnostics"]["hard_violation_count"] = 1
    final = _receipt(
        digest,
        pool=["noise", "expected"],
        a=["noise"],
        b=["noise"],
        c=["expected"],
        attempt_index=1,
    )
    final["recorded_at_ms"] += 1
    gold = _gold(
        digest,
        case_id="natural-window-policy",
        expected=["expected"],
        acceptable=["expected"],
        noise=["noise"],
    )

    report = EVAL.evaluate(
        [first, final],
        [_selection(final)],
        [gold],
        min_cases=1,
        verified_source_turn_digests={gold["source_turn_digest"]},
    )

    assert report["status"] == "failed"
    assert report["hard_violation_count"] == 1
    assert report["gates"]["hard_violations_zero"] is False


def test_final_selection_must_name_an_existing_last_attempt():
    digest = "5" * 64
    receipt = _receipt(
        digest,
        pool=["expected"],
        a=["expected"],
        b=["expected"],
        c=["expected"],
    )
    selection = _selection(receipt)
    selection["attempt_index"] = 1
    gold = _gold(
        digest,
        case_id="natural-missing-final-attempt",
        expected=["expected"],
        acceptable=["expected"],
        noise=[],
    )

    try:
        EVAL.evaluate([receipt], [selection], [gold], min_cases=1)
    except ValueError as exc:
        assert "missing receipt attempt" in str(exc)
    else:
        raise AssertionError("a final selection cannot point to a missing retry")


def test_auxiliary_id_outside_frozen_pool_fails_closed():
    digest = "1" * 64
    receipt = _receipt(
        digest,
        pool=["noise", "expected"],
        a=["noise"],
        b=["noise"],
        c=["expected"],
    )
    selection = _selection(receipt, outside=["timeline-link"])
    gold = _gold(
        digest,
        case_id="natural-auxiliary-id",
        expected=["expected"],
        acceptable=["expected"],
        noise=["noise"],
    )

    report = EVAL.evaluate(
        [receipt],
        [selection],
        [gold],
        min_cases=1,
        verified_source_turn_digests={gold["source_turn_digest"]},
    )

    assert report["status"] == "failed"
    assert report["outside_pool_id_count"] == 1
    assert report["gates"]["outside_pool_ids_zero"] is False


def test_evaluator_marks_small_or_non_improving_cohort_inconclusive_or_failed():
    digest = "c" * 64
    receipt = _receipt(
        digest,
        pool=["expected"],
        a=["expected"],
        b=["expected"],
        c=["expected"],
    )
    gold = _gold(
        digest,
        case_id="natural-1",
        expected=["expected"],
        acceptable=["expected"],
        noise=[],
    )

    selection = _selection(receipt)
    small = EVAL.evaluate([receipt], [selection], [gold], min_cases=20)
    enough_but_flat = EVAL.evaluate([receipt], [selection], [gold], min_cases=1)

    assert small["status"] == "inconclusive"
    assert small["eligible_for_live"] is False
    assert enough_but_flat["status"] == "failed"
    assert enough_but_flat["gates"]["completeness_strictly_better"] is False


def test_evaluator_rejects_partial_gold_and_hard_violations():
    digest = "d" * 64
    receipt = _receipt(
        digest,
        pool=["expected", "unlabelled"],
        a=["expected"],
        b=["expected"],
        c=["expected"],
    )
    gold = _gold(
        digest,
        case_id="natural-1",
        expected=["expected"],
        acceptable=["expected"],
        noise=[],
    )

    try:
        EVAL.evaluate([receipt], [_selection(receipt)], [gold], min_cases=1)
    except ValueError as exc:
        assert "completely label pool" in str(exc)
    else:
        raise AssertionError("partial gold must be rejected")

    broken = copy.deepcopy(receipt)
    broken["diagnostics"]["external_api_delta"] = 1
    try:
        EVAL.evaluate([broken], [_selection(receipt)], [gold], min_cases=1)
    except ValueError as exc:
        assert "receipt contract" in str(exc)
    else:
        raise AssertionError("external API delta must be rejected")


def test_evaluator_rejects_candidate_pool_drift_even_if_diagnostics_lie():
    digest = "e" * 64
    receipt = _receipt(
        digest,
        pool=["expected", "noise"],
        a=["expected"],
        b=["noise"],
        c=["expected"],
    )
    receipt["arms"]["c"] = ["expected", "outside"]
    receipt["first_screen"]["c"] = ["expected"]

    try:
        EVAL.evaluate([
            receipt
        ], [_selection(receipt)], [
            _gold(
                digest,
                case_id="natural-drift",
                expected=["expected"],
                acceptable=["expected"],
                noise=["noise"],
            )
        ], min_cases=1)
    except ValueError as exc:
        assert "receipt contract" in str(exc)
    else:
        raise AssertionError("candidate-pool drift must be rejected")


def test_unlabelled_receipts_fail_the_same_frozen_cohort_gate():
    selected_digest = "3" * 64
    omitted_digest = "4" * 64
    receipt = _receipt(
        selected_digest,
        pool=["noise", "expected"],
        a=["noise"],
        b=["noise"],
        c=["expected"],
    )
    omitted = _receipt(
        omitted_digest,
        pool=["expected"],
        a=["expected"],
        b=["expected"],
        c=["expected"],
        latency=999.0,
    )
    gold = _gold(
        selected_digest,
        case_id="natural-selected",
        expected=["expected"],
        acceptable=["expected"],
        noise=["noise"],
    )

    report = EVAL.evaluate(
        [receipt, omitted],
        [_selection(receipt), _selection(omitted)],
        [gold],
        min_cases=1,
        verified_source_turn_digests={gold["source_turn_digest"]},
    )

    assert report["status"] == "failed"
    assert report["eligible_for_live"] is False
    assert report["unmatched_receipts"] == 1
    assert report["gates"]["complete_cohort_match"] is False


def test_evaluator_rejects_duplicate_case_provenance():
    first = _gold(
        "f" * 64,
        case_id="natural-duplicate",
        expected=["expected"],
        acceptable=["expected"],
        noise=[],
    )
    second = _gold(
        "0" * 64,
        case_id="natural-duplicate",
        expected=["expected"],
        acceptable=["expected"],
        noise=[],
    )
    try:
        EVAL.evaluate([], [], [first, second])
    except ValueError as exc:
        assert "duplicate case_id" in str(exc)
    else:
        raise AssertionError("duplicate cases must be rejected")


def test_gold_evidence_binds_exact_natural_user_line(tmp_path):
    evidence_root = tmp_path / "evidence"
    source = evidence_root / "messages" / "natural.jsonl"
    source.parent.mkdir(parents=True)
    record = {
        "id": "message-natural-1",
        "convId": "dm:claude",
        "eventType": "chat_message",
        "sender": "user",
        "text": "same natural phrase",
        "recordedAt": "2026-08-29T15:00:00+08:00",
    }
    encoded = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    source.write_bytes(encoded + b"\n")
    query_digest = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
    source_turn_digest = hashlib.sha256(
        b"claude\0dm:claude\0message-natural-1"
    ).hexdigest()
    gold = _gold(
        "2" * 64,
        case_id="natural-evidence",
        expected=["expected"],
        acceptable=["expected"],
        noise=[],
    )
    gold["natural_turn_digest"] = query_digest
    gold["source_turn_digest"] = source_turn_digest
    gold["source_evidence"] = {
        "path": "messages/natural.jsonl",
        "line": 1,
        "line_sha256": hashlib.sha256(encoded).hexdigest(),
        "query_sha256": query_digest,
    }

    assert EVAL.verify_gold_evidence([EVAL.validate_gold(gold)], evidence_root) == {
        source_turn_digest
    }

    source.write_bytes(encoded.replace(b"natural", b"changed") + b"\n")
    try:
        EVAL.verify_gold_evidence([gold], evidence_root)
    except ValueError as exc:
        assert "line hash mismatch" in str(exc)
    else:
        raise AssertionError("mutated natural evidence must be rejected")


def test_real_source_line_cannot_be_paired_with_an_unrelated_receipt(tmp_path):
    evidence_root = tmp_path / "evidence"
    source = evidence_root / "messages" / "natural.jsonl"
    source.parent.mkdir(parents=True)
    record = {
        "id": "message-real-20",
        "convId": "dm:claude",
        "eventType": "chat_message",
        "sender": "user",
        "text": "a real but unrelated natural phrase",
        "recordedAt": "2026-08-29T15:00:00+08:00",
    }
    encoded = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    source.write_bytes(encoded + b"\n")
    query_digest = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
    real_source_digest = hashlib.sha256(
        b"claude\0dm:claude\0message-real-20"
    ).hexdigest()
    projection_digest = "6" * 64
    receipt = _receipt(
        projection_digest,
        pool=["noise", "expected"],
        a=["noise"],
        b=["noise"],
        c=["expected"],
    )
    gold = _gold(
        projection_digest,
        case_id="mismatched-natural-turn",
        expected=["expected"],
        acceptable=["expected"],
        noise=["noise"],
    )
    gold["natural_turn_digest"] = query_digest
    gold["source_turn_digest"] = real_source_digest
    gold["source_evidence"] = {
        "path": "messages/natural.jsonl",
        "line": 1,
        "line_sha256": hashlib.sha256(encoded).hexdigest(),
        "query_sha256": query_digest,
    }
    verified = EVAL.verify_gold_evidence([EVAL.validate_gold(gold)], evidence_root)

    try:
        EVAL.evaluate(
            [receipt],
            [_selection(receipt)],
            [gold],
            min_cases=1,
            verified_source_turn_digests=verified,
        )
    except ValueError as exc:
        assert "not bound to receipt turn" in str(exc)
    else:
        raise AssertionError("a real but unrelated source line must not score a receipt")


def test_inconclusive_cli_is_nonzero(tmp_path, capsys):
    receipts = tmp_path / "receipts.jsonl"
    selections = tmp_path / "selections.jsonl"
    gold = tmp_path / "gold.jsonl"
    evidence = tmp_path / "evidence"
    receipts.write_text("", encoding="utf-8")
    selections.write_text("", encoding="utf-8")
    gold.write_text("", encoding="utf-8")
    evidence.mkdir()

    exit_code = EVAL.main([
        "--receipts", str(receipts),
        "--selections", str(selections),
        "--gold", str(gold),
        "--evidence-root", str(evidence),
    ])

    assert exit_code == 2
    assert '"status": "inconclusive"' in capsys.readouterr().out
