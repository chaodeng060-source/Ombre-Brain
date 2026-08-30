#!/usr/bin/env python3
"""Offline A/B/C evaluator for text-free E-chord shadow receipts.

The evaluator performs no retrieval and has no network/provider imports.  It
joins each Ombre shadow attempt to Twin's post-filter final-selection receipt
and separately human-authored gold by the opaque projection digest.  Metrics
use the actual final B injection and locally replayed A/C selections, and only
completely labelled frozen pools are scorable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from e_chord_shadow import (  # noqa: E402
    validate_final_selection,
    validate_shadow_receipt,
)


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_GOLD_KEYS = frozenset({
    "case_id",
    "projection_digest",
    "agent_id",
    "source_turn_digest",
    "expected_ids",
    "acceptable_ids",
    "noise_ids",
    "expected_zero",
    "source_kind",
    "natural_turn_digest",
    "annotation_method",
    "source_evidence",
    "annotated_by",
    "annotated_at",
})
_SOURCE_EVIDENCE_KEYS = frozenset({
    "path",
    "line",
    "line_sha256",
    "query_sha256",
})


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json(raw: str) -> object:
    return json.loads(
        raw,
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {value}")
        ),
    )


def load_jsonl(
    path: str | Path,
    *,
    validator: Callable[[object], Any] | None = None,
) -> list[Any]:
    """Read existing JSONL without creating, locking, or mutating its source."""

    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = _strict_json(raw)
                rows.append(validator(value) if validator else value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL row {source}:{line_number}") from exc
    return rows


def _id_list(value: object, *, name: str) -> list[str]:
    if type(value) is not list:
        raise ValueError(f"gold {name} must be a list")
    if any(type(item) is not str or not item for item in value):
        raise ValueError(f"gold {name} contains an invalid id")
    if len(set(value)) != len(value):
        raise ValueError(f"gold {name} contains duplicates")
    return value


def validate_gold(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _GOLD_KEYS:
        raise ValueError("invalid E chord gold contract")
    case_id = value.get("case_id")
    digest = value.get("projection_digest")
    if type(case_id) is not str or not case_id or len(case_id) > 160:
        raise ValueError("gold case_id is invalid")
    if type(digest) is not str or _HEX64_RE.fullmatch(digest) is None:
        raise ValueError("gold projection_digest is invalid")
    agent_id = value.get("agent_id")
    if type(agent_id) is not str or _SAFE_ID_RE.fullmatch(agent_id) is None:
        raise ValueError("gold agent_id is invalid")
    source_turn_digest = value.get("source_turn_digest")
    if (
        type(source_turn_digest) is not str
        or _HEX64_RE.fullmatch(source_turn_digest) is None
    ):
        raise ValueError("gold source_turn_digest is invalid")
    if type(value.get("expected_zero")) is not bool:
        raise ValueError("gold expected_zero must be boolean")
    if value.get("source_kind") != "natural_conversation":
        raise ValueError("gold source_kind must be natural_conversation")
    if value.get("annotation_method") != "human":
        raise ValueError("gold annotation_method must be human")
    natural_turn_digest = value.get("natural_turn_digest")
    if (
        type(natural_turn_digest) is not str
        or _HEX64_RE.fullmatch(natural_turn_digest) is None
    ):
        raise ValueError("gold natural_turn_digest is invalid")
    source_evidence = value.get("source_evidence")
    if (
        type(source_evidence) is not dict
        or set(source_evidence) != _SOURCE_EVIDENCE_KEYS
        or type(source_evidence.get("path")) is not str
        or not source_evidence["path"]
        or Path(source_evidence["path"]).is_absolute()
        or ".." in Path(source_evidence["path"]).parts
        or type(source_evidence.get("line")) is not int
        or source_evidence["line"] < 1
        or type(source_evidence.get("line_sha256")) is not str
        or _HEX64_RE.fullmatch(source_evidence["line_sha256"]) is None
        or type(source_evidence.get("query_sha256")) is not str
        or _HEX64_RE.fullmatch(source_evidence["query_sha256"]) is None
        or source_evidence["query_sha256"] != natural_turn_digest
    ):
        raise ValueError("gold source_evidence is invalid")
    for field in ("annotated_by", "annotated_at"):
        item = value.get(field)
        if type(item) is not str or not item.strip() or len(item) > 160:
            raise ValueError(f"gold {field} is invalid")
    try:
        annotated_at = datetime.fromisoformat(value["annotated_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("gold annotated_at is invalid") from exc
    if annotated_at.tzinfo is None:
        raise ValueError("gold annotated_at must include timezone")
    expected = _id_list(value.get("expected_ids"), name="expected_ids")
    acceptable = _id_list(value.get("acceptable_ids"), name="acceptable_ids")
    noise = _id_list(value.get("noise_ids"), name="noise_ids")
    if set(acceptable) & set(noise):
        raise ValueError("gold acceptable_ids and noise_ids overlap")
    if not set(expected) <= set(acceptable):
        raise ValueError("gold expected_ids must be acceptable")
    if value["expected_zero"] and (expected or acceptable):
        raise ValueError("gold zero case cannot declare acceptable hits")
    if not value["expected_zero"] and not expected:
        raise ValueError("gold non-zero case requires expected_ids")
    return value


def verify_gold_evidence(
    gold_rows: list[dict[str, Any]],
    evidence_root: str | Path,
) -> set[str]:
    """Bind every gold row and receipt identity to one natural user-message line."""

    root = Path(evidence_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("evidence root must be a directory")
    verified: set[str] = set()
    for gold in gold_rows:
        evidence = gold["source_evidence"]
        source = (root / evidence["path"]).resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"gold evidence path escapes root for {gold['case_id']}")
        target_line = None
        with source.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                if line_number == evidence["line"]:
                    target_line = raw.rstrip(b"\r\n")
                    break
        if target_line is None:
            raise ValueError(f"gold evidence line missing for {gold['case_id']}")
        if hashlib.sha256(target_line).hexdigest() != evidence["line_sha256"]:
            raise ValueError(f"gold evidence line hash mismatch for {gold['case_id']}")
        try:
            record = _strict_json(target_line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"gold evidence is not strict JSON for {gold['case_id']}") from exc
        if type(record) is not dict or record.get("sender") != "user":
            raise ValueError(f"gold evidence is not a user turn for {gold['case_id']}")
        if record.get("automation"):
            raise ValueError(f"gold evidence is automated for {gold['case_id']}")
        if record.get("eventType") not in (None, "chat_message"):
            raise ValueError(f"gold evidence is not a chat message for {gold['case_id']}")
        conv_id = record.get("convId")
        source_event_id = record.get("id")
        if (
            type(conv_id) is not str
            or _SAFE_ID_RE.fullmatch(conv_id) is None
            or type(source_event_id) is not str
            or _SAFE_ID_RE.fullmatch(source_event_id) is None
        ):
            raise ValueError(f"gold evidence has no stable turn identity for {gold['case_id']}")
        source_turn_digest = hashlib.sha256(
            "\0".join((gold["agent_id"], conv_id, source_event_id)).encode("utf-8")
        ).hexdigest()
        if source_turn_digest != gold["source_turn_digest"]:
            raise ValueError(f"gold source turn mismatch for {gold['case_id']}")
        text = record.get("text")
        if type(text) is not str or not text.strip():
            raise ValueError(f"gold evidence has no user text for {gold['case_id']}")
        query_digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if (
            query_digest != evidence["query_sha256"]
            or query_digest != gold["natural_turn_digest"]
        ):
            raise ValueError(f"gold natural turn hash mismatch for {gold['case_id']}")
        observed_at_raw = record.get("recordedAt") or record.get("ts")
        if type(observed_at_raw) is not str:
            raise ValueError(f"gold evidence has no timestamp for {gold['case_id']}")
        try:
            observed_at = datetime.fromisoformat(observed_at_raw.replace("Z", "+00:00"))
            annotated_at = datetime.fromisoformat(
                gold["annotated_at"].replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(f"gold evidence timestamp is invalid for {gold['case_id']}") from exc
        if observed_at.tzinfo is None or annotated_at.tzinfo is None:
            raise ValueError(f"gold evidence timestamp lacks timezone for {gold['case_id']}")
        if annotated_at < observed_at:
            raise ValueError(f"gold predates source turn for {gold['case_id']}")
        verified.add(source_turn_digest)
    return verified


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[index], 6)


def _expected_final_swaps(
    receipt: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Replay receipt-approved pairs with Twin's all-or-nothing fallback."""

    guard_by_id = {
        guard["id"]: guard for guard in receipt["candidate_guards"]
    }
    baseline = list(selection["arms"]["b"])
    working = list(baseline)
    expected: list[dict[str, Any]] = []
    for proposed in receipt["swaps"]:
        demoted_id = proposed["demoted_id"]
        promoted_id = proposed["promoted_id"]
        if demoted_id not in working or promoted_id not in working:
            return [], baseline
        to_index = working.index(demoted_id)
        from_index = working.index(promoted_id)
        if from_index != to_index + 1:
            return [], baseline
        promoted_guard = guard_by_id[promoted_id]
        demoted_guard = guard_by_id[demoted_id]
        shared_locks = set(promoted_guard["event_lock_digests"]) & set(
            demoted_guard["event_lock_digests"]
        )
        safe = (
            proposed["event_lock_digest"] in shared_locks
            and not promoted_guard["is_factual"]
            and not demoted_guard["is_factual"]
            and promoted_guard["author_match"]
            and demoted_guard["author_match"]
            and promoted_guard["e_admissibility"] == "admissible"
            and demoted_guard["e_admissibility"] == "admissible"
            and promoted_guard["e_resonance_milli"] + 30
            >= demoted_guard["e_resonance_milli"]
        )
        if not safe:
            return [], baseline
        expected.append({
            "promoted_id": promoted_id,
            "demoted_id": demoted_id,
            "from_index": from_index,
            "to_index": to_index,
            "event_lock_digest": proposed["event_lock_digest"],
        })
        working[to_index], working[from_index] = (
            working[from_index],
            working[to_index],
        )
    return expected, working


def evaluate(
    receipts: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    min_cases: int = 20,
    p95_budget_ms: float = 5.0,
    verified_source_turn_digests: set[str] | None = None,
) -> dict[str, Any]:
    if type(min_cases) is not int or min_cases < 1:
        raise ValueError("min_cases must be a positive integer")
    if type(p95_budget_ms) not in (int, float) or not math.isfinite(p95_budget_ms) \
            or p95_budget_ms < 0:
        raise ValueError("p95_budget_ms must be finite and non-negative")

    validated_receipts = [validate_shadow_receipt(row) for row in receipts]
    validated_selections = [validate_final_selection(row) for row in selections]
    validated_gold = [validate_gold(row) for row in gold_rows]
    verified_source_turn_digests = set(verified_source_turn_digests or ())
    receipt_groups: dict[str, list[dict[str, Any]]] = {}
    for receipt in validated_receipts:
        digest = receipt["projection_digest"]
        if not digest:
            continue
        receipt_groups.setdefault(digest, []).append(receipt)
    attempts_by_digest: dict[str, list[dict[str, Any]]] = {}
    for digest, group in receipt_groups.items():
        identities = {
            (
                receipt["agent_id"],
                receipt["e_authored_by"],
                receipt["source_turn_digest"],
            )
            for receipt in group
        }
        if len(identities) != 1:
            raise ValueError("retry receipts cross an identity boundary")
        by_attempt: dict[int, dict[str, Any]] = {}
        for receipt in group:
            attempt = receipt["attempt_index"]
            if attempt in by_attempt:
                raise ValueError("duplicate attempt_index in receipt group")
            by_attempt[attempt] = receipt
        attempt_numbers = sorted(by_attempt)
        if attempt_numbers != list(range(len(attempt_numbers))):
            raise ValueError("receipt attempts must be contiguous from zero")
        attempts = [by_attempt[index] for index in attempt_numbers]
        if any(
            later["recorded_at_ms"] < earlier["recorded_at_ms"]
            for earlier, later in zip(attempts, attempts[1:])
        ):
            raise ValueError("receipt attempt timestamps run backwards")
        attempts_by_digest[digest] = attempts
    selection_by_digest: dict[str, dict[str, Any]] = {}
    for selection in validated_selections:
        digest = selection["projection_digest"]
        if digest in selection_by_digest:
            raise ValueError("duplicate projection_digest in final selections")
        selection_by_digest[digest] = selection
    selected_receipt_by_digest: dict[str, dict[str, Any]] = {}
    selected_attempts_by_digest: dict[str, list[dict[str, Any]]] = {}
    for digest, selection in selection_by_digest.items():
        attempts = attempts_by_digest.get(digest)
        if attempts is None:
            continue
        attempt_index = selection["attempt_index"]
        if attempt_index >= len(attempts):
            raise ValueError("final selection points to a missing receipt attempt")
        if len(attempts) != attempt_index + 1:
            raise ValueError("receipt exists after the selected final attempt")
        receipt = attempts[attempt_index]
        if (
            receipt["agent_id"] != selection["agent_id"]
            or receipt["source_turn_digest"] != selection["source_turn_digest"]
            or receipt["pool_ids"] != selection["pool_ids"]
        ):
            raise ValueError("final selection is not bound to its shadow receipt")
        if selection["recorded_at_ms"] < receipt["recorded_at_ms"]:
            raise ValueError("final selection predates its shadow receipt")
        for arm in ("a", "b", "c"):
            if len(selection["arms"][arm]) > receipt["first_screen_limit"]:
                raise ValueError("final selection exceeds its visible cutoff")
        expected_applied_swaps, expected_final_c = _expected_final_swaps(
            receipt,
            selection,
        )
        if selection["applied_swaps"] != expected_applied_swaps:
            raise ValueError("final applied swaps are not bound to the shadow receipt")
        if selection["arms"]["c"] != expected_final_c:
            raise ValueError("final C is not the validated receipt replay over final B")
        # Twin runs one deterministic local filter/reranker over each frozen
        # arm.  Equal arm inputs therefore must have equal outputs.  Enforcing
        # this sound invariant prevents a selection row from manufacturing a
        # C-only gain when Ombre proposed no C ordering change, without
        # assuming that the legitimate local reranker preserves input order.
        for left, right in (("a", "b"), ("a", "c"), ("b", "c")):
            if (
                receipt["arms"][left] == receipt["arms"][right]
                and selection["arms"][left] != selection["arms"][right]
            ):
                raise ValueError(
                    "equal shadow arms produced different final selections"
                )
        selected_receipt_by_digest[digest] = receipt
        selected_attempts_by_digest[digest] = attempts
    gold_by_digest: dict[str, dict[str, Any]] = {}
    for gold in validated_gold:
        digest = gold["projection_digest"]
        if digest in gold_by_digest:
            raise ValueError("duplicate projection_digest in gold")
        gold_by_digest[digest] = gold
    if len({gold["case_id"] for gold in validated_gold}) != len(validated_gold):
        raise ValueError("duplicate case_id in gold")
    if len({gold["source_turn_digest"] for gold in validated_gold}) != len(validated_gold):
        raise ValueError("duplicate source_turn_digest in gold")

    totals = {
        arm: {
            "selected": 0,
            "acceptable": 0,
            "noise": 0,
            "expected_hits": 0,
            "expected_total": 0,
            "correct_zero": 0,
            "zero_total": 0,
            "predicted_zero": 0,
            "predicted_zero_correct": 0,
        }
        for arm in ("a", "b", "c")
    }
    latencies: list[float] = []
    hard_violations = 0
    external_api_delta = 0
    case_ids: list[str] = []
    unscorable_reason_counts: dict[str, int] = {}
    unscorable_turn_count = 0
    for digest, gold in gold_by_digest.items():
        receipt = selected_receipt_by_digest.get(digest)
        selection = selection_by_digest.get(digest)
        if receipt is None or selection is None:
            continue
        if (
            receipt["agent_id"] != gold["agent_id"]
            or receipt["source_turn_digest"] != gold["source_turn_digest"]
        ):
            raise ValueError(f"gold is not bound to receipt turn for {gold['case_id']}")
        pool = set(receipt["pool_ids"])
        labelled = set(gold["acceptable_ids"]) | set(gold["noise_ids"])
        if labelled != pool:
            raise ValueError(f"gold does not completely label pool for {gold['case_id']}")
        if not set(gold["expected_ids"]) <= pool:
            raise ValueError(f"gold expected id is outside pool for {gold['case_id']}")
        if receipt["diagnostics"]["same_candidate_pool"] is not True:
            raise ValueError(f"receipt arms do not share a pool for {gold['case_id']}")

        attempts = selected_attempts_by_digest[digest]
        latencies.append(sum(
            float(attempt["request_path_delta_ms"])
            for attempt in attempts
        ) + float(selection["request_path_delta_ms"]))
        hard_violations += sum(
            int(attempt["diagnostics"]["hard_violation_count"])
            for attempt in attempts
        )
        external_api_delta += sum(
            int(attempt["diagnostics"]["external_api_delta"])
            for attempt in attempts
        )
        unscorable_statuses = {
            attempt["a_cohort_status"]
            for attempt in attempts
            if attempt["a_cohort_status"] != "pure_semantic"
        }
        if selection["final_input_cohort_status"] != "pure_same_cohort":
            unscorable_statuses.add(selection["final_input_cohort_status"])
        unscorable_statuses = sorted(unscorable_statuses)
        if unscorable_statuses:
            unscorable_turn_count += 1
            for reason in unscorable_statuses:
                unscorable_reason_counts[reason] = (
                    unscorable_reason_counts.get(reason, 0) + 1
                )
            continue

        case_ids.append(gold["case_id"])
        acceptable = set(gold["acceptable_ids"])
        noise = set(gold["noise_ids"])
        expected = set(gold["expected_ids"])
        for arm in ("a", "b", "c"):
            selected = selection["arms"][arm]
            arm_totals = totals[arm]
            arm_totals["selected"] += len(selected)
            arm_totals["acceptable"] += len(set(selected) & acceptable)
            arm_totals["noise"] += len(set(selected) & noise)
            arm_totals["expected_hits"] += len(set(selected) & expected)
            arm_totals["expected_total"] += len(expected)
            if gold["expected_zero"]:
                arm_totals["zero_total"] += 1
                arm_totals["correct_zero"] += int(len(selected) == 0)
            if not selected:
                arm_totals["predicted_zero"] += 1
                arm_totals["predicted_zero_correct"] += int(
                    gold["expected_zero"]
                )
    metrics = {}
    for arm, values in totals.items():
        metrics[arm] = {
            "precision": _rate(values["acceptable"], values["selected"]),
            "noise_rate": _rate(values["noise"], values["selected"]),
            "completeness": _rate(values["expected_hits"], values["expected_total"]),
            "correct_zero_rate": _rate(values["correct_zero"], values["zero_total"]),
            "predicted_zero_precision": _rate(
                values["predicted_zero_correct"], values["predicted_zero"]
            ),
            "selected_count": values["selected"],
        }
    p95_delta = _p95(latencies)
    scored = len(case_ids)
    receipt_digests = set(attempts_by_digest)
    selection_digests = set(selection_by_digest)
    gold_digests = set(gold_by_digest)
    jointly_bound = receipt_digests & selection_digests & gold_digests
    unmatched_receipts = len(receipt_digests - jointly_bound)
    unmatched_selections = len(selection_digests - jointly_bound)
    unmatched_gold = len(gold_digests - jointly_bound)
    complete_cohort_match = (
        receipt_digests == selection_digests == gold_digests
    )
    outside_pool_count = sum(
        len(selection["outside_pool_ids"])
        for selection in validated_selections
    )
    natural_evidence_verified = all(
        gold["source_turn_digest"] in verified_source_turn_digests
        for gold in validated_gold
    ) and bool(validated_gold)

    def no_worse(current: float | None, baseline: float | None, *, high_good: bool) -> bool:
        if current is None or baseline is None:
            return False
        return current >= baseline if high_good else current <= baseline

    c_improves_completeness = (
        metrics["c"]["completeness"] is not None
        and metrics["b"]["completeness"] is not None
        and metrics["c"]["completeness"] > metrics["b"]["completeness"]
    )
    gates = {
        "minimum_cases": scored >= min_cases,
        "complete_cohort_match": complete_cohort_match,
        "outside_pool_ids_zero": outside_pool_count == 0,
        "natural_evidence_verified": natural_evidence_verified,
        "nonzero_cases_present": totals["b"]["expected_total"] > 0,
        "zero_controls_present": totals["b"]["zero_total"] > 0,
        "precision_not_worse": no_worse(
            metrics["c"]["precision"], metrics["b"]["precision"], high_good=True
        ),
        "noise_not_worse": no_worse(
            metrics["c"]["noise_rate"], metrics["b"]["noise_rate"], high_good=False
        ),
        "completeness_strictly_better": c_improves_completeness,
        "correct_zero_not_worse": no_worse(
            metrics["c"]["correct_zero_rate"],
            metrics["b"]["correct_zero_rate"],
            high_good=True,
        ),
        "predicted_zero_precision_not_worse": no_worse(
            metrics["c"]["predicted_zero_precision"],
            metrics["b"]["predicted_zero_precision"],
            high_good=True,
        ),
        "hard_violations_zero": hard_violations == 0,
        "external_api_delta_zero": external_api_delta == 0,
        "p95_request_path_delta_within_budget": (
            p95_delta is not None and p95_delta <= p95_budget_ms
        ),
        "all_turns_scorable": not unscorable_reason_counts,
    }
    mechanical_candidate_pass = all(gates.values())
    if scored < min_cases:
        status = "inconclusive"
    elif mechanical_candidate_pass:
        status = "candidate_for_named_review"
    elif all(
        passed
        for name, passed in gates.items()
        if name != "completeness_strictly_better"
    ):
        # Final B and C are required to have the same candidate set, so the
        # set-based completeness gate cannot be established by this version.
        status = "inconclusive"
    else:
        status = "failed"
    return {
        "schema": "e_chord_shadow_eval.v1",
        "status": status,
        # This tool deliberately cannot self-authorize a live ranking change.
        # A named independent review must verify the private source ledger and
        # human annotations outside this process before any separate rollout.
        "eligible_for_live": False,
        "mechanical_candidate_pass": mechanical_candidate_pass,
        "named_review_required": True,
        "scorable_cases": scored,
        "minimum_cases": min_cases,
        "receipt_count": len(validated_receipts),
        "selection_count": len(validated_selections),
        "evaluated_turn_count": len(selected_receipt_by_digest),
        "retry_receipt_count": len(validated_receipts) - len(attempts_by_digest),
        "gold_count": len(validated_gold),
        "unmatched_receipts": unmatched_receipts,
        "unmatched_selections": unmatched_selections,
        "unmatched_gold": unmatched_gold,
        "outside_pool_id_count": outside_pool_count,
        "case_ids": sorted(case_ids),
        "metrics": metrics,
        "p95_request_path_delta_ms": p95_delta,
        "p95_budget_ms": float(p95_budget_ms),
        "hard_violation_count": hard_violations,
        "external_api_delta": external_api_delta,
        "unscorable_turn_count": unscorable_turn_count,
        "unscorable_reason_counts": dict(sorted(unscorable_reason_counts.items())),
        "completeness_gate_reachable": False,
        "gates": gates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--selections", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--p95-budget-ms", type=float, default=5.0)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args(argv)
    gold_rows = load_jsonl(args.gold, validator=validate_gold)
    verified_source_turn_digests = verify_gold_evidence(gold_rows, args.evidence_root)
    report = evaluate(
        load_jsonl(args.receipts, validator=validate_shadow_receipt),
        load_jsonl(args.selections, validator=validate_final_selection),
        gold_rows,
        min_cases=args.min_cases,
        p95_budget_ms=args.p95_budget_ms,
        verified_source_turn_digests=verified_source_turn_digests,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    if report["status"] == "candidate_for_named_review":
        return 3
    return 2 if report["status"] == "inconclusive" else 1


if __name__ == "__main__":
    raise SystemExit(main())
