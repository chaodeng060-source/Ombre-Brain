"""Replay the entity-ticket contract over the frozen 22-request evidence.

This is a bounded, provider-free replay.  The private ledger supplies real
request provenance, Anchor candidates, historical DS counts, and historical
final IDs.  The private entity audit supplies the matched entity's frozen BM25
document frequency; the ledger does not identify which literal term was the
matched entity.  The tool derives ticket eligibility and a bounded primary
projection, but never invents a new DS verdict or end-to-end ON receipt.

Output contains request IDs, bucket IDs, numeric evidence, and verification
states only.  Query text, bucket content, entity names, and prompts are never
emitted.  Every input must be a regular JSON file with no group/world read bit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import stat
from typing import Any


EXPECTED_LEDGER_SHA256 = (
    "ebe1ffd91675e1c5685c0b2a20a098cf527a4cb79bac78cf3767090021c2fddc"
)
EXPECTED_AUDIT_SHA256 = (
    "176faeaed36a790738dc1ab60b07ddbf220fcdbbbb2c5e6e0b0e7c5354d76a2d"
)
EXPECTED_AUDIT_SCHEMA = "entity_score_guard_audit.v1"
EXPECTED_REQUEST_COUNT = 22
EXPECTED_ANCHOR_POSITION_COUNT = 318
EXPECTED_ENTITY_POSITION_COUNT = 15
EXPECTED_ENTITY_UNIQUE_BUCKET_COUNT = 10
EXPECTED_PRE_DS_POSITION_COUNT = 89
EXPECTED_PRE_DS_UNIQUE_COUNT = 88
EXPECTED_HISTORICAL_FINAL_POSITION_COUNT = 52
EXPECTED_OK_REQUEST_COUNT = 18

LITERAL_CANDIDATE_FLOOR = 40.0
ENTITY_MAX_DF_RATIO = 0.2
MAX_TICKETS_PER_REQUEST = 2
ANCHOR_CONVERSATION_MIN_SCORE = 0.25
LITERAL_ONLY_CAP = 0.55

PRODUCTION_EFFECTIVE_QUERY_REQUEST_ID = "7fa0ff67ca954ae58750978fb53699e0"
PRODUCTION_EFFECTIVE_QUERY_SHA256 = (
    "fd83fb69932c422929f8c578a7966386a1461a260bdfff02db113cd5c279808b"
)
NOISE_REQUEST_ID = "5e00777f6b2f4c3e823ec6064aa39d42"
ENGINEERING_NOISE_IDS = ("013da98a75e5", "019af40158f7")
TARGET_REQUEST_ID = "844e6108dc444c228f0b4120569330a8"
TARGET_BUCKET_ID = "6cc5995aea84"

_VERIFIED_HUMAN_GOLD_FIELD = "verified_human_gold_relevance_label"
_RELEVANT_LABELS = {"relevant", "confirmed_relevant"}
_NOISE_LABELS = {"noise", "confirmed_noise"}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_private_json_object(path: Path) -> dict:
    """Read a regular JSON object while refusing group/world-readable input."""
    path = Path(path)
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"private replay input is missing: {path}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"private replay input must be a regular file: {path}")
    mode = stat.S_IMODE(file_stat.st_mode)
    if mode & 0o044:
        raise PermissionError(
            "private replay input must not be group/world-readable: "
            f"{path} is {mode:04o}"
        )
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(ledger_path: Path, audit_path: Path) -> tuple[dict, dict]:
    """Load the fixed ledger and its entity-DF evidence fail-closed."""
    ledger_path = Path(ledger_path)
    ledger = _read_private_json_object(ledger_path)
    ledger_sha256 = _file_sha256(ledger_path)
    if ledger_sha256 != EXPECTED_LEDGER_SHA256:
        raise ValueError(
            "frozen ledger SHA-256 mismatch: "
            f"{ledger_sha256} != {EXPECTED_LEDGER_SHA256}"
        )
    audit_path = Path(audit_path)
    audit = _read_private_json_object(audit_path)
    audit_sha256 = _file_sha256(audit_path)
    if audit_sha256 != EXPECTED_AUDIT_SHA256:
        raise ValueError(
            "frozen entity audit SHA-256 mismatch: "
            f"{audit_sha256} != {EXPECTED_AUDIT_SHA256}"
        )
    return ledger, audit


def _finite_number(value: Any, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _request_id(row: Mapping[str, Any]) -> str:
    return str(row.get("ombre_request_id") or "").strip()


def _bucket_ids(value: Any, *, field: str, request_id: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list for {request_id}")
    bucket_ids = [str(item).strip() for item in value]
    if any(not item for item in bucket_ids):
        raise ValueError(f"{field} contains an empty bucket ID for {request_id}")
    if len(bucket_ids) != len(set(bucket_ids)):
        raise ValueError(f"{field} contains duplicate IDs for {request_id}")
    return bucket_ids


def _query_provenance(row: Mapping[str, Any]) -> str:
    """Validate captured bytes without returning or emitting them."""
    request_id = _request_id(row)
    original = row.get("original_query")
    effective = row.get("effective_query")
    if request_id == PRODUCTION_EFFECTIVE_QUERY_REQUEST_ID:
        if isinstance(original, str) and original:
            raise ValueError(
                f"production effective row unexpectedly has original text: {request_id}"
            )
        if row.get("query_source") != "production_ds_log":
            raise ValueError(f"production effective query source drifted: {request_id}")
        if not isinstance(effective, str) or not effective:
            raise ValueError(f"production effective query is missing: {request_id}")
        digest = hashlib.sha256(effective.encode("utf-8")).hexdigest()
        if digest != PRODUCTION_EFFECTIVE_QUERY_SHA256:
            raise ValueError(f"production effective query hash mismatch: {request_id}")
        return "production_effective_query"

    if not isinstance(original, str) or not original:
        raise ValueError(f"original query is missing: {request_id}")
    prefix = str(row.get("sha") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{12,64}", prefix):
        raise ValueError(f"query hash prefix is invalid: {request_id}")
    if not hashlib.sha256(original.encode("utf-8")).hexdigest().startswith(prefix):
        raise ValueError(f"query hash mismatch: {request_id}")
    return "original_query"


def _validate_anchor(anchor: Any, *, request_id: str) -> tuple[str, bool]:
    if not isinstance(anchor, dict):
        raise ValueError(f"anchor must be an object for {request_id}")
    bucket_id = str(anchor.get("id") or "").strip()
    if not bucket_id:
        raise ValueError(f"anchor bucket ID is missing for {request_id}")
    _finite_number(anchor.get("s"), field=f"anchor score {request_id}/{bucket_id}")
    _finite_number(anchor.get("lit"), field=f"literal score {request_id}/{bucket_id}")
    _finite_number(anchor.get("vec"), field=f"vector score {request_id}/{bucket_id}")
    entity_match = anchor.get("ent")
    if not isinstance(entity_match, bool):
        raise ValueError(f"anchor ent must be boolean for {request_id}/{bucket_id}")
    rare = anchor.get("rare")
    if not isinstance(rare, list) or any(
        not isinstance(term, str) or not term for term in rare
    ):
        raise ValueError(f"anchor rare must be a string list for {request_id}/{bucket_id}")
    term_dfs = anchor.get("term_dfs_local_snapshot")
    if not isinstance(term_dfs, dict):
        raise ValueError(f"frozen term DF map is missing for {request_id}/{bucket_id}")
    for term, raw_df in term_dfs.items():
        if (
            not isinstance(term, str)
            or not term
            or not isinstance(raw_df, int)
            or isinstance(raw_df, bool)
            or raw_df <= 0
        ):
            raise ValueError(f"frozen term DF is invalid for {request_id}/{bucket_id}")
    return bucket_id, entity_match


def _validate_ledger(ledger: Mapping[str, Any]) -> tuple[list[dict], int, Counter]:
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_REQUEST_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_REQUEST_COUNT} ledger rows")
    corpus_count = ledger.get("corpus_count")
    if (
        not isinstance(corpus_count, int)
        or isinstance(corpus_count, bool)
        or corpus_count <= 0
    ):
        raise ValueError("ledger corpus_count must be a positive integer")

    request_ids: list[str] = []
    anchor_positions = 0
    entity_positions = 0
    entity_ids: set[str] = set()
    pre_ds_positions = 0
    pre_ds_ids: set[str] = set()
    historical_final_positions = 0
    provenance = Counter()

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every ledger row must be an object")
        request_id = _request_id(row)
        if not request_id:
            raise ValueError("every ledger row requires a request ID")
        request_ids.append(request_id)
        if str(row.get("policy") or "").strip().lower() != "conversation":
            raise ValueError(f"expected conversation policy for {request_id}")
        provenance[_query_provenance(row)] += 1

        anchors = row.get("anchors")
        if not isinstance(anchors, list):
            raise ValueError(f"anchors must be a list for {request_id}")
        seen_anchor_ids: set[str] = set()
        for anchor in anchors:
            bucket_id, entity_match = _validate_anchor(anchor, request_id=request_id)
            if bucket_id in seen_anchor_ids:
                raise ValueError(f"duplicate anchor bucket for {request_id}")
            seen_anchor_ids.add(bucket_id)
            anchor_positions += 1
            if entity_match:
                entity_positions += 1
                entity_ids.add(bucket_id)

        ids_in = _bucket_ids(row.get("ids_in"), field="ids_in", request_id=request_id)
        ids_out = _bucket_ids(row.get("ids_out"), field="ids_out", request_id=request_id)
        pre_ds_positions += len(ids_in)
        pre_ds_ids.update(ids_in)
        historical_final_positions += len(ids_out)

        outcome = row.get("ds_gate_outcome")
        if outcome not in ("ok", "error", None, ""):
            raise ValueError(f"unsupported historical DS outcome for {request_id}")
        for field in ("ds_gate_in", "ds_gate_out"):
            value = row.get(field)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"invalid {field} for {request_id}")

    if len(request_ids) != len(set(request_ids)):
        raise ValueError("ledger request IDs must be unique")
    if anchor_positions != EXPECTED_ANCHOR_POSITION_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_ANCHOR_POSITION_COUNT} anchor positions")
    if entity_positions != EXPECTED_ENTITY_POSITION_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_ENTITY_POSITION_COUNT} entity positions")
    if len(entity_ids) != EXPECTED_ENTITY_UNIQUE_BUCKET_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_ENTITY_UNIQUE_BUCKET_COUNT} unique entity buckets"
        )
    if pre_ds_positions != EXPECTED_PRE_DS_POSITION_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_PRE_DS_POSITION_COUNT} ids_in positions")
    if len(pre_ds_ids) != EXPECTED_PRE_DS_UNIQUE_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_PRE_DS_UNIQUE_COUNT} unique ids_in buckets")
    if historical_final_positions != EXPECTED_HISTORICAL_FINAL_POSITION_COUNT:
        raise ValueError(
            "expected exactly "
            f"{EXPECTED_HISTORICAL_FINAL_POSITION_COUNT} historical final positions"
        )
    if provenance != Counter({"original_query": 21, "production_effective_query": 1}):
        raise ValueError("frozen query provenance counts drifted")
    return rows, corpus_count, provenance


def _normalise_label(value: Any, *, field: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    label = value.strip().lower()
    if label in _RELEVANT_LABELS:
        return "relevant"
    if label in _NOISE_LABELS:
        return "noise"
    raise ValueError(f"unsupported {field}: {value!r}")


def _validate_audit(
    audit: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    corpus_count: int,
) -> dict[tuple[str, str], dict]:
    if audit.get("schema") != EXPECTED_AUDIT_SCHEMA:
        raise ValueError(f"expected audit schema {EXPECTED_AUDIT_SCHEMA}")
    candidate_rows = audit.get("candidate_rows")
    if (
        not isinstance(candidate_rows, list)
        or len(candidate_rows) != EXPECTED_ENTITY_POSITION_COUNT
    ):
        raise ValueError(
            f"expected exactly {EXPECTED_ENTITY_POSITION_COUNT} audit candidate rows"
        )

    anchors_by_pair = {
        (_request_id(row), str(anchor["id"])): anchor
        for row in rows
        for anchor in row["anchors"]
        if anchor.get("ent") is True
    }
    frozen_dfs_by_request: dict[str, dict[str, set[int]]] = {}
    for row in rows:
        request_id = _request_id(row)
        by_term: dict[str, set[int]] = {}
        for anchor in row["anchors"]:
            for term, raw_df in anchor["term_dfs_local_snapshot"].items():
                by_term.setdefault(str(term), set()).add(int(raw_df))
        frozen_dfs_by_request[request_id] = by_term
    evidence_by_pair: dict[tuple[str, str], dict] = {}
    for raw_evidence in candidate_rows:
        if not isinstance(raw_evidence, dict):
            raise ValueError("every audit candidate row must be an object")
        request_id = str(raw_evidence.get("request_id") or "").strip()
        bucket_id = str(raw_evidence.get("bucket_id") or "").strip()
        pair = (request_id, bucket_id)
        if pair in evidence_by_pair:
            raise ValueError(f"duplicate entity audit pair: {request_id}/{bucket_id}")
        anchor = anchors_by_pair.get(pair)
        if anchor is None:
            raise ValueError(f"audit pair is not an ent=true anchor: {request_id}/{bucket_id}")
        matched_entity = raw_evidence.get("matched_entity")
        if not isinstance(matched_entity, dict):
            raise ValueError(f"matched entity is missing for {request_id}/{bucket_id}")
        canonical_name = matched_entity.get("canonical_name")
        if not isinstance(canonical_name, str) or not canonical_name:
            raise ValueError(f"matched entity name is missing for {request_id}/{bucket_id}")
        entity_df = raw_evidence.get("entity_df")
        if (
            not isinstance(entity_df, int)
            or isinstance(entity_df, bool)
            or entity_df <= 0
        ):
            raise ValueError(f"entity DF is invalid for {request_id}/{bucket_id}")
        if entity_df not in frozen_dfs_by_request[request_id].get(
            canonical_name,
            set(),
        ):
            raise ValueError(
                "audit entity DF is not in the request's frozen term DF "
                f"snapshot for {request_id}/{bucket_id}"
            )
        if raw_evidence.get("corpus_bucket_count") != corpus_count:
            raise ValueError(f"audit corpus denominator drifted for {request_id}/{bucket_id}")
        high_frequency = entity_df / corpus_count >= ENTITY_MAX_DF_RATIO
        if raw_evidence.get("high_frequency") is not high_frequency:
            raise ValueError(f"audit frequency class drifted for {request_id}/{bucket_id}")
        _normalise_label(
            raw_evidence.get("human_relevance_label"),
            field="human_relevance_label",
        )
        _normalise_label(
            raw_evidence.get(_VERIFIED_HUMAN_GOLD_FIELD),
            field=_VERIFIED_HUMAN_GOLD_FIELD,
        )
        evidence_by_pair[pair] = copy.deepcopy(raw_evidence)

    if set(evidence_by_pair) != set(anchors_by_pair):
        raise ValueError("entity audit coverage does not match ent=true anchors")
    return evidence_by_pair


def _no_entity_anchor_score(anchor: Mapping[str, Any]) -> float | None:
    """Recompute only the existing absolute Anchor evidence, minus entity."""
    vector = float(anchor["vec"])
    literal = float(anchor["lit"])
    similarities: list[float] = []
    if vector > 0.0:
        similarities.append(max(0.0, min(1.0, vector)))
    normalised_literal = max(0.0, min(1.0, literal / 100.0))
    if vector <= 0.0 and not anchor.get("rare"):
        normalised_literal = min(normalised_literal, LITERAL_ONLY_CAP)
    similarities.append(normalised_literal)
    if not similarities:
        return None
    return round(0.45 * max(similarities), 6)


def _entity_result(
    anchor: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict:
    literal = float(anchor["lit"])
    vector = float(anchor["vec"])
    entity_df = int(evidence["entity_df"])
    corpus_count = int(evidence["corpus_bucket_count"])
    df_ratio = entity_df / corpus_count
    no_entity_score = _no_entity_anchor_score(anchor)
    ordinary_without_entity = (
        no_entity_score is None or no_entity_score >= ANCHOR_CONVERSATION_MIN_SCORE
    )
    ticket_eligible = (
        vector == 0.0
        and literal < LITERAL_CANDIDATE_FLOOR
        and df_ratio < ENTITY_MAX_DF_RATIO
    )
    return {
        "bucket_id": str(anchor["id"]),
        "literal_score": literal,
        "vector_score": vector,
        "entity_df": entity_df,
        "corpus_bucket_count": corpus_count,
        "entity_df_ratio": round(df_ratio, 12),
        "frequency_class": "high" if df_ratio >= ENTITY_MAX_DF_RATIO else "low",
        "score_without_entity": no_entity_score,
        "ordinary_without_entity": ordinary_without_entity,
        "ticket_eligible": ticket_eligible,
        "ticket_selected": False,
        "prior_manual_relevance": (
            _normalise_label(
                evidence.get("human_relevance_label"),
                field="human_relevance_label",
            )
            or "unlabelled"
        ),
        "verified_human_gold": (
            _normalise_label(
                evidence.get(_VERIFIED_HUMAN_GOLD_FIELD),
                field=_VERIFIED_HUMAN_GOLD_FIELD,
            )
            or "unlabelled"
        ),
    }


def replay(ledger: Mapping[str, Any], audit: Mapping[str, Any]) -> dict:
    """Return the privacy-safe 22-request OFF/ON evidence matrix."""
    rows, corpus_count, provenance = _validate_ledger(ledger)
    evidence_by_pair = _validate_audit(audit, rows, corpus_count=corpus_count)

    request_results: list[dict] = []
    low_frequency_positions = 0
    high_frequency_positions = 0
    eligible_positions = 0
    ticket_positions = 0
    ticket_requests = 0
    verified_gold_positions = 0
    on_ds_status_counts: Counter[str] = Counter()

    for row in rows:
        request_id = _request_id(row)
        pre_ds_ids = [str(item) for item in row["ids_in"]]
        historical_final_ids = [str(item) for item in row["ids_out"]]
        entity_results: list[dict] = []
        entity_by_id: dict[str, dict] = {}
        for anchor in row["anchors"]:
            if anchor.get("ent") is not True:
                continue
            pair = (request_id, str(anchor["id"]))
            entity = _entity_result(anchor, evidence_by_pair[pair])
            entity_results.append(entity)
            entity_by_id[entity["bucket_id"]] = entity
            if entity["frequency_class"] == "low":
                low_frequency_positions += 1
            else:
                high_frequency_positions += 1
            eligible_positions += int(entity["ticket_eligible"])
            verified_gold_positions += int(
                entity["verified_human_gold"] != "unlabelled"
            )

        # EntityStore.linked_bucket_ids is authoritative and orders links by
        # bucket_id.  The current task checkpoint keeps that stable resolver
        # order; it supersedes the incompatible 09:20 score-sort proposal.
        ticket_candidates = sorted(
            (
                item for item in entity_results if item["ticket_eligible"]
            ),
            key=lambda item: item["bucket_id"],
        )
        selected_tickets = ticket_candidates[:MAX_TICKETS_PER_REQUEST]
        ticket_in = [item["bucket_id"] for item in selected_tickets]
        for item in selected_tickets:
            item["ticket_selected"] = True
        ticket_positions += len(ticket_in)
        ticket_requests += bool(ticket_in)

        ordinary_pre_ds_ids = [
            bucket_id
            for bucket_id in pre_ds_ids
            if bucket_id not in entity_by_id
            or entity_by_id[bucket_id]["ordinary_without_entity"]
        ]
        ordinary_input_changed = ordinary_pre_ds_ids != pre_ds_ids
        definite_removed = [
            bucket_id
            for bucket_id in historical_final_ids
            if bucket_id in entity_by_id
            and not entity_by_id[bucket_id]["ordinary_without_entity"]
            and bucket_id not in ticket_in
        ]
        conditional_existing_ticket_ids = [
            bucket_id
            for bucket_id in historical_final_ids
            if bucket_id in ticket_in
        ]
        possible_new_ticket_ids = [
            bucket_id for bucket_id in ticket_in if bucket_id not in historical_final_ids
        ]
        retained_without_ticket_verdict = [
            bucket_id
            for bucket_id in historical_final_ids
            if bucket_id not in definite_removed
            and bucket_id not in conditional_existing_ticket_ids
        ]
        bounded_on_primary_ids = (
            None
            if ticket_in
            else [
                bucket_id
                for bucket_id in historical_final_ids
                if bucket_id not in definite_removed
            ]
        )

        outcome = row.get("ds_gate_outcome")
        historical_outcome = str(outcome) if outcome not in (None, "") else "missing"
        if ticket_in:
            on_ds_result = {
                "status": "unverified_ticket_input",
                "reason": "frozen_ledger_predates_ticket_path_no_per_ticket_verdict",
                "projected_same_call_ticket_ids": ticket_in,
                "historical_outcome": historical_outcome,
                "historical_gate_input_count": row.get("ds_gate_in"),
                "historical_gate_output_count": row.get("ds_gate_out"),
                "historical_same_pair_evidence": [
                    {
                        "bucket_id": bucket_id,
                        "historical_pre_ds_candidate": bucket_id in pre_ds_ids,
                        "historical_final_survivor": bucket_id in historical_final_ids,
                    }
                    for bucket_id in ticket_in
                ],
            }
            final_status = "unverified_ticket_semantic_verdict"
        elif ordinary_input_changed:
            on_ds_result = {
                "status": "unverified_changed_ordinary_input",
                "reason": "entity_score_removal_changed_the_ds_input",
                "historical_outcome": historical_outcome,
                "historical_gate_input_count": row.get("ds_gate_in"),
                "historical_gate_output_count": row.get("ds_gate_out"),
            }
            final_status = "bounded_primary_projection"
        else:
            on_ds_result = {
                "status": "historical_same_input_evidence",
                "reason": "ticket_contract_does_not_change_this_historical_ds_input",
                "historical_outcome": historical_outcome,
                "historical_gate_input_count": row.get("ds_gate_in"),
                "historical_gate_output_count": row.get("ds_gate_out"),
            }
            final_status = "bounded_primary_projection"
        on_ds_status_counts[on_ds_result["status"]] += 1

        request_results.append({
            "request_id": request_id,
            "entity_candidates": entity_results,
            "off": {
                "source": "frozen_historical_ledger",
                "pre_ds_candidate_ids": pre_ds_ids,
                "ds_result": {
                    "outcome": historical_outcome,
                    "input_count": row.get("ds_gate_in"),
                    "output_count": row.get("ds_gate_out"),
                    "exact_output_ids_available": False,
                },
                "final_ids": historical_final_ids,
            },
            "on": {
                "source": "deterministic_contract_projection_not_runtime_replay",
                "ordinary_pre_ds_ids_after_entity_score_removed": ordinary_pre_ds_ids,
                "ticket_in": ticket_in,
                "ds_result": on_ds_result,
                "final_change": {
                    "status": final_status,
                    "definite_removed_ids": definite_removed,
                    "retained_without_ticket_verdict": retained_without_ticket_verdict,
                    "conditional_existing_ticket_ids": conditional_existing_ticket_ids,
                    "possible_new_ticket_ids": possible_new_ticket_ids,
                    "bounded_on_primary_ids": bounded_on_primary_ids,
                    "end_to_end_final_ids": None,
                },
            },
        })

    outcome_counts = Counter(
        (
            str(row.get("ds_gate_outcome"))
            if row.get("ds_gate_outcome") not in (None, "")
            else "missing"
        )
        for row in rows
    )
    if outcome_counts["ok"] != EXPECTED_OK_REQUEST_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_OK_REQUEST_COUNT} historical DS ok rows")

    noise_request = next(
        row for row in request_results if row["request_id"] == NOISE_REQUEST_ID
    )
    target_request = next(
        row for row in request_results if row["request_id"] == TARGET_REQUEST_ID
    )
    noise_ticket_ids = set(noise_request["on"]["ticket_in"])
    noise_bounded_ids = set(
        noise_request["on"]["final_change"]["bounded_on_primary_ids"] or ()
    )
    target_entity = next(
        item
        for item in target_request["entity_candidates"]
        if item["bucket_id"] == TARGET_BUCKET_ID
    )
    target_pair_evidence = next(
        item
        for item in target_request["on"]["ds_result"]["historical_same_pair_evidence"]
        if item["bucket_id"] == TARGET_BUCKET_ID
    )

    verified_gold_unlabelled = EXPECTED_ENTITY_POSITION_COUNT - verified_gold_positions
    acceptance_reasons = [
        f"new_ticket_ds_receipt_missing:{ticket_requests}",
        "changed_ordinary_ds_input_receipt_missing:"
        f"{on_ds_status_counts['unverified_changed_ordinary_input']}",
        f"end_to_end_on_final_receipt_missing:{EXPECTED_REQUEST_COUNT}",
        f"verified_human_gold_incomplete:{verified_gold_unlabelled}",
        "batch_relevant_false_kill_unverified",
    ]

    return {
        "kind": "entity_ticket_frozen_evidence_projection_not_full_pipeline_replay",
        "proof_scope": {
            "off": "historical ledger evidence",
            "on": "eligibility and bounded primary projection only",
            "not_proved": [
                "new ticket-bearing DS semantic verdict",
                "new end-to-end ON final output",
                "batch relevant false-kill zero without verified human gold",
            ],
        },
        "privacy": {
            "private_inputs_not_group_or_world_readable": True,
            "query_text_emitted": False,
            "bucket_content_emitted": False,
            "entity_name_emitted": False,
        },
        "fixed_settings": {
            "literal_candidate_floor": LITERAL_CANDIDATE_FLOOR,
            "entity_max_df_ratio": ENTITY_MAX_DF_RATIO,
            "max_tickets_per_request": MAX_TICKETS_PER_REQUEST,
            "ticket_selection_order": "entity_store_bucket_id_order",
            "weak_entity_changes_score": False,
            "ticket_gate_failure_policy": "drop",
        },
        "provider_call_count": 0,
        "new_on_ds_call_count": 0,
        "new_on_ds_verified_request_count": 0,
        "on_ds_evidence_status_counts": {
            "historical_same_input_evidence": on_ds_status_counts[
                "historical_same_input_evidence"
            ],
            "unverified_changed_ordinary_input": on_ds_status_counts[
                "unverified_changed_ordinary_input"
            ],
            "unverified_ticket_input": on_ds_status_counts[
                "unverified_ticket_input"
            ],
        },
        "request_count": len(rows),
        "anchor_position_count": sum(len(row["anchors"]) for row in rows),
        "entity_position_count": len(evidence_by_pair),
        "entity_unique_bucket_count": len({pair[1] for pair in evidence_by_pair}),
        "low_frequency_entity_position_count": low_frequency_positions,
        "high_frequency_entity_position_count": high_frequency_positions,
        "ticket_eligible_position_count": eligible_positions,
        "ticket_in_position_count": ticket_positions,
        "ticket_request_count": ticket_requests,
        "historical_pre_ds_position_count": sum(len(row["ids_in"]) for row in rows),
        "historical_pre_ds_unique_count": len({
            str(bucket_id) for row in rows for bucket_id in row["ids_in"]
        }),
        "historical_final_position_count": sum(len(row["ids_out"]) for row in rows),
        "historical_ds_outcome_counts": {
            "ok": outcome_counts["ok"],
            "error": outcome_counts["error"],
            "missing": outcome_counts["missing"],
        },
        "query_provenance": {
            "original_query_sha_verified_count": provenance["original_query"],
            "production_effective_query_sha_verified_count": provenance[
                "production_effective_query"
            ],
            "fabricated_query_count": 0,
            "query_text_emitted": False,
        },
        "verified_human_gold_labelled_entity_position_count": verified_gold_positions,
        "verified_human_gold_unlabelled_entity_position_count": verified_gold_unlabelled,
        "batch_relevant_false_kill_count": None,
        "named_ticket_target_false_kill_count": None,
        "checks": {
            "engineering_noise": {
                "request_id": NOISE_REQUEST_ID,
                "bucket_ids": list(ENGINEERING_NOISE_IDS),
                "ticket_in_count": len(noise_ticket_ids.intersection(ENGINEERING_NOISE_IDS)),
                "not_ticketed": not noise_ticket_ids.intersection(ENGINEERING_NOISE_IDS),
                "absent_from_bounded_on_primary_final": not noise_bounded_ids.intersection(
                    ENGINEERING_NOISE_IDS
                ),
                "proof_status": "bounded_primary_projection",
            },
            "named_ticket_target": {
                "request_id": TARGET_REQUEST_ID,
                "bucket_id": TARGET_BUCKET_ID,
                "eligible": target_entity["ticket_eligible"],
                "ticket_in": TARGET_BUCKET_ID in target_request["on"]["ticket_in"],
                "projected_to_same_ds_call": TARGET_BUCKET_ID
                in target_request["on"]["ticket_in"],
                "historical_pre_ds_candidate": target_pair_evidence[
                    "historical_pre_ds_candidate"
                ],
                "historical_final_survivor": target_pair_evidence[
                    "historical_final_survivor"
                ],
                "new_ticket_ds_verdict": "unverified",
            },
        },
        "acceptance": False,
        "acceptance_reasons": acceptance_reasons,
        "minimum_supplement": {
            "same_turn_on_receipts_required": EXPECTED_REQUEST_COUNT,
            "required_receipt_fields": [
                "request_id",
                "ticket_in_ids",
                "ds_input_ids",
                "ds_output_ids",
                "ds_outcome",
                "final_ids",
            ],
            "human_gold_required_for_changed_entity_pairs": True,
            "fabricate_queries_or_gold": False,
        },
        "requests": request_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--entity-audit", required=True, type=Path)
    args = parser.parse_args()

    ledger, audit = load_inputs(args.ledger, args.entity_audit)
    result = replay(ledger, audit)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["acceptance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
