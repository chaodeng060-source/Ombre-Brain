#!/usr/bin/env python3
"""Replay the frozen real recall batch through the Y slot-reservation probe.

The input ledger contains exact production queries and responses.  This tool
does not call Ombre, an embedding provider or an LLM: it reuses the production
bucket mirror and the implementation's own ``_relation_recall_neighbors``
function.  Exact queries, IDs, bodies and concrete edges stay in a mode-0600
private receipt; the committed report is content-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server  # noqa: E402
import eval_relation_readout_baseline as baseline  # noqa: E402


PRIVATE_SCHEMA = "ombre-relation-slot-reservation-private/v1"
PUBLIC_SCHEMA = "ombre-relation-slot-reservation-content-free/v1"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _unique(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _classify(case: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    expected = set(case["expected"])
    acceptable = set(case["acceptable"])
    selected_set = set(selected)
    if case["cohort"] == "zero":
        explicit_noise = selected_set
        unjudged: set[str] = set()
    else:
        explicit_noise = selected_set & set(case["noise"])
        unjudged = selected_set - acceptable - set(case["noise"])
    first_rank = next(
        (rank for rank, item in enumerate(selected, 1) if item in expected),
        None,
    )
    return {
        "selected": len(selected),
        "expected_total": len(expected),
        "expected_hits": len(selected_set & expected),
        "acceptable_selected": len(selected_set & acceptable),
        "explicit_noise": len(explicit_noise),
        "unjudged": len(unjudged),
        "hit_at_1": bool(selected and selected[0] in expected),
        "hit_at_k": bool(selected_set & expected),
        "mrr": (1.0 / first_rank if first_rank else 0.0),
        "correct_zero": bool(case["cohort"] == "zero" and not selected),
    }


def _aggregate(receipts: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    miss = [row for row in receipts if row["cohort"] == "miss"]
    zero = [row for row in receipts if row["cohort"] == "zero"]
    values = [row[arm] for row in receipts]
    selected = sum(row["selected"] for row in values)
    expected_total = sum(row["expected_total"] for row in (item[arm] for item in miss))
    expected_hits = sum(row["expected_hits"] for row in (item[arm] for item in miss))
    explicit_noise = sum(row["explicit_noise"] for row in values)
    acceptable = sum(row["acceptable_selected"] for row in values)
    unjudged = sum(row["unjudged"] for row in values)
    return {
        "cases": len(receipts),
        "selected_items": selected,
        "hit_at_1_rate": round(sum(row[arm]["hit_at_1"] for row in miss) / len(miss), 6),
        "hit_at_k_rate": round(sum(row[arm]["hit_at_k"] for row in miss) / len(miss), 6),
        "mrr": round(sum(row[arm]["mrr"] for row in miss) / len(miss), 6),
        "completeness": round(expected_hits / expected_total, 6) if expected_total else 0.0,
        "expected_hits": expected_hits,
        "expected_total": expected_total,
        "explicit_noise_rate": round(explicit_noise / selected, 6) if selected else 0.0,
        "judged_noise_rate": round(explicit_noise / (explicit_noise + acceptable), 6)
        if explicit_noise + acceptable
        else 0.0,
        "explicit_noise_selected": explicit_noise,
        "unjudged_selected": unjudged,
        "correct_zero_rate": round(
            sum(row[arm]["correct_zero"] for row in zero) / len(zero), 6
        ),
        "correct_zero": sum(row[arm]["correct_zero"] for row in zero),
        "zero_controls": len(zero),
    }


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temp, 0o600)
    os.replace(temp, path)
    os.chmod(path, 0o600)


def _write_public(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _content_free(report: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    forbidden_keys = {
        "query",
        "case_id",
        "case_key",
        "bucket_id",
        "bucket_ids",
        "source_bucket_id",
        "target_bucket_id",
        "memory_body",
        "original_text",
        "raw_result",
    }
    private_values = {
        value
        for case in cases
        for value in (
            case["query"],
            case["case_id"],
            *case["expected"],
            *case["acceptable"],
            *case["noise"],
        )
        if value
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden_keys:
                    raise ValueError(f"public report contains private key: {key}")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value in private_values:
            raise ValueError("public report contains a private exact value")

    walk(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miss-ledger", type=Path, required=True)
    parser.add_argument("--zero-controls", type=Path, required=True)
    parser.add_argument("--baseline-private", type=Path, required=True)
    parser.add_argument("--bucket-store", type=Path, required=True)
    parser.add_argument("--production-policy", type=Path, required=True)
    parser.add_argument("--twin-evaluator", type=Path, required=True)
    parser.add_argument("--private-receipt", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--timing-iterations", type=int, default=5)
    args = parser.parse_args()
    if args.timing_iterations < 1:
        parser.error("--timing-iterations must be positive")

    gold = baseline._load_module(
        "relation_slot_reservation_gold",
        args.twin_evaluator.resolve(strict=True),
    )
    cases = gold.load_gold_cases(
        args.miss_ledger.resolve(strict=True),
        args.zero_controls.resolve(strict=True),
    )
    if (
        sum(case["cohort"] == "miss" for case in cases),
        sum(case["cohort"] == "zero" for case in cases),
    ) != (30, 10):
        raise ValueError("frozen batch must contain exactly 30 miss + 10 zero cases")

    source_ledger = baseline._read_json(args.baseline_private.resolve(strict=True))
    source_rows = {
        str(row.get("case_key")): row
        for row in source_ledger.get("cases", [])
        if isinstance(row, dict) and row.get("case_key")
    }
    if len(source_rows) != 40:
        raise ValueError("baseline private ledger must contain 40 cases")

    graph_buckets, bad_files = baseline._load_active_buckets(
        args.bucket_store.resolve(strict=True)
    )
    if bad_files:
        raise ValueError(f"active bucket graph has {len(bad_files)} unreadable files")
    graph_by_id = {
        str(bucket["id"]): bucket
        for bucket in graph_buckets
        if bucket.get("id")
    }
    relation_rows: list[tuple[str, str, str, str]] = []
    for bucket in graph_buckets:
        source_id = str(bucket.get("id") or "")
        relations = (bucket.get("metadata", {}) or {}).get("relations") or []
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            relation_rows.append(
                (
                    source_id,
                    str(relation.get("type") or ""),
                    str(relation.get("target") or ""),
                    str(relation.get("strength", 1.0)),
                )
            )
    relation_rows.sort()
    relation_manifest_sha256 = _sha_bytes(
        "\n".join("\t".join(row) for row in relation_rows).encode("utf-8")
    )
    policy = baseline._read_json(args.production_policy.resolve(strict=True))
    previous_config = server.config
    server.config = {**server.config, **policy}
    world_filter = {str(policy.get("current_world") or "").strip()}
    max_results = int(source_ledger.get("request_contract", {}).get("max_results", 5))

    private_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            row = source_rows.get(case["case_key"])
            if row is None:
                raise ValueError(f"missing baseline case ordinal {case['ordinal']}")
            entries = baseline._rendered_entries_before_relation(row["raw_result"])
            primary_ids = _unique(
                entry["bucket_id"] for entry in entries if entry["layer"] == "primary"
            )
            state_ids = _unique(
                entry["bucket_id"] for entry in entries if entry["layer"] == "z_lifecycle"
            )
            timeline_ids = _unique(
                entry["bucket_id"] for entry in entries if entry["layer"] == "x_timeline"
            )
            before_ids = _unique(entry["bucket_id"] for entry in entries)
            eligible = bool(
                int(row["relation_depth"]) >= 1
                and max_results > 1
                and len(primary_ids) > 1
                and len(before_ids) >= max_results
            )
            candidates = []
            timings_ms: list[float] = []
            if eligible:
                seed_ids = list(primary_ids[:-1])
                if not row["timeline_rendered"]:
                    seed_ids.extend(state_ids)
                excluded_ids = set(primary_ids) | set(state_ids) | set(timeline_ids)
                for _ in range(args.timing_iterations):
                    started = time.perf_counter_ns()
                    candidates = server._relation_recall_neighbors(
                        graph_buckets,
                        seed_ids,
                        query=row["request_query"],
                        intent=row["effective_intent"],
                        world_filter=world_filter,
                        domain_filter=[],
                        created_after=None,
                        created_before=None,
                        max_depth=int(row["relation_depth"]),
                        max_results=1,
                        excluded_ids=excluded_ids,
                    )
                    timings_ms.append(
                        (time.perf_counter_ns() - started) / 1_000_000
                    )

            reserved = bool(candidates)
            baseline_ids = _unique(row["all_returned_bucket_ids"])
            candidate_ids = list(baseline_ids)
            dropped_primary_id = ""
            edge_rows: list[dict[str, Any]] = []
            if reserved:
                dropped_primary_id = primary_ids[-1]
                selected = candidates[0]
                retained_before = [
                    bucket_id for bucket_id in before_ids
                    if bucket_id != dropped_primary_id
                ]
                trailing = [
                    bucket_id for bucket_id in baseline_ids
                    if bucket_id not in before_ids
                ]
                candidate_ids = _unique(
                    [*retained_before, selected.bucket_id, *trailing]
                )
                source = graph_by_id.get(str(selected.via_id), {})
                target = graph_by_id.get(str(selected.bucket_id), {})
                edge_rows.append(
                    {
                        "relation_type": selected.relation_type,
                        "direction": selected.direction,
                        "depth": selected.depth,
                        "source_bucket_id": selected.via_id,
                        "target_bucket_id": selected.bucket_id,
                        "strength": selected.strength,
                        "source_original_text": str(source.get("content") or ""),
                        "target_original_text": str(target.get("content") or ""),
                    }
                )

            baseline_score = _classify(case, baseline_ids)
            candidate_score = _classify(case, candidate_ids)
            private_rows.append(
                {
                    "ordinal": case["ordinal"],
                    "cohort": case["cohort"],
                    "case_key": case["case_key"],
                    "query": case["query"],
                    "request_query": row["request_query"],
                    "eligible_for_preflight": eligible,
                    "slot_reserved": reserved,
                    "primary_ids": primary_ids,
                    "before_relation_ids": before_ids,
                    "baseline_ids": baseline_ids,
                    "candidate_ids": candidate_ids,
                    "dropped_primary_id": dropped_primary_id,
                    "concrete_edges": edge_rows,
                    "preflight_timings_ms": [round(value, 6) for value in timings_ms],
                    "baseline": baseline_score,
                    "candidate": candidate_score,
                    "baseline_raw_result_sha256": _sha_bytes(
                        str(row["raw_result"]).encode("utf-8")
                    ),
                }
            )
            public_rows.append(
                {
                    "ordinal": case["ordinal"],
                    "cohort": case["cohort"],
                    "eligible_for_preflight": eligible,
                    "slot_reserved": reserved,
                    "concrete_edge_count": len(edge_rows),
                    "selection_changed": baseline_ids != candidate_ids,
                    "result_count_before": len(baseline_ids),
                    "result_count_after": len(candidate_ids),
                    "preflight_median_ms": round(
                        sorted(timings_ms)[len(timings_ms) // 2], 6
                    ) if timings_ms else 0.0,
                    "baseline": baseline_score,
                    "candidate": candidate_score,
                }
            )
    finally:
        server.config = previous_config

    now = datetime.now(timezone.utc).isoformat()
    source = {
        "miss_ledger_sha256": _sha_file(args.miss_ledger),
        "zero_controls_sha256": _sha_file(args.zero_controls),
        "baseline_private_sha256": _sha_file(args.baseline_private),
        "production_policy_sha256": _sha_file(args.production_policy),
        "server_sha256": _sha_file(ROOT / "server.py"),
        "recall_timing_sha256": _sha_file(ROOT / "recall_timing.py"),
        "evaluator_sha256": _sha_file(Path(__file__).resolve()),
        "graph": {
            "active_buckets": len(graph_buckets),
            "relation_rows": len(relation_rows),
            "relation_types": dict(
                sorted(Counter(row[1] for row in relation_rows).items())
            ),
            "relation_manifest_sha256": relation_manifest_sha256,
        },
        "production": source_ledger.get("source", {}),
    }
    private_payload = {
        "schema": PRIVATE_SCHEMA,
        "created_at": now,
        "source": source,
        "cost": {
            "external_api_calls": 0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "provider_calls": 0,
        },
        "cases": private_rows,
    }
    _write_private(args.private_receipt, private_payload)

    baseline_metrics = _aggregate(public_rows, "baseline")
    candidate_metrics = _aggregate(public_rows, "candidate")
    preflight_medians = [row["preflight_median_ms"] for row in public_rows]
    remote_times = [float(row["remote_ms"]) for row in source_ledger["cases"]]
    projected_times = [
        float(source_row["remote_ms"]) + public_row["preflight_median_ms"]
        for source_row, public_row in zip(source_ledger["cases"], public_rows)
    ]
    remote_p95 = _p95(remote_times)
    projected_p95 = _p95(projected_times)
    p95_regression = (
        ((projected_p95 - remote_p95) / remote_p95) * 100.0
        if remote_p95 > 0
        else 0.0
    )
    relation_baseline = sum(
        int(row.get("relation_evidence_count") or 0)
        for row in source_ledger["cases"]
    )
    reserved_cases = sum(row["slot_reserved"] for row in public_rows)
    zero_changed = sum(
        row["selection_changed"]
        for row in public_rows
        if row["cohort"] == "zero"
    )
    public_payload = {
        "schema": PUBLIC_SCHEMA,
        "created_at": now,
        "source": source,
        "private_receipt": {
            "file": args.private_receipt.name,
            "sha256": _sha_file(args.private_receipt),
            "mode": oct(args.private_receipt.stat().st_mode & 0o777),
            "queries_in_public_report": False,
            "bucket_ids_in_public_report": False,
            "memory_bodies_in_public_report": False,
        },
        "scope": {
            "cases": len(public_rows),
            "miss_cases": sum(row["cohort"] == "miss" for row in public_rows),
            "zero_controls": sum(row["cohort"] == "zero" for row in public_rows),
            "frozen_real_production_queries": True,
            "current_production_bucket_mirror": True,
            "live_deployment_evaluated": False,
        },
        "comparison": {
            "eligible_preflight_cases": sum(
                row["eligible_for_preflight"] for row in public_rows
            ),
            "slot_reserved_cases": reserved_cases,
            "concrete_relation_edges": sum(
                row["concrete_edge_count"] for row in public_rows
            ),
            "selection_changed_cases": sum(
                row["selection_changed"] for row in public_rows
            ),
            "zero_control_changes": zero_changed,
            "return_count_changes": sum(
                row["result_count_before"] != row["result_count_after"]
                for row in public_rows
            ),
            "relation_evidence_baseline": relation_baseline,
            "relation_evidence_projected": relation_baseline + reserved_cases,
        },
        "metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        },
        "latency": {
            "production_baseline_p95_ms": round(remote_p95, 6),
            "local_preflight_p95_ms": round(_p95(preflight_medians), 6),
            "projected_total_p95_ms": round(projected_p95, 6),
            "projected_p95_regression_percent": round(p95_regression, 6),
            "production_paired_replay_required": True,
        },
        "cost": {
            "additional_api_calls": 0,
            "additional_llm_calls": 0,
            "additional_embedding_calls": 0,
            "additional_provider_calls": 0,
        },
        "case_receipts": public_rows,
        "batch_gate": {
            "status": "pass_for_independent_review"
            if (
                reserved_cases >= 3
                and zero_changed == 0
                and p95_regression <= 20.0
                and candidate_metrics["hit_at_1_rate"] >= baseline_metrics["hit_at_1_rate"]
                and candidate_metrics["hit_at_k_rate"] >= baseline_metrics["hit_at_k_rate"]
                and candidate_metrics["completeness"] >= baseline_metrics["completeness"]
                and candidate_metrics["explicit_noise_rate"] <= baseline_metrics["explicit_noise_rate"]
            )
            else "hold",
            "production_rollout_authorized": False,
            "reason": "GLM review and Claude paired production replay are still required",
        },
        "release_state": {
            "code_committed": False,
            "pushed": False,
            "deployed": False,
            "process_loaded": False,
            "health_checked": False,
            "real_production_replay_accepted": False,
            "independent_glm_reviewed": False,
        },
    }
    _content_free(public_payload, cases)
    _write_public(args.public_report, public_payload)
    print(
        json.dumps(
            {
                "comparison": public_payload["comparison"],
                "metrics": public_payload["metrics"],
                "latency": public_payload["latency"],
                "batch_gate": public_payload["batch_gate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if public_payload["batch_gate"]["status"] == "pass_for_independent_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
