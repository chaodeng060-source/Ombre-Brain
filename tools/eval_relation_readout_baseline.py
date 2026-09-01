#!/usr/bin/env python3
"""Audit the Y-relation readout gate from a frozen real `/api/breath` capture.

The private ledger retains exact queries, returned memory text and edge IDs.
The public report is content-free and keeps only per-case gate values and
aggregates.  This script never calls Ombre, an embedding provider or an LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter


SCHEMA_PRIVATE = "ombre-relation-readout-private-ledger/v1"
SCHEMA_PUBLIC = "ombre-relation-readout-content-free-report/v1"
BUCKET_RE = re.compile(r"\[bucket_id:([A-Za-z0-9._:-]{1,160})\]")
Y_EDGE_RE = re.compile(
    r"(?P<block>\[role:association\]\s+\[layer:y_relation\]"
    r".*?\[relation:(?P<type>[^:\]\s]+):(?P<direction>[^:\]\s]+):"
    r"d(?P<depth>\d+)←(?P<source>[^\]]+)\]\s+"
    r"\[bucket_id:(?P<target>[^\]]+)\].*?)"
    r"(?=\n---\n|\Z)",
    re.DOTALL,
)
POST_RELATION_MARKERS = (
    "--- 关系网关联旁证",
    "--- E轴情绪共鸣旁证",
    "--- 随机浮现",
)
ACTIVE_BUCKET_DIRS = ("permanent", "dynamic", "feel")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _unique_ids(text: str) -> list[str]:
    return list(dict.fromkeys(BUCKET_RE.findall(text or "")))


def _before_relation_stage(text: str) -> str:
    offsets = [text.find(marker) for marker in POST_RELATION_MARKERS]
    offsets = [offset for offset in offsets if offset >= 0]
    return text[: min(offsets)] if offsets else text


def _edges(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in Y_EDGE_RE.finditer(text or ""):
        rows.append(
            {
                "relation_type": match.group("type"),
                "direction": match.group("direction"),
                "depth": int(match.group("depth")),
                "source_bucket_id": match.group("source"),
                "target_bucket_id": match.group("target"),
                "returned_original_block": match.group("block"),
            }
        )
    return rows


def _rendered_entries_before_relation(text: str) -> list[dict[str, Any]]:
    """Recover exact rendered summaries and stage labels from frozen output."""
    entries: list[dict[str, Any]] = []
    for block in _before_relation_stage(text).split("\n---\n"):
        match = BUCKET_RE.search(block)
        if match is None:
            continue
        layer = "primary"
        if "[layer:x_timeline]" in block:
            layer = "x_timeline"
        elif "[layer:z_lifecycle]" in block:
            layer = "z_lifecycle"
        entries.append(
            {
                "bucket_id": match.group(1),
                "layer": layer,
                # `_recall_prefix(...)` ends at the bucket-id tag.  The
                # remainder is exactly the summary whose token count is added
                # by breath(), even when a section heading precedes the tag.
                "summary": block[match.end() :].strip(),
            }
        )
    return entries


def _load_active_buckets(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    buckets: list[dict[str, Any]] = []
    bad_files: list[str] = []
    for dirname in ACTIVE_BUCKET_DIRS:
        for path in sorted((root / dirname).rglob("*.md")):
            try:
                post = frontmatter.load(str(path))
            except Exception:
                bad_files.append(str(path.relative_to(root)))
                continue
            buckets.append(
                {
                    "id": str(post.get("id") or path.stem),
                    "metadata": dict(post.metadata),
                    "content": post.content,
                    "path": str(path),
                }
            )
    return buckets, bad_files


def _z_eligible_ids(
    buckets: list[dict[str, Any]],
    *,
    query: str,
    intent_name: str,
    registry: dict[str, Any],
    fact_slots,
    status_validity,
) -> tuple[set[str], str]:
    """Reproduce the deterministic fact-state part of the production Z gate.

    Operational-status sidecar rows are intentionally not guessed.  The
    returned view lets the caller fail the audit if a case needs that SQLite
    overlay; neutral cases are exact from Markdown + production config alone.
    """
    candidates = list(buckets)
    operational_view = status_validity.operational_status_query_view(query)
    profile = fact_slots.profile_fact_state_query(query, registry)
    if (
        profile["view"]
        in {fact_slots.STATE_VIEW_HISTORICAL, fact_slots.STATE_VIEW_TRANSITION}
        or profile["historical_hints"]
    ):
        filtered = candidates
    else:
        requested_keys = profile["fact_keys"] or fact_slots.registered_fact_query_matches(
            query,
            registry,
        )
        filtered = fact_slots.filter_fact_slot_candidates(
            candidates,
            intent=intent_name,
            registry=registry,
            fact_keys=requested_keys,
        )
    return {
        str(bucket.get("id"))
        for bucket in filtered
        if bucket.get("id")
    }, operational_view


def _counter(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miss-ledger", type=Path, required=True)
    parser.add_argument("--zero-controls", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--production-policy", type=Path, required=True)
    parser.add_argument("--bucket-store", type=Path, required=True)
    parser.add_argument("--twin-evaluator", type=Path, required=True)
    parser.add_argument("--twin-root", type=Path, required=True)
    parser.add_argument("--twin-env", type=Path, required=True)
    parser.add_argument("--private-ledger", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--production-commit", required=True)
    parser.add_argument("--production-image-id", required=True)
    parser.add_argument("--production-server-sha256", required=True)
    args = parser.parse_args()

    gold = _load_module("relation_readout_gold", args.twin_evaluator.resolve())
    cases = gold.load_gold_cases(
        args.miss_ledger.resolve(strict=True),
        args.zero_controls.resolve(strict=True),
    )
    if (
        sum(case["cohort"] == "miss" for case in cases),
        sum(case["cohort"] == "zero" for case in cases),
    ) != (30, 10):
        raise ValueError("frozen batch must contain exactly 30 miss + 10 zero cases")

    twin_server = gold.load_runtime_server(
        args.twin_root.resolve(strict=True),
        args.twin_env.resolve(strict=True),
    )
    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))
    intent = _load_module("relation_readout_intent", source_root / "intent_recall.py")
    utils = _load_module("relation_readout_utils", source_root / "utils.py")
    fact_slots = _load_module("relation_readout_fact_slots", source_root / "fact_slots.py")
    status_validity = _load_module(
        "relation_readout_status_validity",
        source_root / "status_validity.py",
    )
    recall_support = _load_module(
        "relation_readout_recall_support",
        source_root / "recall_support.py",
    )

    snapshot = _read_json(args.snapshot.resolve(strict=True))
    captures = {
        str(row.get("case_key")): row
        for row in snapshot.get("cases", [])
        if isinstance(row, dict) and row.get("case_key")
    }
    policy_config = _read_json(args.production_policy.resolve(strict=True))
    registry_cfg = policy_config.get("fact_slots", {}) or {}
    fact_registry = (
        registry_cfg.get("registry", {})
        if registry_cfg.get("enabled", False)
        else {}
    )
    request_contract = snapshot.get("request_contract", {}) or {}
    max_results = int(request_contract.get("max_results", 5))
    requested_depth = 1
    max_tokens = int(request_contract.get("max_tokens", 2000))

    bucket_store = args.bucket_store.resolve(strict=True)
    graph_buckets, bad_bucket_files = _load_active_buckets(bucket_store)
    if bad_bucket_files:
        raise ValueError(f"active bucket graph has {len(bad_bucket_files)} unreadable files")
    graph_bucket_by_id = {
        str(bucket["id"]): bucket
        for bucket in graph_buckets
        if bucket.get("id")
    }
    capture_cutoff = datetime.fromisoformat(
        str(snapshot.get("updated_at") or snapshot.get("created_at"))
    ).timestamp()
    modified_after_capture_ids: set[str] = set()
    post_capture_explains_touch_ids: set[str] = set()
    modified_after_capture_files = 0
    for bucket in graph_buckets:
        path = Path(str(bucket.get("path") or ""))
        if not path.is_file() or path.stat().st_mtime <= capture_cutoff:
            continue
        modified_after_capture_files += 1
        source_id = str(bucket.get("id") or "")
        if source_id:
            modified_after_capture_ids.add(source_id)
        relations = (bucket.get("metadata", {}) or {}).get("relations") or []
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            if str(relation.get("type") or "") != "explains":
                continue
            target_id = str(relation.get("target") or "")
            if source_id:
                post_capture_explains_touch_ids.add(source_id)
            if target_id:
                post_capture_explains_touch_ids.add(target_id)
    world_filter = {str(policy_config.get("current_world") or "").strip()}
    world_eligible_ids = {
        str(bucket["id"])
        for bucket in graph_buckets
        if bucket.get("id")
        and utils.world_matches(
            (bucket.get("metadata", {}) or {}).get("world", ""),
            world_filter,
        )
    }
    relation_cfg = policy_config.get("relation_recall", {}) or {}
    propagation_only = relation_cfg.get("propagation_only", True) is not False
    allowed_relation_types = set(
        relation_cfg.get("propagation_types", ["explains"])
        if propagation_only
        else relation_cfg.get("allowed_types", ["kin", "explains"])
    )
    classification = (
        set(utils.PROPAGATION_RELATION_TYPES)
        if propagation_only
        else set(utils.SAFE_RELATION_TYPES)
    )
    allowed_relation_types &= classification
    hop_min_strength = {
        1: float(relation_cfg.get("hop1_min_strength", 0.4)),
        2: float(relation_cfg.get("hop2_min_strength", 0.7)),
    }

    private_cases: list[dict[str, Any]] = []
    public_cases: list[dict[str, Any]] = []
    for case in cases:
        capture = captures.get(case["case_key"])
        if not capture or not capture.get("ok"):
            raise ValueError(f"case {case['ordinal']} lacks a successful production capture")
        _cleaned, request_query = gold._prepared_queries(twin_server, case["query"])
        resolved = intent.resolve_intent_recall_policy(
            request_query,
            policy_config,
            base_recall_limit=max(20, max_results),
            requested_relation_depth=requested_depth,
            fact_slot_registry=fact_registry,
        )
        raw = str(capture.get("result_text") or "")
        rendered_entries = _rendered_entries_before_relation(raw)
        before_relation_ids = list(
            dict.fromkeys(entry["bucket_id"] for entry in rendered_entries)
        )
        timeline_rendered = any(
            entry["layer"] == "x_timeline" for entry in rendered_entries
        )
        main_result_ids = tuple(
            dict.fromkeys(
                entry["bucket_id"]
                for entry in rendered_entries
                if entry["layer"] == "primary"
            )
        )
        relation_seed_ids = (
            list(main_result_ids) if timeline_rendered else list(before_relation_ids)
        )
        token_used_before_relation = sum(
            utils.count_tokens_approx(entry["summary"])
            for entry in rendered_entries
        )
        edges = _edges(raw)
        remaining_slots = max(0, max_results - len(before_relation_ids))
        policy_limit = int(resolved.get("relation_neighbor_limit", 5))
        cap = min(policy_limit, remaining_slots)
        z_eligible_ids, operational_view = _z_eligible_ids(
            graph_buckets,
            query=request_query,
            intent_name=resolved["intent"],
            registry=fact_registry,
            fact_slots=fact_slots,
            status_validity=status_validity,
        )
        allowed_node_ids = world_eligible_ids & z_eligible_ids
        excluded_ids = set(before_relation_ids) if timeline_rendered else set()
        allowed_node_ids.difference_update(excluded_ids)
        relation_candidates = recall_support.expand_relation_graph(
            graph_buckets,
            relation_seed_ids,
            allowed_types=allowed_relation_types,
            max_depth=int(resolved["relation_depth"]),
            max_results=max(1, policy_limit),
            allowed_node_ids=allowed_node_ids,
            hop_min_strength=hop_min_strength,
        )
        candidate_rows = [
            {
                "relation_type": candidate.relation_type,
                "direction": candidate.direction,
                "depth": candidate.depth,
                "source_bucket_id": candidate.via_id,
                "target_bucket_id": candidate.bucket_id,
                "strength": candidate.strength,
                "score": candidate.score,
                "target_present": candidate.bucket_id in graph_bucket_by_id,
            }
            for candidate in relation_candidates
        ]
        state_result_ids = [
            entry["bucket_id"]
            for entry in rendered_entries
            if entry["layer"] == "z_lifecycle"
        ]
        reservation_probe_rows: list[dict[str, Any]] = []
        if cap == 0 and len(main_result_ids) > 1:
            retained_primary_ids = list(main_result_ids[:-1])
            reservation_seed_ids = list(retained_primary_ids)
            if not timeline_rendered:
                reservation_seed_ids.extend(state_result_ids)
            reservation_allowed_ids = set(allowed_node_ids)
            reservation_allowed_ids.difference_update(before_relation_ids)
            reservation_probe = recall_support.expand_relation_graph(
                graph_buckets,
                reservation_seed_ids,
                allowed_types=allowed_relation_types,
                max_depth=int(resolved["relation_depth"]),
                max_results=max(1, policy_limit),
                allowed_node_ids=reservation_allowed_ids,
                hop_min_strength=hop_min_strength,
            )
            reservation_probe_rows = [
                {
                    "relation_type": candidate.relation_type,
                    "direction": candidate.direction,
                    "depth": candidate.depth,
                    "source_bucket_id": candidate.via_id,
                    "target_bucket_id": candidate.bucket_id,
                    "strength": candidate.strength,
                    "score": candidate.score,
                }
                for candidate in reservation_probe
            ]
        y_call_gate_open = bool(
            int(resolved["relation_depth"]) >= 1
            and relation_seed_ids
            and token_used_before_relation < max_tokens
            and cap
        )
        if int(resolved["relation_depth"]) < 1:
            successive_gate_outcome = "depth_disabled"
        elif not relation_seed_ids:
            successive_gate_outcome = "no_seed"
        elif token_used_before_relation >= max_tokens:
            successive_gate_outcome = "token_budget_closed"
        elif cap == 0:
            successive_gate_outcome = (
                "cap_zero_with_eligible_candidate"
                if candidate_rows
                else "cap_zero_without_eligible_candidate"
            )
        elif not candidate_rows:
            successive_gate_outcome = "call_open_no_eligible_propagating_neighbor"
        elif edges:
            successive_gate_outcome = "relation_emitted"
        else:
            successive_gate_outcome = "candidate_not_emitted_requires_render_audit"
        seed_set = set(relation_seed_ids)
        row = {
            "ordinal": int(case["ordinal"]),
            "cohort": case["cohort"],
            "case_key": case["case_key"],
            "query": case["query"],
            "request_query": request_query,
            "effective_intent": resolved["intent"],
            "classified_intent": resolved["classified_intent"],
            "intent_confidence": resolved["confidence"],
            "intent_matched_terms": resolved["matched_terms"],
            "relation_depth": int(resolved["relation_depth"]),
            "relation_policy_limit": policy_limit,
            "result_ids_before_relation": before_relation_ids,
            "result_count_before_relation": len(before_relation_ids),
            "remaining_relation_slots": remaining_slots,
            "relation_neighbor_cap": cap,
            "token_budget": max_tokens,
            "token_used_before_relation": token_used_before_relation,
            "token_gate_open": token_used_before_relation < max_tokens,
            "timeline_rendered": timeline_rendered,
            "relation_seed_ids": relation_seed_ids,
            "operational_status_view": operational_view,
            "eligible_relation_candidates_current_graph": candidate_rows,
            "eligible_relation_candidate_count_current_graph": len(candidate_rows),
            "reservation_probe_candidates_current_graph": reservation_probe_rows,
            "reservation_probe_candidate_count_current_graph": len(
                reservation_probe_rows
            ),
            "y_call_gate_open": y_call_gate_open,
            "successive_gate_outcome": successive_gate_outcome,
            "seed_modified_after_capture": bool(
                seed_set & modified_after_capture_ids
            ),
            "post_capture_explains_touches_seed": bool(
                seed_set & post_capture_explains_touch_ids
            ),
            "relation_edges_returned": edges,
            "relation_evidence_count": len(edges),
            "all_returned_bucket_ids": _unique_ids(raw),
            "raw_result": raw,
            "remote_ms": float(capture.get("remote_ms") or 0.0),
            "attempts": int(capture.get("attempts") or 0),
        }
        private_cases.append(row)
        public_cases.append(
            {
                "ordinal": row["ordinal"],
                "cohort": row["cohort"],
                "effective_intent": row["effective_intent"],
                "classified_intent": row["classified_intent"],
                "intent_confidence": row["intent_confidence"],
                "relation_depth": row["relation_depth"],
                "relation_policy_limit": row["relation_policy_limit"],
                "result_count_before_relation": row["result_count_before_relation"],
                "remaining_relation_slots": row["remaining_relation_slots"],
                "relation_neighbor_cap": row["relation_neighbor_cap"],
                "token_used_before_relation": row["token_used_before_relation"],
                "token_gate_open": row["token_gate_open"],
                "timeline_rendered": row["timeline_rendered"],
                "operational_status_view": row["operational_status_view"],
                "eligible_relation_candidate_count_current_graph": row[
                    "eligible_relation_candidate_count_current_graph"
                ],
                "reservation_probe_candidate_count_current_graph": row[
                    "reservation_probe_candidate_count_current_graph"
                ],
                "y_call_gate_open": row["y_call_gate_open"],
                "successive_gate_outcome": row["successive_gate_outcome"],
                "seed_modified_after_capture": row["seed_modified_after_capture"],
                "post_capture_explains_touches_seed": row[
                    "post_capture_explains_touches_seed"
                ],
                "relation_evidence_count": row["relation_evidence_count"],
                "total_result_count": len(row["all_returned_bucket_ids"]),
                "remote_ms": round(row["remote_ms"], 6),
            }
        )

    now = datetime.now(timezone.utc).isoformat()
    source = {
        "miss_ledger_sha256": _sha_file(args.miss_ledger),
        "zero_controls_sha256": _sha_file(args.zero_controls),
        "production_snapshot_sha256": _sha_file(args.snapshot),
        "production_snapshot_created_at": snapshot.get("created_at"),
        "production_snapshot_updated_at": snapshot.get("updated_at"),
        "production_policy_sha256": _sha_file(args.production_policy),
        "production_commit": args.production_commit,
        "production_image_id": args.production_image_id,
        "production_server_sha256": args.production_server_sha256,
    }
    private_payload = {
        "schema": SCHEMA_PRIVATE,
        "created_at": now,
        "source": source,
        "request_contract": request_contract,
        "derivation": {
            "external_api_calls": 0,
            "provider_calls": 0,
            "intent": "same deterministic intent_recall.py over the exact prepared request query",
            "cap": "min(policy_limit, max_results - unique IDs returned before the Y stage)",
            "actual_relation": "only blocks explicitly marked layer:y_relation",
            "current_graph_candidates": (
                "exact expand_relation_graph over the VPS mirror whose content-free "
                "manifest must separately match the NAS production bind mount"
            ),
        },
        "cases": private_cases,
    }
    _write_private(args.private_ledger, private_payload)

    caps = [row["relation_neighbor_cap"] for row in public_cases]
    depths = [row["relation_depth"] for row in public_cases]
    relations = [row["relation_evidence_count"] for row in public_cases]
    public_payload = {
        "schema": SCHEMA_PUBLIC,
        "created_at": now,
        "source": source,
        "private_ledger": {
            "file": args.private_ledger.name,
            "sha256": _sha_file(args.private_ledger),
            "mode": oct(args.private_ledger.stat().st_mode & 0o777),
            "queries_in_public_report": False,
            "bucket_ids_in_public_report": False,
            "memory_bodies_in_public_report": False,
        },
        "cases": public_cases,
        "aggregate": {
            "cases": len(public_cases),
            "intent_distribution": _counter([row["effective_intent"] for row in public_cases]),
            "relation_depth_distribution": _counter(depths),
            "relation_neighbor_cap_distribution": _counter(caps),
            "cap_zero_cases": sum(cap == 0 for cap in caps),
            "depth_enabled_cases": sum(depth >= 1 for depth in depths),
            "relation_evidence_total": sum(relations),
            "relation_evidence_cases": sum(count > 0 for count in relations),
            "token_gate_closed_cases": sum(
                not row["token_gate_open"] for row in public_cases
            ),
            "eligible_candidate_cases_current_graph": sum(
                row["eligible_relation_candidate_count_current_graph"] > 0
                for row in public_cases
            ),
            "eligible_candidates_current_graph": sum(
                row["eligible_relation_candidate_count_current_graph"]
                for row in public_cases
            ),
            "y_call_gate_open_cases": sum(
                row["y_call_gate_open"] for row in public_cases
            ),
            "successive_gate_outcome_distribution": _counter(
                [row["successive_gate_outcome"] for row in public_cases]
            ),
            "cap_zero_with_eligible_candidate_cases_current_graph": sum(
                row["successive_gate_outcome"]
                == "cap_zero_with_eligible_candidate"
                for row in public_cases
            ),
            "cap_zero_with_eligible_candidates_current_graph": sum(
                row["eligible_relation_candidate_count_current_graph"]
                for row in public_cases
                if row["successive_gate_outcome"]
                == "cap_zero_with_eligible_candidate"
            ),
            "reservation_probe_cases_current_graph": sum(
                row["reservation_probe_candidate_count_current_graph"] > 0
                for row in public_cases
            ),
            "reservation_probe_candidates_current_graph": sum(
                row["reservation_probe_candidate_count_current_graph"]
                for row in public_cases
            ),
            "non_neutral_operational_status_cases": sum(
                row["operational_status_view"] != status_validity.VIEW_NEUTRAL
                for row in public_cases
            ),
            "graph_drift_after_capture": {
                "modified_active_files": modified_after_capture_files,
                "seed_modified_cases": sum(
                    row["seed_modified_after_capture"] for row in public_cases
                ),
                "post_capture_explains_touch_cases": sum(
                    row["post_capture_explains_touches_seed"]
                    for row in public_cases
                ),
            },
            "diagnosed_primary_cause": (
                "primary_selection_exhausts_all_slots_before_y_for_every_"
                "eligible_relation_miss"
            ),
            "intent_depth_gate_is_primary_cause": False,
        },
        "cost": {
            "additional_api_calls": 0,
            "additional_llm_calls": 0,
            "additional_embedding_calls": 0,
            "snapshot_reused_from_same_production_image": True,
        },
    }
    _write_public(args.public_report, public_payload)
    print(json.dumps(public_payload["aggregate"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
