#!/usr/bin/env python3
"""Audit, dry-run, or apply the authoritative Ombre Y-axis backfill."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from bucket_manager import BucketManager
from relation_graph import (
    LEGACY_GENERATION_METHOD,
    PlannedRelation,
    RelationGraphPlan,
    legacy_relation_evidence,
    normalize_relation_evidence,
    plan_relation_graph,
)
from snapshot_manager import SnapshotManager
from utils import RELATION_TYPES, load_config


LEDGER_SCHEMA = "ombre.relation-backfill-ledger/v1"
ACTOR = "operator:y-relation-backfill:v1"


def _metadata(bucket: Mapping[str, Any]) -> Mapping[str, Any]:
    value = bucket.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _bucket_id(bucket: Mapping[str, Any]) -> str:
    return str(bucket.get("id") or _metadata(bucket).get("id") or "").strip()


def _kin_key(source_id: str, target_id: str) -> tuple[str, str, str]:
    left, right = sorted((source_id, target_id))
    return left, "kin", right


def _plan_key(item: PlannedRelation) -> tuple[str, str, str]:
    if item.relation_type == "kin":
        return _kin_key(item.source_id, item.target_id)
    return item.source_id, item.relation_type, item.target_id


def _existing_key(
    source_id: str,
    relation_type: str,
    target_id: str,
) -> tuple[str, str, str]:
    if relation_type == "kin":
        return _kin_key(source_id, target_id)
    return source_id, relation_type, target_id


def audit_existing_relations(
    buckets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ids = {_bucket_id(bucket) for bucket in buckets if _bucket_id(bucket)}
    by_type: Counter[str] = Counter()
    sources: set[str] = set()
    exact_seen: set[tuple[str, str, str]] = set()
    undirected_kin_seen: set[tuple[str, str, str]] = set()
    malformed = 0
    missing_target: list[dict[str, str]] = []
    self_loops: list[dict[str, str]] = []
    exact_duplicates = 0
    reverse_kin_duplicates = 0
    missing_evidence = 0
    missing_generation_method = 0
    relation_count = 0

    for bucket in buckets:
        source_id = _bucket_id(bucket)
        relations = _metadata(bucket).get("relations") or []
        if not isinstance(relations, list):
            malformed += 1
            continue
        for relation in relations:
            if not isinstance(relation, Mapping):
                malformed += 1
                continue
            relation_type = str(relation.get("type") or "").strip()
            target_id = str(relation.get("target") or "").strip()
            if relation_type not in RELATION_TYPES or not target_id:
                malformed += 1
                continue
            relation_count += 1
            sources.add(source_id)
            by_type[relation_type] += 1
            exact = (source_id, relation_type, target_id)
            exact_was_seen = exact in exact_seen
            if exact_was_seen:
                exact_duplicates += 1
            exact_seen.add(exact)
            if relation_type == "kin":
                kin = _kin_key(source_id, target_id)
                if kin in undirected_kin_seen and not exact_was_seen:
                    reverse_kin_duplicates += 1
                undirected_kin_seen.add(kin)
            if target_id not in ids:
                missing_target.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": relation_type,
                })
            if source_id == target_id:
                self_loops.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": relation_type,
                })
            if not relation.get("evidence"):
                missing_evidence += 1
            if not str(relation.get("generation_method") or "").strip():
                missing_generation_method += 1

    return {
        "relation_count": relation_count,
        "source_bucket_count": len(sources),
        "by_type": dict(sorted(by_type.items())),
        "malformed_count": malformed,
        "missing_target_count": len(missing_target),
        "missing_targets": missing_target,
        "self_loop_count": len(self_loops),
        "self_loops": self_loops,
        "exact_duplicate_count": exact_duplicates,
        "reverse_kin_duplicate_count": reverse_kin_duplicates,
        "missing_evidence_count": missing_evidence,
        "missing_generation_method_count": missing_generation_method,
    }


def build_cleanup_plan(
    buckets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, Any]]:
    """Plan exact, recoverable removal of broken or redundant legacy edges."""
    ids = {_bucket_id(bucket) for bucket in buckets if _bucket_id(bucket)}
    kin_edges: dict[tuple[str, str], Mapping[str, Any]] = {}
    prune: dict[str, set[tuple[str, str]]] = defaultdict(set)
    receipts: list[dict[str, Any]] = []

    def receipt(
        source_id: str,
        relation_type: str,
        target_id: str,
        relation: Mapping[str, Any],
        reason: str,
    ) -> None:
        key = (relation_type, target_id)
        if key in prune[source_id]:
            return
        prune[source_id].add(key)
        note = str(relation.get("note") or "")
        receipts.append({
            "source_id": source_id,
            "target_id": target_id,
            "type": relation_type,
            "reason": reason,
            "legacy_note_sha256": hashlib.sha256(
                note.encode("utf-8")
            ).hexdigest() if note else "",
        })

    for bucket in buckets:
        source_id = _bucket_id(bucket)
        for relation in _metadata(bucket).get("relations") or []:
            if not isinstance(relation, Mapping):
                continue
            relation_type = str(relation.get("type") or "").strip()
            target_id = str(relation.get("target") or "").strip()
            if relation_type not in RELATION_TYPES or not target_id:
                continue
            if target_id not in ids:
                receipt(
                    source_id,
                    relation_type,
                    target_id,
                    relation,
                    "missing_target",
                )
            if relation_type == "kin":
                kin_edges[(source_id, target_id)] = relation

    for (source_id, target_id), relation in sorted(kin_edges.items()):
        if source_id >= target_id:
            continue
        reverse = kin_edges.get((target_id, source_id))
        if reverse is None:
            continue
        # ``kin`` is undirected by contract and stored once.  Keep the
        # canonical low-id -> high-id record; the pre-apply snapshot and both
        # note hashes remain in the execution ledger.
        receipt(
            target_id,
            "kin",
            source_id,
            reverse,
            "reverse_kin_duplicate",
        )

    by_reason = Counter(item["reason"] for item in receipts)
    return {
        source_id: sorted(keys)
        for source_id, keys in sorted(prune.items())
    }, {
        "prune_edge_count": len(receipts),
        "prune_source_count": len(prune),
        "by_reason": dict(sorted(by_reason.items())),
        "receipts": sorted(
            receipts,
            key=lambda item: (
                item["reason"],
                item["source_id"],
                item["type"],
                item["target_id"],
            ),
        ),
    }


def _verified_evidence(
    plan: PlannedRelation,
    *,
    stored_source_id: str,
    stored_target_id: str,
) -> dict[str, Any]:
    evidence = dict(plan.evidence)
    evidence.update({
        "verification": "deterministic-backfill:v1",
        "stored_source_id": stored_source_id,
        "stored_target_id": stored_target_id,
    })
    return normalize_relation_evidence(evidence)


def build_relation_patches(
    buckets: Sequence[Mapping[str, Any]],
    plan: RelationGraphPlan,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    plan_by_key = {_plan_key(item): item for item in plan.relations}
    existing_by_key: dict[
        tuple[str, str, str],
        tuple[str, str, Mapping[str, Any]],
    ] = {}
    patches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    existing_count = 0

    for bucket in buckets:
        source_id = _bucket_id(bucket)
        relations = _metadata(bucket).get("relations") or []
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            relation_type = str(relation.get("type") or "").strip()
            target_id = str(relation.get("target") or "").strip()
            if relation_type not in RELATION_TYPES or not target_id or source_id == target_id:
                continue
            key = _existing_key(source_id, relation_type, target_id)
            existing_by_key.setdefault(key, (source_id, target_id, relation))
            matching_plan = plan_by_key.get(key)
            if relation.get("evidence"):
                evidence = relation["evidence"]
            elif matching_plan is not None:
                evidence = _verified_evidence(
                    matching_plan,
                    stored_source_id=source_id,
                    stored_target_id=target_id,
                )
            else:
                evidence = legacy_relation_evidence(relation)
            method = str(relation.get("generation_method") or "").strip()
            document: dict[str, Any] = {
                "type": relation_type,
                "target": target_id,
                "generation_method": method or LEGACY_GENERATION_METHOD,
                "evidence": evidence,
            }
            note = str(relation.get("note") or "").strip()
            if note:
                document["note"] = note
            elif matching_plan is not None:
                document["note"] = matching_plan.note
            if relation.get("strength") is not None:
                document["strength"] = relation["strength"]
            elif matching_plan is not None:
                document["strength"] = matching_plan.strength
            patches[source_id].append(document)
            existing_count += 1

    new_by_type: Counter[str] = Counter()
    new_by_method: Counter[str] = Counter()
    already_by_type: Counter[str] = Counter()
    for item in plan.relations:
        key = _plan_key(item)
        if key in existing_by_key:
            already_by_type[item.relation_type] += 1
            continue
        patches[item.source_id].append(item.edge_document())
        new_by_type[item.relation_type] += 1
        new_by_method[item.generation_method] += 1

    return dict(patches), {
        "existing_edges_scheduled_for_metadata_audit": existing_count,
        "planned_edge_count": len(plan.relations),
        "planned_already_present_count": sum(already_by_type.values()),
        "planned_already_present_by_type": dict(sorted(already_by_type.items())),
        "planned_new_count": sum(new_by_type.values()),
        "planned_new_by_type": dict(sorted(new_by_type.items())),
        "planned_new_by_method": dict(sorted(new_by_method.items())),
        "patch_source_count": len(patches),
    }


def _content_receipt(bucket: Mapping[str, Any]) -> dict[str, Any]:
    content = str(bucket.get("content") or "")
    compact = " ".join(content.split())
    metadata = _metadata(bucket)
    return {
        "bucket_id": _bucket_id(bucket),
        "path": str(bucket.get("path") or ""),
        "name": str(metadata.get("name") or ""),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "excerpt": compact[:240],
    }


def quality_samples(
    buckets: Sequence[Mapping[str, Any]],
    plan: RelationGraphPlan,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_id = {_bucket_id(bucket): bucket for bucket in buckets if _bucket_id(bucket)}
    ordered = sorted(
        plan.relations,
        key=lambda item: (
            item.generation_method,
            hashlib.sha256(
                f"{item.source_id}\x1f{item.relation_type}\x1f{item.target_id}".encode()
            ).hexdigest(),
        ),
    )
    if limit <= 0 or not ordered:
        return []
    methods = sorted({item.generation_method for item in ordered})
    per_method = max(1, limit // max(1, len(methods)))
    selected: list[PlannedRelation] = []
    for method in methods:
        selected.extend([
            item for item in ordered if item.generation_method == method
        ][:per_method])
    if len(selected) < limit:
        selected_keys = {_plan_key(item) for item in selected}
        selected.extend(
            item for item in ordered if _plan_key(item) not in selected_keys
        )
    selected = selected[:limit]
    return [
        {
            "relation": {
                "source_id": item.source_id,
                "target_id": item.target_id,
                "type": item.relation_type,
                "strength": item.strength,
                "generation_method": item.generation_method,
                "evidence": item.evidence,
            },
            "source": _content_receipt(by_id[item.source_id]),
            "target": _content_receipt(by_id[item.target_id]),
        }
        for item in selected
        if item.source_id in by_id and item.target_id in by_id
    ]


def _relations_present(
    buckets: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str, str]]:
    present: set[tuple[str, str, str]] = set()
    for bucket in buckets:
        source_id = _bucket_id(bucket)
        for relation in _metadata(bucket).get("relations") or []:
            if not isinstance(relation, Mapping):
                continue
            relation_type = str(relation.get("type") or "").strip()
            target_id = str(relation.get("target") or "").strip()
            if relation_type in RELATION_TYPES and target_id:
                present.add(_existing_key(source_id, relation_type, target_id))
    return present


def _relations_present_exact(
    buckets: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str, str]]:
    present: set[tuple[str, str, str]] = set()
    for bucket in buckets:
        source_id = _bucket_id(bucket)
        for relation in _metadata(bucket).get("relations") or []:
            if not isinstance(relation, Mapping):
                continue
            relation_type = str(relation.get("type") or "").strip()
            target_id = str(relation.get("target") or "").strip()
            if relation_type in RELATION_TYPES and target_id:
                present.add((source_id, relation_type, target_id))
    return present


async def _collect(manager: BucketManager) -> tuple[
    list[dict[str, Any]],
    RelationGraphPlan,
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    dict[str, list[tuple[str, str]]],
    dict[str, Any],
]:
    buckets = await manager.list_all(include_archive=True, include_nsfw=True)
    plan = plan_relation_graph(buckets)
    audit = audit_existing_relations(buckets)
    patches, patch_report = build_relation_patches(buckets, plan)
    cleanup, cleanup_report = build_cleanup_plan(buckets)
    return buckets, plan, audit, patches, patch_report, cleanup, cleanup_report


async def _apply_patches(
    manager: BucketManager,
    patches: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for source_id, edges in sorted(patches.items()):
        result = await manager.upsert_relations(source_id, list(edges), actor=ACTOR)
        for field in ("requested", "created", "enriched", "unchanged", "failed"):
            totals[field] += int(result.get(field, 0))
        if result.get("failed"):
            failures.append({
                "source_id": source_id,
                "errors": list(result.get("errors") or []),
            })
    return {
        **dict(totals),
        "failed_source_count": len(failures),
        "failures": failures,
    }


async def _apply_cleanup(
    manager: BucketManager,
    cleanup: Mapping[str, list[tuple[str, str]]],
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for source_id, edge_keys in sorted(cleanup.items()):
        result = await manager.prune_relations(
            source_id,
            list(edge_keys),
            actor=ACTOR,
        )
        for field in ("requested", "removed", "missing", "failed"):
            totals[field] += int(result.get(field, 0))
        if result.get("failed"):
            failures.append({
                "source_id": source_id,
                "errors": list(result.get("errors") or []),
            })
    return {
        **dict(totals),
        "failed_source_count": len(failures),
        "failures": failures,
    }


async def run_backfill(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    if args.buckets_dir is not None:
        config["buckets_dir"] = str(args.buckets_dir)
    if not args.apply:
        # Read-only dry-runs must not initialize or append the mutation ledger.
        config["audit"] = {"enabled": False}
    manager = BucketManager(config)
    snapshot = None
    apply_report = None

    if args.apply:
        snapshot_manager = SnapshotManager(config["buckets_dir"], args.snapshot_root)
        async with snapshot_manager.maintenance_barrier.exclusive_async():
            created = snapshot_manager.create_snapshot(args.snapshot_id)
            snapshot = {
                "snapshot_id": created.snapshot_id,
                "snapshot_path": str(created.snapshot_path),
                "manifest_sha256": created.manifest_sha256,
                "file_count": created.file_count,
                "total_bytes": created.total_bytes,
            }
            (
                buckets,
                plan,
                before_audit,
                patches,
                patch_report,
                cleanup,
                cleanup_report,
            ) = await _collect(manager)
            before_present = _relations_present(buckets)
            before_exact = _relations_present_exact(buckets)
            apply_report = await _apply_patches(manager, patches)
            apply_report["cleanup"] = await _apply_cleanup(manager, cleanup)
            after = await manager.list_all(include_archive=True, include_nsfw=True)
    else:
        (
            buckets,
            plan,
            before_audit,
            patches,
            patch_report,
            cleanup,
            cleanup_report,
        ) = await _collect(manager)
        before_present = _relations_present(buckets)
        before_exact = _relations_present_exact(buckets)
        after = buckets

    after_audit = audit_existing_relations(after)
    after_present = _relations_present(after)
    after_exact = _relations_present_exact(after)
    expected_plan = {_plan_key(item) for item in plan.relations}
    cleanup_exact = {
        (source_id, relation_type, target_id)
        for source_id, edge_keys in cleanup.items()
        for relation_type, target_id in edge_keys
    }
    expected_preserved = before_exact - cleanup_exact
    quality_clean = all(
        after_audit[field] == 0
        for field in (
            "malformed_count",
            "missing_target_count",
            "self_loop_count",
            "exact_duplicate_count",
            "reverse_kin_duplicate_count",
        )
    )
    verification = {
        "preexisting_valid_edges_preserved": expected_preserved.issubset(after_exact),
        "preexisting_valid_edges_missing_count": len(expected_preserved - after_exact),
        "planned_edges_present_count": len(expected_plan & after_present),
        "planned_edges_missing_count": len(expected_plan - after_present),
        "all_valid_edges_have_evidence": after_audit["missing_evidence_count"] == 0,
        "all_valid_edges_have_generation_method": (
            after_audit["missing_generation_method_count"] == 0
        ),
        "relation_quality_clean": quality_clean,
    }

    return {
        "schema": LEDGER_SCHEMA,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "dry-run",
        "authority": {
            "kind": "markdown-frontmatter",
            "buckets_dir": str(Path(config["buckets_dir"]).resolve()),
            "second_relation_store_created": False,
        },
        "external_model_calls": 0,
        "snapshot": snapshot,
        "input": {
            "bucket_count": plan.input_count,
            "eligible_bucket_count": plan.eligible_count,
            "unsupported_bucket_count": plan.unsupported_count,
            "skipped_by_reason": plan.skipped_by_reason,
        },
        "before": before_audit,
        "plan": {
            "relation_count": len(plan.relations),
            "by_type": plan.relation_type_counts,
            "by_generation_method": plan.generation_method_counts,
            **patch_report,
        },
        "cleanup_plan": cleanup_report,
        "projected_after": {
            "relation_count": (
                before_audit["relation_count"]
                + patch_report["planned_new_count"]
                - cleanup_report["prune_edge_count"]
            ),
            "missing_evidence_count": 0,
            "missing_generation_method_count": 0,
            "missing_target_count": 0,
            "reverse_kin_duplicate_count": 0,
        },
        "apply": apply_report,
        "after": after_audit,
        "verification": verification,
        "quality_samples": quality_samples(
            buckets,
            plan,
            limit=max(0, int(args.audit_samples)),
        ),
        "remaining_scope": {
            "z_axis_candidates_decided_by_this_run": 0,
            "z_axis_complete": False,
        },
    }


def _write_private_json(path: Path, document: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--buckets-dir", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--audit-samples", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.apply and (args.snapshot_root is None or not args.snapshot_id):
        parser.error("--apply requires --snapshot-root and --snapshot-id")
    if args.apply and args.output is None:
        parser.error("--apply requires --output for the durable execution ledger")

    document = asyncio.run(run_backfill(args))
    if args.output is not None:
        _write_private_json(args.output, document)
        print(json.dumps({
            "schema": document["schema"],
            "mode": document["mode"],
            "output": str(args.output.resolve()),
            "input": document["input"],
            "plan": document["plan"],
            "verification": document["verification"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
