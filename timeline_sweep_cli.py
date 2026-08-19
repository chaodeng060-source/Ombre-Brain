#!/usr/bin/env python3
"""Dry-run or apply the reviewed Ombre X-axis backfill."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from bucket_manager import BucketManager, bucket_revision_hash
from snapshot_manager import SnapshotManager
from timeline_axis import (
    OTHER_THREAD,
    load_thread_hints_from_ledger,
    normalize_thread_hint,
    plan_timeline_assignments,
    run_timeline_sweep,
)
from utils import event_at_from_metadata, load_config


REVIEW_SCHEMA = "ombre.timeline-review/v1"


def _load_reviewed_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"review manifest schema must be {REVIEW_SCHEMA}")
    if not str(document.get("reviewer") or "").strip():
        raise ValueError("review manifest requires reviewer")
    if not str(document.get("reviewed_at") or "").strip():
        raise ValueError("review manifest requires reviewed_at")
    rows = document.get("assignments")
    if not isinstance(rows, list):
        raise ValueError("review manifest assignments must be a list")

    reviewed: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("review assignment must be an object")
        bucket_id = str(row.get("bucket_id") or "").strip()
        thread = normalize_thread_hint(row.get("thread"))
        if not bucket_id or thread == OTHER_THREAD:
            raise ValueError("review assignment requires bucket_id and narrative thread")
        previous = reviewed.get(bucket_id)
        if previous is not None and previous != thread:
            raise ValueError(f"conflicting review assignment: {bucket_id}")
        reviewed[bucket_id] = thread
    return reviewed


def _line_samples(
    buckets: list[Mapping[str, Any]],
    *,
    limit: int,
    reasons: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_thread: dict[str, list[Mapping[str, Any]]] = {}
    for bucket in buckets:
        metadata = bucket.get("metadata", {}) or {}
        thread = str(metadata.get("thread") or OTHER_THREAD).strip() or OTHER_THREAD
        if thread != OTHER_THREAD:
            by_thread.setdefault(thread, []).append(bucket)

    samples: list[dict[str, Any]] = []
    for thread, members in sorted(
        by_thread.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[: max(0, limit)]:
        ordered = sorted(
            members,
            key=lambda bucket: (
                str(event_at_from_metadata(bucket.get("metadata", {}) or {}) or ""),
                str(bucket.get("id") or ""),
            ),
        )
        samples.append({
            "thread": thread,
            "size": len(ordered),
            "members": [
                {
                    "bucket_id": str(bucket.get("id") or ""),
                    "event_at": event_at_from_metadata(
                        bucket.get("metadata", {}) or {}
                    ),
                    "name": str(
                        (bucket.get("metadata", {}) or {}).get("name") or ""
                    ),
                    "reason": reasons.get(str(bucket.get("id") or ""), ""),
                }
                for bucket in ordered
            ],
        })
    return samples


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    manager = BucketManager(config)
    reviewed = _load_reviewed_manifest(args.reviewed_manifest)
    ledger_path = Path(config["buckets_dir"]) / ".lmc5" / "pipeline.sqlite3"
    hints = load_thread_hints_from_ledger(ledger_path)
    snapshot = None

    async def execute() -> tuple[list[dict], Any, Any]:
        before = await manager.list_all(include_archive=False)
        plan = plan_timeline_assignments(
            before,
            thread_hints_by_bucket=hints,
            reviewed_threads_by_bucket=reviewed,
        )
        report = await run_timeline_sweep(
            manager,
            ledger_path=ledger_path,
            reviewed_threads_by_bucket=reviewed,
            apply=args.apply,
            actor="operator:timeline-sweep",
            revision_hash_provider=bucket_revision_hash,
        )
        return before, plan, report

    if args.apply:
        snapshot_manager = SnapshotManager(
            config["buckets_dir"],
            args.snapshot_root,
        )
        async with snapshot_manager.maintenance_barrier.exclusive_async():
            created = snapshot_manager.create_snapshot(args.snapshot_id)
            snapshot = {
                "snapshot_id": created.snapshot_id,
                "snapshot_path": str(created.snapshot_path),
                "manifest_sha256": created.manifest_sha256,
                "file_count": created.file_count,
                "total_bytes": created.total_bytes,
            }
            before, plan, report = await execute()
        sample_buckets = await manager.list_all(include_archive=False)
    else:
        before, plan, report = await execute()
        assignments = {item.bucket_id: item.thread for item in plan.assignments}
        sample_buckets = []
        for bucket in before:
            virtual = dict(bucket)
            virtual["metadata"] = dict(bucket.get("metadata", {}) or {})
            virtual["metadata"]["thread"] = assignments.get(
                str(bucket.get("id") or ""),
                OTHER_THREAD,
            )
            sample_buckets.append(virtual)

    report_document = asdict(report)
    line_sizes = report_document.pop("line_sizes")
    report_document["line_count"] = len(line_sizes)
    report_document["largest_line_size"] = max(line_sizes.values(), default=0)
    reasons = {item.bucket_id: item.reason for item in plan.assignments}
    return {
        "schema": "ombre.timeline-sweep/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "dry-run",
        "snapshot": snapshot,
        "review_manifest": str(args.reviewed_manifest or ""),
        "report": report_document,
        "line_samples": _line_samples(
            sample_buckets,
            limit=args.audit_lines,
            reasons=reasons,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--reviewed-manifest", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--audit-lines", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.apply and (args.snapshot_root is None or not args.snapshot_id):
        parser.error("--apply requires --snapshot-root and --snapshot-id")

    document = asyncio.run(_run(args))
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
