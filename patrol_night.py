#!/usr/bin/env python3
"""Durable nightly wrapper for the strictly read-only M-axis patrol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import patrol as patrol_module
from review_queue import ReviewQueue
from storage_safety import advisory_file_lock, atomic_write_text


DEFAULT_CONFIG = "/app/config.yaml"
DEFAULT_STATE_DIR = "/data/.lmc5/patrol"


def _append_history(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with advisory_file_lock(path.with_suffix(path.suffix + ".lock")):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(path, 0o600)


def _write_status(state_dir: Path, record: dict) -> None:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    atomic_write_text(state_dir / "latest.json", payload)
    _append_history(state_dir / "history.jsonl", record)


def run_nightly_patrol(
    config_path: str | os.PathLike,
    state_dir: str | os.PathLike,
    *,
    clock: Callable[[], datetime] | None = None,
) -> dict:
    """Run patrol, persist both success/failure evidence, and re-raise failure."""
    now = (clock or datetime.now)()
    run_id = now.strftime("patrol-%Y%m%dT%H%M%S")
    state_root = Path(state_dir)
    state_root.mkdir(parents=True, exist_ok=True)
    runs_dir = state_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "schema": "ombre.m-patrol-night/v1",
        "run_id": run_id,
        "started_at": now.isoformat(timespec="seconds"),
        "ok": False,
    }
    try:
        cfg = patrol_module._load_patrol_config(config_path)
        buckets_dir = Path(
            cfg.get("buckets_dir")
            or os.environ.get("OMBRE_BUCKETS_DIR")
            or "/data"
        )
        if not buckets_dir.is_dir():
            raise ValueError(f"bucket directory does not exist: {buckets_dir}")
        registry = ((cfg.get("fact_slots", {}) or {}).get("registry", {}) or {})
        report = patrol_module.patrol(
            buckets_dir,
            now,
            fact_slot_registry=registry,
        )
        rendered = patrol_module.render_md(report, buckets_dir, now)
        queue = ReviewQueue(
            buckets_dir / "review_queue.jsonl",
            maintenance_root=buckets_dir,
        )
        queued = patrol_module.enqueue_metabolism_suggestions(report, queue)
        # Z 轴：同槽新旧候选只入待审队列；fact_status 只有人批准后才会变。
        # 默认关：2026-08-18 真库 dry-run 三轮候选抽样误报率过高，报告里看得到、
        # 但不灌进 review_pending；config.fact_slots.auto_enqueue_z_candidates: true 才入队。
        auto_z = bool((cfg.get("fact_slots", {}) or {}).get("auto_enqueue_z_candidates", False))
        queued_z = patrol_module.enqueue_z_pair_candidates(report, queue) if auto_z else 0
        report_path = runs_dir / f"{run_id}.md"
        atomic_write_text(report_path, rendered + "\n")
        atomic_write_text(state_root / "latest.md", rendered + "\n")
        status.update({
            "ok": True,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "report": os.fspath(report_path),
            "bucket_count": int(report.get("total", 0)),
            "suggestion_count": len(report.get("suggestions", [])),
            "queued_count": queued,
            "z_candidate_count": len(report.get("z_pair_candidates", [])),
            "z_queued_count": queued_z,
        })
        _write_status(state_root, status)
        return status
    except BaseException as exc:
        status.update({
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "error_type": type(exc).__name__,
        })
        _write_status(state_root, status)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run M-axis patrol with durable success/failure evidence"
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("OMBRE_CONFIG", DEFAULT_CONFIG),
    )
    parser.add_argument(
        "--state-dir",
        default=os.environ.get("OMBRE_PATROL_STATE_DIR", DEFAULT_STATE_DIR),
    )
    args = parser.parse_args(argv)
    try:
        result = run_nightly_patrol(args.config, args.state_dir)
    except BaseException as exc:
        print(f"M patrol failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
