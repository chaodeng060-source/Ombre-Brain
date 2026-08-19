from __future__ import annotations

import json
from argparse import Namespace

import pytest

import timeline_sweep_cli


def _manifest(path, assignments):
    path.write_text(
        json.dumps({
            "schema": timeline_sweep_cli.REVIEW_SCHEMA,
            "reviewer": "claude",
            "reviewed_at": "2026-08-19T00:00:00+08:00",
            "assignments": assignments,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_review_manifest_rejects_machine_lines_and_conflicts(tmp_path):
    machine = _manifest(
        tmp_path / "machine.json",
        [{"bucket_id": "a", "thread": "event:1234"}],
    )
    with pytest.raises(ValueError, match="narrative thread"):
        timeline_sweep_cli._load_reviewed_manifest(machine)

    conflict = _manifest(
        tmp_path / "conflict.json",
        [
            {"bucket_id": "a", "thread": "工程演进"},
            {"bucket_id": "a", "thread": "记忆治理"},
        ],
    )
    with pytest.raises(ValueError, match="conflicting"):
        timeline_sweep_cli._load_reviewed_manifest(conflict)


@pytest.mark.asyncio
async def test_dry_run_and_snapshot_apply_share_one_reviewed_plan(
    tmp_path,
    monkeypatch,
    test_config,
    bucket_mgr,
):
    bucket_id = await bucket_mgr.create(content="工程第一阶段", name="stage-one")
    manifest = _manifest(
        tmp_path / "reviewed.json",
        [{"bucket_id": bucket_id, "thread": "工程演进"}],
    )
    monkeypatch.setattr(timeline_sweep_cli, "load_config", lambda: test_config)

    dry = await timeline_sweep_cli._run(Namespace(
        apply=False,
        reviewed_manifest=manifest,
        snapshot_root=None,
        snapshot_id="",
        audit_lines=10,
    ))
    applied = await timeline_sweep_cli._run(Namespace(
        apply=True,
        reviewed_manifest=manifest,
        snapshot_root=tmp_path / "snapshots",
        snapshot_id="timeline-before-test",
        audit_lines=10,
    ))

    assert dry["report"]["assigned_count"] == 1
    assert dry["report"]["new_line_count"] == 1
    assert dry["report"]["orphan_count"] == 0
    assert dry["report"]["updated_count"] == 0
    assert applied["report"]["assigned_count"] == 1
    assert applied["report"]["new_line_count"] == 1
    assert applied["report"]["orphan_count"] == 0
    assert applied["report"]["updated_count"] == 1
    assert applied["snapshot"]["manifest_sha256"]
    assert (await bucket_mgr.get(bucket_id))["metadata"]["thread"] == "工程演进"
