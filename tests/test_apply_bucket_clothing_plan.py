from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import apply_bucket_clothing_plan as apply_clothing


def _write_bucket(
    vault: Path,
    bucket_id: str,
    body: str,
    *,
    name: str | None = None,
) -> Path:
    path = vault / "feel" / "沉淀物" / f"{bucket_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "---\n"
            f"id: {bucket_id}\n"
            f"name: {name or bucket_id}\n"
            "type: feel\n"
            "domain:\n- 未分类\n"
            "---\n"
            f"{body}"
        ),
        encoding="utf-8",
    )
    return path


def _plan_item(
    vault: Path,
    path: Path,
    bucket_id: str,
    body: str,
    keys: list[str],
) -> dict:
    return {
        "bucket_id": bucket_id,
        "path": str(path.relative_to(vault)),
        "status": "propose",
        "current_name": bucket_id,
        "name_action": "replace",
        "suggested_name": f"姐姐_郴州_2026-05-03_{bucket_id}",
        "name_basis": ["姐姐在郴州"],
        "retrieval_keys": [
            {
                "key": key,
                "evidence": f"证据：{key}",
                "sources": ["test"],
                "score": 9.0,
            }
            for key in keys
        ],
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }


def _plan(items: list[dict]) -> dict:
    return {
        "schema": apply_clothing.PLAN_SCHEMA,
        "mode": "dry_run_only",
        "source": {"unchanged": True},
        "items": items,
    }


def test_mechanical_filter_matches_review_condition():
    assert apply_clothing._key_rejection_reason("state.json") == "file_suffix"
    assert apply_clothing._key_rejection_reason("server.py") == "file_suffix"
    assert apply_clothing._key_rejection_reason("AGENTS.md") == "file_suffix"
    assert apply_clothing._key_rejection_reason("v0.3") == "pure_version"
    assert (
        apply_clothing._key_rejection_reason("turn_lock")
        == "pure_english_tech_identifier"
    )
    assert (
        apply_clothing._key_rejection_reason("claude-twin")
        == "pure_english_tech_identifier"
    )
    assert apply_clothing._key_rejection_reason("郴州") == ""
    assert apply_clothing._key_rejection_reason("RRF降噪门槛") == ""


def test_apply_backs_up_exact_file_and_only_changes_name_plus_tail(tmp_path):
    vault = tmp_path / "vault"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    body = (
        "姐姐在郴州吃铁锅炖，也带回一张照片。"
        "技术记录里还有 server.py、turn_lock 和 v0.3。"
    )
    target = _write_bucket(vault, "one", body)
    original = target.read_text(encoding="utf-8")
    plan = _plan([
        _plan_item(
            vault,
            target,
            "one",
            body,
            ["姐姐", "郴州", "照片", "server.py", "turn_lock", "v0.3"],
        )
    ])

    operations, skipped, summary = apply_clothing.build_operations(vault, plan)
    manifest = apply_clothing.apply_operations(
        vault,
        snapshots / "run",
        "approved-plan",
        operations,
        skipped,
        summary,
    )

    updated = target.read_text(encoding="utf-8")
    assert 'name: "姐姐_郴州_2026-05-03_one"\n' in updated
    assert updated.endswith("[检索钥匙: 姐姐/郴州/照片]\n")
    assert body in updated
    assert manifest["status"] == "applied"
    backup = snapshots / "run" / "files" / target.relative_to(vault)
    assert backup.read_text(encoding="utf-8") == original
    assert summary == {
        "approved_proposals": 1,
        "operations": 1,
        "skipped_after_filter": 0,
        "name_changes": 1,
        "key_lines_to_append": 1,
        "keys_kept": 3,
        "keys_filtered": 3,
        "keys_filtered_in_operations": 3,
        "keys_filtered_with_skipped": 0,
        "filtered_reasons": {
            "file_suffix": 1,
            "pure_english_tech_identifier": 1,
            "pure_version": 1,
        },
    }


def test_all_filtered_item_is_skipped_without_backup_or_mutation(tmp_path):
    vault = tmp_path / "vault"
    body = "姐姐在郴州，使用 state.json 和 turn_lock 版本 v0.3。"
    target = _write_bucket(vault, "one", body)
    original = target.read_text(encoding="utf-8")
    plan = _plan([
        _plan_item(
            vault,
            target,
            "one",
            body,
            ["state.json", "turn_lock", "v0.3"],
        )
    ])

    operations, skipped, summary = apply_clothing.build_operations(vault, plan)

    assert operations == []
    assert skipped[0]["reason"] == "all_keys_filtered"
    assert summary["operations"] == 0
    assert summary["keys_filtered"] == 3
    assert summary["keys_filtered_in_operations"] == 0
    assert summary["keys_filtered_with_skipped"] == 3
    assert target.read_text(encoding="utf-8") == original


def test_body_drift_fails_before_any_write(tmp_path):
    vault = tmp_path / "vault"
    body = "姐姐在郴州吃铁锅炖。"
    target = _write_bucket(vault, "one", body)
    plan = _plan([
        _plan_item(vault, target, "one", body, ["姐姐", "郴州", "铁锅炖"])
    ])
    target.write_text(
        target.read_text(encoding="utf-8") + "后来发生了变化。",
        encoding="utf-8",
    )
    before = target.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="body drifted"):
        apply_clothing.build_operations(vault, plan)

    assert target.read_text(encoding="utf-8") == before


def test_approved_plan_hash_mismatch_fails_closed(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan([])), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        apply_clothing._load_approved_plan(plan_path, "0" * 64, 0)
