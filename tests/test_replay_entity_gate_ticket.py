from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from tools import replay_entity_gate_ticket as replay_tool


PRIVATE_ROOT = Path("/opt/claude-twin/.work")
LEDGER_PATH = PRIVATE_ROOT / "recall_noise_ledger_20260905.json"
AUDIT_PATH = PRIVATE_ROOT / "entity_score_guard_audit_20260905.json"


def _real_inputs() -> tuple[dict, dict]:
    if not LEDGER_PATH.is_file() or not AUDIT_PATH.is_file():
        pytest.skip("private 2026-09-05 entity evidence is unavailable")
    return replay_tool.load_inputs(LEDGER_PATH, AUDIT_PATH)


def _all_mapping_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_mapping_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_mapping_keys(item)


def test_real_frozen_replay_covers_all_22_requests_and_ticket_distribution():
    ledger, audit = _real_inputs()

    result = replay_tool.replay(ledger, audit)

    assert result["request_count"] == 22
    assert result["anchor_position_count"] == 318
    assert result["entity_position_count"] == 15
    assert result["entity_unique_bucket_count"] == 10
    assert result["historical_pre_ds_position_count"] == 89
    assert result["historical_pre_ds_unique_count"] == 88
    assert result["historical_final_position_count"] == 52
    assert result["historical_ds_outcome_counts"] == {
        "ok": 18,
        "error": 2,
        "missing": 2,
    }
    assert result["query_provenance"] == {
        "original_query_sha_verified_count": 21,
        "production_effective_query_sha_verified_count": 1,
        "fabricated_query_count": 0,
        "query_text_emitted": False,
    }
    assert result["low_frequency_entity_position_count"] == 5
    assert result["high_frequency_entity_position_count"] == 10
    assert result["ticket_eligible_position_count"] == 4
    assert result["ticket_in_position_count"] == 2
    assert result["ticket_request_count"] == 1
    assert result["on_ds_evidence_status_counts"] == {
        "historical_same_input_evidence": 19,
        "unverified_changed_ordinary_input": 2,
        "unverified_ticket_input": 1,
    }
    assert len(result["requests"]) == 22
    assert all("off" in row and "on" in row for row in result["requests"])
    assert all("ticket_in" in row["on"] for row in result["requests"])
    assert all("ds_result" in row["on"] for row in result["requests"])
    assert all("final_change" in row["on"] for row in result["requests"])


def test_c001_noise_is_not_ticketed_and_6cc_is_projected_to_same_gate():
    ledger, audit = _real_inputs()

    result = replay_tool.replay(ledger, audit)

    noise = result["checks"]["engineering_noise"]
    assert noise == {
        "request_id": replay_tool.NOISE_REQUEST_ID,
        "bucket_ids": list(replay_tool.ENGINEERING_NOISE_IDS),
        "ticket_in_count": 0,
        "not_ticketed": True,
        "absent_from_bounded_on_primary_final": True,
        "proof_status": "bounded_primary_projection",
    }
    noise_request = next(
        row
        for row in result["requests"]
        if row["request_id"] == replay_tool.NOISE_REQUEST_ID
    )
    assert noise_request["on"]["ticket_in"] == []
    assert noise_request["on"]["ds_result"]["status"] == (
        "unverified_changed_ordinary_input"
    )
    assert set(replay_tool.ENGINEERING_NOISE_IDS).issubset(
        noise_request["on"]["final_change"]["definite_removed_ids"]
    )

    target = result["checks"]["named_ticket_target"]
    assert target["request_id"] == replay_tool.TARGET_REQUEST_ID
    assert target["bucket_id"] == replay_tool.TARGET_BUCKET_ID
    assert target["eligible"] is True
    assert target["ticket_in"] is True
    assert target["projected_to_same_ds_call"] is True
    assert target["historical_pre_ds_candidate"] is True
    assert target["historical_final_survivor"] is True
    assert target["new_ticket_ds_verdict"] == "unverified"

    target_request = next(
        row
        for row in result["requests"]
        if row["request_id"] == replay_tool.TARGET_REQUEST_ID
    )
    assert target_request["on"]["ticket_in"] == [
        replay_tool.TARGET_BUCKET_ID,
        "6e997109bad7",
    ]
    assert target_request["on"]["ds_result"]["status"] == (
        "unverified_ticket_input"
    )


def test_missing_on_ds_receipt_and_human_gold_are_machine_visible_failures():
    ledger, audit = _real_inputs()

    result = replay_tool.replay(ledger, audit)

    assert result["provider_call_count"] == 0
    assert result["new_on_ds_call_count"] == 0
    assert result["new_on_ds_verified_request_count"] == 0
    assert result["verified_human_gold_labelled_entity_position_count"] == 0
    assert result["verified_human_gold_unlabelled_entity_position_count"] == 15
    assert result["batch_relevant_false_kill_count"] is None
    assert result["named_ticket_target_false_kill_count"] is None
    assert result["acceptance"] is False
    assert set(result["acceptance_reasons"]) == {
        "new_ticket_ds_receipt_missing:1",
        "changed_ordinary_ds_input_receipt_missing:2",
        "end_to_end_on_final_receipt_missing:22",
        "verified_human_gold_incomplete:15",
        "batch_relevant_false_kill_unverified",
    }
    assert result["minimum_supplement"]["fabricate_queries_or_gold"] is False
    assert result["minimum_supplement"]["same_turn_on_receipts_required"] == 22


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o444])
def test_private_reader_rejects_group_or_world_readable_input(tmp_path, mode):
    path = tmp_path / "private.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(mode)

    with pytest.raises(PermissionError, match="group/world-readable"):
        replay_tool._read_private_json_object(path)


def test_private_reader_accepts_mode_0600(tmp_path):
    path = tmp_path / "private.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o600)

    assert replay_tool._read_private_json_object(path) == {}


def test_private_reader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"label":"noise","label":"relevant"}', encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="duplicate JSON key: label"):
        replay_tool._read_private_json_object(path)


def test_frozen_ledger_digest_mutation_fails_closed(tmp_path):
    ledger, _audit = _real_inputs()
    mutated = copy.deepcopy(ledger)
    mutated["rows"][0]["ids_out"] = []
    path = tmp_path / "mutated-ledger.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="frozen ledger SHA-256 mismatch"):
        replay_tool.load_inputs(path, AUDIT_PATH)


def test_frozen_entity_audit_digest_mutation_fails_closed(tmp_path):
    _ledger, audit = _real_inputs()
    mutated = copy.deepcopy(audit)
    mutated["generated_at"] = "changed"
    path = tmp_path / "mutated-audit.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="frozen entity audit SHA-256 mismatch"):
        replay_tool.load_inputs(LEDGER_PATH, path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("audit_missing_pair", "15 audit candidate rows"),
        ("audit_df_drift", "frozen term DF"),
        ("duplicate_request", "request IDs must be unique"),
        ("anchor_count", "318 anchor positions"),
    ],
)
def test_real_evidence_shape_mutations_fail_closed(mutation, message):
    ledger, audit = _real_inputs()
    mutated_ledger = copy.deepcopy(ledger)
    mutated_audit = copy.deepcopy(audit)

    if mutation == "audit_missing_pair":
        mutated_audit["candidate_rows"].pop()
    elif mutation == "audit_df_drift":
        mutated_audit["candidate_rows"][0]["entity_df"] += 1
    elif mutation == "duplicate_request":
        mutated_ledger["rows"][1]["ombre_request_id"] = mutated_ledger[
            "rows"
        ][0]["ombre_request_id"]
    else:
        mutated_ledger["rows"][0]["anchors"].pop()

    with pytest.raises(ValueError, match=message):
        replay_tool.replay(mutated_ledger, mutated_audit)


def test_result_never_emits_query_or_bucket_content():
    ledger, audit = _real_inputs()

    result = replay_tool.replay(ledger, audit)

    forbidden = {
        "query",
        "original_query",
        "effective_query",
        "content",
        "bucket_content",
    }
    assert forbidden.isdisjoint(_all_mapping_keys(result))
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
    for row in ledger["rows"]:
        for field in ("original_query", "effective_query"):
            value = row.get(field)
            if isinstance(value, str) and value:
                assert value not in payload
    for bucket in ledger["buckets"].values():
        content = bucket.get("content")
        if isinstance(content, str) and content:
            assert content not in payload
    for evidence in audit["candidate_rows"]:
        canonical_name = evidence["matched_entity"]["canonical_name"]
        assert canonical_name not in payload


def test_cli_returns_nonzero_for_machine_visible_unverified_acceptance(
    monkeypatch,
    capsys,
):
    if not LEDGER_PATH.is_file() or not AUDIT_PATH.is_file():
        pytest.skip("private 2026-09-05 entity evidence is unavailable")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_entity_gate_ticket.py",
            str(LEDGER_PATH),
            "--entity-audit",
            str(AUDIT_PATH),
        ],
    )

    assert replay_tool.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["acceptance"] is False
