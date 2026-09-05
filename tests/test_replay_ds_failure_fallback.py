import asyncio
import copy
import json
from pathlib import Path

import pytest

import server
from tools import replay_ds_failure_fallback as replay_tool


PRIVATE_ROOT = Path("/opt/claude-twin/.work")
LEDGER_PATH = PRIVATE_ROOT / "recall_noise_ledger_20260905.json"
SUPPLEMENTAL_PATHS = [
    PRIVATE_ROOT / "recall_noise_supplemental_buckets_20260905.json",
    PRIVATE_ROOT / "ds_replay_success_supplemental_20260905.json",
]


def _real_replay_inputs() -> tuple[dict, dict[str, dict]]:
    paths = [LEDGER_PATH, *SUPPLEMENTAL_PATHS]
    if not all(path.is_file() for path in paths):
        pytest.skip("private 2026-09-05 replay evidence is unavailable")
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    snapshots = replay_tool._merge_bucket_sources(
        ledger.get("buckets"),
        *(
            json.loads(path.read_text(encoding="utf-8")).get("buckets")
            for path in SUPPLEMENTAL_PATHS
        ),
    )
    return ledger, snapshots


def test_baseline_filter_is_extracted_from_fixed_commit():
    baseline, namespace, source_sha256 = replay_tool._load_baseline_filter()

    assert baseline.__name__ == "_baseline_ds_filter_candidates"
    assert namespace["_baseline_ds_filter_candidates"] is baseline
    assert len(source_sha256) == 64
    assert replay_tool.BASELINE_COMMIT == (
        "8be99749d0c24b2249e8502770017c0bf39ecc3d"
    )


def test_real_private_ledger_proves_baseline_patch_contract():
    ledger, snapshots = _real_replay_inputs()

    result = asyncio.run(replay_tool.replay(ledger["rows"], snapshots))

    assert result["request_count"] == 22
    assert result["projection_unique_count"] == 88
    assert result["projection_snapshot_complete_count"] == 88
    assert result["original_query_sha_verified_count"] == 21
    assert result["production_effective_query_count"] == 1
    assert result["ok_request_count"] == 18
    assert result["ok_structural_equivalence_count"] == 18
    assert result["failure_replayed_count"] == 4
    assert result["target_noise_admitted"] == 0
    assert result["target_noise_forced"] == 0
    assert all(
        row["exact_id_equivalence"] and row["partial_hash_equivalence"]
        for row in result["success_results"]
    )
    assert {
        row["request_id"]: row["injected_failure"]
        for row in result["failure_results"]
    } == replay_tool.EXPECTED_FAILURE_INJECTIONS


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_request", "unique"),
        ("query_sha", "query hash mismatch"),
        ("effective_query", "production effective query hash mismatch"),
        ("extra_effective", "production effective"),
        ("failure_status", "failure request set"),
    ],
)
def test_real_ledger_shape_and_provenance_mutations_fail_closed(mutation, match):
    ledger, _snapshots = _real_replay_inputs()
    rows = copy.deepcopy(ledger["rows"])

    if mutation == "duplicate_request":
        rows[1]["ombre_request_id"] = rows[0]["ombre_request_id"]
    elif mutation == "query_sha":
        row = next(row for row in rows if row.get("original_query"))
        row["sha"] = "0" * 12
    elif mutation == "effective_query":
        row = next(
            row
            for row in rows
            if row["ombre_request_id"]
            == replay_tool.PRODUCTION_EFFECTIVE_QUERY_REQUEST_ID
        )
        row["effective_query"] += " mutated"
    elif mutation == "extra_effective":
        row = next(
            row
            for row in rows
            if row["ombre_request_id"]
            != replay_tool.PRODUCTION_EFFECTIVE_QUERY_REQUEST_ID
        )
        row["original_query"] = None
        row["query_source"] = "production_ds_log"
    else:
        row = next(
            row
            for row in rows
            if row["ombre_request_id"] in replay_tool.EXPECTED_FAILURE_INJECTIONS
        )
        row["ds_gate_outcome"] = "ok"

    with pytest.raises(ValueError, match=match):
        replay_tool._validate_ledger_rows(rows)


@pytest.mark.parametrize("mutation", ["missing", "empty_content", "bad_metadata"])
def test_real_candidate_snapshot_mutations_fail_closed(mutation):
    ledger, snapshots = _real_replay_inputs()
    row = next(
        row
        for row in ledger["rows"]
        if row["ombre_request_id"] == replay_tool.TARGET_REQUEST_ID
    )
    mutated = copy.deepcopy(snapshots)
    bucket_id = str(row["ids_in"][0])

    if mutation == "missing":
        del mutated[bucket_id]
    elif mutation == "empty_content":
        mutated[bucket_id]["content"] = ""
    else:
        mutated[bucket_id]["metadata"] = None

    with pytest.raises(ValueError, match="bucket snapshot"):
        replay_tool._build_candidates(row, mutated)


def test_success_contract_detects_current_wrapper_mutation(monkeypatch):
    ledger, snapshots = _real_replay_inputs()
    row = next(
        row
        for row in ledger["rows"]
        if row.get("ds_gate_outcome") == "ok" and len(row.get("ids_in") or []) > 1
    )
    query, _source, _verified = replay_tool._query_provenance(row)
    candidates = replay_tool._build_candidates(row, snapshots)
    baseline, baseline_globals, _source_sha = replay_tool._load_baseline_filter()
    current = server._ds_filter_candidates

    async def reversed_current(*args, **kwargs):
        selected = await current(*args, **kwargs)
        return list(reversed(selected))

    monkeypatch.setattr(server, "_ds_filter_candidates", reversed_current)

    with pytest.raises(AssertionError, match="success path drift"):
        asyncio.run(
            replay_tool._prove_success_equivalence(
                query,
                row,
                candidates,
                baseline,
                baseline_globals,
            )
        )


def test_bucket_source_conflicts_fail_closed():
    with pytest.raises(ValueError, match="conflicting bucket snapshot"):
        replay_tool._merge_bucket_sources(
            {"bucket": {"id": "bucket", "content": "a", "metadata": {}}},
            {"bucket": {"id": "bucket", "content": "b", "metadata": {}}},
        )
