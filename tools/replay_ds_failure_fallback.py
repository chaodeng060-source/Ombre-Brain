"""Replay DS success/failure contracts against the private 2026-09-05 ledger.

This tool makes no provider calls and prints neither query text nor bucket
content.  For successful historical rows it extracts the original
``_ds_filter_candidates`` implementation from the fixed pre-change commit and
compares it with the current implementation across every ordered selector
subset.  Failure rows exercise the current wrapper with their recorded failure
class and render the conservative partial result from real bucket snapshots.

The result is structural regression evidence.  Historical provider decisions
and end-to-end production output are reported separately and are not recreated.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import server
from recall_timing import (
    begin_recall_timing,
    finish_recall_timing,
    reset_recall_timing,
)


BASELINE_COMMIT = "8be99749d0c24b2249e8502770017c0bf39ecc3d"
BASELINE_FUNCTION_NAME = "_ds_filter_candidates"
BASELINE_REPLAY_NAME = "_baseline_ds_filter_candidates"
DEFAULT_FLOOR = "0.450001"
EXPECTED_REQUEST_COUNT = 22
EXPECTED_OK_COUNT = 18
EXPECTED_PROJECTION_UNIQUE_COUNT = 88
PRODUCTION_EFFECTIVE_QUERY_REQUEST_ID = "7fa0ff67ca954ae58750978fb53699e0"
PRODUCTION_EFFECTIVE_QUERY_SHA256 = (
    "fd83fb69932c422929f8c578a7966386a1461a260bdfff02db113cd5c279808b"
)
TARGET_REQUEST_ID = "5e00777f6b2f4c3e823ec6064aa39d42"
TARGET_NOISE_IDS = {"013da98a75e5", "019af40158f7"}
EXPECTED_FAILURE_INJECTIONS = {
    "8d3debaf71d741b09767daaba0657a1f": "cancel",
    "7897a75aff194cfd9caaf266ad5f6b48": "cancel",
    "138ae29d39f1448ba510cd1951bfcc4b": "invalid",
    TARGET_REQUEST_ID: "invalid",
}

FilterFunction = Callable[..., Awaitable[list[dict]]]


def _safe_request_id(row: Mapping[str, Any]) -> str:
    return str(row.get("ombre_request_id") or "").strip()


def _query_provenance(row: Mapping[str, Any]) -> tuple[str, str, bool]:
    """Return captured query bytes after enforcing the frozen provenance rules."""
    request_id = _safe_request_id(row)
    original = row.get("original_query")
    effective = row.get("effective_query")

    if request_id == PRODUCTION_EFFECTIVE_QUERY_REQUEST_ID:
        if isinstance(original, str) and original:
            raise ValueError(
                "known production effective request unexpectedly has original "
                f"text: {request_id}"
            )
        if row.get("query_source") != "production_ds_log":
            raise ValueError(
                f"production effective query source drifted for {request_id}"
            )
        if not isinstance(effective, str) or not effective:
            raise ValueError(
                f"missing captured production effective query for {request_id}"
            )
        effective_digest = hashlib.sha256(effective.encode("utf-8")).hexdigest()
        if effective_digest != PRODUCTION_EFFECTIVE_QUERY_SHA256:
            raise ValueError(
                f"production effective query hash mismatch for {request_id}"
            )
        return effective, "production_effective_query", False

    if not isinstance(original, str) or not original:
        raise ValueError(
            "production effective query is allowed only for "
            f"{PRODUCTION_EFFECTIVE_QUERY_REQUEST_ID}; missing original for "
            f"{request_id}"
        )
    prefix = str(row.get("sha") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{12,64}", prefix):
        raise ValueError(f"invalid query hash prefix for {request_id}")
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    if not digest.startswith(prefix):
        raise ValueError(f"query hash mismatch for {request_id}")
    return original, "original_query", True


def _validate_ledger_rows(
    rows: list[dict],
) -> dict[str, tuple[str, str, bool]]:
    """Reject any drift from the frozen 22-request evidence shape."""
    if not isinstance(rows, list) or len(rows) != EXPECTED_REQUEST_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_REQUEST_COUNT} ledger rows"
        )
    request_ids = [_safe_request_id(row) for row in rows]
    if any(not request_id for request_id in request_ids):
        raise ValueError("every ledger row requires a request ID")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("ledger request IDs must be unique")

    ok_ids = {
        _safe_request_id(row)
        for row in rows
        if row.get("ds_gate_outcome") == "ok"
    }
    failure_ids = set(request_ids) - ok_ids
    if failure_ids != set(EXPECTED_FAILURE_INJECTIONS):
        raise ValueError(
            "failure request set drifted from the frozen four-request set"
        )
    if len(ok_ids) != EXPECTED_OK_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_OK_COUNT} successful DS rows"
        )
    for row in rows:
        request_id = _safe_request_id(row)
        injection = EXPECTED_FAILURE_INJECTIONS.get(request_id)
        outcome = row.get("ds_gate_outcome")
        if injection == "invalid" and outcome != "error":
            raise ValueError(f"invalid-row status drifted for {request_id}")
        if injection == "cancel" and outcome not in (None, ""):
            raise ValueError(f"cancel-row status drifted for {request_id}")

    provenance = {
        _safe_request_id(row): _query_provenance(row)
        for row in rows
    }
    original_verified = sum(
        source == "original_query" and verified
        for _query, source, verified in provenance.values()
    )
    effective_ids = {
        request_id
        for request_id, (_query, source, _verified) in provenance.items()
        if source == "production_effective_query"
    }
    if original_verified != 21:
        raise ValueError("expected 21 SHA-verified original queries")
    if effective_ids != {PRODUCTION_EFFECTIVE_QUERY_REQUEST_ID}:
        raise ValueError(
            "expected exactly the known production effective query request"
        )
    return provenance


def _merge_bucket_sources(
    *sources: Mapping[str, dict] | None,
) -> dict[str, dict]:
    """Merge repeatable snapshot sources while rejecting conflicting evidence."""
    merged: dict[str, dict] = {}
    for source in sources:
        if source is None:
            continue
        if not isinstance(source, Mapping):
            raise ValueError("bucket source must be a mapping")
        for raw_id, snapshot in source.items():
            bucket_id = str(raw_id)
            if bucket_id in merged and merged[bucket_id] != snapshot:
                raise ValueError(f"conflicting bucket snapshot for {bucket_id}")
            merged[bucket_id] = copy.deepcopy(snapshot)
    return merged


def _require_full_snapshot(bucket_id: str, snapshot: Any) -> dict:
    if not isinstance(snapshot, dict):
        raise ValueError(f"bucket snapshot missing or invalid for {bucket_id}")
    if str(snapshot.get("id") or "") != bucket_id:
        raise ValueError(f"bucket snapshot ID mismatch for {bucket_id}")
    content = snapshot.get("content")
    metadata = snapshot.get("metadata")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"bucket snapshot content is empty for {bucket_id}")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(f"bucket snapshot metadata is incomplete for {bucket_id}")
    return snapshot


def _build_candidates(
    row: Mapping[str, Any],
    bucket_snapshots: Mapping[str, dict],
) -> list[dict]:
    """Build ordered real candidates and attach only same-request Anchor scores."""
    request_id = _safe_request_id(row)
    raw_ids = row.get("ids_in")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"ids_in is empty for {request_id}")
    ids = [str(value).strip() for value in raw_ids]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"ids_in is invalid for {request_id}")

    scores: dict[str, float] = {}
    anchors = row.get("anchors") or []
    if not isinstance(anchors, list):
        raise ValueError(f"anchors is invalid for {request_id}")
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        anchor_id = str(anchor.get("id") or "")
        score = anchor.get("s")
        if (
            anchor_id
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
        ):
            scores[anchor_id] = float(score)

    candidates: list[dict] = []
    for bucket_id in ids:
        snapshot = _require_full_snapshot(
            bucket_id,
            bucket_snapshots.get(bucket_id),
        )
        candidate = copy.deepcopy(snapshot)
        if bucket_id in scores:
            candidate["_anchor_adapted_relevance_score"] = scores[bucket_id]
        candidates.append(candidate)
    return candidates


def _replay_cap_count(row: Mapping[str, Any], candidates: list[dict]) -> int:
    """Choose a cap that cannot truncate the recorded downstream projection."""
    raw_gate_in = row.get("ds_gate_in")
    if raw_gate_in is None:
        gate_in = 0
    elif (
        isinstance(raw_gate_in, int)
        and not isinstance(raw_gate_in, bool)
        and raw_gate_in >= 0
    ):
        gate_in = raw_gate_in
    else:
        raise ValueError(f"invalid ds_gate_in for {_safe_request_id(row)}")
    return max(1, len(candidates), gate_in)


def _load_baseline_filter() -> tuple[FilterFunction, dict[str, Any], str]:
    """AST-extract the original wrapper and compile it with current globals."""
    resolved = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "rev-parse",
            f"{BASELINE_COMMIT}^{{commit}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != BASELINE_COMMIT:
        raise RuntimeError("fixed DS baseline commit did not resolve exactly")
    source = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "show",
            f"{BASELINE_COMMIT}:server.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tree = ast.parse(source, filename=f"{BASELINE_COMMIT}:server.py")
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == BASELINE_FUNCTION_NAME
    ]
    if len(matches) != 1:
        raise RuntimeError("fixed baseline must contain exactly one DS wrapper")
    original_node = matches[0]
    original_source = ast.get_source_segment(source, original_node)
    if not original_source:
        raise RuntimeError("could not recover fixed baseline wrapper source")
    replay_node = copy.deepcopy(original_node)
    replay_node.name = BASELINE_REPLAY_NAME
    module = ast.fix_missing_locations(
        ast.Module(body=[replay_node], type_ignores=[])
    )
    namespace = dict(vars(server))
    exec(
        compile(module, f"{BASELINE_COMMIT}:server.py", "exec"),
        namespace,
    )
    baseline = namespace.get(BASELINE_REPLAY_NAME)
    if not callable(baseline):
        raise RuntimeError("compiled baseline DS wrapper is unavailable")
    source_sha256 = hashlib.sha256(original_source.encode("utf-8")).hexdigest()
    return baseline, namespace, source_sha256


@contextmanager
def _replay_settings(floor: str = DEFAULT_FLOOR) -> Iterator[None]:
    env_values = {
        "OMBRE_DS_FILTER_ENABLED": "1",
        "OMBRE_DS_FILTER_MODES": "search",
        "OMBRE_DS_FAILURE_FALLBACK_ENABLED": "1",
        "OMBRE_DS_FAILURE_ANCHOR_FLOOR": floor,
        "OMBRE_DS_FILTER_CACHE_TTL": "0",
    }
    previous = {name: os.environ.get(name) for name in env_values}
    previous_logger_disabled = server.logger.disabled
    os.environ.update(env_values)
    server.logger.disabled = True
    try:
        yield
    finally:
        server.logger.disabled = previous_logger_disabled
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _candidate_ids(candidates: list[dict]) -> list[str]:
    return [str(candidate.get("id") or "") for candidate in candidates]


def _render_partial_sha256(
    candidates: list[dict],
    *,
    max_results: int,
) -> tuple[str, bool]:
    rendered = server._local_partial_recall_text(
        candidates,
        max_results=max_results,
        max_tokens=10_000_000,
        state_profile={},
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest(), bool(rendered)


async def _prove_success_equivalence(
    query: str,
    row: Mapping[str, Any],
    candidates: list[dict],
    baseline: FilterFunction,
    baseline_globals: dict[str, Any],
    *,
    floor: str = DEFAULT_FLOOR,
) -> dict:
    """Compare fixed baseline/current success paths over every ordered subset."""
    request_id = _safe_request_id(row)
    max_results = _replay_cap_count(row, candidates)
    forced_ids = server._exact_retrieval_key_ids(query, candidates)
    projected_ids = _candidate_ids(candidates)
    capped = server._cap_candidates_preserving_forced(
        candidates,
        forced_ids,
        max_results,
    )
    if _candidate_ids(capped) != projected_ids:
        raise AssertionError(
            f"replay cap truncated the recorded projection for {request_id}"
        )

    baseline_digest = hashlib.sha256()
    patched_digest = hashlib.sha256()
    subset_count = 1 << len(capped)
    scenario_count = 0
    id_match_count = 0
    partial_match_count = 0
    previous_current_selector = server._ds_semantic_select
    previous_baseline_selector = baseline_globals.get("_ds_semantic_select")
    try:
        with _replay_settings(floor):
            for mask in range(subset_count):
                for allow_empty in (False, True):
                    async def selector(
                        _query: str,
                        runtime_candidates: list[dict],
                        runtime_forced: set[str],
                        _max_results: int,
                        *,
                        _mask: int = mask,
                    ) -> list[dict]:
                        return [
                            candidate
                            for index, candidate in enumerate(runtime_candidates)
                            if (_mask & (1 << index))
                            or str(candidate.get("id") or "") in runtime_forced
                        ]

                    baseline_globals["_ds_semantic_select"] = selector
                    server._ds_semantic_select = selector
                    baseline_selected = await baseline(
                        query,
                        copy.deepcopy(candidates),
                        mode="search",
                        max_results=max_results,
                        force_keep_ids=set(forced_ids),
                        allow_empty=allow_empty,
                    )
                    patched_selected = await server._ds_filter_candidates(
                        query,
                        copy.deepcopy(candidates),
                        mode="search",
                        max_results=max_results,
                        force_keep_ids=set(forced_ids),
                        allow_empty=allow_empty,
                    )
                    baseline_ids = _candidate_ids(baseline_selected)
                    patched_ids = _candidate_ids(patched_selected)
                    baseline_partial_sha, _ = _render_partial_sha256(
                        baseline_selected,
                        max_results=max_results,
                    )
                    patched_partial_sha, _ = _render_partial_sha256(
                        patched_selected,
                        max_results=max_results,
                    )
                    scenario_count += 1
                    if baseline_ids != patched_ids:
                        raise AssertionError(
                            "success path drift in exact IDs for "
                            f"{request_id} mask={mask} allow_empty={allow_empty}"
                        )
                    id_match_count += 1
                    if baseline_partial_sha != patched_partial_sha:
                        raise AssertionError(
                            "success path drift in rendered partial for "
                            f"{request_id} mask={mask} allow_empty={allow_empty}"
                        )
                    partial_match_count += 1
                    baseline_record = json.dumps(
                        {
                            "mask": mask,
                            "allow_empty": allow_empty,
                            "ids": baseline_ids,
                            "partial_sha256": baseline_partial_sha,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    patched_record = json.dumps(
                        {
                            "mask": mask,
                            "allow_empty": allow_empty,
                            "ids": patched_ids,
                            "partial_sha256": patched_partial_sha,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    baseline_digest.update(baseline_record + b"\n")
                    patched_digest.update(patched_record + b"\n")
    finally:
        server._ds_semantic_select = previous_current_selector
        baseline_globals["_ds_semantic_select"] = previous_baseline_selector

    baseline_matrix_sha = baseline_digest.hexdigest()
    patched_matrix_sha = patched_digest.hexdigest()
    if baseline_matrix_sha != patched_matrix_sha:
        raise AssertionError(f"success path matrix digest drift for {request_id}")
    return {
        "request_id": request_id,
        "candidate_ids": projected_ids,
        "forced_ids": sorted(forced_ids),
        "historical_ids_out": [str(value) for value in row.get("ids_out") or []],
        "historical_ds_gate_in": row.get("ds_gate_in"),
        "historical_ds_gate_out": row.get("ds_gate_out"),
        "replay_max_results": max_results,
        "selector_subset_count": subset_count,
        "scenario_count": scenario_count,
        "exact_id_match_count": id_match_count,
        "partial_hash_match_count": partial_match_count,
        "exact_id_equivalence": id_match_count == scenario_count,
        "partial_hash_equivalence": partial_match_count == scenario_count,
        "baseline_output_matrix_sha256": baseline_matrix_sha,
        "patched_output_matrix_sha256": patched_matrix_sha,
    }


def _timing_receipt_after_failure(
    token: Any,
    *,
    cancelled: bool,
) -> dict:
    try:
        return finish_recall_timing(
            status="deadline" if cancelled else "ok",
            partial=cancelled,
        )
    finally:
        reset_recall_timing(token)


async def _replay_failure(
    query: str,
    row: Mapping[str, Any],
    candidates: list[dict],
    *,
    floor: str = DEFAULT_FLOOR,
) -> dict:
    """Exercise the patched wrapper and actual partial helper for one failure."""
    request_id = _safe_request_id(row)
    injection = EXPECTED_FAILURE_INJECTIONS[request_id]
    max_results = _replay_cap_count(row, candidates)
    forced_ids = server._exact_retrieval_key_ids(query, candidates)
    projected_ids = _candidate_ids(candidates)
    capped = server._cap_candidates_preserving_forced(
        candidates,
        forced_ids,
        max_results,
    )
    if _candidate_ids(capped) != projected_ids:
        raise AssertionError(
            f"replay cap truncated the recorded projection for {request_id}"
        )

    previous_selector = server._ds_semantic_select
    with _replay_settings(floor):
        helper_selected = server._ds_conservative_failure_candidates(
            copy.deepcopy(candidates),
            force_keep_ids=set(forced_ids),
            max_results=max_results,
        )
        helper_ids = _candidate_ids(helper_selected)
        if not helper_ids:
            raise AssertionError(f"failure helper returned empty for {request_id}")
        partial_sha, partial_nonempty = _render_partial_sha256(
            helper_selected,
            max_results=max_results,
        )
        if not partial_nonempty:
            raise AssertionError(
                f"failure helper rendered an empty partial for {request_id}"
            )

        selected: list[dict] | None = None
        cancel_propagated = False
        token = begin_recall_timing()
        try:
            if injection == "invalid":
                async def invalid_selector(*_args: Any, **_kwargs: Any) -> list[dict]:
                    raise server.DSFilterInvalidPayloadError("replay_invalid")

                server._ds_semantic_select = invalid_selector
                selected = await server._ds_filter_candidates(
                    query,
                    copy.deepcopy(candidates),
                    mode="search",
                    max_results=max_results,
                    force_keep_ids=set(forced_ids),
                    allow_empty=True,
                )
            elif injection == "timeout":
                async def timeout_selector(*_args: Any, **_kwargs: Any) -> list[dict]:
                    raise asyncio.TimeoutError()

                server._ds_semantic_select = timeout_selector
                selected = await server._ds_filter_candidates(
                    query,
                    copy.deepcopy(candidates),
                    mode="search",
                    max_results=max_results,
                    force_keep_ids=set(forced_ids),
                    allow_empty=True,
                )
            elif injection == "cancel":
                entered = asyncio.Event()

                async def cancelled_selector(
                    *_args: Any,
                    **_kwargs: Any,
                ) -> list[dict]:
                    entered.set()
                    await asyncio.Event().wait()
                    raise AssertionError("unreachable")

                server._ds_semantic_select = cancelled_selector
                task = asyncio.create_task(
                    server._ds_filter_candidates(
                        query,
                        copy.deepcopy(candidates),
                        mode="search",
                        max_results=max_results,
                        force_keep_ids=set(forced_ids),
                        allow_empty=True,
                    )
                )
                await entered.wait()
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    cancel_propagated = True
                if not cancel_propagated:
                    raise AssertionError(
                        f"outer cancellation was swallowed for {request_id}"
                    )
            else:
                raise ValueError(f"unsupported failure injection for {request_id}")
            receipt = _timing_receipt_after_failure(
                token,
                cancelled=injection == "cancel",
            )
            token = None
        finally:
            if token is not None:
                reset_recall_timing(token)
            server._ds_semantic_select = previous_selector

    if selected is not None and _candidate_ids(selected) != helper_ids:
        raise AssertionError(
            f"patched failure wrapper disagrees with helper for {request_id}"
        )
    expected_status = "timeout" if injection == "cancel" else injection
    if receipt.get("ds_status") != expected_status:
        raise AssertionError(f"DS status timing drift for {request_id}")
    if receipt.get("ds_gate_in") != len(candidates):
        raise AssertionError(f"DS input timing count drift for {request_id}")
    if receipt.get("ds_gate_out") != len(helper_selected):
        raise AssertionError(f"DS output timing count drift for {request_id}")
    return {
        "request_id": request_id,
        "injected_failure": injection,
        "candidate_ids": projected_ids,
        "forced_ids": sorted(forced_ids),
        "historical_ids_out": [str(value) for value in row.get("ids_out") or []],
        "failure_primary_ids_after": helper_ids,
        "rendered_partial_sha256": partial_sha,
        "rendered_partial_nonempty": partial_nonempty,
        "cancel_propagated": cancel_propagated if injection == "cancel" else None,
        "timing_ds_status": receipt.get("ds_status"),
        "timing_ds_gate_in": receipt.get("ds_gate_in"),
        "timing_ds_gate_out": receipt.get("ds_gate_out"),
        "replay_max_results": max_results,
    }


async def replay(
    rows: list[dict],
    bucket_snapshots: Mapping[str, dict],
    *,
    floor: str = DEFAULT_FLOOR,
) -> dict:
    provenance = _validate_ledger_rows(rows)
    projection_ids = {
        str(bucket_id)
        for row in rows
        for bucket_id in (row.get("ids_in") or [])
    }
    if len(projection_ids) != EXPECTED_PROJECTION_UNIQUE_COUNT:
        raise ValueError(
            "expected exactly 88 unique ids_in bucket projections"
        )
    missing = sorted(projection_ids - set(bucket_snapshots))
    if missing:
        raise ValueError(
            "bucket snapshot coverage is incomplete for IDs: "
            + ",".join(missing)
        )

    candidates_by_request = {
        _safe_request_id(row): _build_candidates(row, bucket_snapshots)
        for row in rows
    }
    baseline, baseline_globals, baseline_source_sha = _load_baseline_filter()
    success_results: list[dict] = []
    failure_results: list[dict] = []
    query_results: list[dict] = []

    for row in rows:
        request_id = _safe_request_id(row)
        query, source, verified = provenance[request_id]
        query_results.append({
            "request_id": request_id,
            "query_source": source,
            "query_sha_verified": verified,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        })
        candidates = candidates_by_request[request_id]
        if row.get("ds_gate_outcome") == "ok":
            success_results.append(
                await _prove_success_equivalence(
                    query,
                    row,
                    candidates,
                    baseline,
                    baseline_globals,
                    floor=floor,
                )
            )
        else:
            failure_results.append(
                await _replay_failure(
                    query,
                    row,
                    candidates,
                    floor=floor,
                )
            )

    target = next(
        (
            result
            for result in failure_results
            if result["request_id"] == TARGET_REQUEST_ID
        ),
        None,
    )
    if target is None:
        raise AssertionError("12:14 target request is absent from failure replay")
    target_after = set(target["failure_primary_ids_after"])
    target_forced = set(target["forced_ids"])
    target_noise_admitted = len(TARGET_NOISE_IDS & target_after)
    target_noise_forced = len(TARGET_NOISE_IDS & target_forced)
    if target_noise_admitted:
        raise AssertionError("12:14 engineering noise survived fallback replay")
    if target_noise_forced:
        raise AssertionError("12:14 engineering noise matched a retrieval key")
    if len(success_results) != EXPECTED_OK_COUNT:
        raise AssertionError("successful replay result count drifted")
    if len(failure_results) != len(EXPECTED_FAILURE_INJECTIONS):
        raise AssertionError("failure replay result count drifted")

    original_verified = sum(
        result["query_source"] == "original_query"
        and result["query_sha_verified"]
        for result in query_results
    )
    effective_count = sum(
        result["query_source"] == "production_effective_query"
        for result in query_results
    )
    success_scenarios = sum(
        result["scenario_count"] for result in success_results
    )
    return {
        "kind": (
            "baseline_commit_vs_patch_structural_success_contract_and_"
            "failure_route_replay_not_production_acceptance"
        ),
        "proof_scope": (
            "structural_baseline_vs_patch; no historical provider or "
            "end-to-end reproduction"
        ),
        "historical_ids_out_role": "reported_only",
        "provider_call_count": 0,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_function_source_sha256": baseline_source_sha,
        "floor": float(floor),
        "request_count": len(rows),
        "projection_occurrence_count": sum(
            len(row.get("ids_in") or []) for row in rows
        ),
        "projection_unique_count": len(projection_ids),
        "projection_snapshot_complete_count": len(projection_ids),
        "original_query_sha_verified_count": original_verified,
        "production_effective_query_count": effective_count,
        "production_effective_query_request_id": (
            PRODUCTION_EFFECTIVE_QUERY_REQUEST_ID
        ),
        "ok_request_count": len(success_results),
        "ok_structural_equivalence_count": sum(
            result["exact_id_equivalence"]
            and result["partial_hash_equivalence"]
            for result in success_results
        ),
        "success_selector_scenario_count": success_scenarios,
        "failure_replayed_count": len(failure_results),
        "target_request_id": TARGET_REQUEST_ID,
        "target_noise_admitted": target_noise_admitted,
        "target_noise_forced": target_noise_forced,
        "query_provenance": query_results,
        "success_results": success_results,
        "failure_results": failure_results,
    }


def _read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument(
        "--supplemental-buckets",
        action="append",
        default=[],
        type=Path,
        help=(
            "repeatable private JSON source containing a buckets mapping; "
            "conflicting snapshots are rejected"
        ),
    )
    parser.add_argument("--floor", default=DEFAULT_FLOOR)
    args = parser.parse_args()

    ledger = _read_json_object(args.ledger)
    rows = ledger.get("rows")
    if not isinstance(rows, list):
        raise ValueError("ledger rows must be a list")
    sources: list[Mapping[str, dict] | None] = [ledger.get("buckets")]
    for path in args.supplemental_buckets:
        sources.append(_read_json_object(path).get("buckets"))
    bucket_snapshots = _merge_bucket_sources(*sources)
    result = asyncio.run(replay(rows, bucket_snapshots, floor=args.floor))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
