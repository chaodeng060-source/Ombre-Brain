"""Request-local timing receipt for the read-only recall path.

The receipt intentionally contains no query text, bucket content, model output,
or credentials.  A ContextVar keeps concurrent ``/api/breath`` requests
isolated while allowing lower-level modules such as ``embedding_engine`` to
contribute their own stage timings without changing public call signatures.
"""
from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Iterator
from uuid import uuid4


SCHEMA_VERSION = 1

_recall_timing = contextvars.ContextVar("ombre_recall_timing", default=None)
_DEHYDRATION_OUTCOMES = frozenset({
    "frontmatter_hits",
    "backfilled",
    "computed",
    "passthrough",
    "passthrough_async",
    "persist_failed",
})
_DS_GATE_OUTCOMES = frozenset({
    "ok",
    "timeout",
    "error",
    "noop",
    "disabled",
})
_RECALL_METRIC_MODES = {
    "query_angle_count": "max",
    "keyword_bucket_count": "max",
    "vector_cache_hits": "sum",
    "vector_cache_coalesced_hits": "sum",
    "vector_cache_misses": "sum",
    "vector_cache_cold_loads": "sum",
    "vector_cache_incremental_refreshes": "sum",
    "vector_cache_rows_examined": "sum",
    "vector_cache_rows_loaded": "sum",
    "vector_cache_rows_reused": "sum",
    "vector_cache_rows_removed": "sum",
    "vector_pg_rows": "sum",
    "vector_entries_scanned": "sum",
    "vector_stored_segments_scanned": "sum",
    "vector_segment_comparisons": "sum",
    "vector_invalid_rows": "sum",
    "vector_pg_shadow_busy_skips": "sum",
    "vector_dimension": "max",
    "curated_lexical_shadow_timeouts": "sum",
    "relation_slot_reserved": "sum",
    "relation_primary_restored": "sum",
}


def begin_recall_timing() -> contextvars.Token:
    state = {
        "schema_version": SCHEMA_VERSION,
        "request_id": uuid4().hex,
        "started_at": time.perf_counter(),
        "stages": {},
        "active_stages": {},
        "dehydration": {},
        "metrics": {},
        "degraded": False,
    }
    return _recall_timing.set(state)


def get_recall_request_id() -> str:
    """Return the current content-free request correlation id."""
    state = _recall_timing.get()
    if not isinstance(state, dict):
        return ""
    request_id = state.get("request_id")
    return request_id if isinstance(request_id, str) else ""


def reset_recall_timing(token: contextvars.Token) -> None:
    _recall_timing.reset(token)


def set_recall_partial_result(value: str) -> None:
    state = _recall_timing.get()
    if isinstance(state, dict):
        state["partial_result"] = str(value or "")


def get_recall_partial_result() -> str:
    state = _recall_timing.get()
    if not isinstance(state, dict):
        return ""
    value = state.get("partial_result")
    return value if isinstance(value, str) else ""


def mark_recall_partial() -> None:
    """Mark a completed recall as degraded without storing query content."""
    state = _recall_timing.get()
    if isinstance(state, dict):
        state["degraded"] = True


def recall_is_partial() -> bool:
    state = _recall_timing.get()
    return bool(state.get("degraded", False)) if isinstance(state, dict) else False


def record_recall_stage(name: str, elapsed_seconds: float) -> None:
    state = _recall_timing.get()
    if not isinstance(state, dict):
        return
    stages = state.get("stages")
    if not isinstance(stages, dict):
        return
    elapsed_ms = max(0.0, float(elapsed_seconds) * 1000.0)
    entry = stages.setdefault(name, {"elapsed_ms": 0.0, "calls": 0})
    entry["elapsed_ms"] = round(float(entry["elapsed_ms"]) + elapsed_ms, 3)
    entry["calls"] = int(entry["calls"]) + 1


def record_recall_dehydration(outcome: str, count: int = 1) -> None:
    """Count content-free dehydration outcomes for one recall request."""
    if outcome not in _DEHYDRATION_OUTCOMES:
        raise ValueError(f"unknown dehydration outcome: {outcome}")
    state = _recall_timing.get()
    if not isinstance(state, dict):
        return
    counters = state.get("dehydration")
    if not isinstance(counters, dict):
        return
    counters[outcome] = int(counters.get(outcome, 0)) + max(0, int(count))


def record_recall_metric(name: str, value: int) -> None:
    """Record one allowlisted, content-free diagnostic count."""
    mode = _RECALL_METRIC_MODES.get(name)
    if mode is None:
        raise ValueError(f"unknown recall metric: {name}")
    state = _recall_timing.get()
    if not isinstance(state, dict):
        return
    metrics = state.get("metrics")
    if not isinstance(metrics, dict):
        return
    number = max(0, int(value))
    if mode == "max":
        metrics[name] = max(int(metrics.get(name, 0)), number)
    else:
        metrics[name] = int(metrics.get(name, 0)) + number


def record_recall_ds_gate(
    outcome: str,
    input_count: int,
    output_count: int,
) -> None:
    """Record the content-free semantic-gate result for this recall request."""
    if outcome not in _DS_GATE_OUTCOMES:
        raise ValueError(f"unknown ds gate outcome: {outcome}")
    state = _recall_timing.get()
    if not isinstance(state, dict):
        return
    state["ds_gate"] = {
        "outcome": outcome,
        "input_count": max(0, int(input_count)),
        "output_count": max(0, int(output_count)),
    }


def start_recall_stage(name: str) -> None:
    state = _recall_timing.get()
    if not isinstance(state, dict):
        return
    active = state.get("active_stages")
    if isinstance(active, dict):
        active[name] = time.perf_counter()


def finish_recall_stage(name: str) -> None:
    state = _recall_timing.get()
    if not isinstance(state, dict):
        return
    active = state.get("active_stages")
    if not isinstance(active, dict):
        return
    started_at = active.pop(name, None)
    if isinstance(started_at, (int, float)):
        record_recall_stage(name, time.perf_counter() - started_at)


@contextmanager
def recall_stage(name: str) -> Iterator[None]:
    started_at = time.perf_counter()
    try:
        yield
    finally:
        record_recall_stage(name, time.perf_counter() - started_at)


def finish_recall_timing(*, status: str, partial: bool) -> dict:
    state = _recall_timing.get()
    if not isinstance(state, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": "",
            "status": status,
            "partial": bool(partial),
            "total_ms": 0.0,
            "unattributed_ms": 0.0,
            "stages": {},
            "dehydration": {},
            "metrics": {},
        }

    total_ms = max(0.0, (time.perf_counter() - state["started_at"]) * 1000.0)
    stages = {
        name: {
            "elapsed_ms": round(float(entry.get("elapsed_ms", 0.0)), 3),
            "calls": int(entry.get("calls", 0)),
        }
        for name, entry in state["stages"].items()
    }
    active = state.get("active_stages")
    if isinstance(active, dict) and active:
        now = time.perf_counter()
        for name, started_at in active.items():
            if not isinstance(started_at, (int, float)):
                continue
            entry = stages.setdefault(name, {"elapsed_ms": 0.0, "calls": 0})
            entry["elapsed_ms"] = round(
                float(entry["elapsed_ms"]) + max(0.0, now - started_at) * 1000.0,
                3,
            )
            entry["calls"] = int(entry["calls"]) + 1
    attributed_ms = sum(entry["elapsed_ms"] for entry in stages.values())
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "request_id": state["request_id"],
        "status": status,
        "partial": bool(partial),
        "total_ms": round(total_ms, 3),
        "unattributed_ms": round(max(0.0, total_ms - attributed_ms), 3),
        "stages": stages,
        "metrics": {
            name: int(value)
            for name, value in state.get("metrics", {}).items()
            if name in _RECALL_METRIC_MODES and int(value) >= 0
        },
        "dehydration": {
            name: int(count)
            for name, count in state.get("dehydration", {}).items()
            if name in _DEHYDRATION_OUTCOMES and int(count) > 0
        },
    }
    ds_gate = state.get("ds_gate")
    if (
        isinstance(ds_gate, dict)
        and ds_gate.get("outcome") in _DS_GATE_OUTCOMES
    ):
        receipt.update({
            "ds_gate_outcome": ds_gate["outcome"],
            "ds_gate_in": max(0, int(ds_gate.get("input_count", 0))),
            "ds_gate_out": max(0, int(ds_gate.get("output_count", 0))),
        })
    return receipt
