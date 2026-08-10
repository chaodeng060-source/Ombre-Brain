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


def begin_recall_timing() -> contextvars.Token:
    state = {
        "schema_version": SCHEMA_VERSION,
        "request_id": uuid4().hex,
        "started_at": time.perf_counter(),
        "stages": {},
    }
    return _recall_timing.set(state)


def reset_recall_timing(token: contextvars.Token) -> None:
    _recall_timing.reset(token)


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
        }

    total_ms = max(0.0, (time.perf_counter() - state["started_at"]) * 1000.0)
    stages = {
        name: {
            "elapsed_ms": round(float(entry.get("elapsed_ms", 0.0)), 3),
            "calls": int(entry.get("calls", 0)),
        }
        for name, entry in state["stages"].items()
    }
    attributed_ms = sum(entry["elapsed_ms"] for entry in stages.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": state["request_id"],
        "status": status,
        "partial": bool(partial),
        "total_ms": round(total_ms, 3),
        "unattributed_ms": round(max(0.0, total_ms - attributed_ms), 3),
        "stages": stages,
    }
