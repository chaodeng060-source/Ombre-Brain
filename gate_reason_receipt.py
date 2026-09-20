"""Bounded, request-local model reasons; never participate in gate decisions.

Kept separate from content-free timing and operational logs. Only the HTTP
bridge returns the receipt to its existing authenticated caller; no vault IO.
"""
from __future__ import annotations

import contextvars
import json
import os

MAX_ITEMS = 64
MAX_REASON_CHARS = 240
MAX_BYTES = 32768
_capture = contextvars.ContextVar("gate_reason_receipt", default=None)


def enabled() -> bool:
    return os.getenv("OMBRE_DS_GATE_REASONS_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}


def begin():
    state = {}
    return state, _capture.set(state if enabled() else None)


def reset(token):
    _capture.reset(token)


def current():
    return _capture.get()


class Scores(list):
    """List-compatible internal score vector with aligned diagnostic metadata."""
    def __init__(self, values, reasons):
        super().__init__(values)
        self.reasons = reasons


def missing(status):
    return {"reason": None, "reason_status": status, "reason_truncated": False}


def with_reasons(values, verdicts):
    if not enabled():
        return values
    by_index = {row["candidate"]: row for row in verdicts}
    reasons = []
    for i in range(len(values)):
        row = by_index[i]
        reason = row.get("reason")
        if isinstance(reason, str) and reason:
            # A malformed JSON surrogate must not turn successful selection
            # into an error when the optional receipt is UTF-8 serialized.
            try:
                reason[:MAX_REASON_CHARS].encode("utf-8")
            except UnicodeEncodeError:
                reasons.append(missing("unparsable"))
                continue
            reasons.append({"reason": reason[:MAX_REASON_CHARS], "reason_status": "provided",
                            "reason_truncated": len(reason) > MAX_REASON_CHARS})
        else:
            reasons.append(missing("absent_in_response" if reason is None or reason == "" else "unparsable"))
    return Scores(values, reasons)


def restore(entry):
    values = list(entry[1])
    if not enabled():
        return values
    reasons = entry[2] if len(entry) > 2 else [missing("absent_legacy_cache") for _ in values]
    return Scores(values, reasons)


def cache_entry(timestamp, values):
    if enabled() and isinstance(values, Scores):
        return timestamp, list(values), values.reasons
    return timestamp, list(values)


def combine(parts):
    values = [value for part in parts for value in part]
    if not enabled():
        return values
    reasons = [row for part in parts for row in getattr(part, "reasons", [missing("unparsable") for _ in part])]
    return Scores(values, reasons)


def record_rejections(buckets, scores, threshold):
    state = current()
    if not enabled() or state is None:
        return
    reasons = getattr(scores, "reasons", [missing("unparsable") for _ in scores])
    rejected = [i for i, bucket in enumerate(buckets) if bucket["_ds_gate_final"] < threshold]
    receipt = {"schema_version": 1, "items": [], "omitted": len(rejected)}
    for i in rejected[:MAX_ITEMS]:
        bucket = buckets[i]
        row = {"id": str(bucket.get("id") or "")[:64], "score": int(scores[i]),
               "final": int(bucket["_ds_gate_final"]), **reasons[i]}
        receipt["items"].append(row)
        receipt["omitted"] -= 1
        if len(json.dumps(receipt, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
            receipt["items"].pop()
            receipt["omitted"] += 1
            break
    state["receipt"] = receipt
