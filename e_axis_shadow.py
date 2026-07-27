"""Strict E-axis shadow annotations stored outside the recall corpus."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path


SCORE_FIELDS = frozenset({
    "valence",
    "arousal",
    "tension",
    "confidence",
    "response_tendency",
    "growth_delta",
})
RESPONSE_TENDENCIES = frozenset({"comfort", "engage", "withdraw", "alert"})
GROWTH_DELTAS = frozenset({"growth", "stable", "setback"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _plain_finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def validate_shadow_score(payload, *, min_confidence: float = 0.3):
    """Return ``(normalized_score, error_category)`` without guessing values."""
    if type(payload) is not dict:
        return None, "schema.root"
    keys = set(payload)
    if keys != SCORE_FIELDS:
        if SCORE_FIELDS - keys:
            return None, "schema.missing"
        return None, "schema.unexpected"

    numeric = {}
    for field in ("valence", "arousal", "tension", "confidence"):
        value = payload[field]
        if not _plain_finite_number(value):
            return None, f"schema.{field}"
        numeric[field] = float(value)

    if not -1.0 <= numeric["valence"] <= 1.0:
        return None, "range.valence"
    for field in ("arousal", "tension", "confidence"):
        if not 0.0 <= numeric[field] <= 1.0:
            return None, f"range.{field}"
    if numeric["confidence"] < min_confidence:
        return None, "confidence.low"

    tendency = payload["response_tendency"]
    if type(tendency) is not str or tendency not in RESPONSE_TENDENCIES:
        return None, "enum.response_tendency"
    growth = payload["growth_delta"]
    if type(growth) is not str or growth not in GROWTH_DELTAS:
        return None, "enum.growth_delta"

    return {
        **numeric,
        "response_tendency": tendency,
        "growth_delta": growth,
    }, None


def _required_text(value, field: str, max_length: int = 160) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        raise ValueError(f"{field} must be 1..{max_length} chars")
    return normalized


def build_shadow_annotation(
    *,
    bucket_id: str,
    source_digest: str,
    scorer: str,
    model: str,
    rubric_version: str,
    score,
    scored_at: str | None = None,
    min_confidence: float = 0.3,
):
    """Build one immutable success record or return a categorized failure."""
    try:
        bucket_id = _required_text(bucket_id, "bucket_id")
        scorer = _required_text(scorer, "scorer")
        model = _required_text(model, "model")
        rubric_version = _required_text(rubric_version, "rubric_version")
    except ValueError:
        return None, "schema.provenance"

    source_digest = str(source_digest or "").strip().lower()
    if not _SHA256_RE.fullmatch(source_digest):
        return None, "schema.source_digest"

    normalized_score, error = validate_shadow_score(
        score,
        min_confidence=min_confidence,
    )
    if error:
        return None, error

    if scored_at is None:
        scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        try:
            parsed = datetime.fromisoformat(str(scored_at))
        except (TypeError, ValueError):
            return None, "schema.scored_at"
        if parsed.tzinfo is None:
            return None, "schema.scored_at"
        scored_at = parsed.isoformat(timespec="seconds")

    key_material = "\x1f".join((
        bucket_id,
        source_digest,
        scorer,
        model,
        rubric_version,
    ))
    annotation_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    return {
        "annotation_key": annotation_key,
        "status": "success",
        "bucket_id": bucket_id,
        "source_digest": source_digest,
        "scorer": scorer,
        "model": model,
        "rubric_version": rubric_version,
        "scored_at": scored_at,
        "shadow_only": True,
        "affects_ranking": False,
        "score": normalized_score,
    }, None


def build_failure_record(
    *,
    bucket_id: str,
    source_digest: str,
    scorer: str,
    model: str,
    rubric_version: str,
    category: str,
):
    """Build a failure row without storing model output or memory content."""
    values = {
        "bucket_id": str(bucket_id or "").strip()[:160],
        "source_digest": str(source_digest or "").strip().lower()[:64],
        "scorer": str(scorer or "").strip()[:160],
        "model": str(model or "").strip()[:160],
        "rubric_version": str(rubric_version or "").strip()[:160],
        "category": str(category or "unknown").strip()[:160],
    }
    material = "\x1f".join(values.values())
    return {
        "annotation_key": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "status": "failed",
        **values,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shadow_only": True,
        "affects_ranking": False,
    }


def rank_multiplier(_annotation) -> float:
    """E is annotation-only in stage 1 and can never change ordering."""
    return 1.0


class EAxisShadowStore:
    """Append-only, fsynced, idempotent JSONL store with corruption fail-close."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    @staticmethod
    def _load_locked(handle) -> list[dict]:
        handle.seek(0)
        rows = []
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"corrupt E shadow ledger at line {line_number}"
                ) from exc
            if type(row) is not dict or not row.get("annotation_key"):
                raise ValueError(f"invalid E shadow ledger row at line {line_number}")
            rows.append(row)
        return rows

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return self._load_locked(handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append(self, row: dict) -> bool:
        key = str(row.get("annotation_key") or "").strip()
        if not key:
            raise ValueError("E shadow row requires annotation_key")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            with os.fdopen(fd, "r+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                rows = self._load_locked(handle)
                if any(existing.get("annotation_key") == key for existing in rows):
                    return False
                handle.seek(0, os.SEEK_END)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                os.chmod(self.path, 0o600)
                return True
        except Exception:
            # fd is owned by fdopen only after it succeeds.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
