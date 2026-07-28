"""Strict E-axis shadow annotations stored outside the recall corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from maintenance_barrier import MaintenanceBarrier
from storage_safety import advisory_file_lock


CONTRACT_VERSION = 1
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
_SUCCESS_ROW_FIELDS = frozenset({
    "contract_version",
    "annotation_key",
    "status",
    "bucket_id",
    "source_digest",
    "scorer",
    "model",
    "rubric_version",
    "scored_at",
    "shadow_only",
    "affects_ranking",
    "score",
})
_FAILURE_ROW_FIELDS = frozenset({
    "contract_version",
    "annotation_key",
    "status",
    "bucket_id",
    "source_digest",
    "scorer",
    "model",
    "rubric_version",
    "category",
    "scored_at",
    "shadow_only",
    "affects_ranking",
})


def _plain_finite_number(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def normalize_min_confidence(value):
    """Return a finite ``[0, 1]`` threshold or ``None`` for bad config."""
    if not _plain_finite_number(value):
        return None
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        return None
    return normalized


def _reject_nonfinite_constant(token: str):
    raise ValueError(f"non-finite JSON number: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("JSON number overflows finite float range")
    return value


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def strict_json_loads(raw):
    """Parse JSON without Python's duplicate-key/non-finite extensions."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if type(raw) is not str:
        raise TypeError("JSON input must be text or bytes")
    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_parse_finite_float,
    )


def validate_shadow_score(payload, *, min_confidence: float = 0.3):
    """Return ``(normalized_score, error_category)`` without guessing values."""
    min_confidence = normalize_min_confidence(min_confidence)
    if min_confidence is None:
        return None, "config.min_confidence"
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


def _normalized_timestamp(value, field: str = "scored_at") -> str:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.isoformat(timespec="seconds")


def _success_key(
    bucket_id: str,
    source_digest: str,
    scorer: str,
    model: str,
    rubric_version: str,
) -> str:
    key_material = "\x1f".join((
        bucket_id,
        source_digest,
        scorer,
        model,
        rubric_version,
    ))
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


def _failure_key(
    bucket_id: str,
    source_digest: str,
    scorer: str,
    model: str,
    rubric_version: str,
    category: str,
) -> str:
    material = "\x1f".join((
        bucket_id,
        source_digest,
        scorer,
        model,
        rubric_version,
        category,
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


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
            scored_at = _normalized_timestamp(scored_at)
        except ValueError:
            return None, "schema.scored_at"

    annotation_key = _success_key(
        bucket_id,
        source_digest,
        scorer,
        model,
        rubric_version,
    )
    return {
        "contract_version": CONTRACT_VERSION,
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
    bucket_id = str(bucket_id or "").strip()[:160]
    source_digest = str(source_digest or "").strip().lower()[:64]
    scorer = str(scorer or "").strip()[:160]
    model = str(model or "").strip()[:160]
    rubric_version = str(rubric_version or "").strip()[:160]
    category = str(category or "unknown").strip()[:160]
    return {
        "contract_version": CONTRACT_VERSION,
        "annotation_key": _failure_key(
            bucket_id,
            source_digest,
            scorer,
            model,
            rubric_version,
            category,
        ),
        "status": "failed",
        "bucket_id": bucket_id,
        "source_digest": source_digest,
        "scorer": scorer,
        "model": model,
        "rubric_version": rubric_version,
        "category": category,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shadow_only": True,
        "affects_ranking": False,
    }


def rank_multiplier(_annotation) -> float:
    """E is annotation-only in stage 1 and can never change ordering."""
    return 1.0


class EAxisShadowStore:
    """Append-only, fsynced, idempotent JSONL store with corruption fail-close."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        maintenance_root: str | os.PathLike | None = None,
    ):
        self.path = Path(path)
        self.lock_path = Path(f"{self.path}.lock")
        parent = self.path.parent
        root = (
            Path(maintenance_root)
            if maintenance_root is not None
            else parent.parent if parent.name.startswith(".") else parent
        )
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._maintenance_barrier = MaintenanceBarrier(root)

    @staticmethod
    def _validate_row(row: dict) -> bool:
        if type(row) is not dict:
            return False
        status = row.get("status")
        expected_fields = (
            _SUCCESS_ROW_FIELDS if status == "success"
            else _FAILURE_ROW_FIELDS if status == "failed"
            else None
        )
        if expected_fields is None or set(row) != expected_fields:
            return False
        if type(row.get("contract_version")) is not int \
                or row["contract_version"] != CONTRACT_VERSION:
            return False
        if row.get("shadow_only") is not True \
                or row.get("affects_ranking") is not False:
            return False
        annotation_key = row.get("annotation_key")
        source_digest = row.get("source_digest")
        if type(annotation_key) is not str or not _SHA256_RE.fullmatch(annotation_key):
            return False
        if type(source_digest) is not str or not _SHA256_RE.fullmatch(source_digest):
            return False
        try:
            bucket_id = _required_text(row.get("bucket_id"), "bucket_id")
            scorer = _required_text(row.get("scorer"), "scorer")
            model = _required_text(row.get("model"), "model")
            rubric_version = _required_text(
                row.get("rubric_version"),
                "rubric_version",
            )
            _normalized_timestamp(row.get("scored_at"))
        except ValueError:
            return False

        if status == "success":
            normalized_score, error = validate_shadow_score(
                row.get("score"),
                min_confidence=0.0,
            )
            if error or normalized_score is None:
                return False
            expected_key = _success_key(
                bucket_id,
                source_digest,
                scorer,
                model,
                rubric_version,
            )
        else:
            try:
                category = _required_text(row.get("category"), "category")
            except ValueError:
                return False
            expected_key = _failure_key(
                bucket_id,
                source_digest,
                scorer,
                model,
                rubric_version,
                category,
            )
        return annotation_key == expected_key

    @staticmethod
    def _load_locked(handle) -> list[dict]:
        handle.seek(0)
        rows = []
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = strict_json_loads(raw)
            except (UnicodeDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"corrupt E shadow ledger at line {line_number}"
                ) from exc
            if not EAxisShadowStore._validate_row(row):
                raise ValueError(f"invalid E shadow ledger row at line {line_number}")
            rows.append(row)
        return rows

    def load(self) -> list[dict]:
        with advisory_file_lock(self.lock_path):
            if not self.path.exists():
                return []
            with open(self.path, "r", encoding="utf-8") as handle:
                return self._load_locked(handle)

    def append(self, row: dict) -> bool:
        with self._maintenance_barrier.shared():
            return self._append_locked(row)

    def _append_locked(self, row: dict) -> bool:
        if not self._validate_row(row):
            raise ValueError("invalid E shadow row")
        key = str(row.get("annotation_key") or "").strip()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(self.lock_path):
            fd = None
            try:
                fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
                handle = os.fdopen(fd, "r+", encoding="utf-8")
                fd = None  # os.fdopen owns and closes it from this point.
                with handle:
                    rows = self._load_locked(handle)
                    if any(existing.get("annotation_key") == key for existing in rows):
                        return False
                    handle.seek(0, os.SEEK_END)
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                    os.chmod(self.path, 0o600)
                    return True
            finally:
                if fd is not None:
                    os.close(fd)
