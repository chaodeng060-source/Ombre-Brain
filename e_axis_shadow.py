"""Strict E-axis shadow annotations stored outside the recall corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from e_axis_storage import (
    EAxisStorageError,
    open_secure_e_axis_jsonl,
    secure_e_axis_lock,
)
from maintenance_barrier import MaintenanceBarrier


CONTRACT_VERSION = 2
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
_MACHINE_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_SUCCESS_ROW_FIELDS = frozenset({
    "contract_version",
    "annotation_key",
    "status",
    "bucket_id",
    "source_digest",
    "source_kind",
    "source_run_id",
    "provider",
    "scorer",
    "model",
    "rubric_version",
    "run_id",
    "trigger_reason",
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
    "source_kind",
    "source_run_id",
    "provider",
    "scorer",
    "model",
    "rubric_version",
    "run_id",
    "trigger_reason",
    "category",
    "retryable",
    "scored_at",
    "shadow_only",
    "affects_ranking",
})


class EAxisShadowConflictError(ValueError):
    """An annotation identity was reused with different immutable data."""


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


def _required_machine_text(value, field: str) -> str:
    normalized = _required_text(value, field)
    if _MACHINE_TEXT_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be machine text")
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
    source_kind: str,
    source_run_id: str,
    provider: str,
    scorer: str,
    model: str,
    rubric_version: str,
) -> str:
    key_material = "\x1f".join((
        bucket_id,
        source_digest,
        source_kind,
        source_run_id,
        provider,
        scorer,
        model,
        rubric_version,
    ))
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


def _failure_key(
    bucket_id: str,
    source_digest: str,
    source_kind: str,
    source_run_id: str,
    provider: str,
    scorer: str,
    model: str,
    rubric_version: str,
    category: str,
    retryable: bool,
    run_id: str,
) -> str:
    material = "\x1f".join((
        bucket_id,
        source_digest,
        source_kind,
        source_run_id,
        provider,
        scorer,
        model,
        rubric_version,
        category,
        "retryable" if retryable else "terminal",
        run_id,
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_shadow_annotation(
    *,
    bucket_id: str,
    source_digest: str,
    source_kind: str,
    source_run_id: str,
    provider: str,
    scorer: str,
    model: str,
    rubric_version: str,
    run_id: str,
    trigger_reason: str,
    score,
    scored_at: str | None = None,
    min_confidence: float = 0.3,
):
    """Build one immutable success record or return a categorized failure."""
    try:
        bucket_id = _required_text(bucket_id, "bucket_id")
        source_kind = _required_machine_text(source_kind, "source_kind")
        source_run_id = _required_machine_text(source_run_id, "source_run_id")
        provider = _required_text(provider, "provider")
        scorer = _required_text(scorer, "scorer")
        model = _required_text(model, "model")
        rubric_version = _required_text(rubric_version, "rubric_version")
        run_id = _required_text(run_id, "run_id")
        trigger_reason = _required_text(trigger_reason, "trigger_reason")
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
        source_kind,
        source_run_id,
        provider,
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
        "source_kind": source_kind,
        "source_run_id": source_run_id,
        "provider": provider,
        "scorer": scorer,
        "model": model,
        "rubric_version": rubric_version,
        "run_id": run_id,
        "trigger_reason": trigger_reason,
        "scored_at": scored_at,
        "shadow_only": True,
        "affects_ranking": False,
        "score": normalized_score,
    }, None


def build_failure_record(
    *,
    bucket_id: str,
    source_digest: str,
    source_kind: str,
    source_run_id: str,
    provider: str,
    scorer: str,
    model: str,
    rubric_version: str,
    run_id: str,
    trigger_reason: str,
    category: str,
    retryable: bool,
    scored_at: str | None = None,
):
    """Build a failure row without storing model output or memory content."""
    bucket_id = str(bucket_id or "").strip()[:160]
    source_digest = str(source_digest or "").strip().lower()[:64]
    source_kind = _required_machine_text(source_kind, "source_kind")
    source_run_id = _required_machine_text(source_run_id, "source_run_id")
    provider = str(provider or "").strip()[:160]
    scorer = str(scorer or "").strip()[:160]
    model = str(model or "").strip()[:160]
    rubric_version = str(rubric_version or "").strip()[:160]
    run_id = str(run_id or "").strip()[:160]
    trigger_reason = str(trigger_reason or "").strip()[:160]
    category = str(category or "unknown").strip()[:160]
    if type(retryable) is not bool:
        raise ValueError("retryable must be a boolean")
    if scored_at is None:
        scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        scored_at = _normalized_timestamp(scored_at)
    return {
        "contract_version": CONTRACT_VERSION,
        "annotation_key": _failure_key(
            bucket_id,
            source_digest,
            source_kind,
            source_run_id,
            provider,
            scorer,
            model,
            rubric_version,
            category,
            retryable,
            run_id,
        ),
        "status": "failed",
        "bucket_id": bucket_id,
        "source_digest": source_digest,
        "source_kind": source_kind,
        "source_run_id": source_run_id,
        "provider": provider,
        "scorer": scorer,
        "model": model,
        "rubric_version": rubric_version,
        "run_id": run_id,
        "trigger_reason": trigger_reason,
        "category": category,
        "retryable": retryable,
        "scored_at": scored_at,
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
        self._maintenance_barrier = MaintenanceBarrier(root)

    @staticmethod
    def _normalize_row(row: dict) -> dict:
        if type(row) is not dict:
            raise ValueError("row must be an object")
        status = row.get("status")
        expected_fields = (
            _SUCCESS_ROW_FIELDS if status == "success"
            else _FAILURE_ROW_FIELDS if status == "failed"
            else None
        )
        if expected_fields is None or set(row) != expected_fields:
            raise ValueError("row fields do not match its status")
        if type(row.get("contract_version")) is not int \
                or row["contract_version"] != CONTRACT_VERSION:
            raise ValueError("unsupported contract version")
        if row.get("shadow_only") is not True \
                or row.get("affects_ranking") is not False:
            raise ValueError("row is not permanently shadow-only")
        annotation_key = row.get("annotation_key")
        source_digest = row.get("source_digest")
        if type(annotation_key) is not str or not _SHA256_RE.fullmatch(annotation_key):
            raise ValueError("invalid annotation key")
        if type(source_digest) is not str or not _SHA256_RE.fullmatch(source_digest):
            raise ValueError("invalid source digest")
        bucket_id = _required_text(row.get("bucket_id"), "bucket_id")
        source_kind = _required_machine_text(
            row.get("source_kind"),
            "source_kind",
        )
        source_run_id = _required_machine_text(
            row.get("source_run_id"),
            "source_run_id",
        )
        provider = _required_text(row.get("provider"), "provider")
        scorer = _required_text(row.get("scorer"), "scorer")
        model = _required_text(row.get("model"), "model")
        rubric_version = _required_text(
            row.get("rubric_version"),
            "rubric_version",
        )
        run_id = _required_text(row.get("run_id"), "run_id")
        trigger_reason = _required_text(
            row.get("trigger_reason"),
            "trigger_reason",
        )
        scored_at = _normalized_timestamp(row.get("scored_at"))

        if status == "success":
            normalized_score, error = validate_shadow_score(
                row.get("score"),
                min_confidence=0.0,
            )
            if error or normalized_score is None:
                raise ValueError("invalid shadow score")
            expected_key = _success_key(
                bucket_id,
                source_digest,
                source_kind,
                source_run_id,
                provider,
                scorer,
                model,
                rubric_version,
            )
        else:
            category = _required_text(row.get("category"), "category")
            retryable = row.get("retryable")
            if type(retryable) is not bool:
                raise ValueError("retryable must be a boolean")
            expected_key = _failure_key(
                bucket_id,
                source_digest,
                source_kind,
                source_run_id,
                provider,
                scorer,
                model,
                rubric_version,
                category,
                retryable,
                run_id,
            )
        if annotation_key != expected_key:
            raise ValueError("annotation key does not match row identity")

        normalized = {
            "contract_version": CONTRACT_VERSION,
            "annotation_key": annotation_key,
            "status": status,
            "bucket_id": bucket_id,
            "source_digest": source_digest,
            "source_kind": source_kind,
            "source_run_id": source_run_id,
            "provider": provider,
            "scorer": scorer,
            "model": model,
            "rubric_version": rubric_version,
            "run_id": run_id,
            "trigger_reason": trigger_reason,
            "scored_at": scored_at,
            "shadow_only": True,
            "affects_ranking": False,
        }
        if status == "success":
            normalized["score"] = normalized_score
        else:
            normalized["category"] = category
            normalized["retryable"] = retryable
        return normalized

    @staticmethod
    def _same_logical_row(left: dict, right: dict) -> bool:
        """Ignore per-attempt timestamps while preserving score immutability."""
        left = dict(left)
        right = dict(right)
        for row in (left, right):
            row.pop("run_id", None)
            row.pop("scored_at", None)
        return left == right

    @staticmethod
    def _validate_row(row: dict) -> bool:
        try:
            EAxisShadowStore._normalize_row(row)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _load_locked(handle) -> list[dict]:
        handle.seek(0)
        rows = []
        rows_by_key: dict[str, dict] = {}
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = strict_json_loads(raw)
            except (UnicodeDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"corrupt E shadow ledger at line {line_number}"
                ) from exc
            try:
                normalized = EAxisShadowStore._normalize_row(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid E shadow ledger row at line {line_number}"
                ) from exc
            annotation_key = normalized["annotation_key"]
            existing = rows_by_key.get(annotation_key)
            if existing is not None:
                detail = (
                    "duplicate" if existing == normalized else "conflicting"
                )
                raise ValueError(
                    "corrupt E shadow ledger: "
                    f"{detail} annotation_key at line {line_number}"
                )
            rows_by_key[annotation_key] = normalized
            rows.append(normalized)
        return rows

    def load(self) -> list[dict]:
        try:
            with secure_e_axis_lock(self.lock_path):
                try:
                    self.path.lstat()
                except FileNotFoundError:
                    return []
                except OSError as exc:
                    raise ValueError(
                        "E shadow ledger is unavailable"
                    ) from exc
                with open_secure_e_axis_jsonl(self.path) as handle:
                    return self._load_locked(handle)
        except EAxisStorageError as exc:
            raise ValueError("unsafe E shadow storage") from exc

    def append(self, row: dict) -> bool:
        with self._maintenance_barrier.shared():
            return self._append_locked(row)

    def _append_locked(self, row: dict) -> bool:
        if type(row) is not dict:
            raise ValueError("invalid E shadow row")
        key = row.get("annotation_key")
        if type(key) is not str or not _SHA256_RE.fullmatch(key):
            raise ValueError("invalid E shadow row")
        with secure_e_axis_lock(self.lock_path):
            try:
                with open_secure_e_axis_jsonl(self.path) as handle:
                    rows = self._load_locked(handle)
                    existing = next(
                        (
                            item
                            for item in rows
                            if item.get("annotation_key") == key
                        ),
                        None,
                    )
                    if existing is not None:
                        try:
                            normalized = self._normalize_row(row)
                        except (TypeError, ValueError) as exc:
                            raise EAxisShadowConflictError(
                                "E shadow annotation_key conflict"
                            ) from exc
                        if self._same_logical_row(existing, normalized):
                            return False
                        raise EAxisShadowConflictError(
                            "E shadow annotation_key conflict"
                        )
                    try:
                        normalized = self._normalize_row(row)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("invalid E shadow row") from exc
                    handle.seek(0, os.SEEK_END)
                    handle.write(
                        json.dumps(
                            normalized,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                    return True
            except EAxisStorageError as exc:
                raise ValueError("unsafe E shadow storage") from exc
