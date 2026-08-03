"""Bounded, ranking-neutral E-axis shadow collection.

This process consumes only validated, already-redacted LMC-5 candidate rows.
It never reads raw events, never changes memory buckets, and never participates
in recall. Scores, attempts, and aggregate coverage are separate append-only
sidecars under <buckets_dir>/.axis.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from e_axis_shadow import (
    EAxisShadowConflictError,
    EAxisShadowStore,
    build_failure_record,
    build_shadow_annotation,
    normalize_min_confidence,
    strict_json_loads,
)
from e_axis_storage import (
    EAxisStorageBusy,
    EAxisStorageError,
    open_secure_e_axis_jsonl,
    secure_e_axis_lock,
)
from e_axis_curated_reader import (
    EAxisCuratedError,
    iter_curated_subjects,
)
from e_axis_trigger import (
    EAxisSourceError,
    EAxisSourceScan,
    EAxisSubject,
    iter_candidate_subjects,
)
from lmc5_candidate_reader import (
    ReadOnlyLMC5CandidateLedger,
    ReadOnlyLedgerError,
)
from maintenance_barrier import MaintenanceBarrier
from night_run_runtime import NightRunRuntimeError, OpenAIChatProvider
from redact import redact_text
from utils import load_config


SCORER_NAME = "ombre-e-shadow-v2"
RUBRIC_VERSION = "lmc5-experience-20260731-v1"
DEFAULT_MAX_PER_RUN = 20
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_CONTENT_CHARS = 12_000
SHANGHAI = ZoneInfo("Asia/Shanghai")
_SCORE_FIELDS = frozenset({
    "valence",
    "arousal",
    "tension",
    "confidence",
    "response_tendency",
    "growth_delta",
})
FORMAL_SOURCE_KINDS = frozenset({"lmc5_candidate", "curated_memory"})


class EAxisNightError(RuntimeError):
    """A bounded machine-readable E shadow job failure."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class EAxisNightResult:
    run_id: str
    natural_date: str
    scanned: int
    eligible: int
    skipped: int
    attempted: int
    added: int
    existing_success: int
    existing_terminal: int
    failed_retryable: int
    failed_terminal: int
    remaining: int
    observed_natural_days: int
    promotion_eligible: bool
    skip_reasons: dict[str, int]
    distribution: dict[str, Any]

    @property
    def existing(self) -> int:
        return self.existing_success + self.existing_terminal

    @property
    def failed(self) -> int:
        return self.failed_retryable + self.failed_terminal


def _result_is_healthy(result: EAxisNightResult) -> bool:
    """Keep unresolved terminal failures visible across later night runs."""
    return (
        result.eligible > 0
        and result.failed == 0
        and result.existing_terminal == 0
    )


def _plain_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    if type(value) is not int or not minimum <= value <= maximum:
        raise EAxisNightError("config.integer_invalid")
    return value


def _plain_finite(value: object, *, default: float) -> float:
    if value is None:
        return default
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise EAxisNightError("config.number_invalid")
    normalized = float(value)
    if not 0.0 <= normalized <= 2.0:
        raise EAxisNightError("config.number_invalid")
    return normalized


def _scorer_lineage_name(
    *,
    provider_name: str,
    base_url: str,
    model: str,
    rubric_version: str,
    max_tokens: int,
    max_content_chars: int,
    min_confidence: float,
    temperature: float,
) -> str:
    payload = {
        "base_url": base_url,
        "max_content_chars": max_content_chars,
        "max_tokens": max_tokens,
        "min_confidence": min_confidence,
        "model": model,
        "prompt_contract": SCORER_NAME,
        "provider": provider_name,
        "response_format": "json_object",
        "rubric_version": rubric_version,
        "temperature": temperature,
        "thinking": "disabled",
        "timeout_seconds": 75.0,
    }
    digest = hashlib.sha256(json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return f"{SCORER_NAME}:{digest[:16]}"


def _required_text(value: object, code: str, *, maximum: int = 160) -> str:
    if type(value) is not str:
        raise EAxisNightError(code)
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise EAxisNightError(code)
    return normalized


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_run_id(now: datetime | None = None) -> str:
    current = now or _now()
    return "e-shadow-" + current.strftime("%Y%m%dT%H%M%S%fZ")


def _natural_date(timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError) as exc:
        raise EAxisNightError("report.timestamp_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EAxisNightError("report.timestamp_invalid")
    return parsed.astimezone(SHANGHAI).date().isoformat()


class StrictEAxisScorer:
    """Strict JSON-only E scorer around a provider-agnostic callable."""

    def __init__(
        self,
        provider: Callable[[str], dict[str, Any]],
        *,
        provider_name: str,
        model: str,
        scorer_name: str = SCORER_NAME,
        rubric_version: str = RUBRIC_VERSION,
        min_confidence: float = 0.3,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        self.provider_name = _required_text(
            provider_name,
            "config.provider_name_invalid",
        )
        self.model = _required_text(model, "config.model_invalid")
        self.scorer_name = _required_text(
            scorer_name,
            "config.scorer_invalid",
        )
        self.rubric_version = _required_text(
            rubric_version,
            "config.rubric_invalid",
        )
        normalized_confidence = normalize_min_confidence(min_confidence)
        if normalized_confidence is None:
            raise EAxisNightError("config.min_confidence_invalid")
        self.max_content_chars = _plain_int(
            max_content_chars,
            default=DEFAULT_MAX_CONTENT_CHARS,
            minimum=1_000,
            maximum=50_000,
        )
        self.min_confidence = normalized_confidence
        self.provider = provider

    def _prompt(self, subject: EAxisSubject) -> str:
        if len(subject.content) > self.max_content_chars:
            raise EAxisNightError("source.too_long")
        payload = {
            "memory_type": subject.memory_type,
            "title": redact_text(subject.title),
            "content": redact_text(subject.content),
            "trigger_reason": subject.trigger_reason,
            "output_schema": {
                "valence": "finite number -1..1",
                "arousal": "finite number 0..1",
                "tension": "finite number 0..1",
                "confidence": "finite number 0..1",
                "response_tendency": "comfort|engage|withdraw|alert",
                "growth_delta": "growth|stable|setback",
            },
        }
        rules = (
            "Score only emotional qualities explicitly supported by this "
            "memory. Do not infer hidden motives or new facts. Return exactly "
            "one JSON object with exactly the six output_schema keys, without "
            "markdown or explanation. Use low confidence when ambiguous."
        )
        return rules + "\nINPUT=" + json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _content(envelope: object) -> str:
        if type(envelope) is not dict:
            raise EAxisNightError(
                "provider.invalid_envelope",
                retryable=True,
            )
        choices = envelope.get("choices")
        if type(choices) is not list or len(choices) != 1:
            raise EAxisNightError(
                "provider.invalid_choices",
                retryable=True,
            )
        choice = choices[0]
        if type(choice) is not dict:
            raise EAxisNightError(
                "provider.invalid_choice",
                retryable=True,
            )
        if choice.get("finish_reason") != "stop":
            raise EAxisNightError("provider.incomplete", retryable=True)
        message = choice.get("message")
        if type(message) is not dict or type(message.get("content")) is not str:
            raise EAxisNightError(
                "provider.invalid_message",
                retryable=True,
            )
        content = message["content"].strip()
        if not content:
            raise EAxisNightError("provider.empty", retryable=True)
        return content

    def score(self, subject: EAxisSubject) -> dict[str, Any]:
        try:
            envelope = self.provider(self._prompt(subject))
        except EAxisNightError:
            raise
        except Exception as exc:
            code = (
                "provider.timeout"
                if isinstance(exc, TimeoutError)
                or "timeout" in type(exc).__name__.lower()
                else "provider.transport"
            )
            raise EAxisNightError(code, retryable=True) from exc
        try:
            parsed = strict_json_loads(self._content(envelope))
        except EAxisNightError:
            raise
        except Exception as exc:
            raise EAxisNightError(
                "provider.invalid_json",
                retryable=True,
            ) from exc
        if type(parsed) is not dict or set(parsed) != _SCORE_FIELDS:
            raise EAxisNightError("schema.fields")
        annotation, error = build_shadow_annotation(
            bucket_id="validation-only",
            source_digest="0" * 64,
            source_kind=subject.source_kind,
            source_run_id=subject.source_run_id,
            provider=self.provider_name,
            scorer=self.scorer_name,
            model=self.model,
            rubric_version=self.rubric_version,
            run_id="validation-only",
            trigger_reason=subject.trigger_reason,
            score=parsed,
            min_confidence=self.min_confidence,
        )
        if annotation is None or error:
            raise EAxisNightError(error or "schema.invalid")
        return dict(annotation["score"])


class EAxisRunJournal:
    """Private append-only attempt and aggregate report journals."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        maintenance_root: str | os.PathLike[str],
    ) -> None:
        self.root = Path(root)
        self.attempts_path = self.root / "e-shadow-attempts.jsonl"
        self.reports_path = self.root / "e-shadow-coverage.jsonl"
        self._barrier = MaintenanceBarrier(maintenance_root)

    @staticmethod
    def _load_locked(handle, *, key_field: str) -> dict[str, dict]:
        handle.seek(0)
        rows: dict[str, dict] = {}
        for raw in handle:
            if not raw.strip():
                continue
            try:
                row = strict_json_loads(raw)
            except Exception as exc:
                raise EAxisNightError("journal.corrupt") from exc
            if type(row) is not dict or type(row.get(key_field)) is not str:
                raise EAxisNightError("journal.corrupt")
            key = row[key_field]
            if key in rows:
                raise EAxisNightError("journal.duplicate")
            rows[key] = row
        return rows

    def _append(self, path: Path, key_field: str, row: dict[str, Any]) -> bool:
        key = row.get(key_field)
        if type(key) is not str or not key:
            raise EAxisNightError("journal.key_invalid")
        lock_path = Path(f"{path}.lock")
        with self._barrier.shared():
            with secure_e_axis_lock(lock_path):
                with open_secure_e_axis_jsonl(path) as handle:
                    rows = self._load_locked(handle, key_field=key_field)
                    existing = rows.get(key)
                    if existing is not None:
                        if existing == row:
                            return False
                        raise EAxisNightError("journal.conflict")
                    handle.seek(0, os.SEEK_END)
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
        return True

    def append_attempt(self, row: dict[str, Any]) -> bool:
        return self._append(self.attempts_path, "attempt_key", row)

    def append_report(self, row: dict[str, Any]) -> bool:
        return self._append(self.reports_path, "report_key", row)


def _identity(
    source: EAxisSubject,
    scorer: StrictEAxisScorer,
) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        source.source_id,
        source.source_digest,
        source.source_kind,
        source.source_run_id,
        scorer.provider_name,
        scorer.scorer_name,
        scorer.model,
        scorer.rubric_version,
    )


def _attempt_key(
    *,
    run_id: str,
    source: EAxisSubject,
    scorer: StrictEAxisScorer,
) -> str:
    material = "\x1f".join((
        run_id,
        source.source_kind,
        source.source_id,
        source.source_digest,
        scorer.provider_name,
        scorer.scorer_name,
        scorer.model,
        scorer.rubric_version,
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _attempt_row(
    *,
    run_id: str,
    source: EAxisSubject,
    scorer: StrictEAxisScorer,
    status: str,
    error_code: str | None,
    retryable: bool,
    recorded_at: str,
) -> dict[str, Any]:
    try:
        parsed_at = datetime.fromisoformat(recorded_at)
    except (TypeError, ValueError) as exc:
        raise EAxisNightError("journal.timestamp_invalid") from exc
    if parsed_at.tzinfo is None or parsed_at.utcoffset() is None:
        raise EAxisNightError("journal.timestamp_invalid")
    recorded_at = parsed_at.isoformat(timespec="seconds")
    source_ref_digest = hashlib.sha256(
        f"{source.source_kind}\x1f{source.source_id}".encode("utf-8")
    ).hexdigest()
    return {
        "contract_version": 1,
        "attempt_key": _attempt_key(
            run_id=run_id,
            source=source,
            scorer=scorer,
        ),
        "run_id": run_id,
        "natural_date": _natural_date(recorded_at),
        "source_kind": source.source_kind,
        "source_ref_digest": source_ref_digest,
        "source_digest": source.source_digest,
        "source_run_id": source.source_run_id,
        "memory_type": source.memory_type,
        "trigger_reason": source.trigger_reason,
        "provider": scorer.provider_name,
        "scorer": scorer.scorer_name,
        "model": scorer.model,
        "rubric_version": scorer.rubric_version,
        "status": status,
        "error_code": error_code,
        "retryable": retryable,
        "recorded_at": recorded_at,
        "shadow_only": True,
        "affects_ranking": False,
    }


def _load_shadow_rows(store: EAxisShadowStore) -> list[dict[str, Any]]:
    try:
        return store.load()
    except (EAxisStorageError, OSError, TypeError, ValueError) as exc:
        raise EAxisNightError("shadow_ledger.unavailable") from exc


def _append_shadow_row(
    store: EAxisShadowStore,
    row: dict[str, Any],
) -> bool:
    try:
        return store.append(row)
    except EAxisShadowConflictError:
        raise
    except (EAxisStorageError, OSError, TypeError, ValueError) as exc:
        raise EAxisNightError("shadow_ledger.unavailable") from exc


def _append_attempt(
    journal: EAxisRunJournal,
    row: dict[str, Any],
) -> bool:
    try:
        return journal.append_attempt(row)
    except EAxisNightError:
        raise
    except (EAxisStorageError, OSError, TypeError, ValueError) as exc:
        raise EAxisNightError("journal.unavailable") from exc


def _append_report(
    journal: EAxisRunJournal,
    row: dict[str, Any],
) -> bool:
    try:
        return journal.append_report(row)
    except EAxisNightError:
        raise
    except (EAxisStorageError, OSError, TypeError, ValueError) as exc:
        raise EAxisNightError("journal.unavailable") from exc


def _record_source_failure(
    *,
    store: EAxisShadowStore,
    journal: EAxisRunJournal,
    scorer: StrictEAxisScorer,
    source: EAxisSubject,
    run_id: str,
    recorded_at: str,
    error: EAxisNightError,
) -> None:
    failure = build_failure_record(
        bucket_id=source.source_id,
        source_digest=source.source_digest,
        source_kind=source.source_kind,
        source_run_id=source.source_run_id,
        provider=scorer.provider_name,
        scorer=scorer.scorer_name,
        model=scorer.model,
        rubric_version=scorer.rubric_version,
        run_id=run_id,
        trigger_reason=source.trigger_reason,
        category=error.code,
        retryable=error.retryable,
        scored_at=recorded_at,
    )
    _append_shadow_row(store, failure)
    _append_attempt(journal, _attempt_row(
        run_id=run_id,
        source=source,
        scorer=scorer,
        status="failed",
        error_code=error.code,
        retryable=error.retryable,
        recorded_at=recorded_at,
    ))


def _matching_shadow_rows(
    rows: list[dict[str, Any]],
    source: EAxisSubject,
    scorer: StrictEAxisScorer,
) -> list[dict[str, Any]]:
    identity = _identity(source, scorer)
    return [
        row
        for row in rows
        if (
            str(row.get("bucket_id") or ""),
            str(row.get("source_digest") or ""),
            str(row.get("source_kind") or ""),
            str(row.get("source_run_id") or ""),
            str(row.get("provider") or ""),
            str(row.get("scorer") or ""),
            str(row.get("model") or ""),
            str(row.get("rubric_version") or ""),
        ) == identity
    ]


def _reconcile_attempts(
    *,
    journal: EAxisRunJournal,
    source: EAxisSubject,
    scorer: StrictEAxisScorer,
    rows: list[dict[str, Any]],
) -> None:
    """Repair a crash gap between the score ledger and attempt journal."""
    for row in rows:
        failed = row.get("status") == "failed"
        _append_attempt(journal, _attempt_row(
            run_id=str(row["run_id"]),
            source=source,
            scorer=scorer,
            status="failed" if failed else "success",
            error_code=str(row["category"]) if failed else None,
            retryable=bool(row["retryable"]) if failed else False,
            recorded_at=str(row["scored_at"]),
        ))


def _terminal_status(rows: list[dict[str, Any]]) -> str | None:
    if any(row.get("status") == "success" for row in rows):
        return "success"
    if any(
        row.get("status") == "failed" and row.get("retryable") is False
        for row in rows
    ):
        return "terminal"
    return None


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(
    rows: list[dict[str, Any]],
    scorer: StrictEAxisScorer,
) -> tuple[dict[str, Any], int]:
    cohort = [
        row
        for row in rows
        if row.get("source_kind") in FORMAL_SOURCE_KINDS
        and row.get("provider") == scorer.provider_name
        and row.get("scorer") == scorer.scorer_name
        and row.get("model") == scorer.model
        and row.get("rubric_version") == scorer.rubric_version
    ]
    successes = [row for row in cohort if row.get("status") == "success"]

    def summarize(success_rows: list[dict[str, Any]]) -> dict[str, Any]:
        numeric: dict[str, Any] = {}
        for field in ("valence", "arousal", "tension", "confidence"):
            values = [float(row["score"][field]) for row in success_rows]
            if not values:
                numeric[field] = {"count": 0}
                continue
            numeric[field] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "p05": _percentile(values, 0.05),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
        enums = {
            field: dict(sorted(Counter(
                str(row["score"][field]) for row in success_rows
            ).items()))
            for field in ("response_tendency", "growth_delta")
        }
        return {"numeric": numeric, "enum": enums}

    failures = [row for row in cohort if row.get("status") == "failed"]
    distribution = summarize(successes)
    distribution["by_source_kind"] = {
        source_kind: summarize([
            row for row in successes if row.get("source_kind") == source_kind
        ])
        for source_kind in sorted(FORMAL_SOURCE_KINDS)
    }
    distribution["failures"] = {
        "count": len(failures),
        "by_code": dict(sorted(Counter(
            str(row.get("category") or "failure.unknown")
            for row in failures
        ).items())),
        "by_retryability": {
            "retryable": sum(
                row.get("retryable") is True for row in failures
            ),
            "terminal": sum(
                row.get("retryable") is False for row in failures
            ),
        },
    }
    days = {
        _natural_date(str(row["scored_at"]))
        for row in successes
    }
    return distribution, len(days)


def _merge_source_scans(
    scans: tuple[tuple[str, EAxisSourceScan], ...],
) -> EAxisSourceScan:
    subjects: list[EAxisSubject] = []
    seen: dict[tuple[str, str], EAxisSubject] = {}
    skip_reasons: Counter[str] = Counter()
    scanned = skipped = 0
    for source_kind, scan in scans:
        scanned += scan.scanned
        skipped += scan.skipped
        for reason, count in scan.skip_reasons:
            skip_reasons[f"{source_kind}:{reason}"] += count
        for subject in scan.subjects:
            key = (subject.source_kind, subject.source_id)
            existing = seen.get(key)
            if existing is not None and existing != subject:
                raise EAxisNightError("source.duplicate_conflict")
            if existing is None:
                seen[key] = subject
                subjects.append(subject)
    return EAxisSourceScan(
        subjects=tuple(sorted(
            subjects,
            key=lambda item: (item.created_at, item.source_kind, item.source_id),
        )),
        scanned=scanned,
        skipped=skipped,
        skip_reasons=tuple(sorted(skip_reasons.items())),
    )


async def run_e_axis_shadow(
    *,
    ledger: Any,
    store: EAxisShadowStore,
    journal: EAxisRunJournal,
    scorer: StrictEAxisScorer,
    run_id: str,
    curated_buckets_dir: str | os.PathLike[str] | None = None,
    legacy_naive_timestamps_utc: bool = False,
    max_per_run: int = DEFAULT_MAX_PER_RUN,
    clock: Callable[[], datetime] = _now,
) -> EAxisNightResult:
    """Collect one bounded cohort without mutating the source ledger."""

    run_id = _required_text(run_id, "run.id_invalid")
    if type(legacy_naive_timestamps_utc) is not bool:
        raise EAxisNightError("config.legacy_naive_timestamps_utc_invalid")
    max_per_run = _plain_int(
        max_per_run,
        default=DEFAULT_MAX_PER_RUN,
        minimum=1,
        maximum=100,
    )
    try:
        scans: list[tuple[str, EAxisSourceScan]] = [
            ("lmc5_candidate", iter_candidate_subjects(ledger)),
        ]
        if curated_buckets_dir is not None:
            scans.append((
                "curated_memory",
                iter_curated_subjects(
                    Path(curated_buckets_dir),
                    legacy_naive_timestamps_utc=(
                        legacy_naive_timestamps_utc
                    ),
                ),
            ))
        source_scan = _merge_source_scans(tuple(scans))
    except (EAxisSourceError, EAxisCuratedError) as exc:
        raise EAxisNightError(str(exc)) from exc
    subjects = source_scan.subjects
    scanned = source_scan.scanned
    skipped = source_scan.skipped
    if scanned == 0:
        raise EAxisNightError("source.empty")

    rows_before = _load_shadow_rows(store)
    pending: list[EAxisSubject] = []
    existing_success = 0
    existing_terminal = 0
    for source in subjects:
        source_rows = _matching_shadow_rows(rows_before, source, scorer)
        _reconcile_attempts(
            journal=journal,
            source=source,
            scorer=scorer,
            rows=source_rows,
        )
        status = _terminal_status(source_rows)
        if status == "success":
            existing_success += 1
        elif status == "terminal":
            existing_terminal += 1
        else:
            pending.append(source)

    added = failed_retryable = failed_terminal = 0
    for source in pending[:max_per_run]:
        recorded_at = clock().astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        )
        try:
            score = scorer.score(source)
            row, error = build_shadow_annotation(
                bucket_id=source.source_id,
                source_digest=source.source_digest,
                source_kind=source.source_kind,
                source_run_id=source.source_run_id,
                provider=scorer.provider_name,
                scorer=scorer.scorer_name,
                model=scorer.model,
                rubric_version=scorer.rubric_version,
                run_id=run_id,
                trigger_reason=source.trigger_reason,
                score=score,
                scored_at=recorded_at,
                min_confidence=scorer.min_confidence,
            )
            if row is None or error:
                raise EAxisNightError(error or "schema.invalid")
        except EAxisNightError as exc:
            if exc.retryable:
                failed_retryable += 1
            else:
                failed_terminal += 1
            _record_source_failure(
                store=store,
                journal=journal,
                scorer=scorer,
                source=source,
                run_id=run_id,
                recorded_at=recorded_at,
                error=exc,
            )
            continue

        try:
            inserted = _append_shadow_row(store, row)
        except EAxisShadowConflictError:
            conflict = EAxisNightError("append.conflict")
            failed_terminal += 1
            _record_source_failure(
                store=store,
                journal=journal,
                scorer=scorer,
                source=source,
                run_id=run_id,
                recorded_at=recorded_at,
                error=conflict,
            )
            continue

        if inserted:
            added += 1
        else:
            existing_success += 1
        _append_attempt(journal, _attempt_row(
            run_id=run_id,
            source=source,
            scorer=scorer,
            status="success",
            error_code=None,
            retryable=False,
            recorded_at=recorded_at,
        ))

    rows_after = _load_shadow_rows(store)
    distribution, observed_days = _distribution(rows_after, scorer)
    attempted = min(len(pending), max_per_run)
    remaining = max(0, len(pending) - attempted) + failed_retryable
    now = clock().astimezone(timezone.utc)
    natural_date = now.astimezone(SHANGHAI).date().isoformat()
    result = EAxisNightResult(
        run_id=run_id,
        natural_date=natural_date,
        scanned=scanned,
        eligible=len(subjects),
        skipped=skipped,
        attempted=attempted,
        added=added,
        existing_success=existing_success,
        existing_terminal=existing_terminal,
        failed_retryable=failed_retryable,
        failed_terminal=failed_terminal,
        remaining=remaining,
        observed_natural_days=observed_days,
        promotion_eligible=False,
        skip_reasons=source_scan.skip_reason_counts(),
        distribution=distribution,
    )
    report_payload = asdict(result)
    scored_total = existing_success + added
    terminal_failure_total = existing_terminal + failed_terminal
    unresolved_total = max(
        0,
        len(subjects) - scored_total - terminal_failure_total,
    )
    denominator = len(subjects)
    report_payload.update({
        "contract_version": 1,
        "report_key": hashlib.sha256(
            f"{run_id}\x1f{scorer.provider_name}\x1f"
            f"{scorer.model}\x1f{scorer.rubric_version}".encode("utf-8")
        ).hexdigest(),
        "recorded_at": now.isoformat(timespec="microseconds"),
        "provider": scorer.provider_name,
        "scorer": scorer.scorer_name,
        "model": scorer.model,
        "rubric_version": scorer.rubric_version,
        "by_source_kind": dict(sorted(Counter(
            source.source_kind for source in subjects
        ).items())),
        "by_memory_type": dict(sorted(Counter(
            source.memory_type for source in subjects
        ).items())),
        "by_trigger_reason": dict(sorted(Counter(
            source.trigger_reason for source in subjects
        ).items())),
        "coverage": {
            "eligible": denominator,
            "scored": scored_total,
            "terminal_failure": terminal_failure_total,
            "unresolved": unresolved_total,
            "score_rate": (
                scored_total / denominator if denominator else None
            ),
            "resolved_rate": (
                (scored_total + terminal_failure_total) / denominator
                if denominator
                else None
            ),
            "denominator_zero": denominator == 0,
        },
        "promotion_guards": {
            "minimum_natural_days": 30,
            "coverage_stable": False,
            "distribution_stable": False,
            "provider_calibrated": False,
            "real_query_validation": False,
            "human_approved": False,
        },
        "shadow_only": True,
        "affects_ranking": False,
        "curated_timestamp_policy": (
            "legacy_naive_utc"
            if legacy_naive_timestamps_utc
            else "aware_only"
        ),
    })
    _append_report(journal, report_payload)
    return result


def build_e_axis_runtime(
    config: dict[str, Any],
) -> tuple[
    ReadOnlyLMC5CandidateLedger,
    EAxisShadowStore,
    EAxisRunJournal,
    StrictEAxisScorer,
    int,
    Path,
    bool,
]:
    section = config.get("e_axis_shadow", {}) or {}
    if type(section) is not dict:
        raise EAxisNightError("config.section_invalid")
    if section.get("enabled") is not True:
        raise EAxisNightError("config.disabled")
    legacy_naive_timestamps_utc = section.get(
        "legacy_naive_timestamps_utc",
        False,
    )
    if type(legacy_naive_timestamps_utc) is not bool:
        raise EAxisNightError("config.legacy_naive_timestamps_utc_invalid")
    max_per_run = _plain_int(
        section.get("max_per_run"),
        default=DEFAULT_MAX_PER_RUN,
        minimum=1,
        maximum=100,
    )
    max_tokens = _plain_int(
        section.get("max_tokens"),
        default=DEFAULT_MAX_TOKENS,
        minimum=512,
        maximum=2_048,
    )
    min_confidence = normalize_min_confidence(
        section.get("min_confidence", 0.3)
    )
    if min_confidence is None:
        raise EAxisNightError("config.min_confidence_invalid")

    dehydration = config.get("dehydration", {}) or {}
    if type(dehydration) is not dict:
        raise EAxisNightError("config.provider_invalid")
    provider_name = _required_text(
        section.get("provider_name"),
        "config.provider_name_invalid",
    )
    model = _required_text(
        section.get("model") or dehydration.get("model") or "deepseek-chat",
        "config.model_invalid",
    )
    base_url = _required_text(
        section.get("base_url")
        or dehydration.get("base_url")
        or "https://api.deepseek.com/v1",
        "config.base_url_invalid",
        maximum=2_048,
    )
    rubric_version = _required_text(
        section.get("rubric_version", RUBRIC_VERSION),
        "config.rubric_invalid",
    )
    temperature = _plain_finite(
        section.get("temperature"),
        default=0.0,
    )
    max_content_chars = _plain_int(
        section.get("max_content_chars"),
        default=DEFAULT_MAX_CONTENT_CHARS,
        minimum=1_000,
        maximum=50_000,
    )
    scorer_name = _scorer_lineage_name(
        provider_name=provider_name,
        base_url=base_url,
        model=model,
        rubric_version=rubric_version,
        max_tokens=max_tokens,
        max_content_chars=max_content_chars,
        min_confidence=min_confidence,
        temperature=temperature,
    )
    try:
        provider = OpenAIChatProvider(
            api_key=str(
                section.get("api_key")
                or dehydration.get("api_key")
                or ""
            ),
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=75.0,
            disable_thinking=True,
            json_object=True,
        )
    except NightRunRuntimeError as exc:
        raise EAxisNightError(exc.code) from exc
    root = Path(config["buckets_dir"])
    try:
        ledger = ReadOnlyLMC5CandidateLedger(
            root / ".lmc5" / "pipeline.sqlite3"
        )
    except ReadOnlyLedgerError as exc:
        raise EAxisNightError(str(exc)) from exc
    store = EAxisShadowStore(
        root / ".axis" / "e-shadow.jsonl",
        maintenance_root=root,
    )
    journal = EAxisRunJournal(
        root / ".axis",
        maintenance_root=root,
    )
    scorer = StrictEAxisScorer(
        provider,
        provider_name=provider_name,
        model=model,
        scorer_name=scorer_name,
        rubric_version=rubric_version,
        min_confidence=min_confidence,
        max_content_chars=max_content_chars,
    )
    return (
        ledger,
        store,
        journal,
        scorer,
        max_per_run,
        root,
        legacy_naive_timestamps_utc,
    )


def main() -> int:
    try:
        config = load_config()
        (
            ledger,
            store,
            journal,
            scorer,
            max_per_run,
            root,
            legacy_naive_timestamps_utc,
        ) = build_e_axis_runtime(config)
        with secure_e_axis_lock(
            root / ".axis" / "e-shadow-run.lock",
            blocking=False,
        ):
            result = asyncio.run(run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=scorer,
                run_id=_new_run_id(),
                curated_buckets_dir=root,
                legacy_naive_timestamps_utc=legacy_naive_timestamps_utc,
                max_per_run=max_per_run,
            ))
    except EAxisStorageBusy:
        print('{"ok":false,"code":"run.busy"}')
        return 75
    except EAxisStorageError:
        print('{"ok":false,"code":"run.lock_unavailable"}')
        return 1
    except EAxisNightError as exc:
        print(json.dumps(
            {"ok": False, "code": exc.code},
            sort_keys=True,
            separators=(",", ":"),
        ))
        return 1
    payload = asdict(result)
    payload["existing"] = result.existing
    payload["failed"] = result.failed
    payload["ok"] = _result_is_healthy(result)
    if result.eligible == 0:
        payload["code"] = "source.no_eligible"
    print(json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
