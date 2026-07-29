"""Bounded daily E-axis shadow scoring for LMC-5 night memories.

Only buckets produced by the conservative LMC-5 X writer are eligible.  The
job never scans ordinary/private buckets into the external scorer, never
changes a bucket, and writes annotations only to ``.axis/e-shadow.jsonl``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from bucket_manager import BucketManager
from e_axis_shadow import (
    EAxisShadowStore,
    build_failure_record,
    build_shadow_annotation,
    normalize_min_confidence,
    strict_json_loads,
)
from night_run_runtime import OpenAIChatProvider
from utils import load_config


SCORER_NAME = "ombre-e-shadow-v1"
RUBRIC_VERSION = "experience-rubric-20260729-v1"
DEFAULT_MAX_PER_RUN = 20
DEFAULT_MAX_TOKENS = 512
MAX_CONTENT_CHARS = 2_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCORE_FIELDS = frozenset(
    {
        "valence",
        "arousal",
        "tension",
        "confidence",
        "response_tendency",
        "growth_delta",
    }
)


class EAxisNightError(RuntimeError):
    """A bounded machine-readable E shadow job failure."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class EAxisNightResult:
    eligible: int
    attempted: int
    added: int
    existing: int
    failed: int
    remaining: int


def _plain_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
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


def _source_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _eligible_lmc5_bucket(bucket: object) -> bool:
    if type(bucket) is not dict:
        return False
    bucket_id = bucket.get("id")
    content = bucket.get("content")
    metadata = bucket.get("metadata")
    if (
        type(bucket_id) is not str
        or not bucket_id.strip()
        or type(content) is not str
        or not content.strip()
        or type(metadata) is not dict
    ):
        return False
    tags = metadata.get("tags")
    provenance = metadata.get("x_provenance")
    curated_key = metadata.get("curated_write_key")
    curated_digest = metadata.get("curated_payload_sha256")
    if type(tags) is not list or not {"lmc5", "night", "event"}.issubset(
        {str(tag).strip() for tag in tags}
    ):
        return False
    return bool(
        type(curated_key) is str
        and curated_key.startswith("lmc5-x:v1:")
        and type(curated_digest) is str
        and _SHA256_RE.fullmatch(curated_digest)
        and metadata.get("vector_policy") == "required"
        and metadata.get("lmc5_recall_state") == "ready_vector"
        and type(provenance) is dict
        and provenance.get("source_kind") == "conversation"
        and type(provenance.get("source_session")) is str
        and provenance.get("source_session")
        and type(provenance.get("source_event_ids")) is list
        and provenance.get("source_event_ids")
        and type(provenance.get("source_digest")) is str
        and _SHA256_RE.fullmatch(provenance["source_digest"])
    )


def _bucket_sort_key(bucket: dict[str, Any]) -> tuple[str, str]:
    metadata = bucket.get("metadata") or {}
    return (
        str(metadata.get("created") or metadata.get("last_active") or ""),
        str(bucket.get("id") or ""),
    )


class StrictEAxisScorer:
    """Strict JSON-only E scorer around a synchronous provider callable."""

    def __init__(
        self,
        provider: Callable[[str], dict[str, Any]],
        *,
        model: str,
        scorer_name: str = SCORER_NAME,
        rubric_version: str = RUBRIC_VERSION,
        min_confidence: float = 0.3,
    ) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        if type(model) is not str or not model.strip():
            raise ValueError("model must be non-empty text")
        if type(scorer_name) is not str or not scorer_name.strip():
            raise ValueError("scorer_name must be non-empty text")
        if type(rubric_version) is not str or not rubric_version.strip():
            raise ValueError("rubric_version must be non-empty text")
        normalized_confidence = normalize_min_confidence(min_confidence)
        if normalized_confidence is None:
            raise EAxisNightError("config.min_confidence_invalid")
        self.provider = provider
        self.model = model.strip()
        self.scorer_name = scorer_name.strip()
        self.rubric_version = rubric_version.strip()
        self.min_confidence = normalized_confidence

    @staticmethod
    def _prompt(title: str, content: str) -> str:
        payload = {
            "title": title[:240],
            "content": content[:MAX_CONTENT_CHARS],
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
            "Score only emotional qualities explicitly supported by this memory. "
            "Do not infer hidden motives or facts. Return exactly one JSON object "
            "with exactly the six output_schema keys, no markdown or explanation. "
            "Use low confidence when the evidence is ambiguous."
        )
        return (
            f"{rules}\nINPUT="
            + json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    @staticmethod
    def _content(envelope: object) -> str:
        if type(envelope) is not dict:
            raise EAxisNightError("provider.invalid_envelope", retryable=True)
        choices = envelope.get("choices")
        if type(choices) is not list or len(choices) != 1:
            raise EAxisNightError("provider.invalid_choices", retryable=True)
        choice = choices[0]
        if type(choice) is not dict:
            raise EAxisNightError("provider.invalid_choice", retryable=True)
        if choice.get("finish_reason") != "stop":
            raise EAxisNightError("provider.incomplete", retryable=True)
        message = choice.get("message")
        if type(message) is not dict or type(message.get("content")) is not str:
            raise EAxisNightError("provider.invalid_message", retryable=True)
        content = message["content"].strip()
        if not content:
            raise EAxisNightError("provider.empty", retryable=True)
        return content

    def score(self, *, title: str, content: str) -> dict[str, Any]:
        try:
            envelope = self.provider(self._prompt(title, content))
        except EAxisNightError:
            raise
        except Exception as exc:
            raise EAxisNightError("provider.transport", retryable=True) from exc
        raw = self._content(envelope)
        try:
            parsed = strict_json_loads(raw)
        except Exception as exc:
            raise EAxisNightError("provider.invalid_json") from exc
        if type(parsed) is not dict or set(parsed) != _SCORE_FIELDS:
            raise EAxisNightError("schema.fields")
        annotation, error = build_shadow_annotation(
            bucket_id="validation-only",
            source_digest="0" * 64,
            scorer=self.scorer_name,
            model=self.model,
            rubric_version=self.rubric_version,
            score=parsed,
            min_confidence=self.min_confidence,
        )
        if annotation is None or error:
            raise EAxisNightError(error or "schema.invalid")
        return dict(annotation["score"])


def _terminal_identities(
    rows: list[dict[str, Any]],
) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            str(row.get("bucket_id") or ""),
            str(row.get("source_digest") or ""),
            str(row.get("scorer") or ""),
            str(row.get("model") or ""),
            str(row.get("rubric_version") or ""),
        )
        for row in rows
    }


async def run_e_axis_shadow(
    *,
    bucket_manager: Any,
    store: EAxisShadowStore,
    scorer: StrictEAxisScorer,
    max_per_run: int = DEFAULT_MAX_PER_RUN,
) -> EAxisNightResult:
    max_per_run = _plain_int(
        max_per_run,
        default=DEFAULT_MAX_PER_RUN,
        minimum=1,
        maximum=100,
    )
    before = await bucket_manager.list_all(
        include_archive=False,
        include_nsfw=False,
    )
    eligible = sorted(
        (bucket for bucket in before if _eligible_lmc5_bucket(bucket)),
        key=_bucket_sort_key,
        reverse=True,
    )
    terminal = _terminal_identities(store.load())
    pending: list[tuple[dict[str, Any], str]] = []
    existing = 0
    for bucket in eligible:
        digest = _source_digest(bucket["content"])
        identity = (
            bucket["id"],
            digest,
            scorer.scorer_name,
            scorer.model,
            scorer.rubric_version,
        )
        if identity in terminal:
            existing += 1
        else:
            pending.append((bucket, digest))

    attempted = added = failed = 0
    for bucket, digest in pending[:max_per_run]:
        attempted += 1
        metadata = bucket.get("metadata") or {}
        title = str(metadata.get("name") or bucket["id"])
        try:
            score = scorer.score(title=title, content=bucket["content"])
            row, error = build_shadow_annotation(
                bucket_id=bucket["id"],
                source_digest=digest,
                scorer=scorer.scorer_name,
                model=scorer.model,
                rubric_version=scorer.rubric_version,
                score=score,
                min_confidence=scorer.min_confidence,
            )
            if row is None or error:
                raise EAxisNightError(error or "schema.invalid")
            if store.append(row):
                added += 1
            else:
                existing += 1
        except EAxisNightError as exc:
            failed += 1
            if not exc.retryable:
                failure = build_failure_record(
                    bucket_id=bucket["id"],
                    source_digest=digest,
                    scorer=scorer.scorer_name,
                    model=scorer.model,
                    rubric_version=scorer.rubric_version,
                    category=exc.code,
                )
                store.append(failure)

    after = await bucket_manager.list_all(
        include_archive=False,
        include_nsfw=False,
    )
    after_fingerprints = {
        (bucket["id"], _source_digest(bucket["content"]))
        for bucket in after
        if _eligible_lmc5_bucket(bucket)
    }
    before_fingerprints = {
        (bucket["id"], _source_digest(bucket["content"]))
        for bucket in eligible
    }
    if after_fingerprints != before_fingerprints:
        raise EAxisNightError("bucket_set_changed")

    return EAxisNightResult(
        eligible=len(eligible),
        attempted=attempted,
        added=added,
        existing=existing,
        failed=failed,
        remaining=max(0, len(pending) - attempted),
    )


def build_e_axis_runtime(
    config: dict[str, Any],
) -> tuple[BucketManager, EAxisShadowStore, StrictEAxisScorer, int]:
    section = config.get("e_axis_shadow", {}) or {}
    if type(section) is not dict:
        raise EAxisNightError("config.section_invalid")
    if section.get("enabled", True) is not True:
        raise EAxisNightError("config.disabled")
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
        maximum=1024,
    )
    min_confidence = normalize_min_confidence(
        section.get("min_confidence", 0.3)
    )
    if min_confidence is None:
        raise EAxisNightError("config.min_confidence_invalid")

    dehydration = config.get("dehydration", {}) or {}
    if type(dehydration) is not dict:
        raise EAxisNightError("config.provider_invalid")
    model = str(dehydration.get("model") or "deepseek-chat")
    provider = OpenAIChatProvider(
        api_key=str(dehydration.get("api_key") or ""),
        base_url=str(
            dehydration.get("base_url") or "https://api.deepseek.com/v1"
        ),
        model=model,
        max_tokens=max_tokens,
        temperature=_plain_finite(
            section.get("temperature"),
            default=0.0,
        ),
        timeout_seconds=75.0,
    )
    manager = BucketManager(config)
    store = EAxisShadowStore(
        Path(config["buckets_dir"]) / ".axis" / "e-shadow.jsonl",
        maintenance_root=config["buckets_dir"],
    )
    scorer = StrictEAxisScorer(
        provider,
        model=model,
        min_confidence=min_confidence,
    )
    return manager, store, scorer, max_per_run


def main() -> int:
    try:
        config = load_config()
        manager, store, scorer, max_per_run = build_e_axis_runtime(config)
        result = asyncio.run(
            run_e_axis_shadow(
                bucket_manager=manager,
                store=store,
                scorer=scorer,
                max_per_run=max_per_run,
            )
        )
    except EAxisNightError as exc:
        print(
            json.dumps(
                {"ok": False, "code": exc.code},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1
    print(
        json.dumps(
            {"ok": result.failed == 0, **asdict(result)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
