"""JEV provider at the existing relevance gate; never writes memory facts.

Only the selected query and redacted candidate bodies go to the scoring API.
The host server keeps ownership of retrieval, world isolation and selection.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import jev_reranker

REVISION = "jev-native-gate-v2"
_LAST: dict = {}
DEFAULT_MAX_CANDIDATES = 16
DEFAULT_DEADLINE_SEC = 2.0


@dataclass(frozen=True)
class GateScore:
    """Scored slice of the candidate list plus the raw provider batch."""

    buckets: list[dict]
    scores: list[int]
    batch: object
_LOG = logging.getLogger(__name__)


def _section(config: dict) -> dict:
    value = config.get("jev_reranker", {})
    return value if isinstance(value, dict) else {}


def enabled(config: dict) -> bool:
    override = os.getenv("OMBRE_JEV_RERANK_ENABLED")
    value = override if override is not None else _section(config).get("enabled", False)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _key(config: dict) -> str:
    value = os.getenv("OMBRE_JEV_API_KEY", "").strip() or os.getenv("JEV_API_KEY", "").strip()
    if value:
        return value
    path = _section(config).get("api_key_file")
    if not path:
        raise ValueError("JEV credential is not configured")
    source = Path(path).expanduser()
    try:
        if source.stat().st_mode & 0o077:
            raise ValueError("JEV credential file must be private")
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise ValueError("JEV credential file is unavailable") from None
    for line in lines:
        name, sep, raw = line.partition("=")
        if sep and name.strip() in {"JEV_API_KEY", "OMBRE_JEV_API_KEY"}:
            value = raw.strip().strip("\"'")
            if value:
                return value
    raise ValueError("JEV credential file has no key")


async def score(config: dict, query: str, buckets: list[dict], *, redact, timeout: float):
    """Score the head of the candidate list, bounded by calls and a batch deadline.

    The gate is subtractive and the list arrives in retrieval order, so the tail
    beyond ``max_candidates`` is not scored and is not returned. This mirrors the
    upstream profile (at most 16 paid calls per search) instead of paying for
    every candidate.
    """
    settings = _section(config)
    limit = max(1, int(settings.get("max_candidates", DEFAULT_MAX_CANDIDATES)))
    selected = list(buckets[:limit])
    deadline = float(settings.get("deadline_sec", DEFAULT_DEADLINE_SEC))
    candidates = []
    for bucket in selected:
        metadata = bucket.get("metadata") or {}
        # IDs, file paths, raw session pointers and other private metadata are
        # not needed for relevance. A human title + complete body is enough.
        title = str(metadata.get("name") or "")
        body = str(bucket.get("content") or "")
        candidates.append(redact((title + "\n" + body).strip()))
    batch = await jev_reranker.score_candidates(
        redact(query), candidates, api_key=_key(config),
        model=str(settings.get("model") or "jev-latest"),
        endpoint=str(settings.get("endpoint") or "https://api.typesafe.ai/v1/systemone"),
        timeout=timeout,
        concurrency=int(settings.get("concurrency", 4)),
        cache_ttl=float(settings.get("cache_ttl", 600)),
        cache_max=int(settings.get("cache_max", 512)),
        deadline=deadline,
    )
    _LAST.clear()
    _LAST.update(model=batch.model, candidates=len(candidates), requests=batch.requests,
                 cache_hits=batch.cache_hits, input_tokens=batch.input_tokens,
                 output_tokens=batch.output_tokens)
    _LOG.info("JEV rerank completed model=%s candidates=%d requests=%d cache_hits=%d input_tokens=%d output_tokens=%d",
              batch.model, len(candidates), batch.requests, batch.cache_hits,
              batch.input_tokens, batch.output_tokens)
    return GateScore(selected, batch.scores, batch)


def legacy_select(buckets: list[dict], scores: list[int], keep: set[str], max_results: int) -> list[dict]:
    """Compatibility for older source checkouts without the score-mode selector.

    Production uses the server's existing score-mode selector, threshold,
    bonuses and topic deduplication instead of this compatibility path.
    """
    threshold = int(os.getenv("OMBRE_DS_GATE_SCORE_THRESHOLD", "70"))
    ranked = sorted(enumerate(buckets), key=lambda item: (-scores[item[0]], item[0]))
    eligible = [bucket for i, bucket in ranked if scores[i] >= threshold or bucket.get("id") in keep]
    forced = [bucket for bucket in eligible if bucket.get("id") in keep]
    normal = [bucket for bucket in eligible if bucket.get("id") not in keep]
    room = max(0, max_results - len(forced))
    included = {id(bucket) for bucket in forced + normal[:room]}
    return [bucket for bucket in eligible if id(bucket) in included][:max(0, max_results)]


def selector_mode(config: dict) -> str:
    """Which selector consumes the scores: the host's score-mode selector or legacy."""
    raw = os.getenv("OMBRE_JEV_SELECTOR")
    value = raw if raw is not None else _section(config).get("selector", "score_mode")
    mode = str(value).strip().lower() or "score_mode"
    if mode not in {"score_mode", "legacy"}:
        raise ValueError("JEV selector must be score_mode or legacy")
    return mode


def select(config: dict, result: GateScore, keep: set[str], max_results: int, *,
           score_mode_select=None, score_mode_enabled: bool = False) -> list[dict]:
    """Apply the host's existing selection contract to the JEV scores."""
    if selector_mode(config) == "score_mode" and score_mode_enabled and score_mode_select is not None:
        return score_mode_select(result.buckets, result.scores, keep, max_results)
    return legacy_select(result.buckets, result.scores, keep, max_results)


def status(config: dict) -> dict:
    """Public readiness/version and aggregate counters, never credentials or text."""
    return {"provider": "jev" if enabled(config) else "legacy",
            "enabled": enabled(config), "revision": REVISION,
            "model": str(_section(config).get("model") or "jev-latest"),
            "max_candidates": int(_section(config).get("max_candidates", DEFAULT_MAX_CANDIDATES)),
            "deadline_sec": float(_section(config).get("deadline_sec", DEFAULT_DEADLINE_SEC)),
            "selector": selector_mode(config),
            "score_threshold": os.getenv("OMBRE_DS_GATE_SCORE_THRESHOLD", ""),
            "last_batch": dict(_LAST)}
