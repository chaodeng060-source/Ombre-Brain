"""JEV provider at the existing relevance gate; never writes memory facts.

Only the selected query and redacted candidate bodies go to the scoring API.
The host server keeps ownership of retrieval, world isolation and selection.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import jev_reranker

REVISION = "jev-native-gate-v1"
_LAST: dict = {}
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
    """All candidates are scored once, including content after the old 200-char cut."""
    settings = _section(config)
    candidates = []
    for bucket in buckets:
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
    )
    _LAST.clear()
    _LAST.update(model=batch.model, candidates=len(candidates), requests=batch.requests,
                 cache_hits=batch.cache_hits, input_tokens=batch.input_tokens,
                 output_tokens=batch.output_tokens)
    _LOG.info("JEV rerank completed model=%s candidates=%d requests=%d cache_hits=%d input_tokens=%d output_tokens=%d",
              batch.model, len(candidates), batch.requests, batch.cache_hits,
              batch.input_tokens, batch.output_tokens)
    return batch


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


def status(config: dict) -> dict:
    """Public readiness/version and aggregate counters, never credentials or text."""
    return {"provider": "jev" if enabled(config) else "legacy",
            "enabled": enabled(config), "revision": REVISION,
            "model": str(_section(config).get("model") or "jev-latest"),
            "last_batch": dict(_LAST)}
