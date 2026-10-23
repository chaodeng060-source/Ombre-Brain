"""Native TypeSafe JEV relevance scores; callers redact, rank and set thresholds.

Only digests, scores, resolved model names and timestamps enter the process cache.
Failed batches never populate it. Token counts cover new HTTP calls only; cache
hits count candidate positions served by an existing entry. Identical misses in
one batch share a request. ``timeout`` bounds each request, not the whole batch.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass

import httpx


class JEVError(RuntimeError):
    """Safe failure message without request data, credentials or response bodies."""


@dataclass
class ScoreBatch:
    scores: list[int]
    model: str
    requests: int
    cache_hits: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class _Score:
    score: int
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class _CacheEntry:
    score: int
    model: str
    created_at: float
    expires_at: float


_PROMPT_VERSION = "jev-relevance-v1"
_INSTRUCTIONS = "How useful is this memory for answering the question?"
_CRITERIA = ["Irrelevant", "Loosely related", "Partially answers", "Directly answers"]
_CACHE: OrderedDict[str, _CacheEntry] = OrderedDict()


def _cache_key(endpoint: str, model: str, key_digest: str, query: str, candidate: str) -> str:
    digest = hashlib.sha256()
    for value in (endpoint, model, key_digest, _PROMPT_VERSION, query, candidate):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _prune_cache(now: float, ttl: float, maximum: int) -> None:
    for key, entry in list(_CACHE.items()):
        if now >= entry.expires_at or now - entry.created_at >= ttl:
            del _CACHE[key]
    while len(_CACHE) > maximum:
        _CACHE.popitem(last=False)


def _parse_score(data: object) -> _Score:
    if not isinstance(data, dict):
        raise JEVError("JEV returned an invalid response.")
    model = data.get("model")
    answers = data.get("answers")
    rel = answers.get("rel") if isinstance(answers, dict) else None
    if not isinstance(model, str) or not model.strip() or not isinstance(rel, dict):
        raise JEVError("JEV returned an invalid response.")
    score = rel.get("score")
    if (
        rel.get("type") != "score"
        or type(score) not in (int, float)
        or not 0 <= score <= 3
        or not math.isfinite(score)
    ):
        raise JEVError("JEV returned an invalid relevance score.")
    usage = data.get("usage", {})
    if not isinstance(usage, dict):
        raise JEVError("JEV returned invalid token usage.")
    counts = [usage.get("input_tokens", 0), usage.get("output_tokens", 0)]
    if any(type(count) is not int or count < 0 for count in counts):
        raise JEVError("JEV returned invalid token usage.")
    return _Score(round(score * 100 / 3), model, *counts)


async def _request_score(
    client: httpx.AsyncClient, endpoint: str, model: str, query: str, candidate: str, timeout: float
) -> _Score:
    payload = {
        "model": model,
        "state": "Question: " + query + "\nMemory: " + candidate,
        "questions": {
            "rel": {"type": "score", "instructions": _INSTRUCTIONS, "criteria": _CRITERIA}
        },
    }
    # Raise outside the except block so even the exception context contains no
    # httpx request (which holds the Authorization header and complete state).
    error = None
    try:
        async with asyncio.timeout(timeout):
            response = await client.post(endpoint, json=payload)
    except TimeoutError:
        error = "JEV request timed out."
    except (httpx.HTTPError, httpx.InvalidURL, OSError, ValueError):
        error = "JEV network request failed."
    if error:
        raise JEVError(error)
    if response.status_code in (401, 403):
        raise JEVError("JEV authentication failed.")
    if not response.is_success:
        raise JEVError(f"JEV HTTP request failed (status {response.status_code}).")
    try:
        data = response.json()
    except ValueError:
        data = None
    return _parse_score(data)


async def score_candidates(
    query: str,
    candidates: list[str],
    *,
    api_key: str,
    model: str = "jev-latest",
    endpoint: str = "https://api.typesafe.ai/v1/systemone",
    timeout: float = 12,
    concurrency: int = 4,
    cache_ttl: float = 600,
    cache_max: int = 512,
    deadline: float | None = None,
) -> ScoreBatch:
    """Score the complete batch in input order, or cancel/drain and raise JEVError.

    Empty batches return the requested model because none has been resolved.
    Cache TTL is limited by both the writing and reading call's TTL. Zero TTL or
    size disables caching. A batch spanning different resolved models fails
    because a single ``ScoreBatch.model`` cannot describe it accurately.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        raise JEVError("A JEV API key is required.")
    if type(concurrency) is not int or concurrency < 1:
        raise JEVError("JEV concurrency must be a positive integer.")
    if type(cache_max) is not int or cache_max < 0:
        raise JEVError("JEV cache size must be a nonnegative integer.")
    for value, minimum, name in ((timeout, 0, "timeout"), (cache_ttl, 0, "cache TTL")):
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise JEVError(f"Invalid JEV {name}.")
    if timeout == 0:
        raise JEVError("Invalid JEV timeout.")
    if deadline is not None and (
        type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0
    ):
        raise JEVError("Invalid JEV batch deadline.")
    if not candidates:
        return ScoreBatch([], model, 0, 0, 0, 0)

    now = time.monotonic()
    _prune_cache(now, cache_ttl, cache_max)
    use_cache = cache_ttl > 0 and cache_max > 0
    key_digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    keys = [_cache_key(endpoint, model, key_digest, query, candidate) for candidate in candidates]
    results: dict[str, _Score] = {}
    missing: dict[str, str] = {}
    cache_hits = 0
    for key, candidate in zip(keys, candidates):
        entry = _CACHE.get(key) if use_cache else None
        if entry is not None:
            _CACHE.move_to_end(key)
            results[key] = _Score(entry.score, entry.model)
            cache_hits += 1
        else:
            missing[key] = candidate

    models = {result.model for result in results.values()}
    if len(models) > 1:
        raise JEVError("JEV returned inconsistent resolved models.")
    resolved_model = next(iter(models), None)
    requests = input_tokens = output_tokens = 0
    if missing:
        pending = iter(missing.items())
        failed = False
        async with httpx.AsyncClient(
            headers={"Authorization": "Bearer " + api_key},
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            transport=httpx.AsyncHTTPTransport(retries=0, trust_env=False),
        ) as client:

            async def worker() -> None:
                nonlocal failed, resolved_model, requests, input_tokens, output_tokens
                while not failed:
                    item = next(pending, None)
                    if item is None:
                        return
                    key, candidate = item
                    try:
                        requests += 1
                        result = await _request_score(client, endpoint, model, query, candidate, timeout)
                        if resolved_model is not None and result.model != resolved_model:
                            raise JEVError("JEV returned inconsistent resolved models.")
                        resolved_model = result.model
                        results[key] = result
                        input_tokens += result.input_tokens
                        output_tokens += result.output_tokens
                    except BaseException:
                        failed = True
                        raise

            tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(missing)))]
            try:
                gathered = asyncio.gather(*tasks)
                if deadline is None:
                    await gathered
                else:
                    # The batch deadline is the caller's gate budget, not a per-request
                    # timeout: a slow provider must fail the whole batch so the caller
                    # can fall back inside its own budget instead of paying for a
                    # half-finished rerank.
                    await asyncio.wait_for(gathered, timeout=deadline)
            except asyncio.TimeoutError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise JEVError("JEV batch deadline exceeded.") from None
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        if use_cache:
            now = time.monotonic()
            _prune_cache(now, cache_ttl, cache_max)
            for key in missing:
                result = results[key]
                _CACHE[key] = _CacheEntry(result.score, result.model, now, now + cache_ttl)
                _CACHE.move_to_end(key)
                while len(_CACHE) > cache_max:
                    _CACHE.popitem(last=False)

    return ScoreBatch(
        [results[key].score for key in keys],
        resolved_model or model,
        requests,
        cache_hits,
        input_tokens,
        output_tokens,
    )
