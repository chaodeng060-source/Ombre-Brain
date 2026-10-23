"""Synthetic native-protocol checks. Every HTTP client uses MockTransport."""

import asyncio
import inspect
import json
import logging
import traceback
from dataclasses import is_dataclass
from types import SimpleNamespace

import httpx
import pytest

import jev_reranker as jev


def response(score=3, *, model="jev-1.14.0", usage=None):
    body = {"model": model, "answers": {"rel": {"type": "score", "score": score}}}
    if usage is not None:
        body["usage"] = usage
    # Permit deliberately malformed NaN/Infinity fixtures, which httpx's json=
    # encoder correctly refuses to produce for outgoing requests.
    return httpx.Response(200, content=json.dumps(body))


@pytest.fixture(autouse=True)
def mock_http(monkeypatch):
    jev._CACHE.clear()
    real_client = httpx.AsyncClient
    state = SimpleNamespace(calls=[], clients=[], client_options=[], transport_options=[])

    def unexpected(request):
        raise AssertionError("Unexpected HTTP call to the offline transport")

    state.handler = unexpected

    async def dispatch(request):
        state.calls.append(request)
        result = state.handler(request)
        return await result if inspect.isawaitable(result) else result

    def transport(**kwargs):
        state.transport_options.append(kwargs)
        return httpx.MockTransport(dispatch)

    def client(**kwargs):
        state.client_options.append(kwargs)
        result = real_client(**kwargs)
        state.clients.append(result)
        return result

    monkeypatch.setattr(jev.httpx, "AsyncHTTPTransport", transport)
    monkeypatch.setattr(jev.httpx, "AsyncClient", client)
    yield state
    jev._CACHE.clear()


def assert_safe(error, *private_values):
    rendered = "".join(traceback.format_exception(error))
    for value in private_values:
        assert value not in str(error)
        assert value not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.asyncio
async def test_native_payload_alias_resolution_and_usage(mock_http, caplog):
    query = "synthetic-question: 哪一条能回答？"
    candidates = ["synthetic-memory-" + "长" * 250 + str(i) for i in range(5)]
    raw_scores = [0, 1, 2, 3, 1.5]

    def handler(request):
        body = json.loads(request.content)
        index = candidates.index(body["state"].split("\nMemory: ", 1)[1])
        assert request.method == "POST"
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer synthetic-key"
        assert body == {
            "model": "jev-latest",
            "state": "Question: " + query + "\nMemory: " + candidates[index],
            "questions": {
                "rel": {
                    "type": "score",
                    "instructions": "How useful is this memory for answering the question?",
                    "criteria": ["Irrelevant", "Loosely related", "Partially answers", "Directly answers"],
                }
            },
        }
        assert set(request.extensions["timeout"].values()) == {2.5}
        return response(raw_scores[index], usage={"input_tokens": index + 1, "output_tokens": 2})

    mock_http.handler = handler
    with caplog.at_level(logging.DEBUG):
        batch = await jev.score_candidates(query, candidates, api_key="synthetic-key", timeout=2.5)
    assert is_dataclass(batch)
    assert batch == jev.ScoreBatch([0, 33, 67, 100, 50], "jev-1.14.0", 5, 0, 15, 10)
    assert mock_http.client_options[0]["follow_redirects"] is False
    assert mock_http.client_options[0]["trust_env"] is False
    assert mock_http.transport_options == [{"retries": 0, "trust_env": False}]
    assert all(client.is_closed for client in mock_http.clients)
    for private_value in [query, "synthetic-key", *candidates]:
        assert private_value not in caplog.text
        assert private_value not in repr(jev._CACHE)


@pytest.mark.asyncio
@pytest.mark.parametrize("usage,expected", [(None, (0, 0)), ({}, (0, 0)), ({"input_tokens": 7}, (7, 0)), ({"output_tokens": 4}, (0, 4))])
async def test_optional_usage(mock_http, usage, expected):
    mock_http.handler = lambda request: response(usage=usage)
    batch = await jev.score_candidates("q", ["memory"], api_key="stub")
    assert (batch.input_tokens, batch.output_tokens) == expected


@pytest.mark.asyncio
async def test_empty_batch_never_constructs_http_client(mock_http):
    assert await jev.score_candidates("q", [], api_key="stub") == jev.ScoreBatch([], "jev-latest", 0, 0, 0, 0)
    assert mock_http.calls == mock_http.clients == []


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["", " ", "\t"])
async def test_empty_key_is_safe_even_for_empty_batch(mock_http, key):
    query = "synthetic-private-question"
    for candidates in ([], ["synthetic-private-memory"]):
        with pytest.raises(jev.JEVError, match="key is required") as caught:
            await jev.score_candidates(query, candidates, api_key=key)
        assert_safe(caught.value, query, "synthetic-private-memory")
    assert mock_http.calls == mock_http.clients == []


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [-0.01, 3.01, True, False, None, "2", float("nan"), float("inf"), -float("inf"), [], {}, 10**400])
async def test_invalid_scores_fail_without_caching(mock_http, score):
    mock_http.handler = lambda request: response(score)
    with pytest.raises(jev.JEVError, match="score"):
        await jev.score_candidates("q", ["memory"], api_key="stub")
    assert len(mock_http.calls) == 1
    assert not jev._CACHE


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    None, [], {},
    {"model": "jev-1.14.0", "answers": None},
    {"model": "jev-1.14.0", "answers": {"rel": []}},
    {"model": "jev-1.14.0", "answers": {"rel": {"score": 2}}},
    {"model": "jev-1.14.0", "answers": {"rel": {"type": "text", "score": 2}}},
    {"model": "", "answers": {"rel": {"type": "score", "score": 2}}},
    {"model": True, "answers": {"rel": {"type": "score", "score": 2}}},
])
async def test_invalid_native_response_shapes(mock_http, body):
    mock_http.handler = lambda request: httpx.Response(200, content=json.dumps(body))
    with pytest.raises(jev.JEVError):
        await jev.score_candidates("q", ["memory"], api_key="stub")
    assert not jev._CACHE


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [{"input_tokens": True}, {"input_tokens": -1}, {"input_tokens": "7"}, {"input_tokens": 1.5}, {"output_tokens": False}, {"output_tokens": []}, []])
async def test_malformed_usage_is_not_silently_counted(mock_http, usage):
    mock_http.handler = lambda request: response(usage=usage)
    with pytest.raises(jev.JEVError, match="usage"):
        await jev.score_candidates("q", ["memory"], api_key="stub")
    assert not jev._CACHE


@pytest.mark.asyncio
async def test_invalid_json_does_not_expose_body(mock_http):
    private = "synthetic-private-response"
    mock_http.handler = lambda request: httpx.Response(200, content=private)
    with pytest.raises(jev.JEVError) as caught:
        await jev.score_candidates("q", ["memory"], api_key="stub")
    assert_safe(caught.value, private)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
async def test_http_errors_never_retry_or_follow_redirects(mock_http, status):
    private = ("synthetic-private-question", "synthetic-private-memory", "synthetic-private-key")
    mock_http.handler = lambda request: httpx.Response(
        status, content=" ".join(private), headers={"Location": "https://redirect.invalid/private"}
    )
    with pytest.raises(jev.JEVError) as caught:
        await jev.score_candidates(private[0], [private[1], "queued"], api_key=private[2], concurrency=1)
    assert_safe(caught.value, *private)
    assert len(mock_http.calls) == 1
    assert not jev._CACHE


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError])
async def test_network_errors_have_no_sensitive_exception_context(mock_http, error_type):
    private = ("synthetic-query-marker", "synthetic-memory-marker", "synthetic-key-marker")

    def handler(request):
        raise error_type(" ".join(private), request=request)

    mock_http.handler = handler
    with pytest.raises(jev.JEVError) as caught:
        await jev.score_candidates(private[0], [private[1]], api_key=private[2])
    assert_safe(caught.value, *private)
    assert len(mock_http.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid", "auth", "network"])
async def test_failure_cancels_drains_and_never_caches_partial_batch(mock_http, failure):
    mock_http.handler = lambda request: response(2)
    await jev.score_candidates("q", ["warm"], api_key="stub")
    mock_http.calls.clear()
    slow_started = asyncio.Event()
    slow_drained = asyncio.Event()

    async def handler(request):
        candidate = json.loads(request.content)["state"].split("\nMemory: ", 1)[1]
        if candidate == "good":
            return response(3)
        if candidate == "bad":
            await slow_started.wait()
            if failure == "auth":
                return httpx.Response(401, content="synthetic-sensitive-body")
            if failure == "network":
                raise httpx.ConnectError("synthetic-sensitive-body", request=request)
            return response(True)
        assert candidate == "slow", "Queued work was started after the batch failed"
        slow_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            slow_drained.set()

    mock_http.handler = handler
    with pytest.raises(jev.JEVError):
        await asyncio.wait_for(jev.score_candidates(
            "q", ["warm", "good", "bad", "slow", "queued"], api_key="stub", concurrency=2
        ), 1)
    assert slow_drained.is_set()
    assert len(mock_http.calls) == 3
    assert len(jev._CACHE) == 1
    assert all(client.is_closed for client in mock_http.clients)
    mock_http.handler = lambda request: response(2)
    batch = await jev.score_candidates("q", ["warm", "good"], api_key="stub")
    assert (batch.requests, batch.cache_hits) == (1, 1)


@pytest.mark.asyncio
async def test_caller_cancellation_drains_workers(mock_http):
    both_started = asyncio.Event()
    started = drained = 0

    async def handler(request):
        nonlocal started, drained
        started += 1
        if started == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            drained += 1

    mock_http.handler = handler
    task = asyncio.create_task(jev.score_candidates("q", ["a", "b", "queued"], api_key="stub", concurrency=2))
    try:
        await asyncio.wait_for(both_started.wait(), 1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert drained == 2
    assert len(mock_http.calls) == 2
    assert all(client.is_closed for client in mock_http.clients)
    assert not jev._CACHE


@pytest.mark.asyncio
async def test_timeout_cancels_request_and_is_safe(mock_http):
    drained = asyncio.Event()
    candidate = "synthetic-timeout-memory"
    key = "synthetic-timeout-key"

    async def handler(request):
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    mock_http.handler = handler
    with pytest.raises(jev.JEVError, match="timed out") as caught:
        await asyncio.wait_for(jev.score_candidates("q", [candidate], api_key=key, timeout=0.01), 1)
    assert_safe(caught.value, key, candidate)
    assert drained.is_set()
    assert len(mock_http.calls) == 1


@pytest.mark.asyncio
async def test_concurrency_bound_and_order_under_out_of_order_responses(mock_http):
    started = [asyncio.Event() for _ in range(5)]
    release = [asyncio.Event() for _ in range(5)]
    raw_scores = [0, 3, 1, 2, 1.5]
    active = peak = 0
    finished = []

    async def handler(request):
        nonlocal active, peak
        index = int(json.loads(request.content)["state"].split("\nMemory: ", 1)[1])
        active += 1
        peak = max(peak, active)
        started[index].set()
        try:
            await release[index].wait()
            finished.append(index)
            return response(raw_scores[index])
        finally:
            active -= 1

    mock_http.handler = handler
    task = asyncio.create_task(jev.score_candidates("q", list(map(str, range(5))), api_key="stub", concurrency=2))
    try:
        await asyncio.wait_for(started[0].wait(), 1)
        await asyncio.wait_for(started[1].wait(), 1)
        assert not any(event.is_set() for event in started[2:])
        for index in [1, 2, 3, 4, 0]:
            await asyncio.wait_for(started[index].wait(), 1)
            release[index].set()
        batch = await asyncio.wait_for(task, 1)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert batch.scores == [0, 100, 33, 67, 50]
    assert batch.requests == 5
    assert peak == 2 and active == 0
    assert finished == [1, 2, 3, 4, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["query", "candidate", "api_key", "model", "endpoint", "prompt_version"])
async def test_cache_hits_and_namespace_full_content(mock_http, monkeypatch, changed):
    mock_http.handler = lambda request: response(2, usage={"input_tokens": 9, "output_tokens": 1})
    query = "synthetic-query-private"
    candidate = "synthetic-memory-private" + "a" * 250 + "first"
    key = "synthetic-key-private"
    first = await jev.score_candidates(query, [candidate], api_key=key)
    cached = await jev.score_candidates(query, [candidate], api_key=key)
    assert first.requests == 1
    assert cached == jev.ScoreBatch([67], "jev-1.14.0", 0, 1, 0, 0)
    options = {"api_key": key}
    if changed == "query":
        query += " changed"
    elif changed == "candidate":
        candidate = candidate[:-5] + "other"
    elif changed == "prompt_version":
        monkeypatch.setattr(jev, "_PROMPT_VERSION", "jev-relevance-v2")
    else:
        options[changed] = {"api_key": "different-stub", "model": "jev-1.14.0", "endpoint": "https://alternate.invalid/v1/systemone"}[changed]
    uncached = await jev.score_candidates(query, [candidate], **options)
    assert (uncached.requests, uncached.cache_hits) == (1, 0)
    assert len(mock_http.calls) == 2
    assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest in jev._CACHE)
    for private in ["synthetic-query-private", "synthetic-memory-private", key]:
        assert private not in repr(jev._CACHE)


@pytest.mark.asyncio
async def test_duplicate_candidates_share_requests_and_reuse_cache(mock_http):
    mock_http.handler = lambda request: response(
        2 if json.loads(request.content)["state"].endswith("a") else 1,
        usage={"input_tokens": 5, "output_tokens": 1},
    )
    first = await jev.score_candidates("q", ["a", "b", "a"], api_key="stub")
    assert first == jev.ScoreBatch([67, 33, 67], "jev-1.14.0", 2, 0, 10, 2)
    cached = await jev.score_candidates("q", ["a", "b", "a"], api_key="stub")
    assert cached == jev.ScoreBatch(first.scores, first.model, 0, 3, 0, 0)
    assert len(mock_http.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("write_ttl,read_ttl", [(10, 100), (100, 10)])
async def test_ttl_honors_writer_and_reader_limits(mock_http, monkeypatch, write_ttl, read_ttl):
    clock = [100.0]
    monkeypatch.setattr(jev, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    mock_http.handler = lambda request: response(1)
    await jev.score_candidates("q", ["a"], api_key="stub", cache_ttl=write_ttl)
    clock[0] += 9
    assert (await jev.score_candidates("q", ["a"], api_key="stub", cache_ttl=read_ttl)).cache_hits == 1
    clock[0] += 1
    expired = await jev.score_candidates("q", ["a"], api_key="stub", cache_ttl=read_ttl)
    assert (expired.requests, expired.cache_hits) == (1, 0)


@pytest.mark.asyncio
async def test_lru_cache_remains_bounded(mock_http):
    mock_http.handler = lambda request: response(1)

    async def score(candidates, maximum=2):
        return await jev.score_candidates("q", candidates, api_key="stub", cache_max=maximum)

    await score(["a", "b"])
    assert (await score(["a"])).cache_hits == 1
    await score(["c"])
    assert len(jev._CACHE) == 2
    assert (await score(["a"])).cache_hits == 1
    assert (await score(["b"])).requests == 1
    await score(["d", "e", "f"], maximum=1)
    assert len(jev._CACHE) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{"cache_ttl": 0}, {"cache_max": 0}])
async def test_cache_can_be_disabled(mock_http, options):
    mock_http.handler = lambda request: response()
    for _ in range(2):
        batch = await jev.score_candidates("q", ["a"], api_key="stub", **options)
        assert (batch.requests, batch.cache_hits) == (1, 0)
    assert len(mock_http.calls) == 2
    assert not jev._CACHE


@pytest.mark.asyncio
@pytest.mark.parametrize("warm_cache", [False, True])
async def test_mixed_resolved_models_fail_as_one_batch(mock_http, warm_cache):
    mock_http.handler = lambda request: response(
        model="jev-1.14.0" if json.loads(request.content)["state"].endswith("a") else "jev-1.15.0"
    )
    if warm_cache:
        await jev.score_candidates("q", ["a"], api_key="stub")
    with pytest.raises(jev.JEVError, match="inconsistent resolved models"):
        await jev.score_candidates("q", ["a", "b"], api_key="stub")
    assert len(jev._CACHE) == int(warm_cache)


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{"concurrency": 0}, {"concurrency": True}, {"timeout": 0}, {"cache_max": -1}])
async def test_invalid_resource_limits_do_not_send_or_hang(mock_http, options):
    with pytest.raises(jev.JEVError):
        await jev.score_candidates("q", ["a"], api_key="stub", **options)
    assert not mock_http.calls
