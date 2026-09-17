import asyncio
import copy
import json

import httpx
import pytest

from retrieval_hints import Generator, HintError, lexical_text, parse_hints, source_record


BODY = "周二林禾在银杏馆讨论星舟项目，决定保留旧索引。"


def payload():
    return {
        "schema_version": 1,
        "who": [{"text": "林禾", "evidence": "林禾在银杏馆"}],
        "where": [{"text": "银杏馆", "evidence": "林禾在银杏馆"}],
        "when": [{"text": "周二", "evidence": "周二林禾"}],
        "what": [{"text": "星舟保留旧索引", "evidence": "讨论星舟项目，决定保留旧索引"}],
        "aliases": [{"text": "星舟的旧索引方案", "evidence": "讨论星舟项目，决定保留旧索引"}],
    }


def test_schema_and_literal_evidence():
    value = parse_hints(json.dumps(payload()), BODY)
    assert value == payload()
    assert lexical_text(value).count("林禾") == 1
    assert "schema_version" not in lexical_text(value)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema_version=True),
    lambda p: p.update(extra="untrusted"),
    lambda p: p.pop("when"),
    lambda p: p.update(who=None),
    lambda p: p["who"][0].update(text="林海"),
    lambda p: p["who"][0].update(evidence="不存在的证据"),
    lambda p: p["who"][0].update(extra=1),
    lambda p: p["aliases"][0].update(text="任务"),
    lambda p: p["aliases"][0].update(text="x" * 49),
    lambda p: p["who"].append(copy.deepcopy(p["who"][0])),
])
def test_invalid_is_not_successful_empty(mutation):
    data = payload()
    mutation(data)
    with pytest.raises(HintError):
        parse_hints(json.dumps(data), BODY)


@pytest.mark.parametrize("raw", ["[]", "null", "```json\n{}\n```", '{"schema_version":1,"schema_version":1}'])
def test_no_json_salvage_or_duplicate_fields(raw):
    with pytest.raises(HintError):
        parse_hints(raw, BODY)


def test_empty_is_valid_and_no_date_guessing():
    data = {key: [] for key in ("who", "where", "when", "what", "aliases")}
    data["schema_version"] = 1
    assert parse_hints(json.dumps(data), BODY) == data
    data["when"] = [{"text": "2026年9月15日", "evidence": "周二"}]
    with pytest.raises(HintError):
        parse_hints(json.dumps(data), BODY)


def test_source_metadata_is_local_and_body_bound():
    bucket = {"id": "synthetic-a", "content": BODY, "path": "/tmp/synthetic.md",
              "metadata": {"world": "test", "e_authored_by": "test-author"}}
    assert source_record(bucket)["source_content_sha256"] != source_record({**bucket, "content": BODY + "新"})["source_content_sha256"]
    assert "content" not in source_record(bucket)


def test_source_event_identity_is_preserved_not_replaced_by_bucket_date():
    bucket = {"id": "synthetic-a", "content": BODY, "metadata": {
        "source_event_ids": ["event-old"], "source_session": "room:test",
        "source_kind": "conversation", "source_digest": "source-digest",
        "created": "2026-09-14"}}
    source = source_record(bucket)
    assert source["source_event_ids"] == ["event-old"]
    assert source["source_session"] == "room:test"
    assert source["source_kind"] == "conversation"
    assert source["source_digest"] == "source-digest"
    assert "event_at" not in source  # Admission time is not event time.


@pytest.mark.asyncio
async def test_off_does_not_read_provider_configuration(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("OFF read provider config")
    monkeypatch.setattr("retrieval_hints.provider_config", fail)
    result = await Generator(enabled=False).generate(BODY)
    assert result.status == "disabled"


@pytest.mark.asyncio
async def test_one_request_current_body_only_and_usage():
    requests = []
    async def handler(request):
        data = json.loads(request.content)
        requests.append(data)
        return httpx.Response(200, json={"model": "mock-model", "usage": {"prompt_tokens": 17},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload())}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await Generator(enabled=True, client=client, config=("https://mock.invalid/v1", "mock-key", "approved-model")).generate(BODY)
    assert result.status == "ok"
    assert result.usage == {"prompt_tokens": 17}
    assert len(requests) == 1
    data = requests[0]
    assert data["messages"][1] == {"role": "user", "content": BODY}
    assert len(data["messages"]) == 2
    assert data["thinking"] == {"type": "disabled"}
    assert data["model"] == "approved-model"
    assert data["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_timeout_no_retry_and_unknown_usage():
    calls = 0
    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        raise AssertionError("deadline not enforced")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await Generator(enabled=True, client=client, config=("https://mock.invalid", "mock", "mock"), timeout_s=.01).generate(BODY)
    assert result.status == "timeout" and result.usage is None
    assert calls == 1


@pytest.mark.asyncio
async def test_oversize_does_not_spend_call():
    result = await Generator(enabled=True, config=("https://mock.invalid", "mock", "mock")).generate("字" * 6000)
    assert result.status == "oversize" and result.usage is None
