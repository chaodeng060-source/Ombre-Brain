import json
import threading
from types import SimpleNamespace

import pytest

import server


class _Request:
    def __init__(self, body, *, token="hook-secret"):
        self._body = (
            body
            if isinstance(body, bytes)
            else json.dumps(body, ensure_ascii=False).encode("utf-8")
        )
        self.headers = (
            {"x-ombre-hook-token": token}
            if token is not None
            else {}
        )
        self.client = SimpleNamespace(host="127.0.0.1")

    async def body(self):
        return self._body


def _json(response):
    return json.loads(response.body)


@pytest.fixture(autouse=True)
def _hook_environment(monkeypatch, test_config):
    monkeypatch.setenv("OMBRE_HOOK_TOKEN", "hook-secret")
    monkeypatch.setitem(server.config, "buckets_dir", test_config["buckets_dir"])
    monkeypatch.setattr(server, "_lmc5_ledger", None)


@pytest.mark.asyncio
async def test_raw_event_batch_is_exact_atomic_and_replayable():
    first_line = '{"uuid":"event-1", "message":{"text":"原样 空格"}}'
    second_line = '{"uuid":"event-2","type":"assistant"}'
    request = _Request(
        {
            "schema_version": 1,
            "session_id": "session-a",
            "events": [
                {"source_event_id": "event-1", "payload": first_line},
                {"source_event_id": "event-2", "payload": second_line},
            ],
        }
    )

    response = await server.lmc5_raw_events_hook(request)
    assert response.status_code == 200
    assert _json(response) == {
        "ok": True,
        "session_id": "session-a",
        "acknowledged": 2,
        "inserted": 2,
    }
    rows = server._get_lmc5_ledger().list_uncovered_raw_events()
    assert [row.payload.decode("utf-8") for row in rows] == [
        first_line,
        second_line,
    ]

    replay = await server.lmc5_raw_events_hook(request)
    assert replay.status_code == 200
    assert _json(replay)["acknowledged"] == 2
    assert _json(replay)["inserted"] == 0


@pytest.mark.asyncio
async def test_raw_event_durable_write_runs_off_event_loop(monkeypatch):
    caller_thread = threading.get_ident()
    observed = {}

    class Ledger:
        def append_raw_events(self, events):
            observed["thread"] = threading.get_ident()
            observed["events"] = events
            return (SimpleNamespace(created=True),)

    monkeypatch.setattr(server, "_get_lmc5_ledger", lambda: Ledger())
    response = await server.lmc5_raw_events_hook(
        _Request(
            {
                "schema_version": 1,
                "session_id": "session-a",
                "events": [
                    {
                        "source_event_id": "event-1",
                        "payload": '{"uuid":"event-1"}',
                    }
                ],
            }
        )
    )

    assert response.status_code == 200
    assert observed["thread"] != caller_thread
    assert len(observed["events"]) == 1


@pytest.mark.asyncio
async def test_raw_event_conflict_rolls_back_whole_batch():
    await server.lmc5_raw_events_hook(
        _Request(
            {
                "schema_version": 1,
                "session_id": "session-a",
                "events": [
                    {
                        "source_event_id": "event-1",
                        "payload": '{"uuid":"event-1","value":1}',
                    }
                ],
            }
        )
    )

    response = await server.lmc5_raw_events_hook(
        _Request(
            {
                "schema_version": 1,
                "session_id": "session-a",
                "events": [
                    {
                        "source_event_id": "event-new",
                        "payload": '{"uuid":"event-new"}',
                    },
                    {
                        "source_event_id": "event-1",
                        "payload": '{"uuid":"event-1","value":2}',
                    },
                ],
            }
        )
    )
    assert response.status_code == 409
    report = server._get_lmc5_ledger().coverage_report()
    assert report.total_raw_events == 1
    assert tuple(
        identity.source_event_id for identity in report.uncovered_event_ids
    ) == ("event-1",)


@pytest.mark.asyncio
async def test_raw_event_route_rejects_bad_auth_and_malformed_line(monkeypatch):
    body = {
        "schema_version": 1,
        "session_id": "session-a",
        "events": [
            {"source_event_id": "event-1", "payload": "not-json"}
        ],
    }
    forbidden = await server.lmc5_raw_events_hook(_Request(body, token="wrong"))
    assert forbidden.status_code == 403

    malformed = await server.lmc5_raw_events_hook(_Request(body))
    assert malformed.status_code == 400
    assert server._get_lmc5_ledger().coverage_report().total_raw_events == 0

    monkeypatch.delenv("OMBRE_HOOK_TOKEN")
    unconfigured = await server.lmc5_raw_events_hook(_Request(body))
    assert unconfigured.status_code == 503


@pytest.mark.asyncio
async def test_recall_hook_uses_authoritative_breath_pipeline(monkeypatch):
    observed = {}

    async def fake_breath(**kwargs):
        observed.update(kwargs)
        return "[evidence_role: primary] remembered"

    monkeypatch.setattr(server, "breath", fake_breath)
    response = await server.lmc5_recall_hook(
        _Request(
            {
                "schema_version": 1,
                "prompt": "现在住在哪里？",
                "session_id": "session-a",
            }
        )
    )

    assert response.status_code == 200
    assert _json(response) == {
        "ok": True,
        "context": "[evidence_role: primary] remembered",
    }
    assert observed["query"] == "现在住在哪里？"
    assert observed["session_id"] == "session-a"
    assert observed["relation_depth"] == 2
    assert observed["include_images"] is False
    assert observed["include_body_state"] is False


@pytest.mark.asyncio
async def test_recall_hook_never_turns_failure_into_empty_success(monkeypatch):
    async def broken_breath(**_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(server, "breath", broken_breath)
    response = await server.lmc5_recall_hook(
        _Request({"schema_version": 1, "prompt": "找一段记忆"})
    )
    assert response.status_code == 500
    assert _json(response) == {"error": "recall failed"}


@pytest.mark.asyncio
async def test_recall_hook_rejects_breath_failure_sentinel(monkeypatch):
    async def failed_breath(**_kwargs):
        return "检索过程出错，请稍后重试。"

    monkeypatch.setattr(server, "breath", failed_breath)
    response = await server.lmc5_recall_hook(
        _Request({"schema_version": 1, "prompt": "之前发生了什么"})
    )

    assert response.status_code == 500
    assert _json(response) == {"error": "recall failed"}


@pytest.mark.asyncio
async def test_recall_hook_never_turns_vector_outage_into_empty_success(
    monkeypatch,
):
    async def no_op():
        return None

    async def no_keyword_matches(*_args, **_kwargs):
        return []

    async def vector_unavailable(*_args, **_kwargs):
        raise RuntimeError("vector backend unavailable")

    monkeypatch.setattr(server.decay_engine, "ensure_started", no_op)
    monkeypatch.setattr(server.consolidation_engine, "ensure_started", no_op)
    monkeypatch.setattr(server.episode_engine, "ensure_started", no_op)
    monkeypatch.setattr(server, "_maybe_start_backfill", lambda: None)
    monkeypatch.setattr(server.bucket_mgr, "search", no_keyword_matches)
    monkeypatch.setattr(
        server.embedding_engine,
        "search_similar",
        vector_unavailable,
    )
    monkeypatch.setitem(
        server.config,
        "query_expansion",
        {"enabled": False},
    )

    response = await server.lmc5_recall_hook(
        _Request({"schema_version": 1, "prompt": "以前发生了什么"})
    )

    assert response.status_code == 500
    assert _json(response) == {"error": "recall failed"}
