import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import subprocess

import pytest

from tools import retrieval_hints_batch as cli


@pytest.fixture
def night(monkeypatch):
    class Clock:
        @staticmethod
        def now(tz):
            return datetime(2026, 9, 18, 1, 10, tzinfo=tz)
    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setenv("OMBRE_RETRIEVAL_HINTS_ENABLED", "1")


@pytest.mark.asyncio
async def test_missing_busy_probe_stops_before_binding_or_storage(night, monkeypatch):
    monkeypatch.setattr(cli, "provider_config", lambda: pytest.fail("must not load credentials"))
    with pytest.raises(ValueError, match="busy_probe_required"):
        await cli.run(argparse.Namespace())


@pytest.mark.asyncio
async def test_actual_busy_probe_defers_before_credentials(night, monkeypatch):
    monkeypatch.setattr(cli, "provider_config", lambda: pytest.fail("must not load credentials"))
    assert await cli.run(argparse.Namespace(), busy_probe=lambda: True) == {
        "status": "busy", "outbound_calls": 0}


@pytest.mark.parametrize("output", ['{}', '{"busy":null}', '{"busy":"false"}', 'not json'])
def test_probe_does_not_interpret_unknown_as_idle(monkeypatch, output):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, output, ""))
    with pytest.raises(ValueError):
        cli.command_busy_probe(["/fixture/read-chat-state"] )()


def test_probe_command_is_bounded_shell_free_and_live(monkeypatch):
    seen = []
    def execute(command, **kw):
        seen.append((command, kw))
        return subprocess.CompletedProcess(command, 0, '{"busy":true}', "")
    monkeypatch.setattr(subprocess, "run", execute)
    assert cli.command_busy_probe(["/fixture/read-chat-state", "--json"])() is True
    assert len(seen) == 1 and seen[0][1]["timeout"] == 3
    assert not seen[0][1].get("shell", False)


@pytest.mark.asyncio
async def test_cli_wires_schema2_source_audit_and_busy_probe(night, monkeypatch, tmp_path):
    import httpx
    import psycopg
    import maintenance_barrier
    monkeypatch.setenv("OMBRE_RETRIEVAL_ATTRIBUTION_ENABLED", "1")
    monkeypatch.setenv("OMBRE_PG_RECALL_DSN", "fixture")
    monkeypatch.setattr(cli, "provider_config", lambda: ("https://fixture.invalid/v1", "fixture", "deepseek-chat"))
    monkeypatch.setattr(cli, "existing_host_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(maintenance_barrier, "MaintenanceBarrier", lambda *a: argparse.Namespace(exclusive=nullcontext))
    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: nullcontext(object()))
    monkeypatch.setattr(cli, "PostgresHints", lambda conn: argparse.Namespace(initialize=lambda: None))
    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    inputs = tmp_path / "sources.json"
    inputs.write_text(json.dumps([{"bucket_id": "fixture", "source_path": "/unused"}]))
    audit = tmp_path / "sources.private.json"
    events = [{"event_id": "exact-id", "content": "synthetic original source"}]
    audit.write_text(json.dumps([{"bucket_id": "fixture", "resolved_events": events}]))
    captured = {}
    class Batch:
        def __init__(self, **kw):
            captured.update(kw)
        async def run(self, approved):
            return {"status": "pass_finished", "outbound_calls": 0}
    monkeypatch.setattr(cli, "NightBatch", Batch)
    busy = lambda: False
    args = argparse.Namespace(lock_device=1, lock_inode=1, approved_sources=inputs,
        buckets_dir=tmp_path, host_lock="/unused", limit=200, source_contexts=audit)
    assert (await cli.run(args, busy_probe=busy))["outbound_calls"] == 0
    assert captured["generator"].schema_version == 2
    assert captured["busy"]() is False
    assert captured["load_source_context"]({"id": "fixture"}) == events
    assert captured["load_source_context"]({"id": "missing"}) == ()
