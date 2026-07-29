from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import night_run_trigger
from lmc5_ledger import LMC5Ledger, NIGHT_RUN_FORWARD_STAGES
from night_run_coordinator import (
    NightRunCoordinatorError,
    NightRunOutcome,
)
from night_run_runtime import NightRunRuntime, OpenAIChatProvider
from night_run_trigger import (
    NightTriggerHTTPError,
    _safe_summary,
)


class _CompletingCoordinator:
    def __init__(self, ledger: LMC5Ledger) -> None:
        self.ledger = ledger
        self.maintenance_barrier = ledger._maintenance_barrier
        self.policy = SimpleNamespace(barrier_timeout_seconds=1.0)
        self.calls: list[tuple[str, datetime]] = []

    async def run(
        self,
        *,
        run_id: str,
        cutoff: datetime,
    ) -> NightRunOutcome:
        self.calls.append((run_id, cutoff))
        current = self.ledger.start_night_run(run_id, run_id, counts={})
        for stage in NIGHT_RUN_FORWARD_STAGES[1:]:
            current = self.ledger.record_night_stage(
                run_id,
                stage,
                counts={"processed": 1},
                expected_stage=current.stage,
            )
        return NightRunOutcome(
            run=current,
            snapshot_manifest_sha256="a" * 64,
            cutoff_utc=cutoff.isoformat(timespec="microseconds"),
            counts={"processed": 1},
        )


def _ledger(tmp_path: Path) -> LMC5Ledger:
    root = tmp_path / "vault"
    root.mkdir()
    return LMC5Ledger(
        root / ".lmc5" / "pipeline.sqlite3",
        maintenance_root=root,
    )


@pytest.mark.asyncio
async def test_daily_run_is_idempotent_after_completion(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    coordinator = _CompletingCoordinator(ledger)
    now = datetime(2026, 7, 29, 20, 30, tzinfo=timezone.utc)
    runtime = NightRunRuntime(
        coordinator=coordinator,
        ledger=ledger,
        clock=lambda: now,
    )

    first = await runtime.run_once()
    second = await runtime.run_once()

    assert first.run_id == "lmc5-night-20260730"
    assert first.already_complete is False
    assert second.run_id == first.run_id
    assert second.already_complete is True
    assert second.cutoff_utc == first.cutoff_utc
    assert len(coordinator.calls) == 1


@pytest.mark.asyncio
async def test_daily_completion_survives_runtime_rebuild(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    now = datetime(2026, 7, 29, 20, 30, tzinfo=timezone.utc)
    first_coordinator = _CompletingCoordinator(ledger)
    first_runtime = NightRunRuntime(
        coordinator=first_coordinator,
        ledger=ledger,
        clock=lambda: now,
    )
    await first_runtime.run_once()

    rebuilt_coordinator = _CompletingCoordinator(ledger)
    rebuilt_runtime = NightRunRuntime(
        coordinator=rebuilt_coordinator,
        ledger=ledger,
        clock=lambda: now,
    )
    replay = await rebuilt_runtime.run_once()

    assert replay.already_complete is True
    assert rebuilt_coordinator.calls == []


@pytest.mark.asyncio
async def test_logical_day_starts_at_0430_shanghai(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    coordinator = _CompletingCoordinator(ledger)
    runtime = NightRunRuntime(
        coordinator=coordinator,
        ledger=ledger,
        clock=lambda: datetime(
            2026,
            7,
            29,
            20,
            29,
            59,
            tzinfo=timezone.utc,
        ),
    )

    result = await runtime.run_once()

    assert result.local_date == "2026-07-29"
    assert result.run_id == "lmc5-night-20260729"
    assert coordinator.calls == [
        (
            "lmc5-night-20260729",
            datetime(2026, 7, 28, 20, 30, tzinfo=timezone.utc),
        )
    ]
    assert result.cutoff_utc == "2026-07-28T20:30:00.000000+00:00"


@pytest.mark.asyncio
async def test_retry_uses_identical_cutoff_after_trigger_time_advances(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)

    class FailingCoordinator(_CompletingCoordinator):
        async def run(
            self,
            *,
            run_id: str,
            cutoff: datetime,
        ) -> NightRunOutcome:
            self.calls.append((run_id, cutoff))
            current = self.ledger.start_night_run(
                run_id,
                run_id,
                counts={},
            )
            self.ledger.record_night_stage(
                run_id,
                "error",
                counts={},
                errors=("injected.failure",),
                expected_stage=current.stage,
            )
            raise NightRunCoordinatorError("injected.failure")

    first_coordinator = FailingCoordinator(ledger)
    first_runtime = NightRunRuntime(
        coordinator=first_coordinator,
        ledger=ledger,
        clock=lambda: datetime(
            2026,
            7,
            29,
            21,
            0,
            tzinfo=timezone.utc,
        ),
    )
    with pytest.raises(
        NightRunCoordinatorError,
        match="injected.failure",
    ):
        await first_runtime.run_once()

    retry_coordinator = _CompletingCoordinator(ledger)
    retry_runtime = NightRunRuntime(
        coordinator=retry_coordinator,
        ledger=ledger,
        clock=lambda: datetime(
            2026,
            7,
            30,
            20,
            29,
            tzinfo=timezone.utc,
        ),
    )
    retry = await retry_runtime.run_once()

    expected_cutoff = datetime(
        2026,
        7,
        29,
        20,
        30,
        tzinfo=timezone.utc,
    )
    assert first_coordinator.calls == [
        ("lmc5-night-20260730", expected_cutoff)
    ]
    assert retry_coordinator.calls == [
        ("lmc5-night-20260730-r2", expected_cutoff)
    ]
    assert retry.cutoff_utc == "2026-07-29T20:30:00.000000+00:00"


@pytest.mark.asyncio
async def test_failed_daily_attempt_uses_new_bounded_run_id(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    base = "lmc5-night-20260730"
    ledger.start_night_run(base, base, counts={})
    ledger.record_night_stage(
        base,
        "error",
        counts={},
        errors=("injected.failure",),
        expected_stage="started",
    )
    coordinator = _CompletingCoordinator(ledger)
    runtime = NightRunRuntime(
        coordinator=coordinator,
        ledger=ledger,
        clock=lambda: datetime(
            2026,
            7,
            29,
            20,
            30,
            tzinfo=timezone.utc,
        ),
    )

    result = await runtime.run_once()

    assert result.run_id == base + "-r2"
    assert coordinator.calls[0][0] == base + "-r2"
    assert ledger.get_night_run(base).stage == "error"


@pytest.mark.asyncio
async def test_interrupted_attempt_is_sealed_before_fresh_retry(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    base = "lmc5-night-20260730"
    ledger.start_night_run(base, base, counts={"raw_events": 1})
    coordinator = _CompletingCoordinator(ledger)
    runtime = NightRunRuntime(
        coordinator=coordinator,
        ledger=ledger,
        clock=lambda: datetime(
            2026,
            7,
            29,
            20,
            30,
            tzinfo=timezone.utc,
        ),
    )

    result = await runtime.run_once()

    interrupted = ledger.get_night_run(base)
    assert interrupted.stage == "error"
    assert interrupted.errors == ("run.interrupted",)
    assert result.run_id == base + "-r2"


@pytest.mark.asyncio
async def test_same_runtime_rejects_overlapping_trigger(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingCoordinator(_CompletingCoordinator):
        async def run(self, *, run_id: str, cutoff: datetime):
            entered.set()
            await release.wait()
            return await super().run(run_id=run_id, cutoff=cutoff)

    coordinator = BlockingCoordinator(ledger)
    runtime = NightRunRuntime(
        coordinator=coordinator,
        ledger=ledger,
        clock=lambda: datetime(
            2026,
            7,
            29,
            20,
            30,
            tzinfo=timezone.utc,
        ),
    )
    first = asyncio.create_task(runtime.run_once())
    await entered.wait()

    from night_run_runtime import NightRunRuntimeError

    with pytest.raises(NightRunRuntimeError) as raised:
        await runtime.run_once()
    assert raised.value.code == "run.busy"

    release.set()
    await first
    assert len(coordinator.calls) == 1


@pytest.mark.asyncio
async def test_two_runtimes_serialize_on_durable_barrier(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    now = datetime(2026, 7, 29, 20, 30, tzinfo=timezone.utc)

    class BlockingCoordinator(_CompletingCoordinator):
        async def run(self, *, run_id: str, cutoff: datetime):
            entered.set()
            await release.wait()
            return await super().run(run_id=run_id, cutoff=cutoff)

    first_coordinator = BlockingCoordinator(ledger)
    second_coordinator = _CompletingCoordinator(ledger)
    first_runtime = NightRunRuntime(
        coordinator=first_coordinator,
        ledger=ledger,
        clock=lambda: now,
    )
    second_runtime = NightRunRuntime(
        coordinator=second_coordinator,
        ledger=ledger,
        clock=lambda: now,
    )
    first = asyncio.create_task(first_runtime.run_once())
    await entered.wait()
    second = asyncio.create_task(second_runtime.run_once())
    await asyncio.sleep(0.02)
    assert not second.done()

    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.already_complete is False
    assert second_result.already_complete is True
    assert second_coordinator.calls == []


def test_openai_provider_returns_plain_json_envelope() -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any):
        captured.update(kwargs)
        return SimpleNamespace(
            model_dump=lambda **_: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "{}"},
                    }
                ]
            }
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    provider = OpenAIChatProvider(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        max_tokens=128,
        temperature=0.1,
        timeout_seconds=5,
        client=client,
    )

    result = provider("strict prompt")

    assert type(result) is dict
    assert captured["messages"] == [
        {"role": "user", "content": "strict prompt"}
    ]
    assert captured["model"] == "test-model"


def test_trigger_summary_strips_snapshot_and_internal_fields() -> None:
    summary = _safe_summary(
        {
            "ok": True,
            "run_id": "night-1",
            "counts": {"x_ready": 1},
            "snapshot_manifest_sha256": "secret-path-adjacent",
            "cutoff_utc": "2026-07-29T00:00:00+00:00",
            "internal": {"token": "must-not-print"},
        }
    )

    assert summary == {
        "ok": True,
        "run_id": "night-1",
        "counts": {"x_ready": 1},
    }


def test_trigger_ignores_proxy_environment_and_uses_origin_form(
    monkeypatch,
) -> None:
    observed: dict[str, Any] = {}

    class Response:
        status = 200

        @staticmethod
        def getheader(name: str, default: str = "") -> str:
            assert name == "Content-Type"
            return "application/json; charset=utf-8"

        @staticmethod
        def read(_limit: int) -> bytes:
            return (
                b'{"ok":true,"contract":"lmc5-conservative-stage1",'
                b'"run_id":"night-1","stage":"complete"}'
            )

    class Connection:
        def __init__(self, host: str, port: int, *, timeout: int) -> None:
            observed["connect"] = (host, port, timeout)

        def request(
            self,
            method: str,
            path: str,
            *,
            body: bytes,
            headers: dict[str, str],
        ) -> None:
            observed["request"] = (method, path, body, headers)

        @staticmethod
        def getresponse() -> Response:
            return Response()

        def close(self) -> None:
            observed["closed"] = True

    monkeypatch.setenv("OMBRE_API_TOKEN", "night-secret-token")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:65534")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:65534")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(night_run_trigger, "HTTPConnection", Connection)

    result = night_run_trigger.trigger()

    assert result["ok"] is True
    assert observed["connect"] == ("127.0.0.1", 8000, 3600)
    method, path, body, headers = observed["request"]
    assert method == "POST"
    assert path == "/api/maintenance/lmc5-night"
    assert not path.startswith("http://")
    assert body == b'{"schema_version":1}'
    assert headers["Authorization"] == "Bearer night-secret-token"
    assert observed["closed"] is True


def test_trigger_rejects_redirect_without_following_it(monkeypatch) -> None:
    calls = 0

    class Response:
        status = 302

        @staticmethod
        def getheader(_name: str, default: str = "") -> str:
            return "text/html"

        @staticmethod
        def read(_limit: int) -> bytes:
            return b"redirect"

    class Connection:
        def __init__(self, *_args, **_kwargs) -> None:
            nonlocal calls
            calls += 1

        @staticmethod
        def request(*_args, **_kwargs) -> None:
            return None

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setenv("OMBRE_API_TOKEN", "night-secret-token")
    monkeypatch.setattr(night_run_trigger, "HTTPConnection", Connection)

    with pytest.raises(NightTriggerHTTPError) as raised:
        night_run_trigger.trigger()

    assert raised.value.status == 302
    assert calls == 1


def test_host_cron_wrapper_has_valid_shell_syntax() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "cron"
        / "run-lmc5-night.sh"
    )

    subprocess.run(["sh", "-n", str(script)], check=True)
