from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import night_run_runtime as night_runtime_module
import night_run_trigger
from lmc5_ledger import LMC5Ledger, NIGHT_RUN_FORWARD_STAGES
from night_run_coordinator import (
    NightRunCoordinatorError,
    NightRunOutcome,
)
from night_run_runtime import (
    NightRunRuntime,
    NightRunRuntimeError,
    OpenAIChatProvider,
    _proposer_max_chunks_per_run,
    _proposer_max_tokens,
    _proposer_temperature,
    _proposer_wall_budget_seconds,
    build_night_run_runtime,
)
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


class _DeferredCoordinator(_CompletingCoordinator):
    async def run(
        self,
        *,
        run_id: str,
        cutoff: datetime,
    ) -> NightRunOutcome:
        self.calls.append((run_id, cutoff))
        current = self.ledger.start_night_run(run_id, run_id, counts={})
        for stage in NIGHT_RUN_FORWARD_STAGES[1:-1]:
            current = self.ledger.record_night_stage(
                run_id,
                stage,
                counts={"proposer_pending_after": 1},
                expected_stage=current.stage,
            )
        current = self.ledger.record_night_stage(
            run_id,
            "deferred",
            counts={"proposer_pending_after": 1},
            expected_stage=current.stage,
        )
        return NightRunOutcome(
            run=current,
            snapshot_manifest_sha256="b" * 64,
            cutoff_utc=cutoff.isoformat(timespec="microseconds"),
            counts={"proposer_pending_after": 1},
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
async def test_deferred_daily_attempt_is_preserved_before_fresh_retry(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    now = datetime(2026, 7, 29, 20, 30, tzinfo=timezone.utc)
    first_coordinator = _DeferredCoordinator(ledger)
    first_runtime = NightRunRuntime(
        coordinator=first_coordinator,
        ledger=ledger,
        clock=lambda: now,
    )

    first = await first_runtime.run_once()
    retry_coordinator = _CompletingCoordinator(ledger)
    retry_runtime = NightRunRuntime(
        coordinator=retry_coordinator,
        ledger=ledger,
        clock=lambda: now,
    )
    retry = await retry_runtime.run_once()

    assert first.run_id == "lmc5-night-20260730"
    assert first.stage == "deferred"
    assert first.already_complete is False
    assert retry.run_id == "lmc5-night-20260730-r2"
    assert retry_coordinator.calls[0][1] == first_coordinator.calls[0][1]
    historical = ledger.get_night_run(first.run_id)
    assert historical.stage == "deferred"
    assert historical.errors == ()


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
        max_tokens=4096,
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
    assert captured["max_tokens"] == 4096
    assert captured["temperature"] == 0.1


def test_proposer_budget_is_independent_from_dehydration_budget() -> None:
    assert _proposer_max_tokens(
        {"dehydration": {"max_tokens": 128}}
    ) == 4096
    assert _proposer_max_tokens(
        {
            "dehydration": {"max_tokens": 128},
            "lmc5_night": {"proposer_max_tokens": 6144},
        }
    ) == 6144


@pytest.mark.parametrize("value", (512, 2048, 4096, 8192))
def test_proposer_budget_accepts_plain_integer_boundaries(value: int) -> None:
    assert _proposer_max_tokens(
        {"lmc5_night": {"proposer_max_tokens": value}}
    ) == value


def test_null_night_section_uses_safe_default() -> None:
    assert _proposer_max_tokens({"lmc5_night": None}) == 4096
    assert _proposer_max_chunks_per_run({"lmc5_night": None}) == 16
    assert _proposer_wall_budget_seconds({"lmc5_night": None}) == 3000


def test_proposer_temperature_is_deterministic_and_independent() -> None:
    assert _proposer_temperature(
        {"dehydration": {"temperature": 0.8}}
    ) == 0.0
    assert _proposer_temperature(
        {
            "dehydration": {"temperature": 0.8},
            "lmc5_night": {"proposer_temperature": 0.25},
        }
    ) == 0.25
    assert _proposer_temperature({"lmc5_night": None}) == 0.0


@pytest.mark.parametrize(
    "section",
    (
        [],
        {"proposer_temperature": None},
        {"proposer_temperature": True},
        {"proposer_temperature": False},
        {"proposer_temperature": -0.01},
        {"proposer_temperature": 1.01},
        {"proposer_temperature": float("nan")},
        {"proposer_temperature": float("inf")},
        {"proposer_temperature": "0"},
    ),
)
def test_proposer_temperature_rejects_unsafe_config(
    section: object,
) -> None:
    with pytest.raises(
        NightRunRuntimeError,
        match="^provider\\.temperature_invalid$",
    ):
        _proposer_temperature({"lmc5_night": section})


@pytest.mark.parametrize(
    "section",
    (
        [],
        {"proposer_max_tokens": None},
        {"proposer_max_tokens": True},
        {"proposer_max_tokens": False},
        {"proposer_max_tokens": 0},
        {"proposer_max_tokens": -1},
        {"proposer_max_tokens": 512.0},
        {"proposer_max_tokens": "4096"},
        {"proposer_max_tokens": 511},
        {"proposer_max_tokens": 8193},
    ),
)
def test_proposer_budget_rejects_unsafe_config(section: object) -> None:
    with pytest.raises(
        NightRunRuntimeError,
        match="^provider\\.max_tokens_invalid$",
    ):
        _proposer_max_tokens({"lmc5_night": section})


@pytest.mark.parametrize(
    "value",
    (None, True, False, 0, -1, 17, 16.0, "16"),
)
def test_proposer_chunk_cap_rejects_unsafe_config(value: object) -> None:
    with pytest.raises(
        NightRunRuntimeError,
        match="^proposer\\.chunk_cap_invalid$",
    ):
        _proposer_max_chunks_per_run(
            {"lmc5_night": {"proposer_max_chunks_per_run": value}}
        )


@pytest.mark.parametrize(
    "value",
    (None, True, False, 0, -1, 3600, 3000.0, "3000"),
)
def test_proposer_wall_budget_rejects_unsafe_config(value: object) -> None:
    with pytest.raises(
        NightRunRuntimeError,
        match="^proposer\\.wall_budget_invalid$",
    ):
        _proposer_wall_budget_seconds(
            {"lmc5_night": {"proposer_wall_budget_seconds": value}}
        )


@pytest.mark.parametrize("value", (None, True, False, 0, -1, 511, 8193, 512.0, "512"))
def test_openai_provider_rejects_budget_outside_contract(value: object) -> None:
    with pytest.raises(
        NightRunRuntimeError,
        match="^provider\\.max_tokens_invalid$",
    ):
        OpenAIChatProvider(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            max_tokens=value,  # type: ignore[arg-type]
            temperature=0.1,
            timeout_seconds=5,
            client=object(),
        )


@pytest.mark.parametrize(
    (
        "night_section",
        "expected_tokens",
        "expected_temperature",
        "expected_chunk_cap",
        "expected_wall_budget",
    ),
    (
        (None, 4096, 0.0, 16, 3000),
        (
            {
                "proposer_max_tokens": 2048,
                "proposer_temperature": 0.2,
                "proposer_max_chunks_per_run": 7,
                "proposer_wall_budget_seconds": 900,
            },
            2048,
            0.2,
            7,
            900,
        ),
    ),
)
def test_runtime_builder_wires_dedicated_proposer_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    night_section: dict[str, int | float] | None,
    expected_tokens: int,
    expected_temperature: float,
    expected_chunk_cap: int,
    expected_wall_budget: int,
) -> None:
    captured: dict[str, Any] = {}

    class ProviderSpy:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def __call__(self, prompt: str) -> dict[str, Any]:
            return {}

    class CoordinatorStub:
        def __init__(self, **kwargs: Any) -> None:
            captured["policy"] = kwargs["policy"]
            self.maintenance_barrier = object()
            self.policy = SimpleNamespace(barrier_timeout_seconds=1.0)

    monkeypatch.setenv(
        "OMBRE_LMC5_SNAPSHOT_DIR",
        str(tmp_path / "snapshots"),
    )
    monkeypatch.setattr(
        night_runtime_module,
        "OpenAIChatProvider",
        ProviderSpy,
    )
    monkeypatch.setattr(
        night_runtime_module,
        "StrictOmbreProposer",
        lambda provider, **_: provider,
    )
    monkeypatch.setattr(
        night_runtime_module,
        "SnapshotManager",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        night_runtime_module,
        "CuratedWriteCoordinator",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        night_runtime_module,
        "NightRunCoordinator",
        CoordinatorStub,
    )

    config: dict[str, Any] = {
        "buckets_dir": str(tmp_path / "vault"),
        "dehydration": {
            "api_key": "test-key",
            "model": "test-model",
            "max_tokens": 777,
        },
    }
    if night_section is not None:
        config["lmc5_night"] = night_section

    runtime = build_night_run_runtime(
        config=config,
        ledger=_ledger(tmp_path),
        bucket_manager=object(),
        embedding_engine=object(),
        decay_engine=object(),
        consolidation_engine=object(),
    )

    assert isinstance(runtime, NightRunRuntime)
    assert captured["max_tokens"] == expected_tokens
    assert captured["temperature"] == expected_temperature
    assert captured["policy"].proposer_max_chunks_per_run == expected_chunk_cap
    assert (
        captured["policy"].proposer_wall_budget_seconds
        == expected_wall_budget
    )


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


def test_trigger_summary_and_cli_accept_truthful_deferred(
    monkeypatch,
    capsys,
) -> None:
    payload = {
        "ok": True,
        "contract": "lmc5-conservative-stage1",
        "run_id": "night-1",
        "stage": "deferred",
        "already_complete": False,
        "complete": False,
        "degraded": True,
        "counts": {"proposer_pending_after": 12},
    }

    assert _safe_summary(payload) == payload
    monkeypatch.setattr(night_run_trigger, "trigger", lambda: payload)

    assert night_run_trigger.main() == 0
    assert json.loads(capsys.readouterr().out) == payload


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
