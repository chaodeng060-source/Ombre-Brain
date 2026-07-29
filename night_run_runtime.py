"""Production runtime boundary for the conservative LMC-5 night job.

This module deliberately exposes only the Stage-1 contract:

* snapshot the live vault;
* consolidate exact raw events and apply safe X writes;
* compute M in ``report_only`` mode;
* leave Y/Z/E explicitly deferred;
* fence one logical run per Asia/Shanghai calendar date.

It is not the full five-axis dream pipeline and must not be presented as one.
"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from curated_writer import CuratedWriteCoordinator
from lmc5_ledger import (
    LMC5Ledger,
    LedgerStateError,
    NightRunResult,
)
from lmc5_proposer import StrictOmbreProposer
from maintenance_barrier import MaintenanceBarrierTimeout
from night_run_coordinator import (
    NightRunCoordinator,
    NightRunCoordinatorError,
    NightRunOutcome,
)
from snapshot_manager import SnapshotManager


DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_SCHEDULE_TIME = time(hour=4, minute=30)


class NightRunRuntimeError(RuntimeError):
    """A bounded machine-readable production runtime failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ScheduledNightResult:
    run_id: str
    local_date: str
    stage: str
    already_complete: bool
    cutoff_utc: str
    snapshot_manifest_sha256: str
    counts: dict[str, int]


class OpenAIChatProvider:
    """Synchronous OpenAI-compatible adapter for ``StrictOmbreProposer``."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout_seconds: float,
        client: Any | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise NightRunRuntimeError("provider.unconfigured")
        if not isinstance(base_url, str) or not base_url.strip():
            raise NightRunRuntimeError("provider.base_url_invalid")
        if not isinstance(model, str) or not model.strip():
            raise NightRunRuntimeError("provider.model_invalid")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise NightRunRuntimeError("provider.max_tokens_invalid")
        if isinstance(temperature, bool) or not isinstance(
            temperature, (int, float)
        ) or not math.isfinite(float(temperature)):
            raise NightRunRuntimeError("provider.temperature_invalid")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or float(timeout_seconds) <= 0
        ):
            raise NightRunRuntimeError("provider.timeout_invalid")

        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url.strip(),
                timeout=float(timeout_seconds),
            )
        self._client = client
        self.model = model.strip()
        self.max_tokens = max_tokens
        self.temperature = float(temperature)

    def __call__(self, prompt: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        dump = getattr(response, "model_dump", None)
        if not callable(dump):
            raise RuntimeError("provider response cannot be serialized")
        envelope = dump(mode="json")
        if type(envelope) is not dict:
            raise RuntimeError("provider response is not an object")
        return envelope


class NightRunRuntime:
    """Single-flight daily scheduler boundary around one coordinator."""

    def __init__(
        self,
        *,
        coordinator: NightRunCoordinator,
        ledger: LMC5Ledger,
        timezone_name: str = DEFAULT_TIMEZONE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        try:
            self.timezone = ZoneInfo(timezone_name)
        except Exception as exc:
            raise ValueError("timezone_name must be an IANA timezone") from exc
        if type(max_attempts) is not int or not 1 <= max_attempts <= 32:
            raise ValueError("max_attempts must be a plain integer in 1..32")
        self.coordinator = coordinator
        self.ledger = ledger
        self.max_attempts = max_attempts
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = asyncio.Lock()

    async def run_once(self) -> ScheduledNightResult:
        if self._lock.locked():
            raise NightRunRuntimeError("run.busy")
        async with self._lock:
            try:
                async with self.coordinator.maintenance_barrier.exclusive_async(
                    timeout=float(
                        self.coordinator.policy.barrier_timeout_seconds
                    )
                ):
                    return await self._run_once_locked()
            except MaintenanceBarrierTimeout as exc:
                raise NightRunRuntimeError("run.busy") from exc

    async def _run_once_locked(self) -> ScheduledNightResult:
        now = self.clock()
        if (
            not isinstance(now, datetime)
            or now.tzinfo is None
            or now.utcoffset() is None
        ):
            raise NightRunRuntimeError("clock.invalid")
        local_date, cutoff_utc = self._logical_window(now)
        base_run_id = "lmc5-night-" + local_date.replace("-", "")

        selected_run_id = ""
        for attempt in range(1, self.max_attempts + 1):
            run_id = (
                base_run_id
                if attempt == 1
                else f"{base_run_id}-r{attempt}"
            )
            existing = self._existing(run_id)
            if existing is None:
                selected_run_id = run_id
                break
            if existing.stage == "complete":
                return ScheduledNightResult(
                    run_id=existing.run_id,
                    local_date=local_date,
                    stage=existing.stage,
                    already_complete=True,
                    cutoff_utc=cutoff_utc.isoformat(
                        timespec="microseconds"
                    ),
                    snapshot_manifest_sha256="",
                    counts=dict(existing.counts),
                )
            if existing.stage not in {"error", "rolled_back"}:
                self.ledger.record_night_stage(
                    existing.run_id,
                    "error",
                    counts=existing.counts,
                    errors=("run.interrupted",),
                    expected_stage=existing.stage,
                )
        if not selected_run_id:
            raise NightRunRuntimeError("run.attempts_exhausted")

        try:
            outcome = await self.coordinator.run(
                run_id=selected_run_id,
                cutoff=cutoff_utc,
            )
        except NightRunCoordinatorError as exc:
            if exc.code == "run.reused":
                raise NightRunRuntimeError("run.raced") from exc
            raise
        return self._result(local_date, outcome)

    def _logical_window(self, now: datetime) -> tuple[str, datetime]:
        """Return the stable logical date and cutoff for one daily window.

        A logical night starts at the configured production schedule boundary
        (04:30 Asia/Shanghai by default), not at civil midnight.  Deriving the
        cutoff from that boundary instead of the trigger time guarantees that
        every bounded retry for the same logical night sees the exact same
        input set.
        """

        local_now = now.astimezone(self.timezone)
        logical_day = local_now.date()
        boundary = datetime.combine(
            logical_day,
            DEFAULT_SCHEDULE_TIME,
            tzinfo=self.timezone,
        )
        if local_now < boundary:
            logical_day -= timedelta(days=1)
            boundary = datetime.combine(
                logical_day,
                DEFAULT_SCHEDULE_TIME,
                tzinfo=self.timezone,
            )
        return (
            logical_day.isoformat(),
            boundary.astimezone(timezone.utc),
        )

    def _existing(self, run_id: str) -> NightRunResult | None:
        try:
            return self.ledger.get_night_run(run_id)
        except LedgerStateError:
            return None

    @staticmethod
    def _result(
        local_date: str,
        outcome: NightRunOutcome,
    ) -> ScheduledNightResult:
        return ScheduledNightResult(
            run_id=outcome.run.run_id,
            local_date=local_date,
            stage=outcome.run.stage,
            already_complete=False,
            cutoff_utc=outcome.cutoff_utc,
            snapshot_manifest_sha256=outcome.snapshot_manifest_sha256,
            counts=dict(outcome.counts),
        )


def build_night_run_runtime(
    *,
    config: dict[str, Any],
    ledger: LMC5Ledger,
    bucket_manager: Any,
    embedding_engine: Any,
    decay_engine: Any,
    consolidation_engine: Any,
) -> NightRunRuntime:
    """Build the strict production Stage-1 runtime from live components."""

    snapshot_root = os.environ.get("OMBRE_LMC5_SNAPSHOT_DIR", "").strip()
    if not snapshot_root:
        raise NightRunRuntimeError("snapshot.unconfigured")
    if not os.path.isabs(snapshot_root):
        raise NightRunRuntimeError("snapshot.path_not_absolute")

    dehydration = config.get("dehydration", {}) or {}
    api_key = str(dehydration.get("api_key") or "")
    base_url = str(
        dehydration.get("base_url") or "https://api.deepseek.com/v1"
    )
    model = str(dehydration.get("model") or "deepseek-chat")
    max_tokens = dehydration.get("max_tokens", 1024)
    temperature = dehydration.get("temperature", 0.1)
    provider_timeout = 75.0
    provider = OpenAIChatProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=provider_timeout,
    )
    proposer = StrictOmbreProposer(
        provider,
        timeout_seconds=provider_timeout + 5.0,
        model=model,
        provider_name="openai-compatible",
    )
    snapshots = SnapshotManager(config["buckets_dir"], snapshot_root)
    curated = CuratedWriteCoordinator(bucket_manager, embedding_engine)
    coordinator = NightRunCoordinator(
        ledger=ledger,
        snapshots=snapshots,
        proposer=proposer,
        curated=curated,
        decay_engine=decay_engine,
        consolidation_engine=consolidation_engine,
    )
    return NightRunRuntime(coordinator=coordinator, ledger=ledger)


__all__ = [
    "NightRunRuntime",
    "NightRunRuntimeError",
    "OpenAIChatProvider",
    "ScheduledNightResult",
    "build_night_run_runtime",
]
