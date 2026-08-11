"""Production runtime boundary for the bounded LMC-5 night job.

The runtime exposes the production five-axis contract:

* snapshot the live vault;
* consolidate exact raw events and apply provenance-bound X writes;
* write safe Y edges and queue dangerous Y edges for named review;
* queue registered Z current/history pairs for explicit approval;
* queue non-authoritative E proposals for primary-agent authorship;
* compute M in ``report_only`` mode;
* fence one logical run per Asia/Shanghai calendar date.
"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
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
from review_queue import ReviewQueue
from night_run_coordinator import (
    NightRunCoordinator,
    NightRunCoordinatorError,
    NightRunOutcome,
    NightRunPolicy,
)
from snapshot_manager import SnapshotManager


DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_SCHEDULE_TIME = time(hour=4, minute=30)
DEFAULT_PROPOSER_MAX_TOKENS = 4096
DEFAULT_PROPOSER_TEMPERATURE = 0.0
DEFAULT_PROPOSER_DISABLE_THINKING = False
DEFAULT_PROPOSER_JSON_OBJECT = False
DEFAULT_PROPOSER_MAX_CHUNKS_PER_RUN = 16
DEFAULT_PROPOSER_WALL_BUDGET_SECONDS = 3000
MIN_PROPOSER_MAX_TOKENS = 512
MAX_PROPOSER_MAX_TOKENS = 8192
MAX_NIGHT_ATTEMPTS = 32


class NightRunRuntimeError(RuntimeError):
    """A bounded machine-readable production runtime failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _proposer_max_tokens(config: dict[str, Any]) -> int:
    section = config.get("lmc5_night", {})
    if section is None:
        section = {}
    if type(section) is not dict:
        raise NightRunRuntimeError("provider.max_tokens_invalid")
    value = section.get(
        "proposer_max_tokens",
        DEFAULT_PROPOSER_MAX_TOKENS,
    )
    if (
        type(value) is not int
        or not MIN_PROPOSER_MAX_TOKENS
        <= value
        <= MAX_PROPOSER_MAX_TOKENS
    ):
        raise NightRunRuntimeError("provider.max_tokens_invalid")
    return value


def _proposer_temperature(config: dict[str, Any]) -> float:
    section = config.get("lmc5_night", {})
    if section is None:
        section = {}
    if type(section) is not dict:
        raise NightRunRuntimeError("provider.temperature_invalid")
    value = section.get(
        "proposer_temperature",
        DEFAULT_PROPOSER_TEMPERATURE,
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise NightRunRuntimeError("provider.temperature_invalid")
    return float(value)


def _proposer_disable_thinking(config: dict[str, Any]) -> bool:
    section = config.get("lmc5_night", {})
    if section is None:
        section = {}
    if type(section) is not dict:
        raise NightRunRuntimeError("provider.disable_thinking_invalid")
    value = section.get(
        "proposer_disable_thinking",
        DEFAULT_PROPOSER_DISABLE_THINKING,
    )
    if type(value) is not bool:
        raise NightRunRuntimeError("provider.disable_thinking_invalid")
    return value


def _proposer_json_object(config: dict[str, Any]) -> bool:
    section = config.get("lmc5_night", {})
    if section is None:
        section = {}
    if type(section) is not dict:
        raise NightRunRuntimeError("provider.json_object_invalid")
    value = section.get(
        "proposer_json_object",
        DEFAULT_PROPOSER_JSON_OBJECT,
    )
    if type(value) is not bool:
        raise NightRunRuntimeError("provider.json_object_invalid")
    return value


def _proposer_max_chunks_per_run(config: dict[str, Any]) -> int:
    section = config.get("lmc5_night", {})
    if section is None:
        section = {}
    if type(section) is not dict:
        raise NightRunRuntimeError("proposer.chunk_cap_invalid")
    value = section.get(
        "proposer_max_chunks_per_run",
        DEFAULT_PROPOSER_MAX_CHUNKS_PER_RUN,
    )
    if type(value) is not int or not 1 <= value <= 256:
        raise NightRunRuntimeError("proposer.chunk_cap_invalid")
    return value


def _proposer_wall_budget_seconds(config: dict[str, Any]) -> int:
    section = config.get("lmc5_night", {})
    if section is None:
        section = {}
    if type(section) is not dict:
        raise NightRunRuntimeError("proposer.wall_budget_invalid")
    value = section.get(
        "proposer_wall_budget_seconds",
        DEFAULT_PROPOSER_WALL_BUDGET_SECONDS,
    )
    if type(value) is not int or not 1 <= value < 3600:
        raise NightRunRuntimeError("proposer.wall_budget_invalid")
    return value


def _night_max_attempts(config: dict[str, Any]) -> int:
    section = config.get("lmc5_night", {})
    if section is None:
        section = {}
    if type(section) is not dict:
        raise NightRunRuntimeError("run.max_attempts_invalid")
    value = section.get("max_attempts_per_logical_day", DEFAULT_MAX_ATTEMPTS)
    if type(value) is not int or not 1 <= value <= MAX_NIGHT_ATTEMPTS:
        raise NightRunRuntimeError("run.max_attempts_invalid")
    return value


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
        disable_thinking: bool = False,
        json_object: bool = False,
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise NightRunRuntimeError("provider.unconfigured")
        if not isinstance(base_url, str) or not base_url.strip():
            raise NightRunRuntimeError("provider.base_url_invalid")
        if not isinstance(model, str) or not model.strip():
            raise NightRunRuntimeError("provider.model_invalid")
        if (
            type(max_tokens) is not int
            or not MIN_PROPOSER_MAX_TOKENS
            <= max_tokens
            <= MAX_PROPOSER_MAX_TOKENS
        ):
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
        if type(disable_thinking) is not bool:
            raise NightRunRuntimeError("provider.disable_thinking_invalid")
        if type(json_object) is not bool:
            raise NightRunRuntimeError("provider.json_object_invalid")

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
        self.disable_thinking = disable_thinking
        self.json_object = json_object

    def __call__(self, prompt: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.disable_thinking:
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        if self.json_object:
            request["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**request)
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
            if existing.stage == "deferred":
                # A bounded proposer slice ended truthfully with backlog left.
                # It is a terminal historical result, not an interrupted run:
                # preserve it and allocate the next same-day retry id.
                continue
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
    max_tokens = _proposer_max_tokens(config)
    temperature = _proposer_temperature(config)
    disable_thinking = _proposer_disable_thinking(config)
    json_object = _proposer_json_object(config)
    max_chunks_per_run = _proposer_max_chunks_per_run(config)
    wall_budget_seconds = _proposer_wall_budget_seconds(config)
    max_attempts = _night_max_attempts(config)
    provider_timeout = 75.0
    provider = OpenAIChatProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=provider_timeout,
        disable_thinking=disable_thinking,
        json_object=json_object,
    )
    proposer = StrictOmbreProposer(
        provider,
        timeout_seconds=provider_timeout + 5.0,
        model=model,
        provider_name="openai-compatible",
    )
    snapshots = SnapshotManager(config["buckets_dir"], snapshot_root)
    curated = CuratedWriteCoordinator(bucket_manager, embedding_engine)

    async def relation_targets(text: str) -> frozenset[str]:
        target_ids: list[str] = []
        seen: set[str] = set()
        try:
            for bucket_id, _score in await embedding_engine.search_similar(
                text, top_k=12
            ):
                normalized = str(bucket_id or "").strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    target_ids.append(normalized)
        except Exception:
            pass
        try:
            for bucket in await bucket_manager.search(text, limit=8):
                normalized = str(bucket.get("id") or "").strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    target_ids.append(normalized)
        except Exception:
            pass
        verified: list[str] = []
        for bucket_id in target_ids[:16]:
            if await bucket_manager.get(bucket_id) is not None:
                verified.append(bucket_id)
        return frozenset(verified)

    vault_root = Path(config["buckets_dir"])
    review_queue = ReviewQueue(
        vault_root / "review_queue.jsonl",
        maintenance_root=vault_root,
    )
    fact_slot_section = config.get("fact_slots", {}) or {}
    fact_slot_registry = (
        fact_slot_section.get("registry", {})
        if isinstance(fact_slot_section, dict)
        else {}
    )
    coordinator = NightRunCoordinator(
        ledger=ledger,
        snapshots=snapshots,
        proposer=proposer,
        curated=curated,
        decay_engine=decay_engine,
        consolidation_engine=consolidation_engine,
        bucket_manager=bucket_manager,
        review_queue=review_queue,
        fact_slot_registry=fact_slot_registry,
        relation_target_provider=relation_targets,
        policy=NightRunPolicy(
            proposer_max_chunks_per_run=max_chunks_per_run,
            proposer_wall_budget_seconds=wall_budget_seconds,
        ),
    )
    return NightRunRuntime(
        coordinator=coordinator,
        ledger=ledger,
        max_attempts=max_attempts,
    )


__all__ = [
    "NightRunRuntime",
    "NightRunRuntimeError",
    "OpenAIChatProvider",
    "ScheduledNightResult",
    "build_night_run_runtime",
]
