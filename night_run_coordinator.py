"""Fail-closed coordinator for one bounded LMC-5 night run.

The coordinator keeps every axis inside its production authority contract:

* raw events are redacted and durably chunked before leaving local storage;
* each provider call owns one chunk, with bounded independent concurrency;
* normal X drafts may become recall-visible rows with immutable provenance;
* safe Y edges write idempotently while dangerous edges enter named review;
* Z only proposes registered fact pairs; approval is a separate transaction;
* E only proposes source material; a primary agent must author the E record;
* M is a verified ``report_only`` computation and never mutates memory.

An interrupted run is evidence, not a resume token.  A caller must retry with
a fresh ``run_id`` (and, at the scheduler layer, the same UTC cutoff).
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import math
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence

from curated_writer import CuratedWriteCoordinator, CuratedWriteResult
from lmc5_ledger import (
    CandidateRecord,
    EventIdentity,
    LMC5Ledger,
    LedgerError,
    NightRunResult,
    PendingProposerChunk,
    RawEventRecord,
    TERMINAL_NIGHT_STAGES,
)
from lmc5_proposer import (
    CANDIDATE_TYPES,
    CandidateDraft,
    ProposerBatch,
    ProposerChunk,
    ProposerContractError,
    RelationHint,
    StrictOmbreProposer,
    route_candidate_axes,
)
from fact_slots import (
    FACT_STATUS_HISTORICAL,
    extract_registered_facts,
    is_fact_slot_exempt,
    normalize_fact_slot_registry,
)
from review_queue import (
    make_e_proposal_entry,
    make_relation_entry,
    make_z_pair_entry,
)
from redact import redact_obj
from snapshot_manager import SnapshotManager, SnapshotResult
from bucket_manager import bucket_revision_hash
from timeline_axis import run_timeline_sweep


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_SCHEMA = "ombre.lmc5-axis-candidate/v1"
_SNAPSHOT_RECEIPT_SCHEMA = "ombre.lmc5-snapshot-receipt/v1"
_METABOLISM_RECEIPT_SCHEMA = "ombre.lmc5-metabolism-receipt/v1"
_CHUNK_SCHEMA = "ombre.lmc5-redacted-event/v1"
_REDACTION_VERSION = "redact_obj/v1"
_RETRYABLE_DISPATCH_CODES = frozenset(
    {
        "e.source_write_retryable",
        "x.write_retryable",
        "y.source_write_retryable",
        "z.source_write_retryable",
    }
)
_M_RECEIPT_YIELD_EVERY = 16


class NightRunCoordinatorError(RuntimeError):
    """A bounded machine-readable night-run failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class NightRunPolicy:
    raw_page_size: int = 100
    pending_page_size: int = 100
    proposer_max_chunks_per_run: int = 16
    proposer_concurrency: int = 1
    proposer_wall_budget_seconds: int = 3000
    chunk_bytes: int = 24 * 1024
    barrier_timeout_seconds: float = 60.0
    vector_policy: str = "required"

    def __post_init__(self) -> None:
        for field in (
            "raw_page_size",
            "pending_page_size",
            "proposer_max_chunks_per_run",
            "proposer_concurrency",
            "proposer_wall_budget_seconds",
            "chunk_bytes",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.raw_page_size > 1_000 or self.pending_page_size > 1_000:
            raise ValueError("ledger page sizes cannot exceed 1000")
        if self.chunk_bytes > 256 * 1024:
            raise ValueError("chunk_bytes exceeds the proposer input contract")
        if self.proposer_max_chunks_per_run > 256:
            raise ValueError("proposer run chunk cap cannot exceed 256")
        if self.proposer_concurrency > 8:
            raise ValueError("proposer concurrency cannot exceed 8")
        if self.proposer_wall_budget_seconds >= 3600:
            raise ValueError("proposer wall budget must be below 3600 seconds")
        timeout = self.barrier_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError("barrier_timeout_seconds must be finite and positive")
        if self.vector_policy != "required":
            raise ValueError("night writes require the 'required' vector policy")


@dataclass(frozen=True, slots=True)
class NightRunOutcome:
    run: NightRunResult
    snapshot_manifest_sha256: str
    cutoff_utc: str
    counts: Mapping[str, int]


class _DuplicateKey(ValueError):
    pass


class _NonFinite(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_nonfinite(_: str) -> Any:
    raise _NonFinite


def _load_json_object(payload: str | bytes, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateKey,
        _NonFinite,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise NightRunCoordinatorError(code) from exc
    if type(value) is not dict:
        raise NightRunCoordinatorError(code)
    return value


def _canonical_bytes(value: Any, *, code: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise NightRunCoordinatorError(code) from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: Any, *, code: str) -> str:
    return _sha256(_canonical_bytes(value, code=code))


def _normalize_cutoff(value: datetime) -> tuple[datetime, str]:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise NightRunCoordinatorError("cutoff.invalid")
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise NightRunCoordinatorError("cutoff.invalid") from exc
    if offset is None:
        raise NightRunCoordinatorError("cutoff.invalid")
    normalized = value.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if normalized > now:
        raise NightRunCoordinatorError("cutoff.future")
    return normalized, normalized.isoformat(timespec="microseconds")


def _split_utf8(payload: bytes, limit: int) -> tuple[bytes, ...]:
    if not payload:
        raise NightRunCoordinatorError("raw.empty")
    parts: list[bytes] = []
    start = 0
    while start < len(payload):
        end = min(start + limit, len(payload))
        while end > start:
            try:
                payload[start:end].decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                end -= 1
            else:
                break
        if end == start:
            raise NightRunCoordinatorError("raw.utf8")
        parts.append(payload[start:end])
        start = end
    return tuple(parts)


def _relation_json(relation: RelationHint) -> dict[str, Any]:
    return {
        "reason": relation.reason,
        "relation_type": relation.relation_type,
        "strength": relation.strength,
        "target_id": relation.target_id,
    }


def _draft_json(draft: CandidateDraft) -> dict[str, Any]:
    return {
        "content": draft.content,
        "evidence": draft.evidence,
        "importance": draft.importance,
        "relation_hints": [
            _relation_json(relation) for relation in draft.relation_hints
        ],
        "risk": draft.risk,
        "source_chunk_ids": list(draft.source_chunk_ids),
        "thread_hint": draft.thread_hint,
        "title": draft.title,
        "type": draft.type,
    }


async def _await_daemon_thread(function):
    """Await one blocking leaf without retaining asyncio's default executor.

    The coordinator is also exercised by short-lived CLI and test processes.
    A one-shot daemon worker keeps those processes from hanging on executor
    shutdown while still copying the maintenance lease context into the leaf.
    """

    finished = threading.Event()
    outcome: dict[str, Any] = {}
    context = contextvars.copy_context()

    def run() -> None:
        try:
            outcome["value"] = context.run(function)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(
        target=run,
        name="lmc5-night-blocking-leaf",
        daemon=True,
    ).start()
    while not finished.is_set():
        await asyncio.sleep(0.01)
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


class NightRunCoordinator:
    """Run one append-only, snapshot-first night pipeline."""

    def __init__(
        self,
        *,
        ledger: LMC5Ledger,
        snapshots: SnapshotManager,
        proposer: StrictOmbreProposer,
        curated: CuratedWriteCoordinator,
        decay_engine: Any,
        consolidation_engine: Any,
        bucket_manager: Any | None = None,
        review_queue: Any | None = None,
        fact_slot_registry: Mapping[str, Any] | None = None,
        relation_target_provider: (
            Callable[[str], Awaitable[frozenset[str]]] | None
        ) = None,
        policy: NightRunPolicy | None = None,
    ) -> None:
        self.ledger = ledger
        self.snapshots = snapshots
        self.proposer = proposer
        self.curated = curated
        self.decay_engine = decay_engine
        self.consolidation_engine = consolidation_engine
        self.bucket_manager = bucket_manager
        self.review_queue = review_queue
        self.fact_slot_registry = normalize_fact_slot_registry(
            fact_slot_registry
        )
        self._fact_slot_config = dict(fact_slot_registry or {})
        self.relation_target_provider = relation_target_provider
        self.policy = policy or NightRunPolicy()
        self._barrier = snapshots.maintenance_barrier
        self._assert_contract()

    @property
    def maintenance_barrier(self):
        return self._barrier

    def _assert_contract(self) -> None:
        barriers = [
            getattr(self.ledger, "_maintenance_barrier", None),
            getattr(self.snapshots, "maintenance_barrier", None),
            getattr(self.curated, "_maintenance_barrier", None),
            getattr(
                getattr(self.decay_engine, "bucket_mgr", None),
                "_maintenance_barrier",
                None,
            ),
            getattr(
                getattr(self.consolidation_engine, "bucket_mgr", None),
                "_maintenance_barrier",
                None,
            ),
        ]
        for component in (self.bucket_manager, self.review_queue):
            if component is not None:
                barriers.append(getattr(component, "_maintenance_barrier", None))
        if any(barrier is None for barrier in barriers):
            raise ValueError("all night components need one maintenance barrier")
        lock_paths = {str(barrier.lock_path) for barrier in barriers}
        if len(lock_paths) != 1:
            raise ValueError("night components do not share one maintenance barrier")
        if getattr(self.decay_engine, "metabolism_mode", None) != "report_only":
            raise ValueError("decay engine must be report_only")
        if (
            getattr(self.consolidation_engine, "metabolism_mode", None)
            != "report_only"
        ):
            raise ValueError("consolidation engine must be report_only")

    async def run(self, *, run_id: str, cutoff: datetime) -> NightRunOutcome:
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise NightRunCoordinatorError("run.invalid_id")
        _, cutoff_iso = _normalize_cutoff(cutoff)
        self._assert_report_only()
        counts: dict[str, int] = {}
        started = False
        try:
            async with self._barrier.exclusive_async(
                timeout=float(self.policy.barrier_timeout_seconds)
            ):
                run = self.ledger.start_night_run(run_id, run_id, counts=counts)
                if not run.created:
                    raise NightRunCoordinatorError("run.reused")
                started = True

                snapshot = await _await_daemon_thread(
                    lambda: self.snapshots.create_snapshot(run_id)
                )
                self._seal_snapshot(run_id, cutoff_iso, snapshot, counts)

                self._chunk_uncovered(cutoff_iso, counts)
                self._advance(run_id, "snapshotted", "chunked", counts)

                proposer_watermark = self.ledger.proposer_watermark()
                counts["proposer_watermark"] = proposer_watermark
                await self._propose_pending(
                    run_id,
                    counts,
                    watermark=proposer_watermark,
                )
                self._advance(run_id, "chunked", "proposed", counts)

                await self._dispatch_pending(counts)
                self._advance(run_id, "proposed", "dispatched", counts)

                await self._run_timeline_sweep(counts)

                await self._run_metabolism(counts)
                self._advance(
                    run_id,
                    "dispatched",
                    "metabolism_reported",
                    counts,
                )

                await self._validate(
                    run_id=run_id,
                    cutoff_iso=cutoff_iso,
                    snapshot=snapshot,
                    counts=counts,
                )
                self._advance(
                    run_id,
                    "metabolism_reported",
                    "validated",
                    counts,
                )
                terminal_stage = (
                    "deferred"
                    if (
                        counts["proposer_pending_after"] > 0
                        or counts["dispatch_pending_after"] > 0
                    )
                    else "complete"
                )
                completed = self._advance(
                    run_id, "validated", terminal_stage, counts
                )
                return NightRunOutcome(
                    run=completed,
                    snapshot_manifest_sha256=snapshot.manifest_sha256,
                    cutoff_utc=cutoff_iso,
                    counts=dict(completed.counts),
                )
        except asyncio.CancelledError:
            if started:
                self._mark_error(run_id, "run.cancelled", counts)
            raise
        except NightRunCoordinatorError as exc:
            if started and exc.code != "run.reused":
                self._mark_error(run_id, exc.code, counts)
            raise
        except Exception as exc:
            if started:
                self._mark_error(run_id, "run.internal", counts)
            raise NightRunCoordinatorError("run.internal") from exc

    def _assert_report_only(self) -> None:
        if getattr(self.decay_engine, "metabolism_mode", None) != "report_only":
            raise NightRunCoordinatorError("metabolism.decay_mode_unsafe")
        if (
            getattr(self.consolidation_engine, "metabolism_mode", None)
            != "report_only"
        ):
            raise NightRunCoordinatorError(
                "metabolism.consolidation_mode_unsafe"
            )

    async def _run_timeline_sweep(self, counts: dict[str, int]) -> None:
        """Backfill X after snapshot+dispatch and before report-only M."""

        if self.bucket_manager is None:
            counts.update({
                "timeline_scanned": 0,
                "timeline_assigned": 0,
                "timeline_named": 0,
                "timeline_updated": 0,
                "timeline_new_lines": 0,
                "timeline_orphans": 0,
            })
            return
        report = await run_timeline_sweep(
            self.bucket_manager,
            ledger_path=getattr(self.ledger, "path", None),
            apply=True,
            actor="lmc5:night:timeline-sweep",
            revision_hash_provider=bucket_revision_hash,
        )
        counts.update({
            "timeline_scanned": report.scanned_count,
            "timeline_assigned": report.assigned_count,
            "timeline_named": report.named_count,
            "timeline_updated": report.updated_count,
            "timeline_new_lines": report.new_line_count,
            "timeline_orphans": report.orphan_count,
        })

    def _seal_snapshot(
        self,
        run_id: str,
        cutoff_iso: str,
        snapshot: SnapshotResult,
        counts: dict[str, int],
    ) -> None:
        request = {
            "cutoff_utc": cutoff_iso,
            "schema": _SNAPSHOT_RECEIPT_SCHEMA,
            "snapshot_id": snapshot.snapshot_id,
            "source_root": str(self.snapshots.source_root),
        }
        request_hash = _canonical_digest(
            request, code="snapshot.receipt_invalid"
        )
        with self.ledger.transaction() as tx:
            tx.record_write_receipt(
                f"snapshot:v1:{run_id}",
                request_hash,
                f"snapshot:{run_id}",
                result_hash=snapshot.manifest_sha256,
            )
            tx.record_night_stage(
                run_id,
                "snapshotted",
                counts=counts,
                expected_stage="started",
            )

    def _chunk_uncovered(
        self, cutoff_iso: str, counts: dict[str, int]
    ) -> None:
        after: int | None = None
        while True:
            records = self.ledger.list_uncovered_raw_events(
                limit=self.policy.raw_page_size,
                after=after,
                created_before=cutoff_iso,
            )
            if not records:
                return
            for record in records:
                parts = self._event_parts(record)
                with self.ledger.transaction() as tx:
                    for chunk_id, content in parts:
                        tx.record_event_chunk(
                            chunk_id,
                            content,
                            (record.identity,),
                        )
                counts["raw_events"] = counts.get("raw_events", 0) + 1
                counts["chunks"] = counts.get("chunks", 0) + len(parts)
            after = records[-1].row_id

    def _event_parts(
        self, record: RawEventRecord
    ) -> tuple[tuple[str, bytes], ...]:
        raw = _load_json_object(record.payload, code="raw.invalid_json")
        redacted = redact_obj(raw)
        envelope = {
            "payload": redacted,
            "recorded_at": record.recorded_at,
            "redaction": _REDACTION_VERSION,
            "schema": _CHUNK_SCHEMA,
            "session_id": record.identity.session_id,
            "source_event_id": record.identity.source_event_id,
        }
        canonical = _canonical_bytes(envelope, code="raw.invalid_value")
        split = _split_utf8(canonical, self.policy.chunk_bytes)
        results: list[tuple[str, bytes]] = []
        for ordinal, content in enumerate(split):
            identity = {
                "content_sha256": _sha256(content),
                "ordinal": ordinal,
                "payload_sha256": record.payload_digest,
                "redaction": _REDACTION_VERSION,
                "session_id": record.identity.session_id,
                "source_event_id": record.identity.source_event_id,
            }
            chunk_id = (
                "lmc5-chunk-v1-"
                + _canonical_digest(identity, code="chunk.identity_invalid")
            )
            results.append((chunk_id, content))
        return tuple(results)

    async def _propose_pending(
        self,
        run_id: str,
        counts: dict[str, int],
        *,
        watermark: int,
    ) -> None:
        before = self.ledger.proposer_backlog_stats(through=watermark)
        counts["proposer_pending_before"] = before.pending
        counts["proposer_attempted"] = 0
        counts["proposer_succeeded"] = 0
        counts["proposer_retryable"] = 0
        counts["proposer_circuit_breaker"] = 0
        counts["proposer_wall_budget_exhausted"] = 0

        pending_rows = self.ledger.list_pending_proposer_chunks(
            limit=self.policy.proposer_max_chunks_per_run,
            through=watermark,
            prioritize_retries=True,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.policy.proposer_wall_budget_seconds

        async def propose_one(
            pending: PendingProposerChunk,
            text: str,
            timeout: float,
        ) -> ProposerBatch:
            async def call() -> ProposerBatch:
                relation_targets = frozenset()
                if self.relation_target_provider is not None:
                    relation_targets = await self.relation_target_provider(text)
                    if not isinstance(relation_targets, frozenset):
                        raise NightRunCoordinatorError(
                            "proposer.relation_targets_invalid"
                        )
                return await self.proposer.propose(
                    (ProposerChunk(id=pending.chunk_id, text=text),),
                    relation_targets,
                )

            return await asyncio.wait_for(call(), timeout=timeout)

        decoded: list[tuple[PendingProposerChunk, str]] = []
        for pending in pending_rows:
            if len(pending.source_event_ids) != 1:
                raise NightRunCoordinatorError(
                    "proposer.source_cardinality"
                )
            try:
                text = pending.content.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise NightRunCoordinatorError(
                    "proposer.chunk_utf8"
                ) from exc
            decoded.append((pending, text))

        consecutive_errors = 0
        next_index = 0
        active: dict[
            asyncio.Task[ProposerBatch],
            tuple[int, PendingProposerChunk],
        ] = {}

        def schedule_available() -> None:
            nonlocal next_index
            while (
                next_index < len(decoded)
                and len(active) < self.policy.proposer_concurrency
                and not counts["proposer_circuit_breaker"]
                and not counts["proposer_wall_budget_exhausted"]
            ):
                remaining = deadline - loop.time()
                if remaining <= 0:
                    counts["proposer_wall_budget_exhausted"] = 1
                    return
                pending, text = decoded[next_index]
                task = asyncio.create_task(
                    propose_one(pending, text, remaining)
                )
                active[task] = (next_index, pending)
                next_index += 1
                counts["proposer_attempted"] += 1

        schedule_available()
        try:
            while active:
                done, _pending_tasks = await asyncio.wait(
                    active,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                ordered = sorted(done, key=lambda task: active[task][0])
                for task in ordered:
                    _index, pending = active.pop(task)
                    try:
                        result: ProposerBatch | BaseException = task.result()
                    except BaseException as exc:
                        result = exc
                    if isinstance(result, asyncio.TimeoutError):
                        self._record_proposer_error(
                            run_id,
                            pending,
                            "provider.run_budget",
                        )
                        counts["proposer_retryable"] += 1
                        counts["proposer_errors"] = counts[
                            "proposer_retryable"
                        ]
                        counts["proposer_wall_budget_exhausted"] = 1
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            counts["proposer_circuit_breaker"] = 1
                        continue
                    if isinstance(result, ProposerContractError):
                        self._record_proposer_error(
                            run_id, pending, result.code
                        )
                        counts["proposer_retryable"] += 1
                        counts["proposer_errors"] = counts[
                            "proposer_retryable"
                        ]
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            counts["proposer_circuit_breaker"] = 1
                        continue
                    if isinstance(result, BaseException):
                        raise result
                    consecutive_errors = 0
                    counts["proposer_circuit_breaker"] = 0
                    candidate_specs = self._candidate_specs(
                        run_id=run_id,
                        pending=pending,
                        batch=result,
                    )
                    with self.ledger.transaction() as tx:
                        candidate_keys: list[str] = []
                        for key, axis, payload in candidate_specs:
                            tx.record_candidate(
                                key,
                                axis,
                                payload,
                                (pending.chunk_id,),
                            )
                            candidate_keys.append(key)
                        outcome = (
                            "candidates_persisted"
                            if candidate_keys
                            else "zero_candidates"
                        )
                        outcome_key = self._proposer_outcome_key(
                            run_id=run_id,
                            pending=pending,
                            batch=result,
                            candidate_keys=candidate_keys,
                            outcome=outcome,
                        )
                        tx.record_chunk_proposer_outcome(
                            outcome_key,
                            pending.chunk_id,
                            outcome,
                            candidate_keys=candidate_keys,
                        )
                    counts["proposer_succeeded"] += 1
                    counts["proposer_chunks"] = counts[
                        "proposer_succeeded"
                    ]
                    counts["candidates"] = counts.get(
                        "candidates", 0
                    ) + len(candidate_specs)
                schedule_available()
        finally:
            if active:
                for task in active:
                    task.cancel()
                await asyncio.gather(*active, return_exceptions=True)

        after = self.ledger.proposer_backlog_stats(through=watermark)
        counts["proposer_pending_after"] = after.pending
        counts["proposer_quarantined"] = after.quarantined
        counts["proposer_unattempted_after"] = after.unattempted
        counts.setdefault("proposer_errors", 0)
        counts.setdefault("proposer_chunks", 0)

    def _record_proposer_error(
        self,
        run_id: str,
        pending: PendingProposerChunk,
        provider_code: str,
    ) -> None:
        safe_provider_code = (
            provider_code
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", provider_code)
            else "provider.invalid_code"
        )
        identity = {
            "chunk_id": pending.chunk_id,
            "content_digest": pending.content_digest,
            "error_code": safe_provider_code,
            "run_id": run_id,
        }
        key = "proposer-error:v1:" + _canonical_digest(
            identity, code="proposer.error_identity"
        )
        self.ledger.record_chunk_proposer_outcome(
            key,
            pending.chunk_id,
            "retryable_error",
            error_code=safe_provider_code,
        )

    def _candidate_specs(
        self,
        *,
        run_id: str,
        pending: PendingProposerChunk,
        batch: ProposerBatch,
    ) -> tuple[tuple[str, str, bytes], ...]:
        source = pending.source_event_ids[0]
        specs: list[tuple[str, str, bytes]] = []
        for ordinal, draft in enumerate(batch.candidates):
            if draft.source_chunk_ids != (pending.chunk_id,):
                raise NightRunCoordinatorError("proposer.source_mismatch")
            base = {
                "draft": _draft_json(draft),
                "origin_run_id": run_id,
                "proposer": {
                    "model": batch.model,
                    "output_digest": batch.output_digest,
                    "prompt_digest": batch.prompt_digest,
                    "provider": batch.provider,
                    "schema_version": batch.schema_version,
                },
                "source": {
                    "chunk_content_sha256": pending.content_digest,
                    "chunk_ids": [pending.chunk_id],
                    "created_at": pending.created_at,
                    "session_id": source.session_id,
                    "source_digest": _canonical_digest(
                        {
                            "chunk_content_sha256": pending.content_digest,
                            "chunk_id": pending.chunk_id,
                            "session_id": source.session_id,
                            "source_event_id": source.source_event_id,
                        },
                        code="candidate.source_invalid",
                    ),
                    "source_event_ids": [source.source_event_id],
                },
            }
            base_digest = _canonical_digest(
                base, code="candidate.identity_invalid"
            )
            axes = sorted(route_candidate_axes(draft))
            for axis in axes:
                payload = {
                    "axis": axis,
                    "base_digest": base_digest,
                    "candidate_ordinal": ordinal,
                    "draft": base["draft"],
                    "origin_run_id": run_id,
                    "proposer": base["proposer"],
                    "schema": _CANDIDATE_SCHEMA,
                    "source": base["source"],
                    "x_write_key": f"lmc5-x:v1:{base_digest}",
                }
                encoded = _canonical_bytes(
                    payload, code="candidate.payload_invalid"
                )
                key = (
                    f"candidate:v1:{axis.lower()}:"
                    f"{_sha256(encoded)}"
                )
                specs.append((key, axis, encoded))
        return tuple(specs)

    @staticmethod
    def _proposer_outcome_key(
        *,
        run_id: str,
        pending: PendingProposerChunk,
        batch: ProposerBatch,
        candidate_keys: Sequence[str],
        outcome: str,
    ) -> str:
        identity = {
            "candidate_keys": sorted(candidate_keys),
            "chunk_id": pending.chunk_id,
            "content_digest": pending.content_digest,
            "outcome": outcome,
            "output_digest": batch.output_digest,
            "run_id": run_id,
        }
        return "proposer-success:v1:" + _canonical_digest(
            identity, code="proposer.outcome_identity"
        )

    async def _dispatch_pending(self, counts: dict[str, int]) -> None:
        counts["dispatch_attempted"] = 0
        counts["dispatch_retryable"] = 0
        counts["dispatch_circuit_breaker"] = 0
        consecutive_errors = 0
        after: int | None = None
        while True:
            rows = self.ledger.list_candidates(
                "pending",
                limit=self.policy.pending_page_size,
                after=after,
            )
            if not rows:
                counts["dispatch_pending_after"] = (
                    self._pending_candidate_counts()[0]
                )
                return
            for record in rows:
                # M remains pending until both report-only engines have
                # completed.  Leaving it here must not make this page spin.
                if record.axis == "M":
                    continue
                counts["dispatch_attempted"] += 1
                try:
                    await self._dispatch_candidate(record, counts)
                except NightRunCoordinatorError as exc:
                    if exc.code not in _RETRYABLE_DISPATCH_CODES:
                        raise
                    counts["dispatch_retryable"] += 1
                    error_key = f"dispatch_retryable_{exc.code.replace('.', '_')}"
                    counts[error_key] = counts.get(error_key, 0) + 1
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        counts["dispatch_circuit_breaker"] = 1
                        counts["dispatch_pending_after"] = (
                            self._pending_candidate_counts()[0]
                        )
                        return
                    continue
                consecutive_errors = 0
            after = rows[-1].candidate_id

    def _pending_candidate_counts(self) -> tuple[int, int]:
        dispatch_count = 0
        metabolism_count = 0
        after: int | None = None
        while True:
            rows = self.ledger.list_candidates(
                "pending",
                limit=self.policy.pending_page_size,
                after=after,
            )
            if not rows:
                return dispatch_count, metabolism_count
            dispatch_count += sum(record.axis != "M" for record in rows)
            metabolism_count += sum(record.axis == "M" for record in rows)
            after = rows[-1].candidate_id

    async def _dispatch_candidate(
        self, record: CandidateRecord, counts: dict[str, int]
    ) -> None:
        payload = _load_json_object(
            record.payload, code="candidate.persisted_invalid"
        )
        if (
            payload.get("schema") != _CANDIDATE_SCHEMA
            or payload.get("axis") != record.axis
        ):
            raise NightRunCoordinatorError("candidate.persisted_invalid")
        axis = record.axis
        if axis == "X":
            await self._dispatch_x(record, payload, counts)
            return
        if axis == "Y":
            await self._dispatch_y(record, payload, counts)
            return
        if axis == "Z":
            await self._dispatch_z(record, payload, counts)
            return
        if axis == "E":
            await self._dispatch_e(record, payload, counts)
            return
        if axis == "M":
            return
        raise NightRunCoordinatorError("candidate.axis_unknown")

    async def _dispatch_x(
        self,
        record: CandidateRecord,
        payload: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        draft = payload.get("draft")
        source = payload.get("source")
        if type(draft) is not dict or type(source) is not dict:
            raise NightRunCoordinatorError("candidate.persisted_invalid")
        if draft.get("risk") != "normal":
            self._defer(record, "x.human_review_required", counts)
            return
        if draft.get("type") not in CANDIDATE_TYPES:
            self._defer(record, "x.type_requires_axis_decision", counts)
            return
        relation_hints = draft.get("relation_hints")
        if type(relation_hints) is not list:
            raise NightRunCoordinatorError("candidate.persisted_invalid")
        content = draft.get("content")
        title = draft.get("title")
        importance = draft.get("importance")
        if (
            type(content) is not str
            or not content.strip()
            or type(title) is not str
            or not title.strip()
            or type(importance) is not int
            or isinstance(importance, bool)
            or not 1 <= importance <= 10
        ):
            raise NightRunCoordinatorError("candidate.persisted_invalid")
        session_id = source.get("session_id")
        event_ids = source.get("source_event_ids")
        source_digest = source.get("source_digest")
        write_key = payload.get("x_write_key")
        if (
            type(session_id) is not str
            or type(event_ids) is not list
            or not event_ids
            or any(type(item) is not str or not item for item in event_ids)
            or type(source_digest) is not str
            or not _SHA256_RE.fullmatch(source_digest)
            or type(write_key) is not str
        ):
            raise NightRunCoordinatorError("candidate.persisted_invalid")

        result = await self._write_x_payload(payload)
        if not self._x_result_is_complete(result):
            raise NightRunCoordinatorError("x.write_retryable")
        result_value = asdict(result)
        result_hash = _canonical_digest(
            result_value, code="x.result_invalid"
        )
        with self.ledger.transaction() as tx:
            tx.record_write_receipt(
                f"x-receipt:v1:{record.payload_digest}",
                record.payload_digest,
                f"bucket:{result.bucket_id}",
                result_hash=result_hash,
            )
            tx.transition_candidate(
                record.idempotency_key,
                "ready",
                expected_status="pending",
            )
        counts["x_ready"] = counts.get("x_ready", 0) + 1

    async def _write_x_payload(
        self,
        payload: dict[str, Any],
    ) -> CuratedWriteResult:
        draft = payload.get("draft")
        source = payload.get("source")
        if type(draft) is not dict or type(source) is not dict:
            raise NightRunCoordinatorError("candidate.persisted_invalid")
        content = draft.get("content")
        title = draft.get("title")
        importance = draft.get("importance")
        candidate_type = draft.get("type")
        session_id = source.get("session_id")
        event_ids = source.get("source_event_ids")
        source_digest = source.get("source_digest")
        write_key = payload.get("x_write_key")
        if (
            type(content) is not str
            or not content.strip()
            or type(title) is not str
            or not title.strip()
            or type(importance) is not int
            or isinstance(importance, bool)
            or not 1 <= importance <= 10
            or candidate_type not in CANDIDATE_TYPES
            or type(session_id) is not str
            or type(event_ids) is not list
            or not event_ids
            or any(type(item) is not str or not item for item in event_ids)
            or type(source_digest) is not str
            or not _SHA256_RE.fullmatch(source_digest)
            or type(write_key) is not str
        ):
            raise NightRunCoordinatorError("candidate.persisted_invalid")
        return await self.curated.write(
            idempotency_key=write_key,
            content=content,
            vector_policy=self.policy.vector_policy,
            bucket_options={
                "bucket_type": "dynamic",
                "importance": importance,
                "name": title,
                "tags": ["lmc5", "night", candidate_type],
                "x_provenance": {
                    "source_kind": "conversation",
                    "source_session": session_id,
                    "source_event_ids": event_ids,
                    "source_digest": source_digest,
                },
            },
            actor="lmc5:night",
        )

    async def _dispatch_y(
        self,
        record: CandidateRecord,
        payload: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        if self.bucket_manager is None or self.review_queue is None:
            self._defer(record, "y.storage_unavailable", counts)
            return
        draft = payload.get("draft")
        if type(draft) is not dict or draft.get("risk") != "normal":
            self._defer(record, "y.human_review_required", counts)
            return
        relations = draft.get("relation_hints")
        if type(relations) is not list or not relations:
            self._defer(record, "y.relation_pair_unavailable", counts)
            return
        result = await self._write_x_payload(payload)
        if not self._x_result_is_complete(result):
            raise NightRunCoordinatorError("y.source_write_retryable")
        source_id = str(result.bucket_id)
        safe_count = 0
        review_count = 0
        outcomes: list[dict[str, Any]] = []
        for relation in relations:
            if type(relation) is not dict:
                raise NightRunCoordinatorError("candidate.persisted_invalid")
            rel_type = str(relation.get("relation_type") or "")
            target_id = str(relation.get("target_id") or "")
            reason = str(relation.get("reason") or "")
            strength = relation.get("strength")
            target = await self.bucket_manager.get(target_id)
            if target is None:
                raise NightRunCoordinatorError("y.target_unavailable")
            if rel_type in {"kin", "explains"}:
                ok = await self.bucket_manager.add_relation(
                    source_id,
                    target_id,
                    rel_type,
                    reason,
                    strength=strength,
                    actor="lmc5:night:y-safe",
                )
                if not ok:
                    raise NightRunCoordinatorError("y.safe_write_retryable")
                safe_count += 1
                outcomes.append({"status": "applied", **relation})
                continue
            entry = make_relation_entry(
                source_id,
                target_id,
                rel_type,
                reason,
                source_name=str(draft.get("title") or source_id),
                target_name=str((target.get("metadata") or {}).get("name") or target_id),
                strength=strength,
            )
            self.review_queue.enqueue(entry)
            review_count += 1
            outcomes.append({"status": "pending_review", **relation})
        result_hash = _canonical_digest(outcomes, code="y.result_invalid")
        with self.ledger.transaction() as tx:
            tx.record_write_receipt(
                f"y-receipt:v1:{record.payload_digest}",
                record.payload_digest,
                f"relations:{source_id}",
                result_hash=result_hash,
            )
            tx.transition_candidate(
                record.idempotency_key,
                "ready",
                expected_status="pending",
            )
        counts["y_safe_ready"] = counts.get("y_safe_ready", 0) + safe_count
        counts["y_review_ready"] = counts.get("y_review_ready", 0) + review_count

    async def _dispatch_z(
        self,
        record: CandidateRecord,
        payload: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        if (
            self.bucket_manager is None
            or self.review_queue is None
            or not self.fact_slot_registry
        ):
            self._defer(record, "z.storage_unavailable", counts)
            return
        draft = payload.get("draft")
        if type(draft) is not dict or draft.get("risk") != "normal":
            self._defer(record, "z.human_review_required", counts)
            return
        result = await self._write_x_payload(payload)
        if not self._x_result_is_complete(result):
            raise NightRunCoordinatorError("z.source_write_retryable")
        new_bucket = await self.bucket_manager.get(str(result.bucket_id))
        if new_bucket is None or is_fact_slot_exempt(new_bucket):
            self._defer(record, "z.fact_pair_unavailable", counts)
            return
        facts = extract_registered_facts(
            str(new_bucket.get("content") or ""),
            self._fact_slot_config,
            bucket=new_bucket,
        )
        if len(facts) != 1:
            self._defer(record, "z.fact_pair_unavailable", counts)
            return
        fact_key = next(iter(facts))
        candidates: list[dict[str, Any]] = []
        for bucket in await self.bucket_manager.list_all(include_archive=True):
            if bucket.get("id") == new_bucket.get("id") or is_fact_slot_exempt(bucket):
                continue
            metadata = bucket.get("metadata") or {}
            existing_key = str(metadata.get("fact_key") or "").strip().lower()
            status = str(metadata.get("fact_status") or "current").strip().lower()
            if existing_key == fact_key and status != FACT_STATUS_HISTORICAL:
                candidates.append(bucket)
                continue
            extracted = extract_registered_facts(
                str(bucket.get("content") or ""),
                self._fact_slot_config,
                bucket=bucket,
            )
            if fact_key in extracted:
                candidates.append(bucket)
        if not candidates:
            self._defer(record, "z.fact_pair_unavailable", counts)
            return

        def order_key(bucket: dict[str, Any]) -> tuple[str, str]:
            metadata = bucket.get("metadata") or {}
            timestamp = str(
                metadata.get("event_at")
                or metadata.get("recorded_at")
                or metadata.get("created")
                or ""
            )
            return timestamp, str(bucket.get("id") or "")

        prior = max(candidates, key=order_key)
        new_key = order_key(new_bucket)
        prior_key = order_key(prior)
        current, historical = (
            (new_bucket, prior) if new_key >= prior_key else (prior, new_bucket)
        )
        entry = make_z_pair_entry(
            str(current["id"]),
            str(historical["id"]),
            fact_key=fact_key,
            current_name=str((current.get("metadata") or {}).get("name") or current["id"]),
            historical_name=str((historical.get("metadata") or {}).get("name") or historical["id"]),
            reason="lmc5_night_registered_fact_transition",
            source="lmc5_night",
        )
        self.review_queue.enqueue(entry)
        result_hash = _canonical_digest(entry, code="z.result_invalid")
        with self.ledger.transaction() as tx:
            tx.record_write_receipt(
                f"z-receipt:v1:{record.payload_digest}",
                record.payload_digest,
                f"review:{entry['key']}",
                result_hash=result_hash,
            )
            tx.transition_candidate(
                record.idempotency_key,
                "ready",
                expected_status="pending",
            )
        counts["z_review_ready"] = counts.get("z_review_ready", 0) + 1

    async def _dispatch_e(
        self,
        record: CandidateRecord,
        payload: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        if self.review_queue is None:
            self._defer(record, "e.proposal_storage_unavailable", counts)
            return
        draft = payload.get("draft")
        if type(draft) is not dict:
            raise NightRunCoordinatorError("candidate.persisted_invalid")
        if draft.get("risk") != "normal":
            self._defer(record, "e.human_review_required", counts)
            return
        result = await self._write_x_payload(payload)
        if not self._x_result_is_complete(result):
            raise NightRunCoordinatorError("e.source_write_retryable")
        importance = draft.get("importance")
        if type(importance) is not int or isinstance(importance, bool):
            raise NightRunCoordinatorError("candidate.persisted_invalid")
        entry = make_e_proposal_entry(
            str(result.bucket_id),
            str(draft.get("type") or ""),
            str(draft.get("title") or ""),
            str(draft.get("evidence") or ""),
            suggested_priority=max(1, min(100, importance * 10)),
        )
        self.review_queue.enqueue(entry)
        result_hash = _canonical_digest(entry, code="e.result_invalid")
        with self.ledger.transaction() as tx:
            tx.record_write_receipt(
                f"e-receipt:v1:{record.payload_digest}",
                record.payload_digest,
                f"proposal:{entry['key']}",
                result_hash=result_hash,
            )
            tx.transition_candidate(
                record.idempotency_key,
                "ready",
                expected_status="pending",
            )
        counts["e_proposal_ready"] = counts.get("e_proposal_ready", 0) + 1

    @staticmethod
    def _x_result_is_complete(result: CuratedWriteResult) -> bool:
        return bool(
            isinstance(result, CuratedWriteResult)
            and result.success
            and result.status == "completed"
            and result.bucket_id
            and result.vector_policy == "required"
            and result.recall_state == "ready_vector"
            and result.error_code is None
        )

    def _defer(
        self,
        record: CandidateRecord,
        code: str,
        counts: dict[str, int],
    ) -> None:
        self.ledger.transition_candidate(
            record.idempotency_key,
            "deferred",
            expected_status="pending",
            error_code=code,
        )
        key = f"{record.axis.lower()}_deferred"
        counts[key] = counts.get(key, 0) + 1

    async def _run_metabolism(self, counts: dict[str, int]) -> None:
        self._assert_report_only()
        decay = await self.decay_engine.run_decay_cycle()
        self._assert_report_only()
        consolidation = await self.consolidation_engine.run_consolidation_cycle()
        self._assert_report_only()
        self._validate_metabolism_result(decay, consolidation)
        report = {
            "consolidation": consolidation,
            "decay": decay,
            "schema": _METABOLISM_RECEIPT_SCHEMA,
        }
        report_hash = _canonical_digest(
            report, code="metabolism.report_invalid"
        )
        receipts_since_yield = 0
        while True:
            pending_m = tuple(
                row
                for row in self.ledger.list_candidates(
                    "pending", limit=self.policy.pending_page_size
                )
                if row.axis == "M"
            )
            if not pending_m:
                break
            for record in pending_m:
                with self.ledger.transaction() as tx:
                    tx.record_write_receipt(
                        f"m-receipt:v1:{record.payload_digest}",
                        record.payload_digest,
                        "metabolism:report-only:v1",
                        result_hash=report_hash,
                    )
                    tx.transition_candidate(
                        record.idempotency_key,
                        "ready",
                        expected_status="pending",
                    )
                counts["m_computed"] = counts.get("m_computed", 0) + 1
                receipts_since_yield += 1
                if receipts_since_yield >= _M_RECEIPT_YIELD_EVERY:
                    receipts_since_yield = 0
                    await asyncio.sleep(0)

    @staticmethod
    def _validate_metabolism_result(
        decay: Any, consolidation: Any
    ) -> None:
        if type(decay) is not dict or type(consolidation) is not dict:
            raise NightRunCoordinatorError("metabolism.report_invalid")
        if (
            decay.get("ok") is not True
            or decay.get("mode") != "report_only"
            or type(decay.get("checked")) is not int
            or decay.get("checked") <= 0
            or type(decay.get("archived")) is not int
            or decay.get("archived") != 0
            or type(decay.get("auto_resolved")) is not int
            or decay.get("auto_resolved") != 0
            or type(decay.get("errors")) is not list
            or decay.get("errors")
        ):
            raise NightRunCoordinatorError("metabolism.decay_unsafe")
        if (
            consolidation.get("ok") is not True
            or consolidation.get("mode") != "report_only"
            or type(consolidation.get("auto_digested")) is not int
            or consolidation.get("auto_digested") != 0
            or consolidation.get("report_bucket_id") is not None
            or type(consolidation.get("errors")) is not list
            or consolidation.get("errors")
        ):
            raise NightRunCoordinatorError("metabolism.consolidation_unsafe")

    async def _validate(
        self,
        *,
        run_id: str,
        cutoff_iso: str,
        snapshot: SnapshotResult,
        counts: Mapping[str, int],
    ) -> None:
        verified = await _await_daemon_thread(
            lambda: self.snapshots.verify_snapshot(
                run_id,
                expected_manifest_sha256=snapshot.manifest_sha256,
            )
        )
        if verified.manifest_sha256 != snapshot.manifest_sha256:
            raise NightRunCoordinatorError("snapshot.verify_failed")
        self.ledger.verify_integrity(deep=True)
        if self.ledger.list_uncovered_raw_events(
            limit=1, created_before=cutoff_iso
        ):
            raise NightRunCoordinatorError("validation.raw_uncovered")
        required_counts = (
            "proposer_watermark",
            "proposer_pending_before",
            "proposer_attempted",
            "proposer_succeeded",
            "proposer_retryable",
            "proposer_pending_after",
            "proposer_quarantined",
            "proposer_circuit_breaker",
            "proposer_wall_budget_exhausted",
            "dispatch_attempted",
            "dispatch_retryable",
            "dispatch_pending_after",
            "dispatch_circuit_breaker",
        )
        if any(key not in counts for key in required_counts):
            raise NightRunCoordinatorError("validation.proposer_counts")
        attempted = counts["proposer_attempted"]
        succeeded = counts["proposer_succeeded"]
        retryable = counts["proposer_retryable"]
        pending_before = counts["proposer_pending_before"]
        pending_after = counts["proposer_pending_after"]
        if (
            attempted > self.policy.proposer_max_chunks_per_run
            or attempted != succeeded + retryable
            or pending_before != succeeded + pending_after
        ):
            raise NightRunCoordinatorError("validation.proposer_counts")
        backlog = self.ledger.proposer_backlog_stats(
            through=counts["proposer_watermark"]
        )
        if (
            backlog.pending != pending_after
            or backlog.quarantined != counts["proposer_quarantined"]
        ):
            raise NightRunCoordinatorError("validation.proposer_counts")
        dispatch_attempted = counts["dispatch_attempted"]
        dispatch_retryable = counts["dispatch_retryable"]
        dispatch_pending = counts["dispatch_pending_after"]
        dispatch_breaker = counts["dispatch_circuit_breaker"]
        if (
            dispatch_retryable > dispatch_attempted
            or dispatch_breaker not in {0, 1}
            or (dispatch_breaker and dispatch_retryable < 3)
            or (not dispatch_breaker and dispatch_pending != dispatch_retryable)
            or (dispatch_breaker and dispatch_pending < dispatch_retryable)
        ):
            raise NightRunCoordinatorError("validation.dispatch_counts")
        live_dispatch_pending, live_metabolism_pending = (
            self._pending_candidate_counts()
        )
        if (
            live_dispatch_pending != dispatch_pending
            or live_metabolism_pending != 0
        ):
            raise NightRunCoordinatorError("validation.candidate_pending")
        if self.ledger.list_candidates("error", limit=1):
            raise NightRunCoordinatorError("validation.candidate_error")

    def _advance(
        self,
        run_id: str,
        expected_stage: str,
        stage: str,
        counts: Mapping[str, int],
    ) -> NightRunResult:
        return self.ledger.record_night_stage(
            run_id,
            stage,
            counts=counts,
            expected_stage=expected_stage,
        )

    def _mark_error(
        self, run_id: str, code: str, counts: Mapping[str, int]
    ) -> None:
        try:
            current = self.ledger.get_night_run(run_id)
            if current.stage in TERMINAL_NIGHT_STAGES:
                return
            self.ledger.record_night_stage(
                run_id,
                "error",
                counts=counts,
                errors=(code,),
                expected_stage=current.stage,
            )
        except (LedgerError, OSError, ValueError):
            # The original sanitized coordinator failure is authoritative.
            # A broken ledger must never leak a secondary exception payload.
            return


__all__ = [
    "NightRunCoordinator",
    "NightRunCoordinatorError",
    "NightRunOutcome",
    "NightRunPolicy",
]
