"""Fail-closed coordinator for one conservative LMC-5 night run.

The coordinator deliberately does less than the eventual LMC-5 design:

* raw events are redacted and durably chunked before leaving local storage;
* one chunk is proposed at a time and the candidate fan-out is transactional;
* only normal, relation-free ``event`` drafts may become recall-visible X rows;
* Y/Z/E stay deferred until their stronger storage contracts are available;
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
from typing import Any, Mapping, Sequence

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
    CandidateDraft,
    ProposerBatch,
    ProposerChunk,
    ProposerContractError,
    RelationHint,
    StrictOmbreProposer,
    route_candidate_axes,
)
from redact import redact_obj
from snapshot_manager import SnapshotManager, SnapshotResult


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_SCHEMA = "ombre.lmc5-axis-candidate/v1"
_SNAPSHOT_RECEIPT_SCHEMA = "ombre.lmc5-snapshot-receipt/v1"
_METABOLISM_RECEIPT_SCHEMA = "ombre.lmc5-metabolism-receipt/v1"
_CHUNK_SCHEMA = "ombre.lmc5-redacted-event/v1"
_REDACTION_VERSION = "redact_obj/v1"


class NightRunCoordinatorError(RuntimeError):
    """A bounded machine-readable night-run failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class NightRunPolicy:
    raw_page_size: int = 100
    pending_page_size: int = 100
    chunk_bytes: int = 24 * 1024
    barrier_timeout_seconds: float = 60.0
    vector_policy: str = "required"

    def __post_init__(self) -> None:
        for field in ("raw_page_size", "pending_page_size", "chunk_bytes"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.raw_page_size > 1_000 or self.pending_page_size > 1_000:
            raise ValueError("ledger page sizes cannot exceed 1000")
        if self.chunk_bytes > 256 * 1024:
            raise ValueError("chunk_bytes exceeds the proposer input contract")
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
        policy: NightRunPolicy | None = None,
    ) -> None:
        self.ledger = ledger
        self.snapshots = snapshots
        self.proposer = proposer
        self.curated = curated
        self.decay_engine = decay_engine
        self.consolidation_engine = consolidation_engine
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

                await self._propose_pending(run_id, counts)
                self._advance(run_id, "chunked", "proposed", counts)

                await self._dispatch_pending(counts)
                self._advance(run_id, "proposed", "dispatched", counts)

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
                )
                self._advance(
                    run_id,
                    "metabolism_reported",
                    "validated",
                    counts,
                )
                completed = self._advance(
                    run_id, "validated", "complete", counts
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
        self, run_id: str, counts: dict[str, int]
    ) -> None:
        after: int | None = None
        retryable_errors = 0
        while True:
            pending_rows = self.ledger.list_pending_proposer_chunks(
                limit=self.policy.pending_page_size,
                after=after,
            )
            if not pending_rows:
                break
            for pending in pending_rows:
                after = pending.row_id
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
                try:
                    batch = await self.proposer.propose(
                        (ProposerChunk(id=pending.chunk_id, text=text),),
                        frozenset(),
                    )
                except ProposerContractError as exc:
                    self._record_proposer_error(run_id, pending, exc.code)
                    retryable_errors += 1
                    counts["proposer_errors"] = retryable_errors
                    continue
                candidate_specs = self._candidate_specs(
                    run_id=run_id,
                    pending=pending,
                    batch=batch,
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
                        batch=batch,
                        candidate_keys=candidate_keys,
                        outcome=outcome,
                    )
                    tx.record_chunk_proposer_outcome(
                        outcome_key,
                        pending.chunk_id,
                        outcome,
                        candidate_keys=candidate_keys,
                    )
                counts["proposer_chunks"] = (
                    counts.get("proposer_chunks", 0) + 1
                )
                counts["candidates"] = counts.get("candidates", 0) + len(
                    candidate_specs
                )
        if retryable_errors:
            raise NightRunCoordinatorError("proposer.failed")

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
        after: int | None = None
        while True:
            rows = self.ledger.list_candidates(
                "pending",
                limit=self.policy.pending_page_size,
                after=after,
            )
            if not rows:
                return
            for record in rows:
                # M remains pending until both report-only engines have
                # completed.  Leaving it here must not make this page spin.
                if record.axis == "M":
                    continue
                await self._dispatch_candidate(record, counts)
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
            self._defer(record, "y.strength_storage_unavailable", counts)
            return
        if axis == "Z":
            self._defer(record, "z.fact_pair_unavailable", counts)
            return
        if axis == "E":
            self._defer(record, "e.scorer_unavailable", counts)
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
        if draft.get("type") != "event":
            self._defer(record, "x.type_requires_axis_decision", counts)
            return
        relation_hints = draft.get("relation_hints")
        if type(relation_hints) is not list or relation_hints:
            self._defer(record, "x.relation_review_required", counts)
            return
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

        result = await self.curated.write(
            idempotency_key=write_key,
            content=content,
            vector_policy=self.policy.vector_policy,
            bucket_options={
                "bucket_type": "dynamic",
                "importance": importance,
                "name": title,
                "tags": ["lmc5", "night", "event"],
                "x_provenance": {
                    "source_kind": "conversation",
                    "source_session": session_id,
                    "source_event_ids": event_ids,
                    "source_digest": source_digest,
                },
            },
            actor="lmc5:night",
        )
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
        if self.ledger.list_pending_proposer_chunks(limit=1):
            raise NightRunCoordinatorError("validation.proposer_pending")
        if self.ledger.list_candidates("pending", limit=1):
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
