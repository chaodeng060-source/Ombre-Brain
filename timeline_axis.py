"""LMC-5 X-axis thread storage, reviewed backfill, and recall neighbors.

``thread`` is a mutable narrative label. ``other`` is the incubator. This
module deliberately does not infer production threads from source-event ids,
episodes, free-text notes, or proposer ``thread_hint`` values. A bucket can
leave ``other`` only through an explicit review decision or a typed
``in_thread`` edge connected to exactly one already named line.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from utils import event_at_from_metadata


OTHER_THREAD = "other"
_THREAD_MAX_CHARS = 160
_ROUTING_PREFIXES = (
    "dm:",
    "room:",
    "task_",
    "task-",
    "task:",
    "ntf_",
    "ntf-",
)
_AUTOMATIC_PREFIXES = ("event:", "relation:", "episode:")
_OPAQUE_ID_RE = re.compile(
    r"^(?:[0-9a-f]{12,64}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
_MACHINE_SLUG_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+){2,}$")


@dataclass(frozen=True, slots=True)
class TimelineAssignment:
    bucket_id: str
    thread: str
    reason: str


@dataclass(frozen=True, slots=True)
class TimelineSweepPlan:
    assignments: tuple[TimelineAssignment, ...]
    assigned_count: int
    named_count: int
    new_line_count: int
    orphan_count: int
    preserved_count: int
    candidate_hint_count: int
    line_sizes: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TimelineSweepReport:
    scanned_count: int
    updated_count: int
    assigned_count: int
    named_count: int
    new_line_count: int
    orphan_count: int
    preserved_count: int
    candidate_hint_count: int
    line_sizes: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TimelineNeighbor:
    bucket_id: str
    thread: str
    direction: str
    distance: int
    via_id: str


def normalize_thread(value: object) -> str:
    """Strip a stored thread value and map empty values to ``other``."""

    text = str(value or "").strip()
    if not text or text.lower() == OTHER_THREAD:
        return OTHER_THREAD
    return text


def normalize_thread_hint(value: object) -> str:
    """Validate a proposed/reviewed narrative label without approving it."""

    text = normalize_thread(value)
    if text == OTHER_THREAD:
        return OTHER_THREAD
    lowered = text.lower()
    if lowered.startswith((*_ROUTING_PREFIXES, *_AUTOMATIC_PREFIXES)):
        return OTHER_THREAD
    if _OPAQUE_ID_RE.fullmatch(text):
        return OTHER_THREAD
    if _MACHINE_SLUG_RE.fullmatch(lowered) and not re.search(
        r"[\u3400-\u9fff]", text
    ):
        return OTHER_THREAD
    return text[:_THREAD_MAX_CHARS]


def _bucket_type(bucket: Mapping[str, Any]) -> str:
    metadata = bucket.get("metadata", {}) or {}
    return str(metadata.get("type") or "dynamic").strip().lower()


def _eligible_bucket(bucket: Mapping[str, Any]) -> bool:
    return bool(bucket.get("id")) and _bucket_type(bucket) not in {
        "archived",
        "nsfw",
    }


def _event_time(bucket: Mapping[str, Any]) -> datetime | None:
    raw = event_at_from_metadata(bucket.get("metadata", {}) or {})
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _relation_target(relation: object) -> str:
    """Return a valid live ``in_thread`` target, or an empty string."""

    if not isinstance(relation, Mapping):
        return ""
    if str(relation.get("type") or "").strip().lower() != "in_thread":
        return ""
    if relation.get("valid_until") not in (None, ""):
        return ""
    strength = relation.get("strength")
    if strength is not None:
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            return ""
        if not 0 <= float(strength) <= 1:
            return ""
    return str(
        relation.get("target") or relation.get("target_id") or ""
    ).strip()


def plan_timeline_assignments(
    buckets: Iterable[Mapping[str, Any]],
    *,
    thread_hints_by_bucket: Mapping[str, object] | None = None,
    reviewed_threads_by_bucket: Mapping[str, object] | None = None,
) -> TimelineSweepPlan:
    """Plan a fail-closed X backfill.

    Hints are counted for audit but never assigned. Typed ``in_thread`` edges
    may only propagate one unambiguous existing/reviewed label; an unanchored
    component or a component containing conflicting labels remains unchanged.
    Existing named buckets are preserved unless an explicit reviewed mapping
    names their replacement.  Machine hints and graph propagation can never
    rename an already named bucket.
    """

    candidates = [dict(bucket) for bucket in buckets if _eligible_bucket(bucket)]
    by_id = {str(bucket["id"]): bucket for bucket in candidates}
    existing_threads: dict[str, str] = {}
    desired: dict[str, str] = {}
    reasons: dict[str, str] = {}
    previously_named: set[str] = set()

    for bucket_id, bucket in by_id.items():
        existing = normalize_thread(
            (bucket.get("metadata", {}) or {}).get("thread")
        )
        existing_threads[bucket_id] = existing
        if existing != OTHER_THREAD:
            desired[bucket_id] = existing
            reasons[bucket_id] = "existing"
            previously_named.add(existing)

    for raw_bucket_id, raw_thread in (reviewed_threads_by_bucket or {}).items():
        bucket_id = str(raw_bucket_id)
        thread = normalize_thread_hint(raw_thread)
        explicit_incubator = (
            isinstance(raw_thread, str)
            and raw_thread.strip().casefold() == OTHER_THREAD
        )
        if bucket_id in by_id and (
            thread != OTHER_THREAD or explicit_incubator
        ):
            desired[bucket_id] = thread
            reasons[bucket_id] = "explicit_review"

    parent = {bucket_id: bucket_id for bucket_id in by_id}

    def find(bucket_id: str) -> str:
        root = bucket_id
        while parent[root] != root:
            root = parent[root]
        while parent[bucket_id] != bucket_id:
            next_id = parent[bucket_id]
            parent[bucket_id] = root
            bucket_id = next_id
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    linked: set[str] = set()
    for bucket_id, bucket in by_id.items():
        metadata = bucket.get("metadata", {}) or {}
        for relation in metadata.get("relations") or ():
            target = _relation_target(relation)
            if target in by_id and target != bucket_id:
                linked.update((bucket_id, target))
                union(bucket_id, target)

    components: dict[str, list[str]] = defaultdict(list)
    for bucket_id in linked:
        components[find(bucket_id)].append(bucket_id)
    for member_ids in components.values():
        anchored_threads = {
            desired[bucket_id]
            for bucket_id in member_ids
            if desired.get(bucket_id, OTHER_THREAD) != OTHER_THREAD
            and normalize_thread_hint(desired[bucket_id]) != OTHER_THREAD
        }
        if len(anchored_threads) != 1:
            continue
        thread = next(iter(anchored_threads))
        for bucket_id in member_ids:
            if bucket_id not in desired:
                desired[bucket_id] = thread
                reasons[bucket_id] = "in_thread"

    candidate_hints = {
        str(bucket_id): normalize_thread_hint(value)
        for bucket_id, value in (thread_hints_by_bucket or {}).items()
        if str(bucket_id) in by_id
    }
    candidate_hint_count = sum(
        thread != OTHER_THREAD for thread in candidate_hints.values()
    )

    assignments: list[TimelineAssignment] = []
    line_sizes: Counter[str] = Counter()
    preserved_count = 0
    for bucket_id in sorted(by_id):
        thread = desired.get(bucket_id, OTHER_THREAD)
        reason = reasons.get(bucket_id, "incubator")
        if reason == "existing":
            preserved_count += 1
        assignments.append(TimelineAssignment(bucket_id, thread, reason))
        if thread != OTHER_THREAD:
            line_sizes[thread] += 1

    named_count = sum(line_sizes.values())
    return TimelineSweepPlan(
        assignments=tuple(assignments),
        assigned_count=sum(
            existing_threads[bucket_id] == OTHER_THREAD
            and desired.get(bucket_id, OTHER_THREAD) != OTHER_THREAD
            for bucket_id in by_id
        ),
        named_count=named_count,
        new_line_count=len(set(line_sizes) - previously_named),
        orphan_count=len(assignments) - named_count,
        preserved_count=preserved_count,
        candidate_hint_count=candidate_hint_count,
        line_sizes=dict(sorted(line_sizes.items())),
    )


def load_thread_hints_from_ledger(ledger_path: str | Path) -> dict[str, str]:
    """Read proposer X hints as candidates; this is not a review decision."""

    path = Path(ledger_path)
    if not path.is_file():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15)
    try:
        rows = connection.execute(
            """
            SELECT candidate.payload, receipt.result_ref
              FROM candidates AS candidate
              JOIN write_receipts AS receipt
                ON receipt.idempotency_key =
                   'x-receipt:v1:' || candidate.payload_digest
             WHERE candidate.axis = 'X'
               AND candidate.status = 'ready'
               AND receipt.result_ref LIKE 'bucket:%'
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        connection.close()

    candidates: dict[str, set[str]] = defaultdict(set)
    for payload, result_ref in rows:
        try:
            document = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        thread = normalize_thread_hint(
            (document.get("draft") or {}).get("thread_hint")
        )
        bucket_id = str(result_ref).removeprefix("bucket:").strip()
        if bucket_id and thread != OTHER_THREAD:
            candidates[bucket_id].add(thread)
    return {
        bucket_id: next(iter(threads))
        for bucket_id, threads in candidates.items()
        if len(threads) == 1
    }


async def run_timeline_sweep(
    bucket_manager: Any,
    *,
    ledger_path: str | Path | None = None,
    reviewed_threads_by_bucket: Mapping[str, object] | None = None,
    apply: bool = True,
    actor: str = "lmc5:timeline-sweep",
    revision_hash_provider: Callable[[str, Mapping[str, Any]], str] | None = None,
) -> TimelineSweepReport:
    buckets = await bucket_manager.list_all(include_archive=False)
    hints = (
        load_thread_hints_from_ledger(ledger_path)
        if ledger_path is not None
        else {}
    )
    plan = plan_timeline_assignments(
        buckets,
        thread_hints_by_bucket=hints,
        reviewed_threads_by_bucket=reviewed_threads_by_bucket,
    )
    current = {
        str(bucket.get("id")): str(
            (bucket.get("metadata", {}) or {}).get("thread") or ""
        ).strip()
        for bucket in buckets
    }
    expected_revisions = (
        {
            str(bucket.get("id")): revision_hash_provider(
                str(bucket.get("content") or ""),
                bucket.get("metadata", {}) or {},
            )
            for bucket in buckets
        }
        if revision_hash_provider is not None
        else {}
    )
    updated = 0
    if apply:
        for assignment in plan.assignments:
            if current.get(assignment.bucket_id) == assignment.thread:
                continue
            kwargs: dict[str, Any] = {"actor": actor}
            expected_revision = expected_revisions.get(assignment.bucket_id, "")
            if expected_revision:
                kwargs["expected_revision_hash"] = expected_revision
            changed = await bucket_manager.set_thread(
                assignment.bucket_id,
                assignment.thread,
                **kwargs,
            )
            if not changed:
                raise RuntimeError(
                    f"timeline assignment failed: {assignment.bucket_id}"
                )
            updated += 1
    return TimelineSweepReport(
        scanned_count=len(plan.assignments),
        updated_count=updated,
        assigned_count=plan.assigned_count,
        named_count=plan.named_count,
        new_line_count=plan.new_line_count,
        orphan_count=plan.orphan_count,
        preserved_count=plan.preserved_count,
        candidate_hint_count=plan.candidate_hint_count,
        line_sizes=plan.line_sizes,
    )


def timeline_neighbors(
    buckets: Iterable[Mapping[str, Any]],
    seed_ids: Iterable[str],
    *,
    neighbor_window: int,
    max_results: int,
    allowed_node_ids: Iterable[str] | None = None,
    excluded_ids: Iterable[str] | None = None,
) -> list[TimelineNeighbor]:
    """Return bounded previous/next buckets from the seed's named thread."""

    if max_results <= 0 or neighbor_window <= 0:
        return []
    by_id = {
        str(bucket["id"]): bucket
        for bucket in buckets
        if _eligible_bucket(bucket)
    }
    allowed = (
        {str(value) for value in allowed_node_ids}
        if allowed_node_ids is not None
        else set(by_id)
    )
    excluded = {str(value) for value in (excluded_ids or ())}
    seeds = [str(value) for value in seed_ids if str(value) in by_id]
    excluded.update(seeds)

    by_thread: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for bucket_id, bucket in by_id.items():
        if bucket_id not in allowed:
            continue
        thread = normalize_thread(
            (bucket.get("metadata", {}) or {}).get("thread")
        )
        if thread != OTHER_THREAD:
            by_thread[thread].append(bucket)
    for members in by_thread.values():
        members.sort(
            key=lambda bucket: (
                _event_time(bucket) or datetime.min,
                str(bucket.get("id") or ""),
            )
        )

    result: list[TimelineNeighbor] = []
    seen: set[str] = set(excluded)
    for seed_id in seeds:
        seed = by_id[seed_id]
        thread = normalize_thread(
            (seed.get("metadata", {}) or {}).get("thread")
        )
        if thread == OTHER_THREAD or thread not in by_thread:
            continue
        members = by_thread[thread]
        position = next(
            (
                index
                for index, bucket in enumerate(members)
                if str(bucket.get("id")) == seed_id
            ),
            None,
        )
        if position is None:
            continue
        for distance in range(1, neighbor_window + 1):
            for direction, index in (
                ("previous", position - distance),
                ("next", position + distance),
            ):
                if not 0 <= index < len(members):
                    continue
                bucket_id = str(members[index].get("id") or "")
                if not bucket_id or bucket_id in seen:
                    continue
                seen.add(bucket_id)
                result.append(
                    TimelineNeighbor(
                        bucket_id=bucket_id,
                        thread=thread,
                        direction=direction,
                        distance=distance,
                        via_id=seed_id,
                    )
                )
                if len(result) >= max_results:
                    return result
    return result
