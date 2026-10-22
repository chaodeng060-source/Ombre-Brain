"""Deterministic, reversible consolidation of exact copies of one event.

``plan_exact_consolidation(await manager.list_all(include_nsfw=False))`` is
read-only and returns a JSON-serializable plan without memory bodies. The
runner plans first, then holds the manager's cross-process exclusive lease
from revalidation through every CAS update. Call it outside a shared lease;
an existing exclusive lease in the *same task* is supported.

Only ``digested`` and ``exact_duplicate_of`` are written. No model, vector,
delete, source rewrite or new backup store is involved. Existing mutation
audit must be enabled for crash recovery. Recover receipts after an interrupted
call with ``recovery_records_from_audit(manager.audit_log.list_events())``;
pass these (or a run's ``recovery`` list) to ``restore_exact_consolidation``.
Pending/failed audit rows are included: restore checks the actual post-write
revision, so a journal commit failure cannot strand a durable mutation.

Restore reactivates a copy only if its entire post-write revision still
matches. BucketManager.update cannot delete keys or preserve last_active:
originally missing/null markers become False/"", and last_active is advanced.
The exact original frontmatter/body remain in the existing mutation audit.

Recall integration must exclude digested/exact_duplicate_of entries before
ranking: BucketManager.search itself does not implement that exclusion. This
module deliberately does not patch the manager, server or night coordinator.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import frontmatter
from frontmatter.default_handlers import YAMLHandler

from bucket_manager import RECALL_DERIVED_METADATA_FIELDS, bucket_revision_hash
from maintenance_barrier import MaintenanceBarrierError
from utils import PROTECTED_RESOLVE_DOMAINS
from x_provenance import normalize_source_ref


DEFAULT_ACTOR = "system:exact-metabolism"
DEFAULT_MAX_UPDATES = 20
_PROTECTED_WORLDS = frozenset({"涩涩", "nsfw"})
_UNKNOWN = frozenset({"unknown", "none", "null", "n/a", "?", "未知"})
# This is an allowlist, not a prefix filter. Unknown metadata remains semantic.
# In particular created/event_at/date_* are EVENT times, not write times.
_RUNTIME_FIELDS = frozenset({
    "id", "name", "recorded_at", "last_active", "activation_count",
    "digested", "exact_duplicate_of", *RECALL_DERIVED_METADATA_FIELDS,
})
_MARKERS = ("digested", "exact_duplicate_of")


def _freeze(value: Any) -> tuple:
    """Type-sensitive exact equality, without JSON's bool/int/date coercions."""
    if value is None:
        return ("null",)
    if type(value) in (str, bool, int):
        return (type(value).__name__, value)
    if type(value) is float and math.isfinite(value):
        return ("float", value.hex())
    if type(value) in (datetime, date):
        return (type(value).__name__, value.isoformat())
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(_freeze(v) for v in value))
    if isinstance(value, Mapping) and all(type(k) is str for k in value):
        return ("dict", tuple((k, _freeze(value[k])) for k in sorted(value)))
    raise ValueError("unsupported or non-finite metadata")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _key(bucket: Mapping) -> tuple:
    return (
        bucket["content"],
        _freeze({k: v for k, v in bucket["metadata"].items()
                 if k not in _RUNTIME_FIELDS}),
    )


def _key_hash(key: tuple) -> str:
    return _sha(json.dumps(key, ensure_ascii=False, separators=(",", ":")))


def _revision(bucket: Mapping) -> str:
    return bucket_revision_hash(bucket["content"], bucket["metadata"])


def _text(value: Any) -> bool:
    return (type(value) is str and bool(value.strip())
            and value == value.strip() and value.casefold() not in _UNKNOWN)


def _source_fields(meta: Mapping) -> list[str]:
    fields = []
    for field in ("source_event_ids", "source_chunk_ids"):
        ids = meta.get(field)
        if (type(ids) is list and ids and all(_text(v) for v in ids)
                and len(set(ids)) == len(ids)):
            fields.append(field)
    ref = meta.get("source_ref")
    if _text(ref):
        fields.append("source_ref")
    elif type(ref) is dict:
        try:
            normalize_source_ref(ref)
        except ValueError:
            pass
        else:
            fields.append("source_ref")
    return fields


def _labels(value: Any) -> set[str]:
    if value is None:
        return set()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or any(type(v) is not str for v in values):
        raise ValueError("invalid domain/world/tags")
    return {v.strip().casefold() for v in values}


def _protection(domains, worlds) -> tuple[set[str], set[str]]:
    return (
        _labels(list(PROTECTED_RESOLVE_DOMAINS)) | {"涩涩", "nsfw"} | _labels(domains),
        set(_PROTECTED_WORLDS) | _labels(worlds),
    )


def _candidate(bucket: Mapping, domains: set[str], worlds: set[str]) -> str | None:
    if not isinstance(bucket, Mapping):
        return "invalid_bucket"
    meta = bucket.get("metadata")
    if (not isinstance(meta, Mapping) or not _text(bucket.get("id"))
            or meta.get("id", bucket["id"]) != bucket["id"]
            or type(bucket.get("content")) is not str or not bucket["content"]):
        return "invalid_bucket"
    if any(type(k) is not str for k in meta):
        return "invalid_metadata"
    if meta.get("type") != "dynamic":
        return "non_dynamic"
    for flag in ("pinned", "protected", "resolved", "digested", "permanent",
                 "nsfw", "is_nsfw", "protected_world", "world_protected"):
        if meta.get(flag):
            return flag
    if meta.get("exact_duplicate_of"):
        return "existing_duplicate"
    if any(k.startswith("e_") for k in meta) or meta.get("axis") == "E":
        return "E"
    try:
        if _labels(meta.get("domain")) & domains:
            return "protected_domain"
        if _labels(meta.get("world")) & worlds:
            return "protected_world"
        if _labels(meta.get("tags")) & {"experience", "e_axis", "e-axis"}:
            return "E"
        _key(bucket)
    except (ValueError, TypeError):
        return "invalid_metadata"
    if not _source_fields(meta):
        return "no_explicit_source"
    return None


def _alias_roots(buckets: Iterable[Mapping]) -> set[str]:
    """Never hide an existing active target and create a chain of dead targets."""
    return {meta["exact_duplicate_of"] for bucket in buckets if isinstance(bucket, Mapping)
            and isinstance(meta := bucket.get("metadata"), Mapping)
            and meta.get("digested") and _text(meta.get("exact_duplicate_of"))}


def _body_survives_update(bucket: Mapping) -> bool:
    """Reject parsed-equal files whose raw bodies update() would normalize.

    python-frontmatter strips leading/trailing whitespace and normalizes CRLF
    while reading. Comparing get()['content'] alone could otherwise hide two
    different originals, or change a legacy body during a metadata update.
    The production manager supplies paths; in-memory planning needs none.
    """
    try:
        raw = Path(bucket["path"]).read_bytes().decode("utf-8")
        handler = YAMLHandler()
        _, original_body = handler.split(raw)
        _, serialized_body = handler.split(frontmatter.dumps(frontmatter.Post(bucket["content"])))
        return original_body == serialized_body
    except (OSError, KeyError, TypeError, ValueError, UnicodeError):
        return False


def plan_exact_consolidation(
    buckets: Iterable[Mapping], *, max_updates: int = DEFAULT_MAX_UPDATES,
    protected_domains: Iterable[str] = (), protected_worlds: Iterable[str] = (),
) -> dict:
    """Plan exact copies; an existing alias target, otherwise smallest ID, wins.

    Body, all provenance, tags/domain/retrieval_keys, world, author, fact state,
    relations and unknown metadata must match, including types/list order.
    Protection arguments ADD exemptions; they cannot remove built-in ones.
    The update budget counts losers, never comparisons. Hash grouping avoids
    a pairwise scan. Input content strings are compared verbatim. For proposed
    groups with disk paths, raw bodies must also survive update's serializer;
    the runner repeats that check under the lease before mutation.
    Multiple active alias roots are reported rather than creating dead links.
    """
    if type(max_updates) is not int or max_updates < 0:
        raise ValueError("max_updates must be a non-negative integer")
    domains, worlds = _protection(protected_domains, protected_worlds)
    rows = list(buckets)
    counts = Counter(b.get("id") for b in rows
                     if isinstance(b, Mapping) and type(b.get("id")) is str)
    excluded: Counter = Counter()
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for bucket in rows:
        reason = _candidate(bucket, domains, worlds)
        if reason is None and counts[bucket["id"]] != 1:
            reason = "ambiguous_id"
        if reason:
            excluded[reason] += 1
        else:
            groups[_key(bucket)].append(bucket)
    entries = []
    roots = _alias_roots(rows)
    blocked_groups = []
    available = duplicate_groups = 0
    eligible = sum(len(g) for g in groups.values())
    for members in sorted(groups.values(), key=lambda g: min(b["id"] for b in g)):
        members = sorted(members, key=lambda b: b["id"])
        if len(members) < 2:
            continue
        lossless = [b for b in members if not b.get("path") or _body_survives_update(b)]
        excluded["body_would_normalize"] += len(members) - len(lossless)
        eligible -= len(members) - len(lossless)
        members = lossless
        if len(members) < 2:
            continue
        duplicate_groups += 1
        existing_roots = [b for b in members if b["id"] in roots]
        if len(existing_roots) > 1:
            blocked_groups.append({"ids": [b["id"] for b in existing_roots],
                                   "reason": "multiple_active_alias_roots"})
            continue
        winner = existing_roots[0] if existing_roots else members[0]
        available += len(members) - 1
        for loser in members:
            if loser["id"] == winner["id"]:
                continue
            if len(entries) >= max_updates:
                break
            entries.append({
                "winner_id": winner["id"], "loser_id": loser["id"],
                "winner_revision_hash": _revision(winner),
                "loser_revision_hash": _revision(loser),
                "content_sha256": _sha(winner["content"]),
                "semantic_sha256": _key_hash(_key(winner)),
                "source_fields": _source_fields(winner["metadata"]),
                "reason": "identical_body_provenance_and_semantic_metadata",
            })
    return {
        "schema_version": 1, "scanned": len(rows),
        "eligible": eligible,
        "excluded": {k: n for k, n in sorted(excluded.items()) if n}, "duplicate_groups": duplicate_groups,
        "blocked_groups": blocked_groups,
        "available": available, "planned": len(entries),
        "deferred": available - len(entries), "entries": entries,
    }


def _recovery(before: Mapping, after: Mapping) -> dict:
    meta = before["metadata"]
    return {
        "bucket_id": before["id"], "winner_id": after["metadata"]["exact_duplicate_of"],
        "before_revision_hash": _revision(before), "after_revision_hash": _revision(after),
        "content_sha256": _sha(before["content"]),
        "original_markers": {k: {"present": k in meta, "value": meta.get(k)} for k in _MARKERS},
    }


def recovery_records_from_audit(events: Iterable[Mapping], *, actor: str = DEFAULT_ACTOR) -> list[dict]:
    """Extract compact receipts, including potentially landed pending/failed rows.

    No corpus/body is copied to a new store. Filter by the actor used for the
    run. If a copy has multiple apply events, restore the newest matching
    revision; stale receipts are reported as conflicts, never replayed blindly.
    """
    records = []
    for event in events:
        if event.get("actor") != actor or event.get("action") != "update":
            continue
        try:
            before = json.loads(event["before_json"])
            after = json.loads(event["after_json"])
            old, new = before["metadata"], after["metadata"]
            if (new.get("digested") is not True or not _text(new.get("exact_duplicate_of"))
                    or old.get("digested") or old.get("exact_duplicate_of")
                    or before["content"] != after["content"]):
                continue
            before["id"] = after["id"] = event["bucket_id"]
            record = _recovery(before, after)
            record["audit_event_id"] = event["event_id"]
            records.append(record)
        except (KeyError, TypeError, ValueError):
            continue
    return records


def _require_audit(bucket_mgr) -> None:
    if not bucket_mgr.audit_log.enabled:
        raise ValueError("exact consolidation requires mutation audit for crash recovery")


async def run_exact_consolidation(
    bucket_mgr, *, enabled: bool = True, actor: str = DEFAULT_ACTOR,
    max_updates: int = DEFAULT_MAX_UPDATES,
    protected_domains: Iterable[str] = (), protected_worlds: Iterable[str] = (),
    lease_timeout: float | None = 30.0,
) -> dict:
    """Apply bounded exact duplicates; return plan, applied IDs and undo receipts.

    Night-run contract: ``consolidated`` counts durably observed soft hides;
    ``errors`` is an always-present list of execution/lease/root-conflict
    errors. Exemptions and stale-plan skips remain in the diagnostic fields.
    Barrier hierarchy/timeout errors return ``status='blocked'`` with the real
    reason. Cancellation propagates and releases the lease; durable partial
    work is idempotent on rerun and recoverable from the mutation journal.
    The exclusive lease must stay in this task (do not wrap update in a task).
    """
    result = {"status": "disabled" if not enabled else "completed", "plan": None,
              "consolidated": 0, "errors": [],
              "applied": [], "skipped": [], "failed": [], "recovery": []}
    if not enabled:
        return result
    # Materialize generators once so planning and revalidation share policy.
    protected_domains, protected_worlds = list(protected_domains), list(protected_worlds)
    plan = plan_exact_consolidation(
        await bucket_mgr.list_all(include_archive=False, include_nsfw=False),
        max_updates=max_updates, protected_domains=protected_domains, protected_worlds=protected_worlds,
    )
    result["plan"] = plan
    result["errors"].extend(
        f"{group['reason']}:{','.join(group['ids'])}" for group in plan["blocked_groups"]
    )
    if not plan["entries"]:
        if plan["blocked_groups"]:
            result["status"] = "partial"
        return result
    _require_audit(bucket_mgr)
    domains, worlds = _protection(protected_domains, protected_worlds)
    try:
        async with bucket_mgr._maintenance_barrier.exclusive_async(timeout=lease_timeout):
            # A writer may have added aliases since the advisory plan. Never
            # turn their target into a loser, even if its own revision is equal.
            roots = _alias_roots(await bucket_mgr.list_all(include_archive=False, include_nsfw=False))
            for entry in plan["entries"]:
                winner = await bucket_mgr.get(entry["winner_id"])
                loser = await bucket_mgr.get(entry["loser_id"])
                if (winner is None or loser is None
                        or _candidate(winner, domains, worlds) or _candidate(loser, domains, worlds)
                        or _revision(winner) != entry["winner_revision_hash"]
                        or _revision(loser) != entry["loser_revision_hash"]
                        or _key(winner) != _key(loser) or loser["id"] in roots):
                    result["skipped"].append({"loser_id": entry["loser_id"], "reason": "changed_since_plan"})
                    continue
                if not _body_survives_update(winner) or not _body_survives_update(loser):
                    result["skipped"].append({"loser_id": loser["id"], "reason": "body_would_normalize"})
                    continue
                ok = await bucket_mgr.update(
                    loser["id"], actor=actor, expected_revision_hash=entry["loser_revision_hash"],
                    digested=True, exact_duplicate_of=winner["id"],
                )
                after = await bucket_mgr.get(loser["id"])
                # A file write may land even if committing its audit row failed.
                landed = bool(after and after["metadata"].get("digested") is True
                              and after["metadata"].get("exact_duplicate_of") == winner["id"]
                              and _key(after) == _key(loser))
                if landed:
                    result["applied"].append(loser["id"])
                    result["consolidated"] += 1
                    result["recovery"].append(_recovery(loser, after))
                if not ok or not landed:
                    result["failed"].append({"loser_id": loser["id"], "reason": "update_failed"})
                    result["errors"].append(f"update_failed:{loser['id']}")
                await asyncio.sleep(0)  # explicit cancellation point between atomic writes
    except MaintenanceBarrierError as exc:
        result["status"] = "blocked"
        result["error"] = str(exc)
        result["errors"].append(str(exc))
        return result
    if result["failed"] or result["skipped"] or plan["blocked_groups"]:
        result["status"] = "partial"
    return result


async def restore_exact_consolidation(
    bucket_mgr, recovery: Iterable[Mapping], *, actor: str = "system:exact-metabolism-restore",
    lease_timeout: float | None = 30.0,
) -> dict:
    """Reactivate matching losers under exclusive lease and revision CAS.

    An already restored or subsequently edited bucket is skipped, making
    retries safe. A winner need not still exist to recover an original copy.
    Missing/null original markers use inactive values (see module docstring).
    """
    result = {"status": "completed", "restored": [], "skipped": [], "failed": []}
    try:
        async with bucket_mgr._maintenance_barrier.exclusive_async(timeout=lease_timeout):
            for record in recovery:
                bucket_id = record["bucket_id"]
                current = await bucket_mgr.get(bucket_id)
                if (current is None or _revision(current) != record["after_revision_hash"]
                        or current["metadata"].get("digested") is not True
                        or current["metadata"].get("exact_duplicate_of") != record["winner_id"]):
                    result["skipped"].append({"bucket_id": bucket_id, "reason": "changed_or_already_restored"})
                    continue
                if not _body_survives_update(current):
                    result["skipped"].append({"bucket_id": bucket_id, "reason": "body_would_normalize"})
                    continue
                markers = record["original_markers"]
                digested = markers["digested"]["value"] if markers["digested"]["present"] else False
                duplicate = markers["exact_duplicate_of"]["value"] if markers["exact_duplicate_of"]["present"] else ""
                if digested or duplicate:
                    raise ValueError("recovery record does not describe an originally active copy")
                ok = await bucket_mgr.update(
                    bucket_id, actor=actor, expected_revision_hash=record["after_revision_hash"],
                    digested=False if digested is None else digested,
                    exact_duplicate_of="" if duplicate is None else duplicate,
                )
                result["restored" if ok else "failed"].append(bucket_id)
                await asyncio.sleep(0)
    except MaintenanceBarrierError as exc:
        result.update(status="blocked", error=str(exc))
        return result
    if result["skipped"] or result["failed"]:
        result["status"] = "partial"
    return result
