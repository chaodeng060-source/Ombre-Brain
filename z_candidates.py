#!/usr/bin/env python3
"""Z-axis candidate proposals: same registered fact slot, newer bucket vs older bucket.

Pure helpers.  No BucketManager, MCP server, or queue writes here; callers
(patrol / night run) decide whether to enqueue.  Every proposal is only a
*candidate* for human review — nothing here mutates ``fact_status``.

Why this exists (2026-08-18 audit): the Z pipeline (review_queue z_conflict
entries, lifecycle_updates, historical recall suppression) was complete, but
nothing produced candidates: ``scan_cross_bucket_z_conflicts`` only pairs by
name/domain and reports without a registered ``fact_key`` so nothing could be
queued, and the night run had no Z step.  This module groups buckets by
registered slot, orders them by ``created`` and proposes newer→older pairs
whose contents actually conflict.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Mapping

from fact_conflicts import detect_fact_conflicts, is_z_scan_candidate
from fact_slots import (
    FACT_STATUS_CONTESTED,
    FACT_STATUS_HISTORICAL,
    extract_registered_facts,
    fact_slot_applies_to_bucket,
    is_fact_slot_exempt,
    normalize_fact_slot_registry,
    registered_fact_key,
)

REASON_SLOT_NEWER_SUPERSEDES = "same_fact_slot_newer_supersedes"
MATCH_STRUCTURED = "structured"   # content had a registered `label: value` line
MATCH_METADATA = "metadata"       # bucket metadata.fact_key is a registered slot
MATCH_CONTEXT = "context"         # only registry context (domains/types/tags/name) matched


def _meta(bucket: dict) -> dict:
    meta = bucket.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _bucket_id(bucket: dict) -> str:
    return str(bucket.get("id") or _meta(bucket).get("id") or "").strip()


def _name(bucket: dict) -> str:
    return str(_meta(bucket).get("name") or _bucket_id(bucket))[:160]


def parse_created(value) -> datetime | None:
    """Accept str / datetime / date; normalise to naive datetime (like patrol)."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "")).replace(tzinfo=None)
    except Exception:
        return None


def bucket_created(bucket: dict) -> datetime | None:
    meta = _meta(bucket)
    for key in ("created", "recorded_at", "last_active"):
        parsed = parse_created(meta.get(key))
        if parsed:
            return parsed
    return None


def _spec_has_context(spec) -> bool:
    if not isinstance(spec, Mapping):
        return False
    return any(spec.get(k) for k in ("domains", "types", "tags_any", "name_contains"))


def slot_memberships(bucket: dict, registry: Mapping | None) -> dict[str, dict]:
    """Return {fact_key: {"match": ..., "values": [...]}} for one bucket.

    Strong matches (structured line / metadata.fact_key) always count.  A pure
    context match counts only when the registry entry actually declares context
    constraints — an unconstrained slot would otherwise swallow every bucket.
    """
    if is_fact_slot_exempt(bucket) or not is_z_scan_candidate(bucket):
        return {}
    slots = normalize_fact_slot_registry(registry)
    if not slots:
        return {}
    out: dict[str, dict] = {}
    found = extract_registered_facts(bucket.get("content", "") or "", registry, bucket=bucket)
    for key, values in found.items():
        out[key] = {"match": MATCH_STRUCTURED, "values": list(values)}
    meta_key = registered_fact_key(_meta(bucket).get("fact_key"), registry)
    if meta_key and meta_key not in out and fact_slot_applies_to_bucket(meta_key, bucket, registry):
        value = _meta(bucket).get("fact_value")
        out[meta_key] = {"match": MATCH_METADATA, "values": [str(value)[:240]] if value else []}
    for key in slots:
        if key in out:
            continue
        spec = registry.get(key, {}) if isinstance(registry, Mapping) else {}
        if _spec_has_context(spec) and fact_slot_applies_to_bucket(key, bucket, registry):
            out[key] = {"match": MATCH_CONTEXT, "values": []}
    return out


def _already_linked(newer: dict, older: dict) -> bool:
    nm, om = _meta(newer), _meta(older)
    older_id, newer_id = _bucket_id(older), _bucket_id(newer)
    if str(om.get("superseded_by_bucket_id") or "").strip() == newer_id:
        return True
    supersedes = nm.get("supersedes_bucket_ids") or []
    if isinstance(supersedes, str):
        supersedes = [supersedes]
    return older_id in {str(x).strip() for x in supersedes}


def _status(bucket: dict) -> str:
    return str(_meta(bucket).get("fact_status") or "").strip().lower()


def propose_z_pair_candidates(
    buckets: Iterable[dict],
    registry: Mapping | None,
    *,
    limit: int = 200,
    allow_context_only: bool = True,
) -> dict:
    """Group buckets by registered slot; propose newer→older pairs that conflict.

    Returns a report dict::

        {"candidates": [...], "stats": {...}}

    Each candidate carries what ``review_queue.make_z_pair_entry`` needs
    (fact_key / current_bucket_id / historical_bucket_id / names) plus the
    evidence (created timestamps, matched values, conflicting fields).
    Nothing is written anywhere.
    """
    groups: dict[str, list[tuple[dict, dict]]] = {}
    stats = {
        "buckets_seen": 0,
        "buckets_in_slots": 0,
        "memberships_by_match": {MATCH_STRUCTURED: 0, MATCH_METADATA: 0, MATCH_CONTEXT: 0},
        "slots_with_members": 0,
        "slots_with_pairs": 0,
        "pairs_compared": 0,
        "skipped_no_created": 0,
        "skipped_already_linked": 0,
        "skipped_historical_current": 0,
        "skipped_contested": 0,
        "skipped_no_conflict": 0,
        "skipped_same_values": 0,
        "candidates": 0,
        "hit_limit": False,
    }
    for bucket in buckets:
        stats["buckets_seen"] += 1
        memberships = slot_memberships(bucket, registry)
        if not allow_context_only:
            memberships = {k: v for k, v in memberships.items() if v["match"] != MATCH_CONTEXT}
        if not memberships:
            continue
        stats["buckets_in_slots"] += 1
        for key, info in memberships.items():
            stats["memberships_by_match"][info["match"]] += 1
            groups.setdefault(key, []).append((bucket, info))

    candidates: list[dict] = []
    for fact_key in sorted(groups):
        members = groups[fact_key]
        if len(members) < 2:
            continue
        stats["slots_with_members"] += 1
        dated: list[tuple[datetime, dict, dict]] = []
        for bucket, info in members:
            created = bucket_created(bucket)
            if created is None:
                stats["skipped_no_created"] += 1
                continue
            dated.append((created, bucket, info))
        if len(dated) < 2:
            continue
        dated.sort(key=lambda row: (row[0], _bucket_id(row[1])), reverse=True)
        slot_had_pair = False
        # newest bucket that is not itself historical/contested is the "current" side;
        # every older bucket is a historical candidate against it.
        for idx, (newer_created, newer, newer_info) in enumerate(dated):
            newer_status = _status(newer)
            if newer_status == FACT_STATUS_HISTORICAL:
                stats["skipped_historical_current"] += 1
                continue
            if newer_status == FACT_STATUS_CONTESTED:
                stats["skipped_contested"] += 1
                continue
            for older_created, older, older_info in dated[idx + 1:]:
                stats["pairs_compared"] += 1
                if _status(older) == FACT_STATUS_CONTESTED:
                    stats["skipped_contested"] += 1
                    continue
                if _already_linked(newer, older):
                    stats["skipped_already_linked"] += 1
                    continue
                if _status(older) == FACT_STATUS_HISTORICAL:
                    # already retired by someone else; not our pair to propose
                    stats["skipped_historical_current"] += 1
                    continue
                newer_values = newer_info.get("values") or []
                older_values = older_info.get("values") or []
                if newer_values and older_values and set(newer_values) == set(older_values):
                    stats["skipped_same_values"] += 1
                    continue
                conflicts = detect_fact_conflicts(older.get("content", ""), newer.get("content", ""))
                if not conflicts and not (newer_values and older_values):
                    stats["skipped_no_conflict"] += 1
                    continue
                candidates.append({
                    "fact_key": fact_key,
                    "current_bucket_id": _bucket_id(newer),
                    "current_name": _name(newer),
                    "current_created": newer_created.isoformat(timespec="seconds"),
                    "current_match": newer_info["match"],
                    "current_values": newer_values,
                    "historical_bucket_id": _bucket_id(older),
                    "historical_name": _name(older),
                    "historical_created": older_created.isoformat(timespec="seconds"),
                    "historical_match": older_info["match"],
                    "historical_values": older_values,
                    "conflicts": conflicts,
                    "reason": REASON_SLOT_NEWER_SUPERSEDES,
                })
                slot_had_pair = True
                if len(candidates) >= limit:
                    stats["hit_limit"] = True
                    stats["candidates"] = len(candidates)
                    if slot_had_pair:
                        stats["slots_with_pairs"] += 1
                    return {"candidates": candidates, "stats": stats}
            # only the newest non-retired bucket acts as "current"; stop after it
            break
        if slot_had_pair:
            stats["slots_with_pairs"] += 1
    stats["candidates"] = len(candidates)
    return {"candidates": candidates, "stats": stats}
