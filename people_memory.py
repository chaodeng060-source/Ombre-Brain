"""Evidence-backed people cards, derived from existing identities and buckets.

This is a read view, not another fact authority. Quotes are explicitly memory
body excerpts, not claimed original dialogue. Co-mention never creates a
relationship. Every inferred display row retains its bucket/source pointers.
"""
from __future__ import annotations

import asyncio
import json
import re
import stat
from collections import OrderedDict, defaultdict
from pathlib import Path

from entity_store import ENTITY_TYPES, content_sha256, entity_mention_present, normalize_term
from maintenance_barrier import MaintenanceBarrier
from redact import redact_obj
from storage_safety import atomic_write_text
from utils import world_matches, event_at_from_metadata


def wants_people_profile(query: str) -> bool:
    return bool(re.search(
        r"是谁|谁是|什么人|什么关系|人物|身份|昵称|别名|叫什么|介绍.{0,18}",
        str(query or ""),
    ))


def _sentences(content: str) -> list[str]:
    return [value.strip() for value in re.findall(r"[^\n。！？!?]+[。！？!?]?", content)
            if value.strip()]


def _historical(meta: dict) -> bool:
    lifecycle = meta.get("lifecycle") if isinstance(meta.get("lifecycle"), dict) else {}
    statuses = [meta.get("version_status"), meta.get("fact_status"),
                meta.get("validity_state"), lifecycle.get("status")]
    return any(str(value).lower() in {"historical", "superseded", "retired"}
               for value in statuses) or meta.get("active_fact") is False


def _eligible(bucket: dict, world_filter, include_history: bool) -> bool:
    meta = bucket.get("metadata") or {}
    if meta.get("digested") or meta.get("exact_duplicate_of"):
        return False
    if world_filter is not None and not world_matches(meta.get("world", ""), set(world_filter)):
        return False
    if not include_history and (meta.get("type") in {"archive", "archived"} or meta.get("resolved") or _historical(meta)):
        return False
    return True


def _source_refs(meta: dict) -> dict:
    """Carry existing pointers; absence remains absence, never a fake ref."""
    return {key: meta[key] for key in (
        "source_ref", "source_refs", "source_event_ids", "source_chunk_ids",
        "source_session_id", "source_kind", "source_agent", "raw_source",
    ) if key in meta}


def _alias_owners(catalog):
    owners = defaultdict(set)
    displays = {}
    for person in catalog:
        for alias in (person.canonical_name, *person.aliases):
            owners[normalize_term(alias)].add(person.entity_id)
            displays.setdefault(normalize_term(alias), alias)
    unique = {key: next(iter(ids)) for key, ids in owners.items() if len(ids) == 1}
    return unique, displays


def _relationship_pattern(owners, displays):
    aliases = sorted((displays[key] for key in owners), key=len, reverse=True)
    if not aliases:
        return None
    names = "(?:" + "|".join(re.escape(alias) for alias in aliases) + ")"
    roles = "恋人|男朋友|女朋友|丈夫|妻子|朋友|同事|同学|哥哥|弟弟|姐姐|妹妹|父亲|母亲|老师|学生|邻居"
    return re.compile(
        rf"(?P<subject>{names})(?:\*\*|__)?\s*(?:是|成了|成为(?:了)?)\s*(?:\*\*|__)?"
        rf"(?P<object>{names})(?:\*\*|__)?\s*的\s*(?P<relation>{roles})(?!的|吗|么)", re.IGNORECASE,
    )


def _identity_score(sentence: str, aliases) -> int:
    if re.search(r"不是|并非|不再|曾经|以前|原来", sentence):
        # Retain the exact statement; do not transform negation/history into
        # an asserted present relationship.
        return 2
    for alias in aliases:
        if re.search(re.escape(alias) + r"(?:\*\*|__)?\s*(?:是|叫|的(?:身份|昵称|别名))", sentence, re.I):
            return 10
    return 1


async def build_people_cards(store, bucket_mgr, *, query: str = "", world_filter=("",),
                             include_history: bool = False, limit: int = 20,
                             offset: int = 0, evidence_limit: int = 6) -> dict:
    catalog = store.catalog("person") if store is not None else []
    by_id = {person.entity_id: person for person in catalog}
    ambiguous = []
    if query:
        resolution = store.resolve_query(query) if store is not None else None
        selected_ids = list(resolution.entity_ids) if resolution else []
        ambiguous = list(resolution.ambiguous_terms) if resolution else []
        selected = [by_id[key] for key in selected_ids if key in by_id]
    else:
        selected = catalog
    total = len(selected)
    offset = max(0, int(offset))
    limit = min(100, max(1, int(limit)))
    selected = selected[offset:offset + limit]
    evidence_limit = min(20, max(1, int(evidence_limit)))
    owners, displays = _alias_owners(catalog)
    relation_pattern = _relationship_pattern(owners, displays)
    cards, edges, stale = [], {}, 0
    cache = OrderedDict()
    read_count = 0
    for person in selected:
        names = (person.canonical_name, *person.aliases)
        evidence = []
        source_buckets = 0
        linked = store.linked_sources([person.entity_id])
        for bucket_id in linked:
            if bucket_id not in cache:
                cache[bucket_id] = await bucket_mgr.get(bucket_id)
                if len(cache) > 256:
                    cache.popitem(last=False)
                read_count += 1
                if read_count % 32 == 0:
                    await asyncio.sleep(0)
            bucket = cache[bucket_id]
            if not bucket or not _eligible(bucket, world_filter, include_history):
                continue
            content = str(bucket.get("content") or "")
            if linked[bucket_id] != content_sha256(content):
                stale += 1
                continue
            meta = bucket.get("metadata") or {}
            source_buckets += 1
            date = str(event_at_from_metadata(meta) or "")
            for sentence in _sentences(content):
                if not any(entity_mention_present(sentence, name) for name in names):
                    continue
                item = {
                    "bucket_id": bucket_id, "event_at": date,
                    "world": str(meta.get("world") or ""),
                    "quote": sentence[:320], "quote_truncated": len(sentence) > 320,
                    "source_kind": "memory_body_excerpt", "source_refs": _source_refs(meta),
                    "historical": _historical(meta),
                    "identity_score": _identity_score(sentence, names),
                }
                evidence.append(item)
                if relation_pattern is not None and not re.search(
                    r"不是|并非|不再|曾经|以前|原来|如果|假如|要是|也许|可能|？|\?", sentence,
                ):
                    for match in relation_pattern.finditer(sentence):
                        left = owners.get(normalize_term(match["subject"]))
                        right = owners.get(normalize_term(match["object"]))
                        if not left or not right or left == right:
                            continue
                        key = (left, match["relation"], right, item["world"], item["historical"])
                        edge = edges.setdefault(key, {
                            "subject": left, "subject_name": by_id[left].canonical_name,
                            "relation": match["relation"], "object": right,
                            "object_name": by_id[right].canonical_name,
                            "world": item["world"], "historical": item["historical"],
                            "basis": "explicit_memory_statement", "evidence": [],
                        })
                        if not any(e["bucket_id"] == bucket_id for e in edge["evidence"]):
                            edge["evidence"].append(item)
                            edge["evidence"].sort(key=lambda value: value["event_at"], reverse=True)
                            del edge["evidence"][3:]
            # Keep only a bounded best-evidence set per person, not copies of
            # every mentioning paragraph in a large source history.
            evidence.sort(key=lambda row: (row["identity_score"], row["event_at"], row["bucket_id"]), reverse=True)
            del evidence[evidence_limit:]
        # A registry identity with no evidence in this world is not exposed
        # across world boundaries merely because it exists in the database.
        if source_buckets:
            cards.append({
                "entity_id": person.entity_id, "name": person.canonical_name,
                "aliases": list(dict.fromkeys(names)), "source_bucket_count": source_buckets,
                "evidence": evidence, "evidence_truncated": source_buckets > len(evidence),
            })
    return redact_obj({
        "schema": "ombre.people-evidence/v1", "people": cards,
        "relations": list(edges.values()), "matched_registry_entries": total,
        "offset": offset, "limit": limit, "next_offset": offset + limit if offset + limit < total else None,
        "ambiguous_terms": ambiguous, "stale_links_skipped": stale,
        "source_kind": "derived_memory_view", "memory_written": False,
    })


def evidence_ranked_ids(cards: dict) -> list[tuple[str, float]]:
    """Only explicit identity statements seed the identity-query channel."""
    rows = [row for card in cards.get("people", []) for row in card.get("evidence", [])
            if row.get("identity_score", 0) >= 10]
    rows.sort(key=lambda row: (row["identity_score"], row["event_at"]), reverse=True)
    seen, ranked = set(), []
    for row in rows:
        if row["bucket_id"] not in seen:
            seen.add(row["bucket_id"])
            ranked.append((row["bucket_id"], 1.0))
    return ranked


_PEOPLE_REFRESH_STATE = "people-refresh-catalog.json"


def _declared_catalog(store, people=None):
    # refresh_declared_links considers every entity type. A place/project
    # alias can also make a person's formerly unique alias ambiguous.
    rows = list(store.catalog("person") if people is None else people)
    for entity_type in sorted(ENTITY_TYPES - {"person"}):
        rows.extend(store.catalog(entity_type))
    return rows


def _catalog_signature(catalog) -> str:
    rows = sorted(
        (row.entity_id, row.entity_type, row.canonical_name, sorted(set(row.aliases)))
        for row in catalog
    )
    return content_sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))


def _read_catalog_signature(path: Path) -> str | None:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            return None
        signature = state.get("catalog_sha256")
        if isinstance(signature, str) and re.fullmatch(r"[0-9a-f]{64}", signature):
            return signature
    except (OSError, ValueError, UnicodeError):
        pass
    return None


async def refresh_people_links(store, bucket_mgr) -> dict:
    """Refresh declared links, rebuilding old bodies when the catalog changes.

    One fixed 0600 derived watermark stores only the catalog hash. Missing or
    invalid state forces a rebuild. It is invalidated before the first batch
    write and published atomically only after every batch succeeds against a
    stable catalog. Failures/cancellation therefore force a complete retry,
    even when earlier batches already carry the current body hashes.
    """
    if store is None or not store.catalog("person"):
        return {"checked": 0, "refreshed": 0, "memory_written": False}
    # Reuses the night run's same-task exclusive lease; standalone refreshes
    # also serialize the index+watermark window across processes.
    async with MaintenanceBarrier(store.buckets_dir).exclusive_async(timeout=30):
        people = store.catalog("person")
        if not people:
            return {"checked": 0, "refreshed": 0, "memory_written": False}
        signature = _catalog_signature(_declared_catalog(store, people))
        path = Path(store.entities_dir) / _PEOPLE_REFRESH_STATE
        rebuild = _read_catalog_signature(path) != signature
        checked = refreshed = 0
        sources = {} if rebuild else store.linked_sources()
        batch = []
        invalidated = False

        def flush() -> int:
            nonlocal invalidated
            if not batch:
                return 0
            if not invalidated:
                path.unlink(missing_ok=True)
                invalidated = True
            count = store.refresh_declared_links(batch)
            batch.clear()
            return count

        for bucket in await bucket_mgr.list_all(include_archive=True, include_nsfw=True):
            content = str(bucket.get("content") or "")
            checked += 1
            if rebuild or sources.get(bucket["id"]) != content_sha256(content):
                batch.append((bucket["id"], content))
                if len(batch) >= 100:
                    refreshed += flush()
            if checked % 32 == 0:
                await asyncio.sleep(0)
        refreshed += flush()
        if _catalog_signature(_declared_catalog(store)) != signature:
            path.unlink(missing_ok=True)
            raise RuntimeError("people catalog changed during refresh; retry required")
        if rebuild or invalidated:
            # Remove invalid/old files first, so the existing atomic writer's
            # mkstemp creates 0600 rather than preserving an invalid old mode.
            path.unlink(missing_ok=True)
            atomic_write_text(path, json.dumps({
                "schema_version": 1, "catalog_sha256": signature,
            }, separators=(",", ":")))
        return {"checked": checked, "refreshed": refreshed, "memory_written": False}
