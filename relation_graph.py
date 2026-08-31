"""Deterministic, auditable Y-axis relation planning.

The Markdown bucket frontmatter is Ombre's relation authority.  This module
builds a sparse graph only from evidence already carried by that authority:

* explicit bucket references (episode, saga, and E-axis source anchors),
* shared raw-event or source-digest provenance, and
* a non-``other`` thread which has already been assigned upstream.

No model, embedding, network request, or content similarity participates in
the decision.  Unsupported buckets are reported, not force-linked.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from utils import event_at_from_metadata


RELATION_GRAPH_SCHEMA = "ombre.relation-graph/v1"
LEGACY_GENERATION_METHOD = "legacy:unattributed:v1"
EXPLICIT_GENERATION_METHOD = "deterministic:explicit-reference:v1"
PROVENANCE_GENERATION_METHOD = "deterministic:provenance-forest:v1"
TIMELINE_GENERATION_METHOD = "deterministic:timeline-forest:v1"

_GENERATION_METHOD_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,127}")
_MAX_EVIDENCE_BYTES = 4096
_MAX_EVIDENCE_DEPTH = 4
_MAX_EVIDENCE_ITEMS = 32
_MAX_EVIDENCE_TEXT = 512
_METHOD_PRIORITY = {
    TIMELINE_GENERATION_METHOD: 1,
    PROVENANCE_GENERATION_METHOD: 2,
    EXPLICIT_GENERATION_METHOD: 3,
}


@dataclass(frozen=True)
class PlannedRelation:
    source_id: str
    target_id: str
    relation_type: str
    note: str
    strength: float
    evidence: dict[str, Any]
    generation_method: str

    def edge_document(self) -> dict[str, Any]:
        return {
            "type": self.relation_type,
            "target": self.target_id,
            "note": self.note,
            "strength": self.strength,
            "evidence": self.evidence,
            "generation_method": self.generation_method,
        }


@dataclass(frozen=True)
class RelationGraphPlan:
    schema: str
    input_count: int
    eligible_count: int
    unsupported_count: int
    skipped_by_reason: dict[str, int]
    relations: tuple[PlannedRelation, ...]

    @property
    def relation_type_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(item.relation_type for item in self.relations).items()))

    @property
    def generation_method_counts(self) -> dict[str, int]:
        return dict(
            sorted(Counter(item.generation_method for item in self.relations).items())
        )


class _DisjointSet:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        parent = self._parent.setdefault(value, value)
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root
        return True


def normalize_generation_method(value: Any) -> str:
    method = str(value or "").strip()
    if not _GENERATION_METHOD_RE.fullmatch(method):
        raise ValueError("invalid relation generation_method")
    return method


def _normalize_evidence_value(value: Any, *, depth: int) -> Any:
    if depth > _MAX_EVIDENCE_DEPTH:
        raise ValueError("relation evidence is too deeply nested")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or len(cleaned) > _MAX_EVIDENCE_TEXT:
            raise ValueError("relation evidence text is empty or too long")
        return cleaned
    if isinstance(value, Mapping):
        if not value or len(value) > _MAX_EVIDENCE_ITEMS:
            raise ValueError("relation evidence object has invalid size")
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip()
            if not key or len(key) > 64:
                raise ValueError("relation evidence has an invalid key")
            result[key] = _normalize_evidence_value(raw_value, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if not value or len(value) > _MAX_EVIDENCE_ITEMS:
            raise ValueError("relation evidence list has invalid size")
        return [
            _normalize_evidence_value(item, depth=depth + 1)
            for item in value
        ]
    raise ValueError("relation evidence contains an unsupported value")


def normalize_relation_evidence(value: Any) -> dict[str, Any]:
    normalized = _normalize_evidence_value(value, depth=0)
    if not isinstance(normalized, dict):
        raise ValueError("relation evidence must be an object")
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_EVIDENCE_BYTES:
        raise ValueError("relation evidence is too large")
    return normalized


def legacy_relation_evidence(relation: Mapping[str, Any]) -> dict[str, Any]:
    """Describe an old edge honestly without inventing its original basis."""
    import hashlib

    note = str(relation.get("note") or "")
    evidence: dict[str, Any] = {
        "kind": "legacy_unattributed",
        "reason": "edge predates auditable Y metadata",
    }
    if note:
        evidence["legacy_note_sha256"] = hashlib.sha256(
            note.encode("utf-8")
        ).hexdigest()
    return normalize_relation_evidence(evidence)


def _metadata(bucket: Mapping[str, Any]) -> Mapping[str, Any]:
    value = bucket.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _bucket_id(bucket: Mapping[str, Any]) -> str:
    return str(bucket.get("id") or _metadata(bucket).get("id") or "").strip()


def _id_list(metadata: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = metadata.get(field)
    if not isinstance(value, list):
        return ()
    return tuple(
        dict.fromkeys(
            cleaned
            for item in value
            if (cleaned := str(item or "").strip())
        )
    )


def _bounded_basis_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty relation basis")
    if len(text) <= 256:
        return text
    import hashlib

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _order_key(bucket: Mapping[str, Any]) -> tuple[str, str]:
    metadata = _metadata(bucket)
    return (
        str(event_at_from_metadata(dict(metadata), fallback_last_active=True) or ""),
        _bucket_id(bucket),
    )


def _canonical_edge(
    source_id: str,
    relation_type: str,
    target_id: str,
) -> tuple[str, str, str]:
    if relation_type == "kin" and target_id < source_id:
        return target_id, relation_type, source_id
    return source_id, relation_type, target_id


class _Accumulator:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        method: str,
        strength: float,
        basis: Mapping[str, Any],
        note: str,
    ) -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        source_id, relation_type, target_id = _canonical_edge(
            source_id,
            relation_type,
            target_id,
        )
        key = (source_id, relation_type, target_id)
        item = self._items.setdefault(
            key,
            {
                "method": method,
                "strength": strength,
                "bases": [],
                "notes": [],
            },
        )
        if _METHOD_PRIORITY[method] > _METHOD_PRIORITY[item["method"]]:
            item["method"] = method
        item["strength"] = max(float(item["strength"]), float(strength))
        normalized_basis = normalize_relation_evidence(dict(basis))
        basis_key = json.dumps(normalized_basis, ensure_ascii=False, sort_keys=True)
        if all(
            json.dumps(old, ensure_ascii=False, sort_keys=True) != basis_key
            for old in item["bases"]
        ):
            item["bases"].append(normalized_basis)
        if note and note not in item["notes"]:
            item["notes"].append(note)

    def relations(self) -> tuple[PlannedRelation, ...]:
        planned: list[PlannedRelation] = []
        for (source_id, relation_type, target_id), item in sorted(self._items.items()):
            bases = sorted(
                item["bases"],
                key=lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            evidence = normalize_relation_evidence({
                "kind": "deterministic_relation_basis",
                "source_id": source_id,
                "target_id": target_id,
                "bases": bases,
            })
            note = "; ".join(item["notes"])
            if len(note) > 500:
                note = note[:497] + "..."
            planned.append(PlannedRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                note=note,
                strength=float(item["strength"]),
                evidence=evidence,
                generation_method=item["method"],
            ))
        return tuple(planned)


def _valid_buckets(
    buckets: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for bucket in buckets:
        if not isinstance(bucket, Mapping):
            continue
        bucket_id = _bucket_id(bucket)
        if bucket_id:
            result[bucket_id] = bucket
    return result


def plan_relation_graph(
    buckets: Sequence[Mapping[str, Any]],
) -> RelationGraphPlan:
    """Evaluate every bucket and return a sparse deterministic relation plan."""
    by_id = _valid_buckets(buckets)
    accumulator = _Accumulator()
    eligible: set[str] = set()
    missing_reference: set[str] = set()
    singleton_evidence: set[str] = set()

    event_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    digest_groups: dict[str, list[str]] = defaultdict(list)
    thread_groups: dict[str, list[str]] = defaultdict(list)

    for bucket_id, bucket in by_id.items():
        metadata = _metadata(bucket)
        for field in ("source_buckets", "episode_buckets"):
            for target_id in _id_list(metadata, field):
                if target_id not in by_id:
                    missing_reference.add(bucket_id)
                    continue
                eligible.update((bucket_id, target_id))
                accumulator.add(
                    bucket_id,
                    target_id,
                    "explains",
                    method=EXPLICIT_GENERATION_METHOD,
                    strength=1.0,
                    basis={
                        "kind": "explicit_bucket_reference",
                        "field": field,
                        "value": target_id,
                    },
                    note=f"Y v1: {field} explicitly names the target bucket.",
                )

        e_source = str(metadata.get("e_source_bucket_id") or "").strip()
        if e_source:
            if e_source not in by_id:
                missing_reference.add(bucket_id)
            else:
                eligible.update((bucket_id, e_source))
                accumulator.add(
                    bucket_id,
                    e_source,
                    "explains",
                    method=EXPLICIT_GENERATION_METHOD,
                    strength=1.0,
                    basis={
                        "kind": "explicit_bucket_reference",
                        "field": "e_source_bucket_id",
                        "value": e_source,
                    },
                    note="Y v1: E-axis record explicitly names its source bucket.",
                )

        source_session = str(metadata.get("source_session") or "").strip()
        for event_id in _id_list(metadata, "source_event_ids"):
            if source_session:
                event_groups[(source_session, event_id)].append(bucket_id)
            else:
                singleton_evidence.add(bucket_id)

        source_digest = str(metadata.get("source_digest") or "").strip()
        if source_digest:
            digest_groups[_bounded_basis_value(source_digest)].append(bucket_id)

        thread = str(metadata.get("thread") or "").strip()
        if thread and thread != "other":
            thread_groups[_bounded_basis_value(thread)].append(bucket_id)

    forest = _DisjointSet()

    def connect_group(
        members: Iterable[str],
        *,
        method: str,
        strength: float,
        basis: Mapping[str, Any],
        note: str,
    ) -> None:
        ordered = sorted(
            set(members),
            key=lambda bucket_id: _order_key(by_id[bucket_id]),
        )
        if len(ordered) < 2:
            singleton_evidence.update(ordered)
            return
        eligible.update(ordered)
        anchor = ordered[0]
        for member in ordered[1:]:
            if not forest.union(anchor, member):
                continue
            accumulator.add(
                anchor,
                member,
                "kin",
                method=method,
                strength=strength,
                basis=basis,
                note=note,
            )

    for (source_session, event_id), members in sorted(event_groups.items()):
        connect_group(
            members,
            method=PROVENANCE_GENERATION_METHOD,
            strength=1.0,
            basis={
                "kind": "shared_provenance",
                "field": "source_event_ids",
                "source_session": _bounded_basis_value(source_session),
                "value": _bounded_basis_value(event_id),
            },
            note="Y v1: both buckets derive from the same recorded source event.",
        )

    for source_digest, members in sorted(digest_groups.items()):
        connect_group(
            members,
            method=PROVENANCE_GENERATION_METHOD,
            strength=1.0,
            basis={
                "kind": "shared_provenance",
                "field": "source_digest",
                "value": source_digest,
            },
            note="Y v1: both buckets carry the same source digest.",
        )

    for thread, members in sorted(thread_groups.items()):
        connect_group(
            members,
            method=TIMELINE_GENERATION_METHOD,
            strength=0.8,
            basis={
                "kind": "shared_timeline",
                "field": "thread",
                "value": thread,
            },
            note="Y v1: both buckets belong to the same reviewed timeline thread.",
        )

    skipped = Counter()
    for bucket_id in by_id:
        if bucket_id in eligible:
            continue
        if bucket_id in missing_reference:
            skipped["missing_explicit_target"] += 1
        elif bucket_id in singleton_evidence:
            skipped["singleton_deterministic_evidence"] += 1
        else:
            skipped["no_deterministic_relation_evidence"] += 1

    return RelationGraphPlan(
        schema=RELATION_GRAPH_SCHEMA,
        input_count=len(by_id),
        eligible_count=len(eligible),
        unsupported_count=len(by_id) - len(eligible),
        skipped_by_reason=dict(sorted(skipped.items())),
        relations=accumulator.relations(),
    )


def plan_relations_for_created_bucket(
    created_bucket: Mapping[str, Any],
    buckets: Sequence[Mapping[str, Any]],
) -> tuple[PlannedRelation, ...]:
    """Plan bounded post-create links without recomputing the whole graph."""
    created_id = _bucket_id(created_bucket)
    if not created_id:
        return ()
    by_id = _valid_buckets(buckets)
    by_id[created_id] = created_bucket
    metadata = _metadata(created_bucket)
    accumulator = _Accumulator()

    for field in ("source_buckets", "episode_buckets"):
        for target_id in _id_list(metadata, field):
            if target_id in by_id and target_id != created_id:
                accumulator.add(
                    created_id,
                    target_id,
                    "explains",
                    method=EXPLICIT_GENERATION_METHOD,
                    strength=1.0,
                    basis={
                        "kind": "explicit_bucket_reference",
                        "field": field,
                        "value": target_id,
                    },
                    note=f"Y v1: {field} explicitly names the target bucket.",
                )

    e_source = str(metadata.get("e_source_bucket_id") or "").strip()
    if e_source in by_id and e_source != created_id:
        accumulator.add(
            created_id,
            e_source,
            "explains",
            method=EXPLICIT_GENERATION_METHOD,
            strength=1.0,
            basis={
                "kind": "explicit_bucket_reference",
                "field": "e_source_bucket_id",
                "value": e_source,
            },
            note="Y v1: E-axis record explicitly names its source bucket.",
        )

    session = str(metadata.get("source_session") or "").strip()
    event_ids = set(_id_list(metadata, "source_event_ids")) if session else set()
    digest = str(metadata.get("source_digest") or "").strip()
    thread = str(metadata.get("thread") or "").strip()
    if thread == "other":
        thread = ""

    event_peers: dict[str, list[str]] = defaultdict(list)
    digest_peers: list[str] = []
    thread_peers: list[str] = []
    for other_id, other in by_id.items():
        if other_id == created_id:
            continue
        other_meta = _metadata(other)
        if event_ids and str(other_meta.get("source_session") or "").strip() == session:
            for shared in event_ids.intersection(
                _id_list(other_meta, "source_event_ids")
            ):
                event_peers[shared].append(other_id)
        if digest and str(other_meta.get("source_digest") or "").strip() == digest:
            digest_peers.append(other_id)
        if thread and str(other_meta.get("thread") or "").strip() == thread:
            thread_peers.append(other_id)

    selected: set[str] = set()

    def first_peer(peers: Iterable[str]) -> str | None:
        ordered = sorted(set(peers), key=lambda value: _order_key(by_id[value]))
        return ordered[0] if ordered else None

    for event_id, peers in sorted(event_peers.items()):
        peer = first_peer(peers)
        if peer is None:
            continue
        selected.add(peer)
        accumulator.add(
            created_id,
            peer,
            "kin",
            method=PROVENANCE_GENERATION_METHOD,
            strength=1.0,
            basis={
                "kind": "shared_provenance",
                "field": "source_event_ids",
                "source_session": _bounded_basis_value(session),
                "value": _bounded_basis_value(event_id),
            },
            note="Y v1: both buckets derive from the same recorded source event.",
        )

    digest_peer = first_peer(digest_peers)
    if digest_peer is not None:
        selected.add(digest_peer)
        accumulator.add(
            created_id,
            digest_peer,
            "kin",
            method=PROVENANCE_GENERATION_METHOD,
            strength=1.0,
            basis={
                "kind": "shared_provenance",
                "field": "source_digest",
                "value": _bounded_basis_value(digest),
            },
            note="Y v1: both buckets carry the same source digest.",
        )

    thread_peer = first_peer(thread_peers)
    if thread_peer is not None and thread_peer not in selected:
        accumulator.add(
            created_id,
            thread_peer,
            "kin",
            method=TIMELINE_GENERATION_METHOD,
            strength=0.8,
            basis={
                "kind": "shared_timeline",
                "field": "thread",
                "value": _bounded_basis_value(thread),
            },
            note="Y v1: both buckets belong to the same reviewed timeline thread.",
        )

    return accumulator.relations()
