"""Pure helpers for multi-query recall and typed relation traversal.

This module deliberately owns no storage, model, or MCP state.  The server
passes it ranked channel results and already-loaded Markdown buckets.  Keeping
the graph walk pure makes the recall contract testable without importing the
full service runtime.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


DEFAULT_RELATION_TYPE_WEIGHTS = {
    "kin": 1.0,
    "explains": 0.85,
}


@dataclass(frozen=True)
class RelationNeighbor:
    bucket_id: str
    via_id: str
    relation_type: str
    depth: int
    strength: float
    score: float
    direction: str


def merge_ranked_lists(
    ranked_lists: Sequence[Sequence[tuple[str, float] | str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple rankings with reciprocal-rank fusion.

    The input scores are intentionally ignored.  This is used inside one
    channel to combine the original query and optional expansion angles before
    the existing keyword/vector channel fusion runs.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            bucket_id = str(item[0] if isinstance(item, tuple) else item)
            if not bucket_id or bucket_id in seen_in_list:
                continue
            seen_in_list.add(bucket_id)
            scores[bucket_id] = scores.get(bucket_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def expand_relation_graph(
    buckets: Iterable[dict],
    seed_ids: Iterable[str],
    *,
    allowed_types: Iterable[str],
    max_depth: int = 2,
    max_results: int = 5,
    allowed_node_ids: set[str] | None = None,
    hop_min_strength: dict[int, float] | None = None,
    type_weights: dict[str, float] | None = None,
) -> list[RelationNeighbor]:
    """Walk typed Markdown relation edges in both directions for at most 2 hops.

    Existing Ombre edges do not carry a strength field, so they retain full
    strength (1.0).  Newer weighted edges may set ``strength`` in ``[0, 1]``.
    Invalid strengths and missing endpoints are ignored rather than weakening
    the whole recall request.
    """
    allowed = {str(value) for value in allowed_types}
    if not allowed or max_results <= 0:
        return []

    depth_limit = max(0, min(int(max_depth or 0), 2))
    if depth_limit == 0:
        return []

    bucket_map = {
        str(bucket.get("id")): bucket
        for bucket in buckets
        if isinstance(bucket, dict) and bucket.get("id")
    }
    seeds = {str(value) for value in seed_ids if str(value) in bucket_map}
    if not seeds:
        return []

    eligible = set(bucket_map) if allowed_node_ids is None else set(allowed_node_ids) & set(bucket_map)
    eligible |= seeds
    thresholds = {1: 0.4, 2: 0.7, **(hop_min_strength or {})}
    weights = {**DEFAULT_RELATION_TYPE_WEIGHTS, **(type_weights or {})}

    adjacency: dict[str, list[tuple[str, str, float, str]]] = defaultdict(list)
    for source_id, bucket in bucket_map.items():
        relations = (bucket.get("metadata", {}) or {}).get("relations") or []
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            relation_type = str(relation.get("type") or "")
            target_id = str(relation.get("target") or "")
            if relation_type not in allowed or not target_id or target_id == source_id:
                continue
            if target_id not in bucket_map:
                continue
            try:
                strength = float(relation.get("strength", 1.0))
            except (TypeError, ValueError):
                continue
            if not 0.0 <= strength <= 1.0:
                continue
            adjacency[source_id].append((target_id, relation_type, strength, "out"))
            adjacency[target_id].append((source_id, relation_type, strength, "in"))

    frontier = set(seeds)
    visited_depth = {seed_id: 0 for seed_id in seeds}
    found: dict[str, RelationNeighbor] = {}

    for depth in range(1, depth_limit + 1):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for current_id in sorted(frontier):
            for target_id, relation_type, strength, direction in adjacency.get(current_id, []):
                if target_id in seeds or target_id not in eligible:
                    continue
                if strength < float(thresholds.get(depth, 0.0)):
                    continue
                previous_depth = visited_depth.get(target_id)
                if previous_depth is not None and previous_depth < depth:
                    continue
                score = strength * float(weights.get(relation_type, 0.5)) * (1.0 if depth == 1 else 0.6)
                candidate = RelationNeighbor(
                    bucket_id=target_id,
                    via_id=current_id,
                    relation_type=relation_type,
                    depth=depth,
                    strength=strength,
                    score=round(score, 6),
                    direction=direction,
                )
                previous = found.get(target_id)
                if previous is None or candidate.score > previous.score:
                    found[target_id] = candidate
                if previous_depth is None:
                    visited_depth[target_id] = depth
                    next_frontier.add(target_id)
        frontier = next_frontier

    return sorted(
        found.values(),
        key=lambda item: (-item.score, item.depth, item.bucket_id),
    )[:max_results]
