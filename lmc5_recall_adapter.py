"""Thin Ombre adapter for the vendored LMC-5 recall fusion.

Retrieval stays in Ombre because its stores and guards are asynchronous and
production-specific. Only the rank fusion and cross-channel id merge are
delegated to the byte-identical upstream module.
"""

from __future__ import annotations

from typing import Any

from vendor.lmc5_pgvector.recall_pipeline import RecallHit, RecallPipeline


def fuse_ranked_channels(
    channels: list[tuple[list[tuple[Any, Any]], float]],
    *,
    k: int = 60,
) -> list[tuple[Any, float]]:
    """Fuse Ombre ranked bucket channels through upstream LMC-5 RRF.

    Ombre bucket ids are strings while upstream ids are integers. The mapping
    is call-local and deterministic. Duplicate positions are represented by
    disposable sentinel hits so a duplicate still consumes its original rank
    without casting a second vote, exactly matching Ombre's previous contract.
    """
    source_to_upstream: dict[Any, int] = {}
    upstream_to_source: dict[int, Any] = {}
    adapted_channels: list[tuple[str, list[RecallHit]]] = []
    channel_weights: dict[str, float] = {}
    next_source_id = 1
    next_sentinel_id = -1

    for channel_index, (ranked_items, weight) in enumerate(channels):
        channel_name = f"ombre_{channel_index}"
        channel_weights[channel_name] = float(weight)
        seen_in_channel: set[Any] = set()
        hits: list[RecallHit] = []
        item_count = len(ranked_items)

        for position, item in enumerate(ranked_items):
            source_id = item[0]
            if source_id in seen_in_channel:
                upstream_id = next_sentinel_id
                next_sentinel_id -= 1
            else:
                seen_in_channel.add(source_id)
                if source_id not in source_to_upstream:
                    source_to_upstream[source_id] = next_source_id
                    upstream_to_source[next_source_id] = source_id
                    next_source_id += 1
                upstream_id = source_to_upstream[source_id]

            # Upstream ranks by hit.score before applying RRF. Synthetic scores
            # preserve the caller's supplied order; source scores stay unused.
            hits.append(
                RecallHit(
                    source_id=upstream_id,
                    title="",
                    content="",
                    score=float(item_count - position),
                    channel=channel_name,
                    metadata={"namespace": "curated"},
                )
            )
        adapted_channels.append((channel_name, hits))

    pipeline = RecallPipeline(
        fusion="rrf",
        channel_weights=channel_weights,
        rrf_k=int(k),
        content_fingerprint=None,
    )
    fused_channels = pipeline._apply_score_fusion(adapted_channels)
    merged = pipeline._merge_dedup(fused_channels)
    merged.sort(key=lambda hit: hit.score, reverse=True)

    return [
        (upstream_to_source[int(hit.source_id)], float(hit.score))
        for hit in merged
        if int(hit.source_id) in upstream_to_source
    ]
