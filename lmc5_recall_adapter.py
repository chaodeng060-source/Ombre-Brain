"""Thin Ombre adapter for the vendored LMC-5 score fusion.

The adapter keeps Ombre's asynchronous stores and authority guards in place.
Only cross-channel score fusion is delegated to the vendored upstream module.
"""

from __future__ import annotations

from typing import Any

from vendor.lmc5_pgvector.recall_pipeline import RecallHit, RecallPipeline

_FUSION_MODES = frozenset({"raw", "minmax", "rrf"})


def fuse_ranked_channels(
    channels: list[tuple[list[tuple[Any, Any]], float]],
    *,
    k: int = 60,
    fusion: str = "rrf",
) -> list[tuple[Any, float]]:
    """Fuse ranked Ombre bucket channels through upstream LMC-5.

    ``rrf`` is the production-compatible default and deliberately uses only
    the caller's within-channel order. ``raw`` and ``minmax`` preserve each
    item's real source score; replacing those scores with positional numbers
    would make the three upstream modes a false comparison.
    """
    normalized_fusion = str(fusion or "rrf").strip().lower()
    if normalized_fusion not in _FUSION_MODES:
        raise ValueError(f"unsupported LMC-5 fusion mode: {fusion!r}")

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
                if normalized_fusion != "rrf":
                    # raw/minmax compare actual scores; a disposable duplicate
                    # would change the channel's min/max span. Keep the first
                    # occurrence and drop later copies instead.
                    continue
                upstream_id = next_sentinel_id
                next_sentinel_id -= 1
            else:
                seen_in_channel.add(source_id)
                if source_id not in source_to_upstream:
                    source_to_upstream[source_id] = next_source_id
                    upstream_to_source[next_source_id] = source_id
                    next_source_id += 1
                upstream_id = source_to_upstream[source_id]

            adapted_score = (
                float(item_count - position)
                if normalized_fusion == "rrf"
                else float(item[1])
            )
            hits.append(
                RecallHit(
                    source_id=upstream_id,
                    title="",
                    content="",
                    score=adapted_score,
                    channel=channel_name,
                    metadata={"namespace": "curated"},
                )
            )
        adapted_channels.append((channel_name, hits))

    pipeline = RecallPipeline(
        fusion=normalized_fusion,
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
