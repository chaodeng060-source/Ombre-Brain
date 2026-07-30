from __future__ import annotations

import pytest

from recall_support import (
    rank_within_relevance_bands,
    retain_original_query_supported_candidates,
)


def test_secondary_signal_cannot_cross_relevance_band():
    rows = [
        {"id": "relevant", "relevance": 50.0, "secondary": 0.0},
        {"id": "emotional_noise", "relevance": 33.33, "secondary": 100.0},
    ]

    ranked = rank_within_relevance_bands(
        rows,
        relevance_score=lambda row: row["relevance"],
        tie_break_score=lambda row: row["secondary"],
        band_width=3.0,
    )

    assert [row["id"] for row in ranked] == ["relevant", "emotional_noise"]


def test_secondary_signal_may_rerank_inside_close_band():
    rows = [
        {"id": "slightly_higher", "relevance": 50.0, "secondary": 10.0},
        {"id": "better_tie_break", "relevance": 48.0, "secondary": 90.0},
        {"id": "outside_band", "relevance": 45.0, "secondary": 100.0},
    ]

    ranked = rank_within_relevance_bands(
        rows,
        relevance_score=lambda row: row["relevance"],
        tie_break_score=lambda row: row["secondary"],
        band_width=3.0,
    )

    assert [row["id"] for row in ranked] == [
        "better_tie_break",
        "slightly_higher",
        "outside_band",
    ]


def test_expansion_only_drift_is_removed():
    rows = [
        {"id": "ankle", "literal": 50.0, "original_vector": 0.0},
        {
            "id": "sexual_vector_drift",
            "literal": 33.33,
            "original_vector": 0.0,
        },
    ]

    kept = retain_original_query_supported_candidates(
        rows,
        literal_score=lambda row: row["literal"],
        original_vector_score=lambda row: row["original_vector"],
        literal_floor=40.0,
    )

    assert [row["id"] for row in kept] == ["ankle"]


def test_original_literal_or_vector_support_survives():
    rows = [
        {"id": "literal", "literal": 44.44, "original_vector": 0.0},
        {"id": "semantic", "literal": 12.0, "original_vector": 0.72},
        {"id": "expansion_only", "literal": 12.0, "original_vector": 0.0},
    ]

    kept = retain_original_query_supported_candidates(
        rows,
        literal_score=lambda row: row["literal"],
        original_vector_score=lambda row: row["original_vector"],
        literal_floor=40.0,
    )

    assert [row["id"] for row in kept] == ["literal", "semantic"]


@pytest.mark.asyncio
async def test_bucket_search_uses_topic_threshold_in_relevance_mode(
    bucket_mgr,
    monkeypatch,
):
    relevant = {
        "id": "0a9042bbccab",
        "metadata": {
            "importance": 1,
            "valence": 0.5,
            "arousal": 0.3,
            "last_active": "2020-01-01T00:00:00",
        },
        "content": "脚踝",
    }
    emotional_noise = {
        "id": "6d2514690d7d",
        "metadata": {
            "importance": 10,
            "valence": 0.8,
            "arousal": 0.7,
            "last_active": "2099-01-01T00:00:00",
        },
        "content": "射精值系统",
    }

    async def fake_list_all(*, include_archive=False):
        assert include_archive is False
        return [emotional_noise, relevant]

    monkeypatch.setattr(bucket_mgr, "list_all", fake_list_all)
    monkeypatch.setattr(
        bucket_mgr,
        "_calc_topic_score",
        lambda _query, bucket: (
            0.50 if bucket["id"] == "0a9042bbccab" else 0.3333
        ),
    )

    results = await bucket_mgr.search(
        "你什么时候给我收拾过脚踝",
        limit=10,
        relevance_first=True,
    )

    assert [row["id"] for row in results] == ["0a9042bbccab"]


@pytest.mark.asyncio
async def test_relevance_mode_keeps_useful_literal_candidate_below_legacy_floor(
    bucket_mgr,
    monkeypatch,
):
    sister = {
        "id": "6a881e94365f",
        "metadata": {
            "importance": 5,
            "valence": 0.5,
            "arousal": 0.3,
        },
        "content": "朝灯有一个姐姐",
    }

    async def fake_list_all(*, include_archive=False):
        assert include_archive is False
        return [sister]

    monkeypatch.setattr(bucket_mgr, "list_all", fake_list_all)
    monkeypatch.setattr(bucket_mgr, "_calc_topic_score", lambda _query, _bucket: 0.4444)

    results = await bucket_mgr.search(
        "你记不记得我有一个姐姐",
        limit=10,
        relevance_first=True,
    )

    assert [row["id"] for row in results] == ["6a881e94365f"]
