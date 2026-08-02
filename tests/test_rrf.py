# ============================================================
# Test: RRF Fusion Algorithm — pure local, no LLM needed
# 测试：RRF 融合算法 —— 纯本地,不需要 LLM
#
# Verifies utils.rrf_fuse:
#   - empty inputs → empty output
#   - single channel only → channel's ordering preserved
#   - identical orderings → preserved
#   - complementary (no overlap) → both top buckets surface
#   - opposing orderings → fused mid-rank bucket wins
#   - asymmetric weights → respect per-channel weight
#   - k parameter → larger k flattens score gaps
#
# Pure function tests — no fixtures, no I/O, no LLM. Run fast.
# ============================================================

import pytest

from utils import rrf_fuse, rrf_fuse_channels


def test_rrf_empty_inputs():
    assert rrf_fuse([], []) == []


def test_rrf_keyword_only():
    keyword = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    fused = rrf_fuse(keyword, [])
    assert [bid for bid, _ in fused] == ["a", "b", "c"]


def test_rrf_vector_only():
    vector = [("x", 0.95), ("y", 0.8)]
    fused = rrf_fuse([], vector)
    assert [bid for bid, _ in fused] == ["x", "y"]


def test_rrf_identical_orderings():
    keyword = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    vector = [("a", 0.95), ("b", 0.8), ("c", 0.6)]
    fused = rrf_fuse(keyword, vector)
    assert [bid for bid, _ in fused[:3]] == ["a", "b", "c"]


def test_rrf_complementary():
    # No overlap between channels — both top buckets should surface near front
    keyword = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    vector = [("x", 0.95), ("y", 0.8)]
    fused = rrf_fuse(keyword, vector)
    bids = [bid for bid, _ in fused]
    assert "a" in bids[:2]
    assert "x" in bids[:2]
    assert len(fused) == 5


def test_rrf_dual_channel_wins():
    # b appears in both channels → should beat a (kw-only) and c (vec-only)
    # core RRF property: rewards documents present in both rankings
    keyword = [("a", 0.9), ("b", 0.85)]
    vector = [("b", 0.95), ("c", 0.9)]
    fused = rrf_fuse(keyword, vector)
    assert fused[0][0] == "b"


def test_rrf_asymmetric_weights():
    keyword = [("a", 0.9)]
    vector = [("x", 0.95)]
    fused_kw = rrf_fuse(keyword, vector, keyword_weight=2.0, vector_weight=1.0)
    assert fused_kw[0][0] == "a"
    fused_vec = rrf_fuse(keyword, vector, keyword_weight=1.0, vector_weight=2.0)
    assert fused_vec[0][0] == "x"


def test_rrf_k_parameter():
    # Larger k flattens score differences between consecutive ranks
    pairs = [(f"b{i}", 1.0 - i * 0.1) for i in range(5)]
    fused_small_k = rrf_fuse(pairs, [], k=10)
    fused_large_k = rrf_fuse(pairs, [], k=1000)
    gap_small = fused_small_k[0][1] - fused_small_k[1][1]
    gap_large = fused_large_k[0][1] - fused_large_k[1][1]
    assert gap_large < gap_small


def test_rrf_returns_sorted_descending():
    keyword = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    vector = [("c", 0.95), ("a", 0.8)]
    fused = rrf_fuse(keyword, vector)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_wrapper_preserves_exact_two_channel_result():
    keyword = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    vector = [("c", 0.95), ("a", 0.8), ("d", 0.6)]

    assert rrf_fuse(keyword, vector) == [
        ("a", 1.0 / 61 + 1.0 / 62),
        ("c", 1.0 / 63 + 1.0 / 61),
        ("b", 1.0 / 62),
        ("d", 1.0 / 63),
    ]


def test_rrf_wrapper_preserves_legacy_duplicate_accumulation():
    keyword = [("a", 1.0), ("a", 0.5), ("b", 0.4)]

    assert rrf_fuse(keyword, []) == [
        ("a", 1.0 / 61 + 1.0 / 62),
        ("b", 1.0 / 63),
    ]


def test_rrf_three_channel_consensus_accumulates():
    channels = [
        ([("keyword-only", 1.0), ("shared", 0.9)], 1.0),
        ([("vector-only", 1.0), ("shared", 0.9)], 1.0),
        ([("entity-only", 1.0), ("shared", 0.9)], 1.0),
    ]

    fused = rrf_fuse_channels(channels)

    assert fused[0] == ("shared", 3.0 / 62)
    assert {bucket_id for bucket_id, _ in fused} == {
        "shared",
        "keyword-only",
        "vector-only",
        "entity-only",
    }


def test_rrf_empty_entity_channel_preserves_two_channel_result():
    keyword = [("a", 0.9), ("b", 0.7)]
    vector = [("b", 0.95), ("c", 0.8)]

    assert rrf_fuse_channels(
        [(keyword, 1.5), (vector, 0.75), ([], 2.0)],
        k=42,
    ) == rrf_fuse(
        keyword,
        vector,
        k=42,
        keyword_weight=1.5,
        vector_weight=0.75,
    )


def test_rrf_channels_deduplicates_and_breaks_ties_stably():
    channels = [
        ([("first", 1.0), ("first", 0.5)], 1.0),
        ([("second", 1.0)], 1.0),
    ]

    assert rrf_fuse_channels(channels) == [
        ("first", 1.0 / 61),
        ("second", 1.0 / 61),
    ]
