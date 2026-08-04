import hashlib
import random
from pathlib import Path

import pytest

from lmc5_recall_adapter import fuse_ranked_channels
from utils import rrf_fuse_channels
from vendor.lmc5_pgvector import UPSTREAM_COMMIT, UPSTREAM_PATH
from vendor.lmc5_pgvector.recall_pipeline import RecallPipeline


ROOT = Path(__file__).resolve().parents[1]
VENDORED_PIPELINE = ROOT / "vendor" / "lmc5_pgvector" / "recall_pipeline.py"


def _assert_equivalent(channels, *, k=60):
    expected = rrf_fuse_channels(channels, k=k)
    actual = fuse_ranked_channels(channels, k=k)
    assert [bucket_id for bucket_id, _ in actual] == [
        bucket_id for bucket_id, _ in expected
    ]
    assert [score for _, score in actual] == pytest.approx(
        [score for _, score in expected],
        rel=0,
        abs=1e-15,
    )


def test_vendored_pipeline_is_pinned_byte_for_byte():
    assert UPSTREAM_COMMIT == "53a4aaa944cdc64a1a56eaf62aee0a67d59a46f1"
    assert UPSTREAM_PATH == "extras/pgvector_backend/recall_pipeline.py"
    assert hashlib.sha256(VENDORED_PIPELINE.read_bytes()).hexdigest() == (
        "bbdc108fcab8572134db69c3f4ea3ae8a2b2a796d1029546d7225d7cf07cd567"
    )


@pytest.mark.parametrize(
    "channels,k",
    [
        ([], 60),
        ([([], 1.0), ([], 2.0)], 60),
        (
            [
                ([('keyword-a', 0.9), ('shared', 0.8)], 1.4),
                ([('shared', 0.99), ('vector-b', 0.7)], 0.8),
            ],
            60,
        ),
        (
            [
                ([('keyword-a', 0.9), ('shared', 0.8)], 1.0),
                ([('vector-a', 0.9), ('shared', 0.8)], 1.0),
                ([('entity-a', 0.9), ('shared', 0.8)], 0.65),
            ],
            42,
        ),
        (
            [
                ([('first', 1.0), ('first', 0.5), ('late', 0.1)], 1.0),
                ([('second', 1.0)], 1.0),
            ],
            60,
        ),
    ],
)
def test_upstream_adapter_preserves_existing_rrf_contract(channels, k):
    _assert_equivalent(channels, k=k)


def test_upstream_adapter_matches_legacy_across_ranked_samples():
    rng = random.Random(20260804)
    bucket_ids = [f"bucket-{index}" for index in range(12)]

    for _ in range(100):
        channels = []
        for _channel in range(rng.randint(0, 4)):
            ranked = [
                (rng.choice(bucket_ids), rng.random())
                for _item in range(rng.randint(0, 12))
            ]
            channels.append((ranked, rng.uniform(0.1, 2.0)))
        _assert_equivalent(channels, k=rng.randint(1, 100))


def test_adapter_calls_the_vendored_upstream_fusion(monkeypatch):
    calls = []
    original_fusion = RecallPipeline._apply_score_fusion
    original_merge = RecallPipeline._merge_dedup

    def record_fusion(self, channels):
        calls.append("fusion")
        return original_fusion(self, channels)

    def record_merge(self, channels):
        calls.append("merge")
        return original_merge(self, channels)

    monkeypatch.setattr(RecallPipeline, "_apply_score_fusion", record_fusion)
    monkeypatch.setattr(RecallPipeline, "_merge_dedup", record_merge)

    result = fuse_ranked_channels([([("bucket-a", 1.0)], 1.0)])

    assert result[0][0] == "bucket-a"
    assert calls == ["fusion", "merge"]
