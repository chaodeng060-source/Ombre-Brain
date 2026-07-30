import asyncio

import pytest

import server
from bucket_manager import BucketManager


class _EmbeddingStub:
    def __init__(self):
        self.calls = []

    async def generate_and_store(self, bucket_id, content):
        self.calls.append((bucket_id, content))
        return True


def test_exact_feel_match_is_byte_for_byte_and_feel_only():
    buckets = [
        {"id": "dynamic", "content": "same", "metadata": {"type": "dynamic"}},
        {"id": "feel", "content": "same", "metadata": {"type": "feel"}},
        {"id": "spaced", "content": "same ", "metadata": {"type": "feel"}},
    ]

    assert server._find_exact_feel_bucket(buckets, "same")["id"] == "feel"
    assert server._find_exact_feel_bucket(buckets, "same\n") is None


@pytest.mark.asyncio
async def test_concurrent_exact_feel_replays_create_one_bucket(
    test_config,
    monkeypatch,
):
    manager = BucketManager(test_config)
    embeddings = _EmbeddingStub()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", embeddings)

    results = await asyncio.gather(*(
        server._create_or_reuse_feel_bucket(
            "同一份逐字正文",
            tags=[],
            domain=["沉淀物"],
            valence=0.6,
            arousal=0.4,
        )
        for _ in range(8)
    ))

    bucket_ids = {bucket_id for bucket_id, _created in results}
    assert len(bucket_ids) == 1
    assert sum(created for _bucket_id, created in results) == 1
    assert len(embeddings.calls) == 1
    feels = [
        bucket
        for bucket in await manager.list_all(include_archive=False)
        if bucket["metadata"].get("type") == "feel"
    ]
    assert [bucket["id"] for bucket in feels] == list(bucket_ids)
