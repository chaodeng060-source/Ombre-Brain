import pytest

import server


def _bucket(bucket_id, content, *, domain=None, score=0):
    return {
        "id": bucket_id,
        "content": content,
        "score": score,
        "metadata": {
            "id": bucket_id,
            "name": f"name-{bucket_id}",
            "tags": ["old-tag"],
            "domain": domain or ["health"],
            "type": "dynamic",
            "valence": 0.5,
            "arousal": 0.3,
            "importance": 5,
        },
    }


class FakeBucketMgr:
    def __init__(self, buckets, keyword_matches=None):
        self.buckets = {b["id"]: b for b in buckets}
        self.keyword_matches = keyword_matches or []
        self.updates = []
        self.created = []

    async def search(self, *args, **kwargs):
        return list(self.keyword_matches)

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)

    async def update(self, bucket_id, **kwargs):
        self.updates.append((bucket_id, kwargs))
        bucket = self.buckets[bucket_id]
        if "content" in kwargs:
            bucket["content"] = kwargs["content"]
        for key, value in kwargs.items():
            if key != "content":
                bucket["metadata"][key] = value
        return True

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return "new-bucket"


class FakeEmbedding:
    def __init__(self, hits):
        self.hits = hits
        self.generated = []

    async def search_similar(self, query, top_k=8):
        return list(self.hits)

    async def generate_and_store(self, bucket_id, content):
        self.generated.append((bucket_id, content))


class FakeDehydrator:
    def __init__(self):
        self.merge_calls = []

    async def merge(self, old_content, new_content):
        self.merge_calls.append((old_content, new_content))
        return new_content


@pytest.fixture
def merge_config(monkeypatch):
    monkeypatch.setitem(server.config, "merge_threshold", 90)
    monkeypatch.setitem(
        server.config,
        "merge",
        {
            "keyword_limit": 5,
            "vector_limit": 8,
            "vector_floor": 0.50,
            "vector_threshold": 0.80,
            "candidate_limit": 8,
        },
    )


@pytest.mark.asyncio
async def test_vector_only_conflict_merge_writes_supersedes_audit(monkeypatch, merge_config):
    old = _bucket("old", "period_start: 2026-05-03\nflow: light", domain=["health"])
    mgr = FakeBucketMgr([old], keyword_matches=[])
    emb = FakeEmbedding([("old", 0.93)])
    dh = FakeDehydrator()
    monkeypatch.setattr(server, "bucket_mgr", mgr)
    monkeypatch.setattr(server, "embedding_engine", emb)
    monkeypatch.setattr(server, "dehydrator", dh)

    bucket_id, display, is_merged = await server._merge_or_create(
        content="period_start: 2026-05-04\nflow: light",
        tags=["new-tag"],
        importance=6,
        domain=["health"],
        valence=0.7,
        arousal=0.4,
        name="period",
    )

    assert (bucket_id, display, is_merged) == ("old", "name-old", True)
    assert mgr.created == []
    assert emb.generated == [("old", "period_start: 2026-05-04\nflow: light")]
    assert "[ARBITRATION_CONTEXT]" in dh.merge_calls[0][1]

    update = mgr.updates[-1][1]
    assert update["content"] == "period_start: 2026-05-04\nflow: light"
    assert "new-tag" in update["tags"]
    assert update["importance"] == 6
    fields = {entry["field"]: entry for entry in update["supersedes"]}
    assert fields["period_start"]["old"] == "2026-05-03"
    assert fields["period_start"]["new"] == "2026-05-04"
    assert fields["period_start"]["bucket_id"] == "old"


@pytest.mark.asyncio
async def test_protected_domain_vector_hit_creates_new_bucket(monkeypatch, merge_config):
    protected_domain = sorted(server.PROTECTED_RESOLVE_DOMAINS)[0]
    old = _bucket("old", "rule: keep original wording", domain=[protected_domain])
    mgr = FakeBucketMgr([old], keyword_matches=[])
    emb = FakeEmbedding([("old", 0.98)])
    dh = FakeDehydrator()
    monkeypatch.setattr(server, "bucket_mgr", mgr)
    monkeypatch.setattr(server, "embedding_engine", emb)
    monkeypatch.setattr(server, "dehydrator", dh)

    bucket_id, display, is_merged = await server._merge_or_create(
        content="rule: drifted wording",
        tags=[],
        importance=5,
        domain=[protected_domain],
        valence=0.5,
        arousal=0.3,
        name="protected",
    )

    assert (bucket_id, display, is_merged) == ("new-bucket", "protected", False)
    assert mgr.updates == []
    assert dh.merge_calls == []
    assert mgr.created[-1]["content"] == "rule: drifted wording"
