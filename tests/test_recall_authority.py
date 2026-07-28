"""Fault-focused recall authority tests for Z, archive, and Y boundaries."""

from __future__ import annotations

import pytest

import server


class _NoopLoop:
    async def ensure_started(self):
        return None


class _Decay(_NoopLoop):
    @staticmethod
    def apply_retrieval_decay(score, metadata):
        return score


class _NeverDehydrate:
    async def dehydrate(self, content, metadata):
        raise AssertionError("cold/archive evidence must not reach rendering")


class _ColdBucketManager:
    def __init__(self, bucket, archive_dir):
        self.bucket = bucket
        self.archive_dir = str(archive_dir)
        self.touched = []

    async def search(self, *args, **kwargs):
        return []

    async def get(self, bucket_id):
        return self.bucket if bucket_id == self.bucket["id"] else None

    async def list_all(self, include_archive=False, **kwargs):
        return []

    async def touch(self, bucket_id):
        self.touched.append(bucket_id)


class _ColdEmbedding:
    def __init__(self, bucket_id):
        self.bucket_id = bucket_id

    async def search_similar(self, query, top_k=10):
        return [(self.bucket_id, 0.99)]

    async def search_similar_with_status(self, query, top_k=10):
        return [(self.bucket_id, 0.99)], "ok"


def _cold_bucket(bucket_id, archive_dir, *, metadata_type="dynamic"):
    return {
        "id": bucket_id,
        "metadata": {
            "name": "cold evidence",
            "type": metadata_type,
            "world": "",
        },
        "content": "THIS COLD CONTENT MUST NOT SURFACE",
        "path": str(archive_dir / "生活" / f"{bucket_id}.md"),
    }


@pytest.mark.asyncio
async def test_stale_archive_embedding_cannot_reenter_breath(tmp_path, monkeypatch):
    archive_dir = tmp_path / "archive"
    cold = _cold_bucket("cold-1", archive_dir)
    manager = _ColdBucketManager(cold, archive_dir)

    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "buckets_dir": str(tmp_path),
            "random_surfacing": {},
            "query_expansion": {"enabled": False},
        },
    )
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", _ColdEmbedding(cold["id"]))
    monkeypatch.setattr(server, "dehydrator", _NeverDehydrate())
    monkeypatch.setattr(server, "decay_engine", _Decay())
    monkeypatch.setattr(server, "consolidation_engine", _NoopLoop())
    monkeypatch.setattr(server, "episode_engine", _NoopLoop())
    monkeypatch.setattr(server, "_backfill_started", True)

    result = await server.breath(
        query="cold evidence",
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )

    assert result == "未找到相关记忆。"
    assert manager.touched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_type", ["dynamic", "archived"])
async def test_anchor_status_rejects_cold_keyword_and_vector_evidence(
    tmp_path,
    monkeypatch,
    metadata_type,
):
    archive_dir = tmp_path / "archive"
    cold = _cold_bucket("cold-2", archive_dir, metadata_type=metadata_type)
    manager = _ColdBucketManager(cold, archive_dir)

    async def search(*args, **kwargs):
        return [cold]

    manager.search = search
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "buckets_dir": str(tmp_path),
            "query_expansion": {"enabled": False},
        },
    )
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", _ColdEmbedding(cold["id"]))
    monkeypatch.setattr(server, "_append_recall_status_trace", lambda record: None)

    result = await server._probe_anchor_status("cold evidence")

    assert result["keyword_candidate_count"] == 0
    assert result["vector_candidate_count"] == 1
    assert result["final_candidate_count"] == 0
    assert result["has_evidence"] is False


def test_association_role_is_explicit_even_when_role_markers_are_disabled(monkeypatch):
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "recall_evidence_roles": {"enabled": False},
        },
    )

    association = server._recall_prefix(
        "neighbor",
        "association",
        "y_relation",
        relation="kin:out:d1<-main",
    )
    main = server._recall_prefix("main", "main", "curated_rrf")

    assert "[role:association]" in association
    assert "[authority:supporting_only]" in association
    assert "[layer:y_relation]" in association
    assert "[relation:kin:out:d1<-main]" in association
    assert "[role:main]" not in main
