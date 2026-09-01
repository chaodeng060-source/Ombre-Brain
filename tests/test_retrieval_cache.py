import json
import os
import sqlite3
from pathlib import Path

import frontmatter
import pytest

from embedding_engine import EmbeddingEngine
from maintenance_barrier import MaintenanceBarrier


@pytest.mark.asyncio
async def test_list_all_reuses_parsed_snapshot_without_sharing_mutations(
    bucket_mgr,
    monkeypatch,
):
    first_id = await bucket_mgr.create("第一条正文", name="第一条", domain=["测试"])
    second_id = await bucket_mgr.create("第二条正文", name="第二条", domain=["测试"])
    original_load = bucket_mgr._load_bucket
    loaded = []

    def counting_load(path):
        loaded.append(path)
        return original_load(path)

    monkeypatch.setattr(bucket_mgr, "_load_bucket", counting_load)

    first = await bucket_mgr.list_all(include_archive=False)
    cold_loads = len(loaded)
    assert {row["id"] for row in first} == {first_id, second_id}
    assert cold_loads == 2

    first[0]["content"] = "调用方污染"
    first[0]["metadata"]["name"] = "调用方污染"
    first[0]["score"] = 100
    second = await bucket_mgr.list_all(include_archive=False)

    assert len(loaded) == cold_loads
    assert all(row["content"] != "调用方污染" for row in second)
    assert all(row["metadata"].get("name") != "调用方污染" for row in second)
    assert all("score" not in row for row in second)


@pytest.mark.asyncio
async def test_list_all_snapshot_detects_external_edit_create_and_delete(bucket_mgr):
    bucket_id = await bucket_mgr.create("旧正文", name="外部编辑", domain=["测试"])
    warm = await bucket_mgr.list_all(include_archive=False)
    path = Path(next(row["path"] for row in warm if row["id"] == bucket_id))

    post = frontmatter.load(path)
    post.content = "外部改后的正文"
    frontmatter.dump(post, path)
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))

    edited = await bucket_mgr.list_all(include_archive=False)
    assert next(row for row in edited if row["id"] == bucket_id)["content"] == "外部改后的正文"

    external_path = path.parent / "external-cache-test.md"
    frontmatter.dump(
        frontmatter.Post(
            "外部新建正文",
            id="external-cache-test",
            name="外部新建",
            type="dynamic",
            domain=["测试"],
        ),
        external_path,
    )
    assert "external-cache-test" in {
        row["id"] for row in await bucket_mgr.list_all(include_archive=False)
    }

    external_path.unlink()
    assert "external-cache-test" not in {
        row["id"] for row in await bucket_mgr.list_all(include_archive=False)
    }


@pytest.mark.asyncio
async def test_bucket_mutations_are_immediately_visible_to_warm_cache(bucket_mgr):
    await bucket_mgr.list_all(include_archive=False)
    bucket_id = await bucket_mgr.create("即时可见旧正文", name="缓存失效", domain=["测试"])
    assert bucket_id in {
        row["id"] for row in await bucket_mgr.list_all(include_archive=False)
    }

    await bucket_mgr.list_all(include_archive=False)
    assert await bucket_mgr.update(bucket_id, content="即时可见新正文") is True
    updated = await bucket_mgr.list_all(include_archive=False)
    assert next(row for row in updated if row["id"] == bucket_id)["content"] == "即时可见新正文"

    await bucket_mgr.list_all(include_archive=False)
    assert await bucket_mgr.archive(bucket_id) is True
    assert bucket_id not in {
        row["id"] for row in await bucket_mgr.list_all(include_archive=False)
    }
    assert bucket_id in {
        row["id"] for row in await bucket_mgr.list_all(include_archive=True)
    }

    await bucket_mgr.list_all(include_archive=True)
    assert await bucket_mgr.delete(bucket_id) is True
    assert bucket_id not in {
        row["id"] for row in await bucket_mgr.list_all(include_archive=True)
    }


@pytest.mark.asyncio
async def test_preloaded_search_is_result_equivalent_and_does_not_mutate_candidates(
    bucket_mgr,
):
    await bucket_mgr.create("朝灯喜欢潮汕菜", name="潮汕菜", domain=["生活"])
    await bucket_mgr.create("今晚吃了别的菜", name="晚饭", domain=["生活"])

    expected = await bucket_mgr.search(
        "潮汕菜",
        limit=10,
        relevance_first=True,
        relevance_candidate_floor=0.0,
    )
    candidates = await bucket_mgr.list_all(include_archive=False)
    actual = await bucket_mgr.search(
        "潮汕菜",
        limit=10,
        relevance_first=True,
        relevance_candidate_floor=0.0,
        preloaded_buckets=candidates,
    )

    assert actual == expected
    assert all("score" not in row for row in candidates)
    assert all("_keyword_tie_break_score" not in row for row in candidates)


def _embedding_engine_with_rows(tmp_path, rows):
    engine = object.__new__(EmbeddingEngine)
    engine.enabled = True
    engine.db_path = str(tmp_path / "embeddings.db")
    engine._maintenance_barrier = MaintenanceBarrier(str(tmp_path))
    with sqlite3.connect(engine.db_path) as connection:
        connection.execute(
            "CREATE TABLE embeddings ("
            "bucket_id TEXT PRIMARY KEY, embedding TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO embeddings(bucket_id, embedding, updated_at) VALUES (?, ?, 'now')",
            rows,
        )
    return engine


def test_prepared_similarity_preserves_python_cosine_result_exactly():
    engine = object.__new__(EmbeddingEngine)
    cases = [
        ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
        ([0.0, 0.0], [1.0, 0.0]),
        ([[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [-1.0, 0.0]]),
    ]
    for left, right in cases:
        expected = engine._max_stored_similarity(left, right)
        actual = engine._max_prepared_similarity(
            engine._prepare_embedding_record(left),
            engine._prepare_embedding_record(right),
        )
        assert actual == expected


@pytest.mark.asyncio
async def test_vector_search_cooperative_yield_preserves_results(
    tmp_path,
    monkeypatch,
):
    engine = _embedding_engine_with_rows(
        tmp_path,
        [
            ("bucket-a", json.dumps([1.0, 0.0])),
            ("bucket-b", json.dumps([0.0, 1.0])),
        ],
    )

    async def fake_embedding(_query):
        return [1.0, 0.0], "ok"

    monkeypatch.setattr(engine, "_generate_embedding_with_status", fake_embedding)
    expected = await engine.search_similar("query", top_k=10)
    actual = await engine.search_similar(
        "query",
        top_k=10,
        cooperative_yield_every=1,
    )

    assert actual == expected
    with pytest.raises(
        ValueError,
        match="^cooperative_yield_every must be a positive integer$",
    ):
        await engine.search_similar(
            "query",
            cooperative_yield_every=0,
        )


@pytest.mark.asyncio
async def test_selected_vector_scores_reuse_one_query_embedding(
    tmp_path,
    monkeypatch,
):
    engine = _embedding_engine_with_rows(
        tmp_path,
        [
            ("bucket-a", json.dumps([1.0, 0.0])),
            ("bucket-b", json.dumps([0.0, 1.0])),
            ("bucket-c", json.dumps([-1.0, 0.0])),
        ],
    )
    calls = 0

    async def fake_embedding(_query):
        nonlocal calls
        calls += 1
        return [1.0, 0.0], "ok"

    monkeypatch.setattr(engine, "_generate_embedding_with_status", fake_embedding)
    ranked, status, selected = await engine.search_similar_with_selected_scores(
        "query",
        top_k=1,
        score_bucket_ids={"bucket-a", "bucket-b"},
    )

    assert calls == 1
    assert status == "ok"
    assert ranked == [("bucket-a", 1.0)]
    assert selected == {"bucket-a": 1.0, "bucket-b": 0.0}


@pytest.mark.asyncio
async def test_vector_cache_reuses_parse_and_refreshes_after_external_commit(
    tmp_path,
    monkeypatch,
):
    engine = _embedding_engine_with_rows(
        tmp_path,
        [
            ("bucket-a", json.dumps([1.0, 0.0])),
            ("bucket-b", json.dumps([0.0, 1.0])),
        ],
    )

    async def fake_embedding(_query):
        return [1.0, 0.0], "ok"

    monkeypatch.setattr(engine, "_generate_embedding_with_status", fake_embedding)
    import embedding_engine as embedding_module

    original_loads = embedding_module.json.loads
    parsed = 0

    def counting_loads(payload):
        nonlocal parsed
        parsed += 1
        return original_loads(payload)

    monkeypatch.setattr(embedding_module.json, "loads", counting_loads)

    first, first_status = await engine.search_similar_with_status("query", top_k=10)
    cold_parses = parsed
    second, second_status = await engine.search_similar_with_status("query", top_k=10)
    assert first_status == second_status == "ok"
    assert second == first
    assert cold_parses == 2
    assert parsed == cold_parses

    with sqlite3.connect(engine.db_path) as connection:
        connection.execute(
            "INSERT INTO embeddings(bucket_id, embedding, updated_at) VALUES (?, ?, 'later')",
            ("bucket-c", json.dumps([0.9, 0.1])),
        )

    refreshed, status = await engine.search_similar_with_status("query", top_k=10)
    assert status == "ok"
    assert [bucket_id for bucket_id, _score in refreshed] == [
        "bucket-a",
        "bucket-c",
        "bucket-b",
    ]
    assert parsed == cold_parses + 1


@pytest.mark.asyncio
async def test_vector_store_and_delete_invalidate_warm_cache(tmp_path, monkeypatch):
    engine = _embedding_engine_with_rows(
        tmp_path,
        [("bucket-a", json.dumps([1.0, 0.0]))],
    )

    async def fake_embedding(_query):
        return [1.0, 0.0], "ok"

    monkeypatch.setattr(engine, "_generate_embedding_with_status", fake_embedding)
    import embedding_engine as embedding_module

    original_loads = embedding_module.json.loads
    parsed = 0

    def counting_loads(payload):
        nonlocal parsed
        parsed += 1
        return original_loads(payload)

    monkeypatch.setattr(embedding_module.json, "loads", counting_loads)
    await engine.search_similar_with_status("query", top_k=10)
    cold_parses = parsed

    engine._store_embedding("bucket-b", [0.8, 0.2])
    stored, _status = await engine.search_similar_with_status("query", top_k=10)
    assert [bucket_id for bucket_id, _score in stored] == ["bucket-a", "bucket-b"]
    assert parsed == cold_parses + 1

    engine.delete_embedding("bucket-a")
    remaining, _status = await engine.search_similar_with_status("query", top_k=10)
    assert [bucket_id for bucket_id, _score in remaining] == ["bucket-b"]
    assert parsed == cold_parses + 1
