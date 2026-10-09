import pytest

from bm25_index import BM25Index


def _bucket(bucket_id, name, content):
    return {
        "id": bucket_id,
        "metadata": {"name": name},
        "content": content,
    }


def _assert_matches_full_rebuild(index, buckets, query):
    rebuilt = BM25Index()
    rebuilt.build(buckets)

    assert index._ids == rebuilt._ids
    assert index._postings == rebuilt._postings
    assert index._term_doc_counts == rebuilt._term_doc_counts
    assert index._df_histogram == rebuilt._df_histogram
    assert index._total_doc_length == rebuilt._total_doc_length
    assert index.rare_term_hits(query, max_df=2) == rebuilt.rare_term_hits(
        query, max_df=2
    )
    targets = {bucket["id"] for bucket in buckets}
    assert index.literal_term_df_hits(query, bucket_ids=targets) == (
        rebuilt.literal_term_df_hits(query, bucket_ids=targets)
    )
    assert index.score(query) == pytest.approx(rebuilt.score(query))


def test_copy_on_write_upsert_and_delete_match_full_rebuild():
    pytest.importorskip("rank_bm25")
    initial = [
        _bucket("a", "蚊子夜", "蚊子 睡觉 记录"),
        _bucket("b", "工作计划", "计划 验收 进度"),
        _bucket("c", "蚊子回忆", "蚊子 夏天 记忆"),
    ]
    original = BM25Index()
    original.build(initial)
    original_ids = list(original._ids)

    replacement = _bucket("b", "新计划", "新计划 复盘 稳定")
    after_upsert = original.with_upsert(replacement)
    after_upsert_buckets = [initial[0], replacement, initial[2]]
    _assert_matches_full_rebuild(after_upsert, after_upsert_buckets, "蚊子 新计划")

    # Building the new generation must leave the resident generation intact.
    assert original._ids == original_ids
    assert original.rare_term_hits("蚊子", max_df=2) == {
        "a": ("蚊子",), "c": ("蚊子",),
    }

    added = _bucket("d", "加班记录", "加班 夜里 记录")
    after_add = after_upsert.with_upsert(added)
    after_add_buckets = after_upsert_buckets + [added]
    _assert_matches_full_rebuild(after_add, after_add_buckets, "夜里 加班 蚊子")

    after_delete = after_add.with_delete("a")
    after_delete_buckets = [replacement, initial[2], added]
    _assert_matches_full_rebuild(after_delete, after_delete_buckets, "夜里 加班 蚊子")
    assert after_add._ids == ["a", "b", "c", "d"]
