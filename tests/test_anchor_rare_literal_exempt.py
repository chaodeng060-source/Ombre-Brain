"""稀有词精确命中豁免字面单路压制（2026-09-04 朝灯考「蚊子」现行）。

背景：8/31 把「向量没命中的字面单路」压到 0.2475（<0.25 线）拦常见词撞车；
副作用是整库只出现几次的词被整词命中、而她整句向量掉到 0.5 地板下时，
候选照样被按死——「蚊子」5 条桶字面满分全被闸掉，记了搜不到。

这些用例锁死：只有候选带着 _rare_literal_terms（召回路径按 BM25 df 标注）
才免压；常见词撞车、向量有命中、env 关闭三条路径与 8/31 完全一致。
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

server = importlib.import_module("server")
bm25_index = importlib.import_module("bm25_index")

CONVERSATION_THRESHOLD = 0.25


def score(**bucket):
    return server._anchor_adapted_relevance_score(dict(bucket))


def test_rare_literal_hit_escapes_literal_only_cap():
    s = score(
        _literal_relevance_score=100.0,
        _original_vector_relevance_score=0.0,
        _rare_literal_terms=["蚊子"],
    )
    assert s == pytest.approx(0.45)
    assert s >= CONVERSATION_THRESHOLD


def test_common_word_collision_still_capped():
    """没有稀有词标注的字面单路，8/31 行为一字不改。"""
    s = score(_literal_relevance_score=100.0, _original_vector_relevance_score=0.0)
    assert s == pytest.approx(0.2475)
    assert s < CONVERSATION_THRESHOLD


def test_empty_rare_terms_is_not_an_exemption():
    s = score(
        _literal_relevance_score=100.0,
        _original_vector_relevance_score=0.0,
        _rare_literal_terms=[],
    )
    assert s == pytest.approx(0.2475)


def test_exemption_does_not_force_low_literal_over_the_line():
    """豁免只是不压，字面分本身不够 55.6 仍过不了 0.25。"""
    s = score(
        _literal_relevance_score=50.0,
        _original_vector_relevance_score=0.0,
        _rare_literal_terms=["蚊子"],
    )
    assert s == pytest.approx(0.225)
    assert s < CONVERSATION_THRESHOLD


def test_vector_hit_path_unchanged_with_rare_terms():
    s = score(
        _literal_relevance_score=100.0,
        _original_vector_relevance_score=0.3,
        _rare_literal_terms=["蚊子"],
    )
    assert s == pytest.approx(0.45)


def test_max_df_env_parsing(monkeypatch):
    monkeypatch.delenv("OMBRE_ANCHOR_RARE_LITERAL_MAX_DF", raising=False)
    assert server._anchor_rare_literal_max_df() == 8
    monkeypatch.setenv("OMBRE_ANCHOR_RARE_LITERAL_MAX_DF", "0")
    assert server._anchor_rare_literal_max_df() == 0
    monkeypatch.setenv("OMBRE_ANCHOR_RARE_LITERAL_MAX_DF", "off")
    assert server._anchor_rare_literal_max_df() == 0
    monkeypatch.setenv("OMBRE_ANCHOR_RARE_LITERAL_MAX_DF", "abc")
    assert server._anchor_rare_literal_max_df() == 8
    monkeypatch.setenv("OMBRE_ANCHOR_RARE_LITERAL_MAX_DF", "20")
    assert server._anchor_rare_literal_max_df() == 20


def _build_index():
    pytest.importorskip("rank_bm25")
    idx = bm25_index.BM25Index()
    buckets = [
        {"id": "m1", "metadata": {"name": "蚊子夜"}, "content": "主要是有蚊子 打死它 早点睡"},
        {"id": "m2", "metadata": {"name": "蚊子反思"}, "content": "四月的蚊子没存过 记忆未入库"},
    ]
    # 「记忆」是常见词：塞满语料让它的 df 远超上限
    for i in range(30):
        buckets.append({"id": f"c{i}", "metadata": {"name": f"记忆{i}"}, "content": "记忆 召回 噪音 看看"})
    idx.build(buckets)
    return idx


def test_bm25_rare_term_hits_only_rare_terms():
    idx = _build_index()
    hits = idx.rare_term_hits("我现在问你蚊子 记忆", max_df=8)
    assert set(hits) == {"m1", "m2"}
    assert hits["m1"] == ("蚊子",)
    assert "记忆" not in {t for terms in hits.values() for t in terms}


def test_bm25_rare_term_hits_respects_threshold_and_off():
    idx = _build_index()
    assert idx.rare_term_hits("蚊子", max_df=1) == {}
    assert idx.rare_term_hits("蚊子", max_df=0) == {}
    assert idx.rare_term_hits("", max_df=8) == {}


def test_bm25_rare_term_hits_ignores_single_char_terms():
    idx = _build_index()
    # 「蚊」单字在语料里可能碰巧稀有，单字不算证据
    assert idx.rare_term_hits("蚊", max_df=8) == {}


def test_bucket_manager_lookup_fails_closed(monkeypatch):
    class _Mgr:
        _bm25_mode = "live"
        _bm25 = None

    mgr = _Mgr()
    fn = importlib.import_module("bucket_manager").BucketManager.rare_literal_hits
    assert fn(mgr, "蚊子", max_df=8) == {}

    class _Boom:
        def rare_term_hits(self, *a, **k):
            raise RuntimeError("boom")

    mgr._bm25 = _Boom()
    assert fn(mgr, "蚊子", max_df=8) == {}
    mgr._bm25 = _build_index()
    assert set(fn(mgr, "蚊子", max_df=8)) == {"m1", "m2"}
    mgr._bm25_mode = "off"
    assert fn(mgr, "蚊子", max_df=8) == {}
