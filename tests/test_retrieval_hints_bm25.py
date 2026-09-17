from copy import deepcopy

import jieba
import pytest

from bm25_index import BM25Index, _tokenize
from retrieval_hints import source_record
from retrieval_tokenizer import RetrievalTokenizer


def documents():
    return [{"id": "a", "content": "project comet retained the old index", "metadata": {}},
            {"id": "b", "content": "bananas apples oranges", "metadata": {}},
            {"id": "c", "content": "trains cars boats", "metadata": {}}]


def augmented():
    docs = documents()
    hints = {"schema_version": 1, "who": [], "where": [], "when": [], "what": [],
             "aliases": [{"text": "comet legacy-search choice", "evidence": docs[0]["content"]}]}
    docs[0]["retrieval_hints_v1"] = {**source_record(docs[0]), "payload": hints}
    return docs


def test_off_ignores_hint_fields_and_preserves_exact_scores():
    left, right = BM25Index(), BM25Index()
    left.build(documents())
    right.build(augmented())
    for query in ("legacy-search", "comet", "apples", "index"):
        assert left.score(query) == right.score(query)
        assert left.rare_term_hits(query, max_df=5) == right.rare_term_hits(query, max_df=5)


def test_hints_help_candidate_but_never_become_literal_evidence():
    index = BM25Index(hints_enabled=True)
    index.build(augmented())
    assert "a" in index.score("legacy-search")
    assert "a" in index.hint_matches("legacy-search")
    assert index.rare_term_hits("legacy-search", max_df=5) == {}
    assert index.literal_term_df_hits("legacy-search", bucket_ids={"a"}) == {}
    assert "a" in index.rare_term_hits("comet", max_df=5)


def test_cow_update_removes_stale_hints_and_keeps_old_generation():
    index = BM25Index(hints_enabled=True)
    index.build(augmented())
    changed = {**augmented()[0], "content": "new source with different content"}
    fresh = index.with_upsert(changed)
    assert fresh.score("legacy-search") == {}
    assert "a" in index.score("legacy-search")
    assert fresh.hint_matches("legacy-search") == {}
    removed = index.with_delete("a")
    assert removed.hint_matches("legacy-search") == {}
    assert removed.rare_term_hits("comet", max_df=5) == {}
    added = index.with_upsert({"id": "new", "content": "unicorn orchard", "metadata": {}})
    assert "new" in added.rare_term_hits("unicorn", max_df=5)
    assert "new" not in index.rare_term_hits("unicorn", max_df=5)


def test_private_dictionary_does_not_mutate_global_or_old_generation():
    text = "星帆云栈在银杏馆"
    before = _tokenize(text)
    private = RetrievalTokenizer("星帆云栈 100000 nz\n".encode())
    assert "星帆云栈" in private(text)
    assert _tokenize(text) == before
    index = BM25Index(tokenizer=private)
    index.build([{"id": "a", "content": text, "metadata": {}}, *documents()])
    assert index.lexical_version == private.version
    assert index.with_upsert(documents()[0]).lexical_version == private.version
    assert index._tokenizer is private
    assert BM25Index().lexical_version == "jieba-search-v1"


def test_private_dictionary_uses_same_case_normalization_as_queries():
    private = RetrievalTokenizer(b"OrionRouter 100000 nz\nCedar-Bridge 100000 nz\n")
    assert private._tokenizer.FREQ.get("orionrouter") == 100000
    assert private._tokenizer.FREQ.get("cedar-bridge") == 100000
    assert "orionrouter" in private("ORIONROUTER")
    assert "cedar-bridge" in private("Cedar-Bridge")
    assert "private-lower-v2" in private.version
    with pytest.raises(ValueError, match="duplicate"):
        RetrievalTokenizer(b"OrionRouter 100 nz\norionrouter 100 nz\n")
