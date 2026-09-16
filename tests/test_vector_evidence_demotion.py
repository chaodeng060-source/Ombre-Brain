import pytest

import server


def _pairs(*items):
    return [(bid, score) for bid, score in items]


def test_default_off_returns_baseline_unchanged(monkeypatch):
    monkeypatch.delenv("OMBRE_VECTOR_EVIDENCE_DEMOTION", raising=False)
    fused = _pairs(("lexical", 0.03), ("vec", 0.02))

    assert server._demote_evidenceless_candidates(
        fused, _pairs(("vec", 0.68)), []
    ) == fused


def test_evidenceless_leader_loses_the_top_slot(monkeypatch):
    monkeypatch.setenv("OMBRE_VECTOR_EVIDENCE_DEMOTION", "1")
    fused = _pairs(("lexical", 0.033), ("vec_a", 0.026), ("vec_b", 0.016))

    result = server._demote_evidenceless_candidates(
        fused, _pairs(("vec_a", 0.68), ("vec_b", 0.66)), []
    )

    assert [bid for bid, _ in result] == ["vec_a", "vec_b", "lexical"]
    assert dict(result)["lexical"] == 0.033


def test_entity_hit_counts_as_evidence(monkeypatch):
    monkeypatch.setenv("OMBRE_VECTOR_EVIDENCE_DEMOTION", "1")
    fused = _pairs(("entity_only", 0.03), ("vec", 0.02), ("lexical", 0.01))

    result = server._demote_evidenceless_candidates(
        fused, _pairs(("vec", 0.68)), _pairs(("entity_only", 1.0))
    )

    assert [bid for bid, _ in result] == ["entity_only", "vec", "lexical"]


def test_order_within_each_group_is_preserved(monkeypatch):
    monkeypatch.setenv("OMBRE_VECTOR_EVIDENCE_DEMOTION", "1")
    fused = _pairs(
        ("lex_1", 0.05), ("vec_1", 0.04), ("lex_2", 0.03), ("vec_2", 0.02)
    )

    result = server._demote_evidenceless_candidates(
        fused, _pairs(("vec_1", 0.7), ("vec_2", 0.6)), []
    )

    assert [bid for bid, _ in result] == ["vec_1", "vec_2", "lex_1", "lex_2"]


def test_nothing_is_dropped(monkeypatch):
    monkeypatch.setenv("OMBRE_VECTOR_EVIDENCE_DEMOTION", "1")
    fused = _pairs(("a", 0.05), ("b", 0.04), ("c", 0.03))

    result = server._demote_evidenceless_candidates(fused, _pairs(("b", 0.6)), [])

    assert sorted(result) == sorted(fused)


@pytest.mark.parametrize(
    "vector_ranked, entity_ranked",
    [([], []), ([], None)],
)
def test_no_semantic_retrieval_this_turn_is_a_noop(
    monkeypatch, vector_ranked, entity_ranked
):
    monkeypatch.setenv("OMBRE_VECTOR_EVIDENCE_DEMOTION", "1")
    fused = _pairs(("a", 0.05), ("b", 0.04))

    assert server._demote_evidenceless_candidates(
        fused, vector_ranked, entity_ranked
    ) == fused


def test_all_candidates_evidenced_is_a_noop(monkeypatch):
    monkeypatch.setenv("OMBRE_VECTOR_EVIDENCE_DEMOTION", "1")
    fused = _pairs(("a", 0.05), ("b", 0.04))

    assert server._demote_evidenceless_candidates(
        fused, _pairs(("a", 0.7), ("b", 0.6)), []
    ) == fused


def test_empty_fused_list_is_safe(monkeypatch):
    monkeypatch.setenv("OMBRE_VECTOR_EVIDENCE_DEMOTION", "1")

    assert server._demote_evidenceless_candidates([], _pairs(("a", 0.7)), []) == []
