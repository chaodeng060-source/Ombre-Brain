"""字面单路压制（2026-08-31 朝灯令「没有相关内容就不该召回」）。

背景：实测 330 条真实候选里 44.8% 是 vector 检索没捞到、纯靠字面撞词进来的，
其中 27 条字面吃满 100 直接拿 0.45 满分，跟真正相关的并列坐在最高分档，
导致 conversation 线 0.25 形同虚设（14 轮真实日志里 11 轮 kept == input）。

这些用例锁死：只压「向量没命中时的字面单路」，向量有命中的一律不受影响。
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

server = importlib.import_module("server")

CONVERSATION_THRESHOLD = 0.25


def score(**bucket):
    return server._anchor_adapted_relevance_score(dict(bucket))


def test_pure_literal_hit_falls_below_conversation_threshold():
    """纯撞词（向量没捞到 + 字面满分）必须落到 0.25 线之下。"""
    s = score(_literal_relevance_score=100.0, _original_vector_relevance_score=0.0)
    assert s == pytest.approx(0.2475)
    assert s < CONVERSATION_THRESHOLD


def test_vector_hit_keeps_full_literal_weight():
    """向量有命中时，字面分不受任何压制——回归保护。"""
    s = score(_literal_relevance_score=100.0, _original_vector_relevance_score=0.3)
    assert s == pytest.approx(0.45)
    assert s >= CONVERSATION_THRESHOLD


def test_vector_alone_unchanged():
    """只有向量证据时，行为跟改动前完全一致。"""
    assert score(_original_vector_relevance_score=0.68) == pytest.approx(0.306)


def test_entity_match_still_full_score():
    """实体命中仍按原契约给满分，本次不动它。"""
    s = score(_literal_relevance_score=100.0, _original_vector_relevance_score=0.0, entity_match=True)
    assert s == pytest.approx(0.45)


def test_missing_all_evidence_returns_none():
    """没有任何绝对证据时仍返回 None（上游 fail-open 契约不变）。"""
    assert score(score=123) is None


def test_cap_env_can_fully_roll_back(monkeypatch):
    """把上限设回 1 即完全退回改动前的行为。"""
    monkeypatch.setenv("OMBRE_ANCHOR_LITERAL_ONLY_CAP", "1")
    s = score(_literal_relevance_score=100.0, _original_vector_relevance_score=0.0)
    assert s == pytest.approx(0.45)


def test_cap_env_off_keyword_rolls_back(monkeypatch):
    monkeypatch.setenv("OMBRE_ANCHOR_LITERAL_ONLY_CAP", "off")
    s = score(_literal_relevance_score=100.0, _original_vector_relevance_score=0.0)
    assert s == pytest.approx(0.45)


def test_cap_env_garbage_falls_back_to_default(monkeypatch):
    """坏值不许炸链路，回落默认 0.55。"""
    monkeypatch.setenv("OMBRE_ANCHOR_LITERAL_ONLY_CAP", "abc")
    s = score(_literal_relevance_score=100.0, _original_vector_relevance_score=0.0)
    assert s == pytest.approx(0.2475)


def test_low_literal_without_vector_unchanged():
    """字面分本来就低于上限的，压制不改变它。"""
    s = score(_literal_relevance_score=20.0, _original_vector_relevance_score=0.0)
    assert s == pytest.approx(0.09)


def test_negative_vector_treated_as_no_hit():
    """负向量分不算命中，字面仍受压制。"""
    s = score(_literal_relevance_score=100.0, _original_vector_relevance_score=-0.5)
    assert s == pytest.approx(0.2475)
