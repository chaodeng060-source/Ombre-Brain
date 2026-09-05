"""情绪一致召回·直白版（2026-09-05 朝灯：「我开心的时候我肯定就会想到我之前自己开心的事情」）。

三条合同：
1. OMBRE_MOOD_CONGRUENT_WEIGHT=0（默认）时排序乘数恒为 1.0，旧行为逐字等价；
2. 开权重后，情绪相近（emotion_score→1）抬 1+w，相反（→0）压 1-w，0.5 不动，权重封顶 0.5；
3. 主对话 AI 自己的心情 self_* 与她这句话的情绪合并：都有取平均、只有一方用那一方、都没有为 None。
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

server = importlib.import_module("server")
bucket_manager = importlib.import_module("bucket_manager")
BM = bucket_manager.BucketManager


def test_weight_zero_is_identity(monkeypatch):
    monkeypatch.delenv("OMBRE_MOOD_CONGRUENT_WEIGHT", raising=False)
    assert BM._mood_congruent_weight() == 0.0
    for score in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert BM._mood_congruent_factor(score, BM._mood_congruent_weight()) == 1.0


def test_factor_lifts_congruent_and_lowers_opposite():
    assert BM._mood_congruent_factor(1.0, 0.15) == pytest.approx(1.15)
    assert BM._mood_congruent_factor(0.0, 0.15) == pytest.approx(0.85)
    assert BM._mood_congruent_factor(0.5, 0.15) == pytest.approx(1.0)
    assert BM._mood_congruent_factor(0.75, 0.2) == pytest.approx(1.1)


def test_weight_env_parsing_and_cap(monkeypatch):
    monkeypatch.setenv("OMBRE_MOOD_CONGRUENT_WEIGHT", "0.15")
    assert BM._mood_congruent_weight() == pytest.approx(0.15)
    monkeypatch.setenv("OMBRE_MOOD_CONGRUENT_WEIGHT", "9")
    assert BM._mood_congruent_weight() == 0.5
    monkeypatch.setenv("OMBRE_MOOD_CONGRUENT_WEIGHT", "-1")
    assert BM._mood_congruent_weight() == 0.0
    monkeypatch.setenv("OMBRE_MOOD_CONGRUENT_WEIGHT", "abc")
    assert BM._mood_congruent_weight() == 0.0


def test_factor_tolerates_bad_input():
    assert BM._mood_congruent_factor("x", 0.2) == 1.0
    assert BM._mood_congruent_factor(3.0, 0.2) == pytest.approx(1.2)  # clamp 到 +1


def test_merge_mood_coords_contract():
    merge = server._merge_mood_coords
    assert merge(None, None, -1, -1) == (None, None)
    assert merge(0.9, 0.7, -1, -1) == (0.9, 0.7)
    assert merge(None, None, 0.6, 0.5) == (0.6, 0.5)
    assert merge(0.2, 0.8, 0.6, 0.4) == (pytest.approx(0.4), pytest.approx(0.6))
    # self 越界视为未传
    assert merge(0.2, 0.8, 1.5, 0.4) == (0.2, 0.8)
    assert merge(None, None, 0.6, 2.0) == (None, None)


def test_breath_accepts_self_mood_kwargs():
    import inspect
    params = inspect.signature(server.breath).parameters
    assert params["self_valence"].default == -1
    assert params["self_arousal"].default == -1
