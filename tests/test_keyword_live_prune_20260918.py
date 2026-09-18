"""2026-09-18 第十四版：BM25 live 模式关键词剪枝 `_bounded_topic_scores_live`。
验证：开关关时不调用；开关开时 search 返回的 id 序列、score、tie_break 与全量逐位一致
（随机语料 × 多句 × limit 5/20 × 含 resolved / 钥匙命中 / mood 权重）；且主循环确实只跑保留桶。
"""
from __future__ import annotations

import asyncio
import importlib
import os
import random
import sys
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parent
sys.path.insert(0, str(TREE))
bucket_manager = importlib.import_module("bucket_manager")
BucketManager = bucket_manager.BucketManager


async def _inline_to_thread(function, *args, **kwargs):
    return function(*args, **kwargs)


@pytest.fixture(autouse=True)
def _inline(monkeypatch):
    monkeypatch.setattr(bucket_manager.asyncio, "to_thread", _inline_to_thread)


class _FakeBM25:
    def __init__(self, rows, scores):
        self._index = object()
        self._keyword_score_rows = rows
        self._scores = scores

    def score(self, _query):
        return dict(self._scores)


def _manager(tmp_path) -> BucketManager:
    buckets_dir = tmp_path / "buckets"
    for sub in ("permanent", "dynamic", "archive", "dynamic/feel"):
        (buckets_dir / sub).mkdir(parents=True, exist_ok=True)
    return BucketManager({
        "buckets_dir": str(buckets_dir),
        "audit": {"enabled": False},
        "matching": {"fuzzy_threshold": 50, "max_results": 10},
        "wikilink": {"enabled": False},
        "scoring_weights": {"topic_relevance": 4.0, "emotion_resonance": 2.0,
                            "time_proximity": 2.5, "importance": 1.0, "content_weight": 3.0},
    })


_WORDS = ["粽子", "边牧", "许嵩", "卡兜", "肠粉", "中秋", "游戏", "遛狗", "公园", "蚊子", "宅舞",
          "海马体", "召回", "噪音", "重启", "回家", "老公", "广州", "深圳", "阴阳师", "台式", "显卡"]


def _corpus(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        words = rng.sample(_WORDS, rng.randint(1, 4))
        name = "".join(words) + f"_{i}"
        tags = rng.sample(_WORDS, rng.randint(0, 3))
        content = "。".join(rng.choice(_WORDS) + rng.choice(["很好", "去了", "吃了", "没来"]) for _ in range(rng.randint(3, 30)))
        keys = [rng.choice(_WORDS) + "钥匙"] if rng.random() < 0.05 else []
        out.append({
            "id": f"b{i:04d}",
            "content": content,
            "metadata": {
                "id": f"b{i:04d}", "name": name, "tags": tags,
                "domain": [rng.choice(["生活", "工程", "感情"])],
                "retrieval_keys": keys, "importance": rng.randint(1, 10),
                "resolved": rng.random() < 0.1,
                "valence": rng.random(), "arousal": rng.random(),
            },
        })
    return out


def _install(manager, buckets, rng):
    rows = {b["id"]: manager._build_keyword_score_row(b) for b in buckets}
    scores = {b["id"]: rng.random() for b in buckets if rng.random() < 0.6}
    manager._bm25_mode = "live"
    manager._bm25 = _FakeBM25(rows, scores)
    manager._bm25_dirty = False
    manager._bm25_rebuilding = False
    manager._bm25_unknown_dirty = False
    manager._bm25_dirty_bucket_ids.clear()


QUERIES = ["要中秋了诶，但是我想吃粽子", "宝宝边牧呢", "许嵩要结婚了", "我在打游戏", "有召回吗",
           "卡兜今天遛了吗", "肠粉", "蚊子钥匙", "老公你还回家吗", "显卡台式到了"]


def _run(manager, buckets, query, limit, *, prune: bool, mood=False):
    os.environ["OMBRE_KEYWORD_LIVE_PRUNE"] = "1" if prune else "0"
    kwargs = {}
    if mood:
        kwargs = {"query_valence": 0.8, "query_arousal": 0.6}
    return asyncio.run(manager.search(
        query, limit=limit, relevance_first=True, relevance_candidate_floor=0.0,
        preloaded_buckets=buckets, **kwargs,
    ))


def _sig(result):
    return [(b["id"], b["score"], b["_keyword_tie_break_score"], b["_bm25_relevance_score"]) for b in result]


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("limit", [5, 20])
def test_prune_matches_full_scan(tmp_path, monkeypatch, seed, limit):
    rng = random.Random(seed)
    manager = _manager(tmp_path)
    buckets = _corpus(rng, 400)
    _install(manager, buckets, rng)
    for query in QUERIES:
        full = _sig(_run(manager, buckets, query, limit, prune=False))
        pruned = _sig(_run(manager, buckets, query, limit, prune=True))
        assert pruned == full, query


def test_prune_matches_with_mood_weight(tmp_path, monkeypatch):
    monkeypatch.setenv("OMBRE_MOOD_CONGRUENT_WEIGHT", "0.3")
    rng = random.Random(7)
    manager = _manager(tmp_path)
    buckets = _corpus(rng, 300)
    _install(manager, buckets, rng)
    for query in QUERIES:
        full = _sig(_run(manager, buckets, query, 20, prune=False, mood=True))
        pruned = _sig(_run(manager, buckets, query, 20, prune=True, mood=True))
        assert pruned == full, query


def test_prune_actually_shrinks_main_loop(tmp_path, monkeypatch):
    rng = random.Random(11)
    manager = _manager(tmp_path)
    buckets = _corpus(rng, 400)
    _install(manager, buckets, rng)
    seen = []
    real = manager._calc_emotion_score
    monkeypatch.setattr(manager, "_calc_emotion_score", lambda *a, **k: seen.append(1) or real(*a, **k))
    _run(manager, buckets, "有召回吗", 5, prune=False)
    full_calls = len(seen)
    seen.clear()
    _run(manager, buckets, "有召回吗", 5, prune=True)
    assert full_calls == 400
    assert 0 < len(seen) < 400


def test_flag_off_never_calls_prune(tmp_path, monkeypatch):
    rng = random.Random(5)
    manager = _manager(tmp_path)
    buckets = _corpus(rng, 50)
    _install(manager, buckets, rng)
    monkeypatch.setattr(manager, "_bounded_topic_scores_live",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该调用")))
    _run(manager, buckets, "粽子", 5, prune=False)


def test_prune_failure_falls_back_to_full_scan(tmp_path, monkeypatch):
    rng = random.Random(9)
    manager = _manager(tmp_path)
    buckets = _corpus(rng, 120)
    _install(manager, buckets, rng)
    full = _sig(_run(manager, buckets, "宝宝边牧呢", 5, prune=False))
    monkeypatch.setattr(manager, "_bounded_topic_scores_live",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _sig(_run(manager, buckets, "宝宝边牧呢", 5, prune=True)) == full
