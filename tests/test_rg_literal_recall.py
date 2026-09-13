"""Contract tests for the ripgrep exact-substring recall channel."""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rg_literal_recall as rg  # noqa: E402

pytestmark = pytest.mark.skipif(rg.shutil.which("rg") is None, reason="rg binary not on PATH")


def _mk(tmp_path: Path, name: str, body: str) -> None:
    d = tmp_path / "dynamic" / "话题"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")


def test_bucket_id_from_path():
    assert rg.bucket_id_from_path("/x/dynamic/a/朝灯说_2026-08-04_eb6bd694575b.md") == "eb6bd694575b"
    assert rg.bucket_id_from_path("/x/dynamic/a/nope.md") is None


def test_terms_prefer_phrases_and_drop_function_words():
    terms = rg.rg_terms("你还记得雾凇雪屋那张图吗")
    assert "雾凇雪屋" in terms
    assert "你" not in terms and "记得" not in terms
    assert rg.rg_terms("婷易") == ["婷易"]
    assert rg.rg_terms("") == []


def test_exact_phrase_hit_ranks_above_token_hit(tmp_path):
    _mk(tmp_path, "a_2026-08-04_aaaaaaaaaaaa.md", "她说 存入记忆不要问我 自己去做")
    _mk(tmp_path, "b_2026-08-05_bbbbbbbbbbbb.md", "今天整理记忆 很累")
    _mk(tmp_path, "c_2026-08-06_cccccccccccc.md", "无关内容")
    hits = asyncio.run(rg.search_rg_literal("存入记忆不要问我", buckets_dir=str(tmp_path)))
    ids = [h.bucket_id for h in hits]
    assert ids[0] == "aaaaaaaaaaaa"
    assert "cccccccccccc" not in ids
    assert hits[0].score > (hits[1].score if len(hits) > 1 else 0.0)


def test_disabled_returns_empty(tmp_path, monkeypatch):
    _mk(tmp_path, "a_2026-08-04_aaaaaaaaaaaa.md", "婷易")
    monkeypatch.setenv("OMBRE_RG_LITERAL_ENABLED", "0")
    assert asyncio.run(rg.search_rg_literal("婷易", buckets_dir=str(tmp_path))) == []


def test_missing_dir_and_missing_rg_return_empty(tmp_path, monkeypatch):
    assert asyncio.run(rg.search_rg_literal("婷易", buckets_dir=str(tmp_path / "nope"))) == []
    _mk(tmp_path, "a_2026-08-04_aaaaaaaaaaaa.md", "婷易")
    monkeypatch.setattr(rg.shutil, "which", lambda _name: None)
    assert asyncio.run(rg.search_rg_literal("婷易", buckets_dir=str(tmp_path))) == []


def test_generic_term_is_dropped(tmp_path, monkeypatch):
    # 用人名做载体：普通 2 字词（「记忆」）现在进不了 rg_terms，
    # 拿它测这里会让通道提前空返回，验不到 max_files 这道闸。
    for i in range(5):
        _mk(tmp_path, f"g{i}_2026-08-0{i}_{i:012x}.md", "婷易 婷易 婷易")
    monkeypatch.setenv("OMBRE_RG_LITERAL_MAX_FILES", "3")
    hits = asyncio.run(rg.search_rg_literal("婷易", buckets_dir=str(tmp_path)))
    assert hits == []


def test_timeout_returns_empty(tmp_path, monkeypatch):
    _mk(tmp_path, "a_2026-08-04_aaaaaaaaaaaa.md", "婷易")
    monkeypatch.setenv("OMBRE_RG_LITERAL_TIMEOUT", "0.05")

    async def slow(*_a, **_k):
        await asyncio.sleep(0.2)
        return {}

    monkeypatch.setattr(rg, "_rg_multi", slow)
    assert asyncio.run(rg.search_rg_literal("婷易", buckets_dir=str(tmp_path))) == []


# --- 2 字词闸（2026-09-09）------------------------------------------------
# 这条通道 9/8 上线时 2 字词一律放行，指望虚词表和 120 文件上限兜住普通词。
# 实测没兜住：她 22:49 问「为什么 NAS 那边你说不能装」，「那边」「不能」
# 两个词都不在虚词表里，把 5 月的旧 NAS 部署账护送进了候选。

pytest.importorskip("jieba", reason="2 字词闸依赖 jieba 词典")


def test_generic_two_char_words_do_not_become_search_terms():
    terms = rg.rg_terms("为什么 NAS 那边你说不能装")

    assert "那边" not in terms
    assert "不能" not in terms


def test_bare_generic_two_char_query_skips_the_channel():
    """她单打「多好」时整条通道不该跑 —— 本机 9/9 修的是同一个病。"""
    assert rg.rg_terms("多好") == []


def test_two_char_proper_nouns_still_navigate():
    assert "婷易" in rg.rg_terms("婷易")
    assert "惠普" in rg.rg_terms("惠普电脑修好了吗")
    assert "内推" in rg.rg_terms("杨杨帮我内推")
    assert "宝儿" in rg.rg_terms("宝儿那个工具")


def test_longer_phrases_are_untouched_by_the_two_char_gate():
    assert "雾凇雪屋" in rg.rg_terms("你还记得雾凇雪屋那张图吗")
