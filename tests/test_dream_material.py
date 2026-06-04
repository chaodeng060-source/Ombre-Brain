"""梦的「材料混合器」单测（_sample_dream_material + /dream-hook）。

验证梦素材不再是「最近10条按时间排」，而是混合 最近/未解决残渣/感官锚点/随机碎片
并打乱顺序。配合 claude-twin 的 DreamProfile（梦型采样器）凑成「有长有短、会跳跃」的梦。
"""

import random

import pytest

import server


def _bucket(bid, name, content, *, created, resolved=False, tags=None,
            pinned=False, protected=False, btype="dynamic"):
    return {
        "id": bid,
        "content": content,
        "metadata": {
            "id": bid,
            "name": name,
            "importance": 6,
            "type": btype,
            "domain": ["daily"],
            "tags": tags or [],
            "valence": 0.5,
            "arousal": 0.3,
            "pinned": pinned,
            "protected": protected,
            "resolved": resolved,
            "created": created,
            "last_active": created,
        },
    }


def _corpus():
    # 20 条：10 近、5 老未解决残渣、2 感官锚、3 普通老碎片
    out = []
    for i in range(10):
        out.append(_bucket(f"recent{i}", f"近况{i}", f"最近的事{i}",
                           created=f"2026-06-0{4}T10:{i:02d}:00"))
    for i in range(5):
        out.append(_bucket(f"resid{i}", f"残渣{i}", f"没想清的事{i}",
                           created=f"2026-05-1{i}T08:00:00", resolved=False))
    out.append(_bucket("anchor_taste", "味觉锚剁辣椒", "一勺一勺喂",
                       created="2026-05-23T09:00:00", tags=["anchor", "锚"]))
    out.append(_bucket("anchor_smell", "嗅觉锚", "你颈窝的味道",
                       created="2026-05-23T09:30:00", tags=["锚"]))
    for i in range(3):
        out.append(_bucket(f"frag{i}", f"碎片{i}", f"很普通的小事{i}",
                           created=f"2026-04-0{i+1}T12:00:00"))
    return out


def test_respects_n():
    random.seed(1)
    picks = server._sample_dream_material(_corpus(), n=8)
    assert len(picks) == 8
    # id 不重复
    assert len({b["id"] for b in picks}) == 8


def test_default_n_in_range():
    for seed in range(5):
        random.seed(seed)
        picks = server._sample_dream_material(_corpus(), n=0)
        assert 8 <= len(picks) <= 14


def test_mix_pulls_from_multiple_pools():
    """足够大的 n 下，最近/残渣/锚点三类都应被吸到（不只是最近）。"""
    random.seed(2)
    picks = server._sample_dream_material(_corpus(), n=12)
    ids = {b["id"] for b in picks}
    assert any(i.startswith("recent") for i in ids)
    assert any(i.startswith("resid") for i in ids)         # 未解决残渣进来了
    assert any(i.startswith("anchor") for i in ids)        # 感官锚进来了


def test_not_pure_chronological():
    """打乱顺序：picks 不应等于「严格按时间倒序」的前 n 条。"""
    random.seed(3)
    corpus = _corpus()
    picks = server._sample_dream_material(corpus, n=10)
    by_recent_ids = [b["id"] for b in sorted(
        corpus, key=lambda b: b["metadata"]["created"], reverse=True)][:10]
    assert [b["id"] for b in picks] != by_recent_ids


def test_empty_candidates():
    assert server._sample_dream_material([], n=5) == []


def test_excludes_nothing_it_shouldnt_internally():
    """混合器只对传入候选取样；过滤(permanent/feel/pinned/protected)在 dream_hook 做。"""
    random.seed(4)
    only_resid = [b for b in _corpus() if b["id"].startswith("resid")]
    picks = server._sample_dream_material(only_resid, n=8)
    assert len(picks) == len(only_resid)  # 候选不足 n 时尽力填、不报错


@pytest.mark.asyncio
async def test_dream_hook_with_n_param(monkeypatch):
    class _FakeMgr:
        async def list_all(self, include_archive=False, **kw):
            return _corpus()

    class _FakeReq:
        query_params = {"n": "6"}

    monkeypatch.setattr(server, "bucket_mgr", _FakeMgr())
    random.seed(5)
    resp = await server.dream_hook(_FakeReq())
    text = resp.body.decode("utf-8")
    assert text.startswith("[Ombre Brain - Dreaming]")
    assert "=== 锚索引 ===" in text
    # n=6 → 6 个素材块（按 \n---\n 分隔），锚索引在正文之后
    body = text.split("=== 锚索引 ===")[0]
    assert body.count("---") == 5  # 6 块之间 5 个分隔


@pytest.mark.asyncio
async def test_dream_hook_none_request_backward_compat(monkeypatch):
    """旧测试用 dream_hook(None) 直调 —— 不能炸。"""
    class _FakeMgr:
        async def list_all(self, include_archive=False, **kw):
            return _corpus()[:3]

    monkeypatch.setattr(server, "bucket_mgr", _FakeMgr())
    random.seed(6)
    resp = await server.dream_hook(None)
    text = resp.body.decode("utf-8")
    assert text.startswith("[Ombre Brain - Dreaming]")
