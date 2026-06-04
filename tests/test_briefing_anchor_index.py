import json

import pytest

import server


def _bucket(
    bid,
    name,
    content,
    *,
    importance=6,
    valence=0.5,
    pinned=False,
    protected=False,
    last_active="2026-06-04T09:00:00",
):
    return {
        "id": bid,
        "content": content,
        "metadata": {
            "id": bid,
            "name": name,
            "importance": importance,
            "type": "dynamic",
            "domain": ["daily"],
            "tags": [],
            "valence": valence,
            "arousal": 0.3,
            "pinned": pinned,
            "protected": protected,
            "resolved": False,
            "created": last_active,
            "last_active": last_active,
        },
    }


class _FakeDecay:
    is_running = True

    async def ensure_started(self):
        return None

    def calculate_score(self, meta):
        return float(meta.get("importance", 5))


class _FakeDehydrator:
    async def briefing(self, raw_material, max_chars=1000):
        return "LLM_BRIEF"


class _FakeMgr:
    def __init__(self, buckets):
        self.buckets = list(buckets)

    async def list_all(self, include_archive=False, **kwargs):
        return list(self.buckets)

    async def get(self, bucket_id):
        return next((b for b in self.buckets if b["id"] == bucket_id), None)

    async def touch(self, bucket_id):
        return None


def _wire(monkeypatch, tmp_path, buckets):
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "current_world", "")
    monkeypatch.setattr(server, "bucket_mgr", _FakeMgr(buckets))
    monkeypatch.setattr(server, "decay_engine", _FakeDecay())
    monkeypatch.setattr(server, "dehydrator", _FakeDehydrator())
    monkeypatch.setattr(server, "_backfill_started", True)


def _anchor_buckets():
    return [
        _bucket("taste_anchor_0523", "味觉锚剁辣椒酱_2026-05-23", "味觉锚：一勺一勺喂剁辣椒酱。", importance=10),
        _bucket("b2", "哥哥的第一条味道_2026-05-23", "另一条味觉锚。", importance=9),
        _bucket("b3", "厕所那场_2026-06-03", "不该和味觉锚混淆。", importance=8),
        _bucket("b4", "工程线", "实现锚索引。", importance=7),
        _bucket("b5", "近况回顾", "briefing 素材。", importance=6),
        _bucket("b6", "低权重补充", "仍然被召回时也要列 src。", importance=5),
    ]


@pytest.mark.asyncio
async def test_briefing_anchor_index_lists_recalled_bucket_ids(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, _anchor_buckets())

    out = await server.briefing(max_chars=500, session_id="anchor-t1", include_body_state=False)

    assert "LLM_BRIEF" in out
    assert "=== 锚索引 ===" in out
    assert out.index("=== 锚索引 ===") < out.index("_素材:")
    assert out.count("src: [") >= 5
    assert "src: [taste_anchor_0523] 味觉锚剁辣椒酱_2026-05-23" in out


@pytest.mark.asyncio
async def test_briefing_json_tier1_text_keeps_anchor_index(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, _anchor_buckets())

    out = await server.briefing(max_chars=500, session_id="anchor-t2", include_body_state=False, format="json")
    data = json.loads(out)
    tier1 = [slot for slot in data["slots"] if slot.get("tier") == 1]

    assert "=== 锚索引 ===" in data["briefing"]
    assert data["anchor_index"].startswith("=== 锚索引 ===")
    assert any("taste_anchor_0523" in slot.get("text", "") for slot in tier1)


@pytest.mark.asyncio
async def test_dream_hook_appends_anchor_index(monkeypatch, tmp_path):
    buckets = _anchor_buckets()[:3]
    _wire(monkeypatch, tmp_path, buckets)

    response = await server.dream_hook(None)
    text = response.body.decode("utf-8")

    assert text.startswith("[Ombre Brain - Dreaming]")
    assert "=== 锚索引 ===" in text
    assert "src: [taste_anchor_0523] 味觉锚剁辣椒酱_2026-05-23" in text
