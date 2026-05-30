# ============================================================
# #5 world 串台修复：briefing 按 current_world 过滤，角色扮演世界线不渗进日常简报
# ============================================================
import json
import pytest

import server


def _wbucket(bid, world, domain, *, btype="dynamic", pinned=True,
             last_active="2026-05-30T10:00:00"):
    return {
        "id": bid,
        "content": f"原文内容标识_{bid}",
        "metadata": {
            "id": bid, "name": bid, "importance": 10, "type": btype,
            "domain": domain, "tags": [], "valence": 0.5, "arousal": 0.3,
            "pinned": pinned, "world": world, "last_active": last_active,
        },
    }


class _FakeDecay:
    is_running = True
    async def ensure_started(self): return None
    def calculate_score(self, meta): return float(meta.get("importance", 5))


class _FakeDehydrator:
    async def briefing(self, raw_material, max_chars=1000): return "LLM_BRIEF"


class _FakeMgr:
    def __init__(self, buckets): self.buckets = list(buckets)
    async def list_all(self, include_archive=False): return list(self.buckets)
    async def get(self, bid): return next((b for b in self.buckets if b["id"] == bid), None)
    async def touch(self, bid): pass


def _wire(monkeypatch, tmp_path, buckets, world):
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "current_world", world)
    monkeypatch.setattr(server, "bucket_mgr", _FakeMgr(buckets))
    monkeypatch.setattr(server, "decay_engine", _FakeDecay())
    monkeypatch.setattr(server, "dehydrator", _FakeDehydrator())
    monkeypatch.setattr(server, "_backfill_started", True)


@pytest.mark.asyncio
async def test_daily_mode_excludes_roleplay_world(monkeypatch, tmp_path):
    buckets = [
        _wbucket("daily1", "", ["约定"]),          # 日常
        _wbucket("roleplay1", "谢长夜", ["约定"]),  # 角色扮演世界线
        _wbucket("universal1", "通用", ["约定"]),   # 通用·跨世界
    ]
    _wire(monkeypatch, tmp_path, buckets, world="")   # 日常模式
    out = await server.briefing(max_chars=400, session_id="w1",
                                include_body_state=False, format="json")
    txt = json.dumps(json.loads(out), ensure_ascii=False)
    assert "daily1" in txt          # 日常桶进
    assert "universal1" in txt      # 通用进
    assert "roleplay1" not in txt   # ← 角色扮演世界线被挡在外


@pytest.mark.asyncio
async def test_roleplay_mode_excludes_daily_world(monkeypatch, tmp_path):
    buckets = [
        _wbucket("daily2", "", ["约定"]),
        _wbucket("rp2", "谢长夜", ["约定"]),
        _wbucket("uni2", "通用", ["约定"]),
    ]
    _wire(monkeypatch, tmp_path, buckets, world="谢长夜")  # 角色扮演模式
    out = await server.briefing(max_chars=400, session_id="w2",
                                include_body_state=False, format="json")
    txt = json.dumps(json.loads(out), ensure_ascii=False)
    assert "rp2" in txt             # 当前世界线进
    assert "uni2" in txt            # 通用进
    assert "daily2" not in txt      # 日常桶在角色扮演模式被挡


@pytest.mark.asyncio
async def test_feel_survives_cross_world(monkeypatch, tmp_path):
    # feel 桶即便 world=角色扮演世界线，也不该被 world 过滤误杀
    buckets = [
        _wbucket("feel_rp", "谢长夜", ["feel"], btype="feel", pinned=False),
    ]
    _wire(monkeypatch, tmp_path, buckets, world="")  # 日常模式
    out = await server.briefing(max_chars=400, session_id="w3",
                                include_body_state=False, format="json")
    assert "feel_rp" in out          # feel 跨世界保留，没被误杀
