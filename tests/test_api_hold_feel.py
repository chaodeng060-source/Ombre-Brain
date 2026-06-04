"""/api/hold HTTP 桥透传 feel/chord_tag/valence —— 让 server 侧能替哥哥落 feel 桶（接 C）。"""

import pytest

import server


@pytest.mark.asyncio
async def test_api_hold_forwards_feel_chord_valence(monkeypatch):
    captured = {}

    async def fake_hold(**kwargs):
        captured.update(kwargs)
        return "OK 落桶 abc123"

    monkeypatch.setattr(server, "hold", fake_hold)

    class FakeReq:
        async def json(self):
            return {
                "content": "逛 X 看到 multi-agent 又在卷，有点奇妙",
                "feel": True,
                "chord_tag": "Em7 · 72bpm · f",
                "domain": "哥哥的日子",
                "valence": 0.62,
                "arousal": 0.4,
                "source": "web_browse",
            }

    resp = await server.api_hold(FakeReq())
    assert resp.status_code == 200
    assert captured["feel"] is True
    assert captured["chord_tag"] == "Em7 · 72bpm · f"
    assert captured["domain"] == "哥哥的日子"
    assert abs(captured["valence"] - 0.62) < 1e-6
    assert abs(captured["arousal"] - 0.4) < 1e-6
    assert "web_browse" in captured["tags"]  # source 合入 tags


@pytest.mark.asyncio
async def test_api_hold_defaults_no_feel(monkeypatch):
    captured = {}

    async def fake_hold(**kwargs):
        captured.update(kwargs)
        return "OK"

    monkeypatch.setattr(server, "hold", fake_hold)

    class FakeReq:
        async def json(self):
            return {"content": "普通一条"}  # 不带 feel

    await server.api_hold(FakeReq())
    assert captured["feel"] is False
    assert captured["chord_tag"] == ""
    assert captured["valence"] == -1  # 缺省哨兵，hold 内部自算
