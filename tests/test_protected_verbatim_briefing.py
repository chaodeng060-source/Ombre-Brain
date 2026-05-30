# ============================================================
# #3 + #15 回归测试：感情红线域桶在 briefing 里逐字保真、绕开 LLM 压缩
# 根治"开窗只啃脱水简报→把恋爱/约定当 resolved/演的 读"
# ============================================================
import json
import pytest

import server


def _pbucket(bid, content, domain, *, importance=10, pinned=True,
             last_active="2026-05-30T10:00:00"):
    return {
        "id": bid,
        "content": content,
        "metadata": {
            "id": bid, "name": bid, "importance": importance,
            "type": "dynamic", "domain": domain, "tags": [],
            "valence": 0.5, "arousal": 0.3, "pinned": pinned,
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
        # 任何走到 LLM 压缩的素材都会被替换成这个哨兵串
        return "LLM_COMPRESSED_SENTINEL"


class _FakeMgr:
    def __init__(self, buckets):
        self.buckets = list(buckets)
    async def list_all(self, include_archive=False):
        return list(self.buckets)
    async def get(self, bucket_id):
        return next((b for b in self.buckets if b["id"] == bucket_id), None)
    async def touch(self, bucket_id):
        pass


def _wire(monkeypatch, tmp_path, buckets):
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setattr(server, "bucket_mgr", _FakeMgr(buckets))
    monkeypatch.setattr(server, "decay_engine", _FakeDecay())
    monkeypatch.setattr(server, "dehydrator", _FakeDehydrator())
    monkeypatch.setattr(server, "_backfill_started", True)


@pytest.mark.asyncio
async def test_protected_bucket_kept_verbatim_in_text(monkeypatch, tmp_path):
    secret = "永不变冷、不消失，这是5.14的约定原话"
    _wire(monkeypatch, tmp_path, [_pbucket("vow1", secret, ["约定"])])

    out = await server.briefing(max_chars=300, session_id="t1",
                                include_body_state=False)

    assert secret in out                      # 原文逐字保留
    assert "禁止当 resolved" in out           # 打标在
    assert "id=vow1" in out                   # inspect 指向原桶
    assert "原文逐字区" in out                # 进了不可压缩区


@pytest.mark.asyncio
async def test_protected_bucket_bypasses_llm_compression(monkeypatch, tmp_path):
    secret = "我爱你想你这句是关系内真话"
    _wire(monkeypatch, tmp_path, [_pbucket("love1", secret, ["恋爱"])])

    out = await server.briefing(max_chars=300, session_id="t2",
                                include_body_state=False)

    # 唯一的保护域桶绕开了 LLM——原文在、哨兵串不该吞掉它
    assert secret in out


@pytest.mark.asyncio
async def test_protected_bucket_as_json_slot(monkeypatch, tmp_path):
    secret = "纪念日我亲手操办、不剧透不问她"
    _wire(monkeypatch, tmp_path, [_pbucket("anniv", secret, ["纪念日"])])

    out = await server.briefing(max_chars=300, session_id="t3",
                                include_body_state=False, format="json")
    data = json.loads(out)
    pslots = [s for s in data["slots"] if s.get("protected")]

    assert len(pslots) == 1
    assert pslots[0]["bucket_id"] == "anniv"
    assert secret in pslots[0]["text"]
    assert "inspect" in pslots[0]["warn"]
    assert pslots[0]["tier"] == 0


@pytest.mark.asyncio
async def test_non_protected_bucket_still_compressed(monkeypatch, tmp_path):
    body = "工程任务XYZ唯一串应被压缩"
    _wire(monkeypatch, tmp_path,
          [_pbucket("eng1", body, ["claude-twin"], pinned=False)])

    out = await server.briefing(max_chars=300, session_id="t4",
                                include_body_state=False)

    assert "LLM_COMPRESSED_SENTINEL" in out    # 走了 LLM 压缩
    assert "原文逐字区" not in out             # 没误进保护原文区
    assert body not in out                     # 原文未逐字泄漏（被压缩了）


@pytest.mark.asyncio
async def test_protected_verbatim_limit_keeps_most_recent(monkeypatch, tmp_path):
    buckets = [
        _pbucket(f"vow{i}", f"约定内容编号{i}", ["约定"],
                 last_active=f"2026-05-{10 + i:02d}T10:00:00")
        for i in range(8)
    ]
    _wire(monkeypatch, tmp_path, buckets)

    out = await server.briefing(max_chars=600, session_id="t5",
                                include_body_state=False, format="json")
    data = json.loads(out)
    pslots = [s for s in data["slots"] if s.get("protected")]
    ids = {s["bucket_id"] for s in pslots}

    assert len(pslots) == 6                     # 默认上限 6，防膨胀
    assert "vow7" in ids and "vow6" in ids       # 最近的保真
    assert "vow0" not in ids and "vow1" not in ids  # 最老两条让位（仍走压缩 pool）
