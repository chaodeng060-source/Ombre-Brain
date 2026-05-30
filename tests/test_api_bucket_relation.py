# ============================================================
# #2 去重·建边能力：给 HTTP /api/bucket/{id} POST 补 add/remove_relation
# （原 api_bucket_update 注释吹"镜像 trace"却漏了 relation；这里补齐并回归）
# ============================================================
import json
import pytest

import server


class _FakeReq:
    def __init__(self, bucket_id, body):
        self.path_params = {"bucket_id": bucket_id}
        self._body = body

    async def json(self):
        return self._body


class _RelMgr:
    def __init__(self, ids):
        self._ids = set(ids)
        self.edges = []
        self.removed = []
        self.updated = None

    async def get(self, bid):
        if bid in self._ids:
            return {"id": bid, "metadata": {}, "content": ""}
        return None

    async def add_relation(self, source, target, rel_type, note=""):
        if target not in self._ids:
            return False
        self.edges.append((source, target, rel_type, note))
        return True

    async def remove_relation(self, source, target, rel_type=""):
        self.removed.append((source, target, rel_type))
        return 1

    async def update(self, bid, **kw):
        self.updated = kw
        return True


async def _call(monkeypatch, mgr, bucket_id, body):
    monkeypatch.setattr(server, "bucket_mgr", mgr)
    resp = await server.api_bucket_update(_FakeReq(bucket_id, body))
    return resp.status_code, json.loads(bytes(resp.body).decode("utf-8"))


@pytest.mark.asyncio
async def test_add_kin_edge(monkeypatch):
    mgr = _RelMgr(["d6207", "57140"])
    status, data = await _call(monkeypatch, mgr, "d6207",
                               {"add_relation": "kin:57140:搬进你家那一夜"})
    assert status == 200
    assert data["ok"] is True
    assert ("d6207", "57140", "kin", "搬进你家那一夜") in mgr.edges
    assert data["relations"][0] == {"op": "add", "type": "kin", "target": "57140", "ok": True}


@pytest.mark.asyncio
async def test_relation_only_no_metadata_fields(monkeypatch):
    # 关键修复：只传 add_relation、不带任何 metadata 字段，不该再被"no fields"挡掉
    mgr = _RelMgr(["a", "b"])
    status, data = await _call(monkeypatch, mgr, "a", {"add_relation": "kin:b"})
    assert status == 200
    assert ("a", "b", "kin", "") in mgr.edges
    assert mgr.updated is None  # 没动 metadata


@pytest.mark.asyncio
async def test_add_relation_bad_format(monkeypatch):
    mgr = _RelMgr(["a", "b"])
    status, data = await _call(monkeypatch, mgr, "a", {"add_relation": "b"})  # 缺 type
    assert status == 400
    assert "格式错误" in data["error"]
    assert mgr.edges == []


@pytest.mark.asyncio
async def test_remove_relation(monkeypatch):
    mgr = _RelMgr(["a", "b"])
    status, data = await _call(monkeypatch, mgr, "a", {"remove_relation": "kin:b"})
    assert status == 200
    assert ("a", "b", "kin") in mgr.removed
    assert data["relations"][0]["removed"] == 1


@pytest.mark.asyncio
async def test_relation_plus_field_together(monkeypatch):
    # relation 与 metadata 字段可同时改
    mgr = _RelMgr(["a", "b"])
    status, data = await _call(monkeypatch, mgr, "a",
                               {"add_relation": "updates:b", "importance": 6})
    assert status == 200
    assert mgr.updated == {"importance": 6}
    assert ("a", "b", "updates", "") in mgr.edges


@pytest.mark.asyncio
async def test_bucket_not_found_still_404(monkeypatch):
    mgr = _RelMgr(["a"])
    status, data = await _call(monkeypatch, mgr, "ghost", {"add_relation": "kin:a"})
    assert status == 404
