"""#3 关系闸集成自测 —— 真调 server._auto_infer_edges，monkeypatch 掉
embedding / bucket_mgr / dehydrator，验「闸开时 safe 边照写、危险边改入队不写库」。

不打任何真 API、不碰真库；队列指向 tmp 文件。
"""
import asyncio
import os
import tempfile

import server
from review_queue import ReviewQueue, KIND_RELATION


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeEmbedding:
    async def search_similar(self, content, top_k=8):
        return [("buk_kin", 0.9), ("buk_cause", 0.85)]


class _FakeBucketMgr:
    def __init__(self):
        self.added = []   # 实际写库的边
        self._buckets = {
            "buk_kin": {"id": "buk_kin", "content": "同类桶", "metadata": {"name": "同类桶", "world": ""}},
            "buk_cause": {"id": "buk_cause", "content": "因果桶", "metadata": {"name": "因果桶", "world": ""}},
        }

    async def search(self, content, limit=5):
        return []

    async def get(self, bid):
        return self._buckets.get(bid)

    async def add_relation(self, source_id, target_id, rel_type, note=""):
        self.added.append((source_id, target_id, rel_type))
        return True


class _FakeDehydrator:
    async def dehydrate(self, content, meta):
        return content[:50]

    async def infer_relations(self, content, candidates):
        # 一条安全 kin、一条危险 causes
        return [
            {"type": "kin", "target": "buk_kin", "note": "同类"},
            {"type": "causes", "target": "buk_cause", "note": "导致"},
        ]


def _setup(monkeypatch_gate: bool):
    tmpdir = tempfile.mkdtemp(prefix="relgate_")
    fake_mgr = _FakeBucketMgr()
    server.embedding_engine = _FakeEmbedding()
    server.bucket_mgr = fake_mgr
    server.dehydrator = _FakeDehydrator()
    server._review_queue = ReviewQueue(os.path.join(tmpdir, "review_queue.jsonl"))
    # 直接塞队列实例，避免 _get_review_queue 跟着 config.buckets_dir 跑偏
    server._get_review_queue = lambda: server._review_queue
    server.config = dict(server.config)
    server.config["review_gate"] = {"relation_review": monkeypatch_gate}
    return fake_mgr, server._review_queue


def test_gate_off_writes_all_edges():
    mgr, q = _setup(monkeypatch_gate=False)
    added = run(server._auto_infer_edges("src", "源内容", world=""))
    types_written = sorted(t for _, _, t in mgr.added)
    assert types_written == ["causes", "kin"], types_written   # 闸关：全写
    assert q.list_pending() == []                              # 没入队
    assert sorted(e["type"] for e in added) == ["causes", "kin"]


def test_gate_on_diverts_dangerous_edge():
    mgr, q = _setup(monkeypatch_gate=True)
    added = run(server._auto_infer_edges("src", "源内容", world=""))
    types_written = [t for _, _, t in mgr.added]
    assert types_written == ["kin"], types_written            # 闸开：只安全 kin 写库
    assert [e["type"] for e in added] == ["kin"]              # 返回的也只有写成功的
    pend = q.list_pending(KIND_RELATION)
    assert len(pend) == 1 and pend[0]["rel_type"] == "causes"  # 危险 causes 改入队
    assert pend[0]["source_id"] == "src" and pend[0]["target_id"] == "buk_cause"
    assert pend[0]["target_name"] == "因果桶"


def _tool_fn(t):
    # FastMCP 装饰后可能仍是函数，也可能裹成 Tool 对象（.fn）。两种都兼容。
    return t if callable(t) and not hasattr(t, "fn") else t.fn


def test_review_pending_tool_readonly():
    _, q = _setup(monkeypatch_gate=True)
    from review_queue import make_relation_entry as mr, make_z_conflict_entry as mz
    q.enqueue(mr("src", "buk_cause", "causes", target_name="因果桶"))
    q.enqueue(mz("buk1", "number", "3", "5", bucket_name="数量桶"))
    fn = _tool_fn(server.review_pending)
    out = run(fn())
    assert "因果桶" in out and "数量桶" in out
    # kind 过滤
    rel_only = run(fn(kind="relation"))
    assert "因果桶" in rel_only and "数量桶" not in rel_only
    # 非法 kind 友好报错、不炸
    assert "只能是" in run(fn(kind="garbage"))
