"""review_queue 自测 —— 纯文件存储，不打任何 API、不碰 server。

覆盖：入队幂等去重、按 kind 列 pending、resolve 显式裁决、key 稳定性、
safe/review 关系分级覆盖完整、render_md 不炸。
"""
import os
import tempfile

from review_queue import (
    ReviewQueue, make_relation_entry, make_z_conflict_entry, render_md,
    KIND_RELATION, KIND_Z_CONFLICT,
    STATUS_PENDING, STATUS_APPLIED, STATUS_REJECTED,
)
from utils import (
    RELATION_TYPES, SAFE_RELATION_TYPES, REVIEW_RELATION_TYPES,
)


def _q():
    d = tempfile.mkdtemp(prefix="rq_test_")
    return ReviewQueue(os.path.join(d, "review_queue.jsonl"))


def test_enqueue_idempotent():
    q = _q()
    e = make_relation_entry("a", "b", "causes", "因为下雨")
    assert q.enqueue(e) is True                 # 首次新增
    assert q.enqueue(e) is False                # 同 key 不重复
    # 重新构造同一条（key 只由 source/type/target 决定）也算重复
    assert q.enqueue(make_relation_entry("a", "b", "causes", "别的备注")) is False
    assert len(q.list_pending()) == 1


def test_list_pending_by_kind():
    q = _q()
    q.enqueue(make_relation_entry("a", "b", "causes"))
    q.enqueue(make_z_conflict_entry("buk1", "number", "3", "5"))
    assert len(q.list_pending()) == 2
    assert len(q.list_pending(KIND_RELATION)) == 1
    assert len(q.list_pending(KIND_Z_CONFLICT)) == 1


def test_resolve_removes_from_pending():
    q = _q()
    e = make_z_conflict_entry("buk1", "negation", "affirmed", "negated")
    q.enqueue(e)
    assert len(q.list_pending()) == 1
    assert q.resolve(e["key"], STATUS_APPLIED, verdict_note="确认翻转") is True
    assert q.list_pending() == []               # 不再 pending
    rows = q.all()
    assert rows[0]["status"] == STATUS_APPLIED
    assert rows[0]["verdict_note"] == "确认翻转"
    assert "resolved_at" in rows[0]
    # 已裁决的再 resolve 不命中（只动 pending 行）
    assert q.resolve(e["key"], STATUS_REJECTED) is False


def test_resolve_rejects_bad_status():
    q = _q()
    e = make_relation_entry("a", "b", "updates")
    q.enqueue(e)
    try:
        q.resolve(e["key"], "garbage")
        assert False, "应拒非法状态"
    except ValueError:
        pass


def test_z_conflict_key_varies_by_value():
    # 同桶同字段、不同 old→new 是不同事件，应各记一条
    q = _q()
    q.enqueue(make_z_conflict_entry("buk1", "number", "3", "5"))
    q.enqueue(make_z_conflict_entry("buk1", "number", "5", "9"))
    assert len(q.list_pending()) == 2


def test_entry_shapes():
    r = make_relation_entry("s", "t", "improves", "note", target_name="目标桶")
    assert r["kind"] == KIND_RELATION and r["status"] == STATUS_PENDING
    assert r["rel_type"] == "improves" and r["target_name"] == "目标桶"
    z = make_z_conflict_entry("b", "date", "2026-05-14", "2026-05-13", bucket_name="约定")
    assert z["kind"] == KIND_Z_CONFLICT and z["bucket_name"] == "约定"
    assert z["old"] == "2026-05-14" and z["new"] == "2026-05-13"


def test_enqueue_requires_key():
    q = _q()
    try:
        q.enqueue({"kind": KIND_RELATION})      # 无 key
        assert False, "应要求 key"
    except ValueError:
        pass


def test_render_md_smoke():
    q = _q()
    q.enqueue(make_relation_entry("a", "b", "causes", source_name="桶A", target_name="桶B"))
    q.enqueue(make_z_conflict_entry("buk1", "number", "3", "5", bucket_name="数量桶"))
    md = render_md(q.list_pending())
    assert "关系闸" in md and "Z轴" in md
    assert "桶A" in md and "数量桶" in md
    # 空清单也不炸
    assert "✅ 无" in render_md([])


def test_relation_partition_covers_all_types():
    # 自动建边的每条都要能判 safe/review：两集合恰好覆盖、不漏不叠
    assert SAFE_RELATION_TYPES | REVIEW_RELATION_TYPES == RELATION_TYPES
    assert not (SAFE_RELATION_TYPES & REVIEW_RELATION_TYPES)
    # 危险类就是「因果 / 取代」那四个
    assert REVIEW_RELATION_TYPES == {"causes", "contributes", "improves", "updates"}
    assert SAFE_RELATION_TYPES == {"kin", "explains"}
