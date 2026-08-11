"""review_queue 自测 —— 纯文件存储，不打任何 API、不碰 server。

覆盖：入队幂等去重、按 kind 列 pending、resolve 显式裁决、key 稳定性、
safe/review 关系分级覆盖完整、render_md 不炸。
"""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from review_queue import (
    ReviewQueue, lifecycle_updates, make_e_proposal_entry, make_metabolism_entry, make_relation_entry,
    make_z_conflict_entry,
    make_z_pair_entry, render_md,
    KIND_E_PROPOSAL, KIND_METABOLISM, KIND_RELATION, KIND_Z_CONFLICT,
    STATUS_PENDING, STATUS_APPLIED, STATUS_REJECTED,
    ReviewQueueCorruptError,
)
from utils import (
    RELATION_TYPES, SAFE_RELATION_TYPES, REVIEW_RELATION_TYPES,
    default_graph_relation_allowed,
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


def test_pending_relation_replay_only_enriches_missing_names():
    q = _q()
    original = make_relation_entry("a", "b", "causes", "原备注")
    assert q.enqueue(original) is True

    replay = make_relation_entry(
        "a",
        "b",
        "causes",
        "新备注不覆盖",
        source_name="源桶",
        target_name="目标桶",
    )
    assert q.enqueue(replay) is False

    stored = q.get(original["key"])
    assert stored["source_name"] == "源桶"
    assert stored["target_name"] == "目标桶"
    assert stored["note"] == "原备注"
    assert len(q.all()) == 1


def test_enqueue_is_idempotent_across_concurrent_writers():
    q = _q()
    entry = make_relation_entry("a", "b", "causes")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: q.enqueue(entry), range(32)))
    assert results.count(True) == 1
    assert len(q.list_pending()) == 1


def test_corrupt_row_fails_closed_instead_of_looking_empty():
    q = _q()
    os.makedirs(q.path.parent, exist_ok=True)
    q.path.write_text("{not json}\n", encoding="utf-8")
    try:
        q.list_pending()
        assert False, "corrupt review queue must fail closed"
    except ReviewQueueCorruptError:
        pass


def test_list_pending_by_kind():
    q = _q()
    q.enqueue(make_relation_entry("a", "b", "causes"))
    q.enqueue(make_z_conflict_entry("buk1", "number", "3", "5"))
    q.enqueue(make_metabolism_entry(
        "stale_important",
        "mark_review",
        "高重要度桶长期未激活，只建议复核。",
        bucket_ids=["old"],
    ))
    q.enqueue(make_e_proposal_entry(
        "source",
        "relationship_moment",
        "值得主 AI 复看",
        "这只是机器证据摘要，不是体验正文。",
        suggested_priority=70,
    ))
    assert len(q.list_pending()) == 4
    assert len(q.list_pending(KIND_RELATION)) == 1
    assert len(q.list_pending(KIND_Z_CONFLICT)) == 1
    assert len(q.list_pending(KIND_METABOLISM)) == 1
    assert len(q.list_pending(KIND_E_PROPOSAL)) == 1


def test_resolve_removes_from_pending():
    q = _q()
    e = make_z_pair_entry("new", "old", fact_key="profile.city")
    q.enqueue(e)
    assert len(q.list_pending()) == 1
    assert q.apply_lifecycle(
        e["key"],
        reviewer="哥哥",
        verdict_note="确认翻转",
    ) is True
    assert q.list_pending() == []               # 不再 pending
    rows = q.all()
    assert rows[0]["status"] == STATUS_APPLIED
    assert rows[0]["reviewer"] == "哥哥"
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
    try:
        q.resolve(e["key"], STATUS_APPLIED)
        assert False, "applied must go through the paired lifecycle transaction"
    except ValueError:
        pass


def test_z_conflict_key_varies_by_value():
    # 同桶同字段、不同 old→new 是不同事件，应各记一条
    q = _q()
    q.enqueue(make_z_conflict_entry("buk1", "number", "3", "5"))
    q.enqueue(make_z_conflict_entry("buk1", "number", "5", "9"))
    assert len(q.list_pending()) == 2


def test_z_pair_uses_canonical_fact_contract():
    entry = make_z_pair_entry(
        "new",
        "old",
        fact_key="profile.city",
        current_name="杭州",
        historical_name="北京",
    )
    current, historical = lifecycle_updates(entry)

    assert entry["fact_key"] == "profile.city"
    assert entry["field"] == "fact_status"
    assert current == {
        "fact_status": "current",
        "fact_key": "profile.city",
        "supersedes_bucket_ids": ["old"],
    }
    assert historical == {
        "fact_status": "historical",
        "fact_key": "profile.city",
        "superseded_by_bucket_id": "new",
    }
    assert "active_fact" not in current and "lifecycle" not in historical


def test_z_pair_key_includes_fact_slot():
    city = make_z_pair_entry("new", "old", fact_key="profile.city")
    job = make_z_pair_entry("new", "old", fact_key="profile.job")
    assert city["key"] != job["key"]


def test_z_pair_rejects_missing_fact_key():
    try:
        make_z_pair_entry("new", "old", fact_key="")
        assert False, "fact_key must be explicit"
    except ValueError:
        pass


def test_entry_shapes():
    r = make_relation_entry("s", "t", "improves", "note", target_name="目标桶")
    assert r["kind"] == KIND_RELATION and r["status"] == STATUS_PENDING
    assert r["rel_type"] == "improves" and r["target_name"] == "目标桶"
    z = make_z_conflict_entry("b", "date", "2026-05-14", "2026-05-13", bucket_name="约定")
    assert z["kind"] == KIND_Z_CONFLICT and z["bucket_name"] == "约定"
    assert z["old"] == "2026-05-14" and z["new"] == "2026-05-13"
    m = make_metabolism_entry(
        "oversized_buckets",
        "split_thread",
        "正文超过阈值，仅建议人工拆分。",
        severity="info",
        bucket_ids=["b", "a", "a"],
    )
    assert m["kind"] == KIND_METABOLISM
    assert m["bucket_ids"] == ["a", "b"]
    assert m["reason"] and m["status"] == STATUS_PENDING
    e = make_e_proposal_entry(
        "source",
        "relationship_moment",
        "主 AI 待写",
        "模型只提供证据。",
        suggested_priority=77,
    )
    assert e["kind"] == KIND_E_PROPOSAL
    assert e["authority"] == "proposal_only"


def test_metabolism_entry_requires_reason_and_known_action():
    try:
        make_metabolism_entry("check", "archive", "")
        assert False, "M suggestions require a human-readable reason"
    except ValueError:
        pass
    try:
        make_metabolism_entry("check", "apply_now", "must never auto-apply")
        assert False, "unknown or mutating actions must fail closed"
    except ValueError:
        pass


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
    q.enqueue(make_metabolism_entry(
        "oversized_buckets",
        "split_thread",
        "正文超过阈值，仅建议人工拆分。",
        bucket_ids=["long"],
    ))
    q.enqueue(make_e_proposal_entry(
        "source",
        "relationship_moment",
        "主 AI 待写",
        "模型只提供证据。",
        suggested_priority=77,
    ))
    md = render_md(q.list_pending())
    assert "关系闸" in md and "Z轴" in md and "M轴" in md and "E轴" in md
    assert "桶A" in md and "数量桶" in md
    assert "正文超过阈值" in md
    assert "模型只提供证据" in md
    # 空清单也不炸
    assert "✅ 无" in render_md([])


def test_relation_partition_covers_all_types():
    # 自动建边的每条都要能判 safe/review：两集合恰好覆盖、不漏不叠
    assert SAFE_RELATION_TYPES | REVIEW_RELATION_TYPES == RELATION_TYPES
    assert not (SAFE_RELATION_TYPES & REVIEW_RELATION_TYPES)
    # 危险类就是「因果 / 取代」那四个
    assert REVIEW_RELATION_TYPES == {"causes", "contributes", "improves", "updates"}
    assert SAFE_RELATION_TYPES == {"kin", "explains"}


def test_default_graph_expansion_rejects_review_and_unknown_edges():
    assert all(default_graph_relation_allowed(value) for value in SAFE_RELATION_TYPES)
    assert not any(default_graph_relation_allowed(value) for value in REVIEW_RELATION_TYPES)
    assert default_graph_relation_allowed(" made_up ") is False
    assert default_graph_relation_allowed(None) is False
