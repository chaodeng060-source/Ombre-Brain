"""2026-08-19 小卷复核（task_wr_976a40a17657_02 review）三个 P1 + 一个 P2 的回归。

P1-2  扫描 / candidate API / apply-lifecycle 三层共用一个完整豁免闸：
      permanent(type/flag)、nsfw/is_nsfw、pinned、protected、六个保护域逐项负例。
P1-3  未命中任何已登记槽（fact_keys 空）的 fact 查询必须 fail-open：
      一个 historical 都不删；明确命中槽时仍压旧（正例保留）。
P1-1  夜跑默认把 Z 候选入待审队列（pending）：临时库「有候选 → 只新增 pending →
      桶字节不变 → 二跑幂等」。
P2    review_queue 已 durable applied 之后尾部异常：事务返回 applied/changed=True，
      不再抛错让 API 报 503 / mutated=false。
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import textwrap

import pytest
import yaml

import fact_conflicts
import fact_slots
from fact_slots import (
    fact_slot_applies_to_bucket,
    filter_fact_slot_candidates,
    is_fact_slot_exempt,
)

REGISTRY = yaml.safe_load(
    pathlib.Path(__file__).with_name("config.example.yaml").read_text(encoding="utf-8")
)["fact_slots"]["registry"]

IP_KEY = "infra.zhaodeng_windows.ip"


def _bucket(bid, name, content, *, created, domain=("工程",), btype="dynamic", **extra):
    meta = {"id": bid, "name": name, "type": btype, "domain": list(domain), "created": created}
    meta.update(extra)
    return {"id": bid, "metadata": meta, "content": content}


def _ip_pair(**extra_on_both):
    old = _bucket("ip-old", "朝灯电脑 IP", "IP: 192.168.1.10\n", created="2026-07-05T10:00:00+08:00", **extra_on_both)
    new = _bucket("ip-new", "朝灯 Windows 真实IP", "IP: 192.168.1.52\n", created="2026-08-10T10:00:00+08:00", **extra_on_both)
    return old, new


# ───────────────────────── P1-2：三层同一闸 ─────────────────────────

PROTECTION_CASES = [
    ("permanent_flag", {"permanent": True}),
    ("permanent_type", {"btype": "permanent"}),
    ("nsfw", {"nsfw": True}),
    ("is_nsfw", {"is_nsfw": True}),
    ("pinned", {"pinned": True}),
    ("protected", {"protected": True}),
    *[(f"domain:{d}", {"domain": (d,)}) for d in sorted(fact_slots.PROTECTED_FACT_DOMAINS)],
]


@pytest.mark.parametrize("label,extra", PROTECTION_CASES, ids=[c[0] for c in PROTECTION_CASES])
def test_protection_gate_is_shared_by_scanner_and_apply_validation(label, extra):
    old, new = _ip_pair(**extra)
    # 1) 扫描器不拿它当候选
    assert fact_conflicts.is_z_scan_candidate(old) is False, label
    assert fact_conflicts.is_z_scan_candidate(new) is False, label
    # 2) 豁免闸本身
    assert is_fact_slot_exempt(old) is True, label
    # 3) apply 前重校验走的 fact_slot_applies_to_bucket 同样拒绝 → 手工 candidate/approve 写不进去
    assert fact_slot_applies_to_bucket(IP_KEY, old, REGISTRY) is False, label
    assert fact_slot_applies_to_bucket(IP_KEY, new, REGISTRY) is False, label


def test_unprotected_ip_pair_still_passes_all_three_layers():
    """正例：没有任何保护标记的同槽桶，三层都放行（否则上面的负例是空转）。"""
    old, new = _ip_pair()
    assert fact_conflicts.is_z_scan_candidate(old) and fact_conflicts.is_z_scan_candidate(new)
    assert is_fact_slot_exempt(old) is False
    assert fact_slot_applies_to_bucket(IP_KEY, old, REGISTRY) is True
    assert fact_slot_applies_to_bucket(IP_KEY, new, REGISTRY) is True


def test_server_pair_validation_rejects_protected_buckets(monkeypatch):
    """server._z_pair_validation_error 是 apply-lifecycle 的 validate_pair；对保护桶必须非空。"""
    import server

    monkeypatch.setattr(server, "_fact_slot_registry", lambda: REGISTRY)
    for label, extra in PROTECTION_CASES:
        old, new = _ip_pair(**extra)
        assert server._z_pair_validation_error(new, old, IP_KEY) != "", label
    old, new = _ip_pair()
    assert server._z_pair_validation_error(new, old, IP_KEY) == ""


# ───────────────────────── P1-3：空 fact_keys fail-open ─────────────────────────

def _historical_pair():
    old, new = _ip_pair()
    old["metadata"].update({"fact_key": IP_KEY, "fact_status": "historical", "superseded_by_bucket_id": "ip-new"})
    new["metadata"].update({"fact_key": IP_KEY, "fact_status": "current", "supersedes_bucket_ids": ["ip-old"]})
    return [new, old]


def test_fact_query_without_registered_slot_hit_keeps_historical():
    buckets = _historical_pair()
    for keys in (None, [], ["not.registered.key"]):
        kept = [b["id"] for b in filter_fact_slot_candidates(buckets, intent="fact", registry=REGISTRY, fact_keys=keys)]
        assert kept == ["ip-new", "ip-old"], keys


def test_fact_query_with_explicit_slot_hit_still_suppresses_historical():
    buckets = _historical_pair()
    kept = [b["id"] for b in filter_fact_slot_candidates(buckets, intent="fact", registry=REGISTRY, fact_keys=[IP_KEY])]
    assert kept == ["ip-new"]


def test_neutral_profile_query_keeps_top5_byte_identical(monkeypatch):
    """小卷的复现：注册表只有“城市”，问「具体地址是多少」→ neutral/fact_keys=() → 不许删 historical。

    走 server._filter_z_fact_candidates 这层（不是直接调 fact_slots），max_results=5 逐字对照。
    """
    import server

    registry = {
        "user.profile.city": {"aliases": ["城市", "住在哪"], "types": ["dynamic"], "name_contains": ["城市"]},
    }
    monkeypatch.setattr(server, "_fact_slot_registry", lambda: registry)
    cur = _bucket("cur", "城市·现在", "城市: B 城\n", created="2026-08-01T10:00:00+08:00",
                  fact_key="user.profile.city", fact_status="current", supersedes_bucket_ids=["old"])
    old = _bucket("old", "城市·以前", "城市: A 城\n", created="2026-07-01T10:00:00+08:00",
                  fact_key="user.profile.city", fact_status="historical", superseded_by_bucket_id="cur")
    others = [_bucket(f"m{i}", f"杂记{i}", f"第 {i} 条。", created=f"2026-08-1{i}T10:00:00+08:00", domain=("日常小事",)) for i in range(3)]
    candidates = [cur, old, *others]
    before = json.dumps(candidates, ensure_ascii=False, sort_keys=True)
    filtered = server._filter_z_fact_candidates(candidates, query="具体地址是多少", intent="fact")
    assert [b["id"] for b in filtered][:5] == [b["id"] for b in candidates][:5]
    assert json.dumps(filtered, ensure_ascii=False, sort_keys=True) == before
    # 正例：明确命中「城市」槽，旧版本才被压
    filtered = server._filter_z_fact_candidates(candidates, query="现在住在哪个城市", intent="fact")
    assert "old" not in {b["id"] for b in filtered}
    assert "cur" in {b["id"] for b in filtered}


# ───────────────────────── P1-1：夜跑默认入队（临时库） ─────────────────────────

def _write_bucket_file(root: pathlib.Path, bucket: dict) -> pathlib.Path:
    """真桶格式：buckets_dir/dynamic/<id>.md + YAML frontmatter（patrol 与 BucketManager 都认）。"""
    meta = dict(bucket["metadata"])
    text = "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n" + bucket["content"]
    (root / "dynamic").mkdir(exist_ok=True)
    path = root / "dynamic" / f"{bucket['id']}.md"
    path.write_text(text, encoding="utf-8")
    return path


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_night_patrol_enqueues_z_candidates_by_default_and_is_idempotent(tmp_path, monkeypatch):
    import patrol_night
    from review_queue import ReviewQueue, KIND_Z_CONFLICT, STATUS_PENDING

    buckets_dir = tmp_path / "buckets"
    buckets_dir.mkdir()
    old, new = _ip_pair()
    paths = [_write_bucket_file(buckets_dir, b) for b in (old, new)]
    misc = _bucket("misc", "今天的天气", "下雨了。", created="2026-08-18T10:00:00+08:00", domain=("日常小事",))
    paths.append(_write_bucket_file(buckets_dir, misc))
    before = {p.name: _sha(p) for p in paths}

    cfg = {"buckets_dir": str(buckets_dir), "fact_slots": {"enabled": True, "registry": REGISTRY}}
    # 不写 auto_enqueue_z_candidates —— 验的就是缺省值
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    first = patrol_night.run_nightly_patrol(cfg_path, tmp_path / "state")
    assert first["ok"] is True, first
    assert first["z_candidate_count"] >= 1, first
    assert first["z_queued_count"] == first["z_candidate_count"], first

    queue = ReviewQueue(buckets_dir / "review_queue.jsonl", maintenance_root=buckets_dir)
    pend = [r for r in queue.all() if r.get("kind") == KIND_Z_CONFLICT and r.get("status") == STATUS_PENDING]
    assert len(pend) == first["z_queued_count"]
    # 桶一个字节没变
    assert {p.name: _sha(p) for p in paths} == before

    second = patrol_night.run_nightly_patrol(cfg_path, tmp_path / "state")
    assert second["ok"] is True
    assert second["z_queued_count"] == 0, second
    pend2 = [r for r in queue.all() if r.get("kind") == KIND_Z_CONFLICT and r.get("status") == STATUS_PENDING]
    assert len(pend2) == len(pend)
    assert {p.name: _sha(p) for p in paths} == before


def test_night_patrol_enqueue_can_be_explicitly_disabled(tmp_path):
    import patrol_night

    buckets_dir = tmp_path / "buckets"
    buckets_dir.mkdir()
    for b in _ip_pair():
        _write_bucket_file(buckets_dir, b)
    cfg = {"buckets_dir": str(buckets_dir),
           "fact_slots": {"enabled": True, "registry": REGISTRY, "auto_enqueue_z_candidates": False}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    status = patrol_night.run_nightly_patrol(cfg_path, tmp_path / "state")
    assert status["ok"] is True
    assert status["z_candidate_count"] >= 1
    assert status["z_queued_count"] == 0


# ───────────────────────── P2：durable applied 后尾部异常 ─────────────────────────

def test_apply_returns_applied_when_queue_durable_before_trailing_error(tmp_path, monkeypatch):
    from review_queue import ReviewQueue, make_z_pair_entry
    from z_lifecycle import ZLifecycleTransaction
    import bucket_manager as bm_mod

    buckets_dir = tmp_path / "buckets"
    buckets_dir.mkdir()
    old, new = _ip_pair()
    for b in (old, new):
        _write_bucket_file(buckets_dir, b)
    queue = ReviewQueue(buckets_dir / "review_queue.jsonl", maintenance_root=buckets_dir)
    entry = make_z_pair_entry(new["id"], old["id"], fact_key=IP_KEY,
                              current_name=new["metadata"]["name"], historical_name=old["metadata"]["name"],
                              reason="test", source="test")
    assert queue.enqueue(entry) is True
    key = entry["key"]

    bm = bm_mod.BucketManager({"buckets_dir": str(buckets_dir)})
    tx = ZLifecycleTransaction(buckets_dir, bm, queue)

    # 让「队列已 durable applied」之后的第一步炸：包一层 apply_lifecycle，先真落盘再抛
    real_apply = queue.apply_lifecycle

    def apply_then_explode(*a, **kw):
        changed = real_apply(*a, **kw)
        assert changed is True
        raise OSError("disk hiccup after durable commit")

    monkeypatch.setattr(queue, "apply_lifecycle", apply_then_explode)

    result = tx.apply(key, reviewer="哥哥", verdict_note="p2", validate_pair=lambda c, h, k: "")
    assert result["status"] == "applied"
    assert result["changed"] is True
    assert "recovered_after_error" in result
    assert queue.get(key)["status"] == "applied"
    # 两桶目标已落盘（恢复完成）
    old_text = (buckets_dir / "dynamic" / "ip-old.md").read_text(encoding="utf-8")
    new_text = (buckets_dir / "dynamic" / "ip-new.md").read_text(encoding="utf-8")
    assert "historical" in old_text and "ip-new" in old_text
    assert "current" in new_text and "ip-old" in new_text
