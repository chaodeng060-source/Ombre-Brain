"""Z 轴闭环链路回归：候选 → 入队 entry → 人批准（lifecycle_updates）→ 召回压旧/标注。

验收标准（task_wr_976a40a17657_02）的单测版：
- 三条已知过期事实（记忆库位置 / 朝灯 Windows IP / 本地模型名）被检出、批准后，
  exact-fact 召回把旧桶压掉、只剩新桶；旧桶带 historical 标注。
- 保护域 / feel / permanent / pinned / nsfw 桶零检出、零变动。
- 非 fact intent（普通语义查询）候选集不变——「非 Z 查询 top5 不变」的精神。
- 批准之前（只入队）什么都不变。
"""
import copy
import pathlib

import yaml

from fact_slots import (
    fact_state_label,
    filter_fact_slot_candidates,
    profile_fact_state_query,
    align_fact_state_candidates,
)
from review_queue import KIND_Z_CONFLICT, STATUS_PENDING, lifecycle_updates, make_z_pair_entry
from z_candidates import propose_z_pair_candidates

REGISTRY = yaml.safe_load(
    pathlib.Path(__file__).with_name("config.example.yaml").read_text(encoding="utf-8")
)["fact_slots"]["registry"]


def _bucket(bid, name, content, *, created, domain=("工程",), btype="dynamic", **extra):
    meta = {"id": bid, "name": name, "type": btype, "domain": list(domain), "created": created}
    meta.update(extra)
    return {"id": bid, "metadata": meta, "content": content}


def _library():
    return [
        _bucket("mem-old", "记忆库在NAS", "记忆库: 本地 NAS 上\n哥哥的记忆存在 NAS。", created="2026-07-01T10:00:00+08:00"),
        _bucket("mem-new", "记忆库搬到远程", "记忆库: memory.zhaodeng.xyz\n记忆已经搬到远程。", created="2026-08-01T10:00:00+08:00"),
        _bucket("ip-old", "朝灯电脑 IP", "IP: 192.168.1.10\n", created="2026-07-05T10:00:00+08:00"),
        _bucket("ip-new", "朝灯 Windows 真实IP", "IP: 192.168.1.52\n", created="2026-08-10T10:00:00+08:00"),
        _bucket("llm-old", "本地模型选型", "本地模型: qwen2:1.5b\n", created="2026-06-20T10:00:00+08:00"),
        _bucket("llm-new", "本地模型换了", "本地模型: qwen3:4b\n", created="2026-08-05T10:00:00+08:00"),
        # 保护域 / 类型 / 标记：同字样也不许动
        _bucket("love", "记忆库纪念日", "记忆库: 心里\n", created="2026-07-15T10:00:00+08:00", domain=("恋爱",)),
        _bucket("feel", "记忆库位置", "记忆库: 心里\n", created="2026-07-15T10:00:00+08:00", btype="feel"),
        _bucket("perm", "朝灯 Windows IP", "IP: 1.1.1.1\n", created="2026-06-01T10:00:00+08:00", btype="permanent"),
        _bucket("pin", "本地模型", "本地模型: pinned\n", created="2026-06-01T10:00:00+08:00", pinned=True),
        _bucket("nsfw", "本地模型", "本地模型: x\n", created="2026-08-02T10:00:00+08:00", nsfw=True),
        # 无关桶
        _bucket("misc", "今天的天气", "下雨了。", created="2026-08-18T10:00:00+08:00", domain=("日常小事",)),
    ]


def _approve(buckets, entry):
    """模拟 ZLifecycleTransaction.apply 对两桶 metadata 的写法（字段语义同 lifecycle_updates）。"""
    by_id = {b["id"]: b for b in buckets}
    cur_upd, hist_upd = lifecycle_updates(entry)
    by_id[entry["current_bucket_id"]]["metadata"].update(cur_upd)
    by_id[entry["historical_bucket_id"]]["metadata"].update(hist_upd)


def test_full_chain_candidates_to_recall_suppression():
    buckets = _library()
    before = copy.deepcopy(buckets)

    # 1) 候选：三对，且只有这三对
    report = propose_z_pair_candidates(buckets, REGISTRY)
    pairs = {(c["fact_key"], c["current_bucket_id"], c["historical_bucket_id"]) for c in report["candidates"]}
    assert pairs == {
        ("infra.memory_store.location", "mem-new", "mem-old"),
        ("infra.zhaodeng_windows.ip", "ip-new", "ip-old"),
        ("infra.local_llm.model_name", "llm-new", "llm-old"),
    }

    # 2) 入队 entry 形状（review_queue 合同）——只入队，桶不变
    entries = [
        make_z_pair_entry(c["current_bucket_id"], c["historical_bucket_id"], fact_key=c["fact_key"],
                          current_name=c["current_name"], historical_name=c["historical_name"],
                          reason=c["reason"], source="patrol_z_scan")
        for c in report["candidates"]
    ]
    assert all(e["kind"] == KIND_Z_CONFLICT and e["status"] == STATUS_PENDING for e in entries)
    assert buckets == before, "入队不改桶"

    # 3) 批准前：exact-fact 召回两桶都在（还没有人说旧的过期）
    for key, old_id, new_id in (
        ("infra.memory_store.location", "mem-old", "mem-new"),
        ("infra.zhaodeng_windows.ip", "ip-old", "ip-new"),
        ("infra.local_llm.model_name", "llm-old", "llm-new"),
    ):
        kept = {b["id"] for b in filter_fact_slot_candidates(buckets, intent="fact", registry=REGISTRY, fact_keys=[key])}
        assert {old_id, new_id} <= kept

    # 4) 人批准三对
    for e in entries:
        _approve(buckets, e)

    # 5) 批准后：exact-fact 召回压掉旧桶、留下新桶；旧桶带 historical 标注、新桶 current
    for key, old_id, new_id in (
        ("infra.memory_store.location", "mem-old", "mem-new"),
        ("infra.zhaodeng_windows.ip", "ip-old", "ip-new"),
        ("infra.local_llm.model_name", "llm-old", "llm-new"),
    ):
        kept = {b["id"] for b in filter_fact_slot_candidates(buckets, intent="fact", registry=REGISTRY, fact_keys=[key])}
        assert old_id not in kept, (key, old_id)
        assert new_id in kept, (key, new_id)
        by_id = {b["id"]: b for b in buckets}
        assert fact_state_label(by_id[old_id], REGISTRY) == "historical"
        assert fact_state_label(by_id[new_id], REGISTRY) == "current"
        assert by_id[old_id]["metadata"]["superseded_by_bucket_id"] == new_id
        assert by_id[new_id]["metadata"]["supersedes_bucket_ids"] == [old_id]

    # 6) 保护桶 / 无关桶 一个字节没变
    by_id_before = {b["id"]: b for b in before}
    for pid in ("love", "feel", "perm", "pin", "nsfw", "misc"):
        assert {b["id"]: b for b in buckets}[pid] == by_id_before[pid], pid
    for pid in ("love", "feel", "perm", "pin", "nsfw"):
        assert fact_state_label({b["id"]: b for b in buckets}[pid], REGISTRY) == ""

    # 7) 非 fact intent：候选集原样（非 Z 查询不受影响）
    for intent in ("semantic", "recent", "narrative", ""):
        assert [b["id"] for b in filter_fact_slot_candidates(buckets, intent=intent, registry=REGISTRY)] == [b["id"] for b in buckets]

    # 8) 明确问历史（historical 视角）时旧桶不被压：align 只重排、不删
    profile = profile_fact_state_query("以前记忆库放在哪", REGISTRY)
    aligned = align_fact_state_candidates(buckets, profile=profile, registry=REGISTRY)
    assert {b["id"] for b in aligned} == {b["id"] for b in buckets}


def test_unregistered_or_context_mismatch_never_suppresses():
    """fact_key 没注册 / 桶不在槽约束里，即使标了 historical 也不压——fail open（上游合同）。"""
    stray = _bucket("stray", "随手一记", "IP: 9.9.9.9\n", created="2026-07-01T10:00:00+08:00",
                    fact_key="infra.zhaodeng_windows.ip", fact_status="historical",
                    superseded_by_bucket_id="ip-new")
    unknown = _bucket("unknown", "朝灯 Windows IP 备忘", "IP: 8.8.8.8\n", created="2026-07-01T10:00:00+08:00",
                      fact_key="infra.not_registered", fact_status="historical",
                      superseded_by_bucket_id="ip-new")
    kept = {b["id"] for b in filter_fact_slot_candidates([stray, unknown], intent="fact", registry=REGISTRY,
                                                        fact_keys=["infra.zhaodeng_windows.ip"])}
    assert kept == {"stray", "unknown"}
