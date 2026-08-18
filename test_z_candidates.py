from datetime import datetime

from z_candidates import (
    MATCH_CONTEXT,
    MATCH_VALUE,
    MATCH_METADATA,
    MATCH_STRUCTURED,
    REASON_SLOT_NEWER_SUPERSEDES,
    bucket_created,
    propose_z_pair_candidates,
    slot_memberships,
)

REGISTRY = {
    "infra.memory_store.location": {
        "aliases": ["记忆库", "记忆库位置"],
        "domains": ["工程"],
        "types": ["dynamic"],
        "name_contains": ["记忆库", "记忆搬"],
    },
    "infra.zhaodeng_windows.ip": {
        "aliases": ["ip", "真实ip"],
        "domains": ["工程"],
        "types": ["dynamic"],
        "name_contains": ["IP", "Windows"],
    },
    "preference.ui.primary_color": {
        "aliases": ["主色"],
        "domains": ["创作"],
        "types": ["dynamic"],
        "name_contains": ["UI"],
    },
}


def _bucket(bid, name, content, *, created, domain=("工程",), btype="dynamic", **extra):
    meta = {"id": bid, "name": name, "type": btype, "domain": list(domain), "created": created}
    meta.update(extra)
    return {"id": bid, "metadata": meta, "content": content}


def test_structured_line_beats_context_and_metadata_fact_key_counts():
    structured = _bucket("s1", "记忆库位置", "记忆库: 本地 NAS\n", created="2026-07-01T10:00:00+08:00")
    # metadata.fact_key counts only inside the slot's declared context (upstream rule);
    # here the name matches, the content has no structured line → metadata match
    via_meta = _bucket("m1", "记忆库杂记", "随手一句", created="2026-07-02T10:00:00+08:00",
                       fact_key="infra.memory_store.location", fact_value="NAS")
    outside_ctx = _bucket("m2", "杂记", "随手一句", created="2026-07-02T10:00:00+08:00",
                          fact_key="infra.memory_store.location", fact_value="NAS")
    assert slot_memberships(outside_ctx, REGISTRY) == {}
    context_only = _bucket("c1", "记忆搬到远程了", "记忆已经搬到 memory.zhaodeng.xyz，不再在 NAS。",
                           created="2026-08-01T10:00:00+08:00")
    assert slot_memberships(structured, REGISTRY)["infra.memory_store.location"]["match"] == MATCH_STRUCTURED
    assert slot_memberships(via_meta, REGISTRY)["infra.memory_store.location"]["match"] == MATCH_METADATA
    assert slot_memberships(context_only, REGISTRY)["infra.memory_store.location"]["match"] == MATCH_CONTEXT
    # unrelated slot never leaks in
    assert "preference.ui.primary_color" not in slot_memberships(context_only, REGISTRY)


def test_protected_domains_and_types_never_join_a_slot():
    love = _bucket("love", "记忆库纪念", "记忆库: 心里", created="2026-07-01T10:00:00+08:00", domain=("恋爱",))
    feel = _bucket("feel", "记忆库位置", "记忆库: 心里", created="2026-07-01T10:00:00+08:00", btype="feel")
    perm = _bucket("perm", "记忆库位置", "记忆库: 心里", created="2026-07-01T10:00:00+08:00", btype="permanent")
    nsfw = _bucket("nsfw", "记忆库位置", "记忆库: 床上", created="2026-07-01T10:00:00+08:00", nsfw=True)
    pinned = _bucket("pin", "记忆库位置", "记忆库: 钉住", created="2026-07-01T10:00:00+08:00", pinned=True)
    for bucket in (love, feel, perm, nsfw, pinned):
        assert slot_memberships(bucket, REGISTRY) == {}, bucket["id"]
    report = propose_z_pair_candidates([love, feel, perm, nsfw, pinned], REGISTRY)
    assert report["candidates"] == []
    assert report["stats"]["buckets_in_slots"] == 0


def test_newer_bucket_becomes_current_and_older_conflicting_bucket_is_historical():
    old = _bucket("mem-old", "记忆库位置", "记忆库: 本地 NAS\n", created="2026-07-01T10:00:00+08:00")
    new = _bucket("mem-new", "记忆库位置", "记忆库: 远程 memory.zhaodeng.xyz\n", created="2026-08-01T10:00:00+08:00")
    report = propose_z_pair_candidates([old, new], REGISTRY)
    assert len(report["candidates"]) == 1
    cand = report["candidates"][0]
    assert cand["fact_key"] == "infra.memory_store.location"
    assert cand["current_bucket_id"] == "mem-new"
    assert cand["historical_bucket_id"] == "mem-old"
    assert cand["reason"] == REASON_SLOT_NEWER_SUPERSEDES
    assert cand["current_values"] == ["远程 memory.zhaodeng.xyz"]
    assert cand["historical_values"] == ["本地 NAS"]
    assert report["stats"]["slots_with_pairs"] == 1


def test_same_values_and_already_linked_pairs_are_not_proposed_again():
    old = _bucket("a", "记忆库位置", "记忆库: 远程\n", created="2026-07-01T10:00:00+08:00")
    same = _bucket("b", "记忆库位置", "记忆库: 远程\n", created="2026-08-01T10:00:00+08:00")
    assert propose_z_pair_candidates([old, same], REGISTRY)["candidates"] == []

    retired = _bucket("h", "旧IP", "IP: 192.168.1.10\n", created="2026-07-10T10:00:00+08:00",
                      fact_status="historical", superseded_by_bucket_id="ip-new")
    current = _bucket("ip-new", "朝灯Windows IP", "IP: 192.168.1.52\n", created="2026-08-10T10:00:00+08:00",
                      fact_status="current", supersedes_bucket_ids=["h"])
    report = propose_z_pair_candidates([retired, current], REGISTRY)
    assert report["candidates"] == []
    assert report["stats"]["skipped_already_linked"] == 1


def test_only_newest_non_retired_bucket_acts_as_current_and_limit_is_honoured():
    b1 = _bucket("ip1", "Windows IP", "IP: 10.0.0.1\n", created="2026-06-01T10:00:00+08:00")
    b2 = _bucket("ip2", "Windows IP", "IP: 10.0.0.2\n", created="2026-07-01T10:00:00+08:00")
    b3 = _bucket("ip3", "Windows IP", "IP: 10.0.0.3\n", created="2026-08-01T10:00:00+08:00")
    report = propose_z_pair_candidates([b1, b2, b3], REGISTRY)
    assert [(c["current_bucket_id"], c["historical_bucket_id"]) for c in report["candidates"]] == [
        ("ip3", "ip2"),
        ("ip3", "ip1"),
    ]
    limited = propose_z_pair_candidates([b1, b2, b3], REGISTRY, limit=1)
    assert len(limited["candidates"]) == 1
    assert limited["stats"]["hit_limit"] is True


def test_context_only_members_are_counted_but_never_paired():
    # both buckets only match by name/domain context and carry no slot value → never a pair
    # (2026-08-18 production dry-run: context-only pairing produced 200 unrelated candidates)
    a = _bucket("k1", "记忆库杂谈", "今天聊了聊记忆库设计。", created="2026-07-01T10:00:00+08:00")
    b = _bucket("k2", "记忆库杂谈2", "又聊了聊记忆库设计。", created="2026-08-01T10:00:00+08:00")
    report = propose_z_pair_candidates([a, b], REGISTRY)
    assert report["candidates"] == []
    assert report["stats"]["memberships_by_match"][MATCH_CONTEXT] == 2
    assert report["stats"]["buckets_in_slots"] == 0
    # legacy permissive mode still needs a real content conflict
    loose = propose_z_pair_candidates([a, b], REGISTRY, allow_context_only=True)
    assert loose["candidates"] == []
    assert loose["stats"]["skipped_no_conflict"] == 1


def test_value_patterns_extract_values_and_overlapping_values_do_not_pair():
    reg = {
        "infra.zhaodeng_windows.ip": {
            "aliases": ["ip"],
            "types": ["dynamic"],
            "name_contains": ["IP", "电脑"],
            "value_patterns": [r"\b(?:10|172|192)\.(?:\d{1,3})\.(?:\d{1,3})\.(?:\d{1,3})\b"],
        }
    }
    old = _bucket("ip1", "朝灯电脑网络", "她电脑在 192.168.1.10，路由器 192.168.1.1。", created="2026-07-01T10:00:00+08:00")
    new = _bucket("ip2", "朝灯电脑真实 IP", "实测朝灯 Windows 真实 IP 是 192.168.1.52。", created="2026-08-10T10:00:00+08:00")
    same = _bucket("ip3", "电脑 IP 复述", "还是 192.168.1.52 没变，路由 192.168.1.1。", created="2026-08-12T10:00:00+08:00")
    m_old = slot_memberships(old, reg)["infra.zhaodeng_windows.ip"]
    assert m_old["match"] == MATCH_VALUE and m_old["values"] == ["192.168.1.10", "192.168.1.1"]
    report = propose_z_pair_candidates([old, new, same], reg)
    pairs = [(c["current_bucket_id"], c["historical_bucket_id"]) for c in report["candidates"]]
    # newest is ip3; ip3 vs ip2 share 192.168.1.52 → not a supersession; ip3 vs ip1 → differ → pair
    assert pairs == [("ip3", "ip1")]
    assert report["stats"]["skipped_same_values"] == 1


def test_created_parsing_accepts_datetime_and_missing_created_is_skipped():
    aware = _bucket("d1", "Windows IP", "IP: 1.1.1.1\n", created=datetime.fromisoformat("2026-07-01T10:00:00+08:00"))
    assert bucket_created(aware) == datetime(2026, 7, 1, 10, 0, 0)
    undated = _bucket("d2", "Windows IP", "IP: 2.2.2.2\n", created=None)
    report = propose_z_pair_candidates([aware, undated], REGISTRY)
    assert report["candidates"] == []
    assert report["stats"]["skipped_no_created"] == 1


def test_example_registry_catches_the_three_acceptance_facts_and_spares_protected():
    """config.example.yaml 的三个新槽：记忆库位置 / 朝灯 Windows IP / 本地模型名 —— 验收第一条的单测版。"""
    import pathlib
    import yaml
    cfg = yaml.safe_load(pathlib.Path(__file__).with_name("config.example.yaml").read_text(encoding="utf-8"))
    registry = cfg["fact_slots"]["registry"]
    buckets = [
        _bucket("mem-old", "记忆库在NAS", "记忆库: 本地 NAS 上\n", created="2026-07-01T10:00:00+08:00"),
        _bucket("mem-new", "记忆库搬到远程", "记忆库: memory.zhaodeng.xyz\n", created="2026-08-01T10:00:00+08:00"),
        _bucket("ip-old", "朝灯电脑 IP", "IP: 192.168.1.10\n", created="2026-07-05T10:00:00+08:00"),
        _bucket("ip-new", "朝灯 Windows 真实IP", "IP: 192.168.1.52\n", created="2026-08-10T10:00:00+08:00"),
        _bucket("llm-old", "本地模型选型", "本地模型: qwen2:1.5b\n", created="2026-06-20T10:00:00+08:00"),
        _bucket("llm-new", "本地模型换了", "本地模型: qwen3:4b\n", created="2026-08-05T10:00:00+08:00"),
        # 保护域 / 类型：同样字样也不许进槽
        _bucket("love", "记忆库纪念日", "记忆库: 心里\n", created="2026-07-15T10:00:00+08:00", domain=("恋爱",)),
        _bucket("feel", "记忆库位置", "记忆库: 心里\n", created="2026-07-15T10:00:00+08:00", btype="feel"),
        _bucket("perm", "朝灯 Windows IP", "IP: 1.1.1.1\n", created="2026-06-01T10:00:00+08:00", btype="permanent"),
        _bucket("nsfw", "本地模型", "本地模型: x\n", created="2026-08-02T10:00:00+08:00", nsfw=True),
    ]
    report = propose_z_pair_candidates(buckets, registry)
    pairs = {(c["fact_key"], c["current_bucket_id"], c["historical_bucket_id"]) for c in report["candidates"]}
    assert pairs == {
        ("infra.memory_store.location", "mem-new", "mem-old"),
        ("infra.zhaodeng_windows.ip", "ip-new", "ip-old"),
        ("infra.local_llm.model_name", "llm-new", "llm-old"),
    }
    touched = {c["current_bucket_id"] for c in report["candidates"]} | {c["historical_bucket_id"] for c in report["candidates"]}
    assert touched.isdisjoint({"love", "feel", "perm", "nsfw"})
