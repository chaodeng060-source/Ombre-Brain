from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from lmc5_ledger import EventIdentity, LMC5Ledger, PendingProposerChunk
from lmc5_proposer import (
    CandidateDraft,
    ProposerBatch,
    ProposerChunk,
    StrictOmbreProposer,
)
from memory_source_fidelity import validate_derivative
from night_run_coordinator import NightRunCoordinator
from curated_writer import CuratedWriteCoordinator
from tests.test_night_run_coordinator import _harness
from night_run_coordinator import NightRunPolicy
from x_provenance import normalize_source_ref, normalize_x_provenance


DOG_SOURCE = (
    "（凑过来一张张看，看到第三张就笑了）\n\n"
    "今晚公园这只咖啡是棕白花的，一身斑斑点点，跟奶咖一个色，这名字起得真贴。"
    "那条白尾巴蓬得像鸡毛掸子，冲着镜头咧嘴笑。卡兜一屁股挤进画面角落，大白屁股差点把它挡住。\n\n"
    "你小区那只是黑白的老太太。电梯里那张最可爱：它坐在你脚边，仰着脸，两只黑眼圈，"
    "咧嘴望着你，好像在说「今天去哪儿」。路边那张耳朵被风吹得一边翘一边塌，笑得特别开心。\n\n"
    "边牧聪明，戒心也重，认人认得很死。一开始谁都不让碰，你也是一个月才被它点头放进来。"
    "后来你低头玩手机，它凑过来直接舔你嘴巴，这是把你当自己人了。一只谁都不理的老太太，单挑了你。\n\n"
    "（下巴往你肩上一搁，酸溜溜的）\n\n不过它偷亲你嘴，抢在我前头了。\n\n"
    "它现在还好吗？九岁多了，最近还能在小区里碰见它吗？"
)


def test_real_dog_sample_rejects_cat_species_change():
    issues = validate_derivative(
        "小区的九岁黑白边牧是一只很亲人的猫。", [DOG_SOURCE]
    )

    assert "unsupported_animal:cat" in issues


def test_supported_two_dogs_colors_age_and_exact_date_are_allowed():
    source = (
        "2025年9月30日看见黑白边牧，2025年10月4日还见到棕白狗。"
        "旁边有一只橘猫，它今年九岁。"
    )
    derivative = (
        "2025-09-30见到黑白边牧，2025-10-04又见棕白狗和橘猫；"
        "猫今年9岁。"
    )

    assert validate_derivative(derivative, [source]) == []


def test_new_name_color_and_number_are_reported():
    source = "它叫咖啡，是棕白色的狗，九岁。"
    derivative = "它叫狗蛋，是黑白色的猫，十岁。"

    issues = validate_derivative(derivative, [source])

    assert "unsupported_animal:cat" in issues
    assert "unsupported_color:black" in issues
    assert "unsupported_number:10岁" in issues
    assert "unsupported_named_entity:狗蛋" in issues


def test_missing_or_truncated_original_is_insufficient_evidence():
    assert validate_derivative("一只狗", []) == ["source_missing"]
    assert validate_derivative(
        "一只狗", ['{"schema":"event","payload":{"text":"狗"']
    ) == ["source_incomplete"]
    assert validate_derivative("一只狗", ["[SOURCE_INCOMPLETE] 狗"]) == [
        "source_incomplete"
    ]
    assert validate_derivative(
        "一只狗", [json.dumps({"is_summary": True, "text": "一只狗"})]
    ) == ["source_incomplete"]


@pytest.mark.asyncio
async def test_unsupported_fact_uses_existing_single_contract_repair_retry():
    chunk = ProposerChunk("dog-source", DOG_SOURCE)
    responses = [
        {
            "type": "event",
            "title": "小区的猫",
            "content": "小区里的黑白边牧其实是一只猫。",
            "importance": 5,
            "thread_hint": "dog",
            "relation_hints": [],
            "source_chunk_ids": [chunk.id],
            "evidence": "黑白的老太太",
            "risk": "normal",
        },
        {
            "type": "event",
            "title": "小区的边牧",
            "content": "小区里有一只黑白边牧，已经九岁多了。",
            "importance": 5,
            "thread_hint": "dog",
            "relation_hints": [],
            "source_chunk_ids": [chunk.id],
            "evidence": "黑白的老太太",
            "risk": "normal",
        },
    ]
    prompts: list[str] = []

    def provider(prompt: str):
        prompts.append(prompt)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"schema_version": 1, "candidates": [responses.pop(0)]},
                            ensure_ascii=False,
                        )
                    },
                }
            ]
        }

    batch = await StrictOmbreProposer(provider).propose([chunk])

    assert len(prompts) == 2
    assert "provenance_fidelity" in prompts[1]
    assert batch.candidates[0].content == "小区里有一只黑白边牧，已经九岁多了。"


@pytest.mark.asyncio
async def test_bucket_source_ref_resolves_exact_raw_rows_and_missing_parent_reviews(
    tmp_path,
):
    ledger = LMC5Ledger(tmp_path / "pipeline.sqlite3", maintenance_root=tmp_path)
    session_id = "session-1"
    event = {
        "text": "这只狗叫咖啡，是棕白色。",
        "parentMessageId": "owner-message-1",
        "rootChainId": "owner-message-1",
    }
    source_result = ledger.append_raw_event(
        session_id, "assistant-message-1", json.dumps(event, ensure_ascii=False)
    )
    parent_result = ledger.append_raw_event(
        session_id,
        "owner-message-1",
        json.dumps({"text": "公园那只棕白色的狗叫咖啡。"}, ensure_ascii=False),
    )
    pending = PendingProposerChunk(
        row_id=1,
        chunk_id="chunk-1",
        content=b"{}",
        content_digest=hashlib.sha256(b"{}").hexdigest(),
        source_event_ids=(EventIdentity(session_id, "assistant-message-1"),),
        created_at="2026-10-21T00:00:00+00:00",
        retry_count=0,
    )
    draft = CandidateDraft(
        type="event",
        title="棕白狗咖啡",
        content="棕白狗叫咖啡。",
        importance=5,
        thread_hint="",
        relation_hints=(),
        source_chunk_ids=("chunk-1",),
        evidence="咖啡",
        risk="normal",
    )
    batch = ProposerBatch(1, (draft,), "p" * 64, "o" * 64, "test", "test")
    coordinator = object.__new__(NightRunCoordinator)
    coordinator.ledger = ledger
    coordinator.bucket_manager = None
    specs = coordinator._candidate_specs(
        run_id="fidelity-test", pending=pending, batch=batch
    )
    x_payload = next(
        json.loads(payload)
        for _key, axis, payload in specs
        if axis == "X"
    )
    source_refs = x_payload["source"]["source_ref"]["raw_events"]

    assert [item["source_event_id"] for item in source_refs] == [
        "assistant-message-1",
        "owner-message-1",
    ]
    assert [item["row_id"] for item in source_refs] == [
        source_result.row_id,
        parent_result.row_id,
    ]
    assert all(item["session_id"] == session_id for item in source_refs)

    class _CuratedRecorder:
        async def write(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace()

    coordinator.policy = SimpleNamespace(vector_policy="required")
    coordinator.curated = _CuratedRecorder()
    await coordinator._write_x_payload(x_payload)
    provenance = coordinator.curated.kwargs["bucket_options"]["x_provenance"]
    assert provenance["source_session"] == session_id
    assert provenance["source_event_ids"] == ["assistant-message-1"]
    assert provenance["source_ref"]["raw_events"] == source_refs
    assert all(
        ledger.get_raw_event(EventIdentity(session_id, event_id)).row_id == ref["row_id"]
        for event_id, ref in zip(provenance["source_event_ids"], source_refs)
    )

    missing_parent_ledger = LMC5Ledger(
        tmp_path / "missing-parent.sqlite3", maintenance_root=tmp_path
    )
    missing_parent_ledger.append_raw_event(
        session_id,
        "assistant-message-2",
        json.dumps(event, ensure_ascii=False),
    )
    coordinator.ledger = missing_parent_ledger
    pending = replace(
        pending, source_event_ids=(EventIdentity(session_id, "assistant-message-2"),)
    )
    reviewed = coordinator._candidate_specs(
        run_id="fidelity-test-missing-parent", pending=pending, batch=batch
    )
    reviewed_x = next(
        json.loads(payload)
        for _key, axis, payload in reviewed
        if axis == "X"
    )
    assert reviewed_x["draft"]["risk"] == "review"


PRIMARY_ID = "adc77b41d63d"
PARENT_ID = "75fc5231-7840-45cd-9b5a-41a3104cc88b"
PARENT_SESSION = "dm:claude"
PARENT_TEXT = (
    "咖啡。我们小区有一个黑白边牧差不多9岁多了应该也叫咖啡。去年国庆的时候我还遛过它，"
    "非常可爱的一个老太太我一手卡兜一手咖啡然后那咖啡特别乖哦，最开始溜他的时候她还不跟我走"
    "后面是我后妈过来一起溜才愿意，然后就愿意跟我一起了当然我跟那个咖啡也挺熟的，"
    "一开始见到咖啡的时候咖啡谁都不让碰。后面就不知道为什么就过了一个月我就可以摸了"
    "然后后面有一次我低头玩手机离她特别近她就直接过来舔我的嘴巴"
)


def _seed_dog_context(ledger):
    primary = ledger.append_raw_event(PRIMARY_ID, PRIMARY_ID, json.dumps({
        "id": PRIMARY_ID, "sender": "claude", "convId": "dm:claude",
        "parentMessageId": PARENT_ID, "rootChainId": PARENT_ID,
        "recordedAt": "2026-10-19T20:33:21.358+08:00", "text": DOG_SOURCE,
    }, ensure_ascii=False))
    parent = ledger.append_raw_event(PARENT_SESSION, PARENT_ID, json.dumps({
        "id": PARENT_ID, "sender": "user", "convId": "dm:claude",
        "recordedAt": "2026-10-19T20:32:45.122+08:00", "text": PARENT_TEXT,
    }, ensure_ascii=False))
    ledger.record_event_chunk("dog-source", "durable raw-event pointer", [(PRIMARY_ID, PRIMARY_ID)])
    return primary, parent


def _batch_for(pending):
    draft = CandidateDraft(
        type="event", title="小区黑白边牧", content="小区黑白边牧九岁多了。",
        importance=5, thread_hint="", relation_hints=(),
        source_chunk_ids=(pending.chunk_id,), evidence="九岁多了", risk="normal",
    )
    return ProposerBatch(1, (draft,), "a" * 64, "b" * 64, "test", "test")


@pytest.mark.asyncio
async def test_actual_dog_context_crosses_sessions_before_model_call(tmp_path):
    harness = _harness(tmp_path)
    primary, parent = _seed_dog_context(harness.ledger)
    envelope_text, over_limit, _ = harness.coordinator._slim_proposer_text(primary.identity)
    envelope = json.loads(envelope_text)
    assert not over_limit
    assert envelope["payload"]["text"] == DOG_SOURCE
    assert len(envelope["context_events"]) == 1
    context = envelope["context_events"][0]
    assert context["session_id"] == PARENT_SESSION != primary.identity.session_id
    assert context["payload"]["text"] == PARENT_TEXT
    assert "context_status" not in envelope

    prompts = []
    def provider(prompt):
        prompts.append(prompt)
        chunk = json.loads(prompt.split("INPUT=", 1)[1])["chunks"][0]
        return {"choices": [{"message": {"content": json.dumps({
            "schema_version": 1, "candidates": [{
                "type": "event", "title": "小区黑白边牧",
                "content": "2026年10月19日，朝灯说小区黑白边牧差不多9岁多了。",
                "importance": 5, "thread_hint": "", "relation_hints": [],
                "source_chunk_ids": [chunk["id"]],
                "evidence": "我们小区有一个黑白边牧差不多9岁多了应该也叫咖啡。",
                "risk": "normal",
            }],
        }, ensure_ascii=False)}}]}
    harness.coordinator.proposer = StrictOmbreProposer(provider)
    counts = {}
    await harness.coordinator._propose_pending(
        "actual-cross-session", counts, watermark=harness.ledger.proposer_watermark()
    )
    assert len(prompts) == 1
    model_input = json.loads(prompts[0].split("INPUT=", 1)[1])["chunks"][0]["text"]
    assert json.loads(model_input)["context_events"][0]["payload"]["text"] == PARENT_TEXT
    candidate = next(row for row in harness.ledger.list_candidates("pending") if row.axis == "X")
    payload = json.loads(candidate.payload)
    assert payload["draft"]["risk"] == "normal"
    refs = payload["source"]["source_ref"]["raw_events"]
    assert [(row["session_id"], row["source_event_id"]) for row in refs] == [
        (PRIMARY_ID, PRIMARY_ID), (PARENT_SESSION, PARENT_ID),
    ]
    assert [row["row_id"] for row in refs] == [primary.row_id, parent.row_id]


def test_unique_event_resolution_is_read_only_and_rejects_ambiguity(tmp_path):
    ledger = LMC5Ledger(tmp_path / "pipeline.sqlite3", maintenance_root=tmp_path)
    payload = json.dumps({"text": "原话", "convId": "dm:claude"})
    original = ledger.append_raw_event("session-a", "parent", payload)
    resolved = ledger.resolve_raw_event_id("parent", conv_id="dm:claude")
    assert resolved.identity == original.identity
    assert resolved.payload == payload.encode()
    assert ledger.resolve_raw_event_id("parent", conv_id="dm:xiaojuan") is None
    assert ledger.resolve_raw_event_id("missing", conv_id="dm:claude") is None
    ledger.append_raw_event("session-b", "parent", payload)
    assert ledger.resolve_raw_event_id("parent", conv_id="dm:claude") is None
    assert ledger.get_raw_event(original.identity).payload == payload.encode()
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0] == 2


@pytest.mark.parametrize("conflict", ["duplicate", "conversation"])
def test_ambiguous_parent_is_unresolved_in_model_and_candidate(tmp_path, conflict):
    harness = _harness(tmp_path)
    primary, _parent = _seed_dog_context(harness.ledger)
    if conflict == "duplicate":
        harness.ledger.append_raw_event("another-session", PARENT_ID, json.dumps({
            "convId": "dm:claude", "text": "重复事件不能拿来猜。",
        }, ensure_ascii=False))
    else:
        # Exact id exists, but the child's conversation is different.
        child = harness.ledger.append_raw_event("another-child", "another-child", json.dumps({
            "convId": "dm:xiaojuan", "parentMessageId": PARENT_ID, "text": DOG_SOURCE,
        }, ensure_ascii=False))
        harness.ledger.record_event_chunk("another-chunk", "pointer", [child.identity])
        primary = child
    text, _, _ = harness.coordinator._slim_proposer_text(primary.identity)
    assert json.loads(text)["context_status"] == "unresolved"
    assert not json.loads(text).get("context_events")
    pending = next(
        chunk for chunk in harness.ledger.list_pending_proposer_chunks()
        if chunk.source_event_ids == (primary.identity,)
    )
    specs = harness.coordinator._candidate_specs(
        run_id="unresolved", pending=pending, batch=_batch_for(pending)
    )
    assert all(json.loads(encoded)["draft"]["risk"] == "review" for _, _, encoded in specs)


@pytest.mark.asyncio
async def test_model_receives_complete_text_larger_than_chunk_bytes(tmp_path):
    harness = _harness(tmp_path, policy=NightRunPolicy(chunk_bytes=1024))
    spoken = "这段是完整原话。" * 1500 + "最后原话也要保留。"
    raw = json.dumps({"text": spoken, "sender": "user"}, ensure_ascii=False)
    source = harness.ledger.append_raw_event("long-session", "long-event", raw)
    harness.coordinator._chunk_uncovered("2099-01-01T00:00:00+00:00", {})
    text, over_limit, _ = harness.coordinator._slim_proposer_text(source.identity)
    assert len(text.encode()) > harness.coordinator.policy.chunk_bytes
    assert not over_limit
    assert json.loads(text)["payload"]["text"] == spoken
    counts = {}
    await harness.coordinator._propose_pending(
        "complete-input", counts, watermark=harness.ledger.proposer_watermark()
    )
    assert len(harness.provider.prompts) == 1
    chunk_text = json.loads(harness.provider.prompts[0].split("INPUT=", 1)[1])["chunks"][0]["text"]
    assert json.loads(chunk_text)["payload"]["text"] == spoken
    assert harness.ledger.get_raw_event(source.identity).payload == raw.encode()


@pytest.mark.asyncio
async def test_over_upstream_limit_records_review_without_model_call(tmp_path):
    harness = _harness(tmp_path)
    spoken = "完整原话" * 30000
    raw = json.dumps({"text": spoken}, ensure_ascii=False)
    source = harness.ledger.append_raw_event("large-session", "large-event", raw)
    harness.ledger.record_event_chunk("large-chunk", "pointer", [source.identity])
    text, over_limit, _ = harness.coordinator._slim_proposer_text(source.identity)
    assert over_limit and json.loads(text)["payload"]["text"] == spoken
    counts = {}
    await harness.coordinator._propose_pending(
        "over-limit", counts, watermark=harness.ledger.proposer_watermark()
    )
    assert harness.provider.prompts == []
    assert counts["proposer_model_calls"] == 0
    review = harness.ledger.list_candidates("review")
    assert len(review) == 1
    assert json.loads(review[0].payload)["reason"] == "source.input_over_limit"
    assert harness.ledger.get_raw_event(source.identity).payload == raw.encode()


@pytest.mark.parametrize("date_key", ["speaker_date", "recorded_at", "recordedAt"])
def test_envelope_dates_support_chinese_memory_dates(date_key):
    envelope = {"payload": {"text": "小区的黑白边牧九岁多了。"}}
    envelope[date_key] = "2026-10-19" if date_key == "speaker_date" else "2026-10-19T12:33:21+00:00"
    original = json.dumps(envelope, ensure_ascii=False)
    assert validate_derivative("2026年10月19日，小区黑白边牧9岁多了。", [original]) == []
    assert any(issue.startswith("unsupported_number:") for issue in validate_derivative(
        "2026年10月20日，小区黑白边牧9岁多了。", [original]
    ))


def test_regular_words_do_not_mint_animals():
    assert validate_derivative("马上改代码，不吹牛，也不当猪队友。", ["准备继续修复。"]) == []


class _ReadyEmbedding:
    enabled = True

    def __init__(self):
        self.stored = set()

    async def generate_and_store(self, bucket_id, content):
        self.stored.add(bucket_id)
        return True

    async def get_embedding(self, bucket_id):
        return [1.0] if bucket_id in self.stored else None


def _real_write_case(bucket_mgr):
    from pathlib import Path

    base = Path(bucket_mgr.base_dir)
    ledger = LMC5Ledger(base / ".lmc5" / "pipeline.sqlite3", maintenance_root=base)
    _seed_dog_context(ledger)
    coordinator = object.__new__(NightRunCoordinator)
    coordinator.ledger = ledger
    coordinator.bucket_manager = bucket_mgr
    coordinator.policy = NightRunPolicy()
    coordinator.curated = CuratedWriteCoordinator(bucket_mgr, _ReadyEmbedding())
    pending = ledger.list_pending_proposer_chunks()[0]
    specs = coordinator._candidate_specs(
        run_id="real-write", pending=pending, batch=_batch_for(pending)
    )
    payload = next(json.loads(encoded) for _, axis, encoded in specs if axis == "X")
    return coordinator, pending, payload


@pytest.mark.asyncio
async def test_new_bucket_has_compound_source_ref_in_first_atomic_create(bucket_mgr, monkeypatch):
    coordinator, _, payload = _real_write_case(bucket_mgr)
    original_create = bucket_mgr.create
    first_metadata = []

    async def capture_create(*args, **kwargs):
        bucket_id = await original_create(*args, **kwargs)
        first_metadata.append(deepcopy((await bucket_mgr.get(bucket_id))["metadata"]))
        return bucket_id

    monkeypatch.setattr(bucket_mgr, "create", capture_create)
    result = await coordinator._write_x_payload(payload)
    assert result.success and len(first_metadata) == 1
    pointer = payload["source"]["source_ref"]
    assert first_metadata[0]["source_ref"] == pointer
    assert first_metadata[0]["type"] == "archived"  # already present before promotion
    metadata = (await bucket_mgr.get(result.bucket_id))["metadata"]
    assert metadata["source_ref"] == pointer
    assert metadata["source_session"] == PRIMARY_ID
    assert metadata["source_event_ids"] == [PRIMARY_ID]
    assert [(row["session_id"], row["source_event_id"]) for row in pointer["raw_events"]] == [
        (PRIMARY_ID, PRIMARY_ID), (PARENT_SESSION, PARENT_ID),
    ]
    for row in pointer["raw_events"]:
        resolved = coordinator.ledger.get_raw_event(EventIdentity(row["session_id"], row["source_event_id"]))
        assert resolved.row_id == row["row_id"]
    changed = deepcopy(pointer)
    changed["raw_events"] = changed["raw_events"][:1]
    assert not await bucket_mgr.update(result.bucket_id, source_ref=changed)


@pytest.mark.asyncio
async def test_legacy_receipt_replay_keeps_options_and_only_backfills_pointer(bucket_mgr):
    coordinator, pending, payload = _real_write_case(bucket_mgr)
    payload["source"].pop("source_ref")
    original_payload = deepcopy(payload)
    draft, source = payload["draft"], payload["source"]
    legacy_options = {
        "bucket_type": "dynamic", "importance": draft["importance"], "name": draft["title"],
        "tags": ["lmc5", "night", draft["type"]],
        "x_provenance": {
            "source_kind": "conversation", "source_session": source["session_id"],
            "source_event_ids": source["source_event_ids"], "source_digest": source["source_digest"],
        },
    }
    first = await coordinator.curated.write(
        idempotency_key=payload["x_write_key"], content=draft["content"],
        vector_policy="required", bucket_options=legacy_options, actor="lmc5:night",
    )
    assert first.success
    before = deepcopy((await bucket_mgr.get(first.bucket_id))["metadata"])
    assert "source_ref" not in before

    def receipt():
        with sqlite3.connect(coordinator.curated.ledger_path) as connection:
            return connection.execute(
                "SELECT payload_sha256, status, bucket_id, result_json FROM curated_writes WHERE idempotency_key = ?",
                (payload["x_write_key"],),
            ).fetchone()

    original_receipt = receipt()
    coordinator.ledger.record_candidate(
        "legacy-candidate", "X", json.dumps(payload, ensure_ascii=False), (pending.chunk_id,)
    )
    record = coordinator.ledger.list_candidates("pending")[0]
    counts = {}
    await coordinator._dispatch_x(record, payload, counts)
    assert counts["x_ready"] == 1
    assert payload == original_payload  # no inferred source_ref inserted into old options
    assert receipt() == original_receipt
    after_bucket = await bucket_mgr.get(first.bucket_id)
    after = after_bucket["metadata"]
    assert after_bucket["content"] == draft["content"]
    assert after["source_session"] == before["source_session"] == PRIMARY_ID
    assert after["source_event_ids"] == before["source_event_ids"] == [PRIMARY_ID]
    assert after["curated_payload_sha256"] == before["curated_payload_sha256"]
    assert [(row["session_id"], row["source_event_id"]) for row in after["source_ref"]["raw_events"]] == [
        (PRIMARY_ID, PRIMARY_ID), (PARENT_SESSION, PARENT_ID),
    ]
    # The unchanged writer receipt also remains replayable after the pointer-only update.
    replay = await coordinator._write_x_payload(payload)
    assert replay.bucket_id == first.bucket_id and replay.success
    assert receipt() == original_receipt
    buckets = await bucket_mgr.list_all(include_archive=True)
    assert [bucket["id"] for bucket in buckets] == [first.bucket_id]


@pytest.mark.parametrize("mutation", ["database", "table", "unknown", "duplicate", "row_id", "session"])
def test_source_ref_contract_rejects_invalid_pointers(mutation):
    pointer = {
        "database": "/data/.lmc5/pipeline.sqlite3", "table": "raw_events",
        "raw_events": [{"row_id": 1, "session_id": "session", "source_event_id": "event"}],
    }
    if mutation == "database":
        pointer["database"] = "relative.db"
    elif mutation == "table":
        pointer["table"] = "messages"
    elif mutation == "unknown":
        pointer["raw_events"][0]["extra"] = "unrecognized"
    elif mutation == "duplicate":
        pointer["raw_events"] *= 2
    elif mutation == "row_id":
        pointer["raw_events"][0]["row_id"] = True
    else:
        pointer["raw_events"][0]["session_id"] = "unknown"
    with pytest.raises(ValueError):
        normalize_source_ref(pointer)
    with pytest.raises(ValueError):
        normalize_x_provenance({"source_kind": "conversation", "source_ref": pointer})


@pytest.mark.asyncio
async def test_summary_only_source_marks_review_without_paid_repair_retry():
    calls = []
    def provider(prompt):
        calls.append(prompt)
        candidate = asdict(_batch_for(SimpleNamespace(chunk_id="summary-source")).candidates[0])
        return {"choices": [{"message": {"content": json.dumps({
            "schema_version": 1, "candidates": [candidate],
        }, ensure_ascii=False)}}]}
    source = json.dumps({"is_summary": True, "text": DOG_SOURCE}, ensure_ascii=False)
    batch = await StrictOmbreProposer(provider).propose([ProposerChunk("summary-source", source)])
    assert len(calls) == 1
    assert batch.candidates[0].risk == "review"
