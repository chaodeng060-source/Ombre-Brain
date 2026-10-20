import json
from datetime import datetime, timezone

import pytest

from tests.test_night_run_coordinator import _harness
from lmc5_ledger import EventIdentity
from night_run_coordinator import _memory_payload


REFUSAL = "I cannot generate explicit sexual content."


@pytest.mark.asyncio
async def test_omitted_source_closes_all_chunks_without_a_proposer_call(tmp_path):
    h = _harness(tmp_path)
    h.ledger.append_raw_event("kiro-window", "refusal-event", json.dumps({
        "sender": "claude", "text": REFUSAL,
        "resultMeta": {"memory_backend": "kiro"},
        "thinking": "ignored " * 2000,
    }))
    result = await h.coordinator.run(run_id="omit-kiro", cutoff=datetime.now(timezone.utc))
    assert result.run.stage == "complete"
    assert result.counts["memory_omitted_events"] == 1
    assert result.counts["proposer_model_calls"] == 0
    assert result.counts["proposer_pending_after"] == 0
    assert h.provider.prompts == []
    assert h.curated.calls == []
    raw = h.ledger.get_raw_event(EventIdentity("kiro-window", "refusal-event"))
    assert json.loads(raw.payload)["text"] == REFUSAL
    again = await h.coordinator.run(run_id="omit-kiro-again", cutoff=datetime.now(timezone.utc))
    assert again.counts["proposer_pending_after"] == 0
    assert h.provider.prompts == []


@pytest.mark.asyncio
async def test_cached_candidate_cannot_republish_an_omitted_source(tmp_path):
    h = _harness(tmp_path)
    h.ledger.append_raw_event("kiro-window", "old-refusal", json.dumps({
        "sender": "claude", "text": REFUSAL, "memory_backend": "kiro",
    }))
    h.ledger.record_event_chunk("old-chunk", "previous proposer input", [("kiro-window", "old-refusal")])
    payload = {
        "draft": {"risk": "normal", "type": "event", "title": "Old model boundary", "content": "The model cannot write explicit sexual content.", "importance": 5, "relation_hints": []},
        "source": {"session_id": "kiro-window", "source_event_ids": ["old-refusal"], "source_digest": "a" * 64},
        "x_write_key": "old-refusal-write",
    }
    h.ledger.record_candidate("old-candidate", "X", json.dumps(payload), ["old-chunk"])
    row = h.ledger.list_candidates("pending")[0]
    counts = {}
    await h.coordinator._dispatch_x(row, payload, counts)
    assert h.curated.calls == []
    assert h.ledger.list_candidates("pending") == ()
    assert counts["x_deferred"] == 1
    assert h.ledger.list_candidates("deferred")[0].error_code == "memory.owner_omitted"


@pytest.mark.parametrize("raw", [
    {"sender": "user", "memory_backend": "kiro", "text": REFUSAL},
    {"sender": "claude", "text": REFUSAL},
    {"sender": "claude", "memory_backend": "kiro", "text": "Kiro shell permission denied."},
    {"sender": "claude", "memory_backend": "kiro", "text": "朝灯不想亲密，哥哥尊重她。"},
])
def test_human_unknown_backend_and_unrelated_errors_remain(raw):
    assert _memory_payload(raw) == raw
