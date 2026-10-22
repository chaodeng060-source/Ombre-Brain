import json
from datetime import datetime, timezone

import pytest

from lmc5_ledger import EventIdentity
from night_run_coordinator import _memory_payload
from tests.test_night_run_coordinator import _harness


@pytest.mark.parametrize("metadata", [
    {"sender": "system"}, {"purpose": "compaction_letter"},
    {"automation": "task_stall_nudge"}, {"memoryOrigin": "auxiliary"},
    {"isMeta": True}, {"parentMessageId": "ntf_notice"},
    {"payload": {"purpose": "model_probe"}},
])
@pytest.mark.asyncio
async def test_auxiliary_raw_event_is_retained_but_never_proposed(tmp_path, metadata):
    h = _harness(tmp_path)
    payload = {"sender": "user", "text": "系统自动续接摘要，不是朝灯的承诺。", **metadata}
    original = json.dumps(payload, ensure_ascii=False)
    h.ledger.append_raw_event("old-window", "notice", original)
    result = await h.coordinator.run(run_id="origin-first", cutoff=datetime.now(timezone.utc))
    assert result.run.stage == "complete"
    assert result.counts["memory_omitted_events"] == 1
    assert result.counts["proposer_model_calls"] == 0
    assert result.counts["proposer_pending_after"] == 0
    assert h.curated.calls == []
    assert h.ledger.get_raw_event(EventIdentity("old-window", "notice")).payload == original.encode()
    await h.coordinator.run(run_id="origin-second", cutoff=datetime.now(timezone.utc))
    assert h.provider.prompts == []


def test_real_human_discussion_of_audit_and_promises_is_preserved():
    raw = {"sender": "user", "text": "整理备注不要混到记忆正文里，之前的承诺保留原话。"}
    assert _memory_payload(raw) is raw


@pytest.mark.asyncio
async def test_cached_auxiliary_draft_is_not_republished(tmp_path):
    h = _harness(tmp_path)
    h.ledger.append_raw_event("old-window", "notice", json.dumps({
        "sender": "claude", "text": "系统摘要里的承诺。", "purpose": "continuity_capsule",
    }))
    assert h.coordinator._memory_x_content("系统摘要里的承诺。", "old-window", ["notice"]) == ""
