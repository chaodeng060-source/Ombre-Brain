"""Night-run proposer input no longer includes thinking/tools/resultMeta.

2026-09-15: LMC-5 night-run backlog stats showed one assistant reply is
routinely chunked into several ledger rows, with most of those chunks
landing entirely inside the `thinking`/`tools`/`resultMeta` payload keys the
proposer never needed -- it only ever asks the model to read the redacted
transcript `text`.  ``night_run_coordinator._propose_pending`` now groups
pending chunks by their source event, rebuilds one slim model input straight
from ``raw_events`` for the whole group, and closes every sibling chunk of a
successfully-resolved event in the same ledger transaction -- for both
freshly cut ("slim") chunks and chunks left over from before this change
("fat", still holding thinking/tools/resultMeta on disk).

These tests call ``coordinator._propose_pending`` directly against a hand
seeded ledger (same pattern as
``test_night_run_coordinator.test_report_only_receipt_flush_yields_during_large_backlog``),
bypassing ``_chunk_uncovered`` so a single raw event can be given several
pre-existing ("legacy fat") chunks the way the real backlog has them.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from lmc5_ledger import EventIdentity, LMC5Ledger, RawEventRecord
from night_run_coordinator import NightRunPolicy
from redact import redact_obj
from tests.test_night_run_coordinator import _harness


def _seed_legacy_fat_event(
    harness,
    *,
    session_id: str,
    source_event_id: str,
    payload: dict,
    chunk_count: int,
    chunk_prefix: str,
) -> None:
    """Append one raw event and give it several pre-existing chunk rows.

    Mirrors how a real backlog looks: chunks recorded before this slimming
    change still hold the full (fat) redacted payload on disk.  The exact
    on-disk chunk content is irrelevant to the new proposer path -- it is
    only used for content-digest bookkeeping -- so a small placeholder body
    is enough to exercise the grouped-by-event logic.
    """

    harness.ledger.append_raw_event(
        session_id,
        source_event_id,
        json.dumps(payload, ensure_ascii=False),
    )
    for ordinal in range(chunk_count):
        harness.ledger.record_event_chunk(
            f"{chunk_prefix}-{ordinal}",
            f"legacy fat chunk body {ordinal}",
            [(session_id, source_event_id)],
        )


def _prompt_envelope(prompt: str) -> dict:
    raw_input = json.loads(prompt.split("INPUT=", 1)[1])
    chunk = raw_input["chunks"][0]
    return json.loads(chunk["text"])


@pytest.mark.asyncio
async def test_legacy_multi_chunk_event_calls_model_once_without_slim_keys(
    tmp_path,
) -> None:
    harness = _harness(tmp_path)
    _seed_legacy_fat_event(
        harness,
        session_id="room-main",
        source_event_id="assistant-reply-1",
        payload={
            "text": "朝灯今晚也想看星星呀",
            "thinking": "要不要主动一点" * 200,
            "tools": [{"name": "grow", "args": {"note": "记一下"}}],
            "resultMeta": {"latency_ms": 999, "tokens": 12345},
            "sender": "assistant",
            "eventType": "assistant_reply",
        },
        chunk_count=3,
        chunk_prefix="fat-a",
    )
    counts: dict[str, int] = {}
    watermark = harness.ledger.proposer_watermark()

    await harness.coordinator._propose_pending(
        "night-slim-a", counts, watermark=watermark
    )

    assert len(harness.provider.prompts) == 1
    envelope = _prompt_envelope(harness.provider.prompts[0])
    assert set(envelope["payload"]) & {"thinking", "tools", "resultMeta"} == set()
    assert envelope["payload"]["text"] == "朝灯今晚也想看星星呀"

    assert counts["proposer_events"] == 1
    assert counts["proposer_model_calls"] == 1
    assert counts["proposer_no_text_events"] == 0
    assert counts["proposer_succeeded"] == 3
    assert counts["proposer_attempted"] == 3
    assert counts["proposer_retryable"] == 0
    assert counts["proposer_sibling_chunks_closed"] == 2
    assert counts["proposer_chunks"] == 3
    assert counts["candidates"] == 2  # one "event" draft routes to axes X+M
    assert counts["proposer_pending_before"] == 3
    assert counts["proposer_pending_after"] == 0
    assert harness.ledger.list_pending_proposer_chunks(limit=10) == ()


@pytest.mark.asyncio
async def test_event_with_no_spoken_text_still_calls_model_once(tmp_path) -> None:
    """2026-09-15 correction: a no-text event is no longer a free close.

    An earlier version of this change skipped the model call whenever the
    slimmed payload had no non-blank top-level ``text`` (production only
    has ~26 such events out of ~9000 -- not much saved), but ``text`` is
    proposer shorthand, not a universal event contract; short-circuiting
    on it silently dropped other event shapes (like this legacy
    ``{"message": ...}``-only payload) from proposer coverage instead of
    just costing one extra call.  Every event now spends exactly one real
    model call; ``proposer_no_text_events`` is kept purely as an
    observation counter and never skips the call.
    """
    harness = _harness(tmp_path, empty_provider=True)
    _seed_legacy_fat_event(
        harness,
        session_id="room-main",
        source_event_id="assistant-notext-1",
        payload={
            "thinking": "只在心里盘算，没有说出口" * 100,
            "tools": [{"name": "grow", "args": {}}],
            "sender": "assistant",
            "eventType": "assistant_reply",
        },
        chunk_count=2,
        chunk_prefix="notext-b",
    )
    counts: dict[str, int] = {}
    watermark = harness.ledger.proposer_watermark()

    await harness.coordinator._propose_pending(
        "night-slim-b", counts, watermark=watermark
    )

    assert len(harness.provider.prompts) == 1
    envelope = _prompt_envelope(harness.provider.prompts[0])
    assert set(envelope["payload"]) & {"thinking", "tools", "resultMeta"} == set()
    assert "text" not in envelope["payload"]

    assert counts["proposer_events"] == 1
    assert counts["proposer_model_calls"] == 1
    assert counts["proposer_no_text_events"] == 1
    assert counts["proposer_succeeded"] == 2
    assert counts["proposer_attempted"] == 2
    assert counts["proposer_sibling_chunks_closed"] == 1
    assert counts["candidates"] == 0
    assert harness.ledger.list_pending_proposer_chunks(limit=10) == ()


@pytest.mark.asyncio
async def test_representative_failure_keeps_siblings_pending_then_closes_them(
    tmp_path,
) -> None:
    harness = _harness(tmp_path, invalid_provider=True)
    _seed_legacy_fat_event(
        harness,
        session_id="room-main",
        source_event_id="assistant-retry-1",
        payload={"text": "先失败一次再成功的话", "thinking": "x" * 50},
        chunk_count=3,
        chunk_prefix="retry-c",
    )
    watermark = harness.ledger.proposer_watermark()

    counts_fail: dict[str, int] = {}
    await harness.coordinator._propose_pending(
        "night-slim-c1", counts_fail, watermark=watermark
    )

    assert counts_fail["proposer_retryable"] == 1
    assert counts_fail["proposer_succeeded"] == 0
    assert counts_fail["proposer_sibling_chunks_closed"] == 0
    still_pending = harness.ledger.list_pending_proposer_chunks(limit=10)
    assert {row.chunk_id for row in still_pending} == {
        "retry-c-0",
        "retry-c-1",
        "retry-c-2",
    }
    assert all(row.retry_count == 0 for row in still_pending if row.chunk_id != "retry-c-0")

    harness.provider.invalid = False
    counts_ok: dict[str, int] = {}
    await harness.coordinator._propose_pending(
        "night-slim-c2", counts_ok, watermark=harness.ledger.proposer_watermark()
    )

    assert counts_ok["proposer_model_calls"] == 1
    assert counts_ok["proposer_succeeded"] == 3
    assert counts_ok["proposer_sibling_chunks_closed"] == 2
    assert harness.ledger.list_pending_proposer_chunks(limit=10) == ()


@pytest.mark.asyncio
async def test_quarantined_event_is_skipped_entirely_not_retried(tmp_path) -> None:
    harness = _harness(tmp_path)
    _seed_legacy_fat_event(
        harness,
        session_id="room-main",
        source_event_id="assistant-quarantine-1",
        payload={"text": "被隔离前的最后一句话", "thinking": "t"},
        chunk_count=2,
        chunk_prefix="quarantine-d",
    )
    representative_chunk_id = "quarantine-d-0"
    for attempt in range(3):
        harness.ledger.record_chunk_proposer_outcome(
            f"quarantine-d-retry-{attempt}",
            representative_chunk_id,
            "retryable_error",
            error_code="provider.timeout",
        )
    counts: dict[str, int] = {}
    watermark = harness.ledger.proposer_watermark()

    await harness.coordinator._propose_pending(
        "night-slim-d", counts, watermark=watermark
    )

    assert harness.provider.prompts == []
    assert counts["proposer_events"] == 0
    assert counts["proposer_model_calls"] == 0
    assert counts["proposer_succeeded"] == 0
    assert counts["proposer_retryable"] == 0
    assert counts["proposer_attempted"] == 0
    pending = harness.ledger.list_pending_proposer_chunks(limit=10)
    assert {row.chunk_id for row in pending} == {
        "quarantine-d-0",
        "quarantine-d-1",
    }
    assert counts["proposer_quarantined"] == 1


@pytest.mark.asyncio
async def test_run_budget_counts_every_discovered_event_including_no_text(
    tmp_path,
) -> None:
    """A no-text event now costs one call too, so it counts against the cap.

    Discovery order follows chunk row id: notext-e, then text-a, text-b,
    text-c.  With a cap of 2 model calls, "notext-e" and "text-a" are
    discovered and resolved; "text-b"/"text-c" are never even looked at
    this run and stay pending untouched.
    """
    harness = _harness(
        tmp_path, policy=NightRunPolicy(proposer_max_chunks_per_run=2)
    )
    harness.ledger.append_raw_event(
        "room-main",
        "notext-e",
        json.dumps({"thinking": "没开口"}, ensure_ascii=False),
    )
    harness.ledger.record_event_chunk(
        "notext-e-chunk", "body", [("room-main", "notext-e")]
    )
    for label in ("a", "b", "c"):
        harness.ledger.append_raw_event(
            "room-main",
            f"text-{label}",
            json.dumps({"text": f"消息{label}"}, ensure_ascii=False),
        )
        harness.ledger.record_event_chunk(
            f"text-{label}-chunk", "body", [("room-main", f"text-{label}")]
        )
    counts: dict[str, int] = {}
    watermark = harness.ledger.proposer_watermark()

    await harness.coordinator._propose_pending(
        "night-slim-e", counts, watermark=watermark
    )

    assert len(harness.provider.prompts) == 2
    assert counts["proposer_model_calls"] == 2
    assert counts["proposer_events"] == 2  # notext + a; b/c never discovered
    assert counts["proposer_no_text_events"] == 1
    assert counts["proposer_succeeded"] == 2
    assert counts["proposer_attempted"] == 2
    pending = harness.ledger.list_pending_proposer_chunks(limit=10)
    assert {row.chunk_id for row in pending} == {"text-b-chunk", "text-c-chunk"}


def test_freshly_cut_chunk_drops_slim_keys_and_gets_a_new_chunk_id(
    tmp_path,
) -> None:
    harness = _harness(tmp_path)
    payload = {
        "text": "新切块应该只留这句话",
        "thinking": "不该出现的内心戏",
        "tools": [{"name": "grow", "args": {}}],
        "resultMeta": {"latency_ms": 1},
        "sender": "assistant",
    }
    raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    record = RawEventRecord(
        row_id=1,
        identity=EventIdentity("room-main", "schema-bump-1"),
        payload=raw_bytes,
        payload_digest=hashlib.sha256(raw_bytes).hexdigest(),
        recorded_at="2026-09-15T00:00:00.000000+00:00",
    )

    parts = harness.coordinator._event_parts(record)

    assert len(parts) == 1
    chunk_id, content = parts[0]
    envelope = json.loads(content.decode("utf-8"))
    assert set(envelope["payload"]) & {"thinking", "tools", "resultMeta"} == set()
    assert envelope["payload"]["text"] == "新切块应该只留这句话"
    assert envelope["schema"] == "ombre.lmc5-redacted-event/v2"

    # Replay the pre-slim (v1, un-stripped) envelope/id algorithm exactly to
    # prove the new chunk id cannot collide with an old "fat" chunk already
    # sitting in the ledger for the same raw event.
    old_envelope = {
        "payload": redact_obj(json.loads(raw_bytes)),
        "recorded_at": record.recorded_at,
        "redaction": "redact_obj/v1",
        "schema": "ombre.lmc5-redacted-event/v1",
        "session_id": record.identity.session_id,
        "source_event_id": record.identity.source_event_id,
    }
    old_canonical = json.dumps(
        old_envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    old_identity = {
        "content_sha256": hashlib.sha256(old_canonical).hexdigest(),
        "ordinal": 0,
        "payload_sha256": record.payload_digest,
        "redaction": "redact_obj/v1",
        "session_id": record.identity.session_id,
        "source_event_id": record.identity.source_event_id,
    }
    old_chunk_id = "lmc5-chunk-v1-" + hashlib.sha256(
        json.dumps(
            old_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert chunk_id != old_chunk_id


@pytest.mark.asyncio
async def test_grouped_event_lookup_skips_full_outcomes_table_scan_per_event(
    tmp_path, monkeypatch
) -> None:
    """Regression guard for the 2026-09-15 lock-duration fix.

    ``list_pending_proposer_chunks_for_event`` used to re-run
    ``_verify_proposer_outcomes`` (a full scan of the
    ``chunk_proposer_outcomes`` table) once per distinct event discovered
    in a run.  With the real backlog collecting ~256 events per run, that
    turned a handful of page-level full scans into hundreds, all while
    holding the exclusive maintenance lease.  Ten single-chunk events here
    should still only trigger the page-level scans (before/after backlog
    stats plus the two ``list_pending_proposer_chunks`` page fetches), not
    one extra scan per event.
    """
    harness = _harness(tmp_path)
    for index in range(10):
        harness.ledger.append_raw_event(
            "room-main",
            f"verify-guard-{index}",
            json.dumps({"text": f"消息{index}"}, ensure_ascii=False),
        )
        harness.ledger.record_event_chunk(
            f"verify-guard-{index}-chunk",
            "body",
            [("room-main", f"verify-guard-{index}")],
        )

    calls = {"count": 0}
    original_verify = LMC5Ledger._verify_proposer_outcomes

    def counting_verify(connection):
        calls["count"] += 1
        return original_verify(connection)

    monkeypatch.setattr(
        LMC5Ledger,
        "_verify_proposer_outcomes",
        staticmethod(counting_verify),
    )

    counts: dict[str, int] = {}
    watermark = harness.ledger.proposer_watermark()

    await harness.coordinator._propose_pending(
        "night-slim-verify-guard", counts, watermark=watermark
    )

    assert counts["proposer_events"] == 10
    assert counts["proposer_model_calls"] == 10
    assert calls["count"] < 10
