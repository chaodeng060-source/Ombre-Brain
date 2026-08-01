from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from e_axis_trigger import (
    EAxisSourceError,
    decide_e_axis_trigger,
    iter_candidate_subjects,
)


@pytest.mark.parametrize(
    ("memory_type", "content", "hints", "included", "reason"),
    [
        ("preference", "纯技术字样", [], True, "type.preference"),
        (
            "relationship_moment",
            "没有关键词",
            [],
            True,
            "type.relationship_moment",
        ),
        ("risk_boundary", "没有关键词", [], True, "type.risk_boundary"),
        (
            "fact",
            "服务监听 8000 端口",
            [],
            False,
            "type.fact.no_emotion",
        ),
        (
            "engineering_decision",
            "这个事故让我很焦虑",
            [],
            True,
            "keyword.emotion",
        ),
        (
            "event",
            "普通叙述",
            [{"relation_type": "emotional_link"}],
            True,
            "relation.emotional_link",
        ),
        ("event", "普通叙述", [], False, "gate.no_signal"),
    ],
)
def test_official_trigger_matrix(
    memory_type,
    content,
    hints,
    included,
    reason,
):
    decision = decide_e_axis_trigger(
        memory_type=memory_type,
        title="标题",
        content=content,
        relation_hints=hints,
    )
    assert decision.included is included
    assert decision.reason == reason


def _record(axis: str, candidate_id: int, *, malformed=False):
    base_digest = hashlib.sha256(b"same-draft").hexdigest()
    payload = {
        "axis": axis,
        "base_digest": base_digest,
        "draft": {
            "type": "preference",
            "title": "偏好",
            "content": "我更喜欢清楚的回答。",
            "relation_hints": [],
        },
        "origin_run_id": "lmc5-night-1",
        "schema": "ombre.lmc5-axis-candidate/v1",
        "source": {"created_at": "2026-07-30T00:00:00+00:00"},
    }
    raw = b"not-json" if malformed else json.dumps(payload).encode()
    return SimpleNamespace(
        candidate_id=candidate_id,
        axis=axis,
        payload=raw,
    )


class _Ledger:
    def __init__(self, rows):
        self.rows = rows

    def list_candidates(self, status, *, limit, after=None):
        if status != "pending":
            return ()
        after = after or 0
        return tuple(
            row for row in self.rows if row.candidate_id > after
        )[:limit]


def test_producer_reads_only_x_copy_and_deduplicates_axis_copies():
    scan = iter_candidate_subjects(
        _Ledger([
            _record("X", 1),
            _record("E", 2),
            _record("M", 3),
        ])
    )
    assert scan.scanned == 1
    assert scan.skipped == 0
    assert scan.skip_reason_counts() == {}
    assert len(scan.subjects) == 1
    assert scan.subjects[0].source_kind == "lmc5_candidate"
    assert scan.subjects[0].memory_type == "preference"


def test_producer_reports_explicit_skip_reason_distribution():
    record = _record("X", 1)
    payload = json.loads(record.payload)
    payload["draft"].update({
        "type": "fact",
        "title": "端口",
        "content": "服务监听 8000 端口。",
    })
    record.payload = json.dumps(payload).encode()

    scan = iter_candidate_subjects(_Ledger([record]))

    assert scan.subjects == ()
    assert scan.skipped == 1
    assert scan.skip_reason_counts() == {"type.fact.no_emotion": 1}


def test_bad_persisted_x_payload_fails_closed():
    with pytest.raises(EAxisSourceError, match="candidate.invalid_json"):
        iter_candidate_subjects(_Ledger([_record("X", 1, malformed=True)]))


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("origin_run_id", "bad run", "candidate.invalid_source_run"),
        ("created_at", "not-a-date", "candidate.invalid_created_at"),
        ("created_at", "2026-07-31T00:00:00", "candidate.invalid_created_at"),
    ],
)
def test_bad_candidate_lineage_fails_at_source_boundary(field, value, code):
    record = _record("X", 1)
    payload = json.loads(record.payload)
    if field == "created_at":
        payload["source"][field] = value
    else:
        payload[field] = value
    record.payload = json.dumps(payload).encode()

    with pytest.raises(EAxisSourceError, match=code):
        iter_candidate_subjects(_Ledger([record]))
