"""Fresh-window briefing must not present stale facts as current context."""

from datetime import datetime
import json

import pytest

import server


def _bucket(
    bucket_id: str,
    content: str,
    *,
    domain: list[str] | None = None,
    pinned: bool = True,
    **metadata,
) -> dict:
    event_at = metadata.pop("event_at", datetime.now(server._BJ_TZ).isoformat())
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "name": bucket_id,
            "type": "dynamic",
            "domain": domain or ["日常陪伴"],
            "tags": [],
            "importance": 10,
            "valence": 0.5,
            "arousal": 0.3,
            "pinned": pinned,
            "protected": False,
            "resolved": False,
            "event_at": event_at,
            "created": event_at,
            "last_active": event_at,
            **metadata,
        },
    }


class _FakeDecay:
    is_running = True

    async def ensure_started(self):
        return None

    def calculate_score(self, metadata):
        return float(metadata.get("importance", 5))


class _FakeManager:
    def __init__(self, buckets):
        self.buckets = list(buckets)

    async def list_all(self, include_archive=False):
        return list(self.buckets)


class _FakeValidityStore:
    def __init__(self, markers=None):
        self.markers = dict(markers or {})
        self.attach_calls = 0

    def attach(self, buckets):
        self.attach_calls += 1
        attached = list(buckets)
        for bucket in attached:
            marker = self.markers.get(bucket["id"])
            if marker:
                bucket["metadata"].update(marker)
        return attached


def _wire(monkeypatch, tmp_path, buckets, *, store=None):
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "current_world", "")
    monkeypatch.setitem(server.config, "briefing", {
        "generated_enabled": False,
        "unaudited_status_max_age_days": 1,
    })
    monkeypatch.setitem(server.config, "status_validity", {"enabled": True})
    monkeypatch.setattr(server, "bucket_mgr", _FakeManager(buckets))
    monkeypatch.setattr(server, "decay_engine", _FakeDecay())
    monkeypatch.setattr(server, "_backfill_started", True)
    if store is not None:
        monkeypatch.setattr(
            server,
            "_get_operational_status_validity_store",
            lambda: store,
        )


async def _briefing_text():
    raw = await server.briefing(
        max_chars=1500,
        session_id="fresh-window-currentness",
        include_body_state=False,
        format="json",
        deterministic=True,
    )
    return json.dumps(json.loads(raw), ensure_ascii=False)


@pytest.mark.asyncio
async def test_historical_and_superseded_facts_never_enter_boot_pack(
    monkeypatch, tmp_path,
):
    buckets = [
        _bucket(
            "old_ui_fact",
            "旧配色是纯白。",
            fact_key="ui.palette",
            fact_status="historical",
            superseded_by_bucket_id="current_ui_fact",
        ),
        _bucket(
            "current_ui_fact",
            "当前配色是墨绿和浅绿。",
            fact_key="ui.palette",
            fact_status="current",
            supersedes_bucket_ids=["old_ui_fact"],
        ),
    ]
    _wire(monkeypatch, tmp_path, buckets, store=_FakeValidityStore())

    text = await _briefing_text()

    assert "current_ui_fact" in text
    assert "old_ui_fact" not in text


@pytest.mark.asyncio
async def test_operational_status_requires_current_or_fresh_unresolved_evidence(
    monkeypatch, tmp_path,
):
    buckets = [
        _bucket(
            "stale_unverified_status",
            "任务已经完成并部署。",
            domain=["工程"],
            event_at="2026-07-01T12:00:00+08:00",
        ),
        _bucket(
            "fresh_unresolved_status",
            "这项工程任务仍未完成。",
            domain=["工程"],
        ),
        _bucket("audited_current", "当前生产版本已经部署。", domain=["工程"]),
        _bucket("audited_historical", "旧版本曾经部署。", domain=["工程"]),
    ]
    store = _FakeValidityStore({
        "audited_current": {
            "validity_kind": "operational_status",
            "validity_state": "current",
            "status_key": "deploy.main",
        },
        "audited_historical": {
            "validity_kind": "operational_status",
            "validity_state": "historical",
            "status_key": "deploy.main",
            "validity_superseded_by_bucket_id": "audited_current",
        },
    })
    _wire(monkeypatch, tmp_path, buckets, store=store)

    text = await _briefing_text()

    assert store.attach_calls == 1
    assert "audited_current" in text
    assert "fresh_unresolved_status" in text
    assert "stale_unverified_status" not in text
    assert "audited_historical" not in text


@pytest.mark.asyncio
async def test_old_narrative_memory_is_dated_not_misclassified_as_stale(
    monkeypatch, tmp_path,
):
    old_story = _bucket(
        "old_relationship_story",
        "这是已经发生过、仍有关系意义的共同经历。",
        domain=["日常陪伴"],
        event_at="2026-05-01T12:00:00+08:00",
    )
    _wire(monkeypatch, tmp_path, [old_story], store=_FakeValidityStore())

    text = await _briefing_text()

    assert "old_relationship_story" in text
    assert "2026-05-01" in text
