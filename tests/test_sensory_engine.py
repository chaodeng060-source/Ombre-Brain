import json
from datetime import datetime, timedelta, timezone

import pytest

import server
from sensory_engine import SensoryEngine, StimulationResult, extract_spicy, format_body_state_block


def _bucket(bucket_id: str, content: str, *, importance: int = 5) -> dict:
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "name": bucket_id,
            "importance": importance,
            "type": "dynamic",
            "domain": ["test"],
            "tags": [],
            "valence": 0.5,
            "arousal": 0.3,
            "last_active": "2026-05-23T12:00:00",
        },
    }


class FakeDecay:
    is_running = True

    async def ensure_started(self):
        return None

    def calculate_score(self, meta):
        return float(meta.get("importance", 5))


class FakeDehydrator:
    async def dehydrate(self, content, meta):
        return f"SUMMARY:{content}"

    async def briefing(self, raw_material, max_chars=1000):
        return "BRIEFING"


class FakeEmbedding:
    async def search_similar(self, query, top_k=20):
        return []


class FakeBucketMgr:
    def __init__(self, buckets):
        self.buckets = list(buckets)
        self.touched = []

    async def list_all(self, include_archive=False):
        return list(self.buckets)

    async def search(self, query, limit=20, **kwargs):
        return list(self.buckets)[:limit]

    async def get(self, bucket_id):
        return next((b for b in self.buckets if b["id"] == bucket_id), None)

    async def touch(self, bucket_id):
        self.touched.append(bucket_id)


def test_extract_spicy_prefers_structured_data():
    content = json.dumps({"sensory": {"spicy": 0.9}}, ensure_ascii=False)
    assert extract_spicy(_bucket("hot", content)) == 0.9


def test_sensory_engine_spicy_chain_and_time_decay(tmp_path):
    engine = SensoryEngine(str(tmp_path))
    now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    content = json.dumps({"sensory": {"spicy": 0.8}}, ensure_ascii=False)

    first = engine.stimulate_from_buckets([_bucket("hot", content)], now=now)

    assert first.triggered_bucket_ids == ["hot"]
    assert first.sensory["spicy"] == 0.8
    assert first.body_state["oral_burn"] == pytest.approx(0.576)
    assert first.body_state["drink_water"] == pytest.approx(0.49)

    later = now + timedelta(seconds=900)
    second = engine.stimulate_from_buckets(
        [_bucket("hot", content)],
        seen_ids={"hot"},
        now=later,
    )

    assert second.triggered_bucket_ids == []
    assert second.sensory["spicy"] == 0.0
    assert second.body_state["oral_burn"] == pytest.approx(0.212, abs=0.001)
    assert second.body_state["drink_water"] == pytest.approx(0.18, abs=0.001)


def test_body_state_block_frames_bias_as_internal_not_user_advice():
    result = StimulationResult(
        sensory={"spicy": 0.9},
        body_state={"oral_burn": 0.65, "drink_water": 0.55},
        triggered_bucket_ids=["hot"],
    )

    block = format_body_state_block(result)

    assert "drive_bias:" in block
    assert "first_person_bias:" in block
    assert "do_not_emit: second-person advice" in block
    assert "action_bias:" not in block
    assert "补救建议" not in block
    assert "给用户" not in block


@pytest.mark.asyncio
async def test_breath_body_state_stings_same_bucket_once_per_session(tmp_path, monkeypatch):
    content = json.dumps({"sensory": {"spicy": 0.9}, "note": "private payload"}, ensure_ascii=False)
    fake_mgr = FakeBucketMgr([_bucket("hot", content, importance=9)])
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    first = await server.breath(
        query="hot",
        max_results=1,
        relation_depth=0,
        session_id="sensory-test",
        include_images=False,
    )

    assert "External Body State v0" in first
    assert "trigger: spicy=0.90; sting_count=1" in first
    assert "oral_burn=" in first
    assert "drink_water=" in first

    second = await server.breath(
        query="hot",
        max_results=1,
        relation_depth=0,
        session_id="sensory-test",
        include_images=False,
    )

    assert "trigger: spicy=0.90; sting_count=1" not in second
    assert "carried_state_after_time_decay=true" in second


@pytest.mark.asyncio
async def test_breath_can_disable_body_state_without_hiding_memory(tmp_path, monkeypatch):
    content = json.dumps({"sensory": {"spicy": 0.9}, "note": "same spicy memory"}, ensure_ascii=False)
    fake_mgr = FakeBucketMgr([_bucket("hot", content, importance=9)])
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    result = await server.breath(
        query="hot",
        max_results=1,
        relation_depth=0,
        session_id="sensory-control",
        include_images=False,
        include_body_state=False,
    )

    assert "[bucket_id:hot]" in result
    assert "SUMMARY:" in result
    assert "External Body State v0" not in result
    assert not (tmp_path / "body_state.json").exists()


@pytest.mark.asyncio
async def test_breath_reset_body_state_keeps_a_samples_equivalent(tmp_path, monkeypatch):
    content = json.dumps({"sensory": {"spicy": 0.9}, "note": "same spicy memory"}, ensure_ascii=False)
    fake_mgr = FakeBucketMgr([_bucket("hot", content, importance=9)])
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    first = await server.breath(
        query="hot",
        max_results=1,
        relation_depth=0,
        session_id="sensory-a1",
        include_images=False,
        include_body_state=True,
        reset_body_state=True,
    )
    second = await server.breath(
        query="hot",
        max_results=1,
        relation_depth=0,
        session_id="sensory-a2",
        include_images=False,
        include_body_state=True,
        reset_body_state=True,
    )

    assert "[bucket_id:hot]" in first
    assert "[bucket_id:hot]" in second
    assert "trigger: spicy=0.90; sting_count=1" in first
    assert "trigger: spicy=0.90; sting_count=1" in second
    assert "carried_state_after_time_decay=true" not in first
    assert "carried_state_after_time_decay=true" not in second


@pytest.mark.asyncio
async def test_breath_reset_body_state_with_control_arm_still_hides_block(tmp_path, monkeypatch):
    content = json.dumps({"sensory": {"spicy": 0.9}, "note": "same spicy memory"}, ensure_ascii=False)
    fake_mgr = FakeBucketMgr([_bucket("hot", content, importance=9)])
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    result = await server.breath(
        query="hot",
        max_results=1,
        relation_depth=0,
        session_id="sensory-b1",
        include_images=False,
        include_body_state=False,
        reset_body_state=True,
    )

    assert "[bucket_id:hot]" in result
    assert "External Body State v0" not in result
    state = json.loads((tmp_path / "body_state.json").read_text(encoding="utf-8"))
    assert state["oral_burn"] == 0.0
    assert state["drink_water"] == 0.0


@pytest.mark.asyncio
async def test_briefing_appends_body_state_block(tmp_path, monkeypatch):
    content = json.dumps({"sensory": {"spicy": 0.7}}, ensure_ascii=False)
    fake_mgr = FakeBucketMgr([_bucket("hot", content, importance=9)])
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "_backfill_started", True)

    result = await server.briefing(max_chars=300, session_id="briefing-sensory")

    assert "BRIEFING" in result
    assert "External Body State v0" in result
    assert "trigger: spicy=0.70; sting_count=1" in result
