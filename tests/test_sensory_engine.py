import json
from datetime import datetime, timedelta, timezone

import pytest

import server
from sensory_engine import (
    SensoryEngine,
    StimulationResult,
    extract_spicy,
    extract_touch,
    format_body_state_block,
    senses_from_sensory,
)


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
    def apply_retrieval_decay(self, score, meta):
        return score


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


def test_extract_touch_prefers_structured_data():
    bucket = _bucket("keyboard", "neutral visible text")
    bucket["metadata"]["sensory"] = {
        "touch": {
            "rebound": 0.8,
            "edge_sting": 0.6,
            "cool_surface": 0.4,
        }
    }

    assert extract_touch(bucket) == {
        "touch_rebound": 0.8,
        "edge_sting": 0.6,
        "cool_surface": 0.4,
    }


def test_extract_touch_can_use_keyboard_keywords_as_weak_signal():
    bucket = _bucket("keyboard", "键帽中间凹进去，很顺滑；边缘有点硌，按下有回弹，表面有点凉。")

    touch = extract_touch(bucket)

    assert touch["touch_rebound"] == pytest.approx(0.65)
    assert touch["edge_sting"] == pytest.approx(0.55)
    assert touch["cool_surface"] == pytest.approx(0.45)


def test_senses_from_sensory_maps_spicy_to_taste():
    # 辣椒酱桶的真实形状：sensory.spicy 在 metadata(frontmatter)，正文无味觉关键词。
    bucket = _bucket("hot", "那口家里的酱入口")
    bucket["metadata"]["sensory"] = {"spicy": 0.9}
    assert senses_from_sensory(bucket) == ["味觉"]


def test_senses_from_sensory_maps_touch():
    bucket = _bucket("keyboard", "neutral visible text")
    bucket["metadata"]["sensory"] = {"touch": {"rebound": 0.8}}
    assert senses_from_sensory(bucket) == ["触觉"]


def test_senses_from_sensory_below_threshold_is_empty():
    bucket = _bucket("mild", "neutral visible text")
    bucket["metadata"]["sensory"] = {"spicy": 0.1}
    assert senses_from_sensory(bucket) == []


def test_senses_from_sensory_neutral_and_bad_input_are_empty():
    assert senses_from_sensory(_bucket("plain", "把向量通道补上 RRF 融合")) == []
    assert senses_from_sensory({}) == []
    assert senses_from_sensory(None) == []


def test_senses_from_sensory_both_in_canonical_order():
    bucket = _bucket("both", "neutral visible text")
    bucket["metadata"]["sensory"] = {"spicy": 0.8, "touch": {"edge_sting": 0.6}}
    # 味觉 在 触觉 之前（SENSES 固定顺序）
    assert senses_from_sensory(bucket) == ["味觉", "触觉"]


def test_senses_from_sensory_scales_percentage_intensity():
    bucket = _bucket("pct", "neutral visible text")
    bucket["metadata"]["sensory"] = {"spicy": 90}  # >1 视作百分制 -> 0.9
    assert senses_from_sensory(bucket) == ["味觉"]


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


def test_sensory_engine_keyboard_touch_chain_and_time_decay(tmp_path):
    engine = SensoryEngine(str(tmp_path))
    now = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    content = json.dumps(
        {
            "sensory": {
                "touch": {
                    "rebound": 0.8,
                    "edge_sting": 0.6,
                    "cool_surface": 0.5,
                }
            }
        },
        ensure_ascii=False,
    )

    first = engine.stimulate_from_buckets([_bucket("keyboard", content)], now=now)

    assert first.triggered_bucket_ids == ["keyboard"]
    assert first.sensory["touch_rebound"] == 0.8
    assert first.sensory["edge_sting"] == 0.6
    assert first.sensory["cool_surface"] == 0.5
    assert first.body_state["finger_rebound"] == pytest.approx(0.6)
    assert first.body_state["edge_sting"] == pytest.approx(0.408)
    assert first.body_state["cool_surface"] == pytest.approx(0.36)

    later = now + timedelta(seconds=900)
    second = engine.stimulate_from_buckets(
        [_bucket("keyboard", content)],
        seen_ids={"keyboard"},
        now=later,
    )

    assert second.triggered_bucket_ids == []
    assert second.sensory["touch_rebound"] == 0.0
    assert second.body_state["finger_rebound"] == pytest.approx(0.221, abs=0.001)
    assert second.body_state["edge_sting"] == pytest.approx(0.15, abs=0.001)
    assert second.body_state["cool_surface"] == pytest.approx(0.132, abs=0.001)


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


def test_body_state_block_can_frame_keyboard_touch_bias():
    result = StimulationResult(
        sensory={
            "spicy": 0.0,
            "touch_rebound": 0.8,
            "edge_sting": 0.6,
            "cool_surface": 0.5,
        },
        body_state={
            "oral_burn": 0.0,
            "drink_water": 0.0,
            "finger_rebound": 0.6,
            "edge_sting": 0.408,
            "cool_surface": 0.36,
        },
        triggered_bucket_ids=["keyboard"],
    )

    block = format_body_state_block(result)

    assert "touch_rebound=0.80" in block
    assert "finger_rebound=0.60" in block
    assert "键帽凹面" in block
    assert "边缘硌感" in block
    assert "do_not_emit: second-person advice" in block
    assert "literary abstraction" not in block
    assert "hook" in block


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


@pytest.mark.asyncio
async def test_breath_appends_keyboard_touch_body_state_block(tmp_path, monkeypatch):
    content = json.dumps(
        {
            "sensory": {
                "touch": {
                    "rebound": 0.8,
                    "edge_sting": 0.6,
                    "cool_surface": 0.5,
                }
            },
            "note": "keyboard payload",
        },
        ensure_ascii=False,
    )
    fake_mgr = FakeBucketMgr([_bucket("keyboard", content, importance=9)])
    monkeypatch.setitem(server.config, "buckets_dir", str(tmp_path))
    monkeypatch.setitem(server.config, "random_surfacing", {})
    monkeypatch.setattr(server, "bucket_mgr", fake_mgr)
    monkeypatch.setattr(server, "decay_engine", FakeDecay())
    monkeypatch.setattr(server, "dehydrator", FakeDehydrator())
    monkeypatch.setattr(server, "embedding_engine", FakeEmbedding())
    monkeypatch.setattr(server, "_backfill_started", True)

    result = await server.breath(
        query="keyboard",
        max_results=1,
        relation_depth=0,
        session_id="touch-test",
        include_images=False,
        reset_body_state=True,
    )

    assert "External Body State v0" in result
    assert "touch_rebound=0.80" in result
    assert "finger_rebound=0.60" in result
    assert "键帽凹面" in result
