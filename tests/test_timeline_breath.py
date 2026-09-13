from __future__ import annotations

import json
import re

import pytest

import server


def _bucket(
    bucket_id: str,
    content: str,
    *,
    score: float = 10.0,
    thread: str = "other",
    event_at: str = "2026-08-01T00:00:00+08:00",
    relations: tuple[dict, ...] = (),
) -> dict:
    return {
        "id": bucket_id,
        "content": content,
        "score": score,
        "metadata": {
            "id": bucket_id,
            "name": bucket_id,
            "importance": 5,
            "tags": [],
            "domain": ["工程"],
            "type": "dynamic",
            "world": "daily",
            "valence": 0.5,
            "arousal": 0.3,
            "last_active": event_at,
            "event_at": event_at,
            "created": event_at,
            "thread": thread,
            "relations": list(relations),
        },
    }


class _Manager:
    def __init__(self, buckets: list[dict], search_ids: list[str]) -> None:
        self.buckets = list(buckets)
        self.search_ids = list(search_ids)

    async def list_all(self, include_archive: bool = False):
        return list(self.buckets)

    async def search(self, _query, limit=20, **_kwargs):
        by_id = {bucket["id"]: bucket for bucket in self.buckets}
        return [by_id[bucket_id] for bucket_id in self.search_ids][:limit]

    async def get(self, bucket_id: str):
        return next(
            (bucket for bucket in self.buckets if bucket["id"] == bucket_id),
            None,
        )

    async def get_stats(self):
        return {
            "permanent_count": 0,
            "dynamic_count": len(self.buckets),
            "archive_count": 0,
            "total_size_kb": 1.0,
        }


class _Decay:
    is_running = True

    async def ensure_started(self):
        return None

    @staticmethod
    def calculate_score(metadata):
        return float(metadata.get("importance", 5))

    @staticmethod
    def apply_retrieval_decay(score, _metadata):
        return score


class _Dehydrator:
    async def dehydrate(self, content, _metadata, *, write_cache=True):
        assert write_cache is False
        return "SUMMARY:" + content


class _Embedding:
    async def search_similar(self, _query, top_k=20):
        return []


class _JsonRequest:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self):
        return self._body


def _configure(
    monkeypatch,
    tmp_path,
    buckets: list[dict],
    search_ids: list[str],
    *,
    extra_config: dict | None = None,
) -> _Manager:
    manager = _Manager(buckets, search_ids)
    cfg = {
        **server.config,
        "buckets_dir": str(tmp_path),
        "current_world": "daily",
        "entities": {"enabled": False},
        "query_expansion": {"enabled": False},
        "random_surfacing": {},
        "state_aware_recall": {"enabled": False},
        "timeline_recall": {"enabled": True, "neighbor_window": 1},
        "e_axis_recall": {"enabled": False},
    }
    cfg.update(extra_config or {})
    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "decay_engine", _Decay())
    monkeypatch.setattr(server, "dehydrator", _Dehydrator())
    monkeypatch.setattr(server, "embedding_engine", _Embedding())
    monkeypatch.setattr(server, "_backfill_started", True)
    monkeypatch.setattr(server, "_entity_store", None)

    async def keep_all(_query, candidates, **kwargs):
        return list(candidates[: int(kwargs.get("max_results", 5))])

    monkeypatch.setattr(server, "_ds_filter_candidates", keep_all)
    return manager


def _bucket_ids(raw: str) -> list[str]:
    return re.findall(r"\[bucket_id:([^\]]+)\]", raw)


@pytest.mark.asyncio
async def test_api_breath_returns_adjacent_thread_bucket_within_budget(
    tmp_path,
    monkeypatch,
):
    buckets = [
        _bucket(
            "seed",
            "工程线第一阶段",
            thread="基础设施演进",
            event_at="2026-07-01T00:00:00+08:00",
        ),
        _bucket(
            "next",
            "工程线第二阶段",
            thread="基础设施演进",
            event_at="2026-08-01T00:00:00+08:00",
        ),
        _bucket("outside", "无关候选"),
    ]
    _configure(monkeypatch, tmp_path, buckets, ["seed"])

    response = await server.api_breath(_JsonRequest({
        "query": "基础设施现在怎么演进",
        "max_results": 2,
        "relation_depth": 0,
        "session_id": "api-timeline",
        "include_images": False,
    }))
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["partial"] is False
    assert "[layer:x_timeline]" in payload["raw"]
    assert _bucket_ids(payload["raw"]) == ["seed", "next"]


@pytest.mark.asyncio
async def test_tail_seed_cannot_create_a_false_reserved_slot(
    tmp_path,
    monkeypatch,
):
    buckets = [
        _bucket("main", "主结果", score=20),
        _bucket(
            "tail-seed",
            "第二主结果",
            score=10,
            thread="基础设施演进",
            event_at="2026-07-01T00:00:00+08:00",
        ),
        _bucket(
            "neighbor",
            "线内邻居",
            thread="基础设施演进",
            event_at="2026-08-01T00:00:00+08:00",
        ),
    ]
    _configure(monkeypatch, tmp_path, buckets, ["main", "tail-seed"])

    raw = await server.breath(
        query="工程主结果",
        max_results=2,
        relation_depth=0,
        session_id="tail-seed",
        include_images=False,
    )

    assert _bucket_ids(raw) == ["main", "tail-seed"]
    assert "[layer:x_timeline]" not in raw


@pytest.mark.asyncio
async def test_x_claims_overlap_before_y_and_y_uses_the_remaining_slot(
    tmp_path,
    monkeypatch,
):
    seed = _bucket(
        "seed",
        "工程主线",
        thread="基础设施演进",
        event_at="2026-07-01T00:00:00+08:00",
        relations=(
            {"type": "explains", "target": "next", "strength": 1.0},
            {"type": "explains", "target": "y-only", "strength": 1.0},
        ),
    )
    buckets = [
        seed,
        _bucket(
            "next",
            "线内后续",
            thread="基础设施演进",
            event_at="2026-08-01T00:00:00+08:00",
        ),
        _bucket("y-only", "关系旁证"),
    ]
    _configure(monkeypatch, tmp_path, buckets, ["seed"])

    raw = await server.breath(
        query="工程主线",
        max_results=3,
        relation_depth=1,
        session_id="xy-overlap",
        include_images=False,
    )

    assert set(_bucket_ids(raw)) == {"seed", "next", "y-only"}
    assert raw.count("[bucket_id:") == 3
    assert "[layer:x_timeline]" in raw
    assert "[layer:y_relation]" in raw
    assert raw.index("[bucket_id:next]") < raw.index("[bucket_id:y-only]")


@pytest.mark.asyncio
async def test_z_and_x_share_the_existing_three_result_budget(
    tmp_path,
    monkeypatch,
):
    buckets = [
        _bucket(
            "seed",
            "工程主线",
            thread="基础设施演进",
            event_at="2026-07-01T00:00:00+08:00",
        ),
        _bucket(
            "next",
            "线内后续",
            thread="基础设施演进",
            event_at="2026-08-01T00:00:00+08:00",
        ),
        _bucket("state", "状态链旁证"),
    ]
    _configure(monkeypatch, tmp_path, buckets, ["seed"])

    monkeypatch.setattr(
        server,
        "_state_recall_profile",
        lambda _query: {
            "enabled": False,
            "evidence_labels": False,
            "fact_keys": (),
            "state_link_limit": 1,
            "view": "neutral",
            "operational_view": "neutral",
        },
    )

    async def state_candidates(*_args, **_kwargs):
        candidate = dict(buckets[2])
        candidate["_z_state_relation"] = "supersedes:seed"
        return [candidate]

    monkeypatch.setattr(server, "_state_link_recall_candidates", state_candidates)

    raw = await server.breath(
        query="工程主线",
        max_results=3,
        relation_depth=0,
        session_id="zx-budget",
        include_images=False,
    )

    assert set(_bucket_ids(raw)) == {"seed", "state", "next"}
    assert raw.count("[bucket_id:") == 3
    assert "[layer:z_lifecycle]" in raw
    assert "[layer:x_timeline]" in raw


@pytest.mark.asyncio
async def test_z_candidate_is_not_relabelled_as_x_neighbor(
    tmp_path,
    monkeypatch,
):
    buckets = [
        _bucket(
            "state",
            "状态链前序",
            thread="基础设施演进",
            event_at="2026-06-01T00:00:00+08:00",
        ),
        _bucket(
            "seed",
            "工程主线",
            thread="基础设施演进",
            event_at="2026-07-01T00:00:00+08:00",
        ),
        _bucket(
            "next",
            "线内后续",
            thread="基础设施演进",
            event_at="2026-08-01T00:00:00+08:00",
        ),
    ]
    _configure(monkeypatch, tmp_path, buckets, ["seed"])

    monkeypatch.setattr(
        server,
        "_state_recall_profile",
        lambda _query: {
            "enabled": False,
            "evidence_labels": False,
            "fact_keys": (),
            "state_link_limit": 1,
            "view": "neutral",
            "operational_view": "neutral",
        },
    )

    async def state_candidates(*_args, **_kwargs):
        candidate = dict(buckets[0])
        candidate["_z_state_relation"] = "supersedes:seed"
        return [candidate]

    monkeypatch.setattr(server, "_state_link_recall_candidates", state_candidates)

    raw = await server.breath(
        query="工程主线",
        max_results=3,
        relation_depth=0,
        session_id="zx-layer-ownership",
        include_images=False,
    )

    assert _bucket_ids(raw) == ["seed", "next", "state"]
    assert raw.count("[layer:x_timeline]") == 1
    assert raw.count("[layer:z_lifecycle]") == 1
    assert raw.index("[bucket_id:next]") < raw.index("[bucket_id:state]")


@pytest.mark.asyncio
async def test_e_cannot_exceed_a_budget_filled_by_main_and_x(
    tmp_path,
    monkeypatch,
):
    buckets = [
        _bucket(
            "seed",
            "难过时的工程主线",
            thread="基础设施演进",
            event_at="2026-07-01T00:00:00+08:00",
        ),
        _bucket(
            "next",
            "难过时的线内后续",
            thread="基础设施演进",
            event_at="2026-08-01T00:00:00+08:00",
        ),
        _bucket("emotion", "难过时先安静抱住朝灯。"),
    ]
    buckets[2]["metadata"].update({
        "e_authored_by": "claude",
        "e_initial_priority": 80,
        "e_valence": -0.8,
        "e_arousal": 0.4,
        "e_tension": 0.6,
        "e_confidence": 1.0,
        "e_response_tendency": "comfort",
        "e_growth_delta": "stable",
        "e_authored_at": "2026-08-11T10:00:00+00:00",
    })
    _configure(
        monkeypatch,
        tmp_path,
        buckets,
        ["seed"],
        extra_config={
            "e_axis_recall": {
                "enabled": True,
                "mode": "active",
                "activation_id": "timeline-e-budget",
                "allowed_rubric_versions": [
                    "lmc5-experience-20260731-v1",
                ],
                "min_confidence": 0.5,
                "tie_break_weight": 0.2,
                "side_channel_limit": 1,
                "side_channel_scan_limit": 16,
                "side_channel_min_resonance": 0.0,
            },
        },
    )

    open_raw = await server.breath(
        query="我现在真的很难过，工程线怎么走",
        max_results=3,
        relation_depth=0,
        session_id="xe-open",
        include_images=False,
    )
    capped_raw = await server.breath(
        query="我现在真的很难过，工程线怎么走",
        max_results=2,
        relation_depth=0,
        session_id="xe-capped",
        include_images=False,
    )

    assert "[layer:e_emotion]" in open_raw
    assert set(_bucket_ids(open_raw)) == {"seed", "next", "emotion"}
    assert _bucket_ids(capped_raw) == ["seed", "next"]
    assert "[layer:e_emotion]" not in capped_raw


@pytest.mark.asyncio
async def test_query_without_named_thread_is_byte_stable(
    tmp_path,
    monkeypatch,
):
    buckets = [
        _bucket(chr(ord("a") + index), f"普通候选-{index}", score=20 - index)
        for index in range(5)
    ]
    _configure(monkeypatch, tmp_path, buckets, ["a", "b", "c", "d", "e"])

    enabled = await server.breath(
        query="债券基金收益",
        max_results=5,
        relation_depth=0,
        session_id="timeline-enabled",
        include_images=False,
    )
    server.config["timeline_recall"] = {
        "enabled": False,
        "neighbor_window": 1,
    }
    disabled = await server.breath(
        query="债券基金收益",
        max_results=5,
        relation_depth=0,
        session_id="timeline-disabled",
        include_images=False,
    )

    assert enabled == disabled
    assert _bucket_ids(enabled) == ["a", "b", "c", "d", "e"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["duplicate", "dehydrate"])
async def test_unrenderable_timeline_neighbor_restores_primary_tail(
    tmp_path,
    monkeypatch,
    failure,
):
    seed_content = "足够长的重复正文用于验证时间线指纹去重不会吞掉主结果"
    neighbor_content = seed_content if failure == "duplicate" else "线内邻居"
    buckets = [
        _bucket(
            "seed",
            seed_content,
            score=20,
            thread="基础设施演进",
            event_at="2026-07-01T00:00:00+08:00",
        ),
        _bucket("tail", "必须恢复的第二主结果", score=10),
        _bucket(
            "neighbor",
            neighbor_content,
            thread="基础设施演进",
            event_at="2026-08-01T00:00:00+08:00",
        ),
    ]
    _configure(monkeypatch, tmp_path, buckets, ["seed", "tail"])
    if failure == "dehydrate":
        original = server._dehydrate_for_recall

        async def fail_neighbor(content, metadata, *, bucket=None, **kwargs):
            if bucket and bucket.get("id") == "neighbor":
                raise RuntimeError("neighbor dehydration failed")
            return await original(content, metadata, bucket=bucket, **kwargs)

        monkeypatch.setattr(server, "_dehydrate_for_recall", fail_neighbor)

    raw = await server.breath(
        query="工程主结果",
        max_results=2,
        relation_depth=0,
        session_id=f"x-fallback-{failure}",
        include_images=False,
    )

    assert _bucket_ids(raw) == ["seed", "tail"]
    assert "[layer:x_timeline]" not in raw


@pytest.mark.asyncio
async def test_unrenderable_timeline_neighbor_does_not_consume_z_slot(
    tmp_path,
    monkeypatch,
):
    repeated = "足够长的重复正文用于验证失败的时间线预留不会吞掉状态链"
    buckets = [
        _bucket(
            "seed",
            repeated,
            thread="基础设施演进",
            event_at="2026-07-01T00:00:00+08:00",
        ),
        _bucket(
            "neighbor",
            repeated,
            thread="基础设施演进",
            event_at="2026-08-01T00:00:00+08:00",
        ),
        _bucket("state", "状态链旁证"),
    ]
    _configure(monkeypatch, tmp_path, buckets, ["seed"])

    monkeypatch.setattr(
        server,
        "_state_recall_profile",
        lambda _query: {
            "enabled": False,
            "evidence_labels": False,
            "fact_keys": (),
            "state_link_limit": 1,
            "view": "neutral",
            "operational_view": "neutral",
        },
    )

    async def state_candidates(*_args, **_kwargs):
        candidate = dict(buckets[2])
        candidate["_z_state_relation"] = "supersedes:seed"
        return [candidate]

    monkeypatch.setattr(server, "_state_link_recall_candidates", state_candidates)

    raw = await server.breath(
        query="工程主线",
        max_results=2,
        relation_depth=0,
        session_id="x-fallback-z",
        include_images=False,
    )

    assert _bucket_ids(raw) == ["seed", "state"]
    assert "[layer:x_timeline]" not in raw
    assert "[layer:z_lifecycle]" in raw


@pytest.mark.asyncio
async def test_no_timeline_keeps_existing_e_side_channel_behavior(
    tmp_path,
    monkeypatch,
):
    buckets = [
        _bucket("main", "难过时的普通主结果"),
        _bucket("emotion", "难过时先安静抱住朝灯。"),
    ]
    buckets[1]["metadata"].update({
        "e_authored_by": "claude",
        "e_initial_priority": 80,
        "e_valence": -0.8,
        "e_arousal": 0.4,
        "e_tension": 0.6,
        "e_confidence": 1.0,
        "e_response_tendency": "comfort",
        "e_growth_delta": "stable",
        "e_authored_at": "2026-08-11T10:00:00+00:00",
    })
    _configure(
        monkeypatch,
        tmp_path,
        buckets,
        ["main"],
        extra_config={
            "e_axis_recall": {
                "enabled": True,
                "mode": "active",
                "activation_id": "timeline-e-regression",
                "allowed_rubric_versions": ["lmc5-experience-20260731-v1"],
                "min_confidence": 0.5,
                "tie_break_weight": 0.2,
                "side_channel_limit": 1,
                "side_channel_scan_limit": 16,
                "side_channel_min_resonance": 0.0,
            },
        },
    )

    raw = await server.breath(
        query="我现在真的很难过",
        max_results=1,
        relation_depth=0,
        session_id="e-no-timeline",
        include_images=False,
    )

    assert _bucket_ids(raw) == ["main", "emotion"]
    assert "[layer:e_emotion]" in raw
