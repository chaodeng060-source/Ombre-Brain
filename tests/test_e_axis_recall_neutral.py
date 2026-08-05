"""E0 immutability plus the bounded E1 live recall contract."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

import server
from e_axis_curated_reader import bind_loaded_curated_source
from e_axis_recall import (
    apply_resonance_tie_break,
    derive_response_posture,
    format_response_posture,
    group_candidate_rows,
    infer_query_emotion,
    load_e_axis_recall_config,
    resonance_score,
    select_current_annotation,
)
from e_axis_shadow import build_shadow_annotation, rank_multiplier
from recall_support import rank_within_relevance_bands


ROOT = Path(__file__).resolve().parents[1]
RUBRIC = "lmc5-experience-20260731-v1"


def _live_config(**overrides):
    live = {
        "enabled": True,
        "mode": "active",
        "activation_id": "test-e1",
        "allowed_rubric_versions": [RUBRIC],
        "min_confidence": 0.5,
        "tie_break_weight": 0.2,
        "side_channel_limit": 1,
        "side_channel_scan_limit": 16,
        "side_channel_min_resonance": 0.55,
        **overrides,
    }
    return load_e_axis_recall_config({"e_axis_recall": live})


def _bucket(bucket_id: str, content: str) -> dict:
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "name": "memory-" + bucket_id,
            "type": "preference",
            "tags": [],
            "world": "daily",
            "domain": ["关系"],
            "importance": 5,
        },
    }


def _row(
    bucket: dict,
    *,
    valence: float,
    arousal: float,
    tension: float,
    confidence: float = 0.9,
    tendency: str = "comfort",
    growth: str = "stable",
    rubric: str = RUBRIC,
    scored_at: str = "2026-08-04T08:00:00+00:00",
) -> dict:
    binding = bind_loaded_curated_source(
        bucket["metadata"],
        bucket["content"],
    )
    assert binding is not None
    row, error = build_shadow_annotation(
        bucket_id="bucket:" + bucket["id"],
        source_digest=binding.source_digest,
        source_kind="curated_memory",
        source_run_id="curated:source-run",
        provider="test-provider",
        scorer="test-scorer",
        model="test-model",
        rubric_version=rubric,
        run_id="test-run",
        trigger_reason=binding.trigger_reason,
        scored_at=scored_at,
        score={
            "valence": valence,
            "arousal": arousal,
            "tension": tension,
            "confidence": confidence,
            "response_tendency": tendency,
            "growth_delta": growth,
        },
    )
    assert error is None and row is not None
    return row


def _function_source(path: Path, name: str) -> str:
    raw = path.read_text(encoding="utf-8")
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return ast.get_source_segment(raw, node) or ""
    raise AssertionError(f"{name} not found")


def test_e0_rows_remain_permanently_neutral_by_themselves():
    assert rank_multiplier(None) == 1.0
    assert rank_multiplier({"score": {"arousal": 1.0}}) == 1.0


def test_live_projection_requires_named_active_config_and_stays_bounded():
    assert not load_e_axis_recall_config({}).enabled
    assert not load_e_axis_recall_config({
        "e_axis_recall": {"enabled": True, "mode": "shadow"}
    }).enabled
    with pytest.raises(ValueError, match="activation_id"):
        load_e_axis_recall_config({
            "e_axis_recall": {
                "enabled": True,
                "mode": "active",
                "allowed_rubric_versions": [RUBRIC],
            }
        })
    with pytest.raises(ValueError, match="tie_break_weight"):
        _live_config(tie_break_weight=0.251)


def test_current_digest_and_confidence_gate_live_annotation():
    bucket = _bucket("memory-a", "朝灯难过时，希望先被安稳地接住。")
    accepted = _row(
        bucket,
        valence=-0.7,
        arousal=0.4,
        tension=0.6,
    )
    low = _row(
        bucket,
        valence=-0.7,
        arousal=0.4,
        tension=0.6,
        confidence=0.4,
        scored_at="2026-08-04T09:00:00+00:00",
    )
    cfg = _live_config()
    grouped = group_candidate_rows([accepted, low], cfg)

    annotation = select_current_annotation(grouped["memory-a"], bucket, cfg)
    assert annotation is not None
    assert annotation.confidence == 0.9

    changed = {**bucket, "content": bucket["content"] + "后来规则变了。"}
    assert select_current_annotation(grouped["memory-a"], changed, cfg) is None


def test_manual_candidate_and_old_rubric_never_promote():
    bucket = _bucket("memory-a", "朝灯难过时，希望先被安稳地接住。")
    curated = _row(
        bucket,
        valence=-0.7,
        arousal=0.4,
        tension=0.6,
    )
    manual = dict(curated, source_kind="manual_bucket")
    candidate = dict(curated, source_kind="lmc5_candidate")
    old_rubric = _row(
        bucket,
        valence=-0.7,
        arousal=0.4,
        tension=0.6,
        rubric="old-rubric",
    )

    grouped = group_candidate_rows(
        [manual, candidate, old_rubric, curated],
        _live_config(),
    )

    assert list(grouped) == ["memory-a"]
    assert grouped["memory-a"] == (curated,)


def test_e_reranks_only_inside_existing_relevance_band():
    query = infer_query_emotion("我真的很难过")
    comforting = _bucket("comforting", "难过时先抱住朝灯。")
    cheerful = _bucket("cheerful", "开心时一起庆祝。")
    cfg = _live_config()
    comfort_row = _row(
        comforting,
        valence=-0.7,
        arousal=0.35,
        tension=0.55,
        tendency="comfort",
    )
    cheerful_row = _row(
        cheerful,
        valence=0.9,
        arousal=0.8,
        tension=0.1,
        tendency="engage",
    )
    grouped = group_candidate_rows([comfort_row, cheerful_row], cfg)
    comfort_e = select_current_annotation(
        grouped["comforting"], comforting, cfg
    )
    cheerful_e = select_current_annotation(grouped["cheerful"], cheerful, cfg)
    assert comfort_e is not None and cheerful_e is not None
    rows = [
        {
            "id": "cheerful",
            "relevance": 10.2,
            "tie": apply_resonance_tie_break(
                10.0,
                resonance_score(query, cheerful_e),
                weight=cfg.tie_break_weight,
            ),
        },
        {
            "id": "comforting",
            "relevance": 10.0,
            "tie": apply_resonance_tie_break(
                10.0,
                resonance_score(query, comfort_e),
                weight=cfg.tie_break_weight,
            ),
        },
        {"id": "fact", "relevance": 11.0, "tie": 0.0},
    ]

    ranked = rank_within_relevance_bands(
        rows,
        relevance_score=lambda item: item["relevance"],
        tie_break_score=lambda item: item["tie"],
        band_width=0.35,
    )

    # The closer emotional match wins its narrow band; the stronger factual
    # result is in another band and cannot be crossed by E.
    assert [item["id"] for item in ranked] == [
        "fact", "comforting", "cheerful"
    ]


def test_posture_is_explicitly_non_factual():
    bucket = _bucket("memory-a", "朝灯难过时，希望先被安稳地接住。")
    cfg = _live_config()
    grouped = group_candidate_rows([
        _row(
            bucket,
            valence=-0.7,
            arousal=0.4,
            tension=0.6,
            tendency="comfort",
            growth="growth",
        )
    ], cfg)
    annotation = select_current_annotation(grouped["memory-a"], bucket, cfg)
    assert annotation is not None
    posture = derive_response_posture([(annotation, 0.9)])
    assert posture is not None
    rendered = format_response_posture(posture, activation_id=cfg.activation_id)
    assert "experience only" in rendered
    assert "不可改写事实" in rendered
    assert "tendency:comfort" in rendered


def test_breath_wires_live_e_after_authority_filters():
    breath = _function_source(ROOT / "server.py", "breath")
    assert "load_e_axis_recall_config" in breath
    assert "select_current_annotation" in breath
    assert "apply_resonance_tie_break" in breath
    assert breath.index("_filter_z_fact_candidates") < breath.index(
        "apply_resonance_tie_break"
    )
    assert "supporting experience only" in breath
    assert "format_response_posture" in breath


class _NoopLoop:
    async def ensure_started(self):
        return None


class _Decay(_NoopLoop):
    @staticmethod
    def apply_retrieval_decay(score, metadata):
        return score


class _Dehydrator:
    async def dehydrate(self, content, metadata, *, write_cache=True):
        assert write_cache is False
        return content


class _Embedding:
    async def search_similar(self, query, top_k=20):
        return []


class _Manager:
    def __init__(self, buckets, archive_dir: Path):
        self.buckets = {item["id"]: item for item in buckets}
        self.archive_dir = str(archive_dir)
        self.literal_candidate_floor = 0.0

    async def search(self, query, limit=20, **kwargs):
        return list(self.buckets.values())[:limit]

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)

    async def list_all(self, include_archive=False, **kwargs):
        return list(self.buckets.values())

    @staticmethod
    def _calc_topic_score(query, bucket):
        return 0.5


@pytest.mark.asyncio
async def test_real_breath_changes_close_order_and_injects_posture(
    tmp_path,
    monkeypatch,
):
    cheerful = _bucket("cheerful", "难过时也马上打起精神庆祝。")
    comforting = _bucket("comforting", "难过时先安静抱住朝灯。")
    for item in (cheerful, comforting):
        item["metadata"].update({
            "created": "2026-08-04T00:00:00+00:00",
            "valence": 0.5,
            "arousal": 0.3,
        })
    vault = tmp_path / "vault"
    vault.mkdir()
    cfg = {
        **server.config,
        "buckets_dir": str(vault),
        "current_world": "daily",
        "entities": {"enabled": False},
        "query_expansion": {"enabled": False},
        "random_surfacing": {},
        "matching": {
            "literal_candidate_floor": 0.0,
            "fused_relevance_tie_band": 0.35,
        },
        "e_axis_shadow": {"rubric_version": RUBRIC},
        "e_axis_recall": {
            "enabled": True,
            "mode": "active",
            "activation_id": "test-e1",
            "allowed_rubric_versions": [RUBRIC],
            "min_confidence": 0.5,
            "tie_break_weight": 0.2,
            "side_channel_limit": 1,
            "side_channel_scan_limit": 16,
            "side_channel_min_resonance": 0.55,
        },
    }
    manager = _Manager([cheerful, comforting], tmp_path / "archive")
    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", _Embedding())
    monkeypatch.setattr(server, "dehydrator", _Dehydrator())
    monkeypatch.setattr(server, "decay_engine", _Decay())
    monkeypatch.setattr(server, "consolidation_engine", _NoopLoop())
    monkeypatch.setattr(server, "episode_engine", _NoopLoop())
    monkeypatch.setattr(server, "_backfill_started", True)
    monkeypatch.setattr(server, "_entity_store", None)
    monkeypatch.setattr(server, "_e_axis_shadow_store", None)

    async def passthrough(
        _query,
        values,
        *,
        mode,
        max_results,
        allow_empty=False,
    ):
        return values[:max_results]

    monkeypatch.setattr(server, "_ds_filter_candidates", passthrough)
    store = server._get_e_axis_shadow_store()
    assert store.append(_row(
        cheerful,
        valence=0.9,
        arousal=0.9,
        tension=0.1,
        tendency="engage",
    ))
    assert store.append(_row(
        comforting,
        valence=-0.7,
        arousal=0.35,
        tension=0.55,
        tendency="comfort",
    ))
    assert len(store.load()) == 2

    task = asyncio.create_task(server.breath(
        query="我现在真的很难过",
        max_results=2,
        relation_depth=0,
        world="daily",
        include_images=False,
        include_body_state=False,
    ))
    done, _ = await asyncio.wait({task}, timeout=5)
    if not done:
        task.print_stack()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        pytest.fail("breath did not return within integration-test timeout")
    result = task.result()

    assert result.index("bucket_id:comforting") < result.index(
        "bucket_id:cheerful"
    )
    assert "E轴回应姿态" in result
    assert "tendency:comfort" in result
    assert "不可改写事实" in result
