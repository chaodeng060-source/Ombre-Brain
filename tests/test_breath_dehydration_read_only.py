"""Guards for breath's persistent-cache read-only boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import sqlite3

import pytest

import server
from dehydrator import Dehydrator


ROOT = Path(__file__).resolve().parents[1]


def _function_source(name: str) -> str:
    raw = (ROOT / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return ast.get_source_segment(raw, node) or ""
    raise AssertionError(f"{name} not found")


def test_breath_paths_use_the_read_only_dehydration_helper():
    helper = _function_source("_dehydrate_for_recall")
    breath = _function_source("breath")
    hook = _function_source("breath_hook")

    assert "write_cache=False" in helper.replace(" ", "")
    assert "dehydrator.dehydrate" not in breath
    assert "dehydrator.dehydrate" not in hook
    assert "_dehydrate_for_recall" in breath
    assert "_dehydrate_for_recall" in hook


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Completions:
    def __init__(self):
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        summary = (
            '{"core_facts":["只读召回摘要"],'
            '"summary":"只读召回摘要不会落入缓存"}'
        )
        return type("Response", (), {"choices": [_Choice(summary)]})()


class _NoopLoop:
    async def ensure_started(self):
        return None


class _Decay(_NoopLoop):
    @staticmethod
    def apply_retrieval_decay(score, metadata):
        return score


class _Embedding:
    async def search_similar(self, query, top_k=20):
        return []


class _GraphManager:
    literal_candidate_floor = 0.0

    def __init__(self, main: dict, graph: list[dict], archive_dir: Path):
        self.main = main
        self.graph = graph
        self.by_id = {bucket["id"]: bucket for bucket in graph}
        self.archive_dir = str(archive_dir)

    async def search(self, query, limit=20, **kwargs):
        return [self.main]

    async def get(self, bucket_id):
        return self.by_id.get(bucket_id)

    async def list_all(self, include_archive=False, **kwargs):
        return list(self.graph)

    @staticmethod
    def _calc_topic_score(query, bucket):
        return 1.0


def _long_bucket(
    bucket_id: str,
    marker: str,
    *,
    relations: list[dict] | None = None,
    fact_status: str = "",
) -> dict:
    metadata = {
        "id": bucket_id,
        "name": bucket_id,
        "type": "dynamic",
        "world": "daily",
        "domain": ["生活"],
        "importance": 5,
        "valence": 0.5,
        "arousal": 0.3,
        "tags": [],
        "relations": relations or [],
    }
    if fact_status:
        metadata.update(
            fact_key="profile.city",
            fact_status=fact_status,
        )
    return {
        "id": bucket_id,
        "content": (f"{marker}：朝灯的城市事实与迁移背景必须完整保留。" * 80),
        "score": 100.0,
        "metadata": metadata,
    }


@pytest.mark.asyncio
async def test_breath_y_walk_applies_z_gate_without_writing_dehydration_cache(
    test_config,
    tmp_path,
    monkeypatch,
):
    """Exercise the production Y render path after the Z currentness filter."""
    main = _long_bucket(
        "main",
        "MAIN_ONLY",
        relations=[
            {"type": "explains", "target": "historical", "strength": 1.0},
            {"type": "explains", "target": "current", "strength": 1.0},
        ],
    )
    historical = _long_bucket(
        "historical",
        "HISTORICAL_MUST_NOT_RENDER",
        fact_status="historical",
    )
    current = _long_bucket(
        "current",
        "CURRENT_ONLY",
        fact_status="current",
    )
    manager = _GraphManager(
        main,
        [main, historical, current],
        tmp_path / "archive",
    )

    completions = _Completions()
    dehydrator = Dehydrator(test_config)
    dehydrator.api_available = True
    dehydrator.client = type(
        "Client",
        (),
        {
            "chat": type(
                "Chat",
                (),
                {"completions": completions},
            )()
        },
    )()
    cache_path = Path(dehydrator.cache_db_path)
    before_bytes = cache_path.read_bytes()
    with sqlite3.connect(cache_path) as conn:
        before_rows = conn.execute(
            "SELECT content_hash, summary, model, created_at "
            "FROM dehydration_cache ORDER BY content_hash"
        ).fetchall()

    cfg = {
        **server.config,
        "buckets_dir": test_config["buckets_dir"],
        "current_world": "daily",
        "entities": {"enabled": False},
        "query_expansion": {"enabled": False},
        "random_surfacing": {},
        "relation_recall": {
            "propagation_only": True,
            "propagation_types": ["explains"],
            "hop1_min_strength": 0.4,
            "hop2_min_strength": 0.7,
        },
        "fact_slots": {
            "enabled": True,
            "registry": {
                "profile.city": {"aliases": ["城市"]},
            },
        },
    }
    monkeypatch.setattr(server, "config", cfg)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", _Embedding())
    monkeypatch.setattr(server, "dehydrator", dehydrator)
    monkeypatch.setattr(server, "decay_engine", _Decay())
    monkeypatch.setattr(server, "consolidation_engine", _NoopLoop())
    monkeypatch.setattr(server, "episode_engine", _NoopLoop())
    monkeypatch.setattr(server, "_entity_store", None)
    monkeypatch.setattr(server, "_entity_store_key", None)
    monkeypatch.setattr(server, "_entity_store_initialized", False)
    monkeypatch.setattr(server, "_backfill_started", True)
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "0")
    monkeypatch.setenv("OMBRE_LMC5_NIGHT_ENABLED", "1")

    result = await server.breath(
        query="现在城市是什么",
        max_results=2,
        relation_depth=1,
        world="daily",
        include_images=False,
        include_body_state=False,
    )

    assert "[bucket_id:main]" in result
    assert "[role:association]" in result
    assert "[layer:y_relation]" in result
    assert "[bucket_id:current]" in result
    assert "[bucket_id:historical]" not in result
    assert len(completions.requests) == 2
    rendered_prompts = "\n".join(
        str(request.get("messages", "")) for request in completions.requests
    )
    assert "MAIN_ONLY" in rendered_prompts
    assert "CURRENT_ONLY" in rendered_prompts
    assert "HISTORICAL_MUST_NOT_RENDER" not in rendered_prompts

    assert cache_path.read_bytes() == before_bytes
    with sqlite3.connect(cache_path) as conn:
        after_rows = conn.execute(
            "SELECT content_hash, summary, model, created_at "
            "FROM dehydration_cache ORDER BY content_hash"
        ).fetchall()
    assert after_rows == before_rows
