from __future__ import annotations

from types import SimpleNamespace

import pytest

import server


ENTITY_LOW = "6cc5995aea84"
ENTITY_HIGH_BOUNDARY = "013da98a75e5"
ENTITY_HIGH = "019af40158f7"
NORMAL_KEYWORD = "normal-keyword"
NORMAL_VECTOR = "normal-vector"


def _bucket(bucket_id: str, *, topic_score: float = 0.0) -> dict:
    return {
        "id": bucket_id,
        "content": f"body:{bucket_id}",
        "metadata": {
            "name": bucket_id,
            "type": "dynamic",
            "world": "daily",
            "domain": ["工程"],
            "created": "2026-09-05T00:00:00",
            "importance": 5,
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
        },
        "score": topic_score * 100.0,
        "_topic_score_for_test": topic_score,
    }


class _NoopLoop:
    async def ensure_started(self):
        return None


class _Decay(_NoopLoop):
    @staticmethod
    def apply_retrieval_decay(score, _metadata):
        return score


class _Dehydrator:
    async def dehydrate(self, content, _metadata, *, write_cache=True):
        assert write_cache is False
        return content


class _Embedding:
    def __init__(self):
        self.queries = []

    async def search_similar(self, query, top_k=20):
        self.queries.append((query, top_k))
        return [(NORMAL_VECTOR, 0.70)]

    async def search_similar_with_status(self, query, top_k=20):
        self.queries.append((query, top_k))
        return [(NORMAL_VECTOR, 0.70)], "ok"


class _EntityStore:
    _entity_to_bucket = {
        "entity-low": ENTITY_LOW,
        "entity-high-boundary": ENTITY_HIGH_BOUNDARY,
        "entity-high": ENTITY_HIGH,
    }

    def resolve_query(self, query):
        return SimpleNamespace(
            canonical_query=query,
            entity_ids=tuple(self._entity_to_bucket),
            terms=("rare-name", "common-name", "very-common-name"),
        )

    def linked_bucket_ids(self, *, entity_ids):
        return [
            self._entity_to_bucket[entity_id]
            for entity_id in entity_ids
            if entity_id in self._entity_to_bucket
        ]

    @staticmethod
    def link_is_current(_bucket_id, _content):
        return True


class _Manager:
    literal_candidate_floor = 40.0

    def __init__(self, buckets, tmp_path):
        self.buckets = {bucket["id"]: bucket for bucket in buckets}
        self.archive_dir = str(tmp_path / "archive")
        self.search_calls = []
        self.df_calls = []

    async def list_all(self, include_archive=False, **_kwargs):
        assert include_archive is False
        return list(self.buckets.values())

    async def search(self, query, limit=20, **_kwargs):
        self.search_calls.append((query, limit))
        return [self.buckets[NORMAL_KEYWORD]]

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)

    @staticmethod
    def _calc_topic_score(_query, bucket):
        return float(bucket.get("_topic_score_for_test", 0.0))

    def entity_term_df_stats(self, terms):
        self.df_calls.append(tuple(terms))
        return {
            "rare-name": (1, 10),
            "common-name": (2, 10),
            "very-common-name": (3, 10),
        }


def _configure_breath(tmp_path, monkeypatch, *, guard_enabled: bool):
    buckets = [
        _bucket(NORMAL_KEYWORD, topic_score=0.80),
        _bucket(NORMAL_VECTOR, topic_score=0.0),
        # Frozen target shape: literal=16.8675, original vector=0.
        _bucket(ENTITY_LOW, topic_score=0.168675),
        _bucket(ENTITY_HIGH_BOUNDARY, topic_score=0.0),
        _bucket(ENTITY_HIGH, topic_score=0.0),
    ]
    manager = _Manager(buckets, tmp_path)
    embedding = _Embedding()
    entity_store = _EntityStore()
    captures = {
        "rrf_channels": [],
        "anchor_rows": [],
        "state_seed_ids": [],
        "partials": [],
        "partial_at_ds": [],
        "ds_calls": [],
    }

    monkeypatch.setenv(
        "OMBRE_ENTITY_SCORE_GUARD_ENABLED",
        "1" if guard_enabled else "0",
    )
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")
    monkeypatch.setenv("OMBRE_ANCHOR_QUALITY_GATE_ENABLED", "0")
    monkeypatch.setenv("OMBRE_LITERAL_COLLISION_GUARD_ENABLED", "0")
    monkeypatch.setenv("OMBRE_UPSTREAM_FUSION_SHADOW", "0")
    monkeypatch.setattr(
        server,
        "config",
        {
            **server.config,
            "buckets_dir": str(tmp_path / "vault"),
            "current_world": "daily",
            "entities": {"enabled": True, "rrf_weight": 1.0, "top_k": 20},
            "query_expansion": {"enabled": False},
            "rrf": {"k": 60, "keyword_weight": 1.0, "vector_weight": 1.0},
            "random_surfacing": {},
        },
    )
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", embedding)
    monkeypatch.setattr(server, "dehydrator", _Dehydrator())
    monkeypatch.setattr(server, "decay_engine", _Decay())
    monkeypatch.setattr(server, "consolidation_engine", _NoopLoop())
    monkeypatch.setattr(server, "episode_engine", _NoopLoop())
    monkeypatch.setattr(server, "_backfill_started", True)
    monkeypatch.setattr(server, "_get_entity_store", lambda **_kwargs: entity_store)
    monkeypatch.setattr(server, "pg_lexical_mode", lambda: "off")

    async def no_lexical(*_args, **_kwargs):
        return []

    monkeypatch.setattr(server, "search_curated_lexical", no_lexical)

    original_fuse = server.lmc5_fuse_ranked_channels

    def capture_fuse(channels, *args, **kwargs):
        captures["rrf_channels"].append(
            [
                [str(bucket_id) for bucket_id, _score in ranked]
                for ranked, _weight in channels
            ]
        )
        return original_fuse(channels, *args, **kwargs)

    monkeypatch.setattr(server, "lmc5_fuse_ranked_channels", capture_fuse)

    original_anchor = server._filter_anchor_policy_candidates

    def capture_anchor(rows, policy):
        selected = original_anchor(rows, policy)
        captures["anchor_rows"].append(
            [
                {
                    "id": str(row.get("id")),
                    "entity_match": bool(row.get("entity_match")),
                    "weak_entity": bool(row.get("weak_entity")),
                    "anchor": server._anchor_adapted_relevance_score(row),
                }
                for row in selected
            ]
        )
        return selected

    monkeypatch.setattr(server, "_filter_anchor_policy_candidates", capture_anchor)

    async def no_state_links(rows, **_kwargs):
        captures["state_seed_ids"].append(
            [str(row.get("id")) for row in rows if row.get("id")]
        )
        return []

    monkeypatch.setattr(server, "_state_link_recall_candidates", no_state_links)

    def capture_partial(value):
        captures["partials"].append(value)

    monkeypatch.setattr(server, "set_recall_partial_result", capture_partial)

    async def ds_gate(
        query,
        candidates,
        *,
        mode,
        max_results,
        force_keep_ids=None,
        allow_empty=False,
        **unexpected_kwargs,
    ):
        normal_rows = list(candidates)
        captures["partial_at_ds"].append(
            captures["partials"][-1] if captures["partials"] else ""
        )
        captures["ds_calls"].append(
            {
                "query": query,
                "mode": mode,
                "max_results": max_results,
                "force_keep_ids": set(force_keep_ids or set()),
                "allow_empty": allow_empty,
                "ids": [row["id"] for row in normal_rows],
                "weak_entity_ids": [
                    row["id"] for row in normal_rows
                    if row.get("weak_entity") is True
                ],
                "unexpected_kwargs": unexpected_kwargs,
            }
        )
        return normal_rows[:max_results]

    monkeypatch.setattr(server, "_ds_filter_candidates", ds_gate)

    return manager, embedding, captures


def _flatten_anchor_rows(captures):
    return [row for batch in captures["anchor_rows"] for row in batch]


@pytest.mark.asyncio
async def test_guard_off_is_byte_and_call_shape_equivalent_to_3e16a3f(
    tmp_path,
    monkeypatch,
):
    manager, _embedding, captures = _configure_breath(
        tmp_path,
        monkeypatch,
        guard_enabled=False,
    )

    result = await server.breath(
        query="rare-name",
        max_results=3,
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )

    assert result == (
        "[bucket_id:normal-keyword] body:normal-keyword\n"
        "---\n"
        "[语义关联] [bucket_id:normal-vector] body:normal-vector\n"
        "---\n"
        "[实体关联] [bucket_id:6cc5995aea84] body:6cc5995aea84"
    )
    assert captures["rrf_channels"] == [[
        [NORMAL_KEYWORD],
        [NORMAL_VECTOR],
        [ENTITY_LOW, ENTITY_HIGH_BOUNDARY, ENTITY_HIGH],
    ]]
    anchor_by_id = {row["id"]: row for row in _flatten_anchor_rows(captures)}
    assert anchor_by_id[ENTITY_LOW]["entity_match"] is True
    assert anchor_by_id[ENTITY_LOW]["anchor"] == pytest.approx(0.45)
    assert captures["ds_calls"] == [{
        "query": "rare-name",
        "mode": "search",
        "max_results": 3,
        "force_keep_ids": set(),
        "allow_empty": True,
        "ids": [
            NORMAL_KEYWORD,
            NORMAL_VECTOR,
            ENTITY_LOW,
            ENTITY_HIGH_BOUNDARY,
            ENTITY_HIGH,
        ],
        "weak_entity_ids": [],
        "unexpected_kwargs": {},
    }]
    assert manager.df_calls == []


@pytest.mark.asyncio
async def test_guard_on_ds_receives_only_normal_pool_and_never_issues_ticket(
    tmp_path,
    monkeypatch,
):
    manager, embedding, captures = _configure_breath(
        tmp_path,
        monkeypatch,
        guard_enabled=True,
    )

    result = await server.breath(
        query="rare-name",
        max_results=3,
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )

    assert len(captures["ds_calls"]) == 1
    ds_call = captures["ds_calls"][0]
    assert ds_call["ids"] == [NORMAL_KEYWORD, NORMAL_VECTOR]
    assert ds_call["weak_entity_ids"] == []
    assert ds_call["unexpected_kwargs"] == {}
    assert ds_call["max_results"] == 3
    assert manager.df_calls == []
    assert "[实体门卫复核]" not in result
    assert "weak_entity" not in result


@pytest.mark.asyncio
async def test_guard_on_entity_hits_do_not_enter_rrf_anchor_or_state_seed(
    tmp_path,
    monkeypatch,
):
    manager, embedding, captures = _configure_breath(
        tmp_path,
        monkeypatch,
        guard_enabled=True,
    )

    result = await server.breath(
        query="rare-name",
        max_results=3,
        relation_depth=0,
        include_images=False,
        include_body_state=False,
    )

    assert captures["rrf_channels"] == [[[NORMAL_KEYWORD], [NORMAL_VECTOR]]]
    anchor_ids = {row["id"] for row in _flatten_anchor_rows(captures)}
    assert anchor_ids == {NORMAL_KEYWORD, NORMAL_VECTOR}
    assert captures["state_seed_ids"] == [[NORMAL_KEYWORD, NORMAL_VECTOR]]
    assert manager.search_calls == [("rare-name", server.BREATH_RECALL_POOL_SIZE)]
    assert len(embedding.queries) == 1
    assert manager.df_calls == []

    partial = captures["partial_at_ds"][0]
    assert f"bucket_id:{NORMAL_KEYWORD}" in partial
    assert f"bucket_id:{NORMAL_VECTOR}" in partial
    for entity_id in (ENTITY_LOW, ENTITY_HIGH_BOUNDARY, ENTITY_HIGH):
        assert f"bucket_id:{entity_id}" not in partial

    assert f"[bucket_id:{NORMAL_KEYWORD}]" in result
    assert f"[bucket_id:{NORMAL_VECTOR}]" in result
    for entity_id in (ENTITY_LOW, ENTITY_HIGH_BOUNDARY, ENTITY_HIGH):
        assert f"[bucket_id:{entity_id}]" not in result
