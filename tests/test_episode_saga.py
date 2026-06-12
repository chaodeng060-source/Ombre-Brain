# Tests for episode_engine.py + saga_engine.py — 内核 3 叙事层
# Drive async via asyncio.run() so we don't depend on pytest-asyncio config.
# Core invariants under test:
#   - find_clusters groups recent, semantically-close, unclaimed Event buckets
#   - exempt buckets (feel / chord / pinned / protected / protected-domain /
#     episode / saga) are NEVER swallowed into an episode
#   - events already cited by an episode's source_buckets are not re-clustered
#   - time span and lookback bounds are respected
#   - saga _load_state derives "claimed episodes" from sagas' episode_buckets
#   - saga _append_episode is idempotent (no duplicate ids)

import asyncio
import math
from datetime import datetime, timedelta

from episode_engine import EpisodeEngine
from saga_engine import SagaEngine


def _iso_days_ago(days: float) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat()


def _bucket(bid, *, content="x", days_ago=1.0, importance=5, domain=None,
            btype="dynamic", pinned=False, protected=False, chord=None,
            source_buckets=None, episode_buckets=None, valence=0.5, arousal=0.3):
    meta = {
        "id": bid,
        "name": bid,
        "type": btype,
        "domain": domain or ["工程"],
        "importance": importance,
        "valence": valence,
        "arousal": arousal,
        "created": _iso_days_ago(days_ago),
        "last_active": _iso_days_ago(days_ago),
    }
    if pinned:
        meta["pinned"] = True
    if protected:
        meta["protected"] = True
    if chord:
        meta["chord_tag"] = chord
    if source_buckets is not None:
        meta["source_buckets"] = source_buckets
    if episode_buckets is not None:
        meta["episode_buckets"] = episode_buckets
    return {"id": bid, "metadata": meta, "content": content}


class FakeEmbedding:
    def __init__(self, vecs: dict):
        self.enabled = True
        self._vecs = vecs

    async def get_embedding(self, bid):
        return self._vecs.get(bid)

    @staticmethod
    def _cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0


class FakeBucketMgr:
    def __init__(self, buckets):
        self._buckets = {b["id"]: b for b in buckets}
        self.updates = []
        self._seq = 0

    async def list_all(self, include_archive=False):
        return [dict(b) for b in self._buckets.values()]

    async def get(self, bucket_id):
        b = self._buckets.get(bucket_id)
        return dict(b) if b else None

    async def update(self, bucket_id, **kwargs):
        self.updates.append((bucket_id, kwargs))
        if bucket_id in self._buckets:
            self._buckets[bucket_id]["metadata"].update(
                {k: v for k, v in kwargs.items() if v is not None}
            )
        return True

    async def create(self, content, name=None, tags=None, domain=None,
                     importance=5, valence=0.5, arousal=0.3,
                     bucket_type="dynamic", **kw):
        self._seq += 1
        bid = f"{bucket_type}-{self._seq}"
        self._buckets[bid] = {
            "id": bid,
            "metadata": {"id": bid, "name": name, "type": bucket_type,
                         "domain": domain or ["未分类"], "importance": importance,
                         "valence": valence, "arousal": arousal,
                         "created": _iso_days_ago(0)},
            "content": content,
        }
        return bid


def _ep_engine(buckets, vecs=None, cfg=None):
    bm = FakeBucketMgr(buckets)
    emb = FakeEmbedding(vecs or {})
    eng = EpisodeEngine({"narrative": cfg or {}}, bm, emb, dehydrator=None)
    return eng, bm, emb


# --------------------------------------------------------------------------
# Exemption
# --------------------------------------------------------------------------

def test_is_exempt_covers_emotional_and_derived_layers():
    assert EpisodeEngine._is_exempt({"type": "feel"})
    assert EpisodeEngine._is_exempt({"type": "episode"})
    assert EpisodeEngine._is_exempt({"type": "saga"})
    assert EpisodeEngine._is_exempt({"type": "permanent"})
    assert EpisodeEngine._is_exempt({"pinned": True})
    assert EpisodeEngine._is_exempt({"protected": True})
    assert EpisodeEngine._is_exempt({"chord_tag": "Em(maj7) → A13"})
    assert EpisodeEngine._is_exempt({"domain": ["恋爱"]})   # protected-resolve domain
    assert EpisodeEngine._is_exempt({"domain": "家庭"})
    # plain dynamic engineering bucket is fair game
    assert not EpisodeEngine._is_exempt({"type": "dynamic", "domain": ["工程"]})


# --------------------------------------------------------------------------
# find_clusters — the golden case
# --------------------------------------------------------------------------

def test_find_clusters_groups_close_recent_unclaimed():
    # A,B,C are semantically close and recent -> one episode.
    # D points orthogonal -> excluded. FEEL is exempt. OLD is past lookback.
    buckets = [
        _bucket("A", days_ago=1.0),
        _bucket("B", days_ago=1.2),
        _bucket("C", days_ago=1.5),
        _bucket("D", days_ago=1.0),
        _bucket("FEEL", days_ago=1.0, btype="feel"),
        _bucket("OLD", days_ago=99.0),
    ]
    vecs = {
        "A": [1.0, 0.0, 0.0],
        "B": [0.98, 0.02, 0.0],
        "C": [0.95, 0.05, 0.0],
        "D": [0.0, 1.0, 0.0],
        "FEEL": [1.0, 0.0, 0.0],   # close, but exempt -> still excluded
        "OLD": [1.0, 0.0, 0.0],    # close, but past lookback -> excluded
    }
    eng, _, _ = _ep_engine(buckets, vecs)
    clusters = asyncio.run(eng.find_clusters())
    assert len(clusters) == 1
    ids = {b["id"] for b in clusters[0]}
    assert ids == {"A", "B", "C"}


def test_find_clusters_excludes_claimed_events():
    # EP is an episode citing A,B,C -> those events are claimed, not re-clustered.
    buckets = [
        _bucket("A", days_ago=1.0),
        _bucket("B", days_ago=1.1),
        _bucket("C", days_ago=1.2),
        _bucket("EP", days_ago=0.5, btype="episode", source_buckets=["A", "B", "C"]),
    ]
    vecs = {k: [1.0, 0.0, 0.0] for k in ("A", "B", "C")}
    eng, _, _ = _ep_engine(buckets, vecs)
    clusters = asyncio.run(eng.find_clusters())
    assert clusters == []


def test_find_clusters_respects_time_span():
    # A,B within span; C is 10 days off the seed -> falls outside episode_span_days.
    buckets = [
        _bucket("A", days_ago=1.0),
        _bucket("B", days_ago=1.5),
        _bucket("C", days_ago=11.0),
    ]
    vecs = {k: [1.0, 0.0, 0.0] for k in ("A", "B", "C")}
    eng, _, _ = _ep_engine(buckets, vecs, cfg={"min_cluster_size": 2, "episode_span_days": 3.0})
    clusters = asyncio.run(eng.find_clusters())
    assert len(clusters) == 1
    assert {b["id"] for b in clusters[0]} == {"A", "B"}


def test_claimed_event_ids_derivation():
    buckets = [
        _bucket("EP1", btype="episode", source_buckets=["A", "B"]),
        _bucket("EP2", btype="episode", source_buckets=["C"]),
        _bucket("D"),
    ]
    eng, _, _ = _ep_engine(buckets)
    claimed = asyncio.run(eng._claimed_event_ids(buckets))
    assert claimed == {"A", "B", "C"}


def test_dominant_domain_and_parse_summary():
    cluster = [
        _bucket("A", domain=["工程"]),
        _bucket("B", domain=["工程"]),
        _bucket("C", domain=["生活"]),
    ]
    assert EpisodeEngine._dominant_domain(cluster) == ["工程"]
    name, summ = EpisodeEngine._parse_summary('{"name": "记忆库长跑", "summary": "我把内核3做了"}')
    assert name == "记忆库长跑"
    assert summ == "我把内核3做了"
    # fenced JSON is tolerated
    n2, s2 = EpisodeEngine._parse_summary('```json\n{"name":"x","summary":"y"}\n```')
    assert (n2, s2) == ("x", "y")
    # garbage -> empty, never raises
    assert EpisodeEngine._parse_summary("not json") == ("", "")


# --------------------------------------------------------------------------
# Saga engine
# --------------------------------------------------------------------------

def _saga_engine(buckets):
    bm = FakeBucketMgr(buckets)
    eng = SagaEngine({"narrative": {}}, bm, dehydrator=None)
    return eng, bm


def test_saga_load_state_derives_claimed_episodes():
    buckets = [
        _bucket("S1", btype="saga", episode_buckets=["EP1", "EP2"]),
        _bucket("EP1", btype="episode"),
        _bucket("EP2", btype="episode"),
        _bucket("EP3", btype="episode"),   # unclaimed
    ]
    eng, _ = _saga_engine(buckets)
    sagas, unclaimed = asyncio.run(eng._load_state())
    assert {s["id"] for s in sagas} == {"S1"}
    assert {e["id"] for e in unclaimed} == {"EP3"}


def test_saga_append_episode_is_idempotent():
    buckets = [_bucket("S1", btype="saga", episode_buckets=["EP1"])]
    eng, bm = _saga_engine(buckets)
    saga = asyncio.run(bm.get("S1"))
    # new id appends
    assert asyncio.run(eng._append_episode(saga, "EP2")) is True
    assert bm._buckets["S1"]["metadata"]["episode_buckets"] == ["EP1", "EP2"]
    # duplicate id is a no-op
    saga2 = asyncio.run(bm.get("S1"))
    assert asyncio.run(eng._append_episode(saga2, "EP1")) is False
