# Tests for consolidation_engine.py — 夜班整理引擎
# Drive async via asyncio.run() so we don't depend on pytest-asyncio config.
# Core invariants under test:
#   - find_duplicates pairs only non-exempt buckets above threshold
#   - find_stale flags only non-exempt, unresolved, idle>days buckets
#   - the cycle NEVER deletes or archives
#   - auto-digest is OFF by default, ON only behind the config flag
#   - a report bucket is written only when there is something to review

import asyncio
import math
from datetime import datetime, timedelta

from consolidation_engine import ConsolidationEngine


def _iso_days_ago(days: float) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat()


def _bucket(bid, *, content="x", vec=None, days_ago=1.0, importance=5,
            domain=None, btype="dynamic", pinned=False, protected=False,
            resolved=False):
    meta = {
        "id": bid,
        "name": bid,
        "type": btype,
        "domain": domain or ["工程"],
        "importance": importance,
        "created": _iso_days_ago(days_ago),
        "last_active": _iso_days_ago(days_ago),
        "resolved": resolved,
    }
    if pinned:
        meta["pinned"] = True
    if protected:
        meta["protected"] = True
    return {"id": bid, "metadata": meta, "content": content, "_vec": vec}


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
        self.updates = []          # (id, kwargs)
        self.created = []          # kwargs dicts
        self.archive_called = False
        self.delete_called = False
        self._report_seq = 0

    async def list_all(self, include_archive=False):
        # return shallow copies so callers iterate a stable snapshot
        return [dict(b) for b in self._buckets.values()]

    async def update(self, bucket_id, **kwargs):
        self.updates.append((bucket_id, kwargs))
        if bucket_id in self._buckets:
            self._buckets[bucket_id]["metadata"].update(kwargs)
        return True

    async def create(self, content, name=None, tags=None, domain=None,
                     importance=5, valence=0.5, arousal=0.3,
                     bucket_type="dynamic", **kw):
        self._report_seq += 1
        rid = f"report-{self._report_seq}"
        self.created.append({"id": rid, "name": name, "domain": domain,
                             "importance": importance, "content": content})
        return rid

    async def archive(self, bucket_id):       # must never be called by the cycle
        self.archive_called = True
        return True

    async def delete(self, bucket_id):         # must never be called by the cycle
        self.delete_called = True
        return True


def _engine(buckets, vecs=None, cfg=None):
    bm = FakeBucketMgr(buckets)
    emb = FakeEmbedding(vecs or {})
    eng = ConsolidationEngine({"consolidation": cfg or {}}, bm, emb)
    return eng, bm, emb


# --------------------------------------------------------------------------

def test_find_duplicates_pairs_above_threshold():
    buckets = [_bucket("A"), _bucket("B"), _bucket("C")]
    vecs = {"A": [1.0, 0.0, 0.0], "B": [1.0, 0.0, 0.0], "C": [0.0, 1.0, 0.0]}
    eng, _, _ = _engine(buckets, vecs)
    pairs = asyncio.run(eng.find_duplicates(0.85))
    assert len(pairs) == 1
    ids = {pairs[0]["a_id"], pairs[0]["b_id"]}
    assert ids == {"A", "B"}
    assert pairs[0]["similarity"] >= 0.99


def test_find_duplicates_skips_exempt():
    # B is pinned, D is feel-type, E is protected domain — all identical to A
    buckets = [
        _bucket("A"),
        _bucket("B", pinned=True),
        _bucket("D", btype="feel"),
        _bucket("E", domain=["恋爱"]),
    ]
    same = [1.0, 0.0, 0.0]
    vecs = {k: same for k in ("A", "B", "D", "E")}
    eng, _, _ = _engine(buckets, vecs)
    pairs = asyncio.run(eng.find_duplicates(0.85))
    assert pairs == []  # only A is eligible, no second non-exempt bucket to pair with


def test_find_stale_flags_only_eligible():
    buckets = [
        _bucket("OLD_ENG", days_ago=40, domain=["工程"]),          # flag
        _bucket("OLD_FEEL", days_ago=40, btype="feel"),            # exempt
        _bucket("OLD_PIN", days_ago=40, pinned=True),              # exempt
        _bucket("OLD_LOVE", days_ago=40, domain=["恋爱"]),         # exempt (protected)
        _bucket("OLD_DONE", days_ago=40, resolved=True),           # skip (resolved)
        _bucket("FRESH", days_ago=1, domain=["工程"]),             # too recent
    ]
    eng, _, _ = _engine(buckets)
    stale = asyncio.run(eng.find_stale(14))
    ids = {s["id"] for s in stale}
    assert ids == {"OLD_ENG"}


def test_cycle_never_deletes_or_archives():
    buckets = [_bucket("A"), _bucket("B"), _bucket("OLD", days_ago=40)]
    vecs = {"A": [1.0, 0.0], "B": [1.0, 0.0], "OLD": [0.0, 1.0]}
    eng, bm, _ = _engine(buckets, vecs)
    asyncio.run(eng.run_consolidation_cycle())
    assert bm.delete_called is False
    assert bm.archive_called is False


def test_cycle_writes_report_only_when_findings():
    # findings present → exactly one report bucket in 记忆整理
    buckets = [_bucket("A"), _bucket("B")]
    vecs = {"A": [1.0, 0.0], "B": [1.0, 0.0]}
    eng, bm, _ = _engine(buckets, vecs)
    res = asyncio.run(eng.run_consolidation_cycle())
    assert res["report_bucket_id"] is not None
    assert len(bm.created) == 1
    assert bm.created[0]["domain"] == ["记忆整理"]

    # nothing to review → no report bucket written
    eng2, bm2, _ = _engine([_bucket("solo")], {"solo": [1.0, 0.0]})
    res2 = asyncio.run(eng2.run_consolidation_cycle())
    assert res2["report_bucket_id"] is None
    assert bm2.created == []


def test_auto_digest_off_by_default():
    buckets = [_bucket("LONG", content="x" * 100), _bucket("SHORT", content="x")]
    vecs = {"LONG": [1.0, 0.0], "SHORT": [1.0, 0.0]}  # sim 1.0 ≥ near_identical
    eng, bm, _ = _engine(buckets, vecs)
    asyncio.run(eng.run_consolidation_cycle())
    digest_updates = [u for u in bm.updates if u[1].get("digested")]
    assert digest_updates == []  # OFF by default


def test_auto_digest_hides_shorter_when_enabled():
    buckets = [_bucket("LONG", content="x" * 100), _bucket("SHORT", content="x")]
    vecs = {"LONG": [1.0, 0.0], "SHORT": [1.0, 0.0]}
    eng, bm, _ = _engine(buckets, vecs, cfg={"auto_digest_near_identical": True})
    asyncio.run(eng.run_consolidation_cycle())
    digest_updates = [u for u in bm.updates if u[1].get("digested")]
    assert len(digest_updates) == 1
    assert digest_updates[0][0] == "SHORT"  # the shorter (less complete) one hidden
