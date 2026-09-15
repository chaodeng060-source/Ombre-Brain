# Equivalence tests for the vectorized duplicate scan (2026-09-14 night-run
# lock incident).  The numpy block screen must return exactly what the legacy
# pure-Python scorer returns: the same pairs, the same rounded similarities,
# the same error strings in the same order, and the same _top_pairs result.

import asyncio
import math
import random

import pytest

import consolidation_engine as consolidation_module
from consolidation_engine import ConsolidationEngine
from embedding_engine import EmbeddingEngine

np = pytest.importorskip("numpy")


class StockEmbedding(EmbeddingEngine):
    """Real EmbeddingEngine kernel, in-memory vectors, no DB or network."""

    def __init__(self, vecs: dict):  # noqa: D401 - deliberately skips super()
        self.enabled = True
        self._vecs = vecs

    async def get_embedding(self, bucket_id):
        return self._vecs.get(bucket_id)


class OverriddenKernel(StockEmbedding):
    @staticmethod
    def _max_prepared_similarity(left, right):
        return EmbeddingEngine._max_prepared_similarity(left, right)


class FakeBucketMgr:
    def __init__(self, buckets):
        self._buckets = {bucket["id"]: bucket for bucket in buckets}

    async def list_all(self, include_archive=False):
        return [dict(bucket) for bucket in self._buckets.values()]


def _bucket(bucket_id, content="x"):
    return {
        "id": bucket_id,
        "metadata": {"id": bucket_id, "name": f"name-{bucket_id}", "type": "dynamic", "domain": ["工程"]},
        "content": content,
    }


def _engine(vecs, buckets=None, max_report_pairs=10**9, engine_cls=StockEmbedding):
    buckets = buckets if buckets is not None else [_bucket(key, "x" * (len(key) % 7)) for key in vecs]
    manager = FakeBucketMgr(buckets)
    embedding = engine_cls(vecs)
    engine = ConsolidationEngine(
        {"consolidation": {"max_report_pairs": max_report_pairs}, "metabolism": {"mode": "report_only"}},
        manager,
        embedding,
    )
    return engine, manager, embedding


def _unit(rng, dim):
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _mix(rng, base, cosine):
    """A unit-ish vector whose cosine with ``base`` is close to ``cosine``."""

    noise = _unit(rng, len(base))
    dot = sum(x * y for x, y in zip(noise, base))
    orth = [n - dot * b for n, b in zip(noise, base)]
    orth_norm = math.sqrt(sum(value * value for value in orth)) or 1.0
    orth = [value / orth_norm for value in orth]
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return [cosine * b + sine * o for b, o in zip(base, orth)]


BAD_VALUES = [
    "not-a-list",
    [],
    [[]],
    [True, 0.0],
    [[1.0, 0.0], ["not-a-number", 0.0]],
    [[1.0, 0.0], [1.0, 0.0, 0.0]],
    [10**400, 0.0],
    {"vector": [1.0]},
    None,
]


def _random_dataset(seed, *, count, dim, bad=True, extra_dims=True, extremes=True):
    rng = random.Random(seed)
    vecs = {}
    bases = []
    for index in range(count):
        bucket_id = f"b{seed}-{index:04d}"
        roll = rng.random()
        if bases and roll < 0.25:
            base = rng.choice(bases)
            target = rng.choice([0.85, 0.8500000001, 0.8499999999, 0.9, 0.97, 0.99995, 1.0, 0.90005, 0.5, rng.uniform(0.6, 1.0)])
            vec = _mix(rng, base, target)
        elif roll < 0.30 and bases:
            vec = list(rng.choice(bases))  # exact duplicate
        elif roll < 0.33:
            vec = [-value for value in rng.choice(bases)] if bases else _unit(rng, dim)
        else:
            vec = _unit(rng, dim)
            bases.append(vec)
        segments = rng.choice([1, 1, 1, 1, 2, 3, 4, 10])
        if segments == 1:
            value = vec if rng.random() < 0.5 else [vec]
        else:
            value = [vec] + [_unit(rng, dim) if rng.random() < 0.5 else _mix(rng, vec, rng.uniform(0.7, 1.0)) for _ in range(segments - 1)]
        vecs[bucket_id] = value
    specials = []
    if extremes:
        specials.extend([
            [0.0] * dim,                                   # zero norm
            [[0.0] * dim, bases[0] if bases else _unit(rng, dim)],
            [1e-170] * dim,                                # squares underflow -> norm 0
            [1e-155] * dim,                                # tiny but non-zero norm (unsafe)
            [1e200] * dim,                                 # overflow -> non-finite similarity
            [1e120] + [0.0] * (dim - 1),                   # large norm (unsafe)
            [1] + [0] * (dim - 1),                         # integer components
            [float(rng.randint(-3, 3)) for _ in range(dim)],
        ])
    if extra_dims:
        specials.extend([
            _unit(rng, dim + 1),
            [_unit(rng, dim + 1), _unit(rng, dim + 1)],
            _unit(rng, max(2, dim // 2)),
        ])
    if bad:
        specials.extend(BAD_VALUES)
    for offset, value in enumerate(specials):
        vecs[f"s{seed}-{offset:02d}"] = value
    # interleave specials with normal ids so bad records sit in the middle
    items = list(vecs.items())
    rng.shuffle(items)
    return dict(items)


def _embs_snapshot(vecs):
    # find_duplicates keeps only non-None embeddings, in candidate order
    return {key: value for key, value in vecs.items() if value is not None}


def _legacy_incremental_pair_ids(ids, new_set):
    new_positions = [index for index, bucket_id in enumerate(ids) if bucket_id in new_set]
    pair_ids = []
    for index, left in enumerate(ids):
        if left in new_set:
            pair_ids.extend((left, right) for right in ids[index + 1:])
            continue
        pair_ids.extend((left, ids[position]) for position in new_positions if position > index)
    return pair_ids


def _compare_full(vecs, threshold, max_report_pairs=10**9):
    engine, manager, _ = _engine(vecs, max_report_pairs=max_report_pairs)
    candidates = asyncio.run(manager.list_all())
    embs = _embs_snapshot(vecs)
    legacy = asyncio.run(engine._score_duplicate_pairs(candidates, embs, threshold))
    fast = asyncio.run(engine._score_pairs_vectorized(candidates, embs, threshold))
    return legacy, fast


def _compare_incremental(vecs, new_ids, threshold, max_report_pairs=10**9):
    engine, manager, _ = _engine(vecs, max_report_pairs=max_report_pairs)
    candidates = asyncio.run(manager.list_all())
    embs = _embs_snapshot(vecs)
    ids = list(embs)
    new_set = set(new_ids) & set(ids)
    pair_ids = _legacy_incremental_pair_ids(ids, new_set)
    legacy = asyncio.run(engine._score_pair_ids(candidates, embs, pair_ids, threshold))
    fast = asyncio.run(engine._score_pairs_vectorized(candidates, embs, threshold, new_ids=new_set))
    return legacy, fast


@pytest.fixture
def tiny_blocks(monkeypatch):
    # Force many matrix blocks so block boundaries are exercised.
    monkeypatch.setattr(consolidation_module, "_VECTOR_BLOCK_BYTES", 8 * 40 * 64)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("threshold", [0.85, 0.5, 0.97, 1.0])
def test_full_scan_matches_legacy_with_bad_and_extreme_data(seed, threshold, tiny_blocks):
    vecs = _random_dataset(seed, count=48, dim=12)
    legacy, fast = _compare_full(vecs, threshold)
    assert fast == legacy
    assert [type(pair["similarity"]) for pair in fast[0]] == [float] * len(fast[0])


@pytest.mark.parametrize("seed", range(6))
def test_full_scan_matches_legacy_on_clean_multi_block_data(seed, tiny_blocks):
    vecs = _random_dataset(seed + 100, count=160, dim=16, bad=False, extra_dims=False, extremes=False)
    legacy, fast = _compare_full(vecs, 0.85)
    assert legacy[1] == []
    assert fast == legacy
    assert legacy[0], "fixture should contain duplicate pairs"


@pytest.mark.parametrize("threshold", [0.0, -0.5, 1e-7, 0.9999999])
def test_full_scan_matches_legacy_at_degenerate_thresholds(threshold, tiny_blocks):
    vecs = _random_dataset(7, count=14, dim=6)
    legacy, fast = _compare_full(vecs, threshold)
    assert fast == legacy


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("new_fraction", [0.02, 0.2, 1.0])
def test_incremental_scan_matches_legacy(seed, new_fraction, tiny_blocks):
    vecs = _random_dataset(seed + 200, count=40, dim=10)
    rng = random.Random(seed)
    ids = [key for key, value in vecs.items() if value is not None]
    new_ids = [bucket_id for bucket_id in ids if rng.random() < new_fraction] or ids[:1]
    legacy, fast = _compare_incremental(vecs, new_ids, 0.85)
    assert fast == legacy


def test_default_report_cap_and_ties_match_legacy(tiny_blocks):
    rng = random.Random(5)
    base = _unit(rng, 8)
    vecs = {f"dup-{index:03d}": list(base) for index in range(30)}
    vecs.update({f"other-{index:03d}": _unit(rng, 8) for index in range(30)})
    legacy, fast = _compare_full(vecs, 0.85, max_report_pairs=50)
    assert len(legacy[0]) == 50
    assert fast == legacy


def test_near_threshold_and_rounding_boundaries_are_exact(tiny_blocks):
    vecs = {"anchor": [1.0, 0.0, 0.0]}
    for index, cosine in enumerate([
        0.85, 0.85 + 1e-12, 0.85 - 1e-12, 0.8499999, 0.8500001,
        0.90005, 0.90005 + 1e-13, 0.90005 - 1e-13, 0.99995, 0.12345, 0.85005,
    ]):
        vecs[f"c{index:02d}"] = [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)), 0.0]
    legacy, fast = _compare_full(vecs, 0.85)
    assert fast == legacy


def test_fixtures_exercise_shortcut_exact_and_legacy_branches(monkeypatch, tiny_blocks):
    stats = {"shortcut": 0, "exact": 0, "slow_events": 0}
    original_classify = consolidation_module._VectorScan._classify
    original_slow = consolidation_module._VectorScan.slow_events
    margin = consolidation_module._VECTOR_SCREEN_MARGIN

    def classify(self, members, left, right, values):
        upper = self.threshold + margin
        for value in values.tolist():
            if value >= upper and round(value - margin, 4) == round(value + margin, 4):
                stats["shortcut"] += 1
            else:
                stats["exact"] += 1
        return original_classify(self, members, left, right, values)

    def slow(self):
        events = original_slow(self)
        stats["slow_events"] += len(events)
        return events

    monkeypatch.setattr(consolidation_module._VectorScan, "_classify", classify)
    monkeypatch.setattr(consolidation_module._VectorScan, "slow_events", slow)
    for seed in range(4):
        vecs = _random_dataset(seed, count=48, dim=12)
        legacy, fast = _compare_full(vecs, 0.85)
        assert fast == legacy
    near = {"anchor": [1.0, 0.0, 0.0]}
    for index, cosine in enumerate([0.85, 0.85 + 1e-12, 0.90005, 0.99995]):
        near[f"c{index}"] = [cosine, math.sqrt(1.0 - cosine * cosine), 0.0]
    legacy, fast = _compare_full(near, 0.85)
    assert fast == legacy
    assert stats["shortcut"] > 0
    assert stats["exact"] > 0
    assert stats["slow_events"] > 0


def test_single_malformed_record_errors_keep_legacy_order_and_strings():
    vecs = {
        "GOOD": [1.0, 0.0],
        "BAD": [[1.0, 0.0], ["not-a-number", 0.0]],
        "ALSO": [1.0, 0.0],
        "WIDE": [[1.0, 0.0, 0.0]],
    }
    legacy, fast = _compare_full(vecs, 0.85)
    assert fast == legacy
    assert fast[1] == [
        "find_duplicates.cosine:GOOD:BAD:TypeError",
        "find_duplicates.cosine:GOOD:WIDE:TypeError",
        "find_duplicates.cosine:BAD:ALSO:TypeError",
        "find_duplicates.cosine:BAD:WIDE:TypeError",
        "find_duplicates.cosine:ALSO:WIDE:TypeError",
    ]


def _run_find_duplicates_sequence(vecs_sequence, monkeypatch, use_numpy):
    if not use_numpy:
        monkeypatch.setattr(consolidation_module, "_np", None)
    else:
        monkeypatch.setattr(consolidation_module, "_np", np)
    first_vecs = vecs_sequence[0]
    engine, manager, embedding = _engine(dict(first_vecs), max_report_pairs=40)
    if use_numpy:
        async def legacy_forbidden(*_args, **_kwargs):
            raise AssertionError("vectorized scan fell back to the legacy scorer")

        monkeypatch.setattr(engine, "_score_pair_ids", legacy_forbidden)
    observations = []
    for vecs in vecs_sequence:
        manager._buckets = {key: _bucket(key, "x" * (len(key) % 5)) for key in vecs}
        embedding._vecs = dict(vecs)
        engine._read_errors = []
        pairs = asyncio.run(engine.find_duplicates(0.85))
        observations.append((
            pairs,
            list(engine._read_errors),
            engine._last_duplicate_scan_mode,
            engine._last_duplicate_pairs_scored,
        ))
    return observations


def test_find_duplicates_full_then_incremental_matches_pure_python(monkeypatch, tiny_blocks):
    clean = _random_dataset(301, count=60, dim=12, bad=False, extra_dims=False, extremes=False)
    grown = dict(clean)
    grown.update(_random_dataset(302, count=9, dim=12, bad=False, extra_dims=False, extremes=False))
    with_bad = dict(grown)
    with_bad["zz-bad"] = [[1.0] * 12, [1.0] * 11]
    sequence = [clean, clean, grown, with_bad]
    expected = _run_find_duplicates_sequence(sequence, monkeypatch, use_numpy=False)
    actual = _run_find_duplicates_sequence(sequence, monkeypatch, use_numpy=True)
    assert actual == expected
    assert [item[2] for item in actual] == ["full", "incremental_append", "incremental_append", "incremental_append"]


def test_stock_engine_uses_vectorized_path_and_fakes_keep_legacy(monkeypatch):
    vecs = {"A": [1.0, 0.0], "B": [1.0, 0.0], "C": [0.0, 1.0]}
    engine, _, _ = _engine(vecs)

    async def legacy_forbidden(*_args, **_kwargs):
        raise AssertionError("legacy scorer must not run for the stock kernel")

    monkeypatch.setattr(engine, "_score_pair_ids", legacy_forbidden)
    pairs = asyncio.run(engine.find_duplicates(0.85))
    assert [(pair["a_id"], pair["b_id"], pair["similarity"]) for pair in pairs] == [("A", "B", 1.0)]

    overridden, _, _ = _engine(vecs, engine_cls=OverriddenKernel)
    assert overridden._vectorized_scoring_available(0.85) is False
    patched, _, patched_embedding = _engine(vecs)
    patched_embedding._max_prepared_similarity = EmbeddingEngine._max_prepared_similarity
    assert patched._vectorized_scoring_available(0.85) is False
    assert engine._vectorized_scoring_available(True) is False
    assert engine._vectorized_scoring_available(float("nan")) is False


def test_without_numpy_the_legacy_scorer_runs(monkeypatch):
    monkeypatch.setattr(consolidation_module, "_np", None)
    vecs = {"A": [1.0, 0.0], "B": [1.0, 0.0]}
    engine, _, _ = _engine(vecs)
    calls = []
    original = engine._score_pair_ids

    async def observed(*args, **kwargs):
        calls.append(1)
        return await original(*args, **kwargs)

    monkeypatch.setattr(engine, "_score_pair_ids", observed)
    pairs = asyncio.run(engine.find_duplicates(0.85))
    assert calls == [1]
    assert len(pairs) == 1


def test_vectorized_failure_falls_back_to_exact_legacy(monkeypatch):
    vecs = {"A": [1.0, 0.0], "B": [1.0, 0.0]}
    engine, _, _ = _engine(vecs)

    async def broken(*_args, **_kwargs):
        raise MemoryError("simulated")

    monkeypatch.setattr(engine, "_score_pairs_vectorized", broken)
    pairs = asyncio.run(engine.find_duplicates(0.85))
    assert [(pair["a_id"], pair["b_id"]) for pair in pairs] == [("A", "B")]


def test_event_loop_keeps_running_during_vectorized_scan(monkeypatch):
    monkeypatch.setattr(consolidation_module, "_VECTOR_BLOCK_BYTES", 8 * 64 * 32)
    vecs = _random_dataset(900, count=400, dim=32, bad=False, extra_dims=False, extremes=False)
    engine, manager, _ = _engine(vecs)

    async def scenario():
        candidates = await manager.list_all()
        ticks = 0
        done = False

        async def ticker():
            nonlocal ticks
            while not done:
                ticks += 1
                await asyncio.sleep(0)

        task = asyncio.create_task(ticker())
        result = await engine._score_pairs_vectorized(candidates, _embs_snapshot(vecs), 0.85)
        done = True
        await task
        return result, ticks

    (pairs, errors), ticks = asyncio.run(scenario())
    assert errors == []
    assert ticks > 10
