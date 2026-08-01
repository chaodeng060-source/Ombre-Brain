"""Strict E-axis shadow contract tests."""

import asyncio
import copy
import hashlib
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from e_axis_shadow import (
    CONTRACT_VERSION,
    EAxisShadowConflictError,
    EAxisShadowStore,
    build_failure_record,
    build_shadow_annotation,
    normalize_min_confidence,
    rank_multiplier,
    strict_json_loads,
    validate_shadow_score,
)


def _score(**overrides):
    value = {
        "valence": 0.25,
        "arousal": 0.7,
        "tension": 0.4,
        "confidence": 0.8,
        "response_tendency": "engage",
        "growth_delta": "stable",
    }
    value.update(overrides)
    return value


def _annotation():
    return build_shadow_annotation(
        bucket_id="bucket-1",
        source_digest=hashlib.sha256(b"content").hexdigest(),
        source_kind="lmc5_candidate",
        source_run_id="lmc5-night-20260728",
        provider="provider-a",
        scorer="e-shadow-v2",
        model="model-a",
        rubric_version="rubric-1",
        run_id="run-1",
        trigger_reason="type.preference",
        score=_score(),
        scored_at="2026-07-28T00:00:00+00:00",
    )


def test_strict_score_accepts_exact_schema():
    normalized, error = validate_shadow_score(_score())
    assert error is None
    assert normalized == _score()


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        (_score(arousal=True), "schema.arousal"),
        (_score(tension=float("nan")), "schema.tension"),
        (_score(confidence=float("inf")), "schema.confidence"),
        (_score(valence=2.0), "range.valence"),
        (_score(confidence=0.1), "confidence.low"),
        (_score(response_tendency="obey"), "enum.response_tendency"),
        (_score(growth_delta="unknown"), "enum.growth_delta"),
        ({key: value for key, value in _score().items() if key != "tension"}, "schema.missing"),
        ({**_score(), "explanation": "guess"}, "schema.unexpected"),
    ],
)
def test_strict_score_rejects_invalid_values(payload, category):
    normalized, error = validate_shadow_score(payload)
    assert normalized is None
    assert error == category


def test_annotation_is_permanently_shadow_only():
    annotation, error = _annotation()
    assert error is None
    assert annotation["contract_version"] == CONTRACT_VERSION
    assert annotation["shadow_only"] is True
    assert annotation["affects_ranking"] is False
    assert rank_multiplier(annotation) == 1.0
    assert set(annotation["score"]) == {
        "valence",
        "arousal",
        "tension",
        "confidence",
        "response_tendency",
        "growth_delta",
    }


def test_annotation_requires_bound_provenance():
    annotation, error = build_shadow_annotation(
        bucket_id="bucket-1",
        source_digest="not-a-digest",
        source_kind="lmc5_candidate",
        source_run_id="lmc5-night-20260728",
        provider="provider",
        scorer="scorer",
        model="model",
        rubric_version="v1",
        run_id="run-1",
        trigger_reason="type.preference",
        score=_score(),
    )
    assert annotation is None and error == "schema.source_digest"


@pytest.mark.parametrize(
    ("source_kind", "source_run_id"),
    [
        ("contains space", "lmc5-night-20260728"),
        ("lmc5_candidate", "contains space"),
        ("", "lmc5-night-20260728"),
        ("lmc5_candidate", ""),
        ("x" * 161, "lmc5-night-20260728"),
        ("lmc5_candidate", "x" * 161),
    ],
)
def test_annotation_requires_machine_source_provenance(
    source_kind,
    source_run_id,
):
    annotation, error = build_shadow_annotation(
        bucket_id="bucket-1",
        source_digest="a" * 64,
        source_kind=source_kind,
        source_run_id=source_run_id,
        provider="provider",
        scorer="scorer",
        model="model",
        rubric_version="v1",
        run_id="run-1",
        trigger_reason="type.preference",
        score=_score(),
    )
    assert annotation is None and error == "schema.provenance"


def test_shadow_store_is_fsynced_idempotent_and_private(tmp_path):
    annotation, _ = _annotation()
    path = tmp_path / ".axis" / "e-shadow.jsonl"
    store = EAxisShadowStore(path)

    assert store.append(annotation) is True
    assert store.append(annotation) is False
    assert store.load() == [annotation]
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert store.lock_path.exists()


def test_shadow_success_is_idempotent_across_runs(tmp_path):
    annotation, _ = _annotation()
    store = EAxisShadowStore(tmp_path / "e-shadow.jsonl")
    assert store.append(annotation) is True

    replay = copy.deepcopy(annotation)
    replay["run_id"] = "run-2"
    replay["scored_at"] = "2026-07-29T00:00:00+00:00"
    assert store.append(replay) is False
    assert store.load() == [annotation]


def test_shadow_store_reordered_object_keys_are_the_same_normalized_row(
    tmp_path,
):
    annotation, _ = _annotation()
    path = tmp_path / "e-shadow.jsonl"
    store = EAxisShadowStore(path)
    assert store.append(annotation) is True
    before = path.read_bytes()

    reordered = dict(reversed(list(annotation.items())))
    reordered["score"] = dict(reversed(list(annotation["score"].items())))
    assert store.append(reordered) is False
    assert path.read_bytes() == before
    assert store.load() == [annotation]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row["score"].update(valence=-0.75),
        lambda row: row.update(status="failed", category="provider.error"),
        lambda row: row.update(category="provider.error"),
        lambda row: row.update(bucket_id="bucket-2"),
        lambda row: row.update(source_digest="b" * 64),
        lambda row: row.update(source_kind="manual_bucket"),
        lambda row: row.update(source_run_id="manual-run-2"),
        lambda row: row.update(provider="provider-b"),
        lambda row: row.update(scorer="e-shadow-v3"),
        lambda row: row.update(model="model-b"),
        lambda row: row.update(rubric_version="rubric-2"),
        lambda row: row.update(contract_version=3),
        lambda row: row.update(shadow_only=False),
        lambda row: row.update(affects_ranking=True),
        lambda row: row.update(extra="not-allowed"),
    ],
)
def test_shadow_store_rejects_any_changed_row_for_existing_key(tmp_path, mutate):
    annotation, _ = _annotation()
    path = tmp_path / "e-shadow.jsonl"
    store = EAxisShadowStore(path)
    assert store.append(annotation) is True
    before = path.read_bytes()

    changed = copy.deepcopy(annotation)
    mutate(changed)
    with pytest.raises(
        EAxisShadowConflictError,
        match="annotation_key conflict",
    ):
        store.append(changed)
    assert path.read_bytes() == before
    assert store.load() == [annotation]


def test_shadow_failure_replay_is_per_run_and_idempotent_within_run(tmp_path):
    first = build_failure_record(
        bucket_id="bucket-1",
        source_digest="a" * 64,
        source_kind="lmc5_candidate",
        source_run_id="lmc5-night-20260728",
        provider="provider",
        scorer="scorer",
        model="model",
        rubric_version="v1",
        run_id="run-1",
        trigger_reason="keyword.emotion",
        category="schema.missing",
        retryable=False,
    )
    path = tmp_path / "e-shadow.jsonl"
    store = EAxisShadowStore(path)
    assert store.append(first) is True
    assert store.append(dict(reversed(list(first.items())))) is False

    replay = copy.deepcopy(first)
    replay["scored_at"] = "1900-01-01T00:00:00+00:00"
    assert store.append(replay) is False

    second = build_failure_record(
        bucket_id="bucket-1",
        source_digest="a" * 64,
        source_kind="lmc5_candidate",
        source_run_id="lmc5-night-20260728",
        provider="provider",
        scorer="scorer",
        model="model",
        rubric_version="v1",
        run_id="run-2",
        trigger_reason="keyword.emotion",
        category="schema.missing",
        retryable=False,
        scored_at="2026-07-29T00:00:00+00:00",
    )
    assert second["annotation_key"] != first["annotation_key"]
    assert store.append(second) is True

    for field, changed_value in (("category", "schema.unexpected"),):
        changed = copy.deepcopy(first)
        changed[field] = changed_value
        with pytest.raises(
            EAxisShadowConflictError,
            match="annotation_key conflict",
        ):
            store.append(changed)
    assert store.load() == [first, second]


def test_shadow_store_corruption_fails_closed(tmp_path):
    path = tmp_path / "e-shadow.jsonl"
    annotation, _ = _annotation()
    path.write_text(
        json.dumps(annotation, ensure_ascii=False) + "\nnot-json\n",
        encoding="utf-8",
    )
    store = EAxisShadowStore(path)

    with pytest.raises(ValueError, match="corrupt E shadow ledger"):
        store.load()
    with pytest.raises(ValueError, match="corrupt E shadow ledger"):
        store.append(annotation)
    assert path.read_text(encoding="utf-8").endswith("not-json\n")


def test_shadow_store_conflicting_duplicate_key_is_corruption(tmp_path):
    annotation, _ = _annotation()
    changed = copy.deepcopy(annotation)
    changed["score"]["tension"] = 0.9
    path = tmp_path / "e-shadow.jsonl"
    original = (
        json.dumps(annotation, ensure_ascii=False)
        + "\n"
        + json.dumps(changed, ensure_ascii=False)
        + "\n"
    )
    path.write_text(original, encoding="utf-8")
    store = EAxisShadowStore(path)

    with pytest.raises(
        ValueError,
        match="corrupt E shadow ledger: conflicting annotation_key",
    ):
        store.load()
    with pytest.raises(
        ValueError,
        match="corrupt E shadow ledger: conflicting annotation_key",
    ):
        store.append(annotation)
    assert path.read_text(encoding="utf-8") == original


def test_shadow_store_identical_duplicate_key_is_corruption(tmp_path):
    annotation, _ = _annotation()
    path = tmp_path / "e-shadow.jsonl"
    raw = json.dumps(annotation, ensure_ascii=False) + "\n"
    path.write_text(raw + raw, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="corrupt E shadow ledger: duplicate annotation_key",
    ):
        EAxisShadowStore(path).load()


@pytest.mark.parametrize(
    "raw",
    [
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":-Infinity}',
        '{"a":1e400}',
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite_numbers(raw):
    with pytest.raises(ValueError):
        strict_json_loads(raw)


def test_huge_integer_is_rejected_without_overflow():
    normalized, error = validate_shadow_score(_score(valence=10 ** 400))
    assert normalized is None
    assert error == "schema.valence"


@pytest.mark.parametrize(
    "value",
    [True, float("nan"), float("inf"), 10 ** 400, -0.1, 1.1, "0.3"],
)
def test_min_confidence_must_be_finite_number_in_range(value):
    assert normalize_min_confidence(value) is None
    normalized, error = validate_shadow_score(_score(), min_confidence=value)
    assert normalized is None
    assert error == "config.min_confidence"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.pop("contract_version"),
        lambda row: row.update(contract_version=3),
        lambda row: row.update(extra="not-allowed"),
        lambda row: row.update(shadow_only=False),
        lambda row: row.update(annotation_key="0" * 64),
    ],
)
def test_shadow_store_requires_exact_versioned_row_contract(tmp_path, mutate):
    annotation, _ = _annotation()
    mutate(annotation)
    path = tmp_path / "e-shadow.jsonl"
    path.write_text(json.dumps(annotation) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid E shadow ledger row"):
        EAxisShadowStore(path).load()


@pytest.mark.parametrize(
    "raw",
    [
        '{"contract_version":1,"contract_version":1}',
        '{"contract_version":1,"score":NaN}',
        '{"contract_version":1,"score":1e400}',
    ],
)
def test_shadow_store_strictly_parses_raw_ledger_json(tmp_path, raw):
    path = tmp_path / "e-shadow.jsonl"
    path.write_text(raw + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt E shadow ledger"):
        EAxisShadowStore(path).load()


def test_shadow_store_load_rejects_broken_symlink_instead_of_seeing_empty(
    tmp_path,
):
    path = tmp_path / ".axis" / "e-shadow.jsonl"
    path.parent.mkdir()
    path.symlink_to(tmp_path / "outside-missing.jsonl")

    with pytest.raises(ValueError, match="unsafe E shadow storage"):
        EAxisShadowStore(path, maintenance_root=tmp_path).load()


def test_failure_record_stores_no_payload_or_content():
    row = build_failure_record(
        bucket_id="bucket-1",
        source_digest="a" * 64,
        source_kind="lmc5_candidate",
        source_run_id="lmc5-night-20260728",
        provider="provider",
        scorer="scorer",
        model="model",
        rubric_version="v1",
        run_id="run-1",
        trigger_reason="type.preference",
        category="schema.missing",
        retryable=False,
    )
    assert row["status"] == "failed"
    assert row["contract_version"] == CONTRACT_VERSION
    assert row["shadow_only"] is True
    assert row["affects_ranking"] is False
    assert "score" not in row and "content" not in row and "raw" not in row


@pytest.mark.parametrize(
    ("source_kind", "source_run_id"),
    [
        ("contains space", "lmc5-night-20260728"),
        ("lmc5_candidate", "contains space"),
    ],
)
def test_failure_record_requires_machine_source_provenance(
    source_kind,
    source_run_id,
):
    with pytest.raises(ValueError, match="machine text"):
        build_failure_record(
            bucket_id="bucket-1",
            source_digest="a" * 64,
            source_kind=source_kind,
            source_run_id=source_run_id,
            provider="provider",
            scorer="scorer",
            model="model",
            rubric_version="v1",
            run_id="run-1",
            trigger_reason="type.preference",
            category="schema.missing",
            retryable=False,
        )


class _RawRequest:
    def __init__(self, raw: str):
        self.raw = raw

    async def body(self):
        return self.raw.encode("utf-8")


class _ReadOnlyBucketManager:
    def __init__(self, content: str):
        self.bucket = {
            "id": "bucket-1",
            "content": content,
            "metadata": {
                "last_active": "2026-07-27T00:00:00+00:00",
                "valence": 0.9,
                "arousal": 0.1,
            },
        }
        self.mutations = []

    async def get(self, bucket_id):
        return self.bucket if bucket_id == "bucket-1" else None

    async def create(self, *args, **kwargs):
        self.mutations.append(("create", args, kwargs))
        raise AssertionError("E shadow route must not create memory")

    async def update(self, *args, **kwargs):
        self.mutations.append(("update", args, kwargs))
        raise AssertionError("E shadow route must not update memory")

    async def touch(self, *args, **kwargs):
        self.mutations.append(("touch", args, kwargs))
        raise AssertionError("E shadow route must not touch memory")

    async def archive(self, *args, **kwargs):
        self.mutations.append(("archive", args, kwargs))
        raise AssertionError("E shadow route must not archive memory")


def _route_payload(content: str):
    return {
        "bucket_id": "bucket-1",
        "source_digest": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "provider": "provider-a",
        "scorer": "e-shadow-v2",
        "model": "model-a",
        "rubric_version": "rubric-1",
        "run_id": "manual-run-1",
        "trigger_reason": "manual.api",
        "score": _score(),
    }


async def _call_shadow_route(monkeypatch, tmp_path, raw, *, min_confidence=0.3):
    if "jieba" not in sys.modules:
        fake_jieba = ModuleType("jieba")
        fake_jieba.cut = lambda text: tuple(str(text))
        fake_jieba.initialize = lambda: None
        monkeypatch.setitem(sys.modules, "jieba", fake_jieba)
    if "rapidfuzz" not in sys.modules:
        fake_rapidfuzz = ModuleType("rapidfuzz")
        fake_rapidfuzz.fuzz = SimpleNamespace(
            ratio=lambda *_args, **_kwargs: 0,
            partial_ratio=lambda *_args, **_kwargs: 0,
            token_set_ratio=lambda *_args, **_kwargs: 0,
        )
        monkeypatch.setitem(sys.modules, "rapidfuzz", fake_rapidfuzz)
    if "openai" not in sys.modules:
        fake_openai = ModuleType("openai")
        class _FakeOpenAI:
            def __init__(self, *_args, **_kwargs):
                pass
        fake_openai.OpenAI = _FakeOpenAI
        fake_openai.AsyncOpenAI = _FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
    import server

    async def _inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    manager = _ReadOnlyBucketManager("current memory")
    store = EAxisShadowStore(tmp_path / ".axis" / "e-shadow.jsonl")
    before = copy.deepcopy(manager.bucket)
    monkeypatch.setenv("OMBRE_API_TOKEN", "test-token")
    # Keep direct route tests deterministic; store concurrency is covered by
    # the real append/load tests above.
    monkeypatch.setattr(server.asyncio, "to_thread", _inline_to_thread)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "_get_e_axis_shadow_store", lambda: store)
    monkeypatch.setitem(
        server.config,
        "e_axis_shadow",
        {"enabled": True, "min_confidence": min_confidence},
    )

    response = await server.api_e_axis_shadow(_RawRequest(raw))
    body = json.loads(bytes(response.body).decode("utf-8"))
    assert manager.bucket == before
    assert manager.mutations == []
    return response.status_code, body, store


def test_shadow_route_is_read_only_and_duplicate_write_is_idempotent(
    monkeypatch,
    tmp_path,
):
    payload = _route_payload("current memory")
    raw = json.dumps(payload, separators=(",", ":"))

    status, body, store = asyncio.run(
        _call_shadow_route(monkeypatch, tmp_path, raw)
    )
    assert status == 200
    assert body["added"] is True
    assert body["memory_mutated"] is False

    status, body, _ = asyncio.run(
        _call_shadow_route(monkeypatch, tmp_path, raw)
    )
    assert status == 200
    assert body["added"] is False
    rows = store.load()
    assert len(rows) == 1
    assert rows[0]["status"] == "success"


def test_shadow_route_rejects_stale_digest_without_memory_mutation(
    monkeypatch,
    tmp_path,
):
    payload = _route_payload("stale memory")
    status, body, store = asyncio.run(
        _call_shadow_route(
            monkeypatch,
            tmp_path,
            json.dumps(payload),
        )
    )
    assert status == 409
    assert "does not match" in body["error"]
    rows = store.load()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["category"] == "source_digest.mismatch"


@pytest.mark.parametrize(
    "raw",
    [
        (
            '{"bucket_id":"bucket-1","bucket_id":"bucket-2",'
            '"source_digest":"' + "a" * 64 + '","scorer":"s","model":"m",'
            '"rubric_version":"r","score":{}}'
        ),
        (
            '{"bucket_id":"bucket-1","source_digest":"' + "a" * 64 + '",'
            '"scorer":"s","model":"m","rubric_version":"r",'
            '"score":{"valence":0,"valence":1,"arousal":0,"tension":0,'
            '"confidence":1,"response_tendency":"engage","growth_delta":"stable"}}'
        ),
        (
            '{"bucket_id":"bucket-1","source_digest":"' + "a" * 64 + '",'
            '"scorer":"s","model":"m","rubric_version":"r",'
            '"score":{"valence":NaN,"arousal":0,"tension":0,"confidence":1,'
            '"response_tendency":"engage","growth_delta":"stable"}}'
        ),
        (
            '{"bucket_id":"bucket-1","source_digest":"' + "a" * 64 + '",'
            '"scorer":"s","model":"m","rubric_version":"r",'
            '"score":{"valence":1e400,"arousal":0,"tension":0,"confidence":1,'
            '"response_tendency":"engage","growth_delta":"stable"}}'
        ),
    ],
)
def test_shadow_route_rejects_non_strict_raw_json(monkeypatch, tmp_path, raw):
    status, body, store = asyncio.run(
        _call_shadow_route(
            monkeypatch,
            tmp_path,
            raw,
        )
    )
    assert status == 400
    assert body["error"] == "invalid JSON"
    assert store.load() == []


@pytest.mark.parametrize("min_confidence", [float("nan"), float("inf"), 10 ** 400])
def test_shadow_route_fails_closed_on_invalid_min_confidence(
    monkeypatch,
    tmp_path,
    min_confidence,
):
    payload = _route_payload("current memory")
    status, body, store = asyncio.run(
        _call_shadow_route(
            monkeypatch,
            tmp_path,
            json.dumps(payload),
            min_confidence=min_confidence,
        )
    )
    assert status == 503
    assert body["error"] == "invalid E shadow min_confidence config"
    assert store.load() == []
