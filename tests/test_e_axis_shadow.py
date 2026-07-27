"""Strict E-axis shadow contract tests."""

import hashlib
import os

import pytest

from e_axis_shadow import (
    EAxisShadowStore,
    build_failure_record,
    build_shadow_annotation,
    rank_multiplier,
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
        scorer="e-shadow-v1",
        model="model-a",
        rubric_version="rubric-1",
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
        scorer="scorer",
        model="model",
        rubric_version="v1",
        score=_score(),
    )
    assert annotation is None and error == "schema.source_digest"


def test_shadow_store_is_fsynced_idempotent_and_private(tmp_path):
    annotation, _ = _annotation()
    path = tmp_path / ".axis" / "e-shadow.jsonl"
    store = EAxisShadowStore(path)

    assert store.append(annotation) is True
    assert store.append(annotation) is False
    assert store.load() == [annotation]
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_shadow_store_corruption_fails_closed(tmp_path):
    path = tmp_path / "e-shadow.jsonl"
    path.write_text('{"annotation_key":"ok"}\nnot-json\n', encoding="utf-8")
    store = EAxisShadowStore(path)
    annotation, _ = _annotation()

    with pytest.raises(ValueError, match="corrupt E shadow ledger"):
        store.load()
    with pytest.raises(ValueError, match="corrupt E shadow ledger"):
        store.append(annotation)
    assert path.read_text(encoding="utf-8").endswith("not-json\n")


def test_failure_record_stores_no_payload_or_content():
    row = build_failure_record(
        bucket_id="bucket-1",
        source_digest="a" * 64,
        scorer="scorer",
        model="model",
        rubric_version="v1",
        category="schema.missing",
    )
    assert row["status"] == "failed"
    assert row["shadow_only"] is True
    assert row["affects_ranking"] is False
    assert "score" not in row and "content" not in row and "raw" not in row
