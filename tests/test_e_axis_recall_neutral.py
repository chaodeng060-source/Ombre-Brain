"""Static and contract guards that keep E0 outside recall."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from e_axis_shadow import (
    EAxisShadowStore,
    build_shadow_annotation,
    rank_multiplier,
)


ROOT = Path(__file__).resolve().parents[1]


def _function_source(path: Path, name: str) -> str:
    raw = path.read_text(encoding="utf-8")
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return ast.get_source_segment(raw, node) or ""
    raise AssertionError(f"{name} not found")


def test_breath_does_not_read_e_sidecars_or_apply_e_multiplier():
    breath = _function_source(ROOT / "server.py", "breath")
    lowered = breath.lower()
    assert "e_axis_shadow" not in lowered
    assert "e-shadow" not in lowered
    assert "rank_multiplier" not in lowered


def test_e_annotation_cannot_change_ranking(tmp_path):
    row, error = build_shadow_annotation(
        bucket_id="candidate:" + "a" * 64,
        source_digest=hashlib.sha256(b"payload").hexdigest(),
        source_kind="lmc5_candidate",
        source_run_id="lmc5-night-1",
        provider="fake-provider",
        scorer="fake-scorer",
        model="fake-model",
        rubric_version="fake-rubric",
        run_id="e-run-1",
        trigger_reason="type.preference",
        score={
            "valence": -1.0,
            "arousal": 1.0,
            "tension": 1.0,
            "confidence": 1.0,
            "response_tendency": "alert",
            "growth_delta": "setback",
        },
    )
    assert error is None
    store = EAxisShadowStore(
        tmp_path / ".axis" / "e-shadow.jsonl",
        maintenance_root=tmp_path,
    )
    assert store.append(row)
    assert rank_multiplier(None) == 1.0
    assert rank_multiplier(store.load()[0]) == 1.0
