"""Dependency-light fake-provider smoke for E0 shadow collection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e_axis_night import (  # noqa: E402
    EAxisRunJournal,
    StrictEAxisScorer,
    run_e_axis_shadow,
)
from e_axis_shadow import EAxisShadowStore  # noqa: E402


def candidate():
    base_digest = hashlib.sha256(b"preference").hexdigest()
    payload = json.dumps({
        "axis": "X",
        "base_digest": base_digest,
        "draft": {
            "type": "preference",
            "title": "response preference",
            "content": "I prefer direct answers.",
            "relation_hints": [],
        },
        "origin_run_id": "lmc5-night-smoke",
        "schema": "ombre.lmc5-axis-candidate/v1",
        "source": {"created_at": "2026-07-31T00:00:00+00:00"},
    }, sort_keys=True, separators=(",", ":")).encode()
    return SimpleNamespace(candidate_id=1, axis="X", payload=payload)


class Ledger:
    def list_candidates(self, status, *, limit, after=None):
        if status == "pending" and not after:
            return (candidate(),)
        return ()


class Provider:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        if isinstance(self.output, BaseException):
            raise self.output
        return self.output


def envelope(content):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": content},
        }]
    }


def scorer(provider):
    return StrictEAxisScorer(
        provider,
        provider_name="fake-provider",
        model="fake-model",
        scorer_name="fake-scorer",
        rubric_version="fake-rubric",
    )


async def smoke():
    good = Provider(envelope(json.dumps({
        "valence": 0.4,
        "arousal": 0.3,
        "tension": 0.2,
        "confidence": 0.9,
        "response_tendency": "engage",
        "growth_delta": "stable",
    })))
    with tempfile.TemporaryDirectory(prefix="e-axis-smoke-") as raw:
        root = Path(raw)
        store = EAxisShadowStore(
            root / ".axis" / "e-shadow.jsonl",
            maintenance_root=root,
        )
        journal = EAxisRunJournal(
            root / ".axis",
            maintenance_root=root,
        )
        first = await run_e_axis_shadow(
            ledger=Ledger(),
            store=store,
            journal=journal,
            scorer=scorer(good),
            run_id="smoke-success",
        )
        assert first.added == 1 and first.failed == 0
        assert good.calls == 1
        assert store.load()[0]["affects_ranking"] is False

    for name, output in (
        ("empty", envelope("")),
        ("timeout", TimeoutError()),
    ):
        with tempfile.TemporaryDirectory(prefix=f"e-axis-{name}-") as raw:
            root = Path(raw)
            store = EAxisShadowStore(
                root / ".axis" / "e-shadow.jsonl",
                maintenance_root=root,
            )
            journal = EAxisRunJournal(
                root / ".axis",
                maintenance_root=root,
            )
            result = await run_e_axis_shadow(
                ledger=Ledger(),
                store=store,
                journal=journal,
                scorer=scorer(Provider(output)),
                run_id=f"smoke-{name}",
            )
            assert result.failed_retryable == 1
            assert store.load()[0]["status"] == "failed"


if __name__ == "__main__":
    asyncio.run(smoke())
    print("E_AXIS_NIGHT_SMOKE_OK")
