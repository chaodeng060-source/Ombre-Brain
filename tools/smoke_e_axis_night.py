"""Dependency-light smoke for the production E-axis shadow job."""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
from pathlib import Path

from e_axis_night import (
    EAxisNightError,
    StrictEAxisScorer,
    _eligible_lmc5_bucket,
    run_e_axis_shadow,
)
from e_axis_shadow import EAxisShadowStore


def bucket(bucket_id: str, content: str, *, ordinary: bool = False) -> dict:
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "name": bucket_id,
            "created": "2026-07-29T10:00:00+00:00",
            "tags": (
                ["ordinary"] if ordinary else ["lmc5", "night", "event"]
            ),
            "curated_write_key": "lmc5-x:v1:" + "b" * 64,
            "curated_payload_sha256": "c" * 64,
            "vector_policy": "required",
            "lmc5_recall_state": "ready_vector",
            "x_provenance": {
                "source_kind": "conversation",
                "source_session": "smoke",
                "source_event_ids": ["event-1"],
                "source_digest": "a" * 64,
            },
        },
    }


class Buckets:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = copy.deepcopy(rows)

    async def list_all(self, **kwargs):
        assert kwargs == {
            "include_archive": False,
            "include_nsfw": False,
        }
        return copy.deepcopy(self.rows)


class Provider:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> dict:
        self.calls += 1
        assert "Do not infer hidden motives" in prompt
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "valence": 0.4,
                                "arousal": 0.3,
                                "tension": 0.2,
                                "confidence": 0.9,
                                "response_tendency": "engage",
                                "growth_delta": "stable",
                            }
                        )
                    },
                }
            ]
        }


async def smoke() -> None:
    lmc = bucket("lmc", "redacted derived event")
    ordinary = bucket("ordinary", "private ordinary memory", ordinary=True)
    assert _eligible_lmc5_bucket(lmc)
    assert not _eligible_lmc5_bucket(ordinary)

    provider = Provider()
    scorer = StrictEAxisScorer(
        provider,
        model="smoke-model",
        scorer_name="smoke-scorer",
        rubric_version="smoke-rubric-v1",
    )
    with tempfile.TemporaryDirectory(prefix="e-axis-smoke-") as raw:
        store = EAxisShadowStore(
            Path(raw) / ".axis" / "e-shadow.jsonl",
            maintenance_root=raw,
        )
        manager = Buckets([ordinary, lmc])
        first = await run_e_axis_shadow(
            bucket_manager=manager,
            store=store,
            scorer=scorer,
        )
        assert first.eligible == 1
        assert first.attempted == 1
        assert first.added == 1
        assert first.failed == 0
        assert provider.calls == 1
        assert manager.rows == [ordinary, lmc]
        rows = store.load()
        assert len(rows) == 1
        assert rows[0]["bucket_id"] == "lmc"
        assert rows[0]["shadow_only"] is True
        assert rows[0]["affects_ranking"] is False

        second = await run_e_axis_shadow(
            bucket_manager=manager,
            store=store,
            scorer=scorer,
        )
        assert second.attempted == 0
        assert second.existing == 1
        assert provider.calls == 1

    incomplete = StrictEAxisScorer(
        lambda _: {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "{}"},
                }
            ]
        },
        model="smoke-model",
    )
    try:
        incomplete.score(title="title", content="content")
    except EAxisNightError as exc:
        assert exc.code == "provider.incomplete"
    else:
        raise AssertionError("incomplete provider output was accepted")


if __name__ == "__main__":
    asyncio.run(smoke())
    print("E_AXIS_NIGHT_SMOKE_OK")
