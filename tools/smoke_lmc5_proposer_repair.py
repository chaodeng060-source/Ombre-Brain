#!/usr/bin/env python3
"""Dependency-light smoke test for bounded proposer evidence repair."""

from __future__ import annotations

import asyncio
import json

from lmc5_proposer import (
    ProposerChunk,
    ProposerContractError,
    StrictOmbreProposer,
)


CHUNKS = (
    ProposerChunk(
        "chunk-1",
        "朝灯明确喜欢雨声，也决定采用严格适配器。",
    ),
)


def _candidate(*, evidence: str, importance: int = 7) -> dict:
    return {
        "type": "preference",
        "title": "雨声偏好",
        "content": "朝灯喜欢雨声。",
        "importance": importance,
        "thread_hint": "声音偏好",
        "relation_hints": [],
        "source_chunk_ids": ["chunk-1"],
        "evidence": evidence,
        "risk": "normal",
    }


def _document(*candidates: dict) -> str:
    return json.dumps(
        {"schema_version": 1, "candidates": list(candidates)},
        ensure_ascii=False,
    )


def _envelope(content: str) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content},
            }
        ]
    }


async def _repair_succeeds_once() -> None:
    prompts: list[str] = []
    responses = iter(
        (
            _document(_candidate(evidence="模型改写的雨声证据")),
            _document(_candidate(evidence="喜欢雨声")),
        )
    )

    async def provider(prompt: str) -> dict:
        prompts.append(prompt)
        return _envelope(next(responses))

    batch = await StrictOmbreProposer(provider).propose(CHUNKS)
    assert batch.candidates[0].evidence == "喜欢雨声"
    assert len(prompts) == 2
    assert "REPAIR ONE PROVENANCE ERROR" not in prompts[0]
    assert "REPAIR ONE PROVENANCE ERROR" in prompts[1]


async def _repair_stops_after_second_failure() -> None:
    prompts: list[str] = []
    content = _document(_candidate(evidence="模型改写的雨声证据"))

    async def provider(prompt: str) -> dict:
        prompts.append(prompt)
        return _envelope(content)

    try:
        await StrictOmbreProposer(provider).propose(CHUNKS)
    except ProposerContractError as exc:
        assert exc.code == "provenance_evidence"
    else:
        raise AssertionError("second invalid evidence response was accepted")
    assert len(prompts) == 2


async def _other_errors_do_not_retry() -> None:
    prompts: list[str] = []
    content = _document(_candidate(evidence="喜欢雨声", importance=11))

    async def provider(prompt: str) -> dict:
        prompts.append(prompt)
        return _envelope(content)

    try:
        await StrictOmbreProposer(provider).propose(CHUNKS)
    except ProposerContractError as exc:
        assert exc.code == "schema_candidate"
    else:
        raise AssertionError("invalid candidate schema was accepted")
    assert len(prompts) == 1


async def main() -> None:
    await _repair_succeeds_once()
    await _repair_stops_after_second_failure()
    await _other_errors_do_not_retry()


if __name__ == "__main__":
    asyncio.run(main())
    print("LMC5_PROPOSER_REPAIR_SMOKE_OK")
