#!/usr/bin/env python3
"""Dependency-light smoke for dedicated LMC-5 proposer controls."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import night_run_runtime as runtime_module
from night_run_runtime import (
    NightRunRuntime,
    NightRunRuntimeError,
    _proposer_max_tokens,
    _proposer_temperature,
    build_night_run_runtime,
)


def _expect_temperature_error(section: object) -> None:
    try:
        _proposer_temperature({"lmc5_night": section})
    except NightRunRuntimeError as exc:
        assert exc.code == "provider.temperature_invalid"
    else:
        raise AssertionError("unsafe proposer temperature was accepted")


def _assert_runtime_wiring() -> None:
    captured: dict[str, Any] = {}

    class ProviderSpy:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    class CoordinatorStub:
        def __init__(self, **_: Any) -> None:
            self.maintenance_barrier = object()
            self.policy = SimpleNamespace(barrier_timeout_seconds=1.0)

    originals = {
        "CuratedWriteCoordinator": runtime_module.CuratedWriteCoordinator,
        "NightRunCoordinator": runtime_module.NightRunCoordinator,
        "OpenAIChatProvider": runtime_module.OpenAIChatProvider,
        "SnapshotManager": runtime_module.SnapshotManager,
        "StrictOmbreProposer": runtime_module.StrictOmbreProposer,
    }
    prior_snapshot_root = os.environ.get("OMBRE_LMC5_SNAPSHOT_DIR")
    try:
        os.environ["OMBRE_LMC5_SNAPSHOT_DIR"] = "/tmp/lmc5-smoke-snapshots"
        runtime_module.OpenAIChatProvider = ProviderSpy
        runtime_module.StrictOmbreProposer = lambda provider, **_: provider
        runtime_module.SnapshotManager = lambda *_: object()
        runtime_module.CuratedWriteCoordinator = lambda *_: object()
        runtime_module.NightRunCoordinator = CoordinatorStub
        runtime = build_night_run_runtime(
            config={
                "buckets_dir": "/data",
                "dehydration": {
                    "api_key": "test-key",
                    "temperature": 0.8,
                },
            },
            ledger=object(),
            bucket_manager=object(),
            embedding_engine=object(),
            decay_engine=object(),
            consolidation_engine=object(),
        )
    finally:
        for name, value in originals.items():
            setattr(runtime_module, name, value)
        if prior_snapshot_root is None:
            os.environ.pop("OMBRE_LMC5_SNAPSHOT_DIR", None)
        else:
            os.environ["OMBRE_LMC5_SNAPSHOT_DIR"] = prior_snapshot_root
    assert isinstance(runtime, NightRunRuntime)
    assert captured["max_tokens"] == 4096
    assert captured["temperature"] == 0.0


def main() -> None:
    assert _proposer_max_tokens(
        {"dehydration": {"max_tokens": 128}}
    ) == 4096
    assert _proposer_temperature(
        {"dehydration": {"temperature": 0.8}}
    ) == 0.0
    assert _proposer_temperature(
        {"lmc5_night": {"proposer_temperature": 0.25}}
    ) == 0.25
    for invalid in (
        [],
        {"proposer_temperature": None},
        {"proposer_temperature": True},
        {"proposer_temperature": -0.01},
        {"proposer_temperature": 1.01},
        {"proposer_temperature": float("nan")},
        {"proposer_temperature": "0"},
    ):
        _expect_temperature_error(invalid)
    _assert_runtime_wiring()


if __name__ == "__main__":
    main()
    print("LMC5_RUNTIME_CONTROLS_SMOKE_OK")
