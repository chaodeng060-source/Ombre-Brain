from __future__ import annotations

from pathlib import Path

import server
from utils import load_config


class _RecordingLoop:
    def __init__(self) -> None:
        self.created = 0

    def create_task(self, coroutine):
        self.created += 1
        coroutine.close()
        return object()


def test_startup_relation_backfill_defaults_off(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
    monkeypatch.delenv("OMBRE_STARTUP_RELATION_BACKFILL", raising=False)

    loaded = load_config(str(tmp_path / "missing-config.yaml"))

    assert loaded["maintenance"]["startup_relation_backfill"] is False


def test_startup_relation_backfill_env_requires_explicit_truthy_value(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
    monkeypatch.setenv("OMBRE_STARTUP_RELATION_BACKFILL", "true")

    loaded = load_config(str(tmp_path / "missing-config.yaml"))

    assert loaded["maintenance"]["startup_relation_backfill"] is True


def test_startup_relation_backfill_can_be_enabled_in_yaml(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
    monkeypatch.delenv("OMBRE_STARTUP_RELATION_BACKFILL", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "maintenance:\n  startup_relation_backfill: true\n",
        encoding="utf-8",
    )

    loaded = load_config(str(config_path))

    assert loaded["maintenance"]["startup_relation_backfill"] is True


def test_startup_relation_backfill_env_can_force_yaml_off(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
    monkeypatch.setenv("OMBRE_STARTUP_RELATION_BACKFILL", "0")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "maintenance:\n  startup_relation_backfill: true\n",
        encoding="utf-8",
    )

    loaded = load_config(str(config_path))

    assert loaded["maintenance"]["startup_relation_backfill"] is False


def test_maybe_start_backfill_is_noop_by_default(monkeypatch) -> None:
    monkeypatch.setitem(
        server.config,
        "maintenance",
        {"startup_relation_backfill": False},
    )
    monkeypatch.setattr(server, "_backfill_started", False)

    def _unexpected_loop_lookup():
        raise AssertionError("disabled backfill must not touch the event loop")

    monkeypatch.setattr(server.asyncio, "get_running_loop", _unexpected_loop_lookup)

    server._maybe_start_backfill()
    server._maybe_start_backfill()

    assert server._backfill_started is False


def test_explicit_startup_backfill_is_scheduled_once(monkeypatch) -> None:
    loop = _RecordingLoop()
    monkeypatch.setitem(
        server.config,
        "maintenance",
        {"startup_relation_backfill": True},
    )
    monkeypatch.setattr(server, "_backfill_started", False)
    monkeypatch.setattr(server.asyncio, "get_running_loop", lambda: loop)

    server._maybe_start_backfill()
    server._maybe_start_backfill()

    assert loop.created == 1
    assert server._backfill_started is True
    assert callable(server.backfill_relations)
