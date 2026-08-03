from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

import patrol_night
from utils import load_config


def _config(tmp_path: Path) -> Path:
    buckets = tmp_path / "buckets"
    for directory in ("permanent", "dynamic", "archive", "feel"):
        (buckets / directory).mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(
        f"buckets_dir: {buckets}\nfact_slots:\n  registry: {{}}\n",
        encoding="utf-8",
    )
    return path


def test_nightly_patrol_persists_report_and_success_status(tmp_path, monkeypatch):
    config = _config(tmp_path)
    state = tmp_path / "patrol-state"
    report = {"total": 3, "suggestions": [{"key": "m|one"}]}
    monkeypatch.setattr(patrol_night.patrol_module, "patrol", lambda *_a, **_k: report)
    monkeypatch.setattr(
        patrol_night.patrol_module,
        "render_md",
        lambda *_a, **_k: "# durable patrol report",
    )
    monkeypatch.setattr(
        patrol_night.patrol_module,
        "enqueue_metabolism_suggestions",
        lambda *_a, **_k: 1,
    )

    result = patrol_night.run_nightly_patrol(
        config,
        state,
        clock=lambda: datetime(2026, 8, 3, 4, 30, 0),
    )

    latest = json.loads((state / "latest.json").read_text(encoding="utf-8"))
    history = (state / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert result["ok"] is True
    assert result["bucket_count"] == 3
    assert result["queued_count"] == 1
    assert latest == result
    assert len(history) == 1 and json.loads(history[0])["ok"] is True
    assert (state / "latest.md").read_text(encoding="utf-8") == (
        "# durable patrol report\n"
    )
    assert Path(result["report"]).is_file()


def test_nightly_patrol_persists_failure_and_propagates(tmp_path, monkeypatch):
    config = _config(tmp_path)
    state = tmp_path / "patrol-state"

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected patrol failure")

    monkeypatch.setattr(patrol_night.patrol_module, "patrol", fail)
    with pytest.raises(RuntimeError, match="injected patrol failure"):
        patrol_night.run_nightly_patrol(
            config,
            state,
            clock=lambda: datetime(2026, 8, 3, 4, 30, 0),
        )

    latest = json.loads((state / "latest.json").read_text(encoding="utf-8"))
    assert latest["ok"] is False
    assert latest["error_type"] == "RuntimeError"
    assert not (state / "latest.md").exists()


def _fake_docker(tmp_path: Path) -> tuple[Path, Path]:
    calls = tmp_path / "docker.calls"
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "case \"$*\" in\n"
        "  *patrol_night.py*) exit \"${FAIL_PATROL:-0}\" ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o700)
    return fake, calls


def _run_cron(tmp_path: Path, *, fail_patrol: int = 0):
    fake, calls = _fake_docker(tmp_path)
    script = Path(__file__).resolve().parents[1] / "cron" / "run-lmc5-night.sh"
    env = {
        **os.environ,
        "DOCKER_BIN": str(fake),
        "CALL_LOG": str(calls),
        "FAIL_PATROL": str(fail_patrol),
        "OMBRE_LMC5_LOCK_FILE": str(tmp_path / "night.lock"),
    }
    result = subprocess.run(["sh", str(script)], env=env, check=False)
    return result, calls.read_text(encoding="utf-8").splitlines()


def test_cron_runs_night_then_patrol_only(tmp_path):
    result, calls = _run_cron(tmp_path)

    assert result.returncode == 0
    assert len(calls) == 2
    assert "night_run_trigger.py" in calls[0]
    assert "patrol_night.py" in calls[1]


def test_cron_propagates_patrol_failure(tmp_path):
    result, calls = _run_cron(tmp_path, fail_patrol=23)

    assert result.returncode == 23
    assert len(calls) == 2
    assert "patrol_night.py" in calls[1]


def test_default_config_has_a_real_constrained_fact_slot_registry(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
    config = load_config(str(tmp_path / "missing.yaml"))
    registry = config["fact_slots"]["registry"]

    assert "preference.ui.primary_color" in registry
    assert registry["preference.ui.primary_color"]["domains"] == ["创作"]
    assert registry["preference.ui.primary_color"]["types"] == ["dynamic"]
