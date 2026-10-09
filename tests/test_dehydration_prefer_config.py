from __future__ import annotations

from pathlib import Path

import yaml

from utils import load_config


def _write_config(path: Path, *, prefer_config: bool) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "dehydration": {
                    "prefer_config": prefer_config,
                    "api_key": "file-key",
                    "base_url": "https://file.example.invalid/v1",
                    "model": "file-model",
                }
            }
        ),
        encoding="utf-8",
    )


def _set_environment(monkeypatch) -> None:
    monkeypatch.setenv("OMBRE_API_KEY", "environment-key")
    monkeypatch.setenv("OMBRE_BASE_URL", "https://environment.example.invalid/v1")
    monkeypatch.setenv("OMBRE_MODEL", "environment-model")


def test_environment_overrides_remain_default(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, prefer_config=False)
    _set_environment(monkeypatch)

    config = load_config(str(config_path))

    assert config["dehydration"]["api_key"] == "environment-key"
    assert config["dehydration"]["base_url"] == "https://environment.example.invalid/v1"
    assert config["dehydration"]["model"] == "environment-model"


def test_explicit_prefer_config_uses_file_provider_settings(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, prefer_config=True)
    _set_environment(monkeypatch)

    config = load_config(str(config_path))

    assert config["dehydration"]["api_key"] == "file-key"
    assert config["dehydration"]["base_url"] == "https://file.example.invalid/v1"
    assert config["dehydration"]["model"] == "file-model"
