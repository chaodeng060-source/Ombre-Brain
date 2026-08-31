from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _job_fixture(tmp_path: Path, *, cron_file: Path | None) -> tuple[Path, dict[str, str], Path, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    backup = repo / "scripts" / "backup_nas_data.sh"
    backup.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    backup.chmod(0o700)

    env_file = tmp_path / "production.env"
    values = [
        "OMBRE_CONTAINER_NAME=ombre-vps-mirror",
        f"OMBRE_JOB_ENV_FILE={env_file}",
        f"OMBRE_JOB_REPO_DIR={repo}",
        "OMBRE_JOB_CRON_USER=zhaodeng",
        "OMBRE_PUBLIC_HEALTH_URL=https://memory.zhaodeng.xyz/health",
    ]
    if cron_file is not None:
        values.append(f"OMBRE_JOB_CRON_FILE={cron_file}")
    env_file.write_text("\n".join((*values, "")), encoding="utf-8")

    state = tmp_path / "user-crontab"
    state.write_text("17 * * * * unrelated-safe-job\n", encoding="utf-8")
    calls = tmp_path / "crontab.calls"
    crontab = tmp_path / "crontab"
    crontab.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -l ]; then\n"
        "  printf 'list\\n' >>\"$CRONTAB_CALLS\"\n"
        "  if [ -n \"${CONCURRENT_LINE:-}\" ]; then\n"
        "    grep -Fqx \"$CONCURRENT_LINE\" \"$CRON_STATE\" || printf '%s\\n' \"$CONCURRENT_LINE\" >>\"$CRON_STATE\"\n"
        "  fi\n"
        "  cat \"$CRON_STATE\"; exit 0\n"
        "fi\n"
        "printf 'write\\n' >>\"$CRONTAB_CALLS\"\n"
        "cp \"$1\" \"$CRON_STATE\"\n",
        encoding="utf-8",
    )
    crontab.chmod(0o700)

    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = inspect ] && [ \"$4\" = ombre-brain ]; then\n"
        "  printf '%s\\n' \"${LEGACY_RUNNING:-false}\"; exit 0\n"
        "fi\n"
        "exit 91\n",
        encoding="utf-8",
    )
    docker.chmod(0o700)
    curl = tmp_path / "curl"
    curl.write_text("#!/bin/sh\nexit \"${PUBLIC_HEALTH_STATUS:-0}\"\n", encoding="utf-8")
    curl.chmod(0o700)
    command_env = {
        **os.environ,
        "CRONTAB_BIN": str(crontab),
        "CRON_STATE": str(state),
        "CRONTAB_CALLS": str(calls),
        "TMPDIR": str(tmp_path),
        "DOCKER_BIN": str(docker),
        "CURL_BIN": str(curl),
    }
    return env_file, command_env, state, calls


def _install(env_file: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / "install_nas_jobs.sh"), "--env-file", str(env_file)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("wrapper", ["run-lmc5-night.sh", "run-e-axis-shadow.sh"])
def test_writer_wrapper_rejects_the_legacy_container_before_docker(tmp_path: Path, wrapper: str) -> None:
    calls = tmp_path / "docker.calls"
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >>\"$CALL_LOG\"\n", encoding="utf-8")
    docker.chmod(0o700)

    result = subprocess.run(
        ["sh", str(ROOT / "cron" / wrapper)],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CALL_LOG": str(calls),
            "OMBRE_CONTAINER_NAME": "ombre-brain",
            "OMBRE_LMC5_LOCK_FILE": str(tmp_path / "night.lock"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing non-production container" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize("entrypoint", ["night_run_trigger.py", "patrol_night.py", "e_axis_night.py"])
def test_installer_rejects_direct_unmanaged_docker_writers(tmp_path: Path, entrypoint: str) -> None:
    cron_file = tmp_path / "cron.d" / "ombre-vps-mirror"
    cron_file.parent.mkdir()
    env_file, env, state, _ = _job_fixture(tmp_path, cron_file=cron_file)
    state.write_text(
        f"30 4 * * * /usr/bin/docker exec ombre-brain python /app/{entrypoint}\n",
        encoding="utf-8",
    )

    result = _install(env_file, env)

    assert result.returncode != 0
    assert "unmanaged Ombre writer/backup cron remains" in result.stderr
    assert not cron_file.exists()


def test_production_install_requires_an_independent_cron_drop_in(tmp_path: Path) -> None:
    env_file, env, state, calls = _job_fixture(tmp_path, cron_file=None)
    original = state.read_text(encoding="utf-8")

    result = _install(env_file, env)

    assert result.returncode != 0
    assert "independent cron drop-in" in result.stderr
    assert state.read_text(encoding="utf-8") == original
    assert not calls.exists() or "write" not in calls.read_text(encoding="utf-8")


def test_drop_in_install_never_rewrites_the_shared_user_crontab(tmp_path: Path) -> None:
    cron_file = tmp_path / "cron.d" / "ombre-vps-mirror"
    cron_file.parent.mkdir()
    env_file, env, state, calls = _job_fixture(tmp_path, cron_file=cron_file)
    env["CONCURRENT_LINE"] = "23 * * * * concurrent-safe-job"

    result = _install(env_file, env)

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == ["list"]
    assert state.read_text(encoding="utf-8").splitlines() == [
        "17 * * * * unrelated-safe-job",
        "23 * * * * concurrent-safe-job",
    ]
    installed = cron_file.read_text(encoding="utf-8")
    assert " zhaodeng " in installed
    assert installed.count("run-lmc5-night.sh") == 1
    assert installed.count("run-e-axis-shadow.sh") == 1
    assert installed.count("backup_nas_data.sh") == 1
    assert "ombre-brain" not in installed


def test_drop_in_install_refuses_while_the_legacy_container_is_running(tmp_path: Path) -> None:
    cron_file = tmp_path / "cron.d" / "ombre-vps-mirror"
    cron_file.parent.mkdir()
    env_file, env, state, calls = _job_fixture(tmp_path, cron_file=cron_file)
    env["LEGACY_RUNNING"] = "true"
    original = state.read_text(encoding="utf-8")

    result = _install(env_file, env)

    assert result.returncode != 0
    assert "legacy writer container is still running" in result.stderr
    assert state.read_text(encoding="utf-8") == original
    assert calls.read_text(encoding="utf-8").splitlines() == ["list"]
    assert not cron_file.exists()


def test_cutover_contract_requires_public_routing_to_8001_and_stopping_8000() -> None:
    contract = (ROOT / "docs" / "NAS_8001_PRODUCTION.md").read_text(encoding="utf-8")

    assert "docker stop ombre-brain" in contract
    assert "memory.zhaodeng.xyz" in contract
    assert "127.0.0.1:8001" in contract
    assert "must succeed" in contract
    assert "while 8000 remains stopped" in contract


def test_production_example_declares_root_owned_drop_in_and_public_guard() -> None:
    example = (ROOT / "deploy" / "nas-production.env.example").read_text(
        encoding="utf-8"
    )
    contract = (ROOT / "docs" / "NAS_8001_PRODUCTION.md").read_text(
        encoding="utf-8"
    )

    assert "OMBRE_JOB_CRON_FILE=/etc/cron.d/ombre-vps-mirror" in example
    assert "OMBRE_JOB_CRON_USER=zhaodeng" in example
    assert (
        "OMBRE_PUBLIC_HEALTH_URL=https://memory.zhaodeng.xyz/health" in example
    )
    assert "run the installer as root" in contract
    assert "execute its three jobs as `zhaodeng`" in contract
