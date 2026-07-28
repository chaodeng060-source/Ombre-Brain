from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SCRIPT = REPO_ROOT / "scripts" / "recover_nas.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture()
def recovery_env(tmp_path: Path) -> dict[str, str]:
    volume = tmp_path / "vol1"
    deploy = volume / "ombre-deploy"
    data = volume / "ombre-data"
    docker_root = volume / "docker"
    fake_bin = tmp_path / "bin"
    state = tmp_path / "state"
    for directory in (deploy, data, docker_root, fake_bin, state):
        directory.mkdir(parents=True, exist_ok=True)
    (deploy / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (deploy / "config.yaml").write_text("runtime: preserved\n", encoding="utf-8")
    (deploy / ".env").write_text("RUNTIME_SETTING=preserved\n", encoding="utf-8")
    (data / "existing-memory").write_text("present\n", encoding="utf-8")
    (state / "container").write_text("running\n", encoding="utf-8")
    (state / "commands.log").write_text("", encoding="utf-8")
    (state / "http.log").write_text("", encoding="utf-8")

    _write_executable(
        fake_bin / "docker",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${FAKE_STATE}/commands.log"
if [[ "${1:-}" == "info" ]]; then
  printf '%s\n' "${FAKE_DOCKER_ROOT}"
  exit 0
fi
if [[ "${1:-}" == "inspect" ]]; then
  state="$(cat "${FAKE_STATE}/container")"
  [[ "${state}" != "missing" ]] || exit 1
  if [[ "$*" == *"{{json .Mounts}}"* ]]; then
    printf '[{"Type":"bind","Source":"%s","Destination":"%s","RW":%s}]\n' \
      "${FAKE_CONTAINER_DATA_SOURCE}" "${FAKE_CONTAINER_DATA_TARGET}" \
      "${FAKE_CONTAINER_DATA_RW}"
    exit 0
  fi
  [[ "${state}" == "running" ]] && printf 'true\n' || printf 'false\n'
  exit 0
fi
if [[ "${1:-}" == "compose" ]]; then
  if [[ "$*" == *" config --quiet"* ]]; then
    [[ "${FAKE_COMPOSE_CONFIG_FAIL:-0}" != "1" ]]
    exit
  fi
  if [[ "$*" == *" config --format json"* ]]; then
    printf '{"services":{"ombre-brain":{"volumes":[{"type":"bind","source":"%s","target":"%s"}]}}}\n' \
      "${FAKE_COMPOSE_DATA_SOURCE}" "${FAKE_COMPOSE_DATA_TARGET}"
    exit 0
  fi
  if [[ "$*" == *" ps -q --all "* ]]; then
    state="$(cat "${FAKE_STATE}/container")"
    [[ "${state}" == "missing" ]] || printf 'fake-container\n'
    exit 0
  fi
  if [[ "$*" == *" up -d"* ]]; then
    printf 'running\n' >"${FAKE_STATE}/container"
    exit 0
  fi
fi
printf 'unexpected docker invocation: %s\n' "$*" >&2
exit 64
""",
    )
    _write_executable(
        fake_bin / "curl",
        r"""#!/usr/bin/env bash
set -euo pipefail
output=""
headers=""
method="GET"
fail_on_http=0
while (($#)); do
  if [[ "$1" == "-o" ]]; then
    output="$2"
    shift 2
  elif [[ "$1" == "-D" ]]; then
    headers="$2"
    shift 2
  elif [[ "$1" == "-X" ]]; then
    method="$2"
    shift 2
  elif [[ "$1" == "-f" || "$1" == "-fsS" ]]; then
    fail_on_http=1
    shift
  else
    shift
  fi
done
printf '%s\n' "${method}" >>"${FAKE_STATE}/http.log"
if [[ "${method}" == "DELETE" ]]; then
  printf '%s' "${FAKE_MCP_DELETE_CODE:-204}"
  exit 0
fi
if [[ "${FAKE_MCP_HEALTHY:-1}" == "1" ]]; then
  if [[ -n "${headers}" && -n "${FAKE_MCP_SESSION_ID-fake-session}" ]]; then
    printf 'HTTP/1.1 200 OK\r\nmcp-session-id: %s\r\n\r\n' \
      "${FAKE_MCP_SESSION_ID-fake-session}" >"${headers}"
  fi
  printf '{"result":{"serverInfo":{"name":"Ombre Brain"}}}\n' >"${output}"
  printf '200'
  exit 0
fi
printf '{"error":"unhealthy"}\n' >"${output}"
printf '503'
(( fail_on_http == 0 ))
""",
    )
    _write_executable(
        fake_bin / "findmnt",
        r"""#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"UUID"* ]]; then
  printf '%s\n' "${FAKE_MOUNT_UUID}"
else
  printf '%s\n' "${FAKE_MOUNT_TARGET}"
fi
""",
    )
    _write_executable(
        fake_bin / "logger",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${FAKE_STATE}/events.log"
""",
    )
    _write_executable(
        fake_bin / "crontab",
        r"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
  if [[ "${FAKE_CRONTAB_FAIL_LIST:-0}" == "1" ]]; then
    printf 'permission denied\n' >&2
    exit 2
  fi
  [[ -f "${FAKE_CRONTAB}" ]] || exit 1
  cat "${FAKE_CRONTAB}"
else
  cp "$1" "${FAKE_CRONTAB}"
fi
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_STATE": str(state),
            "FAKE_DOCKER_ROOT": str(docker_root),
            "FAKE_MOUNT_TARGET": str(volume),
            "FAKE_MOUNT_UUID": "fake-volume-uuid",
            "FAKE_COMPOSE_DATA_SOURCE": str(data),
            "FAKE_COMPOSE_DATA_TARGET": "/data",
            "FAKE_CONTAINER_DATA_SOURCE": str(data),
            "FAKE_CONTAINER_DATA_TARGET": "/data",
            "FAKE_CONTAINER_DATA_RW": "true",
            "FAKE_CRONTAB": str(state / "crontab"),
            "OMBRE_COMPOSE_DIR": str(deploy),
            "OMBRE_COMPOSE_FILE": str(deploy / "docker-compose.yml"),
            "OMBRE_DATA_DIR": str(data),
            "OMBRE_EXPECTED_VOLUME_ROOT": str(volume),
            "OMBRE_EXPECTED_VOLUME_UUID": "fake-volume-uuid",
            "OMBRE_EXPECTED_DOCKER_ROOT": str(docker_root),
            "OMBRE_DOCKER_WAIT_SECONDS": "0",
            "OMBRE_MCP_WAIT_SECONDS": "0",
            "OMBRE_POLL_SECONDS": "1",
            "OMBRE_LOCK_FILE": str(state / "recovery.lock"),
        }
    )
    return env


def _run(env: dict[str, str], mode: str = "--recover") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RECOVERY_SCRIPT), mode],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _commands(env: dict[str, str]) -> str:
    return (Path(env["FAKE_STATE"]) / "commands.log").read_text(encoding="utf-8")


def _http_methods(env: dict[str, str]) -> list[str]:
    content = (Path(env["FAKE_STATE"]) / "http.log").read_text(encoding="utf-8")
    return content.splitlines()


def test_healthy_container_is_a_noop(recovery_env: dict[str, str]) -> None:
    result = _run(recovery_env)
    assert result.returncode == 0, result.stderr
    assert " up -d" not in _commands(recovery_env)
    assert "no action needed" in result.stdout
    assert _http_methods(recovery_env) == ["POST", "DELETE"]


def test_stateless_mcp_probe_does_not_attempt_session_delete(
    recovery_env: dict[str, str],
) -> None:
    recovery_env["FAKE_MCP_SESSION_ID"] = ""

    result = _run(recovery_env)

    assert result.returncode == 0, result.stderr
    assert _http_methods(recovery_env) == ["POST"]


def test_mcp_session_cleanup_failure_is_not_retried_or_ignored(
    recovery_env: dict[str, str],
) -> None:
    recovery_env["FAKE_MCP_DELETE_CODE"] = "500"

    result = _run(recovery_env)

    assert result.returncode != 0
    assert "session cleanup returned HTTP 500" in result.stdout
    assert _http_methods(recovery_env) == ["POST", "DELETE"]
    assert " up -d" not in _commands(recovery_env)


def test_missing_container_is_rebuilt_once(recovery_env: dict[str, str]) -> None:
    state = Path(recovery_env["FAKE_STATE"])
    deploy = Path(recovery_env["OMBRE_COMPOSE_DIR"])
    data = Path(recovery_env["OMBRE_DATA_DIR"])
    config_before = (deploy / "config.yaml").read_bytes()
    env_before = (deploy / ".env").read_bytes()
    data_before = (data / "existing-memory").read_bytes()
    (state / "container").write_text("missing\n", encoding="utf-8")

    result = _run(recovery_env)

    assert result.returncode == 0, result.stderr
    commands = _commands(recovery_env)
    assert commands.count(" up -d --build ombre-brain") == 1
    assert all(word not in commands for word in (" down", " stop", " rm", " prune"))
    assert (deploy / "config.yaml").read_bytes() == config_before
    assert (deploy / ".env").read_bytes() == env_before
    assert (data / "existing-memory").read_bytes() == data_before


def test_stopped_container_is_started_without_rebuild(
    recovery_env: dict[str, str],
) -> None:
    state = Path(recovery_env["FAKE_STATE"])
    (state / "container").write_text("stopped\n", encoding="utf-8")

    result = _run(recovery_env)

    assert result.returncode == 0, result.stderr
    commands = _commands(recovery_env)
    assert " up -d ombre-brain" in commands
    assert " --build " not in commands


@pytest.mark.parametrize(
    "guard",
    [
        "docker-root",
        "mount",
        "volume-uuid",
        "empty-data",
        "compose",
        "volume-map",
        "container-map",
    ],
)
def test_guard_failures_never_mutate_docker(
    recovery_env: dict[str, str], tmp_path: Path, guard: str
) -> None:
    if guard == "docker-root":
        wrong = tmp_path / "wrong-docker-root"
        wrong.mkdir()
        recovery_env["FAKE_DOCKER_ROOT"] = str(wrong)
    elif guard == "mount":
        wrong = tmp_path / "wrong-mount"
        wrong.mkdir()
        recovery_env["FAKE_MOUNT_TARGET"] = str(wrong)
    elif guard == "volume-uuid":
        recovery_env["FAKE_MOUNT_UUID"] = "wrong-volume-uuid"
    elif guard == "empty-data":
        data = Path(recovery_env["OMBRE_DATA_DIR"])
        (data / "existing-memory").unlink()
    elif guard == "compose":
        recovery_env["FAKE_COMPOSE_CONFIG_FAIL"] = "1"
    elif guard == "volume-map":
        wrong = tmp_path / "wrong-data-source"
        wrong.mkdir()
        recovery_env["FAKE_COMPOSE_DATA_SOURCE"] = str(wrong)
    elif guard == "container-map":
        wrong = tmp_path / "wrong-container-data-source"
        wrong.mkdir()
        recovery_env["FAKE_CONTAINER_DATA_SOURCE"] = str(wrong)

    result = _run(recovery_env)

    assert result.returncode != 0
    assert " up -d" not in _commands(recovery_env)


def test_running_but_unhealthy_is_reported_without_restart(
    recovery_env: dict[str, str],
) -> None:
    recovery_env["FAKE_MCP_HEALTHY"] = "0"

    result = _run(recovery_env)

    assert result.returncode != 0
    assert "refusing restart loop" in result.stdout
    assert " up -d" not in _commands(recovery_env)


def test_cron_install_is_idempotent_and_preserves_existing_entries(
    recovery_env: dict[str, str],
) -> None:
    crontab = Path(recovery_env["FAKE_CRONTAB"])
    tunnel_line = (
        "* * * * * /home/zhaodeng/.ssh/ensure_vps_reverse_tunnel.sh "
        ">/dev/null 2>&1\n"
    )
    crontab.write_text(tunnel_line, encoding="utf-8")

    first = _run(recovery_env, "--install-cron")
    second = _run(recovery_env, "--install-cron")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    installed = crontab.read_text(encoding="utf-8")
    assert tunnel_line.strip() in installed
    assert installed.count("# BEGIN ombre-brain self-recovery") == 1
    assert "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" in installed
    assert installed.count("@reboot sleep 90") == 1
    assert installed.count("*/5 * * * *") == 1


def test_malformed_cron_markers_are_rejected_without_overwrite(
    recovery_env: dict[str, str],
) -> None:
    crontab = Path(recovery_env["FAKE_CRONTAB"])
    original = (
        "* * * * * /home/zhaodeng/.ssh/ensure_vps_reverse_tunnel.sh\n"
        "# BEGIN ombre-brain self-recovery\n"
        "0 1 * * * important-existing-job\n"
    )
    crontab.write_text(original, encoding="utf-8")

    result = _run(recovery_env, "--install-cron")

    assert result.returncode != 0
    assert "markers are malformed" in result.stdout
    assert crontab.read_text(encoding="utf-8") == original


def test_crontab_read_error_is_rejected_without_overwrite(
    recovery_env: dict[str, str],
) -> None:
    crontab = Path(recovery_env["FAKE_CRONTAB"])
    original = "0 2 * * * important-existing-job\n"
    crontab.write_text(original, encoding="utf-8")
    recovery_env["FAKE_CRONTAB_FAIL_LIST"] = "1"

    result = _run(recovery_env, "--install-cron")

    assert result.returncode != 0
    assert "cannot safely read current crontab" in result.stdout
    assert crontab.read_text(encoding="utf-8") == original
