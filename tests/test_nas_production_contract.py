from __future__ import annotations

import os
import subprocess
import hashlib
import sqlite3
import fcntl
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_shell(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(script)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_e_axis_job_refuses_an_implicit_writer_and_targets_8001(tmp_path):
    script = ROOT / "cron" / "run-e-axis-shadow.sh"
    calls = tmp_path / "docker.calls"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n",
        encoding="utf-8",
    )
    docker.chmod(0o700)
    env = {
        **os.environ,
        "DOCKER_BIN": str(docker),
        "CALL_LOG": str(calls),
    }
    env.pop("OMBRE_CONTAINER_NAME", None)

    missing = _run_shell(script, env)

    assert missing.returncode != 0
    assert "OMBRE_CONTAINER_NAME" in missing.stderr
    assert not calls.exists()

    env["OMBRE_CONTAINER_NAME"] = "ombre-vps-mirror"
    selected = _run_shell(script, env)

    assert selected.returncode == 0
    assert calls.read_text(encoding="utf-8").strip() == (
        "exec ombre-vps-mirror python /app/e_axis_night.py"
    )


def test_compose_can_describe_the_8001_instance_without_hardcoded_data_paths():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "container_name: ${OMBRE_CONTAINER_NAME:-ombre-brain}" in compose
    assert "${OMBRE_HOST_PORT:-8000}:8000" in compose
    assert "${OMBRE_DATA_DIR:-/vol1/ombre-data}:/data" in compose
    assert "${OMBRE_SNAPSHOT_DIR:-/vol1/ombre-night-snapshots}:/snapshots" in compose
    assert "${OMBRE_CONFIG_FILE:-./config.yaml}:/app/config.yaml" in compose
    assert "OMBRE_SOURCE_REVISION: ${OMBRE_DEPLOY_COMMIT:-unknown}" in compose
    assert "network_mode: ${OMBRE_NETWORK_MODE:-bridge}" in compose
    assert 'LABEL org.opencontainers.image.revision="${OMBRE_SOURCE_REVISION}"' in dockerfile


def test_production_example_points_every_writer_and_backup_at_8001():
    example = (ROOT / "deploy" / "nas-production.env.example").read_text(
        encoding="utf-8"
    )

    expected = {
        "OMBRE_CONTAINER_NAME=ombre-vps-mirror",
        "OMBRE_HOST_PORT=8001",
        "OMBRE_ACTIVE_DIR=/vol1/ombre-migrate/code",
        "OMBRE_DATA_DIR=/vol1/ombre-migrate/data",
        "OMBRE_SNAPSHOT_DIR=/vol1/ombre-migrate/snapshots",
        "OMBRE_HEALTH_URL=http://127.0.0.1:8001/health",
        "OMBRE_DEPLOYMENT_ANCHOR_FILE=/vol1/ombre-migrate/deployment-anchor.env",
        "OMBRE_MUTATION_LOCK_FILE=/vol1/ombre-migrate/ombre-production.lock",
    }
    lines = {line.strip() for line in example.splitlines() if line.strip()}

    assert expected <= lines
    assert "TOKEN=" not in example
    assert "PASSWORD=" not in example

    reconciliation = (ROOT / "deploy" / "nas-live-reconciliation.tsv").read_text(
        encoding="utf-8"
    )
    records = [
        line.split("\t")
        for line in reconciliation.splitlines()
        if line and not line.startswith("#")
    ]
    assert all(len(record) == 3 for record in records)
    paths = [record[0] for record in records]
    assert len(paths) == len(set(paths))
    assert {
        "server.py",
        "bucket_manager.py",
        "bm25_index.py",
        "memory_signal.py",
    } <= set(paths)
    for path, live_sha, source_sha in records:
        assert path.endswith(".py")
        assert live_sha == "ABSENT" or (
            len(live_sha) == 64 and all(c in "0123456789abcdef" for c in live_sha)
        )
        assert source_sha in {"UNRECONCILED", "ABSENT"} or (
            len(source_sha) == 64 and all(c in "0123456789abcdef" for c in source_sha)
        )


def test_tracked_nas_scripts_have_valid_shell_syntax():
    scripts = (
        ROOT / "scripts" / "deploy_nas_atomic.sh",
        ROOT / "scripts" / "backup_nas_data.sh",
        ROOT / "scripts" / "install_nas_jobs.sh",
    )

    for script in scripts:
        result = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}"

    deploy = scripts[0].read_text(encoding="utf-8")
    assert "trap rollback EXIT" in deploy
    assert "trap 'exit 130' INT" in deploy
    assert 'compose_project="ombre-vps-mirror-${COMMIT:0:12}-' in deploy
    assert '-p "$compose_project"' in deploy
    assert "-p ombre-vps-mirror" not in deploy


def test_docker_publish_waits_for_grow_and_production_contract_tests():
    workflow = (ROOT / ".github" / "workflows" / "docker-publish.yml").read_text(
        encoding="utf-8"
    )

    assert "test-production-contract:" in workflow
    assert "needs: test-production-contract" in workflow
    assert "tests/test_grow_bucket_id_receipts.py" in workflow
    assert "tests/test_nas_production_contract.py" in workflow
    assert "tests/test_patrol_night.py" in workflow
    assert "OMBRE_SOURCE_REVISION=${{ github.sha }}" in workflow
    assert "deploy_ready: ${{ steps.readiness.outputs.deploy_ready }}" in workflow
    assert "grep -q $'\\tUNRECONCILED$'" in workflow
    assert "needs.test-production-contract.outputs.deploy_ready == 'true'" in workflow

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "rank-bm25==0.2.2" in requirements
    assert "psycopg[binary]==3.3.4" in requirements


def _write_production_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    active = tmp_path / "active"
    data = tmp_path / "data"
    snapshots = tmp_path / "snapshots"
    backup = tmp_path / "backups"
    for directory in (active, data, snapshots, backup):
        directory.mkdir()
    (active / "config.yaml").write_text("buckets_dir: /data\n", encoding="utf-8")
    secret_env = tmp_path / "mirror.env"
    secret_env.write_text("OMBRE_API_TOKEN=test-only\n", encoding="utf-8")
    env_file = tmp_path / "production.env"
    env_file.write_text(
        "\n".join(
            (
                "OMBRE_CONTAINER_NAME=ombre-vps-mirror",
                "OMBRE_SERVICE=ombre-brain",
                "OMBRE_HOST_PORT=8001",
                "OMBRE_HEALTH_URL=http://127.0.0.1:8001/health",
                "OMBRE_NETWORK_MODE=ombre-net",
                f"OMBRE_ACTIVE_DIR={active}",
                f"OMBRE_DATA_DIR={data}",
                f"OMBRE_SNAPSHOT_DIR={snapshots}",
                f"OMBRE_CONFIG_FILE={active / 'config.yaml'}",
                f"OMBRE_SECRET_ENV_FILE={secret_env}",
                f"OMBRE_BACKUP_ROOT={backup}",
                f"OMBRE_DEPLOYMENT_ANCHOR_FILE={tmp_path / 'deployment-anchor.env'}",
                f"OMBRE_MUTATION_LOCK_FILE={tmp_path / 'mutation.lock'}",
                "OMBRE_BUCKETS_VOLUME=test-buckets",
                "OMBRE_BUCKETS_VOLUME_EXTERNAL=true",
                "OMBRE_IMAGE_REPOSITORY=ombre-vps-mirror",
                "OMBRE_EXPECTED_CONTAINER_ID=test-container-id",
                "OMBRE_EXPECTED_IMAGE_ID=sha256:test-image-id",
                "OMBRE_BACKUP_KEEP=7",
                f"OMBRE_JOB_ENV_FILE={env_file}",
                f"OMBRE_JOB_REPO_DIR={ROOT}",
                f"OMBRE_CONTRACT_TEST_ROOT={tmp_path}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return env_file, active, data


def _make_deploy_archive(
    tmp_path: Path,
    *,
    live_anchor: str = "1" * 64,
    source_anchor: str | None = None,
) -> tuple[Path, str, str, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    server = "_bid_suffix = ' [id]'\n# _bid_suffix second grow path\n"
    (source / "server.py").write_text(server, encoding="utf-8")
    (source / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (source / "docker-compose.yml").write_text(
        "services:\n  ombre-brain:\n    build: .\n", encoding="utf-8"
    )
    (source / "requirements.txt").write_text(
        "rank-bm25==0.2.2\npsycopg[binary]==3.2.9\n", encoding="utf-8"
    )
    server_sha = hashlib.sha256(server.encode()).hexdigest()
    if source_anchor is None:
        source_anchor = server_sha
    (source / "deploy").mkdir()
    (source / "deploy" / "nas-live-reconciliation.tsv").write_text(
        f"server.py\t{live_anchor}\t{source_anchor}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "contract@example.invalid"],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "NAS contract test"],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "add", "--", "."], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "contract fixture"], cwd=source, check=True
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    archive = tmp_path / "source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(archive), commit],
        cwd=source,
        check=True,
    )
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, archive_sha, server_sha, live_anchor, commit


def _fake_deploy_docker(tmp_path: Path) -> tuple[Path, Path]:
    calls = tmp_path / "docker.calls"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "case \"$1\" in\n"
        "  inspect)\n"
        "    case \"$3\" in\n"
        "      *'.Id'*) echo test-container-id ;;\n"
        "      *'.Image'*) echo sha256:test-image-id ;;\n"
        "      *'/data'*) echo \"$FAKE_DATA_DIR\" ;;\n"
        "      *'/snapshots'*) echo \"$FAKE_SNAPSHOT_DIR\" ;;\n"
        "      *'/app/config.yaml'*) echo \"$FAKE_CONFIG_FILE\" ;;\n"
        "      *'/app/buckets'*) echo test-buckets ;;\n"
        "      *'NetworkSettings.Networks'*) echo ombre-net ;;\n"
        "    esac ;;\n"
        "  port) echo 127.0.0.1:8001 ;;\n"
        "  diff) printf '%b\\n' \"${FAKE_DIFF:-C /app/server.py}\" ;;\n"
        "  exec)\n"
        "    [ \"${FAKE_LIVE_PY_ABSENT:-0}\" = 1 ] || echo \"$FAKE_LIVE_SHA  /app/server.py\"\n"
        "    [ -z \"${FAKE_EXTRA_LIVE_PY_SHA:-}\" ] || echo \"$FAKE_EXTRA_LIVE_PY_SHA  /app/baked_only.py\" ;;\n"
        "  stop)\n"
        "    if [ \"${FAKE_TERM_PARENT_AFTER_STOP:-0}\" = 1 ]; then kill -TERM \"$PPID\"; fi\n"
        "    exit 0 ;;\n"
        "  start) exit 0 ;;\n"
        "  *) exit 91 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o700)
    return docker, calls


def _fake_cutover_docker(tmp_path: Path) -> tuple[Path, Path, Path]:
    calls = tmp_path / "cutover-docker.calls"
    state = tmp_path / "cutover-docker.state"
    state.write_text("old\n", encoding="utf-8")
    docker = tmp_path / "cutover-docker"
    docker.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "printf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "state=$(cat \"$FAKE_STATE\")\n"
        "case \"$1\" in\n"
        "  inspect)\n"
        "    fmt=$3; target=$4\n"
        "    if [ \"$state\" = renamed ] && [ \"$target\" = ombre-vps-mirror ]; then exit 1; fi\n"
        "    case \"$fmt\" in\n"
        "      *'.Id'*)\n"
        "        case \"$target\" in *-rollback-*) [ \"$state\" != old ] && echo test-container-id || exit 1 ;; *) [ \"$state\" = new ] && echo new-container-id || echo test-container-id ;; esac ;;\n"
        "      *'.Image'*)\n"
        "        case \"$target\" in *-rollback-*) echo sha256:test-image-id ;; *) [ \"$state\" = new ] && echo sha256:new-image-id || echo sha256:test-image-id ;; esac ;;\n"
        "      *'/data'*) echo \"$FAKE_DATA_DIR\" ;;\n"
        "      *'/snapshots'*) echo \"$FAKE_SNAPSHOT_DIR\" ;;\n"
        "      *'/app/config.yaml'*) echo \"$FAKE_CONFIG_FILE\" ;;\n"
        "      *'/app/buckets'*) echo test-buckets ;;\n"
        "      *'NetworkSettings.Networks'*) echo ombre-net ;;\n"
        "    esac ;;\n"
        "  container)\n"
        "    [ \"$2\" = inspect ] || exit 90\n"
        "    [ \"$state\" != old ] ;;\n"
        "  image)\n"
        "    case \"$*\" in\n"
        "      *org.opencontainers.image.revision*) echo \"$OMBRE_DEPLOY_COMMIT\" ;;\n"
        "      *) echo sha256:new-image-id ;;\n"
        "    esac ;;\n"
        "  port) echo 127.0.0.1:8001 ;;\n"
        "  diff) printf 'C /app\\nC /app/server.py\\n' ;;\n"
        "  exec) [ \"${FAKE_LIVE_PY_ABSENT:-0}\" = 1 ] || echo \"$FAKE_LIVE_SHA  /app/server.py\" ;;\n"
        "  stop)\n"
        "    if [ -n \"${FAKE_STOP_COUNT:-}\" ]; then n=$(cat \"$FAKE_STOP_COUNT\" 2>/dev/null || echo 0); n=$((n+1)); echo \"$n\" >\"$FAKE_STOP_COUNT\"; [ \"$n\" != \"${FAKE_TERM_ON_STOP_NUMBER:-0}\" ] || kill -TERM \"$PPID\"; fi ;;\n"
        "  start) : ;;\n"
        "  rename)\n"
        "    if [ \"$2\" = ombre-vps-mirror ] && [ \"${FAKE_FAIL_RENAME:-0}\" = 1 ]; then exit 44; fi\n"
        "    if [ \"$2\" = ombre-vps-mirror ]; then echo renamed >\"$FAKE_STATE\"; else echo old >\"$FAKE_STATE\"; fi ;;\n"
        "  rm) echo renamed >\"$FAKE_STATE\" ;;\n"
        "  compose)\n"
        "    case \" $* \" in *' up '*) echo new >\"$FAKE_STATE\" ;; esac ;;\n"
        "  *) exit 91 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o700)
    return docker, calls, state


def test_deploy_preflight_blocks_unreconciled_live_python_before_build(tmp_path):
    env_file, _active, data = _write_production_env(tmp_path)
    archive, archive_sha, _server_sha, _live_anchor, commit = _make_deploy_archive(tmp_path)
    docker, calls = _fake_deploy_docker(tmp_path)
    script = ROOT / "scripts" / "deploy_nas_atomic.sh"
    env = {
        **os.environ,
        "DOCKER_BIN": str(docker),
        "CALL_LOG": str(calls),
        "FAKE_DATA_DIR": str(data),
        "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
        "FAKE_LIVE_SHA": "f" * 64,
    }

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
            "--preflight",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "live reconciliation anchor changed: server.py" in result.stderr
    recorded = calls.read_text(encoding="utf-8")
    assert "compose" not in recorded
    assert "stop ombre-vps-mirror" not in recorded


def test_deploy_preflight_blocks_an_unreconciled_source_sentinel(tmp_path):
    env_file, _active, data = _write_production_env(tmp_path)
    archive, archive_sha, server_sha, _live_anchor, commit = _make_deploy_archive(
        tmp_path, source_anchor="UNRECONCILED"
    )
    docker, calls = _fake_deploy_docker(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
            "--preflight",
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
            # Equal live/source bytes used to bypass a per-difference-only gate.
            "FAKE_LIVE_SHA": server_sha,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unresolved reconciliation sentinel remains" in result.stderr
    recorded = calls.read_text(encoding="utf-8")
    assert "compose" not in recorded
    assert "stop ombre-vps-mirror" not in recorded


def test_deploy_rejects_an_archive_declared_as_a_different_commit(tmp_path):
    env_file, _active, _data = _write_production_env(tmp_path)
    archive, archive_sha, _server_sha, _live_anchor, commit = _make_deploy_archive(
        tmp_path
    )
    assert commit != "f" * 40

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            "f" * 40,
            "--preflight",
        ],
        env=os.environ,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "archive commit mismatch" in result.stderr


def test_shared_mutation_lock_blocks_deploy_and_backup_before_docker(tmp_path):
    env_file, _active, data = _write_production_env(tmp_path)
    archive, archive_sha, _server_sha, _live_anchor, commit = _make_deploy_archive(
        tmp_path
    )
    docker, calls = _fake_deploy_docker(tmp_path)
    command_env = {
        **os.environ,
        "DOCKER_BIN": str(docker),
        "CALL_LOG": str(calls),
        "FAKE_DATA_DIR": str(data),
        "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
        "FAKE_LIVE_SHA": "1" * 64,
    }

    with (tmp_path / "mutation.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deploy = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
                "--env-file",
                str(env_file),
                "--archive",
                str(archive),
                "--archive-sha256",
                archive_sha,
                "--commit",
                commit,
                "--preflight",
            ],
            env=command_env,
            check=False,
            capture_output=True,
            text=True,
        )
        backup = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts" / "backup_nas_data.sh"),
                "--env-file",
                str(env_file),
                "--check",
            ],
            env=command_env,
            check=False,
            capture_output=True,
            text=True,
        )

    assert deploy.returncode != 0
    assert backup.returncode != 0
    assert "production mutation lock is held" in deploy.stderr
    assert "production mutation lock is held" in backup.stderr
    assert not calls.exists()


def test_deploy_preflight_accepts_an_exact_reconciled_live_hunk(tmp_path):
    env_file, _active, data = _write_production_env(tmp_path)
    archive, archive_sha, server_sha, live_anchor, commit = _make_deploy_archive(tmp_path)
    docker, _calls = _fake_deploy_docker(tmp_path)
    script = ROOT / "scripts" / "deploy_nas_atomic.sh"
    env = {
        **os.environ,
        "DOCKER_BIN": str(docker),
        "CALL_LOG": str(tmp_path / "docker.calls"),
        "FAKE_DATA_DIR": str(data),
        "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
        "FAKE_LIVE_SHA": live_anchor,
        "FAKE_DIFF": (
            "C /app\\n"
            "C /root/.cache/pip\\n"
            "C /app/vendor\\n"
            "A /app/__pycache__/server.cpython-312.pyc\\n"
            "C /app/server.py"
        ),
    }

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
            "--preflight",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "PREFLIGHT_OK" in result.stdout


def test_deploy_preflight_blocks_unknown_app_drift_before_build(tmp_path):
    env_file, _active, data = _write_production_env(tmp_path)
    archive, archive_sha, _server_sha, _live_anchor, commit = _make_deploy_archive(tmp_path)
    docker, calls = _fake_deploy_docker(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
            "--preflight",
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
            "FAKE_LIVE_SHA": "1" * 64,
            "FAKE_DIFF": "C /app\\nA /app/unknown.bin",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unsupported live /app drift remains" in result.stderr
    recorded = calls.read_text(encoding="utf-8")
    assert "compose" not in recorded
    assert "stop ombre-vps-mirror" not in recorded


def test_preflight_blocks_baked_in_python_missing_from_incoming_source(tmp_path):
    env_file, _active, data = _write_production_env(tmp_path)
    archive, archive_sha, _server_sha, live_anchor, commit = _make_deploy_archive(
        tmp_path
    )
    docker, calls = _fake_deploy_docker(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
            "--preflight",
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
            "FAKE_LIVE_SHA": live_anchor,
            "FAKE_EXTRA_LIVE_PY_SHA": "e" * 64,
            "FAKE_DIFF": "C /app",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing or duplicate reconciliation record: baked_only.py" in result.stderr
    recorded = calls.read_text(encoding="utf-8")
    assert "compose" not in recorded
    assert "stop ombre-vps-mirror" not in recorded


def test_deleted_live_python_requires_an_explicit_reintroduction_record(tmp_path):
    env_file, _active, data = _write_production_env(tmp_path)
    archive, archive_sha, _server_sha, _live_anchor, commit = _make_deploy_archive(
        tmp_path, live_anchor="ABSENT"
    )
    docker, calls = _fake_deploy_docker(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
            "--preflight",
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
            "FAKE_LIVE_SHA": "unused",
            "FAKE_LIVE_PY_ABSENT": "1",
            "FAKE_DIFF": "D /app/server.py",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "PREFLIGHT_OK" in result.stdout
    assert "exec ombre-vps-mirror sha256sum /app/server.py" not in calls.read_text(
        encoding="utf-8"
    )


def _prepare_cutover_fixture(tmp_path: Path):
    env_file, active, data = _write_production_env(tmp_path)
    (active / "old-container-only.txt").write_text("preserve me\n", encoding="utf-8")
    database = sqlite3.connect(data / "audit.sqlite3")
    database.execute("create table audit (id integer primary key, value text)")
    database.execute("insert into audit(value) values ('durable')")
    database.commit()
    database.close()
    archive, archive_sha, _server_sha, live_anchor, commit = _make_deploy_archive(
        tmp_path
    )
    docker, calls, state = _fake_cutover_docker(tmp_path)
    return env_file, active, data, archive, archive_sha, live_anchor, commit, docker, calls, state


def test_cutover_keeps_old_container_and_writes_next_backup_anchor(tmp_path):
    (
        env_file,
        active,
        data,
        archive,
        archive_sha,
        live_anchor,
        commit,
        docker,
        calls,
        state,
    ) = _prepare_cutover_fixture(tmp_path)
    curl = tmp_path / "curl-ok"
    curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    curl.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CURL_BIN": str(curl),
            "CALL_LOG": str(calls),
            "FAKE_STATE": str(state),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(active / "config.yaml"),
            "FAKE_LIVE_SHA": live_anchor,
            "OMBRE_BACKUP_LOCK_FILE": str(tmp_path / "backup.lock"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert state.read_text(encoding="utf-8").strip() == "new"
    anchor = (tmp_path / "deployment-anchor.env").read_text(encoding="utf-8")
    assert "OMBRE_EXPECTED_CONTAINER_ID=new-container-id" in anchor
    assert "OMBRE_EXPECTED_IMAGE_ID=sha256:new-image-id" in anchor
    recorded = calls.read_text(encoding="utf-8")
    assert "rename ombre-vps-mirror ombre-vps-mirror-rollback-" in recorded
    assert "tag sha256:test-image-id" not in recorded
    assert "rollback_container=ombre-vps-mirror-rollback-" in result.stdout
    assert f"-p ombre-vps-mirror-{commit[:12]}-" in recorded
    manifest = (active / "deployment-manifest.json").read_text(encoding="utf-8")
    assert f'"compose_project": "ombre-vps-mirror-{commit[:12]}-' in manifest
    assert len(list(tmp_path.glob("previous-*"))) == 1


def test_failed_cutover_renames_and_restarts_the_original_container(tmp_path):
    (
        env_file,
        active,
        data,
        archive,
        archive_sha,
        live_anchor,
        commit,
        docker,
        calls,
        state,
    ) = _prepare_cutover_fixture(tmp_path)
    curl = tmp_path / "curl-by-state"
    curl.write_text(
        "#!/bin/sh\n[ \"$(cat \"$FAKE_STATE\")\" = old ]\n", encoding="utf-8"
    )
    curl.chmod(0o700)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sleep.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DOCKER_BIN": str(docker),
            "CURL_BIN": str(curl),
            "CALL_LOG": str(calls),
            "FAKE_STATE": str(state),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(active / "config.yaml"),
            "FAKE_LIVE_SHA": live_anchor,
            "OMBRE_BACKUP_LOCK_FILE": str(tmp_path / "backup.lock"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert state.read_text(encoding="utf-8").strip() == "old"
    assert (active / "old-container-only.txt").read_text(encoding="utf-8") == "preserve me\n"
    assert not (tmp_path / "deployment-anchor.env").exists()
    recorded = calls.read_text(encoding="utf-8")
    assert "rename ombre-vps-mirror ombre-vps-mirror-rollback-" in recorded
    assert "rename ombre-vps-mirror-rollback-" in recorded
    assert "start ombre-vps-mirror" in recorded


def test_failed_old_container_rename_restarts_the_stopped_original(tmp_path):
    (
        env_file,
        active,
        data,
        archive,
        archive_sha,
        live_anchor,
        commit,
        docker,
        calls,
        state,
    ) = _prepare_cutover_fixture(tmp_path)
    curl = tmp_path / "curl-ok"
    curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    curl.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CURL_BIN": str(curl),
            "CALL_LOG": str(calls),
            "FAKE_STATE": str(state),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(active / "config.yaml"),
            "FAKE_LIVE_SHA": live_anchor,
            "FAKE_FAIL_RENAME": "1",
            "OMBRE_BACKUP_LOCK_FILE": str(tmp_path / "backup.lock"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert state.read_text(encoding="utf-8").strip() == "old"
    recorded = calls.read_text(encoding="utf-8")
    assert "stop ombre-vps-mirror" in recorded
    assert "start ombre-vps-mirror" in recorded
    assert "rm -f ombre-vps-mirror" not in recorded


def test_term_after_deploy_stop_recovers_original_from_observed_state(tmp_path):
    (
        env_file,
        active,
        data,
        archive,
        archive_sha,
        live_anchor,
        commit,
        docker,
        calls,
        state,
    ) = _prepare_cutover_fixture(tmp_path)
    curl = tmp_path / "curl-ok"
    curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    curl.chmod(0o700)
    stop_count = tmp_path / "stop-count"

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CURL_BIN": str(curl),
            "CALL_LOG": str(calls),
            "FAKE_STATE": str(state),
            "FAKE_STOP_COUNT": str(stop_count),
            "FAKE_TERM_ON_STOP_NUMBER": "2",
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(active / "config.yaml"),
            "FAKE_LIVE_SHA": live_anchor,
            "OMBRE_BACKUP_LOCK_FILE": str(tmp_path / "backup.lock"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 143
    assert state.read_text(encoding="utf-8").strip() == "old"
    assert (active / "old-container-only.txt").is_file()
    recorded = calls.read_text(encoding="utf-8")
    assert recorded.count("stop ombre-vps-mirror") == 2
    assert recorded.count("start ombre-vps-mirror") >= 2


def test_source_restore_failure_does_not_skip_original_container_restart(tmp_path):
    (
        env_file,
        active,
        data,
        archive,
        archive_sha,
        live_anchor,
        commit,
        docker,
        calls,
        state,
    ) = _prepare_cutover_fixture(tmp_path)
    curl = tmp_path / "curl-by-state"
    curl.write_text(
        "#!/bin/sh\n[ \"$(cat \"$FAKE_STATE\")\" = old ]\n", encoding="utf-8"
    )
    curl.chmod(0o700)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sleep.chmod(0o700)
    mv = tmp_path / "mv"
    mv.write_text(
        "#!/bin/sh\ncase \"$1\" in *previous-*) exit 77 ;; esac\nexec /bin/mv \"$@\"\n",
        encoding="utf-8",
    )
    mv.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "deploy_nas_atomic.sh"),
            "--env-file",
            str(env_file),
            "--archive",
            str(archive),
            "--archive-sha256",
            archive_sha,
            "--commit",
            commit,
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DOCKER_BIN": str(docker),
            "CURL_BIN": str(curl),
            "CALL_LOG": str(calls),
            "FAKE_STATE": str(state),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(active / "config.yaml"),
            "FAKE_LIVE_SHA": live_anchor,
            "OMBRE_BACKUP_LOCK_FILE": str(tmp_path / "backup.lock"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert state.read_text(encoding="utf-8").strip() == "old"
    recorded = calls.read_text(encoding="utf-8")
    assert "rename ombre-vps-mirror-rollback-" in recorded
    assert "start ombre-vps-mirror" in recorded


def test_backup_check_rejects_a_different_container_data_bind(tmp_path):
    env_file, _active, _data = _write_production_env(tmp_path)
    other_data = tmp_path / "other-data"
    other_data.mkdir()
    docker, calls = _fake_deploy_docker(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "backup_nas_data.sh"),
            "--env-file",
            str(env_file),
            "--check",
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(other_data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
            "FAKE_LIVE_SHA": "f" * 64,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "container /data mismatch" in result.stderr
    assert "stop ombre-vps-mirror" not in calls.read_text(encoding="utf-8")


def test_backup_cold_copies_code_and_data_then_restarts_8001(tmp_path):
    env_file, active, data = _write_production_env(tmp_path)
    database = sqlite3.connect(data / "audit.sqlite3")
    database.execute("create table audit (id integer primary key, value text)")
    database.execute("insert into audit(value) values ('durable')")
    database.commit()
    database.close()
    (active / "server.py").write_text("# tracked source\n", encoding="utf-8")
    docker, calls = _fake_deploy_docker(tmp_path)
    curl = tmp_path / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    curl.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "backup_nas_data.sh"),
            "--env-file",
            str(env_file),
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CURL_BIN": str(curl),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
            "FAKE_LIVE_SHA": "f" * 64,
            "OMBRE_BACKUP_LOCK_FILE": str(tmp_path / "backup.lock"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    backups = list((tmp_path / "backups").glob("backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "code.tar").is_file()
    assert (backups[0] / "data.tar").is_file()
    assert (backups[0] / "SHA256SUMS").is_file()
    checked = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=backups[0],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    recorded = calls.read_text(encoding="utf-8")
    assert "stop ombre-vps-mirror" in recorded
    assert "start ombre-vps-mirror" in recorded
    assert "BACKUP_OK" in result.stdout


def test_backup_failure_after_stop_still_restarts_8001(tmp_path):
    env_file, _active, data = _write_production_env(tmp_path)
    docker, calls = _fake_deploy_docker(tmp_path)
    curl = tmp_path / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    curl.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "backup_nas_data.sh"),
            "--env-file",
            str(env_file),
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CURL_BIN": str(curl),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
            "FAKE_LIVE_SHA": "f" * 64,
            "OMBRE_BACKUP_LOCK_FILE": str(tmp_path / "backup.lock"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    recorded = calls.read_text(encoding="utf-8")
    assert "stop ombre-vps-mirror" in recorded
    assert "start ombre-vps-mirror" in recorded
    assert not list((tmp_path / "backups").glob("backup-*"))
    assert len(list((tmp_path / "backups").glob("failed-*"))) == 1


def test_backup_term_after_stop_still_restarts_8001(tmp_path):
    env_file, _active, data = _write_production_env(tmp_path)
    docker, calls = _fake_deploy_docker(tmp_path)
    curl = tmp_path / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    curl.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "backup_nas_data.sh"),
            "--env-file",
            str(env_file),
        ],
        env={
            **os.environ,
            "DOCKER_BIN": str(docker),
            "CURL_BIN": str(curl),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(tmp_path / "active" / "config.yaml"),
            "FAKE_LIVE_SHA": "f" * 64,
            "FAKE_TERM_PARENT_AFTER_STOP": "1",
            "OMBRE_BACKUP_LOCK_FILE": str(tmp_path / "backup.lock"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 143
    recorded = calls.read_text(encoding="utf-8")
    assert "stop ombre-vps-mirror" in recorded
    assert "start ombre-vps-mirror" in recorded


def _fake_crontab(tmp_path: Path) -> tuple[Path, Path]:
    state = tmp_path / "crontab.txt"
    state.write_text("", encoding="utf-8")
    command = tmp_path / "crontab"
    command.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -l ]; then cat \"$CRON_STATE\"; exit 0; fi\n"
        "cp \"$1\" \"$CRON_STATE\"\n",
        encoding="utf-8",
    )
    command.chmod(0o700)
    return command, state


def test_job_installer_refuses_legacy_8000_lines_then_is_idempotent(tmp_path):
    env_file, active, data = _write_production_env(tmp_path)
    crontab, state = _fake_crontab(tmp_path)
    docker, calls = _fake_deploy_docker(tmp_path)
    installer = ROOT / "scripts" / "install_nas_jobs.sh"
    env = {
        **os.environ,
        "CRONTAB_BIN": str(crontab),
        "CRON_STATE": str(state),
        "TMPDIR": str(tmp_path),
        "DOCKER_BIN": str(docker),
        "CALL_LOG": str(calls),
        "FAKE_DATA_DIR": str(data),
        "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        "FAKE_CONFIG_FILE": str(active / "config.yaml"),
        "FAKE_LIVE_SHA": "f" * 64,
    }
    state.write_text(
        "30 4 * * * /vol1/ombre-deploy/cron/run-lmc5-night.sh\n",
        encoding="utf-8",
    )

    blocked = subprocess.run(
        ["bash", str(installer), "--env-file", str(env_file)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert blocked.returncode != 0
    assert "unmanaged Ombre writer/backup cron remains" in blocked.stderr

    state.write_text("17 * * * * unrelated-safe-job\n", encoding="utf-8")
    for _ in range(2):
        installed = subprocess.run(
            ["bash", str(installer), "--env-file", str(env_file)],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert installed.returncode == 0, installed.stderr

    final = state.read_text(encoding="utf-8")
    assert final.count("# BEGIN ombre-vps-mirror production jobs") == 1
    assert final.count("run-lmc5-night.sh") == 1
    assert final.count("run-e-axis-shadow.sh") == 1
    assert final.count("backup_nas_data.sh") == 1
    assert "ombre-brain" not in final


def test_job_installer_rejects_an_unmanaged_current_job_before_any_write(tmp_path):
    env_file, active, data = _write_production_env(tmp_path)
    crontab, state = _fake_crontab(tmp_path)
    docker, calls = _fake_deploy_docker(tmp_path)
    original = f"12 3 * * * {ROOT}/cron/run-lmc5-night.sh\n17 * * * * safe-job\n"
    state.write_text(original, encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "install_nas_jobs.sh"),
            "--env-file",
            str(env_file),
        ],
        env={
            **os.environ,
            "CRONTAB_BIN": str(crontab),
            "CRON_STATE": str(state),
            "TMPDIR": str(tmp_path),
            "DOCKER_BIN": str(docker),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(active / "config.yaml"),
            "FAKE_LIVE_SHA": "f" * 64,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unmanaged Ombre writer/backup cron remains" in result.stderr
    assert state.read_text(encoding="utf-8") == original


def test_job_installer_rejects_a_different_unvalidated_job_environment(tmp_path):
    env_file, active, data = _write_production_env(tmp_path)
    unvalidated = tmp_path / "unvalidated.env"
    unvalidated.write_text("OMBRE_CONTAINER_NAME=ombre-brain\n", encoding="utf-8")
    env_file.write_text(
        env_file.read_text(encoding="utf-8").replace(
            f"OMBRE_JOB_ENV_FILE={env_file}",
            f"OMBRE_JOB_ENV_FILE={unvalidated}",
        ),
        encoding="utf-8",
    )
    crontab, state = _fake_crontab(tmp_path)
    original = "17 * * * * safe-job\n"
    state.write_text(original, encoding="utf-8")
    docker, calls = _fake_deploy_docker(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "install_nas_jobs.sh"),
            "--env-file",
            str(env_file),
        ],
        env={
            **os.environ,
            "CRONTAB_BIN": str(crontab),
            "CRON_STATE": str(state),
            "TMPDIR": str(tmp_path),
            "DOCKER_BIN": str(docker),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(active / "config.yaml"),
            "FAKE_LIVE_SHA": "f" * 64,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "not the validated --env-file" in result.stderr
    assert state.read_text(encoding="utf-8") == original
    assert not calls.exists()


def test_job_installer_does_not_overwrite_a_concurrent_crontab_edit(tmp_path):
    env_file, active, data = _write_production_env(tmp_path)
    state = tmp_path / "crontab.txt"
    state.write_text("17 * * * * safe-job\n", encoding="utf-8")
    list_count = tmp_path / "list-count"
    writes = tmp_path / "writes"
    crontab = tmp_path / "crontab"
    crontab.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -l ]; then\n"
        "  n=$(cat \"$CRON_LIST_COUNT\" 2>/dev/null || echo 0)\n"
        "  n=$((n+1)); printf '%s\\n' \"$n\" >\"$CRON_LIST_COUNT\"\n"
        "  if [ \"$n\" = 2 ]; then printf '%s\\n' '23 * * * * concurrent-safe-job' >>\"$CRON_STATE\"; fi\n"
        "  cat \"$CRON_STATE\"; exit 0\n"
        "fi\n"
        "printf 'write\\n' >>\"$CRON_WRITES\"\n"
        "cp \"$1\" \"$CRON_STATE\"\n",
        encoding="utf-8",
    )
    crontab.chmod(0o700)
    docker, calls = _fake_deploy_docker(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "install_nas_jobs.sh"),
            "--env-file",
            str(env_file),
        ],
        env={
            **os.environ,
            "CRONTAB_BIN": str(crontab),
            "CRON_STATE": str(state),
            "CRON_LIST_COUNT": str(list_count),
            "CRON_WRITES": str(writes),
            "TMPDIR": str(tmp_path),
            "DOCKER_BIN": str(docker),
            "CALL_LOG": str(calls),
            "FAKE_DATA_DIR": str(data),
            "FAKE_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "FAKE_CONFIG_FILE": str(active / "config.yaml"),
            "FAKE_LIVE_SHA": "f" * 64,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "crontab changed concurrently" in result.stderr
    assert "concurrent-safe-job" in state.read_text(encoding="utf-8")
    assert not writes.exists()
