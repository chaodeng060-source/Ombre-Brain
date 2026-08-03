from __future__ import annotations

import fcntl
import os
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import lmc5_ingest_guard as guard


@pytest.fixture
def guard_paths(monkeypatch):
    suffix = uuid4().hex
    lock = Path(f"/tmp/ombre-lmc5-raw-ingest-test-{suffix}.lock")
    ready = Path(f"/tmp/ombre-lmc5-ready-{suffix}")
    release = Path(f"/tmp/ombre-lmc5-release-{suffix}")
    monkeypatch.setattr(guard, "RAW_INGEST_LOCK_PATH", lock)
    yield lock, ready, release
    for path in (ready, release, lock):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def test_exclusive_blocks_shared_and_exception_releases(guard_paths):
    lock, _ready, _release = guard_paths
    with guard.exclusive_ingest_guard(timeout=0.5):
        lock_stat = lock.stat()
        assert stat.S_ISREG(lock_stat.st_mode)
        assert lock_stat.st_nlink == 1
        assert stat.S_IMODE(lock_stat.st_mode) == 0o600
        with pytest.raises(guard.RawIngestBusy):
            with guard.shared_ingest_guard():
                pass
        with pytest.raises(guard.RawIngestBusy):
            with guard.shared_acceptance_write_guard():
                pass

    with pytest.raises(RuntimeError, match="injected"):
        with guard.shared_ingest_guard():
            raise RuntimeError("injected")

    with guard.exclusive_ingest_guard(timeout=0.5):
        pass

    descriptor = guard._open_lock_file()
    try:
        assert fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    finally:
        guard._release(descriptor)


def test_acceptance_write_guard_preserves_non_posix_write_behavior(monkeypatch):
    monkeypatch.setattr(guard, "os", SimpleNamespace(name="nt"))

    def unexpected_shared_guard():
        raise AssertionError("non-POSIX runtime must not require flock")

    monkeypatch.setattr(guard, "shared_ingest_guard", unexpected_shared_guard)
    with guard.shared_acceptance_write_guard():
        pass


def test_hold_status_probe_and_o_excl_release(guard_paths, capsys):
    _lock, ready, release = guard_paths
    token = "acceptance-token-1"
    failures = []

    def run_holder():
        try:
            guard.hold_guard(
                ready=ready,
                release=release,
                token=token,
                timeout=2,
            )
        except BaseException as exc:
            failures.append(exc)

    holder = threading.Thread(target=run_holder)
    holder.start()
    deadline = time.monotonic() + 1
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert guard.main(["status", "--ready", str(ready), "--token", token]) == 0
    assert guard.main(["probe-shared"]) == 4
    assert guard.main(
        ["release", "--release", str(release), "--token", token]
    ) == 0
    holder.join(timeout=1)

    assert not holder.is_alive()
    assert failures == []
    assert not ready.exists()
    assert not release.exists()
    assert "raw_ingest.busy" in capsys.readouterr().err


def test_status_rejects_stale_token_and_hold_rejects_symlink(
    guard_paths,
    tmp_path,
):
    _lock, ready, release = guard_paths
    guard._create_marker(ready, "old-token")
    assert guard.marker_matches(ready, "new-token") is False

    ready.unlink()
    target = tmp_path / "target"
    target.write_text("untouched", encoding="utf-8")
    ready.symlink_to(target)
    with pytest.raises(guard.RawIngestGuardError):
        guard.hold_guard(
            ready=ready,
            release=release,
            token="new-token",
            timeout=0.5,
        )
    assert target.read_text(encoding="utf-8") == "untouched"
    with guard.shared_ingest_guard():
        pass


def test_lock_rejects_symlink_and_hardlink(guard_paths):
    lock, _ready, _release = guard_paths
    target = lock.with_name(f"{lock.name}.target")
    target.write_text("untouched", encoding="utf-8")
    target.chmod(0o600)
    try:
        lock.symlink_to(target)
        with pytest.raises(guard.RawIngestGuardError):
            with guard.shared_ingest_guard():
                pass
        lock.unlink()

        os.link(target, lock)
        with pytest.raises(guard.RawIngestGuardError):
            with guard.shared_ingest_guard():
                pass
        assert target.read_text(encoding="utf-8") == "untouched"
    finally:
        for path in (lock, target):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def test_markers_cannot_alias_lock_or_escape_dedicated_tmp_names(guard_paths):
    lock, _ready, release = guard_paths
    with pytest.raises(guard.RawIngestGuardError):
        guard.hold_guard(
            ready=lock,
            release=release,
            token="token",
            timeout=0.5,
        )
    with pytest.raises(guard.RawIngestGuardError):
        guard.signal_release("/tmp/unrelated-file", "token")
    with pytest.raises(guard.RawIngestGuardError):
        with guard.exclusive_ingest_guard(timeout=float("nan")):
            pass


def test_release_refuses_to_replace_stale_token(guard_paths):
    _lock, _ready, release = guard_paths
    guard._create_marker(release, "old-token")
    with pytest.raises(guard.RawIngestGuardError):
        guard.signal_release(release, "new-token")
    assert guard.marker_matches(release, "old-token") is True


def test_hold_timeout_cleans_markers_and_releases_lock(guard_paths):
    _lock, ready, release = guard_paths
    with pytest.raises(guard.RawIngestGuardTimeout):
        guard.hold_guard(
            ready=ready,
            release=release,
            token="timeout-token",
            timeout=0.05,
        )
    assert not ready.exists()
    assert not release.exists()
    with guard.shared_ingest_guard():
        pass
