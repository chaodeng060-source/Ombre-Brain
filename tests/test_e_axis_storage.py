from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import e_axis_storage as e_axis_storage_module

from e_axis_storage import (
    EAxisStorageBusy,
    EAxisStorageError,
    ensure_private_e_axis_directory,
    open_secure_e_axis_jsonl,
    secure_e_axis_lock,
)


E_AXIS_SIDECARS = (
    "e-shadow.jsonl",
    "e-shadow-attempts.jsonl",
    "e-shadow-coverage.jsonl",
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _append_line(path: Path, value: str = '{"ok":true}\n') -> None:
    with open_secure_e_axis_jsonl(path) as handle:
        handle.seek(0, os.SEEK_END)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_normal_directory_jsonl_and_lock_are_private(tmp_path):
    axis = ensure_private_e_axis_directory(tmp_path / ".axis")
    data = axis / "e-shadow.jsonl"
    lock = axis / "e-shadow.jsonl.lock"

    with secure_e_axis_lock(lock):
        _append_line(data)
        with pytest.raises(EAxisStorageBusy):
            with secure_e_axis_lock(lock, blocking=False):
                pytest.fail("a second lock unexpectedly succeeded")

    assert data.read_text(encoding="utf-8") == '{"ok":true}\n'
    assert _mode(axis) == 0o700
    assert _mode(data) == 0o600
    assert _mode(lock) == 0o600


@pytest.mark.skipif(
    not hasattr(os, "symlink"),
    reason="symlinks are unavailable",
)
@pytest.mark.parametrize("sidecar", E_AXIS_SIDECARS)
@pytest.mark.parametrize("kind", ["data", "lock"])
def test_file_symlink_is_rejected_without_touching_external_target(
    tmp_path,
    sidecar,
    kind,
):
    axis = ensure_private_e_axis_directory(tmp_path / ".axis")
    external = tmp_path / "external.txt"
    external.write_bytes(b"outside-sentinel")
    before_mode = _mode(external)
    path = axis / (sidecar if kind == "data" else f"{sidecar}.lock")
    path.symlink_to(external)

    with pytest.raises(EAxisStorageError):
        if kind == "data":
            _append_line(path)
        else:
            with secure_e_axis_lock(path):
                pytest.fail("symlink lock unexpectedly succeeded")

    assert external.read_bytes() == b"outside-sentinel"
    assert _mode(external) == before_mode
    assert path.is_symlink()


@pytest.mark.skipif(
    not hasattr(os, "link"),
    reason="hardlinks are unavailable",
)
@pytest.mark.parametrize("sidecar", E_AXIS_SIDECARS)
@pytest.mark.parametrize("kind", ["data", "lock"])
def test_hardlink_is_rejected_without_touching_external_target(
    tmp_path,
    sidecar,
    kind,
):
    axis = ensure_private_e_axis_directory(tmp_path / ".axis")
    external = tmp_path / "external.txt"
    external.write_bytes(b"outside-sentinel")
    before_mode = _mode(external)
    path = axis / (sidecar if kind == "data" else f"{sidecar}.lock")
    os.link(external, path)
    assert external.stat().st_nlink == 2

    with pytest.raises(EAxisStorageError):
        if kind == "data":
            _append_line(path)
        else:
            with secure_e_axis_lock(path):
                pytest.fail("hardlink lock unexpectedly succeeded")

    assert external.read_bytes() == b"outside-sentinel"
    assert _mode(external) == before_mode
    assert external.stat().st_nlink == 2


@pytest.mark.skipif(
    not hasattr(os, "symlink"),
    reason="symlinks are unavailable",
)
def test_symlink_ancestor_is_rejected_without_creating_external_files(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(external, target_is_directory=True)
    target = linked_parent / ".axis" / "e-shadow.jsonl"

    with pytest.raises(EAxisStorageError):
        _append_line(target)

    assert list(external.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="directory fsync is POSIX-only")
def test_first_creation_fsyncs_file_and_parent_directories(tmp_path, monkeypatch):
    real_fsync = os.fsync
    fsynced_modes: list[int] = []

    def recording_fsync(descriptor: int) -> None:
        fsynced_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    data = tmp_path / ".axis" / "e-shadow.jsonl"
    _append_line(data)

    # One directory sync persists .axis, another persists the JSONL entry.
    assert sum(stat.S_ISDIR(mode) for mode in fsynced_modes) >= 2
    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)


def test_existing_private_files_reopen_normally(tmp_path):
    data = tmp_path / ".axis" / "e-shadow.jsonl"
    _append_line(data, '{"n":1}\n')
    _append_line(data, '{"n":2}\n')

    with open_secure_e_axis_jsonl(data) as handle:
        assert handle.read().splitlines() == ['{"n":1}', '{"n":2}']


@pytest.mark.skipif(
    not hasattr(os, "symlink"),
    reason="symlinks are unavailable",
)
@pytest.mark.parametrize("sidecar", E_AXIS_SIDECARS)
@pytest.mark.parametrize("kind", ["data", "lock"])
def test_axis_directory_swap_after_open_cannot_redirect_writes(
    tmp_path,
    monkeypatch,
    sidecar,
    kind,
):
    axis = ensure_private_e_axis_directory(tmp_path / ".axis")
    parked = tmp_path / ".axis-parked"
    external = tmp_path / "external"
    external.mkdir()
    target_name = sidecar if kind == "data" else f"{sidecar}.lock"
    external_target = external / target_name
    external_target.write_bytes(b"outside-sentinel")
    real_open_private = e_axis_storage_module._open_private_e_axis_directory
    swapped = False

    def swap_after_open(directory):
        nonlocal swapped
        opened = real_open_private(directory)
        if not swapped:
            swapped = True
            axis.rename(parked)
            axis.symlink_to(external, target_is_directory=True)
        return opened

    monkeypatch.setattr(
        e_axis_storage_module,
        "_open_private_e_axis_directory",
        swap_after_open,
    )
    target = axis / target_name

    with pytest.raises(EAxisStorageError):
        if kind == "data":
            _append_line(target)
        else:
            with secure_e_axis_lock(target):
                pytest.fail("redirected lock unexpectedly succeeded")

    assert swapped
    assert external_target.read_bytes() == b"outside-sentinel"
    assert axis.is_symlink()


def test_real_ancestor_swap_during_open_fails_before_sidecar_write(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "owned"
    ancestor = parent / "ancestor"
    ancestor.mkdir(parents=True)
    replacement = parent / "replacement"
    replacement.mkdir()
    parked = parent / "parked"
    directory_flags = e_axis_storage_module._directory_flags()
    real_open = e_axis_storage_module.os.open
    swapped = False

    def swap_after_stat(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "ancestor" and dir_fd is not None and not swapped:
            swapped = True
            ancestor.rename(parked)
            replacement.rename(ancestor)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(e_axis_storage_module.os, "open", swap_after_stat)
    monkeypatch.setattr(
        e_axis_storage_module,
        "_directory_flags",
        lambda: directory_flags,
    )
    target = ancestor / ".axis" / "e-shadow.jsonl"

    with pytest.raises(
        EAxisStorageError,
        match="ancestor identity changed",
    ):
        _append_line(target)

    assert swapped
    assert list(ancestor.iterdir()) == []
    assert not (ancestor / ".axis").exists()
