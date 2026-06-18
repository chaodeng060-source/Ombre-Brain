"""Crash-safe filesystem writes used by the memory store."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

import frontmatter


@contextmanager
def advisory_file_lock(lock_path: str | os.PathLike[str]):
    """Hold an exclusive cross-process lock for one bucket."""
    path = os.path.abspath(os.fspath(lock_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if os.path.getsize(path) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _fsync_directory(directory: str) -> None:
    """Persist the directory entry where the platform supports it."""
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(file_path: str | os.PathLike[str], text: str) -> None:
    """Write a complete replacement in the target directory and atomically swap it."""
    target = os.path.abspath(os.fspath(file_path))
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)

    existing_mode = None
    if os.path.exists(target):
        existing_mode = stat.S_IMODE(os.stat(target).st_mode)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{Path(target).name}.",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        fd = -1
        if existing_mode is not None:
            os.chmod(tmp_path, existing_mode)
        os.replace(tmp_path, target)
        _fsync_directory(parent)
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def atomic_write_post(file_path: str | os.PathLike[str], post) -> None:
    """Atomically serialize and replace a python-frontmatter post."""
    atomic_write_text(file_path, frontmatter.dumps(post))
