"""Filesystem primitives for private E-axis append-only sidecars.

The E0 score, attempt, and coverage ledgers live outside the recall corpus but
still contain private operational metadata.  This module gives those ledgers a
small, shared filesystem boundary instead of relying on ordinary ``open()``:

* every existing ancestor must be a real directory, never a symlink;
* the private E directory is owned by this process and mode ``0700`` on POSIX;
* data and lock files are real, singly-linked, process-owned regular files;
* POSIX opens use ``O_NOFOLLOW`` and all platforms request ``O_CLOEXEC`` where
  available;
* newly-created files and their parent directory entries are fsynced.

Callers remain responsible for validating JSONL rows and for flushing/fsyncing
after each append.  ``open_secure_e_axis_jsonl`` only establishes the safe file
identity and returns a text handle positioned wherever the caller chooses.
"""

from __future__ import annotations

import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class EAxisStorageError(RuntimeError):
    """The requested E-axis storage path is unavailable or unsafe."""


class EAxisStorageBusy(EAxisStorageError):
    """A non-blocking E-axis lock is already held."""


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _expected_uid() -> int | None:
    getter = getattr(os, "geteuid", None)
    return int(getter()) if callable(getter) else None


def _require_owner(st: os.stat_result, *, label: str) -> None:
    expected = _expected_uid()
    if expected is not None and st.st_uid != expected:
        raise EAxisStorageError(f"{label} owner is unsafe")


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if (
        not nofollow
        or not directory
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
    ):
        raise EAxisStorageError("secure directory-fd storage is unsupported")
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
        | directory
    )


def _verify_directory(descriptor: int, *, label: str) -> os.stat_result:
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise EAxisStorageError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(opened.st_mode):
        raise EAxisStorageError(f"{label} is unsafe")
    return opened


def _open_directory_chain(directory: Path) -> int:
    """Open an absolute directory one component at a time, never by pathname."""

    absolute = _absolute(directory)
    flags = _directory_flags()
    try:
        descriptor = os.open(absolute.anchor or os.sep, flags)
    except OSError as exc:
        raise EAxisStorageError("E-axis directory anchor is unavailable") from exc
    try:
        _verify_directory(descriptor, label="E-axis directory anchor")
        for component in absolute.parts[1:]:
            try:
                expected = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise EAxisStorageError(
                    "E-axis directory ancestors must exist"
                ) from exc
            if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(
                expected.st_mode
            ):
                raise EAxisStorageError(
                    "E-axis directory ancestors must be real directories"
                )
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise EAxisStorageError(
                    "E-axis directory ancestors must be real directories"
                ) from exc
            try:
                opened = _verify_directory(
                    child,
                    label="E-axis directory ancestor",
                )
                if (opened.st_dev, opened.st_ino) != (
                    expected.st_dev,
                    expected.st_ino,
                ):
                    raise EAxisStorageError(
                        "E-axis directory ancestor identity changed"
                    )
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _verify_directory_entry(
    descriptor: int,
    parent_descriptor: int,
    name: str,
) -> os.stat_result:
    opened = _verify_directory(descriptor, label="E-axis directory")
    try:
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise EAxisStorageError("E-axis directory identity changed") from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise EAxisStorageError("E-axis directory identity changed")
    _require_owner(opened, label="E-axis directory")
    _require_owner(current, label="E-axis directory")
    return opened


def _open_private_e_axis_directory(directory: Path) -> tuple[int, int, str]:
    """Open/create only the final private leaf beneath a held real parent fd."""

    absolute = _absolute(directory)
    if absolute == Path(absolute.anchor):
        raise EAxisStorageError("E-axis directory leaf is invalid")
    parent_descriptor = _open_directory_chain(absolute.parent)
    name = absolute.name
    created = False
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise EAxisStorageError("unable to create E-axis directory") from exc
        try:
            descriptor = os.open(
                name,
                _directory_flags(),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise EAxisStorageError(
                "E-axis directory must be a real directory"
            ) from exc
        try:
            _verify_directory_entry(
                descriptor,
                parent_descriptor,
                name,
            )
            if os.name != "nt":
                os.fchmod(descriptor, 0o700)
                secured = _verify_directory_entry(
                    descriptor,
                    parent_descriptor,
                    name,
                )
                if stat.S_IMODE(secured.st_mode) != 0o700:
                    raise EAxisStorageError(
                        "E-axis directory permissions are unsafe"
                    )
            if created:
                os.fsync(descriptor)
                os.fsync(parent_descriptor)
            return descriptor, parent_descriptor, name
        except Exception:
            os.close(descriptor)
            raise
    except Exception:
        os.close(parent_descriptor)
        raise


def ensure_private_e_axis_directory(
    directory: str | os.PathLike[str],
) -> Path:
    """Return a real process-owned E directory, creating only its final leaf."""

    path = _absolute(directory)
    descriptor, parent_descriptor, name = _open_private_e_axis_directory(path)
    try:
        _verify_directory_entry(descriptor, parent_descriptor, name)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    return path


def _verify_open_regular_file(
    descriptor: int,
    directory_descriptor: int,
    name: str,
    *,
    label: str,
) -> os.stat_result:
    try:
        opened = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise EAxisStorageError(f"{label} identity is unavailable") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or opened.st_nlink != 1
        or current.st_nlink != 1
        or opened.st_dev != current.st_dev
        or opened.st_ino != current.st_ino
    ):
        raise EAxisStorageError(f"{label} identity is unsafe")
    _require_owner(opened, label=label)
    _require_owner(current, label=label)
    return opened


def _open_secure_file(
    path: Path,
    *,
    label: str,
) -> tuple[int, bool, int, int, str, str]:
    directory_descriptor, parent_descriptor, directory_name = (
        _open_private_e_axis_directory(path.parent)
    )
    name = path.name
    common_flags = os.O_RDWR
    common_flags |= getattr(os, "O_BINARY", 0)
    common_flags |= getattr(os, "O_CLOEXEC", 0)
    common_flags |= getattr(os, "O_NOFOLLOW", 0)

    created = False
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                name,
                common_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_descriptor,
            )
            created = True
        except FileExistsError:
            try:
                descriptor = os.open(
                    name,
                    common_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise EAxisStorageError(f"unable to open {label}") from exc
        except OSError as exc:
            raise EAxisStorageError(f"unable to create {label}") from exc

        # Validate before chmod so a pre-existing hardlink cannot change the
        # permissions of an out-of-bound target.
        _verify_open_regular_file(
            descriptor,
            directory_descriptor,
            name,
            label=label,
        )
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
            secured = _verify_open_regular_file(
                descriptor,
                directory_descriptor,
                name,
                label=label,
            )
            if stat.S_IMODE(secured.st_mode) != 0o600:
                raise EAxisStorageError(f"{label} permissions are unsafe")
        if created:
            os.fsync(descriptor)
            os.fsync(directory_descriptor)
        _verify_directory_entry(
            directory_descriptor,
            parent_descriptor,
            directory_name,
        )
        return (
            descriptor,
            created,
            directory_descriptor,
            parent_descriptor,
            directory_name,
            name,
        )
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)
        os.close(parent_descriptor)
        raise


@contextmanager
def open_secure_e_axis_jsonl(
    path: str | os.PathLike[str],
) -> Iterator[TextIO]:
    """Open one safe private JSONL file as UTF-8 ``r+`` text."""

    target = _absolute(path)
    (
        descriptor,
        _created,
        directory_descriptor,
        parent_descriptor,
        directory_name,
        name,
    ) = _open_secure_file(
        target,
        label="E-axis JSONL file",
    )
    try:
        with os.fdopen(
            descriptor,
            "r+",
            encoding="utf-8",
            newline="",
        ) as handle:
            descriptor = -1
            try:
                yield handle
            finally:
                _verify_open_regular_file(
                    handle.fileno(),
                    directory_descriptor,
                    name,
                    label="E-axis JSONL file",
                )
                _verify_directory_entry(
                    directory_descriptor,
                    parent_descriptor,
                    directory_name,
                )
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            os.close(directory_descriptor)
            os.close(parent_descriptor)


def _lock_descriptor(descriptor: int, *, blocking: bool) -> None:
    if os.name == "nt":
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        operation = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        try:
            msvcrt.locking(descriptor, operation, 1)
        except OSError as exc:
            if not blocking:
                raise EAxisStorageBusy("E-axis lock is busy") from exc
            raise EAxisStorageError("unable to acquire E-axis lock") from exc
        return

    import fcntl

    operation = fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(descriptor, operation)
    except OSError as exc:
        if not blocking and exc.errno in {
            errno.EACCES,
            errno.EAGAIN,
            errno.EWOULDBLOCK,
        }:
            raise EAxisStorageBusy("E-axis lock is busy") from exc
        raise EAxisStorageError("unable to acquire E-axis lock") from exc


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def secure_e_axis_lock(
    lock_path: str | os.PathLike[str],
    *,
    blocking: bool = True,
) -> Iterator[None]:
    """Hold an exclusive lock on a safe private E-axis lock file."""

    if type(blocking) is not bool:
        raise TypeError("blocking must be a boolean")
    target = _absolute(lock_path)
    (
        descriptor,
        _created,
        directory_descriptor,
        parent_descriptor,
        directory_name,
        name,
    ) = _open_secure_file(
        target,
        label="E-axis lock file",
    )
    locked = False
    try:
        _lock_descriptor(descriptor, blocking=blocking)
        locked = True
        _verify_open_regular_file(
            descriptor,
            directory_descriptor,
            name,
            label="E-axis lock file",
        )
        _verify_directory_entry(
            directory_descriptor,
            parent_descriptor,
            directory_name,
        )
        yield
        _verify_open_regular_file(
            descriptor,
            directory_descriptor,
            name,
            label="E-axis lock file",
        )
        _verify_directory_entry(
            directory_descriptor,
            parent_descriptor,
            directory_name,
        )
    finally:
        try:
            if locked:
                _unlock_descriptor(descriptor)
        finally:
            try:
                os.close(descriptor)
            finally:
                os.close(directory_descriptor)
                os.close(parent_descriptor)


__all__ = [
    "EAxisStorageBusy",
    "EAxisStorageError",
    "ensure_private_e_axis_directory",
    "open_secure_e_axis_jsonl",
    "secure_e_axis_lock",
]
