"""Dedicated cross-process gate for the LMC-5 raw-ingest endpoint.

The public endpoint takes a non-blocking shared lease for the complete ledger
initialization/append operation.  Deployment acceptance can take the exclusive
lease with ``python -m lmc5_ingest_guard hold`` without pausing unrelated Ombre
writers through the vault-wide maintenance barrier.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import signal
import stat
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


RAW_INGEST_LOCK_PATH = Path("/tmp/ombre-lmc5-raw-ingest.lock")
_SAFE_MARKER_NAME = re.compile(
    r"ombre-lmc5-[A-Za-z0-9][A-Za-z0-9._-]{0,116}\Z"
)
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_POLL_SECONDS = 0.05


class RawIngestGuardError(RuntimeError):
    """The raw-ingest coordination boundary is unsafe or unavailable."""


class RawIngestBusy(RawIngestGuardError):
    """An exclusive acceptance window currently blocks raw ingest."""


class RawIngestGuardTimeout(RawIngestGuardError):
    """The requested exclusive guard window timed out."""


def _tmp_directory_descriptor() -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    if not getattr(os, "O_NOFOLLOW", 0) or not getattr(os, "O_DIRECTORY", 0):
        raise RawIngestGuardError("secure /tmp access is unsupported")
    descriptor = -1
    try:
        descriptor = os.open("/tmp", flags)
        opened = os.fstat(descriptor)
        current = os.lstat("/tmp")
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RawIngestGuardError("/tmp identity is unsafe")
        return descriptor
    except RawIngestGuardError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise RawIngestGuardError("unable to open /tmp safely") from exc


def _validate_tmp_path(path: str | os.PathLike[str], *, field: str) -> Path:
    candidate = Path(os.fspath(path))
    if (
        not candidate.is_absolute()
        or candidate.parent != Path("/tmp")
        or not _SAFE_MARKER_NAME.fullmatch(candidate.name)
    ):
        raise RawIngestGuardError(
            f"{field} must be an absolute /tmp path with a safe basename"
        )
    return candidate


def _validate_lock_path(path: Path) -> Path:
    candidate = _validate_tmp_path(path, field="lock path")
    if candidate.name != RAW_INGEST_LOCK_PATH.name:
        # Tests may replace the module constant with another direct /tmp name;
        # production callers cannot supply or override the path through CLI.
        if candidate != RAW_INGEST_LOCK_PATH:
            raise RawIngestGuardError("raw-ingest lock path is not fixed")
    return candidate


def _validate_token(token: str) -> str:
    if not isinstance(token, str) or not _SAFE_TOKEN.fullmatch(token):
        raise RawIngestGuardError("token must be a short opaque machine token")
    return token


def _reject_lock_marker(path: Path) -> None:
    if path == RAW_INGEST_LOCK_PATH:
        raise RawIngestGuardError("guard marker must not be the lock path")


def _validate_opened_file(
    descriptor: int,
    directory_descriptor: int,
    name: str,
    *,
    kind: str,
) -> os.stat_result:
    try:
        opened = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise RawIngestGuardError(f"{kind} identity is unavailable") from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or opened.st_nlink != 1
        or current.st_nlink != 1
        or opened.st_uid != os.geteuid()
        or current.st_uid != os.geteuid()
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise RawIngestGuardError(f"{kind} identity is unsafe")
    return opened


def _open_lock_file() -> int:
    path = _validate_lock_path(RAW_INGEST_LOCK_PATH)
    directory_descriptor = _tmp_directory_descriptor()
    descriptor = -1
    created = False
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(
                path.name,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_descriptor,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
        _validate_opened_file(
            descriptor,
            directory_descriptor,
            path.name,
            kind="raw-ingest lock",
        )
        os.fchmod(descriptor, 0o600)
        secured = _validate_opened_file(
            descriptor,
            directory_descriptor,
            path.name,
            kind="raw-ingest lock",
        )
        if stat.S_IMODE(secured.st_mode) != 0o600:
            raise RawIngestGuardError("raw-ingest lock permissions are unsafe")
        if created:
            os.fsync(descriptor)
            os.fsync(directory_descriptor)
        result = descriptor
        descriptor = -1
        return result
    except RawIngestGuardError:
        raise
    except OSError as exc:
        raise RawIngestGuardError("unable to secure raw-ingest lock") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


def _acquire(mode: int, *, nonblocking: bool) -> int:
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - deployment is Linux
        raise RawIngestGuardError("POSIX flock is required") from exc
    descriptor = _open_lock_file()
    operation = mode | (fcntl.LOCK_NB if nonblocking else 0)
    try:
        fcntl.flock(descriptor, operation)
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
            raise RawIngestBusy("raw ingest is paused") from exc
        raise RawIngestGuardError("unable to acquire raw-ingest lock") from exc


def _release(descriptor: int) -> None:
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def shared_ingest_guard() -> Iterator[None]:
    """Take the endpoint's non-blocking shared lease."""

    import fcntl

    descriptor = _acquire(fcntl.LOCK_SH, nonblocking=True)
    try:
        yield
    finally:
        _release(descriptor)


@contextmanager
def shared_acceptance_write_guard() -> Iterator[None]:
    """Fence soft read-side writes during a deployment acceptance window.

    The production deployment already holds the raw-ingest guard exclusively
    from the Y/Z before-snapshot through the E after-snapshot.  Reusing that
    same coordination domain avoids a second lock and the gap between taking
    a fingerprint and starting E0.  Non-POSIX runtimes preserve the previous
    best-effort write behaviour because the deployment contract is Linux-only.
    """

    if os.name != "posix":  # pragma: no cover - deployment is POSIX
        yield
        return
    try:
        import fcntl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - defensive POSIX fallback
        raise RawIngestGuardError("POSIX flock is required") from exc
    with shared_ingest_guard():
        yield


@contextmanager
def exclusive_ingest_guard(*, timeout: float | None = None) -> Iterator[None]:
    """Take the deployment window's exclusive lease with an optional timeout."""

    import fcntl

    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise RawIngestGuardError("timeout must be positive")
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            descriptor = _acquire(fcntl.LOCK_EX, nonblocking=True)
            break
        except RawIngestBusy as exc:
            if deadline is not None and time.monotonic() >= deadline:
                raise RawIngestGuardTimeout(
                    "timed out acquiring raw-ingest guard"
                ) from exc
            time.sleep(_POLL_SECONDS)
    try:
        yield
    finally:
        _release(descriptor)


def _safe_marker_state(path: Path) -> tuple[int, int, bytes] | None:
    directory_descriptor = _tmp_directory_descriptor()
    descriptor = -1
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
        except FileNotFoundError:
            return None
        opened = _validate_opened_file(
            descriptor,
            directory_descriptor,
            path.name,
            kind="guard marker",
        )
        if stat.S_IMODE(opened.st_mode) != 0o600 or opened.st_size > 256:
            raise RawIngestGuardError("guard marker permissions or size are unsafe")
        content = os.read(descriptor, 257)
        if len(content) > 256:
            raise RawIngestGuardError("guard marker is too large")
        return opened.st_dev, opened.st_ino, content
    except RawIngestGuardError:
        raise
    except OSError as exc:
        raise RawIngestGuardError("unable to read guard marker safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


def _remove_marker(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
    expected_token: str | None = None,
) -> None:
    state = _safe_marker_state(path)
    if state is None:
        return
    device, inode, content = state
    if expected_identity is not None and (device, inode) != expected_identity:
        return
    if expected_token is not None and content != (expected_token + "\n").encode():
        return
    directory_descriptor = _tmp_directory_descriptor()
    try:
        current = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (current.st_dev, current.st_ino) != (device, inode):
            return
        os.unlink(path.name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RawIngestGuardError("unable to remove guard marker safely") from exc
    finally:
        os.close(directory_descriptor)


def _create_marker(path: Path, token: str) -> tuple[int, int]:
    directory_descriptor = _tmp_directory_descriptor()
    descriptor = -1
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            path.name,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        payload = (token + "\n").encode("ascii")
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        opened = _validate_opened_file(
            descriptor,
            directory_descriptor,
            path.name,
            kind="guard marker",
        )
        if stat.S_IMODE(opened.st_mode) != 0o600:
            raise RawIngestGuardError("guard marker permissions are unsafe")
        os.fsync(directory_descriptor)
        return opened.st_dev, opened.st_ino
    except RawIngestGuardError:
        raise
    except OSError as exc:
        raise RawIngestGuardError("unable to create guard marker safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


def signal_release(release: str | os.PathLike[str], token: str) -> None:
    """Create the authenticated release marker used by a running holder."""

    release_path = _validate_tmp_path(release, field="release")
    _reject_lock_marker(release_path)
    token = _validate_token(token)
    state = _safe_marker_state(release_path)
    if state is not None:
        if state[2] == (token + "\n").encode():
            return
        raise RawIngestGuardError("release marker already exists with another token")
    _create_marker(release_path, token)


def marker_matches(path: str | os.PathLike[str], token: str) -> bool:
    """Return whether a safe marker contains exactly the requested token."""

    marker_path = _validate_tmp_path(path, field="marker")
    _reject_lock_marker(marker_path)
    token = _validate_token(token)
    state = _safe_marker_state(marker_path)
    return state is not None and state[2] == (token + "\n").encode()


def hold_guard(
    *,
    ready: str | os.PathLike[str],
    release: str | os.PathLike[str],
    token: str,
    timeout: float,
) -> None:
    """Hold exclusive ingest access until a matching release marker arrives."""

    ready_path = _validate_tmp_path(ready, field="ready")
    release_path = _validate_tmp_path(release, field="release")
    _reject_lock_marker(ready_path)
    _reject_lock_marker(release_path)
    token = _validate_token(token)
    if ready_path == release_path:
        raise RawIngestGuardError("ready and release markers must differ")
    if not math.isfinite(timeout) or timeout <= 0:
        raise RawIngestGuardError("timeout must be positive")
    deadline = time.monotonic() + timeout
    ready_identity: tuple[int, int] | None = None
    with exclusive_ingest_guard(timeout=timeout):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RawIngestGuardTimeout("raw-ingest guard timed out")
        # Only the exclusive holder may clear stale regular markers. Unsafe
        # symlinks/hardlinks are rejected rather than followed or replaced.
        _remove_marker(ready_path)
        _remove_marker(release_path)
        ready_identity = _create_marker(ready_path, token)
        try:
            while True:
                state = _safe_marker_state(release_path)
                if state is not None and state[2] == (token + "\n").encode():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RawIngestGuardTimeout("raw-ingest guard timed out")
                time.sleep(min(_POLL_SECONDS, remaining))
        finally:
            if ready_identity is not None:
                _remove_marker(ready_path, expected_identity=ready_identity)
            _remove_marker(release_path, expected_token=token)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    hold = commands.add_parser("hold", help="hold the exclusive ingest guard")
    hold.add_argument("--ready", required=True)
    hold.add_argument("--release", required=True)
    hold.add_argument("--token", required=True)
    hold.add_argument("--timeout", type=float, default=180.0)
    release = commands.add_parser("release", help="signal a matching holder")
    release.add_argument("--release", required=True)
    release.add_argument("--token", required=True)
    status = commands.add_parser("status", help="verify an exact ready marker")
    status.add_argument("--ready", required=True)
    status.add_argument("--token", required=True)
    commands.add_parser(
        "probe-shared",
        help="non-blocking shared-lock probe without touching the ledger",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    previous_term = signal.getsignal(signal.SIGTERM)

    def terminate(_signum, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.command == "hold":
            hold_guard(
                ready=args.ready,
                release=args.release,
                token=args.token,
                timeout=args.timeout,
            )
        elif args.command == "release":
            signal_release(args.release, args.token)
        elif args.command == "status":
            matched = marker_matches(args.ready, args.token)
            print(
                json.dumps(
                    {
                        "ok": matched,
                        "command": "status",
                        "code": "ready" if matched else "not_ready",
                    }
                ),
                flush=True,
            )
            return 0 if matched else 1
        else:
            with shared_ingest_guard():
                pass
        print(json.dumps({"ok": True, "command": args.command}), flush=True)
        return 0
    except RawIngestBusy:
        print(
            json.dumps(
                {"ok": False, "command": args.command, "code": "raw_ingest.busy"}
            ),
            file=sys.stderr,
            flush=True,
        )
        return 4
    except RawIngestGuardTimeout as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 3
    except RawIngestGuardError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())
