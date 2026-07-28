"""Re-entrant cross-process maintenance barrier for the Ombre vault.

Ordinary writers take a shared lease.  Operations that need one coherent
filesystem view, such as an LMC-5 snapshot followed by apply, take an exclusive
lease for the complete window.  Nested calls from the same asyncio task/thread
reuse the outer lease so a strict high-level writer can safely call the bucket,
vector, review, and ledger layers without deadlocking.

POSIX uses ``flock(LOCK_SH/LOCK_EX)``.  Windows has no shared equivalent in
``msvcrt``, so both lease types serialize on one byte; this is slower but keeps
the same safety contract.
"""

from __future__ import annotations

import asyncio
import contextvars
import errno
import os
import stat
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Iterator


LOCK_DIRECTORY = ".locks"
LOCK_NAME = "lmc5-maintenance.lock"


class MaintenanceBarrierError(RuntimeError):
    """The maintenance boundary is unsafe or used incorrectly."""


class MaintenanceBarrierTimeout(MaintenanceBarrierError):
    """A lease did not become available within the requested timeout."""


class _MaintenanceBarrierBusy(MaintenanceBarrierError):
    """The non-blocking file-lock probe found an active conflicting lease."""


@dataclass
class _HeldLease:
    mode: str
    depth: int
    handle: BinaryIO
    owner_task_id: int | None
    owner_thread_id: int


_HELD: dict[tuple[str, str], _HeldLease] = {}
_HELD_GUARD = threading.Lock()
_LEASE_CONTEXT: contextvars.ContextVar[tuple[tuple[str, str], ...]] = (
    contextvars.ContextVar("ombre_maintenance_leases", default=())
)


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _current_task_id() -> int | None:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    return id(task) if task is not None else None


def _context_leases() -> dict[str, str]:
    return dict(_LEASE_CONTEXT.get())


def _reset_after_fork() -> None:
    # ``flock`` state is tied to the inherited open-file description.  The
    # child must close its duplicate descriptors without issuing LOCK_UN,
    # which could otherwise disturb the still-running parent.
    # Never acquire the inherited threading.Lock here: another parent thread
    # may have owned it at fork time and that owner no longer exists in the
    # child.  Replacing the guard is the only deadlock-safe reset.
    global _HELD_GUARD
    handles = {id(lease.handle): lease.handle for lease in _HELD.values()}
    _HELD.clear()
    _HELD_GUARD = threading.Lock()
    _LEASE_CONTEXT.set(())
    for handle in handles.values():
        try:
            handle.close()
        except OSError:
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _prepare_lock_path(root: Path) -> Path:
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise MaintenanceBarrierError("maintenance root is unavailable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise MaintenanceBarrierError("maintenance root must be a real directory")

    lock_dir = root / LOCK_DIRECTORY
    try:
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        lock_dir_stat = lock_dir.lstat()
        if (
            stat.S_ISLNK(lock_dir_stat.st_mode)
            or not stat.S_ISDIR(lock_dir_stat.st_mode)
        ):
            raise MaintenanceBarrierError(
                "maintenance lock directory must be real"
            )
        if os.name != "nt":
            os.chmod(lock_dir, 0o700)
    except MaintenanceBarrierError:
        raise
    except OSError as exc:
        raise MaintenanceBarrierError(
            "unable to prepare maintenance lock directory"
        ) from exc
    return lock_dir / LOCK_NAME


def _open_lock_file(path: Path) -> BinaryIO:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            os.close(descriptor)
            raise MaintenanceBarrierError(
                "maintenance lock file identity is unsafe"
            )
        if os.name != "nt":
            os.chmod(path, 0o600)
        return os.fdopen(descriptor, "r+b", buffering=0)
    except MaintenanceBarrierError:
        raise
    except OSError as exc:
        raise MaintenanceBarrierError(
            "unable to open maintenance lock file"
        ) from exc


def _acquire_handle(
    path: Path,
    mode: str,
    *,
    blocking: bool = True,
) -> BinaryIO:
    handle = _open_lock_file(path)
    try:
        if os.name == "nt":
            import msvcrt

            if os.path.getsize(path) == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            operation = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            msvcrt.locking(handle.fileno(), operation, 1)
        else:
            import fcntl

            operation = fcntl.LOCK_SH if mode == "shared" else fcntl.LOCK_EX
            if not blocking:
                operation |= fcntl.LOCK_NB
            fcntl.flock(handle.fileno(), operation)
        return handle
    except OSError as exc:
        handle.close()
        if not blocking and exc.errno in {
            errno.EACCES,
            errno.EAGAIN,
            errno.EWOULDBLOCK,
        }:
            raise _MaintenanceBarrierBusy("maintenance barrier is busy") from exc
        raise MaintenanceBarrierError(
            "unable to acquire maintenance lock"
        ) from exc
    except Exception:
        handle.close()
        raise


def _release_handle(handle: BinaryIO) -> None:
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


def _nested_key(lock_path: Path, mode: str) -> tuple[str, str] | None:
    path_key = os.fspath(lock_path)
    token = _context_leases().get(path_key)
    if token is None:
        return None
    key = (path_key, token)
    current_task_id = _current_task_id()
    current_thread_id = threading.get_ident()
    with _HELD_GUARD:
        held = _HELD.get(key)
        if held is None:
            return None

        # asyncio copies ContextVars into child tasks and asyncio.to_thread.
        # A child task is a competing writer and must acquire its own file
        # lease.  A to_thread call has no asyncio task, so it may reuse the
        # parent task's lease; that is the sync bridge used by the night job.
        if current_task_id is not None:
            if held.owner_task_id is not None:
                if current_task_id != held.owner_task_id:
                    return None
            elif current_thread_id != held.owner_thread_id:
                return None
        elif (
            held.owner_task_id is None
            and current_thread_id != held.owner_thread_id
        ):
            return None

        if held.mode == "shared" and mode == "exclusive":
            raise MaintenanceBarrierError(
                "maintenance lease cannot upgrade shared to exclusive"
            )
        held.depth += 1
        return key


def _register_new(
    lock_path: Path,
    mode: str,
    handle: BinaryIO,
) -> tuple[tuple[str, str], contextvars.Token[tuple[tuple[str, str], ...]]]:
    path_key = os.fspath(lock_path)
    token = uuid.uuid4().hex
    key = (path_key, token)
    task_id = _current_task_id()
    thread_id = threading.get_ident()
    with _HELD_GUARD:
        if key in _HELD:
            raise MaintenanceBarrierError("maintenance lease token collision")
        _HELD[key] = _HeldLease(
            mode=mode,
            depth=1,
            handle=handle,
            owner_task_id=task_id,
            owner_thread_id=thread_id,
        )
    context = _context_leases()
    context[path_key] = token
    context_token = _LEASE_CONTEXT.set(tuple(sorted(context.items())))
    return key, context_token


def _leave(key: tuple[str, str]) -> BinaryIO | None:
    with _HELD_GUARD:
        held = _HELD.get(key)
        if held is None:
            raise MaintenanceBarrierError("maintenance lease is not held")
        held.depth -= 1
        if held.depth > 0:
            return None
        del _HELD[key]
        return held.handle


async def _acquire_handle_async(
    path: Path,
    mode: str,
    *,
    timeout: float | None,
) -> BinaryIO:
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout
    while True:
        try:
            return _acquire_handle(path, mode, blocking=False)
        except _MaintenanceBarrierBusy as exc:
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise MaintenanceBarrierTimeout(
                        "maintenance lease acquisition timed out"
                    ) from exc
                await asyncio.sleep(min(0.025, remaining))
            else:
                await asyncio.sleep(0.025)


class MaintenanceBarrier:
    """Shared writer/exclusive maintenance leases for one vault root."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = _absolute(root)
        self.lock_path = _prepare_lock_path(self.root)

    @contextmanager
    def _sync_lease(self, mode: str) -> Iterator[None]:
        key = _nested_key(self.lock_path, mode)
        context_token = None
        if key is None:
            # A blocking flock in an event-loop task can freeze the loop while
            # its peer task owns the exclusive lease.  Sync leaf calls may
            # still run there, but they fail closed rather than block.
            blocking = _current_task_id() is None
            handle = _acquire_handle(
                self.lock_path,
                mode,
                blocking=blocking,
            )
            key, context_token = _register_new(self.lock_path, mode, handle)
        try:
            yield
        finally:
            handle = _leave(key)
            if handle is not None:
                _release_handle(handle)
            if context_token is not None:
                _LEASE_CONTEXT.reset(context_token)

    @asynccontextmanager
    async def _async_lease(
        self,
        mode: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[None]:
        key = _nested_key(self.lock_path, mode)
        context_token = None
        if key is None:
            handle = await _acquire_handle_async(
                self.lock_path,
                mode,
                timeout=timeout,
            )
            key, context_token = _register_new(self.lock_path, mode, handle)
        try:
            yield
        finally:
            handle = _leave(key)
            if handle is not None:
                _release_handle(handle)
            if context_token is not None:
                _LEASE_CONTEXT.reset(context_token)

    def shared(self):
        return self._sync_lease("shared")

    def exclusive(self):
        return self._sync_lease("exclusive")

    def shared_async(self, *, timeout: float | None = None):
        return self._async_lease("shared", timeout=timeout)

    def exclusive_async(self, *, timeout: float | None = None):
        return self._async_lease("exclusive", timeout=timeout)
