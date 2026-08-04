"""Session-scoped injection history and same-turn content fingerprints.

Adapted from ``wuxuyun0606-collab/lmc-5`` commits 7fc7881 and 53a4aaa.
Ombre uses string bucket ids instead of the reference backend's integer ids,
but keeps the same contracts: hash-only state, expiry, locking, atomic replace,
and fail-open reads/writes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from collections.abc import Iterable
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX import compatibility
    fcntl = None  # type: ignore[assignment]


_CONTENT_SOURCE_PREFIX_RE = re.compile(
    r"""^\s*(?:
        \[(?:knowledge[_\s-]?base|pgvector|pg[_\s-]?fts|vector|fts|imprint|sqlite)\]
        |
        (?:knowledge[_\s-]?base|pgvector|pg[_\s-]?fts|vector|fts|imprint|sqlite)
        (?:\s*[:：|/]\s*|\s+)
    )+""",
    re.IGNORECASE | re.VERBOSE,
)


def _digest(value: str, size: int = 40) -> str:
    return hashlib.sha256((value or "").encode("utf-8", "ignore")).hexdigest()[:size]


def recall_identity(namespace: str, source_id: str) -> str:
    """Stable content-free identity for one source row."""
    return f"{namespace}\x1f{source_id}"


def default_content_fingerprint(content: str) -> str | None:
    """Hash normalized bodies so same content with different ids shares a slot."""
    text = unicodedata.normalize("NFKC", str(content or "")).lower()
    previous = None
    while text != previous:
        previous = text
        text = _CONTENT_SOURCE_PREFIX_RE.sub("", text, count=1)
    compact = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    if len(compact) < 8:
        return None
    return hashlib.sha256(compact.encode("utf-8", "ignore")).hexdigest()


class JsonFileRecallHistory:
    """Process-safe, hash-only history for per-turn recall injections."""

    def __init__(
        self,
        state_dir: Path,
        *,
        ttl_seconds: int = 2 * 86400,
        max_keys_per_session: int = 1024,
    ):
        self.state_dir = Path(state_dir)
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_keys_per_session = max(1, int(max_keys_per_session))

    def _paths(self, session_id: str) -> tuple[Path, Path]:
        token = _digest(session_id, 24)
        return (
            self.state_dir / f"session-{token}.json",
            self.state_dir / f"session-{token}.lock",
        )

    def _ensure_dir(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.state_dir.chmod(0o700)
        except OSError:
            pass

    @staticmethod
    def _read(path: Path) -> dict[str, float]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            seen = payload.get("seen") if isinstance(payload, dict) else None
            if isinstance(seen, dict):
                return {str(key): float(value) for key, value in seen.items()}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return {}

    @staticmethod
    def _hashed(keys: Iterable[str]) -> dict[str, str]:
        return {str(key): _digest(str(key)) for key in keys if key}

    def seen(self, session_id: str, keys: Iterable[str]) -> set[str]:
        if not session_id:
            return set()
        key_hashes = self._hashed(keys)
        if not key_hashes:
            return set()
        state_path, lock_path = self._paths(session_id)
        if (
            not self.state_dir.exists()
            or not state_path.exists()
            or not lock_path.exists()
        ):
            return set()
        try:
            with lock_path.open("r", encoding="utf-8") as lock:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
                stale = (
                    state_path.exists()
                    and time.time() - state_path.stat().st_mtime >= self.ttl_seconds
                )
                known = {} if stale else self._read(state_path)
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError:
            return set()
        return {key for key, hashed in key_hashes.items() if hashed in known}

    def mark(self, session_id: str, keys: Iterable[str]) -> None:
        if not session_id:
            return
        key_hashes = self._hashed(keys)
        if not key_hashes:
            return
        self._ensure_dir()
        state_path, lock_path = self._paths(session_id)
        with lock_path.open("a+", encoding="utf-8") as lock:
            try:
                os.fchmod(lock.fileno(), 0o600)
            except OSError:
                pass
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            now = time.time()
            stale = (
                state_path.exists()
                and now - state_path.stat().st_mtime >= self.ttl_seconds
            )
            known = {} if stale else self._read(state_path)
            for hashed in key_hashes.values():
                known[hashed] = now
            if len(known) > self.max_keys_per_session:
                newest = sorted(known.items(), key=lambda pair: pair[1], reverse=True)
                known = dict(newest[: self.max_keys_per_session])
            payload = json.dumps(
                {"version": 1, "updated_at": now, "seen": known},
                sort_keys=True,
                separators=(",", ":"),
            )
            fd, tmp_name = tempfile.mkstemp(
                prefix=state_path.name + ".",
                dir=str(self.state_dir),
            )
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                    tmp.write(payload)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                os.replace(tmp_name, state_path)
            finally:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        self._cleanup()

    def _cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        try:
            for path in list(self.state_dir.glob("session-*.json"))[:128]:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
        except OSError:
            pass
