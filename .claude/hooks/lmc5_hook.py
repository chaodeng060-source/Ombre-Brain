#!/usr/bin/env python3
"""Claude lifecycle hook client for Ombre's LMC-5 HTTP bridge.

``session-end`` is a required background write.  It validates the complete
transcript before sending anything and exits non-zero unless the server returns
an exact durable acknowledgement.

``user-prompt-submit`` is a foreground enhancement.  It prints recall context
when available, but deliberately exits zero on every operational failure so a
memory outage cannot block the conversation.

The client never logs prompts, transcript bodies, response bodies, tokens, or
filesystem paths.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import secrets
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, TextIO


SCHEMA_VERSION = 1
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 12.0

MAX_HOOK_INPUT_CHARS = 256 * 1024
MAX_TRANSCRIPT_BYTES = 16 * 1024 * 1024
MAX_TRANSCRIPT_EVENTS = 4_096
MAX_TRANSCRIPT_LINE_BYTES = 1024 * 1024
MAX_PROMPT_CHARS = 12_000
MAX_PROMPT_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_OUTBOX_BATCH_BYTES = 32 * 1024 * 1024
MAX_OUTBOX_FLUSH_ITEMS = 32

_SAFE_ID_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,480}$")
_TRIVIAL_MESSAGES = frozenset(
    value.casefold()
    for value in (
        "嗯",
        "嗯嗯",
        "好",
        "好的",
        "好吧",
        "继续",
        "ok",
        "收到",
        "明白",
        "哈哈",
        "笑",
        "对",
        "是的",
        "?",
        "？",
        ".",
        "。",
    )
)
_RECALL_CUES = (
    "记得",
    "回忆",
    "想起来",
    "上次",
    "之前",
    "以前",
    "过去",
    "当时",
    "那次",
    "那件",
    "刚才",
    "昨天",
    "前天",
    "上周",
    "上个月",
    "去年",
    "后来",
    "时间线",
    "变化",
    "搬家",
    "迁移",
    "约定",
    "答应",
    "说过",
    "聊过",
    "发生过",
    "我忘了",
    "不确定",
    "想不起来",
    "是不是",
    "有没有",
)
_QUESTION_CUES = (
    "谁",
    "什么",
    "哪里",
    "哪儿",
    "哪个",
    "哪天",
    "几号",
    "多少",
    "什么时候",
    "为什么",
    "怎么",
    "如何",
    "在哪",
)
_DATED_OR_NAMED_ENTITY_RE = re.compile(
    r"(?:\b[A-Za-z][A-Za-z0-9_.-]{2,}\b|"
    r"\b20\d{2}[-/.年]\d{1,2}(?:[-/.月]\d{1,2})?|"
    r"\b\d{1,2}[:：]\d{2}\b)"
)


class HookInputError(ValueError):
    """A hook event or transcript violates the local input contract."""


class HookTransportError(RuntimeError):
    """The server could not provide a valid explicit acknowledgement."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Never forward the private hook token to a redirect target."""

    def redirect_request(self, *_args, **_kwargs):
        raise HookTransportError("hook_redirect_refused")


def _diagnostic(code: str) -> None:
    """Emit one body-free machine-readable diagnostic."""

    print(f"ombre_lmc5_hook_error={code}", file=sys.stderr)


def _read_hook_event(stream: TextIO) -> dict[str, Any]:
    raw = stream.read(MAX_HOOK_INPUT_CHARS + 1)
    if len(raw) > MAX_HOOK_INPUT_CHARS:
        raise HookInputError("hook_input_too_large")
    if not raw:
        raise HookInputError("hook_input_empty")
    try:
        event = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise HookInputError("hook_input_invalid_json") from exc
    if not isinstance(event, dict):
        raise HookInputError("hook_input_not_object")
    return event


def _session_id(event: Mapping[str, Any], *, required: bool) -> str | None:
    value = event.get("session_id") or event.get("sessionId")
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise HookInputError("session_id_missing" if value is None else "session_id_invalid")
    normalized = value.strip()
    if not _SAFE_ID_RE.fullmatch(normalized):
        raise HookInputError("session_id_invalid")
    return normalized


def _transcript_path(event: Mapping[str, Any]) -> Path:
    value = (
        event.get("transcript_path")
        or event.get("transcriptPath")
        or event.get("session_log")
        or event.get("sessionLog")
    )
    if not isinstance(value, str) or not value or "\x00" in value:
        raise HookInputError("transcript_path_missing")
    return Path(value)


def _explicit_event_id(event: Mapping[str, Any]) -> str | None:
    for key in ("uuid", "event_id", "eventId", "message_id", "messageId", "id"):
        value = event.get(key)
        if isinstance(value, str) and _SAFE_ID_RE.fullmatch(value):
            return f"{key}:{value}"
    message = event.get("message")
    if isinstance(message, Mapping):
        for key in ("uuid", "id", "message_id", "messageId"):
            value = message.get(key)
            if isinstance(value, str) and _SAFE_ID_RE.fullmatch(value):
                return f"message.{key}:{value}"
    return None


def _derived_event_id(session_id: str, line_number: int, exact_line: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(session_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(line_number).encode("ascii"))
    digest.update(b"\x00")
    digest.update(exact_line)
    return f"line-sha256:{digest.hexdigest()}"


def _logical_jsonl_line(raw_line: bytes) -> bytes:
    """Remove only the JSONL record delimiter, preserving the record bytes."""

    if raw_line.endswith(b"\n"):
        raw_line = raw_line[:-1]
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
    return raw_line


def load_transcript_events(
    path: Path,
    session_id: str,
    *,
    opener=open,
) -> list[dict[str, str]]:
    """Load and validate an entire transcript as exact JSONL record strings.

    No prefix is returned on failure.  This is intentional: SessionEnd must
    never acknowledge a partial transcript as though the whole archive landed.
    """

    events: list[dict[str, str]] = []
    used_ids: set[str] = set()
    total_bytes = 0

    try:
        handle: BinaryIO
        with opener(path, "rb") as handle:
            before = None
            try:
                before = os.fstat(handle.fileno())
            except (AttributeError, OSError):
                # Alternate openers used by callers/tests need not expose a
                # descriptor.  A real filesystem transcript always does.
                pass
            for line_number, raw_line in enumerate(handle, start=1):
                total_bytes += len(raw_line)
                if total_bytes > MAX_TRANSCRIPT_BYTES:
                    raise HookInputError("transcript_too_large")

                exact_line = _logical_jsonl_line(raw_line)
                if not exact_line.strip():
                    continue
                if len(exact_line) > MAX_TRANSCRIPT_LINE_BYTES:
                    raise HookInputError("transcript_line_too_large")
                if len(events) >= MAX_TRANSCRIPT_EVENTS:
                    raise HookInputError("transcript_too_many_events")

                try:
                    payload = exact_line.decode("utf-8")
                    parsed = json.loads(payload)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise HookInputError("transcript_invalid_jsonl") from exc
                if not isinstance(parsed, dict):
                    raise HookInputError("transcript_event_not_object")

                source_event_id = _explicit_event_id(parsed)
                if source_event_id is None or source_event_id in used_ids:
                    source_event_id = _derived_event_id(
                        session_id, line_number, exact_line
                    )
                if source_event_id in used_ids:
                    raise HookInputError("transcript_duplicate_event_id")
                used_ids.add(source_event_id)
                events.append(
                    {
                        "source_event_id": source_event_id,
                        "payload": payload,
                    }
                )
            if before is not None:
                after = os.fstat(handle.fileno())
                if (
                    before.st_dev != after.st_dev
                    or before.st_ino != after.st_ino
                    or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                    or total_bytes != after.st_size
                ):
                    raise HookInputError("transcript_changed_during_read")
    except HookInputError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise HookInputError("transcript_unreadable") from exc

    return events


def _base_url() -> str:
    raw = os.environ.get("OMBRE_HOOK_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise HookInputError("hook_url_invalid")
    hostname = parsed.hostname or ""
    if parsed.scheme == "http":
        is_loopback = hostname.casefold() == "localhost"
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise HookInputError("hook_plain_http_forbidden")
    return raw


def _timeout_seconds() -> float:
    raw = os.environ.get("OMBRE_HOOK_TIMEOUT_SECONDS", "")
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise HookInputError("hook_timeout_invalid") from exc
    if not 0.1 <= value <= 60.0:
        raise HookInputError("hook_timeout_invalid")
    return value


def _post_json(
    path: str,
    payload: Mapping[str, Any],
    *,
    urlopen=None,
) -> dict[str, Any]:
    body = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "ombre-lmc5-hook/1",
    }
    token = os.environ.get("OMBRE_HOOK_TOKEN", "")
    if not token:
        raise HookInputError("hook_token_missing")
    headers["X-Ombre-Hook-Token"] = token

    request = urllib.request.Request(
        f"{_base_url()}{path}",
        data=body,
        headers=headers,
        method="POST",
    )
    if urlopen is None:
        urlopen = urllib.request.build_opener(_RejectRedirects()).open
    try:
        with urlopen(request, timeout=_timeout_seconds()) as response:
            status = int(getattr(response, "status", response.getcode()))
            if not 200 <= status < 300:
                raise HookTransportError("hook_http_status")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HookTransportError:
        raise
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        raise HookTransportError("hook_request_failed") from exc
    except Exception as exc:
        raise HookTransportError("hook_request_failed") from exc

    if len(raw) > MAX_RESPONSE_BYTES:
        raise HookTransportError("hook_response_too_large")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HookTransportError("hook_response_invalid_json") from exc
    if not isinstance(result, dict):
        raise HookTransportError("hook_response_not_object")
    return result


def _extract_prompt(event: Mapping[str, Any]) -> str | None:
    for key in ("prompt", "user_message", "message", "content"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _is_trivial(prompt: str) -> bool:
    normalized = prompt.strip().casefold()
    return normalized in _TRIVIAL_MESSAGES


def _should_recall(prompt: str) -> bool:
    """Conservative passive-recall intent gate for companion chat.

    Active ``breath(query=...)`` remains available to the agent.  This gate
    only decides whether a user message should automatically pull historical
    context into the current turn.
    """
    normalized = prompt.strip()
    folded = normalized.casefold()
    if _is_trivial(normalized):
        return False
    if any(cue in folded for cue in _RECALL_CUES):
        return True
    if "?" in normalized or "？" in normalized:
        return True
    if any(cue in folded for cue in _QUESTION_CUES):
        return True
    return bool(_DATED_OR_NAMED_ENTITY_RE.search(normalized))


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if os.name == "nt":
            return
        raise HookInputError("outbox_sync_failed") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if os.name != "nt":
            raise HookInputError("outbox_sync_failed") from exc
    finally:
        os.close(descriptor)


def _secure_outbox_dir(outbox_dir: Path | None = None) -> Path:
    if outbox_dir is None:
        configured = os.environ.get("OMBRE_HOOK_OUTBOX_DIR", "").strip()
        if configured:
            outbox_dir = Path(configured).expanduser()
            if not outbox_dir.is_absolute():
                raise HookInputError("outbox_dir_invalid")
        else:
            state_home = (
                os.environ.get("XDG_STATE_HOME", "").strip()
                or os.environ.get("LOCALAPPDATA", "").strip()
            )
            base = (
                Path(state_home).expanduser()
                if state_home
                else Path.home() / ".local" / "state"
            )
            outbox_dir = base / "ombre-lmc5" / "outbox"
    try:
        outbox_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        directory_stat = outbox_dir.lstat()
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
            directory_stat.st_mode
        ):
            raise HookInputError("outbox_dir_unsafe")
        if os.name != "nt":
            os.chmod(outbox_dir, 0o700)
    except HookInputError:
        raise
    except OSError as exc:
        raise HookInputError("outbox_dir_unavailable") from exc
    return outbox_dir


def _validate_raw_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "session_id",
        "events",
    }:
        raise HookInputError("outbox_payload_invalid")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise HookInputError("outbox_payload_invalid")
    session_id = payload.get("session_id")
    events = payload.get("events")
    if not isinstance(session_id, str) or not _SAFE_ID_RE.fullmatch(session_id):
        raise HookInputError("outbox_payload_invalid")
    if not isinstance(events, list) or not events or len(events) > MAX_TRANSCRIPT_EVENTS:
        raise HookInputError("outbox_payload_invalid")

    normalized_events = []
    for event in events:
        if not isinstance(event, Mapping) or set(event) != {
            "source_event_id",
            "payload",
        }:
            raise HookInputError("outbox_payload_invalid")
        source_event_id = event.get("source_event_id")
        exact_payload = event.get("payload")
        if (
            not isinstance(source_event_id, str)
            or not _SAFE_ID_RE.fullmatch(source_event_id)
            or not isinstance(exact_payload, str)
        ):
            raise HookInputError("outbox_payload_invalid")
        try:
            parsed = json.loads(exact_payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HookInputError("outbox_payload_invalid") from exc
        if not isinstance(parsed, dict):
            raise HookInputError("outbox_payload_invalid")
        normalized_events.append(
            {
                "source_event_id": source_event_id,
                "payload": exact_payload,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "events": normalized_events,
    }


def _serialize_raw_payload(payload: Mapping[str, Any]) -> bytes:
    normalized = _validate_raw_payload(payload)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_OUTBOX_BATCH_BYTES:
        raise HookInputError("outbox_payload_too_large")
    return encoded


def _assert_safe_outbox_file(path: Path) -> None:
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise HookInputError("outbox_file_unavailable") from exc
    if (
        stat.S_ISLNK(file_stat.st_mode)
        or not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_nlink != 1
        or file_stat.st_size > MAX_OUTBOX_BATCH_BYTES
    ):
        raise HookInputError("outbox_file_unsafe")


def _spool_raw_payload(
    payload: Mapping[str, Any],
    *,
    outbox_dir: Path | None = None,
) -> Path:
    encoded = _serialize_raw_payload(payload)
    directory = _secure_outbox_dir(outbox_dir)
    digest = hashlib.sha256(encoded).hexdigest()
    destination = directory / f"batch-{digest}.json"
    if destination.exists() or destination.is_symlink():
        _assert_safe_outbox_file(destination)
        try:
            if destination.read_bytes() != encoded:
                raise HookInputError("outbox_digest_conflict")
        except HookInputError:
            raise
        except OSError as exc:
            raise HookInputError("outbox_file_unavailable") from exc
        return destination

    temporary = directory / (
        f".batch-{digest}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if os.name != "nt":
            os.chmod(destination, 0o600)
        _fsync_directory(directory)
    except HookInputError:
        raise
    except OSError as exc:
        raise HookInputError("outbox_write_failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    _assert_safe_outbox_file(destination)
    return destination


def _load_spooled_payload(path: Path) -> dict[str, Any]:
    _assert_safe_outbox_file(path)
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HookInputError("outbox_payload_corrupt") from exc
    normalized = _validate_raw_payload(payload)
    if _serialize_raw_payload(normalized) != encoded:
        raise HookInputError("outbox_payload_noncanonical")
    expected_name = f"batch-{hashlib.sha256(encoded).hexdigest()}.json"
    if path.name != expected_name:
        raise HookInputError("outbox_identity_invalid")
    return normalized


def _validate_raw_ack(
    payload: Mapping[str, Any],
    response: Mapping[str, Any],
) -> None:
    events = payload["events"]
    session_id = payload["session_id"]
    acknowledged = response.get("acknowledged")
    inserted = response.get("inserted")
    if (
        response.get("ok") is not True
        or response.get("session_id") != session_id
        or isinstance(acknowledged, bool)
        or acknowledged != len(events)
        or isinstance(inserted, bool)
        or not isinstance(inserted, int)
        or not 0 <= inserted <= len(events)
    ):
        raise HookTransportError("raw_ack_invalid")


def flush_outbox(
    *,
    post_json: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    outbox_dir: Path | None = None,
    max_items: int = MAX_OUTBOX_FLUSH_ITEMS,
) -> int:
    if post_json is None:
        post_json = _post_json
    directory = _secure_outbox_dir(outbox_dir)
    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
        raise HookInputError("outbox_flush_limit_invalid")
    paths = sorted(directory.glob("batch-*.json"))[:max_items]
    flushed = 0
    for path in paths:
        payload = _load_spooled_payload(path)
        response = post_json("/lmc5/raw-events", payload)
        _validate_raw_ack(payload, response)
        try:
            path.unlink()
            _fsync_directory(directory)
        except OSError as exc:
            raise HookInputError("outbox_remove_failed") from exc
        flushed += 1
    return flushed


def run_session_end(
    event: Mapping[str, Any],
    *,
    post_json: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    outbox_dir: Path | None = None,
) -> None:
    if post_json is None:
        post_json = _post_json
    session_id = _session_id(event, required=True)
    assert session_id is not None
    events = load_transcript_events(_transcript_path(event), session_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "events": events,
    }
    pending_path = _spool_raw_payload(payload, outbox_dir=outbox_dir)
    flush_outbox(post_json=post_json, outbox_dir=outbox_dir)
    if pending_path.exists():
        raise HookTransportError("raw_batch_pending")


def run_user_prompt_submit(
    event: Mapping[str, Any],
    *,
    post_json: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    output: TextIO | None = None,
) -> None:
    if post_json is None:
        post_json = _post_json
    if output is None:
        output = sys.stdout
    prompt = _extract_prompt(event)
    if prompt is None or not _should_recall(prompt):
        return
    if len(prompt) > MAX_PROMPT_CHARS or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise HookInputError("prompt_too_large")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prompt": prompt,
    }
    session_id = _session_id(event, required=False)
    if session_id is not None:
        payload["session_id"] = session_id
    response = post_json("/lmc5/recall-hook", payload)
    context = response.get("context")
    if response.get("ok") is not True or not isinstance(context, str):
        raise HookTransportError("recall_ack_invalid")
    if context:
        output.write(context)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in {
        "retry-outbox",
        "session-end",
        "user-prompt-submit",
    }:
        _diagnostic("usage")
        return 2
    if os.environ.get("OMBRE_HOOK_SKIP") == "1":
        return 0

    command = args[0]
    try:
        if command == "retry-outbox":
            flush_outbox()
        else:
            event = _read_hook_event(sys.stdin)
        if command == "session-end":
            run_session_end(event)
        elif command == "user-prompt-submit":
            run_user_prompt_submit(event)
    except (HookInputError, HookTransportError) as exc:
        code = str(exc)
        _diagnostic(code if _SAFE_ID_RE.fullmatch(code) else "internal")
        return 0 if command in {"retry-outbox", "user-prompt-submit"} else 1
    except Exception:
        _diagnostic("internal")
        return 0 if command in {"retry-outbox", "user-prompt-submit"} else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
