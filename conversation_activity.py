#!/usr/bin/env python3
"""Build the privacy-minimal VPS conversation activity anchor.

Only user-message dates and counts leave the chat host. Message text is never
copied into the summary consumed by the NAS-side read-only patrol.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


SCHEMA = "ombre.conversation-activity/v1"
TIMEZONE_NAME = "Asia/Shanghai"
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_NAME)
SUPPORTED_LOGS = (
    "room__main.jsonl",
    "dm__claude.jsonl",
    "dm__xiaojuan.jsonl",
    "dm__hajimi.jsonl",
    "dm__glm.jsonl",
)


def _parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _parse_recorded_at(value: object) -> datetime | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(
                LOCAL_TIMEZONE
            )
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    # Legacy records sometimes contain only HH:MM. They cannot be assigned to
    # a natural day safely, so fail closed instead of guessing from file mtime.
    if len(text) < 10 or text[4:5] != "-" or text[7:8] != "-":
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
    return parsed.astimezone(LOCAL_TIMEZONE)


def summarize_message_logs(
    messages_dir: Path,
    *,
    start_date: date,
    generated_at: datetime | None = None,
) -> dict:
    """Count real user chat messages by Asia/Shanghai natural day."""
    counts: Counter[str] = Counter()
    malformed_lines = 0
    timestampless_user_messages = 0
    scanned_logs: list[str] = []

    for name in SUPPORTED_LOGS:
        path = messages_dir / name
        if not path.is_file():
            continue
        scanned_logs.append(name)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (TypeError, json.JSONDecodeError):
                    malformed_lines += 1
                    continue
                if not isinstance(record, dict) or record.get("sender") != "user":
                    continue
                event_type = record.get("eventType")
                if event_type not in {None, "chat_message"}:
                    continue
                stamp = _parse_recorded_at(
                    record.get("recordedAt")
                    if record.get("recordedAt") is not None
                    else record.get("ts")
                )
                if stamp is None:
                    timestampless_user_messages += 1
                    continue
                day = stamp.date()
                if day >= start_date:
                    counts[day.isoformat()] += 1

    now = generated_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return {
        "schema": SCHEMA,
        "timezone": TIMEZONE_NAME,
        "start_date": start_date.isoformat(),
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "daily_user_messages": dict(sorted(counts.items())),
        "source": {
            "kind": "vps_append_only_chat_counts",
            "logs": scanned_logs,
            "malformed_lines": malformed_lines,
            "timestampless_user_messages": timestampless_user_messages,
            "contains_message_text": False,
        },
    }


def validate_activity_summary(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("activity summary must be a JSON object")
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported activity summary schema")
    if payload.get("timezone") != TIMEZONE_NAME:
        raise ValueError("activity summary timezone must be Asia/Shanghai")
    start_date = _parse_date(payload.get("start_date"))
    if start_date is None:
        raise ValueError("activity summary start_date is invalid")
    raw_counts = payload.get("daily_user_messages")
    if not isinstance(raw_counts, dict):
        raise ValueError("daily_user_messages must be an object")
    counts: dict[str, int] = {}
    for raw_day, raw_count in raw_counts.items():
        day = _parse_date(raw_day)
        if day is None or day < start_date:
            raise ValueError("activity summary contains an invalid day")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            raise ValueError("activity counts must be non-negative integers")
        counts[day.isoformat()] = raw_count
    return {
        "schema": SCHEMA,
        "timezone": TIMEZONE_NAME,
        "start_date": start_date.isoformat(),
        "daily_user_messages": dict(sorted(counts.items())),
    }


def load_activity_summary(path: str) -> dict:
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("activity summary is not valid JSON") from exc
    return validate_activity_summary(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只读汇总 VPS 对话活跃日；输出不含消息正文"
    )
    parser.add_argument("--messages-dir", required=True)
    parser.add_argument("--start-date", default="2026-07-30")
    parser.add_argument("--out", default="-", help="默认 stdout；可写隔离 JSON 文件")
    args = parser.parse_args()

    start_date = _parse_date(args.start_date)
    if start_date is None:
        raise SystemExit(f"非法 --start-date：{args.start_date}")
    summary = summarize_message_logs(
        Path(args.messages_dir),
        start_date=start_date,
    )
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.out == "-":
        sys.stdout.write(encoded)
    else:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
