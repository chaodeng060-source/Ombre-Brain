"""Read-only exact-body before/after counts; no application or provider startup."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import frontmatter


def instant(value, naive_zone):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=naive_zone)
    return parsed.astimezone(timezone.utc)


def measure(root, cutover, *, hours=24, naive_timezone, as_of=None):
    zone = ZoneInfo(naive_timezone)
    cutover = instant(cutover, zone)
    as_of = as_of or datetime.now(timezone.utc)
    interval = timedelta(hours=hours)
    records, errors = [], Counter()
    fallback = 0
    # Only vault directories; do not accidentally count operator documents.
    for directory in ("permanent", "dynamic", "feel", "涩涩", "archive"):
        for path in (Path(root) / directory).rglob("*.md"):
            try:
                # frontmatter.parse avoids Post(**metadata) collisions on the
                # legacy YAML 'content' field; the Markdown body stays intact.
                metadata, body = frontmatter.parse(path.read_text(encoding="utf-8"))
                stamp = metadata.get("recorded_at") or metadata.get("created")
                when = instant(stamp, zone)
                fallback += not bool(metadata.get("recorded_at"))
                key = (str(metadata.get("world", "<absent>")),
                       str(metadata.get("type", "<absent>")),
                       hashlib.sha256(body.encode("utf-8")).hexdigest())
                records.append((when, str(metadata.get("id") or path), key))
            except (ValueError, TypeError, OSError) as exc:
                errors[type(exc).__name__] += 1
    records.sort(key=lambda r: (r[0], r[1]))

    def window(start, end):
        earlier, inside = set(), Counter()
        repeated = total = 0
        for when, _, key in records:
            if when >= end or when > as_of:
                continue
            if when < start:
                earlier.add(key)
                continue
            total += 1
            repeated += key in earlier or key in inside
            inside[key] += 1
        return {"start": start.isoformat(), "end": end.isoformat(),
                "window_complete": as_of >= end, "new_buckets": total,
                "body_equal_to_earlier_bucket": repeated,
                "within_window_duplicate_groups": sum(n > 1 for n in inside.values()),
                "within_window_extra_body_copies": sum(n-1 for n in inside.values())}

    return {"as_of": as_of.isoformat(), "hours": hours,
            "naive_timezone": naive_timezone, "parsed_buckets": len(records),
            "created_fallback_buckets": fallback, "read_errors": dict(errors),
            "before": window(cutover-interval, cutover),
            "after": window(cutover, cutover+interval),
            "scope": "world+type+exact body; equality is not event identity or permission to skip/delete",
            "nas_writes": 0, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cutover", required=True, help="actual deployed timestamp with UTC offset")
    parser.add_argument("--naive-timezone", required=True, help="verify the writer clock; no guessed timezone")
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()
    if args.hours <= 0 or datetime.fromisoformat(args.cutover.replace("Z", "+00:00")).tzinfo is None:
        parser.error("positive hours and timezone-aware cutover are required")
    report = measure(args.data_dir, args.cutover, hours=args.hours, naive_timezone=args.naive_timezone)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["read_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
