"""Count strict-summary reuse logs in a timezone-aware, half-open window.

Inputs must include all rotated logs for the window. This counts logical client
calls avoided by successful reuse, not HTTP retries, billing, or causal savings
against the previous release. No cache/vault read or model call is performed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re


EVENT = re.compile(r"recall_summary_cache at=(\d+\.\d+) event=(\w+) key=[0-9a-f]{64} saved_calls=(\d+)")


def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("window boundaries require a timezone")
    return parsed.timestamp()


def summarize(lines, start, end, *, now=None):
    if end <= start:
        raise ValueError("end must be after start")
    counts = Counter()
    saved = 0
    invalid = 0
    first = last = None
    for line in lines:
        if "recall_summary_cache " not in line:
            continue
        match = EVENT.search(line)
        if not match:
            invalid += 1
            continue
        timestamp, event, avoided = match.groups()
        timestamp = float(timestamp)
        if not start <= timestamp < end:
            continue
        first = timestamp if first is None else min(first, timestamp)
        last = timestamp if last is None else max(last, timestamp)
        counts[event] += 1
        saved += int(avoided)
    hits = sum(counts[key] for key in ("memory_hit", "persistent_hit", "coalesced_hit"))
    requests = counts["request"]
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    return {
        "start": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "window_elapsed": now >= end,
        "requested_hours": (end - start) / 3600,
        "eligible_requests": requests,
        "cache_hits": hits,
        "cache_hit_rate": hits / requests if requests and hits <= requests else None,
        "saved_logical_calls": saved,
        "provider_invocations": counts["provider_start"],
        "event_counts": dict(sorted(counts.items())),
        "malformed_cache_log_lines": invalid,
        "first_event_epoch": first,
        "last_event_epoch": last,
        "coverage": "supplied logs only; verify all rotations, process lifetime and cross-boundary requests separately",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    def lines():
        for path in args.logs:
            with Path(path).open() as stream:
                yield from stream
    print(json.dumps(summarize(lines(), parse_time(args.start), parse_time(args.end)), sort_keys=True))


if __name__ == "__main__":
    main()
