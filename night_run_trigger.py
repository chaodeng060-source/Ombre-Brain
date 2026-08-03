"""Authenticated localhost trigger for the scheduled LMC-5 night job.

The API token is read from the container environment and is never accepted on
the command line.  Host cron only needs to run:

    docker exec ombre-brain python /app/night_run_trigger.py
"""

from __future__ import annotations

import json
import os
import sys
from http.client import HTTPConnection


HOST = "127.0.0.1"
PORT = 8000
PATH = "/api/maintenance/lmc5-night"
MAX_RESPONSE_BYTES = 64 * 1024


class NightTriggerHTTPError(RuntimeError):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"night trigger returned HTTP {status}")


def _safe_summary(payload: object) -> dict[str, object]:
    if type(payload) is not dict:
        raise ValueError("night response is not an object")
    allowed = {
        "ok",
        "contract",
        "run_id",
        "local_date",
        "stage",
        "already_complete",
        "complete",
        "degraded",
        "counts",
        "deferred_axes",
        "code",
    }
    return {key: payload[key] for key in allowed if key in payload}


def trigger() -> dict[str, object]:
    token = os.environ.get("OMBRE_API_TOKEN", "")
    if not token:
        raise RuntimeError("api token is unavailable")
    connection = HTTPConnection(HOST, PORT, timeout=3600)
    try:
        connection.request(
            "POST",
            PATH,
            body=b'{"schema_version":1}',
            headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        status = int(response.status)
        content_type = str(response.getheader("Content-Type", ""))
    finally:
        connection.close()
    if status != 200:
        raise NightTriggerHTTPError(status)
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise RuntimeError("night response content type is invalid")
    if len(payload) > MAX_RESPONSE_BYTES:
        raise RuntimeError("night response is too large")
    parsed = json.loads(payload)
    summary = _safe_summary(parsed)
    if summary.get("ok") is not True:
        raise RuntimeError(str(summary.get("code") or "night run failed"))
    return summary


def main() -> int:
    try:
        summary = trigger()
    except NightTriggerHTTPError as exc:
        print(
            f"LMC-5 night trigger failed: HTTP {exc.status}",
            file=sys.stderr,
        )
        return 1
    except (TimeoutError, OSError, ValueError, RuntimeError):
        print("LMC-5 night trigger failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
