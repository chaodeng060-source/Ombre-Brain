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
RESPONSE_TIMEOUT_SECONDS = 2 * 60 * 60


class NightTriggerHTTPError(RuntimeError):
    def __init__(self, status: int, code: str = "") -> None:
        self.status = status
        self.code = code
        detail = f" ({code})" if code else ""
        super().__init__(f"night trigger returned HTTP {status}{detail}")


def _connection_target() -> tuple[str, int]:
    host = os.environ.get("OMBRE_NIGHT_TRIGGER_HOST", HOST).strip()
    raw_port = os.environ.get(
        "OMBRE_NIGHT_TRIGGER_PORT",
        os.environ.get("OMBRE_HOST_PORT", str(PORT)),
    )
    port = int(raw_port)
    if host != "127.0.0.1" or not 1 <= port <= 65535:
        raise ValueError("night trigger target is invalid")
    return host, port


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


def _response_code(payload: bytes) -> str:
    """Read the server's refusal code without trusting the body's shape."""
    try:
        parsed = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return ""
    if type(parsed) is not dict:
        return ""
    code = parsed.get("code")
    if not isinstance(code, str):
        return ""
    code = code.strip()
    return code if 0 < len(code) <= 64 else ""


def trigger() -> dict[str, object]:
    token = os.environ.get("OMBRE_API_TOKEN", "")
    if not token:
        raise RuntimeError("api token is unavailable")
    # A full conservative run can legitimately take more than one hour while
    # draining a historical proposer backlog.  Keep the authenticated local
    # request alive long enough to receive its terminal ledger receipt; an
    # early client timeout does not cancel the server-side run and makes the
    # catch-up wrapper mistake healthy work for a failed round.
    host, port = _connection_target()
    connection = HTTPConnection(host, port, timeout=RESPONSE_TIMEOUT_SECONDS)
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
        # 服务端把拒绝原因放在 body 的 code 里（run.busy / night.unavailable …）。
        # 以前这里直接丢掉 body，日志只剩一个光秃秃的状态码，事后谁都判不了
        # 「是真故障还是撞上了正在跑的一轮」——2026-09-20 查这条 503 时卡在这。
        raise NightTriggerHTTPError(status, _response_code(payload))
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
        # 另一轮正在跑不是故障：single-flight 锁挡住这次调用，说明夜跑本身
        # 是活的。以前这里一律 return 1，cron 每撞上一次就记一笔「失败」，
        # 而容器日志里那一轮其实跑得好好的（2026-09-20 实测：昨夜跑了 5 轮，
        # proposer 积压从 1690 降到 9），把健康的工作记成了坏账。
        if exc.code in {"run.busy", "run.raced"}:
            print(
                f"LMC-5 night run already in flight ({exc.code}); not a failure",
                file=sys.stderr,
            )
            return 0
        detail = f" code={exc.code}" if exc.code else ""
        print(
            f"LMC-5 night trigger failed: HTTP {exc.status}{detail}",
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
