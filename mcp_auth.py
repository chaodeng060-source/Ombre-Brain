"""Fail-closed bearer authentication for network API and MCP transports."""

from __future__ import annotations

import hmac
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any


ASGIScope = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]
PathMatcher = Callable[[object], bool]

_MCP_EXACT_PATHS = frozenset({"/mcp", "/mcp/", "/sse", "/sse/", "/messages"})
_TOKEN_RE = re.compile(rb"^[A-Za-z0-9._~-]{32,}$")
_UNAUTHORIZED_BODY = json.dumps(
    {"error": "unauthorized"},
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")


def is_mcp_transport_path(path: object) -> bool:
    """Return whether an ASGI path belongs to a network MCP transport."""
    normalized = str(path or "")
    return normalized in _MCP_EXACT_PATHS or normalized.startswith("/messages/")


def is_api_path(path: object) -> bool:
    """Return whether an ASGI path belongs to the dashboard/API surface."""
    normalized = str(path or "")
    return normalized == "/api" or normalized.startswith("/api/")


def _validated_token_bytes(token: object, *, variable: str) -> bytes:
    raw = str(token or "")
    if raw != raw.strip():
        raise ValueError(f"{variable} must not contain leading or trailing whitespace")
    try:
        encoded = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{variable} must use the documented ASCII token format") from exc
    if _TOKEN_RE.fullmatch(encoded) is None:
        raise ValueError(
            f"{variable} must contain at least 32 characters from [A-Za-z0-9._~-]"
        )
    return encoded


def require_network_token(
    environment: Mapping[str, str] | None = None,
    *,
    variable: str,
) -> str:
    """Load a network token or refuse to start with a weak/ambiguous value."""
    env = environment if environment is not None else os.environ
    token = str(env.get(variable, "") or "")
    if not token:
        raise RuntimeError(
            f"{variable} is required when Ombre Brain uses a network transport"
        )
    try:
        _validated_token_bytes(token, variable=variable)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return token


def require_mcp_token(
    environment: Mapping[str, str] | None = None,
) -> str:
    return require_network_token(environment, variable="OMBRE_MCP_TOKEN")


def require_api_token(
    environment: Mapping[str, str] | None = None,
) -> str:
    return require_network_token(environment, variable="OMBRE_API_TOKEN")


def _bearer_token(headers: object) -> bytes | None:
    if not isinstance(headers, (list, tuple)):
        return None
    values: list[bytes] = []
    for item in headers:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        name, value = item
        if isinstance(name, bytes) and name.lower() == b"authorization":
            if not isinstance(value, bytes):
                return None
            values.append(value)
    if len(values) != 1:
        return None
    scheme, separator, token = values[0].partition(b" ")
    candidate = token.strip()
    if (
        not separator
        or scheme.lower() != b"bearer"
        or not candidate
        or _TOKEN_RE.fullmatch(candidate) is None
    ):
        return None
    return candidate


class BearerAuthMiddleware:
    """Protect a path family without buffering streaming response bodies."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str,
        path_matcher: PathMatcher,
        realm: str,
    ) -> None:
        self.app = app
        self._token = _validated_token_bytes(token, variable="bearer token")
        self._path_matcher = path_matcher
        self._www_authenticate = (
            f'Bearer realm="{realm}", charset="UTF-8"'.encode("ascii")
        )

    async def __call__(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if (
            scope.get("type") != "http"
            or str(scope.get("method") or "").upper() == "OPTIONS"
            or not self._path_matcher(scope.get("path"))
        ):
            await self.app(scope, receive, send)
            return

        supplied = _bearer_token(scope.get("headers"))
        if supplied is not None and hmac.compare_digest(supplied, self._token):
            await self.app(scope, receive, send)
            return

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                    (b"www-authenticate", self._www_authenticate),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})


class MCPBearerAuthMiddleware(BearerAuthMiddleware):
    """Protect /mcp, /sse, and /messages with a pure ASGI wrapper."""

    def __init__(self, app: ASGIApp, *, token: str) -> None:
        super().__init__(
            app,
            token=token,
            path_matcher=is_mcp_transport_path,
            realm="Ombre Brain MCP",
        )


class APIBearerAuthMiddleware(BearerAuthMiddleware):
    """Protect /api and /api/* with the same strict bearer contract."""

    def __init__(self, app: ASGIApp, *, token: str) -> None:
        super().__init__(
            app,
            token=token,
            path_matcher=is_api_path,
            realm="Ombre Brain API",
        )
