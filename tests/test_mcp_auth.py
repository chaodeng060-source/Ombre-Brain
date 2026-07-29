from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_auth import (
    APIBearerAuthMiddleware,
    MCPBearerAuthMiddleware,
    is_api_path,
    is_mcp_transport_path,
    require_api_token,
    require_mcp_token,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TOKEN = "correct-token-0123456789abcdefghijklmnop"


def _exercise(
    path: str,
    *,
    method: str = "POST",
    headers: list[tuple[bytes, bytes]] | None = None,
    expected_token: str = TEST_TOKEN,
    middleware_class=MCPBearerAuthMiddleware,
) -> tuple[int, list[dict[str, object]]]:
    calls = 0
    messages: list[dict[str, object]] = []

    async def downstream(scope, receive, send) -> None:
        nonlocal calls
        calls += 1
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    middleware = middleware_class(downstream, token=expected_token)
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": method,
                "path": path,
                "headers": headers or [],
            },
            receive,
            send,
        )
    )
    return calls, messages


@pytest.mark.parametrize(
    "path",
    [
        "/mcp",
        "/mcp/",
        "/sse",
        "/sse/",
        "/messages",
        "/messages/",
        "/messages/session-id",
    ],
)
def test_mcp_transport_path_matches_only_real_endpoints(path: str) -> None:
    assert is_mcp_transport_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/",
        "/health",
        "/api/buckets",
        "/mcp-evil",
        "/sse-old",
        "/message",
        "/messages-evil",
    ],
)
def test_mcp_transport_path_rejects_lookalikes(path: str) -> None:
    assert not is_mcp_transport_path(path)


@pytest.mark.parametrize("path", ["/api", "/api/", "/api/buckets"])
def test_api_path_matches_real_endpoints(path: str) -> None:
    assert is_api_path(path)


@pytest.mark.parametrize("path", ["", "/", "/health", "/apix", "/api-evil"])
def test_api_path_rejects_lookalikes(path: str) -> None:
    assert not is_api_path(path)


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"authorization", b"Bearer wrong-token")],
        [(b"authorization", f"Basic {TEST_TOKEN}".encode("ascii"))],
        [(b"authorization", b"Bearer")],
        [(b"authorization", b"Bearer correct token")],
        [(b"authorization", b"Bearer " + b"\xe9" * 40)],
        [
            (b"authorization", f"Bearer {TEST_TOKEN}".encode("ascii")),
            (b"authorization", f"Bearer {TEST_TOKEN}".encode("ascii")),
        ],
    ],
)
def test_unauthorized_mcp_requests_fail_before_downstream(
    headers: list[tuple[bytes, bytes]],
) -> None:
    calls, messages = _exercise("/mcp", headers=headers)

    assert calls == 0
    assert messages[0]["status"] == 401
    response_headers = dict(messages[0]["headers"])
    assert response_headers[b"cache-control"] == b"no-store"
    assert response_headers[b"www-authenticate"].startswith(b"Bearer ")
    serialized = repr(messages)
    assert TEST_TOKEN not in serialized
    assert "wrong-token" not in serialized


@pytest.mark.parametrize("path", ["/mcp", "/sse", "/messages/abc"])
def test_correct_token_reaches_each_mcp_transport(path: str) -> None:
    calls, messages = _exercise(
        path,
        headers=[
            (b"authorization", f"bEaReR {TEST_TOKEN}".encode("ascii"))
        ],
    )

    assert calls == 1
    assert messages[0]["status"] == 204


def test_api_auth_uses_the_same_strict_header_contract() -> None:
    bad_calls, bad_messages = _exercise(
        "/api/buckets",
        headers=[(b"authorization", b"Bearer " + b"\xe9" * 40)],
        middleware_class=APIBearerAuthMiddleware,
    )
    good_calls, good_messages = _exercise(
        "/api/buckets",
        headers=[
            (b"authorization", f"Bearer {TEST_TOKEN}".encode("ascii"))
        ],
        middleware_class=APIBearerAuthMiddleware,
    )

    assert bad_calls == 0
    assert bad_messages[0]["status"] == 401
    assert good_calls == 1
    assert good_messages[0]["status"] == 204


def test_authorized_sse_is_not_buffered_by_either_auth_layer() -> None:
    async def run() -> None:
        first_chunk_sent = asyncio.Event()
        release_stream = asyncio.Event()
        messages: list[dict[str, object]] = []

        async def downstream(scope, receive, send) -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"data: first\n\n",
                    "more_body": True,
                }
            )
            first_chunk_sent.set()
            await release_stream.wait()
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        app = MCPBearerAuthMiddleware(
            APIBearerAuthMiddleware(downstream, token=TEST_TOKEN),
            token=TEST_TOKEN,
        )
        task = asyncio.create_task(
            app(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/sse",
                    "headers": [
                        (
                            b"authorization",
                            f"Bearer {TEST_TOKEN}".encode("ascii"),
                        )
                    ],
                },
                receive,
                send,
            )
        )
        await asyncio.wait_for(first_chunk_sent.wait(), timeout=0.5)
        assert messages[0]["status"] == 200
        assert messages[1]["body"] == b"data: first\n\n"
        assert messages[1]["more_body"] is True
        assert not task.done()
        release_stream.set()
        await asyncio.wait_for(task, timeout=0.5)

    asyncio.run(run())


def test_options_and_non_mcp_paths_remain_available_without_token() -> None:
    options_calls, options_messages = _exercise("/mcp", method="OPTIONS")
    health_calls, health_messages = _exercise("/health", method="GET")

    assert options_calls == 1
    assert options_messages[0]["status"] == 204
    assert health_calls == 1
    assert health_messages[0]["status"] == 204


def test_network_token_is_required_and_never_defaulted() -> None:
    with pytest.raises(RuntimeError, match="OMBRE_MCP_TOKEN is required"):
        require_mcp_token({})

    with pytest.raises(RuntimeError, match="at least 32 characters"):
        require_mcp_token({"OMBRE_MCP_TOKEN": "too-short"})
    with pytest.raises(RuntimeError, match="documented ASCII token format"):
        require_mcp_token({"OMBRE_MCP_TOKEN": "密" * 32})
    with pytest.raises(RuntimeError, match="at least 32 characters"):
        require_api_token({"OMBRE_API_TOKEN": "base64-token-with-padding========"})

    token = "exact-token-0123456789abcdefghijkl"
    assert require_mcp_token({"OMBRE_MCP_TOKEN": token}) == token


def test_server_wires_fail_closed_mcp_auth() -> None:
    source = (REPO_ROOT / "server.py").read_text(encoding="utf-8")

    assert "MCPBearerAuthMiddleware" in source
    assert "_OMBRE_MCP_TOKEN = require_mcp_token()" in source
    assert "_OMBRE_API_TOKEN = require_api_token()" in source
    assert "from starlette.middleware.base import BaseHTTPMiddleware" not in source
    assert "MCP 传输路径（/mcp、/sse、/messages）和 /health 放行" not in source


@pytest.mark.parametrize(
    "filename",
    ["docker-compose.yml", "docker-compose.user.yml"],
)
def test_compose_requires_tokens_and_binds_loopback(filename: str) -> None:
    compose = (REPO_ROOT / filename).read_text(encoding="utf-8")

    assert "${OMBRE_BIND_ADDRESS:-127.0.0.1}" in compose
    assert "OMBRE_API_TOKEN=${OMBRE_API_TOKEN:?required}" in compose
    assert "OMBRE_MCP_TOKEN=${OMBRE_MCP_TOKEN:?required}" in compose
    assert '- "8000:8000"' not in compose


def test_user_compose_builds_the_authenticated_source() -> None:
    compose = (REPO_ROOT / "docker-compose.user.yml").read_text(encoding="utf-8")

    assert "build: ." in compose
    assert "p0luz/ombre-brain:latest" not in compose


def test_render_blueprint_requires_contract_compatible_network_tokens() -> None:
    blueprint = (REPO_ROOT / "render.yaml").read_text(encoding="utf-8")

    assert "key: OMBRE_API_TOKEN" in blueprint
    assert "key: OMBRE_MCP_TOKEN" in blueprint
    assert blueprint.count("sync: false          # Set to: openssl rand -hex 32") == 2
    assert "generateValue: true" not in blueprint


def test_docker_build_context_excludes_secrets_and_runtime_data() -> None:
    patterns = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".env" in patterns
    assert ".env.*" in patterns
    assert "config.yaml" in patterns
    assert "buckets/" in patterns
    assert "data/" in patterns
    assert "*.db" in patterns
    assert "*.sqlite" in patterns
    assert "*.key" in patterns
    assert "*.pem" in patterns


def test_container_pins_and_smokes_the_fastmcp_contract() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "mcp==1.28.1" in requirements.splitlines()
    assert 'RUN python -c "from mcp.server.fastmcp import FastMCP"' in dockerfile
