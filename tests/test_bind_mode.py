"""启动绑定模式：容器默认 0.0.0.0:8000；只有 OMBRE_LOOPBACK_ONLY=1 才走 VPS 单机 loopback。

2026-08-18 NAS 首次部署 1bdba89 失败的根因：p13 loopback 补丁把 uvicorn 绑到
OMBRE_BIND_ADDRESS（compose 把 .env 里给宿主用的 127.0.0.1 传进了容器）→ 容器内 loopback
→ 宿主映射的 8000 无人接 → health 连不上 → 自动回滚。这里把那段逻辑抠成纯函数
在源码级别锁住两种模式，不需要真起 uvicorn。
"""
import os
import re
import textwrap
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "server.py"


def _bind_block() -> str:
    src = SERVER.read_text(encoding="utf-8")
    anchor = src.index("loopback_only = os.environ.get(\"OMBRE_LOOPBACK_ONLY\"")
    start = src.rfind("\n", 0, anchor) + 1          # 从行首切，dedent 才认得出公共缩进
    end = src.index("keepalive_host = ", start)
    end = src.index("\n", end)
    return textwrap.dedent(src[start:end])


def _run(env: dict) -> tuple[str, int, str]:
    block = _bind_block()
    ns = {"os": os}
    saved = {k: os.environ.get(k) for k in ("OMBRE_LOOPBACK_ONLY", "OMBRE_BIND_ADDRESS", "OMBRE_HOST_PORT")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        exec(block, ns)
        return ns["bind_host"], ns["host_port"], ns["keepalive_host"]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_container_default_binds_all_interfaces_even_if_env_leaks_bind_address():
    # NAS 容器真实 env：compose 把 OMBRE_BIND_ADDRESS=127.0.0.1 传进来了，但没设 LOOPBACK_ONLY。
    host, port, ka = _run({"OMBRE_BIND_ADDRESS": "127.0.0.1"})
    assert (host, port, ka) == ("0.0.0.0", 8000, "localhost")


def test_vps_loopback_mode_uses_18080_and_refuses_public_bind():
    host, port, ka = _run({"OMBRE_LOOPBACK_ONLY": "1", "OMBRE_BIND_ADDRESS": "127.0.0.1", "OMBRE_HOST_PORT": "18080"})
    assert (host, port, ka) == ("127.0.0.1", 18080, "127.0.0.1")
    import pytest
    with pytest.raises(RuntimeError):
        _run({"OMBRE_LOOPBACK_ONLY": "1", "OMBRE_BIND_ADDRESS": "0.0.0.0"})


def test_uvicorn_run_uses_computed_bind():
    src = SERVER.read_text(encoding="utf-8")
    assert re.search(r"uvicorn\.run\(_app,\s*host=bind_host,\s*port=host_port\)", src)
    assert 'host="0.0.0.0", port=8000)' not in src.split("uvicorn.run(")[-1][:80]
