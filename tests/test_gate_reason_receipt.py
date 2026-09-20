"""Zero-provider checks of the real score/cache/HTTP path, not another gate."""
import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import runpy
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import gate_reason_receipt as receipt
from recall_timing import begin_recall_timing, finish_recall_timing, reset_recall_timing

ROOT = Path(__file__).resolve().parents[1]
BASE = "5bbff3ef367042af9eafc8ec3c041a23674d6097"


class Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.payload))])


def load(client, source=None, split=0):
    tree = ast.parse(source or (ROOT / "server.py").read_text())
    names = {"_parse_ds_score_verdicts", "_ds_score_one_batch", "_ds_score_mode_select", "_ds_semantic_select", "api_breath"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for n in nodes:
        n.decorator_list = []
    ns = dict(json=json, asyncio=asyncio, hashlib=hashlib, time=time, re=re,
              gate_reason_receipt=receipt, logger=logging.getLogger("reason-test"),
              _DS_SELECT_CACHE={}, DS_FILTER_MAX_TOKENS=2000,
              _DS_GATE_SCORE_PROMPT="score with reason", DSFilterInvalidPayloadError=ValueError,
              _ds_json_payloads=lambda raw: [json.loads(raw)],
              redact_embedding_input=lambda text: text,
              _ds_select_cache_config=lambda: (60, 100),
              _ds_bucket_stored_date=lambda b: "",
              _safe_chat_completion_diagnostics=lambda r: {},
              _ds_gate_score_threshold=lambda: 70,
              _ds_gate_score_bonus=lambda b, keep: b.get("bonus", 0),
              _dedupe_recall_topics=lambda rows, **kw: rows,
              _ds_filter_provider=lambda: ("shared", "mock", client, {}),
              _ds_full_body_windows_enabled=lambda: False,
              _ds_gate_score_mode_enabled=lambda: True,
              _ds_gate_score_split=lambda: split,
              _ds_today_local_date=lambda: "2026-09-20",
              _ds_gate_cite_mode_enabled=lambda: False)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "server.py", "exec"), ns)
    return ns


def buckets(n=2):
    return [{"id": f"bucket{i}", "metadata": {"name": f"candidate{i}"}, "content": f"body{i}"} for i in range(n)]


def raw(reasons=("完全不同的事件", "同词不同含义"), scores=(10, 20)):
    return json.dumps({"scores": [{"candidate": i, "relevance": score, "reason": reason} for i, (score, reason) in enumerate(zip(scores, reasons))]}, ensure_ascii=False)


async def run_gate(ns, rows=None):
    state, token = receipt.begin()
    try:
        chosen = await ns["_ds_semantic_select"]("同一问句", rows or buckets(), set(), 3)
        return chosen, dict(state)
    finally:
        receipt.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("split", [0, 1])
async def test_all_reject_original_reasons_survive_cache_and_batches(split):
    client = Client(raw() if split == 0 else raw(("本批理由",), (10,)))
    ns = load(client, split=split)
    first, state = await run_gate(ns)
    second, cached = await run_gate(ns)
    assert first == second == []
    expected = ["完全不同的事件", "同词不同含义"] if not split else ["本批理由"] * 2
    assert [r["reason"] for r in state["receipt"]["items"]] == expected
    assert state == cached
    assert len(client.calls) == (2 if split else 1)
    assert [r["id"] for r in state["receipt"]["items"]] == ["bucket0", "bucket1"]


@pytest.mark.asyncio
async def test_legacy_cache_never_rescores(monkeypatch):
    client = Client(raw())
    ns = load(client)
    monkeypatch.setenv("OMBRE_DS_GATE_REASONS_ENABLED", "0")
    await run_gate(ns)
    assert all(len(v) == 2 for v in ns["_DS_SELECT_CACHE"].values())
    monkeypatch.delenv("OMBRE_DS_GATE_REASONS_ENABLED")
    selected, state = await run_gate(ns)
    assert selected == [] and len(client.calls) == 1
    assert {r["reason_status"] for r in state["receipt"]["items"]} == {"absent_legacy_cache"}
    assert all(r["reason"] is None for r in state["receipt"]["items"])


@pytest.mark.asyncio
async def test_off_matches_original_requests_cache_decisions(monkeypatch):
    monkeypatch.setenv("OMBRE_DS_GATE_REASONS_ENABLED", "0")
    source = subprocess.check_output(["git", "show", f"{BASE}:server.py"], cwd=ROOT, text=True)
    old_client, new_client = Client(raw(scores=(65, 80))), Client(raw(scores=(65, 80)))
    old, new = load(old_client, source), load(new_client)
    original = await old["_ds_semantic_select"]("同一问句", buckets(), {"bucket0"}, 3)
    changed, state = await run_gate(new)
    assert changed == original
    assert state == {}
    assert old_client.calls == new_client.calls
    assert list(old["_DS_SELECT_CACHE"]) == list(new["_DS_SELECT_CACHE"])
    assert [v[1:] for v in old["_DS_SELECT_CACHE"].values()] == [v[1:] for v in new["_DS_SELECT_CACHE"].values()]


@pytest.mark.asyncio
async def test_missing_reason_is_not_fabricated():
    ns = load(Client('{"scores":[{"candidate":0,"relevance":2},{"candidate":1,"relevance":3,"reason":42}]}'))
    _, state = await run_gate(ns)
    assert [r["reason_status"] for r in state["receipt"]["items"]] == ["absent_in_response", "unparsable"]
    assert all(r["reason"] is None for r in state["receipt"]["items"])


def test_out_of_order_parser_keeps_reason_alignment():
    ns = load(Client(""))
    value = '{"scores":[{"candidate":1,"relevance":20,"reason":"second"},{"candidate":0,"relevance":10,"reason":"first"}]}'
    scores = ns["_parse_ds_score_verdicts"](value, 2)
    assert scores == [10, 20]
    assert [r["reason"] for r in scores.reasons] == ["first", "second"]


@pytest.mark.asyncio
async def test_only_below_final_threshold_is_a_rejection():
    ns = load(Client(raw(scores=(60, 80))))
    rows = buckets()
    rows[0]["bonus"] = 15
    chosen, state = await run_gate(ns, rows)
    assert len(chosen) == 2
    assert state["receipt"]["items"] == []


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_cross_contaminate():
    a, b = load(Client(raw(("a", "a")))), load(Client(raw(("b", "b"))))
    (_, sa), (_, sb) = await asyncio.gather(run_gate(a), run_gate(b))
    assert {r["reason"] for r in sa["receipt"]["items"]} == {"a"}
    assert {r["reason"] for r in sb["receipt"]["items"]} == {"b"}


@pytest.mark.asyncio
async def test_long_escaped_reasons_have_explicit_bounds():
    ns = load(Client(raw(["\x00\\\"中😀" * 1000] * 100, [1] * 100)))
    _, state = await run_gate(ns, buckets(100))
    result = state["receipt"]
    assert len(result["items"]) <= 64
    assert result["omitted"] + len(result["items"]) == 100
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 32768
    assert all(len(r["reason"]) <= 240 and r["reason_truncated"] for r in result["items"])


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["1", "0"])
async def test_api_exports_separate_receipt_and_timing_stays_content_free(monkeypatch, flag, tmp_path):
    monkeypatch.setenv("OMBRE_DS_GATE_REASONS_ENABLED", flag)
    ns = load(Client(raw()))
    import contextvars
    ns.update(begin_recall_timing=begin_recall_timing, finish_recall_timing=finish_recall_timing,
              reset_recall_timing=reset_recall_timing, _normalize_anchor_recall_policy=lambda x: x,
              _e_chord_shadow_response_capture=contextvars.ContextVar("chord", default=None),
              _breath_gate_skip_capture=contextvars.ContextVar("skip", default=None),
              BREATH_DEFAULT_MAX_TOKENS=2000, BREATH_DEFAULT_MAX_RESULTS=10,
              _breath_deadline_sec=lambda: 5, recall_is_partial=lambda: False)
    async def breath(**kwargs):
        await ns["_ds_semantic_select"]("同一问句", buckets(), set(), 3)
        return "未找到相关记忆。"
    async def request_json():
        return {"query": "同一问句"}
    ns["breath"] = breath
    response = await ns["api_breath"](SimpleNamespace(json=request_json))
    payload = json.loads(response.body)
    if flag == "1":
        assert len(payload["gate_rejections"]["items"]) == 2
    else:
        assert "gate_rejections" not in payload
    assert "完全不同的事件" not in json.dumps(payload["timing"], ensure_ascii=False)
    assert receipt.current() is None
    # Optional two-repository contract check. Nothing is installed or deployed.
    twin_root = os.getenv("TWIN_GATE_REASON_SOURCE_ROOT")
    if twin_root and flag == "1":
        consumer = runpy.run_path(str(Path(twin_root) / "app/recall_gate_reasons.py"))
        cleaned = consumer["receipt_for_trace"](payload["gate_rejections"])
        tree = ast.parse((Path(twin_root) / "server.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_recall_trace")
        path = tmp_path / "recall_trace.jsonl"
        namespace = dict(json=json, datetime=datetime, os=os, HEARTBEAT_TZ=ZoneInfo("Asia/Shanghai"),
                         _RECALL_TRACE_PATH=path, _RECALL_TRACE_MAX_BYTES=1000000,
                         recall_visibility=SimpleNamespace(observe_trace=lambda fields: None),
                         recall_gate_reasons=SimpleNamespace(**consumer))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "twin-server.py", "exec"), namespace)
        namespace["_recall_trace"](ds_gate_rejections=cleaned, ds_gate_in=2, ds_gate_out=0)
        record = json.loads(path.read_text())
        assert record["ds_gate_rejections"] == payload["gate_rejections"]
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_invalid_unicode_reason_does_not_change_successful_gate():
    _, state = await run_gate(load(Client(raw(("\ud800", "safe")))))
    assert state["receipt"]["items"][0]["reason_status"] == "unparsable"
    assert state["receipt"]["items"][1]["reason"] == "safe"
