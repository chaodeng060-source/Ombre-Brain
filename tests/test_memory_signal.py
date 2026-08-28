import json
from contextlib import nullcontext

import pytest

import server
from memory_signal import (
    MemorySignalCursorError,
    MemorySignalStore,
    build_signal_entry,
)


def _bucket(bucket_id: str, content: str, **metadata):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "domain": ["工程"],
            "type": "dynamic",
            "event_at": "2026-08-28T07:00:00+00:00",
            **metadata,
        },
    }


def test_signal_entry_is_bounded_and_contains_required_evidence():
    entry = build_signal_entry(
        _bucket(
            "abc123abc123",
            "朝灯已经把海马体切换到 NAS。",
            source="dm:claude",
            validity_state="current",
        ),
        reason="semantic",
    )

    assert len(entry.line) <= 120
    assert "[id:abc123abc123]" in entry.line
    assert "[src:dm:claude]" in entry.line
    assert "[time:current@2026-08-28]" in entry.line
    assert "[why:semantic]" in entry.line
    assert "「朝灯已经把海马体切换到 NAS。" in entry.line
    assert entry.partial is False


def test_signal_entry_marks_long_exact_snippet_partial():
    content = "这是一条必须保留原词序的完整证据。" * 30
    entry = build_signal_entry(
        _bucket("def456def456", content),
        reason="relation",
    )

    assert len(entry.line) <= 120
    assert entry.partial is True
    assert entry.line.endswith("[partial]")
    assert "「这是一条必须保留原词序的完整证据。" in entry.line


def test_signal_entry_uses_honest_unknown_source_without_provenance():
    entry = build_signal_entry(
        _bucket("feedfacefeed", "没有来源字段的原文。"),
        reason="semantic",
    )

    assert "[src:unknown]" in entry.line
    assert "[src:工程]" not in entry.line
    assert "[src:dynamic]" not in entry.line


def test_signal_entry_selects_query_relevant_verbatim_sentence():
    entry = build_signal_entry(
        _bucket(
            "beadbeadbead",
            "这是与查询无关的开场。真正命中的证据是海马体已经迁移完成。",
        ),
        reason="lexical",
        query="海马体",
    )

    assert "真正命中的证据是海马体已经迁移完成。" in entry.line
    assert "这是与查询无关的开场。" not in entry.line


def test_snapshot_cursor_is_fixed_and_does_not_rerank_mutated_inputs():
    store = MemorySignalStore(ttl_seconds=60, max_snapshots=4)
    first = build_signal_entry(_bucket("111111111111", "第一页原文。"), reason="lexical")
    second = build_signal_entry(_bucket("222222222222", "第二页原文。"), reason="semantic")
    candidates = [first, second]

    page_one = store.create(candidates, page_size=1)
    candidates.reverse()
    candidates[0] = build_signal_entry(
        _bucket("222222222222", "后来改写的正文。"),
        reason="random",
    )
    page_two = store.page(page_one["next_cursor"])

    assert page_one["entries"] == [first.line]
    assert page_one["has_more"] is True
    assert page_one["partial"] is True
    assert page_two["entries"] == [second.line]
    assert "第二页原文" in page_two["entries"][0]
    assert "后来改写" not in page_two["entries"][0]
    assert page_two["snapshot_id"] == page_one["snapshot_id"]
    assert page_two["has_more"] is False


def test_only_inspected_snapshot_member_enters_read_receipt():
    store = MemorySignalStore(ttl_seconds=60, max_snapshots=4)
    entries = [
        build_signal_entry(_bucket("111111111111", "第一条。"), reason="lexical"),
        build_signal_entry(_bucket("222222222222", "第二条。"), reason="semantic"),
    ]
    page = store.create(entries, page_size=2)

    assert store.expanded_ids(page["snapshot_id"]) == ()
    assert store.mark_expanded(page["snapshot_id"], "111111111111") is True
    assert store.expanded_ids(page["snapshot_id"]) == ("111111111111",)
    assert store.mark_expanded(page["snapshot_id"], "not-in-snapshot") is False
    assert store.expanded_ids(page["snapshot_id"]) == ("111111111111",)


def test_expired_cursor_fails_explicitly():
    now = [100.0]
    store = MemorySignalStore(
        ttl_seconds=10,
        max_snapshots=4,
        clock=lambda: now[0],
    )
    page = store.create(
        [
            build_signal_entry(_bucket("111111111111", "第一页。"), reason="lexical"),
            build_signal_entry(_bucket("222222222222", "第二页。"), reason="semantic"),
        ],
        page_size=1,
    )
    now[0] = 111.0

    with pytest.raises(MemorySignalCursorError, match="snapshot_expired_or_unknown"):
        store.page(page["next_cursor"])
    assert store.mark_expanded(page["snapshot_id"], "not-in-snapshot") is False
    assert store.expanded_ids(page["snapshot_id"]) == ()


def test_signal_page_is_less_than_half_of_full_bodies():
    store = MemorySignalStore(ttl_seconds=60, max_snapshots=4)
    buckets = [
        _bucket(f"{index:012x}", f"第 {index} 条完整正文。" + "原始细节" * 300)
        for index in range(5)
    ]
    page = store.create(
        [build_signal_entry(bucket, reason="semantic") for bucket in buckets],
        page_size=5,
    )
    thin = json.dumps(page, ensure_ascii=False, separators=(",", ":"))
    full = "\n---\n".join(bucket["content"] for bucket in buckets)

    assert len(thin) <= len(full) * 0.5


@pytest.mark.asyncio
async def test_breath_default_mode_is_byte_stable_passthrough(monkeypatch):
    calls = []

    async def _full(**kwargs):
        calls.append(kwargs)
        return "旧 breath 返回，逐字不变。"

    monkeypatch.setattr(server, "breath", _full)

    result = await server._breath_tool(query="旧查询", include_images=False)

    assert result == "旧 breath 返回，逐字不变。"
    assert len(calls) == 1
    assert calls[0]["query"] == "旧查询"
    assert calls[0]["include_images"] is False
    assert "output_mode" not in calls[0]
    assert "cursor" not in calls[0]


def test_mcp_breath_schema_exposes_signal_without_replacing_core_function():
    tool = server.mcp._tool_manager.get_tool("breath")

    assert tool.fn is server._breath_tool
    assert "output_mode" in tool.parameters["properties"]
    assert "cursor" in tool.parameters["properties"]
    assert "page_size" in tool.parameters["properties"]
    assert server.breath is not server._breath_tool


@pytest.mark.asyncio
async def test_breath_signal_cursor_reuses_snapshot_without_second_search(monkeypatch):
    store = MemorySignalStore(ttl_seconds=60, max_snapshots=4)
    buckets = {
        "111111111111": _bucket("111111111111", "第一页固定原文。"),
        "222222222222": _bucket("222222222222", "第二页固定原文。"),
    }
    full_calls = []

    async def _full(**kwargs):
        full_calls.append(kwargs)
        server._breath_candidate_capture.get().extend([
            {
                "id": "111111111111",
                "bucket": buckets["111111111111"],
                "summary": "脱水一",
                "reason": "lexical",
            },
            {
                "id": "222222222222",
                "bucket": buckets["222222222222"],
                "summary": "脱水二",
                "reason": "semantic",
            },
        ])
        return "很长的旧式完整返回"

    class _Manager:
        async def get(self, bucket_id):
            raise AssertionError(f"signal snapshot must not reread {bucket_id}")

    monkeypatch.setattr(server, "_memory_signal_store", store)
    monkeypatch.setattr(server, "breath", _full)
    monkeypatch.setattr(server, "bucket_mgr", _Manager())

    first_raw = await server._breath_tool(
        query="海马体迁移",
        output_mode="signal",
        page_size=1,
        max_results=2,
    )
    first = json.loads(first_raw)
    buckets["222222222222"]["content"] = "搜索后才变动的正文。"
    second_raw = await server._breath_tool(
        output_mode="signal",
        cursor=first["next_cursor"],
    )
    second = json.loads(second_raw)

    assert len(full_calls) == 1
    assert full_calls[0]["include_images"] is False
    assert full_calls[0]["include_body_state"] is False
    assert full_calls[0]["reset_body_state"] is False
    assert "第一页固定原文" in first["entries"][0]
    assert "第二页固定原文" in second["entries"][0]
    assert "搜索后才变动" not in second["entries"][0]


@pytest.mark.asyncio
async def test_breath_signal_freezes_transient_validity_overlay(monkeypatch):
    store = MemorySignalStore(ttl_seconds=60, max_snapshots=4)
    selected = _bucket(
        "abcabcabcabc",
        "海马体旧地址只用于历史审计。",
        validity_state="historical",
    )

    async def _full(**_kwargs):
        server._capture_breath_candidate(
            selected,
            "旧地址历史审计",
            reason="lexical",
            limit=1,
        )
        selected["metadata"].pop("validity_state")
        return "旧式完整返回"

    class _Manager:
        async def get(self, bucket_id):
            raise AssertionError(f"signal snapshot must not reread {bucket_id}")

    monkeypatch.setattr(server, "_memory_signal_store", store)
    monkeypatch.setattr(server, "breath", _full)
    monkeypatch.setattr(server, "bucket_mgr", _Manager())

    result = json.loads(await server._breath_tool(
        query="海马体旧地址",
        output_mode="signal",
        max_results=1,
    ))

    assert "[time:historical@2026-08-28]" in result["entries"][0]


@pytest.mark.asyncio
async def test_signal_does_not_mark_session_seen_until_valid_inspect(monkeypatch):
    store = MemorySignalStore(ttl_seconds=60, max_snapshots=4)
    first_bucket = _bucket("111111111111", "第一条完整正文。")
    second_bucket = _bucket("222222222222", "第二条完整正文。")

    class _History:
        def __init__(self):
            self.marked = []

        def mark(self, session_id, keys):
            self.marked.append((session_id, tuple(keys)))

    history = _History()

    async def _full(**kwargs):
        server._capture_breath_candidate(
            first_bucket,
            "第一条",
            reason="lexical",
            limit=2,
        )
        server._capture_breath_candidate(
            second_bucket,
            "第二条",
            reason="semantic",
            limit=2,
        )
        server._remember_session_seen_ids(
            kwargs["session_id"],
            [first_bucket["id"], second_bucket["id"]],
        )
        return "旧式完整返回"

    class _Manager:
        async def get(self, bucket_id):
            return {
                first_bucket["id"]: first_bucket,
                second_bucket["id"]: second_bucket,
            }.get(bucket_id)

    monkeypatch.setattr(server, "_memory_signal_store", store)
    monkeypatch.setattr(server, "breath", _full)
    monkeypatch.setattr(server, "bucket_mgr", _Manager())
    monkeypatch.setattr(server, "_session_recall_history", lambda: history)
    monkeypatch.setattr(server, "shared_acceptance_write_guard", nullcontext)
    monkeypatch.setattr(server.decay_engine, "calculate_score", lambda _meta: 1.25)

    page = json.loads(await server._breath_tool(
        query="完整正文",
        output_mode="signal",
        session_id="signal-session",
        max_results=2,
    ))

    assert history.marked == []
    assert store.expanded_ids(page["snapshot_id"]) == ()

    await server.inspect(
        first_bucket["id"],
        signal_snapshot_id=page["snapshot_id"],
    )

    assert store.expanded_ids(page["snapshot_id"]) == (first_bucket["id"],)
    assert len(history.marked) == 1
    assert history.marked[0][0] == "signal-session"
    assert any(first_bucket["id"] in key for key in history.marked[0][1])
    assert all(second_bucket["id"] not in key for key in history.marked[0][1])


@pytest.mark.asyncio
async def test_signal_reports_core_recall_failure_instead_of_complete_empty(monkeypatch):
    async def _failed(**_kwargs):
        return "检索过程出错，请稍后重试。"

    monkeypatch.setattr(server, "breath", _failed)

    result = json.loads(await server._breath_tool(
        query="海马体迁移",
        output_mode="signal",
    ))

    assert result["entries"] == []
    assert result["partial"] is False
    assert result["error"] == "recall_failed"


@pytest.mark.asyncio
async def test_signal_marks_candidate_render_loss_partial(monkeypatch):
    store = MemorySignalStore(ttl_seconds=60, max_snapshots=4)
    good = _bucket("111111111111", "可用原文。")
    bad = _bucket("x" * 200, "无法装进信号行的原文。")

    async def _full(**_kwargs):
        server._breath_candidate_capture.get().extend([
            {"id": good["id"], "bucket": good, "summary": "可用", "reason": "lexical"},
            {"id": bad["id"], "bucket": bad, "summary": "过长", "reason": "semantic"},
        ])
        return "旧式完整返回"

    monkeypatch.setattr(server, "_memory_signal_store", store)
    monkeypatch.setattr(server, "breath", _full)

    result = json.loads(await server._breath_tool(
        query="原文",
        output_mode="signal",
        max_results=2,
    ))

    assert len(result["entries"]) == 1
    assert result["partial"] is True
    assert result["error"] == "signal_entry_build_failed"
    assert result["skipped_count"] == 1


@pytest.mark.asyncio
async def test_inspect_expands_full_body_and_only_then_marks_read(monkeypatch):
    store = MemorySignalStore(ttl_seconds=60, max_snapshots=4)
    first_bucket = _bucket("111111111111", "第一条完整正文，不能被截断。")
    second_bucket = _bucket("222222222222", "第二条完整正文。")
    page = store.create(
        [
            build_signal_entry(first_bucket, reason="semantic"),
            build_signal_entry(second_bucket, reason="lexical"),
        ],
        page_size=2,
    )

    class _Manager:
        async def get(self, bucket_id):
            return {
                "111111111111": first_bucket,
                "222222222222": second_bucket,
            }.get(bucket_id)

    monkeypatch.setattr(server, "_memory_signal_store", store)
    monkeypatch.setattr(server, "bucket_mgr", _Manager())
    monkeypatch.setattr(server.decay_engine, "calculate_score", lambda _meta: 1.25)

    result = await server.inspect(
        "111111111111",
        signal_snapshot_id=page["snapshot_id"],
    )

    assert "第一条完整正文，不能被截断。" in result
    assert "memory_signal_read" in result
    assert "partial:false" in result
    assert store.expanded_ids(page["snapshot_id"]) == ("111111111111",)


@pytest.mark.asyncio
async def test_signal_mode_requires_query_unless_cursor(monkeypatch):
    async def _forbidden(**_kwargs):
        raise AssertionError("empty signal request must not run automatic surfacing")

    monkeypatch.setattr(server, "breath", _forbidden)

    result = json.loads(await server._breath_tool(output_mode="signal"))

    assert result["mode"] == "signal"
    assert result["error"] == "query_required"
    assert result["entries"] == []
