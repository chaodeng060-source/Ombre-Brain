import json

import pytest

import server


def _bucket(bid: str = "briefing-source") -> dict:
    return {
        "id": bid,
        "content": "朝灯今天确认简报要后台预生成。",
        "metadata": {
            "id": bid,
            "name": "简报后台预生成",
            "importance": 9,
            "type": "dynamic",
            "domain": ["daily"],
            "tags": [],
            "valence": 0.7,
            "arousal": 0.3,
            "pinned": True,
            "protected": False,
            "resolved": False,
            "world": "",
            "created": "2026-08-11T14:00:00+08:00",
            "last_active": "2026-08-11T14:00:00+08:00",
        },
    }


class _FakeDecay:
    is_running = True

    async def ensure_started(self):
        return None

    def calculate_score(self, meta):
        return float(meta.get("importance", 5))


class _FakeMgr:
    def __init__(self):
        self.calls = 0
        self.fail = False

    async def list_all(self, include_archive=False):
        self.calls += 1
        if self.fail:
            raise AssertionError("cache hit must not reread the bucket library")
        return [_bucket()]


class _GoodDehydrator:
    def __init__(self):
        self.calls = []

    async def briefing(self, raw_material, max_chars=1000, **kwargs):
        self.calls.append(kwargs)
        return "这是模型压缩后的真简报。"


class _FallbackDehydrator:
    def __init__(self):
        self.calls = []

    async def briefing(self, raw_material, max_chars=1000, **kwargs):
        self.calls.append(kwargs)
        return "【简报压缩未完成，以下为已脱敏素材摘录】\n原始素材"


@pytest.fixture(autouse=True)
def _clear_prebuilt_cache(monkeypatch):
    with server._briefing_cache_lock:
        server._briefing_prebuilt_cache.clear()
    monkeypatch.setattr(server, "decay_engine", _FakeDecay())
    yield
    with server._briefing_cache_lock:
        server._briefing_prebuilt_cache.clear()


@pytest.mark.asyncio
async def test_second_request_reads_prebuilt_cache_and_refreshes_time_header(
    monkeypatch,
):
    manager = _FakeMgr()
    dehydrator = _GoodDehydrator()
    headers = iter([
        "现在 2026-08-11 周二 14:00",
        "现在 2026-08-11 周二 14:07",
    ])
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", dehydrator)
    monkeypatch.setattr(server, "_now_bj_header", lambda: next(headers))

    first = await server.briefing(
        max_chars=1500,
        format="json",
        session_id="cache-session",
        include_body_state=False,
    )
    manager.fail = True
    second = await server.briefing(
        max_chars=1500,
        format="json",
        session_id="cache-session",
        include_body_state=False,
    )
    assert json.loads(first)["time_header"].endswith("14:00")
    assert json.loads(second)["time_header"].endswith("14:07")
    assert json.loads(second)["briefing"].startswith("这是模型压缩后的真简报")
    assert manager.calls == 1
    assert len(dehydrator.calls) == 1


@pytest.mark.asyncio
async def test_new_session_reuses_same_generation_shape_without_model_call(
    monkeypatch,
):
    manager = _FakeMgr()
    dehydrator = _GoodDehydrator()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", dehydrator)

    first = await server.briefing(
        max_chars=1500,
        format="json",
        session_id="session-a",
        include_body_state=False,
    )
    manager.fail = True
    second = await server.briefing(
        max_chars=1500,
        format="json",
        session_id="session-b",
        include_body_state=False,
    )

    assert json.loads(first)["briefing"] == json.loads(second)["briefing"]
    assert manager.calls == 1
    assert len(dehydrator.calls) == 1


@pytest.mark.asyncio
async def test_background_fallback_keeps_previous_good_cache(monkeypatch):
    manager = _FakeMgr()
    dehydrator = _FallbackDehydrator()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", dehydrator)
    profile = server._briefing_profile(1500, "", False, "stable-session", "json")
    old_payload = json.dumps(
        {
            "time_header": "现在 2026-08-11 周二 14:00",
            "slots": [],
            "briefing": "旧的完整真简报",
            "anchor_index": "",
        },
        ensure_ascii=False,
    )
    server._store_briefing_cache_entry(
        profile,
        text=old_payload,
        time_header="现在 2026-08-11 周二 14:00",
        buckets=[],
    )
    before = server._get_briefing_cache_entry(profile)

    await server._refresh_briefing_profile(profile)

    assert server._get_briefing_cache_entry(profile) is before
    assert dehydrator.calls == [
        {"total_timeout_seconds": server.BRIEFING_REFRESH_TIMEOUT_SECONDS}
    ]


def test_cache_profile_separates_size_format_and_session():
    base = server._briefing_profile(1500, "", False, "session-a", "json")

    assert base != server._briefing_profile(1000, "", False, "session-a", "json")
    assert base != server._briefing_profile(1500, "", False, "session-a", "text")
    assert base != server._briefing_profile(1500, "", False, "session-b", "json")


def test_background_store_fans_out_without_merging_session_keys():
    first = server._briefing_profile(1500, "", False, "session-a", "json")
    second = server._briefing_profile(1500, "", False, "session-b", "json")
    server._register_briefing_profile(first)
    server._register_briefing_profile(second)

    server._store_briefing_cache_entry(
        first,
        text=json.dumps({"time_header": "old", "briefing": "GOOD"}),
        time_header="old",
        buckets=[],
    )

    assert first in server._briefing_prebuilt_cache
    assert second in server._briefing_prebuilt_cache
    assert server._briefing_prebuilt_cache[first] is not server._briefing_prebuilt_cache[second]


def test_new_profile_only_reuses_identical_generation_shape():
    source = server._briefing_profile(1500, "daily", False, "session-a", "json")
    same_shape = server._briefing_profile(1500, "daily", False, "session-b", "json")
    different_size = server._briefing_profile(1000, "daily", False, "session-b", "json")
    different_format = server._briefing_profile(1500, "daily", False, "session-b", "text")

    server._store_briefing_cache_entry(
        source,
        text=json.dumps({"time_header": "old", "briefing": "GOOD"}),
        time_header="old",
        buckets=[],
    )
    server._register_briefing_profile(same_shape)
    server._register_briefing_profile(different_size)
    server._register_briefing_profile(different_format)

    assert server._get_briefing_cache_entry(same_shape) is not None
    assert server._get_briefing_cache_entry(different_size) is None
    assert server._get_briefing_cache_entry(different_format) is None


@pytest.mark.asyncio
async def test_dirty_write_wakes_background_refresh(monkeypatch):
    event = server.asyncio.Event()
    loop = server.asyncio.get_running_loop()
    monkeypatch.setattr(server, "_briefing_refresh_event", event)
    monkeypatch.setattr(server, "_briefing_refresh_loop", loop)

    server._mark_briefing_cache_dirty("test_hold")

    assert event.is_set()


@pytest.mark.asyncio
async def test_startup_schedules_immediate_background_generation(monkeypatch):
    started = server.asyncio.Event()
    seen = []

    async def _record_refresh(profile):
        seen.append(profile)
        started.set()

    monkeypatch.setattr(server, "_refresh_briefing_profile", _record_refresh)
    await server._start_briefing_cache_refresh()
    try:
        await server.asyncio.wait_for(started.wait(), timeout=1)
        assert seen[0][0] == 1500
        assert seen[0][4] == "json"
        assert server._briefing_refresh_task is not None
        assert not server._briefing_refresh_task.done()
    finally:
        await server._stop_briefing_cache_refresh()


@pytest.mark.asyncio
async def test_clean_worker_does_not_refresh_again_while_idle(monkeypatch):
    profile = server._briefing_profile(1500, "", False, "", "json")
    event = server.asyncio.Event()
    event.set()
    first_refresh = server.asyncio.Event()
    seen = []

    async def _record_refresh(current):
        seen.append(current)
        first_refresh.set()

    monkeypatch.setattr(server, "_briefing_refresh_event", event)
    monkeypatch.setattr(server, "_briefing_profiles", {profile})
    monkeypatch.setattr(server, "_refresh_briefing_profile", _record_refresh)
    monkeypatch.setattr(server, "BRIEFING_REFRESH_INTERVAL_SECONDS", 0.03)

    task = server.asyncio.create_task(server._briefing_cache_refresh_worker())
    try:
        await server.asyncio.wait_for(first_refresh.wait(), timeout=1)
        await server.asyncio.sleep(0.08)
        assert seen == [profile]
    finally:
        task.cancel()
        with pytest.raises(server.asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_dirty_marks_coalesce_into_one_rate_limited_refresh(monkeypatch):
    profile = server._briefing_profile(1500, "", False, "", "json")
    event = server.asyncio.Event()
    event.set()
    first_refresh = server.asyncio.Event()
    second_refresh = server.asyncio.Event()
    seen = []

    async def _record_refresh(current):
        seen.append((current, server.asyncio.get_running_loop().time()))
        if len(seen) == 1:
            first_refresh.set()
        elif len(seen) == 2:
            second_refresh.set()

    monkeypatch.setattr(server, "_briefing_refresh_event", event)
    monkeypatch.setattr(server, "_briefing_profiles", {profile})
    monkeypatch.setattr(server, "_refresh_briefing_profile", _record_refresh)
    monkeypatch.setattr(server, "BRIEFING_REFRESH_INTERVAL_SECONDS", 0.05)

    task = server.asyncio.create_task(server._briefing_cache_refresh_worker())
    try:
        await server.asyncio.wait_for(first_refresh.wait(), timeout=1)
        event.set()
        await server.asyncio.sleep(0.01)
        event.set()
        event.set()
        await server.asyncio.wait_for(second_refresh.wait(), timeout=1)
        await server.asyncio.sleep(0.08)

        assert len(seen) == 2
        assert seen[1][1] - seen[0][1] >= 0.04
    finally:
        task.cancel()
        with pytest.raises(server.asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_generated_briefing_disabled_forces_zero_model_boot_pack(monkeypatch):
    manager = _FakeMgr()
    dehydrator = _GoodDehydrator()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", dehydrator)
    monkeypatch.setitem(server.config, "briefing", {"generated_enabled": False})

    result = await server.briefing(
        max_chars=1500,
        format="json",
        session_id="fresh-window",
        include_body_state=False,
    )

    payload = json.loads(result)
    assert payload["mode"] == "deterministic"
    assert payload["model_called"] is False
    assert dehydrator.calls == []


@pytest.mark.asyncio
async def test_generated_briefing_disabled_does_not_start_refresh_worker(monkeypatch):
    await server._stop_briefing_cache_refresh()
    monkeypatch.setitem(server.config, "briefing", {"generated_enabled": False})

    await server._start_briefing_cache_refresh()

    assert server._briefing_refresh_task is None
    assert server._briefing_refresh_event is None
