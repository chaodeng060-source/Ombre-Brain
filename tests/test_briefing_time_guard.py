import asyncio
import logging
import types

import pytest

import dehydrator as dehydrator_module
from dehydrator import Dehydrator, _briefing_relative_time_violations


def _response(text: str):
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def _empty_response(*, content=None, finish_reason=None):
    choices = []
    if content is not None:
        message = types.SimpleNamespace(content=content)
        choices = [types.SimpleNamespace(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )]
    return types.SimpleNamespace(
        id="safe-response-id",
        model="deepseek-v4-flash",
        usage=types.SimpleNamespace(prompt_tokens=123, completion_tokens=456),
        choices=choices,
    )


def _install_responses(dehy, *texts):
    calls = []

    async def _create(**kwargs):
        calls.append(kwargs)
        index = min(len(calls) - 1, len(texts) - 1)
        return _response(texts[index])

    dehy.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create),
        )
    )
    return calls


@pytest.fixture
def briefing_dehy(test_config):
    cfg = dict(test_config)
    cfg["dehydration"] = dict(test_config["dehydration"], api_key="test-key")
    return Dehydrator(cfg)


def test_relative_time_violation_detector():
    assert _briefing_relative_time_violations("前两天那件事最近又被提起") == ["前两天", "最近"]
    assert _briefing_relative_time_violations("2026-05-30 的事，上一窗已经说清") == []
    assert _briefing_relative_time_violations("朝灯原话：『最近我很累』") == []


@pytest.mark.asyncio
async def test_briefing_retries_when_relative_time_survives(briefing_dehy):
    calls = _install_responses(
        briefing_dehy,
        "前两天卡兜受伤了。",
        "2026-05-30 卡兜受伤，后来已经痊愈。",
    )

    result = await briefing_dehy._api_briefing("📅 发生于 2026-05-30", 300)

    assert result.startswith("2026-05-30")
    assert len(calls) == 2
    assert "上次输出因含弱相对时间词" in calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_briefing_mechanically_anchors_after_two_bad_outputs(briefing_dehy):
    _install_responses(briefing_dehy, "最近又疼了。", "刚刚又疼了。")

    result = await briefing_dehy._api_briefing("📅 发生于 2026-05-30", 300)

    assert result.endswith("稍早又疼了。")
    assert _briefing_relative_time_violations(result) == []


@pytest.mark.asyncio
async def test_briefing_empty_choices_logs_and_returns_source_fallback(
    briefing_dehy, caplog,
):
    async def _create(**kwargs):
        return _empty_response()

    briefing_dehy.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create),
        )
    )
    with caplog.at_level(logging.ERROR, logger="ombre_brain.dehydrator"):
        result = await briefing_dehy._api_briefing(
            "=== 当前时点 ===\n现在 2026-08-11\n\n=== 上一窗口 ===\n真实素材",
            300,
        )

    assert result
    assert "真实素材" in result
    assert "no choices" in caplog.text
    assert "safe-response-id" in caplog.text


@pytest.mark.asyncio
async def test_briefing_empty_content_logs_finish_reason_and_falls_back(
    briefing_dehy, caplog,
):
    async def _create(**kwargs):
        return _empty_response(content="", finish_reason="length")

    briefing_dehy.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create),
        )
    )
    with caplog.at_level(logging.ERROR, logger="ombre_brain.dehydrator"):
        result = await briefing_dehy._api_briefing("=== 上一窗口 ===\n素材", 300)

    assert result
    assert "素材" in result
    assert '"finish_reason": "length"' in caplog.text


@pytest.mark.asyncio
async def test_briefing_total_timeout_covers_all_attempts_and_falls_back(
    briefing_dehy, monkeypatch, caplog,
):
    started = asyncio.Event()

    async def _create(**kwargs):
        started.set()
        await asyncio.Event().wait()

    briefing_dehy.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create),
        )
    )
    monkeypatch.setattr(
        dehydrator_module,
        "BRIEFING_TOTAL_TIMEOUT_SECONDS",
        0.01,
    )
    with caplog.at_level(logging.ERROR, logger="ombre_brain.dehydrator"):
        result = await briefing_dehy._api_briefing("=== 上一窗口 ===\n素材", 300)

    assert started.is_set()
    assert result
    assert "素材" in result
    assert "exceeded total timeout" in caplog.text
