import types

import pytest

from dehydrator import Dehydrator, _briefing_relative_time_violations


def _response(text: str):
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


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
async def test_briefing_fails_closed_after_two_bad_outputs(briefing_dehy):
    _install_responses(briefing_dehy, "最近又疼了。", "刚刚又疼了。")

    with pytest.raises(RuntimeError, match="相对时间词"):
        await briefing_dehy._api_briefing("📅 发生于 2026-05-30", 300)
