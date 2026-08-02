import hashlib
import logging
import sqlite3

import pytest

from dehydrator import Dehydrator, INFER_RELATIONS_PROMPT


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Completions:
    def __init__(self, content, *, choices=True):
        self.content = content
        self.choices = choices
        self.calls = 0
        self.requests = []

    async def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        response_choices = [_Choice(self.content)] if self.choices else []
        return type("Response", (), {"choices": response_choices})()


def _dehydrator(test_config, content, *, choices=True):
    dehydrator = Dehydrator(test_config)
    dehydrator.api_available = True
    completions = _Completions(content, choices=choices)
    dehydrator.client = type(
        "Client",
        (),
        {
            "chat": type(
                "Chat",
                (),
                {
                    "completions": completions,
                },
            )()
        },
    )()
    return dehydrator, completions


def _long_content():
    return "朝灯问起4.7的第一晚，我要保留具体事件、时间、感受和后续影响。" * 12


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "choices"),
    [
        ("", True),
        (" \n\t ", True),
        ("123456789", True),
        (None, True),
        ("unused", False),
    ],
)
async def test_near_empty_api_result_is_warned_rejected_and_not_cached(
    test_config,
    caplog,
    response,
    choices,
):
    dehydrator, _ = _dehydrator(test_config, response, choices=choices)
    content = _long_content()

    with caplog.at_level(logging.WARNING, logger="ombre_brain.dehydrator"):
        with pytest.raises(RuntimeError, match="空或过短摘要"):
            await dehydrator.dehydrate(content)

    with sqlite3.connect(dehydrator.cache_db_path) as conn:
        count = conn.execute(
            "SELECT count(*) FROM dehydration_cache"
        ).fetchone()[0]
    assert count == 0
    assert "脱水 API 返回空或过短结果" in caplog.text


@pytest.mark.asyncio
async def test_near_empty_cached_summary_is_ignored_and_replaced(
    test_config,
    caplog,
):
    valid = '{"core_facts":["4.7的第一晚发生了具体事件"],"summary":"保留正文摘要"}'
    dehydrator, completions = _dehydrator(test_config, valid)
    content = _long_content()
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    with sqlite3.connect(dehydrator.cache_db_path) as conn:
        conn.execute(
            "INSERT INTO dehydration_cache "
            "(content_hash, summary, model) VALUES (?, ?, ?)",
            (content_hash, "   ", dehydrator.model),
        )
        conn.commit()

    with caplog.at_level(logging.WARNING, logger="ombre_brain.dehydrator"):
        output = await dehydrator.dehydrate(content)

    assert completions.calls == 1
    assert "4.7的第一晚发生了具体事件" in output
    assert "忽略空或过短脱水缓存" in caplog.text
    with sqlite3.connect(dehydrator.cache_db_path) as conn:
        cached = conn.execute(
            "SELECT summary FROM dehydration_cache WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()[0]
    assert cached == valid


def test_cache_writer_defensively_refuses_near_empty_summary(
    test_config,
    caplog,
):
    dehydrator, _ = _dehydrator(test_config, "unused")

    with caplog.at_level(logging.WARNING, logger="ombre_brain.dehydrator"):
        stored = dehydrator._set_cached_summary(_long_content(), "九八七六五四三二一")

    assert stored is False
    assert "拒绝缓存空或过短脱水结果" in caplog.text
    with sqlite3.connect(dehydrator.cache_db_path) as conn:
        count = conn.execute(
            "SELECT count(*) FROM dehydration_cache"
        ).fetchone()[0]
    assert count == 0


@pytest.mark.asyncio
async def test_relation_inference_uses_reasoning_budget(test_config):
    dehydrator, completions = _dehydrator(test_config, "[]")

    result = await dehydrator.infer_relations(
        "朝灯完成了脱水器回归。",
        [{"id": "bucket-1", "name": "脱水器", "summary": "生产修复"}],
    )

    assert result == []
    assert completions.calls == 1
    assert completions.requests[0]["max_tokens"] == 4000


def test_relation_prompt_preserves_production_kin_policy():
    assert "kin（同类）判定要放开" in INFER_RELATIONS_PROMPT
    assert "其余五类（causes/contributes/improves/explains/updates）保持严格" in INFER_RELATIONS_PROMPT
