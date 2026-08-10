import hashlib
import logging
from pathlib import Path
import sqlite3

import pytest

import dehydrator as dehydrator_module
from dehydrator import (
    Dehydrator,
    INFER_RELATIONS_PROMPT,
    RECALL_DEHYDRATION_CACHE_SCHEMA,
    RECALL_DEHYDRATION_CACHE_SCHEMA_V1,
)


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
async def test_recall_cache_miss_writes_only_disposable_derived_cache(
    test_config,
):
    valid = '{"core_facts":["4.7的第一晚发生了具体事件"],"summary":"保留正文摘要"}'
    dehydrator, completions = _dehydrator(test_config, valid)
    cache_path = Path(dehydrator.cache_db_path)
    before = cache_path.read_bytes()

    output = await dehydrator.dehydrate(_long_content(), write_cache=False)

    assert completions.calls == 1
    assert "4.7的第一晚发生了具体事件" in output
    assert cache_path.read_bytes() == before
    with sqlite3.connect(dehydrator.cache_db_path) as conn:
        count = conn.execute(
            "SELECT count(*) FROM dehydration_cache"
        ).fetchone()[0]
    assert count == 0

    recall_path = Path(dehydrator.recall_cache_db_path)
    with sqlite3.connect(recall_path) as conn:
        rows = conn.execute(
            "SELECT content_hash, summary, model, cache_schema "
            "FROM recall_dehydration_cache"
        ).fetchall()
    assert rows == [(
        hashlib.sha256(_long_content().encode()).hexdigest(),
        valid,
        dehydrator.model,
        RECALL_DEHYDRATION_CACHE_SCHEMA,
    )]
    assert _long_content().encode() not in recall_path.read_bytes()

    restarted, restarted_completions = _dehydrator(
        test_config,
        "must not be called",
    )
    restarted_output = await restarted.dehydrate(
        _long_content(),
        write_cache=False,
    )
    assert restarted_output == output
    assert restarted_completions.calls == 0


@pytest.mark.asyncio
async def test_recall_dehydration_disables_deepseek_thinking_only_on_read_path(
    test_config,
):
    valid = '{"core_facts":["长正文仍返回摘要"],"summary":"关闭隐藏思考"}'
    deepseek_config = {
        **test_config,
        "dehydration": {
            **test_config.get("dehydration", {}),
            "base_url": "https://api.deepseek.com/v1",
        },
    }
    recall, recall_completions = _dehydrator(deepseek_config, valid)

    await recall.dehydrate(_long_content(), write_cache=False)

    assert recall_completions.requests[0]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }

    write_content = _long_content() + "写链保持原调用合同。"
    writer, writer_completions = _dehydrator(deepseek_config, valid)
    await writer.dehydrate(write_content, write_cache=True)
    assert "extra_body" not in writer_completions.requests[0]


@pytest.mark.asyncio
async def test_deployed_v1_recall_summary_migrates_without_api_call(test_config):
    content = _long_content()
    summary = '{"core_facts":["v1成功摘要"],"summary":"迁移到v2"}'
    current, completions = _dehydrator(test_config, "must not be called")
    with sqlite3.connect(current.recall_cache_db_path) as conn:
        conn.execute(
            "INSERT INTO recall_dehydration_cache "
            "(cache_key, content_hash, summary, model, cache_schema) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                current._recall_cache_key_v1(content),
                hashlib.sha256(content.encode()).hexdigest(),
                summary,
                current.model,
                RECALL_DEHYDRATION_CACHE_SCHEMA_V1,
            ),
        )
        conn.commit()

    output = await current.dehydrate(content, write_cache=False)

    assert completions.calls == 0
    assert "v1成功摘要" in output
    with sqlite3.connect(current.recall_cache_db_path) as conn:
        rows = conn.execute(
            "SELECT cache_key, cache_schema FROM recall_dehydration_cache "
            "ORDER BY cache_schema"
        ).fetchall()
    assert rows == [
        (current._recall_cache_key_v1(content), RECALL_DEHYDRATION_CACHE_SCHEMA_V1),
        (current._recall_cache_key(content), RECALL_DEHYDRATION_CACHE_SCHEMA),
    ]


@pytest.mark.asyncio
async def test_read_only_dehydrate_reuses_bounded_process_cache(test_config):
    valid = '{"core_facts":["4.7的第一晚发生了具体事件"],"summary":"保留正文摘要"}'
    dehydrator, completions = _dehydrator(test_config, valid)
    cache_path = Path(dehydrator.cache_db_path)
    before = cache_path.read_bytes()

    first = await dehydrator.dehydrate(_long_content(), write_cache=False)
    second = await dehydrator.dehydrate(_long_content(), write_cache=False)

    assert first == second
    assert completions.calls == 1
    assert len(dehydrator._read_only_summary_cache) == 1
    assert cache_path.read_bytes() == before
    with sqlite3.connect(dehydrator.cache_db_path) as conn:
        count = conn.execute(
            "SELECT count(*) FROM dehydration_cache"
        ).fetchone()[0]
    assert count == 0


@pytest.mark.asyncio
async def test_recall_cache_key_invalidates_on_content_model_and_prompt_change(
    test_config,
    monkeypatch,
):
    valid = '{"core_facts":["4.7的第一晚发生了具体事件"],"summary":"保留正文摘要"}'
    first, first_completions = _dehydrator(test_config, valid)
    await first.dehydrate(_long_content(), write_cache=False)
    assert first_completions.calls == 1

    changed_content, changed_content_completions = _dehydrator(test_config, valid)
    await changed_content.dehydrate(
        _long_content() + "正文已经更新。",
        write_cache=False,
    )
    assert changed_content_completions.calls == 1

    other_config = {
        **test_config,
        "dehydration": {
            **test_config.get("dehydration", {}),
            "model": "another-dehydration-model",
        },
    }
    changed_model, changed_model_completions = _dehydrator(other_config, valid)
    await changed_model.dehydrate(_long_content(), write_cache=False)
    assert changed_model_completions.calls == 1

    prior_key = first._recall_cache_key(_long_content())
    monkeypatch.setattr(
        dehydrator_module,
        "DEHYDRATE_PROMPT",
        dehydrator_module.DEHYDRATE_PROMPT + "\n契约版本变化。",
    )
    assert first._recall_cache_key(_long_content()) != prior_key
    legacy_only_content = _long_content() + "旧库未绑定提示词，不能复用。"
    with sqlite3.connect(first.cache_db_path) as conn:
        conn.execute(
            "INSERT INTO dehydration_cache "
            "(content_hash, summary, model) VALUES (?, ?, ?)",
            (
                hashlib.sha256(legacy_only_content.encode()).hexdigest(),
                valid,
                first.model,
            ),
        )
        conn.commit()
    await first.dehydrate(legacy_only_content, write_cache=False)
    assert first_completions.calls == 2

    with sqlite3.connect(first.recall_cache_db_path) as conn:
        rows = conn.execute(
            "SELECT count(*), count(DISTINCT cache_key), "
            "count(DISTINCT content_hash), count(DISTINCT model) "
            "FROM recall_dehydration_cache"
        ).fetchone()
    assert rows == (4, 4, 3, 2)


@pytest.mark.asyncio
async def test_read_only_dehydrate_cache_hit_keeps_database_byte_identical(
    test_config,
):
    cached_summary = (
        '{"core_facts":["4.7的第一晚发生了具体事件"],'
        '"summary":"命中只读缓存"}'
    )
    dehydrator, completions = _dehydrator(test_config, "must not be called")
    content = _long_content()
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    with sqlite3.connect(dehydrator.cache_db_path) as conn:
        conn.execute(
            "INSERT INTO dehydration_cache "
            "(content_hash, summary, model) VALUES (?, ?, ?)",
            (content_hash, cached_summary, dehydrator.model),
        )
        conn.commit()
    cache_path = Path(dehydrator.cache_db_path)
    before = cache_path.read_bytes()

    output = await dehydrator.dehydrate(content, write_cache=False)

    assert completions.calls == 0
    assert "命中只读缓存" in output
    assert cache_path.read_bytes() == before
    with sqlite3.connect(dehydrator.recall_cache_db_path) as conn:
        migrated = conn.execute(
            "SELECT summary, model, cache_schema "
            "FROM recall_dehydration_cache"
        ).fetchall()
    assert migrated == [(
        cached_summary,
        dehydrator.model,
        RECALL_DEHYDRATION_CACHE_SCHEMA,
    )]


@pytest.mark.asyncio
async def test_legacy_cross_model_summary_is_migrated_only_once(test_config):
    cached_summary = (
        '{"core_facts":["旧模型摘要仍是既有生产结果"],'
        '"summary":"兼容迁移后绑定新合同"}'
    )
    content = _long_content()
    current, current_completions = _dehydrator(
        test_config,
        "must not be called",
    )
    with sqlite3.connect(current.cache_db_path) as conn:
        conn.execute(
            "INSERT INTO dehydration_cache "
            "(content_hash, summary, model) VALUES (?, ?, ?)",
            (
                hashlib.sha256(content.encode()).hexdigest(),
                cached_summary,
                "historical-dehydration-model",
            ),
        )
        conn.commit()

    migrated_output = await current.dehydrate(content, write_cache=False)
    assert current_completions.calls == 0
    assert "旧模型摘要仍是既有生产结果" in migrated_output

    changed_config = {
        **test_config,
        "dehydration": {
            **test_config.get("dehydration", {}),
            "model": "future-dehydration-model",
        },
    }
    future, future_completions = _dehydrator(
        changed_config,
        '{"core_facts":["新模型重新生成"],"summary":"不能二次回落旧库"}',
    )
    future_output = await future.dehydrate(content, write_cache=False)
    assert future_completions.calls == 1
    assert "新模型重新生成" in future_output


@pytest.mark.asyncio
async def test_recall_cache_damage_fails_open_and_can_be_rebuilt(
    test_config,
    caplog,
):
    valid = '{"core_facts":["4.7的第一晚发生了具体事件"],"summary":"保留正文摘要"}'
    dehydrator, completions = _dehydrator(test_config, valid)
    recall_path = Path(dehydrator.recall_cache_db_path)
    recall_path.write_bytes(b"not a sqlite database")

    with caplog.at_level(logging.WARNING, logger="ombre_brain.dehydrator"):
        output = await dehydrator.dehydrate(_long_content(), write_cache=False)

    assert completions.calls == 1
    assert "4.7的第一晚发生了具体事件" in output
    assert "Ignoring unreadable recall dehydration cache" in caplog.text

    # Deleting only the disposable cache is sufficient recovery; the running
    # process recreates the schema on the next successful miss.
    recall_path.unlink()
    replacement_content = _long_content() + "删除派生库后重新生成。"
    replacement_output = await dehydrator.dehydrate(
        replacement_content,
        write_cache=False,
    )
    assert completions.calls == 2
    assert "4.7的第一晚发生了具体事件" in replacement_output
    with sqlite3.connect(dehydrator.recall_cache_db_path) as conn:
        count = conn.execute(
            "SELECT count(*) FROM recall_dehydration_cache"
        ).fetchone()[0]
    assert count == 1

    rebuilt, rebuilt_completions = _dehydrator(test_config, "must not be called")
    rebuilt_output = await rebuilt.dehydrate(
        replacement_content,
        write_cache=False,
    )
    assert rebuilt_output == replacement_output
    assert rebuilt_completions.calls == 0


@pytest.mark.asyncio
async def test_recall_cache_lock_fails_open_without_waiting(test_config, caplog):
    valid = '{"core_facts":["4.7的第一晚发生了具体事件"],"summary":"保留正文摘要"}'
    dehydrator, completions = _dehydrator(test_config, valid)
    lock = sqlite3.connect(dehydrator.recall_cache_db_path, timeout=0)
    lock.execute("BEGIN EXCLUSIVE")
    try:
        with caplog.at_level(logging.WARNING, logger="ombre_brain.dehydrator"):
            output = await dehydrator.dehydrate(
                _long_content(),
                write_cache=False,
            )
    finally:
        lock.rollback()
        lock.close()

    assert completions.calls == 1
    assert "4.7的第一晚发生了具体事件" in output
    assert "database is locked" in caplog.text


@pytest.mark.asyncio
async def test_recall_cache_refuses_symlink_without_touching_target(
    test_config,
    tmp_path,
    caplog,
):
    cache_dir = Path(test_config["buckets_dir"]) / ".recall_cache"
    cache_dir.mkdir(mode=0o700)
    outside = tmp_path / "outside.sqlite3"
    outside.write_bytes(b"do not touch")
    cache_path = cache_dir / "recall_dehydration_cache.db"
    try:
        cache_path.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")

    valid = '{"core_facts":["4.7的第一晚发生了具体事件"],"summary":"保留正文摘要"}'
    with caplog.at_level(logging.WARNING, logger="ombre_brain.dehydrator"):
        dehydrator, completions = _dehydrator(test_config, valid)
        output = await dehydrator.dehydrate(_long_content(), write_cache=False)

    assert completions.calls == 1
    assert "4.7的第一晚发生了具体事件" in output
    assert outside.read_bytes() == b"do not touch"
    assert "unsafe recall cache database" in caplog.text


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
