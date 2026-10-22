"""C-001: reject added audit notes at derivative and cache return boundaries."""

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from dehydrator import RECALL_DEHYDRATION_CACHE_SCHEMA_V1
from tests.test_memory_editorial_notes import NOTES, SOURCE
from tests.test_recall_summary_reuse import make


def _entry(content, name='散步计划'):
    return {'name': name, 'content': content, 'entities': []}


@pytest.mark.asyncio
@pytest.mark.parametrize('fail_open', [False, True])
@pytest.mark.parametrize('field', ['name', 'content'])
async def test_digest_rejects_whole_batch_before_any_item_mapping(test_config, fail_open, field):
    instance, provider = make(test_config)
    entries = [_entry(SOURCE), _entry(SOURCE)]
    entries[1][field] = NOTES[1]
    provider.result = json.dumps({'entries': entries}, ensure_ascii=False)
    instance.ensure_self_contained = AsyncMock(side_effect=AssertionError('item mapping called'))
    before = deepcopy(entries)

    with pytest.raises(RuntimeError, match='digest.unsupported_editorial_note'):
        await instance._api_digest(SOURCE, fail_open=fail_open)

    instance.ensure_self_contained.assert_not_awaited()
    assert entries == before
    assert len(provider.requests) == 1
    assert provider.requests[0]['messages'][1]['content'] == SOURCE


@pytest.mark.asyncio
async def test_public_digest_does_not_continue_to_item_mapping_after_bad_batch(test_config):
    instance, provider = make(test_config)
    provider.result = json.dumps({'entries': [_entry(SOURCE), _entry(NOTES[2])]}, ensure_ascii=False)
    instance.ensure_self_contained = AsyncMock(return_value=SOURCE)

    with pytest.raises(RuntimeError, match='digest.unsupported_editorial_note'):
        await instance.digest(SOURCE, fail_open=True)

    # The one pre-existing mapping of the original source is still allowed.
    assert instance.ensure_self_contained.await_count == 1
    assert instance.ensure_self_contained.call_args.args == (SOURCE,)
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_merge_rejects_generated_note_after_one_provider_call(test_config):
    instance, provider = make(test_config)
    originals = (SOURCE, '朝灯说：“明天有空。”')
    provider.result = SOURCE + '；' + NOTES[1] + '。'

    with pytest.raises(RuntimeError, match='merge.unsupported_editorial_note'):
        await instance.merge(*originals)

    assert originals == (SOURCE, '朝灯说：“明天有空。”')
    assert len(provider.requests) == 1
    assert provider.requests[0]['messages'][1]['content'] == f'旧记忆：\n{originals[0]}\n\n新内容：\n{originals[1]}'


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['digest', 'merge'])
@pytest.mark.parametrize('actual_audit', [False, True])
async def test_new_runtime_checks_preserve_audit_speech_without_widening_fact_gates(
    test_config, operation, actual_audit,
):
    instance, provider = make(test_config)
    original = f'朝灯复核时说：“{NOTES[1]}”。' if actual_audit else SOURCE
    # This followup must not add animal/name/number rejection to these paths.
    derivative = original if actual_audit else '朝灯说棕色猫叫阿花，十岁。'
    if operation == 'digest':
        provider.result = json.dumps({'entries': [_entry(derivative)]}, ensure_ascii=False)
        instance.ensure_self_contained = AsyncMock(return_value=derivative)
        result = await instance._api_digest(original)
        assert result[0]['content'] == derivative
        assert instance.ensure_self_contained.await_count == 1
    else:
        provider.result = derivative
        assert await instance.merge(original, '哥哥记录了朝灯的话。') == derivative
    assert len(provider.requests) == 1


def _seed_cache(instance, source, summary, cache_kind):
    if cache_kind == 'legacy':
        instance._set_cached_summary(source, summary)
    elif cache_kind == 'v1':
        with sqlite3.connect(instance.recall_cache_db_path) as connection:
            connection.execute(
                'INSERT INTO recall_dehydration_cache '
                '(cache_key, content_hash, summary, model, cache_schema) VALUES (?, ?, ?, ?, ?)',
                (instance._recall_cache_key_v1(source), hashlib.sha256(source.encode()).hexdigest(),
                 summary, instance.model, RECALL_DEHYDRATION_CACHE_SCHEMA_V1),
            )
    elif cache_kind == 'v2':
        instance._set_recall_cached_summary(source, summary)
    elif cache_kind == 'legacy_memory':
        instance._set_read_only_memory_summary(source, summary)
    else:
        snapshot = instance._recall_summary_request(source, source)
        if cache_kind == 'exact_memory':
            instance._set_read_only_memory_summary(source, summary, cache_key=snapshot.key)
        else:
            instance._set_recall_cached_summary(source, summary, snapshot=snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize('cache_kind', [
    'legacy', 'v1', 'v2', 'legacy_memory', 'exact_memory', 'exact_persistent',
])
@pytest.mark.parametrize('actual_audit', [False, True])
async def test_cached_audit_note_is_sourced_or_replaced_without_provider_calls(
    test_config, cache_kind, actual_audit,
):
    instance, provider = make(test_config)
    source = f'朝灯复核时说：“{NOTES[1]}”。' if actual_audit else SOURCE
    source += '\n朝灯和哥哥一起讨论晚上的散步安排。' * 60
    summary = json.dumps({'core_facts': [source[:30]], 'summary': NOTES[1]}, ensure_ascii=False)
    _seed_cache(instance, source, summary, cache_kind)
    database_paths = [Path(instance.cache_db_path), Path(instance.recall_cache_db_path)]
    before = [path.read_bytes() for path in database_paths]

    if cache_kind.startswith('exact_'):
        result, origin = await instance.dehydrate_recall_with_source(source)
        assert origin == ('memory_hit' if cache_kind == 'exact_memory' else 'persistent_hit')
    else:
        result, origin = await instance.dehydrate_with_source(source, write_cache=cache_kind == 'legacy')
        assert origin == 'cached'

    expected = summary if actual_audit else json.dumps({
        'core_facts': [source[:400]], 'summary': source[:120],
    }, ensure_ascii=False)
    assert json.loads(result) == json.loads(expected)
    assert provider.requests == []
    # Only the pre-existing v1 migration writes another persistent cache row.
    if cache_kind != 'v1':
        assert [path.read_bytes() for path in database_paths] == before


@pytest.mark.asyncio
async def test_cache_with_non_dialogue_json_source_uses_literal_input_without_call(test_config):
    instance, provider = make(test_config)
    source = json.dumps({'core_facts': [SOURCE * 60]}, ensure_ascii=False)
    instance._set_cached_summary(source, json.dumps({'summary': NOTES[1]}, ensure_ascii=False))

    output, origin = await instance.dehydrate_with_source(source)

    assert json.loads(output) == {'core_facts': [source[:400]], 'summary': source[:120]}
    assert origin == 'cached'
    assert provider.requests == []
