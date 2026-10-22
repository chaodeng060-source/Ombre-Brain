"""Generated source audits stay out of recall; original audit speech survives."""

import json

import pytest

from lmc5_proposer import ProposerChunk, ProposerContractError, StrictOmbreProposer
from memory_source_fidelity import validate_derivative
from night_run_coordinator import NightRunCoordinatorError
from source_summary_fidelity import guarded_summary
from tests.test_memory_source_fidelity import _real_write_case
from tests.test_recall_summary_reuse import make


SOURCE = '哥哥说：“朝灯，今晚想陪你散步，明天打算给你发邮件。”'
NOTES = (
    '原话没有‘永远爱老婆’的承诺',
    '原话没有说明是否发送',
    '不能据此确认完成',
)


@pytest.mark.parametrize('note', NOTES)
def test_generated_audit_note_is_not_supported_by_silence(note):
    assert 'unsupported_editorial_note' in validate_derivative(
        SOURCE + '；' + note + '。', [SOURCE]
    )


@pytest.mark.parametrize('note', NOTES)
def test_original_audit_speech_survives_including_parent_context(note):
    original = f'朝灯复核时说：“{note}”。'
    envelope = json.dumps({
        'payload': {'text': SOURCE},
        'context_events': [{'payload': {'text': original}}],
    }, ensure_ascii=False)
    summary = json.dumps({'core_facts': [original], 'summary': original}, ensure_ascii=False)

    assert validate_derivative(original, [envelope]) == []
    assert guarded_summary(summary, envelope) == summary


def test_a_different_original_audit_does_not_authorize_a_new_denial():
    source = '朝灯说：“原话没有说明是否发送”。哥哥说：“永远爱老婆。”'
    assert 'unsupported_editorial_note' in validate_derivative(NOTES[0], [source])
    assert validate_derivative('哥哥说：“永远爱老婆。”', [source]) == []


def test_audit_metadata_is_not_original_speech():
    envelope = json.dumps({
        'payload': {'text': SOURCE}, 'audit_note': NOTES[1],
    }, ensure_ascii=False)
    assert 'unsupported_editorial_note' in validate_derivative(NOTES[1], [envelope])


def test_actual_denial_and_speaker_attributed_plan_are_preserved():
    source = '哥哥说：“我没有答应发邮件，我打算明天陪朝灯散步。”'
    summary = json.dumps({'core_facts': [source], 'summary': source}, ensure_ascii=False)
    assert validate_derivative(source, [source]) == []
    assert guarded_summary(summary, source) == summary


@pytest.mark.parametrize('field,note', tuple(zip(
    ('core_facts', 'summary', 'unspoken_need'), NOTES
)))
def test_summary_falls_back_to_original_without_editorial_explanations(field, note):
    generated = {'core_facts': [SOURCE], 'summary': SOURCE}
    generated[field] = [note] if field == 'core_facts' else note
    repaired = json.loads(guarded_summary(json.dumps(generated, ensure_ascii=False), SOURCE))

    assert repaired == {'core_facts': [SOURCE], 'summary': SOURCE}


@pytest.mark.asyncio
@pytest.mark.parametrize('recall', [False, True])
async def test_dehydration_checks_generated_output_before_caching(test_config, recall):
    instance, provider = make(test_config)
    source = SOURCE + '\n朝灯和哥哥一起讨论晚上的散步安排。' * 60
    provider.result = json.dumps({
        'core_facts': [SOURCE], 'summary': NOTES[1],
    }, ensure_ascii=False)
    dehydrate = instance.dehydrate_recall_with_source if recall else instance.dehydrate_with_source

    output, origin = await dehydrate(source)
    expected = {'core_facts': [source[:400]], 'summary': source[:120]}
    assert json.loads(output) == expected
    assert origin == 'computed'
    assert (await dehydrate(source))[0] == output
    assert len(provider.requests) == 1
    assert provider.requests[0]['messages'][1]['content'] == source[:3000]


@pytest.mark.asyncio
async def test_recall_preserves_actual_audit_speech(test_config):
    instance, provider = make(test_config)
    source = f'朝灯复核时说：“{NOTES[1]}”。' + '\n哥哥记录复核结论。' * 80
    provider.result = json.dumps({
        'core_facts': [f'朝灯复核时说：“{NOTES[1]}”。'],
        'summary': f'朝灯指出：{NOTES[1]}。',
    }, ensure_ascii=False)

    assert await instance.dehydrate_recall_with_source(source) == (provider.result, 'computed')
    assert len(provider.requests) == 1


def _candidate(content):
    return {
        'type': 'event', 'title': '散步与邮件计划', 'content': content,
        'importance': 5, 'thread_hint': '', 'relation_hints': [],
        'source_chunk_ids': ['source'], 'evidence': SOURCE, 'risk': 'normal',
    }


@pytest.mark.asyncio
async def test_proposer_preserves_original_audit_speech_without_retry():
    original = f'朝灯复核时说：“{NOTES[1]}”。'
    candidate = _candidate(original)
    candidate['evidence'] = original
    calls = []

    async def provider(prompt):
        calls.append(prompt)
        return {'choices': [{'message': {'content': json.dumps({
            'schema_version': 1, 'candidates': [candidate],
        }, ensure_ascii=False)}}]}

    batch = await StrictOmbreProposer(provider).propose([ProposerChunk('source', original)])
    assert batch.candidates[0].content == original
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('repair_succeeds', [False, True])
async def test_proposer_uses_only_existing_single_repair_for_editorial_notes(repair_succeeds):
    bad = _candidate(SOURCE + '；' + NOTES[1] + '。')
    replies = [bad, _candidate(SOURCE) if repair_succeeds else bad]
    prompts = []

    async def provider(prompt):
        prompts.append(prompt)
        return {'choices': [{'message': {'content': json.dumps({
            'schema_version': 1, 'candidates': [replies.pop(0)],
        }, ensure_ascii=False)}}]}

    proposer = StrictOmbreProposer(provider)
    if repair_succeeds:
        batch = await proposer.propose([ProposerChunk('source', SOURCE)])
        assert batch.candidates[0].content == SOURCE
    else:
        with pytest.raises(ProposerContractError) as caught:
            await proposer.propose([ProposerChunk('source', SOURCE)])
        assert caught.value.code == 'provenance_fidelity'
    assert len(prompts) == 2
    assert 'provenance_fidelity' in prompts[1]


@pytest.mark.asyncio
async def test_legacy_queued_editorial_note_is_checked_against_untouched_original(bucket_mgr):
    coordinator, pending, payload = _real_write_case(bucket_mgr)
    before = [coordinator.ledger.get_raw_event(identity).payload for identity in pending.source_event_ids]
    payload['source'].pop('source_ref', None)
    payload['draft']['content'] += '；' + NOTES[1] + '。'

    with pytest.raises(NightRunCoordinatorError, match='candidate.source_fidelity'):
        await coordinator._write_x_payload(payload)

    assert await bucket_mgr.list_all(include_archive=True) == []
    assert coordinator.curated.embedding_engine.stored == set()
    assert [coordinator.ledger.get_raw_event(identity).payload for identity in pending.source_event_ids] == before
