from datetime import datetime, timezone
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from bucket_manager import BucketManager
from curated_writer import CuratedWriteCoordinator
from lmc5_proposer import ProposerChunk, ProposerContractError, StrictOmbreProposer, _candidate_as_json
from night_run_coordinator import _draft_json
from tests.test_lmc5_proposer import CHUNKS, _candidate, _document, _provider
from tests.test_night_run_coordinator import _harness, _RetryEmbedding
from tests.test_retrieval_hints import BODY
from tests.test_retrieval_hints_v2 import v2, sources, source_bucket


def document():
    return {'schema_version': 2, 'candidates': [_candidate(type='event', content=BODY,
        evidence='讨论星舟项目', retrieval_hints=v2())]}


@pytest.mark.asyncio
async def test_proposer_v2_source_to_draft_serializers():
    batch = await StrictOmbreProposer(_provider(json.dumps(document())), retrieval_hints_enabled=True).propose(
        (ProposerChunk('chunk-1', BODY),))
    assert batch.schema_version == 2
    for serialize in (_candidate_as_json, _draft_json):
        assert serialize(batch.candidates[0])['retrieval_hints'] == v2()


@pytest.mark.asyncio
@pytest.mark.parametrize('broken', ['anchor', 'foreign_evidence', 'schema'])
async def test_proposer_rejects_unbound_v2(broken):
    value = document()
    hint = value['candidates'][0]['retrieval_hints']
    if broken == 'anchor':
        hint['anchors'] = []
    elif broken == 'foreign_evidence':
        hint['attribution_evidence'] = ['not a source span']
    else:
        value['schema_version'] = 1
    with pytest.raises(ProposerContractError):
        await StrictOmbreProposer(_provider(json.dumps(value)), retrieval_hints_enabled=True).propose(
            (ProposerChunk('chunk-1', BODY),))


@pytest.mark.asyncio
async def test_off_prompt_output_and_candidate_identity_match_prior_commit(monkeypatch):
    # Execute the pre-change implementation, not a second handwritten oracle.
    old = {}
    for filename in ('lmc5_proposer', 'night_run_coordinator'):
        module = ModuleType('_hints_baseline_' + filename)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        source = subprocess.check_output(['git', 'show', 'efa96a7:' + filename + '.py'], text=True)
        exec(compile(source, filename + '.py', 'exec'), module.__dict__)
        old[filename] = module
    output = json.dumps(_document(_candidate()))
    before = old['lmc5_proposer'].StrictOmbreProposer(_provider(output))
    after = StrictOmbreProposer(_provider(output), retrieval_hints_enabled=False)
    old_chunks = tuple(old['lmc5_proposer'].ProposerChunk(c.id, c.text) for c in CHUNKS)
    first, second = await before.propose(old_chunks), await after.propose(CHUNKS)
    assert first.prompt_digest == second.prompt_digest
    assert first.output_digest == second.output_digest
    assert _draft_json(second.candidates[0]) == old['night_run_coordinator']._draft_json(first.candidates[0])
    from night_run_coordinator import NightRunCoordinator
    pending = SimpleNamespace(source_event_ids=[SimpleNamespace(session_id='fixture', source_event_id='event-1')],
        chunk_id='chunk-1', content_digest='a' * 64, created_at='2026-09-17T00:00:00Z')
    args = dict(run_id='fixture-run', pending=pending)
    assert NightRunCoordinator._candidate_specs(None, batch=second, **args) == (
        old['night_run_coordinator'].NightRunCoordinator._candidate_specs(None, batch=first, **args))


@pytest.mark.asyncio
@pytest.mark.parametrize('stage_fails', [False, True])
async def test_real_night_pipeline_retains_hints_without_an_extra_model(tmp_path, test_config, monkeypatch, stage_fails):
    monkeypatch.setenv('OMBRE_RETRIEVAL_HINTS_ENABLED', '1')
    monkeypatch.setenv('OMBRE_RETRIEVAL_ATTRIBUTION_ENABLED', '1')
    harness = _harness(tmp_path)
    config = {**test_config, 'buckets_dir': str(harness.source)}
    manager = BucketManager(config)
    embedding = _RetryEmbedding()
    embedding.calls = 1  # first local embedding succeeds
    harness.coordinator.curated = CuratedWriteCoordinator(manager, embedding)
    harness.coordinator.bucket_manager = manager
    prompts = []
    def provider(prompt):
        prompts.append(prompt)
        data = json.loads(prompt.split('INPUT=', 1)[1])
        output = document()
        output['candidates'][0]['source_chunk_ids'] = [data['chunks'][0]['id']]
        return {'choices': [{'message': {'content': json.dumps(output)}}]}
    harness.coordinator.proposer = StrictOmbreProposer(provider, retrieval_hints_enabled=True,
        model='fixture-model', provider_name='fixture-provider')
    if stage_fails:
        def broken(*a, **kw):
            raise OSError('fixture sidecar unavailable')
        monkeypatch.setattr(manager._retrieval_hints_store, 'stage', broken)
    harness.ledger.append_raw_event('fixture-room', 'fixture-event', json.dumps({'message': BODY}))
    result = await harness.coordinator.run(run_id='fixture-night', cutoff=datetime.now(timezone.utc))
    assert result.counts['x_ready'] == 1 and len(prompts) == 1
    ready = [c for c in harness.ledger.list_candidates('ready') if c.axis == 'X']
    assert len(ready) == 1
    persisted = json.loads(ready[0].payload)
    assert persisted['draft']['retrieval_hints'] == v2()
    buckets = await manager.list_all(include_archive=False)
    assert len(buckets) == 1 and buckets[0]['content'] == BODY
    assert 'retrieval_hints' not in buckets[0]['metadata']
    pending = manager._retrieval_hints_store.pending()
    if stage_fails:
        assert pending == []
        assert manager._retrieval_hints_store.jobs()[0]['state'] == 'queued'
    else:
        assert len(pending) == 1 and pending[0]['payload'] == v2()
        assert pending[0]['generation_origin'] == 'night_proposer'
        assert pending[0]['attribution_source']['events'][0]['event_id'] == 'fixture-event'
        assert pending[0]['proposer']['model'] == 'fixture-model'
        assert manager._retrieval_hints_store.lookup(buckets[0]) is None  # not yet PG-published


@pytest.mark.asyncio
async def test_night_publishes_precomputed_proposer_without_generation(tmp_path):
    from retrieval_hints import Generation, source_record
    from tests.test_retrieval_hints_batch import setup_batch
    bucket = source_bucket()
    from tests.test_retrieval_hints_batch import Model
    from retrieval_hints import PROMPT_VERSION_V2
    generator = Model()
    generator.schema_version = 2
    generator.prompt_version = PROMPT_VERSION_V2
    batch, _, store, model = setup_batch(tmp_path, count=1, load_source=lambda _: bucket,
        generator=generator)
    store.stage(bucket, Generation('ok', v2(), source_context=sources(),
                                  generation_provenance={'model': 'fixture'}))
    outcome = await batch.run([source_record(bucket)])
    assert outcome['published'] == 1 and model.calls == outcome['outbound_calls'] == 0
