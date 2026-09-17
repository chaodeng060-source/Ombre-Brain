import copy
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

import server
from bucket_manager import BucketManager
from retrieval_hints import Generation
from tests.test_retrieval_hints_storage import Peer
from tests.test_retrieval_hints_v2 import BODY, v2, sources, source_bucket
from tests.test_breath_dehydration_read_only import _GraphManager, _Embedding, _Decay, _NoopLoop


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*a, **kw):
        pytest.fail('no real provider calls permitted')
    monkeypatch.setattr(socket.socket, 'connect', denied)


@pytest.fixture
def manager(test_config, monkeypatch):
    monkeypatch.setenv('OMBRE_RETRIEVAL_HINTS_ENABLED', '1')
    monkeypatch.setenv('OMBRE_RETRIEVAL_ATTRIBUTION_ENABLED', '1')
    return BucketManager(test_config)


def publish(manager, bucket, attribution):
    store = manager._retrieval_hints_store
    version = store.stage(bucket, Generation('ok', v2(attribution), source_context=sources()))
    store.publish(bucket['id'], version, Peer(), lambda: bucket)
    manager._retrieval_hint_rows = store.published_snapshot()
    return version


def test_only_published_current_quoted_excluded_and_both_off_restore(manager):
    b = source_bucket()
    store = manager._retrieval_hints_store
    store.stage(b, Generation('ok', v2('quoted'), source_context=sources()))
    assert manager.retrieval_attribution_eligible(b)
    publish(manager, b, 'quoted')
    assert not manager.retrieval_attribution_eligible(b)
    assert manager.retrieval_attribution_eligible({**b, 'content': BODY + 'changed'})
    changed = copy.deepcopy(b)
    changed['metadata']['source_event_ids'] = ['other-event']
    assert manager.retrieval_attribution_eligible(changed)
    for state in ('self', 'unknown'):
        publish(manager, b, state)
        assert manager.retrieval_attribution_eligible(b)
    publish(manager, b, 'quoted')
    manager._retrieval_attribution_enabled = False
    assert manager.retrieval_attribution_eligible(b)


@pytest.mark.asyncio
async def test_lexical_search_filters_even_without_bm25(manager):
    quoted, unknown = source_bucket(), {**source_bucket(), 'id': 'unknown'}
    publish(manager, quoted, 'quoted')
    manager._bm25_mode = 'off'
    result = await manager.search('星舟项目', preloaded_buckets=[quoted, unknown],
        relevance_first=True, relevance_candidate_floor=0)
    assert [b['id'] for b in result] == ['unknown']


@pytest.mark.asyncio
async def test_real_breath_merge_force_keep_and_relation_cannot_reintroduce_quoted(manager, monkeypatch, tmp_path):
    main = {**source_bucket(), 'id': 'main', 'score': 100.0}
    quoted = {**source_bucket(), 'id': 'quoted', 'score': 100.0}
    unknown = {**source_bucket(), 'id': 'unknown', 'score': 100.0}
    for b in (main, quoted, unknown):
        b['metadata'] = {**b['metadata'], 'name': b['id'], 'domain': ['生活'],
            'tags': [], 'importance': 5, 'retrieval_keys': ['星舟项目'], 'world': 'daily'}
    main['metadata']['relations'] = [{'type': 'explains', 'target': 'quoted', 'strength': 1.0},
                                   {'type': 'explains', 'target': 'unknown', 'strength': 1.0}]
    # Distinct bodies avoid unrelated body dedup hiding the quoted failure.
    quoted['content'] += ' quoted fixture'
    unknown['content'] += ' unknown fixture'
    publish(manager, quoted, 'quoted')
    graph = _GraphManager(main, [main, quoted, unknown], tmp_path / 'archive')
    graph._retrieval_attribution_enabled = True
    graph.retrieval_attribution_eligible = manager.retrieval_attribution_eligible
    async def refresh(buckets):
        await manager._refresh_retrieval_hint_index(buckets)
    graph._refresh_retrieval_hint_index = refresh
    # Simulate a stale lexical producer ignoring the new sidecar.
    async def stale_search(*a, **kw):
        return [main, quoted]
    graph.search = stale_search
    monkeypatch.setattr(server, 'bucket_mgr', graph)
    monkeypatch.setattr(server, 'embedding_engine', _Embedding())
    monkeypatch.setattr(server, 'decay_engine', _Decay())
    monkeypatch.setattr(server, 'consolidation_engine', _NoopLoop())
    monkeypatch.setattr(server, 'episode_engine', _NoopLoop())
    monkeypatch.setattr(server, '_backfill_started', True)
    monkeypatch.setattr(server, 'config', {**server.config, 'current_world': 'daily',
        'entities': {'enabled': False}, 'query_expansion': {'enabled': False}, 'random_surfacing': {},
        'relation_recall': {'propagation_only': True, 'propagation_types': ['explains'],
                            'hop1_min_strength': .4, 'hop2_min_strength': .7}})
    monkeypatch.setenv('OMBRE_DS_FILTER_ENABLED', '0')
    monkeypatch.setenv('OMBRE_LMC5_NIGHT_ENABLED', '1')
    rendered = []
    async def dehydrate(content, metadata, *, bucket, **kwargs):
        rendered.append(bucket['id'])
        return content
    monkeypatch.setattr(server, '_dehydrate_for_recall', dehydrate)
    result = await server.breath(query='星舟项目', max_results=3, relation_depth=1, world='daily',
                               include_images=False, include_body_state=False)
    assert '[bucket_id:main]' in result
    assert '[bucket_id:unknown]' in result
    assert '[bucket_id:quoted]' not in result and 'quoted' not in rendered


@pytest.mark.asyncio
async def test_hold_confirmation_is_not_a_recall_and_still_confirms_quoted(manager, monkeypatch):
    bid = await manager.create(content=BODY, world='daily',
                               x_provenance={'source_kind': 'conversation',
                                             'source_event_ids': ['fixture-event']})
    bucket = await manager.get(bid)
    publish(manager, bucket, 'quoted')
    assert not manager.retrieval_attribution_eligible(bucket)
    monkeypatch.setattr(server, 'bucket_mgr', manager)
    result = await server.api_hold_status(SimpleNamespace(query_params={
        'world': 'daily', 'content_sha256': hashlib.sha256(BODY.encode()).hexdigest()}))
    assert json.loads(result.body)['state'] == 'stored'
