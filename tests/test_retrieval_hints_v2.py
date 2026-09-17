import copy
import json

import httpx
import pytest

from retrieval_hints import Generator, HintError, lexical_text, parse_hints
from tests.test_retrieval_hints import BODY, payload


def sources(text=BODY):
    return ({'event_id': 'fixture-event', 'text': text, 'sender': 'user'},)


def v2(attribution='self'):
    value = payload()
    value.update(schema_version=2, keywords=[{'text': '星舟项目', 'evidence': '讨论星舟项目'}],
                 keywords_short_reason='insufficient_evidence',
                 anchors=[{'text': '星舟项目', 'evidence': '讨论星舟项目'}],
                 attribution=attribution, quoted={'self': False, 'quoted': True, 'unknown': None}[attribution],
                 attribution_evidence=['讨论星舟项目'] if attribution != 'unknown' else [])
    return value


def test_v2_keys_and_source_bound_attribution():
    value = parse_hints(json.dumps(v2('quoted')), BODY, source_context=sources())
    assert value['quoted'] is True and '星舟项目' in lexical_text(value)
    assert 'quoted' not in lexical_text(value)


@pytest.mark.parametrize('with_source', [False, True])
def test_short_body_always_unknown_even_when_model_says_quoted(with_source):
    body = '林禾转发银杏馆故事。'
    value = {**{k: [] for k in ('who', 'where', 'when', 'what', 'aliases', 'keywords', 'anchors')},
        'schema_version': 2, 'keywords_short_reason': 'insufficient_evidence',
        'attribution': 'quoted', 'quoted': True, 'attribution_evidence': ['银杏馆故事']}
    parsed = parse_hints(json.dumps(value), body, source_context=sources(body) if with_source else ())
    assert parsed['attribution'] == 'unknown' and parsed['quoted'] is None
    assert parsed['attribution_evidence'] == []


def test_missing_source_cannot_certify_self_or_exclude_quoted():
    for label in ('self', 'quoted'):
        result = parse_hints(json.dumps(v2(label)), BODY)
        assert result['attribution'] == 'unknown' and result['quoted'] is None


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(attribution='mixed'),
    lambda p: p.update(quoted=True),
    lambda p: p.update(attribution_evidence=['a fabricated source']),
    lambda p: p['anchors'][0].update(text='不存在的项目'),
    lambda p: p.update(keywords_short_reason=None),
    lambda p: p['keywords'][0].update(text='决定'),
    lambda p: p.update(keywords=[p['keywords'][0]] * 11),
])
def test_v2_invalid_not_silently_certified(mutate):
    value = copy.deepcopy(v2())
    mutate(value)
    with pytest.raises(HintError):
        parse_hints(json.dumps(value), BODY, source_context=sources())


@pytest.mark.asyncio
async def test_v2_backfill_body_only_no_raw_source_external_input():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                              'message': {'content': json.dumps(v2())}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await Generator(enabled=True, schema_version=2, client=client,
             config=('https://fixture.invalid', 'mock', 'mock')).generate(
                 BODY, source_context=sources('source-private-prefix\n' + BODY))
    assert result.status == 'ok' and result.payload['schema_version'] == 2
    assert result.source_context[0]['event_id'] == 'fixture-event'
    assert calls[0]['messages'][1]['content'] == BODY
    assert 'source-private-prefix' not in json.dumps(calls)
    assert len(calls) == 1


def source_bucket():
    from tests.test_retrieval_hints_storage import bucket
    b = bucket()
    b['metadata']['source_event_ids'] = ['fixture-event']
    return b


def test_v2_sidecar_publication_and_exact_v1_pointer_rollback(tmp_path):
    from retrieval_hints import Generation, PROMPT_VERSION, PROMPT_VERSION_V2
    from retrieval_hints_storage import HintsStore
    from tests.test_retrieval_hints_storage import Peer
    store, peer, b = HintsStore(tmp_path / 'hints'), Peer(), source_bucket()
    old = store.stage(b, Generation('ok', payload()))
    old_row = store.publish(b['id'], old, peer, lambda: b)
    new = store.stage(b, Generation('ok', v2('quoted'), source_context=sources()))
    row = store.publish(b['id'], new, peer, lambda: b)
    assert row['payload']['quoted'] is True and row['attribution_source']['status'] == 'resolved'
    assert row['source_content_sha256'] == old_row['source_content_sha256']
    assert row['prompt_version'] == store.manifest([b])['records'][0]['prompt_version'] == PROMPT_VERSION_V2
    assert 'text' not in row['attribution_source']['events'][0]
    assert 'source_context' not in row
    store.rollback_pointer(b['id'], expected=new, previous=old)
    assert store.lookup(b) == old_row and old_row['prompt_version'] == PROMPT_VERSION


def test_other_event_cannot_certify_attribution_and_source_rebinding_invalidates(tmp_path):
    from retrieval_hints import Generation
    from retrieval_hints_storage import HintsStore, SourceChanged
    from tests.test_retrieval_hints_storage import Peer
    store, peer, b = HintsStore(tmp_path / 'hints'), Peer(), source_bucket()
    foreign = ({**sources()[0], 'event_id': 'other-conversation'},)
    version = store.stage(b, Generation('ok', v2('quoted'), source_context=foreign))
    row = store.publish(b['id'], version, peer, lambda: b)
    assert row['payload']['quoted'] is None and row['attribution_source']['status'] == 'missing'
    b['metadata']['source_event_ids'] = ['rebound-event']
    assert store.lookup(b) is None
    with pytest.raises(SourceChanged):
        store.publish(b['id'], version, peer, lambda: b)


def test_v2_peer_source_proof_must_match_before_admission(tmp_path):
    from retrieval_hints import Generation
    from retrieval_hints_storage import HintsStore, VersionConflict
    from tests.test_retrieval_hints_storage import Peer
    class WrongProof(Peer):
        def get(self, bid, version):
            return {**super().get(bid, version), 'attribution_source': {'status': 'wrong'}}
    store, b = HintsStore(tmp_path / 'hints'), source_bucket()
    version = store.stage(b, Generation('ok', v2('quoted'), source_context=sources()))
    with pytest.raises(VersionConflict):
        store.publish(b['id'], version, WrongProof(), lambda: b)
    assert store.lookup(b) is None
