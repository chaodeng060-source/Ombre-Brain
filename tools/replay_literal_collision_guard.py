"""Diagnostic Anchor replay, not a gold or end-to-end recall evaluation.

Input is a private JSON ledger with rows[{ombre_request_id, ids_out, anchors}].
Each anchor records id/s/lit/vec/ent/rare and term_dfs_local_snapshot. Original
keyword membership is absent from old logs: report both possible outcomes,
never invent that membership. No model, embedding, corpus write or reranking.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


def replay(rows):
    results = []
    for row in rows:
        anchors = {a['id']: a for a in row['anchors']}
        candidates = []
        for bid in row['ids_out']:
            a = anchors.get(bid)
            if a is None:
                candidates.append({'id': bid, 'status': 'no_anchor_receipt'})
                continue
            base = {
                'id': bid, '_literal_relevance_score': a['lit'],
                '_original_vector_relevance_score': a['vec'],
                'entity_match': a['ent'], '_rare_literal_terms': a['rare'],
                '_literal_term_dfs': [
                    {'term': term, 'df': df}
                    for term, df in a['term_dfs_local_snapshot'].items()
                ],
            }
            os.environ['OMBRE_LITERAL_COLLISION_GUARD_ENABLED'] = '0'
            off = server._anchor_adapted_relevance_score(dict(base))
            os.environ['OMBRE_LITERAL_COLLISION_GUARD_ENABLED'] = '1'
            on = {}
            for kw in (False, True):
                b = dict(base, _keyword_channel_match=kw)
                on[str(kw).lower()] = server._anchor_adapted_relevance_score(b)
            candidates.append({
                'id': bid, 'status': 'keyword_membership_unknown',
                'recorded_score': a['s'], 'off_score': off,
                'off_matches_receipt': abs(off - a['s']) <= 0.000001,
                'on_score_if_keyword': on,
            })
        results.append({'request_id': row['ombre_request_id'], 'candidates': candidates})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ledger', type=Path)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text())
    print(json.dumps({'kind': 'diagnostic_bounds_not_acceptance',
        'corpus_count': ledger['corpus_count'], 'results': replay(ledger['rows'])}, indent=2))
