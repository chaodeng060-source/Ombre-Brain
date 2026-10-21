"""Resolve legacy night-memory pointers without rewriting memory prose.

Only exact manifest targets are touched. Existing reviewed corrections win.
Raw events are never modified or deleted. Repeating the same run does no writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def run(plan: dict, apply: bool) -> dict:
    import frontmatter
    from storage_safety import atomic_write_post
    from maintenance_barrier import MaintenanceBarrier

    root = Path('/data')
    if plan.get('schema') != 'ombre.source-refs-backfill/v1':
        raise ValueError('unexpected source manifest')
    ledger = root / '.lmc5/pipeline.sqlite3'
    result = {'targets': len(plan['records']), 'resolved': 0, 'already_present': 0,
              'written': 0, 'pending': [], 'raw_event_mutations': 0, 'body_mutations': 0}
    backup = Path('/snapshots/memory-fidelity-20261021/buckets')
    prepared = []
    with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        by_event = {}
        by_identity = {}
        for r in db.execute('SELECT id,session_id,source_event_id,payload_digest FROM raw_events'):
            row = dict(r)
            by_event.setdefault(row['source_event_id'], []).append(row)
            by_identity[(row['session_id'], row['source_event_id'])] = row
    for record in plan['records']:
        path = (root / record['source_path']).resolve()
        if not path.is_relative_to(root) or path.suffix != '.md':
            raise ValueError('invalid bucket path')
        post = frontmatter.load(path)
        bid = record['id']
        if post.get('id') != bid:
            raise ValueError('bucket identity mismatch')
        ref = post.get('source_ref')
        if isinstance(ref, dict) and ref.get('database') and ref.get('raw_events'):
            result['already_present'] += 1
            continue
        ids = post.get('source_event_ids') or []
        refs = []
        for eid in ids:
            row = by_identity.get((post.get('source_session'), eid))
            if row is None:
                matches = by_event.get(eid, [])
                row = matches[0] if len(matches) == 1 else None
            if row is None:
                break
            refs.append({'row_id': row['id'], 'session_id': row['session_id'],
                         'source_event_id': row['source_event_id'],
                         'payload_digest': row['payload_digest']})
        if not ids or len(refs) != len(ids):
            result['pending'].append({'id': bid, 'source_path': record['source_path'],
                                      'event_ids': ids, 'reason': 'declared_original_missing_or_ambiguous'})
            continue
        source_ref = {'database': str(ledger), 'table': 'raw_events', 'raw_events': refs}
        result['resolved'] += 1
        if apply:
            prepared.append((bid, path, hashlib.sha256(path.read_bytes()).hexdigest(), source_ref))
    if not apply:
        return result
    backup.mkdir(mode=0o700, parents=True, exist_ok=True)
    barrier = MaintenanceBarrier(root)
    for bid, path, expected, refs in prepared:
        with barrier.exclusive():
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                result['pending'].append({'id': bid, 'reason': 'bucket_changed_during_pointer_backfill'})
                continue
            post = frontmatter.loads(raw.decode())
            snap = backup / (bid + '.md')
            if not snap.exists():
                snap.write_bytes(raw); snap.chmod(0o600)
            post['source_ref'] = refs
            post['source_ref_resolution'] = 'declared_source_resolved'
            atomic_write_post(path, post)
            result['written'] += 1
    directory = root / 'provenance'
    directory.mkdir(mode=0o700, exist_ok=True)
    retention = directory / 'retention.json'
    retention.write_text(json.dumps({'schema': 'ombre.original-retention/v1',
        'policy': 'append-only, excluded from automatic cleanup',
        'stores': ['/data/.lmc5/pipeline.sqlite3', '/data/provenance/raw_events.sqlite3'],
        'original_events_deleted': 0}, ensure_ascii=False, indent=2))
    retention.chmod(0o600)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    print(json.dumps(run(json.load(sys.stdin), args.apply), ensure_ascii=False))
