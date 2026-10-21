"""Apply owner-authorized, original-source-reviewed memory corrections on NAS.

Dry-run by default. The plan contains exact IDs, old body hashes, independently
reviewed replacements and actual original events, never arbitrary shell actions.
Original conversations and existing provenance remain append-only.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def backup_db(path: Path, target: Path) -> None:
    if target.exists() or not path.exists():
        return
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst)
    target.chmod(0o600)


def mirror_sources(root: Path, originals: list[dict]) -> dict:
    directory = root / 'provenance'
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / 'raw_events.sqlite3'
    refs = []
    with sqlite3.connect(path) as db:
        db.execute('''CREATE TABLE IF NOT EXISTS raw_events (
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
            source_event_id TEXT NOT NULL, payload BLOB NOT NULL,
            payload_digest TEXT NOT NULL, recorded_at TEXT NOT NULL,
            UNIQUE(session_id, source_event_id, payload_digest))''')
        db.execute('CREATE INDEX IF NOT EXISTS raw_events_event_id ON raw_events(source_event_id)')
        for event in originals:
            eid, text = event['event_id'], event['text']
            if not eid or not text or event.get('sender') in {None, '', 'unknown'}:
                raise ValueError('original event identity or author missing')
            session = event.get('session_id') or event.get('session_ref') or event.get('conv_id') or eid
            payload = json.dumps({k: event.get(k) for k in (
                'event_id', 'sender', 'text', 'conv_id', 'recorded_at', 'origin', 'origin_locator'
            )}, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            sha = digest(payload)
            db.execute('INSERT OR IGNORE INTO raw_events(session_id,source_event_id,payload,payload_digest,recorded_at) VALUES(?,?,?,?,?)',
                       (session, eid, payload, sha, event.get('recorded_at') or ''))
            rowid = db.execute('SELECT id FROM raw_events WHERE session_id=? AND source_event_id=? AND payload_digest=?',
                               (session, eid, sha)).fetchone()[0]
            refs.append({'row_id': rowid, 'session_id': session, 'source_event_id': eid,
                         'payload_digest': sha, 'text_sha256': digest(text)})
    path.chmod(0o600)
    return {'database': str(path), 'table': 'raw_events', 'raw_events': refs}


def candidate_options(root: Path, keys: set[str]) -> dict[str, dict]:
    found = {}
    with sqlite3.connect(f'file:{root / ".lmc5/pipeline.sqlite3"}?mode=ro', uri=True) as db:
        for payload, in db.execute("SELECT payload FROM candidates WHERE axis='X'"):
            try:
                p = json.loads(payload)
                key = p.get('x_write_key')
                if key not in keys:
                    continue
                d, s = p['draft'], p['source']
                found[key] = {'bucket_type': 'dynamic', 'importance': d['importance'],
                              'name': d['title'], 'tags': ['lmc5', 'night', d['type']],
                              'x_provenance': {'source_kind': 'conversation',
                                               'source_session': s['session_id'],
                                               'source_event_ids': s['source_event_ids'],
                                               'source_digest': s['source_digest']}}
                if s.get('source_ref'):
                    found[key]['x_provenance']['source_ref'] = s['source_ref']
            except (ValueError, KeyError, TypeError):
                continue
    return found


def invalidate_old_caches(root: Path, old_hash: str) -> int:
    count = 0
    for path, table, column in (
        (root / 'dehydration_cache.db', 'dehydration_cache', 'content_hash'),
        (root / 'dehydration_cache.db', 'json_cache', 'cache_key'),
        (root / '.recall_cache/recall_dehydration_cache.db', 'recall_dehydration_cache', 'content_hash'),
    ):
        if not path.exists():
            continue
        with sqlite3.connect(path) as db:
            count += db.execute(f'DELETE FROM {table} WHERE {column} LIKE ?', ('%' + old_hash + '%',)).rowcount
    return count


async def run(plan: dict, apply: bool) -> dict:
    import frontmatter
    from bucket_manager import BucketManager
    from curated_writer import CuratedWriteCoordinator, _payload_digest
    from dehydrator import Dehydrator
    from embedding_engine import EmbeddingEngine
    from redact import redact_embedding_input
    from utils import load_config

    root = Path('/data')
    if plan.get('schema') != 'ombre.memory-fidelity-repair/v1' or plan.get('root') != str(root):
        raise ValueError('unexpected plan')
    records = plan['records']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate targets')
    options = candidate_options(root, {r.get('curated_write_key') for r in records if r.get('curated_write_key')})
    prepared, conflicts = [], []
    ledger_path = root / '.curated_writes.db'
    with sqlite3.connect(f'file:{ledger_path}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in records:
            path = (root / row['source_path']).resolve()
            if not path.is_relative_to(root) or path.suffix != '.md':
                raise ValueError('target outside bucket root')
            post = frontmatter.load(path)
            if post.get('id') != row['id']:
                raise ValueError('bucket id mismatch')
            if post.get('fidelity_repair_revision') == row['revision'] and digest(post.content) == digest(row['body']):
                continue
            if digest(post.content) != row['original_body_sha256']:
                conflicts.append({'id': row['id'], 'reason': 'body_changed_since_review'})
                continue
            if post.get('e_authored_by') and row['body'] != post.content:
                conflicts.append({'id': row['id'], 'reason': 'primary_authored_E_requires_successor'})
                continue
            if row.get('review', {}).get('verdict') != 'accept' or not row.get('originals'):
                raise ValueError('unreviewed correction')
            opt = options.get(post.get('curated_write_key'))
            receipt = None
            if post.get('curated_write_key'):
                receipt = db.execute('SELECT * FROM curated_writes WHERE idempotency_key=? AND bucket_id=?',
                                     (post['curated_write_key'], row['id'])).fetchone()
                if receipt:
                    if not opt:
                        conflicts.append({'id': row['id'], 'reason': 'original_curated_options_missing'})
                        continue
                    opt = CuratedWriteCoordinator._normalize_options(opt)
                    old_digest = _payload_digest(content=post.content, vector_policy=receipt['vector_policy'], bucket_options=opt)
                    if old_digest != receipt['payload_sha256']:
                        conflicts.append({'id': row['id'], 'reason': 'original_curated_options_mismatch'})
                        continue
            prepared.append((row, path, post, opt, dict(receipt) if receipt else None))
    result = {'status': 'dry-run', 'targets': len(records), 'prepared': len(prepared),
              'conflicts': conflicts, 'applied': 0, 'changed_bodies': 0, 'summaries_rebuilt': 0,
              'cache_rows_removed': 0, 'embedding_requests': 0}
    if not apply or not prepared:
        return result

    backup = Path('/snapshots/memory-fidelity-20261021')
    backup.mkdir(mode=0o700, exist_ok=True)
    (backup / 'buckets').mkdir(mode=0o700, exist_ok=True)
    for path in (root / '.curated_writes.db', root / 'embeddings.db', root / 'dehydration_cache.db',
                 root / '.recall_cache/recall_dehydration_cache.db', root / '.lmc5/pipeline.sqlite3'):
        backup_db(path, backup / path.name)
    config = load_config()
    manager, engine, dehydrator = BucketManager(config), EmbeddingEngine(config), Dehydrator(config)
    if not engine.enabled:
        raise RuntimeError('embedding engine disabled')
    receipts_path = backup / 'applied.jsonl'
    for row, path, old_post, opt, ledger in prepared:
        bid = row['id']
        changed = row['body'] != old_post.content
        vector = None
        if changed:
            result['embedding_requests'] += 1
            vector = await engine._generate_embedding(row['body'])
            if vector is None:
                result['conflicts'].append({'id': bid, 'reason': 'replacement_embedding_unavailable'})
                continue
        refs = mirror_sources(root, row['originals'])
        summary = json.dumps(row['summary_object'], ensure_ascii=False, separators=(',', ':'))
        now = datetime.now(timezone.utc).isoformat()
        with manager._maintenance_barrier.exclusive():
            current = frontmatter.load(path)
            if digest(current.content) != row['original_body_sha256']:
                result['conflicts'].append({'id': bid, 'reason': 'body_changed_before_apply'})
                continue
            snap = backup / 'buckets' / (bid + '.md')
            if not snap.exists():
                snap.write_bytes(path.read_bytes()); snap.chmod(0o600)
            before = manager._post_snapshot(current, path)
            current.content = row['body']
            # The original immutable source_* declaration remains intact. This
            # explicit owner maintenance operation adds the resolved correction
            # evidence and retains both chains in the snapshot and audit log.
            current['source_ref'] = refs
            current['fidelity_repair_revision'] = row['revision']
            current['fidelity_repair_at'] = now
            current['fidelity_review_source'] = row['review']['artifact']
            current['dehydrated_summary'] = summary
            current['dehydrated_content_hash'] = digest(row['body'])
            current['name'] = row['name']
            source_text = '\n'.join(e['text'] for e in row['originals'])
            structural = [t for t in current.get('tags', []) if t in {'lmc5', 'night', 'event', 'relationship_moment', 'fact', 'preference', 'decision'} or (isinstance(t, str) and t in source_text)]
            if 'lmc5' in current.get('tags', []):
                current['tags'] = list(dict.fromkeys(structural + row['keywords']))
            current['retrieval_keys'] = row['keywords']
            if ledger:
                new_hash = _payload_digest(content=row['body'], vector_policy=ledger['vector_policy'], bucket_options=opt)
                current['curated_payload_sha256'] = new_hash
            audit_id = manager.audit_log.begin(actor='xiaojuan:owner-source-fidelity-repair', action='update',
                                               bucket_id=bid, before=before, after=manager._post_snapshot(current, path),
                                               details={'reviewed_originals': [r['event_id'] for r in row['originals']],
                                                        'revision': row['revision'], 'backup': str(snap)})
            if vector is not None:
                engine._store_embedding(bid, vector)
            manager._atomic_write_post(path, current, bm25_content_changed=True)
            if ledger:
                with sqlite3.connect(ledger_path) as db:
                    db.execute('UPDATE curated_writes SET payload_sha256=?,updated_at=? WHERE idempotency_key=? AND bucket_id=?',
                               (new_hash, now, ledger['idempotency_key'], bid))
            result['cache_rows_removed'] += invalidate_old_caches(root, row['original_body_sha256'])
            dehydrator._set_cached_summary(redact_embedding_input(row['body']), summary)
            if hasattr(dehydrator, '_recall_summary_request'):
                snapshot = dehydrator._recall_summary_request(redact_embedding_input(row['body']), row['body'])
                dehydrator._set_recall_cached_summary('', summary, snapshot=snapshot)
            manager.audit_log.commit(audit_id)
            verify = frontmatter.load(path)
            if verify.content != row['body'] or verify.get('dehydrated_summary') != summary:
                raise RuntimeError('readback differs: ' + bid)
            with receipts_path.open('a') as f:
                f.write(json.dumps({'id': bid, 'revision': row['revision'], 'body_sha256': digest(row['body']),
                                    'source_ref': refs, 'changed_body': changed, 'at': now}, ensure_ascii=False) + '\n')
        result['applied'] += 1
        result['changed_bodies'] += int(changed)
        result['summaries_rebuilt'] += 1
        print(json.dumps({'progress': result['applied'], 'id': bid}), file=sys.stderr, flush=True)
    result.update(status='applied', backup=str(backup), originals_retained=True)
    return result


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    print(json.dumps(asyncio.run(run(json.load(sys.stdin), args.apply)), ensure_ascii=False), flush=True)
