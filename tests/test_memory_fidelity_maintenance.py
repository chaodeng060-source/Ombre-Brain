import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


spec = importlib.util.spec_from_file_location(
    'repair_memory_fidelity', Path(__file__).resolve().parents[1] / 'tools/repair_memory_fidelity.py')
maintenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maintenance)


def test_original_mirror_is_exact_resolvable_and_deduplicated(tmp_path):
    original = {'event_id': 'source-1', 'session_id': 'session-1', 'sender': 'user',
                'text': '咖啡是黑白边牧。\n九岁多了。\u2028完整原话。',
                'recorded_at': '2026-10-19T20:32:45+08:00', 'origin': 'original-transcript'}
    one = maintenance.mirror_sources(tmp_path, [original])
    two = maintenance.mirror_sources(tmp_path, [original])
    assert one == two
    with sqlite3.connect(one['database']) as db:
        assert db.execute('SELECT count(*) FROM raw_events').fetchone()[0] == 1
        row = db.execute('SELECT payload,payload_digest FROM raw_events WHERE id=?',
                         (one['raw_events'][0]['row_id'],)).fetchone()
    assert json.loads(row[0])['text'] == original['text']
    assert hashlib.sha256(row[0]).hexdigest() == row[1]


def test_missing_speaker_cannot_be_presented_as_original(tmp_path):
    with pytest.raises(ValueError, match='author missing'):
        maintenance.mirror_sources(tmp_path, [{'event_id': 'x', 'text': '不能猜是谁说的。', 'sender': 'unknown'}])


def test_pointer_is_small_not_embedded_original_session(tmp_path):
    text = '原始消息。' * 1000
    refs = maintenance.mirror_sources(tmp_path, [
        {'event_id': 'source-2', 'session_id': 'session-2', 'sender': 'claude', 'text': text}])
    pointer = json.dumps(refs, ensure_ascii=False)
    assert len(pointer) < 800
    assert text not in pointer
