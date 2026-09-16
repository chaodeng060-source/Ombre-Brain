import hashlib
import json
import sqlite3

from tools.pending_body_duplicates import collect, digest, summarize


def test_census_counts_bodies_not_envelopes_and_is_read_only(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE event_chunks (chunk_id TEXT PRIMARY KEY);
            CREATE TABLE chunk_proposer_outcomes (chunk_id TEXT, outcome TEXT);
            CREATE TABLE raw_events (id INTEGER, payload BLOB, payload_digest TEXT, source_event_id TEXT);
            CREATE TABLE chunk_sources (chunk_id TEXT, raw_event_id INTEGER);
        """)
        for number, body in enumerate(["same body", "same body", "different", None, "terminal body"], 1):
            payload = json.dumps({"text": body, "id": f"synthetic-{number}"}).encode()
            db.execute("INSERT INTO raw_events VALUES (?,?,?,?)", (number, payload, hashlib.sha256(payload).hexdigest(), f"event-{number}"))
            db.execute("INSERT INTO event_chunks VALUES (?)", (str(number),))
            db.execute("INSERT INTO chunk_sources VALUES (?,?)", (str(number), number))
        db.execute("INSERT INTO chunk_sources VALUES ('2',1)")
        db.execute("INSERT INTO chunk_proposer_outcomes VALUES ('5','zero_candidates')")
    before = path.read_bytes()
    result = collect(path, {"target": {"body_sha256": [digest("same body")], "source_event_sha256": [digest("event-1")]}})
    assert path.read_bytes() == before
    summary = result["summary"]
    assert summary["pending_chunks"] == 4
    assert summary["pending_distinct_events"] == 4
    assert summary["events_with_nonblank_text"] == 3
    assert summary["duplicate_body_members"] == 2
    assert summary["duplicate_body_extra_copies"] == 1
    assert summary["duplicate_pending_chunks"] == 2  # not double counted
    assert summary["targets"]["target"]["same_source_event_events"] == 1
    assert summarize(result["records"], result["pending_chunk_ids"], result["targets"]) == summary
    assert "same body" not in json.dumps(result)
