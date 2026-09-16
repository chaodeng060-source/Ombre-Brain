"""Read-only full pending-event body census; stdlib only, no application import.

Run with --ledger PATH --records to retain a hash-only recomputable snapshot.
Targets JSON maps labels to body_sha256 and/or source_event_sha256 (lists).
No trimming or semantic normalization is used for body equality.
"""
import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


PENDING = """SELECT ec.chunk_id FROM event_chunks ec WHERE NOT EXISTS (
    SELECT 1 FROM chunk_proposer_outcomes cpo WHERE cpo.chunk_id=ec.chunk_id
    AND cpo.outcome IN ('zero_candidates','candidates_persisted'))"""


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def summarize(records, pending_chunks, targets):
    groups = defaultdict(list)
    for record in records:
        if record["body_sha256"]:
            groups[record["body_sha256"]].append(record)
    duplicates = [rows for rows in groups.values() if len(rows) > 1]
    nonblank = sum(map(len, groups.values()))
    members = sum(map(len, duplicates))
    extras = sum(len(rows) - 1 for rows in duplicates)
    duplicate_chunks = {cid for rows in duplicates for row in rows for cid in row["chunk_ids"]}
    matches = {}
    for label, target in targets.items():
        body = set(target.get("body_sha256", []))
        source = set(target.get("source_event_sha256", []))
        by_body = [r for r in records if r["body_sha256"] in body]
        by_source = [r for r in records if source.intersection(r["source_event_sha256"])]
        matches[label] = {
            "exact_body_events": len(by_body),
            "exact_body_pending_chunks": len({c for r in by_body for c in r["chunk_ids"]}),
            "same_source_event_events": len(by_source),
            "same_source_event_pending_chunks": len({c for r in by_source for c in r["chunk_ids"]}),
        }
    return {
        "pending_chunks": len(pending_chunks),
        "pending_distinct_events": len(records),
        "events_with_nonblank_text": nonblank,
        "events_without_nonblank_text": len(records) - nonblank,
        "duplicate_body_groups": len(duplicates),
        "duplicate_body_members": members,
        "duplicate_body_extra_copies": extras,
        "duplicate_member_fraction": members / nonblank if nonblank else None,
        "extra_copy_fraction": extras / nonblank if nonblank else None,
        "duplicate_pending_chunks": len(duplicate_chunks),
        "duplicate_pending_chunk_fraction": len(duplicate_chunks) / len(pending_chunks) if pending_chunks else None,
        "largest_body_group": max(map(len, duplicates), default=0),
        "targets": matches,
        "meaning": "Exact text equality across distinct events is not permission to skip or delete events.",
    }


def collect(ledger, targets=None):
    targets = targets or {}
    with closing(sqlite3.connect(f"{Path(ledger).absolute().as_uri()}?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        chunks = sorted(row[0] for row in db.execute(PENDING))
        rows = db.execute(f"""WITH pending AS ({PENDING})
            SELECT re.id,re.payload,re.payload_digest,re.source_event_id,cs.chunk_id
            FROM pending JOIN chunk_sources cs ON cs.chunk_id=pending.chunk_id
            JOIN raw_events re ON re.id=cs.raw_event_id ORDER BY re.id,cs.chunk_id""")
        events = {}
        for rid, raw, payload_digest, source_event, chunk_id in rows:
            if rid not in events:
                raw = bytes(raw)
                if hashlib.sha256(raw).hexdigest() != payload_digest:
                    raise ValueError("raw event payload digest mismatch")
                obj = json.loads(raw)
                body = obj.get("text")
                sources = {str(v) for v in (source_event, obj.get("sourceEventId"), obj.get("id")) if v}
                events[rid] = {
                    "event_rowid": rid,
                    "body_sha256": digest(body) if isinstance(body, str) and body.strip() else None,
                    "source_event_sha256": sorted(digest(v) for v in sources),
                    "chunk_ids": [],
                }
            events[rid]["chunk_ids"].append(chunk_id)
        records = list(events.values())
        for record in records:
            record["chunk_ids"] = sorted(set(record["chunk_ids"]))
        db.rollback()
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "schema": "pending-body-census/v1",
        "summary": summarize(records, chunks, targets),
        "targets": targets,
        "pending_chunk_ids": chunks,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ledger")
    source.add_argument("--snapshot", help="recompute from a prior hash-only snapshot")
    parser.add_argument("--targets-json", default="{}")
    parser.add_argument("--records", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        result = json.loads(Path(args.snapshot).read_text())
        result["summary"] = summarize(result["records"], result["pending_chunk_ids"], result["targets"])
    else:
        result = collect(args.ledger, json.loads(args.targets_json))
    if not args.records:
        result = {key: value for key, value in result.items() if key not in {"records", "pending_chunk_ids", "targets"}}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
