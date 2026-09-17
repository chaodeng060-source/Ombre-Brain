"""Private derived sidecars. Never writes bucket, receipt or vector tables.

SQLite's publication pointer is authoritative. PG consumers MUST intersect
ready versions with that pointer (or its versioned manifest); ready alone is
not admission. Old versions are retained for bounded compare-and-restore.
The production runner holds the host production lock across publication.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

from retrieval_hints import (
    PROMPT_VERSION, Generation, canonical_json, content_hash, lexical_text,
    parse_hints, source_record,
)


class SourceChanged(ValueError):
    pass


class VersionConflict(ValueError):
    pass


def active(bucket) -> bool:
    return bool(bucket) and (bucket.get("metadata") or {}).get("type") != "archived"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _job_key(source):
    return content_hash(canonical_json([source["bucket_id"], source["source_content_sha256"], source["prompt_version"]]))


class HintsStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "index.sqlite3"
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with sqlite3.connect(self.path, timeout=.1) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_key TEXT PRIMARY KEY, bucket_id TEXT NOT NULL,
                    source_json TEXT NOT NULL, kind TEXT NOT NULL,
                    state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    bucket_id TEXT NOT NULL, version TEXT NOT NULL,
                    record_json TEXT NOT NULL, state TEXT NOT NULL,
                    PRIMARY KEY(bucket_id,version)
                );
                CREATE TABLE IF NOT EXISTS published (
                    bucket_id TEXT PRIMARY KEY, version TEXT NOT NULL
                );
            """)
        os.chmod(self.path, 0o600)
        self._initialized = True

    @contextmanager
    def _connect(self, *, write=False):
        if write:
            self._initialize()
            db = sqlite3.connect(self.path, timeout=.1)
        else:
            db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=.1)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, bucket, *, kind="backfill"):
        if not active(bucket):
            return None
        if kind not in {"new_write", "backfill"}:
            raise ValueError("job kind")
        source = source_record(bucket)
        key = _job_key(source)
        with self._connect(write=True) as db:
            db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?,?,?, 'queued',0,NULL,?)",
                       (key, source["bucket_id"], canonical_json(source), kind, _now()))
            if kind == "new_write":
                db.execute("UPDATE jobs SET kind='new_write' WHERE job_key=?", (key,))
        return key

    def jobs(self):
        if not self.path.exists():
            return []
        with self._connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM jobs ORDER BY CASE kind WHEN 'new_write' THEN 0 ELSE 1 END,created_at,job_key")]

    def reserve(self, key):
        with self._connect(write=True) as db:
            return db.execute("UPDATE jobs SET state='inflight',attempts=attempts+1 "
                              "WHERE job_key=? AND state='queued' AND attempts=0", (key,)).rowcount == 1

    def finish_job(self, key, result: Generation):
        receipt = {"status": result.status, "usage": result.usage, "requested_model": result.requested_model,
                   "response_model": result.response_model, "elapsed_ms": result.elapsed_ms,
                   "error": result.error, "outbound_calls": result.outbound_calls}
        with self._connect(write=True) as db:
            db.execute("UPDATE jobs SET state=?, result_json=? WHERE job_key=?",
                       ("generated" if result.status == "ok" else result.status, canonical_json(receipt), key))

    def stage(self, bucket, result: Generation, *, lexical_version="jieba-search-v1", tokens=None):
        if result.status != "ok" or not active(bucket):
            raise ValueError("only successful active sources can be staged")
        payload = parse_hints(canonical_json(result.payload), bucket.get("content") or "")
        source = source_record(bucket)
        terms = lexical_text(payload)
        row = {**source, "payload": payload, "payload_sha256": content_hash(canonical_json(payload)),
               "lexical_text": " ".join(tokens(terms)) if tokens else terms,
               "lexical_version": lexical_version}
        version = content_hash(canonical_json(row))
        row.update(version=version, generated_at=_now(), usage=result.usage,
                   requested_model=result.requested_model, response_model=result.response_model)
        with self._connect(write=True) as db:
            db.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?,'pending')",
                       (source["bucket_id"], version, canonical_json(row)))
        return version

    def pending(self):
        if not self.path.exists():
            return []
        with self._connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT record_json FROM versions WHERE state='pending' ORDER BY bucket_id,version")]

    def _version(self, bid, version):
        with self._connect() as db:
            row = db.execute("SELECT record_json FROM versions WHERE bucket_id=? AND version=?", (bid, version)).fetchone()
        if row is None:
            raise VersionConflict("unknown_version")
        return json.loads(row[0])

    @staticmethod
    def _check_source(row, bucket):
        if (not active(bucket) or str(bucket["id"]) != row["bucket_id"]
                or content_hash(bucket.get("content") or "") != row["source_content_sha256"]):
            raise SourceChanged("source_changed_or_inactive")

    def publish(self, bid, version, peer, load_current):
        row = self._version(bid, version)
        self._check_source(row, load_current())
        peer.stage(row)
        mirrored = peer.get(bid, version)
        if (not mirrored or canonical_json(mirrored["payload"]) != canonical_json(row["payload"])
                or mirrored["source_content_sha256"] != row["source_content_sha256"]
                or mirrored["lexical_text"] != row["lexical_text"]
                or mirrored["lexical_version"] != row["lexical_version"]):
            raise VersionConflict("peer_mismatch")
        peer.ready(bid, version)
        self._check_source(row, load_current())
        with self._connect(write=True) as db:
            db.execute("UPDATE versions SET state='published' WHERE bucket_id=? AND version=?", (bid, version))
            db.execute("INSERT INTO published VALUES (?,?) ON CONFLICT(bucket_id) DO UPDATE SET version=excluded.version", (bid, version))
        return row

    def published_snapshot(self):
        if not self.path.exists():
            return {}
        with self._connect() as db:
            return {row[0]: json.loads(row[1]) for row in db.execute(
                "SELECT p.bucket_id,v.record_json FROM published p JOIN versions v "
                "ON p.bucket_id=v.bucket_id AND p.version=v.version WHERE v.state='published'")}

    def lookup(self, bucket):
        if not active(bucket) or not self.path.exists():
            return None
        with self._connect() as db:
            found = db.execute("SELECT v.record_json FROM published p JOIN versions v "
                "ON p.bucket_id=v.bucket_id AND p.version=v.version "
                "WHERE p.bucket_id=? AND v.state='published'", (str(bucket["id"]),)).fetchone()
        if found is None:
            return None
        row = json.loads(found[0])
        try:
            self._check_source(row, bucket)
        except SourceChanged:
            return None
        return row

    def manifest(self, buckets):
        snapshot = self.published_snapshot()
        records = []
        seen = set()
        for bucket in buckets:
            row = snapshot.get(str(bucket["id"]))
            if not row or row["bucket_id"] in seen:
                continue
            try:
                self._check_source(row, bucket)
            except SourceChanged:
                continue
            source = source_record(bucket)  # Current author/world/path, not stale generation metadata.
            records.append({**row, **source, "source_kind": "bucket_body",
                            "source_link_status": "present_unverified" if source["source_links"] else "missing"})
            seen.add(row["bucket_id"])
        records.sort(key=lambda row: row["bucket_id"])
        return {"schema_version": 1, "publication_version": content_hash(canonical_json(records)), "records": records}

    def rollback_pointer(self, bid, *, expected, previous):
        with self._connect(write=True) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT version FROM published WHERE bucket_id=?", (bid,)).fetchone()
            if not row or row[0] != expected:
                raise VersionConflict("later_publication_exists")
            if previous is None:
                db.execute("DELETE FROM published WHERE bucket_id=? AND version=?", (bid, expected))
            else:
                old = db.execute("SELECT state FROM versions WHERE bucket_id=? AND version=?", (bid, previous)).fetchone()
                if not old or old[0] != "published":
                    raise VersionConflict("rollback_version_missing")
                db.execute("UPDATE published SET version=? WHERE bucket_id=?", (previous, bid))
            # Retain both derived versions as before-images; never delete source data.
            db.execute("UPDATE versions SET state='withdrawn' WHERE bucket_id=? AND version=?", (bid, expected))


class PostgresHints:
    """A dedicated connection for only the new side table; caller closes it."""
    def __init__(self, connection):
        self.connection = connection

    def initialize(self):
        with self.connection.transaction():
            self.connection.execute("""CREATE TABLE IF NOT EXISTS ombre_retrieval_hints (
                bucket_id TEXT NOT NULL, version TEXT NOT NULL,
                source_content_sha256 TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                keys_json JSONB NOT NULL, provenance_json JSONB NOT NULL,
                lexical_text TEXT NOT NULL, lexical_version TEXT NOT NULL,
                search_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, lexical_text)) STORED,
                ready BOOLEAN NOT NULL DEFAULT FALSE, PRIMARY KEY(bucket_id,version))""")
            self.connection.execute("CREATE INDEX IF NOT EXISTS ombre_retrieval_hints_search_idx "
                                    "ON ombre_retrieval_hints USING GIN(search_tsv)")

    def stage(self, row):
        with self.connection.transaction():
            self.connection.execute("""INSERT INTO ombre_retrieval_hints
                (bucket_id,version,source_content_sha256,payload_sha256,keys_json,provenance_json,lexical_text,lexical_version)
                VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s)
                ON CONFLICT(bucket_id,version) DO NOTHING""",
                (row["bucket_id"], row["version"], row["source_content_sha256"], row["payload_sha256"],
                 canonical_json(row["payload"]), canonical_json({k: v for k, v in row.items() if k != "payload"}),
                 row["lexical_text"], row["lexical_version"]))

    def get(self, bid, version):
        with self.connection.transaction():
            found = self.connection.execute("SELECT provenance_json,keys_json,ready FROM ombre_retrieval_hints "
                "WHERE bucket_id=%s AND version=%s", (bid, version)).fetchone()
        return {**found[0], "payload": found[1], "ready": found[2]} if found else None

    def ready(self, bid, version):
        with self.connection.transaction():
            self.connection.execute("UPDATE ombre_retrieval_hints SET ready=TRUE WHERE bucket_id=%s AND version=%s", (bid, version))
