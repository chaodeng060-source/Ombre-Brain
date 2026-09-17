#!/usr/bin/env python3
"""Private, scoped sidecar receipts and compare-and-restore; never restore bodies.

Stop the worker and disable readers first. Capture before and after under the
same production lock. Immutable versions and usage ledgers are retained. PG and
SQLite are separate transactions; the same receipt safely resumes interruption.
"""
import argparse
import base64
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_hints import canonical_json, enabled
from retrieval_hints_batch import existing_host_lock
from retrieval_hints_storage import HintsStore, PostgresHints, VersionConflict
from tools.retrieval_hints_batch import worker_lease


def _sqlite_rows(store, ids, db=None):
    if not store.path.exists():
        return {"versions": [], "published": []}
    if db is None:
        with store._connect() as connection:
            return _sqlite_rows(store, ids, connection)
    slots = ",".join("?" for _ in ids)
    return {table: [dict(row) for row in db.execute(
        f"SELECT * FROM {table} WHERE bucket_id IN ({slots}) ORDER BY bucket_id"
        + (",version" if table == "versions" else ""), ids)]
        for table in ("versions", "published")}


def _pg_rows(peer, ids):
    with peer.connection.transaction():
        if peer.connection.execute("SELECT to_regclass('ombre_retrieval_hints')").fetchone()[0] is None:
            return []
        rows = peer.connection.execute("""SELECT bucket_id,version,source_content_sha256,
            payload_sha256,keys_json,provenance_json,lexical_text,lexical_version,ready
            FROM ombre_retrieval_hints WHERE bucket_id=ANY(%s) ORDER BY bucket_id,version""", (ids,)).fetchall()
    fields = ("bucket_id", "version", "source_content_sha256", "payload_sha256", "keys_json",
              "provenance_json", "lexical_text", "lexical_version", "ready")
    return [dict(zip(fields, row)) for row in rows]


def _dictionary(store):
    path = store.directory / "userdict.txt"
    if path.is_symlink():
        raise ValueError("dictionary_symlink")
    return base64.b64encode(path.read_bytes()).decode("ascii") if path.exists() else None


def capture(store, peer, ids):
    ids = sorted(ids)
    if (not 1 <= len(ids) <= 200 or len(set(ids)) != len(ids)
            or any(not isinstance(bid, str) or not bid for bid in ids)):
        raise ValueError("bounded_distinct_bucket_ids_required")
    return {"schema_version": 1, "store_directory": str(store.directory.resolve()), "bucket_ids": ids,
            "sqlite": _sqlite_rows(store, ids), "postgres": _pg_rows(peer, ids),
            "dictionary": _dictionary(store)}


def write_receipt(path, receipt):
    # Exclusive and private from the first byte; do not overwrite a before-image.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(canonical_json(receipt) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _target_rows(before, after, state_field, new_state):
    old = {(row["bucket_id"], row["version"]): row for row in before}
    current = {(row["bucket_id"], row["version"]): row for row in after}
    if len(old) != len(before) or len(current) != len(after) or not old.keys() <= current.keys():
        raise VersionConflict("receipt_versions_missing")
    target = copy.deepcopy(after)
    for row in target:
        previous = old.get((row["bucket_id"], row["version"]))
        if previous is not None:
            if {k: v for k, v in row.items() if k != state_field} != {
                    k: v for k, v in previous.items() if k != state_field}:
                raise VersionConflict("immutable_version_changed")
            row[state_field] = previous[state_field]
        else:
            row[state_field] = new_state
    return target


def _restore_sqlite(store, ids, expected, target):
    if expected == target:
        return
    with store._connect(write=True) as db:
        db.execute("BEGIN IMMEDIATE")
        current = _sqlite_rows(store, ids, db)
        if current == target:
            return
        if current != expected:
            raise VersionConflict("later_sqlite_change")
        for row in target["versions"]:
            db.execute("UPDATE versions SET state=? WHERE bucket_id=? AND version=?",
                       (row["state"], row["bucket_id"], row["version"]))
        for bid in ids:
            db.execute("DELETE FROM published WHERE bucket_id=?", (bid,))
        db.executemany("INSERT INTO published(bucket_id,version) VALUES (?,?)",
                       [(row["bucket_id"], row["version"]) for row in target["published"]])


def _restore_dictionary(store, expected, target):
    current = _dictionary(store)
    if current == target:
        return
    if current != expected:
        raise VersionConflict("later_dictionary_change")
    path = store.directory / "userdict.txt"
    if target is None:
        path.unlink()  # Exact newly-created dictionary; bytes remain in after receipt.
        return
    with tempfile.NamedTemporaryFile(dir=store.directory, prefix=".userdict-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(base64.b64decode(target, validate=True))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def restore(store, peer, before, after, *, apply=True):
    for key in ("schema_version", "store_directory", "bucket_ids"):
        if before.get(key) != after.get(key):
            raise ValueError("receipt_scope_mismatch")
    if before["schema_version"] != 1 or before["store_directory"] != str(store.directory.resolve()):
        raise ValueError("receipt_target_mismatch")
    ids = before["bucket_ids"]
    current = capture(store, peer, ids)
    for receipt in (before, after):
        for rows in (receipt["sqlite"]["versions"], receipt["sqlite"]["published"], receipt["postgres"]):
            if any(row["bucket_id"] not in ids for row in rows):
                raise ValueError("receipt_row_outside_scope")
        if receipt["dictionary"] is not None:
            base64.b64decode(receipt["dictionary"], validate=True)
    target_sqlite = {
        "versions": _target_rows(before["sqlite"]["versions"], after["sqlite"]["versions"], "state", "withdrawn"),
        "published": before["sqlite"]["published"],
    }
    target_pg = _target_rows(before["postgres"], after["postgres"], "ready", False)
    targets = {"sqlite": target_sqlite, "postgres": target_pg, "dictionary": before["dictionary"]}
    # Preflight every component before touching any; allow already-restored
    # components so a PG-committed / SQLite-interrupted operation can resume.
    for key, target in targets.items():
        if current[key] != after[key] and current[key] != target:
            raise VersionConflict("later_" + key + "_change")
    if not apply:
        return {"status": "ready", "buckets": len(ids), "applied": False}
    if current["postgres"] != target_pg:
        with peer.connection.transaction():
            # Operators hold the shared host/maintenance locks. A table lock
            # also excludes PG writers that do not participate in those leases.
            peer.connection.execute("LOCK TABLE ombre_retrieval_hints IN SHARE ROW EXCLUSIVE MODE")
            if _pg_rows(peer, ids) != after["postgres"]:
                raise VersionConflict("later_postgres_change")
            for row in target_pg:
                peer.connection.execute("UPDATE ombre_retrieval_hints SET ready=%s WHERE bucket_id=%s AND version=%s",
                                        (row["ready"], row["bucket_id"], row["version"]))
    _restore_sqlite(store, ids, after["sqlite"], target_sqlite)
    _restore_dictionary(store, after["dictionary"], before["dictionary"])
    return {"status": "restored", "buckets": len(ids), "applied": True, "usage_ledger_retained": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "restore"))
    parser.add_argument("--buckets-dir", required=True)
    parser.add_argument("--host-lock", required=True)
    parser.add_argument("--lock-device", required=True, type=int)
    parser.add_argument("--lock-inode", required=True, type=int)
    parser.add_argument("--approved-sources")
    parser.add_argument("--output")
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--apply", action="store_true", help="restore defaults to read-only comparison")
    args = parser.parse_args(argv)
    try:
        if any(enabled(name) for name in ("OMBRE_RETRIEVAL_HINTS_ENABLED", "OMBRE_RETRIEVAL_HINTS_BACKFILL_ENABLED",
                                         "OMBRE_RETRIEVAL_ATTRIBUTION_ENABLED", "OMBRE_PRIVATE_RECALL_DICT_ENABLED")):
            raise ValueError("disable_hints_readers_and_worker_first")
        root = Path(args.buckets_dir).resolve()
        if not root.is_dir():
            raise ValueError("source_root_missing")
        import psycopg
        from maintenance_barrier import MaintenanceBarrier
        store = HintsStore(root / ".retrieval_hints")
        with existing_host_lock(args.host_lock, expected_device=args.lock_device, expected_inode=args.lock_inode):
            with worker_lease(store), MaintenanceBarrier(root).exclusive():
                with psycopg.connect(os.environ["OMBRE_PG_RECALL_DSN"], autocommit=True, connect_timeout=3,
                                      options="-c statement_timeout=2000 -c lock_timeout=250") as conn:
                    peer = PostgresHints(conn)
                    if args.command == "capture":
                        approved = json.loads(Path(args.approved_sources).read_text())
                        rows = approved["sources"] if isinstance(approved, dict) else approved
                        receipt = capture(store, peer, [row["bucket_id"] for row in rows])
                        write_receipt(args.output, receipt)
                        result = {"status": "captured", "buckets": len(rows)}
                    else:
                        before = json.loads(Path(args.before).read_text())
                        after = json.loads(Path(args.after).read_text())
                        result = restore(store, peer, before, after, apply=args.apply)
    except Exception as exc:
        print(canonical_json({"status": "error", "error_type": type(exc).__name__}))
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
