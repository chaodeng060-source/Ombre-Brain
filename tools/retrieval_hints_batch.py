#!/usr/bin/env python3
"""Explicitly invoked night pilot. This command never installs itself.

The production host lock must already be bound into this execution namespace;
device/inode are verified against the operator's read-only host receipt.
"""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_hints import Generator, canonical_json, enabled, provider_config
from retrieval_hints_batch import NightBatch, TZ, existing_host_lock, night_seconds_left
from retrieval_hints_storage import HintsStore, PostgresHints


def load_source(path, root):
    import frontmatter
    source = Path(path).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        return None
    if source.relative_to(root).parts[0] not in {"permanent", "dynamic", "feel", "涩涩"}:
        return None
    post = frontmatter.load(source)
    return {"id": str(post.get("id", source.stem)), "content": post.content,
            "metadata": dict(post.metadata), "path": str(source)}


@contextmanager
def worker_lease(store):
    import fcntl
    store.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = store.directory / "worker.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


async def run(args):
    if not enabled():
        return {"status": "disabled", "outbound_calls": 0}
    if not night_seconds_left(datetime.now(TZ)):
        return {"status": "outside_night_window", "outbound_calls": 0}
    binding = provider_config()  # Only existing NAS binding; no copying keys.
    dsn = os.environ.get("OMBRE_PG_RECALL_DSN", "").strip()
    if not dsn:
        raise ValueError("existing_pg_binding_missing")
    if args.lock_device is None or args.lock_inode is None:
        raise ValueError("host_lock_identity_receipt_required")
    approved = json.loads(Path(args.approved_sources).read_text(encoding="utf-8"))
    if isinstance(approved, dict):
        approved = approved["sources"]
    if not isinstance(approved, list) or not 1 <= len(approved) <= 200:
        raise ValueError("approved_pilot_source_list_required")
    root = Path(args.buckets_dir).resolve()
    if not root.is_dir():
        raise ValueError("source_root_missing")
    store = HintsStore(root / ".retrieval_hints")
    paths = {row["bucket_id"]: row["source_path"] for row in approved}
    lease = lambda: existing_host_lock(args.host_lock, expected_device=args.lock_device, expected_inode=args.lock_inode)
    import httpx
    import psycopg
    from maintenance_barrier import MaintenanceBarrier
    tokenizer = None
    if enabled("OMBRE_PRIVATE_RECALL_DICT_ENABLED"):
        from retrieval_tokenizer import RetrievalTokenizer
        tokenizer = RetrievalTokenizer.from_file(store.directory / "userdict.txt")
    with lease():
        barrier = MaintenanceBarrier(root)
    async with httpx.AsyncClient(timeout=15.0) as client:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=3,
                              options="-c statement_timeout=2000 -c lock_timeout=250") as conn:
            peer = PostgresHints(conn)
            with lease(), barrier.exclusive():
                if night_seconds_left(datetime.now(TZ)) <= 2:
                    return {"status": "window_closing", "outbound_calls": 0}
                peer.initialize()
            batch = NightBatch(enabled=True, backfill_enabled=enabled("OMBRE_RETRIEVAL_HINTS_BACKFILL_ENABLED"),
                store=store, generator=Generator(enabled=True, client=client, config=binding), peer=peer,
                load_source=lambda bid: load_source(paths[bid], root), publish_lock=lease,
                worker_lock=lambda: worker_lease(store), maintenance_lock=barrier.exclusive,
                tokenizer=tokenizer, limit=args.limit)
            return await batch.run(approved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets-dir", required=True)
    parser.add_argument("--approved-sources", required=True)
    parser.add_argument("--host-lock", required=True)
    parser.add_argument("--lock-device", type=int)
    parser.add_argument("--lock-inode", type=int)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        # Never print DSN, provider response bodies, credentials or source text.
        print(canonical_json({"status": "error", "error_type": type(exc).__name__}))
        return 2
    print(canonical_json(result))
    return 0 if result["status"] in {"disabled", "outside_night_window", "pass_finished"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
