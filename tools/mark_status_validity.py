#!/usr/bin/env python3
"""Apply audited operational-status validity markers to real bucket IDs."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from status_validity import OperationalStatusValidityStore


_BUCKET_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _locate_bucket(root: Path, bucket_id: str) -> Path:
    if not _BUCKET_ID_RE.fullmatch(bucket_id):
        raise ValueError(f"invalid bucket id: {bucket_id}")
    frontmatter_id = re.compile(
        rf"(?m)^id:\s*['\"]?{re.escape(bucket_id)}['\"]?\s*$"
    )
    matches = []
    for path in root.rglob("*.md"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                head = handle.read(32768)
        except (OSError, UnicodeError):
            continue
        if frontmatter_id.search(head):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one Markdown bucket for {bucket_id}, found {len(matches)}"
        )
    return matches[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets-dir", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    current = subparsers.add_parser("current")
    current.add_argument("--bucket-id", required=True)
    current.add_argument("--status-key", required=True)
    current.add_argument("--valid-at", required=True)
    current.add_argument("--source-ref", required=True)

    historical = subparsers.add_parser("historical")
    historical.add_argument("--bucket-id", required=True)
    historical.add_argument("--status-key", required=True)
    historical.add_argument("--valid-at", required=True)
    historical.add_argument("--invalid-at", required=True)
    historical.add_argument("--source-ref", required=True)
    historical.add_argument("--superseded-by", default="")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(args.buckets_dir).resolve()
    if not root.is_dir():
        raise RuntimeError(f"buckets directory does not exist: {root}")
    bucket_path = _locate_bucket(root, args.bucket_id)
    store = OperationalStatusValidityStore(
        str(root / ".validity" / "operational_status.sqlite3")
    )
    if args.command == "current":
        store.mark_current(
            args.bucket_id,
            status_key=args.status_key,
            valid_at=args.valid_at,
            source_ref=args.source_ref,
        )
    else:
        if args.superseded_by:
            _locate_bucket(root, args.superseded_by)
        store.mark_historical(
            args.bucket_id,
            status_key=args.status_key,
            valid_at=args.valid_at,
            invalid_at=args.invalid_at,
            source_ref=args.source_ref,
            superseded_by_bucket_id=args.superseded_by,
        )
    marker = store.lookup_many([args.bucket_id])[args.bucket_id]
    print(json.dumps({
        "bucket_id": args.bucket_id,
        "bucket_path": str(bucket_path),
        "marker": marker,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
