#!/usr/bin/env python3
"""Print a content-free manifest of Markdown relation frontmatter.

The command reads only the active recall directories.  It emits counts and a
SHA-256 over normalized source/type/target/strength tuples; memory bodies,
names and paths never leave the process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import frontmatter


ACTIVE_DIRS = ("permanent", "dynamic", "feel")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buckets-dir",
        type=Path,
        default=Path(os.environ.get("OMBRE_BUCKETS_DIR", "buckets")),
    )
    args = parser.parse_args()

    root = args.buckets_dir.resolve(strict=True)
    paths = sorted(
        path
        for dirname in ACTIVE_DIRS
        for path in (root / dirname).rglob("*.md")
    )
    rows: list[tuple[str, str, str, str]] = []
    bucket_ids: set[str] = set()
    bad_files = 0
    for path in paths:
        try:
            post = frontmatter.load(str(path))
        except Exception:
            bad_files += 1
            continue
        source_id = str(post.get("id") or path.stem)
        bucket_ids.add(source_id)
        relations = post.get("relations") or []
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            rows.append(
                (
                    source_id,
                    str(relation.get("type") or ""),
                    str(relation.get("target") or ""),
                    str(relation.get("strength", 1.0)),
                )
            )

    rows.sort()
    normalized = "\n".join("\t".join(row) for row in rows).encode("utf-8")
    payload = {
        "schema": "ombre-relation-graph-manifest/v1",
        "active_directories": list(ACTIVE_DIRS),
        "markdown_files": len(paths),
        "unique_bucket_ids": len(bucket_ids),
        "relation_rows": len(rows),
        "relation_types": dict(sorted(Counter(row[1] for row in rows).items())),
        "relation_manifest_sha256": hashlib.sha256(normalized).hexdigest(),
        "bad_files": bad_files,
        "contains_memory_bodies": False,
        "contains_bucket_ids": False,
        "contains_paths": False,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if bad_files == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
