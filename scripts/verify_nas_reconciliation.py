#!/usr/bin/env python3
"""Verify the Git/source side of the NAS live-reconciliation manifest.

This check intentionally does not claim that the manifest covers every live
container difference: CI has no live-container baseline.  The NAS deployment
preflight remains responsible for comparing the live/source Python union and
rejecting missing records.  Here we only prove that every tracked source anchor
still describes the checkout which would be published.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path


INVALID = 2
NOT_READY = 3
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TOP_LEVEL_PY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.py\Z")


class ManifestError(ValueError):
    """The manifest cannot safely describe this source checkout."""


@dataclass(frozen=True)
class Record:
    relative_path: str
    live_sha256: str
    source_sha256: str
    line_number: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _is_anchor(value: str) -> bool:
    return value == "ABSENT" or SHA256_RE.fullmatch(value) is not None


def _read_records(manifest: Path) -> list[Record]:
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read reconciliation manifest: {exc}") from exc

    records: list[Record] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3 or any(not field for field in fields):
            raise ManifestError(
                f"invalid reconciliation record at line {line_number}: expected 3 tab-separated fields"
            )
        relative_path, live_sha256, source_sha256 = fields
        if TOP_LEVEL_PY_RE.fullmatch(relative_path) is None:
            raise ManifestError(f"unsafe top-level Python path: {relative_path}")
        if relative_path in seen:
            raise ManifestError(f"duplicate reconciliation record: {relative_path}")
        seen.add(relative_path)
        if not _is_anchor(live_sha256):
            raise ManifestError(
                f"invalid live reconciliation anchor: {relative_path}"
            )
        if source_sha256 != "UNRECONCILED" and not _is_anchor(source_sha256):
            raise ManifestError(
                f"invalid source reconciliation anchor: {relative_path}"
            )
        if live_sha256 == source_sha256 == "ABSENT":
            raise ManifestError(
                f"both reconciliation anchors cannot be ABSENT: {relative_path}"
            )
        records.append(
            Record(relative_path, live_sha256, source_sha256, line_number)
        )

    if not records:
        raise ManifestError("reconciliation manifest has no records")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(source_root: Path, manifest: Path) -> tuple[bool, list[str]]:
    try:
        source_root = source_root.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"cannot resolve source root: {exc}") from exc
    if not source_root.is_dir():
        raise ManifestError(f"source root is not a directory: {source_root}")

    records = _read_records(manifest)
    unresolved: list[str] = []
    for record in records:
        source_path = source_root / record.relative_path
        if record.source_sha256 == "UNRECONCILED":
            unresolved.append(record.relative_path)
            continue
        if record.source_sha256 == "ABSENT":
            if source_path.exists() or source_path.is_symlink():
                raise ManifestError(
                    f"source reconciliation anchor changed: {record.relative_path}"
                )
            continue
        if not source_path.is_file() or source_path.is_symlink():
            raise ManifestError(
                f"source reconciliation anchor changed: {record.relative_path}"
            )
        try:
            actual_sha256 = _sha256(source_path)
        except OSError as exc:
            raise ManifestError(
                f"cannot hash source reconciliation path: {record.relative_path}: {exc}"
            ) from exc
        if actual_sha256 != record.source_sha256:
            raise ManifestError(
                f"source reconciliation anchor changed: {record.relative_path}"
            )
    return not unresolved, unresolved


def main() -> int:
    args = _parse_args()
    try:
        ready, unresolved = verify(args.source_root, args.manifest)
    except ManifestError as exc:
        print(f"deploy_ready=false error={exc}", file=sys.stderr)
        return INVALID

    if not ready:
        print(
            "deploy_ready=false status=UNRECONCILED paths=" + ",".join(unresolved)
        )
        return NOT_READY
    print("deploy_ready=true status=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
