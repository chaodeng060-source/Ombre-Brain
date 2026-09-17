#!/usr/bin/env python3
"""Export only published, source-current hints to a separate private directory."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_hints import canonical_json
from retrieval_hints_storage import HintsStore
from tools.retrieval_hints_batch import load_source


def write_manifest(manifest, directory):
    version = manifest.get("publication_version")
    if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("invalid_publication_version")
    directory = Path(directory).resolve()
    if directory.is_relative_to((Path.home() / "imprint-mirror").resolve()):
        raise ValueError("cannot_write_conversation_originals")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = "manifest-" + version + ".json"
    target = directory / name
    data = canonical_json(manifest) + "\n"
    try:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(data)
    except FileExistsError:
        if target.read_text(encoding="utf-8") != data:
            raise ValueError("existing_version_content_mismatch")
    target.chmod(0o400)
    pointer = {"schema_version": 1, "manifest_file": name, "publication_version": manifest["publication_version"]}
    current = directory / "CURRENT.json"
    if current.exists() and set(json.loads(current.read_text())) != set(pointer):
        raise ValueError("foreign_pointer_file")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".CURRENT-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(canonical_json(pointer) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o400)
    os.replace(temporary, current)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets-dir", required=True)
    parser.add_argument("--approved-sources", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.buckets_dir).resolve()
    approved = json.loads(Path(args.approved_sources).read_text())
    approved = approved["sources"] if isinstance(approved, dict) else approved
    buckets = [bucket for row in approved if (bucket := load_source(row["source_path"], root))]
    manifest = HintsStore(root / ".retrieval_hints").manifest(buckets)
    target = write_manifest(manifest, args.output_dir)
    print(canonical_json({"manifest": str(target), "records": len(manifest["records"])}))


if __name__ == "__main__":
    main()
