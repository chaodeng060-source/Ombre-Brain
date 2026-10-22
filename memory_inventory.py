"""Count the current vault by unique bucket ID, separately from vectors/backups."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path


LAYERS = {"permanent": "permanent", "dynamic": "dynamic", "feel": "feel",
          "archive": "archive", "world_isolated": "涩涩"}


def _identity(path: Path) -> tuple[str, str]:
    fields: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        if stream.readline().strip() == "---":
            for line in stream:
                if line.strip() == "---":
                    break
                match = re.match(r"^(id|duplicate_of|exact_duplicate_of):\s*(.*?)\s*$", line)
                if match:
                    fields[match[1]] = match[2].strip("\"'")
    fallback = re.search(r"([0-9a-f]{12})$", path.stem)
    return fields.get("id", fallback[1] if fallback else path.stem), fields.get("duplicate_of", "") or fields.get("exact_duplicate_of", "")


def build_inventory(base_dir: str | Path) -> dict:
    root = Path(base_dir)
    physical: dict[str, set[str]] = {}
    records: dict[str, set[str]] = {}
    alias_ids: set[str] = set()
    file_count = 0
    size_bytes = 0
    domains: dict[str, int] = {}
    for label, directory in LAYERS.items():
        physical[label], records[label] = set(), set()
        for path in (root / directory).rglob("*.md"):
            bucket_id, duplicate_of = _identity(path)
            file_count += 1
            size_bytes += path.stat().st_size
            physical[label].add(bucket_id)
            if duplicate_of:
                alias_ids.add(bucket_id)
            else:
                records[label].add(bucket_id)
                if path.parent.name != directory:
                    domains[path.parent.name] = domains.get(path.parent.name, 0) + 1
    all_ids = set().union(*physical.values())
    logical_ids = set().union(*records.values())
    indexed: set[str] = set()
    database = root / "embeddings.db"
    if database.exists():
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            indexed = {row[0] for row in connection.execute("SELECT bucket_id FROM embeddings")}
    layers = {label: len(ids) for label, ids in records.items()}
    coverage = {label: {"indexed": len(ids & indexed), "missing": len(ids - indexed)}
                for label, ids in records.items()}
    regular_ids = records["permanent"] | records["dynamic"]
    return {"schema": "ombre.memory-inventory/v1", "total_records": len(logical_ids),
            "record_ids": len(all_ids), "alias_count": len(alias_ids),
            "regular_count": len(regular_ids), "layers": layers,
            "index": {"records_indexed": len(logical_ids & indexed),
                      "missing_regular": len(regular_ids - indexed),
                      "missing_feel": len(records["feel"] - indexed),
                      "missing_archive": len(records["archive"] - indexed),
                      "orphan_rows": len(indexed - all_ids), "by_layer": coverage},
            "physical_files": file_count, "duplicate_id_files": file_count - len(all_ids),
            "domains": domains,
            "total_size_kb": size_bytes / 1024,
            "count_basis": "Current vault unique IDs; aliases, backups and vector rows are separate."}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir")
    print(json.dumps(build_inventory(parser.parse_args().base_dir), ensure_ascii=False, indent=2))
