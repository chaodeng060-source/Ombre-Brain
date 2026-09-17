#!/usr/bin/env python3
"""Zero-provider four-way BM25 candidate comparison, NOT final breath evidence."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm25_index import BM25Index
from retrieval_hints import canonical_json, content_hash
from retrieval_tokenizer import RetrievalTokenizer


def compare(snapshot, manifest, dictionary, turns, *, limit=20):
    rows = {row["bucket_id"]: row for row in manifest["records"]}
    augmented = [{**bucket, "retrieval_hints_v1": rows[bucket["id"]]}
                 if bucket["id"] in rows else bucket for bucket in snapshot]
    private = RetrievalTokenizer.from_file(dictionary)
    indexes = {
        "OFF": BM25Index(), "dictionary_only": BM25Index(tokenizer=private),
        "keys_only": BM25Index(hints_enabled=True),
        "both": BM25Index(tokenizer=private, hints_enabled=True),
    }
    for name, index in indexes.items():
        index.build(augmented if name in {"keys_only", "both"} else snapshot)
    for turn in turns:
        query = turn["user_text"]
        result = {"n": turn["n"], "segment": turn["segment"], "ts": turn["ts"], "user_text": query,
                  "layer": "bm25_candidates_only", "provider_calls": 0,
                  "temporal_scope": "same_current_snapshot_not_historical_reconstruction",
                  "delivery_status": turn.get("delivery_status", "not_replayed"),
                  "final_breath": "pending_night_pilot", "conditions": {}}
        for name, index in indexes.items():
            start = time.monotonic()
            scores = index.score(query)
            selected = sorted(scores, key=lambda bid: (-scores[bid], bid))[:limit]
            hints = index.hint_matches(query)
            result["conditions"][name] = {
                "bucket_ids": selected, "scores": [scores[bid] for bid in selected],
                "hint_match": {bid: hints[bid] for bid in selected if bid in hints},
                "elapsed_ms": (time.monotonic() - start) * 1000,
                "lexical_version": index.lexical_version,
            }
        yield result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "manifest", "dictionary", "turns", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    snapshot = [json.loads(row) for row in Path(args.snapshot).read_text().splitlines()]
    manifest = json.loads(Path(args.manifest).read_text())
    turns = [json.loads(row) for row in Path(args.turns).read_text().splitlines()]
    if len({row["id"] for row in snapshot}) != len(snapshot):
        raise ValueError("duplicate_snapshot_ids")
    if any(row["source_content_sha256"] != content_hash(row["content"]) for row in snapshot):
        raise ValueError("snapshot_content_mismatch")
    with Path(args.output).open("x", encoding="utf-8") as stream:
        for row in compare(snapshot, manifest, args.dictionary, turns):
            stream.write(canonical_json(row) + "\n")
    Path(args.output).chmod(0o600)
    print(canonical_json({"rows": len(turns), "provider_calls": 0, "layer": "bm25_candidates_only"}))


if __name__ == "__main__":
    main()
