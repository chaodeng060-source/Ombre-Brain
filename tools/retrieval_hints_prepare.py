#!/usr/bin/env python3
"""Prepare a deterministic private pilot from a read-only full-corpus snapshot.

Inputs contain source records, not model prompts or per-query gold answers.
Dictionary proposals are review material; this tool never installs userdict.txt.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_hints import GENERIC_HINTS, canonical_json, content_hash, source_record
from retrieval_hints_storage import active


def select_sources(records, required_ids, *, count=200, recent_month, seed="retrieval-hints-pilot-v1"):
    by_id = {}
    duplicates = set()
    excluded = {}
    for row in records:
        bid = str(row["id"])
        if bid in by_id:
            duplicates.add(bid)
        by_id[bid] = row
        if not active(row):
            excluded[bid] = "inactive"
        elif any(marker in (row.get("content") or "") for marker in
                 ("task_200", "retrieval_keys_plan_20260917", "recall-noise-compare-20260917", "CLAUDE_LABELS_BC.md")):
            excluded[bid] = "pilot_or_evaluation_material"
    for bid in duplicates:
        excluded[bid] = "duplicate_active_id"
    selected = []
    required_status = []
    for bid in dict.fromkeys(required_ids):
        status = "included" if bid in by_id and bid not in excluded else excluded.get(bid, "not_active_in_snapshot")
        required_status.append({"bucket_id": bid, "status": status})
        if status == "included":
            selected.append(by_id[bid])
    if len(selected) > count:
        raise ValueError("required_ids_exceed_sample_size")
    chosen = {row["id"] for row in selected}
    groups = defaultdict(list)
    for bid, row in by_id.items():
        if bid in chosen or bid in excluded:
            continue
        meta = row.get("metadata") or {}
        created = str(meta.get("created") or "")
        size = len((row.get("content") or "").encode())
        band = "oversize" if size > 16384 else "long" if size > 4096 else "short"
        key = (created.startswith(recent_month), canonical_json(meta.get("domain") or []),
               str(meta.get("world") or ""), band)
        groups[key].append(row)
    # Balance recent/older samples first; hundreds of distinct older domains
    # must not consume the sample before the first recent-domain queue is read.
    # Within each age band, round-robin domain/world/length strata as before.
    queues = {False: deque(), True: deque()}
    for key, group in sorted(groups.items(), key=lambda pair: pair[0]):
        queues[key[0]].append(deque(sorted(group, key=lambda row: content_hash(seed + str(row["id"])))))
    counts = {False: 0, True: 0}
    for row in selected:
        counts[str((row.get("metadata") or {}).get("created") or "").startswith(recent_month)] += 1
    while len(selected) < count and any(queues.values()):
        age = min((age for age in queues if queues[age]), key=lambda age: (counts[age], not age))
        group = queues[age].popleft()
        selected.append(group.popleft())
        counts[age] += 1
        if group:
            queues[age].append(group)
    if len(selected) != count:
        raise ValueError("not_enough_eligible_sources")
    return selected, required_status, excluded


def dictionary_proposals(records, *, limit=80):
    import jieba.posseg
    proposals = {}
    for row in records:
        body = row.get("content") or ""
        # Wikilinks and POS-tagged proper nouns are proposals, not identity truth.
        words = [(term, "source_wikilink") for term in re.findall(r"\[\[([^\[\]|\n]{2,24})\]\]", body)]
        words += [(word, tag) for word, tag in jieba.posseg.cut(body) if tag in {"nr", "ns", "nt", "nz"}]
        for word, kind in words:
            if not 2 <= len(word) <= 24 or word in GENERIC_HINTS or word not in body or any(c.isspace() for c in word):
                continue
            proposal = proposals.setdefault(word, {"term": word, "kind": kind, "approval": "pending", "sources": []})
            if any(source["bucket_id"] == row["id"] for source in proposal["sources"]):
                continue
            start = body.index(word)
            proposal["sources"].append({"bucket_id": row["id"], "evidence": body[max(0,start-24):start+len(word)+24]})
    return sorted(proposals.values(), key=lambda p: (-len(p["sources"]), -len(p["term"]), p["term"]))[:limit]


def write_private(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--required-ids", required=True)
    parser.add_argument("--recent-month", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    raw = Path(args.snapshot).read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    if any("snapshot_error" in row for row in records):
        raise ValueError("incomplete_snapshot")
    required = json.loads(Path(args.required_ids).read_text(encoding="utf-8"))
    selected, required_status, excluded = select_sources(records, required, recent_month=args.recent_month)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    proposals = dictionary_proposals(selected)
    write_private(output / "approved_sources.proposed.json", {
        "schema_version": 1, "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot_records": len(records), "selection_count": len(selected),
        "source_freeze": "per-file-read-only-not-atomic-global-snapshot",
        "selection_rule": "required-first; recent/older balanced; domain/world/length round-robin; seeded ID order",
        "recent_month": args.recent_month,
        "recent_count": sum(str((r.get("metadata") or {}).get("created") or "").startswith(args.recent_month)
                            for r in selected),
        "sources": [source_record(row) for row in selected],
        "required_bc_cases": required_status, "excluded_count": len(excluded),
    })
    write_private(output / "dictionary.proposals.json", proposals)
    write_private(output / "selected_bodies.private.json", selected)
    print(canonical_json({"selected": len(selected), "required_bc_cases": len(required_status),
        "required_included": sum(r["status"] == "included" for r in required_status),
        "proposals_pending_review": len(proposals), "oversize": sum(len(r["content"].encode()) > 16384 for r in selected)}))


if __name__ == "__main__":
    main()
