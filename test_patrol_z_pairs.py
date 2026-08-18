"""patrol → Z 轴同槽新旧候选 → review_queue（只入队、不改桶）的集成测试。"""
import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path

import patrol
from review_queue import KIND_Z_CONFLICT, STATUS_PENDING, ReviewQueue

REGISTRY = {
    "infra.memory_store.location": {
        "aliases": ["记忆库", "记忆库位置"],
        "domains": ["工程"],
        "types": ["dynamic"],
        "name_contains": ["记忆库"],
    },
}


def _write(dir_path, rel, text):
    p = Path(dir_path) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _seed(d):
    files = [
        _write(d, "dynamic/mem-old.md",
               "---\nid: mem-old\nname: 记忆库位置\ntype: dynamic\ndomain: [工程]\n"
               "created: '2026-07-01T10:00:00+08:00'\n---\n记忆库: 本地 NAS\n"),
        _write(d, "dynamic/mem-new.md",
               "---\nid: mem-new\nname: 记忆库位置\ntype: dynamic\ndomain: [工程]\n"
               "created: '2026-08-01T10:00:00+08:00'\n---\n记忆库: 远程 memory.zhaodeng.xyz\n"),
        # protected: feel type / 恋爱 domain / permanent — never a Z candidate
        _write(d, "feel/love.md",
               "---\nid: love\nname: 记忆库纪念\ntype: feel\ndomain: [恋爱]\n"
               "created: '2026-07-15T10:00:00+08:00'\n---\n记忆库: 心里\n"),
        _write(d, "permanent/perm.md",
               "---\nid: perm\nname: 记忆库位置\ntype: permanent\ndomain: [工程]\n"
               "created: '2026-06-01T10:00:00+08:00'\n---\n记忆库: 石头上\n"),
    ]
    return files


def _digest(paths):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_patrol_proposes_and_enqueues_z_pairs_without_touching_buckets():
    with tempfile.TemporaryDirectory() as d:
        files = _seed(d)
        before = _digest(files)
        report = patrol.patrol(Path(d), datetime(2026, 8, 18, 22, 0, 0), fact_slot_registry=REGISTRY)

        cands = report["z_pair_candidates"]
        assert [(c["current_bucket_id"], c["historical_bucket_id"], c["fact_key"]) for c in cands] == [
            ("mem-new", "mem-old", "infra.memory_store.location"),
        ]
        assert {"love", "perm"}.isdisjoint(
            {c["current_bucket_id"] for c in cands} | {c["historical_bucket_id"] for c in cands}
        )
        assert report["z_pair_stats"]["candidates"] == 1

        queue_path = Path(d) / "review_queue.jsonl"
        queue = ReviewQueue(queue_path)
        assert patrol.enqueue_z_pair_candidates(report, queue) == 1
        # idempotent: second pass adds nothing
        assert patrol.enqueue_z_pair_candidates(report, queue) == 0

        rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        z_rows = [r for r in rows if r.get("kind") == KIND_Z_CONFLICT]
        assert len(z_rows) == 1
        row = z_rows[0]
        assert row["status"] == STATUS_PENDING
        assert row["current_bucket_id"] == "mem-new"
        assert row["historical_bucket_id"] == "mem-old"
        assert row["fact_key"] == "infra.memory_store.location"
        assert row["source"] == patrol.Z_PAIR_QUEUE_SOURCE
        assert row["field"] == "fact_status"

        # buckets are byte-identical: patrol never writes fact_status
        assert _digest(files) == before

        rendered = patrol.render_md(report, Path(d), datetime(2026, 8, 18, 22, 0, 0))
        assert "Z轴同槽新旧候选" in rendered
        assert "mem-new" in rendered and "mem-old" in rendered


def test_patrol_without_registry_produces_no_z_pairs():
    with tempfile.TemporaryDirectory() as d:
        _seed(d)
        report = patrol.patrol(Path(d), datetime(2026, 8, 18, 22, 0, 0), fact_slot_registry={})
        assert report["z_pair_candidates"] == []
        assert report["z_pair_stats"]["buckets_in_slots"] == 0
        assert patrol.z_pair_entries(report) == []
