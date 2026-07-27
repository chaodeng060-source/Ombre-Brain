#!/usr/bin/env python3
"""pending 审计队列 review_queue —— #2 Z轴事实演化 + #3 关系闸的共用「待审」存储
（对位 lmc-5 的 z_conflict_audits + relation review-plan）。

设计铁律（直接抄 lmc-5，也守咱家 5.10/5.14 教训）：
  1. **机器只入队、不落库。** 自动推断出的「危险」边、合并时检出的事实冲突，
     都先挂成 pending 行给人过目，绝不静默改写真相
     （"keep fact changes reviewable instead of silently rewriting truth"）。
  2. **append-only + 入队去重。** 同一 (来源, 类型, 目标/字段值) 只挂一次，
     不刷屏；幂等，重复 enqueue 返回 False。
  3. **lifecycle 显式。** pending → reviewed/applied/rejected 只能由人显式 resolve，
     没有自动流转；resolve 是唯一会重写文件的写入路径。
  4. 队列本身不删任何桶、不动任何边——它只是一张「待人看」的清单。

存储：一行一个 JSON 对象的 .jsonl，落在 <buckets_dir>/review_queue.jsonl。

A pending-review queue shared by Z-axis fact evolution (#2) and relation safety
gating (#3). Machines only enqueue candidates here; the queue is append-only with
enqueue-dedup, and resolve() is the only path that changes a row's pending status.
(Note: #2 is an audit trail — ordinary merges still proceed; the queue records them
for review rather than blocking the truth store.  Safety gates default enabled.)
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


# 队列里两类条目（kind）：
KIND_RELATION = "relation"    # #3：机器自动推断的「危险」关系边（因果/取代类）
KIND_Z_CONFLICT = "z_conflict"  # #2：合并时检出的事实冲突（数字/日期/否定翻转）

# 状态机：机器只写 pending，其余只能人显式 resolve。
STATUS_PENDING = "pending"
STATUS_REVIEWED = "reviewed"   # 人看过、判定保留候选（不应用）
STATUS_APPLIED = "applied"     # 人确认应用（真去建边 / 真去 supersede）
STATUS_REJECTED = "rejected"   # 人否决候选

# REST/UI may only acknowledge or reject a candidate.  ``applied`` is kept in
# the storage state machine for a future explicit memory-write transaction, but
# marking a queue row applied is not itself such a transaction.
REST_SAFE_RESOLVE_STATUSES = {STATUS_REVIEWED, STATUS_REJECTED}


def rest_resolve_status_allowed(status: str) -> bool:
    return str(status or "").strip() in REST_SAFE_RESOLVE_STATUSES


def lifecycle_updates(entry: dict) -> tuple[dict, dict]:
    """Build paired metadata updates for an approved cross-bucket candidate."""
    if entry.get("candidate_type") != "cross_bucket_lifecycle":
        raise ValueError("not a cross-bucket lifecycle candidate")
    current_id = str(entry.get("current_bucket_id") or "").strip()
    historical_id = str(entry.get("historical_bucket_id") or "").strip()
    fact_key = str(entry.get("key") or "").strip()
    if not current_id or not historical_id or not fact_key or current_id == historical_id:
        raise ValueError("invalid lifecycle candidate")
    return (
        {
            "lifecycle": "current",
            "active_fact": True,
            "fact_key": fact_key,
            "supersedes_bucket_ids": [historical_id],
        },
        {
            "lifecycle": "historical",
            "active_fact": False,
            "fact_key": fact_key,
            "superseded_by_bucket_id": current_id,
        },
    )


def historical_recall_suppressed(metadata: dict) -> bool:
    """Suppress only when all explicit Z lifecycle signals agree."""
    if not isinstance(metadata, dict):
        return False
    return (
        str(metadata.get("lifecycle") or "").strip().lower() == "historical"
        and metadata.get("active_fact") is False
        and bool(str(metadata.get("fact_key") or "").strip())
        and bool(str(metadata.get("superseded_by_bucket_id") or "").strip())
    )


HISTORICAL_QUERY_CUES = (
    "以前", "过去", "上次", "历史", "当时", "之前", "曾经", "那次",
    "old", "previous", "historical", "before", "back then",
)


def query_requests_history(query: str) -> bool:
    normalized = str(query or "").strip().lower()
    return any(cue in normalized for cue in HISTORICAL_QUERY_CUES)


def make_currentness_overlay_entry(entry: dict, now: Optional[datetime] = None) -> dict:
    current_update, historical_update = lifecycle_updates(entry)
    return {
        "key": entry["key"],
        "status": "active",
        "current_bucket_id": entry["current_bucket_id"],
        "historical_bucket_id": entry["historical_bucket_id"],
        "fact_key": current_update["fact_key"],
        "source": "reviewed_protected_overlay",
        "created": _now_iso(now),
    }


class CurrentnessOverlay:
    """Append-only reviewed currentness map that never rewrites protected buckets."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except Exception:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows

    def add(self, entry: dict) -> bool:
        key = str(entry.get("key") or "").strip()
        if not key:
            raise ValueError("overlay entry requires key")
        if any(row.get("key") == key and row.get("status") == "active" for row in self._load()):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(self.path, 0o600)
        return True

    def suppresses(self, bucket_id: str, query: str) -> bool:
        if query_requests_history(query):
            return False
        return any(
            row.get("status") == "active"
            and row.get("historical_bucket_id") == bucket_id
            and row.get("current_bucket_id")
            for row in self._load()
        )


def _now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def _short_hash(*parts: str) -> str:
    h = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


def make_relation_entry(
    source_id: str,
    target_id: str,
    rel_type: str,
    note: str = "",
    *,
    source_name: str = "",
    target_name: str = "",
    now: Optional[datetime] = None,
) -> dict:
    """#3：一条等待人审的机器自动推断关系边。"""
    return {
        "key": "rel|" + _short_hash(source_id, rel_type, target_id),
        "kind": KIND_RELATION,
        "status": STATUS_PENDING,
        "source_id": source_id,
        "source_name": source_name,
        "target_id": target_id,
        "target_name": target_name,
        "rel_type": rel_type,
        "note": note or "",
        "created": _now_iso(now),
    }


def make_z_conflict_entry(
    bucket_id: str,
    field: str,
    old: str,
    new: str,
    *,
    bucket_name: str = "",
    reason: str = "",
    now: Optional[datetime] = None,
) -> dict:
    """#2：一条等待人审的事实演化冲突（合并时 old→new 翻转）。"""
    old, new = str(old), str(new)
    return {
        # 同桶同字段、同一组 old→new 才算同一事件；值变了就是新事件，值得再记一次。
        "key": "z|" + _short_hash(bucket_id, field, old, new),
        "kind": KIND_Z_CONFLICT,
        "status": STATUS_PENDING,
        "bucket_id": bucket_id,
        "bucket_name": bucket_name,
        "field": field,
        "old": old[:240],
        "new": new[:240],
        "reason": reason,
        "created": _now_iso(now),
    }


def make_z_pair_entry(
    current_bucket_id: str,
    historical_bucket_id: str,
    *,
    current_name: str = "",
    historical_name: str = "",
    reason: str = "cross_bucket_currentness",
    source: str = "quality_benchmark",
    now: Optional[datetime] = None,
) -> dict:
    """Cross-bucket currentness candidate; enqueue only, never mutates buckets."""
    current_bucket_id = str(current_bucket_id).strip()
    historical_bucket_id = str(historical_bucket_id).strip()
    if not current_bucket_id or not historical_bucket_id:
        raise ValueError("both bucket ids are required")
    if current_bucket_id == historical_bucket_id:
        raise ValueError("current and historical candidates must differ")
    return {
        "key": "zpair|" + _short_hash(current_bucket_id, historical_bucket_id),
        "kind": KIND_Z_CONFLICT,
        "status": STATUS_PENDING,
        "candidate_type": "cross_bucket_lifecycle",
        "bucket_id": current_bucket_id,
        "bucket_name": current_name,
        "current_bucket_id": current_bucket_id,
        "historical_bucket_id": historical_bucket_id,
        "historical_bucket_name": historical_name,
        "field": "lifecycle",
        "old": historical_name or "历史候选",
        "new": current_name or "当前候选",
        "reason": reason,
        "source": source,
        "created": _now_iso(now),
    }


class ReviewQueue:
    """append-only 的待审队列；enqueue 幂等去重，resolve 是唯一重写路径。"""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    # ---- 读 ----
    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    # 坏行不致命：跳过（队列是辅助清单，不该因一行脏数据炸掉）。
                    continue
        return out

    def _keys(self) -> set[str]:
        return {e.get("key") for e in self._load() if e.get("key")}

    def list_pending(self, kind: Optional[str] = None) -> list[dict]:
        items = [e for e in self._load() if e.get("status") == STATUS_PENDING]
        if kind:
            items = [e for e in items if e.get("kind") == kind]
        return items

    def all(self) -> list[dict]:
        return self._load()

    # ---- 写 ----
    def enqueue(self, entry: dict) -> bool:
        """挂一条 pending 行。已存在同 key 则不重复（幂等）。返回是否新增。"""
        key = entry.get("key")
        if not key:
            raise ValueError("review_queue entry 缺 key")
        if key in self._keys():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True

    def resolve(self, key: str, status: str, *, verdict_note: str = "",
                now: Optional[datetime] = None) -> bool:
        """人显式裁决一条：把它的 status 改成 reviewed/applied/rejected。
        唯一会重写文件的路径——机器永不调它。返回是否命中。"""
        if status not in (STATUS_REVIEWED, STATUS_APPLIED, STATUS_REJECTED):
            raise ValueError(f"非法 resolve 状态: {status}")
        rows = self._load()
        hit = False
        for r in rows:
            if r.get("key") == key and r.get("status") == STATUS_PENDING:
                r["status"] = status
                r["resolved_at"] = _now_iso(now)
                if verdict_note:
                    r["verdict_note"] = verdict_note
                hit = True
        if hit:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
        return hit


def render_md(items: list[dict], now: Optional[datetime] = None) -> str:
    """把 pending 清单渲染成给人/agent 看的只读报告。"""
    now = now or datetime.now()
    L = [f"# 海马体待审队列 · {now:%Y-%m-%d %H:%M}", ""]
    rels = [e for e in items if e.get("kind") == KIND_RELATION]
    zs = [e for e in items if e.get("kind") == KIND_Z_CONFLICT]

    L.append(f"## 🔶 关系闸 · 待审危险边（{len(rels)}）")
    L.append("> 机器自动推断的因果/取代类边，未写库，等人确认。")
    if not rels:
        L.append("- ✅ 无")
    else:
        for e in rels:
            sn = e.get("source_name") or e["source_id"]
            tn = e.get("target_name") or e["target_id"]
            note = f" —— {e['note']}" if e.get("note") else ""
            L.append(f"- `{e['key']}` {sn} --{e['rel_type']}--> {tn}{note}")
    L.append("")

    L.append(f"## ⚠️ Z轴 · 待审事实演化（{len(zs)}）")
    L.append("> 合并时检出的事实冲突（数字/日期/否定翻转），旧值已留 historical，挂此待人核。")
    if not zs:
        L.append("- ✅ 无")
    else:
        for e in zs:
            bn = e.get("bucket_name") or e["bucket_id"]
            L.append(f"- `{e['key']}` {bn} · {e['field']}: {e['old']} → {e['new']}")
    L.append("")
    L.append("---")
    L.append("*review_queue 只列待审，永不自动改库/建边/supersede——裁决只能由人显式 resolve。*")
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="海马体待审队列只读查看（永不改库）")
    default_dir = os.environ.get("OMBRE_BUCKETS_DIR", "/data/buckets")
    ap.add_argument("--path", default=os.path.join(default_dir, "review_queue.jsonl"),
                    help="队列文件（默认 <OMBRE_BUCKETS_DIR>/review_queue.jsonl）")
    ap.add_argument("--out", default=None, help="报告落点（默认打印 stdout）")
    args = ap.parse_args()

    q = ReviewQueue(args.path)
    md = render_md(q.list_pending())
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(md, encoding="utf-8")
        print(f"报告已写 → {outp}")
    else:
        print(md)


if __name__ == "__main__":
    main()
