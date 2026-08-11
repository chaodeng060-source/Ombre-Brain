#!/usr/bin/env python3
"""pending 审计队列 review_queue —— M 轴巡检、Z轴事实演化与关系闸共用的「待审」存储
（对位 lmc-5 的 z_conflict_audits + relation review-plan）。

设计铁律（直接抄 lmc-5，也守咱家 5.10/5.14 教训）：
  1. **机器只入队、不落库。** 自动推断出的「危险」边、合并时检出的事实冲突，
     都先挂成 pending 行给人过目，绝不静默改写真相
     （"keep fact changes reviewable instead of silently rewriting truth"）。
  2. **append-only + 入队去重。** 同一 (来源, 类型, 目标/字段值) 只挂一次，
     不刷屏；幂等，重复 enqueue 返回 False。
  3. **lifecycle 显式。** pending → reviewed/rejected 由人显式 resolve；
     applied 只能由带崩溃恢复日志的 Z 双桶事务落成。
  4. 队列本身不删任何桶、不动任何边——它只是一张「待人看」的清单。

存储：一行一个 JSON 对象的 .jsonl，落在 <buckets_dir>/review_queue.jsonl。

A pending-review queue shared by Z-axis fact evolution (#2) and relation safety
gating (#3). Machines only enqueue candidates here; the queue is append-only with
enqueue-dedup. Generic resolve() can acknowledge/reject; applied is reserved for
the paired lifecycle transaction.
(Z conflicts stay as separate facts until explicit approval.  Discovery defaults
to dry-run; apply only creates a pending row.)
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from maintenance_barrier import MaintenanceBarrier
from storage_safety import advisory_file_lock, atomic_write_text


# 队列里的条目类型（kind）：
KIND_RELATION = "relation"    # #3：机器自动推断的「危险」关系边（因果/取代类）
KIND_Z_CONFLICT = "z_conflict"  # #2：合并时检出的事实冲突（数字/日期/否定翻转）
KIND_METABOLISM = "metabolism"  # M：只读巡检建议，永不自动执行
KIND_E_PROPOSAL = "e_proposal"  # E：模型只提建议，主 AI 亲自写权威体验

METABOLISM_ACTIONS = frozenset({
    "promote",
    "demote",
    "split_thread",
    "mark_review",
    "archive",
})
METABOLISM_SEVERITIES = frozenset({"info", "warning", "critical"})

# 状态机：机器只写 pending，其余只能人显式 resolve。
STATUS_PENDING = "pending"
STATUS_REVIEWED = "reviewed"   # 人看过、判定保留候选（不应用）
STATUS_APPLIED = "applied"     # 人确认应用（真去建边 / 真去 supersede）
STATUS_REJECTED = "rejected"   # 人否决候选

# Generic REST/UI resolve may only acknowledge or reject a candidate.
# ``applied`` is reserved for the crash-recoverable paired lifecycle transaction;
# marking a queue row applied by itself is never sufficient.
REST_SAFE_RESOLVE_STATUSES = {STATUS_REVIEWED, STATUS_REJECTED}


class ReviewQueueCorruptError(RuntimeError):
    """The durable review ledger cannot be trusted and must not be treated as empty."""


def _maintenance_root(path: Path) -> Path:
    parent = path.parent
    root = parent.parent if parent.name.startswith(".") else parent
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    return root


def rest_resolve_status_allowed(status: str) -> bool:
    return str(status or "").strip() in REST_SAFE_RESOLVE_STATUSES


def lifecycle_updates(entry: dict) -> tuple[dict, dict]:
    """Build paired metadata updates for an approved cross-bucket candidate."""
    if entry.get("candidate_type") != "cross_bucket_lifecycle":
        raise ValueError("not a cross-bucket lifecycle candidate")
    current_id = str(entry.get("current_bucket_id") or "").strip()
    historical_id = str(entry.get("historical_bucket_id") or "").strip()
    fact_key = str(entry.get("fact_key") or "").strip().lower()
    if not current_id or not historical_id or not fact_key or current_id == historical_id:
        raise ValueError("invalid lifecycle candidate")
    return (
        {
            "fact_status": "current",
            "fact_key": fact_key,
            "supersedes_bucket_ids": [historical_id],
        },
        {
            "fact_status": "historical",
            "fact_key": fact_key,
            "superseded_by_bucket_id": current_id,
        },
    )


def historical_recall_suppressed(metadata: dict) -> bool:
    """Legacy helper for callers that already validated the fact-key registry.

    New recall code must additionally validate ``fact_key`` against the
    configured registry.  The old ``lifecycle/active_fact`` pair is not a
    second truth source.
    """
    if not isinstance(metadata, dict):
        return False
    return (
        str(metadata.get("fact_status") or "").strip().lower() == "historical"
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

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        maintenance_root: str | os.PathLike | None = None,
    ):
        self.path = Path(path)
        self._maintenance_barrier = MaintenanceBarrier(
            Path(maintenance_root)
            if maintenance_root is not None
            else _maintenance_root(self.path)
        )

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
        with self._maintenance_barrier.shared():
            return self._add_locked(entry)

    def _add_locked(self, entry: dict) -> bool:
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
    strength: float | None = None,
    now: Optional[datetime] = None,
) -> dict:
    """#3：一条等待人审的机器自动推断关系边。"""
    if strength is not None:
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise ValueError("relation strength must be a number in 0..1")
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("relation strength must be a number in 0..1")
    entry = {
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
    if strength is not None:
        entry["strength"] = strength
    return entry


def make_metabolism_entry(
    check: str,
    action: str,
    reason: str,
    *,
    severity: str = "info",
    bucket_ids: Optional[list[str]] = None,
    details: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Build one idempotent M-axis suggestion; it never applies the action."""
    check = str(check or "").strip()
    action = str(action or "").strip()
    severity = str(severity or "").strip()
    reason = str(reason or "").strip()
    if not check:
        raise ValueError("metabolism check is required")
    if action not in METABOLISM_ACTIONS:
        raise ValueError(f"invalid metabolism action: {action}")
    if severity not in METABOLISM_SEVERITIES:
        raise ValueError(f"invalid metabolism severity: {severity}")
    if not reason:
        raise ValueError("metabolism reason is required")
    normalized_ids = sorted({
        str(value).strip()
        for value in (bucket_ids or [])
        if str(value).strip()
    })
    return {
        "key": "m|" + _short_hash(check, action, ",".join(normalized_ids)),
        "kind": KIND_METABOLISM,
        "status": STATUS_PENDING,
        "check": check,
        "action": action,
        "severity": severity,
        "reason": reason,
        "bucket_ids": normalized_ids,
        "details": dict(details or {}),
        "source": "patrol_read_only",
        "created": _now_iso(now),
    }


def make_e_proposal_entry(
    source_bucket_id: str,
    candidate_type: str,
    title: str,
    evidence: str,
    *,
    suggested_priority: int,
    now: Optional[datetime] = None,
) -> dict:
    """Create a non-authoritative E proposal for the primary agent."""
    source_bucket_id = str(source_bucket_id or "").strip()
    candidate_type = str(candidate_type or "").strip()
    title = str(title or "").strip()
    evidence = str(evidence or "").strip()
    if not source_bucket_id or not candidate_type or not title or not evidence:
        raise ValueError("E proposal requires source, type, title and evidence")
    if type(suggested_priority) is not int or not 1 <= suggested_priority <= 100:
        raise ValueError("suggested_priority must be a plain integer in 1..100")
    return {
        "key": "e|" + _short_hash(source_bucket_id, candidate_type, evidence),
        "kind": KIND_E_PROPOSAL,
        "status": STATUS_PENDING,
        "source_bucket_id": source_bucket_id,
        "candidate_type": candidate_type,
        "title": title[:240],
        "evidence": evidence[:500],
        "suggested_priority": suggested_priority,
        "authority": "proposal_only",
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
    fact_key: str,
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
    fact_key = str(fact_key or "").strip().lower()
    if not fact_key:
        raise ValueError("fact_key is required")
    return {
        "key": "zpair|" + _short_hash(fact_key, current_bucket_id, historical_bucket_id),
        "kind": KIND_Z_CONFLICT,
        "status": STATUS_PENDING,
        "candidate_type": "cross_bucket_lifecycle",
        "fact_key": fact_key,
        "bucket_id": current_bucket_id,
        "bucket_name": current_name,
        "current_bucket_id": current_bucket_id,
        "historical_bucket_id": historical_bucket_id,
        "historical_bucket_name": historical_name,
        "field": "fact_status",
        "old": historical_name or "历史候选",
        "new": current_name or "当前候选",
        "reason": reason,
        "source": source,
        "created": _now_iso(now),
    }


class ReviewQueue:
    """append-only 的待审队列；enqueue 幂等去重，resolve 是唯一重写路径。"""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        maintenance_root: str | os.PathLike | None = None,
    ):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._maintenance_barrier = MaintenanceBarrier(
            Path(maintenance_root)
            if maintenance_root is not None
            else _maintenance_root(self.path)
        )

    # ---- 读 ----
    def _load_unlocked(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise ReviewQueueCorruptError(
                        f"review queue contains invalid JSON at line {line_number}"
                    ) from exc
                if not isinstance(row, dict) or not str(row.get("key") or "").strip():
                    raise ReviewQueueCorruptError(
                        f"review queue contains an invalid row at line {line_number}"
                    )
                out.append(row)
        return out

    def _load(self) -> list[dict]:
        with advisory_file_lock(self.lock_path):
            return self._load_unlocked()

    @staticmethod
    def _keys(rows: list[dict]) -> set[str]:
        return {e.get("key") for e in rows if e.get("key")}

    def list_pending(self, kind: Optional[str] = None) -> list[dict]:
        items = [e for e in self._load() if e.get("status") == STATUS_PENDING]
        if kind:
            items = [e for e in items if e.get("kind") == kind]
        return items

    def all(self) -> list[dict]:
        return self._load()

    def get(self, key: str) -> Optional[dict]:
        """Return one durable row by key without treating absence as pending."""
        key = str(key or "").strip()
        if not key:
            return None
        return next((entry for entry in self._load() if entry.get("key") == key), None)

    # ---- 写 ----
    def enqueue(self, entry: dict) -> bool:
        """挂一条 pending 行。已存在同 key 则不重复（幂等）。返回是否新增。"""
        with self._maintenance_barrier.shared():
            return self._enqueue_locked(entry)

    def _enqueue_locked(self, entry: dict) -> bool:
        key = entry.get("key")
        if not key:
            raise ValueError("review_queue entry 缺 key")
        with advisory_file_lock(self.lock_path):
            rows = self._load_unlocked()
            existing = next((row for row in rows if row.get("key") == key), None)
            if existing is not None:
                enriched = False
                if (
                    existing.get("status") == STATUS_PENDING
                    and existing.get("kind") == KIND_RELATION
                    and entry.get("kind") == KIND_RELATION
                ):
                    for field in ("source_name", "target_name"):
                        value = str(entry.get(field) or "").strip()
                        if value and not str(existing.get(field) or "").strip():
                            existing[field] = value[:160]
                            enriched = True
                if enriched:
                    atomic_write_text(
                        self.path,
                        "".join(
                            json.dumps(row, ensure_ascii=False) + "\n"
                            for row in rows
                        ),
                    )
                    os.chmod(self.path, 0o600)
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.chmod(self.path, 0o600)
            return True

    def resolve(self, key: str, status: str, *, verdict_note: str = "",
                reviewer: str = "", now: Optional[datetime] = None) -> bool:
        """Acknowledge or reject one row without applying memory changes."""
        if status == STATUS_APPLIED:
            raise ValueError(
                "applied is reserved for an explicit memory transaction"
            )
        with self._maintenance_barrier.shared():
            return self._resolve_locked(
                key,
                status,
                verdict_note=verdict_note,
                reviewer=reviewer,
                require_lifecycle=False,
                require_relation=False,
                now=now,
            )

    def apply_lifecycle(self, key: str, *, reviewer: str,
                        verdict_note: str = "",
                        now: Optional[datetime] = None) -> bool:
        """Mark a row applied from inside the paired Z transaction only."""
        reviewer = str(reviewer or "").strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        with self._maintenance_barrier.shared():
            return self._resolve_locked(
                key,
                STATUS_APPLIED,
                verdict_note=verdict_note,
                reviewer=reviewer,
                require_lifecycle=True,
                require_relation=False,
                now=now,
            )

    def apply_relation(self, key: str, *, reviewer: str,
                       verdict_note: str = "",
                       now: Optional[datetime] = None) -> bool:
        """Mark a relation row applied from its crash-recoverable transaction."""
        reviewer = str(reviewer or "").strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        with self._maintenance_barrier.shared():
            return self._resolve_locked(
                key,
                STATUS_APPLIED,
                verdict_note=verdict_note,
                reviewer=reviewer,
                require_lifecycle=False,
                require_relation=True,
                now=now,
            )

    def _resolve_locked(self, key: str, status: str, *, verdict_note: str = "",
                        reviewer: str = "", require_lifecycle: bool = False,
                        require_relation: bool = False,
                        now: Optional[datetime] = None) -> bool:
        if status not in (STATUS_REVIEWED, STATUS_APPLIED, STATUS_REJECTED):
            raise ValueError(f"非法 resolve 状态: {status}")
        if require_lifecycle and require_relation:
            raise ValueError("review transaction kind is ambiguous")
        with advisory_file_lock(self.lock_path):
            rows = self._load_unlocked()
            hit = False
            for r in rows:
                if r.get("key") == key and r.get("status") == STATUS_PENDING:
                    if (
                        require_lifecycle
                        and r.get("candidate_type") != "cross_bucket_lifecycle"
                    ):
                        raise ValueError(
                            "applied requires a cross-bucket lifecycle candidate"
                        )
                    if require_relation and r.get("kind") != KIND_RELATION:
                        raise ValueError(
                            "applied requires a relation review candidate"
                        )
                    r["status"] = status
                    r["resolved_at"] = _now_iso(now)
                    if verdict_note:
                        r["verdict_note"] = verdict_note
                    if reviewer:
                        r["reviewer"] = str(reviewer).strip()[:120]
                    hit = True
            if hit:
                atomic_write_text(
                    self.path,
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                )
                os.chmod(self.path, 0o600)
            return hit


def render_md(items: list[dict], now: Optional[datetime] = None) -> str:
    """把 pending 清单渲染成给人/agent 看的只读报告。"""
    now = now or datetime.now()
    L = [f"# 海马体待审队列 · {now:%Y-%m-%d %H:%M}", ""]
    rels = [e for e in items if e.get("kind") == KIND_RELATION]
    zs = [e for e in items if e.get("kind") == KIND_Z_CONFLICT]
    metabolism = [e for e in items if e.get("kind") == KIND_METABOLISM]
    e_proposals = [e for e in items if e.get("kind") == KIND_E_PROPOSAL]

    L.append(f"## 🩺 M轴 · 待审代谢建议（{len(metabolism)}）")
    L.append("> 巡检只提出建议，不改桶；晋升、降级、拆分或归档均需人显式执行。")
    if not metabolism:
        L.append("- ✅ 无")
    else:
        for entry in metabolism:
            bucket_ids = ", ".join(entry.get("bucket_ids") or []) or "全局"
            L.append(
                f"- `{entry['key']}` [{entry['severity']}] "
                f"`{entry['action']}` · {bucket_ids} —— {entry['reason']}"
            )
    L.append("")

    L.append(f"## 🫧 E轴 · 主 AI 待写体验提案（{len(e_proposals)}）")
    L.append("> 模型输出没有 E 权威；主 AI 需自己措辞并选择初始优先级。")
    if not e_proposals:
        L.append("- ✅ 无")
    else:
        for entry in e_proposals:
            L.append(
                f"- `{entry['key']}` {entry['title']} · "
                f"source={entry['source_bucket_id']} · "
                f"建议优先级 {entry['suggested_priority']} —— {entry['evidence']}"
            )
    L.append("")

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
    L.append("> 待审事实冲突尚未改变真值；只有人明确批准后，旧事实才会标 historical。")
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
