#!/usr/bin/env python3
"""只读巡检 patrol —— 海马体代谢健康报告（对位 lmc-5 metabolism patrol）

设计铁律：**只读，永不改库**。巡检只看一眼、出报告给人/agent 看，
任何"该不该拆/该不该并/该不该忘"的决定都留给人或专门的写入链路。
这条克制路径直接抄 lmc-5 的 read-only patrol，也守咱家 5.10 教训
（一个 CC self 自作主张 resolve 了 13 个桶）。

用法：
    # 本地拿备份副本看（最安全）
    python patrol.py --buckets /c/Users/HP/Ombre-Brain-backups/2026-06-15_0111/buckets

    # NAS 活库上看（只读，cron-able）
    python patrol.py --config /app/config.yaml

    # 把报告落到文件
    python patrol.py --buckets <dir> --out notes/patrol_2026-06-15.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import frontmatter
import yaml

from conversation_activity import load_activity_summary, validate_activity_summary
from fact_slots import audit_fact_slots
from fact_conflicts import scan_cross_bucket_z_conflicts
from review_queue import (
    ReviewQueue,
    make_metabolism_entry,
)
from utils import RELATION_TYPES

# 与 utils.PROTECTED_RESOLVE_DOMAINS 保持一致（resolve=遗忘的禁区）。
# 这里硬编一份副本，让 patrol 不依赖 server 运行时即可独立巡检。
PROTECTED_RESOLVE_DOMAINS = frozenset({"恋爱", "纪念日", "约定", "家庭", "自省", "feel"})

# 报告阈值（保守，宁可漏报不误报；全部可调）
OVERSIZED_CHARS = 1500          # content 超此长度 → 拆线候选（只提示）
STALE_DAYS = 90                 # 高重要度桶超此天数没激活 → 提示（绝不自动忘）
STALE_IMPORTANCE = 7            # 仅对 importance>=此值的桶报陈旧（重要的才值得提醒）
AMNESIA_MONITOR_START_DATE = date(2026, 7, 30)
AMNESIA_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _load_patrol_config(path: str | os.PathLike) -> dict:
    """Read only the patrol config; unlike server load_config, create nothing."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("patrol config must be a YAML mapping")
    return raw


def _sample_ids(items: list[dict], *keys: str, limit: int = 30) -> list[str]:
    values: list[str] = []
    for item in items:
        for key in keys:
            value = str(item.get(key) or "").strip()
            if value and value not in values:
                values.append(value)
                if len(values) >= limit:
                    return values
    return values


def build_metabolism_suggestions(report: dict, now: datetime) -> list[dict]:
    """Convert read-only patrol findings into reviewable, reasoned M entries."""
    suggestions: list[dict] = []

    def add(
        check: str,
        action: str,
        severity: str,
        items: list[dict],
        reason: str,
        *id_keys: str,
    ) -> None:
        if not items:
            return
        suggestions.append(make_metabolism_entry(
            check,
            action,
            reason,
            severity=severity,
            bucket_ids=_sample_ids(items, *id_keys),
            details={"count": len(items)},
            now=now,
        ))

    add(
        "broken_bucket_files",
        "mark_review",
        "critical",
        report["broken"],
        f"{len(report['broken'])} 个桶文件无法解析，需人工修复后再参与代谢。",
        "__broken__",
    )
    add(
        "protected_resolved",
        "mark_review",
        "critical",
        report["protected_resolved"],
        f"{len(report['protected_resolved'])} 个保护域桶被标记 resolved；禁止自动降级或归档。",
        "id",
    )
    add(
        "dangling_relations",
        "mark_review",
        "critical",
        report["dangling"],
        f"{len(report['dangling'])} 条关系指向不存在的桶，需人工核对来源与目标。",
        "from",
        "target",
    )
    add(
        "relation_self_loops",
        "mark_review",
        "critical",
        report["self_loops"],
        f"{len(report['self_loops'])} 条关系自环无有效语义，需人工确认后清理。",
        "from",
    )
    add(
        "duplicate_relation_edges",
        "mark_review",
        "warning",
        report["duplicate_edges"],
        f"{len(report['duplicate_edges'])} 条关系边重复，需人工确认保留项。",
        "from",
        "target",
    )
    add(
        "reciprocal_kin_edges",
        "mark_review",
        "warning",
        report["reciprocal_kin"],
        f"{len(report['reciprocal_kin'])} 组 kin 双向重复存储，需人工确认保留一条。",
        "from",
        "target",
    )
    add(
        "invalid_relation_types",
        "mark_review",
        "warning",
        report["invalid_relation_types"],
        f"{len(report['invalid_relation_types'])} 条关系使用未知类型，不能自动迁移。",
        "from",
        "target",
    )
    add(
        "invalid_relation_strengths",
        "mark_review",
        "warning",
        report["invalid_relation_strengths"],
        f"{len(report['invalid_relation_strengths'])} 条关系强度不在 0..1，需人工校正。",
        "from",
        "target",
    )
    add(
        "oversized_buckets",
        "split_thread",
        "info",
        report["oversized"],
        f"{len(report['oversized'])} 个桶正文超过 {OVERSIZED_CHARS} 字，仅建议人工拆分。",
        "id",
    )

    duplicate_rows = [
        {"name": name, "id": bucket_id}
        for name, bucket_ids in report["duplicates"].items()
        for bucket_id in bucket_ids
    ]
    add(
        "duplicate_bucket_names",
        "mark_review",
        "warning",
        duplicate_rows,
        f"{len(report['duplicates'])} 组桶重名，需人工判断是否同一事件。",
        "id",
    )

    fact_rows = [
        {"fact_key": fact_key, "id": bucket_id}
        for fact_key, bucket_ids in report["fact_conflicts"].items()
        for bucket_id in bucket_ids
    ]
    add(
        "duplicate_current_fact_slots",
        "mark_review",
        "critical",
        fact_rows,
        f"{len(report['fact_conflicts'])} 个 fact_key 同时存在多个 current 值，需人工裁决。",
        "id",
    )
    add(
        "fact_slot_migration_candidates",
        "mark_review",
        "info",
        report["migration_candidates"],
        f"{len(report['migration_candidates'])} 个桶可迁移到明确 fact_key，未自动写入。",
        "id",
    )
    add(
        "ambiguous_fact_slot_candidates",
        "mark_review",
        "warning",
        report["ambiguous_candidates"],
        f"{len(report['ambiguous_candidates'])} 个桶同时命中多个 fact_key，需人工拆分。",
        "id",
    )
    add(
        "invalid_fact_keys",
        "mark_review",
        "warning",
        report["invalid_fact_keys"],
        f"{len(report['invalid_fact_keys'])} 个 fact_key 未登记，禁止自动晋升为当前事实。",
        "id",
    )
    add(
        "invalid_fact_statuses",
        "mark_review",
        "warning",
        report["invalid_fact_statuses"],
        f"{len(report['invalid_fact_statuses'])} 个 fact_status 非法，需人工修正。",
        "id",
    )
    add(
        "legacy_active_fact",
        "mark_review",
        "warning",
        report["legacy_active_fact"],
        f"{len(report['legacy_active_fact'])} 个桶仍使用遗留 active_fact 字段，未自动迁移。",
        "id",
    )
    add(
        "stale_important",
        "mark_review",
        "info",
        report["stale_important"],
        (
            f"{len(report['stale_important'])} 个高重要度桶超过 {STALE_DAYS} 天未激活；"
            "只建议复核，绝不自动归档。"
        ),
        "id",
    )
    add(
        "curated_without_vector",
        "mark_review",
        "critical",
        report.get("curated_without_vector", []),
        (
            f"{len(report.get('curated_without_vector', []))} 个有正文的桶缺少可用向量；"
            "仅报告，等待人工裁决是否补写。"
        ),
        "id",
    )
    zero_deposition = report.get("zero_deposition", {})
    if zero_deposition.get("status") == "alert":
        suggestions.append(make_metabolism_entry(
            "zero_deposition_with_activity",
            "mark_review",
            (
                f"连续 {zero_deposition['streak_days']} 个应沉淀日有对话但新增桶为 0；"
                "可能存在静默写入断档，仅报告等待人工排查。"
            ),
            severity="critical",
            details={
                "start_date": zero_deposition.get("streak_start"),
                "end_date": zero_deposition.get("streak_end"),
                "streak_days": zero_deposition.get("streak_days"),
            },
            now=now,
        ))
    return suggestions


def enqueue_metabolism_suggestions(report: dict, queue: ReviewQueue) -> int:
    """Append new pending M suggestions; bucket contents remain untouched."""
    added = 0
    for entry in report.get("suggestions", []):
        if queue.enqueue(entry):
            added += 1
    return added


def _safe_frontmatter(path: Path):
    """读 .md 桶 frontmatter，容忍 YAML 头里混入 'content' 键的脏数据。
    对齐 bucket_manager._safe_load_post —— 让 patrol 不依赖 server 运行时
    即可独立解析真实桶（避免把已知可容忍的脏头误报成坏文件）。"""
    try:
        return frontmatter.load(str(path))
    except TypeError as e:
        if "content" not in str(e):
            raise
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise
        end = text.find("\n---\n", 4)
        if end < 0:
            raise
        yaml_part, body = text[4:end], text[end + 5:]
        cleaned, skip = [], False
        for line in yaml_part.splitlines(keepends=True):
            if skip:
                if line and line[0] in " \t":
                    continue
                skip = False
            if line.startswith("content:"):
                skip = True
                continue
            cleaned.append(line)
        cleaned_yaml = "".join(cleaned)
        # closing --- 必须独占一行：若保留的末行无尾换行（yaml_part 本就不带
        # 尾换行），重组会把它和 --- 黏成一行 → YAML 头解析失败、metadata 整段
        # 被吞进 body=静默丢 id/domain/relations/resolved，巡检假干净。
        if cleaned_yaml and not cleaned_yaml.endswith("\n"):
            cleaned_yaml += "\n"
        return frontmatter.loads("---\n" + cleaned_yaml + "---\n" + body)


def _load_buckets(buckets_dir: Path) -> list[dict]:
    # 真桶是 .md + YAML frontmatter，散在 permanent/dynamic/feel/archive/...
    # 及世界线子目录下（对齐 bucket_manager.list_all 的 os.walk）。
    # 递归扫，把保护域被 resolve 的桶藏在子目录里也能巡到。
    out = []
    for p in sorted(buckets_dir.rglob("*.md")):
        try:
            post = _safe_frontmatter(p)
            meta = dict(post.metadata)
            out.append({
                "id": meta.get("id", p.stem),
                "metadata": meta,
                "content": post.content,
            })
        except Exception as e:  # 坏文件也是巡检要报的
            out.append({"__broken__": str(p.name), "__error__": str(e)})

    # 本地/NAS 备份工具会把每个 Markdown 桶序列化成 <12hex>.json 快照。
    # patrol 同时读取这种只读备份格式；body_state.json 等运行时 sidecar
    # 没有 bucket schema，直接忽略，避免把它们误报成坏桶。
    for p in sorted(buckets_dir.rglob("*.json")):
        looks_like_snapshot = bool(re.fullmatch(r"[0-9a-fA-F]{12}", p.stem))
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            if looks_like_snapshot:
                out.append({"__broken__": str(p.name), "__error__": str(e)})
            continue
        if not isinstance(data, dict):
            continue
        meta = data.get("metadata")
        if not isinstance(meta, dict) or "content" not in data:
            if looks_like_snapshot:
                out.append({"__broken__": str(p.name), "__error__": "invalid bucket snapshot schema"})
            continue
        out.append({
            "id": data.get("id") or meta.get("id") or p.stem,
            "metadata": dict(meta),
            "content": data.get("content", "") or "",
        })
    return out


def _parse_dt(s) -> datetime | None:
    if not s:
        return None
    # frontmatter/YAML 会把未加引号的 ISO 时间直接读成 datetime/date 对象
    # （只有带引号的才是 str）；两种都要吃，否则非字符串输入悄悄返回 None
    # → stale_important 检查被跳过=漏检。
    if isinstance(s, datetime):
        return s.replace(tzinfo=None)
    try:
        # 真桶的时间戳带时区（如 +08:00），now=datetime.now() 是 naive；
        # 统一剥掉 tzinfo 归一成 naive，避免 aware/naive 相减崩溃。
        return datetime.fromisoformat(str(s).replace("Z", "")).replace(tzinfo=None)
    except Exception:
        return None


def _bucket_recorded_at(metadata: dict) -> datetime | None:
    """Return the body-write timestamp used by incremental health scans."""
    return _parse_dt(
        metadata.get("recorded_at")
        or metadata.get("created_at")
        or metadata.get("created")
    )


def _local_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if len(text) < 10 or text[4:5] != "-" or text[7:8] != "-":
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=AMNESIA_TIMEZONE)
    return parsed.astimezone(AMNESIA_TIMEZONE)


def _bucket_created_day(metadata: dict) -> date | None:
    recorded = (
        metadata.get("recorded_at")
        or metadata.get("recordedAt")
        or metadata.get("created_at")
        or metadata.get("created")
    )
    parsed = _local_datetime(recorded)
    return parsed.date() if parsed else None


def _zero_deposition_report(
    buckets: list[dict],
    now: datetime,
    activity_summary: dict | None,
    *,
    monitor_start_date: date,
) -> dict:
    """Compare active chat days with bucket creation days, without writes."""
    note = (
        "第一刀将纯寒暄也计为活跃；若连续三天只有无需沉淀的寒暄，可能敏感误报，"
        "但不会自动修复或写库。"
    )
    if activity_summary is None:
        return {
            "status": "not_configured",
            "monitor_start_date": monitor_start_date.isoformat(),
            "streak_days": 0,
            "streak_start": None,
            "streak_end": None,
            "days": [],
            "note": note,
        }

    summary = validate_activity_summary(activity_summary)
    summary_start = date.fromisoformat(summary["start_date"])
    start_date = max(monitor_start_date, summary_start)
    local_now = (
        now.replace(tzinfo=AMNESIA_TIMEZONE)
        if now.tzinfo is None
        else now.astimezone(AMNESIA_TIMEZONE)
    )
    today = local_now.date()

    bucket_counts: Counter[str] = Counter()
    for bucket in buckets:
        day = _bucket_created_day(bucket.get("metadata", {}))
        if day is not None and start_date <= day <= today:
            bucket_counts[day.isoformat()] += 1

    activity_counts = summary["daily_user_messages"]
    rows: list[dict] = []
    streak = 0
    cursor = start_date
    while cursor <= today:
        day_key = cursor.isoformat()
        user_messages = int(activity_counts.get(day_key, 0))
        new_buckets = int(bucket_counts.get(day_key, 0))
        expected = user_messages > 0
        silent = expected and new_buckets == 0
        streak = streak + 1 if silent else 0
        rows.append({
            "date": day_key,
            "user_messages": user_messages,
            "new_buckets": new_buckets,
            "expected_deposition": expected,
            "silent": silent,
        })
        cursor += timedelta(days=1)

    if len(rows) < 3:
        status = "insufficient_window"
    else:
        status = "alert" if streak >= 3 else "healthy"
    streak_start = (
        (today - timedelta(days=streak - 1)).isoformat()
        if streak
        else None
    )
    return {
        "status": status,
        "monitor_start_date": start_date.isoformat(),
        "streak_days": streak,
        "streak_start": streak_start,
        "streak_end": today.isoformat() if streak else None,
        "days": rows[-14:],
        "note": note,
    }


def _read_vector_index(embeddings_path: Path) -> dict:
    """Read vector presence and zero-dimensional rows without mutating SQLite.

    Normal vectors can be large, so the audit reads every bucket id but only
    loads payloads short enough to represent an empty/zero-dimensional value.
    """
    result = {
        "status": "ok",
        "path": str(embeddings_path),
        "vector_rows": 0,
        "bucket_ids": set(),
        "zero_dimension": set(),
        "error": "",
    }
    if not embeddings_path.is_file():
        result.update(status="unavailable", error="embeddings.db not found")
        return result

    try:
        uri = embeddings_path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("PRAGMA query_only = ON")
            bucket_ids = {
                str(row[0])
                for row in conn.execute("SELECT bucket_id FROM embeddings")
                if row[0]
            }
            suspicious = conn.execute(
                """
                SELECT bucket_id, embedding
                FROM embeddings
                WHERE embedding IS NULL OR length(trim(embedding)) <= 32
                """
            ).fetchall()
        zero_dimension: set[str] = set()
        for bucket_id, raw in suspicious:
            try:
                decoded = json.loads(raw) if raw is not None else None
            except (TypeError, json.JSONDecodeError):
                continue
            if decoded == [] or (
                isinstance(decoded, list)
                and decoded
                and all(isinstance(segment, list) and not segment for segment in decoded)
            ):
                zero_dimension.add(str(bucket_id))
        result.update(
            vector_rows=len(bucket_ids),
            bucket_ids=bucket_ids,
            zero_dimension=zero_dimension,
        )
    except (OSError, sqlite3.Error) as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


def _scan_curated_without_vector(
    buckets: list[dict],
    embeddings_path: Path,
    *,
    since: datetime | None = None,
) -> tuple[list[dict], dict]:
    """List bodies whose required/legacy vector record is absent or zero-dim."""
    vector_index = _read_vector_index(embeddings_path)
    audit = {
        key: value
        for key, value in vector_index.items()
        if key not in {"bucket_ids", "zero_dimension"}
    }
    audit["since"] = since.isoformat() if since else None
    audit["scanned_bodies"] = 0
    if vector_index["status"] != "ok":
        return [], audit

    findings: list[dict] = []
    indexed_ids = vector_index["bucket_ids"]
    zero_dimension = vector_index["zero_dimension"]
    for bucket in buckets:
        metadata = bucket.get("metadata", {})
        content = str(bucket.get("content") or "")
        bucket_id = str(bucket.get("id") or metadata.get("id") or "").strip()
        if not bucket_id or not content.strip():
            continue
        # fts_only is an explicit promise that no vector should exist.
        if metadata.get("vector_policy") == "fts_only":
            continue
        recorded_at = _bucket_recorded_at(metadata)
        if since is not None and recorded_at is not None and recorded_at < since:
            continue
        audit["scanned_bodies"] += 1
        if bucket_id in zero_dimension:
            reason = "zero_dimension"
        elif bucket_id not in indexed_ids:
            reason = "missing_vector"
        else:
            continue
        findings.append({
            "id": bucket_id,
            "name": metadata.get("name", "(无名)"),
            "reason": reason,
            "recorded_at": recorded_at.isoformat() if recorded_at else None,
            "vector_policy": metadata.get("vector_policy") or "legacy",
        })
    findings.sort(key=lambda item: (item["reason"], item["id"]))
    audit["findings"] = len(findings)
    return findings, audit


def patrol(
    buckets_dir: Path,
    now: datetime,
    fact_slot_registry: dict | None = None,
    *,
    embeddings_path: Path | None = None,
    vector_since: datetime | None = None,
    activity_summary: dict | None = None,
    monitor_start_date: date = AMNESIA_MONITOR_START_DATE,
) -> dict:
    raw = _load_buckets(buckets_dir)
    broken = [b for b in raw if b.get("__broken__")]
    buckets = [b for b in raw if not b.get("__broken__")]
    curated_without_vector, vector_audit = _scan_curated_without_vector(
        buckets,
        embeddings_path or (buckets_dir / "embeddings.db"),
        since=vector_since,
    )
    zero_deposition = _zero_deposition_report(
        buckets,
        now,
        activity_summary,
        monitor_start_date=monitor_start_date,
    )

    ids = set()
    for b in buckets:
        bid = b.get("id") or b.get("metadata", {}).get("id")
        if bid:
            ids.add(bid)

    by_type: Counter = Counter()
    by_domain: Counter = Counter()
    dangling: list[dict] = []          # 关系指向不存在的桶（断边）
    non_reciprocal: list[dict] = []    # A→B 有边、B→A 没有（信息性，不一定是病）
    self_loops: list[dict] = []        # 自环永远无效
    duplicate_edges: list[dict] = []   # 同 source/type/target 重复
    reciprocal_kin: list[dict] = []    # kin 是对称关系，反向重复存储是脏边
    invalid_relation_types: list[dict] = []
    invalid_relation_strengths: list[dict] = []
    oversized: list[dict] = []         # content 过长 → 拆线候选
    name_index: defaultdict = defaultdict(list)  # 重名 → 重复候选
    protected_resolved: list[dict] = []          # 保护域被 resolve（5.10 守卫验证）
    stale_important: list[dict] = []   # 重要但久未激活（只提示）

    # 先建反向关系索引，判互惠
    fwd_edges: defaultdict = defaultdict(set)
    typed_edges: set[tuple[str, str, str]] = set()
    for b in buckets:
        meta = b.get("metadata", {})
        bid = b.get("id") or meta.get("id")
        for rel in meta.get("relations", []) or []:
            if not isinstance(rel, dict):
                continue
            tgt = rel.get("target")
            rel_type = rel.get("type")
            if tgt:
                fwd_edges[bid].add(tgt)
                typed_edges.add((str(bid), str(tgt), str(rel_type or "")))

    for b in buckets:
        meta = b.get("metadata", {})
        bid = b.get("id") or meta.get("id")
        name = meta.get("name", "(无名)")
        content = b.get("content", "") or ""
        domains = meta.get("domain", []) or []
        if isinstance(domains, str):
            domains = [domains]

        by_type[meta.get("type", "?")] += 1
        for d in domains:
            by_domain[d] += 1

        name_index[name].append(bid)

        # 断边 / 互惠
        seen_edges: set[tuple[str, str]] = set()
        for rel in meta.get("relations", []) or []:
            if not isinstance(rel, dict):
                invalid_relation_types.append({"from": bid, "target": "", "type": type(rel).__name__})
                continue
            tgt = rel.get("target")
            rel_type = str(rel.get("type") or "")
            edge_key = (rel_type, str(tgt or ""))
            if edge_key in seen_edges:
                duplicate_edges.append({"from": bid, "target": tgt, "type": rel_type})
            seen_edges.add(edge_key)
            if tgt == bid:
                self_loops.append({"from": bid, "target": tgt, "type": rel_type})
            if rel_type not in RELATION_TYPES:
                invalid_relation_types.append({"from": bid, "target": tgt, "type": rel_type})
            if "strength" in rel:
                try:
                    strength = float(rel["strength"])
                    valid_strength = 0.0 <= strength <= 1.0
                except (TypeError, ValueError):
                    valid_strength = False
                if not valid_strength:
                    invalid_relation_strengths.append({
                        "from": bid, "target": tgt, "type": rel_type, "strength": rel.get("strength")
                    })
            if tgt and tgt not in ids:
                dangling.append({"from": bid, "name": name, "target": tgt, "type": rel_type})
            elif tgt and bid not in fwd_edges.get(tgt, set()):
                non_reciprocal.append({"from": bid, "target": tgt, "type": rel_type})

        # 拆线候选
        if len(content) > OVERSIZED_CHARS:
            oversized.append({"id": bid, "name": name, "chars": len(content)})

        # 保护域被 resolve（绝不该出现）
        if meta.get("resolved") and any(d in PROTECTED_RESOLVE_DOMAINS for d in domains):
            protected_resolved.append({"id": bid, "name": name, "domains": domains})

        # 陈旧但重要（只提示）
        imp = meta.get("importance", 0) or 0
        la = _parse_dt(meta.get("last_active"))
        if imp >= STALE_IMPORTANCE and la and (now - la).days >= STALE_DAYS:
            stale_important.append({"id": bid, "name": name, "importance": imp,
                                    "days": (now - la).days})

    duplicates = {n: ids_ for n, ids_ in name_index.items() if len(ids_) > 1}
    fact_report = audit_fact_slots(buckets, fact_slot_registry or {})
    z_conflicts = scan_cross_bucket_z_conflicts(buckets)
    for source_id, target_id, rel_type in sorted(typed_edges):
        if rel_type == "kin" and (target_id, source_id, rel_type) in typed_edges and source_id < target_id:
            reciprocal_kin.append({"from": source_id, "target": target_id, "type": rel_type})

    report = {
        "total": len(buckets),
        "broken": broken,
        "by_type": dict(by_type.most_common()),
        "by_domain": dict(by_domain.most_common(12)),
        "dangling": dangling,
        "non_reciprocal": non_reciprocal,
        "self_loops": self_loops,
        "duplicate_edges": duplicate_edges,
        "reciprocal_kin": reciprocal_kin,
        "invalid_relation_types": invalid_relation_types,
        "invalid_relation_strengths": invalid_relation_strengths,
        "oversized": sorted(oversized, key=lambda x: -x["chars"])[:15],
        "duplicates": duplicates,
        **fact_report,
        "z_conflicts": z_conflicts,
        "protected_resolved": protected_resolved,
        "stale_important": sorted(stale_important, key=lambda x: -x["days"])[:20],
        "curated_without_vector": curated_without_vector,
        "vector_audit": vector_audit,
        "zero_deposition": zero_deposition,
    }
    report["suggestions"] = build_metabolism_suggestions(report, now)
    return report


def render_md(report: dict, buckets_dir: Path, now: datetime) -> str:
    L = []
    L.append(f"# 海马体只读巡检 · {now:%Y-%m-%d %H:%M}")
    L.append("")
    L.append(f"> 来源：`{buckets_dir}` · **只读，未改任何桶**")
    L.append("")
    L.append(f"- 桶总数：**{report['total']}**")
    if report["broken"]:
        L.append(f"- ⚠️ 坏文件：**{len(report['broken'])}** 个 —— {[b['__broken__'] for b in report['broken']]}")
    L.append(f"- 按类型：{report['by_type']}")
    L.append(f"- 按 domain（Top12）：{report['by_domain']}")
    L.append("")

    L.append(f"## 📋 M轴待审建议（{len(report['suggestions'])}）")
    if not report["suggestions"]:
        L.append("- ✅ 无")
    else:
        for entry in report["suggestions"]:
            bucket_ids = ", ".join(entry["bucket_ids"]) or "全局"
            L.append(
                f"- [{entry['severity']}] `{entry['action']}` · {bucket_ids}"
                f" —— {entry['reason']}"
            )
    L.append("")

    def section(title, items, fmt, empty="无"):
        L.append(f"## {title}（{len(items)}）")
        if not items:
            L.append(f"- ✅ {empty}")
        else:
            for it in items[:30]:
                L.append(f"- {fmt(it)}")
        L.append("")

    def fmt_z_conflict(item):
        fields = ", ".join(
            f"{c['field']}: {c['old']} → {c['new']}"
            for c in item.get("fields", [])
        )
        return (
            f"`{item['left_id']}` {item['left_name']} ↔ "
            f"`{item['right_id']}` {item['right_name']} —— {fields}"
        )

    section("🔴 保护域被 resolve（5.10 守卫·必须为 0）", report["protected_resolved"],
            lambda x: f"`{x['id']}` {x['name']} —— domains={x['domains']}",
            empty="守卫完好，无保护域被遗忘")
    section("🔗 断边（关系指向不存在的桶）", report["dangling"],
            lambda x: f"`{x['from']}` ({x['name']}) --{x['type']}--> `{x['target']}` ❌不存在")
    section("⛔ 关系自环（必须为 0）", report["self_loops"],
            lambda x: f"`{x['from']}` --{x['type']}--> 自己")
    section("♻️ 重复关系边", report["duplicate_edges"],
            lambda x: f"`{x['from']}` --{x['type']}--> `{x['target']}` 重复")
    section("↔️ kin 双向重复存储", report["reciprocal_kin"],
            lambda x: f"`{x['from']}` ↔ `{x['target']}`（只需存一条）")
    section("❓ 未知关系类型", report["invalid_relation_types"],
            lambda x: f"`{x['from']}` --{x['type']}--> `{x['target']}`")
    section("📏 非法关系强度", report["invalid_relation_strengths"],
            lambda x: f"`{x['from']}` --{x['type']}--> `{x['target']}` strength={x['strength']}")
    section("✂️ 拆线候选（content 过长，仅提示）", report["oversized"],
            lambda x: f"`{x['id']}` {x['name']} —— {x['chars']} 字")
    dups = report["duplicates"]
    L.append(f"## ♊ 重名候选（{len(dups)}）")
    if not dups:
        L.append("- ✅ 无重名")
    else:
        for n, ids_ in list(dups.items())[:20]:
            L.append(f"- 「{n}」×{len(ids_)}：{ids_}")
    L.append("")
    fact_conflicts = report["fact_conflicts"]
    L.append(f"## 🧾 重复当前事实槽（{len(fact_conflicts)}）")
    if not fact_conflicts:
        L.append("- ✅ 无重复 current fact_key")
    else:
        for fact_key, bucket_ids in list(fact_conflicts.items())[:20]:
            L.append(f"- `{fact_key}`：{bucket_ids}")
    L.append("")
    section("🧭 事实槽迁移候选（只读建议）", report["migration_candidates"],
            lambda x: f"`{x['id']}` → `{x['fact_key']}` values={x['values']}")
    section("⚠️ 多槽迁移候选（需人工拆分）", report["ambiguous_candidates"],
            lambda x: f"`{x['id']}` → {x['fact_keys']}")
    section("❓ 未登记 fact_key", report["invalid_fact_keys"],
            lambda x: f"`{x['id']}` fact_key=`{x['fact_key']}`")
    section("❓ 非法 fact_status", report["invalid_fact_statuses"],
            lambda x: f"`{x['id']}` `{x['fact_key']}` status=`{x['status']}`")
    section("🧹 遗留 active_fact 字段", report["legacy_active_fact"],
            lambda x: f"`{x['id']}` active_fact={x['value']}（只报告，不作为真值）")
    section("🛡️ 已排除的保护/叙事事实元数据", report["exempt_fact_metadata"],
            lambda x: f"`{x['id']}` fact_key=`{x['fact_key']}`")
    section("⚠️ Z轴跨桶事实冲突候选（只报告，不入队、不改库）", report.get("z_conflicts", []),
            fmt_z_conflict,
            empty="未发现同名/同域跨桶事实冲突候选")
    zero_deposition = report.get("zero_deposition", {})
    L.append("## 🚨 连续零沉淀监控（对照真实对话活跃）")
    status = zero_deposition.get("status", "not_configured")
    if status == "not_configured":
        L.append("- ⚪ 未提供 VPS 活跃日计数，本轮不作零沉淀判断。")
    else:
        L.append(
            f"- 状态：**{status}** · 起算："
            f"`{zero_deposition.get('monitor_start_date')}` · "
            f"当前连续静默：**{zero_deposition.get('streak_days', 0)} 天**"
        )
        for row in zero_deposition.get("days", [])[-7:]:
            L.append(
                f"- `{row['date']}`：user={row['user_messages']} · "
                f"new_buckets={row['new_buckets']} · "
                f"{'⚠️ 应沉淀但为零' if row['silent'] else '正常/无需沉淀'}"
            )
    L.append(f"- 边界备注：{zero_deposition.get('note', '')}")
    L.append("")
    vector_audit = report.get("vector_audit", {})
    if vector_audit.get("status") == "ok":
        section(
            "🧠 正文有而向量缺失/零维（curated_without_vector）",
            report.get("curated_without_vector", []),
            lambda x: (
                f"`{x['id']}` {x['name']} —— {x['reason']}, "
                f"policy={x['vector_policy']}, recorded_at={x['recorded_at'] or '?'}"
            ),
            empty="已扫描正文均有可用向量（显式 fts_only 不计）",
        )
    else:
        L.append("## 🧠 正文/向量对账（未完成）")
        L.append(
            f"- ⚠️ vector store {vector_audit.get('status', 'unknown')}: "
            f"{vector_audit.get('error') or 'unknown error'}"
        )
        L.append("")
    section("🕰️ 陈旧但重要（importance≥{}, >{}天未激活·只提示绝不自动忘）".format(STALE_IMPORTANCE, STALE_DAYS),
            report["stale_important"],
            lambda x: f"`{x['id']}` {x['name']} —— imp={x['importance']}, {x['days']}天")
    L.append(f"## ↔️ 非互惠关系边（{len(report['non_reciprocal'])}，信息性，不一定是病）")
    L.append(f"- 共 {len(report['non_reciprocal'])} 条单向边（A→B 有、B→A 无）。多数正常（updates/causes 本就有方向）。")
    L.append("")
    L.append("---")
    L.append("*patrol 只读巡检 —— 任何拆/并/忘都需人或专门写入链路决定，patrol 永不动手。*")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="海马体只读巡检（永不改库）")
    ap.add_argument(
        "--buckets",
        default=None,
        help="桶目录（优先级：显式参数 > config.buckets_dir > $OMBRE_BUCKETS_DIR > /data）",
    )
    ap.add_argument("--out", default=None, help="报告落点（默认打印到 stdout）")
    ap.add_argument("--now", default=None, help="覆盖当前时间（ISO，便于测试）")
    ap.add_argument(
        "--embeddings",
        default=None,
        help="向量 SQLite（默认 <buckets>/embeddings.db；严格只读打开）",
    )
    ap.add_argument(
        "--vector-since",
        default=None,
        help="只对账该写入时间之后的桶（ISO；首次基线不传即全量）",
    )
    ap.add_argument(
        "--activity-summary",
        default=None,
        help="VPS 只读汇总的 date/count JSON；'-' 表示 stdin，不含消息正文",
    )
    ap.add_argument(
        "--amnesia-start-date",
        default=AMNESIA_MONITOR_START_DATE.isoformat(),
        help="零沉淀监控起算日（默认 2026-07-30，不回溯历史告警）",
    )
    ap.add_argument("--config", default=os.environ.get("OMBRE_CONFIG"),
                    help="可选 config.yaml，用于读取 fact_slots.registry")
    ap.add_argument("--review-queue", default=None,
                    help="可选待审队列路径；只入 pending 建议，不修改任何桶")
    ap.add_argument("--apply", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.apply:
        raise SystemExit("M 巡检严格只读：不支持 --apply")

    cfg = {}
    registry = {}
    if args.config:
        cfg = _load_patrol_config(args.config)
        registry = ((cfg.get("fact_slots", {}) or {}).get("registry", {}) or {})
    buckets_dir = Path(
        args.buckets
        or cfg.get("buckets_dir")
        or os.environ.get("OMBRE_BUCKETS_DIR")
        or "/data"
    )
    if not buckets_dir.is_dir():
        raise SystemExit(f"桶目录不存在：{buckets_dir}")
    now = _parse_dt(args.now) or datetime.now()

    vector_since = _parse_dt(args.vector_since)
    if args.vector_since and vector_since is None:
        raise SystemExit(f"非法 --vector-since：{args.vector_since}")
    try:
        monitor_start_date = date.fromisoformat(args.amnesia_start_date)
    except ValueError as exc:
        raise SystemExit(
            f"非法 --amnesia-start-date：{args.amnesia_start_date}"
        ) from exc
    try:
        activity_summary = (
            load_activity_summary(args.activity_summary)
            if args.activity_summary
            else None
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"无法读取活跃锚：{exc}") from exc
    report = patrol(
        buckets_dir,
        now,
        fact_slot_registry=registry,
        embeddings_path=Path(args.embeddings) if args.embeddings else None,
        vector_since=vector_since,
        activity_summary=activity_summary,
        monitor_start_date=monitor_start_date,
    )
    md = render_md(report, buckets_dir, now)
    queued = 0
    if args.review_queue:
        queued = enqueue_metabolism_suggestions(
            report,
            ReviewQueue(args.review_queue),
        )

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(md, encoding="utf-8")
        print(f"报告已写 → {outp}")
    else:
        print(md)
    if args.review_queue:
        print(f"待审建议新增 {queued} 条；记忆桶未改。")


if __name__ == "__main__":
    main()
