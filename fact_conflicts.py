#!/usr/bin/env python3
"""Fact evolution conflict helpers.

Pure helpers shared by merge-time supersedes audit and read-only patrol scans.
No BucketManager, MCP server, or queue imports here.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Iterable

BJ_TZ = timezone(timedelta(hours=8))

KV_CONFLICT_RE = re.compile(r"(?im)^\s*([A-Za-z0-9_\-\u4e00-\u9fff]{2,40})\s*[:：=]\s*([^\n#;；]+?)\s*$")
DATE_CONFLICT_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}月\d{1,2}[日号]?\b")
NUMBER_CONFLICT_RE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*(?:kg|g|mg|ml|cm|mm|km|斤|克|毫克|毫升|天|次|小时|分钟|分|度|℃|%|元|块|h|m)\b",
    re.IGNORECASE,
)
NEGATION_CONFLICT_RE = re.compile(r"\b(no|not|never|none|without|cancelled|canceled)\b|不再|不是|没有|没|无|取消|别|不要")
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

PROTECTED_Z_SCAN_DOMAINS = frozenset({"恋爱", "纪念日", "约定", "家庭", "自省", "feel"})


def strip_wikilinks_local(text: str) -> str:
    """Local wikilink stripper so this module stays independent from utils.py."""
    def repl(match: re.Match) -> str:
        return match.group(2) or match.group(1)

    return WIKILINK_RE.sub(repl, text or "")


def extract_key_values_for_conflict(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for match in KV_CONFLICT_RE.finditer(text or ""):
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key and value:
            pairs[key] = value[:180]
    return pairs


def detect_fact_conflicts(old_content: str, new_content: str) -> list[dict]:
    old_text = strip_wikilinks_local(old_content or "")
    new_text = strip_wikilinks_local(new_content or "")
    conflicts: list[dict] = []
    seen_fields: set[str] = set()

    old_kv = extract_key_values_for_conflict(old_text)
    new_kv = extract_key_values_for_conflict(new_text)
    for field, old_value in old_kv.items():
        new_value = new_kv.get(field)
        if new_value is not None and old_value != new_value:
            conflicts.append({"field": field, "old": old_value, "new": new_value})
            seen_fields.add(field)

    old_dates = sorted({m.group(0) for m in DATE_CONFLICT_RE.finditer(old_text)})
    new_dates = sorted({m.group(0) for m in DATE_CONFLICT_RE.finditer(new_text)})
    if old_dates and new_dates and old_dates != new_dates and "date" not in seen_fields:
        conflicts.append({
            "field": "date",
            "old": ", ".join(old_dates)[:180],
            "new": ", ".join(new_dates)[:180],
        })

    old_without_dates = DATE_CONFLICT_RE.sub(" ", old_text)
    new_without_dates = DATE_CONFLICT_RE.sub(" ", new_text)
    old_numbers = sorted({m.group(0).strip() for m in NUMBER_CONFLICT_RE.finditer(old_without_dates)})
    new_numbers = sorted({m.group(0).strip() for m in NUMBER_CONFLICT_RE.finditer(new_without_dates)})
    if old_numbers and new_numbers and old_numbers != new_numbers and "number" not in seen_fields:
        conflicts.append({
            "field": "number",
            "old": ", ".join(old_numbers)[:180],
            "new": ", ".join(new_numbers)[:180],
        })

    old_negated = bool(NEGATION_CONFLICT_RE.search(old_text))
    new_negated = bool(NEGATION_CONFLICT_RE.search(new_text))
    if old_negated != new_negated:
        conflicts.append({
            "field": "negation",
            "old": "negated" if old_negated else "affirmed",
            "new": "negated" if new_negated else "affirmed",
        })
    return conflicts


def build_supersedes_audit(bucket: dict, new_content: str, *, now: datetime | None = None) -> list[dict]:
    conflicts = detect_fact_conflicts(bucket.get("content", ""), new_content)
    if not conflicts:
        return []
    at = (now or datetime.now(BJ_TZ)).isoformat(timespec="seconds")
    return [
        {
            "field": c["field"],
            "old": c["old"],
            "new": c["new"],
            "at": at,
            "bucket_id": bucket.get("id", ""),
        }
        for c in conflicts
    ]


def _domains(meta: dict) -> list[str]:
    raw = meta.get("domain", []) or []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw if x]


def is_z_scan_candidate(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) or {}
    domains = set(_domains(meta))
    if meta.get("resolved") or meta.get("pinned") or meta.get("protected") or meta.get("permanent"):
        return False
    if meta.get("type") in {"feel", "permanent"}:
        return False
    if meta.get("nsfw") or meta.get("is_nsfw"):
        return False
    if domains & PROTECTED_Z_SCAN_DOMAINS:
        return False
    return bool(bucket.get("content"))


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip().lower())


def _bucket_group_keys(bucket: dict) -> set[str]:
    meta = bucket.get("metadata", {}) or {}
    keys: set[str] = set()
    name = _norm_name(meta.get("name") or bucket.get("id") or "")
    if name:
        keys.add(f"name:{name}")
    for domain in _domains(meta):
        if domain:
            keys.add(f"domain:{domain}")
    return keys


def scan_cross_bucket_z_conflicts(buckets: Iterable[dict], *, limit: int = 50) -> list[dict]:
    """Read-only cross-bucket conflict candidates for patrol.

    Conservative by design: only compare buckets sharing a normalized name or a
    domain label, skip protected/emotional buckets, and return reports only.
    """
    candidates = [b for b in buckets if is_z_scan_candidate(b)]
    reports: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, left in enumerate(candidates):
        left_id = left.get("id") or left.get("metadata", {}).get("id")
        if not left_id:
            continue
        left_keys = _bucket_group_keys(left)
        if not left_keys:
            continue
        for right in candidates[i + 1:]:
            right_id = right.get("id") or right.get("metadata", {}).get("id")
            if not right_id:
                continue
            pair = tuple(sorted((str(left_id), str(right_id))))
            if pair in seen_pairs:
                continue
            if not (left_keys & _bucket_group_keys(right)):
                continue
            conflicts = detect_fact_conflicts(left.get("content", ""), right.get("content", ""))
            if not conflicts:
                continue
            seen_pairs.add(pair)
            reports.append({
                "left_id": left_id,
                "left_name": left.get("metadata", {}).get("name", left_id),
                "right_id": right_id,
                "right_name": right.get("metadata", {}).get("name", right_id),
                "fields": conflicts,
                "reason": "cross_bucket_same_name_or_domain",
            })
            if len(reports) >= limit:
                return reports
    return reports
