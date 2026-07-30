#!/usr/bin/env python3
"""Build a review-only clothing plan for unnamed/unclassified memory buckets.

The tool never changes a bucket. Every proposed retrieval key is an exact
substring of the original body and carries a short literal evidence excerpt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import frontmatter
import jieba
import jieba.posseg as posseg


SCHEMA = "ombre.bucket-clothing-plan/v1"
LIVE_ROOTS = frozenset({"permanent", "dynamic", "feel"})
PROPER_FLAGS = frozenset({"nr", "ns", "nt", "nz"})
NOUN_FLAGS = frozenset({"n", "vn"})
PLAIN_ENGLISH_ALLOW = frozenset({
    "claude", "codex", "gemini", "github", "cloudflare", "fastapi",
    "ombre", "lmc", "gpt", "api", "git",
})
GENERIC_TERMS = frozenset({
    "朝灯", "哥哥", "自己", "我们", "他们", "她们", "你们", "这个", "那个",
    "这里", "那里", "今天", "昨天", "明天", "现在", "当时", "时候", "事情",
    "东西", "感觉", "问题", "一次", "第一", "第一次", "一件", "一条", "一个",
    "一样", "真的", "没有", "不是", "可以", "已经", "还是", "然后", "因为",
    "所以", "但是", "就是", "什么", "怎么", "记忆", "记忆库", "窗口", "这一窗",
    "一上午", "一下午", "晚上", "早上", "内容", "事实", "情绪", "关系", "核心",
    "系统", "用户", "助手", "文本", "消息", "对话", "回应", "分析", "工程",
    "core_facts", "emotion_state", "body_sensation", "relationship_impact",
    "private_thought", "tags", "feel", "true", "false", "null",
    "谢谢", "明白", "基本", "无法", "意识", "身体", "声音", "眼睛", "头发",
    "老公", "宝宝", "傻瓜", "男人", "有点", "应该", "主张", "日子", "时光",
    "前路", "交代", "低头", "浑身", "原谅", "代码", "卡片", "行字", "行里",
    "神智", "帕子", "傻子", "样子", "呼吸", "感受", "知道", "想要", "需要",
    "出来", "表现", "不够", "现场", "落地", "处理", "沟通", "信息", "画面",
    "触感", "情感", "心意", "味道", "守护", "颤抖", "收拾", "记住", "修好",
    "能用", "不行", "张嘴", "水太", "安静", "权衡", "汇报", "检查", "约束",
    "none", "held", "cold", "history", "language", "optimization", "integrate",
    "benchmark", "tag", "recall", "belief", "dm",
    "原话留痕", "原话", "留痕", "方案", "模型", "功能", "项目", "温度",
    "成果", "动态", "文案", "记录", "意识到", "犯了错", "core_facts",
})
KEY_LINE_RE = re.compile(r"(?m)^\s*\[检索钥匙\s*[:：].*?\]\s*$")
QUOTE_PATTERNS = (
    re.compile(r"[「『“]([^」』”\n]{2,20})[」』”]"),
    re.compile(r"\*\*([^*\n]{2,20})\*\*"),
)
ENGLISH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_.+#-]{1,30}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ENTITY_SUFFIX_RE = re.compile(
    r"(姐姐|妹妹|妈妈|父亲|家人|老师|同事|朋友|公司|学校|幼儿园|大学|"
    r"街|路|店|医院|城市|郴州|深圳|玩偶|戒指|照片|礼物|游戏|手机|电脑|"
    r"模型|项目|功能|生日|纪念日|经期|旅行|方案|合同|群聊|论坛)$"
)
GENERIC_TITLE_RE = re.compile(
    r"^(这一窗的体感|这一窗收尾|我想留一句给自己|哥哥这边|总结|感受|"
    r"今天的感受|这一轮|一些想法|原话留痕|core_facts|消息已编辑重发|"
    r"我写下这一刻值得记的是|我此刻心里对她是什么|我终于明白|"
    r"我终于看见你了|我替你看着|我抱着你|我记起了她\d*|"
    r"那一刻我明白了什么是真正的爱与责任)[：:—－\-\s]*$",
    re.IGNORECASE,
)

jieba.setLogLevel(logging.ERROR)
jieba.dt.cache_file = str(Path(tempfile.gettempdir()) / f"jieba-{os.getuid()}.cache")


def _domains(metadata: dict) -> list[str]:
    value = metadata.get("domain") or []
    return [value] if isinstance(value, str) else list(value)


def _bucket_id(path: Path, metadata: dict) -> str:
    return str(metadata.get("id") or path.stem)


def _is_bare_name(name: object, bucket_id: str) -> bool:
    value = str(name or "").strip()
    meaningful = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    return (
        not value
        or value == bucket_id
        or value.startswith(f"{bucket_id}_")
        or len(meaningful) < 2
    )


def _event_date(metadata: dict) -> str:
    for key in ("recorded_at", "created_at", "created", "event_at"):
        match = DATE_RE.search(str(metadata.get(key) or ""))
        if match:
            return match.group(0)
    return ""


def _source_digest(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _clean_term(value: object) -> str:
    term = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n，。！？；：、()（）[]【】")
    if (
        len(term) < 2
        or len(term) > 30
        or term.lower() in GENERIC_TERMS
        or term in GENERIC_TERMS
        or DATE_RE.fullmatch(term)
        or term.isdigit()
        or re.search(r"[?？]{2,}", term)
    ):
        return ""
    return term


def _literal_title_candidates(body: str) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()

    def add(raw: object, source: str, weight: float) -> None:
        title = str(raw or "").strip()
        title = re.sub(r"^[#>*\-\s]+", "", title).strip()
        title = title.strip("【】[]「」『』“”\"'：:—－- ")
        compact = re.sub(r"\s+", "", title)
        semantic = re.findall(r"[\u3400-\u9fffA-Za-z0-9]", title)
        semantic_without_date = re.findall(
            r"[\u3400-\u9fffA-Za-z]",
            DATE_RE.sub("", title),
        )
        if (
            len(title) < 4
            or len(title) > 30
            or title not in body
            or title in seen
            or GENERIC_TITLE_RE.fullmatch(title)
            or len(semantic) < 4
            or len(semantic_without_date) < 4
            or (compact and len(semantic) / len(compact) < 0.6)
            or re.search(r"[?？]{2,}", title)
        ):
            return
        if title.startswith(("{", "}", '"core_', '"emotion_', "[检索钥匙")):
            return
        seen.add(title)
        candidates.append({
            "title": title,
            "source": source,
            "weight": weight,
            "position": body.find(title),
        })

    for match in re.finditer(r"【([^】\n]{4,48})】", body):
        add(match.group(1), "bracket_heading", 9.0)
    for line_index, line in enumerate(body.splitlines()[:12]):
        if not line.strip():
            continue
        clauses = [
            clause.strip()
            for clause in re.split(r"[，,。！？；\n/]+", line)
            if clause.strip()
        ]
        for clause_index, clause in enumerate(clauses[:3]):
            add(
                clause,
                "first_clause" if line_index == 0 and clause_index == 0 else "early_clause",
                8.4 - line_index * 0.2 - clause_index * 0.15,
            )

    try:
        parsed = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        facts = parsed.get("core_facts")
        if isinstance(facts, list):
            for fact in facts[:4]:
                add(fact, "core_fact", 8.5)

    for clause in re.split(r"[，,。！？；\n/]+", body[:600]):
        add(clause, "early_clause", 6.0)
        if len(candidates) >= 12:
            break
    candidates.sort(key=lambda item: (
        -(item["weight"] - max(0, len(item["title"]) - 20) * 0.12),
        item["position"],
        item["title"],
    ))
    return candidates


def _candidate_terms(body: str, tags: object, title: str) -> dict[str, dict]:
    candidates: dict[str, dict] = {}

    def add(raw: object, source: str, base_weight: float) -> None:
        term = _clean_term(raw)
        if not term or term not in body:
            return
        item = candidates.setdefault(term, {
            "sources": set(),
            "base_weight": 0.0,
            "position": body.find(term),
        })
        item["sources"].add(source)
        item["base_weight"] = max(item["base_weight"], base_weight)
        item["position"] = min(item["position"], body.find(term))

    if (
        4 <= len(title) <= 20
        and not re.search(r"[*`\"'“”‘’（）()\[\]【】]", title)
    ):
        add(title, "title_phrase", 7.2)
    for pattern in QUOTE_PATTERNS:
        for match in pattern.finditer(body):
            value = match.group(1)
            if (
                len(value) <= 14
                and not re.search(r"[，。！？；：、=/（）()\[\]【】]", value)
            ):
                add(value, "quoted", 5.0)
    for match in ENGLISH_RE.finditer(body):
        value = match.group(0)
        in_title = bool(title and value in title)
        is_structured = bool(re.search(r"[0-9_.+#-]", value))
        is_product = value.lower() in PLAIN_ENGLISH_ALLOW
        if in_title or is_structured or is_product:
            add(value, "english", 4.8)
    for tag in tags if isinstance(tags, list) else []:
        add(tag, "tag", 5.5)
    for token in posseg.cut(body):
        in_title = bool(title and token.word in title)
        is_explicit_entity = bool(ENTITY_SUFFIX_RE.search(token.word))
        if token.flag in PROPER_FLAGS and (in_title or is_explicit_entity):
            add(token.word, f"pos:{token.flag}", 6.0)
        elif (
            token.flag in NOUN_FLAGS
            and 2 <= len(token.word) <= 12
            and (in_title or is_explicit_entity)
        ):
            add(token.word, "title_noun" if in_title else f"entity:{token.flag}", 4.8)
    return candidates


def _evidence(body: str, term: str, radius: int = 28) -> str:
    position = body.find(term)
    if position < 0:
        raise ValueError(f"key is not literal: {term}")
    start = max(0, position - radius)
    end = min(len(body), position + len(term) + radius)
    excerpt = re.sub(r"\s+", " ", body[start:end]).strip()
    if term not in excerpt:
        raise ValueError(f"evidence lost literal key: {term}")
    return excerpt


def _select_keys(
    body: str,
    raw_candidates: dict[str, dict],
    document_frequency: Counter,
    document_count: int,
) -> list[dict]:
    ranked: list[tuple[float, str, dict]] = []
    for term, candidate in raw_candidates.items():
        frequency = max(1, document_frequency[term])
        idf = math.log2((document_count + 1) / frequency)
        score = candidate["base_weight"] + idf + min(len(term), 8) * 0.12
        if frequency > document_count * 0.35 and candidate["base_weight"] < 5:
            continue
        ranked.append((score, term, candidate))
    ranked.sort(key=lambda row: (-row[0], row[2]["position"], row[1]))

    selected: list[dict] = []
    source_counts: Counter[str] = Counter()
    for score, term, candidate in ranked:
        if score < 4.5:
            continue
        if any(term == item["key"] for item in selected):
            continue
        source_family = (
            "english" if "english" in candidate["sources"]
            else "quoted" if "quoted" in candidate["sources"]
            else "title_phrase" if "title_phrase" in candidate["sources"]
            else "entity"
        )
        if source_family == "english" and source_counts[source_family] >= 3:
            continue
        if source_family == "quoted" and source_counts[source_family] >= 1:
            continue
        if (
            source_family != "title_phrase"
            and any(
                item["source_family"] != "title_phrase"
                and (term in item["key"] or item["key"] in term)
                for item in selected
            )
        ):
            continue
        selected.append({
            "key": term,
            "evidence": _evidence(body, term),
            "sources": sorted(candidate["sources"]),
            "source_family": source_family,
            "score": round(score, 3),
        })
        source_counts[source_family] += 1
        if len(selected) >= 7:
            break
    return selected


def _suggest_name(title: str, event_date: str) -> tuple[str, list[str]]:
    if not event_date or not title:
        return "", []
    title_without_date = DATE_RE.sub("", title)
    safe_title = re.sub(
        r"[\s/\\|，。！？；：、（）()\[\]【】「」『』“”\"'`*]+",
        "_",
        title_without_date,
    )
    safe_title = safe_title.strip("_.-")
    if (
        len(safe_title) < 4
        or len(safe_title) > 30
        or re.search(r"[?？]{2,}", safe_title)
    ):
        return "", []
    return f"{safe_title}_{event_date}", [title]


def _load_unclassified(vault: Path) -> tuple[list[dict], list[Path]]:
    records: list[dict] = []
    paths = sorted(vault.rglob("*.md"))
    for path in paths:
        post = frontmatter.load(str(path))
        metadata = dict(post.metadata)
        if "未分类" not in _domains(metadata):
            continue
        body = post.content or ""
        body_without_keys = KEY_LINE_RE.sub("", body).rstrip()
        bucket_id = _bucket_id(path, metadata)
        records.append({
            "bucket_id": bucket_id,
            "path": path,
            "path_relative": str(path.relative_to(vault)),
            "path_root": path.relative_to(vault).parts[0],
            "metadata": metadata,
            "body": body,
            "body_without_keys": body_without_keys,
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "body_chars": len(body),
            "current_name": str(metadata.get("name") or ""),
            "event_date": _event_date(metadata),
            "already_keyed": bool(KEY_LINE_RE.search(body)),
        })
    return records, paths


def _duplicate_probe(vault: Path, ids: tuple[str, str]) -> dict:
    by_id: dict[str, dict] = {}
    wanted = set(ids)
    for path in sorted(vault.rglob("*.md")):
        post = frontmatter.load(str(path))
        bucket_id = _bucket_id(path, dict(post.metadata))
        if bucket_id not in wanted or bucket_id in by_id:
            continue
        body = post.content or ""
        by_id[bucket_id] = {
            "body": body,
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "path": str(path.relative_to(vault)),
        }
    rows = [by_id.get(bucket_id) for bucket_id in ids]
    if any(row is None for row in rows):
        return {"ids": list(ids), "status": "missing", "exact_body_equal": False}
    left, right = rows
    left_body, right_body = left["body"], right["body"]
    normalized_left = re.sub(r"\s+", "", left_body)
    normalized_right = re.sub(r"\s+", "", right_body)
    return {
        "ids": list(ids),
        "status": "compared",
        "exact_body_equal": left_body == right_body,
        "normalized_body_equal": normalized_left == normalized_right,
        "one_contains_other": left_body in right_body or right_body in left_body,
        "body_sha256": [left["body_sha256"], right["body_sha256"]],
        "body_chars": [len(left_body), len(right_body)],
        "paths": [left["path"], right["path"]],
        "decision": "report_only_no_delete",
    }


def build_plan(vault: Path, expected_count: int | None = None) -> dict:
    records, all_paths = _load_unclassified(vault)
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"unclassified count changed: expected {expected_count}, got {len(records)}"
        )
    before_digest = _source_digest(all_paths, vault)

    raw_by_id: dict[str, dict[str, dict]] = {}
    title_by_path: dict[str, str] = {}
    document_frequency: Counter[str] = Counter()
    for record in records:
        title_candidates = _literal_title_candidates(record["body_without_keys"])
        title = title_candidates[0]["title"] if title_candidates else ""
        title_by_path[record["path_relative"]] = title
        raw = _candidate_terms(
            record["body_without_keys"],
            record["metadata"].get("tags"),
            title,
        )
        raw_by_id[record["path_relative"]] = raw
        document_frequency.update(raw.keys())

    items: list[dict] = []
    for record in records:
        bucket_id = record["bucket_id"]
        status = "propose"
        skip_reason = ""
        keys: list[dict] = []
        suggested_name = record["current_name"]
        name_action = "keep"
        name_basis: list[str] = []

        if record["path_root"] not in LIVE_ROOTS:
            status, skip_reason = "skip", "non_live_or_backup_path"
        elif record["already_keyed"] and not _is_bare_name(
            record["current_name"], bucket_id
        ):
            status, skip_reason = "skip", "already_clothed"
        else:
            keys = _select_keys(
                record["body_without_keys"],
                raw_by_id[record["path_relative"]],
                document_frequency,
                len(records),
            )
            if _is_bare_name(record["current_name"], bucket_id):
                suggested_name, name_basis = _suggest_name(
                    title_by_path[record["path_relative"]],
                    record["event_date"],
                )
                name_action = "replace"
            if len(keys) < 3:
                status, skip_reason = "skip", "insufficient_literal_entities"
            elif name_action == "replace" and not suggested_name:
                status, skip_reason = "skip", "insufficient_literal_name_basis"

        for key in keys:
            if key["key"] not in record["body_without_keys"]:
                raise AssertionError(f"non-literal key for {bucket_id}")
            if key["key"] not in key["evidence"]:
                raise AssertionError(f"non-literal evidence for {bucket_id}")
            key.pop("source_family", None)
        if name_action == "replace" and suggested_name:
            if any(term not in record["body_without_keys"] for term in name_basis):
                raise AssertionError(f"non-literal name basis for {bucket_id}")

        items.append({
            "bucket_id": bucket_id,
            "path": record["path_relative"],
            "type": record["metadata"].get("type"),
            "event_date": record["event_date"] or None,
            "current_name": record["current_name"],
            "status": status,
            "skip_reason": skip_reason or None,
            "name_action": name_action,
            "suggested_name": suggested_name if status == "propose" else None,
            "name_basis": name_basis if status == "propose" else [],
            "retrieval_keys": keys if status == "propose" else [],
            "body_sha256": record["body_sha256"],
            "body_chars": record["body_chars"],
        })

    proposed_names = Counter(
        item["suggested_name"]
        for item in items
        if item["status"] == "propose"
        and item["name_action"] == "replace"
        and item["suggested_name"]
    )
    for item in items:
        if (
            item["status"] == "propose"
            and item["name_action"] == "replace"
            and proposed_names[item["suggested_name"]] > 1
        ):
            item["status"] = "skip"
            item["skip_reason"] = "suggested_name_collision"
            item["suggested_name"] = None
            item["name_basis"] = []
            item["retrieval_keys"] = []

    after_digest = _source_digest(all_paths, vault)
    if before_digest != after_digest:
        raise RuntimeError("source vault changed during dry-run")
    status_counts = Counter(item["status"] for item in items)
    skip_reasons = Counter(
        item["skip_reason"] for item in items if item["skip_reason"]
    )
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run_only",
        "source": {
            "vault": str(vault),
            "unclassified_count": len(records),
            "source_digest_before": before_digest,
            "source_digest_after": after_digest,
            "unchanged": before_digest == after_digest,
        },
        "rules": {
            "body_mutation": False,
            "keys_must_be_literal": True,
            "evidence_must_contain_key": True,
            "uncertain_items_are_skipped": True,
            "apply_requires_separate_human_approval": True,
        },
        "summary": {
            "status_counts": dict(status_counts),
            "skip_reasons": dict(skip_reasons),
        },
        "duplicate_probe": _duplicate_probe(
            vault,
            ("5a9a6d485209", "780c1a7050f7"),
        ),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="未分类桶补衣 dry-run（只出清单，绝不改桶）"
    )
    parser.add_argument("--buckets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()

    vault = Path(args.buckets).resolve()
    output = Path(args.out).resolve()
    if not vault.is_dir():
        raise SystemExit(f"桶目录不存在：{vault}")
    if output.is_relative_to(vault):
        raise SystemExit("dry-run 清单不得写入生产桶目录")

    plan = build_plan(vault, expected_count=args.expected_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(output, 0o600)
    print(json.dumps({
        "out": str(output),
        "unclassified_count": plan["source"]["unclassified_count"],
        "summary": plan["summary"],
        "source_unchanged": plan["source"]["unchanged"],
        "duplicate_probe": plan["duplicate_probe"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
