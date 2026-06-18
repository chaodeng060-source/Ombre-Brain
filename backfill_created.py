#!/usr/bin/env python3
"""
Migrate legacy `created` metadata to the split v2 time model.

为存量桶补齐 `event_at`、`recorded_at`、`date_precision`、`date_source`、
`date_confidence`。事件日期优先从桶名、tags（可选正文）推断；只能沿用旧
`created` 时会明确标成 `legacy_created`、置信度 0.3，避免把历史歧义伪装成真相。

与 bucket_mgr.update 的区别（关键）：
    update() 会无条件把 last_active 刷成 now。一旦回填全库，等于把所有桶的衰减
    时钟集体清零、搅乱未解决权重池。本脚本**只写时间模型字段、原样保留 last_active**，
    直接复用 bucket_manager 的 frontmatter 序列化，不走 update。

安全性：
    - 幂等：时间模型字段完整的桶跳过，不覆盖。
    - 保守：只认 name → tags（默认）里的日期；--scan-content 才扫正文（置信度低，
      正文里的日期未必是事件日期），且日期必须真实存在、不晚于今天、年份 ≥ 2024。
    - 推断不出日期的桶**不写**，让它在简报里诚实显示「无确切日期」，绝不瞎填。
    - 默认 dry-run，只打印将怎么改；真正写入要带 --go。
    - 写入使用同目录临时文件 + fsync + os.replace，避免中断留下半个 frontmatter。

Usage:
    OMBRE_BUCKETS_DIR=/data python backfill_created.py                 # dry-run（默认）
    OMBRE_BUCKETS_DIR=/data python backfill_created.py --scan-content  # dry-run + 扫正文
    OMBRE_BUCKETS_DIR=/data python backfill_created.py --go            # 实际写入
    docker exec ombre-brain python /app/backfill_created.py            # NAS 容器 dry-run
"""

import argparse
import asyncio
import os
import re
import sys
from datetime import date, datetime

sys.path.insert(0, ".")
from utils import load_config           # noqa: E402
from bucket_manager import BucketManager  # noqa: E402
from storage_safety import atomic_write_post as _atomic_write_post  # noqa: E402
from utils import normalize_event_at, now_iso  # noqa: E402

_MIN_YEAR = 2024

# 支持 2026-05-30 / 2026.5.30 / 2026/5/30；统一按文本位置取第一条。
# 数字边界避免从 12026-05-300 这类更长编号里截出伪日期。
_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{4})([-./])(\d{1,2})\2(\d{1,2})(?!\d)"
)


def _first_valid_date(text, today: date) -> date | None:
    """文本里第一个合法、年份≥2024、不晚于今天的日期；没有→None。"""
    if not isinstance(text, str):
        return None
    for m in _DATE_PATTERN.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(3)), int(m.group(4))
        if y < _MIN_YEAR:
            continue
        try:
            dt = date(y, mo, d)
        except ValueError:
            continue
        if dt > today:        # 未来日期不可能是事件发生时间
            continue
        return dt
    return None


def infer_created(meta: dict, content: str = "", today: date = None,
                  scan_content: bool = False) -> tuple[str | None, str]:
    """从桶元数据推断 created（事件发生时间）。

    返回 (iso_str_or_None, source)。source ∈
    {already-has-created, name, tag, content, no-date-found}。
    优先级：已有合法 created > 桶名 > tags > （可选）正文。
    iso 格式对齐 now_iso()：'YYYY-MM-DDT00:00:00'（naive，时刻补零）。
    """
    today = today or date.today()

    existing = meta.get("created")
    if existing:
        try:
            datetime.fromisoformat(str(existing).replace("Z", "+00:00"))
            return None, "already-has-created"   # 幂等：不覆盖好数据
        except Exception:
            pass  # 损坏的 created，往下重新推断

    dt = _first_valid_date(meta.get("name", ""), today)
    if dt:
        return f"{dt.isoformat()}T00:00:00", "name"

    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    elif not isinstance(tags, (list, tuple, set)):
        tags = []
    for t in tags:
        dt = _first_valid_date(str(t), today)
        if dt:
            return f"{dt.isoformat()}T00:00:00", "tag"

    if scan_content:
        dt = _first_valid_date(content, today)
        if dt:
            return f"{dt.isoformat()}T00:00:00", "content"

    return None, "no-date-found"


def infer_time_metadata(
    meta: dict,
    *,
    content: str = "",
    today: date = None,
    scan_content: bool = False,
    recorded_at: str = None,
) -> dict:
    """Build missing v2 time fields without overstating legacy date certainty."""
    today = today or date.today()
    updates = {}

    event_at = meta.get("event_at")
    inferred_source = ""
    if event_at:
        try:
            event_at, inferred_precision = normalize_event_at(event_at)
            inferred_source = meta.get("date_source") or "existing_event_at"
        except (TypeError, ValueError):
            event_at = None

    if not event_at:
        inference_meta = dict(meta)
        inference_meta.pop("created", None)
        event_at, inferred_source = infer_created(
            inference_meta,
            content=content,
            today=today,
            scan_content=scan_content,
        )
        inferred_precision = "day" if event_at else "unknown"

    if not event_at and meta.get("created"):
        try:
            event_at, inferred_precision = normalize_event_at(meta["created"])
            inferred_source = "legacy_created"
        except (TypeError, ValueError):
            event_at = None

    if event_at:
        confidence_by_source = {
            "existing_event_at": 1.0,
            "name": 0.9,
            "tag": 0.8,
            "content": 0.5,
            "legacy_created": 0.3,
        }
        updates["event_at"] = event_at
        updates["created"] = event_at
        updates["date_precision"] = meta.get("date_precision") or inferred_precision
        updates["date_source"] = meta.get("date_source") or inferred_source
        updates["date_confidence"] = meta.get(
            "date_confidence",
            confidence_by_source.get(inferred_source, 0.5),
        )

    if not meta.get("recorded_at"):
        updates["recorded_at"] = recorded_at or now_iso()

    return {
        key: value
        for key, value in updates.items()
        if meta.get(key) != value
    }


async def backfill(go: bool = False, scan_content: bool = False):
    config = load_config()
    mgr = BucketManager(config)

    buckets = await mgr.list_all(include_archive=True)
    today = date.today()
    print(f"Total buckets: {len(buckets)}")

    todo, skipped, nodate = [], 0, []
    for b in buckets:
        fp = b.get("path") or mgr._find_bucket_file(b["id"])
        file_recorded_at = None
        if fp:
            try:
                file_recorded_at = datetime.fromtimestamp(
                    os.path.getmtime(fp)
                ).isoformat(timespec="seconds")
            except OSError:
                pass
        updates = infer_time_metadata(
            b["metadata"],
            content=b.get("content", ""),
            today=today,
            scan_content=scan_content,
            recorded_at=file_recorded_at,
        )
        if not updates:
            skipped += 1
        elif "event_at" not in updates and not b["metadata"].get("event_at"):
            nodate.append(b)
            todo.append((b, updates))
        else:
            todo.append((b, updates))

    print(f"时间字段已完整（跳过）: {skipped}")
    print(f"可迁移: {len(todo)}    推断不出事件日期: {len(nodate)}")
    for b, updates in todo:
        name = b["metadata"].get("name", b["id"])
        print(
            f"  [{updates.get('date_source', 'recorded-only'):16}] "
            f"{b['id'][:12]} ({name[:30]}) -> "
            f"event_at={updates.get('event_at', '(unknown)')}"
        )
    if nodate:
        print("  -- 无日期可推断（不写，简报里诚实显示「无确切日期」）--")
        for b in nodate[:50]:
            name = b["metadata"].get("name", b["id"])
            print(f"     {b['id'][:12]} ({name[:30]})")

    if not go:
        print("\n(dry-run) 没有写入。确认无误后加 --go 实际写入。")
        return

    ok = fail = 0
    for b, updates in todo:
        try:
            fp = mgr._find_bucket_file(b["id"])
            if not fp:
                fail += 1
                print(f"  FAIL (file not found): {b['id']}")
                continue
            async with mgr._write_guard(b["id"]):
                post = mgr._safe_load_post(fp)
                before = mgr._post_snapshot(post, fp)
                for key, value in updates.items():
                    post[key] = value
                event_id = mgr.audit_log.begin(
                    actor="migration:time_fields_v2",
                    action="migrate_time_fields",
                    bucket_id=b["id"],
                    before=before,
                    after=mgr._post_snapshot(post, fp),
                    details={"changed_fields": sorted(updates)},
                )
                try:
                    mgr._atomic_write_post(fp, post)
                    mgr.audit_log.commit(event_id)
                except Exception as write_error:
                    mgr.audit_log.fail(event_id, write_error)
                    raise
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  ERROR: {b['id']}: {e}")
    print(f"\n=== Done: {ok} 回填, {fail} 失败 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true",
                        help="actually write (default: dry-run)")
    parser.add_argument("--scan-content", action="store_true",
                        help="also scan bucket content for dates (lower confidence)")
    args = parser.parse_args()
    asyncio.run(backfill(go=args.go, scan_content=args.scan_content))
