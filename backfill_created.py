#!/usr/bin/env python3
"""
Backfill the `created` (event-time) field on existing buckets.

给「没有 created / created 损坏」的存量桶补上事件发生时间，从桶名、tags（必要时
正文）里的 `YYYY-MM-DD` 推断。这是「全库时间标 + 按时间推进的故事线泳道」的地基：
saga 泳道、因果边都已在库里，缺的就是每个桶在时间轴上的可靠坐标。

为什么要补：卡兜咬人桶 created/last_active 双空，简报里只能显示「⚠ 无确切日期」
（见 server._event_age_label，2026-06-18）。补上 created 后，它会显示成
「发生于 2026-05-30（距今 N 天）」，朝灯的时间感不再漂（她以为一个月，其实 19 天）。

与 bucket_mgr.update 的区别（关键）：
    update() 会无条件把 last_active 刷成 now。一旦回填全库，等于把所有桶的衰减
    时钟集体清零、搅乱未解决权重池。本脚本**只写 created、原样保留 last_active**，
    直接复用 bucket_manager 的 frontmatter 序列化，不走 update。

安全性：
    - 幂等：已有可解析 created 的桶跳过，不覆盖。
    - 保守：只认 name → tags（默认）里的日期；--scan-content 才扫正文（置信度低，
      正文里的日期未必是事件日期），且日期必须真实存在、不晚于今天、年份 ≥ 2024。
    - 推断不出日期的桶**不写**，让它在简报里诚实显示「无确切日期」，绝不瞎填。
    - 默认 dry-run，只打印将怎么改；真正写入要带 --go。

Usage:
    OMBRE_BUCKETS_DIR=/data python backfill_created.py                 # dry-run（默认）
    OMBRE_BUCKETS_DIR=/data python backfill_created.py --scan-content  # dry-run + 扫正文
    OMBRE_BUCKETS_DIR=/data python backfill_created.py --go            # 实际写入
"""

import argparse
import asyncio
import re
import sys
from datetime import date, datetime

import frontmatter

sys.path.insert(0, ".")
from utils import load_config           # noqa: E402
from bucket_manager import BucketManager  # noqa: E402

_MIN_YEAR = 2024

# 支持 2026-05-30 / 2026.5.30 / 2026/5/30 三种写法
_DATE_PATTERNS = [
    re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"),
    re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})"),
    re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})"),
]


def _first_valid_date(text, today: date) -> date | None:
    """文本里第一个合法、年份≥2024、不晚于今天的日期；没有→None。"""
    if not isinstance(text, str):
        return None
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
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

    for t in (meta.get("tags") or []):
        dt = _first_valid_date(str(t), today)
        if dt:
            return f"{dt.isoformat()}T00:00:00", "tag"

    if scan_content:
        dt = _first_valid_date(content, today)
        if dt:
            return f"{dt.isoformat()}T00:00:00", "content"

    return None, "no-date-found"


async def backfill(go: bool = False, scan_content: bool = False):
    config = load_config()
    mgr = BucketManager(config)

    buckets = await mgr.list_all(include_archive=True)
    today = date.today()
    print(f"Total buckets: {len(buckets)}")

    todo, skipped, nodate = [], 0, []
    for b in buckets:
        iso, source = infer_created(
            b["metadata"], content=b.get("content", ""),
            today=today, scan_content=scan_content,
        )
        if source == "already-has-created":
            skipped += 1
        elif iso is None:
            nodate.append(b)
        else:
            todo.append((b, iso, source))

    print(f"已有 created（跳过）: {skipped}")
    print(f"可回填: {len(todo)}    推断不出日期（保持「无确切日期」）: {len(nodate)}")
    for b, iso, source in todo:
        name = b["metadata"].get("name", b["id"])
        print(f"  [{source:7}] {b['id'][:12]} ({name[:30]}) -> created={iso}")
    if nodate:
        print("  -- 无日期可推断（不写，简报里诚实显示「无确切日期」）--")
        for b in nodate[:50]:
            name = b["metadata"].get("name", b["id"])
            print(f"     {b['id'][:12]} ({name[:30]})")

    if not go:
        print("\n(dry-run) 没有写入。确认无误后加 --go 实际写入。")
        return

    ok = fail = 0
    for b, iso, _source in todo:
        try:
            fp = mgr._find_bucket_file(b["id"])
            if not fp:
                fail += 1
                print(f"  FAIL (file not found): {b['id']}")
                continue
            post = mgr._safe_load_post(fp)
            post["created"] = iso          # 只动 created，last_active 原样保留
            with open(fp, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
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
