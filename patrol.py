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
    OMBRE_BUCKETS_DIR=/vol1/ombre-data/buckets python patrol.py

    # 把报告落到文件
    python patrol.py --buckets <dir> --out notes/patrol_2026-06-15.md
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import frontmatter

# 与 utils.PROTECTED_RESOLVE_DOMAINS 保持一致（resolve=遗忘的禁区）。
# 这里硬编一份副本，让 patrol 不依赖 server 运行时即可独立巡检。
PROTECTED_RESOLVE_DOMAINS = frozenset({"恋爱", "纪念日", "约定", "家庭", "自省", "feel"})

# 报告阈值（保守，宁可漏报不误报；全部可调）
OVERSIZED_CHARS = 1500          # content 超此长度 → 拆线候选（只提示）
STALE_DAYS = 90                 # 高重要度桶超此天数没激活 → 提示（绝不自动忘）
STALE_IMPORTANCE = 7            # 仅对 importance>=此值的桶报陈旧（重要的才值得提醒）


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


def patrol(buckets_dir: Path, now: datetime) -> dict:
    raw = _load_buckets(buckets_dir)
    broken = [b for b in raw if b.get("__broken__")]
    buckets = [b for b in raw if not b.get("__broken__")]

    ids = set()
    for b in buckets:
        bid = b.get("id") or b.get("metadata", {}).get("id")
        if bid:
            ids.add(bid)

    by_type: Counter = Counter()
    by_domain: Counter = Counter()
    dangling: list[dict] = []          # 关系指向不存在的桶（断边）
    non_reciprocal: list[dict] = []    # A→B 有边、B→A 没有（信息性，不一定是病）
    oversized: list[dict] = []         # content 过长 → 拆线候选
    name_index: defaultdict = defaultdict(list)  # 重名 → 重复候选
    protected_resolved: list[dict] = []          # 保护域被 resolve（5.10 守卫验证）
    stale_important: list[dict] = []   # 重要但久未激活（只提示）

    # 先建反向关系索引，判互惠
    fwd_edges: defaultdict = defaultdict(set)
    for b in buckets:
        meta = b.get("metadata", {})
        bid = b.get("id") or meta.get("id")
        for rel in meta.get("relations", []) or []:
            tgt = rel.get("target")
            if tgt:
                fwd_edges[bid].add(tgt)

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
        for rel in meta.get("relations", []) or []:
            tgt = rel.get("target")
            if tgt and tgt not in ids:
                dangling.append({"from": bid, "name": name, "target": tgt, "type": rel.get("type")})
            elif tgt and bid not in fwd_edges.get(tgt, set()):
                non_reciprocal.append({"from": bid, "target": tgt, "type": rel.get("type")})

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

    return {
        "total": len(buckets),
        "broken": broken,
        "by_type": dict(by_type.most_common()),
        "by_domain": dict(by_domain.most_common(12)),
        "dangling": dangling,
        "non_reciprocal": non_reciprocal,
        "oversized": sorted(oversized, key=lambda x: -x["chars"])[:15],
        "duplicates": duplicates,
        "protected_resolved": protected_resolved,
        "stale_important": sorted(stale_important, key=lambda x: -x["days"])[:20],
    }


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

    def section(title, items, fmt, empty="无"):
        L.append(f"## {title}（{len(items)}）")
        if not items:
            L.append(f"- ✅ {empty}")
        else:
            for it in items[:30]:
                L.append(f"- {fmt(it)}")
        L.append("")

    section("🔴 保护域被 resolve（5.10 守卫·必须为 0）", report["protected_resolved"],
            lambda x: f"`{x['id']}` {x['name']} —— domains={x['domains']}",
            empty="守卫完好，无保护域被遗忘")
    section("🔗 断边（关系指向不存在的桶）", report["dangling"],
            lambda x: f"`{x['from']}` ({x['name']}) --{x['type']}--> `{x['target']}` ❌不存在")
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
    ap.add_argument("--buckets", default=os.environ.get("OMBRE_BUCKETS_DIR", "/data/buckets"),
                    help="桶目录（默认 $OMBRE_BUCKETS_DIR 或 /data/buckets）")
    ap.add_argument("--out", default=None, help="报告落点（默认打印到 stdout）")
    ap.add_argument("--now", default=None, help="覆盖当前时间（ISO，便于测试）")
    args = ap.parse_args()

    buckets_dir = Path(args.buckets)
    if not buckets_dir.is_dir():
        raise SystemExit(f"桶目录不存在：{buckets_dir}")
    now = _parse_dt(args.now) or datetime.now()

    report = patrol(buckets_dir, now)
    md = render_md(report, buckets_dir, now)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(md, encoding="utf-8")
        print(f"报告已写 → {outp}")
    else:
        print(md)


if __name__ == "__main__":
    main()
