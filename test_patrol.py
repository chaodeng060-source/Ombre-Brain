"""patrol 只读巡检自测 —— 不打 API、不碰 server、不改任何桶。

回归锁定（小卷 review 抓的两处静默失效）：
  - 脏 content 键 frontmatter 重组不丢 metadata（closing --- 须独占行）
  - _parse_dt 吃 frontmatter 直接给的 datetime 对象（不止 str）
另覆盖：递归扫 .md（非旧的顶层 .json）、保护域 resolved 命中、
       陈旧重要命中、整体跑通不炸。
"""
import tempfile
from datetime import datetime
from pathlib import Path

import patrol


def _write(dir_path, rel, text):
    p = Path(dir_path) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---- Q2: _parse_dt 吃多种输入 ----
def test_parse_dt_accepts_datetime_object():
    # frontmatter 把未加引号的 ISO 时间读成 aware datetime；必须接住并归一 naive
    dt = datetime.fromisoformat("2026-01-01T00:00:00+08:00")
    out = patrol._parse_dt(dt)
    assert out is not None
    assert out.tzinfo is None
    assert (out.year, out.month, out.day) == (2026, 1, 1)


def test_parse_dt_accepts_quoted_str():
    out = patrol._parse_dt("2026-05-24T11:00:00+08:00")
    assert out is not None and out.tzinfo is None


def test_parse_dt_none_on_empty_or_garbage():
    assert patrol._parse_dt(None) is None
    assert patrol._parse_dt("") is None
    assert patrol._parse_dt("not-a-date") is None


# ---- Q1: 脏 content 键 frontmatter 不丢 metadata ----
def test_safe_frontmatter_dirty_content_preserves_metadata():
    md = (
        "---\n"
        "id: dirty1\n"
        "domain:\n- 恋爱\n"
        "resolved: true\n"
        "content: collides with body positional arg\n"
        "last_active: 2026-01-01T00:00:00+08:00\n"
        "---\n"
        "real body here\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "permanent/dirty1.md", md)
        post = patrol._safe_frontmatter(p)
        meta = dict(post.metadata)
    assert meta.get("id") == "dirty1"
    assert meta.get("domain") == ["恋爱"]
    assert meta.get("resolved") is True
    assert "last_active" in meta          # 字段没被吞进 body
    assert post.content.strip() == "real body here"


# ---- 递归扫 .md（旧版扫顶层 .json 会得 0 桶） ----
def test_load_buckets_recursive_md():
    with tempfile.TemporaryDirectory() as d:
        _write(d, "permanent/a.md", "---\nid: a\ntype: permanent\n---\nbody a\n")
        _write(d, "dynamic/生活/b.md", "---\nid: b\ntype: dynamic\n---\nbody b\n")
        _write(d, "stale.json", '{"id": "ignored"}')   # 旧路径产物：不该被当桶
        loaded = patrol._load_buckets(Path(d))
    ids = {b.get("id") for b in loaded if not b.get("__broken__")}
    assert ids == {"a", "b"}


# ---- 端到端：保护域 resolved + 陈旧重要 都命中，且不炸 ----
def test_patrol_end_to_end_flags():
    with tempfile.TemporaryDirectory() as d:
        # 保护域被 resolve（5.10 守卫必须能抓到）
        _write(d, "permanent/love.md",
               "---\nid: love\nname: 约定\ntype: permanent\n"
               "domain:\n- 恋爱\nresolved: true\n"
               "importance: 10\nlast_active: 2020-01-01T00:00:00+08:00\n"
               "---\n誓约\n")
        # 重要但久未激活（未加引号时间→datetime，依赖 Q2 修复才不漏）
        _write(d, "dynamic/old.md",
               "---\nid: old\nname: 旧事\ntype: dynamic\n"
               "importance: 9\nlast_active: 2020-01-01T00:00:00+08:00\n"
               "---\n很久以前\n")
        now = datetime(2026, 6, 16, 0, 0, 0)
        rep = patrol.patrol(Path(d), now)
    assert rep["total"] == 2
    assert any(x["id"] == "love" for x in rep["protected_resolved"])
    assert any(x["id"] == "old" for x in rep["stale_important"])
    assert "巡检" in patrol.render_md(rep, Path(d), now)   # render 不炸
