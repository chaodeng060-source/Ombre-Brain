"""patrol 只读巡检自测 —— 不打 API、不碰 server、不改任何桶。

回归锁定（小卷 review 抓的两处静默失效）：
  - 脏 content 键 frontmatter 重组不丢 metadata（closing --- 须独占行）
  - _parse_dt 吃 frontmatter 直接给的 datetime 对象（不止 str）
另覆盖：递归扫 .md、读取备份 JSON 快照并忽略 sidecar、保护域 resolved
       命中、陈旧重要命中、整体跑通不炸。
"""
import tempfile
import hashlib
import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

import patrol
from conversation_activity import SCHEMA, TIMEZONE_NAME
from review_queue import ReviewQueue


def _write(dir_path, rel, text):
    p = Path(dir_path) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _write_embedding_db(path: Path, rows: list[tuple[str, object]]) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE embeddings (
                bucket_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO embeddings(bucket_id, embedding, updated_at) VALUES (?, ?, ?)",
            [
                (bucket_id, json.dumps(embedding), "2026-07-30T12:00:00+08:00")
                for bucket_id, embedding in rows
            ],
        )
    return path


def _activity_summary(counts: dict[str, int]) -> dict:
    return {
        "schema": SCHEMA,
        "timezone": TIMEZONE_NAME,
        "start_date": "2026-07-30",
        "daily_user_messages": counts,
    }


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
    assert rep["suggestions"]
    assert all(item["reason"] for item in rep["suggestions"])
    assert "巡检" in patrol.render_md(rep, Path(d), now)   # render 不炸


def test_patrol_reports_relation_hygiene_and_duplicate_fact_slots():
    with tempfile.TemporaryDirectory() as d:
        _write(d, "dynamic/a.md",
               "---\nid: a\nname: A\ntype: dynamic\nfact_key: profile.city\n"
               "relations:\n"
               "- {type: kin, target: b, strength: 0.9}\n"
               "- {type: kin, target: b, strength: 0.9}\n"
               "- {type: kin, target: a}\n"
               "- {type: made_up, target: b}\n"
               "- {type: explains, target: b, strength: 2.0}\n"
               "---\nA body\n")
        _write(d, "dynamic/b.md",
               "---\nid: b\nname: B\ntype: dynamic\nfact_key: profile.city\n"
               "relations:\n- {type: kin, target: a}\n"
               "---\nB body\n")
        rep = patrol.patrol(
            Path(d),
            datetime(2026, 6, 19, 0, 0, 0),
            fact_slot_registry={"profile.city": {"aliases": ["城市"]}},
        )

    assert rep["self_loops"]
    assert rep["duplicate_edges"]
    assert rep["reciprocal_kin"] == [{"from": "a", "target": "b", "type": "kin"}]
    assert any(item["type"] == "made_up" for item in rep["invalid_relation_types"])
    assert any(item["strength"] == 2.0 for item in rep["invalid_relation_strengths"])
    assert rep["fact_conflicts"] == {"profile.city": ["a", "b"]}


def test_patrol_reads_json_backup_snapshots_and_ignores_sidecars():
    with tempfile.TemporaryDirectory() as d:
        snapshot = {
            "id": "abcdef123456",
            "metadata": {
                "id": "abcdef123456",
                "name": "城市事实",
                "type": "dynamic",
                "domain": ["生活"],
            },
            "content": "居住城市: 杭州",
        }
        Path(d, "abcdef123456.json").write_text(
            json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
        )
        Path(d, "body_state.json").write_text(
            json.dumps({"arousal": 0.5}), encoding="utf-8"
        )

        rep = patrol.patrol(
            Path(d),
            datetime(2026, 6, 19, 0, 0, 0),
            fact_slot_registry={"profile.city": {"aliases": ["居住城市"]}},
        )

    assert rep["total"] == 1
    assert rep["broken"] == []
    assert rep["migration_candidates"] == [
        {"id": "abcdef123456", "fact_key": "profile.city", "values": ["杭州"]}
    ]


def test_patrol_is_read_only_and_queues_reasoned_suggestions_idempotently():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as qd:
        bucket = _write(
            d,
            "dynamic/long.md",
            "---\nid: long\nname: 长桶\ntype: dynamic\n---\n" + ("内容" * 900),
        )
        before = bucket.read_bytes()
        report = patrol.patrol(Path(d), datetime(2026, 7, 30, 12, 0, 0))
        after = bucket.read_bytes()

        assert before == after
        assert report["suggestions"]
        assert all(
            entry["action"] and entry["severity"] and entry["reason"]
            for entry in report["suggestions"]
        )
        queue = ReviewQueue(Path(qd) / "review_queue.jsonl")
        first = patrol.enqueue_metabolism_suggestions(report, queue)
        second = patrol.enqueue_metabolism_suggestions(report, queue)
        assert first >= 1
        assert second == 0
        pending = queue.list_pending("metabolism")
        assert len(pending) == first
        assert all(entry["reason"] for entry in pending)


def test_patrol_cli_rejects_apply_before_reading_target(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["patrol.py", "--apply", "--buckets", "/definitely/not/a/vault"],
    )
    with pytest.raises(SystemExit, match="严格只读"):
        patrol.main()


def test_patrol_cli_prefers_configured_vault_over_stale_environment(
    monkeypatch,
    tmp_path,
):
    vault = tmp_path / "vault"
    _write(
        vault,
        "dynamic/one.md",
        "---\nid: one\nname: One\ntype: dynamic\n---\nbody",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"buckets_dir: {vault}\nfact_slots:\n  registry: {{}}\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.md"
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", "/stale/empty/mount")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "patrol.py",
            "--config",
            str(config_path),
            "--out",
            str(report_path),
        ],
    )

    patrol.main()

    assert "桶总数：**1**" in report_path.read_text(encoding="utf-8")


def test_patrol_reports_missing_and_zero_dim_vectors_without_writing(tmp_path):
    vault = tmp_path / "vault"
    missing = _write(
        vault,
        "dynamic/missing.md",
        "---\nid: missing\nname: Missing\ntype: dynamic\n"
        "recorded_at: 2026-07-30T08:00:00+08:00\n---\n正文存在\n",
    )
    zero = _write(
        vault,
        "dynamic/zero.md",
        "---\nid: zero\nname: Zero\ntype: dynamic\n"
        "recorded_at: 2026-07-30T09:00:00+08:00\n---\n正文存在\n",
    )
    healthy = _write(
        vault,
        "dynamic/healthy.md",
        "---\nid: healthy\nname: Healthy\ntype: dynamic\n"
        "recorded_at: 2026-07-30T10:00:00+08:00\n---\n正文存在\n",
    )
    fts_only = _write(
        vault,
        "dynamic/fts.md",
        "---\nid: fts\nname: FTS\ntype: dynamic\nvector_policy: fts_only\n"
        "recorded_at: 2026-07-30T11:00:00+08:00\n---\n明确不需要向量\n",
    )
    db = _write_embedding_db(
        vault / "embeddings.db",
        [("zero", [[]]), ("healthy", [[0.1, 0.2]])],
    )
    bucket_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (missing, zero, healthy, fts_only)
    }
    db_hash = hashlib.sha256(db.read_bytes()).hexdigest()

    report = patrol.patrol(vault, datetime(2026, 7, 30, 12, 0, 0))

    assert report["vector_audit"]["status"] == "ok"
    assert report["vector_audit"]["scanned_bodies"] == 3
    assert [
        (item["id"], item["reason"])
        for item in report["curated_without_vector"]
    ] == [("missing", "missing_vector"), ("zero", "zero_dimension")]
    suggestion = next(
        item
        for item in report["suggestions"]
        if item["check"] == "curated_without_vector"
    )
    assert suggestion["bucket_ids"] == ["missing", "zero"]
    assert "curated_without_vector" in patrol.render_md(
        report, vault, datetime(2026, 7, 30, 12, 0, 0)
    )
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (missing, zero, healthy, fts_only)
    } == bucket_hashes
    assert hashlib.sha256(db.read_bytes()).hexdigest() == db_hash


def test_vector_reconciliation_supports_incremental_recorded_at_window(tmp_path):
    vault = tmp_path / "vault"
    _write(
        vault,
        "dynamic/old.md",
        "---\nid: old\ntype: dynamic\n"
        "recorded_at: 2026-07-29T23:59:59+08:00\n---\n旧正文\n",
    )
    _write(
        vault,
        "dynamic/new.md",
        "---\nid: new\ntype: dynamic\n"
        "recorded_at: 2026-07-30T00:00:00+08:00\n---\n新正文\n",
    )
    _write_embedding_db(vault / "embeddings.db", [])

    report = patrol.patrol(
        vault,
        datetime(2026, 7, 30, 12, 0, 0),
        vector_since=datetime(2026, 7, 30, 0, 0, 0),
    )

    assert [item["id"] for item in report["curated_without_vector"]] == ["new"]
    assert report["vector_audit"]["since"] == "2026-07-30T00:00:00"


def test_zero_deposition_alerts_after_three_active_days_without_buckets(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_embedding_db(vault / "embeddings.db", [])
    report = patrol.patrol(
        vault,
        datetime(2026, 8, 1, 23, 59, 0),
        activity_summary=_activity_summary({
            "2026-07-30": 4,
            "2026-07-31": 2,
            "2026-08-01": 3,
        }),
        monitor_start_date=date(2026, 7, 30),
    )

    monitor = report["zero_deposition"]
    assert monitor["status"] == "alert"
    assert monitor["streak_days"] == 3
    assert monitor["streak_start"] == "2026-07-30"
    assert monitor["streak_end"] == "2026-08-01"
    assert "纯寒暄" in monitor["note"]
    rendered = patrol.render_md(
        report,
        vault,
        datetime(2026, 8, 1, 23, 59, 0),
    )
    assert "连续零沉淀监控" in rendered
    assert "状态：**alert**" in rendered
    assert "纯寒暄" in rendered
    suggestion = next(
        item
        for item in report["suggestions"]
        if item["check"] == "zero_deposition_with_activity"
    )
    assert suggestion["severity"] == "critical"
    assert suggestion["bucket_ids"] == []


def test_zero_deposition_healthy_controls_do_not_false_alarm(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_embedding_db(vault / "embeddings.db", [])
    # Healthy control A: she was away for one day, so the streak must break.
    away = patrol.patrol(
        vault,
        datetime(2026, 8, 1, 23, 59, 0),
        activity_summary=_activity_summary({
            "2026-07-30": 4,
            "2026-07-31": 0,
            "2026-08-01": 3,
        }),
    )
    assert away["zero_deposition"]["status"] == "healthy"
    assert away["zero_deposition"]["streak_days"] == 1

    # Healthy control B: active every day, but a real bucket landed today.
    bucket = _write(
        vault,
        "dynamic/landed.md",
        "---\nid: landed\ntype: dynamic\n"
        "recorded_at: 2026-08-01T08:00:00+08:00\n---\n已沉淀\n",
    )
    before = bucket.read_bytes()
    landed = patrol.patrol(
        vault,
        datetime(2026, 8, 1, 23, 59, 0),
        activity_summary=_activity_summary({
            "2026-07-30": 4,
            "2026-07-31": 2,
            "2026-08-01": 3,
        }),
    )
    assert landed["zero_deposition"]["status"] == "healthy"
    assert landed["zero_deposition"]["streak_days"] == 0
    assert landed["zero_deposition"]["days"][-1]["new_buckets"] == 1
    assert bucket.read_bytes() == before
