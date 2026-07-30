#!/usr/bin/env python3
"""Isolated acceptance smoke for the read-only amnesia monitor."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import patrol  # noqa: E402
from conversation_activity import SCHEMA, TIMEZONE_NAME  # noqa: E402


def _write_bucket(vault: Path, bucket_id: str, recorded_at: str, body: str) -> Path:
    path = vault / "dynamic" / f"{bucket_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "---\n"
            f"id: {bucket_id}\n"
            f"name: {bucket_id}\n"
            "type: dynamic\n"
            f"recorded_at: {recorded_at}\n"
            "---\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )
    return path


def _embedding_db(vault: Path, rows: list[tuple[str, object]]) -> Path:
    path = vault / "embeddings.db"
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
            "INSERT INTO embeddings VALUES (?, ?, ?)",
            [
                (bucket_id, json.dumps(vector), "2026-07-30T12:00:00+08:00")
                for bucket_id, vector in rows
            ],
        )
    return path


def _activity(**counts: int) -> dict:
    return {
        "schema": SCHEMA,
        "timezone": TIMEZONE_NAME,
        "start_date": "2026-07-30",
        "daily_user_messages": {
            key.replace("_", "-"): value
            for key, value in counts.items()
        },
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="ombre-amnesia-smoke-",
        dir="/tmp",
    ) as raw_root:
        root = Path(raw_root)

        vector_vault = root / "vector-case"
        missing = _write_bucket(
            vector_vault,
            "missing",
            "2026-07-30T08:00:00+08:00",
            "正文已落盘但没有向量",
        )
        healthy = _write_bucket(
            vector_vault,
            "healthy",
            "2026-07-30T09:00:00+08:00",
            "正文与向量都完整",
        )
        vector_db = _embedding_db(
            vector_vault,
            [("healthy", [[0.1, 0.2]])],
        )
        before = {path.name: _sha(path) for path in (missing, healthy, vector_db)}
        vector_report = patrol.patrol(
            vector_vault,
            datetime(2026, 7, 30, 12, 0, 0),
        )
        findings = vector_report["curated_without_vector"]
        assert [(item["id"], item["reason"]) for item in findings] == [
            ("missing", "missing_vector")
        ]
        assert before == {
            path.name: _sha(path)
            for path in (missing, healthy, vector_db)
        }

        zero_vault = root / "zero-case"
        zero_vault.mkdir()
        _embedding_db(zero_vault, [])
        alert = patrol.patrol(
            zero_vault,
            datetime(2026, 8, 1, 23, 59, 0),
            activity_summary=_activity(
                **{"2026_07_30": 4, "2026_07_31": 2, "2026_08_01": 3}
            ),
        )
        assert alert["zero_deposition"]["status"] == "alert"
        assert alert["zero_deposition"]["streak_days"] == 3

        away = patrol.patrol(
            zero_vault,
            datetime(2026, 8, 1, 23, 59, 0),
            activity_summary=_activity(
                **{"2026_07_30": 4, "2026_07_31": 0, "2026_08_01": 3}
            ),
        )
        assert away["zero_deposition"]["status"] == "healthy"

        landed = _write_bucket(
            zero_vault,
            "landed",
            "2026-08-01T08:00:00+08:00",
            "当天已有真实沉淀",
        )
        landed_before = _sha(landed)
        healthy_report = patrol.patrol(
            zero_vault,
            datetime(2026, 8, 1, 23, 59, 0),
            activity_summary=_activity(
                **{"2026_07_30": 4, "2026_07_31": 2, "2026_08_01": 3}
            ),
        )
        assert healthy_report["zero_deposition"]["status"] == "healthy"
        assert healthy_report["zero_deposition"]["days"][-1]["new_buckets"] == 1
        assert _sha(landed) == landed_before

        print(json.dumps({
            "vector_case": {
                "status": "caught",
                "bucket_ids": [item["id"] for item in findings],
            },
            "zero_deposition_case": {
                "status": alert["zero_deposition"]["status"],
                "streak_days": alert["zero_deposition"]["streak_days"],
            },
            "healthy_controls": {
                "away_day": away["zero_deposition"]["status"],
                "bucket_landed": healthy_report["zero_deposition"]["status"],
            },
            "read_only_hashes_unchanged": True,
            "production_data_touched": False,
        }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
