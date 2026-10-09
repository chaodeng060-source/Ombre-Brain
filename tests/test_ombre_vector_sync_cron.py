from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


def _load_sync_module(monkeypatch):
    monkeypatch.setenv("OMBRE_PG_RECALL_DSN", "postgresql://test.invalid/test")
    source = Path(__file__).resolve().parents[1] / "ombre_vector_sync_cron.py"
    spec = importlib.util.spec_from_file_location("ombre_vector_sync_cron_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vector_sync_inventory_skips_merged_bucket_directories(
    tmp_path: Path, monkeypatch
) -> None:
    buckets = tmp_path / "buckets"
    active = buckets / "permanent" / "active_aabbccddeeff.md"
    merged_root = buckets / ".merged-2026-10-01" / "permanent" / "old_112233445566.md"
    merged_nested = (
        buckets
        / "dynamic"
        / ".merged-rollback"
        / "older_778899aabbcc.md"
    )
    for path in (active, merged_root, merged_nested):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    module = _load_sync_module(monkeypatch)
    monkeypatch.setattr(module, "BUCKETS_DIR", buckets)

    assert module.bucket_ids_on_disk() == {"aabbccddeeff"}


class _FakePg:
    def __init__(self, vector_rows, body_ids=()):
        self.vector_rows = list(vector_rows)
        self.body_ids = set(body_ids)
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1


class _FakeCursor:
    def __init__(self, pg):
        self.pg = pg
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params=None):
        sql = " ".join(statement.split()).lower()
        if sql.startswith(
            "select bucket_id, max(source_updated_at) from ombre_vectors"
        ):
            latest = {}
            for bucket_id, _segment, _vector, source_updated_at in self.pg.vector_rows:
                latest[bucket_id] = max(
                    latest.get(bucket_id, ""), source_updated_at
                )
            self.result = list(latest.items())
        elif sql.startswith("delete from ombre_vectors where bucket_id = any"):
            removed = set(params[0])
            self.pg.vector_rows = [
                row for row in self.pg.vector_rows if row[0] not in removed
            ]
        elif sql.startswith("delete from ombre_bodies where bucket_id = any"):
            self.pg.body_ids.difference_update(params[0])
        elif sql.startswith("delete from ombre_vectors where bucket_id="):
            bucket_id = params[0]
            self.pg.vector_rows = [
                row for row in self.pg.vector_rows if row[0] != bucket_id
            ]
        elif sql.startswith("insert into ombre_vectors"):
            bucket_id, segment_idx, vector, source_updated_at = params
            self.pg.vector_rows.append(
                (bucket_id, segment_idx, vector, source_updated_at)
            )
        elif sql.startswith("select count(distinct bucket_id) from ombre_vectors"):
            self.result = (len({row[0] for row in self.pg.vector_rows}),)
        else:
            raise AssertionError(f"Unexpected SQL: {statement}")

    def fetchall(self):
        return self.result

    def fetchone(self):
        return self.result


def _prepare_sync_fixture(module, buckets: Path, *, live_ids: tuple[str, ...]):
    buckets.mkdir(parents=True, exist_ok=True)
    for bucket_id in live_ids:
        (buckets / f"memory_{bucket_id}.md").write_text(
            "synthetic bucket\n", encoding="utf-8"
        )
    module.BUCKETS_DIR = buckets
    module.EMBEDDINGS_DB = buckets / "embeddings.db"
    with sqlite3.connect(module.EMBEDDINGS_DB) as connection:
        connection.execute(
            "CREATE TABLE embeddings "
            "(bucket_id TEXT PRIMARY KEY, embedding TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )


def test_execute_syncs_new_and_stale_vectors_and_removes_ghosts(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_sync_module(monkeypatch)
    new_id = "aaaaaaaaaaaa"
    stale_id = "bbbbbbbbbbbb"
    ghost_id = "cccccccccccc"
    no_embedding_id = "dddddddddddd"
    buckets = tmp_path / "buckets"
    _prepare_sync_fixture(
        module,
        buckets,
        live_ids=(new_id, stale_id, no_embedding_id),
    )
    with sqlite3.connect(module.EMBEDDINGS_DB) as connection:
        connection.executemany(
            "INSERT INTO embeddings VALUES (?, ?, ?)",
            [
                (new_id, json.dumps([[0.25] * 1024]), "2026-10-08T00:00:00+00:00"),
                (stale_id, json.dumps([[0.5] * 1024]), "2026-10-09T00:00:00+00:00"),
                (ghost_id, json.dumps([[0.75] * 1024]), "2026-10-09T00:00:00+00:00"),
            ],
        )

    pg = _FakePg(
        [
            (stale_id, 0, "[0.0]", "2026-10-01T00:00:00+00:00"),
            (ghost_id, 0, "[0.0]", "2026-10-01T00:00:00+00:00"),
        ],
        body_ids={ghost_id},
    )
    monkeypatch.setattr(module.psycopg, "connect", lambda *_args, **_kwargs: pg)
    monkeypatch.setattr(module, "EXECUTE", True)

    assert module.main() == 0

    assert {row[0] for row in pg.vector_rows} == {new_id, stale_id}
    assert len(pg.vector_rows) == 2
    assert all(
        row[3]
        in {"2026-10-08T00:00:00+00:00", "2026-10-09T00:00:00+00:00"}
        for row in pg.vector_rows
    )
    assert pg.body_ids == set()
    with sqlite3.connect(module.EMBEDDINGS_DB) as connection:
        remaining = {
            row[0] for row in connection.execute("SELECT bucket_id FROM embeddings")
        }
    assert remaining == {new_id, stale_id}


def test_dry_run_reports_sync_work_without_mutating_sources(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_sync_module(monkeypatch)
    live_id = "aaaaaaaaaaaa"
    buckets = tmp_path / "buckets"
    _prepare_sync_fixture(module, buckets, live_ids=(live_id,))
    with sqlite3.connect(module.EMBEDDINGS_DB) as connection:
        connection.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?)",
            (live_id, json.dumps([[0.25] * 1024]), "2026-10-08T00:00:00+00:00"),
        )
    pg = _FakePg([])
    monkeypatch.setattr(module.psycopg, "connect", lambda *_args, **_kwargs: pg)
    monkeypatch.setattr(module, "EXECUTE", False)

    assert module.main() == 0

    assert pg.vector_rows == []
    assert pg.commits == 0
    assert "dry-run" in capsys.readouterr().out
    with sqlite3.connect(module.EMBEDDINGS_DB) as connection:
        assert connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1


def test_empty_bucket_mount_fails_before_database_access(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_sync_module(monkeypatch)
    module.BUCKETS_DIR = tmp_path / "empty-buckets"
    monkeypatch.setattr(
        module.psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty mount must not connect to PostgreSQL")
        ),
    )

    assert module.main() == 2
    assert "FATAL" in capsys.readouterr().err
