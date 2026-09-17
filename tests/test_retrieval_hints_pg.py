"""Real local PG transactions; no NAS, provider or existing database access."""
from pathlib import Path
import subprocess
import tempfile

import pytest

from retrieval_hints import Generation
from retrieval_hints_storage import HintsStore, PostgresHints
from tests.test_retrieval_hints import payload
from tests.test_retrieval_hints_storage import bucket


@pytest.fixture(scope="module")
def pg_dsn():
    pg = Path("/usr/lib/postgresql/17/bin")
    if not (pg / "initdb").exists():
        pytest.skip("local PG server binaries unavailable")
    with tempfile.TemporaryDirectory(prefix="hints-pg-") as root:
        data = Path(root) / "data"
        subprocess.run([str(pg / "initdb"), "-D", str(data), "-A", "trust", "-U", "test",
                        "--no-locale", "-E", "UTF8"], check=True, capture_output=True, text=True)
        subprocess.run([str(pg / "pg_ctl"), "-D", str(data), "-l", str(Path(root) / "pg.log"),
                        "-o", f"-k {root} -c listen_addresses='' -c max_connections=6 -c shared_buffers=16MB",
                        "-w", "start"], check=True, capture_output=True, text=True)
        try:
            yield f"host={root} dbname=postgres user=test"
        finally:
            subprocess.run([str(pg / "pg_ctl"), "-D", str(data), "-m", "fast", "-w", "stop"],
                           check=True, capture_output=True, text=True)


def test_real_dual_publication_preserves_body_vector_tables(tmp_path, pg_dsn):
    psycopg = pytest.importorskip("psycopg")
    store = HintsStore(tmp_path / "hints")
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute("CREATE TABLE ombre_bodies (bucket_id TEXT PRIMARY KEY, body TEXT)")
        conn.execute("CREATE TABLE ombre_vectors (bucket_id TEXT PRIMARY KEY, embedding TEXT)")
        conn.execute("INSERT INTO ombre_bodies VALUES ('synthetic-a','unchanged-body')")
        conn.execute("INSERT INTO ombre_vectors VALUES ('synthetic-a','unchanged-vector')")
        peer = PostgresHints(conn)
        peer.initialize()
        version = store.stage(bucket(), Generation("ok", payload()))
        store.publish("synthetic-a", version, peer, bucket)
        assert peer.get("synthetic-a", version)["payload"] == store.lookup(bucket())["payload"]
        assert conn.execute("SELECT * FROM ombre_bodies").fetchall() == [("synthetic-a", "unchanged-body")]
        assert conn.execute("SELECT * FROM ombre_vectors").fetchall() == [("synthetic-a", "unchanged-vector")]
        assert conn.execute("SELECT search_tsv IS NOT NULL FROM ombre_retrieval_hints").fetchone()[0]
        store.rollback_pointer("synthetic-a", expected=version, previous=None)
        assert store.lookup(bucket()) is None
        assert conn.execute("SELECT body FROM ombre_bodies").fetchone()[0] == "unchanged-body"


def test_pg_disconnect_keeps_outbox_then_resumes(tmp_path, pg_dsn):
    psycopg = pytest.importorskip("psycopg")
    store = HintsStore(tmp_path / "hints")
    version = store.stage(bucket(), Generation("ok", payload()))
    conn = psycopg.connect(pg_dsn, autocommit=True)
    peer = PostgresHints(conn)
    conn.close()
    with pytest.raises(psycopg.OperationalError):
        store.publish("synthetic-a", version, peer, bucket)
    assert store.lookup(bucket()) is None and len(store.pending()) == 1
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        store.publish("synthetic-a", version, PostgresHints(conn), bucket)
    assert store.lookup(bucket())["version"] == version and store.pending() == []


def test_pg_ready_without_sqlite_pointer_cannot_export(tmp_path, pg_dsn, monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    store = HintsStore(tmp_path / "hints")
    version = store.stage(bucket(), Generation("ok", payload()))
    original = store._connect
    def fail_sqlite(*, write=False):
        if write:
            raise OSError("simulated process interruption before pointer commit")
        return original()
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        peer = PostgresHints(conn)
        monkeypatch.setattr(store, "_connect", fail_sqlite)
        with pytest.raises(OSError):
            store.publish("synthetic-a", version, peer, bucket)
        assert peer.get("synthetic-a", version)["ready"]
        assert store.manifest([bucket()])["records"] == []
        monkeypatch.setattr(store, "_connect", original)
        store.publish("synthetic-a", version, peer, bucket)
        assert len(store.manifest([bucket()])["records"]) == 1
