"""Scoped rollback against real, isolated SQLite and PostgreSQL."""
import copy
import json

import pytest

from retrieval_hints import Generation
from retrieval_hints_storage import HintsStore, PostgresHints, VersionConflict
from tests.test_retrieval_hints import payload
from tests.test_retrieval_hints_storage import bucket
from tests.test_retrieval_hints_pg import pg_dsn
from tools.retrieval_hints_rollback import capture, restore, write_receipt


@pytest.fixture
def setup(tmp_path, pg_dsn):
    import psycopg
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        # Each test has its own schema, not an existing application database.
        schema = "test_" + tmp_path.name.replace("-", "_")
        conn.execute(f'CREATE SCHEMA "{schema}"')
        conn.execute(f'SET search_path TO "{schema}"')
        store = HintsStore(tmp_path / "hints")
        peer = PostgresHints(conn)
        yield store, peer


def publish(store, peer, *, variant=0, bid="synthetic-a"):
    current = {**bucket(), "id": bid}
    value = payload()
    if variant:
        value["aliases"] = []
    if variant == 2:
        value["where"] = []
    version = store.stage(current, Generation("ok", value))
    store.publish(bid, version, peer, lambda: current)
    return version


def test_capture_missing_tables_is_read_only(setup, tmp_path):
    store, peer = setup
    before = capture(store, peer, ["synthetic-a"])
    assert not store.path.exists()
    assert before["sqlite"] == {"versions": [], "published": []}
    assert before["postgres"] == []
    target = tmp_path / "before.json"
    write_receipt(target, before)
    assert target.stat().st_mode & 0o777 == 0o600
    assert json.loads(target.read_text()) == before
    with pytest.raises(FileExistsError):
        write_receipt(target, before)


def test_restore_new_rows_retains_ledger_and_is_repeatable(setup):
    store, peer = setup
    before = capture(store, peer, ["synthetic-a"])
    peer.initialize()
    peer.connection.execute("CREATE TABLE ombre_bodies (body TEXT)")
    peer.connection.execute("CREATE TABLE ombre_vectors (embedding TEXT)")
    peer.connection.execute("INSERT INTO ombre_bodies VALUES ('original body')")
    peer.connection.execute("INSERT INTO ombre_vectors VALUES ('original vector')")
    key = store.register(bucket())
    assert store.reserve(key)
    version = publish(store, peer)
    store.finish_job(key, Generation("ok", payload(), {"output_tokens": 3}))
    after = capture(store, peer, ["synthetic-a"])
    jobs = store.jobs()
    assert restore(store, peer, before, after, apply=False)["status"] == "ready"
    assert capture(store, peer, ["synthetic-a"]) == after
    restore(store, peer, before, after)
    assert store.lookup(bucket()) is None
    assert not peer.get("synthetic-a", version)["ready"]
    assert store.jobs() == jobs
    assert store.pending() == []
    assert store._version("synthetic-a", version)["payload"] == payload()
    assert restore(store, peer, before, after)["status"] == "restored"
    assert peer.connection.execute("SELECT body FROM ombre_bodies").fetchone()[0] == "original body"
    assert peer.connection.execute("SELECT embedding FROM ombre_vectors").fetchone()[0] == "original vector"


def test_new_dictionary_is_removed_but_recoverable_in_receipt(setup):
    store, peer = setup
    before = capture(store, peer, ["synthetic-a"])
    store.directory.mkdir()
    (store.directory / "userdict.txt").write_bytes(b"synthetic 10\n")
    after = capture(store, peer, ["synthetic-a"])
    restore(store, peer, before, after)
    assert not (store.directory / "userdict.txt").exists()
    import base64
    assert base64.b64decode(after["dictionary"]) == b"synthetic 10\n"


def test_cli_refuses_enabled_runtime_flags_before_connections(monkeypatch, tmp_path):
    from tools.retrieval_hints_rollback import main
    monkeypatch.setenv("OMBRE_RETRIEVAL_HINTS_ENABLED", "1")
    assert main(["capture", "--buckets-dir", str(tmp_path), "--host-lock", "/not-opened",
                 "--lock-device", "1", "--lock-inode", "2"]) == 2
    assert not (tmp_path / ".retrieval_hints").exists()


def test_restores_previous_publication_dictionary_and_preserves_unrelated(setup):
    store, peer = setup
    peer.initialize()
    old = publish(store, peer)
    dictionary = store.directory / "userdict.txt"
    dictionary.write_bytes(b"old-word 10\n")
    before = capture(store, peer, ["synthetic-a"])
    new = publish(store, peer, variant=1)
    unrelated = publish(store, peer, bid="synthetic-b")
    dictionary.write_bytes(b"new-word 20\n")
    after = capture(store, peer, ["synthetic-a"])
    restore(store, peer, before, after)
    assert store.lookup(bucket())["version"] == old
    assert not peer.get("synthetic-a", new)["ready"]
    assert peer.get("synthetic-b", unrelated)["ready"]
    assert store.lookup({**bucket(), "id": "synthetic-b"})["version"] == unrelated
    assert dictionary.read_bytes() == b"old-word 10\n"


@pytest.mark.parametrize("changed", ["publication", "dictionary"])
def test_later_changes_refuse_before_any_mutation(setup, changed):
    store, peer = setup
    before = capture(store, peer, ["synthetic-a"])
    peer.initialize()
    publish(store, peer)
    after = capture(store, peer, ["synthetic-a"])
    if changed == "publication":
        publish(store, peer, variant=1)
    else:
        (store.directory / "userdict.txt").write_bytes(b"later-owner\n")
    current = capture(store, peer, ["synthetic-a"])
    with pytest.raises(VersionConflict):
        restore(store, peer, before, after)
    assert capture(store, peer, ["synthetic-a"]) == current


def test_pg_done_sqlite_interruption_can_resume(setup, monkeypatch):
    store, peer = setup
    before = capture(store, peer, ["synthetic-a"])
    peer.initialize()
    version = publish(store, peer)
    after = capture(store, peer, ["synthetic-a"])
    import tools.retrieval_hints_rollback as rollback
    original = rollback._restore_sqlite
    monkeypatch.setattr(rollback, "_restore_sqlite", lambda *a: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError):
        restore(store, peer, before, after)
    assert not peer.get("synthetic-a", version)["ready"]
    assert store.lookup(bucket()) is not None
    monkeypatch.setattr(rollback, "_restore_sqlite", original)
    restore(store, peer, before, after)
    assert store.lookup(bucket()) is None


def test_receipt_cannot_switch_target_or_rewrite_existing_payload(setup):
    store, peer = setup
    peer.initialize()
    publish(store, peer)
    before = capture(store, peer, ["synthetic-a"])
    after = copy.deepcopy(before)
    after["store_directory"] += "-other"
    with pytest.raises(ValueError):
        restore(store, peer, before, after)
    after = copy.deepcopy(before)
    after["sqlite"]["versions"][0]["record_json"] = "{}"
    with pytest.raises(VersionConflict):
        restore(store, peer, before, after)
