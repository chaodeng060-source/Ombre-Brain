from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

import snapshot_manager as snapshot_module
from snapshot_manager import (
    SnapshotIntegrityError,
    SnapshotLimitError,
    SnapshotManager,
    SnapshotSecurityError,
    SnapshotValidationError,
)


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "buckets"
    backup = tmp_path / "backups"
    (source / "dynamic" / "日常").mkdir(parents=True)
    (source / "archive").mkdir()
    return source, backup


def _make_sqlite(path: Path, value: str = "remembered") -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("CREATE TABLE facts (value TEXT NOT NULL)")
    connection.execute("INSERT INTO facts VALUES (?)", (value,))
    connection.commit()
    return connection


def _read_manifest(snapshot: Path) -> dict:
    return json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))


def _replace_manifest(snapshot: Path, manifest: dict) -> str:
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    (snapshot / "manifest.json").write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_snapshot_hashes_files_and_uses_sqlite_backup_for_wal_data(tmp_path):
    source, backup = _roots(tmp_path)
    body = source / "dynamic" / "日常" / "memory.md"
    body.write_text("---\nid: one\n---\nexact body\n", encoding="utf-8")
    connection = _make_sqlite(source / "embeddings.db")
    try:
        # Keep the WAL connection open. A raw copy of embeddings.db can miss
        # the committed row; SQLite backup must include it.
        wal_path = source / "embeddings.db-wal"
        assert wal_path.exists()

        manager = SnapshotManager(source, backup)
        result = manager.create_snapshot("night-20260728")
    finally:
        connection.close()

    assert result.file_count == 2
    assert not (result.snapshot_path / "files" / "embeddings.db-wal").exists()
    manifest = _read_manifest(result.snapshot_path)
    assert manifest["file_count"] == 2
    assert manifest["total_bytes"] == result.total_bytes
    assert manifest["source_root_sha256"] == hashlib.sha256(
        os.fspath(source).encode("utf-8")
    ).hexdigest()

    entries = {item["path"]: item for item in manifest["files"]}
    assert entries["embeddings.db"]["kind"] == "sqlite"
    assert entries["dynamic/日常/memory.md"]["kind"] == "file"
    for relative, item in entries.items():
        payload = result.snapshot_path / "files" / Path(relative)
        assert payload.stat().st_size == item["size"]
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == item["sha256"]

    with sqlite3.connect(result.snapshot_path / "files" / "embeddings.db") as db:
        assert db.execute("SELECT value FROM facts").fetchall() == [("remembered",)]
    assert manager.verify_snapshot(
        "night-20260728",
        expected_manifest_sha256=result.manifest_sha256,
    ) == result


def test_snapshot_captures_nested_audit_and_pipeline_databases(tmp_path):
    source, backup = _roots(tmp_path)
    (source / ".audit").mkdir()
    (source / ".lmc5").mkdir()
    audit = _make_sqlite(source / ".audit" / "mutations.sqlite3", "audit")
    ledger = _make_sqlite(source / ".lmc5" / "pipeline.sqlite3", "ledger")
    try:
        result = SnapshotManager(source, backup).create_snapshot("nested")
    finally:
        audit.close()
        ledger.close()

    paths = {item.path for item in result.files}
    assert ".audit/mutations.sqlite3" in paths
    assert ".lmc5/pipeline.sqlite3" in paths


def test_lock_directory_is_excluded_but_a_symlink_source_is_rejected(tmp_path):
    source, backup = _roots(tmp_path)
    (source / ".locks").mkdir()
    (source / ".locks" / "bucket.lock").write_bytes(b"\0")
    outside = tmp_path / "outside.txt"
    outside.write_text("do not follow", encoding="utf-8")
    link = source / "dynamic" / "日常" / "link.md"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(SnapshotSecurityError, match="symlink"):
        SnapshotManager(source, backup).create_snapshot("unsafe")
    assert not (backup / "unsafe").exists()
    assert list(backup.glob(".unsafe.*.tmp")) == []


def test_all_known_coordination_lock_paths_are_excluded(tmp_path):
    source, backup = _roots(tmp_path)
    memory = source / "dynamic" / "日常" / "memory.md"
    memory.write_text("remember this", encoding="utf-8")
    (source / ".import_state.lock").write_bytes(b"")
    (source / "review_queue.jsonl.lock").write_bytes(b"")
    (source / ".axis").mkdir()
    (source / ".axis" / "e-shadow.jsonl.lock").write_bytes(b"")
    (source / ".curated-write-locks").mkdir()
    (source / ".curated-write-locks" / "one.lock").write_bytes(b"")

    result = SnapshotManager(source, backup).create_snapshot("without-locks")

    assert {item.path for item in result.files} == {
        "dynamic/日常/memory.md",
    }


def test_runtime_only_entries_are_excluded_and_policy_is_manifested(tmp_path):
    source, backup = _roots(tmp_path)
    memory = source / "dynamic" / "日常" / "memory.md"
    memory.write_text("remember this", encoding="utf-8")
    for name in snapshot_module._EXCLUDED_ROOT_FILES:
        (source / name).write_bytes(b"runtime-only")
    for name in snapshot_module._EXCLUDED_ROOT_DIRECTORIES:
        runtime_dir = source / name
        runtime_dir.mkdir()
        (runtime_dir / "state.json").write_text("runtime-only", encoding="utf-8")

    result = SnapshotManager(source, backup).create_snapshot("without-runtime")
    manifest = _read_manifest(result.snapshot_path)

    assert {item.path for item in result.files} == {
        "dynamic/日常/memory.md",
    }
    assert manifest["exclusion_policy"] == (
        snapshot_module._manifest_exclusion_policy()
    )


def test_manifest_exclusion_policy_is_strictly_anchored(tmp_path):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "remember this",
        encoding="utf-8",
    )
    manager = SnapshotManager(source, backup)
    result = manager.create_snapshot("policy-bound")
    manifest = _read_manifest(result.snapshot_path)
    manifest["exclusion_policy"]["root_files"].remove("body_state.json")
    rewritten_digest = _replace_manifest(result.snapshot_path, manifest)

    with pytest.raises(SnapshotIntegrityError, match="exclusion policy"):
        manager.verify_snapshot(
            "policy-bound",
            expected_manifest_sha256=rewritten_digest,
        )


def test_database_suffix_with_non_sqlite_content_fails_closed(tmp_path):
    source, backup = _roots(tmp_path)
    (source / "embeddings.db").write_bytes(b"not a database")

    with pytest.raises(SnapshotIntegrityError, match="not a valid SQLite"):
        SnapshotManager(source, backup).create_snapshot("corrupt")
    assert not (backup / "corrupt").exists()


def test_bounds_abort_without_publishing_partial_snapshot(tmp_path):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "a.md").write_bytes(b"a")
    (source / "dynamic" / "日常" / "b.md").write_bytes(b"b")
    manager = SnapshotManager(source, backup, max_files=1)

    with pytest.raises(SnapshotLimitError, match="max_files|traversal bound"):
        manager.create_snapshot("bounded")
    assert not (backup / "bounded").exists()
    assert list(backup.glob(".bounded.*.tmp")) == []


def test_source_and_backup_roots_must_be_disjoint_and_real(tmp_path):
    source, _backup = _roots(tmp_path)
    with pytest.raises(SnapshotSecurityError, match="disjoint"):
        SnapshotManager(source, source / "backups")
    assert not (source / "backups").exists()

    real_backup = tmp_path / "real-backup"
    real_backup.mkdir()
    link_backup = tmp_path / "link-backup"
    try:
        link_backup.symlink_to(real_backup, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(SnapshotSecurityError, match="symlink"):
        SnapshotManager(source, link_backup)


@pytest.mark.parametrize("snapshot_id", ["../escape", "/absolute", ".", "", "a/b"])
def test_snapshot_id_rejects_path_escape(tmp_path, snapshot_id):
    source, backup = _roots(tmp_path)
    manager = SnapshotManager(source, backup)

    with pytest.raises(SnapshotValidationError):
        manager.create_snapshot(snapshot_id)


def test_restore_isolated_recreates_files_and_sqlite_via_backup(tmp_path):
    source, backup = _roots(tmp_path)
    body = source / "dynamic" / "日常" / "memory.md"
    body.write_text("body", encoding="utf-8")
    database = _make_sqlite(source / "embeddings.db", "vector")
    try:
        manager = SnapshotManager(source, backup)
        snapshot = manager.create_snapshot("restore-me")
    finally:
        database.close()

    destination = tmp_path / "isolated-restore"
    restored = manager.restore_isolated(
        "restore-me",
        destination,
        expected_manifest_sha256=snapshot.manifest_sha256,
    )
    assert restored.destination == destination
    assert restored.file_count == snapshot.file_count
    assert (destination / "dynamic" / "日常" / "memory.md").read_text(
        encoding="utf-8"
    ) == "body"
    with sqlite3.connect(destination / "embeddings.db") as db:
        assert db.execute("SELECT value FROM facts").fetchall() == [("vector",)]


def test_restore_refuses_live_backup_existing_and_symlink_destinations(tmp_path):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "body", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    snapshot = manager.create_snapshot("guarded")

    with pytest.raises(SnapshotSecurityError, match="overlaps"):
        manager.restore_isolated(
            "guarded",
            source / "restored",
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
    with pytest.raises(SnapshotSecurityError, match="overlaps"):
        manager.restore_isolated(
            "guarded",
            backup / "restored",
            expected_manifest_sha256=snapshot.manifest_sha256,
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(SnapshotSecurityError, match="must not already exist"):
        manager.restore_isolated(
            "guarded",
            existing,
            expected_manifest_sha256=snapshot.manifest_sha256,
        )

    real = tmp_path / "real-destination"
    real.mkdir()
    link = tmp_path / "linked-destination"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(SnapshotSecurityError, match="symlink"):
        manager.restore_isolated(
            "guarded",
            link,
            expected_manifest_sha256=snapshot.manifest_sha256,
        )


def test_tampered_payload_fails_verification_and_restore(tmp_path):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "original", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    snapshot = manager.create_snapshot("tampered")
    payload = snapshot.snapshot_path / "files" / "dynamic" / "日常" / "memory.md"
    payload.write_text("changed", encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError, match="hash mismatch"):
        manager.verify_snapshot(
            "tampered",
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
    destination = tmp_path / "must-not-exist"
    with pytest.raises(SnapshotIntegrityError, match="hash mismatch"):
        manager.restore_isolated(
            "tampered",
            destination,
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
    assert not destination.exists()


def test_sqlite_change_during_restore_is_detected_and_not_published(
    tmp_path, monkeypatch
):
    source, backup = _roots(tmp_path)
    database = _make_sqlite(source / "embeddings.db", "original")
    try:
        manager = SnapshotManager(source, backup)
        snapshot = manager.create_snapshot("sqlite-race")
    finally:
        database.close()

    original_backup = manager._backup_sqlite

    def mutate_then_backup(entry, target):
        with sqlite3.connect(entry.source) as db:
            db.execute("INSERT INTO facts VALUES ('tampered')")
            db.commit()
        return original_backup(entry, target)

    monkeypatch.setattr(manager, "_backup_sqlite", mutate_then_backup)
    destination = tmp_path / "race-restore"
    with pytest.raises(SnapshotIntegrityError, match="changed during restore"):
        manager.restore_isolated(
            "sqlite-race",
            destination,
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
    assert not destination.exists()


def test_manifest_traversal_and_unlisted_files_fail_closed(tmp_path):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "original", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    snapshot = manager.create_snapshot("manifest-guard")
    manifest_path = snapshot.snapshot_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../escape"
    tampered_digest = _replace_manifest(snapshot.snapshot_path, manifest)

    with pytest.raises(SnapshotIntegrityError, match="escapes"):
        manager.verify_snapshot(
            "manifest-guard",
            expected_manifest_sha256=tampered_digest,
        )

    second = manager.create_snapshot("extra-file")
    extra = second.snapshot_path / "files" / "unlisted.txt"
    extra.write_text("not in manifest", encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError, match="exactly match"):
        manager.verify_snapshot(
            "extra-file",
            expected_manifest_sha256=second.manifest_sha256,
        )


def test_manifest_digest_anchor_rejects_rewritten_payload_and_manifest(tmp_path):
    source, backup = _roots(tmp_path)
    body = source / "dynamic" / "日常" / "memory.md"
    body.write_text("original", encoding="utf-8")
    manager = SnapshotManager(source, backup)
    snapshot = manager.create_snapshot("anchored")

    payload = snapshot.snapshot_path / "files" / "dynamic" / "日常" / "memory.md"
    payload.write_text("attacker rewrite", encoding="utf-8")
    manifest = _read_manifest(snapshot.snapshot_path)
    entry = manifest["files"][0]
    entry["size"] = payload.stat().st_size
    entry["sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    rewritten_digest = _replace_manifest(snapshot.snapshot_path, manifest)
    assert rewritten_digest != snapshot.manifest_sha256

    with pytest.raises(SnapshotIntegrityError, match="manifest digest mismatch"):
        manager.verify_snapshot(
            "anchored",
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
    destination = tmp_path / "anchor-restore"
    with pytest.raises(SnapshotIntegrityError, match="manifest digest mismatch"):
        manager.restore_isolated(
            "anchored",
            destination,
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
    assert not destination.exists()


def test_snapshot_root_rejects_unlisted_top_level_entry(tmp_path):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "body", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    snapshot = manager.create_snapshot("root-layout")
    (snapshot.snapshot_path / "rogue").write_text("extra", encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError, match="only manifest"):
        manager.verify_snapshot(
            "root-layout",
            expected_manifest_sha256=snapshot.manifest_sha256,
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX filename and mode semantics")
def test_create_rejects_unverifiable_filename_and_special_mode(tmp_path):
    source, backup = _roots(tmp_path)
    bad_name = source / "dynamic" / "日常" / "bad\\name.md"
    bad_name.write_text("body", encoding="utf-8")
    manager = SnapshotManager(source, backup)
    with pytest.raises(SnapshotIntegrityError, match="invalid file path"):
        manager.create_snapshot("bad-name")
    assert not (backup / "bad-name").exists()

    bad_name.unlink()
    special = source / "dynamic" / "日常" / "special.md"
    special.write_text("body", encoding="utf-8")
    special.chmod(0o4755)
    with pytest.raises(SnapshotSecurityError, match="special permission"):
        manager.create_snapshot("special-mode")
    assert not (backup / "special-mode").exists()


def test_create_rejects_oversized_manifest_before_publication(
    tmp_path, monkeypatch
):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "body", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    monkeypatch.setattr(snapshot_module, "_MAX_MANIFEST_BYTES", 64)

    with pytest.raises(SnapshotLimitError, match="manifest is too large"):
        manager.create_snapshot("large-manifest")
    assert not (backup / "large-manifest").exists()
    assert list(backup.glob(".large-manifest.*.tmp")) == []


def test_manifest_is_bound_to_source_root_and_strict_accounting(tmp_path):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "original", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    first = manager.create_snapshot("source-bound")
    manifest_path = first.snapshot_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_root_sha256"] = "0" * 64
    tampered_digest = _replace_manifest(first.snapshot_path, manifest)
    with pytest.raises(SnapshotIntegrityError, match="different source"):
        manager.verify_snapshot(
            "source-bound",
            expected_manifest_sha256=tampered_digest,
        )

    second = manager.create_snapshot("strict-count")
    second_manifest_path = second.snapshot_path / "manifest.json"
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    second_manifest["file_count"] = True
    tampered_digest = _replace_manifest(second.snapshot_path, second_manifest)
    with pytest.raises(SnapshotIntegrityError, match="file_count"):
        manager.verify_snapshot(
            "strict-count",
            expected_manifest_sha256=tampered_digest,
        )


def test_snapshot_payload_symlink_is_rejected_before_restore(tmp_path):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "original", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    snapshot = manager.create_snapshot("payload-link")
    payload = snapshot.snapshot_path / "files" / "dynamic" / "日常" / "memory.md"
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    payload.unlink()
    try:
        payload.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(SnapshotSecurityError, match="symlink"):
        manager.verify_snapshot(
            "payload-link",
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
    destination = tmp_path / "no-restore"
    with pytest.raises(SnapshotSecurityError, match="symlink"):
        manager.restore_isolated(
            "payload-link",
            destination,
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
    assert not destination.exists()


def test_injected_copy_failure_cleans_staging_and_never_publishes(
    tmp_path, monkeypatch
):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "body", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)

    def fail_copy(_entry, _target):
        raise OSError("injected")

    monkeypatch.setattr(manager, "_copy_regular", fail_copy)
    with pytest.raises(OSError, match="injected"):
        manager.create_snapshot("atomic")
    assert not (backup / "atomic").exists()
    assert list(backup.glob(".atomic.*.tmp")) == []


def test_snapshot_publication_never_replaces_racing_empty_directory(
    tmp_path,
    monkeypatch,
):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "body",
        encoding="utf-8",
    )
    manager = SnapshotManager(source, backup)
    destination = backup / "racing-snapshot"
    original_publish = snapshot_module._rename_no_replace
    racing_identity = None

    def create_racer_then_publish(staging, target):
        nonlocal racing_identity
        target.mkdir()
        target_stat = target.stat()
        racing_identity = (target_stat.st_dev, target_stat.st_ino)
        return original_publish(staging, target)

    monkeypatch.setattr(
        snapshot_module,
        "_rename_no_replace",
        create_racer_then_publish,
    )
    with pytest.raises(SnapshotIntegrityError, match="raced"):
        manager.create_snapshot("racing-snapshot")

    destination_stat = destination.stat()
    assert (destination_stat.st_dev, destination_stat.st_ino) == racing_identity
    assert list(backup.glob(".racing-snapshot.*.tmp")) == []


def test_restore_publication_never_replaces_racing_empty_directory(
    tmp_path,
    monkeypatch,
):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "body",
        encoding="utf-8",
    )
    manager = SnapshotManager(source, backup)
    snapshot = manager.create_snapshot("restore-race-source")
    destination = tmp_path / "racing-restore"
    original_publish = snapshot_module._rename_no_replace
    racing_identity = None

    def create_racer_then_publish(staging, target):
        nonlocal racing_identity
        target.mkdir()
        target_stat = target.stat()
        racing_identity = (target_stat.st_dev, target_stat.st_ino)
        return original_publish(staging, target)

    monkeypatch.setattr(
        snapshot_module,
        "_rename_no_replace",
        create_racer_then_publish,
    )
    with pytest.raises(SnapshotSecurityError, match="appeared"):
        manager.restore_isolated(
            "restore-race-source",
            destination,
            expected_manifest_sha256=snapshot.manifest_sha256,
        )

    destination_stat = destination.stat()
    assert (destination_stat.st_dev, destination_stat.st_ino) == racing_identity
    assert list(
        destination.parent.glob(f".{destination.name}.restore.*.tmp")
    ) == []


def test_cleanup_refuses_external_temp_lookalike(tmp_path):
    source, backup = _roots(tmp_path)
    manager = SnapshotManager(source, backup)
    outside = tmp_path / "outside"
    outside.mkdir()
    lookalike = outside / ".atomic.attacker.tmp"
    lookalike.mkdir()
    marker = lookalike / "keep.txt"
    marker.write_text("must survive", encoding="utf-8")

    with pytest.raises(SnapshotSecurityError, match="unowned staging"):
        manager._cleanup_private_tree(
            lookalike,
            owned_parent=backup,
            name_prefix=".atomic.",
        )
    assert marker.read_text(encoding="utf-8") == "must survive"


def test_post_rename_fsync_failure_removes_formal_snapshot_path(
    tmp_path, monkeypatch
):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "body", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    original_fsync = snapshot_module._fsync_directory
    parent_sync_states = []

    def fail_after_publish(path):
        if Path(path) == backup:
            published_exists = (backup / "fsync-fail").exists()
            parent_sync_states.append(published_exists)
            if published_exists:
                raise OSError("injected post-rename fsync failure")
        return original_fsync(path)

    monkeypatch.setattr(snapshot_module, "_fsync_directory", fail_after_publish)
    with pytest.raises(OSError, match="post-rename"):
        manager.create_snapshot("fsync-fail")
    assert not (backup / "fsync-fail").exists()
    assert list(backup.glob(".fsync-fail.*.tmp")) == []
    assert parent_sync_states == [True, False]


def test_restore_post_rename_fsync_failure_removes_destination(
    tmp_path, monkeypatch
):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "body", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    snapshot = manager.create_snapshot("restore-fsync")
    destination = tmp_path / "restore-fsync-destination"
    original_fsync = snapshot_module._fsync_directory
    parent_sync_states = []

    def fail_after_publish(path):
        if Path(path) == destination.parent:
            published_exists = destination.exists()
            parent_sync_states.append(published_exists)
            if published_exists:
                raise OSError("injected restore fsync failure")
        return original_fsync(path)

    monkeypatch.setattr(snapshot_module, "_fsync_directory", fail_after_publish)
    with pytest.raises(OSError, match="restore fsync"):
        manager.restore_isolated(
            "restore-fsync",
            destination,
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
    assert not destination.exists()
    assert list(
        destination.parent.glob(f".{destination.name}.restore.*.tmp")
    ) == []
    assert parent_sync_states == [True, False]


def test_source_tree_change_during_snapshot_is_not_published(tmp_path, monkeypatch):
    source, backup = _roots(tmp_path)
    (source / "dynamic" / "日常" / "memory.md").write_text(
        "body", encoding="utf-8"
    )
    manager = SnapshotManager(source, backup)
    original_copy = manager._copy_regular
    injected = False

    def copy_then_add(entry, target):
        nonlocal injected
        result = original_copy(entry, target)
        if not injected:
            injected = True
            (source / "dynamic" / "日常" / "late.md").write_text(
                "arrived during snapshot", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(manager, "_copy_regular", copy_then_add)
    with pytest.raises(SnapshotIntegrityError, match="source tree changed"):
        manager.create_snapshot("moving-source")
    assert not (backup / "moving-source").exists()
    assert list(backup.glob(".moving-source.*.tmp")) == []
