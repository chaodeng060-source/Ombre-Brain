"""Synthetic inventories preserve physical IDs while counting both alias markers."""

from __future__ import annotations

import sqlite3

import pytest

from memory_inventory import LAYERS, build_inventory


def bucket(root, relative, bucket_id=None, *, markers=None, body="原文逐字保留。"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [] if bucket_id is None else [f"id: '{bucket_id}'"]
    fields.extend(f"{key}: '{value}'" for key, value in (markers or {}).items())
    path.write_text("---\n" + "\n".join(fields) + "\n---\n\n" + body, encoding="utf-8")
    return path


def embeddings(root, *bucket_ids):
    with sqlite3.connect(root / "embeddings.db") as connection:
        connection.execute("CREATE TABLE embeddings (bucket_id TEXT NOT NULL)")
        connection.executemany("INSERT INTO embeddings VALUES (?)", [(key,) for key in bucket_ids])


def test_old_and_exact_aliases_keep_physical_ids_without_counting_independent_records(tmp_path):
    files = [
        bucket(tmp_path, "dynamic/工程/winner.md", "winner"),
        bucket(tmp_path, "dynamic/工程/old.md", "old-id", markers={"duplicate_of": "winner"}),
        bucket(tmp_path, "dynamic/工程/exact.md", "exact-id", markers={"exact_duplicate_of": "winner"}),
    ]
    before = {path: path.read_bytes() for path in files}
    embeddings(tmp_path, "winner", "winner", "old-id", "exact-id", "orphan", "orphan")
    result = build_inventory(tmp_path)
    assert result["schema"] == "ombre.memory-inventory/v1"
    assert result["physical_files"] == result["record_ids"] == 3
    assert result["duplicate_id_files"] == 0
    assert result["total_records"] == result["regular_count"] == 1
    assert result["alias_count"] == 2
    assert result["layers"] == {key: int(key == "dynamic") for key in LAYERS}
    assert result["domains"] == {"工程": 1}
    assert result["index"] == {
        "records_indexed": 1, "missing_regular": 0, "missing_feel": 0,
        "missing_archive": 0, "orphan_rows": 1,
        "by_layer": {key: {"indexed": int(key == "dynamic"), "missing": 0} for key in LAYERS},
    }
    assert result["total_size_kb"] == sum(len(raw) for raw in before.values()) / 1024
    assert {path: path.read_bytes() for path in files} == before
    assert build_inventory(tmp_path) == result


def test_exact_alias_exclusion_applies_to_all_existing_layers(tmp_path):
    for layer, directory in LAYERS.items():
        bucket(tmp_path, f"{directory}/测试/{layer}.md", layer)
        bucket(tmp_path, f"{directory}/测试/{layer}-copy.md", layer + "-copy",
               markers={"exact_duplicate_of": layer})
    result = build_inventory(tmp_path)
    assert result["physical_files"] == result["record_ids"] == 10
    assert result["total_records"] == result["alias_count"] == 5
    assert result["regular_count"] == 2
    assert result["layers"] == dict.fromkeys(LAYERS, 1)
    assert result["domains"] == {"测试": 5}
    assert result["index"]["missing_regular"] == 2
    assert result["index"]["missing_feel"] == result["index"]["missing_archive"] == 1
    assert result["index"]["by_layer"] == {key: {"indexed": 0, "missing": 1} for key in LAYERS}
    assert not (tmp_path / "embeddings.db").exists()


@pytest.mark.parametrize("markers,aliases", [
    ({"duplicate_of": "winner", "exact_duplicate_of": ""}, 1),
    ({"duplicate_of": "", "exact_duplicate_of": "winner"}, 1),
    ({"duplicate_of": "winner", "exact_duplicate_of": "another-id"}, 1),
    ({"duplicate_of": "", "exact_duplicate_of": ""}, 0),
])
def test_both_markers_preserve_old_alias_detection_and_allow_exact_fallback(tmp_path, markers, aliases):
    bucket(tmp_path, "dynamic/winner.md", "winner")
    bucket(tmp_path, "dynamic/copy.md", "copy", markers=markers)
    result = build_inventory(tmp_path)
    assert result["record_ids"] == result["physical_files"] == 2
    assert result["alias_count"] == aliases
    assert result["total_records"] == result["regular_count"] == 2 - aliases


def test_backup_and_snapshot_trees_do_not_enter_the_current_vault_inventory(tmp_path):
    bucket(tmp_path, "dynamic/工程/live.md", "live")
    expected = build_inventory(tmp_path)
    for backup in ("backups", "snapshots", ".snapshots", "backup_20260405_2124"):
        bucket(tmp_path, f"{backup}/dynamic/工程/copy.md", "backup-only")
        bucket(tmp_path, f"{backup}/permanent/old.md", "backup-alias",
               markers={"exact_duplicate_of": "live"})
    assert build_inventory(tmp_path) == expected


@pytest.mark.parametrize("marker", [None, "duplicate_of", "exact_duplicate_of"])
def test_same_id_physical_copies_keep_existing_unique_id_and_domain_count_basis(tmp_path, marker):
    markers = {marker: "target"} if marker else None
    bucket(tmp_path, "dynamic/工程/one.md", "same-id", markers=markers)
    bucket(tmp_path, "permanent/核心/two.md", "same-id", markers=markers)
    embeddings(tmp_path, "same-id", "same-id")
    result = build_inventory(tmp_path)
    independent = int(marker is None)
    assert result["physical_files"] == 2
    assert result["record_ids"] == result["duplicate_id_files"] == 1
    assert result["alias_count"] == 1 - independent
    assert result["total_records"] == result["regular_count"] == independent
    assert result["layers"] == {
        key: independent if key in {"permanent", "dynamic"} else 0 for key in LAYERS}
    assert result["domains"] == ({"工程": 1, "核心": 1} if independent else {})
    assert result["index"]["records_indexed"] == independent
    assert result["index"]["orphan_rows"] == 0


@pytest.mark.parametrize("marker", ["duplicate_of", "exact_duplicate_of"])
def test_same_id_regular_and_alias_files_keep_existing_overlapping_id_sets(tmp_path, marker):
    bucket(tmp_path, "dynamic/工程/one.md", "same-id")
    bucket(tmp_path, "feel/two.md", "same-id", markers={marker: "target"})
    result = build_inventory(tmp_path)
    assert result["physical_files"] == 2
    assert result["record_ids"] == result["duplicate_id_files"] == 1
    # Preserve the baseline's independent set unions: one physical ID may
    # appear in both sets when its separate files carry different markers.
    assert result["total_records"] == result["alias_count"] == 1
    assert result["regular_count"] == result["layers"]["dynamic"] == 1
    assert result["layers"]["feel"] == 0


def test_filename_id_fallback_and_body_marker_text_keep_existing_behavior(tmp_path):
    fallback_id = "012345abcdef"
    bucket(tmp_path, f"dynamic/无元数据前缀_{fallback_id}.md",
           body="正文中的字符串不算别名：\nexact_duplicate_of: winner\nduplicate_of: winner")
    bucket(tmp_path, "dynamic/plain-name.md", body="没有十六进制 ID 时使用文件名。")
    embeddings(tmp_path, fallback_id, "plain-name")
    result = build_inventory(tmp_path)
    assert result["total_records"] == result["record_ids"] == 2
    assert result["alias_count"] == 0
    assert result["index"]["records_indexed"] == 2
    assert result["index"]["missing_regular"] == result["index"]["orphan_rows"] == 0
