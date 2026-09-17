import json
from pathlib import Path

import pytest

from tools.retrieval_hints_export import write_manifest
from tools.retrieval_hints_compare import compare
from tests.test_retrieval_hints_bm25 import augmented, documents


def test_export_is_versioned_readonly_idempotent_and_never_originals(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = {"schema_version": 1, "publication_version": "a" * 64, "records": []}
    target = write_manifest(manifest, tmp_path / "keys-mirror")
    assert target.stat().st_mode & 0o222 == 0
    assert write_manifest(manifest, tmp_path / "keys-mirror") == target
    assert json.loads((target.parent / "CURRENT.json").read_text())["manifest_file"] == target.name
    with pytest.raises(ValueError, match="originals"):
        write_manifest(manifest, tmp_path / "imprint-mirror")
    assert not (tmp_path / "imprint-mirror").exists()


@pytest.mark.parametrize("version", ["../escape", "a/../../escape", "", None])
def test_export_rejects_invalid_version_before_creating_directory(tmp_path, version):
    with pytest.raises(ValueError, match="publication_version"):
        write_manifest({"publication_version": version, "records": []}, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_four_groups_keep_full_corpus_and_unknown_delivery(tmp_path):
    dictionary = tmp_path / "synthetic.txt"
    dictionary.write_text("星帆云栈 100000 nz\n")
    manifest = {"records": [augmented()[0]["retrieval_hints_v1"]]}
    turns = [{"n": 7, "segment": "C", "ts": "2026-01-01", "user_text": "legacy-search", "delivery_status": "unknown"}]
    result = list(compare(documents(), manifest, dictionary, turns))[0]
    assert result["provider_calls"] == 0 and result["delivery_status"] == "unknown"
    assert set(result["conditions"]) == {"OFF", "dictionary_only", "keys_only", "both"}
    assert result["conditions"]["OFF"]["bucket_ids"] == []
    assert result["conditions"]["keys_only"]["bucket_ids"] == ["a"]
    assert result["final_breath"] == "pending_night_pilot"


def test_comparison_does_not_infer_delivery_from_row_number(tmp_path):
    dictionary = tmp_path / "synthetic.txt"
    dictionary.write_text("星帆云栈 100000 nz\n")
    turns = [{"n": 96, "segment": "C", "ts": "2026-01-01", "user_text": "legacy-search"}]
    result = next(compare(documents(), {"records": []}, dictionary, turns))
    assert result["delivery_status"] == "not_replayed"
