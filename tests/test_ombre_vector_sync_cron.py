from __future__ import annotations

import importlib.util
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
