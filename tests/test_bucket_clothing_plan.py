from __future__ import annotations

import hashlib
from pathlib import Path

from tools import build_bucket_clothing_plan as clothing


def _write_bucket(
    vault: Path,
    root: str,
    bucket_id: str,
    body: str,
    *,
    name: str | None = None,
) -> Path:
    path = vault / root / f"{bucket_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "---\n"
            f"id: {bucket_id}\n"
            f"name: {name or bucket_id}\n"
            "type: feel\n"
            "domain:\n- 未分类\n"
            "created: 2026-05-03T12:00:00+08:00\n"
            "---\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )
    return path


def _hashes(vault: Path) -> dict[str, str]:
    return {
        str(path.relative_to(vault)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(vault.rglob("*.md"))
    }


def test_plan_uses_only_literal_keys_and_never_changes_vault(tmp_path):
    vault = tmp_path / "vault"
    _write_bucket(
        vault,
        "feel",
        "clear",
        "姐姐在郴州中山西街的步步高吃铁锅炖，也认真听了小克咪的事情。",
    )
    _write_bucket(vault, "feel", "uncertain", "我今天还好。")
    _write_bucket(
        vault,
        "_backup",
        "backup",
        "姐姐在郴州中山西街吃铁锅炖。",
    )
    _write_bucket(
        vault,
        "feel",
        "clothed",
        "姐姐在郴州吃铁锅炖。\n[检索钥匙: 姐姐/郴州/铁锅炖]",
        name="姐姐_郴州_铁锅炖_2026-05-03",
    )
    _write_bucket(vault, "feel", "5a9a6d485209", "第一份正文。")
    _write_bucket(vault, "feel", "780c1a7050f7", "第二份不同正文。")
    before = _hashes(vault)

    plan = clothing.build_plan(vault, expected_count=6)

    assert _hashes(vault) == before
    assert plan["source"]["unchanged"] is True
    by_id = {item["bucket_id"]: item for item in plan["items"]}
    clear = by_id["clear"]
    assert clear["status"] == "propose"
    assert len(clear["retrieval_keys"]) >= 3
    source_body = (vault / clear["path"]).read_text(encoding="utf-8")
    for item in clear["retrieval_keys"]:
        assert item["key"] in source_body
        assert item["key"] in item["evidence"]
    for term in clear["name_basis"]:
        assert term in source_body
    assert clear["suggested_name"].endswith("2026-05-03")

    assert by_id["uncertain"]["status"] == "skip"
    assert by_id["uncertain"]["skip_reason"] == "insufficient_literal_entities"
    assert by_id["backup"]["skip_reason"] == "non_live_or_backup_path"
    assert by_id["clothed"]["skip_reason"] == "already_clothed"
    assert plan["duplicate_probe"]["exact_body_equal"] is False
    assert plan["duplicate_probe"]["decision"] == "report_only_no_delete"


def test_expected_count_fails_closed(tmp_path):
    vault = tmp_path / "vault"
    _write_bucket(vault, "feel", "one", "姐姐在郴州吃铁锅炖。")

    try:
        clothing.build_plan(vault, expected_count=2)
    except ValueError as exc:
        assert "expected 2, got 1" in str(exc)
    else:
        raise AssertionError("count mismatch must fail closed")


def test_gibberish_name_and_body_are_skipped(tmp_path):
    vault = tmp_path / "vault"
    _write_bucket(
        vault,
        "feel",
        "gibberish",
        "???????????,??????????? git commit?",
        name="??????????",
    )

    plan = clothing.build_plan(vault, expected_count=1)

    item = plan["items"][0]
    assert item["status"] == "skip"
    assert item["suggested_name"] is None


def test_name_does_not_repeat_event_date(tmp_path):
    vault = tmp_path / "vault"
    _write_bucket(
        vault,
        "feel",
        "dated",
        "2026-05-03 中午，姐姐在郴州中山西街吃铁锅炖。",
    )

    plan = clothing.build_plan(vault, expected_count=1)

    item = plan["items"][0]
    assert item["status"] == "propose"
    assert item["suggested_name"].count("2026-05-03") == 1
    assert len(item["suggested_name"]) <= 41


def test_colliding_suggested_names_are_skipped(tmp_path):
    vault = tmp_path / "vault"
    body = "姐姐在郴州中山西街吃铁锅炖，后来带回一张照片。"
    _write_bucket(vault, "feel", "one", body)
    _write_bucket(vault, "feel", "two", body)

    plan = clothing.build_plan(vault, expected_count=2)

    assert {item["status"] for item in plan["items"]} == {"skip"}
    assert {
        item["skip_reason"] for item in plan["items"]
    } == {"suggested_name_collision"}
