from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from tools.hold_duplicate_window import measure


def test_equal_windows_count_body_copies_without_merging_distinct_events(tmp_path):
    root = tmp_path / "dynamic"
    root.mkdir()
    for i, day, body, world in [
        (0, 13, "same", "one"), (1, 14, "same", "one"),
        (2, 14, "different", "one"), (3, 15, "same", "one"),
        (4, 15, "same", "one"), (5, 15, "same", "two"),
    ]:
        (root / f"{i}.md").write_text(
            f"---\nid: '{i}'\nrecorded_at: '2026-09-{day}T12:00:00'\nworld: {world}\ntype: dynamic\n---\n{body}"
        )
    # Unrelated documents outside the vault directories never enter counts.
    (tmp_path / "README.md").write_text("not a memory")
    before = {p: p.read_bytes() for p in root.iterdir()}
    result = measure(tmp_path, "2026-09-15T00:00:00Z", naive_timezone="UTC",
                     as_of=datetime(2026, 9, 16, tzinfo=timezone.utc))
    assert result["parsed_buckets"] == 6
    assert result["before"]["new_buckets"] == 2
    assert result["before"]["body_equal_to_earlier_bucket"] == 1
    assert result["after"]["new_buckets"] == 3
    assert result["after"]["body_equal_to_earlier_bucket"] == 2
    assert result["after"]["within_window_extra_body_copies"] == 1
    assert result["after"]["window_complete"] is True
    assert {p: p.read_bytes() for p in root.iterdir()} == before
    partial = measure(tmp_path, "2026-09-15T00:00:00Z", naive_timezone="UTC",
                      as_of=datetime(2026, 9, 15, 13, tzinfo=timezone.utc))
    assert partial["after"]["window_complete"] is False


def test_bad_or_missing_time_is_reported_not_silently_counted(tmp_path):
    (tmp_path / "feel").mkdir()
    (tmp_path / "feel" / "bad.md").write_text("---\nid: invalid\n---\nbody")
    result = measure(tmp_path, "2026-09-15T00:00:00Z", naive_timezone="UTC")
    assert result["read_errors"] == {"ValueError": 1}
    assert result["parsed_buckets"] == 0


def test_legacy_header_content_does_not_replace_markdown_body(tmp_path):
    (tmp_path / "feel").mkdir()
    (tmp_path / "feel" / "old.md").write_text(
        "---\nid: old\ncontent: header\ncreated: '2026-09-15T09:00:00'\n---\nactual body"
    )
    result = measure(tmp_path, "2026-09-15T00:00:00Z", naive_timezone="Asia/Shanghai")
    assert result["parsed_buckets"] == 1
    assert result["created_fallback_buckets"] == 1
    assert result["after"]["new_buckets"] == 1


def test_deploy_and_rollback_scripts_parse_and_reject_missing_arguments():
    root = Path(__file__).resolve().parents[1]
    for name in ("deploy_hold_idempotency.sh", "rollback_hold_idempotency.sh"):
        script = root / "scripts" / name
        check = subprocess.run(["bash", "-n", str(script)], capture_output=True)
        assert check.returncode == 0, check.stderr
        missing = subprocess.run(["bash", str(script)], capture_output=True)
        assert missing.returncode != 0
