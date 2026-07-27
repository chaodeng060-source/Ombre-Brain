from datetime import datetime

from fact_conflicts import (
    build_supersedes_audit,
    detect_fact_conflicts,
    is_z_scan_candidate,
    scan_cross_bucket_z_conflicts,
)


def _bucket(bucket_id, content, *, name="period", domain=None, **meta):
    metadata = {
        "id": bucket_id,
        "name": name,
        "domain": domain or ["health"],
        "type": "dynamic",
    }
    metadata.update(meta)
    return {"id": bucket_id, "content": content, "metadata": metadata}


def test_detect_fact_conflicts_matches_merge_time_fields():
    conflicts = detect_fact_conflicts(
        "period_start: 2026-05-03\nflow: light\nweight: 50kg",
        "period_start: 2026-05-04\nflow: light\nweight: 51kg",
    )
    by_field = {c["field"]: c for c in conflicts}

    assert by_field["period_start"]["old"] == "2026-05-03"
    assert by_field["period_start"]["new"] == "2026-05-04"
    assert by_field["weight"]["old"] == "50kg"
    assert by_field["weight"]["new"] == "51kg"


def test_build_supersedes_audit_keeps_existing_shape():
    bucket = _bucket("old", "period_start: 2026-05-03")
    audit = build_supersedes_audit(
        bucket,
        "period_start: 2026-05-04",
        now=datetime.fromisoformat("2026-07-08T14:00:00+08:00"),
    )

    by_field = {entry["field"]: entry for entry in audit}
    assert by_field["period_start"] == {
        "field": "period_start",
        "old": "2026-05-03",
        "new": "2026-05-04",
        "at": "2026-07-08T14:00:00+08:00",
        "bucket_id": "old",
    }


def test_scan_cross_bucket_z_conflicts_reports_same_topic_candidates():
    buckets = [
        _bucket("a", "period_start: 2026-06-02", name="period"),
        _bucket("b", "period_start: 2026-06-30", name="period"),
        _bucket("c", "unrelated: 1", name="other", domain=["ops"]),
    ]

    reports = scan_cross_bucket_z_conflicts(buckets)

    assert len(reports) == 1
    assert reports[0]["left_id"] == "a"
    assert reports[0]["right_id"] == "b"
    assert reports[0]["fields"][0]["field"] == "period_start"


def test_z_scan_skips_protected_and_feel_buckets():
    protected = _bucket("p", "date: 2026-06-02", domain=["恋爱"])
    feel = _bucket("f", "date: 2026-06-30", type="feel")
    pinned = _bucket("pin", "date: 2026-06-30", pinned=True)

    assert is_z_scan_candidate(protected) is False
    assert is_z_scan_candidate(feel) is False
    assert is_z_scan_candidate(pinned) is False
    assert scan_cross_bucket_z_conflicts([protected, feel, pinned]) == []
