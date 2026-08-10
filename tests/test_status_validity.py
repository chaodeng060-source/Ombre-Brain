import os

from status_validity import (
    OperationalStatusValidityStore,
    VIEW_CURRENT,
    VIEW_HISTORICAL,
    VIEW_NEUTRAL,
    bucket_is_operational_status,
    operational_status_query_view,
    validity_label,
)


def _bucket(bucket_id, content="Ombre 缓存已上线", domain=None):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "type": "dynamic",
            "domain": domain or ["工程"],
        },
    }


def test_real_status_questions_activate_only_the_narrow_view():
    assert operational_status_query_view("Ombre 缓存上线了吗") == VIEW_CURRENT
    assert operational_status_query_view("assembly 提速做完了吗") == VIEW_CURRENT
    assert operational_status_query_view("以前 assembly 的进度怎么样") == VIEW_HISTORICAL
    assert operational_status_query_view("想起 Ombre 那一夜") == VIEW_NEUTRAL


def test_operational_classifier_excludes_protected_narrative_memory():
    assert bucket_is_operational_status(_bucket("a", "服务已经部署完成"))
    protected = _bucket("b", "纪念日安排已经完成", ["纪念日"])
    assert not bucket_is_operational_status(protected)


def test_reading_absent_sidecar_does_not_create_it(tmp_path):
    path = tmp_path / ".validity" / "operational_status.sqlite3"
    store = OperationalStatusValidityStore(str(path))

    assert store.lookup_many(["missing"]) == {}
    assert not path.exists()
    assert not path.parent.exists()


def test_supersession_keeps_old_history_and_exposes_bitemporal_fields(tmp_path):
    path = tmp_path / ".validity" / "operational_status.sqlite3"
    store = OperationalStatusValidityStore(str(path))

    outcome = store.mark_supersession(
        old_bucket_id="old-status",
        new_bucket_id="new-status",
        old_valid_at="2026-08-10T01:00:00+00:00",
        new_valid_at="2026-08-11T01:00:00+00:00",
        source_ref="test:real-transition",
    )

    markers = store.lookup_many(["old-status", "new-status"])
    assert outcome == {
        "status_key": "status.old-status",
        "current_bucket_id": "new-status",
    }
    assert markers["old-status"]["validity_state"] == "historical"
    assert markers["old-status"]["validity_invalid_at"] == "2026-08-11T01:00:00+00:00"
    assert markers["old-status"]["validity_expired_at"]
    assert markers["old-status"]["validity_superseded_by_bucket_id"] == "new-status"
    assert markers["new-status"]["validity_state"] == "current"
    assert markers["new-status"]["validity_valid_at"] == "2026-08-11T01:00:00+00:00"
    assert markers["new-status"]["validity_supersedes_bucket_ids"] == ["old-status"]
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_backfilled_older_event_cannot_replace_newer_current_status(tmp_path):
    store = OperationalStatusValidityStore(
        str(tmp_path / ".validity" / "operational_status.sqlite3")
    )
    store.mark_current(
        "known-current",
        status_key="status.project",
        valid_at="2026-08-11T05:00:00+00:00",
        source_ref="test:current",
    )

    outcome = store.mark_supersession(
        old_bucket_id="older-status",
        new_bucket_id="backfilled-event",
        old_valid_at="2026-08-09T01:00:00+00:00",
        new_valid_at="2026-08-10T01:00:00+00:00",
        status_key="status.project",
        source_ref="test:backfill",
    )

    markers = store.lookup_many(["known-current", "backfilled-event"])
    assert outcome["current_bucket_id"] == "known-current"
    assert markers["known-current"]["validity_state"] == "current"
    assert markers["backfilled-event"]["validity_state"] == "historical"
    assert markers["backfilled-event"]["validity_invalid_at"] == "2026-08-11T05:00:00+00:00"


def test_unmarked_status_is_unknown_and_marked_status_is_current(tmp_path):
    store = OperationalStatusValidityStore(
        str(tmp_path / ".validity" / "operational_status.sqlite3")
    )
    bucket = _bucket("cache-status")
    assert validity_label(bucket, view=VIEW_CURRENT) == {"state": "unknown"}

    store.mark_current(
        "cache-status",
        status_key="status.ombre-cache",
        valid_at="2026-08-10T12:00:00+00:00",
        source_ref="test:acceptance",
    )
    store.attach([bucket])

    assert validity_label(bucket, view=VIEW_CURRENT) == {
        "state": "current",
        "valid_at": "2026-08-10T12:00:00+00:00",
    }
