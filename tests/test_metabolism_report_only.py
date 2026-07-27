"""M-axis report-only tests for decay candidates."""

import asyncio
from datetime import datetime, timedelta

from decay_engine import DecayEngine


def _old_bucket(bucket_id="old"):
    timestamp = (datetime.now() - timedelta(days=90)).isoformat()
    return {
        "id": bucket_id,
        "content": "old",
        "metadata": {
            "id": bucket_id,
            "name": bucket_id,
            "type": "dynamic",
            "importance": 1,
            "activation_count": 1,
            "arousal": 0.0,
            "created": timestamp,
            "last_active": timestamp,
            "resolved": False,
        },
    }


class SpyBucketManager:
    def __init__(self, buckets=None, error=None):
        self.buckets = list(buckets or [])
        self.error = error
        self.updates = []
        self.archives = []

    async def list_all(self, include_archive=False):
        if self.error:
            raise self.error
        return list(self.buckets)

    async def update(self, bucket_id, **kwargs):
        self.updates.append((bucket_id, kwargs))
        return True

    async def archive(self, bucket_id):
        self.archives.append(bucket_id)
        return True


def _engine(manager, mode="report_only"):
    return DecayEngine(
        {
            "metabolism": {"mode": mode},
            "decay": {"threshold": 999.0},
        },
        manager,
    )


def test_decay_report_only_lists_candidates_without_mutating():
    manager = SpyBucketManager([_old_bucket()])
    result = asyncio.run(_engine(manager).run_decay_cycle())

    assert result["ok"] is True
    assert result["mode"] == "report_only"
    assert result["would_auto_resolve"] == ["old"]
    assert result["would_archive"] == ["old"]
    assert result["auto_resolved"] == 0 and result["archived"] == 0
    assert manager.updates == [] and manager.archives == []


def test_decay_apply_requires_explicit_mode():
    manager = SpyBucketManager([_old_bucket()])
    result = asyncio.run(_engine(manager, mode="apply").run_decay_cycle())

    assert result["mode"] == "apply"
    assert result["auto_resolved"] == 1 and result["archived"] == 1
    assert manager.updates == [("old", {"resolved": True})]
    assert manager.archives == ["old"]


def test_decay_read_failure_is_red():
    manager = SpyBucketManager(error=RuntimeError("offline"))
    result = asyncio.run(_engine(manager).run_decay_cycle())

    assert result["ok"] is False
    assert result["errors"] == ["list_all:RuntimeError"]
    assert result["would_auto_resolve"] == []
    assert result["would_archive"] == []


def test_decay_rejects_non_enum_mode():
    try:
        _engine(SpyBucketManager(), mode="yes")
        assert False, "mode must be an exact enum"
    except ValueError:
        pass
