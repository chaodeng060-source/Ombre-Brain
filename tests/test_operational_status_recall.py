import server


def _bucket(bucket_id, content, state=""):
    metadata = {
        "id": bucket_id,
        "type": "dynamic",
        "domain": ["工程"],
    }
    if state:
        metadata.update({
            "validity_kind": "operational_status",
            "validity_state": state,
            "status_key": "status.assembly",
            "validity_valid_at": "2026-08-10T01:00:00+00:00",
        })
        if state == "historical":
            metadata.update({
                "validity_invalid_at": "2026-08-11T01:00:00+00:00",
                "validity_expired_at": "2026-08-11T02:00:00+00:00",
                "validity_superseded_by_bucket_id": "current",
            })
    return {"id": bucket_id, "content": content, "metadata": metadata}


class _AttachedStore:
    def attach(self, buckets):
        return list(buckets)


def test_current_status_query_filters_historical_but_keeps_unknown(monkeypatch):
    monkeypatch.setattr(
        server,
        "_get_operational_status_validity_store",
        lambda: _AttachedStore(),
    )
    historical = _bucket("old", "assembly 跑了三分之一", "historical")
    current = _bucket("current", "assembly 尝试已回滚结案", "current")
    unknown = _bucket("unknown", "assembly 验收全绿")

    kept = server._filter_z_fact_candidates(
        [historical, current, unknown],
        query="assembly 提速做完了吗",
        intent="fact",
    )

    assert [bucket["id"] for bucket in kept] == ["current", "unknown"]


def test_status_prefix_never_presents_unmarked_snapshot_as_current():
    profile = server._state_recall_profile("Ombre 缓存上线了吗")
    current = _bucket("current", "Ombre 缓存已上线", "current")
    unknown = _bucket("unknown", "Ombre 另一项改造未部署")
    protected_history = _bucket(
        "feel-history",
        "旧项目已经 commit、push 并部署完成",
    )
    protected_history["metadata"]["type"] = "feel"
    progress_history = _bucket(
        "feel-progress",
        "当前进度为444/522，剩78条，预计40分钟完成",
    )
    progress_history["metadata"]["type"] = "feel"

    current_prefix = server._recall_prefix(
        "current",
        "main",
        "curated_rrf",
        bucket=current,
        state_profile=profile,
    )
    unknown_prefix = server._recall_prefix(
        "unknown",
        "main",
        "curated_rrf",
        bucket=unknown,
        state_profile=profile,
    )
    protected_prefix = server._recall_prefix(
        "feel-history",
        "main",
        "curated_rrf",
        bucket=protected_history,
        state_profile=profile,
    )
    progress_prefix = server._recall_prefix(
        "feel-progress",
        "main",
        "curated_rrf",
        bucket=progress_history,
        state_profile=profile,
    )

    assert "[validity:current]" in current_prefix
    assert "[authority:current_status]" in current_prefix
    assert "[valid_at:2026-08-10T01:00:00+00:00]" in current_prefix
    assert "[validity:unknown]" in unknown_prefix
    assert "[authority:not_current_status]" in unknown_prefix
    assert "[validity:unknown]" in protected_prefix
    assert "[authority:not_current_status]" in protected_prefix
    assert "[validity:unknown]" in progress_prefix
    assert "[authority:not_current_status]" in progress_prefix


def test_historical_status_query_retains_and_labels_old_snapshot(monkeypatch):
    monkeypatch.setattr(
        server,
        "_get_operational_status_validity_store",
        lambda: _AttachedStore(),
    )
    historical = _bucket("old", "assembly 跑了三分之一", "historical")

    kept = server._filter_z_fact_candidates(
        [historical],
        query="以前 assembly 的进度怎么样",
        intent="fact",
    )
    prefix = server._recall_prefix(
        "old",
        "main",
        "curated_rrf",
        bucket=historical,
        state_profile=server._state_recall_profile("以前 assembly 的进度怎么样"),
    )

    assert kept == [historical]
    assert "[validity:historical]" in prefix
    assert "[authority:historical_status]" in prefix
    assert "[invalid_at:2026-08-11T01:00:00+00:00]" in prefix
