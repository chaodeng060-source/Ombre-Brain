from tools.retrieval_hints_prepare import select_sources, dictionary_proposals


def test_deterministic_strata_required_ids_and_no_resurrection():
    rows = [{"id": str(i), "content": "银杏馆的星舟会议", "metadata": {"created": "2026-09-01", "domain": [str(i % 3)]}}
            for i in range(20)]
    rows[0]["metadata"]["type"] = "archived"
    rows[1]["content"] = "task_200 evaluation derivation"
    first, statuses, _ = select_sources(rows, ["0", "1", "3", "missing"], count=10, recent_month="2026-09")
    second, _, _ = select_sources(list(reversed(rows)), ["0", "1", "3", "missing"], count=10, recent_month="2026-09")
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert first[0]["id"] == "3"
    assert {s["status"] for s in statuses} == {"included", "inactive", "pilot_or_evaluation_material", "not_active_in_snapshot"}


def test_dictionary_is_only_source_bound_pending_proposals():
    source = {"id": "synthetic-a", "content": "[[星帆云栈]]在[[银杏馆]]开会。"}
    result = dictionary_proposals([source])
    assert any(row["term"] == "星帆云栈" for row in result)
    assert all(row["approval"] == "pending" for row in result)
    assert all(row["term"] in item["evidence"] and item["evidence"] in source["content"]
               for row in result for item in row["sources"])


def test_many_old_domains_do_not_starve_recent_sources():
    old = [{"id": f"old-{i}", "content": "银杏馆会议", "metadata": {
        "created": "2026-07-01", "domain": [f"domain-{i}"]}} for i in range(30)]
    recent = [{"id": f"new-{i}", "content": "星舟会议", "metadata": {
        "created": "2026-09-01", "domain": ["shared"]}} for i in range(20)]
    selected, _, _ = select_sources(old + recent, ["new-0"], count=10, recent_month="2026-09")
    reverse, _, _ = select_sources(list(reversed(old + recent)), ["new-0"], count=10, recent_month="2026-09")
    assert [r["id"] for r in selected] == [r["id"] for r in reverse]
    assert selected[0]["id"] == "new-0"
    assert sum(r["id"].startswith("new-") for r in selected) == 5


def test_recency_balance_preserves_required_and_exhausts_available_strata():
    rows = [{"id": str(i), "content": "银杏馆会议", "metadata": {
        "created": "2026-09-01" if i < 3 else "2026-07-01"}} for i in range(12)]
    selected, _, _ = select_sources(rows, ["0", "1", "2"], count=10, recent_month="2026-09")
    assert [r["id"] for r in selected[:3]] == ["0", "1", "2"]
    assert len({r["id"] for r in selected}) == 10
    assert sum(r["metadata"]["created"].startswith("2026-09") for r in selected) == 3
