import copy

import pytest

import server
from night_run_coordinator import NightRunCoordinator, NightRunCoordinatorError, NightRunPolicy
from tests.test_entity_recall import _bucket, _configure_server


@pytest.mark.asyncio
async def test_dream_boot_and_old_boot_cache_never_reintroduce_exact_copy(tmp_path, monkeypatch):
    active = _bucket("a", "朝灯在白板上画了三条线。")
    alias = copy.deepcopy(active)
    alias["id"] = "b"
    alias["metadata"].update(name="b", digested=True, exact_duplicate_of="a")
    _configure_server(tmp_path, monkeypatch, [active, alias])
    monkeypatch.setattr(server.decay_engine, "calculate_score", lambda meta: 1.0, raising=False)
    monkeypatch.setattr(server.embedding_engine, "enabled", False, raising=False)
    boot = await server.briefing(deterministic=True, include_body_state=False)
    assert "[a]" in boot
    assert "[b]" not in boot
    dream = await server.dream()
    assert "ID: a" in dream
    assert "ID: b" not in dream
    assert await server._briefing_cache_uses_exact_copy({"buckets": [active, alias]}) is True
    assert await server._briefing_cache_uses_exact_copy({"buckets": [active]}) is False


@pytest.mark.asyncio
async def test_night_reports_alias_root_conflict_instead_of_claiming_success(monkeypatch):
    import exact_metabolism
    async def conflict(_manager):
        return {"status": "partial", "applied": [], "failed": [],
                "errors": ["multiple_active_alias_roots:a,b"], "plan": {"deferred": 0}}
    monkeypatch.setattr(exact_metabolism, "run_exact_consolidation", conflict)
    coordinator = object.__new__(NightRunCoordinator)
    coordinator.bucket_manager = object()
    coordinator.policy = NightRunPolicy(exact_dedup_enabled=True)
    with pytest.raises(NightRunCoordinatorError, match="metabolism.exact_failed"):
        await coordinator._run_derived_maintenance({})
