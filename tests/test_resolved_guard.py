# ============================================================
# Test: Resolved Guard — protected-domain buckets cannot resolve
# 测试：resolved 守卫 —— 保护域桶禁止 resolved=1
#
# Backstory: 5.10 incident — 13 protected-domain buckets were
# wrongly resolved by a CC self. This guard makes it impossible.
# ============================================================

import pytest
from utils import ResolvedGuardError, PROTECTED_RESOLVE_DOMAINS


@pytest.mark.parametrize("domain", sorted(PROTECTED_RESOLVE_DOMAINS - {"feel"}))
@pytest.mark.asyncio
async def test_protected_domain_refuses_resolved(bucket_mgr, domain):
    bid = await bucket_mgr.create(
        content=f"protected {domain} bucket",
        domain=[domain],
        importance=8,
    )
    with pytest.raises(ResolvedGuardError):
        await bucket_mgr.update(bid, resolved=True)


@pytest.mark.asyncio
async def test_feel_type_refuses_resolved(bucket_mgr):
    bid = await bucket_mgr.create(
        content="feel bucket",
        domain=["未分类"],
        bucket_type="feel",
    )
    with pytest.raises(ResolvedGuardError):
        await bucket_mgr.update(bid, resolved=True)


@pytest.mark.asyncio
async def test_unprotected_domain_allows_resolved(bucket_mgr):
    bid = await bucket_mgr.create(
        content="engineering bucket",
        domain=["工程"],
        importance=5,
    )
    assert await bucket_mgr.update(bid, resolved=True) is True


@pytest.mark.asyncio
async def test_resolved_false_always_allowed_on_protected(bucket_mgr):
    """Unresolving must always work — used to recover the 5.10 wrongly-resolved buckets."""
    bid = await bucket_mgr.create(
        content="protected bucket",
        domain=["恋爱"],
        importance=8,
    )
    assert await bucket_mgr.update(bid, resolved=False) is True


@pytest.mark.asyncio
async def test_other_updates_still_work_on_protected(bucket_mgr):
    """Guard only blocks resolved=True, not other field edits."""
    bid = await bucket_mgr.create(
        content="protected bucket",
        domain=["约定"],
        importance=8,
    )
    assert await bucket_mgr.update(bid, importance=10, pinned=True) is True
