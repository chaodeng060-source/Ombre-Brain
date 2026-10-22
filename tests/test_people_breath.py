"""Exercise people evidence through the real breath fusion and filter path."""
import pytest

import server
from entity_store import EntityStore
from tests.test_entity_recall import _bucket, _configure_server


@pytest.mark.asyncio
async def test_explicit_identity_reaches_breath_beyond_generic_entity_top_twenty(tmp_path, monkeypatch):
    rows = [_bucket(f"a{i:02d}", "我和朝灯今天聊了天气。") for i in range(25)]
    identity = _bucket("zz-identity", "朝灯是这间工作室的主人。", created="2026-07-01T00:00:00")
    rows += [identity, _bucket("zz-other-world", "朝灯是城堡的女王。", world="role")]
    cfg, manager = _configure_server(tmp_path, monkeypatch, rows)
    store = EntityStore(cfg)
    for row in rows:
        store.resolve_and_link(row["id"], row["content"])
    result = await server.breath(query="朝灯是谁", world="daily", relation_depth=0,
                                max_results=10, include_images=False, include_body_state=False)
    assert "[bucket_id:zz-identity]" in result
    assert "[bucket_id:zz-other-world]" not in result
    # It is evidence from an existing memory, not a new model-authored card.
    assert manager.buckets["zz-identity"] == identity


@pytest.mark.asyncio
async def test_people_identity_reuses_domain_filter_in_real_breath(tmp_path, monkeypatch):
    allowed = _bucket("allowed", "朝灯是工作室的主人。", domain=["工程"])
    denied = _bucket("denied", "朝灯是故事里的主人公。", domain=["小说"])
    cfg, _ = _configure_server(tmp_path, monkeypatch, [allowed, denied])
    store = EntityStore(cfg)
    for row in (allowed, denied):
        store.resolve_and_link(row["id"], row["content"])
    result = await server.breath(query="朝灯是谁", world="daily", domain="工程", relation_depth=0,
                                include_images=False, include_body_state=False)
    assert "[bucket_id:allowed]" in result
    assert "[bucket_id:denied]" not in result
