import asyncio

import httpx
from starlette.applications import Starlette
from starlette.routing import Route

from entity_store import EntityStore
from mcp_auth import APIBearerAuthMiddleware
from night_run_coordinator import NightRunCoordinator, NightRunPolicy
from people_memory import build_people_cards, refresh_people_links


class Buckets:
    def __init__(self, path):
        self.base_dir = str(path)
        self.rows = {"a": {"id": "a", "content": "小墨是朝灯的朋友。", "metadata": {
            "type": "dynamic", "world": "", "source_ref": "raw:event-1",
        }}}

    async def get(self, key):
        return self.rows.get(key)

    async def list_all(self, **_kwargs):
        return list(self.rows.values())


def setup(tmp_path):
    root = tmp_path / "vault"
    store = EntityStore({"buckets_dir": str(root), "entities": {"seeds": [
        {"canonical_name": "小墨", "type": "person", "aliases": ["墨总"]},
        {"canonical_name": "朝灯", "type": "person", "aliases": []},
    ]}})
    return store, Buckets(root)


def test_night_hook_refreshes_old_people_links_without_rewriting_memory(tmp_path):
    store, buckets = setup(tmp_path)
    before = str(buckets.rows)
    coordinator = object.__new__(NightRunCoordinator)
    coordinator.bucket_manager = buckets
    coordinator.policy = NightRunPolicy(people_refresh_enabled=True)
    counts = {}
    asyncio.run(coordinator._run_derived_maintenance(counts))
    assert counts == {"people_links_refreshed": 1, "people_sources_checked": 1}
    result = asyncio.run(build_people_cards(store, buckets, query="墨总是谁"))
    assert result["people"][0]["evidence"][0]["source_refs"]["source_ref"] == "raw:event-1"
    assert str(buckets.rows) == before
    again = asyncio.run(refresh_people_links(store, buckets))
    assert again["refreshed"] == 0


def test_people_api_requires_existing_api_bearer_and_returns_actual_evidence(tmp_path, monkeypatch):
    import server
    store, buckets = setup(tmp_path)
    asyncio.run(refresh_people_links(store, buckets))
    monkeypatch.setattr(server, "bucket_mgr", buckets)
    monkeypatch.setattr(server, "_get_entity_store", lambda **_kwargs: store)
    monkeypatch.setitem(server.config, "current_world", "")
    app = Starlette(routes=[Route("/api/people", server.api_people)])
    app.add_middleware(APIBearerAuthMiddleware, token="synthetic-test-token-for-people-view-0000")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            denied = await client.get("/api/people", params={"q": "小墨是谁"})
            assert denied.status_code == 401
            assert "小墨" not in denied.text
            result = await client.get("/api/people", params={"q": "小墨是谁"},
                                      headers={"Authorization": "Bearer synthetic-test-token-for-people-view-0000"})
            assert result.status_code == 200
            body = result.json()
            assert body["people"][0]["name"] == "小墨"
            assert body["relations"][0]["relation"] == "朋友"
            assert body["memory_written"] is False
            invalid = await client.get("/api/people?limit=nope", headers={"Authorization": "Bearer synthetic-test-token-for-people-view-0000"})
            assert invalid.status_code == 400
    asyncio.run(scenario())
