import json
from pathlib import Path

import pytest

from bucket_manager import BucketManager
from retrieval_hints import Generation
from tests.test_retrieval_hints_bm25 import augmented, documents
from tests.test_retrieval_hints_storage import Peer


@pytest.mark.asyncio
async def test_off_creates_no_sidecar_and_search_is_byte_equal(test_config, monkeypatch):
    monkeypatch.setenv("OMBRE_RETRIEVAL_HINTS_ENABLED", "0")
    monkeypatch.setenv("OMBRE_PRIVATE_RECALL_DICT_ENABLED", "1")
    manager = BucketManager(test_config)
    bid = await manager.create(content="synthetic body", domain=["测试"])
    assert await manager.update(bid, content="synthetic body updated")
    assert manager._retrieval_hints_store is None
    assert not (Path(test_config["buckets_dir"]) / ".retrieval_hints").exists()
    manager._bm25_mode = "live"
    manager._bm25_dirty = manager._bm25_unknown_dirty = False
    manager._bm25 = manager._build_bm25_index(documents())
    first = await manager.search("comet", preloaded_buckets=documents(), relevance_first=True, relevance_candidate_floor=0)
    manager._bm25 = manager._build_bm25_index(augmented())
    second = await manager.search("comet", preloaded_buckets=documents(), relevance_first=True, relevance_candidate_floor=0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


@pytest.mark.asyncio
async def test_failed_registration_keeps_successful_source_write(test_config, monkeypatch):
    monkeypatch.setenv("OMBRE_RETRIEVAL_HINTS_ENABLED", "1")
    manager = BucketManager(test_config)
    def unavailable(*args, **kwargs):
        raise OSError("unavailable local sidecar")
    monkeypatch.setattr(manager._retrieval_hints_store, "register", unavailable)
    bid = await manager.create(content="original synthetic source", domain=["测试"])
    assert bid and (await manager.get(bid))["content"] == "original synthetic source"
    assert await manager.update(bid, content="updated synthetic source")
    assert (await manager.get(bid))["content"] == "updated synthetic source"
    assert not manager._retrieval_hints_store.path.exists()


@pytest.mark.asyncio
async def test_new_write_registers_only_and_difference_scan_can_recover(test_config, monkeypatch):
    monkeypatch.setenv("OMBRE_RETRIEVAL_HINTS_ENABLED", "1")
    manager = BucketManager(test_config)
    bid = await manager.create(content="synthetic source", domain=["测试"])
    jobs = manager._retrieval_hints_store.jobs()
    assert len(jobs) == 1 and jobs[0]["bucket_id"] == bid
    assert jobs[0]["state"] == "queued" and jobs[0]["attempts"] == 0
    assert manager._retrieval_hints_store.lookup(await manager.get(bid)) is None
    assert manager._retrieval_hints_store.register(await manager.get(bid)) == jobs[0]["job_key"]


@pytest.mark.asyncio
async def test_published_hint_enters_candidate_not_literal_force(test_config, monkeypatch):
    monkeypatch.setenv("OMBRE_RETRIEVAL_HINTS_ENABLED", "1")
    manager = BucketManager(test_config)
    source = augmented()[0]
    store = manager._retrieval_hints_store
    version = store.stage(source, Generation("ok", source["retrieval_hints_v1"]["payload"]))
    peer = Peer()
    manager._bm25 = manager._build_hinted_bm25_index(documents())
    manager._bm25_mode = "live"
    manager._bm25_dirty = manager._bm25_unknown_dirty = False
    assert manager._bm25.score("legacy-search") == {}
    store.publish("a", version, peer, lambda: source)
    await manager._refresh_retrieval_hint_index(documents())
    assert "a" in manager._bm25.score("legacy-search")
    assert manager._bm25._literal_retrieval_rows == ()
    assert manager._bm25.rare_term_hits("legacy-search", max_df=8) == {}
    result = await manager.search("legacy-search", preloaded_buckets=documents(), relevance_first=True, relevance_candidate_floor=0)
    assert result[0]["id"] == "a" and result[0]["hint_match"]
    assert "retrieval_keys" not in result[0]["metadata"]


@pytest.mark.asyncio
async def test_withdrawn_hint_refresh_keeps_body(test_config, monkeypatch):
    monkeypatch.setenv("OMBRE_RETRIEVAL_HINTS_ENABLED", "1")
    manager = BucketManager(test_config)
    source = augmented()[0]
    store = manager._retrieval_hints_store
    version = store.stage(source, Generation("ok", source["retrieval_hints_v1"]["payload"]))
    store.publish("a", version, Peer(), lambda: source)
    manager._bm25_mode = "live"
    manager._bm25 = manager._build_hinted_bm25_index(documents())
    manager._retrieval_hint_rows = store.published_snapshot()
    store.rollback_pointer("a", expected=version, previous=None)
    await manager._refresh_retrieval_hint_index(documents())
    assert manager._bm25.score("legacy-search") == {}
    assert "a" in manager._bm25.score("comet")
