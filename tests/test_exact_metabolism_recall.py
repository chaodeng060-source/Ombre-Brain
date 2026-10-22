import asyncio
from pathlib import Path

import frontmatter

from bucket_manager import BucketManager
from exact_metabolism import restore_exact_consolidation, run_exact_consolidation


def test_automatic_copies_disappear_from_search_and_side_channels_but_keep_old_ids(tmp_path):
    import server
    root = tmp_path / "vault"
    root.mkdir()
    manager = BucketManager({"buckets_dir": str(root), "audit": {"enabled": True}, "wikilink": {"enabled": False}})
    directory = Path(manager.dynamic_dir) / "synthetic"
    directory.mkdir(parents=True)
    body = "原始事件：我在白板上画了三条线。\n逐字保留这一行。"
    for key in ["a", "b"]:
        post = frontmatter.Post(body, id=key, name=key, type="dynamic", domain=["工程"],
                                tags=["三条线"], world="", created="2026-10-22T00:00:00+00:00",
                                source_event_ids=["event-1"])
        (directory / ("copy_" + key + ".md")).write_text(frontmatter.dumps(post))

    async def scenario():
        # Chat recall uses relevance-first; legacy weighted search can reject
        # this partial multiword query before any duplicate exists.
        before = await manager.search("白板三条线", limit=10, relevance_first=True)
        assert {row["id"] for row in before} == {"a", "b"}
        result = await run_exact_consolidation(manager)
        assert result["applied"] == ["b"]
        found = await manager.search("白板三条线", limit=10, relevance_first=True)
        assert [row["id"] for row in found] == ["a"]
        old = await manager.get("b")
        assert old["content"] == body
        assert not server._is_main_recall_bucket(old)
        assert not server._passes_merge_candidate_filters(old, ["工程"], [""])
        restored = await restore_exact_consolidation(manager, result["recovery"])
        assert restored["restored"] == ["b"]
        found = await manager.search("白板三条线", limit=10, relevance_first=True)
        assert {row["id"] for row in found} == {"a", "b"}
        assert (await manager.get("b"))["content"] == body
    asyncio.run(scenario())
