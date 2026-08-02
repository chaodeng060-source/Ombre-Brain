from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from entity_store import (
    EntityStore,
    UnsafeEntityStore,
    content_sha256,
    entity_mention_present,
    normalize_term,
)


def _config(tmp_path: Path, seeds=()) -> dict:
    buckets_dir = tmp_path / "buckets"
    buckets_dir.mkdir(exist_ok=True)
    return {
        "buckets_dir": str(buckets_dir),
        "entities": {"seeds": list(seeds)},
    }


def _zhaodeng_seed() -> dict:
    return {
        "canonical_name": "朝灯",
        "type": "person",
        "aliases": ["Rosita", "老婆", "宝宝"],
    }


def test_schema_sidecar_permissions_and_pragmas(tmp_path):
    config = _config(tmp_path, [_zhaodeng_seed()])
    store = EntityStore(config)

    assert store.db_path == str(tmp_path / "buckets" / ".entities" / "entities.sqlite3")
    assert stat.S_IMODE(os.lstat(store.entities_dir).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(store.db_path).st_mode) == 0o600

    with sqlite3.connect(store.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"entities", "entity_aliases", "bucket_entities", "entity_events"} <= tables
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "delete"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        foreign_keys = {
            row[2]
            for table in ("entity_aliases", "bucket_entities", "entity_events")
            for row in conn.execute(f"PRAGMA foreign_key_list({table})")
        }
        assert foreign_keys == {"entities"}


def test_nfkc_casefold_and_whitespace_normalization(tmp_path):
    assert normalize_term("  ＶＡＥ\t  许嵩  ") == "vae 许嵩"
    store = EntityStore(
        _config(
            tmp_path,
            [{"canonical_name": "许嵩", "type": "person", "aliases": ["Vae"]}],
        )
    )
    resolved = store.resolve_query("ｖＡＥ")
    assert resolved.canonical_query == "许嵩"
    assert resolved.terms == ("许嵩",)


def test_seed_initialization_is_idempotent(tmp_path):
    config = _config(tmp_path, [_zhaodeng_seed()])
    first = EntityStore(config)
    with sqlite3.connect(first.db_path) as conn:
        before = tuple(
            conn.execute(
                "SELECT (SELECT count(*) FROM entities),"
                "       (SELECT count(*) FROM entity_aliases),"
                "       (SELECT count(*) FROM entity_events)"
            ).fetchone()
        )
    EntityStore(config)
    with sqlite3.connect(first.db_path) as conn:
        after = tuple(
            conn.execute(
                "SELECT (SELECT count(*) FROM entities),"
                "       (SELECT count(*) FROM entity_aliases),"
                "       (SELECT count(*) FROM entity_events)"
            ).fetchone()
        )
    assert before == after


def test_alias_is_not_globally_unique_and_collision_is_ambiguous(tmp_path):
    store = EntityStore(_config(tmp_path))
    first = store.resolve_or_create("甲", ["小名"], "person")
    second = store.resolve_or_create("乙", ["小名"], "person")
    assert first.entity_id != second.entity_id

    result = store.resolve_query("小名")
    assert result.entity_ids == ()
    assert result.ambiguous_terms == ("小名",)
    assert result.canonical_query == "小名"
    assert store.linked_bucket_ids("小名") == []


def test_cross_type_alias_collision_never_lets_model_type_choose_owner(tmp_path):
    store = EntityStore(_config(tmp_path))
    person = store.resolve_or_create("阿雾", ["Ombre"], "person")
    project = store.resolve_or_create("Ombre Brain", ["Ombre"], "project")
    assert person.entity_id != project.entity_id
    assert store.resolve_query("Ombre").entity_ids == ()

    linked = store.resolve_and_link(
        "project-bucket",
        "Ombre 完成了发布。",
        [{"mention": "Ombre", "type": "project"}],
    )
    assert linked == ()
    assert store.linked_bucket_ids(entity_ids=[person.entity_id, project.entity_id]) == []


def test_model_type_disagreement_cannot_poison_seed_alias(tmp_path):
    store = EntityStore(
        _config(
            tmp_path,
            [{"canonical_name": "Ombre Brain", "type": "project", "aliases": ["Ombre"]}],
        )
    )
    before = store.resolve_query("Ombre")

    linked = store.resolve_and_link(
        "wrong-model-type",
        "Ombre 今天发布。",
        [{"mention": "Ombre", "type": "person"}],
    )

    after = store.resolve_query("Ombre")
    assert [record.entity_id for record in linked] == list(before.entity_ids)
    assert after.entity_ids == before.entity_ids
    assert after.canonical_query == "Ombre Brain"


def test_write_recognizes_seed_alias_and_query_channels_share_buckets(tmp_path):
    store = EntityStore(_config(tmp_path, [_zhaodeng_seed()]))
    records = store.resolve_and_link("core-1", "朝灯今天做完了 Phase 2。")
    assert [record.canonical_name for record in records] == ["朝灯"]

    expected = ["core-1"]
    for query in ("Rosita", "朝灯", "老婆"):
        assert store.canonicalize_query(query) == "朝灯"
        assert store.linked_bucket_ids(query) == expected


def test_candidates_accept_mentions_but_reject_model_aliases_and_hallucinations(tmp_path):
    store = EntityStore(_config(tmp_path))
    record = store.resolve_and_link(
        "vae-1",
        "Vae 发布了新歌。",
        [{"mention": "Vae", "type": "person"}],
    )[0]
    assert record.canonical_name == "Vae"

    with pytest.raises(ValueError, match="cannot define aliases"):
        store.resolve_and_link(
            "bad-1",
            "Vae 发布了新歌。",
            [{"mention": "Vae", "type": "person", "aliases": ["许嵩"]}],
        )
    with pytest.raises(ValueError, match="not present"):
        store.resolve_and_link(
            "bad-2",
            "今天发布了新歌。",
            [{"mention": "Vae", "type": "person"}],
        )


def test_alias_boundaries_do_not_match_compounds_or_ascii_suffixes(tmp_path):
    store = EntityStore(
        _config(
            tmp_path,
            [
                _zhaodeng_seed(),
                {"canonical_name": "许嵩", "type": "person", "aliases": ["Vae"]},
            ],
        )
    )
    result = store.resolve_query("买老婆饼，听 Vae2 的歌")
    assert result.entity_ids == ()
    assert result.canonical_query == "买老婆饼，听 Vae2 的歌"
    assert store.canonicalize_query("OpenAI  API") == "OpenAI  API"

    result = store.resolve_query("老婆今天听Vae唱歌")
    assert set(result.terms) == {"朝灯", "许嵩"}
    assert result.canonical_query == "朝灯今天听许嵩唱歌"
    assert not entity_mention_present("老婆饼和朝灯", "老婆")
    assert entity_mention_present("老婆饼和朝灯", "朝灯")


def test_unicode_words_and_single_cjk_names_require_real_boundaries(tmp_path):
    store = EntityStore(_config(tmp_path))
    store.resolve_or_create("É", (), "person")
    store.resolve_or_create("朝", (), "person")
    store.resolve_or_create("朝灯", (), "person")
    store.resolve_or_create("苹果", (), "project")

    for query in (
        "caféine",
        "préface",
        "朝阳今天晴",
        "朝灯塔方向走",
        "苹果汁怎么做",
        "苹果酱配方",
    ):
        assert store.resolve_query(query).entity_ids == ()

    assert store.resolve_query("É").terms == ("É",)
    assert store.resolve_query("朝").terms == ("朝",)
    assert store.resolve_query("我和朝灯今天见").terms == ("朝灯",)


def test_content_hash_marks_stale_links_and_relink_replaces_old_entities(tmp_path):
    store = EntityStore(_config(tmp_path, [_zhaodeng_seed()]))
    old = "老婆今天很开心。"
    changed = "今天只讨论发布流程。"
    store.resolve_and_link("bucket-1", old)
    assert store.link_is_current("bucket-1", old)
    assert not store.link_is_current("bucket-1", changed)
    assert store.linked_bucket_ids(
        "老婆", content_hashes={"bucket-1": content_sha256(changed)}
    ) == []

    assert store.resolve_and_link("bucket-1", changed) == ()
    assert store.linked_bucket_ids("老婆") == []
    assert not store.link_is_current("bucket-1", changed)


def test_deleted_bucket_links_are_audited_and_removed(tmp_path):
    store = EntityStore(_config(tmp_path, [_zhaodeng_seed()]))
    store.resolve_and_link("bucket-1", "老婆记住了这件事。")

    assert store.unlink_bucket("bucket-1") == 1
    assert store.linked_bucket_ids("老婆") == []
    assert store.unlink_bucket("bucket-1") == 0


def test_link_hash_is_exact_content_not_normalized_content(tmp_path):
    store = EntityStore(_config(tmp_path, [_zhaodeng_seed()]))
    store.resolve_and_link("bucket-1", "老婆 今天")
    assert store.link_is_current("bucket-1", "老婆 今天")
    assert not store.link_is_current("bucket-1", "老婆\t今天")


def test_readonly_missing_or_corrupt_store_never_creates_or_repairs(tmp_path):
    config = _config(tmp_path)
    store = EntityStore(config, initialize=False)
    assert store.resolve_query("老婆").entity_ids == ()
    assert store.canonicalize_query("老婆") == "老婆"
    assert store.linked_bucket_ids("老婆") == []
    assert not store.link_is_current("none", "content")
    assert not Path(store.entities_dir).exists()

    Path(store.entities_dir).mkdir(mode=0o700)
    Path(store.db_path).write_bytes(b"not sqlite")
    os.chmod(store.db_path, 0o600)
    before = Path(store.db_path).read_bytes()
    assert store.resolve_query("老婆").entity_ids == ()
    assert Path(store.db_path).read_bytes() == before


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_initialize_refuses_symlink_database(tmp_path):
    config = _config(tmp_path)
    entities_dir = tmp_path / "buckets" / ".entities"
    entities_dir.mkdir(mode=0o700)
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    os.symlink(target, entities_dir / "entities.sqlite3")
    with pytest.raises(UnsafeEntityStore, match="non-regular"):
        EntityStore(config)


@pytest.mark.skipif(not hasattr(os, "link"), reason="hardlinks unavailable")
def test_initialize_refuses_hardlinked_database(tmp_path):
    config = _config(tmp_path)
    entities_dir = tmp_path / "buckets" / ".entities"
    entities_dir.mkdir(mode=0o700)
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    os.link(target, entities_dir / "entities.sqlite3")
    with pytest.raises(UnsafeEntityStore, match="hardlinked"):
        EntityStore(config)


def test_multiple_instances_initialize_and_write_concurrently(tmp_path):
    config = _config(tmp_path, [_zhaodeng_seed()])

    def work(index: int):
        store = EntityStore(config)
        store.resolve_and_link(f"bucket-{index}", f"老婆完成了第 {index} 项。")
        return store.resolve_query("Rosita").entity_ids

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(16)))
    assert all(result == results[0] and len(result) == 1 for result in results)

    store = EntityStore(config)
    assert store.linked_bucket_ids("朝灯") == sorted(f"bucket-{i}" for i in range(16))
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM entities").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM bucket_entities").fetchone()[0] == 16


def test_queries_do_not_change_database_bytes_or_mtime(tmp_path):
    store = EntityStore(_config(tmp_path, [_zhaodeng_seed()]))
    store.resolve_and_link("bucket-1", "老婆记住了这件事。")

    def fingerprint():
        return {
            path.name: (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_size,
                path.stat().st_mtime_ns,
                stat.S_IMODE(path.stat().st_mode),
            )
            for path in Path(store.entities_dir).iterdir()
            if path.is_file()
        }

    before = fingerprint()

    for _ in range(10):
        assert store.resolve_query("老婆").terms == ("朝灯",)
        assert store.canonicalize_query("Rosita") == "朝灯"
        assert store.linked_bucket_ids("朝灯") == ["bucket-1"]
        assert store.link_is_current("bucket-1", "老婆记住了这件事。")

    assert fingerprint() == before


def test_long_bucket_content_links_and_same_type_collision_skips(tmp_path):
    store = EntityStore(_config(tmp_path, [_zhaodeng_seed()]))
    long_content = "前情" * 300 + "，老婆完成了这件事。"

    linked = store.resolve_and_link("long-bucket", long_content)

    assert [record.canonical_name for record in linked] == ["朝灯"]
    assert store.link_is_current("long-bucket", long_content)

    store.resolve_or_create("甲", ["共同小名"], "person")
    store.resolve_or_create("乙", ["共同小名"], "person")
    assert store.resolve_and_link(
        "ambiguous-bucket",
        "共同小名今天来了。",
        [{"mention": "共同小名", "type": "person"}],
    ) == ()
