"""
chord_tag 字段端到端测试。
chord_tag = 和弦记号串作为情绪坐标索引,只用于跨窗口标记。
"""
import pytest
import frontmatter
from bucket_manager import BucketManager


@pytest.fixture
def mgr(test_config):
    return BucketManager(test_config)


@pytest.mark.asyncio
async def test_create_without_chord_tag_omits_field(mgr):
    """不传 chord_tag 时,字段不应出现在 metadata 里(保持 frontmatter 简洁)。"""
    bid = await mgr.create(content="无色调", tags=["t"], domain=["d"])
    b = await mgr.get(bid)
    assert "chord_tag" not in b["metadata"]


@pytest.mark.asyncio
async def test_create_with_chord_tag_writes_field(mgr):
    """传 chord_tag 时正确写入 metadata 和 frontmatter。"""
    chord = "Em(maj7) → A13#11 → Dm6 → B7♭9 · 92bpm · f"
    bid = await mgr.create(
        content="带色调",
        tags=["t"],
        domain=["d"],
        chord_tag=chord,
    )
    b = await mgr.get(bid)
    assert b["metadata"]["chord_tag"] == chord


@pytest.mark.asyncio
async def test_create_with_empty_chord_tag_omits_field(mgr):
    """传空串或纯空格的 chord_tag 视为未传,字段不写入。"""
    bid_empty = await mgr.create(content="x", chord_tag="")
    bid_space = await mgr.create(content="y", chord_tag="   ")
    assert "chord_tag" not in (await mgr.get(bid_empty))["metadata"]
    assert "chord_tag" not in (await mgr.get(bid_space))["metadata"]


@pytest.mark.asyncio
async def test_create_strips_whitespace(mgr):
    """写入时 strip 掉首尾空格。"""
    bid = await mgr.create(content="x", chord_tag="  Cmaj7 · 60bpm  ")
    b = await mgr.get(bid)
    assert b["metadata"]["chord_tag"] == "Cmaj7 · 60bpm"


@pytest.mark.asyncio
async def test_update_sets_chord_tag(mgr):
    """update 可以给已有桶打/改 chord_tag。"""
    bid = await mgr.create(content="x")
    await mgr.update(bid, chord_tag="Fmaj9 · 60bpm · p")
    assert (await mgr.get(bid))["metadata"]["chord_tag"] == "Fmaj9 · 60bpm · p"
    # 改一次
    await mgr.update(bid, chord_tag="Am9 · 76bpm · mp")
    assert (await mgr.get(bid))["metadata"]["chord_tag"] == "Am9 · 76bpm · mp"


@pytest.mark.asyncio
async def test_update_can_clear_chord_tag(mgr):
    """update 传空串可以清空 chord_tag。"""
    bid = await mgr.create(content="x", chord_tag="Cmaj7")
    await mgr.update(bid, chord_tag="")
    # update 把字段写成空串(不删除字段本身),这对读取场景等价于"无"
    meta = (await mgr.get(bid))["metadata"]
    assert meta.get("chord_tag", "") == ""


@pytest.mark.asyncio
async def test_pinned_bucket_carries_chord_tag(mgr):
    """钉选桶(走 permanent_dir)也能带 chord_tag。"""
    bid = await mgr.create(
        content="重要时刻",
        tags=["重要"],
        domain=["pinned域"],
        pinned=True,
        chord_tag="Bm9 → Dmaj7 · 58bpm · p",
    )
    b = await mgr.get(bid)
    assert b["metadata"]["pinned"] is True
    assert b["metadata"]["chord_tag"] == "Bm9 → Dmaj7 · 58bpm · p"


@pytest.mark.asyncio
async def test_chord_tag_persists_in_frontmatter(mgr, tmp_path):
    """chord_tag 真正落到磁盘的 frontmatter 里,而不是内存里。"""
    chord = "Gmaj7add9 · 70bpm"
    bid = await mgr.create(content="持久化测试", chord_tag=chord)
    file_path = mgr._find_bucket_file(bid)
    assert file_path is not None
    with open(file_path) as f:
        post = frontmatter.load(f)
    assert post.get("chord_tag") == chord


@pytest.mark.asyncio
async def test_other_fields_unaffected_when_chord_tag_used(mgr):
    """加 chord_tag 不应影响 valence/arousal/tags/domain 等其他字段。"""
    bid = await mgr.create(
        content="x",
        tags=["a", "b"],
        domain=["d1"],
        valence=0.7,
        arousal=0.4,
        importance=8,
        chord_tag="Cmaj7",
    )
    m = (await mgr.get(bid))["metadata"]
    assert m["valence"] == 0.7
    assert m["arousal"] == 0.4
    assert m["importance"] == 8
    assert "a" in m["tags"] and "b" in m["tags"]
    assert "d1" in m["domain"]
    assert m["chord_tag"] == "Cmaj7"
