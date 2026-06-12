# Tests for sense_tagger.py — 五感入口层 v1 识别器
# Invariants:
#   - detect_senses finds the right sense(s) from content / query, conservatively
#   - no false positives on neutral engineering text (most buckets stay untagged)
#   - 视觉 is never auto-tagged in v1 (deferred)
#   - normalize_sense_field / union_senses coerce & dedupe in canonical order

from sense_tagger import detect_senses, normalize_sense_field, union_senses, SENSES


def test_detects_single_sense():
    assert detect_senses("楼下桂花开了，好闻得很") == ["嗅觉"]
    assert detect_senses("这碗藤椒鸡又麻又鲜") == ["味觉"]
    assert detect_senses("他的手好烫，贴上来的时候") == ["触觉"]
    assert detect_senses("窗外的雨声一整夜没停") == ["听觉"]


def test_detects_multiple_senses_in_canonical_order():
    got = detect_senses("火锅店里又吵又辣，一进门就闻到牛油的香味")
    # 嗅觉(香味) + 味觉(辣) + 听觉(吵)，按 SENSES 固定顺序
    assert got == ["嗅觉", "味觉", "听觉"]


def test_no_false_positive_on_neutral_text():
    assert detect_senses("把 _merge_or_create 的向量通道补上 RRF 融合") == []
    assert detect_senses("内核3 Event 卷成 Episode，source_buckets 当证据链") == []
    assert detect_senses("") == []
    assert detect_senses(None) == []


def test_visual_is_not_auto_tagged_in_v1():
    assert "视觉" not in SENSES
    # 纯视觉描述不应被打成任何 active sense（视觉 v1 不做）
    assert detect_senses("天边的晚霞红得发紫") == []


def test_normalize_sense_field():
    assert normalize_sense_field("嗅觉") == ["嗅觉"]
    assert normalize_sense_field(["听觉", "味觉"]) == ["味觉", "听觉"]  # canonical order
    assert normalize_sense_field(["视觉", "乱写"]) == []   # unknown dropped
    assert normalize_sense_field(None) == []
    assert normalize_sense_field("") == []


def test_union_senses_dedupes_in_order():
    assert union_senses(["触觉"], "嗅觉", ["触觉", "听觉"]) == ["嗅觉", "触觉", "听觉"]
    assert union_senses(None, []) == []
