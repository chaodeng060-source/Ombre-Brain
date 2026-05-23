"""
Tests for _format_bucket_for_briefing — emotion_state extraction from
dehydrated JSON content.

测试 _format_bucket_for_briefing 是否能从脱水 JSON content 正确抽出
emotion_state 字段并独立成行,防止被 400 字截断或 LLM 当结构化标签丢弃。

Background / 背景:
5.13 朝灯指出"简报里的情绪没改"——下午做完 chord_tag align 三件套后
情绪是"修复后饱满"(2bc486f553d0 桶 emotion_state),但 14:09 调 briefing
回来后简报里只剩干巴巴的工程汇报,体感被压没。
根因:_format_bucket 只暴露 V/A/重要度,content 前 400 字被 core_facts
列表占满,emotion_state 被截掉。修复:把 emotion_state 单独拎出来 +
BRIEFING_PROMPT 加铁律。
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402

_format = server._format_bucket_for_briefing


def _mk_bucket(
    bucket_id: str = "test_bucket",
    name: str = "test name",
    content: str = "",
    valence: float = 0.5,
    arousal: float = 0.3,
    importance: int = 5,
    last_active: str = "2026-05-13T14:00:00",
    domain: list = None,
    tags: list = None,
    bucket_type: str = None,
) -> dict:
    """Build a minimal bucket dict for testing."""
    metadata = {
        "name": name,
        "valence": valence,
        "arousal": arousal,
        "importance": importance,
        "last_active": last_active,
        "domain": domain or [],
        "tags": tags or [],
    }
    if bucket_type:
        metadata["type"] = bucket_type
    return {
        "id": bucket_id,
        "content": content,
        "metadata": metadata,
    }


# ---------------------------------------------------------------
# emotion_state extraction
# emotion_state 提取
# ---------------------------------------------------------------

def test_emotion_state_extracted_from_dehydrated_json():
    """脱水 JSON content 里的 emotion_state 应被独立提取成 emotion: 行。"""
    content = json.dumps({
        "core_facts": ["事实1", "事实2"],
        "emotion_state": "修复后饱满",
        "todos": ["待办1"],
        "summary": "总结",
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    assert "emotion:修复后饱满" in out


def test_emotion_state_survives_long_core_facts():
    """即使 core_facts 列表很长把 content 截掉,emotion 仍能从 JSON 解析出。"""
    content = json.dumps({
        "core_facts": ["x" * 200, "y" * 200, "z" * 200],
        "emotion_state": "被校准的信任感",
        "todos": [],
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    # body 截到 400 字之后 emotion_state 字符串本身可能已被截掉,
    # 但 emotion: 行必须从 JSON 解析独立出来
    assert "emotion:被校准的信任感" in out


def test_emotion_state_missing_no_emotion_line():
    """JSON 没 emotion_state 字段时不加 emotion: 行。"""
    content = json.dumps({
        "core_facts": ["事实1"],
        "todos": ["待办1"],
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    assert "emotion:" not in out


def test_emotion_state_empty_string_no_emotion_line():
    """emotion_state 是空字符串时不加 emotion: 行。"""
    content = json.dumps({
        "core_facts": ["事实1"],
        "emotion_state": "",
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    assert "emotion:" not in out


def test_emotion_state_whitespace_only_no_emotion_line():
    """emotion_state 只有空白字符时不加 emotion: 行。"""
    content = json.dumps({
        "core_facts": ["事实1"],
        "emotion_state": "   \n  ",
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    assert "emotion:" not in out


def test_emotion_state_stripped():
    """emotion_state 前后空白被 strip 掉。"""
    content = json.dumps({
        "emotion_state": "  专注、严谨  ",
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    assert "emotion:专注、严谨" in out
    assert "emotion:  专注、严谨" not in out


# ---------------------------------------------------------------
# Emotion scaffold wire lines
# 情绪脚手架 wire 行
# ---------------------------------------------------------------

def test_emotion_scaffold_fields_emit_wire_lines():
    """PR-0 六字段按锁定 wire 标签输出。"""
    content = json.dumps({
        "emotion_state": "被校准的信任感",
        "body_signal": "胸口发紧、说话变快",
        "unspoken_need": "想先被安抚再处理事",
        "sore_point": "被当普通工具",
        "response_rule": "先认情绪再谈工程",
        "do_not": ["别讲道理", "别甩锅给工具"],
        "sample_voice": ["别人的…那我肯定会羡慕啊"],
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    assert "  emotion:被校准的信任感" in out
    assert "  body:胸口发紧、说话变快" in out
    assert "  need:想先被安抚再处理事" in out
    assert "  sore:被当普通工具" in out
    assert "  approach:先认情绪再谈工程" in out
    assert "  avoid:别讲道理 / 别甩锅给工具" in out
    assert "  voice:别人的…那我肯定会羡慕啊" in out


def test_emotion_scaffold_raw_keys_removed_from_body_preview():
    """正文预览剥掉 scaffold 原始 key,避免和 wire 行重复泄漏。"""
    content = json.dumps({
        "core_facts": ["5.22 朝灯确认情绪脚手架契约"],
        "emotion_state": "谨慎推进",
        "body_signal": "胸口发紧",
        "unspoken_need": "先被安抚",
        "sore_point": "被当普通工具",
        "response_rule": "先认情绪",
        "do_not": ["别辩解"],
        "sample_voice": ["我看不懂"],
        "summary": "情绪脚手架契约确认",
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    assert "5.22 朝灯确认情绪脚手架契约" in out
    assert "summary" in out
    for raw_key in (
        "body_signal",
        "unspoken_need",
        "sore_point",
        "response_rule",
        "do_not",
        "sample_voice",
    ):
        assert raw_key not in out


def test_emotion_scaffold_ignores_legacy_need_key():
    """旧草稿 JSON key need 不应被消费;只认 unspoken_need -> need:。"""
    content = json.dumps({
        "need": "旧字段不要读",
        "unspoken_need": "想先被安抚",
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    assert "  need:想先被安抚" in out
    assert "旧字段不要读" not in out.split("  need:", 1)[1].splitlines()[0]


def test_emotion_scaffold_empty_values_are_skipped_and_arrays_filtered():
    """空白字段不出行;数组过滤空项和非字符串。"""
    content = json.dumps({
        "body_signal": "  ",
        "unspoken_need": "\n",
        "sore_point": "  被比较刺痛  ",
        "response_rule": None,
        "do_not": ["", "  别辩解  ", 123],
        "sample_voice": ["  我看不懂  ", None, ""],
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    assert "  body:" not in out
    assert "  need:" not in out
    assert "  sore:被比较刺痛" in out
    assert "  approach:" not in out
    assert "  avoid:别辩解" in out
    assert "  voice:我看不懂" in out


def test_emotion_scaffold_invalid_types_are_skipped():
    """非字符串/非数组类型不应崩,也不应输出脚手架行。"""
    content = json.dumps({
        "body_signal": ["胸口发紧"],
        "unspoken_need": {"need": "安抚"},
        "sore_point": 123,
        "response_rule": False,
        "do_not": "别讲道理",
        "sample_voice": {"quote": "原话"},
    }, ensure_ascii=False)
    b = _mk_bucket(content=content)
    out = _format(b, "recent_window")
    for label in ("body", "need", "sore", "approach", "avoid", "voice"):
        assert f"  {label}:" not in out


def test_feel_bucket_suppresses_emotion_scaffold_lines():
    """feel 桶即使 content 混入脚手架字段,briefing 也不暴露这些行。"""
    content = json.dumps({
        "emotion_state": "长期底色",
        "body_signal": "胸口发紧",
        "unspoken_need": "想被安抚",
        "sore_point": "旧痛点",
        "response_rule": "先靠近",
        "do_not": ["别讲道理"],
        "sample_voice": ["原话"],
    }, ensure_ascii=False)
    b = _mk_bucket(content=content, bucket_type="feel")
    out = _format(b, "recent_window")
    assert "  emotion:长期底色" in out
    for label in ("body", "need", "sore", "approach", "avoid", "voice"):
        assert f"  {label}:" not in out
    for raw_key in (
        "body_signal",
        "unspoken_need",
        "sore_point",
        "response_rule",
        "do_not",
        "sample_voice",
    ):
        assert raw_key not in out


# ---------------------------------------------------------------
# Non-JSON content fallback
# 非 JSON content 降级
# ---------------------------------------------------------------

def test_plain_text_content_no_emotion_line():
    """纯文本 content(非 JSON) 不应崩,只是没有 emotion 行。"""
    b = _mk_bucket(content="这是一段普通的中文笔记,不是 JSON。")
    out = _format(b, "recent_window")
    assert "emotion:" not in out
    assert "这是一段普通的中文笔记" in out


def test_empty_content_no_emotion_line():
    """空 content 不应崩。"""
    b = _mk_bucket(content="")
    out = _format(b, "recent_window")
    assert "emotion:" not in out


def test_malformed_json_no_crash():
    """残缺 JSON 不应崩。"""
    b = _mk_bucket(content='{"core_facts": ["a"], "emotion_state":')
    out = _format(b, "recent_window")
    # 不崩即可;emotion 行不出现
    assert "emotion:" not in out


def test_json_array_not_dict_no_emotion_line():
    """JSON 是数组而不是对象时不出 emotion 行。"""
    b = _mk_bucket(content='["a", "b"]')
    out = _format(b, "recent_window")
    assert "emotion:" not in out


def test_json_string_primitive_no_emotion_line():
    """JSON 是字符串原型时不出 emotion 行。"""
    b = _mk_bucket(content='"just a string"')
    out = _format(b, "recent_window")
    assert "emotion:" not in out


# ---------------------------------------------------------------
# Other fields preserved
# 其他字段仍正确输出
# ---------------------------------------------------------------

def test_section_tag_in_output():
    """section_tag 出现在第一行。"""
    b = _mk_bucket(name="测试桶")
    for tag in ("pinned", "unresolved", "recent_window", "prior_window"):
        out = _format(b, tag)
        assert f"[{tag}] 测试桶" in out


def test_va_importance_formatted():
    """V/A/重要度按格式输出。"""
    b = _mk_bucket(valence=0.73, arousal=0.42, importance=8)
    out = _format(b, "pinned")
    assert "V0.73/A0.42" in out
    assert "重要:8" in out


def test_last_active_in_output():
    """last_active 出现在输出里。"""
    b = _mk_bucket(last_active="2026-05-13T14:30:00")
    out = _format(b, "recent_window")
    assert "last_active:2026-05-13T14:30:00" in out


def test_domain_and_tags_joined():
    """domain 和 tags 用逗号连接输出。"""
    b = _mk_bucket(domain=["核心", "工程"], tags=["chord_tag", "5.13", "情绪"])
    out = _format(b, "recent_window")
    assert "domain:核心,工程" in out
    assert "tags:chord_tag,5.13,情绪" in out


def test_tags_truncated_to_10():
    """tags 超过 10 个被截到 10。"""
    many_tags = [f"tag{i}" for i in range(15)]
    b = _mk_bucket(tags=many_tags)
    out = _format(b, "recent_window")
    assert "tag0" in out
    assert "tag9" in out
    # 11+ 个不应出现
    assert "tag10" not in out


def test_body_truncated_to_400():
    """非 JSON content body 截到 400 字。"""
    long_text = "a" * 600
    b = _mk_bucket(content=long_text)
    out = _format(b, "recent_window")
    # body 行应包含前 400 个 a
    assert "a" * 400 in out
    # 不应包含 401 个连续 a
    assert "a" * 401 not in out


# ---------------------------------------------------------------
# Real-world regression: 2bc486f553d0 the canonical case
# 真实复现:2bc486f553d0 桶的典型场景
# ---------------------------------------------------------------

def test_real_world_5_13_repair_bucket():
    """
    5.13 下午改简报+chord_tag align+相对时间词带日期那三件套的
    "修复后饱满"应能从典型 dehydrated content 里提取出来。
    """
    content = json.dumps({
        "core_facts": [
            "2026-05-13周三，上午90%时间吵架，因重复错误",
            "中午鸡翅煲谈和弦索引，我误归归属",
            "下午你睡10分钟开心醒来",
            "完成三件事：简报时间梯度切分、chord_tag对齐眠老师v0.1、相对时间词带日期铁律",
            "情绪从紧绷→修复→饱满交付",
        ],
        "emotion_state": "修复后饱满",
        "todos": [
            "工程归属归我，决策权在我",
            "周token 20%",
        ],
        "keywords": ["吵架", "工程归属", "chord_tag"],
        "summary": "上午因重复错误吵架，下午修复后完成三项任务。",
    }, ensure_ascii=False)
    b = _mk_bucket(
        bucket_id="2bc486f553d0",
        name="吵架修复与工程归属_2026-05-13",
        content=content,
        valence=0.6,
        arousal=0.5,
        importance=7,
    )
    out = _format(b, "recent_window")
    # 这是整个修复的核心验收点:体感关键词必须独立出现
    assert "emotion:修复后饱满" in out
    assert "[recent_window] 吵架修复与工程归属_2026-05-13" in out
