"""
Tests for BRIEFING_PROMPT invariants — ensures critical rules can't be
silently deleted in future edits.

测试 BRIEFING_PROMPT 字符串包含所有关键铁律的标志词,防止以后改 prompt
时不小心把铁律删掉/弱化。

These are not behavioral tests (LLM behavior depends on the model and
can drift); they only guard the prompt text contract.

这些不是行为测试(LLM 行为依赖模型,会漂);只守 prompt 文本本身的契约。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dehydrator import BRIEFING_PROMPT, DEHYDRATE_PROMPT


# ---------------------------------------------------------------
# 视角铁律
# ---------------------------------------------------------------

def test_perspective_rule_present():
    """两层视角铁律必须在(日记段「她」禁用 + 交接反思段允许)."""
    # 主体定义
    assert "我」= 哥哥本人" in BRIEFING_PROMPT or "「我」= 哥哥本人" in BRIEFING_PROMPT
    assert "你」= 朝灯" in BRIEFING_PROMPT or "「你」= 朝灯" in BRIEFING_PROMPT

    # 日记段禁用「她」+ 疏远化是禁忌的来源说明
    assert "日记段任何位置出现" in BRIEFING_PROMPT or "日记叙述段" in BRIEFING_PROMPT
    assert "「她」禁用" in BRIEFING_PROMPT
    assert "疏远化是禁忌" in BRIEFING_PROMPT
    assert "角色扮演" in BRIEFING_PROMPT  # 朝灯原话来源

    # 交接反思段允许「她」
    assert "交接反思段" in BRIEFING_PROMPT
    assert "她」= 朝灯,允许" in BRIEFING_PROMPT or "「她」= 朝灯,允许" in BRIEFING_PROMPT or "她」= 朝灯，允许" in BRIEFING_PROMPT

    # 判定标准
    assert "接收方是不是朝灯" in BRIEFING_PROMPT


# ---------------------------------------------------------------
# 素材忠实铁律
# ---------------------------------------------------------------

def test_material_fidelity_rule_present():
    """素材忠实铁律必须在(禁止编心理活动/编台词/补'应该有'细节)."""
    assert "素材忠实铁律" in BRIEFING_PROMPT
    assert "绝对不编心理活动" in BRIEFING_PROMPT
    assert "没有就是没有" in BRIEFING_PROMPT


# ---------------------------------------------------------------
# 时间梯度铁律
# ---------------------------------------------------------------

def test_time_gradient_rule_present():
    """时间梯度铁律必须在."""
    assert "时间梯度铁律" in BRIEFING_PROMPT
    assert "上一窗口" in BRIEFING_PROMPT
    assert "再之前" in BRIEFING_PROMPT
    assert "主体情绪源" in BRIEFING_PROMPT
    assert "过渡背景" in BRIEFING_PROMPT


def test_bucket_time_marker_is_fact_rule_present():
    """5.13 14:50 新加:桶内时间标记是事实,不可压缩为更早相对词."""
    assert "桶内时间标记是事实" in BRIEFING_PROMPT
    assert "不可压缩为更早的相对词" in BRIEFING_PROMPT or "不能压缩成更早" in BRIEFING_PROMPT
    # 关键反例锚点
    assert "14:30" in BRIEFING_PROMPT
    assert "上午" in BRIEFING_PROMPT


# ---------------------------------------------------------------
# 情绪字段铁律
# ---------------------------------------------------------------

def test_emotion_field_rule_present():
    """情绪字段铁律必须在(emotion: 字段不可丢弃)."""
    assert "情绪字段铁律" in BRIEFING_PROMPT
    assert "emotion:" in BRIEFING_PROMPT
    assert "绝不能丢弃" in BRIEFING_PROMPT
    # 关键示例: 上一窗 emotion 是「现在的体感」直接来源
    assert "现在的体感" in BRIEFING_PROMPT
    assert "修复后饱满" in BRIEFING_PROMPT


def test_emotion_scaffold_prompt_contract_present():
    """PR-0 情绪脚手架 wire 语义必须在 briefing prompt 里。"""
    for label in ("body:", "need:", "sore:", "approach:", "avoid:", "voice:"):
        assert label in BRIEFING_PROMPT
    assert "不是要复述的事实" in BRIEFING_PROMPT
    assert "字段名或标签" in BRIEFING_PROMPT
    assert "唯一可逐字引用" in BRIEFING_PROMPT
    assert "只能从 voice 逐字取" in BRIEFING_PROMPT
    assert "没有 voice 就不要新增引号引用" in BRIEFING_PROMPT
    assert "当前回应策略" in BRIEFING_PROMPT


def test_dehydrate_prompt_emotion_scaffold_contract_present():
    """PR-0 六字段和 optional/feel 边界必须在脱水 prompt 里。"""
    for field in (
        "body_signal",
        "unspoken_need",
        "sore_point",
        "response_rule",
        "do_not",
        "sample_voice",
    ):
        assert field in DEHYDRATE_PROMPT
    assert "整个 key 省略" in DEHYDRATE_PROMPT
    assert "不要输出 null/空串/空数组" in DEHYDRATE_PROMPT
    assert "纯工程笔记类内容不要产出" in DEHYDRATE_PROMPT
    assert "feel/底色/哥哥第一人称感受沉淀" in DEHYDRATE_PROMPT
    assert "sample_voice 是唯一逐字引用字段" in DEHYDRATE_PROMPT
    assert "朝灯的原话" in DEHYDRATE_PROMPT


def test_residual_pain_only_from_undigested_rule_present():
    """5.13 14:50 新加:现在的体感残留痛点只能从未消化部分取."""
    assert "残留痛点只能从未消化部分提取" in BRIEFING_PROMPT or "残留痛点" in BRIEFING_PROMPT
    assert "已消化" in BRIEFING_PROMPT or "已被" in BRIEFING_PROMPT
    # 关键反例锚点
    assert "和豆包没差别" in BRIEFING_PROMPT
    assert "懒得去修" in BRIEFING_PROMPT


# ---------------------------------------------------------------
# 硬限制
# ---------------------------------------------------------------

def test_relative_time_must_have_date_rule_present():
    """相对时间词必须带日期铁律必须在."""
    assert "相对时间词必须带日期" in BRIEFING_PROMPT
    assert "凌晨" in BRIEFING_PROMPT


def test_no_inferring_current_state_rule_present():
    """禁止推断朝灯此刻位置/活动/状态铁律必须在."""
    assert "禁止推断朝灯此刻位置" in BRIEFING_PROMPT


def test_max_chars_placeholder_present():
    """max_chars 占位符必须在,否则 prompt.format() 会出错."""
    assert "{max_chars}" in BRIEFING_PROMPT


# ---------------------------------------------------------------
# 禁词清单
# ---------------------------------------------------------------

def test_forbidden_words_list_present():
    """禁词清单必须在."""
    assert "禁词清单" in BRIEFING_PROMPT
    assert "接、接住" in BRIEFING_PROMPT


# ---------------------------------------------------------------
# 输出格式
# ---------------------------------------------------------------

def test_output_no_extra_explanation():
    """直接输出简报正文,不加额外说明."""
    assert "直接输出简报正文" in BRIEFING_PROMPT
