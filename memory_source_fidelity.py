"""Deterministic, deliberately limited checks for generated memory text.

An empty result means only that these local lexical checks found no issue; it
is not a semantic proof that a derivative is fully supported.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any


_ANIMAL_GROUPS = {
    "dog": ("边境牧羊犬", "边牧", "法斗", "柯基", "金毛", "拉布拉多", "柴犬", "萨摩耶", "哈士奇", "阿拉斯加", "狗狗", "狗子", "狗", "犬"),
    "cat": ("猫咪", "猫猫", "猫", "喵"),
    "rabbit": ("兔子", "兔兔", "家兔"),
    "bird": ("鹦鹉", "鸟儿", "小鸟"),
    "fox": ("狐狸",),
    "horse": ("马儿", "骏马", "一匹马"),
    "pig": ("猪猪", "小猪", "家猪", "野猪"),
    "cow": ("奶牛", "牛犊", "一头牛"),
    "hamster": ("仓鼠",),
    "turtle": ("乌龟", "海龟"),
    "fish": ("金鱼", "小鱼", "鲤鱼"),
}
_COLOR_PATTERNS = (
    ("red", re.compile(r"红(?:色|毛|棕|白|黑|陨石)")),
    ("orange", re.compile(r"(?:橘|橙)(?:色|黄|白|猫)")),
    ("yellow", re.compile(r"黄(?:色|毛|白|黑)")),
    ("green", re.compile(r"绿(?:色|毛)")),
    ("blue", re.compile(r"蓝(?:色|毛|灰)")),
    ("purple", re.compile(r"紫(?:色|毛)")),
    ("brown", re.compile(r"棕(?:色|白|黑|毛|褐)|咖啡色|奶咖色|巧克力色")),
    ("black", re.compile(r"黑(?:色|白|棕|灰|毛|眼圈)")),
    ("white", re.compile(r"白(?:色|毛|尾|爪|脸|肚|屁股|花|棕|黑)")),
    ("gray", re.compile(r"灰(?:色|白|黑|毛)|银灰色")),
)
_NUMBER = re.compile(
    r"(?:\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?|"
    r"\d+(?:\.\d+)?(?:岁|年|个月|月|天|日|次|只|张|个|条|分钟|小时)?|"
    r"[零〇一二两三四五六七八九十百千万]+(?:岁|年|个月|月|天|日|次|只|张|个|条|分钟|小时))"
)
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_NAMING = re.compile(r"(?:名叫|叫作|叫|名字是|名字叫|取名为|称为)\s*[「『“‘]?([\w\u4e00-\u9fff]{1,16})")
_QUOTED_NAME = re.compile(r"[「『“‘]([\w\u4e00-\u9fff]{1,16})[」』”’]")
_INCOMPLETE_MARKERS = (
    "[source_incomplete]",
    "[原文不完整]",
    "[截断原文]",
    "[summary_only]",
)
# Deliberately narrow source-audit constructions, not ordinary denials or
# uncertainty. Check generated clauses against original dialogue; never strip
# text or treat an audit phrase in metadata as original speech.
_EDITORIAL_SOURCE_AUDIT = re.compile(
    r"(?:(?:原话|原文|原始对话|来源文本)(?:中|里)?(?:并|也)?"
    r"(?:没有|未|无法|不能|不足以|不支持)"
    r"|(?:原话|原文)(?:表达|描述|记录|呈现|说明)"
    r"[^。！？；\n\"“‘「『]{0,60}(?:没有|并未|未说明|未确认|不支持)"
    r"|(?:不能|无法|不足以)(?:据此|由此)(?:确认|认定|推断|证明))"
    r"[^。！？；\n]*"
)


def _envelope_date(envelope: dict, payload: dict) -> str:
    for value in (
        envelope.get("speaker_date"),
        payload.get("recordedAt"),
        payload.get("ts"),
        envelope.get("recordedAt"),
        envelope.get("recorded_at"),
    ):
        if value is None or isinstance(value, bool):
            continue
        try:
            if isinstance(value, (int, float)):
                seconds = value / 1000 if value > 1e11 else value
                moment = datetime.fromtimestamp(seconds, tz=timezone.utc)
            else:
                moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return moment.astimezone(timezone(timedelta(hours=8))).date().isoformat()
        except (ValueError, OverflowError, OSError):
            continue
    return ""


def _envelope_text(envelope: dict) -> tuple[str, bool]:
    payload = envelope.get("payload")
    payload = payload if type(payload) is dict else envelope
    if any(
        obj.get("truncated") is True or obj.get("is_summary") is True
        for obj in (envelope, payload)
    ):
        return "", False
    spoken = next(
        (payload[field] for field in ("text", "message", "content", "body")
         if type(payload.get(field)) is str and payload[field].strip()),
        "",
    )
    if not spoken:
        return "", False
    parts = [spoken]
    source_date = _envelope_date(envelope, payload)
    if source_date:
        parts.append(source_date)
    contexts = envelope.get("context_events", [])
    if type(contexts) is not list:
        return "", False
    for context in contexts:
        if type(context) is not dict:
            return "", False
        text, complete = _envelope_text(context)
        if not complete:
            return "", False
        parts.append(text)
    return "\n".join(parts), True


def _dialogue_text(value: str) -> tuple[str, bool]:
    stripped = value.strip()
    if not stripped:
        return "", False
    try:
        parsed: Any = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        # A serialized event envelope cut off mid-JSON is not complete source.
        if stripped.startswith("{") and '"payload"' in stripped:
            return "", False
        return stripped, True
    if type(parsed) is not dict:
        return stripped, True
    return _envelope_text(parsed)


def _chinese_number(value: str) -> int | None:
    if not value:
        return None
    if all(char in _CN_DIGITS for char in value):
        return int("".join(str(_CN_DIGITS[char]) for char in value))
    total = section = number = 0
    for char in value:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _UNIT:
            unit = _UNIT[char]
            if unit == 10000:
                section = (section + (number or 1)) * unit
                total += section
                section = number = 0
            else:
                section += (number or 1) * unit
                number = 0
        else:
            return None
    return total + section + number


def _normalized_number(token: str) -> str:
    date_match = re.fullmatch(r"(\d{4})[-/.年](\d{1,2})(?:[-/.月](\d{1,2})日?)?", token)
    if date_match:
        year, month, day = date_match.groups()
        return f"{year}-{int(month):02d}" + (
            f"-{int(day):02d}" if day is not None else ""
        )
    match = re.fullmatch(r"([零〇一二两三四五六七八九十百千万]+)(.*)", token)
    if match:
        number = _chinese_number(match.group(1))
        if number is not None:
            return f"{number}{match.group(2)}"
    return token.replace("／", "/")


def _animal_groups(text: str) -> set[str]:
    return {
        group
        for group, terms in _ANIMAL_GROUPS.items()
        if any(term in text for term in terms)
    }


def _colors(text: str) -> set[str]:
    found: set[str] = set()
    for color, pattern in _COLOR_PATTERNS:
        if pattern.search(text):
            found.add(color)
    return found


def _numbers(text: str) -> set[str]:
    return {_normalized_number(match.group(0)) for match in _NUMBER.finditer(text)}


def _named_entities(text: str) -> set[str]:
    return {match.group(1).rstrip("，。！？；：、") for pattern in (_NAMING, _QUOTED_NAME) for match in pattern.finditer(text)}


def validate_derivative(text: str, originals: Sequence[str]) -> list[str]:
    """Check bounded fact classes and added source-audit commentary.

    ``originals`` must contain full raw message text or full serialized LMC-5
    event envelopes. Missing, malformed or explicitly incomplete source is
    reported as insufficient evidence. No model or paid service is called.
    """

    if type(text) is not str or not text.strip():
        return ["derivative_missing"]
    if isinstance(originals, (str, bytes)) or not isinstance(originals, Sequence) or not originals:
        return ["source_missing"]
    source_parts: list[str] = []
    for original in originals:
        if type(original) is not str or any(marker in original.lower() for marker in _INCOMPLETE_MARKERS):
            return ["source_incomplete"]
        part, complete = _dialogue_text(original)
        if not complete or not part.strip():
            return ["source_incomplete"]
        source_parts.append(part)
    source = "\n".join(source_parts)
    issues: list[str] = []
    for match in _EDITORIAL_SOURCE_AUDIT.finditer(text):
        clause = match.group(0).rstrip(' \t\r\"\'”’」』）)]')
        if not any(clause in part for part in source_parts):
            issues.append("unsupported_editorial_note")
            break
    unsupported_animals = _animal_groups(text) - _animal_groups(source)
    if unsupported_animals:
        issues.append("unsupported_animal:" + ",".join(sorted(unsupported_animals)))
    unsupported_colors = _colors(text) - _colors(source)
    if unsupported_colors:
        issues.append("unsupported_color:" + ",".join(sorted(unsupported_colors)))
    unsupported_numbers = _numbers(text) - _numbers(source)
    if unsupported_numbers:
        issues.append("unsupported_number:" + ",".join(sorted(unsupported_numbers)))
    unsupported_names = {
        name for name in _named_entities(text) if name not in source
    }
    if unsupported_names:
        issues.append("unsupported_named_entity:" + ",".join(sorted(unsupported_names)))
    return issues


__all__ = ["validate_derivative"]
