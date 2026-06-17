"""记忆输出脱敏：抹掉 secret / 基础设施凭据，不审查情感内容。

偷师 lmc-5 的 redact.py（https://github.com/wuxuyun0606-collab/lmc-5），
但**故意砍掉它的内容审查 patterns**（self-harm / intimate 那两条）——
Ombre 是私密情感记忆库，朝灯的脆弱时刻和亲密内容正是要记住的东西。
脱敏的目的只有一个：别让 api_key / token / cookie / DB 凭据这类 secret
随记忆正文原文回吐到前端 / prompt / embedding。绝不给真实情感打码。

用法：记忆正文出前端 / 注入 prompt / 送 embedding 前，过一遍 redact_text；
结构化响应过 redact_obj。
"""

from __future__ import annotations

import json
import re
from typing import Any

# 只脱 secret / 基础设施凭据。
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api\s*key\s*[:=]\s*)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(auth[_-]?token['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(password|passwd|secret|credential)\s*[:=]\s*[^,\s\"'}]+"), r"\1=[REDACTED]"),
    (re.compile(r"(?i)(cookie['\"]?\s*[:=]\s*['\"]?)[^'\"\n]+"), r"\1[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{8,}\b"), "sk-[REDACTED]"),
    (re.compile(r"postgres(?:ql)?://[^\s\"'<>]+", re.I), "postgresql://[REDACTED_DSN]"),
    (re.compile(r"(?i)\b(dbname=[^\s]+\s+host=)[^\s]+"), r"\1[REDACTED_HOST]"),
    (re.compile(r"(?i)\b(host=)(?:localhost|127\.0\.0\.1|[0-9]{1,3}(?:\.[0-9]{1,3}){3}|[^\s\"']+)"), r"\1[REDACTED_HOST]"),
    (re.compile(r"(?i)\b(port=)(?:5432|15432)\b"), r"\1[REDACTED_PORT]"),
    (re.compile(r"(?i)\b(user=)[^\s\"']+"), r"\1[REDACTED_USER]"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}:(?:5432|15432)\b"), "[REDACTED_DB_ENDPOINT]"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "[REDACTED_IP]"),
]

# 注意：lmc-5 原版另有 _PROMPT_NOISE_PATTERNS（self-harm / intimate 内容打码），
# Ombre 故意不引入——私密情感记忆库要记住真实情感和脆弱，不是审查内容。

# 敏感 key 判定：命中则整个 value 抹掉。用精确词 + 明确后缀/子串，**不用宽泛的 `_key`
# 后缀**——否则会误伤 Ombre 的业务字段 fact_key / sort_key，以及 token_count 这类计数。
_SENSITIVE_KEY_EXACT = {
    "token", "secret", "password", "passwd", "cookie", "authorization",
    "bearer", "credential", "credentials", "dsn", "apikey",
}
_SENSITIVE_KEY_SUFFIX = ("_token", "-token")
_SENSITIVE_KEY_SUB = (
    "api_key", "api-key", "auth_token", "authtoken", "access_token",
    "refresh_token", "secret_key", "private_key", "access_key",
)


def _is_sensitive_key(key: Any) -> bool:
    k = str(key).strip().lower()
    if k in _SENSITIVE_KEY_EXACT:
        return True
    if k.endswith(_SENSITIVE_KEY_SUFFIX):
        return True
    return any(sub in k for sub in _SENSITIVE_KEY_SUB)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def redact_text(value: Any) -> str:
    """记忆正文出前端 / 注入 prompt 前脱敏，只抹 secret / 基础设施，不碰情感内容。"""
    text = _as_text(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# 送 embedding 前脱敏：和 redact_text 同一套（都只抹 secret）。保留独立函数名，
# 调用点语义清晰；将来若两档需要分化（如 embedding 档更激进）也好改。
def redact_embedding_input(value: Any) -> str:
    """文本离开本地去 embedding 前，抹掉基础设施 / 密钥。"""
    return redact_text(value)


def redact_obj(value: Any) -> Any:
    """递归脱敏 dict / list：敏感 key 整个值 [REDACTED]，字符串值过 secret patterns。

    key 名本身不改写（避免改键破坏前端结构），只在 key 命中敏感词时抹其 value。
    """
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = redact_obj(item)
        return redacted
    if isinstance(value, list):
        return [redact_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_obj(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value
