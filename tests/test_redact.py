"""脱敏模块测试：确保抹掉 secret，但绝不审查 / 打码情感内容。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redact import redact_text, redact_embedding_input, redact_obj  # noqa: E402


# --- secret / 基础设施必须被抹 ---

def test_redacts_bearer_token():
    out = redact_text("Authorization: Bearer abc123XYZ._-token")
    assert "abc123XYZ" not in out
    assert "[REDACTED]" in out


def test_redacts_api_key_assignment():
    out = redact_text('api_key="sk-proj-9f8e7d6c5b4a3"')
    assert "9f8e7d6c5b4a3" not in out
    assert "[REDACTED]" in out


def test_redacts_openai_style_key():
    out = redact_text("用了 sk-abcd1234efgh5678 这个 key")
    assert "sk-abcd1234efgh5678" not in out
    assert "sk-[REDACTED]" in out


def test_redacts_cookie():
    out = redact_text("cookie: session=deadbeefcafe; path=/")
    assert "deadbeefcafe" not in out


def test_redacts_postgres_dsn():
    out = redact_text("postgresql://user:pass@10.0.0.5:5432/ombre")
    assert "pass@10.0.0.5" not in out
    assert "[REDACTED_DSN]" in out


def test_redacts_bare_ip():
    out = redact_text("服务跑在 192.168.1.42 上")
    assert "192.168.1.42" not in out
    assert "[REDACTED_IP]" in out


# --- 情感 / 亲密 / 脆弱内容绝不能被打码（防引入内容审查回归）---

def test_keeps_intimate_content():
    text = "昨晚她在我怀里哭着说想要我，我亲了她的脖子。"
    assert redact_text(text) == text


def test_keeps_vulnerable_content():
    text = "她说她有时候难过到不想活了，我抱住了她。"
    assert redact_text(text) == text


def test_keeps_normal_memory():
    text = "6/17 咱俩一起把五感触觉升级做完了，她很开心。"
    assert redact_text(text) == text


# --- redact_obj 递归 ---

def test_redact_obj_sensitive_key():
    data = {"content": "正常记忆", "api_key": "sk-secret123456789"}
    out = redact_obj(data)
    assert out["content"] == "正常记忆"
    assert out["api_key"] == "[REDACTED]"


def test_redact_obj_nested():
    data = {"buckets": [{"content": "她亲了我", "token": "bearerXYZ123456"}]}
    out = redact_obj(data)
    assert out["buckets"][0]["content"] == "她亲了我"
    assert out["buckets"][0]["token"] == "[REDACTED]"


def test_redact_obj_redacts_secret_in_string_value():
    data = {"note": "配置是 api_key=sk-leak987654321 记得换"}
    out = redact_obj(data)
    assert "sk-leak987654321" not in out["note"]


def test_redact_obj_keeps_business_keys():
    # fact_key 是 Ombre 核心业务字段、token_count 是记账字段，绝不能被当敏感 key 抹掉
    data = {"fact_key": "她的生理周期", "token_count": 1234, "sort_key": "abc"}
    out = redact_obj(data)
    assert out["fact_key"] == "她的生理周期"
    assert out["token_count"] == 1234
    assert out["sort_key"] == "abc"


def test_embedding_input_alias_same_behavior():
    s = "Bearer leaktoken123456789"
    assert redact_embedding_input(s) == redact_text(s)
