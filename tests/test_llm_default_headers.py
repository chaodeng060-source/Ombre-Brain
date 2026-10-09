from __future__ import annotations

from utils import llm_default_headers


def test_opencode_go_gets_stable_purpose_session_header() -> None:
    assert llm_default_headers(
        "https://opencode.ai/zen/go/v1", "night-run"
    ) == {"x-opencode-session": "ombre-brain-night-run"}


def test_other_providers_and_invalid_urls_get_no_extra_headers() -> None:
    assert llm_default_headers("https://api.deepseek.com/v1", "night-run") is None
    assert llm_default_headers("https://opencode.ai.evil.test/zen/go", "night-run") is None
    assert llm_default_headers("not a url", "night-run") is None
