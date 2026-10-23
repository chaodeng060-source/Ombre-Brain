"""Provider switch tests: real gate entry, no second model on success."""
from types import SimpleNamespace

import pytest

import jev_memory_gate as gate
import server


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv("OMBRE_JEV_RERANK_ENABLED", raising=False)
    monkeypatch.delenv("OMBRE_JEV_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    gate._LAST.clear()
    yield
    gate._LAST.clear()


def bucket(name, text):
    return {"id": name, "metadata": {"name": name}, "content": text}


def test_switch_is_opt_in_and_env_can_roll_back():
    assert not gate.enabled({})
    assert gate.enabled({"jev_reranker": {"enabled": True}})


def test_explicit_env_false_overrides_config(monkeypatch):
    monkeypatch.setenv("OMBRE_JEV_RERANK_ENABLED", "0")
    assert not gate.enabled({"jev_reranker": {"enabled": True}})


def test_private_key_file_and_safe_errors(tmp_path):
    path = tmp_path / "jev.env"
    path.write_text("JEV_API_KEY=never-log-this-secret\n")
    path.chmod(0o600)
    config = {"jev_reranker": {"api_key_file": str(path)}}
    assert gate._key(config) == "never-log-this-secret"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="must be private") as exc:
        gate._key(config)
    assert "never-log" not in str(exc.value)


@pytest.mark.asyncio
async def test_complete_redacted_body_and_no_source_metadata(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "stub")
    captured = {}
    async def fake(query, candidates, **kwargs):
        captured.update(query=query, candidates=candidates, kwargs=kwargs)
        return SimpleNamespace(scores=[96], model="jev-1.14.0", requests=1,
                               cache_hits=0, input_tokens=27, output_tokens=4)
    monkeypatch.setattr(gate.jev_reranker, "score_candidates", fake)
    item = bucket("Title", "a" * 250 + " SECRET final original evidence")
    item["source_ref"] = "/private/raw/session-7"
    await gate.score({}, "SECRET question", [item],
                     redact=lambda s: s.replace("SECRET", "[redacted]"), timeout=12)
    assert captured["query"] == "[redacted] question"
    assert captured["candidates"][0].endswith("[redacted] final original evidence")
    assert len(captured["candidates"][0]) > 250
    assert "/private/raw" not in captured["candidates"][0]
    assert gate.status({})["last_batch"]["model"] == "jev-1.14.0"
    assert "stub" not in str(gate.status({}))
    assert "evidence" not in str(gate.status({}))


@pytest.mark.asyncio
async def test_native_success_skips_legacy_provider_and_sorts(monkeypatch):
    monkeypatch.setitem(server.config, "jev_reranker", {"enabled": True})
    monkeypatch.setenv("OMBRE_DS_GATE_SCORE_THRESHOLD", "70")
    def legacy():
        pytest.fail("successful JEV must not also call DeepSeek/Gemini")
    monkeypatch.setattr(server, "_ds_filter_provider", legacy)
    async def fake(*args, **kwargs):
        return SimpleNamespace(scores=[20, 94, 82])
    monkeypatch.setattr(gate, "score", fake)
    # Exercise the compatibility path even when the test is run on live NAS.
    monkeypatch.setattr(server, "_ds_score_mode_select", None, raising=False)
    items = [bucket("noise", "unrelated"), bucket("exact", "answer"), bucket("related", "related")]
    result = await server._ds_semantic_select("question", items, set(), 2)
    assert [b["id"] for b in result] == ["exact", "related"]


@pytest.mark.asyncio
async def test_production_retains_its_existing_score_selector(monkeypatch):
    monkeypatch.setitem(server.config, "jev_reranker", {"enabled": True})
    items = [bucket("a", "a"), bucket("b", "b")]
    async def fake(*args, **kwargs):
        return SimpleNamespace(scores=[93, 20])
    monkeypatch.setattr(gate, "score", fake)
    received = []
    def existing(buckets, scores, keep, limit):
        received.append((buckets, scores, keep, limit))
        return [buckets[0]]
    monkeypatch.setattr(server, "_ds_score_mode_select", existing, raising=False)
    monkeypatch.setattr(server, "_ds_gate_score_mode_enabled", lambda: True, raising=False)
    result = await server._ds_semantic_select("q", items, {"b"}, 1)
    assert result == [items[0]]
    assert received == [(items, [93, 20], {"b"}, 1)]


@pytest.mark.asyncio
async def test_explicit_fallback_line_does_not_repeat_jev(monkeypatch):
    monkeypatch.setitem(server.config, "jev_reranker", {"enabled": True})
    monkeypatch.setenv("OMBRE_DS_GATE_SCORE_MODE", "0")
    monkeypatch.setenv("OMBRE_DS_GATE_CITE_MODE", "0")
    monkeypatch.setenv("OMBRE_DS_FILTER_CACHE_TTL", "0")
    async def native(*args, **kwargs):
        pytest.fail("fallback must not resend the failed native JEV request")
    monkeypatch.setattr(gate, "score", native)
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"keep":[0]}'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    items = [bucket("a", "a"), bucket("b", "b")]
    result = await server._ds_semantic_select("q", items, set(), 2,
        provider_override=("fallback-deepseek", "deepseek-chat", client, {}))
    assert result == [items[0]]
    assert len(calls) == 1


def test_old_source_forced_contract_and_stable_ties(monkeypatch):
    monkeypatch.setenv("OMBRE_DS_GATE_SCORE_THRESHOLD", "70")
    items = [bucket("a", "a"), bucket("forced", "b"), bucket("c", "c")]
    result = gate.legacy_select(items, [90, 0, 90], {"forced"}, 2)
    assert [b["id"] for b in result] == ["a", "forced"]


def test_disabled_status_never_serializes_config_secret():
    status = gate.status({"jev_reranker": {"api_key_file": "/private/secret", "api_key": "hidden"}})
    assert status["provider"] == "legacy"
    assert "hidden" not in str(status) and "/private" not in str(status)
