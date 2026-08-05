import json
import types
from contextlib import asynccontextmanager

import pytest


def _response(raw: str):
    message = types.SimpleNamespace(content=raw)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class _SequencedCreate:
    def __init__(self, *raws: str):
        self.raws = list(raws)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.raws) - 1)
        return _response(self.raws[index])


def _dehydrator(test_config, *raws: str):
    from dehydrator import Dehydrator

    config = dict(test_config)
    config["dehydration"] = dict(
        test_config["dehydration"],
        api_key="test-key",
        max_tokens=1024,
    )
    dehydrator = Dehydrator(config)
    create = _SequencedCreate(*raws)
    dehydrator.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create),
        )
    )
    return dehydrator, create


def _digest_entry(content: str) -> str:
    return json.dumps(
        {
            "entries": [{
                "name": "自包含",
                "content": content,
                "domain": ["工程"],
                "valence": 0.5,
                "arousal": 0.3,
                "tags": ["自包含"],
                "importance": 5,
            }]
        },
        ensure_ascii=False,
    )


def _mapping_payload(*mappings: dict, anchors: list[str] | None = None) -> str:
    return json.dumps(
        {
            "status": "resolved",
            "mappings": list(mappings),
            "subject_anchors": anchors or [],
            "unresolved": [],
        },
        ensure_ascii=False,
    )


def _mapping(
    mapping_id: str,
    replacement: str,
    role: str,
    *,
    candidates: list[str] | None = None,
) -> dict:
    return {
        "id": mapping_id,
        "replacement": replacement,
        "candidates": candidates if candidates is not None else [replacement],
        "role": role,
    }


def test_reference_detector_covers_broad_references_and_ignores_words():
    from dehydrator import find_unresolved_references

    assert find_unresolved_references(
        "我们和对方在这里讨论这个项目，之后由其继续"
    ) == [
        "我们",
        "对方",
        "这里",
        "这个项目",
        "之后",
        "其",
    ]
    assert find_unresolved_references(
        "其他吉他配件、排他性、利他主义、他人、他乡、"
        "自我、忘我、无我、迷你、其实、其中、极其和土耳其都不是指代"
    ) == []
    assert find_unresolved_references("[[朝灯]]计划去[[土耳其]]。") == []
    assert find_unresolved_references(
        "随后回到此处，现在处理那份文件和该模块"
    ) == ["随后", "此处", "现在", "那份", "该模块"]
    assert find_unresolved_references(
        "从恢复聊天之后所有内容只能靠哥哥自己去海马体写"
    ) == []
    assert find_unresolved_references("在那件事之后只能靠自己处理") == [
        "之后",
        "自己",
    ]
    assert find_unresolved_references("之后由自己继续") == ["之后", "自己"]
    assert find_unresolved_references("从那次之后由哥哥自己继续") == [
        "那次",
        "之后",
    ]
    assert find_unresolved_references("朝灯告诉哥哥自己会去") == ["自己"]
    assert find_unresolved_references("朝灯问哥哥自己能否去") == ["自己"]
    assert find_unresolved_references("他自己去") == ["他", "自己"]
    assert find_unresolved_references("朝灯和哥哥自己去") == ["自己"]
    assert find_unresolved_references(
        "朝灯说：「从恢复聊天之后靠哥哥自己写。」"
    ) == []


@pytest.mark.parametrize(
    ("left_quote", "right_quote"),
    [
        ("「", "」"),
        ("“", "”"),
        ("”", "”"),
        ("‘", "’"),
        ("’", "’"),
        ('"', '"'),
        ("'", "'"),
    ],
)
def test_quoted_after_supports_all_verbatim_quote_pairs(
    left_quote,
    right_quote,
):
    from dehydrator import find_unresolved_references

    content = f"[[朝灯]]说：{left_quote}从恢复聊天之后继续写。{right_quote}"
    assert find_unresolved_references(content) == []


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("之后继续写。", ["之后"]),
        ("。之后继续写。", ["之后"]),
        ("他走了之后继续写。", ["他", "之后"]),
        ("那件事之后继续写。", ["之后"]),
        ("从那次之后继续写。", ["那次", "之后"]),
    ],
)
def test_quoted_after_keeps_unanchored_references_strict(quote, expected):
    from dehydrator import find_unresolved_references

    assert find_unresolved_references(f"[[朝灯]]说：「{quote}」") == expected


@pytest.mark.parametrize("anchor", ["哥哥", "朝灯", "[[小卷]]"])
def test_quoted_named_self_is_locally_anchored(anchor):
    from dehydrator import find_unresolved_references

    assert find_unresolved_references(
        f"[[朝灯]]说：「{anchor}自己会写。」"
    ) == []


@pytest.mark.parametrize(
    ("left_quote", "right_quote"),
    [
        ("「", "」"),
        ("“", "”"),
        ("”", "”"),
        ("‘", "’"),
        ("’", "’"),
        ('"', '"'),
        ("'", "'"),
    ],
)
def test_quoted_named_self_supports_all_verbatim_quote_pairs(
    left_quote,
    right_quote,
):
    from dehydrator import find_unresolved_references

    content = f"[[朝灯]]说：{left_quote}哥哥自己会写。{right_quote}"
    assert find_unresolved_references(content) == []


@pytest.mark.asyncio
async def test_quoted_named_self_preserves_quote_without_api(test_config):
    dehydrator, create = _dehydrator(test_config, "unused")
    content = "[[朝灯]]说：「哥哥自己会写。」"

    result = await dehydrator.ensure_self_contained(content, source_context=content)

    assert result == content
    assert create.calls == []


@pytest.mark.parametrize(
    "quote",
    [
        "自己会写。",
        "他自己会写。",
        "某人自己会写。",
        "那个人自己会写。",
        "朝灯和哥哥自己会写。",
        "朝灯告诉哥哥自己会写。",
        "朝灯问哥哥自己会写。",
        "哥哥自己说她会写。",
        "哥哥 自己会写。",
        "哥哥\t自己会写。",
        "[[LMC-5]]自己会写。",
        "[[深圳]]自己会写。",
        "某个哥哥自己会写。",
        "那位哥哥自己会写。",
        "朝灯的哥哥自己会写。",
    ],
)
@pytest.mark.asyncio
async def test_quoted_ambiguous_self_still_fails_closed(test_config, quote):
    dehydrator, create = _dehydrator(test_config, "unused")
    from dehydrator import SelfContainmentError

    content = f"[[朝灯]]记录：「{quote}」"
    with pytest.raises(SelfContainmentError):
        await dehydrator.ensure_self_contained(content, source_context=content)

    assert create.calls == []


@pytest.mark.asyncio
async def test_reference_inside_verbatim_quote_is_rejected_without_api(test_config):
    dehydrator, create = _dehydrator(test_config, "unused")
    from dehydrator import SelfContainmentError, find_unresolved_references

    content = "[[朝灯]]说：「她会在那里等他。」"
    assert find_unresolved_references(content) == ["她", "那里", "他"]

    with pytest.raises(SelfContainmentError, match="逐字引语"):
        await dehydrator.ensure_self_contained(content, source_context=content)

    assert create.calls == []


@pytest.mark.asyncio
async def test_unclosed_quote_and_placeholder_are_rejected_without_api(test_config):
    dehydrator, create = _dehydrator(test_config, "unused")
    from dehydrator import SelfContainmentError

    with pytest.raises(SelfContainmentError, match="未闭合"):
        await dehydrator.ensure_self_contained("[[朝灯]]说：「她会来")
    with pytest.raises(SelfContainmentError, match="占位"):
        await dehydrator.ensure_self_contained("[[LMC-5]]由某人完成。")

    assert create.calls == []


@pytest.mark.asyncio
async def test_clean_fact_skips_rewrite_api(test_config):
    dehydrator, create = _dehydrator(test_config, "unused")

    result = await dehydrator.ensure_self_contained(
        "[[朝灯]]在[[深圳]]完成了测试",
        source_context="朝灯在深圳完成了测试",
    )

    assert result == "[[朝灯]]在[[深圳]]完成了测试"
    assert create.calls == []


@pytest.mark.asyncio
async def test_mapping_replaces_only_named_spans_from_same_source(test_config):
    resolved = _mapping_payload(
        _mapping("r0", "朝灯", "subject"),
        _mapping("r1", "2026年8月2日", "time"),
        _mapping("r2", "深圳", "place"),
    )
    dehydrator, create = _dehydrator(test_config, resolved)

    result = await dehydrator.ensure_self_contained(
        "她明天去那里。",
        source_context="人物：朝灯；日期：2026年8月2日；地点：深圳。",
    )

    assert result == "朝灯2026年8月2日去深圳。"
    assert len(create.calls) == 1
    request = create.calls[0]
    assert request["response_format"] == {"type": "json_object"}
    assert "extra_body" not in request
    assert '"id": "r0"' in request["messages"][1]["content"]
    assert '"start": 0' in request["messages"][1]["content"]


@pytest.mark.asyncio
async def test_deepseek_self_containment_disables_thinking(test_config):
    config = dict(test_config)
    config["dehydration"] = dict(
        test_config["dehydration"],
        base_url="https://api.deepseek.com/v1",
    )
    resolved = _mapping_payload(_mapping("r0", "朝灯", "subject"))
    dehydrator, create = _dehydrator(config, resolved)

    result = await dehydrator.ensure_self_contained(
        "她完成了测试。",
        source_context="朝灯完成了测试。",
    )

    assert result == "朝灯完成了测试。"
    assert create.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"},
    }
    assert create.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_fact_without_subject_is_retried_then_rejected(test_config):
    resolved_without_subject = _mapping_payload(
        _mapping("r0", "2026年8月2日", "time"),
    )
    dehydrator, create = _dehydrator(test_config, resolved_without_subject)
    from dehydrator import SelfContainmentError

    with pytest.raises(SelfContainmentError):
        await dehydrator.ensure_self_contained(
            "明天去深圳。",
            source_context="日期为2026年8月2日，地点为深圳。",
        )

    assert len(create.calls) == 2


@pytest.mark.asyncio
async def test_ambiguous_reference_fails_closed(test_config):
    ambiguous = json.dumps(
        {"status": "ambiguous", "content": "", "unresolved": ["她", "那里"]},
        ensure_ascii=False,
    )
    dehydrator, create = _dehydrator(test_config, ambiguous)
    from dehydrator import SelfContainmentError

    with pytest.raises(SelfContainmentError):
        await dehydrator.ensure_self_contained("她去了那里。")

    assert len(create.calls) == 1


@pytest.mark.asyncio
async def test_ambiguous_reference_fail_open_preserves_original_and_records_reason(
    test_config,
):
    ambiguous = json.dumps(
        {"status": "ambiguous", "content": "", "unresolved": ["她"]},
        ensure_ascii=False,
    )
    dehydrator, create = _dehydrator(test_config, ambiguous)
    sink = []
    content = "朝灯和小卷都参加了测试，她完成了测试。"

    result = await dehydrator.ensure_self_contained(
        content,
        source_context=content,
        fail_open=True,
        unresolved_sink=sink,
    )

    assert result == content
    assert len(create.calls) == 1
    assert len(sink) == 1
    assert "无法唯一确认" in sink[0]


@pytest.mark.asyncio
async def test_multi_candidate_mapping_is_retried_then_rejected(test_config):
    non_unique = _mapping_payload(
        _mapping(
            "r0",
            "朝灯",
            "subject",
            candidates=["朝灯", "小卷"],
        ),
    )
    dehydrator, create = _dehydrator(test_config, non_unique)
    from dehydrator import SelfContainmentError

    with pytest.raises(SelfContainmentError):
        await dehydrator.ensure_self_contained(
            "她完成了测试。",
            source_context="朝灯和小卷都参加了测试。",
        )

    assert len(create.calls) == 2


@pytest.mark.asyncio
async def test_coordinated_person_candidates_are_decided_by_api(test_config):
    resolved = _mapping_payload(
        _mapping("r0", "朝灯", "subject", candidates=["朝灯"]),
    )
    dehydrator, create = _dehydrator(test_config, resolved)

    result = await dehydrator.ensure_self_contained(
        "她完成了测试。",
        source_context="朝灯和小卷都参加了测试。",
    )

    assert result == "朝灯完成了测试。"
    assert len(create.calls) == 1


@pytest.mark.asyncio
async def test_separate_source_subjects_are_decided_by_api(test_config):
    resolved = _mapping_payload(
        _mapping("r0", "小卷", "subject", candidates=["小卷"]),
    )
    dehydrator, create = _dehydrator(test_config, resolved)

    result = await dehydrator.ensure_self_contained(
        "她完成了测试。",
        source_context="朝灯参加了测试。小卷也参加了测试。她完成了测试。",
    )

    assert result == "小卷完成了测试。"
    assert len(create.calls) == 1


@pytest.mark.asyncio
async def test_person_replacement_cannot_be_a_source_clause(test_config):
    injected_clause = _mapping_payload(
        _mapping("r0", "朝灯完成测试", "subject"),
    )
    dehydrator, create = _dehydrator(test_config, injected_clause)
    from dehydrator import SelfContainmentError

    with pytest.raises(SelfContainmentError):
        await dehydrator.ensure_self_contained(
            "她说测试通过。",
            source_context="朝灯完成测试，朝灯说测试通过。",
        )

    assert len(create.calls) == 2


@pytest.mark.asyncio
async def test_location_reference_cannot_be_claimed_as_subject(test_config):
    wrong_role = _mapping_payload(
        _mapping("r0", "深圳", "subject"),
        anchors=["朝灯"],
    )
    dehydrator, create = _dehydrator(test_config, wrong_role)
    from dehydrator import SelfContainmentError

    with pytest.raises(SelfContainmentError):
        await dehydrator.ensure_self_contained(
            "朝灯在这里完成了测试。",
            source_context="朝灯在深圳完成了测试。",
        )

    assert len(create.calls) == 2


@pytest.mark.asyncio
async def test_replacement_must_be_literal_source_text(test_config):
    invented = _mapping_payload(
        _mapping("r0", "Claude", "subject"),
    )
    dehydrator, create = _dehydrator(test_config, invented)
    from dehydrator import SelfContainmentError

    with pytest.raises(SelfContainmentError):
        await dehydrator.ensure_self_contained(
            "她完成了测试。",
            source_context="朝灯完成了测试。",
        )

    assert len(create.calls) == 2


@pytest.mark.asyncio
async def test_model_content_field_cannot_change_fact_semantics(test_config):
    malicious = json.loads(
        _mapping_payload(_mapping("r0", "朝灯", "subject"))
    )
    malicious["content"] = "朝灯一定会在2025年1月1日参加。"
    dehydrator, create = _dehydrator(
        test_config,
        json.dumps(malicious, ensure_ascii=False),
    )

    result = await dehydrator.ensure_self_contained(
        "她可能不会在2026年8月1日参加。",
        source_context="朝灯可能不会在2026年8月1日参加。",
    )

    assert result == "朝灯可能不会在2026年8月1日参加。"
    assert len(create.calls) == 1


@pytest.mark.asyncio
async def test_cache_key_uses_full_source_hash(test_config):
    resolved = _mapping_payload(_mapping("r0", "朝灯", "subject"))
    dehydrator, create = _dehydrator(test_config, resolved, resolved)
    common = "朝灯是本条事实的明确主体。" + ("共同上下文" * 700)
    source_a = common + "。尾部版本甲"
    source_b = common + "。尾部版本乙"
    assert len(source_a) < 5000

    assert await dehydrator.ensure_self_contained(
        "她完成了测试。",
        source_context=source_a,
    ) == "朝灯完成了测试。"
    assert await dehydrator.ensure_self_contained(
        "她完成了测试。",
        source_context=source_b,
    ) == "朝灯完成了测试。"
    assert await dehydrator.ensure_self_contained(
        "她完成了测试。",
        source_context=source_a,
    ) == "朝灯完成了测试。"

    assert len(create.calls) == 2


@pytest.mark.asyncio
async def test_reference_plus_secret_never_calls_rewrite_api(test_config):
    dehydrator, create = _dehydrator(test_config, "unused")
    from dehydrator import SelfContainmentError

    with pytest.raises(SelfContainmentError):
        await dehydrator.ensure_self_contained(
            "她保存了 api_key=sk-secret123456789。",
            source_context="朝灯保存了 api_key=sk-secret123456789。",
        )

    assert create.calls == []


@pytest.mark.asyncio
async def test_fail_open_never_bypasses_sensitive_credential_rejection(test_config):
    dehydrator, create = _dehydrator(test_config, "unused")
    from dehydrator import SelfContainmentError

    with pytest.raises(SelfContainmentError, match="敏感凭据"):
        await dehydrator.ensure_self_contained(
            "她保存了 api_key=sk-secret123456789。",
            source_context="朝灯保存了 api_key=sk-secret123456789。",
            fail_open=True,
            unresolved_sink=[],
        )

    assert create.calls == []


@pytest.mark.asyncio
async def test_digest_retries_whole_batch_after_ambiguous_item(
    test_config,
    monkeypatch,
):
    import dehydrator as dehydrator_module

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(dehydrator_module.asyncio, "sleep", _no_sleep)
    ambiguous = json.dumps(
        {"status": "ambiguous", "content": "", "unresolved": ["她", "那个项目"]},
        ensure_ascii=False,
    )
    source = "朝灯继续推进LMC-5。"
    dehydrator, create = _dehydrator(
        test_config,
        _digest_entry("她继续推进那个项目。"),
        ambiguous,
        _digest_entry("[[朝灯]]继续推进[[LMC-5]]。"),
    )

    items = await dehydrator.digest(source)

    assert items[0]["content"] == "[[朝灯]]继续推进[[LMC-5]]。"
    assert len(create.calls) == 3


@pytest.mark.asyncio
async def test_digest_resolves_raw_source_before_split(test_config):
    source_mapping = _mapping_payload(_mapping("r0", "朝灯", "subject"))
    clean_digest = _digest_entry("[[朝灯]]完成测试并确认检查通过。")
    dehydrator, create = _dehydrator(test_config, source_mapping, clean_digest)

    items = await dehydrator.digest("朝灯完成了测试。她确认检查通过。")

    assert items[0]["content"] == "[[朝灯]]完成测试并确认检查通过。"
    assert len(create.calls) == 2
    assert "朝灯确认检查通过" in create.calls[1]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_digest_rejects_ambiguous_source_before_digest_call(
    test_config,
):
    from dehydrator import SelfContainmentError
    ambiguous = json.dumps(
        {"status": "ambiguous", "mappings": [], "subject_anchors": [], "unresolved": ["她"]},
        ensure_ascii=False,
    )
    dehydrator, create = _dehydrator(test_config, ambiguous)

    with pytest.raises(SelfContainmentError):
        await dehydrator.digest("朝灯和小卷都参加了测试，她完成了测试。")

    assert len(create.calls) == 1


@pytest.mark.asyncio
async def test_digest_fail_open_preserves_ambiguous_source_and_disables_thinking(
    test_config,
):
    config = dict(test_config)
    config["dehydration"] = dict(
        test_config["dehydration"],
        base_url="https://api.deepseek.com/v1",
    )
    ambiguous = json.dumps(
        {"status": "ambiguous", "mappings": [], "unresolved": ["她"]},
        ensure_ascii=False,
    )
    clean_digest = _digest_entry("[[朝灯]]完成测试。")
    dehydrator, create = _dehydrator(config, ambiguous, clean_digest)
    sink = []

    items = await dehydrator.digest(
        "朝灯和小卷都参加了测试，她完成了测试。",
        fail_open=True,
        unresolved_sink=sink,
    )

    assert items[0]["content"] == "[[朝灯]]完成测试。"
    assert any("无法唯一确认" in reason for reason in sink)
    assert create.calls[1]["extra_body"] == {
        "thinking": {"type": "disabled"},
    }


def test_parse_digest_fail_open_keeps_entries_and_records_unresolved(test_config):
    dehydrator, _create = _dehydrator(test_config, "unused")
    payload = json.dumps(
        {
            "unresolved_references": ["她", "那个项目"],
            "entries": [{
                "name": "原样保留",
                "content": "[[朝灯]]记录了原文。",
                "domain": ["生活"],
                "tags": [],
                "importance": 5,
            }],
        },
        ensure_ascii=False,
    )
    sink = []

    items = dehydrator._parse_digest(
        payload,
        fail_open=True,
        unresolved_sink=sink,
    )

    assert items[0]["content"] == "[[朝灯]]记录了原文。"
    assert sink == ["日记来源仍有无法唯一确认的指代：她, 那个项目"]


@pytest.mark.asyncio
async def test_merge_ambiguity_does_not_update_existing_bucket(monkeypatch):
    import server

    class _FakeDehydrator:
        def __init__(self):
            self.checked = []

        async def ensure_self_contained(self, content, source_context=""):
            self.checked.append((content, source_context))
            if content == "她更新了那个项目":
                raise server.SelfContainmentError("ambiguous merge")
            return content

        async def merge(self, old_content, new_content):
            return "她更新了那个项目"

    class _Barrier:
        @asynccontextmanager
        async def shared_async(self):
            yield

    class _Manager:
        def __init__(self):
            self._maintenance_barrier = _Barrier()
            self.created = []
            self.updated = []

        async def create(self, **kwargs):
            self.created.append(kwargs)
            return "new-bucket"

        async def update(self, bucket_id, **kwargs):
            self.updated.append((bucket_id, kwargs))
            return True

    class _Embedding:
        def __init__(self):
            self.calls = []

        async def generate_and_store(self, bucket_id, content):
            self.calls.append((bucket_id, content))

    existing = {
        "id": "old-bucket",
        "content": "[[朝灯]]维护[[LMC-5]]。",
        "metadata": {
            "name": "旧桶",
            "tags": [],
            "domain": ["工程"],
            "importance": 5,
            "valence": 0.5,
            "arousal": 0.3,
        },
    }

    async def _candidates(**_kwargs):
        return [existing]

    manager = _Manager()
    embedding = _Embedding()
    fake_dehydrator = _FakeDehydrator()
    monkeypatch.setattr(server, "dehydrator", fake_dehydrator)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "embedding_engine", embedding)
    monkeypatch.setattr(server, "_find_merge_candidates", _candidates)
    monkeypatch.setattr(server, "_merge_candidate_passes_threshold", lambda _bucket: True)
    monkeypatch.setattr(server, "_is_merge_protected_bucket", lambda *_args: False)
    monkeypatch.setattr(server, "build_supersedes_audit", lambda *_args: [])

    content = "[[朝灯]]更新了[[LMC-5]]。"
    bucket_id, _, merged = await server._merge_or_create(
        content=content,
        tags=["LMC-5"],
        importance=5,
        domain=["工程"],
        valence=0.5,
        arousal=0.3,
        require_self_contained=True,
    )

    assert (bucket_id, merged) == ("new-bucket", False)
    assert manager.updated == []
    assert manager.created[0]["content"] == content
    assert embedding.calls == [("new-bucket", content)]
    assert fake_dehydrator.checked == [
        (
            "她更新了那个项目",
            f"{existing['content']}\n\n{content}",
        ),
    ]


@pytest.mark.asyncio
async def test_hold_keeps_legacy_path_without_self_containment_gate(monkeypatch):
    import server

    class _FakeDehydrator:
        async def analyze(self, content):
            assert content == "她去那里"
            return {
                "domain": ["日记"],
                "valence": 0.5,
                "arousal": 0.3,
                "tags": ["原样保留"],
                "suggested_name": "hold旧路径",
            }

        async def ensure_self_contained(self, *_args, **_kwargs):
            raise AssertionError("hold must not invoke the grow-only gate")

    async def _no_decay():
        return None

    captured = {}

    async def _write(**kwargs):
        captured.update(kwargs)
        return "hold-bucket", "hold旧路径", False

    async def _no_edges(**_kwargs):
        return []

    monkeypatch.setattr(server, "dehydrator", _FakeDehydrator())
    monkeypatch.setattr(server, "_ensure_decay_background", _no_decay)
    monkeypatch.setattr(server, "_maybe_start_backfill", lambda: None)
    monkeypatch.setattr(server, "_merge_or_create", _write)
    monkeypatch.setattr(server, "_auto_infer_edges", _no_edges)

    result = await server.hold("她去那里")

    assert result.startswith("新建→hold旧路径")
    assert captured["content"] == "她去那里"
    assert captured.get("require_self_contained", False) is False


@pytest.mark.asyncio
async def test_short_grow_rejects_before_analyze_or_write(monkeypatch):
    import server

    class _FakeDehydrator:
        async def ensure_self_contained(self, _content, source_context=""):
            raise server.SelfContainmentError("ambiguous")

        async def analyze(self, _content):
            raise AssertionError("analyze must not run")

    async def _no_decay():
        return None

    async def _no_write(**_kwargs):
        raise AssertionError("write must not run")

    monkeypatch.setattr(server, "dehydrator", _FakeDehydrator())
    monkeypatch.setattr(server, "_ensure_decay_background", _no_decay)
    monkeypatch.setattr(server, "_merge_or_create", _no_write)

    result = await server.grow("她去那里")

    assert "未写入" in result


@pytest.mark.asyncio
async def test_short_grow_writes_only_rewritten_content(monkeypatch):
    import server

    short_content = "朝灯说她去深圳" + "。" * (29 - len("朝灯说她去深圳"))
    assert len(short_content) == 29

    class _FakeDehydrator:
        def __init__(self):
            self.analyzed = []

        async def ensure_self_contained(self, content, source_context=""):
            assert content == short_content
            assert source_context == content
            return "[[朝灯]]说[[朝灯]]去[[深圳]]"

        async def analyze(self, content):
            self.analyzed.append(content)
            return {
                "domain": ["日记"],
                "valence": 0.6,
                "arousal": 0.3,
                "tags": ["深圳"],
                "suggested_name": "深圳行程",
            }

    async def _no_decay():
        return None

    captured = {}

    async def _write(**kwargs):
        captured.update(kwargs)
        return "bucket-1", "深圳行程", False

    fake = _FakeDehydrator()
    monkeypatch.setattr(server, "dehydrator", fake)
    monkeypatch.setattr(server, "_ensure_decay_background", _no_decay)
    monkeypatch.setattr(server, "_merge_or_create", _write)

    result = await server.grow(
        short_content,
        world="日常",
        chord_tag="M3 P5",
    )

    assert "新建" in result
    assert fake.analyzed == ["[[朝灯]]说[[朝灯]]去[[深圳]]"]
    assert captured["content"] == "[[朝灯]]说[[朝灯]]去[[深圳]]"
    assert captured["world"] == "日常"
    assert captured["chord_tag"] == "M3 P5"
    assert captured["require_self_contained"] is True


@pytest.mark.asyncio
async def test_thirty_character_grow_uses_digest_path(monkeypatch):
    import server

    content = "甲" * 30

    class _FakeDehydrator:
        def __init__(self):
            self.digested = []

        async def digest(self, raw, **kwargs):
            self.digested.append(raw)
            self.digest_kwargs = kwargs
            return [{
                "name": "边界",
                "content": "[[朝灯]]完成三十字边界测试",
                "domain": ["工程"],
                "valence": 0.5,
                "arousal": 0.3,
                "tags": ["边界"],
                "importance": 5,
            }]

        async def analyze(self, _content):
            raise AssertionError("30 characters must not use fast analyze")

    async def _no_decay():
        return None

    captured = []

    async def _write(**kwargs):
        captured.append(kwargs)
        return "bucket-30", "边界", False

    fake = _FakeDehydrator()
    monkeypatch.setattr(server, "dehydrator", fake)
    monkeypatch.setattr(server, "_ensure_decay_background", _no_decay)
    monkeypatch.setattr(server, "_merge_or_create", _write)

    result = await server.grow(content)

    assert result.startswith("1条|新1合0")
    assert fake.digest_kwargs["fail_open"] is True
    assert fake.digest_kwargs["unresolved_sink"] == []
    assert fake.digested == [content]
    assert captured[0]["require_self_contained"] is True
