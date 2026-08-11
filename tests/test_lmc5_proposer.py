from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping

import pytest

from lmc5_proposer import (
    CANDIDATE_TYPES,
    _PROVIDER_POLL_INTERVAL_SECONDS,
    CandidateDraft,
    ProposerChunk,
    ProposerContractError,
    RelationHint,
    StrictOmbreProposer,
    route_candidate_axes,
)
from utils import RELATION_TYPES


CHUNKS = (
    ProposerChunk("chunk-1", "朝灯明确喜欢雨声，也决定采用严格适配器。"),
    ProposerChunk("chunk-2", "第二段证据：部署前必须先验收。"),
)


def _candidate(**changes):
    value = {
        "type": "preference",
        "title": "雨声偏好",
        "content": "朝灯喜欢雨声。",
        "importance": 7,
        "thread_hint": "声音偏好",
        "relation_hints": [],
        "source_chunk_ids": ["chunk-1"],
        "evidence": "喜欢雨声",
        "risk": "normal",
    }
    value.update(changes)
    return value


def _document(*candidates):
    return {"schema_version": 1, "candidates": list(candidates)}


def _provider(content):
    async def provide(_prompt):
        return {"choices": [{"message": {"content": content}}]}

    return provide


def _proposer(content, **kwargs):
    return StrictOmbreProposer(
        _provider(content),
        model="test-model",
        provider_name="test-provider",
        **kwargs,
    )


class _AdversarialMapping(Mapping):
    def __iter__(self):
        return iter(("choices",))

    def __len__(self):
        return 1

    def __getitem__(self, _key):
        raise RuntimeError("secret-provider-payload")

    def get(self, _key, _default=None):
        time.sleep(0.2)
        raise RuntimeError("secret-provider-payload")


async def _error_code(proposer, **kwargs):
    with pytest.raises(ProposerContractError) as caught:
        await proposer.propose(CHUNKS, **kwargs)
    return caught.value.code


@pytest.mark.asyncio
async def test_exact_valid_empty_is_the_only_empty_success():
    batch = await _proposer(json.dumps(_document())).propose(CHUNKS)

    assert batch.candidates == ()
    assert batch.schema_version == 1
    assert batch.model == "test-model"
    assert batch.provider == "test-provider"
    assert len(batch.prompt_digest) == len(batch.output_digest) == 64


@pytest.mark.asyncio
async def test_provider_timeout_is_hard_bounded():
    async def slow(_prompt):
        await asyncio.sleep(1)

    proposer = StrictOmbreProposer(slow, timeout_seconds=0.01)
    assert await _error_code(proposer) == "provider.timeout"


def test_provider_poll_interval_avoids_busy_loop_without_blunting_deadline():
    assert 0.01 <= _PROVIDER_POLL_INTERVAL_SECONDS <= 0.05


@pytest.mark.asyncio
async def test_provider_cannot_suppress_the_hard_timeout():
    release = asyncio.Event()

    async def suppress_cancel(_prompt):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await release.wait()
        return {"choices": [{"message": {"content": json.dumps(_document())}}]}

    loop = asyncio.get_running_loop()
    started = loop.time()
    proposer = StrictOmbreProposer(suppress_cancel, timeout_seconds=0.01)
    assert await _error_code(proposer) == "provider.timeout"
    assert loop.time() - started < 0.1
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_provider_transport_failure_is_safe():
    async def broken(_prompt):
        raise RuntimeError("secret credential and response")

    proposer = StrictOmbreProposer(broken)
    with pytest.raises(ProposerContractError) as caught:
        await proposer.propose(CHUNKS)
    assert caught.value.code == "provider.transport"
    assert "secret" not in caught.value.detail


@pytest.mark.asyncio
async def test_provider_internal_cancellation_is_safe_transport_failure():
    async def self_cancel(_prompt):
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    proposer = StrictOmbreProposer(self_cancel)
    assert await _error_code(proposer) == "provider.transport"


@pytest.mark.asyncio
async def test_caller_cancellation_still_propagates():
    async def slow(_prompt):
        await asyncio.sleep(1)

    task = asyncio.create_task(StrictOmbreProposer(slow).propose(CHUNKS))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_synchronous_provider_block_cannot_defeat_timeout():
    def block(_prompt):
        time.sleep(0.2)
        return {"choices": [{"message": {"content": json.dumps(_document())}}]}

    loop = asyncio.get_running_loop()
    started = loop.time()
    proposer = StrictOmbreProposer(block, timeout_seconds=0.01)
    assert await _error_code(proposer) == "provider.timeout"
    assert loop.time() - started < 0.1


@pytest.mark.asyncio
async def test_async_cpu_block_before_first_await_cannot_defeat_timeout():
    async def block(_prompt):
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            pass
        await asyncio.sleep(0)
        return {"choices": [{"message": {"content": json.dumps(_document())}}]}

    loop = asyncio.get_running_loop()
    started = loop.time()
    proposer = StrictOmbreProposer(block, timeout_seconds=0.01)
    assert await _error_code(proposer) == "provider.timeout"
    assert loop.time() - started < 0.1


@pytest.mark.asyncio
async def test_adversarial_mapping_cannot_escape_or_extend_provider_contract():
    async def provider(_prompt):
        return _AdversarialMapping()

    loop = asyncio.get_running_loop()
    started = loop.time()
    code = await _error_code(
        StrictOmbreProposer(provider, timeout_seconds=0.01)
    )
    assert code in {"provider.no_choices", "provider.timeout"}
    assert loop.time() - started < 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("envelope", "code"),
    [
        ({}, "provider.no_choices"),
        ({"choices": []}, "provider.no_choices"),
        ({"choices": [{}]}, "provider.no_message"),
        ({"choices": [{"message": {"content": None}}]}, "provider.no_message"),
    ],
)
async def test_provider_envelope_is_strict(envelope, code):
    async def provider(_prompt):
        return envelope

    assert await _error_code(StrictOmbreProposer(provider)) == code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("envelope", "code"),
    [
        (
            {
                "error": {"message": "secret provider detail"},
                "choices": [{"message": {"content": json.dumps(_document())}}],
            },
            "provider.error",
        ),
        (
            {
                "refusal": "no",
                "choices": [{"message": {"content": json.dumps(_document())}}],
            },
            "provider.refusal",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(_document()),
                            "refusal": "no",
                        }
                    }
                ]
            },
            "provider.refusal",
        ),
        (
            {
                "choices": [
                    {
                        "error": {"message": "secret provider detail"},
                        "message": {"content": json.dumps(_document())},
                    }
                ]
            },
            "provider.error",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "error": {"message": "secret provider detail"},
                            "content": json.dumps(_document()),
                        }
                    }
                ]
            },
            "provider.error",
        ),
        (
            {
                "choices": [
                    {
                        "refusal": "no",
                        "message": {"content": json.dumps(_document())},
                    }
                ]
            },
            "provider.refusal",
        ),
        (
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": json.dumps(_document())},
                    }
                ]
            },
            "provider.incomplete",
        ),
    ],
)
async def test_provider_failure_metadata_cannot_masquerade_as_empty(envelope, code):
    async def provider(_prompt):
        return envelope

    with pytest.raises(ProposerContractError) as caught:
        await StrictOmbreProposer(provider).propose(CHUNKS)
    assert caught.value.code == code
    assert "secret provider detail" not in caught.value.detail


@pytest.mark.asyncio
async def test_incomplete_response_gets_one_compact_retry():
    prompts = []
    responses = iter(
        (
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "truncated"},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(_document(_candidate()))
                        },
                    }
                ]
            },
        )
    )

    async def provider(prompt):
        prompts.append(prompt)
        return next(responses)

    batch = await StrictOmbreProposer(provider).propose(CHUNKS)

    assert batch.candidates[0].title == "雨声偏好"
    assert len(prompts) == 2
    assert "RETRY ONE INCOMPLETE MODEL RESPONSE" not in prompts[0]
    assert "RETRY ONE INCOMPLETE MODEL RESPONSE" in prompts[1]
    assert "return at most 4 high-signal candidates" in prompts[1]
    assert "Return at most 4 high-signal candidates" in prompts[1]
    assert "Return at most 8 high-signal candidates" not in prompts[1]


@pytest.mark.asyncio
async def test_incomplete_response_is_retried_only_once_and_stays_retryable():
    prompts = []

    async def provider(prompt):
        prompts.append(prompt)
        return {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "truncated"},
                }
            ]
        }

    with pytest.raises(ProposerContractError) as caught:
        await StrictOmbreProposer(provider).propose(CHUNKS)

    assert caught.value.code == "provider.incomplete"
    assert len(prompts) == 2


@pytest.mark.asyncio
async def test_compact_retry_enforces_its_smaller_candidate_bound():
    prompts = []
    responses = iter(
        (
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "truncated"},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                _document(*(_candidate() for _ in range(5)))
                            )
                        },
                    }
                ]
            },
        )
    )

    async def provider(prompt):
        prompts.append(prompt)
        return next(responses)

    with pytest.raises(ProposerContractError) as caught:
        await StrictOmbreProposer(provider).propose(CHUNKS)

    assert caught.value.code == "schema_root"
    assert len(prompts) == 2


@pytest.mark.asyncio
async def test_candidate_count_is_bounded_in_prompt_and_validator():
    prompts = []
    too_many = json.dumps(_document(*(_candidate() for _ in range(9))))

    async def provider(prompt):
        prompts.append(prompt)
        return {"choices": [{"message": {"content": too_many}}]}

    with pytest.raises(ProposerContractError) as caught:
        await StrictOmbreProposer(provider).propose(CHUNKS)

    assert caught.value.code == "schema_root"
    assert len(prompts) == 2
    assert "Return at most 8 high-signal candidates" in prompts[0]


@pytest.mark.parametrize(
    "timeout_seconds",
    [True, 0, -1, float("nan"), float("inf"), 10**1000],
)
def test_timeout_configuration_is_bounded_without_raw_numeric_errors(
    timeout_seconds,
):
    with pytest.raises(ValueError, match="finite and positive"):
        StrictOmbreProposer(
            lambda _prompt: None,
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", ""),
        ("model", " model"),
        ("model", "model\x00"),
        ("model", "\ud800"),
        ("model", "m" * 257),
        ("provider_name", ""),
        ("provider_name", "provider "),
        ("provider_name", "provider\x7f"),
        ("provider_name", "\udfff"),
        ("provider_name", "p" * 257),
    ],
)
def test_provider_metadata_is_bounded_exact_utf8(field, value):
    kwargs = {"model": "model", "provider_name": "provider"}
    kwargs[field] = value
    with pytest.raises(ValueError, match="bounded UTF-8"):
        StrictOmbreProposer(lambda _prompt: None, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", " \n\t"])
async def test_blank_response_is_failure(content):
    assert await _error_code(_proposer(content)) == "provider.empty_response"


@pytest.mark.asyncio
async def test_oversized_response_is_rejected_before_parse():
    proposer = _proposer(" " * 33, max_response_bytes=32)
    assert await _error_code(proposer) == "provider.response_too_large"


@pytest.mark.asyncio
async def test_malformed_json_is_rejected():
    assert await _error_code(_proposer('{"schema_version":')) == "parse_json"


@pytest.mark.asyncio
async def test_json_integer_digit_limit_is_a_contract_error():
    content = (
        '{"schema_version":1,"candidates":[{'
        '"type":"preference","title":"雨声偏好","content":"朝灯喜欢雨声。",'
        '"importance":7,"thread_hint":"声音偏好","relation_hints":[{'
        '"relation_type":"explains","target_id":"bucket-1","strength":'
        + ("9" * 5000)
        + ',"reason":"same decision"}],"source_chunk_ids":["chunk-1"],'
        '"evidence":"喜欢雨声","risk":"normal"}]}'
    )
    proposer = _proposer(content)
    assert (
        await _error_code(
            proposer, allowed_relation_targets=frozenset({"bucket-1"})
        )
        == "parse_json"
    )


@pytest.mark.asyncio
async def test_raw_lone_surrogate_is_a_contract_error():
    assert (
        await _error_code(_proposer('{"schema_version":1,"candidates":[]}\ud800'))
        == "provider.invalid_text"
    )


@pytest.mark.asyncio
async def test_duplicate_key_is_rejected_at_any_depth():
    content = (
        '{"schema_version":1,"candidates":[{'
        '"type":"fact","type":"event","title":"t","content":"c",'
        '"importance":1,"thread_hint":"","relation_hints":[],'
        '"source_chunk_ids":["chunk-1"],"evidence":"喜欢雨声","risk":"normal"}]}'
    )
    assert await _error_code(_proposer(content)) == "parse_duplicate_key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "number", ["NaN", "Infinity", "-Infinity", "1e999"]
)
async def test_nonfinite_json_number_is_rejected(number):
    content = json.dumps(_document(_candidate())).replace(
        '"importance": 7', f'"importance": {number}'
    )
    assert await _error_code(_proposer(content)) == "parse_nonfinite"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "root",
    [
        [],
        {"schema_version": 1},
        {"schema_version": True, "candidates": []},
        {"schema_version": 1, "candidates": [], "extra": 1},
        {"schema_version": 1, "candidates": {}},
    ],
)
async def test_root_schema_is_exact_without_coercion(root):
    assert await _error_code(_proposer(json.dumps(root))) == "schema_root"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(extra=True),
        _candidate(importance=True),
        _candidate(importance="7"),
        _candidate(importance=0),
        _candidate(type="unknown"),
        _candidate(relation_hints={}),
        _candidate(source_chunk_ids="chunk-1"),
        _candidate(axis="Z"),
        _candidate(fact_status="accepted"),
        _candidate(E=0.9),
        _candidate(protected=True),
        _candidate(archive=True),
        _candidate(delete=True),
    ],
)
async def test_candidate_schema_is_exact_and_has_no_model_control_fields(candidate):
    assert (
        await _error_code(_proposer(json.dumps(_document(candidate))))
        == "schema_candidate"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", ""),
        ("title", " "),
        ("title", " 雨声"),
        ("title", "雨声 "),
        ("title", "雨\n声"),
        ("title", "雨\x7f声"),
        ("title", "\ud800"),
        ("content", ""),
        ("content", " "),
        ("content", " 朝灯喜欢雨声。"),
        ("content", "朝灯喜欢雨声。\t"),
        ("thread_hint", " "),
        ("thread_hint", "声音偏好 "),
        ("thread_hint", "声音\x00偏好"),
    ],
)
async def test_candidate_text_is_exact_nonblank_bounded_utf8_without_controls(
    field, value
):
    proposer = _proposer(json.dumps(_document(_candidate(**{field: value}))))
    assert await _error_code(proposer) == "schema_candidate"


@pytest.mark.asyncio
async def test_empty_thread_hint_is_a_valid_optional_hint():
    batch = await _proposer(
        json.dumps(_document(_candidate(thread_hint="")))
    ).propose(CHUNKS)

    assert batch.candidates[0].thread_hint == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "t" * 513),
        ("content", "c" * (16 * 1024 + 1)),
        ("thread_hint", "h" * 513),
    ],
)
async def test_candidate_text_fields_are_bounded(field, value):
    proposer = _proposer(json.dumps(_document(_candidate(**{field: value}))))
    assert await _error_code(proposer) == "schema_candidate"


@pytest.mark.asyncio
@pytest.mark.parametrize("risk", ["low", "NORMAL", " normal", "normal "])
async def test_risk_is_exact_normal_or_review(risk):
    proposer = _proposer(json.dumps(_document(_candidate(risk=risk))))
    assert await _error_code(proposer) == "schema_candidate"


@pytest.mark.asyncio
@pytest.mark.parametrize("risk", ["normal", "review"])
async def test_valid_risk_values_are_preserved(risk):
    batch = await _proposer(
        json.dumps(_document(_candidate(risk=risk)))
    ).propose(CHUNKS)
    assert batch.candidates[0].risk == risk


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_ids",
    [
        [],
        ["chunk-1", "chunk-1"],
        ["missing"],
        [1],
        [" chunk-1"],
        ["chunk-1 "],
        ["chunk-\x001"],
        ["\ud800"],
    ],
)
async def test_source_ids_are_unique_nonempty_input_subset(source_ids):
    proposer = _proposer(
        json.dumps(_document(_candidate(source_chunk_ids=source_ids)))
    )
    assert await _error_code(proposer) == "provenance_source"


@pytest.mark.asyncio
@pytest.mark.parametrize("source_ids", ([], ["model-typo"]))
async def test_single_chunk_source_is_bound_locally(source_ids):
    proposer = _proposer(
        json.dumps(_document(_candidate(source_chunk_ids=source_ids)))
    )

    batch = await proposer.propose((CHUNKS[0],))

    assert batch.candidates[0].source_chunk_ids == ("chunk-1",)


@pytest.mark.asyncio
async def test_single_chunk_source_hint_stays_structurally_bounded():
    proposer = _proposer(
        json.dumps(
            _document(
                _candidate(source_chunk_ids=["model-typo", "second"])
            )
        )
    )

    with pytest.raises(ProposerContractError) as caught:
        await proposer.propose((CHUNKS[0],))

    assert caught.value.code == "provenance_source"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evidence",
    [
        "",
        " ",
        "喜欢雨声 ",
        "喜欢\n雨声",
        "喜欢\x7f雨声",
        "\ud800",
        "e" * (4 * 1024 + 1),
        "模型捏造的证据",
    ],
)
async def test_evidence_must_be_literal_in_a_cited_chunk(evidence):
    proposer = _proposer(json.dumps(_document(_candidate(evidence=evidence))))
    assert await _error_code(proposer) == "provenance_evidence"


@pytest.mark.asyncio
async def test_evidence_must_match_one_of_the_cited_chunks_only():
    candidate = _candidate(
        source_chunk_ids=["chunk-2"], evidence="喜欢雨声"
    )
    assert (
        await _error_code(_proposer(json.dumps(_document(candidate))))
        == "provenance_evidence"
    )


def _relation(**changes):
    value = {
        "relation_type": "explains",
        "target_id": "bucket-1",
        "strength": 0.8,
        "reason": "same decision",
    }
    value.update(changes)
    return value


@pytest.mark.asyncio
async def test_relation_requires_prebound_target_and_empty_allowlist_means_none():
    candidate = _candidate(relation_hints=[_relation()])
    proposer = _proposer(json.dumps(_document(candidate)))
    assert await _error_code(proposer) == "relation_target"


@pytest.mark.asyncio
async def test_relation_target_must_be_in_nonempty_allowlist():
    candidate = _candidate(relation_hints=[_relation(target_id="bucket-2")])
    proposer = _proposer(json.dumps(_document(candidate)))
    assert (
        await _error_code(
            proposer, allowed_relation_targets=frozenset({"bucket-1"})
        )
        == "relation_target"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relation",
    [
        _relation(extra=True),
        _relation(relation_type="invented"),
        _relation(strength=True),
        _relation(strength="0.8"),
        _relation(strength=-0.1),
        _relation(strength=1.1),
    ],
)
async def test_relation_schema_is_exact_and_numeric_values_are_strict(relation):
    candidate = _candidate(relation_hints=[relation])
    proposer = _proposer(json.dumps(_document(candidate)))
    assert (
        await _error_code(
            proposer, allowed_relation_targets=frozenset({"bucket-1"})
        )
        == "schema_relation"
    )


@pytest.mark.asyncio
async def test_huge_integer_relation_strength_is_a_contract_error():
    candidate = _candidate(relation_hints=[_relation(strength=10**1000)])
    proposer = _proposer(json.dumps(_document(candidate)))
    assert (
        await _error_code(
            proposer, allowed_relation_targets=frozenset({"bucket-1"})
        )
        == "schema_relation"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relation",
    [
        _relation(target_id=""),
        _relation(target_id=" bucket-1"),
        _relation(target_id="bucket-1 "),
        _relation(target_id="bucket-\x001"),
        _relation(target_id="\ud800"),
        _relation(target_id="b" * 257),
        _relation(reason=""),
        _relation(reason=" "),
        _relation(reason=" same decision"),
        _relation(reason="same decision "),
        _relation(reason="same\ndecision"),
        _relation(reason="same\x7fdecision"),
        _relation(reason="\ud800"),
        _relation(reason="r" * (2 * 1024 + 1)),
    ],
)
async def test_relation_ids_and_reason_are_exact_bounded_utf8_without_controls(
    relation,
):
    candidate = _candidate(relation_hints=[relation])
    proposer = _proposer(json.dumps(_document(candidate)))
    assert (
        await _error_code(
            proposer, allowed_relation_targets=frozenset({"bucket-1"})
        )
        == "schema_relation"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("relation_type", sorted(RELATION_TYPES))
async def test_all_ombre_relation_types_are_reused(relation_type):
    candidate = _candidate(
        relation_hints=[_relation(relation_type=relation_type)]
    )
    batch = await _proposer(json.dumps(_document(candidate))).propose(
        CHUNKS, frozenset({"bucket-1"})
    )
    assert batch.candidates[0].relation_hints[0].relation_type == relation_type


@pytest.mark.asyncio
async def test_one_invalid_candidate_fails_whole_batch():
    valid = _candidate(title="valid")
    invalid = _candidate(title="invalid", evidence="not present")
    proposer = _proposer(json.dumps(_document(valid, invalid)))
    assert await _error_code(proposer) == "provenance_evidence"


@pytest.mark.asyncio
async def test_evidence_error_gets_one_full_batch_repair():
    prompts = []
    responses = iter(
        (
            json.dumps(
                _document(_candidate(evidence="模型改写的雨声证据"))
            ),
            json.dumps(_document(_candidate(evidence="喜欢雨声"))),
        )
    )

    async def provider(prompt):
        prompts.append(prompt)
        return {
            "choices": [{"message": {"content": next(responses)}}]
        }

    batch = await StrictOmbreProposer(provider).propose(CHUNKS)

    assert batch.candidates[0].evidence == "喜欢雨声"
    assert len(prompts) == 2
    assert "REPAIR ONE MODEL CONTRACT ERROR" not in prompts[0]
    assert "REPAIR ONE MODEL CONTRACT ERROR (provenance_evidence)" in prompts[1]
    assert "Regenerate the whole JSON object" in prompts[1]
    assert batch.prompt_digest == hashlib.sha256(
        prompts[1].encode("utf-8")
    ).hexdigest()


@pytest.mark.asyncio
async def test_evidence_repair_is_attempted_only_once():
    prompts = []
    content = json.dumps(
        _document(_candidate(evidence="模型改写的雨声证据"))
    )

    async def provider(prompt):
        prompts.append(prompt)
        return {"choices": [{"message": {"content": content}}]}

    with pytest.raises(ProposerContractError) as caught:
        await StrictOmbreProposer(provider).propose(CHUNKS)

    assert caught.value.code == "provenance_evidence"
    assert len(prompts) == 2


@pytest.mark.asyncio
async def test_schema_candidate_error_gets_one_full_batch_repair():
    prompts = []
    responses = iter(
        (
            json.dumps(_document(_candidate(importance=11))),
            json.dumps(_document(_candidate())),
        )
    )

    async def provider(prompt):
        prompts.append(prompt)
        return {
            "choices": [{"message": {"content": next(responses)}}]
        }

    batch = await StrictOmbreProposer(provider).propose(CHUNKS)

    assert batch.candidates[0].importance == 7
    assert len(prompts) == 2
    assert "schema_candidate" in prompts[1]


@pytest.mark.asyncio
async def test_provenance_source_error_gets_one_full_batch_repair():
    prompts = []
    responses = iter(
        (
            json.dumps(
                _document(
                    _candidate(source_chunk_ids=["altered-chunk-id"])
                )
            ),
            json.dumps(_document(_candidate())),
        )
    )

    async def provider(prompt):
        prompts.append(prompt)
        return {
            "choices": [{"message": {"content": next(responses)}}]
        }

    batch = await StrictOmbreProposer(provider).propose(CHUNKS)

    assert batch.candidates[0].source_chunk_ids == ("chunk-1",)
    assert len(prompts) == 2
    assert "provenance_source" in prompts[1]


@pytest.mark.asyncio
async def test_nonrepairable_parse_error_does_not_retry():
    prompts = []

    async def provider(prompt):
        prompts.append(prompt)
        return {
            "choices": [{"message": {"content": '{"schema_version":'}}]
        }

    with pytest.raises(ProposerContractError) as caught:
        await StrictOmbreProposer(provider).propose(CHUNKS)

    assert caught.value.code == "parse_json"
    assert len(prompts) == 1


def _draft(candidate_type, with_relation=False):
    relations = (
        (RelationHint("kin", "bucket-1", 0.5, "related"),)
        if with_relation
        else ()
    )
    return CandidateDraft(
        type=candidate_type,
        title="t",
        content="c",
        importance=5,
        thread_hint="",
        relation_hints=relations,
        source_chunk_ids=("chunk-1",),
        evidence="喜欢雨声",
        risk="normal",
    )


@pytest.mark.parametrize(
    ("candidate_type", "expected"),
    [
        ("event", {"X", "M"}),
        ("fact", {"X", "Z", "M"}),
        ("preference", {"X", "Z", "E", "M"}),
        ("engineering_decision", {"X", "Z", "M"}),
        ("relationship_moment", {"X", "E", "M"}),
        ("risk_boundary", {"X", "Z", "E", "M"}),
    ],
)
def test_local_axis_route_is_deterministic(candidate_type, expected):
    assert route_candidate_axes(_draft(candidate_type)) == expected


def test_bound_relation_adds_y_to_local_axis_route():
    for candidate_type in CANDIDATE_TYPES:
        axes = route_candidate_axes(_draft(candidate_type, with_relation=True))
        assert {"X", "M", "Y"} <= axes


@pytest.mark.asyncio
async def test_output_digest_is_stable_across_json_format_and_key_order():
    candidate = _candidate(
        relation_hints=[_relation(strength=1)],
        source_chunk_ids=["chunk-1", "chunk-2"],
    )
    first = json.dumps(_document(candidate), ensure_ascii=False)
    reordered = {
        "candidates": [
            {
                key: candidate[key]
                for key in reversed(list(candidate))
            }
        ],
        "schema_version": 1,
    }
    second = json.dumps(reordered, ensure_ascii=False, indent=2)
    kwargs = {"allowed_relation_targets": frozenset({"bucket-1"})}
    batch_a = await _proposer(first).propose(CHUNKS, **kwargs)
    batch_b = await _proposer(second).propose(CHUNKS, **kwargs)
    assert batch_a.output_digest == batch_b.output_digest
    assert batch_a.prompt_digest == batch_b.prompt_digest


@pytest.mark.asyncio
async def test_negative_zero_strength_has_canonical_zero_digest():
    negative = _candidate(relation_hints=[_relation(strength=-0.0)])
    positive = _candidate(relation_hints=[_relation(strength=0.0)])
    kwargs = {"allowed_relation_targets": frozenset({"bucket-1"})}
    batch_negative = await _proposer(
        json.dumps(_document(negative))
    ).propose(CHUNKS, **kwargs)
    batch_positive = await _proposer(
        json.dumps(_document(positive))
    ).propose(CHUNKS, **kwargs)
    assert batch_negative.candidates[0].relation_hints[0].strength == 0.0
    assert batch_negative.output_digest == batch_positive.output_digest


@pytest.mark.asyncio
async def test_prompt_is_deterministic_and_contains_only_local_schema_controls():
    prompts = []

    async def provider(prompt):
        prompts.append(prompt)
        return {"choices": [{"message": {"content": json.dumps(_document())}}]}

    proposer = StrictOmbreProposer(provider)
    first = await proposer.propose(CHUNKS, frozenset({"bucket-2", "bucket-1"}))
    second = await proposer.propose(CHUNKS, frozenset({"bucket-1", "bucket-2"}))
    assert prompts[0] == prompts[1]
    assert first.prompt_digest == second.prompt_digest
    assert "Never emit axis" in prompts[0]


@pytest.mark.asyncio
async def test_prompt_limit_fails_before_provider_is_called():
    called = False

    async def provider(_prompt):
        nonlocal called
        called = True

    proposer = StrictOmbreProposer(provider, max_prompt_bytes=32)
    assert await _error_code(proposer) == "prompt_too_large"
    assert called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        (),
        (ProposerChunk("", "text"),),
        (ProposerChunk("same", "one"), ProposerChunk("same", "two")),
        (ProposerChunk("id", ""),),
        (ProposerChunk("id", " \n"),),
        ("not-a-chunk",),
    ],
)
async def test_invalid_chunk_input_fails_before_provider(chunks):
    proposer = _proposer(json.dumps(_document()))
    with pytest.raises(ProposerContractError) as caught:
        await proposer.propose(chunks)
    assert caught.value.code == "input_invalid"


@pytest.mark.asyncio
async def test_chunk_count_limit_is_input_invalid():
    proposer = _proposer(json.dumps(_document()), max_chunks=1)
    assert await _error_code(proposer) == "input_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "targets", ["bucket-1", frozenset({""}), frozenset({1})]
)
async def test_relation_allowlist_input_is_strict(targets):
    proposer = _proposer(json.dumps(_document()))
    with pytest.raises(ProposerContractError) as caught:
        await proposer.propose(CHUNKS, targets)
    assert caught.value.code == "input_invalid"
