"""Strict, fail-closed model-output adapter for LMC-5 candidate proposals.

The proposer deliberately stops at typed candidate drafts.  It does not choose
fact status, write any axis, or mutate Ombre state.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import queue
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

from utils import RELATION_TYPES


SCHEMA_VERSION = 1
CANDIDATE_TYPES = frozenset(
    {
        "event",
        "fact",
        "preference",
        "engineering_decision",
        "relationship_moment",
        "risk_boundary",
    }
)

_ROOT_FIELDS = frozenset({"schema_version", "candidates"})
_CANDIDATE_FIELDS = frozenset(
    {
        "type",
        "title",
        "content",
        "importance",
        "thread_hint",
        "relation_hints",
        "source_chunk_ids",
        "evidence",
        "risk",
    }
)
_RELATION_FIELDS = frozenset(
    {"relation_type", "target_id", "strength", "reason"}
)
_RISKS = frozenset({"normal", "review"})
_MAX_ID_BYTES = 256
_MAX_CHUNK_TEXT_BYTES = 256 * 1024
_MAX_TITLE_BYTES = 512
_MAX_CONTENT_BYTES = 16 * 1024
_MAX_THREAD_HINT_BYTES = 512
_MAX_EVIDENCE_BYTES = 4 * 1024
_MAX_RELATION_REASON_BYTES = 2 * 1024
_MAX_CANDIDATES = 8
_COMPACT_RETRY_MAX_CANDIDATES = 4
_REPAIRABLE_MODEL_CODES = frozenset(
    {
        "provenance_evidence",
        "provenance_source",
        "relation_target",
        "schema_candidate",
        "schema_relation",
        "schema_root",
    }
)
_INCOMPLETE_RETRY_INSTRUCTION = (
    "RETRY ONE INCOMPLETE MODEL RESPONSE: Regenerate the whole JSON object "
    "from the original INPUT and return at most "
    f"{_COMPACT_RETRY_MAX_CANDIDATES} high-signal candidates. Prefer fewer "
    "candidates over a truncated response. Keep title, content, thread_hint, "
    "evidence, and relation reasons concise while preserving literal evidence "
    "and every original schema/provenance rule. Return JSON only."
)
_CONTRACT_REPAIR_INSTRUCTION = (
    "REPAIR ONE MODEL CONTRACT ERROR ({code}): Regenerate the whole JSON "
    "object; do not patch or discuss the prior output. Match every root, "
    "candidate, and relation key and value type in output_schema exactly. "
    "Copy every source_chunk_id verbatim from INPUT.chunks[].id and use only "
    "those ids. For every candidate, copy evidence verbatim as one exact "
    "contiguous substring from the text of one cited source_chunk_id, with "
    "identical punctuation and whitespace. Never paraphrase, summarize, "
    "translate, normalize, splice, or invent evidence. All other schema and "
    "contract rules remain unchanged."
)
_TYPE_AXES = {
    "event": frozenset({"X", "M"}),
    "fact": frozenset({"X", "Z", "M"}),
    "preference": frozenset({"X", "Z", "E", "M"}),
    "engineering_decision": frozenset({"X", "Z", "M"}),
    "relationship_moment": frozenset({"X", "E", "M"}),
    "risk_boundary": frozenset({"X", "Z", "E", "M"}),
}


class ProposerContractError(Exception):
    """A bounded machine-readable failure without provider/model payloads."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        # Keep diagnostics suitable for logs: single-line, short, and never raw
        # provider output.  All adapter call sites use fixed details.
        self.detail = " ".join(str(detail).split())[:240]
        super().__init__(f"{self.code}: {self.detail}")


@dataclass(frozen=True, slots=True)
class ProposerChunk:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class RelationHint:
    relation_type: str
    target_id: str
    strength: float
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    type: str
    title: str
    content: str
    importance: int
    thread_hint: str
    relation_hints: tuple[RelationHint, ...]
    source_chunk_ids: tuple[str, ...]
    evidence: str
    risk: str


@dataclass(frozen=True, slots=True)
class ProposerBatch:
    schema_version: int
    candidates: tuple[CandidateDraft, ...]
    prompt_digest: str
    output_digest: str
    model: str
    provider: str


def route_candidate_axes(candidate: CandidateDraft) -> frozenset[str]:
    """Return the deterministic local route; the model cannot set axes.

    Every draft participates in timeline/provenance (X) and lifecycle (M).
    Fact-like drafts enter Z; preference and risk-boundary drafts enter both Z
    and E; relationship moments enter E.  Only a validated relation adds Y.
    """

    axes = _TYPE_AXES.get(candidate.type)
    if axes is None:
        raise ProposerContractError(
            "schema_candidate", "candidate type has no local axis route"
        )
    if candidate.relation_hints:
        return axes | {"Y"}
    return axes


class _DuplicateKey(ValueError):
    pass


class _NonFinite(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result


def _reject_nonfinite_constant(_: str) -> Any:
    raise _NonFinite


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    return False


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        # Integers are finite without converting them to float.  Converting an
        # adversarially large integer can itself raise OverflowError.
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_bounded_text(value: Any, max_bytes: int) -> bool:
    """Accept exact, nonblank UTF-8 text without silently normalizing it."""

    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)
    ):
        return False
    try:
        return len(value.encode("utf-8", errors="strict")) <= max_bytes
    except UnicodeError:
        return False


def _is_utf8_bounded(value: Any, max_bytes: int) -> bool:
    if type(value) is not str:
        return False
    try:
        return len(value.encode("utf-8", errors="strict")) <= max_bytes
    except UnicodeError:
        return False


def _is_optional_bounded_text(value: Any, max_bytes: int) -> bool:
    return value == "" or _is_bounded_text(value, max_bytes)


def _expect_exact_mapping(
    value: Any, fields: frozenset[str], code: str, detail: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != fields:
        raise ProposerContractError(code, detail)
    return value


def _candidate_as_json(candidate: CandidateDraft) -> dict[str, Any]:
    return {
        "type": candidate.type,
        "title": candidate.title,
        "content": candidate.content,
        "importance": candidate.importance,
        "thread_hint": candidate.thread_hint,
        "relation_hints": [
            {
                "relation_type": relation.relation_type,
                "target_id": relation.target_id,
                "strength": relation.strength,
                "reason": relation.reason,
            }
            for relation in candidate.relation_hints
        ],
        "source_chunk_ids": list(candidate.source_chunk_ids),
        "evidence": candidate.evidence,
        "risk": candidate.risk,
    }


class StrictOmbreProposer:
    """Call an injected provider and validate each whole response atomically."""

    def __init__(
        self,
        provider: Callable[[str], Any],
        *,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 64 * 1024,
        max_prompt_bytes: int = 256 * 1024,
        max_chunks: int = 256,
        model: str = "unspecified",
        provider_name: str = "injected",
    ) -> None:
        if not callable(provider):
            raise ValueError("provider must be callable")
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        try:
            safe_timeout = float(timeout_seconds)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "timeout_seconds must be finite and positive"
            ) from exc
        if not math.isfinite(safe_timeout) or safe_timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        for name, value in (
            ("max_response_bytes", max_response_bytes),
            ("max_prompt_bytes", max_prompt_bytes),
            ("max_chunks", max_chunks),
        ):
            if not _is_plain_int(value) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not _is_bounded_text(model, _MAX_ID_BYTES) or not _is_bounded_text(
            provider_name, _MAX_ID_BYTES
        ):
            raise ValueError(
                "model and provider_name must be bounded UTF-8 identifiers"
            )

        self._provider = provider
        self.timeout_seconds = safe_timeout
        self.max_response_bytes = max_response_bytes
        self.max_prompt_bytes = max_prompt_bytes
        self.max_chunks = max_chunks
        self.model = model
        self.provider_name = provider_name

    async def propose(
        self,
        chunks: Sequence[ProposerChunk],
        allowed_relation_targets: frozenset[str] = frozenset(),
    ) -> ProposerBatch:
        chunk_tuple, targets = self._validate_input(
            chunks, allowed_relation_targets
        )
        prompt = self._build_prompt(chunk_tuple, targets)
        self._validate_prompt_size(prompt)

        effective_prompt = prompt
        try:
            candidates = await self._call_and_validate(
                prompt,
                chunk_tuple,
                targets,
                max_candidates=_MAX_CANDIDATES,
            )
        except ProposerContractError as exc:
            if exc.code == "provider.incomplete":
                effective_prompt = self._build_incomplete_retry_prompt(prompt)
                max_candidates = _COMPACT_RETRY_MAX_CANDIDATES
            elif exc.code in _REPAIRABLE_MODEL_CODES:
                effective_prompt = self._build_contract_repair_prompt(
                    prompt, exc.code
                )
                max_candidates = _MAX_CANDIDATES
            else:
                raise
            self._validate_prompt_size(effective_prompt)
            candidates = await self._call_and_validate(
                effective_prompt,
                chunk_tuple,
                targets,
                max_candidates=max_candidates,
            )

        normalized = {
            "schema_version": SCHEMA_VERSION,
            "candidates": [
                _candidate_as_json(candidate) for candidate in candidates
            ],
        }
        return ProposerBatch(
            schema_version=SCHEMA_VERSION,
            candidates=candidates,
            prompt_digest=_digest_text(effective_prompt),
            output_digest=_digest_text(_canonical_json(normalized)),
            model=self.model,
            provider=self.provider_name,
        )

    def _validate_prompt_size(self, prompt: str) -> None:
        try:
            prompt_size = len(prompt.encode("utf-8", errors="strict"))
        except UnicodeError as exc:
            raise ProposerContractError(
                "input_invalid", "canonical proposer prompt is not UTF-8"
            ) from exc
        if prompt_size > self.max_prompt_bytes:
            raise ProposerContractError(
                "prompt_too_large", "canonical proposer prompt exceeds limit"
            )

    async def _call_and_validate(
        self,
        prompt: str,
        chunks: tuple[ProposerChunk, ...],
        targets: frozenset[str],
        *,
        max_candidates: int,
    ) -> tuple[CandidateDraft, ...]:
        envelope = await self._call_provider(prompt)
        response = self._extract_message(envelope)
        try:
            encoded = response.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ProposerContractError(
                "provider.invalid_text", "provider message is not valid UTF-8"
            ) from exc
        if len(encoded) > self.max_response_bytes:
            raise ProposerContractError(
                "provider.response_too_large", "provider message exceeds limit"
            )
        if not response.strip():
            raise ProposerContractError(
                "provider.empty_response", "provider message was blank"
            )
        parsed = self._parse_json(response)
        return self._validate_root(
            parsed,
            chunks,
            targets,
            max_candidates=max_candidates,
        )

    async def _call_provider(self, prompt: str) -> Any:
        caller_loop = asyncio.get_running_loop()
        deadline = caller_loop.time() + self.timeout_seconds
        completion: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        abandoned = threading.Event()

        def publish(ok: bool, value: Any) -> None:
            if abandoned.is_set():
                return
            try:
                completion.put_nowait((ok, value))
            except queue.Full:
                pass

        def worker() -> None:
            try:
                result = self._provider(prompt)
                if inspect.isawaitable(result):
                    result = asyncio.run(self._await_provider(result))
            except BaseException:
                # CancelledError raised by the provider is an internal transport
                # failure.  Never attach its payload to caller-visible state.
                publish(False, None)
            else:
                publish(True, result)

        thread = threading.Thread(
            target=worker,
            name="lmc5-proposer-provider",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            raise ProposerContractError(
                "provider.transport", "provider call failed"
            ) from exc

        try:
            while True:
                try:
                    ok, result = completion.get_nowait()
                    break
                except queue.Empty:
                    remaining = deadline - caller_loop.time()
                    if remaining <= 0:
                        abandoned.set()
                        raise ProposerContractError(
                            "provider.timeout",
                            "provider exceeded hard deadline",
                        )
                    await asyncio.sleep(min(0.002, remaining))
        except asyncio.CancelledError:
            abandoned.set()
            raise
        if not ok:
            raise ProposerContractError(
                "provider.transport", "provider call failed"
            )
        return result

    @staticmethod
    async def _await_provider(pending: Awaitable[Any]) -> Any:
        return await pending

    def _validate_input(
        self,
        chunks: Sequence[ProposerChunk],
        allowed_relation_targets: frozenset[str],
    ) -> tuple[tuple[ProposerChunk, ...], frozenset[str]]:
        if (
            isinstance(chunks, (str, bytes))
            or not isinstance(chunks, Sequence)
            or not chunks
            or len(chunks) > self.max_chunks
        ):
            raise ProposerContractError(
                "input_invalid", "chunks must be a non-empty bounded sequence"
            )
        seen: set[str] = set()
        validated: list[ProposerChunk] = []
        for chunk in chunks:
            if (
                not isinstance(chunk, ProposerChunk)
                or not _is_bounded_text(chunk.id, _MAX_ID_BYTES)
                or chunk.id in seen
                or not _is_utf8_bounded(chunk.text, _MAX_CHUNK_TEXT_BYTES)
                or not chunk.text.strip()
            ):
                raise ProposerContractError(
                    "input_invalid", "chunk ids/text must be valid and ids unique"
                )
            seen.add(chunk.id)
            validated.append(chunk)

        if (
            isinstance(allowed_relation_targets, (str, bytes))
            or not isinstance(allowed_relation_targets, Set)
            or any(
                not _is_bounded_text(target, _MAX_ID_BYTES)
                for target in allowed_relation_targets
            )
        ):
            raise ProposerContractError(
                "input_invalid", "relation target allowlist is invalid"
            )
        return tuple(validated), frozenset(allowed_relation_targets)

    def _build_prompt(
        self,
        chunks: tuple[ProposerChunk, ...],
        targets: frozenset[str],
    ) -> str:
        schema = {
            "schema_version": 1,
            "candidates": [
                {
                    "type": sorted(CANDIDATE_TYPES),
                    "title": "string",
                    "content": "string",
                    "importance": "integer 1..10",
                    "thread_hint": "string",
                    "relation_hints": [
                        {
                            "relation_type": sorted(RELATION_TYPES),
                            "target_id": "one allowed_relation_target",
                            "strength": "finite number 0..1",
                            "reason": "string",
                        }
                    ],
                    "source_chunk_ids": ["unique cited chunk ids"],
                    "evidence": "non-empty literal substring of a cited chunk",
                    "risk": sorted(_RISKS),
                }
            ],
        }
        payload = {
            "allowed_relation_targets": sorted(targets),
            "chunks": [
                {"id": chunk.id, "text": chunk.text} for chunk in chunks
            ],
            "output_schema": schema,
        }
        rules = (
            "Return exactly one JSON object and no markdown. Root keys must be "
            "exactly schema_version,candidates. Candidate and relation keys "
            f"must exactly match output_schema. Return at most {_MAX_CANDIDATES} "
            "high-signal candidates and prefer fewer concise candidates over "
            "a long or truncated response. Never emit axis, fact status, E, "
            "protected, archive, or delete controls. If no candidate is "
            "supported, return the exact schema with candidates:[]. Every "
            "candidate needs non-empty unique source_chunk_ids and non-empty "
            "source_chunk_ids must be copied exactly from INPUT.chunks[].id "
            "without alteration. evidence must be copied verbatim as one exact "
            "contiguous substring from the text of one cited chunk, with "
            "identical punctuation and whitespace; never paraphrase, "
            "summarize, translate, normalize, splice, or invent evidence. "
            "relation_hints must be empty when allowed_relation_targets is "
            "empty. risk must be exactly normal or review."
        )
        return f"{rules}\nINPUT={_canonical_json(payload)}"

    @staticmethod
    def _build_contract_repair_prompt(prompt: str, code: str) -> str:
        instruction = _CONTRACT_REPAIR_INSTRUCTION.format(code=code)
        return f"{instruction}\n{prompt}"

    @staticmethod
    def _build_incomplete_retry_prompt(prompt: str) -> str:
        return f"{_INCOMPLETE_RETRY_INSTRUCTION}\n{prompt}"

    @staticmethod
    def _extract_message(envelope: Any) -> str:
        # Only accept plain JSON container types.  Arbitrary Mapping/list
        # subclasses can execute code from ``get``/index operations after the
        # provider deadline and can leak their own exception payloads.
        if type(envelope) is not dict:
            raise ProposerContractError(
                "provider.no_choices", "provider envelope has no choices"
            )
        if "error" in envelope:
            raise ProposerContractError(
                "provider.error", "provider returned an error envelope"
            )
        if envelope.get("refusal") is not None:
            raise ProposerContractError(
                "provider.refusal", "provider refused the request"
            )
        choices = envelope.get("choices")
        if (
            type(choices) is not list
            or not choices
            or type(choices[0]) is not dict
        ):
            raise ProposerContractError(
                "provider.no_choices", "provider envelope has no choices"
            )
        choice = choices[0]
        if "error" in choice:
            raise ProposerContractError(
                "provider.error", "provider returned an error choice"
            )
        if choice.get("refusal") is not None:
            raise ProposerContractError(
                "provider.refusal", "provider refused the request"
            )
        if (
            "finish_reason" in choice
            and choice["finish_reason"] != "stop"
        ):
            raise ProposerContractError(
                "provider.incomplete", "provider response did not finish normally"
            )
        message = choice.get("message")
        if type(message) is not dict:
            raise ProposerContractError(
                "provider.no_message", "first choice has no message"
            )
        if "error" in message:
            raise ProposerContractError(
                "provider.error", "provider returned an error message"
            )
        if message.get("refusal") is not None:
            raise ProposerContractError(
                "provider.refusal", "provider refused the request"
            )
        content = message.get("content")
        if type(content) is not str:
            raise ProposerContractError(
                "provider.no_message", "message content is not text"
            )
        return content

    @staticmethod
    def _parse_json(response: str) -> Any:
        try:
            parsed = json.loads(
                response,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
        except _DuplicateKey as exc:
            raise ProposerContractError(
                "parse_duplicate_key", "JSON contains a duplicate key"
            ) from exc
        except _NonFinite as exc:
            raise ProposerContractError(
                "parse_nonfinite", "JSON contains a non-finite number"
            ) from exc
        except (UnicodeError, RecursionError, ValueError) as exc:
            raise ProposerContractError(
                "parse_json", "message is not one valid JSON value"
            ) from exc
        if _contains_nonfinite(parsed):
            raise ProposerContractError(
                "parse_nonfinite", "JSON contains a non-finite number"
            )
        return parsed

    def _validate_root(
        self,
        parsed: Any,
        chunks: tuple[ProposerChunk, ...],
        targets: frozenset[str],
        *,
        max_candidates: int,
    ) -> tuple[CandidateDraft, ...]:
        root = _expect_exact_mapping(
            parsed,
            _ROOT_FIELDS,
            "schema_root",
            "root object does not exactly match schema",
        )
        if (
            not _is_plain_int(root["schema_version"])
            or root["schema_version"] != SCHEMA_VERSION
            or not isinstance(root["candidates"], list)
            or len(root["candidates"]) > max_candidates
        ):
            raise ProposerContractError(
                "schema_root", "root values do not match schema"
            )
        chunk_by_id = {chunk.id: chunk.text for chunk in chunks}
        # Build the whole tuple before returning: one invalid item rejects the
        # entire model batch and no partial result can escape.
        return tuple(
            self._validate_candidate(candidate, chunk_by_id, targets)
            for candidate in root["candidates"]
        )

    def _validate_candidate(
        self,
        raw: Any,
        chunk_by_id: Mapping[str, str],
        targets: frozenset[str],
    ) -> CandidateDraft:
        candidate = _expect_exact_mapping(
            raw,
            _CANDIDATE_FIELDS,
            "schema_candidate",
            "candidate does not exactly match schema",
        )
        string_fields = ("type", "title", "content", "thread_hint", "evidence", "risk")
        if any(not isinstance(candidate[field], str) for field in string_fields):
            raise ProposerContractError(
                "schema_candidate", "candidate field types do not match schema"
            )
        if candidate["type"] not in CANDIDATE_TYPES:
            raise ProposerContractError(
                "schema_candidate", "candidate type is unsupported"
            )
        if (
            not _is_bounded_text(candidate["title"], _MAX_TITLE_BYTES)
            or not _is_bounded_text(candidate["content"], _MAX_CONTENT_BYTES)
            or not _is_optional_bounded_text(
                candidate["thread_hint"], _MAX_THREAD_HINT_BYTES
            )
            or candidate["risk"] not in _RISKS
        ):
            raise ProposerContractError(
                "schema_candidate", "candidate text values do not match schema"
            )
        if (
            not _is_plain_int(candidate["importance"])
            or not 1 <= candidate["importance"] <= 10
            or not isinstance(candidate["relation_hints"], list)
            or not isinstance(candidate["source_chunk_ids"], list)
        ):
            raise ProposerContractError(
                "schema_candidate", "candidate field types do not match schema"
            )

        source_ids = candidate["source_chunk_ids"]
        if len(chunk_by_id) == 1:
            # The production coordinator proposes one chunk at a time. Bind
            # that sole source locally so a model typo cannot corrupt or block
            # provenance. The persisted draft always receives the real input
            # id, while the model hint remains structurally bounded.
            if (
                len(source_ids) > 1
                or any(
                    not _is_bounded_text(source_id, _MAX_ID_BYTES)
                    for source_id in source_ids
                )
            ):
                raise ProposerContractError(
                    "provenance_source",
                    "single-chunk source hint is not structurally bounded",
                )
            bound_source_ids = tuple(chunk_by_id)
        elif (
            not source_ids
            or any(
                not _is_bounded_text(source_id, _MAX_ID_BYTES)
                for source_id in source_ids
            )
            or len(source_ids) != len(set(source_ids))
            or any(source_id not in chunk_by_id for source_id in source_ids)
        ):
            raise ProposerContractError(
                "provenance_source",
                "candidate sources must be a unique non-empty input subset",
            )
        else:
            bound_source_ids = tuple(source_ids)
        evidence = candidate["evidence"]
        if not _is_bounded_text(evidence, _MAX_EVIDENCE_BYTES) or not any(
            evidence in chunk_by_id[source_id]
            for source_id in bound_source_ids
        ):
            raise ProposerContractError(
                "provenance_evidence",
                "candidate evidence is not literal in a cited chunk",
            )

        relations = tuple(
            self._validate_relation(relation, targets)
            for relation in candidate["relation_hints"]
        )
        return CandidateDraft(
            type=candidate["type"],
            title=candidate["title"],
            content=candidate["content"],
            importance=candidate["importance"],
            thread_hint=candidate["thread_hint"],
            relation_hints=relations,
            source_chunk_ids=bound_source_ids,
            evidence=evidence,
            risk=candidate["risk"],
        )

    @staticmethod
    def _validate_relation(
        raw: Any, targets: frozenset[str]
    ) -> RelationHint:
        relation = _expect_exact_mapping(
            raw,
            _RELATION_FIELDS,
            "schema_relation",
            "relation does not exactly match schema",
        )
        if (
            not isinstance(relation["relation_type"], str)
            or relation["relation_type"] not in RELATION_TYPES
            or not _is_bounded_text(relation["target_id"], _MAX_ID_BYTES)
            or not _is_bounded_text(
                relation["reason"], _MAX_RELATION_REASON_BYTES
            )
            or not _is_finite_number(relation["strength"])
            or not 0 <= relation["strength"] <= 1
        ):
            raise ProposerContractError(
                "schema_relation", "relation values do not match schema"
            )
        if not targets or relation["target_id"] not in targets:
            raise ProposerContractError(
                "relation_target", "relation target is not pre-authorized"
            )
        strength = float(relation["strength"])
        if strength == 0:
            strength = 0.0
        return RelationHint(
            relation_type=relation["relation_type"],
            target_id=relation["target_id"],
            strength=strength,
            reason=relation["reason"],
        )


__all__ = [
    "CANDIDATE_TYPES",
    "CandidateDraft",
    "ProposerBatch",
    "ProposerChunk",
    "ProposerContractError",
    "RelationHint",
    "SCHEMA_VERSION",
    "StrictOmbreProposer",
    "route_candidate_axes",
]
