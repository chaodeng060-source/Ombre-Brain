import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from curated_writer import CuratedWriteCoordinator
from maintenance_barrier import MaintenanceBarrier


REFUSAL = "Kiro replied: I cannot generate explicit sexual content."


@pytest.mark.asyncio
async def test_curated_replay_omits_before_existing_ledger_or_vector_access():
    writer = object.__new__(CuratedWriteCoordinator)
    writer._write_locked = AsyncMock(side_effect=AssertionError("storage touched"))
    result = await writer.write(
        idempotency_key="old-refusal", content=REFUSAL, vector_policy="required"
    )
    assert result.status == "omitted"
    assert result.bucket_id is None
    assert result.recall_state == "absent"
    writer._write_locked.assert_not_awaited()


@pytest.mark.asyncio
async def test_unrelated_curated_payload_is_byte_preserved(tmp_path):
    writer = object.__new__(CuratedWriteCoordinator)
    writer._maintenance_barrier = MaintenanceBarrier(tmp_path)
    writer._write_locked = AsyncMock(return_value="accepted")
    original = "  A quiet evening with a familiar companion.\n\n"
    assert await writer.write(
        idempotency_key="ordinary", content=original, vector_policy="required"
    ) == "accepted"
    assert writer._write_locked.await_args.kwargs["content"] == original


@pytest.mark.asyncio
async def test_corrected_cached_digest_matches_frontmatter_canonical_body(tmp_path,monkeypatch):
    rules=tmp_path/'reviewed.json'
    rules.write_text(json.dumps({'schema':'twin.memory-omission-overrides/v1','replacements':[{'old':'- reviewed old refusal','new':''}]}))
    monkeypatch.setenv('MEMORY_OMISSION_RULES_FILE',str(rules))
    writer=object.__new__(CuratedWriteCoordinator)
    writer._maintenance_barrier=MaintenanceBarrier(tmp_path)
    writer._write_locked=AsyncMock(return_value='accepted')
    assert await writer.write(idempotency_key='cached-digest',content='- keep the useful fact\n- reviewed old refusal',vector_policy='required')=='accepted'
    assert writer._write_locked.await_args.kwargs['content']=='- keep the useful fact'


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["hold", "grow"])
async def test_mcp_ingress_omits_without_model_or_writer(name):
    # Execute the real entrypoint body without importing the production server
    # (which would initialize its models, network clients and vault).
    path = Path(__file__).parents[1] / "server.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    result = await namespace[name](REFUSAL)
    assert "未写入" in result
