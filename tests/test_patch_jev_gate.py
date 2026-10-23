"""Only requested entry points change; older and customized deployments work."""
import ast

import pytest

from tools.patch_jev_gate import patch_source


SOURCE = '''import httpx
KEEP_UNRELATED = "a deployed custom fix"
async def _ds_semantic_select(query, buckets, keep, max_results, *, provider_override=None):
    """A customized deployment's existing docstring."""
    provider, model, client, provider_kwargs = resolve_provider()
    return await old_gate(query, buckets, keep, max_results)

async def health_check(request):
    try:
        return JSONResponse({
            "status": "ok",
            "memory_counts": {"physical_ids": 123},
            "decay_engine": "running" if decay_engine.is_running else "stopped",
        })
    except Exception:
        return error_response()

def unrelated():
    return "never replace this code"
'''


def test_custom_code_and_docstring_are_preserved():
    result = patch_source(SOURCE)
    assert '"physical_ids": 123' in result
    assert 'return "never replace this code"' in result
    assert "a deployed custom fix" in result
    assert "deployment's existing docstring" in result
    assert 'return await old_gate(query, buckets, keep, max_results)' in result
    assert 'provider_override is None and jev_memory_gate.enabled(config)' in result
    ast.parse(result)
    assert patch_source(result) == result


def test_signature_mismatch_does_not_generate_a_guessed_patch():
    with pytest.raises(ValueError, match="signature"):
        patch_source(SOURCE.replace(", *, provider_override=None", ""))
