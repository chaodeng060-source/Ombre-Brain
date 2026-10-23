"""Apply only the JEV entry points to an already customized server.py.

This avoids replacing a deployed module with an older Git baseline. Pass a
path and redirect stdout to a new file; this tool never writes the input.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ENTRY = '''    if provider_override is None and jev_memory_gate.enabled(config):
        batch = await jev_memory_gate.score(
            config, query, buckets, redact=redact_embedding_input,
            timeout=_ds_gate_timeout(),
        )
        # Current NAS builds already own threshold/bonuses/topic dedup here.
        # Older source checkouts retain their forced-ID selection contract.
        score_selector = globals().get("_ds_score_mode_select")
        if score_selector is not None and _ds_gate_score_mode_enabled():
            return score_selector(buckets, batch.scores, keep, max_results)
        return jev_memory_gate.legacy_select(buckets, batch.scores, keep, max_results)
'''


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    if "import jev_memory_gate\n" in source:
        if ENTRY not in source or '"reranker": jev_memory_gate.status(config),' not in source:
            raise ValueError("Incomplete existing JEV patch")
        return source
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    target = definitions["_ds_semantic_select"]
    if not any(arg.arg == "provider_override" for arg in target.args.kwonlyargs):
        raise ValueError("Unsupported semantic-gate signature")
    body = target.body
    index = 1 if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str) else 0
    lines = source.splitlines(keepends=True)
    lines.insert(body[index].lineno - 1, ENTRY)
    result = "".join(lines)
    if result.count("import httpx\n") != 1:
        raise ValueError("Expected one httpx import")
    result = result.replace("import httpx\n", "import httpx\nimport jev_memory_gate\n", 1)
    health = definitions["health_check"]
    old_health = "".join(source.splitlines(keepends=True)[health.lineno - 1:health.end_lineno])
    marker = '            "decay_engine": "running" if decay_engine.is_running else "stopped",\n'
    if old_health.count(marker) != 1:
        raise ValueError("Unsupported health response")
    new_health = old_health.replace(marker, marker + '            "reranker": jev_memory_gate.status(config),\n', 1)
    result = result.replace(old_health, new_health, 1)
    ast.parse(result)
    return result


if __name__ == "__main__":
    print(patch_source(Path(sys.argv[1]).read_text(encoding="utf-8")), end="")
