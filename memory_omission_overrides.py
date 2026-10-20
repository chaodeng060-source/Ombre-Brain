"""Exact owner-reviewed corrections; private text never ships in this module."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path


@lru_cache(maxsize=4)
def _rules(path: str, mtime_ns: int, size: int) -> tuple[tuple[str, str], ...]:
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 'twin.memory-omission-overrides/v1':
        raise ValueError('invalid reviewed memory omission schema')
    rows = data.get('replacements', [])
    result = []
    for row in rows:
        if not isinstance(row.get('old'), str) or not row['old'] or not isinstance(row.get('new'), str):
            raise ValueError('invalid reviewed memory omission replacement')
        result.append((row['old'], row['new']))
    return tuple(result)


def apply_reviewed_omissions(text: str) -> str:
    configured = os.environ.get('MEMORY_OMISSION_RULES_FILE')
    paths = [Path(configured)] if configured else [
        Path(__file__).resolve().parent / 'data' / 'memory_omissions.json',
        Path('/data/.maintenance/memory-omissions.json'),
    ]
    for path in paths:
        if not path.is_file():
            continue
        stat = path.stat()
        for old, new in _rules(str(path), stat.st_mtime_ns, stat.st_size):
            text = text.replace(old, new)
    return text
