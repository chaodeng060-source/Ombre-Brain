"""Private, immutable dictionary generations; never mutates jieba.dt."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from bm25_index import _keep_token


class RetrievalTokenizer:
    def __init__(self, dictionary: bytes):
        import jieba
        text = dictionary.decode("utf-8")
        seen = set()
        for line in text.splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) > 3 or fields[0] in seen:
                raise ValueError("invalid_or_duplicate_dictionary_entry")
            if len(fields) >= 2 and (not fields[1].isdigit() or int(fields[1]) <= 0):
                raise ValueError("invalid_dictionary_frequency")
            seen.add(fields[0])
        self.version = "jieba-search-v1+private-" + hashlib.sha256(dictionary).hexdigest()
        self._tokenizer = jieba.Tokenizer()
        self._tokenizer.load_userdict(io.StringIO(text))

    @classmethod
    def from_file(cls, path):
        return cls(Path(path).read_bytes())

    def __call__(self, text):
        if not text:
            return []
        return [token for token in self._tokenizer.cut_for_search(text.lower())
                if token.strip() and _keep_token(token.strip())]
