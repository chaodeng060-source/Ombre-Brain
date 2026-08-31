"""PostgreSQL curated FTS + short literal recall for the Ombre read path.

This is the local adapter for LMC-5's existing cascade:

* literal search is independent of the vector score for short CJK/proper names;
* curated FTS runs only when the caller's top vector score is below its floor;
* the adapter returns bucket ids and scores only.  Bucket bodies remain in the
  existing BucketManager materialization and authority filters.

No model is called here.  PostgreSQL is a rebuildable index over Markdown.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from vendor.lmc5_pgvector.recall_pipeline import (
    literal_query_terms,
    should_run_literal_search,
)

logger = logging.getLogger("ombre_brain.curated_lexical")

try:
    import jieba

    jieba.setLogLevel(logging.WARNING)
except ImportError:  # pragma: no cover - runtime requirements install jieba
    jieba = None  # type: ignore[assignment]

try:
    from recall_timing import recall_stage
except ImportError:  # pragma: no cover - standalone contract tests
    from contextlib import nullcontext

    def recall_stage(_name: str):
        return nullcontext()


@dataclass(frozen=True)
class CuratedLexicalHit:
    bucket_id: str
    score: float
    channel: str
    original_support: float


_QUERY_PREFIXES = (
    "你还记得", "你帮我", "帮我", "能不能", "可不可以", "你能不能", "你能",
    "麻烦", "请问", "刚才", "上次", "之前", "以前", "这个", "那个",
    "记得", "还记得", "说过", "聊过", "提过", "关于", "查一下",
    "搜一下", "找一下", "搜索", "检索", "查", "搜", "找",
)
_QUERY_SUFFIXES = (
    "是怎么回事", "怎么回事", "是什么", "在哪里", "在哪儿", "还在吗",
    "的事", "的事情", "这件事", "吗", "呢", "啊", "呀", "么", "吧",
)
_GENERIC_TERMS = frozenset({
    "事情", "东西", "情况", "问题", "内容", "时候", "一下", "这个", "那个",
    "什么", "怎么", "为什么", "记得", "说过", "聊过", "提过", "之前", "以前",
    "刚才", "今天", "昨天", "现在", "我们", "你们", "他们", "自己", "可以",
    "还是", "已经", "还有", "没有", "一个", "就是", "然后", "真的", "知道",
    "单子", "任务", "进度", "修改", "处理", "情况", "新的", "干嘛", "工作",
    "继续", "完成", "问题", "代码", "测试", "结果", "目前", "已经", "又在",
    "不是", "不要", "不能", "不会", "不想", "不用", "不行", "需要", "只要",
})
_EDGE_PUNCTUATION = " \t\r\n：:，,。.!！?？、；;“”‘’'\"`「」『』（）()[]【】"
_QUOTED_LITERAL_RE = re.compile(r"[「『“\"'`]([^」』”\"'`]{2,80})[」』”\"'`]")
_EXPLICIT_LITERAL_RE = re.compile(
    r"(?:记得|还记得|说过|聊过|提过|关于|搜一下|查一下|找一下|搜索|检索|搜|查|找)"
)


def _strip_query_shell(text: str) -> str:
    value = str(text or "").strip(_EDGE_PUNCTUATION)
    changed = True
    while changed and value:
        changed = False
        for prefix in _QUERY_PREFIXES:
            if value.startswith(prefix) and len(value) - len(prefix) >= 2:
                value = value[len(prefix):].strip(_EDGE_PUNCTUATION)
                changed = True
                break
        for suffix in _QUERY_SUFFIXES:
            if value.endswith(suffix) and len(value) - len(suffix) >= 2:
                value = value[:-len(suffix)].strip(_EDGE_PUNCTUATION)
                changed = True
                break
    return value


def substantive_literal_terms(query: str, max_terms: int = 3) -> list[str]:
    """Keep only the strongest upstream literal terms, never command ngrams."""
    stripped_query = _strip_query_shell(query)
    if stripped_query and _EXPLICIT_LITERAL_RE.search(str(query or "")):
        # For explicit navigation, the shell-stripped phrase is the evidence.
        # Upstream ngrams can otherwise cut through a cue boundary and produce
        # fragments such as ``得毕业答辩`` / ``记得毕``.
        return [stripped_query.casefold()][: max(1, min(int(max_terms), 3))]

    candidates = literal_query_terms(query, max_terms=16)
    if stripped_query:
        candidates.insert(0, stripped_query)

    normalized: list[str] = []
    for candidate in candidates:
        value = _strip_query_shell(candidate).casefold()
        compact = re.sub(r"\s+", " ", value).strip()
        if (
            len(compact) < 2
            or len(compact) > 48
            or compact in _GENERIC_TERMS
            or compact in normalized
        ):
            continue
        normalized.append(compact)

    # Upstream can emit overlapping command ngrams. Keep the longest semantic
    # phrase and discard shorter substrings such as "答辩" beside "毕业答辩".
    selected: list[str] = []
    for value in sorted(normalized, key=lambda item: (-len(item), normalized.index(item))):
        if any(value in kept for kept in selected):
            continue
        selected.append(value)
        if len(selected) >= max(1, min(int(max_terms), 3)):
            break
    return selected


def should_run_local_literal(query: str) -> bool:
    """Narrow upstream's helper to actual navigation, names, and codenames.

    The upstream generic ASCII branch would treat a long complaint containing
    an English word (for example ``safety``) as a literal-memory request. That
    produced three demonstrably unrelated lunch memories in the real 20-turn
    shadow, so the local adapter keeps the same intended use but not that leak.
    """
    text = str(query or "").strip()
    if not should_run_literal_search(text):
        return False
    if _QUOTED_LITERAL_RE.search(text):
        return True
    if len(text) <= 40 and _EXPLICIT_LITERAL_RE.search(text):
        return True
    compact = re.sub(r"[\s：:，,。.!！?？、；;“”‘’'\"`「」『』（）()【】\[\]]+", "", text)
    if re.fullmatch(r"[\u3400-\u9fff]{2,8}", compact):
        return compact not in _GENERIC_TERMS
    if len(text) <= 40 and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,40}", text):
        return True
    return False


def lexical_query(text: str) -> str:
    """Use the same pre-tokenization contract as pg_mirror_body_sync.py."""
    value = _strip_query_shell(text).casefold().strip()
    if not value or len(str(text or "").strip()) > 80:
        return ""
    if jieba is None:
        return ""
    tokens: list[str] = []
    for token in jieba.cut_for_search(value):
        token = token.strip()
        if len(token) < 2 or token in _GENERIC_TERMS or token in tokens:
            continue
        tokens.append(token)
    return " ".join(tokens)


def pg_lexical_mode() -> str:
    """Return off/shadow/live; default off, independent of PG vector rollout."""
    value = os.environ.get("OMBRE_PG_LEXICAL_MODE", "off").strip().lower()
    return value if value in {"off", "shadow", "live"} else "off"


def _merge_curated_rows(
    fts_rows: list[tuple[str, float]],
    literal_rows: list[tuple[str, float]],
    *,
    limit: int,
) -> list[CuratedLexicalHit]:
    """Merge bounded PG rows while preserving their evidence semantics."""
    merged: dict[str, CuratedLexicalHit] = {}
    for bucket_id, score in fts_rows:
        merged[bucket_id] = CuratedLexicalHit(
            bucket_id=bucket_id,
            score=max(0.0, min(1.0, score)),
            channel="curated_fts",
            # FTS is a candidate generator. The server still requires the
            # existing original-query literal/topic support before ranking.
            original_support=0.0,
        )
    for bucket_id, score in literal_rows:
        # Exact literal evidence wins over an FTS copy of the same bucket.
        merged[bucket_id] = CuratedLexicalHit(
            bucket_id=bucket_id,
            score=max(0.0, min(1.0, score)),
            channel="literal",
            original_support=1.0,
        )

    return sorted(
        merged.values(),
        key=lambda hit: (
            hit.channel == "literal",
            hit.score,
            hit.bucket_id,
        ),
        reverse=True,
    )[:limit]


async def search_curated_lexical(
    query: str,
    *,
    top_k: int = 20,
    include_fts: bool,
    dsn: str | None = None,
) -> list[CuratedLexicalHit]:
    """Return a bounded union of literal and (when requested) curated FTS hits.

    A database failure returns an empty list so the existing keyword/vector
    channels remain available.  Query/body text is never logged or returned.
    """
    if pg_lexical_mode() == "off" or not str(query or "").strip():
        return []
    try:
        import psycopg
    except ImportError:
        logger.warning("PG lexical recall enabled but psycopg is unavailable")
        return []

    limit = max(1, min(int(top_k), 20))
    literal_limit = min(3, limit)
    dsn = dsn or os.environ.get(
        "OMBRE_PG_RECALL_DSN",
        "postgresql:///ombre_mirror",
    )
    run_literal = should_run_local_literal(query)
    terms = substantive_literal_terms(query) if run_literal else []
    token_query = lexical_query(query) if include_fts else ""
    if not terms and not token_query:
        return []

    literal_rows: list[tuple[str, float]] = []
    fts_rows: list[tuple[str, float]] = []
    try:
        with recall_stage("curated_lexical_pg"):
            async with await psycopg.AsyncConnection.connect(
                dsn,
                connect_timeout=2,
                options=(
                    "-c default_transaction_read_only=on "
                    "-c statement_timeout=500ms -c lock_timeout=100ms"
                ),
            ) as conn:
                async with conn.cursor() as cur:
                    if terms:
                        # phraseto_tsquery uses the GIN index even for two-CJK
                        # terms; ILIKE is reserved for >=3 chars where pg_trgm
                        # has an indexable trigram. Upstream literal stays top3.
                        term_queries = [lexical_query(term) for term in terms]
                        term_queries = [term for term in term_queries if term]
                        patterns = [f"%{term}%" for term in terms if len(term) >= 3]
                        await cur.execute(
                            "WITH phrase_q AS ("
                            "  SELECT phraseto_tsquery('simple', term) AS tsq "
                            "  FROM unnest(%s::text[]) AS term"
                            ") "
                            "SELECT b.bucket_id, 1.0::double precision "
                            "FROM ombre_bodies b "
                            "WHERE b.lexical_version = 'jieba-search-v1' AND ("
                            "  EXISTS (SELECT 1 FROM phrase_q q WHERE b.search_tsv @@ q.tsq) "
                            "  OR (cardinality(%s::text[]) > 0 "
                            "      AND b.lexical_source_text ILIKE ANY(%s::text[]))"
                            ") "
                            "ORDER BY b.source_mtime DESC, b.bucket_id "
                            "LIMIT %s",
                            (term_queries, patterns, patterns, literal_limit),
                        )
                        literal_rows = [
                            (str(bucket_id), float(score))
                            for bucket_id, score in await cur.fetchall()
                        ]
                    if token_query:
                        await cur.execute(
                            "WITH q AS (SELECT plainto_tsquery('simple', %s) AS tsq) "
                            "SELECT bucket_id, "
                            "       ts_rank_cd(search_tsv, q.tsq, 32)::double precision "
                            "FROM ombre_bodies, q "
                            "WHERE lexical_version = 'jieba-search-v1' "
                            "  AND search_tsv @@ q.tsq "
                            "ORDER BY 2 DESC, bucket_id "
                            "LIMIT %s",
                            (token_query, limit),
                        )
                        fts_rows = [
                            (str(bucket_id), float(score))
                            for bucket_id, score in await cur.fetchall()
                        ]
    except Exception as exc:  # noqa: BLE001 - fail-soft index, never hide main recall
        logger.warning(
            "PG lexical recall failed (%s); keeping existing channels",
            type(exc).__name__,
        )
        return []

    return _merge_curated_rows(fts_rows, literal_rows, limit=limit)
