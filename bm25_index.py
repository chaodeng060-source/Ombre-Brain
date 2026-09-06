"""bm25_index.py — BM25 稀疏检索，配合 jieba 中文分词。

照上游 P0luz/Ombre-Brain `src/bm25_index.py` 复刻（2026-08-25 一手读取）。
上游那份被 `src/bucket_manager.py`、`src/tools/dream/feel_rank.py` 引用，
带 `tests/test_bm25_async_rebuild.py`，写进 `docs/INTERNALS.md` 与
`update_manifest.json` —— 是正式部件，不是实验代码。本地生产 `code/` 零命中。

为什么补这个（四叶草老师 2026-08-24 14:28 原话）：
    「分词也没做好」
蛋同场补：「她连同义词都没过关」

本地现状：词法通道是 `bucket_manager._calc_topic_score()` 的
jieba 命中计数 + `fuzz.ratio` 字符串相似度。它按「字面像不像」给分，
不按「这个词在这个库里有多罕见」给分。于是高频词一撞就满分：
    「有东西」  → Lamp present in scene
    「没反应」  → 任务停滞 / 延迟抱怨
    「不删」    → 不删_2026-08-11
BM25 的 IDF 项正是治这个：满库都有的词权重压到极低，罕见词才顶分。

rank_bm25 / jieba 均为软依赖：未安装时所有方法静默 no-op，
不影响其余检索维度——**装不上也不会让召回变差**，这条是上游的设计，照抄。

状态：**未验证。** 本机缺 rank_bm25 与 jieba，且无人值守下装不了包、
读不到生产桶数据，无法拿真实病例跑前后对比。不得据此宣称召回已改善。
"""
from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger("ombre_brain.bm25")

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi = None  # type: ignore
    _BM25_AVAILABLE = False
    logger.info("[bm25] rank_bm25 未安装 — BM25 关键词检索已禁用（pip install rank-bm25 启用）")

try:
    import jieba as _jieba
    _jieba.setLogLevel(logging.WARNING)
    _JIEBA_AVAILABLE = True
except ImportError:
    _jieba = None  # type: ignore
    _JIEBA_AVAILABLE = False
    logger.info("[bm25] jieba 未安装 — 回退空格分词（pip install jieba 启用中文分词）")


# 分词卫生:纯标点、单个拉丁字符、口癖/笑声 token 不进语料——
# 「O.o」碎成 o/./o、「哈哈哈哈」枚举一串子词,字面撞口癖是实测过的噪音源
# (账本零控题 t2793 的假命中全靠撞 O.o)。IDF 治不了它们:口癖中频、语义为零。
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")
_STOP_TOKENS = {
    "tvt", "qaq", "qwq", "orz", "emm", "emmm", "hhh", "hhhh",
    "哈哈", "哈哈哈", "哈哈哈哈", "嘿嘿", "嘻嘻", "呜呜", "呜呜呜",
    "嘤嘤", "咳咳", "唔唔", "嘿嘿嘿", "嘻嘻嘻",
    # 口语框架词:账本实测的噪音源——睡美人题的假命中全靠撞「刚才/这样子/诶」。
    # 这些词中频(IDF 治不动)、语义为零,query 和语料两侧都滤。
    "刚才", "这样子", "那样子", "这样", "那样", "其实", "居然", "好像",
    "真的", "就是", "但是", "然后", "所以", "可能", "应该", "现在",
    "一下", "时候", "什么", "怎么", "怎么样", "为什么", "能不能",
    "是不是", "要不要", "可不可以", "还是", "还有", "有点", "一点",
}


def _keep_token(t: str) -> bool:
    if _PUNCT_ONLY_RE.match(t):
        return False
    if len(t) == 1 and not ("一" <= t <= "鿿"):
        return False
    if t in _STOP_TOKENS:
        return False
    return True


def _tokenize(text: str) -> list[str]:
    """中文 jieba 分词 + 空格切割英文，小写，过滤空串。"""
    if not text:
        return []
    text = text.lower()
    if _JIEBA_AVAILABLE:
        tokens = list(_jieba.cut_for_search(text))
    else:
        tokens = text.split()
    return [t for t in tokens if t.strip() and _keep_token(t.strip())]


class _LazyIdf:
    """Mapping-compatible exact Okapi IDF view over incremental df stats."""

    def __init__(
        self,
        term_doc_counts: dict[str, int],
        corpus_size: int,
        epsilon: float,
        average_idf: float,
    ):
        self._term_doc_counts = term_doc_counts
        self._corpus_size = corpus_size
        self._epsilon = epsilon
        self._average_idf = average_idf

    def __len__(self) -> int:
        return len(self._term_doc_counts)

    def __iter__(self):
        return iter(self._term_doc_counts)

    def __getitem__(self, term: str) -> float:
        frequency = self._term_doc_counts[term]
        value = (
            math.log(self._corpus_size - frequency + 0.5)
            - math.log(frequency + 0.5)
        )
        return (
            self._epsilon * self._average_idf
            if value < 0
            else value
        )

    def get(self, term: str, default=None):
        if term not in self._term_doc_counts:
            return default
        return self[term]


class BM25Index:
    """In-memory BM25 with copy-on-write single-document generations."""

    def __init__(self):
        self._index = None
        self._ids: list[str] = []
        # term -> immutable bucket-id set.  Only touched postings are copied by
        # an incremental generation; readers keep the previous complete map.
        self._postings: dict[str, frozenset[str]] = {}
        self._term_doc_counts: dict[str, int] = {}
        self._df_histogram: dict[int, int] = {}
        self._total_doc_length = 0

    @property
    def available(self) -> bool:
        return _BM25_AVAILABLE

    @staticmethod
    def _bucket_tokens(bucket: dict) -> list[str]:
        meta = bucket.get("metadata", {}) or {}
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        domain = meta.get("domain") or []
        if isinstance(domain, str):
            domain = [domain]
        text = " ".join([
            meta.get("name") or "",
            (bucket.get("content") or "")[:1200],
            " ".join(str(tag) for tag in tags),
            " ".join(str(value) for value in domain),
        ])
        return _tokenize(text)

    @staticmethod
    def _frequencies(tokens: list[str]) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    @staticmethod
    def _average_idf(
        corpus_size: int,
        df_histogram: dict[int, int],
    ) -> float:
        term_count = sum(df_histogram.values())
        if corpus_size <= 0 or term_count <= 0:
            return 0.0
        return math.fsum(
            count
            * (
                math.log(corpus_size - frequency + 0.5)
                - math.log(frequency + 0.5)
            )
            for frequency, count in sorted(df_histogram.items())
        ) / term_count

    @classmethod
    def _rank_index_from_parts(
        cls,
        template,
        doc_freqs: list[dict[str, int]],
        doc_len: list[int],
        term_doc_counts: dict[str, int],
        df_histogram: dict[int, int],
    ):
        if not doc_freqs:
            return None
        index = _BM25Okapi.__new__(_BM25Okapi)
        index.k1 = template.k1
        index.b = template.b
        index.epsilon = template.epsilon
        index.corpus_size = len(doc_freqs)
        index.doc_freqs = doc_freqs
        index.doc_len = doc_len
        index.avgdl = sum(doc_len) / len(doc_len)
        index.tokenizer = getattr(template, "tokenizer", None)
        index.average_idf = cls._average_idf(
            index.corpus_size,
            df_histogram,
        )
        index.idf = _LazyIdf(
            term_doc_counts,
            index.corpus_size,
            index.epsilon,
            index.average_idf,
        )
        return index

    def build(self, buckets: list[dict]) -> None:
        """Build one complete generation with the unchanged corpus contract."""
        if not _BM25_AVAILABLE:
            return
        corpus: list[list[str]] = []
        ids: list[str] = []
        for bucket in buckets:
            tokens = self._bucket_tokens(bucket)
            if tokens:
                corpus.append(tokens)
                ids.append(bucket["id"])

        self._index = _BM25Okapi(corpus) if corpus else None
        self._ids = ids
        postings: dict[str, set[str]] = {}
        if self._index is not None:
            for bucket_id, frequencies in zip(ids, self._index.doc_freqs):
                for term in frequencies:
                    postings.setdefault(term, set()).add(bucket_id)
        self._postings = {
            term: frozenset(bucket_ids)
            for term, bucket_ids in postings.items()
        }
        self._term_doc_counts = {
            term: len(bucket_ids)
            for term, bucket_ids in self._postings.items()
        }
        self._df_histogram = {}
        for frequency in self._term_doc_counts.values():
            self._df_histogram[frequency] = (
                self._df_histogram.get(frequency, 0) + 1
            )
        self._total_doc_length = (
            sum(self._index.doc_len) if self._index is not None else 0
        )
        if self._index is not None:
            self._index.average_idf = self._average_idf(
                self._index.corpus_size,
                self._df_histogram,
            )
            self._index.idf = _LazyIdf(
                self._term_doc_counts,
                self._index.corpus_size,
                self._index.epsilon,
                self._index.average_idf,
            )

    def _replace_document(self, bucket_id: str, tokens: list[str]):
        if not _BM25_AVAILABLE or self._index is None:
            raise RuntimeError("incremental BM25 requires a complete resident generation")

        ids = list(self._ids)
        doc_freqs = list(self._index.doc_freqs)
        doc_len = list(self._index.doc_len)
        postings = dict(self._postings)
        term_doc_counts = dict(self._term_doc_counts)
        df_histogram = dict(self._df_histogram)
        new_frequencies = self._frequencies(tokens)

        try:
            position = ids.index(bucket_id)
        except ValueError:
            position = -1
        old_frequencies = doc_freqs[position] if position >= 0 else {}

        for term in set(old_frequencies) | set(new_frequencies):
            old_df = term_doc_counts.get(term, 0)
            bucket_ids = set(postings.get(term, ()))
            if term in old_frequencies:
                bucket_ids.discard(bucket_id)
            if term in new_frequencies:
                bucket_ids.add(bucket_id)
            if bucket_ids:
                postings[term] = frozenset(bucket_ids)
                term_doc_counts[term] = len(bucket_ids)
            else:
                postings.pop(term, None)
                term_doc_counts.pop(term, None)
            new_df = len(bucket_ids)
            if old_df != new_df:
                if old_df:
                    remaining = df_histogram.get(old_df, 0) - 1
                    if remaining > 0:
                        df_histogram[old_df] = remaining
                    else:
                        df_histogram.pop(old_df, None)
                if new_df:
                    df_histogram[new_df] = df_histogram.get(new_df, 0) + 1

        if position >= 0 and new_frequencies:
            doc_freqs[position] = new_frequencies
            doc_len[position] = len(tokens)
        elif position >= 0:
            ids.pop(position)
            doc_freqs.pop(position)
            doc_len.pop(position)
        elif new_frequencies:
            ids.append(bucket_id)
            doc_freqs.append(new_frequencies)
            doc_len.append(len(tokens))

        fresh = BM25Index()
        fresh._index = self._rank_index_from_parts(
            self._index,
            doc_freqs,
            doc_len,
            term_doc_counts,
            df_histogram,
        )
        fresh._ids = ids
        fresh._postings = postings
        fresh._term_doc_counts = term_doc_counts
        fresh._df_histogram = df_histogram
        fresh._total_doc_length = sum(doc_len)
        return fresh

    def with_upsert(self, bucket: dict):
        """Return a complete generation with one bucket added or replaced."""
        bucket_id = str(bucket.get("id", ""))
        if not bucket_id:
            raise ValueError("incremental BM25 upsert requires bucket id")
        return self._replace_document(bucket_id, self._bucket_tokens(bucket))

    def with_delete(self, bucket_id: str):
        """Return a complete generation without one bucket (idempotent)."""
        return self._replace_document(str(bucket_id), [])

    def rare_term_hits(
        self,
        query: str,
        *,
        max_df: int,
        min_term_chars: int = 2,
    ) -> dict[str, tuple[str, ...]]:
        """bucket_id -> query terms that are rare in the corpus and occur in it.

        2026-09-04 朝灯考「蚊子」：整库只有 3 条桶带这个词，字面全中，但她整句
        的向量落在 0.5 地板之下，候选变成「字面单路」被 0.55 上限压到 0.2475、
        卡在 conversation 线 0.25 之下——记了搜不到。8/31 那道上限要拦的是
        常见词撞车（刚才/这样子/看看），稀有词精确命中恰恰是最强的字面证据。
        这里只回答「哪些候选带着稀有查询词」，不打分、不排序；阈值由调用方给，
        单字词不算（单字太容易撞），rank_bm25 缺席或索引未建时返回空 = 不豁免。
        """
        if max_df <= 0 or not self._postings or not self._term_doc_counts:
            return {}
        hits: dict[str, list[str]] = {}
        seen: set[str] = set()
        for term in _tokenize(query):
            if term in seen or len(term) < max(1, min_term_chars):
                continue
            seen.add(term)
            df = self._term_doc_counts.get(term, 0)
            if df <= 0 or df > max_df:
                continue
            for bucket_id in self._postings.get(term, ()):
                hits.setdefault(bucket_id, []).append(term)
        return {bucket_id: tuple(terms) for bucket_id, terms in hits.items()}

    def literal_term_df_hits(
        self,
        query: str,
        *,
        bucket_ids: set[str],
        min_term_chars: int = 1,
    ) -> dict[str, tuple[tuple[str, int], ...]]:
        """Return exact query-term hits and corpus DF for selected buckets.

        This is evidence only: it does not score or rank.  Restricting the
        lookup to the already-fused candidate IDs avoids expanding a common
        posting list (for example ``任务``) into thousands of request-local
        rows.  Missing/incomplete indexes return no evidence so callers can
        fail open rather than deleting memories without a DF witness.
        """
        targets = {str(bucket_id) for bucket_id in bucket_ids if bucket_id}
        if not targets or not self._postings or not self._term_doc_counts:
            return {}

        hits: dict[str, list[tuple[str, int]]] = {}
        seen: set[str] = set()
        for term in _tokenize(query):
            if term in seen or len(term) < max(1, min_term_chars):
                continue
            seen.add(term)
            df = int(self._term_doc_counts.get(term, 0) or 0)
            if df <= 0:
                continue
            posting = self._postings.get(term, ())
            for bucket_id in targets:
                if bucket_id in posting:
                    hits.setdefault(bucket_id, []).append((term, df))
        return {bucket_id: tuple(terms) for bucket_id, terms in hits.items()}

    def score(self, query: str) -> dict[str, float]:
        """Return normalized BM25 scores; the rank_bm25 formula is unchanged."""
        if not _BM25_AVAILABLE or self._index is None:
            return {}
        tokens = _tokenize(query)
        if not tokens:
            return {}
        raw = self._index.get_scores(tokens)
        max_s = float(raw.max()) if raw.size > 0 else 0.0
        if max_s <= 0:
            return {}
        return {
            bucket_id: float(score) / max_s
            for bucket_id, score in zip(self._ids, raw)
            if score > 0
        }
