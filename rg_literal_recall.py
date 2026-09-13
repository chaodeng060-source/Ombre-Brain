"""ripgrep exact-substring recall channel over the bucket Markdown files.

朝灯 2026-09-08 19:59：「有现成的好东西不用非得用差的」「直接做召回的时候做搜索的时候
他在那里面跑不就可以了」。

Why this exists: the curated PG literal channel is ``off`` in production and,
even when on, it can only annotate buckets that the vector/keyword channels
already produced (server.py looks the id up in the existing candidate caches
and ``continue``s otherwise).  A bucket that contains her exact words but is
missed by embeddings and jieba tokens therefore never enters recall at all.

This channel runs ``rg -l -F`` over ``<buckets_dir>/dynamic`` with the same
substantive literal terms the PG channel would use, maps file names back to
bucket ids, and hands the ids to server.py, which escorts them into ``matches``
after the retention cutoff (same shape as the vector-strong escort).

Contract:
- ``OMBRE_RG_LITERAL_ENABLED`` default ``1``; ``0`` disables (= rollback).
- ``OMBRE_RG_LITERAL_TOP_K`` default 3, ``OMBRE_RG_LITERAL_TIMEOUT`` seconds
  default 0.6 for the whole channel.
- A term matching more than ``OMBRE_RG_LITERAL_MAX_FILES`` (default 40) files
  is too generic to be navigation evidence and is dropped.
- Any failure (no rg binary, timeout, bad dir) returns ``[]``; never raises.
- Query text is never logged.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
from dataclasses import dataclass

from curated_lexical_recall import substantive_literal_terms

_BUCKET_ID_RE = re.compile(r"_([0-9a-f]{12})\.md$")


@dataclass(frozen=True)
class RgLiteralHit:
    bucket_id: str
    score: float  # 0..1, share of matched term length
    term: str  # longest matched term (for logging term length only)


def rg_literal_enabled() -> bool:
    return os.environ.get("OMBRE_RG_LITERAL_ENABLED", "1").strip() not in {"0", "false", "off", ""}


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.05, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def bucket_id_from_path(path: str) -> str | None:
    m = _BUCKET_ID_RE.search(str(path or ""))
    return m.group(1) if m else None


_CLAUSE_SPLIT_RE = re.compile(
    r"[\s，,。.!！?？、；;：:“”‘’'\"`「」『』（）()【】\[\]…—\-~～]+"
)


try:  # same dependency the PG literal channel already uses
    import jieba  # type: ignore

    jieba.setLogLevel(60)
    jieba.initialize()  # ~1 s once at import, not on her first message
except Exception:  # pragma: no cover
    jieba = None

# Function words that carry no navigation value; dropped before joining
# adjacent content tokens into sub-phrases.
_FUNC_WORDS = set(
    "你 我 他 她 它 我们 你们 他们 咱们 咱 哥哥 老公 还 记得 记不记得 那张 那个 这个 那些 "
    "这些 什么 怎么 为什么 吗 吧 呢 啊 呀 哦 嗯 的 了 着 过 是 在 和 跟 就 都 也 不 没 要 "
    "有 没有 这 那 一下 一个 一次 一点 今天 昨天 明天 现在 刚才 之前 以后 时候 可以 不是 "
    "是不是 有没有 知道 觉得 感觉 看看 看 说 想 做 去 来 给 把 被 让 对 很 太 好 还是 "
    "但是 然后 所以 因为 如果 而且 或者 到底 其实 真的 已经 又 再 才 只 就是 这样 那样 "
    "怎么样 如何 为啥 啥 谁 哪 哪里 哪个 多少 几 图 张 个 条 次 件 不要 可以 需要 应该 "
    "一起 一样 东西 事情 问题 时间 地方 这里 那里 怎样 为何 自己 别人 大家 一直 总是 "
    "还有 还要 而已 或 及 与 等 之 于 从 到 向 往 比 像 好像 似乎 大概 可能 一定 肯定 "
    "非常 特别 比较 有点 一些 那么 这么 多 少 大 小 上 下 里 外 中 前 后 左 右".split()
)


_TWO_CHAR_PROPER_CACHE: dict[str, bool] = {}


def _is_all_cjk(term: str) -> bool:
    return bool(term) and all("一" <= ch <= "鿿" for ch in term)


def _is_proper_two_char(term: str) -> bool:
    """2 字中文词里，只有人名/专名配当字面导航词。

    9/8 上线这条通道时我把 2 字词全放行，指望 ``_FUNC_WORDS`` 和 120 文件上限
    兜住普通词。实测没兜住：她 9/8 22:49 问「为什么 NAS 那边你说不能装」，
    「那边」「不能」两个词都不在虚词表里、命中文件数也没到上限，于是把 5 月的
    旧 NAS 部署账护送进了候选（2026-09-09 朝灯：「没有噪音」）。

    判据跟本机 ``app/recall_literal_mirror.py`` 一致：jieba 词典里查不到这个词
    （未登录词，多半是名字），或者词性是 nr/nz/ns/nt。
    「惠普」「婷易」「内推」进得来，「那边」「不能」「多好」进不来。

    jieba 缺失时一律放行 —— 维持这条通道上线时的行为，宁可留点噪音，
    也不能悄悄把人名全砍掉换成「召回不完整」。
    """
    if jieba is None:
        return True
    cached = _TWO_CHAR_PROPER_CACHE.get(term)
    if cached is not None:
        return cached
    try:
        import jieba.posseg as pseg

        # 模块导入时已 jieba.initialize()；FREQ 空的时候 get 一律 None，
        # 「多好」这种普通词会被误判成专名。
        if jieba.dt.FREQ.get(term) is None:
            verdict = True
        else:
            verdict = any(
                p.flag in ("nr", "nz", "ns", "nt") for p in pseg.cut(term)
            )
    except Exception:
        verdict = True
    _TWO_CHAR_PROPER_CACHE[term] = verdict
    return verdict


def _sub_phrases(clause: str, *, min_len: int) -> list[str]:
    """Adjacent content tokens joined back into phrases, plus the tokens."""
    if jieba is None:
        return []
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        phrase = "".join(run)
        if len(phrase) >= min_len:
            out.append(phrase)
        for tok in run:
            if len(tok) >= 2:
                out.append(tok)
        run.clear()

    for tok in jieba.cut(clause):
        tok = tok.strip()
        if not tok or tok in _FUNC_WORDS or not re.search(r"[\w㐀-鿿]", tok):
            flush()
            continue
        # A lone character may follow a word as its suffix (``海马``+``体``
        # → ``海马体``, ``招商``+``岗``), but a word never continues a lone
        # character: ``体``+``人名`` must not become ``体人名``.
        if run and len(tok) > 1 and len(run[-1]) == 1 and len(run) == 1:
            flush()
        elif run and len(tok) > 1 and len(run[-1]) == 1:
            flush()
        run.append(tok)
    flush()
    return out


def rg_terms(query: str, max_terms: int = 6) -> list[str]:
    """Phrases worth an exact-substring search; [] means skip the channel.

    Exact substring wants real phrases, not the upstream command ngrams
    (those produced ``今天吃什`` / ``safety成精了是`` + ``是吧`` in the smoke).
    Terms = whole clauses (strongest evidence when they hit) + sub-phrases made
    of adjacent content tokens (``雾凇雪屋`` out of ``你还记得雾凇雪屋那张图吗``).
    A 2-char term is allowed only when the whole query is short (a name such
    as ``婷易`` typed alone is navigation); inside a sentence the 120-file cap
    in the search drops anything that generic anyway.
    """
    text = str(query or "").strip()
    if not text:
        return []
    # 2-char tokens are allowed everywhere (``惠普``, ``内推`` are exactly the
    # names worth navigating by); common ones are killed by the per-term file
    # cap in the search, and function words by _FUNC_WORDS.
    min_len = 2
    seen: set[str] = set()
    terms: list[str] = []

    def add(term: str) -> None:
        term = term.strip().casefold()
        if not term or term in seen or len(term) > 48:
            return
        if len(term) < min_len or (term.isascii() and len(term) < 4):
            return
        if len(term) == 2 and _is_all_cjk(term) and not _is_proper_two_char(term):
            return
        seen.add(term)
        terms.append(term)

    clauses = [c for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
    for clause in clauses:
        # A whole sentence is never stored verbatim; only short clauses can hit.
        if len(clause.strip()) <= 16:
            add(clause)
    for clause in clauses:
        for phrase in _sub_phrases(clause, min_len=min_len):
            add(phrase)
    if not terms:
        try:
            for t in substantive_literal_terms(text, max_terms=max_terms):
                add(t)
        except Exception:
            return []
    # Longest first: a whole clause or a joined phrase beats a lone token.
    terms.sort(key=lambda t: (-len(t), t))
    return terms[: max(1, min(int(max_terms), 6))]


async def _rg_multi(
    terms: list[str], root: str, *, timeout: float
) -> dict[str, dict[str, int]]:
    """One rg pass for all terms: path -> {term: occurrences}. {} on failure.

    ``-o --null`` prints ``path\\0matched_text`` per occurrence, so one ~100 ms
    scan serves every term instead of one scan per term.
    """
    rg = shutil.which("rg")
    if not rg or not terms:
        return {}
    args = [rg, "-o", "-i", "-F", "--null", "--no-messages", "-g", "*.md"]
    for term in terms:
        args += ["-e", term]
    args += ["--", root]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError):
        proc.kill()
        return {}
    by_term = {t.casefold(): t for t in terms}
    files: dict[str, dict[str, int]] = {}
    for line in out.decode("utf-8", "replace").splitlines():
        path, sep, matched = line.partition("\0")
        if not sep or not path:
            continue
        term = by_term.get(matched.casefold())
        if term is None:
            continue
        per_file = files.setdefault(path, {})
        per_file[term] = per_file.get(term, 0) + 1
    return files


async def search_rg_literal(
    query: str,
    *,
    buckets_dir: str,
    top_k: int | None = None,
    timeout: float | None = None,
) -> list[RgLiteralHit]:
    if not rg_literal_enabled() or not buckets_dir:
        return []
    root = os.path.join(str(buckets_dir), "dynamic")
    if not os.path.isdir(root):
        return []
    terms = rg_terms(query)
    if not terms:
        return []
    top_k = top_k or _int_env("OMBRE_RG_LITERAL_TOP_K", 3)
    budget = timeout or _float_env("OMBRE_RG_LITERAL_TIMEOUT", 0.8)
    max_files = _int_env("OMBRE_RG_LITERAL_MAX_FILES", 120)
    total_len = float(sum(len(t) for t in terms)) or 1.0
    # Candidates may be generous (the downstream gate reads full bodies and
    # judges relevance); what must not happen is a bucket that literally
    # contains her words never being seen.  Only 1-char evidence is refused.
    min_hit_term_len = 2
    scored: dict[str, float] = {}
    occurrences: dict[str, int] = {}
    longest: dict[str, str] = {}
    try:
        files = await _rg_multi(terms, root, timeout=budget)
        # A term hitting more files than max_files is too generic to navigate by.
        term_files: dict[str, int] = {}
        for per_file in files.values():
            for term in per_file:
                term_files[term] = term_files.get(term, 0) + 1
        generic = {t for t, n in term_files.items() if n > max_files}
        for path, per_file in files.items():
            bid = bucket_id_from_path(path)
            if not bid:
                continue
            for term, count in per_file.items():
                if term in generic:
                    continue
                scored[bid] = scored.get(bid, 0.0) + len(term) / total_len
                occurrences[bid] = occurrences.get(bid, 0) + int(count)
                if len(term) > len(longest.get(bid, "")):
                    longest[bid] = term
        for bid in [b for b, t in longest.items() if len(t) < min_hit_term_len]:
            scored.pop(bid, None)
            occurrences.pop(bid, None)
            longest.pop(bid, None)
    except Exception:
        return []
    hits = [
        RgLiteralHit(bucket_id=bid, score=round(min(1.0, score), 6), term=longest.get(bid, ""))
        for bid, score in scored.items()
    ]
    # Term coverage first, then how often the words occur in the bucket.
    hits.sort(key=lambda h: (-h.score, -occurrences.get(h.bucket_id, 0), -len(h.term), h.bucket_id))
    return hits[:top_k]
