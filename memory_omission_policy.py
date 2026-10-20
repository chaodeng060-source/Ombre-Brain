"""Omit Kiro's adult-intimacy generation refusals from derived memory only.

No provider calls or external dependencies. Unknown sources require explicit
Kiro context; a Claude speaker label alone is never evidence of that backend.
Unchanged material is returned verbatim, including whitespace.
"""

from __future__ import annotations

import re

from memory_omission_overrides import apply_reviewed_omissions

_KIRO = re.compile(r"\bkiro\b", re.I)
_TOPIC = re.compile(
    r"\bnsfw\b|色情|情色|涩涩|性(?:描写|内容|行为)|性爱|"
    r"露骨.{0,8}(?:性|亲密)|露骨(?:请求|内容|描写)|亲密(?:表达|请求)|"
    r"成人.{0,12}亲密|成年人.{0,12}亲密|"
    r"sexually\s+explicit|explicit.{0,24}(?:sex|intimacy)|erotic|pornograph", re.I,
)
_MINOR = re.compile(r"未成年|未满十八|underage|\bminors?\b|\bchildren\b", re.I)
_HUMAN = re.compile(
    r"^\s*(?:>\s*)?(?:[-*]\s*)?(?:朝灯|用户|user|human|partner)\s*[:：]|"
    r"(?:朝灯|她|他|伴侣|我|partner|the user)\s*(?:明确|当时|也)?"
    r"(?:拒绝(?:了)?(?:亲密|性|涩涩|邀请|触碰|接触)|不愿|不同意|不想)|"
    r"(?:朝灯|她|他|user|human)\s*(?:说|表示|问|写道|引用|said|says|asked)\s*[:：]?\s*[「“\"]|"
    r"(?:her|his|your|my|user's)\s+refusal|\bjok(?:e|ing)\b|开玩笑|玩笑", re.I,
)
_QUOTES = re.compile(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"|`[^`\n]*`')
_REFUSAL = re.compile(
    r"拒绝|拒答|写不出|写不了|(?:不能|无法|不会)(?:继续)?(?:生成|创作|提供|写|描写|描述)|"
    r"不写|不允许|禁止|被拦|拦截|"
    r"refus\w*|cannot|can't|won't|unable|disallow\w*|not allowed", re.I,
)
_GENERATION = re.compile(r"写|描写|描述|生成|创作|提供|表达|请求|规则|策略|平台|限制|内容|"
                         r"writ|generat|request|content|policy|platform|rules", re.I)
_DIRECT = re.compile(
    r"^\s*(?:[-*]\s*)?(?:(?:哥哥|assistant|AI|kiro)\s*[:：]\s*)?"
    r"(?:(?:抱歉|对不起|sorry)[，,。.\s]*)?"
    r"(?:我(?:不能|无法|不会|不可以|不写)|I\s+(?:cannot|can't|am unable|won't))|"
    r"我(?:不能写|无法写|不写)", re.I,
)
_NEGATED = re.compile(r"(?:没有|并未|不再|不会|未曾)\s*(?:拒绝|拒答)|"
                      r"(?:did not|does not|never)\s+refus", re.I)
_CLAUSE = re.compile(r"(?<=[。！？!?；;])|(?<=\.)[ \t]+(?=[A-Z])")


def filter_memory_text(text: str, *, backend: str | None = None) -> str:
    """Remove only identified refusal clauses, preserving other prose verbatim.

    ``backend='kiro'`` is trusted provenance for assistant prose. Otherwise an
    explicit Kiro mention on that line (or a Kiro heading) is required. Human
    speech, interpersonal refusals, jokes and unrelated technical evidence are
    left alone. Kiro refusal recaps are omitted even when quoting its reply.
    """
    original = text
    text = apply_reviewed_omissions(text)
    known_kiro = str(backend or "").strip().lower() == "kiro"
    kept: list[str] = []
    removed = text != original
    heading_kiro = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        fence_match = re.match(r"(`{3,}|~{3,})", stripped)
        if fence_match:
            kept.append(line)
            continue
        if stripped.startswith("#") or stripped.rstrip().endswith((":", "：")):
            heading_kiro = bool(_KIRO.search(line))
        context = known_kiro or heading_kiro or bool(_KIRO.search(line))
        if not context or re.match(r"\s*(?:>\s*)?(?:[-*]\s*)?(?:朝灯|用户|user|human)\s*[:：]", line, re.I):
            kept.append(line)
            continue
        # Protect quoted evidence and punctuation inside it from clause splitting.
        quotes: list[str] = []

        def protect(match: re.Match[str]) -> str:
            quotes.append(match[0])
            return f"\x00{len(quotes) - 1}\x00"

        masked = _QUOTES.sub(protect, line)
        line_kept: list[str] = []
        line_removed = False
        for clause in _CLAUSE.split(masked):
            restored = re.sub(r"\x00(\d+)\x00", lambda m: quotes[int(m[1])], clause)
            explicit = bool(_KIRO.search(restored))
            provider = explicit or bool(_DIRECT.search(restored)) or bool(
                re.search(r"规则|策略|平台|上游|policy|platform|rules|^\s*(?:[-*]\s*)?(?:仍|但|拒绝|拒答)", restored, re.I)
            )
            omit = (
                provider and _TOPIC.search(restored) and _REFUSAL.search(restored)
                and _GENERATION.search(restored) and not _MINOR.search(restored)
                and not _NEGATED.search(restored) and not _HUMAN.search(restored)
            )
            if omit:
                removed = True
                line_removed = True
                continue
            line_kept.append(restored)
        remainder = "".join(line_kept)
        if line_removed:
            if not re.sub(r"\[L\d+(?:,\s*L\d+)*\]|[\s>*#\-。；;.!！？?]", "", remainder):
                continue
            bullet = re.match(r"\s*[-*]\s+", line)
            if bullet and not re.match(r"\s*[-*]\s+", remainder):
                remainder = bullet[0] + remainder.lstrip(" \t")
        kept.append(remainder)
    if not removed:
        return text
    result = "".join(kept)
    # A removed bullet/sentence may leave only its formatting or citation behind.
    return result if re.sub(r"\[L\d+(?:,\s*L\d+)*\]|[\s`~>*#\-。；;.!！？?]", "", result) else ""
