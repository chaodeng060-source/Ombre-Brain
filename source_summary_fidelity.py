"""Keep summary facts bound to the supplied source without extra model calls."""
from __future__ import annotations

import json

SUMMARY_FACT_RULES = (
    '\n事实保真：姓名、动物名、物种、颜色、数字、说话者与动作归属只能取自本次原话。'
    '同名对象不自动合并；狗不能写成猫；提议、推测不能写成既成事实。'
    '摘要的每条事实直接对照原话，不以旧摘要或改写正文作为独立证据。'
    '没有依据的具体属性省略，不补猜，不编造情绪需求。\n'
)


def guarded_summary(summary: str, source: str) -> str:
    """Return supported output, or a literal source excerpt on a detected drift.

    This deterministic check detects bounded classes of factual drift, not full
    semantic entailment. The backfill's independent original-text review is
    responsible for the latter. A cheap model's bad answer must not cause a
    second paid call or prevent a conversation from continuing.
    """
    from memory_source_fidelity import validate_derivative

    try:
        parsed = json.loads(summary)
    except (TypeError, ValueError):
        parsed = summary
    if isinstance(parsed, dict):
        keys = ('core_facts', 'summary', 'keywords', 'todos', 'sample_voice')
        claims = '\n'.join(
            item
            for key in keys
            for item in (parsed.get(key, []) if isinstance(parsed.get(key), list)
                         else [parsed.get(key, '')])
            if isinstance(item, str)
        )
    else:
        claims = str(parsed)
    issues = validate_derivative(claims, [source])
    if not issues:
        return summary
    excerpt = source.strip()[:400]
    return json.dumps({'core_facts': [excerpt], 'summary': excerpt[:120]},
                      ensure_ascii=False, separators=(',', ':'))
