"""Keep summary facts bound to the supplied source without extra model calls."""
from __future__ import annotations

import json

MEMORY_PROSE_RULES = (
    '\n记忆正文和摘要只描述原话支持的事实、感受和注明说话者的计划。'
    '缺失或无依据的说法直接省略，不附加来源审校、证据不足或完成状态核验的解释。'
    '这类审校解释只能放在记忆之外的独立诊断/审计信息中，不进入本次记忆输出的任何字段。'
    '片段未提及某事，不代表从未承诺、没有感情或事情没有完成。'
    '若原对话本就在讨论审校，按原说话者归属保留实际说过的审校内容。\n'
)

SUMMARY_FACT_RULES = (
    '\n事实保真：姓名、动物名、物种、颜色、数字、说话者与动作归属只能取自本次原话。'
    '同名对象不自动合并；狗不能写成猫；提议、推测不能写成既成事实。'
    '摘要的每条事实直接对照原话，不以旧摘要或改写正文作为独立证据。'
    '没有依据的具体属性省略，不补猜，不编造情绪需求。\n'
) + MEMORY_PROSE_RULES


def guarded_summary(summary: str, source: str, *, editorial_only: bool = False) -> str:
    """Return supported output, or a literal source excerpt on a detected drift.

    This deterministic check detects bounded classes of factual drift, not full
    semantic entailment. The backfill's independent original-text review is
    responsible for the latter. A cheap model's bad answer must not cause a
    second paid call or prevent a conversation from continuing.

    Existing cache returns use ``editorial_only`` to reject added audit notes
    without adding animal/name/number gates. Unusable source still falls back
    literally because it cannot support a generated audit explanation.
    """
    from memory_source_fidelity import validate_derivative

    try:
        parsed = json.loads(summary)
    except (TypeError, ValueError):
        parsed = summary
    if isinstance(parsed, dict):
        keys = (
            'core_facts', 'summary', 'keywords', 'todos', 'sample_voice',
            'emotion_state', 'body_signal', 'unspoken_need', 'sore_point',
            'response_rule', 'do_not',
        )
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
    if editorial_only:
        issues = [issue for issue in issues if issue in {
            'unsupported_editorial_note', 'source_missing', 'source_incomplete',
        }]
    if not issues:
        return summary
    excerpt = source.strip()[:400]
    return json.dumps({'core_facts': [excerpt], 'summary': excerpt[:120]},
                      ensure_ascii=False, separators=(',', ':'))
