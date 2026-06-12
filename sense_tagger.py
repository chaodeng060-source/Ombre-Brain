# ============================================================
# Module: Sense Tagger (sense_tagger.py)
# 模块：五感入口识别器（五感入口层 v1）
#
# Keyword-channel detector: given a piece of memory content (or a breath query),
# return which senses it touches — 嗅觉 / 味觉 / 触觉 / 听觉. Pure & deterministic.
# 关键词通道识别：给一段记忆内容（或 breath query），判它触到哪些感官。纯函数。
#
# Design / 设计：
#   - This is the WRITE-SIDE recognizer only. The "experience, not transcription"
#     framing (哥哥第一人称体验 vs 复读朝灯的话) lives in the twin-side prompt,
#     NOT here — here we just attach a `sense:` label so breath can surface by it.
#     这只是写入侧识别器。"体验而非转录"在 twin 侧 prompt 做，这里只贴 sense 标签。
#   - 视觉 is intentionally OUT in v1 (看/颜色/亮 等太泛、误报高；文档「视觉暂不动」)。
#     视觉 v1 故意不做（词太泛易误报，对齐作战文档）。
#   - Conservative: a sense is tagged only on a real keyword hit, so most buckets
#     get NO sense tag (sense memories are the exception, not the rule).
#     保守：命中真实感官词才打标，绝大多数桶不带 sense（感官记忆是少数）。
#
# Used by: server.py (_merge_or_create write hook + breath query boost).
# ============================================================

# Active senses in v1 (视觉 deferred). 嗅觉 直连海马体——普鲁斯特效应的主角。
SENSES = ("嗅觉", "味觉", "触觉", "听觉")

# Per-sense keyword vocab. Substring match on the lowered text.
# 每感官的关键词表，按子串匹配（文本转小写后）。词选得具体、避免泛词误报。
_SENSE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "嗅觉": (
        "气味", "味道", "闻到", "闻起来", "香味", "香气", "花香", "桂花", "栀子",
        "茉莉", "薄荷", "焦味", "霉味", "腥味", "汗味", "烟味", "香水", "古龙",
        "体香", "奶香", "土腥", "雨后的味道", "好闻", "难闻", "芳香", "馊",
    ),
    "味觉": (
        "甜", "酸", "苦", "咸", "鲜", "涩", "辣", "麻", "藤椒", "花椒", "口感",
        "尝起来", "吃起来", "回甘", "下饭", "咸香", "酸甜", "苦涩", "齁", "腻",
        "味蕾", "舌尖", "入口",
    ),
    "触觉": (
        "摸", "碰", "触", "贴", "蹭", "软", "硬", "烫", "滚烫", "温热", "凉",
        "冰凉", "冰冷", "暖", "滑", "粗糙", "细腻", "痒", "刺痛", "回弹", "硌",
        "质感", "毛茸茸", "湿", "黏", "酥麻", "颤", "手感", "肌肤", "体温",
    ),
    "听觉": (
        "声音", "听到", "听见", "响", "吵", "安静", "寂静", "音乐", "歌声",
        "噪音", "雨声", "风声", "心跳声", "嗡", "叮", "铃", "脚步声", "呼吸声",
        "嘈杂", "悄悄话", "低语", "喘", "哼", "动静",
    ),
}


def detect_senses(text: str) -> list[str]:
    """Return the senses this text touches, in canonical SENSES order. [] if none.
    返回这段文本触到的感官（按 SENSES 固定顺序去重）。没有则空列表。"""
    if not text or not isinstance(text, str):
        return []
    low = text.lower()
    hit = [s for s in SENSES if any(kw in low for kw in _SENSE_KEYWORDS[s])]
    return hit


def normalize_sense_field(value) -> list[str]:
    """Coerce a frontmatter `sense` value (str | list | None) into a clean list.
    把 frontmatter 的 sense 字段（字符串/列表/None）规整成干净的合法列表。"""
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [s for s in SENSES if s in {str(v).strip() for v in value}]


def union_senses(*groups) -> list[str]:
    """Union several sense lists, preserving canonical order. 合并多组 sense，保持固定顺序。"""
    seen: set[str] = set()
    for g in groups:
        for s in normalize_sense_field(g):
            seen.add(s)
    return [s for s in SENSES if s in seen]
