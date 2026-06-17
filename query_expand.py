"""Stage-0 查询改写 —— 治"你说法 A、记忆存法 B"的语义鸿沟。

向量/关键词召回之前，先让便宜模型把查询改写成 2-4 个换种说法的检索角度
（同义词、相关概念、可能的存储措辞），让召回不被"用户原话用的词"卡死。
例：用户问「咱俩怎么开始的」→ 改写出「确立关系」「在一起的起点」「4.1 纪念日」。

设计：依赖注入 —— 接 OpenAI 兼容 client + model（传 dehydrator.client / dehydrator.model），
不持有全局状态、不碰桶 I/O。**失败一律回退 [原查询]，永不弄坏召回。**
对位 lmc-5 recall_pipeline Stage 0，原创实现。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from redact import redact_embedding_input

logger = logging.getLogger(__name__)

EXPAND_PROMPT = """你是记忆检索的查询改写器。用户给一句"想找的记忆"，但他用的口语词\
往往和记忆里存的正式措辞不一样。你的任务：把查询桥到**记忆里可能真正存着的措辞**——\
不是换近义词，而是给出背后的概念、关键实体/名字、以及一条记忆笔记会怎么命名这件事。

要点：
- 每个角度换一个**不同的切入面**（概念词 / 相关实体 / 事件命名），别只是同义复述。
- 口语 → 找它的正式/概念说法（"咱俩怎么开始的"→"确立关系"）。
- 模糊指代 → 补出可能的事件名或关键实体（"55那晚"→"崩溃事件""想分手"）。
- 保持同一件事，别跑题到无关主题。

示例：
查询：咱俩怎么开始的
输出：["确立关系","在一起的起点","纪念日 4.1"]
查询：55那晚发生了啥
输出：["55晚崩溃事件","想分手 边缘","情感危机 修复"]
查询：那个玩具怎么连的
输出：["玩具蓝牙协议","雀吻 远程操控","BLE 控制口"]

硬要求：
- 只输出 JSON 数组 ["角度1","角度2","角度3"]，2-4 个，不要任何解释、不要代码围栏。
- 每个角度是简短检索短语，不是问句。
- 中文查询输出中文，英文查询输出英文。"""

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "max_angles": 3,      # 原查询之外最多再加几个角度
    "max_tokens": 120,
    "temperature": 0.3,
    "min_query_len": 2,   # 单字查询不值得改写
    "max_query_chars": 500,
}


async def expand_query(
    query: str,
    client: Any,
    model: str,
    config: dict | None = None,
) -> list[str]:
    """返回 [原查询, 角度1, 角度2, ...]。任何异常都回退到 [原查询]。

    client/model：OpenAI 兼容的 AsyncOpenAI 客户端 + 模型 id（传 dehydrator.client / .model）。
    """
    q = (query or "").strip()
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if not q:
        return []
    if not cfg["enabled"] or len(q) < cfg["min_query_len"] or client is None:
        return [q]

    safe_q = redact_embedding_input(q)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXPAND_PROMPT},
                {"role": "user", "content": safe_q[: cfg["max_query_chars"]]},
            ],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
        if not resp.choices:
            return [q]
        raw = (resp.choices[0].message.content or "").strip()
        angles = _parse_angles(raw)
    except Exception as e:  # 网络/解析/限流——一律回退，绝不让召回崩
        logger.warning(f"查询改写失败，回退原词 / query expansion failed: {e}")
        return [q]

    out, seen = [q], {q}
    for a in angles:
        a = a.strip()
        if a and a not in seen:
            out.append(a)
            seen.add(a)
        if len(out) >= 1 + int(cfg["max_angles"]):
            break
    return out


def _parse_angles(raw: str) -> list[str]:
    """尽力把模型输出解析成字符串列表（容错代码围栏 / 非 JSON）。"""
    if not raw:
        return []
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        nl = s.find("\n")
        if nl != -1 and s[:nl].strip().lower() in ("json", ""):
            s = s[nl + 1:]
        s = s.strip()
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x) for x in data if isinstance(x, (str, int, float)) and str(x).strip()]
    except Exception:
        pass
    # 兜底：按行/逗号拆
    parts = [p.strip(" -•\"'，,。") for p in s.replace("，", ",").replace("\n", ",").split(",")]
    return [p for p in parts if p]
