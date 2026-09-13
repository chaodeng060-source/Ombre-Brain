"""写入侧人名闸：桶里冒出户口本之外的人名，挂进待审队列等人过目。

治的是「婷易」那类病 —— 模型整理记忆时凭空编一个人名，写进桶就成了「记忆」，
几个月后朝灯偶然翻到才发现。9/9 那次全靠她自己看见，不能再指望这个。

不发明新机制，用仓里现成的两个零件：
  · ``entity_store``  —— 模块开头写死「an LLM is never allowed to invent aliases」，
    由 config.yaml 的 ``entities.seeds`` 显式播种，这就是户口本。
  · ``review_queue``  —— 铁律「机器只入队、不落库」，pending 等人显式 resolve。

**闸不拦写入**。桶照常落地，只多挂一条待审。误报的代价是我多看一眼，不是丢记忆。

判据为什么是双链而不是分词：容器里实测 ``jieba.posseg`` 把「婷易」切成
``婷/x`` + ``易/a`` 两个碎片，把「朝灯」「小卷」标成普通名词 ``n`` —— 没有户口本时
分词对咱家的人名基本失灵，靠词性根本抓不到当年那个幻觉。而 dehydrator 的整理模板
本来就要求给人名/专名打 ``[[双链]]``，9/9 查到的原文正是 ``[[婷易]]在15:02提到``：
**模型自己标出了「这是个专名」，这个信号零猜测，比任何词性推断都硬。**

抽名字的原则抄 2026-07-28 眠和Opia《语法召回实现教程》第三、四节（朝灯给的）：
先分词/看标记再判，绝不做 n-gram 枚举 —— 「们现」那种碎渣根本不该出生；
户口本上的人不看运行时工牌，工牌只审陌生人。
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

# dehydrator 打的双链：[[名字]]，也兼容 [[名字|显示文本]]
_WIKILINK_RE = re.compile(r"\[\[([^\]\n|]{1,40})(?:\|[^\]\n]{0,40})?\]\]")

# 代词被打成双链是 dehydrator 的另一个毛病（全库 [[我]] 95 次、[[她]] 11 次），
# 跟人名幻觉不是同一件事，不在这道闸里刷屏。
_PRONOUNS = frozenset("我你他她它咱谁人")

# 技术名/项目名/文件名不是人名幻觉。含 ASCII 字母数字或这些符号的一律放行，
# 避免 mcp_excalidraw、lmc5:night、Fable 5.1 这类每写一个桶就挂一条。
_NON_PERSON_CHARS = re.compile(r"[A-Za-z0-9_:./\-#@]")

# 人名极少超过 6 个汉字；更长的是事件名/短语（「记忆系统大修收官」这种）。
_MAX_PERSON_CHARS = 6


def _normalize(value: str) -> str:
    return str(value or "").strip()


def roster_from_config(entities_cfg: Mapping | None) -> frozenset[str]:
    """户口本 = entities.seeds 的正名与别名 + entities.known_links 白名单。

    seeds 同时喂给 ``entity_store``，两处共用一份名单，不各记各的。
    """
    cfg = entities_cfg or {}
    roster: set[str] = set()

    seeds = cfg.get("seeds") or []
    items: Iterable = seeds.values() if isinstance(seeds, Mapping) else seeds
    if isinstance(seeds, Mapping):
        roster.update(_normalize(k) for k in seeds)
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = _normalize(item.get("canonical_name") or item.get("canonical"))
        if name:
            roster.add(name)
        aliases = item.get("aliases") or ()
        if isinstance(aliases, str):
            aliases = [aliases]
        if isinstance(aliases, Sequence):
            roster.update(_normalize(a) for a in aliases if _normalize(a))

    known = cfg.get("known_links") or ()
    if isinstance(known, str):
        known = [known]
    if isinstance(known, Sequence):
        roster.update(_normalize(k) for k in known if _normalize(k))

    roster.discard("")
    return frozenset(roster)


_ROSTER_CACHE: frozenset[str] | None = None


def default_roster() -> frozenset[str]:
    """进程内缓存一次的户口本。改了 config.yaml 要重启才生效，跟别的配置一样。"""
    global _ROSTER_CACHE
    if _ROSTER_CACHE is None:
        try:
            from utils import load_config

            _ROSTER_CACHE = roster_from_config((load_config() or {}).get("entities"))
        except Exception:
            # 配置读不出来时闸自己闭嘴，绝不影响桶写入。
            _ROSTER_CACHE = frozenset()
    return _ROSTER_CACHE


def mentions_needing_review(content: str) -> tuple[str, ...]:
    """写入路径用的一行入口：正文里打了双链、又不在户口本上的疑似人名。

    户口本为空（没播种）时返回空 —— 没配名单就等于这道闸没开，
    与 ``entities.seeds`` 缺省时 entity_store 保持零 sidecar 行为一致。
    """
    roster = default_roster()
    if not roster:
        return ()
    return unknown_person_mentions(content, roster)


def unknown_person_mentions(content: str, roster: Iterable[str]) -> tuple[str, ...]:
    """返回正文里打了双链、却不在户口本上的疑似人名，按出现顺序去重。"""
    known = {_normalize(r) for r in roster or ()}
    seen: list[str] = []
    for raw in _WIKILINK_RE.findall(str(content or "")):
        name = _normalize(raw)
        if not name or name in known or name in seen:
            continue
        if len(name) == 1 and name in _PRONOUNS:
            continue
        if _NON_PERSON_CHARS.search(name):
            continue
        if len(name) > _MAX_PERSON_CHARS:
            continue
        seen.append(name)
    return tuple(seen)
