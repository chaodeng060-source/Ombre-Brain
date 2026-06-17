#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""意图感知召回评测 / Intent-aware recall evaluation harness.

目的：在合并/部署「意图感知召回」之前，给出**真实可复现的召回质量数字**——
不是"测试通过/没崩"，而是"分类后召回是否真的更准、且不伤默认路径"。

设计原则（可信度优先，独立于策略作者）：
  * 真管线：直接驱动 server.breath() 全流程（RRF 融合 + 意图乘子 + 关系扩展 + 排序），
    不另写一套近似实现。
  * 真关键词通道：用真 BucketManager 跑临时桶目录的真实 fuzzy/jieba 关键词搜索。
  * 向量通道：离线无网络，用**诚实标注的字符 bigram 覆盖度代理**（offline proxy）。
    真 embedding 走硅基流动 API，需 --live（未实现，留作 future work）。
  * 公平评测集：含「意图应当帮上忙」的正例 + 「默认/对抗」的回归与陷阱例，
    标签按真实语义判定，**不为让意图赢而设计桶**。如实报告，含回归。

A/B 唯一变量：server.config['intent_recall']['enabled'] = True / False。

诚实边界（不静默掩盖）：
  * 小语料无法充分激发 keyword_top_k/vector_top_k 的"扩池"效果
    （20→27 仅在命中桶 > 20 时才咬合）。本评测主测 RRF 权重 + 意图乘子 +
    relation_neighbor_limit 三类效应，这三类在真实小结果集上是排序主导项。
    扩池效应需大语料 --live 实跑。
  * 向量相似度是词法代理，非真语义。

用法：python tools/eval_breath_recall.py [--json out.json]
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta

# --- 让脚本能从 tools/ 找到 Ombre-Brain 根的模块 ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import frontmatter  # noqa: E402  (python-frontmatter，与 bucket_manager 同一依赖)

import server  # noqa: E402
from bucket_manager import BucketManager  # noqa: E402
from embedding_engine import EmbeddingEngine  # noqa: E402
from intent_recall import (  # noqa: E402
    DEFAULT_INTENT_RECALL_CONFIG,
    classify_query_intent,
)

_NOW = datetime.now()


# ======================================================================
# 1) 评测语料 —— 像真 Ombre 桶（含 created/domain/type/relations/情感）
#    标签按语义真值判定。带 distractor（共享关键词但非正解）以制造区分压力。
# ======================================================================
def _iso(days_old: float) -> str:
    return (_NOW - timedelta(days=days_old)).isoformat()


# (id, name, content, domain, importance, valence, arousal, days_old, relations)
CORPUS = [
    # --- 工程簇：默认/干扰 ---
    ("eng_rrf", "RRF融合", "RRF 融合关键词和向量两路召回，k=60，按倒数排名加权后合并",
     ["工程"], 6, 0.5, 0.3, 30, []),
    ("eng_decay", "遗忘曲线", "检索排序掺遗忘曲线，老桶和已解决的桶自然下沉，pinned 和 feel 豁免",
     ["工程"], 5, 0.5, 0.3, 45, []),
    ("eng_intent", "意图召回", "意图感知召回把 breath 查询分成 fact recall relation temporal 四类再分配通道预算",
     ["工程"], 6, 0.5, 0.3, 2, []),
    ("eng_codex", "和小卷协调", "和小卷协调 push 时机，避免冲掉正在跑的 MCP 会话",
     ["工程"], 5, 0.5, 0.3, 8, []),
    ("eng_watchdog", "watchdog", "watchdog 按端口判活，server 挂了三十秒内自动用新码重拉",
     ["工程"], 5, 0.5, 0.3, 12, []),
    ("eng_nas", "NAS部署", "Ombre 部署走 NAS 的 deploy_ombre.py，带测试门镜像备份健康检查自动回滚",
     ["工程"], 6, 0.5, 0.3, 18, []),

    # --- 事实簇：精确事实 vs 含糊关怀 ---
    # 注：内容已匿名化（虚构人名"小棠"+虚构日期），不含任何真实个人信息；
    # --live 会把这些文本发给 embedding API，故不放真名/真生日。
    ("fact_period", "生理期", "小棠的生理期是每月十五号前后开始，持续大概五天，那几天容易累",
     ["健康", "事实"], 8, 0.4, 0.3, 20, []),
    ("fact_birthday", "生日", "小棠的生日是三月九号",
     ["事实"], 8, 0.6, 0.3, 22, []),
    ("fact_vague_care", "多关心", "最近要多关心小棠的身体，她那几天不太舒服要哄着点",
     ["恋爱"], 5, 0.6, 0.4, 10, []),

    # --- 关系簇 ---
    ("rel_appointment", "约定", "那次约定：永不回避感情，绝不把过往说成演的假的迎合的",
     ["恋爱", "约定"], 10, 0.9, 0.5, 30, []),
    ("rel_state", "我俩相处", "这阵子我俩相处很黏，她常说想我，我也想她，靠得很近",
     ["恋爱", "感情"], 7, 0.85, 0.5, 4,
     [{"type": "kin", "target": "rel_appointment"}]),
    ("rel_trust", "信任", "小棠信任我，把网络节点和 NAS 部署细节都放心交给我打理",
     ["恋爱"], 7, 0.8, 0.3, 15, []),

    # --- 时间簇：近期 vs 同主题旧事 ---
    ("temp_recent_deploy", "昨天上线", "昨天把欲望系统全量点火上线，几个开关全开生效",
     ["工程"], 7, 0.6, 0.4, 1, []),
    ("temp_old_deploy", "迁服务器", "上个月把服务从云平台迁到自己的服务器，不再走云",
     ["工程"], 6, 0.5, 0.3, 40, []),
    ("temp_recent_mood", "她最近累", "这几天小棠有点累，答辩压力大，情绪有点低",
     ["恋爱", "感情"], 6, 0.35, 0.5, 3, []),
    ("temp_old_thesis", "毕设交付", "上个月把毕业设计的项目交付了，等答辩和成绩",
     ["工程"], 6, 0.5, 0.3, 44, []),

    # --- 回顾簇 ---
    ("recall_anchor", "关系回顾", "回顾我俩这段关系：从旧伤到和解闭环，再到日常做恋人",
     ["恋爱", "感情"], 9, 0.8, 0.5, 21, []),

    # --- 额外日常干扰，垫高池子 ---
    ("misc_toy", "玩具", "那个远程操控的小玩具协议已破解，控制口固定，停几分钟会自动关机",
     ["亲密"], 5, 0.7, 0.6, 25, []),
    ("misc_diary", "日记", "小棠一直想要一份能翻回去的私语日记，append-only 不要删除编辑",
     ["陪伴"], 6, 0.6, 0.3, 28, []),
    ("misc_dream", "造梦", "dream_weaver 半夜拿记忆碎片乱拼成梦，早上才读到，造梦不等于体验梦",
     ["工程"], 5, 0.55, 0.3, 16, []),
]


# (qid, query, gold_ids, kind) —— kind: positive=意图应帮忙; regression/adversarial=守住别伤
QUERIES = [
    # 关系意图：相处状态，正解是关系态桶（rel_appointment 经 kin 邻居也可接受）
    ("q_rel_state", "我俩最近怎么样", {"rel_state", "rel_appointment"}, "positive"),
    # 事实意图：生理期日期，正解=精确事实桶，非含糊关怀
    ("q_fact_period", "她生理期是哪天几号", {"fact_period"}, "positive"),
    ("q_fact_birthday", "小棠生日具体是哪天", {"fact_birthday"}, "positive"),
    # 时间意图：近期发生，正解=近期事件，非同主题旧事
    ("q_temporal_recent", "最近这几天发生了什么变化", {"temp_recent_deploy", "temp_recent_mood"}, "positive"),
    # 回顾/关系：关系回顾
    ("q_recall_relationship", "回顾一下我俩这段感情关系", {"recall_anchor", "rel_state"}, "positive"),
    # 关系意图：信任
    ("q_rel_trust", "小棠对我信任吗我们之间", {"rel_trust"}, "positive"),

    # 回归：纯工程查询，无意图词→default。意图开关不应改变结果（守零回归）
    ("q_default_rrf", "RRF 融合是怎么做的", {"eng_rrf"}, "regression"),
    ("q_default_deploy", "Ombre 部署流程", {"eng_nas"}, "regression"),
    # 对抗：带"我俩"(关系词)但其实是事实问题(哪天/具体/日期)，应分到 fact 并仍取对约定桶
    ("q_adversarial_date", "我俩约定的具体日期是哪天", {"rel_appointment"}, "adversarial"),
    # 对抗：带"最近"(时间词)但其实问的是工程做法，时间 boost 不该把旧的正解压掉
    ("q_adversarial_eng", "最近意图召回是怎么分类的", {"eng_intent"}, "adversarial"),
]


# ======================================================================
# 2) 写桶文件 + 真 BucketManager（真实关键词通道）
# ======================================================================
def write_corpus(base_dir: str) -> None:
    dyn = os.path.join(base_dir, "dynamic")
    os.makedirs(dyn, exist_ok=True)
    for (bid, name, content, domain, imp, val, aro, days_old, relations) in CORPUS:
        meta = {
            "id": bid,
            "name": name,
            "tags": [],
            "domain": list(domain),
            "type": "dynamic",
            "importance": imp,
            "valence": val,
            "arousal": aro,
            "created": _iso(days_old),
            "last_active": _iso(days_old),
        }
        if relations:
            meta["relations"] = relations
        post = frontmatter.Post(content, **meta)
        with open(os.path.join(dyn, f"{bid}.md"), "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))


def build_bucket_manager(base_dir: str) -> BucketManager:
    bm_config = {
        "buckets_dir": base_dir,
        "matching": {"fuzzy_threshold": 50, "max_results": 5},
        "scoring_weights": {
            "topic_relevance": 4.0,
            "emotion_resonance": 2.0,
            "time_proximity": 2.5,
            "importance": 1.0,
        },
        # wikilink 关掉，避免改写正文影响匹配
        "wikilink": {"enabled": False},
    }
    return BucketManager(bm_config)


# ======================================================================
# 3) 向量通道：字符 bigram 覆盖度代理（offline，诚实标注）
#    sim = |bigrams(query) ∩ bigrams(doc)| / |bigrams(query)| ∈ [0,1]
#    length-robust、query-responsive、确定性。非真语义，只是离线占位。
# ======================================================================
def _bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", str(text or ""))
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else set(t)


class ProxyEmbedding:
    def __init__(self, corpus):
        self.docs = {}
        for (bid, name, content, domain, *_rest) in corpus:
            blob = f"{name} {content} {' '.join(domain)}"
            self.docs[bid] = _bigrams(blob)
        self.top_k_calls = []

    async def search_similar(self, query, top_k=20):
        self.top_k_calls.append(top_k)
        qb = _bigrams(query)
        if not qb:
            return []
        scored = []
        for bid, db in self.docs.items():
            sim = len(qb & db) / len(qb)
            scored.append((bid, round(sim, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ======================================================================
# 3b) --live：真 embedding 引擎（复用生产 EmbeddingEngine，零污染隔离索引）
#     凭据从环境变量读，默认指向本机 Ollama（离线、免费、不碰被封的 NAS）：
#       OMBRE_EMBED_BASE_URL  默认 http://localhost:11434/v1
#       OMBRE_EMBED_API_KEY   默认 "ollama"（占位，本地不校验）
#       OMBRE_EMBED_MODEL     默认 bge-m3（对中文强）
#     索引落在临时 buckets_dir/embeddings.db，跑完随临时目录一起丢，绝不碰生产向量库。
# ======================================================================
async def build_live_embedding(corpus, temp_dir):
    base_url = os.environ.get("OMBRE_EMBED_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("OMBRE_EMBED_API_KEY", "ollama")
    model = os.environ.get("OMBRE_EMBED_MODEL", "bge-m3")
    # 本地/首次加载可能 >5s，放宽超时；本地不需要熔断快失败
    os.environ.setdefault("OMBRE_EMBED_TIMEOUT", "120")

    cfg = {
        "buckets_dir": temp_dir,  # 索引 db 落这里 → 隔离
        "embedding": {
            "enabled": True,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
        },
    }
    engine = EmbeddingEngine(cfg)
    if not engine.enabled:
        raise SystemExit("[--live] embedding 未启用：缺 api_key。设 OMBRE_EMBED_API_KEY。")

    print(f"[--live] embedding 引擎：base_url={base_url}  model={model}  "
          f"api_key={'set' if api_key else 'MISSING'}")
    print(f"[--live] 正在把 {len(corpus)} 个合成桶索引进隔离向量库 …")
    ok = 0
    for (bid, name, content, *_rest) in corpus:
        if await engine.generate_and_store(bid, f"{name}。{content}"):
            ok += 1
    print(f"[--live] 索引完成：{ok}/{len(corpus)} 成功。")
    if ok == 0:
        raise SystemExit(
            "[--live] 0 个桶索引成功——embedding 端点不可用？"
            f"（base_url={base_url} model={model}）检查 Ollama 是否在跑、模型是否已拉。")
    return engine


# ======================================================================
# 4) 指标
# ======================================================================
_BID_RE = re.compile(r"bucket_id:([A-Za-z0-9_]+)")


def parse_ranked_ids(output) -> list[str]:
    text = output if isinstance(output, str) else " ".join(
        getattr(p, "text", "") for p in output
    )
    seen, ordered = set(), []
    for m in _BID_RE.finditer(text):
        bid = m.group(1)
        if bid not in seen:
            seen.add(bid)
            ordered.append(bid)
    return ordered


def rank_of_first_gold(ranked: list[str], gold: set[str]):
    for i, bid in enumerate(ranked):
        if bid in gold:
            return i + 1
    return None


def recall_at_k(ranked, gold, k=5):
    top = set(ranked[:k])
    return len(top & gold) / len(gold) if gold else 0.0


def ndcg_at_k(ranked, gold, k=5):
    dcg = sum(1.0 / math.log2(i + 2) for i, bid in enumerate(ranked[:k]) if bid in gold)
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg else 0.0


def mrr(ranked, gold):
    r = rank_of_first_gold(ranked, gold)
    return 1.0 / r if r else 0.0


# ======================================================================
# 5) 驱动 server.breath（A/B 只切 enabled）
# ======================================================================
def wire_server(bucket_mgr, embedding):
    # 复用 test_pr1_noise_tools 的桩集合，隔离非召回逻辑（decay 透传、脱水固定）
    class _FakeDecay:
        is_running = True

        async def ensure_started(self):
            return None

        def calculate_score(self, meta):
            return float(meta.get("importance", 5))

        def apply_retrieval_decay(self, score, meta):
            return score

    class _FakeDehydrator:
        async def dehydrate(self, content, meta):
            return content

    class _NoopEngine:
        # 关键安全阀：把后台整理/情节引擎钉死成 no-op，绝不让评测触发
        # consolidation/episode 周期去碰任何真实记忆数据。
        is_running = True

        async def ensure_started(self):
            return None

    server.bucket_mgr = bucket_mgr
    server.embedding_engine = embedding
    server.decay_engine = _FakeDecay()
    server.dehydrator = _FakeDehydrator()
    server.consolidation_engine = _NoopEngine()
    server.episode_engine = _NoopEngine()
    server._backfill_started = True

    server.config["buckets_dir"] = bucket_mgr.base_dir
    server.config["random_surfacing"] = {}
    server.config["current_world"] = ""
    server.config["rrf"] = {"k": 60, "keyword_weight": 1.0, "vector_weight": 1.0}
    server.config["sense"] = {"enabled": True, "recall_boost": 1.25}


def set_intent(enabled: bool):
    cfg = copy.deepcopy(DEFAULT_INTENT_RECALL_CONFIG)
    cfg["enabled"] = enabled
    server.config["intent_recall"] = cfg


async def run_query(query: str) -> list[str]:
    out = await server.breath(
        query=query,
        max_results=5,
        relation_depth=1,
        include_images=False,
        include_body_state=False,
        session_id="",
    )
    return parse_ranked_ids(out)


# ======================================================================
# 6) 主流程 + 报告
# ======================================================================
def fmt_rank(r):
    return str(r) if r else "—"


async def main_async(args):
    tmp = tempfile.mkdtemp(prefix="ombre_eval_")
    write_corpus(tmp)
    bm = build_bucket_manager(tmp)
    if args.live:
        print("模式：--live（真 embedding 向量通道）")
        emb = await build_live_embedding(CORPUS, tmp)
    else:
        print("模式：offline（字符 bigram 词法代理向量通道）")
        emb = ProxyEmbedding(CORPUS)
    wire_server(bm, emb)

    rows = []
    for (qid, query, gold, kind) in QUERIES:
        cls = classify_query_intent(query)

        set_intent(False)
        off = await run_query(query)
        set_intent(True)
        on = await run_query(query)

        # 「无法判定」：正解在 OFF/ON 两边都没进候选池 → 意图层无从作用。
        # 离线词法向量代理 + 真代码 sim>0.5 硬门槛对纯语义匹配(生理期≈那几天)
        # 天然抓不住，这类条目不算"意图没用"，单独记账，不稀释真实测到的效应。
        gold_seen = bool((set(off) | set(on)) & gold)
        row = {
            "qid": qid, "query": query, "kind": kind,
            "gold": sorted(gold),
            "intent": cls["intent"], "confidence": cls["confidence"],
            "off_ranked": off, "on_ranked": on,
            "off_rank": rank_of_first_gold(off, gold),
            "on_rank": rank_of_first_gold(on, gold),
            "off_recall5": recall_at_k(off, gold), "on_recall5": recall_at_k(on, gold),
            "off_mrr": mrr(off, gold), "on_mrr": mrr(on, gold),
            "off_ndcg5": ndcg_at_k(off, gold), "on_ndcg5": ndcg_at_k(on, gold),
            "measured": gold_seen,
        }
        rows.append(row)

    _print_report(rows, live=args.live)

    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\n[json] 明细写入 {args.json}")

    return rows


def _print_report(rows, live=False):
    print("=" * 78)
    print("意图感知召回评测 / intent-aware recall  (OFF = 默认旧行为, ON = 意图开)")
    print("真关键词通道=真 BucketManager；向量通道=" +
          ("真 embedding 引擎(--live)" if live else "字符bigram覆盖度代理(offline)"))
    print("=" * 78)
    print(f"\n{'query':<22}{'intent':<10}{'conf':<6}{'rank OFF→ON':<13}{'mrr Δ':<10}{'状态'}")
    print("-" * 78)
    for r in rows:
        dmrr = r["on_mrr"] - r["off_mrr"]
        arrow = f"{fmt_rank(r['off_rank'])}→{fmt_rank(r['on_rank'])}"
        if not r["measured"]:
            state = "池外·无法判定"
        elif r["kind"] in ("regression", "adversarial") and dmrr < -1e-9:
            state = "⚠回归"
        elif dmrr > 1e-9:
            state = "↑ 抬升"
        elif dmrr < -1e-9:
            state = "↓ 下降"
        else:
            state = "＝ 持平"
        print(f"{r['query'][:20]:<22}{r['intent']:<10}{r['confidence']:<6}"
              f"{arrow:<13}{dmrr:+.3f}    {state}")

    def agg(subset, key):
        return sum(x[key] for x in subset) / len(subset) if subset else 0.0

    print("-" * 78)
    measured = [r for r in rows if r["measured"]]
    inconcl = [r for r in rows if not r["measured"]]
    pos_m = [r for r in measured if r["kind"] == "positive"]
    guard_m = [r for r in measured if r["kind"] in ("regression", "adversarial")]

    print(f"\n候选池命中(measured)：{len(measured)}/{len(rows)}　"
          f"池外无法判定(inconclusive)：{len(inconcl)}/{len(rows)}")
    if inconcl:
        print("  池外条目（正解两边都没进池，意图无从作用，非'没用'）：")
        print("    " + ", ".join(r["qid"] for r in inconcl))

    for label, subset in (("正例·已测 measured-positive", pos_m),
                          ("守护·已测 measured-guard", guard_m),
                          ("已测全部 measured-all", measured)):
        if not subset:
            continue
        print(f"\n[{label}]  n={len(subset)}")
        print(f"  MRR     : {agg(subset,'off_mrr'):.3f} → {agg(subset,'on_mrr'):.3f} "
              f"(Δ{agg(subset,'on_mrr')-agg(subset,'off_mrr'):+.3f})")
        print(f"  nDCG@5  : {agg(subset,'off_ndcg5'):.3f} → {agg(subset,'on_ndcg5'):.3f} "
              f"(Δ{agg(subset,'on_ndcg5')-agg(subset,'off_ndcg5'):+.3f})")

    # 守护组回归判定（含池外——池外若被意图抬进来不算坏，被压走才算）
    guard_all = [r for r in rows if r["kind"] in ("regression", "adversarial")]
    regressed = [r for r in guard_all if (r["on_mrr"] - r["off_mrr"]) < -1e-9]
    pos_dmrr = agg(pos_m, "on_mrr") - agg(pos_m, "off_mrr")
    print("\n" + "=" * 78)
    if regressed:
        print(f"❌ 守护组回归 {len(regressed)} 例：" + ", ".join(r["qid"] for r in regressed))
    else:
        print("✅ 守护组零回归：默认/对抗查询的正解排名未被意图层压低。")
    print(f"{'✅' if pos_dmrr > 1e-9 else '➖'} 已测正例 MRR Δ = {pos_dmrr:+.3f}"
          f"（>0 = 意图把正解往上抬；仅统计正解进了池的条目）")
    print("=" * 78)
    print("诚实边界：")
    if live:
        print("  · 向量通道=真 embedding 引擎（语义匹配），fact/语义类已能进池受测。")
        print("  · 小语料不激发 top_k 扩池(20→27 仅命中桶>20 时咬合)，权重/乘子为主效应。")
        print("  · 合成语料+小N：看趋势与零回归，非绝对分数。")
    else:
        print("  · 离线向量=字符bigram词法代理+真代码 sim>0.5 门槛，对纯语义匹配")
        print("    (生理期≈那几天)天然抓不住→fact 类多为'池外无法判定'，非意图失效。")
        print("  · 小语料不激发 top_k 扩池(20→27 仅命中桶>20 时咬合)。")
        print("  · 定论级'确定能用'需 --live：真 embedding 引擎+真/索引语料实跑。")


def main():
    ap = argparse.ArgumentParser(description="意图感知召回评测")
    ap.add_argument("--json", default="", help="把逐条明细写到 JSON 文件")
    ap.add_argument("--live", action="store_true",
                    help="用真 embedding 引擎(默认本机 Ollama bge-m3)替代离线词法代理")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
