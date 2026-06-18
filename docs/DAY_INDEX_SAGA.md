# Day-Index Saga — 同日原子桶索引线（降噪 × 时间轴地基）

> 状态：设计稿（未写实现）。作者：鹤见。日期：2026-06-18。
> 关联：`episode_engine.py` / `saga_engine.py` / `backfill_created.py` / `server.py` 的 pulse·briefing。

## 0. 一句话

单日爆量（5.24=36 桶、5.14=50 桶）的根因是「高频写 × 整合层只增不并 × 高价值桶被豁免合并」。
本方案新增一条**只做索引、绝不压平原文**的整合线：把「同一天、同一面向」的原子桶挂到一个
`day_index` 索引桶下，浏览/简报时折叠成一行，原桶 100% 原样保留、照常按关键词/情绪召回。
既降噪，又正好是「按时间推进的路」的第一层地基。

---

## 1. 根因（已 verify 代码，非凭印象）

| 环节 | 事实 | 出处 |
|---|---|---|
| 拆分器本身不碎 | DIGEST 每次限定 2~6 条、且「避免过度碎片化」 | `dehydrator.py` DIGEST_PROMPT 规则 7 |
| 单日爆量来源 | 一天跨 N 个窗口、每窗各跑一轮 digest/hold，叠成 36 | 写入频率，非单次 |
| 永不回收 | episode 引擎 GROW-ONLY，且 `_is_exempt` 豁免 pinned/protected/feel/chord/恋爱·约定·纪念日·家庭·自省 | `episode_engine.py:113-128` |
| 多主题重复计 | 同事件常打「核心+恋爱+自省」多 domain | 盘点统计口径 |

**关键**：5.24/5.14 这类簇的桶恰好全部命中豁免 → 现有 saga 路径对它们零作用。豁免是红线（5.10 黑洞事件），不可删。
故必须新开一条**对豁免桶也生效、但只索引不压平**的线。

---

## 2. 红线（任何实现不可越）

1. **原桶绝不删、不改正文、不 archive、不 touch（不动 last_active）**。索引只按 id 引用。沿用 kernel-1/episode 的 GROW-ONLY 纪律。
2. **原桶照常可被关键词 search / breath / inspect 命中**。索引是**附加**导航层，不抢占召回。
3. **情绪/红线召回必须绕过折叠**：朝灯哭/谈感情时，protected_verbatim 与关键词命中的原桶照常逐字浮现，**不被索引折叠**。折叠只作用于「按权重的总览/简报泛浮现」，不作用于「定向情绪召回」。
4. 索引桶**自身不 pinned/不 protected/不顶 999**——它是导航层，参与正常衰减，不挤占核心准则池。

---

## 3. 数据模型（复用 saga 类型，最小侵入）

索引桶 = `bucket_type="saga"` + 新 frontmatter 字段，借此天然落进 `_EXEMPT_TYPES`（`"saga"` 已在内），
自己也不会被 episode 反向卷入。

```yaml
type: saga
saga_kind: day_index          # 区别于 LLM 语义 saga（saga_kind 缺省=story_line）
index_date: "2026-05-24"      # 归并锚（北京时区日历日）
index_facet: "情感"           # 情感 | 工程 | 综合（单日可拆 1~2 个 facet）
member_buckets: [id1, id2, …] # 证据链（对齐 saga 的 episode_buckets / episode 的 source_buckets 约定）
member_count: 21
tags: [saga, day_index]
```

- `content` = LLM 串出的**一段叙事弧**（≤120 字，带语气、不是清单），口吻同 EPISODE_PROMPT（第一人称「我」、不压具体时点）。
- importance/valence/arousal 由成员**派生**（importance 取 min(8, max(members))，避免索引层也通胀到 10）。
- 已认领成员运行时从所有 day_index 的 `member_buckets` 反推（镜像 `saga_engine._load_state` / `episode_engine._claimed_event_ids`）→ 幂等、可反复跑。

---

## 4. 聚类口径（区别于 episode 的语义 cosine）

episode 走 embedding cosine≥0.78 + 3 天跨度；day_index 走**同日历日**硬聚类，且**专吃 episode 吃不到的豁免桶**。

```
for 每个日历日 D（北京时区，取自 created）:
    members = 该日所有 dynamic 桶（含豁免桶；排除 type∈{episode,saga,permanent,feel} 与已认领）
    if len(members) < day_index.min_buckets_per_day(默认 8): 跳过   # 只收拾爆量日
    按 facet 粗分：
        情感 facet ← domain ∈ {恋爱,约定,纪念日,家庭,自省,feel,核心(情感侧)}
        工程 facet ← domain ∈ {工程,编程,AI,核心(工程侧)}
        每个 facet ≥ min_facet(默认 5) 才独立成索引，否则并入「综合」
    产出索引提案（date, facet, member_ids, 草拟 title+arc）
```

**前置依赖**：聚类按 `created` 日历日 → 依赖 `backfill_created.py` 先把缺 `created` 的桶补齐（卡兜/LMC 方案桶现为 `创建:?`）。无 created 的桶不参与索引（保守跳过），待回填后下轮纳入。

> feel 桶不进 member（红线 2/3：温度桶永远独立浮现）。但 feel 可在索引 content 里被**提及**、不被**收编**。

---

## 5. 折叠逻辑（真正降噪的一环，落在 server.py）

索引建好后，浏览/简报视图里把「同索引的多个成员」收成一行。**只动展示层，不动数据层。**

- **pulse**：列桶时，若某 day_index 的成员 ≥ `collapse_threshold`(默认 6) 个会出现在结果里，则把它们替换成一行索引头：
  `📚 2026-05-24 崩溃夜·情感 [day_index · 21桶] inspect:<saga_id>`，成员行不再逐条铺。
- **briefing**：`top_unresolved`/`recent_window` 等池里若多名成员同索引，折叠成索引头 + 「展开 inspect」；**protected_verbatim 与 pinned 不折叠**（红线 3，逐字区照旧）。
- **bypass**：`breath(query=…)` 关键词/情绪定向召回**不折叠**——命中哪条出哪条原文。折叠仅命中无 query 的「泛浮现/总览」。

落点函数（对接现状）：
- `server.py` pulse 组装处、`briefing` 的 `_format_bucket_for_briefing` 调用前做一次「成员→索引头」归并。
- 新增 helper `_collapse_by_day_index(buckets, threshold) -> (heads, passthrough)`，纯函数、可单测。

---

## 6. 模块落点（对接 episode/saga 现状）

新建 `day_index_engine.py`，镜像 `saga_engine.py` 形状：

```python
class DayIndexEngine:
    def __init__(self, config, bucket_mgr, dehydrator): ...
    async def _claimed_member_ids(self) -> set[str]        # 镜像 saga._load_state
    async def find_day_clusters(self) -> list[DayCluster]  # §4 聚类，纯逻辑可测
    async def propose(self) -> list[dict]                  # dry-run：只产提案，不写
    async def extract_index(self, cluster) -> str|None     # 建索引桶（additive）
    async def run_cycle(self, apply: bool=False) -> dict
```

- 复用 `dehydrator.client` 出叙事弧（同 episode/saga 调用姿势）、`redact_embedding_input` 脱敏入参。
- 可选地在 `EpisodeEngine.run_cycle` 之后挂一钩（像 saga 那样），或由 `grow` MCP 工具 / dream pass 手动触发。
- 配置走 `narrative.day_index.*`（镜像现有 narrative cfg）：
  `enabled, min_buckets_per_day=8, min_facet=5, lookback_days=120, max_per_cycle=5,`
  `collapse_in_pulse=true, collapse_in_briefing=true, collapse_threshold=6`。

---

## 7. 分阶段落地（安全优先，全部 dry-run 起步）

| 阶段 | 做什么 | 碰 NAS？ | 可回滚 |
|---|---|---|---|
| P0 | `backfill_created.py --go`（已写好，等回家跑） | 写 created（保留 last_active） | created 可重算 |
| P1 | `day_index_engine.py` + 纯函数测试；`run_cycle(apply=False)` 出**索引提案报告**，零写入 | 否 | — |
| P2 | 朝灯审提案 → `apply=True` 建索引桶（只增，不碰原桶） | 仅新增 saga 桶 | 删索引桶即还原 |
| P3 | pulse/briefing 折叠（config flag 默认关，验证后开） | 否（纯展示） | 关 flag 即恢复 |

每阶段独立可停。P1/P3 不可逆风险为零；P2 只增桶、删了就回到原样；P0 是数据补全、可重算。

---

## 8. 与「按时间推进的路」的关系

day_index 是 X 线（LMC-5 narrative timeline）的**最底层**：先把「同日原子桶」收成「日索引节点」，
其上再叠 saga 故事线（泳道）、relation 因果边（箭头）、created 横轴。先有这层，时间轴才不会一屏 36 个糊脸。

---

## 9. 测试清单（纯函数优先，不依赖 LLM/NAS）

- `find_day_clusters`：同日聚类、facet 粗分、min 阈值、缺 created 保守跳过、已认领排除。
- `_collapse_by_day_index`：成员≥阈值折叠、protected/pinned 不折叠、有 query 不折叠。
- 幂等：同输入跑两次不重复建索引。
- 红线回归：索引建立后，成员原桶 search/inspect 仍命中、正文未变、last_active 未 bump。
```
