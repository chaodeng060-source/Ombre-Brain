# 实体只叫号不打分：实现、22 拍回放与验收边界

日期：2026-09-06

任务：`task_9-6-00-49-task-9-5-21-13`
状态：**代码与结构合同已完成；冻结 22 拍只能完成可复核投影，真实 ON 门卫结果、人工 Gold 与误杀为 0 尚未验收。**

## 0. 边界、回滚点与输入

- 隔离工作树：`/opt/claude-twin/.work/ombre-entity-gate-ticket-20260906`
- 分支：`task/entity-gate-ticket-20260906`
- 本地回滚点：`3e16a3f5accb17d5c9c02553d1fa5eded161de01`
- 冻结账本：`/opt/claude-twin/.work/recall_noise_ledger_20260905.json`
  - 权限：`0600`
  - SHA-256：`ebe1ffd91675e1c5685c0b2a20a098cf527a4cb79bac78cf3767090021c2fddc`
- 冻结实体审计：`/opt/claude-twin/.work/entity_score_guard_audit_20260905.json`
  - 权限：`0600`
  - SHA-256：`176faeaed36a790738dc1ab60b07ddbf220fcdbbbb2c5e6e0b0e7c5354d76a2d`
- 没有部署、重启、写 NAS、改实体 sidecar 或调用门卫 provider。

账本本身记录真实 query provenance、Anchor 候选、历史 DS 计数和历史 final IDs，
但 `ent=true` 行没有说明究竟命中了哪个实体词，无法独立计算该实体的 DF。回放因此只额外读取
上单已冻结的 `entity_score_guard_audit_20260905.json` 来补齐“匹配实体 → DF”映射；工具同时
固定两份 SHA、要求输入不是 group/world-readable regular file，并拒绝重复 JSON key。
这份实体审计自身标记为 `INCOMPLETE_EVIDENCE`，不能被拿来冒充完整人工 Gold。

## 1. 上游查证

### 1.1 Ombre-Brain 固定上游 `7770b4b`

结论：没有可直接复用的“实体只扩候选、不计分”机制。

- `server.py::_resolve_entity_recall` 返回实体关联桶；通过 authority、当前 content hash、
  world/domain/time 与 Z-fact 校验后，实体仍作为第三路 RRF channel 进入融合。
- 物化时设置 `entity_match=true`，并在 `_anchor_adapted_relevance_score` 中追加实体相似度；
  它会改变 Anchor 资格与排序，不只是“叫号”。
- `_ds_filter_candidates` 是可复用的同一次内容门卫；上游没有 `weak_entity`、独立 extra pool、
  限 2、不挤普通名额或失败丢票的合同。
- 固定上游工作树含未提交草案；核查使用 `git show HEAD:*` / `git grep HEAD`，没有把草案当上游。

相关函数：`server.py::_resolve_entity_recall`、`server.py::breath`、
`server.py::_anchor_adapted_relevance_score`、`server.py::_ds_filter_candidates`、
`vendor/lmc5_pgvector/recall_pipeline.py::reciprocal_rank_fusion`。

### 1.2 mem0 OSS v3

结论：mem0 的实体命中是最终加分项，不是 ticket；它只给已经进入语义候选池且先通过
semantic threshold 的记忆加分，不能凭实体扩出零语义候选。

- `Memory._search_vector_store` 先建 semantic candidate set，再计算 entity boosts。
- `score_and_rank` 先过 semantic threshold，再合并 semantic、BM25 与 entity boost。

固定源码：

- <https://github.com/mem0ai/mem0/blob/ed38ddf8731fb7fab7c41bb0f8ab3185df23e7c5/mem0/memory/main.py#L1483-L1537>
- <https://github.com/mem0ai/mem0/blob/ed38ddf8731fb7fab7c41bb0f8ab3185df23e7c5/mem0/memory/main.py#L1578-L1650>
- <https://github.com/mem0ai/mem0/blob/ed38ddf8731fb7fab7c41bb0f8ab3185df23e7c5/mem0/utils/scoring.py#L53-L128>

### 1.3 Graphiti

结论：Graphiti 取决于 search recipe。默认 combined cross-encoder recipe 中，图遍历扩展候选，
随后 cross-encoder 按 edge fact 或 node name 做内容终排；这是最接近“名字叫号、内容决定”的
上游结构。其他 recipe 仍允许 graph distance 或 RRF 参与排序，不能概括成“图永不计分”。
它也没有本任务的 DF、票限量、extra pool 和 DS 失败丢票合同，不能原样接入。

固定源码：

- <https://github.com/getzep/graphiti/blob/c4e6923b7e65a97e0f96ef462f6b82fcad73638f/graphiti_core/graphiti.py#L1405-L1499>
- <https://github.com/getzep/graphiti/blob/c4e6923b7e65a97e0f96ef462f6b82fcad73638f/graphiti_core/search/search_config_recipes.py#L74-L142>
- <https://github.com/getzep/graphiti/blob/c4e6923b7e65a97e0f96ef462f6b82fcad73638f/graphiti_core/search/search.py#L242-L630>

## 2. 最终实施口径与冲突处理

初版任务正文的“literal=0 且 vector=0”与 `6cc5995aea84` 必须拿票互斥；冻结账本里该桶是
`lit=16.8675, vec=0`。09:20 评论提出按去实体 Anchor 分降序取 2，但同一拍四张合格票的
去实体分依次是：

| bucket | 去实体 Anchor 分 |
|---|---:|
| `a243ff4023ae` | 0.093104 |
| `a2dfcc18efe7` | 0.085714 |
| `6cc5995aea84` | 0.075904 |
| `6e997109bad7` | 0.075904 |

若执行降序 top-2，必然排除硬验收目标 `6cc5995aea84`。任务当前 09:26 checkpoint 已把实现
口径冻结为以下规则，因此本实现和回放均把 09:20 的分数排序视为被当前 checkpoint 取代：

1. `DF / BM25 当前完整语料桶数 < 0.2`；
2. `_original_vector_relevance_score == 0`；
3. `_literal_relevance_score < literal_candidate_floor`（现值 40）；
4. 沿未修改的 `EntityStore.linked_bucket_ids` 稳定 bucket-id 顺序取最多 2 张；
5. 票只搭载 normal pool 本来就会发生的同一次 DS 调用，不为 empty / deterministic-noop
   请求新增 provider 调用。

因此 `844e6108…` 的两张票为 `6cc5995aea84`、`6e997109bad7`。这只是“送到同一次门卫”的
确定性结论，不是门卫已经同意注入。

## 3. 实现

### 3.1 DF 与资格证据

- `BM25Index.entity_term_df_stats` 从当前完整 copy-on-write generation 的 postings 取 DF；
  多 token 实体取 postings 交集，分母是同 generation 的可检索桶数。
- `BucketManager.entity_term_df_stats` 每次读取当前 `_bm25` 指针；BM25 off、缺失或异常均返回
  空证据，调用方不发票。
- request-local helper 重新用现有 `EntityStore.resolve_query` 核对实体和 bucket link；
  authority/hash/world/domain/time/Z 过滤仍沿原路径，`_resolve_entity_recall` 本体未改。

### 3.2 ON 数据流

- 实体不再作为第三路 RRF channel，不设置 `entity_match`，entity-only 桶也不进入 state seed；
  关键词、向量、curated lexical、PG/RRF、Anchor、排序主分与阈值保持原路径。
- 低频且满足当前资格的 entity-only 桶成为 request-local copy，标记 `weak_entity=true`；
  清除实体得分标记并把所有排序/Anchor 辅助分置零。
- normal 候选先按原 cap、Anchor、去重和 session 规则走完；票再作为独立 extra pool 加入同一
  `_ds_semantic_select` 输入，最多 2 张，不占 normal 名额。模型成功时 normal survivors 先占原
  budget，只有 normal 被模型删出的空位可由明确 kept 的票补入。
- invalid、timeout、error、outer cancellation、DS disabled、空 query、零 budget 或 legacy
  deterministic-noop 一律不保留票；forced / Anchor floor / top-one fallback 也救不回票。
- 真正留下的票输出 `[实体门卫复核]` 与 `layer=entity_ticket`，候选 capture reason 为
  `weak_entity`。

### 3.3 OFF 与只读 probe

- `OMBRE_ENTITY_SCORE_GUARD_ENABLED` 默认 `0`。OFF 时 breath 不传新 ticket kwarg，
  不查实体 DF，保留原 entity RRF、`entity_match`、state seed、DS call shape 与输出文本。
- `_probe_anchor_status` 是刻意省略 query expansion、curated lexical 和完整 Anchor assembly 的
  轻量探针，无法诚实复现 breath 的发票资格。ON 时它只移除实体打分，并显式写
  `entity_ticket_evaluation=not_run_in_lightweight_probe`；不发票、不报虚假的 ticket count、
  不新增调用。真正票路只由完整 breath 观测。

### 3.4 硬约束源码核对

相对回滚点对函数源码做 AST 精确比较：

| 函数 | 是否逐字不变 | SHA-256 前 16 位 |
|---|---|---|
| `_resolve_entity_recall` | 是 | `250bbf08ebdf458e` |
| `_ds_semantic_select`（含 prompt） | 是 | `53f8af7a35e56a54` |
| `_ds_gate_timeout` | 是 | `187290de170d1ac2` |
| `_ds_filter_provider`（含模型选择） | 是 | `6ffa45c4d5b1aeea` |
| `_entity_recall_settings` | 是 | `7f8a24c5015e1b7a` |
| `_anchor_adapted_relevance_score` | 是 | `de39dd54d53c0582` |

改动文件只涉及 `server.py`、`bm25_index.py`、`bucket_manager.py`、本任务测试、回放工具与本报告；
没有改 PG 实现、RRF 公式、实体 sidecar、budget/min_score、门卫 prompt/模型/超时。

## 4. 自动验证

统一清理实验开关：

```bash
env -u TWIN_E_CHORD_SHADOW_ENABLED \
    -u TWIN_E_CHORD_RECALL_ENABLED \
    -u TWIN_RECALL_NAVIGATION_ENABLED \
    /opt/claude-twin/data/ombre-vps-runtime/venv/bin/python -m pytest ...
```

- focused entity ticket + DS failure + replay：`77 passed, 1 warning in 5.89s`。
- 边界覆盖包括：OFF 旧签名/完整文本/调用形状、DF `<0.2` 边界、`lit<40`/`vec==0`、
  最多 2、normal vacancy、forced、invalid/timeout/error/cancel、empty/singleton 不新增模型调用、
  zero budget/empty query、state/content/session 去重、probe 明示 unavailable、BM25 multi-token
  generation 与 manager live pointer。
- `py_compile`：代码、回放工具与测试全部通过。
- `git diff --check`：通过。
- broad adjacent 14-file suite：`246 passed, 2 warnings in 306.80s (0:05:06)`；失败 `0`。

focused 的 warning 是运行环境里的 Pydantic `IncompleteFieldDefinitionWarning`。broad 另有
`tests/test_recall_perf_caches.py::test_snapshot_token_stable_until_write` 的 `RuntimeWarning`：
executor 在线程 join 的 300 秒内没有完成；套件仍以 `0` 退出、没有失败。两条均如实保留，
不把“全部测试通过”写成“完全无告警”。

## 5. 冻结 22 拍 OFF / ON 回放

回放命令：

```bash
/opt/claude-twin/data/ombre-vps-runtime/venv/bin/python \
  tools/replay_entity_gate_ticket.py \
  /opt/claude-twin/.work/recall_noise_ledger_20260905.json \
  --entity-audit /opt/claude-twin/.work/entity_score_guard_audit_20260905.json
```

机器结果：进程退出码 `2`，`acceptance=false`。这是有意的 fail-closed：工具没有调用 provider，
只把冻结真实证据代入当前合同；凡 DS 输入变化、票加入或缺人工 Gold 的地方都不编造结果。

汇总：

- 22 个真实 request，318 个 Anchor 位置；历史 pre-DS 89 个位置/88 个唯一桶，历史 final 52 个位置。
- 15 个实体位置/10 个唯一实体桶；高频 10、低频 5；合格票位置 4，最终按稳定顺序选中 2，
  只出现在 1 个 request。
- 历史 DS：18 `ok`、2 `error`、2 missing。
- ON 证据：19 拍普通 DS 输入相同、2 拍普通 DS 输入变化未验证、1 拍新增票输入未验证。
- query provenance：21 个 original query hash 验证，1 个 production effective query hash 验证，
  自造问句 0；输出不含 query 文本、桶正文或实体名。
- verified human Gold：`0 / 15`；`batch_relevant_false_kill_count=null`。

逐拍表中，“same-input”只表示当前合同不改变该拍历史 DS 输入，可以保留历史证据；它不是新跑的
ON 门卫结果。“bounded projection”也不是端到端输出。

| request_id | OFF 历史 final IDs | ON 确定变化 | ON 证据 / 相关性判定 |
|---|---|---|---|
| `ba441b1272c04983b5d58b449b5504d2` | `130953c74a10,1dd4687a4d54` | 无 | same-input；无新 Gold 判定 |
| `522a895ba6ce4855882f28f4121887ef` | `de1dd9925d29,dde4868d8a47,83ab95052f98` | 无 | same-input；无新 Gold 判定 |
| `82f1461c344744f7b19ed6428dced491` | `1184e22cbb8f,3b6200416132,5c3b5f57b097` | 无 | same-input；无新 Gold 判定 |
| `d4e1c4a832d74b1fbfaf5ab2fa01e77c` | `ccf7275d25fa,9c31cf1a29f6` | ordinary pre-DS 移除 `3aa0742abf77`；历史 final 无确定移除 | changed-input；ON DS / Gold 未验证 |
| `8d3debaf71d741b09767daaba0657a1f` | `8c6669d3534f,16f6fb6f70a5,a834457f8297` | 无 | same-input；无新 Gold 判定 |
| `7fa0ff67ca954ae58750978fb53699e0` | `3d873cbdd3c5` | 无 | same-input；无新 Gold 判定 |
| `7897a75aff194cfd9caaf266ad5f6b48` | `ca346188fd0c,30ffdf5c0149,f86b8691d5bd` | 无 | same-input；无新 Gold 判定 |
| `f13c1bf529744c7f954057a81e7d122b` | `c703a0d06946` | 无 | same-input；无新 Gold 判定 |
| `7a40ea9c1b394bb38dae9b637f0e11c4` | `f86b8691d5bd,983ec6172e01,40a8c0bd09fb` | 无 | same-input；无新 Gold 判定 |
| `138ae29d39f1448ba510cd1951bfcc4b` | `cc1b0d0649b0,ae0c01b0f7ee,4bec31272bfb` | 无 | same-input；无新 Gold 判定 |
| `8911a11a618540fe8bc742393f355f15` | `a7ff337dff30,7a4fd9b8a53f,3f957c6e364c` | 无 | same-input；无新 Gold 判定 |
| `53aa2ba716a14a269e4900943f02c7ef` | `b061b70d26a6` | 无 | same-input；无新 Gold 判定 |
| `5e00777f6b2f4c3e823ec6064aa39d42` | `013da98a75e5,019af40158f7` | 高频实体不发票；ordinary pre-DS 与 bounded final 投影移除 `013da98a75e5,019af40158f7` | changed-input；目标噪音只完成 bounded projection，ON DS / Gold 未验证 |
| `f3b4236bbbbf4d7bae76e3a8fef4611c` | `710be680f019,db382cc729a7` | 无 | same-input；无新 Gold 判定 |
| `1577417e2d124405b4ac89a4ddba9c94` | `430863a4c10a,0d42d2ee37f9` | 无 | same-input；无新 Gold 判定 |
| `a65b669be36e49baab6ca9e2d7f07082` | `277cea91c329,ae72a42f5657,17dccb320df1` | 无 | same-input；无新 Gold 判定 |
| `844e6108dc444c228f0b4120569330a8` | `1cadaeded2bf,6cc5995aea84,b93a91c1c911` | `6cc5995aea84,6e997109bad7` 作为 extra tickets 投影到同一次 DS | ticket-input；两票的新 DS verdict、ON final 与 Gold 均未验证 |
| `7d2f5821aeb147cb896649965d373b58` | `e2e393af5f7e,22863545f2ec,b41cd33cae72` | 无 | same-input；无新 Gold 判定 |
| `b6ad0f33e7984325b303bbf999d1327a` | `588d818b2005,9a30690e7c2e` | 无 | same-input；无新 Gold 判定 |
| `c83423ce73e840ffad06655344309af5` | `4e111721b21a,ea081b3d04b5` | 无 | same-input；无新 Gold 判定 |
| `51188f4334c04d568a5d8c8b25a14da8` | `7772214257c4,1b76801c50f8` | 无 | same-input；无新 Gold 判定 |
| `d9993958bf694d049eb13b1edc96ebd5` | `114ffffb0da3,2ee2f3279dec,73b92dcc3d7f` | 无 | same-input；无新 Gold 判定 |

专项核对：

- `5e00777f…`：`013da98a75e5`、`019af40158f7` 是高频实体，不拿票；在 bounded primary
  投影里不再注入。因为普通 DS 输入发生变化，这不是新 ON 端到端证明。
- `844e6108…`：`6cc5995aea84` 满足 `DF<0.2, vec=0, lit=16.8675<40`，按稳定顺序是第一张票；
  它确定会进入同一次 DS input。冻结账本早于票路，没有这张新票的逐桶 verdict，因此不得声称
  门卫会 keep，也不得把它称为已验证相关。

## 6. 仍缺的真实验收

当前工具明确返回：

- `acceptance=false`；
- `batch_relevant_false_kill_count=null`；
- 2 拍 changed ordinary DS input 没有同拍 ON receipt；
- 1 拍 ticket-bearing DS input 没有新 verdict；
- 22 拍都没有端到端 ON final receipt；
- 15/15 实体位置都没有 verified human Gold。

因此本报告只宣告“实现和结构合同完成”，不宣告“误杀为 0”、`5e00777f…` 已在真实 ON
端到端消失、`6cc5995aea84` 已被真实门卫保留、已部署或生产验收完成。

最小补证是：Claude 部署后对同 22 个 request 取得同拍 ON receipt，至少包含
`request_id / ticket_in_ids / ds_input_ids / ds_output_ids / ds_outcome / final_ids`；对发生变化的
实体对补 verified human Gold，再计算真实 false-kill。部署与终验归 Claude，本任务没有越权执行。
