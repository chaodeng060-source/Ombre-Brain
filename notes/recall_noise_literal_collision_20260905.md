# 9/5 字面撞车降噪：生产归因与默认关闭实现

任务：`task_9-5-15-53`。实施小卷，复核/终验 Claude。时间均为北京时间。

## 结论和交付边界

- **未通过降噪验收，不应开启或部署。** 本提交提供默认关闭的泛词闸、同次 DS 调用的提示词修订、测试和诊断工具，不代表生产改善。
- 9 个点名噪音不是同一种来源：3 个由字面分主导但也有向量证据，4 个靠向量分独立过线，2 个由实体通道保送。9 个均无 rare 豁免。
- 12:14 两个实体桶遇到门卫无效响应后回退保留，不是一次有效语义判定的“放行”。15:44 的噪音在当前 BM25 索引文本中没有检索词命中。
- 任务要求的 `13 噪音全部消失 / 42 正确至少保住 40` 目前不可证明。机器窗是 22 个非空 request、52 个注入位置，不是 55；没有完整逐条人工标签。
- 不改变实体通道、DS 错误回退、Twin 查询清洗；这三处超出明确锁定的单一路线，必须另行授权。

## 来源与可复算性

写入前回滚点：`20ac91ca03666c2fb3178b28619a892a8ae3d767`。提交前再次通过 `git ls-remote origin refs/heads/main` 核实。目标远端：`chaodeng060-source/Ombre-Brain` main；未操作生产文件或进程。

生产容器 `ombre-vps-mirror` 的 Docker 原始日志通过已配置 NAS SSH 只读获取：UTC `2026-09-04T14:30:00Z` 至 `2026-09-05T08:00:00Z`。647575 bytes / 4648 行，其中 52 行 Anchor 评分、55 行 breath timing。原始日志 SHA256：`f03bf7071cf9a74bd13074c0481b853a1a0245ba3dd42e626c6bd890f181d412`。

`data/recall_trace.jsonl` 的 `ombre_request_id` 与生产 `breath_timing.request_id` 精确连接；Anchor 行本身没有 request ID，只能以该请求结束时间和 total_ms 界定窗口。本次每个非空请求窗口均找到一行 Anchor 数组；这不是新增的同拍来源追踪字段。

原问句通过 trace 的 SHA256 前 12 位与消息正文匹配，21/22 成功；剩下一条 08:54 是拼接后的自动催办（`stitch=1`），不能手抄成原始问句。18 条请求有 DS query 日志，但日志可能截短，不等于全量原请求。错误回退请求没有成功 DS query 行。trace 自身不保存明文 query。

DF 由当前本地镜像的 `permanent/dynamic/feel` 共 **13914** 桶复算，复用 `BucketManager._load_bucket` 与 `BM25Index._bucket_tokens`，保留完整 frontmatter/body 解析、标题、正文前 1200 字、tags、domain 和现行 tokenizer；未改语料。**这是当前镜像 DF，不是请求发生时的生产 DF。** 生产 timing 四拍分别记录 13900/13905/13907/13907 桶，期间新增导致计数漂移。

原日志、原问句、桶正文和私有账本仅留在 VPS `.work/recall_noise_prod_logs_20260905.gz`、`.work/recall_noise_ledger_20260905.json`（0600），不提交 GitHub。本报告只给请求/桶 ID 和必要分数。

## 四拍 9 个明确噪音的逐条归因

`lit` 是绝对字面证据分，未必等于 BM25 分；`vec` 是原查询向量相似度。Anchor 使用 `0.45 × max(归一化字面, vec, 实体)`，纯字面已有 0.55 上限，conversation 门槛为 0.25。

| 时间 / request_id | bucket_id | lit / vec / Anchor | 过线原因与 DS | 当前镜像 DF |
|---|---|---|---|---|
| 00:33 / `82f1461c344744f7b19ed6428dced491` | `1184e22cbb8f` | 97.8466 / 0.637387 / 0.440310 | 字面主导；向量也足以过线。DS ok 5→5 | 单子 173，老板 34 |
| 同上 | `3b6200416132` | 16.0000 / 0.637938 / 0.287072 | 向量独立过线；不是字面单路。DS ok 5→5 | 单子 173 |
| 同上 | `5c3b5f57b097` | 21.3333 / 0.641598 / 0.288719 | 向量独立过线。DS ok 5→5 | 单子 173 |
| 12:14 / `5e00777f6b2f4c3e823ec6064aa39d42` | `013da98a75e5` | 8.2759 / 0 / 0.450000 | ent=true 实体保送；DS invalid payload → error fallback 5→5 | 原问句中的哥哥 5565；仅原句 DF，非完整规范化查询证明 |
| 同上 | `019af40158f7` | 5.9701 / 0 / 0.450000 | ent=true 实体保送；同次错误回退 | 原问句仅我 5582、和 2770；没有二字实词命中 |
| 13:09 / `7d2f5821aeb147cb896649965d373b58` | `e2e393af5f7e` | 78.5534 / 0.596399 / 0.353490 | 字面主导；向量也足以过线。DS ok 5→5 | 验收 659 |
| 同上 | `22863545f2ec` | 75.6508 / 0.593605 / 0.340429 | 字面主导；向量也足以过线。DS ok 5→5 | 验收 659 |
| 同上 | `b41cd33cae72` | 17.7778 / 0.672842 / 0.302779 | 向量独立过线。DS ok 5→5 | 验收 659 |
| 15:44 / `d9993958bf694d049eb13b1edc96ebd5` | `73b92dcc3d7f` | 8.4211 / 0.670128 / 0.301558 | 向量独立过线；DS ok 5→4 后仍在 Twin ids_out | 当前索引文本对有效查询没有 token 命中；不能编造一个慢/修的 DF 作该桶命中证据 |

00:33 的实际检索仍含“老板打单子”。13:09 的 DS query 已变成“的验收 的验收 的验收 的施工 可以开始认领”，原消息的 84/142/138/135/141/120 已不在其中。只能证明编号到门卫前丢失，未在本单改动清洗器，也不把缺失上下文伪造回回放输入。

剩余 4 条人工账未确认：00:27 的 `1dd4687a4d54`、11:03 三桶中未指名的一条、12:41 的 `277cea91c329`，以及前夜 22:40 的 `b685e9cf6a84` 是待复核线索，**不是本单人工金标**。前夜一条不在今天窗内，不能直接拿来凑 13/55。

## 实现

复用现有 BM25 postings 和 DF，不新增分词器、检索器或模型调用：

1. `literal_term_df_hits` 只对已融合候选返回现成 token 命中与 DF，支持单字；没有 DF 证据时不删除候选。
2. `OMBRE_LITERAL_COLLISION_GUARD_ENABLED=0` 默认关闭。打开后，只有实际 keyword 通道候选、非 rare、非 entity、有高于现有 rare 上限的 DF，且原查询向量低于专用 floor，才使用既有 literal-only cap，不再让弱向量解除此 cap。纯向量候选不经过此闸。
3. `OMBRE_LITERAL_COLLISION_VECTOR_FLOOR=0.71` 是**未校准的试验值，不是验收后的生产阈值**。没有迭代调阈值直至凑过账本。
4. 同一个 DS 调用追加优先级明确的语境规则：共享泛词或同一人物名字不足以判相关；生活/亲密/身体语境下纯工程记录须拒绝，除非明确指向同一任务/事件。关闭开关保持旧 prompt/cache key；不改模型、次数、超时或回退。
5. 开启时 Anchor 诊断行增加 `lit_df/kw/collision`。不修改召回 budget、min_score、PG 查询、RRF 或排序主分。

回滚：保持或恢复 `OMBRE_LITERAL_COLLISION_GUARD_ENABLED=0`；当前没有部署，线上无需执行任何回滚操作。

## 真实数据复算：只报能证明的部分

`tools/replay_literal_collision_guard.py` 对私有账本运行 Anchor 诊断，不调用模型、不访问生产、不重新生成 top：

```sh
python tools/replay_literal_collision_guard.py /opt/claude-twin/.work/recall_noise_ledger_20260905.json
```

52 个真实注入位置有 51 个对应 Anchor 评分；关闭开关复算 **51/51** 与生产记录吻合（误差 ≤0.000001）。一条无对应 Anchor 记录，标为未知，不补造。

历史日志没有 keyword membership，DF 又是后来的镜像，因此诊断必须分别报告 `kw=false/true` 的边界，而非擅自选一个当作真实 ON 结果：

- 00:33 三个噪音在 `kw=true` 时降到 0.2475/0.072/0.096，`kw=false` 不变。
- 13:09 三个噪音在 `kw=true` 时降到 0.2475/0.2475/0.08，`kw=false` 不变。
- 12:14 两个实体噪音两种情况都仍是 0.45。
- 15:44 噪音两种情况都仍是 0.301558；同拍另外两桶在 `kw=true` 时反而落到 0.2475/0.048，说明误杀风险不能忽略。
- 全部 52 个注入中，假设有证据的桶都来自 keyword，最多有 32 个低于门槛。**这不是实际删除数，更不是 32 条噪音**，不能据此打开开关。

未进行新的 DS 成对调用或完整 top 回放，也没有报告“13→0 / 42→40”。缺少完整源查询、实际 keyword 来源、冻结当时语料及人工标签时，模型重跑不能填平这些证据缺口。

## 真实注入账：22 request / 52 位置

以下均为修前 Twin `ids_out`，不是修后结果。重复桶按注入位置计数，不把候选池当最终注入。

| 时间 | request_id | ids_out |
|---|---|---|
| 00:27:30 | `ba441b1272c04983b5d58b449b5504d2` | `130953c74a10`, `1dd4687a4d54` |
| 00:32:05 | `522a895ba6ce4855882f28f4121887ef` | `de1dd9925d29`, `dde4868d8a47`, `83ab95052f98` |
| 00:33:32 | `82f1461c344744f7b19ed6428dced491` | `1184e22cbb8f`, `3b6200416132`, `5c3b5f57b097` |
| 01:26:01 | `d4e1c4a832d74b1fbfaf5ab2fa01e77c` | `ccf7275d25fa`, `9c31cf1a29f6` |
| 01:37:35 | `8d3debaf71d741b09767daaba0657a1f` | `8c6669d3534f`, `16f6fb6f70a5`, `a834457f8297` |
| 08:54:04 | `7fa0ff67ca954ae58750978fb53699e0` | `3d873cbdd3c5` |
| 08:55:58 | `7897a75aff194cfd9caaf266ad5f6b48` | `ca346188fd0c`, `30ffdf5c0149`, `f86b8691d5bd` |
| 10:43:31 | `f13c1bf529744c7f954057a81e7d122b` | `c703a0d06946` |
| 10:49:25 | `7a40ea9c1b394bb38dae9b637f0e11c4` | `f86b8691d5bd`, `983ec6172e01`, `40a8c0bd09fb` |
| 11:02:37 | `138ae29d39f1448ba510cd1951bfcc4b` | `cc1b0d0649b0`, `ae0c01b0f7ee`, `4bec31272bfb` |
| 11:03:26 | `8911a11a618540fe8bc742393f355f15` | `a7ff337dff30`, `7a4fd9b8a53f`, `3f957c6e364c` |
| 11:27:00 | `53aa2ba716a14a269e4900943f02c7ef` | `b061b70d26a6` |
| 12:14:27 | `5e00777f6b2f4c3e823ec6064aa39d42` | `013da98a75e5`, `019af40158f7` |
| 12:35:51 | `f3b4236bbbbf4d7bae76e3a8fef4611c` | `710be680f019`, `db382cc729a7` |
| 12:39:07 | `1577417e2d124405b4ac89a4ddba9c94` | `430863a4c10a`, `0d42d2ee37f9` |
| 12:41:00 | `a65b669be36e49baab6ca9e2d7f07082` | `277cea91c329`, `ae72a42f5657`, `17dccb320df1` |
| 12:42:40 | `844e6108dc444c228f0b4120569330a8` | `1cadaeded2bf`, `6cc5995aea84`, `b93a91c1c911` |
| 13:09:07 | `7d2f5821aeb147cb896649965d373b58` | `e2e393af5f7e`, `22863545f2ec`, `b41cd33cae72` |
| 13:44:22 | `b6ad0f33e7984325b303bbf999d1327a` | `588d818b2005`, `9a30690e7c2e` |
| 15:37:30 | `c83423ce73e840ffad06655344309af5` | `4e111721b21a`, `ea081b3d04b5` |
| 15:37:53 | `51188f4334c04d568a5d8c8b25a14da8` | `7772214257c4`, `1b76801c50f8` |
| 15:44:00 | `d9993958bf694d049eb13b1edc96ebd5` | `114ffffb0da3`, `2ee2f3279dec`, `73b92dcc3d7f` |

## 测试与复核交接

144 项相关测试通过，包含真实 breath 函数的 ON/OFF 接线测试、DF、单字、rare/entity/纯向量豁免、DS prompt/cache、现有 Anchor、实体、时间线、只读脱水、状态与权威过滤、BM25 热修复。

```sh
env -u TWIN_E_CHORD_SHADOW_ENABLED -u TWIN_E_CHORD_RECALL_ENABLED -u TWIN_RECALL_NAVIGATION_ENABLED \
  /opt/claude-twin/.venv/bin/python -m pytest -q \
  tests/test_literal_collision_guard.py tests/test_anchor_rare_literal_exempt.py \
  tests/test_anchor_literal_only_cap.py tests/test_ds_filter_cache.py \
  tests/test_pr1_noise_tools.py tests/test_relevance_first_ranking.py \
  tests/test_entity_recall.py tests/test_timeline_breath.py \
  tests/test_breath_dehydration_read_only.py tests/test_state_aware_recall.py \
  tests/test_recall_authority.py tests/test_live_reconciliation_hotfixes.py
```

`test_anchor_quality_gate_uses_adapted_absolute_score` 在未改的 `20ac91c` 同样失败：旧断言要求纯字面 60 分桶通过，已与已有 literal-only cap 冲突。本提交只修正该断言并明确验证 0.2475，不放宽生产门槛。未宣称全仓全绿或真实质量达标。

下一步由 Claude 在任务卡补齐逐请求的人工标签与 55 条口径，并复核是否另开实体误匹配、DS 错误回退、编号清洗修复。未授权前不在此单绕开这些边界。任务保持未验收；不做三次无依据调阈值，不转速度优化。
