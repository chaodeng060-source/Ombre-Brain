# 9/5 DS 门卫失败不再全放行：生产归因、保守回退与真实请求回放

任务：`task_9-5-17-31`。实施小卷；复核、部署和生产终验归 Claude。时间均为北京时间，除非明确写 UTC。

## 结论与边界

- 根因已钉死：旧 `_ds_filter_candidates` 把 invalid、内层 timeout、其他异常都回退为完整 `capped`；外层 `/api/breath` deadline 取消门卫时又返回 DS 前完整 partial snapshot。因此门卫失败等于门卫不存在。
- 本次只有一条修复路线：失败时复用已记录的 Anchor 绝对分，保留 `score >= 0.450001` 与 exact retrieval-key forced 候选；若没有任何候选过线，确定性保底当前顺序第 1 条。内部 invalid/timeout/error 与外层取消返回的 partial 共用同一 helper。
- 新开关 `OMBRE_DS_FAILURE_FALLBACK_ENABLED=0` 默认关闭；专用线为 `OMBRE_DS_FAILURE_ANCHOR_FLOOR=0.450001`。未改门卫 8 秒超时、provider/model、调用次数、召回 budget、既有 min_score、PG、RRF、主排序或正常 `ok`/合法空结果。
- 22 条真实请求验证分两层：18 条历史 `ok` 把固定回滚点 `8be9974` 的旧 `_ds_filter_candidates` 与当前补丁函数做结构差分；在每拍真实 query、真实候选投影与真实 forced-key 上穷举全部有序 selector 子集，并同时覆盖 `allow_empty=false/true`，共 **808/808** 组 exact IDs 与渲染 partial SHA 相同。历史 served 列表只作冻结参照，不冒充重放出的 provider 结果。4 条失败/外层取消用真实 query、完整候选桶快照和实际 forced-key 对当前 wrapper 注入真实故障类别，均收为 top-1；12:14 两条明确工程噪音 `013da98a75e5`、`019af40158f7` 在 failure primary 中为 **0 条**。
- 另对 9/5 全量冻结快照做了下游回流核对：两噪音桶均不能从 X/Z/E 入选；Y 的实际 seed `dd279da3beee` 在 960 条 `explains` 边上深度 1/2 都没有邻居，到两噪音均无路径。因此这份冻结快照的最终候选链不会把它们捞回；这仍不是已部署生产验收。
- 这是代码、测试和冻结账本回放，不是已部署生产验收。本次按任务要求不部署；Claude 仍需部署后亲抽 12:14 同类请求做最终注入验收。

## 写入前回滚点与生产证据

Ombre-Brain 目标 worktree 写入前为 clean，分支基点、本地 `origin/main` 均为：

`8be99749d0c24b2249e8502770017c0bf39ecc3d`

远端为 `chaodeng060-source/Ombre-Brain` 的 `main`。生产只读，没有改 NAS、容器、服务、环境变量或数据。

先按约定的 SSH alias 直连，当前执行沙箱真实返回 `socket: Operation not permitted`。随后使用该 alias 的同一 `127.0.0.1:2222`、用户、专用 key、`IdentitiesOnly=yes`、`BatchMode=yes`、`StrictHostKeyChecking=yes` 和既有 `known_hosts` 显式展开连接成功；没有关闭主机指纹校验。生产容器回执：

- 容器：`ombre-vps-mirror`
- 状态：running；restart count `0`
- 启动：`2026-09-04T04:16:48.541960978Z`
- 镜像：`ombre-vps-mirror:f669ac8`

冻结最近整 24 小时窗口：UTC `2026-09-04T10:10:55Z` 至 `2026-09-05T10:10:55Z`，即北京时间 `2026-09-04 18:10:55` 至 `2026-09-05 18:10:55`。`docker logs --since ... --until ...` 原始流共 6700 行 / 771706 bytes，SHA256：

`479cb3f9e20c8124c5bd81677cda73b2554e0d039017101e5f546c96c3e3c0b5`

这段原始流已通过配置的 NAS 直连重新读取并逐字节核对同一 SHA，保存到私有 mode-0600 文件 `/opt/claude-twin/.work/ds_filter_frozen_24h_20260905_181055.log.gz`，供复核重算，不提交 GitHub。

## 最近 24 小时门卫账

这里统计门卫裁决结果，`ok` 包含成功缓存命中，不能据此反推 provider 网络调用次数。`ok` 的“输入”同时给门卫结果日志里的原始 input 和真正受门卫裁决的 capped input；失败调用从 timing 取得 capped 前后数。

| 旧状态 | 次数 | 旧放行数量 | 证据口径 |
|---|---:|---:|---|
| `ok` | 81 | 原始 689，capped **375 → 227** | `DS filter mode=... input/capped/kept` |
| invalid payload（旧 timing 误记 `error`） | 8 | **30 → 30** | 每次都有同拍 invalid 原始元数据行、专用 ValueError 回退行和 timing |
| 内层 `TimeoutError` | 2 | **10 → 10** | 两次均 5→5 |
| 其他 caught error | 0 | 0 | 排除上述 ValueError 与 TimeoutError 后无剩余回退行 |
| 外层 deadline 在 DS 内取消、旧代码无 DS 状态 | 3 | 无旧 `ds_gate_in/out`；返回完整 pre-DS partial | timing 有 `ds_filter` 活跃阶段，但没有任何 `ds_gate_*` |

另有 11 次 `noop`，没有实际模型判定，不混进上述调用状态。

同窗 `breath_timing` 自身包含 70 次 `ok`（324→184），81 次总账来自所有入口的 `DS filter mode=...` 行，含没有独立 `breath_timing` 的旧 Anchor 入口，不能只数 HTTP timing 代替总账。

### 8 次 invalid 的逐请求放行账

| 时间 | request_id | 旧门卫 |
|---|---|---:|
| 9/4 22:29 | `cc4a9502cf7a45a28abe25ea7828d8f5` | 5→5 |
| 9/4 22:34 | `885ce80961104715af946da43291d711` | 1→1 |
| 9/4 22:37 | `7687b8b307174b66b40a91118c600e86` | 4→4 |
| 9/4 22:44 | `647b70b7ae0d473ebbebd5b356d466a0` | 4→4 |
| 9/4 23:46 | `cd0eab54bd5348ca904b1a02bbe87722` | 5→5 |
| 9/5 11:02 | `138ae29d39f1448ba510cd1951bfcc4b` | 5→5 |
| 9/5 12:07 | `804b68ba4cf8450a9f2f59acfaebf005` | 1→1 |
| 9/5 12:14 | `5e00777f6b2f4c3e823ec6064aa39d42` | 5→5 |

### 2 次内层 timeout 与 3 次外层取消

| 类型 / 时间 | request_id | 旧留痕与放行 |
|---|---|---|
| 内层 timeout / 9/4 21:55 | `4f46ee5213374af9b259730faf2a17f2` | `timeout`, 5→5 |
| 内层 timeout / 9/5 16:42 | `33c781e137e840ed9ed20a9ac98400ae` | `timeout`, 5→5 |
| 外层取消 / 9/4 19:41 | `da4084d9445c46b79b4ba63eed9b322e` | DS 已运行 3502.526 ms；无 DS 状态/前后数，返回全量 pre-DS partial |
| 外层取消 / 9/5 01:37 | `8d3debaf71d741b09767daaba0657a1f` | DS 已运行 2935.884 ms；同上 |
| 外层取消 / 9/5 08:55 | `7897a75aff194cfd9caaf266ad5f6b48` | DS 已运行 1190.645 ms；同上 |

## invalid 原文调查：能确认什么，不能伪造什么

8 次 invalid 的保留证据完全一致：

- `raw_chars=322`
- `raw_sha256=2aa1d8035362584ad6c763c45c936519aab861828f5d50583c65bb7b30d1b4ed`
- model `gemini-3.7-flash-tiered`
- `finish_reason=stop`
- `completion_tokens=87`
- `content_chars=322`
- `reasoning_content_chars=0`、`refusal_chars=0`、`tool_call_count=0`

**原文无法从现存生产证据还原；这是本交付物明确保留的证据缺口。** 旧实现有意只记录长度、SHA256、脱敏 head 和响应元数据，任意模型正文从未落日志；SHA256 不可逆。报告不能把猜测冒充“收到的原文”。同一个 322 字符 SHA 在不同 prompt 上重复，支持“固定 provider 文本/模板”的推断，但仍不能证明具体文字。

解析链的实际失败点是 `_parse_ds_keep_indices(raw, count)` 返回 `None`，随后旧 `_ds_semantic_select` 抛 `ValueError`：

1. `_ds_json_payloads` 会扫描短 prose 或 Markdown 围栏中的完整 `{...}` / `[...]` JSON；围栏本身不会导致失败。
2. `_parse_ds_keep_indices` 接受 `{"keep": [...]}` 或裸数组，合法 `{"keep": []}` 也成功；它只在没有完整 JSON、没有 list 合同、或非空 list 中没有任何有效范围内索引时返回 `None`。
3. `finish_reason=stop`、固定 322 字符和充足输出预算排除了普通 token-budget `length` 截断；但不能排除 provider 主动返回了语法不完整文本。
4. 因为原文已脱敏，无法再区分“无完整 JSON / wrong key 或类型 / 全部越界索引”三类。Markdown 围栏包着合法 JSON 则可明确排除，因为现行解析器本就能解析。

本次没有为未知原文乱加宽解析合同。新代码把解析失败改为专用 `DSFilterInvalidPayloadError`，未来在不记录正文的前提下写出 `parse_reason=empty_content|non_string_content|no_complete_json|missing_keep_list|no_valid_indices`、是否围栏和完整 JSON 值数量；invalid 不再混入 generic error，也不进入成功缓存。

## 为什么保守门槛是 0.450001

真实 22 请求账本共有 318 个 Anchor 评分：min 0.220432、P50 0.2630835、P90 0.4198322、P95 0.45、max 0.45；26 个恰好为 0.45，**没有一个 >= 0.450001**。`ids_in` 中能同拍关联的 78 个评分同样 max 0.45，其中 14 个恰好 0.45；另 11 个旧 trace 输入没有同拍 Anchor 行，不补造分数。

Anchor 适配公式的理论上限也是 `0.45 * 1.0 = 0.45`。12:14 两个工程噪音因为 `entity_match=true` 都拿到 0.45；真正相关的首位 `dd279da3beee` 为 0.444237。于是只靠既有 Anchor 分时存在一个不可绕开的事实：

- 门槛 `<= 0.45` 会保留两个明确噪音；
- 门槛 `> 0.45` 才能先拒绝全部得分候选，再由“至少保底 1 条”确定性留下当前排序第 1 条 `dd279da3beee`。

因此取现有六位小数精度的下一格 `0.450001`。这不是暗改 Anchor 主门槛，也不是新排序公式；只在 DS 已经失败且专用开关开启时使用。实际含义必须说透：按当前分值上限，失败路径通常会收为 top-1；exact retrieval-key forced 候选仍遵守既有强制保留合同。

## 实现与留痕

1. `_ds_conservative_failure_candidates` 是 invalid、内层 timeout、generic error、外层取消 partial 的唯一回退 helper；默认关闭时仍返回旧 capped 结果。
2. `_ds_filter_candidates` 单独捕获 invalid；内层 `TimeoutError`、generic error 同路回退。捕获 `asyncio.CancelledError` 时先记录 `timeout + fallback 前后数`，再重新抛出，绝不吞掉外层取消。
3. `breath` 在进入 DS 前，用同一 helper 生成 deadline 可返回的 partial snapshot；正常 DS 完成后仍由原成功结果覆盖后续路径。
4. `breath_timing` 和 `/api/anchor-status` trace 都保留旧兼容字段 `ds_gate_outcome/ds_gate_in/ds_gate_out`；invalid 在旧 outcome 中仍映射为 `error`，避免现有 Twin 白名单丢掉整组计数；实际门卫调用再增加精确的 `ds_status=ok|invalid|timeout|error`。每次裁决同时写无正文的 `DS gate diagnostic` 行，覆盖直接 MCP 调用及调用方被取消、来不及写最终 trace 的情况。disabled/noop 没有模型调用，不伪造 `ds_status`。
5. 正常 `ok`、合法 `keep: []`、缓存命中、forced ID、E-chord decision capture、最大结果数的原合同保持不变。

回滚开关：保持/恢复 `OMBRE_DS_FAILURE_FALLBACK_ENABLED=0`。本次未部署，所以线上当前无需做任何回滚动作。

## 22 条真实请求确定性回放

私有账本：`/opt/claude-twin/.work/recall_noise_ledger_20260905.json`，mode 0600，不提交 GitHub。回放工具只输出 request/bucket ID 和 query 指纹，不输出原问句或桶正文：

```sh
env -u TWIN_E_CHORD_SHADOW_ENABLED -u TWIN_E_CHORD_RECALL_ENABLED \
  -u TWIN_RECALL_NAVIGATION_ENABLED PYTHONDONTWRITEBYTECODE=1 \
  /opt/claude-twin/.venv/bin/python tools/replay_ds_failure_fallback.py \
  /opt/claude-twin/.work/recall_noise_ledger_20260905.json \
  --supplemental-buckets \
  /opt/claude-twin/.work/recall_noise_supplemental_buckets_20260905.json \
  --supplemental-buckets \
  /opt/claude-twin/.work/ds_replay_success_supplemental_20260905.json
```

结果：22 requests / 89 次候选出现 / 88 个唯一候选快照全部齐全；21 条账本原问句逐条 SHA 前缀吻合；1 条 08:54 自动催办的 original text 在旧 trace 中不可恢复，只使用生产 DS 日志留下的真实 effective query，未自造替代文本。4 条失败路径全部有原问句且 SHA 验证通过。18 条成功样本的固定旧函数 vs 补丁函数结构差分共 808 个 selector/`allow_empty` 场景，exact IDs 与 `_local_partial_recall_text` SHA 都为 **808/808**；4 条失败样本实际注入 invalid/cancel，wrapper 留痕与 helper 输出吻合；12:14 明确工程噪音 forced `0`、failure-primary admitted `0`。工具明确记录 `provider_call_count=0`，没有把结构证明冒充 provider 重放。

12:14 缺失的 `c0731e844589` 当前活跃桶已不存在；从 NAS 只读快照 `/snapshots/lmc5-night-20260905/.../时间窗根因定位_2026-08-17_c0731e844589.md` 找回，原文件 SHA256 `c2570d4590d6f199766ae92292c5a6131e521649878eba4960203619e3b02f65`。只抽取其检索钥匙写入 mode-0600 的私有 supplemental，不提交 GitHub。对 4 个失败请求调用生产同款 `_exact_retrieval_key_ids`，已知 forced 集合均为空，目标两噪音也明确不是 forced。

成功样本另外缺失的 10 个候选桶，已从同一 `/snapshots/lmc5-night-20260905/files/` 只读恢复真实正文与完整 frontmatter。私有 `ds_replay_success_supplemental_20260905.json` 为 mode 0600、40,548 bytes，SHA256 `e1fbd4fd5ef6d1251bdc5330577dcf4da7e45c8557b710416b12dedae82b5aa1`；逐桶原文件路径、SHA 与长度均在文件中留证，不提交 GitHub。

### 18 条历史 ok：served 参照 + 固定旧函数/补丁结构差分

历史 served IDs 只是冻结参照，不是工具直接复制成“修后结果”。工具从 Git 对象 `8be99749d0c24b2249e8502770017c0bf39ecc3d:server.py` AST 提取旧 wrapper（源码 SHA256 `22b8699619f584bdbdcab81fb4c87f8d8205d6c9e0a9ddcc4711df1a23adebde`），再与补丁 wrapper 使用完全相同的成功 selector。对每拍真实候选投影枚举 `2^n` 个有序 keep 子集，并各跑 `allow_empty=false/true`；最右列同时比较返回 exact IDs 和 `_local_partial_recall_text` SHA。

| request_id | 历史 served 参照 | 旧函数 = 补丁（IDs + partial SHA） |
|---|---|---|
| `ba441b1272c04983b5d58b449b5504d2` | `130953c74a10`, `1dd4687a4d54` | 16/16 |
| `522a895ba6ce4855882f28f4121887ef` | `de1dd9925d29`, `dde4868d8a47`, `83ab95052f98` | 64/64 |
| `82f1461c344744f7b19ed6428dced491` | `1184e22cbb8f`, `3b6200416132`, `5c3b5f57b097` | 64/64 |
| `d4e1c4a832d74b1fbfaf5ab2fa01e77c` | `ccf7275d25fa`, `9c31cf1a29f6` | 64/64 |
| `7fa0ff67ca954ae58750978fb53699e0` | `3d873cbdd3c5` | 4/4 |
| `f13c1bf529744c7f954057a81e7d122b` | `c703a0d06946` | 16/16 |
| `7a40ea9c1b394bb38dae9b637f0e11c4` | `f86b8691d5bd`, `983ec6172e01`, `40a8c0bd09fb` | 16/16 |
| `8911a11a618540fe8bc742393f355f15` | `a7ff337dff30`, `7a4fd9b8a53f`, `3f957c6e364c` | 64/64 |
| `53aa2ba716a14a269e4900943f02c7ef` | `b061b70d26a6` | 4/4 |
| `f3b4236bbbbf4d7bae76e3a8fef4611c` | `710be680f019`, `db382cc729a7` | 16/16 |
| `1577417e2d124405b4ac89a4ddba9c94` | `430863a4c10a`, `0d42d2ee37f9` | 16/16 |
| `a65b669be36e49baab6ca9e2d7f07082` | `277cea91c329`, `ae72a42f5657`, `17dccb320df1` | 16/16 |
| `844e6108dc444c228f0b4120569330a8` | `1cadaeded2bf`, `6cc5995aea84`, `b93a91c1c911` | 128/128 |
| `7d2f5821aeb147cb896649965d373b58` | `e2e393af5f7e`, `22863545f2ec`, `b41cd33cae72` | 64/64 |
| `b6ad0f33e7984325b303bbf999d1327a` | `588d818b2005`, `9a30690e7c2e` | 32/32 |
| `c83423ce73e840ffad06655344309af5` | `4e111721b21a`, `ea081b3d04b5` | 32/32 |
| `51188f4334c04d568a5d8c8b25a14da8` | `7772214257c4`, `1b76801c50f8` | 128/128 |
| `d9993958bf694d049eb13b1edc96ebd5` | `114ffffb0da3`, `2ee2f3279dec`, `73b92dcc3d7f` | 64/64 |

### 4 条失败/取消：失败主候选回放

| request_id | 修前 served IDs | 修后 failure primary IDs |
|---|---|---|
| `8d3debaf71d741b09767daaba0657a1f` | `8c6669d3534f`, `16f6fb6f70a5`, `a834457f8297` | `02347186115b` |
| `7897a75aff194cfd9caaf266ad5f6b48` | `ca346188fd0c`, `30ffdf5c0149`, `f86b8691d5bd` | `30ffdf5c0149` |
| `138ae29d39f1448ba510cd1951bfcc4b` | `cc1b0d0649b0`, `ae0c01b0f7ee`, `4bec31272bfb` | `cc1b0d0649b0` |
| `5e00777f6b2f4c3e823ec6064aa39d42` | `013da98a75e5`, `019af40158f7` | `dd279da3beee`；两条明确工程噪音 0 |

这里的“修后 failure primary”来自真实 query/候选投影：两条历史 invalid 对当前 wrapper 注入 `DSFilterInvalidPayloadError`；两条外层 deadline 让实际 selector 挂起后取消 task，并确认 `CancelledError` 继续上抛、timing 留 `ds_status=timeout`、5/4→1。wrapper 返回（invalid）或取消前记录的 fallback 数量（cancel）均与实际新 helper 一致，渲染 partial 均非空。forced 不是假定为空，而是用真实 query 和完整桶快照调用现行 `_exact_retrieval_key_ids` 算得 4/4 为空。12:14 的第五个 `ids_in` 在旧 Anchor 行中缺分，但其桶快照与检索钥匙已补回且确认非 forced；保底仍取现有第一名。

### 12:14 下游 X/Y/Z/E 回流核对

只证明 DS primary 为 0 仍不够，因为 X/Y/Z/E 有各自的候选来源。对同一份 9/5 冻结快照继续做只读核对：

- X：两工程桶的 `thread=other`，当前 timeline 规则明确不选 `other`。
- Z：两桶都没有 `fact_key/fact_status`，不属于 lifecycle state link。
- E：两桶都没有 `e_authored_by/e_initial_priority`，不进入 E 候选分组。
- Y：扫描快照 14,071 个 Markdown 文件、13,973 个唯一 bucket ID、960 条有效 `explains` 边；从本拍实际 primary seed `dd279da3beee` 双向 BFS，depth 1 = 0、depth 2 = 0，到 `013da98a75e5` / `019af40158f7` 均无路径。快照里确有 `c0731e844589 -> 013da98a75e5`，但 `c0731e844589` 不是本拍 seed、也不在该 seed 的两跳可达集合，不能把 013 捞回。

所以在冻结的 12:14 query、候选、元数据、关系图与现行 X/Y/Z/E 合同下，最终候选链中的两条明确工程噪音为 **0**。这是一份离线冻结证据，不代替部署后 live metadata/config 下的最终 rendered capture。

## 测试与交接

红测先证明旧代码的内部失败仍全放行、外层 deadline partial 仍含噪音；补丁后结果如下：

- 完整核心相关回归先跑到 `159 passed, 1 warning in 306.42s`，覆盖新失败合同、DS cache/旧合同、timing、E-chord 与 E-axis 邻接路径。唯一 warning 是既有 `test_keyword_scoring_yields_to_request_deadline` 的 executor 线程 join 超过 300 秒；测试本身通过。
- 加入固定旧函数结构差分与 fail-closed 变异测试后的最终相关复验为 `175 passed, 1 deselected in 5.71s`；只排除上面已经通过、但每次会等待 300 秒线程 join 的同一测试。
- 新回放工具自身测试 `12 passed in 5.24s`；覆盖固定 commit AST 提取、真实私有账本全批、query/snapshot/status 变异拒绝、补丁漂移探针与 supplemental 冲突拒绝。
- 22 条真实请求 CLI 以退出码 0 完成：18 条成功样本共 808/808 组 fixed-baseline vs patch exact IDs + partial SHA 相同；2 条 invalid 与 2 条 outer cancel 实际注入完成，取消继续上抛且 timing 前后数为 5/4→1；目标两条工程噪音 failure-primary admitted/forced 均为 0。
- `git diff --check` 通过。

提交与 push SHA 写入 TaskRail 交接回执，避免在同一个 Git commit 内伪造自指 SHA。

没有部署。Claude 的终验必须把“代码已推”“生产已加载”“真实同类请求注入验收”分开报告。
