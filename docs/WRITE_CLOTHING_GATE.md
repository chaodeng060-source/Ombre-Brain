# 写入侧补衣闸

## 不变量

所有新桶最终都经过 `BucketManager.create()`。它在原子写入前只接受正文中可逐字命中的
检索钥匙；候选来自调用方传入的实体、`name` 和 `tags`。泛词、纯数字、正文中不存在的词
不会被当成检索钥匙。

- 有可靠字面钥匙：写入可读 `name`，并在正文末尾追加
  `[检索钥匙: key1 / key2]`，同时把钥匙写入 `metadata.retrieval_keys`。
- 没有可靠字面钥匙：正文原样保留，名称使用 `待补衣_YYYY-MM-DD`，写入
  `needs_clothing: true` 和 `clothing_reason: no_literal_retrieval_key`，并向人工队列追加
  一条不含正文、只含正文 SHA-256 的 `kind=clothing` 记录。
- 人工队列不可用不会拒绝记忆：桶上的 `needs_clothing` 仍是可观测兜底，服务端记录
  warning。任何新桶都不得再以 `name == id` 静默落库。

## 产桶入口与覆盖证明

| 入口 | 产桶路径 | 如何经过补衣闸 |
|---|---|---|
| Twin imprint / 千问 worker | Twin `imprint_dispatch` → `memory_gateway.hold` → Ombre `/api/hold` | 千问只做质量判定，真正写入进入下方 `hold` 路径，不能直接写 Markdown |
| `hold` 普通写入 | `server._merge_or_create` | 新建分支调用 `BucketManager.create`；Ombre 消化器 `analyze()` 的已验证实体作为候选钥匙 |
| `hold(pinned=True)` | `server.hold` 直接新建 | 直接调用 `BucketManager.create`，并传入 `analyze()` 实体 |
| `hold(feel=True)` | `server.hold` 直接新建 | 直接调用 `BucketManager.create`；无可靠字面钥匙时保正文并排队 |
| `grow` 短文本 | `analyze` → `_merge_or_create` | 与普通 hold 共用中心新建分支，并传入实体 |
| `grow` 长文本 / Ombre 消化器 | `digest` → 每条 `_merge_or_create` | 模型拆分只负责候选；每条最终仍经过中心闸，模型空名或幻觉名不能绕过 |
| recall-before-write `new` | `_merge_or_create` 新建分支 | 经过中心闸 |
| recall-before-write 状态 `supersede` | `_create_operational_status_successor` | 直接调用中心闸并传入实体 |
| `experience` | `server.experience` | 直接调用中心闸 |
| 历史导入 | `ImportEngine` raw/new 两条新建分支 | 两条都调用中心闸 |
| curated writer | `CuratedWriter` staged create | 调用中心闸；两阶段可见性规则不变 |
| episode / saga | `EpisodeEngine`、`SagaEngine` | 调用中心闸 |
| 夜班 consolidation 报告 | `ConsolidationEngine` | 调用中心闸 |
| 手工 CLI | `write_memory.py` | 实例化 `BucketManager` 后调用中心闸 |

更新、合并、归档和恢复不产生新 bucket id，不属于“新桶补衣”入口；它们继续走原有
`BucketManager.update` 或恢复事务。

## 人工队列查询

命令行只看待补衣项：

```bash
python3 review_queue.py \
  --path "$OMBRE_BUCKETS_DIR/review_queue.jsonl" \
  --kind clothing
```

服务端查询：

```text
GET /api/review_queue?kind=clothing
```

真实库中新增裸桶的观察口径：以部署时刻为分界，扫描部署后创建的 Markdown frontmatter，
统计 `name == id`；目标始终为 `0`。`needs_clothing` 数量单独报告，不能拿它冒充拒收或
裸桶。
