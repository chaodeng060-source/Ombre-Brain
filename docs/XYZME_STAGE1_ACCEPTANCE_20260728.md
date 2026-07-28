# Ombre-Brain · LMC-5 XYZME Stage 1 验收

日期：2026-07-28

## 结论

XYZME Stage 1 已在 `chaodeng060-source/Ombre-Brain` 的 `main` 分支完成。
本次是对 Ombre 现有 Markdown/YAML、SQLite 与 recall 管线的增量加固，不引入
第二套数据库，也不把 LMC-5 参考仓库的 PostgreSQL 实现直接复制进来。

最终本地全量回归：

```text
460 passed, 7 skipped, 1 warning in 12.27s
```

7 个 skip 全部来自需要外部 `OMBRE_API_KEY` 的 LLM 质量基准；
没有功能测试失败。Python 编译与 `git diff --check` 通过。

## 五轴交付

### X · Timeline / provenance

- 保留 Ombre 既有 `event_at / date_precision / date_source / world /
  domain / tags` 时间线坐标。
- episode、saga 与 import 新建记忆在第一次原子写入时绑定来源证据；
  不再先创建无来源桶、再用第二次 update 补链。
- provenance 创建后不可改写；saga 只允许无重复的严格前缀追加。
- import digest 绑定模型实际收到的脱敏、12000 字符上限输入；
  合并进旧桶时不冒充一对一精确来源。

提交：`3005ee0`

### Y · Relations

- 安全关系类型进入双向、最多两跳的图扩展。
- 只以真正进入结果集的桶为 seed，并继续执行世界、领域、时间、
  Z 当前性和 session 边界。
- `contradicts / supports / cause_effect` 等审计关系不能进入默认扩图，
  图扩展结果保持 association 身份，不偷换主召回排序。

提交：`6e45bcf`

### Z · Fact evolution

- 当前性统一使用既有 `fact_status=current|historical` 与显式注册的
  `fact_key`，不新增另一套事实真源。
- 精确事实问题过滤 historical；历史问题仍可召回历史版本。
- 未注册槽位、上下文不匹配、保护域或叙事记忆均 fail-open，不误删。
- 缺少真正跨文件事务时，危险的事实 apply 与保护层改写端点固定
  fail-closed，避免半写状态。

提交：`39546ca`、`aa6f8d3`、`6e45bcf`

### M · Metabolism

- `report_only` 是默认模式：只报告 would-resolve、would-archive、
  duplicate 与 stale 候选，不修改、归档、digest 或写回报告桶。
- 失败显式进入 `ok=false / errors`，不能用空结果冒充成功。
- 旧 apply 行为仅在管理员精确配置 `metabolism.mode=apply` 时保留，
  以维持现有部署兼容性。

提交：`988b6e6`

### E · Experience

- E 只写 `<buckets_dir>/.axis/e-shadow.jsonl` 旁路账本；
  永久 `shadow_only=true`、`affects_ranking=false`。
- 标注绑定桶正文 SHA-256、scorer、model 与 rubric；旧正文评分拒绝。
- 严格拒绝重复 JSON key、NaN、Infinity、浮点溢出、超大整数、
  非法配置与损坏账本。
- 跨平台文件锁、fsync、0600、幂等 key 与精确版本行合同已验证。
- E 路由只读取桶，不调用 create/update/touch/archive，不覆盖现有
  `valence / arousal`，也没有自动启用排序的时间闸。

提交：`f976c4d`、`9d7d6fc`

## 共同边界

- 没有访问、修改或重启 NAS。
- 没有部署生产服务或改写现有记忆数据。
- 没有把 GitHub 版本覆盖到 NAS。
- GitHub 基线已经包含此前筛选回仓的 NAS 代码恢复提交；
  本次只在该基线上追加 XYZME 安全增量。
- `82eef9b` 另修复了 review pending 读取时的 executor 挂起，
  是本轮验收的前置稳定性修复。

部署前仍需在能够登录 NAS 后执行：只读差异核对、代码与数据快照、
目标环境回归、显式部署与健康检查。Stage 1 代码验收不等于 NAS 已上线。
