# Z 轴事实槽契约

## 字段

- `fact_key`：可选，只接受 `config.fact_slots.registry` 里登记的规范 key。规范 key 使用带命名空间的 ASCII 形式，例如 `profile.city`。
- `fact_status`：可选，只有 `current`、`historical`、`contested`。缺省按 `current` 解释。
- 不使用 `active_fact`。巡检只报告这个遗留字段，不把它当真值来源。
- `resolved` 与 Z 轴正交：它控制注意力/衰减，不代表事实已经过期。

## 来源与迁移

规范槽只能来自人工维护的注册表，模型不能临时创造槽名。注册表中的 `aliases` 可让巡检从 `key: value` 形式正文提出只读迁移候选；可选的 `domains`、`types`、`tags_any`、`name_contains` 约束用于避免“主色”“位置”这类局部字段跨项目串槽。各约束之间为 AND，同一约束内为 OR。巡检不写 frontmatter。

空注册表不改变任何现有桶或召回结果。未登记 key、非法 status 和多槽候选都 fail-open，只报告，不隐藏记忆。

## 召回

精确事实意图只隐藏已登记槽的 `historical` 版本；`current` 和 `contested` 都保留，避免争议事实被静默裁决。时间线、回忆和关系意图保留历史版本。

## 红线

pinned/protected、保护域（恋爱、纪念日、约定、家庭、自省、feel）以及 `feel`、`episode`、`saga` 类型不进入 Z 轴审计或过滤。

当前阶段没有自动 supersede、自动改 status 或自动写 `fact_key`。review queue
只接受 `config.fact_slots.registry` 已登记的规范 key 并生成待审候选，不改桶。
`apply-lifecycle` 暂停使用：Markdown 双桶写入尚无持久事务与崩溃恢复，不能用
“写两次再按字节回滚”冒充原子提交。`lifecycle/active_fact` 是遗留字段，不再由
新流程写入，也不参与召回真值判断。

`breath` 只在精确事实意图且查询未要求历史时过滤已登记的 `historical`。空注册表、
未登记 key、非法 status 均 fail-open；受保护桶仍不由普通 lifecycle 事务改写。
旧的 `apply-protected-overlay` 接口已停用：保护记忆不会再被旁路 overlay 隐藏。

`patrol.py` 可读取生产 Markdown 桶，也可读取备份工具生成的 `<12hex>.json` 桶快照；`body_state.json` 等不符合桶 schema 的 sidecar 会被忽略。两种输入都只读。
