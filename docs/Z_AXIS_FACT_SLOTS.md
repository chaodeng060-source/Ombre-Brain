# Z 轴事实槽契约

## 字段

- `fact_key`：可选，只接受 `config.fact_slots.registry` 里登记的规范 key。规范 key 使用带命名空间的 ASCII 形式，例如 `profile.city`。
- `fact_status`：可选，只有 `current`、`historical`、`contested`。缺省按 `current` 解释。
- 不使用 `active_fact`。巡检只报告这个遗留字段，不把它当真值来源。
- `resolved` 与 Z 轴正交：它控制注意力/衰减，不代表事实已经过期。

## 来源与迁移

规范槽只能来自人工维护的注册表，模型不能临时创造槽名。注册表中的 `aliases` 可让巡检从 `key: value` 形式正文提出只读迁移候选；可选的 `domains`、`types`、`tags_any`、`name_contains` 约束用于避免“主色”“位置”这类局部字段跨项目串槽。各约束之间为 AND，同一约束内为 OR。巡检不写 frontmatter。

内置注册表只启用两条有严格 domain/type/tag/name 上下文的 UI 偏好槽，与
`config.example.yaml` 一致，避免无私有配置的生产安装把 Z 轴静默退化为空壳。部署可显式
设置 `registry: {}` 关闭全部事实槽；此时不改变任何现有桶或召回结果。未登记 key、非法
status 和多槽候选都 fail-open，只报告，不隐藏记忆。

## 召回

精确事实意图只隐藏已登记槽的 `historical` 版本；`current` 和 `contested` 都保留，避免争议事实被静默裁决。时间线、回忆和关系意图保留历史版本。

## 红线

pinned/protected、保护域（恋爱、纪念日、约定、家庭、自省、feel）以及 `feel`、`episode`、`saga` 类型不进入 Z 轴审计或过滤。

没有任何自动 supersede、自动改 status 或自动写 `fact_key` 的路径。发现事实翻转时，
新旧内容保留为两个独立桶。`POST /api/review_queue/candidate` 默认
`mode=dry-run`，只返回候选且不写队列/桶；显式 `mode=apply` 也只把候选幂等写成
pending，不改变事实状态。review queue 只接受 `config.fact_slots.registry` 已登记的
规范 key。

`POST /api/review_queue/apply-lifecycle` 是唯一可落事实 lifecycle 的入口：必须提交
pending key、非空 reviewer 和裁决备注，事务会再次校验注册槽与保护边界，然后把新事实
标 `current`、旧事实标 `historical`。双桶原文、目标内容和状态机都先写入
`.z-lifecycle-transactions/` 持久日志；中途失败自动恢复两桶，进程崩溃后启动时根据
队列的 durable status 决定回滚或完成提交。精确重放幂等。

机器推断的危险关系边（`causes/contributes/improves/updates`）另走
`POST /api/review_queue/apply-relation`：请求必须包含 pending key、非空 reviewer 和可选
裁决备注。事务在 `.relation-approval-transactions/` 保存源桶原文和目标原文，具名裁决与
关系边一起收口；任一落盘失败会恢复原桶并保留 pending，崩溃重启则按 durable status
完成或回滚。相同 key 重放不重复建边。安全关系类型不能借该入口绕过危险边合同。

`lifecycle/active_fact` 是遗留字段，不再由新流程写入，也不参与召回真值判断。

`breath` 只在精确事实意图且查询未要求历史时过滤已登记的 `historical`。空注册表、
未登记 key、非法 status 均 fail-open；受保护桶仍不由普通 lifecycle 事务改写。
旧的 `apply-protected-overlay` 接口已停用：保护记忆不会再被旁路 overlay 隐藏。

`patrol.py` 可读取生产 Markdown 桶，也可读取备份工具生成的 `<12hex>.json` 桶快照；`body_state.json` 等不符合桶 schema 的 sidecar 会被忽略。两种输入都只读。
