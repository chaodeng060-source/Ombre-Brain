# XYZME · M 轴只读代谢

`metabolism.mode` 是严格枚举：

- `report_only`（默认）：衰减与整理只返回候选，不调用 `update`、`archive`、
  `create`、`delete` 或合并，也不会把“夜班报告”写回可召回的记忆桶。
- `apply`：保留旧版自动 resolve/archive/digest/报告桶行为，只能由维护者显式开启。

`report_only` 的返回值会区分：

- 合法空结果：`ok=true`，候选数组为空；
- 读取/计算失败：`ok=false`，`errors` 非空；
- 候选：`would_auto_resolve`、`would_archive`、`would_digest`，
  仅供人审，不代表已经修改。

M 轴不读取或修改 Z 轴事实状态，也不会根据 E 轴分值自动销账。需要真实应用候选时，
必须另走带快照、持久事务/回执和人工确认的维护流程。
