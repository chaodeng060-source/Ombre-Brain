# XYZME · E 轴 Shadow 合同

E 轴第一阶段只保存体验标注，不影响事实与召回顺序。

写入端点：`POST /api/e-axis/shadow`。请求必须精确包含：

- `bucket_id`
- `source_digest`：当前桶正文的 SHA-256
- `scorer`
- `model`
- `rubric_version`
- `score`：精确字段 `valence`、`arousal`、`tension`、`confidence`、
  `response_tendency`、`growth_delta`

数值必须是有限 JSON number；布尔不是数字。缺字段、额外字段、NaN/Infinity、
浮点溢出、重复 JSON key、越界、非法枚举和低置信度均拒绝，不补默认值。
`e_axis_shadow.min_confidence` 也必须是有限的 `[0, 1]` number，配置错误时端点
fail-closed。正文哈希不匹配时拒绝旧分数。

成功与失败记录都写在 `<buckets_dir>/.axis/e-shadow.jsonl`，不写 Markdown 桶；
失败记录不保存模型原文或记忆正文。账本 append-only、文件锁、fsync、同一
`bucket + source_digest + scorer + model + rubric` 幂等；账本损坏时停止写入。
锁复用跨平台 `storage_safety.advisory_file_lock`。每行都必须精确符合
`contract_version=1` 的 success/failure 模式；重复 key、非有限数、缺字段、
多字段或错误版本都会把账本判为损坏并停止追加。

硬约束：每行都带 `shadow_only=true`、`affects_ranking=false`。现有顶层
`valence/arousal` 仍是旧的在线情绪坐标，E shadow 不读取、不覆盖，也没有
“满 30 天自动开排序”的路径。
