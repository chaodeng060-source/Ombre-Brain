# X 轴来源链（v1）

X 轴不是一个新字段，也不等于 provenance。Ombre 原有的
`event_at / date_precision / date_source / world / domain / tags` 继续负责
“何时发生、属于哪段世界与主题”的时间线定位。本补丁补的是另一块：
**派生记忆从哪里来，并且这条来源链不能在事后被伪造。**

## 创建时原子写

`BucketManager.create(..., x_provenance={...})` 会先严格校验，再把以下字段
和正文放进同一次原子写：

- `x_schema_version: 1`
- `source_kind`
- 与来源类型匹配的 `source_buckets / episode_buckets`
- 有真实时间证据时的 `span_start / span_end`
- 导入来源的 `source_digest / source_chunk_ordinal`
- 可选的真实 session、thread、event id

episode 必须在创建时同时带齐源桶和起止时间；saga 必须在创建时带上第一
个 episode。不会再先建一个无证据桶、随后用第二次 `update` 补链。

## 更新边界

- `x_schema_version / source_kind / source_buckets / span / digest / ordinal`
  创建后不可修改，也不可用普通 `update` 后补。
- saga 的 `episode_buckets` 只允许追加：新数组必须完整保留旧数组作为前缀，
  且不允许重复、删除或重排。
- 守卫在 `last_active` 改动之前执行；非法请求不会把桶伪装成“刚活跃”。

旧 saga 已有的 `episode_buckets` 仍可按相同的只追加规则延长，但普通 update
不能把一个无来源的旧桶包装成 v1 provenance。

## 导入真实性

导入新建桶记录的是**实际送入提取器的文本**：

- `source_digest` 是完成 secret 脱敏与 12000 字符上限后、实际模型输入的
  完整 SHA-256；
- `source_chunk_ordinal` 是稳定的 0 基序号；
- 只有源文件里确实存在且能解析的起止时间才写 span；
- 有真实起始时间时，沿用现有 `event_at` 时间模型。

如果导入内容合并进已有桶，不写这组精确 provenance。合并结果混合了多个
来源，冒充成当前 chunk 的一对一派生会比不写更危险。

## 非目标

- 不回填或猜测旧桶来源。
- 不把文件名、当前时间或模型输出当作 source id。
- 不改变召回排序。
- 不替代 Z 的事实当前性、Y 的关系图、M 的代谢报告或 E 的 shadow 标注。
