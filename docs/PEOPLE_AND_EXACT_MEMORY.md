# 人物证据与同源副本整理

## 人物证据视图（Y 的补充）

`GET /api/people?q=名字是谁&world=世界` 使用既有 `/api/*` Bearer 认证。
不带 q 可分页浏览，`limit` 最大 100，`evidence_limit` 最大 20；
`history=1` 才展示已标记历史的内容。世界规则沿用现有召回。

它复用 `.entities/entities.sqlite3` 已登记的身份和别名，展示当前桶中
逐字的记忆正文片段、事件时间、桶 ID 和已有原始事件指针。
**正文片段不是原始对话，也不是新增的权威人物事实。** 没有原始指针就不编造。
只有直接写明的简单关系句才产生关系证据；同场出现、否定、假设、历史句不猜关系。
冲突别名不自动选人，过期内容哈希的关联不使用。

`breath` 仅在明确询问身份/关系时加入人物证据通道，并继续执行原有
世界、领域、时间、Z 和后续筛选；日常提到人名不会每轮注入整张人物卡。
`lmc5_night.people_refresh_enabled: true` 使既有夜跑在快照与独占维护边界内
补齐旧正文对已登记名字的派生关联。不新建人物，不生成关系事实，不调用模型。

## 精确副本整理（M 的确定性补充）

`lmc5_night.exact_dedup_enabled: true` 才启用，每晚默认最多处理 20 条。
`plan_exact_consolidation` 可只读演练，`run_exact_consolidation` 执行。
正文逐字相同、明确来源相同，且所有语义元数据相同，才可能成为一组。
作者、世界、事件日期、检索词、关系、未识别语义字段的差异均阻止收拢。
受保护、feel/E、永久、钉住的记录不处理。语义相似的记录继续只报告。

只给副本写 `digested` 与 `exact_duplicate_of` 标记；不删除文件、不改正文。
普通关键词、主召回和合并候选只使用保留项，物理清单与旧 ID 直接读取仍保留。
快照、独占维护锁、更新前版本核对和既有 mutation audit 支持恢复。
`restore_exact_consolidation` 只恢复仍匹配该次更新版本的记录，已被别人改过的跳过。
审计恢复入口为 `recovery_records_from_audit`，无需反复复制整份会话备份。

## 设计出处与范围

- 既有 [LMC-5 五轴](https://github.com/chaodeng060-source/Ombre-Brain/blob/main/docs/CONNECTING_XYZEM.md)：沿用 Y/Z/M 权限分工，不把派生视图当新事实。
- [Aelios](https://github.com/wusaki0723/Aelios/tree/ac8c64f5a6738ae383948bc4d389dd55f3a2bcd5)：借鉴 glossary、dedup gate、版本历史的入口组织；本实现未搬入它的语义合并策略。
- [Aelios-resident](https://github.com/wusaki0723/Aelios-resident/tree/60dc8b3a82a7556bb6b5198d988f46db7a05f3df)：沿用来源区分及后台维护思路。
- [Agent Memory Workbench v0.2.0](https://github.com/yanyan1115/agent-memory-workbench/tree/16cc5c0e134e08b4130f48ace9f4776c6882511e)：Twin 两个私库单独采用人物/别名检索、来源保留和可审阅恢复，不向共享海马体倾倒私库。

这里的具体匹配、锁和回滚行为是本项目实现；上游思路不是上游对本实现正确性的保证。
