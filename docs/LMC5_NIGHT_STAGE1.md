# LMC-5 保守夜班 Stage 1

这条生产夜班只承诺一个可验证的小闭环：

1. 在同一把维护栅栏内创建并校验数据快照。
2. 把永久 raw-event 账本截至本轮 cutoff 的未覆盖事件脱敏、分块。
3. 用严格 proposer 生成候选。
4. 只自动写入 `risk=normal`、无关系边、`type=event` 的 X 候选。
5. M 只运行 `report_only`，不会归档、digest、resolve 或写报告桶。
6. 校验快照、账本、raw coverage、候选终态后，冻结水位内已清空才标记
   `complete`；仍有 proposer 积压则如实标记 `deferred`。

Y、Z、E 在这条无人值守链上明确保持 `deferred`。这不是完整五轴夜班，
也不能以完整五轴名义验收。

## 生产开关

服务端需要：

```env
OMBRE_LMC5_NIGHT_ENABLED=1
OMBRE_LMC5_SNAPSHOT_DIR=/snapshots
```

`OMBRE_LMC5_SNAPSHOT_DIR` 必须是绝对路径，并挂载到独立持久目录。启用后，
旧的进程内 24 小时 decay/consolidation 漂移循环不再启动，避免同一晚跑两次。

入口是受现有 `/api/*` Bearer 鉴权保护的：

```text
POST /api/maintenance/lmc5-night
{"schema_version":1}
```

宿主机不保存或传递 Token。定时器用 `/usr/bin/flock` 防止任务重叠，
再进入容器；触发器从容器环境读取 `OMBRE_API_TOKEN`。生产任务不得直接运行
旧 `/vol1/ombre-deploy` wrapper，也不得依赖脚本的隐式容器名。统一入口由
`scripts/install_nas_jobs.sh` 安装，并从受保护的
`/vol1/ombre-migrate/nas-production.env` 显式加载
`OMBRE_CONTAINER_NAME=ombre-vps-mirror`。

04:00 冷备必须覆盖同一 8001 实例的 `/vol1/ombre-migrate/data` 与权威源码，
且通过 SQLite 解包回查；旧的 `/home/zhaodeng/ombre_daily_backup.sh` 只备份
8000 数据，不能作为这条夜班的前置备份。LMC-5 在 04:30 运行，E-axis 在
05:30 运行。完整安装、迁移闸和回滚步骤见
[`NAS_8001_PRODUCTION.md`](NAS_8001_PRODUCTION.md)。

夜班的“同一天”按 `Asia/Shanghai` 的 04:30 边界计算，不按午夜计算：
从本日 04:30 到次日 04:30 前都属于同一个逻辑日。本轮 cutoff 永远固定
在这个逻辑日开始时的 04:30；触发时间变晚、进程重启或改用 `-rN` 重试
都不会扩大输入集合。04:30 之后到达的事件留到下一个逻辑日处理。

## 失败语义

- 同一逻辑日完成过一次后，重复触发只返回原终态，不重复写入。
- proposer 每个 run 最多处理 16 个 chunk，且总墙钟预算低于 3600 秒；
  本轮开始时冻结 `event_chunks` rowid 水位，运行中新增的 chunk 留给下一轮。
- `deferred` 是干净的历史终态，不是失败也不会被改写成 `error`。同一逻辑日
  再触发会使用新的 `-rN`，优先处理从未尝试或重试较少的 chunk。
- 单块 provider/合同错误只记录为 `retryable_error`；连续三块错误会熔断本轮。
  重试三次以上的块计入 `proposer_quarantined` 以便审计，但仍保持 pending，
  不伪造成功终态，也不永久丢弃。
- 失败或被中断的 run 永久保留为证据；重试使用有界的新 run ID，
  且所有 `-rN` 使用完全相同的固定 cutoff。
- X 正文先进入冷仓，required 向量成功后才提升；相同 idempotency key
  重试不会创建第二个可见桶。
- 任何失败都保留本轮快照和机器错误码，不把失败伪装成空成功。
- Stage 1 不自动做破坏性的 live restore。恢复必须由人核对快照清单后执行。

## 验收

上线前后都要满足：

- 匿名 `/api/maintenance/lmc5-night` 返回 401。
- 容器内带 Token 触发返回 `contract=lmc5-conservative-stage1` 和
  `stage=complete|deferred`；前者必须是 `complete=true,degraded=false`，
  后者必须是 `complete=false,degraded=true`。
- 账本必须满足 `attempted=succeeded+retryable`、
  `pending_before=succeeded+pending_after`，且 `attempted<=16`；
  `complete` 要求 `pending_after=0`，`deferred` 要求 `pending_after>0`。
- 本轮不存在未派发的 candidate；未处理的 proposer chunk 只能作为有精确
  计数的 `deferred` 积压存在。
- 旧记忆正文与元数据未因 M 改写。
- 只有同一逻辑日已经 `complete` 时，第二次触发才返回
  `already_complete=true`；若上一轮 `deferred`，则必须创建新的 `-rN` 续跑。
- cron 只有一条 LMC-5 任务，显式写入 8001，并与同一 8001 数据的 04:00
  冷备错峰；不得残留旧 8000 writer/backup 行。
