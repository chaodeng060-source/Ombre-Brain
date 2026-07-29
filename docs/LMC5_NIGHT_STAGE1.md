# LMC-5 保守夜班 Stage 1

这条生产夜班只承诺一个可验证的小闭环：

1. 在同一把维护栅栏内创建并校验数据快照。
2. 把永久 raw-event 账本截至本轮 cutoff 的未覆盖事件脱敏、分块。
3. 用严格 proposer 生成候选。
4. 只自动写入 `risk=normal`、无关系边、`type=event` 的 X 候选。
5. M 只运行 `report_only`，不会归档、digest、resolve 或写报告桶。
6. 校验快照、账本、raw coverage、候选终态后才标记 `complete`。

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
再进入容器；触发器从容器环境读取 `OMBRE_API_TOKEN`：

```sh
/vol1/ombre-deploy/cron/run-lmc5-night.sh
```

朝灯 NAS 已有 04:00 数据备份任务，因此 LMC-5 安排在 04:30，避免两份
I/O 密集任务争抢同一数据目录：

```cron
30 4 * * * /vol1/ombre-deploy/cron/run-lmc5-night.sh >>/home/zhaodeng/ombre-lmc5-night.log 2>&1
```

夜班的“同一天”按 `Asia/Shanghai` 的 04:30 边界计算，不按午夜计算：
从本日 04:30 到次日 04:30 前都属于同一个逻辑日。本轮 cutoff 永远固定
在这个逻辑日开始时的 04:30；触发时间变晚、进程重启或改用 `-rN` 重试
都不会扩大输入集合。04:30 之后到达的事件留到下一个逻辑日处理。

## 失败语义

- 同一逻辑日完成过一次后，重复触发只返回原终态，不重复写入。
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
  `stage=complete`。
- 首轮运行后不存在 `pending` raw chunk/candidate。
- 旧记忆正文与元数据未因 M 改写。
- 同一逻辑日第二次触发 `already_complete=true`，且不新增 X 桶。
- cron 只有一条 LMC-5 任务，并与 04:00 备份错峰。
