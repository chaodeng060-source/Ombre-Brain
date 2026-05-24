# ─── Ombre 夜班 Runner (Windows) ───
# 每晚定时拉起 claude CLI，跑 nightly-consolidation.md 这条记忆整理流水线。
# 从 claude-imprint/cron-task.ps1 改来，砍掉 telegram/DB，留：预算闸 + 日志 + 静默。
#
# 装法（Task Scheduler，每晚比如 03:30）：
#   powershell -ExecutionPolicy Bypass -File C:\Users\HP\Ombre-Brain\cron\run-night-shift.ps1
#
# 前置（一次性）：
#   1) ~/.claude/cron-token 放 OAuth token（Max Plan）或 API key
#   2) 本目录放 ombre-mcp.json —— 只挂 Ombre 记忆库这一个 MCP server。
#      ⚠️ 占位：把你 ~/.claude.json 里 "claude.ai 记忆库" 那个 server 块抠出来填进去。
#      只挂 Ombre，别挂 phone-control / 聊天室那些，省 token 也防误触。

$ErrorActionPreference = "Stop"

$Dir        = Split-Path -Parent $MyInvocation.MyCommand.Path
$PromptFile = Join-Path $Dir "nightly-consolidation.md"
$McpConfig  = Join-Path $Dir "ombre-mcp.json"
$LogDir     = Join-Path $Dir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile    = Join-Path $LogDir "night-shift.log"

$TS = Get-Date -Format "yyyy-MM-dd HH:mm"
Add-Content $LogFile "[$TS] === 夜班 start ==="

# ─── Auth ───
$TokenFile = Join-Path $env:USERPROFILE ".claude\cron-token"
if (Test-Path $TokenFile) {
    $env:CLAUDE_CODE_OAUTH_TOKEN = (Get-Content $TokenFile -Raw).Trim()
}

if (-not (Test-Path $PromptFile)) { Add-Content $LogFile "[$TS] ERROR: prompt 不在: $PromptFile"; exit 1 }
if (-not (Test-Path $McpConfig))  { Add-Content $LogFile "[$TS] ERROR: 还没建 ombre-mcp.json（见文件头占位说明）"; exit 1 }
$Prompt = Get-Content $PromptFile -Raw

# ─── 跑 claude CLI ───
# 从 HOME 跑，避免吃到项目级 .mcp.json。预算闸 0.50 刀；夜班是只读+整理，禁删已写进 prompt。
$TmpOut = [System.IO.Path]::GetTempFileName()
Push-Location $env:USERPROFILE
try {
    $Prompt | claude -p --mcp-config $McpConfig --dangerously-skip-permissions --max-budget-usd 0.50 --output-format text >> $TmpOut
} catch {
    # claude 可能非零退出，继续抓输出
}
Pop-Location

$Output = if (Test-Path $TmpOut) { Get-Content $TmpOut -Raw } else { "" }
Remove-Item $TmpOut -ErrorAction SilentlyContinue
Add-Content $LogFile "[$TS] 输出: $Output"
Add-Content $LogFile "[$TS] === 夜班 done ==="
