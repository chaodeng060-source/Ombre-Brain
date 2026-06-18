#requires -Version 5.1
<#
Deploy Ombre Brain to the NAS host.

Default target:
  zhaodeng@192.168.1.207:/home/zhaodeng/ombre-brain

Examples:
  .\scripts\deploy_nas.ps1
  .\scripts\deploy_nas.ps1 -DryRun -SkipTests -NoPush

The script deploys tracked files from the current Git commit only. Local
untracked folders such as docs/, notes/, and tools/ are ignored until they are
intentionally committed.
#>

[CmdletBinding()]
param(
    [string]$RemoteHost = "192.168.1.207",
    [string]$RemoteUser = "zhaodeng",
    [string]$KeyPath = "$env:USERPROFILE\.ssh\nas_zhaodeng",
    [string]$RemoteDir = "/home/zhaodeng/ombre-brain",
    [string]$BackupDir = "/home/zhaodeng/ombre-backups",
    [string]$Service = "ombre-brain",
    [string]$Branch = "main",
    [string]$HealthUrl = "https://memory.zhaodeng.xyz/health",
    [switch]$SkipTests,
    [switch]$NoPush,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string[]]$Command
    )

    Write-Host "==> $Title"
    Write-Host ("    " + ($Command -join " "))
    if ($DryRun) {
        return
    }

    $exe = $Command[0]
    $argv = @()
    if ($Command.Count -gt 1) {
        $argv = $Command[1..($Command.Count - 1)]
    }

    & $exe @argv
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Remote {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Script
    )

    $target = "${RemoteUser}@${RemoteHost}"
    # OpenSSH concatenates remote argv without preserving argument boundaries.
    # Passing `bash -lc $Script` therefore turns only the first word into bash's
    # command and lets later lines run in the parent shell without `set -e`.
    # Base64 keeps the complete script as one payload and pipes it to bash stdin.
    $normalized = $Script -replace "`r`n", "`n"
    $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
    $remoteCommand = "printf '%s' '$payload' | base64 -d | bash -se"
    Invoke-Checked $Title @("ssh", "-i", $KeyPath, $target, $remoteCommand)
}

function Assert-CleanTrackedWorktree {
    $dirty = & git status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0) {
        throw "git status failed"
    }
    if ($dirty) {
        throw "Tracked files have uncommitted changes. Commit or stash them before deploy."
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$currentBranch = (& git rev-parse --abbrev-ref HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Cannot determine current Git branch"
}
if ($currentBranch -ne $Branch) {
    throw "Refusing to deploy branch '$currentBranch'. Switch to '$Branch' or pass -Branch explicitly."
}

Assert-CleanTrackedWorktree

$head = (& git rev-parse --short HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Cannot determine Git HEAD"
}

if (-not $SkipTests) {
    Invoke-Checked "Run full test suite" @("python", "-m", "pytest")
}

Invoke-Checked "Fetch remote refs" @("git", "fetch", "origin", "--prune")

if (-not $NoPush) {
    Invoke-Checked "Push $Branch to origin" @("git", "push", "origin", $Branch)
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archiveName = "ombre-brain-$head-$timestamp.tar"
$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) $archiveName
$remoteArchive = "/tmp/$archiveName"
$backupPath = "$BackupDir/ombre-brain-code-pre-deploy-$timestamp.tgz"

try {
    Invoke-Checked "Create deploy archive from HEAD $head" @(
        "git", "archive", "--format=tar", "-o", $archivePath, "HEAD"
    )

    Invoke-Checked "Upload deploy archive to NAS" @(
        "scp", "-i", $KeyPath, $archivePath, "${RemoteUser}@${RemoteHost}:$remoteArchive"
    )

    $backupScript = @"
set -euo pipefail
mkdir -p '$BackupDir'
tar --exclude='.env*' --exclude='__pycache__' --exclude='.pytest_cache' -czf '$backupPath' -C '$RemoteDir' .
test -f '$backupPath'
tar -tzf '$backupPath' >/dev/null
"@
    Invoke-Remote "Backup current NAS source" $backupScript

    $deployScript = @"
set -euo pipefail
mkdir -p '$RemoteDir'
tar -xf '$remoteArchive' -C '$RemoteDir'
rm -f '$remoteArchive'
cd '$RemoteDir'
printf '%s\n' '$head' > '.ombre-deployed-commit'
docker compose -f docker-compose.yml build '$Service'
docker stop '$Service' >/dev/null 2>&1 || true
docker rm '$Service' >/dev/null 2>&1 || true
docker compose -f docker-compose.yml up -d '$Service'
"@
    Invoke-Remote "Install new source and restart container" $deployScript

    $healthScript = @"
set -euo pipefail
for i in `$(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health; then
    exit 0
  fi
  sleep 2
done
exit 1
"@
    Invoke-Remote "Verify NAS local health" $healthScript

    if ($HealthUrl) {
        Invoke-Checked "Verify public health" @("curl.exe", "-fsS", "--max-time", "20", $HealthUrl)
    }

    if ($DryRun) {
        Write-Host "Dry run complete: no local archive was created and no remote changes were made."
    } else {
        Write-Host "Deploy complete: $head"
        Write-Host "NAS backup: $backupPath"
    }
}
finally {
    if ((-not $DryRun) -and (Test-Path $archivePath)) {
        Remove-Item -LiteralPath $archivePath -Force
    }
}
