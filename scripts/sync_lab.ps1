<#
.SYNOPSIS
    本地与实验室服务器之间的选择性同步（基于 Cygwin rsync + Windows OpenSSH 免密）。

.DESCRIPTION
    push：把跑实验需要的源码推到服务器（从调用所在的工作区推）。
    pull：把服务器上的 output 结果拉回本地分析（始终落到主检出，不论从哪个工作区调用，
          避免 rsync 覆盖工作区里指向主检出的 output 符号链接）。
    白名单方式，只同步配置区里列出的内容，其余一律不带。
    默认是 dry-run（只列会动哪些文件，不真正传输），确认后加 -Apply 才真传。

.EXAMPLE
    # 看推送会动哪些文件（dry-run）
    pwsh scripts/sync_lab.ps1 push

    # 真正推送源码到服务器
    pwsh scripts/sync_lab.ps1 push -Apply

    # 把服务器 output 拉回本地（dry-run）
    pwsh scripts/sync_lab.ps1 pull

    # 真正拉回，并临时覆盖主机
    pwsh scripts/sync_lab.ps1 pull -Apply -RemoteHost user@10.0.0.5

.NOTES
    --delete 默认关闭，需要服务器严格镜像本地时再加 -Delete（只对 push 有意义，
    pull 永远不删本地）。datasets / .venv / docs / references 等都不在白名单里。
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('push', 'pull', 'pull-env')]
    [string]$Direction = 'push',

    # 默认 dry-run；加 -Apply 才真正传输
    [switch]$Apply,

    # 加 -Delete 启用 --delete（让目标端镜像源端、删除多余文件）。pull 方向忽略此项以保护本地。
    [switch]$Delete,

    # 临时覆盖配置区里的主机 / 远程目录
    [string]$RemoteHost,
    [string]$RemoteDir
)

$ErrorActionPreference = 'Stop'

# ===================== 可配置区 =====================

# 本机 Cygwin rsync 可执行文件
$RsyncExe = 'E:\Program Files\Cygwin64\bin\rsync.exe'

# 服务器免密目标，形如 user@host；请填成你平时 ssh 连的那串
$DefaultHost = 'dongyajie@10.108.25.128'

# 项目在服务器上的目录（~ 会在服务器端展开）
$DefaultDir = '/home/dongyajie/work/mfuzz'

# 留空则自动构造一条走 Cygwin ssh 的命令（用下面的密钥、复用 Windows 的 known_hosts）。
# 需要完全自定义时再填，例如 'ssh -i /cygdrive/c/Users/xxx/.ssh/id_rsa'
$SshCmd = ''

# 免密用的私钥文件名，位于 %USERPROFILE%\.ssh\ 下
$SshKey = 'id_ed25519'

# 留空则用服务器默认 rsync；若非交互 ssh 找不到 rsync，填绝对路径，例如 '/usr/bin/rsync'
$RemoteRsync = ''

# push：只把这些推到服务器（rsync filter，自上而下第一条匹配生效）
# 目录用 /名字/*** 表示连同内容，单文件直接写 /文件名
$PushFilter = @(
    '--exclude=__pycache__/'
    '--exclude=*.pyc'
    '--include=/mfuzz/***'
    '--include=/scripts/***'
    '--include=/configs/***'
    '--include=/tests/***'
    '--include=/.python-version'
    '--exclude=*'
)

# pull：只从服务器取 output* 目录，但默认跳过 samples/（变异图 + 抽查标注图，最占地方）。
# rsync filter 自上而下第一条匹配生效，所以排除 samples/ 必须排在 include 之前。
# 需要某个 run 的样本图时，临时把 samples/ 那条注释掉，或单独跑一条带具体路径的 rsync。
$PullFilter = @(
    '--exclude=samples/'
    '--include=/output*/***'
    '--exclude=*'
)

# pull-env：从服务器取回 Linux 那份环境文件（pyproject.toml + uv.lock）
$PullEnvFilter = @(
    '--include=/pyproject.toml'
    '--include=/uv.lock'
    '--exclude=*'
)

# ===================== 可配置区结束 =====================

# 把 Windows 路径转成 Cygwin 的 /cygdrive/<盘>/... 形式
function ConvertTo-Cygdrive([string]$winPath) {
    $full = (Resolve-Path -LiteralPath $winPath).Path
    $drive = $full.Substring(0, 1).ToLower()
    $rest = $full.Substring(2) -replace '\\', '/'
    return "/cygdrive/$drive$rest"
}

# 主检出（jj 默认工作区）根：pull 永远落到这里，不管从哪个工作区调用。
# jj 次级工作区的 .jj/repo 是个文件，内容是指向 <主检出>/.jj/repo 的相对路径；
# 主检出的 .jj/repo 是目录。据此回推主检出根，避免把绝对路径写死。
function Get-MainCheckoutRoot([string]$repoRoot) {
    $jjRepo = Join-Path $repoRoot '.jj/repo'
    if (Test-Path -LiteralPath $jjRepo -PathType Container) {
        return $repoRoot                      # 主检出：.jj/repo 是目录
    }
    if (Test-Path -LiteralPath $jjRepo -PathType Leaf) {
        $rel = (Get-Content -LiteralPath $jjRepo -Raw).Trim()
        $store = (Resolve-Path -LiteralPath (Join-Path (Join-Path $repoRoot '.jj') $rel)).Path
        return Split-Path -Parent (Split-Path -Parent $store)   # <主检出>/.jj/repo -> 上两级
    }
    return $repoRoot                          # 兜底：当作主检出
}

# 解析主机 / 远程目录（参数优先于配置）
$targetHost = if ($RemoteHost) { $RemoteHost } else { $DefaultHost }
$targetDir = if ($RemoteDir) { $RemoteDir } else { $DefaultDir }

if ($targetHost -eq 'user@server') {
    Write-Error "请先在 scripts/sync_lab.ps1 的配置区把 `$DefaultHost 改成你的免密目标，或用 -RemoteHost 传入。"
    exit 1
}

if (-not (Test-Path -LiteralPath $RsyncExe)) {
    Write-Error "找不到 rsync：$RsyncExe，请在配置区修正 `$RsyncExe。"
    exit 1
}

# 让 rsync 用 Cygwin 自带的 ssh（与 rsync 同属 Cygwin，管道模型一致）。
# Windows 自带的 OpenSSH 与 Cygwin rsync 不兼容，组合会报 "connection unexpectedly closed (0 bytes)"。
$cygBin = Split-Path -Parent $RsyncExe
if (-not (Test-Path -LiteralPath (Join-Path $cygBin 'ssh.exe'))) {
    Write-Warning "未在 $cygBin 找到 Cygwin 的 ssh.exe。请用 Cygwin 安装器加装 openssh 包，否则 rsync 会回落到不兼容的 Windows OpenSSH 而失败。"
}
$env:PATH = "$cygBin;$env:PATH"   # 让 rsync 优先找到 Cygwin ssh，而非 System32 的 Windows OpenSSH

# Cygwin ssh 按 passwd 家目录（/home/...）找密钥，不读 $HOME，所以显式指向 Windows 的 .ssh，
# 并把 known_hosts 指过去、用 accept-new，复用你现有的免密配置。
if (-not $SshCmd) {
    $winSsh = ConvertTo-Cygdrive (Join-Path $env:USERPROFILE '.ssh')
    $SshCmd = "ssh -i $winSsh/$SshKey -o IdentitiesOnly=yes -o UserKnownHostsFile=$winSsh/known_hosts -o StrictHostKeyChecking=accept-new"
}

# 仓库根 = 本脚本所在 scripts/ 的上一级（可能是主检出，也可能是某个 jj 工作区）
$repoRoot = Split-Path -Parent $PSScriptRoot
# push 从调用所在工作区推源码；pull 一律落到主检出，避免覆盖工作区的 output 符号链接
$mainRoot = Get-MainCheckoutRoot $repoRoot
$repoCyg = (ConvertTo-Cygdrive $repoRoot).TrimEnd('/') + '/'
$mainCyg = (ConvertTo-Cygdrive $mainRoot).TrimEnd('/') + '/'
$remoteSpec = "${targetHost}:${targetDir}/"

# 组装 rsync 参数（--mkpath 让缺失的目标父目录自动创建）
$rsyncArgs = @('-a', '-v', '-h', '--progress', '--mkpath')
if (-not $Apply) { $rsyncArgs += '-n' }        # dry-run
if ($SshCmd) { $rsyncArgs += @('-e', $SshCmd) }
if ($RemoteRsync) { $rsyncArgs += "--rsync-path=$RemoteRsync" }

switch ($Direction) {
    'push' {
        if ($Delete) { $rsyncArgs += '--delete' }
        $rsyncArgs += $PushFilter
        $src = $repoCyg          # 推调用所在工作区的源码
        $dst = $remoteSpec
    }
    'pull' {
        # 拉回时绝不删本地，忽略 -Delete；目标强制为主检出，避免覆盖工作区 output 符号链接。
        # 服务器端 output 是指向 NAS 的符号链接（root 盘满后迁到 /home/nas511）：rsync -a 默认
        # 不跟随符号链接、不会下钻进 output，加 --copy-dirlinks 把"指向目录的符号链接"当真目录
        # 下钻，才能把 NAS 上的产物拉回。只对 pull 生效（pull 源在服务器）。
        $rsyncArgs += '--copy-dirlinks'
        $rsyncArgs += $PullFilter
        $src = $remoteSpec
        $dst = $mainCyg
    }
    'pull-env' {
        # 拉回 Linux 的 pyproject.toml + uv.lock，绝不删本地；目标同样为主检出
        $rsyncArgs += $PullEnvFilter
        $src = $remoteSpec
        $dst = $mainCyg
    }
}

$rsyncArgs += @($src, $dst)

# 打印将要执行的命令
$mode = if ($Apply) { '真实传输' } else { 'DRY-RUN（不传输，加 -Apply 才真传）' }
Write-Host "方向 : $Direction" -ForegroundColor Cyan
Write-Host "模式 : $mode" -ForegroundColor Cyan
Write-Host "源   : $src" -ForegroundColor DarkGray
Write-Host "目标 : $dst" -ForegroundColor DarkGray
Write-Host ("命令 : `"{0}`" {1}" -f $RsyncExe, ($rsyncArgs -join ' ')) -ForegroundColor DarkGray
if ($Direction -ne 'push' -and $mainRoot -ne $repoRoot) {
    Write-Host "注意 : 已从工作区 $repoRoot 自动定向到主检出 $mainRoot（保护工作区 output 符号链接）" -ForegroundColor Yellow
}
Write-Host ''

& $RsyncExe @rsyncArgs
exit $LASTEXITCODE
