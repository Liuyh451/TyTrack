$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$LogDir = Join-Path $ScriptDir "monitor_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$StdoutLog = Join-Path $LogDir "monitor_runner_stdout.log"
$StderrLog = Join-Path $LogDir "monitor_runner_stderr.log"

python monitor_forecast_availability.py --loop --interval 3600 1>> $StdoutLog 2>> $StderrLog
