$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ScriptDir "..\human-imitation-learning\.venv\Scripts\python.exe"
$BackendScript = Join-Path $ScriptDir "backend\run.py"

Write-Host "[Booster K1 Control Platform] Starting Flask + Socket.IO Backend on http://localhost:5000 ..." -ForegroundColor Cyan
& $PythonExe $BackendScript
