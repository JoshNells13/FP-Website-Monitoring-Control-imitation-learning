$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendDir = Join-Path $ScriptDir "frontend"

Write-Host "[Booster K1 Control Platform] Starting Next.js Frontend on http://localhost:3000 ..." -ForegroundColor Cyan
Set-Location $FrontendDir
npm run dev
