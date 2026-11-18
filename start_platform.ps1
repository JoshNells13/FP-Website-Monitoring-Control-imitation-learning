$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Booster K1 (22-DoF) Humanoid Imitation Learning Control Platform" -ForegroundColor Cyan
Write-Host "  Backend API & Socket.IO : http://localhost:5000" -ForegroundColor Green
Write-Host "  Frontend Web Application: http://localhost:3000" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan

Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $ScriptDir "start_backend.ps1")
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $ScriptDir "start_frontend.ps1")
