$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$OutputPath = Join-Path $RepoRoot "outputs\deploy_smoke.json"

& (Join-Path $ScriptDir "run_search.ps1") `
    -Query "graph search systems" `
    -Device "directml" `
    -Format "json" `
    -JsonOut $OutputPath

Write-Host "Smoke check completed: $OutputPath"
