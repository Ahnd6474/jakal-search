$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    py -3.12 -m venv (Join-Path $RepoRoot ".venv")
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -e "$RepoRoot[models,directml]"

Write-Host "Installed deployment environment at $RepoRoot"
