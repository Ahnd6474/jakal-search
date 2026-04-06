param(
    [Parameter(Mandatory = $true)]
    [string]$Query,
    [string]$Device = "directml",
    [ValidateSet("report", "urls", "tree", "json")]
    [string]$Format = "report",
    [string]$JsonOut = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ModelsDir = Join-Path $RepoRoot "outputs\models"

if (-not (Test-Path $PythonExe)) {
    throw "Python venv not found. Run deploy\install_windows.ps1 first."
}

$ArgsList = @(
    "-m", "jakal_search",
    $Query,
    "--device", $Device,
    "--trust-model", (Join-Path $ModelsDir "trust_head_sentence-transformers__all-MiniLM-L6-v2.pt"),
    "--claim-model", (Join-Path $ModelsDir "claim_falsehood_head_sentence-transformers__all-MiniLM-L6-v2.pt"),
    "--branch-model", (Join-Path $ModelsDir "branch_head.pt"),
    "--topic-reranker-model", (Join-Path $ModelsDir "topic_reranker_head.pt"),
    "--format", $Format
)

if ($JsonOut -ne "") {
    $ArgsList += @("--json-out", $JsonOut)
}

& $PythonExe @ArgsList
