param(
    [string]$Destination = "dist\jakal-search-deploy"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$TargetRoot = if ([System.IO.Path]::IsPathRooted($Destination)) { $Destination } else { Join-Path $RepoRoot $Destination }
$ModelsDir = Join-Path $RepoRoot "outputs\models"

$FilesToCopy = @(
    "pyproject.toml",
    "README.md"
)

$ModelFiles = @(
    "trust_head_sentence-transformers__all-MiniLM-L6-v2.pt",
    "trust_head_sentence-transformers__all-MiniLM-L6-v2.json",
    "claim_falsehood_head_sentence-transformers__all-MiniLM-L6-v2.pt",
    "claim_falsehood_head_sentence-transformers__all-MiniLM-L6-v2.json",
    "branch_head.pt",
    "branch_head.json",
    "topic_reranker_head.pt",
    "topic_reranker_head.json"
)

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $TargetRoot "outputs\models") | Out-Null

foreach ($RelativePath in $FilesToCopy) {
    Copy-Item -Path (Join-Path $RepoRoot $RelativePath) -Destination (Join-Path $TargetRoot $RelativePath) -Force
}

Copy-Item -Path (Join-Path $RepoRoot "jakal_search") -Destination (Join-Path $TargetRoot "jakal_search") -Recurse -Force
Copy-Item -Path (Join-Path $RepoRoot "deploy") -Destination (Join-Path $TargetRoot "deploy") -Recurse -Force

foreach ($ModelFile in $ModelFiles) {
    Copy-Item -Path (Join-Path $ModelsDir $ModelFile) -Destination (Join-Path $TargetRoot "outputs\models\$ModelFile") -Force
}

$Manifest = @{
    created_from = $RepoRoot
    models = @{
        trust = "outputs/models/trust_head_sentence-transformers__all-MiniLM-L6-v2.pt"
        claim_falsehood = "outputs/models/claim_falsehood_head_sentence-transformers__all-MiniLM-L6-v2.pt"
        branch = "outputs/models/branch_head.pt"
        topic_reranker = "outputs/models/topic_reranker_head.pt"
    }
    runtime = @{
        install = "deploy/install_windows.ps1"
        run = "deploy/run_search.ps1"
        verify = "deploy/verify_bundle.ps1"
    }
}

$Manifest | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $TargetRoot "deploy\bundle.manifest.json") -Encoding UTF8

Write-Host "Deployment bundle exported to $TargetRoot"
