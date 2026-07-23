param(
    [string]$PythonPath = "",
    [switch]$SkipDocker
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if ($PythonPath) { $PythonPath } else { (Get-Command python).Source }
$scripts = @(
    "ingestion_PN.py",
    "ingestion_PG.py",
    "ingestion_CH.py",
    "ingestion_QD.py",
    "ingestion_LanceDB.py",
    "ingestion_SQLite.py"
)

Write-Host "Using Python: $python"
Write-Host "Repo: $repoRoot"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}

if (-not $SkipDocker) {
    Write-Host "Starting Docker services for PostgreSQL and Qdrant..."
    docker compose up -d postgres qdrant
}

Write-Host "Checking ingestion dependencies..."
& $python -c "from langchain_qdrant import QdrantVectorStore; import lancedb; print('Qdrant and LanceDB deps OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Missing dependencies. Run: & '$python' -m pip install -r requirements.txt --upgrade"
}

foreach ($script in $scripts) {
    $scriptPath = Join-Path $repoRoot $script
    if (-not (Test-Path -LiteralPath $scriptPath)) {
        Write-Warning "Skipping missing script: $script"
        continue
    }

    $command = @"
`$env:PYTHONIOENCODING='utf-8'; Set-Location -LiteralPath '$repoRoot'; Write-Host 'Starting: $script' -ForegroundColor Cyan; & '$python' '$scriptPath'; if (`$LASTEXITCODE -ne 0) { Write-Host ''; Write-Host 'FAILED: $script' -ForegroundColor Red; Read-Host 'Press Enter to close' } else { Write-Host ''; Write-Host 'COMPLETED: $script' -ForegroundColor Green; Read-Host 'Press Enter to close' }
"@

    Start-Process powershell.exe -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $command)
}

Write-Host "Started $($scripts.Count) ingestion terminal(s) in parallel."
