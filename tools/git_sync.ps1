param(
    [string]$Message = ""
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Title,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Title"
    & $Command
}

$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot

Run-Step "Run tests" {
    python -m pytest
}

Run-Step "Stage non-ignored project changes" {
    git add -A
}

git diff --cached --quiet
$hasStagedChanges = $LASTEXITCODE -ne 0

if ($hasStagedChanges) {
    if ([string]::IsNullOrWhiteSpace($Message)) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $Message = "chore: sync project update $stamp"
    }

    Run-Step "Commit changes" {
        git commit -m $Message
    }
} else {
    Write-Host ""
    Write-Host "No staged changes to commit."
}

Run-Step "Push main to GitHub" {
    git push
}

Write-Host ""
Write-Host "GitHub sync complete."
