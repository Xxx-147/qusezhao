param(
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = "Stop"

$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot

Write-Host "Watching repository for non-ignored changes."
Write-Host "Press Ctrl+C to stop."

while ($true) {
    $status = git status --porcelain
    if ($status) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        & "$PSScriptRoot\git_sync.ps1" "chore: auto sync project update $stamp"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
