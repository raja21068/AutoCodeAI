# eval/wait_and_run.ps1
# ---------------------
# Wait until the LLM provider will accept requests, then start the grid.
#
# The grid is blocked on an exhausted provider balance, not on anything in
# the pipeline: images are cached, the environment works, and the
# generate-then-grade loop is proven. Rather than leave that to be noticed
# and restarted by hand, poll cheaply until the provider answers and then
# hand off to the overnight driver.
#
# The probe costs one token and is the same check the driver's preflight
# runs, so a run never starts into a provider that will refuse it.
#
# Usage:
#     powershell -File eval/wait_and_run.ps1
#     powershell -File eval/wait_and_run.ps1 -IntervalMinutes 10 -MaxHours 48

param(
    [int]$IntervalMinutes = 10,
    [int]$MaxHours        = 48,
    [string]$Subset       = "eval/subsets/lite50.json",
    # Passed through to the grid. Must name the provider that .env is
    # configured for: the runner sets this per call and it overrides the
    # per-role defaults, so a stale value here asks one provider for another
    # provider's model and every request is rejected.
    [string]$Model        = "deepseek/deepseek-chat",
    # Groq's free tier throttles hard on requests and tokens per minute.
    # Three concurrent instances is fine against DeepSeek and produces a wall
    # of 429s against Groq, so the caller sets this to match the provider.
    [int]$Workers         = 3
)

$ErrorActionPreference = "Continue"
$Repo = "D:\AutoResearch\AutoCodeAI"
$Log  = Join-Path $Repo "eval\results\wait_and_run.log"
$Py   = "/opt/agentforge-venv/bin/python"

New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

function Test-Provider {
    wsl.exe -d Ubuntu -u root -- bash -c `
        "cd /mnt/d/AutoResearch/AutoCodeAI && $Py -m eval.check_provider" 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}

Say "waiting for provider; probing every $IntervalMinutes min, giving up after $MaxHours h"
Say "will run with model=$Model workers=$Workers subset=$Subset"

$deadline = (Get-Date).AddHours($MaxHours)
$attempt  = 0

while ((Get-Date) -lt $deadline) {
    $attempt++
    if (Test-Provider) {
        Say "provider available after $attempt probe(s) - starting the grid (model=$Model, workers=$Workers)"
        & powershell -ExecutionPolicy Bypass -File (Join-Path $Repo "eval\overnight_run.ps1") `
            -Subset $Subset -Model $Model -Workers $Workers
        Say "grid driver returned; wait_and_run done"
        exit 0
    }
    if ($attempt -eq 1 -or $attempt % 6 -eq 0) {
        Say "still unavailable (probe $attempt) - top up the provider account to start"
    }
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}

Say "gave up after $MaxHours h without the provider becoming available"
exit 1
