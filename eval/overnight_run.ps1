# eval/overnight_run.ps1
# -----------------------
# Unattended driver: prefetch -> run -> grade -> regenerate tables -> recompile.
#
# Runs the experimental grid on a fixed instance subset, one configuration at a
# time, in priority order. After EVERY graded configuration it regenerates the
# manuscript tables and rebuilds the PDF, so the paper is in a valid, honest
# state at any moment the process is interrupted: configurations that finished
# show real numbers, configurations that did not still show TBD placeholders.
# Nothing is ever written into a table that a run did not produce.
#
# Order matters. The four factorial arms come first because they carry the
# paper's central claim (role decomposition vs. execution scheduling); the
# reference arms follow; ablations last. A night that only gets partway
# through therefore still yields the rows the argument depends on.
#
# Usage:
#     powershell -File eval/overnight_run.ps1
#     powershell -File eval/overnight_run.ps1 -Subset eval/subsets/lite50.json

param(
    [string]$Subset  = "eval/subsets/lite50.json",
    [string]$Model   = "deepseek/deepseek-chat",
    [int]$Workers    = 3,      # 3 x 4GB container cap against 13GB available
    [int]$GradeWorkers = 2
)

$ErrorActionPreference = "Continue"
$Repo   = "D:\AutoResearch\AutoCodeAI"
$Py     = "/opt/agentforge-venv/bin/python"
$LogDir = Join-Path $Repo "eval\results"
$Log    = Join-Path $LogDir "overnight.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

function Invoke-Wsl($cmd, $LogFile) {
    # wsl.exe, not wsl: PowerShell resolves command names case-insensitively
    # and prefers functions over external programs, so a function named Wsl
    # calling "wsl" calls itself until the call stack overflows.
    #
    # The output must be diverted to a file rather than left on the output
    # stream. A PowerShell function returns everything written to that
    # stream, so letting the command's stdout through makes the caller's
    # "$code = Invoke-Wsl ..." an array of every log line with the exit code
    # appended -- and "$code -ne 0" is then true no matter what happened.
    # That silently skipped grading for every configuration in the first run.
    if ($LogFile) {
        wsl.exe -d Ubuntu -u root -- bash -c "cd /mnt/d/AutoResearch/AutoCodeAI && $cmd" `
            2>&1 | Out-File -FilePath $LogFile -Append -Encoding utf8
    } else {
        wsl.exe -d Ubuntu -u root -- bash -c "cd /mnt/d/AutoResearch/AutoCodeAI && $cmd" `
            2>&1 | Out-Null
    }
    return $LASTEXITCODE
}

# Priority order: factorial first, then reference arms, then ablations.
$Configs = @(
    "agentforge",
    "single_forced",
    "multi_optional",
    "single_optional",
    "react",
    "single_call",
    "no_critic",
    "no_debugger",
    "no_tester",
    "no_planner",
    "no_retrieval",
    "no_generated_tests"
)

function Rebuild-Paper {
    # Tables are regenerated from artifacts on disk; a configuration with no
    # official report emits placeholders rather than a blank row.
    Push-Location $Repo
    python -m eval.export_tables --results_dir eval/results --out AgentForge/tables 2>&1 |
        ForEach-Object { Say "  tables: $_" }
    Pop-Location

    # Build into a scratch directory, then copy the result out.
    #
    # Building straight onto AgentForge_Polished_main.pdf fails outright when
    # that file is open in a viewer -- pdflatex cannot write it and aborts.
    # Over a long unattended run that would mean one open PDF window silently
    # costs every rebuild for the rest of the night. The build always
    # succeeds here; only the final copy can fail, and if it does the fresh
    # PDF is still on disk under build/.
    Push-Location (Join-Path $Repo "AgentForge")

    # Build under a different job name in the SAME directory, then copy.
    #
    # -output-directory would be the obvious choice, but it puts the .aux and
    # .bbl somewhere pdflatex does not search, so every citation silently
    # resolves to [?]. Keeping all auxiliary files beside the source
    # reproduces the known-good build exactly; only the final copy can fail,
    # which is the one operation that touches the possibly-locked PDF.
    $job = "paper_build"
    # Build the flag as its own string. PowerShell does not expand $job inside
    # a bare -jobname=$job argument; it reaches pdflatex literally and the
    # build lands in files actually named '$job.pdf'.
    $jobArg = "-jobname=$job"

    pdflatex -interaction=nonstopmode $jobArg AgentForge_Polished_main.tex 2>&1 | Out-Null
    bibtex $job 2>&1 | Out-Null
    pdflatex -interaction=nonstopmode $jobArg AgentForge_Polished_main.tex 2>&1 | Out-Null
    $out = pdflatex -interaction=nonstopmode $jobArg AgentForge_Polished_main.tex 2>&1

    $written = $out | Select-String -Pattern "Output written"
    if ($written) {
        $undef = (Select-String -Path "$job.log" `
                  -Pattern 'Citation .* undefined|Reference .* undefined' -ErrorAction SilentlyContinue)
        $n = if ($undef) { $undef.Count } else { 0 }
        Say "  paper: built ($($written.Line.Trim())); undefined refs: $n"
        try {
            Copy-Item "$job.pdf" "AgentForge_Polished_main.pdf" -Force -ErrorAction Stop
        } catch {
            Say "  paper: AgentForge_Polished_main.pdf is locked (open in a viewer?); fresh build is at AgentForge\$job.pdf"
        }
    } else {
        Say "  paper: BUILD FAILED - see AgentForge\$job.log"
    }
    Pop-Location
}

Say "=== overnight run start | subset=$Subset model=$Model workers=$Workers ==="

# ---------------------------------------------------------------------------
# 1. Images. Everything downstream blocks on this, and cached images are
#    skipped instantly, so it is safe to re-enter.
# ---------------------------------------------------------------------------
Say "prefetching images for $Subset"
$code = Invoke-Wsl "$Py -u -m eval.prefetch_images --instance_file $Subset --workers 3" `
                   (Join-Path $LogDir "prefetch.log")
if ($code -ne 0) {
    Say "prefetch reported failures (exit $code); continuing - missing images surface per instance"
}
Say "prefetch done"

# ---------------------------------------------------------------------------
# 2. Grid. Each configuration is generated, graded, and folded into the paper
#    before the next one starts.
# ---------------------------------------------------------------------------
foreach ($cfg in $Configs) {
    $cfgLog = Join-Path $LogDir "$cfg.run.log"

    Say "--- $cfg : generating ---"
    $code = Invoke-Wsl "$Py -u -m eval.swebench_runner --split lite --config $cfg --model $Model --run_id $cfg --output_dir eval/results --instance_file $Subset --workers $Workers --resume" $cfgLog
    if ($code -ne 0) {
        Say "$cfg : generation exited $code - skipping grade (see $cfg.run.log)"
        continue
    }

    # A provider billing failure does not stop the runner: each instance
    # records an error and the run still "succeeds". Grading that would
    # report a resolve rate over instances that never ran, which measures
    # the account balance rather than the system. Check before spending
    # hours grading a run that cannot mean anything.
    $billing = Select-String -Path $cfgLog -Pattern 'Insufficient Balance|RateLimitError|AuthenticationError' -ErrorAction SilentlyContinue
    if ($billing) {
        Say "$cfg : ABORTING GRID - provider refused requests ($($billing.Count) occurrences); results would be invalid"
        Say "        top up the provider account, then re-run; --resume keeps completed instances"
        break
    }

    Say "--- $cfg : grading ---"
    $code = Invoke-Wsl "$Py -u -m eval.run_official_eval --run_id $cfg --max_workers $GradeWorkers" $cfgLog
    if ($code -ne 0) {
        Say "$cfg : grading exited $code"
        continue
    }

    Say "--- $cfg : rebuilding paper ---"
    Rebuild-Paper
    Say "$cfg : complete"
}

Say "=== overnight run finished ==="
