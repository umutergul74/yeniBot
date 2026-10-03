param(
    [ValidateSet('A','B','C','D')][string]$Experiment = 'A',
    [string]$Config = 'configs/advisor_ablation.yaml'
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$pythonExecutable = (Get-Command python -ErrorAction Stop).Source
$logDirectory = Join-Path $workspace 'output/advisor_experiments/logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outputLog = Join-Path $logDirectory "$Experiment-$runStamp.stdout.log"
$errorLog = Join-Path $logDirectory "$Experiment-$runStamp.stderr.log"
# All command arguments are fixed tokens or validated values; paths below are relative.
if ($Config -notmatch '^[a-zA-Z0-9_./-]+$') { throw 'Config path must be a simple relative workspace path.' }
$worker = Start-Process -FilePath $pythonExecutable -WorkingDirectory $workspace -WindowStyle Hidden -PassThru `
    -ArgumentList @('-u','-m','yenibot.training.advisor','run','--config',$Config,'--experiment',$Experiment) `
    -RedirectStandardOutput $outputLog -RedirectStandardError $errorLog
$launchRecord = @{
    process_id = $worker.Id
    experiment = $Experiment
    config = $Config
    started_at = (Get-Date).ToString('o')
    stdout = $outputLog
    stderr = $errorLog
    resume = "powershell -NoProfile -File scripts/run_advisor_experiment.ps1 -Experiment $Experiment -Config $Config"
}
$launchRecord | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $logDirectory "$Experiment-$runStamp.launch.json")
Write-Output "Started process $($worker.Id). Log: $outputLog"
Write-Output "Same command resumes saved epochs and skips completed folds."
