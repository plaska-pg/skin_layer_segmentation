<#
  Launches predict.py FULLY DETACHED for the previously identified Raw images
  folders that lack results.csv, and prevents Windows from sleeping while it runs.
  Before launch, each target is checked recursively and skipped if results.csv
  has appeared since the list was created.
#>
$ProjectDir = $PSScriptRoot
$Python     = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$RawImagesRoot = "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images"
$LogDir     = Join-Path $ProjectDir "runs\background_logs"
$Timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$Log        = Join-Path $LogDir "run_missing_results_$Timestamp.log"
$SourceNames = @(
  "S-Exp_10_24_D3_H_E",
  "S-EXP_Clobertasol_Pre-Treatment_03_25_D4_TIFF",
  "S-EX_SEP_2024_D2_H_E",
  "S_ex1-24tRA_Clob",
  "S_Exp 6.24 D5",
  "S_EXP_SEPT_24_D1_H_E",
  "S_Exp_SEP_2024_D3_H_E",
  "S_EX_NOV_24_D1_H_E",
  "S_EX_NOV_24_D2_H_E"
)

if (-not (Test-Path -LiteralPath $RawImagesRoot -PathType Container)) {
  throw "Raw images directory not found: $RawImagesRoot"
}

$SourceDirs = @()
foreach ($SourceName in $SourceNames) {
  $SourceDir = Join-Path $RawImagesRoot $SourceName
  if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
    Write-Warning "Target folder not found; skipping: $SourceDir"
    continue
  }

  $ResultsCsv = Get-ChildItem -LiteralPath $SourceDir -File -Filter "results.csv" -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($null -eq $ResultsCsv) {
    $SourceDirs += $SourceDir
  } else {
    Write-Output "Already has results.csv; skipping: $SourceDir"
  }
}

if ($SourceDirs.Count -eq 0) {
  Write-Output "None of the target folders are currently missing results.csv. Nothing launched."
  exit 0
}

$ExistingJobs = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -match '(?i)predict\.py' })
if ($ExistingJobs.Count -gt 0) {
  $Pids = ($ExistingJobs.ProcessId -join ", ")
  throw "predict.py is already running (PID(s): $Pids). Wait for it to finish before launching this batch."
}

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Wrapper = Join-Path $LogDir "_run_missing_results_$Timestamp.generated.cmd"
$Commands = foreach ($SourceDir in $SourceDirs) {
  $Source = $SourceDir
  @"
echo.>> "$Log"
echo ===== %date% %time% - $Source ===== >> "$Log"
"$Python" -u predict.py --source "$Source" >> "$Log" 2>&1
echo [%date% %time%] Exit code: %errorlevel% >> "$Log"
"@
}

@"
@echo off
cd /d "$ProjectDir"
echo Started %date% %time% > "$Log"
$($Commands -join "`r`n")
echo.>> "$Log"
echo Finished %date% %time% >> "$Log"
"@ | Set-Content -Path $Wrapper -Encoding ASCII

$null = powercfg /requestsoverride PROCESS python.exe SYSTEM 2>&1
if ($LASTEXITCODE -eq 0) {
  Write-Output "Sleep prevention enabled for python.exe."
} else {
  Write-Warning "Could not enable sleep prevention (needs admin rights) - continuing without it."
}

$p = Start-Process -FilePath $Wrapper -WindowStyle Hidden -PassThru
Write-Output "Launched $($SourceDirs.Count) folders sequentially (wrapper PID $($p.Id))."
Write-Output "Each output is written inside its source as <folder>\predicted_<folder>."
Write-Output "Log: $Log"