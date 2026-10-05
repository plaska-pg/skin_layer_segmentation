<#
  Launches predict.py FULLY DETACHED and prevents Windows from sleeping while it
  runs - safe to run from any terminal (including one that gets closed right after),
  since by the time this script returns, python.exe is already running independently.

  Internally this writes a tiny throwaway .cmd into $LogDir and hands that to
  Start-Process - that's the most reliable way found to get a truly detached process
  with stdout+stderr merged into one log file:
    - Start-Process -FilePath powershell.exe -ArgumentList "-File",<this script's path>
      was unreliable here (silently failed to actually launch anything in testing).
    - Start-Process's own -RedirectStandardOutput/-RedirectStandardError can't merge
      both streams into a single file (cmd.exe's "> log 2>&1" can).
  You never need to touch that generated .cmd yourself - just run this .ps1.
#>
$ProjectDir = "C:\Users\plas.ka\OneDrive - Procter and Gamble\Desktop\yolo_skin_seg"
$Python     = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$RawImagesRoot = "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images"
$LogDir     = Join-Path $ProjectDir "runs\background_logs"
$Timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$Log        = Join-Path $LogDir "run_$Timestamp.log"
$ImageExtensions = @(".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs")

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$SourceDirs = @("C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S_EX_B4_D3_MARCH_2025")

$ExistingJobs = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -match '(?i)predict\.py' })
if ($ExistingJobs.Count -gt 0) {
  $Pids = ($ExistingJobs.ProcessId -join ", ")
  throw "predict.py is already running (PID(s): $Pids). Wait for it to finish before launching this batch."
}

$Wrapper = Join-Path $LogDir "_run_all_$Timestamp.generated.cmd"
$Commands = foreach ($SourceDir in $SourceDirs) {
  $Source = $SourceDir
  # NOTE: never delete the existing "<folder>_predicted" output here - --resume below relies on its
  # saved labels/steps.jpg/results.csv to skip images already done (e.g. after a crash mid-batch).
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

# don't let the machine sleep while this is running - it only blocks SLEEP (the display
# can still turn off); revert anytime with: powercfg /requestsoverride PROCESS python.exe
# requires admin rights - silently continue without sleep-prevention if not elevated,
# instead of leaking powercfg's raw permission error to the console
$null = powercfg /requestsoverride PROCESS python.exe SYSTEM 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Output "Sleep prevention enabled for python.exe."
} else {
    Write-Warning "Could not enable sleep prevention (needs admin rights) - continuing without it."
}

$p = Start-Process -FilePath $Wrapper -WindowStyle Hidden -PassThru
Write-Output "Launched $($SourceDirs.Count) folders sequentially from scratch (wrapper PID $($p.Id))."
Write-Output "Each output will be created beside its source as <folder>_predicted."
Write-Output "Log: $Log"


# to kill: taskkill /F /PID 27148