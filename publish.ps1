<#
  Publishes the last commit (HEAD) as a clean copy for users, e.g. into the shared OneDrive folder:
    .\publish.ps1 -Destination "C:\Users\<you>\OneDrive - Procter and Gamble\<shared folder>"
  Users then double-click run_app.cmd there. Only committed files are published (no .git, .venv,
  runs, datasets). The destination is mirrored - files no longer in the repo are deleted there - so it
  must be empty or a previous publish (marked by .skin_seg_publish).
#>
param([Parameter(Mandatory = $true)][string]$Destination)
$ErrorActionPreference = "Stop"
$Destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination).TrimEnd("\")
Set-Location $PSScriptRoot
$Marker = ".skin_seg_publish"

if ($Destination -like "$PSScriptRoot*") { throw "Destination must be outside the repo." }
if (git status --porcelain --untracked-files=no) { throw "There are uncommitted changes; commit them first (only HEAD is published)." }
if ((Test-Path $Destination) -and (Get-ChildItem -Force $Destination | Select-Object -First 1) -and
    -not (Test-Path (Join-Path $Destination $Marker))) {
    throw "$Destination is not empty and was not created by publish.ps1; refusing to overwrite it."
}

$Commit = (git rev-parse --short HEAD).Trim()
$Stage = Join-Path $env:TEMP "skin_seg_publish_$Commit"
$Tar = "$Stage.tar"
Remove-Item -Recurse -Force $Stage, $Tar -ErrorAction SilentlyContinue
New-Item -ItemType Directory $Stage | Out-Null
git archive --format=tar -o $Tar HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
tar -xf $Tar -C $Stage
if ($LASTEXITCODE -ne 0) { throw "tar extract failed" }
"$Commit ($(git log -1 --format=%cs HEAD))" | Set-Content (Join-Path $Stage "VERSION") -Encoding ascii
New-Item -ItemType File (Join-Path $Stage $Marker) | Out-Null

# desktop.ini is OneDrive's own; /XF also keeps /MIR from deleting it
robocopy $Stage $Destination /MIR /XF desktop.ini /NFL /NDL /NJH /NP
$Code = $LASTEXITCODE
Remove-Item -Recurse -Force $Stage, $Tar
if ($Code -ge 8) { throw "robocopy failed (exit code $Code)" }
Write-Host "Published $Commit to $Destination" -ForegroundColor Green
exit 0
