<#
  Starts the Streamlit app. Users double-click run_app.cmd, which runs this script.
  User mode (no .git, e.g. the published OneDrive copy): the Python environment, caches and app
  settings go to %LOCALAPPDATA%\skin_layer_segmentation, so nothing is written to the shared folder.
  Dev mode (git clone): uses the repo's .venv and runs\interface, and pulls the latest code first.
  First run installs Python 3.14 (via winget, after asking), PyTorch (CUDA build if an NVIDIA GPU is
  found, else CPU) and requirements.txt; later runs only reinstall when requirements.txt changes.
#>
$Repo = $PSScriptRoot
Set-Location $Repo

$PythonVersion = "3.14"
$TorchPins = @("torch==2.11.0", "torchvision==0.26.0")  # keep in sync with README
$Port = 8501
$Url = "http://localhost:$Port"

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

$DevMode = Test-Path (Join-Path $Repo ".git")
if ($DevMode) {
    $Venv = Join-Path $Repo ".venv"
} else {
    $Local = Join-Path $env:LOCALAPPDATA "skin_layer_segmentation"
    $Venv = Join-Path $Local ".venv"
    $env:PYTHONPYCACHEPREFIX = Join-Path $Local "pycache"
    $env:SKIN_SEG_STATE_DIR = Join-Path $Local "interface"
}
$VenvPython = Join-Path $Venv "Scripts\python.exe"

$VersionFile = Join-Path $Repo "VERSION"
if (Test-Path $VersionFile) { Step "Skin layer segmentation, version $((Get-Content $VersionFile -Raw).Trim())" }

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Step "The app is already running; opening $Url"
    Start-Process $Url
    exit 0
}

if ($DevMode -and (Get-Command git -ErrorAction SilentlyContinue)) {
    Step "Updating code (git pull)"
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { Write-Warning "git pull failed; starting with the current code" }
}

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $exe = & py "-$PythonVersion" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return "$exe".Trim() }
    }
    $candidates = @(Get-Command python -All -CommandType Application -ErrorAction SilentlyContinue |
        ForEach-Object Source) + (Join-Path $env:LOCALAPPDATA "Programs\Python\Python$($PythonVersion -replace '\.')\python.exe")
    foreach ($exe in $candidates) {
        # WindowsApps\python.exe is the Microsoft Store placeholder, not a real interpreter
        if (-not (Test-Path $exe) -or $exe -like "*\WindowsApps\*") { continue }
        $ver = & $exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and "$ver".Trim() -eq $PythonVersion) { return $exe }
    }
    return $null
}

if (-not (Test-Path $VenvPython)) {
    $BasePython = Find-Python
    if (-not $BasePython) {
        Write-Host "Python $PythonVersion was not found on this computer."
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            Fail "Install Python $PythonVersion from https://www.python.org/downloads/ and run this again."
        }
        $answer = Read-Host "Install Python $PythonVersion now (for your user only)? [Y/n]"
        if ($answer -match '^\s*n') { Fail "Python $PythonVersion is required to run the app." }
        winget install -e --id "Python.Python.$PythonVersion" --scope user --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
        $BasePython = Find-Python
        if (-not $BasePython) { Fail "Python $PythonVersion is still not found; close this window and double-click run_app.cmd again." }
    }
    Step "Creating the Python environment in $Venv (first run only)"
    & $BasePython -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create $Venv" }
}

& $VenvPython -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    $Gpu = $false
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi *> $null
        $Gpu = $LASTEXITCODE -eq 0
    }
    $Index = if ($Gpu) { "cu128" } else { "cpu" }
    Step "Installing PyTorch for $(if ($Gpu) { 'NVIDIA GPU' } else { 'CPU only' }) (first run only, large download)"
    & $VenvPython -m pip install --disable-pip-version-check @TorchPins --index-url "https://download.pytorch.org/whl/$Index"
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed (check the network / proxy and run this again)." }
}

$Requirements = Join-Path $Repo "requirements.txt"
$Stamp = Join-Path $Venv ".requirements.sha256"
$Hash = (Get-FileHash $Requirements -Algorithm SHA256).Hash
if (-not (Test-Path $Stamp) -or (Get-Content $Stamp -Raw).Trim() -ne $Hash) {
    Step "Installing packages from requirements.txt"
    & $VenvPython -m pip install --disable-pip-version-check -r $Requirements
    if ($LASTEXITCODE -ne 0) { Fail "Package install failed (check the network / proxy and run this again)." }
    Set-Content -Path $Stamp -Value $Hash -Encoding ascii
}

# opens the browser once the server is listening (first start can be slow while OneDrive downloads files)
$Opener = "for (`$i = 0; `$i -lt 180; `$i++) { if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { Start-Process '$Url'; break }; Start-Sleep -Seconds 1 }"
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-EncodedCommand",
    ([Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Opener)))

Step "Starting the app at $Url (keep this window open; close it to stop the app)"
# localhost only: the app can read/write the user's files, so don't expose it to the network
$StreamlitArgs = @("--server.address", "localhost", "--server.port", $Port, "--server.headless", "true", "--browser.gatherUsageStats", "false")
if (-not $DevMode) { $StreamlitArgs += @("--server.fileWatcherType", "none") }
& $VenvPython -m streamlit run (Join-Path $Repo "interface\app.py") @StreamlitArgs
exit $LASTEXITCODE
