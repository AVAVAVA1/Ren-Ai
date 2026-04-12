# Start backend: prefer server/.venv, then repo-root .venv, else PATH python
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvServer = Join-Path $root "server\.venv\Scripts\python.exe"
$venvRoot = Join-Path $root ".venv\Scripts\python.exe"
$venvPy = $null
if (Test-Path $venvServer) { $venvPy = $venvServer }
elseif (Test-Path $venvRoot) { $venvPy = $venvRoot }

Set-Location (Join-Path $root "server")
if ($venvPy) {
    Write-Host "Using venv: $venvPy"
    & $venvPy -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
} else {
    Write-Host 'No .venv found; using python on PATH. Create: cd server; python -m venv .venv; pip install -r requirements.txt'
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}
