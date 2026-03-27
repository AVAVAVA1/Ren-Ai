# Start backend (prefer repo-root .venv)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    Write-Host "Using venv: $venvPy"
    Set-Location (Join-Path $root "server")
    & $venvPy -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
} else {
    Write-Host 'No .venv found; using python on PATH. Create: python -m venv .venv then pip install -r server/requirements.txt'
    Set-Location (Join-Path $root "server")
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}
