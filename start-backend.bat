@echo off
setlocal
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
echo Starting Backend Server...
cd /d "%ROOT%server"
if exist "%VENV_PY%" (
  echo Using venv: %VENV_PY%
  "%VENV_PY%" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
) else (
  echo No .venv found, using python on PATH
  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
)
