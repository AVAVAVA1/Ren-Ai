@echo off
setlocal
set "ROOT=%~dp0"
set "VENV_SERVER=%ROOT%server\.venv\Scripts\python.exe"
set "VENV_ROOT=%ROOT%.venv\Scripts\python.exe"
echo Starting Backend Server...
cd /d "%ROOT%server"
if exist "%VENV_SERVER%" (
  echo Using venv: %VENV_SERVER%
  "%VENV_SERVER%" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
) else if exist "%VENV_ROOT%" (
  echo Using venv: %VENV_ROOT%
  "%VENV_ROOT%" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
) else (
  echo No .venv found, using python on PATH
  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
)
