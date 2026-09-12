@echo off
title FOVEAX 2.5D Real-Time PyQt5 Dashboard
cd /d "%~dp0"
echo ========================================================
echo  Launching FOVEAX 2.5D PyQt5 + Open3D Real-Time Dashboard
echo ========================================================

if exist "..\foveax_venv\Scripts\python.exe" (
    "..\foveax_venv\Scripts\python.exe" src\13_realtime_dashboard.py %*
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" src\13_realtime_dashboard.py %*
) else (
    python src\13_realtime_dashboard.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Dashboard exited with error code %ERRORLEVEL%.
    pause
)
