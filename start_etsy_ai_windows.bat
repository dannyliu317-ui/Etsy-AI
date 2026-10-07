@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo        Etsy AI - V1 to V10 Launcher
echo ==========================================
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=py -3"
) else (
    set "PYTHON=python"
)

%PYTHON% --version
if %errorlevel% neq 0 (
    echo.
    echo Python 3 is not installed or is not available in PATH.
    echo Please install Python 3.11 and run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [1/3] Creating virtual environment...
    %PYTHON% -m venv .venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo.
echo [2/3] Installing / updating dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to update pip.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements.txt.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m pip install opencv-python-headless
if %errorlevel% neq 0 (
    echo Failed to install OpenCV.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Etsy AI...
echo.
echo Browser address: http://localhost:8501
echo Keep this window open while using Etsy AI.
echo Close this window to stop the app.
echo.

.venv\Scripts\python.exe -m streamlit run app.py

echo.
echo Etsy AI has stopped.
pause
