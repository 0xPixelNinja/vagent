@echo off
setlocal EnableDelayedExpansion

REM Get the directory of this script
set "SCRIPT_DIR=%~dp0"
REM Go up one level to project root
cd /d "%SCRIPT_DIR%.."
set "PROJECT_ROOT=%CD%"

set "VENV_DIR=%PROJECT_ROOT%\.venv"
set "NVIDIA_DIR=%VENV_DIR%\Lib\site-packages\nvidia"

REM Add NVIDIA DLL folders (cuDNN, cuBLAS, etc.) to PATH
REM This is required on Windows so Python extensions can locate required DLLs.
if exist "%NVIDIA_DIR%" (
    for /d %%p in ("%NVIDIA_DIR%\*") do (
        if exist "%%p\bin" (
            set "PATH=%%p\bin;!PATH!"
        )
    )
)

REM Enable latency timing logs by default (override by setting VAGENT_LATENCY beforehand)
if not defined VAGENT_LATENCY set VAGENT_LATENCY=1

REM Activate venv and run agent
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    echo Virtual environment not found at %VENV_DIR%
    exit /b 1
)

python -m vagent.agent start
