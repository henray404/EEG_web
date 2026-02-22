@echo off
title EEG Analysis Tool

:: Cek Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Python tidak ditemukan. Menginstall...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] Install Python gagal. Download manual: https://python.org
        pause
        exit /b 1
    )
    echo [OK] Python terinstall. Silakan jalankan start.bat lagi.
    pause
    exit /b 0
)

:: Jalankan GUI launcher
pythonw launcher.py
