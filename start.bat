@echo off
title EEG Analysis Tool
echo.
echo ============================================
echo        EEG Analysis Tool - Launcher
echo ============================================
echo.

:: ========== CEK & INSTALL PYTHON ==========
python --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Python tidak ditemukan. Menginstall otomatis...
    echo.
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo [ERROR] Gagal install Python otomatis.
        echo         Silakan install manual dari: https://www.python.org/downloads/
        echo         PENTING: Centang "Add Python to PATH" saat instalasi!
        pause
        exit /b 1
    )
    echo.
    echo [OK] Python berhasil diinstall.
    echo [INFO] Menutup dan membuka ulang launcher untuk menerapkan PATH...
    echo.
    pause
    start "" "%~f0"
    exit /b 0
)
echo [OK] Python ditemukan.

:: ========== CEK & INSTALL GIT ==========
git --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Git tidak ditemukan. Menginstall otomatis...
    echo.
    winget install Git.Git --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo [WARNING] Gagal install Git otomatis.
        echo           Auto-update tidak akan berfungsi.
        echo           Install manual dari: https://git-scm.com/downloads
        echo.
    ) else (
        echo.
        echo [OK] Git berhasil diinstall.
        echo [INFO] Menutup dan membuka ulang launcher untuk menerapkan PATH...
        echo.
        pause
        start "" "%~f0"
        exit /b 0
    )
) else (
    echo [OK] Git ditemukan.
)

:: ========== SETUP VENV ==========
if not exist ".venv" (
    echo.
    echo [SETUP] Membuat virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment dibuat.
)

:: Aktivasi venv
call .venv\Scripts\activate.bat

:: Install dependencies jika belum
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo.
    echo [SETUP] Menginstall dependensi (pertama kali, mohon tunggu)...
    pip install -r requirements.txt -q
    echo [OK] Dependensi terinstall.
)

:: ========== JALANKAN ==========
echo.
python launcher.py

pause
