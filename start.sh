#!/bin/bash

echo ""
echo "============================================"
echo "       EEG Analysis Tool - Launcher"
echo "============================================"
echo ""

# Pindah ke direktori script
cd "$(dirname "$0")"

# ========== CEK PYTHON ==========
if ! command -v python3 &> /dev/null; then
    echo "[INFO] Python tidak ditemukan."
    echo "       Menginstall via Homebrew..."
    
    # Install Homebrew jika belum ada
    if ! command -v brew &> /dev/null; then
        echo "[INFO] Menginstall Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew install python@3.12
    echo "[OK] Python terinstall."
fi
echo "[OK] Python ditemukan: $(python3 --version)"

# ========== CEK GIT ==========
if ! command -v git &> /dev/null; then
    echo "[INFO] Git tidak ditemukan. Menginstall..."
    xcode-select --install 2>/dev/null || brew install git
fi
echo "[OK] Git ditemukan."

# ========== SETUP VENV ==========
if [ ! -d ".venv" ]; then
    echo ""
    echo "[SETUP] Membuat virtual environment..."
    python3 -m venv .venv
    echo "[OK] Virtual environment dibuat."
fi

# Aktivasi venv
source .venv/bin/activate

# Install dependencies jika belum
if ! pip show streamlit &> /dev/null; then
    echo ""
    echo "[SETUP] Menginstall dependensi (pertama kali, mohon tunggu)..."
    pip install -r requirements.txt -q
    echo "[OK] Dependensi terinstall."
fi

# ========== JALANKAN ==========
echo ""
python launcher.py
