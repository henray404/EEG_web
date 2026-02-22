#!/bin/bash
cd "$(dirname "$0")"

# Cek Python
if ! command -v python3 &> /dev/null; then
    echo "Python tidak ditemukan. Menginstall..."
    if command -v brew &> /dev/null; then
        brew install python@3.12
    else
        echo "Install Homebrew dulu: https://brew.sh"
        echo "Lalu jalankan ulang start.sh"
        exit 1
    fi
fi

# Jalankan GUI launcher
python3 launcher.py
