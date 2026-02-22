import urllib.request
import subprocess
import os
import sys

# --- KONFIGURASI ---
LOCAL_VERSION = "1.0"
VERSION_URL = "https://raw.githubusercontent.com/henray404/EEG_web/master/version.txt"

print("=" * 50)
print("       EEG Analysis Tool")
print("=" * 50)
print(f"\n  Versi lokal: {LOCAL_VERSION}")
print("\nMemeriksa pembaruan...")

try:
    response = urllib.request.urlopen(VERSION_URL, timeout=5)
    latest_version = response.read().decode('utf-8').strip()

    if latest_version != LOCAL_VERSION:
        print(f"\n  [UPDATE] Versi baru tersedia: {latest_version}")
        print("  Mengunduh pembaruan otomatis...")
        
        result = subprocess.run(
            ["git", "pull", "origin", "master"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print("  [OK] Pembaruan berhasil diunduh!")
            print("  Menginstall dependensi baru...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        else:
            print(f"  [WARNING] Gagal update otomatis: {result.stderr.strip()}")
            print("  Melanjutkan dengan versi lama...")
    else:
        print("  [OK] Sudah versi terbaru.")

except Exception as e:
    print("\n  [INFO] Tidak ada koneksi internet. Melewati cek update.")

# --- MENJALANKAN STREAMLIT ---
print("\nMemulai server...\n")
os.system(f'"{sys.executable}" -m streamlit run app.py')