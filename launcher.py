"""
EEG Analysis Tool - GUI Launcher
Tkinter-based launcher with auto-setup, update check, and one-click start.
"""
import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import urllib.request
import ssl
import os
import sys
import re
import webbrowser
import logging
import traceback

# --- FILE LOGGING ---
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launcher.log")
logging.basicConfig(
    filename=LOG_FILE, level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("=" * 50)
logging.info("Launcher started")

# --- KONFIGURASI ---
LOCAL_VERSION = "1.9"
VERSION_URL = "https://raw.githubusercontent.com/henray404/EEG_web/master/version.txt"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(APP_DIR, ".venv")
CHANGELOG_FILE = os.path.join(APP_DIR, "CHANGELOG.md")

if sys.platform == "win32":
    VENV_PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
    VENV_PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")
    VENV_PIP = os.path.join(VENV_DIR, "bin", "pip")


def _read_current_changelog():
    """Baca changelog untuk versi saat ini dari CHANGELOG.md."""
    if not os.path.exists(CHANGELOG_FILE):
        return []
    try:
        with open(CHANGELOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        # Cari section untuk versi saat ini
        pattern = rf"## v{re.escape(LOCAL_VERSION)}.*?\n(.*?)(?=\n## v|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            lines = match.group(1).strip().split("\n")
            return [ln.lstrip("- ").strip() for ln in lines if ln.strip().startswith("-")]
    except Exception:
        pass
    return []


class LauncherApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EEG Analysis Tool")
        self.root.geometry("540x580")
        self.root.resizable(False, False)
        self.root.configure(bg="#0B1120")

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 540) // 2
        y = (self.root.winfo_screenheight() - 580) // 2
        self.root.geometry(f"540x580+{x}+{y}")

        self._build_ui()
        self.server_process = None
    
    def _build_ui(self):
        bg = "#0B1120"
        card = "#111827"
        accent = "#1E88E5"
        text = "#F1F5F9"
        muted = "#94A3B8"

        # Header
        header = tk.Frame(self.root, bg=bg)
        header.pack(fill="x", padx=20, pady=(20, 5))

        tk.Label(header, text="EEG Analysis Tool", font=("Segoe UI", 18, "bold"),
                 bg=bg, fg=text).pack(anchor="w")
        tk.Label(header, text=f"Versi {LOCAL_VERSION}", font=("Segoe UI", 10),
                 bg=bg, fg=muted).pack(anchor="w")

        # Changelog card (scrollable, fixed height)
        changelog_items = _read_current_changelog()
        if changelog_items:
            cl_frame = tk.Frame(self.root, bg=card, highlightbackground="#1E293B",
                                highlightthickness=1)
            cl_frame.pack(fill="x", padx=20, pady=(10, 0))

            tk.Label(cl_frame, text=f"Pembaruan v{LOCAL_VERSION}",
                     font=("Segoe UI", 10, "bold"),
                     bg=card, fg=accent).pack(anchor="w", padx=12, pady=(8, 2))

            changelog_text = "\n".join(f"  - {item}" for item in changelog_items)
            cl_textbox = tk.Text(cl_frame, bg=card, fg=muted,
                                 font=("Consolas", 8), bd=0,
                                 highlightthickness=0, wrap="word",
                                 height=5, state="normal")
            cl_textbox.insert("1.0", changelog_text)
            cl_textbox.configure(state="disabled")
            cl_textbox.pack(fill="x", padx=12, pady=(0, 8))

        # Status card
        status_frame = tk.Frame(self.root, bg=card, highlightbackground="#1E293B",
                                highlightthickness=1)
        status_frame.pack(fill="both", expand=True, padx=20, pady=10)

        tk.Label(status_frame, text="Status", font=("Segoe UI", 10, "bold"),
                 bg=card, fg=muted).pack(anchor="w", padx=12, pady=(10, 0))

        self.log_text = tk.Text(status_frame, bg=card, fg=text,
                                font=("Consolas", 9), bd=0,
                                highlightthickness=0, wrap="word",
                                state="disabled", height=8)
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(5, 10))

        # Progress bar
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=card, background=accent,
                        bordercolor=card, lightcolor=accent,
                        darkcolor=accent)

        self.progress = ttk.Progressbar(self.root, mode="determinate",
                                         style="Custom.Horizontal.TProgressbar",
                                         maximum=100)
        self.progress.pack(fill="x", padx=20, pady=(0, 10))

        # Buttons
        btn_frame = tk.Frame(self.root, bg=bg)
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))

        self.start_btn = tk.Button(
            btn_frame, text="Mulai Aplikasi", font=("Segoe UI", 11, "bold"),
            bg=accent, fg="white", activebackground="#1565C0",
            activeforeground="white", bd=0, padx=20, pady=8,
            cursor="hand2", command=self._on_start,
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.quit_btn = tk.Button(
            btn_frame, text="Keluar", font=("Segoe UI", 11),
            bg="#1E293B", fg=text, activebackground="#374151",
            activeforeground=text, bd=0, padx=20, pady=8,
            cursor="hand2", command=self._on_quit,
        )
        self.quit_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

    # ---- Logging ----
    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()
        logging.info(msg)

    def set_progress(self, value):
        self.progress["value"] = value
        self.root.update_idletasks()

    # ---- Setup steps ----
    def _run_setup(self):
        self.start_btn.configure(state="disabled", bg="#374151")

        # Step 1: Check venv
        self.log("Memeriksa virtual environment...")
        self.set_progress(10)
        if not os.path.exists(VENV_PYTHON):
            self.log("Membuat virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR],
                           cwd=APP_DIR, capture_output=True)
            self.log("  Virtual environment dibuat.")
        else:
            self.log("  Virtual environment tersedia.")
        self.set_progress(25)

        # Step 2: Check dependencies
        self.log("Memeriksa dependensi...")
        result = subprocess.run([VENV_PIP, "show", "streamlit"],
                                capture_output=True, text=True, cwd=APP_DIR)
        if result.returncode != 0:
            self.log("Menginstall dependensi (mohon tunggu)...")
            proc = subprocess.run(
                [VENV_PIP, "install", "-r", "requirements.txt", "-q"],
                capture_output=True, text=True, cwd=APP_DIR
            )
            if proc.returncode == 0:
                self.log("  Dependensi terinstall.")
            else:
                self.log(f"  Gagal: {proc.stderr[:200]}")
                self.start_btn.configure(state="normal", bg="#1E88E5")
                return
        else:
            self.log("  Dependensi tersedia.")
        self.set_progress(50)

        # Step 3: Check updates
        self.log("Memeriksa pembaruan...")
        try:
            # Coba dulu dengan SSL verification normal
            try:
                response = urllib.request.urlopen(VERSION_URL, timeout=10)
            except (urllib.error.URLError, ssl.SSLError):
                # Fallback: skip SSL verification (umum di WiFi kampus/proxy)
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                response = urllib.request.urlopen(VERSION_URL, timeout=10, context=ctx)
            
            latest = response.read().decode("utf-8").strip()
            if latest != LOCAL_VERSION:
                self.log(f"  Versi baru tersedia: {latest}")
                # Try git pull
                gp = subprocess.run(["git", "pull", "origin", "master"],
                                    capture_output=True, text=True, cwd=APP_DIR)
                if gp.returncode == 0:
                    self.log("  Update berhasil diunduh!")
                    subprocess.run([VENV_PIP, "install", "-r", "requirements.txt", "-q"],
                                   capture_output=True, cwd=APP_DIR)
                else:
                    self.log("  Auto-update gagal. Melanjutkan versi lama.")
            else:
                self.log("  Sudah versi terbaru.")
        except Exception as e:
            err_name = type(e).__name__
            logging.warning(f"Update check failed: {err_name}: {e}")
            self.log(f"  Gagal cek update ({err_name}). Melewati.")
        self.set_progress(75)

        # Step 4: Launch Streamlit
        self.log("\nMemulai server EEG Analysis Tool...")
        self.set_progress(90)

        kwargs = dict(
            cwd=APP_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        # Hide console window on Windows
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self.server_process = subprocess.Popen(
            [VENV_PYTHON, "-m", "streamlit", "run", "app.py",
             "--server.headless", "true"],
            **kwargs,
        )

        # Wait for server ready, then open browser
        server_ready = False
        for line in iter(self.server_process.stdout.readline, ""):
            logging.debug(f"streamlit: {line.strip()}")
            if "Local URL" in line or "Network URL" in line:
                url = line.strip().split()[-1]
                self.log(f"  Server berjalan: {url}")
                webbrowser.open("http://localhost:8501")
                server_ready = True
                break
            if "error" in line.lower():
                self.log(f"  {line.strip()}")
                break

        # Keep draining stdout in background so Streamlit doesn't block
        def _drain():
            try:
                for line in iter(self.server_process.stdout.readline, ""):
                    logging.debug(f"streamlit: {line.strip()}")
            except Exception:
                pass

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()

        self.set_progress(100)
        self.log("\nAplikasi berjalan! Buka browser jika belum terbuka.")
        self.log("Tekan 'Keluar' untuk menghentikan server.")

        self.start_btn.configure(text="Buka Browser", state="normal",
                                 bg="#10B981",
                                 command=lambda: webbrowser.open("http://localhost:8501"))

    def _on_start(self):
        thread = threading.Thread(target=self._run_setup, daemon=True)
        thread.start()

    def _on_quit(self):
        if self.server_process:
            self.server_process.terminate()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self.root.mainloop()


if __name__ == "__main__":
    LauncherApp().run()