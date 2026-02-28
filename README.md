# EEG Analysis Tool

Aplikasi web berbasis Streamlit untuk analisis sinyal EEG (Electroencephalography). Dirancang untuk membandingkan data EEG antara subjek ALS dan Normal melalui ekstraksi fitur dan analisis delta.

## Fitur Utama

- **Single File Analysis** - Upload dan analisis satu file EDF secara interaktif
- **Batch Processing** - Upload ZIP berisi banyak file EDF, proses secara paralel (multithreaded)
- **Analisis Delta** - Hitung selisih fitur antar task (misal: Thinking vs Resting)
- **Scatter Plot per Subjek** - Visualisasi data individual setiap subjek
- **Heatmap Delta** - Heatmap perubahan fitur per channel & subband
- **Transition Delta** - Delta per-subject lalu rata-rata per group

## Struktur Proyek

```
web/
├── app.py                    # Entry point Streamlit
├── config.py                 # Konfigurasi (subband, fitur, warna)
├── launcher.py               # GUI Launcher (Tkinter)
├── start.bat                 # Launcher Windows
├── start.sh                  # Launcher macOS/Linux
├── CHANGELOG.md              # Riwayat perubahan per versi
├── DISABLED_FEATURES.md      # Rekap fitur yang di-disable
├── processing/
│   ├── loader.py             # Load EDF/ZIP, metadata detection
│   ├── filters.py            # Bandpass, notch, ICA, bad channel
│   ├── features.py           # Ekstraksi fitur time-domain
│   ├── delta.py              # Perhitungan delta antar task
│   └── statistics.py         # Uji statistik
├── visualization/
│   ├── signal_plots.py       # Sinyal mentah, PSD, distribusi
│   ├── feature_plots.py      # Bar, box, grouped bar
│   └── comparison_plots.py   # Delta chart, heatmap, transition
└── ui/
    ├── styles.py             # CSS injection
    ├── sidebar.py            # Panel sidebar
    ├── single_file.py        # Halaman analisis file tunggal
    └── batch.py              # Halaman analisis batch
```

## Pipeline Pemrosesan

```
File EDF --> Load (MNE) --> Bandpass Filter (0.5-49 Hz)
         --> Extract DataFrame (time x channels + marker)
         --> Filter per Task (berdasarkan annotation/marker)
         --> Subband Filtering (Delta, Theta, Alpha, Beta, Gamma)
         --> Ekstraksi Fitur (MAV, Variance, STD)
         --> Analisis Delta (Task A - Task B)
```

## Fitur Statistik

| Fitur        | Rumus         | Keterangan                                                             |
| ------------ | ------------- | ---------------------------------------------------------------------- |
| **MAV**      | `mean(\|x\|)` | Mean Absolute Value - rata-rata amplitudo tanpa mempedulikan polaritas |
| **Variance** | `var(x)`      | Variasi sinyal - mengukur seberapa tersebar data                       |
| **STD**      | `std(x)`      | Standar deviasi - akar dari variance                                   |

## Subband EEG

| Subband | Rentang (Hz) | Keterangan                 |
| ------- | ------------ | -------------------------- |
| Delta   | 0.5 - 4      | Tidur dalam                |
| Theta   | 4 - 8        | Mengantuk, meditasi        |
| Alpha   | 8 - 13       | Rileks, mata tertutup      |
| Beta    | 13 - 30      | Fokus, berpikir aktif      |
| Gamma   | 30 - 49      | Pemrosesan kognitif tinggi |

## Instalasi

### Prasyarat

- Python 3.10 atau lebih baru
- `tkinter` (biasanya sudah termasuk di Python, kecuali di beberapa distro Linux)
- Git (opsional, untuk auto-update)

### Quick Start (Windows)

```bash
# 1. Clone repository
git clone https://github.com/henray404/EEG_web.git
cd EEG_web

# 2. Double-click start.bat
# 3. Klik tombol "Mulai Aplikasi"
```

> **Note**: `start.bat` akan otomatis install Python jika belum ada (via `winget`).

### Quick Start (macOS)

```bash
# 1. Clone repository
git clone https://github.com/henray404/EEG_web.git
cd EEG_web

# 2. Jalankan launcher
bash start.sh

# 3. Klik tombol "Mulai Aplikasi"
```

### Quick Start (Linux)

```bash
# 1. Clone repository
git clone https://github.com/henray404/EEG_web.git
cd EEG_web

# 2. Install Python dan tkinter jika belum ada
# Ubuntu / Debian:
sudo apt update
sudo apt install python3 python3-venv python3-tk

# Fedora:
sudo dnf install python3 python3-tkinter

# Arch Linux:
sudo pacman -S python tk

# 3. Jalankan launcher
python3 launcher.py

# 4. Klik tombol "Mulai Aplikasi"
```

> **Note**: Di Linux, `tkinter` tidak selalu terinstall secara default.
> Jika muncul error `ModuleNotFoundError: No module named 'tkinter'`,
> install paket `python3-tk` (Debian/Ubuntu) atau `python3-tkinter` (Fedora).

### Manual Setup (Semua OS)

Jika tidak ingin menggunakan GUI Launcher, bisa setup manual:

```bash
git clone https://github.com/henray404/EEG_web.git
cd EEG_web

# Buat virtual environment
python3 -m venv .venv

# Aktivasi
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependensi
pip install -r requirements.txt

# Jalankan Streamlit
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

### GUI Launcher

Aplikasi dilengkapi **GUI Launcher** (`launcher.py`) yang menangani setup secara otomatis:

- Membuat virtual environment
- Menginstall semua dependensi
- Memeriksa dan mengunduh update dari GitHub (via `git pull`)
- Menjalankan server dan membuka browser
- Menampilkan changelog untuk versi terkini

## Penggunaan

### Mode Single File

1. Pilih **File EDF** di sidebar
2. Upload file `.edf`
3. Klik **Proses Data**
4. Lihat sinyal mentah, subband, dan hasil fitur

### Mode Batch (ZIP)

1. Pilih **File ZIP (dataset)** di sidebar
2. Upload file ZIP berisi file EDF dengan struktur folder:
   ```
   dataset.zip
   ├── ALS/
   │   ├── ALS01_time1_scenario1/EEG.edf
   │   └── ...
   └── Normal/
       ├── id1_time1_scenario1/EEG.edf
       └── ...
   ```
3. Aktifkan toggle **Batch Analisis (semua file)**
4. Klik **Proses Batch**
5. Gunakan panel filter (Kategori, Skenario, Time, Subband, Channel) untuk eksplorasi data
6. Lihat hasil di **Analisis Delta**

## Konfigurasi

Edit `config.py` untuk menyesuaikan:

- **Subband**: Tambah/hapus subband EEG
- **Fitur**: Ubah fitur statistik yang diekstrak
- **Tema**: Sesuaikan warna tampilan

Edit `.streamlit/config.toml` untuk:

- `maxUploadSize`: Batas ukuran upload (default: 8192 MB)
- `theme`: Warna tema Streamlit

## Dependensi Utama

| Library       | Fungsi                             |
| ------------- | ---------------------------------- |
| **Streamlit** | Framework web app                  |
| **MNE**       | Membaca dan memproses file EDF/EEG |
| **NumPy**     | Operasi numerik                    |
| **Pandas**    | Manipulasi data tabular            |
| **SciPy**     | Uji statistik                      |
| **Plotly**    | Visualisasi interaktif             |
