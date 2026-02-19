# EEG Analysis Tool

Aplikasi web berbasis Streamlit untuk analisis sinyal EEG (Electroencephalography). Dirancang untuk membandingkan data EEG antara subjek ALS dan Normal melalui ekstraksi fitur, analisis delta, dan uji statistik.

## Fitur Utama

- **Single File Analysis** - Upload dan analisis satu file EDF secara interaktif
- **Batch Processing** - Upload ZIP berisi banyak file EDF, proses secara paralel (multithreaded)
- **Distribusi Fitur** - Visualisasi distribusi fitur per task, channel, dan subband
- **Analisis Delta** - Hitung selisih fitur antar task (misal: Thinking vs Resting)
- **ALS vs Normal** - Perbandingan statistik antara kelompok ALS dan Normal menggunakan Mann-Whitney U test
- **Scatter Plot per Subjek** - Visualisasi data individual setiap subjek

## Struktur Proyek

```
web/
├── app.py              # Aplikasi utama Streamlit (UI + logic)
├── eeg_processor.py    # Backend pemrosesan EEG (load, filter, fitur, statistik)
├── eeg_visualizer.py   # Modul visualisasi (plotly charts)
├── config.py           # Konfigurasi default (subband, fitur, warna tema)
├── requirements.txt    # Dependensi Python
├── .streamlit/
│   └── config.toml     # Konfigurasi Streamlit (tema, upload size)
└── .gitignore
```

## Pipeline Pemrosesan

```
File EDF ──→ Load (MNE) ──→ Bandpass Filter (1-50 Hz)
         ──→ Extract DataFrame (time × channels + marker)
         ──→ Filter per Task (berdasarkan annotation/marker)
         ──→ Subband Filtering (Alpha, Beta, Gamma, dll.)
         ──→ Ekstraksi Fitur (MAV, Variance, STD)
         ──→ Analisis Delta (Task A − Task B)
         ──→ Perbandingan ALS vs Normal (Mann-Whitney U)
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
- pip (Python package manager)

### Langkah Setup

```bash
# 1. Clone repository
git clone https://github.com/<username>/<repo>.git
cd <repo>

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktivasi virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 4. Install dependensi
pip install -r requirements.txt

# 5. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

## Penggunaan

### Mode Single File

1. Pilih mode **Single File** di sidebar
2. Upload file `.edf`
3. Pilih channel dan subband yang diinginkan
4. Lihat sinyal mentah, subband, dan hasil fitur ekstraksi

### Mode Batch (ZIP)

1. Pilih mode **Batch (ZIP)** di sidebar
2. Upload file ZIP berisi file EDF dengan struktur folder:
   ```
   dataset.zip
   ├── ALS/
   │   ├── ALS01_time1_scenario1/EEG.edf
   │   ├── ALS01_time1_scenario2/EEG.edf
   │   └── ...
   └── Normal/
       ├── id1_time1_scenario1/EEG.edf
       ├── id1_time1_scenario2/EEG.edf
       └── ...
   ```
3. Sistem akan mendeteksi kategori (ALS/Normal), subjek, time, dan scenario dari nama folder
4. Gunakan tab **Distribusi Fitur**, **Analisis Delta**, atau **Data Mentah** untuk menganalisis

### Perbandingan ALS vs Normal

1. Di tab **Analisis Delta**, centang checkbox **Bandingkan ALS vs Normal**
2. Pilih mode perbandingan (Delta, Z-score, atau keduanya)
3. Atur jumlah sampel ALS dan Normal
4. Hasil: bar chart, p-value table, dan scatter plot per subjek

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
| **SciPy**     | Uji statistik (Mann-Whitney U)     |
| **Plotly**    | Visualisasi interaktif             |
