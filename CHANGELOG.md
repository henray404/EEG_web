# Changelog

## v1.8 (2026-03-05)

- Filter batch (Skenario, Time, Subband, Channel) default kosong -- tidak semua terpilih saat awal
- Tambah fitur "Fokus Satu Fitur" di Tabel Delta Lengkap: pilih satu fitur untuk tabel ringkas dengan download terpisah
- Tambah pewarnaan kolom pada tabel delta (lengkap & fokus): biru muda (meta), hijau muda (Task A), oranye muda (Task B), pink muda (Delta)

## v1.3.1 (2026-03-01)

- Fix batch task: task yang muncul sekarang semua (union), bukan hanya yang ada di semua file (intersection)
- Fix launcher: tombol tidak terlihat karena changelog terlalu panjang, sekarang changelog scrollable
- Tambah scikit-learn ke requirements.txt (dibutuhkan ICA fastica)

## v1.3 (2026-02-28)

- Hapus tab Distribusi Fitur dari batch analysis
- Hapus Z-score Normalisasi dari panel filter
- Hapus perbandingan ALS vs Normal dan tab Statistik
- Hapus selector Channel, Task, Subband, Fitur dari sidebar (proses semua default)
- Modularisasi batch.py: pecah fungsi delta tab jadi sub-fungsi terpisah
- Bersihkan kode: hapus dead code, blok `if False`, dan komentar tidak perlu
- batch.py: 953 baris -> 353 baris, sidebar.py: 241 baris -> 184 baris
- Tambah CHANGELOG.md dengan tampilan di GUI Launcher
- Tambah DISABLED_FEATURES.md (rekap fitur yang di-disable)
- Update README.md dengan instruksi Linux (Ubuntu, Fedora, Arch)

## v1.2 (2026-02-28)

- Tambah analisis per-occurrence task (Resting_1, Resting_2, dst)
- Tambah mode Agregat Occurrence (rata-rata semua occurrence)
- Tambah filter subband dan channel pada hasil batch
- Tambah panel konfigurasi batch (3 kolom: Kategori, Skenario, Time)
- Nonaktifkan sementara: fitur frekuensi, tab statistik, data mentah
- Hapus fitur RMS dari analisis
- Hapus semua emoji dari antarmuka
- Tambah panduan lengkap (GUIDE.md)

## v1.1 (2026-02-14)

- Refactor arsitektur modular (processing/, visualization/, ui/)
- Integrasi fitur pipeline: bad channel detection, band power, ratios
- Tambah statistical tests: T-test, Cohen's d, FDR correction
- Tambah per-subject delta calculation
- Tambah ALS vs Normal comparison (sementara nonaktif)
- Tambah normalisasi Z-score per subjek

## v1.0 (2026-02-13)

- Rilis awal EEG Analysis Tool
- Analisis file EDF tunggal dan batch (ZIP)
- Filtering: bandpass, notch, ICA
- Ekstraksi fitur time-domain: MAV, variance, std
- Visualisasi: sinyal mentah, PSD, distribusi, korelasi
- Delta analysis antar task
- GUI Launcher dengan auto-setup
