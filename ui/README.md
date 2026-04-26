# Ui Module

This directory contains detailed information about the functions and classes in the `ui` module.

## `batch.py`

_Modul batch -- Halaman analisis batch (ZIP)._

- **`run_batch_processing(cfg)`**
  - Jalankan batch analysis: baca semua EDF dari ZIP, hitung fitur.

- **`render_batch_results(cfg)`**
  - Tampilkan hasil batch analysis & delta comparison.

## `batch_openbci.py`

_Modul batch_openbci — Batch processing khusus data OpenBCI TXT._

- **`run_openbci_batch(cfg)`**
  - Jalankan batch processing untuk dataset OpenBCI TXT dari ZIP.

- **`render_openbci_batch_results(cfg)`**
  - Tampilkan hasil batch analysis OpenBCI.

## `sidebar.py`

_Modul sidebar -- Panel konfigurasi di sidebar Streamlit._

- **`init_state()`**
  - Inisialisasi session state dengan default values.

- **`render_sidebar()`**
  - Render panel sidebar dan return konfigurasi.

## `single_file.py`

_Modul single_file — Halaman analisis file EDF tunggal._

- **`render_single_file(cfg)`**
  - Render halaman analisis file tunggal.

- **`render_overview(info, loader)`**
  - Kartu ringkasan data EDF di bagian atas dashboard.

- **`run_processing(loader, cfg)`**
  - Jalankan pipeline filtering, ICA, dan ekstraksi data.

- **`render_results(loader, cfg)`**
  - Tampilkan semua hasil.

## `styles.py`

_Modul styles — CSS custom untuk dashboard._

- **`inject_css()`**
  - Inject custom CSS into Streamlit page.

