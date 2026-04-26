# Processing Module

This directory contains detailed information about the functions and classes in the `processing` module.

## `chunking.py`

_Modul chunking — Pipeline Chunking & Chain Encoding untuk sinyal EEG._

### `class ChunkingPipeline`
Pipeline Chunking & Chain Encoding untuk sinyal EEG.

**Methods:**
- **`compute_chunked_subband_features(df, channels, sfreq, chunk_duration, subbands, features)`**: Hitung fitur per chunk per channel per subband.
- **`compute_task_chunked_features(loader, df, channels, tasks, chunk_duration, subbands, features)`**: Hitung fitur per chunk per task per channel per subband.
- **`compute_chain_encoding(chunked_features_df, features)`**: Encode tren antar chunk berturutan.
- **`summarize_chain_encoding(chain_df, features)`**: Ringkasan chain encoding: sequence string + ratio kenaikan.
- **`process_single_file(loader, df, channels, chunk_duration, subbands, features, chain_features, use_task_segmentation, tasks, filename, subject_id, scenario, scenario_id)`**: Proses satu file EDF: chunking + FE + chain encoding.
- **`generate_cross_file_summary(all_chain_df)`**: Buat summary perbandingan chain sequence antar file berbeda.

- **`process_dataset(dataset_root, subject_range, scenarios, chunk_duration, subbands, features, chain_features, use_task_segmentation, progress_callback)`**
  - Batch chunking+chain encoding untuk EEGET-ALS Dataset.

- **`process_cached_item(df, channels, sfreq, tasks, chunk_duration, subbands, features, chain_features, use_task_segmentation, scale_x10k, filename, subject_id, scenario, scenario_id)`**
  - Varian process_single_file yang tidak butuh loader object.

- **`process_cached_items(cached_items, chunk_duration, subbands, features, chain_features, use_task_segmentation, scale_x10k, progress_callback)`**
  - Batch chunking+chain encoding dari cached items.

- **`process_and_export(dataset_root, output_dir, features_name, chain_name, summary_name, **kwargs)`**
  - Wrapper: jalankan ``process_dataset`` lalu simpan 3 CSV.

## `connectivity.py`

_Modul connectivity — Analisis konektivitas fungsional EEG (PLI / wPLI)._

### `class ConnectivityAnalyzer`
Analisis konektivitas fungsional EEG (PLI/wPLI).

**Methods:**
- **`compute_connectivity(data, sfreq, channels, method, subbands)`**: Hitung connectivity matrix per subband.
- **`compute_task_connectivity(loader, df, channels, tasks, sfreq, method, subbands, use_epoching, epoch_duration)`**: Hitung connectivity per task.
- **`connectivity_to_dataframe(conn_dict, channels, method, task_name)`**: Konversi connectivity matrices ke DataFrame.
- **`all_tasks_to_dataframe(task_conn_dict, channels, method)`**: Konversi semua task connectivity ke satu DataFrame.

## `delta.py`

_Modul delta — Perhitungan delta antar task._

### `class DeltaCalculator`
Kumpulan metode untuk menghitung delta antar task.

**Methods:**
- **`calculate_task_delta(batch_df, task_a, task_b, feature_cols)`**: Hitung delta (task_a − task_b) per filename/channel/subband.
- **`compute_subject_delta(batch_df, subject_id, from_task, to_task, feature_col, subband, channels)`**: Hitung delta untuk satu subjek (per task transition).
- **`compute_group_transition_deltas(batch_df, from_task, to_task, feature_col, subband, channels, scenarios, sessions)`**: Hitung transition delta per group (ALS / Normal).
- **`compute_transition_table(batch_df, from_task, to_task, feature_col, subbands, channels, scenarios, sessions)`**: Hitung transition delta untuk semua subband.

## `encoding.py`

_Modul encoding — Batch encoding sinyal EEG ke matriks fitur._

- **`encode_single_edf(edf_path, window_size, overlap, subbands, features, include_frequency, include_gamma_bursts, psd_method, psd_fmin, psd_fmax)`**
  - Encode satu file EDF menjadi DataFrame fitur per window.

- **`flatten_features(encoded_df)`**
  - Pivot dari format panjang ke format lebar.

- **`encode_dataset(dataset_root, subject_range, scenarios, window_size, overlap, subbands, features, include_frequency, include_gamma_bursts, psd_method, psd_fmin, psd_fmax, progress_callback)`**
  - Encode seluruh EEGET-ALS dataset menjadi satu DataFrame besar.

- **`encode_cached_items(cached_items, window_size, overlap, subbands, features, include_frequency, include_gamma_bursts, psd_method, psd_fmin, psd_fmax, progress_callback)`**
  - Encode dari cached items (hasil batch ZIP yang sudah di-preprocess).

- **`encode_and_export(dataset_root, output_path, flatten, **kwargs)`**
  - Encode dataset dan simpan ke CSV.

## `epoching.py`

_Modul epoching — Epoching dan Sliding Windows untuk data EEG._

### `class EpochEngine`
Epoching dan Sliding Windows untuk data EEG.

**Methods:**
- **`create_fixed_epochs(df, channels, sfreq, epoch_duration)`**: Potong DataFrame menjadi epoch-epoch dengan durasi tetap.
- **`create_sliding_windows(df, channels, sfreq, window_size, overlap)`**: Potong DataFrame menggunakan sliding window.
- **`create_task_epochs(loader, df, channels, sfreq, tasks, epoch_duration)`**: Epoching per task: potong setiap segmen task menjadi epoch.
- **`create_task_sliding_windows(loader, df, channels, sfreq, tasks, window_size, overlap)`**: Sliding window per task.
- **`reject_bad_epochs(epochs, channels, threshold_uv, sfreq)`**: Buang epoch yang amplitude-nya melebihi threshold.
- **`compute_epoch_features(epochs, channels, sfreq, subbands, features, include_frequency, psd_method, psd_fmin, psd_fmax, psd_n_fft)`**: Hitung fitur per epoch, lalu rata-ratakan.
- **`compute_windowed_features(windows, channels, sfreq, subbands, features, include_frequency, psd_method, psd_fmin, psd_fmax, psd_n_fft)`**: Hitung fitur per sliding window (time-resolved).
- **`compute_epoched_task_features(loader, df, channels, tasks, sfreq, subbands, features, epoch_duration, reject_threshold, include_frequency, psd_method, psd_fmin, psd_fmax, psd_n_fft)`**: Hitung fitur per task menggunakan epoching.
- **`compute_windowed_task_features(loader, df, channels, tasks, sfreq, subbands, features, window_size, overlap, include_frequency, psd_method, psd_fmin, psd_fmax, psd_n_fft)`**: Hitung fitur per task menggunakan sliding window (time-resolved).
- **`aggregate_windowed_features(windowed_df)`**: Agregasi fitur windowed menjadi 1 baris per task/channel/subband.

## `features.py`

_Modul features — Ekstraksi fitur EEG per channel/subband._

### `class EEGFeatures`
Kumpulan metode untuk ekstraksi fitur EEG.

**Methods:**
- **`compute_band_power(signal, sfreq, low, high)`**: Hitung band power absolut menggunakan FFT.
- **`compute_relative_power(signal, sfreq, low, high)`**: Hitung relative power (% total power).
- **`compute_peak_frequency(signal, sfreq, low, high)`**: Hitung peak frequency dalam subband.
- **`compute_subband_features(df, channels, sfreq, subbands, features, include_frequency, psd_method, psd_fmin, psd_fmax, psd_n_fft)`**: Hitung fitur per channel per subband.
- **`compute_task_features(loader, df, channels, tasks, subbands, features, include_frequency, psd_method, psd_fmin, psd_fmax, psd_n_fft)`**: Hitung fitur per task per channel per subband.
- **`compute_occurrence_features(loader, df, channels, tasks, subbands, features, include_frequency, psd_method, psd_fmin, psd_fmax, psd_n_fft)`**: Hitung fitur per occurrence per task per channel per subband.
- **`compute_aggregated_occurrence_features(loader, df, channels, tasks, subbands, features, include_frequency, psd_method, psd_fmin, psd_fmax, psd_n_fft)`**: Hitung fitur dengan rata-rata semua occurrence per task.
- **`compute_first_occurrence_features(loader, df, channels, tasks, subbands, features, include_frequency, psd_method, psd_fmin, psd_fmax, psd_n_fft)`**: Hitung fitur hanya dari occurrence pertama tiap task.
- **`compute_erd_ers(batch_df, baseline_task, feature_col)`**: Hitung ERD/ERS relatif terhadap baseline task.
- **`compute_erd_ers_paired(loader, df, channels, task_name, subbands, baseline_task)`**: Hitung ERD/ERS menggunakan pasangan Resting→Task yang berurutan.
- **`compute_band_ratios(features_df, ratios)`**: Hitung rasio power antar subband.

## `filters.py`

_Modul filters — Bandpass, notch, ICA, dan bad channel detection._

### `class EEGFilters`
Kumpulan metode filtering untuk data EEG.

**Methods:**
- **`pick_channels(loader, channels)`**: Pilih subset channel dari raw data.
- **`detect_bad_channels(raw, threshold)`**: Deteksi channel buruk berdasarkan variance outlier (MAD threshold).
- **`apply_amplitude_filter(loader, max_uv)`**: Clipping sinyal EEG ke ±max_uv µV.
- **`apply_notch(loader, freq, quality)`**: Terapkan notch filter untuk menghapus noise powerline.
- **`apply_bandpass(loader, low_freq, high_freq, order)`**: Terapkan bandpass filter pada raw data.
- **`bandpass_array(data, sfreq, low, high, order)`**: Bandpass filter pada array numpy 1-D.
- **`apply_ica(loader, n_components, method, random_state, auto_detect_artifacts)`**: Jalankan ICA dan hapus komponen artefak otomatis.
- **`apply_car(loader)`**: Common Average Reference: re-referencing ke rata-rata semua channel.

## `gamma_bursts.py`

_Gamma burst detection from Superlet time-frequency power maps._

### `class Burst`
A single burst interval and its summary statistics.

**Methods:**
- **`duration_s(self, sfreq)`**: No description provided.

### `class GammaBurstDetector`
Detect gamma bursts from TFR power and aggregate per window.

**Methods:**
- **`__init__(self, sfreq, freqs, mad_k, min_duration_ms, merge_gap_ms)`**: No description provided.
- **`detect(self, power)`**: Detect bursts from power map with shape (n_freqs, n_samples).
- **`aggregate_in_window(self, bursts, window_start_sample, window_end_sample)`**: Aggregate burst features for a sample interval.

## `loader.py`

_Modul loader — Load data EEG dari file EDF atau arsip ZIP. Fungsi deteksi metadata (kategori, subject, time, scenario) juga ada di sini._

### `class EEGLoader`
Loader untuk file EDF dan ZIP berisi EDF.

**Methods:**
- **`__init__(self)`**: No description provided.
- **`load_edf(self, file_source)`**: Load file EDF dari path string atau file-like object (upload).
- **`list_edf_in_zip(zip_buffer)`**: Temukan semua file EDF di dalam arsip ZIP.
- **`load_edf_from_zip(self, zip_buffer, edf_path_in_zip)`**: Ekstrak satu file EDF dari ZIP lalu load.
- **`get_raw_info(self)`**: Return dict ringkasan raw data.
- **`get_processing_log(self)`**: No description provided.
- **`get_task_list(self)`**: Return daftar nama task unik dari annotations.
- **`extract_dataframe(self)`**: Konversi raw data ke DataFrame dengan kolom waktu, channel, marker.
- **`extract_task_segments(self, df, task_name)`**: Filter DataFrame ke segment yang berisi task tertentu.
- **`get_task_summary(self, df)`**: Hitung statistik ringkasan per task.
- **`get_task_occurrences(self)`**: Return daftar occurrence per task, dalam urutan waktu.
- **`extract_occurrence_segment(self, df, task_name, occurrence_num)`**: Ekstrak data untuk occurrence spesifik dari satu task.
- **`get_occurrence_pairs(self, task_a, task_b)`**: Temukan pasangan sequential occurrence dari dua task.
- **`detect_category(edf_path_in_zip)`**: Deteksi kategori (ALS/Normal) dan metadata dari path di ZIP.
- **`load_openbci_txt(self, file_source, channel_map)`**: Load file TXT OpenBCI Cyton.
- **`load_txt_from_zip(self, zip_buffer, txt_path_in_zip, channel_map)`**: Ekstrak satu file TXT dari ZIP lalu load sebagai OpenBCI.
- **`list_txt_in_zip(zip_buffer)`**: Temukan semua file TXT di dalam arsip ZIP.
- **`detect_openbci_metadata(txt_path)`**: Deteksi subject ID dan kondisi dari path/nama file TXT OpenBCI.
- **`scan_openbci_zip(zip_buffer)`**: Scan ZIP untuk struktur folder OpenBCI (Baseline/Familiar/Unfamiliar).

## `psd.py`

_Modul psd — Analisis Power Spectral Density (PSD)._

### `class PSDAnalyzer`
Kelas untuk menghitung PSD menggunakan Welch atau Multitaper.

**Methods:**
- **`compute_psd_array(data, sfreq, method, fmin, fmax, n_fft)`**: Hitung PSD dari array numpy 2-D (n_channels × n_samples).
- **`compute_psd_raw(raw, method, fmin, fmax, n_fft)`**: Hitung PSD dari MNE raw object.
- **`compute_psd_per_task(loader, df, method, fmin, fmax, n_fft)`**: Hitung PSD per task dari DataFrame.
- **`compute_band_power_from_psd(psds, freqs, ch_names, subbands)`**: Ekstrak band power numerik dari PSD yang sudah dihitung.
- **`compute_task_band_power(loader, df, channels, tasks, subbands, method, fmin, fmax, n_fft)`**: Hitung band power per task per channel per subband dari PSD.

## `statistics.py`

_Modul statistics — Uji statistik untuk perbandingan ALS vs Normal._

- **`cohens_d(group1, group2)`**
  - Hitung Cohen's d (effect size).

- **`interpret_cohens_d(d)`**
  - Interpretasi ukuran efek Cohen's d.

### `class StatisticalTests`
Kumpulan uji statistik untuk perbandingan group.

**Methods:**
- **`normalize_per_subject(batch_df, feature_cols, method, scope)`**: Normalisasi fitur per subjek.
- **`compare_als_vs_normal(batch_df, active_task, baseline_task, feature_cols, mode, apply_fdr, compute_effect_size, include_ttest)`**: Bandingkan fitur antara ALS dan Normal.

## `superlets.py`

_Superlet time-frequency transform utilities._

### `class SuperletTFR`
Fractional Adaptive Superlet Transform (FASLT) on 1D signals.

**Methods:**
- **`__init__(self, sfreq, freqs, c_base, order_min, order_max)`**: No description provided.
- **`compute(self, signal)`**: Compute Superlet power map.

- **`build_frequency_grid(low, high, n_freqs, spacing)`**
  - Build frequency grid for time-frequency transforms.

- **`compute_faslt(signal, sfreq, freqs, c_base, order_min, order_max)`**
  - Convenience wrapper for one-shot FASLT computation.

