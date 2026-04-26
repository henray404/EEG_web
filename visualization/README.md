# Visualization Module

This directory contains detailed information about the functions and classes in the `visualization` module.

## `comparison_plots.py`

_Modul comparison_plots — Delta bar/scatter/heatmap, ALS vs Normal._

### `class ComparisonPlots`
Visualisasi perbandingan: delta, ALS vs Normal.

**Methods:**
- **`plot_delta_bar(agg_df, feature_name, task_a, task_b, title)`**: Bar chart delta agregat per channel/subband.
- **`plot_delta_heatmap(agg_df, feature_name, task_a, task_b, title)`**: Heatmap delta mean per channel (rows) x subband (cols).
- **`plot_per_file_delta(delta_df, feature_name, task_a, task_b, title)`**: Bar chart delta per file per channel/subband (top N).
- **`plot_als_vs_normal(stats_df, feature_name, active_task, baseline_task, title, use_sem)`**: Grouped bar ALS vs Normal per channel/subband + p-value annotations.
- **`plot_transition_deltas(transition_df, feature_name, title)`**: Bar chart transition deltas per group (ALS vs Normal) with SEM.

## `feature_plots.py`

_Modul feature_plots — Visualisasi distribusi dan perbandingan fitur._

### `class FeaturePlots`
Visualisasi fitur EEG: bar, box, pie, grouped bar.

**Methods:**
- **`plot_feature_comparison(features_df, feature_name, title)`**: Grouped bar chart fitur per channel dan subband.
- **`plot_task_feature_comparison(task_features_df, feature_name, title, task_col)`**: Grouped bar chart fitur per task, channel, dan subband.
- **`plot_task_pie(task_summary_df, title)`**: Pie chart proporsi waktu per task.
- **`plot_batch_overview(batch_df, feature_name, title)`**: Box plot fitur per task di seluruh dataset.
- **`plot_grouped_bar(df, feature_name, group_col, facet_col, x_col, title)`**: Grouped bar chart faceted by channel.
- **`plot_band_ratios(ratios_df, title)`**: Bar chart rasio antar subband per channel.

## `signal_plots.py`

_Modul signal_plots — Visualisasi sinyal mentah, PSD, distribusi._

### `class SignalPlots`
Visualisasi sinyal EEG.

**Methods:**
- **`plot_raw_signal(df, channels, time_range, title)`**: Line chart sinyal EEG per channel.
- **`plot_task_signal(df, channels, task_name, time_range)`**: Plot sinyal untuk satu task tertentu.
- **`plot_psd(raw, method, fmin, fmax, n_fft, title)`**: PSD plot dari raw MNE object dengan pilihan metode.
- **`plot_psd_per_task(psd_per_task, ch_names, title)`**: Overlay PSD dari beberapa task dalam satu plot.
- **`plot_psd_band_power_bars(band_power_df, title)`**: Bar chart band power per subband per channel dari hasil PSD.
- **`plot_signal_distribution(df, channels, title)`**: Histogram distribusi amplitudo per channel.
- **`plot_channel_correlation(df, channels, title)`**: Heatmap korelasi antar channel.
- **`plot_annotation_summary(annotations, title)`**: Bar chart jumlah kemunculan setiap annotation.
- **`plot_spectrogram(windowed_features_df, channel, feature_name, title)`**: Time-frequency heatmap dari sliding window features.
- **`plot_epoch_summary(epoch_stats_df, feature_name, title)`**: Bar chart fitur per task dari epoched analysis (dengan error bar).
- **`plot_connectivity_matrix(conn_matrix, channels, method, subband, task, title)`**: Heatmap matrix konektivitas antar channel.
- **`plot_connectivity_comparison(task_conn_dict, channels, subband, method)`**: Subplot heatmap per task untuk membandingkan konektivitas.

