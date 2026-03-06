"""
Modul single_file — Halaman analisis file EDF tunggal.
"""

import streamlit as st
import pandas as pd

from config import DEFAULT_SUBBANDS, DEFAULT_FEATURES, TASK_COLORS, ACCENT_LIGHT

from processing.loader import EEGLoader
from processing.filters import EEGFilters
from processing.features import EEGFeatures

from visualization.signal_plots import SignalPlots
from visualization.feature_plots import FeaturePlots


def render_single_file(cfg):
    """Render halaman analisis file tunggal.

    Dipanggil dari render_main() ketika tidak dalam batch mode.
    """
    if st.session_state.processor is None:
        st.info("Silakan unggah file EDF atau ZIP melalui panel samping.")
        return

    loader: EEGLoader = st.session_state.processor
    info = st.session_state.raw_info

    # --- Overview cards ---
    render_overview(info, loader)

    # --- Process ---
    if cfg["process"]:
        run_processing(loader, cfg)

    # --- Results ---
    if st.session_state.processed:
        render_results(loader, cfg)


def render_overview(info, loader: EEGLoader):
    """Kartu ringkasan data EDF di bagian atas dashboard."""
    st.markdown(
        '<p class="section-title">Ringkasan Data</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Channel", info["n_channels"])
    c2.metric("Sampling Rate", f'{info["sfreq"]:.0f} Hz')
    c3.metric("Durasi", f'{info["duration_s"]:.1f} s')
    c4.metric("Annotations", info["n_annotations"])

    # Task badges
    tasks = loader.get_task_list()
    if tasks:
        badge_html = ""
        for t in tasks:
            color = TASK_COLORS.get(t, ACCENT_LIGHT)
            badge_html += (
                f'<span class="task-badge" style="background:{color}">{t}</span>'
            )
        st.markdown(
            f'<div style="margin-top:6px">'
            f'<span class="info-label">Task yang terdeteksi</span>'
            f'<div style="margin-top:6px">{badge_html}</div></div>',
            unsafe_allow_html=True,
        )


def run_processing(loader: EEGLoader, cfg):
    """Jalankan pipeline filtering, ICA, dan ekstraksi data."""
    loader.raw = loader.raw_original.copy()
    loader.processing_log = ["File EDF berhasil dimuat"]

    with st.spinner("Memproses data..."):
        if cfg["channels"]:
            EEGFilters.pick_channels(loader, cfg["channels"])

        # Bad channel detection (BARU)
        if cfg.get("detect_bad"):
            bad_chs = EEGFilters.detect_bad_channels(loader.raw)
            if bad_chs:
                loader.processing_log.append(
                    f"Bad channel ditemukan: {bad_chs}"
                )
                good_chs = [
                    ch for ch in loader.raw.ch_names if ch not in bad_chs
                ]
                if good_chs:
                    loader.raw.pick_channels(good_chs)
                    loader.processing_log.append(
                        f"Channel dieksklusi: {bad_chs}"
                    )
                st.warning(f"Bad channel ditemukan dan dieksklusi: {bad_chs}")

        # Amplitude filter
        if cfg.get("use_amplitude"):
            EEGFilters.apply_amplitude_filter(loader)

        if cfg["use_notch"]:
            EEGFilters.apply_notch(loader, freq=cfg["notch_freq"])

        EEGFilters.apply_bandpass(
            loader, cfg["bp_low"], cfg["bp_high"], order=cfg["bp_order"]
        )

        if cfg["use_ica"]:
            EEGFilters.apply_ica(
                loader, n_components=cfg["ica_n"], method=cfg["ica_method"]
            )

        df = loader.extract_dataframe()
        st.session_state.df_data = df

        channels = cfg["channels"] or loader.channel_names
        subbands = cfg["subbands"] or DEFAULT_SUBBANDS
        features = cfg["features"] or DEFAULT_FEATURES
        include_freq = cfg.get("include_frequency", True)

        # Fitur keseluruhan
        features_df = EEGFeatures.compute_subband_features(
            df, channels, loader.sfreq, subbands, features,
            include_frequency=include_freq,
        )
        st.session_state.features_df = features_df

        # Fitur per task
        tasks = cfg["tasks"]
        if tasks:
            task_features_df = EEGFeatures.compute_task_features(
                loader, df, channels, tasks, subbands, features,
                include_frequency=include_freq,
            )
            task_summary_df = loader.get_task_summary(df)
            st.session_state.task_features_df = task_features_df
            st.session_state.task_summary_df = task_summary_df
        else:
            st.session_state.task_features_df = pd.DataFrame()
            st.session_state.task_summary_df = pd.DataFrame()

        # Band ratios (BARU)
        if include_freq and not features_df.empty:
            ratios_df = EEGFeatures.compute_band_ratios(features_df)
            st.session_state.ratios_df = ratios_df
        else:
            st.session_state.ratios_df = pd.DataFrame()

        st.session_state.processed = True

    st.success("Pemrosesan selesai.")


def render_results(loader: EEGLoader, cfg):
    """Tampilkan semua hasil."""
    df = st.session_state.df_data
    features_df = st.session_state.features_df
    task_features_df = st.session_state.task_features_df
    task_summary_df = st.session_state.task_summary_df
    ratios_df = st.session_state.get("ratios_df", pd.DataFrame())
    channels = cfg["channels"] or loader.channel_names
    info = st.session_state.raw_info
    tasks = cfg["tasks"]

    # --- Filter Subband & Channel ---
    all_subbands = []
    all_channels_feat = []
    if not features_df.empty:
        if "subband" in features_df.columns:
            all_subbands = sorted(features_df["subband"].unique().tolist())
        if "channel" in features_df.columns:
            all_channels_feat = sorted(features_df["channel"].unique().tolist())

    if all_subbands or all_channels_feat:
        st.markdown(
            '<p class="section-title">Filter Hasil</p>',
            unsafe_allow_html=True,
        )
        fc1, fc2 = st.columns(2)

        selected_subbands = all_subbands
        selected_channels_feat = all_channels_feat

        with fc1:
            if all_subbands:
                selected_subbands = st.multiselect(
                    "Subband", all_subbands, default=all_subbands,
                    key="sf_subband_filter",
                )
        with fc2:
            if all_channels_feat:
                selected_channels_feat = st.multiselect(
                    "Channel", all_channels_feat, default=all_channels_feat,
                    key="sf_channel_filter",
                )

        # Apply subband filter
        if selected_subbands and len(selected_subbands) < len(all_subbands):
            features_df = features_df[
                features_df["subband"].isin(selected_subbands)
            ].copy()
            if (
                task_features_df is not None
                and not task_features_df.empty
                and "subband" in task_features_df.columns
            ):
                task_features_df = task_features_df[
                    task_features_df["subband"].isin(selected_subbands)
                ].copy()

        # Apply channel filter
        if selected_channels_feat and len(selected_channels_feat) < len(all_channels_feat):
            features_df = features_df[
                features_df["channel"].isin(selected_channels_feat)
            ].copy()
            if (
                task_features_df is not None
                and not task_features_df.empty
                and "channel" in task_features_df.columns
            ):
                task_features_df = task_features_df[
                    task_features_df["channel"].isin(selected_channels_feat)
                ].copy()

        # Recompute ratios from filtered features
        if not ratios_df.empty and "channel" in ratios_df.columns:
            ratios_df = EEGFeatures.compute_band_ratios(features_df)


    # --- Processing log ---
    st.markdown(
        '<p class="section-title">Log Pemrosesan</p>',
        unsafe_allow_html=True,
    )
    log_html = "<br>".join(f"&bull; {entry}" for entry in loader.get_processing_log())
    st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    # --- Signal plot ---
    st.markdown(
        '<p class="section-title">Sinyal EEG (Terfilter)</p>',
        unsafe_allow_html=True,
    )
    max_t = float(df["time"].max())
    window = min(10.0, max_t)
    t_range = st.slider(
        "Rentang Waktu (s)", 0.0, max_t, (0.0, window), step=0.5,
    )
    fig_signal = SignalPlots.plot_raw_signal(
        df, channels, time_range=t_range, title="Sinyal Terfilter",
    )
    st.plotly_chart(fig_signal, use_container_width=True)

    # --- Task section ---
    if tasks and task_summary_df is not None and not task_summary_df.empty:
        _render_task_section(
            loader, df, channels, tasks, task_features_df, task_summary_df, cfg,
        )

    # --- Data table ---
    st.markdown(
        '<p class="section-title">Tabel Data</p>', unsafe_allow_html=True,
    )
    st.dataframe(df, use_container_width=True, height=320)

    # --- EDA tabs ---
    st.markdown(
        '<p class="section-title">Exploratory Data Analysis</p>',
        unsafe_allow_html=True,
    )
    tab_dist, tab_psd, tab_corr, tab_annot = st.tabs(
        ["Distribusi", "PSD", "Korelasi", "Marker"]
    )

    with tab_dist:
        fig_dist = SignalPlots.plot_signal_distribution(df, channels)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)

    with tab_psd:
        fig_psd = SignalPlots.plot_psd(loader.raw)
        st.plotly_chart(fig_psd, use_container_width=True)

    with tab_corr:
        fig_corr = SignalPlots.plot_channel_correlation(df, channels)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Pilih minimal 2 channel untuk melihat korelasi.")

    with tab_annot:
        annotations = info.get("annotations", [])
        fig_annot = SignalPlots.plot_annotation_summary(annotations)
        if fig_annot:
            st.plotly_chart(fig_annot, use_container_width=True)
        else:
            st.info("File ini tidak memiliki annotation/marker.")

    # --- Feature summary ---
    st.markdown(
        '<p class="section-title">Ringkasan Fitur</p>',
        unsafe_allow_html=True,
    )
    if not features_df.empty:
        st.dataframe(features_df, use_container_width=True, height=280)
        feat_cols = [
            c for c in features_df.columns if c not in ("channel", "subband")
        ]
        if feat_cols:
            selected_feat = st.selectbox(
                "Visualisasi fitur", feat_cols, key="feat_select",
            )
            fig_feat = FeaturePlots.plot_feature_comparison(
                features_df, selected_feat,
            )
            if fig_feat:
                st.plotly_chart(fig_feat, use_container_width=True)

    # --- Band Ratios (BARU) ---
    if not ratios_df.empty:
        st.markdown(
            '<p class="section-title">Rasio Subband</p>',
            unsafe_allow_html=True,
        )
        fig_ratios = FeaturePlots.plot_band_ratios(ratios_df)
        if fig_ratios:
            st.plotly_chart(fig_ratios, use_container_width=True)
        st.dataframe(ratios_df, use_container_width=True, hide_index=True)


def _render_task_section(loader, df, channels, tasks, task_features_df,
                          task_summary_df, cfg):
    """Section khusus analisis berbasis task."""
    st.markdown(
        '<p class="section-title">Analisis per Task</p>',
        unsafe_allow_html=True,
    )

    # Task summary row
    col_pie, col_table = st.columns([1, 2])

    with col_pie:
        fig_pie = FeaturePlots.plot_task_pie(task_summary_df)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_table:
        st.markdown("**Ringkasan Task**")
        display_df = task_summary_df[task_summary_df["task"] != "none"].copy()
        if not display_df.empty:
            display_df.columns = ["Task", "Jumlah Sample", "Durasi (s)", "% Total"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada task yang ditemukan.")

    # --- Occurrence mode toggle ---
    st.divider()
    occ_mode = st.radio(
        "Mode Analisis Task",
        ["Gabungan", "Per Occurrence", "Agregat Occurrence"],
        horizontal=True,
        key="occ_mode",
        help=(
            "**Gabungan**: Semua segmen digabung jadi satu (cara lama). "
            "**Per Occurrence**: Analisis tiap kemunculan terpisah "
            "(Resting₁, Resting₂, dst). "
            "**Agregat Occurrence**: Hitung fitur per occurrence dulu, "
            "lalu rata-rata hasilnya (lebih robust)."
        ),
    )

    # --- Occurrence info table ---
    occurrences = loader.get_task_occurrences()
    if occurrences and occ_mode != "Gabungan":
        occ_table = pd.DataFrame(occurrences)
        occ_table = occ_table[occ_table["task"].isin(tasks)]
        if not occ_table.empty:
            occ_table["label"] = (
                occ_table["task"] + "_" + occ_table["occurrence"].astype(str)
            )
            occ_display = occ_table[["label", "task", "occurrence", "onset", "duration"]].copy()
            occ_display.columns = ["Label", "Task", "#", "Onset (s)", "Durasi (s)"]
            occ_display["Onset (s)"] = occ_display["Onset (s)"].round(2)
            occ_display["Durasi (s)"] = occ_display["Durasi (s)"].round(2)
            with st.expander("Timeline Occurrence", expanded=False):
                st.dataframe(occ_display, use_container_width=True, hide_index=True)

    # --- Per-task signal viewer ---
    if occ_mode == "Per Occurrence":
        # Show signal per occurrence
        occ_labels = []
        for occ in occurrences:
            if occ["task"] in tasks:
                occ_labels.append(f"{occ['task']}_{occ['occurrence']}")

        if occ_labels:
            sel_occ = st.selectbox(
                "Lihat sinyal occurrence", ["-- Semua --"] + occ_labels,
                key="occ_signal_view",
            )
            if sel_occ != "-- Semua --":
                parts = sel_occ.rsplit("_", 1)
                t_name, t_num = parts[0], int(parts[1])
                occ_df = loader.extract_occurrence_segment(df, t_name, t_num)
                if not occ_df.empty:
                    fig_occ = SignalPlots.plot_raw_signal(
                        occ_df, channels, title=f"Sinyal - {sel_occ}",
                    )
                    st.plotly_chart(fig_occ, use_container_width=True)
                else:
                    st.info(f"Tidak ada data untuk '{sel_occ}'.")
    else:
        # Original task signal viewer
        task_view = st.selectbox(
            "Lihat sinyal task", ["-- Semua --"] + list(tasks), key="task_view",
        )
        if task_view != "-- Semua --":
            task_df = loader.extract_task_segments(df, task_view)
            if not task_df.empty:
                fig_task = SignalPlots.plot_raw_signal(
                    task_df, channels, title=f"Sinyal - {task_view}",
                )
                st.plotly_chart(fig_task, use_container_width=True)
            else:
                st.info(f"Tidak ada data untuk task '{task_view}'.")

    # --- Feature computation based on mode ---
    subbands = cfg["subbands"] or DEFAULT_SUBBANDS
    features_list = cfg["features"] or DEFAULT_FEATURES
    include_freq = cfg.get("include_frequency", True)

    if occ_mode == "Per Occurrence":
        active_feat_df = EEGFeatures.compute_occurrence_features(
            loader, df, channels, tasks, subbands, features_list,
            include_frequency=include_freq,
        )
        task_col = "task_occ"  # Gunakan label Resting_1, Typing_1, dst
        meta_exclude = {"task", "occurrence", "task_occ", "channel", "subband"}
    elif occ_mode == "Agregat Occurrence":
        active_feat_df = EEGFeatures.compute_aggregated_occurrence_features(
            loader, df, channels, tasks, subbands, features_list,
            include_frequency=include_freq,
        )
        task_col = "task"
        meta_exclude = {"task", "channel", "subband"}
    else:
        active_feat_df = task_features_df
        task_col = "task"
        meta_exclude = {"task", "channel", "subband"}

    # --- Feature comparison ---
    if active_feat_df is not None and not active_feat_df.empty:
        # Subband + channel filters
        filter_cols = st.columns(2)
        with filter_cols[0]:
            all_sb = sorted(active_feat_df["subband"].unique().tolist()) if "subband" in active_feat_df.columns else []
            if all_sb:
                sel_sb_task = st.multiselect(
                    "Filter Subband (Task)",
                    all_sb, default=all_sb,
                    key="task_subband_filter",
                )
                if sel_sb_task and len(sel_sb_task) < len(all_sb):
                    active_feat_df = active_feat_df[
                        active_feat_df["subband"].isin(sel_sb_task)
                    ].copy()
        with filter_cols[1]:
            all_ch = sorted(active_feat_df["channel"].unique().tolist()) if "channel" in active_feat_df.columns else []
            if all_ch:
                sel_ch_task = st.multiselect(
                    "Filter Channel (Task)",
                    all_ch, default=all_ch,
                    key="task_channel_filter",
                )
                if sel_ch_task and len(sel_ch_task) < len(all_ch):
                    active_feat_df = active_feat_df[
                        active_feat_df["channel"].isin(sel_ch_task)
                    ].copy()

        st.markdown(f"**Perbandingan Fitur antar {occ_mode}**")
        feat_cols = [
            c for c in active_feat_df.columns if c not in meta_exclude
        ]
        if feat_cols:
            sel_feat = st.selectbox(
                "Fitur", feat_cols, key="task_feat_select",
            )
            # Use task_occ for Per Occurrence mode grouping
            fig_tf = FeaturePlots.plot_task_feature_comparison(
                active_feat_df, sel_feat, task_col=task_col,
            )
            if fig_tf:
                st.plotly_chart(fig_tf, use_container_width=True)

        st.dataframe(
            active_feat_df, use_container_width=True, height=280,
            hide_index=True,
        )

