"""
Modul batch — Halaman analisis batch (ZIP).

Enhanced:
- T-test + Cohen's d (BARU dari pipeline)
- SEM error bars (BARU dari pipeline)
- Per-subject delta (BARU dari pipeline)
- Transition delta table (BARU dari pipeline)
"""

import io
import os
import random
import threading

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import DEFAULT_SUBBANDS, DEFAULT_FEATURES, TASK_COLORS, ACCENT_LIGHT

from processing.loader import EEGLoader
from processing.filters import EEGFilters
from processing.features import EEGFeatures
from processing.delta import DeltaCalculator
from processing.statistics import StatisticalTests

from visualization.feature_plots import FeaturePlots
from visualization.comparison_plots import ComparisonPlots


def _process_single_edf(zip_bytes, edf_path, channels, subbands, features,
                         include_frequency=True):
    """Worker: proses satu file EDF dari ZIP bytes. Thread-safe."""
    meta = EEGLoader.detect_category(edf_path)
    loader = EEGLoader()

    try:
        buf = io.BytesIO(zip_bytes)
        loader.load_edf_from_zip(buf, edf_path)
    except Exception:
        return None, []

    ch_list = channels if channels else loader.channel_names
    ch_list = [c for c in ch_list if c in loader.channel_names]
    if not ch_list:
        loader._cleanup_tmp()
        return None, []

    try:
        low_all = min(v[0] for v in subbands.values())
        high_all = max(v[1] for v in subbands.values())
        EEGFilters.apply_bandpass(loader, low_all, high_all)
    except Exception:
        pass

    df = loader.extract_dataframe()
    tasks_found = [t for t in loader.get_task_list() if t != "none"]

    if not tasks_found:
        loader._cleanup_tmp()
        return None, tasks_found

    feat_df = EEGFeatures.compute_task_features(
        loader, df, ch_list, tasks_found, subbands, features,
        include_frequency=include_frequency,
    )
    loader._cleanup_tmp()

    if feat_df.empty:
        return None, tasks_found

    feat_df.insert(0, "filename", edf_path)
    feat_df.insert(1, "category", meta["category"])
    feat_df.insert(2, "subject", meta["subject"])
    feat_df.insert(3, "time", meta["time"])
    feat_df.insert(4, "scenario", meta["scenario"])

    return feat_df, tasks_found


def run_batch_processing(cfg):
    """Jalankan batch analysis: baca semua EDF dari ZIP, hitung fitur."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    uploaded = cfg.get("uploaded")
    if uploaded is None:
        st.warning("File ZIP belum diunggah.")
        return

    subbands = cfg["subbands"] or DEFAULT_SUBBANDS
    features = cfg["features"] or DEFAULT_FEATURES
    channels = cfg["channels"] if cfg["channels"] else None
    include_freq = cfg.get("include_frequency", True)

    progress_bar = st.progress(0, text="Memulai batch processing...")

    # Baca ZIP ke bytes sekali untuk thread-safe sharing
    uploaded.seek(0)
    zip_bytes = uploaded.read()

    edf_list = EEGLoader.list_edf_in_zip(io.BytesIO(zip_bytes))
    total = len(edf_list)
    all_frames = []
    all_tasks_per_file = []

    counter_lock = threading.Lock()
    counter = [0]

    def _progress(current, total_n, fname):
        if total_n > 0:
            progress_bar.progress(
                current / total_n,
                text=f"Memproses {current}/{total_n}: {fname}",
            )

    def _on_done(edf_path):
        with counter_lock:
            counter[0] += 1
            _progress(counter[0], total, os.path.basename(edf_path))

    max_workers = min(8, os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {}
        for edf_path in edf_list:
            fut = executor.submit(
                _process_single_edf,
                zip_bytes, edf_path, channels, subbands, features,
                include_freq,
            )
            future_to_path[fut] = edf_path

        for future in as_completed(future_to_path):
            edf_path = future_to_path[future]
            _on_done(edf_path)
            try:
                feat_df, tasks_found = future.result()
                if tasks_found:
                    cat = EEGLoader.detect_category(edf_path).get(
                        "category", "Unknown"
                    )
                    all_tasks_per_file.append((cat, set(tasks_found)))
                if feat_df is not None:
                    all_frames.append(feat_df)
            except Exception:
                continue

    _progress(total, total, "Selesai")
    progress_bar.empty()

    if not all_frames:
        st.error("Tidak ada data fitur yang berhasil diekstrak dari ZIP.")
        return

    batch_df = pd.concat(all_frames, ignore_index=True)

    # Irisan task
    if all_tasks_per_file:
        from collections import defaultdict
        tasks_by_cat = defaultdict(set)
        for cat, tset in all_tasks_per_file:
            tasks_by_cat[cat].update(tset)
        if len(tasks_by_cat) > 1:
            common_tasks = set.intersection(*tasks_by_cat.values())
        else:
            common_tasks = list(tasks_by_cat.values())[0]
    else:
        common_tasks = set()

    st.session_state.batch_df = batch_df
    st.session_state.batch_tasks = sorted(common_tasks)
    st.session_state.batch_processed = True
    st.success(
        f"Batch processing selesai: {batch_df['filename'].nunique()} file, "
        f"{len(common_tasks)} task ditemukan."
    )


def render_batch_results(cfg):
    """Tampilkan hasil batch analysis & delta comparison."""
    batch_df = st.session_state.batch_df
    batch_tasks = st.session_state.batch_tasks

    if batch_df is None or batch_df.empty:
        return

    # --- Category filter ---
    categories = (
        sorted(batch_df["category"].unique().tolist())
        if "category" in batch_df.columns else []
    )

    if len(categories) > 1:
        st.markdown(
            '<p class="section-title">Filter Kategori</p>',
            unsafe_allow_html=True,
        )
        cat_counts = batch_df.groupby("category")["subject"].nunique()
        cat_labels = [f"Semua ({batch_df['subject'].nunique()})"]
        for cat in categories:
            n = cat_counts.get(cat, 0)
            cat_labels.append(f"{cat} ({n})")

        selected_cat_label = st.radio(
            "Kategori", cat_labels, horizontal=True,
            label_visibility="collapsed", key="cat_filter",
        )

        if selected_cat_label.startswith("Semua"):
            filtered_df = batch_df
            active_category = "Semua"
        else:
            active_category = selected_cat_label.split(" (")[0]
            filtered_df = batch_df[
                batch_df["category"] == active_category
            ].copy()
    else:
        filtered_df = batch_df
        active_category = categories[0] if categories else "Semua"

    # --- Scenario filter ---
    scenarios = (
        sorted(filtered_df["scenario"].unique().tolist())
        if "scenario" in filtered_df.columns else []
    )
    selected_scenarios = scenarios
    if len(scenarios) > 1:
        selected_scenarios = st.multiselect(
            "Filter Skenario", scenarios, default=scenarios,
            key="scenario_filter",
        )
        if selected_scenarios:
            filtered_df = filtered_df[
                filtered_df["scenario"].isin(selected_scenarios)
            ].copy()

    # --- Time filter ---
    times = (
        sorted(filtered_df["time"].unique().tolist())
        if "time" in filtered_df.columns else []
    )
    selected_times = times
    if len(times) > 1:
        selected_times = st.multiselect(
            "Filter Time", times, default=times, key="time_filter",
        )
        if selected_times:
            filtered_df = filtered_df[
                filtered_df["time"].isin(selected_times)
            ].copy()

    # --- Normalization toggle ---
    meta_cols = {
        "filename", "category", "subject", "time", "scenario",
        "task", "channel", "subband",
    }
    feat_cols = [c for c in filtered_df.columns if c not in meta_cols]

    norm_col1, norm_col2, norm_col3 = st.columns([1, 1, 2])
    with norm_col1:
        use_norm = st.toggle("Z-score Normalisasi", value=False, key="norm_toggle")
    with norm_col2:
        if use_norm:
            norm_scope = st.radio(
                "Scope", ["Per Subjek", "Per Subjek+Skenario"],
                horizontal=True, label_visibility="collapsed",
                key="norm_scope",
            )
            norm_scope_val = (
                "subject" if norm_scope == "Per Subjek" else "subject_scenario"
            )
        else:
            norm_scope_val = "subject"
    with norm_col3:
        if use_norm:
            if norm_scope_val == "subject":
                st.caption("Normalisasi per orang (semua skenario digabung).")
            else:
                st.caption("Normalisasi per orang per skenario (lebih granular).")

    if use_norm and feat_cols:
        filtered_df = StatisticalTests.normalize_per_subject(
            filtered_df, feat_cols, method="zscore", scope=norm_scope_val,
        )

    # --- Subband & Channel filter ---
    all_subbands = (
        sorted(filtered_df["subband"].unique().tolist())
        if "subband" in filtered_df.columns else []
    )
    all_channels_batch = (
        sorted(filtered_df["channel"].unique().tolist())
        if "channel" in filtered_df.columns else []
    )

    if all_subbands or all_channels_batch:
        st.markdown(
            '<p class="section-title">Filter Hasil</p>',
            unsafe_allow_html=True,
        )
        fc1, fc2 = st.columns(2)

        selected_subbands = all_subbands
        selected_channels_batch = all_channels_batch

        with fc1:
            if all_subbands:
                selected_subbands = st.multiselect(
                    "Subband", all_subbands, default=all_subbands,
                    key="batch_subband_filter",
                )
        with fc2:
            if all_channels_batch:
                selected_channels_batch = st.multiselect(
                    "Channel", all_channels_batch, default=all_channels_batch,
                    key="batch_channel_filter",
                )

        if selected_subbands and len(selected_subbands) < len(all_subbands):
            filtered_df = filtered_df[
                filtered_df["subband"].isin(selected_subbands)
            ].copy()
            feat_cols = [c for c in filtered_df.columns if c not in meta_cols]

        if selected_channels_batch and len(selected_channels_batch) < len(all_channels_batch):
            filtered_df = filtered_df[
                filtered_df["channel"].isin(selected_channels_batch)
            ].copy()
            feat_cols = [c for c in filtered_df.columns if c not in meta_cols]



    st.markdown(
        '<p class="section-title">Ringkasan Batch</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kategori", active_category)
    c2.metric(
        "Subjek",
        filtered_df["subject"].nunique()
        if "subject" in filtered_df.columns else "–",
    )
    c3.metric("Task Ditemukan", len(batch_tasks))
    c4.metric("Total Baris", f"{len(filtered_df):,}")

    # Task badges
    badge_html = ""
    for t in batch_tasks:
        color = TASK_COLORS.get(t, ACCENT_LIGHT)
        badge_html += (
            f'<span class="task-badge" style="background:{color}">{t}</span>'
        )
    st.markdown(
        f'<div style="margin:8px 0">{badge_html}</div>',
        unsafe_allow_html=True,
    )

    # --- Tabs ---
    tab_overview, tab_delta, tab_stats, tab_data = st.tabs(
        ["Distribusi Fitur", "Analisis Delta", "Statistik", "Data Mentah"]
    )

    # ---- Tab 1: Overview (Grouped Bar) ----
    with tab_overview:
        if feat_cols:
            sel_feat = st.selectbox("Fitur", feat_cols, key="batch_feat_overview")

            available_tasks = (
                sorted(filtered_df["task"].unique().tolist())
                if "task" in filtered_df.columns else []
            )
            if available_tasks:
                sel_tasks_dist = st.multiselect(
                    "Filter Task", available_tasks, default=available_tasks,
                    key="dist_task_filter",
                )
                dist_df = (
                    filtered_df[filtered_df["task"].isin(sel_tasks_dist)].copy()
                    if sel_tasks_dist else filtered_df
                )
            else:
                dist_df = filtered_df

            st.markdown("**Pilih Visualisasi:**")
            show_grouped = st.checkbox(
                "Grouped Bar (per Task & Subband)",
                value=False, key="show_grouped_bar",
            )
            show_box = st.checkbox(
                "Box Plot Overview", value=False, key="show_box_plot",
            )
            show_summary_table = st.checkbox(
                "Tabel Rata-rata per Task", value=False, key="show_summary_tbl",
            )

            if show_grouped:
                fig_grouped = FeaturePlots.plot_grouped_bar(
                    dist_df, sel_feat, group_col="task",
                    facet_col="channel", x_col="subband",
                    title=f"{sel_feat.capitalize()} per Task & Subband",
                )
                if fig_grouped:
                    st.plotly_chart(fig_grouped, use_container_width=True)

            if show_box:
                fig_box = FeaturePlots.plot_batch_overview(dist_df, sel_feat)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)

            if show_summary_table:
                st.markdown("**Rata-rata Fitur per Task**")
                summary = dist_df.groupby("task")[feat_cols].agg(["mean", "std"])
                summary.columns = [f"{f} ({s})" for f, s in summary.columns]
                st.dataframe(summary, use_container_width=True)

    # ---- Tab 2: Delta ----
    with tab_delta:
        _render_delta_tab(
            filtered_df, batch_df, batch_tasks, feat_cols,
            categories, selected_scenarios, selected_times,
        )

    # ---- Tab 3: Statistik (BARU) ----
    with tab_stats:
        _render_stats_tab(filtered_df, batch_tasks, feat_cols)

    # ---- Tab 4: Raw Data ----
    with tab_data:
        st.markdown("**Seluruh Data Fitur Batch**")
        st.caption(
            f"Total: {len(filtered_df):,} baris × {len(filtered_df.columns)} kolom"
        )

        MAX_DISPLAY = 500
        if len(filtered_df) > MAX_DISPLAY:
            st.dataframe(
                filtered_df.head(MAX_DISPLAY), use_container_width=True,
                height=400, hide_index=True,
            )
            st.info(
                f"Menampilkan {MAX_DISPLAY} baris pertama dari "
                f"{len(filtered_df):,} total. "
                "Gunakan tombol download untuk data lengkap."
            )
        else:
            st.dataframe(
                filtered_df, use_container_width=True, height=400,
                hide_index=True,
            )

        @st.cache_data
        def _convert_to_csv(df):
            return df.to_csv(index=False).encode("utf-8")

        csv_data = _convert_to_csv(filtered_df)
        st.download_button(
            "Download CSV", csv_data,
            file_name="batch_features.csv", mime="text/csv",
        )


def _render_delta_tab(filtered_df, batch_df, batch_tasks, feat_cols,
                       categories, selected_scenarios, selected_times):
    """Render tab analisis delta."""
    if len(batch_tasks) < 2:
        st.info("Diperlukan minimal 2 task untuk analisis delta.")
        return

    st.markdown(
        '<p class="section-title">Perbandingan Delta antar Task</p>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        task_a = st.selectbox(
            "Task A (Kondisi)", batch_tasks, index=0, key="delta_task_a",
        )
    with col_b:
        default_b = 1 if len(batch_tasks) > 1 else 0
        task_b = st.selectbox(
            "Task B (Baseline)", batch_tasks, index=default_b, key="delta_task_b",
        )

    if task_a == task_b:
        st.warning("Pilih dua task yang berbeda.")
        return

    sel_delta_feat = st.selectbox(
        "Fitur Delta", feat_cols, key="delta_feat_select",
    )

    delta_df, agg_df = DeltaCalculator.calculate_task_delta(
        filtered_df, task_a, task_b, feat_cols,
    )

    if delta_df.empty:
        st.warning(f"Tidak ada data yang cocok untuk {task_a} vs {task_b}.")
        return

    # Pilih visualisasi
    st.markdown("**Pilih Visualisasi:**")
    show_delta_bar = st.checkbox(
        "Bar Chart Delta", value=False, key="show_delta_bar",
    )
    show_delta_scatter = st.checkbox(
        "Scatter Plot per Subjek", value=False, key="show_delta_scatter",
    )
    show_delta_heat = st.checkbox(
        "Heatmap Delta", value=False, key="show_delta_heat",
    )
    show_transition = st.checkbox(
        "Transition Delta per Group (BARU)", value=False, key="show_transition",
    )
    show_delta_table = st.checkbox(
        "Tabel Delta & Agregat", value=False, key="show_delta_tbl",
    )

    dcol = f"delta_{sel_delta_feat}"

    if show_delta_bar:
        fig_bar = ComparisonPlots.plot_delta_bar(
            agg_df, sel_delta_feat, task_a, task_b,
        )
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- Transition Delta per Group (BARU dari pipeline) ---
    if show_transition:
        st.markdown("**Transition Delta per Group (Per-Subject)**")
        st.caption(
            "Delta dihitung per pasien dulu, lalu rata-rata per group. "
            "Lebih akurat secara statistik."
        )
        transition_df = DeltaCalculator.compute_transition_table(
            filtered_df, task_a, task_b, sel_delta_feat,
            scenarios=selected_scenarios if selected_scenarios else None,
            sessions=selected_times if selected_times else None,
        )
        if not transition_df.empty:
            fig_trans = ComparisonPlots.plot_transition_deltas(
                transition_df, sel_delta_feat,
                title=f"Transition: {task_a} → {task_b}",
            )
            if fig_trans:
                st.plotly_chart(fig_trans, use_container_width=True)
            st.dataframe(
                transition_df, use_container_width=True, hide_index=True,
            )

    if show_delta_scatter and dcol in delta_df.columns:
        # Build meta_map only from columns that exist
        meta_cols_available = ["filename"]
        for mc in ["subject", "category"]:
            if mc in filtered_df.columns:
                meta_cols_available.append(mc)

        meta_map = filtered_df.drop_duplicates(
            subset=["filename"]
        )[meta_cols_available].copy()

        # Avoid merge conflicts if delta_df already has subject/category
        merge_cols = [c for c in meta_cols_available if c not in delta_df.columns or c == "filename"]
        extra_cols = [c for c in meta_cols_available if c != "filename" and c not in delta_df.columns]
        if extra_cols:
            scatter_df = delta_df.merge(meta_map[["filename"] + extra_cols], on="filename", how="left")
        else:
            scatter_df = delta_df.copy()

        scatter_df["ch_sub"] = (
            scatter_df["channel"] + " / " + scatter_df["subband"]
        )

        has_category = "category" in scatter_df.columns
        has_subject = "subject" in scatter_df.columns

        if has_category:
            scatter_cats = sorted(scatter_df["category"].dropna().unique().tolist())
            if scatter_cats:
                sel_scatter_cat = st.multiselect(
                    "Filter Kategori (Scatter)", scatter_cats,
                    default=scatter_cats, key="scatter_cat_filter",
                )
                if sel_scatter_cat:
                    scatter_df = scatter_df[
                        scatter_df["category"].isin(sel_scatter_cat)
                    ]

        fig_scatter = go.Figure()

        if has_subject:
            subjects = sorted(scatter_df["subject"].dropna().unique().tolist())
            for subj in subjects:
                s_data = scatter_df[scatter_df["subject"] == subj]
                cat_label = s_data["category"].iloc[0] if has_category and not s_data.empty else ""
                fig_scatter.add_trace(go.Scatter(
                    x=s_data["ch_sub"], y=s_data[dcol],
                    mode="markers",
                    name=f"{subj} ({cat_label})" if cat_label else subj,
                    marker=dict(size=10, opacity=0.8),
                    hovertemplate=(
                        f"<b>{subj}</b>" + (f" ({cat_label})" if cat_label else "") + "<br>"
                        "Channel/Subband: %{x}<br>"
                        f"Δ {sel_delta_feat}: " + "%{y:.6e}<br>"
                        "<extra></extra>"
                    ),
                ))
        else:
            # Fallback: plot by filename
            filenames = sorted(scatter_df["filename"].dropna().unique().tolist())
            for fname in filenames[:20]:  # Limit to 20 files
                s_data = scatter_df[scatter_df["filename"] == fname]
                short_name = fname.replace(".edf", "").split("/")[-1]
                fig_scatter.add_trace(go.Scatter(
                    x=s_data["ch_sub"], y=s_data[dcol],
                    mode="markers",
                    name=short_name,
                    marker=dict(size=10, opacity=0.8),
                ))

        fig_scatter.update_layout(
            title=f"Δ {sel_delta_feat.capitalize()} per Subjek ({task_a} − {task_b})",
            xaxis_title="Channel / Subband",
            yaxis_title=f"Δ {sel_delta_feat}",
            template="plotly_dark",
            height=500,
            legend_title="Subjek" if has_subject else "File",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_scatter.add_hline(
            y=0, line_dash="dash", line_color="gray", opacity=0.5,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    if show_delta_heat:
        fig_heat = ComparisonPlots.plot_delta_heatmap(
            agg_df, sel_delta_feat, task_a, task_b,
        )
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)

    if show_delta_table:
        with st.expander("Tabel Delta Lengkap", expanded=True):
            st.dataframe(
                delta_df, use_container_width=True,
                height=320, hide_index=True,
            )
            dl1, dl2 = st.columns(2)
            with dl1:
                csv_data = delta_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV", csv_data,
                    file_name=f"delta_{task_a}_vs_{task_b}.csv",
                    mime="text/csv", key="dl_delta_csv",
                )
            with dl2:
                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    delta_df.to_excel(writer, index=False, sheet_name="Delta")
                    agg_df.to_excel(writer, index=False, sheet_name="Agregat")
                st.download_button(
                    "Download Excel", excel_buf.getvalue(),
                    file_name=f"delta_{task_a}_vs_{task_b}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_delta_xlsx",
                )
        with st.expander("Statistik Agregat per Channel/Subband"):
            st.dataframe(
                agg_df, use_container_width=True, hide_index=True,
            )

    # --- Perbandingan ALS vs Normal ---
    has_both_cats = (
        "category" in batch_df.columns
        and set(batch_df["category"].unique()) >= {"ALS", "Normal"}
    )
    if has_both_cats:
        st.divider()
        st.markdown(
            '<p class="section-title">Perbandingan ALS vs Normal</p>',
            unsafe_allow_html=True,
        )

        show_als_compare = st.checkbox(
            "Tampilkan Perbandingan ALS vs Normal",
            value=False, key="show_als_compare",
        )

        if show_als_compare:
            _render_als_comparison(
                batch_df, filtered_df, task_a, task_b,
                sel_delta_feat, feat_cols,
                categories, selected_scenarios, selected_times,
            )


def _render_als_comparison(batch_df, filtered_df, task_a, task_b,
                            sel_delta_feat, feat_cols,
                            categories, selected_scenarios, selected_times):
    """Render perbandingan ALS vs Normal."""
    cmp_mode_label = st.radio(
        "Metode Perbandingan",
        ["Delta saja", "Z-score saja", "Z-score + Delta"],
        horizontal=True, key="cmp_mode",
        help="Delta: selisih Task A − Task B. "
             "Z-score: normalisasi per subjek. "
             "Z-score + Delta: normalisasi dulu, baru hitung selisih.",
    )
    cmp_mode_map = {
        "Delta saja": "delta",
        "Z-score saja": "zscore",
        "Z-score + Delta": "both",
    }
    cmp_mode = cmp_mode_map[cmp_mode_label]

    # --- Opsi statistik (sekarang aktif!) ---
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        use_fdr = st.checkbox(
            "Koreksi FDR (Benjamini-Hochberg)",
            value=True, key="use_fdr",
        )
    with opt_col2:
        use_effect = st.checkbox(
            "Hitung Effect Size (Cohen's d)",
            value=True, key="use_effect",
        )

    use_ttest = st.checkbox("T-test Independent (BARU)", value=True, key="use_ttest")

    # --- Data source ---
    compare_src = batch_df.copy()
    if selected_scenarios and len(selected_scenarios) > 0:
        if "scenario" in compare_src.columns:
            compare_src = compare_src[
                compare_src["scenario"].isin(selected_scenarios)
            ]
    if selected_times and len(selected_times) > 0:
        if "time" in compare_src.columns:
            compare_src = compare_src[
                compare_src["time"].isin(selected_times)
            ]

    # --- Common time control ---
    if "time" in compare_src.columns:
        use_common_time = st.checkbox(
            "Hanya gunakan time yang dimiliki semua subjek",
            value=False, key="common_time",
        )
        if use_common_time:
            subj_times = compare_src.groupby("subject")["time"].apply(set)
            if len(subj_times) > 0:
                common_times = set.intersection(*subj_times)
                if common_times:
                    compare_src = compare_src[
                        compare_src["time"].isin(common_times)
                    ]
                    st.info(
                        f"Time yang sama di semua subjek: {sorted(common_times)}"
                    )
                else:
                    st.warning("Tidak ada time yang sama di semua subjek.")

    # --- Sampling ---
    als_subjects = sorted(
        compare_src[compare_src["category"] == "ALS"]["subject"].unique()
    )
    norm_subjects = sorted(
        compare_src[compare_src["category"] == "Normal"]["subject"].unique()
    )

    st.markdown(
        f"Tersedia: **{len(als_subjects)} ALS**, "
        f"**{len(norm_subjects)} Normal**"
    )

    samp_col1, samp_col2 = st.columns(2)
    with samp_col1:
        n_als = st.number_input(
            "Jumlah Sampel ALS", min_value=1,
            max_value=max(len(als_subjects), 1),
            value=max(min(len(als_subjects), len(norm_subjects)), 1),
            key="delta_n_als",
        )
    with samp_col2:
        n_norm = st.number_input(
            "Jumlah Sampel Normal", min_value=1,
            max_value=max(len(norm_subjects), 1),
            value=max(min(len(als_subjects), len(norm_subjects)), 1),
            key="delta_n_norm",
        )

    seed = st.number_input(
        "Random Seed", value=42, key="delta_seed",
        help="Ganti seed untuk sampling berbeda",
    )
    rng = random.Random(seed)

    sampled_als = (
        rng.sample(als_subjects, min(n_als, len(als_subjects)))
        if als_subjects else []
    )
    sampled_norm = (
        rng.sample(norm_subjects, min(n_norm, len(norm_subjects)))
        if norm_subjects else []
    )

    sampled_subjects = sampled_als + sampled_norm
    sampled_df = compare_src[compare_src["subject"].isin(sampled_subjects)]

    if sampled_df.empty:
        st.warning("Tidak ada data setelah sampling.")
        return

    avs_cmp_df, avs_stats_df = StatisticalTests.compare_als_vs_normal(
        sampled_df, task_a, baseline_task=task_b,
        feature_cols=feat_cols, mode=cmp_mode,
        apply_fdr=use_fdr, compute_effect_size=use_effect,
        include_ttest=use_ttest,
    )

    if avs_cmp_df.empty or avs_stats_df.empty:
        st.warning("Tidak cukup data untuk perbandingan.")
        return

    # -- Metrics --
    col_p = f"p_{sel_delta_feat}"
    col_d = f"cohend_{sel_delta_feat}"

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Sampel ALS", len(sampled_als))
    mc2.metric("Sampel Normal", len(sampled_norm))
    if col_p in avs_stats_df.columns:
        sig_raw = (avs_stats_df[col_p] <= 0.05).sum()
        tot_n = avs_stats_df[col_p].notna().sum()
        mc3.metric("Signifikan (p≤0.05)", f"{sig_raw}/{tot_n}")
    if use_fdr:
        col_fdr = f"p_fdr_{sel_delta_feat}"
        if col_fdr in avs_stats_df.columns:
            sig_fdr = (avs_stats_df[col_fdr] <= 0.05).sum()
            mc4.metric("Signifikan (FDR≤0.05)", f"{sig_fdr}/{tot_n}")

    # -- Grouped bar chart --
    fig_avs = ComparisonPlots.plot_als_vs_normal(
        avs_stats_df, sel_delta_feat, task_a, task_b,
        use_sem=True,
    )
    if fig_avs:
        st.plotly_chart(fig_avs, use_container_width=True)

    # -- P-value table --
    p_cols = [
        c for c in avs_stats_df.columns
        if c.startswith("p_") and not c.startswith("p_fdr_")
    ]
    if p_cols:
        disp = avs_stats_df[["channel", "subband"] + p_cols].copy()

        def _hl(val):
            if isinstance(val, float) and val <= 0.05:
                return (
                    "background-color: #065F46; "
                    "color: #34D399; font-weight: bold"
                )
            return ""

        styled = disp.style.map(_hl, subset=p_cols).format(
            {c: "{:.4f}" for c in p_cols}, na_rep="–",
        )
        st.markdown("**Tabel P-value (Mann-Whitney U)**")
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # -- T-test table (BARU) --
    if use_ttest:
        t_cols = [c for c in avs_stats_df.columns if c.startswith("t_pval_")]
        if t_cols:
            t_disp = avs_stats_df[["channel", "subband"] + t_cols].copy()

            def _hl_t(val):
                if isinstance(val, float) and val <= 0.05:
                    return (
                        "background-color: #1E3A5F; "
                        "color: #60A5FA; font-weight: bold"
                    )
                return ""

            styled_t = t_disp.style.map(_hl_t, subset=t_cols).format(
                {c: "{:.4f}" for c in t_cols}, na_rep="–",
            )
            st.markdown("**Tabel P-value (T-test)**")
            st.dataframe(styled_t, use_container_width=True, hide_index=True)

    # -- Effect size table (BARU — sekarang aktif!) --
    if use_effect:
        d_cols = [c for c in avs_stats_df.columns if c.startswith("cohend_")]
        e_cols = [c for c in avs_stats_df.columns if c.startswith("effect_")]
        if d_cols:
            d_disp = avs_stats_df[
                ["channel", "subband"] + d_cols + e_cols
            ].copy()

            def _hl_d(val):
                if isinstance(val, float) and abs(val) >= 0.8:
                    return (
                        "background-color: #1E3A5F; "
                        "color: #60A5FA; font-weight: bold"
                    )
                return ""

            styled_d = d_disp.style.map(_hl_d, subset=d_cols).format(
                {c: "{:.3f}" for c in d_cols}, na_rep="–",
            )
            st.markdown("**Effect Size (Cohen's d)**")
            st.caption(
                "Interpretasi: |d| < 0.2 = Sangat Kecil, "
                "< 0.5 = Kecil, < 0.8 = Sedang, ≥ 0.8 = Besar"
            )
            st.dataframe(styled_d, use_container_width=True, hide_index=True)

    # -- Excel download dengan multi-sheet (BARU dari pipeline) --
    st.divider()
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        avs_cmp_df.to_excel(writer, index=False, sheet_name="Group Comparison")
        avs_stats_df.to_excel(writer, index=False, sheet_name="Statistics")
    st.download_button(
        "Download Laporan Excel",
        excel_buf.getvalue(),
        file_name=f"als_vs_normal_{task_a}_vs_{task_b}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_avs_xlsx",
    )


def _render_stats_tab(filtered_df, batch_tasks, feat_cols):
    """Tab statistik (BARU) — ringkasan uji statistik."""
    has_both = (
        "category" in filtered_df.columns
        and set(filtered_df["category"].unique()) >= {"ALS", "Normal"}
    )
    if not has_both:
        st.info(
            "Tab ini memerlukan data dari kedua kategori (ALS dan Normal)."
        )
        return

    if len(batch_tasks) < 1:
        st.info("Diperlukan minimal 1 task.")
        return

    st.markdown(
        '<p class="section-title">Ringkasan Statistik per Task</p>',
        unsafe_allow_html=True,
    )

    sel_task_stat = st.selectbox(
        "Task", batch_tasks, key="stat_task_select",
    )
    sel_feat_stat = st.selectbox(
        "Fitur", feat_cols, key="stat_feat_select",
    )

    # Compute stats per subband
    from processing.statistics import cohens_d, interpret_cohens_d
    from scipy.stats import mannwhitneyu, ttest_ind, sem

    task_data = filtered_df[filtered_df["task"] == sel_task_stat]
    subbands = sorted(task_data["subband"].unique()) if "subband" in task_data.columns else []

    stat_rows = []
    for sb in subbands:
        sb_data = task_data[task_data["subband"] == sb]
        als_vals = sb_data[sb_data["category"] == "ALS"][sel_feat_stat].dropna()
        norm_vals = sb_data[sb_data["category"] == "Normal"][sel_feat_stat].dropna()

        row = {
            "Subband": sb,
            "ALS Mean": f"{als_vals.mean():.4e}" if len(als_vals) else "–",
            "ALS SEM": f"{sem(als_vals):.4e}" if len(als_vals) >= 2 else "–",
            "ALS N": len(als_vals),
            "Normal Mean": f"{norm_vals.mean():.4e}" if len(norm_vals) else "–",
            "Normal SEM": f"{sem(norm_vals):.4e}" if len(norm_vals) >= 2 else "–",
            "Normal N": len(norm_vals),
        }

        if len(als_vals) >= 2 and len(norm_vals) >= 2:
            _, u_p = mannwhitneyu(als_vals, norm_vals, alternative="two-sided")
            _, t_p = ttest_ind(als_vals, norm_vals)
            d = cohens_d(als_vals.values, norm_vals.values)
            row["U p-value"] = f"{u_p:.4f}"
            row["T p-value"] = f"{t_p:.4f}"
            row["Cohen's d"] = f"{d:.3f}"
            row["Effect Size"] = interpret_cohens_d(d)
        else:
            row["U p-value"] = "–"
            row["T p-value"] = "–"
            row["Cohen's d"] = "–"
            row["Effect Size"] = "–"

        stat_rows.append(row)

    if stat_rows:
        stat_summary = pd.DataFrame(stat_rows)
        st.dataframe(stat_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada data yang cukup.")
