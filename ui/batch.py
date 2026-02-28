"""
Modul batch -- Halaman analisis batch (ZIP).
"""

import io
import os
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

from visualization.feature_plots import FeaturePlots
from visualization.comparison_plots import ComparisonPlots


# ---------------------------------------------------------------------------
# Worker: proses satu file EDF
# ---------------------------------------------------------------------------

def _process_single_edf(zip_bytes, edf_path, channels, subbands, features,
                         include_frequency=False):
    """Worker: proses satu file EDF dari ZIP bytes (thread-safe)."""
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


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

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
    include_freq = cfg.get("include_frequency", False)

    progress_bar = st.progress(0, text="Memulai batch processing...")

    uploaded.seek(0)
    zip_bytes = uploaded.read()
    uploaded.seek(0)
    edf_list = EEGLoader.list_edf_in_zip(uploaded)

    if not edf_list:
        st.error("Tidak ditemukan file EDF dalam ZIP.")
        return

    all_dfs = []
    common_tasks = set()
    n_total = len(edf_list)

    max_workers = min(4, max(1, os.cpu_count() or 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_edf, zip_bytes, path,
                channels, subbands, features, include_freq,
            ): path
            for path in edf_list
        }

        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            progress_bar.progress(
                i / n_total,
                text=f"Memproses {i}/{n_total}: {path.split('/')[-1]}",
            )
            try:
                feat_df, tasks_found = future.result()
                if feat_df is not None and not feat_df.empty:
                    all_dfs.append(feat_df)
                    if not common_tasks:
                        common_tasks = set(tasks_found)
                    else:
                        common_tasks &= set(tasks_found)
            except Exception:
                pass

    progress_bar.empty()

    if not all_dfs:
        st.error("Tidak ada data fitur yang berhasil diekstrak.")
        return

    batch_df = pd.concat(all_dfs, ignore_index=True)
    if common_tasks is None:
        common_tasks = set()

    st.session_state.batch_df = batch_df
    st.session_state.batch_tasks = sorted(common_tasks)
    st.session_state.batch_processed = True
    st.success(
        f"Batch processing selesai: {batch_df['filename'].nunique()} file, "
        f"{len(common_tasks)} task ditemukan."
    )


# ---------------------------------------------------------------------------
# Render hasil batch
# ---------------------------------------------------------------------------

def render_batch_results(cfg):
    """Tampilkan hasil batch analysis & delta comparison."""
    batch_df = st.session_state.batch_df
    batch_tasks = st.session_state.batch_tasks

    if batch_df is None or batch_df.empty:
        return

    # Kumpulkan opsi filter
    categories = sorted(batch_df["category"].unique().tolist()) if "category" in batch_df.columns else []
    scenarios = sorted(batch_df["scenario"].unique().tolist()) if "scenario" in batch_df.columns else []
    times = sorted(batch_df["time"].unique().tolist()) if "time" in batch_df.columns else []
    all_subbands = sorted(batch_df["subband"].unique().tolist()) if "subband" in batch_df.columns else []
    all_channels = sorted(batch_df["channel"].unique().tolist()) if "channel" in batch_df.columns else []

    # Panel filter
    with st.expander("Konfigurasi & Filter Batch", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            active_category = "Semua"
            if len(categories) > 1:
                cat_counts = batch_df.groupby("category")["subject"].nunique()
                cat_labels = [f"Semua ({batch_df['subject'].nunique()})"]
                for cat in categories:
                    cat_labels.append(f"{cat} ({cat_counts.get(cat, 0)})")
                selected_cat_label = st.selectbox("Kategori", cat_labels, key="cat_filter")
                if not selected_cat_label.startswith("Semua"):
                    active_category = selected_cat_label.split(" (")[0]
            elif categories:
                active_category = categories[0]

        with col2:
            selected_scenarios = scenarios
            if len(scenarios) > 1:
                selected_scenarios = st.multiselect("Filter Skenario", scenarios, default=scenarios, key="scenario_filter")
            st.markdown("---")
            selected_subbands = all_subbands
            if all_subbands:
                selected_subbands = st.multiselect("Filter Subband", all_subbands, default=all_subbands, key="batch_subband")

        with col3:
            selected_times = times
            if len(times) > 1:
                selected_times = st.multiselect("Filter Time", times, default=times, key="time_filter")
            st.markdown("---")
            selected_channels_batch = all_channels
            if all_channels:
                selected_channels_batch = st.multiselect("Filter Channel", all_channels, default=all_channels, key="batch_channel")

    # Aplikasikan filter
    filtered_df = batch_df.copy()
    if active_category != "Semua":
        filtered_df = filtered_df[filtered_df["category"] == active_category]
    if selected_scenarios and len(selected_scenarios) < len(scenarios):
        filtered_df = filtered_df[filtered_df["scenario"].isin(selected_scenarios)]
    if selected_times and len(selected_times) < len(times):
        filtered_df = filtered_df[filtered_df["time"].isin(selected_times)]
    if selected_subbands and len(selected_subbands) < len(all_subbands):
        filtered_df = filtered_df[filtered_df["subband"].isin(selected_subbands)]
    if selected_channels_batch and len(selected_channels_batch) < len(all_channels):
        filtered_df = filtered_df[filtered_df["channel"].isin(selected_channels_batch)]

    meta_cols = {"filename", "category", "subject", "time", "scenario", "task", "channel", "subband"}
    feat_cols = [c for c in filtered_df.columns if c not in meta_cols]

    # Ringkasan
    st.markdown('<p class="section-title">Ringkasan Batch</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kategori", active_category)
    c2.metric("Subjek", filtered_df["subject"].nunique() if "subject" in filtered_df.columns else "-")
    c3.metric("Task Ditemukan", len(batch_tasks))
    c4.metric("Total Baris", f"{len(filtered_df):,}")

    # Task badges
    badge_html = ""
    for t in batch_tasks:
        color = TASK_COLORS.get(t, ACCENT_LIGHT)
        badge_html += f'<span class="task-badge" style="background:{color}">{t}</span>'
    st.markdown(f'<div style="margin:8px 0">{badge_html}</div>', unsafe_allow_html=True)

    # Tab: hanya Analisis Delta
    _render_delta_tab(
        filtered_df, batch_df, batch_tasks, feat_cols,
        categories, selected_scenarios, selected_times,
    )


# ---------------------------------------------------------------------------
# Tab Analisis Delta
# ---------------------------------------------------------------------------

def _render_delta_tab(filtered_df, batch_df, batch_tasks, feat_cols,
                       categories, selected_scenarios, selected_times):
    """Render analisis delta."""
    if len(batch_tasks) < 2:
        st.info("Diperlukan minimal 2 task untuk analisis delta.")
        return

    st.markdown('<p class="section-title">Perbandingan Delta antar Task</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        task_a = st.selectbox("Task A (Kondisi)", batch_tasks, index=0, key="delta_task_a")
    with col_b:
        default_b = 1 if len(batch_tasks) > 1 else 0
        task_b = st.selectbox("Task B (Baseline)", batch_tasks, index=default_b, key="delta_task_b")

    if task_a == task_b:
        st.warning("Pilih dua task yang berbeda.")
        return

    sel_delta_feat = st.selectbox("Fitur Delta", feat_cols, key="delta_feat_select")

    delta_df, agg_df = DeltaCalculator.calculate_task_delta(filtered_df, task_a, task_b, feat_cols)

    if delta_df.empty:
        st.warning(f"Tidak ada data yang cocok untuk {task_a} vs {task_b}.")
        return

    # Visualisasi checkboxes
    st.markdown("**Pilih Visualisasi:**")
    show_delta_table = st.checkbox("Tabel Delta", value=False, key="show_delta_tbl")
    show_transition = st.checkbox("Transition Delta per Group", value=False, key="show_transition")
    show_delta_scatter = st.checkbox("Scatter Plot per Subjek", value=False, key="show_delta_scatter")
    show_delta_heat = st.checkbox("Heatmap Delta", value=False, key="show_delta_heat")

    dcol = f"delta_{sel_delta_feat}"

    # Tabel
    if show_delta_table:
        _render_delta_tables(delta_df, agg_df, task_a, task_b)

    # Transition
    if show_transition:
        _render_transition_delta(
            filtered_df, task_a, task_b, sel_delta_feat,
            selected_scenarios, selected_times,
        )

    # Scatter
    if show_delta_scatter and dcol in delta_df.columns:
        _render_scatter_plot(delta_df, filtered_df, dcol, sel_delta_feat, task_a, task_b)

    # Heatmap
    if show_delta_heat:
        fig_heat = ComparisonPlots.plot_delta_heatmap(agg_df, sel_delta_feat, task_a, task_b)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------------
# Sub-renders untuk delta tab
# ---------------------------------------------------------------------------

def _render_delta_tables(delta_df, agg_df, task_a, task_b):
    """Render tabel delta dan agregat."""
    with st.expander("Tabel Delta Lengkap", expanded=True):
        st.dataframe(delta_df, use_container_width=True, height=320, hide_index=True)
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



def _render_transition_delta(filtered_df, task_a, task_b, sel_delta_feat,
                              selected_scenarios, selected_times):
    """Render transition delta per group."""
    st.markdown("**Transition Delta per Group (Per-Subject)**")
    st.caption("Delta dihitung per pasien dulu, lalu rata-rata per group.")
    transition_df = DeltaCalculator.compute_transition_table(
        filtered_df, task_a, task_b, sel_delta_feat,
        scenarios=selected_scenarios if selected_scenarios else None,
        sessions=selected_times if selected_times else None,
    )
    if not transition_df.empty:
        fig_trans = ComparisonPlots.plot_transition_deltas(
            transition_df, sel_delta_feat,
            title=f"Transition: {task_a} -> {task_b}",
        )
        if fig_trans:
            st.plotly_chart(fig_trans, use_container_width=True)
        st.dataframe(transition_df, use_container_width=True, hide_index=True)


def _render_scatter_plot(delta_df, filtered_df, dcol, sel_delta_feat, task_a, task_b):
    """Render scatter plot delta per subjek."""
    meta_cols_available = ["filename"]
    for mc in ["subject", "category"]:
        if mc in filtered_df.columns:
            meta_cols_available.append(mc)

    meta_map = filtered_df.drop_duplicates(subset=["filename"])[meta_cols_available].copy()

    extra_cols = [c for c in meta_cols_available if c != "filename" and c not in delta_df.columns]
    if extra_cols:
        scatter_df = delta_df.merge(meta_map[["filename"] + extra_cols], on="filename", how="left")
    else:
        scatter_df = delta_df.copy()

    scatter_df["ch_sub"] = scatter_df["channel"] + " / " + scatter_df["subband"]

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
                scatter_df = scatter_df[scatter_df["category"].isin(sel_scatter_cat)]

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
            ))
    else:
        filenames = sorted(scatter_df["filename"].dropna().unique().tolist())
        for fname in filenames[:20]:
            s_data = scatter_df[scatter_df["filename"] == fname]
            short_name = fname.replace(".edf", "").split("/")[-1]
            fig_scatter.add_trace(go.Scatter(
                x=s_data["ch_sub"], y=s_data[dcol],
                mode="markers", name=short_name,
                marker=dict(size=10, opacity=0.8),
            ))

    fig_scatter.update_layout(
        title=f"Delta {sel_delta_feat.capitalize()} per Subjek ({task_a} - {task_b})",
        xaxis_title="Channel / Subband",
        yaxis_title=f"Delta {sel_delta_feat}",
        template="plotly_dark",
        height=500,
        legend_title="Subjek" if has_subject else "File",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig_scatter, use_container_width=True)
