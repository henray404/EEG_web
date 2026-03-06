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
from processing.features import EEGFeatures

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

    feat_df = EEGFeatures.compute_first_occurrence_features(
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
                        common_tasks |= set(tasks_found)
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
        # --- Select All / Hapus Semua helpers ---
        def _select_all(key, options):
            st.session_state[key] = list(options)
        def _clear_all(key):
            st.session_state[key] = []

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
                sa1, sa2 = st.columns(2)
                sa1.button("Semua", key="sa_sc", on_click=_select_all, args=("scenario_filter", scenarios))
                sa2.button("Hapus", key="cl_sc", on_click=_clear_all, args=("scenario_filter",))
                selected_scenarios = st.multiselect("Filter Skenario", scenarios, default=[], key="scenario_filter")
            st.markdown("---")
            selected_subbands = all_subbands
            if all_subbands:
                sb1, sb2 = st.columns(2)
                sb1.button("Semua", key="sa_sb", on_click=_select_all, args=("batch_subband", all_subbands))
                sb2.button("Hapus", key="cl_sb", on_click=_clear_all, args=("batch_subband",))
                selected_subbands = st.multiselect("Filter Subband", all_subbands, default=[], key="batch_subband")

        with col3:
            selected_times = times
            if len(times) > 1:
                ta1, ta2 = st.columns(2)
                ta1.button("Semua", key="sa_tm", on_click=_select_all, args=("time_filter", times))
                ta2.button("Hapus", key="cl_tm", on_click=_clear_all, args=("time_filter",))
                selected_times = st.multiselect("Filter Time", times, default=[], key="time_filter")
            st.markdown("---")
            selected_channels_batch = all_channels
            if all_channels:
                ca1, ca2 = st.columns(2)
                ca1.button("Semua", key="sa_ch", on_click=_select_all, args=("batch_channel", all_channels))
                ca2.button("Hapus", key="cl_ch", on_click=_clear_all, args=("batch_channel",))
                selected_channels_batch = st.multiselect("Filter Channel", all_channels, default=[], key="batch_channel")

    # Aplikasikan filter
    filtered_df = batch_df.copy()
    if active_category != "Semua":
        filtered_df = filtered_df[filtered_df["category"] == active_category]
    if selected_scenarios:
        filtered_df = filtered_df[filtered_df["scenario"].isin(selected_scenarios)]
    if selected_times:
        filtered_df = filtered_df[filtered_df["time"].isin(selected_times)]
    if selected_subbands:
        filtered_df = filtered_df[filtered_df["subband"].isin(selected_subbands)]
    if selected_channels_batch:
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

    # Tab: Analisis Delta + Fitur + ERD/ERS
    _render_delta_tab(
        filtered_df, batch_df, batch_tasks, feat_cols,
        categories, selected_scenarios, selected_times,
    )

    # Tabel Fitur per Task (non-delta)
    _render_feature_per_task_table(filtered_df, batch_tasks, feat_cols)

    # ERD/ERS
    _render_erd_ers(filtered_df, batch_tasks, feat_cols)


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
        # Warna per kolom untuk tabel lengkap
        meta_cols = {"filename", "category", "subject", "time", "scenario",
                     "task", "channel", "subband"}

        def _color_full_cols(col):
            if col.name in meta_cols:
                return ["background-color: #e3f2fd"] * len(col)   # biru muda
            elif col.name.endswith(f"_{task_a}"):
                return ["background-color: #e8f5e9"] * len(col)   # hijau muda
            elif col.name.endswith(f"_{task_b}"):
                return ["background-color: #fff3e0"] * len(col)   # oranye muda
            elif col.name.startswith("delta_"):
                return ["background-color: #fce4ec"] * len(col)   # pink muda
            return [""] * len(col)

        styled_full = delta_df.style.apply(_color_full_cols)
        st.dataframe(styled_full, use_container_width=True, height=320, hide_index=True)
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
                _format_micro_units(delta_df).to_excel(writer, index=False, sheet_name="Delta")
                _format_micro_units(agg_df).to_excel(writer, index=False, sheet_name="Agregat")
            st.download_button(
                "Download Excel", excel_buf.getvalue(),
                file_name=f"delta_{task_a}_vs_{task_b}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_delta_xlsx",
            )

        # --- Fokus satu fitur ---
        st.markdown("---")
        st.markdown("**Fokus Satu Fitur**")
        st.caption("Pilih satu fitur untuk menampilkan tabel yang lebih ringkas.")

        meta_cols = {"filename", "category", "subject", "time", "scenario",
                     "task", "channel", "subband"}
        # Deteksi fitur unik dari kolom delta_*
        delta_feat_cols = [c for c in delta_df.columns
                          if c.startswith("delta_")]
        base_features = [c.replace("delta_", "", 1) for c in delta_feat_cols]

        if base_features:
            sel_focus = st.selectbox(
                "Pilih Fitur", base_features, key="focus_feat_select",
            )

            # Kolom yang relevan: meta + task_a value + task_b value + delta
            col_a = f"{sel_focus}_{task_a}"
            col_b = f"{sel_focus}_{task_b}"
            col_delta = f"delta_{sel_focus}"

            # Kumpulkan kolom meta yang ada di delta_df
            keep_cols = [c for c in delta_df.columns if c in meta_cols]
            # Tambahkan kolom fitur yang ada
            for fc in [col_a, col_b, col_delta]:
                if fc in delta_df.columns:
                    keep_cols.append(fc)
            # Fallback: jika kolom per-task tidak ada, cari kolom asli
            if col_a not in delta_df.columns and col_b not in delta_df.columns:
                if sel_focus in delta_df.columns:
                    keep_cols.append(sel_focus)
                if col_delta in delta_df.columns and col_delta not in keep_cols:
                    keep_cols.append(col_delta)

            focus_df = delta_df[keep_cols].copy()

            # Warna per kolom
            def _color_focus_cols(col):
                if col.name in meta_cols:
                    return ["background-color: #e3f2fd"] * len(col)  # biru muda (meta)
                elif col.name == col_a:
                    return ["background-color: #e8f5e9"] * len(col)  # hijau muda (Task A)
                elif col.name == col_b:
                    return ["background-color: #fff3e0"] * len(col)  # oranye muda (Task B)
                elif col.name == col_delta:
                    return ["background-color: #fce4ec"] * len(col)  # pink muda (Delta)
                return [""] * len(col)

            styled_focus = focus_df.style.apply(_color_focus_cols)
            st.dataframe(styled_focus, use_container_width=True, height=320,
                         hide_index=True)

            fl1, fl2 = st.columns(2)
            with fl1:
                csv_focus = focus_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV (Fokus)", csv_focus,
                    file_name=f"delta_{sel_focus}_{task_a}_vs_{task_b}.csv",
                    mime="text/csv", key="dl_focus_csv",
                )
            with fl2:
                excel_focus = io.BytesIO()
                with pd.ExcelWriter(excel_focus, engine="openpyxl") as writer:
                    _format_micro_units(focus_df).to_excel(writer, index=False,
                                     sheet_name=sel_focus[:31])
                st.download_button(
                    "Download Excel (Fokus)", excel_focus.getvalue(),
                    file_name=f"delta_{sel_focus}_{task_a}_vs_{task_b}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_focus_xlsx",
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
        template="plotly_white",
        height=500,
        legend_title="Subjek" if has_subject else "File",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------------------------
# Helper: Excel micro-unit formatting
# ---------------------------------------------------------------------------

def _format_micro_units(df):
    """Konversi kolom numerik yang nilainya sangat kecil ke satuan µ (micro).

    Jika rata-rata absolut kolom < 1e-3, kalikan dengan 1e6 dan tambah (µ)
    pada nama kolom agar lebih mudah dibaca.
    """
    df_out = df.copy()
    rename_map = {}
    for col in df_out.select_dtypes(include=[np.number]).columns:
        mean_abs = df_out[col].abs().mean()
        if 0 < mean_abs < 1e-3:
            df_out[col] = df_out[col] * 1e6
            rename_map[col] = f"{col} (µ)"
    if rename_map:
        df_out.rename(columns=rename_map, inplace=True)
    return df_out


# ---------------------------------------------------------------------------
# Non-delta: Tabel Fitur per Task
# ---------------------------------------------------------------------------

def _render_feature_per_task_table(filtered_df, batch_tasks, feat_cols):
    """Render tabel fitur per task tanpa delta (format pivot)."""
    if not batch_tasks or not feat_cols:
        return

    st.markdown('<p class="section-title">Tabel Fitur per Task</p>', unsafe_allow_html=True)
    st.caption("Data fitur mentah per task, tanpa perhitungan delta.")

    show_feat_table = st.checkbox("Tampilkan Tabel Fitur per Task", value=False, key="show_feat_per_task")
    if not show_feat_table:
        return

    sel_task_table = st.selectbox("Pilih Task", batch_tasks, key="feat_task_select")
    sel_feat_table = st.selectbox("Pilih Fitur", feat_cols, key="feat_table_feat")

    task_df = filtered_df[filtered_df["task"] == sel_task_table].copy()
    if task_df.empty:
        st.warning(f"Tidak ada data untuk task '{sel_task_table}'.")
        return

    # Pivot: baris = subject, kolom = scenario, group by channel+subband
    has_scenario = "scenario" in task_df.columns
    has_subject = "subject" in task_df.columns
    has_channel = "channel" in task_df.columns
    has_subband = "subband" in task_df.columns

    if not (has_subject and has_channel and has_subband):
        st.dataframe(task_df, use_container_width=True, hide_index=True)
        return

    # Subband color mapping
    subband_colors = {
        "Mu": "#FFF3E0",
        "Low_Beta": "#E8F5E9",
        "High_Beta": "#E3F2FD",
        "Alpha": "#FCE4EC",
        "Beta": "#F3E5F5",
        "Delta": "#FFFDE7",
        "Theta": "#E0F7FA",
        "Gamma": "#FBE9E7",
    }

    channels_in_data = sorted(task_df["channel"].unique())
    subbands_in_data = sorted(task_df["subband"].unique())

    for ch in channels_in_data:
        st.markdown(f"**Channel: {ch}**")
        for sb in subbands_in_data:
            sb_df = task_df[
                (task_df["channel"] == ch) & (task_df["subband"] == sb)
            ]
            if sb_df.empty:
                continue

            if has_scenario:
                try:
                    pivot = sb_df.pivot_table(
                        index="subject", columns="scenario",
                        values=sel_feat_table, aggfunc="first",
                    )
                    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
                except Exception:
                    pivot = sb_df[["subject", sel_feat_table]].set_index("subject")
            else:
                pivot = sb_df[["subject", sel_feat_table]].set_index("subject")

            bg_color = subband_colors.get(sb, "#FFFFFF")
            styled = pivot.style.set_properties(**{"background-color": bg_color})
            st.markdown(f"*{sb}* — {sel_feat_table}")
            st.dataframe(styled, use_container_width=True, height=min(35 * (len(pivot) + 1), 400))

        st.markdown("---")

    # Download Excel (Custom Layout)
    import re
    from openpyxl.styles import PatternFill, Alignment, Font

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        has_cat = "category" in task_df.columns
        categories = sorted(task_df["category"].dropna().astype(str).unique().tolist()) if has_cat else ["Semua"]
        if not categories:
            categories = ["Semua"]

        ch_colors = {
            0: "FFE699",  # yellow
            1: "9BC2E6",  # blue
            2: "C6E0B4",  # green
            3: "F4B084",  # orange
            4: "D9D9D9",  # gray
            5: "B4A7D6",  # purple
        }

        for cat in categories:
            if cat == "Semua":
                df_cat = task_df
            else:
                df_cat = task_df[task_df["category"] == cat]
            
            if df_cat.empty:
                continue

            safe_feat = re.sub(r'[\\/*?:\[\]]', '', str(sel_feat_table))
            safe_task = re.sub(r'[\\/*?:\[\]]', '', str(sel_task_table))
            safe_cat = re.sub(r'[\\/*?:\[\]]', '', str(cat))
            
            sheet_name = f"{safe_feat}_{safe_task}_{safe_cat}"[:31]
            
            # Buat sheet via pandas dict biar aman
            pd.DataFrame().to_excel(writer, sheet_name=sheet_name)
            ws = writer.sheets[sheet_name]
            
            # Hapus konten dari cell awal pandas (index & blank)
            for row in ws.iter_rows():
                for cell in row:
                    cell.value = None

            ws.cell(row=1, column=1, value=sel_task_table).font = Font(bold=True)
            
            channels_arr = sorted(df_cat["channel"].unique().tolist()) if "channel" in df_cat.columns else []
            subbands_arr = sorted(df_cat["subband"].unique().tolist()) if "subband" in df_cat.columns else []
            subjects_arr = sorted(df_cat["subject"].unique().tolist()) if "subject" in df_cat.columns else []
            
            has_scen = "scenario" in df_cat.columns
            scenarios_arr = sorted(df_cat["scenario"].dropna().unique().tolist()) if has_scen else ["Value"]
            
            if not subjects_arr or not channels_arr or not subbands_arr:
                continue
                
            # Cek apakai pakai unit micro (µ)
            feat_vals = pd.to_numeric(df_cat[sel_feat_table], errors="coerce").dropna()
            use_micro = False
            if not feat_vals.empty:
                if 0 < feat_vals.abs().mean() < 1e-3:
                    use_micro = True
            disp_feat_name = f"{sel_feat_table} (µ)" if use_micro else sel_feat_table
            
            # Mapping nilai
            val_map = {}
            for _, r in df_cat.iterrows():
                val = r.get(sel_feat_table)
                if pd.isna(val):
                    continue
                try:
                    val = float(val)
                    if use_micro:
                        val *= 1e6
                except ValueError:
                    pass
                ch = r.get("channel")
                sb = r.get("subband")
                subj = r.get("subject")
                scen = r.get("scenario") if has_scen else "Value"
                val_map[(ch, sb, subj, scen)] = val

            for c_idx, ch in enumerate(channels_arr):
                grid_row = c_idx // 2
                grid_col = c_idx % 2
                
                # Menentukan posisi baris dan kolom untuk channel ini
                start_r = 3 + grid_row * (len(subbands_arr) * (len(subjects_arr) + 3))
                start_c = 1 + grid_col * (len(scenarios_arr) + 3)
                
                ch_color = PatternFill("solid", fgColor=ch_colors.get(c_idx % 6, "FFFFFF"))
                sb_color = PatternFill("solid", fgColor="F4B084") # orange
                
                # Menghitung sampai mana cell channel akan digabung
                end_r_for_ch = start_r + len(subbands_arr) * (len(subjects_arr) + 3) - 2 
                
                c_cell = ws.cell(row=start_r, column=start_c, value=ch)
                c_cell.alignment = Alignment(horizontal="center", vertical="center", text_rotation=90)
                c_cell.fill = ch_color
                c_cell.font = Font(bold=True)
                if end_r_for_ch > start_r:
                    ws.merge_cells(start_row=start_r, start_column=start_c, end_row=end_r_for_ch, end_column=start_c)
                    
                curr_r = start_r
                for sb in subbands_arr:
                    # Header: Subband name
                    sb_cell = ws.cell(row=curr_r, column=start_c+1, value=sb)
                    sb_cell.fill = sb_color
                    sb_cell.font = Font(bold=True)
                    
                    # Header: Feature name (spanning scenarios)
                    f_cell = ws.cell(row=curr_r, column=start_c+2, value=disp_feat_name)
                    f_cell.font = Font(bold=True)
                    f_cell.alignment = Alignment(horizontal="center")
                    if len(scenarios_arr) > 1:
                        ws.merge_cells(start_row=curr_r, start_column=start_c+2, end_row=curr_r, end_column=start_c+1+len(scenarios_arr))
                        
                    curr_r += 1
                    # Header: "ID" and scenarios names
                    ws.cell(row=curr_r, column=start_c+1, value="ID").font = Font(bold=True)
                    for s_idx, scen in enumerate(scenarios_arr):
                        sc_cell = ws.cell(row=curr_r, column=start_c+2+s_idx, value=scen)
                        sc_cell.font = Font(bold=True)
                        sc_cell.alignment = Alignment(horizontal="center")
                        
                    # Data Row (Subjects and Values)
                    for subj in subjects_arr:
                        curr_r += 1
                        ws.cell(row=curr_r, column=start_c+1, value=subj)
                        for s_idx, scen in enumerate(scenarios_arr):
                            v = val_map.get((ch, sb, subj, scen), "")
                            cell = ws.cell(row=curr_r, column=start_c+2+s_idx, value=v)
                            if isinstance(v, float):
                                cell.number_format = '0.0000'
                                
                    curr_r += 2 # Menambah 1 row kosong stl data sbg jarak ke subband berikutnya
                    
    st.download_button(
        "Download Excel (Fitur per Task - Grid Format)", excel_buf.getvalue(),
        file_name=f"fitur_{sel_task_table}_{sel_feat_table}_grid.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_feat_task_xlsx_grid",
    )


# ---------------------------------------------------------------------------
# ERD/ERS Analysis
# ---------------------------------------------------------------------------

def _render_erd_ers(filtered_df, batch_tasks, feat_cols):
    """Render analisis ERD/ERS."""
    if "Resting" not in batch_tasks or len(batch_tasks) < 2:
        return

    st.markdown('<p class="section-title">ERD/ERS Analysis</p>', unsafe_allow_html=True)
    st.caption(
        "Event-Related Desynchronization (ERD) / Synchronization (ERS). "
        "Negatif = ERD (penurunan power), Positif = ERS (kenaikan power)."
    )

    show_erd = st.checkbox("Tampilkan ERD/ERS", value=False, key="show_erd_ers")
    if not show_erd:
        return

    sel_erd_feat = st.selectbox("Fitur untuk ERD/ERS", feat_cols, key="erd_feat_sel")

    erd_df = EEGFeatures.compute_erd_ers(
        filtered_df, baseline_task="Resting", feature_col=sel_erd_feat,
    )

    if erd_df.empty:
        st.warning("Tidak ada data ERD/ERS (pastikan task Resting ada).")
        return

    # Warna: negatif = merah (ERD), positif = hijau (ERS)
    def _color_erd(val):
        if isinstance(val, (int, float)):
            if val < 0:
                return "background-color: #FFCDD2"  # merah muda
            elif val > 0:
                return "background-color: #C8E6C9"  # hijau muda
        return ""

    styled_erd = erd_df.style.map(_color_erd, subset=["erd_ers_pct"])
    st.dataframe(styled_erd, use_container_width=True, height=400, hide_index=True)

    # Download
    fl1, fl2 = st.columns(2)
    with fl1:
        csv_erd = erd_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV (ERD/ERS)", csv_erd,
            file_name=f"erd_ers_{sel_erd_feat}.csv",
            mime="text/csv", key="dl_erd_csv",
        )
    with fl2:
        excel_erd = io.BytesIO()
        with pd.ExcelWriter(excel_erd, engine="openpyxl") as writer:
            out_erd = _format_micro_units(erd_df)
            out_erd.to_excel(writer, index=False, sheet_name="ERD_ERS")
        st.download_button(
            "Download Excel (ERD/ERS)", excel_erd.getvalue(),
            file_name=f"erd_ers_{sel_erd_feat}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_erd_xlsx",
        )
