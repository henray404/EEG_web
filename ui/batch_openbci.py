"""
Modul batch_openbci — Batch processing khusus data OpenBCI TXT.

Mendukung folder structure: Baseline/ Familiar/ Unfamiliar/
Setiap file .txt = 1 subjek × 1 kondisi.
"""

import io
import os

import streamlit as st
import pandas as pd
import numpy as np

from config import DEFAULT_SUBBANDS, DEFAULT_FEATURES, ACCENT_LIGHT
from processing.loader import EEGLoader
from processing.filters import EEGFilters
from processing.features import EEGFeatures
from processing.epoching import EpochEngine
from processing.connectivity import ConnectivityAnalyzer


# ---------------------------------------------------------------------------
# Worker: proses satu file TXT OpenBCI
# ---------------------------------------------------------------------------

def _process_single_txt(zip_bytes, txt_path, subbands, features,
                        use_notch=False, notch_freq=50.0,
                        use_car=False,
                        use_amplitude=False,
                        bp_low=0.5, bp_high=50.0, bp_order=5,
                        use_ica=False, ica_n=None, ica_method="fastica",
                        include_frequency=True,
                        psd_method="welch", psd_fmin=0.0,
                        psd_fmax=49.0, psd_n_fft=None,
                        use_epoching=False, epoch_mode="fixed",
                        epoch_duration=2.0, window_overlap=0.5,
                        use_epoch_reject=False, epoch_reject_threshold=100.0,
                        use_connectivity=False, connectivity_method="wpli",
                        connectivity_channels=None):
    """Worker: proses satu file TXT OpenBCI dari ZIP bytes (thread-safe).

    Returns
    -------
    tuple  (feat_df, conn_df, meta)
           feat_df : fitur per channel/subband (atau None)
           conn_df : DataFrame connectivity (atau None)
           meta    : dict {"subject", "condition"}
    """
    meta = EEGLoader.detect_openbci_metadata(txt_path)
    loader = EEGLoader()

    try:
        buf = io.BytesIO(zip_bytes)
        loader.load_txt_from_zip(buf, txt_path)
    except Exception as e:
        meta["error"] = f"Load gagal: {e}"
        return None, None, meta

    ch_list = loader.channel_names
    if not ch_list:
        meta["error"] = "Filter channel kosong."
        return None, None, meta

    # --- Preprocessing ---
    # Amplitude filter
    if use_amplitude:
        try:
            EEGFilters.apply_amplitude_filter(loader)
        except Exception:
            pass

    # Notch filter
    if use_notch:
        try:
            EEGFilters.apply_notch(loader, freq=notch_freq)
        except Exception:
            pass

    # Bandpass filter
    try:
        EEGFilters.apply_bandpass(loader, bp_low, bp_high, order=bp_order)
    except Exception:
        pass

    # Common Average Reference
    if use_car:
        try:
            EEGFilters.apply_car(loader)
        except Exception:
            pass

    # ICA
    if use_ica:
        try:
            EEGFilters.apply_ica(loader, n_components=ica_n, method=ica_method)
        except Exception:
            pass

    # --- Extract DataFrame ---
    df = loader.extract_dataframe()

    # Data TXT tidak punya task/marker → seluruh data = 1 segment
    # Kita set marker "all" untuk seluruh data agar fitur bisa dihitung
    sfreq = loader.sfreq

    # --- Fitur statistik ---
    try:
        feat_df = EEGFeatures.compute_subband_features(
            df, ch_list, sfreq=sfreq, subbands=subbands, features=features,
            include_frequency=include_frequency,
            psd_method=psd_method,
            psd_fmin=psd_fmin,
            psd_fmax=psd_fmax,
            psd_n_fft=psd_n_fft,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        meta["error"] = f"Fitur gagal: {e}"
        return None, None, meta

    # --- Connectivity ---
    conn_df = None
    if use_connectivity:
        ch_list_conn = ch_list
        if connectivity_channels:
            ch_list_conn = [c for c in ch_list if c in connectivity_channels]
            if not ch_list_conn:
                ch_list_conn = ch_list

        raw_data = df[ch_list_conn].values.T  # (n_channels, n_times)

        # Auto-chunk 2 detik untuk PLI/wPLI
        chunk_duration = epoch_duration if use_epoching else 2.0
        samples_per_epoch = int(chunk_duration * sfreq)
        n_samples = raw_data.shape[1]
        n_epochs = n_samples // samples_per_epoch

        if n_epochs < 2:
            if n_samples >= int(sfreq * 1.0):
                samples_per_epoch = n_samples // 2
                n_epochs = 2
                trimmed = raw_data[:, :n_epochs * samples_per_epoch]
                data_3d = trimmed.reshape(
                    raw_data.shape[0], n_epochs, samples_per_epoch
                ).transpose(1, 0, 2)
            else:
                data_3d = raw_data[np.newaxis, :, :]
        else:
            trimmed = raw_data[:, :n_epochs * samples_per_epoch]
            data_3d = trimmed.reshape(
                raw_data.shape[0], n_epochs, samples_per_epoch
            ).transpose(1, 0, 2)

        conn_dict = ConnectivityAnalyzer.compute_connectivity(
            data_3d, sfreq, ch_list_conn,
            method=connectivity_method, subbands=subbands,
        )
        if conn_dict:
            conn_df = ConnectivityAnalyzer.connectivity_to_dataframe(
                conn_dict, ch_list_conn, method=connectivity_method,
            )

    loader._cleanup_tmp()

    # Tambahkan metadata ke hasil
    if feat_df is not None and not feat_df.empty:
        feat_df.insert(0, "subject", meta["subject"])
        feat_df.insert(1, "condition", meta["condition"])
        feat_df.insert(2, "filename", os.path.basename(txt_path))

    if conn_df is not None and not conn_df.empty:
        conn_df.insert(0, "subject", meta["subject"])
        conn_df.insert(1, "condition", meta["condition"])
        conn_df.insert(2, "filename", os.path.basename(txt_path))

    return feat_df, conn_df, meta


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

def run_openbci_batch(cfg):
    """Jalankan batch processing untuk dataset OpenBCI TXT dari ZIP."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    uploaded = cfg.get("uploaded")
    if uploaded is None:
        st.warning("File ZIP belum diunggah.")
        return

    subbands = cfg.get("subbands") or DEFAULT_SUBBANDS
    features = cfg.get("features") or DEFAULT_FEATURES

    progress_bar = st.progress(0, text="Memulai batch processing OpenBCI...")

    uploaded.seek(0)
    zip_bytes = uploaded.read()
    uploaded.seek(0)

    # Scan ZIP
    subjects_map, skipped = EEGLoader.scan_openbci_zip(uploaded)

    if not subjects_map:
        st.error("Tidak ditemukan file TXT yang sesuai format dalam ZIP.")
        return

    # Kumpulkan semua file yang harus diproses
    all_tasks = []
    for subj_id, conditions in subjects_map.items():
        for condition, txt_path in conditions.items():
            all_tasks.append((subj_id, condition, txt_path))

    n_total = len(all_tasks)
    all_feat_dfs = []
    all_conn_dfs = []

    cpu_count = max(1, os.cpu_count() or 1)
    max_workers = min(cpu_count, n_total)

    failed_files = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for subj_id, condition, txt_path in all_tasks:
            future = executor.submit(
                _process_single_txt,
                zip_bytes, txt_path,
                subbands, features,
                cfg.get("use_notch", False),
                cfg.get("notch_freq", 50.0),
                cfg.get("use_car", False),
                cfg.get("use_amplitude", False),
                cfg.get("bp_low", 0.5),
                cfg.get("bp_high", 50.0),
                cfg.get("bp_order", 5),
                cfg.get("use_ica", False),
                cfg.get("ica_n", None),
                cfg.get("ica_method", "fastica"),
                cfg.get("include_frequency", True),
                cfg.get("psd_method", "welch"),
                cfg.get("psd_fmin", 0.0),
                cfg.get("psd_fmax", 49.0),
                cfg.get("psd_n_fft", None),
                cfg.get("use_epoching", False),
                cfg.get("epoch_mode", "fixed"),
                cfg.get("epoch_duration", 2.0),
                cfg.get("window_overlap", 0.5),
                cfg.get("use_epoch_reject", False),
                cfg.get("epoch_reject_threshold", 100.0),
                cfg.get("use_connectivity", False),
                cfg.get("connectivity_method", "wpli"),
                cfg.get("connectivity_channels", []),
            )
            futures[future] = (subj_id, condition, txt_path)

        for i, future in enumerate(as_completed(futures), 1):
            subj_id, condition, txt_path = futures[future]
            progress_bar.progress(
                i / n_total,
                text=f"Memproses {i}/{n_total}: {subj_id} ({condition})",
            )
            try:
                feat_df, conn_df, meta = future.result()
                if feat_df is not None and not feat_df.empty:
                    all_feat_dfs.append(feat_df)
                else:
                    err_msg = meta.get("error", "Unknown Error") if meta else "No returns"
                    failed_files[txt_path] = err_msg
                
                if conn_df is not None and not conn_df.empty:
                    all_conn_dfs.append(conn_df)
            except Exception as e:
                import traceback
                print(f"Error in batch worker for {subj_id} ({condition}): {e}")
                traceback.print_exc()
                failed_files[txt_path] = f"Fatal crash: {e}"

    progress_bar.empty()

    if failed_files:
        with st.expander("Beberapa file gagal diekstrak (Klik untuk detail)"):
            for p, err in failed_files.items():
                st.write(f"- `{p}`: {err}")

    if not all_feat_dfs:
        st.error("Tidak ada data fitur yang berhasil diekstrak.")
        return

    batch_feat_df = pd.concat(all_feat_dfs, ignore_index=True)
    batch_conn_df = pd.DataFrame()
    if all_conn_dfs:
        batch_conn_df = pd.concat(all_conn_dfs, ignore_index=True)

    # Simpan ke session state
    st.session_state.openbci_feat_df = batch_feat_df
    st.session_state.openbci_conn_df = batch_conn_df
    st.session_state.openbci_subjects_map = subjects_map
    st.session_state.openbci_batch_processed = True

    n_subj = batch_feat_df["subject"].nunique()
    n_cond = batch_feat_df["condition"].nunique()
    st.markdown(f'''
    <div style="padding: 16px; border-radius: 8px; background-color: #d1e7dd; color: #0f5132; display: flex; align-items: flex-start; gap: 12px; margin: 16px 0; border: 1px solid #badbcc;">
        <svg style="width: 24px; height: 24px; flex-shrink: 0;" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
        </svg>
        <span style="font-size: 14px; line-height: 1.5;">Batch processing selesai: <strong>{n_subj} subjek</strong>, <strong>{n_cond} kondisi</strong>, <strong>{len(batch_feat_df):,} baris</strong> data fitur.</span>
    </div>
    ''', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render hasil batch OpenBCI
# ---------------------------------------------------------------------------

def render_openbci_batch_results(cfg):
    """Tampilkan hasil batch analysis OpenBCI."""
    feat_df = st.session_state.get("openbci_feat_df")
    conn_df = st.session_state.get("openbci_conn_df")

    if feat_df is None or feat_df.empty:
        return

    # --- Ringkasan ---
    st.markdown(
        '<p class="section-title">Ringkasan Batch OpenBCI</p>',
        unsafe_allow_html=True,
    )

    conditions = sorted(feat_df["condition"].unique().tolist())
    subjects = sorted(feat_df["subject"].unique().tolist())
    all_channels = sorted(feat_df["channel"].unique().tolist()) if "channel" in feat_df.columns else []
    all_subbands = sorted(feat_df["subband"].unique().tolist()) if "subband" in feat_df.columns else []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjek", len(subjects))
    c2.metric("Kondisi", len(conditions))
    c3.metric("Channel", len(all_channels))
    c4.metric("Total Baris Fitur", f"{len(feat_df):,}")

    # Badges kondisi
    cond_colors = {
        "baseline": "#1E88E5",
        "familiar": "#10B981",
        "unfamiliar": "#F59E0B",
    }
    badge_html = ""
    for cond in conditions:
        color = cond_colors.get(cond, ACCENT_LIGHT)
        badge_html += f'<span class="task-badge" style="background:{color}">{cond.capitalize()}</span>'
    st.markdown(f'<div style="margin:8px 0">{badge_html}</div>', unsafe_allow_html=True)

    # --- Filter panel ---
    with st.expander("Filter & Konfigurasi", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_conditions = st.multiselect(
                "Kondisi", conditions, default=conditions, key="openbci_cond_filter"
            )
        with fc2:
            sel_subbands = st.multiselect(
                "Subband", all_subbands, default=[], key="openbci_sb_filter"
            )
        with fc3:
            sel_channels = st.multiselect(
                "Channel", all_channels, default=[], key="openbci_ch_filter"
            )

    # Apply filters
    filtered_df = feat_df.copy()
    if sel_conditions:
        filtered_df = filtered_df[filtered_df["condition"].isin(sel_conditions)]
    if sel_subbands:
        filtered_df = filtered_df[filtered_df["subband"].isin(sel_subbands)]
    if sel_channels:
        filtered_df = filtered_df[filtered_df["channel"].isin(sel_channels)]

    # --- Helper Scaling Units ---
    def _format_micro_units(df):
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

    # --- Tabel Fitur per Subjek & Kondisi ---
    st.markdown(
        '<p class="section-title">Tabel Fitur per Subjek & Kondisi</p>',
        unsafe_allow_html=True,
    )

    meta_cols = {"subject", "condition", "filename", "channel", "subband", "task"}
    feat_cols = [c for c in filtered_df.columns if c not in meta_cols]

    show_table = st.checkbox("Tampilkan Tabel Fitur", value=True, key="openbci_show_feat")
    if show_table and not filtered_df.empty:
        # Scale units for readability
        formatted_df = _format_micro_units(filtered_df)
            
        # Color styling by column prefix
        def _color_flat_cols(col):
            if col.name in meta_cols:
                return ["background-color: #e3f2fd"] * len(col)  # biru muda
            elif col.name.startswith("mav") or "mav (µ)" in col.name:
                return ["background-color: #e8f5e9"] * len(col)  # hijau muda
            elif col.name.startswith("variance") or "variance (µ)" in col.name:
                return ["background-color: #fff3e0"] * len(col)  # oranye muda
            elif col.name.startswith("std") or "std (µ)" in col.name:
                return ["background-color: #fce4ec"] * len(col)  # pink muda
            else:
                return ["background-color: #DBCDF0C9"] * len(col)  # abu-abu Default
        
        styled_df = formatted_df.style.apply(_color_flat_cols)
        st.dataframe(styled_df, use_container_width=True, height=450, hide_index=True)
        feat_df_dl = formatted_df
    else:
        feat_df_dl = _format_micro_units(filtered_df)

    # --- Download ---
    st.markdown("### Download Hasil")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.markdown('''
        <div style="display:flex; align-items:center; margin-bottom: 8px;">
            <svg style="width: 20px; height: 20px; color: #1976d2; margin-right: 8px;" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
            <span style="font-weight: 600; font-size: 14px; color: #424242;">Download CSV Tabel Ini</span>
        </div>
        ''', unsafe_allow_html=True)
        csv_data = feat_df_dl.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Fitur CSV", csv_data,
            file_name="openbci_features.csv",
            mime="text/csv", key="dl_openbci_feat_csv",
            use_container_width=True
        )
    with dl2:
        st.markdown('''
        <div style="display:flex; align-items:center; margin-bottom: 8px;">
            <svg style="width: 20px; height: 20px; color: #2e7d32; margin-right: 8px;" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
            <span style="font-weight: 600; font-size: 14px; color: #424242;">Download Referensi Lengkap (Excel)</span>
        </div>
        ''', unsafe_allow_html=True)
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            feat_df_dl.to_excel(writer, index=False, sheet_name="Fitur")
            if conn_df is not None and not conn_df.empty:
                conn_df.to_excel(writer, index=False, sheet_name="Connectivity")
        st.download_button(
            "Download Excel (Fitur + Konekt.)",
            excel_buf.getvalue(),
            file_name="openbci_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_openbci_xlsx",
            use_container_width=True
        )

    # --- Perbandingan antar Kondisi ---
    if len(conditions) >= 2 and feat_cols:
        _render_condition_comparison(filtered_df, conditions, feat_cols)

    # --- Connectivity ---
    if conn_df is not None and not conn_df.empty:
        _render_connectivity_openbci(conn_df, sel_conditions)


def _render_condition_comparison(filtered_df, conditions, feat_cols):
    """Render perbandingan fitur antar kondisi dengan Scatter Plot."""
    import plotly.express as px

    st.markdown(
        '<p class="section-title">Sebaran Data per Subjek & Kondisi (Scatter Plot)</p>',
        unsafe_allow_html=True,
    )

    sel_feat = st.selectbox("Pilih Fitur untuk Scatter", feat_cols, key="openbci_compare_feat")

    if sel_feat not in filtered_df.columns:
        return

    # Visualisasi Scatter plot
    fig = px.scatter(
        filtered_df,
        x="subject",
        y=sel_feat,
        color="condition",
        symbol="subband" if "subband" in filtered_df.columns else None,
        hover_data=["channel"],
        title=f"Distribusi {sel_feat} per Subjek",
        color_discrete_map={
            "baseline": "#1E88E5",
            "familiar": "#10B981",
            "unfamiliar": "#F59E0B"
        }
    )
    
    fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title="Subjek",
        yaxis_title=sel_feat,
        template="plotly_white",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="Kondisi"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rata-rata per kondisi × subband
    group_cols = ["condition"]
    if "subband" in filtered_df.columns:
        group_cols.append("subband")

    agg_df = (
        filtered_df.groupby(group_cols)[sel_feat]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    if agg_df.empty:
        st.warning("Tidak ada data untuk tabel agregat.")
        return

    # Tabel agregat
    show_agg = st.checkbox("Tampilkan Tabel Agregat", value=False, key="openbci_show_agg")
    if show_agg:
        meta_cols = {"condition", "subband", "channel"}
        
        def _color_agg_cols(col):
            if col.name in meta_cols:
                return ["background-color: #e3f2fd"] * len(col)
            elif col.name == "mean":
                return ["background-color: #e8f5e9"] * len(col)
            elif col.name == "std":
                return ["background-color: #fff3e0"] * len(col)
            elif col.name == "count":
                return ["background-color: #fce4ec"] * len(col)
            else:
                return [""] * len(col)
                
        styled_agg = agg_df.style.apply(_color_agg_cols)
        st.dataframe(styled_agg, use_container_width=True, hide_index=True)


def _render_connectivity_openbci(conn_df, sel_conditions):
    """Render connectivity results for OpenBCI batch."""
    st.markdown(
        '<p class="section-title">Konektivitas (PLI/wPLI)</p>',
        unsafe_allow_html=True,
    )

    # Filter by conditions
    filtered_conn = conn_df.copy()
    if sel_conditions:
        filtered_conn = filtered_conn[filtered_conn["condition"].isin(sel_conditions)]

    if filtered_conn.empty:
        st.info("Tidak ada data konektivitas.")
        return

    show_conn = st.checkbox("Tampilkan Tabel Konektivitas", value=False, key="openbci_show_conn")
    if show_conn:
        meta_cols = {"subject", "condition", "filename", "task", "time", "method", "channel_1", "channel_2", "subband"}
        
        def _color_conn_cols(col):
            if col.name in meta_cols:
                return ["background-color: #e3f2fd"] * len(col)
            else:
                return ["background-color: #fff3e0"] * len(col) # Oranye muda untuk nilai
                
        styled_conn = filtered_conn.style.apply(_color_conn_cols)
        st.dataframe(styled_conn, use_container_width=True, height=400, hide_index=True)

    # Download konektivitas
    st.markdown('''
    <div style="display:flex; align-items:center; margin-top: 16px; margin-bottom: 8px;">
        <svg style="width: 20px; height: 20px; color: #1976d2; margin-right: 8px;" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
        <span style="font-weight: 600; font-size: 14px; color: #424242;">Download Connectivity Data</span>
    </div>
    ''', unsafe_allow_html=True)
    csv_conn = filtered_conn.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Konektivitas CSV", csv_conn,
        file_name="openbci_connectivity.csv",
        mime="text/csv", key="dl_openbci_conn_csv",
    )
