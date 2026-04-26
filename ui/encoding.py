"""
Modul UI encoding â€” Step Post-Batch: Chunking + Chain Encoding.

Dipanggil setelah ``run_batch_processing`` selesai. Memakai cache
DataFrame preprocessed dari ``st.session_state.batch_cached_files``
(lihat ``ui/batch.py::run_batch_processing``) sehingga tidak load
EDF ulang.

Output:
  - output_chain.csv   : chain sequence (1/0) per (task, channel, subband, feature)
  - output_summary.csv : cross-file summary (butuh >= 2 file)

Catatan: ``output_features.csv`` tidak disimpan â€” hanya dipakai
internal untuk menurunkan chain & summary.

Kontrak struktur dataset: hanya mendukung ZIP EEGET-ALS
(``idN/time1/scenarioN/EEG.edf``). Subject + scenario sudah di-parse
pada step batch via ``EEGLoader.detect_category``.
"""

import os
import time

import streamlit as st
import pandas as pd
import plotly.express as px

from config import (
    DEFAULT_ENCODING_WINDOW,
    EEGET_ALS_SCENARIOS,
    ACCENT_PRIMARY, ACCENT_LIGHT,
)
from processing.chunking import (
    process_cached_items as chunking_process_cached_items,
    DEFAULT_CHUNK_FEATURES,
)


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def render_post_batch_encoding():
    """Render section Chunking + Chain Encoding setelah batch selesai.

    Harus dipanggil dari ``ui/batch.py::render_batch_results`` setelah
    batch berhasil. Mengkonsumsi ``st.session_state.batch_cached_files``
    yang di-populate oleh ``run_batch_processing``.
    """
    cached_files = st.session_state.get("batch_cached_files") or []

    st.divider()
    st.markdown("## Chunking + Chain Encoding")
    st.caption(
        "Step ini memakai data batch yang sudah di-preprocess (tanpa "
        "load EDF ulang). Hanya mendukung struktur ZIP EEGET-ALS "
        "(`idN/time1/scenarioN/EEG.edf`). Output: **chain** + **summary**."
    )

    if not cached_files:
        st.info(
            "Cache batch kosong. Jalankan **Proses Batch** dulu pada ZIP "
            "berformat EEGET-ALS untuk mengaktifkan step ini."
        )
        return

    # Filter hanya file yang punya scenario_id valid (struktur EEGET-ALS)
    valid_items = [
        item for item in cached_files
        if int(item.get("scenario_id", -1)) >= 1
    ]
    n_valid = len(valid_items)
    n_total = len(cached_files)

    col_a, col_b = st.columns(2)
    col_a.metric("File Batch (total)", n_total)
    col_b.metric("Format EEGET-ALS", n_valid)

    if n_valid == 0:
        st.warning(
            "Tidak ada file dengan struktur `scenarioN/EEG.edf` yang "
            "terdeteksi. Post-batch encoding hanya mendukung ZIP "
            "EEGET-ALS untuk saat ini."
        )
        return

    # ============================================================== #
    #  Filter subject & scenario                                      #
    # ============================================================== #

    available_scenarios = sorted({
        int(item["scenario_id"]) for item in valid_items
    })
    scenario_options = {
        sc: f"{sc}. {EEGET_ALS_SCENARIOS.get(sc, '?')}"
        for sc in available_scenarios
    }
    selected_scenarios = st.multiselect(
        "Skenario",
        options=available_scenarios,
        default=available_scenarios,
        format_func=lambda x: scenario_options[x],
    )
    if not selected_scenarios:
        st.warning("Pilih minimal 1 skenario.")
        return

    # Filter item sementara berdasarkan skenario untuk mendapatkan metadata unik
    filtered_items = [
        item for item in valid_items
        if int(item["scenario_id"]) in selected_scenarios
    ]

    # Collect available channels, tasks, subbands from filtered_items
    available_channels = list(sorted(set(
        c for item in filtered_items for c in (item.get("channels") or [])
    )))
    available_tasks = list(sorted(set(
        t for item in filtered_items for t in (item.get("tasks") or [])
    )))
    
    from config import DEFAULT_SUBBANDS
    available_subbands = list(DEFAULT_SUBBANDS.keys())

    st.markdown("#### Filter Data Spasial & Temporal")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    selected_channels = col_opt1.multiselect(
        "Channel",
        options=available_channels,
        default=available_channels,
        help="Pilih elektroda yang akan di-chunk dan dienkode."
    )
    selected_tasks = col_opt2.multiselect(
        "Task (Stimulus/Rest)",
        options=available_tasks,
        default=available_tasks,
        help="Pilih fase rekaman yang akan diekstrak."
    )
    selected_subbands = col_opt3.multiselect(
        "Subband Frekuensi",
        options=available_subbands,
        default=available_subbands,
        help="Filter gelombang otak."
    )

    if not selected_channels or not selected_tasks or not selected_subbands:
        st.warning("Pilih setidaknya 1 Channel, 1 Task, dan 1 Subband.")
        return

    # --- Chunk settings ---
    col_win, col_info = st.columns(2)
    chunk_duration = col_win.slider(
        "Chunk Duration (detik)",
        min_value=0.1, max_value=2.0,
        value=DEFAULT_ENCODING_WINDOW, step=0.1,
        help="Durasi setiap chunk. 0.3s = 38 sampel pada 128 Hz.",
    )
    col_info.info("Chunking mode: non-overlapping (tanpa overlap).")

    # --- Feature options ---
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    use_task_segmentation = col_feat1.toggle(
        "Task Segmentation",
        value=True,
        help="Chunking dilakukan per segment task (Resting, Thinking, "
             "Typing, Think_Acting) dari annotations EDF.",
    )
    include_freq = col_feat2.toggle(
        "Fitur Frequency-Domain",
        value=True,
        help="Tambahkan band_power, relative_power, peak_frequency.",
    )
    scale_x10k = col_feat3.toggle(
        "Scale ×10.000",
        value=False,
        help="Kalikan semua nilai fitur dengan 10.000 sebelum chain encoding. "
             "Hasilnya ditambahkan sebagai kolom chain_sequence_x10k dan "
             "chain_ratio_x10k di CSV yang sama untuk perbandingan.",
    )

    all_features = list(DEFAULT_CHUNK_FEATURES)
    if not include_freq:
        all_features = [
            f for f in all_features if f in ("mav", "variance", "std")
        ]
    chain_features = st.multiselect(
        "Fitur yang di-chain",
        options=all_features,
        default=all_features,
        help="Pilih fitur yang dibandingkan antar chunk (encoding 1/0).",
    )
    if not chain_features:
        chain_features = all_features

    # ============================================================== #
    #  Filter cached items sesuai skenario terpilih                   #
    # ============================================================== #

    filtered_items = [
        item for item in valid_items
        if int(item["scenario_id"]) in selected_scenarios
    ]
    st.info(
        f"**Akan diproses:** {len(filtered_items)} file "
        f"({len(selected_scenarios)} skenario dari "
        f"{len({i['subject_id'] for i in filtered_items})} subjek)."
    )

    # ============================================================== #
    #  Tombol Mulai                                                   #
    # ============================================================== #

    st.divider()

    if st.button("Mulai Chunking + Chain Encoding", type="primary",
                 use_container_width=True):
        # Siapkan items dengan df = path pickle dan filter channel/task
        items_for_run = [
            {
                "df": item["cache_path"],
                "channels": [c for c in (item.get("channels") or []) if c in selected_channels],
                "sfreq": item["sfreq"],
                "tasks": [t for t in (item.get("tasks") or []) if t in selected_tasks],
                "subject_id": item.get("subject_id", ""),
                "scenario": item.get("scenario", "unknown"),
                "scenario_id": int(item.get("scenario_id", -1)),
                "filename": item.get("filename", "EEG.edf"),
            }
            for item in filtered_items
        ]

        from config import DEFAULT_SUBBANDS
        passed_subbands = {k: v for k, v in DEFAULT_SUBBANDS.items() if k in selected_subbands}

        _run_chunking(
            items=items_for_run,
            chunk_duration=chunk_duration,
            include_freq=include_freq,
            use_task_segmentation=use_task_segmentation,
            chain_features=chain_features,
            subbands=passed_subbands,
            scale_x10k=scale_x10k,
        )

    # ============================================================== #
    #  Tampilkan hasil jika sudah ada                                 #
    # ============================================================== #

    if st.session_state.get("chunking_done"):
        _render_chunking_results()

        # Perbandingan chain encoding antar subjek
        from ui.comparison import render_comparison_section
        render_comparison_section()


# ------------------------------------------------------------------ #
#  Run: Chunking + Chain Encoding                                     #
# ------------------------------------------------------------------ #

def _run_chunking(items, chunk_duration, include_freq,
                  use_task_segmentation, chain_features, subbands,
                  scale_x10k=False):
    """Jalankan chunking + chain encoding dari cached items."""

    # Clear stale state (termasuk key lama dari pipeline simple yang sudah dihapus)
    for key in (
        "chunking_done", "chunking_chain_df", "chunking_summary_df",
        "chunking_run_summary", "chunking_elapsed", "chunking_paths",
        # Legacy keys
        "chunking_features_df",
        "encoding_done", "encoding_result", "encoding_summary",
        "encoding_elapsed", "encoding_output_path", "encoding_raw",
        # Comparison state (clear saat re-encode)
        "comparison_done", "comparison_summary_df",
        "comparison_details_df", "comparison_elapsed", "comparison_meta",
    ):
        st.session_state.pop(key, None)

    progress_bar = st.progress(0, text="Memulai chunking...")
    status_text = st.empty()
    start_time = time.time()

    def progress_callback(subject_id, scenario_id, current, total):
        pct = current / total
        progress_bar.progress(
            pct,
            text=f"Chunking {subject_id}/scenario{scenario_id}... ({current}/{total})",
        )
        status_text.text(
            f"File {current}/{total} â€” {subject_id} skenario {scenario_id}"
        )

    if include_freq:
        features = list(DEFAULT_CHUNK_FEATURES)
    else:
        features = ["mav", "variance", "std"]

    features_df, chain_df, summary_df, run_summary = chunking_process_cached_items(
        cached_items=items,
        chunk_duration=chunk_duration,
        subbands=subbands,
        features=features,
        chain_features=chain_features,
        use_task_segmentation=use_task_segmentation,
        scale_x10k=scale_x10k,
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time

    if chain_df.empty:
        progress_bar.empty()
        status_text.empty()
        st.error(
            "Chunking gagal â€” tidak ada chain sequence yang berhasil dibuat. "
            "Pastikan tiap file punya â‰¥ 2 chunk per (task, channel, subband)."
        )
        return

    # features_df hanya dibuang â€” tidak disimpan ke disk maupun session_state.
    del features_df

    progress_bar.progress(1.0, text="Chunking selesai!")
    status_text.text(f"Selesai dalam {elapsed:.1f} detik")

    st.session_state.chunking_done = True
    st.session_state.chunking_chain_df = chain_df
    st.session_state.chunking_summary_df = summary_df
    st.session_state.chunking_run_summary = run_summary
    st.session_state.chunking_elapsed = elapsed


# ------------------------------------------------------------------ #
#  Render: Chunking results                                           #
# ------------------------------------------------------------------ #

def _render_chunking_results():
    """Tampilkan hasil chunking + chain encoding."""

    chain_df = st.session_state.chunking_chain_df
    summary_df = st.session_state.chunking_summary_df
    run = st.session_state.chunking_run_summary
    elapsed = st.session_state.chunking_elapsed

    st.divider()
    st.markdown("### Hasil Chunking + Chain Encoding")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total File", run["n_total_files"])
    col2.metric("Berhasil", run["n_success"])
    col3.metric("Gagal/Skip", run["n_failed"])
    col4.metric("Total Chunks", f"{run['n_total_chunks']:,}")

    col5, col6 = st.columns(2)
    col5.metric("Chain rows", f"{run['n_chain_rows']:,}")
    # Removed Summary rows metric

    st.info(f"**Waktu:** {elapsed:.1f} detik")

    # Use just a container or single expander instead of tabs
    st.markdown(f"#### Chain Data ({chain_df.shape[0]:,} rows)")

    import io

    def _style_dataframe(df_to_style):
        # Palet warna pastel/soft yang sangat lembut agar menyatu elegan dengan UI Streamlit
        colors = ["#F8FAFC", "#F1F5F9", "#F0FDF4", "#F0F9FF", "#EFF6FF", "#EEF2FF", "#F5F3FF", "#FAF5FF"]
        styler = df_to_style.style
        for i, col in enumerate(df_to_style.columns):
            styler = styler.set_properties(subset=[col], **{"background-color": colors[i % len(colors)], "color": "#1E293B"})
        return styler

    def _get_excel_bytes(df_to_export):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            colors = ["#F8FAFC", "#F1F5F9", "#F0FDF4", "#F0F9FF", "#EFF6FF", "#EEF2FF", "#F5F3FF", "#FAF5FF"]
            workbook = writer.book

            header_format = workbook.add_format({
                'bold': True,
                'border': 1,
                'bg_color': '#E2E8F0',  # Slate 200
                'font_color': '#0F172A'
            })

            # Check if 'feature' column exists
            if "feature" in df_to_export.columns:
                feature_groups = df_to_export.groupby("feature")
                for feat, feat_df in feature_groups:
                    # Truncate string for Excel 32767 chars limit
                    feat_df = feat_df.copy()
                    for col in feat_df.select_dtypes(include=["object"]):
                        feat_df[col] = feat_df[col].apply(lambda x: x[:32700] if isinstance(x, str) and len(x) > 32700 else x)

                    # Excel sheet names have max limit of 31 chars
                    sheet_name = str(feat).translate(str.maketrans("", "", "[]:*?/\\"))[:31]
                    if not sheet_name:
                        sheet_name = "Sheet"
                    feat_df.to_excel(writer, index=False, sheet_name=sheet_name)
                    worksheet = writer.sheets[sheet_name]

                    for i, col in enumerate(feat_df.columns):
                        col_format = workbook.add_format({
                            'bg_color': colors[i % len(colors)],
                            'font_color': '#1E293B',
                            'border': 1
                        })
                        col_width = max(len(str(col)) + 4, 15)
                        if not feat_df.empty:
                            max_data_len = feat_df[col].astype(str).map(len).max()
                            col_width = min(max(col_width, max_data_len + 2), 60)
                        worksheet.set_column(i, i, col_width, col_format)

                    for col_num, value in enumerate(feat_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
            else:
                # Truncate string for Excel 32767 chars limit
                df_safe = df_to_export.copy()
                for col in df_safe.select_dtypes(include=["object"]):
                    df_safe[col] = df_safe[col].apply(lambda x: x[:32700] if isinstance(x, str) and len(x) > 32700 else x)
                
                df_safe.to_excel(writer, index=False, sheet_name='Sheet1')
                worksheet = writer.sheets['Sheet1']

                for i, col in enumerate(df_safe.columns):
                    col_format = workbook.add_format({
                        'bg_color': colors[i % len(colors)],
                        'font_color': '#1E293B',
                        'border': 1
                    })
                    col_width = max(len(str(col)) + 4, 15)
                    if not df_safe.empty:
                        max_data_len = df_safe[col].astype(str).map(len).max()
                        col_width = min(max(col_width, max_data_len + 2), 60)
                    worksheet.set_column(i, i, col_width, col_format)

                for col_num, value in enumerate(df_safe.columns.values):
                    worksheet.write(0, col_num, value, header_format)

        return buffer.getvalue()

    if True:
        if chain_df.empty:
            st.info("Chain kosong â€” data tidak cukup untuk encoding chain.")
        else:
            st.dataframe(_style_dataframe(chain_df.head(200)), use_container_width=True, height=400)
            
            c_csv, c_xl = st.columns(2)
            c_csv.download_button(
                label="Download Full Data (CSV)",
                data=chain_df.to_csv(index=False).encode("utf-8"),
                file_name="output_chain.csv",
                mime="text/csv",
                use_container_width=True,
            )
            try:
                c_xl.download_button(
                    label="Download Result (Excel)",
                    data=_get_excel_bytes(chain_df),
                    file_name="output_chain.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Gagal Export Excel (Chain): {type(e).__name__} - {str(e)}")

    if False:
        if summary_df.empty:
            st.info(
                "Summary kosong â€” butuh â‰¥ 2 file dengan "
                "scenario/task/channel/subband/feature sama."
            )
        else:
            st.dataframe(_style_dataframe(summary_df.head(200)), use_container_width=True, height=400)
            
            c_csv, c_xl = st.columns(2)
            c_csv.download_button(
                label="Download Full Data (CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="output_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
            try:
                c_xl.download_button(
                    label="Download Result (Excel)",
                    data=_get_excel_bytes(summary_df),
                    file_name="output_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Gagal Export Excel (Summary): {type(e).__name__} - {str(e)}")

    # ------------------ Visualisasi pakai chain_df -------------------
    if "scenario_id" in chain_df.columns:
        st.markdown("#### Distribusi Chain Sequences per Skenario")
        scen_stats = (
            chain_df.groupby(["scenario_id"])
            .size()
            .reset_index(name="n_chain_rows")
        )
        scen_stats["scenario_name"] = scen_stats["scenario_id"].map(
            lambda x: f"{int(x)}. {EEGET_ALS_SCENARIOS.get(int(x), '?')}"
        )
        fig = px.bar(
            scen_stats,
            x="scenario_name", y="n_chain_rows",
            labels={"scenario_name": "Skenario",
                    "n_chain_rows": "Jumlah Chain Rows"},
            color_discrete_sequence=[ACCENT_PRIMARY],
        )
        fig.update_layout(
            xaxis_tickangle=-30,
            height=350,
            margin=dict(t=20, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    if "task" in chain_df.columns and chain_df["task"].any():
        tasks_present = [t for t in chain_df["task"].unique() if t]
        if tasks_present:
            st.markdown("#### Distribusi per Task")
            task_stats = (
                chain_df[chain_df["task"] != ""]
                .groupby("task")
                .size()
                .reset_index(name="n_chain_rows")
            )
            fig_t = px.bar(
                task_stats,
                x="task", y="n_chain_rows",
                labels={"task": "Task", "n_chain_rows": "Chain Rows"},
                color_discrete_sequence=[ACCENT_LIGHT],
            )
            fig_t.update_layout(height=300, margin=dict(t=20, b=40))
            st.plotly_chart(fig_t, use_container_width=True)
