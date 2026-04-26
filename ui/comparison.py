"""
Modul UI comparison — Perbandingan chain sequence antar subjek.

Halaman ini menampilkan perbandingan bit-string chain_sequence antar
semua pasangan subjek yang telah di-encode melalui Chunking + Chain
Encoding. User dapat memfilter scenario, task, channel, subband, dan
feature, lalu melihat segmen bit berturut-turut yang identik.

Posisi di UI: Tab "Encoding" di halaman batch results, di bawah
section Chunking + Chain Encoding (setelah encoding selesai).
"""

import io
import time

import streamlit as st
import pandas as pd

from config import (
    EEGET_ALS_SCENARIOS,
    ACCENT_PRIMARY, ACCENT_LIGHT, ACCENT_LIGHTER,
)
from processing.comparison import compare_all_pairs


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def render_comparison_section():
    """Render section perbandingan chain sequence antar subjek.

    Dipanggil dari ``ui/encoding.py`` setelah chunking selesai dan
    ``st.session_state.chunking_chain_df`` tersedia.
    """
    chain_df = st.session_state.get("chunking_chain_df")

    st.divider()
    st.markdown("## Perbandingan Chain Encoding antar Subjek")
    st.caption(
        "Bandingkan bit-string chain sequence antar semua pasangan subjek "
        "untuk menemukan segmen berturut-turut yang identik. "
        "Sequence akan di-truncate ke panjang terpendek agar adil."
    )

    if chain_df is None or chain_df.empty:
        st.info(
            "Data chain belum tersedia. Jalankan **Chunking + Chain Encoding** "
            "di atas terlebih dahulu."
        )
        return

    # Verifikasi kolom yang dibutuhkan
    required_cols = {"subject_id", "chain_sequence"}
    if not required_cols.issubset(set(chain_df.columns)):
        st.warning(
            "Data chain tidak memiliki kolom yang diperlukan "
            f"({', '.join(required_cols)}). Pastikan encoding berjalan dengan benar."
        )
        return

    # Cek minimal 2 subjek
    n_subjects = chain_df["subject_id"].nunique()
    if n_subjects < 2:
        st.warning(
            f"Hanya ditemukan {n_subjects} subjek. Perbandingan membutuhkan "
            "minimal 2 subjek."
        )
        return

    # ============================================================== #
    #  Filter panel                                                   #
    # ============================================================== #

    st.markdown("#### Filter Perbandingan")
    st.caption(
        "Semua filter default ke **Semua** — perbandingan dilakukan "
        "untuk setiap kombinasi unik (scenario, task, channel, subband, feature) "
        "yang terpilih."
    )

    # Scenario
    available_scenarios = []
    if "scenario_id" in chain_df.columns:
        available_scenarios = sorted(
            chain_df["scenario_id"].dropna().unique().tolist()
        )
    scenario_labels = {
        sc: f"{int(sc)}. {EEGET_ALS_SCENARIOS.get(int(sc), '?')}"
        for sc in available_scenarios
    }

    # Task
    available_tasks = []
    if "task" in chain_df.columns:
        available_tasks = sorted([
            t for t in chain_df["task"].dropna().unique().tolist()
            if t  # exclude empty strings
        ])

    # Channel
    available_channels = []
    if "channel" in chain_df.columns:
        available_channels = sorted(
            chain_df["channel"].dropna().unique().tolist()
        )

    # Subband
    available_subbands = []
    if "subband" in chain_df.columns:
        available_subbands = sorted(
            chain_df["subband"].dropna().unique().tolist()
        )

    # Feature
    available_features = []
    if "feature" in chain_df.columns:
        available_features = sorted(
            chain_df["feature"].dropna().unique().tolist()
        )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_scenarios = st.multiselect(
            "Skenario",
            options=available_scenarios,
            default=available_scenarios,
            format_func=lambda x: scenario_labels.get(x, str(x)),
            key="comp_scenario",
        ) if available_scenarios else available_scenarios

        selected_channels = st.multiselect(
            "Channel",
            options=available_channels,
            default=available_channels,
            key="comp_channel",
        ) if available_channels else available_channels

    with col_f2:
        selected_tasks = st.multiselect(
            "Task",
            options=available_tasks,
            default=available_tasks,
            key="comp_task",
        ) if available_tasks else available_tasks

        selected_subbands = st.multiselect(
            "Subband",
            options=available_subbands,
            default=available_subbands,
            key="comp_subband",
        ) if available_subbands else available_subbands

    col_feat, col_min = st.columns(2)
    with col_feat:
        selected_features = st.multiselect(
            "Feature",
            options=available_features,
            default=available_features,
            key="comp_feature",
        ) if available_features else available_features

    with col_min:
        min_length = st.slider(
            "Min. Panjang Segmen",
            min_value=1, max_value=10, value=2,
            key="comp_min_length",
            help="Minimum jumlah bit berturut-turut yang identik "
                 "untuk dicatat sebagai segmen matching. "
                 "Default 2 (exclude kecocokan 1-bit acak).",
        )

    # Preview: filter chain_df dan hitung subjek
    filtered_chain = chain_df.copy()
    if selected_scenarios and "scenario_id" in filtered_chain.columns:
        filtered_chain = filtered_chain[filtered_chain["scenario_id"].isin(selected_scenarios)]
    if selected_tasks and "task" in filtered_chain.columns:
        filtered_chain = filtered_chain[filtered_chain["task"].isin(selected_tasks)]
    if selected_channels and "channel" in filtered_chain.columns:
        filtered_chain = filtered_chain[filtered_chain["channel"].isin(selected_channels)]
    if selected_subbands and "subband" in filtered_chain.columns:
        filtered_chain = filtered_chain[filtered_chain["subband"].isin(selected_subbands)]
    if selected_features and "feature" in filtered_chain.columns:
        filtered_chain = filtered_chain[filtered_chain["feature"].isin(selected_features)]

    n_subj = filtered_chain["subject_id"].nunique()
    n_pairs = n_subj * (n_subj - 1) // 2

    # Hitung jumlah grup unik
    grp_cols = [c for c in ["scenario_id", "task", "channel", "subband", "feature"]
                if c in filtered_chain.columns and filtered_chain[c].nunique() > 0]
    n_groups = filtered_chain.groupby(grp_cols).ngroups if grp_cols else 1

    st.info(
        f"**{n_subj}** subjek × **{n_groups:,}** kombinasi "
        f"→ **{n_pairs * n_groups:,}** total perbandingan pasangan."
    )

    if not selected_scenarios and available_scenarios:
        st.warning("Pilih minimal 1 skenario.")
        return
    if not selected_tasks and available_tasks:
        st.warning("Pilih minimal 1 task.")
        return
    if not selected_channels and available_channels:
        st.warning("Pilih minimal 1 channel.")
        return
    if not selected_subbands and available_subbands:
        st.warning("Pilih minimal 1 subband.")
        return
    if not selected_features and available_features:
        st.warning("Pilih minimal 1 feature.")
        return

    # ============================================================== #
    #  Tombol Mulai                                                   #
    # ============================================================== #

    if st.button(
        "Mulai Perbandingan", type="primary",
        use_container_width=True, key="btn_start_comparison",
    ):
        _run_comparison(
            chain_df=filtered_chain,
            min_length=min_length,
        )


    # ============================================================== #
    #  Tampilkan hasil                                                #
    # ============================================================== #

    if st.session_state.get("comparison_done"):
        _render_comparison_results()


# ------------------------------------------------------------------ #
#  Run comparison                                                     #
# ------------------------------------------------------------------ #

def _run_comparison(chain_df, min_length):
    """Jalankan perbandingan dan simpan ke session state."""
    # Clear previous results
    for key in ("comparison_done", "comparison_summary_df",
                "comparison_details_df", "comparison_elapsed",
                "comparison_meta"):
        st.session_state.pop(key, None)

    progress_bar = st.progress(0, text="Memulai perbandingan...")
    status_text = st.empty()
    start_time = time.time()

    def progress_callback(current, total):
        pct = current / total if total > 0 else 1.0
        progress_bar.progress(
            pct,
            text=f"Memproses grup {current}/{total}...",
        )
        status_text.text(f"Grup {current}/{total}")

    # Pass None for all filters — chain_df sudah di-filter di UI
    summary_df, details_df = compare_all_pairs(
        chain_df=chain_df,
        min_length=min_length,
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time

    if summary_df.empty:
        progress_bar.empty()
        status_text.empty()
        st.warning(
            "Tidak ada pasangan yang ditemukan. Pastikan filter yang "
            "dipilih menghasilkan data dari minimal 2 subjek."
        )
        return

    progress_bar.progress(1.0, text="Perbandingan selesai!")
    status_text.text(f"Selesai dalam {elapsed:.1f} detik")

    st.session_state.comparison_done = True
    st.session_state.comparison_summary_df = summary_df
    st.session_state.comparison_details_df = details_df
    st.session_state.comparison_elapsed = elapsed
    st.session_state.comparison_meta = {
        "min_length": min_length,
        "n_subjects": chain_df["subject_id"].nunique(),
    }


# ------------------------------------------------------------------ #
#  Render results                                                     #
# ------------------------------------------------------------------ #

def _render_comparison_results():
    """Tampilkan hasil perbandingan."""
    summary_df = st.session_state.comparison_summary_df
    details_df = st.session_state.comparison_details_df
    elapsed = st.session_state.comparison_elapsed
    meta = st.session_state.comparison_meta

    st.divider()
    st.markdown("### Hasil Perbandingan")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pasangan", f"{len(summary_df):,}")
    col2.metric("Avg Match %", f"{summary_df['match_percentage'].mean():.1f}%"
                if not summary_df.empty else "–")
    col3.metric("Max Match %", f"{summary_df['match_percentage'].max():.1f}%"
                if not summary_df.empty else "–")
    col4.metric("Waktu", f"{elapsed:.1f}s")

    # Filter info
    filter_parts = []
    filter_parts.append(f"{meta.get('n_subjects', '?')} subjek")
    filter_parts.append(f"Min. segmen: {meta.get('min_length', 2)} bit")

    # Hitung unique values from summary
    for col, label in [("scenario_id", "Skenario"), ("task", "Task"),
                        ("channel", "Channel"), ("subband", "Subband"),
                        ("feature", "Feature")]:
        if col in summary_df.columns:
            n_unique = summary_df[col].nunique()
            filter_parts.append(f"{n_unique} {label}")

    st.caption(f"{'  |  '.join(filter_parts)}")

    # ---- Summary Table ----
    st.markdown("#### Tabel Summary (Semua Pasangan)")

    if not summary_df.empty:
        # Kolom yang ditampilkan — include group cols since results span all combinations
        display_cols = [
            c for c in [
                "scenario_id", "task", "channel", "subband", "feature",
                "subject_a", "subject_b", "compared_length",
                "n_matching_segments", "matching_segments_desc",
                "total_matching_bits", "match_percentage",
                "total_identical_bits", "identity_percentage",
            ] if c in summary_df.columns
        ]

        display_df = summary_df[display_cols].copy()

        # Sort by match_percentage descending
        if "match_percentage" in display_df.columns:
            display_df = display_df.sort_values(
                "match_percentage", ascending=False
            ).reset_index(drop=True)

        # Color styling
        def _color_match_pct(val):
            """Warna gradient untuk match_percentage."""
            if pd.isna(val):
                return ""
            if val >= 70:
                return "background-color: #A7F3D0; color: #065F46"  # green
            elif val >= 40:
                return "background-color: #FDE68A; color: #92400E"  # yellow
            else:
                return "background-color: #FECACA; color: #991B1B"  # red

        styled = display_df.style
        if "match_percentage" in display_df.columns:
            styled = styled.map(
                _color_match_pct, subset=["match_percentage"]
            )
        if "identity_percentage" in display_df.columns:
            styled = styled.map(
                _color_match_pct, subset=["identity_percentage"]
            )

        st.dataframe(
            styled,
            use_container_width=True,
            height=min(400, 40 + len(display_df) * 35),
        )

        # Download summary
        c_csv, c_xl = st.columns(2)
        with c_csv:
            csv_data = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Summary (CSV)",
                data=csv_data,
                file_name="comparison_summary.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_comp_summary_csv",
            )
        with c_xl:
            try:
                xl_data = _summary_to_excel(summary_df, details_df)
                st.download_button(
                    "Download Summary + Detail (Excel)",
                    data=xl_data,
                    file_name="comparison_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_comp_xlsx",
                )
            except Exception as e:
                st.error(f"Gagal export Excel: {e}")

    # ---- Detail per Pasangan ----
    st.markdown("#### Detail Segmen Matching per Pasangan")

    if summary_df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    # Sort summary by match_percentage desc for selection
    sorted_summary = summary_df.copy()
    if "match_percentage" in sorted_summary.columns:
        sorted_summary = sorted_summary.sort_values(
            "match_percentage", ascending=False
        ).reset_index(drop=True)

    # Build labels for selectbox — include group info
    pair_labels = []
    for idx, row in sorted_summary.iterrows():
        parts = []
        for col in ["scenario_id", "task", "channel", "subband", "feature"]:
            if col in row.index and pd.notna(row[col]):
                parts.append(str(row[col]))
        group_str = "/".join(parts) if parts else ""
        label = (
            f"[{group_str}] "
            f"{row['subject_a']} vs {row['subject_b']} "
            f"({row.get('n_matching_segments', 0)} segmen, "
            f"{row.get('match_percentage', 0):.1f}%)"
        )
        pair_labels.append(label)

    selected_label = st.selectbox(
        "Pilih Pasangan",
        options=pair_labels,
        key="comp_pair_select",
    )

    # Find selected row
    selected_idx = pair_labels.index(selected_label)
    sel_row = sorted_summary.iloc[selected_idx]
    sel_a = sel_row["subject_a"]
    sel_b = sel_row["subject_b"]

    # Build mask for details using all group columns
    if not details_df.empty:
        mask = (
            (details_df["subject_a"] == sel_a) &
            (details_df["subject_b"] == sel_b)
        )
        for col in ["scenario_id", "task", "channel", "subband", "feature"]:
            if col in details_df.columns and col in sel_row.index:
                mask = mask & (details_df[col] == sel_row[col])
        pair_details = details_df[mask].copy()
    else:
        pair_details = pd.DataFrame()

    # Show pair info
    ps = sel_row
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Panjang Dibandingkan", f"{ps.get('compared_length', 0)} bit")
    col_p2.metric("Segmen Matching", ps.get("n_matching_segments", 0))
    col_p3.metric("Match %", f"{ps.get('match_percentage', 0):.1f}%")

    # Show sequences with alignment
    seq_a = ps.get("seq_a", "")
    seq_b = ps.get("seq_b", "")
    if seq_a and seq_b:
        n = min(len(seq_a), len(seq_b))
        s1 = seq_a[:n]
        s2 = seq_b[:n]

        # Build alignment string
        match_line = ""
        for i in range(n):
            if s1[i] == s2[i]:
                match_line += "│"
            else:
                match_line += "╳"

        with st.expander("Lihat Alignment Sequence", expanded=False):
            # Tampilkan dalam chunks agar mudah dibaca
            chunk_size = 50
            alignment_parts = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                pos_label = f"[{start+1:>4}–{end:<4}]"
                alignment_parts.append(
                    f"{pos_label} {sel_a}: {s1[start:end]}\n"
                    f"{'':>13} {''.join(match_line[start:end])}\n"
                    f"{'':>13}{sel_b}: {s2[start:end]}"
                )
            st.code("\n\n".join(alignment_parts), language=None)

    # Detail table
    if not pair_details.empty:
        detail_display_cols = [
            c for c in ["segment_no", "bit_range", "length", "matching_bits"]
            if c in pair_details.columns
        ]
        detail_display = pair_details[detail_display_cols].copy()

        # Style
        _PASTEL_COLORS = [
            "#F0FDF4", "#EFF6FF", "#FEF3C7", "#FCE7F3",
            "#F5F3FF", "#ECFDF5", "#FFF7ED", "#F0F9FF",
        ]

        def _color_segments(df_to_style):
            styler = df_to_style.style
            for i, col in enumerate(df_to_style.columns):
                styler = styler.set_properties(
                    subset=[col],
                    **{
                        "background-color": _PASTEL_COLORS[i % len(_PASTEL_COLORS)],
                        "color": "#1E293B",
                    },
                )
            return styler

        st.dataframe(
            _color_segments(detail_display),
            use_container_width=True,
            height=min(400, 40 + len(detail_display) * 35),
        )
    else:
        st.info("Tidak ada segmen matching yang memenuhi threshold untuk pasangan ini.")

    # Download detail for selected pair
    if not pair_details.empty:
        csv_detail = pair_details.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download Detail {sel_a} vs {sel_b} (CSV)",
            data=csv_detail,
            file_name=f"comparison_detail_{sel_a}_vs_{sel_b}.csv",
            mime="text/csv",
            key="dl_comp_detail_csv",
        )


# ------------------------------------------------------------------ #
#  Excel export helpers                                               #
# ------------------------------------------------------------------ #

def _summary_to_excel(summary_df, details_df):
    """Export summary + detail ke Excel multi-sheet."""
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Header format
        header_fmt = workbook.add_format({
            "bold": True,
            "border": 1,
            "bg_color": "#E2E8F0",
            "font_color": "#0F172A",
        })

        # --- Sheet 1: Summary ---
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        ws_summary = writer.sheets["Summary"]

        colors = [
            "#F8FAFC", "#F1F5F9", "#F0FDF4", "#F0F9FF",
            "#EFF6FF", "#EEF2FF", "#F5F3FF", "#FAF5FF",
        ]
        for i, col in enumerate(summary_df.columns):
            col_fmt = workbook.add_format({
                "bg_color": colors[i % len(colors)],
                "font_color": "#1E293B",
                "border": 1,
            })
            col_width = max(len(str(col)) + 4, 15)
            if not summary_df.empty:
                max_data = summary_df[col].astype(str).map(len).max()
                col_width = min(max(col_width, max_data + 2), 60)
            ws_summary.set_column(i, i, col_width, col_fmt)

        for col_num, value in enumerate(summary_df.columns.values):
            ws_summary.write(0, col_num, value, header_fmt)

        # --- Sheet 2: All Details ---
        if not details_df.empty:
            # Truncate detail for Excel limit
            details_safe = details_df.copy()
            for col in details_safe.select_dtypes(include=["object"]):
                details_safe[col] = details_safe[col].apply(
                    lambda x: x[:32700] if isinstance(x, str) and len(x) > 32700 else x
                )
            details_safe.to_excel(
                writer, index=False, sheet_name="Detail Segments"
            )
            ws_detail = writer.sheets["Detail Segments"]

            for i, col in enumerate(details_safe.columns):
                col_fmt = workbook.add_format({
                    "bg_color": colors[i % len(colors)],
                    "font_color": "#1E293B",
                    "border": 1,
                })
                col_width = max(len(str(col)) + 4, 15)
                if not details_safe.empty:
                    max_data = details_safe[col].astype(str).map(len).max()
                    col_width = min(max(col_width, max_data + 2), 60)
                ws_detail.set_column(i, i, col_width, col_fmt)

            for col_num, value in enumerate(details_safe.columns.values):
                ws_detail.write(0, col_num, value, header_fmt)

    return buffer.getvalue()
