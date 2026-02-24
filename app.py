"""
EEG Analysis Tool — Streamlit Entry Point

Aplikasi utama hanya routing antara:
  1. Analisis file tunggal  (ui.single_file)
  2. Analisis batch / ZIP   (ui.batch)

Semua logika pemrosesan, visualisasi, dan UI
berada di modul terpisah (processing/, visualization/, ui/).
"""

import streamlit as st

st.set_page_config(
    page_title="EEG Analysis Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.styles import inject_css
from ui.sidebar import render_sidebar, init_state
from ui.single_file import render_single_file
from ui.batch import run_batch_processing, render_batch_results


def main():
    inject_css()
    init_state()

    # ── Sidebar ──
    cfg = render_sidebar()

    # ── Header ──
    st.markdown(
        '<p class="dashboard-header">EEG Analysis Tool</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="dashboard-subtitle">'
        'Analisis sinyal EEG berbasis Streamlit — '
        'filtering, ICA, ekstraksi fitur, delta, dan statistik ALS vs Normal'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Routing ──
    if cfg["batch_mode"]:
        if cfg["batch_process"]:
            run_batch_processing(cfg)
        if st.session_state.batch_processed:
            render_batch_results(cfg)
    else:
        render_single_file(cfg)


if __name__ == "__main__":
    main()
