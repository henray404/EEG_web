"""
Modul sidebar -- Panel konfigurasi di sidebar Streamlit.
"""

import streamlit as st

from config import (
    DEFAULT_SUBBANDS, DEFAULT_FEATURES,
    ICA_METHODS, NOTCH_FREQUENCIES,
)
from processing.loader import EEGLoader


def init_state():
    """Inisialisasi session state dengan default values."""
    defaults = {
        "processor": None,
        "raw_info": None,
        "df_data": None,
        "features_df": None,
        "task_features_df": None,
        "task_summary_df": None,
        "processed": False,
        "batch_mode": False,
        "batch_df": None,
        "batch_tasks": [],
        "batch_processed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_data(uploaded, upload_type, edf_in_zip):
    """Load EDF lazily."""
    cache_key = f"{uploaded.name}_{edf_in_zip}"
    if st.session_state.get("_loaded_key") == cache_key:
        return

    loader = EEGLoader()
    try:
        if upload_type == "File EDF":
            info = loader.load_edf(uploaded)
        else:
            uploaded.seek(0)
            info = loader.load_edf_from_zip(uploaded, edf_in_zip)
        st.session_state.processor = loader
        st.session_state.raw_info = info
        st.session_state.processed = False
        st.session_state._loaded_key = cache_key
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")


def render_sidebar():
    """Render panel sidebar dan return konfigurasi.

    Returns
    -------
    dict  Konfigurasi yang dipilih user.
    """
    with st.sidebar:
        st.markdown("# EEG Analysis Tool")

        # --- File Upload ---
        st.markdown("### Unggah File")
        upload_type = st.radio(
            "Tipe", ["File EDF", "File ZIP (dataset)"],
            horizontal=True, label_visibility="collapsed",
        )

        uploaded = None
        edf_in_zip = None
        batch_mode = False

        if upload_type == "File EDF":
            uploaded = st.file_uploader(
                "Pilih file EDF", type=["edf"],
                label_visibility="collapsed",
            )
        else:
            uploaded = st.file_uploader(
                "Pilih file ZIP", type=["zip"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                uploaded.seek(0)
                try:
                    with st.spinner("Membaca struktur ZIP..."):
                        edf_list = EEGLoader.list_edf_in_zip(uploaded)

                    if edf_list:
                        if len(edf_list) > 1:
                            batch_mode = st.toggle(
                                "Batch Analisis (semua file)",
                                value=st.session_state.batch_mode,
                                key="batch_toggle",
                            )
                            st.session_state.batch_mode = batch_mode

                        if not batch_mode:
                            edf_in_zip = st.selectbox("File EDF dalam ZIP", edf_list)
                        else:
                            st.info(
                                f"{len(edf_list)} file EDF ditemukan. "
                                "Klik **Proses Batch** untuk memproses semua."
                            )
                    else:
                        st.warning("Tidak ada file EDF dalam ZIP.")
                        uploaded = None
                except Exception as e:
                    st.error(f"Gagal membaca ZIP: {e}")
                    uploaded = None

        if uploaded is not None and not batch_mode:
            _load_data(uploaded, upload_type, edf_in_zip)

        # Channel & Task: proses semua, filter setelah hasil keluar
        selected_channels = []
        selected_tasks = []
        if st.session_state.processor:
            tasks = st.session_state.processor.get_task_list()
            if tasks:
                selected_tasks = tasks

        # --- Filter Config ---
        st.markdown("### Filter")

        use_notch = st.toggle("Notch Filter", value=False)
        notch_freq = 50.0
        if use_notch:
            notch_freq = st.selectbox("Frekuensi Notch (Hz)", NOTCH_FREQUENCIES)

        use_amplitude = st.toggle(
            "Amplitude Filter", value=False,
            help="Clipping sinyal ke ±100 µV untuk menghilangkan artefak amplitudo tinggi",
        )

        bp_mode = st.radio(
            "Bandpass", ["Preset Subband", "Custom Range"],
            horizontal=True, label_visibility="collapsed",
        )

        bp_low, bp_high, bp_order = 0.5, 49.0, 5
        selected_subbands = {}

        if bp_mode == "Preset Subband":
            selected_subbands = dict(DEFAULT_SUBBANDS)
            bp_low = min(v[0] for v in selected_subbands.values())
            bp_high = max(v[1] for v in selected_subbands.values())
        else:
            col_a, col_b = st.columns(2)
            bp_low = col_a.number_input("Low (Hz)", 0.1, 100.0, 0.5, 0.1)
            bp_high = col_b.number_input("High (Hz)", 0.2, 100.0, 49.0, 0.1)
            bp_order = st.slider("Order", 1, 10, 5)

        # --- Bad Channel Detection ---
        st.markdown("### Deteksi Bad Channel")
        detect_bad = st.toggle(
            "Deteksi Bad Channel", value=False,
            help="Deteksi dan eksklusi channel buruk "
                 "berdasarkan variance (MAD threshold)",
        )

        # --- ICA ---
        st.markdown("### ICA")
        use_ica = st.toggle("Aktifkan ICA", value=False)
        ica_n, ica_method = None, "fastica"
        if use_ica:
            ica_auto = st.checkbox("Komponen otomatis", value=True)
            if not ica_auto:
                max_comp = max(1, len(selected_channels)) if selected_channels else 10
                ica_n = st.slider("Komponen", 1, max_comp, min(5, max_comp))
            ica_method = st.selectbox("Metode", ICA_METHODS)

        # Fitur: pakai semua default (MAV, variance, std)
        selected_features = list(DEFAULT_FEATURES)
        include_freq = False

        # --- Process ---
        st.divider()
        if batch_mode:
            batch_btn = st.button(
                "Proses Batch", type="primary",
                use_container_width=True,
                disabled=(uploaded is None),
            )
            process_btn = False
        else:
            batch_btn = False
            process_btn = st.button(
                "Proses Data", type="primary",
                use_container_width=True,
                disabled=(st.session_state.processor is None),
            )

        return {
            "process": process_btn,
            "batch_process": batch_btn,
            "batch_mode": batch_mode,
            "uploaded": uploaded,
            "channels": selected_channels,
            "tasks": selected_tasks,
            "use_notch": use_notch,
            "notch_freq": notch_freq,
            "bp_low": bp_low,
            "bp_high": bp_high,
            "bp_order": bp_order,
            "detect_bad": detect_bad,
            "use_amplitude": use_amplitude,
            "use_ica": use_ica,
            "ica_n": ica_n,
            "ica_method": ica_method,
            "features": selected_features,
            "include_frequency": include_freq,
            "subbands": selected_subbands if bp_mode == "Preset Subband" else None,
        }
