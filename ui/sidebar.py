"""
Modul sidebar -- Panel konfigurasi di sidebar Streamlit.
"""

import streamlit as st

from config import (
    DEFAULT_SUBBANDS, DEFAULT_FEATURES,
    ICA_METHODS, NOTCH_FREQUENCIES,
    PSD_METHODS, DEFAULT_PSD_METHOD,
    DEFAULT_PSD_FMIN, DEFAULT_PSD_FMAX,
    DEFAULT_EPOCH_DURATION, MIN_EPOCH_DURATION, MAX_EPOCH_DURATION,
    DEFAULT_WINDOW_SIZE, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE,
    DEFAULT_WINDOW_OVERLAP, DEFAULT_EPOCH_REJECT_UV,
    CONNECTIVITY_METHODS, DEFAULT_CONNECTIVITY_METHOD,
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
            "Tipe", ["File EDF", "File TXT (OpenBCI)", "File ZIP (dataset)"],
            horizontal=True, label_visibility="collapsed",
        )

        uploaded = None
        edf_in_zip = None
        batch_mode = False
        is_openbci = upload_type == "File TXT (OpenBCI)"
        is_openbci_batch = False

        if upload_type == "File EDF":
            uploaded = st.file_uploader(
                "Pilih file EDF", type=["edf"],
                label_visibility="collapsed",
            )
        elif upload_type == "File TXT (OpenBCI)":
            uploaded = st.file_uploader(
                "Pilih file TXT OpenBCI", type=["txt"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                # Load TXT OpenBCI
                cache_key = f"openbci_{uploaded.name}"
                if st.session_state.get("_loaded_key") != cache_key:
                    loader = EEGLoader()
                    try:
                        uploaded.seek(0)
                        info = loader.load_openbci_txt(uploaded)
                        st.session_state.processor = loader
                        st.session_state.raw_info = info
                        st.session_state.processed = False
                        st.session_state._loaded_key = cache_key
                    except Exception as e:
                        st.error(f"Gagal memuat TXT: {e}")
                        uploaded = None
        else:
            uploaded = st.file_uploader(
                "Pilih file ZIP", type=["zip"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                uploaded.seek(0)
                try:
                    with st.spinner("Membaca struktur ZIP..."):
                        # Cek apakah ZIP berisi TXT (OpenBCI) atau EDF
                        edf_list = EEGLoader.list_edf_in_zip(uploaded)
                        uploaded.seek(0)
                        txt_list = EEGLoader.list_txt_in_zip(uploaded)

                    if txt_list and not edf_list:
                        # ZIP berisi TXT → mode OpenBCI batch
                        is_openbci = True
                        is_openbci_batch = True
                        batch_mode = True
                        st.session_state.batch_mode = True
                        uploaded.seek(0)
                        subjects_map, skipped = EEGLoader.scan_openbci_zip(uploaded)
                        n_subj = len(subjects_map)
                        n_files = sum(len(v) for v in subjects_map.values())
                        st.info(
                            f"📁 OpenBCI dataset: **{n_subj} subjek**, "
                            f"**{n_files} file TXT** ditemukan."
                        )
                        if skipped:
                            st.warning(f"{len(skipped)} file di-skip (format tidak dikenali).")
                    elif edf_list:
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
                        st.warning("Tidak ada file EDF atau TXT dalam ZIP.")
                        uploaded = None
                except Exception as e:
                    st.error(f"Gagal membaca ZIP: {e}")
                    uploaded = None

        if uploaded is not None and not batch_mode and not is_openbci:
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

        use_car = st.toggle(
            "Common Average Reference (CAR)", value=False,
            help="Re-referencing ke rata-rata semua channel. "
                 "Disarankan untuk data OpenBCI.",
        )

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
        include_freq = True

        # --- PSD Settings ---
        st.markdown("### PSD (Power Spectral Density)")

        psd_method = st.selectbox(
            "Metode PSD", PSD_METHODS,
            index=PSD_METHODS.index(DEFAULT_PSD_METHOD),
            help="Welch: segmented FFT averaging. "
                 "Multitaper: multiple taper averaging (lebih halus).",
        )

        psd_col1, psd_col2 = st.columns(2)
        psd_fmin = psd_col1.number_input(
            "fmin (Hz)", 0.0, 20.0, DEFAULT_PSD_FMIN, 0.5,
            help="Frekuensi minimum untuk analisis PSD.",
        )
        psd_fmax = psd_col2.number_input(
            "fmax (Hz)", 10.0, 100.0, DEFAULT_PSD_FMAX, 1.0,
            help="Frekuensi maksimum untuk analisis PSD.",
        )

        psd_n_fft_auto = st.checkbox("n_fft Otomatis", value=True,
            help="Jika aktif, n_fft = 2 × sampling rate "
                 "(resolusi ~0.5 Hz). Jika nonaktif, atur manual.",
        )
        psd_n_fft = None
        if not psd_n_fft_auto:
            psd_n_fft = st.slider(
                "n_fft", 64, 4096, 512, step=64,
                help="Panjang FFT. Semakin besar → resolusi frekuensi "
                     "lebih halus, tapi butuh data lebih panjang.",
            )

        # --- Epoching & Sliding Windows ---
        st.markdown("### Epoching & Sliding Windows")
        use_epoching = st.toggle("Aktifkan Epoching", value=False)
        
        epoch_mode = "fixed"
        epoch_duration = DEFAULT_EPOCH_DURATION
        window_overlap = DEFAULT_WINDOW_OVERLAP
        use_epoch_reject = False
        epoch_reject_threshold = DEFAULT_EPOCH_REJECT_UV
        
        if use_epoching:
            epoch_mode_str = st.radio(
                "Mode Epoch", ["Fixed Epoch", "Sliding Window"],
                horizontal=True, label_visibility="collapsed"
            )
            epoch_mode = "fixed" if epoch_mode_str == "Fixed Epoch" else "sliding"
            
            if epoch_mode == "fixed":
                epoch_duration = st.slider(
                    "Durasi Epoch (s)", 
                    MIN_EPOCH_DURATION, MAX_EPOCH_DURATION, 
                    DEFAULT_EPOCH_DURATION, 0.5
                )
            else:
                epoch_duration = st.slider(
                    "Window Size (s)", 
                    MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, 
                    DEFAULT_WINDOW_SIZE, 0.5
                )
                window_overlap = st.slider(
                    "Overlap (%)", 0, 75, int(DEFAULT_WINDOW_OVERLAP * 100), 25
                ) / 100.0
            
            use_epoch_reject = st.checkbox("Aktifkan Epoch Rejection", value=False)
            if use_epoch_reject:
                default_reject = 75.0 if use_amplitude else DEFAULT_EPOCH_REJECT_UV
                epoch_reject_threshold = st.slider(
                    "Threshold Rejection (µV)", 25.0, 200.0, default_reject, 5.0,
                    help="Buang epoch yang nilai amplitudo maksimalnya melebihi threshold ini."
                )

        # --- Connectivity (PLI / wPLI) ---
        st.markdown("### Connectivity")
        use_connectivity = st.toggle("Analisis PLI/wPLI", value=False)
        connectivity_method = DEFAULT_CONNECTIVITY_METHOD
        connectivity_channels = []
        if use_connectivity:
            connectivity_method = st.selectbox(
                "Metode Konektivitas",
                CONNECTIVITY_METHODS,
                index=CONNECTIVITY_METHODS.index(DEFAULT_CONNECTIVITY_METHOD),
                format_func=lambda x: x.upper(),
                help="PLI: Phase Lag Index\nwPLI: Weighted Phase Lag Index (Lebih tahan terhadap noise dan volume conduction)"
            )
            available_channels = []
            if st.session_state.processor:
                available_channels = st.session_state.processor.channel_names
            else:
                # Fallback standard 10-20 jika dari ZIP batch & file belum diproses
                available_channels = [
                    "Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz", 
                    "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                    "F7", "F8", "T7", "T8", "P7", "P8",
                    "FC1", "FC2", "FC5", "FC6", "CP1", "CP2", "CP5", "CP6"
                ]
            
            connectivity_channels = st.multiselect(
                "Channel Spesifik", 
                options=available_channels,
                help="Pilih channel yang ingin dianalisis konektivitasnya. Biarkan kosong untuk semua channel (sangat lambat!). Jika dalam mode Batch, channel standar 10-20 disediakan."
            )

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
            "use_car": use_car,
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
            "psd_method": psd_method,
            "psd_fmin": psd_fmin,
            "psd_fmax": psd_fmax,
            "psd_n_fft": psd_n_fft,
            "use_epoching": use_epoching,
            "epoch_mode": epoch_mode,
            "epoch_duration": epoch_duration,
            "window_overlap": window_overlap,
            "use_epoch_reject": use_epoch_reject,
            "epoch_reject_threshold": epoch_reject_threshold,
            "use_connectivity": use_connectivity,
            "connectivity_method": connectivity_method,
            "connectivity_channels": connectivity_channels,
            "is_openbci": is_openbci,
            "is_openbci_batch": is_openbci_batch,
        }
