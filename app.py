"""
EEG Analysis Tool - Streamlit Application
Dashboard untuk analisa data EEG secara interaktif.
Desain terinspirasi dashboard AeuxGlobal dengan aksen warna ITS Robocon.
"""

import streamlit as st
import pandas as pd

from eeg_processor import EEGProcessor
from eeg_visualizer import EEGVisualizer
from config import (
    DEFAULT_SUBBANDS, DEFAULT_FEATURES, ICA_METHODS, NOTCH_FREQUENCIES,
    ACCENT_PRIMARY, ACCENT_SECONDARY, ACCENT_LIGHT, ACCENT_LIGHTER,
    ACCENT_PALE, BG_DARK, BG_CARD_DARK, BG_SIDEBAR, BORDER_DARK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TASK_COLORS,
)

# ------------------------------------------------------------------ #
#  Page config                                                        #
# ------------------------------------------------------------------ #
st.set_page_config(page_title="EEG Analysis Tool", layout="wide", page_icon="brain")

# ------------------------------------------------------------------ #
#  Dashboard CSS                                                      #
# ------------------------------------------------------------------ #
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* === Global === */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {BG_DARK};
}}

/* === Sidebar === */
section[data-testid="stSidebar"] {{
    background: {BG_SIDEBAR};
    border-right: 1px solid {BORDER_DARK};
}}

section[data-testid="stSidebar"] .stMarkdown h1 {{
    font-size: 1.25rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {ACCENT_LIGHT};
    padding-bottom: 10px;
    margin-bottom: 18px;
    letter-spacing: -0.02em;
}}

section[data-testid="stSidebar"] .stMarkdown h3 {{
    font-size: 0.88rem;
    font-weight: 600;
    color: {ACCENT_PALE};
    margin-top: 20px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* === Dashboard metric cards === */
div[data-testid="stMetric"] {{
    background: {BG_CARD_DARK};
    border: 1px solid {BORDER_DARK};
    border-radius: 14px;
    padding: 18px 20px;
    transition: border-color 0.2s;
}}

div[data-testid="stMetric"]:hover {{
    border-color: {ACCENT_LIGHT};
}}

div[data-testid="stMetric"] label {{
    font-weight: 500;
    color: {TEXT_SECONDARY};
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-weight: 800;
    color: {TEXT_PRIMARY};
    font-size: 1.6rem;
}}

/* === Cards / containers === */
div[data-testid="stExpander"] {{
    background: {BG_CARD_DARK};
    border: 1px solid {BORDER_DARK};
    border-radius: 14px;
}}

/* === Tabs === */
button[data-baseweb="tab"] {{
    font-weight: 500;
    font-size: 0.9rem;
}}

/* === Data table === */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER_DARK};
    border-radius: 12px;
    overflow: hidden;
}}

/* === Primary button === */
button[kind="primary"] {{
    background: linear-gradient(135deg, {ACCENT_SECONDARY}, {ACCENT_PRIMARY});
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: all 0.25s;
    letter-spacing: 0.01em;
}}

button[kind="primary"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(13, 71, 161, 0.45);
}}

/* === Custom elements === */
.dashboard-header {{
    font-size: 1.9rem;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.03em;
    margin-bottom: 2px;
}}

.dashboard-subtitle {{
    font-size: 0.92rem;
    color: {TEXT_MUTED};
    margin-bottom: 22px;
}}

.section-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    border-left: 3px solid {ACCENT_LIGHT};
    padding-left: 12px;
    margin: 28px 0 14px 0;
    letter-spacing: -0.01em;
}}

.log-box {{
    background: {BG_CARD_DARK};
    border: 1px solid {BORDER_DARK};
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.83rem;
    color: {TEXT_SECONDARY};
    line-height: 1.8;
}}

.card {{
    background: {BG_CARD_DARK};
    border: 1px solid {BORDER_DARK};
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
}}

.task-badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px 4px;
    color: white;
}}

.info-label {{
    font-size: 0.78rem;
    font-weight: 500;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}}

.info-value {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Session state                                                      #
# ------------------------------------------------------------------ #
def init_state():
    defaults = {
        "processor": None,
        "raw_info": None,
        "df_data": None,
        "features_df": None,
        "task_features_df": None,
        "task_summary_df": None,
        "processed": False,
        # Batch analysis state
        "batch_mode": False,
        "batch_df": None,
        "batch_tasks": [],
        "batch_processed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ------------------------------------------------------------------ #
#  Sidebar                                                            #
# ------------------------------------------------------------------ #
def render_sidebar():
    with st.sidebar:
        st.markdown("# EEG Analysis Tool")

        # --- File Upload ---
        st.markdown("### Unggah File")
        upload_type = st.radio("Tipe", ["File EDF", "File ZIP (dataset)"],
                               horizontal=True, label_visibility="collapsed")

        uploaded = None
        edf_in_zip = None
        batch_mode = False

        if upload_type == "File EDF":
            uploaded = st.file_uploader("Pilih file EDF", type=["edf"],
                                        label_visibility="collapsed")
        else:
            uploaded = st.file_uploader("Pilih file ZIP", type=["zip"],
                                        label_visibility="collapsed")
            if uploaded is not None:
                # Reset pointer just in case
                uploaded.seek(0)
                try:
                    with st.spinner("Membaca struktur ZIP..."):
                        edf_list = EEGProcessor.list_edf_in_zip(uploaded)

                    if edf_list:
                        # Batch toggle — only if multiple files in ZIP
                        if len(edf_list) > 1:
                            batch_mode = st.toggle("Batch Analisis (semua file)",
                                                   value=st.session_state.batch_mode,
                                                   key="batch_toggle")
                            st.session_state.batch_mode = batch_mode

                        if not batch_mode:
                            edf_in_zip = st.selectbox("File EDF dalam ZIP", edf_list)
                        else:
                            st.info(f"{len(edf_list)} file EDF ditemukan. Klik **Proses Batch** untuk memproses semua.")
                    else:
                        st.warning("Tidak ada file EDF dalam ZIP.")
                        uploaded = None
                except Exception as e:
                    st.error(f"Gagal membaca ZIP: {e}")
                    uploaded = None

        if uploaded is not None and not batch_mode:
            _load_data(uploaded, upload_type, edf_in_zip)

        # --- Channel Selection ---
        selected_channels = []
        if st.session_state.raw_info:
            st.markdown("### Channel")
            all_ch = st.session_state.raw_info["channels"]
            selected_channels = st.multiselect(
                "Channel", all_ch, label_visibility="collapsed",
            )

        # --- Task Selection ---
        selected_tasks = []
        if st.session_state.processor:
            tasks = st.session_state.processor.get_task_list()
            if tasks:
                st.markdown("### Task / Marker")
                selected_tasks = st.multiselect(
                    "Task", tasks, label_visibility="collapsed",
                )

        # --- Filter Config ---
        st.markdown("### Filter")

        use_notch = st.toggle("Notch Filter", value=False)
        notch_freq = 50.0
        if use_notch:
            notch_freq = st.selectbox("Frekuensi Notch (Hz)", NOTCH_FREQUENCIES)

        bp_mode = st.radio("Bandpass", ["Preset Subband", "Custom Range"],
                           horizontal=True, label_visibility="collapsed")

        bp_low, bp_high, bp_order = 0.5, 49.0, 5
        selected_subbands = {}

        if bp_mode == "Preset Subband":
            sb_names = st.multiselect(
                "Subband", list(DEFAULT_SUBBANDS.keys()),
                label_visibility="collapsed",
            )
            selected_subbands = {k: DEFAULT_SUBBANDS[k] for k in sb_names}
            if sb_names:
                bp_low = min(v[0] for v in selected_subbands.values())
                bp_high = max(v[1] for v in selected_subbands.values())
        else:
            col_a, col_b = st.columns(2)
            bp_low = col_a.number_input("Low (Hz)", 0.1, 100.0, 0.5, 0.1)
            bp_high = col_b.number_input("High (Hz)", 0.2, 100.0, 49.0, 0.1)
            bp_order = st.slider("Order", 1, 10, 5)

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

        # --- Features ---
        st.markdown("### Fitur Statistik")
        selected_features = st.multiselect(
            "Fitur", DEFAULT_FEATURES,
            label_visibility="collapsed",
        )

        # --- Process ---
        st.divider()
        if batch_mode:
            batch_btn = st.button("Proses Batch", type="primary",
                                  use_container_width=True,
                                  disabled=(uploaded is None))
            process_btn = False
        else:
            batch_btn = False
            process_btn = st.button("Proses Data", type="primary",
                                    use_container_width=True,
                                    disabled=(st.session_state.processor is None))

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
            "use_ica": use_ica,
            "ica_n": ica_n,
            "ica_method": ica_method,
            "features": selected_features,
            "subbands": selected_subbands if bp_mode == "Preset Subband" else None,
        }


def _load_data(uploaded, upload_type, edf_in_zip):
    """Load EDF lazily."""
    cache_key = f"{uploaded.name}_{edf_in_zip}"
    if st.session_state.get("_loaded_key") == cache_key:
        return

    proc = EEGProcessor()
    try:
        if upload_type == "File EDF":
            info = proc.load_edf(uploaded)
        else:
            uploaded.seek(0)
            info = proc.load_edf_from_zip(uploaded, edf_in_zip)
        st.session_state.processor = proc
        st.session_state.raw_info = info
        st.session_state.processed = False
        st.session_state._loaded_key = cache_key
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")


# ------------------------------------------------------------------ #
#  Main content                                                       #
# ------------------------------------------------------------------ #
def render_main(cfg):
    st.markdown('<p class="dashboard-header">EEG Analysis Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">Unggah file EDF atau ZIP untuk memulai analisis sinyal EEG</p>',
                unsafe_allow_html=True)

    # ---- Batch mode ----
    if cfg.get("batch_mode"):
        if cfg.get("batch_process"):
            run_batch_processing(cfg)
        if st.session_state.batch_processed:
            render_batch_results(cfg)
        elif not cfg.get("batch_process"):
            st.info("Klik **Proses Batch** di panel samping untuk memproses semua file EDF dalam ZIP.")
        return

    # ---- Single file mode ----
    if st.session_state.processor is None:
        st.info("Silakan unggah file EDF atau ZIP melalui panel samping.")
        return

    proc: EEGProcessor = st.session_state.processor
    info = st.session_state.raw_info

    # --- Overview cards ---
    render_overview(info, proc)

    # --- Process ---
    if cfg["process"]:
        run_processing(proc, cfg)

    # --- Results ---
    if st.session_state.processed:
        render_results(proc, cfg)


def render_overview(info, proc: EEGProcessor):
    """Kartu ringkasan data EDF di bagian atas dashboard."""
    st.markdown('<p class="section-title">Ringkasan Data</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Channel", info["n_channels"])
    c2.metric("Sampling Rate", f'{info["sfreq"]:.0f} Hz')
    c3.metric("Durasi", f'{info["duration_s"]:.1f} s')
    c4.metric("Annotations", info["n_annotations"])

    # Task badges
    tasks = proc.get_task_list()
    if tasks:
        badge_html = ""
        for t in tasks:
            color = TASK_COLORS.get(t, ACCENT_LIGHT)
            badge_html += f'<span class="task-badge" style="background:{color}">{t}</span>'
        st.markdown(
            f'<div style="margin-top:6px"><span class="info-label">Task yang terdeteksi</span>'
            f'<div style="margin-top:6px">{badge_html}</div></div>',
            unsafe_allow_html=True,
        )


def run_processing(proc: EEGProcessor, cfg):
    """Jalankan pipeline filtering, ICA, dan ekstraksi data."""
    proc.raw = proc.raw_original.copy()
    proc.processing_log = ["File EDF berhasil dimuat"]

    with st.spinner("Memproses data..."):
        if cfg["channels"]:
            proc.pick_channels(cfg["channels"])

        if cfg["use_notch"]:
            proc.apply_notch(freq=cfg["notch_freq"])

        proc.apply_bandpass(cfg["bp_low"], cfg["bp_high"], order=cfg["bp_order"])

        if cfg["use_ica"]:
            proc.apply_ica(n_components=cfg["ica_n"], method=cfg["ica_method"])

        df = proc.extract_dataframe()
        st.session_state.df_data = df

        channels = cfg["channels"] or proc.channel_names
        subbands = cfg["subbands"] or DEFAULT_SUBBANDS
        features = cfg["features"] or DEFAULT_FEATURES

        # Fitur keseluruhan
        features_df = proc.compute_subband_features(df, channels, subbands, features)
        st.session_state.features_df = features_df

        # Fitur per task
        tasks = cfg["tasks"]
        if tasks:
            task_features_df = proc.compute_task_features(df, channels, tasks, subbands, features)
            task_summary_df = proc.get_task_summary(df)
            st.session_state.task_features_df = task_features_df
            st.session_state.task_summary_df = task_summary_df
        else:
            st.session_state.task_features_df = pd.DataFrame()
            st.session_state.task_summary_df = pd.DataFrame()

        st.session_state.processed = True

    st.success("Pemrosesan selesai.")


def render_results(proc: EEGProcessor, cfg):
    """Tampilkan semua hasil."""
    df = st.session_state.df_data
    features_df = st.session_state.features_df
    task_features_df = st.session_state.task_features_df
    task_summary_df = st.session_state.task_summary_df
    channels = cfg["channels"] or proc.channel_names
    info = st.session_state.raw_info
    tasks = cfg["tasks"]

    # --- Processing log ---
    st.markdown('<p class="section-title">Log Pemrosesan</p>', unsafe_allow_html=True)
    log_html = "<br>".join(f"&bull; {entry}" for entry in proc.get_processing_log())
    st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    # --- Signal plot ---
    st.markdown('<p class="section-title">Sinyal EEG (Terfilter)</p>', unsafe_allow_html=True)
    max_t = float(df["time"].max())
    window = min(10.0, max_t)
    t_range = st.slider("Rentang Waktu (s)", 0.0, max_t, (0.0, window), step=0.5)
    fig_signal = EEGVisualizer.plot_raw_signal(df, channels, time_range=t_range,
                                               title="Sinyal Terfilter")
    st.plotly_chart(fig_signal, use_container_width=True)

    # --- Task section ---
    if tasks and task_summary_df is not None and not task_summary_df.empty:
        render_task_section(proc, df, channels, tasks, task_features_df, task_summary_df, cfg)

    # --- Data table ---
    st.markdown('<p class="section-title">Tabel Data</p>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=320)

    # --- EDA tabs ---
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    tab_dist, tab_psd, tab_corr, tab_annot = st.tabs(
        ["Distribusi", "PSD", "Korelasi", "Marker"]
    )

    with tab_dist:
        fig_dist = EEGVisualizer.plot_signal_distribution(df, channels)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)

    with tab_psd:
        fig_psd = EEGVisualizer.plot_psd(proc.raw)
        st.plotly_chart(fig_psd, use_container_width=True)

    with tab_corr:
        fig_corr = EEGVisualizer.plot_channel_correlation(df, channels)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Pilih minimal 2 channel untuk melihat korelasi.")

    with tab_annot:
        annotations = info.get("annotations", [])
        fig_annot = EEGVisualizer.plot_annotation_summary(annotations)
        if fig_annot:
            st.plotly_chart(fig_annot, use_container_width=True)
        else:
            st.info("File ini tidak memiliki annotation/marker.")

    # --- Feature summary ---
    st.markdown('<p class="section-title">Ringkasan Fitur</p>', unsafe_allow_html=True)
    if not features_df.empty:
        st.dataframe(features_df, use_container_width=True, height=280)
        feat_cols = [c for c in features_df.columns if c not in ("channel", "subband")]
        if feat_cols:
            selected_feat = st.selectbox("Visualisasi fitur", feat_cols, key="feat_select")
            fig_feat = EEGVisualizer.plot_feature_comparison(features_df, selected_feat)
            if fig_feat:
                st.plotly_chart(fig_feat, use_container_width=True)


def render_task_section(proc, df, channels, tasks, task_features_df, task_summary_df, cfg):
    """Section khusus analisis berbasis task."""
    st.markdown('<p class="section-title">Analisis per Task</p>', unsafe_allow_html=True)

    # Task summary row
    col_pie, col_table = st.columns([1, 2])

    with col_pie:
        fig_pie = EEGVisualizer.plot_task_pie(task_summary_df)
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

    # Per-task signal viewer
    task_view = st.selectbox("Lihat sinyal task", ["-- Semua --"] + list(tasks), key="task_view")
    if task_view != "-- Semua --":
        task_df = proc.extract_task_segments(df, task_view)
        if not task_df.empty:
            fig_task = EEGVisualizer.plot_raw_signal(task_df, channels,
                                                     title=f"Sinyal - {task_view}")
            st.plotly_chart(fig_task, use_container_width=True)
        else:
            st.info(f"Tidak ada data untuk task '{task_view}'.")

    # Per-task feature comparison
    if task_features_df is not None and not task_features_df.empty:
        st.markdown("**Perbandingan Fitur antar Task**")
        feat_cols = [c for c in task_features_df.columns if c not in ("task", "channel", "subband")]
        if feat_cols:
            sel_feat = st.selectbox("Fitur", feat_cols, key="task_feat_select")
            fig_tf = EEGVisualizer.plot_task_feature_comparison(task_features_df, sel_feat)
            if fig_tf:
                st.plotly_chart(fig_tf, use_container_width=True)

        st.dataframe(task_features_df, use_container_width=True, height=280, hide_index=True)


# ------------------------------------------------------------------ #
#  Batch processing & results                                         #
# ------------------------------------------------------------------ #

def run_batch_processing(cfg):
    """Jalankan batch analysis: baca semua EDF dari ZIP, hitung fitur."""
    uploaded = cfg.get("uploaded")
    if uploaded is None:
        st.warning("File ZIP belum diunggah.")
        return

    subbands = cfg["subbands"] or DEFAULT_SUBBANDS
    features = cfg["features"] or DEFAULT_FEATURES
    channels = cfg["channels"] if cfg["channels"] else None

    progress_bar = st.progress(0, text="Memulai batch processing...")

    def _progress(current, total, fname):
        if total > 0:
            progress_bar.progress(current / total,
                                  text=f"Memproses {current}/{total}: {fname}")

    batch_df, batch_tasks = EEGProcessor.process_batch_zip(
        uploaded, channels=channels, subbands=subbands,
        features=features, progress_cb=_progress,
    )

    progress_bar.empty()

    if batch_df.empty:
        st.error("Tidak ada data fitur yang berhasil diekstrak dari ZIP.")
        return

    st.session_state.batch_df = batch_df
    st.session_state.batch_tasks = batch_tasks
    st.session_state.batch_processed = True
    st.success(f"Batch processing selesai: {batch_df['filename'].nunique()} file, "
               f"{len(batch_tasks)} task ditemukan.")


def render_batch_results(cfg):
    """Tampilkan hasil batch analysis & delta comparison."""
    batch_df = st.session_state.batch_df
    batch_tasks = st.session_state.batch_tasks

    if batch_df is None or batch_df.empty:
        return

    # --- Category filter ---
    categories = sorted(batch_df["category"].unique().tolist()) if "category" in batch_df.columns else []

    if len(categories) > 1:
        st.markdown('<p class="section-title">Filter Kategori</p>', unsafe_allow_html=True)

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
            filtered_df = batch_df[batch_df["category"] == active_category].copy()
    else:
        filtered_df = batch_df
        active_category = categories[0] if categories else "Semua"

    # --- Scenario filter ---
    scenarios = sorted(filtered_df["scenario"].unique().tolist()) if "scenario" in filtered_df.columns else []
    if len(scenarios) > 1:
        selected_scenarios = st.multiselect(
            "Filter Skenario", scenarios, default=scenarios,
            key="scenario_filter",
        )
        if selected_scenarios:
            filtered_df = filtered_df[filtered_df["scenario"].isin(selected_scenarios)].copy()

    # --- Time filter ---
    times = sorted(filtered_df["time"].unique().tolist()) if "time" in filtered_df.columns else []
    if len(times) > 1:
        selected_times = st.multiselect(
            "Filter Time", times, default=times,
            key="time_filter",
        )
        if selected_times:
            filtered_df = filtered_df[filtered_df["time"].isin(selected_times)].copy()

    # --- Normalization toggle ---
    meta_cols = {"filename", "category", "subject", "time", "scenario",
                 "task", "channel", "subband"}
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
            norm_scope_val = "subject" if norm_scope == "Per Subjek" else "subject_scenario"
        else:
            norm_scope_val = "subject"
    with norm_col3:
        if use_norm:
            if norm_scope_val == "subject":
                st.caption("Normalisasi per orang (semua skenario digabung).")
            else:
                st.caption("Normalisasi per orang per skenario (lebih granular).")

    if use_norm and feat_cols:
        filtered_df = EEGProcessor.normalize_per_subject(
            filtered_df, feat_cols, method="zscore", scope=norm_scope_val
        )

    # --- Ringkasan ---
    st.markdown('<p class="section-title">Ringkasan Batch</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kategori", active_category)
    c2.metric("Subjek", filtered_df["subject"].nunique() if "subject" in filtered_df.columns else "–")
    c3.metric("Task Ditemukan", len(batch_tasks))
    c4.metric("Total Baris", f"{len(filtered_df):,}")

    # Task badges
    badge_html = ""
    for t in batch_tasks:
        color = TASK_COLORS.get(t, ACCENT_LIGHT)
        badge_html += f'<span class="task-badge" style="background:{color}">{t}</span>'
    st.markdown(f'<div style="margin:8px 0">{badge_html}</div>', unsafe_allow_html=True)

    # --- Tabs ---
    tab_overview, tab_delta, tab_data = st.tabs(
        ["Distribusi Fitur", "Analisis Delta", "Data Mentah"]
    )

    # ---- Tab 1: Overview (Grouped Bar) ----
    with tab_overview:
        if feat_cols:
            sel_feat = st.selectbox("Fitur", feat_cols, key="batch_feat_overview")

            # Filter task
            available_tasks = sorted(filtered_df["task"].unique().tolist()) if "task" in filtered_df.columns else []
            if available_tasks:
                sel_tasks_dist = st.multiselect(
                    "Filter Task", available_tasks, default=available_tasks,
                    key="dist_task_filter",
                )
                dist_df = filtered_df[filtered_df["task"].isin(sel_tasks_dist)].copy() if sel_tasks_dist else filtered_df
            else:
                dist_df = filtered_df

            # Pilih chart yang ingin ditampilkan
            st.markdown("**Pilih Visualisasi:**")
            show_grouped = st.checkbox("Grouped Bar (per Task & Subband)",
                                       value=False, key="show_grouped_bar")
            show_box = st.checkbox("Box Plot Overview", value=False,
                                   key="show_box_plot")
            show_summary_table = st.checkbox("Tabel Rata-rata per Task",
                                              value=False, key="show_summary_tbl")

            if show_grouped:
                fig_grouped = EEGVisualizer.plot_grouped_bar(
                    dist_df, sel_feat, group_col="task",
                    facet_col="channel", x_col="subband",
                    title=f"{sel_feat.capitalize()} per Task & Subband",
                )
                if fig_grouped:
                    st.plotly_chart(fig_grouped, use_container_width=True)

            if show_box:
                fig_box = EEGVisualizer.plot_batch_overview(dist_df, sel_feat)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)

            if show_summary_table:
                st.markdown("**Rata-rata Fitur per Task**")
                summary = dist_df.groupby("task")[feat_cols].agg(["mean", "std"])
                summary.columns = [f"{f} ({s})" for f, s in summary.columns]
                st.dataframe(summary, use_container_width=True)

    # ---- Tab 2: Delta ----
    with tab_delta:
        if len(batch_tasks) < 2:
            st.info("Diperlukan minimal 2 task untuk analisis delta.")
        else:
            st.markdown('<p class="section-title">Perbandingan Delta antar Task</p>',
                        unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                task_a = st.selectbox("Task A (Kondisi)", batch_tasks,
                                     index=0, key="delta_task_a")
            with col_b:
                default_b = 1 if len(batch_tasks) > 1 else 0
                task_b = st.selectbox("Task B (Baseline)", batch_tasks,
                                     index=default_b, key="delta_task_b")

            if task_a == task_b:
                st.warning("Pilih dua task yang berbeda.")
            else:
                sel_delta_feat = st.selectbox("Fitur Delta", feat_cols,
                                              key="delta_feat_select")

                delta_df, agg_df = EEGProcessor.calculate_task_delta(
                    filtered_df, task_a, task_b, feat_cols,
                )

                if delta_df.empty:
                    st.warning(f"Tidak ada data yang cocok untuk {task_a} vs {task_b}.")
                else:
                    # Pilih visualisasi
                    st.markdown("**Pilih Visualisasi:**")
                    show_delta_metrics = st.checkbox("Metrik Delta",
                                                     value=False, key="show_delta_metrics")
                    show_delta_bar = st.checkbox("Bar Chart Delta",
                                                 value=False, key="show_delta_bar")
                    show_delta_scatter = st.checkbox("Scatter Plot per Subjek",
                                                     value=False, key="show_delta_scatter")
                    show_delta_heat = st.checkbox("Heatmap Delta",
                                                  value=False, key="show_delta_heat")
                    show_delta_table = st.checkbox("Tabel Delta & Agregat",
                                                   value=False, key="show_delta_tbl")

                    dcol = f"delta_{sel_delta_feat}"

                    if show_delta_metrics and dcol in delta_df.columns:
                        m1, m2, m3 = st.columns(3)
                        m1.metric(f"Mean Δ {sel_delta_feat}",
                                  f"{delta_df[dcol].mean():+.4e}")
                        m2.metric(f"Std Δ {sel_delta_feat}",
                                  f"{delta_df[dcol].std():.4e}")
                        m3.metric("Pasangan Data", len(delta_df))

                    if show_delta_bar:
                        fig_bar = EEGVisualizer.plot_delta_bar(
                            agg_df, sel_delta_feat, task_a, task_b)
                        if fig_bar:
                            st.plotly_chart(fig_bar, use_container_width=True)

                    if show_delta_scatter and dcol in delta_df.columns:
                        import plotly.graph_objects as go

                        # Map filename → subject & category
                        meta_map = filtered_df.drop_duplicates(
                            subset=["filename"]
                        )[["filename", "subject", "category"]].copy()
                        scatter_df = delta_df.merge(meta_map, on="filename", how="left")
                        scatter_df["ch_sub"] = (scatter_df["channel"] + " / "
                                                + scatter_df["subband"])

                        # Filter kategori
                        scatter_cats = sorted(scatter_df["category"].dropna().unique().tolist())
                        sel_scatter_cat = st.multiselect(
                            "Filter Kategori (Scatter)", scatter_cats,
                            default=scatter_cats, key="scatter_cat_filter",
                        )
                        if sel_scatter_cat:
                            scatter_df = scatter_df[
                                scatter_df["category"].isin(sel_scatter_cat)
                            ]

                        fig_scatter = go.Figure()
                        subjects = sorted(scatter_df["subject"].dropna().unique().tolist())
                        for subj in subjects:
                            s_data = scatter_df[scatter_df["subject"] == subj]
                            cat_label = s_data["category"].iloc[0] if not s_data.empty else ""
                            fig_scatter.add_trace(go.Scatter(
                                x=s_data["ch_sub"],
                                y=s_data[dcol],
                                mode="markers",
                                name=f"{subj} ({cat_label})",
                                marker=dict(size=10, opacity=0.8),
                                hovertemplate=(
                                    f"<b>{subj}</b> ({cat_label})<br>"
                                    "Channel/Subband: %{x}<br>"
                                    f"Δ {sel_delta_feat}: " + "%{y:.6e}<br>"
                                    "<extra></extra>"
                                ),
                            ))

                        fig_scatter.update_layout(
                            title=f"Δ {sel_delta_feat.capitalize()} per Subjek ({task_a} − {task_b})",
                            xaxis_title="Channel / Subband",
                            yaxis_title=f"Δ {sel_delta_feat}",
                            template="plotly_dark",
                            height=500,
                            legend_title="Subjek",
                        )
                        fig_scatter.add_hline(y=0, line_dash="dash",
                                              line_color="gray", opacity=0.5)
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    if show_delta_heat:
                        fig_heat = EEGVisualizer.plot_delta_heatmap(
                            agg_df, sel_delta_feat, task_a, task_b)
                        if fig_heat:
                            st.plotly_chart(fig_heat, use_container_width=True)

                    if show_delta_table:
                        with st.expander("Tabel Delta Lengkap", expanded=True):
                            st.dataframe(delta_df, use_container_width=True,
                                         height=320, hide_index=True)
                        with st.expander("Statistik Agregat per Channel/Subband"):
                            st.dataframe(agg_df, use_container_width=True,
                                         hide_index=True)

                    # --- Perbandingan ALS vs Normal pada delta ini ---
                    has_both_cats = (
                        "category" in batch_df.columns
                        and set(batch_df["category"].unique()) >= {"ALS", "Normal"}
                    )
                    if has_both_cats:
                        st.divider()
                        st.markdown('<p class="section-title">'
                                    'Perbandingan ALS vs Normal</p>',
                                    unsafe_allow_html=True)

                        show_als_compare = st.checkbox(
                            "Tampilkan Perbandingan ALS vs Normal",
                            value=False, key="show_als_compare",
                        )

                        if show_als_compare:
                            # -- Opsi metode perbandingan --
                            cmp_mode_label = st.radio(
                                "Metode Perbandingan",
                                ["Delta saja", "Z-score saja", "Z-score + Delta"],
                                horizontal=True, key="cmp_mode",
                                help="Delta: selisih Task A − Task B. "
                                     "Z-score: normalisasi per subjek lalu "
                                     "bandingkan langsung. "
                                     "Z-score + Delta: normalisasi dulu, "
                                     "baru hitung selisih.",
                            )
                            cmp_mode_map = {
                                "Delta saja": "delta",
                                "Z-score saja": "zscore",
                                "Z-score + Delta": "both",
                            }
                            cmp_mode = cmp_mode_map[cmp_mode_label]

                            st.caption({
                                "delta": f"Menghitung Δ ({task_a} − {task_b}) "
                                         "per subjek, lalu bandingkan ALS vs Normal.",
                                "zscore": f"Z-score per subjek pada {task_a}, "
                                          "lalu bandingkan ALS vs Normal.",
                                "both": f"Z-score per subjek, lalu hitung "
                                        f"Δ ({task_a} − {task_b}), "
                                        "lalu bandingkan ALS vs Normal.",
                            }[cmp_mode])

                            # -- Opsi FDR & Effect Size (disabled for now) --
                            # opt_col1, opt_col2 = st.columns(2)
                            # with opt_col1:
                            #     use_fdr = st.checkbox(
                            #         "Koreksi FDR (Benjamini-Hochberg)",
                            #         value=True, key="use_fdr",
                            #     )
                            # with opt_col2:
                            #     use_effect = st.checkbox(
                            #         "Hitung Effect Size (Cohen's d)",
                            #         value=True, key="use_effect",
                            #     )
                            use_fdr = False
                            use_effect = False

                            # -- Data source --
                            compare_src = batch_df.copy()
                            if len(scenarios) > 1 and selected_scenarios:
                                compare_src = compare_src[
                                    compare_src["scenario"].isin(selected_scenarios)
                                ]
                            # Apply time filter if active
                            if "time" in compare_src.columns:
                                sel_times = (selected_times
                                             if "selected_times" in dir() else None)
                                if sel_times:
                                    compare_src = compare_src[
                                        compare_src["time"].isin(sel_times)
                                    ]

                            # -- Kontrol time: hanya time yang dimiliki semua subjek --
                            if "time" in compare_src.columns:
                                use_common_time = st.checkbox(
                                    "Hanya gunakan time yang dimiliki semua subjek",
                                    value=False, key="common_time",
                                    help="Memfilter data agar hanya time yang "
                                         "ada di SEMUA subjek yang dianalisis.",
                                )
                                if use_common_time:
                                    subj_times = compare_src.groupby("subject")["time"].apply(set)
                                    if len(subj_times) > 0:
                                        common_times = set.intersection(*subj_times)
                                        if common_times:
                                            compare_src = compare_src[
                                                compare_src["time"].isin(common_times)
                                            ]
                                            st.info(f"Time yang sama di semua subjek: "
                                                    f"{sorted(common_times)}")
                                        else:
                                            st.warning("Tidak ada time yang sama "
                                                       "di semua subjek.")

                            # -- Sampling --
                            als_subjects = sorted(
                                compare_src[compare_src["category"] == "ALS"]["subject"].unique()
                            )
                            norm_subjects = sorted(
                                compare_src[compare_src["category"] == "Normal"]["subject"].unique()
                            )

                            st.markdown(f"Tersedia: **{len(als_subjects)} ALS**, "
                                        f"**{len(norm_subjects)} Normal**")

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

                            import random
                            seed = st.number_input("Random Seed", value=42,
                                                   key="delta_seed",
                                                   help="Ganti seed untuk sampling berbeda")
                            rng = random.Random(seed)

                            sampled_als = (rng.sample(als_subjects, min(n_als, len(als_subjects)))
                                           if als_subjects else [])
                            sampled_norm = (rng.sample(norm_subjects, min(n_norm, len(norm_subjects)))
                                            if norm_subjects else [])

                            sampled_subjects = sampled_als + sampled_norm
                            sampled_df = compare_src[
                                compare_src["subject"].isin(sampled_subjects)
                            ]

                            if sampled_df.empty:
                                st.warning("Tidak ada data setelah sampling.")
                            else:
                                avs_cmp_df, avs_stats_df = EEGProcessor.compare_als_vs_normal(
                                    sampled_df, task_a, baseline_task=task_b,
                                    feature_cols=feat_cols,
                                    mode=cmp_mode,
                                    apply_fdr=use_fdr,
                                    compute_effect_size=use_effect,
                                )

                                if avs_cmp_df.empty or avs_stats_df.empty:
                                    st.warning("Tidak cukup data untuk perbandingan.")
                                else:
                                    # -- Metrics --
                                    col_p = f"p_{sel_delta_feat}"
                                    col_fdr = f"p_fdr_{sel_delta_feat}"
                                    col_d = f"cohend_{sel_delta_feat}"

                                    mc1, mc2, mc3, mc4 = st.columns(4)
                                    mc1.metric("Sampel ALS", len(sampled_als))
                                    mc2.metric("Sampel Normal", len(sampled_norm))
                                    if col_p in avs_stats_df.columns:
                                        sig_raw = (avs_stats_df[col_p] <= 0.05).sum()
                                        tot_n = avs_stats_df[col_p].notna().sum()
                                        mc3.metric("Signifikan (p≤0.05)",
                                                   f"{sig_raw}/{tot_n}")
                                    # if use_fdr and col_fdr in avs_stats_df.columns:
                                    #     sig_fdr = (avs_stats_df[col_fdr] <= 0.05).sum()
                                    #     mc4.metric("Signifikan (FDR≤0.05)",
                                    #                f"{sig_fdr}/{tot_n}")

                                    # -- Grouped bar chart --
                                    fig_avs = EEGVisualizer.plot_als_vs_normal(
                                        avs_stats_df, sel_delta_feat, task_a, task_b,
                                    )
                                    if fig_avs:
                                        st.plotly_chart(fig_avs, use_container_width=True)

                                    # -- P-value table --
                                    p_cols = [c for c in avs_stats_df.columns
                                              if c.startswith("p_") and not c.startswith("p_fdr_")]
                                    if p_cols:
                                        disp = avs_stats_df[
                                            ["channel", "subband"] + p_cols
                                        ].copy()

                                        def _hl(val):
                                            if isinstance(val, float) and val <= 0.05:
                                                return ("background-color: #065F46; "
                                                        "color: #34D399; "
                                                        "font-weight: bold")
                                            return ""

                                        styled = disp.style.applymap(
                                            _hl, subset=p_cols
                                        ).format(
                                            {c: "{:.4f}" for c in p_cols},
                                            na_rep="–",
                                        )
                                        st.markdown("**Tabel P-value**")
                                        st.dataframe(styled,
                                                     use_container_width=True,
                                                     hide_index=True)

                                    # -- Effect size table (disabled for now) --
                                    # d_cols = [c for c in avs_stats_df.columns
                                    #           if c.startswith("cohend_")]
                                    # if d_cols:
                                    #     def _interpret_d(d):
                                    #         if pd.isna(d):
                                    #             return "–"
                                    #         ad = abs(d)
                                    #         if ad < 0.2:
                                    #             return "Sangat Kecil"
                                    #         elif ad < 0.5:
                                    #             return "Kecil"
                                    #         elif ad < 0.8:
                                    #             return "Sedang"
                                    #         else:
                                    #             return "Besar"
                                    #
                                    #     d_disp = avs_stats_df[
                                    #         ["channel", "subband"] + d_cols
                                    #     ].copy()
                                    #     for dc in d_cols:
                                    #         interp_col = dc.replace("cohend_", "efek_")
                                    #         d_disp[interp_col] = d_disp[dc].apply(_interpret_d)
                                    #
                                    #     def _hl_d(val):
                                    #         if isinstance(val, float) and abs(val) >= 0.8:
                                    #             return ("background-color: #1E3A5F; "
                                    #                     "color: #60A5FA; "
                                    #                     "font-weight: bold")
                                    #         return ""
                                    #
                                    #     styled_d = d_disp.style.applymap(
                                    #         _hl_d, subset=d_cols
                                    #     ).format(
                                    #         {c: "{:.3f}" for c in d_cols},
                                    #         na_rep="–",
                                    #     )
                                    #     st.markdown("**Effect Size (Cohen's d)**")
                                    #     st.caption("Interpretasi: |d| < 0.2 = Sangat Kecil, "
                                    #                "< 0.5 = Kecil, < 0.8 = Sedang, ≥ 0.8 = Besar")
                                    #     st.dataframe(styled_d,
                                    #                  use_container_width=True,
                                    #                  hide_index=True)

    # ---- Tab 3: Raw Data ----
    with tab_data:
        st.markdown("**Seluruh Data Fitur Batch**")
        st.caption(f"Total: {len(filtered_df):,} baris × {len(filtered_df.columns)} kolom")

        MAX_DISPLAY = 500
        if len(filtered_df) > MAX_DISPLAY:
            st.dataframe(filtered_df.head(MAX_DISPLAY), use_container_width=True,
                         height=400, hide_index=True)
            st.info(f"Menampilkan {MAX_DISPLAY} baris pertama dari {len(filtered_df):,} total. "
                    "Gunakan tombol download untuk data lengkap.")
        else:
            st.dataframe(filtered_df, use_container_width=True, height=400,
                         hide_index=True)

        @st.cache_data
        def _convert_to_csv(df):
            return df.to_csv(index=False).encode("utf-8")

        csv_data = _convert_to_csv(filtered_df)
        st.download_button("Download CSV", csv_data,
                           file_name="batch_features.csv",
                           mime="text/csv")


# ------------------------------------------------------------------ #
#  Run                                                                #
# ------------------------------------------------------------------ #
cfg = render_sidebar()
render_main(cfg)
