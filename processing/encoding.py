"""
Modul encoding — Batch encoding sinyal EEG ke matriks fitur.

Pipeline:
  Load EDF → Sliding Window (0.3s) → Feature Extraction → Flatten → CSV

Mendukung encoding batch seluruh EEGET-ALS Dataset (170 subjek × 9 skenario).
Output: matriks X [n_windows × n_features] + vektor y [n_windows] siap ML.
"""

import os
import json
import logging

import numpy as np
import pandas as pd
import mne

from config import (
    DEFAULT_SUBBANDS, DEFAULT_FEATURES,
    DEFAULT_ENCODING_WINDOW, DEFAULT_ENCODING_OVERLAP,
    EEGET_ALS_SCENARIOS,
)
from processing.epoching import EpochEngine
from processing.loader import EEGLoader

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Encode single EDF file                                             #
# ------------------------------------------------------------------ #

def encode_single_edf(edf_path, window_size=None, overlap=None,
                      subbands=None, features=None,
                      include_frequency=True,
                      psd_method="welch", psd_fmin=0.0, psd_fmax=49.0):
    """Encode satu file EDF menjadi DataFrame fitur per window.

    Parameters
    ----------
    edf_path : str
        Path ke file EDF.
    window_size : float
        Durasi window dalam detik (default: 0.3).
    overlap : float
        Rasio overlap 0.0–0.75 (default: 0.0).
    subbands : dict | None
        Subband definitions.
    features : list[str] | None
        Fitur time-domain.
    include_frequency : bool
        Hitung fitur frequency-domain (band_power, relative_power, peak_frequency).
    psd_method : str
        Metode PSD ('welch' atau 'multitaper').
    psd_fmin, psd_fmax : float
        Rentang frekuensi PSD.

    Returns
    -------
    pd.DataFrame
        Kolom: window_idx, start_time, end_time, channel, subband, + fitur.
        Kosong jika file tidak bisa di-load atau data terlalu pendek.
    dict
        Metadata: sfreq, n_channels, ch_names, duration_s, n_windows.
    """
    if window_size is None:
        window_size = DEFAULT_ENCODING_WINDOW
    if overlap is None:
        overlap = DEFAULT_ENCODING_OVERLAP
    if subbands is None:
        subbands = DEFAULT_SUBBANDS
    if features is None:
        features = DEFAULT_FEATURES

    # Load EDF
    loader = EEGLoader()
    try:
        loader.load_edf(edf_path)
    except Exception as e:
        logger.warning("Gagal load %s: %s", edf_path, e)
        return pd.DataFrame(), {}

    df = loader.extract_dataframe()
    channels = loader.raw.ch_names
    sfreq = loader.sfreq

    # Buat sliding windows
    windows = EpochEngine.create_sliding_windows(
        df, channels, sfreq,
        window_size=window_size, overlap=overlap,
    )

    if not windows:
        logger.warning("Tidak ada window dari %s (data terlalu pendek?)", edf_path)
        return pd.DataFrame(), {}

    # Ekstrak fitur per window
    feat_df = EpochEngine.compute_windowed_features(
        windows, channels, sfreq,
        subbands=subbands, features=features,
        include_frequency=include_frequency,
        psd_method=psd_method, psd_fmin=psd_fmin, psd_fmax=psd_fmax,
    )

    metadata = {
        "sfreq": sfreq,
        "n_channels": len(channels),
        "ch_names": list(channels),
        "duration_s": float(df["time"].iloc[-1] - df["time"].iloc[0]),
        "n_windows": len(windows),
    }

    return feat_df, metadata


# ------------------------------------------------------------------ #
#  Flatten: long format → wide format (1 row = 1 window)              #
# ------------------------------------------------------------------ #

def flatten_features(encoded_df):
    """Pivot dari format panjang ke format lebar.

    Input (long format):
        window_idx | channel | subband | mav | variance | std | band_power | ...
        0          | Cz      | Alpha   | 0.2 | 0.05     | 0.1 | 1.2        |
        0          | Cz      | Beta    | 0.1 | 0.03     | 0.08| 0.8        |

    Output (wide format):
        window_idx | Cz_Alpha_mav | Cz_Alpha_variance | Cz_Beta_mav | ...
        0          | 0.2          | 0.05              | 0.1         |

    Parameters
    ----------
    encoded_df : pd.DataFrame
        Output dari encode_single_edf (long format).

    Returns
    -------
    pd.DataFrame
        Wide format: 1 baris = 1 window, kolom = channel_subband_feature.
    """
    if encoded_df.empty:
        return pd.DataFrame()

    # Identifikasi kolom meta vs fitur
    meta_cols = {"window_idx", "start_time", "end_time", "channel", "subband",
                 "subject_id", "scenario"}
    feat_cols = [c for c in encoded_df.columns if c not in meta_cols]

    if not feat_cols:
        return pd.DataFrame()

    # Buat kolom gabungan channel_subband
    df = encoded_df.copy()
    df["ch_sb"] = df["channel"] + "_" + df["subband"]

    # Tentukan kolom group (window identifier)
    group_cols = ["window_idx"]
    for extra in ["subject_id", "scenario", "start_time", "end_time"]:
        if extra in df.columns:
            group_cols.append(extra)

    # Pivot setiap fitur
    pivoted_parts = []
    for feat in feat_cols:
        pivot = df.pivot_table(
            index=group_cols, columns="ch_sb", values=feat,
            aggfunc="first",
        )
        # Rename kolom: ch_sb → ch_sb_feat
        pivot.columns = [f"{col}_{feat}" for col in pivot.columns]
        pivoted_parts.append(pivot)

    result = pd.concat(pivoted_parts, axis=1).reset_index()
    return result


# ------------------------------------------------------------------ #
#  Encode seluruh dataset EEGET-ALS                                   #
# ------------------------------------------------------------------ #

def _find_edf_path(dataset_root, subject_id, scenario_num):
    """Cari file EDF untuk subjek dan skenario tertentu.

    Struktur: dataset_root/subject_id/time1/scenarioN/EEG.edf

    Returns
    -------
    str | None
    """
    scenario_dir = os.path.join(
        dataset_root, subject_id, "time1",
        f"scenario{scenario_num}",
    )
    edf_path = os.path.join(scenario_dir, "EEG.edf")
    if os.path.isfile(edf_path):
        return edf_path
    return None


def _get_subject_ids(dataset_root, subject_range=None):
    """Ambil daftar subject_id dari folder dataset.

    Parameters
    ----------
    dataset_root : str
    subject_range : tuple(int, int) | None
        (start, end) inclusive. Misal (1, 10) = id1..id10.
        None = semua subjek normal (id*).

    Returns
    -------
    list[str]  Sorted subject IDs.
    """
    all_dirs = []
    for name in os.listdir(dataset_root):
        full = os.path.join(dataset_root, name)
        if os.path.isdir(full) and name.startswith("id"):
            all_dirs.append(name)

    # Sort secara numerik
    def _sort_key(s):
        num = s.replace("id", "")
        try:
            return int(num)
        except ValueError:
            return 999999

    all_dirs.sort(key=_sort_key)

    if subject_range is not None:
        start, end = subject_range
        all_dirs = [
            d for d in all_dirs
            if start <= _sort_key(d) <= end
        ]

    return all_dirs


def encode_dataset(dataset_root, subject_range=None, scenarios=None,
                   window_size=None, overlap=None,
                   subbands=None, features=None,
                   include_frequency=True,
                   psd_method="welch", psd_fmin=0.0, psd_fmax=49.0,
                   progress_callback=None):
    """Encode seluruh EEGET-ALS dataset menjadi satu DataFrame besar.

    Parameters
    ----------
    dataset_root : str
        Root folder dataset (misal: 'C:/Users/.../EEGET-ALS Dataset').
    subject_range : tuple(int, int) | None
        Range subjek (1, 170). None = semua.
    scenarios : list[int] | None
        Skenario yang di-encode (1-9). None = semua 9 skenario.
    window_size, overlap : float
        Parameter sliding window.
    subbands, features : dict, list
        Parameter feature extraction.
    include_frequency : bool
    psd_method, psd_fmin, psd_fmax : parameter PSD.
    progress_callback : callable | None
        Fungsi callback(subject_id, scenario, current, total) untuk progress bar.

    Returns
    -------
    pd.DataFrame
        Long format: window_idx, channel, subband, fitur, subject_id, scenario.
    dict
        Summary: n_subjects, n_scenarios, n_total_windows, n_failed.
    """
    if scenarios is None:
        scenarios = list(range(1, 10))

    subject_ids = _get_subject_ids(dataset_root, subject_range)
    total_tasks = len(subject_ids) * len(scenarios)
    current = 0
    n_failed = 0
    n_total_windows = 0

    all_dfs = []

    for subj_id in subject_ids:
        for sc_num in scenarios:
            current += 1

            edf_path = _find_edf_path(dataset_root, subj_id, sc_num)
            if edf_path is None:
                logger.info("EDF tidak ditemukan: %s scenario%d", subj_id, sc_num)
                n_failed += 1
                if progress_callback:
                    progress_callback(subj_id, sc_num, current, total_tasks)
                continue

            feat_df, meta = encode_single_edf(
                edf_path, window_size=window_size, overlap=overlap,
                subbands=subbands, features=features,
                include_frequency=include_frequency,
                psd_method=psd_method, psd_fmin=psd_fmin, psd_fmax=psd_fmax,
            )

            if feat_df.empty:
                n_failed += 1
                if progress_callback:
                    progress_callback(subj_id, sc_num, current, total_tasks)
                continue

            # Tambahkan metadata subjek dan skenario
            feat_df.insert(0, "subject_id", subj_id)
            feat_df.insert(1, "scenario", sc_num)

            all_dfs.append(feat_df)
            n_total_windows += meta.get("n_windows", 0)

            if progress_callback:
                progress_callback(subj_id, sc_num, current, total_tasks)

            logger.info(
                "Encoded %s/scenario%d: %d windows",
                subj_id, sc_num, meta.get("n_windows", 0),
            )

    summary = {
        "n_subjects": len(subject_ids),
        "n_scenarios": len(scenarios),
        "n_total_files": total_tasks,
        "n_failed": n_failed,
        "n_success": total_tasks - n_failed,
        "n_total_windows": n_total_windows,
    }

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        return result, summary

    return pd.DataFrame(), summary


# ------------------------------------------------------------------ #
#  Export helpers                                                      #
# ------------------------------------------------------------------ #

def encode_cached_items(cached_items, window_size=None, overlap=None,
                         subbands=None, features=None,
                         include_frequency=True,
                         psd_method="welch", psd_fmin=0.0, psd_fmax=49.0,
                         progress_callback=None):
    """Encode dari cached items (hasil batch ZIP yang sudah di-preprocess).

    Setiap ``cached_items`` berisi DataFrame sinyal yang sudah di-filter,
    ICA, dll. Step ini hanya melakukan sliding window + feature extraction
    (tanpa load ulang EDF).

    Parameters
    ----------
    cached_items : list[dict]
        Tiap item minimal harus punya keys:
            ``df`` (pd.DataFrame), ``channels`` (list),
            ``sfreq`` (float), ``subject_id`` (str),
            ``scenario`` (str), ``scenario_id`` (int),
            ``filename`` (str).
        Catatan: ``df`` bisa berupa objek DataFrame langsung, atau
        path string ke pickle file (akan di-load otomatis).
    progress_callback : callable | None
        ``(subject_id, scenario_id, current, total)``.

    Returns
    -------
    tuple (pd.DataFrame, dict)
        Long-format feature DataFrame + summary dict.
    """
    if window_size is None:
        window_size = DEFAULT_ENCODING_WINDOW
    if overlap is None:
        overlap = DEFAULT_ENCODING_OVERLAP
    if subbands is None:
        subbands = DEFAULT_SUBBANDS
    if features is None:
        features = DEFAULT_FEATURES

    total = len(cached_items)
    all_dfs = []
    n_failed = 0
    n_total_windows = 0

    for idx, item in enumerate(cached_items, start=1):
        subj = item.get("subject_id", "")
        scen_id = item.get("scenario_id", -1)

        # Load df dari pickle jika path string
        raw_df = item["df"]
        if isinstance(raw_df, str):
            try:
                raw_df = pd.read_pickle(raw_df)
            except Exception as exc:
                logger.warning("Gagal load cache %s: %s", raw_df, exc)
                n_failed += 1
                if progress_callback:
                    progress_callback(subj, scen_id, idx, total)
                continue

        channels = item.get("channels") or [
            c for c in raw_df.columns if c not in ("time", "marker")
        ]
        sfreq = item["sfreq"]

        try:
            windows = EpochEngine.create_sliding_windows(
                raw_df, channels, sfreq,
                window_size=window_size, overlap=overlap,
            )
        except Exception as exc:
            logger.warning("Sliding window gagal untuk %s: %s",
                           item.get("filename"), exc)
            n_failed += 1
            if progress_callback:
                progress_callback(subj, scen_id, idx, total)
            continue

        if not windows:
            n_failed += 1
            if progress_callback:
                progress_callback(subj, scen_id, idx, total)
            continue

        feat_df = EpochEngine.compute_windowed_features(
            windows, channels, sfreq,
            subbands=subbands, features=features,
            include_frequency=include_frequency,
            psd_method=psd_method, psd_fmin=psd_fmin, psd_fmax=psd_fmax,
        )

        if feat_df.empty:
            n_failed += 1
            if progress_callback:
                progress_callback(subj, scen_id, idx, total)
            continue

        feat_df.insert(0, "subject_id", subj)
        feat_df.insert(1, "scenario", scen_id)
        all_dfs.append(feat_df)
        n_total_windows += len(windows)

        if progress_callback:
            progress_callback(subj, scen_id, idx, total)

    summary = {
        "n_total_files": total,
        "n_failed": n_failed,
        "n_success": total - n_failed,
        "n_total_windows": n_total_windows,
    }

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True), summary
    return pd.DataFrame(), summary


def encode_and_export(dataset_root, output_path, flatten=True, **kwargs):
    """Encode dataset dan simpan ke CSV.

    Parameters
    ----------
    dataset_root : str
        Root folder EEGET-ALS Dataset.
    output_path : str
        Path output CSV.
    flatten : bool
        Jika True, flatten ke wide format (1 row = 1 window).
        Jika False, simpan long format.
    **kwargs
        Parameter tambahan untuk encode_dataset.

    Returns
    -------
    dict  Summary encoding.
    """
    encoded_df, summary = encode_dataset(dataset_root, **kwargs)

    if encoded_df.empty:
        logger.warning("Tidak ada data yang berhasil di-encode.")
        return summary

    if flatten:
        result = flatten_features(encoded_df)
    else:
        result = encoded_df

    # Buat direktori output jika belum ada
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.to_csv(output_path, index=False)

    summary["output_path"] = output_path
    summary["output_shape"] = result.shape
    summary["flatten"] = flatten

    logger.info("Encoding selesai. Output: %s, Shape: %s", output_path, result.shape)
    return summary
