"""
Modul pemrosesan EEG.
Menyediakan class EEGProcessor untuk loading, filtering, ICA, dan ekstraksi fitur.
Mendukung analisis berbasis task (resting, typing, thinking, dll).
"""

import io
import zipfile
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from scipy.signal import butter, filtfilt, iirnotch

from config import DEFAULT_SUBBANDS, DEFAULT_FEATURES


def _benjamini_hochberg(pvals):
    """Koreksi Benjamini-Hochberg (FDR) untuk array p-value.

    Parameters
    ----------
    pvals : array-like
        Array p-value mentah (tanpa NaN).

    Returns
    -------
    np.ndarray
        P-value yang sudah dikoreksi (adjusted).
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    adjusted = pvals * n / ranked
    # Enforce monotonicity from largest to smallest rank
    adjusted = np.minimum.accumulate(adjusted[np.argsort(-ranked)])[
        np.argsort(np.argsort(-ranked))
    ]
    return np.clip(adjusted, 0, 1)


class EEGProcessor:
    """Class utama untuk memproses data EEG dari file EDF."""

    def __init__(self):
        self.raw = None
        self.raw_original = None
        self.sfreq = None
        self.channel_names = []
        self.processing_log = []
        self._tmp_path = None

    def _cleanup_tmp(self):
        """Hapus tempfile sebelumnya jika ada."""
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
            self._tmp_path = None

    # ------------------------------------------------------------------ #
    #  Loading                                                            #
    # ------------------------------------------------------------------ #

    def load_edf(self, file_source):
        """Load file EDF dari path string atau file-like object (upload)."""
        try:
            self._cleanup_tmp()
            if isinstance(file_source, (str, os.PathLike)):
                self.raw = mne.io.read_raw_edf(str(file_source), preload=True, verbose=False)
            else:
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    tmp.write(file_source.read())
                    self._tmp_path = tmp.name
                self.raw = mne.io.read_raw_edf(self._tmp_path, preload=True, verbose=False)

            self.raw_original = self.raw.copy()
            self.sfreq = self.raw.info["sfreq"]
            self.channel_names = self.raw.ch_names
            self.processing_log = ["File EDF berhasil dimuat"]
            return self.get_raw_info()
        except Exception as exc:
            raise RuntimeError(f"Gagal memuat file EDF: {exc}") from exc

    @staticmethod
    def list_edf_in_zip(zip_buffer):
        """Temukan semua file EDF di dalam arsip ZIP."""
        edf_files = []
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".edf"):
                    edf_files.append(name)
        return sorted(edf_files)

    def load_edf_from_zip(self, zip_buffer, edf_path_in_zip):
        """Ekstrak satu file EDF dari ZIP lalu load."""
        self._cleanup_tmp()
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            with zf.open(edf_path_in_zip) as edf_file:
                data = edf_file.read()
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(data)
            self._tmp_path = tmp.name
        self.raw = mne.io.read_raw_edf(self._tmp_path, preload=True, verbose=False)

        self.raw_original = self.raw.copy()
        self.sfreq = self.raw.info["sfreq"]
        self.channel_names = self.raw.ch_names
        self.processing_log = [f"File EDF dimuat dari ZIP: {edf_path_in_zip}"]
        return self.get_raw_info()

    # ------------------------------------------------------------------ #
    #  Info & state                                                       #
    # ------------------------------------------------------------------ #

    def get_raw_info(self):
        """Return dict ringkasan raw data."""
        if self.raw is None:
            return {}
        annotations = self.raw.annotations
        return {
            "channels": list(self.raw.ch_names),
            "n_channels": len(self.raw.ch_names),
            "sfreq": self.raw.info["sfreq"],
            "duration_s": self.raw.times[-1],
            "n_annotations": len(annotations),
            "annotations": [
                {"onset": a["onset"], "duration": a["duration"], "description": a["description"]}
                for a in annotations
            ],
        }

    def get_processing_log(self):
        return list(self.processing_log)

    def get_task_list(self):
        """Return daftar nama task unik dari annotations."""
        if self.raw is None:
            return []
        descs = [a["description"] for a in self.raw.annotations]
        # Normalisasi nama marker
        descs = ["Think_Acting" if d == "Thinking and Acting" else d for d in descs]
        return sorted(set(descs))

    # ------------------------------------------------------------------ #
    #  Filtering                                                          #
    # ------------------------------------------------------------------ #

    def pick_channels(self, channels):
        """Pilih subset channel dari raw data."""
        if self.raw is None:
            raise RuntimeError("Data belum dimuat.")
        available = [ch for ch in channels if ch in self.raw.ch_names]
        if not available:
            raise ValueError("Tidak ada channel yang valid.")
        self.raw.pick_channels(available)
        self.processing_log.append(f"Channel dipilih: {available}")

    def apply_notch(self, freq=50.0, quality=30.0):
        """Terapkan notch filter untuk menghapus noise powerline."""
        if self.raw is None:
            raise RuntimeError("Data belum dimuat.")
        self.raw.notch_filter(freqs=freq, verbose=False)
        self.processing_log.append(f"Notch filter: {freq} Hz")

    def apply_bandpass(self, low_freq, high_freq, order=5):
        """Terapkan bandpass filter pada raw data."""
        if self.raw is None:
            raise RuntimeError("Data belum dimuat.")
        self.raw.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
        self.processing_log.append(f"Bandpass: {low_freq}-{high_freq} Hz")

    def apply_ica(self, n_components=None, method="fastica", random_state=42):
        """Jalankan ICA dan hapus komponen artefak otomatis."""
        if self.raw is None:
            raise RuntimeError("Data belum dimuat.")
        ica = ICA(n_components=n_components, method=method, random_state=random_state)
        ica.fit(self.raw, verbose=False)
        try:
            eog_indices, _ = ica.find_bads_eog(self.raw, verbose=False)
            if eog_indices:
                ica.exclude = eog_indices
        except Exception:
            pass
        ica.apply(self.raw, verbose=False)
        n_excluded = len(ica.exclude)
        self.processing_log.append(
            f"ICA (metode={method}, komponen={ica.n_components_}, artefak={n_excluded})"
        )

    # ------------------------------------------------------------------ #
    #  Data extraction                                                    #
    # ------------------------------------------------------------------ #

    def extract_dataframe(self):
        """Konversi raw data ke DataFrame dengan kolom waktu, channel, dan marker."""
        if self.raw is None:
            raise RuntimeError("Data belum dimuat.")

        data, times = self.raw.get_data(return_times=True)
        ch_names = self.raw.ch_names

        df = pd.DataFrame(data.T, columns=ch_names)
        df.insert(0, "time", times)
        df["marker"] = "none"

        annotations = self.raw.annotations
        if len(annotations) > 0:
            for annot in annotations:
                onset = annot["onset"]
                duration = annot["duration"]
                desc = annot["description"]
                if duration > 0:
                    mask = (df["time"] >= onset) & (df["time"] < onset + duration)
                else:
                    idx = np.argmin(np.abs(df["time"].values - onset))
                    mask = df.index == idx
                df.loc[mask, "marker"] = desc

        # Normalisasi nama marker
        df["marker"] = df["marker"].replace("Thinking and Acting", "Think_Acting")

        self.processing_log.append("DataFrame diekstrak")
        return df

    def extract_task_segments(self, df, task_name):
        """Filter DataFrame ke segment yang berisi task tertentu.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame hasil extract_dataframe().
        task_name : str
            Nama task/marker untuk difilter.

        Returns
        -------
        pd.DataFrame  Baris yang marker-nya sesuai task_name.
        """
        return df[df["marker"] == task_name].copy()

    def get_task_summary(self, df):
        """Hitung statistik ringkasan per task.

        Returns
        -------
        pd.DataFrame  Kolom: task, jumlah_sample, durasi_s, persen_total.
        """
        total = len(df)
        sfreq = self.sfreq or 1.0
        tasks = df["marker"].value_counts()
        rows = []
        for task_name, count in tasks.items():
            rows.append({
                "task": task_name,
                "jumlah_sample": int(count),
                "durasi_s": round(count / sfreq, 2),
                "persen_total": round(100.0 * count / total, 1),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Subband & fitur                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bandpass_array(data, sfreq, low, high, order=5):
        """Bandpass filter pada array numpy 1-D."""
        nyq = 0.5 * sfreq
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, data)

    def compute_subband_features(self, df, channels, subbands=None, features=None):
        """Hitung fitur statistik per channel per subband."""
        if subbands is None:
            subbands = DEFAULT_SUBBANDS
        if features is None:
            features = DEFAULT_FEATURES

        sfreq = self.sfreq
        rows = []

        for ch in channels:
            if ch not in df.columns:
                continue
            signal = df[ch].values
            for sb_name, (low, high) in subbands.items():
                filtered = self._bandpass_array(signal, sfreq, low, high)
                row = {"channel": ch, "subband": sb_name}
                for feat in features:
                    if feat == "mav":
                        row[feat] = float(np.mean(np.abs(filtered)))
                    elif feat == "variance":
                        row[feat] = float(np.var(filtered))
                    elif feat == "std":
                        row[feat] = float(np.std(filtered))
                    elif feat == "rms":
                        row[feat] = float(np.sqrt(np.mean(filtered ** 2)))
                rows.append(row)

        return pd.DataFrame(rows)

    def compute_task_features(self, df, channels, tasks, subbands=None, features=None):
        """Hitung fitur per task per channel per subband.

        Returns
        -------
        pd.DataFrame  Kolom: task, channel, subband, + fitur.
        """
        if subbands is None:
            subbands = DEFAULT_SUBBANDS
        if features is None:
            features = DEFAULT_FEATURES

        all_rows = []
        for task in tasks:
            seg = self.extract_task_segments(df, task)
            if seg.empty:
                continue
            feat_df = self.compute_subband_features(seg, channels, subbands, features)
            feat_df.insert(0, "task", task)
            all_rows.append(feat_df)

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Batch processing (ZIP)                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_category(edf_path_in_zip):
        """Deteksi kategori (ALS/Normal) dan metadata dari path di ZIP.

        Mendukung berbagai kedalaman folder:
          ALS05/time5/scenario4/EEG.edf
          EEGET-ALS Dataset/ALS05/time5/scenario4/EEG.edf
          id9/time1/scenario4/EEG.edf
          root_folder/id9/time1/scenario4/EEG.edf

        Returns
        -------
        dict  {"category", "subject", "time", "scenario"}
        """
        import re
        # Normalize separators
        parts = edf_path_in_zip.replace("\\", "/").split("/")
        # Remove empty parts and the filename itself
        parts = [p for p in parts if p and not p.lower().endswith(".edf")]

        # Cari subject folder: folder yang cocok ALS* atau id*
        subject = "unknown"
        category = "Normal"
        time_val = ""
        scenario_val = ""

        for p in parts:
            p_upper = p.upper()
            p_lower = p.lower()

            # Deteksi subject: ALS01, ALS02, ... atau id1, id2, ...
            if re.match(r'^ALS\d*', p_upper):
                subject = p
                category = "ALS"
            elif re.match(r'^id\d+$', p_lower) and subject == "unknown":
                subject = p
                category = "Normal"
            elif p_lower.startswith("time"):
                time_val = p
            elif p_lower.startswith("scenario"):
                scenario_val = p

        return {
            "category": category,
            "subject": subject,
            "time": time_val,
            "scenario": scenario_val,
        }

    @staticmethod
    def _process_single_edf(zip_bytes, edf_path, channels, subbands, features):
        """Worker: proses satu file EDF dari ZIP bytes. Thread-safe."""
        meta = EEGProcessor._detect_category(edf_path)
        proc = EEGProcessor()

        try:
            buf = io.BytesIO(zip_bytes)
            proc.load_edf_from_zip(buf, edf_path)
        except Exception:
            return None, []

        ch_list = channels if channels else proc.channel_names
        ch_list = [c for c in ch_list if c in proc.channel_names]
        if not ch_list:
            proc._cleanup_tmp()
            return None, []

        try:
            low_all = min(v[0] for v in subbands.values())
            high_all = max(v[1] for v in subbands.values())
            proc.apply_bandpass(low_all, high_all)
        except Exception:
            pass

        df = proc.extract_dataframe()
        tasks_found = [t for t in proc.get_task_list() if t != "none"]

        if not tasks_found:
            proc._cleanup_tmp()
            return None, tasks_found

        feat_df = proc.compute_task_features(df, ch_list, tasks_found,
                                              subbands, features)
        proc._cleanup_tmp()

        if feat_df.empty:
            return None, tasks_found

        feat_df.insert(0, "filename", edf_path)
        feat_df.insert(1, "category", meta["category"])
        feat_df.insert(2, "subject", meta["subject"])
        feat_df.insert(3, "time", meta["time"])
        feat_df.insert(4, "scenario", meta["scenario"])

        return feat_df, tasks_found

    @staticmethod
    def process_batch_zip(zip_buffer, channels=None, subbands=None,
                          features=None, progress_cb=None, max_workers=None):
        """Proses semua file EDF dalam ZIP secara paralel (multithreaded).

        Parameters
        ----------
        zip_buffer : file-like
            Buffer ZIP yang berisi file EDF.
        channels : list[str] | None
            Jika None, gunakan semua channel yang tersedia.
        subbands, features : lihat compute_subband_features.
        progress_cb : callable(current, total, filename) | None
            Callback untuk update progress bar.
        max_workers : int | None
            Jumlah thread paralel. Default: min(8, cpu_count).

        Returns
        -------
        pd.DataFrame  Kolom: filename, category, subject, time, scenario,
                       task, channel, subband, + fitur.
        list[str]  Daftar task unik yang ditemukan di seluruh dataset.
        """
        from config import DEFAULT_SUBBANDS, DEFAULT_FEATURES
        if subbands is None:
            subbands = DEFAULT_SUBBANDS
        if features is None:
            features = DEFAULT_FEATURES
        if max_workers is None:
            max_workers = min(8, os.cpu_count() or 4)

        # Baca ZIP ke bytes sekali untuk thread-safe sharing
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.read()

        edf_list = EEGProcessor.list_edf_in_zip(io.BytesIO(zip_bytes))
        total = len(edf_list)
        all_frames = []
        all_tasks_per_file = []

        # Thread-safe counter untuk progress
        counter_lock = threading.Lock()
        counter = [0]

        def _on_done(edf_path):
            with counter_lock:
                counter[0] += 1
                if progress_cb:
                    progress_cb(counter[0], total, os.path.basename(edf_path))

        # Submit semua file ke thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {}
            for edf_path in edf_list:
                fut = executor.submit(
                    EEGProcessor._process_single_edf,
                    zip_bytes, edf_path, channels, subbands, features,
                )
                future_to_path[fut] = edf_path

            for future in as_completed(future_to_path):
                edf_path = future_to_path[future]
                _on_done(edf_path)
                try:
                    feat_df, tasks_found = future.result()
                    if tasks_found:
                        cat = EEGProcessor._detect_category(edf_path).get(
                            "category", "Unknown"
                        )
                        all_tasks_per_file.append((cat, set(tasks_found)))
                    if feat_df is not None:
                        all_frames.append(feat_df)
                except Exception:
                    continue

        if progress_cb:
            progress_cb(total, total, "Selesai")

        if all_frames:
            batch_df = pd.concat(all_frames, ignore_index=True)
        else:
            batch_df = pd.DataFrame()

        # Irisan task: union per kategori, lalu intersect antar kategori
        if all_tasks_per_file:
            from collections import defaultdict
            tasks_by_cat = defaultdict(set)
            for cat, tset in all_tasks_per_file:
                tasks_by_cat[cat].update(tset)  # union dalam kategori
            if len(tasks_by_cat) > 1:
                common_tasks = set.intersection(*tasks_by_cat.values())
            else:
                # Hanya 1 kategori → ambil semua task-nya
                common_tasks = list(tasks_by_cat.values())[0]
        else:
            common_tasks = set()

        return batch_df, sorted(common_tasks)

    @staticmethod
    def calculate_task_delta(batch_df, task_a, task_b, feature_cols=None):
        """Hitung delta (task_a - task_b) per filename/channel/subband.

        Parameters
        ----------
        batch_df : pd.DataFrame
            Hasil dari process_batch_zip.
        task_a, task_b : str
            Nama task yang akan dibandingkan.
        feature_cols : list[str] | None
            Kolom fitur yang akan dihitung delta-nya. Default semua fitur numerik.

        Returns
        -------
        pd.DataFrame  Kolom: filename, channel, subband, delta_{feat}, ...
        pd.DataFrame  Statistik agregat: channel, subband, mean_delta_{feat}, std_delta_{feat}, ...
        """
        if batch_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if feature_cols is None:
            exclude = {"filename", "task", "channel", "subband"}
            feature_cols = [c for c in batch_df.columns if c not in exclude]

        df_a = batch_df[batch_df["task"] == task_a].copy()
        df_b = batch_df[batch_df["task"] == task_b].copy()

        if df_a.empty or df_b.empty:
            return pd.DataFrame(), pd.DataFrame()

        merge_keys = ["filename", "channel", "subband"]
        merged = pd.merge(
            df_a[merge_keys + feature_cols],
            df_b[merge_keys + feature_cols],
            on=merge_keys, suffixes=("_a", "_b"),
        )

        delta_df = merged[merge_keys].copy()
        for feat in feature_cols:
            delta_df[f"{feat}_{task_a}"] = merged[f"{feat}_a"].values
            delta_df[f"{feat}_{task_b}"] = merged[f"{feat}_b"].values
            delta_df[f"delta_{feat}"] = merged[f"{feat}_a"] - merged[f"{feat}_b"]

        # Agregasi statistik per channel/subband (rata-rata di semua file)
        delta_cols = [c for c in delta_df.columns if c.startswith("delta_")]
        agg_dict = {}
        for dc in delta_cols:
            agg_dict[dc] = ["mean", "std"]

        agg_df = delta_df.groupby(["channel", "subband"]).agg(agg_dict)
        agg_df.columns = [f"{stat}_{col}" for col, stat in agg_df.columns]
        agg_df = agg_df.reset_index()

        return delta_df, agg_df

    # ------------------------------------------------------------------ #
    #  Normalisasi & Perbandingan ALS vs Normal                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_per_subject(batch_df, feature_cols, method="zscore",
                               scope="subject"):
        """Normalisasi fitur per subjek.

        Parameters
        ----------
        batch_df : pd.DataFrame
            DataFrame batch dengan kolom 'subject'.
        feature_cols : list[str]
            Kolom fitur yang akan dinormalisasi.
        method : str
            'zscore' (default) atau 'minmax'.
        scope : str
            'subject' — normalisasi per orang (semua scenario digabung).
            'subject_scenario' — normalisasi per orang per skenario.

        Returns
        -------
        pd.DataFrame  DataFrame baru dengan fitur yang sudah dinormalisasi.
        """
        if batch_df.empty or "subject" not in batch_df.columns:
            return batch_df.copy()

        df = batch_df.copy()

        if scope == "subject_scenario" and "scenario" in df.columns:
            group_keys = ["subject", "scenario"]
        else:
            group_keys = ["subject"]

        for _, idx in df.groupby(group_keys).groups.items():
            grp_data = df.loc[idx, feature_cols]
            if method == "zscore":
                mu = grp_data.mean()
                sigma = grp_data.std().replace(0, 1)
                df.loc[idx, feature_cols] = (grp_data - mu) / sigma
            elif method == "minmax":
                mn = grp_data.min()
                mx = grp_data.max()
                rng = (mx - mn).replace(0, 1)
                df.loc[idx, feature_cols] = (grp_data - mn) / rng

        return df

    @staticmethod
    def compare_als_vs_normal(batch_df, active_task, baseline_task="Resting",
                               feature_cols=None, mode="delta",
                               apply_fdr=True, compute_effect_size=True):
        """Bandingkan fitur antara ALS dan Normal.

        Parameters
        ----------
        batch_df : pd.DataFrame
            DataFrame batch dengan kolom 'category', 'subject', dll.
        active_task : str
            Task aktif (misal 'Thinking').
        baseline_task : str
            Task baseline (default 'Resting').
        feature_cols : list[str] | None
            Kolom fitur.
        mode : str
            'delta'  — bandingkan delta (active − baseline).
            'zscore' — bandingkan fitur z-scored langsung (hanya active_task).
            'both'   — z-score dulu, lalu hitung delta.
        apply_fdr : bool
            Jika True, tambahkan kolom p-value yang dikoreksi FDR
            (Benjamini-Hochberg).
        compute_effect_size : bool
            Jika True, tambahkan kolom Cohen's d.

        Returns
        -------
        compare_df : pd.DataFrame
            Data per subject/channel/subband dengan kolom 'category'.
            Kolom fitur diawali 'delta_' (mode delta/both) atau asli (zscore).
        stats_df : pd.DataFrame
            Statistik: channel, subband, mean_als, mean_normal, p_value,
            p_fdr, cohend per fitur.
        """
        from scipy.stats import mannwhitneyu

        if batch_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        meta_cols = {"filename", "category", "subject", "time", "scenario",
                     "task", "channel", "subband"}
        if feature_cols is None:
            feature_cols = [c for c in batch_df.columns if c not in meta_cols]

        group_keys = ["category", "subject", "channel", "subband"]

        if mode == "zscore":
            # --- Z-score only: bandingkan fitur active task langsung ---
            df_task = batch_df[batch_df["task"] == active_task].copy()
            if df_task.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Z-score per subject
            df_task = EEGProcessor.normalize_per_subject(
                df_task, feature_cols, method="zscore", scope="subject"
            )
            compare_df = df_task.groupby(group_keys)[feature_cols].mean().reset_index()
            value_cols = feature_cols

        elif mode == "both":
            # --- Z-score + Delta ---
            df_two = batch_df[
                batch_df["task"].isin([active_task, baseline_task])
            ].copy()
            if df_two.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Z-score per subject (across both tasks for fair scaling)
            df_two = EEGProcessor.normalize_per_subject(
                df_two, feature_cols, method="zscore", scope="subject"
            )

            df_active = df_two[df_two["task"] == active_task]
            df_base = df_two[df_two["task"] == baseline_task]
            if df_active.empty or df_base.empty:
                return pd.DataFrame(), pd.DataFrame()

            agg_active = df_active.groupby(group_keys)[feature_cols].mean().reset_index()
            agg_base = df_base.groupby(group_keys)[feature_cols].mean().reset_index()

            merged = pd.merge(agg_active, agg_base,
                              on=group_keys, suffixes=("_act", "_base"))
            compare_df = merged[group_keys].copy()
            for feat in feature_cols:
                compare_df[f"delta_{feat}"] = (
                    merged[f"{feat}_act"] - merged[f"{feat}_base"]
                )
            value_cols = [f"delta_{feat}" for feat in feature_cols]

        else:
            # --- Delta only (original) ---
            df_active = batch_df[batch_df["task"] == active_task].copy()
            df_base = batch_df[batch_df["task"] == baseline_task].copy()
            if df_active.empty or df_base.empty:
                return pd.DataFrame(), pd.DataFrame()

            agg_active = df_active.groupby(group_keys)[feature_cols].mean().reset_index()
            agg_base = df_base.groupby(group_keys)[feature_cols].mean().reset_index()

            merged = pd.merge(agg_active, agg_base,
                              on=group_keys, suffixes=("_act", "_base"))
            compare_df = merged[group_keys].copy()
            for feat in feature_cols:
                compare_df[f"delta_{feat}"] = (
                    merged[f"{feat}_act"] - merged[f"{feat}_base"]
                )
            value_cols = [f"delta_{feat}" for feat in feature_cols]

        # ---- Statistik ALS vs Normal per channel/subband ----
        stats_rows = []
        for (ch, sb), grp in compare_df.groupby(["channel", "subband"]):
            als_data = grp[grp["category"] == "ALS"]
            norm_data = grp[grp["category"] == "Normal"]

            row = {"channel": ch, "subband": sb}
            for vc in value_cols:
                feat_name = vc.replace("delta_", "") if vc.startswith("delta_") else vc
                als_vals = als_data[vc].dropna()
                norm_vals = norm_data[vc].dropna()

                row[f"mean_als_{feat_name}"] = (
                    als_vals.mean() if len(als_vals) else np.nan
                )
                row[f"mean_normal_{feat_name}"] = (
                    norm_vals.mean() if len(norm_vals) else np.nan
                )

                # Mann-Whitney U test
                if len(als_vals) >= 2 and len(norm_vals) >= 2:
                    try:
                        _, p = mannwhitneyu(als_vals, norm_vals,
                                            alternative="two-sided")
                        row[f"p_{feat_name}"] = p
                    except Exception:
                        row[f"p_{feat_name}"] = np.nan
                else:
                    row[f"p_{feat_name}"] = np.nan

                # Cohen's d effect size
                if compute_effect_size and len(als_vals) >= 2 and len(norm_vals) >= 2:
                    m1, m2 = als_vals.mean(), norm_vals.mean()
                    s1, s2 = als_vals.std(ddof=1), norm_vals.std(ddof=1)
                    n1, n2 = len(als_vals), len(norm_vals)
                    # Pooled standard deviation
                    pooled = np.sqrt(
                        ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
                    )
                    row[f"cohend_{feat_name}"] = (
                        (m1 - m2) / pooled if pooled > 0 else 0.0
                    )
                elif compute_effect_size:
                    row[f"cohend_{feat_name}"] = np.nan

            stats_rows.append(row)

        stats_df = pd.DataFrame(stats_rows)

        # ---- FDR correction (Benjamini-Hochberg) per fitur ----
        if apply_fdr and not stats_df.empty:
            from scipy.stats import false_discovery_control
            p_cols = [c for c in stats_df.columns if c.startswith("p_")]
            for pc in p_cols:
                raw_p = stats_df[pc].values.copy()
                valid_mask = ~np.isnan(raw_p)
                if valid_mask.sum() >= 2:
                    try:
                        adjusted = np.full_like(raw_p, np.nan)
                        # Benjamini-Hochberg
                        adjusted[valid_mask] = _benjamini_hochberg(
                            raw_p[valid_mask]
                        )
                        fdr_col = pc.replace("p_", "p_fdr_")
                        stats_df[fdr_col] = adjusted
                    except Exception:
                        pass

        return compare_df, stats_df
