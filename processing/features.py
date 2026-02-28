"""
Modul features — Ekstraksi fitur EEG per channel/subband.

Fitur baru dari pipeline EEG-ALS- (03_extract_features.py):
- Band power (absolute)
- Relative power (% terhadap total)
- Peak frequency
- Rasio antar subband (alpha/beta, theta/alpha, delta/theta)
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from config import DEFAULT_SUBBANDS, DEFAULT_FEATURES, BAND_RATIOS


def _bandpass_array(data, sfreq, low, high, order=5):
    """Bandpass filter pada array numpy 1-D."""
    nyq = 0.5 * sfreq
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, data)


class EEGFeatures:
    """Kumpulan metode untuk ekstraksi fitur EEG."""

    # ------------------------------------------------------------------ #
    #  Band power (BARU – dari pipeline)                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_band_power(signal, sfreq, low, high):
        """Hitung band power absolut menggunakan FFT.

        Dari pipeline EEG-ALS-/03_extract_features.py.

        Parameters
        ----------
        signal : np.ndarray
            Sinyal 1-D.
        sfreq : float
            Sampling frequency.
        low, high : float
            Batas frekuensi subband.

        Returns
        -------
        float  Absolute band power (µV²/Hz).
        """
        n = len(signal)
        if n < 4:
            return 0.0
        fft_vals = np.fft.rfft(signal)
        power_spectrum = (np.abs(fft_vals) ** 2) / n
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        band_mask = (freqs >= low) & (freqs <= high)
        return float(np.mean(power_spectrum[band_mask])) if band_mask.any() else 0.0

    @staticmethod
    def compute_relative_power(signal, sfreq, low, high):
        """Hitung relative power (% total power).

        Dari pipeline EEG-ALS-/03_extract_features.py.

        Returns
        -------
        float  Relative power (0–1).
        """
        n = len(signal)
        if n < 4:
            return 0.0
        fft_vals = np.fft.rfft(signal)
        power_spectrum = (np.abs(fft_vals) ** 2) / n
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        band_mask = (freqs >= low) & (freqs <= high)

        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0.0
        band_power = np.sum(power_spectrum[band_mask])
        return float(band_power / total_power)

    @staticmethod
    def compute_peak_frequency(signal, sfreq, low, high):
        """Hitung peak frequency dalam subband.

        Dari pipeline EEG-ALS-/03_extract_features.py.

        Returns
        -------
        float  Frekuensi puncak (Hz) dalam subband.
        """
        n = len(signal)
        if n < 4:
            return 0.0
        fft_vals = np.fft.rfft(signal)
        power_spectrum = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        band_mask = (freqs >= low) & (freqs <= high)

        if not band_mask.any():
            return 0.0
        band_power = power_spectrum[band_mask]
        band_freqs = freqs[band_mask]
        return float(band_freqs[np.argmax(band_power)])

    # ------------------------------------------------------------------ #
    #  Fitur per subband                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_subband_features(df, channels, sfreq, subbands=None,
                                  features=None, include_frequency=True):
        """Hitung fitur per channel per subband.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame sinyal (kolom per channel).
        channels : list[str]
            Nama channel.
        sfreq : float
            Sampling frequency.
        subbands : dict | None
            Subband: {"Delta": (0.5, 4), ...}. Default dari config.
        features : list[str] | None
            Fitur time-domain: ["mav", "variance", "std", "rms"].
        include_frequency : bool
            Jika True, hitung juga band_power, relative_power, peak_frequency.

        Returns
        -------
        pd.DataFrame  Kolom: channel, subband, + fitur.
        """
        if subbands is None:
            subbands = DEFAULT_SUBBANDS
        if features is None:
            features = DEFAULT_FEATURES

        rows = []

        for ch in channels:
            if ch not in df.columns:
                continue
            signal = df[ch].values

            for sb_name, (low, high) in subbands.items():
                filtered = _bandpass_array(signal, sfreq, low, high)
                row = {"channel": ch, "subband": sb_name}

                # --- Time-domain features ---
                for feat in features:
                    if feat == "mav":
                        row[feat] = float(np.mean(np.abs(filtered)))
                    elif feat == "variance":
                        row[feat] = float(np.var(filtered))
                    elif feat == "std":
                        row[feat] = float(np.std(filtered))

                # --- Frequency-domain features (BARU) ---
                if include_frequency:
                    row["band_power"] = EEGFeatures.compute_band_power(
                        signal, sfreq, low, high
                    )
                    row["relative_power"] = EEGFeatures.compute_relative_power(
                        signal, sfreq, low, high
                    )
                    row["peak_frequency"] = EEGFeatures.compute_peak_frequency(
                        signal, sfreq, low, high
                    )

                rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Fitur per task                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_task_features(loader, df, channels, tasks, subbands=None,
                               features=None, include_frequency=True):
        """Hitung fitur per task per channel per subband.

        Parameters
        ----------
        loader : EEGLoader
            Loader instance (untuk extract_task_segments).
        df : pd.DataFrame
        channels : list[str]
        tasks : list[str]
        subbands, features : lihat compute_subband_features.
        include_frequency : bool

        Returns
        -------
        pd.DataFrame  Kolom: task, channel, subband, + fitur.
        """
        all_rows = []
        for task in tasks:
            seg = loader.extract_task_segments(df, task)
            if seg.empty:
                continue
            feat_df = EEGFeatures.compute_subband_features(
                seg, channels, loader.sfreq, subbands, features,
                include_frequency=include_frequency,
            )
            feat_df.insert(0, "task", task)
            all_rows.append(feat_df)

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Fitur per occurrence (BARU)                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_occurrence_features(loader, df, channels, tasks,
                                     subbands=None, features=None,
                                     include_frequency=True):
        """Hitung fitur per occurrence per task per channel per subband.

        Setiap occurrence diberi label task_occ (misal 'Resting_1').

        Parameters
        ----------
        loader : EEGLoader
        df : pd.DataFrame
        channels, tasks, subbands, features, include_frequency: sama.

        Returns
        -------
        pd.DataFrame  Kolom: task, occurrence, task_occ, channel, subband, + fitur.
        """
        occurrences = loader.get_task_occurrences()
        if not occurrences:
            return pd.DataFrame()

        all_rows = []
        for occ in occurrences:
            task_name = occ["task"]
            occ_num = occ["occurrence"]

            if task_name not in tasks:
                continue

            seg = loader.extract_occurrence_segment(df, task_name, occ_num)
            if seg.empty or len(seg) < 4:
                continue

            feat_df = EEGFeatures.compute_subband_features(
                seg, channels, loader.sfreq, subbands, features,
                include_frequency=include_frequency,
            )
            if feat_df.empty:
                continue

            feat_df.insert(0, "task", task_name)
            feat_df.insert(1, "occurrence", occ_num)
            feat_df.insert(2, "task_occ", f"{task_name}_{occ_num}")
            all_rows.append(feat_df)

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    @staticmethod
    def compute_aggregated_occurrence_features(loader, df, channels, tasks,
                                                subbands=None, features=None,
                                                include_frequency=True):
        """Hitung fitur dengan rata-rata semua occurrence per task.

        Untuk setiap task: hitung fitur per occurrence dulu, lalu
        rata-rata hasilnya.  Dibandingkan dengan compute_task_features
        yang menggabungkan semua sample jadi satu segmen besar.

        Returns
        -------
        pd.DataFrame  Kolom: task, channel, subband, + fitur (mean dari occurrences).
        """
        occ_df = EEGFeatures.compute_occurrence_features(
            loader, df, channels, tasks, subbands, features,
            include_frequency=include_frequency,
        )
        if occ_df.empty:
            return pd.DataFrame()

        # Group by task + channel + subband, ambil mean dari semua occurrence
        group_cols = ["task", "channel", "subband"]
        meta_cols = {"task", "occurrence", "task_occ", "channel", "subband"}
        feat_cols = [c for c in occ_df.columns if c not in meta_cols]

        agg = occ_df.groupby(group_cols, as_index=False)[feat_cols].mean()
        return agg



    @staticmethod
    def compute_band_ratios(features_df, ratios=None):
        """Hitung rasio power antar subband.

        Dari pipeline EEG-ALS-/03_extract_features.py.

        Parameters
        ----------
        features_df : pd.DataFrame
            Hasil dari compute_subband_features (harus ada kolom band_power).
        ratios : dict | None
            Mapping nama_ratio -> (subband_atas, subband_bawah).
            Default dari config BAND_RATIOS.

        Returns
        -------
        pd.DataFrame  Kolom: channel, ratio_name, value.
        """
        if ratios is None:
            ratios = BAND_RATIOS

        if "band_power" not in features_df.columns:
            return pd.DataFrame()

        rows = []
        for ch, grp in features_df.groupby("channel"):
            power_map = dict(zip(grp["subband"], grp["band_power"]))

            for ratio_name, (sb_num, sb_den) in ratios.items():
                num = power_map.get(sb_num, 0.0)
                den = power_map.get(sb_den, 0.0)
                value = float(num / den) if den > 0 else 0.0
                rows.append({
                    "channel": ch,
                    "ratio_name": ratio_name,
                    "value": value,
                })

        return pd.DataFrame(rows)
