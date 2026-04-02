"""
Modul epoching — Epoching dan Sliding Windows untuk data EEG.

Menyediakan:
- Fixed-length epoching (potong data jadi klip-klip durasi tetap)
- Sliding window (jendela geser dengan overlap)
- Epoch rejection (buang epoch yang mengandung artefak)
- Komputasi fitur per epoch / per window
"""

import numpy as np
import pandas as pd

from config import (
    DEFAULT_SUBBANDS, DEFAULT_FEATURES,
    DEFAULT_EPOCH_DURATION, DEFAULT_EPOCH_REJECT_UV,
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_OVERLAP,
)


class EpochEngine:
    """Epoching dan Sliding Windows untuk data EEG."""

    # ------------------------------------------------------------------ #
    #  Fixed-length Epoching                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_fixed_epochs(df, channels, sfreq, epoch_duration=None):
        """Potong DataFrame menjadi epoch-epoch dengan durasi tetap.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame sinyal (kolom time, channel, marker).
        channels : list[str]
            Channel yang diproses.
        sfreq : float
            Sampling frequency.
        epoch_duration : float
            Durasi per epoch dalam detik.

        Returns
        -------
        list[dict]  Setiap item: {
            "epoch_idx": int,
            "start_time": float,
            "end_time": float,
            "data": pd.DataFrame
        }
        """
        if epoch_duration is None:
            epoch_duration = DEFAULT_EPOCH_DURATION

        samples_per_epoch = int(epoch_duration * sfreq)
        if samples_per_epoch < 4:
            return []

        n_samples = len(df)
        epochs = []
        epoch_idx = 0

        for start in range(0, n_samples, samples_per_epoch):
            end = start + samples_per_epoch
            if end > n_samples:
                break  # Buang sisa yang kurang dari 1 epoch penuh

            epoch_df = df.iloc[start:end].copy()
            start_time = float(epoch_df["time"].iloc[0]) if "time" in epoch_df.columns else start / sfreq
            end_time = float(epoch_df["time"].iloc[-1]) if "time" in epoch_df.columns else end / sfreq

            epochs.append({
                "epoch_idx": epoch_idx,
                "start_time": start_time,
                "end_time": end_time,
                "data": epoch_df,
            })
            epoch_idx += 1

        return epochs

    # ------------------------------------------------------------------ #
    #  Sliding Windows                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_sliding_windows(df, channels, sfreq,
                                window_size=None, overlap=None):
        """Potong DataFrame menggunakan sliding window.

        Parameters
        ----------
        df : pd.DataFrame
        channels : list[str]
        sfreq : float
        window_size : float
            Lebar jendela dalam detik.
        overlap : float
            Rasio overlap (0.0 – 0.75). Misal 0.5 = 50% overlap.

        Returns
        -------
        list[dict]  Setiap item: {
            "window_idx": int,
            "start_time": float,
            "end_time": float,
            "data": pd.DataFrame
        }
        """
        if window_size is None:
            window_size = DEFAULT_WINDOW_SIZE
        if overlap is None:
            overlap = DEFAULT_WINDOW_OVERLAP

        samples_per_window = int(window_size * sfreq)
        step_samples = int(samples_per_window * (1.0 - overlap))

        if samples_per_window < 4 or step_samples < 1:
            return []

        n_samples = len(df)
        windows = []
        window_idx = 0

        start = 0
        while start + samples_per_window <= n_samples:
            end = start + samples_per_window
            win_df = df.iloc[start:end].copy()

            start_time = float(win_df["time"].iloc[0]) if "time" in win_df.columns else start / sfreq
            end_time = float(win_df["time"].iloc[-1]) if "time" in win_df.columns else end / sfreq

            windows.append({
                "window_idx": window_idx,
                "start_time": start_time,
                "end_time": end_time,
                "data": win_df,
            })
            window_idx += 1
            start += step_samples

        return windows

    # ------------------------------------------------------------------ #
    #  Per-task Epoching & Sliding Windows                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_task_epochs(loader, df, channels, sfreq,
                           tasks, epoch_duration=None):
        """Epoching per task: potong setiap segmen task menjadi epoch.

        Returns
        -------
        dict  task_name -> list[dict] (sama format create_fixed_epochs)
        """
        result = {}
        for task_name in tasks:
            seg = loader.extract_task_segments(df, task_name)
            if seg.empty:
                continue
            epochs = EpochEngine.create_fixed_epochs(
                seg, channels, sfreq, epoch_duration
            )
            if epochs:
                result[task_name] = epochs
        return result

    @staticmethod
    def create_task_sliding_windows(loader, df, channels, sfreq,
                                    tasks, window_size=None, overlap=None):
        """Sliding window per task.

        Returns
        -------
        dict  task_name -> list[dict] (sama format create_sliding_windows)
        """
        result = {}
        for task_name in tasks:
            seg = loader.extract_task_segments(df, task_name)
            if seg.empty:
                continue
            windows = EpochEngine.create_sliding_windows(
                seg, channels, sfreq, window_size, overlap
            )
            if windows:
                result[task_name] = windows
        return result

    # ------------------------------------------------------------------ #
    #  Epoch Rejection                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def reject_bad_epochs(epochs, channels, threshold_uv=None, sfreq=None):
        """Buang epoch yang amplitude-nya melebihi threshold.

        Epoch dianggap buruk jika ada satu sampel pun di channel mana pun
        yang amplitude absolutnya melebihi threshold.

        Parameters
        ----------
        epochs : list[dict]
            Epoch dari create_fixed_epochs atau create_sliding_windows.
        channels : list[str]
            Channel yang dicek.
        threshold_uv : float
            Threshold dalam µV. None = default dari config.
        sfreq : float | None
            Sampling frequency (untuk info log). Tidak dipakai langsung.

        Returns
        -------
        clean_epochs : list[dict]
            Epoch yang lolos rejection.
        n_rejected : int
            Jumlah epoch yang dibuang.
        """
        if threshold_uv is None:
            threshold_uv = DEFAULT_EPOCH_REJECT_UV

        # MNE menyimpan data dalam Volt, konversi threshold µV → V
        threshold_v = threshold_uv * 1e-6

        clean = []
        n_rejected = 0

        for epoch in epochs:
            data = epoch["data"]
            ch_cols = [ch for ch in channels if ch in data.columns]
            if not ch_cols:
                clean.append(epoch)
                continue

            # Cek apakah ada nilai yang melebihi threshold
            max_amp = data[ch_cols].abs().max().max()
            if max_amp > threshold_v:
                n_rejected += 1
            else:
                clean.append(epoch)

        return clean, n_rejected

    # ------------------------------------------------------------------ #
    #  Compute features per epoch (averaged)                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_epoch_features(epochs, channels, sfreq, subbands=None,
                               features=None, include_frequency=True,
                               psd_method="welch", psd_fmin=0.0,
                               psd_fmax=49.0, psd_n_fft=None):
        """Hitung fitur per epoch, lalu rata-ratakan.

        Parameters
        ----------
        epochs : list[dict]
            Epoch dari create_fixed_epochs (sudah di-reject jika perlu).
        channels : list[str]
        sfreq : float
        subbands, features, include_frequency : lihat EEGFeatures.
        psd_method, psd_fmin, psd_fmax, psd_n_fft : parameter PSD.

        Returns
        -------
        pd.DataFrame
            Kolom: channel, subband, + fitur (mean dari semua epoch),
            + fitur_std (std dari semua epoch), n_epochs.
        """
        from processing.features import EEGFeatures

        if subbands is None:
            subbands = DEFAULT_SUBBANDS
        if features is None:
            features = DEFAULT_FEATURES

        if not epochs:
            return pd.DataFrame()

        # Hitung fitur per epoch
        epoch_dfs = []
        for epoch in epochs:
            feat_df = EEGFeatures.compute_subband_features(
                epoch["data"], channels, sfreq, subbands, features,
                include_frequency=include_frequency,
                psd_method=psd_method, psd_fmin=psd_fmin,
                psd_fmax=psd_fmax, psd_n_fft=psd_n_fft,
            )
            if not feat_df.empty:
                epoch_dfs.append(feat_df)

        if not epoch_dfs:
            return pd.DataFrame()

        all_epochs = pd.concat(epoch_dfs, ignore_index=True)

        # Rata-ratakan per channel+subband
        group_cols = ["channel", "subband"]
        meta_cols = set(group_cols)
        feat_cols = [c for c in all_epochs.columns if c not in meta_cols]

        agg_dict = {}
        for fc in feat_cols:
            agg_dict[fc] = ["mean", "std"]

        agg = all_epochs.groupby(group_cols).agg(agg_dict)
        # Flatten column names
        agg.columns = [
            f"{col}" if stat == "mean" else f"{col}_std"
            for col, stat in agg.columns
        ]
        agg = agg.reset_index()
        agg["n_epochs"] = len(epoch_dfs)

        return agg

    # ------------------------------------------------------------------ #
    #  Compute features per sliding window (time-resolved)                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_windowed_features(windows, channels, sfreq, subbands=None,
                                  features=None, include_frequency=True,
                                  psd_method="welch", psd_fmin=0.0,
                                  psd_fmax=49.0, psd_n_fft=None):
        """Hitung fitur per sliding window (time-resolved).

        Parameters
        ----------
        windows : list[dict]
            Windows dari create_sliding_windows.
        channels, sfreq, subbands, features, include_frequency : sama.
        psd_method, psd_fmin, psd_fmax, psd_n_fft : parameter PSD.

        Returns
        -------
        pd.DataFrame
            Kolom: window_idx, start_time, end_time,
            channel, subband, + fitur.
        """
        from processing.features import EEGFeatures

        if subbands is None:
            subbands = DEFAULT_SUBBANDS
        if features is None:
            features = DEFAULT_FEATURES

        if not windows:
            return pd.DataFrame()

        all_rows = []
        for win in windows:
            feat_df = EEGFeatures.compute_subband_features(
                win["data"], channels, sfreq, subbands, features,
                include_frequency=include_frequency,
                psd_method=psd_method, psd_fmin=psd_fmin,
                psd_fmax=psd_fmax, psd_n_fft=psd_n_fft,
            )
            if feat_df.empty:
                continue

            feat_df.insert(0, "window_idx", win["window_idx"])
            feat_df.insert(1, "start_time", win["start_time"])
            feat_df.insert(2, "end_time", win["end_time"])
            all_rows.append(feat_df)

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Epoched task features (with rejection)                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_epoched_task_features(loader, df, channels, tasks,
                                      sfreq, subbands=None, features=None,
                                      epoch_duration=None,
                                      reject_threshold=None,
                                      include_frequency=True,
                                      psd_method="welch", psd_fmin=0.0,
                                      psd_fmax=49.0, psd_n_fft=None):
        """Hitung fitur per task menggunakan epoching.

        Alur: task segment → epoch → reject → compute per epoch → mean

        Returns
        -------
        pd.DataFrame
            Kolom: task, channel, subband, + fitur, + fitur_std,
            n_epochs, n_rejected
        """
        task_epochs = EpochEngine.create_task_epochs(
            loader, df, channels, sfreq, tasks, epoch_duration
        )

        all_rows = []
        for task_name, epochs in task_epochs.items():
            n_rejected = 0
            if reject_threshold is not None:
                epochs, n_rejected = EpochEngine.reject_bad_epochs(
                    epochs, channels, reject_threshold
                )

            if not epochs:
                continue

            feat_df = EpochEngine.compute_epoch_features(
                epochs, channels, sfreq, subbands, features,
                include_frequency=include_frequency,
                psd_method=psd_method, psd_fmin=psd_fmin,
                psd_fmax=psd_fmax, psd_n_fft=psd_n_fft,
            )
            if feat_df.empty:
                continue

            feat_df.insert(0, "task", task_name)
            feat_df["n_rejected"] = n_rejected
            all_rows.append(feat_df)

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Windowed task features (time-resolved)                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_windowed_task_features(loader, df, channels, tasks,
                                       sfreq, subbands=None, features=None,
                                       window_size=None, overlap=None,
                                       include_frequency=True,
                                       psd_method="welch", psd_fmin=0.0,
                                       psd_fmax=49.0, psd_n_fft=None):
        """Hitung fitur per task menggunakan sliding window (time-resolved).

        Returns
        -------
        pd.DataFrame
            Kolom: task, window_idx, start_time, end_time,
            channel, subband, + fitur
        """
        task_windows = EpochEngine.create_task_sliding_windows(
            loader, df, channels, sfreq, tasks, window_size, overlap
        )

        all_rows = []
        for task_name, windows in task_windows.items():
            feat_df = EpochEngine.compute_windowed_features(
                windows, channels, sfreq, subbands, features,
                include_frequency=include_frequency,
                psd_method=psd_method, psd_fmin=psd_fmin,
                psd_fmax=psd_fmax, psd_n_fft=psd_n_fft,
            )
            if feat_df.empty:
                continue

            feat_df.insert(0, "task", task_name)
            all_rows.append(feat_df)

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Aggregated windowed features (for batch)                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def aggregate_windowed_features(windowed_df):
        """Agregasi fitur windowed menjadi 1 baris per task/channel/subband.

        Digunakan untuk batch processing agar DataFrame tidak terlalu besar.

        Parameters
        ----------
        windowed_df : pd.DataFrame
            Output dari compute_windowed_task_features.

        Returns
        -------
        pd.DataFrame
            Kolom: task, channel, subband, + fitur (mean), n_windows
        """
        if windowed_df.empty:
            return pd.DataFrame()

        group_cols = ["task", "channel", "subband"]
        meta_cols = {"task", "window_idx", "start_time", "end_time",
                     "channel", "subband"}
        feat_cols = [c for c in windowed_df.columns if c not in meta_cols]

        if not feat_cols:
            return pd.DataFrame()

        agg = windowed_df.groupby(group_cols, as_index=False)[feat_cols].mean()
        # Hitung jumlah windows per group
        counts = windowed_df.groupby(group_cols).size().reset_index(name="n_windows")
        agg = agg.merge(counts, on=group_cols, how="left")

        return agg
