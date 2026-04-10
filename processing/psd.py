"""
Modul psd — Analisis Power Spectral Density (PSD).

Mendukung metode Welch dan Multitaper.
Menyediakan:
- PSD dari MNE raw object atau numpy array
- PSD per-task (dari annotations)
- Ekstraksi band power numerik dari PSD
"""

import numpy as np
import pandas as pd
from scipy.signal import welch as scipy_welch

from config import (
    DEFAULT_SUBBANDS,
    DEFAULT_PSD_METHOD,
    DEFAULT_PSD_FMIN,
    DEFAULT_PSD_FMAX,
    DEFAULT_PSD_N_FFT,
)

# Kompatibilitas untuk numpy >= 2.0 di mana np.trapz diubah menjadi np.trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

def _auto_n_fft(sfreq, n_samples, n_fft=None):
    """Tentukan n_fft otomatis.

    - Jika n_fft=None  → 2 × sfreq (resolusi 0.5 Hz)
    - n_fft tidak boleh melebihi panjang data
    - Minimal 64 agar PSD tidak terlalu kasar

    Returns
    -------
    int  Nilai n_fft yang aman.
    """
    if n_fft is None:
        n_fft = int(2 * sfreq)
    n_fft = max(64, n_fft)
    n_fft = min(n_fft, n_samples)
    return n_fft


class PSDAnalyzer:
    """Kelas untuk menghitung PSD menggunakan Welch atau Multitaper."""

    # ------------------------------------------------------------------ #
    #  Core: hitung PSD dari array numpy                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_psd_array(data, sfreq, method="welch",
                          fmin=0.0, fmax=49.0, n_fft=None):
        """Hitung PSD dari array numpy 2-D (n_channels × n_samples).

        Parameters
        ----------
        data : np.ndarray
            Sinyal EEG.  Shape (n_channels, n_samples) atau (n_samples,).
        sfreq : float
            Sampling frequency (Hz).
        method : str
            'welch' atau 'multitaper'.
        fmin, fmax : float
            Rentang frekuensi yang dikembalikan.
        n_fft : int | None
            Panjang FFT.  None = otomatis (2 × sfreq).

        Returns
        -------
        psds : np.ndarray  Shape (n_channels, n_freqs).
        freqs : np.ndarray  Shape (n_freqs,).
        """
        if data.ndim == 1:
            data = data[np.newaxis, :]

        n_channels, n_samples = data.shape

        if n_samples < 8:
            # Data terlalu pendek
            return np.zeros((n_channels, 1)), np.array([0.0])

        n_fft_safe = _auto_n_fft(sfreq, n_samples, n_fft)

        if method == "multitaper":
            psds, freqs = PSDAnalyzer._psd_multitaper(
                data, sfreq, fmin, fmax, n_fft_safe
            )
        else:
            psds, freqs = PSDAnalyzer._psd_welch(
                data, sfreq, fmin, fmax, n_fft_safe
            )

        return psds, freqs

    # ------------------------------------------------------------------ #
    #  Welch implementation                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _psd_welch(data, sfreq, fmin, fmax, n_fft):
        """PSD via Welch's method (scipy)."""
        n_channels, n_samples = data.shape

        # nperseg = n_fft; overlap 50%
        nperseg = min(n_fft, n_samples)
        noverlap = nperseg // 2

        all_psds = []
        freqs_out = None

        for ch_idx in range(n_channels):
            freqs, psd = scipy_welch(
                data[ch_idx],
                fs=sfreq,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=n_fft,
            )
            # Crop ke fmin-fmax
            mask = (freqs >= fmin) & (freqs <= fmax)
            if freqs_out is None:
                freqs_out = freqs[mask]
            all_psds.append(psd[mask])

        if freqs_out is None:
            return np.zeros((n_channels, 1)), np.array([0.0])

        return np.array(all_psds), freqs_out

    # ------------------------------------------------------------------ #
    #  Multitaper implementation                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _psd_multitaper(data, sfreq, fmin, fmax, n_fft):
        """PSD via Multitaper (MNE)."""
        try:
            from mne.time_frequency import psd_array_multitaper

            psds, freqs = psd_array_multitaper(
                data, sfreq=sfreq,
                fmin=fmin, fmax=fmax,
                n_jobs=1, verbose=False,
            )
            return psds, freqs
        except Exception:
            # Fallback ke Welch jika Multitaper gagal
            return PSDAnalyzer._psd_welch(data, sfreq, fmin, fmax, n_fft)

    # ------------------------------------------------------------------ #
    #  PSD dari MNE raw object                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_psd_raw(raw, method="welch", fmin=0.0, fmax=49.0,
                        n_fft=None):
        """Hitung PSD dari MNE raw object.

        Parameters
        ----------
        raw : mne.io.Raw
        method, fmin, fmax, n_fft : sama dengan compute_psd_array.

        Returns
        -------
        psds : np.ndarray  (n_channels, n_freqs)
        freqs : np.ndarray (n_freqs,)
        ch_names : list[str]
        """
        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        psds, freqs = PSDAnalyzer.compute_psd_array(
            data, sfreq, method=method,
            fmin=fmin, fmax=fmax, n_fft=n_fft,
        )
        return psds, freqs, list(raw.ch_names)

    # ------------------------------------------------------------------ #
    #  PSD per-task                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_psd_per_task(loader, df, method="welch",
                             fmin=0.0, fmax=49.0, n_fft=None):
        """Hitung PSD per task dari DataFrame.

        Parameters
        ----------
        loader : EEGLoader
            Loader instance (sudah load file EDF).
        df : pd.DataFrame
            DataFrame sinyal (kolom time, channel, marker).
        method, fmin, fmax, n_fft : parameter PSD.

        Returns
        -------
        dict  task_name -> (psds, freqs)
              psds shape: (n_channels, n_freqs)
        list[str]  Channel names.
        """
        tasks = loader.get_task_list()
        sfreq = loader.sfreq
        ch_names = [ch for ch in loader.raw.ch_names if ch in df.columns]

        result = {}

        for task_name in tasks:
            seg = loader.extract_task_segments(df, task_name)
            if seg.empty or len(seg) < 8:
                continue

            # Ambil data channel saja (hapus time & marker)
            data = seg[ch_names].values.T  # (n_channels, n_samples)
            psds, freqs = PSDAnalyzer.compute_psd_array(
                data, sfreq, method=method,
                fmin=fmin, fmax=fmax, n_fft=n_fft,
            )
            result[task_name] = (psds, freqs)

        return result, ch_names

    # ------------------------------------------------------------------ #
    #  Band Power dari PSD                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_band_power_from_psd(psds, freqs, ch_names, subbands=None):
        """Ekstrak band power numerik dari PSD yang sudah dihitung.

        Parameters
        ----------
        psds : np.ndarray  (n_channels, n_freqs)
        freqs : np.ndarray (n_freqs,)
        ch_names : list[str]
        subbands : dict  {"Delta": (0.5, 4), ...}

        Returns
        -------
        pd.DataFrame  Kolom: channel, subband, band_power, relative_power,
                       peak_frequency
        """
        if subbands is None:
            subbands = DEFAULT_SUBBANDS

        freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        rows = []

        for ch_idx, ch in enumerate(ch_names):
            psd_ch = psds[ch_idx]
            total_power = float(_trapz(psd_ch, dx=freq_resolution) if len(psd_ch) > 0 else 0.0)

            for sb_name, (low, high) in subbands.items():
                band_mask = (freqs >= low) & (freqs <= high)

                if not band_mask.any():
                    rows.append({
                        "channel": ch, "subband": sb_name,
                        "band_power": 0.0,
                        "relative_power": 0.0,
                        "peak_frequency": 0.0,
                    })
                    continue

                band_psd = psd_ch[band_mask]
                band_freqs = freqs[band_mask]

                band_power = float(_trapz(band_psd, dx=freq_resolution) if len(band_psd) > 0 else 0.0)
                relative_power = (
                    float(band_power / total_power)
                    if total_power > 0 else 0.0
                )
                peak_freq = float(band_freqs[np.argmax(band_psd)])

                rows.append({
                    "channel": ch,
                    "subband": sb_name,
                    "band_power": band_power,
                    "relative_power": relative_power,
                    "peak_frequency": peak_freq,
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Band Power per task                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_task_band_power(loader, df, channels, tasks, subbands=None,
                                method="welch", fmin=0.0, fmax=49.0,
                                n_fft=None):
        """Hitung band power per task per channel per subband dari PSD.

        Parameters
        ----------
        loader : EEGLoader
        df : pd.DataFrame
        channels : list[str]
        tasks : list[str]
        subbands : dict | None
        method, fmin, fmax, n_fft : parameter PSD.

        Returns
        -------
        pd.DataFrame  Kolom: task, channel, subband,
                       band_power, relative_power, peak_frequency
        """
        if subbands is None:
            subbands = DEFAULT_SUBBANDS

        sfreq = loader.sfreq
        all_rows = []

        for task_name in tasks:
            seg = loader.extract_task_segments(df, task_name)
            if seg.empty or len(seg) < 8:
                continue

            ch_list = [ch for ch in channels if ch in seg.columns]
            if not ch_list:
                continue

            data = seg[ch_list].values.T  # (n_channels, n_samples)

            psds, freqs = PSDAnalyzer.compute_psd_array(
                data, sfreq, method=method,
                fmin=fmin, fmax=fmax, n_fft=n_fft,
            )

            bp_df = PSDAnalyzer.compute_band_power_from_psd(
                psds, freqs, ch_list, subbands
            )
            bp_df.insert(0, "task", task_name)
            all_rows.append(bp_df)

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()
