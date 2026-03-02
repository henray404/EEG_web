"""
Modul filters — Bandpass, notch, ICA, dan bad channel detection.

Bad channel detection diambil dari pipeline EEG-ALS- (02_split_to_csv.py).
"""

import numpy as np
import mne
from mne.preprocessing import ICA
from scipy.signal import butter, filtfilt

from config import BAD_CHANNEL_THRESHOLD


class EEGFilters:
    """Kumpulan metode filtering untuk data EEG."""

    # ------------------------------------------------------------------ #
    #  Channel selection                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def pick_channels(loader, channels):
        """Pilih subset channel dari raw data.

        Parameters
        ----------
        loader : EEGLoader
            Loader instance dengan raw data.
        channels : list[str]
            Channel yang dipilih.
        """
        if loader.raw is None:
            raise RuntimeError("Data belum dimuat.")
        available = [ch for ch in channels if ch in loader.raw.ch_names]
        if not available:
            raise ValueError("Tidak ada channel yang valid.")
        loader.raw.pick_channels(available)
        loader.processing_log.append(f"Channel dipilih: {available}")

    # ------------------------------------------------------------------ #
    #  Bad channel detection (BARU – dari pipeline)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_bad_channels(raw, threshold=None):
        """Deteksi channel buruk berdasarkan variance outlier (MAD threshold).

        Dari pipeline EEG-ALS-/02_split_to_csv.py:detect_bad_channels().

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MNE object.
        threshold : float
            MAD multiplier. Default dari config.BAD_CHANNEL_THRESHOLD.

        Returns
        -------
        list[str]  Nama channel yang terdeteksi buruk.
        """
        if threshold is None:
            threshold = BAD_CHANNEL_THRESHOLD

        data = raw.get_data()
        variances = np.var(data, axis=1)
        median_var = np.median(variances)
        mad = np.median(np.abs(variances - median_var))

        if mad == 0:
            return []

        bad_idx = np.where(np.abs(variances - median_var) > threshold * mad)[0]
        bad_channels = [raw.ch_names[i] for i in bad_idx]
        return bad_channels

    # ------------------------------------------------------------------ #
    #  Notch filter                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_notch(loader, freq=50.0, quality=30.0):
        """Terapkan notch filter untuk menghapus noise powerline."""
        if loader.raw is None:
            raise RuntimeError("Data belum dimuat.")
        loader.raw.notch_filter(freqs=freq, verbose=False)
        loader.processing_log.append(f"Notch filter: {freq} Hz")

    # ------------------------------------------------------------------ #
    #  Bandpass filter                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_bandpass(loader, low_freq, high_freq, order=5):
        """Terapkan bandpass filter pada raw data."""
        if loader.raw is None:
            raise RuntimeError("Data belum dimuat.")
        loader.raw.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
        loader.processing_log.append(f"Bandpass: {low_freq}-{high_freq} Hz")

    @staticmethod
    def bandpass_array(data, sfreq, low, high, order=5):
        """Bandpass filter pada array numpy 1-D."""
        nyq = 0.5 * sfreq
        low_n = max(low / nyq, 0.001)
        high_n = min(high / nyq, 0.999)
        b, a = butter(order, [low_n, high_n], btype="band")
        return filtfilt(b, a, data)

    # ------------------------------------------------------------------ #
    #  ICA                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_ica(loader, n_components=None, method="fastica",
                  random_state=42, auto_detect_artifacts=True):
        """Jalankan ICA dan hapus komponen artefak otomatis.

        Deteksi artefak:
          - EOG (kedipan mata) via channel Fp1/Fp2/F7/F8
          - Muscle (gerakan otot) via channel T7/T8/TP9/TP10
        Kedua jenis artefak dideteksi bersamaan dan semua komponen
        yang teridentifikasi akan dihapus tanpa batas jumlah.

        Parameters
        ----------
        loader : EEGLoader
        n_components : int | None
        method : str
        random_state : int
        auto_detect_artifacts : bool
            Jika True, gunakan deteksi artefak lengkap dari pipeline.
        """
        if loader.raw is None:
            raise RuntimeError("Data belum dimuat.")

        n_channels = len(loader.raw.ch_names)
        if n_components is not None:
            n_comp = min(n_components, n_channels - 1)
        else:
            n_comp = min(15, n_channels - 1)

        if n_comp < 2:
            loader.processing_log.append("ICA dilewati (channel terlalu sedikit)")
            return

        ica = ICA(n_components=n_comp, method=method,
                  random_state=random_state, max_iter=500)
        ica.fit(loader.raw, verbose=False)

        bad_ica = []

        if auto_detect_artifacts:
            # --- EOG artifacts (kedipan mata) ---
            eog_indices = []
            for ch in loader.raw.ch_names:
                ch_lower = ch.lower()
                if any(x in ch_lower for x in ["eog", "fp1", "fp2", "f7", "f8"]):
                    try:
                        eog_idx, _ = ica.find_bads_eog(
                            loader.raw, ch_name=ch, verbose=False
                        )
                        eog_indices.extend(eog_idx)
                    except Exception:
                        continue

            # --- Muscle artifacts (gerakan otot kepala) ---
            muscle_indices = []
            for ch in loader.raw.ch_names:
                ch_lower = ch.lower()
                if any(x in ch_lower for x in ["t7", "t8", "tp9", "tp10"]):
                    try:
                        muscle_idx, _ = ica.find_bads_muscle(
                            loader.raw, verbose=False
                        )
                        muscle_indices.extend(muscle_idx)
                    except Exception:
                        continue

            # Gabungkan semua artefak (EOG + Muscle) tanpa batas
            bad_ica = list(set(eog_indices + muscle_indices))
        else:
            try:
                eog_indices, _ = ica.find_bads_eog(loader.raw, verbose=False)
                if eog_indices:
                    bad_ica = eog_indices
            except Exception:
                pass

        if bad_ica:
            ica.exclude = bad_ica
        ica.apply(loader.raw, verbose=False)

        n_excluded = len(ica.exclude)
        loader.processing_log.append(
            f"ICA (metode={method}, komponen={ica.n_components_}, "
            f"artefak={n_excluded})"
        )
