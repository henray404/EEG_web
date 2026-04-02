"""
Modul connectivity — Analisis konektivitas fungsional EEG (PLI / wPLI).

Menggunakan mne-connectivity untuk komputasi spectral connectivity.
Menyediakan:
- Perhitungan PLI/wPLI per subband
- Analisis per task
- Konversi ke DataFrame untuk export
"""

import numpy as np
import pandas as pd

from config import DEFAULT_SUBBANDS, DEFAULT_CONNECTIVITY_METHOD


class ConnectivityAnalyzer:
    """Analisis konektivitas fungsional EEG (PLI/wPLI)."""

    # ------------------------------------------------------------------ #
    #  Core: hitung connectivity matrix                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_connectivity(data, sfreq, channels, method=None,
                             subbands=None):
        """Hitung connectivity matrix per subband.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_channels, n_times) untuk continuous,
            atau (n_epochs, n_channels, n_times) untuk epoched.
        sfreq : float
            Sampling frequency.
        channels : list[str]
            Nama channel.
        method : str
            "pli" atau "wpli". Default dari config.
        subbands : dict
            Misal {"Alpha": (8, 13), "Beta": (13, 30)}.

        Returns
        -------
        dict   subband_name -> np.ndarray shape (n_channels, n_channels)
        """
        from mne_connectivity import spectral_connectivity_epochs

        if method is None:
            method = DEFAULT_CONNECTIVITY_METHOD
        if subbands is None:
            subbands = DEFAULT_SUBBANDS

        # Pastikan data 3D: (n_epochs, n_channels, n_times)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]  # 1 epoch

        n_epochs, n_ch, n_times = data.shape

        # Siapkan fmin/fmax arrays dari subbands
        subband_names = list(subbands.keys())
        fmin_arr = [subbands[sb][0] for sb in subband_names]
        fmax_arr = [subbands[sb][1] for sb in subband_names]

        # Hitung spectral connectivity
        try:
            conn = spectral_connectivity_epochs(
                data, method=method, sfreq=sfreq,
                fmin=fmin_arr, fmax=fmax_arr,
                faverage=True,  # rata-ratakan dalam setiap band
                verbose=False,
            )
        except Exception as e:
            print(f"[ConnectivityAnalyzer] Error: {e}")
            return {}

        # Ambil data konektivitas
        conn_data = conn.get_data(output="dense")
        # Shape: (n_channels, n_channels, n_bands)

        result = {}
        for i, sb_name in enumerate(subband_names):
            matrix = conn_data[:, :, i]
            # Pastikan simetris dan diagonal = 0
            matrix = (matrix + matrix.T) / 2.0
            np.fill_diagonal(matrix, 0.0)
            result[sb_name] = matrix

        return result

    # ------------------------------------------------------------------ #
    #  Per-task connectivity                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_task_connectivity(loader, df, channels, tasks, sfreq,
                                  method=None, subbands=None,
                                  use_epoching=False, epoch_duration=2.0):
        """Hitung connectivity per task.

        Parameters
        ----------
        loader : EEGLoader
        df : pd.DataFrame
        channels : list[str]
        tasks : list[str]
        sfreq : float
        method : str
        subbands : dict
        use_epoching : bool
            Jika True, segmen task dipotong jadi epoch dulu.
        epoch_duration : float
            Durasi epoch dalam detik (jika use_epoching=True).

        Returns
        -------
        dict   task_name -> dict[subband_name -> np.ndarray]
        """
        if subbands is None:
            subbands = DEFAULT_SUBBANDS

        result = {}
        for task_name in tasks:
            seg = loader.extract_task_segments(df, task_name)
            if seg.empty:
                continue

            ch_cols = [ch for ch in channels if ch in seg.columns]
            if not ch_cols:
                continue

            raw_data = seg[ch_cols].values.T  # (n_channels, n_times)

            # Konektivitas matematis mutlak mensyaratkan > 1 epoch (ensemble) 
            # untuk distribusi fase. Jika epoching dimatikan user, kita paksa 
            # pemotongan 2 detik secara internal khusus untuk PLI/wPLI.
            current_epoch_duration = epoch_duration if use_epoching else 2.0
            
            samples_per_epoch = int(current_epoch_duration * sfreq)
            n_samples = raw_data.shape[1]
            n_epochs = n_samples // samples_per_epoch

            if n_epochs < 2:
                # Jika data total masih kurang dari 4 detik (n_epochs < 2),
                # kita paksa potong tepat di tengah agar mendapat 2 epoch (minimal valid).
                if n_samples >= int(sfreq * 1.0): # minimal 1 detik total data
                    samples_per_epoch = n_samples // 2
                    n_epochs = 2
                    trimmed = raw_data[:, :n_epochs * samples_per_epoch]
                    data_3d = trimmed.reshape(raw_data.shape[0], n_epochs, samples_per_epoch).transpose(1, 0, 2)
                else:
                    # Terlalu pendek, pasrah 1 epoch (nilai pasti 0.0 atau 0.5)
                    data_3d = raw_data[np.newaxis, :, :]
            else:
                trimmed = raw_data[:, :n_epochs * samples_per_epoch]
                data_3d = trimmed.reshape(
                    raw_data.shape[0], n_epochs, samples_per_epoch
                ).transpose(1, 0, 2)

            conn = ConnectivityAnalyzer.compute_connectivity(
                data_3d, sfreq, ch_cols, method=method, subbands=subbands,
            )
            if conn:
                result[task_name] = conn

        return result

    # ------------------------------------------------------------------ #
    #  Konversi ke DataFrame                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def connectivity_to_dataframe(conn_dict, channels, method="wpli",
                                  task_name=None):
        """Konversi connectivity matrices ke DataFrame.

        Parameters
        ----------
        conn_dict : dict
            subband_name -> np.ndarray (n_channels × n_channels)
        channels : list[str]
        method : str
        task_name : str | None

        Returns
        -------
        pd.DataFrame
            Kolom: [task], channel_a, channel_b, subband, value
        """
        rows = []
        n_ch = len(channels)

        for sb_name, matrix in conn_dict.items():
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    
                    val = float(matrix[i, j])
                    ket = "Lemah"
                    if abs(val) >= 0.5:
                        ket = "Kuat"
                    elif abs(val) >= 0.3:
                        ket = "Sedang"
                        
                    row = {
                        "channel_a": channels[i],
                        "channel_b": channels[j],
                        "subband": sb_name,
                        "method": method,
                        "value": val,
                        "keterangan": ket,
                    }
                    if task_name is not None:
                        row["task"] = task_name
                    rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Urutkan kolom
        col_order = []
        if task_name is not None:
            col_order.append("task")
        col_order.extend(["channel_a", "channel_b", "subband", "method", "value", "keterangan"])
        df = df[col_order]
        return df

    @staticmethod
    def all_tasks_to_dataframe(task_conn_dict, channels, method="wpli"):
        """Konversi semua task connectivity ke satu DataFrame.

        Parameters
        ----------
        task_conn_dict : dict
            task_name -> dict[subband -> matrix]
        channels : list[str]
        method : str

        Returns
        -------
        pd.DataFrame
        """
        all_dfs = []
        for task_name, conn_dict in task_conn_dict.items():
            df = ConnectivityAnalyzer.connectivity_to_dataframe(
                conn_dict, channels, method=method, task_name=task_name,
            )
            if not df.empty:
                all_dfs.append(df)

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()
