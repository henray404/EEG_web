"""
Modul loader — Load data EEG dari file EDF atau arsip ZIP.
Fungsi deteksi metadata (kategori, subject, time, scenario) juga ada di sini.
"""

import io
import os
import re
import zipfile
import tempfile

import mne
import numpy as np
import pandas as pd

from config import OPENBCI_CHANNEL_MAP, OPENBCI_SFREQ, OPENBCI_CONDITIONS


class EEGLoader:
    """Loader untuk file EDF dan ZIP berisi EDF."""

    def __init__(self):
        self.raw = None
        self.raw_original = None
        self.sfreq = None
        self.channel_names = []
        self.processing_log = []
        self._tmp_path = None

    # ------------------------------------------------------------------ #
    #  Temp file management                                               #
    # ------------------------------------------------------------------ #

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
                self.raw = mne.io.read_raw_edf(
                    str(file_source), preload=True, verbose=False
                )
            else:
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    tmp.write(file_source.read())
                    self._tmp_path = tmp.name
                self.raw = mne.io.read_raw_edf(
                    self._tmp_path, preload=True, verbose=False
                )

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
        self.raw = mne.io.read_raw_edf(
            self._tmp_path, preload=True, verbose=False
        )

        self.raw_original = self.raw.copy()
        self.sfreq = self.raw.info["sfreq"]
        self.channel_names = self.raw.ch_names
        self.processing_log = [f"File EDF dimuat dari ZIP: {edf_path_in_zip}"]
        return self.get_raw_info()

    # ------------------------------------------------------------------ #
    #  Info & metadata                                                    #
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
                {
                    "onset": a["onset"],
                    "duration": a["duration"],
                    "description": a["description"],
                }
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
        descs = [
            "Think_Acting" if d == "Thinking and Acting" else d for d in descs
        ]
        return sorted(set(descs))

    # ------------------------------------------------------------------ #
    #  Data extraction                                                    #
    # ------------------------------------------------------------------ #

    def extract_dataframe(self):
        """Konversi raw data ke DataFrame dengan kolom waktu, channel, marker."""
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

        df["marker"] = df["marker"].replace("Thinking and Acting", "Think_Acting")

        self.processing_log.append("DataFrame diekstrak")
        return df

    def extract_task_segments(self, df, task_name):
        """Filter DataFrame ke segment yang berisi task tertentu."""
        return df[df["marker"] == task_name].copy()

    def get_task_summary(self, df):
        """Hitung statistik ringkasan per task."""
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
    #  Per-occurrence analysis                                             #
    # ------------------------------------------------------------------ #

    def get_task_occurrences(self):
        """Return daftar occurrence per task, dalam urutan waktu.

        Returns
        -------
        list[dict]  Tiap dict: {"task", "occurrence", "onset", "duration"}
                    Diurutkan berdasarkan onset.
        """
        if self.raw is None:
            return []

        annotations = self.raw.annotations
        if len(annotations) == 0:
            return []

        counter = {}
        occurrences = []

        for annot in annotations:
            desc = annot["description"]
            if desc == "Thinking and Acting":
                desc = "Think_Acting"
            if desc not in counter:
                counter[desc] = 0
            counter[desc] += 1

            occurrences.append({
                "task": desc,
                "occurrence": counter[desc],
                "onset": annot["onset"],
                "duration": annot["duration"],
            })

        return occurrences

    def extract_occurrence_segment(self, df, task_name, occurrence_num):
        """Ekstrak data untuk occurrence spesifik dari satu task.

        Parameters
        ----------
        df : pd.DataFrame  DataFrame dengan kolom 'time' dan 'marker'.
        task_name : str     Nama task (misal 'Resting').
        occurrence_num : int  Nomor occurrence (1-based).

        Returns
        -------
        pd.DataFrame  Subset data untuk occurrence tersebut.
        """
        task_df = df[df["marker"] == task_name].copy()
        if task_df.empty:
            return pd.DataFrame()

        # Cari batas-batas segments berdasarkan gap di index
        indices = task_df.index.tolist()
        segments = []
        seg_start = indices[0]

        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] > 1:
                segments.append((seg_start, indices[i - 1]))
                seg_start = indices[i]
        segments.append((seg_start, indices[-1]))

        if occurrence_num < 1 or occurrence_num > len(segments):
            return pd.DataFrame()

        start_idx, end_idx = segments[occurrence_num - 1]
        return df.loc[start_idx:end_idx].copy()

    def get_occurrence_pairs(self, task_a, task_b):
        """Temukan pasangan sequential occurrence dari dua task.

        Mencari occurrence task_a yang diikuti (atau mendahului)
        task_b secara langsung berdasarkan urutan waktu.

        Returns
        -------
        list[tuple]  [(occ_a, occ_b), ...]
                      occ_a = nomor occurrence task_a (1-based)
                      occ_b = nomor occurrence task_b (1-based)
        """
        occurrences = self.get_task_occurrences()
        if not occurrences:
            return []

        pairs = []
        n = len(occurrences)

        for i in range(n - 1):
            curr = occurrences[i]
            nxt = occurrences[i + 1]
            if curr["task"] == task_a and nxt["task"] == task_b:
                pairs.append((curr["occurrence"], nxt["occurrence"]))
            elif curr["task"] == task_b and nxt["task"] == task_a:
                pairs.append((nxt["occurrence"], nxt["occurrence"]))

        # Deduplicate
        seen = set()
        unique_pairs = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)

        return unique_pairs

    # ------------------------------------------------------------------ #
    #  Metadata detection (untuk batch / ZIP)                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_category(edf_path_in_zip):
        """Deteksi kategori (ALS/Normal) dan metadata dari path di ZIP.

        Mendukung berbagai kedalaman folder:
          ALS05/time5/scenario4/EEG.edf
          EEGET-ALS Dataset/ALS05/time5/scenario4/EEG.edf
          id9/time1/scenario4/EEG.edf

        Returns
        -------
        dict  {"category", "subject", "time", "scenario"}
        """
        parts = edf_path_in_zip.replace("\\", "/").split("/")
        parts = [p for p in parts if p and not p.lower().endswith(".edf")]

        subject = "unknown"
        category = "Normal"
        time_val = ""
        scenario_val = ""

        for p in parts:
            p_upper = p.upper()
            p_lower = p.lower()

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

    # ------------------------------------------------------------------ #
    #  OpenBCI TXT loading                                                #
    # ------------------------------------------------------------------ #

    def load_openbci_txt(self, file_source, channel_map=None):
        """Load file TXT OpenBCI Cyton.

        Parameters
        ----------
        file_source : str | PathLike | file-like
            Path atau upload buffer ke file .txt OpenBCI.
        channel_map : dict | None
            Mapping kolom CSV → nama channel EEG.
            Default: OPENBCI_CHANNEL_MAP (EXG Channel 2–7 → T3–O2).

        Returns
        -------
        dict  Info ringkasan (channels, sfreq, duration_s, ...).
        """
        if channel_map is None:
            channel_map = OPENBCI_CHANNEL_MAP

        try:
            self._cleanup_tmp()

            # Baca file content
            if isinstance(file_source, (str, os.PathLike)):
                with open(str(file_source), "r") as f:
                    lines = f.readlines()
            else:
                content = file_source.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                lines = content.splitlines(keepends=True)

            # Skip header lines starting with %
            data_lines = []
            header_line = None
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("%"):
                    continue
                if header_line is None:
                    header_line = stripped
                    continue
                data_lines.append(stripped)

            if header_line is None or not data_lines:
                raise ValueError("File TXT kosong atau tidak valid.")

            # Parse header
            header_cols = [c.strip() for c in header_line.split(",")]

            # Parse data rows
            parsed_rows = []
            for line in data_lines:
                vals = line.split(",")
                if len(vals) >= len(header_cols):
                    row = {}
                    for idx, col_name in enumerate(header_cols):
                        try:
                            row[col_name] = float(vals[idx].strip())
                        except (ValueError, IndexError):
                            row[col_name] = 0.0
                    parsed_rows.append(row)

            if not parsed_rows:
                raise ValueError("Tidak ada data valid dalam file TXT.")

            df_raw = pd.DataFrame(parsed_rows)

            # Ekstrak channel yang ada di channel_map
            available_cols = [c for c in channel_map.keys() if c in df_raw.columns]
            if not available_cols:
                raise ValueError(
                    f"Tidak ditemukan channel OpenBCI yang diharapkan. "
                    f"Kolom file: {list(df_raw.columns)[:10]}"
                )

            ch_names = [channel_map[c] for c in available_cols]
            data = df_raw[available_cols].values.T  # (n_channels, n_times)

            # Konversi ke µV jika nilainya terlalu besar (OpenBCI default sudah µV)
            # Buat MNE RawArray
            sfreq = OPENBCI_SFREQ
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=sfreq,
                ch_types="eeg",
            )
            # Data OpenBCI sudah dalam µV, MNE expects V → konversi
            data_volts = data * 1e-6
            self.raw = mne.io.RawArray(data_volts, info, verbose=False)
            self.raw_original = self.raw.copy()
            self.sfreq = sfreq
            self.channel_names = ch_names
            self.processing_log = [
                f"File TXT OpenBCI dimuat: {len(ch_names)} channel, "
                f"{data.shape[1]} sampel ({data.shape[1]/sfreq:.1f}s)"
            ]
            return self.get_raw_info()

        except Exception as exc:
            raise RuntimeError(f"Gagal memuat file TXT OpenBCI: {exc}") from exc

    def load_txt_from_zip(self, zip_buffer, txt_path_in_zip, channel_map=None):
        """Ekstrak satu file TXT dari ZIP lalu load sebagai OpenBCI.

        Parameters
        ----------
        zip_buffer : file-like
            Buffer ZIP.
        txt_path_in_zip : str
            Path file .txt di dalam ZIP.
        channel_map : dict | None
            Mapping kolom → channel name.

        Returns
        -------
        dict  Info ringkasan.
        """
        self._cleanup_tmp()
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            with zf.open(txt_path_in_zip) as txt_file:
                data = txt_file.read()

        return self.load_openbci_txt(
            io.BytesIO(data),
            channel_map=channel_map,
        )

    @staticmethod
    def list_txt_in_zip(zip_buffer):
        """Temukan semua file TXT di dalam arsip ZIP.

        Returns
        -------
        list[str]  Daftar path .txt dalam ZIP (sorted).
        """
        txt_files = []
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".txt"):
                    # Skip file yang namanya dimulai CATATAN atau bukan data EEG
                    basename = os.path.basename(name).lower()
                    if basename.startswith("catatan"):
                        continue
                    txt_files.append(name)
        return sorted(txt_files)

    @staticmethod
    def detect_openbci_metadata(txt_path):
        """Deteksi subject ID dan kondisi dari path/nama file TXT OpenBCI.

        Mendukung format:
          - Baseline/1baseline.txt → subject=1, condition=baseline
          - Familiar/alenbaseline.txt → subject=alen, condition=familiar
          - Unfamiliar/4nonfamiliar.txt → subject=4, condition=unfamiliar

        Returns
        -------
        dict  {"subject": str, "condition": str}
        """
        basename = os.path.basename(txt_path)
        name_no_ext = os.path.splitext(basename)[0].strip().lower()

        # Deteksi kondisi dari nama file atau path
        condition = "unknown"
        subject_id = name_no_ext

        # Coba cari kondisi di nama file (prioritas: paling panjang dulu)
        conditions_sorted = sorted(OPENBCI_CONDITIONS, key=len, reverse=True)
        for cond in conditions_sorted:
            if cond in name_no_ext:
                condition = cond
                # Hapus kondisi dari nama untuk mendapat subject ID
                subject_id = name_no_ext.replace(cond, "").strip()
                break

        # Jika tidak ketemu dari nama file, coba dari folder parent
        if condition == "unknown":
            parts = txt_path.replace("\\", "/").split("/")
            for p in parts:
                p_lower = p.lower().strip()
                if p_lower in ["baseline", "familiar", "unfamiliar"]:
                    condition = p_lower
                    break

        # Normalize: nonfamiliar → unfamiliar
        if condition == "nonfamiliar":
            condition = "unfamiliar"

        # Clean subject_id
        subject_id = subject_id.strip("_- ")
        if not subject_id:
            subject_id = "unknown"

        return {
            "subject": subject_id,
            "condition": condition,
        }

    @staticmethod
    def scan_openbci_zip(zip_buffer):
        """Scan ZIP untuk struktur folder OpenBCI (Baseline/Familiar/Unfamiliar).

        Returns
        -------
        dict  {subject_id: {"baseline": path, "familiar": path, "unfamiliar": path}}
              Hanya subjek yang memiliki setidaknya 1 kondisi saja.
        list  Daftar file yang di-skip (tidak sesuai format).
        """
        zip_buffer.seek(0)
        txt_files = EEGLoader.list_txt_in_zip(zip_buffer)

        subjects = {}
        skipped = []

        for txt_path in txt_files:
            meta = EEGLoader.detect_openbci_metadata(txt_path)
            subj = meta["subject"]
            cond = meta["condition"]

            if cond == "unknown" or subj == "unknown":
                skipped.append(txt_path)
                continue

            if subj not in subjects:
                subjects[subj] = {}
            subjects[subj][cond] = txt_path

        return subjects, skipped

