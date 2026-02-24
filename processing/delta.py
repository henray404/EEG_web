"""
Modul delta — Perhitungan delta antar task.

Mengimplementasikan dua metode:
1. Delta klasik: (task_a − task_b) per file/channel/subband.
2. Delta per-subject (BARU dari pipeline): hitung delta per pasien dulu,
   lalu rata-rata per group → statistik lebih akurat.
"""

import numpy as np
import pandas as pd
from scipy import stats


class DeltaCalculator:
    """Kumpulan metode untuk menghitung delta antar task."""

    # ------------------------------------------------------------------ #
    #  Delta klasik (existing)                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_task_delta(batch_df, task_a, task_b, feature_cols=None):
        """Hitung delta (task_a − task_b) per filename/channel/subband.

        Returns
        -------
        delta_df : pd.DataFrame
            Delta per file/channel/subband.
        agg_df : pd.DataFrame
            Statistik agregat per channel/subband.
        """
        if batch_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if feature_cols is None:
            exclude = {"filename", "task", "channel", "subband",
                        "category", "subject", "time", "scenario"}
            feature_cols = [c for c in batch_df.columns if c not in exclude]

        df_a = batch_df[batch_df["task"] == task_a].copy()
        df_b = batch_df[batch_df["task"] == task_b].copy()

        if df_a.empty or df_b.empty:
            return pd.DataFrame(), pd.DataFrame()

        merge_keys = ["filename", "channel", "subband"]
        # Include category and subject if present
        for col in ["category", "subject"]:
            if col in df_a.columns and col in df_b.columns:
                merge_keys.append(col)

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

        # Agregasi per channel/subband
        delta_cols = [c for c in delta_df.columns if c.startswith("delta_")]
        agg_dict = {}
        for dc in delta_cols:
            agg_dict[dc] = ["mean", "std"]

        agg_df = delta_df.groupby(["channel", "subband"]).agg(agg_dict)
        agg_df.columns = [f"{stat}_{col}" for col, stat in agg_df.columns]
        agg_df = agg_df.reset_index()

        return delta_df, agg_df

    # ------------------------------------------------------------------ #
    #  Delta per-subject (BARU – dari pipeline)                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_subject_delta(batch_df, subject_id, from_task, to_task,
                               feature_col, subband=None, channels=None):
        """Hitung delta untuk satu subjek (per task transition).

        Dari pipeline EEG-ALS-/04_experiment_plot.py:compute_subject_delta_band().

        Parameters
        ----------
        batch_df : pd.DataFrame
        subject_id : str
        from_task, to_task : str
        feature_col : str
        subband : str | None
        channels : list[str] | None

        Returns
        -------
        dict | None  {"from_val", "to_val", "delta"} atau None.
        """
        subj_df = batch_df[batch_df["subject"] == subject_id]
        if subj_df.empty:
            return None

        if subband and "subband" in subj_df.columns:
            subj_df = subj_df[subj_df["subband"] == subband]

        if channels and "channel" in subj_df.columns:
            subj_df = subj_df[subj_df["channel"].isin(channels)]

        from_df = subj_df[subj_df["task"] == from_task]
        to_df = subj_df[subj_df["task"] == to_task]

        if from_df.empty or to_df.empty:
            return None

        if feature_col not in from_df.columns:
            return None

        from_mean = from_df[feature_col].mean()
        to_mean = to_df[feature_col].mean()

        return {
            "from_val": float(from_mean),
            "to_val": float(to_mean),
            "delta": float(to_mean - from_mean),
        }

    @staticmethod
    def compute_group_transition_deltas(batch_df, from_task, to_task,
                                         feature_col, subband=None,
                                         channels=None, scenarios=None,
                                         sessions=None):
        """Hitung transition delta per group (ALS / Normal).

        Dari pipeline EEG-ALS-/04_experiment_plot.py:compute_group_transition_deltas().

        Menghitung delta per subject → kemudian mean, SEM, n per group.

        Returns
        -------
        dict  {"ALS": {mean, sem, n, deltas}, "Normal": {...}}.
        """
        df = batch_df.copy()

        if scenarios and "scenario" in df.columns:
            df = df[df["scenario"].isin(scenarios)]
        if sessions and "time" in df.columns:
            df = df[df["time"].isin(sessions)]

        results = {}
        for group in ["ALS", "Normal"]:
            if "category" not in df.columns:
                continue
            group_df = df[df["category"] == group]
            subjects = sorted(group_df["subject"].unique())

            subject_deltas = []
            for subj in subjects:
                result = DeltaCalculator.compute_subject_delta(
                    group_df, subj, from_task, to_task,
                    feature_col, subband, channels,
                )
                if result is not None:
                    subject_deltas.append(result["delta"])

            if subject_deltas:
                arr = np.array(subject_deltas)
                results[group] = {
                    "mean": float(np.mean(arr)),
                    "sem": float(stats.sem(arr)) if len(arr) > 1 else 0.0,
                    "n": len(arr),
                    "deltas": subject_deltas,
                }

        return results

    @staticmethod
    def compute_transition_table(batch_df, from_task, to_task,
                                  feature_col, subbands=None,
                                  channels=None, scenarios=None,
                                  sessions=None):
        """Hitung transition delta untuk semua subband.

        Returns
        -------
        pd.DataFrame  Kolom: subband, als_mean, als_sem, als_n,
                              normal_mean, normal_sem, normal_n.
        """
        if subbands is None:
            from config import DEFAULT_SUBBANDS
            subbands = DEFAULT_SUBBANDS

        rows = []
        for sb_name in subbands:
            group_data = DeltaCalculator.compute_group_transition_deltas(
                batch_df, from_task, to_task, feature_col,
                subband=sb_name, channels=channels,
                scenarios=scenarios, sessions=sessions,
            )
            row = {
                "subband": sb_name,
                "transition": f"{from_task} → {to_task}",
            }
            for grp in ["ALS", "Normal"]:
                prefix = grp.lower()
                if grp in group_data:
                    row[f"{prefix}_mean"] = group_data[grp]["mean"]
                    row[f"{prefix}_sem"] = group_data[grp]["sem"]
                    row[f"{prefix}_n"] = group_data[grp]["n"]
                else:
                    row[f"{prefix}_mean"] = np.nan
                    row[f"{prefix}_sem"] = np.nan
                    row[f"{prefix}_n"] = 0
            rows.append(row)

        return pd.DataFrame(rows)
