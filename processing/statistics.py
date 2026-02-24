"""
Modul statistics — Uji statistik untuk perbandingan ALS vs Normal.

Fitur:
- Mann-Whitney U test (existing)
- Independent t-test (BARU dari pipeline)
- Cohen's d effect size (BARU dari pipeline)
- Standard Error of Mean (BARU dari pipeline)
- FDR correction Benjamini-Hochberg (existing)
- Normalisasi per subjek
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, sem


def _benjamini_hochberg(pvals):
    """Koreksi Benjamini-Hochberg (FDR) untuk array p-value."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    adjusted = pvals * n / ranked
    adjusted = np.minimum.accumulate(adjusted[np.argsort(-ranked)])[
        np.argsort(np.argsort(-ranked))
    ]
    return np.clip(adjusted, 0, 1)


def cohens_d(group1, group2):
    """Hitung Cohen's d (effect size).

    Dari pipeline EEG-ALS-/04_experiment_plot.py.

    Returns
    -------
    float  Cohen's d. Positif berarti group1 > group2.
    """
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((m1 - m2) / pooled)


def interpret_cohens_d(d):
    """Interpretasi ukuran efek Cohen's d."""
    ad = abs(d)
    if ad < 0.2:
        return "Sangat Kecil"
    elif ad < 0.5:
        return "Kecil"
    elif ad < 0.8:
        return "Sedang"
    else:
        return "Besar"


class StatisticalTests:
    """Kumpulan uji statistik untuk perbandingan group."""

    # ------------------------------------------------------------------ #
    #  Normalisasi                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_per_subject(batch_df, feature_cols, method="zscore",
                               scope="subject"):
        """Normalisasi fitur per subjek.

        Parameters
        ----------
        method : str  'zscore' | 'minmax'
        scope : str  'subject' | 'subject_scenario'
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

    # ------------------------------------------------------------------ #
    #  Perbandingan ALS vs Normal (enhanced)                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compare_als_vs_normal(batch_df, active_task, baseline_task="Resting",
                               feature_cols=None, mode="delta",
                               apply_fdr=True, compute_effect_size=True,
                               include_ttest=True):
        """Bandingkan fitur antara ALS dan Normal.

        Parameters
        ----------
        mode : str
            'delta'  — bandingkan delta (active − baseline).
            'zscore' — bandingkan z-scored langsung.
            'both'   — z-score dulu, lalu delta.
        apply_fdr : bool
            Koreksi FDR (Benjamini-Hochberg).
        compute_effect_size : bool
            Hitung Cohen's d.
        include_ttest : bool  (BARU)
            Hitung independent t-test selain Mann-Whitney U.

        Returns
        -------
        compare_df, stats_df : pd.DataFrame
        """
        if batch_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        meta_cols = {"filename", "category", "subject", "time", "scenario",
                     "task", "channel", "subband"}
        if feature_cols is None:
            feature_cols = [c for c in batch_df.columns if c not in meta_cols]

        group_keys = ["category", "subject", "channel", "subband"]

        # ---- Prepare data ----
        if mode == "zscore":
            df_task = batch_df[batch_df["task"] == active_task].copy()
            if df_task.empty:
                return pd.DataFrame(), pd.DataFrame()
            df_task = StatisticalTests.normalize_per_subject(
                df_task, feature_cols, method="zscore", scope="subject"
            )
            compare_df = df_task.groupby(group_keys)[feature_cols].mean().reset_index()
            value_cols = feature_cols

        elif mode == "both":
            df_two = batch_df[
                batch_df["task"].isin([active_task, baseline_task])
            ].copy()
            if df_two.empty:
                return pd.DataFrame(), pd.DataFrame()
            df_two = StatisticalTests.normalize_per_subject(
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
            # Delta only (original)
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

                # SEM (BARU)
                row[f"sem_als_{feat_name}"] = (
                    float(sem(als_vals)) if len(als_vals) >= 2 else np.nan
                )
                row[f"sem_normal_{feat_name}"] = (
                    float(sem(norm_vals)) if len(norm_vals) >= 2 else np.nan
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

                # T-test (BARU dari pipeline)
                if include_ttest and len(als_vals) >= 2 and len(norm_vals) >= 2:
                    try:
                        t_stat, t_pval = ttest_ind(als_vals, norm_vals)
                        row[f"t_stat_{feat_name}"] = float(t_stat)
                        row[f"t_pval_{feat_name}"] = float(t_pval)
                    except Exception:
                        row[f"t_stat_{feat_name}"] = np.nan
                        row[f"t_pval_{feat_name}"] = np.nan

                # Cohen's d (BARU dari pipeline, sekarang selalu aktif)
                if compute_effect_size and len(als_vals) >= 2 and len(norm_vals) >= 2:
                    row[f"cohend_{feat_name}"] = cohens_d(
                        als_vals.values, norm_vals.values
                    )
                    row[f"effect_{feat_name}"] = interpret_cohens_d(
                        row[f"cohend_{feat_name}"]
                    )
                elif compute_effect_size:
                    row[f"cohend_{feat_name}"] = np.nan
                    row[f"effect_{feat_name}"] = "–"

            stats_rows.append(row)

        stats_df = pd.DataFrame(stats_rows)

        # ---- FDR correction ----
        if apply_fdr and not stats_df.empty:
            p_cols = [c for c in stats_df.columns if c.startswith("p_")]
            for pc in p_cols:
                raw_p = stats_df[pc].values.copy()
                valid_mask = ~np.isnan(raw_p)
                if valid_mask.sum() >= 2:
                    try:
                        adjusted = np.full_like(raw_p, np.nan)
                        adjusted[valid_mask] = _benjamini_hochberg(
                            raw_p[valid_mask]
                        )
                        fdr_col = pc.replace("p_", "p_fdr_")
                        stats_df[fdr_col] = adjusted
                    except Exception:
                        pass

        return compare_df, stats_df
