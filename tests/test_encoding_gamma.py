import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdt

from processing.encoding import encode_cached_items


class TestEncodingGammaBursts(unittest.TestCase):
    def _make_cached_item(self):
        sfreq = 128.0
        duration_s = 3.0
        n_samples = int(sfreq * duration_s)
        t = np.arange(n_samples) / sfreq

        rng = np.random.default_rng(42)
        envelope = np.zeros(n_samples)
        envelope[int(0.6 * sfreq) : int(1.0 * sfreq)] = 1.0
        envelope[int(1.8 * sfreq) : int(2.2 * sfreq)] = 1.0

        gamma_carrier = np.sin(2.0 * np.pi * 40.0 * t)
        cz = (0.2e-6 * rng.standard_normal(n_samples)) + (8e-6 * envelope * gamma_carrier)
        pz = 0.2e-6 * rng.standard_normal(n_samples)

        raw_df = pd.DataFrame({
            "time": t,
            "Cz": cz,
            "Pz": pz,
        })

        return {
            "df": raw_df,
            "channels": ["Cz", "Pz"],
            "sfreq": sfreq,
            "subject_id": "id_test",
            "scenario": 1,
            "scenario_id": 1,
            "filename": "synthetic.edf",
        }

    def test_gamma_burst_columns_are_additive_and_scoped(self):
        item = self._make_cached_item()
        common_kwargs = {
            "cached_items": [item],
            "window_size": 0.3,
            "overlap": 0.0,
            "subbands": {"Alpha": (8.0, 13.0), "Gamma": (30.0, 49.5)},
            "features": ["mav", "variance", "std"],
            "include_frequency": False,
        }

        baseline_df, _ = encode_cached_items(
            include_gamma_bursts=False,
            **common_kwargs,
        )
        gamma_df, _ = encode_cached_items(
            include_gamma_bursts=True,
            **common_kwargs,
        )

        self.assertFalse(baseline_df.empty)
        self.assertFalse(gamma_df.empty)

        pdt.assert_frame_equal(
            baseline_df.reset_index(drop=True),
            gamma_df[baseline_df.columns].reset_index(drop=True),
            check_exact=True,
        )

        new_cols = [
            "burst_count",
            "burst_peak_amp",
            "burst_duration_ratio",
            "burst_mean_freq",
        ]
        for col in new_cols:
            self.assertIn(col, gamma_df.columns)

        gamma_rows = gamma_df[gamma_df["subband"] == "Gamma"]
        non_gamma_rows = gamma_df[gamma_df["subband"] != "Gamma"]

        self.assertFalse(gamma_rows.empty)
        self.assertFalse(gamma_rows[["burst_count", "burst_peak_amp", "burst_duration_ratio"]].isna().any().any())
        self.assertTrue(non_gamma_rows[new_cols].isna().all().all())

        self.assertTrue((gamma_rows["burst_count"] >= 0).all())
        self.assertTrue((gamma_rows["burst_duration_ratio"] >= 0.0).all())
        self.assertTrue((gamma_rows["burst_duration_ratio"] <= 1.0).all())
        self.assertTrue(np.allclose(gamma_rows["burst_count"], np.floor(gamma_rows["burst_count"])))


if __name__ == "__main__":
    unittest.main()
