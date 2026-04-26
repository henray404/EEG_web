import unittest

import numpy as np

from processing.gamma_bursts import Burst, GammaBurstDetector


class TestGammaBurstDetector(unittest.TestCase):
    def setUp(self):
        self.sfreq = 128.0
        self.freqs = np.array([30.0, 40.0, 49.5])
        self.detector = GammaBurstDetector(
            sfreq=self.sfreq,
            freqs=self.freqs,
            mad_k=2.0,
            min_duration_ms=25.0,
            merge_gap_ms=25.0,
        )

    def _make_power(self, n_samples, intervals):
        """Build synthetic power map with high-power burst intervals."""
        rng = np.random.default_rng(7)
        p = 1.0 + 0.05 * rng.standard_normal((len(self.freqs), n_samples))

        for start, end, amp in intervals:
            p[:, start:end] += amp
            p[1, start:end] += amp * 0.25  # make 40 Hz slightly dominant

        return p

    def test_detect_single_burst(self):
        p = self._make_power(256, [(80, 108, 4.0)])
        bursts = self.detector.detect(p)

        self.assertEqual(len(bursts), 1)
        b = bursts[0]
        self.assertGreaterEqual(b.end_sample - b.start_sample + 1, 3)
        self.assertGreater(b.peak_amp, 1.0)
        self.assertGreaterEqual(b.mean_freq, 30.0)
        self.assertLessEqual(b.mean_freq, 49.5)

    def test_merge_close_bursts(self):
        # Gap of 2 samples (< 25 ms at 128 Hz -> 3 samples) should merge.
        p = self._make_power(256, [(80, 95, 4.0), (97, 112, 4.0)])
        bursts = self.detector.detect(p)
        self.assertEqual(len(bursts), 1)

    def test_reject_short_burst(self):
        # 2-sample spike should be rejected by min duration filter.
        p = self._make_power(256, [(100, 102, 5.0)])
        bursts = self.detector.detect(p)
        self.assertEqual(len(bursts), 0)

    def test_aggregate_in_window(self):
        bursts = [
            Burst(start_sample=50, end_sample=74, peak_amp=3.0, mean_freq=40.0),
            Burst(start_sample=120, end_sample=149, peak_amp=5.0, mean_freq=42.0),
        ]
        agg = self.detector.aggregate_in_window(
            bursts,
            window_start_sample=40,
            window_end_sample=130,
        )

        self.assertEqual(agg["burst_count"], 2)
        self.assertAlmostEqual(agg["burst_peak_amp"], 5.0)
        self.assertGreater(agg["burst_duration_ratio"], 0.0)
        self.assertGreaterEqual(agg["burst_mean_freq"], 40.0)

    def test_aggregate_empty(self):
        agg = self.detector.aggregate_in_window([], 0, 127)
        self.assertEqual(agg["burst_count"], 0)
        self.assertEqual(agg["burst_peak_amp"], 0.0)
        self.assertEqual(agg["burst_duration_ratio"], 0.0)
        self.assertTrue(np.isnan(agg["burst_mean_freq"]))


if __name__ == "__main__":
    unittest.main()
