import unittest

import numpy as np

from processing.superlets import SuperletTFR, build_frequency_grid


class TestSuperlets(unittest.TestCase):
    def setUp(self):
        self.sfreq = 128.0
        self.duration_s = 2.0
        self.n_samples = int(self.sfreq * self.duration_s)
        self.t = np.arange(self.n_samples) / self.sfreq

    def test_build_frequency_grid(self):
        linear = build_frequency_grid(30.0, 49.5, 10, spacing="linear")
        self.assertEqual(linear.shape[0], 10)
        self.assertAlmostEqual(float(linear[0]), 30.0)
        self.assertAlmostEqual(float(linear[-1]), 49.5)

        log_grid = build_frequency_grid(30.0, 49.5, 10, spacing="log")
        self.assertEqual(log_grid.shape[0], 10)
        self.assertTrue(np.all(np.diff(log_grid) > 0))

    def test_pure_tone_peaks_near_carrier(self):
        signal = np.sin(2.0 * np.pi * 40.0 * self.t)
        freqs = np.array([30.0, 35.0, 40.0, 45.0, 49.0])

        tfr = SuperletTFR(self.sfreq, freqs, c_base=3, order_min=1, order_max=6)
        power = tfr.compute(signal)

        center = self.n_samples // 2
        peak_freq = float(freqs[np.argmax(power[:, center])])
        self.assertAlmostEqual(peak_freq, 40.0, delta=1.0)

    def test_transient_burst_is_time_localized(self):
        burst_start = int(0.9 * self.sfreq)
        burst_end = int(1.1 * self.sfreq)

        signal = np.zeros_like(self.t)
        signal[burst_start:burst_end] = np.sin(
            2.0 * np.pi * 40.0 * self.t[burst_start:burst_end]
        )

        rng = np.random.default_rng(123)
        signal += 0.25 * rng.standard_normal(self.n_samples)

        freqs = np.array([30.0, 35.0, 40.0, 45.0, 49.0])
        tfr = SuperletTFR(self.sfreq, freqs, c_base=3, order_min=1, order_max=6)
        power = tfr.compute(signal)

        idx_40 = int(np.where(freqs == 40.0)[0][0])
        power_40 = power[idx_40]
        peak_sample = int(np.argmax(power_40))
        peak_time = peak_sample / self.sfreq

        self.assertAlmostEqual(peak_time, 1.0, delta=0.2)

        burst_power = float(np.mean(power_40[burst_start:burst_end]))
        outside_power = float(
            np.mean(np.r_[power_40[: burst_start - 10], power_40[burst_end + 10 :]])
        )
        self.assertGreater(burst_power, outside_power)


if __name__ == "__main__":
    unittest.main()
