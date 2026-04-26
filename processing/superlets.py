"""Superlet time-frequency transform utilities.

Reference:
Moca et al. (2021), Time-frequency super-resolution with superlets.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


class SuperletTFR:
    """Fractional Adaptive Superlet Transform (FASLT) on 1D signals."""

    def __init__(
        self,
        sfreq: float,
        freqs: np.ndarray,
        c_base: int = 3,
        order_min: float = 1.0,
        order_max: float = 6.0,
    ) -> None:
        self.sfreq = float(sfreq)
        self.freqs = np.asarray(freqs, dtype=float)
        self.c_base = int(c_base)
        self.order_min = float(order_min)
        self.order_max = float(order_max)

        if self.sfreq <= 0:
            raise ValueError("sfreq must be > 0")
        if self.freqs.ndim != 1 or self.freqs.size == 0:
            raise ValueError("freqs must be a non-empty 1D array")
        if np.any(self.freqs <= 0):
            raise ValueError("all frequencies must be > 0")
        if self.c_base < 1:
            raise ValueError("c_base must be >= 1")
        if self.order_min < 1 or self.order_max < self.order_min:
            raise ValueError("order range must satisfy 1 <= order_min <= order_max")

    @staticmethod
    def _morlet(sfreq: float, freq: float, cycles: int) -> np.ndarray:
        """Build an L2-normalized complex Morlet kernel."""
        sigma = cycles / (2.0 * np.pi * freq)
        half_len = int(np.ceil(4.0 * sigma * sfreq))
        t = np.arange(-half_len, half_len + 1, dtype=float) / sfreq

        envelope = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
        carrier = np.exp(1j * 2.0 * np.pi * freq * t)
        wavelet = envelope * carrier
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
        return wavelet

    def _wavelet_power(self, signal: np.ndarray, freq: float, cycles: int) -> np.ndarray:
        """Return |x * psi|^2(t) for one frequency/cycle pair."""
        kernel = self._morlet(self.sfreq, freq, cycles)
        conv = fftconvolve(signal, kernel, mode="same")
        return np.abs(conv) ** 2

    def _integer_order_geomean(
        self,
        signal: np.ndarray,
        freq: float,
        order: int,
    ) -> np.ndarray:
        """Geometric mean response for integer superlet order."""
        tiny = np.finfo(float).tiny
        log_sum = np.zeros(signal.size, dtype=float)

        for k in range(1, order + 1):
            cycles = self.c_base * k
            power = self._wavelet_power(signal, freq, cycles)
            log_sum += np.log(np.maximum(power, tiny))

        return np.exp(log_sum / float(order))

    def _faslt_at_freq(self, signal: np.ndarray, freq: float, order: float) -> np.ndarray:
        """Fractional-order superlet response at one frequency."""
        k = int(np.floor(order))
        k = max(k, 1)
        r = float(order - k)

        gm_k = self._integer_order_geomean(signal, freq, k)
        if r <= 0.0:
            return gm_k

        gm_k1 = self._integer_order_geomean(signal, freq, k + 1)
        tiny = np.finfo(float).tiny
        return np.exp(
            (1.0 - r) * np.log(np.maximum(gm_k, tiny))
            + r * np.log(np.maximum(gm_k1, tiny))
        )

    def _adaptive_orders(self) -> np.ndarray:
        """Map frequency bins to adaptive fractional orders."""
        if self.freqs.size == 1:
            return np.array([self.order_max], dtype=float)
        return np.linspace(self.order_min, self.order_max, self.freqs.size)

    def compute(self, signal: np.ndarray) -> np.ndarray:
        """Compute Superlet power map.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples,)
            Single-channel time series.

        Returns
        -------
        np.ndarray, shape (n_freqs, n_samples)
            Superlet power map.
        """
        x = np.asarray(signal, dtype=float)
        if x.ndim != 1:
            raise ValueError("signal must be a 1D array")
        if x.size < 4:
            raise ValueError("signal is too short for superlet computation")

        power = np.empty((self.freqs.size, x.size), dtype=float)
        orders = self._adaptive_orders()
        for i, (freq, order) in enumerate(zip(self.freqs, orders)):
            power[i, :] = self._faslt_at_freq(x, freq, order)
        return power


def build_frequency_grid(
    low: float,
    high: float,
    n_freqs: int,
    spacing: str = "linear",
) -> np.ndarray:
    """Build frequency grid for time-frequency transforms."""
    low = float(low)
    high = float(high)
    if n_freqs < 2:
        raise ValueError("n_freqs must be >= 2")
    if low <= 0 or high <= low:
        raise ValueError("frequency bounds must satisfy 0 < low < high")

    spacing_norm = spacing.lower().strip()
    if spacing_norm == "linear":
        return np.linspace(low, high, n_freqs)
    if spacing_norm == "log":
        return np.geomspace(low, high, n_freqs)

    raise ValueError("spacing must be 'linear' or 'log'")


def compute_faslt(
    signal: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    c_base: int = 3,
    order_min: float = 1.0,
    order_max: float = 6.0,
) -> np.ndarray:
    """Convenience wrapper for one-shot FASLT computation."""
    tfr = SuperletTFR(
        sfreq=sfreq,
        freqs=freqs,
        c_base=c_base,
        order_min=order_min,
        order_max=order_max,
    )
    return tfr.compute(signal)
