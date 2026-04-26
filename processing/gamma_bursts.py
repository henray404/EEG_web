"""Gamma burst detection from Superlet time-frequency power maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import (
    BURST_MAD_THRESHOLD,
    BURST_MERGE_GAP_MS,
    BURST_MIN_DURATION_MS,
)


@dataclass
class Burst:
    """A single burst interval and its summary statistics."""

    start_sample: int
    end_sample: int
    peak_amp: float
    mean_freq: float

    def duration_s(self, sfreq: float) -> float:
        return float((self.end_sample - self.start_sample + 1) / sfreq)


class GammaBurstDetector:
    """Detect gamma bursts from TFR power and aggregate per window."""

    def __init__(
        self,
        sfreq: float,
        freqs: np.ndarray,
        mad_k: float = BURST_MAD_THRESHOLD,
        min_duration_ms: float = BURST_MIN_DURATION_MS,
        merge_gap_ms: float = BURST_MERGE_GAP_MS,
    ) -> None:
        self.sfreq = float(sfreq)
        self.freqs = np.asarray(freqs, dtype=float)
        self.mad_k = float(mad_k)

        if self.sfreq <= 0:
            raise ValueError("sfreq must be > 0")
        if self.freqs.ndim != 1 or self.freqs.size == 0:
            raise ValueError("freqs must be a non-empty 1D array")

        self.min_dur_samples = max(1, int(round((min_duration_ms / 1000.0) * self.sfreq)))
        self.merge_gap_samples = max(0, int(round((merge_gap_ms / 1000.0) * self.sfreq)))

    def detect(self, power: np.ndarray) -> List[Burst]:
        """Detect bursts from power map with shape (n_freqs, n_samples)."""
        p = np.asarray(power, dtype=float)
        if p.ndim != 2 or p.shape[0] != self.freqs.size:
            raise ValueError("power must have shape (n_freqs, n_samples)")
        if p.shape[1] == 0:
            return []

        p_gamma = p.mean(axis=0)
        med = float(np.median(p_gamma))
        mad = float(np.median(np.abs(p_gamma - med)))

        robust_scale = 1.4826 * mad
        if robust_scale == 0.0:
            robust_scale = float(np.std(p_gamma))

        threshold = med + self.mad_k * robust_scale
        if robust_scale == 0.0:
            threshold = med

        mask = p_gamma > threshold
        intervals = self._contiguous_true(mask)
        intervals = self._merge_close(intervals, self.merge_gap_samples)
        intervals = [
            (start, end)
            for (start, end) in intervals
            if (end - start + 1) >= self.min_dur_samples
        ]

        bursts: List[Burst] = []
        for start, end in intervals:
            segment = p_gamma[start : end + 1]
            peak_amp = float(np.max(segment))

            # Mean carrier frequency over burst by per-sample argmax frequency.
            freq_idx = np.argmax(p[:, start : end + 1], axis=0)
            mean_freq = float(np.mean(self.freqs[freq_idx]))
            bursts.append(
                Burst(
                    start_sample=int(start),
                    end_sample=int(end),
                    peak_amp=peak_amp,
                    mean_freq=mean_freq,
                )
            )

        return bursts

    @staticmethod
    def _contiguous_true(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Get inclusive index intervals where boolean mask is True."""
        if mask.size == 0 or not np.any(mask):
            return []

        diff = np.diff(mask.astype(int))
        starts = list(np.where(diff == 1)[0] + 1)
        ends = list(np.where(diff == -1)[0])

        if mask[0]:
            starts.insert(0, 0)
        if mask[-1]:
            ends.append(mask.size - 1)

        return list(zip(starts, ends))

    @staticmethod
    def _merge_close(intervals: List[Tuple[int, int]], gap_samples: int) -> List[Tuple[int, int]]:
        if not intervals:
            return []

        merged: List[Tuple[int, int]] = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if (start - prev_end - 1) <= gap_samples:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged

    def aggregate_in_window(
        self,
        bursts: List[Burst],
        window_start_sample: int,
        window_end_sample: int,
    ) -> Dict[str, float]:
        """Aggregate burst features for a sample interval."""
        if window_end_sample < window_start_sample:
            raise ValueError("window_end_sample must be >= window_start_sample")

        overlapping = [
            b
            for b in bursts
            if b.end_sample >= window_start_sample and b.start_sample <= window_end_sample
        ]

        if not overlapping:
            return {
                "burst_count": 0,
                "burst_peak_amp": 0.0,
                "burst_duration_ratio": 0.0,
                "burst_mean_freq": float("nan"),
            }

        win_len = window_end_sample - window_start_sample + 1
        covered_samples = 0
        for burst in overlapping:
            clipped_start = max(burst.start_sample, window_start_sample)
            clipped_end = min(burst.end_sample, window_end_sample)
            covered_samples += clipped_end - clipped_start + 1

        return {
            "burst_count": int(len(overlapping)),
            "burst_peak_amp": float(max(b.peak_amp for b in overlapping)),
            "burst_duration_ratio": float(covered_samples / win_len),
            "burst_mean_freq": float(np.mean([b.mean_freq for b in overlapping])),
        }
