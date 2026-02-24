"""
Processing package — backend pemrosesan EEG.

Modul:
- loader   : Load EDF, ZIP, deteksi metadata
- filters  : Bandpass, notch, ICA, bad channel detection
- features : Ekstraksi fitur (time-domain + frequency-domain)
- delta    : Delta antar task
- statistics : Uji statistik (Mann-Whitney, t-test, Cohen's d, FDR)
"""

from processing.loader import EEGLoader
from processing.filters import EEGFilters
from processing.features import EEGFeatures
from processing.delta import DeltaCalculator
from processing.statistics import StatisticalTests

__all__ = [
    "EEGLoader",
    "EEGFilters",
    "EEGFeatures",
    "DeltaCalculator",
    "StatisticalTests",
]
