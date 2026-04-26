"""
Processing package — backend pemrosesan EEG.

Modul:
- loader     : Load EDF, ZIP, deteksi metadata
- filters    : Bandpass, notch, ICA, bad channel detection
- features   : Ekstraksi fitur (time-domain + frequency-domain)
- psd        : Analisis Power Spectral Density (Welch / Multitaper)
- epoching   : Epoching & Sliding Windows
- connectivity: Konektivitas fungsional (PLI / wPLI)
- delta      : Delta antar task
- statistics : Uji statistik (Mann-Whitney, t-test, Cohen's d, FDR)
"""

from processing.loader import EEGLoader
from processing.filters import EEGFilters
from processing.features import EEGFeatures
from processing.psd import PSDAnalyzer
from processing.epoching import EpochEngine
from processing.connectivity import ConnectivityAnalyzer
from processing.delta import DeltaCalculator
from processing.statistics import StatisticalTests
from processing.superlets import SuperletTFR
from processing.gamma_bursts import GammaBurstDetector

__all__ = [
    "EEGLoader",
    "EEGFilters",
    "EEGFeatures",
    "PSDAnalyzer",
    "EpochEngine",
    "ConnectivityAnalyzer",
    "DeltaCalculator",
    "StatisticalTests",
    "SuperletTFR",
    "GammaBurstDetector",
]


