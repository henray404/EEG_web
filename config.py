"""
Konfigurasi default untuk EEG Analysis Tool.
Aksen warna berdasarkan logo ITS Robocon (biru).
"""

# ------------------------------------------------------------------ #
#  Subband & Fitur                                                    #
# ------------------------------------------------------------------ #

# Subband EEG standar (Hz)
DEFAULT_SUBBANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Mu": (8, 12),
    "Alpha": (8, 13),
    "Low_Beta": (12, 16),
    "High_Beta": (20, 30),
    "Beta": (13, 30),
    "Gamma": (30, 49.5),
}

# Threshold amplitudo EEG (µV) — sinyal di atas ini dianggap artefak
AMPLITUDE_MAX_UV = 100.0

# Fitur statistik dasar (time‑domain)
DEFAULT_FEATURES = ["mav", "variance", "std"]

# Fitur frekuensi‑domain (baru dari pipeline)
FREQUENCY_FEATURES = ["band_power", "relative_power", "peak_frequency"]

# Rasio antar subband (baru dari pipeline)
BAND_RATIOS = {
    "alpha_beta":  ("Alpha", "Beta"),
    "theta_alpha": ("Theta", "Alpha"),
    "delta_theta": ("Delta", "Theta"),
}

# Transisi task default untuk analisis delta
DEFAULT_TRANSITIONS = [
    ("Resting", "Thinking"),
    ("Resting", "Typing"),
    ("Thinking", "Think_Acting"),
    ("Resting", "Think_Acting"),
]

# ------------------------------------------------------------------ #
#  Filter & ICA                                                       #
# ------------------------------------------------------------------ #

ICA_METHODS = ["fastica", "infomax", "picard"]
MIN_FILTER_ORDER = 1
MAX_FILTER_ORDER = 10
DEFAULT_FILTER_ORDER = 5
NOTCH_FREQUENCIES = [50, 60]

# Bad channel detection threshold (MAD multiplier)
BAD_CHANNEL_THRESHOLD = 3.0

# ------------------------------------------------------------------ #
#  PSD (Power Spectral Density)                                       #
# ------------------------------------------------------------------ #

PSD_METHODS = ["welch", "multitaper"]
DEFAULT_PSD_METHOD = "welch"
DEFAULT_PSD_FMIN = 0.0
DEFAULT_PSD_FMAX = 49.5
DEFAULT_PSD_N_FFT = None   # None = auto (2 * sfreq, capped by data length)

# ------------------------------------------------------------------ #
#  Epoching & Sliding Windows                                         #
# ------------------------------------------------------------------ #

DEFAULT_EPOCH_DURATION = 2.0    # detik
MIN_EPOCH_DURATION = 0.5
MAX_EPOCH_DURATION = 10.0

DEFAULT_WINDOW_SIZE = 2.0       # detik
DEFAULT_WINDOW_OVERLAP = 0.5    # 50% overlap (rasio 0–0.75)
MIN_WINDOW_SIZE = 0.1           # diturunkan dari 0.5 agar support 0.3s encoding
MAX_WINDOW_SIZE = 10.0

# ------------------------------------------------------------------ #
#  Encoding (Batch Feature Extraction)                                #
# ------------------------------------------------------------------ #

DEFAULT_ENCODING_WINDOW = 0.3   # detik per window untuk encoding
DEFAULT_ENCODING_OVERLAP = 0.0  # tanpa overlap (baseline)
EEGET_ALS_SFREQ = 128.0        # sampling frequency EEGET-ALS dataset
EEGET_ALS_N_CHANNELS = 32      # jumlah channel EEGET-ALS dataset

# Superlet TFR (Gamma burst extraction)
SUPERLET_C_BASE = 3
SUPERLET_ORDER_MIN = 1
SUPERLET_ORDER_MAX = 6
SUPERLET_N_FREQS = 10
SUPERLET_FREQ_SPACING = "linear"  # "linear" | "log"

# Gamma burst detection
GAMMA_WINDOW_SECONDS = 1.0
BURST_MAD_THRESHOLD = 2.0
BURST_MIN_DURATION_MS = 25.0
BURST_MERGE_GAP_MS = 25.0
BURST_ENABLE_DEFAULT = False

# Mapping skenario EEGET-ALS ke label numerik
EEGET_ALS_SCENARIOS = {
    1: "Lifting left hand",
    2: "Lifting right hand",
    3: "Lifting left leg",
    4: "Lifting right leg",
    5: "Opening mouth",
    6: "Nodding head",
    7: "Shaking head",
    8: "Desire to drink water",
    9: "Desire to use bathroom",
}

DEFAULT_EPOCH_REJECT_UV = 100.0  # threshold rejection epoch (µV)

# ------------------------------------------------------------------ #
#  Connectivity (PLI / wPLI)                                          #
# ------------------------------------------------------------------ #

CONNECTIVITY_METHODS = ["pli", "wpli"]
DEFAULT_CONNECTIVITY_METHOD = "wpli"

# ------------------------------------------------------------------ #
#  Tema warna - ITS Robocon (Blue)                                    #
# ------------------------------------------------------------------ #

ACCENT_PRIMARY   = "#0D47A1"
ACCENT_SECONDARY = "#1565C0"
ACCENT_LIGHT     = "#1E88E5"
ACCENT_LIGHTER   = "#42A5F5"
ACCENT_PALE      = "#90CAF9"

BG_DARK       = "#F5F8FF"
BG_CARD_DARK  = "#FFFFFF"
BG_CARD_HOVER = "#EDF2FF"
BG_SIDEBAR    = "#E8EEFF"

TEXT_PRIMARY   = "#0F172A"
TEXT_SECONDARY = "#475569"
TEXT_MUTED     = "#64748B"

BORDER_DARK   = "#CBD5E1"
BORDER_ACCENT = "#93C5FD"

COLOR_SUCCESS = "#10B981"
COLOR_WARNING = "#F59E0B"
COLOR_ERROR   = "#EF4444"
COLOR_INFO    = "#38BDF8"

CHANNEL_COLORS = [
    "#1E88E5", "#42A5F5", "#90CAF9", "#64B5F6",
    "#29B6F6", "#4FC3F7", "#81D4FA", "#03A9F4",
    "#0288D1", "#0277BD", "#01579B", "#039BE5",
    "#00ACC1", "#0097A7", "#00838F", "#006064",
]

# ------------------------------------------------------------------ #
#  OpenBCI Cyton                                                       #
# ------------------------------------------------------------------ #

OPENBCI_CHANNEL_MAP = {
    "EXG Channel 2": "T3",
    "EXG Channel 3": "T4",
    "EXG Channel 4": "T5",
    "EXG Channel 5": "T6",
    "EXG Channel 6": "O1",
    "EXG Channel 7": "O2",
}
OPENBCI_SFREQ = 250.0

# Regex patterns untuk deteksi kondisi dari nama file OpenBCI
OPENBCI_CONDITIONS = ["baseline", "familiar", "unfamiliar", "nonfamiliar"]

TASK_COLORS = {
    "Resting":      "#1E88E5",
    "Thinking":     "#7C3AED",
    "Acting":       "#10B981",
    "Typing":       "#F59E0B",
    "Think_Acting": "#EC4899",
}
