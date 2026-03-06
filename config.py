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
    "Gamma": (30, 49),
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

TASK_COLORS = {
    "Resting":      "#1E88E5",
    "Thinking":     "#7C3AED",
    "Acting":       "#10B981",
    "Typing":       "#F59E0B",
    "Think_Acting": "#EC4899",
}
