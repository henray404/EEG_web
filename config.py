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
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 49),
}

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

BG_DARK       = "#0B1120"
BG_CARD_DARK  = "#111827"
BG_CARD_HOVER = "#172033"
BG_SIDEBAR    = "#0D1525"

TEXT_PRIMARY   = "#F1F5F9"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED     = "#64748B"

BORDER_DARK   = "#1E293B"
BORDER_ACCENT = "#1E3A5F"

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
