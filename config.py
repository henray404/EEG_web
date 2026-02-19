"""
Konfigurasi default untuk EEG Analysis Tool.
Aksen warna berdasarkan logo ITS Robocon (biru).
"""

# Subband EEG standar (Hz)
DEFAULT_SUBBANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 49),
}

# Fitur statistik yang tersedia
DEFAULT_FEATURES = ["mav", "variance", "std"]

# Metode ICA yang tersedia
ICA_METHODS = ["fastica", "infomax", "picard"]

# Range order filter
MIN_FILTER_ORDER = 1
MAX_FILTER_ORDER = 10
DEFAULT_FILTER_ORDER = 5

# Notch filter
NOTCH_FREQUENCIES = [50, 60]

# ------------------------------------------------------------------ #
#  Tema warna - ITS Robocon (Blue)                                    #
# ------------------------------------------------------------------ #

# Warna utama dari logo ITS Robocon
ACCENT_PRIMARY = "#0D47A1"       # deep navy blue
ACCENT_SECONDARY = "#1565C0"     # medium blue
ACCENT_LIGHT = "#1E88E5"         # bright blue
ACCENT_LIGHTER = "#42A5F5"       # light blue
ACCENT_PALE = "#90CAF9"          # pale blue

# Dark mode background
BG_DARK = "#0B1120"
BG_CARD_DARK = "#111827"
BG_CARD_HOVER = "#172033"
BG_SIDEBAR = "#0D1525"

# Text
TEXT_PRIMARY = "#F1F5F9"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#64748B"

# Borders
BORDER_DARK = "#1E293B"
BORDER_ACCENT = "#1E3A5F"

# Success/warning/info
COLOR_SUCCESS = "#10B981"
COLOR_WARNING = "#F59E0B"
COLOR_ERROR = "#EF4444"
COLOR_INFO = "#38BDF8"

# Warna channel untuk visualisasi
CHANNEL_COLORS = [
    "#1E88E5", "#42A5F5", "#90CAF9", "#64B5F6",
    "#29B6F6", "#4FC3F7", "#81D4FA", "#03A9F4",
    "#0288D1", "#0277BD", "#01579B", "#039BE5",
    "#00ACC1", "#0097A7", "#00838F", "#006064",
]

# Task colors
TASK_COLORS = {
    "Resting":       "#1E88E5",
    "Thinking":      "#7C3AED",
    "Acting":        "#10B981",
    "Typing":        "#F59E0B",
    "Think_Acting":  "#EC4899",
}

