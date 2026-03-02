"""
Modul styles — CSS custom untuk dashboard.
"""

from config import (
    ACCENT_PRIMARY, ACCENT_SECONDARY, ACCENT_LIGHT, ACCENT_LIGHTER,
    ACCENT_PALE, BG_DARK, BG_CARD_DARK, BG_SIDEBAR, BORDER_DARK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
)


CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* === Global === */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {BG_DARK};
}}

/* === Sidebar === */
section[data-testid="stSidebar"] {{
    background: {BG_SIDEBAR};
    border-right: 1px solid {BORDER_DARK};
}}

section[data-testid="stSidebar"] .stMarkdown h1 {{
    font-size: 1.25rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {ACCENT_LIGHT};
    padding-bottom: 10px;
    margin-bottom: 18px;
    letter-spacing: -0.02em;
}}

section[data-testid="stSidebar"] .stMarkdown h3 {{
    font-size: 0.88rem;
    font-weight: 600;
    color: {ACCENT_PRIMARY};
    margin-top: 20px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* === Dashboard metric cards === */
div[data-testid="stMetric"] {{
    background: {BG_CARD_DARK};
    border: 1px solid {BORDER_DARK};
    border-radius: 14px;
    padding: 18px 20px;
    transition: border-color 0.2s;
}}

div[data-testid="stMetric"]:hover {{
    border-color: {ACCENT_LIGHT};
}}

div[data-testid="stMetric"] label {{
    font-weight: 500;
    color: {TEXT_SECONDARY};
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-weight: 800;
    color: {TEXT_PRIMARY};
    font-size: 1.6rem;
}}

/* === Cards / containers === */
div[data-testid="stExpander"] {{
    background: {BG_CARD_DARK};
    border: 1px solid {BORDER_DARK};
    border-radius: 14px;
}}

/* === Tabs === */
button[data-baseweb="tab"] {{
    font-weight: 500;
    font-size: 0.9rem;
}}

/* === Data table === */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER_DARK};
    border-radius: 12px;
    overflow: hidden;
}}

/* === Primary button === */
button[kind="primary"] {{
    background: linear-gradient(135deg, {ACCENT_SECONDARY}, {ACCENT_PRIMARY});
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: all 0.25s;
    letter-spacing: 0.01em;
}}

button[kind="primary"] p {{
    color: #FFFFFF !important;
}}

button[kind="primary"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(13, 71, 161, 0.25);
}}

/* === Secondary / default button === */
button[kind="secondary"] {{
    color: {ACCENT_PRIMARY} !important;
    border-color: {BORDER_DARK};
    font-weight: 600;
}}

button[kind="secondary"] p {{
    color: {ACCENT_PRIMARY} !important;
}}

/* === Download button === */
button[data-testid="stDownloadButton"] button {{
    color: {ACCENT_PRIMARY} !important;
    font-weight: 600;
}}

/* === All button text readable === */
.stButton button p,
.stDownloadButton button p {{
    color: inherit !important;
}}

/* === Custom elements === */
.dashboard-header {{
    font-size: 1.9rem;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.03em;
    margin-bottom: 2px;
}}

.dashboard-subtitle {{
    font-size: 0.92rem;
    color: {TEXT_MUTED};
    margin-bottom: 22px;
}}

.section-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    border-left: 3px solid {ACCENT_LIGHT};
    padding-left: 12px;
    margin: 28px 0 14px 0;
    letter-spacing: -0.01em;
}}

.log-box {{
    background: {BG_CARD_DARK};
    border: 1px solid {BORDER_DARK};
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.83rem;
    color: {TEXT_SECONDARY};
    line-height: 1.8;
}}

.card {{
    background: {BG_CARD_DARK};
    border: 1px solid {BORDER_DARK};
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
}}

.task-badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px 4px;
    color: white;
}}

.info-label {{
    font-size: 0.78rem;
    font-weight: 500;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}}

.info-value {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}
</style>
"""


def inject_css():
    """Inject custom CSS into Streamlit page."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
