"""
UI package — Komponen Streamlit.

Modul:
- styles      : CSS dashboard
- sidebar     : Panel konfigurasi sidebar
- single_file : Analisis file EDF tunggal
- batch       : Analisis batch (ZIP)
"""

from ui.styles import inject_css
from ui.sidebar import render_sidebar, init_state
from ui.single_file import render_single_file, render_overview, run_processing, render_results
from ui.batch import run_batch_processing, render_batch_results

__all__ = [
    "inject_css",
    "render_sidebar",
    "init_state",
    "render_single_file",
    "render_overview",
    "run_processing",
    "render_results",
    "run_batch_processing",
    "render_batch_results",
]
