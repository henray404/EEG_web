"""
Modul signal_plots — Visualisasi sinyal mentah, PSD, distribusi.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import CHANNEL_COLORS, TASK_COLORS, ACCENT_LIGHT

# Template dasar Plotly untuk dark mode
_TEMPLATE = "plotly_dark"
_PLOT_BG = "rgba(0,0,0,0)"
_PAPER_BG = "rgba(0,0,0,0)"


def _base_layout(**kwargs):
    """Merge layout defaults untuk semua chart."""
    base = dict(
        template=_TEMPLATE,
        plot_bgcolor=_PLOT_BG,
        paper_bgcolor=_PAPER_BG,
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        margin=dict(l=50, r=20, t=44, b=40),
    )
    base.update(kwargs)
    return base


class SignalPlots:
    """Visualisasi sinyal EEG."""

    @staticmethod
    def plot_raw_signal(df, channels, time_range=None, title="Sinyal EEG"):
        """Line chart sinyal EEG per channel."""
        plot_df = df.copy()
        if time_range:
            plot_df = plot_df[
                (plot_df["time"] >= time_range[0]) & (plot_df["time"] <= time_range[1])
            ]

        fig = make_subplots(
            rows=len(channels), cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
        )
        for i, ch in enumerate(channels):
            if ch not in plot_df.columns:
                continue
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=plot_df["time"], y=plot_df[ch], name=ch,
                    line=dict(width=1, color=color),
                ),
                row=i + 1, col=1,
            )
            fig.update_yaxes(
                title_text=ch, row=i + 1, col=1,
                gridcolor="#1E293B", zerolinecolor="#1E293B",
            )

        fig.update_xaxes(
            title_text="Waktu (s)", row=len(channels), col=1,
            gridcolor="#1E293B",
        )
        fig.update_layout(**_base_layout(
            title=title, height=180 * len(channels), showlegend=False,
        ))
        return fig

    @staticmethod
    def plot_task_signal(df, channels, task_name, time_range=None):
        """Plot sinyal untuk satu task tertentu."""
        task_df = df[df["marker"] == task_name].copy()
        if task_df.empty:
            return None
        title = f"Sinyal - {task_name}"
        return SignalPlots.plot_raw_signal(task_df, channels, time_range, title)

    @staticmethod
    def plot_psd(raw, fmin=0, fmax=60, title="Power Spectral Density"):
        """PSD plot dari raw MNE object."""
        spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        psds_db = 10 * np.log10(psds + 1e-20)

        fig = go.Figure()
        for i, ch in enumerate(raw.ch_names):
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            fig.add_trace(go.Scatter(
                x=freqs, y=psds_db[i], name=ch,
                line=dict(width=1.5, color=color),
            ))
        fig.update_layout(**_base_layout(
            title=title, xaxis_title="Frekuensi (Hz)",
            yaxis_title="Power (dB)", height=380,
        ))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    @staticmethod
    def plot_signal_distribution(df, channels, title="Distribusi Amplitudo"):
        """Histogram distribusi amplitudo per channel."""
        cols = [c for c in channels if c in df.columns]
        if not cols:
            return None
        fig = go.Figure()
        for i, ch in enumerate(cols):
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            fig.add_trace(go.Histogram(
                x=df[ch], name=ch, opacity=0.65,
                marker_color=color, nbinsx=80,
            ))
        fig.update_layout(**_base_layout(
            barmode="overlay", title=title, height=380,
            xaxis_title="Amplitudo", yaxis_title="Frekuensi",
        ))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    @staticmethod
    def plot_channel_correlation(df, channels, title="Korelasi Antar Channel"):
        """Heatmap korelasi antar channel."""
        cols = [c for c in channels if c in df.columns]
        if len(cols) < 2:
            return None
        corr = df[cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="Blues",
            zmin=-1, zmax=1,
            labels=dict(color="Korelasi"),
        )
        fig.update_layout(**_base_layout(title=title, height=420))
        return fig

    @staticmethod
    def plot_annotation_summary(annotations, title="Distribusi Marker"):
        """Bar chart jumlah kemunculan setiap annotation."""
        if not annotations:
            return None
        descs = [a["description"] for a in annotations]
        unique, counts = np.unique(descs, return_counts=True)
        colors = [TASK_COLORS.get(u, ACCENT_LIGHT) for u in unique]
        fig = go.Figure(go.Bar(
            x=unique, y=counts, marker_color=colors,
            marker_line_width=0, text=counts, textposition="outside",
        ))
        fig.update_layout(**_base_layout(
            title=title, height=350, showlegend=False,
            xaxis_title="Marker", yaxis_title="Jumlah",
        ))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig
