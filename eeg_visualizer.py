"""
Modul visualisasi EEG menggunakan Plotly.
Semua method bersifat static, menerima data dan mengembalikan figure Plotly.
Chart menggunakan dark theme sesuai desain dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import CHANNEL_COLORS, TASK_COLORS, ACCENT_LIGHT, ACCENT_PALE, BG_CARD_DARK

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


class EEGVisualizer:
    """Kumpulan static method untuk membuat chart Plotly dari data EEG."""

    @staticmethod
    def plot_raw_signal(df, channels, time_range=None, title="Sinyal EEG"):
        """Line chart sinyal EEG per channel."""
        plot_df = df.copy()
        if time_range:
            plot_df = plot_df[(plot_df["time"] >= time_range[0]) & (plot_df["time"] <= time_range[1])]

        fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True,
                            vertical_spacing=0.03)
        for i, ch in enumerate(channels):
            if ch not in plot_df.columns:
                continue
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            fig.add_trace(
                go.Scatter(x=plot_df["time"], y=plot_df[ch], name=ch,
                           line=dict(width=1, color=color)),
                row=i + 1, col=1,
            )
            fig.update_yaxes(title_text=ch, row=i + 1, col=1,
                             gridcolor="#1E293B", zerolinecolor="#1E293B")

        fig.update_xaxes(title_text="Waktu (s)", row=len(channels), col=1,
                         gridcolor="#1E293B")
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
        return EEGVisualizer.plot_raw_signal(task_df, channels, time_range, title)

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
        """Bar chart jumlah kemunculan setiap deskripsi annotation."""
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
    def plot_feature_comparison(features_df, feature_name, title=None):
        """Grouped bar chart fitur per channel dan subband."""
        if feature_name not in features_df.columns:
            return None
        if title is None:
            title = f"Perbandingan {feature_name.capitalize()} per Channel & Subband"
        fig = px.bar(
            features_df, x="subband", y=feature_name,
            color="channel", barmode="group",
            color_discrete_sequence=CHANNEL_COLORS,
            labels={"subband": "Subband", feature_name: feature_name.capitalize(),
                    "channel": "Channel"},
        )
        fig.update_layout(**_base_layout(title=title, height=380))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    @staticmethod
    def plot_task_feature_comparison(task_features_df, feature_name, title=None):
        """Grouped bar chart fitur per task, channel, dan subband."""
        if task_features_df.empty or feature_name not in task_features_df.columns:
            return None
        if title is None:
            title = f"{feature_name.capitalize()} per Task & Subband"

        fig = px.bar(
            task_features_df, x="subband", y=feature_name,
            color="task", barmode="group",
            facet_col="channel",
            color_discrete_map=TASK_COLORS,
            labels={"subband": "Subband", feature_name: feature_name.capitalize(),
                    "task": "Task", "channel": "Channel"},
        )
        fig.update_layout(**_base_layout(title=title, height=420))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    @staticmethod
    def plot_task_pie(task_summary_df, title="Proporsi Task"):
        """Pie chart proporsi waktu per task."""
        if task_summary_df.empty:
            return None
        colors = [TASK_COLORS.get(t, ACCENT_LIGHT) for t in task_summary_df["task"]]
        fig = go.Figure(go.Pie(
            labels=task_summary_df["task"],
            values=task_summary_df["jumlah_sample"],
            marker=dict(colors=colors),
            hole=0.45,
            textinfo="label+percent",
            textfont=dict(size=13),
        ))
        fig.update_layout(**_base_layout(title=title, height=360, showlegend=True))
        return fig

    # ------------------------------------------------------------------ #
    #  Batch / Delta visualizations                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_batch_overview(batch_df, feature_name, title=None):
        """Box plot fitur per task di seluruh dataset (semua file)."""
        if batch_df.empty or feature_name not in batch_df.columns:
            return None
        if title is None:
            title = f"Distribusi {feature_name.capitalize()} per Task (Batch)"
        fig = px.box(
            batch_df, x="task", y=feature_name, color="channel",
            color_discrete_sequence=CHANNEL_COLORS,
            labels={"task": "Task", feature_name: feature_name.capitalize(),
                    "channel": "Channel"},
            points="outliers",
        )
        fig.update_layout(**_base_layout(title=title, height=440))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    @staticmethod
    def plot_delta_bar(agg_df, feature_name, task_a, task_b, title=None):
        """Bar chart delta agregat per channel/subband."""
        col_mean = f"mean_delta_{feature_name}"
        col_std = f"std_delta_{feature_name}"
        if agg_df.empty or col_mean not in agg_df.columns:
            return None
        if title is None:
            title = f"Δ {feature_name.capitalize()} ({task_a} – {task_b})"

        plot_df = agg_df.copy()
        plot_df["label"] = plot_df["channel"] + " / " + plot_df["subband"]
        plot_df = plot_df.sort_values(col_mean, key=abs, ascending=False)

        colors = [CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                  for i in range(len(plot_df))]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_df["label"], y=plot_df[col_mean],
            error_y=dict(type="data", array=plot_df[col_std].abs().tolist(),
                         visible=True) if col_std in plot_df.columns else None,
            marker_color=colors,
            text=[f"{v:+.2e}" for v in plot_df[col_mean]],
            textposition="outside",
        ))
        fig.update_layout(**_base_layout(
            title=title, height=440, showlegend=False,
            xaxis_title="Channel / Subband", yaxis_title=f"Δ {feature_name}",
        ))
        fig.update_xaxes(gridcolor="#1E293B", tickangle=-45)
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    @staticmethod
    def plot_delta_heatmap(agg_df, feature_name, task_a, task_b, title=None):
        """Heatmap delta mean per channel (rows) x subband (cols)."""
        col_mean = f"mean_delta_{feature_name}"
        if agg_df.empty or col_mean not in agg_df.columns:
            return None
        if title is None:
            title = f"Heatmap Δ {feature_name.capitalize()} ({task_a} – {task_b})"

        pivot = agg_df.pivot_table(
            index="channel", columns="subband", values=col_mean,
        )
        fig = px.imshow(
            pivot, text_auto=".2e",
            color_continuous_scale="RdBu_r",
            labels=dict(x="Subband", y="Channel", color=f"Δ {feature_name}"),
        )
        fig.update_layout(**_base_layout(title=title, height=400))
        return fig

    @staticmethod
    def plot_per_file_delta(delta_df, feature_name, task_a, task_b, title=None):
        """Bar chart delta per file per channel/subband (top N)."""
        col = f"delta_{feature_name}"
        if delta_df.empty or col not in delta_df.columns:
            return None
        if title is None:
            title = f"Top 20 Δ {feature_name.capitalize()} per File ({task_a} – {task_b})"

        plot_df = delta_df.copy()
        plot_df["label"] = (plot_df["filename"].str.replace(".edf", "", case=False)
                            + " | " + plot_df["channel"] + "_" + plot_df["subband"])
        plot_df = plot_df.reindex(plot_df[col].abs().sort_values(ascending=False).index)
        plot_df = plot_df.head(20)

        colors = ["#EF4444" if v < 0 else "#10B981" for v in plot_df[col]]

        fig = go.Figure(go.Bar(
            y=plot_df["label"], x=plot_df[col],
            orientation="h", marker_color=colors,
            text=[f"{v:+.2e}" for v in plot_df[col]],
            textposition="outside",
        ))
        fig.update_layout(**_base_layout(
            title=title, height=max(400, 26 * len(plot_df)),
            showlegend=False, yaxis=dict(autorange="reversed"),
            xaxis_title=f"Δ {feature_name}",
        ))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    # ------------------------------------------------------------------ #
    #  ALS vs Normal Visualizations                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_grouped_bar(df, feature_name, group_col="task", facet_col="channel",
                         x_col="subband", title=None):
        """Grouped bar chart faceted by channel (style seperti screenshot user).

        Parameters
        ----------
        df : pd.DataFrame
        feature_name : str  kolom fitur untuk y-axis
        group_col : str  kolom untuk warna/group (default 'task')
        facet_col : str  kolom untuk facet (default 'channel')
        x_col : str  kolom untuk x-axis (default 'subband')
        """
        if df.empty or feature_name not in df.columns:
            return None
        if title is None:
            title = f"{feature_name.capitalize()} per {group_col.capitalize()} & {x_col.capitalize()}"

        # Aggregate: mean per group
        agg = df.groupby([facet_col, x_col, group_col])[feature_name].mean().reset_index()

        facets = sorted(agg[facet_col].unique())
        groups = sorted(agg[group_col].unique())

        # Color palette
        palette = ["#60A5FA", "#A78BFA", "#F59E0B", "#10B981",
                   "#F472B6", "#34D399", "#FB923C", "#818CF8"]
        color_map = {}
        for i, g in enumerate(groups):
            # Use TASK_COLORS if available
            color_map[g] = TASK_COLORS.get(g, palette[i % len(palette)])

        fig = make_subplots(
            rows=1, cols=len(facets),
            subplot_titles=[f"{facet_col}={f}" for f in facets],
            shared_yaxes=True,
        )

        for col_idx, facet_val in enumerate(facets, 1):
            facet_data = agg[agg[facet_col] == facet_val]
            for g in groups:
                g_data = facet_data[facet_data[group_col] == g]
                fig.add_trace(go.Bar(
                    x=g_data[x_col], y=g_data[feature_name],
                    name=g, marker_color=color_map[g],
                    legendgroup=g,
                    showlegend=(col_idx == 1),
                ), row=1, col=col_idx)

        fig.update_layout(**_base_layout(
            title=title,
            height=450,
            barmode="group",
            legend=dict(title=group_col.capitalize()),
        ))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        if facets:
            fig.update_yaxes(title_text=feature_name, row=1, col=1)
        return fig

    @staticmethod
    def plot_als_vs_normal(stats_df, feature_name, active_task, baseline_task,
                           title=None):
        """Grouped bar ALS vs Normal per channel/subband + p-value annotations.

        Parameters
        ----------
        stats_df : pd.DataFrame
            Output dari compare_als_vs_normal (kolom: channel, subband,
            mean_als_{feat}, mean_normal_{feat}, p_{feat}).
        feature_name : str
        """
        col_als = f"mean_als_{feature_name}"
        col_norm = f"mean_normal_{feature_name}"
        col_p = f"p_{feature_name}"

        if stats_df.empty or col_als not in stats_df.columns:
            return None

        if title is None:
            title = (f"Δ {feature_name.capitalize()}: ALS vs Normal "
                     f"({active_task} − {baseline_task})")

        channels = sorted(stats_df["channel"].unique())

        fig = make_subplots(
            rows=1, cols=len(channels),
            subplot_titles=[f"Channel={ch}" for ch in channels],
            shared_yaxes=True,
        )

        for col_idx, ch in enumerate(channels, 1):
            ch_data = stats_df[stats_df["channel"] == ch].sort_values("subband")

            # ALS bars
            fig.add_trace(go.Bar(
                x=ch_data["subband"], y=ch_data[col_als],
                name="ALS", marker_color="#EF4444",
                legendgroup="ALS", showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

            # Normal bars
            fig.add_trace(go.Bar(
                x=ch_data["subband"], y=ch_data[col_norm],
                name="Normal", marker_color="#60A5FA",
                legendgroup="Normal", showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

            # P-value annotations
            if col_p in ch_data.columns:
                for _, row in ch_data.iterrows():
                    p = row[col_p]
                    if pd.notna(p):
                        max_y = max(abs(row[col_als] or 0), abs(row[col_norm] or 0))
                        star = "★" if p <= 0.05 else f"p={p:.3f}"
                        fig.add_annotation(
                            x=row["subband"],
                            y=max_y * 1.15,
                            text=star,
                            showarrow=False,
                            font=dict(
                                size=12,
                                color="#FBBF24" if p <= 0.05 else "#94A3B8",
                            ),
                            row=1, col=col_idx,
                        )

        fig.update_layout(**_base_layout(
            title=title,
            height=480,
            barmode="group",
        ))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        if channels:
            fig.update_yaxes(title_text=f"Mean Δ {feature_name}", row=1, col=1)
        return fig
