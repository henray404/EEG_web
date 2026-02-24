"""
Modul feature_plots — Visualisasi distribusi dan perbandingan fitur.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import CHANNEL_COLORS, TASK_COLORS, ACCENT_LIGHT

_TEMPLATE = "plotly_dark"
_PLOT_BG = "rgba(0,0,0,0)"
_PAPER_BG = "rgba(0,0,0,0)"


def _base_layout(**kwargs):
    base = dict(
        template=_TEMPLATE,
        plot_bgcolor=_PLOT_BG,
        paper_bgcolor=_PAPER_BG,
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        margin=dict(l=50, r=20, t=44, b=40),
    )
    base.update(kwargs)
    return base


class FeaturePlots:
    """Visualisasi fitur EEG: bar, box, pie, grouped bar."""

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
            labels={
                "subband": "Subband",
                feature_name: feature_name.capitalize(),
                "channel": "Channel",
            },
        )
        fig.update_layout(**_base_layout(title=title, height=380))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    @staticmethod
    def plot_task_feature_comparison(task_features_df, feature_name,
                                      title=None, task_col="task"):
        """Grouped bar chart fitur per task, channel, dan subband."""
        if task_features_df.empty or feature_name not in task_features_df.columns:
            return None
        if task_col not in task_features_df.columns:
            task_col = "task"
        if title is None:
            title = f"{feature_name.capitalize()} per Task & Subband"

        fig = px.bar(
            task_features_df, x="subband", y=feature_name,
            color=task_col, barmode="group",
            facet_col="channel",
            color_discrete_map=TASK_COLORS,
            labels={
                "subband": "Subband",
                feature_name: feature_name.capitalize(),
                task_col: "Task",
                "channel": "Channel",
            },
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
        colors = [
            TASK_COLORS.get(t, ACCENT_LIGHT) for t in task_summary_df["task"]
        ]
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

    @staticmethod
    def plot_batch_overview(batch_df, feature_name, title=None):
        """Box plot fitur per task di seluruh dataset."""
        if batch_df.empty or feature_name not in batch_df.columns:
            return None
        if title is None:
            title = f"Distribusi {feature_name.capitalize()} per Task (Batch)"
        fig = px.box(
            batch_df, x="task", y=feature_name, color="channel",
            color_discrete_sequence=CHANNEL_COLORS,
            labels={
                "task": "Task",
                feature_name: feature_name.capitalize(),
                "channel": "Channel",
            },
            points="outliers",
        )
        fig.update_layout(**_base_layout(title=title, height=440))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    @staticmethod
    def plot_grouped_bar(df, feature_name, group_col="task", facet_col="channel",
                         x_col="subband", title=None):
        """Grouped bar chart faceted by channel."""
        if df.empty or feature_name not in df.columns:
            return None
        if title is None:
            title = (
                f"{feature_name.capitalize()} per "
                f"{group_col.capitalize()} & {x_col.capitalize()}"
            )

        agg = (df.groupby([facet_col, x_col, group_col])[feature_name]
               .mean().reset_index())

        facets = sorted(agg[facet_col].unique())
        groups = sorted(agg[group_col].unique())

        palette = [
            "#60A5FA", "#A78BFA", "#F59E0B", "#10B981",
            "#F472B6", "#34D399", "#FB923C", "#818CF8",
        ]
        color_map = {}
        for i, g in enumerate(groups):
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
            title=title, height=450, barmode="group",
            legend=dict(title=group_col.capitalize()),
        ))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        if facets:
            fig.update_yaxes(title_text=feature_name, row=1, col=1)
        return fig

    @staticmethod
    def plot_band_ratios(ratios_df, title="Rasio Antar Subband"):
        """Bar chart rasio antar subband per channel.

        Parameters
        ----------
        ratios_df : pd.DataFrame
            Output dari EEGFeatures.compute_band_ratios().
        """
        if ratios_df.empty:
            return None
        fig = px.bar(
            ratios_df, x="ratio_name", y="value",
            color="channel", barmode="group",
            color_discrete_sequence=CHANNEL_COLORS,
            labels={
                "ratio_name": "Rasio",
                "value": "Nilai Rasio",
                "channel": "Channel",
            },
        )
        fig.update_layout(**_base_layout(title=title, height=400))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig
