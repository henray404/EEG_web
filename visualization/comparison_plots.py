"""
Modul comparison_plots — Delta bar/scatter/heatmap, ALS vs Normal.

Enhanced: Mendukung SEM error bars dan t-test annotation dari pipeline.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import CHANNEL_COLORS, ACCENT_LIGHT

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


class ComparisonPlots:
    """Visualisasi perbandingan: delta, ALS vs Normal."""

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

        colors = [
            CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            for i in range(len(plot_df))
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_df["label"], y=plot_df[col_mean],
            error_y=(
                dict(type="data", array=plot_df[col_std].abs().tolist(), visible=True)
                if col_std in plot_df.columns else None
            ),
            marker_color=colors,
            text=[f"{v:+.2e}" for v in plot_df[col_mean]],
            textposition="outside",
        ))
        fig.update_layout(**_base_layout(
            title=title, height=440, showlegend=False,
            xaxis_title="Channel / Subband",
            yaxis_title=f"Δ {feature_name}",
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
            pivot, text_auto=".2e", color_continuous_scale="RdBu_r",
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
        plot_df["label"] = (
            plot_df["filename"].str.replace(".edf", "", case=False)
            + " | " + plot_df["channel"] + "_" + plot_df["subband"]
        )
        plot_df = plot_df.reindex(
            plot_df[col].abs().sort_values(ascending=False).index
        )
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
    #  ALS vs Normal (enhanced with SEM and t-test)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_als_vs_normal(stats_df, feature_name, active_task, baseline_task,
                           title=None, use_sem=True):
        """Grouped bar ALS vs Normal per channel/subband + p-value annotations.

        Enhanced: Mendukung SEM error bars (baru dari pipeline).

        Parameters
        ----------
        use_sem : bool
            Jika True, gunakan SEM error bars. Jika False, tanpa error bars.
        """
        col_als = f"mean_als_{feature_name}"
        col_norm = f"mean_normal_{feature_name}"
        col_p = f"p_{feature_name}"
        col_sem_als = f"sem_als_{feature_name}"
        col_sem_norm = f"sem_normal_{feature_name}"

        if stats_df.empty or col_als not in stats_df.columns:
            return None

        if title is None:
            title = (
                f"Δ {feature_name.capitalize()}: ALS vs Normal "
                f"({active_task} − {baseline_task})"
            )

        channels = sorted(stats_df["channel"].unique())

        fig = make_subplots(
            rows=1, cols=len(channels),
            subplot_titles=[f"Channel={ch}" for ch in channels],
            shared_yaxes=True,
        )

        for col_idx, ch in enumerate(channels, 1):
            ch_data = stats_df[stats_df["channel"] == ch].sort_values("subband")

            # ALS bars with SEM
            als_error = None
            if use_sem and col_sem_als in ch_data.columns:
                als_error = dict(
                    type="data",
                    array=ch_data[col_sem_als].abs().tolist(),
                    visible=True,
                )

            fig.add_trace(go.Bar(
                x=ch_data["subband"], y=ch_data[col_als],
                name="ALS", marker_color="#EF4444",
                error_y=als_error,
                legendgroup="ALS", showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

            # Normal bars with SEM
            norm_error = None
            if use_sem and col_sem_norm in ch_data.columns:
                norm_error = dict(
                    type="data",
                    array=ch_data[col_sem_norm].abs().tolist(),
                    visible=True,
                )

            fig.add_trace(go.Bar(
                x=ch_data["subband"], y=ch_data[col_norm],
                name="Normal", marker_color="#60A5FA",
                error_y=norm_error,
                legendgroup="Normal", showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

            # P-value annotations
            if col_p in ch_data.columns:
                for _, row in ch_data.iterrows():
                    p = row[col_p]
                    if pd.notna(p):
                        max_y = max(
                            abs(row[col_als] or 0), abs(row[col_norm] or 0)
                        )
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
            title=title, height=480, barmode="group",
        ))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        if channels:
            fig.update_yaxes(
                title_text=f"Mean Δ {feature_name}", row=1, col=1,
            )
        return fig

    @staticmethod
    def plot_transition_deltas(transition_df, feature_name, title=None):
        """Bar chart transition deltas per group (ALS vs Normal) with SEM.

        Parameters
        ----------
        transition_df : pd.DataFrame
            Output dari DeltaCalculator.compute_transition_table().
        """
        if transition_df.empty:
            return None
        if title is None:
            title = f"Transition Delta — {feature_name.capitalize()}"

        fig = go.Figure()

        # ALS bars
        fig.add_trace(go.Bar(
            x=transition_df["subband"],
            y=transition_df["als_mean"],
            name="ALS",
            marker_color="#EF4444",
            error_y=dict(
                type="data",
                array=transition_df["als_sem"].tolist(),
                visible=True,
            ) if "als_sem" in transition_df.columns else None,
        ))

        # Normal bars
        fig.add_trace(go.Bar(
            x=transition_df["subband"],
            y=transition_df["normal_mean"],
            name="Normal",
            marker_color="#60A5FA",
            error_y=dict(
                type="data",
                array=transition_df["normal_sem"].tolist(),
                visible=True,
            ) if "normal_sem" in transition_df.columns else None,
        ))

        fig.update_layout(**_base_layout(
            title=title, height=440, barmode="group",
            xaxis_title="Subband",
            yaxis_title=f"Mean Δ {feature_name}",
        ))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig
