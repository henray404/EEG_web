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

    # ------------------------------------------------------------------ #
    #  PSD Plots                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_psd(raw, method="welch", fmin=0, fmax=49, n_fft=None,
                 title=None):
        """PSD plot dari raw MNE object dengan pilihan metode.

        Parameters
        ----------
        raw : mne.io.Raw
        method : str  'welch' atau 'multitaper'
        fmin, fmax : float
        n_fft : int | None
        title : str | None
        """
        from processing.psd import PSDAnalyzer

        psds, freqs, ch_names = PSDAnalyzer.compute_psd_raw(
            raw, method=method, fmin=fmin, fmax=fmax, n_fft=n_fft,
        )
        psds_db = 10 * np.log10(psds + 1e-20)

        if title is None:
            title = f"Power Spectral Density ({method.capitalize()})"

        fig = go.Figure()
        for i, ch in enumerate(ch_names):
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
    def plot_psd_per_task(psd_per_task, ch_names, title=None):
        """Overlay PSD dari beberapa task dalam satu plot.

        Setiap task diberi warna berbeda.  Jika banyak channel,
        tampilkan subplot per channel agar tidak terlalu ramai.

        Parameters
        ----------
        psd_per_task : dict  task_name -> (psds, freqs)
            psds shape: (n_channels, n_freqs)
        ch_names : list[str]
        title : str | None
        """
        if not psd_per_task:
            return None

        if title is None:
            title = "PSD per Task (Overlay)"

        n_channels = len(ch_names)

        # Palette untuk task
        task_names = list(psd_per_task.keys())
        palette = [
            "#60A5FA", "#A78BFA", "#F59E0B", "#10B981",
            "#F472B6", "#34D399", "#FB923C", "#818CF8",
        ]

        fig = make_subplots(
            rows=n_channels, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=[f"Channel: {ch}" for ch in ch_names],
        )

        for task_idx, task_name in enumerate(task_names):
            psds, freqs = psd_per_task[task_name]
            psds_db = 10 * np.log10(psds + 1e-20)
            color = TASK_COLORS.get(
                task_name, palette[task_idx % len(palette)]
            )

            for ch_idx, ch in enumerate(ch_names):
                if ch_idx >= psds_db.shape[0]:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=freqs, y=psds_db[ch_idx],
                        name=task_name,
                        legendgroup=task_name,
                        showlegend=(ch_idx == 0),
                        line=dict(width=1.5, color=color),
                    ),
                    row=ch_idx + 1, col=1,
                )

        for ch_idx in range(n_channels):
            fig.update_yaxes(
                title_text="Power (dB)", row=ch_idx + 1, col=1,
                gridcolor="#1E293B",
            )

        fig.update_xaxes(
            title_text="Frekuensi (Hz)", row=n_channels, col=1,
            gridcolor="#1E293B",
        )
        fig.update_layout(**_base_layout(
            title=title,
            height=max(300, 200 * n_channels),
            showlegend=True,
        ))
        return fig

    @staticmethod
    def plot_psd_band_power_bars(band_power_df, title=None):
        """Bar chart band power per subband per channel dari hasil PSD.

        Parameters
        ----------
        band_power_df : pd.DataFrame
            Output dari PSDAnalyzer.compute_band_power_from_psd().
            Kolom: channel, subband, band_power, relative_power, peak_frequency
        """
        if band_power_df.empty:
            return None

        if title is None:
            title = "Band Power per Subband (dari PSD)"

        fig = px.bar(
            band_power_df, x="subband", y="band_power",
            color="channel", barmode="group",
            color_discrete_sequence=CHANNEL_COLORS,
            labels={
                "subband": "Subband",
                "band_power": "Band Power",
                "channel": "Channel",
            },
        )
        fig.update_layout(**_base_layout(title=title, height=400))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    # ------------------------------------------------------------------ #
    #  Distribution & Correlation                                         #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Epoching & Sliding Windows Plots                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_spectrogram(windowed_features_df, channel, feature_name="band_power", 
                         title=None):
        """Time-frequency heatmap dari sliding window features.
        
        X-axis: start_time
        Y-axis: subband
        Color: nilai fitur (misal band_power)
        
        Parameters
        ----------
        windowed_features_df : pd.DataFrame
            Harus memuat kolom 'start_time', 'subband', 'channel', dan feature_name.
        channel : str
        feature_name : str
        
        Returns
        -------
        go.Figure | None
        """
        ch_df = windowed_features_df[windowed_features_df["channel"] == channel].copy()
        if ch_df.empty or feature_name not in ch_df.columns:
            return None

        # Pivot data: index=subband, columns=start_time, values=feature_name
        pivot = ch_df.pivot_table(
            index="subband", columns="start_time", 
            values=feature_name, aggfunc="mean"
        )
        
        # Urutkan subband secara logis dari frekuensi rendah ke tinggi
        subband_order = ["Delta", "Theta", "Mu", "Alpha", "Low_Beta", "High_Beta", "Beta", "Gamma"]
        ordered_index = [sb for sb in subband_order if sb in pivot.index]
        # Tambahkan sisa jika ada yang tidak terdaftar
        for sb in pivot.index:
            if sb not in ordered_index:
                ordered_index.append(sb)
                
        pivot = pivot.reindex(ordered_index)

        if title is None:
            title = f"Spectrogram ({feature_name.replace('_', ' ').capitalize()}) - Channel {channel}"

        # Perhatikan: pivot terbalik secara vertikal untuk imshow (y-axis terendah di bawah)
        pivot = pivot.iloc[::-1]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Viridis",
            hoverongaps=False,
            colorbar=dict(title=feature_name)
        ))
        
        fig.update_layout(**_base_layout(
            title=title,
            xaxis_title="Waktu (detik)",
            yaxis_title="Subband",
            height=400,
        ))
        
        return fig

    @staticmethod
    def plot_epoch_summary(epoch_stats_df, feature_name="band_power", title=None):
        """Bar chart fitur per task dari epoched analysis (dengan error bar).
        
        Parameters
        ----------
        epoch_stats_df : pd.DataFrame
            Harus memuat kolom 'task', 'channel', 'subband', feature_name, 
            dan feature_name_std.
        feature_name : str
        
        Returns
        -------
        go.Figure | None
        """
        if epoch_stats_df.empty or feature_name not in epoch_stats_df.columns:
            return None
            
        std_col = f"{feature_name}_std"
        has_std = std_col in epoch_stats_df.columns

        if title is None:
            title = f"Rata-rata {feature_name.replace('_', ' ').capitalize()} per Epoch"

        # Karena bisa jadi ada banyak channel, kita ambil rata-rata antar epoch per task+subband
        # (Idealnya df sudah difilter untuk 1 channel sebelum dipanggil ke plot ini)
        
        fig = px.bar(
            epoch_stats_df, x="subband", y=feature_name,
            color="task", barmode="group",
            error_y=std_col if has_std else None,
            color_discrete_map=TASK_COLORS,
            labels={
                "subband": "Subband",
                feature_name: feature_name.capitalize(),
                "task": "Task",
            },
        )
        fig.update_layout(**_base_layout(title=title, height=400))
        fig.update_xaxes(gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")
        return fig

    # ------------------------------------------------------------------ #
    #  Connectivity Plots (PLI / wPLI)                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_connectivity_matrix(conn_matrix, channels, method="wpli",
                                 subband="Alpha", task=None, title=None):
        """Heatmap matrix konektivitas antar channel.

        Parameters
        ----------
        conn_matrix : np.ndarray
            Shape (n_channels, n_channels).
        channels : list[str]
        method : str
        subband : str
        task : str | None

        Returns
        -------
        go.Figure
        """
        if title is None:
            parts = [f"Connectivity ({method.upper()}) — {subband}"]
            if task:
                parts.append(f"Task: {task}")
            title = " | ".join(parts)

        fig = go.Figure(data=go.Heatmap(
            z=conn_matrix,
            x=channels,
            y=channels,
            colorscale="Viridis",
            zmin=0, zmax=1,
            text=np.round(conn_matrix, 3),
            texttemplate="%{text:.3f}",
            textfont=dict(size=10),
            hoverongaps=False,
            colorbar=dict(title=method.upper()),
        ))
        fig.update_layout(**_base_layout(
            title=title, height=420,
        ))
        return fig

    @staticmethod
    def plot_connectivity_comparison(task_conn_dict, channels, subband,
                                     method="wpli"):
        """Subplot heatmap per task untuk membandingkan konektivitas.

        Parameters
        ----------
        task_conn_dict : dict
            task_name -> dict[subband_name -> np.ndarray]
        channels : list[str]
        subband : str
        method : str

        Returns
        -------
        go.Figure | None
        """
        # Filter task yang memiliki subband ini
        valid_tasks = [
            t for t, conn in task_conn_dict.items()
            if subband in conn
        ]
        if not valid_tasks:
            return None

        n_tasks = len(valid_tasks)
        fig = make_subplots(
            rows=1, cols=n_tasks,
            subplot_titles=[f"{t}" for t in valid_tasks],
            horizontal_spacing=0.05,
        )

        for i, task_name in enumerate(valid_tasks, 1):
            matrix = task_conn_dict[task_name][subband]
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=channels,
                    y=channels,
                    colorscale="Viridis",
                    zmin=0, zmax=1,
                    text=np.round(matrix, 2),
                    texttemplate="%{text:.2f}",
                    textfont=dict(size=9),
                    showscale=(i == n_tasks),  # colorbar hanya di kanan
                    colorbar=dict(title=method.upper()) if i == n_tasks else None,
                ),
                row=1, col=i,
            )

        fig.update_layout(**_base_layout(
            title=f"Perbandingan {method.upper()} — Subband {subband}",
            height=420,
        ))
        return fig

