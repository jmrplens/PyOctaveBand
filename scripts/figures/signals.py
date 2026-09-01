#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the sound level meter chain: filters, weightings and levels.

The instrument chain as the site teaches it -- octave and fractional-octave
filter banks, the frequency and time weightings, the level metrics they feed,
and the class limit masks each of those stages is verified against.
Everything here is embedded by a page under ``signals/filters/`` or
``signals/levels/``; the analyses applied to the recorded signal afterwards
have modules of their own (:mod:`figures.spectral_estimation`,
:mod:`figures.correlation_analysis`, :mod:`figures.system_measurement`), and
the checks made around the measurement are in :mod:`figures.metrology`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import signal as scipy_signal

from phonometry import filters
from phonometry._plot.common import format_frequency_axis, theme_fill

from .i18n import _fmt_minus
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_QUATERNARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    LABEL_LEVEL_DB,
    apply_axis_styling,
    measure_weighting_response,
    save_figure,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def generate_filter_type_comparison(output_dir: str) -> None:
    """Compare different filter architectures with a zoom inset."""
    print("Generating filter_type_comparison.png...")
    fs = 48000
    fraction = 1
    order = 6

    # We want exactly the 1000Hz band
    limits = [800.0, 1200.0]

    designs = [
        ("butter", "Butterworth", COLOR_PRIMARY, "-"),
        ("cheby1", "Chebyshev I", COLOR_SECONDARY, "--"),
        ("cheby2", "Chebyshev II", COLOR_TERTIARY, ":"),
        ("ellip", "Elliptic", "#9467bd", "-."),
        ("bessel", "Bessel", "#8c564b", "-"),
    ]

    _, ax = plt.subplots(figsize=(10, 7))

    # Create inset axis for zoom (increased height to 45%)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(ax, width="35%", height="45%", loc="upper left", borderpad=3)
    axins.set_xscale("log")  # Explicitly set log scale

    for f_type, label, color, style in designs:
        bank = filters.OctaveFilterBank(
            fs,
            fraction=fraction,
            order=order,
            limits=limits,
            design=filters.FilterDesign(filter_type=f_type),
        )

        # Find index of 1000Hz band
        idx = np.argmin(np.abs(np.array(bank.freq) - 1000))

        fsd = fs / bank.factor[idx]
        w, h = scipy_signal.sosfreqz(bank.sos[idx], worN=16384, fs=fsd)
        mag_db = 20 * np.log10(np.abs(h) + 1e-9)

        ax.semilogx(w, mag_db, label=label, color=color, linestyle=style)
        axins.plot(w, mag_db, color=color, linestyle=style)

    ax.axhline(-3, color=COLOR_FG, linestyle=":", alpha=0.3, label="−3 dB")
    axins.axhline(-3, color=COLOR_FG, linestyle=":", alpha=0.3)

    apply_axis_styling(
        ax,
        "Filter Architecture Comparison (Order 6, 1kHz Band)",
        xlim=(100, 8000),
        ylim=(-80, 5),
    )

    # Sub-plot styling (Zoom around 1kHz and -3dB)
    axins.set_xlim(650, 1500)
    axins.set_ylim(-4, 0.5)  # Adjusted: from -4 to 0.5
    axins.grid(True, which="both", alpha=0.3)
    axins.set_title("Zoom at −3 dB (Log Scale)", fontsize=9)

    # Fix x-ticks for log scale zoom to look right
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    axins.xaxis.set_major_formatter(ScalarFormatter())
    axins.xaxis.set_minor_formatter(NullFormatter())  # Hide minor tick labels
    axins.xaxis.get_major_formatter().set_scientific(
        False
    )  # Disable scientific notation
    axins.set_xticks([707, 1000, 1414])
    axins.set_xticklabels(["707", "1000", "1414"], fontsize=8)

    ax.legend(loc="lower right")
    save_figure(output_dir, "filter_type_comparison.png")
    plt.close()


def generate_filter_responses(output_dir: str) -> None:
    """Generate plots for the filter bank responses for different filter types."""
    fs = 48000

    # Filter types to generate
    filter_types = [
        ("butter", "butter"),
        ("cheby1", "cheby1"),
        ("cheby2", "cheby2"),
        ("ellip", "ellip"),
        ("bessel", "bessel"),
    ]

    configs = [
        (1, 6),
        (3, 6),
    ]

    for f_type_name, f_type in filter_types:
        for fraction, order in configs:
            filename = f"filter_{f_type_name}_fraction_{fraction}_order_{order}.png"
            print(f"Generating {filename}...")
            bank = filters.OctaveFilterBank(
                fs=fs,
                fraction=fraction,
                order=order,
                limits=[12.0, 20000.0],
                design=filters.FilterDesign(filter_type=f_type),
            )

            from phonometry.filters.design import _showfilter

            # Draw first, then save through save_figure so the Spanish
            # translation pass runs on the finished figure (it rewrites the
            # live figure's text artists right before the save).
            _showfilter(
                bank.sos,
                bank.freq,
                bank.freq_u,
                bank.freq_d,
                fs,
                bank.factor,
                show=False,
                plot_file=None,
                close=False,
            )
            save_figure(output_dir, filename, dpi=150, bbox_inches="tight")
            plt.close("all")


def generate_signal_responses(output_dir: str) -> None:
    """Generate spectral analysis plots for a complex signal."""
    fs = 48000
    duration = 5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    freqs = [20, 100, 500, 2000, 4000, 15000]
    y = 100 * np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

    for frac, filename, title in [
        (3, "signal_response_fraction_3.png", "1/3 Octave Band Analysis"),
    ]:
        print(f"Generating {filename}...")
        bank = filters.OctaveFilterBank(
            fs=fs, fraction=frac, order=6, limits=[12.0, 20000.0]
        )
        spl, freq = bank.filter(y)

        _, ax = plt.subplots()

        # Plot PSD of raw signal in background
        # We need to scale PSD to comparable levels.
        # A simple hack for visualization is to align the max of PSD to max of SPL
        f_psd, Pxx = scipy_signal.welch(y, fs, nperseg=8192)
        Pxx_db = 10 * np.log10(Pxx + 1e-12)
        # Shift PSD to match SPL peak roughly
        Pxx_db += (np.max(spl) - np.max(Pxx_db)) - 5  # Shift slightly below

        ax.semilogx(
            f_psd,
            Pxx_db,
            color="gray",
            alpha=0.6,
            linewidth=1.2,
            label="Raw Signal Spectrum (PSD)",
            zorder=0,
        )

        ax.semilogx(
            freq,
            spl,
            marker="o",
            markersize=5,
            linestyle="-",
            color=COLOR_PRIMARY,
            linewidth=1.5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=f"Measured 1/{frac} Octave Bands",
        )
        apply_axis_styling(ax, title, xlim=(11, 25000))
        ax.legend(loc="lower right")
        save_figure(output_dir, filename)
        plt.close()


def generate_multichannel_response(output_dir: str) -> None:
    """Generate analysis plot for a stereo signal with separate subplots."""
    print("Generating signal_response_multichannel.png...")
    fs = 48000
    duration = 5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    rng = np.random.default_rng(42)
    # Channel 1: Pink Noise (Voss-McCartney simplified)
    # Good enough for visualization
    white = rng.standard_normal(len(t))
    b, a = scipy_signal.butter(1, 0.04)  # -3dB/oct approx
    ch1 = scipy_signal.lfilter(b, a, white)
    ch1 = (ch1 - np.mean(ch1)) / np.max(np.abs(ch1))

    # Channel 2: Logarithmic Sine Sweep
    ch2 = scipy_signal.chirp(t, f0=50, t1=duration, f1=10000, method="logarithmic")

    x = np.vstack((ch1, ch2))
    bank = filters.OctaveFilterBank(fs=fs, fraction=3, order=6, limits=[20.0, 20000.0])
    spl, freq = bank.filter(x)

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Calculate PSDs for background
    f_psd1, Pxx1 = scipy_signal.welch(x[0], fs, nperseg=4096)
    Pxx_db1 = 10 * np.log10(Pxx1 + 1e-12)
    Pxx_db1 += np.max(spl[0]) - np.max(Pxx_db1)  # Align peaks

    f_psd2, Pxx2 = scipy_signal.welch(x[1], fs, nperseg=4096)
    Pxx_db2 = 10 * np.log10(Pxx2 + 1e-12)
    Pxx_db2 += np.max(spl[1]) - np.max(Pxx_db2)  # Align peaks

    # Plot Left Channel
    ax1.semilogx(
        f_psd1,
        Pxx_db1,
        color="gray",
        alpha=0.6,
        linewidth=1.2,
        label="Raw PSD",
        zorder=0,
    )
    ax1.semilogx(
        freq,
        spl[0],
        marker="o",
        markersize=5,
        label="Left Channel: Pink Noise",
        color=COLOR_PRIMARY,
        linestyle="-",
        linewidth=1.5,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    # Use standard styling but override title
    apply_axis_styling(ax1, "Multichannel Analysis (Stereo Input)", xlim=(16, 20000))
    ax1.legend(loc="lower right")
    # Let Y-axis autoscale

    # Plot Right Channel
    ax2.semilogx(
        f_psd2,
        Pxx_db2,
        color="gray",
        alpha=0.6,
        linewidth=1.2,
        label="Raw PSD",
        zorder=0,
    )
    ax2.semilogx(
        freq,
        spl[1],
        marker="s",
        markersize=5,
        label="Right Channel: Log Sine Sweep",
        color=COLOR_SECONDARY,
        linestyle="-",
        linewidth=1.5,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    apply_axis_styling(ax2, "", xlim=(16, 20000))
    ax2.set_title("")  # Remove title from bottom plot
    ax2.legend(loc="lower right")
    # Let Y-axis autoscale

    plt.tight_layout()
    save_figure(output_dir, "signal_response_multichannel.png")
    plt.close()


def generate_decomposition_plot(output_dir: str) -> None:
    """Generate time-domain decomposition plot comparing two filter types (Butterworth vs Chebyshev II)."""
    print(
        "Generating signal_decomposition.png with comparison (Butter vs Cheby2) @ 48kHz..."
    )
    fs = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Signal: sum of 250Hz and 1000Hz sines
    y = np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 1000 * t)

    # Filter into 1/1 octave bands with two different architectures
    # We use Chebyshev II (flat passband, no ripple)
    bank_butter = filters.OctaveFilterBank(
        fs=fs,
        fraction=1,
        order=6,
        limits=[100.0, 2000.0],
        design=filters.FilterDesign(filter_type="butter"),
    )
    bank_cheby2 = filters.OctaveFilterBank(
        fs=fs,
        fraction=1,
        order=6,
        limits=[100.0, 2000.0],
        design=filters.FilterDesign(filter_type="cheby2"),
    )

    # Cast to 3-tuple to satisfy mypy unpacking
    _, freq, xb_butter = bank_butter.filter(y, sigbands=True)

    _, _, xb_cheby2 = bank_cheby2.filter(y, sigbands=True)

    if xb_butter is None or xb_cheby2 is None:
        msg = "Signal bands should not be None"
        raise ValueError(msg)

    num_plots = len(xb_butter) + 2  # +1 for original, +1 for impulse response
    _fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.2 * num_plots), sharex=False)

    # Fixed Y limits for decomposition
    y_lim = (-2.8, 2.8)

    # 1. Original Signal
    axes[0].plot(t, y, color=COLOR_FG, linewidth=1.5)
    axes[0].set_title("Original Signal (250 Hz + 1000 Hz Sum) @ 48 kHz")
    axes[0].set_ylim(y_lim)
    axes[0].set_xlim(0, 0.04)

    # 2. Filtered Bands Comparison
    for i, (f_center) in enumerate(freq):
        axes[i + 1].plot(
            t,
            xb_butter[i],
            color=COLOR_PRIMARY,
            linewidth=1.5,
            label="Butterworth (Flat)",
        )
        axes[i + 1].plot(
            t,
            xb_cheby2[i],
            color=COLOR_SECONDARY,
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
            label="Chebyshev II",
        )
        axes[i + 1].set_title(f"Octave Band: {f_center:.0f} Hz", fontsize=11)
        axes[i + 1].set_ylim(y_lim)
        axes[i + 1].set_xlim(0, 0.04)
        if i == 0:
            axes[i + 1].legend(loc="upper right", fontsize=9, framealpha=0.8)

    # 3. Impulse Response (Stability/Transient Visualization)
    impulse = np.zeros(len(t))
    impulse[0] = 1.0
    _, _, ir_butter = bank_butter.filter(impulse, sigbands=True)
    _, _, ir_cheby2 = bank_cheby2.filter(impulse, sigbands=True)

    idx_1000 = np.argmin(np.abs(np.array(freq) - 1000))
    axes[-1].plot(
        t, ir_butter[idx_1000], color=COLOR_PRIMARY, linewidth=1.5, label="Butterworth"
    )
    axes[-1].plot(
        t,
        ir_cheby2[idx_1000],
        color=COLOR_SECONDARY,
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
        label="Chebyshev II",
    )
    axes[-1].set_title(
        f"Impulse Response ({freq[idx_1000]:.0f} Hz Band) - Transient/Stability Comparison"
    )
    axes[-1].set_xlim(0, 0.04)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.8)

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(True, which="both", alpha=0.4, linestyle=":")

    plt.tight_layout()
    save_figure(output_dir, "signal_decomposition.png")
    plt.close()


def _draw_weighting_curves(
    ax: Axes,
    fs: int,
    curves: tuple[tuple[str, str, str, str, float], ...],
    inset: Axes | None = None,
) -> None:
    """Draw ``(code, label, colour, linestyle, linewidth)`` curves measured at *fs*.

    Shared by the two weighting-family figures (the IEC 61672-1 A/C/Z chart and
    the special B/D/AU chart), which differ only in the curves they hold and in
    the axis they hold them on. Curves are drawn in the order given, so a wide
    reference line listed first sits behind the curves that follow it.
    """
    for code, label, color, style, width in curves:
        # measure_weighting_response is covered by tests/test_graph_measurements.py
        w, mag_db = measure_weighting_response(fs, code)
        ax.semilogx(
            w, mag_db, label=label, color=color, linestyle=style, linewidth=width
        )
        if inset is not None:
            inset.plot(w, mag_db, color=color, linestyle=style, linewidth=width)
    ax.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)


def generate_weighting_responses(output_dir: str) -> None:
    """Plot the IEC 61672-1 A/C/Z weighting frequency responses."""
    print("Generating weighting_responses.png...")
    fs = 48000

    _, ax = plt.subplots(figsize=(10, 7))

    # Zoom inset: the A curve is POSITIVE (+1.27 dB max at ~2.5 kHz per
    # IEC 61672-1 Table 2), invisible at the full -72..15 dB scale, so all
    # three curves of this figure are redrawn in it.
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(ax, width="42%", height="34%", loc="lower center", borderpad=2)
    axins.set_xscale("log")

    # The special B, D and AU curves have their own figure
    # (generate_special_weighting_responses); the infrasound G curve has
    # generate_g_weighting_response.
    curves = (
        ("A", "A-Weighting", COLOR_PRIMARY, "-", 1.8),
        ("C", "C-Weighting", COLOR_SECONDARY, "-", 1.8),
        ("Z", "Z-Weighting (Flat)", COLOR_FG, "--", 1.8),
    )
    _draw_weighting_curves(ax, fs, curves, inset=axins)

    # -72 dB, not the -50 the six-curve version used: with only A, C and Z left
    # nothing forces the floor up, and A reaches -70.4 dB at 10 Hz, so the whole
    # curve now fits on the axis instead of running off the bottom-left corner.
    apply_axis_styling(
        ax, "Frequency Weighting Curves (IEC 61672-1)", xlim=(10, 22000), ylim=(-72, 15)
    )

    axins.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.4, linewidth=1)
    axins.set_xlim(500, 8000)
    axins.set_ylim(-3, 2)
    axins.grid(True, which="both", alpha=0.3)
    axins.set_title(
        "Zoom: A-weighting is positive (max +1.27 dB @ 2.5 kHz)", fontsize=9
    )
    axins.annotate(
        "+1.27 dB",
        xy=(2500, 1.27),
        xytext=(4200, 1.55),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    axins.xaxis.set_major_formatter(ScalarFormatter())
    axins.xaxis.set_minor_formatter(NullFormatter())
    axins.set_xticks([500, 1000, 2500, 5000, 8000])
    axins.set_xticklabels(["500", "1k", "2.5k", "5k", "8k"], fontsize=8)

    ax.legend(loc="upper left", fontsize=9)
    save_figure(output_dir, "weighting_responses.png")
    plt.close()


def generate_special_weighting_responses(output_dir: str) -> None:
    """Plot the special B/D/AU weighting curves against the A reference."""
    print("Generating special_weighting_responses.png...")
    # 96 kHz, not the 48 kHz of the IEC 61672-1 figure: the point of AU is the
    # IEC 61012 U low-pass, specified up to 40 kHz, and a 48 kHz axis would cut
    # it off mid-slope (the guide says the same about measuring it).
    fs = 96000

    _, ax = plt.subplots(figsize=(10, 7))

    # A is drawn first, as a wide pale line, so it reads as the reference the
    # other three are described against: AU runs inside it up to 10 kHz and
    # then leaves, B stays above it, D crosses it at its 3.15 kHz hump.
    curves = (
        ("A", "A-Weighting (reference)", COLOR_MUTED, "-", 4.0),
        ("B", "B-Weighting (historical)", COLOR_TERTIARY, "--", 1.8),
        ("D", "D-Weighting (aircraft, withdrawn)", "#9467bd", "-.", 1.8),
        ("AU", "AU-Weighting (audible + ultrasound)", "#ff7f0e", "-", 1.8),
    )
    _draw_weighting_curves(ax, fs, curves)

    apply_axis_styling(
        ax, "Special Weighting Curves (B, D, AU)", xlim=(10, 40000), ylim=(-90, 18)
    )
    # apply_axis_styling stops labelling at 16 kHz; this axis runs into the
    # ultrasonic decade AU exists for, so it needs the 40 kHz end labelled.
    ax.set_xticks([16, 63, 250, 1000, 4000, 16000, 40000])
    ax.set_xticklabels(["16", "63", "250", "1k", "4k", "16k", "40k"])

    # The two numbers the guide quotes from the standards: the D hump of
    # IEC 537 (+11.5 dB at 3.15 kHz, NASA CR-3406 Table SLD-I) and the U
    # low-pass of IEC 61012 (-13 dB at 16 kHz, Table 1).
    ax.annotate(
        "+11.5 dB @ 3.15 kHz",
        xy=(3150, 11.6),
        xytext=(150, 13.5),
        fontsize=9,
        color="#9467bd",
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#9467bd"},
    )
    ax.annotate(
        "AU is 13 dB below A at 16 kHz",
        xy=(16000, -21.0),
        xytext=(700, -35.0),
        fontsize=9,
        color="#ff7f0e",
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#ff7f0e"},
    )

    ax.legend(loc="lower center", fontsize=9)
    save_figure(output_dir, "special_weighting_responses.png")
    plt.close()


def generate_g_weighting_response(output_dir: str) -> None:
    """Plot the ISO 7196 G-weighting curve against the Table 2 nominals."""
    print("Generating g_weighting_response.png...")
    from scipy import signal as sp_signal

    fs = 48000
    # ISO 7196:1995 Table 2 - nominal one-third-octave frequency, response dB
    table2 = [
        (0.25, -88.0),
        (0.5, -64.3),
        (1.0, -43.0),
        (2.0, -28.3),
        (4.0, -16.0),
        (8.0, -4.0),
        (10.0, 0.0),
        (16.0, 7.7),
        (20.0, 9.0),
        (31.5, -4.0),
        (63.0, -28.0),
        (125.0, -52.0),
        (250.0, -76.0),
    ]
    freqs = np.logspace(np.log10(0.1), np.log10(1000), 800)
    sos = filters.WeightingFilter(fs, "G").sos
    _, h = sp_signal.sosfreqz(sos, worN=freqs, fs=fs)
    mag_db = 20 * np.log10(np.abs(h))

    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs, mag_db, color=COLOR_PRIMARY, label="G-weighting (ISO 7196)")
    tf = [f for f, _ in table2]
    tv = [v for _, v in table2]
    ax.plot(
        tf,
        tv,
        "o",
        color=COLOR_SECONDARY,
        markersize=5,
        label="ISO 7196 Table 2 nominals",
        zorder=5,
    )
    ax.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)
    ax.axvline(10, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)
    ax.annotate(
        "0 dB @ 10 Hz",
        xy=(10, 0),
        xytext=(20, -18),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    apply_axis_styling(
        ax,
        "G Frequency Weighting for Infrasound (ISO 7196:1995)",
        xlim=(0.1, 1000),
        ylim=(-95, 15),
    )
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    ticks = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 125, 315, 1000]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(
        ["0.1", "0.25", "0.5", "1", "2", "5", "10", "20", "50", "125", "315", "1k"]
    )
    ax.legend(loc="upper right")
    save_figure(output_dir, "g_weighting_response.png")
    plt.close()


def generate_time_weighting_plot(output_dir: str) -> None:
    """Visualize Fast, Slow and Impulse time weighting response to a burst."""
    print("Generating time_weighting_analysis.png...")
    fs = 1000
    t = np.linspace(0, 4, fs * 4, endpoint=False)

    # 500ms burst of noise starting at 1.0s
    rng = np.random.default_rng(42)
    x = np.zeros_like(t)
    start_idx = int(fs * 1.0)
    end_idx = int(fs * 1.5)
    x[start_idx:end_idx] = rng.standard_normal(end_idx - start_idx)

    # Square for energy
    x_sq = x**2
    fast = filters.time_weighting(x, fs, mode="fast")
    slow = filters.time_weighting(x, fs, mode="slow")
    impulse = filters.time_weighting(x, fs, mode="impulse")

    _, ax = plt.subplots()
    # Normalize for better visualization
    # We normalized x_sq to peak at 1 for the plot
    peak = np.max(x_sq)
    x_sq /= peak
    fast /= peak
    slow /= peak
    impulse /= peak

    ax.plot(t, x_sq, color="#9e9e9e", alpha=0.6, label="Input Burst (Normalized)")
    ax.plot(t, fast, color=COLOR_PRIMARY, label="Fast (125ms)")
    ax.plot(t, slow, color=COLOR_SECONDARY, label="Slow (1000ms)")
    # The CSS "purple" is #800080, dark enough to be read against the white
    # page and not against the dark one; the palette's own purple is the same
    # hue with the luminance to carry a line on either.
    ax.plot(
        t,
        impulse,
        color=COLOR_QUATERNARY,
        linestyle="-.",
        linewidth=1.5,
        label="Impulse (35ms/1.5s)",
    )

    ax.set_title("Time Weighting Ballistics (IEC 61672-1)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Response")
    ax.legend(loc="upper right")
    ax.set_xlim(0.8, 3.5)
    save_figure(output_dir, "time_weighting_analysis.png")
    plt.close()


def generate_crossover_plot(output_dir: str) -> None:
    """Visualize Linkwitz-Riley 4th Order Crossover."""
    print("Generating crossover_lr4.png...")
    fs = 48000

    # Frequency analysis
    # Measure response using IR
    impulse = np.zeros(fs)
    impulse[0] = 1.0
    lp_ir, hp_ir = filters.linkwitz_riley(impulse, fs, freq=1000, order=4)

    w, h_lp = scipy_signal.freqz(lp_ir, worN=8192, fs=fs)
    _, h_hp = scipy_signal.freqz(hp_ir, worN=8192, fs=fs)

    _, ax = plt.subplots()
    ax.semilogx(
        w,
        20 * np.log10(np.abs(h_lp) + 1e-9),
        color=COLOR_PRIMARY,
        label="Low Pass (LR4)",
    )
    ax.semilogx(
        w,
        20 * np.log10(np.abs(h_hp) + 1e-9),
        color=COLOR_SECONDARY,
        label="High Pass (LR4)",
    )
    ax.semilogx(
        w,
        20 * np.log10(np.abs(h_lp + h_hp) + 1e-9),
        color=COLOR_FG,
        linestyle="--",
        label="Sum (Flat)",
    )

    apply_axis_styling(
        ax,
        "Linkwitz-Riley Crossover (4th Order @ 1kHz)",
        xlim=(20, 20000),
        ylim=(-60, 5),
    )
    ax.legend(loc="lower right")
    save_figure(output_dir, "crossover_lr4.png")
    plt.close()


def generate_parametric_eq_family(output_dir: str) -> None:
    """Magnitude responses of the RBJ Audio EQ Cookbook biquad family."""
    print("Generating parametric_eq_family.png...")
    fs = 48000

    family = [
        (
            filters.EQSection("peaking", 1000.0, gain_db=6.0, q=1.4),
            "Peaking +6 dB ($Q$ = 1.4)",
            COLOR_PRIMARY,
            "-",
        ),
        (
            filters.EQSection("lowshelf", 125.0, gain_db=6.0),
            "Low shelf +6 dB",
            COLOR_TERTIARY,
            "-",
        ),
        (
            filters.EQSection("highshelf", 4000.0, gain_db=-6.0),
            "High shelf −6 dB",
            "#9467bd",
            "-",
        ),
        (
            filters.EQSection("lowpass", 10000.0),
            "Low-pass ($Q$ = 0.707)",
            COLOR_SECONDARY,
            "--",
        ),
        (
            filters.EQSection("highpass", 50.0),
            "High-pass ($Q$ = 0.707)",
            "#8c564b",
            "--",
        ),
        (
            filters.EQSection("bandpass", 500.0, q=2.0),
            "Band-pass ($Q$ = 2)",
            "#ff7f0e",
            "-.",
        ),
        (filters.EQSection("notch", 2000.0, q=6.0), "Notch ($Q$ = 6)", "#17becf", "-."),
    ]

    _, ax = plt.subplots(figsize=(10, 6))
    for section, label, color, style in family:
        res = filters.ParametricEQ(fs, section).response(f_min=20.0, f_max=20000.0)
        ax.semilogx(
            res.frequencies, res.magnitude_db, label=label, color=color, linestyle=style
        )

    ax.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)
    apply_axis_styling(
        ax,
        "Parametric EQ Biquads (RBJ Audio EQ Cookbook)",
        xlim=(20, 20000),
        ylim=(-27, 9),
    )
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.set_ylabel("Magnitude [dB]")
    ax.legend(loc="lower center", fontsize=9, ncols=2)
    save_figure(output_dir, "parametric_eq_family.png")
    plt.close()


def generate_spectrogram_example(output_dir: str) -> None:
    """Visualize OctaveFilterBank.spectrogram on a time-varying signal."""
    print("Generating spectrogram_example.png...")
    fs = 48000
    duration = 4.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Log sweep + two tone bursts to show time-frequency localization
    rng = np.random.default_rng(42)
    x = 0.5 * scipy_signal.chirp(t, f0=80, t1=duration, f1=8000, method="logarithmic")
    x[int(1.0 * fs) : int(1.3 * fs)] += np.sin(
        2 * np.pi * 4000 * t[int(1.0 * fs) : int(1.3 * fs)]
    )
    x[int(2.5 * fs) : int(2.8 * fs)] += np.sin(
        2 * np.pi * 250 * t[int(2.5 * fs) : int(2.8 * fs)]
    )
    x += 0.01 * rng.standard_normal(len(t))

    # 1/12-octave bands stepped at 1/8 of the window: the Fast (125 ms)
    # integration still sets the time resolution, but the mesh is sampled four
    # times finer on each axis than the band spacing and hop it replaces, so
    # the sweep reads as a line instead of a staircase of cells.
    bank = filters.OctaveFilterBank(fs=fs, fraction=12, order=6, limits=[50.0, 12000.0])
    levels, freq, times = bank.spectrogram(x, window_time=0.125, overlap=0.875)

    _, ax = plt.subplots()
    mesh = ax.pcolormesh(times, freq, levels, shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_title("1/12 Octave Spectrogram (Fast windows, 87.5% overlap)", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(LABEL_FREQ_HZ)
    yticks = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["63", "125", "250", "500", "1k", "2k", "4k", "8k"])
    plt.colorbar(mesh, ax=ax, label=LABEL_LEVEL_DB)
    save_figure(output_dir, "spectrogram_example.png")
    plt.close()


def generate_ln_levels_example(output_dir: str) -> None:
    """Visualize statistical LN levels over the Fast envelope."""
    print("Generating ln_levels_example.png...")
    fs = 8000
    duration = 30.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    from phonometry import signals

    # Fluctuating "traffic-like" noise: background + random events
    rng = np.random.default_rng(42)
    x = 0.05 * rng.standard_normal(len(t))
    for _ in range(12):
        start = rng.uniform(1, duration - 3)
        length = rng.uniform(0.5, 2.0)
        idx = (t >= start) & (t < start + length)
        envelope = np.hanning(int(idx.sum()))
        x[idx] += envelope * rng.uniform(0.3, 1.0) * rng.standard_normal(int(idx.sum()))

    envelope_ms = filters.time_weighting(x, fs, mode="fast")
    level_t = 10 * np.log10(np.maximum(envelope_ms, 1e-12) / (2e-5) ** 2)
    stats = signals.ln_levels(x, fs, n=(10, 50, 90))

    _, ax = plt.subplots()
    ax.plot(t, level_t, color=COLOR_PRIMARY, linewidth=0.8, label="Fast level $L_p(t)$")
    for n_value, color, style in [
        (10, COLOR_SECONDARY, "--"),
        (50, COLOR_FG, "-"),
        (90, COLOR_TERTIARY, "-."),
    ]:
        ax.axhline(
            float(stats[n_value]),
            color=color,
            linestyle=style,
            linewidth=1.5,
            label=f"$L_{{{n_value}}}$ = {float(stats[n_value]):.1f} dB",
        )
    ax.set_title(
        "Statistical Levels $L_{10}$ / $L_{50}$ / $L_{90}$ (Fast envelope)", pad=12
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(LABEL_LEVEL_DB)
    ax.set_xlim(0, duration)
    ax.legend(loc="lower right")
    save_figure(output_dir, "ln_levels_example.png")
    plt.close()


def generate_zero_phase_comparison(output_dir: str) -> None:
    """Compare causal vs zero-phase band filtering of a tone burst."""
    print("Generating zero_phase_comparison.png...")
    fs = 48000
    duration = 0.15
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 250 Hz tone burst in the middle of the frame
    x = np.zeros_like(t)
    start, end = int(0.05 * fs), int(0.10 * fs)
    x[start:end] = np.sin(2 * np.pi * 250 * t[start:end]) * np.hanning(end - start)

    bank = filters.OctaveFilterBank(fs=fs, fraction=1, order=6, limits=[200.0, 300.0])
    _, _, bands_fwd = bank.filter(x, sigbands=True, calculate_level=False)
    _, _, bands_zp = bank.filter(
        x, sigbands=True, calculate_level=False, zero_phase=True
    )

    _, ax = plt.subplots()
    ax.plot(t, x, color="gray", alpha=0.5, linewidth=1.0, label="Input burst (250 Hz)")
    ax.plot(
        t,
        bands_fwd[0],
        color=COLOR_PRIMARY,
        linewidth=1.3,
        label="Causal filtering (group delay)",
    )
    ax.plot(
        t,
        bands_zp[0],
        color=COLOR_SECONDARY,
        linewidth=1.3,
        linestyle="--",
        label="zero_phase=True (aligned)",
    )
    ax.set_title("Zero-Phase Filtering: Group Delay Elimination (250 Hz Band)", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    save_figure(output_dir, "zero_phase_comparison.png")
    plt.close()


def generate_weighting_accuracy_hf(output_dir: str) -> None:
    """Compare A-weighting HF accuracy: analytic vs bilinear vs the fitted design."""
    print("Generating weighting_accuracy_hf.png...")
    fs = 48000

    freqs = np.logspace(np.log10(1000), np.log10(20000), 40)

    def analytic_a(f: np.ndarray) -> np.ndarray:
        ra = (12194**2 * f**4) / (
            (f**2 + 20.6**2)
            * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
            * (f**2 + 12194**2)
        )
        return np.asarray(20 * np.log10(ra) + 2.0)

    def measured_gains(wf: filters.WeightingFilter) -> np.ndarray:
        gains = []
        for f0 in freqs:
            tt = np.arange(int(fs * 0.2)) / fs
            x = np.sin(2 * np.pi * f0 * tt)
            y = wf.filter(x)
            n0 = int(0.05 * fs)  # skip filter transient
            gains.append(20 * np.log10(np.std(y[n0:]) / np.std(x[n0:])))
        return np.array(gains)

    legacy = measured_gains(filters.WeightingFilter(fs, "A", high_accuracy=False))
    accurate = measured_gains(filters.WeightingFilter(fs, "A"))
    reference = analytic_a(freqs)

    _, (ax, ax_err) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax.semilogx(
        freqs,
        reference,
        color=COLOR_FG,
        linewidth=2,
        label="IEC 61672-1 analytic curve",
    )
    ax.semilogx(
        freqs,
        legacy,
        color=COLOR_SECONDARY,
        linestyle="--",
        label="Plain bilinear (high_accuracy=False)",
    )
    ax.semilogx(
        freqs,
        accurate,
        color=COLOR_PRIMARY,
        linestyle="-.",
        label="Fitted at the sample rate (high_accuracy=True)",
    )
    ax.set_title(
        rf"A-Weighting High-Frequency Accuracy @ $f_{{\mathrm{{s}}}}$ = {fs // 1000} kHz",
        pad=12,
    )
    ax.set_ylabel(LABEL_LEVEL_DB)
    ax.legend(loc="lower left")

    ax_err.semilogx(
        freqs,
        legacy - reference,
        color=COLOR_SECONDARY,
        linestyle="--",
        label="Bilinear error",
    )
    ax_err.semilogx(
        freqs,
        accurate - reference,
        color=COLOR_PRIMARY,
        linestyle="-.",
        label="Fitted-design error",
    )
    ax_err.axhline(
        -2.5, color="gray", linestyle=":", label="Class 1 lower limit @ 12.5 kHz"
    )
    ax_err.set_ylabel("Error [dB]")
    ax_err.set_xlabel(LABEL_FREQ_HZ)
    ax_err.set_ylim(-8, 2)
    ax_err.legend(loc="lower left")

    for a in (ax, ax_err):
        xticks = [1000, 2000, 4000, 8000, 12500, 16000, 20000]
        a.set_xticks(xticks)
        a.set_xticklabels(["1k", "2k", "4k", "8k", "12.5k", "16k", "20k"])

    plt.tight_layout()
    save_figure(output_dir, "weighting_accuracy_hf.png")
    plt.close()


def generate_group_delay_comparison(output_dir: str) -> None:
    """Group delay of the 1 kHz band for every architecture (docs: filter-banks)."""
    print("Generating group_delay_comparison.png...")
    fs = 48000
    limits = [800.0, 1200.0]

    designs = [
        ("butter", "Butterworth", COLOR_PRIMARY, "-"),
        ("cheby1", "Chebyshev I", COLOR_SECONDARY, "--"),
        ("cheby2", "Chebyshev II", COLOR_TERTIARY, ":"),
        ("ellip", "Elliptic", "#9467bd", "-."),
        ("bessel", "Bessel", "#8c564b", "-"),
    ]

    _, ax = plt.subplots()
    for f_type, label, color, style in designs:
        bank = filters.OctaveFilterBank(
            fs,
            fraction=1,
            order=6,
            limits=limits,
            design=filters.FilterDesign(filter_type=f_type),
        )
        idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
        fsd = fs / bank.factor[idx]
        # Group delay of an SOS cascade = sum of the sections' group delays.
        w = np.logspace(np.log10(500), np.log10(2000), 1024)
        gd = np.zeros_like(w)
        for section in bank.sos[idx]:
            _w_s, gd_s = scipy_signal.group_delay(
                (section[:3], section[3:]), w=w, fs=fsd
            )
            gd += gd_s
        ax.semilogx(w, gd / fsd * 1000, label=label, color=color, linestyle=style)

    ax.set_title("Group Delay Comparison (1 kHz Octave Band, Order 6)", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Group delay [ms]")
    ax.set_xlim(500, 2000)
    from matplotlib.ticker import NullFormatter

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([500, 707, 1000, 1414, 2000])
    ax.set_xticklabels(["500", "707", "1k", "1.41k", "2k"])
    ax.legend(loc="upper right")
    save_figure(output_dir, "group_delay_comparison.png")
    plt.close()


def generate_tone_burst_iec(output_dir: str) -> None:
    """FAST envelope response to 4 kHz tonebursts vs IEC 61672-1 Table 4 targets."""
    print("Generating tone_burst_iec.png...")
    fs = 48000

    cases = [(0.2, -1.0), (0.05, -4.8), (0.01, -11.1)]  # Table 4, class 1 rows
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    t_all = np.arange(int(fs * 2.0)) / fs
    steady = np.sin(2 * np.pi * 4000 * t_all)
    ref = filters.time_weighting(steady, fs, mode="fast")[int(1.5 * fs) :].mean()

    for ax, (duration, target) in zip(axes, cases, strict=True):
        burst = np.zeros_like(t_all)
        start = int(0.5 * fs)
        burst[start : start + round(duration * fs)] = steady[
            start : start + round(duration * fs)
        ]
        env_db = 10 * np.log10(
            np.maximum(filters.time_weighting(burst, fs, mode="fast") / ref, 1e-6)
        )
        # The Error column of the guide's results table, reproducible from here.
        print(
            f"  {duration * 1000:g} ms: max {env_db.max():+.2f} dB, "
            f"error {env_db.max() - target:+.2f} dB vs target {target:+.1f}"
        )

        ax.plot(
            t_all, env_db, color=COLOR_PRIMARY, linewidth=1.3, label="FAST envelope"
        )
        ax.axhline(
            target,
            color=COLOR_SECONDARY,
            linestyle="--",
            linewidth=1.2,
            label=f"IEC target {_fmt_minus(target, '.1f')} dB",
        )
        ax.set_title(f"{duration * 1000:g} ms burst", fontsize=11)
        ax.set_xlim(0.4, 1.4)
        ax.set_ylim(-30, 3)
        ax.set_xlabel("Time [s]")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Level re steady state [dB]")

    fig.suptitle("4 kHz Toneburst Response vs IEC 61672-1 Table 4 (FAST)")
    plt.tight_layout()
    save_figure(output_dir, "tone_burst_iec.png")
    plt.close()


def generate_block_processing_continuity(output_dir: str) -> None:
    """Stateful vs stateless block processing (docs: block-processing)."""
    print("Generating block_processing_continuity.png...")
    fs = 8000
    n_blocks, block = 4, 1000
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n_blocks * block)
    t = np.arange(len(x)) / fs

    def band_output(stateful: bool) -> np.ndarray:
        bank = filters.OctaveFilterBank(
            fs,
            fraction=1,
            limits=[900, 1100],
            design=filters.FilterDesign(resample=False),
            block_processing=filters.BlockProcessing(stateful=stateful),
        )
        if stateful:
            parts = [
                bank.filter(
                    x[i * block : (i + 1) * block],
                    sigbands=True,
                    detrend=False,
                    calculate_level=False,
                )[2][0]
                for i in range(n_blocks)
            ]
        else:
            parts = []
            for i in range(n_blocks):
                b2 = filters.OctaveFilterBank(
                    fs,
                    fraction=1,
                    limits=[900, 1100],
                    design=filters.FilterDesign(resample=False),
                )
                parts.append(
                    b2.filter(
                        x[i * block : (i + 1) * block],
                        sigbands=True,
                        detrend=False,
                        calculate_level=False,
                    )[2][0]
                )
        return np.concatenate(parts)

    continuous = filters.OctaveFilterBank(
        fs,
        fraction=1,
        limits=[900, 1100],
        design=filters.FilterDesign(resample=False),
    ).filter(x, sigbands=True, detrend=False, calculate_level=False)[2][0]
    y_stateful = band_output(stateful=True)
    y_stateless = band_output(stateful=False)

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    zoom = slice(int(0.9 * block), int(1.4 * block))  # around the first boundary

    ax1.plot(
        t[zoom],
        continuous[zoom],
        color=COLOR_FG,
        linewidth=2.2,
        alpha=0.35,
        label="Continuous (whole signal)",
    )
    ax1.plot(
        t[zoom],
        y_stateful[zoom],
        color=COLOR_PRIMARY,
        linewidth=1.1,
        label="Stateful blocks (state carried)",
    )
    ax1.set_title(
        "stateful=True: block outputs equal the continuous result", fontsize=11
    )
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(
        t[zoom],
        continuous[zoom],
        color=COLOR_FG,
        linewidth=2.2,
        alpha=0.35,
        label="Continuous (whole signal)",
    )
    ax2.plot(
        t[zoom],
        y_stateless[zoom],
        color=COLOR_SECONDARY,
        linewidth=1.1,
        label="Independent blocks (state reset)",
    )
    ax2.axvline(block / fs, color=COLOR_FG, linestyle=":", alpha=0.6)
    # The callout lands on top of the dense trace, so it carries a solid
    # panel of its own instead of relying on the gaps between the waveforms.
    ax2.annotate(
        "block boundary:\nfilter transient restarts",
        xy=(block / fs, 0),
        xytext=(
            block / fs + 0.02,
            ax2.get_ylim()[0] * 0.55 if ax2.get_ylim()[0] < 0 else -1,
        ),
        fontsize=9.5,
        color=COLOR_FG,
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG},
    )
    ax2.set_title("No state: each block restarts the filter transient", fontsize=11)
    ax2.set_xlabel("Time [s]")
    ax2.legend(loc="upper right", fontsize=9)
    for a in (ax1, ax2):
        a.set_ylabel("Amplitude")

    plt.tight_layout()
    save_figure(output_dir, "block_processing_continuity.png")
    plt.close()


def generate_class_mask_overlay(output_dir: str) -> None:
    """Band response against the IEC 61260-1:2014 class limit mask."""
    print("Generating class_mask_overlay.png...")
    fs = 48000

    from phonometry.filters.compliance import class_limits

    bank = filters.OctaveFilterBank(
        fs,
        fraction=1,
        order=6,
        limits=[800, 1200],
        design=filters.FilterDesign(filter_type="butter"),
    )
    idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
    fm = bank.freq[idx]
    fsd = fs / bank.factor[idx]
    w, h = scipy_signal.sosfreqz(bank.sos[idx], worN=2**15, fs=fsd)
    attenuation = -20 * np.log10(np.abs(h) + 1e-12)
    a_ref = float(np.interp(fm, w, attenuation))
    omega = w / fm
    valid = (omega > 0.05) & (omega < 8)
    omega, delta_a = omega[valid], (attenuation - a_ref)[valid]

    grid = np.logspace(np.log10(0.05), np.log10(8), 2000)
    lo1, hi1 = class_limits(1.0, 1, grid)
    lo2, _ = class_limits(1.0, 2, grid)

    _, ax = plt.subplots(figsize=(10, 6.5))
    # Forbidden regions for class 1: below the minimum required attenuation
    # (stop band) and above the maximum allowed attenuation (pass band).
    ax.fill_between(
        grid,
        -10,
        lo1,
        color=theme_fill(COLOR_SECONDARY, ax),
        zorder=0,
        label="Forbidden for class 1 (too little attenuation)",
    )
    finite = np.isfinite(hi1)
    ax.fill_between(
        grid[finite],
        hi1[finite],
        90,
        color=theme_fill("#9467bd", ax),
        zorder=0,
        label="Forbidden for class 1 (too much attenuation)",
    )
    ax.plot(
        grid,
        lo2,
        color=COLOR_TERTIARY,
        linestyle=":",
        linewidth=1.2,
        label="Class 2 minimum attenuation",
    )

    ax.plot(
        omega,
        delta_a,
        color=COLOR_PRIMARY,
        linewidth=1.6,
        label="Butterworth order 6 (1 kHz octave band)",
    )

    ax.set_xscale("log")
    ax.set_xlim(0.08, 8)
    ax.set_ylim(-6, 90)
    ax.set_title("Relative Attenuation vs IEC 61260-1:2014 Class Limits", pad=12)
    ax.set_xlabel(r"Normalized frequency  $f\,/\,f_{\mathrm{m}}$")
    ax.set_ylabel(r"Relative attenuation $\Delta A$ [dB]")
    ax.set_xticks([0.125, 0.25, 0.5, 0.707, 1, 1.414, 2, 4, 8])
    ax.set_xticklabels(["0.125", "0.25", "0.5", "0.707", "1", "1.41", "2", "4", "8"])
    ax.legend(loc="upper left", fontsize=9)
    save_figure(output_dir, "class_mask_overlay.png")
    plt.close()


def generate_filter_class0_mask(output_dir: str) -> None:
    """Pass-band class 0/1/2 maximum corridors (IEC 61260:1995 / ANSI S1.11-2004)."""
    print("Generating filter_class0_mask...")
    fs = 48000
    from phonometry.filters.compliance import class_limits

    bank = filters.OctaveFilterBank(
        fs,
        fraction=1,
        order=6,
        limits=[800, 1200],
        design=filters.FilterDesign(filter_type="butter"),
    )
    idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
    fm = bank.freq[idx]
    fsd = fs / bank.factor[idx]
    w, h = scipy_signal.sosfreqz(bank.sos[idx], worN=2**15, fs=fsd)
    attenuation = -20 * np.log10(np.abs(h) + 1e-12)
    a_ref = float(np.interp(fm, w, attenuation))
    omega = w / fm

    # Restrict to the pass-band [G**-1/2, G**+1/2] where a finite max applies
    # (beyond the band edges the maximum limit is +inf, so plotting there would
    # misleadingly show the filter's natural roll-off "exceeding" a corridor).
    g_octave = 10 ** (3 / 10)  # octave ratio G (IEC 61260)
    edge_lo, edge_hi = g_octave**-0.5, g_octave**0.5
    pb = (omega >= edge_lo) & (omega <= edge_hi)
    omega, delta_a = omega[pb], (attenuation - a_ref)[pb]
    grid = np.linspace(edge_lo, edge_hi, 1500)

    _, ax = plt.subplots(figsize=(10, 6.5))
    # Nested min/max corridors: class 0 (+-0.15 dB reference) is the tightest.
    for cls, colour, name in (
        (2, COLOR_TERTIARY, "Class 2 corridor"),
        (1, COLOR_SECONDARY, "Class 1 corridor"),
        (0, COLOR_PRIMARY, "Class 0 corridor"),
    ):
        lo, hi = class_limits(1.0, cls, grid, edition="1995")
        ax.plot(grid, hi, color=colour, linewidth=1.4, label=name)
        ax.plot(grid, lo, color=colour, linewidth=1.4)
    ax.plot(
        omega,
        delta_a,
        color=COLOR_FG,
        linewidth=2.2,
        label="Butterworth order 6 (1 kHz octave band)",
    )

    ax.set_xscale("log")
    ax.set_xlim(edge_lo, edge_hi)
    ax.set_ylim(-0.7, 6)
    ax.set_title(
        "Pass-band Class 0/1/2 Limits (IEC 61260:1995 / ANSI S1.11-2004)", pad=12
    )
    ax.set_xlabel(r"Normalized frequency  $f\,/\,f_{\mathrm{m}}$")
    ax.set_ylabel(r"Relative attenuation $\Delta A$ [dB]")
    ax.set_xticks([0.707, 0.841, 1, 1.189, 1.414])
    ax.set_xticklabels(["0.707", "0.841", "1", "1.189", "1.414"])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())  # keep only explicit ticks
    ax.grid(which="major", color=COLOR_GRID, linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", fontsize=9)
    save_figure(output_dir, "filter_class0_mask.png")
    plt.close()


def generate_weighting_class_mask(output_dir: str) -> None:
    """A/C weighting deviation against the IEC 61672-1:2013 Table 3 mask."""
    print("Generating weighting_class_mask.png...")

    freqs, lower1, upper1 = filters.weighting_class_limits(1)
    _, lower2, upper2 = filters.weighting_class_limits(2)
    floor, ceil = -7.0, 7.0  # plotting bounds; -inf limits clip to the floor
    lo1 = np.clip(lower1, floor, ceil)
    lo2 = np.clip(lower2, floor, ceil)

    _, ax = plt.subplots(figsize=(10, 6.5))
    # Allowed corridor for class 1 (between lower and upper limit).
    ax.fill_between(
        freqs,
        lo1,
        upper1,
        color=theme_fill(COLOR_PRIMARY, ax),
        step="mid",
        label="Class 1 acceptance region",
    )
    ax.plot(
        freqs,
        upper1,
        color=COLOR_SECONDARY,
        linewidth=1.3,
        drawstyle="steps-mid",
        label="Class 1 upper/lower limit",
    )
    ax.plot(freqs, lo1, color=COLOR_SECONDARY, linewidth=1.3, drawstyle="steps-mid")
    ax.plot(
        freqs,
        upper2,
        color=COLOR_TERTIARY,
        linestyle=":",
        linewidth=1.1,
        drawstyle="steps-mid",
        label="Class 2 upper/lower limit",
    )
    ax.plot(
        freqs,
        lo2,
        color=COLOR_TERTIARY,
        linestyle=":",
        linewidth=1.1,
        drawstyle="steps-mid",
    )

    for curve, colour, marker in (("A", COLOR_PRIMARY, "o"), ("C", "#9467bd", "s")):
        result = filters.verify_weighting_class(filters.WeightingFilter(48000, curve))
        f = np.array([b["freq"] for b in result["bands"]])
        dev = np.array([b["deviation_db"] for b in result["bands"]])
        ax.plot(
            f,
            dev,
            color=colour,
            linewidth=1.6,
            marker=marker,
            markersize=4,
            label=f"{curve} weighting deviation (48 kHz)",
        )

    ax.set_xscale("log")
    ax.set_xlim(10, 20000)
    ax.set_ylim(floor, ceil)
    format_frequency_axis(ax, 10, 20000)
    ax.set_title("Weighting Deviation vs IEC 61672-1:2013 Table 3 Limits", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Deviation from design goal [dB]")
    ax.grid(which="both", color=COLOR_GRID, linestyle=":", alpha=0.4)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    save_figure(output_dir, "weighting_class_mask.png")
    plt.close()


def generate_sel_concept(output_dir: str) -> None:
    """SEL: the whole event compressed into one second of equal energy."""
    print("Generating sel_concept.png...")
    from phonometry import signals

    fs = 48000
    seconds = 8.0
    tt = np.arange(int(fs * seconds)) / fs
    rng = np.random.default_rng(11)
    # A vehicle pass-by: noise with a gaussian energy envelope
    envelope = np.exp(-0.5 * ((tt - 4.0) / 1.1) ** 2)
    x = envelope * rng.standard_normal(tt.size) * 0.3

    env = filters.time_weighting(x, fs, mode="fast")
    level = 10 * np.log10(np.maximum(env, 1e-12))
    l_sel = float(signals.sel(x, fs, dbfs=True))
    l_eq = float(signals.leq(x, fs, dbfs=True))

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        tt, level, color=COLOR_PRIMARY, linewidth=1.2, label="Fast level of the event"
    )
    ax.hlines(
        l_eq,
        0,
        seconds,
        color=COLOR_TERTIARY,
        linestyle="--",
        linewidth=1.6,
        label=r"$L_{\mathrm{eq}}$ over the whole event",
    )
    # SEL: same energy squeezed into 1 s (drawn as a 1 s block)
    ax.fill_between([3.5, 4.5], -55, l_sel, color=COLOR_SECONDARY, alpha=0.25)
    ax.hlines(
        l_sel,
        3.5,
        4.5,
        color=COLOR_SECONDARY,
        linewidth=2.2,
        label="SEL: same energy in 1 s",
    )
    ax.annotate(
        "equal energy",
        xy=(4.5, l_sel - 3),
        xytext=(5.6, l_sel - 1),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.set_title("Sound Exposure Level: the event normalized to 1 s", pad=12)
    ax.set_xlim(0, seconds)
    ax.set_ylim(-55, l_sel + 6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level [dBFS]")
    ax.legend(loc="lower left", fontsize=9)
    save_figure(output_dir, "sel_concept.png")
    plt.close()


def _slm_walkthrough_signals() -> tuple[int, np.ndarray, float]:
    """The two recordings of ``signals/sound-level-meter.mdx``, verbatim.

    The walkthrough page synthesizes a calibrator tone and a ten-second
    "street" recording so it runs anywhere; both figures below are built from
    exactly those arrays, so every number printed on them is a number the
    reader's own run produces.
    """
    from phonometry import metrology
    from phonometry import signals as level_signals

    fs = 48000
    calibrator = np.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(3 * fs) / fs)
    recording = level_signals.noise_signal(fs, 10.0, color="pink", rms=0.02, seed=7)
    recording[4 * fs : 5 * fs] += (
        0.2 * np.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)
    )
    cal = float(metrology.sensitivity(calibrator, target_spl=94.0, fs=fs))
    return fs, recording, cal


def generate_slm_level_track(output_dir: str) -> None:
    """The step-4 readouts drawn on the level history that produced them."""
    print("Generating slm_level_track.png...")
    from phonometry import signals as level_signals

    fs, recording, cal = _slm_walkthrough_signals()
    weighted = filters.weighting_filter(cal * recording, fs, curve="A")
    envelope = filters.time_weighting(weighted, fs, mode="fast")
    laf_t = 10 * np.log10(np.maximum(envelope, 1e-12) / (2e-5) ** 2)

    la_eq = float(level_signals.laeq(recording, fs, calibration_factor=cal))
    ln = level_signals.ln_levels(
        recording, fs, n=(10, 50, 90), weighting="A", calibration_factor=cal
    )
    lae = float(level_signals.sel(recording, fs, weighting="A", calibration_factor=cal))

    t = np.arange(recording.size) / fs
    _, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(4.0, 5.0, color=COLOR_SECONDARY, alpha=0.20)
    ax.annotate(
        "the 1 s event", xy=(4.5, 82.5), ha="center", color=COLOR_SECONDARY, fontsize=9
    )
    ax.plot(
        t,
        laf_t,
        color=COLOR_PRIMARY,
        linewidth=0.8,
        label=r"$L_{\mathrm{AF}}(t)$, Fast A-weighted level",
    )
    for value, name, color, style in [
        (la_eq, r"$L_{\mathrm{Aeq}}$", COLOR_FG, "--"),
        (float(ln[10]), "$L_{10}$", COLOR_SECONDARY, ":"),
        (float(ln[50]), "$L_{50}$", COLOR_TERTIARY, "-."),
        (float(ln[90]), "$L_{90}$", COLOR_QUATERNARY, (0, (6, 2))),
    ]:
        ax.axhline(
            value,
            color=color,
            linestyle=style,
            linewidth=1.4,
            label=f"{name} = {value:.1f} dB",
        )
    ax.annotate(
        rf"$L_{{\mathrm{{AE}}}}$ = {lae:.1f} dB: the whole event energy in 1 s",
        xy=(5.0, lae),
        xytext=(6.0, lae + 3.0),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.set_title("What step 4 reports, drawn on the recording it read", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level [dB re 20 µPa]")
    ax.set_xlim(0, 10)
    ax.set_ylim(40, 92)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-")
    ax.legend(loc="lower right", fontsize=9, ncols=2)
    save_figure(output_dir, "slm_level_track.png")
    plt.close()


def generate_slm_third_octave(output_dir: str) -> None:
    """The step-5 spectrum, unweighted and A-weighted, on one axis."""
    print("Generating slm_third_octave.png...")

    fs, recording, cal = _slm_walkthrough_signals()
    spl_z, centres = filters.octave_filter(
        recording,
        fs,
        fraction=3,
        calibration=filters.LevelCalibration(factor=cal),
    )
    weighted = filters.weighting_filter(cal * recording, fs, curve="A")
    spl_a, _ = filters.octave_filter(weighted, fs, fraction=3)

    total_z = 10 * np.log10(np.sum(10 ** (np.asarray(spl_z) / 10)))
    total_a = 10 * np.log10(np.sum(10 ** (np.asarray(spl_a) / 10)))

    _, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(891.3, 1122.5, color=COLOR_SECONDARY, alpha=0.20)
    ax.step(
        centres,
        spl_z,
        where="mid",
        color=COLOR_PRIMARY,
        linewidth=1.6,
        label=f"Z (unweighted): bands sum to {total_z:.1f} dB",
    )
    ax.step(
        centres,
        spl_a,
        where="mid",
        color=COLOR_TERTIARY,
        linewidth=1.6,
        linestyle="--",
        label=f"A-weighted: bands sum to {total_a:.1f} dB",
    )
    ax.annotate(
        "the 1 kHz band holds the event",
        xy=(1000, 70.0),
        xytext=(1900, 62.0),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.annotate(
        "pink background: equal energy per band",
        xy=(160, 43.2),
        xytext=(60, 33.0),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.set_title("One-third-octave spectrum of the same ten seconds", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Band level [dB re 20 µPa]")
    ax.set_ylim(-25, 78)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-")
    ax.grid(which="minor", color=COLOR_GRID, linestyle=":", alpha=0.4)
    format_frequency_axis(ax, 11.0, 23000.0)
    ax.legend(loc="lower left", fontsize=9)
    save_figure(output_dir, "slm_third_octave.png")
    plt.close()


def generate_energy_vs_arithmetic_mean(output_dir: str) -> None:
    """Why decibels are combined as energy, and what the shortcut costs."""
    print("Generating energy_vs_arithmetic_mean.png...")
    levels = np.array([60.0, 80.0])
    arithmetic = float(levels.mean())
    energetic = float(10 * np.log10(np.mean(10 ** (levels / 10))))

    sigma = np.linspace(0.0, 12.0, 121)
    gaussian = (np.log(10) / 20.0) * sigma**2
    two_level = 10 * np.log10(np.cosh(sigma * np.log(10) / 10))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_left.bar(
        ["first half", "second half"],
        levels,
        width=0.55,
        color=COLOR_PRIMARY,
        alpha=0.65,
    )
    ax_left.axhline(
        arithmetic,
        color=COLOR_MUTED,
        linestyle=":",
        linewidth=1.6,
        label=f"arithmetic mean of the dB values = {arithmetic:.0f} dB",
    )
    ax_left.axhline(
        energetic,
        color=COLOR_SECONDARY,
        linestyle="--",
        linewidth=1.8,
        label=rf"$L_{{\mathrm{{eq}}}}$ (energy mean) = {energetic:.0f} dB",
    )
    ax_left.annotate(
        "",
        xy=(1.45, energetic),
        xytext=(1.45, arithmetic),
        arrowprops={"arrowstyle": "<->", "lw": 1.2},
    )
    ax_left.text(
        1.5,
        0.5 * (arithmetic + energetic),
        f"{energetic - arithmetic:.0f} dB",
        va="center",
        fontsize=10,
    )
    ax_left.set_title("Two equal periods, 60 dB and 80 dB", pad=12)
    ax_left.set_ylabel(LABEL_LEVEL_DB)
    ax_left.set_ylim(50, 86)
    ax_left.set_xlim(-0.6, 2.1)
    ax_left.legend(loc="lower left", fontsize=9)
    ax_left.grid(axis="y", color=COLOR_GRID, linestyle="-")

    ax_right.plot(
        sigma,
        gaussian,
        color=COLOR_PRIMARY,
        linewidth=1.8,
        label=r"Gaussian spread: $0.115\,\sigma^2$",
    )
    ax_right.plot(
        sigma,
        two_level,
        color=COLOR_TERTIARY,
        linewidth=1.8,
        linestyle="--",
        label=r"two levels, one $\sigma$ either side",
    )
    ax_right.axvline(5.0, color=COLOR_MUTED, linewidth=1.0)
    ax_right.annotate(
        "levels spread over 10 dB",
        xy=(5.0, 2.5),
        xytext=(5.6, 1.0),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax_right.set_title(
        "The error of averaging decibels, and it never changes sign", pad=12
    )
    ax_right.set_xlabel("Standard deviation of the levels [dB]")
    ax_right.set_ylabel(r"$L_{\mathrm{eq}}$ minus the arithmetic dB mean [dB]")
    ax_right.set_xlim(0, 12)
    ax_right.set_ylim(0, 18)
    ax_right.grid(color=COLOR_GRID, linestyle="-")
    ax_right.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    save_figure(output_dir, "energy_vs_arithmetic_mean.png")
    plt.close()


def _distribution_pair() -> tuple[int, np.ndarray, np.ndarray]:
    """Two 60 s signals with the same LAeq and opposite level distributions."""
    fs = 8000
    duration = 60.0
    n = int(fs * duration)
    rng = np.random.default_rng(21)
    t = np.arange(n) / fs

    # (A) steady traffic-like noise: the level wanders with sigma = 3 dB.
    slow = np.zeros(n)
    for k in range(1, 12):
        slow += rng.normal() * np.sin(
            2 * np.pi * k * t / duration + rng.uniform(0.0, 2 * np.pi)
        )
    slow = 3.0 * slow / np.std(slow)
    steady = 0.02 * 10 ** (slow / 20) * rng.standard_normal(n)

    # (B) a quiet background with three short events 25 dB above it.
    peaky = 0.004 * rng.standard_normal(n)
    for centre in (12.0, 31.0, 47.0):
        i0, i1 = int((centre - 0.4) * fs), int((centre + 0.4) * fs)
        peaky[i0:i1] += (
            0.004 * 10 ** (25 / 20) * np.hanning(i1 - i0) * rng.standard_normal(i1 - i0)
        )
    return fs, steady, peaky


def generate_level_distribution(output_dir: str) -> None:
    """Same LAeq, opposite distributions: the percentiles read the shape."""
    print("Generating level_distribution.png...")
    from phonometry import signals

    fs, steady, peaky = _distribution_pair()
    peaky = peaky * 10 ** (
        (float(signals.laeq(steady, fs)) - float(signals.laeq(peaky, fs))) / 20
    )
    la_eq = float(signals.laeq(steady, fs))

    percentages = tuple(range(1, 100))
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.8))
    notes: list[tuple[str, str]] = []
    styles = [
        ("steady noise", steady, COLOR_PRIMARY, "-"),
        ("quiet background, three events", peaky, COLOR_SECONDARY, "-"),
    ]
    for name, x, color, style in styles:
        env = filters.time_weighting(
            filters.weighting_filter(x, fs, curve="A"), fs, mode="fast"
        )
        track = 10 * np.log10(np.maximum(env, 1e-12) / (2e-5) ** 2)
        t = np.arange(x.size) / fs
        # The Fast track is smooth on a 125 ms constant, so 80 samples per
        # second of it is a faithful trace at a fraction of the file size.
        ax_left.plot(
            t[::100],
            track[::100],
            color=color,
            linestyle=style,
            linewidth=0.7,
            label=name,
        )

        curve = signals.ln_levels(x, fs, n=percentages, weighting="A")
        values = np.array([float(curve[p]) for p in percentages])
        ax_right.plot(percentages, values, color=color, linewidth=1.8, label=name)
        for p in (10, 50, 90):
            ax_right.plot(p, float(curve[p]), "o", color=color, markersize=5)
            ax_right.annotate(
                f"$L_{{{p}}}$",
                xy=(p, float(curve[p])),
                fontsize=8,
                color=color,
                textcoords="offset points",
                xytext=(4, 5),
            )
        spread = float(curve[10]) - float(curve[90])
        gap = la_eq - float(curve[50])
        note = (
            f"{name}:  $L_{{10}} - L_{{90}}$ = {spread:.1f} dB"
            rf"   |   $L_{{\mathrm{{Aeq}}}} - L_{{50}}$ = {_fmt_minus(gap, '.1f')} dB"
        )
        notes.append((note, color))

    ax_left.axhline(
        la_eq,
        color=COLOR_FG,
        linestyle="--",
        linewidth=1.6,
        label=rf"$L_{{\mathrm{{Aeq}}}}$ = {la_eq:.1f} dB (both)",
    )
    ax_left.set_title(r"Two noises with the same $L_{\mathrm{Aeq}}$", pad=12)
    ax_left.set_xlabel("Time [s]")
    ax_left.set_ylabel(LABEL_LEVEL_DB)
    ax_left.set_xlim(0, 60)
    ax_left.set_ylim(48, 84)
    ax_left.grid(color=COLOR_GRID, linestyle="-")
    ax_left.legend(loc="upper left", fontsize=8, ncols=2)

    for row, (text, color) in enumerate(notes):
        ax_right.text(
            0.03,
            0.11 - 0.07 * row,
            text,
            transform=ax_right.transAxes,
            fontsize=9,
            color=color,
        )
    ax_right.set_title("Their exceedance curves", pad=12)
    ax_right.set_xlabel("Percentage of the time exceeded [%]")
    ax_right.set_ylabel(LABEL_LEVEL_DB)
    ax_right.set_xlim(0, 100)
    ax_right.set_ylim(44, 84)
    ax_right.grid(color=COLOR_GRID, linestyle="-")
    ax_right.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    save_figure(output_dir, "level_distribution.png")
    plt.close()


def generate_peak_oversampling(output_dir: str) -> None:
    """What the on-grid maximum misses, and what oversample=8 recovers."""
    print("Generating peak_oversampling.png...")
    from phonometry import signals

    fs = 48000
    f_tone = 8000.0
    t_fine = np.linspace(0.0, 2.5 / f_tone, 2000)
    n_grid = np.arange(0, int(2.5 * fs / f_tone) + 1)
    t_grid = n_grid / fs
    # Six samples per cycle: this phase puts the crest exactly midway between
    # two of them, which is the worst case the grid can produce.
    phase = np.pi / 3
    fine = np.sin(2 * np.pi * f_tone * t_fine + phase)
    grid = np.sin(2 * np.pi * f_tone * t_grid + phase)
    on_grid = float(np.max(grid))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_left.plot(
        t_fine * 1e3,
        fine,
        color=COLOR_PRIMARY,
        linewidth=1.6,
        label="the continuous 8 kHz tone",
    )
    ax_left.plot(
        t_grid * 1e3,
        grid,
        "o",
        color=COLOR_SECONDARY,
        markersize=6,
        label="its samples at 48 kHz (6 per cycle)",
    )
    ax_left.axhline(
        1.0, color=COLOR_FG, linestyle="--", linewidth=1.2, label="true peak"
    )
    ax_left.axhline(
        on_grid,
        color=COLOR_SECONDARY,
        linestyle=":",
        linewidth=1.4,
        label=f"largest sample: {_fmt_minus(20 * np.log10(on_grid), '.2f')} dB low",
    )
    ax_left.set_title("The crest falls between two samples", pad=12)
    ax_left.set_xlabel("Time [ms]")
    ax_left.set_ylabel("Amplitude")
    ax_left.set_ylim(-1.25, 1.35)
    ax_left.grid(color=COLOR_GRID, linestyle="-")
    ax_left.legend(loc="lower right", fontsize=8)

    t_tone = np.arange(int(0.05 * fs)) / fs
    per_cycle = np.array([3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0])
    phases = np.linspace(0.0, 2 * np.pi, 46)
    colors = [COLOR_SECONDARY, COLOR_QUATERNARY, COLOR_TERTIARY, COLOR_PRIMARY]
    for factor, color in zip((1, 2, 4, 8), colors, strict=True):
        worst = []
        for n_samples in per_cycle:
            tones = [np.sin(2 * np.pi * (fs / n_samples) * t_tone + p) for p in phases]
            worst.append(
                max(
                    float(signals.lc_peak(tone, fs, oversample=16))
                    - float(signals.lc_peak(tone, fs, oversample=factor))
                    for tone in tones
                )
            )
        ax_right.semilogx(
            per_cycle,
            worst,
            marker="o",
            color=color,
            linewidth=1.8,
            label=f"oversample={factor}",
        )
    ax_right.axhline(0.1, color=COLOR_MUTED, linestyle=":", linewidth=1.2)
    ax_right.annotate("0.1 dB", xy=(30, 0.16), fontsize=9, color=COLOR_MUTED)
    ax_right.set_title("Worst case over the phase of the tone", pad=12)
    ax_right.set_xlabel("Samples per cycle")
    ax_right.set_ylabel("Under-read of the peak [dB]")
    ax_right.xaxis.set_major_locator(mticker.FixedLocator(per_cycle.tolist()))
    ax_right.xaxis.set_minor_locator(mticker.NullLocator())
    ax_right.xaxis.set_major_formatter(
        mticker.FixedFormatter([f"{v:.0f}" for v in per_cycle])
    )
    ax_right.set_ylim(0, 3.2)
    ax_right.grid(color=COLOR_GRID, linestyle="-")
    ax_right.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    save_figure(output_dir, "peak_oversampling.png")
    plt.close()


def generate_dose_exchange(output_dir: str) -> None:
    """Iso-exposure contours: the equal-energy trade of level against time."""
    print("Generating dose_exchange.png...")
    from phonometry import signals

    fs = 48000
    hours = np.geomspace(1.0 / 60.0, 8.0, 60)
    sample = 0.4946 * np.random.default_rng(1).standard_normal(2 * fs)
    base = float(signals.laeq(sample, fs))
    # lex_8h(sample, duration_hours=h) - LAeq(sample) is the 10 log10(h/8)
    # normalisation, so the contour is the target minus that offset.
    offsets = np.array(
        [float(signals.lex_8h(sample, fs, duration_hours=h)) - base for h in hours]
    )

    _, ax = plt.subplots(figsize=(10, 5.4))
    styles = [
        (80.0, COLOR_TERTIARY, ":"),
        (85.0, COLOR_PRIMARY, "-"),
        (87.0, COLOR_SECONDARY, "--"),
    ]
    for target, color, style in styles:
        exposure = 8.0 * (20e-6 * 10 ** (target / 20)) ** 2
        ax.semilogx(
            hours,
            target - offsets,
            color=color,
            linestyle=style,
            linewidth=1.9,
            label=rf"$L_{{\mathrm{{EX,8h}}}}$ = {target:.0f} dB  ($E$ = {exposure:.2f} Pa²·h)",
        )
    ax.plot(8.0, 90.0, "o", color=COLOR_FG, markersize=7)
    ax.annotate(
        "90 dB(A) for 8 h = 3.20 Pa²·h",
        xy=(8.0, 90.0),
        xytext=(1.1, 96.0),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.annotate(
        "+3 dB for every halving of the duration",
        xy=(2.0, 88.0),
        xytext=(1.15, 79.0),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.set_title(
        "Equal-energy exchange: every point on a line is the same daily exposure",
        pad=12,
    )
    ax.set_xlabel("Exposure duration [h]")
    ax.set_ylabel("A-weighted level [dB]")
    ax.set_xlim(1.0 / 60.0, 8.0)
    ax.set_ylim(77, 113)
    ax.set_xticks([1 / 60, 5 / 60, 15 / 60, 0.5, 1.0, 2.0, 4.0, 8.0])
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter(
            ["1 min", "5 min", "15 min", "30 min", "1 h", "2 h", "4 h", "8 h"]
        )
    )
    ax.grid(color=COLOR_GRID, linestyle="-")
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "dose_exchange.png")
    plt.close()


def generate_ballistics_vs_duration(output_dir: str) -> None:
    """How much of an event each detector keeps, against event duration."""
    print("Generating ballistics_vs_duration.png...")

    fs = 48000
    t = np.arange(int(3.0 * fs)) / fs
    tone = np.sin(2 * np.pi * 4000 * t)
    durations = np.geomspace(0.001, 2.0, 42)
    start = int(0.5 * fs)
    # The steady reference needs many time constants: Slow is still 8 % short
    # of it after 2.5 s, which would bias the whole S curve by 0.4 dB.
    long_t = np.arange(int(12.0 * fs)) / fs
    long_tone = np.sin(2 * np.pi * 4000 * long_t)
    steady = {
        m: float(np.mean(filters.time_weighting(long_tone, fs, mode=m)[-fs:]))
        for m in ("fast", "slow", "impulse")
    }

    _, ax = plt.subplots(figsize=(10, 5.4))
    modes = [
        ("fast", 0.125, COLOR_PRIMARY, "-"),
        ("slow", 1.0, COLOR_TERTIARY, "-"),
        ("impulse", None, COLOR_SECONDARY, "-"),
    ]
    measured: dict[str, np.ndarray] = {}
    for mode, tau, color, style in modes:
        reference = steady[mode]
        peaks = []
        for t_b in durations:
            burst = np.zeros_like(t)
            stop = start + int(t_b * fs)
            burst[start:stop] = tone[start:stop]
            peaks.append(
                10
                * np.log10(
                    float(np.max(filters.time_weighting(burst, fs, mode=mode)))
                    / reference
                )
            )
        measured[mode] = np.asarray(peaks)
        ax.semilogx(
            durations * 1e3,
            measured[mode],
            color=color,
            linestyle=style,
            linewidth=1.9,
            label=f"{mode} (measured)",
        )
        if tau is not None:
            table = np.array(
                [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            )
            ax.plot(
                table * 1e3,
                10 * np.log10(1 - np.exp(-table / tau)),
                "o",
                color=color,
                markersize=5,
                markerfacecolor="none",
            )

    # The F/S gap at exactly 100 ms, measured rather than interpolated.
    at_100ms = {}
    for mode in ("fast", "slow", "impulse"):
        reference = steady[mode]
        burst = np.zeros_like(t)
        burst[start : start + int(0.1 * fs)] = tone[start : start + int(0.1 * fs)]
        at_100ms[mode] = 10 * np.log10(
            float(np.max(filters.time_weighting(burst, fs, mode=mode))) / reference
        )
    gap = at_100ms["fast"] - at_100ms["slow"]
    ax.axvline(100.0, color=COLOR_MUTED, linewidth=1.0)
    ax.annotate(
        f"at 100 ms, F reads {gap:.1f} dB above S",
        xy=(100.0, -6.0),
        xytext=(150.0, -22.0),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.annotate(
        f"even Impulse, with its 35 ms attack,\n"
        f"loses {abs(measured['impulse'][0]):.0f} dB on a 1 ms burst",
        xy=(1.05, measured["impulse"][0]),
        xytext=(1.6, -10.0),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.set_title("Every detector under-reads every short event", pad=12)
    ax.set_xlabel("Toneburst duration [ms]")
    ax.set_ylabel("Peak level re the steady reading [dB]")
    ax.set_xlim(1.0, 2000.0)
    ax.set_ylim(-32, 2)
    ax.xaxis.set_major_locator(
        mticker.FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter(
            ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1k", "2k"]
        )
    )
    ax.grid(color=COLOR_GRID, linestyle="-")
    ax.plot(
        [],
        [],
        "o",
        color=COLOR_FG,
        markersize=5,
        markerfacecolor="none",
        label="IEC 61672-1 Table 4, Equation (7)",
    )
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "ballistics_vs_duration.png")
    plt.close()


def generate_c_minus_a_spectrum(output_dir: str) -> None:
    """What a C - A difference of 23 dB looks like as a band spectrum."""
    print("Generating c_minus_a_spectrum.png...")
    from phonometry import signals

    fs = 48000
    t = np.arange(10 * fs) / fs
    rng = np.random.default_rng(1)
    rumble = 0.2 * np.sin(2 * np.pi * 50 * t) + 0.01 * rng.standard_normal(t.size)
    pink = signals.noise_signal(fs, 10.0, color="pink", rms=0.02, seed=5)

    scenes = [
        ("A 50 Hz rumble under a light hiss", rumble),
        ("Broadband pink noise", pink),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    curves = [
        ("Z", COLOR_MUTED, "-"),
        ("C", COLOR_PRIMARY, "--"),
        ("A", COLOR_SECONDARY, ":"),
    ]
    for ax, (title, x) in zip(axes, scenes, strict=True):
        for curve, color, style in curves:
            weighted = filters.weighting_filter(x, fs, curve=curve)
            band_levels, centres = filters.octave_filter(weighted, fs, fraction=3)
            ax.semilogx(
                centres,
                band_levels,
                color=color,
                linestyle=style,
                linewidth=1.8,
                label=f"{curve}-weighted bands",
            )
        la = float(signals.leq(filters.weighting_filter(x, fs, curve="A")))
        lc = float(signals.leq(filters.weighting_filter(x, fs, curve="C")))
        ax.text(
            0.03,
            0.05,
            f"$L_{{\\mathrm{{Aeq}}}}$ {la:.1f} dB\n$L_{{\\mathrm{{Ceq}}}}$ {lc:.1f} dB\n"
            f"C − A = {_fmt_minus(lc - la, '.1f')} dB",
            transform=ax.transAxes,
            fontsize=10,
            bbox={
                "boxstyle": "round",
                "facecolor": COLOR_PANEL,
                "edgecolor": COLOR_MUTED,
            },
        )
        ax.set_title(title, pad=12)
        ax.set_xlabel(LABEL_FREQ_HZ)
        ax.set_ylim(-5, 82)
        ax.grid(which="major", color=COLOR_GRID, linestyle="-")
        ax.grid(which="minor", color=COLOR_GRID, linestyle=":", alpha=0.4)
        format_frequency_axis(ax, 11.0, 23000.0)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("Band level [dB]")
    fig.tight_layout()
    save_figure(output_dir, "c_minus_a_spectrum.png")
    plt.close()


def generate_pole_migration(output_dir: str) -> None:
    """Why the bank decimates: where the poles of a narrow band actually sit."""
    print("Generating pole_migration.png...")
    from scipy.signal import sos2zpk

    fs = 48000
    bank = filters.OctaveFilterBank(fs=fs, fraction=3)
    full = filters.OctaveFilterBank(
        fs=fs, fraction=3, design=filters.FilterDesign(resample=False)
    )
    idx = int(np.argmin(np.abs(np.asarray(bank.freq) - 25.0)))
    f_m = float(bank.freq[idx])
    factor = int(bank.factor[idx])

    _, p_full, _ = sos2zpk(np.asarray(full.sos[idx]))
    _, p_dec, _ = sos2zpk(np.asarray(bank.sos[idx]))

    fig = plt.figure(figsize=(12, 5.2))
    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2)
    ax_c = fig.add_subplot(1, 3, 3)

    for ax, poles, _rate, title in (
        (ax_a, p_full, fs, f"Designed at {fs / 1000:.0f} kHz"),
        (ax_b, p_dec, fs / factor, f"As the bank realizes it, at {fs / factor:.0f} Hz"),
    ):
        theta = np.linspace(0, 2 * np.pi, 512)
        ax.plot(np.cos(theta), np.sin(theta), color=COLOR_MUTED, linewidth=1.2)
        ax.plot(
            np.real(poles),
            np.imag(poles),
            "x",
            color=COLOR_SECONDARY,
            markersize=9,
            markeredgewidth=2.0,
            label="poles",
        )
        ax.plot(
            [0.0] * 4,
            [0.0] * 4,
            "o",
            color=COLOR_PRIMARY,
            markerfacecolor="none",
            markersize=9,
            label="zeros (at $z = 0$)",
        )
        gap = 1.0 - float(np.max(np.abs(poles)))
        mantissa, exponent = f"{gap:.1e}".split("e")
        ax.set_title(
            f"{title}\n$1 - r = {mantissa}\\times10^{{{int(exponent)}}}$", pad=10
        )
        ax.set_xlabel(r"$\mathrm{Re}\,z$")
        ax.set_ylabel(r"$\mathrm{Im}\,z$")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.grid(color=COLOR_GRID, linestyle="-")
        # One row along the bottom: the corners of a unit-circle plot are the
        # only space free of the circle, and they are too small for a stacked
        # legend -- at "lower left" the box swallowed the outermost pole of
        # the lower conjugate group and a stretch of the circle itself.
        ax.legend(loc="lower center", ncol=2, fontsize=8)

    # The zoom that makes the collapsed pair visible.  Its top stays clear of
    # the parent's top spine so the inset title has a line of its own.
    inset = ax_a.inset_axes((0.52, 0.60, 0.44, 0.34))
    inset.plot(
        np.real(p_full),
        np.imag(p_full),
        "x",
        color=COLOR_SECONDARY,
        markersize=8,
        markeredgewidth=1.8,
    )
    theta = np.linspace(-0.01, 0.01, 200)
    inset.plot(np.cos(theta), np.sin(theta), color=COLOR_MUTED, linewidth=1.0)
    inset.set_xlim(0.9985, 1.0004)
    inset.set_ylim(-0.0045, 0.0045)
    inset.tick_params(labelsize=7)
    inset.set_title("zoom at $z = 1$", fontsize=8, pad=3)
    indicator = ax_a.indicate_inset_zoom(inset, edgecolor=COLOR_FG)
    # Of the two leaders matplotlib picks, the one from the lower-left corner
    # runs along the inset's own tick labels on its way to z = 1; the
    # lower-right one reaches the same point over empty ground.
    leaders = indicator.connectors
    if leaders is not None:
        leaders[0].set_visible(False)

    radius_dec = [
        1.0 - float(np.max(np.abs(sos2zpk(np.asarray(s))[1]))) for s in bank.sos
    ]
    radius_full = [
        1.0 - float(np.max(np.abs(sos2zpk(np.asarray(s))[1]))) for s in full.sos
    ]
    ax_c.loglog(
        bank.freq,
        radius_full,
        "o-",
        color=COLOR_SECONDARY,
        markersize=4,
        linewidth=1.6,
        label="resample=False (one rate)",
    )
    ax_c.loglog(
        bank.freq,
        radius_dec,
        "s-",
        color=COLOR_PRIMARY,
        markersize=4,
        linewidth=1.6,
        label="the default multirate bank",
    )
    ax_c.set_title("Every band of the 1/3-octave bank", pad=10)
    ax_c.set_xlabel(LABEL_FREQ_HZ)
    ax_c.set_ylabel("1 − (largest pole radius)")
    ax_c.grid(which="both", color=COLOR_GRID, linestyle="-")
    ax_c.legend(loc="lower right", fontsize=8)
    format_frequency_axis(ax_c, 11.0, 23000.0, minor=None)
    # Eleven octave labels over a third of this figure's width: at the body
    # size "125 250 500" closes up into a single number.
    ax_c.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        f"The {f_m:.0f} Hz one-third-octave band, before and after "
        f"decimation by {factor}"
    )
    fig.tight_layout()
    save_figure(output_dir, "pole_migration.png")
    plt.close()


def generate_parametric_eq_cascade(output_dir: str) -> None:
    """The three sections of the guide's snippet and the cascade they sum to."""
    print("Generating parametric_eq_cascade.png...")

    fs = 48000
    eq = filters.ParametricEQ(
        fs,
        [
            filters.EQSection("lowshelf", 100.0, gain_db=4.0),
            filters.EQSection("peaking", 1000.0, gain_db=-6.0, bw=1.0),
            filters.EQSection("highshelf", 8000.0, gain_db=3.0),
        ],
    )
    result = eq.response(f_min=20.0, f_max=20000.0)
    result.plot(show_sections=True)
    # The result object drew through pyplot, so the figure it made is the
    # current one; asking the axes for it hands back a union this cannot use.
    fig = plt.gcf()
    fig.set_size_inches(10, 6.4)
    fig.tight_layout()
    save_figure(output_dir, "parametric_eq_cascade.png")
    plt.close()


def generate_architecture_tradeoff(output_dir: str) -> None:
    """The architecture choice as two numbers: rejection and group delay."""
    print("Generating architecture_tradeoff.png...")
    from scipy.signal import sosfreqz

    fs = 48000
    names = ("butter", "cheby1", "cheby2", "ellip", "bessel")
    two_fm, four_fm, delays = [], [], []
    for ftype in names:
        bank = filters.OctaveFilterBank(
            fs,
            fraction=1,
            order=6,
            limits=[800, 1200],
            design=filters.FilterDesign(filter_type=ftype, resample=False),
        )
        idx = int(np.argmin(np.abs(np.asarray(bank.freq) - 1000.0)))
        f_m = float(bank.freq[idx])
        w, h = sosfreqz(bank.sos[idx], worN=1 << 17, fs=fs)
        mag = 20 * np.log10(np.abs(h) + 1e-15)
        mag -= mag.max()
        two_fm.append(float(mag[int(np.argmin(np.abs(w - 2 * f_m)))]))
        four_fm.append(float(mag[int(np.argmin(np.abs(w - 4 * f_m)))]))
        grid = np.linspace(f_m * 0.95, f_m * 1.05, 4001)
        _, h_c = sosfreqz(bank.sos[idx], worN=2 * np.pi * grid / fs)
        delay = -np.gradient(np.unwrap(np.angle(h_c)), 2 * np.pi * grid)
        delays.append(float(delay[len(delay) // 2]) * 1e3)

    x = np.arange(len(names))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.0))
    ax_a.bar(
        x - 0.19, two_fm, 0.36, color=COLOR_PRIMARY, label=r"at $2 f_{\mathrm{m}}$"
    )
    ax_a.bar(
        x + 0.19, four_fm, 0.36, color=COLOR_TERTIARY, label=r"at $4 f_{\mathrm{m}}$"
    )
    for xi, value in zip(x - 0.19, two_fm, strict=True):
        ax_a.annotate(
            _fmt_minus(value, ".0f"),
            (xi, value),
            ha="center",
            va="top",
            fontsize=9,
            xytext=(0, -4),
            textcoords="offset points",
        )
    for xi, value in zip(x + 0.19, four_fm, strict=True):
        ax_a.annotate(
            _fmt_minus(value, ".0f"),
            (xi, value),
            ha="center",
            va="top",
            fontsize=9,
            xytext=(0, -4),
            textcoords="offset points",
        )
    ax_a.set_title("Relative attenuation out of band", pad=12)
    ax_a.set_ylabel("Relative attenuation [dB]")
    ax_a.set_ylim(-115, 6)
    ax_a.set_xticks(x, names)
    ax_a.grid(axis="y", color=COLOR_GRID, linestyle="-")
    ax_a.legend(loc="lower right", fontsize=9)

    ax_b.bar(x, delays, 0.5, color=COLOR_SECONDARY)
    for xi, value in zip(x, delays, strict=True):
        ax_b.annotate(
            f"{value:.2f}",
            (xi, value),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )
    ax_b.set_title("Group delay at the band mid frequency", pad=12)
    ax_b.set_ylabel("Group delay [ms]")
    ax_b.set_ylim(0, 2.4)
    ax_b.set_xticks(x, names)
    ax_b.grid(axis="y", color=COLOR_GRID, linestyle="-")

    fig.suptitle("The 1 kHz octave band at 48 kHz, order 6")
    fig.tight_layout()
    save_figure(output_dir, "architecture_tradeoff.png")
    plt.close()


def generate_class_mask_architectures(output_dir: str) -> None:
    """The four architecture verdicts of the table, drawn on their own mask."""
    print("Generating class_mask_architectures.png...")

    fs = 48000
    cases = (
        ("butter", "Butterworth: passes"),
        ("cheby1", "Chebyshev I: passband ripple"),
        ("ellip", "Elliptic: ripple in both bands"),
        ("bessel", "Bessel: roll-off too slow"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (ftype, title) in zip(axes.ravel(), cases, strict=True):
        bank = filters.OctaveFilterBank(
            fs,
            fraction=1,
            order=6,
            limits=[800, 1200],
            design=filters.FilterDesign(filter_type=ftype),
        )
        result = filters.filter_class_compliance(bank)
        result.plot(ax=ax)
        verdict = result.overall_class
        ax.set_title(f"{title}   (overall_class = {verdict})", pad=10)
    fig.suptitle(
        "The same 1 kHz octave band, order 6, on the IEC 61260-1 acceptance mask"
    )
    fig.tight_layout()
    save_figure(output_dir, "class_mask_architectures.png")
    plt.close()


def generate_leakage_floor(output_dir: str) -> None:
    """What a 70 dB stopband looks like as a measured band spectrum."""
    print("Generating filter_leakage_floor.png...")
    from phonometry import signals

    fs = 48000
    t = np.arange(int(8.0 * fs)) / fs
    # A 1 kHz tone at 100 dB SPL over pink noise whose band levels are 25 dB,
    # so the real floor and the filter's own skirt are within reach of each
    # other and the crossing is the point of the figure.
    tone = np.sqrt(2) * (2e-5 * 10 ** (100 / 20)) * np.sin(2 * np.pi * 1000 * t)
    floor = signals.noise_signal(fs, 8.0, color="pink", rms=2.04e-3, seed=4)
    x = tone + floor

    calibration = filters.LevelCalibration(factor=1.0)
    _, ax = plt.subplots(figsize=(10, 5.4))
    for order, color, style in ((6, COLOR_PRIMARY, "-"), (10, COLOR_TERTIARY, "--")):
        levels, centres = filters.octave_filter(
            x, fs, fraction=3, order=order, calibration=calibration
        )
        ax.semilogx(
            centres,
            levels,
            color=color,
            linestyle=style,
            linewidth=1.8,
            label=f"measured band levels, order {order}",
        )
    truth, centres = filters.octave_filter(
        floor, fs, fraction=3, order=6, calibration=calibration
    )
    ax.semilogx(
        centres,
        truth,
        color=COLOR_SECONDARY,
        linestyle=":",
        linewidth=1.8,
        label="the noise actually present",
    )
    ax.annotate(
        "bands on the skirt are measuring\nthe filter, not the sound",
        xy=(500.0, 34.0),
        xytext=(30.0, 58.0),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.annotate(
        "here the skirt has fallen below the noise:\n"
        "these bands are measuring the sound",
        xy=(11000.0, 22.5),
        xytext=(1500.0, 10.0),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.set_title("A 1 kHz tone at 100 dB over a pink-noise floor", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Band level [dB re 20 µPa]")
    ax.set_ylim(5, 108)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-")
    ax.grid(which="minor", color=COLOR_GRID, linestyle=":", alpha=0.4)
    format_frequency_axis(ax, 11.0, 23000.0)
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "filter_leakage_floor.png")
    plt.close()


def generate_streaming_level_seams(output_dir: str) -> None:
    """The three streaming pitfalls, in the level domain the reader watches."""
    print("Generating streaming_level_seams.png...")

    fs, block = 48000, 4800  # 100 ms blocks
    rng = np.random.default_rng(17)
    steps = np.array([0.02, 0.02, 0.08, 0.08, 0.02, 0.02, 0.05, 0.05])
    x = np.concatenate([a * rng.standard_normal(block) for a in steps])
    t = np.arange(x.size) / fs
    n_blocks = steps.size

    def to_db(env: np.ndarray) -> np.ndarray:
        return np.asarray(10 * np.log10(np.maximum(env, 1e-16) / (2e-5) ** 2))

    # Every path here takes the default design: with the resampling gone,
    # streaming and single-shot weighting are the same cascade of sections at
    # the input rate, so what the figure shows is a state difference and
    # nothing else.
    continuous = to_db(
        filters.time_weighting(
            filters.weighting_filter(x, fs, curve="A"),
            fs,
            mode="fast",
        )
    )

    aw = filters.WeightingFilter(fs, "A", stateful=True)
    tw = filters.TimeWeighting(fs, mode="fast")
    stateful = np.concatenate(
        [
            to_db(np.asarray(tw.process(aw.filter(x[i * block : (i + 1) * block]))))
            for i in range(n_blocks)
        ]
    )

    stateless = np.concatenate(
        [
            to_db(
                filters.time_weighting(
                    filters.weighting_filter(
                        x[i * block : (i + 1) * block],
                        fs,
                        curve="A",
                    ),
                    fs,
                    mode="fast",
                )
            )
            for i in range(n_blocks)
        ]
    )

    seam = max(
        abs(float(stateless[i * block] - continuous[i * block]))
        for i in range(1, n_blocks)
    )
    exact = float(np.max(np.abs(stateful - continuous)))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [2.1, 1.0]}
    )
    for i in range(1, n_blocks):
        ax_a.axvline(i * block / fs, color=COLOR_GRID, linestyle=":", linewidth=1.0)
    ax_a.plot(
        t,
        continuous,
        color=COLOR_MUTED,
        linewidth=3.2,
        alpha=0.5,
        label="one continuous pass",
    )
    ax_a.plot(
        t,
        stateful,
        color=COLOR_PRIMARY,
        linewidth=1.4,
        label=f"stateful=True (max error {exact:.2g} dB)",
    )
    ax_a.plot(
        t,
        stateless,
        color=COLOR_SECONDARY,
        linewidth=1.2,
        label=f"stateful=False (up to {seam:.0f} dB at a seam)",
    )
    ax_a.set_title("Eight 100 ms blocks of a level-stepping signal", pad=12)
    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylabel(r"$L_{\mathrm{AF}}$ [dB re 20 µPa]")
    ax_a.set_xlim(0, t[-1])
    ax_a.set_ylim(30, 85)
    ax_a.grid(color=COLOR_GRID, linestyle="-")
    ax_a.legend(loc="lower right", fontsize=8)

    aw_s = filters.WeightingFilter(fs, "A", stateful=True, steady_ic=True)
    tw_s = filters.TimeWeighting(fs, mode="fast")
    shown = 8
    steady = np.concatenate(
        [
            to_db(np.asarray(tw_s.process(aw_s.filter(x[i * block : (i + 1) * block]))))
            for i in range(shown)
        ]
    )
    ax_b.axvspan(0.0, 0.625, color=COLOR_SECONDARY, alpha=0.20)
    ax_b.plot(
        t[: shown * block],
        continuous[: shown * block],
        color=COLOR_MUTED,
        linewidth=3.2,
        alpha=0.5,
        label="one continuous pass",
    )
    ax_b.plot(
        t[: shown * block],
        stateful[: shown * block],
        color=COLOR_PRIMARY,
        linewidth=1.4,
        label="steady_ic=False",
    )
    ax_b.plot(
        t[: shown * block],
        steady,
        color=COLOR_TERTIARY,
        linewidth=1.4,
        linestyle="--",
        label="steady_ic=True",
    )
    ax_b.annotate(
        r"$5\tau$ on Fast: 0.63 s",
        xy=(0.31, 82.0),
        ha="center",
        fontsize=9,
        color=COLOR_SECONDARY,
    )
    ax_b.set_title("The settling ramp, magnified", pad=12)
    ax_b.set_xlabel("Time [s]")
    ax_b.set_xlim(0, shown * block / fs)
    ax_b.set_ylim(30, 85)
    ax_b.grid(color=COLOR_GRID, linestyle="-")
    ax_b.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    save_figure(output_dir, "streaming_level_seams.png")
    plt.close()


def generate_survey_channel_average(output_dir: str) -> None:
    """Energy average against dB average over a five-position room survey."""
    print("Generating survey_channel_average.png...")
    from phonometry import signals

    fs = 48000
    rng = np.random.default_rng(23)
    # Five positions in one room: the same source, a modal field below about
    # 300 Hz where the positions disagree, and a diffuse field above it.
    base = signals.noise_signal(fs, 6.0, color="pink", rms=0.05, seed=1)
    survey = []
    for _ in range(5):
        modal = np.zeros_like(base)
        for f_mode in rng.uniform(40.0, 300.0, 40):
            t = np.arange(base.size) / fs
            modal += rng.uniform(0.03, 0.07) * np.sin(
                2 * np.pi * f_mode * t + rng.uniform(0, 2 * np.pi)
            )
        survey.append(base * rng.uniform(0.85, 1.15) + modal)
    x = np.stack(survey)

    spl, centres = filters.octave_filter(x, fs, fraction=3, limits=[25.0, 10000.0])
    spl = np.asarray(spl)
    energetic = 10 * np.log10(np.mean(10 ** (spl / 10), axis=0))
    arithmetic = np.mean(spl, axis=0)
    error = energetic - arithmetic
    worst = int(np.argmax(error))

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(10, 7.0), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0]}
    )
    for i, row in enumerate(spl):
        ax_a.semilogx(
            centres,
            row,
            color=COLOR_MUTED,
            linewidth=0.9,
            label="the five positions" if i == 0 else None,
        )
    ax_a.semilogx(
        centres,
        energetic,
        color=COLOR_PRIMARY,
        linewidth=2.4,
        label="energy average (correct)",
    )
    ax_a.semilogx(
        centres,
        arithmetic,
        color=COLOR_SECONDARY,
        linewidth=2.0,
        linestyle="--",
        label="arithmetic mean of the dB values",
    )
    ax_a.set_title("A five-position room survey in one octave_filter call", pad=12)
    ax_a.set_ylabel("Band level [dB re 20 µPa]")
    ax_a.grid(which="major", color=COLOR_GRID, linestyle="-")
    ax_a.grid(which="minor", color=COLOR_GRID, linestyle=":", alpha=0.4)
    ax_a.legend(loc="lower left", fontsize=9)

    ax_b.semilogx(centres, error, color=COLOR_SECONDARY, linewidth=1.8)
    ax_b.axhline(0.0, color=COLOR_MUTED, linewidth=1.0)
    ax_b.annotate(
        f"worst under-read {error[worst]:.1f} dB at {centres[worst]:.0f} Hz",
        xy=(centres[worst], error[worst]),
        xytext=(centres[worst] * 2.2, error[worst] * 0.85),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax_b.set_xlabel(LABEL_FREQ_HZ)
    ax_b.set_ylabel("Energy minus\ndB average [dB]")
    ax_b.grid(which="major", color=COLOR_GRID, linestyle="-")
    ax_b.grid(which="minor", color=COLOR_GRID, linestyle=":", alpha=0.4)
    format_frequency_axis(ax_b, 22.0, 12000.0)

    fig.tight_layout()
    save_figure(output_dir, "survey_channel_average.png")
    plt.close()
