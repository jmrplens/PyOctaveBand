#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the signals guides: filters, weightings, levels and spectra.

The instrument chain as the site teaches it -- octave and fractional-octave
filter banks, the frequency and time weightings, the level metrics they feed,
and the spectral, cepstral and correlation analyses applied to the recorded
signal. Everything here is embedded by a page under ``signals/``.
"""

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import signal as scipy_signal

from phonometry import OctaveFilterBank
from phonometry._plot.common import (
    format_frequency_axis,
    theme_fill,
    theme_fill_alpha,
)

from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    LABEL_LEVEL_DB,
    apply_axis_styling,
    measure_weighting_response,
    save_figure,
)


def generate_filter_type_comparison(output_dir: str) -> None:
    """Compare different filter architectures with a zoom inset."""
    print("Generating filter_type_comparison.png...")
    fs = 48000
    fraction = 1
    order = 6
    
    # We want exactly the 1000Hz band
    limits = [800.0, 1200.0]
    
    filters = [
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
    axins.set_xscale("log") # Explicitly set log scale
    
    for f_type, label, color, style in filters:
        bank = OctaveFilterBank(fs, fraction=fraction, order=order, limits=limits, filter_type=f_type)
        
        # Find index of 1000Hz band
        idx = np.argmin(np.abs(np.array(bank.freq) - 1000))
        
        fsd = fs / bank.factor[idx]
        w, h = scipy_signal.sosfreqz(bank.sos[idx], worN=16384, fs=fsd)
        mag_db = 20 * np.log10(np.abs(h) + 1e-9)
        
        ax.semilogx(w, mag_db, label=label, color=color, linestyle=style)
        axins.plot(w, mag_db, color=color, linestyle=style)

    ax.axhline(-3, color=COLOR_FG, linestyle=":", alpha=0.3, label="-3 dB")
    axins.axhline(-3, color=COLOR_FG, linestyle=":", alpha=0.3)
    
    apply_axis_styling(ax, "Filter Architecture Comparison (Order 6, 1kHz Band)", xlim=(100, 8000), ylim=(-80, 5))
    
    # Sub-plot styling (Zoom around 1kHz and -3dB)
    axins.set_xlim(650, 1500)
    axins.set_ylim(-4, 0.5)  # Adjusted: from -4 to 0.5
    axins.grid(True, which="both", alpha=0.3)
    axins.set_title("Zoom at -3 dB (Log Scale)", fontsize=9)

    # Fix x-ticks for log scale zoom to look right
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    axins.xaxis.set_major_formatter(ScalarFormatter())
    axins.xaxis.set_minor_formatter(NullFormatter())  # Hide minor tick labels
    axins.xaxis.get_major_formatter().set_scientific(False)  # Disable scientific notation
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
            bank = OctaveFilterBank(fs=fs, fraction=fraction, order=order, limits=[12.0, 20000.0], filter_type=f_type)
            
            from phonometry.filters.design import _showfilter
            # Draw first, then save through save_figure so the Spanish
            # translation pass runs on the finished figure (it rewrites the
            # live figure's text artists right before the save).
            _showfilter(bank.sos, bank.freq, bank.freq_u, bank.freq_d, fs,
                        bank.factor, show=False, plot_file=None, close=False)
            save_figure(output_dir, filename, dpi=150,
                        bbox_inches="tight")
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
        bank = OctaveFilterBank(fs=fs, fraction=frac, order=6, limits=[12.0, 20000.0])
        spl, freq = bank.filter(y)

        _, ax = plt.subplots()
        
        # Plot PSD of raw signal in background
        # We need to scale PSD to comparable levels. 
        # A simple hack for visualization is to align the max of PSD to max of SPL
        f_psd, Pxx = scipy_signal.welch(y, fs, nperseg=8192)
        Pxx_db = 10 * np.log10(Pxx + 1e-12)
        # Shift PSD to match SPL peak roughly
        Pxx_db += (np.max(spl) - np.max(Pxx_db)) - 5 # Shift slightly below
        
        ax.semilogx(f_psd, Pxx_db, color="gray", alpha=0.6, linewidth=1.2, label="Raw Signal Spectrum (PSD)", zorder=0)
        
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
            label=f"Measured 1/{frac} Octave Bands"
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
    b, a = scipy_signal.butter(1, 0.04) # -3dB/oct approx
    ch1 = scipy_signal.lfilter(b, a, white)
    ch1 = (ch1 - np.mean(ch1)) / np.max(np.abs(ch1))

    # Channel 2: Logarithmic Sine Sweep
    ch2 = scipy_signal.chirp(t, f0=50, t1=duration, f1=10000, method="logarithmic")

    x = np.vstack((ch1, ch2))
    bank = OctaveFilterBank(fs=fs, fraction=3, order=6, limits=[20.0, 20000.0])
    spl, freq = bank.filter(x)

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Calculate PSDs for background
    f_psd1, Pxx1 = scipy_signal.welch(x[0], fs, nperseg=4096)
    Pxx_db1 = 10 * np.log10(Pxx1 + 1e-12)
    Pxx_db1 += (np.max(spl[0]) - np.max(Pxx_db1)) # Align peaks
    
    f_psd2, Pxx2 = scipy_signal.welch(x[1], fs, nperseg=4096)
    Pxx_db2 = 10 * np.log10(Pxx2 + 1e-12)
    Pxx_db2 += (np.max(spl[1]) - np.max(Pxx_db2)) # Align peaks

    # Plot Left Channel
    ax1.semilogx(f_psd1, Pxx_db1, color="gray", alpha=0.6, linewidth=1.2, label="Raw PSD", zorder=0)
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
    ax2.semilogx(f_psd2, Pxx_db2, color="gray", alpha=0.6, linewidth=1.2, label="Raw PSD", zorder=0)
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
    ax2.set_title("") # Remove title from bottom plot
    ax2.legend(loc="lower right")
    # Let Y-axis autoscale

    plt.tight_layout()
    save_figure(output_dir, "signal_response_multichannel.png")
    plt.close()


def generate_decomposition_plot(output_dir: str) -> None:
    """Generate time-domain decomposition plot comparing two filter types (Butterworth vs Chebyshev II)."""
    print("Generating signal_decomposition.png with comparison (Butter vs Cheby2) @ 48kHz...")
    fs = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Signal: sum of 250Hz and 1000Hz sines
    y = np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 1000 * t)

    # Filter into 1/1 octave bands with two different architectures
    # We use Chebyshev II (flat passband, no ripple)
    bank_butter = OctaveFilterBank(fs=fs, fraction=1, order=6, limits=[100.0, 2000.0], filter_type="butter")
    bank_cheby2 = OctaveFilterBank(fs=fs, fraction=1, order=6, limits=[100.0, 2000.0], filter_type="cheby2")
    
    # Cast to 3-tuple to satisfy mypy unpacking
    _, freq, xb_butter = bank_butter.filter(y, sigbands=True)
    
    _, _, xb_cheby2 = bank_cheby2.filter(y, sigbands=True)

    if xb_butter is None or xb_cheby2 is None:
        raise ValueError("Signal bands should not be None")

    num_plots = len(xb_butter) + 2 # +1 for original, +1 for impulse response
    _fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.2 * num_plots), sharex=False)


    # Fixed Y limits for decomposition
    y_lim = (-2.8, 2.8)

    # 1. Original Signal
    axes[0].plot(t, y, color=COLOR_FG, linewidth=1.5)
    axes[0].set_title("Original Signal (250 Hz + 1000 Hz Sum) @ 48 kHz", fontweight="bold")
    axes[0].set_ylim(y_lim)
    axes[0].set_xlim(0, 0.04)

    # 2. Filtered Bands Comparison
    for i, (f_center) in enumerate(freq):
        axes[i + 1].plot(t, xb_butter[i], color=COLOR_PRIMARY, linewidth=1.5, label="Butterworth (Flat)")
        axes[i + 1].plot(t, xb_cheby2[i], color=COLOR_SECONDARY, linewidth=1.2, linestyle="--", alpha=0.9, label="Chebyshev II")
        axes[i + 1].set_title(f"Octave Band: {f_center:.0f} Hz", fontsize=11, fontweight="bold")
        axes[i + 1].set_ylim(y_lim)
        axes[i + 1].set_xlim(0, 0.04)
        if i == 0:
            axes[i+1].legend(loc="upper right", fontsize=9, framealpha=0.8)

    # 3. Impulse Response (Stability/Transient Visualization)
    impulse = np.zeros(len(t))
    impulse[0] = 1.0
    _, _, ir_butter = bank_butter.filter(impulse, sigbands=True)
    _, _, ir_cheby2 = bank_cheby2.filter(impulse, sigbands=True)
    
    idx_1000 = np.argmin(np.abs(np.array(freq) - 1000))
    axes[-1].plot(t, ir_butter[idx_1000], color=COLOR_PRIMARY, linewidth=1.5, label="Butterworth")
    axes[-1].plot(t, ir_cheby2[idx_1000], color=COLOR_SECONDARY, linewidth=1.2, linestyle="--", alpha=0.9, label="Chebyshev II")
    axes[-1].set_title(f"Impulse Response ({freq[idx_1000]:.0f} Hz Band) - Transient/Stability Comparison", fontweight="bold")
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
    ax: Any,
    fs: int,
    curves: tuple[tuple[str, str, str, str, float], ...],
    inset: Any = None,
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
        ax.semilogx(w, mag_db, label=label, color=color, linestyle=style, linewidth=width)
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
    apply_axis_styling(ax, "Frequency Weighting Curves (IEC 61672-1)",
                       xlim=(10, 22000), ylim=(-72, 15))

    axins.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.4, linewidth=1)
    axins.set_xlim(500, 8000)
    axins.set_ylim(-3, 2)
    axins.grid(True, which="both", alpha=0.3)
    axins.set_title("Zoom: A-weighting is positive (max +1.27 dB @ 2.5 kHz)", fontsize=9)
    axins.annotate(
        "+1.27 dB", xy=(2500, 1.27), xytext=(4200, 1.55), fontsize=8,
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

    apply_axis_styling(ax, "Special Weighting Curves (B, D, AU)",
                       xlim=(10, 40000), ylim=(-90, 18))
    # apply_axis_styling stops labelling at 16 kHz; this axis runs into the
    # ultrasonic decade AU exists for, so it needs the 40 kHz end labelled.
    ax.set_xticks([16, 63, 250, 1000, 4000, 16000, 40000])
    ax.set_xticklabels(["16", "63", "250", "1k", "4k", "16k", "40k"])

    # The two numbers the guide quotes from the standards: the D hump of
    # IEC 537 (+11.5 dB at 3.15 kHz, NASA CR-3406 Table SLD-I) and the U
    # low-pass of IEC 61012 (-13 dB at 16 kHz, Table 1).
    ax.annotate(
        "+11.5 dB @ 3.15 kHz", xy=(3150, 11.6), xytext=(150, 13.5), fontsize=9,
        color="#9467bd", arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#9467bd"},
    )
    ax.annotate(
        "AU is 13 dB below A at 16 kHz", xy=(16000, -21.0), xytext=(700, -35.0),
        fontsize=9, color="#ff7f0e",
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#ff7f0e"},
    )

    ax.legend(loc="lower center", fontsize=9)
    save_figure(output_dir, "special_weighting_responses.png")
    plt.close()


def generate_g_weighting_response(output_dir: str) -> None:
    """Plot the ISO 7196 G-weighting curve against the Table 2 nominals."""
    print("Generating g_weighting_response.png...")
    from scipy import signal as sp_signal

    from phonometry import WeightingFilter

    fs = 48000
    # ISO 7196:1995 Table 2 - nominal one-third-octave frequency, response dB
    table2 = [
        (0.25, -88.0), (0.5, -64.3), (1.0, -43.0), (2.0, -28.3),
        (4.0, -16.0), (8.0, -4.0), (10.0, 0.0), (16.0, 7.7), (20.0, 9.0),
        (31.5, -4.0), (63.0, -28.0), (125.0, -52.0), (250.0, -76.0),
    ]
    freqs = np.logspace(np.log10(0.1), np.log10(1000), 800)
    sos = WeightingFilter(fs, "G").sos
    _, h = sp_signal.sosfreqz(sos, worN=freqs, fs=fs)
    mag_db = 20 * np.log10(np.abs(h))

    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs, mag_db, color=COLOR_PRIMARY, label="G-weighting (ISO 7196)")
    tf = [f for f, _ in table2]
    tv = [v for _, v in table2]
    ax.plot(tf, tv, "o", color=COLOR_SECONDARY, markersize=5,
            label="ISO 7196 Table 2 nominals", zorder=5)
    ax.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)
    ax.axvline(10, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)
    ax.annotate("0 dB @ 10 Hz", xy=(10, 0), xytext=(20, -18), fontsize=9,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    apply_axis_styling(
        ax, "G Frequency Weighting for Infrasound (ISO 7196:1995)",
        xlim=(0.1, 1000), ylim=(-95, 15),
    )
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ticks = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 125, 315, 1000]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "5", "10", "20", "50", "125", "315", "1k"])
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
    
    from phonometry import time_weighting
    
    # Square for energy
    x_sq = x**2
    fast = time_weighting(x, fs, mode="fast")
    slow = time_weighting(x, fs, mode="slow")
    impulse = time_weighting(x, fs, mode="impulse")
    
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
    ax.plot(t, impulse, color="purple", linestyle="-.", linewidth=1.5, label="Impulse (35ms/1.5s)")
    
    ax.set_title("Time Weighting Ballistics (IEC 61672-1)", fontweight="bold")
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
    
    from phonometry import linkwitz_riley
    
    # Frequency analysis
    # Measure response using IR
    impulse = np.zeros(fs)
    impulse[0] = 1.0
    lp_ir, hp_ir = linkwitz_riley(impulse, fs, freq=1000, order=4)
    
    w, h_lp = scipy_signal.freqz(lp_ir, worN=8192, fs=fs)
    _, h_hp = scipy_signal.freqz(hp_ir, worN=8192, fs=fs)
    
    _, ax = plt.subplots()
    ax.semilogx(w, 20 * np.log10(np.abs(h_lp) + 1e-9), color=COLOR_PRIMARY, label="Low Pass (LR4)")
    ax.semilogx(w, 20 * np.log10(np.abs(h_hp) + 1e-9), color=COLOR_SECONDARY, label="High Pass (LR4)")
    ax.semilogx(w, 20 * np.log10(np.abs(h_lp + h_hp) + 1e-9), color=COLOR_FG, linestyle="--", label="Sum (Flat)")

    apply_axis_styling(ax, "Linkwitz-Riley Crossover (4th Order @ 1kHz)", xlim=(20, 20000), ylim=(-60, 5))
    ax.legend(loc="lower right")
    save_figure(output_dir, "crossover_lr4.png")
    plt.close()


def generate_parametric_eq_family(output_dir: str) -> None:
    """Magnitude responses of the RBJ Audio EQ Cookbook biquad family."""
    print("Generating parametric_eq_family.png...")
    fs = 48000

    from phonometry import EQSection, ParametricEQ

    family = [
        (EQSection("peaking", 1000.0, gain_db=6.0, q=1.4),
         "Peaking +6 dB (Q = 1.4)", COLOR_PRIMARY, "-"),
        (EQSection("lowshelf", 125.0, gain_db=6.0),
         "Low shelf +6 dB", COLOR_TERTIARY, "-"),
        (EQSection("highshelf", 4000.0, gain_db=-6.0),
         "High shelf -6 dB", "#9467bd", "-"),
        (EQSection("lowpass", 10000.0),
         "Low-pass (Q = 0.707)", COLOR_SECONDARY, "--"),
        (EQSection("highpass", 50.0),
         "High-pass (Q = 0.707)", "#8c564b", "--"),
        (EQSection("bandpass", 500.0, q=2.0),
         "Band-pass (Q = 2)", "#ff7f0e", "-."),
        (EQSection("notch", 2000.0, q=6.0),
         "Notch (Q = 6)", "#17becf", "-."),
    ]

    _, ax = plt.subplots(figsize=(10, 6))
    for section, label, color, style in family:
        res = ParametricEQ(fs, section).response(f_min=20.0, f_max=20000.0)
        ax.semilogx(res.frequencies, res.magnitude_db,
                    label=label, color=color, linestyle=style)

    ax.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)
    apply_axis_styling(ax, "Parametric EQ Biquads (RBJ Audio EQ Cookbook)",
                       xlim=(20, 20000), ylim=(-27, 9))
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
    x[int(1.0 * fs):int(1.3 * fs)] += np.sin(2 * np.pi * 4000 * t[int(1.0 * fs):int(1.3 * fs)])
    x[int(2.5 * fs):int(2.8 * fs)] += np.sin(2 * np.pi * 250 * t[int(2.5 * fs):int(2.8 * fs)])
    x += 0.01 * rng.standard_normal(len(t))

    # 1/12-octave bands stepped at 1/8 of the window: the Fast (125 ms)
    # integration still sets the time resolution, but the mesh is sampled four
    # times finer on each axis than the band spacing and hop it replaces, so
    # the sweep reads as a line instead of a staircase of cells.
    bank = OctaveFilterBank(fs=fs, fraction=12, order=6, limits=[50.0, 12000.0])
    levels, freq, times = bank.spectrogram(x, window_time=0.125, overlap=0.875)

    _, ax = plt.subplots()
    mesh = ax.pcolormesh(times, freq, levels, shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_title("1/12 Octave Spectrogram (Fast windows, 87.5% overlap)",
                 fontweight="bold", pad=12)
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

    from phonometry import ln_levels, time_weighting

    # Fluctuating "traffic-like" noise: background + random events
    rng = np.random.default_rng(42)
    x = 0.05 * rng.standard_normal(len(t))
    for _ in range(12):
        start = rng.uniform(1, duration - 3)
        length = rng.uniform(0.5, 2.0)
        idx = (t >= start) & (t < start + length)
        envelope = np.hanning(int(idx.sum()))
        x[idx] += envelope * rng.uniform(0.3, 1.0) * rng.standard_normal(int(idx.sum()))

    envelope_ms = time_weighting(x, fs, mode="fast")
    level_t = 10 * np.log10(np.maximum(envelope_ms, 1e-12) / (2e-5) ** 2)
    stats = ln_levels(x, fs, n=(10, 50, 90))

    _, ax = plt.subplots()
    ax.plot(t, level_t, color=COLOR_PRIMARY, linewidth=0.8, label="Fast level $L_p(t)$")
    for n_value, color, style in [(10, COLOR_SECONDARY, "--"), (50, COLOR_FG, "-"), (90, COLOR_TERTIARY, "-.")]:
        ax.axhline(
            float(stats[n_value]), color=color, linestyle=style, linewidth=1.5,
            label=f"L{n_value} = {float(stats[n_value]):.1f} dB",
        )
    ax.set_title("Statistical Levels L10 / L50 / L90 (Fast envelope)", fontweight="bold", pad=12)
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

    bank = OctaveFilterBank(fs=fs, fraction=1, order=6, limits=[200.0, 300.0])
    _, _, bands_fwd = bank.filter(x, sigbands=True, calculate_level=False)
    _, _, bands_zp = bank.filter(x, sigbands=True, calculate_level=False, zero_phase=True)

    _, ax = plt.subplots()
    ax.plot(t, x, color="gray", alpha=0.5, linewidth=1.0, label="Input burst (250 Hz)")
    ax.plot(t, bands_fwd[0], color=COLOR_PRIMARY, linewidth=1.3, label="Causal filtering (group delay)")
    ax.plot(t, bands_zp[0], color=COLOR_SECONDARY, linewidth=1.3, linestyle="--", label="zero_phase=True (aligned)")
    ax.set_title("Zero-Phase Filtering: Group Delay Elimination (250 Hz Band)", fontweight="bold", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    save_figure(output_dir, "zero_phase_comparison.png")
    plt.close()


def generate_weighting_accuracy_hf(output_dir: str) -> None:
    """Compare A-weighting HF accuracy: analytic vs bilinear vs high_accuracy."""
    print("Generating weighting_accuracy_hf.png...")
    fs = 48000

    from phonometry import WeightingFilter

    freqs = np.logspace(np.log10(1000), np.log10(20000), 40)

    def analytic_a(f: np.ndarray) -> np.ndarray:
        ra = (12194**2 * f**4) / (
            (f**2 + 20.6**2)
            * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
            * (f**2 + 12194**2)
        )
        return np.asarray(20 * np.log10(ra) + 2.0)

    def measured_gains(wf: WeightingFilter) -> np.ndarray:
        gains = []
        for f0 in freqs:
            tt = np.arange(int(fs * 0.2)) / fs
            x = np.sin(2 * np.pi * f0 * tt)
            y = wf.filter(x)
            n0 = int(0.05 * fs)  # skip filter transient
            gains.append(20 * np.log10(np.std(y[n0:]) / np.std(x[n0:])))
        return np.array(gains)

    legacy = measured_gains(WeightingFilter(fs, "A", high_accuracy=False))
    accurate = measured_gains(WeightingFilter(fs, "A"))
    reference = analytic_a(freqs)

    _, (ax, ax_err) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax.semilogx(freqs, reference, color=COLOR_FG, linewidth=2, label="IEC 61672-1 analytic curve")
    ax.semilogx(freqs, legacy, color=COLOR_SECONDARY, linestyle="--", label="Plain bilinear (high_accuracy=False)")
    ax.semilogx(freqs, accurate, color=COLOR_PRIMARY, linestyle="-.", label="Oversampled (high_accuracy=True)")
    ax.set_title(f"A-Weighting High-Frequency Accuracy @ fs={fs//1000} kHz", fontweight="bold", pad=12)
    ax.set_ylabel(LABEL_LEVEL_DB)
    ax.legend(loc="lower left")

    ax_err.semilogx(freqs, legacy - reference, color=COLOR_SECONDARY, linestyle="--", label="Bilinear error")
    ax_err.semilogx(freqs, accurate - reference, color=COLOR_PRIMARY, linestyle="-.", label="high_accuracy error")
    ax_err.axhline(-2.5, color="gray", linestyle=":", label="Class 1 lower limit @ 12.5 kHz")
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

    filters = [
        ("butter", "Butterworth", COLOR_PRIMARY, "-"),
        ("cheby1", "Chebyshev I", COLOR_SECONDARY, "--"),
        ("cheby2", "Chebyshev II", COLOR_TERTIARY, ":"),
        ("ellip", "Elliptic", "#9467bd", "-."),
        ("bessel", "Bessel", "#8c564b", "-"),
    ]

    _, ax = plt.subplots()
    for f_type, label, color, style in filters:
        bank = OctaveFilterBank(fs, fraction=1, order=6, limits=limits, filter_type=f_type)
        idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
        fsd = fs / bank.factor[idx]
        # Group delay of an SOS cascade = sum of the sections' group delays.
        w = np.logspace(np.log10(500), np.log10(2000), 1024)
        gd = np.zeros_like(w)
        for section in bank.sos[idx]:
            _w_s, gd_s = scipy_signal.group_delay((section[:3], section[3:]), w=w, fs=fsd)
            gd += gd_s
        ax.semilogx(w, gd / fsd * 1000, label=label, color=color, linestyle=style)

    ax.set_title("Group Delay Comparison (1 kHz Octave Band, Order 6)", fontweight="bold", pad=12)
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

    from phonometry import time_weighting

    cases = [(0.2, -1.0), (0.05, -4.8), (0.01, -11.1)]  # Table 4, class 1 rows
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    t_all = np.arange(int(fs * 2.0)) / fs
    steady = np.sin(2 * np.pi * 4000 * t_all)
    ref = time_weighting(steady, fs, mode="fast")[int(1.5 * fs):].mean()

    for ax, (duration, target) in zip(axes, cases):
        burst = np.zeros_like(t_all)
        start = int(0.5 * fs)
        burst[start:start + round(duration * fs)] = steady[start:start + round(duration * fs)]
        env_db = 10 * np.log10(np.maximum(time_weighting(burst, fs, mode="fast") / ref, 1e-6))

        ax.plot(t_all, env_db, color=COLOR_PRIMARY, linewidth=1.3, label="FAST envelope")
        ax.axhline(target, color=COLOR_SECONDARY, linestyle="--", linewidth=1.2,
                   label=f"IEC target {target} dB")
        ax.set_title(f"{duration * 1000:g} ms burst", fontsize=11, fontweight="bold")
        ax.set_xlim(0.4, 1.4)
        ax.set_ylim(-30, 3)
        ax.set_xlabel("Time [s]")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Level re steady state [dB]")

    fig.suptitle("4 kHz Toneburst Response vs IEC 61672-1 Table 4 (FAST)", fontweight="bold")
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
        bank = OctaveFilterBank(fs, fraction=1, limits=[900, 1100],
                                stateful=stateful, resample=False)
        if stateful:
            parts = [
                bank.filter(x[i * block:(i + 1) * block], sigbands=True,
                            detrend=False, calculate_level=False)[2][0]
                for i in range(n_blocks)
            ]
        else:
            parts = []
            for i in range(n_blocks):
                b2 = OctaveFilterBank(fs, fraction=1, limits=[900, 1100], resample=False)
                parts.append(b2.filter(x[i * block:(i + 1) * block], sigbands=True,
                                       detrend=False, calculate_level=False)[2][0])
        return np.concatenate(parts)

    continuous = OctaveFilterBank(fs, fraction=1, limits=[900, 1100], resample=False).filter(
        x, sigbands=True, detrend=False, calculate_level=False)[2][0]
    y_stateful = band_output(stateful=True)
    y_stateless = band_output(stateful=False)

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    zoom = slice(int(0.9 * block), int(1.4 * block))  # around the first boundary

    ax1.plot(t[zoom], continuous[zoom], color=COLOR_FG, linewidth=2.2, alpha=0.35,
             label="Continuous (whole signal)")
    ax1.plot(t[zoom], y_stateful[zoom], color=COLOR_PRIMARY, linewidth=1.1,
             label="Stateful blocks (state carried)")
    ax1.set_title("stateful=True: block outputs equal the continuous result",
                  fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(t[zoom], continuous[zoom], color=COLOR_FG, linewidth=2.2, alpha=0.35,
             label="Continuous (whole signal)")
    ax2.plot(t[zoom], y_stateless[zoom], color=COLOR_SECONDARY, linewidth=1.1,
             label="Independent blocks (state reset)")
    ax2.axvline(block / fs, color=COLOR_FG, linestyle=":", alpha=0.6)
    # The callout lands on top of the dense trace, so it carries a solid
    # panel of its own instead of relying on the gaps between the waveforms.
    ax2.annotate("block boundary:\nfilter transient restarts", xy=(block / fs, 0),
                 xytext=(block / fs + 0.02, ax2.get_ylim()[0] * 0.55 if ax2.get_ylim()[0] < 0 else -1),
                 fontsize=9.5, color=COLOR_FG, zorder=6,
                 bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                       "edgecolor": COLOR_GRID},
                 arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax2.set_title("No state: each block restarts the filter transient",
                  fontsize=11, fontweight="bold")
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

    bank = OctaveFilterBank(fs, fraction=1, order=6, limits=[800, 1200], filter_type="butter")
    idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
    fm = bank.freq[idx]
    fsd = fs / bank.factor[idx]
    w, h = scipy_signal.sosfreqz(bank.sos[idx], worN=2 ** 15, fs=fsd)
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
    ax.fill_between(grid, -10, lo1, color=theme_fill(COLOR_SECONDARY, ax),
                    zorder=0,
                    label="Forbidden for class 1 (too little attenuation)")
    finite = np.isfinite(hi1)
    ax.fill_between(grid[finite], hi1[finite], 90,
                    color=theme_fill("#9467bd", ax), zorder=0,
                    label="Forbidden for class 1 (too much attenuation)")
    ax.plot(grid, lo2, color=COLOR_TERTIARY, linestyle=":", linewidth=1.2,
            label="Class 2 minimum attenuation")

    ax.plot(omega, delta_a, color=COLOR_PRIMARY, linewidth=1.6,
            label="Butterworth order 6 (1 kHz octave band)")

    ax.set_xscale("log")
    ax.set_xlim(0.08, 8)
    ax.set_ylim(-6, 90)
    ax.set_title("Relative Attenuation vs IEC 61260-1:2014 Class Limits", fontweight="bold", pad=12)
    ax.set_xlabel("Normalized frequency  f / fm")
    ax.set_ylabel("Relative attenuation ΔA [dB]")
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

    bank = OctaveFilterBank(fs, fraction=1, order=6, limits=[800, 1200], filter_type="butter")
    idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
    fm = bank.freq[idx]
    fsd = fs / bank.factor[idx]
    w, h = scipy_signal.sosfreqz(bank.sos[idx], worN=2 ** 15, fs=fsd)
    attenuation = -20 * np.log10(np.abs(h) + 1e-12)
    a_ref = float(np.interp(fm, w, attenuation))
    omega = w / fm

    # Restrict to the pass-band [G**-1/2, G**+1/2] where a finite max applies
    # (beyond the band edges the maximum limit is +inf, so plotting there would
    # misleadingly show the filter's natural roll-off "exceeding" a corridor).
    g_octave = 10 ** (3 / 10)  # octave ratio G (IEC 61260)
    edge_lo, edge_hi = g_octave ** -0.5, g_octave ** 0.5
    pb = (omega >= edge_lo) & (omega <= edge_hi)
    omega, delta_a = omega[pb], (attenuation - a_ref)[pb]
    grid = np.linspace(edge_lo, edge_hi, 1500)

    _, ax = plt.subplots(figsize=(10, 6.5))
    # Nested min/max corridors: class 0 (+-0.15 dB reference) is the tightest.
    for cls, colour, name in ((2, COLOR_TERTIARY, "Class 2 corridor"),
                              (1, COLOR_SECONDARY, "Class 1 corridor"),
                              (0, COLOR_PRIMARY, "Class 0 corridor")):
        lo, hi = class_limits(1.0, cls, grid, edition="1995")
        ax.plot(grid, hi, color=colour, linewidth=1.4, label=name)
        ax.plot(grid, lo, color=colour, linewidth=1.4)
    ax.plot(omega, delta_a, color=COLOR_FG, linewidth=2.2,
            label="Butterworth order 6 (1 kHz octave band)")

    ax.set_xscale("log")
    ax.set_xlim(edge_lo, edge_hi)
    ax.set_ylim(-0.7, 6)
    ax.set_title("Pass-band Class 0/1/2 Limits (IEC 61260:1995 / ANSI S1.11-2004)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Normalized frequency  f / fm")
    ax.set_ylabel("Relative attenuation ΔA [dB]")
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
    from phonometry import (
        WeightingFilter,
        verify_weighting_class,
        weighting_class_limits,
    )

    freqs, lower1, upper1 = weighting_class_limits(1)
    _, lower2, upper2 = weighting_class_limits(2)
    floor, ceil = -7.0, 7.0  # plotting bounds; -inf limits clip to the floor
    lo1 = np.clip(lower1, floor, ceil)
    lo2 = np.clip(lower2, floor, ceil)

    _, ax = plt.subplots(figsize=(10, 6.5))
    # Allowed corridor for class 1 (between lower and upper limit).
    ax.fill_between(freqs, lo1, upper1, color=theme_fill(COLOR_PRIMARY, ax),
                    step="mid", label="Class 1 acceptance region")
    ax.plot(freqs, upper1, color=COLOR_SECONDARY, linewidth=1.3, drawstyle="steps-mid",
            label="Class 1 upper/lower limit")
    ax.plot(freqs, lo1, color=COLOR_SECONDARY, linewidth=1.3, drawstyle="steps-mid")
    ax.plot(freqs, upper2, color=COLOR_TERTIARY, linestyle=":", linewidth=1.1,
            drawstyle="steps-mid", label="Class 2 upper/lower limit")
    ax.plot(freqs, lo2, color=COLOR_TERTIARY, linestyle=":", linewidth=1.1,
            drawstyle="steps-mid")

    for curve, colour, marker in (("A", COLOR_PRIMARY, "o"), ("C", "#9467bd", "s")):
        result = verify_weighting_class(WeightingFilter(48000, curve))
        f = np.array([b["freq"] for b in result["bands"]])
        dev = np.array([b["deviation_db"] for b in result["bands"]])
        ax.plot(f, dev, color=colour, linewidth=1.6, marker=marker, markersize=4,
                label=f"{curve} weighting deviation (48 kHz)")

    ax.set_xscale("log")
    ax.set_xlim(10, 20000)
    ax.set_ylim(floor, ceil)
    format_frequency_axis(ax, 10, 20000)
    ax.set_title("Weighting Deviation vs IEC 61672-1:2013 Table 3 Limits",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Deviation from design goal [dB]")
    ax.grid(which="both", color=COLOR_GRID, linestyle=":", alpha=0.4)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    save_figure(output_dir, "weighting_class_mask.png")
    plt.close()


def generate_calibration_stability(output_dir: str) -> None:
    """Stable vs unstable calibration tone against the IEC 60942 limit."""
    print("Generating calibration_stability.png...")
    from phonometry import time_weighting

    fs = 48000
    seconds = 6.0
    tt = np.arange(int(fs * seconds)) / fs
    stable = 0.5 * np.sin(2 * np.pi * 1000 * tt)
    # 3 % amplitude modulation at 2 Hz: ~0.14 dB deviation, clearly over
    unstable = 0.5 * (1 + 0.03 * np.sin(2 * np.pi * 2.0 * tt)) * np.sin(2 * np.pi * 1000 * tt)

    _, ax = plt.subplots(figsize=(10, 6))
    skip = fs  # discard the F-integrator attack (~8*tau = 1 s)
    for x, color, label in [
        (stable, COLOR_PRIMARY, "Stable tone (good coupling)"),
        (unstable, COLOR_SECONDARY, "3% AM tone (loose coupling)"),
    ]:
        env = time_weighting(x, fs, mode="fast")[skip:]
        level = 10 * np.log10(np.maximum(env, np.finfo(float).eps))
        rel = level - np.mean(level)
        ax.plot(tt[skip:], rel, color=color, linewidth=1.4, label=label)

    ax.axhline(0.07, color=COLOR_FG, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(-0.07, color=COLOR_FG, linestyle="--", linewidth=1.2, alpha=0.7,
               label="IEC 60942:2017 class 1 limit (deviation from mean)")
    ax.fill_between([1, seconds], -0.07, 0.07, color=COLOR_PRIMARY, alpha=0.06)
    ax.set_title("Calibration Tone Stability Check (IEC 60942:2017, 5.3.3)",
                 fontweight="bold", pad=12)
    ax.set_xlim(1, seconds)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("F-weighted level re mean [dB]")
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "calibration_stability.png")
    plt.close()


def generate_sel_concept(output_dir: str) -> None:
    """SEL: the whole event compressed into one second of equal energy."""
    print("Generating sel_concept.png...")
    from phonometry import leq, sel, time_weighting

    fs = 48000
    seconds = 8.0
    tt = np.arange(int(fs * seconds)) / fs
    rng = np.random.default_rng(11)
    # A vehicle pass-by: noise with a gaussian energy envelope
    envelope = np.exp(-0.5 * ((tt - 4.0) / 1.1) ** 2)
    x = envelope * rng.standard_normal(tt.size) * 0.3

    env = time_weighting(x, fs, mode="fast")
    level = 10 * np.log10(np.maximum(env, 1e-12))
    l_sel = float(sel(x, fs, dbfs=True))
    l_eq = float(leq(x, fs, dbfs=True))

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tt, level, color=COLOR_PRIMARY, linewidth=1.2,
            label="Fast level of the event")
    ax.hlines(l_eq, 0, seconds, color=COLOR_TERTIARY, linestyle="--",
              linewidth=1.6, label="Leq over the whole event")
    # SEL: same energy squeezed into 1 s (drawn as a 1 s block)
    ax.fill_between([3.5, 4.5], -55, l_sel, color=COLOR_SECONDARY, alpha=0.25)
    ax.hlines(l_sel, 3.5, 4.5, color=COLOR_SECONDARY, linewidth=2.2,
              label="SEL: same energy in 1 s")
    ax.annotate("equal energy", xy=(4.5, l_sel - 3), xytext=(5.6, l_sel - 1),
                fontsize=10, arrowprops={"arrowstyle": "->", "lw": 0.9})
    ax.set_title("Sound Exposure Level: the event normalized to 1 s",
                 fontweight="bold", pad=12)
    ax.set_xlim(0, seconds)
    ax.set_ylim(-55, l_sel + 6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level [dBFS]")
    ax.legend(loc="lower left", fontsize=9)
    save_figure(output_dir, "sel_concept.png")
    plt.close()


def generate_psd_confidence_smoothing(output_dir: str) -> None:
    """Calibrated PSD of pink noise: chi-square CI plus 1/3-oct smoothing."""
    print("Generating psd_confidence_smoothing...")
    from phonometry import (
        fractional_octave_smoothing,
        noise_signal,
        power_spectral_density,
    )

    fs = 48000.0
    x = noise_signal(fs, 20.0, color="pink", seed=11)
    res = power_spectral_density(x, fs, nperseg=4096)
    band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    freqs = res.frequencies[band]
    smooth = fractional_octave_smoothing(res.frequencies, res.psd, 3.0)[band]
    # The exact -3.01 dB/oct power law through the level at 1 kHz.
    i0 = int(np.argmin(np.abs(freqs - 1000.0)))
    ref_db = 10.0 * np.log10(smooth[i0]) - 10.0 * np.log10(freqs / freqs[i0])

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(
        freqs,
        10.0 * np.log10(res.ci_lower[band]),
        10.0 * np.log10(res.ci_upper[band]),
        color=COLOR_PRIMARY, alpha=0.28, lw=0.0,
        label="95 % chi-square confidence interval")
    ax.semilogx(freqs, 10.0 * np.log10(res.psd[band]), color=COLOR_PRIMARY,
                linewidth=1.0, alpha=0.85, label="Welch PSD estimate")
    ax.semilogx(freqs, 10.0 * np.log10(smooth), color=COLOR_SECONDARY,
                linewidth=2.2, label="1/3-octave smoothed")
    ax.semilogx(freqs, ref_db, color=COLOR_FG, linestyle="--", linewidth=1.4,
                alpha=0.7, label="Exact -3.01 dB/octave power law")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB re 1/Hz]")
    ax.set_title("Calibrated Spectral Density of Pink Noise (Bendat & Piersol)",
                 fontweight="bold", pad=12)
    ax.set_xlim(20.0, 20000.0)
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.985, 0.965,
            f"$n_d$ = {res.n_averages:.0f} averages, "
            f"$\\varepsilon_r$ = {100.0 * res.random_error:.1f} %",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "psd_confidence_smoothing.svg")


def generate_multitaper_psd_confidence(output_dir: str) -> None:
    """Thomson multitaper PSD of a short record vs a single-taper estimate."""
    print("Generating multitaper_psd_confidence...")
    from phonometry import multitaper_psd, noise_signal

    fs = 48000.0
    n = 8192  # a genuinely short record: 171 ms
    x = noise_signal(fs, n / fs, color="pink", seed=11)
    single = multitaper_psd(x, fs, n_tapers=1, adaptive=False)
    res = multitaper_psd(x, fs)
    band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    freqs = res.frequencies[band]
    # The exact -3.01 dB/oct power law through the mean level around 1 kHz.
    anchor = (freqs >= 800.0) & (freqs <= 1250.0)
    level_1k = float(np.mean(10.0 * np.log10(res.psd[band][anchor])))
    ref_db = level_1k - 10.0 * np.log10(freqs / 1000.0)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.semilogx(freqs, 10.0 * np.log10(single.psd[band]), color="gray",
                alpha=0.45, linewidth=0.7,
                label="Single Slepian taper ($K$ = 1, $\\nu$ = 2)")
    ax.fill_between(
        freqs,
        10.0 * np.log10(res.ci_lower[band]),
        10.0 * np.log10(res.ci_upper[band]),
        color=COLOR_PRIMARY, alpha=0.28, lw=0.0,
        label="95 % chi-square confidence interval")
    ax.semilogx(freqs, 10.0 * np.log10(res.psd[band]), color=COLOR_PRIMARY,
                linewidth=1.2, label="Multitaper estimate ($K$ = 7, adaptive)")
    ax.semilogx(freqs, ref_db, color=COLOR_FG, linestyle="--", linewidth=1.4,
                alpha=0.7, label="Exact -3.01 dB/octave power law")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB re 1/Hz]")
    ax.set_title(
        "Thomson Multitaper Density of a Short Record (Percival & Walden)",
        fontweight="bold", pad=12)
    ax.set_xlim(20.0, 20000.0)
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    nu = float(np.mean(res.degrees_of_freedom[1:-1]))
    ax.text(0.985, 0.965,
            f"171 ms record, $NW$ = 4, "
            f"$\\bar\\nu$ = {nu:.1f} equivalent dof",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "multitaper_psd_confidence.svg")
def generate_calibrated_spectrogram(output_dir: str) -> None:
    """Calibrated STFT spectrogram of a nonstationary signal, in dB SPL."""
    print("Generating calibrated_spectrogram...")
    from phonometry import noise_signal, spectrogram

    fs = 16000.0
    duration = 4.0
    t = np.arange(int(fs * duration)) / fs
    p_ref = 2e-5

    # A siren sweeping 600-1200 Hz at 70 dB SPL, an impact at t = 2.5 s
    # and a pink-noise floor at 45 dB SPL, all in pascals.
    siren_rms = p_ref * 10.0 ** (70.0 / 20.0)
    phase = 2.0 * np.pi * 900.0 * t - 600.0 * np.cos(np.pi * t)
    x = siren_rms * np.sqrt(2.0) * np.cos(phase)
    rng = np.random.default_rng(9)
    n_imp = int(0.06 * fs)
    impact = rng.standard_normal(n_imp) * np.exp(
        -np.arange(n_imp) / (0.012 * fs)
    )
    x[int(2.5 * fs):int(2.5 * fs) + n_imp] += 0.4 * impact
    x += noise_signal(fs, duration, color="pink",
                      rms=p_ref * 10.0 ** (45.0 / 20.0), seed=10)

    res = spectrogram(x, fs, nperseg=1024, overlap=0.75, scaling="spectrum")
    level = 10.0 * np.log10(res.power / p_ref**2)
    vmax = float(np.ceil(level.max()))

    fig, ax = plt.subplots(figsize=(10, 6.2))
    half_hop = 0.5 * res.hop / fs
    df = float(res.frequencies[1] - res.frequencies[0])
    img = ax.imshow(
        level, cmap="magma", vmin=vmax - 55.0, vmax=vmax, aspect="auto",
        origin="lower", interpolation="nearest",
        extent=(float(res.times[0]) - half_hop,
                float(res.times[-1]) + half_hop,
                0.0, float(res.frequencies[-1]) + 0.5 * df),
    )
    fig.colorbar(img, ax=ax, label="Sound pressure level [dB SPL]")
    ax.set_ylim(0.0, 3000.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(LABEL_FREQ_HZ)
    ax.set_title("Calibrated Spectrogram in dB SPL (Bendat & Piersol)",
                 fontweight="bold", pad=12)
    ax.text(0.02, 0.965,
            "a siren, an impact and a pink-noise floor:\n"
            "every cell reads an absolute level",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color="white")
    plt.tight_layout()
    save_figure(output_dir, "calibrated_spectrogram.png")
    plt.close()


def generate_zoom_fft_resolution(output_dir: str) -> None:
    """Zoom FFT resolving two tones closer than a coarse FFT bin."""
    print("Generating zoom_fft_resolution...")
    from scipy import signal as sp_signal

    from phonometry import zoom_fft

    fs = 8192.0
    t = np.arange(8192) / fs  # 1 s record: 1 Hz true resolution
    x = 0.8 * np.cos(2.0 * np.pi * 997.0 * t) + 0.5 * np.cos(
        2.0 * np.pi * 1000.0 * t
    )

    # Coarse view: a 1024-point FFT of the same record (8 Hz bins).
    w = sp_signal.get_window("hann", 1024)
    coarse = 2.0 * np.abs(np.fft.rfft(x[:1024] * w)) / np.sum(w)
    coarse_f = np.fft.rfftfreq(1024, 1.0 / fs)
    band = (coarse_f >= 950.0) & (coarse_f <= 1050.0)

    # 0.25 Hz grid: four points per record-length resolution, so the two
    # mainlobes are drawn as smooth curves with their exact peaks.
    res = zoom_fft(x, fs, 980.0, 1016.0, n_points=145)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coarse_f[band], 20.0 * np.log10(np.maximum(coarse[band], 1e-12)),
            color=COLOR_SECONDARY, marker="o",
            ms=4.0, lw=1.2, ls="--", label="1024-point FFT (8 Hz bins)")
    ax.plot(res.frequencies,
            20.0 * np.log10(np.maximum(res.amplitude, 1e-12)),
            color=COLOR_PRIMARY, lw=1.6,
            label="Zoom FFT of the same record")
    for f0 in (997.0, 1000.0):
        ax.axvline(f0, color=COLOR_FG, ls=":", lw=1.0, alpha=0.6)
    ax.set_xlim(950.0, 1050.0)
    ax.set_ylim(-70.0, 5.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title("Zoom FFT Resolves Tones One Coarse Bin Apart "
                 "(Bendat & Piersol)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.015, 0.965,
            "997 and 1000 Hz, 3 Hz apart:\n"
            "one lump on the 8 Hz grid,\n"
            "two exact lines on the zoom grid",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "zoom_fft_resolution.svg")


def generate_tone_burst_train(output_dir: str) -> None:
    """IEC 60268-1 tone bursts: one gated burst and the repetitive train."""
    print("Generating tone_burst_train...")
    from phonometry import tone_burst

    fs = 48000.0
    single = tone_burst(fs, 5000.0, 25, pre_silence=0.001, post_silence=0.001)
    train = tone_burst(fs, 5000.0, 25, repetitions=4, repetition_rate=10.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
    t_ms = 1e3 * np.arange(single.signal.size) / single.fs
    axes[0].plot(t_ms, single.signal, color=COLOR_PRIMARY, linewidth=0.9)
    axes[0].plot(t_ms, single.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--", label="Gating envelope")
    axes[0].plot(t_ms, -single.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--")
    axes[0].set_title("Single 5 ms burst of 5 kHz tone (25 full periods)",
                      fontweight="bold")
    axes[0].set_xlabel("Time [ms]")
    axes[0].legend(loc="upper right", fontsize=9)

    t_s = np.arange(train.signal.size) / train.fs
    axes[1].plot(t_s, train.signal, color=COLOR_PRIMARY, linewidth=0.5)
    axes[1].plot(t_s, train.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--", label="Gating envelope")
    axes[1].plot(t_s, -train.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--")
    axes[1].set_title("Repetitive train: 10 bursts per second (duty cycle 5 %)",
                      fontweight="bold")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Tone-Burst Test Signal (IEC 60268-1)", fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "tone_burst_train.svg")
    plt.close()


def generate_window_functions_tradeoff(output_dir: str) -> None:
    """Window spectra with their Harris figures of merit in the legend."""
    print("Generating window_functions_tradeoff...")
    from phonometry import window_metrics

    n, oversample = 1024, 256
    cases = [
        ("boxcar", COLOR_FG, "-"),
        ("hann", COLOR_PRIMARY, "-"),
        ("hamming", COLOR_SECONDARY, "-"),
        ("blackman", COLOR_TERTIARY, "-"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for name, color, style in cases:
        res = window_metrics(name, n)
        spectrum = np.abs(np.fft.rfft(res.taps, n=n * oversample))
        level = 20.0 * np.log10(spectrum / spectrum[0])
        bins = np.arange(level.size) / oversample
        shown = bins <= 16.0
        ax.plot(bins[shown], level[shown], color=color, linestyle=style,
                linewidth=1.4, alpha=0.9,
                label=(f"{name}: ENBW {res.enbw_bins:.2f} bins, "
                       f"sidelobe {res.highest_sidelobe_db:.1f} dB"))
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(-100.0, 5.0)
    ax.set_xlabel("Frequency offset [DFT bins]")
    ax.set_ylabel("Level re main lobe [dB]")
    ax.set_title("Window Functions: The Spectral Trade-off (Harris 1978)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "window_functions_tradeoff.svg")
    plt.close()


def generate_gcc_phat_delay(output_dir: str) -> None:
    """GCC-PHAT vs direct correlation for TDE on a colored signal pair."""
    print("Generating gcc_phat_delay...")
    from scipy import signal as sp_signal

    from phonometry import noise_signal, time_delay

    fs = 8192.0
    delay = 20  # samples -> 2.44 ms
    # Colored common signal: a Butterworth roll-off keeps some power in
    # every band (the Knapp & Carter condition for a usable PHAT phase).
    b, a = sp_signal.butter(2, 800.0 / (fs / 2.0))
    s = sp_signal.lfilter(b, a, noise_signal(fs, 4.0, color="white", seed=10))
    noise_x = noise_signal(fs, 4.0, color="white", rms=0.02, seed=11)
    noise_y = noise_signal(fs, 4.0, color="white", rms=0.02, seed=12)
    x = s + noise_x
    y = np.roll(s, delay) + noise_y

    direct = time_delay(x, y, fs, method="direct", max_delay=0.01)
    phat = time_delay(x, y, fs, method="gcc", weighting="phat",
                      nperseg=2048, max_delay=0.01)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(1e3 * direct.lags,
            direct.correlation / np.max(np.abs(direct.correlation)),
            color=COLOR_PRIMARY, linewidth=1.6,
            label="Direct cross-correlation")
    ax.plot(1e3 * phat.lags, phat.correlation, color=COLOR_SECONDARY,
            linewidth=1.6, label="GCC-PHAT")
    ax.axvline(1e3 * delay / fs, color=COLOR_FG, linestyle="--",
               linewidth=1.4, alpha=0.7, label="True delay (20 samples)")
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Normalized correlation")
    ax.set_title("Time-Delay Estimation: GCC-PHAT vs Direct Correlation "
                 "(Knapp & Carter)", fontweight="bold", pad=12)
    ax.set_xlim(-10.0, 10.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.985, 0.965,
            "colored signal: the plain correlator smears the peak,\n"
            "PHAT prewhitens the cross-spectrum and restores it",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "gcc_phat_delay.svg")
    plt.close()


def generate_cepstrum_echo(output_dir: str) -> None:
    """Echo detection on the power cepstrum of an IR with one reflection."""
    print("Generating cepstrum_echo...")
    from scipy import signal as sp_signal

    from phonometry import echo_detection, noise_signal

    fs = 48000.0
    delay_s = 0.008  # one floor reflection 8 ms after the direct sound
    a = 0.5
    # Broadband direct sound: a band-passed click, plus the scaled echo and
    # a -80 dB noise floor.
    n = 12000
    impulse = np.zeros(n)
    impulse[0] = 1.0
    b, bb = sp_signal.butter(2, [0.004, 0.9], btype="bandpass")
    direct = sp_signal.lfilter(b, bb, impulse)
    ir = direct + a * np.roll(direct, round(delay_s * fs))
    ir += noise_signal(fs, n / fs, color="white", rms=1e-4, seed=13)

    res = echo_detection(ir, fs, min_quefrency=0.002)

    _fig, ax = plt.subplots(figsize=(10, 6))
    half = res.nfft // 2 + 1
    ax.plot(1e3 * res.quefrencies[:half], res.cepstrum[:half],
            color=COLOR_PRIMARY, linewidth=1.1, label="Power cepstrum")
    ax.axvspan(1e3 * res.search_range[0], 1e3 * res.search_range[1],
               color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
               label="Searched band")
    ax.axvline(1e3 * delay_s, color=COLOR_FG, linestyle="--", linewidth=1.3,
               alpha=0.7, label="True echo delay (8 ms)")
    ax.plot([1e3 * res.delay], [res.reflection_coefficient], "v",
            color=COLOR_SECONDARY, markersize=10,
            label="Detected peak (height = reflection a)")
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(-0.3, 0.65)
    ax.set_xlabel("Quefrency [ms]")
    ax.set_ylabel("Cepstrum")
    ax.set_title("Echo Detection on the Power Cepstrum (Quefrency Analysis)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.985, 0.60,
            "spectral ripple of period 1/(8 ms) collapses to one\n"
            "spike at 8 ms whose height reads the reflection",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "cepstrum_echo.svg")
    plt.close()


def generate_envelope_spectrum(output_dir: str) -> None:
    """Envelope spectrum of an AM tone in noise: the line at fm."""
    print("Generating envelope_spectrum...")
    from phonometry import envelope_spectrum, noise_signal

    fs = 8192.0
    seconds = 4.0
    t = np.arange(int(seconds * fs)) / fs
    a0, m, fm = 1.0, 0.4, 25.0
    x = a0 * (1.0 + m * np.cos(2.0 * np.pi * fm * t)) * np.cos(
        2.0 * np.pi * 1000.0 * t
    )
    x += noise_signal(fs, seconds, color="white", rms=0.03, seed=8)

    res = envelope_spectrum(x, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.frequencies, res.amplitude, color=COLOR_PRIMARY,
            linewidth=1.4, label="Envelope spectrum")
    ax.axvline(fm, color=COLOR_FG, linestyle="--", linewidth=1.3, alpha=0.7,
               label="Modulation frequency (25 Hz)")
    ax.axhline(a0 * m, color=COLOR_SECONDARY, linestyle=":", linewidth=1.4,
               label=r"Exact line amplitude $A_0 m$ = 0.4")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Modulation amplitude")
    ax.set_title("Envelope Spectrum of an AM Tone (Bendat & Piersol 13.3)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.985, 0.70,
            "the carrier is at 1 kHz; its amplitude modulation\n"
            "appears as one line at exactly $f_m$",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "envelope_spectrum.svg")
    plt.close()


def generate_synchronous_average(output_dir: str) -> None:
    """TSA: a periodic waveform pulled from noise, and the comb filter."""
    print("Generating synchronous_average...")
    from phonometry import (
        comb_filter_response,
        noise_signal,
        time_synchronous_average,
    )

    fs = 8192.0
    period = 1.0 / 32.0  # one revolution: 256 samples at this rate
    m = 256
    n_avg = 40
    phase = np.arange((n_avg + 1) * m) / m
    periodic = (
        np.cos(2.0 * np.pi * phase)
        + 0.5 * np.cos(2.0 * np.pi * 3.0 * phase + 0.4)
        - 0.3 * np.cos(2.0 * np.pi * 6.0 * phase)
    )
    signal = periodic + noise_signal(fs, phase.size / fs, rms=0.9, seed=11)
    res = time_synchronous_average(signal, fs, period, n_averages=n_avg)
    true_one = periodic[:m]

    _fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel (a): one noisy period against the recovered average.
    t_ms = 1e3 * res.times
    ax0.plot(t_ms, signal[:m], color=COLOR_GRID, linewidth=1.0,
             label="One noisy period")
    ax0.plot(t_ms, res.period_waveform, color=COLOR_PRIMARY, linewidth=1.8,
             label=f"Average of N = {n_avg} periods")
    ax0.plot(t_ms, true_one, color=COLOR_SECONDARY, linestyle="--",
             linewidth=1.2, label="True periodic waveform")
    ax0.set_xlim(0.0, 1e3 * period)
    ax0.set_xlabel("Time [ms]")
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Periodic Waveform Extracted from Noise",
                  fontweight="bold", pad=10)
    ax0.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax0.set_axisbelow(True)
    ax0.legend(loc="upper right", fontsize=8)
    ax0.text(0.02, 0.03,
             "averaging N periods lowers the asynchronous\n"
             "noise by $\\sqrt{N}$ in amplitude",
             transform=ax0.transAxes, va="bottom", ha="left", fontsize=8.5,
             color=COLOR_FG)

    # Panel (b): comb filter, node selection at 32.05 orders.
    orders = np.linspace(31.0, 33.0, 4000)
    freqs = orders / period
    c20 = comb_filter_response(freqs, period, 20)
    c32 = comb_filter_response(freqs, period, 32)
    ax1.plot(orders, c32, color=COLOR_TERTIARY, linewidth=1.2,
             label="N = 32 (power of two)")
    ax1.plot(orders, c20, color=COLOR_PRIMARY, linewidth=1.4,
             label="N = 20 (node on 32.05)")
    ax1.axvline(32.05, color=COLOR_SECONDARY, linestyle=":", linewidth=1.3,
                label="Interfering tone (32.05)")
    ax1.set_xlim(31.0, 33.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("Frequency [orders]")
    ax1.set_ylabel("Comb filter magnitude")
    ax1.set_title("Rejecting a Tone by Choosing N (McFadden 1987)",
                  fontweight="bold", pad=10)
    ax1.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.text(0.02, 0.55,
             "N = 20 puts a node on 32.05 orders and removes\n"
             "it; the power-of-two N = 32 lets it through",
             transform=ax1.transAxes, va="top", ha="left", fontsize=8.5,
             color=COLOR_FG)

    plt.tight_layout()
    save_figure(output_dir, "synchronous_average.svg")
    plt.close()


def generate_miso_coherence(output_dir: str) -> None:
    """MISO coherence: which correlated source dominates each band."""
    print("Generating miso_coherence...")
    from scipy import signal as sp_signal

    from phonometry import miso_coherence, noise_signal

    fs = 8192.0
    seconds = 32.0
    # x1 drives a low-frequency path; x2 = 0.7*x1 + independent noise drives a
    # high-frequency path. x2 is correlated with x1, so its ORDINARY coherence
    # with the output is inflated in the low band (borrowed through x1), while
    # its PARTIAL coherence is clean once x1 is conditioned out.
    x1 = noise_signal(fs, seconds, color="white", seed=1)
    x2 = 0.7 * x1 + noise_signal(fs, seconds, color="white", seed=2)
    low = sp_signal.butter(4, 400.0, fs=fs, output="sos")
    high = sp_signal.butter(4, 1500.0, btype="high", fs=fs, output="sos")
    noise = noise_signal(fs, seconds, color="white", rms=0.05, seed=3)
    y = sp_signal.sosfilt(low, x1) + sp_signal.sosfilt(high, x2) + noise
    res = miso_coherence([x1, x2], y, fs, nperseg=2048)

    f = res.frequencies
    band = (f >= 20.0) & (f <= 4000.0)
    fb = f[band]

    _fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)

    def db(v: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(v)

    ax_top.semilogx(fb, db(res.output_psd[band]), color="gray", linewidth=1.4,
                    label="Measured output")
    floor = db(res.output_psd[band]).min() - 5.0
    for i, color in ((0, COLOR_PRIMARY), (1, COLOR_TERTIARY)):
        level = db(res.coherent_output_spectra[i][band])
        # Both contributions rise from the same floor, so they stay
        # translucent and only their opacity follows the page.
        ax_top.fill_between(fb, floor, level, color=color, lw=0.0,
                            alpha=theme_fill_alpha(color, ax_top))
        ax_top.semilogx(fb, level, color=color, linewidth=1.4,
                        label=f"Input {i + 1} contribution")
    ax_top.semilogx(fb, db(res.noise_psd[band]), color=COLOR_SECONDARY,
                    linestyle="--", linewidth=1.1, label="Residual noise")
    ax_top.set_ylim(floor, db(res.output_psd[band]).max() + 3.0)
    ax_top.set_ylabel("Coherent output [dB re 1/Hz]")
    ax_top.set_title(
        "Multiple-Input Coherence: Which Source Dominates Each Band "
        "(Bendat & Piersol Ch. 7)", fontweight="bold", pad=12)
    ax_top.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_top.set_axisbelow(True)
    ax_top.legend(loc="lower center", fontsize=8.5, ncol=2)

    ax_bot.semilogx(fb, res.ordinary_coherence[1][band], color=COLOR_TERTIARY,
                    linestyle=":", linewidth=1.6,
                    label=r"Input 2 ordinary $\gamma^2_{2y}$ (inflated by x1)")
    ax_bot.semilogx(fb, res.partial_coherence[1][band], color=COLOR_TERTIARY,
                    linewidth=1.8,
                    label=r"Input 2 partial $\gamma^2_{2y\cdot 1}$ (x1 removed)")
    ax_bot.semilogx(fb, res.multiple_coherence[band], color=COLOR_FG,
                    linewidth=1.4, alpha=0.8,
                    label=r"Multiple $\gamma^2_{y:x}$")
    ax_bot.set_ylim(0.0, 1.05)
    ax_bot.set_xlim(20.0, 4000.0)
    ax_bot.set_xlabel("Frequency [Hz]")
    ax_bot.set_ylabel("Coherence")
    ax_bot.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_bot.set_axisbelow(True)
    ax_bot.legend(loc="center right", fontsize=8.5)
    ax_bot.text(0.015, 0.05,
                "conditioning removes the shared x1 component:\n"
                "the low-band ordinary coherence of x2 collapses",
                transform=ax_bot.transAxes, va="bottom", ha="left",
                fontsize=8.5, color=COLOR_FG)
    for ax in (ax_top, ax_bot):
        format_frequency_axis(ax, 20.0, 4000.0)
    plt.tight_layout()
    save_figure(output_dir, "miso_coherence.svg")
    plt.close()


def generate_trend_test(output_dir: str) -> None:
    """Reverse arrangement trend test: B&P Example 4.4 vs a rising drift."""
    print("Generating trend_test...")
    from phonometry import trend_test

    # B&P Example 4.4: twenty observations, A = 86, accepted (no trend).
    example = np.array([
        5.2, 6.2, 3.7, 6.4, 3.9, 4.0, 3.9, 5.3, 4.0, 4.6,
        5.9, 6.5, 4.3, 5.7, 3.1, 5.6, 5.2, 3.9, 6.2, 5.0,
    ])
    # The same fluctuations with a slow upward drift: fewer reverse
    # arrangements (A = 38, below the Table A.6 lower bound of 64), rejected.
    drifting = example + np.linspace(0.0, 4.0, example.size)
    res_flat = trend_test(example)
    res_drift = trend_test(drifting)

    _fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(1, example.size + 1)
    ax.plot(index, res_flat.values, "o-", color=COLOR_PRIMARY,
            linewidth=1.2, markersize=5,
            label="B&P Example 4.4: A = 86, accepted (no trend)")
    ax.plot(index, res_drift.values, "s-", color=COLOR_SECONDARY,
            linewidth=1.2, markersize=5,
            label="Added rising drift: A = 38, rejected (trend)")
    ax.set_xticks(index[1::2])
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sequence value")
    ax.set_title("Nonparametric Trend Test by Reverse Arrangements (B&P 4.5.2)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.80,
            "20 observations; the count A of pairs i < j with x[i] > x[j]\n"
            "must fall in (64, 125] at the 5 % level (Table A.6). A rising\n"
            "trend depresses A below the acceptance region",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "trend_test.svg")
    plt.close()


def generate_stationarity_test(output_dir: str) -> None:
    """Reverse arrangement stationarity test: steady noise vs a gain ramp."""
    print("Generating stationarity_test...")
    from phonometry import stationarity_test

    fs = 8192.0
    n = 1 << 16
    steady = np.random.default_rng(42).standard_normal(n)
    # The B&P Example 10.3 scenario: the same noise with a slow +20 % gain
    # increase over the record.
    ramp = np.random.default_rng(42).standard_normal(n) * np.linspace(
        1.0, 1.2, n
    )
    res_steady = stationarity_test(steady, fs)
    res_ramp = stationarity_test(ramp, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(1, res_steady.n_segments + 1)
    ax.plot(index, res_steady.segment_values, "o-", color=COLOR_PRIMARY,
            linewidth=1.2, markersize=5,
            label="Steady noise: A = 91, accepted (stationary)")
    ax.plot(index, res_ramp.segment_values, "s-", color=COLOR_SECONDARY,
            linewidth=1.2, markersize=5,
            label="+20 % gain ramp: A = 7, rejected (nonstationary)")
    ax.set_xticks(index[1::2])
    ax.set_xlabel("Segment index")
    ax.set_ylabel("Segment mean square")
    ax.set_title("Stationarity Test by Reverse Arrangements (B&P 10.3.1.1)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.80,
            "20 segment mean squares; the count A of pairs i < j with\n"
            "x[i] > x[j] must fall in (64, 125] at the 5 % level (Table A.6)",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "stationarity_test.svg")
    plt.close()


def _bandlimited_gaussian_figure_record(
    seed: int, fs: float, n: int, f1: float, f2: float
) -> np.ndarray:
    """Exactly bandlimited unit-variance Gaussian noise (FFT synthesis)."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(
        freqs.size
    )
    spec[(freqs < f1) | (freqs > f2)] = 0.0
    x = np.fft.irfft(spec, n)
    return np.asarray(x / np.std(x))


def generate_rice_level_crossings(output_dir: str) -> None:
    """Measured level-crossing rates against the Rice curve."""
    print("Generating rice_level_crossings...")
    from phonometry import level_crossing_rate

    fs = 20480.0
    x = _bandlimited_gaussian_figure_record(0, fs, 1 << 19, 800.0, 1200.0)
    res = level_crossing_rate(x, fs, levels=np.linspace(-3.5, 3.5, 29))

    _fig, ax = plt.subplots(figsize=(10, 6))
    order = np.argsort(res.levels)
    ax.plot(res.levels[order], res.rice_rates[order], color=COLOR_SECONDARY,
            linewidth=1.6,
            label=r"Rice: $N_0\,\exp(-a^2/2\sigma_x^2)$ (Eq. 5.196)")
    ax.plot(res.levels, res.rates, "o", color=COLOR_PRIMARY, markersize=6,
            label="Measured crossing rate")
    ax.set_yscale("log")
    ax.set_xlabel("Level a [signal units]")
    ax.set_ylabel("Crossings per second [1/s]")
    ax.set_title("Level-Crossing Rates of Bandlimited Gaussian Noise (Rice)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9)
    ax.text(0.975, 0.965,
            "800-1200 Hz Gaussian band: 2014 zero crossings/s, an\n"
            r"apparent frequency $N_0/2 \approx$ 1007 Hz"
            " (B&P Example 5.13)",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "rice_level_crossings.svg")
    plt.close()


def generate_rice_peak_distribution(output_dir: str) -> None:
    """Peak-height exceedance between the Gaussian and Rayleigh limits."""
    print("Generating rice_peak_distribution...")
    from phonometry import peak_statistics
    from phonometry.metrology.data_qualification import _rice_peak_exceedance

    fs = 20480.0
    x = _bandlimited_gaussian_figure_record(3, fs, 1 << 19, 0.0, 2000.0)
    res = peak_statistics(x, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    peaks = res.peak_values
    exceedance = 1.0 - np.arange(1, peaks.size + 1) / peaks.size
    z = np.linspace(-2.5, 4.5, 400)
    ax.plot(z, _rice_peak_exceedance(z, 1.0), color=COLOR_FG, linewidth=1.0,
            linestyle="--", alpha=0.6, label="Rayleigh limit (r = 1, narrowband)")
    ax.plot(z, _rice_peak_exceedance(z, 0.0), color=COLOR_FG, linewidth=1.0,
            linestyle=":", alpha=0.6, label="Gaussian limit (r = 0, wideband)")
    ax.plot(z, res.peak_exceedance(z), color=COLOR_SECONDARY, linewidth=1.7,
            label="Rice mixture at r = 0.746 (Eq. 5.223)")
    ax.plot(peaks, exceedance, drawstyle="steps-post", color=COLOR_PRIMARY,
            linewidth=1.2, label="Empirical peak exceedance (0-2 kHz noise)")
    ax.set_yscale("log")
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel(r"Standardized peak height $z = a/\sigma_x$")
    ax.set_ylabel("Prob[peak > z]")
    ax.set_title("Peak-Height Distribution and the Irregularity Factor (Rice)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.975, 0.965,
            r"low-pass noise: $r = N_0/2M = \sqrt{5}/3$;"
            " negative maxima exist,\nso the peak law sits between"
            " Gaussian and Rayleigh (B&P 5.5.4)",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "rice_peak_distribution.svg")
    plt.close()


def generate_uncertainty(output_dir: str) -> None:
    """GUM uncertainty budget and Monte Carlo distribution (Guide 98-3 + S1)."""
    print("Generating uncertainty_budget.png...")
    import phonometry as ph

    # A-weighted level: reading plus calibration, instrument and positional
    # corrections (all zero-mean); the model is their sum.
    quantities = [
        ph.Quantity(74.0, 0.0, name="Reading"),
        ph.rectangular(0.0, 0.20, name="Calibration"),
        ph.rectangular(0.0, 0.30, name="Instrument"),
        ph.Quantity(0.0, 0.35, dof=9, name="Position (Type A)"),
    ]
    model = lambda a, b, c, d: a + b + c + d

    result = ph.combine_uncertainty(model, quantities)
    mc = ph.monte_carlo(model, quantities, trials=1_000_000, coverage=0.95, seed=1)
    k, big = result.expanded(0.95)

    _fig, (ax_b, ax_m) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: uncertainty budget (contributions). ---
    contrib = result.contributions
    names = list(result.names)
    pos = np.arange(len(names))
    ax_b.barh(pos, contrib, color=COLOR_PRIMARY, zorder=2)
    ax_b.axvline(result.combined_uncertainty, color=COLOR_SECONDARY, ls="--",
                 label=f"$u_c$ = {result.combined_uncertainty:.3f} dB")
    ax_b.set_yticks(pos)
    ax_b.set_yticklabels(names)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Contribution to combined uncertainty [dB]")
    ax_b.set_title("GUM uncertainty budget", fontweight="bold", pad=10)
    ax_b.grid(which="major", axis="x", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower right")

    # --- Right: Monte Carlo output vs the GUM Gaussian. ---
    rng = np.random.default_rng(1)
    samples = (74.0 + rng.uniform(-0.20, 0.20, 200000)
               + rng.uniform(-0.30, 0.30, 200000)
               + rng.normal(0.0, 0.35, 200000))
    ax_m.hist(samples, bins=120, density=True, color=COLOR_PRIMARY, alpha=0.35,
              label="Monte Carlo (Suppl 1)")
    grid = np.linspace(samples.min(), samples.max(), 400)
    gauss = (np.exp(-0.5 * ((grid - result.value) / result.combined_uncertainty) ** 2)
             / (result.combined_uncertainty * np.sqrt(2 * np.pi)))
    ax_m.plot(grid, gauss, color=COLOR_SECONDARY, lw=2, label="GUM Gaussian")
    ax_m.axvspan(mc.interval[0], mc.interval[1], zorder=0,
                 color=theme_fill(COLOR_PRIMARY, ax_m),
                 label="95 % coverage interval")
    ax_m.set_xlabel("A-weighted level [dB]")
    ax_m.set_ylabel("Probability density")
    ax_m.set_title(f"Y = {result.value:.2f} dB,  U = {big:.2f} dB (k = {k:.2f})",
                   fontweight="bold", pad=10)
    ax_m.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_m.set_axisbelow(True)
    ax_m.legend(loc="upper right")

    plt.tight_layout()
    save_figure(output_dir, "uncertainty_budget.png")
    plt.close()


# Every documentation figure, in the order the sequential generator has always
# produced them. This registry is the single source of truth for both the
# sequential path (``generate_all``) and the parallel runner (``--jobs``),
# and for the ``--figure`` name filter.


def generate_regularized_inversion(output_dir: str) -> None:
    """Kirkeby inversion of a loudspeaker-like band-pass response."""
    print("Generating regularized_inversion...")
    from scipy import signal as sp_signal

    from phonometry import regularized_inverse_filter

    fs = 48000.0
    b, bb = sp_signal.butter(2, [100.0, 8000.0], btype="bandpass", fs=fs)
    imp = np.zeros(2048)
    imp[0] = 1.0
    h = sp_signal.lfilter(b, bb, imp)

    res = regularized_inverse_filter(h, fs, f_range=(200.0, 4000.0))

    freqs = res.frequencies
    pos = freqs > 0.0
    tiny = np.finfo(np.float64).tiny
    h_mag = np.abs(res.response_spectrum)
    peak = float(np.max(h_mag))
    inv_mag = np.abs(res.spectrum)
    eq_mag = h_mag * inv_mag

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs[pos],
                20.0 * np.log10(np.maximum(h_mag[pos], tiny) / peak),
                color=COLOR_PRIMARY, linewidth=1.4,
                label="Measured response $|H|$")
    ax.semilogx(freqs[pos],
                20.0 * np.log10(np.maximum(inv_mag[pos] * peak, tiny)),
                color=COLOR_SECONDARY, linewidth=1.4,
                label=r"Inverse filter $|H_{\mathrm{inv}}|$")
    ax.semilogx(freqs[pos],
                20.0 * np.log10(np.maximum(eq_mag[pos], tiny)),
                color=COLOR_TERTIARY, linewidth=1.8,
                label=r"Equalized $|H \cdot H_{\mathrm{inv}}|$")
    ax.axvspan(200.0, 4000.0, color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
               label="Equalized band (200 Hz - 4 kHz)")
    ax.set_xlim(20.0, fs / 2.0)
    ax.set_ylim(-50.0, 15.0)
    format_frequency_axis(ax, 20.0, fs / 2.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Regularized Spectral Inversion (Kirkeby Frequency-"
                 "Dependent Regularization)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9)
    ax.text(0.985, 0.97,
            "unity in-band; outside, the frequency-dependent\n"
            "regularization caps the gain instead of boosting noise",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "regularized_inversion.svg")
    plt.close()


def generate_shaped_sweep(output_dir: str) -> None:
    """A pink shaped sweep: waveform and spectrum against the target."""
    print("Generating shaped_sweep...")
    from scipy import signal as sp_signal

    from phonometry import shaped_sweep_signal

    fs = 48000
    res = shaped_sweep_signal(fs, 50.0, 5000.0, 2.0, target="pink")
    x = np.asarray(res)
    t = np.arange(x.size) / fs

    nperseg = 8192
    freqs_w, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg,
                                   noverlap=3 * nperseg // 4)
    tiny = np.finfo(np.float64).tiny
    band_w = (freqs_w >= 50.0) & (freqs_w <= 5000.0)
    welch_db = 10.0 * np.log10(np.maximum(psd, tiny))
    welch_db -= float(np.max(welch_db[band_w]))
    grid = res.frequencies
    band_g = (grid >= 50.0) & (grid <= 5000.0)
    target_db = 20.0 * np.log10(np.maximum(res.magnitude, tiny))
    target_db -= float(np.max(target_db[band_g]))

    _fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(t, x, color=COLOR_PRIMARY, linewidth=0.5)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0.0, float(t[-1]))
    axes[0].set_title("Shaped Sweep with an Arbitrary Target Spectrum "
                      "(Group-Delay Synthesis)", fontweight="bold", pad=12)
    axes[0].grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    axes[0].set_axisbelow(True)
    axes[0].text(0.985, 0.95,
                 "nearly constant envelope: the energy shaping lives\n"
                 "in the dwell time, not in the amplitude",
                 transform=axes[0].transAxes, va="top", ha="right",
                 fontsize=8.5, color=COLOR_FG)

    posw = freqs_w > 0.0
    axes[1].semilogx(freqs_w[posw], welch_db[posw], color=COLOR_PRIMARY,
                     linewidth=1.3, label="Welch spectrum of the sweep")
    posg = grid > 0.0
    axes[1].semilogx(grid[posg], target_db[posg], color=COLOR_SECONDARY,
                     linewidth=1.5, linestyle="--",
                     label="Pink target (-3 dB per octave)")
    axes[1].axvspan(50.0, 5000.0, color=theme_fill(COLOR_PRIMARY, axes[1]),
                    zorder=0, label="Sweep band (50 Hz - 5 kHz)")
    axes[1].set_xlim(20.0, 20000.0)
    axes[1].set_ylim(-60.0, 8.0)
    format_frequency_axis(axes[1], 20.0, 20000.0)
    axes[1].set_xlabel(LABEL_FREQ_HZ)
    axes[1].set_ylabel("Level re in-band max [dB]")
    axes[1].grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "shaped_sweep.svg")
    plt.close()
def generate_resampling_antialias(output_dir: str) -> None:
    """Polyphase resampling: the delivered anti-alias filter vs its spec."""
    print("Generating resampling_antialias...")
    from phonometry import noise_signal, resample_signal

    x = noise_signal(44100, 5.0, color="pink", seed=1)
    res = resample_signal(x, 44100, 48000)      # 120 dB alias rejection

    fs_up = res.original_fs * res.up
    freqs, h = scipy_signal.freqz(res.filter_taps, worN=1 << 18, fs=fs_up)
    tiny = np.finfo(np.float64).tiny
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), tiny))
    f_lo, f_hi = res.stopband_edge_hz / 8.0, 4.0 * res.stopband_edge_hz
    view = (freqs > 0.0) & (freqs <= f_hi)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs[view], mag_db[view], color=COLOR_PRIMARY, linewidth=1.2,
                label="Anti-alias filter $|H(f)|$")
    ax.axvline(res.passband_edge_hz, color=COLOR_TERTIARY, linestyle="--",
               linewidth=1.4, label="Passband edge")
    ax.axvline(res.stopband_edge_hz, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.4, label="Stopband edge (alias fold)")
    ax.axhline(-res.stopband_attenuation_db, color=COLOR_FG, linestyle=":",
               linewidth=1.2, alpha=0.7, label="Design attenuation -120 dB")
    ax.axvspan(res.stopband_edge_hz, f_hi, color=theme_fill(COLOR_SECONDARY, ax),
               zorder=0,
               label="Rejected band (would fold back as aliases)")
    ax.set_xlim(f_lo, f_hi)
    ax.set_ylim(-170.0, 10.0)
    format_frequency_axis(ax, f_lo, f_hi)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Polyphase Resampling 44.1 kHz → 48 kHz: "
                 "the Delivered Anti-Alias Filter", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "resampling_antialias.svg")
    plt.close()


def generate_cepstrum_variants(output_dir: str) -> None:
    """Power, real and complex cepstra of one echo-carrying record."""
    print("Generating cepstrum_variants...")
    from phonometry import cepstrum

    fs = 48000.0
    # A band-limited source wavelet (its cepstrum concentrates below 1 ms)
    # plus one reflection at 8 ms with reflection coefficient a = 0.5.
    b, a_coef = scipy_signal.butter(2, 0.3)
    s = np.zeros(4096)
    s[37: 37 + 256] = scipy_signal.lfilter(
        b, a_coef, np.r_[1.0, np.zeros(255)]
    )
    x = s + 0.5 * np.roll(s, 384)                # echo: 8 ms, a = 0.5

    _fig, ax = plt.subplots(figsize=(10, 6))
    axins = ax.inset_axes((0.60, 0.28, 0.26, 0.42))
    variants = (
        ("power", COLOR_PRIMARY, "-", "Power cepstrum"),
        ("real", COLOR_TERTIARY, "--", "Real cepstrum (exactly half the power)"),
        ("complex", COLOR_SECONDARY, ":", "Complex cepstrum"),
    )
    for kind, colour, style, label in variants:
        res = cepstrum(x, fs, kind=kind)
        q_ms = 1e3 * res.quefrencies
        mask = (q_ms > 0.5) & (q_ms <= 20.0)
        ax.plot(q_ms[mask], res.cepstrum[mask], color=colour, linestyle=style,
                linewidth=1.1, label=label)
        zoom = (q_ms > 7.8) & (q_ms <= 8.2)
        axins.plot(q_ms[zoom], res.cepstrum[zoom], color=colour,
                   linestyle=style, linewidth=1.3, marker="o", markersize=3)
    axins.set_xlim(7.85, 8.15)
    axins.set_ylim(-0.05, 0.55)
    axins.tick_params(labelsize=7)
    axins.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.indicate_inset_zoom(axins, edgecolor=COLOR_FG, alpha=0.5)
    ax.annotate("first rahmonic at 8 ms:\nheight ≈ a on the power cepstrum",
                xy=(8.0, 0.5), xytext=(2.5, 0.42), fontsize=9,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax.annotate("second rahmonic: $-a^2/2$",
                xy=(16.0, -0.125), xytext=(12.0, -0.22), fontsize=9,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax.set_xlabel("Quefrency [ms]")
    ax.set_ylabel("Cepstrum")
    ax.set_xlim(0.5, 20.0)
    ax.set_ylim(-0.3, 0.6)
    ax.set_title("The Three Cepstrum Variants of One Echo-Carrying Record",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "cepstrum_variants.svg")
    plt.close()


def generate_lifter_split(output_dir: str) -> None:
    """Lowpass/highpass liftering of a log spectrum with an 8 ms echo."""
    print("Generating lifter_split...")
    from phonometry import lifter

    fs = 48000.0
    # The same wavelet-plus-echo record as the cepstrum-variants figure:
    # a smooth source envelope carrying a pure 8 ms echo ripple (a = 0.5),
    # so the highpass ripple swings exactly between the closed forms.
    b, a_coef = scipy_signal.butter(2, 0.3)
    s = np.zeros(4096)
    s[37: 37 + 256] = scipy_signal.lfilter(
        b, a_coef, np.r_[1.0, np.zeros(255)]
    )
    x = s + 0.5 * np.roll(s, 384)                # the same 8 ms echo

    low = lifter(x, fs, cutoff=0.004, mode="lowpass")
    high = lifter(x, fs, cutoff=0.004, mode="highpass")
    band = (low.frequencies >= 500.0) & (low.frequencies <= 2000.0)

    _fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogx(low.frequencies[band], low.spectrum_db[band],
                     color="#7f7f7f", linewidth=0.7, alpha=0.8,
                     label="Log spectrum of the record")
    axes[0].semilogx(low.frequencies[band], low.liftered_db[band],
                     color=COLOR_PRIMARY, linewidth=2.0,
                     label="Lowpass lifter: spectral envelope")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].semilogx(high.frequencies[band], high.liftered_db[band],
                     color=COLOR_SECONDARY, linewidth=1.1,
                     label="Highpass lifter: the echo ripple alone")
    for bound in (20.0 * np.log10(1.5), 20.0 * np.log10(0.5)):
        axes[1].axhline(bound, color=COLOR_TERTIARY, linestyle="--",
                        linewidth=1.2,
                        label=(r"closed-form ripple bounds $20\log_{10}(1\pm a)$"
                               if bound > 0 else None))
    axes[1].set_ylabel("Magnitude [dB]")
    axes[1].set_xlabel(LABEL_FREQ_HZ)
    axes[1].legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
        ax.set_axisbelow(True)
        format_frequency_axis(ax, 500.0, 2000.0)
    axes[0].set_title("Liftering at 4 ms: Envelope Versus Echo Ripple",
                      fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "lifter_split.svg")
    plt.close()


def generate_correlation_normalizations(output_dir: str) -> None:
    """The three correlation normalizations of a two-sensor delay model."""
    print("Generating correlation_normalizations...")
    from phonometry import correlation, noise_signal

    fs = 8192.0
    delay = 102                                   # 12.45 ms
    x = noise_signal(fs, 2.0, seed=4)
    interference = noise_signal(fs, 2.0, rms=0.5, seed=5)
    y = 0.8 * np.concatenate([np.zeros(delay), x[:-delay]]) + interference

    coeff = correlation(x, y, fs, normalization="coefficient", max_lag=0.05)
    biased = correlation(x, y, fs, normalization="biased")
    unbiased = correlation(x, y, fs, normalization="unbiased")

    _fig, (ax_c, ax_n) = plt.subplots(2, 1, figsize=(10, 7))
    ax_c.plot(1e3 * coeff.lags, coeff.values, color=COLOR_PRIMARY,
              linewidth=1.1,
              label=r"Coefficient $\rho_{xy}(\tau)$ (bounded by $\pm 1$)")
    ax_c.axvline(1e3 * delay / fs, color=COLOR_SECONDARY, linestyle="--",
                 linewidth=1.2, label="true delay +12.5 ms")
    ax_c.set_xlabel("Lag [ms]")
    ax_c.set_ylabel("Correlation")
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.legend(loc="upper left", fontsize=9)

    ax_n.plot(unbiased.lags, unbiased.values, color=COLOR_SECONDARY,
              linewidth=0.5, alpha=0.9,
              label=r"Unbiased $1/(N-|r|)$ (variance grows at the ends)")
    ax_n.plot(biased.lags, biased.values, color=COLOR_PRIMARY,
              linewidth=0.5,
              label=r"Biased $1/N$ (tapers toward the ends)")
    ax_n.set_xlabel("Lag [s]")
    ax_n.set_ylabel("Correlation")
    ax_n.legend(loc="upper left", fontsize=9)

    for ax in (ax_c, ax_n):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    ax_c.set_title("Correlation Normalizations of a Two-Sensor Delay Model",
                   fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "correlation_normalizations.svg")
    plt.close()


def generate_ir_alignment(output_dir: str) -> None:
    """Sub-sample alignment of an impulse response onto a reference."""
    print("Generating ir_alignment...")
    from phonometry import align_impulse_responses, fractional_delay

    fs = 48000.0
    t = np.arange(int(0.03 * fs)) / fs
    rng = np.random.default_rng(6)
    # Band-limited reference pulse: a 2 kHz Gaussian tone burst at 5 ms.
    ir_a = scipy_signal.gausspulse(t - 0.005, fc=2000.0, bw=0.5)
    ir_b = fractional_delay(ir_a, 7.37)[: ir_a.size]
    ir_b += 0.005 * rng.standard_normal(ir_a.size)

    res = align_impulse_responses(ir_b, ir_a, fs)
    t_ms = 1e3 * t

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_ms, res.reference, color=COLOR_PRIMARY, linewidth=1.6,
            label="Reference IR")
    ax.plot(t_ms, ir_b, color="#7f7f7f", linewidth=1.0, linestyle="--",
            alpha=0.8, label="Measured IR (delayed 7.37 samples)")
    ax.plot(t_ms, res.aligned[: t.size], color=COLOR_SECONDARY,
            linewidth=1.0, linestyle=":", label="Aligned IR (delay removed)")
    ax.text(0.985, 0.05,
            f"estimated delay removed: {res.delay_samples:.2f} samples",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            color=COLOR_FG)
    ax.set_xlim(2.0, 9.0)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sub-Sample Impulse-Response Alignment",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ir_alignment.svg")
    plt.close()


def generate_hilbert_envelope(output_dir: str) -> None:
    """Hilbert envelope and instantaneous frequency of a decaying mode."""
    print("Generating hilbert_envelope...")
    from phonometry import envelope

    fs = 8192.0
    t = np.arange(int(0.4 * fs)) / fs
    rng = np.random.default_rng(7)
    decay = np.exp(-t / 0.1)                     # a struck 250 Hz mode
    x = decay * np.sin(2.0 * np.pi * 250.0 * t)
    x += 0.001 * rng.standard_normal(t.size)

    # The envelope of a narrowband signal is low-frequency: the anti-aliased
    # x8 decimation keeps the outputs compact without losing the decay.
    res = envelope(x, fs, decimation_factor=8)

    _fig, (ax_e, ax_f) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_e.plot(t, res.signal, color=COLOR_PRIMARY, linewidth=0.6,
              label="Signal")
    ax_e.plot(res.times, res.envelope, color=COLOR_SECONDARY, linewidth=1.8,
              label="Envelope $A(t)$")
    ax_e.plot(res.times, -res.envelope, color=COLOR_SECONDARY, linewidth=1.8)
    ax_e.set_ylabel("Amplitude")
    ax_e.legend(loc="upper right", fontsize=9)

    ax_f.plot(res.times, res.instantaneous_frequency, color=COLOR_PRIMARY,
              linewidth=0.9, label="Instantaneous frequency $f(t)$")
    ax_f.axhline(250.0, color=COLOR_TERTIARY, linestyle="--", linewidth=1.4,
                 label="carrier 250 Hz")
    ax_f.set_ylim(230.0, 270.0)
    ax_f.set_ylabel("Instantaneous frequency [Hz]")
    ax_f.set_xlabel("Time [s]")
    ax_f.legend(loc="upper right", fontsize=9)

    for ax in (ax_e, ax_f):
        ax.set_xlim(0.0, 0.3)
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    ax_e.set_title("Hilbert Envelope and Instantaneous Frequency",
                   fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "hilbert_envelope.svg")
    plt.close()


def generate_cross_spectral_density_delay(output_dir: str) -> None:
    """Cross-spectral density of a delay path: magnitude and linear phase."""
    print("Generating cross_spectral_density...")
    from phonometry import cross_spectral_density, noise_signal

    fs = 8000.0
    tau = 0.002                                   # 2 ms = 16 samples
    delay = int(tau * fs)
    x = noise_signal(fs, 8.0, seed=8)
    noise = noise_signal(fs, 8.0, rms=0.3, seed=9)
    y = 0.9 * np.concatenate([np.zeros(delay), x[:-delay]]) + noise

    res = cross_spectral_density(x, y, fs)
    tiny = np.finfo(np.float64).tiny
    band = (res.frequencies >= 20.0) & (res.frequencies <= 3500.0)
    freqs = res.frequencies[band]

    _fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_m.semilogx(freqs, 10.0 * np.log10(np.maximum(res.magnitude[band], tiny)),
                  color=COLOR_PRIMARY, linewidth=1.1,
                  label=r"$|\hat{G}_{xy}|$ (Welch estimate)")
    ax_m.set_ylabel("Magnitude [dB]")
    ax_m.legend(loc="lower left", fontsize=9)

    ax_p.semilogx(freqs, res.phase[band], color=COLOR_PRIMARY, linewidth=1.1,
                  label="Unwrapped phase")
    ax_p.fill_between(freqs, res.phase[band] - res.phase_std[band],
                      res.phase[band] + res.phase_std[band],
                      color=COLOR_PRIMARY, alpha=0.25,
                      label=r"$\pm 1$ s.d. band (Eq. 9.52)")
    ax_p.semilogx(freqs, -2.0 * np.pi * freqs * tau, color=COLOR_SECONDARY,
                  linestyle="--", linewidth=1.4,
                  label=r"slope $-2\pi f\tau$ ($\tau$ = 2 ms)")
    ax_p.set_ylabel("Phase [rad]")
    ax_p.set_xlabel(LABEL_FREQ_HZ)
    ax_p.legend(loc="lower left", fontsize=9)

    for ax in (ax_m, ax_p):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
        ax.set_axisbelow(True)
        format_frequency_axis(ax, 20.0, 3500.0)
    ax_m.set_title("Cross-Spectral Density of a 2 ms Delay Path",
                   fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "cross_spectral_density.svg")
    plt.close()


def generate_coherent_output_snr(output_dir: str) -> None:
    """Coherent output spectrum split and the spectral SNR."""
    print("Generating coherent_output_snr...")
    from phonometry import coherent_output_spectrum, noise_signal

    fs = 48000.0
    x = noise_signal(fs, 8.0, color="white", seed=1)
    noise = noise_signal(fs, 8.0, color="white", rms=0.5, seed=2)
    y = 0.8 * x + noise                      # SNR = 0.64/0.25 per band

    res = coherent_output_spectrum(x, y, fs, nperseg=2048)
    tiny = np.finfo(np.float64).tiny
    band = np.flatnonzero(
        (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    )
    # Thin the flat spectra to ~1000 log-spaced bins: on a log axis the
    # linear Welch grid bunches thousands of vertices at the right edge
    # without adding visual information (and bloats the SVG).
    band = band[np.unique(np.geomspace(1, band.size, 1000).astype(int) - 1)]
    freqs = res.frequencies[band]

    _fig, (ax_g, ax_s) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for values, colour, style, label in (
        (res.output_psd, COLOR_PRIMARY, "-",
         r"$\hat{G}_{yy}$ (measured output)"),
        (res.coherent_psd, COLOR_TERTIARY, "--",
         r"$\hat{G}_{vv} = \gamma^2\hat{G}_{yy}$ (coherent part)"),
        (res.noise_psd, COLOR_SECONDARY, ":",
         r"$\hat{G}_{nn}$ (uncorrelated noise)"),
    ):
        ax_g.semilogx(freqs,
                      10.0 * np.log10(np.maximum(values[band], tiny)),
                      color=colour, linestyle=style, linewidth=1.1,
                      label=label)
    ax_g.set_ylabel("Spectral density [dB re 1/Hz]")
    ax_g.legend(loc="lower left", fontsize=9)

    ax_s.semilogx(freqs, res.snr_db[band], color=COLOR_PRIMARY,
                  linewidth=1.1, label="Spectral SNR [dB]")
    ax_s.axhline(10.0 * np.log10(0.64 / 0.25), color=COLOR_SECONDARY,
                 linestyle="--", linewidth=1.4,
                 label=r"closed form $10\log_{10}(2.56)$ = 4.1 dB")
    ax_s.set_ylabel("Spectral SNR [dB]")
    ax_s.set_xlabel(LABEL_FREQ_HZ)
    ax_s.legend(loc="lower left", fontsize=9)

    for ax in (ax_g, ax_s):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
        ax.set_axisbelow(True)
        format_frequency_axis(ax, 20.0, 20000.0)
    ax_g.set_title(
        "Coherent Output Spectrum and Spectral SNR (Bendat & Piersol 9.2.2)",
        fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "coherent_output_snr.svg")
    plt.close()


def generate_golay_ir(output_dir: str) -> None:
    """Golay-pair impulse response: exact complementary recovery."""
    print("Generating golay_ir...")
    from phonometry import golay_impulse_response, golay_pair

    fs = 48000
    pair = golay_pair(14)                        # two 16384-sample codes
    b, a = scipy_signal.butter(2, [200.0, 2000.0], btype="bandpass", fs=fs)
    length = pair[0].size
    rec_a = scipy_signal.lfilter(b, a, np.tile(pair[0], 3))[2 * length:]
    rec_b = scipy_signal.lfilter(b, a, np.tile(pair[1], 3))[2 * length:]

    ir = np.asarray(golay_impulse_response(rec_a, rec_b, pair, fs=fs))
    impulse = np.zeros(length)
    impulse[0] = 1.0
    true_ir = scipy_signal.lfilter(b, a, impulse)
    err = float(np.max(np.abs(ir - true_ir)))

    t_ms = 1e3 * np.arange(length) / fs
    view = t_ms <= 6.0
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_ms[view], ir[view], color=COLOR_PRIMARY, linewidth=1.6,
            label="Recovered IR (golay_impulse_response)")
    ax.plot(t_ms[view], true_ir[view], color=COLOR_SECONDARY, linewidth=1.2,
            linestyle="--", label="True system response")
    ax.text(0.985, 0.05,
            f"max |recovered - true| = {err:.1e}\n"
            "noise-free closed-form identity",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Golay-Pair Impulse Response: Exact Complementary Recovery",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "golay_ir.svg")
    plt.close()


def generate_tsa_noise_reduction(output_dir: str) -> None:
    """TSA: measured error of the average against the ideal 1/sqrt(N)."""
    print("Generating tsa_noise_reduction...")
    from phonometry import time_synchronous_average

    fs = 8192.0
    samples = 256
    period = samples / fs
    m = np.arange(samples) / fs
    true = (np.cos(2.0 * np.pi * m / period)
            + 0.5 * np.cos(2.0 * np.pi * 3.0 * m / period + 0.7)
            + 0.25 * np.cos(2.0 * np.pi * 5.0 * m / period + 1.1))
    rng = np.random.default_rng(5)
    n_max = 128
    x = np.tile(true, n_max) + rng.standard_normal(n_max * samples)

    counts = [1, 2, 4, 8, 16, 32, 64, 128]
    errors = []
    for n in counts:
        res = time_synchronous_average(x[: n * samples], fs, period,
                                       n_averages=n)
        errors.append(float(np.sqrt(np.mean(
            (res.period_waveform - true) ** 2))))

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(counts, errors, "o-", color=COLOR_PRIMARY, linewidth=1.6,
              markersize=6, label="Measured RMS error of the average")
    ax.loglog(counts, 1.0 / np.sqrt(np.asarray(counts, dtype=np.float64)),
              color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
              label=r"Ideal $\sigma/\sqrt{N}$")
    ax.set_xticks(counts)
    ax.set_xticklabels([str(n) for n in counts])
    ax.set_xlabel("Number of averages N")
    ax.set_ylabel("RMS error of the averaged waveform")
    ax.set_title(r"TSA Noise Reduction: the $\sqrt{N}$ Law (McFadden 1987)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "tsa_noise_reduction.svg")
    plt.close()


def generate_runs_test(output_dir: str) -> None:
    """Runs test about the median: accepted noise vs rejected alternation."""
    print("Generating runs_test...")
    from phonometry import trend_test

    rng = np.random.default_rng(3)
    sequences = [rng.standard_normal(40), np.tile([1.0, -1.0], 10)]

    _fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, seq in zip(axes, sequences):
        res = trend_test(seq, method="runs")
        idx = np.arange(1, seq.size + 1)
        median = float(np.median(seq))
        above = seq > median
        ax.plot(idx, seq, color="#7f7f7f", linewidth=0.7, alpha=0.7,
                zorder=1)
        ax.scatter(idx[above], seq[above], color=COLOR_PRIMARY, s=28,
                   zorder=3, label="Above the median")
        ax.scatter(idx[~above], seq[~above], color=COLOR_SECONDARY, s=28,
                   zorder=3, label="Below the median")
        ax.axhline(median, color=COLOR_FG, linestyle="--", linewidth=1.1,
                   alpha=0.7, label="Sequence median")
        verdict = "trend-free" if res.trend_free else "rejected"
        ax.set_title(f"r = {res.statistic} runs, accept "
                     f"({res.bounds[0]}, {res.bounds[1]}]: {verdict}",
                     fontweight="bold", pad=10)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sequence value")
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=8.5)
    plt.suptitle("Runs Test About the Median (Wald & Wolfowitz)",
                 fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "runs_test.svg")
    plt.close()
