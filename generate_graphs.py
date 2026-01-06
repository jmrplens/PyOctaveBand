import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

# Add src to path to use the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from pyoctaveband import OctaveFilterBank

# Constants for professional styling
LABEL_FREQ_HZ = "Frequency [Hz]"
LABEL_LEVEL_DB = "Level [dB]"
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#d62728"
COLOR_TERTIARY = "#2ca02c"
COLOR_GRID = "#e0e0e0"

# Global matplotlib configuration
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def apply_axis_styling(ax, title, xlim=None, ylim=None):
    """Apply consistent styling to plots."""
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(LABEL_LEVEL_DB)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-")
    ax.grid(which="minor", color=COLOR_GRID, linestyle=":", alpha=0.4)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Standard Octave Ticks
    xticks = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    xticklabels = ["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


def generate_filter_type_comparison(output_dir: str) -> None:
    """Compare different filter architectures with a zoom inset."""
    print("Generating filter_type_comparison.png...")
    fs = 48000
    fraction = 1
    order = 6
    
    # We want exactly the 1000Hz band
    limits = [800, 1200]
    
    filters = [
        ("butter", "Butterworth", COLOR_PRIMARY, "-"),
        ("cheby1", "Chebyshev I (0.1dB ripple)", COLOR_SECONDARY, "--"),
        ("ellip", "Elliptic (0.1dB ripple, 60dB atten)", COLOR_TERTIARY, "-."),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
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
        axins.plot(w, mag_db, color=color, linestyle=style) # plot is fine if scale is set

    ax.axhline(-3, color="black", linestyle=":", alpha=0.3, label="-3 dB")
    axins.axhline(-3, color="black", linestyle=":", alpha=0.3)
    
    apply_axis_styling(ax, "Filter Architecture Comparison (Order 6, 1kHz Band)", xlim=(100, 8000), ylim=(-80, 5))
    
    # Sub-plot styling (Zoom around 1kHz and -3dB)
    axins.set_xlim(650, 1500)
    axins.set_ylim(-6, 0.5) # As requested: from -6 to 0.5
    axins.grid(True, which="both", alpha=0.3)
    axins.set_title("Zoom at -3 dB (Log Scale)", fontsize=9)
    
    # Fix x-ticks for log scale zoom to look right
    from matplotlib.ticker import ScalarFormatter
    axins.xaxis.set_major_formatter(ScalarFormatter())
    axins.set_xticks([707, 1000, 1414])
    axins.set_xticklabels(["707", "1k", "1.4k"], fontsize=8)
    
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, "filter_type_comparison.png"))
    plt.close()


def generate_filter_responses(output_dir: str) -> None:
    """Generate plots for the filter bank responses for different filter types."""
    fs = 48000
    
    # Filter types to generate
    filter_types = [
        ("butter", "butter"),
        ("cheby1", "cheby1"),
        ("ellip", "ellip"),
    ]

    configs = [
        (1, 6),
        (3, 6),
    ]

    for f_type_name, f_type in filter_types:
        for fraction, order in configs:
            filename = f"filter_{f_type_name}_fraction_{fraction}_order_{order}.png"
            print(f"Generating {filename}...")
            bank = OctaveFilterBank(fs=fs, fraction=fraction, order=order, limits=[12, 20000], filter_type=f_type)
            
            from pyoctaveband.filter_design import _showfilter
            _showfilter(bank.sos, bank.freq, bank.freq_u, bank.freq_d, fs, bank.factor, 
                       show=False, plot_file=os.path.join(output_dir, filename))
            plt.close("all")


def generate_signal_responses(output_dir: str) -> None:
    """Generate spectral analysis plots for a complex signal."""
    fs = 48000
    duration = 5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    freqs = [20, 100, 500, 2000, 4000, 15000]
    y = 100 * np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

    for frac, filename, title in [
        (1, "signal_response_fraction_1.png", "1/1 Octave Band Analysis"),
        (3, "signal_response_fraction_3.png", "1/3 Octave Band Analysis"),
    ]:
        print(f"Generating {filename}...")
        bank = OctaveFilterBank(fs=fs, fraction=frac, order=6, limits=[12, 20000])
        spl, freq = bank.filter(y)

        _, ax = plt.subplots()
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
        )
        apply_axis_styling(ax, title, xlim=(11, 25000))
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def generate_multichannel_response(output_dir: str) -> None:
    """Generate analysis plot for a stereo signal."""
    print("Generating signal_response_multichannel.png...")
    fs = 48000
    duration = 5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    rng = np.random.default_rng(42)
    # Channel 1: Pink Noise (Voss-McCartney)
    num_cols = 16
    array = np.empty((len(t), num_cols))
    array.fill(np.nan)
    array[0, :] = rng.random(num_cols)
    array[:, 0] = rng.random(len(t))
    for i in range(1, len(t)):
        for j in range(1, num_cols):
            if i % (2**j) == 0:
                array[i, j] = rng.random()
            else:
                array[i, j] = array[i - 1, j]
    ch1 = np.sum(array, axis=1)
    ch1 = (ch1 - np.mean(ch1)) / np.max(np.abs(ch1))

    # Channel 2: Logarithmic Sine Sweep
    ch2 = scipy_signal.chirp(t, f0=50, t1=duration, f1=10000, method="logarithmic")

    x = np.vstack((ch1, ch2))
    bank = OctaveFilterBank(fs=fs, fraction=3, order=6, limits=[20, 20000])
    spl, freq = bank.filter(x)

    _, ax = plt.subplots()
    ax.semilogx(
        freq,
        spl[0],
        marker="o",
        markersize=5,
        label="Left: Pink Noise",
        color=COLOR_PRIMARY,
        alpha=0.8,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    ax.semilogx(
        freq,
        spl[1],
        marker="s",
        markersize=5,
        label="Right: Log Sweep",
        color=COLOR_SECONDARY,
        alpha=0.8,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )

    apply_axis_styling(ax, "Multichannel Analysis (1/3 Octave)", xlim=(16, 20000))
    ax.legend(frameon=True, loc="upper right")

    plt.savefig(os.path.join(output_dir, "signal_response_multichannel.png"))
    plt.close()


def generate_decomposition_plot(output_dir: str) -> None:
    """Generate time-domain decomposition plot with impulse response."""
    print("Generating signal_decomposition.png...")
    fs = 8000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Signal: sum of 250Hz and 1000Hz sines
    y = np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 1000 * t)

    # Filter into 1/1 octave bands
    bank = OctaveFilterBank(fs=fs, fraction=1, order=6, limits=[100, 2000])
    _, freq, xb = bank.filter(y, sigbands=True)

    num_plots = len(xb) + 2 # +1 for original, +1 for impulse response
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 1.8 * num_plots), sharex=False)

    # Fixed Y limits for decomposition
    y_lim = (-2.5, 2.5)

    # 1. Original Signal
    axes[0].plot(t, y, color="black", linewidth=1.2)
    axes[0].set_title("Original Signal (250 Hz + 1000 Hz Sum)", fontweight="bold")
    axes[0].set_ylim(y_lim)
    axes[0].set_xlim(0, 0.04)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # 2. Filtered Bands
    for i, (band_signal, f_center) in enumerate(zip(xb, freq)):
        axes[i + 1].plot(t, band_signal, color=colors[i % len(colors)], linewidth=1.2)
        axes[i + 1].set_title(f"Octave Band: {f_center:.0f} Hz", fontsize=10)
        axes[i + 1].set_ylim(y_lim)
        axes[i + 1].set_xlim(0, 0.04)

    # 3. Impulse Response (Stability Visualization)
    # Generate a separate impulse
    impulse = np.zeros(len(t))
    impulse[0] = 1.0
    _, _, ir_bands = bank.filter(impulse, sigbands=True)
    # Plot IR of the 1000Hz band (index 3 if bands are 125, 250, 500, 1000, 2000)
    # Let's find index of 1000Hz
    idx_1000 = np.argmin(np.abs(np.array(freq) - 1000))
    axes[-1].plot(t, ir_bands[idx_1000], color=COLOR_SECONDARY, linewidth=1.2)
    axes[-1].set_title(f"Impulse Response ({freq[idx_1000]:.0f} Hz Band) - Stability View", fontweight="bold")
    axes[-1].set_xlim(0, 0.04)
    axes[-1].set_xlabel("Time [s]")

    for ax in axes:
        ax.set_ylabel("Amp")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_decomposition.png"))
    plt.close()


def generate_weighting_responses(output_dir: str) -> None:
    """Plot A, C and Z weighting frequency responses."""
    print("Generating weighting_responses.png...")
    fs = 48000
    f = np.logspace(np.log10(10), np.log10(22000), 1000)
    
    from pyoctaveband import weighting_filter
    
    fig, ax = plt.subplots()
    
    curves = [
        ("A", "A-Weighting", COLOR_PRIMARY),
        ("C", "C-Weighting", COLOR_SECONDARY),
        ("Z", "Z-Weighting (Flat)", "black")
    ]
    
    for code, label, color in curves:
        # We need to measure response. Simplest way: IR then FFT
        impulse = np.zeros(fs)
        impulse[0] = 1.0
        weighted = weighting_filter(impulse, fs, curve=code)
        
        # Frequency response
        w, h = scipy_signal.freqz(weighted, [1], worN=8192, fs=fs)
        ax.semilogx(w, 20 * np.log10(np.abs(h) + 1e-9), label=label, color=color)

    apply_axis_styling(ax, "Frequency Weighting Curves (IEC 61672-1)", xlim=(10, 22000), ylim=(-50, 5))
    ax.legend()
    plt.savefig(os.path.join(output_dir, "weighting_responses.png"))
    plt.close()


def generate_time_weighting_plot(output_dir: str) -> None:
    """Visualize Fast and Slow time weighting response to a burst."""
    print("Generating time_weighting_analysis.png...")
    fs = 1000
    t = np.linspace(0, 3, fs * 3, endpoint=False)
    
    # 500ms burst of noise
    rng = np.random.default_rng(42)
    x = np.zeros_like(t)
    x[fs:fs+int(fs*0.5)] = rng.standard_normal(int(fs*0.5))
    
    from pyoctaveband import time_weighting
    
    # Square for energy
    x_sq = x**2
    fast = time_weighting(x, fs, mode="fast")
    slow = time_weighting(x, fs, mode="slow")
    
    fig, ax = plt.subplots()
    ax.plot(t, x_sq, color=COLOR_GRID, alpha=0.5, label="Instantaneous Energy ($x^2$)")
    ax.plot(t, fast, color=COLOR_PRIMARY, label="Fast (125ms)")
    ax.plot(t, slow, color=COLOR_SECONDARY, label="Slow (1000ms)")
    
    ax.set_title("Time Weighting Ballistics (Fast vs Slow)", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Squared Amplitude")
    ax.legend()
    ax.set_xlim(0.8, 3.0)
    plt.savefig(os.path.join(output_dir, "time_weighting_analysis.png"))
    plt.close()


def generate_crossover_plot(output_dir: str) -> None:
    """Visualize Linkwitz-Riley 4th Order Crossover."""
    print("Generating crossover_lr4.png...")
    fs = 48000
    # Log sweep for testing
    t = np.linspace(0, 1, fs, endpoint=False)
    x = scipy_signal.chirp(t, f0=20, t1=1, f1=20000, method='logarithmic')
    
    from pyoctaveband import linkwitz_riley
    
    lp, hp = linkwitz_riley(x, fs, freq=1000, order=4)
    
    # Frequency analysis
    w_lp, h_lp = scipy_signal.freqz(lp, x, worN=8192, fs=fs) # This is not correct for freqz, 
    # better to use IR
    impulse = np.zeros(fs)
    impulse[0] = 1.0
    lp_ir, hp_ir = linkwitz_riley(impulse, fs, freq=1000, order=4)
    
    w, h_lp = scipy_signal.freqz(lp_ir, worN=8192, fs=fs)
    _, h_hp = scipy_signal.freqz(hp_ir, worN=8192, fs=fs)
    
    fig, ax = plt.subplots()
    ax.semilogx(w, 20 * np.log10(np.abs(h_lp) + 1e-9), color=COLOR_PRIMARY, label="Low Pass (LR4)")
    ax.semilogx(w, 20 * np.log10(np.abs(h_hp) + 1e-9), color=COLOR_SECONDARY, label="High Pass (LR4)")
    ax.semilogx(w, 20 * np.log10(np.abs(h_lp + h_hp) + 1e-9), color="black", linestyle="--", label="Sum (Flat)")

    apply_axis_styling(ax, "Linkwitz-Riley Crossover (4th Order @ 1kHz)", xlim=(20, 20000), ylim=(-60, 5))
    ax.legend()
    plt.savefig(os.path.join(output_dir, "crossover_lr4.png"))
    plt.close()


if __name__ == "__main__":
    img_dir = ".github/images"
    os.makedirs(img_dir, exist_ok=True)
    
    # Generate all plots
    generate_filter_type_comparison(img_dir)
    generate_filter_responses(img_dir)
    generate_signal_responses(img_dir)
    generate_multichannel_response(img_dir)
    generate_decomposition_plot(img_dir)
    
    # NEW PLOTS
    generate_weighting_responses(img_dir)
    generate_time_weighting_plot(img_dir)
    generate_crossover_plot(img_dir)
    
    print("Graphics generated successfully.")