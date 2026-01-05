import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

# Add src to path to use the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from pyoctaveband import octavefilter

# Constants for professional styling
LABEL_FREQ_HZ = "Frequency [Hz]"
LABEL_LEVEL_DB = "Level [dB]"
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#d62728"
COLOR_GRID = "#e0e0e0"

# Global matplotlib configuration
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.bbox": "tight"
})

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

def generate_filter_responses(output_dir: str) -> None:
    """Generate plots for the filter bank responses."""
    fs = 48000
    rng = np.random.default_rng(42)
    x = rng.standard_normal(fs)

    configs = [
        ("filter_fraction_1_order_6.png", 1, 6),
        ("filter_fraction_1_order_16.png", 1, 16),
        ("filter_fraction_3_order_6.png", 3, 6),
        ("filter_fraction_3_order_16.png", 3, 16),
        ("filter_fraction_1.5_order_6.png", 1.5, 6),
        ("filter_fraction_1.5_order_16.png", 1.5, 16),
        ("filter_fraction_12_order_6.png", 12, 6),
        ("filter_fraction_24_order_6.png", 24, 6),
    ]

    for filename, fraction, order in configs:
        print(f"Generating {filename}...")
        octavefilter(
            x,
            fs=fs,
            fraction=fraction,
            order=order,
            limits=[12, 20000],
            show=False,
            plot_file=os.path.join(output_dir, filename),
        )
        plt.close("all")

def generate_signal_responses(output_dir: str) -> None:
    """Generate spectral analysis plots for a complex signal."""
    fs = 48000
    duration = 5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    freqs = [20, 100, 500, 2000, 4000, 15000]
    y = 100 * np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

    for frac, filename, title in [(1, "signal_response_fraction_1.png", "1/1 Octave Band Analysis"), 
                                  (3, "signal_response_fraction_3.png", "1/3 Octave Band Analysis")]:
        print(f"Generating {filename}...")
        spl, freq = octavefilter(y, fs=fs, fraction=frac, order=6, limits=[12, 20000], show=False)
        
        _, ax = plt.subplots()
        ax.semilogx(freq, spl, marker="o", markersize=5, linestyle="-", color=COLOR_PRIMARY, 
                    linewidth=1.5, markerfacecolor="white", markeredgewidth=1.5)
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
    spl, freq = octavefilter(x, fs=fs, fraction=3, order=6, limits=[20, 20000])

    _, ax = plt.subplots()
    ax.semilogx(freq, spl[0], marker="o", markersize=5, label="Left: Pink Noise", 
                color=COLOR_PRIMARY, alpha=0.8, markerfacecolor="white", markeredgewidth=1.2)
    ax.semilogx(freq, spl[1], marker="s", markersize=5, label="Right: Log Sweep", 
                color=COLOR_SECONDARY, alpha=0.8, markerfacecolor="white", markeredgewidth=1.2)
    
    apply_axis_styling(ax, "Multichannel Analysis (1/3 Octave)", xlim=(16, 20000))
    ax.legend(frameon=True, loc="upper right")
    
    plt.savefig(os.path.join(output_dir, "signal_response_multichannel.png"))
    plt.close()

def generate_decomposition_plot(output_dir: str) -> None:
    """Generate time-domain decomposition plot with synchronized axes."""
    print("Generating signal_decomposition.png...")
    fs = 8000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Signal: sum of 250Hz and 1000Hz sines
    y = np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 1000 * t)
    
    # Filter into 1/1 octave bands
    _, freq, xb = octavefilter(y, fs=fs, fraction=1, order=6, limits=[100, 2000], sigbands=True)
    
    num_plots = len(xb) + 1
    _, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots), sharex=True)
    
    # Fixed Y limits for all plots to allow direct comparison
    # The composite signal has peak at 2.0, individual bands at 1.0.
    y_lim = (-2.5, 2.5)
    
    # Original Signal
    axes[0].plot(t, y, color="black", linewidth=1.2)
    axes[0].set_title("Original Signal (250 Hz + 1000 Hz Sum)", fontweight="bold")
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # Filtered Bands
    for i, (band_signal, f_center) in enumerate(zip(xb, freq)):
        axes[i+1].plot(t, band_signal, color=colors[i % len(colors)], linewidth=1.2)
        axes[i+1].set_title(f"Octave Band: {f_center:.0f} Hz", fontsize=10)
        
    for ax in axes:
        ax.set_ylim(y_lim)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(0, 0.04) # Show first 40ms for clear waveform detail

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_decomposition.png"))
    plt.close()

if __name__ == "__main__":
    img_dir = ".github/images"
    os.makedirs(img_dir, exist_ok=True)
    generate_filter_responses(img_dir)
    generate_signal_responses(img_dir)
    generate_multichannel_response(img_dir)
    generate_decomposition_plot(img_dir)
    print("Graphics generated successfully.")