import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import pyoctaveband as PyOctaveBand

# Constants for common labels and styling
LABEL_FREQ_HZ = r"Frequency [Hz]"
LABEL_LEVEL_DB = "Level [dB]"
COLOR_PRIMARY = "#1f77b4"    # Professional Blue
COLOR_SECONDARY = "#d62728"  # Professional Red
COLOR_TERTIARY = "#2ca02c"   # Professional Green
COLOR_GRID = "#e0e0e0"

# Set a professional style
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.bbox": "tight"
})

def apply_common_styling(ax, title, xlim=None, ylim=None):
    """Apply standard styling to a plot axis."""
    ax.set_title(title, fontweight="bold", pad=15)
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
    """Generate plots for different filter configurations."""
    fs = 48000
    rng = np.random.default_rng()
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
        PyOctaveBand.octavefilter(
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
    """Generate plots for single channel signal analysis."""
    fs = 48000
    duration = 5
    x = np.arange(np.round(fs * duration)) / fs
    f1, f2, f3, f4, f5, f6 = 20, 100, 500, 2000, 4000, 15000
    y = 100 * (
        np.sin(2 * np.pi * f1 * x)
        + np.sin(2 * np.pi * f2 * x)
        + np.sin(2 * np.pi * f3 * x)
        + np.sin(2 * np.pi * f4 * x)
        + np.sin(2 * np.pi * f5 * x)
        + np.sin(2 * np.pi * f6 * x)
    )

    for frac, filename, title in [(1, "signal_response_fraction_1.png", "1/1 Octave Band Analysis"), 
                                  (3, "signal_response_fraction_3.png", "1/3 Octave Band Analysis")]:
        print(f"Generating {filename}...")
        spl, freq = PyOctaveBand.octavefilter(y, fs=fs, fraction=frac, order=6, limits=[12, 20000], show=False)
        
        fig, ax = plt.subplots()
        ax.semilogx(freq, spl, marker="o", markersize=5, linestyle="-", color=COLOR_PRIMARY, 
                    linewidth=1.5, markerfacecolor="white", markeredgewidth=1.5)
        apply_common_styling(ax, title, xlim=(11, 25000))
        plt.savefig(os.path.join(output_dir, filename))
        plt.close(fig)

def generate_multichannel_response(output_dir: str) -> None:
    """Generate plot for multichannel signal analysis."""
    print("Generating signal_response_multichannel.png...")
    fs = 48000
    duration = 5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Channel 1: Pink Noise
    rng = np.random.default_rng(42)
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
    pink_noise = np.sum(array, axis=1)
    pink_noise = pink_noise - np.mean(pink_noise)
    ch1 = pink_noise / np.max(np.abs(pink_noise))

    # Channel 2: Sine Sweep (Logarithmic) from 50Hz to 10kHz
    ch2 = scipy_signal.chirp(t, f0=50, t1=duration, f1=10000, method="logarithmic")

    x = np.vstack((ch1, ch2))
    spl, freq = PyOctaveBand.octavefilter(x, fs=fs, fraction=3, order=6, limits=[20, 20000])

    fig, ax = plt.subplots()
    ax.semilogx(freq, spl[0], marker="o", markersize=5, label="Left Ch: Pink Noise", 
                color=COLOR_PRIMARY, alpha=0.8, markerfacecolor="white", markeredgewidth=1.2)
    ax.semilogx(freq, spl[1], marker="s", markersize=5, label="Right Ch: Log Sweep (50Hz-10kHz)", 
                color=COLOR_SECONDARY, alpha=0.8, markerfacecolor="white", markeredgewidth=1.2)
    
    apply_common_styling(ax, "Multichannel 1/3 Octave Analysis", xlim=(16, 20000))
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    
    plt.savefig(os.path.join(output_dir, "signal_response_multichannel.png"))
    plt.close(fig)

def generate_time_domain_example(output_dir: str) -> None:
    """Show signal decomposition in time domain with consistent axis limits."""
    print("Generating signal_decomposition.png...")
    fs = 8000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Composite signal
    f1, f2 = 250, 1000
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    
    # Get bands in time domain
    _, freq, xb = PyOctaveBand.octavefilter(y, fs=fs, fraction=1, order=6, limits=[100, 2000], sigbands=True)
    
    num_plots = len(xb) + 1
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots), sharex=True, sharey=True)
    
    # Common Y limits for comparison
    y_margin = 0.2
    y_lim = (np.min(y) - y_margin, np.max(y) + y_margin)
    
    # Original
    axes[0].plot(t, y, color="black", linewidth=1.2, label="Original")
    axes[0].set_title("Original Composite Signal (250 Hz + 1000 Hz)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amp")
    axes[0].grid(True, alpha=0.3)
    
    # Colors for bands
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#ff7f0e", "#9467bd"]
    
    # Bands
    for i, (band_signal, f_center) in enumerate(zip(xb, freq)):
        color = colors[i % len(colors)]
        axes[i+1].plot(t, band_signal, color=color, linewidth=1.2)
        axes[i+1].set_title(f"Filtered Band: {f_center:.0f} Hz", fontsize=10)
        axes[i+1].set_ylabel("Amp")
        axes[i+1].grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(0, 0.05) # Show only first 50ms for detail
    
    # Set uniform limits
    for ax in axes:
        ax.set_ylim(y_lim)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_decomposition.png"))
    plt.close(fig)

if __name__ == "__main__":
    output_dir_main = ".github/images"
    os.makedirs(output_dir_main, exist_ok=True)
    generate_filter_responses(output_dir_main)
    generate_signal_responses(output_dir_main)
    generate_multichannel_response(output_dir_main)
    generate_time_domain_example(output_dir_main)
    print("Done.")
