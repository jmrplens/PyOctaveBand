import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

import pyoctaveband as PyOctaveBand


def generate_filter_responses(output_dir):
    fs = 48000
    # Dummy signal
    x = np.random.randn(fs)

    # (filename, fraction, order)
    # Using consistent naming: filter_fraction_{fraction}_order_{order}.png
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
            show=0,
            plot_file=os.path.join(output_dir, filename),
        )
        plt.close("all")


def generate_signal_responses(output_dir):
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

    # Single Channel 1 Octave
    print("Generating signal_response_fraction_1.png...")
    spl, freq = PyOctaveBand.octavefilter(y, fs=fs, fraction=1, order=6, limits=[12, 20000], show=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(freq, spl, "b-o")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel("Level [dB]")
    plt.xlim(11, 25000)
    ax.set_xticks([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
    ax.set_title("1/1 Octave Band Analysis")
    plt.savefig(os.path.join(output_dir, "signal_response_fraction_1.png"))
    plt.close()

    # Single Channel 1/3 Octave
    print("Generating signal_response_fraction_3.png...")
    spl, freq = PyOctaveBand.octavefilter(y, fs=fs, fraction=3, order=6, limits=[12, 20000], show=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(freq, spl, "b-o")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel("Level [dB]")
    plt.xlim(11, 25000)
    ax.set_xticks([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
    ax.set_title("1/3 Octave Band Analysis")
    plt.savefig(os.path.join(output_dir, "signal_response_fraction_3.png"))
    plt.close()


def generate_multichannel_response(output_dir):
    print("Generating signal_response_multichannel.png...")
    fs = 48000
    duration = 5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Channel 1: Pink Noise (High energy in low freqs, dropping 3dB/oct)
    # Using the Voss-McCartney algorithm implementation for consistency
    num_cols = 16
    array = np.empty((len(t), num_cols))
    array.fill(np.nan)
    array[0, :] = np.random.random(num_cols)
    array[:, 0] = np.random.random(len(t))
    for i in range(1, len(t)):
        for j in range(1, num_cols):
            if i % (2**j) == 0:
                array[i, j] = np.random.random()
            else:
                array[i, j] = array[i - 1, j]
    pink_noise = np.sum(array, axis=1)
    pink_noise = pink_noise - np.mean(pink_noise)
    ch1 = pink_noise / np.max(np.abs(pink_noise))

    # Channel 2: Sine Sweep (Logarithmic) from 50Hz to 10kHz
    ch2 = scipy_signal.chirp(t, f0=50, t1=duration, f1=10000, method="logarithmic")

    # Stack into stereo signal (2, N)
    x = np.vstack((ch1, ch2))

    # Filter with 1/3 Octave
    spl, freq = PyOctaveBand.octavefilter(x, fs=fs, fraction=3, order=6, limits=[20, 20000])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Channel 1
    ax.semilogx(freq, spl[0], "b-o", label="Left Ch: Pink Noise", alpha=0.7)
    # Plot Channel 2
    ax.semilogx(freq, spl[1], "r-s", label="Right Ch: Log Sweep (50Hz-10kHz)", alpha=0.7)

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel("Level [dB]")
    ax.set_title("Multichannel 1/3 Octave Band Analysis")
    ax.legend()

    plt.xlim(16, 20000)
    ax.set_xticks([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])

    plt.savefig(os.path.join(output_dir, "signal_response_multichannel.png"))
    plt.close()


if __name__ == "__main__":
    output_dir = ".github/images"
    os.makedirs(output_dir, exist_ok=True)
    generate_filter_responses(output_dir)
    generate_signal_responses(output_dir)
    generate_multichannel_response(output_dir)
    print("Done.")
