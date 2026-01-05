#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Advanced audio processing tests including Pink Noise spectral analysis.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure local package is used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pyoctaveband import octavefilter


def generate_pink_noise(samples: int) -> np.ndarray:
    """
    Generate pink noise (1/f noise) using the Voss-McCartney algorithm.
    
    :param samples: Number of samples to generate.
    :return: Normalized pink noise array.
    """
    num_cols = 16
    rng = np.random.default_rng(42)  # Seeded for reproducibility
    array = np.empty((samples, num_cols))
    array.fill(np.nan)
    array[0, :] = rng.random(num_cols)
    array[:, 0] = rng.random(samples)

    # Update random values at geometric intervals
    for i in range(1, samples):
        for j in range(1, num_cols):
            if i % (2**j) == 0:
                array[i, j] = rng.random()
            else:
                array[i, j] = array[i - 1, j]

    pink = np.sum(array, axis=1)
    pink = pink - np.mean(pink)
    return pink / np.max(np.abs(pink))


def test_pink_noise_flatness() -> None:
    """
    Verify that pink noise has approximately equal energy per fractional octave.
    The resulting SPL spectrum should be relatively flat.
    """
    print("Generating Pink Noise...")
    fs = 48000
    duration = 5.0
    samples = int(fs * duration)
    x = generate_pink_noise(samples)

    print("Analyzing with 1/3 octave bands...")
    spl, freq = octavefilter(x, fs=fs, fraction=3, order=6, limits=[20, 20000])

    mean_spl = np.mean(spl)
    std_spl = np.std(spl)

    print(f"Mean SPL: {mean_spl:.2f} dB")
    print(f"Standard Deviation: {std_spl:.2f} dB")

    # Plot results for visual verification
    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freq, spl, "b-o", label="Measured SPL", markerfacecolor="white")
    ax.axhline(mean_spl, color="r", linestyle="--", label="Mean SPL", alpha=0.7)

    ax.set_title("Pink Noise 1/3 Octave Spectrum (Flatness Check)", fontweight="bold")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level [dB]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.xlim(16, 20000)
    xticks = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    xticklabels = ["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    output_file = "tests/pink_noise_test.png"
    plt.savefig(output_file, dpi=150)
    print(f"Test plot saved to {output_file}")

    # Check flatness in central bands (ignoring edges)
    valid_spl = np.array(spl)[2:-2]
    deviation = np.max(np.abs(valid_spl - np.mean(valid_spl)))
    print(f"Max deviation in central bands: {deviation:.2f} dB")

    if deviation < 3.0:
        print("PASS: Spectrum is approximately flat.")
    else:
        print("FAIL: Spectrum deviation is too high.")


if __name__ == "__main__":
    test_pink_noise_flatness()