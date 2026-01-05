#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Advanced audio processing tests including Pink Noise spectral analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from pyoctaveband import octavefilter


def generate_pink_noise(samples: int) -> np.ndarray:
    """
    Generate pink noise (1/f noise) using the Voss-McCartney algorithm.
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
    fs = 48000
    duration = 2.0  # Reduced for faster test
    samples = int(fs * duration)
    x = generate_pink_noise(samples)

    spl, freq = octavefilter(x, fs=fs, fraction=3, order=6, limits=[20, 20000])

    mean_spl = np.mean(spl)

    # Check flatness in central bands (ignoring edges)
    valid_spl = np.array(spl)[2:-2]
    deviation = np.max(np.abs(valid_spl - np.mean(valid_spl)))

    # Plot results for visual verification (optional in CI)
    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freq, spl, "b-o", label="Measured SPL", markerfacecolor="white")
    ax.axhline(mean_spl, color="r", linestyle="--", label="Mean SPL", alpha=0.7)
    ax.set_title("Pink Noise 1/3 Octave Spectrum (Flatness Check)", fontweight="bold")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level [dB]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.savefig("tests/pink_noise_test.png", dpi=150)
    plt.close()

    assert deviation < 3.0, f"Spectrum deviation too high: {deviation:.2f} dB"
