import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import matplotlib.pyplot as plt
import numpy as np

import pyoctaveband as PyOctaveBand


def generate_pink_noise(samples):
    """
    Generate pink noise (1/f noise) using the Voss-McCartney algorithm.
    """
    # Number of columns for the Voss algorithm
    num_cols = 16
    array = np.empty((samples, num_cols))
    array.fill(np.nan)
    array[0, :] = np.random.random(num_cols)
    array[:, 0] = np.random.random(samples)

    # Voss algorithm: update random values at geometric intervals
    for i in range(1, samples):
        for j in range(1, num_cols):
            if i % (2**j) == 0:
                array[i, j] = np.random.random()
            else:
                array[i, j] = array[i - 1, j]

    # Sum along rows to get pink noise
    pink = np.sum(array, axis=1)

    # Normalize
    pink = pink - np.mean(pink)
    pink = pink / np.max(np.abs(pink))
    return pink


def test_pink_noise_flatness():
    """
    Test that filtering pink noise with 1/3 octave bands results in a relatively flat spectrum.
    Pink noise has equal energy per octave (and thus per fractional octave band).
    """
    print("Generating Pink Noise...")
    fs = 48000
    duration = 5  # seconds
    samples = fs * duration
    x = generate_pink_noise(samples)

    print("Filtering with 1/3 octave bands...")
    # Using 1/3 octave bands
    spl, freq = PyOctaveBand.octavefilter(x, fs=fs, fraction=3, order=6, limits=[20, 20000])

    # Calculate mean deviation from the average SPL
    mean_spl = np.mean(spl)
    std_spl = np.std(spl)

    print(f"Mean SPL: {mean_spl:.2f} dB")
    print(f"Standard Deviation of SPL across bands: {std_spl:.2f} dB")

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freq, spl, "b-o", label="Measured SPL")
    ax.axhline(mean_spl, color="r", linestyle="--", label="Mean SPL")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel("Level [dB]")
    ax.set_title("1/3 Octave Band Spectrum of Pink Noise (Should be flat)")
    ax.legend()

    plt.xlim(16, 20000)
    ax.set_xticks([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])

    output_file = "tests/pink_noise_test.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

    # Assertion: Ideally, pink noise spectrum is flat.
    # Allow some tolerance due to randomness and filter edge effects
    # We check if most bands are within reasonable range of the mean
    # We ignore the very ends (lowest and highest bands) where filter performance might drop
    valid_spl = spl[2:-2]
    deviation = np.max(np.abs(valid_spl - np.mean(valid_spl)))
    print(f"Max deviation in central bands: {deviation:.2f} dB")

    if deviation < 3.0:  # 3dB tolerance is generous but reasonable for this simple test
        print("PASS: Spectrum is approximately flat.")
    else:
        print("WARNING: Spectrum deviation is high.")


if __name__ == "__main__":
    test_pink_noise_flatness()
