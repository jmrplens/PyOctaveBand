import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pyoctaveband as PyOctaveBand


def test_multichannel() -> None:
    """Test multichannel signal processing."""
    fs = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Channel 1: Pure Tone at 500 Hz
    ch1 = np.sin(2 * np.pi * 500 * t)

    # Channel 2: White Noise (High energy across all bands)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    ch2 = rng.standard_normal(len(t))

    x = np.vstack((ch1, ch2))

    print("Testing multichannel input shape:", x.shape)
    spl, freq = PyOctaveBand.octavefilter(x, fs, fraction=3)

    print("SPL shape:", spl.shape)
    assert spl.shape == (2, len(freq))

    # Analysis
    # Channel 1 should have a peak around 500Hz
    idx500 = np.argmin(np.abs(np.array(freq) - 500))
    spl_ch1_max_idx = np.argmax(spl[0])

    print(f"Channel 1 Max SPL Index: {spl_ch1_max_idx} (Freq: {freq[spl_ch1_max_idx]:.1f} Hz)")
    print(f"Target 500Hz Index: {idx500} (Freq: {freq[idx500]:.1f} Hz)")

    # Check if the peak is close to 500Hz (within 1 band index)
    assert abs(spl_ch1_max_idx - idx500) <= 1, "Channel 1 should peak near 500Hz"

    # Channel 2 (White Noise) should have significant energy across many bands
    # Let's check that the standard deviation of SPL is lower than the pure tone channel
    std_ch1 = np.std(spl[0])
    std_ch2 = np.std(spl[1])

    print(f"Std Dev SPL Channel 1 (Tone): {std_ch1:.2f} dB")
    print(f"Std Dev SPL Channel 2 (Noise): {std_ch2:.2f} dB")

    assert std_ch1 > std_ch2, "Pure tone should have higher spectral variance than white noise"

    print("Multichannel SPL test passed!")

    print("Testing sigbands with multichannel...")
    spl, freq, xb = PyOctaveBand.octavefilter(x, fs, fraction=3, sigbands=True)
    if xb is None:
        raise ValueError("xb should not be None")

    print(f"Number of bands: {len(xb)}")
    print(f"Shape of first band signal: {xb[0].shape}")
    assert len(xb) == len(freq)
    assert xb[0].shape == (2, len(t))
    print("Multichannel sigbands test passed!")


if __name__ == "__main__":
    test_multichannel()
