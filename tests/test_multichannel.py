#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Multichannel processing validation tests.
"""

import os
import sys

import numpy as np

# Ensure local package is used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pyoctaveband import octavefilter


def test_multichannel() -> None:
    """Validate processing of signals with multiple channels (stereo)."""
    fs = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Channel 1: Pure Tone at 500 Hz (Distinct peak expected)
    ch1 = np.sin(2 * np.pi * 500 * t)

    # Channel 2: White Noise (Broadband energy expected)
    rng = np.random.default_rng(42)
    ch2 = rng.standard_normal(len(t))

    x = np.vstack((ch1, ch2))

    print(f"Testing stereo input with shape: {x.shape}")
    spl, freq = octavefilter(x, fs, fraction=3)

    print(f"Resulting SPL shape: {spl.shape}")
    assert spl.shape == (2, len(freq)), "Output SPL should maintain channel count"

    # Verify Channel 1 (Tone) peaks near 500Hz
    target_idx = np.argmin(np.abs(np.array(freq) - 500))
    max_idx = np.argmax(spl[0])
    print(f"Ch 1 Peak: {freq[max_idx]:.1f} Hz (Target: {freq[target_idx]:.1f} Hz)")
    assert abs(max_idx - target_idx) <= 1, "Tone peak not detected in correct band"

    # Verify Channel 2 (Noise) has lower variance than Tone
    std_tone = np.std(spl[0])
    std_noise = np.std(spl[1])
    print(f"SPL Std Dev - Tone: {std_tone:.2f}, Noise: {std_noise:.2f}")
    assert std_tone > std_noise, "Tone should have higher spectral variance than noise"

    print("Multichannel SPL test: OK")

    # Verify time-domain band splitting
    print("Testing 'sigbands' mode...")
    _, _, xb = octavefilter(x, fs, fraction=3, sigbands=True)
    if xb is None:
        raise ValueError("xb should not be None when sigbands=True")

    print(f"Extracted {len(xb)} bands.")
    assert xb[0].shape == (2, len(t)), "Bands should maintain stereo shape and length"
    print("Multichannel sigbands test: OK")


if __name__ == "__main__":
    test_multichannel()