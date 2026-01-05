#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Multichannel processing validation tests.
"""

import numpy as np

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

    spl, freq = octavefilter(x, fs, fraction=3)

    assert spl.shape == (2, len(freq)), "Output SPL should maintain channel count"

    # Verify Channel 1 (Tone) peaks near 500Hz
    target_idx = np.argmin(np.abs(np.array(freq) - 500))
    max_idx = np.argmax(spl[0])
    assert abs(max_idx - target_idx) <= 1, "Tone peak not detected in correct band"

    # Verify Channel 2 (Noise) has lower variance than Tone
    std_tone = np.std(spl[0])
    std_noise = np.std(spl[1])
    assert std_tone > std_noise, "Tone should have higher spectral variance than noise"

    # Verify time-domain band splitting
    _, _, xb = octavefilter(x, fs, fraction=3, sigbands=True)
    assert xb is not None, "xb should not be None when sigbands=True"
    assert xb[0].shape == (2, len(t)), "Bands should maintain stereo shape and length"
