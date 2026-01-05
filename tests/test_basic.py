#  Copyright (c) 2020. Jose M. Requena-Plens

"""
Basic test and usage example for pyoctaveband.
"""

import numpy as np
import pyoctaveband as PyOctaveBand

def test_octave_filter_basic():
    # Configuration
    fs = 48000
    duration = 1.0  # Reduced for faster testing
    
    # Generate multi-tone signal
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    freqs = [20, 100, 500, 2000, 4000, 15000]
    y = 100 * np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

    # 1. Filter and get only SPL spectrum
    spl, freq = PyOctaveBand.octavefilter(y, fs=fs, fraction=3, order=6, limits=[12, 20000], show=False)
    
    assert len(spl) == len(freq)
    assert len(freq) > 0
    assert not np.isnan(spl).any()

def test_octave_filter_sigbands():
    fs = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    y = np.sin(2 * np.pi * 1000 * t)

    # 2. Filter and get signals in time-domain bands
    spl, freq, xb = PyOctaveBand.octavefilter(
        y, fs=fs, fraction=1, order=6, limits=[500, 2000], show=False, sigbands=True
    )

    assert len(xb) == len(freq)
    assert xb[0].shape == y.shape
