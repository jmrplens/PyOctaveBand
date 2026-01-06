#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Tests for parametric filters: Weighting (A, C), Time Weighting and Linkwitz-Riley.
"""

import numpy as np

from pyoctaveband import calculate_sensitivity, linkwitz_riley, octavefilter, time_weighting, weighting_filter


def test_calibration_logic():
    """Verify that calibration correctly maps digital RMS to target SPL."""
    fs = 48000
    # Create a 'recording' of a 94dB tone (RMS = 0.5 for example)
    rms_ref = 0.5
    t = np.linspace(0, 1, fs)
    ref_signal = rms_ref * np.sqrt(2) * np.sin(2 * np.pi * 1000 * t)
    
    # Calculate sensitivity
    factor = calculate_sensitivity(ref_signal, target_spl=94.0)
    
    # Analyze same signal with that factor
    spl, freq = octavefilter(ref_signal, fs, fraction=1, limits=[800, 1200], calibration_factor=factor)
    
    # It should be exactly 94 dB
    assert abs(spl[0] - 94.0) < 0.01

def test_dbfs_logic():
    """Verify dBFS mode returns RMS relative to 1.0."""
    fs = 48000
    # Sine wave with peak 1.0 -> RMS 0.707 -> -3.01 dBFS
    t = np.linspace(0, 1, fs)
    x = np.sin(2 * np.pi * 1000 * t)
    
    spl, freq = octavefilter(x, fs, fraction=1, limits=[800, 1200], dbfs=True)
    
    assert abs(spl[0] - (-3.01)) < 0.05

def test_a_weighting_response():
    """
    Verify A-weighting frequency response.
    Standard values:
    100Hz: -19.1 dB
    1000Hz: 0 dB
    8000Hz: -1.1 dB
    """
    fs = 48000
    duration = 1.0
    t = np.linspace(0, duration, fs, endpoint=False)
    
    test_freqs = [100, 1000, 8000]
    expected_gain = [-19.1, 0.0, -1.1]
    
    for f, expected in zip(test_freqs, expected_gain):
        # Generate tone
        x = np.sin(2 * np.pi * f * t)
        y = weighting_filter(x, fs, curve="A")
        
        # Calculate gain in dB (RMS)
        gain_db = 20 * np.log10(np.std(y) / np.std(x))
        assert abs(gain_db - expected) < 1.0, f"A-weighting at {f}Hz failed. Got {gain_db:.1f}dB, expected {expected}dB"

def test_c_weighting_response():
    """
    Verify C-weighting frequency response.
    Standard values:
    31.5Hz: -3.0 dB
    1000Hz: 0 dB
    8000Hz: -3.0 dB
    """
    fs = 48000
    duration = 1.0
    t = np.linspace(0, duration, fs, endpoint=False)
    
    test_freqs = [31.5, 1000, 8000]
    expected_gain = [-3.0, 0.0, -3.0]
    
    for f, expected in zip(test_freqs, expected_gain):
        x = np.sin(2 * np.pi * f * t)
        y = weighting_filter(x, fs, curve="C")
        
        gain_db = 20 * np.log10(np.std(y) / np.std(x))
        assert abs(gain_db - expected) < 1.0, f"C-weighting at {f}Hz failed. Got {gain_db:.1f}dB, expected {expected}dB"

def test_time_weighting_fast():
    """
    Verify Fast (125ms) time weighting response to a step.
    The signal should reach ~63% of its final value in one tau (125ms).
    """
    fs = 1000
    tau = 0.125
    x = np.ones(int(fs * 2)) # Step signal
    
    y = time_weighting(x, fs, mode="fast")
    
    # Check at index corresponding to tau
    idx_tau = int(fs * tau)
    # Expected: 1 - exp(-1) approx 0.632
    assert abs(y[idx_tau] - 0.632) < 0.05

def test_linkwitz_riley_sum():
    """
    Verify that the sum of Linkwitz-Riley bands is flat.
    The gain of the sum should be 1.0 (0 dB).
    """
    fs = 48000
    x = np.random.randn(fs)
    
    # Split at 1000 Hz
    lp, hp = linkwitz_riley(x, fs, freq=1000, order=4)
    
    # Sum of bands
    y_sum = lp + hp
    
    # Check RMS ratio
    gain_db = 20 * np.log10(np.std(y_sum) / np.std(x))
    assert abs(gain_db) < 0.1, f"Linkwitz-Riley sum not flat: {gain_db:.2f} dB"

def test_weighting_z_bypass():
    """Verify Z-weighting is a bypass."""
    x = np.random.randn(1000)
    y = weighting_filter(x, 48000, curve="Z")
    assert np.all(x == y)