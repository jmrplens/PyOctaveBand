#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Stress tests and signal processing theory verification.
Focuses on edge cases that might be problematic in DSP.
"""

import numpy as np
import pytest

from pyoctaveband import octavefilter


def test_nyquist_frequency_content() -> None:
    """
    Theory: A signal exactly at fs/2 (Nyquist) is ambiguous in digital systems.
    Verification: Pass a Nyquist-frequency signal through the filter bank.
    Expectation: The library should not crash, and energy should be handled gracefully (attenuated).
    """
    fs = 48000
    t = np.linspace(0, 1, fs, endpoint=False)
    # Signal at fs/2: alternating [1, -1, 1, -1]
    x = np.cos(np.pi * fs * t) 
    
    # Request analysis up to Nyquist
    spl, _ = octavefilter(x, fs, limits=[100, 23000])
    assert not np.isnan(spl).any()


def test_aliasing_behavior() -> None:
    """
    Theory: Frequencies above fs/2 appear as aliased lower frequencies.
    Verification: Pass a signal at fs * 0.75 (above Nyquist).
    Expectation: The filter bank should treat the aliased frequency (fs * 0.25) correctly.
    """
    fs = 1000
    target_freq = 750 # > Nyquist (500)
    aliased_freq = 250
    t = np.linspace(0, 1, fs, endpoint=False)
    x = np.sin(2 * np.pi * target_freq * t)
    
    # Analyze. 250Hz is a standard band for 1/1 octave? 
    # Mid bands: ..., 125, 250, 500
    spl, freq = octavefilter(x, fs, fraction=1, limits=[100, 400])
    
    # Find 250Hz band
    idx_250 = np.argmin(np.abs(np.array(freq) - aliased_freq))
    assert np.argmax(spl) == idx_250 # Aliased signal peaks at 250Hz


def test_extreme_high_order_stability() -> None:
    """
    Theory: Very high order IIR filters can become numerically unstable even with SOS.
    Verification: Request order 50 and 100 (extreme).
    Expectation: SOS should maintain stability (no NaNs), but we check output sanity.
    """
    fs = 48000
    x = np.random.default_rng(42).standard_normal(fs)
    
    for order in [50, 100]:
        spl, _ = octavefilter(x, fs, order=order)
        assert not np.isnan(spl).any()
        assert np.all(np.isfinite(spl))


def test_dc_offset_rejection() -> None:
    """
    Theory: Band-pass filters should reject DC (0 Hz).
    Verification: Pass a constant signal (x = 1.0) with detrend=True.
    Expectation: The signal should be centered around 0 before filtering, 
    leading to extremely low energy levels.
    """
    fs = 8000
    x = np.ones(fs) # Pure DC
    
    # 1. With detrend=True (default)
    spl, _ = octavefilter(x, fs, limits=[100, 2000], detrend=True)
    # SPL should be extremely low (near noise floor)
    assert np.all(spl < -100)

    # 2. With detrend=False
    # The step response at t=0 will still generate some transient energy,
    # but we verify it's at least not a crash.
    spl_nodetrend, _ = octavefilter(x, fs, limits=[100, 2000], detrend=False)
    assert not np.isnan(spl_nodetrend).any()


def test_extreme_sampling_rates() -> None:
    """
    Verification: Test extremely low and extremely high sampling rates.
    """
    # Extremely low (e.g. 100Hz)
    fs_low = 100
    x_low = np.zeros(fs_low)
    _, freq_low = octavefilter(x_low, fs_low, limits=[10, 40])
    assert len(freq_low) > 0
    
    # Extremely high (e.g. 1MHz)
    fs_high = 1000000
    x_high = np.zeros(fs_high // 10)
    _, freq_high = octavefilter(x_high, fs_high, limits=[1000, 20000])
    assert len(freq_high) > 0


def test_huge_calibration_factor() -> None:
    """
    Verification: Sensitivity factor of 1e10 (extreme scaling).
    """
    fs = 8000
    x = np.random.default_rng(42).standard_normal(fs)
    spl, _ = octavefilter(x, fs, calibration_factor=1e10)
    assert np.all(spl > 100) # Should be massive but not Inf


def test_multichannel_mismatched_lengths() -> None:
    """
    Verification: Pass a 2D array where channels have different sizes (not possible in numpy, 
    but we check if it handles non-standard list of lists if we support it).
    Actually, _typesignal ensures it's an array.
    """
    fs = 8000
    # Numpy arrays must have rectangular shape, so we test with a list of lists of different lengths
    x = [[1.0, 2.0, 3.0], [1.0, 2.0]]
    # This should probably raise an error or handle it via numpy's default behavior
    with pytest.raises(Exception):
        # type ignore because list of lists of floats is technically not what we hint, but what user might pass
        octavefilter(x, fs) # type: ignore


def test_sos_stability_at_low_freq_high_fs() -> None:
    """
    Theory: Filtering at 16Hz with fs=192kHz is extremely difficult.
    Verification: Check stability for this case.
    """
    fs = 192000
    x = np.random.default_rng(42).standard_normal(fs)
    # The bank should use a very high decimation factor
    spl, freq = octavefilter(x, fs, limits=[15, 30])
    assert not np.isnan(spl).any()
    assert 16.0 in np.round(freq)
