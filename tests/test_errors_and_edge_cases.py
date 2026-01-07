#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Tests for error handling and edge cases across all modules.
"""

import numpy as np
import pytest

from pyoctaveband import (
    OctaveFilterBank,
    calculate_sensitivity,
    linkwitz_riley,
    octavefilter,
    time_weighting,
    weighting_filter,
)


def test_octave_filter_bank_invalid_init() -> None:
    """
    Verify that OctaveFilterBank raises appropriate errors for invalid parameters.

    **Purpose:**
    Ensure that the class constructor robustly validates all input arguments to prevent
    unstable configurations or math errors during processing.

    **Verification:**
    - Pass invalid `fs`, `fraction`, `order`.
    - Pass invalid `limits` (too few elements, negative values, reversed).
    - Pass an unknown `filter_type`.

    **Expectation:**
    - Each invalid call must raise a `ValueError` with a specific error message.
    """
    with pytest.raises(ValueError, match="fs' must be positive"):
        OctaveFilterBank(fs=0)
    
    with pytest.raises(ValueError, match="fraction' must be positive"):
        OctaveFilterBank(fs=48000, fraction=-1)
        
    with pytest.raises(ValueError, match="order' must be positive"):
        OctaveFilterBank(fs=48000, order=0)
        
    with pytest.raises(ValueError, match="list of two frequencies"):
        OctaveFilterBank(fs=48000, limits=[1000])
        
    with pytest.raises(ValueError, match="must be positive"):
        OctaveFilterBank(fs=48000, limits=[-10, 1000])
        
    with pytest.raises(ValueError, match="less than the upper limit"):
        OctaveFilterBank(fs=48000, limits=[2000, 1000])
        
    with pytest.raises(ValueError, match="Invalid filter_type"):
        OctaveFilterBank(fs=48000, filter_type="invalid")


def test_weighting_filter_invalid() -> None:
    """
    Verify error handling for invalid weighting curves.

    **Purpose:**
    Ensure that only supported IEC curves (A, C, Z) are accepted.

    **Verification:**
    - Call `weighting_filter` with an unsupported curve name.

    **Expectation:**
    - Raise `ValueError`.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="must be 'A', 'C' or 'Z'"):
        weighting_filter(x, 48000, curve="B")


def test_time_weighting_invalid() -> None:
    """
    Verify error handling for invalid time weighting modes.

    **Purpose:**
    Ensure that only standardized ballistic modes (Fast, Slow, Impulse) are accepted.

    **Verification:**
    - Call `time_weighting` with an unsupported mode string.

    **Expectation:**
    - Raise `ValueError`.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="Invalid time weighting mode"):
        time_weighting(x, 48000, mode="instant")


def test_linkwitz_riley_invalid() -> None:
    """
    Verify error handling for Linkwitz-Riley order.

    **Purpose:**
    Linkwitz-Riley filters require an even order to ensure correct phase alignment.

    **Verification:**
    - Call `linkwitz_riley` with an odd order.

    **Expectation:**
    - Raise `ValueError`.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="order must be even"):
        linkwitz_riley(x, 48000, freq=1000, order=3)


def test_calculate_sensitivity_silent() -> None:
    """
    Verify error handling for silent reference signal.

    **Purpose:**
    Prevent division by zero during calibration when the reference signal is empty or silent.

    **Verification:**
    - Pass an array of zeros to `calculate_sensitivity`.

    **Expectation:**
    - Raise `ValueError`.
    """
    x = np.zeros(1000)
    with pytest.raises(ValueError, match="Reference signal is silent"):
        calculate_sensitivity(x)


def test_octave_filter_vs_class_consistency() -> None:
    """
    Verify that octavefilter function and OctaveFilterBank class yield identical results.

    **Purpose:**
    Ensure that the functional wrapper correctly proxies all arguments to the underlying
    class implementation, maintaining behavioral parity.

    **Verification:**
    - Process the same signal using both the function and the class.
    - Compare output SPL and frequencies.

    **Expectation:**
    - Arrays should be numerically identical.
    """
    fs = 44100
    rng = np.random.default_rng(42)
    x = rng.standard_normal(fs)
    fraction = 3
    order = 6
    filter_type = "butter"
    
    # 1. Using function
    spl_func, freq_func = octavefilter(x, fs=fs, fraction=fraction, order=order, filter_type=filter_type)
    
    # 2. Using class
    bank = OctaveFilterBank(fs=fs, fraction=fraction, order=order, filter_type=filter_type)
    spl_class, freq_class = bank.filter(x)
    
    assert np.allclose(spl_func, spl_class)
    assert np.allclose(freq_func, freq_class)


def test_single_sample_signal() -> None:
    """
    Verify handling of extremely short signals.

    **Purpose:**
    Ensure the library doesn't crash when provided with a single-sample signal,
    which might occur in edge-case stream processing.

    **Verification:**
    - Pass a single-element array to `octavefilter`.

    **Expectation:**
    - The code should return valid (though likely low) SPL values without crashing.
    """
    fs = 48000
    x = np.array([1.0])
    spl, freq = octavefilter(x, fs)
    assert len(spl) == len(freq)
    assert not np.isnan(spl).any()


def test_multichannel_consistency() -> None:
    """
    Verify that processing 2 channels together is same as processing them separately.

    **Purpose:**
    Confirm that the multichannel implementation correctly isolates channels and does
    not introduce cross-channel artifacts.

    **Verification:**
    - Create two independent signals.
    - Process them separately.
    - Process them as a stereo pair.
    - Compare results.

    **Expectation:**
    - SPL values for each channel should match exactly.
    """
    fs = 16000
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal(fs)
    x2 = rng.standard_normal(fs)
    x_stereo = np.vstack((x1, x2))
    
    bank = OctaveFilterBank(fs, fraction=1)
    
    # Separate
    spl1, _ = bank.filter(x1)
    spl2, _ = bank.filter(x2)
    
    # Together
    spl_stereo, _ = bank.filter(x_stereo)
    
    assert np.allclose(spl_stereo[0], spl1)
    assert np.allclose(spl_stereo[1], spl2)
