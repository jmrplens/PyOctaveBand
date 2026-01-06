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


def test_octave_filter_bank_invalid_init():
    """Verify that OctaveFilterBank raises appropriate errors for invalid parameters."""
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

def test_weighting_filter_invalid():
    """Verify error handling for invalid weighting curves."""
    x = np.random.randn(1000)
    with pytest.raises(ValueError, match="must be 'A', 'C' or 'Z'"):
        weighting_filter(x, 48000, curve="B")

def test_time_weighting_invalid():
    """Verify error handling for invalid time weighting modes."""
    x = np.random.randn(1000)
    with pytest.raises(ValueError, match="Invalid time weighting mode"):
        time_weighting(x, 48000, mode="instant")

def test_linkwitz_riley_invalid():
    """Verify error handling for Linkwitz-Riley order."""
    x = np.random.randn(1000)
    with pytest.raises(ValueError, match="order must be even"):
        linkwitz_riley(x, 48000, freq=1000, order=3)

def test_calculate_sensitivity_silent():
    """Verify error handling for silent reference signal."""
    x = np.zeros(1000)
    with pytest.raises(ValueError, match="Reference signal is silent"):
        calculate_sensitivity(x)

def test_octave_filter_vs_class_consistency():
    """Verify that octavefilter function and OctaveFilterBank class yield identical results."""
    fs = 44100
    x = np.random.randn(fs)
    params = {
        "fs": fs,
        "fraction": 3,
        "order": 6,
        "filter_type": "butter"
    }
    
    # 1. Using function
    spl_func, freq_func = octavefilter(x, **params)
    
    # 2. Using class
    bank = OctaveFilterBank(**params)
    spl_class, freq_class = bank.filter(x)
    
    assert np.allclose(spl_func, spl_class)
    assert np.allclose(freq_func, freq_class)

def test_single_sample_signal():
    """Verify handling of extremely short signals."""
    fs = 48000
    x = np.array([1.0])
    # Should not crash, though results might be meaningless
    spl, freq = octavefilter(x, fs)
    assert len(spl) == len(freq)
    assert not np.isnan(spl).any()

def test_multichannel_consistency():
    """Verify that processing 2 channels together is same as processing them separately."""
    fs = 16000
    x1 = np.random.randn(fs)
    x2 = np.random.randn(fs)
    x_stereo = np.vstack((x1, x2))
    
    bank = OctaveFilterBank(fs, fraction=1)
    
    # Separate
    spl1, _ = bank.filter(x1)
    spl2, _ = bank.filter(x2)
    
    # Together
    spl_stereo, _ = bank.filter(x_stereo)
    
    assert np.allclose(spl_stereo[0], spl1)
    assert np.allclose(spl_stereo[1], spl2)
