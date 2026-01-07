import numpy as np
import pytest
from pyoctaveband.utils import _resample_to_length
from pyoctaveband.parametric_filters import time_weighting, weighting_filter

def test_resample_coverage():
    # Test 1D padding
    x = np.ones(10)
    # Factor 1 means no resampling, so length stays 10. Target 12 -> Pad
    y = _resample_to_length(x, 1, 12)
    assert len(y) == 12
    assert np.all(y[10:] == 0)

    # Test 2D padding
    x2 = np.ones((2, 10))
    y2 = _resample_to_length(x2, 1, 12)
    assert y2.shape == (2, 12)
    assert np.all(y2[:, 10:] == 0)

    # Test 1D slicing
    y3 = _resample_to_length(x, 1, 8)
    assert len(y3) == 8

    # Test 2D slicing
    y4 = _resample_to_length(x2, 1, 8)
    assert y4.shape == (2, 8)

def test_time_weighting_impulse_multichannel():
    # Test impulse mode with multichannel input
    fs = 1000
    x = np.zeros((2, fs))
    x[:, 0] = 1.0 # Impulse
    
    # This exercises the loop over time in the impulse branch
    y = time_weighting(x, fs, mode="impulse")
    assert y.shape == x.shape
    # Check that it decayed
    assert y[0, 100] < y[0, 0]

def test_weighting_filter_c_coverage():
    # Ensure C-weighting path is covered if it wasn't
    fs = 48000
    x = np.random.randn(fs)
    y = weighting_filter(x, fs, curve="C")
    assert len(y) == len(x)
