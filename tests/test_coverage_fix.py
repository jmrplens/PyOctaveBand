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

def test_showfilter_visual() -> None:
    from pyoctaveband.filter_design import _showfilter
    import os
    from unittest.mock import patch
    import matplotlib.pyplot as plt
    
    fs = 8000
    sos = [np.array([[1, 0, 0, 1, 0, 0]])] # Dummy SOS
    freq = [1000.0]
    freq_u = [1414.0]
    freq_d = [707.0]
    factor = np.array([1])
    
    plot_path = "tests/test_plot_coverage.png"
    if os.path.exists(plot_path):
        os.remove(plot_path)
        
    try:
        # Use plot_file to cover the save branch
        _showfilter(sos, freq, freq_u, freq_d, fs, factor, show=False, plot_file=plot_path)
        assert os.path.exists(plot_path)
        
        # Mock plt.show to cover the line without opening a window
        with patch.object(plt, 'show') as mock_show:
            _showfilter(sos, freq, freq_u, freq_d, fs, factor, show=True, plot_file=None)
            mock_show.assert_called_once()
            
        # Also cover the case where nothing is provided (early exit from save/show logic)
        _showfilter(sos, freq, freq_u, freq_d, fs, factor, show=False, plot_file=None)
    finally:
        if os.path.exists(plot_path):
            os.remove(plot_path)

def test_all_filter_architectures_design() -> None:
    from pyoctaveband.filter_design import _design_sos_filter
    types = ["butter", "cheby1", "cheby2", "ellip", "bessel"]
    fs = 48000
    for ft in types:
        sos = _design_sos_filter(
            freq=[1000], freq_d=[707], freq_u=[1414], fs=fs, order=4, 
            factor=np.array([1]), filter_type=ft, ripple=1.0, attenuation=40.0
        )
        assert len(sos) == 1
        assert len(sos[0]) > 0

def test_normalizedfreq_coverage() -> None:
    from pyoctaveband.frequencies import normalizedfreq
    res = normalizedfreq(1)
    assert 1000 in res
    with pytest.raises(ValueError):
        normalizedfreq(5)

def test_weighting_filter_class():
    from pyoctaveband import WeightingFilter
    fs = 48000
    wf = WeightingFilter(fs, curve="A")
    x = np.random.randn(2, fs)
    y = wf.filter(x)
    assert y.shape == x.shape
    
    wf_z = WeightingFilter(fs, curve="Z")
    y_z = wf_z.filter(x)
    assert np.all(y_z == x)
    
    with pytest.raises(ValueError):
        WeightingFilter(0)
    with pytest.raises(ValueError):
        WeightingFilter(fs, curve="invalid")

def test_octavefilterbank_repr() -> None:
    from pyoctaveband.core import OctaveFilterBank
    bank = OctaveFilterBank(48000)
    r = repr(bank)
    assert "OctaveFilterBank" in r
    assert "fs=48000" in r

def test_design_sos_with_internal_plot() -> None:
    from pyoctaveband.filter_design import _design_sos_filter
    import os
    # We provide a plot_file to trigger the internal call at line 62
    plot_path = "tests/test_design_plot.png"
    try:
        _ = _design_sos_filter([1000], [707], [1414], 8000, 2, np.array([1]), "butter", 0.1, 60, plot_file=plot_path)
        assert os.path.exists(plot_path)
    finally:
        if os.path.exists(plot_path):
            os.remove(plot_path)

def test_octavefilter_limits_none():
    from pyoctaveband import octavefilter
    from pyoctaveband.frequencies import getansifrequencies
    # Calling with limits=None covers frequencies.py:26
    spl, freq = octavefilter(np.random.randn(1000), 1000, limits=None)
    assert len(spl) > 0
    # Also directly call it
    f1, f2, f3 = getansifrequencies(1, limits=None)
    assert len(f1) > 0

def test_calculate_level_invalid():
    from pyoctaveband.core import OctaveFilterBank
    bank = OctaveFilterBank(48000)
    # This should hit core.py:218
    with pytest.raises(ValueError):
        bank._calculate_level(np.array([1.0]), "invalid_mode")
