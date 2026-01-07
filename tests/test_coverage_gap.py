#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Specific tests to close coverage gaps in core logic and visualization.
"""

import os
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyoctaveband.core import OctaveFilterBank
from pyoctaveband.filter_design import _design_sos_filter, _showfilter
from pyoctaveband.frequencies import getansifrequencies, normalizedfreq
from pyoctaveband.utils import _resample_to_length, _typesignal


def test_typesignal_bypass() -> None:
    """Cover the branch where input is already a numpy array (identity check)."""
    x = np.array([1.0, 2.0])
    y = _typesignal(x)
    assert x is y # Confirms it doesn't re-wrap if already an ndarray


def test_typesignal_list() -> None:
    """Cover the branch where input is a list (needs conversion)."""
    x = [1.0, 2.0]
    y = _typesignal(x)
    assert isinstance(y, np.ndarray)
    assert y[0] == 1.0


def test_resample_to_length_padding() -> None:
    """Cover the branch where padding is needed in _resample_to_length."""
    # Create a case where length < target_length
    # signal length 10, factor 2 -> resampled 20.
    # Target 25 -> needs padding of 5.
    x = np.ones(10)
    target = 25
    y = _resample_to_length(x, 2, target)
    assert len(y) == target
    assert np.all(y[20:] == 0) # Verify padding


def test_showfilter_visual() -> None:
    """
    Test the visualization logic (_showfilter).
    
    **Purpose:**
    Ensure that the plotting logic works without errors, correctly handles the branch 
    where no output is requested, and correctly produces a file when a path is provided.
    """
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


def test_design_sos_with_internal_plot() -> None:
    """Cover the branch in _design_sos_filter that calls _showfilter."""
    # We provide a plot_file to trigger the internal call
    plot_path = "tests/test_design_plot.png"
    try:
        _ = _design_sos_filter([1000], [707], [1414], 8000, 2, np.array([1]), "butter", 0.1, 60, plot_file=plot_path)
        assert os.path.exists(plot_path)
    finally:
        if os.path.exists(plot_path):
            os.remove(plot_path)


def test_getansifrequencies_default_limits() -> None:
    """Cover the branch where limits is None in getansifrequencies."""
    freq, _, _ = getansifrequencies(fraction=1, limits=None)
    assert len(freq) > 0
    assert freq[0] > 10 # Default range starts at ~16Hz


def test_normalizedfreq_success() -> None:
    """Cover the success branch of normalizedfreq."""
    res = normalizedfreq(1)
    assert len(res) > 0
    assert 1000 in res


def test_normalizedfreq_error() -> None:
    """Directly cover the ValueError in normalizedfreq."""
    with pytest.raises(ValueError, match="Normalized frequencies only available"):
        normalizedfreq(5)


def test_design_sos_invalid_type() -> None:
    """Forced test for invalid filter type in the internal design function."""
    # The internal function returns empty arrays for unknown types instead of raising
    # because the validation is done at the class level.
    res = _design_sos_filter([100], [70], [140], 8000, 2, np.array([1]), "invalid", 0.1, 60)
    assert len(res) == 1
    assert res[0].size == 0


def test_calculate_level_invalid_mode() -> None:
    """Directly cover the invalid mode branch in _calculate_level."""
    bank = OctaveFilterBank(48000)
    with pytest.raises(ValueError, match="Invalid mode"):
        bank._calculate_level(np.array([1.0]), "unknown_mode")


def test_even_fraction_logic() -> None:
    """
    Cover the 'round(b) % 2 == 0' branch in frequencies.py (_initindex, _ratio).
    """
    # fraction=2 (1/2 octave) is even.
    freq, _, _ = getansifrequencies(fraction=2, limits=[100, 1000])
    assert len(freq) > 0
    # Basic check that frequencies are increasing
    assert np.all(np.diff(freq) > 0)


def test_resample_to_length_truncation() -> None:
    """
    Cover the branch where truncation is needed in _resample_to_length.
    (len(y_resampled) > target_length).
    """
    # Signal length 10, factor 2 -> resampled ~20.
    # Target 15 -> needs truncation.
    x = np.ones(10)
    target = 15
    y = _resample_to_length(x, 2, target)
    assert len(y) == target
    # The default resample_poly might produce boundary effects, but length is key here.


def test_low_fs_warning() -> None:
    """
    Cover the warning branch in _deleteouters when bands exceed Nyquist.
    """
    fs = 1000  # Nyquist 500
    # Request bands up to 2000 Hz
    with pytest.warns(UserWarning, match="frequencies above fs/2 removed"):
        freq, _, _ = getansifrequencies(fraction=1, limits=[100, 2000])
        # Manually verify removal happened if we were calling _genfreqs, 
        # but here we test the warning mechanism which is triggered in higher level calls
        # actually getansifrequencies doesn't warn, _genfreqs does.
        # Let's call OctaveFilterBank which calls _genfreqs
        OctaveFilterBank(fs, limits=[100, 2000])


def test_typesignal_tuple() -> None:
    """Cover the branch where input is a tuple."""
    x = (1.0, 2.0, 3.0)
    y = _typesignal(x)
    assert isinstance(y, np.ndarray)
    assert len(y) == 3


def test_all_filter_architectures_design() -> None:
    """
    Ensure _design_sos_filter runs without error for all supported types
    and parameters.
    """
    types = ["butter", "cheby1", "cheby2", "ellip", "bessel"]
    fs = 48000
    for ft in types:
        # We just check it doesn't crash and returns valid SOS
        sos = _design_sos_filter(
            freq=[1000], freq_d=[707], freq_u=[1414], fs=fs, order=4, 
            factor=np.array([1]), filter_type=ft, ripple=1.0, attenuation=40.0
        )
        assert len(sos) == 1
        assert len(sos[0]) > 0