#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for IEC 61260-1 nominal frequency helpers and opt-in nominal label support.
"""

import numpy as np
import pytest

from phonometry import OctaveFilterBank, normalized_frequencies, octave_filter
from phonometry.metrology.frequencies import (
    _format_nominal_freq,
    _iec_e3_round,
    _infer_band_fraction,
    _nominal_freq_for_band,
    nominal_frequencies,
)

# --- _iec_e3_round ---

def test_iec_e3_round_msd_1_to_4():
    assert _iec_e3_round(1234.5) == pytest.approx(1230.0)
    assert _iec_e3_round(456.7) == pytest.approx(457.0)


def test_iec_e3_round_msd_5_to_9():
    assert _iec_e3_round(5678.0) == pytest.approx(5700.0)
    assert _iec_e3_round(99.1) == pytest.approx(99.0)


def test_iec_e3_round_nonpositive():
    assert _iec_e3_round(0) == 0
    assert _iec_e3_round(-1) == -1


# --- _nominal_freq_for_band ---

def test_nominal_freq_fraction1():
    assert _nominal_freq_for_band(15.849, 1) == pytest.approx(16.0)
    assert _nominal_freq_for_band(997.2, 1) == pytest.approx(1000.0)


def test_nominal_freq_fraction3():
    assert _nominal_freq_for_band(12.589, 3) == pytest.approx(12.5)
    assert _nominal_freq_for_band(1995.3, 3) == pytest.approx(2000.0)


def test_nominal_freq_other_fraction():
    assert _nominal_freq_for_band(706.0, 2) == pytest.approx(710.0)


# --- _format_nominal_freq ---

def test_format_below_1k():
    assert _format_nominal_freq(31.5) == "31.5"
    assert _format_nominal_freq(500.0) == "500"


def test_format_1k_and_above():
    assert _format_nominal_freq(1000.0) == "1k"
    assert _format_nominal_freq(16000.0) == "16k"
    assert _format_nominal_freq(1250.0) == "1.25k"


# --- nominal_frequencies — 4-tuple ---

def test_getansifrequencies_returns_labels():
    freq, _, _, labels = nominal_frequencies(fraction=3)
    assert isinstance(labels, list)
    assert all(isinstance(label, str) for label in labels)
    assert len(labels) == len(freq)
    assert "1k" in labels
    assert "31.5" in labels


def test_getansifrequencies_fraction1_labels():
    freq, _, _, labels = nominal_frequencies(fraction=1)
    assert "1k" in labels
    assert len(labels) == len(freq)


# --- OctaveFilterBank.nominal_freq attribute ---

def test_filterbank_nominal_freq_attribute():
    fb = OctaveFilterBank(fs=48000, fraction=1)
    assert hasattr(fb, "nominal_freq")
    assert isinstance(fb.nominal_freq, list)
    assert all(isinstance(label, str) for label in fb.nominal_freq)
    assert "1k" in fb.nominal_freq
    assert len(fb.nominal_freq) == fb.num_bands


# --- filter(nominal=False) — default exact floats ---

def test_filter_nominal_false_returns_floats():
    fb = OctaveFilterBank(fs=48000, fraction=3)
    x = np.zeros(4800)
    _, freq = fb.filter(x, nominal=False)
    assert all(isinstance(f, float) for f in freq)


# --- filter(nominal=True) — nominal string labels ---

def test_filter_nominal_true_returns_strings():
    fb = OctaveFilterBank(fs=48000, fraction=3)
    x = np.zeros(4800)
    _, freq = fb.filter(x, nominal=True)
    assert all(isinstance(f, str) for f in freq)
    assert "1k" in freq


def test_filter_nominal_true_with_sigbands():
    fb = OctaveFilterBank(fs=48000, fraction=1)
    x = np.zeros(4800)
    _, freq, xb = fb.filter(x, sigbands=True, nominal=True)
    assert all(isinstance(f, str) for f in freq)
    assert len(xb) == len(freq)


# --- octave_filter(nominal=True) ---

def test_octavefilter_nominal_true():
    x = np.zeros(4800)
    _, freq = octave_filter(x, fs=48000, fraction=3, nominal=True)
    assert all(isinstance(f, str) for f in freq)
    assert "1k" in freq


def test_octavefilter_nominal_false_default():
    x = np.zeros(4800)
    _, freq = octave_filter(x, fs=48000, fraction=3)
    assert all(isinstance(f, float) for f in freq)


def test_normalizedfreq_fraction1_and_invalid_fraction() -> None:
    freq = normalized_frequencies(1)
    assert 1000 in freq

    with pytest.raises(ValueError):
        normalized_frequencies(5)


def test_annex_e34_worked_rounding_examples() -> None:
    """IEC 61260-1:2014 E.3.4: 41,567 -> 41,6 (MSD 4, three significant
    figures) and 8 785,2 -> 8 800 (MSD 8, two significant figures)."""
    from reference_data import IEC61260_E34_EXAMPLES

    from phonometry.metrology.frequencies import _iec_e3_round

    for raw, printed in IEC61260_E34_EXAMPLES:
        assert _iec_e3_round(raw) == printed


def test_infer_band_fraction_octave_third_and_single() -> None:
    """Octave centres classify as fraction 1, one-third-octave as 3.

    A single band cannot be told apart from its neighbours' ratio, so the
    helper returns the conventional octave default (1).
    """
    octave = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    third = np.array([100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0])
    assert _infer_band_fraction(octave) == 1
    assert _infer_band_fraction(third) == 3
    # A wide octave set (>6 bands) is still octave, not one-third-octave.
    assert _infer_band_fraction(np.array([16.0, 31.5, 63.0, 125.0, 250.0,
                                          500.0, 1000.0, 2000.0])) == 1
    assert _infer_band_fraction(np.array([500.0])) == 1
