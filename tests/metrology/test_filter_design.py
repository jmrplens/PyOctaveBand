#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for filter design helpers and response plotting utilities.
"""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import signal as sg

from phonometry import OctaveFilterBank, octave_filter
from phonometry.metrology.filter_design import _design_sos_filter, _showfilter


def _edge_gains_db(bank: OctaveFilterBank, band_idx: int) -> tuple[float, float]:
    """Measured gain (dB) at the band's lower and upper edge frequencies."""
    fsd = bank.fs / bank.factor[band_idx]
    w, h = sg.sosfreqz(bank.sos[band_idx], worN=2**16, fs=fsd)
    mag = 20 * np.log10(np.abs(h) + 1e-300)

    def gain_at(f: float) -> float:
        return float(mag[np.argmin(np.abs(w - f))])

    return gain_at(bank.freq_d[band_idx]), gain_at(bank.freq_u[band_idx])


@pytest.mark.parametrize("fraction", [1, 3])
def test_cheby2_minus3db_at_band_edges(fraction: float) -> None:
    """
    Verify cheby2 places its -3 dB points at the ANSI band edges.

    **Purpose:**
    ``Wn`` in Chebyshev II is the STOPBAND edge. Passing the band edges
    directly attenuated them by the full ``attenuation`` (-60 dB) instead
    of ~-3 dB, violating ANSI S1.11 band definitions.

    **Verification:**
    - Gain at every band's lower and upper edge is -3 dB (+/- 0.4 dB).
    """
    bank = OctaveFilterBank(
        fs=48000, fraction=fraction, order=6,
        limits=[100, 5000], filter_type="cheby2", attenuation=60.0,
    )
    for idx in range(bank.num_bands):
        g_lo, g_hi = _edge_gains_db(bank, idx)
        assert g_lo == pytest.approx(-3.01, abs=0.4), f"band {idx} lower edge: {g_lo:.2f} dB"
        assert g_hi == pytest.approx(-3.01, abs=0.4), f"band {idx} upper edge: {g_hi:.2f} dB"


def test_cheby2_broadband_levels_match_butter() -> None:
    """
    Verify cheby2 band levels agree with butter for broadband noise.

    **Purpose:**
    With the stopband placed on the band edges, cheby2 underestimated
    white-noise band levels by ~2.8 dB versus butter.

    **Verification:**
    - Max deviation between cheby2 and butter band levels < 0.5 dB.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(48000 * 5)
    spl_b, _ = octave_filter(x, 48000, fraction=3, filter_type="butter", limits=[100, 10000])
    spl_c, _ = octave_filter(x, 48000, fraction=3, filter_type="cheby2", limits=[100, 10000])
    diff = np.asarray(spl_c) - np.asarray(spl_b)
    assert np.abs(diff).max() < 0.5, f"max deviation {np.abs(diff).max():.2f} dB"


@pytest.mark.parametrize("fraction", [1, 3])
def test_bessel_minus3db_at_band_edges(fraction: float) -> None:
    """
    Verify bessel places its -3 dB points at the ANSI band edges.

    **Purpose:**
    ``norm="phase"`` shifted the -3 dB crossings inward (~-10 dB at the
    edges); ``norm="mag"`` defines them exactly at ``Wn``.

    **Verification:**
    - Gain at every band's lower and upper edge is -3 dB (+/- 0.6 dB).
    """
    bank = OctaveFilterBank(
        fs=48000, fraction=fraction, order=6,
        limits=[100, 5000], filter_type="bessel",
    )
    for idx in range(bank.num_bands):
        g_lo, g_hi = _edge_gains_db(bank, idx)
        assert g_lo == pytest.approx(-3.01, abs=0.6), f"band {idx} lower edge: {g_lo:.2f} dB"
        assert g_hi == pytest.approx(-3.01, abs=0.6), f"band {idx} upper edge: {g_hi:.2f} dB"


def test_showfilter_saves_shows_and_noops(tmp_path) -> None:
    """Verify _showfilter handles save, show, and no-output branches."""
    fs = 8000
    sos = [np.array([[1, 0, 0, 1, 0, 0]])]
    freq = [1000.0]
    freq_u = [1414.0]
    freq_d = [707.0]
    factor = np.array([1])
    plot_path = tmp_path / "test_plot_coverage.png"

    _showfilter(sos, freq, freq_u, freq_d, fs, factor, show=False, plot_file=str(plot_path))
    assert plot_path.exists()

    with patch.object(plt, "show") as mock_show:
        _showfilter(sos, freq, freq_u, freq_d, fs, factor, show=True, plot_file=None)
        mock_show.assert_called_once()

    _showfilter(sos, freq, freq_u, freq_d, fs, factor, show=False, plot_file=None)


def test_all_filter_architectures_design() -> None:
    """Verify all supported architectures produce SOS coefficients."""
    filter_types = ["butter", "cheby1", "cheby2", "ellip", "bessel"]

    for filter_type in filter_types:
        sos = _design_sos_filter(
            freq=[1000],
            freq_d=[707],
            freq_u=[1414],
            fs=48000,
            order=4,
            factor=np.array([1]),
            filter_type=filter_type,
            ripple=1.0,
            attenuation=40.0,
        )
        assert len(sos) == 1
        assert len(sos[0]) > 0


def test_design_sos_with_internal_plot(tmp_path) -> None:
    """Verify _design_sos_filter forwards plot output to _showfilter."""
    plot_path = tmp_path / "test_design_plot.png"

    _design_sos_filter(
        [1000],
        [707],
        [1414],
        8000,
        2,
        np.array([1]),
        "butter",
        0.1,
        60,
        plot_file=str(plot_path),
    )

    assert plot_path.exists()

def test_cheby2_low_attenuation_raises() -> None:
    """attenuation <= 3.01 dB has no -3 dB point: must raise, not produce NaN."""
    from phonometry.metrology.filter_design import _cheby2_transition_ratio

    with pytest.raises(ValueError, match="3.01"):
        _cheby2_transition_ratio(order=6, attenuation=3.0)
    with pytest.raises(ValueError, match="3.01"):
        OctaveFilterBank(fs=48000, fraction=3, filter_type="cheby2", attenuation=2.0)


def test_cheby2_stopband_edges_near_nyquist_stay_valid() -> None:
    """Pre-warped mapping must keep f1 < f2 < Nyquist even for bands near fs/2."""
    from phonometry.metrology.filter_design import _cheby2_stopband_edges

    fs = 2400.0
    fu = 0.9999 * fs / 2
    fd = fu / 1.26
    f1, f2 = _cheby2_stopband_edges(fd, fu, order=6, attenuation=60.0, fs=fs)
    assert 0 < f1 < fd
    assert fu < f2 < fs / 2


@pytest.mark.parametrize("fraction", [1, 3])
def test_cheby2_default_bank_meets_class1(fraction: float) -> None:
    """The default cheby2 bank must pass IEC 61260-1:2014 class 1 (audit N1 A5).

    scipy's cheby2 pins the deep-stopband floor at exactly ``attenuation`` dB.
    The former default of 60 dB sat 10 dB inside the class-1 requirement
    (>= 70 dB for Omega >= G^4). The default was raised to 72 dB, which the
    audit demonstrated passes class 1 with the same +0.400 dB passband margin.
    """
    from phonometry import verify_filter_class

    bank = OctaveFilterBank(
        fs=48000, fraction=fraction, order=6, limits=[100, 5000], filter_type="cheby2"
    )
    result = verify_filter_class(bank)
    assert result["overall_class"] == 1, result


def test_functional_octavefilter_cheby2_default_meets_class1() -> None:
    """The functional octave_filter() wrapper defaults attenuation to 72 dB so a
    cheby2 call meets IEC 61260-1 class 1, matching OctaveFilterBank (F1 follow-up:
    the wrapper previously still defaulted to 60 dB)."""
    import inspect

    from phonometry import verify_filter_class

    default_att = inspect.signature(octave_filter).parameters["attenuation"].default
    assert default_att == 72.0
    # A bank built with the wrapper's own default passes class 1; and the
    # functional call itself runs without raising.
    bank = OctaveFilterBank(
        fs=48000, fraction=1, order=6, limits=[100, 5000],
        filter_type="cheby2", attenuation=default_att,
    )
    assert verify_filter_class(bank)["overall_class"] == 1
    x = np.random.default_rng(0).standard_normal(48000)
    spl, _ = octave_filter(x, 48000, fraction=1, filter_type="cheby2", limits=[100, 5000])
    assert np.all(np.isfinite(spl))
