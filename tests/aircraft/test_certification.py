#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for aircraft-noise EPNL (ICAO Annex 16 Vol. I, Appendix 2).

The oracles are independent of the implementation: the analytic noy breakpoints
of Table A2-3, and the worked examples of the ICAO Doc 9501 Environmental
Technical Manual Vol. I (2018) -- the turbofan tone-correction Table 3-7 and the
integrated-method EPNL Table 4-4.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

from phonometry.aircraft.certification import (
    NOY_BANDS,
    _tone_background,
    effective_perceived_noise_level,
    epnl_from_pnlt,
    perceived_noise_level,
    perceived_noisiness,
    tone_correction,
)

# ICAO Doc 9501 ETM Vol. I (2018) Table 4-4 integrated-method EPNL example.
_PNLTR_44 = [
    84.62,
    85.84,
    85.37,
    88.57,
    88.82,
    88.03,
    88.76,
    87.06,
    86.92,
    90.39,
    89.89,
    91.00,
    90.08,
    89.71,
    89.61,
    90.21,
    91.14,
    92.10,
    93.68,
    94.89,
    95.87,
    97.06,
    97.40,
    96.23,
    94.73,
    92.30,
    88.75,
    86.96,
    85.41,
    83.88,
    83.01,
]
_DTR_44 = [
    0.3950,
    0.3950,
    0.3951,
    0.3951,
    0.3952,
    0.3953,
    0.3954,
    0.3956,
    0.3957,
    0.3960,
    0.3963,
    0.3967,
    0.3973,
    0.3981,
    0.3992,
    0.4009,
    0.4033,
    0.4066,
    0.4108,
    0.4153,
    0.4196,
    0.4231,
    0.4256,
    0.4273,
    0.4285,
    0.4294,
    0.4299,
    0.4304,
    0.4307,
    0.4309,
    0.4311,
]

_B = list(NOY_BANDS)

# ICAO Doc 9501 ETM Vol. I (2018) Table 3-7 turbofan spectrum (bands 1-2 blank).
_SPL_37 = [
    -999.0,
    -999.0,
    70.0,
    62.0,
    70.0,
    80.0,
    82.0,
    83.0,
    76.0,
    80.0,
    80.0,
    79.0,
    78.0,
    80.0,
    78.0,
    76.0,
    79.0,
    85.0,
    79.0,
    78.0,
    71.0,
    60.0,
    54.0,
    45.0,
]


def test_noy_breakpoints() -> None:
    # Table A2-3, band 14 (1000 Hz): SPL(b)=40 -> n=1, SPL(e)=25 -> n=0.3,
    # SPL(d)=16 -> n=0.1.
    b14 = _B.index(1000.0)
    spl = np.full(24, -999.0)
    spl[b14] = 40.0
    assert perceived_noisiness(spl)[b14] == pytest.approx(1.0, rel=1e-6)
    spl[b14] = 25.0
    assert perceived_noisiness(spl)[b14] == pytest.approx(0.3, rel=1e-6)
    spl[b14] = 16.0
    assert perceived_noisiness(spl)[b14] == pytest.approx(0.1, rel=1e-6)


def test_noy_floor_below_spld_is_zero() -> None:
    b14 = _B.index(1000.0)
    spl = np.full(24, -999.0)
    spl[b14] = 10.0  # < SPL(d) = 16
    assert perceived_noisiness(spl)[b14] == 0.0


def test_noy_requires_24_bands() -> None:
    with pytest.raises(
        ValueError, match="'spl' must contain the 24 one-third-octave-band levels"
    ):
        perceived_noisiness([1.0, 2.0, 3.0])


def test_pnl_closed_form() -> None:
    # One band at 1000 Hz, SPL = SPL(b) = 40 -> n = 1, others 0.
    # N = 0.85*1 + 0.15*1 = 1.0 -> PNL = 40 + (10/lg2)*lg(1) = 40.
    b14 = _B.index(1000.0)
    spl = np.full(24, -999.0)
    spl[b14] = 40.0
    assert perceived_noise_level(spl) == pytest.approx(40.0, abs=1e-6)


def test_pnl_two_equal_bands() -> None:
    # Two bands each n = 1 -> n_max = 1, Σn = 2 -> N = 0.85 + 0.15*2 = 1.15.
    b14 = _B.index(1000.0)
    b11 = _B.index(500.0)  # SPL(b) @ 500 Hz = 40
    spl = np.full(24, -999.0)
    spl[b14] = 40.0
    spl[b11] = 40.0
    expected = 40.0 + (10.0 / np.log10(2.0)) * np.log10(1.15)
    assert perceived_noise_level(spl) == pytest.approx(expected, rel=1e-9)


def test_tone_correction_etm_table_37() -> None:
    # ETM Vol. I Table 3-7: the max tone correction is C = 2.0 dB at 2500 Hz.
    assert tone_correction(_SPL_37) == pytest.approx(2.0, abs=1e-6)


def test_tone_correction_background_column() -> None:
    # SPL'' (Step 7) column of Table 3-7 at 125, 160, 2500, 10000 Hz; F at 2500.
    spl_dd, excess = _tone_background(_SPL_37)
    assert spl_dd[4] == pytest.approx(71.0, abs=0.05)  # 125 Hz
    assert spl_dd[5] == pytest.approx(77.0 + 2.0 / 3.0, abs=0.05)  # 160 Hz
    assert spl_dd[17] == pytest.approx(79.0, abs=0.05)  # 2500 Hz
    assert spl_dd[23] == pytest.approx(45.0, abs=0.05)  # 10 kHz
    assert excess[17] == pytest.approx(6.0, abs=0.05)  # F = 85 − 79


def test_tone_correction_none_when_flat() -> None:
    assert tone_correction(np.full(24, 60.0)) == 0.0


def test_epnl_table_44() -> None:
    # ETM Vol. I Table 4-4: integrated-method reference EPNL = 92.61892 EPNdB,
    # PNLTM = 97.40, window records 4-28 (0-based 3-27).
    epnl, pnltm, kf, kl = epnl_from_pnlt(np.array(_PNLTR_44), np.array(_DTR_44))
    assert pnltm == pytest.approx(97.40, abs=1e-6)
    assert (kf, kl) == (3, 27)
    assert epnl == pytest.approx(92.61892, abs=1e-2)


def test_epnl_uniform_dt_matches_energy_sum() -> None:
    # With dt=0.5 and T0=10, EPNL = 10 lg(Σ 0.5·10^(PNLT/10)) − 10 lg(10).
    p = np.array([90.0, 95.0, 90.0])  # all within 10 dB of the peak
    epnl, _pnltm, kf, kl = epnl_from_pnlt(p, 0.5)
    expected = 10.0 * np.log10(np.sum(0.5 * 10.0 ** (p / 10.0))) - 10.0 * np.log10(10.0)
    assert epnl == pytest.approx(expected, rel=1e-9)
    assert (kf, kl) == (0, 2)


def test_effective_perceived_noise_level_bundle() -> None:
    rng = np.random.default_rng(0)
    peak = 20.0 * np.exp(-((np.arange(20) - 10.0) ** 2) / 8.0)
    spectra = 60.0 + peak[:, None] + rng.standard_normal((20, 24))
    res = effective_perceived_noise_level(spectra)
    assert res.pnlt.shape == (20,)
    assert res.epnl <= res.pnltm  # duration correction is <= 0 here
    assert res.band_limits[0] <= int(np.argmax(res.pnlt)) <= res.band_limits[1]


def test_epnl_single_element_dt_treated_as_scalar() -> None:
    # A one-element dt (list or 1-D array) broadcasts like a scalar.
    p = np.array([90.0, 95.0, 90.0])
    a, *_ = epnl_from_pnlt(p, [0.5])
    b, *_ = epnl_from_pnlt(p, 0.5)
    assert a == pytest.approx(b, rel=1e-12)


def test_effective_perceived_noise_level_rejects_bad_shape() -> None:
    two_dimensional = np.zeros((5, 10))
    with pytest.raises(ValueError, match=r"'spectra' must have shape \(K, 24\)"):
        effective_perceived_noise_level(two_dimensional)


def test_helicopter_procedure_catches_low_frequency_rotor_tone() -> None:
    """App. 2 4.3.1 Step 1: helicopters start the tone correction at 50 Hz.

    A rotor tone in the 63 Hz band sits below the aeroplane procedure's
    80 Hz start band, so the aeroplane chain must apply no tone correction
    for it while the helicopter chain must (C > 0), raising the EPNL.
    """
    spectrum = np.full(24, 70.0)
    spectrum[1] += 12.0  # 63 Hz rotor tone, below the aeroplane start band
    spectra = np.tile(spectrum, (10, 1))

    assert tone_correction(spectrum) == 0.0  # aeroplane start (80 Hz)
    assert tone_correction(spectrum, start_band=0) > 0.0  # helicopter start

    aeroplane = effective_perceived_noise_level(spectra)
    helicopter = effective_perceived_noise_level(spectra, procedure="helicopter")
    assert np.all(aeroplane.tone_correction == 0.0)
    assert np.all(helicopter.tone_correction > 0.0)
    assert helicopter.epnl > aeroplane.epnl


def test_effective_perceived_noise_level_rejects_bad_procedure() -> None:
    spectra = np.zeros((5, 24))
    with pytest.raises(ValueError, match="procedure"):
        effective_perceived_noise_level(spectra, procedure="rocket")


def test_epnl_result_plot_smoke() -> None:
    rng = np.random.default_rng(0)
    peak = 20.0 * np.exp(-((np.arange(20) - 10.0) ** 2) / 8.0)
    spectra = 60.0 + peak[:, None] + rng.standard_normal((20, 24))
    res = effective_perceived_noise_level(spectra)
    assert res.plot() is not None


def test_epnl_exposed_at_package_top_level() -> None:
    import phonometry

    assert (
        phonometry.aircraft.effective_perceived_noise_level
        is effective_perceived_noise_level
    )


def test_bandsharing_adjustment_applied() -> None:
    # App. 2 section 4.4.2/4.4.3 (ETM GM/AMC A2 4.4.2): when the tone
    # correction at the PNLTM record dips below the average of the five
    # records within one second of it, Delta_B is the shortfall, added to
    # PNLTM before the 10 dB-down window and included in EPNL. Hand-built
    # case: C = [2, 2, 0, 2, 2] around the peak -> Delta_B = 1.6.
    from phonometry.aircraft.certification import epnl_from_pnlt

    pnlt = np.array([80.0, 90.0, 95.0, 100.0, 95.0, 90.0, 80.0])
    c = np.array([0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0])
    plain_epnl, plain_pnltm, _, _ = epnl_from_pnlt(pnlt, 0.5)
    adj_epnl, adj_pnltm, _, _ = epnl_from_pnlt(pnlt, 0.5, tone_corrections=c)
    assert adj_pnltm - plain_pnltm == pytest.approx(1.6)
    assert adj_epnl - plain_epnl == pytest.approx(1.6)
    # No suppression at the peak -> Delta_B = 0.
    c2 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    same_epnl, same_pnltm, _, _ = epnl_from_pnlt(pnlt, 0.5, tone_corrections=c2)
    assert same_pnltm == pytest.approx(plain_pnltm)
    assert same_epnl == pytest.approx(plain_epnl)


def test_bandsharing_window_scales_with_record_duration() -> None:
    # The one-second window is time-based, not a fixed five-record slice:
    # with 1 s records only the immediate neighbours fall within +/- 1 s.
    from phonometry.aircraft.certification import epnl_from_pnlt

    pnlt = np.array([80.0, 90.0, 95.0, 100.0, 95.0, 90.0, 80.0])
    c = np.array([0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0])
    _, pnltm_1s, _, _ = epnl_from_pnlt(pnlt, 1.0, tone_corrections=c)
    plain = float(np.max(pnlt))
    # Window = [2, 0, 2] -> mean 4/3 -> Delta_B = 4/3.
    assert pnltm_1s - plain == pytest.approx(4.0 / 3.0)
    # Per-record durations follow the same rule (uniform 1 s array).
    _, pnltm_arr, _, _ = epnl_from_pnlt(pnlt, np.full(7, 1.0), tone_corrections=c)
    assert pnltm_arr == pytest.approx(pnltm_1s)


def test_effective_epnl_exposes_bandsharing_field() -> None:
    rng = np.random.default_rng(7)
    spectra = 60.0 + 5.0 * rng.standard_normal((9, 24))
    from phonometry import aircraft

    res = aircraft.effective_perceived_noise_level(spectra, 0.5)
    assert res.bandsharing_adjustment >= 0.0
    assert res.pnltm == pytest.approx(
        float(np.max(res.pnlt)) + res.bandsharing_adjustment
    )
