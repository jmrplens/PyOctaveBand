#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.vibration.human.exposure`.

The frequency weightings are validated against the ISO 8041-1:2017 Annex B
design-goal factors (Tables B.1-B.9) and the Table 1 reference-frequency
factors, cross-checked against the ISO 2631-1/-2 and ISO 5349-1 tabulated
weightings.  The exposure arithmetic reproduces the ISO 5349-2 Annex E worked
examples, and the assessment follows the Directive 2002/44/EC action/limit
values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from reference_data import (
    ISO8041_1_ANNEX_B_FACTORS,
    ISO8041_1_TABLE4_TRANSITIONS,
    ISO8041_1_TABLE5_TOLERANCES,
)

from phonometry.vibration.human import exposure as hv


def _fc(n: int) -> float:
    """True IEC 61260 one-third-octave centre ``10^(n/10)`` Hz."""
    return 10.0 ** (n / 10.0)


# ---------------------------------------------------------------------------
# Frequency weightings vs ISO 8041-1:2017 Annex B (Tables B.1-B.9).
# ---------------------------------------------------------------------------
# (name, band index n, design-goal factor) sampled across every weighting.
_ANNEX_B = [
    ("Wk", 0, 0.4825),
    ("Wk", 8, 1.054),
    ("Wk", 20, 0.08873),
    ("Wk", -10, 0.03121),
    ("Wd", 0, 1.011),
    ("Wd", 13, 0.1004),
    ("Wd", 26, 0.0003164),
    ("Wc", 1, 1.000),
    ("Wc", 8, 0.9739),
    ("Wc", 20, 0.05665),
    ("We", 0, 0.8798),
    ("We", 7, 0.2012),
    ("We", 20, 0.007071),
    ("Wj", 15, 1.000),
    ("Wj", 0, 0.4844),
    ("Wj", 20, 0.7075),
    ("Wb", 8, 1.054),
    ("Wb", 0, 0.3853),
    ("Wb", 20, 0.1154),
    ("Wf", -8, 1.004),
    ("Wf", -10, 0.6951),
    ("Wf", 0, 0.02352),
    ("Wh", 8, 0.7272),
    ("Wh", 10, 0.9514),
    ("Wh", 20, 0.1602),
    ("Wh", 30, 0.01346),
    ("Wm", 2, 0.9342),
    ("Wm", 10, 0.4941),
    ("Wm", -1, 0.7003),
    ("Wm", 20, 0.04013),
]


@pytest.mark.parametrize(("name", "n", "expected"), _ANNEX_B)
def test_annex_b_design_goal_factors(name: str, n: int, expected: float) -> None:
    """|H| at the true band centre matches the Annex B four-figure factor."""
    got = hv.weighting_factors(name, _fc(n))[0]
    # 0,1 % relative + a floor for the deep-attenuation four-figure entries.
    assert got == pytest.approx(expected, rel=1e-3, abs=5e-6)


# (name, exact reference frequency Hz, Table 1 weighting factor at ref).
_TABLE_1 = [
    ("Wh", 500.0 / (2.0 * math.pi), 0.2020),
    ("Wk", 100.0 / (2.0 * math.pi), 0.7718),
    ("Wb", 100.0 / (2.0 * math.pi), 0.8126),
    ("Wc", 100.0 / (2.0 * math.pi), 0.5145),
    ("Wd", 100.0 / (2.0 * math.pi), 0.1261),
    ("We", 100.0 / (2.0 * math.pi), 0.06287),
    ("Wj", 100.0 / (2.0 * math.pi), 1.019),
    ("Wm", 100.0 / (2.0 * math.pi), 0.3362),
    ("Wf", 2.5 / (2.0 * math.pi), 0.3888),
]


@pytest.mark.parametrize(("name", "freq", "expected"), _TABLE_1)
def test_table_1_reference_frequency_factors(
    name: str, freq: float, expected: float
) -> None:
    """|H| at the Table 1 reference frequency matches the tabulated factor."""
    got = hv.weighting_factors(name, freq)[0]
    assert got == pytest.approx(expected, rel=1.5e-3)


def test_annex_b_full_tables_reproduce() -> None:
    """Every printed Annex B factor (Tables B.1-B.9, 318 bands) reproduces."""
    for name, rows in ISO8041_1_ANNEX_B_FACTORS.items():
        for n, printed in rows:
            got = float(hv.weighting_factors(name, _fc(n))[0])
            # The tables print four significant figures (<= 0,05 % rounding).
            assert got == pytest.approx(printed, rel=1e-3), (name, n)


def _tolerance_region(freq: float, name: str) -> int:
    """Table 5 tolerance region index of ``freq`` for weighting ``name``."""
    ft1, ft2, ft3, ft4 = ISO8041_1_TABLE4_TRANSITIONS[name]
    if freq <= ft1:
        return 0
    if freq < ft2:
        return 1
    if freq <= ft3:
        return 2
    if freq < ft4:
        return 3
    return 4


def test_table_5_tolerance_envelope() -> None:
    """|H| stays inside the Table 5 envelope of the printed design goal
    at every Annex B one-third-octave band, for all nine weightings.
    """
    for name, rows in ISO8041_1_ANNEX_B_FACTORS.items():
        for n, printed in rows:
            freq = _fc(n)
            upper, lower = ISO8041_1_TABLE5_TOLERANCES[_tolerance_region(freq, name)]
            ratio = float(hv.weighting_factors(name, freq)[0]) / printed - 1.0
            assert -lower <= ratio <= upper, (name, n, ratio)


def test_iso5349_1_wh_third_octave_table_a2() -> None:
    """The Wh factors match ISO 5349-1 Table A.2 to its three figures."""
    # (nominal band centre, Whi) from ISO 5349-1:2001 Table A.2.
    a2 = {
        6.3: 0.727,
        8.0: 0.873,
        10.0: 0.951,
        12.5: 0.958,
        16.0: 0.896,
        20.0: 0.782,
        25.0: 0.647,
        31.5: 0.519,
        40.0: 0.411,
        100.0: 0.160,
        250.0: 0.0634,
        1000.0: 0.0135,
    }
    # Evaluate at the *true* centres the table is computed at.
    n_of = {
        6.3: 8,
        8.0: 9,
        10.0: 10,
        12.5: 11,
        16.0: 12,
        20.0: 13,
        25.0: 14,
        31.5: 15,
        40.0: 16,
        100.0: 20,
        250.0: 24,
        1000.0: 30,
    }
    for nominal, expected in a2.items():
        got = hv.weighting_factors("Wh", _fc(n_of[nominal]))[0]
        assert got == pytest.approx(expected, rel=2e-3, abs=5e-4)


def test_iso2631_2_wm_table_a1() -> None:
    """Wm matches ISO 2631-2 Table A.1 at representative true centres."""
    a1 = {2: 0.934, 3: 0.932, 10: 0.494, 18: 0.0834, -3: 0.368}
    for n, expected in a1.items():
        got = hv.weighting_factors("Wm", _fc(n))[0]
        assert got == pytest.approx(expected, rel=2e-3, abs=5e-4)


def test_weighting_response_fields_and_db() -> None:
    resp = hv.frequency_weighting("Wk", [1.0, 6.3096, 100.0])
    assert resp.name == "Wk"
    assert resp.frequencies.shape == (3,)
    assert np.allclose(resp.magnitude, np.abs(resp.response))
    assert np.allclose(resp.magnitude_db, 20.0 * np.log10(resp.magnitude))
    assert resp.plot  # method exists


def test_unknown_weighting_raises() -> None:
    with pytest.raises(ValueError, match="Unknown weighting"):
        hv.frequency_weighting("Wz", [10.0])


def test_frequency_weighting_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        hv.frequency_weighting("Wk", [])


def test_dc_and_negative_frequencies_are_blocked() -> None:
    resp = hv.frequency_weighting("Wk", [0.0, -1.0, 1.0])
    assert resp.magnitude[0] == 0.0  # DC blocked by the high-pass
    assert resp.magnitude[1] == 0.0
    assert resp.magnitude[2] > 0.0


# ---------------------------------------------------------------------------
# apply_weighting (frequency-domain application of the exact response).
# ---------------------------------------------------------------------------
def test_apply_weighting_scales_sine_by_magnitude() -> None:
    fs = 2000.0
    f0 = 80.0
    t = np.arange(int(4 * fs)) / fs
    x = np.sqrt(2.0) * np.sin(2.0 * math.pi * f0 * t)  # unit-r.m.s. amplitude
    y = hv.apply_weighting(x, fs, name="Wk")
    factor = hv.weighting_factors("Wk", f0)[0]
    # Interior r.m.s. (drop edges) equals |H(f0)| * input r.m.s. (~1).
    interior = y[int(0.5 * fs) : -int(0.5 * fs)]
    assert float(np.sqrt(np.mean(interior**2))) == pytest.approx(factor, rel=2e-2)


def test_apply_weighting_validates() -> None:
    two_dimensional = np.zeros((2, 2))
    with pytest.raises(ValueError, match="1-D"):
        hv.apply_weighting(two_dimensional, 1000.0, name="Wk")
    with pytest.raises(ValueError, match="positive"):
        hv.apply_weighting([1.0, 2.0], 0.0, name="Wk")


# ---------------------------------------------------------------------------
# Band method a_w (ISO 2631-1 Eq. (9) / ISO 5349-1 Eq. (A.1)).
# ---------------------------------------------------------------------------
def test_weighted_acceleration_matches_manual_sum() -> None:
    freqs = np.array([8.0, 16.0, 31.5, 63.0])
    accel = np.array([0.5, 0.8, 0.3, 0.1])
    result = hv.weighted_acceleration(accel, freqs, "Wk")
    factors = hv.weighting_factors("Wk", freqs)
    expected = math.sqrt(np.sum((factors * accel) ** 2))
    assert result.overall == pytest.approx(expected)
    assert np.allclose(result.weighted, factors * accel)
    assert result.weighting_name == "Wk"


def test_weighted_acceleration_single_band_equals_factor_times_level() -> None:
    result = hv.weighted_acceleration([1.0], [10.0], "Wh")
    assert result.overall == pytest.approx(hv.weighting_factors("Wh", 10.0)[0])


def test_weighted_acceleration_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        hv.weighted_acceleration([1.0, 2.0], [8.0], "Wk")


# ---------------------------------------------------------------------------
# Time-domain metrics.
# ---------------------------------------------------------------------------
def test_rms_metrics_on_pure_sine() -> None:
    fs = 1000.0
    t = np.arange(int(10 * fs)) / fs
    a = 2.0
    x = a * np.sin(2.0 * math.pi * 20.0 * t)  # r.m.s. = a/sqrt(2)
    rms = a / math.sqrt(2.0)
    # Crest factor of a sine is sqrt(2) (discrete peak sampling -> ~0,2 %).
    assert hv.crest_factor(x) == pytest.approx(math.sqrt(2.0), rel=5e-3)
    # VDV = (integral a_w^4 dt)^(1/4); for a sine of duration T:
    # mean of sin^4 = 3/8 -> VDV = a*(3/8*T)^(1/4).
    duration = t[-1] + 1.0 / fs
    expected_vdv = a * (3.0 / 8.0 * duration) ** 0.25
    assert hv.vibration_dose_value(x, fs) == pytest.approx(expected_vdv, rel=1e-2)
    # MSDV = (integral a_w^2 dt)^(1/2) = rms * sqrt(T).
    assert hv.motion_sickness_dose_value(x, fs) == pytest.approx(
        rms * math.sqrt(duration), rel=1e-2
    )


def test_running_rms_of_constant_power_signal() -> None:
    fs = 500.0
    # Steady sine -> running r.m.s. settles to the signal r.m.s.
    t = np.arange(int(5 * fs)) / fs
    x = math.sqrt(2.0) * np.sin(2.0 * math.pi * 15.0 * t)
    for method in ("linear", "exponential"):
        r = hv.running_rms(x, fs, integration_time=1.0, method=method)
        assert r.shape == x.shape
        assert float(r[-1]) == pytest.approx(1.0, rel=5e-2)
    assert hv.mtvv(x, fs) == pytest.approx(float(np.max(hv.running_rms(x, fs))))


def test_running_rms_validation() -> None:
    with pytest.raises(ValueError, match="method"):
        hv.running_rms([1.0, 2.0], 100.0, method="bogus")
    with pytest.raises(ValueError, match="integration_time"):
        hv.running_rms([1.0, 2.0], 100.0, integration_time=0.0)


def test_crest_factor_zero_signal() -> None:
    assert hv.crest_factor(np.zeros(10)) == 0.0


def test_crest_factor_warns_above_nine() -> None:
    # A lone spike gives crest = 10 / (10/sqrt(200)) = sqrt(200) ~ 14.1 > 9.
    x = np.zeros(200)
    x[0] = 10.0
    with pytest.warns(hv.HumanVibrationWarning, match="exceeds 9"):
        cf = hv.crest_factor(x)
    assert cf > 9.0


# ---------------------------------------------------------------------------
# Vector sum and daily exposure (ISO 2631-1 Eq. (10); ISO 5349-1/-2).
# ---------------------------------------------------------------------------
def test_vibration_total_value_unweighted() -> None:
    # ISO 5349-1 Eq. (1): a_hv = sqrt(x^2+y^2+z^2).
    assert hv.vibration_total_value([3.0, 4.0, 0.0]) == pytest.approx(5.0)


def test_vibration_total_value_with_k_factors() -> None:
    # ISO 2631-1 7.2.3 health seated: k = 1,4 / 1,4 / 1,0.
    got = hv.vibration_total_value([0.5, 0.3, 0.8], k=[1.4, 1.4, 1.0])
    expected = math.sqrt((1.4 * 0.5) ** 2 + (1.4 * 0.3) ** 2 + (1.0 * 0.8) ** 2)
    assert got == pytest.approx(expected)


def test_wbv_exposure_basis_x_axis_dominates() -> None:
    # Directive 2002/44/EC Annex Part B point 1: A(8) is based on the highest
    # of 1,4*a_wx, 1,4*a_wy, a_wz. Here 1,4*0,5 = 0,7 beats a_wz = 0,6.
    assert hv.wbv_exposure_basis(0.5, 0.2, 0.6) == pytest.approx(0.7)


def test_wbv_exposure_basis_z_axis_dominates_and_differs_from_vector_total() -> None:
    # Same axis set as the docs example: max(0,49; 0,392; 0,62) = 0,62, well
    # below the ISO 2631-1 Eq. (10) vector total 0,882 for the same axes.
    basis = hv.wbv_exposure_basis(0.35, 0.28, 0.62)
    assert basis == pytest.approx(0.62)
    total = hv.vibration_total_value([0.35, 0.28, 0.62], k=[1.4, 1.4, 1.0])
    assert total == pytest.approx(0.882, abs=5e-4)
    assert basis < total


def test_wbv_exposure_basis_rejects_negative_or_nonfinite() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        hv.wbv_exposure_basis(-0.1, 0.2, 0.3)
    with pytest.raises(ValueError, match="non-negative"):
        hv.wbv_exposure_basis(0.1, math.nan, 0.3)


def test_iso5349_2_example_e2_1_single_tool() -> None:
    """ISO 5349-2 E.2.1: a_hv=7,4 m/s2, T=2,5 h -> A(8)=4,1 m/s2."""
    a8 = hv.daily_exposure(7.4, 2.5 * 3600.0)
    assert a8 == pytest.approx(4.1, abs=0.05)


def test_iso5349_2_example_e2_4_burst() -> None:
    """ISO 5349-2 E.2.4: a_hv=14,6 m/s2, T=4000 s (= 1,1 h) -> A(8)=5,4 m/s2."""
    # T = (1000 nuts/day / 5 nuts measured) * 20 s = 4000 s per Eq. (E.4); the
    # standard rounds 4000 s to "1,1 h". A(8) = 14,6*sqrt(4000/28800) = 5,4.
    a8 = hv.daily_exposure(14.6, 4000.0)
    assert a8 == pytest.approx(5.4, abs=0.05)


def test_iso5349_2_example_e3_forestry_multi_tool() -> None:
    """ISO 5349-2 E.3: three tasks combine to A(8)=3,6 m/s2."""
    partials = [
        hv.partial_exposure(4.6, 2 * 3600.0),  # brush-saw  -> 2,3
        hv.partial_exposure(6.0, 1 * 3600.0),  # felling    -> 2,1
        hv.partial_exposure(3.6, 2 * 3600.0),  # stripping  -> 1,8
    ]
    assert partials[0] == pytest.approx(2.3, abs=0.05)
    assert partials[1] == pytest.approx(2.1, abs=0.05)
    assert partials[2] == pytest.approx(1.8, abs=0.05)
    assert hv.combine_partial_exposures(partials) == pytest.approx(3.6, abs=0.05)


def test_iso5349_1_example_multi_operation() -> None:
    """ISO 5349-1 5.3 worked example: 1 h/3 h/0,5 h -> A(8)=3,4 m/s2."""
    a8 = hv.hav_daily_exposure([2.0, 3.5, 10.0], [3600.0, 3 * 3600.0, 0.5 * 3600.0])
    assert a8 == pytest.approx(3.4, abs=0.05)


def test_hav_daily_exposure_matches_partial_combination() -> None:
    values = [3.0, 5.0]
    durations = [2 * 3600.0, 1 * 3600.0]
    direct = hv.hav_daily_exposure(values, durations)
    partials = [
        hv.partial_exposure(v, d) for v, d in zip(values, durations, strict=True)
    ]
    assert direct == pytest.approx(hv.combine_partial_exposures(partials))


def test_energy_equivalent_acceleration() -> None:
    # ISO 2631-1 Eq. (B.3).
    got = hv.energy_equivalent_acceleration([1.0, 2.0], [1.0, 3.0])
    assert got == pytest.approx(math.sqrt((1 + 4 * 3) / 4))


def test_energy_equivalent_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        hv.energy_equivalent_acceleration([1.0, 2.0], [1.0, -1.0])


def test_hav_vwf_lifetime_matches_table_c1() -> None:
    """ISO 5349-1 Table C.1 / Eq. (C.1): A(8)=7 -> ~4 years; 3,7 -> ~8."""
    assert hv.hav_vwf_lifetime_years(7.0) == pytest.approx(4.0, abs=0.3)
    assert hv.hav_vwf_lifetime_years(3.7) == pytest.approx(8.0, abs=0.6)
    assert hv.hav_vwf_lifetime_years(14.0) == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------------
# Directive 2002/44/EC assessment.
# ---------------------------------------------------------------------------
def test_exposure_assessment_hav_zones() -> None:
    below = hv.exposure_assessment(2.0, kind="hav")
    assert below.zone == "below action"
    assert not below.exceeds_action
    action = hv.exposure_assessment(3.0, kind="hav")
    assert action.zone == "action"
    assert action.exceeds_action
    assert not action.exceeds_limit
    limit = hv.exposure_assessment(5.5, kind="hav")
    assert limit.zone == "limit"
    assert limit.exceeds_limit
    assert action.action_value == hv.HAV_EAV_A8
    assert action.limit_value == hv.HAV_ELV_A8


def test_exposure_assessment_wbv_a8_and_vdv() -> None:
    a = hv.exposure_assessment(0.6, kind="wbv")
    assert a.action_value == hv.WBV_EAV_A8
    assert a.limit_value == hv.WBV_ELV_A8
    assert a.zone == "action"
    v = hv.exposure_assessment(22.0, kind="wbv", metric="vdv")
    assert v.action_value == hv.WBV_EAV_VDV
    assert v.limit_value == hv.WBV_ELV_VDV
    assert v.zone == "limit"


def test_exposure_assessment_invalid() -> None:
    with pytest.raises(ValueError, match="kind"):
        hv.exposure_assessment(1.0, kind="hav", metric="vdv")
    with pytest.raises(ValueError, match="non-negative"):
        hv.exposure_assessment(-1.0, kind="hav")


def test_daily_vibration_exposure_result() -> None:
    result = hv.daily_vibration_exposure(
        [4.6, 6.0, 3.6],
        [2 * 3600.0, 1 * 3600.0, 2 * 3600.0],
        kind="hav",
        labels=["brush-saw", "felling", "stripping"],
    )
    assert result.a8 == pytest.approx(3.6, abs=0.05)
    assert result.labels == ("brush-saw", "felling", "stripping")
    assert result.assessment.zone == "action"
    assert result.partials.shape == (3,)
    assert result.plot  # method exists


def test_daily_vibration_exposure_default_labels_and_mismatch() -> None:
    result = hv.daily_vibration_exposure([2.0], [3600.0], kind="wbv")
    assert result.labels == ("op 1",)
    with pytest.raises(ValueError, match="labels"):
        hv.daily_vibration_exposure(
            [2.0, 3.0], [1.0, 1.0], kind="hav", labels=["only-one"]
        )


def test_weighting_names_complete() -> None:
    assert set(hv.WEIGHTING_NAMES) == {
        "Wb",
        "Wc",
        "Wd",
        "We",
        "Wf",
        "Wh",
        "Wj",
        "Wk",
        "Wm",
    }


# ---------------------------------------------------------------------------
# Remaining validation branches and the .plot() renderers.
# ---------------------------------------------------------------------------
def test_all_nonpositive_frequencies_return_zero_response() -> None:
    resp = hv.frequency_weighting("Wk", [0.0, -1.0])
    assert np.all(resp.magnitude == 0.0)


def test_apply_weighting_rejects_empty_signal() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        hv.apply_weighting([], 1000.0, name="Wk")


def test_running_rms_rejects_empty_signal() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        hv.running_rms([], 1000.0)


def test_vibration_total_value_validates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        hv.vibration_total_value([])
    with pytest.raises(ValueError, match="same length"):
        hv.vibration_total_value([1.0, 2.0], k=[1.4])


def test_daily_exposure_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        hv.daily_exposure(-1.0, 3600.0)


def test_combine_partial_exposures_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        hv.combine_partial_exposures([])


def test_hav_daily_exposure_validates() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        hv.hav_daily_exposure([2.0, 3.0], [3600.0])
    with pytest.raises(ValueError, match="non-negative"):
        hv.hav_daily_exposure([2.0, -3.0], [3600.0, 3600.0])


def test_energy_equivalent_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        hv.energy_equivalent_acceleration([1.0, 2.0], [3600.0])


def test_energy_equivalent_rejects_zero_total_duration() -> None:
    with pytest.raises(ValueError, match="total duration"):
        hv.energy_equivalent_acceleration([1.0, 2.0], [0.0, 0.0])


def test_hav_vwf_lifetime_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="positive"):
        hv.hav_vwf_lifetime_years(0.0)


def test_daily_vibration_exposure_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        hv.daily_vibration_exposure([2.0, 3.0], [3600.0], kind="hav")


def test_weighting_response_plot_returns_axes() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    ax = hv.frequency_weighting("Wk", [1.0, 10.0, 100.0]).plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_weighted_spectrum_plot_returns_axes() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    result = hv.weighted_acceleration([0.5, 0.8, 0.3], [16.0, 31.5, 63.0], "Wk")
    ax = result.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_daily_vibration_exposure_plot_returns_axes() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    result = hv.daily_vibration_exposure([2.5, 3.0], [3600.0, 1800.0], kind="hav")
    ax = result.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")
