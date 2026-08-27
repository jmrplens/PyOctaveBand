#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for NPD event-level interpolation (ECAC Doc 29 §4.2).

Oracles (independent of the implementation): recovery of the tabulated nodes,
hand-computed log-linear (distance) and linear (power) interpolation, monotonic
decrease with distance, and the terminal-segment extrapolation.
"""

from __future__ import annotations

import dataclasses

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

from phonometry.aircraft.airport_noise import (
    FlightSegmentState,
    FlyoverResult,
    NoiseContourResult,
    NpdLevelResult,
    _segment_geometry,
    duration_correction,
    engine_installation_correction,
    event_level,
    lateral_attenuation,
    noise_contour,
    noise_fraction,
    npd_curve,
    npd_level,
    start_of_roll_directivity,
)

# Two powers, four log-spaced distances; levels drop 6 dB per doubling.
_P = [1000.0, 2000.0]
_D = [200.0, 400.0, 800.0, 1600.0]
_L = [[100.0, 94.0, 88.0, 82.0], [110.0, 104.0, 98.0, 92.0]]


def test_recovers_tabulated_nodes() -> None:
    assert npd_level(_P, _D, _L, 1000.0, 200.0)[0] == pytest.approx(100.0)
    assert npd_level(_P, _D, _L, 2000.0, 1600.0)[0] == pytest.approx(92.0)
    assert npd_level(_P, _D, _L, 1000.0, 800.0)[0] == pytest.approx(88.0)


def test_log_linear_distance_interpolation() -> None:
    # Midpoint in log distance between 200 and 400 is sqrt(200*400); the level is
    # the arithmetic mean of the two node levels.
    dm = np.sqrt(200.0 * 400.0)
    assert npd_level(_P, _D, _L, 1000.0, dm)[0] == pytest.approx(97.0)


def test_linear_power_interpolation() -> None:
    # Halfway in power (1500) at a tabulated distance is the mean of the rows.
    assert npd_level(_P, _D, _L, 1500.0, 200.0)[0] == pytest.approx(105.0)
    assert npd_level(_P, _D, _L, 1500.0, 800.0)[0] == pytest.approx(93.0)


def test_monotonic_decrease_with_distance() -> None:
    lv = npd_level(_P, _D, _L, 1500.0, np.array([200.0, 400.0, 800.0, 1600.0]))
    # At the tabulated nodes the P = 1500 row is the mean of the two power rows.
    assert np.allclose(lv, [105.0, 99.0, 93.0, 87.0])
    assert np.all(np.diff(lv) < 0.0)


def test_distance_extrapolation_uses_terminal_slope() -> None:
    # Below 6 dB/doubling on the last segment (800->1600), one more doubling to
    # 3200 m gives 82 - 6 = 76 dB at P = 1000.
    assert npd_level(_P, _D, _L, 1000.0, 3200.0)[0] == pytest.approx(76.0)
    # And inward from 200 m: one halving to 100 m gives 100 + 6 = 106 dB.
    assert npd_level(_P, _D, _L, 1000.0, 100.0)[0] == pytest.approx(106.0)


def test_power_extrapolation() -> None:
    # Beyond the last power (2000): linear on the terminal power segment. At
    # P = 3000, d = 200: 100 + 2*(110-100) = 120 dB.
    assert npd_level(_P, _D, _L, 3000.0, 200.0)[0] == pytest.approx(120.0)
    # Below the first power (1000): linear extrapolation downward. At P = 500,
    # d = 200: 100 + (-0.5)*(110-100) = 95 dB.
    assert npd_level(_P, _D, _L, 500.0, 200.0)[0] == pytest.approx(95.0)


def test_npd_curve_result_and_plot() -> None:
    res = npd_curve(_P, _D, _L, 1500.0)
    assert isinstance(res, NpdLevelResult)
    assert res.distance.shape == res.level.shape
    assert res.table_distances.shape == res.table_levels.shape
    # tabulated levels at P=1500 are the row means
    assert np.allclose(res.table_levels, [105.0, 99.0, 93.0, 87.0])
    assert res.plot() is not None


def test_npd_curve_levels_are_pinned_to_their_own_distances() -> None:
    """Each level array is refused unless it covers the distances beside it.

    The sweep and the tabulated points are independent axes, and a level array
    short of its own distances surfaces only at ``ax.plot``, as "x and y must
    have same first dimension" over two bare shapes: which field was short, and
    which of the two axes it belonged to, are left for the reader to work out.
    """
    res = npd_curve(_P, _D, _L, 1500.0)
    short_sweep = res.level[:-1]
    with pytest.raises(ValueError, match=r"'level'.*one value per query distance"):
        dataclasses.replace(res, level=short_sweep)
    short_table = res.table_levels[:-1]
    with pytest.raises(
        ValueError, match=r"'table_levels'.*one value per tabulated distance"
    ):
        dataclasses.replace(res, table_levels=short_table)


def test_invalid_table_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        npd_level(_P, _D, [[100.0, 94.0]], 1000.0, 200.0)  # wrong levels shape
    with pytest.raises(ValueError, match="increasing"):
        npd_level([2000.0, 1000.0], _D, _L[::-1], 1500.0, 200.0)
    with pytest.raises(ValueError, match="distance"):
        npd_level(_P, _D, _L, 1000.0, -100.0)


# --- Single-event correction terms (ECAC Doc 29 §4.5) ----------------------

# NPD used by the flight-path tests: two powers, SEL and Lmax vs distance.
_NP = [8000.0, 12000.0]
_ND = [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0]
_NSEL = [
    [98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0],
    [102.0, 96.0, 90.0, 84.0, 78.0, 72.0, 66.0],
]
_NMAX = [
    [94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0],
    [98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0],
]
_VREF = 160.0 * 0.514444


def test_lateral_attenuation_limits() -> None:
    assert lateral_attenuation(60.0, 2000.0) == pytest.approx(0.0)  # β >= 50°
    assert lateral_attenuation(-5.0, 2000.0) == pytest.approx(10.857)  # β < 0
    assert lateral_attenuation(0.0, 2000.0) == pytest.approx(10.857)  # ℓ > 914, β = 0
    # Γ(914) = 1.089·(1−e^-2.505) ≈ 1.0, so Λ(0°,914) ≈ Λ(0°) = 10.857.
    assert lateral_attenuation(0.0, 914.0) == pytest.approx(10.857, abs=0.02)


def test_engine_installation_values() -> None:
    assert engine_installation_correction(0.0, "propeller") == pytest.approx(0.0)
    # ΔI(0) = 10·b·log10(a): wing and fuselage.
    assert engine_installation_correction(0.0, "wing") == pytest.approx(
        10.0 * 0.062 * np.log10(0.0039), abs=1e-6
    )
    assert engine_installation_correction(0.0, "fuselage") == pytest.approx(
        10.0 * 0.329 * np.log10(0.1225), abs=1e-6
    )
    # For φ < 0 the correction is clamped to ΔI(0).
    assert engine_installation_correction(-15.0, "wing") == pytest.approx(
        engine_installation_correction(0.0, "wing")
    )


def test_duration_correction() -> None:
    assert duration_correction(_VREF, _VREF) == pytest.approx(0.0)
    assert duration_correction(_VREF, 2.0 * _VREF) == pytest.approx(
        -10.0 * np.log10(2.0)
    )


def test_noise_fraction_limits() -> None:
    # A long symmetric segment captures ~all the energy: ΔF -> 0.
    assert noise_fraction(5000.0, 10000.0, 100.0) == pytest.approx(0.0, abs=1e-3)
    # A half-infinite segment (perpendicular foot at the start) -> half energy.
    assert noise_fraction(0.0, 10000.0, 100.0) == pytest.approx(
        -10.0 * np.log10(2.0), abs=1e-3
    )


def test_impedance_adjustment() -> None:
    from phonometry.aircraft.airport_noise import impedance_adjustment

    # Under the standard atmosphere (15 °C, 101.325 kPa) the adjustment is the
    # ECAC-documented +0.074 dB (Doc 29 Vol 2 §4.2.1).
    assert impedance_adjustment() == pytest.approx(0.074, abs=5e-4)
    assert impedance_adjustment(15.0, 101.325) == pytest.approx(0.074, abs=5e-4)
    # Hotter, lower-pressure air is less dense: negative adjustment.
    assert impedance_adjustment(35.0, 95.0) < 0.0
    with pytest.raises(ValueError, match="pressure"):
        impedance_adjustment(15.0, 0.0)


def test_event_level_long_level_flyover_matches_infinite_path() -> None:
    # A long straight level segment through the CPA reduces to the infinite-path
    # baseline plus the geometry corrections (ΔF -> 0, ΔV = 0 at Vref):
    #   SEL = LE∞(P, dp) + ΔI(β) − Λ(β, ℓ).
    xs = np.linspace(-40000.0, 40000.0, 801)
    path = np.column_stack(
        [
            xs,
            np.zeros_like(xs),
            np.full_like(xs, 300.0),
            np.full_like(xs, 10000.0),
            np.full_like(xs, _VREF),
        ]
    )
    res = event_level(
        path, [0.0, 300.0, 0.0], _NP, _ND, _NSEL, _NMAX, metric="exposure"
    )
    assert isinstance(res, FlyoverResult)
    dp = np.hypot(300.0, 300.0)
    lateral, beta = 300.0, np.degrees(np.arccos(300.0 / dp))
    from phonometry.aircraft.airport_noise import impedance_adjustment

    expected = (
        npd_level(_NP, _ND, _NSEL, 10000.0, dp)[0]
        + impedance_adjustment()
        + engine_installation_correction(beta, "wing")
        - lateral_attenuation(beta, lateral)
    )
    assert res.level == pytest.approx(float(expected), abs=1e-3)


def test_event_level_maximum_metric() -> None:
    xs = np.linspace(-20000.0, 20000.0, 401)
    path = np.column_stack(
        [
            xs,
            np.zeros_like(xs),
            np.full_like(xs, 300.0),
            np.full_like(xs, 10000.0),
            np.full_like(xs, _VREF),
        ]
    )
    res = event_level(path, [0.0, 300.0, 0.0], _NP, _ND, _NSEL, _NMAX, metric="maximum")
    dp = np.hypot(300.0, 300.0)
    lateral, beta = 300.0, np.degrees(np.arccos(300.0 / dp))
    from phonometry.aircraft.airport_noise import impedance_adjustment

    expected = (
        npd_level(_NP, _ND, _NMAX, 10000.0, dp)[0]
        + impedance_adjustment()
        + engine_installation_correction(beta, "wing")
        - lateral_attenuation(beta, lateral)
    )
    assert res.level == pytest.approx(float(expected), abs=1e-3)


def test_event_level_duplicate_waypoint_is_skipped() -> None:
    # A repeated path point makes a zero-length segment; it must be dropped
    # cleanly (no NaN/divide warning) and not change the result.
    base = [
        [-20000.0, 0.0, 300.0, 10000.0, _VREF],
        [0.0, 0.0, 300.0, 10000.0, _VREF],
        [20000.0, 0.0, 300.0, 10000.0, _VREF],
    ]
    dup = [base[0], base[1], base[1], base[2]]  # duplicate the middle point
    obs = [0.0, 300.0, 0.0]
    r_base = event_level(base, obs, _NP, _ND, _NSEL, _NMAX).level
    with np.errstate(all="raise"):  # any NaN/divide would raise here
        r_dup = event_level(dup, obs, _NP, _ND, _NSEL, _NMAX).level
    assert r_dup == pytest.approx(r_base, abs=1e-9)


def test_flyover_plot() -> None:
    xs = np.linspace(-20000.0, 20000.0, 201)
    path = np.column_stack(
        [
            xs,
            np.zeros_like(xs),
            np.full_like(xs, 300.0),
            np.full_like(xs, 10000.0),
            np.full_like(xs, _VREF),
        ]
    )
    res = event_level(path, [0.0, 300.0, 0.0], _NP, _ND, _NSEL, _NMAX)
    assert isinstance(res, FlyoverResult)
    assert res.segment_levels.size == 200
    assert res.plot() is not None


def test_flyover_arrays_are_refused_a_second_axis() -> None:
    """Neither event array may be built with an axis it should not have.

    A two-dimensional ``segment_levels`` reaches ``ax.bar`` as heights against
    a flat bar index and fails there, in Matplotlib's words rather than the
    library's; a two-dimensional ``observer`` is read by nothing at all and
    would reach the caller unnoticed.
    """
    xs = np.linspace(-20000.0, 20000.0, 21)
    path = np.column_stack(
        [
            xs,
            np.zeros_like(xs),
            np.full_like(xs, 300.0),
            np.full_like(xs, 10000.0),
            np.full_like(xs, _VREF),
        ]
    )
    res = event_level(path, [0.0, 300.0, 0.0], _NP, _ND, _NSEL, _NMAX)
    stacked_levels = np.column_stack([res.segment_levels] * 2)
    with pytest.raises(ValueError, match=r"'segment_levels' must have one axis"):
        dataclasses.replace(res, segment_levels=stacked_levels)
    stacked_observer = np.column_stack([res.observer] * 2)
    with pytest.raises(ValueError, match=r"'observer' must have one axis"):
        dataclasses.replace(res, observer=stacked_observer)


def test_flyover_metric_outside_the_two_the_type_states_is_refused() -> None:
    """The figure asks one question of ``metric``, and everything else is LAmax.

    Left unpinned the tag does not fail at all: the y-axis and the total line
    of the rendered figure are labelled as a maximum level the result never
    claimed to hold.
    """
    xs = np.linspace(-20000.0, 20000.0, 21)
    path = np.column_stack(
        [
            xs,
            np.zeros_like(xs),
            np.full_like(xs, 300.0),
            np.full_like(xs, 10000.0),
            np.full_like(xs, _VREF),
        ]
    )
    res = event_level(path, [0.0, 300.0, 0.0], _NP, _ND, _NSEL, _NMAX)
    with pytest.raises(ValueError, match="'metric' must be one of"):
        dataclasses.replace(res, metric="bogus")


def test_event_level_farther_receiver_is_quieter() -> None:
    xs = np.linspace(-20000.0, 20000.0, 401)
    path = np.column_stack(
        [
            xs,
            np.zeros_like(xs),
            np.full_like(xs, 300.0),
            np.full_like(xs, 10000.0),
            np.full_like(xs, _VREF),
        ]
    )
    near = event_level(path, [0.0, 300.0, 0.0], _NP, _ND, _NSEL, _NMAX).level
    far = event_level(path, [0.0, 3000.0, 0.0], _NP, _ND, _NSEL, _NMAX).level
    assert far < near


# --- Reference workbook oracle (ECAC Doc 29 5th ed. Vol 3 Part 1) ----------
#
# Segment geometry and per-segment corrections extracted from the official ECAC
# Doc 29 reference workbook (sheet B-2_Segment_Results, case JETFAC, receptor
# R02), independent of this implementation. Coordinates are in feet, the
# receptor sits at (0, 656.168, 0) ft. Each row lists the segment endpoints and
# the workbook's β (deg), φ (deg), lateral attenuation Λ (dB) and engine
# installation ΔI (dB, fuselage-mounted). Segment 8 is level (β = φ); segment 1
# climbs, so β and φ differ.
_WB_OBS_FT = (0.0, 656.1679790026246, 0.0)
_WB_SEGMENTS = [
    # S1 (ft),                          S2 (ft),                        β,      φ,      Λ,      ΔI
    (
        (-81363.2829, -328077.7538, 18311.6034),
        (-81363.2829, -76348.2804, 6000.0),
        4.2225708673,
        1.5708068190,
        6.3768594165,
        -2.9923618950,
    ),
    (
        (-80116.5875, -13598.8229, 3000.0),
        (-78594.3067, -10334.4492, 3000.0),
        2.5797296415,
        2.5797296415,
        7.8166038498,
        -2.9794464291,
    ),
]


# Noise-fraction ΔF from the same workbook rows: (q, length, LE∞−Lmax, ΔF), all
# in feet. dλ = d0·10^((LE∞−Lmax)/10) with d0 = (2/π)·Vref·t0 in feet.
_WB_NOISE_FRACTION = [
    (329236.0, 252030.4, 45.86 - 12.85, -5.787),
    (28482.0, 3607.3, 46.15 - 13.32, -21.635),
    (35032.1, 546.2, 46.97 - 14.58, -29.447),
]


# Start-of-roll directivity ΔSOR from the workbook (sheet B-2, cases JETFDC and
# PROPDC): (ψ°, dSOR[m], ΔSOR, engine). ψ = arccos(q/dSOR), all dSOR < 762 m so
# no distance scaling. ECAC Doc 29 Vol 3 Part 1 reference workbook.
_WB_SOR = [
    (96.031600, 201.1120, -0.80450, "jet"),
    (112.889545, 217.0934, 0.31961, "jet"),
    (133.519719, 275.8080, 0.00558, "jet"),
    (101.133998, 203.8352, -0.98974, "turboprop"),
    (128.182381, 254.4361, 1.09434, "turboprop"),
    (150.518805, 406.3875, -7.09359, "turboprop"),
]


@pytest.mark.parametrize(("psi", "dsor", "sor_ref", "engine"), _WB_SOR)
def test_reference_workbook_start_of_roll(
    psi: float,
    dsor: float,
    sor_ref: float,
    engine: str,
) -> None:
    assert start_of_roll_directivity(psi, dsor, engine) == pytest.approx(
        sor_ref, abs=0.01
    )


def test_start_of_roll_directivity_properties() -> None:
    # Ahead of / abeam the aircraft (ψ < 90°): no rearward directivity.
    assert start_of_roll_directivity(45.0, 500.0, "jet") == 0.0
    assert start_of_roll_directivity(89.9, 500.0, "turboprop") == 0.0
    # Beyond the 762 m normalising distance the correction scales by dSOR,0/dSOR.
    near = start_of_roll_directivity(180.0, 762.0, "jet")
    assert start_of_roll_directivity(180.0, 1524.0, "jet") == pytest.approx(near / 2.0)
    # "propeller"/"prop" are accepted aliases for the turboprop curve.
    assert start_of_roll_directivity(120.0, 300.0, "prop") == pytest.approx(
        start_of_roll_directivity(120.0, 300.0, "turboprop")
    )
    with pytest.raises(ValueError, match="engine"):
        start_of_roll_directivity(120.0, 300.0, "rocket")
    with pytest.raises(ValueError, match="distance_m"):
        start_of_roll_directivity(120.0, -1.0, "jet")


def test_event_level_ground_roll_applies_directivity() -> None:
    # A takeoff ground-roll segment along +x from the origin; a receiver behind
    # the start of roll (negative x) must gain the ΔSOR directivity relative to
    # the same geometry without the ground-roll flag.
    path = [[0.0, 0.0, 0.0, 12000.0, 40.0], [1500.0, 0.0, 0.0, 12000.0, 60.0]]
    obs = [-400.0, 150.0, 0.0]
    plain = event_level(path, obs, _NP, _ND, _NSEL, _NMAX).level
    rolled = event_level(
        path,
        obs,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        segments=FlightSegmentState(ground_roll=[True]),
    ).level
    # Behind SOR, ψ is obtuse; the two differ by exactly ΔSOR plus the reduced
    # noise-fraction switch, so the ground-roll level is not equal to the plain one.
    assert rolled != pytest.approx(plain)
    # Ahead of the segment no ΔSOR applies, but the runway segment still uses
    # the Eq. 4-13b average speed, so the flagged level differs from the plain
    # airborne treatment exactly by the duration-correction delta.
    ahead = [2000.0, 150.0, 0.0]  # q > λ: truly ahead of the segment
    from phonometry.aircraft.airport_noise import duration_correction as _dc

    plain_ahead = event_level(path, ahead, _NP, _ND, _NSEL, _NMAX).level
    flagged_ahead = event_level(
        path,
        ahead,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        segments=FlightSegmentState(ground_roll=[True]),
    ).level
    vref = 160.0 * 0.514444
    expected_delta = _dc(vref, 0.5 * (40.0 + 60.0)) - _dc(vref, 60.0)
    assert flagged_ahead - plain_ahead == pytest.approx(expected_delta, abs=1e-9)
    mismatched_roll = FlightSegmentState(ground_roll=[True, False])
    with pytest.raises(ValueError, match="ground_roll"):
        event_level(path, obs, _NP, _ND, _NSEL, _NMAX, segments=mismatched_roll)
    # The directivity is also applied on the maximum metric (Eq. 4-9a) and with
    # a propeller mounting (turboprop ΔSOR curve).
    max_plain = event_level(path, obs, _NP, _ND, _NSEL, _NMAX, metric="maximum").level
    max_roll = event_level(
        path,
        obs,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        metric="maximum",
        segments=FlightSegmentState(ground_roll=[True]),
    ).level
    assert max_roll != pytest.approx(max_plain)
    prop_roll = event_level(
        path,
        obs,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        mounting="propeller",
        segments=FlightSegmentState(ground_roll=[True]),
    ).level
    jet_roll = event_level(
        path,
        obs,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        mounting="wing",
        segments=FlightSegmentState(ground_roll=[True]),
    ).level
    assert prop_roll != pytest.approx(jet_roll)  # different SOR curves


def test_segment_state_fields_must_cover_the_same_segments() -> None:
    """A bundle whose own fields disagree is refused where it is put together.

    The event entry points compare each field against the path they were handed
    and only there, one at a time, so an inconsistent bundle is blamed on
    whichever field happens to be validated first, and blamed afresh for every
    path it is reused with. The count is of values, not of shape: a column of
    one value per segment is ravelled by the entry points and accepted here.
    """
    with pytest.raises(ValueError, match=r"'bank'.*one value per segment"):
        FlightSegmentState(ground_roll=[True, False, False], bank=[0.0, 8.0])
    column = FlightSegmentState(bank=np.array([[0.0], [4.0], [8.0]]))
    assert np.size(column.bank) == 3


@pytest.mark.parametrize(("q", "length", "le_minus_lmax", "df_ref"), _WB_NOISE_FRACTION)
def test_reference_workbook_noise_fraction(
    q: float,
    length: float,
    le_minus_lmax: float,
    df_ref: float,
) -> None:
    d0_ft = (2.0 / np.pi) * 270.05 * 1.0  # Vref = 160 kn = 270.05 ft/s, t0 = 1 s
    d_lambda = d0_ft * 10.0 ** (le_minus_lmax / 10.0)
    # abs=0.01: LE∞−Lmax is hardcoded to the workbook's 2-decimal precision.
    assert noise_fraction(q, length, d_lambda) == pytest.approx(df_ref, abs=1e-2)


@pytest.mark.parametrize(
    ("s1", "s2", "beta_ref", "phi_ref", "lam_ref", "di_ref"), _WB_SEGMENTS
)
def test_reference_workbook_segment_terms(
    s1: tuple[float, float, float],
    s2: tuple[float, float, float],
    beta_ref: float,
    phi_ref: float,
    lam_ref: float,
    di_ref: float,
) -> None:
    ft = 0.3048
    _, _, _, _, beta, phi, lateral = _segment_geometry(
        np.array(s1), np.array(s2), np.array(_WB_OBS_FT)
    )
    # β and φ match the reference to <0.01° (the small residual on the climbing
    # segment is the workbook's equivalent-level-path height convention).
    assert beta == pytest.approx(beta_ref, abs=0.01)
    assert phi == pytest.approx(phi_ref, abs=0.01)
    # The physical corrections reproduce the reference to <0.01 dB. Λ's Γ term
    # needs the lateral displacement in metres.
    assert lateral_attenuation(beta, lateral * ft) == pytest.approx(lam_ref, abs=0.01)
    assert engine_installation_correction(phi, "fuselage") == pytest.approx(
        di_ref, abs=0.01
    )


def test_noise_contour_shape_and_plot() -> None:
    xs = np.linspace(0.0, 15000.0, 40)
    path = np.column_stack(
        [
            xs,
            np.zeros_like(xs),
            np.clip(xs * 0.1, 0.0, 2000.0),
            np.full_like(xs, 10000.0),
            np.full_like(xs, _VREF),
        ]
    )
    res = noise_contour(
        path,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        x=np.linspace(-2000.0, 16000.0, 24),
        y=np.linspace(-4000.0, 4000.0, 20),
    )
    assert isinstance(res, NoiseContourResult)
    assert res.level.shape == (20, 24)
    # Highest level is near the track (y = 0); much lower far to the side.
    iy0 = int(np.argmin(np.abs(res.y)))
    assert np.max(res.level[iy0]) > np.max(res.level[0])
    assert res.plot() is not None


def test_noise_contour_grid_is_pinned_to_both_axes() -> None:
    """A grid that does not cover the x and y beside it is refused when built.

    ``contourf`` does raise on a grid off by a row or a column, but with a
    ``TypeError`` that compares a bare length with a count of rows and names
    neither the result nor the field; a grid flattened to one axis raises there
    about a dimension, equally anonymously.
    """
    xs = np.linspace(0.0, 15000.0, 12)
    path = np.column_stack(
        [
            xs,
            np.zeros_like(xs),
            np.clip(xs * 0.1, 0.0, 2000.0),
            np.full_like(xs, 10000.0),
            np.full_like(xs, _VREF),
        ]
    )
    res = noise_contour(
        path,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        x=np.linspace(-2000.0, 16000.0, 7),
        y=np.linspace(-4000.0, 4000.0, 5),
    )
    short_rows = res.level[:-1]
    with pytest.raises(ValueError, match=r"'level'.*one value per grid row"):
        dataclasses.replace(res, level=short_rows)
    short_columns = res.level[:, :-1]
    with pytest.raises(
        ValueError, match=r"'level \(axis 1\)'.*one value per grid column"
    ):
        dataclasses.replace(res, level=short_columns)
    flattened = res.level.ravel()
    with pytest.raises(ValueError, match=r"'level' must have 2 axes"):
        dataclasses.replace(res, level=flattened)


def test_noise_contour_metric_outside_the_two_the_type_states_is_refused() -> None:
    """The colorbar reads the tag one-sidedly, with LAmax as its fallback.

    An unknown tag labels a rendered contour map with a metric the grid was
    never computed in, and nothing raises.
    """
    xs = np.linspace(0.0, 15000.0, 12)
    path = np.column_stack(
        [
            xs,
            np.zeros_like(xs),
            np.clip(xs * 0.1, 0.0, 2000.0),
            np.full_like(xs, 10000.0),
            np.full_like(xs, _VREF),
        ]
    )
    res = noise_contour(
        path,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        x=np.linspace(-2000.0, 16000.0, 7),
        y=np.linspace(-4000.0, 4000.0, 5),
    )
    with pytest.raises(ValueError, match="'metric' must be one of"):
        dataclasses.replace(res, metric="LAeq")


def test_event_level_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="path"):
        event_level(
            [[0.0, 0.0, 0.0, 1.0, 1.0]], [0.0, 0.0, 0.0], _NP, _ND, _NSEL, _NMAX
        )
    good = [[0.0, 0.0, 300.0, 10000.0, _VREF], [1000.0, 0.0, 300.0, 10000.0, _VREF]]
    with pytest.raises(ValueError, match="metric"):
        event_level(good, [0.0, 100.0, 0.0], _NP, _ND, _NSEL, _NMAX, metric="Lden")
    with pytest.raises(ValueError, match="lateral_m"):
        lateral_attenuation(30.0, -1.0)
    with pytest.raises(ValueError, match="mounting"):
        engine_installation_correction(20.0, "tail")
    # Non-finite path, negative power and non-positive speed are rejected.
    with pytest.raises(ValueError, match="finite"):
        event_level(
            [[0.0, 0.0, np.inf, 1e4, _VREF], good[1]],
            [0.0, 100.0, 0.0],
            _NP,
            _ND,
            _NSEL,
            _NMAX,
        )
    with pytest.raises(ValueError, match="power"):
        event_level(
            [[0.0, 0.0, 300.0, -1.0, _VREF], good[1]],
            [0.0, 100.0, 0.0],
            _NP,
            _ND,
            _NSEL,
            _NMAX,
        )
    with pytest.raises(ValueError, match="speed"):
        event_level(
            [[0.0, 0.0, 300.0, 1e4, 0.0], good[1]],
            [0.0, 100.0, 0.0],
            _NP,
            _ND,
            _NSEL,
            _NMAX,
        )


def test_impedance_adjustment_rejects_absolute_zero() -> None:
    from phonometry.aircraft.airport_noise import impedance_adjustment

    with pytest.raises(ValueError, match="absolute zero"):
        impedance_adjustment(-300.0, 101.325)


# --------------------------------------------------------------------------- #
# Doc 29 Vol 3 Part 1 reference workbook: seven-receptor assembly oracle
# --------------------------------------------------------------------------- #
from doc29_workbook_data import B1, SEGMENTS


@pytest.mark.parametrize("case", sorted(B1))
def test_workbook_segment_assembly_reproduces_event_sel(case: tuple[str, str]) -> None:
    # Eq. 4-9b assembly from the workbook's own per-segment terms must
    # reproduce each segment SEL and the B-1 event total (Eq. 4-11), for all
    # seven branch-covering receptor events.
    totals = []
    for row in SEGMENTS[case]:
        (_, _, _, di, lam, base, dv, df, sor, imp, seg_sel) = row
        assembled = base + dv + di - lam + df + imp + sor
        assert assembled == pytest.approx(seg_sel, abs=2e-3)
        totals.append(seg_sel)
    total = 10.0 * np.log10(np.sum(10.0 ** (np.asarray(totals) / 10.0)))
    assert total == pytest.approx(B1[case], abs=6e-3)


def test_workbook_behind_sor_geometry_matches_h1_fix() -> None:
    # JETFDC/R03 seg 1 (behind the start of roll on the centreline): the
    # nearest-end geometry gives beta1 = asin(z1/d1) ~ 0.115 deg and
    # l = OC1 = d1, hence Lambda = 8.689 dB and Delta_I(fuselage) = -3.000 dB
    # (workbook sheet B-2). The old track-perpendicular geometry returned
    # lateral = 0 and beta = 90 deg, zeroing both terms.
    d1_ft, beta_ref = 1640.4, 0.1146
    ft = 0.3048
    z1 = d1_ft * np.sin(np.radians(beta_ref))
    oc1 = np.sqrt(d1_ft**2 - z1**2)
    assert lateral_attenuation(beta_ref, oc1 * ft) == pytest.approx(8.689, abs=2e-3)
    assert engine_installation_correction(beta_ref, "fuselage") == pytest.approx(
        -3.000, abs=2e-3
    )
    # End-to-end: a centreline receptor behind a takeoff-roll segment now gets
    # a materially attenuated level instead of the unattenuated one.
    path = [[0.0, 0.0, 0.5, 12000.0, 0.0], [1500.0, 0.0, 0.5, 12000.0, 40.0]]
    behind = [-500.0, 0.0, 0.0]
    lvl = event_level(
        path,
        behind,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        mounting="fuselage",
        segments=FlightSegmentState(ground_roll=[True]),
    ).level
    assert np.isfinite(lvl)
    seg = event_level(
        path,
        behind,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        mounting="fuselage",
        segments=FlightSegmentState(ground_roll=[True]),
    ).segment_levels
    assert seg.size == 1


def test_landing_roll_ahead_branch() -> None:
    # Ahead of a landing rollout: ds baseline, reduced fraction Eq. 4-21b,
    # nearest-end geometry and no SOR. The flagged result must differ from the
    # plain treatment and must NOT include a SOR term (identical for jet and
    # turboprop mounting up to the Delta_I difference).
    path = [[0.0, 0.0, 0.5, 8000.0, 70.0], [1200.0, 0.0, 0.5, 8000.0, 30.0]]
    ahead = [3000.0, 400.0, 0.0]
    plain = event_level(path, ahead, _NP, _ND, _NSEL, _NMAX).level
    landing = event_level(
        path,
        ahead,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        segments=FlightSegmentState(landing_roll=[True]),
    ).level
    assert landing != pytest.approx(plain)
    # Zero-speed start is legitimate on runway segments (Eq. 4-13b mean speed).
    v0 = [
        [0.0, 0.0, 0.5, 12000.0, 0.0],
        [800.0, 0.0, 0.5, 12000.0, 30.0],
        [4000.0, 0.0, 250.0, 12000.0, 80.0],
    ]
    lvl = event_level(
        v0,
        [2000.0, 600.0, 0.0],
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        segments=FlightSegmentState(ground_roll=[True, False]),
    ).level
    assert np.isfinite(lvl)


def test_npd_floor_observer_on_track() -> None:
    # M4: a receiver exactly under the (ground-level) path no longer crashes;
    # the NPD lookup clamps to the recommended 30 m floor (Doc 29 section 4.2).
    path = [[0.0, 0.0, 0.0, 10000.0, _VREF], [2000.0, 0.0, 0.0, 10000.0, _VREF]]
    lvl = event_level(path, [1000.0, 0.0, 0.0], _NP, _ND, _NSEL, _NMAX).level
    assert np.isfinite(lvl)


def test_bank_angle_side_asymmetry() -> None:
    # L2: with bank, starboard and port receivers see different Delta_I
    # (phi = beta +/- epsilon per section 4.5.2); without bank they are equal.
    path = [[0.0, 0.0, 300.0, 10000.0, _VREF], [3000.0, 0.0, 400.0, 10000.0, _VREF]]
    banked = FlightSegmentState(bank=[10.0])
    port = event_level(
        path, [1500.0, 500.0, 0.0], _NP, _ND, _NSEL, _NMAX, segments=banked
    ).level
    stbd = event_level(
        path, [1500.0, -500.0, 0.0], _NP, _ND, _NSEL, _NMAX, segments=banked
    ).level
    same_p = event_level(path, [1500.0, 500.0, 0.0], _NP, _ND, _NSEL, _NMAX).level
    same_s = event_level(path, [1500.0, -500.0, 0.0], _NP, _ND, _NSEL, _NMAX).level
    assert port != pytest.approx(stbd)
    assert same_p == pytest.approx(same_s)
    mismatched_bank = FlightSegmentState(bank=[1.0, 2.0])
    with pytest.raises(ValueError, match="bank"):
        event_level(
            path, [0.0, 500.0, 0.0], _NP, _ND, _NSEL, _NMAX, segments=mismatched_bank
        )


def test_bank_angle_sign_follows_doc29_sides() -> None:
    # Doc 29 section 4.5.2: phi = beta + eps for observers to starboard (right
    # of the flight direction) and phi = beta - eps for observers to port.
    # Level segment along +x at 300 m altitude, 400 m lateral offset:
    # beta = arctan(300/400) = 36.87 deg; with eps = 20 deg the starboard
    # receiver sees phi = 56.87 deg and the port receiver phi = 16.87 deg.
    s1 = np.array([0.0, 0.0, 300.0])
    s2 = np.array([1000.0, 0.0, 300.0])
    starboard = np.array([500.0, -400.0, 0.0])  # right of the +x heading
    port = np.array([500.0, 400.0, 0.0])  # left of the +x heading
    *_, beta_s, phi_s, _ = _segment_geometry(s1, s2, starboard, bank_deg=20.0)
    *_, beta_p, phi_p, _ = _segment_geometry(s1, s2, port, bank_deg=20.0)
    assert beta_s == pytest.approx(36.87, abs=0.01)
    assert beta_p == pytest.approx(36.87, abs=0.01)
    assert phi_s == pytest.approx(56.87, abs=0.01)  # beta + eps (starboard)
    assert phi_p == pytest.approx(16.87, abs=0.01)  # beta - eps (port)
    # eps = 0 regression: phi falls back to the equivalent elevation angle.
    *_, beta0, phi0, _ = _segment_geometry(s1, s2, starboard, bank_deg=0.0)
    assert phi0 == pytest.approx(beta0)


def test_banked_segment_installation_favours_starboard() -> None:
    # Integration: eps > 0 (left turn, starboard wing up) increases phi on the
    # starboard side, so Delta_I recovers towards 0 dB there and the starboard
    # receiver reads a higher level than the mirrored port receiver, for both
    # wing- and fuselage-mounted engines (the inverted sign mirrored banked
    # contours). Propeller aircraft have no installation effect, so no
    # asymmetry appears.
    path = [[0.0, 0.0, 300.0, 10000.0, _VREF], [3000.0, 0.0, 300.0, 10000.0, _VREF]]
    banked15 = FlightSegmentState(bank=[15.0])
    for mounting in ("wing", "fuselage"):
        stbd = event_level(
            path,
            [1500.0, -500.0, 0.0],
            _NP,
            _ND,
            _NSEL,
            _NMAX,
            mounting=mounting,
            segments=banked15,
        ).level
        port = event_level(
            path,
            [1500.0, 500.0, 0.0],
            _NP,
            _ND,
            _NSEL,
            _NMAX,
            mounting=mounting,
            segments=banked15,
        ).level
        assert stbd > port, mounting
    stbd = event_level(
        path,
        [1500.0, -500.0, 0.0],
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        mounting="propeller",
        segments=banked15,
    ).level
    port = event_level(
        path,
        [1500.0, 500.0, 0.0],
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        mounting="propeller",
        segments=banked15,
    ).level
    assert stbd == pytest.approx(port)


def test_noise_contour_accepts_bank() -> None:
    # API parity with event_level: bank threads through to the contour grid
    # and produces the same port/starboard asymmetry.
    path = [[0.0, 0.0, 300.0, 10000.0, _VREF], [3000.0, 0.0, 400.0, 10000.0, _VREF]]
    kw = {"x": [1400.0, 1600.0], "y": [-500.0, 500.0]}
    banked = noise_contour(
        path, _NP, _ND, _NSEL, _NMAX, segments=FlightSegmentState(bank=[10.0]), **kw
    )
    level = noise_contour(path, _NP, _ND, _NSEL, _NMAX, **kw)
    assert banked.level[0, 0] != pytest.approx(banked.level[1, 0])
    assert level.level[0, 0] == pytest.approx(level.level[1, 0])
    mismatched_bank = FlightSegmentState(bank=[1.0, 2.0])
    with pytest.raises(ValueError, match="bank"):
        noise_contour(path, _NP, _ND, _NSEL, _NMAX, segments=mismatched_bank, **kw)


def test_contour_grid_matches_scalar_event_level() -> None:
    # The vectorised grid kernel must reproduce the per-point scalar path to
    # machine precision for every feature combination: takeoff roll (SOR,
    # Eq. 4-21a), landing rollout (Eq. 4-21b), bank, both metrics and both
    # engine mountings.
    path = [
        [-500.0, 0.0, 0.0, 8000.0, 0.0],  # takeoff ground roll
        [800.0, 0.0, 0.0, 10000.0, 80.0],  # rotation
        [2500.0, 100.0, 300.0, 10000.0, _VREF],  # climb, slight turn
        [4500.0, 400.0, 700.0, 9000.0, _VREF],
    ]
    segments = FlightSegmentState(
        ground_roll=[True, False, False],
        landing_roll=[False, False, False],
        bank=[0.0, 0.0, 8.0],
    )
    x = np.linspace(-2000.0, 5000.0, 9)
    y = np.linspace(-1500.0, 1500.0, 7)
    for metric in ("exposure", "maximum"):
        for mounting in ("wing", "propeller"):
            res = noise_contour(
                path,
                _NP,
                _ND,
                _NSEL,
                _NMAX,
                x=x,
                y=y,
                metric=metric,
                mounting=mounting,
                segments=segments,
            )
            for iy, yv in enumerate(y):
                for ix, xv in enumerate(x):
                    ref = event_level(
                        path,
                        [xv, yv, 0.0],
                        _NP,
                        _ND,
                        _NSEL,
                        _NMAX,
                        metric=metric,
                        mounting=mounting,
                        segments=segments,
                    ).level
                    assert res.level[iy, ix] == pytest.approx(ref, abs=1e-9), (
                        metric,
                        mounting,
                        xv,
                        yv,
                    )
    # Landing rollout branch, separately (ahead-of-rollout geometry).
    land = [
        [0.0, 0.0, 300.0, 9000.0, _VREF],
        [2000.0, 0.0, 0.0, 8000.0, 70.0],
        [3200.0, 0.0, 0.0, 4000.0, 0.0],
    ]
    lmask = FlightSegmentState(landing_roll=[False, True])
    res = noise_contour(
        land,
        _NP,
        _ND,
        _NSEL,
        _NMAX,
        x=[1000.0, 4000.0],
        y=[-200.0, 200.0],
        segments=lmask,
    )
    for iy, yv in enumerate([-200.0, 200.0]):
        for ix, xv in enumerate([1000.0, 4000.0]):
            ref = event_level(
                land, [xv, yv, 0.0], _NP, _ND, _NSEL, _NMAX, segments=lmask
            ).level
            assert res.level[iy, ix] == pytest.approx(ref, abs=1e-9)
