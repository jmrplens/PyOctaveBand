#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the baffled circular piston (Beranek & Mellow 2e, §4.19 / §13.7).

Oracles are the exact closed forms (``R1 = 1 - 2 J1(x)/x``, ``X1 = 2 H1(x)/x``
with ``x = 2ka``), their low- and high-frequency limits (Eqs. (13.117),
(13.118), radiation mass Eq. (4.151)), the first directivity null at the first
zero of ``J1`` (3.8317), and the half-space baffle directivity index of 3.01 dB
at ``ka -> 0``. Point values were cross-checked against scipy in the source
extraction.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import special

from phonometry import electroacoustics
from phonometry.electroacoustics.piston import J1_FIRST_ZERO


def test_resistance_reactance_closed_form() -> None:
    x = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    assert np.allclose(
        electroacoustics.piston_resistance(x), 1.0 - 2.0 * special.j1(x) / x
    )
    assert np.allclose(
        electroacoustics.piston_reactance(x), 2.0 * special.struve(1, x) / x
    )


def test_digitized_reference_points() -> None:
    # Points verified from Eqs. (13.117)/(13.118) in the source extraction.
    ref = {
        0.1: (0.001249, 0.042413),
        1.0: (0.119899, 0.396915),
        3.0: (0.773961, 0.680073),
        5.0: (1.131032, 0.323125),
        10.0: (0.991305, 0.178366),
    }
    for x, (r1, x1) in ref.items():
        assert electroacoustics.piston_resistance(x) == pytest.approx(r1, abs=1e-5)
        assert electroacoustics.piston_reactance(x) == pytest.approx(x1, abs=1e-5)


def test_low_frequency_limits() -> None:
    # R1 -> (ka)^2/2 = x^2/8 and X1 -> (8/3pi) ka = (4/3pi) x as x -> 0.
    x = 1e-3
    assert electroacoustics.piston_resistance(x) == pytest.approx(x**2 / 8.0, rel=1e-4)
    assert electroacoustics.piston_reactance(x) == pytest.approx(
        4.0 / (3.0 * math.pi) * x, rel=1e-4
    )


def test_high_frequency_limit_resistive() -> None:
    # R1 -> 1 and X1 -> 0 at large x: Z_r -> rho c S (purely resistive).
    assert electroacoustics.piston_resistance(80.0) == pytest.approx(1.0, abs=0.02)
    assert abs(electroacoustics.piston_reactance(80.0)) < 0.05


def test_zero_argument_regular() -> None:
    assert electroacoustics.piston_resistance(0.0) == 0.0
    assert electroacoustics.piston_reactance(0.0) == 0.0
    assert not np.any(
        np.isnan(electroacoustics.piston_resistance(np.array([0.0, 1.0])))
    )


def test_directivity_onaxis_and_first_null() -> None:
    assert electroacoustics.piston_directivity(5.0, 0.0) == pytest.approx(1.0)
    ka = 6.0
    theta_null = math.asin(J1_FIRST_ZERO / ka)
    assert abs(electroacoustics.piston_directivity(ka, theta_null)) < 1e-9
    # No null exists when ka < first zero of J1.
    theta = np.linspace(0.0, math.pi / 2, 200)
    assert np.min(np.abs(electroacoustics.piston_directivity(3.0, theta))) > 0.1


def test_radiation_mass_coefficient() -> None:
    a, rho = 0.12, 1.206
    res = electroacoustics.radiating_piston(a, [100.0], density=rho)
    assert res.radiation_mass == pytest.approx(8.0 * rho * a**3 / 3.0)


def test_radiation_reactance_tends_to_mass_times_omega() -> None:
    # At low ka the mechanical reactance rho c S X1 = omega M_r.
    a = 0.05
    res = electroacoustics.radiating_piston(
        a, [20.0], density=1.206, speed_of_sound=343.0
    )
    omega = 2.0 * math.pi * 20.0
    assert res.radiation_reactance[0] == pytest.approx(
        omega * res.radiation_mass, rel=2e-3
    )


def test_directivity_index_half_space_limit() -> None:
    # DI -> 10 log10(2) = 3.01 dB as ka -> 0 (radiation into the half-space).
    res = electroacoustics.radiating_piston(0.01, [1.0])
    assert res.directivity_index[0] == pytest.approx(3.0103, abs=1e-3)


def test_directivity_index_high_ka() -> None:
    # DI -> 20 log10(ka) at high ka (Q = (ka)^2).
    a, f, c = 1.0, 5000.0, 343.0
    res = electroacoustics.radiating_piston(a, [f], speed_of_sound=c)
    ka = 2.0 * math.pi * f * a / c
    assert res.directivity_index[0] == pytest.approx(20.0 * math.log10(ka), abs=0.3)


def test_result_shapes_and_plot() -> None:
    res = electroacoustics.radiating_piston(
        0.1, [100.0, 1000.0], angles=[0.0, 0.3, 0.6]
    )
    assert isinstance(res, electroacoustics.RadiatingPistonResult)
    assert res.directivity is not None
    assert res.directivity.shape == (2, 3)
    assert res.ka.shape == (2,)
    import matplotlib as mpl

    mpl.use("Agg")
    ax = res.plot()
    assert ax.get_ylabel()


def test_a_directivity_index_column_of_another_length_is_refused() -> None:
    """The per-frequency quantities are one table, and the data sheet reads it
    by position.

    ``loudspeaker_characteristics`` locates the polar frequency it was asked
    for in ``frequencies`` alone, then indexes ``directivity_index`` and
    ``directivity`` with that position; neither length is checked anywhere
    downstream. A directivity index column one short slides every entry up by
    one, and the IEC 60268-5 sheet prints the index of the next frequency under
    the stated one: 15.19 dB at 800 Hz for the piston below, whose index there
    is 4.56 dB.
    """
    result = electroacoustics.radiating_piston(
        0.1,
        [200.0, 800.0, 3200.0, 6400.0],
        angles=np.radians(np.linspace(-90.0, 90.0, 37)),
    )
    one_frequency_short = result.directivity_index[1:]
    with pytest.raises(ValueError, match=r"'directivity_index' \(3\).*per frequency"):
        dataclasses.replace(result, directivity_index=one_frequency_short)


def test_directivity_pattern_result_and_properties() -> None:
    # Default grid is 361 points over the front hemisphere -90 deg .. +90 deg.
    res = electroacoustics.piston_directivity_pattern([3.0, 8.0])
    assert res.angles.size == 361
    assert isinstance(res, electroacoustics.PistonDirectivity)
    assert res.ka.shape == (2,)
    assert res.directivity.shape == (2, res.angles.size)
    assert res.directivity_db.shape == res.directivity.shape
    # D = 1 (0 dB) on axis (theta = 0) for every ka.
    i0 = int(np.argmin(np.abs(res.angles)))
    assert np.allclose(res.directivity[:, i0], 1.0)
    assert np.allclose(res.directivity_db[:, i0], 0.0)
    # dB echoes the linear directivity: 20 log10 |D|.
    assert np.allclose(
        res.directivity_db, 20.0 * np.log10(np.abs(res.directivity)), atol=1e-9
    )


def test_directivity_pattern_first_null() -> None:
    # First null at ka sin(theta) = 3.8317 (the first zero of J1); it exists
    # only once ka > 3.8317. Sample the exact null angle for ka = 6.
    ka = 6.0
    theta_null = math.asin(J1_FIRST_ZERO / ka)
    res = electroacoustics.piston_directivity_pattern([ka], angles=[0.0, theta_null])
    assert abs(res.directivity[0, 1]) < 1e-9
    assert res.directivity_db[0, 1] < -120.0
    # No null for a scalar ka below the first zero of J1.
    below = electroacoustics.piston_directivity_pattern(3.0)
    assert np.min(np.abs(below.directivity)) > 0.1


def test_directivity_pattern_plot() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    res = electroacoustics.piston_directivity_pattern([2.0, 5.0, 10.0])
    ax = res.plot()
    # A polar axes carrying one curve per ka value.
    assert ax.name == "polar"
    assert len(ax.lines) == 3
    assert "piston directivity" in ax.get_title()
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")


def test_directivity_pattern_plot_kwargs_single_ka_only() -> None:
    # User plot kwargs restyle a single-ka curve, but are ignored for a
    # multi-ka family: one color/label would collapse the per-curve styling.
    import matplotlib as mpl

    mpl.use("Agg")
    single = electroacoustics.piston_directivity_pattern(5.0)
    ax = single.plot(color="crimson", label="mine")
    assert ax.lines[0].get_color() == "crimson"
    assert ax.lines[0].get_label() == "mine"

    family = electroacoustics.piston_directivity_pattern([2.0, 5.0, 10.0])
    ax_family = family.plot(color="crimson")
    colors = {line.get_color() for line in ax_family.lines}
    assert "crimson" not in colors
    assert len(colors) == 3


def test_directivity_pattern_plot_rejects_a_cartesian_axes() -> None:
    # The drawer calls polar-only methods; a plain axes must be refused by
    # name, not left to die in matplotlib's AttributeError.
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    res = electroacoustics.piston_directivity_pattern([1.0, 3.0])
    _fig, cartesian = plt.subplots()
    with pytest.raises(ValueError, match="'ax' must be a polar axes"):
        res.plot(ax=cartesian)
    plt.close("all")


def test_a_pattern_matrix_with_a_surplus_row_is_refused() -> None:
    """The matrix rows and the ``ka`` values are one enumeration, and only the
    surplus is quiet.

    :meth:`plot` loops over ``ka`` and draws row ``i`` as the pattern of
    ``ka[i]``. A missing row runs that loop off the end, as numpy's "index 2 is
    out of bounds for axis 0 with size 2", and a column count that disagrees
    with ``angles`` is refused by matplotlib; both name no field, but both are
    heard. A surplus row is not: the loop stops at the last ``ka``, so the
    third pattern of a two-``ka`` bundle reaches neither the figure nor the
    legend.
    """
    result = electroacoustics.piston_directivity_pattern([1.0, 3.0, 8.0])
    two_of_the_three_ka = result.ka[:2]
    with pytest.raises(ValueError, match=r"'directivity' \(3\).*per pattern"):
        dataclasses.replace(result, ka=two_of_the_three_ka)


def test_directivity_pattern_validation() -> None:
    # Build the arguments up front so only the throwing call sits in each block.
    ka_one = [1.0]
    infinite_angles = [np.inf]
    empty_ka = np.empty(0)
    with pytest.raises(ValueError, match="'ka' must be non-negative"):
        electroacoustics.piston_directivity_pattern(-1.0)
    with pytest.raises(ValueError, match="'angles' must be finite"):
        electroacoustics.piston_directivity_pattern(ka_one, angles=infinite_angles)
    with pytest.raises(ValueError, match="'ka' must be a non-empty"):
        electroacoustics.piston_directivity_pattern(empty_ka)


def test_directivity_rejects_mismatched_shapes() -> None:
    # A shape mismatch used to surface as numpy's broadcast message, which
    # names neither argument.
    three_ka = np.array([1.0, 2.0, 3.0])
    two_angles = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="'ka' and 'theta' must broadcast together"):
        electroacoustics.piston_directivity(three_ka, two_angles)


def test_validation() -> None:
    with pytest.raises(ValueError, match="'radius' must be positive"):
        electroacoustics.radiating_piston(-1.0, [100.0])
    with pytest.raises(ValueError, match="'frequencies' must be positive"):
        electroacoustics.radiating_piston(0.1, [0.0])
    with pytest.raises(ValueError, match="'x' must be non-negative"):
        electroacoustics.piston_resistance(-1.0)
