#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for plane-wave seabed reflection (fluid-fluid Rayleigh model).

Oracles (independent of the implementation): the normal-incidence impedance
reflection coefficient ``R = (Z2 − Z1)/(Z2 + Z1)``; the analytic critical grazing
angle ``arccos(c1/c2)``; and total reflection (``|R| = 1``) below it.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry.underwater.propagation.seabed_reflection import (
    BottomLossResult,
    SeabedReflection,
    bottom_reflection_loss,
    critical_angle,
    reflection_coefficient,
    seabed_reflection,
)

# Water over a fast sandy bottom.
_WATER = {"rho1": 1000.0, "c1": 1500.0}
_SAND = {"rho2": 1900.0, "c2": 1650.0}


def test_normal_incidence_matches_impedance_formula() -> None:
    # At 90 deg grazing (normal incidence) R = (Z2 - Z1)/(Z2 + Z1), Z = rho*c.
    z1 = 1000.0 * 1500.0
    z2 = 1900.0 * 1650.0
    r_expected = (z2 - z1) / (z2 + z1)  # 0.35275
    r = reflection_coefficient(90.0, **_WATER, **_SAND)
    assert float(np.real(r[0])) == pytest.approx(r_expected, abs=1e-6)
    assert float(np.abs(r[0])) == pytest.approx(0.35275, abs=1e-4)


def test_normal_incidence_bottom_loss() -> None:
    # BL = -20 log10|R| ~= 9.05 dB for the sand/water pair at normal incidence.
    res = bottom_reflection_loss(90.0, **_WATER, **_SAND)
    assert isinstance(res, BottomLossResult)
    assert float(res.reflection_loss[0]) == pytest.approx(9.055, abs=1e-2)


def test_critical_angle_analytic() -> None:
    # phi_c = arccos(c1/c2) = arccos(1500/1650) = 24.620 deg.
    assert critical_angle(1500.0, 1650.0) == pytest.approx(
        np.degrees(np.arccos(1500.0 / 1650.0)), abs=1e-6
    )
    assert critical_angle(1500.0, 1650.0) == pytest.approx(24.620, abs=1e-3)


def test_total_reflection_below_critical_angle() -> None:
    # Below the critical grazing angle the wave is totally reflected: |R| = 1,
    # bottom loss ~ 0 dB.
    phi_c = critical_angle(1500.0, 1650.0)
    res = bottom_reflection_loss(np.array([5.0, 15.0, 24.0]), **_WATER, **_SAND)
    assert np.all(np.array([5.0, 15.0, 24.0]) < phi_c)
    assert np.allclose(np.abs(res.reflection_coefficient), 1.0, atol=1e-9)
    assert np.allclose(res.reflection_loss, 0.0, atol=1e-6)
    assert res.critical_angle == pytest.approx(phi_c)


def test_no_critical_angle_for_slow_bottom() -> None:
    # A slower bottom (mud, c2 < c1) has no critical angle.
    with pytest.raises(ValueError, match="c2 > c1"):
        critical_angle(1500.0, 1450.0)
    res = bottom_reflection_loss(45.0, rho1=1000.0, c1=1500.0, rho2=1500.0, c2=1450.0)
    assert res.critical_angle is None
    assert np.all(res.reflection_loss > 0.0)


def test_zero_grazing_equal_speeds_no_nan() -> None:
    # Singular case: phi = 0 and c1 == c2 gives 0/0; the analytic limit is the
    # angle-independent normal-incidence coefficient (z2 − z1)/(z2 + z1).
    res = bottom_reflection_loss(0.0, rho1=1000.0, c1=1500.0, rho2=1900.0, c2=1500.0)
    z1, z2 = 1000.0 * 1500.0, 1900.0 * 1500.0
    expected_r = (z2 - z1) / (z2 + z1)
    assert np.isfinite(res.reflection_coefficient[0])
    assert float(np.real(res.reflection_coefficient[0])) == pytest.approx(
        expected_r, abs=1e-9
    )
    assert float(res.reflection_loss[0]) == pytest.approx(
        -20.0 * np.log10(abs(expected_r)), abs=1e-9
    )


def test_intromission_angle_slow_bottom_closed_form() -> None:
    # Slow mud bottom (rho2 = 1300, c2 = 1400): R = 0 at the angle of
    # intromission. Closed form from R's numerator = 0 with Snell's law:
    # sin(phi_I) = sqrt((z1^2 - (rho1 c2)^2) / (z2^2 - (rho1 c2)^2)),
    # which gives phi_I = 27.585 deg for this pair. |R| vanishes there and the
    # bottom loss blows up; no warning may leak.
    rho1, c1, rho2, c2 = 1000.0, 1500.0, 1300.0, 1400.0
    z1, z2, z12 = rho1 * c1, rho2 * c2, rho1 * c2
    phi_i = float(np.degrees(np.arcsin(np.sqrt((z1**2 - z12**2) / (z2**2 - z12**2)))))
    assert phi_i == pytest.approx(27.585, abs=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = bottom_reflection_loss(phi_i, rho1=rho1, c1=c1, rho2=rho2, c2=c2)
    assert float(np.abs(res.reflection_coefficient[0])) == pytest.approx(0.0, abs=1e-12)
    assert float(res.reflection_loss[0]) > 100.0  # inf at the exact zero of R
    assert res.critical_angle is None


def test_exact_intromission_zero_gives_inf_loss_without_warning() -> None:
    # When |R| underflows to exactly 0 the loss is legitimately +inf; the
    # log10(0) RuntimeWarning must be silenced (identical media force R = 0
    # exactly at every angle).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = bottom_reflection_loss(
            30.0, rho1=1000.0, c1=1500.0, rho2=1000.0, c2=1500.0
        )
        wrapped = seabed_reflection(
            30.0, rho1=1000.0, c1=1500.0, rho2=1000.0, c2=1500.0
        )
    assert res.reflection_loss[0] == np.inf
    assert wrapped.magnitude[0] == 0.0
    assert wrapped.bottom_loss[0] == np.inf


def test_grazing_angle_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="grazing_angle"):
        reflection_coefficient(120.0, **_WATER, **_SAND)


def test_bottom_loss_plot_smoke() -> None:
    res = bottom_reflection_loss(np.linspace(0.0, 90.0, 91), **_WATER, **_SAND)
    assert res.plot() is not None


def test_seabed_reflection_result_bundles_the_maths() -> None:
    # The plottable wrapper re-runs no maths: R, |R| and bottom loss must match
    # the bare functions, and it carries the interface parameters.
    phi = np.linspace(0.0, 90.0, 91)
    res = seabed_reflection(phi, **_WATER, **_SAND)
    assert isinstance(res, SeabedReflection)
    r = reflection_coefficient(phi, **_WATER, **_SAND)
    assert np.allclose(res.reflection_coefficient, r)
    assert np.allclose(res.magnitude, np.abs(r))
    assert np.allclose(res.bottom_loss, -20.0 * np.log10(np.abs(r)))
    assert (res.rho1, res.c1, res.rho2, res.c2) == (1000.0, 1500.0, 1900.0, 1650.0)
    assert res.critical_angle == pytest.approx(critical_angle(1500.0, 1650.0))


def test_seabed_reflection_total_reflection_below_critical() -> None:
    # |R| = 1 below the critical grazing angle for a faster sediment.
    res = seabed_reflection(np.array([5.0, 15.0, 24.0]), **_WATER, **_SAND)
    assert np.all(np.array([5.0, 15.0, 24.0]) < res.critical_angle)
    assert np.allclose(res.magnitude, 1.0, atol=1e-9)


def test_seabed_reflection_normal_incidence_magnitude() -> None:
    # At 90 deg grazing (normal incidence) |R| = |Z2 - Z1|/(Z2 + Z1).
    res = seabed_reflection(90.0, **_WATER, **_SAND)
    z1, z2 = 1000.0 * 1500.0, 1900.0 * 1650.0
    assert float(res.magnitude[-1]) == pytest.approx((z2 - z1) / (z2 + z1), abs=1e-6)


def test_seabed_reflection_no_critical_angle_for_slow_bottom() -> None:
    res = seabed_reflection(45.0, rho1=1000.0, c1=1500.0, rho2=1500.0, c2=1450.0)
    assert res.critical_angle is None
    assert np.all(res.magnitude < 1.0)


def test_seabed_reflection_plot_smoke() -> None:
    res = seabed_reflection(np.linspace(0.0, 90.0, 91), **_WATER, **_SAND)
    assert res.plot() is not None
