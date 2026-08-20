#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the theoretical point mobilities of Cremer Table 5.1.

Anchored on the closed-form compilation of driving-point impedances and
mobilities of infinite structures (Cremer, Heckl & Petersson 2005, Structure-
Borne Sound 3e, Table 5.1): the plate impedance ``Z = 8 sqrt(B' m'')`` (real,
frequency independent), the beam mobility ``Y = (1 - j)/(4 m' cB)`` (45 degrees,
falling as omega**-1/2), the rod impedance ``rho cL S``, and the injected power
``W = 0.5 |F|**2 Re{Y}`` (Cremer Eq. 5.23).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import vibration
from phonometry.vibration.structural.point_mobility import (
    beam_bending_wave_speed,
    infinite_beam_impedance,
    longitudinal_rod_mobility,
    plate_bending_wave_speed,
)

BP = 1.0e4  # plate bending stiffness per width B' [N.m]
M2 = 10.0  # plate mass per unit area m'' [kg/m2]
B_BEAM = 200.0  # beam bending stiffness B = E I [N.m2]
M1 = 5.0  # beam mass per unit length m' [kg/m]


# ---------------------------------------------------------------------------
# Thin plate (Cremer Table 5.1): Z = C sqrt(B' m''), real.
# ---------------------------------------------------------------------------
def test_plate_impedance_centre() -> None:
    assert vibration.infinite_plate_impedance(BP, M2) == pytest.approx(
        8.0 * math.sqrt(BP * M2)
    )


def test_plate_impedance_edge() -> None:
    assert vibration.infinite_plate_impedance(BP, M2, location="edge") == pytest.approx(
        3.5 * math.sqrt(BP * M2)
    )


def test_plate_mobility_is_reciprocal_and_real() -> None:
    y = vibration.infinite_plate_mobility(BP, M2)
    assert y == pytest.approx(1.0 / (8.0 * math.sqrt(BP * M2)))
    # The plate mobility carries no reactance (pure resistance to a point force).
    assert y * vibration.infinite_plate_impedance(BP, M2) == pytest.approx(1.0)


def test_plate_mobility_frequency_independent() -> None:
    res = vibration.infinite_plate_point_mobility([50.0, 200.0, 1000.0], BP, M2)
    assert np.allclose(res.magnitude, res.magnitude[0])
    assert np.allclose(res.mobility.imag, 0.0)


# ---------------------------------------------------------------------------
# Slender beam (Cremer Table 5.1): Y = (1 - j)/(4 m' cB), 45 degrees.
# ---------------------------------------------------------------------------
def test_beam_bending_wave_speed() -> None:
    f = 100.0
    omega = 2.0 * math.pi * f
    assert beam_bending_wave_speed(f, B_BEAM, M1) == pytest.approx(
        (B_BEAM * omega**2 / M1) ** 0.25
    )


def test_beam_mobility_phase_is_minus_45_degrees() -> None:
    y = complex(vibration.infinite_beam_mobility(137.0, B_BEAM, M1))
    assert math.degrees(math.atan2(y.imag, y.real)) == pytest.approx(-45.0)


def test_beam_mobility_real_part() -> None:
    f = 100.0
    c_b = float(beam_bending_wave_speed(f, B_BEAM, M1))
    y = complex(vibration.infinite_beam_mobility(f, B_BEAM, M1))
    assert y.real == pytest.approx(1.0 / (4.0 * M1 * c_b))
    assert y.imag == pytest.approx(-1.0 / (4.0 * M1 * c_b))


def test_beam_end_four_times_centre() -> None:
    # Table 5.1: end mobility magnitude is 4x the centre value.
    yc = abs(complex(vibration.infinite_beam_mobility(200.0, B_BEAM, M1)))
    ye = abs(
        complex(vibration.infinite_beam_mobility(200.0, B_BEAM, M1, location="end"))
    )
    assert ye / yc == pytest.approx(4.0)


def test_beam_mobility_falls_as_sqrt_omega() -> None:
    y1 = abs(complex(vibration.infinite_beam_mobility(100.0, B_BEAM, M1)))
    y4 = abs(complex(vibration.infinite_beam_mobility(400.0, B_BEAM, M1)))
    # cB ~ sqrt(omega) so |Y| ~ omega**-1/2: quadrupling f halves |Y|.
    assert y1 / y4 == pytest.approx(2.0)


def test_beam_impedance_reciprocal() -> None:
    z = complex(infinite_beam_impedance(90.0, B_BEAM, M1))
    y = complex(vibration.infinite_beam_mobility(90.0, B_BEAM, M1))
    assert z * y == pytest.approx(1.0)


def test_beam_moment_mobility_phase_plus_45() -> None:
    # Y_M = omega (1 + j)/(4 B kB): +45 degrees.
    y = complex(vibration.infinite_beam_moment_mobility(120.0, B_BEAM, M1))
    assert math.degrees(math.atan2(y.imag, y.real)) == pytest.approx(45.0)


def test_beam_point_mobility_result() -> None:
    res = vibration.infinite_beam_point_mobility([50.0, 100.0], B_BEAM, M1)
    assert res.driving_point
    assert res.mobility.shape == (2,)


# ---------------------------------------------------------------------------
# Longitudinal rod (Cremer Table 5.1): Z = rho cL S.
# ---------------------------------------------------------------------------
def test_rod_impedance() -> None:
    assert vibration.longitudinal_rod_impedance(
        7800.0, 5100.0, 1.0e-3
    ) == pytest.approx(7800.0 * 5100.0 * 1.0e-3)
    assert longitudinal_rod_mobility(7800.0, 5100.0, 1.0e-3) == pytest.approx(
        1.0 / (7800.0 * 5100.0 * 1.0e-3)
    )


# ---------------------------------------------------------------------------
# Injected power (Cremer Eq. 5.23): W = 0.5 |F|**2 Re{Y}.
# ---------------------------------------------------------------------------
def test_injected_power_plate() -> None:
    y = vibration.infinite_plate_mobility(BP, M2)
    w = vibration.injected_power(10.0, y)
    assert float(w) == pytest.approx(0.5 * 100.0 * y)
    # Explicit closed form W = |F|**2 / (16 sqrt(B' m'')).
    assert float(w) == pytest.approx(100.0 / (16.0 * math.sqrt(BP * M2)))


def test_injected_power_uses_only_conductance() -> None:
    # A purely reactive mobility injects no net power.
    assert float(vibration.injected_power(3.0, 1j * 0.5)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def test_plate_bending_stiffness() -> None:
    # B' = E h**3 / (12 (1 - nu**2)).
    b = vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    assert b == pytest.approx(6.2e10 * 0.006**3 / (12.0 * (1.0 - 0.24**2)))


def test_plate_bending_wave_speed() -> None:
    f = 250.0
    omega = 2.0 * math.pi * f
    assert plate_bending_wave_speed(f, BP, M2) == pytest.approx(
        (BP * omega**2 / M2) ** 0.25
    )


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
def test_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="bending_stiffness"):
        vibration.infinite_plate_impedance(-1.0, M2)
    with pytest.raises(ValueError, match="location"):
        vibration.infinite_plate_impedance(BP, M2, location="corner")
    with pytest.raises(ValueError, match="frequency"):
        vibration.infinite_beam_mobility(0.0, B_BEAM, M1)
