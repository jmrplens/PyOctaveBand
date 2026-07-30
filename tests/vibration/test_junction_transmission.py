#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for the rigid plate-junction bending-wave transmission coefficients.

Hopkins (2007), *Sound Insulation*, Section 5.2.1.3 (Eqs 5.10-5.14, 5.6, 5.7,
5.116; and 2.154 for the coupling loss factor); Cremer et al. (1973); Craik
(1981, 1996).

**Clean-room oracle.** The transmission coefficients are deterministic closed
forms; every expected value below is derived *by hand* from the equations,
independent of the implementation:

* For **identical plates** ``chi = psi = 1`` the corner (Eq. 5.12) and straight
  (Eq. 5.13) denominators collapse. With ``J2 psi = 1`` the obliquity term is
  ``sqrt(1 + s2) sqrt(1 + s2) + sqrt(1 - s2) sqrt(1 - s2) = (1 + s2) +
  (1 - s2) = 2`` (``s2 = sin**2 theta``), so the denominator is
  ``1 + 1 + 1*2 = 4``. The X-junction corner numerator is
  ``0.5 * 1 * 1 * cos(theta) * sqrt(1 - s2) = 0.5 cos**2(theta)`` and the
  straight numerator is ``0.5 * 1 * cos**2(theta)``, hence
  ``tau12 = tau13 = cos**2(theta) / 8`` at *every* angle. Their angular average
  (Eq. 5.6) is ``(1/8) integral_0^(pi/2) cos**3(theta) d(theta) = (1/8)(2/3) =
  1/12``. For the **L-junction** ``J1 = 4`` doubles the corner numerator to
  ``2 cos**2(theta)`` so ``tau12 = cos**2(theta) / 2`` and the average is
  ``1/3``.
* At **normal incidence** ``theta = 0`` the obliquity term is
  ``sqrt(1) sqrt(chi**2) + sqrt(1) sqrt(chi**2) = 2 chi``, so both denominators
  become the perfect square ``(J2 psi + chi)**2`` / ``(J3 psi + chi)**2``, giving
  the reduced closed forms ``tau12(0) = 0.5 J1 J2 psi chi / (J2 psi + chi)**2``
  and ``tau13(0) = 0.5 chi**2 / (J3 psi + chi)**2`` for arbitrary ``chi``,
  ``psi`` -- an expression algebraically different from the implementation.
* **Reciprocity** (SEA consistency, Eq. 5.7): for the X- and L-junctions the two
  directions must satisfy ``tau_bar_12 = chi tau_bar_21`` exactly.
* **Cut-off**: the corner coefficient is ``0`` for ``chi < sin(theta)``.
* **Kij** (Eq. 5.116) is ``10 lg(1 / tau) + 5 lg(fc_j / f_ref)`` with
  ``f_ref = 1000 Hz``; combined with the Eq. 5.7 reciprocity it must be
  symmetric, ``K_ij = K_ji``, for every plate pair and junction type.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from phonometry import (
    angular_average_transmission_coefficient,
    corner_transmission_coefficient,
    coupling_loss_factor,
    inline_transmission_coefficient,
    junction_transmission,
    junction_wave_parameters,
    plate_bending_stiffness,
    point_connection_coupling_loss_factor,
    right_angle_transmission_coefficient,
    straight_transmission_coefficient,
    wave_vibration_reduction_index,
)
from phonometry.vibration import JunctionTransmissionResult
from phonometry.vibration.experimental_sea import flat_plate_modal_density
from phonometry.vibration.point_mobility import plate_bending_wave_speed

# Junction constants (Hopkins Eq. 5.12/5.13) reproduced here so the oracle never
# reads them from the module under test.
J1J2J3 = {"X": (1.0, 1.0, 1.0), "T1": (2.0, 0.5, 0.5), "T2": (2.0, 2.0, None),
          "L": (4.0, 1.0, None)}


# ---------------------------------------------------------------------------
# Wave parameters chi, psi (Hopkins Eqs 5.10, 5.11).
# ---------------------------------------------------------------------------
def test_wave_parameters_closed_form() -> None:
    # h1 cL1 = 0.1*3200, h2 cL2 = 0.2*3200 -> chi = sqrt(0.5), psi = 2*(480/240).
    chi, psi = junction_wave_parameters(0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
    assert chi == pytest.approx(math.sqrt(0.1 * 3200.0 / (0.2 * 3200.0)))
    assert chi == pytest.approx(math.sqrt(0.5))
    assert psi == pytest.approx((0.2 * 3200.0 * 480.0) / (0.1 * 3200.0 * 240.0))
    assert psi == pytest.approx(4.0)


def test_wave_parameters_identical_plates_are_unity() -> None:
    chi, psi = junction_wave_parameters(0.15, 3100.0, 375.0, 0.15, 3100.0, 375.0)
    assert chi == pytest.approx(1.0)
    assert psi == pytest.approx(1.0)


def test_chi_equals_sqrt_critical_frequency_ratio() -> None:
    # chi = sqrt(fc2 / fc1); fc ~ 1 / (cL h), so fc2/fc1 = (h1 cL1)/(h2 cL2).
    chi, _ = junction_wave_parameters(0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
    fc1 = 1.0 / (3200.0 * 0.1)
    fc2 = 1.0 / (3200.0 * 0.2)
    assert chi == pytest.approx(math.sqrt(fc2 / fc1))


# ---------------------------------------------------------------------------
# Identical X-junction: tau12(theta) = tau13(theta) = cos**2(theta) / 8.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deg", [0.0, 30.0, 45.0, 60.0, 90.0])
def test_identical_x_corner_is_cos_squared_over_eight(deg: float) -> None:
    theta = math.radians(deg)
    expected = math.cos(theta) ** 2 / 8.0
    got = float(corner_transmission_coefficient(theta, 1.0, 1.0, "X"))
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("deg", [0.0, 30.0, 45.0, 60.0, 90.0])
def test_identical_x_straight_is_cos_squared_over_eight(deg: float) -> None:
    theta = math.radians(deg)
    expected = math.cos(theta) ** 2 / 8.0
    got = float(straight_transmission_coefficient(theta, 1.0, 1.0, "X"))
    assert got == pytest.approx(expected)


def test_identical_x_specific_angles() -> None:
    # cos**2/8 at 0, 45, 60 deg -> 1/8, 1/16, 1/32.
    assert float(corner_transmission_coefficient(0.0, 1.0, 1.0, "X")) == pytest.approx(0.125)
    assert float(
        corner_transmission_coefficient(math.radians(45.0), 1.0, 1.0, "X")
    ) == pytest.approx(1.0 / 16.0)
    assert float(
        corner_transmission_coefficient(math.radians(60.0), 1.0, 1.0, "X")
    ) == pytest.approx(1.0 / 32.0)


def test_identical_x_angular_average_is_one_twelfth() -> None:
    corner = angular_average_transmission_coefficient(1.0, 1.0, "X", section="corner")
    straight = angular_average_transmission_coefficient(1.0, 1.0, "X", section="straight")
    assert corner == pytest.approx(1.0 / 12.0)
    assert straight == pytest.approx(1.0 / 12.0)


# ---------------------------------------------------------------------------
# Identical L-junction: tau12(theta) = cos**2(theta) / 2, average = 1/3.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deg", [0.0, 30.0, 45.0, 60.0])
def test_identical_l_corner_is_cos_squared_over_two(deg: float) -> None:
    theta = math.radians(deg)
    got = float(corner_transmission_coefficient(theta, 1.0, 1.0, "L"))
    assert got == pytest.approx(math.cos(theta) ** 2 / 2.0)


def test_identical_l_angular_average_is_one_third() -> None:
    avg = angular_average_transmission_coefficient(1.0, 1.0, "L", section="corner")
    assert avg == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Normal incidence: perfect-square denominator (arbitrary chi, psi).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("junction", ["X", "T1", "T2", "L"])
def test_corner_normal_incidence_perfect_square(junction: str) -> None:
    chi, psi = 1.5, 0.8
    j1, j2, _ = J1J2J3[junction]
    expected = 0.5 * j1 * j2 * psi * chi / (j2 * psi + chi) ** 2
    got = float(corner_transmission_coefficient(0.0, chi, psi, junction))
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("junction", ["X", "T1"])
def test_straight_normal_incidence_perfect_square(junction: str) -> None:
    chi, psi = 1.5, 0.8
    _, _, j3 = J1J2J3[junction]
    assert j3 is not None
    expected = 0.5 * chi**2 / (j3 * psi + chi) ** 2
    got = float(straight_transmission_coefficient(0.0, chi, psi, junction))
    assert got == pytest.approx(expected)


def test_t_junctions_identical_normal_incidence_two_ninths() -> None:
    # tau12(0) = 0.5 J1 J2 / (J2 + 1)**2 with chi = psi = 1:
    #   T1: 0.5*2*0.5/(1.5)**2 = 2/9;  T2: 0.5*2*2/(3)**2 = 2/9.
    assert float(
        corner_transmission_coefficient(0.0, 1.0, 1.0, "T1")
    ) == pytest.approx(2.0 / 9.0)
    assert float(
        corner_transmission_coefficient(0.0, 1.0, 1.0, "T2")
    ) == pytest.approx(2.0 / 9.0)
    assert float(
        straight_transmission_coefficient(0.0, 1.0, 1.0, "T1")
    ) == pytest.approx(2.0 / 9.0)


# ---------------------------------------------------------------------------
# In-line junction (Hopkins Eq. 5.14).
# ---------------------------------------------------------------------------
def test_inline_identical_plates_transmit_fully() -> None:
    assert inline_transmission_coefficient(1.0, 1.0) == pytest.approx(1.0)


def test_inline_asymmetric_closed_form() -> None:
    chi, psi = 1.5, 0.8
    ratio = 2.0 * (1.0 + chi) * (1.0 + psi) * math.sqrt(chi * psi) / (
        chi * (1.0 + psi) ** 2 + 2.0 * psi * (1.0 + chi**2)
    )
    assert inline_transmission_coefficient(chi, psi) == pytest.approx(ratio**2)


def test_inline_is_between_zero_and_one() -> None:
    for chi in (0.3, 0.7, 1.0, 2.0, 5.0):
        for psi in (0.2, 1.0, 3.0):
            tau = inline_transmission_coefficient(chi, psi)
            assert 0.0 <= tau <= 1.0


# ---------------------------------------------------------------------------
# Cut-off angle: corner coefficient is zero for chi < sin(theta).
# ---------------------------------------------------------------------------
def test_corner_cut_off_is_zero() -> None:
    # chi = 0.5, theta = 45 deg (sin 45 ~ 0.707 > 0.5) -> no transmitted wave.
    assert float(
        corner_transmission_coefficient(math.radians(45.0), 0.5, 1.0, "X")
    ) == 0.0
    # Just above the cut-off arcsin(0.5) = 30 deg the coefficient vanishes.
    assert float(
        corner_transmission_coefficient(math.radians(31.0), 0.5, 1.0, "X")
    ) == 0.0
    # Just below it there is a finite transmitted wave.
    assert float(
        corner_transmission_coefficient(math.radians(29.0), 0.5, 1.0, "X")
    ) > 0.0


def test_straight_is_continuous_across_cut_off() -> None:
    # The two branches of Eq. 5.13 must join at chi = sin(theta).
    chi, psi = 0.8, 0.7
    tco = math.asin(chi)
    below = float(straight_transmission_coefficient(tco - 1e-6, chi, psi, "X"))
    above = float(straight_transmission_coefficient(tco + 1e-6, chi, psi, "X"))
    assert below == pytest.approx(above, abs=1e-4)


# ---------------------------------------------------------------------------
# Reciprocity (Hopkins Eq. 5.7): tau_bar_12 = chi tau_bar_21 for X and L.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("junction", ["X", "L"])
def test_corner_reciprocity(junction: str) -> None:
    chi, psi = 1.5, 0.8
    forward = angular_average_transmission_coefficient(chi, psi, junction, section="corner")
    # Reverse direction: incident on plate 2 -> chi' = 1/chi, psi' = 1/psi.
    reverse = angular_average_transmission_coefficient(
        1.0 / chi, 1.0 / psi, junction, section="corner"
    )
    assert forward == pytest.approx(chi * reverse, rel=1e-6)


# ---------------------------------------------------------------------------
# Energy: transmission to the three other plates of an identical X cannot
# exceed the incident energy at any angle (reflection stays non-negative).
# ---------------------------------------------------------------------------
def test_identical_x_energy_bound() -> None:
    theta = np.radians(np.linspace(0.0, 89.0, 90))
    corner = corner_transmission_coefficient(theta, 1.0, 1.0, "X")
    straight = straight_transmission_coefficient(theta, 1.0, 1.0, "X")
    total = 2.0 * corner + straight  # two corner plates + one straight plate
    assert np.all(total <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# Coupling loss factor (Hopkins Eq. 2.154).
# ---------------------------------------------------------------------------
def test_coupling_loss_factor_closed_form() -> None:
    cg, lij, f, area, tau = 200.0, 4.0, 500.0, 10.0, 1.0 / 12.0
    expected = cg * lij * tau / (2.0 * math.pi**2 * f * area)
    got = float(coupling_loss_factor(tau, cg, lij, f, area))
    assert got == pytest.approx(expected)


def test_coupling_loss_factor_broadcasts_over_frequency() -> None:
    freqs = np.array([250.0, 500.0, 1000.0])
    eta = coupling_loss_factor(0.1, 300.0, 3.0, freqs, 8.0)
    assert eta.shape == freqs.shape
    # eta ~ 1/f, so doubling the frequency halves the coupling loss factor.
    assert eta[1] == pytest.approx(eta[0] / 2.0)
    assert eta[2] == pytest.approx(eta[0] / 4.0)


# ---------------------------------------------------------------------------
# Vibration reduction index (Hopkins Eq. 5.116).
# ---------------------------------------------------------------------------
def _critical_frequency_oracle(thickness: float, wave_speed: float) -> float:
    # Thin plate: fc = sqrt(12) c0**2 / (2 pi h cL), c0 = 343 m/s.
    return math.sqrt(12.0) * 343.0**2 / (2.0 * math.pi * thickness * wave_speed)


def test_kij_closed_form() -> None:
    # K = 10 lg(1/tau) + 5 lg(fc_j / 1000), fc_j the receiving plate's fc.
    tau, fcj = 0.05, 200.0
    expected = 10.0 * math.log10(1.0 / tau) + 5.0 * math.log10(fcj / 1000.0)
    assert float(wave_vibration_reduction_index(tau, fcj)) == pytest.approx(expected)


def test_kij_at_reference_frequency_is_ten_lg_inverse_tau() -> None:
    # fc_j = f_ref = 1000 Hz makes the correction term vanish.
    assert float(wave_vibration_reduction_index(1.0 / 12.0, 1000.0)) == pytest.approx(
        10.0 * math.log10(12.0)
    )
    assert float(wave_vibration_reduction_index(1.0 / 3.0, 1000.0)) == pytest.approx(
        10.0 * math.log10(3.0)
    )


def test_kij_identical_plates_uses_common_critical_frequency() -> None:
    # Identical plates: K = 10 lg(1/tau) + 5 lg(fc/1000) with the common fc.
    fc = _critical_frequency_oracle(0.1, 3200.0)
    res = junction_transmission("X", 0.1, 3200.0, 240.0, 0.1, 3200.0, 240.0)
    expected = 10.0 * math.log10(12.0) + 5.0 * math.log10(fc / 1000.0)
    assert res.corner_reduction_index == pytest.approx(expected)


def test_kij_rejects_nonpositive_critical_frequency() -> None:
    with pytest.raises(ValueError):
        wave_vibration_reduction_index(0.1, 0.0)
    with pytest.raises(ValueError):
        wave_vibration_reduction_index(0.1, -100.0)


def test_kij_l_junction_concrete_oracle_is_symmetric() -> None:
    # 100 mm (cL 3800, 220 kg/m^2) / 215 mm (cL 3200, 430 kg/m^2) concrete
    # L-junction: hand evaluation of Eqs 5.6 + 5.12 + 5.116 gives
    # K_12 = K_21 = 2.966 dB.
    plate_a = (0.100, 3800.0, 220.0)
    plate_b = (0.215, 3200.0, 430.0)
    k_ab = junction_transmission("L", *plate_a, *plate_b).corner_reduction_index
    k_ba = junction_transmission("L", *plate_b, *plate_a).corner_reduction_index
    assert k_ab == pytest.approx(k_ba, abs=1e-9)
    assert k_ab == pytest.approx(2.9664, abs=1e-3)


# For a T-junction the reverse direction swaps the junction constants:
# T1 sends plate 1 (a through plate) into the perpendicular plate 2, so the
# return path 2 -> 1 is a T2 corner, and vice versa.
_REVERSE_JUNCTION = {"X": "X", "L": "L", "T1": "T2", "T2": "T1"}


@pytest.mark.parametrize("junction", ["X", "L", "T1", "T2"])
def test_kij_symmetry_over_random_plate_pairs(junction: str) -> None:
    # Eq. 5.116 with f_ref = 1000 Hz and the Eq. 5.7 reciprocity of the
    # angular-average tau make K_ij = K_ji for every plate pair.
    rng = np.random.default_rng(20260721)
    for _ in range(10):
        plate_a = (
            float(rng.uniform(0.05, 0.30)),
            float(rng.uniform(1500.0, 4000.0)),
            float(rng.uniform(100.0, 600.0)),
        )
        plate_b = (
            float(rng.uniform(0.05, 0.30)),
            float(rng.uniform(1500.0, 4000.0)),
            float(rng.uniform(100.0, 600.0)),
        )
        k_ab = junction_transmission(
            junction, *plate_a, *plate_b
        ).corner_reduction_index
        k_ba = junction_transmission(
            _REVERSE_JUNCTION[junction], *plate_b, *plate_a
        ).corner_reduction_index
        assert k_ab == pytest.approx(k_ba, abs=1e-9)


# ---------------------------------------------------------------------------
# Result object and builder.
# ---------------------------------------------------------------------------
def test_result_identical_x() -> None:
    res = junction_transmission("X", 0.1, 3200.0, 240.0, 0.1, 3200.0, 240.0)
    assert isinstance(res, JunctionTransmissionResult)
    assert res.junction == "X"
    assert res.chi == pytest.approx(1.0)
    assert res.psi == pytest.approx(1.0)
    assert res.corner_average == pytest.approx(1.0 / 12.0)
    assert res.straight_average == pytest.approx(1.0 / 12.0)
    assert res.corner[0] == pytest.approx(0.125)
    fc = _critical_frequency_oracle(0.1, 3200.0)
    assert res.critical_frequency1 == pytest.approx(fc)
    assert res.critical_frequency2 == pytest.approx(fc)
    assert res.corner_reduction_index == pytest.approx(
        10.0 * math.log10(12.0) + 5.0 * math.log10(fc / 1000.0)
    )


def test_result_l_has_no_straight_section() -> None:
    res = junction_transmission("L", 0.12, 2000.0, 200.0, 0.2, 3200.0, 500.0)
    assert res.straight is None
    assert res.straight_average is None
    assert res.corner.shape == res.angles_deg.shape


def test_result_is_frozen() -> None:
    res = junction_transmission("X", 0.1, 3200.0, 240.0, 0.1, 3200.0, 240.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.chi = 2.0  # type: ignore[misc]


def test_result_accepts_custom_angles() -> None:
    res = junction_transmission(
        "X", 0.1, 3200.0, 240.0, 0.15, 3000.0, 360.0, angles_deg=[0.0, 45.0, 90.0]
    )
    assert res.angles_deg.tolist() == [0.0, 45.0, 90.0]
    assert res.corner.shape == (3,)


def test_plot_returns_axes() -> None:
    import matplotlib
    matplotlib.use("Agg")

    res = junction_transmission("X", 0.1, 3200.0, 240.0, 0.15, 3000.0, 360.0)
    ax = res.plot()
    assert ax.get_xlabel() == "Incidence angle [deg]"
    ax_es = res.plot(language="es")
    assert "ngulo" in ax_es.get_xlabel()


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
def test_unknown_junction_rejected() -> None:
    with pytest.raises(ValueError):
        corner_transmission_coefficient(0.0, 1.0, 1.0, "Z")


def test_negative_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        corner_transmission_coefficient(0.0, -1.0, 1.0, "X")
    with pytest.raises(ValueError):
        junction_wave_parameters(-0.1, 3200.0, 240.0, 0.1, 3200.0, 240.0)


def test_out_of_range_angle_rejected() -> None:
    with pytest.raises(ValueError):
        corner_transmission_coefficient(2.0, 1.0, 1.0, "X")  # > pi/2
    with pytest.raises(ValueError):
        corner_transmission_coefficient(-0.1, 1.0, 1.0, "X")


def test_straight_section_undefined_for_t2_and_l() -> None:
    with pytest.raises(ValueError):
        straight_transmission_coefficient(0.0, 1.0, 1.0, "T2")
    with pytest.raises(ValueError):
        straight_transmission_coefficient(0.0, 1.0, 1.0, "L")


# ---------------------------------------------------------------------------
# Norton & Karczub right-angle and point-connected junctions (Section 6.6.1)
#
# M. P. Norton and D. G. Karczub, *Fundamentals of Noise and Vibration Analysis
# for Engineers* (2nd ed., CUP 2003), Eqs. 6.52 to 6.56, with the published
# answer to problem 6.13 (printed p. 594; answer p. 617).
#
# **Problem 6.13.** Two aluminium plates coupled at right angles along a 1,2 m
# edge: plate 1 is 3 mm thick and 2,5 m x 1,2 m, plate 2 is 5,5 mm thick and
# 2,0 m x 1,2 m. Printed octave-band eta_12 from 125 Hz to 2 kHz:
#
#   welded  3,15e-3  2,23e-3  1,58e-3  1,12e-3  7,89e-4
#   bolted  1,44e-2  7,19e-3  3,60e-3  1,80e-3  8,99e-4  (twelve bolts)
#
# Aluminium from Norton's Appendix 4: rho = 2700 kg/m^3, E = 7,1e10 Pa,
# nu = 0,33, so the plate wave speed of Eq. (6.25) is 5432,3 m/s.
#
# The printed eta_21 column is *not* reproduced here and is not used as an
# oracle: it equals the eta_12 column times h2/h1 exactly, i.e. it applies the
# reciprocity relationship with n proportional to 1/h alone and drops the plate
# area ratio S1/S2 = 1,25 that Eq. (6.25) carries. See docs/ERRATA.md.
# ---------------------------------------------------------------------------

_AL_RHO = 2700.0
_AL_CL = math.sqrt(7.1e10 / (_AL_RHO * (1.0 - 0.33**2)))
_P613_BANDS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0])
_P613_WELDED = np.array([3.15e-3, 2.23e-3, 1.58e-3, 1.12e-3, 7.89e-4])
_P613_BOLTED = np.array([1.44e-2, 7.19e-3, 3.60e-3, 1.80e-3, 8.99e-4])
_H1, _H2 = 0.003, 0.0055
_S1, _S2, _L, _NBOLTS = 2.5 * 1.2, 2.0 * 1.2, 1.2, 12


def _norton_tau12(incidence: str = "random") -> float:
    return right_angle_transmission_coefficient(
        _H1, _H2, density1=_AL_RHO, density2=_AL_RHO,
        wave_speed1=_AL_CL, wave_speed2=_AL_CL, incidence=incidence,
    )


def test_norton_right_angle_normal_incidence_closed_form() -> None:
    # Eqs. (6.53)/(6.54) with equal materials collapse to psi = (h1/h2)**2.5,
    # an expression written here independently of the implementation.
    psi = (_H1 / _H2) ** 2.5
    expected = 2.0 / (math.sqrt(psi) + 1.0 / math.sqrt(psi)) ** 2
    assert _norton_tau12("normal") == pytest.approx(expected, rel=1e-12)


def test_norton_right_angle_normal_incidence_is_symmetric() -> None:
    # tau12(0) does not see which plate is which: swapping them inverts psi and
    # the (sqrt(psi) + 1/sqrt(psi)) sum is unchanged.
    swapped = right_angle_transmission_coefficient(
        _H2, _H1, density1=_AL_RHO, density2=_AL_RHO,
        wave_speed1=_AL_CL, wave_speed2=_AL_CL, incidence="normal",
    )
    assert swapped == pytest.approx(_norton_tau12("normal"), rel=1e-12)


def test_norton_right_angle_identical_plates() -> None:
    # Identical plates give psi = 1, hence tau12(0) = 2/4 = 0,5, and X = 1 so
    # the random-incidence factor is 2,754/4,24.
    same = right_angle_transmission_coefficient(
        0.004, 0.004, density1=_AL_RHO, density2=_AL_RHO,
        wave_speed1=_AL_CL, wave_speed2=_AL_CL, incidence="normal",
    )
    assert same == pytest.approx(0.5, rel=1e-12)
    random = right_angle_transmission_coefficient(
        0.004, 0.004, density1=_AL_RHO, density2=_AL_RHO,
        wave_speed1=_AL_CL, wave_speed2=_AL_CL,
    )
    assert random == pytest.approx(0.5 * 2.754 / 4.24, rel=1e-12)


def test_norton_problem_613_welded_line_junction() -> None:
    # Eq. (6.52) via the library's coupling loss factor (Hopkins Eq. 2.154 is
    # the same expression with cg = 2 cB), against the printed eta_12 column.
    tau = _norton_tau12()
    stiffness = plate_bending_stiffness(7.1e10, _H1, 0.33)
    c_b = plate_bending_wave_speed(_P613_BANDS, stiffness, _AL_RHO * _H1)
    eta = np.array(
        [
            float(coupling_loss_factor(tau, 2.0 * c, _L, f, _S1))
            for c, f in zip(c_b, _P613_BANDS, strict=True)
        ]
    )
    np.testing.assert_allclose(eta, _P613_WELDED, rtol=0.005)


def test_norton_line_junction_falls_as_inverse_root_frequency() -> None:
    # cB rises as sqrt(f) and Eq. (6.52) divides by omega, so eta ~ 1/sqrt(f):
    # the printed column falls by 1/sqrt(2) per octave, to the precision its
    # three significant figures allow.
    np.testing.assert_allclose(
        _P613_WELDED[1:] / _P613_WELDED[:-1], 1.0 / math.sqrt(2.0), rtol=0.005
    )


def test_norton_problem_613_bolted_point_connections() -> None:
    # Eq. (6.56) with twelve bolts, against the printed bolted eta_12 column.
    eta = point_connection_coupling_loss_factor(
        _P613_BANDS, _NBOLTS, thickness1=_H1, thickness2=_H2,
        surface_density1=_AL_RHO * _H1, surface_density2=_AL_RHO * _H2,
        wave_speed1=_AL_CL, wave_speed2=_AL_CL, plate_area1=_S1,
    )
    np.testing.assert_allclose(eta, _P613_BOLTED, rtol=0.007)


def test_norton_point_junction_falls_as_inverse_frequency() -> None:
    # Eq. (6.56) has omega alone in the denominator: the printed column halves
    # every octave exactly.
    np.testing.assert_allclose(
        _P613_BOLTED[1:] / _P613_BOLTED[:-1], 0.5, rtol=0.003
    )


def test_norton_point_junction_scales_with_the_bolt_count() -> None:
    kwargs = {
        "thickness1": _H1, "thickness2": _H2,
        "surface_density1": _AL_RHO * _H1,
        "surface_density2": _AL_RHO * _H2, "wave_speed1": _AL_CL,
        "wave_speed2": _AL_CL, "plate_area1": _S1,
    }
    one = point_connection_coupling_loss_factor(500.0, 1, **kwargs)  # type: ignore[arg-type]
    twelve = point_connection_coupling_loss_factor(500.0, 12, **kwargs)  # type: ignore[arg-type]
    np.testing.assert_allclose(twelve, 12.0 * one, rtol=1e-12)


def test_norton_reciprocity_needs_the_plate_areas() -> None:
    # The correct consistency relationship n1 eta_12 = n2 eta_21 uses
    # n = S sqrt(12)/(2 cL h), so eta_21/eta_12 = (S1 h2)/(S2 h1) = 2,29 and
    # not the 1,83 = h2/h1 that the printed eta_21 column of problem 6.13
    # implies. This pins the ratio the library uses.
    n_1 = flat_plate_modal_density(_S1, _H1, _AL_CL)
    n_2 = flat_plate_modal_density(_S2, _H2, _AL_CL)
    assert n_1 / n_2 == pytest.approx((_S1 * _H2) / (_S2 * _H1), rel=1e-12)
    assert n_1 / n_2 == pytest.approx(2.2916666, rel=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thickness1": 0.0},
        {"thickness2": -1.0},
        {"density1": 0.0},
        {"density2": 0.0},
        {"wave_speed1": 0.0},
        {"wave_speed2": 0.0},
        {"incidence": "grazing"},
    ],
)
def test_right_angle_validation(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "thickness1": _H1, "thickness2": _H2, "density1": _AL_RHO,
        "density2": _AL_RHO, "wave_speed1": _AL_CL, "wave_speed2": _AL_CL,
    }
    base.update(kwargs)
    thickness1 = base.pop("thickness1")
    thickness2 = base.pop("thickness2")
    with pytest.raises(ValueError):
        right_angle_transmission_coefficient(thickness1, thickness2, **base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_connections": 0},
        {"thickness1": 0.0},
        {"surface_density1": 0.0},
        {"wave_speed2": -1.0},
        {"plate_area1": 0.0},
    ],
)
def test_point_connection_validation(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "n_connections": 12, "thickness1": _H1, "thickness2": _H2,
        "surface_density1": _AL_RHO * _H1,
        "surface_density2": _AL_RHO * _H2,
        "wave_speed1": _AL_CL, "wave_speed2": _AL_CL, "plate_area1": _S1,
    }
    base.update(kwargs)
    n = base.pop("n_connections")
    with pytest.raises(ValueError):
        point_connection_coupling_loss_factor(500.0, n, **base)  # type: ignore[arg-type]


def test_point_connection_rejects_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        point_connection_coupling_loss_factor(
            [500.0, 0.0], 12, thickness1=_H1, thickness2=_H2,
            surface_density1=_AL_RHO * _H1, surface_density2=_AL_RHO * _H2,
            wave_speed1=_AL_CL, wave_speed2=_AL_CL, plate_area1=_S1,
        )
