#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the experimental (power-injection) statistical energy analysis.

M. P. Norton and D. G. Karczub, *Fundamentals of Noise and Vibration Analysis
for Engineers* (2nd ed., Cambridge University Press, 2003), Sections 6.3.3,
6.3.4 and 6.4.1 (Eqs. 6.8, 6.10 to 6.17 and 6.23 to 6.29), with the published
answer to problem 6.10.

**Clean-room oracle.**

* **Problem 6.10** (printed pp. 593-594; answer p. 617). A satellite platform
  (5 mm aluminium plate) carrying an aluminium cylinder 2 m long, 1,5 m mean
  diameter, 3 mm wall, in the 500 Hz octave: ``eta_1 = 4,4e-3``,
  ``eta_2 = 2,4e-3``, ``v_1 = 27,2 mm/s``, ``v_2 = 13,2 mm/s``. Printed
  answers ``eta_12 = 4,26e-4``, ``eta_21 = 3,92e-4`` and ``Pi_in = 1,31 W``.

  The problem statement gives the platform as 3,5 m x 3 m, but that area is
  **not** consistent with the printed answers: it fixes ``E_1/E_2 = 7,88``
  whereas the three printed values together require ``6,55``. The free plate
  area implied by the answers is 8,73 m^2, which is exactly 3,5 x 3 minus the
  0,75 m radius footprint of the cylinder that Fig. P6.10 shows standing on
  the platform. With that area all three printed answers come out within
  0,4 %. See ``docs/ERRATA.md``.

* **Reciprocity** (Eq. 6.8): ``n_1 eta_12 = n_2 eta_21`` must hold exactly for
  the single-drive inversion, and the printed pair satisfies
  ``eta_21/eta_12 = n_1/n_2`` to 0,3 %.

* **Energy balance**: the input power of Eq. (6.10) must equal the total
  dissipated power ``omega (eta_1 E_1 + eta_2 E_2)``, because substituting
  Eq. (6.11) into Eq. (6.10) cancels the coupling terms. That identity is
  checked independently of the printed value.

* **Round trip**: a forward SEA model built by hand from chosen loss factors
  and modal densities produces energies which, fed back through the inversion,
  must return the original loss factors. The forward model is written from
  Eqs. (6.10) and (6.11) directly, not from the inversion code.

* **Modal densities** (Eqs. 6.23 to 6.29) are closed forms; the flat-plate one
  is additionally checked against the independent EN 12354-4 expression
  ``n = pi S fc / c0^2`` already in the library.

Aluminium properties are Norton's Appendix 4 (printed p. 605):
``rho = 2700 kg/m^3``, ``E = 7,1e10 Pa``, ``nu = 0,33``, hence the plate wave
speed ``cL = sqrt(E/(rho(1 - nu^2))) = 5432,3 m/s`` of Eq. (6.25).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry.building.measurement.flanking_transmission import (
    modal_density as en12354_n,
)
from phonometry.vibration.structural.experimental_sea import (
    bar_modal_density,
    beam_modal_density,
    cylindrical_shell_modal_density,
    flat_plate_modal_density,
    power_injection_clf,
    power_injection_matrix,
    ring_frequency,
)

# --- Norton Appendix 4, aluminium ---
_RHO = 2700.0
_NU = 0.33
_E = 7.1e10
_CL = math.sqrt(_E / (_RHO * (1.0 - _NU**2)))

# --- Norton problem 6.10 geometry ---
_T1, _T2 = 0.005, 0.003
_RADIUS, _LENGTH = 0.75, 2.0
_S2 = 2.0 * math.pi * _RADIUS * _LENGTH
#: Free platform area: 3,5 m x 3 m minus the cylinder footprint (see module
#: docstring and docs/ERRATA.md).
_S1 = 3.5 * 3.0 - math.pi * _RADIUS**2
_BAND = 500.0
#: A single-band frequency axis for the matrix validation tests.
_ONE_BAND = np.array([500.0])


def _problem_610_inputs() -> tuple[float, float, float, float]:
    """``(E_1, E_2, n_1, n_2)`` of problem 6.10, all from the book's data."""
    e_1 = _RHO * _T1 * _S1 * 0.0272**2
    e_2 = _RHO * _T2 * _S2 * 0.0132**2
    n_1 = flat_plate_modal_density(_S1, _T1, _CL)
    n_2 = float(
        cylindrical_shell_modal_density(_BAND, _S2, _T2, _RADIUS, _CL, band="octave")[0]
    )
    return e_1, e_2, n_1, n_2


class TestNortonProblem610:
    """The single-drive inversion against the printed answers."""

    @pytest.fixture(scope="class")
    def result(self):  # type: ignore[no-untyped-def]
        e_1, e_2, n_1, n_2 = _problem_610_inputs()
        return power_injection_clf(_BAND, e_1, e_2, 4.4e-3, 2.4e-3, n_1, n_2)

    def test_coupling_loss_factor_12(self, result) -> None:  # type: ignore[no-untyped-def]
        """Printed answer eta_12 = 4,26e-4."""
        assert float(result.coupling_loss_factor12[0]) == pytest.approx(
            4.26e-4, rel=0.005
        )

    def test_coupling_loss_factor_21(self, result) -> None:  # type: ignore[no-untyped-def]
        """Printed answer eta_21 = 3,92e-4."""
        assert float(result.coupling_loss_factor21[0]) == pytest.approx(
            3.92e-4, rel=0.005
        )

    def test_input_power(self, result) -> None:  # type: ignore[no-untyped-def]
        """Printed answer Pi_in = 1,31 W."""
        assert float(result.input_power[0]) == pytest.approx(1.31, rel=0.005)

    def test_ring_frequency_places_the_band_below_it(self) -> None:
        """``fr = cL / (2 pi a_m)`` = 1153 Hz, so 500 Hz uses Eq. (6.27)."""
        f_r = ring_frequency(_RADIUS, _CL)
        assert f_r == pytest.approx(1152.8, rel=1e-3)
        assert _BAND / f_r < 0.48

    def test_reciprocity_holds_exactly(self, result) -> None:  # type: ignore[no-untyped-def]
        """Eq. (6.8): ``n_1 eta_12 = n_2 eta_21``."""
        assert float((result.modal_density1 * result.coupling_loss_factor12)[0]) == (
            pytest.approx(
                float((result.modal_density2 * result.coupling_loss_factor21)[0]),
                rel=1e-12,
            )
        )

    def test_printed_pair_satisfies_reciprocity(self) -> None:
        """The two printed loss factors imply the computed modal-density ratio."""
        _, _, n_1, n_2 = _problem_610_inputs()
        assert 3.92e-4 / 4.26e-4 == pytest.approx(n_1 / n_2, rel=0.005)

    def test_input_power_equals_dissipated_power(self, result) -> None:  # type: ignore[no-untyped-def]
        """Eq. (6.10) collapses to the total dissipation in the steady state."""
        np.testing.assert_allclose(
            result.input_power, result.dissipated_power, rtol=1e-12
        )

    def test_transmitted_power_leaves_subsystem_two_in_balance(self, result) -> None:  # type: ignore[no-untyped-def]
        """Eq. (6.11): the power crossing the junction is dissipated in 2."""
        omega = 2.0 * np.pi * result.frequencies
        np.testing.assert_allclose(
            result.transmitted_power,
            omega * result.internal_loss_factor2 * result.energy2,
            rtol=1e-12,
        )

    def test_coupling_is_weak(self, result) -> None:  # type: ignore[no-untyped-def]
        """``eta_12 << eta_1``: the two-subsystem model is trustworthy here."""
        assert float(result.coupling_strength[0]) < 0.15


class TestSingleDriveRoundTrip:
    """A hand-built forward SEA model inverts back to its own loss factors."""

    @staticmethod
    def _forward(
        f: np.ndarray,
        eta_1: float,
        eta_2: float,
        eta_12: float,
        n_1: float,
        n_2: float,
        power: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Energies from Eqs. (6.10) and (6.11), written out directly."""
        eta_21 = eta_12 * n_1 / n_2
        omega = 2.0 * np.pi * f
        # From Eq. (6.11): E2 = E1 eta_12 / (eta_2 + eta_21).
        ratio = eta_12 / (eta_2 + eta_21)
        # Eq. (6.10) with Pi_1 = power.
        e_1 = power / (omega * ((eta_1 + eta_12) - ratio * eta_21))
        return e_1, ratio * e_1

    @pytest.mark.parametrize(
        ("eta_1", "eta_2", "eta_12", "n_1", "n_2"),
        [
            (4.4e-3, 2.4e-3, 4.26e-4, 0.557, 0.606),
            (1.0e-2, 1.0e-2, 1.0e-3, 2.0, 2.0),
            (5.0e-3, 2.0e-2, 3.0e-3, 0.8, 3.2),
        ],
    )
    def test_round_trip(
        self, eta_1: float, eta_2: float, eta_12: float, n_1: float, n_2: float
    ) -> None:
        f = np.array([250.0, 500.0, 1000.0])
        e_1, e_2 = self._forward(f, eta_1, eta_2, eta_12, n_1, n_2, 1.0)
        res = power_injection_clf(f, e_1, e_2, eta_1, eta_2, n_1, n_2)
        np.testing.assert_allclose(res.coupling_loss_factor12, eta_12, rtol=1e-10)
        np.testing.assert_allclose(
            res.coupling_loss_factor21, eta_12 * n_1 / n_2, rtol=1e-10
        )
        np.testing.assert_allclose(res.input_power, 1.0, rtol=1e-10)

    def test_scalar_loss_factors_broadcast_over_bands(self) -> None:
        f = np.array([125.0, 250.0, 500.0, 1000.0])
        e_1, e_2 = self._forward(f, 5e-3, 5e-3, 1e-3, 1.0, 1.0, 2.0)
        res = power_injection_clf(f, e_1, e_2, 5e-3, 5e-3, 1.0, 1.0)
        assert res.coupling_loss_factor12.shape == f.shape
        np.testing.assert_allclose(res.coupling_loss_factor12, 1e-3, rtol=1e-10)

    def test_modal_density_ratio_property(self) -> None:
        f = np.array([500.0])
        e_1, e_2 = self._forward(f, 5e-3, 5e-3, 1e-3, 0.8, 3.2, 1.0)
        res = power_injection_clf(f, e_1, e_2, 5e-3, 5e-3, 0.8, 3.2)
        np.testing.assert_allclose(res.modal_density_ratio, 0.8 / 3.2, rtol=1e-10)


class TestTwoDriveMatrix:
    """The full power-injection matrix recovers all four loss factors."""

    @staticmethod
    def _energies(
        f: np.ndarray,
        eta_1: float,
        eta_2: float,
        eta_12: float,
        eta_21: float,
        p_1: float,
        p_2: float,
    ) -> np.ndarray:
        """Solve Eqs. (6.10)/(6.11) forwards for both drive configurations."""
        omega = 2.0 * np.pi * f
        e = np.empty((2, 2, f.size))
        a = np.array(
            [[eta_1 + eta_12, -eta_21], [-eta_12, eta_2 + eta_21]], dtype=np.float64
        )
        inv = np.linalg.inv(a)
        for j, drive in enumerate(((p_1, 0.0), (0.0, p_2))):
            rhs = np.array(drive, dtype=np.float64)[:, None] / omega[None, :]
            e[:, j, :] = inv @ rhs
        return e

    def test_recovers_all_four_loss_factors(self) -> None:
        f = np.array([250.0, 500.0, 1000.0])
        eta_1, eta_2, eta_12, eta_21 = 4.4e-3, 2.4e-3, 4.26e-4, 3.92e-4
        e = self._energies(f, eta_1, eta_2, eta_12, eta_21, 1.31, 0.7)
        powers = np.stack([np.full(f.size, 1.31), np.full(f.size, 0.7)])
        res = power_injection_matrix(f, e, powers)
        np.testing.assert_allclose(res.coupling_loss_factor12, eta_12, rtol=1e-9)
        np.testing.assert_allclose(res.coupling_loss_factor21, eta_21, rtol=1e-9)
        np.testing.assert_allclose(res.internal_loss_factor1, eta_1, rtol=1e-9)
        np.testing.assert_allclose(res.internal_loss_factor2, eta_2, rtol=1e-9)
        assert res.method == "two-drive"
        assert res.modal_density1 is None

    def test_reciprocity_becomes_a_check_not_an_input(self) -> None:
        """``eta_21/eta_12`` recovers the modal-density ratio it was built with."""
        f = np.array([500.0])
        n_1, n_2 = 0.557, 0.606
        eta_12 = 4.26e-4
        e = self._energies(f, 4.4e-3, 2.4e-3, eta_12, eta_12 * n_1 / n_2, 1.31, 0.7)
        res = power_injection_matrix(f, e, np.stack([[1.31], [0.7]]))
        np.testing.assert_allclose(res.modal_density_ratio, n_1 / n_2, rtol=1e-8)


class TestModalDensities:
    """Norton Eqs. 6.23 to 6.29."""

    def test_bar_is_frequency_independent(self) -> None:
        """``n(f) = 2 L / cL`` (Eq. 6.23)."""
        assert bar_modal_density(10.0, 5150.0) == pytest.approx(
            2.0 * 10.0 / 5150.0, rel=1e-12
        )

    def test_beam_decreases_with_frequency(self) -> None:
        """``n(f) = L (rho A/E I)^(1/4) / sqrt(2 pi f)`` (Eq. 6.24)."""
        n = beam_modal_density([100.0, 400.0], 2.0, 7.8, 1.2e4)
        assert n[1] == pytest.approx(n[0] / 2.0, rel=1e-12)

    def test_flat_plate_matches_the_en12354_form(self) -> None:
        """``S sqrt(12)/(2 cL t)`` equals ``pi S fc / c0^2`` exactly.

        The two are the same quantity once ``fc = c0^2 sqrt(12)/(2 pi cL t)``,
        so this pins Eq. (6.25) against an implementation written from a
        different standard (EN 12354-4 Formula (5)).
        """
        c_0, area, t = 343.0, 10.5, 0.005
        f_c = c_0**2 * math.sqrt(12.0) / (2.0 * math.pi * _CL * t)
        assert flat_plate_modal_density(area, t, _CL) == pytest.approx(
            en12354_n(area, f_c, speed_of_sound=c_0), rel=1e-12
        )

    def test_cylinder_below_ring_frequency(self) -> None:
        """Eq. (6.27) rises as ``sqrt(f)`` up to ``f/fr = 0,48``."""
        f_r = ring_frequency(_RADIUS, _CL)
        f = np.array([0.1, 0.4]) * f_r
        n = cylindrical_shell_modal_density(f, _S2, _T2, _RADIUS, _CL)
        assert n[1] / n[0] == pytest.approx(2.0, rel=1e-12)
        expected = 5.0 * _S2 / (math.pi * _CL * _T2) * math.sqrt(0.1)
        assert n[0] == pytest.approx(expected, rel=1e-12)

    def test_cylinder_middle_regime_is_linear(self) -> None:
        """Eq. (6.28) is proportional to ``f`` between 0,48 and 0,83 fr."""
        f_r = ring_frequency(_RADIUS, _CL)
        f = np.array([0.5, 0.8]) * f_r
        n = cylindrical_shell_modal_density(f, _S2, _T2, _RADIUS, _CL)
        assert n[1] / n[0] == pytest.approx(0.8 / 0.5, rel=1e-12)
        assert n[0] == pytest.approx(7.2 * _S2 / (math.pi * _CL * _T2) * 0.5, rel=1e-12)

    def test_cylinder_tends_to_the_flat_plate_far_above_the_ring(self) -> None:
        """Eq. (6.29) approaches the flat-plate value within 10 % at ``f >> fr``.

        Both arc-cosines tend to ``pi/2``, so the bracket tends to
        ``2 + 0,596 pi/2 = 2,936`` and ``n -> 5,873 S/(pi cL t)`` against the
        flat plate's ``sqrt(12)/2 = 5,441 S/(pi cL t)``.
        """
        f_r = ring_frequency(_RADIUS, _CL)
        n = float(cylindrical_shell_modal_density(2e4 * f_r, _S2, _T2, _RADIUS, _CL)[0])
        plate = flat_plate_modal_density(_S2, _T2, _CL)
        assert n == pytest.approx(plate, rel=0.10)
        assert n == pytest.approx(
            2.0 * _S2 / (math.pi * _CL * _T2) * (2.0 + 0.596 * math.pi / 2.0),
            rel=1e-6,
        )

    @pytest.mark.parametrize(("band", "factor"), [("octave", 1.414), ("third", 1.122)])
    def test_bandwidth_factor_enters_only_above_the_ring(
        self, band: str, factor: float
    ) -> None:
        f_r = ring_frequency(_RADIUS, _CL)
        low = cylindrical_shell_modal_density(
            0.2 * f_r, _S2, _T2, _RADIUS, _CL, band=band
        )
        assert float(low[0]) == pytest.approx(
            5.0 * _S2 / (math.pi * _CL * _T2) * math.sqrt(0.2), rel=1e-12
        )
        x = 2.0
        high = float(
            cylindrical_shell_modal_density(x * f_r, _S2, _T2, _RADIUS, _CL, band=band)[
                0
            ]
        )
        shape = 0.596 / (factor - 1.0 / factor)
        expected = (
            2.0
            * _S2
            / (math.pi * _CL * _T2)
            * (
                2.0
                + shape
                * (
                    factor * math.acos(1.745 / (factor**2 * x**2))
                    - math.acos(min(1.745 * factor**2 / x**2, 1.0)) / factor
                )
            )
        )
        assert high == pytest.approx(expected, rel=1e-12)


class TestValidation:
    """Invalid measurements are rejected rather than silently inverted."""

    def test_receiver_richer_in_modal_energy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="modal energy"):
            power_injection_clf(500.0, 1.0, 2.0, 4.4e-3, 2.4e-3, 1.0, 1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_inputs(self, bad: float) -> None:
        with pytest.raises(ValueError):
            power_injection_clf(500.0, bad, 0.1, 4.4e-3, 2.4e-3, 1.0, 1.0)

    def test_band_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="frequency bands"):
            power_injection_clf(
                [125.0, 250.0], [1.0, 0.9, 0.8], 0.1, 4.4e-3, 2.4e-3, 1.0, 1.0
            )

    def test_matrix_rejects_a_bad_energy_shape(self) -> None:
        f, energies, powers = _ONE_BAND, np.ones((2, 2, 2)), np.ones((2, 1))
        with pytest.raises(ValueError, match=r"shape \(2, 2, 1\)"):
            power_injection_matrix(f, energies, powers)

    def test_matrix_rejects_a_bad_power_shape(self) -> None:
        f, energies, powers = _ONE_BAND, np.ones((2, 2, 1)), np.ones((2, 2))
        with pytest.raises(ValueError, match=r"shape \(2, 1\)"):
            power_injection_matrix(f, energies, powers)

    def test_matrix_rejects_non_positive_energies(self) -> None:
        energies, powers = np.ones((2, 2, 1)), np.ones((2, 1))
        energies[0, 0, 0] = 0.0
        with pytest.raises(ValueError, match="'energies'"):
            power_injection_matrix(_ONE_BAND, energies, powers)

    def test_matrix_rejects_non_positive_powers(self) -> None:
        energies, powers = np.ones((2, 2, 1)), np.zeros((2, 1))
        with pytest.raises(ValueError, match="'input_powers'"):
            power_injection_matrix(_ONE_BAND, energies, powers)

    def test_matrix_rejects_indistinguishable_drive_tests(self) -> None:
        """Two tests with proportional energy distributions are singular."""
        energies = np.array([[[1.0], [2.0]], [[0.5], [1.0]]])
        powers = np.ones((2, 1))
        with pytest.raises(ValueError, match="singular"):
            power_injection_matrix(_ONE_BAND, energies, powers)

    def test_unknown_band(self) -> None:
        with pytest.raises(ValueError):
            cylindrical_shell_modal_density(500.0, 9.4, 0.003, 0.75, _CL, band="half")

    @pytest.mark.parametrize(
        "call",
        [
            lambda: bar_modal_density(0.0, 5150.0),
            lambda: bar_modal_density(1.0, 0.0),
            lambda: beam_modal_density(100.0, 0.0, 7.8, 1.2e4),
            lambda: flat_plate_modal_density(1.0, 0.0, 5150.0),
            lambda: ring_frequency(0.0, 5150.0),
        ],
    )
    def test_non_positive_geometry(self, call) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError):
            call()
