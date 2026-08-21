#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Porous-material models and multilayer absorber prediction.

Oracle strategy (no free fitting anywhere):

- Delany-Bazley and Miki are polynomial regressions, so the printed
  coefficients themselves are the oracle: the models are pinned digit-exact
  at a mid-range digitization point recomputed by hand from Bies 5e
  Table D.1 / Mechel 2e Sect. G.11 Eq. (2) and Miki 1990 Eqs. (30)-(34)
  (``tests/reference_data/``).
- JCA is pinned to its exact closed-form limits (Johnson et al. 1987):
  ``j w rho_e -> sigma`` at DC, ``rho_e -> (T rho/phi)(1 + (1-j) delta_v/L)``
  at high frequency and the isothermal/adiabatic bulk-modulus limits, plus
  the published Cox & D'Antonio Figs. 6.19-6.21 comparison (JCA with
  ``phi = 0.98``, ``T = 1``, ``L' = L``, ``s = 1`` tracks Delany-Bazley over
  the fit range).
- The transfer-matrix solver is pinned to the closed-form hard-backed layer
  ``Zs = -j Zc cot(k d)`` (Mechel D.3 Eq. (1); Bies Eq. (D.94)), to an
  independent implementation of the Bies Eq. (D.95) impedance recursion, to
  the lossless air-layer limits (``|R| = 1`` including the lambda/4
  resonance), and cross-checked against the ASTM E2611 machinery of
  ``impedance_tube`` (Song-Bolton recovery of ``Zc``/``k`` and the
  hard-backed absorption).
- Resonators are pinned to the closed-form resonance frequencies
  (Cox & D'Antonio Eqs. (7.4)/(7.9)/(7.10)) and to Maa 1998: the exact
  Eq. (2) against Maa's own wide-range approximation Eq. (4) (stated max
  error ~6 %), the closed-form absorption Eq. (9) and its oblique variant
  Eq. (23) as machine-precision identities of the TMM chain, the Eq. (11)
  resonance condition, the printed Table I values, and the Fig. 5 design
  example (d = t = 0.2 mm, b = 2.5 mm, D = 6 cm; also Cox & D'Antonio
  Fig. 7.28).
- Energy and reciprocity: ``0 <= alpha <= 1``, ``|R| <= 1`` and
  ``det(T) = 1`` over randomized stacks on a dense grid.
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref

from phonometry.materials.absorbers.four_microphone import (
    TransferMatrix,
)

# The termination admittance the equivalent-fluid recursion consumes, checked
# bit for bit in TestLayeredAbsorber.
from phonometry.materials.absorbers.layered import (
    AirLayer,
    MembraneLayer,
    MicroperforatedPlateLayer,
    PerforatedPlateLayer,
    PorousLayer,
    _termination_admittance,
    diffuse_field_absorption,
    layered_absorber,
    statistical_absorption,
)
from phonometry.materials.absorbers.porous import (
    DELANY_BAZLEY_COEFFICIENTS,
    DELANY_BAZLEY_VALIDITY,
    PorousAbsorberWarning,
    PorousMediumResult,
    delany_bazley,
    helmholtz_resonance_frequency,
    johnson_champoux_allard,
    membrane_impedance,
    membrane_resonance_frequency,
    microperforated_plate_impedance,
    miki,
    perforated_plate_impedance,
    perforation_end_correction,
)

RHO0 = 1.205
C0 = 343.0
RC = RHO0 * C0


def _grid(lo: float = 50.0, hi: float = 4000.0, n: int = 300) -> np.ndarray:
    return np.linspace(lo, hi, n)


# ---------------------------------------------------------------------------
# Delany-Bazley
# ---------------------------------------------------------------------------
class TestDelanyBazley:
    def test_printed_coefficients_at_digitization_point(self) -> None:
        """Bies Table D.1 / Mechel G.11 Eq. (2), X = 0.1, digit-exact."""
        sigma = 20000.0
        f = np.array([ref.POROUS_DB_X_POINT * sigma / RHO0])
        res = delany_bazley(f, sigma, speed_of_sound=C0, air_density=RHO0)
        np.testing.assert_allclose(
            res.normalized_impedance, [ref.POROUS_DB_ZC_EXPECTED], rtol=1e-12
        )
        np.testing.assert_allclose(
            res.normalized_wavenumber, [ref.POROUS_DB_K_EXPECTED], rtol=1e-12
        )

    def test_high_x_approaches_air(self) -> None:
        """X -> upper fit edge: Zc -> rho c and k -> k0 monotonically."""
        sigma = 5000.0
        f = np.array([0.02, 0.2, 1.0]) * sigma / RHO0
        res = delany_bazley(f, sigma, air_density=RHO0)
        dev = np.abs(res.normalized_impedance - 1.0)
        assert dev[0] > dev[1] > dev[2]
        assert dev[2] < 0.15

    def test_passivity_in_fit_range(self) -> None:
        """Re(Zc) > 0 and Im(k) < 0 (decaying wave) inside 0.01 < X < 1."""
        sigma = 20000.0
        x = np.linspace(*DELANY_BAZLEY_VALIDITY, 200)
        res = delany_bazley(x * sigma / RHO0, sigma, air_density=RHO0)
        assert np.all(res.characteristic_impedance.real > 0.0)
        assert np.all(res.wavenumber.imag < 0.0)

    @pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
    def test_layer_inadmissibility_below_fit_range(self) -> None:
        """Mechel G.12: DB gives a negative-Re layer impedance at low X."""
        sigma = 100000.0
        f = np.linspace(5.0, 200.0, 100)
        with pytest.warns(PorousAbsorberWarning):
            med = delany_bazley(f, sigma, air_density=RHO0)
        zs = layered_absorber(f, [PorousLayer(0.01, med)]).surface_impedance
        assert np.min(zs.real) < 0.0

    def test_warns_outside_validity(self) -> None:
        with pytest.warns(PorousAbsorberWarning, match="fit range"):
            delany_bazley(np.array([10.0]), 50000.0)
        with pytest.warns(PorousAbsorberWarning, match="fit range"):
            delany_bazley(np.array([20000.0]), 1000.0)

    def test_presets_and_custom_coefficients(self) -> None:
        f = np.array([1000.0])
        for name in DELANY_BAZLEY_COEFFICIENTS:
            res = delany_bazley(f, 10000.0, coefficients=name)
            assert res.characteristic_impedance.real > 0.0
        explicit = delany_bazley(
            f, 10000.0, coefficients=DELANY_BAZLEY_COEFFICIENTS["garai_pompoli"]
        )
        preset = delany_bazley(f, 10000.0, coefficients="garai_pompoli")
        np.testing.assert_array_equal(
            explicit.characteristic_impedance, preset.characteristic_impedance
        )

    def test_rejects_bad_inputs(self) -> None:
        f = np.array([100.0])
        with pytest.raises(ValueError, match="preset"):
            delany_bazley(f, 10000.0, coefficients="nope")
        with pytest.raises(ValueError, match="8 values"):
            delany_bazley(f, 10000.0, coefficients=(1.0, 2.0))
        with pytest.raises(ValueError, match="'flow_resistivity' must be positive"):
            delany_bazley(f, -1.0)


# ---------------------------------------------------------------------------
# Miki
# ---------------------------------------------------------------------------
class TestMiki:
    def test_printed_coefficients_at_digitization_point(self) -> None:
        """Miki 1990 Eqs. (30)-(34), f/sigma = 0.1, digit-exact."""
        sigma = 20000.0
        f = np.array([ref.POROUS_MIKI_Y_POINT * sigma])
        res = miki(f, sigma, speed_of_sound=C0, air_density=RHO0)
        np.testing.assert_allclose(
            res.normalized_impedance, [ref.POROUS_MIKI_ZC_EXPECTED], rtol=1e-12
        )
        np.testing.assert_allclose(
            res.normalized_wavenumber, [ref.POROUS_MIKI_K_EXPECTED], rtol=1e-12
        )

    @pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
    def test_positive_real_below_fit_range(self) -> None:
        """The Miki constraint: Re(Zc) > 0 well below f/sigma = 0.01.

        The original Delany-Bazley layer impedance turns negative there
        (Mechel G.12); Miki's regression was rebuilt to stay passive. The
        hard-backed layer of Miki's Fig. 3 (sigma = 100 kPa s/m2, l = 10 mm)
        keeps a positive real part over the plotted 0-5 kHz range.
        """
        sigma = 100000.0
        f = np.linspace(50.0, 5000.0, 200)
        with pytest.warns(PorousAbsorberWarning):
            med = miki(f, sigma, air_density=RHO0)
        assert np.all(med.characteristic_impedance.real > 0.0)
        zs = layered_absorber(f, [PorousLayer(0.01, med)]).surface_impedance
        assert np.all(zs.real > -1e-9)

    def test_agrees_with_delany_bazley_in_fit_range(self) -> None:
        """Both regressions describe the same data set (within ~15 %)."""
        sigma = 20000.0
        y = np.linspace(0.05, 0.8, 60)
        m = miki(y * sigma, sigma, air_density=RHO0)
        d = delany_bazley(y * sigma, sigma, air_density=RHO0)
        rel = np.abs(m.characteristic_impedance - d.characteristic_impedance)
        assert np.max(rel / np.abs(d.characteristic_impedance)) < 0.15


# ---------------------------------------------------------------------------
# Johnson-Champoux-Allard
# ---------------------------------------------------------------------------
class TestJohnsonChampouxAllard:
    SIGMA = 20000.0
    PHI = 0.98
    ETA = 1.84e-5

    def _jca(self, f: np.ndarray, tort: float = 1.2) -> PorousMediumResult:
        lam = np.sqrt(8.0 * tort * self.ETA / (self.PHI * self.SIGMA))
        return johnson_champoux_allard(
            f,
            self.SIGMA,
            porosity=self.PHI,
            tortuosity=tort,
            viscous_length=lam,
            thermal_length=2.0 * lam,
            air_density=RHO0,
            viscosity=self.ETA,
        )

    def test_viscous_dc_limit(self) -> None:
        """Johnson et al. 1987: j w rho_e -> sigma as w -> 0 (exact)."""
        res = self._jca(np.array([1e-3]))
        val = 1j * 2.0 * np.pi * 1e-3 * res.effective_density
        np.testing.assert_allclose(val.real, [self.SIGMA], rtol=1e-9)

    def test_high_frequency_density_asymptote(self) -> None:
        """rho_e -> (T rho/phi)(1 + (1-j) delta_v / L) as w -> inf."""
        tort = 1.2
        f = np.array([1e7])
        res = self._jca(f, tort)
        lam = np.sqrt(8.0 * tort * self.ETA / (self.PHI * self.SIGMA))
        delta_v = np.sqrt(2.0 * self.ETA / (RHO0 * 2.0 * np.pi * f))
        expected = tort * RHO0 / self.PHI * (1.0 + (1.0 - 1.0j) * delta_v / lam)
        np.testing.assert_allclose(res.effective_density, expected, rtol=1e-4)

    def test_bulk_modulus_limits(self) -> None:
        """K_e: isothermal P0/phi at DC, adiabatic gamma P0/phi at HF."""
        p0 = 101325.0
        lo = self._jca(np.array([1e-3]))
        hi = self._jca(np.array([1e9]))
        np.testing.assert_allclose(lo.bulk_modulus.real, [p0 / self.PHI], rtol=1e-6)
        np.testing.assert_allclose(
            hi.bulk_modulus.real, [1.4 * p0 / self.PHI], rtol=1e-3
        )

    def test_tracks_delany_bazley_over_fit_range(self) -> None:
        """Cox Figs. 6.19-6.21: JCA (phi=0.98, T=1, L'=L, s=1) ~ DB."""
        lam = np.sqrt(8.0 * self.ETA / (self.PHI * self.SIGMA))
        x = np.logspace(-2, 0, 50)
        f = x * self.SIGMA / RHO0
        jca = johnson_champoux_allard(
            f,
            self.SIGMA,
            porosity=self.PHI,
            tortuosity=1.0,
            viscous_length=lam,
            thermal_length=lam,
            air_density=RHO0,
            viscosity=self.ETA,
        )
        db = delany_bazley(f, self.SIGMA, air_density=RHO0)
        rel_z = np.abs(jca.characteristic_impedance - db.characteristic_impedance)
        rel_k = np.abs(jca.wavenumber - db.wavenumber)
        assert np.max(rel_z / np.abs(db.characteristic_impedance)) < 0.15
        assert np.max(rel_k / np.abs(db.wavenumber)) < 0.10
        a_j = layered_absorber(f, [PorousLayer(0.05, jca)]).absorption
        a_d = layered_absorber(f, [PorousLayer(0.05, db)]).absorption
        assert np.max(np.abs(a_j - a_d)) < 0.06

    def test_consistency_zc_k_density_modulus(self) -> None:
        """Zc = sqrt(rho_e K_e) and k = w sqrt(rho_e/K_e) hold internally."""
        f = _grid(100.0, 2000.0, 20)
        res = self._jca(f)
        np.testing.assert_allclose(
            res.characteristic_impedance,
            np.sqrt(res.effective_density * res.bulk_modulus),
            rtol=1e-12,
        )
        omega = 2.0 * np.pi * f
        np.testing.assert_allclose(
            res.wavenumber,
            omega * np.sqrt(res.effective_density / res.bulk_modulus),
            rtol=1e-12,
        )

    def test_rejects_bad_parameters(self) -> None:
        f = np.array([100.0])
        with pytest.raises(ValueError, match="porosity"):
            johnson_champoux_allard(
                f,
                1e4,
                porosity=1.2,
                tortuosity=1.0,
                viscous_length=1e-4,
                thermal_length=2e-4,
            )
        with pytest.raises(ValueError, match="tortuosity"):
            johnson_champoux_allard(
                f,
                1e4,
                porosity=0.9,
                tortuosity=0.5,
                viscous_length=1e-4,
                thermal_length=2e-4,
            )


# ---------------------------------------------------------------------------
# Transfer-matrix solver
# ---------------------------------------------------------------------------
class TestLayeredAbsorber:
    def test_hard_backed_layer_closed_form(self) -> None:
        """Zs = -j Zc cot(k d) (Mechel D.3 Eq. (1); Bies Eq. (D.94))."""
        f = _grid(200.0, 4000.0)
        med = delany_bazley(f, 20000.0, air_density=RHO0)
        res = layered_absorber(
            f, [PorousLayer(0.05, med)], speed_of_sound=C0, air_density=RHO0
        )
        zs_ref = -1j * med.characteristic_impedance / np.tan(med.wavenumber * 0.05)
        np.testing.assert_allclose(res.surface_impedance, zs_ref, rtol=1e-12)

    def test_matches_bies_impedance_recursion(self) -> None:
        """Independent Bies Eq. (D.95) recursion reproduces the TMM stack."""
        f = _grid(300.0, 3000.0, 200)
        med1 = delany_bazley(f, 30000.0, air_density=RHO0)
        med2 = miki(f, 8000.0, air_density=RHO0)
        layers = [PorousLayer(0.02, med1), AirLayer(0.03), PorousLayer(0.04, med2)]
        res = layered_absorber(f, layers, speed_of_sound=C0, air_density=RHO0)
        # Bies (D.95): Z_i = Zm (Z_{i-1} + j Zm tan(km l)) / (Zm + j Z_{i-1} tan)
        k0 = 2.0 * np.pi * f / C0
        z_l = np.full(f.shape, np.inf + 0j)
        media = [
            (med2.characteristic_impedance, med2.wavenumber, 0.04),
            (RC + 0j * f, k0 + 0j, 0.03),
            (med1.characteristic_impedance, med1.wavenumber, 0.02),
        ]
        for zm, km, d in media:
            tan = np.tan(km * d)
            with np.errstate(invalid="ignore"):
                z_l = np.where(
                    np.isinf(z_l),
                    zm / (1j * tan),
                    zm * (z_l + 1j * zm * tan) / (zm + 1j * z_l * tan),
                )
        np.testing.assert_allclose(res.surface_impedance, z_l, rtol=1e-9)

    def test_air_layer_over_rigid_is_lossless(self) -> None:
        """|R| = 1 and alpha = 0 for a pure air gap, lambda/4 included."""
        d = 0.1
        f_quarter = C0 / (4.0 * d)
        f = np.linspace(20.0, 4000.0, 500)
        f = np.sort(np.append(f, f_quarter))
        res = layered_absorber(f, [AirLayer(d)], speed_of_sound=C0)
        np.testing.assert_allclose(np.abs(res.reflection), 1.0, atol=1e-12)
        np.testing.assert_allclose(res.absorption, 0.0, atol=1e-12)

    def test_zero_thickness_layers_are_transparent(self) -> None:
        f = _grid(200.0, 1000.0, 50)
        med = delany_bazley(f, 20000.0, air_density=RHO0)
        with_zero = layered_absorber(
            f, [AirLayer(0.0), PorousLayer(0.0, med), PorousLayer(0.05, med)]
        )
        without = layered_absorber(f, [PorousLayer(0.05, med)])
        np.testing.assert_array_equal(with_zero.reflection, without.reflection)

    @pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
    def test_very_high_resistivity_approaches_rigid_wall(self) -> None:
        f = _grid(100.0, 1000.0, 50)
        with pytest.warns(PorousAbsorberWarning):
            med = miki(f, 1e9, air_density=RHO0)
        res = layered_absorber(f, [PorousLayer(0.05, med)])
        assert np.max(res.absorption) < 0.01

    def test_free_termination_of_bare_air_is_reflectionless(self) -> None:
        """An air 'layer' radiating into free air behind reflects nothing."""
        f = _grid(100.0, 1000.0, 50)
        res = layered_absorber(f, [AirLayer(0.1)], termination="free")
        np.testing.assert_allclose(np.abs(res.reflection), 0.0, atol=1e-12)
        res_ob = layered_absorber(f, [AirLayer(0.1)], termination="free", angle=0.7)
        np.testing.assert_allclose(np.abs(res_ob.reflection), 0.0, atol=1e-12)

    def test_free_termination_admittance_is_the_literal_cosine_over_rho_c(
        self,
    ) -> None:
        """The equivalent-fluid path must not be reassociated.

        The global-matrix assembly needs the termination as an impedance,
        ``rho c / cos(theta)``, while the admittance recursion has always used
        ``cos(theta) / rho c``. Those are not the same double: on a sweep of
        4001 angles ``1 / (rho c / cos theta)`` differs in bits from
        ``cos(theta) / rho c`` for 997 of them. Bit equality, not a tolerance,
        is the assertion, because a few ulps is exactly what such a
        reassociation costs and exactly what no tolerance would catch.
        """
        f = _grid(100.0, 1000.0, 3)
        for theta in np.linspace(0.0, np.pi / 2.0, 4001, endpoint=False):
            cos_t = float(np.cos(theta))
            assert np.array_equal(
                _termination_admittance("free", f, cos_t=cos_t, rc=RC),
                np.full(f.shape, cos_t / RC, dtype=np.complex128),
            )

    def test_impedance_termination_matches_rigid_limit(self) -> None:
        f = _grid(200.0, 1000.0, 50)
        med = delany_bazley(f, 20000.0, air_density=RHO0)
        rigid = layered_absorber(f, [PorousLayer(0.05, med)])
        huge = layered_absorber(f, [PorousLayer(0.05, med)], termination=1e12 + 0j)
        np.testing.assert_allclose(huge.reflection, rigid.reflection, atol=1e-6)

    def test_cross_check_with_impedance_tube_machinery(self) -> None:
        """ASTM E2611 recovery returns the model Zc/k from the layer matrix."""
        f = _grid(200.0, 800.0, 60)  # keep Re(k d) < pi for arccos branch
        med = delany_bazley(f, 20000.0, air_density=RHO0)
        d = 0.05
        res = layered_absorber(
            f, [PorousLayer(d, med)], speed_of_sound=C0, air_density=RHO0
        )
        tm = res.transfer_matrix
        t = TransferMatrix(t11=tm[0, 0], t12=tm[0, 1], t21=tm[1, 0], t22=tm[1, 1])
        np.testing.assert_allclose(
            t.characteristic_impedance_material(),
            med.characteristic_impedance,
            rtol=1e-10,
        )
        np.testing.assert_allclose(t.material_wavenumber(d), med.wavenumber, rtol=1e-10)
        np.testing.assert_allclose(
            t.absorption_hard_backed(RC), res.absorption, atol=1e-12
        )

    @pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
    def test_energy_and_reciprocity_random_stacks(self) -> None:
        """0 <= alpha <= 1, |R| <= 1 and det(T) = 1 over random stacks."""
        rng = np.random.default_rng(20260717)
        f = _grid(50.0, 8000.0, 400)
        for _ in range(12):
            med = miki(f, float(rng.uniform(3000, 80000)), air_density=RHO0)
            layers = [
                MicroperforatedPlateLayer(
                    float(rng.uniform(2e-4, 1e-3)),
                    float(rng.uniform(5e-5, 3e-4)),
                    float(rng.uniform(0.005, 0.05)),
                ),
                AirLayer(float(rng.uniform(0.0, 0.05))),
                PorousLayer(float(rng.uniform(0.01, 0.1)), med),
                MembraneLayer(float(rng.uniform(0.1, 5.0))),
                AirLayer(float(rng.uniform(0.01, 0.1))),
            ]
            angle = float(rng.uniform(0.0, 1.4))
            res = layered_absorber(f, layers, angle=angle)
            assert np.all(res.absorption >= -1e-9)
            assert np.all(res.absorption <= 1.0 + 1e-9)
            assert np.all(np.abs(res.reflection) <= 1.0 + 1e-9)
            t = res.transfer_matrix
            det = t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]
            # det(T) = 1 exactly; the float error scales with the size of
            # the cancelled products, so the tolerance is relative to them.
            scale = np.abs(t[0, 0] * t[1, 1]) + np.abs(t[0, 1] * t[1, 0])
            assert np.all(np.abs(det - 1.0) <= 1e-12 * np.maximum(1.0, scale))

    def test_recursion_equals_matrix_product_all_terminations(self) -> None:
        """The admittance recursion equals the chain-matrix solution.

        For a mixed five-layer stack, R computed from the exposed
        ``transfer_matrix`` (with each termination closure) must equal the
        recursion's ``reflection`` to machine precision at every angle.
        """
        f = _grid(250.0, 4000.0, 300)
        med = miki(f, 20000.0, air_density=RHO0)
        layers = [
            MicroperforatedPlateLayer(3e-4, 1e-4, 0.01),
            AirLayer(0.02),
            PorousLayer(0.03, med),
            MembraneLayer(1.2),
            AirLayer(0.04),
        ]
        for termination in ("rigid", "free", 800.0 + 200.0j):
            for angle in (0.0, 0.7, 1.3):
                res = layered_absorber(f, layers, angle=angle, termination=termination)
                t = res.transfer_matrix
                cos_t = np.cos(angle)
                if termination == "rigid":
                    p, u = t[0, 0], t[1, 0]
                else:
                    zl = RC / cos_t if termination == "free" else termination
                    p = t[0, 0] * zl + t[0, 1]
                    u = t[1, 0] * zl + t[1, 1]
                r = (p * cos_t - RC * u) / (p * cos_t + RC * u)
                np.testing.assert_allclose(r, res.reflection, atol=1e-13)

    def test_oblique_incidence_air_cavity_closed_form(self) -> None:
        """Air gap at angle: Zs = -j (rho c/cos t) cot(k d cos t) (Mechel D.4)."""
        f = _grid(100.0, 2000.0, 100)
        theta = 0.9
        d = 0.04
        res = layered_absorber(f, [AirLayer(d)], angle=theta)
        k0 = 2.0 * np.pi * f / C0
        zs_ref = -1j * (RC / np.cos(theta)) / np.tan(k0 * d * np.cos(theta))
        np.testing.assert_allclose(res.surface_impedance, zs_ref, rtol=1e-10)

    @pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
    def test_rejects_bad_inputs(self) -> None:
        f = _grid(100.0, 200.0, 5)
        med = miki(f, 20000.0)
        layers = [PorousLayer(0.05, med)]
        with pytest.raises(ValueError, match="at least one layer"):
            layered_absorber(f, [])
        with pytest.raises(ValueError, match="angle"):
            layered_absorber(f, layers, angle=np.pi / 2.0)
        with pytest.raises(ValueError, match="termination"):
            layered_absorber(f, layers, termination="soft")
        other = miki(_grid(100.0, 200.0, 7), 20000.0)
        mismatched = [PorousLayer(0.05, other)]
        with pytest.raises(ValueError, match="frequency vector"):
            layered_absorber(f, mismatched)


# ---------------------------------------------------------------------------
# Resonant sheets
# ---------------------------------------------------------------------------
class TestResonantSheets:
    def test_end_correction_limits(self) -> None:
        """delta -> 0.85 for an isolated hole; decreases with open area."""
        assert perforation_end_correction(1e-9) == pytest.approx(0.85, abs=1e-4)
        assert perforation_end_correction(0.3) < perforation_end_correction(0.05)

    def test_helmholtz_resonance_closed_form_vs_tmm(self) -> None:
        """f0 = (c/2pi) sqrt(eps/(t' d)) matches the TMM peak (Cox Eq. 7.4)."""
        plate = PerforatedPlateLayer(0.006, 0.0025, 0.05)
        d = 0.025
        f0 = helmholtz_resonance_frequency(
            cavity_depth=d,
            plate_thickness=plate.thickness,
            hole_radius=plate.hole_radius,
            open_area=plate.open_area,
        )
        f = np.linspace(0.3 * f0, 2.0 * f0, 4000)
        res = layered_absorber(f, [plate, AirLayer(d)])
        # Resonance = zero crossing of Im(Zs); alpha peaks there too.
        f_peak = f[int(np.argmax(res.absorption))]
        assert abs(f_peak - f0) / f0 < 0.12

    def test_maa_microperforated_example(self) -> None:
        """Maa 1998 Fig. 5 (= Cox Fig. 7.28): d=t=0.2 mm, b=2.5 mm, D=6 cm.

        The published curve shows near-total absorption at the first
        resonance in the high hundreds of hertz; the shallow-cavity closed
        form (Cox Eq. (7.4), which ignores the viscous plug-mass increase
        and the finite cavity depth) sits somewhat above the full model.
        """
        eps = (np.pi / 4.0) * (ref.MAA_FIG5_DIAMETER / ref.MAA_FIG5_SEPARATION) ** 2
        f = np.linspace(100.0, 4000.0, 2000)
        res = layered_absorber(
            f,
            [
                MicroperforatedPlateLayer(
                    ref.MAA_FIG5_THICKNESS, ref.MAA_FIG5_DIAMETER / 2.0, eps
                ),
                AirLayer(ref.MAA_FIG5_CAVITY),
            ],
        )
        i = int(np.argmax(res.absorption))
        assert 550.0 < f[i] < 950.0
        assert res.absorption[i] > 0.9
        f0 = helmholtz_resonance_frequency(
            cavity_depth=ref.MAA_FIG5_CAVITY,
            plate_thickness=ref.MAA_FIG5_THICKNESS,
            hole_radius=ref.MAA_FIG5_DIAMETER / 2.0,
            open_area=eps,
            end_correction=0.85,
        )
        assert f[i] < f0 < 1.35 * f[i]

    def test_maa_exact_vs_wide_range_approximation(self) -> None:
        """Maa 1998 Eq. (2) vs the Eq. (4) approximation: max error ~6 %.

        Maa combined the Crandall small/large-``k`` limits into
        ``Z1 = (32 eta t / d^2) sqrt(1 + k^2/32)
        + j w rho t (1 + (9 + k^2/2)^(-1/2))`` (Eq. (4), the plate-internal
        part before end corrections; Eq. (5b) misprints the 9 as 1, see
        docs/ERRATA.md) and states a maximum error of about 6 % against the
        exact Bessel solution over all k. Pin both parts to 8 % over k in
        [0.5, 10].
        """
        rho0, eta = RHO0, 1.84e-5
        a, t = 0.1e-3, 0.2e-3
        d = 2.0 * a
        k_perf = np.linspace(0.5, 10.0, 40)
        f = (k_perf / d) ** 2 * 4.0 * eta / (rho0 * 2.0 * np.pi)
        omega = 2.0 * np.pi * f
        # Exact hole impedance = eps * (sheet impedance without the end
        # corrections), recovered by passing eps = 1 and delta = 0.
        z_exact = (
            microperforated_plate_impedance(
                f,
                thickness=t,
                hole_radius=a,
                open_area=1.0,
                end_correction=0.0,
                air_density=rho0,
                viscosity=eta,
            )
            - np.sqrt(2.0 * omega * rho0 * eta) / 2.0
        )
        r_approx = (32.0 * eta * t / d**2) * np.sqrt(1.0 + k_perf**2 / 32.0)
        x_approx = omega * rho0 * t * (1.0 + (9.0 + k_perf**2 / 2.0) ** -0.5)
        assert np.max(np.abs(z_exact.real - r_approx) / r_approx) < 0.08
        assert np.max(np.abs(z_exact.imag - x_approx) / x_approx) < 0.08

    def test_maa_closed_form_absorption_identity(self) -> None:
        """TMM chain == Maa 1998 Eq. (9) (and Eq. (23) at oblique).

        For an MPP over a cavity the normalised surface impedance is
        ``z = r + j w m - j cot(w D / c)`` and
        ``alpha = 4 r / ((1 + r)^2 + (w m - cot(w D / c))^2)``; at oblique
        incidence every term is multiplied by ``cos(theta)`` and the
        cotangent argument becomes ``(w D / c) cos(theta)`` (Eq. (23)).
        """
        eps = (np.pi / 4.0) * (ref.MAA_FIG5_DIAMETER / ref.MAA_FIG5_SEPARATION) ** 2
        f = np.linspace(100.0, 4000.0, 400)
        mpp = MicroperforatedPlateLayer(
            ref.MAA_FIG5_THICKNESS, ref.MAA_FIG5_DIAMETER / 2.0, eps
        )
        z_sheet = (
            microperforated_plate_impedance(
                f,
                thickness=mpp.thickness,
                hole_radius=mpp.hole_radius,
                open_area=mpp.open_area,
                air_density=RHO0,
            )
            / RC
        )
        k0 = 2.0 * np.pi * f / C0
        for theta in (0.0, 0.9):
            res = layered_absorber(f, [mpp, AirLayer(ref.MAA_FIG5_CAVITY)], angle=theta)
            cos_t = np.cos(theta)
            reactance = z_sheet.imag - (
                1.0 / np.tan(k0 * ref.MAA_FIG5_CAVITY * cos_t) / cos_t
            )
            alpha_maa = (4.0 * z_sheet.real * cos_t) / (
                (cos_t * z_sheet.real + 1.0) ** 2 + (cos_t * reactance) ** 2
            )
            np.testing.assert_allclose(res.absorption, alpha_maa, atol=1e-12)

    def test_maa_resonance_condition(self) -> None:
        """Maa 1998 Eq. (11): w0 m = cot(w0 D / c) locates the TMM peak."""
        eps = (np.pi / 4.0) * (ref.MAA_FIG5_DIAMETER / ref.MAA_FIG5_SEPARATION) ** 2
        f = np.linspace(200.0, 2000.0, 8000)
        mpp = MicroperforatedPlateLayer(
            ref.MAA_FIG5_THICKNESS, ref.MAA_FIG5_DIAMETER / 2.0, eps
        )
        res = layered_absorber(f, [mpp, AirLayer(ref.MAA_FIG5_CAVITY)])
        xm = (
            microperforated_plate_impedance(
                f,
                thickness=mpp.thickness,
                hole_radius=mpp.hole_radius,
                open_area=mpp.open_area,
                air_density=RHO0,
            ).imag
            / RC
        )
        cot = 1.0 / np.tan(2.0 * np.pi * f * ref.MAA_FIG5_CAVITY / C0)
        i_root = int(np.argmin(np.abs(xm - cot)))
        i_peak = int(np.argmax(res.absorption))
        assert abs(f[i_peak] - f[i_root]) / f[i_root] < 0.02

    def test_maa_table_i_printed_values(self) -> None:
        """Maa 1998 Table I: alpha0 (Eq. (10)) and B = f2/f1 (Eq. (21))."""
        for r, alpha0, b in zip(
            ref.MAA_TABLE_I_R,
            ref.MAA_TABLE_I_ALPHA0,
            ref.MAA_TABLE_I_BANDWIDTH,
            strict=True,
        ):
            assert 4.0 * r / (1.0 + r) ** 2 == pytest.approx(alpha0, abs=0.005)
            arccot = np.arctan(1.0 / (1.0 + r))
            # The printed Table I truncates to two decimals (e.g. 14.9152
            # is printed as 14.91), so allow just above half a hundredth.
            assert np.pi / arccot - 1.0 == pytest.approx(b, abs=0.0075)

    def test_microperforated_impedance_positive_resistance(self) -> None:
        f = _grid(50.0, 8000.0, 200)
        z = microperforated_plate_impedance(
            f, thickness=0.2e-3, hole_radius=0.1e-3, open_area=0.005
        )
        assert np.all(z.real > 0.0)
        assert np.all(z.imag > 0.0)

    def test_membrane_resonance_constants(self) -> None:
        """Cox Eqs. (7.9)/(7.10): 60/sqrt(md) adiabatic, 50/sqrt(md) isothermal."""
        m, d = 5.0, 0.05
        f_ad = membrane_resonance_frequency(surface_density=m, cavity_depth=d)
        f_iso = membrane_resonance_frequency(
            surface_density=m, cavity_depth=d, isothermal=True
        )
        assert f_ad == pytest.approx(60.0 / np.sqrt(m * d), rel=0.02)
        assert f_iso == pytest.approx(50.0 / np.sqrt(m * d), rel=0.02)

    @pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
    def test_membrane_over_cavity_peaks_at_resonance(self) -> None:
        m, d = 5.0, 0.05
        f0 = membrane_resonance_frequency(surface_density=m, cavity_depth=d)
        f = np.linspace(20.0, 500.0, 3000)
        med = delany_bazley(f, 10000.0, air_density=RHO0)
        res = layered_absorber(
            f, [MembraneLayer(m), AirLayer(0.02), PorousLayer(0.03, med)]
        )
        f_peak = f[int(np.argmax(res.absorption))]
        assert abs(f_peak - f0) / f0 < 0.10

    def test_membrane_impedance_is_pure_mass(self) -> None:
        f = np.array([100.0])
        z = membrane_impedance(f, surface_density=2.0)
        assert z.real == pytest.approx(0.0)
        assert z.imag == pytest.approx(2.0 * np.pi * 100.0 * 2.0)

    def test_effectively_grazing_angle_rejected(self) -> None:
        """Angles within ~1e-6 of pi/2 would drive an air layer's kx to
        exactly zero (inf * 0 = nan); they are rejected up front.
        """
        f = np.array([1000.0])
        layers = [AirLayer(0.05)]
        with pytest.raises(ValueError, match="angle"):
            layered_absorber(f, layers, angle=np.pi / 2.0 - 1e-9)

    def test_termination_array_length_mismatch_rejected(self) -> None:
        f = np.array([500.0, 1000.0])
        layers = [AirLayer(0.05)]
        termination = np.array([800.0 + 0j] * 3)
        with pytest.raises(ValueError, match="termination"):
            layered_absorber(f, layers, termination=termination)

    def test_perforated_plate_open_area_one_is_nearly_transparent(self) -> None:
        """eps -> 1: only the small visco-thermal terms remain."""
        f = np.array([1000.0])
        z = perforated_plate_impedance(
            f, thickness=1e-3, hole_radius=5e-3, open_area=1.0
        )
        assert np.abs(z)[0] < 0.05 * RC


# ---------------------------------------------------------------------------
# Random incidence (Paris integral) and the locally reacting closed form
# ---------------------------------------------------------------------------
class TestRandomIncidence:
    def test_statistical_absorption_matches_numeric_paris(self) -> None:
        """Mechel D.5 Eq. (10) == direct quadrature of Eq. (9), any limit."""
        rng = np.random.default_rng(7)
        z = rng.uniform(0.3, 5.0, 20) + 1j * rng.uniform(-4.0, 4.0, 20)
        # Include exactly real impedances so the g2 = 0 limit branch is
        # verified against the reference quadrature too.
        z = np.append(z, [0.7 + 0.0j, 2.0 + 0.0j, 5.0 + 0.0j])
        for lim in (np.pi / 2.0, np.radians(78.0)):
            closed = statistical_absorption(z, angle_limit=lim)
            theta = np.linspace(0.0, lim, 20001)
            g = 1.0 / z[:, None]
            cos = np.cos(theta)[None, :]
            alpha_t = 4.0 * g.real * cos / ((cos + g.real) ** 2 + g.imag**2)
            integrand = alpha_t * np.cos(theta) * np.sin(theta)
            numeric = 2.0 * np.trapezoid(integrand, theta, axis=1) / np.sin(lim) ** 2
            np.testing.assert_allclose(closed, numeric, atol=1e-6)

    def test_statistical_absorption_real_impedance_branch(self) -> None:
        """The printed g2 = 0 special case agrees with g2 -> 0.

        The stabilised arctan-difference form (a single arctan of
        ``g2 (a - b)/(g2^2 + a b)``) must stay continuous down to the
        underflow edge; Mechel's printed two-arctan difference loses
        ~7e-4 already at g2 = 1e-12.
        """
        exact = statistical_absorption(np.array([2.0 + 0.0j]))
        for g2 in (1e-9, 1e-12, 1e-15, -1e-12):
            near = statistical_absorption(np.array([2.0 + 1j * g2]))
            np.testing.assert_allclose(exact, near, atol=1e-9)

    def test_statistical_absorption_real_impedance_closed_form(self) -> None:
        """Exactly real z against the classical g2 = 0 closed form.

        Integrating ``alpha(theta) = 4 xi cos(theta) / (1 + xi cos(theta))^2``
        over the Paris weight gives, for real ``z = xi``,
        ``alpha_dif = (8/xi)(1 + 1/(1 + xi)) - (16/xi^2) ln(1 + xi)``
        (the printed g2 = 0 special case of Mechel D.5 Eq. (10); also
        Kuttruff, *Room Acoustics*). Pins the exact-zero-imaginary-part
        branch independently of the complex formula.
        """
        xi = np.array([0.7, 1.0, 2.0, 5.0])
        expected = (8.0 / xi) * (1.0 + 1.0 / (1.0 + xi)) - (16.0 / xi**2) * np.log(
            1.0 + xi
        )
        alpha = statistical_absorption(xi.astype(complex))
        np.testing.assert_allclose(alpha, expected, rtol=1e-13)

    def test_published_maximum(self) -> None:
        """Mechel D.5: max alpha_dif over locally reacting planes = 0.951."""
        z = np.linspace(1.0, 3.0, 2001)
        alpha = statistical_absorption(z.astype(complex))
        assert np.max(alpha) == pytest.approx(
            ref.POROUS_STATISTICAL_ALPHA_MAX, abs=5e-4
        )

    def test_diffuse_field_bounds_and_value(self) -> None:
        f = np.array([250.0, 500.0, 1000.0, 2000.0])
        med = delany_bazley(f, 20000.0, air_density=RHO0)
        res = diffuse_field_absorption(f, [PorousLayer(0.05, med)])
        assert np.all(res.absorption >= 0.0)
        assert np.all(res.absorption <= 1.0)
        # Thick porous layer: diffuse alpha exceeds normal incidence at mid f.
        normal = layered_absorber(f, [PorousLayer(0.05, med)]).absorption
        assert res.absorption[1] > normal[1]

    def test_diffuse_field_quadrature_converges(self) -> None:
        f = np.array([500.0])
        med = miki(f, 15000.0)
        a64 = diffuse_field_absorption(f, [PorousLayer(0.05, med)]).absorption
        a128 = diffuse_field_absorption(
            f, [PorousLayer(0.05, med)], quadrature_points=128
        ).absorption
        np.testing.assert_allclose(a64, a128, atol=1e-8)

    @pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
    def test_rejects_bad_inputs(self) -> None:
        f = np.array([500.0])
        med = miki(f, 15000.0)
        layers = [PorousLayer(0.05, med)]
        with pytest.raises(ValueError, match="angle_limit"):
            diffuse_field_absorption(f, layers, angle_limit=2.0)
        with pytest.raises(ValueError, match="quadrature_points"):
            diffuse_field_absorption(f, layers, quadrature_points=1)
        negative_real = np.array([-1.0 + 0.5j])
        with pytest.raises(ValueError, match="real part"):
            statistical_absorption(negative_real)
