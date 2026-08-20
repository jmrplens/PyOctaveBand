#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Slow-sound slit + Helmholtz-resonator perfect absorbers (Jimenez et al.).

Oracle strategy (clean-room from the documented transfer-matrix formulas of
Jimenez, Groby, Pagneux and Romero-Garcia, Appl. Sci. 2017, 7, 618, and the
resonator model of Jimenez et al., Appl. Phys. Lett. 2016, 109, 121902):

- The visco-thermal effective parameters are pinned to their exact analytic
  limits: the effective density tends to ``rho0`` and the bulk modulus to
  ``kappa0`` as the boundary layers vanish, and ``j w rho`` tends to the
  Poiseuille flow resistivity of the channel as ``w -> 0``
  (``12 eta / h^2`` for the slit, ``28.454 eta / w^2`` for a square duct).
  The square-duct series is cross-checked against the slit ``tanh`` model in
  the wide-duct limit.
- The critical-coupling design helper is the strongest self-check: it places
  the reflection zero on the real-frequency axis, so the modelled absorption
  at the design frequency must equal 1 (perfect absorption). This is exact by
  construction and independent of any digitized figure.
- Energy and passivity: ``0 <= alpha <= 1`` and ``|R| <= 1`` over randomized
  geometries on a dense grid; the resonator impedance reduces to the
  loss-free lumped Helmholtz limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.materials.absorbers.porous import AirProperties
from phonometry.materials.absorbers.slow_sound import (
    CriticalCouplingResult,
    HelmholtzResonator,
    SlitResonatorAbsorberResult,
    SlowSoundAbsorberWarning,
    critical_coupling_design,
    helmholtz_resonator_impedance,
    rectangular_duct_properties,
    slit_effective_properties,
    slit_helmholtz_absorber,
)

_ETA = 1.84e-5
_RHO0 = 1.205
_C0 = 343.0
_GAMMA = 1.4
_P0 = 101325.0
_AIR = AirProperties(
    speed_of_sound=_C0,
    density=_RHO0,
    viscosity=_ETA,
    heat_capacity_ratio=_GAMMA,
    atmospheric_pressure=_P0,
)


def _base_resonator() -> HelmholtzResonator:
    return HelmholtzResonator(
        neck_length=1.0e-3,
        neck_side=3.0e-3,
        cavity_length=30.0e-3,
        cavity_side=27.0e-3,
    )


# ---------------------------------------------------------------------------
# Visco-thermal effective-parameter anchors
# ---------------------------------------------------------------------------
def test_slit_dc_flow_resistivity() -> None:
    """j w rho_s -> 12 eta / h^2 (parallel-plate Poiseuille) as w -> 0."""
    h = 1.2e-3
    f = np.array([1.0e-2])
    rho_s, _ = slit_effective_properties(f, slit_height=h, air=_AIR)
    sigma = float((1j * 2.0 * np.pi * f * rho_s)[0].real)
    assert sigma == pytest.approx(12.0 * _ETA / h**2, rel=1e-4)


def test_slit_high_frequency_limits() -> None:
    """rho_s -> rho0 and kappa_s -> kappa0 as the boundary layers vanish."""
    f = np.array([2.0e4])
    rho_s, kap_s = slit_effective_properties(f, slit_height=5.0e-2, air=_AIR)
    assert rho_s[0].real == pytest.approx(_RHO0, rel=2e-2)
    assert kap_s[0].real == pytest.approx(_GAMMA * _P0, rel=2e-2)


def test_square_duct_dc_flow_resistivity() -> None:
    """j w rho -> 28.454 eta / side^2 (square-duct Poiseuille) as w -> 0."""
    side = 3.0e-3
    f = np.array([1.0e-2])
    rho, _ = rectangular_duct_properties(f, side=side, air=_AIR)
    sigma = float((1j * 2.0 * np.pi * f * rho)[0].real)
    assert sigma == pytest.approx(28.454 * _ETA / side**2, rel=1e-3)


def test_square_duct_high_frequency_limits() -> None:
    """rho -> rho0 and kappa -> kappa0 for a wide duct at high frequency."""
    f = np.array([5.0e4])
    rho, kap = rectangular_duct_properties(
        f,
        side=5.0e-2,
        air=_AIR,
        sum_terms=120,
    )
    assert rho[0].real == pytest.approx(_RHO0, rel=1e-2)
    assert kap[0].real == pytest.approx(_GAMMA * _P0, rel=1e-2)


def test_square_duct_passive_convention() -> None:
    """The duct is passive in e^{+jwt}: Im(k) < 0 and Re(Z) > 0."""
    f = np.linspace(100.0, 3000.0, 40)
    rho, kap = rectangular_duct_properties(f, side=4.0e-3)
    k = 2.0 * np.pi * f * np.sqrt(rho / kap)
    z = np.sqrt(kap * rho)
    assert np.all(k.imag < 0.0)
    assert np.all(k.real > 0.0)
    assert np.all(z.real > 0.0)


def test_duct_reduces_to_slit_in_wide_limit() -> None:
    """A duct with one very wide side matches the slit tanh model.

    Rectangular Eqs. (7)-(8) with b >> a and a = h must reproduce the
    parallel-plate Eq. (6). Uses a moderate wide side so the transverse
    series over the wide axis stays converged with a finite term count.
    """
    h = 3.0e-3
    f = np.array([1500.0])
    rho_s, kap_s = slit_effective_properties(f, slit_height=h)
    # Rebuild the rectangular series directly with a != b (b wide), on the
    # same air the model defaults to.
    air = AirProperties()
    idx = np.arange(400)
    a2 = ((2 * idx + 1) * np.pi / h) ** 2
    b2 = ((2 * idx + 1) * np.pi / 0.08) ** 2
    ak2, bm2 = a2[:, None], b2[None, :]
    base = ak2 * bm2
    sab = ak2 + bm2
    w = 2 * np.pi * f[0]
    g2r = 1j * w * _RHO0 / air.viscosity
    g2k = 1j * w * air.prandtl_number * _RHO0 / air.viscosity
    sr = np.sum(1.0 / (base * (sab - g2r)))
    sk = np.sum(1.0 / (base * (sab - g2k)))
    rho_d = np.conj(-_RHO0 * h**2 * 0.08**2 / (64.0 * g2r * sr))
    kap_d = np.conj(
        air.heat_capacity_ratio
        * air.atmospheric_pressure
        / (
            air.heat_capacity_ratio
            + (64.0 * (air.heat_capacity_ratio - 1.0) * g2k / (h**2 * 0.08**2)) * sk
        )
    )
    assert rho_d == pytest.approx(rho_s[0], rel=3e-2)
    assert kap_d == pytest.approx(kap_s[0], rel=3e-2)


# ---------------------------------------------------------------------------
# Helmholtz-resonator impedance
# ---------------------------------------------------------------------------
def test_resonator_impedance_lumped_helmholtz_limit() -> None:
    """Loss-free resonance matches the lumped Helmholtz frequency.

    With visco-thermal losses switched off (large channels so the boundary
    layers are negligible) the reactance ``Im(Z_HR)`` crosses zero near the
    lumped ``f0 = (c0 / 2 pi) sqrt(S_n / (V_c (l_n + neck end corrections)))``.
    """
    res = HelmholtzResonator(
        neck_length=5.0e-3,
        neck_side=6.0e-3,
        cavity_length=40.0e-3,
        cavity_side=30.0e-3,
    )
    f = np.linspace(150.0, 900.0, 4000)
    z = helmholtz_resonator_impedance(
        f,
        res,
        air=AirProperties(density=_RHO0, viscosity=_ETA),
    )
    # resonance = reactance sign change
    react = z.imag
    cross = np.where(np.sign(react[:-1]) != np.sign(react[1:]))[0]
    assert cross.size >= 1
    f_res = float(f[cross[0]])
    sn = res.neck_side**2
    vc = res.cavity_side**2 * res.cavity_length
    # neck end corrections (same area-equivalent radii as the model)
    from phonometry.materials.absorbers import slow_sound as ss

    dl = ss._neck_cavity_correction(res.neck_side, res.cavity_side)
    leff = res.neck_length + dl
    f0 = (_C0 / (2.0 * np.pi)) * np.sqrt(sn / (vc * leff))
    assert f_res == pytest.approx(f0, rel=0.12)


# ---------------------------------------------------------------------------
# Panel prediction: passivity and shapes
# ---------------------------------------------------------------------------
def test_panel_absorption_bounds_and_shapes() -> None:
    f = np.linspace(100.0, 900.0, 200)
    res = _base_resonator()
    out = slit_helmholtz_absorber(
        f,
        res,
        slit_height=1.0e-3,
        lattice_step=3.0e-2,
        period=5.0e-2,
    )
    assert isinstance(out, SlitResonatorAbsorberResult)
    assert out.absorption.shape == f.shape
    assert out.transfer_matrix.shape == (2, 2, f.size)
    assert np.all(out.absorption >= 0.0)
    assert np.all(out.absorption <= 1.0)
    assert np.all(np.abs(out.reflection) <= 1.0 + 1e-9)


def test_panel_optional_correction_branches_stay_passive() -> None:
    """The end_correction=False and slit_radiation=False branches stay passive.

    The default path (both True) is covered above; disabling each correction
    exercises the alternate branches in slit_helmholtz_absorber and, for the
    end correction, in helmholtz_resonator_impedance, and must still yield a
    physical (passive, bounded) absorber.
    """
    f = np.linspace(100.0, 900.0, 128)
    res = _base_resonator()
    for end_correction, slit_radiation in (
        (False, True),
        (True, False),
        (False, False),
    ):
        out = slit_helmholtz_absorber(
            f,
            res,
            slit_height=1.0e-3,
            lattice_step=3.0e-2,
            period=5.0e-2,
            end_correction=end_correction,
            slit_radiation=slit_radiation,
        )
        assert out.absorption.shape == f.shape
        assert np.all(out.absorption >= 0.0)
        assert np.all(out.absorption <= 1.0 + 1e-9)
        assert np.all(np.abs(out.reflection) <= 1.0 + 1e-9)


def test_slit_radiation_correction_lowers_resonance() -> None:
    """The slit-radiation end correction is an added mass: it lowers the peak.

    The papers print the series impedance as ``-i w dl rho0 / (phi S0)``;
    conjugated to the e^{+j w t} convention of this module it is
    ``+j w rho0 dl / (phi S0)``, an added radiation mass, which must move the
    slit-panel resonance down. Hand-derived pin for the base geometry
    (h = 1 mm, a = 30 mm, d = 50 mm): the absorption peak sits at 378.55 Hz
    without the correction and at 370.85 Hz with it (a -7.7 Hz shift).
    Transcribing the printed sign literally would instead raise the peak to
    386.8 Hz, which this test guards against.
    """
    f = np.linspace(300.0, 450.0, 3001)
    res = _base_resonator()
    kwargs = {"slit_height": 1.0e-3, "lattice_step": 3.0e-2, "period": 5.0e-2}
    without = slit_helmholtz_absorber(f, res, slit_radiation=False, **kwargs)
    with_rad = slit_helmholtz_absorber(f, res, slit_radiation=True, **kwargs)
    peak_without = float(f[int(np.argmax(without.absorption))])
    peak_with = float(f[int(np.argmax(with_rad.absorption))])
    # Direction: the radiation mass lowers the resonance.
    assert peak_with < peak_without
    # Pinned values (grid step 0.05 Hz).
    assert peak_without == pytest.approx(378.55, abs=0.1)
    assert peak_with == pytest.approx(370.85, abs=0.1)


def test_panel_passivity_random_geometries() -> None:
    rng = np.random.default_rng(20240724)
    f = np.linspace(80.0, 900.0, 60)
    for _ in range(40):
        h = float(rng.uniform(0.5e-3, 4.0e-3))
        a = float(rng.uniform(8.0e-3, 30.0e-3))
        d = float(rng.uniform(max(1.5 * h, 25.0e-3), 80.0e-3))
        res = HelmholtzResonator(
            neck_length=float(rng.uniform(0.3e-3, 15.0e-3)),
            neck_side=float(rng.uniform(1.5e-3, 0.7 * a)),
            cavity_length=float(rng.uniform(8.0e-3, 60.0e-3)),
            cavity_side=float(rng.uniform(0.4 * a, 0.9 * a)),
        )
        out = slit_helmholtz_absorber(
            f,
            res,
            slit_height=h,
            lattice_step=a,
            period=d,
            angle=0.35,
        )
        assert np.all(out.absorption >= -1e-9)
        assert np.all(out.absorption <= 1.0 + 1e-9)
        assert np.all(np.abs(out.reflection) <= 1.0 + 1e-9)


def test_panel_accepts_multiple_resonators() -> None:
    f = np.linspace(100.0, 800.0, 120)
    res = [_base_resonator(), _base_resonator()]
    out = slit_helmholtz_absorber(
        f,
        res,
        slit_height=1.2e-3,
        lattice_step=1.2e-2,
        period=7.0e-2,
    )
    # slit depth L = N * a is reflected in the effective wavenumber array
    assert out.effective_wavenumber.shape == f.shape
    assert np.all(out.absorption <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# Critical coupling: the analytic perfect-absorption anchor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "f0,angle", [(250.0, 0.0), (300.0, 0.0), (350.0, np.radians(20.0))]
)
def test_critical_coupling_gives_perfect_absorption(f0: float, angle: float) -> None:
    """The designed geometry yields alpha ~ 1 at the design frequency."""
    res = _base_resonator()
    design = critical_coupling_design(
        f0,
        res,
        lattice_step=3.0e-2,
        period=5.0e-2,
        angle=angle,
    )
    assert isinstance(design, CriticalCouplingResult)
    assert design.converged
    assert design.absorption == pytest.approx(1.0, abs=1e-3)
    assert abs(design.normalized_impedance) == pytest.approx(1.0, abs=1e-3)
    # the full panel must reproduce alpha ~ 1 at f0 with the designed geometry
    out = slit_helmholtz_absorber(
        np.array([f0]),
        design.resonator,
        slit_height=design.slit_height,
        lattice_step=3.0e-2,
        period=5.0e-2,
        angle=angle,
    )
    assert float(out.absorption[0]) == pytest.approx(1.0, abs=1e-3)


def test_critical_coupling_peak_at_design_frequency() -> None:
    """The absorption spectrum peaks at (near) the design frequency."""
    res = _base_resonator()
    f0 = 300.0
    design = critical_coupling_design(f0, res, lattice_step=3.0e-2, period=5.0e-2)
    f = np.linspace(200.0, 450.0, 1000)
    out = slit_helmholtz_absorber(
        f,
        design.resonator,
        slit_height=design.slit_height,
        lattice_step=3.0e-2,
        period=5.0e-2,
    )
    f_peak = float(f[int(np.argmax(out.absorption))])
    assert f_peak == pytest.approx(f0, abs=8.0)


def test_critical_coupling_warns_when_infeasible() -> None:
    """Impossible bounds raise a warning and report converged=False."""
    res = _base_resonator()
    with pytest.warns(SlowSoundAbsorberWarning):
        design = critical_coupling_design(
            300.0,
            res,
            lattice_step=3.0e-2,
            period=5.0e-2,
            cavity_length_bounds=(1.0e-3, 1.2e-3),
            slit_height_bounds=(0.5e-3, 0.6e-3),
        )
    assert not design.converged


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_invalid_inputs() -> None:
    res = _base_resonator()
    f = np.array([300.0])
    with pytest.raises(ValueError):
        slit_helmholtz_absorber(f, [], slit_height=1e-3, lattice_step=3e-2, period=5e-2)
    with pytest.raises(ValueError):
        slit_helmholtz_absorber(
            f, res, slit_height=6e-2, lattice_step=3e-2, period=5e-2
        )
    with pytest.raises(ValueError):
        slit_helmholtz_absorber(
            f, res, slit_height=1e-3, lattice_step=3e-2, period=5e-2, angle=np.pi / 2.0
        )
