#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the Biot poroelastic layer (Allard & Atalla 2e, ch. 6 and 11).

**Honest statement of the oracle.** Allard & Atalla contains exactly one table
of computed numbers in the whole book (Table 10.2, printed p. 234, Rayleigh
waves in a transversely isotropic frame in vacuum). Every prediction-versus-
measurement pair of the poroelastic chapters, Figures 6.6, 6.7, 6.10, 6.11 and
the whole of chapter 11 included, is a *figure*, and a search of the wider
literature found no published table of ``Zs(f)`` or ``alpha(f)`` for a fully
specified Biot layer anywhere. Published implementations exist, but their
output is not admissible as an oracle here. The model is therefore anchored on
four kinds of evidence, in decreasing strength:

1. **Exact limits against already-anchored code.** As the frame is made
   infinitely stiff and heavy the layer must converge on the
   Johnson-Champoux-Allard equivalent fluid, whose own conformance is pinned on
   published digits; as the frame stiffness is taken to zero it must converge
   on :func:`~phonometry.materials.absorbers.porous.limp_frame`. Both are tested
   as *convergence* over a swept frequency range and a swept stiffness, not as
   spot checks, and both are first order in the swept parameter.
2. **Two independent derivations of the same physics.** The chapter 6 closed
   form for a hard-backed layer at normal incidence, Eqs. (6.107)-(6.108)
   (printed p. 128), is written against the four compressional amplitudes and
   the boundary conditions by hand; the chapter 11 global-matrix assembly goes
   through the six-variable ``[Gamma]`` of Table 11.1, the coupling matrices of
   Sect. 11.4 and a linear solve. They share no code path beyond
   :func:`~phonometry.materials.absorbers.biot.biot_waves` and agree to machine
   precision.
3. **Three printed output digits** for the fully specified glass wool of
   Table 6.1 (printed p. 124), the only "computed" numbers the Biot chapter
   publishes: the airborne wave changes branch from ``(delta1, mu1)`` to
   ``(delta2, mu2)`` at **495 Hz**, ``|mu_a| > 40`` above 50 Hz, and
   ``mu_b`` runs from **1,0 at 50 Hz to 0,82 at 1500 Hz** (all printed
   pp. 124-125). These are genuine independent oracles: they come out of the
   book's own run of the full Biot model on a material whose every parameter
   is printed. The third is matched by ``Re(mu_b)``, not by ``|mu_b|``, even
   though the printed sentence says "modulus"; see ``docs/ERRATA.md``.
4. **The published resonance frequencies**: the surface-impedance peak that no
   equivalent-fluid model can produce sits "around 470 Hz" for a 10 cm layer
   and at **860 Hz** for a 5,6 cm layer (printed p. 129), against the closed
   form Eq. (6.110) which puts the frame ``lambda/4`` resonance at 459,9 Hz and
   821,3 Hz respectively.

What is **not** anchored on digits: the absolute level of ``Zs(f)`` away from
those features, the oblique-incidence behaviour (only its rigid-frame limit is
checked), and the shear wave beyond its role in the transfer matrix.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from phonometry import (
    AirLayer,
    MembraneLayer,
    PoroelasticLayer,
    PorousLayer,
    PorousMediumResult,
    biot_surface_impedance,
    biot_waves,
    frame_bulk_modulus,
    frame_elastic_coefficient,
    frame_quarter_wave_resonance,
    johnson_champoux_allard,
    layered_absorber,
    limp_frame,
    poroelastic_transfer_matrix,
)

# The private [Gamma] of A&A Table 11.1: the field-versus-amplitude matrix the
# transfer matrix is built from, checked directly against the equations it
# transcribes in test_gamma_matches_the_field_rebuilt_from_the_potentials, and
# the [Ipp] interface of Eq. (11.67), checked against the continuity conditions
# of Eq. (11.65). The module itself is imported for the block budget.
from phonometry.materials.absorbers import biot as biot_module
from phonometry.materials.absorbers.biot import _gamma, _porous_porous_matrix

# ---------------------------------------------------------------------------
# A&A Table 6.1 (printed p. 124): glass wool "Domisol Coffrage", with the
# characteristic lengths the book derives in prose on printed p. 123 from
# Eqs. (5.29)-(5.30) (fibre diameter 12 um, Lambda = 0,56e-4 m,
# Lambda' = 2 Lambda). The shear modulus is printed in N/cm2.
# ---------------------------------------------------------------------------
TABLE_6_1_TORTUOSITY = 1.06
TABLE_6_1_FRAME_DENSITY = 130.0
TABLE_6_1_RESISTIVITY = 40_000.0
TABLE_6_1_POROSITY = 0.94
TABLE_6_1_SHEAR_MODULUS = 220.0e4 * (1.0 + 0.1j)
TABLE_6_1_POISSON_RATIO = 0.0
TABLE_6_1_VISCOUS_LENGTH = 0.56e-4
TABLE_6_1_THERMAL_LENGTH = 1.1e-4

#: A&A Table 11.2 (printed p. 254): soft fibrous material, 50 mm thick, the
#: input set behind Figure 11.2 and the material of the limp-frame tests.
TABLE_11_2 = {
    "porosity": 0.98,
    "tortuosity": 1.02,
    "viscous_length": 90e-6,
    "thermal_length": 180e-6,
}
TABLE_11_2_RESISTIVITY = 25.0e3
TABLE_11_2_FRAME_DENSITY = 30.0
TABLE_11_2_THICKNESS = 0.050


def _glass_wool_medium(frequency: np.ndarray) -> PorousMediumResult:
    """Rigid-frame JCA equivalent fluid of the A&A Table 6.1 glass wool."""
    return johnson_champoux_allard(
        frequency,
        TABLE_6_1_RESISTIVITY,
        porosity=TABLE_6_1_POROSITY,
        tortuosity=TABLE_6_1_TORTUOSITY,
        viscous_length=TABLE_6_1_VISCOUS_LENGTH,
        thermal_length=TABLE_6_1_THERMAL_LENGTH,
    )


def _glass_wool_waves(frequency: np.ndarray):
    return biot_waves(
        _glass_wool_medium(frequency),
        porosity=TABLE_6_1_POROSITY,
        tortuosity=TABLE_6_1_TORTUOSITY,
        frame_density=TABLE_6_1_FRAME_DENSITY,
        shear_modulus=TABLE_6_1_SHEAR_MODULUS,
        poisson_ratio=TABLE_6_1_POISSON_RATIO,
    )


def _glass_wool_layer(
    frequency: np.ndarray, thickness: float, *, scale: float = 1.0
) -> PoroelasticLayer:
    return PoroelasticLayer(
        thickness,
        _glass_wool_medium(frequency),
        TABLE_6_1_POROSITY,
        TABLE_6_1_TORTUOSITY,
        TABLE_6_1_FRAME_DENSITY * scale,
        TABLE_6_1_SHEAR_MODULUS * scale,
        TABLE_6_1_POISSON_RATIO,
    )


def _closed_form_resonance(thickness: float) -> float:
    """Eq. (6.110) on the Table 6.1 glass wool."""
    return frame_quarter_wave_resonance(
        thickness,
        shear_modulus=TABLE_6_1_SHEAR_MODULUS,
        poisson_ratio=TABLE_6_1_POISSON_RATIO,
        frame_density=TABLE_6_1_FRAME_DENSITY,
    )


def _impedance_peak(thickness: float) -> float:
    """Frequency of the ``Im(Zs)`` peak of a hard-backed glass-wool layer.

    Searched in the octave-wide window ``[0,6 fr, 1,5 fr]`` around the closed
    form Eq. (6.110), because ``Im(Zs)`` also rises monotonically towards high
    frequency once the layer becomes acoustically thick.
    """
    resonance = _closed_form_resonance(thickness)
    frequency = np.arange(0.6 * resonance, 1.5 * resonance, 0.25)
    impedance = biot_surface_impedance(_glass_wool_waves(frequency), thickness)
    return float(frequency[int(np.argmax(impedance.imag))])


# ---------------------------------------------------------------------------
# Closed forms for the frame in vacuum
# ---------------------------------------------------------------------------
def test_frame_moduli_match_the_printed_closed_forms() -> None:
    """A&A Eqs. (6.29) and (6.111), transcribed from the printed pages.

    ``Kb = 2 N (nu + 1)/(3(1 - 2 nu))`` (printed p. 116) and
    ``Kc = 2(1 - nu) N/(1 - 2 nu)`` (printed p. 130), which must also satisfy
    the definition ``Kc = Kb + 4 N/3`` of Eq. (1.76).
    """
    n_mod = 3.0e5 * (1.0 + 0.07j)
    for nu in (-0.2, 0.0, 0.3, 0.45):
        k_b = frame_bulk_modulus(n_mod, nu)
        k_c = frame_elastic_coefficient(n_mod, nu)
        assert k_b == pytest.approx(
            2.0 * n_mod * (nu + 1.0) / (3.0 * (1.0 - 2.0 * nu)), rel=1e-14
        )
        assert k_c == pytest.approx(
            2.0 * (1.0 - nu) * n_mod / (1.0 - 2.0 * nu), rel=1e-14
        )
        assert k_c == pytest.approx(k_b + 4.0 * n_mod / 3.0, rel=1e-13)


def test_frame_elastic_coefficient_of_the_table_6_1_glass_wool() -> None:
    """With ``nu = 0`` the printed forms collapse to ``Kc = 2 N``.

    A&A Table 6.1 prints ``N = 220 (1 + j 0,1)`` N/cm2 and ``nu = 0``, so
    ``Re(Kc) = 2 x 2,2e6 = 4,4e6`` Pa: pure arithmetic on the printed values.
    """
    k_c = frame_elastic_coefficient(
        TABLE_6_1_SHEAR_MODULUS, TABLE_6_1_POISSON_RATIO
    )
    assert k_c.real == pytest.approx(4.4e6, rel=1e-14)


def test_frame_quarter_wave_resonance_allard_eq_6_110() -> None:
    """A&A Eq. (6.110), printed p. 129: ``fr = (1/4l) sqrt(Re(Kc)/rho1)``.

    On the Table 6.1 glass wool, ``sqrt(4,4e6/130) = 183,97`` m/s, so a 10 cm
    layer resonates at **459,9 Hz** and a 5,6 cm layer at 821,3 Hz. The
    arithmetic is pure and the two are the closed-form anchors of this module.
    """
    assert frame_quarter_wave_resonance(
        0.10,
        shear_modulus=TABLE_6_1_SHEAR_MODULUS,
        poisson_ratio=TABLE_6_1_POISSON_RATIO,
        frame_density=TABLE_6_1_FRAME_DENSITY,
    ) == pytest.approx(459.9, abs=0.05)
    assert frame_quarter_wave_resonance(
        0.056,
        shear_modulus=TABLE_6_1_SHEAR_MODULUS,
        poisson_ratio=TABLE_6_1_POISSON_RATIO,
        frame_density=TABLE_6_1_FRAME_DENSITY,
    ) == pytest.approx(821.3, abs=0.05)


def test_frame_quarter_wave_resonance_scales_as_the_closed_form() -> None:
    """``fr`` is inverse in ``l``, and goes as ``sqrt(Kc/rho1)``."""
    kwargs = {
        "shear_modulus": TABLE_6_1_SHEAR_MODULUS,
        "poisson_ratio": TABLE_6_1_POISSON_RATIO,
        "frame_density": TABLE_6_1_FRAME_DENSITY,
    }
    base = frame_quarter_wave_resonance(0.10, **kwargs)
    assert frame_quarter_wave_resonance(0.05, **kwargs) == pytest.approx(
        2.0 * base
    )
    quadrupled = dict(kwargs, shear_modulus=4.0 * TABLE_6_1_SHEAR_MODULUS)
    assert frame_quarter_wave_resonance(0.10, **quadrupled) == pytest.approx(
        2.0 * base
    )
    heavier = dict(kwargs, frame_density=4.0 * TABLE_6_1_FRAME_DENSITY)
    assert frame_quarter_wave_resonance(0.10, **heavier) == pytest.approx(
        0.5 * base
    )


# ---------------------------------------------------------------------------
# The three waves: the printed output digits of A&A Sect. 6.5.4
# ---------------------------------------------------------------------------
def test_airborne_branch_swaps_at_the_published_495_hz() -> None:
    """A&A printed p. 124, on the Table 6.1 glass wool.

    "At high frequencies, for the airborne wave, ... the wave number is
    ``delta2`` given by Equation (6.68). At frequencies lower than **495 Hz**,
    the airborne wave is related to ``mu1`` and ``delta1``."

    This is one of only three computed numbers the Biot chapter prints, and it
    is a three-digit output of the full model on a fully specified material.
    The tolerance is 1 % of the printed value.
    """
    grid = np.arange(300.0, 700.0, 0.1)
    waves = _glass_wool_waves(grid)
    swaps = np.flatnonzero(np.diff(waves.airborne_is_second.astype(int)) != 0)
    assert swaps.size == 1
    crossing = 0.5 * (grid[swaps[0]] + grid[swaps[0] + 1])
    assert crossing == pytest.approx(495.0, rel=0.01)
    # Below the crossing the airborne wave is delta1, above it delta2.
    assert not bool(waves.airborne_is_second[0])
    assert bool(waves.airborne_is_second[-1])


def test_airborne_velocity_ratio_exceeds_the_published_forty() -> None:
    """A&A printed p. 124: "|mu_a| is larger than 40 for f > 50 Hz".

    ``mu`` is the ratio of the fluid velocity to the frame velocity
    (Eq. (6.71)), so a large ``|mu_a|`` is what makes the airborne wave behave
    like the wave of a rigid-frame material.
    """
    frequency = np.geomspace(50.0, 1500.0, 60)
    waves = _glass_wool_waves(frequency)
    assert float(np.min(np.abs(waves.airborne_velocity_ratio))) > 40.0


def test_frame_borne_velocity_ratio_matches_the_two_published_values() -> None:
    """A&A printed p. 125, read as ``Re(mu_b)``, not as ``|mu_b|``.

    The printed sentence is "the ratio **modulus** ``|mu_b|`` of the velocities
    of the frame and the air for the frame-borne wave decreases from 1,0 at
    50 Hz to 0,82 at 1500 Hz", but the two values it quotes are the real part:
    the model gives ``mu_b(1500) = 0,811 + 0,473 j``, whose real part is 1,1 %
    from 0,82 while its modulus, 0,939, is 14,5 % away. ``docs/ERRATA.md``
    records the sentence and the evidence, including the parameter sweep that
    fails to bring ``|mu_b|`` anywhere near 0,82. The assertions here are
    therefore written against ``Re(mu_b)`` and the test also states what
    ``|mu_b|`` really does, so a future change cannot quietly reinterpret it.

    The frame-borne wave drags a comparable amount of air with it, which is
    exactly what an equivalent fluid cannot represent. The book prints one
    decimal at 50 Hz and two at 1500 Hz; it does not state the air constants it
    used, so the comparison carries a 2 % tolerance.
    """
    waves = _glass_wool_waves(np.array([50.0, 1500.0]))
    ratio = waves.frame_borne_velocity_ratio
    assert float(ratio[0].real) == pytest.approx(1.0, rel=0.02)
    assert float(ratio[1].real) == pytest.approx(0.82, rel=0.02)
    assert float(ratio[1].real) < float(ratio[0].real)
    # The modulus reading of the same sentence, pinned so the erratum stays
    # honest: it is nowhere near 0,82 at 1500 Hz.
    assert abs(ratio[1]) == pytest.approx(0.939, rel=0.01)
    assert abs(abs(ratio[1]) / 0.82 - 1.0) > 0.1


def test_velocity_ratios_satisfy_the_second_printed_form() -> None:
    """A&A prints ``mu_i`` twice, Eq. (6.71) and Eq. (6.72), printed p. 121.

    Only Eq. (6.71) is implemented; Eq. (6.72),
    ``mu_i = (Q delta_i^2 - w^2 rho12)/(w^2 rho22 - R delta_i^2)``, is an
    algebraically equivalent form that holds only if ``delta_i^2`` really is a
    root of the characteristic equation, so it checks the eigenvalue solve as
    well as the ratio.
    """
    waves = _glass_wool_waves(np.geomspace(20.0, 5000.0, 40))
    omega = 2.0 * np.pi * waves.frequency
    for delta, mu in (
        (waves.compressional_wavenumber_1, waves.velocity_ratio_1),
        (waves.compressional_wavenumber_2, waves.velocity_ratio_2),
    ):
        other = (waves.elastic_q * delta**2 - omega**2 * waves.density_12) / (
            omega**2 * waves.density_22 - waves.elastic_r * delta**2
        )
        assert np.allclose(mu, other, rtol=1e-9)


def test_compressional_wavenumbers_solve_the_characteristic_equation() -> None:
    """A&A Eq. (6.65): ``delta^2`` are the eigenvalues of ``w^2 [M]^-1 [rho]``.

    Checked as the two invariants of that 2x2 matrix, so a slip in the
    ``-sqrt(Delta)`` / ``+sqrt(Delta)`` split of Eqs. (6.67)-(6.68) or in
    ``Delta`` itself (Eq. (6.69)) breaks it.
    """
    waves = _glass_wool_waves(np.geomspace(20.0, 5000.0, 40))
    omega = 2.0 * np.pi * waves.frequency
    p_c, q_c, r_c = waves.elastic_p, waves.elastic_q, waves.elastic_r
    r11, r12, r22 = waves.density_11, waves.density_12, waves.density_22
    det_m = p_c * r_c - q_c**2
    d1_sq = waves.compressional_wavenumber_1**2
    d2_sq = waves.compressional_wavenumber_2**2
    trace = omega**2 * (p_c * r22 + r_c * r11 - 2.0 * q_c * r12) / det_m
    product = omega**4 * (r11 * r22 - r12**2) / det_m
    assert np.allclose(d1_sq + d2_sq, trace, rtol=1e-9)
    assert np.allclose(d1_sq * d2_sq, product, rtol=1e-9)


def test_shear_velocity_ratio_matches_the_corrected_second_printed_form() -> None:
    """A&A Eqs. (6.84) and (6.85), printed pp. 122-123, with an erratum.

    Eq. (6.85) prints ``mu3 = (N delta3^2 - w^2 rho11)/(w^2 rho22)``, which
    contradicts Eq. (6.84) ``mu3 = -rho12/rho22`` by a factor
    ``rho12 / rho22``. Eq. (6.80) gives the derivation:
    ``(N delta3^2 - w^2 rho11) psi_s = w^2 rho12 psi_f``, so the denominator is
    ``w^2 rho12``. With that reading the two printed forms agree exactly, and
    they only agree for the ``delta3`` of Eq. (6.83), so this checks the shear
    eigenvalue as well as the ratio. Recorded in ``docs/ERRATA.md``.
    """
    waves = _glass_wool_waves(np.geomspace(20.0, 5000.0, 40))
    omega = 2.0 * np.pi * waves.frequency
    corrected = (
        waves.shear_modulus * waves.shear_wavenumber**2
        - omega**2 * waves.density_11
    ) / (omega**2 * waves.density_12)
    assert np.allclose(waves.velocity_ratio_3, corrected, rtol=1e-9)
    as_printed = (
        waves.shear_modulus * waves.shear_wavenumber**2
        - omega**2 * waves.density_11
    ) / (omega**2 * waves.density_22)
    assert not np.allclose(waves.velocity_ratio_3, as_printed, rtol=1e-3)


def test_modified_densities_reproduce_the_printed_sum_rules() -> None:
    """A&A Eqs. (6.37), (6.39) and (6.40), printed p. 118.

    ``rho11 + 2 rho12 + rho22 = rho1 + phi rho0``, ``rho12 + rho22 = phi rho0``
    only for the inviscid part; with the viscous term the sum rules the book
    derives survive as ``rho11 - rho12 = rho1 + rho_a - (- rho_a) - ...``. The
    two that hold verbatim in the frequency domain are
    ``rho11 + rho12 = rho1`` and ``rho12 + rho22 = phi rho0`` (Eqs. (6.40) and
    (6.39) with the viscous term cancelling between the pair).
    """
    waves = _glass_wool_waves(np.geomspace(20.0, 5000.0, 25))
    rho0 = _glass_wool_medium(waves.frequency).air_density
    assert np.allclose(
        waves.density_11 + waves.density_12, TABLE_6_1_FRAME_DENSITY, rtol=1e-12
    )
    assert np.allclose(
        waves.density_12 + waves.density_22,
        TABLE_6_1_POROSITY * rho0,
        rtol=1e-12,
    )


def test_airborne_wave_follows_the_rigid_frame_wavenumber() -> None:
    """A&A printed p. 124: ``ka`` and ``ka'`` "are represented by the same curve".

    Eq. (6.86) gives the rigid-frame airborne wavenumber
    ``ka' = w sqrt(rho22/R)``, which for this parameterisation is exactly the
    Johnson-Champoux-Allard wavenumber. Figure 6.6 plots them on top of each
    other for the Table 6.1 glass wool; here the two agree to better than 2 %
    over the plotted 0 to 1,5 kHz.
    """
    frequency = np.geomspace(50.0, 1500.0, 60)
    waves = _glass_wool_waves(frequency)
    rigid = _glass_wool_medium(frequency).wavenumber
    assert np.max(np.abs(waves.airborne_wavenumber / rigid - 1.0)) < 0.02


# ---------------------------------------------------------------------------
# Surface impedance: the two published resonance frequencies
# ---------------------------------------------------------------------------
def test_impedance_peak_of_the_ten_centimetre_layer() -> None:
    """A&A printed p. 129: the peak sits "around 470 Hz" for ``l = 10 cm``.

    Figure 6.10 shows it as a dip and a peak in ``Re(Z)`` straddling a sharp
    maximum of ``Im(Z)`` just below 0,5 kHz. The closed form Eq. (6.110) puts
    the underlying frame ``lambda/4`` resonance at 459,9 Hz, and the book warns
    that ``delta_b`` is only "very close" to the in-vacuum wavenumber, so the
    computed peak is expected a few per cent above it. Tolerance: 5 % of the
    printed 470 Hz.
    """
    peak = _impedance_peak(0.10)
    assert peak == pytest.approx(470.0, rel=0.05)
    assert peak > _closed_form_resonance(0.10)


def test_impedance_peak_of_the_thin_layer_resolves_the_printed_thickness() -> None:
    """A&A printed p. 129 prints the peak of the second sample at **860 Hz**.

    The same paragraph is internally inconsistent about that sample: its first
    sentence says ``l = 5,4 cm`` and its third says ``l = 5,6 cm``, and the
    caption of Figure 6.11 says 5,6 cm. Computing the peak from the printed
    material settles it: 5,6 cm gives 863,5 Hz (0,4 % from the printed 860 Hz)
    while 5,4 cm gives 896 Hz (4,2 % away). Only the 5,6 cm reading is
    consistent, so it is the one pinned here, to 2 %.
    """
    thin = _impedance_peak(0.056)
    thicker = _impedance_peak(0.054)
    assert thin == pytest.approx(860.0, rel=0.02)
    assert abs(thin - 860.0) < abs(thicker - 860.0)


def test_impedance_peak_tracks_the_frame_resonance_across_thicknesses() -> None:
    """The peak follows ``1/l``, as Eq. (6.110) requires.

    An equivalent fluid produces no such peak at all, so this also checks the
    peak is a frame feature and not a quarter-wavelength of the pore fluid
    (which for this material would scale differently, because the airborne wave
    is far slower and heavily damped). Checked over the range of thicknesses the
    book measured, 5 to 10 cm: the product ``l fr`` is constant to 2 %, and the
    peak sits 4 to 6 % above the closed form throughout.
    """
    lengths = (0.10, 0.056, 0.05)
    products = [length * _impedance_peak(length) for length in lengths]
    assert max(products) / min(products) == pytest.approx(1.0, abs=0.02)
    for length in lengths:
        excess = _impedance_peak(length) / _closed_form_resonance(length) - 1.0
        assert 0.03 < excess < 0.07


def test_rigid_frame_layer_shows_no_impedance_peak() -> None:
    """The peak is a frame resonance: freeze the frame and it disappears.

    With the frame made 10^6 times stiffer and heavier, ``Im(Zs)`` becomes
    monotonic through the band where the peak sat, exactly as the dashed
    rigid-frame curves of Figures 6.10 and 6.11 do.
    """
    frequency = np.arange(300.0, 700.0, 1.0)
    medium = _glass_wool_medium(frequency)
    flexible = layered_absorber(frequency, [_glass_wool_layer(frequency, 0.10)])
    frozen = layered_absorber(
        frequency, [_glass_wool_layer(frequency, 0.10, scale=1e6)]
    )
    rigid = layered_absorber(frequency, [PorousLayer(0.10, medium)])
    assert np.max(np.diff(np.sign(np.diff(frozen.surface_impedance.imag)))) == 0
    assert np.any(np.diff(np.sign(np.diff(flexible.surface_impedance.imag))))
    assert np.allclose(
        frozen.surface_impedance, rigid.surface_impedance, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# Two independent derivations of the same physics
# ---------------------------------------------------------------------------
def test_global_assembly_reproduces_the_chapter_6_closed_form() -> None:
    """A&A Eq. (6.107) (ch. 6) against the Sect. 11.5 assembly (ch. 11).

    Two derivations written a hundred pages apart: the first solves the 3x3
    determinant Eq. (6.106) for the four compressional amplitudes of a glued
    layer, the second builds the six-variable ``[Gamma]`` of Table 11.1, couples
    it to the free air through ``[Ipf]``/``[Jpf]`` (Eq. (11.73)) and closes it
    with ``[Y p]`` (Eq. (11.81)). They share only
    :func:`~phonometry.materials.absorbers.biot.biot_waves`, so agreement at machine
    precision exercises the whole ``[Gamma]``, the coupling matrices and the
    linear solve at once.
    """
    frequency = np.geomspace(20.0, 5000.0, 120)
    waves = _glass_wool_waves(frequency)
    for thickness in (0.02, 0.056, 0.10, 0.25):
        closed = biot_surface_impedance(waves, thickness)
        assembled = layered_absorber(
            frequency, [_glass_wool_layer(frequency, thickness)]
        ).surface_impedance
        assert np.max(np.abs(assembled / closed - 1.0)) < 1e-10


def test_transfer_matrix_composes() -> None:
    """``[T p]`` of Eq. (11.34) is a propagator: ``T(h1) T(h2) = T(h1 + h2)``.

    That identity holds only if the six columns of ``[Gamma]`` are solutions of
    the same constant-coefficient system, so a ``sin``/``cos`` swap or a
    misplaced ``j`` inside a column breaks it. The comparison against the
    assembly is the test above, which checks the closed form of Eq. (6.107)
    against ``layered_absorber`` at four thicknesses.
    """
    frequency = np.geomspace(100.0, 2000.0, 20)
    waves = _glass_wool_waves(frequency)
    k_t = 2.0 * np.pi * frequency / 343.0 * np.sin(0.4)
    t_one = poroelastic_transfer_matrix(waves, 0.02, transverse_wavenumber=k_t)
    t_two = poroelastic_transfer_matrix(waves, 0.03, transverse_wavenumber=k_t)
    t_all = poroelastic_transfer_matrix(waves, 0.05, transverse_wavenumber=k_t)
    assert np.allclose(t_one @ t_two, t_all, rtol=1e-7, atol=1e-7)


def _reference_field(
    waves, x3: float, k_t: float, amplitudes: np.ndarray
) -> np.ndarray:
    """``[v1s, v3s, v3f, s33s, s13s, s33f]`` rebuilt from the potentials.

    An independent construction of what Table 11.1 tabulates, written from the
    *equations* instead: the displacement potentials of Eqs. (11.22)-(11.25),
    the velocity definitions of Eq. (11.27) and the stress-strain relations
    Eqs. (11.39)-(11.41), with every derivative taken analytically
    (``d/dx1 = -j kt`` and ``grad^2 phi_i = -delta_i^2 phi_i``). Nothing here
    passes through the table, so agreement pins every one of its 36 entries,
    including the ``kt`` terms that vanish at normal incidence.
    """
    omega = 2.0 * np.pi * float(waves.frequency[0])
    d1, d2 = (
        complex(waves.compressional_wavenumber_1[0]),
        complex(waves.compressional_wavenumber_2[0]),
    )
    d3 = complex(waves.shear_wavenumber[0])
    mu1, mu2, mu3 = (
        complex(waves.velocity_ratio_1[0]),
        complex(waves.velocity_ratio_2[0]),
        complex(waves.velocity_ratio_3[0]),
    )
    p_c, q_c, r_c = (
        complex(waves.elastic_p[0]),
        complex(waves.elastic_q[0]),
        complex(waves.elastic_r[0]),
    )
    n_mod = waves.shear_modulus
    k13, k23, k33 = (np.sqrt(d**2 - k_t**2) for d in (d1, d2, d3))
    a1p, a1m, a2p, a2m, a3p, a3m = amplitudes

    def potential(ap, am, kx):
        value = ap * np.cos(kx * x3) - 1j * am * np.sin(kx * x3)
        slope = -kx * (ap * np.sin(kx * x3) + 1j * am * np.cos(kx * x3))
        return value, slope

    phi1, dphi1 = potential(a1p, a1m, k13)
    phi2, dphi2 = potential(a2p, a2m, k23)
    psi, dpsi = potential(a3p, a3m, k33)

    div_s = -(d1**2) * phi1 - d2**2 * phi2
    div_f = -mu1 * d1**2 * phi1 - mu2 * d2**2 * phi2
    v1s = 1j * omega * (-1j * k_t * (phi1 + phi2) - dpsi)
    v3s = 1j * omega * (dphi1 + dphi2 - 1j * k_t * psi)
    v3f = 1j * omega * (mu1 * dphi1 + mu2 * dphi2 - 1j * k_t * mu3 * psi)
    du3s_dx3 = -(k13**2) * phi1 - k23**2 * phi2 - 1j * k_t * dpsi
    du1s_dx3 = -1j * k_t * (dphi1 + dphi2) + k33**2 * psi
    du3s_dx1 = -1j * k_t * (dphi1 + dphi2 - 1j * k_t * psi)
    s33s = (p_c - 2.0 * n_mod) * div_s + q_c * div_f + 2.0 * n_mod * du3s_dx3
    s13s = n_mod * (du1s_dx3 + du3s_dx1)
    s33f = r_c * div_f + q_c * div_s
    return np.array([v1s, v3s, v3f, s33s, s13s, s33f])


@pytest.mark.parametrize("angle", [0.0, 0.4, 1.2])
def test_gamma_matches_the_field_rebuilt_from_the_potentials(angle: float) -> None:
    """A&A Table 11.1 against Eqs. (11.22)-(11.28), on the same amplitudes.

    The table is a transcription of those equations, so re-deriving the six
    field variables from the potentials and comparing them entry by entry is
    the sharpest available check of the table: it reaches the ``kt``-weighted
    and shear-column entries that a normal-incidence test can never see.
    """
    rng = np.random.default_rng(20260730)
    for frequency in (120.0, 900.0, 3000.0):
        waves = _glass_wool_waves(np.array([frequency]))
        k_t = 2.0 * np.pi * frequency / 343.0 * np.sin(angle)
        for x3 in (0.0, -0.013, -0.05):
            gamma_x = _gamma(waves, x3, np.asarray(k_t, dtype=np.complex128))
            amplitudes = rng.normal(size=6) + 1j * rng.normal(size=6)
            expected = _reference_field(waves, x3, k_t, amplitudes)
            obtained = gamma_x[0] @ amplitudes
            # Component-by-component and relative, so a spurious term is caught
            # even where it is small against the rest of its own row.
            assert np.allclose(obtained, expected, rtol=1e-9, atol=0.0)


def test_bonded_poroelastic_layers_split_without_changing_the_result() -> None:
    """A&A Eq. (11.67): the ``[Ipp]`` interface of two bonded equal frames.

    Splitting one 10 cm layer into two bonded 5 cm layers of the same material
    must change nothing, which pins ``[Ipp]`` at equal porosity (its unit-matrix
    case) and the block chaining of the assembly.
    """
    frequency = np.geomspace(50.0, 3000.0, 40)
    single = layered_absorber(frequency, [_glass_wool_layer(frequency, 0.10)])
    split = layered_absorber(
        frequency,
        [_glass_wool_layer(frequency, 0.05), _glass_wool_layer(frequency, 0.05)],
    )
    assert np.allclose(
        split.surface_impedance, single.surface_impedance, rtol=1e-9
    )


def test_bonded_interface_satisfies_the_printed_continuity_conditions() -> None:
    """A&A Eq. (11.65), printed p. 257: ``[Ipp]`` at unequal porosity.

    The test above pins ``[Ipp]`` only where the two frames share a porosity,
    which is the unit-matrix case the book itself flags; every off-diagonal
    entry is then zero. Here the matrix is checked against the six continuity
    conditions it is built from, transcribed from the printed equations rather
    than from the printed matrix, on an arbitrary field vector and a porosity
    ratio far from one:

    ``v1s`` and ``v3s`` continuous, ``phi (v3f - v3s)`` continuous (the
    relative volume flow), ``s33s + s33f`` continuous (the total normal
    stress), ``s13s`` continuous and ``s33f / phi`` continuous (the pore
    pressure).
    """
    phi_left, phi_right = 0.62, 0.97
    i_pp = _porous_porous_matrix(phi_left, phi_right)
    # An arbitrary right-hand field vector [v1s, v3s, v3f, s33s, s13s, s33f];
    # [Ipp] maps it onto the left-hand one, V(M2) = [Ipp] V(M3).
    right = np.array(
        [0.7 - 0.3j, -1.9 + 0.4j, 2.6 + 1.1j, -4.5e3 + 2e3j, 8e2 - 5e2j,
         -1.3e3 - 9e2j],
        dtype=np.complex128,
    )
    left = i_pp @ right

    assert left[0] == pytest.approx(right[0])
    assert left[1] == pytest.approx(right[1])
    assert phi_left * (left[2] - left[1]) == pytest.approx(
        phi_right * (right[2] - right[1])
    )
    assert left[3] + left[5] == pytest.approx(right[3] + right[5])
    assert left[4] == pytest.approx(right[4])
    assert left[5] / phi_left == pytest.approx(right[5] / phi_right)
    # The equal-porosity case really is the unit matrix (printed note below
    # Eq. (11.67)), so the check above is the only one that sees the couplings.
    assert np.array_equal(_porous_porous_matrix(0.8, 0.8), np.eye(6))


def test_two_bonded_materials_reduce_to_the_two_layer_equivalent_fluid() -> None:
    """The ``[Ipp]`` couplings survive a real two-material stack.

    Two *different* bonded poroelastic layers, 0,94 and 0,98 porosity on two
    different Johnson-Champoux-Allard media, driven to the rigid-frame limit of
    A&A Sect. 11.3.4. The result must converge on the same two-layer stack
    built from :class:`~phonometry.materials.absorbers.layered.PorousLayer`,
    whose equivalent-fluid path is pinned on published digits and never touches
    ``[Ipp]``. Unlike the equal-porosity split, this exercises the ``phi2/phi1``
    and ``phi1/phi2`` entries of Eq. (11.67), and it is a convergence test: the
    residual must fall by two decades for every two decades of frame scaling.
    """
    frequency = np.geomspace(50.0, 5000.0, 40)
    glass_wool = _glass_wool_medium(frequency)
    fibrous = _soft_fibrous(frequency)
    reference = layered_absorber(
        frequency,
        [PorousLayer(0.04, glass_wool), PorousLayer(0.03, fibrous)],
    ).surface_impedance

    residuals = []
    for scale in (1e2, 1e4, 1e6, 1e8):
        stack = layered_absorber(
            frequency,
            [
                _glass_wool_layer(frequency, 0.04, scale=scale),
                PoroelasticLayer(
                    0.03,
                    fibrous,
                    TABLE_11_2["porosity"],
                    TABLE_11_2["tortuosity"],
                    TABLE_11_2_FRAME_DENSITY * scale,
                    TABLE_6_1_SHEAR_MODULUS * scale,
                    TABLE_6_1_POISSON_RATIO,
                ),
            ],
        )
        residuals.append(
            float(np.max(np.abs(stack.surface_impedance / reference - 1.0)))
        )
    assert residuals[-1] < 1e-8
    # The first scale is not yet in the asymptotic regime, so the pairwise
    # ratio is only first order from the second onwards, and the band is the
    # one the limp-limit test uses for the same reason.
    for coarse, fine in itertools.pairwise(residuals[1:]):
        assert 75.0 < coarse / fine < 125.0


# ---------------------------------------------------------------------------
# Strongly attenuating blocks: the assembly must not lose the stack
# ---------------------------------------------------------------------------
#: A dense fibrous material, 500 kPa s/m2, whose in-depth attenuation reaches
#: 410 nepers per metre at 20 kHz. Any run of it thicker than about 1,7 m
#: overflows a raw ``float64`` chain matrix.
_DENSE_RESISTIVITY = 500.0e3


def _dense_medium(frequency: np.ndarray) -> PorousMediumResult:
    return johnson_champoux_allard(
        frequency,
        _DENSE_RESISTIVITY,
        porosity=TABLE_6_1_POROSITY,
        tortuosity=TABLE_6_1_TORTUOSITY,
        viscous_length=TABLE_6_1_VISCOUS_LENGTH,
        thermal_length=TABLE_6_1_THERMAL_LENGTH,
    )


def _dense_poroelastic(
    frequency: np.ndarray, thickness: float
) -> PoroelasticLayer:
    return PoroelasticLayer(
        thickness,
        _dense_medium(frequency),
        TABLE_6_1_POROSITY,
        TABLE_6_1_TORTUOSITY,
        TABLE_6_1_FRAME_DENSITY,
        TABLE_6_1_SHEAR_MODULUS,
        TABLE_6_1_POISSON_RATIO,
    )


def test_attenuating_fluid_run_in_front_of_a_biot_layer_gives_the_half_space() -> (
    None
):
    """A fluid run that overflows its chain matrix must not poison the solve.

    Two metres of the dense material attenuate by 245 nepers at 2 kHz and by
    821 nepers at 20 kHz, so over the top of the band ``cos(kx d)`` and
    ``sin(kx d)``, of order ``e^{|Im kx| d}``, are past the ``1,8e308`` of
    ``float64``. Nothing behind such a run can be heard, so the oracle is the
    definition of the characteristic impedance: the surface impedance of a
    half-space at normal incidence is ``Zc`` of its medium, to machine
    precision, whatever closes the stack. Before the run was split into blocks
    of bounded attenuation the assembly returned ``inf``/``NaN`` here and
    ``alpha`` collapsed to zero.
    """
    frequency = np.geomspace(2000.0, 20000.0, 12)
    medium = _dense_medium(frequency)
    result = layered_absorber(
        frequency,
        [PorousLayer(2.0, medium), _dense_poroelastic(frequency, 0.05)],
    )
    assert np.all(np.isfinite(result.surface_impedance))
    assert np.allclose(
        result.surface_impedance,
        np.asarray(medium.characteristic_impedance),
        rtol=1e-12,
        atol=0.0,
    )


def test_attenuating_fluid_run_behind_a_biot_layer_acts_as_its_termination() -> (
    None
):
    """The same defect on the other side of the layer, where it bites sooner.

    A fluid run *behind* the poroelastic layer enters the system through the
    porous-fluid coupling of Eq. (11.73), and a single half-metre of the dense
    material was already enough to return a negative absorption coefficient
    for a passive stack. A run this attenuating is a half-space, and a
    half-space of characteristic impedance ``Zc`` behind the layer is exactly
    what ``termination=Zc`` builds through Eqs. (11.84)-(11.85), a branch of
    the assembly that holds no fluid block at all.
    """
    frequency = np.geomspace(2000.0, 20000.0, 12)
    medium = _dense_medium(frequency)
    layer = _dense_poroelastic(frequency, 0.05)
    half_space = layered_absorber(
        frequency, [layer], termination=np.asarray(medium.characteristic_impedance)
    ).surface_impedance
    for thickness in (0.5, 1.0, 2.0):
        backed = layered_absorber(
            frequency, [layer, PorousLayer(thickness, medium)]
        )
        assert np.all(np.isfinite(backed.surface_impedance))
        assert np.all(backed.absorption >= 0.0)
        assert np.allclose(
            backed.surface_impedance, half_space, rtol=1e-11, atol=0.0
        )


def test_thick_poroelastic_layer_hides_whatever_is_behind_it() -> None:
    """The six-variable block has the same dynamic range as the fluid one.

    ``[Gamma(0)]`` is of order one while ``[Gamma(-h)]`` grows as ``e^{|Im k| h}``
    with the most damped Biot wave, so a thick layer put the same two decades
    per row into the system. A 40 cm layer with a 2 cm air gap behind it
    returned a negative absorption; here the air gap must simply become
    invisible, and the residual against the hard-backed stack must fall
    monotonically towards zero as the layer thickens.
    """
    frequency = np.geomspace(2000.0, 20000.0, 12)
    residuals = []
    for thickness in (0.3, 0.6, 1.2, 2.4):
        layer = _dense_poroelastic(frequency, thickness)
        hard = layered_absorber(frequency, [layer]).surface_impedance
        gap = layered_absorber(frequency, [layer, AirLayer(0.02)])
        assert np.all(np.isfinite(gap.surface_impedance))
        assert np.all(gap.absorption >= 0.0)
        assert np.all(gap.absorption <= 1.0)
        residuals.append(float(np.max(np.abs(gap.surface_impedance / hard - 1.0))))
    assert residuals[-1] < 1e-7
    for coarse, fine in itertools.pairwise(residuals):
        assert fine < coarse / 4.0


def test_an_unresolvable_stack_is_refused_instead_of_exhausting_memory() -> None:
    """The block budget must not turn a bad input into an unbounded solve.

    The shear wavenumber of Eq. (6.87) goes as ``omega sqrt(rho_c / N)``, so
    a frame stiffness driven towards zero, the limit the guide invites the
    reader to explore, makes the shear attenuation across even a thin layer
    diverge. Left uncapped the assembly would ask for thousands of blocks and
    die on the allocation; it has to refuse, and say why. The same guard
    covers a fluid run nobody could hear through.
    """
    frequency = np.geomspace(500.0, 5000.0, 10)
    medium = _glass_wool_medium(frequency)
    floppy = PoroelasticLayer(
        0.05,
        medium,
        TABLE_6_1_POROSITY,
        TABLE_6_1_TORTUOSITY,
        TABLE_6_1_FRAME_DENSITY,
        1.0e-4 * (1.0 + 0.1j),
        TABLE_6_1_POISSON_RATIO,
    )
    deaf_run = [
        PorousLayer(400.0, _dense_medium(frequency)),
        _dense_poroelastic(frequency, 0.05),
    ]
    with pytest.raises(ValueError, match="nepers"):
        layered_absorber(frequency, [floppy])
    with pytest.raises(ValueError, match="nepers"):
        layered_absorber(frequency, deaf_run)


def test_splitting_a_block_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cutting the stack into more blocks is algebra, not approximation.

    A homogeneous fluid layer of phase ``kx d`` is the product of ``m`` layers
    of phase ``kx d / m``, and two bonded halves of one poroelastic material
    couple through the unit matrix (Eq. (11.67) at equal porosity), so the
    block budget must not show up in the answer. Driving it from "never split"
    down to one neper has to leave a well-conditioned stack unchanged.
    """
    frequency = np.geomspace(500.0, 8000.0, 20)
    medium = _glass_wool_medium(frequency)
    layers = [
        AirLayer(0.01),
        PorousLayer(0.03, medium),
        _glass_wool_layer(frequency, 0.05),
        AirLayer(0.02),
    ]
    monkeypatch.setattr(biot_module, "_BLOCK_NEPERS", 1.0e9)
    unsplit = layered_absorber(frequency, layers, angle=0.4).surface_impedance
    for budget in (5.0, 2.0, 1.0):
        monkeypatch.setattr(biot_module, "_BLOCK_NEPERS", budget)
        split = layered_absorber(frequency, layers, angle=0.4).surface_impedance
        assert np.allclose(split, unsplit, rtol=1e-11, atol=0.0)


# ---------------------------------------------------------------------------
# The rigid-frame limit: convergence on the already-anchored JCA fluid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [0.0, np.pi / 6.0, np.pi / 4.0, np.pi / 3.0])
def test_rigid_frame_limit_converges_on_the_jca_equivalent_fluid(
    angle: float,
) -> None:
    """The strongest anchor: an exact limit against oracled code.

    A&A Sect. 11.3.4 (printed p. 251): "the rigid frame limit depicts the
    dynamic behaviour of the material when its frame is supposed motionless",
    and it is then represented by the equivalent-fluid matrix Eq. (11.9) with
    ``rho_eq``, ``K_eq`` - which is exactly what
    :class:`~phonometry.materials.absorbers.layered.PorousLayer` builds from a
    :func:`~phonometry.materials.absorbers.porous.johnson_champoux_allard`
    medium, whose conformance is pinned on published digits.

    Making the frame infinitely stiff **and** heavy must therefore drive the
    Biot layer onto that path. This is tested as a genuine convergence: the
    stiffness and density are scaled together over eight decades on a
    50 Hz to 5 kHz sweep, and the residual must fall by a factor of ten for
    every decade of scaling (first order), not merely be small once.
    """
    frequency = np.geomspace(50.0, 5000.0, 60)
    medium = _glass_wool_medium(frequency)
    reference = layered_absorber(
        frequency, [PorousLayer(0.05, medium)], angle=angle
    ).surface_impedance
    residuals = []
    for scale in (1e2, 1e4, 1e6, 1e8):
        result = layered_absorber(
            frequency,
            [_glass_wool_layer(frequency, 0.05, scale=scale)],
            angle=angle,
        )
        residuals.append(
            float(np.max(np.abs(result.surface_impedance / reference - 1.0)))
        )
    assert residuals[-1] < 1e-8
    for coarse, fine in itertools.pairwise(residuals):
        assert fine == pytest.approx(coarse / 100.0, rel=0.05)


def test_rigid_frame_limit_degrades_when_the_limit_is_perturbed() -> None:
    """The convergence test is a test, not a tautology.

    Backing off the limit must make the disagreement grow, and grow in
    proportion: a frame only 100 times stiffer than the real glass wool already
    differs from the equivalent fluid by more than a part in a thousand, and the
    real frame by more than 20 %. If the Biot path had silently collapsed onto
    the equivalent-fluid path this test would fail.
    """
    frequency = np.geomspace(50.0, 5000.0, 60)
    reference = layered_absorber(
        frequency, [PorousLayer(0.05, _glass_wool_medium(frequency))]
    ).surface_impedance
    residuals = [
        float(
            np.max(
                np.abs(
                    layered_absorber(
                        frequency, [_glass_wool_layer(frequency, 0.05, scale=s)]
                    ).surface_impedance
                    / reference
                    - 1.0
                )
            )
        )
        for s in (1.0, 1e2, 1e4)
    ]
    assert residuals[0] > 0.2
    assert residuals[1] > 1e-3
    assert residuals[0] > residuals[1] > residuals[2]


def test_rigid_frame_limit_holds_through_a_multilayer_stack() -> None:
    """The limit must survive the coupling matrices, not only a bare layer.

    A membrane, an air gap and a poroelastic layer over a rigid wall: in the
    rigid-frame limit the whole stack has to reduce to the same stack built
    with an equivalent-fluid layer, which exercises the fluid-to-porous
    coupling ``[Ipf]``/``[Jpf]`` and the fluid block chaining.
    """
    frequency = np.geomspace(50.0, 5000.0, 40)
    medium = _glass_wool_medium(frequency)
    head: list = [MembraneLayer(0.2), AirLayer(0.02)]
    reference = layered_absorber(
        frequency, [*head, PorousLayer(0.05, medium)], angle=0.3
    )
    frozen = layered_absorber(
        frequency,
        [*head, _glass_wool_layer(frequency, 0.05, scale=1e8)],
        angle=0.3,
    )
    assert np.allclose(
        frozen.surface_impedance, reference.surface_impedance, rtol=1e-7
    )


def test_rigid_frame_limit_holds_for_a_non_rigid_termination() -> None:
    """Same limit with the stack radiating into free air behind it.

    This is the ``[I(n)f]``/``[J(n)f]`` plus Eq. (11.84) branch of the
    assembly, which the hard-wall tests never reach.
    """
    frequency = np.geomspace(50.0, 5000.0, 40)
    medium = _glass_wool_medium(frequency)
    reference = layered_absorber(
        frequency, [PorousLayer(0.05, medium)], termination="free"
    )
    frozen = layered_absorber(
        frequency,
        [_glass_wool_layer(frequency, 0.05, scale=1e8)],
        termination="free",
    )
    assert np.allclose(
        frozen.surface_impedance, reference.surface_impedance, rtol=1e-7
    )
    assert np.allclose(frozen.absorption, reference.absorption, atol=1e-8)


# ---------------------------------------------------------------------------
# The limp limit: convergence on the shipped limp-frame equivalent fluid
# ---------------------------------------------------------------------------
def _soft_fibrous(frequency: np.ndarray) -> PorousMediumResult:
    return johnson_champoux_allard(
        frequency, TABLE_11_2_RESISTIVITY, **TABLE_11_2
    )


def test_limp_limit_converges_on_the_limp_frame_equivalent_fluid() -> None:
    """A&A Sect. 11.3.4: "assuming a limp frame, the stress tensor ... neglected".

    Eq. (11.53) then leaves an equivalent fluid with the density ``rho_limp`` of
    Eq. (11.54), which
    :func:`~phonometry.materials.absorbers.porous.limp_frame` implements in
    Panneton's algebraically identical form Eq. (11.55). Driving the shear
    modulus of the Biot layer towards zero must reproduce it. Tested as a
    convergence over six decades of ``N`` on a 50 Hz to 4 kHz sweep, on the
    soft fibrous material of Table 11.2 the limp model was written for.

    The residual is first order in ``N`` and the test says so: from
    ``N = 1e3`` down it falls by a factor between 7,6 and 10 for every decade,
    not merely "by some". Above that the layer is not limp yet (only 3,2 from
    1e5 to 1e4), and below ``N = 1e-1`` the shear wavenumber
    ``delta3 = omega sqrt(rho_c / N)`` of Eq. (6.87) has diverged so far that
    the assembly refuses the layer outright, which
    :func:`test_an_unresolvable_stack_is_refused_instead_of_exhausting_memory`
    covers.
    """
    frequency = np.geomspace(50.0, 4000.0, 50)
    medium = _soft_fibrous(frequency)
    reference = layered_absorber(
        frequency,
        [PorousLayer(TABLE_11_2_THICKNESS,
                     limp_frame(medium, TABLE_11_2_FRAME_DENSITY,
                                porosity=TABLE_11_2["porosity"]))],
    ).surface_impedance
    residuals = []
    for shear in (1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1):
        result = layered_absorber(
            frequency,
            [PoroelasticLayer(
                TABLE_11_2_THICKNESS, medium, TABLE_11_2["porosity"],
                TABLE_11_2["tortuosity"], TABLE_11_2_FRAME_DENSITY,
                shear * (1.0 + 0.1j),
            )],
        ).surface_impedance
        residuals.append(float(np.max(np.abs(result / reference - 1.0))))
    assert residuals[0] > 0.3
    assert residuals[-1] < 3e-6
    # Not limp yet at 1e5 -> 1e4, first order everywhere below 1e3.
    assert residuals[0] / residuals[1] < 5.0
    for coarse, fine in itertools.pairwise(residuals[2:]):
        assert 7.5 < coarse / fine < 12.0


def test_limp_and_rigid_limits_of_the_same_layer_differ_at_low_frequency() -> None:
    """A&A printed p. 253: the two models "differ mainly at low frequencies".

    A cross-check that the two limits tested above are not the same limit: on
    the Table 11.2 material the limp and rigid equivalent fluids disagree by
    more than 20 % below 200 Hz.
    """
    frequency = np.array([50.0, 100.0, 200.0])
    medium = _soft_fibrous(frequency)
    limp = limp_frame(
        medium, TABLE_11_2_FRAME_DENSITY, porosity=TABLE_11_2["porosity"]
    )
    deviation = np.abs(limp.effective_density / medium.effective_density - 1.0)
    assert float(np.min(deviation)) > 0.2


# ---------------------------------------------------------------------------
# Structure, plotting and input validation
# ---------------------------------------------------------------------------
def test_absorption_is_physical_and_reflection_is_passive() -> None:
    """A passive layer cannot reflect more energy than it receives."""
    frequency = np.geomspace(50.0, 5000.0, 80)
    for angle in (0.0, np.pi / 4.0):
        result = layered_absorber(
            frequency, [_glass_wool_layer(frequency, 0.05)], angle=angle
        )
        assert np.all(result.absorption >= 0.0)
        assert np.all(result.absorption <= 1.0)
        assert np.all(np.abs(result.reflection) <= 1.0 + 1e-12)
        assert np.all(result.surface_impedance.real > 0.0)


def test_layered_result_keeps_the_layers_and_flags_the_missing_chain_matrix() -> None:
    """A six-variable stack has no 2x2 chain matrix; the field says so."""
    frequency = np.geomspace(100.0, 1000.0, 5)
    layer = _glass_wool_layer(frequency, 0.05)
    result = layered_absorber(frequency, [layer])
    assert result.layers == (layer,)
    assert np.all(np.isnan(result.transfer_matrix.real))


def test_zero_thickness_poroelastic_layer_is_ignored() -> None:
    """A degenerate layer contributes nothing, as the fluid layers already do."""
    frequency = np.geomspace(100.0, 1000.0, 8)
    medium = _glass_wool_medium(frequency)
    with_gap = layered_absorber(
        frequency, [_glass_wool_layer(frequency, 0.0), PorousLayer(0.05, medium)]
    )
    without = layered_absorber(frequency, [PorousLayer(0.05, medium)])
    assert np.allclose(with_gap.surface_impedance, without.surface_impedance)


def test_plot_returns_axes_in_both_languages() -> None:
    """Every result object exposes ``.plot()``; both languages must render."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    frequency = np.geomspace(50.0, 3000.0, 40)
    waves = _glass_wool_waves(frequency)
    for language in ("en", "es"):
        axes = waves.plot(language=language)
        assert axes.get_xlabel()
        assert len(axes.get_lines()) == 5
        plt.close(axes.figure)
    layer = _glass_wool_layer(frequency, 0.05)
    axes = layer.plot()
    assert axes is not None
    plt.close(axes.figure)


def test_biot_waves_rejects_bad_input() -> None:
    frequency = np.geomspace(100.0, 1000.0, 4)
    medium = _glass_wool_medium(frequency)
    kwargs = {
        "porosity": TABLE_6_1_POROSITY,
        "tortuosity": TABLE_6_1_TORTUOSITY,
        "frame_density": TABLE_6_1_FRAME_DENSITY,
        "shear_modulus": TABLE_6_1_SHEAR_MODULUS,
    }
    with pytest.raises(ValueError, match="must not exceed 1"):
        biot_waves(medium, **{**kwargs, "porosity": 1.5})
    with pytest.raises(ValueError, match="must be >= 1"):
        biot_waves(medium, **{**kwargs, "tortuosity": 0.5})
    with pytest.raises(ValueError, match="must be positive"):
        biot_waves(medium, **{**kwargs, "frame_density": 0.0})
    with pytest.raises(ValueError, match="positive real part"):
        biot_waves(medium, **{**kwargs, "shear_modulus": -1.0})
    with pytest.raises(ValueError, match="non-negative imaginary part"):
        biot_waves(medium, **{**kwargs, "shear_modulus": 1.0 - 1.0j})
    with pytest.raises(ValueError, match="must be finite"):
        biot_waves(medium, **{**kwargs, "shear_modulus": complex("nan")})
    with pytest.raises(ValueError, match="-1 < nu < 0,5"):
        biot_waves(medium, **kwargs, poisson_ratio=0.5)


def test_layer_rejects_a_medium_on_another_grid() -> None:
    frequency = np.geomspace(100.0, 1000.0, 8)
    other = _glass_wool_medium(np.geomspace(100.0, 1000.0, 9))
    layer = PoroelasticLayer(
        0.05, other, TABLE_6_1_POROSITY, TABLE_6_1_TORTUOSITY,
        TABLE_6_1_FRAME_DENSITY, TABLE_6_1_SHEAR_MODULUS,
    )
    with pytest.raises(ValueError, match="different frequency"):
        layered_absorber(frequency, [layer])


def test_surface_impedance_and_transfer_matrix_reject_bad_thickness() -> None:
    frequency = np.geomspace(100.0, 1000.0, 4)
    waves = _glass_wool_waves(frequency)
    with pytest.raises(ValueError, match="must be positive"):
        biot_surface_impedance(waves, 0.0)
    with pytest.raises(ValueError, match="must be positive"):
        poroelastic_transfer_matrix(waves, -0.01)
    with pytest.raises(ValueError, match="must be positive"):
        frame_quarter_wave_resonance(
            0.0, shear_modulus=1.0, poisson_ratio=0.0, frame_density=30.0
        )


def test_packing_refuses_more_blocks_than_the_assembly_resolves() -> None:
    """The block cap has to hold after packing, not only before it.

    The pre-check bounds the sum of the per-layer losses, but the greedy
    next-fit grouping needs up to twice the optimal number of bins: items just
    over half a budget open a block each. Six terms of 10,5 nepers sum to 63,
    inside a 20-neper budget over four blocks, and still pack into six.
    """
    from phonometry.materials.absorbers.layered import _split_fluid_run

    terms = [("fluid", 1 + 0j, 10.5j)] * 6
    with pytest.raises(ValueError, match="pack into 6 chain blocks"):
        _split_fluid_run(terms, 20.0, 4)  # type: ignore[arg-type]
