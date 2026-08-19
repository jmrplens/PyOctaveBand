#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound absorbers: porous, multilayer and slow-sound.

The empirical and phenomenological models of a porous layer - Delany-Bazley,
Miki, Johnson-Champoux-Allard - and what is built out of them: multilayer
assemblies with air gaps, limp and rigid backings, microperforated panels
(Maa) and the transfer-matrix machinery of Allard & Atalla. All are evaluated
at the one digitization point the printed curves are read at, with the
Bies Appendix D air state, so a model and its published figure meet at the
same abscissa.

The slow-sound perfect absorbers of Jimenez et al. (slitted panels loaded with
Helmholtz resonators) close the module: same transfer-matrix formalism, same
air state, driven to the critical-coupling condition where the reflection
vanishes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register

_POROUS = "Porous & multilayer absorbers (Mechel / Bies / Cox & D'Antonio)"

# Shared porous-domain constants: the digitization point X = rho f / sigma
# = 0.1 with sigma = 20 kPa s/m2 and the Bies 5e Appendix D air state.
_PA_SIGMA = 20000.0
_PA_RHO0 = 1.205
_PA_C0 = 343.0


@register(
    _POROUS,
    "Bies 5e App. D Table D.1 / Mechel 2e G.11 (2)",
    "Delany-Bazley normalised Zc at X = 0.1, real part",
)
def _chk_porous_db_real() -> Outcome:
    f = np.array([0.1 * _PA_SIGMA / _PA_RHO0])
    res = ph.materials.delany_bazley(
        f, _PA_SIGMA, speed_of_sound=_PA_C0, air_density=_PA_RHO0
    )
    return numeric(
        ref.POROUS_DB_ZC_EXPECTED.real,
        float(res.normalized_impedance[0].real), 1e-9,
    )


@register(
    _POROUS,
    "Bies 5e App. D Table D.1 / Mechel 2e G.11 (2)",
    "Delany-Bazley normalised Zc at X = 0.1, imaginary part",
)
def _chk_porous_db_imag() -> Outcome:
    f = np.array([0.1 * _PA_SIGMA / _PA_RHO0])
    res = ph.materials.delany_bazley(
        f, _PA_SIGMA, speed_of_sound=_PA_C0, air_density=_PA_RHO0
    )
    return numeric(
        ref.POROUS_DB_ZC_EXPECTED.imag,
        float(res.normalized_impedance[0].imag), 1e-9,
    )


@register(
    _POROUS,
    "Miki 1990 Eqs. (30)-(34)",
    "Miki normalised wavenumber at f/sigma = 0.1, real part",
)
def _chk_porous_miki() -> Outcome:
    f = np.array([0.1 * _PA_SIGMA])
    res = ph.materials.miki(
        f, _PA_SIGMA, speed_of_sound=_PA_C0, air_density=_PA_RHO0
    )
    return numeric(
        ref.POROUS_MIKI_K_EXPECTED.real,
        float(res.normalized_wavenumber[0].real), 1e-9,
    )


@register(
    _POROUS,
    "Johnson et al. 1987 / Cox & D'Antonio 3e Eq. (6.19)",
    "JCA static viscous limit j w rho_e -> sigma, Pa s/m2",
)
def _chk_porous_jca_dc() -> Outcome:
    f = np.array([1e-3])
    res = ph.materials.johnson_champoux_allard(
        f, _PA_SIGMA, porosity=0.95, tortuosity=1.3,
        viscous_length=6e-5, thermal_length=1.2e-4, air_density=_PA_RHO0,
    )
    value = float((1j * 2.0 * math.pi * f * res.effective_density)[0].real)
    return numeric(_PA_SIGMA, value, 1e-4, rel=True, unit="Pa s/m2", places=1)


@register(
    _POROUS,
    "Mechel 2e Sect. D.3 Eq. (1)",
    "Hard-backed layer: TMM vs -j Zc cot(kd), max rel deviation",
)
def _chk_porous_rigid_backed() -> Outcome:
    f = np.linspace(200.0, 4000.0, 200)
    med = ph.materials.delany_bazley(f, _PA_SIGMA, air_density=_PA_RHO0)
    res = ph.materials.layered_absorber(
        f, [ph.materials.PorousLayer(0.05, med)],
        speed_of_sound=_PA_C0, air_density=_PA_RHO0,
    )
    zs_ref = -1j * med.characteristic_impedance / np.tan(med.wavenumber * 0.05)
    dev = float(np.max(np.abs(res.surface_impedance - zs_ref) / np.abs(zs_ref)))
    return numeric(0.0, dev, 1e-10, places=4)


@register(
    _POROUS,
    "Lossless-layer limit (Mechel 2e Sect. D.3-D.4)",
    "Air cavity over a rigid wall at lambda/4: alpha",
)
def _chk_porous_air_cavity() -> Outcome:
    d = 0.1
    f = np.array([_PA_C0 / (4.0 * d)])
    res = ph.materials.layered_absorber(
        f,
        [ph.materials.AirLayer(d)],
        speed_of_sound=_PA_C0,
        air_density=_PA_RHO0,
    )
    return numeric(0.0, float(res.absorption[0]), 1e-12, places=4)


@register(
    _POROUS,
    "Mechel 2e Sect. D.5",
    "Maximum statistical absorption of a locally reacting plane",
)
def _chk_porous_statistical_max() -> Outcome:
    z = np.linspace(1.0, 3.0, 2001).astype(complex)
    value = float(np.max(ph.materials.statistical_absorption(z)))
    return numeric(ref.POROUS_STATISTICAL_ALPHA_MAX, value, 1e-3, places=3)


@register(
    _POROUS,
    "Cox & D'Antonio 3e Eq. (7.9)",
    "Membrane resonance 60/sqrt(m d), m = 5 kg/m2, d = 5 cm, Hz",
)
def _chk_porous_membrane_resonance() -> Outcome:
    value = ph.materials.membrane_resonance_frequency(
        surface_density=5.0, cavity_depth=0.05,
        speed_of_sound=_PA_C0, air_density=_PA_RHO0,
    )
    return numeric(60.0 / math.sqrt(5.0 * 0.05), value, 0.02, rel=True,
                   unit="Hz", places=2)


@register(
    _POROUS,
    "Maa 1998 Fig. 5 / Cox & D'Antonio 3e Fig. 7.28",
    "Microperforated panel (d=t=0.2 mm, b=2.5 mm, D=6 cm): peak alpha",
)
def _chk_porous_mpp_peak() -> Outcome:
    eps = (math.pi / 4.0) * (ref.MAA_FIG5_DIAMETER / ref.MAA_FIG5_SEPARATION) ** 2
    f = np.linspace(100.0, 4000.0, 2000)
    res = ph.materials.layered_absorber(
        f,
        [
            ph.materials.MicroperforatedPlateLayer(
                ref.MAA_FIG5_THICKNESS, ref.MAA_FIG5_DIAMETER / 2.0, eps
            ),
            ph.materials.AirLayer(ref.MAA_FIG5_CAVITY),
        ],
        speed_of_sound=_PA_C0, air_density=_PA_RHO0,
    )
    return numeric(0.95, float(np.max(res.absorption)), 0.05, places=3)


@register(
    _POROUS,
    "Maa 1998 Eqs. (5a)/(10)",
    "MPP peak absorption vs 4r/(1+r)^2 with Maa's printed resistance",
)
def _chk_porous_maa_peak_closed_form() -> Outcome:
    # Independent expected value: the relative resistance r comes from
    # Maa's printed wide-range approximation Eq. (5a) (with the surface
    # end correction), not from the library's exact Bessel kernel; the
    # peak absorption must then satisfy Eq. (10), alpha0 = 4r/(1+r)^2,
    # within the paper's stated ~6 % accuracy of the approximation.
    eta = 1.84e-5
    d = ref.MAA_FIG5_DIAMETER
    t = ref.MAA_FIG5_THICKNESS
    eps = (math.pi / 4.0) * (d / ref.MAA_FIG5_SEPARATION) ** 2
    f = np.linspace(400.0, 1200.0, 4000)
    res = ph.materials.layered_absorber(
        f,
        [
            ph.materials.MicroperforatedPlateLayer(t, d / 2.0, eps),
            ph.materials.AirLayer(ref.MAA_FIG5_CAVITY),
        ],
        speed_of_sound=_PA_C0, air_density=_PA_RHO0,
    )
    i = int(np.argmax(res.absorption))
    omega = 2.0 * math.pi * float(f[i])
    k_perf = d * math.sqrt(omega * _PA_RHO0 / (4.0 * eta))
    k_r = math.sqrt(1.0 + k_perf**2 / 32.0) + (
        math.sqrt(2.0) / 32.0
    ) * k_perf * (d / t)
    r_rel = 32.0 * eta * t * k_r / (eps * _PA_RHO0 * _PA_C0 * d**2)
    expected = 4.0 * r_rel / (1.0 + r_rel) ** 2
    return numeric(expected, float(res.absorption[i]), 0.02, places=3,
                   expected_label=f"4r/(1+r)^2 = {expected:.3f}")


# Limp-frame equivalent fluid (Allard & Atalla 2e Sect. 11.3.4). The book
# publishes no table of computed limp densities (every comparison is a figure),
# so the anchor is the printed Eq. (11.55) itself, transcribed term by term in
# tests/materials/absorbers/test_limp_frame.py. The two limits the book states in prose on
# printed p. 253 and checked below corroborate that transcription without
# pinning it: a sign-flipped variant of Eq. (11.55) satisfies both. The
# decoupling frequency on the fully specified Table 6.1 glass wool is pure
# arithmetic.
_AA_TABLE_11_2 = {
    "porosity": 0.98,
    "tortuosity": 1.02,
    "viscous_length": 90e-6,
    "thermal_length": 180e-6,
}
_AA_TABLE_11_2_SIGMA = 25.0e3
_AA_TABLE_11_2_RHO1 = 30.0


@register(
    _POROUS,
    "Allard & Atalla 2e Sect. 11.3.4 (Eq. 6.90), Table 6.1 glass wool",
    "Zwikker-Kosten decoupling frequency Fd, Hz",
)
def _chk_limp_decoupling_frequency() -> Outcome:
    fd = ph.materials.decoupling_frequency(
        40.0e3, porosity=0.94, frame_density=130.0
    )
    return numeric(43.27, fd, 0.005, unit="Hz", places=3)


@register(
    _POROUS,
    "Allard & Atalla 2e Eq. (11.55), printed p. 253 (prose limit)",
    "Limp effective density at DC = apparent total density rho_t, kg/m3",
)
def _chk_limp_low_frequency_limit() -> Outcome:
    rigid = ph.materials.johnson_champoux_allard(
        np.array([1.0e-4]), _AA_TABLE_11_2_SIGMA,
        speed_of_sound=_PA_C0, air_density=_PA_RHO0, **_AA_TABLE_11_2,
    )
    limp = ph.materials.limp_frame(
        rigid, _AA_TABLE_11_2_RHO1, porosity=_AA_TABLE_11_2["porosity"]
    )
    expected = _AA_TABLE_11_2_RHO1 + _AA_TABLE_11_2["porosity"] * _PA_RHO0
    return numeric(expected, float(np.real(limp.effective_density[0])),
                   1e-4, rel=True, unit="kg/m3", places=4)


@register(
    _POROUS,
    "Allard & Atalla 2e Eq. (11.55), printed p. 253 (prose limit)",
    "Heavy frame recovers the rigid-frame Zc (relative deviation)",
)
def _chk_limp_heavy_frame_limit() -> Outcome:
    f = np.array([50.0, 125.0, 500.0, 2000.0])
    rigid = ph.materials.johnson_champoux_allard(
        f, _AA_TABLE_11_2_SIGMA, speed_of_sound=_PA_C0,
        air_density=_PA_RHO0, **_AA_TABLE_11_2,
    )
    limp = ph.materials.limp_frame(
        rigid, 1.0e12, porosity=_AA_TABLE_11_2["porosity"]
    )
    deviation = float(np.max(np.abs(
        limp.characteristic_impedance / rigid.characteristic_impedance - 1.0
    )))
    return numeric(0.0, deviation, 1e-5, places=8)


@register(
    _POROUS,
    "Allard & Atalla 2e printed p. 254 (Doutres et al. 2007)",
    "Limp-frame bulk-modulus limit for air, kPa",
)
def _chk_limp_frame_criterion_limit() -> Outcome:
    # The book states "lower than 20 kPa" for |Kc/Kf| < 0.2 with Kf = P0.
    limit = 0.2 * 101325.0
    ok = ph.materials.limp_frame_applicable(
        limit
    ) and not ph.materials.limp_frame_applicable(limit * (1.0 + 1e-9))
    return numeric(20.0, limit / 1000.0 if ok else float("nan"),
                   0.3, unit="kPa", places=2)


# Biot poroelastic layer (Allard & Atalla 2e, ch. 6 and 11). The book prints
# no table of computed surface impedances, so these rows pin the closed forms
# it does print, the three output digits its Sect. 6.5.4 states in prose for the
# fully specified Table 6.1 glass wool, and the two exact limits (rigid frame
# onto the digit-anchored JCA equivalent fluid, and the chapter 11 assembly onto
# the chapter 6 closed form Eq. (6.107)).
_AA_TABLE_6_1 = {
    "porosity": 0.94,
    "tortuosity": 1.06,
    "viscous_length": 0.56e-4,
    "thermal_length": 1.1e-4,
}
_AA_TABLE_6_1_SIGMA = 40_000.0
_AA_TABLE_6_1_RHO1 = 130.0
_AA_TABLE_6_1_SHEAR = 220.0e4 * (1.0 + 0.1j)


def _aa_glass_wool_medium(frequency: np.ndarray) -> Any:
    """The rigid-frame JCA equivalent fluid of the Table 6.1 glass wool."""
    return ph.materials.johnson_champoux_allard(
        frequency, _AA_TABLE_6_1_SIGMA, **_AA_TABLE_6_1
    )


def _aa_glass_wool_waves(frequency: np.ndarray) -> Any:
    return ph.materials.biot_waves(
        _aa_glass_wool_medium(frequency),
        porosity=_AA_TABLE_6_1["porosity"],
        tortuosity=_AA_TABLE_6_1["tortuosity"],
        frame_density=_AA_TABLE_6_1_RHO1,
        shear_modulus=_AA_TABLE_6_1_SHEAR,
    )


def _aa_glass_wool_layer(
    medium: Any, thickness: float, scale: float = 1.0
) -> Any:
    """The same material as a poroelastic layer, optionally frozen stiff."""
    return ph.materials.PoroelasticLayer(
        thickness,
        medium,
        _AA_TABLE_6_1["porosity"],
        _AA_TABLE_6_1["tortuosity"],
        _AA_TABLE_6_1_RHO1 * scale,
        _AA_TABLE_6_1_SHEAR * scale,
    )


@register(
    _POROUS,
    "Allard & Atalla 2e Eq. (6.110), Table 6.1 glass wool",
    "Frame lambda/4 resonance of a 10 cm layer, Hz",
)
def _chk_biot_frame_resonance() -> Outcome:
    value = ph.materials.frame_quarter_wave_resonance(
        0.10, shear_modulus=_AA_TABLE_6_1_SHEAR, poisson_ratio=0.0,
        frame_density=_AA_TABLE_6_1_RHO1,
    )
    return numeric(459.9, value, 0.05, unit="Hz", places=2)


@register(
    _POROUS,
    "Allard & Atalla 2e Sect. 6.5.4 (Biot model output), pp. 124-125",
    "Airborne compressional branch changes root at 495 Hz",
)
def _chk_biot_branch_crossing() -> Outcome:
    grid = np.arange(400.0, 600.0, 0.1)
    waves = _aa_glass_wool_waves(grid)
    swap = np.flatnonzero(np.diff(waves.airborne_is_second.astype(int)) != 0)
    crossing = float(grid[swap[0]]) if swap.size == 1 else float("nan")
    return numeric(495.0, crossing, 0.01, rel=True, unit="Hz", places=1)


@register(
    _POROUS,
    "Allard & Atalla 2e Sect. 6.5.4 (Biot model output), pp. 124-125",
    "Frame-borne velocity ratio Re(mu_b) at 1500 Hz (see ERRATA)",
)
def _chk_biot_frame_borne_ratio() -> Outcome:
    waves = _aa_glass_wool_waves(np.array([1500.0]))
    value = float(np.real(waves.frame_borne_velocity_ratio[0]))
    return numeric(0.82, value, 0.02, rel=True, places=3)


@register(
    _POROUS,
    "Allard & Atalla 2e Sect. 6.6.3 (Biot model output), p. 129",
    "Surface-impedance peak of a 5,6 cm layer, Hz",
)
def _chk_biot_impedance_peak() -> Outcome:
    grid = np.arange(500.0, 1200.0, 0.25)
    impedance = ph.materials.biot_surface_impedance(
        _aa_glass_wool_waves(grid), 0.056
    )
    peak = float(grid[int(np.argmax(impedance.imag))])
    return numeric(860.0, peak, 0.02, rel=True, unit="Hz", places=1)


@register(
    _POROUS,
    "Allard & Atalla 2e Sect. 11.3.4 (rigid-frame limit)",
    "Stiff, heavy frame recovers the JCA layer (max rel deviation)",
)
def _chk_biot_rigid_frame_limit() -> Outcome:
    frequency = np.geomspace(50.0, 5000.0, 40)
    medium = _aa_glass_wool_medium(frequency)
    reference = ph.materials.layered_absorber(
        frequency,
        [ph.materials.PorousLayer(0.05, medium)],
        angle=math.pi / 4.0,
    ).surface_impedance
    frozen = ph.materials.layered_absorber(
        frequency,
        [_aa_glass_wool_layer(medium, 0.05, 1e8)],
        angle=math.pi / 4.0,
    ).surface_impedance
    deviation = float(np.max(np.abs(frozen / reference - 1.0)))
    return numeric(0.0, deviation, 1e-7, places=10)


@register(
    _POROUS,
    "Allard & Atalla 2e Eq. (6.107) vs Sect. 11.5 assembly",
    "Two independent derivations of Zs (max rel deviation)",
)
def _chk_biot_assembly_vs_closed_form() -> Outcome:
    frequency = np.geomspace(20.0, 5000.0, 80)
    medium = _aa_glass_wool_medium(frequency)
    closed = ph.materials.biot_surface_impedance(
        _aa_glass_wool_waves(frequency), 0.10
    )
    assembled = ph.materials.layered_absorber(
        frequency, [_aa_glass_wool_layer(medium, 0.10)]
    ).surface_impedance
    deviation = float(np.max(np.abs(assembled / closed - 1.0)))
    return numeric(0.0, deviation, 1e-10, places=12)


# ===========================================================================
# Slow-sound slit + Helmholtz-resonator perfect absorbers (Jimenez et al.)
# ===========================================================================
_SLOW_SOUND = "Slow-sound perfect absorbers (Jimenez et al. Appl. Sci. 2017)"


@register(
    _SLOW_SOUND,
    "Jimenez et al. Appl. Sci. 2017 Eq. (9)",
    "Critical coupling: alpha at the design frequency (300 Hz, normal)",
)
def _chk_slow_sound_perfect_absorption() -> Outcome:
    res = ph.materials.HelmholtzResonator(
        neck_length=1.0e-3, neck_side=3.0e-3,
        cavity_length=30.0e-3, cavity_side=27.0e-3,
    )
    air = ph.materials.AirProperties(density=_PA_RHO0, speed_of_sound=_PA_C0)
    design = ph.materials.critical_coupling_design(
        300.0, res, lattice_step=3.0e-2, period=5.0e-2, air=air,
    )
    out = ph.materials.slit_helmholtz_absorber(
        np.array([300.0]), design.resonator, slit_height=design.slit_height,
        lattice_step=3.0e-2, period=5.0e-2, air=air,
    )
    # The check requires the solver to have converged: a non-converged design
    # fails the check outright rather than silently reporting its (imperfect)
    # absorption.
    computed = float(out.absorption[0]) if design.converged else float("nan")
    return numeric(1.0, computed, 1e-3, places=4)


@register(
    _SLOW_SOUND,
    "Poiseuille limit (Stinson 1991)",
    "Slit: j w rho_s -> 12 eta / h^2 as w -> 0 (h = 1.2 mm)",
)
def _chk_slow_sound_slit_resistivity() -> Outcome:
    eta = 1.84e-5
    h = 1.2e-3
    f = np.array([1.0e-2])
    rho_s, _ = ph.materials.slit_effective_properties(
        f, slit_height=h,
        air=ph.materials.AirProperties(density=_PA_RHO0, viscosity=eta),
    )
    sigma = float((1j * 2.0 * math.pi * f * rho_s)[0].real)
    return numeric(12.0 * eta / h**2, sigma, 1e-3, rel=True,
                   unit="Pa s/m2", places=1)


@register(
    _SLOW_SOUND,
    "Poiseuille limit (Stinson 1991)",
    "Square duct: j w rho -> 28.454 eta / w^2 as w -> 0 (w = 3 mm)",
)
def _chk_slow_sound_duct_resistivity() -> Outcome:
    eta = 1.84e-5
    side = 3.0e-3
    f = np.array([1.0e-2])
    rho, _ = ph.materials.rectangular_duct_properties(
        f, side=side,
        air=ph.materials.AirProperties(density=_PA_RHO0, viscosity=eta),
    )
    sigma = float((1j * 2.0 * math.pi * f * rho)[0].real)
    return numeric(28.454 * eta / side**2, sigma, 2e-3, rel=True,
                   unit="Pa s/m2", places=1)
