#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Domain 8 - Outdoor propagation, occupational exposure and materials.

Sound travelling outdoors and the exposure it produces: atmospheric absorption
over the full ISO 9613-1 Table 1 grid, the ISO 9613-2 engineering attenuation
terms, and the ISO 9612 task-based determination of occupational noise
exposure.

The material and surface subjects that the propagation terms consume sit with
them: the ISO 11654 absorption rating, the ISO 9053-2 airflow resistance and
the ISO 10534-1/-2 impedance tube; the ISO 17497-1/-2 scattering and diffusion
coefficients, whose external anchor is the published BEM data of Cox &
D'Antonio Appendix B rather than the library's own model; the ISO 13472-1/-2
in-situ measurement of road-surface absorption; and the precision sound-power
methods of ISO 3745 and ISO 9614-3.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register


def _iso9613_table1(point: tuple[float, float, float, float]) -> Outcome:
    """Compare air_attenuation against an ISO 9613-1 Table 1 grid point (dB/km)."""
    temp, rh, freq, alpha_km = point
    computed = float(
        ph.air_attenuation(freq, temp, rh, exact_midband=True)[()]
    ) * 1000.0  # dB/m -> dB/km
    # Tolerance = 1 in the last printed (3-significant-figure) digit.
    tol = 10.0 ** (math.floor(math.log10(alpha_km)) - 2)
    return numeric(alpha_km, computed, tol, unit="dB/km", places=3)


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-1:1993 Table 1",
    "Air attenuation @ 10 degC, 70 %, 1 kHz",
)
def _chk_iso9613_table1_mid() -> Outcome:
    return _iso9613_table1(ref.ISO9613_1_TABLE1_MID)


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-1:1993 Table 1",
    "Air attenuation @ 0 degC, 20 %, 2 kHz",
)
def _chk_iso9613_table1_corner() -> Outcome:
    return _iso9613_table1(ref.ISO9613_1_TABLE1_CORNER)


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 Table 2",
    "Atmospheric attenuation grid, 6 conditions x 8 octave bands, dB/km",
)
def _chk_iso9613_2_table2_grid() -> Outcome:
    """Every printed Table 2 cell (exact midbands) to half its last digit.

    Worst residual in units of the per-cell tolerance; the documented
    15 degC / 80 % / 1 kHz print quirk carries a 0.06 dB/km tolerance
    (printed 4,1 vs exact-midband 4,151).
    """
    worst = 0.0
    for (temp, rh), row in ref.ISO9613_2_TABLE2.items():
        alpha = ph.air_attenuation(
            ref.ISO9613_2_TABLE2_BANDS, temp, rh, 101.325, exact_midband=True
        ) * 1000.0
        for got, printed, band in zip(alpha, row, ref.ISO9613_2_TABLE2_BANDS):
            tol = 0.5 if printed >= 100.0 else 0.05
            if (temp, rh, band) == (15.0, 80.0, 1000.0):
                tol = 0.06
            residual = abs(float(got) - printed) / tol
            if not math.isfinite(residual):
                return Outcome(expected="finite Table 2 residuals",
                               computed=f"non-finite at {band} Hz", delta="inf",
                               passed=False)
            worst = max(worst, residual)
    return Outcome(
        expected="all 48 cells within half a printed digit",
        computed=f"worst residual {worst:.3f} x tolerance",
        delta=f"{worst:.3f} x",
        passed=worst <= 1.0,
    )



@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 Eq. (7)",
    "Geometrical divergence Adiv = 20 lg(d/d0) + 11 at 100 m",
)
def _chk_iso9613_2_adiv() -> Outcome:
    computed = ph.geometric_divergence(100.0)
    return numeric(ref.ISO9613_2_ADIV_100M, computed, 1e-9, unit="dB", places=6)


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 Table 3",
    "Ground b'(0) porous limit -> Agr(250 Hz) = 2(-1.5 + 10.1)",
)
def _chk_iso9613_2_ground_limit() -> Outcome:
    # Porous ground both sides (Gs = Gr = 1), source and receiver on the ground
    # (hs = hr = 0), fully-developed path (dp -> inf): the 250 Hz band isolates
    # the Table 3 limit b'(0) = 1,5 + 8,6 = 10,1, so Agr = 2(-1,5 + 10,1) = 17,2.
    big = 1.0e7
    agr = ph.ground_attenuation(big, 0.0, 0.0, [250.0], 1.0, 1.0, 1.0,
                                projected_distance=big)
    return numeric(
        ref.ISO9613_2_GROUND_AGR_250_POROUS, float(agr[0]), 1e-6, unit="dB",
        places=4,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 clause 7.4",
    "Single-edge diffraction saturates at the 20 dB cap",
)
def _chk_iso9613_2_barrier_single_cap() -> Outcome:
    b = ph.Barrier(source_to_edge=50.0, edge_to_receiver=50.0)
    dz = ph.barrier_attenuation(b, 60.0, ph.DEFAULT_FREQUENCIES)
    return numeric(
        ref.ISO9613_2_BARRIER_CAP_SINGLE, float(np.max(dz)), 1e-9, unit="dB",
        places=6,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 clause 7.4",
    "Double-edge diffraction saturates at the 25 dB cap",
)
def _chk_iso9613_2_barrier_double_cap() -> Outcome:
    b = ph.Barrier(source_to_edge=50.0, edge_to_receiver=50.0, edge_separation=5.0)
    dz = ph.barrier_attenuation(b, 60.0, ph.DEFAULT_FREQUENCIES)
    return numeric(
        ref.ISO9613_2_BARRIER_CAP_DOUBLE, float(np.max(dz)), 1e-9, unit="dB",
        places=6,
    )


def _iso9612_annex_d_tasks() -> list[ph.Task]:
    """Rebuild the ISO 9612 Annex D Task objects from the shared input table."""
    from phonometry.hearing.occupational_exposure import Task

    tasks = []
    for samples, duration, drange in ref.ISO9612_ANNEX_D_TASKS:
        tasks.append(Task(samples=samples, duration_hours=duration,
                          duration_range=drange))
    return tasks


def _lex_and_u(lex: float, u: float, exp_lex: float, exp_u: float,
               tol_lex: float, tol_u: float) -> Outcome:
    """Combined LEX,8h + expanded-uncertainty outcome for one ISO 9612 example."""
    ok = abs(lex - exp_lex) <= tol_lex and abs(u - exp_u) <= tol_u
    return Outcome(
        expected=f"LEX,8h {exp_lex:.1f}; U {exp_u:.1f} dB",
        computed=f"LEX,8h {lex:.1f}; U {u:.1f} dB",
        delta=f"{lex - exp_lex:+.2f}; {u - exp_u:+.2f} dB",
        passed=ok,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9612:2009 Annex D",
    "Task-based LEX,8h + U (welder day, case a)",
)
def _chk_iso9612_annex_d() -> Outcome:
    res = ph.task_based_exposure(
        _iso9612_annex_d_tasks(), include_duration_uncertainty=False, warn=False
    )
    return _lex_and_u(
        res.lex_8h, res.expanded_uncertainty,
        ref.ISO9612_ANNEX_D_LEX_8H, ref.ISO9612_ANNEX_D_U, 0.05, 0.1,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9612:2009 Annex E",
    "Job-based LEX,8h + U (production line, 18 workers)",
)
def _chk_iso9612_annex_e() -> Outcome:
    res = ph.job_based_exposure(
        list(ref.ISO9612_ANNEX_E_SAMPLES), ref.ISO9612_ANNEX_E_TE_HOURS
    )
    return _lex_and_u(
        res.lex_8h, res.expanded_uncertainty,
        ref.ISO9612_ANNEX_E_LEX_8H, ref.ISO9612_ANNEX_E_U, 0.1, 0.05,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9612:2009 Annex F",
    "Full-day LEX,8h + U (forklift drivers)",
)
def _chk_iso9612_annex_f() -> Outcome:
    res = ph.full_day_exposure(
        list(ref.ISO9612_ANNEX_F_SAMPLES), ref.ISO9612_ANNEX_F_TE_HOURS
    )
    return _lex_and_u(
        res.lex_8h, res.expanded_uncertainty,
        ref.ISO9612_ANNEX_F_LEX_8H, ref.ISO9612_ANNEX_F_U, 0.05, 0.05,
    )


# ---------------------------------------------------------------------------
# Materials: absorption rating, airflow resistance & impedance tube
# ---------------------------------------------------------------------------
_MATERIALS = "Materials: absorption, airflow & impedance"


@register(_MATERIALS, "ISO 11654:1997 Annex A.1", "Weighted absorption alpha_w (no indicator)")
def _chk_iso11654_a1() -> Outcome:
    res = ph.weighted_absorption(list(ref.ISO11654_ANNEX_A1_ALPHA_P))
    out = numeric(ref.ISO11654_ANNEX_A1_ALPHA_W, res.alpha_w, 5e-4, places=2)
    # alpha_w matches AND no shape indicator applies AND class is C.
    ok = (
        out.passed
        and res.shape_indicator == ref.ISO11654_ANNEX_A1_INDICATOR
        and res.absorption_class == ref.ISO11654_ANNEX_A1_CLASS
    )
    return Outcome(
        expected=f"{ref.ISO11654_ANNEX_A1_ALPHA_W:.2f} (class C, no indic.)",
        computed=f"{res.alpha_w:.2f} (class {res.absorption_class}, '{res.shape_indicator}')",
        delta=out.delta,
        passed=ok,
    )


@register(_MATERIALS, "ISO 11654:1997 Annex A.2", "Weighted absorption alpha_w with M indicator")
def _chk_iso11654_a2() -> Outcome:
    res = ph.weighted_absorption(list(ref.ISO11654_ANNEX_A2_ALPHA_P))
    out = numeric(ref.ISO11654_ANNEX_A2_ALPHA_W, res.alpha_w, 5e-4, places=2)
    ok = out.passed and res.shape_indicator == ref.ISO11654_ANNEX_A2_INDICATOR
    return Outcome(
        expected=f"{ref.ISO11654_ANNEX_A2_ALPHA_W:.2f}(M)",
        computed=res.rating_label,
        delta=out.delta,
        passed=ok,
    )


@register(_MATERIALS, "ISO 9053-2:2020 Annex A.3", "Thermal boundary-layer thickness b")
def _chk_iso9053_2_boundary() -> Outcome:
    b = ph.thermal_boundary_layer_thickness(frequency=ref.ISO9053_2_ANNEX_A_FREQUENCY)
    return numeric(
        ref.ISO9053_2_ANNEX_A_BOUNDARY_LAYER, b, 5e-6, unit="m", places=5,
    )


@register(_MATERIALS, "ISO 9053-2:2020 Annex A.3", "Effective ratio of specific heats kappa'")
def _chk_iso9053_2_kappa() -> Outcome:
    kp = ph.effective_kappa(
        cavity_surface=ref.ISO9053_2_ANNEX_A_SURFACE,
        cavity_volume=ref.ISO9053_2_ANNEX_A_VOLUME,
        frequency=ref.ISO9053_2_ANNEX_A_FREQUENCY,
    )
    return numeric(ref.ISO9053_2_ANNEX_A_KAPPA_PRIME, kp, 5e-4, places=3)


@register(_MATERIALS, "ISO 10534-1:1996 Eqs (9)/(13)/(14)", "Absorption from standing-wave ratio s=3")
def _chk_iso10534_1_swr() -> Outcome:
    alpha = float(ph.standing_wave_absorption(ref.ISO10534_1_SWR))
    # The intermediate |r| = (s-1)/(s+1) (Eq. (13)) must match its shared
    # oracle too, so both steps of the chain are pinned.
    from phonometry.materials.absorbers.standing_wave import (
        standing_wave_reflection_magnitude,
    )

    r_mag = float(standing_wave_reflection_magnitude(ref.ISO10534_1_SWR))
    out = numeric(ref.ISO10534_1_ABSORPTION, alpha, 1e-9, places=4)
    r_ok = abs(r_mag - ref.ISO10534_1_REFLECTION_MAGNITUDE) <= 1e-9
    # Show both chained values so a |r| failure is visible in the report.
    expected = (
        f"alpha {out.expected}, |r| {ref.ISO10534_1_REFLECTION_MAGNITUDE:g}"
    )
    computed = f"alpha {out.computed}, |r| {r_mag:.4f}"
    return Outcome(expected, computed, out.delta, out.passed and r_ok)


@register(
    _MATERIALS,
    "ISO 10534-2 Eq. (17) / Annex D",
    "Two-microphone round trip recovers a known reflection factor",
)
def _chk_iso10534_2_roundtrip() -> Outcome:
    # Synthesise the transfer function H12 of a known complex r via the
    # Annex D field equations (Eq. (D.7)), then recover r with the library's
    # Eq. (17) reduction. Synthesis and reduction share only the plane-wave
    # field model, so this is an algebraic identity: the only residual is
    # float rounding, hence the 1e-9 tolerance.
    from phonometry.materials.absorbers.impedance_tube import (
        reflection_factor,
        tube_wavenumber,
    )

    f = np.array([500.0, 1000.0, 1800.0])
    x1, spacing, c0 = 0.12, 0.03, 343.2
    r_true = 0.3 - 0.4j
    k0 = np.asarray(tube_wavenumber(f, c0))
    x2 = x1 - spacing
    h12 = (
        (np.exp(1j * k0 * x2) + r_true * np.exp(-1j * k0 * x2))
        / (np.exp(1j * k0 * x1) + r_true * np.exp(-1j * k0 * x1))
    )
    r = reflection_factor(h12, spacing=spacing, x1=x1, wavenumber=k0)
    err = float(np.max(np.abs(np.asarray(r) - r_true)))
    # NOTE: no pipe characters in the label (it lands in a Markdown table cell).
    return numeric(0.0, err, 1e-9, places=9,
                   expected_label="abs(r - (0.3-0.4j)) = 0 (identity, +/-1e-9)")


# ---------------------------------------------------------------------------
# Scattering & diffusion (ISO 17497-1/-2)
# ---------------------------------------------------------------------------
_SCATTERING = "Scattering & diffusion (ISO 17497)"


@register(_SCATTERING, "ISO 17497-1:2004 Eq (2)", "Reference speed of sound at 20 C")
def _chk_iso17497_1_speed() -> Outcome:
    c = float(ph.speed_of_sound(20.0))
    return numeric(ref.ISO17497_1_SPEED_OF_SOUND_20C, c, 1e-6, unit="m/s", places=4)


@register(_SCATTERING, "ISO 17497-1:2004 Eqs (1)/(4)/(5)", "Scattering coefficient (synthetic chain)")
def _chk_iso17497_1_scattering() -> Outcome:
    t1, t2, t3, t4 = ref.ISO17497_1_CHAIN_T
    c = ref.ISO17497_1_CHAIN_C
    alpha_s = ph.random_incidence_absorption(
        ref.ISO17497_1_CHAIN_V, ref.ISO17497_1_CHAIN_S, c1=c, t1=t1, c2=c, t2=t2
    )
    alpha_spec = ph.specular_absorption_coefficient(
        ref.ISO17497_1_CHAIN_V, ref.ISO17497_1_CHAIN_S, c3=c, t3=t3, c4=c, t4=t4
    )
    s = float(ph.scattering_coefficient(alpha_spec, alpha_s))
    return numeric(ref.ISO17497_1_CHAIN_SCATTERING, s, 1e-9, places=4)


@register(_SCATTERING, "ISO 17497-1:2004 Annex A.5", "Expanded uncertainty of scattering coefficient")
def _chk_iso17497_1_uncertainty() -> Outcome:
    u = float(ph.scattering_coefficient_uncertainty(
        ref.ISO17497_1_A5_ALPHA_SPEC,
        ref.ISO17497_1_A5_ALPHA_S,
        ref.ISO17497_1_A5_U_ALPHA_SPEC,
        ref.ISO17497_1_A5_U_ALPHA_S,
    ).u_scattering)
    return numeric(ref.ISO17497_1_A5_U_SCATTERING, u, 1e-6, places=5)


# The ISO 17497-2 arc levels in ``reference_data`` are generated by the
# library's own Fraunhofer model for a published geometry (Cox & D'Antonio 3e
# Appendix B section 7: N = 7 QRD, 6 periods, 3.6 m wide, 0.2 m deep; the
# commercial QRD of Hargreaves et al. 2000, Table I), so these three rows are
# arithmetic oracles for Formulas (5)/(7) on committed levels - the external
# anchor against published third-party BEM data is the Appendix B rows below.
@register(_SCATTERING, "ISO 17497-2:2012 Formula (5)", "Directional diffusion coefficient (QRD, model arc)")
def _chk_iso17497_2_diffusion_qrd() -> Outcome:
    d = float(ph.directional_diffusion_coefficient(list(ref.ISO17497_2_QRD_LEVELS)))
    return numeric(ref.ISO17497_2_QRD_DIFFUSION, d, 1e-6, places=4)


@register(_SCATTERING, "ISO 17497-2:2012 Formula (5)", "Directional diffusion coefficient (flat reference)")
def _chk_iso17497_2_diffusion_flat() -> Outcome:
    d = float(ph.directional_diffusion_coefficient(list(ref.ISO17497_2_FLAT_LEVELS)))
    return numeric(ref.ISO17497_2_FLAT_DIFFUSION, d, 1e-6, places=4)


@register(_SCATTERING, "ISO 17497-2:2012 Formula (7)", "Normalised diffusion coefficient (QRD, model arc)")
def _chk_iso17497_2_diffusion_normalized() -> Outcome:
    d_qrd = ph.directional_diffusion_coefficient(list(ref.ISO17497_2_QRD_LEVELS))
    d_flat = ph.directional_diffusion_coefficient(list(ref.ISO17497_2_FLAT_LEVELS))
    d_n = float(ph.normalized_diffusion_coefficient(d_qrd, d_flat))
    return numeric(ref.ISO17497_2_NORMALIZED_DIFFUSION, d_n, 1e-6, places=4)


# External anchor against published third-party data: Cox & D'Antonio 3e
# Appendix B (pp. 481-485), section 7, "N = 7 QRD, 6 periods, 0.2 m deep",
# normal incidence: 2D BEM normalised diffusion coefficients per one-third-
# octave band. The library's Fraunhofer prediction (band = energy average of
# seven single-frequency responses, as in section 5.2.5) matches the published
# 200-400 Hz bands within 0.01. Low-band anchor only: across the full
# published 100-5000 Hz range the model-vs-BEM mean absolute deviation is
# ~0.09, because edge diffraction and near-grazing effects are outside the
# far-field phase-grating model.
def _make_cox_appendix_b_check(band: float, published: float) -> Callable[[], Outcome]:
    def check() -> Outcome:
        from diffuser_prediction import predicted_band_normalized_diffusion

        d_n = predicted_band_normalized_diffusion(band)
        return numeric(published, d_n, ref.COX3E_APPENDIX_B_TOLERANCE, places=3)
    return check


for _band, _published in zip(
    ref.COX3E_APPENDIX_B_QRD_BANDS, ref.COX3E_APPENDIX_B_QRD_DN, strict=True
):
    register(
        _SCATTERING,
        "Cox & D'Antonio 3e App. B (2D BEM)",
        f"Normalised diffusion d_n, N=7 QRD x 6 periods, {_band:g} Hz band "
        "(low-band anchor)",
    )(_make_cox_appendix_b_check(_band, _published))


@register(_SCATTERING, "ISO 17497-2:2012 Formula (8)", "Zenith area factor (radians convention)")
def _chk_iso17497_2_area_factor() -> Outcome:
    n = ph.area_factors([0.0, 30.0, 60.0, 90.0], delta_theta=5.0)
    return numeric(ref.ISO17497_2_AREA_FACTOR_ZENITH, float(n[0]), 1e-6, places=5)


@register(_SCATTERING, "Cox & D'Antonio Eq (10.3)", "QRD deepest well depth (N=7, f0=500 Hz)")
def _chk_diffuser_qrd_depth() -> Outcome:
    d = ph.qrd_well_depths(7, 500.0, speed_of_sound=343.0)
    return numeric(ref.DIFFUSER_QRD7_MAX_DEPTH, float(d.max()), 1e-12, unit="m", places=4)


@register(_SCATTERING, "Cox & D'Antonio Eq (5.8) + ISO 17497-2 Formula (7)",
          "Flat-panel predicted normalised diffusion (self-reference zero)")
def _chk_diffuser_flat_normalized() -> Outcome:
    spectrum = ph.predicted_diffusion_spectrum(
        0.10, [2000.0], depths=[0.0] * 7, periods=5
    )
    assert spectrum.normalized is not None
    return numeric(
        ref.DIFFUSER_FLAT_NORMALIZED_DIFFUSION,
        float(spectrum.normalized[0]), 1e-12, places=6,
    )


@register(_SCATTERING, "Cox & D'Antonio Eq (5.8) + ISO 17497-2 Formula (7)",
          "QRD predicted normalised diffusion at 2 kHz (above flat panel)")
def _chk_diffuser_qrd_normalized() -> Outcome:
    depths = ph.qrd_well_depths(7, 500.0, speed_of_sound=343.0)
    spectrum = ph.predicted_diffusion_spectrum(
        0.10, [2000.0], depths=depths, periods=5
    )
    assert spectrum.normalized is not None
    return numeric(
        ref.DIFFUSER_QRD7_NORMALIZED_DIFFUSION_2K,
        float(spectrum.normalized[0]), 1e-9, places=4,
    )


# ---------------------------------------------------------------------------
# In-situ road-surface absorption (ISO 13472-1/-2)
# ---------------------------------------------------------------------------
_ROAD = "In-situ road absorption (ISO 13472)"


@register(_ROAD, "ISO 13472-1:2002 Clause 4.2", "Geometrical-spreading factor Kr")
def _chk_iso13472_1_kr() -> Outcome:
    kr = ph.geometric_spreading_factor()
    return numeric(ref.ISO13472_1_KR, kr, 1e-12, places=4)


@register(_ROAD, "ISO 13472-1:2002 Annex A", "Maximum-sampled-area radius")
def _chk_iso13472_1_msa() -> Outcome:
    r = ph.max_sampled_area_radius(ref.ISO13472_1_MSA_WINDOW)
    return numeric(ref.ISO13472_1_MSA_RADIUS, r, 1e-6, unit="m", places=4)


@register(_ROAD, "ISO 13472-2:2010 Clause 5.4.1", "Spot-tube upper usable frequency f_u")
def _chk_iso13472_2_fu() -> Outcome:
    fu = ph.spot_tube_upper_frequency(
        ref.ISO13472_2_SPOT_DIAMETER, ref.ISO13472_2_SPOT_SPEED
    )
    return numeric(ref.ISO13472_2_SPOT_FU, fu, 0.1, unit="Hz", places=1)


# ---------------------------------------------------------------------------
# Precision sound power (ISO 3745 / ISO 9614-3)
# ---------------------------------------------------------------------------
_PRECISION_POWER = "Precision sound power (ISO 3745 / 9614-3)"


@register(_PRECISION_POWER, "ISO 3745:2012 Clause 10.5 EXAMPLE", "Expanded uncertainty U (k=2)")
def _chk_iso3745_uncertainty() -> Outcome:
    u = float(ph.precision_uncertainty(
        ref.ISO3745_U_SIGMA_R0, ref.ISO3745_U_SIGMA_OMC, ref.ISO3745_U_COVERAGE
    ))
    return numeric(ref.ISO3745_U_EXPANDED, u, 1e-3, unit="dB", places=3)


@register(_PRECISION_POWER, "ISO 3745:2012 Eq (11)", "K1 background floor (6 dB edge band)")
def _chk_iso3745_k1_floor() -> Outcome:
    k1 = ph.precision_background_correction(
        np.array([[ref.ISO3745_K1_EDGE_LEVEL]]),
        np.array([[ref.ISO3745_K1_EDGE_BACKGROUND]]),
        np.array([ref.ISO3745_K1_EDGE_FREQUENCY]),
    )
    return numeric(ref.ISO3745_K1_EDGE_FLOOR, float(k1[0, 0]), 1e-4, unit="dB", places=4)


@register(_PRECISION_POWER, "ISO 3745:2012 Eq (16)", "Meteorological C1 at 23 C reference")
def _chk_iso3745_c1() -> Outcome:
    c1 = ph.meteorological_corrections(23.0, 101.325).c1
    return numeric(ref.ISO3745_C1_REFERENCE, c1, 1e-4, unit="dB", places=4)


@register(_PRECISION_POWER, "ISO 9614-3:2002 Eqs (5)/(8)/(9)", "Uniform-intensity LW recovery")
def _chk_iso9614_3_uniform() -> Outcome:
    areas = np.array(ref.ISO9614_3_UNIFORM_AREAS, dtype=float)
    i_n = np.full(areas.shape, ref.ISO9614_3_UNIFORM_POWER / float(areas.sum()))
    res = ph.sound_power_intensity_precision(i_n, areas)
    return numeric(ref.ISO9614_3_UNIFORM_LW, float(res.sound_power_level[0]), 1e-9, unit="dB", places=4)
