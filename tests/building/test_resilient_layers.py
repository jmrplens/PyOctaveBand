#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for the resilient-layer prediction chain.

Sources and page references used as oracles below (printed page / pdf page):

* **H** = Hopkins, *Sound Insulation* (Butterworth-Heinemann, 2007);
  printed page + 27 = pdf page.
* **V** = Vigran, *Building Acoustics* (Taylor & Francis, 2008);
  printed page + 22 = pdf page.
* **ISO 12354-1:2017** Annex D, printed pp. 37-40 / pdf pp. 43-46.
* **ISO 12354-2:2017** Annex C, printed pp. 23-25 / pdf pp. 29-31, and
  Annex G Table G.4, printed p. 39 / pdf p. 45.

Hopkins Table A2 (printed p. 608 / pdf p. 635) supplies the material data of
the four walking surfaces used throughout: concrete cast in situ
(rho 2 200 kg/m3, cL 3 800 m/s, nu 0,2), sand-cement screed (2 000, 3 250,
0,2), chipboard (760, 2 200, nu 0,3 estimated) and OSB (590, 2 570, 0,3).
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref

from phonometry import (
    TAPPING_HAMMER_MASS,
    TAPPING_HAMMER_RADIUS,
    combined_dynamic_stiffness,
    covering_contact_stiffness,
    covering_improvement,
    double_floating_floor_resonances,
    floating_floor_improvement_spectrum,
    floating_floor_resonance_frequency,
    force_pulse,
    hammer_impact_velocity,
    hammer_limiting_frequency,
    infinite_plate_impedance,
    lining_improvement,
    lining_improvement_in_situ,
    lining_resonance_frequency,
    mass_spring_mass_resonance,
    plate_bending_stiffness,
    plate_contact_stiffness,
    resilient_mount_improvement,
    short_pulse_mean_square_force,
    tapping_cut_off_frequency,
    tapping_force_spectrum,
    weighted_floating_floor_improvement,
    weighted_lining_improvement,
)

#: Hopkins Table A2 material data as ``(rho, cL, nu)`` (printed p. 608).
_A2 = {
    "concrete": (2200.0, 3800.0, 0.2),
    "screed": (2000.0, 3250.0, 0.2),
    "chipboard": (760.0, 2200.0, 0.3),
    "osb": (590.0, 2570.0, 0.3),
    "plywood": (710.0, 3850.0, 0.3),
}


def _youngs_modulus(rho: float, c_l: float, nu: float) -> float:
    """``E`` from the quasi-longitudinal plate speed ``cL = sqrt(E/(rho(1-nu^2)))``."""
    return rho * c_l**2 * (1.0 - nu**2)


def _plate(name: str, thickness: float) -> tuple[float, float]:
    """``(contact stiffness K, driving-point impedance Zdp)`` of a Table A2 plate."""
    rho, c_l, nu = _A2[name]
    modulus = _youngs_modulus(rho, c_l, nu)
    stiffness = plate_contact_stiffness(modulus, poisson_ratio=nu)
    impedance = infinite_plate_impedance(
        plate_bending_stiffness(modulus, thickness, nu), rho * thickness
    )
    return stiffness, impedance


# ===========================================================================
# The ISO tapping machine as a mass-spring-dashpot (Hopkins 3.6.3)
# ===========================================================================
def test_impact_velocity_is_the_printed_0_886() -> None:
    """H printed p. 276: "the hammer velocity at impact ... will be 0,886 m/s"."""
    assert hammer_impact_velocity() == pytest.approx(0.886, abs=0.0005)


def test_short_pulse_coefficient_reproduces_eq_3_92() -> None:
    """H Eq. (3.92) prints ``F2rms = 3,9 B``.

    Independent derivation from the two printed equations it is built on:
    ``|Fn| = 2 m vo/Ti`` (Eq. 3.90, printed p. 277) and
    ``F2rms = |Fn|^2 B/(2 fi)`` (Eq. 3.91, printed p. 278) give
    ``F2rms = 2 m^2 vo^2 fi B = 3,925 B``, printed rounded to 3,9.
    """
    v0 = hammer_impact_velocity()
    peak = 2.0 * TAPPING_HAMMER_MASS * v0 / 0.1
    derived = peak**2 / (2.0 * 10.0)
    assert derived == pytest.approx(3.925, abs=5e-3)
    # The library's own 3,9 B must agree with the printed coefficient.
    bandwidth = 0.23 * 500.0
    assert short_pulse_mean_square_force(500.0)[0] == pytest.approx(
        3.9 * bandwidth, rel=1e-12
    )
    assert derived == pytest.approx(3.9 * 1.0064, rel=2e-3)


def test_force_limits_differ_by_6_db_in_mean_square() -> None:
    """H printed p. 282: "the difference between the lower and upper limit is 6 dB".

    ``|Fn|lower = m vo/Ti`` (Eq. 3.99) and ``|Fn|upper = 2 m vo/Ti`` (Eq. 3.100).
    """
    stiffness, impedance = _plate("concrete", 0.14)
    result = tapping_force_spectrum([100.0], stiffness, impedance)
    ratio_db = 20.0 * np.log10(result.upper_limit / result.lower_limit)
    assert ratio_db == pytest.approx(6.0206, abs=1e-3)


@pytest.mark.parametrize(
    ("name", "thickness", "mass_per_area", "over_critical"),
    [
        # H printed p. 280-281 and Figs. 3.30/3.31: the concrete slab and the
        # screed rebound with an under-critical oscillation, "for the chipboard
        # and OSB plates there is no distinct rebound ... due to the
        # over-critical oscillation". The masses are the printed figure legends.
        ("concrete", 0.14, 308.0, False),
        ("screed", 0.065, 130.0, False),
        ("chipboard", 0.022, 17.0, True),
        ("osb", 0.015, 9.0, True),
    ],
)
def test_hopkins_four_plates_critical_case(
    name: str, thickness: float, mass_per_area: float, over_critical: bool
) -> None:
    """The printed over/under-critical classification of H Figs. 3.30 and 3.31.

    The legends of H Figs. 3.30/3.31 (printed p. 281) also print the mass per
    unit area of each plate, which the Table A2 density must reproduce.
    """
    rho = _A2[name][0]
    assert rho * thickness == pytest.approx(mass_per_area, abs=0.5)
    stiffness, impedance = _plate(name, thickness)
    result = tapping_force_spectrum([100.0], stiffness, impedance)
    assert result.over_critical is over_critical


def test_concrete_slab_force_is_within_1_db_of_the_upper_limit() -> None:
    """H printed p. 282, 140 mm concrete slab.

    "Below 4000 Hz the mean-square force values are within 1 dB of the upper
    limit, |Fn|upper. For concrete floor slabs of at least 100 mm thickness it
    is reasonable to estimate the mean-square force using Eq. 3.92".

    Checked on the one-third-octave centres of the building acoustics range
    below 4 kHz; the statement is tight, the worst deviation being 0,8 dB at
    3 150 Hz and passing 1 dB in the 4 kHz band itself.
    """
    stiffness, impedance = _plate("concrete", 0.14)
    freqs = np.array([
        50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
        500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0,
    ])
    result = tapping_force_spectrum(freqs, stiffness, impedance)
    deviation = 20.0 * np.log10(result.peak_force / result.upper_limit)
    assert np.max(np.abs(deviation)) <= 1.0
    # ... and Eq. (3.92) is then an adequate estimate of the band force.
    short = short_pulse_mean_square_force(freqs)
    assert np.max(np.abs(10.0 * np.log10(result.mean_square_force / short))) <= 1.0


@pytest.mark.parametrize(("name", "thickness"), [("chipboard", 0.022), ("osb", 0.015)])
def test_lightweight_plates_tend_to_the_lower_limit(name: str, thickness: float) -> None:
    """H printed p. 282: "with |Fn| tending towards the lower limit ... below 100 Hz"."""
    stiffness, impedance = _plate(name, thickness)
    result = tapping_force_spectrum([50.0, 63.0, 80.0, 100.0], stiffness, impedance)
    deviation = 20.0 * np.log10(result.peak_force / result.lower_limit)
    assert np.max(np.abs(deviation)) <= 0.7
    # ... and, unlike the concrete slab, well away from the upper limit.
    assert np.min(np.abs(
        20.0 * np.log10(result.peak_force / result.upper_limit)
    )) >= 5.0


def test_over_critical_cut_off_is_the_lower_root() -> None:
    """H printed p. 282, chipboard and OSB.

    "For the 22 mm chipboard and 15 mm OSB plates, the longer duration pulse
    means that the force spectrum is not flat and decreases above 100 Hz."
    Equation (3.101) has two roots; only the lower one is compatible with that
    statement, the upper one landing above 8 kHz. The branch is also continuous
    with the under-critical Eq. (3.102): at the critical boundary
    ``K m = 4 Zdp^2`` the discriminant vanishes and both give
    ``sqrt(K/m)/(2 pi)``.
    """
    for name, thickness in (("chipboard", 0.022), ("osb", 0.015)):
        stiffness, impedance = _plate(name, thickness)
        decay = stiffness / (2.0 * impedance)
        root = np.sqrt(decay**2 - stiffness / TAPPING_HAMMER_MASS)
        fco = tapping_cut_off_frequency(stiffness, impedance)
        assert fco == pytest.approx((decay - root) / (2.0 * np.pi))
        assert fco < 1000.0
        assert (decay + root) / (2.0 * np.pi) > 8000.0

    # Continuity across the critical boundary, from both sides.
    stiffness = 1.0e8
    critical = np.sqrt(stiffness * TAPPING_HAMMER_MASS) / 2.0
    undamped = np.sqrt(stiffness / TAPPING_HAMMER_MASS) / (2.0 * np.pi)
    assert tapping_cut_off_frequency(stiffness, critical * (1.0 - 1e-9)) == (
        pytest.approx(undamped, rel=1e-4)
    )
    assert tapping_cut_off_frequency(stiffness, critical * (1.0 + 1e-9)) == (
        pytest.approx(undamped, rel=1e-4)
    )


def test_bare_concrete_cut_off_is_about_7000_hz() -> None:
    """H printed p. 514: bare 140 mm slab, "fco = 7000 Hz" (Eqs. 3.97/3.102)."""
    stiffness, impedance = _plate("concrete", 0.14)
    fco = tapping_cut_off_frequency(stiffness, impedance)
    assert fco == pytest.approx(7000.0, rel=0.02)


def test_power_input_rises_3_db_per_doubling_below_1_khz() -> None:
    """H printed p. 284, concrete slab and screed.

    "the power input in one-third-octave or octave-bands increases by 3 dB per
    doubling of the band centre frequency. This occurs for the 140 mm concrete
    slab and the 65 mm sand-cement screed below 1000 Hz."
    """
    for name, thickness in (("concrete", 0.14), ("screed", 0.065)):
        stiffness, impedance = _plate(name, thickness)
        freqs = np.array([125.0, 250.0, 500.0, 1000.0])
        level = tapping_force_spectrum(freqs, stiffness, impedance).power_input_level
        assert np.allclose(np.diff(level), 3.0103, atol=0.35)


def test_limiting_frequency_matches_the_printed_figure_4_70_caption() -> None:
    """H Fig. 4.70 caption, printed p. 521.

    "Walking surface: 22 mm chipboard. Resilient layer: s = 4 MN/m3 ... Base
    floor: 140 mm concrete slab", with "fms = 83 Hz" and "flimit = 521 Hz".
    The two printed frequencies come from different formulae (Eq. 4.120 and
    Eq. 3.106) but from one specimen, so the mass per unit area implied by
    ``fms`` must also reproduce ``flimit`` through the Table A2 chipboard
    quasi-longitudinal speed.
    """
    printed_fms, printed_flimit = 83.0, 521.0
    thickness, c_l = 0.022, _A2["chipboard"][1]
    mass_per_area = 4.0e6 / (2.0 * np.pi * printed_fms) ** 2
    impedance = 8.0 * np.sqrt(
        plate_bending_stiffness(
            _youngs_modulus(mass_per_area / thickness, c_l, 0.3), thickness, 0.3
        )
        * mass_per_area
    )
    assert hammer_limiting_frequency(impedance) == pytest.approx(
        printed_flimit, rel=0.01
    )


def test_critical_damping_matches_vigran_pi_m_f0() -> None:
    """Two books, one criterion.

    H (printed p. 279) splits the cases at ``K m = 4 Zdp^2``; V (printed
    p. 320) states for the same spring-in-series-with-a-resistance model that
    "critical damping is obtained when the damping coefficient c is equal to
    pi m f0", with ``f0 = sqrt(s/m)/(2 pi)``. Setting ``Zdp = pi m f0`` in
    Hopkins's criterion must land exactly on the boundary.
    """
    mass, stiffness = TAPPING_HAMMER_MASS, 3.2e5
    f0 = np.sqrt(stiffness / mass) / (2.0 * np.pi)
    critical = np.pi * mass * f0
    assert stiffness * mass == pytest.approx(4.0 * critical**2, rel=1e-12)
    # Just either side of it the two branches must disagree.
    assert tapping_force_spectrum([100.0], stiffness, critical * 0.99).over_critical
    assert not tapping_force_spectrum([100.0], stiffness, critical * 1.01).over_critical


def test_hammer_contact_area_matches_vigran_seven_square_centimetres() -> None:
    """H Eq. (3.98) uses ``r = 15 mm``; V Eq. (8.51) uses ``Sh = 7 cm2``.

    V printed p. 318: "the effective stiffness of the layer related to the area
    of the hammer(s) Sh, the latter being 7 cm2", with ``s = E Sh/d``. The two
    printed forms agree to 1 %.
    """
    area = np.pi * TAPPING_HAMMER_RADIUS**2
    assert area == pytest.approx(7.0e-4, rel=0.02)
    modulus, thickness = 2.8e8 * 0.005, 0.005
    vigran = modulus * 7.0e-4 / thickness
    hopkins = covering_contact_stiffness(modulus, thickness)
    assert hopkins == pytest.approx(vigran, rel=0.02)


def test_force_pulse_transform_matches_the_closed_form_spectrum() -> None:
    """The closed-form spectrum equals a numerical transform of the pulse.

    :func:`tapping_force_spectrum` evaluates the analytic Fourier transform of
    Eqs. (3.95)/(3.96); this recomputes it by quadrature over the pulse of
    :func:`force_pulse`, truncated at the first zero for the under-critical
    case as H printed p. 280 requires ("only the initial force pulse that has
    zero or positive force values is used").
    """
    stiffness, impedance = _plate("concrete", 0.14)
    decay = stiffness / (2.0 * impedance)
    beta = np.sqrt(stiffness / TAPPING_HAMMER_MASS - decay**2)
    time = np.linspace(0.0, np.pi / beta, 200001)
    pulse = force_pulse(time, stiffness, impedance)
    assert np.all(pulse >= -1e-6)
    freqs = np.array([100.0, 1000.0, 5000.0])
    quadrature = np.array([
        np.trapezoid(pulse * np.exp(-2.0j * np.pi * f * time), time) for f in freqs
    ])
    result = tapping_force_spectrum(freqs, stiffness, impedance)
    assert np.allclose(result.peak_force, np.abs(quadrature) * 10.0, rtol=2e-3)


def test_over_critical_pulse_stays_positive() -> None:
    """H printed p. 279: "For over-critical oscillations, the force pulse decays
    to zero and takes only positive values"."""
    stiffness, impedance = _plate("chipboard", 0.022)
    time = np.linspace(0.0, 0.02, 5000)
    pulse = force_pulse(time, stiffness, impedance)
    assert np.all(pulse >= 0.0)
    assert pulse[-1] < pulse.max() * 1e-3


def test_troughs_occur_at_odd_multiples_of_the_cut_off() -> None:
    """H printed p. 514: "deep troughs in the force spectra above the cut-off
    frequency; these occur at frequencies n fco where n = 3, 5, 7, etc."."""
    impedance = _plate("concrete", 0.14)[1]
    stiffness = covering_contact_stiffness(2.8e8 * 0.005, 0.005)
    result = tapping_force_spectrum([1.0], stiffness, impedance)
    fco = result.cut_off_frequency
    freqs = np.logspace(np.log10(fco), np.log10(9.0 * fco), 40001)
    force = tapping_force_spectrum(freqs, stiffness, impedance).peak_force
    minima = freqs[1:-1][(force[1:-1] < force[:-2]) & (force[1:-1] < force[2:])]
    for n in (3, 5, 7):
        assert np.min(np.abs(minima / fco - n)) < 0.05


# ===========================================================================
# Soft floor coverings on a heavyweight floor (Hopkins 4.4.3.1)
# ===========================================================================
#: The two coverings of H Fig. 4.64 (printed p. 513): "For covering No. 1,
#: E/d = 1,5e11 N/m3, which is indicative of a few millimetres of solid PVC.
#: For covering No. 2, E/d = 2,8e8 N/m3", with the printed cut-off frequencies
#: "fco = 2300 Hz for covering No. 1, and fco = 100 Hz for covering No. 2"
#: (printed p. 514).
_COVERINGS = ((1.5e11, 2300.0), (2.8e8, 100.0))


@pytest.mark.parametrize(("stiffness_per_volume", "printed_fco"), _COVERINGS)
def test_covering_cut_off_frequencies(
    stiffness_per_volume: float, printed_fco: float
) -> None:
    """H printed p. 514: soft coverings on a 140 mm concrete slab."""
    impedance = _plate("concrete", 0.14)[1]
    thickness = 0.005
    stiffness = covering_contact_stiffness(
        stiffness_per_volume * thickness, thickness
    )
    fco = tapping_cut_off_frequency(stiffness, impedance)
    assert fco == pytest.approx(printed_fco, rel=0.01)


def test_covering_improvement_is_zero_below_the_cut_off() -> None:
    """H printed p. 514: "Below fco the soft floor covering does not
    significantly alter the force input compared to the bare slab; hence it
    does not improve the impact sound insulation ... Below fco, DeltaL is
    approximately 0 dB"."""
    plate_stiffness, impedance = _plate("concrete", 0.14)
    thickness = 0.005
    covering = covering_contact_stiffness(2.8e8 * thickness, thickness)
    freqs = np.array([50.0, 63.0, 80.0])
    result = covering_improvement(freqs, covering, plate_stiffness, impedance)
    assert np.all(np.abs(result.improvement) < 1.0)
    assert np.all(result.two_line == 0.0)


def test_two_line_estimate_rises_12_db_per_octave() -> None:
    """H printed p. 514: "the curves will tend towards a straight slope of
    12 dB/octave (equivalent to 40 dB/decade) for f >= fco"."""
    plate_stiffness, impedance = _plate("concrete", 0.14)
    thickness = 0.005
    covering = covering_contact_stiffness(2.8e8 * thickness, thickness)
    freqs = np.array([200.0, 400.0, 800.0, 1600.0])
    result = covering_improvement(freqs, covering, plate_stiffness, impedance)
    assert np.allclose(np.diff(result.two_line), 12.0411, atol=1e-3)
    # The force-ratio model approaches the same asymptote from above: 14,0 dB
    # over the first octave after fco, 12,1 dB three octaves later.
    assert np.allclose(np.diff(result.improvement), 12.0, atol=2.0)
    far = covering_improvement(
        np.array([800.0, 1600.0, 3200.0]), covering, plate_stiffness, impedance
    )
    assert np.allclose(np.diff(far.improvement), 12.0411, atol=0.3)


@pytest.mark.parametrize(
    ("stiffness", "printed_f0"),
    [
        # V printed p. 320: "using a covering of stiffness s equal to 3,2e5
        # N/m, giving a resonance frequency f0 of approximately 130 Hz with a
        # hammer mass of 0,5 kg"; printed p. 321: "a stiffness of 5,2e6 N/m is
        # used, equivalent to a resonance frequency of approximately 510 Hz".
        (3.2e5, 130.0),
        (5.2e6, 510.0),
    ],
)
def test_vigran_covering_resonance_frequencies(
    stiffness: float, printed_f0: float
) -> None:
    """V section 8.4.5, Figs. 8.36/8.37 (printed pp. 319-321).

    The covering resonance is Hopkins's under-critical cut-off, Eq. (3.102).
    """
    # A perfectly rigid base floor: the covering alone sets the cut-off.
    impedance = 1e9
    fco = tapping_cut_off_frequency(stiffness, impedance)
    assert fco == pytest.approx(printed_f0, rel=0.03)


def test_covering_improvement_result_carries_both_cut_offs() -> None:
    """The bare slab's own cut-off (about 7 kHz) is reported alongside the
    covering's, since above it the bare force falls too (H printed p. 514)."""
    plate_stiffness, impedance = _plate("concrete", 0.14)
    thickness = 0.005
    covering = covering_contact_stiffness(2.8e8 * thickness, thickness)
    result = covering_improvement([500.0], covering, plate_stiffness, impedance)
    assert result.bare_cut_off_frequency == pytest.approx(7000.0, rel=0.02)
    assert result.cut_off_frequency == pytest.approx(100.0, rel=0.01)
    assert result.bare.over_critical is False


# ===========================================================================
# Floating floors (ISO 12354-2 Annex C, Hopkins 4.4.4, Vigran 8.4)
# ===========================================================================
def test_annex_g_resonance_frequency_is_52_8_hz() -> None:
    """ISO 12354-2:2017 Table G.4 input block, printed p. 39.

    "Floating floor, m' = 73,5 kg/m2, s' = 8 MN/m3, f0 = 52,8 Hz" via
    Formula (C.2), ``fo = 160 sqrt(s'/m')``.
    """
    f0 = floating_floor_resonance_frequency(
        ref.ISO12354_ANNEX_L_FLOATING_STIFFNESS * 1e6,
        ref.ISO12354_ANNEX_L_FLOATING_MASS,
    )
    assert f0 == pytest.approx(ref.ISO12354_ANNEX_L_FLOATING_F0, abs=0.05)


def test_annex_g4_printed_delta_l_per_band() -> None:
    """ISO 12354-2:2017 Table G.4, column DeltaL,situ (printed p. 39 / pdf p. 45).

    All 21 printed one-third-octave values of Formula (C.1) for the Annex G
    floating floor. ISO 12354-1:2017 Table L.4 prints the identical column as
    DeltaRd,situ for the same floor.
    """
    bands = np.asarray(ref.ISO12354_ANNEX_L_BANDS)
    f0 = floating_floor_resonance_frequency(
        ref.ISO12354_ANNEX_L_FLOATING_STIFFNESS * 1e6,
        ref.ISO12354_ANNEX_L_FLOATING_MASS,
    )
    result = floating_floor_improvement_spectrum(bands, resonance_frequency=f0)
    assert np.max(
        np.abs(result.improvement - np.asarray(ref.ISO12354_ANNEX_G4_DELTA_L))
    ) <= 0.05


def test_weighted_improvement_matches_the_printed_32_2_db() -> None:
    """ISO 12354-2:2017 Formula (C.4) with the Annex G floating floor.

    ISO 12354-1:2017 Table L.10 / ISO 12354-2:2017 Table G.10 print
    "DeltaLw = 32,2 dB" for m' = 73,5 kg/m2 on s' = 8 MN/m3.
    """
    value = weighted_floating_floor_improvement(
        ref.ISO12354_ANNEX_L_FLOATING_MASS,
        ref.ISO12354_ANNEX_L_FLOATING_STIFFNESS * 1e6,
    )
    assert value == pytest.approx(ref.ISO12354_ANNEX_G10_DELTA_LW, abs=0.05)


def test_asphalt_branch_is_the_40_lg_law() -> None:
    """ISO 12354-2:2017 Formula (C.3), ``DeltaL = 40 lg(f/fo)`` for asphalt and
    dry floating floors; H Eq. (4.119) is the same law from Cremer's
    infinite-plate derivation, "12 dB per octave" (V printed p. 308)."""
    freqs = np.array([100.0, 200.0, 400.0, 800.0])
    result = floating_floor_improvement_spectrum(
        freqs, resonance_frequency=50.0, model="cremer"
    )
    assert np.allclose(np.diff(result.improvement), 12.0411, atol=1e-3)
    assert result.improvement[0] == pytest.approx(40.0 * np.log10(2.0), abs=1e-9)


def test_en12354_branch_is_the_30_lg_law() -> None:
    """ISO 12354-2:2017 Formula (C.1) / H Eq. (4.124): 30 lg(f/fo), i.e. the
    "9 dB per octave" of V printed p. 308, and 0 dB at and below ``fo``."""
    freqs = np.array([25.0, 50.0, 100.0, 200.0])
    result = floating_floor_improvement_spectrum(freqs, resonance_frequency=50.0)
    assert result.improvement[0] == 0.0
    assert result.improvement[1] == 0.0
    assert np.allclose(np.diff(result.improvement[1:]), [9.0309, 9.0309], atol=1e-3)


def test_combined_dynamic_stiffness_is_springs_in_series() -> None:
    """ISO 12354-2:2017 Formula (C.6) and H Eq. (4.121): ``s'tot = (sum 1/s'i)^-1``.

    Two identical layers halve the stiffness, so ``fo`` drops by sqrt(2).
    """
    assert combined_dynamic_stiffness([8e6, 8e6]) == pytest.approx(4e6, rel=1e-12)
    assert combined_dynamic_stiffness([1e7, 4e7]) == pytest.approx(8e6, rel=1e-12)
    single = floating_floor_resonance_frequency(8e6, 73.5)
    doubled = floating_floor_resonance_frequency(
        combined_dynamic_stiffness([8e6, 8e6]), 73.5
    )
    assert doubled == pytest.approx(single / np.sqrt(2.0), rel=1e-12)


def test_double_floating_floor_matches_printed_74_and_195_hz() -> None:
    """H Fig. 4.73 caption, printed p. 523.

    "Walking surface: 18 mm plywood. Resilient layer: s = 7,25 MN/m3 (25 mm
    reconstituted foam). Base floor: 140 mm concrete slab ... Single floating
    floor fms = 118 Hz. Double floating floor fmsms = 74 Hz and 195 Hz."
    The mass per unit area comes from the Table A2 plywood density
    (710 kg/m3), independently of the printed frequencies.
    """
    mass_per_area = _A2["plywood"][0] * 0.018
    stiffness = 7.25e6
    single = np.sqrt(stiffness / mass_per_area) / (2.0 * np.pi)
    assert single == pytest.approx(118.0, rel=0.02)
    lower, upper = double_floating_floor_resonances(
        stiffness, mass_per_area, stiffness, mass_per_area
    )
    assert lower == pytest.approx(74.0, rel=0.02)
    assert upper == pytest.approx(195.0, rel=0.02)


def test_double_floating_floor_identical_layers_give_the_golden_ratio() -> None:
    """Closed-form identity of H Eq. (4.125) for two identical floating floors.

    ``X = 3 wo^2`` and ``X^2 - 4 wo^4 = 5 wo^4``, so the roots are
    ``fms sqrt((3 -+ sqrt 5)/2)``, that is ``fms/phi`` and ``fms phi`` with
    ``phi`` the golden ratio.
    """
    stiffness, mass_per_area = 7.25e6, 12.78
    fms = np.sqrt(stiffness / mass_per_area) / (2.0 * np.pi)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    lower, upper = double_floating_floor_resonances(
        stiffness, mass_per_area, stiffness, mass_per_area
    )
    assert lower == pytest.approx(fms / phi, rel=1e-12)
    assert upper == pytest.approx(fms * phi, rel=1e-12)


@pytest.mark.parametrize(
    ("mass_per_area", "printed_f0"),
    [
        # V printed p. 317-318 ("Examples") and printed p. 310: a 50 mm
        # concrete slab (m' about 115 kg/m2) on 25 mm mineral wool of total
        # dynamic stiffness 8,0 MPa/m gives "a resonance frequency of
        # approximately 40 Hz"; the same layer under 22 mm chipboard plus
        # 13 mm plasterboard (m' about 28 kg/m2) gives "approximately 90 Hz".
        (115.0, 40.0),
        (28.0, 90.0),
    ],
)
def test_vigran_floating_floor_resonances(
    mass_per_area: float, printed_f0: float
) -> None:
    """V section 8.4.4, the printed floating-floor examples."""
    f0 = floating_floor_resonance_frequency(8.0e6, mass_per_area)
    assert f0 == pytest.approx(printed_f0, rel=0.07)


def test_cremer_hammer_branch_tends_to_18_db_per_octave() -> None:
    """V printed p. 312, Eq. (8.48) / H Eq. (4.123).

    "At sufficiently high frequencies, the frequency dependence will be as high
    as 18 dB per octave, a result completely determined by the specific mass of
    the hammer." Well above ``flimit`` the hammer term adds 6 dB per octave to
    the 12 dB per octave of the 40 lg law.
    """
    freqs = np.array([4000.0, 8000.0, 16000.0])
    result = floating_floor_improvement_spectrum(
        freqs, resonance_frequency=83.0, model="cremer_hammer",
        limiting_frequency=521.0,
    )
    assert np.allclose(np.diff(result.improvement), 18.0, atol=0.1)
    # Well below flimit the hammer term vanishes and the 40 lg law is recovered.
    low = np.array([100.0, 130.0])
    plain = floating_floor_improvement_spectrum(low, resonance_frequency=83.0)
    hammer = floating_floor_improvement_spectrum(
        low, resonance_frequency=83.0, model="cremer_hammer",
        limiting_frequency=521.0,
    )
    assert np.all(hammer.improvement - plain.improvement * 4.0 / 3.0 < 0.3)


def test_resilient_mount_model_rises_30_db_per_decade() -> None:
    """H printed p. 519, Eq. (4.118).

    "The equation usefully shows that above the mass-spring-mass resonance
    frequency, DeltaL increases at 30 dB/decade" (9 dB per octave), against the
    40 dB/decade of a continuous resilient layer.
    """
    freqs = np.array([100.0, 1000.0])
    values = resilient_mount_improvement(
        freqs, impedance=3.8e5, mass_per_area=115.0, loss_factor=0.02,
        mount_stiffness=2.0e6, mount_density=4.0,
    )
    assert values[1] - values[0] == pytest.approx(30.0, abs=1e-9)


def test_ver_model_agrees_between_hopkins_and_vigran() -> None:
    """Two books, one SEA model, two different algebraic statements.

    H Eq. (4.118), printed p. 519:
    ``DeltaL = 10 lg(2,3 rhos1^2 cL1 h1 eta1 S1 omega^3/(N k^2))`` with ``N``
    the total number of mounts over the area ``S1``.
    V Eq. (8.45), printed p. 309, dominant term:
    ``10 lg(Z1 eta1 N f^3/(2 pi m1 fo^4))`` with ``N`` mounts per unit area and
    ``fo = sqrt(N s/m1)/(2 pi)``.
    Substituting ``fo`` shows the two are the same expression; the test pins
    that identity rather than either transcription.
    """
    rho, c_l, thickness = 2200.0, 3800.0, 0.05
    mass_per_area = rho * thickness
    impedance_hopkins = 2.3 * rho * c_l * thickness**2
    eta, stiffness, per_area, area = 0.02, 2.0e6, 4.0, 20.0
    freqs = np.array([100.0, 250.0, 630.0, 1600.0])
    omega = 2.0 * np.pi * freqs

    hopkins = 10.0 * np.log10(
        2.3 * mass_per_area**2 * c_l * thickness * eta * area * omega**3
        / (per_area * area * stiffness**2)
    )
    f0 = np.sqrt(per_area * stiffness / mass_per_area) / (2.0 * np.pi)
    vigran = 10.0 * np.log10(
        impedance_hopkins * eta * per_area * freqs**3
        / (2.0 * np.pi * mass_per_area * f0**4)
    )
    assert np.allclose(hopkins, vigran, atol=1e-9)

    library = resilient_mount_improvement(
        freqs,
        impedance=infinite_plate_impedance(
            plate_bending_stiffness(_youngs_modulus(rho, c_l, 0.2), thickness, 0.2),
            mass_per_area,
        ),
        mass_per_area=mass_per_area, loss_factor=eta,
        mount_stiffness=stiffness, mount_density=per_area,
    )
    # 8 sqrt(B' m'') = 2,3094 rho cL h^2 against Hopkins's rounded 2,3.
    assert np.allclose(library, hopkins, atol=0.02)


def test_asphalt_weighted_improvement_exceeds_the_screed_branch() -> None:
    """ISO 12354-2:2017 Formula (C.5) has no printed worked example.

    Figures C.1 and C.2 are nomograms, so only the qualitative relation the
    standard states can be pinned: asphalt and dry floating floors follow the
    steeper 40 lg law (Formula C.3) and therefore rate higher than a
    sand-cement screed of the same mass on the same resilient layer.
    """
    screed = weighted_floating_floor_improvement(73.5, 8e6)
    asphalt = weighted_floating_floor_improvement(73.5, 8e6, floor="asphalt")
    assert asphalt > screed
    # Both fall as the resilient layer stiffens.
    assert weighted_floating_floor_improvement(73.5, 2e7) < screed
    # The transcription itself is pinned by restating the printed formula,
    # DeltaLw = ((-0,21 m') - 5,45) lg(s') + (0,46 m') + 23,8, with s' in
    # MN/m3, independently of the implementation.
    for mass, stiffness in ((73.5, 8.0), (120.0, 20.0), (25.0, 3.0)):
        printed = (
            (-0.21 * mass - 5.45) * np.log10(stiffness) + 0.46 * mass + 23.8
        )
        assert weighted_floating_floor_improvement(
            mass, stiffness * 1e6, floor="asphalt"
        ) == pytest.approx(printed, abs=1e-9)


# ===========================================================================
# Wall linings and additional layers (ISO 12354-1 Annex D)
# ===========================================================================
def test_formula_d1_reproduces_the_printed_hopkins_lining_resonance() -> None:
    """H Fig. 4.48(c) caption, printed p. 486.

    "100 mm aircrete block wall (rhos = 51 kg/m2) ... Lining: 9,5 mm
    plasterboard (6,3 kg/m2) laminated with 32 mm expanded polystyrene
    (0,4 kg/m2, s = 65 MN/m3) attached with gypsum based adhesive dabs ...
    Mass-resilient material-mass: fmsm = 542 Hz."
    This is ISO 12354-1:2017 Formula (D.1) with the resilient layer bonded
    directly to the wall.
    """
    f0 = lining_resonance_frequency(51.0, 6.3, dynamic_stiffness=65e6)
    assert f0 == pytest.approx(542.0, rel=0.01)


@pytest.mark.parametrize(
    ("base", "lining", "gap", "printed"),
    [
        # H Fig. 4.48 captions, printed p. 486, all on the same 100 mm aircrete
        # block wall (rhos = 51 kg/m2):
        # (a) 12,5 mm plasterboard (10,8 kg/m2), 37 mm cavity with rock fibre
        #     quilt -> "Mass-air-mass: fmsm = 104 Hz";
        # (b) 9,5 mm plasterboard + 32 mm XPS (6,3 + 1,1 kg/m2), 37 mm empty
        #     cavity -> "fmsm = 123 Hz";
        # (c) 9,5 mm plasterboard + 32 mm EPS (6,3 + 0,4 kg/m2), 10 mm empty
        #     cavity -> "Mass-air-mass: fmsm = 247 Hz".
        (51.0, 10.8, 0.037, 104.0),
        (51.0, 7.4, 0.037, 123.0),
        (51.0, 6.7, 0.010, 247.0),
    ],
)
def test_hopkins_mass_air_mass_lining_resonances(
    base: float, lining: float, gap: float, printed: float
) -> None:
    """The adiabatic cavity stiffness ``rho0 c0^2/d`` of H Eq. (4.72)."""
    assert mass_spring_mass_resonance(base, lining, gap) == pytest.approx(
        printed, rel=0.01
    )


def test_formula_d2_is_formula_d1_with_the_filled_cavity_stiffness() -> None:
    """ISO 12354-1:2017 Formula (D.2), printed p. 38.

    A stud cavity filled with a porous layer of ``r >= 5 kPa s/m2`` is given the
    stiffness ``0,111/d`` MN/m3, i.e. a near-isothermal value: it sits about
    10 % above the isothermal ``p0/d = 0,1013/d`` and about 22 % below the
    adiabatic ``rho0 c0^2/d`` that H uses for an empty cavity, so a filled
    cavity resonates below an empty one of the same depth. Formula (D.2) has no
    printed worked example; only this bracketing and its identity with
    Formula (D.1) can be pinned.
    """
    base, lining, depth = 51.0, 10.8, 0.037
    filled = lining_resonance_frequency(base, lining, cavity_depth=depth)
    assert filled == pytest.approx(
        lining_resonance_frequency(base, lining, dynamic_stiffness=0.111e6 / depth),
        rel=1e-12,
    )
    isothermal = lining_resonance_frequency(
        base, lining, dynamic_stiffness=101325.0 / depth
    )
    adiabatic = mass_spring_mass_resonance(base, lining, depth)
    assert isothermal < filled < adiabatic


def test_table_d1_low_frequency_branch() -> None:
    """ISO 12354-1:2017 Table D.1, printed p. 39.

    "30 <= f0 <= 160 -> DeltaRw = 74,4 - 20 lg(f0) - Rw/2", with NOTE 1 "For
    resonance frequencies below 200 Hz, the minimum value of DeltaRw is 0 dB".
    """
    assert weighted_lining_improvement(50.0, 50.0) == pytest.approx(
        74.4 - 20.0 * np.log10(50.0) - 25.0, abs=1e-9
    )
    assert weighted_lining_improvement(100.0, 40.0) == pytest.approx(
        74.4 - 40.0 - 20.0, abs=1e-9
    )
    # NOTE 1's floor. Inside the stated validity box (30 Hz to 160 Hz,
    # 20 dB <= Rw <= 60 dB) the branch never reaches 0 dB: its minimum is
    # 74,4 - 20 lg(160) - 30 = 0,32 dB, so the floor only binds for an Rw
    # above the range Clause D.2.2 states.
    assert weighted_lining_improvement(160.0, 60.0) == pytest.approx(
        74.4 - 20.0 * np.log10(160.0) - 30.0, abs=1e-9
    )
    assert weighted_lining_improvement(160.0, 62.0) == 0.0


@pytest.mark.parametrize(
    ("resonance", "printed"),
    [(200.0, -1.0), (250.0, -3.0), (315.0, -5.0), (400.0, -7.0), (500.0, -9.0),
     (630.0, -10.0), (1000.0, -10.0), (1250.0, -10.0), (2000.0, -5.0),
     (5000.0, -5.0)],
)
def test_table_d1_fixed_rows(resonance: float, printed: float) -> None:
    """ISO 12354-1:2017 Table D.1, the rows above 160 Hz (printed p. 39).

    Above 200 Hz the improvement is negative and independent of ``Rw``.
    """
    assert weighted_lining_improvement(resonance, 45.0) == printed
    assert weighted_lining_improvement(resonance, 30.0) == printed


def test_table_d1_rounds_to_the_third_octave_band() -> None:
    """ISO 12354-1:2017 Clause D.2.2, printed p. 38: ``fo`` is "rounded to the
    centre frequency of the one-third-octave band in which fo falls"."""
    assert weighted_lining_improvement(238.0, 45.0) == -3.0
    assert weighted_lining_improvement(279.0, 45.0) == -3.0
    assert weighted_lining_improvement(281.0, 45.0) == -5.0


def test_table_d1_1600_hz_overlap_takes_the_conservative_value() -> None:
    """ISO 12354-1:2017 Table D.1 prints 1 600 Hz in two rows with different
    values, "630 to 1 600 -> -10" and "1 600 <= f0 <= 5 000 -> -5"; see
    docs/ERRATA.md. The library takes the more conservative -10 dB."""
    assert weighted_lining_improvement(1600.0, 45.0) == -10.0
    assert weighted_lining_improvement(2000.0, 45.0) == -5.0


@pytest.mark.parametrize("resonance", [20.0, 6000.0])
def test_table_d1_rejects_untabulated_resonances(resonance: float) -> None:
    """Table D.1 covers 30 Hz to 5 000 Hz only."""
    with pytest.raises(ValueError, match="Table D.1"):
        weighted_lining_improvement(resonance, 45.0)


def test_exterior_lining_formulae_d3_and_d4() -> None:
    """ISO 12354-1:2017 Formulae (D.3)/(D.4), printed p. 40.

    Mineral wool: -36 lg(fo) + 82,5 / -42 lg(fo) + 92,0 / -39 lg(fo) + 87,7,
    each ">= -4"; foams: -33 lg(fo) + 76,0 / -33 lg(fo) + 74,0 /
    -36 lg(fo) + 77,0, each ">= -3".
    """
    wool = lining_improvement(100.0)
    assert wool.ratings == pytest.approx((82.5 - 72.0, 92.0 - 84.0, 87.7 - 78.0))
    foam = lining_improvement(100.0, system="foam")
    assert foam.ratings == pytest.approx((76.0 - 66.0, 74.0 - 66.0, 77.0 - 72.0))
    # The printed floors bind at a high resonance frequency.
    assert lining_improvement(1000.0).ratings == pytest.approx((-4.0, -4.0, -4.0))
    assert lining_improvement(1000.0, system="foam").ratings == pytest.approx(
        (-3.0, -3.0, -3.0)
    )


def test_anchor_and_glued_area_corrections_d5_and_d6() -> None:
    """ISO 12354-1:2017 Formulae (D.5)/(D.6), printed p. 40.

    (D.5) ``0,66 DeltaRw,ref - 1,2``, ``0,62 DeltaRA,ref - 1,3``,
    ``0,54 DeltaRA,tr,ref - 1,6``; (D.6) ``DeltaR - 0,05 %So + 2,0``, which is
    the identity at the 40 % reference glued area.
    """
    reference = lining_improvement(100.0)
    anchored = lining_improvement(100.0, anchors=True)
    assert anchored.delta_rw == pytest.approx(0.66 * reference.delta_rw - 1.2)
    assert anchored.delta_ra == pytest.approx(0.62 * reference.delta_ra - 1.3)
    assert anchored.delta_ratr == pytest.approx(0.54 * reference.delta_ratr - 1.6)
    at_reference = lining_improvement(100.0, glued_area=40.0)
    assert at_reference.ratings == pytest.approx(reference.ratings)
    # More glued area couples the layer to the wall and costs improvement.
    assert lining_improvement(100.0, glued_area=80.0).delta_rw < reference.delta_rw


def test_stud_system_formula_d7() -> None:
    """ISO 12354-1:2017 Formula (D.7), printed p. 41: ``-20 lg(fo) + 48``,
    ``-22 lg(fo) + 51``, ``-24 lg(fo) + 54``, each ">= -4"."""
    studs = lining_improvement(100.0, system="studs")
    assert studs.ratings == pytest.approx((48.0 - 40.0, 51.0 - 44.0, 54.0 - 48.0))
    assert lining_improvement(5000.0, system="studs").delta_rw == -4.0


def test_in_situ_transfer_formula_d8() -> None:
    """ISO 12354-1:2017 Formula (D.8), printed p. 41.

    ``DeltaRsitu = DeltaRlab + a X`` with ``a = 1,35 lg(fo) - 3,5 <= 0`` and
    ``X = Rw,situ - 53`` clamped to ``[-10, +7]``. The cap on ``a`` bites above
    ``fo = 10^(3,5/1,35) = 392 Hz``, where the transfer becomes the identity.
    """
    a = 1.35 * np.log10(100.0) - 3.5
    assert lining_improvement_in_situ(10.0, 100.0, 60.0) == pytest.approx(
        10.0 + a * 7.0
    )
    assert lining_improvement_in_situ(10.0, 100.0, 53.0) == pytest.approx(10.0)
    # X clamps at -10 and +7.
    assert lining_improvement_in_situ(10.0, 100.0, 20.0) == pytest.approx(
        10.0 + a * -10.0
    )
    assert lining_improvement_in_situ(10.0, 100.0, 70.0) == pytest.approx(
        10.0 + a * 7.0
    )
    # a is capped at 0 above about 392 Hz.
    assert lining_improvement_in_situ(10.0, 500.0, 30.0) == pytest.approx(10.0)


# ===========================================================================
# Result plumbing, plots and error handling
# ===========================================================================
def test_plots_return_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every public result exposes a working ``.plot()`` in both languages."""
    import matplotlib

    matplotlib.use("Agg")
    plate_stiffness, impedance = _plate("concrete", 0.14)
    covering = covering_contact_stiffness(2.8e8 * 0.005, 0.005)
    freqs = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    results = [
        tapping_force_spectrum(freqs, plate_stiffness, impedance),
        covering_improvement(freqs, covering, plate_stiffness, impedance),
        floating_floor_improvement_spectrum(
            freqs, resonance_frequency=52.8, mass_per_area=73.5,
            dynamic_stiffness=8e6,
        ),
        lining_improvement(120.0),
    ]
    for result in results:
        for language in ("en", "es"):
            ax = result.plot(language=language)
            assert ax is not None
            ax.figure.clf()


def test_power_input_level_is_consistent_with_the_power_input() -> None:
    stiffness, impedance = _plate("concrete", 0.14)
    result = tapping_force_spectrum([500.0], stiffness, impedance)
    assert result.power_input_level[0] == pytest.approx(
        10.0 * np.log10(result.power_input[0] / 1e-12)
    )
    assert result.mean_square_force[0] == pytest.approx(
        result.power_input[0] * impedance
    )


def test_octave_bandwidth_factor() -> None:
    """H Eq. (3.91): ``B = 0,23 f`` for thirds, ``B = 0,707 f`` for octaves."""
    third = short_pulse_mean_square_force(1000.0, band="third")[0]
    octave = short_pulse_mean_square_force(1000.0, band="octave")[0]
    assert octave / third == pytest.approx(0.707 / 0.23, rel=1e-12)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: hammer_impact_velocity(-1.0), "drop_height"),
        (lambda: plate_contact_stiffness(1e9, poisson_ratio=1.5), "poisson_ratio"),
        (lambda: covering_contact_stiffness(1e9, 0.0), "thickness"),
        (lambda: tapping_force_spectrum([100.0], 1e6, 1e3, band="half"), "band"),
        (lambda: force_pulse([-1.0], 1e6, 1e3), "time"),
        (lambda: combined_dynamic_stiffness([]), "layers"),
        (lambda: weighted_floating_floor_improvement(1.0, 1e6, floor="cork"), "floor"),
        (lambda: lining_improvement(100.0, system="cork"), "system"),
        (lambda: lining_resonance_frequency(50.0, 10.0), "exactly one"),
        (
            lambda: lining_resonance_frequency(
                50.0, 10.0, dynamic_stiffness=1e6, cavity_depth=0.05
            ),
            "exactly one",
        ),
        (
            lambda: floating_floor_improvement_spectrum(
                [100.0], resonance_frequency=50.0, model="cremer_hammer"
            ),
            "limiting_frequency",
        ),
        (
            lambda: floating_floor_improvement_spectrum(
                [100.0], resonance_frequency=50.0, model="ver"
            ),
            "model",
        ),
        (lambda: weighted_lining_improvement(100.0, float("nan")), "base_rating"),
        (lambda: lining_improvement_in_situ(float("inf"), 100.0, 50.0), "finite"),
    ],
)
def test_invalid_inputs_raise(call: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()  # type: ignore[operator]
