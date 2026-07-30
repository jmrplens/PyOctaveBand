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

import warnings

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
from phonometry.building.resilient_layers import (
    _THIRD_OCTAVE_CENTRES,
    _THIRD_OCTAVE_FIRST_INDEX,
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


def test_limiting_frequency_matches_the_printed_figure_4_70_panel() -> None:
    """H Fig. 4.70, printed p. 521 / pdf p. 548.

    The construction and both frequencies are printed *inside the plot panel*,
    not in the caption: "Walking surface: 22 mm chipboard", "Resilient layer:
    s' = 4 MN/m3 (45 mm reconstituted foam formed from two layers of foam)",
    "Base floor: 140 mm concrete slab", "fms = 83 Hz", "flimit = 521 Hz".

    The two frequencies come from different formulae (Eq. 4.120 and Eq. 3.106)
    but from one specimen, so the mass per unit area implied by ``fms`` must
    also reproduce ``flimit`` through the Table A2 chipboard quasi-longitudinal
    speed. That is what is pinned here, and it is a *consistency* anchor
    rather than a transcription one.

    It is worth being explicit about how strong it is. Working back from
    ``fms = 83 Hz`` gives ``rho_s = 14,71 kg/m2``, i.e. a chipboard density of
    668,5 kg/m3, and working back from ``flimit = 521 Hz`` gives 668,3 kg/m3:
    the two agree to 0,03 %, which is why the identity holds. But neither is
    the 760 kg/m3 that Table A2 prints for chipboard and that this file uses
    everywhere else; with 760 the same construction would give 77,9 Hz and
    592 Hz, not 83 and 521. Hopkins gives no density for the Fig. 4.70
    specimen anywhere (the "22 mm chipboard (17 kg/m2)" of Figs. 3.31 to 3.33
    is a third value again, 773 kg/m3), so the anchor pins the two formulae
    against each other and not against Table A2.
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


def test_over_critical_pulse_is_finite_over_the_whole_impact_period() -> None:
    """The pulse must survive the machine's own 0,1 s between impacts.

    The over-critical solution of Eq. (3.95) is
    ``vo K e^(-a t) sinh(gamma t)/gamma``. Evaluated as printed, the
    exponential underflows to zero while the hyperbolic sine overflows, and
    ``0 * inf`` is NaN. For Hopkins's 22 mm chipboard (over-critical, printed
    p. 280) that happens at t = 0,0278 s, well inside ``Ti = 1/fi = 0,1 s``,
    so a caller sampling one impact period would get NaN for most of it.
    """
    stiffness, impedance = _plate("chipboard", 0.022)
    time = np.linspace(0.0, 1.0 / 10.0, 1001)
    pulse = force_pulse(time, stiffness, impedance)
    assert np.all(np.isfinite(pulse))
    assert np.all(pulse >= 0.0)
    assert np.isfinite(force_pulse([0.1], stiffness, impedance)).all()
    # The tail is a pure decay, so it must be monotonic after the peak and
    # reach zero rather than diverge.
    peak = int(np.argmax(pulse))
    assert np.all(np.diff(pulse[peak:]) <= 0.0)
    assert pulse[-1] == pytest.approx(0.0, abs=1e-12)


def test_over_critical_pulse_is_still_equation_3_95() -> None:
    """The stable evaluation changes the rounding, not the function.

    Two independent checks of Eq. (3.95),
    ``F1 = vo K e^(-a t) sinh(gamma t)/gamma``:

    * where the printed form is representable in double precision (up to
      about 0,025 s for this plate) the two agree to 1e-12 relative;
    * beyond that, Eq. (3.95) is analytically a pure exponential of rate
      ``a - gamma``, since the second exponential of
      ``sinh = (e^(gamma t) - e^(-gamma t))/2`` has long since vanished. The
      logarithmic slope of the returned pulse must equal ``-(a - gamma)``.
    """
    stiffness, impedance = _plate("chipboard", 0.022)
    v0 = hammer_impact_velocity()
    decay = stiffness / (2.0 * impedance)
    gamma = np.sqrt(decay**2 - stiffness / TAPPING_HAMMER_MASS)

    safe = np.linspace(1e-7, 0.02, 4001)
    printed = v0 * stiffness * np.exp(-decay * safe) * np.sinh(gamma * safe) / gamma
    assert np.all(np.isfinite(printed))
    assert np.allclose(force_pulse(safe, stiffness, impedance), printed, rtol=1e-12)

    tail = np.array([0.03, 0.05, 0.08, 0.1])
    values = force_pulse(tail, stiffness, impedance)
    assert np.all(values > 0.0)
    slope = np.diff(np.log(values)) / np.diff(tail)
    assert np.allclose(slope, -(decay - gamma), rtol=1e-9)
    # And the amplitude of that exponential is vo K/(2 gamma).
    assert np.allclose(
        values, v0 * stiffness * np.exp(-(decay - gamma) * tail) / (2.0 * gamma),
        rtol=1e-12,
    )


def test_under_critical_pulse_is_truncated_at_the_first_zero_crossing() -> None:
    """H printed p. 280: "only the initial force pulse that has zero or
    positive force values is used ... with all subsequent values of F1(t) due
    to the oscillations set to zero".

    The cut is at ``t = pi/beta`` with ``beta = sqrt(K/m - (K/2 Zdp)^2)``, the
    first zero of the sine of Eq. (3.96). Neither ``2 pi/beta`` (which keeps a
    whole negative lobe), nor ``pi/omega_o`` (which cuts the positive lobe
    short), nor leaving the pulse untruncated satisfies the printed rule.
    """
    stiffness, impedance = _plate("concrete", 0.14)
    decay = stiffness / (2.0 * impedance)
    omega0 = np.sqrt(stiffness / TAPPING_HAMMER_MASS)
    beta = np.sqrt(omega0**2 - decay**2)
    duration = np.pi / beta
    # Positive right up to the cut, exactly zero just after it.
    assert force_pulse([duration * 0.999], stiffness, impedance)[0] > 0.0
    assert force_pulse([duration * 1.001], stiffness, impedance)[0] == 0.0
    # An untruncated Eq. (3.96) would be negative in the next half period.
    negative = duration * 1.5
    assert np.sin(beta * negative) < 0.0
    assert force_pulse([negative], stiffness, impedance)[0] == 0.0
    # pi/omega_o is a different, shorter time, and the pulse is still positive
    # and rising nowhere near zero there.
    assert np.pi / omega0 < duration
    assert force_pulse([np.pi / omega0], stiffness, impedance)[0] > 0.0
    assert np.all(force_pulse(np.linspace(0.0, duration, 401), stiffness,
                              impedance) >= 0.0)


def test_spectrum_truncation_uses_the_same_duration_as_the_pulse() -> None:
    """The closed-form transform must integrate exactly the truncated pulse.

    :func:`tapping_force_spectrum` multiplies by
    ``1 + e^(-a T) e^(-i omega T)`` with ``T = pi/beta``. Any other ``T``
    breaks the agreement with a quadrature of :func:`force_pulse`, which is
    what this pins; the previous check evaluated the quadrature only over
    ``[0, pi/beta]``, so a wrong ``T`` in the transform stayed invisible.
    """
    stiffness, impedance = _plate("concrete", 0.14)
    decay = stiffness / (2.0 * impedance)
    beta = np.sqrt(stiffness / TAPPING_HAMMER_MASS - decay**2)
    duration = np.pi / beta
    # Integrate over three times the pulse duration: the extra range is zero
    # in the pulse, so a transform truncated at 2 pi/beta or pi/omega_o can no
    # longer agree with it.
    time = np.linspace(0.0, 3.0 * duration, 300001)
    pulse = force_pulse(time, stiffness, impedance)
    assert np.all(pulse[time > duration] == 0.0)
    freqs = np.array([200.0, 1500.0, 4000.0, 9000.0])
    quadrature = np.array([
        np.trapezoid(pulse * np.exp(-2.0j * np.pi * f * time), time) for f in freqs
    ])
    result = tapping_force_spectrum(freqs, stiffness, impedance)
    assert np.allclose(result.peak_force, np.abs(quadrature) * 10.0, rtol=3e-3)


def test_spectrum_truncation_holds_for_a_heavily_damped_impact() -> None:
    """The same identity where ``pi/beta`` and ``pi/omega_o`` are far apart.

    On a concrete slab the hammer impact is barely damped: ``a/omega_o`` is
    0,029, so ``beta = sqrt(omega_o^2 - a^2)`` is within 0,04 % of
    ``omega_o`` and *any* test on that specimen accepts either as the
    truncation time. Just below the critical point they differ by a factor of
    five, which is where the distinction is actually observable.
    """
    stiffness, mass = 1.0e8, TAPPING_HAMMER_MASS
    impedance = 3600.0                     # 4 Zdp^2 = 5,18e7 > K m = 5,0e7
    assert stiffness * mass < 4.0 * impedance**2
    decay = stiffness / (2.0 * impedance)
    omega0 = np.sqrt(stiffness / mass)
    beta = np.sqrt(omega0**2 - decay**2)
    assert np.pi / beta > 5.0 * np.pi / omega0
    duration = np.pi / beta

    time = np.linspace(0.0, 2.0 * duration, 400001)
    pulse = force_pulse(time, stiffness, impedance)
    assert np.all(pulse[time > duration] == 0.0)
    assert pulse[time < duration].min() >= 0.0
    freqs = np.array([300.0, 1200.0, 3000.0])
    quadrature = np.array([
        np.trapezoid(pulse * np.exp(-2.0j * np.pi * f * time), time) for f in freqs
    ])
    result = tapping_force_spectrum(freqs, stiffness, impedance)
    assert result.over_critical is False
    assert np.allclose(result.peak_force, np.abs(quadrature) * 10.0, rtol=3e-3)


def test_critical_case_is_inclusive_and_finite() -> None:
    """H printed p. 279 splits the cases at "K m >= 4 Z_dp^2", inclusively.

    The sign is load-bearing rather than cosmetic. At exact equality the
    under-critical branch has ``beta = 0``: its truncation time ``pi/beta`` is
    infinite and its spectrum is ``nan + nan j``. The inclusive ``>=`` sends
    that case to the over-critical branch, whose critically damped limit
    ``F1 = vo K t e^(-a t)`` is finite.
    """
    mass = TAPPING_HAMMER_MASS
    stiffness = 1.0e8
    critical = np.sqrt(stiffness * mass) / 2.0        # K m == 4 Zdp^2 exactly
    assert stiffness * mass == pytest.approx(4.0 * critical**2, rel=1e-12)
    result = tapping_force_spectrum([100.0, 1000.0], stiffness, critical)
    assert result.over_critical is True
    assert np.all(np.isfinite(result.peak_force))
    assert np.all(result.peak_force > 0.0)
    pulse = force_pulse([0.0, 1e-4, 1e-3, 0.1], stiffness, critical)
    assert np.all(np.isfinite(pulse))
    assert pulse[0] == 0.0
    # The critically damped limit is the t -> 0 limit of both neighbours.
    nearby = force_pulse([1e-4], stiffness, critical * (1.0 + 1e-9))[0]
    assert pulse[1] == pytest.approx(nearby, rel=1e-6)


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
    # The band model approaches the same asymptote from above. Octave bands
    # average the truncation ripple (whose period is 4 fco = 400 Hz) far more
    # evenly than one-third-octave bands do, so the convergence is read there.
    octaves = covering_improvement(
        np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0]),
        covering, plate_stiffness, impedance, band="octave",
    )
    steps = np.diff(octaves.improvement)
    assert np.all(steps > 11.0)
    assert np.all(np.diff(steps) < 0.0)          # monotonically approaching
    assert steps[-1] == pytest.approx(12.0411, abs=0.6)
    # Above the *bare* slab's own cut-off (about 7 kHz) the uncovered force
    # falls too and DeltaL stops rising, so the asymptote is not read there.
    far = covering_improvement(
        np.array([8000.0, 16000.0]), covering, plate_stiffness, impedance,
        band="octave",
    )
    assert np.diff(far.improvement)[0] < 6.0


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


def test_covering_improvement_is_a_band_value_not_a_single_fourier_line() -> None:
    """The tapping machine excites lines at multiples of ``fi = 10 Hz``.

    H Eq. (4.114) is a statement about one Fourier component, so a band value
    is the ratio of the band mean-square forces (Eq. 3.91), summed over the
    lines the band contains. Evaluating the ratio at the band centre alone is
    not that, and the difference is not small: the undamped model's transform
    has exact nulls at odd multiples of ``fco`` (H printed p. 514, "deep
    troughs ... at frequencies n fco where n = 3, 5, 7"), so a band centre
    that lands on one reads tens of dB high.

    Hopkins's covering No. 2 has ``fco = 100,13 Hz``, which puts 500 Hz within
    0,2 % of ``5 fco``. Every earlier test of this function used
    200/400/800/1600/3200 Hz, all *even* multiples, so none of them saw it.
    """
    plate_stiffness, impedance = _plate("concrete", 0.14)
    thickness = 0.005
    covering = covering_contact_stiffness(2.8e8 * thickness, thickness)
    odd = np.array([300.0, 500.0, 700.0])          # 3 fco, 5 fco, 7 fco
    result = covering_improvement(odd, covering, plate_stiffness, impedance)
    assert result.cut_off_frequency == pytest.approx(100.0, rel=0.01)

    # The per-line ratio still carries the nulls, and is where they belong.
    line = result.line_improvement[np.isin(result.lines, odd)]
    assert np.all(line - result.two_line > 35.0)

    # The band value does not: it stays within a few dB of the design
    # estimate, as the two-line rule of H printed p. 514 requires.
    assert np.all(np.abs(result.improvement - result.two_line) < 10.0)
    assert result.improvement[1] < line[1] - 30.0

    # The nulls are genuinely nulls: the covered force at 500 Hz is more than
    # 30 dB below its neighbours 20 Hz away, so nothing here is a rounding
    # artefact of the tolerance.
    index = int(np.argmin(np.abs(result.lines - 500.0)))
    trough = result.covered.peak_force[index]
    assert 20.0 * np.log10(result.covered.peak_force[index - 2] / trough) > 25.0


def test_covering_band_average_is_independent_of_where_the_grid_falls() -> None:
    """Neighbouring band centres must not disagree by tens of dB.

    A line-spectrum attribute jumps by 39 dB between 400 Hz and 500 Hz on this
    specimen; a band average moves smoothly, which is the property that makes
    the number usable on the standard one-third-octave grid.
    """
    plate_stiffness, impedance = _plate("concrete", 0.14)
    thickness = 0.005
    covering = covering_contact_stiffness(2.8e8 * thickness, thickness)
    grid = np.array([125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0,
                     800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0])
    result = covering_improvement(grid, covering, plate_stiffness, impedance)
    assert np.all(result.improvement > 0.0)
    steps = np.diff(result.improvement)
    assert np.all(steps > -11.0)
    assert np.all(steps < 17.0)
    # The line ratio on the same grid is far wilder.
    lines = result.line_improvement[np.isin(result.lines, grid)]
    assert np.max(np.abs(np.diff(lines))) > 35.0


@pytest.mark.parametrize(
    ("band", "exponent"), [("third", 1.0 / 6.0), ("octave", 0.5)]
)
def test_covering_bands_are_the_iec_61260_base_ten_bands(
    band: str, exponent: float
) -> None:
    """The lines a band averages over are set by the IEC 61260-1 band edges.

    IEC 61260-1 base-ten bands use ``G = 10^(3/10)``, so a one-``b``-th octave
    band about ``fc`` runs from ``fc G^(-1/(2b))`` to ``fc G^(1/(2b))``: a
    factor ``10^0,05`` for one-third octaves and ``10^0,15`` for octaves. The
    base-two convention would give ``2^(1/6)`` and ``sqrt(2)``, 0,04 % and
    0,12 % wider, which changes which Fourier lines a band contains.

    The edges are pinned through their observable consequence: the band value
    must equal the mean-square ratio over exactly the lines the IEC definition
    selects, and nothing else.
    """
    plate_stiffness, impedance = _plate("concrete", 0.14)
    thickness = 0.005
    covering = covering_contact_stiffness(2.8e8 * thickness, thickness)
    ratio = (10.0 ** (3.0 / 10.0)) ** exponent
    centres = np.array([200.0, 1000.0, 4000.0])
    result = covering_improvement(
        centres, covering, plate_stiffness, impedance, band=band,  # type: ignore[arg-type]
    )
    for i, fc in enumerate(centres):
        chosen = (result.lines >= fc / ratio) & (result.lines <= fc * ratio)
        expected = 10.0 * np.log10(
            np.sum(result.bare.peak_force[chosen] ** 2)
            / np.sum(result.covered.peak_force[chosen] ** 2)
        )
        assert result.improvement[i] == pytest.approx(expected, abs=1e-9)
    # The octave band at 1 kHz holds exactly 71 lines of the 10 Hz comb,
    # 710 Hz to 1410 Hz. Edges 1,5 % wider reach 700 Hz and 1430 Hz and hold
    # 74, which is what makes the constant observable at all.
    if band == "octave":
        inside = (result.lines >= 1000.0 / ratio) & (result.lines <= 1000.0 * ratio)
        assert int(inside.sum()) == 71
        assert result.lines[inside][0] == 710.0
        assert result.lines[inside][-1] == 1410.0
        wide = 1.015 * ratio
        wider = (result.lines >= 1000.0 / wide) & (result.lines <= 1000.0 * wide)
        assert int(wider.sum()) == 74


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
    """H Fig. 4.73, printed p. 524 / pdf p. 551 (Eq. 4.125 itself is on 523).

    The construction and all three frequencies are printed *inside the plot
    panel*, not in the caption: "Walking surface: 18 mm plywood", "Resilient
    layer: s' = 7,25 MN/m3 (25 mm reconstituted foam)", "Base floor: 140 mm
    concrete slab", "Single floating floor fms = 118 Hz", "Double floating
    floor fmsms = 74 Hz and 195 Hz". The double floor is the same floor
    doubled, so both layers are identical.

    The mass per unit area comes from the Table A2 plywood density
    (710 kg/m3), independently of the printed frequencies. The three printed
    numbers are mutually consistent only to about 1,6 %, so the tolerance is
    2 % rather than tighter: ``sqrt(s'/rho_s)/(2 pi)`` gives 119,9 Hz against
    the printed 118 Hz, and the printed 74 and 195 imply ``fms = 120,1 Hz``
    through the identity ``fmsms,lower * fmsms,upper = fms^2`` that holds for
    two identical floors. The 118 is the odd one out. Values read off a plot
    panel carry that much slack; the closed-form identity below is the exact
    anchor.
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


def test_double_floating_floor_asymmetric_matches_the_two_degree_eigenproblem() -> None:
    """H Eq. (4.125) against the 2-DOF eigenvalue problem it solves.

    Every other test of this function uses two identical floors, where the
    middle term of ``X`` is unobservable: ``s'2/rho_s1`` and ``s'1/rho_s2``
    are then the same number, so transposing it survives. This one uses
    different masses *and* different stiffnesses.

    The oracle is built from the equations of motion rather than from
    Eq. (4.125). With subsystem 1 the lower floating floor (H Fig. 4.72,
    printed p. 523, whose layers are labelled ``rho_s2, s'2, rho_s1, s'1``
    from the top), the displacements obey
    ``rho_s1 x1'' = -s'1 x1 + s'2 (x2 - x1)`` and
    ``rho_s2 x2'' = -s'2 (x2 - x1)``, i.e. ``M = diag(rho_s1, rho_s2)`` and
    ``K = [[s'1 + s'2, -s'2], [-s'2, s'2]]``. The resonances are
    ``sqrt(eig(M^-1 K))/(2 pi)``, computed here by :func:`numpy.linalg.eigvals`.
    """
    # Lower floor: a 45 mm sand-cement screed (110 kg/m2) on a stiff 30 MN/m3
    # layer. Upper floor: 22 mm chipboard (17 kg/m2) on a soft 4 MN/m3 layer,
    # the resilient layer of H Fig. 4.70.
    s1, m1, s2, m2 = 30.0e6, 110.0, 4.0e6, 17.0
    stiffness_matrix = np.array([[s1 + s2, -s2], [-s2, s2]])
    mass_matrix = np.diag([m1, m2])
    omega_sq = np.sort(np.linalg.eigvals(np.linalg.inv(mass_matrix) @ stiffness_matrix))
    expected = np.sqrt(np.real(omega_sq)) / (2.0 * np.pi)

    lower, upper = double_floating_floor_resonances(s1, m1, s2, m2)
    assert lower == pytest.approx(expected[0], rel=1e-10)
    assert upper == pytest.approx(expected[1], rel=1e-10)
    # The two orderings of the middle term are far apart on this specimen, so
    # a transposition cannot hide inside the tolerance.
    transposed_x = s1 / m1 + s1 / m2 + s2 / m2
    scale = 1.0 / (2.0**1.5 * np.pi)
    root = np.sqrt(transposed_x**2 - 4.0 * s1 * s2 / (m1 * m2))
    assert abs(scale * np.sqrt(transposed_x - root) - lower) > 30.0
    assert abs(scale * np.sqrt(transposed_x + root) - upper) > 100.0


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


@pytest.mark.parametrize(
    ("model", "expected_floor", "kwargs"),
    [
        ("en12354", "screed", {}),
        ("cremer", "asphalt", {}),
        ("cremer_hammer", "asphalt", {"limiting_frequency": 521.0}),
    ],
)
def test_spectrum_picks_the_weighted_fit_of_its_own_construction(
    model: str, expected_floor: str, kwargs: dict[str, float]
) -> None:
    """The 30 lg law is the screed branch (C.1/C.4); the two 40 lg laws are the
    asphalt and dry floating floors (C.3/C.5), so ``delta_lw`` must follow.

    Every other test supplies either no mass and stiffness at all, or supplies
    them only with the default model, so swapping the two fits is invisible.
    Here the mapping is checked on a specimen where the two formulae are
    9,5 dB apart, well outside any plausible tolerance.
    """
    mass, stiffness = 73.5, 8.0e6
    result = floating_floor_improvement_spectrum(
        np.array([100.0, 500.0]), resonance_frequency=52.8, model=model,
        mass_per_area=mass, dynamic_stiffness=stiffness, **kwargs,
    )
    expected = weighted_floating_floor_improvement(
        mass, stiffness, floor=expected_floor  # type: ignore[arg-type]
    )
    other = weighted_floating_floor_improvement(
        mass, stiffness,
        floor="asphalt" if expected_floor == "screed" else "screed",  # type: ignore[arg-type]
    )
    assert abs(expected - other) > 5.0
    assert result.delta_lw == pytest.approx(expected, abs=1e-9)


def test_mount_model_is_zero_at_and_below_its_own_resonance() -> None:
    """The Ver model is the *dominant* term of V Eq. (8.45), valid above ``fo``.

    Below ``fo = sqrt(N s/m1)/(2 pi)`` the dominant term alone falls without
    bound and goes negative, which would say a resilient mounting makes the
    impact insulation worse the softer it is. H printed p. 521 handles the
    same regime by convention: "it is simplest to assume that DeltaL = 0 dB in
    all frequency bands below the band containing fms".
    """
    impedance, mass_per_area = 3.8e5, 115.0
    stiffness, density = 2.0e6, 4.0
    f0 = np.sqrt(density * stiffness / mass_per_area) / (2.0 * np.pi)
    assert f0 == pytest.approx(41.98, abs=0.05)
    values = resilient_mount_improvement(
        np.array([1.0, 10.0, f0, f0 * 1.01, 100.0]), impedance=impedance,
        mass_per_area=mass_per_area, loss_factor=0.02,
        mount_stiffness=stiffness, mount_density=density,
    )
    assert np.all(values[:3] == 0.0)
    assert values[3] > 0.0
    assert values[4] > values[3]


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
    with pytest.warns(UserWarning, match="outside the 20 dB"):
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


@pytest.mark.parametrize(
    ("lower_band", "upper_band", "lower_value", "upper_value"),
    [
        # ISO 266 band edges are 10^((n +- 0,5)/10) about the exact midband
        # frequencies 10^(n/10), NOT the geometric mean of the rounded nominal
        # labels. The two disagree by up to 1,4 % and straddle real inputs.
        (250.0, 315.0, -3.0, -5.0),
        (315.0, 400.0, -5.0, -7.0),
        (400.0, 500.0, -7.0, -9.0),
        (500.0, 630.0, -9.0, -10.0),
    ],
)
def test_table_d1_reads_the_band_containing_fo_not_the_nearest_label(
    lower_band: float, upper_band: float, lower_value: float, upper_value: float
) -> None:
    """ISO 12354-1:2017 Clause D.2.2, printed p. 38: ``fo`` is "rounded to the
    centre frequency of the one-third-octave band in which fo falls".

    Band *containment*, not proximity to a label. The edge between two
    one-third-octave bands ``n`` and ``n + 1`` is ``10^((n + 0,5)/10)``, which
    for 250|315 is 281,838 Hz while the geometric mean of the labels is
    280,624 Hz: a resonance at 281 Hz belongs to the 250 Hz band and reads
    -3 dB, and a nearest-label rounding would answer -5 dB.
    """
    index = int(np.round(10.0 * np.log10(lower_band)))
    edge = 10.0 ** ((index + 0.5) / 10.0)
    assert weighted_lining_improvement(edge * 0.999, 45.0) == lower_value
    assert weighted_lining_improvement(edge * 1.001, 45.0) == upper_value
    # Every nominal centre must read its own row.
    assert weighted_lining_improvement(lower_band, 45.0) == lower_value
    assert weighted_lining_improvement(upper_band, 45.0) == upper_value


def test_table_d1_low_branch_evaluates_at_the_rounded_nominal_not_raw_fo() -> None:
    """Clause D.2.2 rounds ``fo`` *before* Table D.1 is read, and the
    ``74,4 - 20 lg(fo) - Rw/2`` row is part of Table D.1.

    So the formula is evaluated at the band's nominal centre, not at the raw
    resonance frequency: two linings whose resonances fall in the same band
    must get the same rating. Using raw ``fo`` instead is what lets a
    band-edge error through unnoticed, since the answer then varies smoothly
    and no boundary test can see it.
    """
    # 100 Hz band: 89,1 Hz to 112,2 Hz (ISO 266 edges).
    expected = 74.4 - 20.0 * np.log10(100.0) - 22.5
    for f0 in (89.2, 95.0, 100.0, 108.0, 112.0):
        assert weighted_lining_improvement(f0, 45.0) == pytest.approx(
            expected, abs=1e-9
        )
    # Raw fo would spread these over 2 dB.
    raw = [74.4 - 20.0 * np.log10(f) - 22.5 for f in (89.2, 112.0)]
    assert raw[0] - raw[1] == pytest.approx(1.98, abs=0.02)


def test_table_d1_band_edges_beat_the_nominal_geometric_mean() -> None:
    """The two rounding conventions disagree on real inputs, by up to 8,8 dB.

    The worst case is the 160|200 boundary, where Table D.1 changes from the
    ``74,4 - 20 lg(fo) - Rw/2`` branch to the fixed -1 dB row. The ISO 266
    edge is ``10^(2,25) = 177,83 Hz``; the geometric mean of the nominal
    labels 160 and 200 is 178,89 Hz. A resonance at 178,5 Hz belongs to the
    200 Hz band.
    """
    assert 10.0**2.25 < 178.5 < np.sqrt(160.0 * 200.0)
    assert weighted_lining_improvement(178.5, 45.0) == -1.0
    assert weighted_lining_improvement(177.0, 45.0) == pytest.approx(
        74.4 - 20.0 * np.log10(160.0) - 22.5, abs=1e-9
    )
    swing = (74.4 - 20.0 * np.log10(160.0) - 22.5) - (-1.0)
    assert swing == pytest.approx(8.8, abs=0.05)


def test_table_d1_63_to_80_boundary() -> None:
    """The same distinction inside the low-frequency branch, worth 2,1 dB.

    The 63 Hz band ends at ``10^1,85 = 70,79 Hz``; the geometric mean of the
    labels 63 and 80 is 70,99 Hz. A lining resonance of 70,81 Hz, which the
    guide's worked stud lining produces, therefore reads off the 80 Hz row.
    """
    assert 10.0**1.85 < 70.8115 < np.sqrt(63.0 * 80.0)
    assert weighted_lining_improvement(70.8115, 45.0) == pytest.approx(
        74.4 - 20.0 * np.log10(80.0) - 22.5, abs=1e-9
    )
    difference = 20.0 * np.log10(80.0 / 63.0)
    assert difference == pytest.approx(2.08, abs=0.01)


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


@pytest.mark.parametrize("rating", [-50.0, 0.0, 19.9, 60.1, 80.0])
def test_table_d1_warns_outside_the_stated_rw_box(rating: float) -> None:
    """ISO 12354-1:2017 Clause D.2.2, printed p. 38, states Table D.1 "for
    basic structural elements with a weighted sound reduction index in the
    range of 20 dB <= Rw <= 60 dB".

    The low branch is linear in ``Rw`` and returns nonsense outside it: an
    ``Rw`` of -50 dB yields +65,4 dB of improvement from a lining. The value
    is still returned, because the formula is what it is, but the caller is
    told the standard does not cover it.
    """
    with pytest.warns(UserWarning, match="outside the 20 dB"):
        weighted_lining_improvement(50.0, rating)


@pytest.mark.parametrize("rating", [20.0, 45.0, 60.0])
def test_table_d1_is_silent_inside_the_stated_rw_box(rating: float) -> None:
    """No warning at either endpoint, both of which Clause D.2.2 includes."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        weighted_lining_improvement(50.0, rating)
        weighted_lining_improvement(500.0, rating)


@pytest.mark.parametrize("resonance", [1.0, 29.0, 5001.0, 20000.0])
def test_lining_formulae_warn_outside_the_annex_d_range(resonance: float) -> None:
    """Formulae (D.3), (D.4) and (D.7) are monotonic in ``lg(fo)`` and
    unbounded below the range Annex D puts lining resonances on.

    At ``fo = 1 Hz`` the mineral-wool fit returns +82,5 dB, which is more
    improvement than any lining has ever given. Table D.1 is the only place
    the annex bounds ``fo``, at 30 Hz to 5 000 Hz, so that is the envelope
    used.
    """
    with pytest.warns(UserWarning, match="outside the 30 Hz"):
        lining_improvement(resonance)
    with pytest.warns(UserWarning, match="outside the 30 Hz"):
        lining_improvement(resonance, system="studs")


@pytest.mark.parametrize("resonance", [30.0, 100.0, 5000.0])
def test_lining_formulae_are_silent_inside_the_annex_d_range(
    resonance: float,
) -> None:
    """Both endpoints included, so plotting the whole range stays quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for system in ("mineral_wool", "foam", "studs"):
            lining_improvement(resonance, system=system)  # type: ignore[arg-type]


def test_stacked_d5_and_d6_corrections_can_fall_below_the_reference_floor() -> None:
    """Annex D places the ``>= -4 dB`` floor inside Formulae (D.3)/(D.4) and
    does not restate it after (D.5) or (D.6), so the library does not either.

    This pins the literal reading, and the size of the excursion, so that
    changing it later is a deliberate act rather than a silent one.
    """
    reference = lining_improvement(5000.0)
    assert reference.ratings == (-4.0, -4.0, -4.0)
    corrected = lining_improvement(5000.0, anchors=True, glued_area=100.0)
    assert corrected.delta_rw == pytest.approx(0.66 * -4.0 - 1.2 - 5.0 + 2.0, abs=1e-9)
    assert corrected.delta_rw < -4.0
    assert corrected.delta_rw == pytest.approx(-6.84, abs=0.01)


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


@pytest.mark.parametrize(
    ("system", "floor"),
    [("mineral_wool", -4.0), ("foam", -3.0), ("studs", -4.0)],
)
def test_every_annex_d_rating_has_its_own_printed_floor(
    system: str, floor: float
) -> None:
    """Formulae (D.3), (D.4) and (D.7) print the floor on **all three**
    ratings, not only on ``DeltaRw``.

    Each is ">= -4 dB" for mineral wool and for studs, ">= -3 dB" for the
    foams. Testing only ``delta_rw`` leaves the other two floors unpinned, and
    every fit is steep enough (20 to 42 dB per decade) that 5 000 Hz is well
    into the floored regime for all three.
    """
    result = lining_improvement(5000.0, system=system)  # type: ignore[arg-type]
    assert result.ratings == (floor, floor, floor)
    # Unfloored, all three would be far below it.
    unfloored = lining_improvement(100.0, system=system)  # type: ignore[arg-type]
    assert all(value > floor + 3.0 for value in unfloored.ratings)


def test_third_octave_centres_are_the_iso_266_nominal_series() -> None:
    """Every entry of the rounding table is the ISO 266 nominal label of its
    band, i.e. within about 1 % of the exact midband frequency ``10^(n/10)``.

    Without this the entries below 31,5 Hz and above 5 000 Hz are unreachable
    through Table D.1 and could hold any value at all.
    """
    for offset, nominal in enumerate(_THIRD_OCTAVE_CENTRES):
        exact = 10.0 ** ((offset + _THIRD_OCTAVE_FIRST_INDEX) / 10.0)
        assert abs(nominal / exact - 1.0) < 0.012, (offset, nominal, exact)
    assert _THIRD_OCTAVE_CENTRES[0] == 12.5
    assert _THIRD_OCTAVE_CENTRES[-1] == 10000.0
    assert len(set(_THIRD_OCTAVE_CENTRES)) == len(_THIRD_OCTAVE_CENTRES)


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
