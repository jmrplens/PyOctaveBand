#  Copyright (c) 2026. Jose M. Requena-Plens
"""Fluid-solid coupling of the elastic FDTD solver against exact oracles.

The oracles are transcriptions from Brekhovskikh & Godin, *Acoustics of
Layered Media I: Plane and Quasi-Plane Waves* (Springer 1990) and are
independent of the implementation:

* the oblique plane-wave reflection coefficient of a fluid over an elastic
  half-space (Eqs. 4.2.22-4.2.26 and the limits 4.2.27-4.2.31), with the
  P-to-SV mode conversion active between the critical angles;
* the Scholte interface wave from the exact characteristic equation
  (Eq. 4.4.20 in the notation of Eq. 4.4.18), exposed as
  :func:`~phonometry.simulation.scholte_speed`, including the light-fluid
  asymptotics of Eqs. 4.4.21-4.4.22;
* the exact three-media transmission of an immersed elastic plate at normal
  incidence (Eqs. 2.4.10 and 2.4.14; no shear is excited at normal
  incidence, Eq. 4.2.27) with its half-wave thickness resonances
  ``f_n = n c_P / (2 h)`` (Eq. 2.4.19), cross-checked in its thin-plate
  band against
  :func:`~phonometry.building.panel_transmission.mass_law_transmission_loss`.

The Scholte time-of-flight case runs the fluid-solid benchmark
configuration of van Vossen, Robertsson & Chapman, "Finite-difference
modeling of wave propagation in a fluid-solid configuration", *Geophysics*
67(2), 618-624 (2002): water (1500 m/s, 1000 kg/m3) over a soft solid
(3500/2000/2500), where the interface wave travels at 0.957 c and is
confined within ~0.55 wavelengths of the contact, so it separates from the
water arrival over a few hundred metres. Over stiff beds (water over
steel) the Scholte deficit is only 0.027 % of c and the wave cannot be
separated by time of flight in a finite domain; those pairs are pinned
through the characteristic-equation roots instead. Air over a solid is the
extreme-contrast stress test: the interface wave is indistinguishable from
the grazing airborne wave (deficit ~1e-12 c), so only stability, the
reflection coefficient and the transmitted order of magnitude are asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from phonometry.building.panel_transmission import mass_law_transmission_loss
from phonometry.simulation import (
    AIR,
    ALUMINIUM,
    STEEL,
    WATER,
    ElasticFDTD2D,
    ExplosionSource,
    Material,
    scholte_speed,
)

#: The van Vossen et al. (2002) benchmark media (their METHOD section).
VV_FLUID = Material(c_p=1500.0, c_s=0.0, rho=1000.0)
VV_SOLID = Material(c_p=3500.0, c_s=2000.0, rho=2500.0)


def _reflection_coefficient(theta_deg: float, fluid: Material,
                            solid: Material) -> complex:
    """Plane-wave reflection of a fluid over an elastic half-space.

    Brekhovskikh & Godin (1990) Eqs. 4.2.22-4.2.24: Snell angles for the
    transmitted P and SV waves, the per-wave impedances
    ``Z = rho c / cos(theta)`` and

    ``V = (Zl cos^2(2 theta_t) + Zt sin^2(2 theta_t) - Z) /
    (Zl cos^2(2 theta_t) + Zt sin^2(2 theta_t) + Z)``.

    Past a critical angle the corresponding cosine is continued as
    ``i sqrt(sin^2 - 1)`` (numpy's principal square root of a negative
    real), the decaying evanescent branch, which yields the complex V of
    Eq. 4.2.29 between the critical angles and ``|V| = 1`` beyond the S
    critical angle (Eq. 4.2.31).
    """
    theta = np.radians(theta_deg)
    sin_l = np.complex128(solid.c_p / fluid.c_p * np.sin(theta))
    sin_t = np.complex128(solid.c_s / fluid.c_p * np.sin(theta))
    cos_l = np.sqrt(1.0 - sin_l**2)
    cos_t = np.sqrt(1.0 - sin_t**2)
    z = fluid.rho * fluid.c_p / np.cos(theta)
    z_l = solid.rho * solid.c_p / cos_l
    z_t = solid.rho * solid.c_s / cos_t
    cos_2t = 1.0 - 2.0 * sin_t**2
    sin_2t = 2.0 * sin_t * cos_t
    z_total = z_l * cos_2t**2 + z_t * sin_2t**2
    return complex((z_total - z) / (z_total + z))


def _plate_transmission(f: float, h: float, fluid: Material,
                        solid: Material) -> complex:
    """Three-media transmission of an immersed plate at normal incidence.

    B&G Eq. 2.4.14 with ``Z1 = Z3`` (same fluid on both sides) and
    ``phi = omega h / c_P`` (at normal incidence no shear is excited,
    Eq. 4.2.27, so the elastic plate is exactly the fluid layer of
    Section 2.4 with the plate's ``rho`` and ``c_P``)::

        W = 4 Z1 Z2 / ((Z1 + Z2)^2 e^{-i phi} - (Z1 - Z2)^2 e^{i phi})

    ``|W| = 1`` at the half-wave resonances ``phi = n pi`` (Eq. 2.4.19).
    """
    z1 = fluid.rho * fluid.c_p
    z2 = solid.rho * solid.c_p
    phi = 2.0 * np.pi * f * h / solid.c_p
    return complex(4.0 * z1 * z2 / ((z1 + z2) ** 2 * np.exp(-1j * phi)
                                    - (z1 - z2) ** 2 * np.exp(1j * phi)))


# --- The reference oracle itself: B&G limiting cases ------------------------


def test_reflection_coefficient_limiting_cases() -> None:
    # The reference formula must satisfy the closed-form limits of B&G
    # 4.2.27-4.2.31 before it is allowed to judge the solver. At normal
    # incidence the solid behaves as a liquid of (rho1, c_l1); at the angle
    # where theta_t = pi/4 only SV is excited and V = (Zt - Z)/(Zt + Z);
    # beyond the S critical angle the reflection is total.
    z1 = WATER.rho * WATER.c_p
    z2 = STEEL.rho * STEEL.c_p
    v0 = _reflection_coefficient(0.0, WATER, STEEL)
    assert v0.real == pytest.approx((z2 - z1) / (z2 + z1), rel=1e-12)
    assert v0.imag == 0.0

    # theta with theta_t = pi/4: sin(theta) = c / (c_t sqrt(2)).
    theta_45 = float(np.degrees(np.arcsin(
        WATER.c_p / (STEEL.c_s * np.sqrt(2.0)))))
    z_t = STEEL.rho * STEEL.c_s * np.sqrt(2.0)
    z = z1 / np.cos(np.radians(theta_45))
    v45 = _reflection_coefficient(theta_45, WATER, STEEL)
    assert abs(v45 - (z_t - z) / (z_t + z)) < 1e-9

    # Below the P critical angle V is real; past the S critical angle
    # |V| = 1 with a phase (total reflection, the Scholte wave's origin).
    assert _reflection_coefficient(10.0, WATER, STEEL).imag == 0.0
    for theta in (30.0, 45.0, 60.0):
        v = _reflection_coefficient(theta, WATER, STEEL)
        assert abs(v) == pytest.approx(1.0, abs=1e-12)
        assert abs(np.angle(v)) > 0.0


# --- Scholte characteristic roots (no FDTD) ---------------------------------


def test_scholte_speed_matches_characteristic_roots() -> None:
    # Roots of B&G Eq. 4.4.20 resolved independently (sign sweep + Brent
    # on the transcribed characteristic, float64): water/steel 1479.6013,
    # water/aluminium 1476.7649 and the van Vossen pair 1435.9734 m/s. The
    # root always lies below both the fluid speed and the solid shear
    # speed (a true interface wave, B&G Section 4.4.3).
    cases = [
        (WATER, STEEL, 1479.6013),
        (WATER, ALUMINIUM, 1476.7649),
        (VV_FLUID, VV_SOLID, 1435.9734),
    ]
    for fluid, solid, expected in cases:
        v = scholte_speed(fluid, solid)
        assert v == pytest.approx(expected, abs=0.01)
        assert v < fluid.c_p
        assert v < solid.c_s


def test_scholte_speed_light_fluid_asymptotics() -> None:
    # For a gas over a solid (m r >> 1) the root hugs the sound speed:
    # v ~ c (1 - 1 / (8 m^2 r^2 (1 - q)^2)) (B&G Eqs. 4.4.21-4.4.22), a
    # deficit of ~8e-13 c for air over steel. The exact root must
    # reproduce that deficit, which is why the air-solid interface wave is
    # indistinguishable from the grazing airborne wave and is not testable
    # by time of flight.
    for solid in (STEEL, ALUMINIUM):
        q = solid.c_s**2 / solid.c_p**2
        r = solid.c_s**2 / AIR.c_p**2
        m = solid.rho / AIR.rho
        deficit_asymptotic = AIR.c_p / (8.0 * m**2 * r**2 * (1.0 - q) ** 2)
        deficit = AIR.c_p - scholte_speed(AIR, solid)
        assert deficit == pytest.approx(deficit_asymptotic, rel=0.25)


def test_scholte_speed_rejects_wrong_material_roles() -> None:
    with pytest.raises(ValueError, match="fluid must have"):
        scholte_speed(STEEL, ALUMINIUM)
    with pytest.raises(ValueError, match="solid must have"):
        scholte_speed(WATER, (1500.0, 0.0, 1000.0))
    # Triples are accepted in both roles.
    v = scholte_speed((1500.0, 0.0, 1000.0), (3500.0, 2000.0, 2500.0))
    assert v == pytest.approx(1435.9734, abs=0.01)


# --- Oblique reflection with mode conversion (FDTD) -------------------------


def _oblique_water_trace(theta_deg: float, with_steel: bool,
                         ) -> tuple[np.ndarray, float]:
    """Probe pressure of a tilted carrier beam over water or water/steel.

    A Gaussian-envelope 25 kHz beam (transverse 1/e half-width 0.35 m,
    3.1 degrees angular spread) is written as a leapfrog-consistent
    initial condition travelling down-right at ``theta_deg`` from the
    normal. The reference run replaces the steel by more water at the
    same time step (matched ``cfl``), so the run difference isolates the
    reflected field at the probe exactly, side-wall effects included.
    """
    theta = np.radians(theta_deg)
    dx, f0 = 0.004, 25e3
    ny_w = 113                          # 0.452 m of water
    ny, nx = ny_w + 70, 600
    regions = []
    cfl = 0.6
    if with_steel:
        regions = [((slice(ny_w, None), slice(None)), STEEL)]
    else:
        cfl = 0.6 * WATER.c_p / STEEL.c_p
    sim = ElasticFDTD2D.from_regions(
        (ny, nx), dx, background=WATER, regions=regions,
        sponge_width=30, sponge_sides=("bottom",), cfl=cfl)
    y_interface = ny_w * dx
    height = 0.28                       # beam-centre launch height
    x_hit = 0.5 * nx * dx               # beam axis meets the interface here
    x0 = x_hit - height * np.tan(theta)
    y0 = y_interface - height
    k = 2.0 * np.pi * f0 / WATER.c_p
    w_env, sigma_t = 0.06, 0.35

    def packet(x: np.ndarray, y: np.ndarray, shift: float) -> np.ndarray:
        xi = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta) + shift
        eta = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
        return np.asarray(np.exp(-(xi / w_env) ** 2)
                          * np.exp(-(eta / sigma_t) ** 2) * np.sin(k * xi))

    x_c = (np.arange(nx) + 0.5) * dx
    y_c = (np.arange(ny) + 0.5) * dx
    x_g, y_g = np.meshgrid(x_c, y_c)
    p0 = packet(x_g, y_g, 0.0)
    sim.txx[:] = -p0
    sim.tyy[:] = -p0
    # Leapfrog-consistent particle velocity, half a time step back.
    shift = 0.5 * WATER.c_p * sim.dt
    x_gf, y_gf = np.meshgrid(np.arange(1, nx) * dx, y_c)
    sim.vx += (packet(x_gf, y_gf, shift) / (WATER.rho * WATER.c_p)
               * np.sin(theta))
    x_gv, y_gv = np.meshgrid(x_c, np.arange(1, ny) * dx)
    sim.vy += (packet(x_gv, y_gv, shift) / (WATER.rho * WATER.c_p)
               * np.cos(theta))

    # The probe sits on the beam axis' interface hit point, 0.12 m up:
    # the incident and reflected beam centres cross it at symmetric
    # transverse offsets, so the envelope factors cancel in the ratio.
    ix_p = round(x_hit / dx - 0.5)
    iy_p = round((y_interface - 0.12) / dx - 0.5)
    n_steps = round(0.44e-3 / sim.dt)
    trace = np.zeros(n_steps)
    for i in range(n_steps):
        sim.step()
        trace[i] = -0.5 * (sim.txx[iy_p, ix_p] + sim.tyy[iy_p, ix_p])
    return trace, sim.dt


@pytest.mark.parametrize("theta_deg", [10.0, 20.0])
def test_oblique_reflection_matches_brekhovskikh_godin(
    theta_deg: float,
) -> None:
    # |V| against B&G 4.2.24 at two non-critical angles of water/steel
    # (critical angles 14.5 and 27.5 degrees): 10 degrees is fully
    # propagating (|V| = 0.937018) and 20 degrees sits between the
    # critical angles, where the transmitted P is evanescent and the SV
    # conversion carries the transmitted energy (|V| = 0.918330; a
    # shear-free bottom of the same c_p would reflect totally there, so
    # this value is only reproduced with the mode conversion active).
    # Amplitudes are compared per frequency band on the run difference;
    # measured: +0.07 % at 10 degrees, -0.25 % at 20 degrees.
    reference, _ = _oblique_water_trace(theta_deg, with_steel=False)
    with_steel, _ = _oblique_water_trace(theta_deg, with_steel=True)
    reflected = with_steel - reference
    spectrum_i = np.fft.rfft(reference)
    spectrum_r = np.fft.rfft(reflected)
    band = np.abs(spectrum_i) >= 0.3 * np.abs(spectrum_i).max()
    weights = np.abs(spectrum_i[band])
    ratio = np.abs(spectrum_r[band]) / weights
    v_measured = float(np.sum(ratio * weights) / np.sum(weights))
    v_exact = abs(_reflection_coefficient(theta_deg, WATER, STEEL))
    assert v_measured == pytest.approx(v_exact, rel=0.02)


# --- Scholte wave by time of flight (van Vossen benchmark) ------------------


def test_scholte_wave_speed_van_vossen_configuration() -> None:
    # Water (1500/1000) over the van Vossen soft solid (3500/2000/2500):
    # an explosive 50 Hz Ricker 10 m above the contact excites the
    # interface wave, and two interface probes 300 m (10.4 Scholte
    # wavelengths) apart time its envelope peak. The solid shear
    # wavelength is resolved at 40 cells and the Scholte wavelength at 29
    # (the O(2,2) staggered scheme needs >= 15 per van Vossen; with 19
    # points the wave still arrives visibly delayed, -3.2 %). Oracle:
    # 1435.97 m/s from the exact characteristic; measured: -1.5 %.
    dx = 1.0
    ny, nx = 190, 550
    iy_interface = 90                   # solid from y = 90 m down
    sim = ElasticFDTD2D.from_regions(
        (ny, nx), dx, background=VV_FLUID,
        regions=[((slice(iy_interface, None), slice(None)), VV_SOLID)],
        sponge_width=20, sponge_sides=("top", "bottom"))
    f0, t0 = 50.0, 0.030

    def ricker(t: float) -> float:
        a = (np.pi * f0 * (t - t0)) ** 2
        return float((1.0 - 2.0 * a) * np.exp(-a))

    sim.add_source(ExplosionSource(ix=50, iy=80, waveform=ricker,
                                   amplitude=1e3))
    # Two fluid cells on the contact, plus one 30 m up into the water.
    probes = ((150, 89), (450, 89), (450, 59))
    n_steps = round(0.360 / sim.dt)
    record = np.zeros((3, n_steps))
    for i in range(n_steps):
        sim.step()
        for j, (ix, iy) in enumerate(probes):
            record[j, i] = -0.5 * (sim.txx[iy, ix] + sim.tyy[iy, ix])
    times = (np.arange(n_steps) + 1) * sim.dt
    envelope = np.abs(hilbert(record, axis=1))
    t1 = float(times[envelope[0].argmax()])
    t2 = float(times[envelope[1].argmax()])
    distance = (probes[1][0] - probes[0][0]) * dx
    v_exact = scholte_speed(VV_FLUID, VV_SOLID)
    assert distance / (t2 - t1) == pytest.approx(v_exact, rel=0.02)
    # The timed train really is the interface wave, not the water
    # arrival: it is evanescent into the fluid (1/e depth 0.55
    # wavelengths, ~16 m here), so 30 m above the contact the same
    # passage is a small fraction of the interface amplitude (measured:
    # 0.17, consistent with exp(-30/15.8) plus the direct-wave residue).
    window = (times > t2 - 0.01) & (times < t2 + 0.01)
    ratio = float(envelope[2][window].max() / envelope[1][window].max())
    assert ratio < 0.35


# --- Immersed steel plate: exact three-media transmission -------------------


def test_immersed_plate_transmission_and_thickness_resonance() -> None:
    # A 10 mm steel plate in water, normal incidence, one broadband run:
    # a 1.5 mm Gaussian plane pulse carries usable energy up to ~340 kHz.
    # The incident spectrum is time-gated at a probe above the plate
    # (compact pulse, exact gating) and the transmitted spectrum recorded
    # below it; TL(f) = 20 lg |I/T|. Oracles (B&G 2.4.14): TL = 18.06 dB
    # at 50 kHz (measured +0.01 dB), the mass law 5.77 dB at 10 kHz in
    # its thin-plate band (measured -0.04 dB,
    # mass_law_transmission_loss with water as the ambient fluid), and
    # total transmission at the first thickness resonance
    # f_1 = c_P / (2 h) = 295.0 kHz (measured 294.7 kHz, -0.09 %, with
    # TL 0.005 dB at the dip).
    dx = 0.5e-3
    h = 0.010
    ny, nx = round(0.75 / dx), 3
    iy0 = round(0.35 / dx)
    sim = ElasticFDTD2D.from_regions(
        (ny, nx), dx, background=WATER,
        regions=[((slice(iy0, iy0 + round(h / dx)), slice(None)), STEEL)])
    y = (np.arange(ny) + 0.5) * dx
    width, y0 = 1.5e-3, 0.25
    p0 = np.exp(-(((y - y0) / width) ** 2))
    sim.txx[:] = -p0[:, np.newaxis]
    sim.tyy[:] = -p0[:, np.newaxis]
    y_face = np.arange(1, ny) * dx + 0.5 * WATER.c_p * sim.dt
    v0 = np.exp(-(((y_face - y0) / width) ** 2)) / (WATER.rho * WATER.c_p)
    sim.vy += v0[:, np.newaxis]
    n_steps = round(0.45e-3 / sim.dt)
    iy_above = round(0.32 / dx)         # sees incident then reflected
    iy_below = round(0.41 / dx)         # sees the transmitted ring-down
    trace_above = np.zeros(n_steps)
    trace_below = np.zeros(n_steps)
    for i in range(n_steps):
        sim.step()
        trace_above[i] = -0.5 * (sim.txx[iy_above, 1] + sim.tyy[iy_above, 1])
        trace_below[i] = -0.5 * (sim.txx[iy_below, 1] + sim.tyy[iy_below, 1])
    times = (np.arange(n_steps) + 1) * sim.dt
    # Incident passes the upper probe at 47 us, its plate reflection at
    # 88 us: both are ~10 us compact, so a 65 us gate separates them
    # exactly. The lower probe stays echo-free for the whole record.
    incident = np.where(times < 65e-6, trace_above, 0.0)
    n_fft = 1 << 18
    spectrum_i = np.abs(np.fft.rfft(incident, n_fft))
    spectrum_t = np.abs(np.fft.rfft(trace_below, n_fft))
    freqs = np.fft.rfftfreq(n_fft, sim.dt)
    tl = 20.0 * np.log10(spectrum_i / spectrum_t)

    tl_50k = float(tl[np.argmin(np.abs(freqs - 50e3))])
    w_50k = abs(_plate_transmission(50e3, h, WATER, STEEL))
    assert tl_50k == pytest.approx(-20.0 * np.log10(w_50k), abs=1.0)

    tl_10k = float(tl[np.argmin(np.abs(freqs - 10e3))])
    tl_mass_law = float(mass_law_transmission_loss(
        10e3, STEEL.rho * h, incidence="normal",
        speed_of_sound=WATER.c_p, air_density=WATER.rho))
    assert tl_10k == pytest.approx(tl_mass_law, abs=1.0)

    band = (freqs >= 250e3) & (freqs <= 340e3)
    f_dip = float(freqs[band][np.argmin(tl[band])])
    f_resonance = STEEL.c_p / (2.0 * h)
    assert f_resonance == 295e3
    assert f_dip == pytest.approx(f_resonance, rel=0.03)
    assert float(np.min(tl[band])) == pytest.approx(0.0, abs=1.0)


# --- Air over steel: the extreme-contrast stress test -----------------------


def test_air_solid_interface_is_stable_and_reflects_totally() -> None:
    # Air over steel is the hardest contrast the averaging has to survive:
    # density 6542:1, impedance ~1.1e5:1, moduli ~2e6:1. What is really
    # under test is the harmonic-mu / arithmetic-rho effective medium: the
    # run must stay finite over 10 000 steps and the reflected pulse must
    # keep its amplitude (|V| = 0.999982) within 0.5 % (measured:
    # +0.0005 %). The transmitted stress (amplitude ~2 incident, energy
    # tau_I = 3.6e-5) sits near the scheme's numerical floor, so only its
    # order of magnitude is checked, never its waveform; the air-solid
    # Scholte wave is unobservable (speed deficit ~1e-12 c, see the
    # asymptotics test) and is deliberately not asserted.
    dx = 5e-3
    ny, nx = 1440, 3
    iy_interface = 240                  # steel from y = 1.2 m down
    sim = ElasticFDTD2D.from_regions(
        (ny, nx), dx, background=AIR,
        regions=[((slice(iy_interface, None), slice(None)), STEEL)])
    y = (np.arange(ny) + 0.5) * dx
    p0 = np.exp(-(((y - 0.7) / 0.15) ** 2))
    sim.txx[:] = -p0[:, np.newaxis]
    sim.tyy[:] = -p0[:, np.newaxis]
    y_face = np.arange(1, ny) * dx + 0.5 * AIR.c_p * sim.dt
    v0 = np.exp(-(((y_face - 0.7) / 0.15) ** 2)) / (AIR.rho * AIR.c_p)
    sim.vy += v0[:, np.newaxis]
    n_steps = 10_000
    trace_air = np.zeros(n_steps)
    trace_solid = np.zeros(n_steps)
    for i in range(n_steps):
        sim.step()
        trace_air[i] = -0.5 * (sim.txx[190, 1] + sim.tyy[190, 1])
        trace_solid[i] = -sim.tyy[340, 1]
    assert np.all(np.isfinite(sim.txx))
    assert np.all(np.isfinite(sim.txy))
    assert np.all(np.isfinite(sim.vx)) and np.all(np.isfinite(sim.vy))
    times = (np.arange(n_steps) + 1) * sim.dt
    z1, z2 = AIR.rho * AIR.c_p, STEEL.rho * STEEL.c_p
    incident = float(trace_air[times < 1.4e-3].max())
    window = trace_air[(times > 1.9e-3) & (times < 2.6e-3)]
    reflected = float(window[np.abs(window).argmax()])
    assert reflected / incident == pytest.approx((z2 - z1) / (z2 + z1),
                                                 rel=0.005)
    solid_window = trace_solid[times < 3.3e-3]
    transmitted = float(solid_window[np.abs(solid_window).argmax()])
    expected = 2.0 * z2 / (z1 + z2) * incident
    assert expected / 2.0 < transmitted < expected * 2.0


# --- from_regions and Material ergonomics -----------------------------------


def test_from_regions_matches_manual_maps() -> None:
    # from_regions is sugar over the map constructor: same maps, same dt,
    # bit-identical stepping.
    ny, nx = 40, 60
    c_p = np.full((ny, nx), WATER.c_p)
    c_s = np.zeros((ny, nx))
    rho = np.full((ny, nx), WATER.rho)
    c_p[20:], c_s[20:], rho[20:] = STEEL.c_p, STEEL.c_s, STEEL.rho
    manual = ElasticFDTD2D(c_p, c_s, 0.01, rho=rho)
    sugar = ElasticFDTD2D.from_regions(
        (ny, nx), 0.01, background=WATER,
        regions=[((slice(20, None), slice(None)), STEEL)])
    np.testing.assert_array_equal(sugar.c_p, manual.c_p)
    np.testing.assert_array_equal(sugar.c_s, manual.c_s)
    np.testing.assert_array_equal(sugar.rho, manual.rho)
    assert sugar.dt == manual.dt
    for sim in (manual, sugar):
        sim.add_source(ExplosionSource(
            ix=30, iy=10,
            waveform=lambda t: float(np.exp(-((t - 5e-5) / 1e-5) ** 2))))
        sim.run(200)
    np.testing.assert_array_equal(sugar.txx, manual.txx)
    np.testing.assert_array_equal(sugar.vy, manual.vy)
    assert float(np.abs(sugar.txx).max()) > 0.0


def test_from_regions_paints_in_order_and_accepts_masks() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:6] = True
    sim = ElasticFDTD2D.from_regions(
        (10, 12), 0.01, background=(343.0, 0.0, 1.2),
        regions=[((slice(0, 6), slice(None)), ALUMINIUM), (mask, STEEL)])
    assert sim.c_p[0, 0] == ALUMINIUM.c_p          # first region
    assert sim.c_p[3, 4] == STEEL.c_p              # later region wins
    assert sim.rho[3, 4] == STEEL.rho
    assert sim.c_p[8, 0] == 343.0                  # background triple


def test_from_regions_and_material_validation() -> None:
    with pytest.raises(ValueError, match="mask must be a boolean"):
        ElasticFDTD2D.from_regions(
            (10, 10), 0.01, background=WATER,
            regions=[(np.zeros((3, 3), dtype=bool), STEEL)])
    with pytest.raises(ValueError, match="background must be a Material"):
        ElasticFDTD2D.from_regions((10, 10), 0.01, background="water")
    with pytest.raises(ValueError, match=r"regions\[0\] must be a Material"):
        ElasticFDTD2D.from_regions(
            (10, 10), 0.01, background=WATER,
            regions=[(np.s_[0:2, :], (1480.0, 0.0))])
    with pytest.raises(ValueError, match="at least 2 x 2"):
        ElasticFDTD2D.from_regions((1, 10), 0.01, background=WATER)
    with pytest.raises(ValueError, match="must be an integer"):
        ElasticFDTD2D.from_regions((10.5, 10), 0.01,  # type: ignore[arg-type]
                                   background=WATER)
    with pytest.raises(ValueError, match="c_s must be non-negative"):
        Material(c_p=1480.0, c_s=-1.0, rho=1000.0)
    with pytest.raises(ValueError, match=r"c_p\*\*2 must be at least"):
        Material(c_p=3000.0, c_s=2500.0, rho=1000.0)
    with pytest.raises(ValueError, match="rho must be positive"):
        Material(c_p=1480.0, c_s=0.0, rho=0.0)
    assert WATER.is_fluid and AIR.is_fluid
    assert not STEEL.is_fluid and not ALUMINIUM.is_fluid
