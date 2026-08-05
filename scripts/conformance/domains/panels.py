#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Panel & aperture sound insulation (Bies / Hopkins / Cremer).

The transmission loss of a partition from first principles: the mass law and
its incidence variants, the coincidence and critical frequencies, damping and
the plateau method, double-leaf panels with their mass-air-mass resonance,
composite partitions and the leaks and apertures that dominate them.

These are the closed forms the EN 12354 predictions of Domain 7 consume as
element data, checked here against the textbook expressions and their limits.
"""

from __future__ import annotations

import math

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register

_PANEL = "Panel & aperture sound insulation (Bies / Hopkins / Cremer)"


@register(_PANEL, "Bies 5e Eq. 7.40 (mass law)", "6 dB per octave (500 -> 1000 Hz)")
def _chk_mass_law_octave_slope() -> Outcome:
    lo = float(ph.mass_law_transmission_loss(500.0, 20.0, incidence="normal"))
    hi = float(ph.mass_law_transmission_loss(1000.0, 20.0, incidence="normal"))
    return numeric(6.0206, hi - lo, 0.01, unit="dB")


@register(_PANEL, "Bies 5e Eq. 7.40 (mass law)", "6 dB per doubling of mass")
def _chk_mass_law_mass_slope() -> Outcome:
    lo = float(ph.mass_law_transmission_loss(500.0, 20.0, incidence="normal"))
    hi = float(ph.mass_law_transmission_loss(500.0, 40.0, incidence="normal"))
    return numeric(6.0206, hi - lo, 0.01, unit="dB")


@register(_PANEL, "Bies 5e Eq. 7.42 (field incidence)", "One-third-octave correction 5.5 dB")
def _chk_field_incidence_correction() -> Outcome:
    n = float(ph.mass_law_transmission_loss(500.0, 20.0, incidence="normal"))
    fld = float(ph.mass_law_transmission_loss(500.0, 20.0, incidence="field"))
    return numeric(5.5, n - fld, 0.001, unit="dB")


@register(_PANEL, "Hopkins Eq. 2.201 / Bies Eq. 7.3", "Coincidence frequency, 6 mm glass")
def _chk_coincidence_frequency_glass() -> Outcome:
    bp = ph.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = ph.coincidence_frequency(2500.0 * 0.006, bp)
    return numeric(2079.0, fc, 0.03, rel=True, unit="Hz")


@register(_PANEL, "Cremer Table 5.1", "Thin-plate point impedance Z = 8 sqrt(B' m'')")
def _chk_plate_point_impedance() -> Outcome:
    z = ph.infinite_plate_impedance(1.0e4, 10.0)
    return numeric(8.0 * math.sqrt(1.0e4 * 10.0), z, 1e-6, unit="N.s/m")


@register(_PANEL, "Cremer Table 5.1", "Infinite-beam mobility phase -45 deg")
def _chk_beam_mobility_phase() -> Outcome:
    y = complex(ph.infinite_beam_mobility(137.0, 200.0, 5.0))
    return numeric(-45.0, math.degrees(math.atan2(y.imag, y.real)), 1e-6, unit="deg")


@register(_PANEL, "Hopkins Eq. 2.229 (Leppington/Maidanik)",
          "Radiation efficiency at f = 2 fc")
def _chk_radiation_above_coincidence() -> Outcome:
    res = ph.radiation_efficiency([4000.0], 1.5, 1.25, 2000.0)
    return numeric(1.0 / math.sqrt(1.0 - 0.5), float(res.radiation_efficiency[0]),
                   1e-9)


@register(_PANEL, "Bies Eq. 7.62 / Hopkins Eq. 4.73",
          "Mass-air-mass resonance f0, empty cavity")
def _chk_mass_spring_mass() -> Outcome:
    f0 = ph.mass_spring_mass_resonance(12.16, 12.16, 0.1)
    expected = 60.0 * math.sqrt((12.16 + 12.16) / (12.16 * 12.16 * 0.1))
    return numeric(expected, f0, 0.005, rel=True, unit="Hz")


@register(_PANEL, "Bies Eq. 7.64 (double wall)",
          "Below f0 = mass law of the combined mass")
def _chk_double_wall_low_frequency() -> Outcome:
    f0 = ph.mass_spring_mass_resonance(12.16, 12.16, 0.1)
    dw = float(ph.double_wall_transmission_loss([0.5 * f0], 12.16, 12.16, 0.1)
               .transmission_loss[0])
    ml = float(ph.mass_law_transmission_loss(0.5 * f0, 24.32))
    return numeric(ml, dw, 1e-6, unit="dB")


@register(_PANEL, "Hopkins Eq. 4.92 (composite)",
          "1 % open area caps R at 10 lg(S/Sa)")
def _chk_composite_open_area_limit() -> Outcome:
    r = float(ph.composite_transmission_loss([0.99, 0.01], [60.0, 0.0]))
    return numeric(10.0 * math.log10(1.0 / 0.01), r, 0.05, unit="dB")


@register(_PANEL, "Vigran Building Acoustics Eq. (3.109), printed p. 96",
          "Flat 1 mm steel plate 1 m x 1 m, f(1,1)")
def _chk_flat_plate_first_mode() -> Outcome:
    b = ph.plate_bending_stiffness(2.1e11, 1.0e-3, 0.3)
    f11 = ph.orthotropic_plate_resonance(
        1, 1, length_x=1.0, length_z=1.0, mass_per_area=7.8,
        bending_stiffness_x=b, bending_stiffness_z=b, bending_stiffness_xz=b,
    )
    return numeric(4.9, f11, 0.05, unit="Hz", places=2)


@register(_PANEL, "Vigran Eqs. (3.113)/(3.115), printed p. 96",
          "Corrugated 1 mm steel plate (H = 10 mm, L = 100 mm), f(2,2)")
def _chk_corrugated_plate_mode_22() -> Outcome:
    b_x, b_z, b_xz = ph.corrugated_plate_stiffness(
        1.0e-3, 0.010, 0.100, youngs_modulus=2.1e11, poisson_ratio=0.3
    )
    mass = 7.8 * ph.corrugated_plate_mass_factor(0.010, 0.100)
    f22 = ph.orthotropic_plate_resonance(
        2, 2, length_x=1.0, length_z=1.0, mass_per_area=mass,
        bending_stiffness_x=b_x, bending_stiffness_z=b_z,
        bending_stiffness_xz=b_xz,
    )
    return numeric(102.0, f22, 0.1, unit="Hz", places=2)


@register(_PANEL, "Bies 5e Eq. (7.59) / Vigran Eq. (6.112)",
          "Heckl coincidence-branch constant, dB (rho c = 414)")
def _chk_heckl_coincidence_constant() -> Outcome:
    f, mass, fc1, fc2 = 800.0, 7.5, 400.0, 4000.0
    tl = float(ph.orthotropic_transmission_loss(
        [f], mass, critical_frequency_lower=fc1,
        critical_frequency_upper=fc2, method="heckl",
        air_density=414.0 / 343.0,
    ).transmission_loss[0])
    constant = (
        tl - 20.0 * math.log10(f) - 10.0 * math.log10(mass)
        + 10.0 * math.log10(fc1)
        + 20.0 * math.log10(math.log(4.0 * f / fc1))
    )
    return numeric(-13.2, constant, 0.02, unit="dB", places=3)


@register(_PANEL, "Bies 5e Eq. (7.60) / Vigran Eq. (6.112)",
          "Heckl recovery-branch constant, dB (rho c = 414)")
def _chk_heckl_recovery_constant() -> Outcome:
    f, mass, fc1, fc2 = 12500.0, 7.5, 400.0, 4000.0
    tl = float(ph.orthotropic_transmission_loss(
        [f], mass, critical_frequency_lower=fc1,
        critical_frequency_upper=fc2, method="heckl",
        air_density=414.0 / 343.0,
    ).transmission_loss[0])
    constant = (
        tl - 20.0 * math.log10(f) - 10.0 * math.log10(mass)
        + 5.0 * math.log10(fc1) + 5.0 * math.log10(fc2)
    )
    return numeric(-23.0, constant, 0.2, unit="dB", places=3)


@register(_PANEL, "Vigran Eq. (6.111) / Bies Eq. (7.38)",
          "Orthotropic diffuse integral below fc1 vs its exact mass-law form")
def _chk_orthotropic_mass_law_integral() -> Outcome:
    mass, f, angle = 7.5, 50.0, 78.0
    tl = float(ph.orthotropic_transmission_loss(
        [f], mass, critical_frequency_lower=4.0e5,
        critical_frequency_upper=4.0e6, limiting_angle=angle,
    ).transmission_loss[0])
    q = 2.0 * math.pi * f * mass / (2.0 * 1.205 * 343.0)
    u = math.sin(math.radians(angle)) ** 2
    tau = math.log((1.0 + q**2) / (1.0 + q**2 * (1.0 - u))) / q**2
    return numeric(-10.0 * math.log10(tau), tl, 1e-6, unit="dB", places=6)


@register(_PANEL, "Hopkins Table A2, printed p. 608",
          "h.fc products of 25 building-material rows, worst deviation")
def _chk_hopkins_table_a2_products() -> Outcome:
    rho, nu, h = 2500.0, 0.24, 0.01
    worst = 0.0
    for c_l, product in ref.HOPKINS_TABLE_A2_H_FC:
        b = ph.plate_bending_stiffness(rho * c_l**2 * (1.0 - nu**2), h, nu)
        worst = max(worst, abs(h * ph.coincidence_frequency(rho * h, b) - product))
    return numeric(0.0, worst, 0.06, unit="m.Hz", places=4)


@register(_PANEL, "Hopkins Eq. 4.99/4.101 (Gomperts slit)",
          "Transmission maximum at first resonance")
def _chk_slit_resonance() -> Outcome:
    fr = float(ph.slit_resonance_frequencies(0.1, 0.005, orders=1)[0])
    f = np.linspace(fr - 300.0, fr + 300.0, 601)
    res = ph.slit_transmission_coefficient(f, 0.005, 0.1, field="normal")
    peak = float(f[int(np.argmax(res.transmission_coefficient))])
    return numeric(fr, peak, 15.0, unit="Hz")
