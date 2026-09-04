#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Atmospheric refraction (Salomons rays / GFPE).

Sound bent by a sound-speed gradient: the ray solution for a linear profile,
whose turning-point range and curvature radius are closed forms, and the
Green's-function parabolic equation, checked in the homogeneous limit where it
must reproduce the coherent two-ray field over a rigid ground.
"""

from __future__ import annotations

import cmath
import math

import phonometry as ph

from ..registry import Outcome, numeric, register

_ATM_REFRACTION = "Atmospheric refraction (Salomons rays / GFPE)"


@register(
    _ATM_REFRACTION,
    "Salomons Sec. 4.4 (ray turning height, linear profile)",
    "Turning height of a 10 deg ray vs Rc(1 - cos theta0) (circular arc), m",
)
def _chk_atm_ray_turning() -> Outcome:
    c0, gradient, angle = 343.0, 0.2, 10.0
    rc = c0 / (gradient * math.cos(math.radians(angle)))
    turn = rc * (1.0 - math.cos(math.radians(angle)))
    prof = ph.environment.linear_sound_speed_profile(
        gradient, ground_speed=c0, max_height=3000.0
    )
    res = ph.environment.atmospheric_ray_paths(
        prof,
        source_height=0.0,
        launch_angles_deg=[angle],
        max_range=600.0,
        n_steps=8000,
    )
    return numeric(turn, float(res.heights[0].max()), 0.1, unit="m", places=3)


@register(
    _ATM_REFRACTION,
    "Salomons Eq. (3.4) (GFPE vs spherical-wave ground effect, homogeneous)",
    "PE relative level at 500 m over grassland vs Weyl-Van der Pol, dB",
)
def _chk_atm_pe_ground_effect() -> Outcome:
    freq, zs, zr, z = 250.0, 1.0, 1.0, 11.0 + 8.0j
    flat = ph.environment.linear_sound_speed_profile(
        1e-12, ground_speed=343.0, max_height=200.0
    )
    pe = ph.environment.atmospheric_parabolic_equation(
        freq,
        flat,
        source_height=zs,
        impedance=z,
        max_range=520.0,
        max_height=40.0,
    )
    i = int(min(range(pe.ranges.size), key=lambda k: abs(pe.ranges[k] - 500.0)))
    got = float(pe.level_at_height(zr)[i])
    oracle = float(
        ph.environment.ground_effect(
            [freq],
            zs,
            zr,
            float(pe.ranges[i]),
            impedance=z,
            fluid=ph.materials.PUBLISHED_AIR,
        ).excess_attenuation[0]
    )
    return numeric(oracle, got, 0.5, unit="dB", places=3)


@register(
    _ATM_REFRACTION,
    "Salomons Eq. (3.4) (GFPE hard ground vs two-ray, homogeneous)",
    "PE relative level at 500 m over a rigid ground vs the coherent two-ray, dB",
)
def _chk_atm_pe_hard_ground() -> Outcome:
    freq, zs, zr = 500.0, 2.0, 2.0
    flat = ph.environment.linear_sound_speed_profile(
        1e-12, ground_speed=343.0, max_height=200.0
    )
    pe = ph.environment.atmospheric_parabolic_equation(
        freq,
        flat,
        source_height=zs,
        impedance=1e6 + 0j,
        max_range=520.0,
        max_height=150.0,
    )
    i = int(min(range(pe.ranges.size), key=lambda k: abs(pe.ranges[k] - 500.0)))
    r = float(pe.ranges[i])
    r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
    k = 2.0 * math.pi * freq / 343.0
    two_ray = 20.0 * math.log10(abs(1.0 + (r1 / r2) * cmath.exp(1j * k * (r2 - r1))))
    return numeric(two_ray, float(pe.level_at_height(zr)[i]), 0.6, unit="dB", places=3)
