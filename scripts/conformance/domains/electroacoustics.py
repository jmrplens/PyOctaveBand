#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Electroacoustics: the baffled piston (Beranek & Mellow 2e).

The radiation impedance of a rigid circular piston in an infinite baffle - the
resistance and reactance functions and their small- and large-argument limits -
its far-field directivity and directivity index, and the on-axis pressure.

The expected values are the series and Bessel-function forms printed in
Beranek & Mellow, evaluated at the arguments the book tabulates.
"""

from __future__ import annotations

import math

import phonometry as ph

from ..registry import Outcome, numeric, register

_ELECTROACOUSTICS = "Electroacoustics"


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (13.117)",
    "Piston resistance R1(x) = 1 - 2 J1(x)/x at x = 2ka = 2",
)
def _chk_piston_resistance() -> Outcome:
    return numeric(
        0.423275,
        float(ph.electroacoustics.piston_resistance(2.0)),
        1e-5,
        places=6,
    )


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (13.118)",
    "Piston reactance X1(x) = 2 H1(x)/x at x = 2ka = 2",
)
def _chk_piston_reactance() -> Outcome:
    return numeric(
        0.646764,
        float(ph.electroacoustics.piston_reactance(2.0)),
        1e-5,
        places=6,
    )


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (13.117) (low-frequency limit)",
    "R1 -> (ka)^2/2 as ka -> 0 (x = 0.02, ka = 0.01)",
)
def _chk_piston_resistance_limit() -> Outcome:
    ka = 0.01
    return numeric(
        ka**2 / 2.0,
        float(ph.electroacoustics.piston_resistance(2.0 * ka)),
        1e-4,
        rel=True,
        places=8,
    )


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (4.151)",
    "Radiation mass M = 8 rho a^3 / 3  (a = 0.1 m, rho = 1.206)",
)
def _chk_piston_radiation_mass() -> Outcome:
    res = ph.electroacoustics.radiating_piston(0.1, [100.0], density=1.206)
    return numeric(8.0 * 1.206 * 0.1**3 / 3.0, res.radiation_mass, 1e-9,
                   unit="kg", places=8)


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (13.102), Table 14.1",
    "First directivity null at ka sin(theta) = 3.8317 (first zero of J1)",
)
def _chk_piston_directivity_null() -> Outcome:
    # ka sin(theta) = 3.8317 at ka = 3.8317, theta = pi/2.
    d = float(
        ph.electroacoustics.piston_directivity(
            3.8317059702075125, math.pi / 2.0
        )
    )
    return numeric(0.0, d, 1e-6, places=8)


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e §4.19 (half-space baffle)",
    "Directivity index DI -> 10 lg 2 = 3.01 dB as ka -> 0",
)
def _chk_piston_directivity_index() -> Outcome:
    res = ph.electroacoustics.radiating_piston(0.01, [1.0])
    return numeric(10.0 * math.log10(2.0), float(res.directivity_index[0]),
                   1e-3, unit="dB")


@register(
    _ELECTROACOUSTICS,
    "Long, Architectural Acoustics 2e, Eq. (18.21)",
    "Omnidirectional mic at Zs = -6 dB: L(H-M) <= L(H-L) - 4 dB",
)
def _chk_feedback_omnidirectional() -> Outcome:
    res = ph.electroacoustics.feedback_stability(-6.0, 76.0, 80.0)
    return numeric(80.0 - 4.0, res.maximum_level_at_microphone, 1e-9,
                   unit="dB", places=6)


@register(
    _ELECTROACOUSTICS,
    "Long, Architectural Acoustics 2e, Eq. (18.22)",
    "Cardioid mic (DM = -2 dB) at Zs = -6 dB: L(H-M) <= L(H-L) - 2 dB",
)
def _chk_feedback_cardioid() -> Outcome:
    res = ph.electroacoustics.feedback_stability(
        -6.0, 78.0, 80.0, microphone_directivity=-2.0
    )
    return numeric(80.0 - 2.0, res.maximum_level_at_microphone, 1e-9,
                   unit="dB", places=6)


@register(
    _ELECTROACOUSTICS,
    "Long, Architectural Acoustics 2e, Eq. (18.23)",
    "Number-of-open-microphones correction 10 lg Nm at Nm = 4",
)
def _chk_open_microphone_correction() -> Outcome:
    return numeric(
        10.0 * math.log10(4.0),
        ph.electroacoustics.open_microphone_correction(4),
        1e-12,
        unit="dB",
        places=6,
    )
