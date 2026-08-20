#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Bending-wave plate-junction transmission (Cremer / Craik / Hopkins).

Vibrational power crossing a rigid junction of plates: the transmission
coefficients of a cross, T and corner junction as functions of the
characteristic-impedance ratio, and the vibration reduction index derived from
them.

The oracle is the reciprocity the coefficients must satisfy between the two
directions of a path, plus the equal-plate limit where the answer is known by
symmetry.
"""

from __future__ import annotations

import math

import phonometry as ph

from ..registry import Outcome, numeric, register

_JUNCTION = "Bending-wave plate-junction transmission (Cremer / Craik / Hopkins)"


@register(
    _JUNCTION,
    "Hopkins Eq. 5.12 (identical plates)",
    "X-junction corner tau12(0 deg) = 1/8",
)
def _chk_junction_x_corner_normal() -> Outcome:
    # chi = psi = 1 -> denominator (J2 psi + chi)**2 = 4, numerator 0.5.
    tau = float(ph.vibration.corner_transmission_coefficient(0.0, 1.0, 1.0, "X"))
    return numeric(0.125, tau, 1e-9)


@register(
    _JUNCTION,
    "Hopkins Eqs 5.12 + 5.6 (identical plates)",
    "X-junction corner angular average = 1/12",
)
def _chk_junction_x_average() -> Outcome:
    avg = ph.vibration.angular_average_transmission_coefficient(
        1.0, 1.0, "X", section="corner"
    )
    return numeric(1.0 / 12.0, avg, 1e-6)


@register(
    _JUNCTION,
    "Hopkins Eqs 5.12 + 5.6 (identical plates)",
    "L-junction corner angular average = 1/3",
)
def _chk_junction_l_average() -> Outcome:
    avg = ph.vibration.angular_average_transmission_coefficient(
        1.0, 1.0, "L", section="corner"
    )
    return numeric(1.0 / 3.0, avg, 1e-6)


@register(
    _JUNCTION,
    "Hopkins Eq. 5.14 (identical plates)",
    "In-line junction tau12(0 deg) = 1",
)
def _chk_junction_inline_identical() -> Outcome:
    return numeric(1.0, ph.vibration.inline_transmission_coefficient(1.0, 1.0), 1e-9)


@register(
    _JUNCTION,
    "Hopkins Eq. 5.7 (SEA consistency)",
    "X-junction reciprocity tau_bar_12 / tau_bar_21 = chi",
)
def _chk_junction_reciprocity() -> Outcome:
    chi = 1.5
    fwd = ph.vibration.angular_average_transmission_coefficient(
        chi, 0.8, "X", section="corner"
    )
    rev = ph.vibration.angular_average_transmission_coefficient(
        1.0 / chi, 1.0 / 0.8, "X", section="corner"
    )
    return numeric(chi, fwd / rev, 1e-6)


@register(
    _JUNCTION,
    "Hopkins Eq. 5.116 (identical plates, fc_j = f_ref)",
    "X-junction vibration reduction index = 10 lg(12)",
)
def _chk_junction_kij() -> Outcome:
    # fc_j = f_ref = 1000 Hz cancels the 5 lg(fc_j / f_ref) correction term.
    kij = float(ph.vibration.wave_vibration_reduction_index(1.0 / 12.0, 1000.0))
    return numeric(10.0 * math.log10(12.0), kij, 1e-6, unit="dB")
