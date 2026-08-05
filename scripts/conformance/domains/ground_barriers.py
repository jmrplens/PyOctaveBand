#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Spherical ground effect & barriers (Attenborough / Salomons / Bies).

Sound over an impedance ground and around a screen: the Weyl-Van der Pol
spherical-wave reflection with its boundary-loss factor, and the barrier
diffraction models from the Kurze-Anderson engineering curve to the exact
thin-screen solution.

The oracles are the analytic limits these models are required to collapse to -
the two-ray interference over a rigid ground, the 6 dB of a half-plane at the
shadow boundary, the 10 dB per decade of Fresnel number - rather than
digitized curves.
"""

from __future__ import annotations

import warnings

import phonometry as ph

from ..registry import Outcome, numeric, register

_GROUND_BARRIERS = "Spherical ground & barriers (Attenborough / Salomons / Bies)"


@register(
    _GROUND_BARRIERS,
    "Attenborough 2e Eq. (2.40c) (spherical Q, hard-ground limit)",
    "abs(Q) as Z grows large (Rp -> 1 so (1 - Rp) -> 0 and Q -> 1)",
)
def _chk_hard_ground_q_unity() -> Outcome:
    q = ph.spherical_reflection_coefficient([500.0], 1e12, 1.0, 1.5, 50.0)
    return numeric(1.0, float(abs(q[0])), 1e-6, places=6)


@register(
    _GROUND_BARRIERS,
    "Salomons 2001 Sec. 3.4 (two-ray field over a rigid ground)",
    "dL enhancement at small path difference (constructive, +6 dB)",
)
def _chk_hard_ground_6db() -> Outcome:
    res = ph.ground_effect([31.5], 0.5, 0.5, 200.0, impedance=1e12)
    return numeric(6.0206, float(res.excess_attenuation[0]), 0.1, unit="dB")


@register(
    _GROUND_BARRIERS,
    "Salomons 2001 Eq. (D.59) (plane-wave Rp, grazing incidence)",
    "Re(Rp) at grazing (hs, hr -> 0, cos(theta) -> 0 so Rp -> -1)",
)
def _chk_grazing_rp_minus_one() -> Outcome:
    res = ph.ground_effect([500.0], 1e-4, 1e-4, 100.0, impedance=12.0 + 6.0j)
    return numeric(-1.0, float(res.plane_reflection_coefficient[0].real), 1e-3)


@register(
    _GROUND_BARRIERS,
    "Salomons 2001 Fig. D.3 (grassland ground dip, sigma = 200 kPa s/m2)",
    "Minimum dL for hs = hr = 2 m, r = 100 m (dip near 395 Hz), dB",
)
def _chk_salomons_fig_d3_dip() -> Outcome:
    freqs = [float(f) for f in range(380, 411)]
    with warnings.catch_warnings():
        # The bands sit below the Delany-Bazley published fit range for this
        # sigma; the model extrapolates there by design (Salomons Sec. 3.1).
        warnings.simplefilter("ignore")
        res = ph.ground_effect(freqs, 2.0, 2.0, 100.0, flow_resistivity=2e5)
    return numeric(-12.7, float(res.excess_attenuation.min()), 0.3, unit="dB",
                   places=2)


@register(
    _GROUND_BARRIERS,
    "Bies 5e Eq. (5.138) (Kurze-Anderson, N -> 0)",
    "Barrier attenuation at the shadow boundary N = 0",
)
def _chk_kurze_anderson_zero() -> Outcome:
    return numeric(5.0, float(ph.kurze_anderson_attenuation(0.0)), 1e-9, unit="dB")


@register(
    _GROUND_BARRIERS,
    "Bies 5e Eq. (5.138) (Kurze-Anderson, large-N slope)",
    "Delta(N=10) - Delta(N=1) vs the 10 lg(10) = 10 dB decade growth",
)
def _chk_kurze_anderson_slope() -> Outcome:
    d1 = float(ph.kurze_anderson_attenuation(1.0))
    d10 = float(ph.kurze_anderson_attenuation(10.0))
    return numeric(10.0, d10 - d1, 0.5, unit="dB")


@register(
    _GROUND_BARRIERS,
    "Attenborough 2e Eqs. (9.19)-(9.20) (rigid half-plane, shadow boundary)",
    "Exact thin-screen insertion loss at grazing (field halved, 6 dB)",
)
def _chk_exact_screen_shadow_boundary() -> Outcome:
    il = ph.barrier_insertion_loss([500.0], 1.0, 50.0, 1.0 + 1e-3, 100.0, 1.0,
                                   method="exact")
    return numeric(6.0206, float(il.insertion_loss[0]), 0.6, unit="dB")
