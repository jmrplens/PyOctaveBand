#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Swept-sine distortion & phase utilities (Farina / Novak).

Exponential-sweep measurement: the Farina deconvolution that separates the
harmonic impulse responses from the linear one, the Novak synchronized sweep
and its higher-harmonic frequency responses, and the phase utilities the
analysis rests on - group delay, minimum-phase decomposition and the all-pass
factor a pure latency reduces to.

The excitation is synthesized through a polynomial nonlinearity with known
coefficients, so the harmonic amplitudes the deconvolution should recover are
closed forms.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal as sg

import phonometry as ph

from ..registry import Outcome, numeric, register

_SWEPT_SINE = "Swept-sine distortion & phase utilities (Farina / Novak)"

#: Chebyshev oracle for y = x + a2 x^2 + a3 x^3 driven by a unit sweep.
_SS_A2, _SS_A3 = 0.1, 0.2


def _swept_sine_polynomial() -> Any:
    fs, f1, f2, seconds = 48000, 20.0, 6000.0, 2.0
    x = ph.synchronized_sweep_signal(fs, f1, f2, seconds)
    y = x + _SS_A2 * x**2 + _SS_A3 * x**3
    return ph.swept_sine_distortion(y, fs, f1, f2, seconds, n_harmonics=3)


def _ss_response_at(res: Any, order: int, freq: float) -> complex:
    idx = int(np.argmin(np.abs(res.frequencies - freq)))
    return complex(res.harmonic_responses[order - 1][idx])


@register(
    _SWEPT_SINE,
    "Farina 2000 / Novak et al. 2015 (Chebyshev identity)",
    "3rd-harmonic response H3 magnitude of a cubic polynomial, re a3/4",
)
def _chk_swept_sine_h3_magnitude() -> Outcome:
    res = _swept_sine_polynomial()
    return numeric(
        _SS_A3 / 4.0, abs(_ss_response_at(res, 3, 3000.0)), 5e-4, places=5
    )


@register(
    _SWEPT_SINE,
    "Novak et al. 2015, JAES 63(10), Eqs. 18/49",
    "Synchronized-sweep phase of H3 (Chebyshev: -sin(3wt)), rad",
)
def _chk_swept_sine_h3_phase() -> Outcome:
    res = _swept_sine_polynomial()
    return numeric(
        math.pi, abs(np.angle(_ss_response_at(res, 3, 3000.0))), 5e-3,
        unit="rad", places=4,
    )


@register(
    _SWEPT_SINE,
    "Farina 2000, AES 108th Conv. (THD from one sweep)",
    "THD(1 kHz) of the polynomial vs sqrt((a2/2)^2+(a3/4)^2)/(1+3a3/4)",
)
def _chk_swept_sine_thd_closed_form() -> Outcome:
    res = _swept_sine_polynomial()
    idx = int(np.argmin(np.abs(res.thd_frequencies - 1000.0)))
    expected = math.hypot(_SS_A2 / 2.0, _SS_A3 / 4.0) / (1.0 + 3.0 * _SS_A3 / 4.0)
    return numeric(expected, float(res.thd[idx]), 1e-3, places=5)


@register(
    _SWEPT_SINE,
    "Farina 2000 (distortion rejected from the linear IR)",
    "THD floor of a purely linear path (gain 0.5), max over 100-2000 Hz",
)
def _chk_swept_sine_linear_floor() -> Outcome:
    # A longer sweep than the polynomial checks: the floor is deconvolution
    # residue, and 4 s of sweep leaves 3x headroom under the 1e-3 bound.
    fs, f1, f2, seconds = 48000, 20.0, 6000.0, 4.0
    x = ph.synchronized_sweep_signal(fs, f1, f2, seconds)
    res = ph.swept_sine_distortion(0.5 * x, fs, f1, f2, seconds, n_harmonics=3)
    band = (res.thd_frequencies > 100.0) & (res.thd_frequencies < 2000.0)
    return numeric(0.0, float(np.max(res.thd[band])), 1e-3, places=5)


@register(
    _SWEPT_SINE,
    "Bendat & Piersol, Random Data 4e Sec. 13.1.4 (Hilbert relation)",
    "Min-phase reconstruction of a strictly min-phase biquad, max err, rad",
)
def _chk_minimum_phase_biquad() -> Outcome:
    w0, q, gain = 0.3 * math.pi, 1.5, 10.0 ** (6.0 / 40.0)
    alpha = math.sin(w0) / (2.0 * q)
    b = np.array([1.0 + alpha * gain, -2.0 * math.cos(w0), 1.0 - alpha * gain])
    a = np.array([1.0 + alpha / gain, -2.0 * math.cos(w0), 1.0 - alpha / gain])
    _, resp = sg.freqz(b, a, worN=np.linspace(0.0, math.pi, 4097))
    rec = ph.minimum_phase(resp)
    err = float(np.max(np.abs(np.angle(rec) - np.angle(resp))))
    return numeric(0.0, err, 1e-9, unit="rad", places=6)


@register(
    _SWEPT_SINE,
    "First-order allpass closed form (1-a^2)/(1+2a cos w+a^2)",
    "Group delay of the a = 0.5 allpass at w = pi/2, samples",
)
def _chk_group_delay_allpass() -> Outcome:
    a_coef = 0.5
    grid = np.linspace(0.0, math.pi, 4097)
    _, resp = sg.freqz([a_coef, 1.0], [1.0, a_coef], worN=grid)
    tau = ph.group_delay(resp, 1.0)  # fs = 1 -> samples
    idx = int(np.argmin(np.abs(grid - math.pi / 2.0)))
    expected = (1.0 - a_coef**2) / (1.0 + a_coef**2)
    return numeric(expected, float(tau[idx]), 1e-5, places=5)


@register(
    _SWEPT_SINE,
    "All-pass decomposition of a pure latency (B&P Sec. 13.1.4)",
    "Excess group delay of a biquad delayed 7.25 samples, samples",
)
def _chk_excess_group_delay() -> Outcome:
    w0, q, gain = 0.3 * math.pi, 1.5, 10.0 ** (6.0 / 40.0)
    alpha = math.sin(w0) / (2.0 * q)
    b = np.array([1.0 + alpha * gain, -2.0 * math.cos(w0), 1.0 - alpha * gain])
    a = np.array([1.0 + alpha / gain, -2.0 * math.cos(w0), 1.0 - alpha / gain])
    grid = np.linspace(0.0, math.pi, 4097)
    _, resp = sg.freqz(b, a, worN=grid)
    delayed = resp * np.exp(-1j * grid * 7.25)
    res = ph.phase_decomposition(delayed, 1.0)  # fs = 1 -> samples
    return numeric(7.25, float(res.excess_group_delay[2048]), 1e-6, places=5)
