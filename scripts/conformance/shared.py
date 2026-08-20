#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Filter class and weighting deviation, computed once for two readers.

The registry checks of :mod:`conformance.domains.filters` and the "Numerical
validation - filters & weightings" showcase at the head of the report need the
same two things: the IEC 61260-1 class verdict of a designed filter bank, and
how far a designed weighting curve sits from the normative one. They are
computed here so the two can never disagree - a showcase that computed its own
numbers could print a class-1 table above a failing check row.

Each returns a small frozen record carrying not just the verdict and the
margin but the *binding* band or frequency behind it, with the measured value
and the acceptance limit it is compared against, so the report can show the
number and the range it must sit in rather than a margin alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import reference_data as ref
from scipy import signal as sg

from phonometry import filters
from phonometry.filters.compliance import class_limits, verify_filter_class
from phonometry.filters.weighting import _runtime_frequency_response

_FILTER_ARCHS = ["butter", "cheby1", "cheby2", "ellip", "bessel"]


@dataclass(frozen=True)
class FilterClass:
    """IEC 61260-1 class verification summary for one architecture.

    Beyond the pass/fail verdict and margins, this captures the *binding*
    band and point - the measured relative attenuation there and the class-1
    acceptance limit it is compared against - so the report can show the
    number and the range it must sit in, not just the margin.
    """

    overall_class: int | None
    min_margin1: float
    min_margin2: float
    bind_freq: float
    bind_measured_db: float
    bind_limit_db: float
    bind_side: str  # "ceil" (upper limit), "floor" (pass-band min) or
    # "stop" (stop-band min) - the side of the acceptance band that binds


@dataclass(frozen=True)
class WeightingDeviation:
    """Weighting deviation split into an informational maximum and the
    compliance margin evaluated at the *binding* frequency.

    ``worst_*`` is the largest |designed - nominal| across the band (usually
    at a frequency extreme, where the class-1 tolerance is widest and
    asymmetric). ``bind_*`` is taken at the frequency with the least headroom
    to its acceptance band - where deviation, the +/- tolerance and the
    headroom are co-located, so "value vs range" is unambiguous.
    """

    worst_freq: float
    worst_dev: float
    bind_freq: float
    bind_dev: float
    bind_lower: float
    bind_upper: float
    min_headroom: float


def _filter_class(arch: str, fraction: float) -> FilterClass:
    """IEC 61260-1 class verification summary for one architecture.

    The pass/fail verdict and margins come from the library's
    ``verify_filter_class`` (authoritative). The binding measured value and
    limit are re-derived here with the same public ``class_limits`` on the
    same designed SOS, so they cannot disagree with the library margin (a
    smoke-test guard asserts the re-derived margin equals the library's).
    """
    bank = filters.OctaveFilterBank(
        48000,
        fraction=fraction,
        order=6,
        limits=[100, 10000],
        design=filters.FilterDesign(filter_type=arch),
    )
    result = verify_filter_class(bank)
    bands = result["bands"]
    worst = min(bands, key=lambda b: b["margin_class1_db"])
    idx = [b["freq"] for b in bands].index(worst["freq"])
    fm = float(bank.freq[idx])
    fsd = bank.fs / float(bank.factor[idx])
    w, h = sg.sosfreqz(bank.sos[idx], worN=2**15, fs=fsd)
    attenuation = -20.0 * np.log10(np.abs(h) + np.finfo(float).eps)
    a_ref = float(np.interp(fm, w, attenuation))
    delta = attenuation - a_ref
    omega = w / fm
    valid = omega > 0
    omega, delta = omega[valid], delta[valid]
    minimum, maximum = class_limits(bank.fraction, 1, omega)
    low_margin = delta - minimum
    finite = np.isfinite(maximum)
    high_margin = np.where(finite, maximum - delta, np.inf)
    point_margin = np.minimum(low_margin, high_margin)
    j = int(np.argmin(point_margin))
    omega_h = 1.0 / omega[j] if omega[j] < 1.0 else omega[j]
    if high_margin[j] < low_margin[j]:
        bind_side, bind_limit = "ceil", float(maximum[j])
    else:
        bind_side = "floor" if omega_h <= _pass_edge(bank.fraction) else "stop"
        bind_limit = float(minimum[j])
    return FilterClass(
        overall_class=result["overall_class"],
        min_margin1=min(b["margin_class1_db"] for b in bands),
        min_margin2=min(b["margin_class2_db"] for b in bands),
        bind_freq=fm,
        bind_measured_db=float(delta[j]),
        bind_limit_db=bind_limit,
        bind_side=bind_side,
    )


def _pass_edge(fraction: float) -> float:
    """Normalized pass-band edge G**(1/(2b)) for the given bandwidth b."""
    return float((10.0 ** (3.0 / 10.0)) ** (1.0 / (2.0 * fraction)))


def _weighting_deviation(curve: str, fs: int) -> WeightingDeviation:
    """Weighting deviation vs the normative curve, informational maximum plus
    the binding-frequency compliance margin.

    The weighting filter is evaluated over the whole path a signal travels
    through ``filter()`` - the designed SOS *and* the resampling stages the
    default high-accuracy mode wraps around them, whose anti-alias filter
    dominates the response above roughly 0.9 x fs/2 - against the standard's
    nominal response. For A/C the normative band is the IEC 61672-1 Table 3
    class-1 acceptance limits; for G it is the ISO 7196 Annex A.3 +/-1 dB
    instrumentation tolerance.
    """
    wf = filters.WeightingFilter(fs, curve)
    if curve == "G":
        rows = [r for r in ref.ISO7196_TABLE2 if r[0] < fs / 2]
        # Table 2 lists nominal one-third-octave labels; evaluate at the
        # exact base-10 frequencies 10**(n/10) as IEC 61672-1 Annex D does.
        freqs = np.array([10 ** (round(10 * math.log10(r[0])) / 10) for r in rows])
        nominal = np.array([r[1] for r in rows])
        upper = np.full(nominal.shape, ref.ISO7196_G_TOLERANCE_DB)
        lower = np.full(nominal.shape, -ref.ISO7196_G_TOLERANCE_DB)
    elif curve == "B":
        # ANSI S1.4-1983: Table IV design goals (B column) against the
        # strictest Table V mask (Type 0, laboratory grade).
        rows_b = [
            (t4, t5)
            for t4, t5 in zip(ref.ANSIS14_TABLE4_B, ref.ANSIS14_TABLE5, strict=True)
            if t4[0] < fs / 2
        ]
        freqs = np.array(
            [10 ** (round(10 * math.log10(t4[0])) / 10) for t4, _ in rows_b]
        )
        nominal = np.array([t4[1] for t4, _ in rows_b])
        upper = np.array([t5[1] for _, t5 in rows_b])
        lower = np.array([t5[2] for _, t5 in rows_b])
    elif curve == "AU":
        # IEC 61012:1990: nominal AU = nominal A + nominal U (Table 1), with
        # the subclause 2.2 explicit values above 20 kHz, against the Table 1
        # separate-unit tolerances. The 1 kHz reference row carries zero
        # tolerance and the normalized deviation there is identically zero,
        # so it is skipped to keep the binding-frequency row informative.
        a_nom = {r[0]: r[1] for r in ref.IEC61672_TABLE3}
        rows_u = [r for r in ref.IEC61012_TABLE1 if r[0] < fs / 2 and r[0] != 1000]
        freqs = np.array([10 ** (round(10 * math.log10(r[0])) / 10) for r in rows_u])
        nominal = np.array(
            [ref.IEC61012_AU_HF.get(r[0], a_nom.get(r[0], 0.0) + r[1]) for r in rows_u]
        )
        upper = np.array([r[2] for r in rows_u])
        lower = np.array([r[3] for r in rows_u])
    else:
        col = 1 if curve == "A" else 2
        rows = [r for r in ref.IEC61672_TABLE3 if r[0] < fs / 2]
        # Table 3 NOTE: the design goals are computed at the exact base-10
        # frequencies 1000 * 10^(0.1 (n - 30)) behind the nominal labels
        # (15 848.9 Hz for "16 k"); evaluate the filter there, as the G
        # branch above and IEC 61672-3:2013 subclause 13.3 do.
        freqs = np.array([10 ** (round(10 * math.log10(r[0])) / 10) for r in rows])
        nominal = np.array([r[col] for r in rows])
        upper = np.array([r[3] for r in rows])
        lower = np.array([r[4] for r in rows])

    response = 20.0 * np.log10(np.abs(_runtime_frequency_response(wf, freqs)))
    deviation = response - nominal
    worst_idx = int(np.argmax(np.abs(deviation)))
    headroom = np.minimum(upper - deviation, deviation - lower)
    bind_idx = int(np.argmin(headroom))
    return WeightingDeviation(
        worst_freq=float(freqs[worst_idx]),
        worst_dev=float(deviation[worst_idx]),
        bind_freq=float(freqs[bind_idx]),
        bind_dev=float(deviation[bind_idx]),
        bind_lower=float(lower[bind_idx]),
        bind_upper=float(upper[bind_idx]),
        min_headroom=float(headroom[bind_idx]),
    )
