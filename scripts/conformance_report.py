#  Copyright (c) 2026. Jose M. Requena-Plens
"""Numerical conformance report for phonometry.

A maintainable registry of numerical conformance checks. Each entry pins one
(standard, quantity) pair to:

* the standard designation + clause/table citation,
* the normative expected value or range (from the standard's own worked
  examples, or a closed form synthesized to a known result),
* a callable that computes the library's result, and
* the tolerance.

Running the harness emits Markdown for a GitHub PR comment: a headline
summary followed only by collapsible sections, so the comment stays compact
by default. First a "Numerical validation - filters & weightings" showcase
(per filter architecture IEC 61260-1 class margins and A/C/G weighting
worst-case deviation vs the analytic/normative curve), then one conformance
table per domain (Standard | Quantity | Expected | Computed | Delta |
Status). Every section stays collapsed while all of its rows pass and opens
automatically when any row fails.

Expected values are pulled from a single source of truth wherever the tests
already encode them: the shared ``tests/reference_data`` tables and the
ISO 532-1 data fixtures. Where the reference is a closed form, the harness
synthesizes an input with a known output, so the check is self-verifying.

Design goals: deterministic, fast (< 1 min), no network, pure library calls.
"""

from __future__ import annotations

import cmath
import json
import math
import pathlib
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from scipy import signal as sg

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TESTS = _ROOT / "tests"
_DATA = _TESTS / "data"
for _p in (str(_TESTS),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reference_data as ref

import phonometry as ph
from phonometry import OctaveFilterBank, WeightingFilter
from phonometry.hearing.sti import _sti_from_mtf
from phonometry.metrology.compliance import (
    class_limits,
    verify_filter_class,
)
from phonometry.psychoacoustics.sharpness import reference_sound


# ===========================================================================
# Registry primitives
# ===========================================================================
@dataclass(frozen=True)
class Outcome:
    """The rendered result of a single conformance check."""

    expected: str
    computed: str
    delta: str
    passed: bool


@dataclass(frozen=True)
class Check:
    """A registered (standard, quantity) conformance check."""

    domain: str
    standard: str
    quantity: str
    run: Callable[[], Outcome]


CHECKS: list[Check] = []


def register(domain: str, standard: str, quantity: str) -> Callable[
    [Callable[[], Outcome]], Callable[[], Outcome]
]:
    """Register a check callable under a domain / standard / quantity."""

    def deco(fn: Callable[[], Outcome]) -> Callable[[], Outcome]:
        CHECKS.append(Check(domain, standard, quantity, fn))
        return fn

    return deco


# Decimal places for the informational delta column. Coarse on purpose so the
# committed docs/CONFORMANCE.md stays byte-stable across numpy/scipy/BLAS
# builds (see the note in ``numeric``).
_DELTA_PLACES = 3


def _fmt(value: float, unit: str = "", places: int = 4) -> str:
    """Compact fixed/again-significant formatting with an optional unit."""
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return f"{text} {unit}".strip()


def numeric(
    expected: float,
    computed: float,
    tol: float,
    *,
    unit: str = "",
    rel: bool = False,
    places: int = 4,
    expected_label: str | None = None,
) -> Outcome:
    """Build an Outcome for ``|computed - expected| <= tol`` (abs or rel)."""
    delta = computed - expected
    limit = tol * abs(expected) if rel else tol
    passed = abs(delta) <= limit
    tol_txt = f"{tol * 100:g}%" if rel else _fmt(tol, unit, places)
    exp_txt = expected_label or _fmt(expected, unit, places)
    return Outcome(
        expected=f"{exp_txt} (+/-{tol_txt})" if not expected_label else exp_txt,
        computed=_fmt(computed, unit, places),
        # The delta column is informational (pass/fail comes from ``passed``),
        # so it is coarsened to 3 decimals. This collapses sub-milli residuals
        # of the heavy DSP chains (ECMA/Moore-Glasberg/Zwicker, FFT intensity),
        # whose 4th-5th digit is BLAS/FFT-build dependent, to a stable value -
        # keeping the committed report diff-stable without hiding a regression
        # (that still shows in the Computed column and flips the status).
        delta=_fmt(delta, unit, _DELTA_PLACES),
        passed=passed,
    )


# ===========================================================================
# Shared compute helpers (called by both the registry and the numerical
# validation section, so the two can never disagree).
# ===========================================================================
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
    bank = OctaveFilterBank(
        48000, fraction=fraction, order=6, limits=[100, 10000], filter_type=arch
    )
    result = verify_filter_class(bank)
    bands = result["bands"]
    worst = min(bands, key=lambda b: b["margin_class1_db"])
    idx = [b["freq"] for b in bands].index(worst["freq"])
    fm = float(bank.freq[idx])
    fsd = bank.fs / float(bank.factor[idx])
    w, h = sg.sosfreqz(bank.sos[idx], worN=2 ** 15, fs=fsd)
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

    The weighting filter's designed SOS (at its internal oversampled rate)
    is evaluated against the standard's nominal response. For A/C the
    normative band is the IEC 61672-1 Table 3 class-1 acceptance limits;
    for G it is the ISO 7196 Annex A.3 +/-1 dB instrumentation tolerance.
    """
    wf = WeightingFilter(fs, curve)
    design_fs = wf.fs * wf._oversample
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
            for t4, t5 in zip(ref.ANSIS14_TABLE4_B, ref.ANSIS14_TABLE5)
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
        # (15 848.9 Hz for "16 k"); evaluate the SOS there, as the G branch
        # above and IEC 61672-3:2013 subclause 13.3 do.
        freqs = np.array([10 ** (round(10 * math.log10(r[0])) / 10) for r in rows])
        nominal = np.array([r[col] for r in rows])
        upper = np.array([r[3] for r in rows])
        lower = np.array([r[4] for r in rows])

    _, h = sg.sosfreqz(wf.sos, worN=freqs, fs=design_fs)
    response = 20.0 * np.log10(np.abs(h))
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


# ===========================================================================
# Domain 1 - Filters & weightings
# ===========================================================================
def _filter_class_check(arch: str, fraction: float, label: str) -> Outcome:
    res = _filter_class(arch, fraction)
    margin = res.min_margin1
    ok = res.overall_class == 1
    return Outcome(
        expected="class 1",
        computed=(f"class {res.overall_class}" if res.overall_class else "none")
        + f" (margin {margin:+.3f} dB)",
        delta=f"{margin:+.3f} dB",
        passed=ok,
    )


@register(
    "Filters & weightings",
    "IEC 61260-1:2014 Table 1",
    "Octave-band filter class (butterworth, fs=48 kHz)",
)
def _chk_butter_octave() -> Outcome:
    return _filter_class_check("butter", 1, "octave")


@register(
    "Filters & weightings",
    "IEC 61260-1:2014 Table 1",
    "One-third-octave filter class (butterworth, fs=48 kHz)",
)
def _chk_butter_third() -> Outcome:
    return _filter_class_check("butter", 3, "third")


@register(
    "Filters & weightings",
    "IEC 61260:1995 / ANSI S1.11-2004 Table 1",
    "Class 0 (strictest) octave-band filter (butterworth, fs=48 kHz)",
)
def _chk_butter_class0_1995() -> Outcome:
    bank = OctaveFilterBank(48000, fraction=1, order=6, limits=[100, 10000], filter_type="butter")
    result = ph.verify_filter_class(bank, edition="1995")
    margin = min(b["margin_class0_db"] for b in result["bands"])
    ok = result["overall_class"] == 0
    return Outcome(
        expected="class 0",
        computed=(f"class {result['overall_class']}" if result["overall_class"] is not None
                  else "none") + f" (margin {margin:+.3f} dB)",
        delta=f"{margin:+.3f} dB",
        passed=ok,
    )


@register(
    "Filters & weightings",
    "IEC 61260-1:2014 Table F.1",
    "Formula (9) breakpoint mapping, b=3, Omega at G**(1/2)",
)
def _chk_map_breakpoint_table_f1() -> Outcome:
    from phonometry.metrology.compliance import _map_breakpoint

    return numeric(
        ref.IEC61260_TABLE_F1[0.5][0], _map_breakpoint(0.5, 3), 5e-6, places=5
    )


def _weighting_check(curve: str, fs: int) -> Outcome:
    res = _weighting_deviation(curve, fs)
    headroom = res.min_headroom
    band = f"[{res.bind_lower:+.2f}, {res.bind_upper:+.2f}] dB"
    return Outcome(
        expected=f"deviation within limits @ {res.bind_freq:.0f} Hz",
        computed=f"{_snap(res.bind_dev):+.3f} dB in {band}",
        delta=f"headroom {headroom:+.3f} dB",
        passed=headroom >= 0.0,
    )


@register(
    "Filters & weightings",
    "IEC 61672-1:2013 Table 3",
    "A-weighting deviation vs class-1 limits (fs=48 kHz)",
)
def _chk_a_weighting() -> Outcome:
    return _weighting_check("A", 48000)


@register(
    "Filters & weightings",
    "IEC 61672-1:2013 Table 3",
    "C-weighting deviation vs class-1 limits (fs=48 kHz)",
)
def _chk_c_weighting() -> Outcome:
    return _weighting_check("C", 48000)


@register(
    "Filters & weightings",
    "ISO 7196:1995 Table 2 / A.3",
    "G-weighting deviation vs +/-1 dB tolerance (fs=48 kHz)",
)
def _chk_g_weighting() -> Outcome:
    return _weighting_check("G", 48000)


@register(
    "Filters & weightings",
    "ANSI S1.4-1983 Tables IV/V",
    "B-weighting (historical) deviation vs Type 0 limits (fs=48 kHz)",
)
def _chk_b_weighting() -> Outcome:
    return _weighting_check("B", 48000)


@register(
    "Filters & weightings",
    "IEC 61012:1990 Table 1 / 2.2",
    "AU-weighting deviation vs separate-unit tolerances (fs=96 kHz)",
)
def _chk_au_weighting() -> Outcome:
    # 96 kHz so the 25/31.5/40 kHz rows (exact base-10 frequencies up to
    # 39 811 Hz) fall below Nyquist and the full Table 1 range is checked.
    return _weighting_check("AU", 96000)


@register(
    "Filters & weightings",
    "IEC 537:1976 (withdrawn) via NASA CR-3406 Table SLD-I",
    "D-weighting response vs the published tabulated curve (fs=48 kHz)",
)
def _chk_d_weighting() -> Outcome:
    # IEC 537 is withdrawn and published no surviving tolerance table, so the
    # D response is pinned against the tabulated curve republished in the
    # NASA Handbook of Aircraft Noise Metrics (Table SLD-I, printed at the
    # integer nominal frequencies to 0.1 dB). The rational transfer function
    # reproduces every row within 0.1 dB except 1600/2500 Hz, which appear to
    # round a different source curve; the realized filter adds bilinear
    # residuals below 0.1 dB, so the acceptance bound is 0.2 dB (0.45 dB at
    # the two outlier cells).
    wf = WeightingFilter(48000, "D")
    design_fs = wf.fs * wf._oversample
    freqs = np.array([r[0] for r in ref.IEC537_NASA_TABLE_SLD1], dtype=float)
    table = np.array([r[1] for r in ref.IEC537_NASA_TABLE_SLD1], dtype=float)
    _, h = sg.sosfreqz(wf.sos, worN=np.concatenate([freqs, [1000.0]]), fs=design_fs)
    gain = 20.0 * np.log10(np.abs(h))
    dev = (gain[:-1] - gain[-1]) - table
    bound = np.where(np.isin(freqs, (1600.0, 2500.0)), 0.45, 0.2)
    worst = int(np.argmax(np.abs(dev) / bound))
    ok = bool(np.all(np.abs(dev) <= bound))
    return Outcome(
        expected="abs(response - table) <= 0.2 dB (0.45 dB at 1600/2500 Hz)",
        computed=(
            f"{dev[worst]:+.3f} dB @ {freqs[worst]:.0f} Hz "
            f"(bound {bound[worst]:.2f} dB)"
        ),
        delta=f"headroom {bound[worst] - abs(dev[worst]):+.3f} dB",
        passed=ok,
    )


# ===========================================================================
# Domain 2 - Levels & dosimetry
# ===========================================================================
_FS = 48000


def _tone(freq: float, seconds: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(_FS * seconds)) / _FS
    return amp * np.sin(2.0 * np.pi * freq * t)


@register(
    "Levels & dosimetry",
    "IEC 61672-1:2013 (Leq)",
    "Leq of a 1 Pa 1 kHz sine",
)
def _chk_leq_sine() -> Outcome:
    # 20*log10((1/sqrt2) / 20e-6) = 90.97 dB.
    computed = float(ph.leq(_tone(1000.0)))
    return numeric(90.97, computed, 0.05, unit="dB", places=3)


@register(
    "Levels & dosimetry",
    "IEC 61252:1995 (LEX,8h)",
    "8 h exposure to 90 dB(A) noise",
)
def _chk_lex_8h() -> Outcome:
    rms = 2e-5 * 10 ** (90.0 / 20.0)
    x = math.sqrt(2) * rms * _tone(1000.0, seconds=2.0)
    computed = float(ph.lex_8h(x, _FS, duration_hours=8.0))
    return numeric(90.0, computed, 0.05, unit="dB", places=3)


@register(
    "Levels & dosimetry",
    "ISO 1996-1:2016 3.6.4",
    "Lden, constant 60 dB in day/evening/night",
)
def _chk_lden() -> Outcome:
    offset = 10.0 * math.log10((12 + 4 * 10**0.5 + 8 * 10) / 24)
    computed = float(ph.lden(60.0, 60.0, 60.0))
    return numeric(60.0 + offset, computed, 1e-6, unit="dB", places=4)


@register(
    "Levels & dosimetry",
    "ISO 1996-2:2007 Annex C.5 Example 1",
    "Tonal audibility ΔLta (Formula C.3), 4 kHz tone",
)
def _chk_iso1996_2_tonal_audibility() -> Outcome:
    lpt, lpn, fc, delta_expected, _kt = ref.ISO1996_2_TONAL_EXAMPLES[0]
    computed = ph.tonal_audibility(lpt, lpn, fc)
    return numeric(delta_expected, computed, 0.05, unit="dB", places=2)


@register(
    "Levels & dosimetry",
    "ISO 1996-2:2007 Annex C.5 Example 1",
    "Tonal adjustment Kt (Formulae C.4-C.6)",
)
def _chk_iso1996_2_tonal_adjustment() -> Outcome:
    lpt, lpn, fc, _delta, kt_expected = ref.ISO1996_2_TONAL_EXAMPLES[0]
    computed = ph.tonal_adjustment(ph.tonal_audibility(lpt, lpn, fc))
    return numeric(kt_expected, computed, 1e-9, unit="dB", places=2)


@register(
    "Levels & dosimetry",
    "ISO 1996-2:2017 Annex G.2",
    "Combined measurement uncertainty u = √(Σ(cj·uj)²)",
)
def _chk_iso1996_2_uncertainty() -> Outcome:
    computed = ph.combined_standard_uncertainty(ref.ISO1996_2_G2_CONTRIBUTIONS)
    return numeric(ref.ISO1996_2_G2_COMBINED, computed, 0.01, unit="dB", places=2)


# ---------------------------------------------------------------------------
# Spanish noise regulation (RD 1367/2007) and building code (CTE DB-HR).
# Oracles: the printed limit tables and procedures of the two legal texts, and
# the worked examples of Aviles Lopez & Perera Martin, "Manual de acustica
# ambiental y arquitectonica" (Paraninfo), Ejemplos 3.1-3.3 and 7.2 / 7.4.
# ---------------------------------------------------------------------------
#: Manual Ejemplo 3.1: the day period of an activity on residential land,
#: split into 2 h shut down, 6 h with the noisy machine (LAeq 50 dB, Kt 6,
#: Kf 3) and 4 h with the remaining sources (LAeq 48 dB, Kt 3, Kf 3).
_RD1367_DAY_PHASES = [
    (2.0, 0.0, 0.0, 0.0),
    (6.0, 50.0, 6.0, 3.0),
    (4.0, 48.0, 3.0, 3.0),
]
#: Manual Ejemplo 7.2: the measured apparent sound reduction index R' of a
#: field test, one-third-octave bands 100 Hz to 5 kHz.
_DBHR_R_PRIME = [
    36.2, 41.5, 36.9, 40.4, 44.7, 42.4, 45.7, 46.1, 47.1,
    52.3, 54.3, 57.5, 57.8, 57.3, 59.0, 62.8, 64.7, 65.3,
]


@register(
    "Levels & dosimetry",
    "RD 1367/2007 Annex IV A.3.4.2 b",
    "Corrected period level LKeq,d (Manual Ejemplo 3.1: 3 noise phases, 12 h)",
)
def _chk_rd1367_period_level() -> Outcome:
    phases = [
        ph.NoisePhase(hours, laeq, kt=kt, kf=kf)
        for hours, laeq, kt, kf in _RD1367_DAY_PHASES
    ]
    level = ph.evaluation_period_level(phases, hours=12.0)
    return numeric(57.0, float(ph.round_reported_level(level)), 1e-9, unit="dB")


@register(
    "Levels & dosimetry",
    "RD 1367/2007 Annex I A.2 d",
    "Long-term level LK,d (Manual Ejemplo 3.2: 303 operating days of 365)",
)
def _chk_rd1367_long_term_level() -> Outcome:
    level = ph.long_term_corrected_level([57.0, 0.0], weights=[303.0, 62.0])
    return numeric(56.0, float(ph.round_reported_level(level)), 1e-9, unit="dB")


@register(
    "Levels & dosimetry",
    "RD 1367/2007 Annex III Table B1, Article 25",
    "Activity verdict (Manual Ejemplo 3.3: area type a, LK,d 56 dB over 55 dB)",
)
def _chk_rd1367_activity_verdict() -> Outcome:
    day = [
        ph.NoisePhase(hours, laeq, kt=kt, kf=kf)
        for hours, laeq, kt, kf in _RD1367_DAY_PHASES
    ]
    evening = [ph.NoisePhase(2.0, 48.0, kt=3.0, kf=3.0), ph.NoisePhase(2.0, 0.0)]
    verdict = ph.assess_activity(
        {"day": day, "evening": evening},
        ph.activity_limits("a"),
        operating_days=303,
    )
    # The book: the phase and daily criteria pass, the annual one does not, so
    # a new activity does not comply while one already in operation does.
    computed = (
        verdict.periods[0].phase_pass
        and verdict.periods[0].daily_pass
        and not verdict.periods[0].long_term_pass
        and not verdict.complies
    )
    return Outcome(
        expected="phase and daily pass, annual fails, activity not compliant",
        computed=(
            "phase and daily pass, annual fails, activity not compliant"
            if computed
            else "verdict differs"
        ),
        delta="-",
        passed=bool(computed),
    )


@register(
    "Room & building acoustics",
    "CTE DB-HR Annex A, Formula (A.5)",
    "Global index R'A for pink noise (Manual Ejemplo 7.2)",
)
def _chk_dbhr_ra() -> Outcome:
    computed = ph.ra(_DBHR_R_PRIME).intermediate
    return numeric(51.4, float(computed), 0.05, unit="dBA", places=2)


@register(
    "Room & building acoustics",
    "CTE DB-HR Annex A, Formula (A.6)",
    "Global index D2m,nT,Atr for road traffic (Manual Ejercicio 7.1)",
)
def _chk_dbhr_d2m_nt_atr() -> Outcome:
    values = [
        28.5, 28.5, 18.9, 23.7, 30.7, 31.3, 37.8, 35.2, 34.7,
        38.5, 37.7, 43.1, 42.3, 44.2, 41.9, 37.5, 39.4, 41.5,
    ]
    computed = ph.d2m_nt_atr(values).intermediate
    return numeric(32.8, float(computed), 0.05, unit="dBA", places=2)


@register(
    "Room & building acoustics",
    "Manual de acustica ambiental y arquitectonica, Ejemplo 7.1",
    "Reported R'A of the field-test wall (printed 51 dBA = R'w 52 + C -1)",
)
def _chk_dbhr_route_agreement() -> Outcome:
    # The book prints the pink-noise figure as 51 dBA, so 51 is the oracle;
    # comparing the direct route against the library's own ISO 717-1 engine
    # would be a self-consistency check, not an external validation. The unit
    # test pins R'w = 52 and C = -1 against the same printed example.
    computed = float(ph.ra(_DBHR_R_PRIME).reported)
    return numeric(51.0, computed, 1e-9, unit="dBA", places=1)


@register(
    "Room & building acoustics",
    "Manual de acustica ambiental y arquitectonica, Ejemplo 7.1",
    "Reported R'A,tr of the same wall (printed 47 dBA = R'w 52 + Ctr -5)",
)
def _chk_dbhr_ra_tr() -> Outcome:
    computed = float(ph.ra_tr(_DBHR_R_PRIME).reported)
    return numeric(47.0, computed, 1e-9, unit="dBA", places=1)


@register(
    "Room & building acoustics",
    "CTE Catalogo de Elementos Constructivos",
    "Window size correction of RA (Manual Ejemplo 7.4: 4 m2 window, -2 dB)",
)
def _chk_dbhr_window_correction() -> Outcome:
    computed = 26.0 + ph.window_size_correction(4.0)
    return numeric(24.0, float(computed), 1e-9, unit="dBA", places=1)


# ---------------------------------------------------------------------------
# Reverberation-time prediction (Sabine / Eyring / Millington / Fitzroy /
# Arau-Puchades). No source carries a machine-readable worked example, so the
# checks anchor on hand-computed closed-form values and the model identities.
# ---------------------------------------------------------------------------
# Shoebox 8x5x3 m: V = 120 m3, S = 158 m2. Values hand-derived with the default
# c0 = 343 m/s (Sabine constant k = 24 ln10 / c0 = 0.161113...).
_RT_DIMS = (8.0, 5.0, 3.0)
_RT_VOLUME = 120.0
_RT_SURFACES = [
    (40.0, 0.2), (40.0, 0.2), (24.0, 0.2), (24.0, 0.2), (15.0, 0.2), (15.0, 0.2)
]


@register(
    "Room acoustics",
    "Sabine (W. C. Sabine, 1922)",
    "Reverberation time T = k·V/A  (V=120 m³, S=158 m², α=0.2)",
)
def _chk_sabine_rt() -> Outcome:
    computed = float(ph.sabine_reverberation_time(_RT_VOLUME, _RT_SURFACES))
    return numeric(0.6118246547, computed, 1e-6, unit="s", places=6)


@register(
    "Room acoustics",
    "Long, Architectural Acoustics 2e, Table 8.1",
    "Room modes of a 7 x 5 x 3 m room: the six printed frequencies, Hz",
)
def _chk_long_room_modes() -> Outcome:
    # Long's Chapter 2 quotes c0 = 344 m/s at 20 degC; the printed table is
    # consistent with 344.7 m/s, so the tolerance is one printed digit.
    printed = np.array([24.6, 34.5, 42.4, 49.2, 57.4, 60.1])
    res = ph.room_modes((7.0, 5.0, 3.0), max_frequency=61.0, speed_of_sound=344.0)
    computed = np.asarray(res.frequencies, dtype=float)
    worst = int(np.argmax(np.abs(computed - printed)))
    return numeric(printed[worst], computed[worst], 0.13, unit="Hz", places=2)


@register(
    "Room acoustics",
    "Long, Architectural Acoustics 2e, Eq. (8.46)",
    "Modal density of a 7 x 5 x 3 m room at 1 kHz = 34 modes/Hz",
)
def _chk_long_modal_density() -> Outcome:
    density = float(
        np.asarray(
            ph.room_modal_density(1000.0, (7.0, 5.0, 3.0), speed_of_sound=344.0)
        )[()]
    )
    return numeric(34.0, density, 0.5, unit="modes/Hz", places=2)


@register(
    "Room acoustics",
    "Long, Architectural Acoustics 2e, Eq. (17.51)",
    "Restaurant self-noise, 20 talkers over 20 metric sabins = 76 dB",
)
def _chk_long_restaurant_self_noise() -> Outcome:
    level = float(np.asarray(ph.crowd_noise_level(20, 20.0))[()])
    return numeric(76.0, level, 0.05, unit="dB", places=3)


@register(
    "Room acoustics",
    "Long, Architectural Acoustics 2e, Eq. (17.54)",
    "Privacy bound A_tab < 3.16 rt^2 (Q = 2, L_SN = -9 dB)",
)
def _chk_long_privacy_bound() -> Outcome:
    a_tab = float(np.asarray(ph.absorption_per_table(1.0, -9.0))[()])
    return numeric(3.16, a_tab, 0.005, unit="m^2", places=4)


@register(
    "Room acoustics",
    "Everest, Master Handbook of Acoustics 4th ed, Fig. 7-22",
    "Sabine RT, worked Example 1 @ 1 kHz (untreated 23.3×16×10 ft room, SI)",
)
def _chk_sabine_everest() -> Outcome:
    surfaces = [
        (ref.EVEREST_EX1_FLOOR_AREA, ref.EVEREST_EX1_FLOOR_ALPHA[3]),
        (ref.EVEREST_EX1_SHELL_AREA, ref.EVEREST_EX1_SHELL_ALPHA[3]),
    ]
    computed = float(ph.sabine_reverberation_time(ref.EVEREST_EX1_VOLUME, surfaces))
    return numeric(ref.EVEREST_EX1_RT[3], computed, 0.02, unit="s", places=3)


@register(
    "Room acoustics",
    "Eyring (Norris-Eyring, 1930)",
    "Reverberation time T = k·V/(-S·ln(1-ᾱ))  (α=0.2)",
)
def _chk_eyring_rt() -> Outcome:
    computed = float(ph.eyring_reverberation_time(_RT_VOLUME, _RT_SURFACES))
    return numeric(0.5483686633, computed, 1e-6, unit="s", places=6)


@register(
    "Room acoustics",
    "Arau-Puchades (Acustica 65, 1988, Formula 18)",
    "T (α=0.5/0.1/0.1 per wall pair, dims 8×5×3 m)",
)
def _chk_arau_rt() -> Outcome:
    computed = float(ph.arau_puchades_reverberation_time(_RT_DIMS, (0.5, 0.1, 0.1)))
    return numeric(0.8121469281, computed, 1e-6, unit="s", places=6)


@register(
    "Room acoustics",
    "Model identity (uniform absorption)",
    "Arau-Puchades ≡ Eyring when ᾱ is uniform",
)
def _chk_arau_eyring_identity() -> Outcome:
    eyring = float(ph.eyring_reverberation_time(_RT_VOLUME, _RT_SURFACES))
    arau = float(ph.arau_puchades_reverberation_time(_RT_DIMS, (0.2, 0.2, 0.2)))
    return numeric(eyring, arau, 1e-9, unit="s", places=6,
                   expected_label=f"{eyring:.6f} s (= Eyring)")


@register(
    "Room acoustics",
    "Vorlander Auralization 2e, Eq. (11.38)-(11.39)",
    "Image-source direct-sound amplitude 1/(4πr) and delay r/c (r = 4 m)",
)
def _chk_image_source_direct() -> Outcome:
    res = ph.image_source_rir((8.0, 5.0, 3.0), (2.0, 2.5, 1.5),
                              (6.0, 2.5, 1.5), 0.2, fs=48000, max_order=2)
    amp = float(np.atleast_1d(res.amplitudes)[0])
    return numeric(1.0 / (4.0 * math.pi * 4.0), amp, 1e-9, places=7)


@register(
    "Room acoustics",
    "Kuttruff Room Acoustics 6e, Eq. (9.23)",
    "Audible shoebox image count up to order 10 (= 1560)",
)
def _chk_image_source_count() -> Outcome:
    computed = float(ph.audible_image_count(10))
    return numeric(1560.0, computed, 0.0, places=0)


@register(
    "Room acoustics",
    "Kuttruff Room Acoustics 6e, Eq. (4.6)",
    "Temporal reflection density dN/dt = 4πc³t²/V (t = 0.1 s, V = 120 m³)",
)
def _chk_reflection_density() -> Outcome:
    computed = float(ph.reflection_density(0.1, 120.0))
    expected = 4.0 * math.pi * 343.0**3 * 0.1**2 / 120.0
    return numeric(expected, computed, 1e-6, unit="1/s", places=2)


@register(
    "Room acoustics",
    "Bies Engineering Noise Control 5e, Eq. (6.44)",
    "Room constant R = Sᾱ/(1-ᾱ)  (S = 100 m², ᾱ = 0.2 → 25 m²)",
)
def _chk_room_constant() -> Outcome:
    return numeric(25.0, float(ph.room_constant(100.0, 0.2)), 1e-9,
                   unit="m²", places=6)


@register(
    "Room acoustics",
    "Bies Engineering Noise Control 5e, Eq. (6.43)",
    "Critical distance rc: direct field = reverberant field (R = 25, Q = 1)",
)
def _chk_critical_distance_crossover() -> Outcome:
    rc = float(ph.critical_distance(25.0))
    direct = 1.0 / (4.0 * math.pi * rc**2)
    reverberant = 4.0 / 25.0
    return numeric(reverberant, direct, 1e-9, places=6,
                   expected_label=f"{reverberant:.6f} (= reverberant term)")


@register(
    "Room acoustics",
    "Kuttruff Room Acoustics 6e, Eq. (3.44)",
    "Schroeder frequency f_s = 2000√(T/V)  (V = 200 m³, T = 1 s)",
)
def _chk_schroeder_frequency() -> Outcome:
    computed = float(ph.schroeder_frequency(1.0, 200.0))
    return numeric(2000.0 * math.sqrt(1.0 / 200.0), computed, 1e-6,
                   unit="Hz", places=3)


@register(
    "Room acoustics",
    "Bies Engineering Noise Control 5e, Eq. (6.43)",
    "Steady-state SPL Lp = Lw + 10lg(Q/4πr² + 4/R)  (Lw=90, r=1, R=25, Q=1)",
)
def _chk_steady_state_spl() -> Outcome:
    computed = float(ph.steady_state_spl(90.0, 1.0, 25.0))
    expected = 90.0 + 10.0 * math.log10(1.0 / (4.0 * math.pi) + 4.0 / 25.0)
    return numeric(expected, computed, 1e-6, unit="dB", places=4)


# ===========================================================================
# Domain 3 - Psychoacoustics
# ===========================================================================
def _iso532_stationary_expected() -> tuple[float, float, float]:
    data = json.loads((_DATA / "iso532_1" / "iso532_1_annexB_expected.json").read_text())
    entry = data["Test signal 1.txt"]
    return entry["N"], entry["Nmin"], entry["Nmax"]


def _iso532_levels() -> np.ndarray:
    levels = []
    for line in (_DATA / "iso532_1" / "iso532_1_test_signal_1_levels.txt").read_text().splitlines():
        if ":" in line and not line.strip().startswith("#"):
            levels.append(float(line.split(":")[1]))
    return np.array(levels)


@register(
    "Psychoacoustics",
    "Moore, Psychology of Hearing 6e, p. 77 (Glasberg & Moore 1990)",
    "ERB_N number of 1000 Hz = 15.59 Cam",
)
def _chk_cam_of_1_khz() -> Outcome:
    cam = float(np.asarray(ph.cam_from_frequency(1000.0))[()])
    return numeric(15.59, cam, 0.005, unit="Cam", places=4)


@register(
    "Psychoacoustics",
    "Moore, Psychology of Hearing 6e, p. 76 (Glasberg & Moore 1990)",
    "ERB_N at 1 kHz vs the printed 24.7(4.37F + 1), Hz",
)
def _chk_erb_bandwidth_1_khz() -> Outcome:
    published = 24.7 * (4.37 * 1.0 + 1.0)
    erb = float(np.asarray(ph.erb_bandwidth(1000.0))[()])
    return numeric(published, erb, 3e-3, unit="Hz", rel=True, places=3)


@register(
    "Psychoacoustics",
    "ISO 532-1:2017 Annex B.2",
    "Zwicker loudness N, stationary test signal 1",
)
def _chk_iso532_stationary() -> Outcome:
    expected_n, nmin, nmax = _iso532_stationary_expected()
    res = ph.loudness_zwicker_from_spectrum(_iso532_levels(), field="free")
    computed = float(res.loudness)
    out = numeric(expected_n, computed, 0.001, unit="sone", rel=True, places=4)
    within_band = nmin <= computed <= nmax
    return Outcome(out.expected, out.computed, out.delta, out.passed and within_band)


def _iso532_b5_signal(
    num: int,
) -> tuple[np.ndarray, int, float, Literal["free", "diffuse"]]:
    """Load a recorded ISO 532-1 Annex B.5 technical signal and its expected values.

    Returns the pressure signal, its sample rate, the expected maximum loudness
    Nmax (sone) and the sound field the ISO results workbook was computed in.
    The 16-bit WAV is scaled per Annex B.1 (full scale = 100 dB SPL, so one
    full-scale unit is 2*sqrt(2) Pa peak).
    """
    import glob
    import wave

    data = json.loads((_DATA / "iso532_1" / "iso532_1_annexB_expected.json").read_text())
    entry = data[f"Test signal {num}"]
    matches = glob.glob(str(_DATA / "iso532_1" / "Annex B.5" / f"Test signal {num} *.wav"))
    if not matches:
        raise FileNotFoundError(
            f"No ISO 532-1 Annex B.5 WAV found for Test signal {num} "
            "(see tests/data/iso532_1/README.md)."
        )
    # WAV/RIFF is little-endian, so the dtype is fixed to '<i2'; a multi-channel
    # file keeps only channel 0.
    with wave.open(matches[0]) as handle:
        fs = handle.getframerate()
        n_channels = handle.getnchannels()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    if n_channels > 1:
        raw = raw.reshape(-1, n_channels)[:, 0]
    signal = raw.astype(np.float64) / 32768.0 * (2.0 * math.sqrt(2.0))
    field = str(entry["field"])
    if field not in ("free", "diffuse"):
        raise ValueError(f"unexpected sound field {field!r} in the workbook")
    return signal, int(fs), float(entry["Nmax"]), cast(Literal["free", "diffuse"], field)


@register(
    "Psychoacoustics",
    "ISO 532-1:2017 Annex B.5",
    "Time-varying loudness Nmax, technical signal 14 (aircraft, free field)",
)
def _chk_iso532_b5_free() -> Outcome:
    signal, fs, nmax, field = _iso532_b5_signal(14)
    res = ph.loudness_zwicker(signal, fs, field=field)
    return numeric(nmax, float(res.loudness), 0.001, unit="sone", rel=True, places=4)


@register(
    "Psychoacoustics",
    "ISO 532-1:2017 Annex B.5",
    "Time-varying loudness Nmax, technical signal 15 (vehicle interior, diffuse field)",
)
def _chk_iso532_b5_diffuse() -> Outcome:
    signal, fs, nmax, field = _iso532_b5_signal(15)
    res = ph.loudness_zwicker(signal, fs, field=field)
    return numeric(nmax, float(res.loudness), 0.001, unit="sone", rel=True, places=4)


@register(
    "Psychoacoustics",
    "DIN 45692:2009 Clause 6",
    "Sharpness of the standard 1 kHz reference signal",
)
def _chk_sharpness_reference() -> Outcome:
    # Definitional: the clause 5.2 constant k is derived so this very signal
    # reads 1.00 acum. The independent Table A.2 oracle is checked below.
    computed = float(ph.sharpness_din(reference_sound(), _FS))
    return numeric(1.0, computed, 1e-9, unit="acum", places=6)


@register(
    "Psychoacoustics",
    "DIN 45692:2009 Table A.2",
    "Sharpness of critical-band noise at 2.5 kHz (2320-2700 Hz, 4 sone)",
)
def _chk_sharpness_table_a2() -> Outcome:
    # Independent (non-definitional) oracle: the hearing-test Sollwert for
    # the 2.5 kHz critical-band noise is 1.78 acum at the clause 6 loudness
    # of 4 sone; permitted deviation 5 % or 0.05 acum.
    from scipy import signal as sp_signal

    def narrowband(level_db: float) -> np.ndarray:
        rng = np.random.default_rng(7)
        white = rng.standard_normal(_FS * 2)
        sos = sp_signal.butter(
            8, [2320.0, 2700.0], btype="band", fs=_FS, output="sos"
        )
        nb = sp_signal.sosfilt(sos, white)
        return np.asarray(
            nb / np.sqrt(np.mean(nb**2)) * 2e-5 * 10 ** (level_db / 20)
        )

    lo, hi = 30.0, 90.0
    for _ in range(13):  # set the clause 6 loudness of 4 sone
        mid = (lo + hi) / 2
        n = ph.loudness_zwicker(narrowband(mid), _FS, stationary=True).loudness
        lo, hi = (mid, hi) if n < 4.0 else (lo, mid)
    computed = float(ph.sharpness_din(narrowband((lo + hi) / 2), _FS))
    return numeric(1.78, computed, 0.05 * 1.78, unit="acum", places=3)


@register(
    "Psychoacoustics",
    "ISO 226:2023 Table B.1",
    "Equal-loudness contour, 60 phon @ 100 Hz",
)
def _chk_iso226_contour() -> Outcome:
    # Anchored off 1 kHz (where SPL == phon is a trivial identity) at a
    # tabulated point that exercises the Table 1 contour formula.
    phon, freq, spl_ref = ref.ISO226_2023_TABLE_B1_ANCHOR
    freqs, spl = ph.equal_loudness_contour(phon)
    computed = float(spl[np.asarray(freqs) == freq][0])
    return numeric(spl_ref, computed, 0.05, unit="dB SPL", places=3)


# --- Block-A psychoacoustics: calibrated tones (pressure in pascals) --------
def _spl_tone(freq: float, level_db: float, seconds: float) -> np.ndarray:
    """Pure tone whose RMS is ``level_db`` dB re 20 uPa (pressure in Pa)."""
    t = np.arange(int(_FS * seconds)) / _FS
    amp = math.sqrt(2.0) * 2e-5 * 10.0 ** (level_db / 20.0)
    return np.asarray(amp * np.sin(2.0 * np.pi * freq * t))


def _am_tone(
    fc: float, fmod: float, depth: float, level_db: float, seconds: float
) -> np.ndarray:
    """Amplitude-modulated tone at an OVERALL RMS level (pressure in Pa).

    ECMA-418-2 Clause 7 states the calibration level as the sound pressure
    level of the signal, i.e. the overall RMS level of the modulated waveform.
    """
    t = np.arange(int(_FS * seconds)) / _FS
    x = (1.0 + depth * np.cos(2.0 * np.pi * fmod * t)) * np.sin(
        2.0 * np.pi * fc * t
    )
    return np.asarray(
        x * (2e-5 * 10.0 ** (level_db / 20.0)) / np.sqrt(np.mean(x**2))
    )


@register(
    "Psychoacoustics",
    "ECMA-418-2:2025 Clause 5.1.8",
    "HMS loudness of a 1 kHz / 40 dB tone (c_N=0.0211964)",
)
def _chk_ecma_loudness() -> Outcome:
    # With the full Clause 6.2.3 band averaging the chain computes 0.9845
    # sone_HMS for the calibration tone; the residual's origin is documented
    # in the loudness_ecma module docstring (c_N stays the verbatim value).
    computed = float(ph.loudness_ecma(_spl_tone(1000.0, 40.0, 0.6), _FS).loudness)
    return numeric(
        ref.ECMA418_2_LOUDNESS_1KHZ_40DB_SONE,
        computed,
        0.03,
        unit="sone_HMS",
        places=4,
    )


@register(
    "Psychoacoustics",
    "ECMA-418-2:2025 Clause 6.2.8",
    "HMS tonality of a 1 kHz / 40 dB tone (c_T=2.8758615)",
)
def _chk_ecma_tonality() -> Outcome:
    computed = float(ph.tonality_ecma(_spl_tone(1000.0, 40.0, 0.7), _FS).tonality)
    return numeric(
        ref.ECMA418_2_TONALITY_1KHZ_40DB_TU,
        computed,
        0.03,
        unit="tu_HMS",
        places=4,
    )


@register(
    "Psychoacoustics",
    "ECMA-418-2:2025 Clause 7",
    "HMS roughness of a 1 kHz / 70 Hz / m=1 / overall 60 dB tone (c_R=0.0180685)",
)
def _chk_ecma_roughness() -> Outcome:
    # Clause 7 calibration: the reference signal at an overall sound pressure
    # level of 60 dB SPL is 1 asper (computed: 0.9999 with the tabulated c_R).
    sig = _am_tone(1000.0, 70.0, 1.0, 60.0, 2.0)
    computed = float(ph.roughness_ecma(sig, _FS).roughness)
    return numeric(
        ref.ECMA418_2_ROUGHNESS_1KHZ_70HZ_60DB_ASPER,
        computed,
        0.01,
        unit="asper",
        places=4,
    )


@register(
    "Psychoacoustics",
    "ISO 532-2:2017 Clause 3.17 / Annex B.1",
    "Moore-Glasberg loudness of a 1 kHz / 40 dB tone (C=0.0617)",
)
def _chk_mg_loudness() -> Outcome:
    computed = float(
        ph.loudness_moore_glasberg_from_spectrum([(1000.0, 40.0)]).loudness
    )
    return numeric(
        ref.ISO532_2_ANCHOR_1KHZ_40DB_SONE, computed, 0.01, unit="sone", places=4
    )


@register(
    "Psychoacoustics",
    "ISO 532-3:2023 Annex C.1",
    "Moore-Glasberg-Schlittenlacher peak LTL, steady 1 kHz / 40 dB",
)
def _chk_mg_time_loudness() -> Outcome:
    fs = 32000.0
    t = np.arange(round(0.8 * fs)) / fs
    x = math.sqrt(2.0) * 2e-5 * 10.0 ** (40.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    computed = float(ph.loudness_moore_glasberg_time(x, fs).n_max)
    return numeric(
        ref.ISO532_3_ANCHOR_1KHZ_40DB_SONE, computed, 0.02, unit="sone", places=4
    )


@register(
    "Psychoacoustics",
    "ECMA-418-2:2025 Clause 9",
    "HMS fluctuation strength of a 1 kHz / 4 Hz / m=1 / overall 60 dB tone (c_F=0.003840572)",
)
def _chk_ecma_fluctuation_strength() -> Outcome:
    # Clause 9 calibration: the reference signal at an overall sound pressure
    # level of 60 dB SPL is 1 vacil_HMS (computed: 0.9931 for this 5 s signal
    # with the tabulated c_F; 0.9958 converged for longer signals).
    sig = _am_tone(1000.0, 4.0, 1.0, 60.0, 5.0)
    computed = float(ph.fluctuation_strength_ecma(sig, _FS).fluctuation_strength)
    return numeric(
        ref.ECMA418_2_FLUCTUATION_1KHZ_4HZ_60DB_VACIL,
        computed,
        0.01,
        unit="vacil_HMS",
        places=4,
    )


# ===========================================================================
# Domain 4 - Speech transmission (IEC 60268-16)
# ===========================================================================
_NUM_STI_BANDS = 7
_NUM_MOD_FREQS = 14


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 A.2.2",
    "STI weighting-factor pair (500 Hz + 1 kHz bands)",
)
def _chk_sti_weighting_pair() -> Outcome:
    mtf = np.zeros((_NUM_STI_BANDS, _NUM_MOD_FREQS))
    mtf[[2, 3], :] = 1.0
    computed = float(_sti_from_mtf(mtf).sti)
    return numeric(0.398, computed, 0.001, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 A.3.1.2",
    "Uniform MTF m=0.5 maps to STI=0.5",
)
def _chk_sti_uniform() -> Outcome:
    mtf = np.full((_NUM_STI_BANDS, _NUM_MOD_FREQS), 0.5)
    computed = float(_sti_from_mtf(mtf).sti)
    return numeric(0.5, computed, 0.01, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16 Annex M",
    "Full-STI worked example: printed MTF + speech/noise spectra -> STI",
)
def _chk_sti_annex_m() -> Outcome:
    """End-to-end A.5.3-A.5.6 chain on the annex's own worked example."""
    mtf = np.asarray(ref.IEC60268_16_ANNEX_M_MTF, dtype=float).T
    result = _sti_from_mtf(
        mtf,
        level=np.asarray(ref.IEC60268_16_ANNEX_M_LEVEL, dtype=float),
        ambient=np.asarray(ref.IEC60268_16_ANNEX_M_AMBIENT, dtype=float),
    )
    mti_delta = float(
        np.max(
            np.abs(
                np.round(result.mti, 2)
                - np.asarray(ref.IEC60268_16_ANNEX_M_MTI, dtype=float)
            )
        )
    )
    outcome = numeric(ref.IEC60268_16_ANNEX_M_STI, float(result.sti), 0.005, places=3)
    return Outcome(
        expected=f"STI {ref.IEC60268_16_ANNEX_M_STI} (MTI row of step 4c)",
        computed=f"STI {result.sti:.3f} (max MTI dev {mti_delta:.2f})",
        delta=outcome.delta,
        passed=outcome.passed and mti_delta <= 0.01,
    )


# Ed.5 STIPA verification-signal parameters (restating the standard, not
# the implementation): Table B.1 modulation pairs, A.6.1 male spectrum.
_STI_CENTERS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
_STI_F1 = [1.60, 1.00, 0.63, 2.00, 1.25, 0.80, 2.50]
_STI_F2 = [8.00, 5.00, 3.15, 10.00, 6.25, 4.00, 12.50]
_STI_LEVELS = [-2.5, 0.5, 0.0, -6.0, -12.0, -18.0, -24.0]
# m <-> STI staircase of the Ed.5 verification bench (m in 0,1 steps;
# TI = (10 lg(m/(1-m)) + 15)/30, rounded as published).
_STI_STAIRCASE = {0.0: 0.00, 0.1: 0.18, 0.2: 0.30, 0.3: 0.38, 0.4: 0.44,
                  0.5: 0.50, 0.6: 0.56, 0.7: 0.62, 0.8: 0.70, 0.9: 0.82,
                  1.0: 1.00}


def _stipa_sine_signal(
    m: float,
    seconds: float = 16.0,
    bands: tuple[int, ...] | None = None,
    edge_carriers: bool = False,
    flat_levels: bool = False,
) -> np.ndarray:
    """Ed.5 Formula (C.1)-style verification signal with sine carriers.

    A_k(t) = g_k c_k(t) sqrt(0,5 (1 + 0,55 m (sin 2 pi f1_k t -
    sin 2 pi f2_k t))). ``bands`` limits the modulation (m = 0 elsewhere,
    as in the A.2.2 pair signals); ``edge_carriers`` uses the two
    half-octave edge carriers fc 2^(+/-1/4) per band (A.3.1.2);
    ``flat_levels`` uses g_k = 1 as the A.2.2/A.3.1.2 bench does.
    """
    t = np.arange(round(seconds * _FS)) / _FS
    x = np.zeros(t.size)
    for k, (fc, fa, fb, level) in enumerate(
        zip(_STI_CENTERS, _STI_F1, _STI_F2, _STI_LEVELS)
    ):
        mk = m if bands is None or k in bands else 0.0
        env = 0.5 * (
            1.0 + 0.55 * mk * (np.sin(2 * np.pi * fa * t) - np.sin(2 * np.pi * fb * t))
        )
        if edge_carriers:
            carrier = np.sin(2 * np.pi * fc / 2**0.25 * t) + np.sin(
                2 * np.pi * fc * 2**0.25 * t
            )
        else:
            carrier = np.sin(2 * np.pi * fc * t)
        gain = 1.0 if flat_levels else 10.0 ** (level / 20.0)
        x += gain * carrier * np.sqrt(np.maximum(env, 0.0))
    return x


def _chk_stipa_staircase(m: float) -> Outcome:
    # Ed.5 modulation-depth verification: the Formula (C.1) signal with
    # sinusoidal carriers at the A.6.1 male levels and modulation scale m
    # must read the published staircase STI through the full stipa()
    # audio path (octave bank, envelopes, correlation, TI chain). The
    # certified stipa.info WAV bench (same construction) measures a worst
    # deviation of 0,0031 STI; tolerance +/-0,01.
    computed = float(ph.stipa(_stipa_sine_signal(m), _FS).sti)
    return numeric(_STI_STAIRCASE[m], computed, 0.01, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.3.2",
    "STIPA direct method, Formula (C.1) signal at m=0.2",
)
def _chk_stipa_staircase_m02() -> Outcome:
    return _chk_stipa_staircase(0.2)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.3.2",
    "STIPA direct method, Formula (C.1) signal at m=0.5",
)
def _chk_stipa_staircase_m05() -> Outcome:
    return _chk_stipa_staircase(0.5)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.3.2",
    "STIPA direct method, Formula (C.1) signal at m=0.8",
)
def _chk_stipa_staircase_m08() -> Outcome:
    return _chk_stipa_staircase(0.8)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.3.3",
    "Indirect method: exponential decay RT60=1 s vs Schroeder MTF",
)
def _chk_sti_indirect_expdecay() -> Outcome:
    # Ed.5 indirect-method verification: sine carriers in all seven bands
    # with an amplitude decay 1000^(-t/RT60) (60 dB intensity decay in
    # RT60) must reproduce the closed-form Schroeder MTF
    # m(F) = 1/sqrt(1 + (2 pi F RT60 / (6 ln 10))^2) through the octave
    # bank and Schroeder integral. Certified WAV bench worst deviation:
    # 0,0002 STI; tolerance +/-0,005.
    rt60 = 1.0
    t = np.arange(int(3.0 * _FS)) / _FS
    x = np.sum(
        [np.sin(2 * np.pi * fc * t) for fc in _STI_CENTERS], axis=0
    ) * 10.0 ** (-3.0 * t / rt60)
    # The 14 modulation frequencies 0,63 - 12,5 Hz (Ed.5 A.2.2).
    mod_freqs = np.array(
        [0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30,
         8.00, 10.0, 12.5]
    )
    m_formula = 1.0 / np.sqrt(
        1.0 + (2.0 * np.pi * mod_freqs * rt60 / (6.0 * np.log(10.0))) ** 2
    )
    expected = float(_sti_from_mtf(np.tile(m_formula, (_NUM_STI_BANDS, 1))).sti)
    computed = float(ph.sti_from_impulse_response(x, _FS).sti)
    return numeric(expected, computed, 0.005, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 C.4.2",
    "Filter-bank slope: +41 dB unmodulated tone one octave below 125 Hz",
)
def _chk_sti_filter_slope() -> Outcome:
    # Ed.5 filter-slope verification (worst case of the certified bench
    # pre-fix): a fully modulated 125 Hz carrier plus an unmodulated
    # 62,5 Hz tone 41 dB louder must still read m >= 0,5 in the 125 Hz
    # band, i.e. the analysis filter must be > 41 dB down one octave out
    # (steeper than IEC 61260-1 class 1; the zero-phase bank achieves
    # m >= 0,94 on all 14 certified signals).
    t = np.arange(int(16.0 * _FS)) / _FS
    env = 0.5 * (
        1.0 + 0.55 * (np.sin(2 * np.pi * 1.60 * t) - np.sin(2 * np.pi * 8.00 * t))
    )
    x = 10.0 ** (-41.0 / 20.0) * np.sin(2 * np.pi * 125.0 * t) * np.sqrt(
        np.maximum(env, 0.0)
    ) + np.sin(2 * np.pi * 62.5 * t)
    with warnings.catch_warnings():
        # Bands other than 125 Hz are legitimately (near-)empty here.
        warnings.simplefilter("ignore", UserWarning)
        m_observed = float(np.min(ph.stipa(x, _FS).mtf[0]))
    return Outcome(
        expected="m >= 0.5 (C.4.2 pass criterion)",
        computed=_fmt(m_observed, places=4),
        delta=_fmt(m_observed - 0.5, "", _DELTA_PLACES),
        passed=m_observed >= 0.5,
    )


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 A.2.2 (audio path)",
    "Weighting factors: modulated 500 Hz + 1 kHz pair through stipa()",
)
def _chk_sti_weighting_pair_audio() -> Outcome:
    # Ed.5 weighting-factor verification through the full audio path:
    # sine carriers in all seven bands, only the 500/1000 Hz pair
    # modulated (m = 1) -> STI = alpha_3 + alpha_4 - beta_3 = 0,398.
    # Certified WAV bench worst deviation vs the identity: 0,0002 STI.
    x = _stipa_sine_signal(1.0, bands=(2, 3), flat_levels=True)
    computed = float(ph.stipa(x, _FS).sti)
    return numeric(0.398, computed, 0.005, places=4)


@register(
    "Speech transmission (IEC 60268-16)",
    "IEC 60268-16:2020 A.3.1.2 (audio path)",
    "Filter-bank phase: half-octave edge carriers at TI=0.9",
)
def _chk_sti_edge_carriers() -> Outcome:
    # Ed.5 filter-bank phase-distortion verification: two sine carriers
    # per band at fc 2^(+/-1/4) (the extremes of the half-octave STIPA
    # generation band), modulation depth m = 0,94065 (TI = 0,9). The
    # normative criterion is |STI bias| < 0,01 over TI = 0,1 .. 0,9; the
    # zero-phase bank measures -0,0029 worst on the certified WAVs.
    x = _stipa_sine_signal(0.94065, edge_carriers=True, flat_levels=True)
    computed = float(ph.stipa(x, _FS).sti)
    return numeric(0.9, computed, 0.01, places=4)


# ===========================================================================
# System measurement (Golay pairs / regularized inversion / shaped sweeps)
# ===========================================================================
_SYSMEAS = "System measurement (Golay / Kirkeby / Mueller-Massarani)"


@register(
    _SYSMEAS,
    "Havelock 2008 Part I Ch. 6 (Xiang), Eq. (2)",
    "Golay pair: sum of periodic autocorrelations = 2L*delta (L = 4096)",
)
def _chk_golay_complementary() -> Outcome:
    a, b = ph.golay_pair(12)
    length = a.size
    acorr = np.fft.irfft(
        np.abs(np.fft.rfft(a)) ** 2 + np.abs(np.fft.rfft(b)) ** 2, length
    )
    peak_ok = abs(acorr[0] - 2.0 * length) <= 1e-8
    sidelobe = float(np.max(np.abs(acorr[1:])))
    if not peak_ok:
        return numeric(2.0 * length, float(acorr[0]), 1e-8, places=4)
    return numeric(0.0, sidelobe, 1e-10, places=6,
                   expected_label="0 (algebraic identity, +/-1e-10)")


@register(
    _SYSMEAS,
    "Havelock 2008 Part I Ch. 6 (Xiang), Eq. (4)",
    "Golay chain recovers a delay+gain system IR (noiseless, exact)",
)
def _chk_golay_recovery() -> Outcome:
    pair = ph.golay_pair(10)
    gain, delay = 0.75, 37
    res = ph.golay_impulse_response(
        gain * np.roll(pair[0], delay), gain * np.roll(pair[1], delay), pair
    )
    expected = np.zeros(pair[0].size)
    expected[delay] = gain
    err = float(np.max(np.abs(np.asarray(res) - expected)))
    return numeric(0.0, err, 1e-13, places=6,
                   expected_label="0 (machine precision, +/-1e-13)")


def _sysmeas_inverse() -> Any:
    b, a = sg.butter(2, [100.0, 8000.0], btype="bandpass", fs=float(_FS))
    imp = np.zeros(1024)
    imp[0] = 1.0
    return ph.regularized_inverse_filter(
        sg.lfilter(b, a, imp), float(_FS), f_range=(200.0, 4000.0)
    )


@register(
    _SYSMEAS,
    "Kirkeby & Nelson 1999 Eq. (17) / Mueller-Massarani 2001 Sec. 3.1",
    "In-band equalization residue equals eps/(|H|^2 + eps) bin by bin",
)
def _chk_kirkeby_in_band() -> Outcome:
    res = _sysmeas_inverse()
    band = (res.frequencies >= 200.0) & (res.frequencies <= 4000.0)
    product = np.abs(res.response_spectrum * res.spectrum)[band]
    power = np.abs(res.response_spectrum[band]) ** 2
    residue = res.regularization[band] / (power + res.regularization[band])
    err = float(np.max(np.abs((1.0 - product) - residue)))
    return numeric(0.0, err, 1e-12, places=6,
                   expected_label="0 (closed form, +/-1e-12)")


@register(
    _SYSMEAS,
    "Kirkeby & Nelson 1999 (max of x/(x^2+eps) = 1/(2*sqrt(eps)))",
    "Out-of-band inverse-filter gain within the regularization cap",
)
def _chk_kirkeby_out_of_band() -> Outcome:
    res = _sysmeas_inverse()
    cap_db = -20.0 * math.log10(2.0)  # eps_outside = 1.0 -> -6.021 dB
    headroom = cap_db - res.max_gain_db
    return Outcome(
        expected=f"<= {cap_db:.3f} dB (analytic cap)",
        computed=f"{res.max_gain_db:.3f} dB",
        delta=f"headroom {headroom:+.3f} dB",
        passed=headroom >= 0.0,
    )


@register(
    _SYSMEAS,
    "Mueller-Massarani 2001 Secs. 4.2-4.3 (group-delay synthesis)",
    "Shaped sweep's Welch spectrum follows the pink target, in-band",
)
def _chk_shaped_sweep_pink() -> Outcome:
    res = ph.shaped_sweep_signal(_FS, 50.0, 5000.0, 2.0, target="pink")
    nperseg = 8192
    freqs, psd = sg.welch(
        np.asarray(res), fs=float(_FS), nperseg=nperseg,
        noverlap=3 * nperseg // 4,
    )
    third = 2.0 ** (1.0 / 3.0)
    band = (freqs >= 50.0 * third) & (freqs <= 5000.0 / third)
    diff = 10.0 * np.log10(psd[band]) + 10.0 * np.log10(freqs[band] / 50.0)
    dev = float(np.max(np.abs(diff - np.median(diff))))
    return numeric(0.0, dev, 0.5, unit="dB", places=4,
                   expected_label="0 dB in-band deviation (+/-0.5 dB)")


# ===========================================================================
# Domain 5 - Intensity & sound power
# ===========================================================================
def _plane_wave_pair(
    delay_s: float, seconds: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n = int(_FS * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / _FS)
    spec = np.zeros(freqs.size, dtype=complex)
    band = (freqs >= 50.0) & (freqs <= 2000.0)
    spec[band] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, int(band.sum())))
    p1 = np.fft.irfft(spec, n)
    p2 = np.fft.irfft(spec * np.exp(-2j * np.pi * freqs * delay_s), n)
    scale = 1.0 / np.sqrt(np.mean(p1**2))
    return p1 * scale, p2 * scale


@register(
    "Intensity & sound power",
    "IEC 61043:1993 Clause 5",
    "Plane-wave intensity I = p^2 / (rho c)",
)
def _chk_plane_wave_intensity() -> Outcome:
    rho, c, spacing = 1.204, 343.0, 0.012
    p1, p2 = _plane_wave_pair(spacing / c)
    res = ph.sound_intensity(p1, p2, _FS, spacing, rho=rho, c=c)
    expected = float(np.mean(((p1 + p2) / 2.0) ** 2)) / (rho * c)
    return numeric(
        expected, float(res.total_intensity), 0.015, unit="W/m^2", rel=True, places=5
    )


@register(
    "Intensity & sound power",
    "ISO 3744:2010 Eq. 18",
    "Monopole hemisphere recovers LW (r=4 m)",
)
def _chk_monopole_lw() -> Outcome:
    lw_true, r = 95.0, 4.0
    lp = lw_true - 10.0 * math.log10(2.0 * math.pi * r**2)
    res = ph.sound_power_pressure(np.full((10, 1), lp), "hemisphere", radius=r)
    return numeric(lw_true, float(res.sound_power_level[0]), 1e-9, unit="dB", places=6)


@register(
    "Intensity & sound power",
    "ISO 9614-2:1996 Eq. 12",
    "Intensity scan recovers LW of an enclosed source",
)
def _chk_intensity_scan_lw() -> Outcome:
    w = 1.0e-3  # 90 dB re 1 pW
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = np.full((4, 1), w / areas.sum())
    res = ph.sound_power_intensity(intensity, areas)
    return numeric(90.0, float(res.sound_power_level[0]), 1e-6, unit="dB", places=6)


@register(
    "Intensity & sound power",
    "IEC 61043:1993 Table 2",
    "Minimum delta_pI0 per band, probe/processor/instrument, class 1/2",
)
def _chk_iec61043_table2() -> Outcome:
    """All 22 bands x 3 device kinds x 2 classes against the printed table."""
    columns = {"probe": (1, 2), "processor": (3, 4), "instrument": (5, 6)}
    worst = 0.0
    for device, (col1, col2) in columns.items():
        _, class1, class2 = ph.residual_index_limits(device)
        for i, row in enumerate(ref.IEC61043_TABLE2):
            worst = max(
                worst,
                abs(float(class1[i]) - row[col1]),
                abs(float(class2[i]) - row[col2]),
            )
    n = len(ref.IEC61043_TABLE2) * len(columns) * 2
    return Outcome(
        expected=f"{n} tabulated minima reproduced",
        computed=f"max absolute deviation {worst:.3f} dB",
        delta=_fmt(worst, "dB", _DELTA_PLACES),
        passed=worst == 0.0,
    )


@register(
    "Intensity & sound power",
    "IEC 61043:1993 Table 2 Note 1",
    "Separation rule +10 lg(x/25) on the 25 mm minima (x = 50 mm)",
)
def _chk_iec61043_spacing_rule() -> Outcome:
    _, base, _ = ph.residual_index_limits("instrument")
    _, wide, _ = ph.residual_index_limits("instrument", spacing=0.050)
    return numeric(
        10.0 * math.log10(2.0),
        float(np.max(np.abs(wide - base))),
        1e-12,
        unit="dB",
        places=6,
    )


@register(
    "Intensity & sound power",
    "Fahy, Sound Intensity 2e, 6.8",
    "delta_pI0 = 20 dB is a phase mismatch of 0.26 deg (1 kHz, 25 mm)",
)
def _chk_iec61043_phase_mismatch() -> Outcome:
    phi = ph.phase_mismatch_from_residual_index(
        ref.IEC61043_PHASE_INDEX_DB,
        ref.IEC61043_PHASE_FREQUENCY_HZ,
        ref.IEC61043_PHASE_SPACING_M,
    )
    return numeric(
        ref.IEC61043_PHASE_MISMATCH_DEG, float(phi), 0.005, unit="deg", places=4
    )


@register(
    "Intensity & sound power",
    "ISO 9614-1:1993 Eqs (A.1)/(A.2)",
    "Temporal variability F1 is the coefficient of variation of M samples",
)
def _chk_iso9614_1_f1() -> Outcome:
    samples = np.array([1.2e-5, 0.9e-5, 1.5e-5, 1.1e-5, 1.3e-5, 1.0e-5])
    # Equations (A.1)/(A.2) written out: sample standard deviation (M - 1
    # denominator) over the algebraic mean of the M short-time samples.
    mean = float(np.sum(samples) / samples.size)
    expected = (
        math.sqrt(float(np.sum((samples - mean) ** 2)) / (samples.size - 1)) / mean
    )
    return numeric(
        expected, float(ph.temporal_variability_indicator(samples)), 1e-12, places=6
    )


@register(
    "Intensity & sound power",
    "ISO 4871:1996 clause 3.15 / Annex B",
    "Declared L_WAd = L_WA + K_WA (Annex B, L_WA=88, K_WA=2)",
)
def _chk_iso4871_declared_value() -> Outcome:
    mode = ph.OperatingModeDeclaration("Operating mode 1", 88.0, 2.0)
    return numeric(90.0, float(mode.declared_sound_power_level), 0.0, unit="dB", places=1)


@register(
    "Intensity & sound power",
    "ISO 4871:1996 clause 6.2",
    "Single-machine verification boundary L_1 <= L_WAd",
)
def _chk_iso4871_verification() -> Outcome:
    at_boundary = ph.OperatingModeDeclaration("m", 88.0, 2.0, verification_level=90.0)
    just_over = ph.OperatingModeDeclaration("m", 88.0, 2.0, verification_level=91.0)
    ok = at_boundary.verified is True and just_over.verified is False
    return Outcome(
        expected="L_1=90 verified, L_1=91 rejected (L_WAd=90)",
        computed=f"90->{at_boundary.verified}, 91->{just_over.verified}",
        delta="boundary L_1 = L_WAd",
        passed=ok,
    )


def _reverb_bracket(
    t60: np.ndarray, volume: float, surface: float, freq: np.ndarray,
    theta: float, ps: float,
) -> np.ndarray:
    """Independent re-implementation of the ISO 3741 Eq. (20) bracket.

    The two constants below are deliberately different and mirror the library
    (:func:`phonometry.emission.sound_power_reverberation._speed_of_sound` and its C1/C2
    terms): the speed of sound uses the rounded 273 of ISO 3741 clause 9.1.4
    (``c = 20,05*sqrt(273 + theta)``), while the C1/C2 barometric corrections
    use the exact absolute-zero offset 273.15 in their temperature ratios.
    Matching the library keeps the "expected" bracket and the computed value on
    the same convention.
    """
    ps0, theta0, theta1 = 101.325, 314.0, 296.0
    c = 20.05 * math.sqrt(273.0 + theta)  # ISO 3741 clause 9.1.4 (rounded 273)
    a = (55.26 / c) * (volume / t60)
    waterhouse = 10.0 * np.log10(1.0 + surface * c / (8.0 * volume * freq))
    # C1/C2 use 273.15 (absolute temperature ratios), as in the library.
    c1 = -10.0 * math.log10(ps / ps0) + 5.0 * math.log10((273.15 + theta) / theta0)
    c2 = -10.0 * math.log10(ps / ps0) + 15.0 * math.log10((273.15 + theta) / theta1)
    return 10.0 * np.log10(a) + 4.34 * (a / surface) + waterhouse + c1 + c2 - 6.0


@register(
    "Intensity & sound power",
    "ISO 3741:2010 Eq. 20",
    "Reverberation-room method inverts to a known LW",
)
def _chk_reverberation_lw() -> Outcome:
    volume, surface = 200.0, 210.0
    freqs = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
    t60 = np.array([2.0, 1.8, 1.5, 1.0, 0.6])
    theta, ps = 23.0, 101.325
    lw_target = np.array([80.0, 85.0, 90.0, 82.0, 75.0])
    lp = lw_target - _reverb_bracket(t60, volume, surface, freqs, theta, ps)
    res = ph.sound_power_reverberation(
        lp, t60, volume, surface, freqs, temperature=theta, static_pressure=ps
    )
    worst = float(np.max(np.abs(np.asarray(res.sound_power_level) - lw_target)))
    return numeric(0.0, worst, 1e-9, unit="dB", places=9, expected_label="0 dB error")


# ===========================================================================
# Domain 6 - Room & building acoustics
# ===========================================================================
_A60 = 6.0 * math.log(10.0)


def _exponential_ir(t60: float, seconds: float) -> np.ndarray:
    t = np.arange(round(seconds * _FS)) / _FS
    return np.asarray(np.exp(-0.5 * _A60 * t / t60))


@register(
    "Room & building acoustics",
    "ISO 3382-2:2008 5.3.3",
    "T30 from a synthetic exponential decay (T=1.0 s)",
)
def _chk_room_t30() -> Outcome:
    t60 = 1.0
    res = ph.room_parameters(_exponential_ir(t60, 3.0 * t60), _FS, limits=None)
    return numeric(t60, float(res.t30[0]), 0.01, unit="s", rel=True, places=4)


@register(
    "Room & building acoustics",
    "ISO 18233:2006 (swept-sine method)",
    "Sweep deconvolution recovers a known IIR response",
)
def _chk_iso18233_sweep_deconvolution() -> Outcome:
    # Closed-form identity: an exponential sweep through a known Butterworth
    # band-pass, deconvolved back, must reproduce the filter's freqz response.
    b, a = sg.butter(4, [200.0, 2000.0], btype="band", fs=_FS)
    x = ph.sweep_signal(_FS, 20.0, 20000.0, 2.0)
    y = sg.lfilter(b, a, x)
    ir = np.asarray(ph.impulse_response(y, x, _FS, length=16384))
    freqs = np.fft.rfftfreq(ir.size, d=1.0 / _FS)
    h_est = np.fft.rfft(ir)
    _, h_true = sg.freqz(b, a, worN=freqs, fs=_FS)
    mask = (freqs >= 300.0) & (freqs <= 1500.0)
    worst = float(np.max(np.abs(
        20.0 * np.log10(np.abs(h_est[mask])) - 20.0 * np.log10(np.abs(h_true[mask]))
    )))
    # Linear deconvolution is exact in-band up to windowing/regularisation
    # leakage; 0.1 dB is the demonstrated in-band bound (tests/test_room_ir.py)
    # with the same 300-1500 Hz evaluation band, well inside the sweep edges.
    return numeric(0.0, worst, 0.1, unit="dB", places=4,
                   expected_label="0 dB in-band error (+/-0.1 dB)")


@register(
    "Room & building acoustics",
    "ISO 717-1 Annex C, Table C.1",
    "Weighted sound reduction index Rw (C;Ctr)",
)
def _chk_iso717_rw() -> Outcome:
    exp = ref.ISO717_1_ANNEX_C_EXPECTED
    res = ph.weighted_rating(ref.ISO717_1_ANNEX_C_R)
    ok = res.rating == exp["rw"] and res.c == exp["c"] and res.ctr == exp["ctr"]
    return Outcome(
        expected=f"Rw {exp['rw']} (C {exp['c']}; Ctr {exp['ctr']})",
        computed=f"Rw {res.rating} (C {res.c}; Ctr {res.ctr})",
        delta=f"sum {res.unfavourable_sum:.1f} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 717-1:2020 Annex C, Table C.2",
    "Enlarged range 50-5000 Hz: Rw (C; Ctr; C50-5000; Ctr,50-5000)",
)
def _chk_iso717_1_extended() -> Outcome:
    exp = ref.ISO717_1_ANNEX_C2_EXPECTED
    freqs = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
             630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
    res = ph.weighted_rating_extended(ref.ISO717_1_ANNEX_C2_R_50_5000, freqs)
    ok = (
        res.rating == exp["rw"] and res.c == exp["c"] and res.ctr == exp["ctr"]
        and res.c_50_5000 == exp["c_50_5000"]
        and res.ctr_50_5000 == exp["ctr_50_5000"]
    )
    return Outcome(
        expected=(
            f"Rw {exp['rw']} (C {exp['c']}; Ctr {exp['ctr']}; "
            f"C50-5000 {exp['c_50_5000']}; Ctr,50-5000 {exp['ctr_50_5000']})"
        ),
        computed=(
            f"Rw {res.rating:g} (C {res.c:g}; Ctr {res.ctr:g}; "
            f"C50-5000 {res.c_50_5000:g}; Ctr,50-5000 {res.ctr_50_5000:g})"
        ),
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 717-2 Annex C, Table C.1",
    "Weighted impact sound pressure level Ln,w (CI)",
)
def _chk_iso717_2_lnw() -> Outcome:
    # Worked example: Ln,w = 79 dB, CI = -11 dB, unfavourable sum 28,0 dB.
    # CI = -11 is the ISO 717-2:2013 Annex C print; the 2020 reprint of this
    # example is internally inconsistent with its own A.2.1 (it sums the
    # 3 150 Hz band into Ln,sum and prints CI = -10).
    # Integer ratings and CI must match exactly; the unfavourable sum is a
    # one-decimal tabulated intermediate, so 1e-9 = exact up to float noise.
    exp = ref.ISO717_2_ANNEX_C1_EXPECTED
    res = ph.weighted_impact_rating(ref.ISO717_2_ANNEX_C1_LN)
    sum_ok = abs(res.unfavourable_sum - exp["unfavourable_sum"]) <= 1e-9
    ok = res.rating == exp["ln_w"] and res.ci == exp["ci"] and sum_ok
    return Outcome(
        expected=f"Ln,w {exp['ln_w']} (CI {exp['ci']}; sum {exp['unfavourable_sum']:.1f} dB)",
        computed=f"Ln,w {res.rating} (CI {res.ci}; sum {res.unfavourable_sum:.1f} dB)",
        delta=f"{res.rating - exp['ln_w']:+d} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 717-2 Annex C, Table C.1 (covered)",
    "Weighted impact level of the floor WITH covering Ln,w (CI)",
)
def _chk_iso717_2_lnw_covered() -> Outcome:
    exp = ref.ISO717_2_ANNEX_C1_COVERED_EXPECTED
    res = ph.weighted_impact_rating(ref.ISO717_2_ANNEX_C1_COVERED_LN)
    sum_ok = abs(res.unfavourable_sum - exp["unfavourable_sum"]) <= 1e-9
    ok = res.rating == exp["ln_w"] and res.ci == exp["ci"] and sum_ok
    return Outcome(
        expected=f"Ln,w {exp['ln_w']} (CI {exp['ci']}; sum {exp['unfavourable_sum']:.1f} dB)",
        computed=f"Ln,w {res.rating} (CI {res.ci}; sum {res.unfavourable_sum:.1f} dB)",
        delta=f"{res.rating - exp['ln_w']:+d} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 717-2 Annex C, Table C.2",
    "Floor-covering improvement ΔLw and CI,Δ (Formulae (2)/(A.4); CI,Δ from"
    " the normative Table 4 floor, not the 2020 print's misprinted C.2 chain)",
)
def _chk_iso717_2_c2_improvement() -> Outcome:
    dlw = ph.weighted_impact_improvement(ref.ISO717_2_ANNEX_C2_DELTA_L)
    ci_d = ph.impact_improvement_adaptation_term(ref.ISO717_2_ANNEX_C2_DELTA_L)
    ok = (
        dlw == ref.ISO717_2_ANNEX_C2_DELTA_LW
        and ci_d == ref.ISO717_2_ANNEX_C2_CI_DELTA
    )
    return Outcome(
        expected=(
            f"ΔLw {ref.ISO717_2_ANNEX_C2_DELTA_LW} dB; "
            f"CI,Δ {ref.ISO717_2_ANNEX_C2_CI_DELTA} dB (Table 4 reference floor)"
        ),
        computed=f"ΔLw {dlw} dB; CI,Δ {ci_d} dB",
        delta=f"{dlw - ref.ISO717_2_ANNEX_C2_DELTA_LW:+d} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 354:2003 Eq. 5/8",
    "Sabine inversion recovers absorption area",
)
def _chk_iso354_absorption() -> Outcome:
    v, c, t = 200.0, 343.0, 3.5
    expected = 55.3 * v / (c * t)
    computed = float(np.asarray(ph.absorption_area(t, v, speed_of_sound=c))[()])
    return numeric(expected, computed, 1e-9, unit="m^2", places=6)


@register(
    "Room & building acoustics",
    "ISO 3382-3:2012 Clause 6.2",
    "Open-plan spatial decay rate D2,S (-6 dB/doubling)",
)
def _chk_open_plan_d2s() -> Outcome:
    r = np.array([2.0, 4.0, 8.0, 16.0])
    lp = 70.0 - 6.0 * np.log2(r)
    sti = 0.6 - 0.02 * r
    res = ph.open_plan_metrics(r, lp, sti)
    return numeric(6.0, float(res.d2s), 1e-9, unit="dB", places=6)


@register(
    "Room & building acoustics",
    "ISO 16283-3:2016 Clause 3.12",
    "Facade R'45 isolates the -1.5 dB incidence correction (S=A)",
)
def _chk_facade_r45() -> Outcome:
    # With S = A the 10 lg(S/A) coupling term vanishes, so R' = L1,s - L2 - 1,5.
    n = 3
    res = ph.facade_insulation(
        np.full(n, 55.0),
        np.full(n, ref.ISO16283_3_R45_RECEIVE_LEVEL_DB),
        np.full(n, ref.ISO16283_3_R45_REVERB_TIME_S),
        area=ref.ISO16283_3_R45_AREA_M2,
        volume=ref.ISO16283_3_R45_VOLUME_M3,
        surface_level=np.full(n, ref.ISO16283_3_R45_SURFACE_LEVEL_DB),
    )
    assert res.r_prime is not None
    # The expected value is rebuilt from its components, so the -1,5 dB
    # oblique-incidence correction constant is exercised explicitly.
    expected = (
        ref.ISO16283_3_R45_SURFACE_LEVEL_DB
        - ref.ISO16283_3_R45_RECEIVE_LEVEL_DB
        - ref.ISO16283_3_R45_LOUDSPEAKER_CORRECTION_DB
    )
    assert expected == ref.ISO16283_3_R45_EXPECTED_DB
    computed = float(np.asarray(res.r_prime)[0])
    return numeric(expected, computed, 1e-9, unit="dB", places=6)


@register(
    "Room & building acoustics",
    "ISO 10140-2:2010 Formula (2)",
    "Lab airborne R on the ISO 717-1 reference shape -> Rw = 54",
)
def _chk_lab_airborne_rw() -> Outcome:
    # S = A (A = 0,16*50/0,8 = 10 = area) => R = L1 - L2 = the reference curve.
    ref_r = np.asarray(ref.ISO10140_2_REF_AIRBORNE_R, dtype=float)
    res = ph.lab_airborne_insulation(
        np.full(16, 90.0), 90.0 - ref_r, np.full(16, 0.8), area=10.0, volume=50.0
    )
    assert res.rating is not None
    # R lands exactly on the reference; guard that before reading the rating.
    on_curve = bool(np.allclose(np.asarray(res.r), ref_r))
    expected = ref.ISO10140_2_REF_AIRBORNE_RW
    return Outcome(
        expected=f"Rw {expected} dB",
        computed=f"Rw {res.rating.rating} dB",
        delta=f"{res.rating.rating - expected:+d} dB",
        passed=on_curve and res.rating.rating == expected,
    )


@register(
    "Room & building acoustics",
    "ISO 10140-5:2010+A1 Annex B, Table B.1",
    "Reference elements end-to-end: printed Rw (C; Ctr) of all three",
)
def _chk_iso10140_5_reference_elements() -> Outcome:
    rows = [
        (ref.ISO10140_5_B1_HEAVY_WALL_R, ref.ISO10140_5_B1_HEAVY_WALL_RATING),
        (ref.ISO10140_5_B1_HEAVY_FLOOR_R, ref.ISO10140_5_B1_HEAVY_FLOOR_RATING),
        (ref.ISO10140_5_B1_LIGHT_WALL_R, ref.ISO10140_5_B1_LIGHT_WALL_RATING),
    ]
    computed = []
    ok = True
    for r, expected in rows:
        # S = A (10 m2) so the ISO 10140-2 chain returns R = L1 - L2 exactly.
        res = ph.lab_airborne_insulation(
            np.full(16, 90.0), 90.0 - np.asarray(r, dtype=float),
            np.full(16, 0.8), area=10.0, volume=50.0,
        )
        assert res.rating is not None
        got = (res.rating.rating, res.rating.c, res.rating.ctr)
        computed.append(got)
        ok = ok and got == expected
    return Outcome(
        expected="Rw(C;Ctr) = 53(-1;-5) / 52(-1;-5) / 33(-1;-2)",
        computed=" / ".join(f"{rw}({c};{ctr})" for rw, c, ctr in computed),
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 10140-5:2010+A1 Annex C, Table C.1",
    "Reference floors end-to-end: printed Ln,t,r,0,w (CI) of both",
)
def _chk_iso10140_5_reference_floors() -> Outcome:
    rows = [
        (ref.ISO10140_5_C1_FLOOR_C1C2_LN, ref.ISO10140_5_C1_FLOOR_C1C2_RATING),
        (ref.ISO10140_5_C1_FLOOR_C3_LN, ref.ISO10140_5_C1_FLOOR_C3_RATING),
    ]
    computed = []
    ok = True
    for ln, expected in rows:
        # A = A0 (V = 31,25 m3, T = 0,5 s) so Ln equals the receiving level.
        res = ph.lab_impact_insulation(
            np.asarray(ln, dtype=float), np.full(16, 0.5), volume=31.25
        )
        assert res.rating is not None
        got = (res.rating.rating, res.rating.ci)
        computed.append(got)
        ok = ok and got == expected
    return Outcome(
        expected="Ln,t,r,0,w(CI) = 72(0) / 75(-3)",
        computed=" / ".join(f"{lnw}({ci})" for lnw, ci in computed),
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 15186-1:2000 Formula (7)",
    "Intensity RI on the ISO 717-1 reference shape -> RI,w = 30",
)
def _chk_intensity_ri_rw() -> Outcome:
    # Hand-computed scalar anchor pinning the Formula (7) constants (the
    # curve construction below inverts the same formula, so -6 dB and
    # 10 lg(Sm/S) would cancel there): Lp1 = 80, LIn = 40, Sm = S
    # -> RI = 80 - 6 - 40 - 0 = 34 dB exactly.
    scalar = ph.intensity_sound_reduction(
        [80.0], [40.0], measurement_area=10.0, area=10.0
    )
    scalar_ok = abs(float(scalar.r_i[0]) - 34.0) <= 1e-9
    # Choose LIn so that RI = Lp1 - 6 - [LIn + 10 lg(Sm/S)] lands exactly on
    # the ISO 717-1 Annex C curve; the rating engine must then return Rw = 30.
    ref_ri = np.asarray(ref.ISO15186_1_REF_RI, dtype=float)
    lp1, sm, s = ref.ISO15186_1_REF_LP1, ref.ISO15186_1_REF_SM, ref.ISO15186_1_REF_S
    lin = lp1 - 6.0 - 10.0 * np.log10(sm / s) - ref_ri
    res = ph.intensity_sound_reduction(
        np.full(16, lp1), lin, measurement_area=sm, area=s
    )
    assert res.rating is not None
    on_curve = bool(np.allclose(np.asarray(res.r_i), ref_ri))
    expected = ref.ISO15186_1_REF_RIW
    return Outcome(
        expected=f"RI,w {expected} dB (scalar anchor RI = 34 dB)",
        computed=f"RI,w {res.rating.rating} dB (RI = {float(scalar.r_i[0]):g} dB)",
        delta=f"{res.rating.rating - expected:+d} dB",
        passed=scalar_ok and on_curve and res.rating.rating == expected,
    )


@register(
    "Room & building acoustics",
    "ISO 15186-1:2000 Annex B, Table B.1",
    "Adaptation term Kc: all 21 printed rows; (B.1) reduces to (B.2)",
)
def _chk_intensity_kc_annexb() -> Outcome:
    # The printed Table B.1 (21 one-third-octave rows, one decimal) is the
    # independent oracle; additionally Formula (B.1) with Sb2 = 117 m²,
    # V2 = 81 m³, c = 340 m/s must reduce to (B.2) Kc = 10 lg(1 + 61,4/f).
    b2 = ph.adaptation_term_kc(ref.ISO15186_1_KC_BANDS)
    b1 = ph.adaptation_term_kc(
        ref.ISO15186_1_KC_BANDS, boundary_area=117.0, volume=81.0
    )
    printed = np.asarray(ref.ISO15186_1_KC_B1_PRINTED, dtype=float)
    worst = float(np.max(np.abs(b2 - printed)))
    delta = float(np.max(np.abs(b1 - b2)))
    passed = worst <= 0.05 and delta <= 1e-3
    return Outcome(
        expected="max abs(Kc - Table B.1) <= 0,05 dB (1 dp print)",
        computed=f"{worst:.3f} dB (B.1 vs B.2: {delta:.2e} dB)",
        delta=f"{worst:.3f} dB",
        passed=passed,
    )


@register(
    "Room & building acoustics",
    "ISO 10052:2021 Clause 3.6",
    "Survey R' applies the V/7,5 minimum-area rule",
)
def _chk_survey_rprime_area_rule() -> Outcome:
    # V/7,5 = 120/7,5 = 16 m^2 > S = 5 m^2, so the larger value replaces S.
    # With k = 0 (T = T0), R' = D + 10 lg(16 * T0 / (0,16 V)).
    v, s, d = 120.0, 5.0, 30.0
    res = ph.survey_airborne_insulation(
        np.full(5, 70.0), np.full(5, 40.0), np.zeros(5), volume=v, area=s
    )
    assert res.r_prime is not None
    s_eff = v / 7.5
    expected = d + 10.0 * np.log10(s_eff * 0.5 / (0.16 * v))
    return numeric(expected, float(res.r_prime[0]), 1e-9, unit="dB", places=6)


@register(
    "Room & building acoustics",
    "ISO 10052:2021 Clause 3.16",
    "Service-equipment LXY is the 3-position energy average",
)
def _chk_survey_service_equipment() -> Outcome:
    # Energy average of 35 / 30 / 32 dB(A), then standardized by k.
    levels = [35.0, 30.0, 32.0]
    res = ph.survey_service_equipment_level(levels, 3.0, volume=50.0)
    expected = 10.0 * np.log10(sum(10.0 ** (0.1 * x) for x in levels) / 3.0)
    return numeric(expected, float(np.asarray(res.l_xy)[()]), 1e-9, unit="dB", places=6)


@register(
    "Room & building acoustics",
    "ISO 10052:2021 Table 4",
    "Reverberation-index estimate (35 <= V < 60, type g)",
)
def _chk_survey_reverberation_estimate() -> Outcome:
    # Table 4 row 'g' for 35 <= V < 60: 4,5 / 5 / 5,5 / 5,5 / 5,5 dB.
    expected = [4.5, 5.0, 5.5, 5.5, 5.5]
    got = np.asarray(ph.estimate_reverberation_index(50.0, "g"), dtype=float)
    ok = bool(np.array_equal(got, expected))
    return Outcome(
        expected=f"k = {expected} dB",
        computed=f"k = {got.tolist()} dB",
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 717-2:2020 Table 4 / Clause 5.2",
    "Reference-floor weighted level Ln,r,0,w and CI (ISO 16251-1 ΔLw anchor)",
)
def _chk_iso717_2_reference_floor() -> Outcome:
    res = ph.weighted_impact_rating(ref.ISO717_2_REFERENCE_FLOOR_LN_R0)
    ok = (
        res.rating == ref.ISO717_2_REFERENCE_FLOOR_LN_R0_W
        and res.ci == ref.ISO717_2_REFERENCE_FLOOR_CI
    )
    return Outcome(
        expected=f"Ln,r,0,w = {ref.ISO717_2_REFERENCE_FLOOR_LN_R0_W} dB, "
        f"CI = {ref.ISO717_2_REFERENCE_FLOOR_CI} dB",
        computed=f"Ln,r,0,w = {res.rating} dB, CI = {res.ci} dB",
        delta="exact",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "ISO 16251-1:2014 / ISO 717-2 Formula (2)",
    "Floor-covering ΔLw: zero improvement gives ΔLw = 0",
)
def _chk_iso16251_zero_improvement() -> Outcome:
    import numpy as _np

    dlw = ph.weighted_impact_improvement(_np.zeros(16))
    return Outcome(
        expected="ΔLw = 0 dB (ΔL = 0 -> Ln,r = Ln,r,0)",
        computed=f"ΔLw = {dlw} dB",
        delta="exact",
        passed=(dlw == 0),
    )


@register(
    "Room & building acoustics",
    "ISO 16251-1 / ISO 717-2 (Foret et al. 2011, carpet)",
    "Measured textile-carpet improvement rates to ΔLw = 29 dB",
)
def _chk_iso16251_carpet_foret2011() -> Outcome:
    # Real measurement: the Delta-L spectrum digitized from Figure 4 (vector
    # chart) of Foret, Chene & Guigou-Carter, Forum Acusticum 2011, run through
    # the full ISO 16251-1 improvement path (a flat bare-plate level minus the
    # measured Delta-L reconstructs the specimen level), then rated per ISO
    # 717-2 on the 100-3150 Hz sub-range. See tests/reference_data.py for the
    # provenance and the +/- 0,5 dB digitization tolerance.
    bare = np.full(len(ref.FORET2011_CARPET_FREQ), 100.0)
    delta_l = np.asarray(ref.FORET2011_CARPET_ISO16251_DELTA_L, dtype=float)
    res = ph.impact_improvement(bare, bare - delta_l, ref.FORET2011_CARPET_FREQ)
    dlw = res.delta_lw
    return Outcome(
        expected=f"ΔLw = {ref.FORET2011_CARPET_ISO16251_DELTA_LW} dB (paper, ISO 16251-1)",
        computed=f"ΔLw = {dlw} dB",
        delta=f"{(dlw or 0) - ref.FORET2011_CARPET_ISO16251_DELTA_LW:+d} dB",
        passed=(dlw == ref.FORET2011_CARPET_ISO16251_DELTA_LW),
    )


@register(
    "Room & building acoustics",
    "ISO 10848-1:2006 Formula (14)",
    "Flanking Kij (simplified) matches closed form",
)
def _chk_iso10848_kij_simplified() -> Outcome:
    # No worked example in the standard; anchor on the closed form recomputed
    # independently here (delta is "exact" to keep the report byte-stable).
    res = ph.vibration_reduction_index(
        [ref.ISO10848_KIJ_DBAR],
        ref.ISO10848_KIJ_LIJ,
        ref.ISO10848_KIJ_AREA,
        ref.ISO10848_KIJ_AREA,
    )
    computed = float(res.k_ij[0])
    expected = ref.ISO10848_KIJ_DBAR + 10.0 * math.log10(
        ref.ISO10848_KIJ_LIJ / math.sqrt(ref.ISO10848_KIJ_AREA**2)
    )
    return Outcome(
        expected=f"Kij = {expected:.4f} dB",
        computed=f"Kij = {computed:.4f} dB",
        delta="exact",
        passed=abs(computed - expected) < 1e-9,
    )


@register(
    "Room & building acoustics",
    "ISO 10848-1:2006 Formula (12)",
    "Flanking equivalent absorption length aj at f_ref",
)
def _chk_iso10848_absorption_length() -> Outcome:
    a = ph.equivalent_absorption_length(
        ref.ISO10848_ABS_AREA,
        ref.ISO10848_ABS_TS,
        [1000.0],
        speed_of_sound=ref.ISO10848_ABS_C0,
    )
    computed = float(a[0])
    # aj at f = f_ref (sqrt(f_ref/f) = 1): aj = 2,2·π²·S/(Ts·c0).
    expected = (
        2.2 * math.pi**2 * ref.ISO10848_ABS_AREA
        / (ref.ISO10848_ABS_TS * ref.ISO10848_ABS_C0)
    )
    return Outcome(
        expected=f"aj = {expected:.4f} m",
        computed=f"aj = {computed:.4f} m",
        delta="exact",
        passed=abs(computed - expected) < 1e-9,
    )


@register(
    "Room & building acoustics",
    "ISO 10848-1:2006 Clause 7.3.1",
    "Flanking total loss factor η = 2,2/(f·Ts)",
)
def _chk_iso10848_loss_factor() -> Outcome:
    eta = ph.total_loss_factor([1000.0], [0.5])
    computed = float(eta[0])
    expected = 2.2 / (1000.0 * 0.5)
    return Outcome(
        expected=f"η = {expected:.4f}",
        computed=f"η = {computed:.4f}",
        delta="exact",
        passed=abs(computed - expected) < 1e-12,
    )


@register(
    "Room & building acoustics",
    "ISO 12354-1:2017 Formula (20) vs Hopkins Eq. 2.201 (6 mm glass)",
    "Flanking critical frequency (c0²/1,8·cL·h) vs plate coincidence "
    "(c0²/2π · sqrt(m''/B'))",
)
def _chk_flanking_critical_frequency() -> Outcome:
    # The 1,8 constant rounds 2π/√12, so for a plate whose bending stiffness
    # and mass are mutually consistent the two independent formulas must
    # agree to within that rounding (< 1 %).
    e, rho, nu, h, c0 = 6.2e10, 2500.0, 0.24, 0.006, 343.0
    c_l = math.sqrt(e / (rho * (1.0 - nu**2)))
    fc_flank = ph.critical_frequency(c_l, h, speed_of_sound=c0)
    fc_coinc = ph.coincidence_frequency(
        rho * h, ph.plate_bending_stiffness(e, h, nu), speed_of_sound=c0
    )
    return numeric(fc_coinc, fc_flank, 0.01, rel=True, unit="Hz", places=1)


# --- Dynamic stiffness of resilient materials (EN 29052-1:1992) ---
@register(
    "Room & building acoustics",
    "EN 29052-1:1992 Formula 4",
    "Apparent dynamic stiffness s't = 4π²·m't·fr²  (m't=200 kg/m², fr=25 Hz)",
)
def _chk_en29052_apparent() -> Outcome:
    computed = float(ph.apparent_dynamic_stiffness(25.0, 200.0)) / 1e6
    expected = 4.0 * math.pi**2 * 200.0 * 25.0**2 / 1e6
    return numeric(expected, computed, 1e-6, unit="MN/m³", places=6)


@register(
    "Room & building acoustics",
    "EN 29052-1:1992 clause 8.2 NOTE",
    "Enclosed-gas stiffness s'a·d = 111 MN·mm/m³ (p₀=0,1 MPa, ε=0,9)",
)
def _chk_en29052_enclosed_gas() -> Outcome:
    # NOTE: s'a = 111/d MN/m3 for d in mm; the closed form gives 100/0,9 = 111.11.
    sa_mn = float(ph.enclosed_gas_stiffness(0.020, 0.9)) / 1e6   # d = 20 mm
    return numeric(111.111111 / 20.0, sa_mn, 1e-4, unit="MN/m³", places=5)


@register(
    "Room & building acoustics",
    "EN 29052-1:1992 Formula 2",
    "Floating-floor natural frequency f0 = (1/2π)√(s'/m')  (s'=10 MN/m³, m'=100 kg/m²)",
)
def _chk_en29052_resonance() -> Outcome:
    computed = float(ph.natural_frequency(10.0e6, 100.0))
    expected = math.sqrt(10.0e6 / 100.0) / (2.0 * math.pi)
    return numeric(expected, computed, 1e-6, unit="Hz", places=5)


# --- Mechanical mobility (ISO 7626-1:2011) ---
# Closed-form SDOF resonator (consistent with the ISO 7626-1 Table 1 / 3.1.2
# FRF definitions): m=2 kg, k=8000 N/m, c=5 N.s/m; f0 = sqrt(k/m)/(2pi).
_MOB_M, _MOB_K, _MOB_C = 2.0, 8000.0, 5.0
_MOB_F0 = math.sqrt(_MOB_K / _MOB_M) / (2.0 * math.pi)


@register(
    "Room & building acoustics",
    "ISO 7626-1:2011 Table 1 / 3.1.2",
    "Closed-form SDOF driving-point mobility peak mag(Y(f0)) = 1/c  (c=5 N·s/m)",
)
def _chk_iso7626_mobility_peak() -> Outcome:
    y0 = complex(ph.sdof_mobility(_MOB_F0, _MOB_M, _MOB_K, _MOB_C))
    return numeric(1.0 / _MOB_C, abs(y0), 1e-6, unit="m/(N·s)", places=6)


@register(
    "Room & building acoustics",
    "ISO 7626-1:2011 Table 1 / 3.1.2",
    "Closed-form SDOF static receptance H(0) = 1/k  (k=8000 N/m)",
)
def _chk_iso7626_static_receptance() -> Outcome:
    h = complex(ph.sdof_receptance(1e-6, _MOB_M, _MOB_K, _MOB_C))
    return numeric(1.0 / _MOB_K, h.real, 1e-6, unit="m/N", rel=True, places=8)


@register(
    "Room & building acoustics",
    "ISO 7626-1:2011 Table 1",
    "FRF reciprocity: impedance × mobility = 1  (at 37 Hz)",
)
def _chk_iso7626_reciprocity() -> Outcome:
    y = complex(ph.sdof_mobility(37.0, _MOB_M, _MOB_K, _MOB_C))
    z = complex(ph.convert_frf(y, 37.0, "mobility", "impedance"))
    return numeric(1.0, abs(z * y), 1e-9, expected_label="1 (= Z·Y)")


# --- Dynamic transfer stiffness of resilient elements (ISO 10846) ---
@register(
    "Room & building acoustics",
    "ISO 10846-2:2008 3.17",
    "Transfer-stiffness level Lk = 20 lg(|k|/k0), k0 = 1 N/m  (|k| = 1 MN/m)",
)
def _chk_iso10846_level() -> Outcome:
    lk = float(ph.transfer_stiffness_level(1.0e6))
    return numeric(120.0, lk, 1e-6, unit="dB")


@register(
    "Room & building acoustics",
    "ISO 10846-3:2002 Formula (1)",
    "Indirect method k2,1 = -(2πf)²·m2·T  (f=500 Hz, m2=10 kg, T=0,01)",
)
def _chk_iso10846_indirect() -> Outcome:
    f, m2, t = 500.0, 10.0, 0.01
    expected = -((2.0 * math.pi * f) ** 2) * m2 * t
    computed = complex(ph.transfer_stiffness_indirect(f, t + 0j, m2)).real
    return numeric(expected, computed, 1e-3, rel=True, unit="N/m", places=1)


@register(
    "Room & building acoustics",
    "ISO 10846-1:2008 Table A.2",
    "FRF relation k = jω·Z at 250 Hz  (|k| recovered from impedance)",
)
def _chk_iso10846_stiffness_impedance() -> Outcome:
    f = 250.0
    w = 2.0 * math.pi * f
    k = 1.0e6 + 1j * 5.0e4
    z = complex(ph.convert_frf(k, f, "dynamic_stiffness", "impedance"))
    return numeric(abs(k), abs(1j * w * z), 1e-6, rel=True, unit="N/m", places=1)


@register(
    "Room & building acoustics",
    "ISO 7626-2:2015 7.5.2",
    "Rigid-mass calibration: accelerance mag(A) = 1/m  (m=10 kg)",
)
def _chk_iso7626_2_rigid_mass_accelerance() -> Outcome:
    res = ph.rigid_mass_calibration_check(
        [ref.ISO7626_2_CAL_ACCELERANCE], [100.0], ref.ISO7626_2_CAL_MASS_KG
    )
    value = float(res.expected[0]) if res.passed else math.nan
    return numeric(ref.ISO7626_2_CAL_ACCELERANCE, value, 1e-9, unit="1/kg", places=3)


@register(
    "Room & building acoustics",
    "ISO 7626-2:2015 7.5.2",
    "Rigid-mass calibration: mobility mag(Y) = 1/(2πf·m) at 100 Hz  (m=10 kg)",
)
def _chk_iso7626_2_rigid_mass_mobility() -> Outcome:
    res = ph.rigid_mass_calibration_check(
        [ref.ISO7626_2_CAL_MOBILITY_100HZ],
        [100.0],
        ref.ISO7626_2_CAL_MASS_KG,
        quantity="mobility",
    )
    value = float(res.expected[0]) if res.passed else math.nan
    return numeric(
        ref.ISO7626_2_CAL_MOBILITY_100HZ, value, 1e-5, rel=True,
        unit="m/(N·s)", places=7,
    )


@register(
    "Room & building acoustics",
    "ISO 7626-2:2015 Annex A",
    "Normalized random error ε = √((1−γ²)/(2nγ²)): γ²=0,8, n=75 → 4,08 % (< 5 %)",
)
def _chk_iso7626_2_random_error() -> Outcome:
    eps = float(ph.random_error_percent(0.8, 75))
    return numeric(ref.ISO7626_2_RANDOM_ERROR_PCT, eps, 0.005, unit="%", places=2)


@register(
    "Room & building acoustics",
    "ISO 7626-1:2011 Table 1",
    "Rigid 1 kg mass at ω = 1000 rad/s: mobility 1e-3, compliance 1e-6 (decades)",
)
def _chk_iso7626_decade_identity() -> Outcome:
    f = ref.ISO7626_1_DECADE_FREQ_HZ
    y = abs(complex(ph.convert_frf(1.0, f, "apparent_mass", "mobility")))
    h = abs(complex(ph.convert_frf(1.0, f, "apparent_mass", "receptance")))
    ok = abs(h - ref.ISO7626_1_DECADE_COMPLIANCE) <= 1e-15
    return numeric(
        ref.ISO7626_1_DECADE_MOBILITY, y if ok else math.nan, 1e-9, rel=True,
        unit="m/(N·s)", places=4,
    )


@register(
    "Room & building acoustics",
    "ISO 10846-3:2002 6.1 Inequality (2)",
    "Indirect-method validity limit mag(T) = 0,1 ↔ ΔL1,2 = 20 dB",
)
def _chk_iso10846_3_validity_threshold() -> Outcome:
    delta_l = 20.0 * math.log10(1.0 / ph.TRANSMISSIBILITY_LIMIT)
    return numeric(
        ref.ISO10846_3_LIMIT_DELTA_L_DB, delta_l, 1e-9, unit="dB", places=1
    )


@register(
    "Room & building acoustics",
    "ISO 10846-3:2002 6.1",
    "Model bias at the validity limit: k_ind/k = 1,1 (0,83 dB ≤ 1 dB, 10 % ≤ 12 %)",
)
def _chk_iso10846_3_validity_bias() -> Outcome:
    # Undamped mass-spring model at omega^2 m = 11 k, i.e. T = -0,1 exactly.
    k, m = 1.0e6, 1.0
    f = math.sqrt(11.0 * k / m) / (2.0 * math.pi)
    t = complex(ph.base_transmissibility(f, m, k))
    k_ind = abs(complex(ph.transfer_stiffness_indirect(f, t, m)))
    ratio = k_ind / k
    bias_ok = (
        20.0 * math.log10(ratio) <= ref.ISO10846_3_ACCURACY_DB
        and ratio - 1.0 <= ref.ISO10846_3_ACCURACY_FRACTION
    )
    return numeric(
        ref.ISO10846_3_LIMIT_BIAS_RATIO, ratio if bias_ok else math.nan,
        1e-9, rel=True, places=4,
    )


@register(
    "Room & building acoustics",
    "ISO 10846-1:2008 Equation (6)",
    "Delivered/blocking force F2/F2,b = 1/1,1 at mag(k2,2/kt) = 0,1 (within 10 %)",
)
def _chk_iso10846_1_blocking_force() -> Outcome:
    value = abs(complex(ph.blocking_force_ratio(1.0e5, 1.0e6)))
    return numeric(ref.ISO10846_1_EQ6_FORCE_RATIO, value, 1e-9, places=4)


@register(
    "Room & building acoustics",
    "ISO 10846-2:2008 / -3:2002 7.6",
    "Linearity: ΔLk ≤ 1,5 dB for input spectra 10 dB apart (linear element: 0)",
)
def _chk_iso10846_linearity() -> Outcome:
    k, u_a = 1.0e6 + 3.0e4j, 1.0e-6 + 0j
    u_b = u_a * 10.0 ** (-ref.ISO10846_LINEARITY_STEP_DB / 20.0)
    lk_a = float(ph.transfer_stiffness_level(ph.transfer_stiffness_direct(k * u_a, u_a)))
    lk_b = float(ph.transfer_stiffness_level(ph.transfer_stiffness_direct(k * u_b, u_b)))
    return numeric(
        0.0, abs(lk_a - lk_b), ref.ISO10846_LINEARITY_TOL_DB, unit="dB", places=3,
        expected_label="ΔLk ≤ 1,5 dB (7.6 c)",
    )


# --- Sound power from surface vibration (ISO/TS 7849-1/-2) ---
@register(
    "Room & building acoustics",
    "ISO/TS 7849-1:2009 Formula (8)",
    "Calibration L_v from â = 9,81 m/s² at 100 Hz  (standard's EXAMPLE)",
)
def _chk_iso7849_calibration() -> Outcome:
    lv = float(ph.velocity_level_from_acceleration(9.81, 100.0))
    return numeric(106.9, lv, 0.05, unit="dB", places=1)


@register(
    "Room & building acoustics",
    "ISO/TS 7849-2:2009 Formula (15)",
    "L_W from L_v via measured radiation factor = 10 lg(P/P0)  (round-trip)",
)
def _chk_iso7849_power_round_trip() -> Outcome:
    p, s, v2 = 3.0e-4, 2.0, (1.0e-3) ** 2
    eps = float(ph.radiation_factor(p, s, v2))
    lv = float(ph.velocity_level(math.sqrt(v2)))
    lw = float(ph.radiated_sound_power_level(lv, s, radiation_factor=eps))
    return numeric(10.0 * math.log10(p / 1e-12), lw, 1e-6, unit="dB", places=3)


@register(
    "Room & building acoustics",
    "ISO/TS 7849-1:2009 Formula (12)",
    "Impedance term: L_W − L_v = 10 lg(411/400) at ε = 1, S = S0",
)
def _chk_iso7849_impedance_term() -> Outcome:
    lw = float(ph.radiated_sound_power_level(80.0, 1.0))
    return numeric(10.0 * math.log10(411.0 / 400.0), lw - 80.0, 1e-9, unit="dB")


# --- Structure-borne sound power of building equipment (EN 15657) ---
@register(
    "Room & building acoustics",
    "EN 15657:2018 Formula (14)",
    "Reception-plate L_Ws = resonant-plate power P = ωη(mS)⟨v²⟩  (round-trip)",
)
def _chk_en15657_power_balance() -> Outcome:
    lv, f, m, s, eta = 82.0, 800.0, 15.0, 1.5, 0.02
    lw = float(ph.structure_borne_power_level(lv, f, m, s, eta))
    v2 = (1e-9) ** 2 * 10.0 ** (0.1 * lv)
    p = 2.0 * math.pi * f * eta * (m * s) * v2
    return numeric(10.0 * math.log10(p / 1e-12), lw, 1e-6, unit="dB", places=3)


@register(
    "Room & building acoustics",
    "EN 15657:2018 Formula (13)",
    "Plate loss factor η = 2,2/(f·Ts) at 1 kHz, Ts = 0,3 s",
)
def _chk_en15657_loss_factor() -> Outcome:
    eta = float(ph.plate_loss_factor([1000.0], 0.3)[0])
    return numeric(2.2 / (1000.0 * 0.3), eta, 1e-9)


@register(
    "Room & building acoustics",
    "EN 15657:2018 Formulae (15)/(17) + EN 12354-5 Annex I.3",
    "Source conversion chain reproduces Table I.8 (wall, installed)",
)
def _chk_en15657_conversion_chain() -> Outcome:
    # Measured plate power (Y_plate = 5,34e-6) -> blocked force (15) ->
    # characteristic reception-plate level (17, Y_R,inf,low = 5e-6) ->
    # Annex I mobility correction to the wall (Y_wall = 24,1e-6). The printed
    # Table I.8 row is the oracle (one-decimal intermediates, +/-0,15 dB).
    lwsn = ph.characteristic_reception_plate_power(
        ph.equivalent_blocked_force_level(
            ref.EN12354_5_I8_WALL_LWS, ref.EN12354_5_I8_PLATE_MOBILITY
        )
    )
    installed = ph.installed_power_from_reception_plate(
        lwsn, ref.EN12354_5_I8_Y_WALL
    )
    worst = float(np.max(np.abs(
        np.asarray(installed) - np.asarray(ref.EN12354_5_I8_WALL_INSTALLED)
    )))
    return numeric(0.0, worst, ref.EN12354_5_ANNEX_I_TOL, unit="dB", places=3,
                   expected_label="max abs(L_Ws,inst - Table I.8) <= 0,15 dB")


@register(
    "Room & building acoustics",
    "ISO 9611:1996 eq. (9)",
    "Mean free velocity level (energy mean, v0 = 5e-8 m/s)",
)
def _chk_iso9611_mean_velocity() -> Outcome:
    computed = float(ph.mean_free_velocity_level(ref.ISO9611_MEAN_LEVELS))
    return numeric(ref.ISO9611_MEAN_EXPECTED, computed, 1e-9, unit="dB", places=4)


# --- Installed structure-borne sound from equipment (EN 12354-5) ---
@register(
    "Room & building acoustics",
    "EN 12354-5:2009 Formula (19b/19c)",
    "Coupling term → force-source limit 10 lg(mag(Ys)/Re{Yi}) as mag(Ys) ≫ mag(Yi)",
)
def _chk_en12354_5_coupling_limit() -> Outcome:
    ys, yi = 1e-3 + 0j, 1e-7 + 0j
    dc = float(ph.coupling_term(ys, yi))
    limit = float(ph.coupling_term_force_source(ys, yi))
    return numeric(limit, dc, 1e-2, unit="dB", places=3)


@register(
    "Room & building acoustics",
    "EN 12354-5:2009 Annex I.3, Table I.9",
    "Flushing cistern: four paths + Formula (17) total -> 29 dB(A)",
)
def _chk_en12354_5_annex_i9() -> Outcome:
    # The standard's own end-to-end worked example (replaces the former
    # formula-restatement checks of Formulae (18a)/(18b), which could not
    # catch a mistranscribed constant): both power components through
    # D_C (Table I.9), Formula (18a) per path and the energetic total.
    tol = ref.EN12354_5_ANNEX_I_TOL
    inst_wall = ph.installed_structure_borne_power_level(
        ref.EN12354_5_I8_WALL_LWSC, ref.EN12354_5_I9_DC_WALL
    )
    inst_floor = ph.installed_structure_borne_power_level(
        ref.EN12354_5_I8_FLOOR_LWSC, ref.EN12354_5_I9_DC_FLOOR
    )
    paths = [
        (inst_wall, ref.EN12354_5_I9_DSA_WALL, ref.EN12354_5_I9_R_WALL_FLOOR,
         ref.EN12354_5_I9_S_WALL, ref.EN12354_5_I9_LNS_WALL_FLOOR),
        (inst_wall, ref.EN12354_5_I9_DSA_WALL, ref.EN12354_5_I9_R_WALL_WALL,
         ref.EN12354_5_I9_S_WALL, ref.EN12354_5_I9_LNS_WALL_WALL),
        (inst_floor, ref.EN12354_5_I9_DSA_FLOOR, ref.EN12354_5_I9_R_FLOOR_FLOOR,
         ref.EN12354_5_I9_S_FLOOR, ref.EN12354_5_I9_LNS_FLOOR_FLOOR),
        (inst_floor, ref.EN12354_5_I9_DSA_FLOOR, ref.EN12354_5_I9_R_FLOOR_WALL,
         ref.EN12354_5_I9_S_FLOOR, ref.EN12354_5_I9_LNS_FLOOR_WALL),
    ]
    worst = 0.0
    rows = []
    for inst, dsa, rij, s_i, expected in paths:
        lns = ph.structure_borne_pressure_level_path(inst, dsa, rij, s_i)
        worst = max(worst, float(np.max(np.abs(lns - np.asarray(expected)))))
        rows.append(np.asarray(lns))
    total = ph.total_structure_borne_pressure_level(np.vstack(rows))
    worst = max(worst, float(np.max(np.abs(
        total - np.asarray(ref.EN12354_5_I9_LNS_TOTAL)
    ))))
    a_weights = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2])
    lns_a = float(10.0 * np.log10(np.sum(10.0 ** (0.1 * (total + a_weights)))))
    ok = worst <= tol and round(lns_a) == ref.EN12354_5_I9_LNS_TOTAL_A
    return Outcome(
        expected=f"max path/total dev <= {tol} dB; total "
        f"{ref.EN12354_5_I9_LNS_TOTAL_A} dB(A)",
        computed=f"{worst:.3f} dB; {lns_a:.1f} dB(A)",
        delta=f"{worst:.3f} dB",
        passed=ok,
    )


@register(
    "Room & building acoustics",
    "EN 12354-5:2009 Annex I.2, Table I.6a",
    "Whirlpool floor component: mobility correction + path 11",
)
def _chk_en12354_5_annex_i6a() -> Outcome:
    tol = ref.EN12354_5_ANNEX_I_TOL
    inst = ph.installed_power_from_reception_plate(
        ref.EN12354_5_I6A_LWSN_FLOOR, ref.EN12354_5_I6A_Y_FLOOR
    )
    dev_inst = float(np.max(np.abs(
        np.asarray(inst) - np.asarray(ref.EN12354_5_I6A_LWSN_INST_FLOOR)
    )))
    lns = ph.structure_borne_pressure_level_path(
        inst, ref.EN12354_5_I6A_DSA_FLOOR, ref.EN12354_5_I6A_R11, 10.0
    )
    dev_path = float(np.max(np.abs(
        np.asarray(lns) - np.asarray(ref.EN12354_5_I6A_LNS_11)
    )))
    worst = max(dev_inst, dev_path)
    return numeric(0.0, worst, tol, unit="dB", places=3,
                   expected_label="max abs(dev vs Table I.6a) <= 0,15 dB")


# ===========================================================================
# Domain 7 - Building prediction & uncertainty
# ===========================================================================
def _annex_h3_paths() -> list[ph.FlankingPath]:
    """The EN 12354-1 Annex H.3 flanking paths from the shared input table."""
    ss = ref.EN12354_1_ANNEX_H3_SEPARATING_AREA
    paths: list[ph.FlankingPath] = []
    for label, rw, kff, kfd, lf in ref.EN12354_1_ANNEX_H3_ELEMENTS:
        ff, df, fd = ph.flanking_element(
            label=label, r_flanking=rw, r_separating=ref.EN12354_1_ANNEX_H3_R_DIRECT,
            k_ff=kff, k_fd=kfd, k_df=kfd, separating_area=ss, coupling_length=lf,
        )
        paths += [ff, df, fd]
    return paths


@register(
    "Building prediction & uncertainty",
    "EN 12354-1:2000 Annex H.3",
    "Airborne prediction R'w (direct + 12 flanking paths)",
)
def _chk_en12354_1_airborne() -> Outcome:
    res = ph.predicted_airborne_insulation(
        r_direct=ref.EN12354_1_ANNEX_H3_R_DIRECT, flanking_paths=_annex_h3_paths()
    )
    expected = ref.EN12354_1_ANNEX_H3_RPRIME_W
    computed = float(res.r_prime_w)
    paths_ok = len(res.paths) == ref.EN12354_1_ANNEX_H3_NUM_PATHS
    return Outcome(
        expected=f"R'w {expected} dB ({ref.EN12354_1_ANNEX_H3_NUM_PATHS} paths)",
        computed=f"R'w {round(computed)} dB ({len(res.paths)} paths, {computed:.2f})",
        delta=f"{computed - expected:+.2f} dB",
        passed=paths_ok and round(computed) == expected,
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-1:2000 Annex H.3 (paths)",
    "All 12 printed flanking-path values Rij,w",
)
def _chk_en12354_1_h3_paths() -> Outcome:
    res = ph.predicted_airborne_insulation(
        r_direct=ref.EN12354_1_ANNEX_H3_R_DIRECT, flanking_paths=_annex_h3_paths()
    )
    by_label = {p.label: p.r_w for p in res.paths}
    worst = 0.0
    for element, (r_ff, r_cross) in ref.EN12354_1_ANNEX_H3_PATH_RW.items():
        for suffix, expected in (("Ff", r_ff), ("Fd", r_cross), ("Df", r_cross)):
            worst = max(worst, abs(by_label[f"{element}-{suffix}"] - expected))
    return numeric(0.0, worst, 0.05, unit="dB", places=3,
                   expected_label="max abs(Rij,w - printed) <= 0,05 dB")


@register(
    "Building prediction & uncertainty",
    "EN 12354-1:2000 Formula (5b) / Annex H.3",
    "DnT,w closure from R'w (both H.3 examples -> 54 dB)",
)
def _chk_en12354_1_dnt_closure() -> Outcome:
    v = ref.EN12354_1_ANNEX_H3_VOLUME
    ss = ref.EN12354_1_ANNEX_H3_SEPARATING_AREA
    first = float(ph.standardized_level_difference(52.2, v, ss))
    second = float(ph.standardized_level_difference(52.7, v, ss))
    ok = (
        round(first) == ref.EN12354_1_ANNEX_H3_DNT_W
        and round(second) == ref.EN12354_1_ANNEX_H3_DNT_W_SECOND
        # The printed 53,8 dB uses the standard's own V/(3 S) rounding of the
        # exact 0,32 V/Ss factor (0,18 dB apart).
        and abs(first - ref.EN12354_1_ANNEX_H3_DNT_W_PRINTED) <= 0.2
    )
    return Outcome(
        expected=f"DnT,w {ref.EN12354_1_ANNEX_H3_DNT_W} dB (printed 53,8/54,3)",
        computed=f"DnT,w {first:.2f} / {second:.2f} dB",
        delta=f"{first - ref.EN12354_1_ANNEX_H3_DNT_W_PRINTED:+.2f} dB vs printed",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-2:2000 Annex E.3",
    "Impact prediction L'n,w = Ln,w,eq - dLw + K",
)
def _chk_en12354_2_impact() -> Outcome:
    ln_eq = ph.equivalent_impact_level(ref.EN12354_2_ANNEX_E3_MASS)
    k = ph.impact_flanking_correction(
        ref.EN12354_2_ANNEX_E3_MASS, ref.EN12354_2_ANNEX_E3_FLANKING_MEAN_MASS
    )
    res = ph.predicted_impact_insulation(
        ln_w_eq=round(ln_eq), delta_l_w=ref.EN12354_2_ANNEX_E3_DELTA_LW,
        k_correction=k,
    )
    k_ok = int(k) == ref.EN12354_2_ANNEX_E3_K
    computed = float(res.l_prime_n_w)
    out = numeric(
        ref.EN12354_2_ANNEX_E3_LPRIME_N_W, computed, 1e-9, unit="dB", places=6
    )
    return Outcome(out.expected, out.computed, out.delta, out.passed and k_ok)


@register(
    "Building prediction & uncertainty",
    "EN 12354-2:2000 Formula (3) / Annex E.3",
    "Standardized impact level L'nT,w (exact 0,032 V form -> 43 dB)",
)
def _chk_en12354_2_standardized() -> Outcome:
    # Exact Formula (3): 45 - 10 lg(0,032 x 50) = 42,96 dB. The E.3 chain's own
    # "10 lg(V/30)" rounding gives 42,8 dB; both round to 43 dB.
    lnt = float(ph.standardized_impact_level(45.0, 50.0))
    ok = round(lnt) == 43 and abs(lnt - 42.96) <= 0.01
    return Outcome(
        expected="L'nT,w 43 dB (exact 42,96; E.3 prints 42,8)",
        computed=f"L'nT,w {lnt:.2f} dB",
        delta=f"{lnt - 42.96:+.3f} dB",
        passed=ok,
    )


def _en12354_3_annex_f() -> ph.FacadePredictionResult:
    """The EN 12354-3 Annex F facade prediction from the shared input table."""
    elements = [
        ph.FacadeElement(name=name, area=area, r=r)
        for name, area, r in ref.EN12354_3_ANNEX_F_ELEMENTS
    ]
    elements.append(ph.FacadeElement(name="inlet", dn_e=ref.EN12354_3_ANNEX_F_INLET_DNE))
    return ph.facade_sound_reduction(
        elements,
        area=ref.EN12354_3_ANNEX_F_AREA,
        volume=ref.EN12354_3_ANNEX_F_VOLUME,
        frequencies=ref.EN12354_3_ANNEX_F_BANDS,
        bands="octave",
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-3:2000 Annex F",
    "Facade airborne prediction (R'tr,s,w / D2m,nT,w single numbers)",
)
def _chk_en12354_3_facade() -> Outcome:
    res = _en12354_3_annex_f()
    # Anchor on the digit-exact low bands and the single-number ratings.
    low_ok = np.allclose(
        np.asarray(res.r_prime)[:3], ref.EN12354_3_ANNEX_F_RPRIME_LOW, atol=0.05
    )
    nums_ok = (
        res.r_tr_s_w == ref.EN12354_3_ANNEX_F_RTRS_W
        and res.c_tr == ref.EN12354_3_ANNEX_F_CTR
        and res.d_2m_nt_w == ref.EN12354_3_ANNEX_F_D2MNT_W
    )
    return Outcome(
        expected=(
            f"R'tr,s,w {ref.EN12354_3_ANNEX_F_RTRS_W} "
            f"(Ctr {ref.EN12354_3_ANNEX_F_CTR}); D2m,nT,w {ref.EN12354_3_ANNEX_F_D2MNT_W} dB"
        ),
        computed=f"R'tr,s,w {res.r_tr_s_w} (Ctr {res.c_tr}); D2m,nT,w {res.d_2m_nt_w} dB",
        delta="0",
        passed=bool(low_ok and nums_ok),
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-4:2000 Annex G / Formula (2)",
    "Radiated LW of a wall+door segment (side 1, low bands)",
)
def _chk_en12354_4_radiated() -> Outcome:
    res = ph.radiated_sound_power(
        [
            ph.FacadeElement(
                name="wall",
                area=ref.EN12354_4_ANNEX_G_SEGMENT_AREA - ref.EN12354_4_ANNEX_G_DOOR_AREA,
                r=ref.EN12354_4_ANNEX_G_CONCRETE_R,
            ),
            ph.FacadeElement(
                name="door", area=ref.EN12354_4_ANNEX_G_DOOR_AREA,
                r=ref.EN12354_4_ANNEX_G_DOOR_R,
            ),
        ],
        lp_in=ref.EN12354_4_ANNEX_G_LP_IN,
        area=ref.EN12354_4_ANNEX_G_SEGMENT_AREA,
        c_d=ref.EN12354_4_ANNEX_G_CD,
        r_prime_cap=ref.EN12354_4_ANNEX_G_RPRIME_CAP,
        octave_bands=[int(f) for f in ref.EN12354_4_ANNEX_G_BANDS],
    )
    rp_ok = np.allclose(
        np.asarray(res.r_prime)[:3], ref.EN12354_4_ANNEX_G_SIDE1_RPRIME_LOW, atol=0.05
    )
    lw = np.asarray(res.l_w)[:2]
    exp = np.asarray(ref.EN12354_4_ANNEX_G_SIDE1_LW_LOW)
    out = numeric(0.0, float(np.max(np.abs(lw - exp))), 0.1, unit="dB", places=3)
    return Outcome(
        expected=f"LW 63/125 Hz {ref.EN12354_4_ANNEX_G_SIDE1_LW_LOW} dB (+/-0.1)",
        computed=f"LW {np.round(lw, 1).tolist()} dB",
        delta=out.delta,
        passed=bool(rp_ok and out.passed),
    )


@register(
    "Building prediction & uncertainty",
    "EN 12354-4:2000 Annex E / Table G.9",
    "Exterior level of all four Table G.9 reception cells",
)
def _chk_en12354_4_propagation() -> Outcome:
    # (width, height, distance, printed A'tot, side LWA, printed Lp).
    cells = [
        (*ref.EN12354_4_ANNEX_G_ATTENUATION[0],
         ref.EN12354_4_ANNEX_G_SIDE1_LWA, ref.EN12354_4_ANNEX_G_LP_SIDE1_D5),
        (*ref.EN12354_4_ANNEX_G_ATTENUATION[1],
         ref.EN12354_4_ANNEX_G_SIDE1_LWA, ref.EN12354_4_ANNEX_G_LP_SIDE1_D25),
        (*ref.EN12354_4_ANNEX_G_ATTENUATION[2],
         ref.EN12354_4_ANNEX_G_SIDE4_LWA, ref.EN12354_4_ANNEX_G_LP_SIDE4_D5),
        (*ref.EN12354_4_ANNEX_G_ATTENUATION[3],
         ref.EN12354_4_ANNEX_G_SIDE4_LWA, ref.EN12354_4_ANNEX_G_LP_SIDE4_D25),
    ]
    worst = 0.0
    computed_lp = []
    for w, h, d, a_tot, lwa, lp_expected in cells:
        att = float(ph.outdoor_attenuation(w, h, d))
        lp = float(ph.outdoor_level(lwa, att))
        worst = max(worst, abs(att - a_tot), abs(lp - lp_expected))
        computed_lp.append(lp)
    return Outcome(
        expected="Lp 36,6 / 28,5 / 44,6 / 37,3 dB (+/-0,05)",
        computed="Lp " + " / ".join(f"{v:.1f}" for v in computed_lp) + " dB",
        delta=f"{worst:.3f} dB",
        passed=worst <= 0.05,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-1:2020 Table 2",
    "Airborne band uncertainty, situation A @ 1 kHz",
)
def _chk_iso12999_table2_band() -> Outcome:
    res = ph.band_uncertainty("airborne", "A")
    idx = list(res.frequencies).index(1000)
    computed = float(res.uncertainties[idx])
    return numeric(
        ref.ISO12999_1_TABLE2_AIRBORNE_A_1000HZ, computed, 1e-9, unit="dB", places=3
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-1:2020 Annex B, Table B.2",
    "One-decimal single numbers Rw / Rw+C50-5000 / Rw+Ctr,50-5000",
)
def _chk_iso12999_annex_b_values() -> Outcome:
    res = ph.weighted_rating_extended(
        ref.ISO12999_1_ANNEX_B_RI, ref.ISO12999_1_ANNEX_B_FREQ,
        one_decimal=True,
    )
    assert res.c_50_5000 is not None and res.ctr_50_5000 is not None
    rw = float(res.rating)
    rw_c = rw + float(res.c_50_5000)
    rw_ctr = rw + float(res.ctr_50_5000)
    ok = (
        abs(rw - ref.ISO12999_1_ANNEX_B_RW) <= 1e-9
        and abs(rw_c - ref.ISO12999_1_ANNEX_B_RW_C50_5000) <= 1e-9
        and abs(rw_ctr - ref.ISO12999_1_ANNEX_B_RW_CTR50_5000) <= 1e-9
    )
    return Outcome(
        expected=(
            f"{ref.ISO12999_1_ANNEX_B_RW} / {ref.ISO12999_1_ANNEX_B_RW_C50_5000}"
            f" / {ref.ISO12999_1_ANNEX_B_RW_CTR50_5000} dB"
        ),
        computed=f"{rw:.1f} / {rw_c:.1f} / {rw_ctr:.1f} dB",
        delta=f"{rw - ref.ISO12999_1_ANNEX_B_RW:+.2f} dB",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-1:2020 Annex B, Formulae (B.2)/(B.6)",
    "Single-number uncertainties (uncorrelated 0,6/0,8; correlated u(Rw) 1,9)",
)
def _chk_iso12999_annex_b_uncertainties() -> Outcome:
    from phonometry.building.insulation import (
        _SPECTRUM1_50_5000,
        _SPECTRUM2_50_5000,
    )

    ri = np.asarray(ref.ISO12999_1_ANNEX_B_RI, dtype=float)
    ui = np.asarray(ref.ISO12999_1_ANNEX_B_UI, dtype=float)
    u_c = float(ph.single_number_uncertainty_uncorrelated(
        ui, np.asarray(_SPECTRUM1_50_5000, dtype=float) - ri
    ))
    u_ctr = float(ph.single_number_uncertainty_uncorrelated(
        ui, np.asarray(_SPECTRUM2_50_5000, dtype=float) - ri
    ))
    up = ph.weighted_rating_extended(
        ri + ui, ref.ISO12999_1_ANNEX_B_FREQ, one_decimal=True
    ).rating
    down = ph.weighted_rating_extended(
        ri - ui, ref.ISO12999_1_ANNEX_B_FREQ, one_decimal=True
    ).rating
    u_rw = (float(up) - float(down)) / 2.0
    ok = (
        round(u_c, 1) == ref.ISO12999_1_ANNEX_B_U_UNCORR_C
        and round(u_ctr, 1) == ref.ISO12999_1_ANNEX_B_U_UNCORR_CTR
        and abs(u_rw - ref.ISO12999_1_ANNEX_B_U_CORR_RW) <= 1e-9
    )
    return Outcome(
        expected=(
            f"u_uncorr {ref.ISO12999_1_ANNEX_B_U_UNCORR_C} / "
            f"{ref.ISO12999_1_ANNEX_B_U_UNCORR_CTR} dB; "
            f"u_corr(Rw) {ref.ISO12999_1_ANNEX_B_U_CORR_RW} dB"
        ),
        computed=f"{u_c:.2f} / {u_ctr:.2f} dB; {u_rw:.2f} dB",
        delta=f"{u_rw - ref.ISO12999_1_ANNEX_B_U_CORR_RW:+.2f} dB",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-1:2020 Clause 8 / Table 8",
    "Expanded uncertainty U = 1.96 u (95 % two-sided, Rw sit. A)",
)
def _chk_iso12999_expanded() -> Outcome:
    u = ref.ISO12999_1_RW_A_STANDARD_UNCERTAINTY
    expected = ref.ISO12999_1_COVERAGE_K_95 * u
    computed = float(ph.insulation_expanded_uncertainty(u, coverage=0.95))
    return numeric(expected, computed, 1e-9, unit="dB", places=6)


@register(
    "Building prediction & uncertainty",
    "ISO 12999-2:2020 Table 4 / Formula (1)",
    "Absorption coefficient +/-U (k=2), reproducibility, 20 x 1/3-oct bands",
)
def _chk_iso12999_2_table4() -> Outcome:
    res = ph.sound_absorption_coefficient_uncertainty(
        ref.ISO12999_2_TABLE4_ALPHA_S, ref.ISO12999_2_TABLE4_FREQ, confidence=0.95
    )
    got = res.reported_expanded_uncertainty
    expected = np.asarray(ref.ISO12999_2_TABLE4_U_K2, dtype=float)
    ok = bool(np.array_equal(got, expected))
    return Outcome(
        expected=f"U(k=2) = {expected.tolist()}",
        computed=f"U(k=2) = {got.tolist()}",
        delta="exact",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-2:2020 Table 5 / Formula (4)",
    "Practical coefficient +/-U (k=2), reproducibility, 5 octave bands",
)
def _chk_iso12999_2_table5() -> Outcome:
    res = ph.practical_coefficient_uncertainty(
        ref.ISO12999_2_TABLE5_ALPHA_P, ref.ISO12999_2_TABLE5_FREQ
    )
    got = res.reported_expanded_uncertainty
    expected = np.asarray(ref.ISO12999_2_TABLE5_U_K2, dtype=float)
    ok = bool(np.array_equal(got, expected))
    return Outcome(
        expected=f"U(k=2) = {expected.tolist()}",
        computed=f"U(k=2) = {got.tolist()}",
        delta="exact",
        passed=ok,
    )


@register(
    "Building prediction & uncertainty",
    "ISO 12999-2:2020 Clause 7, Examples 1/2",
    "Single-number U (k=2): alpha_w and DLalpha,NRD",
)
def _chk_iso12999_2_single_numbers() -> Outcome:
    u_aw = float(
        ph.weighted_coefficient_uncertainty(
            ref.ISO12999_2_ALPHA_W_EXAMPLE
        ).reported_expanded_uncertainty[0]
    )
    u_dl = float(
        ph.single_number_rating_uncertainty(
            ref.ISO12999_2_DLALPHA_EXAMPLE
        ).reported_expanded_uncertainty[0]
    )
    ok = (
        u_aw == ref.ISO12999_2_ALPHA_W_U_K2 and u_dl == ref.ISO12999_2_DLALPHA_U_K2
    )
    return Outcome(
        expected=f"alpha_w +/-{ref.ISO12999_2_ALPHA_W_U_K2}, "
        f"DLalpha +/-{ref.ISO12999_2_DLALPHA_U_K2} dB",
        computed=f"alpha_w +/-{u_aw}, DLalpha +/-{u_dl} dB",
        delta="exact",
        passed=ok,
    )


# ===========================================================================
# Domain 8 - Outdoor propagation & occupational exposure
# ===========================================================================
def _iso9613_table1(point: tuple[float, float, float, float]) -> Outcome:
    """Compare air_attenuation against an ISO 9613-1 Table 1 grid point (dB/km)."""
    temp, rh, freq, alpha_km = point
    computed = float(
        ph.air_attenuation(freq, temp, rh, exact_midband=True)[()]
    ) * 1000.0  # dB/m -> dB/km
    # Tolerance = 1 in the last printed (3-significant-figure) digit.
    tol = 10.0 ** (math.floor(math.log10(alpha_km)) - 2)
    return numeric(alpha_km, computed, tol, unit="dB/km", places=3)


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-1:1993 Table 1",
    "Air attenuation @ 10 degC, 70 %, 1 kHz",
)
def _chk_iso9613_table1_mid() -> Outcome:
    return _iso9613_table1(ref.ISO9613_1_TABLE1_MID)


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-1:1993 Table 1",
    "Air attenuation @ 0 degC, 20 %, 2 kHz",
)
def _chk_iso9613_table1_corner() -> Outcome:
    return _iso9613_table1(ref.ISO9613_1_TABLE1_CORNER)


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 Table 2",
    "Atmospheric attenuation grid, 6 conditions x 8 octave bands, dB/km",
)
def _chk_iso9613_2_table2_grid() -> Outcome:
    """Every printed Table 2 cell (exact midbands) to half its last digit.

    Worst residual in units of the per-cell tolerance; the documented
    15 degC / 80 % / 1 kHz print quirk carries a 0.06 dB/km tolerance
    (printed 4,1 vs exact-midband 4,151).
    """
    worst = 0.0
    for (temp, rh), row in ref.ISO9613_2_TABLE2.items():
        alpha = ph.air_attenuation(
            ref.ISO9613_2_TABLE2_BANDS, temp, rh, 101.325, exact_midband=True
        ) * 1000.0
        for got, printed, band in zip(alpha, row, ref.ISO9613_2_TABLE2_BANDS):
            tol = 0.5 if printed >= 100.0 else 0.05
            if (temp, rh, band) == (15.0, 80.0, 1000.0):
                tol = 0.06
            residual = abs(float(got) - printed) / tol
            if not math.isfinite(residual):
                return Outcome(expected="finite Table 2 residuals",
                               computed=f"non-finite at {band} Hz", delta="inf",
                               passed=False)
            worst = max(worst, residual)
    return Outcome(
        expected="all 48 cells within half a printed digit",
        computed=f"worst residual {worst:.3f} x tolerance",
        delta=f"{worst:.3f} x",
        passed=worst <= 1.0,
    )



@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 Eq. (7)",
    "Geometrical divergence Adiv = 20 lg(d/d0) + 11 at 100 m",
)
def _chk_iso9613_2_adiv() -> Outcome:
    computed = ph.geometric_divergence(100.0)
    return numeric(ref.ISO9613_2_ADIV_100M, computed, 1e-9, unit="dB", places=6)


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 Table 3",
    "Ground b'(0) porous limit -> Agr(250 Hz) = 2(-1.5 + 10.1)",
)
def _chk_iso9613_2_ground_limit() -> Outcome:
    # Porous ground both sides (Gs = Gr = 1), source and receiver on the ground
    # (hs = hr = 0), fully-developed path (dp -> inf): the 250 Hz band isolates
    # the Table 3 limit b'(0) = 1,5 + 8,6 = 10,1, so Agr = 2(-1,5 + 10,1) = 17,2.
    big = 1.0e7
    agr = ph.ground_attenuation(big, 0.0, 0.0, [250.0], 1.0, 1.0, 1.0,
                                projected_distance=big)
    return numeric(
        ref.ISO9613_2_GROUND_AGR_250_POROUS, float(agr[0]), 1e-6, unit="dB",
        places=4,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 clause 7.4",
    "Single-edge diffraction saturates at the 20 dB cap",
)
def _chk_iso9613_2_barrier_single_cap() -> Outcome:
    b = ph.Barrier(source_to_edge=50.0, edge_to_receiver=50.0)
    dz = ph.barrier_attenuation(b, 60.0, ph.DEFAULT_FREQUENCIES)
    return numeric(
        ref.ISO9613_2_BARRIER_CAP_SINGLE, float(np.max(dz)), 1e-9, unit="dB",
        places=6,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9613-2:1996 clause 7.4",
    "Double-edge diffraction saturates at the 25 dB cap",
)
def _chk_iso9613_2_barrier_double_cap() -> Outcome:
    b = ph.Barrier(source_to_edge=50.0, edge_to_receiver=50.0, edge_separation=5.0)
    dz = ph.barrier_attenuation(b, 60.0, ph.DEFAULT_FREQUENCIES)
    return numeric(
        ref.ISO9613_2_BARRIER_CAP_DOUBLE, float(np.max(dz)), 1e-9, unit="dB",
        places=6,
    )


def _iso9612_annex_d_tasks() -> list[ph.Task]:
    """Rebuild the ISO 9612 Annex D Task objects from the shared input table."""
    from phonometry.hearing.occupational_exposure import Task

    tasks = []
    for samples, duration, drange in ref.ISO9612_ANNEX_D_TASKS:
        tasks.append(Task(samples=samples, duration_hours=duration,
                          duration_range=drange))
    return tasks


def _lex_and_u(lex: float, u: float, exp_lex: float, exp_u: float,
               tol_lex: float, tol_u: float) -> Outcome:
    """Combined LEX,8h + expanded-uncertainty outcome for one ISO 9612 example."""
    ok = abs(lex - exp_lex) <= tol_lex and abs(u - exp_u) <= tol_u
    return Outcome(
        expected=f"LEX,8h {exp_lex:.1f}; U {exp_u:.1f} dB",
        computed=f"LEX,8h {lex:.1f}; U {u:.1f} dB",
        delta=f"{lex - exp_lex:+.2f}; {u - exp_u:+.2f} dB",
        passed=ok,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9612:2009 Annex D",
    "Task-based LEX,8h + U (welder day, case a)",
)
def _chk_iso9612_annex_d() -> Outcome:
    res = ph.task_based_exposure(
        _iso9612_annex_d_tasks(), include_duration_uncertainty=False, warn=False
    )
    return _lex_and_u(
        res.lex_8h, res.expanded_uncertainty,
        ref.ISO9612_ANNEX_D_LEX_8H, ref.ISO9612_ANNEX_D_U, 0.05, 0.1,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9612:2009 Annex E",
    "Job-based LEX,8h + U (production line, 18 workers)",
)
def _chk_iso9612_annex_e() -> Outcome:
    res = ph.job_based_exposure(
        list(ref.ISO9612_ANNEX_E_SAMPLES), ref.ISO9612_ANNEX_E_TE_HOURS
    )
    return _lex_and_u(
        res.lex_8h, res.expanded_uncertainty,
        ref.ISO9612_ANNEX_E_LEX_8H, ref.ISO9612_ANNEX_E_U, 0.1, 0.05,
    )


@register(
    "Outdoor propagation & occupational exposure",
    "ISO 9612:2009 Annex F",
    "Full-day LEX,8h + U (forklift drivers)",
)
def _chk_iso9612_annex_f() -> Outcome:
    res = ph.full_day_exposure(
        list(ref.ISO9612_ANNEX_F_SAMPLES), ref.ISO9612_ANNEX_F_TE_HOURS
    )
    return _lex_and_u(
        res.lex_8h, res.expanded_uncertainty,
        ref.ISO9612_ANNEX_F_LEX_8H, ref.ISO9612_ANNEX_F_U, 0.05, 0.05,
    )


# ---------------------------------------------------------------------------
# Materials: absorption rating, airflow resistance & impedance tube
# ---------------------------------------------------------------------------
_MATERIALS = "Materials: absorption, airflow & impedance"


@register(_MATERIALS, "ISO 11654:1997 Annex A.1", "Weighted absorption alpha_w (no indicator)")
def _chk_iso11654_a1() -> Outcome:
    res = ph.weighted_absorption(list(ref.ISO11654_ANNEX_A1_ALPHA_P))
    out = numeric(ref.ISO11654_ANNEX_A1_ALPHA_W, res.alpha_w, 5e-4, places=2)
    # alpha_w matches AND no shape indicator applies AND class is C.
    ok = (
        out.passed
        and res.shape_indicator == ref.ISO11654_ANNEX_A1_INDICATOR
        and res.absorption_class == ref.ISO11654_ANNEX_A1_CLASS
    )
    return Outcome(
        expected=f"{ref.ISO11654_ANNEX_A1_ALPHA_W:.2f} (class C, no indic.)",
        computed=f"{res.alpha_w:.2f} (class {res.absorption_class}, '{res.shape_indicator}')",
        delta=out.delta,
        passed=ok,
    )


@register(_MATERIALS, "ISO 11654:1997 Annex A.2", "Weighted absorption alpha_w with M indicator")
def _chk_iso11654_a2() -> Outcome:
    res = ph.weighted_absorption(list(ref.ISO11654_ANNEX_A2_ALPHA_P))
    out = numeric(ref.ISO11654_ANNEX_A2_ALPHA_W, res.alpha_w, 5e-4, places=2)
    ok = out.passed and res.shape_indicator == ref.ISO11654_ANNEX_A2_INDICATOR
    return Outcome(
        expected=f"{ref.ISO11654_ANNEX_A2_ALPHA_W:.2f}(M)",
        computed=res.rating_label,
        delta=out.delta,
        passed=ok,
    )


@register(_MATERIALS, "ISO 9053-2:2020 Annex A.3", "Thermal boundary-layer thickness b")
def _chk_iso9053_2_boundary() -> Outcome:
    b = ph.thermal_boundary_layer_thickness(frequency=ref.ISO9053_2_ANNEX_A_FREQUENCY)
    return numeric(
        ref.ISO9053_2_ANNEX_A_BOUNDARY_LAYER, b, 5e-6, unit="m", places=5,
    )


@register(_MATERIALS, "ISO 9053-2:2020 Annex A.3", "Effective ratio of specific heats kappa'")
def _chk_iso9053_2_kappa() -> Outcome:
    kp = ph.effective_kappa(
        cavity_surface=ref.ISO9053_2_ANNEX_A_SURFACE,
        cavity_volume=ref.ISO9053_2_ANNEX_A_VOLUME,
        frequency=ref.ISO9053_2_ANNEX_A_FREQUENCY,
    )
    return numeric(ref.ISO9053_2_ANNEX_A_KAPPA_PRIME, kp, 5e-4, places=3)


@register(_MATERIALS, "ISO 10534-1:1996 Eqs (9)/(13)/(14)", "Absorption from standing-wave ratio s=3")
def _chk_iso10534_1_swr() -> Outcome:
    alpha = float(ph.standing_wave_absorption(ref.ISO10534_1_SWR))
    # The intermediate |r| = (s-1)/(s+1) (Eq. (13)) must match its shared
    # oracle too, so both steps of the chain are pinned.
    from phonometry.materials.impedance_tube import standing_wave_reflection_magnitude

    r_mag = float(standing_wave_reflection_magnitude(ref.ISO10534_1_SWR))
    out = numeric(ref.ISO10534_1_ABSORPTION, alpha, 1e-9, places=4)
    r_ok = abs(r_mag - ref.ISO10534_1_REFLECTION_MAGNITUDE) <= 1e-9
    # Show both chained values so a |r| failure is visible in the report.
    expected = (
        f"alpha {out.expected}, \\|r\\| {ref.ISO10534_1_REFLECTION_MAGNITUDE:g}"
    )
    computed = f"alpha {out.computed}, \\|r\\| {r_mag:.4f}"
    return Outcome(expected, computed, out.delta, out.passed and r_ok)


@register(
    _MATERIALS,
    "ISO 10534-2 Eq. (17) / Annex D",
    "Two-microphone round trip recovers a known reflection factor",
)
def _chk_iso10534_2_roundtrip() -> Outcome:
    # Synthesise the transfer function H12 of a known complex r via the
    # Annex D field equations (Eq. (D.7)), then recover r with the library's
    # Eq. (17) reduction. Synthesis and reduction share only the plane-wave
    # field model, so this is an algebraic identity: the only residual is
    # float rounding, hence the 1e-9 tolerance.
    from phonometry.materials.impedance_tube import reflection_factor, tube_wavenumber

    f = np.array([500.0, 1000.0, 1800.0])
    x1, spacing, c0 = 0.12, 0.03, 343.2
    r_true = 0.3 - 0.4j
    k0 = np.asarray(tube_wavenumber(f, c0))
    x2 = x1 - spacing
    h12 = (
        (np.exp(1j * k0 * x2) + r_true * np.exp(-1j * k0 * x2))
        / (np.exp(1j * k0 * x1) + r_true * np.exp(-1j * k0 * x1))
    )
    r = reflection_factor(h12, spacing=spacing, x1=x1, wavenumber=k0)
    err = float(np.max(np.abs(np.asarray(r) - r_true)))
    # NOTE: no pipe characters in the label (it lands in a Markdown table cell).
    return numeric(0.0, err, 1e-9, places=9,
                   expected_label="abs(r - (0.3-0.4j)) = 0 (identity, +/-1e-9)")


# ---------------------------------------------------------------------------
# Scattering & diffusion (ISO 17497-1/-2)
# ---------------------------------------------------------------------------
_SCATTERING = "Scattering & diffusion (ISO 17497)"


@register(_SCATTERING, "ISO 17497-1:2004 Eq (2)", "Reference speed of sound at 20 C")
def _chk_iso17497_1_speed() -> Outcome:
    c = float(ph.speed_of_sound(20.0))
    return numeric(ref.ISO17497_1_SPEED_OF_SOUND_20C, c, 1e-6, unit="m/s", places=4)


@register(_SCATTERING, "ISO 17497-1:2004 Eqs (1)/(4)/(5)", "Scattering coefficient (synthetic chain)")
def _chk_iso17497_1_scattering() -> Outcome:
    t1, t2, t3, t4 = ref.ISO17497_1_CHAIN_T
    c = ref.ISO17497_1_CHAIN_C
    alpha_s = ph.random_incidence_absorption(
        ref.ISO17497_1_CHAIN_V, ref.ISO17497_1_CHAIN_S, c1=c, T1=t1, c2=c, T2=t2
    )
    alpha_spec = ph.specular_absorption_coefficient(
        ref.ISO17497_1_CHAIN_V, ref.ISO17497_1_CHAIN_S, c3=c, T3=t3, c4=c, T4=t4
    )
    s = float(ph.scattering_coefficient(alpha_spec, alpha_s))
    return numeric(ref.ISO17497_1_CHAIN_SCATTERING, s, 1e-9, places=4)


@register(_SCATTERING, "ISO 17497-1:2004 Annex A.5", "Expanded uncertainty of scattering coefficient")
def _chk_iso17497_1_uncertainty() -> Outcome:
    u = float(ph.scattering_coefficient_uncertainty(
        ref.ISO17497_1_A5_ALPHA_SPEC,
        ref.ISO17497_1_A5_ALPHA_S,
        ref.ISO17497_1_A5_U_ALPHA_SPEC,
        ref.ISO17497_1_A5_U_ALPHA_S,
    ).u_scattering)
    return numeric(ref.ISO17497_1_A5_U_SCATTERING, u, 1e-6, places=5)


# The ISO 17497-2 arc levels in ``reference_data`` are generated by the
# library's own Fraunhofer model for a published geometry (Cox & D'Antonio 3e
# Appendix B section 7: N = 7 QRD, 6 periods, 3.6 m wide, 0.2 m deep; the
# commercial QRD of Hargreaves et al. 2000, Table I), so these three rows are
# arithmetic oracles for Formulas (5)/(7) on committed levels - the external
# anchor against published third-party BEM data is the Appendix B rows below.
@register(_SCATTERING, "ISO 17497-2:2012 Formula (5)", "Directional diffusion coefficient (QRD, model arc)")
def _chk_iso17497_2_diffusion_qrd() -> Outcome:
    d = float(ph.directional_diffusion_coefficient(list(ref.ISO17497_2_QRD_LEVELS)))
    return numeric(ref.ISO17497_2_QRD_DIFFUSION, d, 1e-6, places=4)


@register(_SCATTERING, "ISO 17497-2:2012 Formula (5)", "Directional diffusion coefficient (flat reference)")
def _chk_iso17497_2_diffusion_flat() -> Outcome:
    d = float(ph.directional_diffusion_coefficient(list(ref.ISO17497_2_FLAT_LEVELS)))
    return numeric(ref.ISO17497_2_FLAT_DIFFUSION, d, 1e-6, places=4)


@register(_SCATTERING, "ISO 17497-2:2012 Formula (7)", "Normalised diffusion coefficient (QRD, model arc)")
def _chk_iso17497_2_diffusion_normalized() -> Outcome:
    d_qrd = ph.directional_diffusion_coefficient(list(ref.ISO17497_2_QRD_LEVELS))
    d_flat = ph.directional_diffusion_coefficient(list(ref.ISO17497_2_FLAT_LEVELS))
    d_n = float(ph.normalized_diffusion_coefficient(d_qrd, d_flat))
    return numeric(ref.ISO17497_2_NORMALIZED_DIFFUSION, d_n, 1e-6, places=4)


# External anchor against published third-party data: Cox & D'Antonio 3e
# Appendix B (pp. 481-485), section 7, "N = 7 QRD, 6 periods, 0.2 m deep",
# normal incidence: 2D BEM normalised diffusion coefficients per one-third-
# octave band. The library's Fraunhofer prediction (band = energy average of
# seven single-frequency responses, as in section 5.2.5) matches the published
# 200-400 Hz bands within 0.01. Low-band anchor only: across the full
# published 100-5000 Hz range the model-vs-BEM mean absolute deviation is
# ~0.09, because edge diffraction and near-grazing effects are outside the
# far-field phase-grating model.
def _make_cox_appendix_b_check(band: float, published: float) -> Callable[[], Outcome]:
    def check() -> Outcome:
        from diffuser_prediction import predicted_band_normalized_diffusion

        d_n = predicted_band_normalized_diffusion(band)
        return numeric(published, d_n, ref.COX3E_APPENDIX_B_TOLERANCE, places=3)
    return check


for _band, _published in zip(
    ref.COX3E_APPENDIX_B_QRD_BANDS, ref.COX3E_APPENDIX_B_QRD_DN, strict=True
):
    register(
        _SCATTERING,
        "Cox & D'Antonio 3e App. B (2D BEM)",
        f"Normalised diffusion d_n, N=7 QRD x 6 periods, {_band:g} Hz band "
        "(low-band anchor)",
    )(_make_cox_appendix_b_check(_band, _published))


@register(_SCATTERING, "ISO 17497-2:2012 Formula (8)", "Zenith area factor (radians convention)")
def _chk_iso17497_2_area_factor() -> Outcome:
    n = ph.area_factors([0.0, 30.0, 60.0, 90.0], delta_theta=5.0)
    return numeric(ref.ISO17497_2_AREA_FACTOR_ZENITH, float(n[0]), 1e-6, places=5)


@register(_SCATTERING, "Cox & D'Antonio Eq (10.3)", "QRD deepest well depth (N=7, f0=500 Hz)")
def _chk_diffuser_qrd_depth() -> Outcome:
    d = ph.qrd_well_depths(7, 500.0, speed_of_sound=343.0)
    return numeric(ref.DIFFUSER_QRD7_MAX_DEPTH, float(d.max()), 1e-12, unit="m", places=4)


@register(_SCATTERING, "Cox & D'Antonio Eq (5.8) + ISO 17497-2 Formula (7)",
          "Flat-panel predicted normalised diffusion (self-reference zero)")
def _chk_diffuser_flat_normalized() -> Outcome:
    spectrum = ph.predicted_diffusion_spectrum(
        0.10, [2000.0], depths=[0.0] * 7, periods=5
    )
    assert spectrum.normalized is not None
    return numeric(
        ref.DIFFUSER_FLAT_NORMALIZED_DIFFUSION,
        float(spectrum.normalized[0]), 1e-12, places=6,
    )


@register(_SCATTERING, "Cox & D'Antonio Eq (5.8) + ISO 17497-2 Formula (7)",
          "QRD predicted normalised diffusion at 2 kHz (above flat panel)")
def _chk_diffuser_qrd_normalized() -> Outcome:
    depths = ph.qrd_well_depths(7, 500.0, speed_of_sound=343.0)
    spectrum = ph.predicted_diffusion_spectrum(
        0.10, [2000.0], depths=depths, periods=5
    )
    assert spectrum.normalized is not None
    return numeric(
        ref.DIFFUSER_QRD7_NORMALIZED_DIFFUSION_2K,
        float(spectrum.normalized[0]), 1e-9, places=4,
    )


# ---------------------------------------------------------------------------
# In-situ road-surface absorption (ISO 13472-1/-2)
# ---------------------------------------------------------------------------
_ROAD = "In-situ road absorption (ISO 13472)"


@register(_ROAD, "ISO 13472-1:2002 Clause 4.2", "Geometrical-spreading factor Kr")
def _chk_iso13472_1_kr() -> Outcome:
    kr = ph.geometric_spreading_factor()
    return numeric(ref.ISO13472_1_KR, kr, 1e-12, places=4)


@register(_ROAD, "ISO 13472-1:2002 Annex A", "Maximum-sampled-area radius")
def _chk_iso13472_1_msa() -> Outcome:
    r = ph.max_sampled_area_radius(ref.ISO13472_1_MSA_WINDOW)
    return numeric(ref.ISO13472_1_MSA_RADIUS, r, 1e-6, unit="m", places=4)


@register(_ROAD, "ISO 13472-2:2010 Clause 5.4.1", "Spot-tube upper usable frequency f_u")
def _chk_iso13472_2_fu() -> Outcome:
    fu = ph.spot_tube_upper_frequency(
        ref.ISO13472_2_SPOT_DIAMETER, ref.ISO13472_2_SPOT_SPEED
    )
    return numeric(ref.ISO13472_2_SPOT_FU, fu, 0.1, unit="Hz", places=1)


# ---------------------------------------------------------------------------
# Precision sound power (ISO 3745 / ISO 9614-3)
# ---------------------------------------------------------------------------
_PRECISION_POWER = "Precision sound power (ISO 3745 / 9614-3)"


@register(_PRECISION_POWER, "ISO 3745:2012 Clause 10.5 EXAMPLE", "Expanded uncertainty U (k=2)")
def _chk_iso3745_uncertainty() -> Outcome:
    u = float(ph.precision_uncertainty(
        ref.ISO3745_U_SIGMA_R0, ref.ISO3745_U_SIGMA_OMC, ref.ISO3745_U_COVERAGE
    ))
    return numeric(ref.ISO3745_U_EXPANDED, u, 1e-3, unit="dB", places=3)


@register(_PRECISION_POWER, "ISO 3745:2012 Eq (11)", "K1 background floor (6 dB edge band)")
def _chk_iso3745_k1_floor() -> Outcome:
    k1 = ph.precision_background_correction(
        np.array([[ref.ISO3745_K1_EDGE_LEVEL]]),
        np.array([[ref.ISO3745_K1_EDGE_BACKGROUND]]),
        np.array([ref.ISO3745_K1_EDGE_FREQUENCY]),
    )
    return numeric(ref.ISO3745_K1_EDGE_FLOOR, float(k1[0, 0]), 1e-4, unit="dB", places=4)


@register(_PRECISION_POWER, "ISO 3745:2012 Eq (16)", "Meteorological C1 at 23 C reference")
def _chk_iso3745_c1() -> Outcome:
    c1 = ph.meteorological_corrections(23.0, 101.325).c1
    return numeric(ref.ISO3745_C1_REFERENCE, c1, 1e-4, unit="dB", places=4)


@register(_PRECISION_POWER, "ISO 9614-3:2002 Eqs (5)/(8)/(9)", "Uniform-intensity LW recovery")
def _chk_iso9614_3_uniform() -> Outcome:
    areas = np.array(ref.ISO9614_3_UNIFORM_AREAS, dtype=float)
    i_n = np.full(areas.shape, ref.ISO9614_3_UNIFORM_POWER / float(areas.sum()))
    res = ph.sound_power_intensity_precision(i_n, areas)
    return numeric(ref.ISO9614_3_UNIFORM_LW, float(res.sound_power_level[0]), 1e-9, unit="dB", places=4)


# ===========================================================================
# Human vibration (ISO 8041-1 / ISO 2631 / ISO 5349 / Directive 2002/44/EC)
# ===========================================================================
_HUMAN_VIB = "Human vibration (ISO 8041 / 2631 / 5349)"


def _true_centre(n: int) -> float:
    """True IEC 61260 one-third-octave centre ``10^(n/10)`` Hz."""
    return float(10.0 ** (n / 10.0))


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table B.8", "Wk design-goal factor at 6,31 Hz")
def _chk_iso8041_wk_annex_b() -> Outcome:
    factor = float(ph.weighting_factors("Wk", _true_centre(8))[0])
    return numeric(ref.ISO8041_1_WK_FACTOR_6P31HZ, factor, 1e-3, rel=True, places=4)


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table B.9", "Wm design-goal factor at 1,585 Hz")
def _chk_iso8041_wm_annex_b() -> Outcome:
    factor = float(ph.weighting_factors("Wm", _true_centre(2))[0])
    return numeric(ref.ISO8041_1_WM_FACTOR_1P585HZ, factor, 1e-3, rel=True, places=4)


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table 1", "Wh factor at the 500 rad/s reference")
def _chk_iso8041_wh_reference() -> Outcome:
    factor = float(ph.weighting_factors("Wh", ref.ISO8041_1_WH_REF_FREQ_HZ)[0])
    return numeric(ref.ISO8041_1_WH_REF_FACTOR, factor, 1.5e-3, rel=True, places=4)


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table B.1", "Wb design-goal factor at 6,31 Hz")
def _chk_iso8041_wb_annex_b() -> Outcome:
    factor = float(ph.weighting_factors("Wb", _true_centre(8))[0])
    return numeric(ref.ISO8041_1_WB_FACTOR_6P31HZ, factor, 1e-3, rel=True, places=4)


@register(
    _HUMAN_VIB, "ISO 8041-1:2017 Table B.1", "Wb design-goal factors at 1 / 100 Hz"
)
def _chk_iso8041_wb_annex_b_edges() -> Outcome:
    worst = max(
        abs(float(ph.weighting_factors("Wb", _true_centre(n))[0]) / expected - 1.0)
        for n, expected in (
            (0, ref.ISO8041_1_WB_FACTOR_1HZ),
            (20, ref.ISO8041_1_WB_FACTOR_100HZ),
        )
    )
    return numeric(
        0.0, worst, 1e-3, places=6, expected_label="max rel dev ≤ 0,1 %"
    )


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table 1", "Wc factor at the 100 rad/s reference")
def _chk_iso8041_wc_reference() -> Outcome:
    factor = float(ph.weighting_factors("Wc", ref.ISO8041_1_WBV_REF_FREQ_HZ)[0])
    return numeric(ref.ISO8041_1_WC_REF_FACTOR, factor, 1e-3, rel=True, places=4)


@register(
    _HUMAN_VIB,
    "ISO 8041-1:2017 Table 1 + Table B.3",
    "Wd factors at the 100 rad/s reference and 1 Hz",
)
def _chk_iso8041_wd_reference_and_annex_b() -> Outcome:
    worst = max(
        abs(
            float(ph.weighting_factors("Wd", freq)[0]) / expected - 1.0
        )
        for freq, expected in (
            (ref.ISO8041_1_WBV_REF_FREQ_HZ, ref.ISO8041_1_WD_REF_FACTOR),
            (_true_centre(0), ref.ISO8041_1_WD_FACTOR_1HZ),
        )
    )
    return numeric(
        0.0, worst, 1e-3, places=6, expected_label="max rel dev ≤ 0,1 %"
    )


@register(_HUMAN_VIB, "ISO 8041-1:2017 Table B.4", "We design-goal factor at 8 Hz")
def _chk_iso8041_we_annex_b() -> Outcome:
    factor = float(ph.weighting_factors("We", _true_centre(9))[0])
    return numeric(ref.ISO8041_1_WE_FACTOR_8HZ, factor, 1e-3, rel=True, places=4)


@register(
    _HUMAN_VIB, "ISO 8041-1:2017 Table B.5", "Wf design-goal factors at 0,1585 / 0,1 Hz"
)
def _chk_iso8041_wf_annex_b() -> Outcome:
    worst = max(
        abs(float(ph.weighting_factors("Wf", _true_centre(n))[0]) / expected - 1.0)
        for n, expected in (
            (-8, ref.ISO8041_1_WF_FACTOR_0P1585HZ),
            (-10, ref.ISO8041_1_WF_FACTOR_0P1HZ),
        )
    )
    return numeric(
        0.0, worst, 1e-3, places=6, expected_label="max rel dev ≤ 0,1 %"
    )


@register(
    _HUMAN_VIB, "ISO 8041-1:2017 Table B.7", "Wj design-goal factors at 6,31 / 8 Hz"
)
def _chk_iso8041_wj_annex_b() -> Outcome:
    worst = max(
        abs(float(ph.weighting_factors("Wj", _true_centre(n))[0]) / expected - 1.0)
        for n, expected in (
            (8, ref.ISO8041_1_WJ_FACTOR_6P31HZ),
            (9, ref.ISO8041_1_WJ_FACTOR_8HZ),
        )
    )
    return numeric(
        0.0, worst, 1e-3, places=6, expected_label="max rel dev ≤ 0,1 %"
    )


@register(
    _HUMAN_VIB,
    "ISO 8041-1:2017 Table 5 + Annex B",
    "All nine weightings inside the tolerance envelope (318 printed bands)",
)
def _chk_iso8041_table5_envelope() -> Outcome:
    violations = 0
    for name, rows in ref.ISO8041_1_ANNEX_B_FACTORS.items():
        ft1, ft2, ft3, ft4 = ref.ISO8041_1_TABLE4_TRANSITIONS[name]
        for n, printed in rows:
            freq = _true_centre(n)
            if freq <= ft1:
                region = 0
            elif freq < ft2:
                region = 1
            elif freq <= ft3:
                region = 2
            elif freq < ft4:
                region = 3
            else:
                region = 4
            upper, lower = ref.ISO8041_1_TABLE5_TOLERANCES[region]
            ratio = float(ph.weighting_factors(name, freq)[0]) / printed - 1.0
            if not -lower <= ratio <= upper:
                violations += 1
    return numeric(
        0.0, float(violations), 0.0, places=0,
        expected_label="0 bands outside the Table 5 tolerances",
    )


@register(_HUMAN_VIB, "ISO 5349-2:2001 Example E.2.1", "Single-tool daily exposure A(8)")
def _chk_iso5349_e21() -> Outcome:
    a8 = ph.daily_exposure(7.4, 2.5 * 3600.0)
    return numeric(ref.ISO5349_2_E21_A8, a8, 0.05, unit="m/s^2", places=2)


@register(_HUMAN_VIB, "ISO 5349-2:2001 Example E.3", "Forestry three-task A(8)")
def _chk_iso5349_e3() -> Outcome:
    a8 = ph.hav_daily_exposure(
        [4.6, 6.0, 3.6], [2 * 3600.0, 1 * 3600.0, 2 * 3600.0]
    )
    return numeric(ref.ISO5349_2_E3_A8, a8, 0.05, unit="m/s^2", places=2)


@register(_HUMAN_VIB, "ISO 5349-1:2001 Eq. (C.1)", "VWF 10 % lifetime Dy at A(8)=7")
def _chk_iso5349_vwf() -> Outcome:
    dy = ph.hav_vwf_lifetime_years(ref.ISO5349_1_VWF_A8)
    return numeric(ref.ISO5349_1_VWF_DY_YEARS, dy, 0.1, unit="yr", places=2)


@register(_HUMAN_VIB, "Directive 2002/44/EC Art. 3", "HAV/WBV action & limit values")
def _chk_directive_2002_44() -> Outcome:
    hav = ph.exposure_assessment(1.0, kind="hav")
    wbv = ph.exposure_assessment(0.1, kind="wbv")
    ok = (
        hav.action_value == ref.DIRECTIVE_2002_44_HAV_EAV
        and hav.limit_value == ref.DIRECTIVE_2002_44_HAV_ELV
        and wbv.action_value == ref.DIRECTIVE_2002_44_WBV_EAV
        and wbv.limit_value == ref.DIRECTIVE_2002_44_WBV_ELV
    )
    exp = (
        f"HAV {ref.DIRECTIVE_2002_44_HAV_EAV}/{ref.DIRECTIVE_2002_44_HAV_ELV}, "
        f"WBV {ref.DIRECTIVE_2002_44_WBV_EAV}/{ref.DIRECTIVE_2002_44_WBV_ELV} m/s^2"
    )
    got = (
        f"HAV {hav.action_value}/{hav.limit_value}, "
        f"WBV {wbv.action_value}/{wbv.limit_value} m/s^2"
    )
    return Outcome(expected=exp, computed=got, delta="0", passed=ok)


# ---------------------------------------------------------------------------
# Speech Intelligibility Index (ANSI S3.5-1997)
# ---------------------------------------------------------------------------
_SII = "Speech intelligibility (ANSI S3.5-1997)"


@register(_SII, "ANSI S3.5-1997 Table 3", "Band-importance function normalisation")
def _chk_sii_band_importance_sum() -> Outcome:
    total = float(ph.hearing.sii.BAND_IMPORTANCE.sum())
    return numeric(ref.ANSIS3_5_BAND_IMPORTANCE_SUM, total, 1e-9, places=6)


@register(_SII, "ANSI S3.5-1997 clause 5.4", "Equivalent masking spectrum level at 200 Hz")
def _chk_sii_masking() -> Outcome:
    result = ph.speech_intelligibility_index("normal")
    return numeric(ref.ANSIS3_5_MASKING_Z_200HZ, float(result.masking[1]), 1e-3, places=3)


@register(_SII, "ANSI S3.5-1997 clause 5.6", "Equivalent disturbance in quiet at 5000 Hz")
def _chk_sii_disturbance_quiet() -> Outcome:
    # In quiet Di = max(Zi, Xi') = Xi' = -23.6 dB (Table 3) at 5000 Hz; an
    # energy-sum disturbance would read above the reference internal noise.
    result = ph.hearing.sii.speech_intelligibility_index("normal")
    return numeric(
        ref.ANSIS3_5_DISTURBANCE_5000HZ, float(result.disturbance[15]), 1e-2,
        unit="dB", places=2,
    )


@register(_SII, "ANSI S3.5-1997 clause 6", "SII, noise 30 dB plus hearing loss 40 dB")
def _chk_sii_noise_plus_loss() -> Outcome:
    result = ph.hearing.sii.speech_intelligibility_index(
        "normal",
        noise_spectrum=np.full(18, 30.0),
        threshold=np.full(18, 40.0),
    )
    return numeric(ref.ANSIS3_5_NOISE_PLUS_LOSS, result.sii, 1e-4, places=4)


@register(_SII, "R CRAN 'SII' Example C.2", "One-third-octave method, independent oracle")
def _chk_sii_r_example() -> Outcome:
    result = ph.hearing.sii.speech_intelligibility_index(
        np.full(18, 54.0),
        np.array([40.0, 30.0, 20.0] + [0.0] * 15),
        threshold=np.zeros(18),
    )
    return numeric(ref.ANSIS3_5_R_EXAMPLE_C2, result.sii, 1e-4, places=6)


@register(_SII, "ANSI S3.5-1997 clause 6", "SII, standard speech in quiet, normal hearing")
def _chk_sii_standard_quiet() -> Outcome:
    result = ph.speech_intelligibility_index("normal")
    return numeric(ref.ANSIS3_5_STANDARD_QUIET, result.sii, 1e-6, places=8)


@register(_SII, "ANSI S3.5-1997 Table 3", "Loud-effort speech spectrum level at 1 kHz")
def _chk_sii_loud_spectrum() -> Outcome:
    from phonometry.hearing.sii import standard_speech_spectrum

    value = float(standard_speech_spectrum("loud")[8])
    return numeric(ref.ANSIS3_5_LOUD_1KHZ, value, 1e-9, unit="dB", places=2)


# ---------------------------------------------------------------------------
# Short-time objective intelligibility (STOI / ESTOI)
# ---------------------------------------------------------------------------
_STOI = "Objective intelligibility (STOI / ESTOI)"


def _stoi_speech_like(seed: int) -> np.ndarray:
    """A deterministic speech-like signal at the 10 kHz STOI internal rate."""
    fs = ph.hearing.objective_intelligibility.SAMPLE_RATE
    rng = np.random.default_rng(seed)
    t = np.arange(3 * fs) / fs
    sig = np.zeros_like(t)
    for f0 in (200.0, 400.0, 700.0, 1100.0, 1800.0, 2600.0):
        depth = 0.5 * (1.0 + np.sin(2.0 * np.pi * rng.uniform(2.0, 6.0) * t
                                    + rng.uniform(0.0, 2.0 * np.pi)))
        sig += depth * np.sin(2.0 * np.pi * f0 * t + rng.uniform(0.0, 2.0 * np.pi))
    return np.asarray(sig, dtype=np.float64)


@register(
    _STOI,
    "Taal et al. 2011 (Eq. 6, degenerate)",
    "STOI of a signal against itself = 1 (perfect correlation)",
)
def _chk_stoi_identity() -> Outcome:
    x = _stoi_speech_like(1)
    return numeric(1.0, ph.stoi(x, x, ph.hearing.objective_intelligibility.SAMPLE_RATE).value,
                   1e-6, places=6)


@register(
    _STOI,
    "Jensen & Taal 2016 (Eq. 8, degenerate)",
    "ESTOI of a signal against itself = 1 (perfect spectral correlation)",
)
def _chk_estoi_identity() -> Outcome:
    x = _stoi_speech_like(1)
    fs = ph.hearing.objective_intelligibility.SAMPLE_RATE
    return numeric(1.0, ph.stoi(x, x, fs, extended=True).value, 1e-6, places=6)


@register(
    _STOI,
    "Taal et al. 2011 (monotonicity with SNR)",
    "STOI rises from -15 dB to +25 dB SNR speech-shaped noise",
)
def _chk_stoi_monotonic() -> Outcome:
    fs = ph.hearing.objective_intelligibility.SAMPLE_RATE
    x = _stoi_speech_like(2)
    rng = np.random.default_rng(10)
    noise = rng.standard_normal(x.size)
    scale = np.sqrt(np.mean(x**2)) / np.sqrt(np.mean(noise**2))
    lo = ph.stoi(x, x + scale * 10.0 ** (15.0 / 20.0) * noise, fs).value  # -15 dB
    hi = ph.stoi(x, x + scale * 10.0 ** (-25.0 / 20.0) * noise, fs).value  # +25 dB
    return Outcome(
        expected="STOI(+25 dB) - STOI(-15 dB) > 0.2",
        computed=f"{hi - lo:.3f} ({lo:.3f} -> {hi:.3f})",
        delta="0",
        passed=(hi - lo) > 0.2,
    )


_NTA = "Impulsive-sound prominence (NT ACOU 112)"


@register(_NTA, "NT ACOU 112:2002 Formula 1", "Predicted prominence, OR=1000 dB/s, LD=30 dB")
def _chk_impulse_prominence() -> Outcome:
    value = float(ph.predicted_prominence(1000.0, 30.0))
    return numeric(ref.NTACOU112_PROMINENCE, value, 1e-4, places=4)


@register(_NTA, "NT ACOU 112:2002 Formula 2", "Adjustment KI to LAeq at prominence P=10")
def _chk_impulse_adjustment() -> Outcome:
    value = float(ph.impulse_adjustment(10.0))
    return numeric(ref.NTACOU112_ADJUSTMENT_P10, value, 1e-9, unit="dB", places=3)


_ISO1996_3 = "Impulsive-sound prominence (ISO/PAS 1996-3)"


def _iso1996_3_ramp_onset() -> Any:
    """Detected onset of a 30 dB LpAF ramp over 0.30 s (dt = 20 ms)."""
    import numpy as _np

    from phonometry.environmental.impulsive_sound import detect_onsets

    dt = 0.02
    pre = _np.full(round(0.2 / dt), 40.0)
    rise = 40.0 + 30.0 * (_np.arange(1, round(0.3 / dt) + 1) / round(0.3 / dt))
    post = _np.full(round(0.3 / dt), 70.0)
    return detect_onsets(_np.concatenate([pre, rise, post]), dt)[0]


@register(_ISO1996_3, "ISO/PAS 1996-3:2022 3.5", "Onset rate of a 30 dB ramp over 0.30 s")
def _chk_iso1996_3_onset_rate() -> Outcome:
    return numeric(ref.ISO1996_3_RAMP_ONSET_RATE, _iso1996_3_ramp_onset().onset_rate, 1e-6, unit="dB/s")


@register(_ISO1996_3, "ISO/PAS 1996-3:2022 Formula 3", "Adjustment KI of the ramp onset")
def _chk_iso1996_3_adjustment() -> Outcome:
    value = float(ph.impulse_adjustment(_iso1996_3_ramp_onset().prominence))
    return numeric(ref.ISO1996_3_RAMP_ADJUSTMENT, value, 1e-6, unit="dB", places=4)


_RN = "Room noise (ANSI S12.2-2019)"


@register(_RN, "ANSI S12.2-2019 Table 1", "NC-40 curve, tangency self-consistency")
def _chk_rn_nc_self() -> Outcome:
    rating = ph.noise_criterion(ph.nc_curve(40.0)).tangency_rating
    return numeric(ref.ANSIS12_2_NC40_SELF, rating, 1e-9, places=3)


@register(_RN, "ANSI S12.2-2019 Table D.1", "RC-31 Mark II curve, 63 Hz level")
def _chk_rn_rc_curve() -> Outcome:
    return numeric(ref.ANSIS12_2_RC31_63HZ, float(ph.rc_curve(31.0)[2]), 1e-9, places=3)


@register(_RN, "ANSI S12.2-2019 clause D.4", "RC-35 curve, mid-frequency average LMF")
def _chk_rn_rc_lmf() -> Outcome:
    return numeric(ref.ANSIS12_2_RC35_LMF, ph.room_criterion(ph.rc_curve(35.0)).lmf, 1e-9, places=3)


_HEAR = "Hearing threshold (ISO 7029 / ISO 389-7)"


@register(_HEAR, "ISO 7029:2017 Table 1", "Median threshold, male age 60 at 4 kHz")
def _chk_hearing_median() -> Outcome:
    value = float(ph.age_threshold(60, "male", 0.5).median[8])
    return numeric(ref.ISO7029_MEDIAN_MALE_60_4KHZ, value, 1e-3, unit="dB", places=3)


@register(_HEAR, "ISO 7029:2017 Table 2", "Upper spread su, male age 60 at 1 kHz")
def _chk_hearing_spread() -> Outcome:
    value = float(ph.age_threshold(60, "male", 0.5).spread_upper[4])
    return numeric(ref.ISO7029_SU_MALE_60_1KHZ, value, 1e-3, unit="dB", places=3)


@register(_HEAR, "ISO 389-7:2005 Table 1", "Free-field reference threshold at 1 kHz")
def _chk_hearing_reference() -> Outcome:
    value = float(ph.reference_threshold("free-field")[4])
    return numeric(ref.ISO389_7_REF_FREE_1KHZ, value, 1e-9, unit="dB", places=3)


_GUM = "Measurement uncertainty (GUM / Supplement 1)"


@register(_GUM, "ISO/IEC Guide 98-3-1 clause 9.2", "Combined uncertainty, additive model")
def _chk_gum_additive() -> Outcome:
    quantities = [ph.Quantity(0.0, 1.0) for _ in range(4)]
    result = ph.combine_uncertainty(lambda a, b, c, d: a + b + c + d, quantities)
    return numeric(ref.GUM_ADDITIVE_UC, result.combined_uncertainty, 1e-9, places=4)


@register(_GUM, "ISO/IEC Guide 98-3 Table G.2", "Coverage factor, p=0.99, v=16")
def _chk_gum_coverage() -> Outcome:
    from phonometry.metrology.uncertainty import coverage_factor

    return numeric(ref.GUM_COVERAGE_K99_16, coverage_factor(0.99, 16), 5e-3, places=3)


@register(_GUM, "ISO/IEC Guide 98-3 Annex G.4", "Welch-Satterthwaite effective dof")
def _chk_gum_welch() -> Outcome:
    quantities = [ph.Quantity(0.0, 1.0, dof=10) for _ in range(4)]
    result = ph.combine_uncertainty(lambda a, b, c, d: a + b + c + d, quantities)
    return numeric(ref.GUM_WELCH_VEFF, result.effective_dof, 1e-6, places=3)


def _gum_h1_result() -> Any:
    quantities = [ph.Quantity(v, unc, dof=dof) for v, unc, dof in ref.GUM_H1_INPUTS]
    with warnings.catch_warnings():
        # alphaS and theta are genuinely flat directions at the H.1 estimates.
        warnings.simplefilter("ignore")
        return ph.combine_uncertainty(
            lambda ls, d, a_s, th, da, dth: ls + d - ls * (da * th + a_s * dth),
            quantities,
        )


@register(_GUM, "ISO/IEC Guide 98-3 Annex H.1", "End-gauge combined uncertainty uc, nm")
def _chk_gum_h1_uc() -> Outcome:
    result = _gum_h1_result()
    return numeric(ref.GUM_H1_UC, result.combined_uncertainty, 0.01, unit="nm", places=2)


@register(_GUM, "ISO/IEC Guide 98-3 Annex H.1", "End-gauge expanded uncertainty U99, nm")
def _chk_gum_h1_u99() -> Outcome:
    result = _gum_h1_result()
    _, big = result.expanded(0.99)
    return numeric(ref.GUM_H1_U99, big, 0.1, unit="nm", places=1)


@register(
    _GUM,
    "ISO/IEC Guide 98-3 Annex H.2 (Table H.3)",
    "Correlated V/I/phi budget: uc(R), ohm",
)
def _chk_gum_h2_correlated() -> Outcome:
    obs = np.array(ref.GUM_H2_OBSERVATIONS)
    obs[:, 1] *= 1e-3  # mA -> A
    means = obs.mean(axis=0)
    u_means = obs.std(axis=0, ddof=1) / math.sqrt(obs.shape[0])
    r = np.corrcoef(obs.T)
    quantities = [ph.Quantity(m, s) for m, s in zip(means, u_means)]
    result = ph.combine_uncertainty(
        lambda v, i, p: v / i * math.cos(p), quantities, correlation=r
    )
    return numeric(
        ref.GUM_H2_RESULTS["R"][1], result.combined_uncertainty, 1e-3,
        unit="ohm", places=3,
    )


@register(
    _GUM,
    "ISO/IEC Guide 98-3-1 Table 3 (clause 9.2.3)",
    "Seeded Monte Carlo, rectangular sum: 95 % interval endpoint",
)
def _chk_gum_s1_table3_monte_carlo() -> Outcome:
    quantities = [ph.Quantity(0.0, 1.0, "rectangular") for _ in range(4)]
    mc = ph.monte_carlo(
        lambda a, b, c, d: a + b + c + d, quantities,
        trials=1_000_000, coverage=0.95, seed=1996,
    )
    endpoint = 0.5 * (mc.interval[1] - mc.interval[0])
    ok_u = abs(mc.standard_uncertainty - ref.GUMS1_TABLE3_U) <= 0.01
    outcome = numeric(ref.GUMS1_TABLE3_INTERVAL_95, endpoint, 0.03, places=3)
    return Outcome(
        expected=f"+/-{ref.GUMS1_TABLE3_INTERVAL_95} (u = {ref.GUMS1_TABLE3_U})",
        computed=f"+/-{endpoint:.3f} (u = {mc.standard_uncertainty:.3f})",
        delta=outcome.delta,
        passed=outcome.passed and ok_u,
    )


_NIHL = "Noise-induced hearing loss (ISO 1999)"


@register(_NIHL, "ISO 1999:2013 Table D.2", "Median NIPTS, 4 kHz, 90 dB, 20 yr")
def _chk_nihl_median() -> Outcome:
    value = float(ph.nipts(90.0, 20.0, 0.5).value[4])
    return numeric(ref.ISO1999_N50_4K_90_20, value, 0.5, unit="dB", places=1)


@register(_NIHL, "ISO 1999:2013 Table D.2", "Worst-10 % NIPTS, 4 kHz, 90 dB, 20 yr")
def _chk_nihl_fractile() -> Outcome:
    value = float(ph.nipts(90.0, 20.0, 0.9).value[4])
    return numeric(ref.ISO1999_N10_4K_90_20, value, 0.5, unit="dB", places=1)


@register(_NIHL, "ISO 1999:2013 Table D.4", "Worst-10 % NIPTS, 3 kHz, 100 dB, 40 yr")
def _chk_nihl_high() -> Outcome:
    value = float(ph.nipts(100.0, 40.0, 0.9).value[3])
    return numeric(ref.ISO1999_N10_3K_100_40, value, 0.5, unit="dB", places=1)


@register(
    _NIHL,
    "ISO 1999:2013 Annex C, Formulae (C.6) to (C.8)",
    "NIPTS at 1/2/4 kHz, 90 dB, 30 yr, Q = 10 % (annex inputs)",
)
def _chk_nihl_annex_c_nipts() -> Outcome:
    """The Table D.2 shifts the Annex C example takes as its noise input."""
    value = np.round(
        ph.nipts(90.0, 30.0, 0.9, frequencies=[1000.0, 2000.0, 4000.0]).value
    )
    expected = np.asarray(ref.ISO1999_ANNEX_C_N, dtype=float)
    delta = float(np.max(np.abs(value - expected)))
    return Outcome(
        expected=", ".join(f"{v:.0f}" for v in expected) + " dB",
        computed=", ".join(f"{v:.0f}" for v in value) + " dB",
        delta=f"{delta:.0f} dB",
        passed=delta <= 0.0,
    )


@register(
    _NIHL,
    "ISO 1999:2013 Annex C, Formula (C.5)",
    "Compressed 4 kHz shift, Formula (1) with the annex's H = 36 dB",
)
def _chk_nihl_annex_c_compression() -> Outcome:
    """The annex reduces the 19 dB shift at 4 kHz by the H*N/120 term."""
    h = ref.ISO1999_ANNEX_C_H[2]
    n = ref.ISO1999_ANNEX_C_N[2]
    # H' - H is what Formula (1) leaves of the noise component at that band.
    value = float(ph.combine_age_and_noise(h, n)) - h
    return numeric(
        ref.ISO1999_ANNEX_C_N_4K_COMPRESSED, value, 0.05, unit="dB", places=1
    )


@register(
    _NIHL,
    "ISO 1999:2013 Annex C, Formula (C.11)",
    "Hearing threshold level with age and noise, 1/2/4 kHz mean, Q = 10 %",
)
def _chk_nihl_annex_c_htlan() -> Outcome:
    """The end of the annex chain: the mean H plus the mean compressed shift.

    The age component is the annex's own Table A.3 selection rather than the
    library's ISO 7029:2017 evaluation, so this exercises the Formula (1)
    combination on the standard's stated inputs (see the source note the
    ISO 1999 fiche prints). The annex applies the compression term only where
    it matters, taking the shift straight from Table D.2 "when (H + N) < 40 dB"
    (its Formula (C.4) approximation); here that leaves only the 4 kHz band
    compressed.
    """
    h = np.asarray(ref.ISO1999_ANNEX_C_H, dtype=float)
    n = np.asarray(ref.ISO1999_ANNEX_C_N, dtype=float)
    compressed = ph.combine_age_and_noise(h, n) - h
    noise = np.where(h + n > ref.ISO1999_ANNEX_C_COMPRESSION_FENCE, compressed, n)
    value = float(np.mean(h) + np.mean(noise))
    return numeric(ref.ISO1999_ANNEX_C_HTLAN, value, 0.05, unit="dB", places=1)


_MSV = "Multiple-shock whole-body vibration (ISO 2631-5)"


@register(_MSV, "ISO 2631-5:2018 Formula 3", "Daily acceleration dose, 5 x 40 m/s2 peaks")
def _chk_multiple_shock_dose() -> Outcome:
    value = ph.dose_from_peaks([40.0] * 5)
    return numeric(ref.ISO2631_5_DZD_MALE, value, 0.01, unit="m/s2", places=2)


@register(_MSV, "ISO 2631-5:2018 Formula C.3", "Stress variable R, Annex C male example")
def _chk_multiple_shock_risk() -> Outcome:
    sd = ph.compression_dose(ph.dose_from_peaks([40.0] * 5))
    value = ph.injury_risk(sd, start_age=20, years=20, days_per_year=120, sex="male")
    return numeric(ref.ISO2631_5_R_MALE, value, 0.01, places=2)


@register(_MSV, "ISO 2631-5:2018 Formula C.5", "Injury probability, Annex C male example")
def _chk_multiple_shock_probability() -> Outcome:
    sd = ph.compression_dose(ph.dose_from_peaks([40.0] * 5))
    r = ph.injury_risk(sd, start_age=20, years=20, days_per_year=120, sex="male")
    return numeric(ref.ISO2631_5_PI_MALE, float(ph.injury_probability(r)), 0.01, places=2)


@register(
    _MSV, "ISO 2631-5:2018 Annex C NOTE 5", "Compressive stress Sd, female example"
)
def _chk_multiple_shock_female_sd() -> Outcome:
    from phonometry.vibration.multiple_shock_vibration import MZ_FEMALE

    sd = ph.compression_dose(ph.dose_from_peaks([40.0] * 5), mz=MZ_FEMALE)
    return numeric(ref.ISO2631_5_SD_FEMALE, sd, 0.01, unit="MPa", places=2)


@register(
    _MSV, "ISO 2631-5:2018 Annex C NOTE 5", "Stress variable R, female example"
)
def _chk_multiple_shock_female_r() -> Outcome:
    from phonometry.vibration.multiple_shock_vibration import MZ_FEMALE

    sd = ph.compression_dose(ph.dose_from_peaks([40.0] * 5), mz=MZ_FEMALE)
    r = ph.injury_risk(sd, start_age=20, years=20, days_per_year=120, sex="female")
    return numeric(ref.ISO2631_5_R_FEMALE, r, 0.01, places=2)


@register(
    _MSV,
    "ISO 2631-5:2018 Formula 1 vs Annex D Table D.1",
    "Seat-to-spine transfer vs the 256 Hz digital filter (0,5-80 Hz)",
)
def _chk_multiple_shock_annex_d_filter() -> Outcome:
    freqs = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0])
    formula = np.abs(ph.seat_to_spine_transfer(freqs))
    _, h = sg.freqz(
        ref.ISO2631_5_ANNEX_D_B,
        ref.ISO2631_5_ANNEX_D_A,
        worN=2.0 * np.pi * freqs / ref.ISO2631_5_ANNEX_D_FS,
    )
    worst = float(np.max(np.abs(formula - np.abs(h))))
    return numeric(
        0.0, worst, 0.04, places=3,
        expected_label="max abs(Formula 1 - filter) ≤ 0,04",
    )


_ABS = "Sound absorption in enclosed spaces (EN 12354-6)"


@register(_ABS, "EN 12354-6:2003 Formula 1", "Equivalent absorption area, Annex E bare room")
def _chk_enclosed_space_area() -> Outcome:
    value = float(ph.equivalent_absorption_area(ref.EN12354_6_ANNEX_E_BARE_SURFACES))
    return numeric(ref.EN12354_6_A_BARE, value, 0.01, unit="m2", places=2)


@register(_ABS, "EN 12354-6:2003 Formula 5", "Reverberation time, Annex E bare room")
def _chk_enclosed_space_rt() -> Outcome:
    area = ph.equivalent_absorption_area(ref.EN12354_6_ANNEX_E_BARE_SURFACES)
    value = float(ph.reverberation_time(area, ref.EN12354_6_ANNEX_E_VOLUME))
    return numeric(ref.EN12354_6_T_BARE, value, 0.05, unit="s", places=1)


_TONES = "Prominent discrete tones (ECMA-418-1)"


@register(_TONES, "ECMA-418-1:2024 Clause 10 Formula (2)", "Critical band at 1 kHz (f1,c / f2,c / dfc)")
def _chk_ecma418_1_critical_band() -> Outcome:
    from phonometry.psychoacoustics.tonality import _critical_band

    f1, f2, dfc = _critical_band(1000.0)
    # 0.05 Hz = half a unit in the last printed digit (the clause EXAMPLE
    # values are given to one decimal: 922,2 / 1084,4 / 162,2 Hz).
    out = numeric(ref.ECMA418_1_DFC_1KHZ, float(dfc), 0.05, unit="Hz", places=2)
    edges_ok = (
        abs(float(f1) - ref.ECMA418_1_F1_1KHZ) <= 0.05
        and abs(float(f2) - ref.ECMA418_1_F2_1KHZ) <= 0.05
    )
    return Outcome(
        expected=(
            f"dfc {out.expected}; edges {ref.ECMA418_1_F1_1KHZ:g}"
            f"-{ref.ECMA418_1_F2_1KHZ:g} Hz"
        ),
        computed=f"dfc {out.computed}; edges {float(f1):.1f}-{float(f2):.1f} Hz",
        delta=out.delta,
        passed=out.passed and edges_ok,
    )


@register(_TONES, "ECMA-418-1:2024 Clause 11.6 Formula (14)", "Proximity spacing dfprox at 150 / 850 Hz")
def _chk_ecma418_1_proximity_spacing() -> Outcome:
    from phonometry.psychoacoustics.tonality import _proximity_spacing

    v150 = float(_proximity_spacing(150.0))
    v850 = float(_proximity_spacing(850.0))
    # 0.5 Hz = half a unit in the last printed digit of the coarser EXAMPLE
    # value (the standard prints 23 Hz with no decimals; 63,8 Hz with one).
    ok = (
        abs(v150 - ref.ECMA418_1_PROX_150HZ) <= 0.5
        and abs(v850 - ref.ECMA418_1_PROX_850HZ) <= 0.5
    )
    return Outcome(
        expected=(
            f"{ref.ECMA418_1_PROX_150HZ:g} Hz @ 150 Hz; "
            f"{ref.ECMA418_1_PROX_850HZ:g} Hz @ 850 Hz (+/-0.5 Hz)"
        ),
        computed=f"{v150:.1f} Hz; {v850:.1f} Hz",
        delta=(
            f"{v150 - ref.ECMA418_1_PROX_150HZ:+.3f}; "
            f"{v850 - ref.ECMA418_1_PROX_850HZ:+.3f} Hz"
        ),
        passed=ok,
    )


_TONE_AUD = "Tonal audibility (ISO/PAS 20065)"


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formulae (12)-(14)", "Audibility at 137.3 Hz, Annex E spectrum 1")
def _chk_iso20065_audibility() -> Outcome:
    fT, ls, lt, expected = ref.ISO20065_ANNEX_E_TONES[1]  # 137.3 Hz tone
    value = ph.tone_audibility(lt, ls, fT, ref.ISO20065_LINE_SPACING)
    # 0.05 dB absorbs the standard's 2-decimal table rounding of LS/LT/LG/av.
    return numeric(expected, value, 0.05, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (13)", "Masking index av at 137.3 / 592.2 Hz")
def _chk_iso20065_masking_index() -> Outcome:
    av137 = ph.masking_index(137.3)
    av592 = ph.masking_index(592.2)
    ok = (
        abs(av137 - ref.ISO20065_AV_137) <= 0.005
        and abs(av592 - ref.ISO20065_AV_592) <= 0.005
    )
    return Outcome(
        expected=f"{ref.ISO20065_AV_137:g} dB @ 137.3 Hz; "
        f"{ref.ISO20065_AV_592:g} dB @ 592.2 Hz (+/-0.005 dB)",
        computed=f"{av137:.3f} dB; {av592:.3f} dB",
        delta=f"{av137 - ref.ISO20065_AV_137:+.3f}; "
        f"{av592 - ref.ISO20065_AV_592:+.3f} dB",
        passed=ok,
    )


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (20)", "Mean audibility of the five spectra, Annex E")
def _chk_iso20065_mean_audibility() -> Outcome:
    value = ph.mean_audibility(ref.ISO20065_DECISIVE_AUDIBILITIES)
    # 0.05 dB absorbs the 2-decimal rounding of the tabulated decisive values.
    return numeric(ref.ISO20065_MEAN_AUDIBILITY, value, 0.05, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (6)", "Mean narrow-band level LS from spectrum, Table E.1")
def _chk_iso20065_mean_narrowband_level() -> Outcome:
    value = ph.mean_narrowband_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3
    )
    # Iterative Formula (6) with Hanning correction; 0.02 dB absorbs rounding.
    return numeric(ref.ISO20065_E1_LS, value, 0.02, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Clause 6", "Extended uncertainty U of the 137.3 Hz tone, Table E.2")
def _chk_iso20065_uncertainty() -> Outcome:
    res = ph.analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    assert res.extended_uncertainties is not None
    by_freq = dict(zip(res.tone_frequencies, res.extended_uncertainties))
    # Table E.2, run index k = 2: U = 2.79 dB (90 % bilateral coverage).
    return numeric(ref.ISO20065_E2_U[1], float(by_freq[137.3]), 0.02, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formulae (28)-(29)", "Extended uncertainty of the mean audibility, Annex E Step 4")
def _chk_iso20065_mean_uncertainty() -> Outcome:
    u_j = [row[6] for row in ref.ISO20065_E4_DECISIVE_ROWS]
    value = ph.mean_audibility_uncertainty(ref.ISO20065_DECISIVE_AUDIBILITIES, u_j)
    return numeric(
        ref.ISO20065_E4_MEAN_UNCERTAINTY, value, 0.01, unit="dB", places=2
    )


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (8)", "Tone level LT from spectrum, Table E.1")
def _chk_iso20065_tone_level() -> Outcome:
    ls = ph.mean_narrowband_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3
    )
    value = ph.tone_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3, ls
    )
    return numeric(ref.ISO20065_E1_LT, value, 0.02, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Clause 5.3.8", "Tone detection over the spectrum, Table E.1")
def _chk_iso20065_peak_detection() -> Outcome:
    result = ph.analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    assert result.group_sizes is not None
    singles = result.group_sizes == 1
    found = sorted(round(float(f), 1) for f in result.tone_frequencies[singles])
    expected = sorted(ref.ISO20065_E1_TONE_FREQUENCIES)
    ok = found == expected
    return Outcome(
        expected=f"tones at {expected} Hz",
        computed=f"tones at {found} Hz",
        delta="exact" if ok else "mismatch",
        passed=ok,
    )


@register(
    _TONE_AUD,
    "ISO/PAS 20065:2016 Clause 5.3.8 Step 3",
    "Same-band FG combination inside analyze_spectrum, Table E.2 row 2 FG",
)
def _chk_iso20065_step3_fg() -> Outcome:
    result = ph.analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    assert result.group_sizes is not None
    fg = result.group_sizes > 1
    value = float(result.tone_levels[fg][0]) if int(fg.sum()) == 1 else float("nan")
    return numeric(ref.ISO20065_E1_LT_FG, value, 0.02, unit="dB", places=2)


@register(_TONE_AUD, "ISO/PAS 20065:2016 Formula (17)", "Multi-tone FG combination, Table E.1")
def _chk_iso20065_fg_combination() -> Outcome:
    value = ph.combined_tone_level(
        ref.ISO20065_E1_LEVELS,
        ref.ISO20065_E1_FREQUENCIES,
        ref.ISO20065_E1_TONE_FREQUENCIES,
        ref.ISO20065_E1_TONE_LS,
    )
    return numeric(ref.ISO20065_E1_LT_FG, value, 0.02, unit="dB", places=2)


@register(
    _TONE_AUD,
    "ISO/PAS 20065:2016 Formulae (18)/(19)",
    "Two-tone separation fD (DIN 45681 Annex J), 137.3 / 212 Hz",
)
def _chk_iso20065_two_tone_separation() -> Outcome:
    fd_137 = ph.two_tone_separation_frequency(137.3)
    fd_212 = ph.two_tone_separation_frequency(212.0)
    # Annex E consistency: the 118.4/137.3 Hz pair is combined, not separated
    # (|Δf| = 18.9 Hz < fD ≈ 24 Hz at the more prominent tone).
    annex_e_combined = not ph.resolve_tones_separately(118.4, 137.3, 4.0, 5.0)
    ok = (
        round(fd_137, 2) == ref.ISO20065_FD_137
        and round(fd_212, 2) == ref.ISO20065_FD_212
        and annex_e_combined
    )
    return Outcome(
        expected=(
            f"fD(137.3)={ref.ISO20065_FD_137}, fD(212)={ref.ISO20065_FD_212} Hz; "
            "Annex E pair combined"
        ),
        computed=(
            f"fD(137.3)={fd_137:.2f}, fD(212)={fd_212:.2f} Hz; "
            f"Annex E pair {'combined' if annex_e_combined else 'separated'}"
        ),
        delta="exact" if ok else "mismatch",
        passed=ok,
    )


# ===========================================================================
# Psychoacoustic annoyance & fluctuation strength (Fastl & Zwicker)
# ===========================================================================
_PA_FS = "Psychoacoustic annoyance & fluctuation strength (Fastl & Zwicker)"


@register(
    _PA_FS,
    "Fastl & Zwicker Eqs (16.2)-(16.4)",
    "Psychoacoustic annoyance, worked (N5,S,F,R) tuple",
)
def _chk_psychoacoustic_annoyance() -> Outcome:
    n5, s, f, r = ref.PA_WORKED_INPUT
    value = ph.psychoacoustic_annoyance(n5, s, f, r).annoyance
    return numeric(ref.PA_WORKED_VALUE, value, 1e-3, places=4)


@register(
    _PA_FS,
    "Fastl & Zwicker Eq (10.2)",
    "Fluctuation strength of AM broadband noise (60 dB, m=1, 4 Hz)",
)
def _chk_fluctuation_strength_am_noise() -> Outcome:
    value = ph.fluctuation_strength_am_noise(60.0, 1.0, 4.0)
    return numeric(ref.FS_BBN_60_1_4, value, 1e-3, unit="vacil", places=4)


@register(
    _PA_FS,
    "Fastl & Zwicker Ch. 10 / Osses et al. 2016",
    "Fluctuation-strength calibration: 1 kHz / 60 dB / m=1 / 4 Hz AM tone",
)
def _chk_fluctuation_strength_calibration() -> Outcome:
    # The signal model is anchored (clean-room) so the 1-vacil reference tone
    # returns 1.00 vacil through its own front-end. No numeric standard exists.
    fs = 48000
    t = np.arange(int(fs * 2.0)) / fs
    tone = (1.0 + np.sin(2.0 * np.pi * 4.0 * t)) * np.sin(2.0 * np.pi * 1000.0 * t)
    tone = tone / np.sqrt(np.mean(tone**2)) * 2e-5 * 10.0 ** (60.0 / 20.0)
    value = ph.fluctuation_strength(tone, fs).fluctuation_strength
    return numeric(
        ref.FS_CALIBRATION_VACIL, value, 0.05, unit="vacil", places=3
    )


# ===========================================================================
# Electroacoustics: distortion & frequency response (IEC 60268-3 / Bendat)
# ===========================================================================
_ELECTRO = "Electroacoustics: distortion & frequency response"


def _electro_fs() -> int:
    return 48000


def _electro_tone(t: np.ndarray, freq: float, amp: float) -> np.ndarray:
    """A single sine of amplitude ``amp`` at ``freq`` over the time base ``t``."""
    return amp * np.sin(2.0 * np.pi * freq * t)


def _electro_harmonic_signal() -> np.ndarray:
    fs = _electro_fs()
    t = np.arange(fs) / fs  # 1 s, tones on integer bins
    a1, a2, a3, a4 = ref.DISTORTION_HARMONICS
    sig = (
        _electro_tone(t, 1000.0, a1)
        + _electro_tone(t, 2000.0, a2)
        + _electro_tone(t, 3000.0, a3)
        + _electro_tone(t, 4000.0, a4)
    )
    return np.asarray(sig, dtype=np.float64)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.3.2)",
    "THD (rel. total RMS, the R convention the clause defines)",
)
def _chk_thd_r() -> Outcome:
    value = ph.thd(_electro_harmonic_signal(), _electro_fs(), 1000.0, kind="R")
    return numeric(ref.DISTORTION_THD_R, value, 1e-4, places=6)


@register(
    _ELECTRO,
    "Closed-form harmonic synthesis (THD_F convention)",
    "THD (rel. fundamental, the widespread datasheet convention)",
)
def _chk_thd_f() -> Outcome:
    value = ph.thd(_electro_harmonic_signal(), _electro_fs(), 1000.0, kind="F")
    return numeric(ref.DISTORTION_THD_F, value, 1e-4, places=6)


def _loudspeaker_flat_response() -> tuple[np.ndarray, np.ndarray]:
    """Flat 90 dB on-axis response with ramps crossing 80 dB at 50/18000 Hz."""
    f = np.geomspace(20.0, 20000.0, 400)
    spl = np.full_like(f, 90.0)
    below = f < 80.0
    spl[below] = 90.0 - 10.0 * (np.log2(80.0 / f[below]) / np.log2(80.0 / 50.0))
    above = f > 15000.0
    spl[above] = 90.0 - 10.0 * (np.log2(f[above] / 15000.0) / np.log2(18000.0 / 15000.0))
    return f, spl


@register(
    _ELECTRO,
    "IEC 60268-5:2003 (20.3/20.4)",
    "Characteristic sensitivity level, 1 W into 8 ohm at 1 m (flat 90 dB)",
)
def _chk_loudspeaker_sensitivity() -> Outcome:
    f, spl = _loudspeaker_flat_response()
    result = ph.loudspeaker_characteristics(f, spl, 8.0, sensitivity_band=(200.0, 4000.0))
    # A flat 90 dB response driven at sqrt(8) V (1 W) at 1 m has a
    # characteristic sensitivity level of 90 dB exactly (the corrections vanish).
    return numeric(90.0, result.sensitivity_level_db, 1e-6, unit="dB", places=6)


@register(
    _ELECTRO,
    "IEC 60268-5:2003 (21.2)",
    "Effective frequency range = -10 dB crossings (50 Hz / 18 kHz)",
)
def _chk_loudspeaker_effective_range() -> Outcome:
    f, spl = _loudspeaker_flat_response()
    result = ph.loudspeaker_characteristics(f, spl, 8.0, sensitivity_band=(200.0, 4000.0))
    lo, hi = result.effective_range
    ok = abs(lo - 50.0) <= 5e-3 and abs(hi - 18000.0) <= 2.0
    return Outcome(
        expected="50 Hz / 18000 Hz (ref -10 dB crossings)",
        computed=f"{lo:.3f} Hz / {hi:.1f} Hz",
        delta=f"{lo - 50.0:.3f} / {hi - 18000.0:.3f} Hz",
        passed=ok,
    )


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.5)",
    "2nd-order harmonic distortion d2 (rel. total)",
)
def _chk_harmonic_d2() -> Outcome:
    value = ph.harmonic_distortion(_electro_harmonic_signal(), _electro_fs(), 1000.0, 2)
    return numeric(ref.DISTORTION_D2, value, 1e-4, places=6)


def _microphone_flat_response() -> tuple[np.ndarray, np.ndarray]:
    """Flat 0 dB relative response with ramps crossing -3 dB at 40/18000 Hz."""
    f = np.geomspace(20.0, 20000.0, 400)
    rel = np.zeros_like(f)
    below = f < 63.0
    rel[below] = -3.0 * (np.log2(63.0 / f[below]) / np.log2(63.0 / 40.0))
    above = f > 15000.0
    rel[above] = -3.0 * (np.log2(f[above] / 15000.0) / np.log2(18000.0 / 15000.0))
    return f, rel


@register(
    _ELECTRO,
    "IEC 60268-4:2014 (11.1/11.3)",
    "Microphone sensitivity level, 12.5 mV/Pa -> 20 lg 0.0125 dB re 1 V/Pa",
)
def _chk_microphone_sensitivity_level() -> Outcome:
    f, rel = _microphone_flat_response()
    result = ph.microphone_characteristics(f, rel, 12.5, tolerance_db=3.0)
    # Hand-computed: 20 lg 0.0125 = -38.061800 dB re 1 V/Pa.
    return numeric(-38.061800, result.sensitivity_level_db, 1e-5, unit="dB", places=6)


@register(
    _ELECTRO,
    "IEC 60268-4:2014 (12.2)",
    "Effective frequency range = +/-3 dB tolerance crossings (40 Hz / 18 kHz)",
)
def _chk_microphone_effective_range() -> Outcome:
    f, rel = _microphone_flat_response()
    result = ph.microphone_characteristics(f, rel, 12.5, tolerance_db=3.0)
    lo, hi = result.effective_range
    ok = abs(lo - 40.0) <= 5e-3 and abs(hi - 18000.0) <= 2.0
    return Outcome(
        expected="40 Hz / 18000 Hz (+/-3 dB tolerance crossings)",
        computed=f"{lo:.3f} Hz / {hi:.1f} Hz",
        delta=f"{lo - 40.0:.3f} / {hi - 18000.0:.3f} Hz",
        passed=ok,
    )


@register(
    _ELECTRO,
    "IEC 60268-4:2014 (13.2.2)",
    "Directivity index of the ideal cardioid, 10 lg 3 dB (11.2.2 a integral)",
)
def _chk_microphone_cardioid_di() -> Outcome:
    f, rel = _microphone_flat_response()
    angles = np.linspace(0.0, 179.9, 1800)
    pattern = 20.0 * np.log10((1.0 + np.cos(np.radians(angles))) / 2.0)
    result = ph.microphone_characteristics(
        f, rel, 12.5, tolerance_db=3.0, polar=(angles, pattern)
    )
    di = result.directivity_index_db
    if di is None:
        return Outcome(
            expected="4.771213 dB", computed="None", delta="n/a", passed=False
        )
    # Closed form: D = 10 lg 3 = 4.771213 dB.
    return numeric(4.771213, di, 5e-3, unit="dB", places=6)


@register(
    _ELECTRO,
    "IEC 60268-4:2014 (17.2)",
    "Equivalent noise level, 2.5 uV over 12.5 mV/Pa -> 200 uPa = 20 dB SPL",
)
def _chk_microphone_equivalent_noise() -> Outcome:
    f, rel = _microphone_flat_response()
    result = ph.microphone_characteristics(
        f, rel, 12.5, tolerance_db=3.0, noise_voltage=2.5e-6
    )
    noise = result.equivalent_noise_level_db
    if noise is None:
        return Outcome(
            expected="20 dB SPL", computed="None", delta="n/a", passed=False
        )
    return numeric(20.0, noise, 1e-9, unit="dB SPL", places=6)


def _electro_smpte_signal() -> tuple[np.ndarray, float, float]:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    fl, fh = 250.0, 8000.0
    x = (
        _electro_tone(t, fl, 1.0)
        + _electro_tone(t, fh, 0.25)
        + _electro_tone(t, fh + fl, 0.02)
        + _electro_tone(t, fh - fl, 0.02)
        + _electro_tone(t, fh + 2 * fl, 0.01)
        + _electro_tone(t, fh - 2 * fl, 0.01)
    )
    return x, fl, fh


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.7.2 g)",
    "Modulation distortion d_m,2 (arithmetic sideband sum over U_2,f2)",
)
def _chk_modulation_d2() -> Outcome:
    x, fl, fh = _electro_smpte_signal()
    # Sidebands 0.02 + 0.02 over the 0.25 carrier: d_m,2 = 0.16 exactly.
    value = ph.modulation_distortion(x, _electro_fs(), fl, fh).d2
    return numeric(0.16, value, 1e-4, places=6)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.7.2 h)",
    "Modulation distortion d_m,3 (arithmetic sideband sum over U_2,f2)",
)
def _chk_modulation_d3() -> Outcome:
    x, fl, fh = _electro_smpte_signal()
    # Sidebands 0.01 + 0.01 over the 0.25 carrier: d_m,3 = 0.08 exactly.
    value = ph.modulation_distortion(x, _electro_fs(), fl, fh).d3
    return numeric(0.08, value, 1e-4, places=6)


def _electro_dfd_signal() -> tuple[np.ndarray, float, float]:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    f1, f2 = 13000.0, 14000.0
    x = (
        _electro_tone(t, f1, 0.5)
        + _electro_tone(t, f2, 0.5)
        + _electro_tone(t, f2 - f1, 0.03)
        + _electro_tone(t, 2 * f1 - f2, 0.02)
        + _electro_tone(t, 2 * f2 - f1, 0.02)
    )
    return x, f1, f2


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.8.1 a)",
    "Difference-frequency distortion d_d,2 (over U_2,ref = 2 U_2,f2)",
)
def _chk_dfd_d2() -> Outcome:
    x, f1, f2 = _electro_dfd_signal()
    # Product 0.03 over the tone-amplitude sum 1.0: d_d,2 = 0.03 exactly.
    value = ph.difference_frequency_distortion(x, _electro_fs(), f1, f2, order=2)
    return numeric(0.03, value, 1e-4, places=6)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.8.1 b)",
    "Difference-frequency distortion d_d,3 (arithmetic product sum)",
)
def _chk_dfd_d3() -> Outcome:
    x, f1, f2 = _electro_dfd_signal()
    # Products 0.02 + 0.02 over the tone-amplitude sum 1.0: d_d,3 = 0.04.
    value = ph.difference_frequency_distortion(x, _electro_fs(), f1, f2, order=3)
    return numeric(0.04, value, 1e-4, places=6)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.10)",
    "Total difference-frequency distortion (8 kHz / 11.95 kHz tones)",
)
def _chk_tdfd() -> Outcome:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    f1, f2 = 8000.0, 11950.0
    x = (
        _electro_tone(t, f1, 0.5)
        + _electro_tone(t, f2, 0.5)
        + _electro_tone(t, f2 - f1, 0.02)
        + _electro_tone(t, 2 * f1 - f2, 0.03)
    )
    # Only the in-band products at f0 -/+ delta (3950/4050 Hz) enter:
    # d_TDFD = sqrt(0.02^2 + 0.03^2) / (0.5 + 0.5) = sqrt(0.0013).
    value = ph.total_difference_frequency_distortion(x, fs)
    return numeric(0.03605551275463989, value, 1e-4, places=8)


@register(
    _ELECTRO,
    "ITU-R BS.468-4 Table 1",
    "Weighting network response at the 6.3 kHz peak (14.12.11 network)",
)
def _chk_itu_468_peak() -> Outcome:
    value = float(ph.itu_r_468_weighting([6300.0])[0])
    return numeric(12.2, value, 1e-9, unit="dB", places=2)


@register(
    _ELECTRO,
    "IEC 60268-3:2013 (14.12.9)",
    "DIM of the 15 kHz / 3.15 kHz signal (Table 2, 9 products)",
)
def _chk_dim() -> Outcome:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    fsine, fsq = 15000.0, 3150.0
    comps = sorted(
        round(abs(k * fsq - fsine), 6) for k in range(1, 10) if abs(k * fsq - fsine) < fsine
    )
    amps = [0.01 * (i + 1) for i in range(len(comps))]
    # 15 kHz sine + the strong 3.15 kHz fundamental + the nine products.
    x = _electro_tone(t, fsine, 1.0) + _electro_tone(t, fsq, 0.8)
    for c, a in zip(comps, amps):
        x = x + _electro_tone(t, c, a)
    expected = math.sqrt(sum(a**2 for a in amps))
    return numeric(expected, ph.dynamic_intermodulation_distortion(x, fs), 1e-4, places=6)


@register(
    _ELECTRO,
    "Bendat & Piersol, Random Data 4e",
    "H1 recovers a known first-order IIR gain at 1 kHz",
)
def _chk_h1_gain() -> Outcome:
    fs = _electro_fs()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200000)
    b, a = sg.butter(1, 2000.0 / (fs / 2.0), btype="low")
    y = sg.lfilter(b, a, x)
    res = ph.transfer_function(x, y, fs, estimator="H1")
    _, h = sg.freqz(b, a, worN=res.frequencies, fs=fs)
    idx = int(np.argmin(np.abs(res.frequencies - 1000.0)))
    return numeric(
        float(np.abs(h[idx])), float(np.abs(res.response[idx])), 0.02, rel=True, places=4
    )


@register(
    _ELECTRO,
    "Bendat & Piersol, Random Data 4e",
    "Ordinary coherence = 1 for a noiseless LTI path",
)
def _chk_coherence_unity() -> Outcome:
    fs = _electro_fs()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200000)
    b, a = sg.butter(1, 2000.0 / (fs / 2.0), btype="low")
    y = sg.lfilter(b, a, x)
    f, g = ph.coherence(x, y, fs)
    band = (f > 100.0) & (f < 5000.0)
    return numeric(1.0, float(np.mean(g[band])), 1e-3, places=6)


@register(
    _ELECTRO,
    "AES17-2015 (6.4.2 / 5.2.7)",
    "Idle channel noise, 1 kHz -20 dBFS tone (CCIR-RMS -5.63 dB offset)",
)
def _chk_aes17_idle_noise() -> Outcome:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    # 468 is 0 dB at 1 kHz, so CCIR-RMS reads -5.63 dB there: a -20 dBFS tone
    # measures -25.63 dBFS CCIR-RMS in closed form.
    sig = _electro_tone(t, 1000.0, 10.0 ** (-20.0 / 20.0))
    return numeric(-25.63, ph.idle_channel_noise(sig, fs), 1e-2, unit="dB", places=2)


@register(
    _ELECTRO,
    "AES17-2015 (6.4.1)",
    "Dynamic range, full-scale reference over a -40 dBFS residual at 2 kHz",
)
def _chk_aes17_dynamic_range() -> Outcome:
    fs = _electro_fs()
    t = np.arange(fs) / fs
    # 997 Hz test tone at -60 dBFS plus a lone 2 kHz residual at -40 dBFS: the
    # CCIR-RMS filter is unity at 2 kHz and the 997 Hz notch is negligible
    # there, so the ratio of the full-scale sine to the residual is ~40 dB
    # (a small notch lift aside).
    sig = _electro_tone(t, 997.0, 10.0 ** (-60.0 / 20.0))
    sig = sig + _electro_tone(t, 2000.0, 10.0 ** (-40.0 / 20.0))
    return numeric(40.0, ph.dynamic_range(sig, fs, 997.0), 0.6, unit="dB", places=2)


# ===========================================================================
# Calibrated spectral analysis (Bendat & Piersol, Random Data 4e)
# ===========================================================================
_SPECTRA = "Calibrated spectral analysis (Bendat & Piersol)"


def _spectra_fs() -> float:
    return 8192.0


def _spectra_white(seed: int, rms: float = 1.0) -> np.ndarray:
    return ph.noise_signal(_spectra_fs(), 4.0, color="white", rms=rms, seed=seed)


@register(
    _SPECTRA,
    "Bendat & Piersol, Random Data 4e Eq. (5.67)",
    "White-noise autospectral density = sigma^2/(fs/2)",
)
def _chk_psd_white_level() -> Outcome:
    fs = _spectra_fs()
    res = ph.power_spectral_density(_spectra_white(1, rms=2.0), fs, nperseg=1024)
    band = (res.frequencies > 200.0) & (res.frequencies < 3800.0)
    expected = 4.0 / (fs / 2.0)
    return numeric(
        expected, float(np.mean(res.psd[band])), 0.03, rel=True, places=6
    )


@register(
    _SPECTRA,
    "Bendat & Piersol, Random Data 4e Eq. (8.158)",
    "PSD random error = 1/sqrt(nd) (Monte Carlo, 100 seeded records)",
)
def _chk_psd_random_error() -> Outcome:
    fs = _spectra_fs()
    estimates = []
    nd = 0.0
    for seed in range(100):
        res = ph.power_spectral_density(
            _spectra_white(100 + seed), fs, nperseg=1024, overlap=0.0
        )
        estimates.append(res.psd[50:200])
        nd = res.n_averages
    stack = np.asarray(estimates)
    empirical = float(np.mean(np.std(stack, axis=0) / np.mean(stack, axis=0)))
    return numeric(1.0 / math.sqrt(nd), empirical, 0.06, rel=True, places=4)


@register(
    _SPECTRA,
    "Bendat & Piersol, Random Data 4e Eq. (8.163)",
    "95% chi-square confidence interval coverage (Monte Carlo)",
)
def _chk_psd_ci_coverage() -> Outcome:
    fs = _spectra_fs()
    true_psd = 1.0 / (fs / 2.0)
    hits, total = 0, 0
    for seed in range(150):
        res = ph.power_spectral_density(_spectra_white(300 + seed), fs, nperseg=1024)
        for b in (60, 120, 240):
            hits += int(res.ci_lower[b] <= true_psd <= res.ci_upper[b])
            total += 1
    return numeric(0.95, hits / total, 0.025, places=4)


@register(
    _SPECTRA,
    "Bendat & Piersol, Random Data 4e Eqs. (9.55)/(6.39)",
    "Coherent output spectrum of a known-SNR path: gamma^2 = SNR/(1+SNR)",
)
def _chk_coherent_output_snr() -> Outcome:
    fs = _spectra_fs()
    x = _spectra_white(11)
    noise = ph.noise_signal(fs, 4.0, color="white", rms=0.5, seed=12)
    res = ph.coherent_output_spectrum(x, 0.8 * x + noise, fs, nperseg=1024)
    snr = 0.64 / 0.25
    band = slice(50, 400)
    return numeric(
        snr / (1.0 + snr),
        float(np.median(res.coherence[band])),
        0.03,
        places=4,
    )


@register(
    _SPECTRA,
    "Closed-form power-law slope (10*lg(2) dB/octave per unit exponent)",
    "Pink-noise PSD slope over 20 Hz - 20 kHz, dB/octave",
)
def _chk_pink_noise_slope() -> Outcome:
    fs = 48000.0
    x = ph.noise_signal(fs, 40.0, color="pink", seed=3)
    res = ph.power_spectral_density(x, fs, nperseg=8192)
    band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    slope = float(
        np.polyfit(
            np.log2(res.frequencies[band]), 10.0 * np.log10(res.psd[band]), 1
        )[0]
    )
    return numeric(-10.0 * math.log10(2.0), slope, 0.05, unit="dB/oct", places=4)


@register(
    _SPECTRA,
    "IEC 60268-1:1985 Clause A2.1 / Table AII",
    "5 ms burst of 5 kHz tone at 48 kHz: gate RMS = A/sqrt(2) (integral periods)",
)
def _chk_tone_burst_rms() -> Outcome:
    # Clause A2.1: zero-crossing start, integral number of full periods.
    # Over exactly 25 full periods (240 samples) the mean square of the
    # sine is exactly 1/2, so the gate RMS is A/sqrt(2) to machine
    # precision.
    res = ph.tone_burst(48000.0, 5000.0, 25, amplitude=1.0)
    if res.burst_samples != 240:  # 5 ms at 48 kHz, from Table AII
        return numeric(240.0, float(res.burst_samples), 0.0, places=0)
    rms = float(np.sqrt(np.mean(res.signal[:240] ** 2)))
    return numeric(1.0 / math.sqrt(2.0), rms, 1e-12, places=6)


@register(
    _SPECTRA,
    "Harris 1978 closed form (DFT-even Hann)",
    "Hann window ENBW = n*sum(w^2)/sum(w)^2 = 3/2 exactly",
)
def _chk_hann_enbw() -> Outcome:
    res = ph.window_metrics("hann", 1024)
    return numeric(1.5, float(res.enbw_bins), 1e-12, places=6)


@register(
    _SPECTRA,
    "Constant-power 1/n-octave kernel (closed form)",
    "1/3-octave smoothed line level = P*df/(f0*(2^(1/6)-2^(-1/6)))",
)
def _chk_smoothing_line_level() -> Outcome:
    f = np.arange(1.0, 4001.0)
    power = np.zeros_like(f)
    i0 = 999  # 1000 Hz
    power[i0] = 5.0
    out = ph.fractional_octave_smoothing(f, power, 3.0)
    width = 1000.0 * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
    return numeric(5.0 / width, float(out[i0]), 1e-9, rel=True, places=6)


@register(
    _SPECTRA,
    "Percival & Walden 1993, Table 382",
    "Slepian taper concentration lambda_14(31, 8/31), quadruple-precision table",
)
def _chk_multitaper_dpss_eigenvalue() -> Outcome:
    from scipy.signal.windows import dpss

    _, ratios = dpss(31, 8.0, Kmax=15, return_ratios=True)
    return numeric(
        0.929438220819848052, float(ratios[14]), 1e-12, places=12
    )


@register(
    _SPECTRA,
    "Percival & Walden 1993, Section 7.2 / Eq. (333)",
    "Multitaper white-noise density = sigma^2/(fs/2), NW=4, K=7 tapers",
)
def _chk_multitaper_white_level() -> Outcome:
    fs = _spectra_fs()
    res = ph.multitaper_psd(np.asarray(_spectra_white(41, rms=2.0))[:8192], fs)
    expected = 4.0 / (fs / 2.0)
    return numeric(
        expected, float(np.mean(res.psd[1:-1])), 0.03, rel=True, places=6
    )


@register(
    _SPECTRA,
    "Percival & Walden 1993, Eq. (369a) tone calibration",
    "Multitaper 'spectrum' scaling reads a sinusoid peak at A^2/2",
)
def _chk_multitaper_tone_peak() -> Outcome:
    fs = _spectra_fs()
    t = np.arange(4096) / fs
    x = 3.0 * np.sin(2.0 * np.pi * 1024.0 * t)
    res = ph.multitaper_psd(x, fs, scaling="spectrum", adaptive=False)
    return numeric(
        4.5, float(res.psd[int(np.argmax(res.psd))]), 1e-4, rel=True, places=6
    )


@register(
    _SPECTRA,
    "Percival & Walden 1993, Eq. (370b)",
    "Adaptive multitaper dof -> 2K on white noise (weights -> uniform)",
)
def _chk_multitaper_adaptive_dof() -> Outcome:
    fs = _spectra_fs()
    res = ph.multitaper_psd(np.asarray(_spectra_white(42))[:4096], fs)
    return numeric(
        2.0 * res.n_tapers,
        float(np.mean(res.degrees_of_freedom[1:-1])),
        0.02,
        rel=True,
        places=4,
    )


# ===========================================================================
# Multiple-input/output coherence (Bendat & Piersol, Random Data 4e, Ch. 7)
# ===========================================================================
_MISO = "Multiple-input coherence (Bendat & Piersol)"


def _miso_problem_7_2() -> tuple[float, float, float]:
    """Conditioned spectra of Problem 7.2 via the public conditioning path.

    Returns ``(Gv1, Gv2, gamma2_2y.1)`` computed by the module's
    Gaussian-elimination conditioning on the hand-set augmented matrix.
    """
    from phonometry.metrology.miso import _condition

    mat = np.zeros((1, 3, 3), dtype=np.complex128)
    mat[0, 0, 0] = 3.0
    mat[0, 1, 1] = 2.0
    mat[0, 2, 2] = 10.0
    mat[0, 0, 1] = 1.0 + 1.0j
    mat[0, 1, 0] = 1.0 - 1.0j
    mat[0, 0, 2] = 4.0 + 1.0j
    mat[0, 2, 0] = 4.0 - 1.0j
    mat[0, 1, 2] = 3.0 - 1.0j
    mat[0, 2, 1] = 3.0 + 1.0j
    partial, coherent, _noise = _condition(mat, (0, 1))
    return float(coherent[0, 0]), float(coherent[1, 0]), float(partial[1, 0])


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Problem 7.2 / Eqs. (7.86)/(7.94)",
    "Conditioned coherent output of the 2nd input abs(G2y.1)^2/G22.1 = 4/3 exactly",
)
def _chk_miso_problem_7_2_conditioned() -> Outcome:
    _gv1, gv2, _p2 = _miso_problem_7_2()
    return numeric(4.0 / 3.0, gv2, 1e-12, places=9)


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Problem 7.2 / Eqs. (7.87)/(7.116)",
    "Partial coherence gamma^2_2y.1 = 2/15 and multiple coherence = 0.7",
)
def _chk_miso_problem_7_2_multiple() -> Outcome:
    gv1, gv2, _p2 = _miso_problem_7_2()
    # gamma^2_{y:x} = (Gv1 + Gv2)/Gyy = (17/3 + 4/3)/10 = 0.7 (Eq. 7.116).
    return numeric(0.7, (gv1 + gv2) / 10.0, 1e-12, places=9)


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Eq. (7.35) with Eqs. (6.40)/(6.41)",
    "Multiple coherence of a known-SNR system: gamma^2_{y:x} = SNR/(1+SNR)",
)
def _chk_miso_multiple_snr() -> Outcome:
    fs = _spectra_fs()
    x1 = _spectra_white(201)
    x2 = _spectra_white(202)
    noise = ph.noise_signal(fs, 4.0, color="white", rms=0.5, seed=203)
    res = ph.miso_coherence([x1, x2], x1 + x2 + noise, fs, nperseg=1024)
    band = (res.frequencies > 200.0) & (res.frequencies < 3800.0)
    snr = 2.0 / 0.25  # Gvv = 2*sigma^2, Gnn = 0.5^2, both flat
    return numeric(
        snr / (1.0 + snr),
        float(np.median(res.multiple_coherence[band])),
        0.03,
        places=4,
    )


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Eq. (7.117)",
    "Uncorrelated inputs: multiple coherence = sum of ordinary coherences",
)
def _chk_miso_independent_sum() -> Outcome:
    fs = _spectra_fs()
    x1 = _spectra_white(211)
    x2 = _spectra_white(212)
    noise = ph.noise_signal(fs, 4.0, color="white", rms=0.4, seed=213)
    res = ph.miso_coherence([x1, x2], x1 + 0.7 * x2 + noise, fs, nperseg=1024)
    band = (res.frequencies > 200.0) & (res.frequencies < 3800.0)
    diff = (res.multiple_coherence
            - res.ordinary_coherence.sum(axis=0))[band]
    # O(q/nd) coherence bias (Section 9.3); nd is a few hundred here.
    return numeric(0.0, float(np.median(diff)), 0.02, places=4)


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Eqs. (7.88)/(7.121)",
    "Output-power decomposition Gyy = sum of Gvi + Gnn (exact)",
)
def _chk_miso_output_decomposition() -> Outcome:
    fs = _spectra_fs()
    x1 = _spectra_white(221)
    x2 = 0.5 * x1 + _spectra_white(222)
    noise = ph.noise_signal(fs, 4.0, color="white", rms=0.3, seed=223)
    res = ph.miso_coherence([x1, x2], x1 + x2 + noise, fs, nperseg=1024)
    reconstructed = res.coherent_output_spectra.sum(axis=0) + res.noise_psd
    resid = float(np.max(np.abs(reconstructed - res.output_psd)))
    return numeric(0.0, resid, 1e-12, places=12)


# ===========================================================================
# Time-frequency analysis (Bendat & Piersol, Random Data 4e)
# ===========================================================================
_TIME_FREQ = "Time-frequency analysis (Bendat & Piersol)"


@register(
    _TIME_FREQ,
    "Bendat & Piersol, Random Data 4e Eq. (12.173)",
    "Spectrogram of an on-bin tone reads its mean square A^2/2 in every column",
)
def _chk_spectrogram_tone_mean_square() -> Outcome:
    fs = _spectra_fs()
    t = np.arange(int(4 * fs)) / fs
    x = 2.0 * np.cos(2.0 * np.pi * 1024.0 * t)  # bin 128 of a 1024-segment
    res = ph.spectrogram(x, fs, nperseg=1024, scaling="spectrum")
    b = int(np.argmin(np.abs(res.frequencies - 1024.0)))
    worst = res.power[b][int(np.argmax(np.abs(res.power[b] - 2.0)))]
    return numeric(2.0, float(worst), 1e-9, rel=True, places=6)


@register(
    _TIME_FREQ,
    "Parseval + COLA identity (Hann taper, 75% overlap)",
    "Time-integrated STFT power = time-domain energy of an interior burst",
)
def _chk_spectrogram_parseval_cola() -> Outcome:
    fs = _spectra_fs()
    x = np.zeros(8192)
    x[2048:4096] = np.asarray(_spectra_white(21))[:2048]
    res = ph.spectrogram(x, fs, nperseg=256, overlap=0.75)
    df = float(res.frequencies[1] - res.frequencies[0])
    stft_energy = (res.hop / fs) * float(np.sum(res.power)) * df
    return numeric(
        float(np.sum(x**2)) / fs,
        stft_energy,
        1e-12,
        rel=True,
        places=6,
    )


@register(
    _TIME_FREQ,
    "Bendat & Piersol, Random Data 4e Eqs. (11.128)-(11.130)",
    "Zoom FFT tone amplitude = demodulate-decimate-DFT chain, machine precision",
)
def _chk_zoom_fft_demodulation_chain() -> Outcome:
    fs = _spectra_fs()
    n = 4096
    t = np.arange(n) / fs
    x = 0.7 * np.cos(2.0 * np.pi * 1100.0 * t + 0.3)
    res = ph.zoom_fft(x, fs, 1000.0, 1256.0, n_points=257, window="boxcar")
    peak = int(np.argmax(res.amplitude))
    # Eqs. (11.128)-(11.130): demodulate by exp(-j*2*pi*1000*t), decimate
    # by d = fs/(2B) = 16 and read bin 50 ((1100-1000) Hz / 2 Hz) of the
    # decimated record's DFT.
    idx = np.arange(n)
    v = (x * np.exp(-2j * np.pi * 1000.0 * idx / fs))[::16]
    m = np.arange(v.size)
    bin50 = np.sum(v * np.exp(-2j * np.pi * 50.0 * m / v.size))
    amp_bp = 2.0 * abs(bin50) / v.size
    return numeric(amp_bp, float(res.amplitude[peak]), 1e-12, rel=True, places=6)


# ===========================================================================
# Correlation, time delay and envelope (Bendat & Piersol / Knapp & Carter)
# ===========================================================================
_CORRELATION = "Correlation, time delay and envelope (B&P / Knapp & Carter)"


def _corr_fs() -> float:
    return 8192.0


def _corr_fractional_pair(
    seed: int, shift: float
) -> tuple[np.ndarray, np.ndarray]:
    """White noise and its exact circular fractional delay by ``shift``."""
    fs = _corr_fs()
    x = ph.noise_signal(fs, 4.0, color="white", seed=seed)
    ramp = np.exp(-2j * np.pi * np.fft.rfftfreq(x.size) * shift)
    return x, np.fft.irfft(np.fft.rfft(x) * ramp, x.size)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Eq. (5.21)",
    "Cross-correlation peak of a 16-sample pure delay, samples",
)
def _chk_tde_integer_delay() -> Outcome:
    fs = _corr_fs()
    x = ph.noise_signal(fs, 4.0, color="white", seed=40)
    res = ph.time_delay(x, np.roll(x, 16), fs, method="direct")
    return numeric(16.0, res.delay_samples, 1e-3, places=4)


@register(
    _CORRELATION,
    "Knapp & Carter 1976, Table I (PHAT) + sub-sample interpolation",
    "GCC-PHAT estimate of an exact 12.25-sample fractional delay, samples",
)
def _chk_tde_gcc_phat_fractional() -> Outcome:
    x, y = _corr_fractional_pair(41, 12.25)
    res = ph.time_delay(
        x, y, _corr_fs(), method="gcc", weighting="phat", nperseg=2048,
        upsample=16,
    )
    return numeric(12.25, res.delay_samples, 5e-3, places=4)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Eq. (5.101)",
    "Cross-spectrum phase-slope estimate of the same fractional delay",
)
def _chk_tde_phase_slope_fractional() -> Outcome:
    x, y = _corr_fractional_pair(41, 12.25)
    res = ph.time_delay(x, y, _corr_fs(), method="phase", nperseg=2048)
    return numeric(12.25, res.delay_samples, 1e-3, places=4)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Eq. (8.120)",
    "BLWN autocorrelation coefficient at 3 samples vs sin(2piBt)/(2piBt)",
)
def _chk_blwn_autocorrelation_sinc() -> Outcome:
    fs = _corr_fs()
    bandwidth = fs / 5.0
    x = ph.noise_signal(fs, 4.0, color="white", seed=41)
    spectrum = np.fft.rfft(x)
    spectrum[np.fft.rfftfreq(x.size, 1.0 / fs) > bandwidth] = 0.0
    xb = np.fft.irfft(spectrum, x.size)
    res = ph.correlation(xb, fs=fs, normalization="coefficient",
                         max_lag=0.005)
    lag = int(np.argmin(np.abs(res.lags))) + 3
    arg = 2.0 * math.pi * bandwidth * res.lags[lag]
    return numeric(
        math.sin(arg) / arg, float(res.coefficient[lag]), 0.02, places=4
    )


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Example 8.5",
    "Random error of the correlation peak: B=100 Hz, T=5 s, M/S=N/S=10",
)
def _chk_correlation_random_error_example_8_5() -> Outcome:
    # rho_peak = S/sqrt((S+M)(S+N)) = 1/11 (Eq. 8.115); the book gives 0.35.
    eps = ph.correlation_random_error(1.0 / 11.0, 100.0, 5.0)
    return numeric(0.35, eps, 1e-3, places=4)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Table 13.1",
    "Hilbert transform of cos recovers sin: max interior error",
)
def _chk_hilbert_cos_to_sin() -> Outcome:
    fs = _corr_fs()
    n = 16384
    t = np.arange(n) / fs
    res = ph.envelope(np.cos(2.0 * np.pi * 500.0 * t), fs)
    interior = slice(1024, n - 1024)
    reconstructed = res.envelope * np.sin(res.phase)
    err = float(np.max(np.abs(
        reconstructed[interior] - np.sin(2.0 * np.pi * 500.0 * t)[interior]
    )))
    return numeric(0.0, err, 1e-9, places=6)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Eq. (13.27)",
    "Envelope of an AM waveform recovers 1 + m*cos(2pi*fm*t) exactly",
)
def _chk_am_envelope_exact() -> Outcome:
    fs = _corr_fs()
    n = 16384
    t = np.arange(n) / fs
    exact = 1.0 + 0.5 * np.cos(2.0 * np.pi * 10.0 * t)
    res = ph.envelope(exact * np.cos(2.0 * np.pi * 1000.0 * t), fs)
    interior = slice(1024, n - 1024)
    err = float(np.max(np.abs(res.envelope[interior] - exact[interior])))
    return numeric(0.0, err, 1e-9, places=6)


# ===========================================================================
# Cepstral analysis and envelope spectrum (Havelock 2008 / Bendat & Piersol)
# ===========================================================================
_CEPSTRUM = "Cepstrum, liftering and envelope spectrum (Havelock / B&P)"


def _cepstrum_echo_signal() -> np.ndarray:
    """delta[n] + a*delta[n-d]: DFT exactly 1 + a*exp(-j*2pi*k*d/N)."""
    x = np.zeros(4096)
    x[0] = 1.0
    x[313] = 0.4
    return x


@register(
    _CEPSTRUM,
    "Havelock 2008 Ch. 27 Fig. 21 + Mercator series of ln(1+a*e^{-j*theta})",
    "Power-cepstrum height at the echo delay = reflection coefficient a",
)
def _chk_power_cepstrum_echo() -> Outcome:
    res = ph.echo_detection(_cepstrum_echo_signal(), 8192.0)
    if res.delay_samples != 313:
        return numeric(313.0, float(res.delay_samples), 0.5, places=0)
    return numeric(0.4, res.reflection_coefficient, 1e-10, places=6)


@register(
    _CEPSTRUM,
    "Havelock 2008 Ch. 87 Eq. (14): complex cepstrum, series term n = 2",
    "Second rahmonic of a reflection a = 0.4 equals -a^2/2",
)
def _chk_complex_cepstrum_rahmonic() -> Outcome:
    res = ph.cepstrum(_cepstrum_echo_signal(), 8192.0, kind="complex")
    return numeric(-0.08, float(res.cepstrum[2 * 313]), 1e-10, places=6)


@register(
    _CEPSTRUM,
    "Bendat & Piersol, Random Data 4e Sec. 13.3 (Fig. 13.11)",
    "Envelope-spectrum line of an AM tone (A0 = 2, m = 0.35) at fm",
)
def _chk_envelope_spectrum_am_line() -> Outcome:
    fs = 8192.0
    n = 16384
    t = np.arange(n) / fs
    x = 2.0 * (1.0 + 0.35 * np.cos(2.0 * np.pi * 16.0 * t)) * np.cos(
        2.0 * np.pi * 1000.0 * t
    )
    res = ph.envelope_spectrum(x, fs)
    line = float(res.amplitude[round(16.0 * n / fs)])
    return numeric(0.7, line, 2e-3, places=4)


# ===========================================================================
# Time synchronous averaging (McFadden 1987)
# ===========================================================================
_TSA = "Time synchronous averaging (McFadden 1987)"


@register(
    _TSA,
    r"McFadden 1987 Eq. 8 / Eq. 9: comb filter \|C(f)\| at a harmonic k/T",
    "Comb-filter tooth height at a harmonic equals unity (any N)",
)
def _chk_tsa_comb_tooth() -> Outcome:
    period = 1.0 / 32.0
    value = float(ph.comb_filter_response(np.array([16.0 / period]), period, 8)[0])
    return numeric(1.0, value, 1e-10, places=8)


@register(
    _TSA,
    "McFadden 1987 Eq. 8: comb filter one quarter-order from a tooth, N = 2",
    "Comb-filter magnitude = 1/sqrt(2) at order 0.25",
)
def _chk_tsa_comb_midbin() -> Outcome:
    period = 1.0 / 32.0
    value = float(
        ph.comb_filter_response(np.array([0.25 / period]), period, 2)[0]
    )
    return numeric(1.0 / math.sqrt(2.0), value, 1e-10, places=8)


@register(
    _TSA,
    "McFadden 1987 Sec. 4 (Fig. 5): node selection, tone at 32.05 orders",
    r"N = 20 places a comb node on 32.05 orders (\|C\| = 0), not the power-of-2 N = 32",
)
def _chk_tsa_node_selection() -> Outcome:
    period = 1.0 / 32.0
    freq = np.array([32.05 / period])
    c20 = float(ph.comb_filter_response(freq, period, 20)[0])
    c32 = float(ph.comb_filter_response(freq, period, 32)[0])
    if not c32 > 0.15:  # sanity: the power-of-two choice does not reject it
        return numeric(0.0, c32, 0.0, places=8)
    return numeric(0.0, c20, 1e-10, places=10)


@register(
    _TSA,
    "McFadden 1987 Eq. 5: exact recovery, integer samples per period",
    "Noiseless periodic waveform (M = 256) recovered to machine precision",
)
def _chk_tsa_exact_recovery() -> Outcome:
    fs = 8192.0
    period = 1.0 / 32.0
    m = 256
    phase = np.arange(m) / m
    one = np.cos(2.0 * np.pi * phase) + 0.5 * np.cos(
        2.0 * np.pi * 3.0 * phase + 0.4
    )
    res = ph.time_synchronous_average(np.tile(one, 24), fs, period)
    err = float(np.max(np.abs(res.period_waveform - one)))
    return numeric(0.0, err, 1e-10, places=12)


@register(
    _TSA,
    "McFadden 1987 Sec. 1: asynchronous-noise variance reduced by 1/N",
    "Residual noise std of the average falls as sigma/sqrt(N), N = 64",
)
def _chk_tsa_sqrt_n_law() -> Outcome:
    fs = 8192.0
    period = 1.0 / 32.0
    m = 256
    n_avg = 64
    phase = np.arange(m) / m
    one = np.cos(2.0 * np.pi * phase)
    rng = np.random.default_rng(2024)
    noise = rng.standard_normal(n_avg * m)
    res = ph.time_synchronous_average(
        np.tile(one, n_avg) + noise, fs, period, n_averages=n_avg
    )
    measured = float(np.std(res.period_waveform - one))
    return numeric(1.0 / math.sqrt(n_avg), measured, 0.15, rel=True, places=5)


# ===========================================================================
# Data qualification and Rice statistics (Bendat & Piersol Chs. 4, 5, 10)
# ===========================================================================
_RANDOM_DATA = "Data qualification and Rice statistics (Bendat & Piersol)"

#: B&P Example 4.4: twenty observations with A = 86 reverse arrangements,
#: accepted as trend-free at the 5 % level of significance.
_BP_EXAMPLE_4_4 = [
    5.2, 6.2, 3.7, 6.4, 3.9, 4.0, 3.9, 5.3, 4.0, 4.6,
    5.9, 6.5, 4.3, 5.7, 3.1, 5.6, 5.2, 3.9, 6.2, 5.0,
]


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Example 4.4",
    "Reverse arrangements of the 20-observation sequence",
)
def _chk_reverse_arrangements_example_4_4() -> Outcome:
    res = ph.trend_test(_BP_EXAMPLE_4_4)
    if not res.trend_free:  # the book accepts the hypothesis at 5 %
        return numeric(1.0, 0.0, 0.0, places=0)
    return numeric(86.0, float(res.statistic), 0.0, places=0)


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Table A.6",
    "Lower percentage point A(20; 0.975) at alpha = 0.05",
)
def _chk_table_a6_lower() -> Outcome:
    res = ph.trend_test(_BP_EXAMPLE_4_4)
    return numeric(64.0, float(res.bounds[0]), 0.0, places=0)


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Table A.6",
    "Upper percentage point A(20; 0.025) at alpha = 0.05",
)
def _chk_table_a6_upper() -> Outcome:
    res = ph.trend_test(_BP_EXAMPLE_4_4)
    return numeric(125.0, float(res.bounds[1]), 0.0, places=0)


def _runs_reference_result() -> ph.TrendTestResult:
    rng = np.random.default_rng(20)
    return ph.trend_test(rng.standard_normal(20), method="runs")


@register(
    _RANDOM_DATA,
    "Wald & Wolfowitz 1940 exact run distribution",
    "Runs acceptance region for n1 = n2 = 10, alpha = 0.05: lower point",
)
def _chk_runs_bounds_lower() -> Outcome:
    return numeric(
        6.0, float(_runs_reference_result().bounds[0]), 0.0, places=0
    )


@register(
    _RANDOM_DATA,
    "Wald & Wolfowitz 1940 exact run distribution",
    "Runs acceptance region for n1 = n2 = 10, alpha = 0.05: upper point",
)
def _chk_runs_bounds_upper() -> Outcome:
    return numeric(
        15.0, float(_runs_reference_result().bounds[1]), 0.0, places=0
    )


def _bandlimited_gaussian_record(
    seed: int, fs: float, n: int, f1: float, f2: float
) -> np.ndarray:
    """Exactly bandlimited unit-variance Gaussian noise (FFT synthesis)."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(
        freqs.size
    )
    spec[(freqs < f1) | (freqs > f2)] = 0.0
    x = np.fft.irfft(spec, n)
    return np.asarray(x / np.std(x))


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Example 5.13 / Eq. (5.195)",
    "Zero-crossing rate of bandlimited noise (fc = 1 kHz, B = 400 Hz)",
)
def _chk_rice_zero_crossings_bandpass() -> Outcome:
    x = _bandlimited_gaussian_record(0, 20480.0, 1 << 19, 800.0, 1200.0)
    res = ph.level_crossing_rate(x, 20480.0, levels=[0.0])
    expected = 2.0 * float(np.sqrt(1000.0**2 + 400.0**2 / 12.0))
    return numeric(expected, res.zero_crossing_rate, 0.01, rel=True, places=0)


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Example 5.12",
    "Apparent frequency of low-pass noise (B = 2 kHz) = 0.577 B",
)
def _chk_rice_apparent_frequency_lowpass() -> Outcome:
    x = _bandlimited_gaussian_record(1, 20480.0, 1 << 19, 0.0, 2000.0)
    res = ph.level_crossing_rate(x, 20480.0, levels=[0.0])
    expected = 2000.0 / float(np.sqrt(3.0))
    return numeric(expected, res.apparent_frequency, 0.01, rel=True, places=0)


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Example 5.14 / Eq. (5.206)",
    "Prob[positive peak > 4 sigma] of a narrow bandwidth record",
)
def _chk_rice_narrowband_peak_exceedance() -> Outcome:
    fs = 8192.0
    t = np.arange(1 << 16) / fs
    res = ph.peak_statistics(np.sin(2.0 * np.pi * 60.0 * t), fs)
    return numeric(
        float(np.exp(-8.0)),
        float(res.peak_exceedance(4.0)[0]),
        1e-5,
        places=6,
    )


# ===========================================================================
# Underwater acoustics (ISO 18405 / 17208 / 18406)
# ===========================================================================
_UNDERWATER = "Underwater acoustics (ISO 18405/17208/18406)"


@register(
    _UNDERWATER,
    "ISO 18405:2017 / ISO 18406 Formula 7",
    "Sound pressure level of a synthetic tone, dB re 1 µPa",
)
def _chk_uw_spl() -> Outcome:
    fs = 48000
    t = np.arange(fs) / fs
    amp = 2.0  # Pa
    x = amp * np.sin(2.0 * np.pi * 500.0 * t)
    expected = 20.0 * math.log10((amp / math.sqrt(2.0)) / 1e-6)
    return numeric(expected, ph.sound_pressure_level(x), 1e-4, places=4)


@register(
    _UNDERWATER,
    "ISO 18405:2017 / ISO 18406 Formulae 3-4",
    "Sound exposure level of a 2 s tone, dB re 1 µPa²·s",
)
def _chk_uw_sel() -> Outcome:
    fs = 48000
    t = np.arange(2 * fs) / fs
    amp = 1.0
    x = amp * np.sin(2.0 * np.pi * 500.0 * t)
    spl = 20.0 * math.log10((amp / math.sqrt(2.0)) / 1e-6)
    expected = spl + 10.0 * math.log10(2.0)
    return numeric(expected, ph.sound_exposure_level(x, fs), 1e-3, places=4)


@register(
    _UNDERWATER,
    "ISO 18406:2017 (6.4.2.1.3)",
    "Peak sound pressure level of a known waveform, dB re 1 µPa",
)
def _chk_uw_peak() -> Outcome:
    fs = 48000
    t = np.arange(fs) / fs
    amp = 3.0
    x = amp * np.sin(2.0 * np.pi * 500.0 * t)
    expected = 20.0 * math.log10(amp / 1e-6)
    return numeric(expected, ph.peak_sound_pressure_level(x), 1e-4, places=4)


@register(
    _UNDERWATER,
    "ISO 17208-1:2016",
    "Radiated noise level from RMS pressure and distance, dB re 1 µPa·m",
)
def _chk_uw_rnl() -> Outcome:
    expected = 20.0 * math.log10(2.0) + 40.0  # p = 2 µPa, r = 100 m
    return numeric(expected, ph.radiated_noise_level(2e-6, 100.0), 1e-4, places=4)


@register(
    _UNDERWATER,
    "ISO 17208-2:2019 (Formula 3)",
    "Lloyd's-mirror surface correction ΔL at a known k·d_s",
)
def _chk_uw_delta_l() -> Outcome:
    draught, c, f = 10.0, 1500.0, 200.0
    ds = 0.7 * draught
    u = 2.0 * math.pi * f / c * ds
    expected = -10.0 * math.log10((2 * u**4 + 14 * u**2) / (14 + 2 * u**2 + u**4))
    res = ph.monopole_source_level(120.0, f, draught, c=c)
    return numeric(expected, float(res.surface_correction[0]), 1e-4, places=4)


@register(
    _UNDERWATER,
    "ISO 18406:2017 (Formulae 8-9)",
    "Cumulative SEL of N identical strikes = SEL_ss + 10·lg(N)",
)
def _chk_uw_cumulative_sel() -> Outcome:
    return numeric(
        180.0 + 10.0 * math.log10(50),
        ph.cumulative_sel_identical(180.0, 50),
        1e-6,
        places=4,
    )


# ===========================================================================
# Underwater sound propagation (transmission loss, closed-form)
# ===========================================================================
_UW_PROP = "Underwater sound propagation (transmission loss)"


@register(
    _UW_PROP,
    "Mackenzie (1981) nine-term equation",
    "Speed of sound at 25 °C, 35 ‰, 1000 m (canonical check value), m/s",
)
def _chk_uwp_mackenzie() -> Outcome:
    return numeric(
        1550.744,
        ph.sea_water_sound_speed(25.0, 35.0, 1000.0, model="mackenzie"),
        1e-2,
        unit="m/s",
        places=3,
    )


@register(
    _UW_PROP,
    "UNESCO/Chen-Millero vs Mackenzie",
    "Sound-speed agreement at 10 °C, 35 ‰, 1000 m (cross-model), m/s",
)
def _chk_uwp_unesco() -> Outcome:
    expected = ph.sea_water_sound_speed(10.0, 35.0, 1000.0, model="mackenzie")
    got = ph.sea_water_sound_speed(10.0, 35.0, 1000.0, model="unesco")
    return numeric(expected, got, 1.0, unit="m/s", places=3)


@register(
    _UW_PROP,
    "Del Grosso (1974) vs Mackenzie",
    "Sound-speed agreement at 10 °C, 35 ‰, 1000 m (cross-model), m/s",
)
def _chk_uwp_del_grosso() -> Outcome:
    expected = ph.sea_water_sound_speed(10.0, 35.0, 1000.0, model="mackenzie")
    got = ph.sea_water_sound_speed(10.0, 35.0, 1000.0, model="del_grosso")
    return numeric(expected, got, 1.0, unit="m/s", places=3)


@register(
    _UW_PROP,
    "Spherical spreading 20·lg(R)",
    "Geometrical spreading loss at R = 1000 m, dB",
)
def _chk_uwp_spreading() -> Outcome:
    return numeric(
        20.0 * math.log10(1000.0),
        float(ph.spreading_loss([1000.0], law="spherical")[0]),
        1e-9,
        unit="dB",
        places=4,
    )


@register(
    _UW_PROP,
    "Thorp (1967) absorption",
    "Volume absorption α at 10 kHz (cold deep water), dB/km",
)
def _chk_uwp_thorp() -> Outcome:
    f = 10.0  # kHz
    expected = 1.0936 * (0.1 * f**2 / (1 + f**2) + 40 * f**2 / (4100 + f**2))
    got = float(ph.seawater_absorption(10_000.0, model="thorp")[0])
    return numeric(expected, got, 1e-6, unit="dB/km", places=4)


@register(
    _UW_PROP,
    "Ainslie-McColm (1998) vs Francois-Garrison (1982)",
    "Absorption agreement at 10 kHz, 10 °C, 35 ‰, 0 m, pH 8, dB/km",
)
def _chk_uwp_absorption_agreement() -> Outcome:
    kw = {"temperature": 10.0, "salinity": 35.0, "depth": 0.0, "ph": 8.0}
    fg = float(ph.seawater_absorption(10_000.0, model="francois-garrison", **kw)[0])
    am = float(ph.seawater_absorption(10_000.0, model="ainslie-mccolm", **kw)[0])
    return numeric(fg, am, 0.1 * fg, unit="dB/km", places=4)


@register(
    _UW_PROP,
    "Francois-Garrison (1982) Part II Table IV",
    "Absorption α at 100 kHz, 10 °C, 35 ‰, 0 m, pH 8 (printed value), dB/km",
)
def _chk_uwp_fg_printed_table() -> Outcome:
    # Oracle: the printed absorption table of the source paper (J. Acoust.
    # Soc. Am. 72(6), 1982); tolerance is half a unit of the last printed
    # digit, i.e. the print's own rounding.
    kw = {"temperature": 10.0, "salinity": 35.0, "depth": 0.0, "ph": 8.0}
    got = float(ph.seawater_absorption(100_000.0, model="francois-garrison", **kw)[0])
    return numeric(33.6, got, 0.05, unit="dB/km", places=3)


@register(
    _UW_PROP,
    "Del Grosso refit (Wong-Zhu 1995 Table IV)",
    "c(t90 = 20 °C, S = 35, P = 500 bar) vs the printed check table, m/s",
)
def _chk_uwp_del_grosso_printed_check() -> Outcome:
    # Oracle: the printed ITS-90 check table of the refit the module
    # implements (J. Acoust. Soc. Am. 97(3), 1995); the table lists pressure
    # in bars, Del Grosso's polynomial takes kg/cm² (1 bar = 1.019716 kg/cm²).
    from phonometry.underwater.sound_speed import _del_grosso

    got = float(_del_grosso(20.0, 35.0, 500.0 * 1.019716))
    return numeric(1603.679, got, 1e-3, unit="m/s", places=3)


@register(
    _UW_PROP,
    "Wales-Heitmeyer (2002) ensemble spectrum",
    "Merchant-ship source PSD at 100 Hz (printed equation), dB re 1 µPa²/Hz",
)
def _chk_uwp_wales_heitmeyer() -> Outcome:
    # Oracle: the mean-spectrum closed form printed in J. Acoust. Soc. Am.
    # 111(3), 2002, hand-evaluated at 100 Hz.
    s = ph.ship_source_spectrum(model="wales-heitmeyer", frequency_hz=[100.0])
    return numeric(158.4504, float(s.source_psd[0]), 1e-3, unit="dB", places=3)


@register(
    _UW_PROP,
    "Passive sonar equation (Urick/Etter)",
    "Figure of merit SL − (NL − DI) − DT, dB",
)
def _chk_uwp_sonar() -> Outcome:
    res = ph.passive_sonar_equation(140.0, 80.0, 60.0, directivity_index=10.0, detection_threshold=5.0)
    return numeric(85.0, res.figure_of_merit, 1e-9, unit="dB", places=4)


@register(
    _UW_PROP,
    "Seabed reflection (Rayleigh, normal incidence)",
    "Bottom loss at 90° grazing, sand ρ=1900 c=1650 over water, dB",
)
def _chk_uwp_seabed() -> Outcome:
    # Normal-incidence oracle: R = (Z2 − Z1)/(Z2 + Z1), BL = −20·lg|R|.
    z1, z2 = 1000.0 * 1500.0, 1900.0 * 1650.0
    expected = -20.0 * math.log10(abs((z2 - z1) / (z2 + z1)))
    res = ph.bottom_reflection_loss(90.0, rho1=1000.0, c1=1500.0, rho2=1900.0, c2=1650.0)
    return numeric(expected, float(res.reflection_loss[0]), 1e-6, unit="dB", places=4)


@register(
    _UW_PROP,
    "Wenz wind noise (rule of fives)",
    "Wind spectrum level at 1 kHz, 5 kn (canonical anchor), dB re 1 µPa²/Hz",
)
def _chk_uwp_wind_noise() -> Outcome:
    # Wenz/Knudsen "25 dB (5 x 5)" is re 0.0002 dyn/cm2 = 20 uPa; re 1 uPa
    # (ISO 18405) the anchor is 25 + 20*lg(20) = 51.0206 dB (matches the
    # published Wenz chart: ~50 dB at 1 kHz for 4-6 kn).
    got = float(ph.wind_noise_spectrum(1000.0, 5.0)[0])
    return numeric(51.0206, got, 1e-4, unit="dB", places=4)


@register(
    _UW_PROP,
    "Mellen thermal noise",
    "Thermal spectrum level at 50 kHz, 16.85 °C (physical), dB re 1 µPa²/Hz",
)
def _chk_uwp_thermal_noise() -> Outcome:
    f, t, rho, c = 5.0e4, 16.85, 1025.0, 1500.0
    p2 = 4.0 * math.pi * 1.380649e-23 * (t + 273.15) * rho * f**2 / c
    expected = 10.0 * math.log10(p2 / (1e-6) ** 2)
    got = float(ph.thermal_noise_spectrum(f, temperature=t, density=rho, sound_speed=c)[0])
    return numeric(expected, got, 1e-6, unit="dB", places=4)


@register(
    _UW_PROP,
    "JOMOPANS-ECHO ship source level",
    "Bulker V=13.5 kn L=211 m band level at 1 kHz (File S1 oracle), dB re 1 µPa m",
)
def _chk_uwp_ship_traffic() -> Outcome:
    # Oracle: authors' Excel reference calculator (File S1), decidecade band.
    s = ph.ship_source_spectrum(13.5, 211.0, vessel_class="bulker", model="jomopans-echo")
    idx = int(min(range(len(s.frequency)), key=lambda i: abs(s.frequency[i] - 1000.0)))
    return numeric(161.394, float(s.band_level[idx]), 1e-2, unit="dB", places=3)


# ===========================================================================
# Underwater numerical propagation (Jensen et al., modes / rays / PE)
# ===========================================================================
@register(
    _UW_PROP,
    "UNESCO sound speed (EOS-80 canonical value)",
    "SVEL(S = 40, T68 = 40 °C, P = 1000 bar) vs Fofonoff & Millard 1983, m/s",
)
def _chk_uwp_unesco_canonical() -> Outcome:
    # Published canonical check of the UNESCO algorithm; the module implements
    # the Wong-Zhu ITS-90 refit, so T90 = T68/1.00024 and the tolerance covers
    # the published refit residual.
    from phonometry.underwater.sound_speed import _unesco

    got = float(_unesco(40.0 / 1.00024, 40.0, 1000.0))
    return numeric(1731.995, got, 0.02, unit="m/s", places=3)


_UW_NUM = "Underwater numerical propagation (modes / rays / PE)"


@register(
    _UW_NUM,
    "Normal modes vs ideal waveguide",
    "Fundamental horizontal wavenumber kr1 at 20 Hz, 100 m (analytic), rad/m",
)
def _chk_uwn_modes() -> Outcome:
    d, c, f = 100.0, 1500.0, 20.0
    k = 2.0 * math.pi * f / c
    kr1 = math.sqrt(k**2 - (math.pi / d) ** 2)
    res = ph.normal_modes(f, [0.0, d], [c, c], source_depth=36.0, receiver_depth=46.0,
                          bottom="pressure-release", n_depth_points=800)
    return numeric(kr1, float(res.wavenumbers[0]), 1e-4, unit="rad/m", places=6)


@register(
    _UW_NUM,
    "Normal modes vs image-source oracle",
    "Absolute TL at 1 km in the ideal waveguide (converged image sum), dB",
)
def _chk_uwn_modes_absolute() -> Outcome:
    # Independent absolute anchor (does not share the Eq. 5.14 prefactor with
    # the implementation): converged image-source sum for D = 100 m, f = 20 Hz,
    # zs = 36 m, zr = 46 m gives TL(1 km) = 48.238 dB.
    res = ph.normal_modes(20.0, [0.0, 100.0], [1500.0, 1500.0], source_depth=36.0,
                          receiver_depth=46.0, ranges_m=[1000.0],
                          n_depth_points=3000)
    return numeric(48.238, float(res.transmission_loss[0]), 0.02, unit="dB", places=3)


@register(
    _UW_NUM,
    "Ray tracing vs linear gradient",
    "Turning depth of a 10° ray, c = 1500 + 0.05z (circular arc), m",
)
def _chk_uwn_rays() -> Outcome:
    c0, g = 1500.0, 0.05
    xi = math.cos(math.radians(10.0)) / c0
    z_turn = (1.0 / xi - c0) / g
    res = ph.ray_trace([0.0, 2000.0], [c0, c0 + g * 2000.0], source_depth=0.0,
                       launch_angles_deg=[10.0], max_range=10_500.0, n_steps=20_000)
    return numeric(z_turn, float(res.depths[0].max()), 1.0, unit="m", places=2)


@register(
    _UW_NUM,
    "Parabolic equation vs free field",
    "PE transmission loss at 2 km, homogeneous medium (spherical spreading), dB",
)
def _chk_uwn_pe() -> Outcome:
    res = ph.parabolic_equation(50.0, [0.0, 20_000.0], [1500.0, 1500.0],
                                source_depth=10_000.0, max_range=3000.0,
                                range_step=2.0, n_depth_points=8192)
    zi = int(min(range(res.depths.size), key=lambda i: abs(res.depths[i] - 10_000.0)))
    ri = int(min(range(res.ranges.size), key=lambda i: abs(res.ranges[i] - 2000.0)))
    return numeric(20.0 * math.log10(2000.0), float(res.transmission_loss[zi][ri]),
                   0.1, unit="dB", places=3)


# ===========================================================================
# Aircraft noise (ICAO Annex 16 EPNL / IEC 61265)
# ===========================================================================
_AIRCRAFT = "Aircraft noise (ICAO Annex 16 / IEC 61265)"


@register(
    _AIRCRAFT,
    "ECAC Doc 29 noise fraction (half path)",
    "Finite-segment correction ΔF for a perpendicular foot at the segment start, dB",
)
def _chk_ecac_noise_fraction() -> Outcome:
    # A half-infinite segment (Sp at the start) receives half the energy: −3.01 dB.
    got = float(ph.noise_fraction(0.0, 10_000.0, 100.0))
    return numeric(-10.0 * math.log10(2.0), got, 1e-3, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 single-event chain",
    "SEL of a long level flyover vs the infinite-path limit LE∞ + ΔI − Λ, dB",
)
def _chk_ecac_event_level() -> Outcome:
    # A long straight level segment through the CPA reduces to the infinite-path
    # baseline plus the geometry corrections (ΔF → 0, ΔV = 0 at Vref).
    npd_p = [8000.0, 12000.0]
    npd_d = [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0]
    sel = [[98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0], [102.0, 96.0, 90.0, 84.0, 78.0, 72.0, 66.0]]
    lmax = [[94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0], [98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0]]
    vref = 160.0 * 0.514444
    import numpy as _np

    xs = _np.linspace(-40000.0, 40000.0, 801)
    path = _np.column_stack([xs, _np.zeros_like(xs), _np.full_like(xs, 300.0),
                             _np.full_like(xs, 10000.0), _np.full_like(xs, vref)])
    got = ph.event_level(path, [0.0, 300.0, 0.0], npd_p, npd_d, sel, lmax, metric="exposure").level
    dp = math.hypot(300.0, 300.0)
    beta = math.degrees(math.acos(300.0 / dp))
    expected = (float(ph.npd_level(npd_p, npd_d, sel, 10000.0, dp)[0])
                + ph.impedance_adjustment()
                + ph.engine_installation_correction(beta, "wing")
                - ph.lateral_attenuation(beta, 300.0))
    return numeric(expected, float(got), 1e-2, unit="dB", places=3)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 impedance adjustment (standard atmosphere)",
    "Acoustic-impedance adjustment of NPD data at 15 °C / 101.325 kPa (Eq. 4-6/4-7), dB",
)
def _chk_ecac_impedance() -> Outcome:
    # ECAC Doc 29 Vol 2 §4.2.1 states the standard-atmosphere adjustment is
    # +0.074 dB (ρc = 416.86, reference impedance 409.81 N·s/m³).
    return numeric(0.074, float(ph.impedance_adjustment()), 5e-4, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 reference workbook (segment Λ)",
    "Lateral attenuation of a climbing segment vs the ECAC Vol 3 Part 1 workbook, dB",
)
def _chk_ecac_workbook_segment() -> Outcome:
    # ECAC Doc 29 5th ed. Vol 3 Part 1 reference workbook, sheet
    # B-2_Segment_Results, case JETFAC receptor R02 segment 1 (climbing): the
    # workbook lists β = 4.2226°, lateral displacement 81363.28 ft (24799.6 m)
    # and Λ = 6.3769 dB. Here we anchor the Λ(β, ℓ) formula to those reference
    # values (the β/φ geometry itself is validated in tests/test_airport_noise).
    beta_ref, lateral_m = 4.2225708673, 81363.2829 * 0.3048
    got = float(ph.lateral_attenuation(beta_ref, lateral_m))
    return numeric(6.3768594165, got, 1e-2, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 start-of-roll directivity (jet)",
    "ΔSOR behind a takeoff ground-roll segment vs the Vol 3 Part 1 workbook, dB",
)
def _chk_ecac_start_of_roll() -> Outcome:
    # ECAC Doc 29 5th ed. Vol 3 Part 1 workbook, case JETFDC: at ψ = 112.8895°,
    # dSOR = 217.09 m the turbofan directivity (Eq. 4-24a) is +0.3196 dB.
    got = float(ph.start_of_roll_directivity(112.889545, 217.0934, "jet"))
    return numeric(0.31961, got, 1e-2, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 start-of-roll directivity (turboprop)",
    "ΔSOR behind a takeoff ground-roll segment (turboprop, Eq. 4-24b), dB",
)
def _chk_ecac_start_of_roll_prop() -> Outcome:
    # Same workbook, case PROPDC: at ψ = 128.1824°, dSOR = 254.44 m the turboprop
    # directivity (Eq. 4-24b) is +1.0943 dB.
    got = float(ph.start_of_roll_directivity(128.182381, 254.4361, "turboprop"))
    return numeric(1.09434, got, 1e-2, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 workbook event assembly (JETFDS/R03, behind SOR)",
    "Energy sum of the reference per-segment SELs vs the B-1 event total, dB",
)
def _chk_doc29_event_assembly() -> Outcome:
    # Doc 29 5th ed. Vol 3 Part 1 workbook, departure case JETFDS receptor R03
    # (centreline behind the start of roll): the Eq. 4-11 energy sum of the 29
    # reference segment SELs must reproduce the B-1 total 74.73 dB. No oracle
    # exists for per-event LAmax, non-zero bank angles or the Annex 16
    # bandsharing adjustment (no ETM worked example); registered gaps.
    import sys as _sys
    from pathlib import Path as _P

    tests_dir = str(_P(__file__).resolve().parent.parent / "tests" / "aircraft")
    if tests_dir not in _sys.path:
        _sys.path.insert(0, tests_dir)
    from doc29_workbook_data import B1, SEGMENTS

    rows = SEGMENTS[("JETFDS", "R03")]
    total = 10.0 * math.log10(sum(10.0 ** (r[-1] / 10.0) for r in rows))
    return numeric(B1[("JETFDS", "R03")], total, 1e-2, unit="dB", places=3)


@register(
    _AIRCRAFT,
    "SAE ARP 5534 band-attenuation continuity",
    "SAE-Method δ_B at the 150 dB branch split (Eq. 7 vs Eq. 8), dB",
)
def _chk_arp5534_continuity() -> Outcome:
    # The two SAE-Method branches meet at δ_t = 150 dB (Eqs. 7-8).
    a, b, c, d, e = 0.867942, 0.111761, 0.95824, 0.008191, 1.6
    eq7 = a * 150.0 * (1.0 + b * (c - d * 150.0)) ** e
    eq8 = 9.2 + 0.765 * 150.0
    return numeric(eq8, eq7, 0.01, unit="dB", places=3)


@register(
    _AIRCRAFT,
    "EASA ANP database round-trip",
    "Interpolated NPD level at a tabulated node vs the published ANP value, dB",
)
def _chk_anp_round_trip() -> Outcome:
    # Independent oracle: the EASA ANP database's own published NPD values
    # (curated subset shipped under aircraft/data/anp). At a tabulated
    # (power, distance) node the loader/interpolation must recover the
    # published value exactly. Boeing 747-100 (JT9DBD), departure SEL,
    # 28000 lb corrected net thrust, 2000 ft slant distance = 98.8 dB.
    curves = ph.load_anp_database().npd_curves("747100", "departure", "SEL")
    got = float(curves.level(28000.0, 2000.0 * 0.3048)[0])
    return numeric(98.8, got, 1e-9, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 NPD interpolation",
    "Log-linear NPD level at the log-midpoint distance (Eq. 4-4), dB",
)
def _chk_ecac_npd() -> Outcome:
    # Log-midpoint distance -> arithmetic mean of the bracketing node levels.
    p, d = [1000.0, 2000.0], [200.0, 400.0, 800.0, 1600.0]
    lv = [[100.0, 94.0, 88.0, 82.0], [110.0, 104.0, 98.0, 92.0]]
    got = float(ph.npd_level(p, d, lv, 1000.0, math.sqrt(200.0 * 400.0))[0])
    return numeric(97.0, got, 1e-9, unit="dB", places=4)


_ROTORCRAFT = "Rotorcraft noise (ECAC Doc 32 / NORAH2)"


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 atmospheric attenuation (Table 4)",
    "ΔLa over a 1 km excess path at 1 kHz vs the NORAH2 guidance Table 4, dB",
)
def _chk_doc32_atmospheric() -> Outcome:
    # NORAH2 guidance Table 4: 6.3 dB/km at 1 kHz (ICAO reference conditions).
    got = -float(ph.atmospheric_adjustment([1000.0], 1000.0 + 60.0)[0])
    return numeric(6.3, got, 0.2, unit="dB", places=3)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 spherical spreading",
    "ΔLs at ten times the 60 m hemisphere reference distance (Eq. 24), dB",
)
def _chk_doc32_spreading() -> Outcome:
    return numeric(-20.0, float(ph.spherical_spreading_adjustment(600.0)), 1e-9,
                   unit="dB", places=3)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 ground effect (rigid limit)",
    "ΔLg over a rigid surface at grazing incidence tends to +6 dB (Eq. 29), dB",
)
def _chk_doc32_ground() -> Outcome:
    # Over a near-rigid surface (Zs -> inf), coherent reflection at a small path
    # difference reinforces the direct ray towards the +6.02 dB pressure-doubling
    # limit. A low frequency and near-grazing geometry approaches it.
    got = float(ph.ground_effect_adjustment([20.0], 50.0, 1.2, 400.0, flow_resistivity="H")[0])
    return numeric(6.0, got, 1.0, unit="dB", places=2)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 propagation chain (NORAH2 prototype)",
    "LA of a single-hemisphere emission vs the NORAH2 prototype single-event"
    " history (R22 approach, 223.66 m slant), dB(A)",
)
def _chk_doc32_chain() -> Outcome:
    # NORAH2 prototype ARP Case 4 (R22_H1_APP_STD2_NE, mic at the origin,
    # hr = 0.2 m, sigma = 1e6 Pa·s/m2), row t = 831.26 s: the nearest-hemisphere
    # source spectrum propagated with dLs + dLa + dLg and A-weighted reproduces
    # the tabulated LA = 55.87 dB(A). Spectrum: R22_Approach_53kts_12deg at
    # (phi, theta) = (-88.70, 108.43) deg (EASA.2020.FC.06, (c) EASA).
    bands = np.array([10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0,
                      80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
                      500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0,
                      2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0])
    spec = np.array([35.8, 52.2, 70.0, 63.6, 48.4, 59.2, 53.1, 63.3, 61.2,
                     74.6, 68.7, 61.8, 65.7, 59.7, 59.7, 63.6, 57.9, 57.7,
                     58.6, 61.0, 61.9, 64.7, 65.7, 65.6, 64.1, 61.6, 57.9,
                     55.8, 56.4, 53.4, 52.9])
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217

    def _ra(x: np.ndarray) -> np.ndarray:
        return (f4**2 * x**4) / ((x**2 + f1**2)
                                 * np.sqrt((x**2 + f2**2) * (x**2 + f3**2))
                                 * (x**2 + f4**2))

    level = (spec + float(ph.spherical_spreading_adjustment(223.66))
             + ph.atmospheric_adjustment(bands, 223.66)
             + ph.ground_effect_adjustment(bands, 5.0, 0.2, 223.607,
                                           flow_resistivity=1.0e6)
             + 20.0 * np.log10(_ra(bands) / _ra(np.array(1000.0))))
    got = float(10.0 * np.log10(np.sum(10.0 ** (level / 10.0))))
    return numeric(55.87, got, 0.1, unit="dB(A)", places=3)


def _uniform_hemisphere(level: float, bands: list[float] | None = None) -> Any:
    """A synthetic hemisphere with one uniform level on the standard 10° grid."""
    freqs = np.asarray(bands if bands is not None else [50.0], dtype=np.float64)
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    return ph.RotorcraftHemisphere(
        freqs, az, po, np.full((az.size, po.size, freqs.size), level))


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 flight-condition interpolation (NORAH2 Eq. 8)",
    "Distance-scaled triangle blend of three uniform hemispheres, hand-checked, dB",
)
def _chk_doc32_flight_condition() -> Outcome:
    # Conditions (V, gamma) = (50, 0), (70, 0), (60, 10); query (60, 2.5).
    # Normalised (Eq. 3-6, Ffc = 2, spans 20 kt / 10 deg): points (2.5, 0),
    # (3.5, 0), (3.0, 2), query (3.0, 0.5) -> deltas sqrt(0.5), sqrt(0.5), 1.5
    # (Eq. 7) -> weights 0.404629, 0.404629, 0.190743 (Eq. 8). Uniform levels
    # 100 / 90 / 95 dB blend to 10*lg(sum w*10^(L/10)) = 97.0367 dB by hand.
    hems = [_uniform_hemisphere(100.0), _uniform_hemisphere(90.0),
            _uniform_hemisphere(95.0)]
    got = float(ph.interpolated_source_level(
        hems, [50.0, 70.0, 60.0], [0.0, 0.0, 10.0], 60.0, 2.5, 0.0, 90.0)[0])
    return numeric(97.0367, got, 1e-3, unit="dB", places=4)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 flight-path kinematics (Eq. 17)",
    "Airspeed of a straight climbing track, 40 m/s ground speed at a 5° path angle, m/s",
)
def _chk_doc32_kinematics() -> Outcome:
    # Constant-velocity track: Vg = 40 m/s at heading 30 deg, path angle 5 deg
    # -> VA = 40/cos(5 deg) = 40.152786 m/s (Eq. 16/17), by hand.
    t = np.arange(0.0, 10.5, 0.5)
    vg, gamma = 40.0, np.radians(5.0)
    heading = np.radians(30.0)
    pos = np.column_stack([vg * np.sin(heading) * t, vg * np.cos(heading) * t,
                           100.0 + vg * np.tan(gamma) * t])
    kin = ph.flight_path_kinematics(t, pos)
    return numeric(40.152786, float(kin.airspeed[10]), 1e-4, unit="m/s", places=5)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 retarded time (Eq. 22)",
    "Recorded-time delay at 100 m slant distance, r/c with c = 346.1 m/s, s",
)
def _chk_doc32_retarded_time() -> Outcome:
    # Level flyover directly over the receiver: at the closest-approach step the
    # slant distance is 101.2 - (0 + 1.2) = 100.0 m and t_r - t_e = 100/346.1
    # = 0.288934 s, by hand.
    t = np.arange(0.0, 20.5, 0.5)
    pos = np.column_stack([np.zeros_like(t), 50.0 * (t - 10.0),
                           np.full_like(t, 101.2)])
    res = ph.rotorcraft_event_level(
        [_uniform_hemisphere(100.0)], [50.0], [0.0], t, pos, (0.0, 0.0),
        receiver_height=1.2, flow_resistivity="H")
    k = int(np.argmin(res.distance))
    return numeric(0.288934, float(res.times[k] - res.emission_times[k]), 1e-5,
                   unit="s", places=6)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 single event (Eq. 27)",
    "SEL − LASmax of a constant-speed level flyover, 10·lg(π·d/V) closed form, dB",
)
def _chk_doc32_event_sel() -> Outcome:
    # Single 31.5 Hz band over class-H ground with a 0.1 m receiver: dLg stays
    # within 0.03 dB of the +6 dB pressure doubling along the whole path, so the
    # exposure integral is the Lorentzian closed form SEL - LASmax =
    # 10·lg(pi*d/V) = 7.982 dB for d = 100 m and V = 50 m/s (truncation and
    # absorption stay below 0.1 dB with the track spanning +-6 km).
    t = np.arange(0.0, 240.1, 0.5)
    pos = np.column_stack([np.zeros_like(t), 50.0 * (t - 120.0),
                           np.full_like(t, 100.1)])
    res = ph.rotorcraft_event_level(
        [_uniform_hemisphere(100.0, [31.5])], [50.0], [0.0], t, pos, (0.0, 0.0),
        receiver_height=0.1, flow_resistivity="H")
    return numeric(7.982, res.sel - res.la_max, 0.1, unit="dB", places=3)


@register(
    _ROTORCRAFT,
    "NORAH2 guidance mean ground plane (Eq. 36-40)",
    "Intercept of the plane fitted to a symmetric 20 m roofline, hand-checked, m",
)
def _chk_doc32_mean_plane() -> Outcome:
    # Roof (0,0)-(100,20)-(200,0): by symmetry the continuous least-squares
    # line is horizontal at the mean height, area/span = 2000/200 = 10 m.
    res = ph.mean_ground_plane([0.0, 100.0, 200.0], [0.0, 20.0, 0.0])
    return numeric(10.0, res.intercept, 1e-6, unit="m", places=4)


@register(
    _ROTORCRAFT,
    "NORAH2 guidance mean flow resistivity (Eq. 41)",
    "Log-average of equal 1e4 and 1e6 Pa·s/m2 halves, hand-checked, Pa·s/m2",
)
def _chk_doc32_mean_sigma() -> Outcome:
    # Equal lengths: sigma_bar = 10^((log 1e4 + log 1e6)/2) = 1e5 by hand.
    got = ph.mean_flow_resistivity([120.0, 120.0], [1.0e4, 1.0e6])
    return numeric(1.0e5, got, 1e-3, unit="Pa·s/m²", places=1)


@register(
    _ROTORCRAFT,
    "NORAH2 guidance diffraction at grazing (Eq. 42)",
    "Pure diffraction with the edge on the line of sight, 10·lg 3, dB",
)
def _chk_doc32_diffraction_grazing() -> Outcome:
    got = float(ph.diffraction_attenuation([1000.0], 0.0, edge_height=10.0)[0])
    return numeric(4.7712, got, 1e-4, unit="dB", places=4)


@register(
    _ROTORCRAFT,
    "NORAH2 guidance screening path difference (§A.4.5)",
    "Rubber-band delta over a 40 m hill, hand-checked geometry, m",
)
def _chk_doc32_screening_delta() -> Outcome:
    # Source (0, 20), receiver (400, 1.2), single edge at (200, 40):
    # delta = SO + OR - SR = sqrt(200^2+20^2) + sqrt(200^2+38.8^2)
    #         - sqrt(400^2+18.8^2) = 4.28480 m by hand.
    res = ph.terrain_screening_adjustment(
        [500.0], (0.0, 20.0), (400.0, 1.2),
        [0.0, 190.0, 200.0, 210.0, 400.0], [0.0, 0.0, 40.0, 0.0, 0.0])
    expected = (np.hypot(200.0, 20.0) + np.hypot(200.0, 38.8)
                - np.hypot(400.0, 18.8))
    return numeric(float(expected), res.path_difference, 1e-9, unit="m", places=5)


@register(
    _AIRCRAFT,
    "SAE ARP 5534 pure-tone coefficient (ISO 9613-1)",
    "Mid-band α at 1 kHz, 25 °C, 70 % RH, 101.325 kPa, dB/m",
)
def _chk_arp5534_coefficient() -> Outcome:
    # ARP 5534 §3.1: the pure-tone coefficient is the ISO 9613-1 one.
    expected = float(ph.air_attenuation(1000.0, 25.0, 70.0, 101.325, exact_midband=True))
    res = ph.sae_band_attenuation([1000.0], 1000.0, temperature=25.0, relative_humidity=70.0)
    return numeric(expected, float(res.coefficient[0]), 1e-9, unit="dB/m", places=6)

# ICAO Doc 9501 ETM Vol. I (2018) Table 3-7 turbofan spectrum (bands 1-2 blank).
_ETM_SPL_37 = [
    -999.0, -999.0, 70.0, 62.0, 70.0, 80.0, 82.0, 83.0, 76.0, 80.0,
    80.0, 79.0, 78.0, 80.0, 78.0, 76.0, 79.0, 85.0, 79.0, 78.0, 71.0, 60.0, 54.0, 45.0,
]
# ICAO Doc 9501 ETM Vol. I (2018) Table 4-4 integrated-method EPNL example.
_ETM_PNLTR_44 = [
    84.62, 85.84, 85.37, 88.57, 88.82, 88.03, 88.76, 87.06, 86.92, 90.39, 89.89,
    91.00, 90.08, 89.71, 89.61, 90.21, 91.14, 92.10, 93.68, 94.89, 95.87, 97.06,
    97.40, 96.23, 94.73, 92.30, 88.75, 86.96, 85.41, 83.88, 83.01,
]
_ETM_DTR_44 = [
    0.3950, 0.3950, 0.3951, 0.3951, 0.3952, 0.3953, 0.3954, 0.3956, 0.3957, 0.3960,
    0.3963, 0.3967, 0.3973, 0.3981, 0.3992, 0.4009, 0.4033, 0.4066, 0.4108, 0.4153,
    0.4196, 0.4231, 0.4256, 0.4273, 0.4285, 0.4294, 0.4299, 0.4304, 0.4307, 0.4309,
    0.4311,
]


@register(
    _AIRCRAFT,
    "ICAO Annex 16 Vol. I App. 2 Table A2-3",
    "Perceived noisiness at SPL(b), 1 kHz band, in noys",
)
def _chk_ac_noy() -> Outcome:
    spl = np.full(24, -999.0)
    spl[13] = 40.0  # 1000 Hz, SPL(b) -> n = 1
    return numeric(1.0, float(ph.perceived_noisiness(spl)[13]), 1e-6, places=4)


@register(
    _AIRCRAFT,
    "ICAO Doc 9501 ETM Vol. I Table 3-7",
    "Tone correction of the turbofan example, dB",
)
def _chk_ac_tone() -> Outcome:
    return numeric(2.0, ph.tone_correction(_ETM_SPL_37), 1e-6, places=4)


@register(
    _AIRCRAFT,
    "ICAO Doc 9501 ETM Vol. I Table 4-4",
    "Integrated-method reference EPNL, EPNdB",
)
def _chk_ac_epnl() -> Outcome:
    epnl, _, _, _ = ph.epnl_from_pnlt(np.array(_ETM_PNLTR_44), np.array(_ETM_DTR_44))
    return numeric(92.61892, epnl, 1e-2, unit="EPNdB", places=3)


@register(
    _AIRCRAFT,
    "IEC 61265:1995 Table 1",
    "Directional-response tolerance at 4 kHz / 90°, dB",
)
def _chk_ac_iec61265() -> Outcome:
    from phonometry.metrology.compliance import _iec61265_directional_limit

    return numeric(2.0, _iec61265_directional_limit(4000.0, 90.0), 1e-9, unit="dB", places=1)


# ===========================================================================
# Wind-turbine noise (IEC 61400-11)
# ===========================================================================
_WIND_TURBINE = "Wind-turbine noise (IEC 61400-11)"


@register(
    _WIND_TURBINE,
    "IEC 61400-11:2012 Formula 30",
    "Critical bandwidth about a 500 Hz tone, Hz",
)
def _chk_wt_critical_bandwidth() -> Outcome:
    from phonometry.environmental.wind_turbine_noise import critical_bandwidth

    expected = 25.0 + 75.0 * (1.0 + 1.4 * (500.0 / 1000.0) ** 2) ** 0.69
    return numeric(expected, critical_bandwidth(500.0), 1e-6, unit="Hz", places=3)


@register(
    _WIND_TURBINE,
    "IEC 61400-11:2012 Formula 26",
    "Apparent sound power level of a single band, dB re 1 pW",
)
def _chk_wt_apparent_power() -> Outcome:
    r1 = 150.0
    expected = 100.0 - 6.0 + 10.0 * math.log10(4.0 * math.pi * r1**2)
    return numeric(expected, ph.apparent_sound_power_level([100.0], r1), 1e-4, unit="dB", places=4)


@register(
    _WIND_TURBINE,
    "IEC 61400-11:2012 Formulae 31-34",
    "Tonal audibility of a synthetic clean tone, dB",
)
def _chk_wt_tonal_audibility() -> Outcome:
    df = 2.0
    freqs = np.arange(440.0, 560.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[int(np.argmin(np.abs(freqs - 500.0)))] = 60.0
    res = ph.wind_turbine_tonality(levels, freqs)
    return numeric(16.38, res.tonal_audibility, 6e-2, unit="dB", places=2)


_POROUS = "Porous & multilayer absorbers (Mechel / Bies / Cox & D'Antonio)"

# Shared porous-domain constants: the digitization point X = rho f / sigma
# = 0.1 with sigma = 20 kPa s/m2 and the Bies 5e Appendix D air state.
_PA_SIGMA = 20000.0
_PA_RHO0 = 1.205
_PA_C0 = 343.0


@register(
    _POROUS,
    "Bies 5e App. D Table D.1 / Mechel 2e G.11 (2)",
    "Delany-Bazley normalised Zc at X = 0.1, real part",
)
def _chk_porous_db_real() -> Outcome:
    f = np.array([0.1 * _PA_SIGMA / _PA_RHO0])
    res = ph.delany_bazley(
        f, _PA_SIGMA, speed_of_sound=_PA_C0, air_density=_PA_RHO0
    )
    return numeric(
        ref.POROUS_DB_ZC_EXPECTED.real,
        float(res.normalized_impedance[0].real), 1e-9,
    )


@register(
    _POROUS,
    "Bies 5e App. D Table D.1 / Mechel 2e G.11 (2)",
    "Delany-Bazley normalised Zc at X = 0.1, imaginary part",
)
def _chk_porous_db_imag() -> Outcome:
    f = np.array([0.1 * _PA_SIGMA / _PA_RHO0])
    res = ph.delany_bazley(
        f, _PA_SIGMA, speed_of_sound=_PA_C0, air_density=_PA_RHO0
    )
    return numeric(
        ref.POROUS_DB_ZC_EXPECTED.imag,
        float(res.normalized_impedance[0].imag), 1e-9,
    )


@register(
    _POROUS,
    "Miki 1990 Eqs. (30)-(34)",
    "Miki normalised wavenumber at f/sigma = 0.1, real part",
)
def _chk_porous_miki() -> Outcome:
    f = np.array([0.1 * _PA_SIGMA])
    res = ph.miki(f, _PA_SIGMA, speed_of_sound=_PA_C0, air_density=_PA_RHO0)
    return numeric(
        ref.POROUS_MIKI_K_EXPECTED.real,
        float(res.normalized_wavenumber[0].real), 1e-9,
    )


@register(
    _POROUS,
    "Johnson et al. 1987 / Cox & D'Antonio 3e Eq. (6.19)",
    "JCA static viscous limit j w rho_e -> sigma, Pa s/m2",
)
def _chk_porous_jca_dc() -> Outcome:
    f = np.array([1e-3])
    res = ph.johnson_champoux_allard(
        f, _PA_SIGMA, porosity=0.95, tortuosity=1.3,
        viscous_length=6e-5, thermal_length=1.2e-4, air_density=_PA_RHO0,
    )
    value = float((1j * 2.0 * math.pi * f * res.effective_density)[0].real)
    return numeric(_PA_SIGMA, value, 1e-4, rel=True, unit="Pa s/m2", places=1)


@register(
    _POROUS,
    "Mechel 2e Sect. D.3 Eq. (1)",
    "Hard-backed layer: TMM vs -j Zc cot(kd), max rel deviation",
)
def _chk_porous_rigid_backed() -> Outcome:
    f = np.linspace(200.0, 4000.0, 200)
    med = ph.delany_bazley(f, _PA_SIGMA, air_density=_PA_RHO0)
    res = ph.layered_absorber(
        f, [ph.PorousLayer(0.05, med)],
        speed_of_sound=_PA_C0, air_density=_PA_RHO0,
    )
    zs_ref = -1j * med.characteristic_impedance / np.tan(med.wavenumber * 0.05)
    dev = float(np.max(np.abs(res.surface_impedance - zs_ref) / np.abs(zs_ref)))
    return numeric(0.0, dev, 1e-10, places=4)


@register(
    _POROUS,
    "Lossless-layer limit (Mechel 2e Sect. D.3-D.4)",
    "Air cavity over a rigid wall at lambda/4: alpha",
)
def _chk_porous_air_cavity() -> Outcome:
    d = 0.1
    f = np.array([_PA_C0 / (4.0 * d)])
    res = ph.layered_absorber(
        f, [ph.AirLayer(d)], speed_of_sound=_PA_C0, air_density=_PA_RHO0
    )
    return numeric(0.0, float(res.absorption[0]), 1e-12, places=4)


@register(
    _POROUS,
    "Mechel 2e Sect. D.5",
    "Maximum statistical absorption of a locally reacting plane",
)
def _chk_porous_statistical_max() -> Outcome:
    z = np.linspace(1.0, 3.0, 2001).astype(complex)
    value = float(np.max(ph.statistical_absorption(z)))
    return numeric(ref.POROUS_STATISTICAL_ALPHA_MAX, value, 1e-3, places=3)


@register(
    _POROUS,
    "Cox & D'Antonio 3e Eq. (7.9)",
    "Membrane resonance 60/sqrt(m d), m = 5 kg/m2, d = 5 cm, Hz",
)
def _chk_porous_membrane_resonance() -> Outcome:
    value = ph.membrane_resonance_frequency(
        surface_density=5.0, cavity_depth=0.05,
        speed_of_sound=_PA_C0, air_density=_PA_RHO0,
    )
    return numeric(60.0 / math.sqrt(5.0 * 0.05), value, 0.02, rel=True,
                   unit="Hz", places=2)


@register(
    _POROUS,
    "Maa 1998 Fig. 5 / Cox & D'Antonio 3e Fig. 7.28",
    "Microperforated panel (d=t=0.2 mm, b=2.5 mm, D=6 cm): peak alpha",
)
def _chk_porous_mpp_peak() -> Outcome:
    eps = (math.pi / 4.0) * (ref.MAA_FIG5_DIAMETER / ref.MAA_FIG5_SEPARATION) ** 2
    f = np.linspace(100.0, 4000.0, 2000)
    res = ph.layered_absorber(
        f,
        [
            ph.MicroperforatedPlateLayer(
                ref.MAA_FIG5_THICKNESS, ref.MAA_FIG5_DIAMETER / 2.0, eps
            ),
            ph.AirLayer(ref.MAA_FIG5_CAVITY),
        ],
        speed_of_sound=_PA_C0, air_density=_PA_RHO0,
    )
    return numeric(0.95, float(np.max(res.absorption)), 0.05, places=3)


@register(
    _POROUS,
    "Maa 1998 Eqs. (5a)/(10)",
    "MPP peak absorption vs 4r/(1+r)^2 with Maa's printed resistance",
)
def _chk_porous_maa_peak_closed_form() -> Outcome:
    # Independent expected value: the relative resistance r comes from
    # Maa's printed wide-range approximation Eq. (5a) (with the surface
    # end correction), not from the library's exact Bessel kernel; the
    # peak absorption must then satisfy Eq. (10), alpha0 = 4r/(1+r)^2,
    # within the paper's stated ~6 % accuracy of the approximation.
    eta = 1.84e-5
    d = ref.MAA_FIG5_DIAMETER
    t = ref.MAA_FIG5_THICKNESS
    eps = (math.pi / 4.0) * (d / ref.MAA_FIG5_SEPARATION) ** 2
    f = np.linspace(400.0, 1200.0, 4000)
    res = ph.layered_absorber(
        f,
        [
            ph.MicroperforatedPlateLayer(t, d / 2.0, eps),
            ph.AirLayer(ref.MAA_FIG5_CAVITY),
        ],
        speed_of_sound=_PA_C0, air_density=_PA_RHO0,
    )
    i = int(np.argmax(res.absorption))
    omega = 2.0 * math.pi * float(f[i])
    k_perf = d * math.sqrt(omega * _PA_RHO0 / (4.0 * eta))
    k_r = math.sqrt(1.0 + k_perf**2 / 32.0) + (
        math.sqrt(2.0) / 32.0
    ) * k_perf * (d / t)
    r_rel = 32.0 * eta * t * k_r / (eps * _PA_RHO0 * _PA_C0 * d**2)
    expected = 4.0 * r_rel / (1.0 + r_rel) ** 2
    return numeric(expected, float(res.absorption[i]), 0.02, places=3,
                   expected_label=f"4r/(1+r)^2 = {expected:.3f}")


# ===========================================================================
# Slow-sound slit + Helmholtz-resonator perfect absorbers (Jimenez et al.)
# ===========================================================================
_SLOW_SOUND = "Slow-sound perfect absorbers (Jimenez et al. Appl. Sci. 2017)"


@register(
    _SLOW_SOUND,
    "Jimenez et al. Appl. Sci. 2017 Eq. (9)",
    "Critical coupling: alpha at the design frequency (300 Hz, normal)",
)
def _chk_slow_sound_perfect_absorption() -> Outcome:
    res = ph.HelmholtzResonator(
        neck_length=1.0e-3, neck_side=3.0e-3,
        cavity_length=30.0e-3, cavity_side=27.0e-3,
    )
    design = ph.critical_coupling_design(
        300.0, res, lattice_step=3.0e-2, period=5.0e-2,
        air_density=_PA_RHO0, speed_of_sound=_PA_C0,
    )
    out = ph.slit_helmholtz_absorber(
        np.array([300.0]), design.resonator, slit_height=design.slit_height,
        lattice_step=3.0e-2, period=5.0e-2,
        air_density=_PA_RHO0, speed_of_sound=_PA_C0,
    )
    # The check requires the solver to have converged: a non-converged design
    # fails the check outright rather than silently reporting its (imperfect)
    # absorption.
    computed = float(out.absorption[0]) if design.converged else float("nan")
    return numeric(1.0, computed, 1e-3, places=4)


@register(
    _SLOW_SOUND,
    "Poiseuille limit (Stinson 1991)",
    "Slit: j w rho_s -> 12 eta / h^2 as w -> 0 (h = 1.2 mm)",
)
def _chk_slow_sound_slit_resistivity() -> Outcome:
    eta = 1.84e-5
    h = 1.2e-3
    f = np.array([1.0e-2])
    rho_s, _ = ph.slit_effective_properties(
        f, slit_height=h, air_density=_PA_RHO0, viscosity=eta,
    )
    sigma = float((1j * 2.0 * math.pi * f * rho_s)[0].real)
    return numeric(12.0 * eta / h**2, sigma, 1e-3, rel=True,
                   unit="Pa s/m2", places=1)


@register(
    _SLOW_SOUND,
    "Poiseuille limit (Stinson 1991)",
    "Square duct: j w rho -> 28.454 eta / w^2 as w -> 0 (w = 3 mm)",
)
def _chk_slow_sound_duct_resistivity() -> Outcome:
    eta = 1.84e-5
    side = 3.0e-3
    f = np.array([1.0e-2])
    rho, _ = ph.rectangular_duct_properties(
        f, side=side, air_density=_PA_RHO0, viscosity=eta,
    )
    sigma = float((1j * 2.0 * math.pi * f * rho)[0].real)
    return numeric(28.454 * eta / side**2, sigma, 2e-3, rel=True,
                   unit="Pa s/m2", places=1)


# ===========================================================================
# Program loudness (ITU-R BS.1770-5 / EBU R 128)
# ===========================================================================
_PROGRAM_LOUDNESS = "Program loudness (ITU-R BS.1770 / EBU R 128)"
_EBU_FS = 48000


def _ebu_tone(level_dbfs: float, duration_s: float, freq: float = 1000.0) -> np.ndarray:
    """A sine with per-sample peak level ``level_dbfs`` re full scale."""
    t = np.arange(round(duration_s * _EBU_FS)) / _EBU_FS
    return np.asarray(10.0 ** (level_dbfs / 20.0) * np.sin(2 * np.pi * freq * t))


def _ebu_stereo_steps(segments: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Concatenated 1 kHz tone steps applied in phase to both channels."""
    x = np.concatenate([_ebu_tone(lvl, dur) for lvl, dur in segments])
    return np.vstack([x, x])


@register(
    _PROGRAM_LOUDNESS,
    "ITU-R BS.1770-5 Annex 1",
    "997 Hz sine at 0 dB FS on the left channel, LKFS",
)
def _chk_bs1770_anchor() -> Outcome:
    x = np.vstack([_ebu_tone(0.0, 10.0, freq=997.0), np.zeros(10 * _EBU_FS)])
    computed = ph.integrated_loudness(x, _EBU_FS)
    return numeric(ref.BS1770_ANCHOR_997_LKFS, computed, 0.01, unit="LKFS", places=2)


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 1",
    "Integrated loudness of the -23 dBFS stereo sine, LUFS",
)
def _chk_tech3341_case1() -> Outcome:
    _, segments, expected = ref.EBU_TECH3341_INTEGRATED_CASES[0]
    computed = ph.integrated_loudness(_ebu_stereo_steps(segments), _EBU_FS)
    return numeric(expected, computed, ref.EBU_TECH3341_TOL_LU, unit="LUFS", places=2)


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 5",
    "Gated integrated loudness of the -26/-20/-26 dBFS steps, LUFS",
)
def _chk_tech3341_case5() -> Outcome:
    _, segments, expected = ref.EBU_TECH3341_INTEGRATED_CASES[4]
    computed = ph.integrated_loudness(_ebu_stereo_steps(segments), _EBU_FS)
    return numeric(expected, computed, ref.EBU_TECH3341_TOL_LU, unit="LUFS", places=2)


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 6",
    "Integrated loudness of the 5.0-channel sine (Table 3 weights), LUFS",
)
def _chk_tech3341_case6() -> Outcome:
    x = np.vstack([_ebu_tone(lvl, 20.0) for lvl in ref.EBU_TECH3341_CASE6_LEVELS])
    computed = ph.integrated_loudness(x, _EBU_FS)
    return numeric(
        ref.EBU_TECH3341_CASE6_EXPECTED,
        computed,
        ref.EBU_TECH3341_TOL_LU,
        unit="LUFS",
        places=2,
    )


def _ebu_true_peak_case(index: int) -> tuple[float, float]:
    """Synthesize a Tech 3341 true-peak tone case; return (expected, computed)."""
    _, freq_ratio, amplitude, phase_deg, expected = ref.EBU_TECH3341_TRUE_PEAK_CASES[
        index
    ]
    t = np.arange(_EBU_FS) / _EBU_FS
    x = amplitude * np.sin(
        2 * np.pi * (freq_ratio * _EBU_FS) * t + np.deg2rad(phase_deg)
    )
    n = int(0.01 * _EBU_FS)
    x[:n] *= np.linspace(0.0, 1.0, n)
    x[-n:] *= np.linspace(1.0, 0.0, n)
    return expected, float(ph.true_peak_level(x, _EBU_FS))


def _true_peak_outcome(expected: float, computed: float) -> Outcome:
    """Tech 3341 true-peak verdict with its asymmetric +0.2/-0.4 dB window."""
    delta = computed - expected
    passed = -ref.EBU_TECH3341_TP_TOL_DOWN <= delta <= ref.EBU_TECH3341_TP_TOL_UP
    return Outcome(
        expected=(
            f"{_fmt(expected, 'dBTP', 2)} "
            f"(+{ref.EBU_TECH3341_TP_TOL_UP:g}/-{ref.EBU_TECH3341_TP_TOL_DOWN:g} dB)"
        ),
        computed=_fmt(computed, "dBTP", 2),
        delta=_fmt(delta, "dBTP", _DELTA_PLACES),
        passed=passed,
    )


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 15",
    "True-peak level of the fs/4 sine at 0.5 FFS, dBTP",
)
def _chk_tech3341_case15() -> Outcome:
    return _true_peak_outcome(*_ebu_true_peak_case(0))


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 19",
    "True-peak level of the fs/4 sine at 1.41 FFS, dBTP",
)
def _chk_tech3341_case19() -> Outcome:
    return _true_peak_outcome(*_ebu_true_peak_case(4))


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3342:2023 Table 1 case 1",
    "Loudness range of the -20/-30 dBFS tone steps, LU",
)
def _chk_tech3342_case1() -> Outcome:
    _, levels, expected = ref.EBU_TECH3342_LRA_CASES[0]
    x = _ebu_stereo_steps(tuple((lvl, 20.0) for lvl in levels))
    res = ph.program_loudness(x, _EBU_FS)
    return numeric(
        expected, res.loudness_range, ref.EBU_TECH3342_TOL_LU, unit="LU", places=2
    )


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3342:2023 Table 1 case 3",
    "Loudness range of the -40/-20 dBFS tone steps, LU",
)
def _chk_tech3342_case3() -> Outcome:
    _, levels, expected = ref.EBU_TECH3342_LRA_CASES[2]
    x = _ebu_stereo_steps(tuple((lvl, 20.0) for lvl in levels))
    res = ph.program_loudness(x, _EBU_FS)
    return numeric(
        expected, res.loudness_range, ref.EBU_TECH3342_TOL_LU, unit="LU", places=2
    )
_FDTD = "2D FDTD wave simulation (Attenborough & Van Renterghem 2021, Ch. 4)"


@register(
    _FDTD,
    "Rigid rectangular box eigenfrequency",
    "Mode (1,1) of a 1.0 x 0.7 m rigid box, f = (c/2)*sqrt(1/lx^2 + 1/ly^2), Hz",
)
def _chk_fdtd_box_mode() -> Outcome:
    lx, ly, dx, c = 1.0, 0.7, 0.02, 343.0
    nx, ny = round(lx / dx), round(ly / dx)
    res = ph.fdtd_simulation(
        c, dx, 0.35, shape=(ny, nx),
        sources=[ph.GaussianPulse(ix=7, iy=5, width=2.0e-4)],
        probes=[(nx - 4, ny - 3)])
    expected = 0.5 * c * math.hypot(1.0 / lx, 1.0 / ly)
    pressure = res.pressures[0]
    spec = np.abs(np.fft.rfft(pressure * np.hanning(pressure.size),
                              n=8 * pressure.size))
    freqs = np.fft.rfftfreq(8 * pressure.size, res.dt)
    sel = (freqs > 0.93 * expected) & (freqs < 1.07 * expected)
    measured = float(freqs[sel][np.argmax(spec[sel])])
    return numeric(expected, measured, 1.5, unit="Hz", places=2)


@register(
    _FDTD,
    "Free-field pulse arrival delay",
    "Probe-to-probe delay of a pulse over 0.6 m of air, (r2 - r1)/c, ms",
)
def _chk_fdtd_pulse_delay() -> Outcome:
    c, dx = 343.0, 0.01
    res = ph.fdtd_simulation(
        c, dx, 6.5e-3, shape=(200, 300),
        sources=[ph.GaussianPulse(ix=40, iy=100, width=1.5e-4)],
        probes=[(100, 100), (160, 100)],
        boundaries="absorbing", absorbing_layer_cells=30)
    t1 = res.times[int(np.argmax(res.pressures[0]))]
    t2 = res.times[int(np.argmax(res.pressures[1]))]
    expected = (160 - 100) * dx / c * 1e3
    return numeric(expected, (t2 - t1) * 1e3, 0.05, unit="ms", places=3)


_NTFF_CACHE: dict[str, Any] = {}


def _ntff_monopole() -> tuple[Any, complex, float]:
    """Far-field pattern of an enclosed CW monopole and its analytic level.

    One steady-state 2 kHz run: a contour probe around the source feeds
    the Kirchhoff-Helmholtz far-field integral, and an independent probe
    cell 0.5 m away (open air, clear of the sponge ramp) fits the
    amplitude ``A`` of the analytic line-source field ``p = A H0(2)(k r)``,
    whose exact far-field pattern is the omnidirectional
    ``A sqrt(2 / (pi k)) exp(j pi / 4)``.
    """
    if "monopole" in _NTFF_CACHE:
        out = _NTFF_CACHE["monopole"]
        return cast("tuple[Any, complex, float]", out)
    from scipy.special import hankel2

    c, dx, f = 343.0, 0.005, 2000.0
    k = 2.0 * np.pi * f / c
    sim = ph.FDTD2D(c, dx, shape=(300, 300), sponge_width=40)
    sim.add_source(ph.CWSource(ix=150, iy=150, frequency=f, ramp_cycles=4.0))
    probe = sim.add_contour_probe(90, 210, 90, 210, frequencies=[f])
    sim.run(round(4.5e-3 / sim.dt))
    probe.reset()
    probe_col = 250          # open air: the sponge ramp starts at 260
    acc = 0.0 + 0.0j
    for _ in range(round(10.0 / f / sim.dt)):
        sim.step()
        acc += sim.p[150, probe_col] * np.exp(
            -2j * np.pi * f * sim.n * sim.dt)
    amplitude = (2.0 * acc / probe.samples) / hankel2(
        0, k * (probe_col - 150) * dx)
    pattern = ph.far_field_from_contour(
        probe.phasors(f), np.arange(0.0, 360.0, 5.0),
        origin=(150.5 * dx, 150.5 * dx))
    expected = abs(amplitude) * math.sqrt(2.0 / (math.pi * k))
    result = (pattern, complex(amplitude), float(expected))
    _NTFF_CACHE["monopole"] = result
    return result


@register(
    _FDTD,
    "2D Kirchhoff-Helmholtz NTFF: monopole directivity",
    "Far-field pattern ripple of an enclosed line source, dB",
)
def _chk_ntff_monopole_ripple() -> Outcome:
    pattern, _, _ = _ntff_monopole()
    levels = 20.0 * np.log10(np.abs(pattern))
    return numeric(0.0, float(levels.max() - levels.min()), 0.2,
                   unit="dB", places=3)


@register(
    _FDTD,
    "2D Kirchhoff-Helmholtz NTFF: monopole level",
    "NTFF far-field level vs the 2D Green function A sqrt(2/(pi k)), dB",
)
def _chk_ntff_monopole_level() -> Outcome:
    pattern, _, expected = _ntff_monopole()
    mean_level = 20.0 * np.log10(float(np.mean(np.abs(pattern))) / expected)
    return numeric(0.0, mean_level, 0.3, unit="dB", places=3)


# ===========================================================================
# Swept-sine nonlinear analysis & phase utilities (Farina / Novak / B&P)
# ===========================================================================
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


# ===========================================================================
# Spherical-wave ground effect & advanced barriers (Attenborough / Salomons /
# Bies). Oracles are the analytic limits of the Weyl-Van der Pol and diffraction
# closed forms (hard ground, grazing, N -> 0), which are exact.
# ===========================================================================
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


@register(_PANEL, "Hopkins Eq. 4.99/4.101 (Gomperts slit)",
          "Transmission maximum at first resonance")
def _chk_slit_resonance() -> Outcome:
    fr = float(ph.slit_resonance_frequencies(0.1, 0.005, orders=1)[0])
    f = np.linspace(fr - 300.0, fr + 300.0, 601)
    res = ph.slit_transmission_coefficient(f, 0.005, 0.1, field="normal")
    peak = float(f[int(np.argmax(res.transmission_coefficient))])
    return numeric(fr, peak, 15.0, unit="Hz")


# ===========================================================================
# Bending-wave transmission at rigid plate junctions (Cremer / Craik / Hopkins)
# ===========================================================================
_JUNCTION = "Bending-wave plate-junction transmission (Cremer / Craik / Hopkins)"


@register(_JUNCTION, "Hopkins Eq. 5.12 (identical plates)",
          "X-junction corner tau12(0 deg) = 1/8")
def _chk_junction_x_corner_normal() -> Outcome:
    # chi = psi = 1 -> denominator (J2 psi + chi)**2 = 4, numerator 0.5.
    tau = float(ph.corner_transmission_coefficient(0.0, 1.0, 1.0, "X"))
    return numeric(0.125, tau, 1e-9)


@register(_JUNCTION, "Hopkins Eqs 5.12 + 5.6 (identical plates)",
          "X-junction corner angular average = 1/12")
def _chk_junction_x_average() -> Outcome:
    avg = ph.angular_average_transmission_coefficient(1.0, 1.0, "X", section="corner")
    return numeric(1.0 / 12.0, avg, 1e-6)


@register(_JUNCTION, "Hopkins Eqs 5.12 + 5.6 (identical plates)",
          "L-junction corner angular average = 1/3")
def _chk_junction_l_average() -> Outcome:
    avg = ph.angular_average_transmission_coefficient(1.0, 1.0, "L", section="corner")
    return numeric(1.0 / 3.0, avg, 1e-6)


@register(_JUNCTION, "Hopkins Eq. 5.14 (identical plates)",
          "In-line junction tau12(0 deg) = 1")
def _chk_junction_inline_identical() -> Outcome:
    return numeric(1.0, ph.inline_transmission_coefficient(1.0, 1.0), 1e-9)


@register(_JUNCTION, "Hopkins Eq. 5.7 (SEA consistency)",
          "X-junction reciprocity tau_bar_12 / tau_bar_21 = chi")
def _chk_junction_reciprocity() -> Outcome:
    chi = 1.5
    fwd = ph.angular_average_transmission_coefficient(chi, 0.8, "X", section="corner")
    rev = ph.angular_average_transmission_coefficient(
        1.0 / chi, 1.0 / 0.8, "X", section="corner")
    return numeric(chi, fwd / rev, 1e-6)


@register(_JUNCTION, "Hopkins Eq. 5.116 (identical plates, fc_j = f_ref)",
          "X-junction vibration reduction index = 10 lg(12)")
def _chk_junction_kij() -> Outcome:
    # fc_j = f_ref = 1000 Hz cancels the 5 lg(fc_j / f_ref) correction term.
    kij = float(ph.wave_vibration_reduction_index(1.0 / 12.0, 1000.0))
    return numeric(10.0 * math.log10(12.0), kij, 1e-6, unit="dB")


# ===========================================================================
# Atmospheric refraction (Salomons rays + GFPE parabolic equation)
# ===========================================================================
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
    prof = ph.linear_sound_speed_profile(gradient, ground_speed=c0, max_height=3000.0)
    res = ph.atmospheric_ray_paths(prof, source_height=0.0, launch_angles_deg=[angle],
                                   max_range=600.0, n_steps=8000)
    return numeric(turn, float(res.heights[0].max()), 0.1, unit="m", places=3)


@register(
    _ATM_REFRACTION,
    "Salomons Eq. (3.4) (GFPE vs spherical-wave ground effect, homogeneous)",
    "PE relative level at 500 m over grassland vs Weyl-Van der Pol, dB",
)
def _chk_atm_pe_ground_effect() -> Outcome:
    freq, zs, zr, z = 250.0, 1.0, 1.0, 11.0 + 8.0j
    flat = ph.linear_sound_speed_profile(1e-12, ground_speed=343.0, max_height=200.0)
    pe = ph.atmospheric_parabolic_equation(freq, flat, source_height=zs,
                                           impedance=z, max_range=520.0,
                                           max_height=40.0)
    i = int(min(range(pe.ranges.size), key=lambda k: abs(pe.ranges[k] - 500.0)))
    got = float(pe.level_at_height(zr)[i])
    oracle = float(ph.ground_effect([freq], zs, zr, float(pe.ranges[i]),
                                    impedance=z, speed_of_sound=343.0).excess_attenuation[0])
    return numeric(oracle, got, 0.5, unit="dB", places=3)


@register(
    _ATM_REFRACTION,
    "Salomons Eq. (3.4) (GFPE hard ground vs two-ray, homogeneous)",
    "PE relative level at 500 m over a rigid ground vs the coherent two-ray, dB",
)
def _chk_atm_pe_hard_ground() -> Outcome:
    freq, zs, zr = 500.0, 2.0, 2.0
    flat = ph.linear_sound_speed_profile(1e-12, ground_speed=343.0, max_height=200.0)
    pe = ph.atmospheric_parabolic_equation(freq, flat, source_height=zs,
                                           impedance=1e6 + 0j, max_range=520.0,
                                           max_height=150.0)
    i = int(min(range(pe.ranges.size), key=lambda k: abs(pe.ranges[k] - 500.0)))
    r = float(pe.ranges[i])
    r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
    k = 2.0 * math.pi * freq / 343.0
    two_ray = 20.0 * math.log10(abs(1.0 + (r1 / r2) * cmath.exp(1j * k * (r2 - r1))))
    return numeric(two_ray, float(pe.level_at_height(zr)[i]), 0.6, unit="dB", places=3)
# Domain - Electroacoustics (baffled piston, Beranek & Mellow 2e)
# ===========================================================================
_ELECTROACOUSTICS = "Electroacoustics"


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (13.117)",
    "Piston resistance R1(x) = 1 - 2 J1(x)/x at x = 2ka = 2",
)
def _chk_piston_resistance() -> Outcome:
    return numeric(0.423275, float(ph.piston_resistance(2.0)), 1e-5, places=6)


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (13.118)",
    "Piston reactance X1(x) = 2 H1(x)/x at x = 2ka = 2",
)
def _chk_piston_reactance() -> Outcome:
    return numeric(0.646764, float(ph.piston_reactance(2.0)), 1e-5, places=6)


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (13.117) (low-frequency limit)",
    "R1 -> (ka)^2/2 as ka -> 0 (x = 0.02, ka = 0.01)",
)
def _chk_piston_resistance_limit() -> Outcome:
    ka = 0.01
    return numeric(ka**2 / 2.0, float(ph.piston_resistance(2.0 * ka)), 1e-4,
                   rel=True, places=8)


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (4.151)",
    "Radiation mass M = 8 rho a^3 / 3  (a = 0.1 m, rho = 1.206)",
)
def _chk_piston_radiation_mass() -> Outcome:
    res = ph.radiating_piston(0.1, [100.0], density=1.206)
    return numeric(8.0 * 1.206 * 0.1**3 / 3.0, res.radiation_mass, 1e-9,
                   unit="kg", places=8)


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e Eq. (13.102), Table 14.1",
    "First directivity null at ka sin(theta) = 3.8317 (first zero of J1)",
)
def _chk_piston_directivity_null() -> Outcome:
    # ka sin(theta) = 3.8317 at ka = 3.8317, theta = pi/2.
    d = float(ph.piston_directivity(3.8317059702075125, math.pi / 2.0))
    return numeric(0.0, d, 1e-6, places=8)


@register(
    _ELECTROACOUSTICS,
    "Beranek & Mellow 2e §4.19 (half-space baffle)",
    "Directivity index DI -> 10 lg 2 = 3.01 dB as ka -> 0",
)
def _chk_piston_directivity_index() -> Outcome:
    res = ph.radiating_piston(0.01, [1.0])
    return numeric(10.0 * math.log10(2.0), float(res.directivity_index[0]),
                   1e-3, unit="dB")


@register(
    _ELECTROACOUSTICS,
    "Long, Architectural Acoustics 2e, Eq. (18.21)",
    "Omnidirectional mic at Zs = -6 dB: L(H-M) <= L(H-L) - 4 dB",
)
def _chk_feedback_omnidirectional() -> Outcome:
    res = ph.feedback_stability(-6.0, 76.0, 80.0)
    return numeric(80.0 - 4.0, res.maximum_level_at_microphone, 1e-9,
                   unit="dB", places=6)


@register(
    _ELECTROACOUSTICS,
    "Long, Architectural Acoustics 2e, Eq. (18.22)",
    "Cardioid mic (DM = -2 dB) at Zs = -6 dB: L(H-M) <= L(H-L) - 2 dB",
)
def _chk_feedback_cardioid() -> Outcome:
    res = ph.feedback_stability(-6.0, 78.0, 80.0, microphone_directivity=-2.0)
    return numeric(80.0 - 2.0, res.maximum_level_at_microphone, 1e-9,
                   unit="dB", places=6)


@register(
    _ELECTROACOUSTICS,
    "Long, Architectural Acoustics 2e, Eq. (18.23)",
    "Number-of-open-microphones correction 10 lg Nm at Nm = 4",
)
def _chk_open_microphone_correction() -> Outcome:
    return numeric(10.0 * math.log10(4.0), ph.open_microphone_correction(4),
                   1e-12, unit="dB", places=6)


# ===========================================================================
# Domain - Industrial noise control (Bies 5e; silencers, HVAC, enclosures)
# ===========================================================================
_NOISE_CONTROL = "Industrial noise control"


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.111)",
    "Expansion-chamber peak TL = 10 lg[1 + (1/4)(m - 1/m)^2], m = 4 at kL = pi/2",
)
def _chk_expansion_chamber_peak() -> Outcome:
    c, length, s_duct = 343.0, 0.3, 0.01
    f = np.array([c / (4.0 * length)])  # kL = pi/2
    res = ph.expansion_chamber(f, length, 4.0 * s_duct, s_duct, speed_of_sound=c)
    expected = 10.0 * math.log10(1.0 + 0.25 * (4.0 - 0.25) ** 2)
    return numeric(expected, float(res.transmission_loss[0]), 1e-6, unit="dB")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.111)",
    "Expansion-chamber trough TL = 0 at kL = pi (chamber transparent)",
)
def _chk_expansion_chamber_trough() -> Outcome:
    c, length, s_duct = 343.0, 0.3, 0.01
    f = np.array([c / (2.0 * length)])  # kL = pi
    res = ph.expansion_chamber(f, length, 4.0 * s_duct, s_duct, speed_of_sound=c)
    return numeric(0.0, float(res.transmission_loss[0]), 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.44) / Example 8.1",
    "Quarter-wave tube tuning f = c/(4 l_e), l_e = 1.516 m -> 56.6 Hz",
)
def _chk_quarter_wave_tuning() -> Outcome:
    area = math.pi * 0.05**2 / 4.0
    res = ph.quarter_wave_resonator([100.0], area, 1.516, area, speed_of_sound=343.24)
    assert res.resonances is not None
    return numeric(56.6, float(res.resonances[0]), 0.1, unit="Hz", places=2)


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.46)",
    "Helmholtz resonance f0 = (c/2pi) sqrt(S/(l_e V))  (S=1e-4, l_e=0.02, V=1e-3)",
)
def _chk_helmholtz_resonance() -> Outcome:
    c = 343.0
    res = ph.helmholtz_resonator([100.0], 0.01, 1e-4, 0.02, 1e-3, speed_of_sound=c)
    assert res.resonances is not None
    expected = c / (2.0 * math.pi) * math.sqrt(1e-4 / (0.02 * 1e-3))
    return numeric(expected, float(res.resonances[0]), 1e-6, unit="Hz", places=3)


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.73)",
    "Side-branch TL = 20 lg abs(1 + rho c/(2 Sd Zb)) (QWT branch, closed form)",
)
def _chk_side_branch_closed_form() -> Outcome:
    from phonometry.noise_control import silencers as _sl

    f = np.array([120.0])
    zb = _sl.quarter_wave_impedance(f, 0.5, 0.002)
    t = _sl.shunt_matrix(zb)
    tl = float(_sl.transmission_loss(t, inlet_area=0.01, outlet_area=0.01)[0])
    closed = 20.0 * math.log10(abs(1.0 + 1.206 * 343.0 / (2.0 * 0.01 * zb[0])))
    return numeric(closed, tl, 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eqs. (8.141)/(8.148) (four-pole insertion loss)",
    "Insertion loss = transmission loss for the anechoic reference Zs=Zr=rho c/S",
)
def _chk_insertion_loss_equals_tl() -> Outcome:
    from phonometry.noise_control import silencers as _sl

    c, rho, s = 343.0, 1.206, 0.01
    z = rho * c / s
    f = np.array([232.0])  # a frequency with a substantial, positive TL
    t = _sl.expansion_chamber(f, 0.3, 0.04, s).transfer_matrix
    tl = float(_sl.transmission_loss(t, inlet_area=s, outlet_area=s)[0])
    il = float(_sl.insertion_loss(t, source_impedance=z, radiation_impedance=z)[0])
    return numeric(tl, il, 1e-9, unit="dB",
                   expected_label=f"{tl:.4f} dB (= TL)")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.275) (Wells' plenum method)",
    "Plenum TL = -10 lg[S_out(cos0/pi r^2 + (1-a)/(Sw a))] (S_out=.1,r=1,Sw=20,a=.2)",
)
def _chk_plenum_wells() -> Outcome:
    tl = float(ph.plenum_attenuation(0.1, 1.0, 20.0, 0.2))
    direct = 1.0 / (math.pi * 1.0**2)
    reverb = (1.0 - 0.2) / (20.0 * 0.2)
    expected = -10.0 * math.log10(0.1 * (direct + reverb))
    return numeric(expected, tl, 1e-6, unit="dB")


@register(
    _NOISE_CONTROL,
    "Bies 5e Table 8.14 (ASHRAE end reflection, flush)",
    "Duct end reflection D = 200 mm at 125 Hz = 10 dB (table node)",
)
def _chk_end_reflection_table() -> Outcome:
    res = ph.end_reflection_loss([125.0], 0.200, termination="flush")
    return numeric(10.0, float(res.values[0]), 1e-6, unit="dB")


@register(
    _NOISE_CONTROL,
    "Long 2e Eq. 13.1 with Table 13.5 (ASHRAE 1987 fan model)",
    "Forward-curved fan at Q_REF, P_REF, peak efficiency -> K_F + C_BFI at 500 Hz",
)
def _chk_fan_sound_power_reference_point() -> Outcome:
    res = ph.fan_sound_power(
        0.472e-3, 249.0, fan_type="forward_curved", relative_efficiency=100.0
    )
    # Table 13.5 forward-curved 500 Hz entry (36 dB) plus the Table 13.7
    # blade frequency increment of the 500 Hz octave (2 dB).
    return numeric(38.0, float(res.values[3]), 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Long 2e Eq. 14.12 with Table 14.2 (Reynolds lined rectangular duct)",
    "18 x 12 in duct, 6 ft, 1 in lining at 1 kHz -> 1.77 (10/3)^0.695 6 dB",
)
def _chk_lined_rectangular_duct() -> Outcome:
    res = ph.lined_rectangular_duct_attenuation(
        None, 18.0 * 0.0254, 12.0 * 0.0254, 6.0 * 0.3048, 0.0254
    )
    expected = 1.7700 * (10.0 / 3.0) ** 0.695 * 6.0
    return numeric(expected, float(res.values[4]), 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Long 2e Table 14.4 (ASHRAE 1995 lined flexible duct)",
    "8 in diameter, 9 ft long -> 6/8/16/25/28/28/18 dB (table node)",
)
def _chk_flexible_duct_table() -> Outcome:
    res = ph.flexible_duct_insertion_loss(None, 8.0 * 0.0254, 9.0 * 0.3048)
    printed = np.array([6, 8, 16, 25, 28, 28, 18], dtype=float)
    worst = float(np.max(np.abs(res.values - printed)))
    return numeric(0.0, worst, 1e-9, unit="dB",
                   expected_label="0 dB (max |diff| over the 7 bands)")


@register(
    _NOISE_CONTROL,
    "Long 2e Eq. 14.17 (branch power division)",
    "25 per cent split with area-matched branches -> -10 lg 0.25 = 6.02 dB",
)
def _chk_split_loss() -> Outcome:
    loss = ph.split_loss(0.6, [0.15, 0.15, 0.15, 0.15], branch=0)
    return numeric(-10.0 * math.log10(0.25), loss, 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Long 2e Table 14.9 (worked duct-borne sheet, supply path)",
    "Fan to room, 8 octave bands -> 52/42/30/18/9/-2/-2/-1 dB at the receiver",
)
def _chk_duct_path_table_14_9() -> Outcome:
    from phonometry.noise_control.duct_path import DuctElement, duct_path
    from phonometry.noise_control.hvac import OCTAVE_BANDS

    res = duct_path(
        OCTAVE_BANDS,
        [90.0, 86.0, 82.0, 79.0, 77.0, 75.0, 71.0, 61.0],
        [
            DuctElement("Elbow", [0, 1, 2, 3, 3, 3, 3, 3],
                        [41, 39, 36, 29, 20, 6, 0, 0]),
            DuctElement("Silencer", [7, 12, 16, 28, 35, 35, 28, 17],
                        [49, 43, 44, 42, 42, 45, 35, 24]),
            DuctElement("Lined duct 36x24", [2, 2, 3, 7, 15, 12, 11, 9]),
            DuctElement("Split 25%", 6.0),
            DuctElement("Lined duct 18x12", [3, 3, 5, 11, 25, 22, 16, 13]),
            DuctElement("Flexible duct", [14, 14, 16, 15, 17, 22, 16, 13]),
            DuctElement("Diffuser", 0.0, [33, 32, 29, 23, 15, 4, 0, 0]),
        ],
        room_effect=[6, 6, 5, 5, 6, 7, 6, 6],
    )
    printed = np.array([52, 42, 30, 18, 9, -2, -2, -1], dtype=float)
    worst = float(np.max(np.abs(np.round(res.received_level) - printed)))
    # The printed sheet carries whole decibels and its own rounding is not
    # always self-consistent, so 1 dB is the published resolution.
    return numeric(0.0, worst, 1.0, unit="dB",
                   expected_label="0 dB +/-1 (max |diff| over the 8 bands)")


@register(
    _NOISE_CONTROL,
    "Long 2e Eqs. 13.27-13.33 (Reynolds diffuser self-noise)",
    "24 x 24 in rectangular diffuser, 312 cfm, 0.05 in pd -> the 33/32/29/23/15 dB "
    "row of Table 14.9",
)
def _chk_diffuser_sound_power() -> Outcome:
    res = ph.diffuser_sound_power(
        None, (24.0 * 0.0254) ** 2, 312.0 * 0.0004719474432, 0.05 * 249.0
    )
    printed = np.array([33.0, 32.0, 29.0, 23.0, 15.0])
    worst = float(np.max(np.abs(res.values[:5] - printed)))
    return numeric(0.0, worst, 1.0, unit="dB",
                   expected_label="0 dB +/-1 (max |diff| over the five bands)")


@register(
    _NOISE_CONTROL,
    "ASHRAE 2019 Applications Ch. 49 Table 9",
    "Max neck velocity of a supply outlet for design RC(30) -> 2.2 m/s",
)
def _chk_air_terminal_velocity() -> Outcome:
    return numeric(2.2, ph.air_terminal_velocity_limit(30), 1e-9, unit="m/s")


@register(
    _NOISE_CONTROL,
    "Norton & Karczub 2e Eqs. 7.6/7.8/7.9 (problem 7.1 answer)",
    "254 mm duct, steam, 200 m/s: (1,0) cut-on 812 Hz and k_x = -8.23 1/m",
)
def _chk_duct_cut_on_with_flow() -> Outcome:
    res = ph.circular_duct_cut_on(
        0.254, flow_velocity=200.0, speed_of_sound=405.0, count=1
    )
    worst = max(
        abs(float(res.cut_on[0]) - 812.0),
        abs(float(res.axial_wavenumber[0]) + 8.23) * 100.0,
    )
    return numeric(0.0, worst, 1.0,
                   expected_label="0 +/-1 (Hz, and 1/m x100)")


@register(
    _NOISE_CONTROL,
    "Norton & Karczub 2e Eq. 7.10 (problem 7.2 answer)",
    "0.65 x 0.4 m duct, 15 m/s: first three cut-on 264 / 428 / 503 Hz",
)
def _chk_rectangular_duct_cut_on() -> Outcome:
    res = ph.rectangular_duct_cut_on(0.65, 0.4, flow_velocity=15.0, count=3)
    printed = np.array([264.0, 428.0, 503.0])
    worst = float(np.max(np.abs(np.round(res.cut_on) - printed)))
    return numeric(0.0, worst, 1e-9, unit="Hz",
                   expected_label="0 Hz (max |diff| over the 3 modes)")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eqs. (7.103), (7.111) (enclosure, fully absorbing limit)",
    "Enclosure correction C -> 10 lg 0.3 = -5.23 dB as alpha_i -> 1",
)
def _chk_enclosure_floor() -> Outcome:
    res = ph.enclosure_insertion_loss([40.0], 6.0, 5.0, 0.999999)
    return numeric(10.0 * math.log10(0.3), float(res.correction[0]), 1e-3,
                   unit="dB")


# ===========================================================================
# Markdown rendering
# ===========================================================================
def _snap(value: float, eps: float = 5e-4) -> float:
    """Snap a near-zero value to +0 so displays avoid a spurious ``-0.00``."""
    return 0.0 if abs(value) < eps else value


def _status(passed: bool) -> str:
    return "&#9989;" if passed else "&#10060;"


def _domains() -> list[str]:
    seen: list[str] = []
    for chk in CHECKS:
        if chk.domain not in seen:
            seen.append(chk.domain)
    return seen


# Architectures that cannot meet the IEC 61260-1 mask by construction, with
# the reason (for the "By design" label, not a failure verdict).
_BY_DESIGN: dict[str, str] = {
    "cheby1": "passband ripple",
    "ellip": "passband ripple",
    "bessel": "soft rolloff",
}


def _filter_verdict(arch: str, fc: FilterClass) -> str:
    if fc.overall_class == 1:
        return "Class 1 (default)" if arch == "butter" else "Class 1"
    if fc.overall_class == 2:
        return "Class 2"
    reason = _BY_DESIGN.get(arch)
    return f"By design ({reason})" if reason else "not compliant"


def _numerical_validation_section(filters_ok: bool) -> str:
    # Collapsed like the per-domain groups so the report stays compact by
    # default; it springs open whenever the filters/weightings domain has a
    # failing row, exactly like those groups do.
    emoji = "&#9989;" if filters_ok else "&#10060;"
    opened = "" if filters_ok else " open"
    lines: list[str] = []
    lines.append(f"<details{opened}>")
    lines.append(
        f"<summary>{emoji} <b>Numerical validation - filters &amp; "
        "weightings</b>: class showcase (IEC 61260-1 · IEC 61672-1 · "
        "ISO 7196)</summary>"
    )
    lines.append("")
    lines.append(
        "**IEC 61260-1:2014 class per filter architecture** (order 6, "
        "one-third-octave, 100 Hz-10 kHz, fs = 48 kHz). For each architecture "
        "the table shows, at its *binding* band, the measured relative "
        "attenuation and the class-1 limit it must clear, so the number and "
        "the range it must sit in are both visible. A positive margin means "
        "the acceptance limits are met with that much room."
    )
    lines.append("")
    lines.append(
        "| Architecture | Class verdict | Binding band | Measured rel. atten. "
        "| Class-1 limit | Margin cl.1 | Margin cl.2 |"
    )
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for arch in _FILTER_ARCHS:
        fc = _filter_class(arch, 3)
        if fc.bind_side == "ceil":
            req = f"&le; {fc.bind_limit_db:+.2f} dB"
        else:
            req = f"&ge; {fc.bind_limit_db:+.2f} dB"
        lines.append(
            f"| {arch} | {_filter_verdict(arch, fc)} | {fc.bind_freq:.0f} Hz "
            f"| {_snap(fc.bind_measured_db, 5e-3):+.2f} dB | {req} "
            f"| {fc.min_margin1:+.3f} dB | {fc.min_margin2:+.3f} dB |"
        )
    lines.append("")
    lines.append(
        "Only **Butterworth** (the library default) and **Chebyshev-II** are "
        "class-compliant architectures. Chebyshev-I and elliptic trade the "
        "mask for passband ripple, and Bessel for a maximally-flat group delay "
        "(soft rolloff); they cannot satisfy the IEC 61260-1 Class 1/2 "
        "attenuation mask by construction, so they are labelled *By design* - "
        "this is expected, not a failure or regression."
    )
    lines.append("")
    lines.append(
        "**Frequency-weighting conformance** (A/C: IEC 61672-1 Table 3; "
        "G: ISO 7196 A.3). The *max deviation from nominal* is informational "
        "(it falls at a frequency extreme where the tolerance is widest and "
        "asymmetric); compliance is judged at the *binding* frequency - the "
        "one with the least headroom - where the deviation, the applicable "
        "tolerance band and the headroom are shown together."
    )
    lines.append("")
    lines.append(
        "| Curve | fs | Max dev. from nominal (info) | Binding freq "
        "| Deviation there | Tolerance band | Headroom |"
    )
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for curve, fs in [("A", 48000), ("A", 96000), ("C", 48000), ("G", 48000)]:
        wd = _weighting_deviation(curve, fs)
        band = f"[{wd.bind_lower:+.2f}, {wd.bind_upper:+.2f}] dB"
        lines.append(
            f"| {curve} | {fs // 1000} kHz "
            f"| {wd.worst_dev:+.3f} dB @ {wd.worst_freq:.0f} Hz "
            f"| {wd.bind_freq:.0f} Hz | {_snap(wd.bind_dev):+.3f} dB | {band} "
            f"| {wd.min_headroom:+.3f} dB |"
        )
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def render_markdown() -> tuple[str, int, int]:
    """Render the full conformance report. Returns (markdown, passed, total)."""
    results = [(chk, chk.run()) for chk in CHECKS]
    passed = sum(1 for _, o in results if o.passed)
    total = len(results)

    filters_ok = all(
        o.passed for c, o in results if c.domain == "Filters & weightings"
    )
    headline_emoji = "&#9989;" if passed == total else "&#10060;"

    out: list[str] = []
    out.append("## Numerical conformance report")
    out.append("")
    summary = (
        f"**{passed}/{total} conformance checks pass** across "
        f"{len(_domains())} domains and {len({c.standard.split(':')[0].split(' Annex')[0] for c, _ in results})} standards"
    )
    if filters_ok:
        summary += " - filters class 1 - weightings within IEC 61672-1 class 1"
    out.append(f"{headline_emoji} {summary}.")
    out.append("")
    out.append(
        "<sub>Each row pins a standard clause to its expected normative value "
        "and the value the library computes. Every section below is "
        "collapsible and stays collapsed while all of its rows pass; a "
        "section with any failing row opens automatically.</sub>"
    )
    out.append("")
    out.append(_numerical_validation_section(filters_ok))
    out.append("")

    for domain in _domains():
        rows = [(chk, o) for chk, o in results if chk.domain == domain]
        passed_d = sum(1 for _, o in rows if o.passed)
        total_d = len(rows)
        pct = 100.0 * passed_d / total_d if total_d else 100.0
        emoji = "&#9989;" if passed_d == total_d else "&#10060;"
        # Each domain is a collapsible group labelled with its compliance
        # percentage (100 % = every row passes). Groups with any failing row are
        # opened by default so regressions stay visible.
        opened = " open" if passed_d != total_d else ""
        domain_html = domain.replace("&", "&amp;")
        out.append(f"<details{opened}>")
        out.append(
            f"<summary>{emoji} <b>{domain_html}</b>: {pct:.0f}% "
            f"({passed_d}/{total_d})</summary>"
        )
        out.append("")
        out.append("| Standard | Quantity | Expected (norm) | Computed | &#916; | Status |")
        out.append("|:---|:---|:---|:---|:---|:---:|")
        for chk, outcome in rows:
            out.append(
                f"| {chk.standard} | {chk.quantity} | {outcome.expected} "
                f"| {outcome.computed} | {outcome.delta} | {_status(outcome.passed)} |"
            )
        out.append("")
        out.append("</details>")
        out.append("")

    return "\n".join(out), passed, total


# Header prepended to the committed docs/CONFORMANCE.md (via `--file-header`,
# used by `make conformance`). Emitted only for the committed file, not for the
# CI PR-comment body, so the PR comment stays header-free.
_DOC_HEADER = """<!--
  AUTO-GENERATED FILE - DO NOT EDIT BY HAND.
  Regenerate with `make conformance` (runs scripts/conformance_report.py).
  CI regenerates it on every pull request and fails the build if it drifts.
-->

> **Auto-generated conformance report - do not hand-edit.** Produced by
> `make conformance` from the library's own computations checked against the
> referenced standards. CI regenerates it on every pull request and fails if it
> is out of date, so edit the checks in `scripts/conformance_report.py`, not this
> file. Each row pins a standard and clause to its expected normative value and
> the value the library computes. Full standards list and methodology:
> [Theory](https://github.com/jmrplens/phonometry/blob/main/docs/theory.md) -
> [Why phonometry](https://github.com/jmrplens/phonometry/blob/main/docs/why-phonometry.md).

"""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    markdown, passed, total = render_markdown()
    # The root artifact feeds the CI PR comment; keep it header-free.
    (_ROOT / "conformance_report.md").write_text(markdown + "\n")
    output = _DOC_HEADER + markdown if "--file-header" in args else markdown
    print(output)
    print(f"\n[conformance] {passed}/{total} checks passed", file=sys.stderr)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
