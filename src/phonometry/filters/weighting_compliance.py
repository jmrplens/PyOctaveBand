#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
IEC 61672-1:2013 frequency-weighting class verification.

A/C/Z frequency-weighting acceptance limits transcribed from
BS EN 61672-1:2013, **Table 3** (standard page 22): the design-goal responses
and the class 1 and class 2 upper/lower limits at the 34 nominal frequencies
from 10 Hz to 20 kHz. A lower limit of ``-inf`` means only the upper limit
applies (subclause 5.5.6 checks measured deviations at the nominal frequencies).

The historical **B weighting** is verified against ANSI S1.4-1983: design
goals from the B column of **Table IV** (whose A and C columns equal IEC
61672-1:2013 Table 3 digit for digit) and tolerance limits from **Table V**,
whose instrument Types 1 and 2 fill the class 1 / class 2 verdict slots (the
stricter laboratory Type 0 mask is exercised by the CI conformance report).
The **AU weighting** is verified against IEC 61012:1990: design goals are the
sum of the nominal A response and the **Table 1** nominal U response (with the
subclause 2.2 explicit AU values at 25/31.5/40 kHz), checked against the
Table 1 tolerances for the filter as a separate unit, the tighter of the two
tolerance readings the standard offers. IEC 61012 publishes a single
tolerance set, so both verdict slots carry the same margin for AU.

One subject: the weighting network a sound level meter applies to the whole
signal, whose acceptance limits qualify the deviation of its measured relative
response from a design goal at the nominal frequencies. The band-filter class
limits of IEC 61260-1, which qualify a relative attenuation against a mask
around each mid-band frequency, live in :mod:`phonometry.filters.compliance`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from .weighting import WeightingFilter

__all__ = [
    "verify_weighting_class",
    "weighting_class_limits",
]

_INF = float("inf")

# BS EN 61672-1:2013 Table 3 (standard page 22): design-goal frequency
# weightings and class 1 / class 2 acceptance limits at the 34 nominal
# frequencies. Columns: (nominal Hz, A dB, C dB, class1 upper, class1 lower,
# class2 upper, class2 lower); Z is 0.0 dB at every frequency. A lower limit
# of -inf means only the upper limit applies.
_WEIGHTING_TABLE3: list[tuple[float, float, float, float, float, float, float]] = [
    (10.0, -70.4, -14.3, 3.0, -_INF, 5.0, -_INF),
    (12.5, -63.4, -11.2, 2.5, -_INF, 5.0, -_INF),
    (16.0, -56.7, -8.5, 2.0, -4.0, 5.0, -_INF),
    (20.0, -50.5, -6.2, 2.0, -2.0, 3.0, -3.0),
    (25.0, -44.7, -4.4, 2.0, -1.5, 3.0, -3.0),
    (31.5, -39.4, -3.0, 1.5, -1.5, 3.0, -3.0),
    (40.0, -34.6, -2.0, 1.0, -1.0, 2.0, -2.0),
    (50.0, -30.2, -1.3, 1.0, -1.0, 2.0, -2.0),
    (63.0, -26.2, -0.8, 1.0, -1.0, 2.0, -2.0),
    (80.0, -22.5, -0.5, 1.0, -1.0, 2.0, -2.0),
    (100.0, -19.1, -0.3, 1.0, -1.0, 1.5, -1.5),
    (125.0, -16.1, -0.2, 1.0, -1.0, 1.5, -1.5),
    (160.0, -13.4, -0.1, 1.0, -1.0, 1.5, -1.5),
    (200.0, -10.9, 0.0, 1.0, -1.0, 1.5, -1.5),
    (250.0, -8.6, 0.0, 1.0, -1.0, 1.5, -1.5),
    (315.0, -6.6, 0.0, 1.0, -1.0, 1.5, -1.5),
    (400.0, -4.8, 0.0, 1.0, -1.0, 1.5, -1.5),
    (500.0, -3.2, 0.0, 1.0, -1.0, 1.5, -1.5),
    (630.0, -1.9, 0.0, 1.0, -1.0, 1.5, -1.5),
    (800.0, -0.8, 0.0, 1.0, -1.0, 1.5, -1.5),
    (1000.0, 0.0, 0.0, 0.7, -0.7, 1.0, -1.0),
    (1250.0, 0.6, 0.0, 1.0, -1.0, 1.5, -1.5),
    (1600.0, 1.0, -0.1, 1.0, -1.0, 2.0, -2.0),
    (2000.0, 1.2, -0.2, 1.0, -1.0, 2.0, -2.0),
    (2500.0, 1.3, -0.3, 1.0, -1.0, 2.5, -2.5),
    (3150.0, 1.2, -0.5, 1.0, -1.0, 2.5, -2.5),
    (4000.0, 1.0, -0.8, 1.0, -1.0, 3.0, -3.0),
    (5000.0, 0.5, -1.3, 1.5, -1.5, 3.5, -3.5),
    (6300.0, -0.1, -2.0, 1.5, -2.0, 4.5, -4.5),
    (8000.0, -1.1, -3.0, 1.5, -2.5, 5.0, -5.0),
    (10000.0, -2.5, -4.4, 2.0, -3.0, 5.0, -_INF),
    (12500.0, -4.3, -6.2, 2.0, -5.0, 5.0, -_INF),
    (16000.0, -6.6, -8.5, 2.5, -16.0, 5.0, -_INF),
    (20000.0, -9.3, -11.2, 3.0, -_INF, 5.0, -_INF),
]

_WEIGHTING_COL = {"A": 1, "C": 2, "Z": None}

# ---------------------------------------------------------------------------
# ANSI S1.4-1983 - historical B weighting (dropped when IEC 61672-1 replaced
# the older sound-level-meter standards).
# ---------------------------------------------------------------------------

# ANSI S1.4-1983 Table IV (standard page 6): random-incidence relative
# response level of the B weighting at the 34 nominal frequencies. The A and
# C columns of Table IV equal IEC 61672-1:2013 Table 3 digit for digit, so
# only the B column is transcribed here. Row = (nominal Hz, B dB).
_ANSI_S14_TABLE4_B: list[tuple[float, float]] = [
    (10.0, -38.2), (12.5, -33.2), (16.0, -28.5), (20.0, -24.2),
    (25.0, -20.4), (31.5, -17.1), (40.0, -14.2), (50.0, -11.6),
    (63.0, -9.3), (80.0, -7.4), (100.0, -5.6), (125.0, -4.2),
    (160.0, -3.0), (200.0, -2.0), (250.0, -1.3), (315.0, -0.8),
    (400.0, -0.5), (500.0, -0.3), (630.0, -0.1), (800.0, 0.0),
    (1000.0, 0.0), (1250.0, 0.0), (1600.0, 0.0), (2000.0, -0.1),
    (2500.0, -0.2), (3150.0, -0.4), (4000.0, -0.7), (5000.0, -1.2),
    (6300.0, -1.9), (8000.0, -2.9), (10000.0, -4.3), (12500.0, -6.1),
    (16000.0, -8.4), (20000.0, -11.1),
]

# ANSI S1.4-1983 Table V (standard page 6): tolerance limits on relative
# response levels for Type 1 and Type 2 instruments; the verifier maps them
# to the class 1 / class 2 verdict slots. (The stricter laboratory Type 0
# column lives in tests/reference_data/ and is pinned by the CI
# conformance report.) Row = (nominal Hz, type1 upper, type1 lower,
# type2 upper, type2 lower); a -inf lower limit means upper-only.
# Transcription note, 20 Hz Type 2: the standard prints a bare "+3" there,
# read here as +3/upper-only (matching the surrounding upper-only rows); a
# plausible alternative reading is +/-3. The B-weighting response at 20 Hz
# sits only 0.05 dB below nominal, so either reading yields the same
# verdict.
_ANSI_S14_TABLE5_12: list[tuple[float, float, float, float, float]] = [
    (10.0, 4.0, -4.0, 5.0, -_INF),
    (12.5, 3.5, -3.5, 5.0, -_INF),
    (16.0, 3.0, -3.0, 5.0, -_INF),
    (20.0, 2.5, -2.5, 3.0, -_INF),
    (25.0, 2.0, -2.0, 3.0, -3.0),
    (31.5, 1.5, -1.5, 3.0, -3.0),
    (40.0, 1.5, -1.5, 2.0, -2.0),
    (50.0, 1.0, -1.0, 2.0, -2.0),
    (63.0, 1.0, -1.0, 2.0, -2.0),
    (80.0, 1.0, -1.0, 2.0, -2.0),
    (100.0, 1.0, -1.0, 1.5, -1.5),
    (125.0, 1.0, -1.0, 1.5, -1.5),
    (160.0, 1.0, -1.0, 1.5, -1.5),
    (200.0, 1.0, -1.0, 1.5, -1.5),
    (250.0, 1.0, -1.0, 1.5, -1.5),
    (315.0, 1.0, -1.0, 1.5, -1.5),
    (400.0, 1.0, -1.0, 1.5, -1.5),
    (500.0, 1.0, -1.0, 1.5, -1.5),
    (630.0, 1.0, -1.0, 1.5, -1.5),
    (800.0, 1.0, -1.0, 1.5, -1.5),
    (1000.0, 1.0, -1.0, 1.5, -1.5),
    (1250.0, 1.0, -1.0, 1.5, -1.5),
    (1600.0, 1.0, -1.0, 2.0, -2.0),
    (2000.0, 1.0, -1.0, 2.0, -2.0),
    (2500.0, 1.0, -1.0, 2.5, -2.5),
    (3150.0, 1.0, -1.0, 2.5, -2.5),
    (4000.0, 1.0, -1.0, 3.0, -3.0),
    (5000.0, 1.5, -1.5, 3.5, -3.5),
    (6300.0, 1.5, -2.0, 4.5, -4.5),
    (8000.0, 1.5, -3.0, 5.0, -5.0),
    (10000.0, 2.0, -4.0, 5.0, -_INF),
    (12500.0, 3.0, -6.0, 5.0, -_INF),
    (16000.0, 3.0, -_INF, 5.0, -_INF),
    (20000.0, 3.0, -_INF, 5.0, -_INF),
]

# ANSI S1.4-1983 Appendix C: the B weighting is the C weighting with one
# extra zero at the origin and one extra real pole at f5 (Formula C2).
_F5 = 158.48932

# ---------------------------------------------------------------------------
# IEC 61012:1990 - AU weighting (audible sound in the presence of ultrasound)
# ---------------------------------------------------------------------------

# IEC 61012:1990 Table 1 (standard page 11): nominal relative response and
# tolerances of the U weighting as a separate filter unit, at the 37 nominal
# frequencies from 10 Hz to 40 kHz. The tolerance is zero at the 1 kHz
# reference frequency (Table 1 note; IEC 651 subclause 3.7) and the -inf
# lower limit at 40 kHz means upper-only. Row = (nominal Hz, U dB, upper
# tolerance, lower tolerance).
_IEC61012_TABLE1: list[tuple[float, float, float, float]] = [
    (10.0, 0.0, 3.0, -3.0), (12.5, 0.0, 3.0, -3.0), (16.0, 0.0, 3.0, -3.0),
    (20.0, 0.0, 3.0, -3.0), (25.0, 0.0, 2.0, -2.0), (31.5, 0.0, 1.0, -1.0),
    (40.0, 0.0, 1.0, -1.0), (50.0, 0.0, 1.0, -1.0), (63.0, 0.0, 1.0, -1.0),
    (80.0, 0.0, 1.0, -1.0), (100.0, 0.0, 1.0, -1.0), (125.0, 0.0, 1.0, -1.0),
    (160.0, 0.0, 1.0, -1.0), (200.0, 0.0, 1.0, -1.0), (250.0, 0.0, 1.0, -1.0),
    (315.0, 0.0, 1.0, -1.0), (400.0, 0.0, 1.0, -1.0), (500.0, 0.0, 1.0, -1.0),
    (630.0, 0.0, 1.0, -1.0), (800.0, 0.0, 1.0, -1.0), (1000.0, 0.0, 0.0, 0.0),
    (1250.0, 0.0, 1.0, -1.0), (1600.0, 0.0, 1.0, -1.0), (2000.0, 0.0, 1.0, -1.0),
    (2500.0, 0.0, 1.0, -1.0), (3150.0, 0.0, 1.0, -1.0), (4000.0, 0.0, 1.0, -1.0),
    (5000.0, 0.0, 1.0, -1.0), (6300.0, 0.0, 1.0, -1.0), (8000.0, 0.0, 1.0, -1.0),
    (10000.0, 0.0, 1.0, -1.0), (12500.0, -2.8, 2.0, -2.0),
    (16000.0, -13.0, 3.0, -3.0), (20000.0, -25.3, 3.0, -6.0),
    (25000.0, -37.6, 3.0, -6.0), (31500.0, -49.7, 3.0, -10.0),
    (40000.0, -61.8, 3.0, -_INF),
]

# IEC 61012:1990 subclause 2.2: explicit nominal AU values at the three
# frequencies above the last IEC 651 A-weighting row (20 kHz), prescribed
# directly because A + U cannot be summed from tabulated columns there.
_IEC61012_AU_HF = {25000.0: -50.0, 31500.0: -65.4, 40000.0: -81.1}

# IEC 61012:1990 Table 2: pole locations of the U weighting (Hz).
_U_POLES_HZ = np.array(
    [
        -12200.0,
        -12200.0,
        -7850.0 + 8800.0j,
        -7850.0 - 8800.0j,
        -2900.0 + 12150.0j,
        -2900.0 - 12150.0j,
    ]
)

# IEC 61672-1:2013 Annex E pole frequencies of the analytic A/C design goals
# (E.4.1-E.4.8); identical to the constants the WeightingFilter design uses.
_F1 = 20.598997
_F2 = 107.65265
_F3 = 737.86223
_F4 = 12194.217


def _exact_base10(frequencies: np.ndarray) -> np.ndarray:
    r"""Exact base-10 frequencies :math:`1000 \cdot 10^{n/10}`.

    IEC 61672-1:2013 Table 3 NOTE: the tabulated weightings are computed
    at the exact frequencies
    :math:`f = 1000 \cdot 10^{0.1 (n - 30)}`, not at the nominal
    labels (e.g. 15 848.9 Hz behind "16 kHz").
    """
    return np.asarray(
        10.0 ** (np.round(10.0 * np.log10(frequencies)) / 10.0), dtype=np.float64
    )


def _analytic_weighting_db(curve: str, frequencies: np.ndarray) -> np.ndarray:
    """Analytic design-goal weighting, re 1 kHz.

    For A/C(/Z) this evaluates the exact transfer-function magnitudes of IEC
    61672-1:2013 Annex E (E.4.1/E.4.2) at ``frequencies`` and normalizes to
    the 1 kHz value, reproducing every Table 3 design goal after 0.1 dB
    rounding. ``B`` adds the ANSI S1.4-1983 Appendix C Formula (C2) factor to
    the C response, and ``AU`` cascades the A response with the U low-pass
    built from the IEC 61012:1990 Table 2 poles (which reproduces every
    Table 1 nominal value within 0.05 dB).
    """
    f = np.asarray(frequencies, dtype=np.float64)
    if curve == "Z":
        return np.zeros_like(f)

    def _c_gain(x: np.ndarray) -> np.ndarray:
        x2 = x**2
        return np.asarray(
            (_F4**2 * x2) / ((x2 + _F1**2) * (x2 + _F4**2)), dtype=np.float64
        )

    def _a_gain(x: np.ndarray) -> np.ndarray:
        x2 = x**2
        return np.asarray(
            _c_gain(x) * x2 / np.sqrt((x2 + _F2**2) * (x2 + _F3**2)),
            dtype=np.float64,
        )

    def _b_gain(x: np.ndarray) -> np.ndarray:
        # ANSI S1.4-1983 Formula (C2): W_B = 10 lg(K2 f^2/(f^2 + f5^2)) + W_C
        # (the constant K2 cancels in the 1 kHz normalization below).
        return np.asarray(
            _c_gain(x) * x / np.sqrt(x**2 + _F5**2), dtype=np.float64
        )

    def _u_gain(x: np.ndarray) -> np.ndarray:
        # Magnitude of the all-pole U weighting from the Table 2 pole
        # coordinates in Hz (the 2*pi scale cancels in the normalization).
        den = np.prod(1j * x[:, None] - _U_POLES_HZ[None, :], axis=1)
        return np.asarray(1.0 / np.abs(den), dtype=np.float64)

    def _au_gain(x: np.ndarray) -> np.ndarray:
        return np.asarray(_a_gain(x) * _u_gain(x), dtype=np.float64)

    gain = {"A": _a_gain, "B": _b_gain, "C": _c_gain, "AU": _au_gain}[curve]
    ref = gain(np.asarray([1000.0]))[0]
    return np.asarray(20.0 * np.log10(gain(f) / ref), dtype=np.float64)


def weighting_class_limits(
    weighting_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IEC 61672-1:2013 Table 3 acceptance limits for a performance class.

    The limits apply to every IEC 61672-1 weighting (A, C, Z); they qualify
    the deviation of the measured relative response from the design goal at
    each nominal frequency, not the response itself. (The B and AU masks that
    ``verify_weighting_class`` uses come from ANSI S1.4-1983 Table V and
    IEC 61012:1990 Table 1 instead and are not returned here.)

    :param weighting_class: 1 or 2 (IEC 61672-1:2013 performance class).
    :return: Tuple ``(frequencies, lower, upper)`` of the 34 nominal
        frequencies (Hz) and the lower/upper deviation limits in dB. A lower
        limit of ``-inf`` means only the upper limit applies.
    """
    if weighting_class not in (1, 2):
        raise ValueError("weighting_class must be 1 or 2.")
    up_col, lo_col = (3, 4) if weighting_class == 1 else (5, 6)
    freqs = np.array([row[0] for row in _WEIGHTING_TABLE3], dtype=np.float64)
    upper = np.array([row[up_col] for row in _WEIGHTING_TABLE3], dtype=np.float64)
    lower = np.array([row[lo_col] for row in _WEIGHTING_TABLE3], dtype=np.float64)
    return freqs, lower, upper


def _curve_design_and_limits(
    curve: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Nominal frequencies, design goals and the two acceptance masks.

    A/C/Z read IEC 61672-1:2013 Table 3 (classes 1 and 2). B reads ANSI
    S1.4-1983 Table IV (design goals) and Table V (Types 1 and 2 fill the
    two mask slots). AU reads IEC 61012:1990 Table 1: the design goal is
    nominal A + nominal U (with the subclause 2.2 explicit values above
    20 kHz) and the single Table 1 tolerance set fills both mask slots.
    """
    if curve in _WEIGHTING_COL:
        col = _WEIGHTING_COL[curve]
        nominal = np.array([row[0] for row in _WEIGHTING_TABLE3], dtype=np.float64)
        design = (
            np.zeros_like(nominal)
            if col is None
            else np.array([row[col] for row in _WEIGHTING_TABLE3], dtype=np.float64)
        )
        _, lower1, upper1 = weighting_class_limits(1)
        _, lower2, upper2 = weighting_class_limits(2)
    elif curve == "B":
        nominal = np.array([row[0] for row in _ANSI_S14_TABLE4_B], dtype=np.float64)
        design = np.array([row[1] for row in _ANSI_S14_TABLE4_B], dtype=np.float64)
        upper1 = np.array([row[1] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        lower1 = np.array([row[2] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        upper2 = np.array([row[3] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        lower2 = np.array([row[4] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
    else:  # AU
        a_design = {row[0]: row[1] for row in _WEIGHTING_TABLE3}
        nominal = np.array([row[0] for row in _IEC61012_TABLE1], dtype=np.float64)
        design = np.array(
            [
                _IEC61012_AU_HF.get(row[0], a_design.get(row[0], 0.0) + row[1])
                for row in _IEC61012_TABLE1
            ],
            dtype=np.float64,
        )
        upper1 = np.array([row[2] for row in _IEC61012_TABLE1], dtype=np.float64)
        lower1 = np.array([row[3] for row in _IEC61012_TABLE1], dtype=np.float64)
        upper2, lower2 = upper1, lower1
    return nominal, design, (lower1, upper1), (lower2, upper2)


def _weighting_response_db(wf: WeightingFilter, frequencies: np.ndarray) -> np.ndarray:
    """Relative steady-state response of *wf* in dB, normalized to 1 kHz."""
    if wf.curve == "Z" or wf.sos.size == 0:
        return np.zeros_like(frequencies)
    # The SOS are designed at the (possibly oversampled) processing rate.
    fs_proc = wf.fs * wf._oversample
    worn = np.concatenate([frequencies, [1000.0]])
    _, h = signal.sosfreqz(wf.sos, worN=worn, fs=fs_proc)
    gain_db = 20.0 * np.log10(np.abs(h) + np.finfo(float).eps)
    return np.asarray(gain_db[:-1] - gain_db[-1], dtype=np.float64)  # relative to 1 kHz


def _band_class(margin1: float, margin2: float) -> int | None:
    """Narrowest class the band still meets, or ``None`` if it meets neither.

    :param margin1: Distance to the nearer class 1 limit, in decibels.
    :param margin2: Distance to the nearer class 2 limit, in decibels.
    :return: ``1``, ``2`` or ``None``.
    """
    if margin1 >= 0:
        return 1
    if margin2 >= 0:
        return 2
    return None


def _weighting_band_verdicts(
    freqs_nom: np.ndarray,
    deviation: np.ndarray,
    limits1: tuple[np.ndarray, np.ndarray],
    limits2: tuple[np.ndarray, np.ndarray],
) -> list[dict[str, Any]]:
    """Per-band class verdicts against the Table 3 acceptance limits.

    Margin = distance to the nearer limit; a -inf lower limit makes that side
    non-binding (its term is +inf), i.e. an upper-only limit.
    """
    lower1, upper1 = limits1
    lower2, upper2 = limits2
    bands: list[dict[str, Any]] = []
    for i, fm in enumerate(freqs_nom):
        m1 = min(upper1[i] - deviation[i], deviation[i] - lower1[i])
        m2 = min(upper2[i] - deviation[i], deviation[i] - lower2[i])
        band_class = _band_class(m1, m2)
        bands.append(
            {
                "freq": float(fm),
                "class": band_class,
                "deviation_db": float(deviation[i]),
                "margin_class1_db": float(m1),
                "margin_class2_db": float(m2),
            }
        )
    return bands


def _between_nominals_sweep(
    wf: WeightingFilter,
    freqs_exact: np.ndarray,
    limits1: tuple[np.ndarray, np.ndarray],
    limits2: tuple[np.ndarray, np.ndarray],
    sweep_points: int,
) -> dict[str, float]:
    """Subclause 5.5.7 sweep between adjacent exact nominal frequencies.

    The acceptance limits between two adjacent nominal frequencies are the
    larger of the two adjacent Table 3 limits; the design goal there is the
    analytic Annex E response.
    """
    lower1, upper1 = limits1
    lower2, upper2 = limits2
    grid = np.geomspace(freqs_exact[0], freqs_exact[-1], sweep_points)
    sweep_dev = _weighting_response_db(wf, grid) - _analytic_weighting_db(
        wf.curve, grid
    )
    seg = np.clip(np.searchsorted(freqs_exact, grid, side="right") - 1, 0,
                  freqs_exact.size - 2)
    up1 = np.maximum(upper1[seg], upper1[seg + 1])
    lo1 = np.minimum(lower1[seg], lower1[seg + 1])
    up2 = np.maximum(upper2[seg], upper2[seg + 1])
    lo2 = np.minimum(lower2[seg], lower2[seg + 1])
    sweep_m1 = np.minimum(up1 - sweep_dev, sweep_dev - lo1)
    sweep_m2 = np.minimum(up2 - sweep_dev, sweep_dev - lo2)
    worst = int(np.argmin(sweep_m1))
    return {
        "worst_freq": float(grid[worst]),
        "margin_class1_db": float(np.min(sweep_m1)),
        "margin_class2_db": float(np.min(sweep_m2)),
    }


def verify_weighting_class(
    wf: WeightingFilter, *, sweep_points: int = 4096
) -> dict[str, Any]:
    r"""
    Verify a frequency-weighting filter against its standard's tolerances.

    ``A``/``C``/``Z`` are checked against IEC 61672-1:2013 Table 3 (classes 1
    and 2). The historical ``B`` weighting is checked against ANSI S1.4-1983:
    Table IV design goals with the Table V tolerance limits, whose instrument
    Types 1 and 2 fill the class 1 / class 2 verdict slots (an
    ``overall_class`` of 1 then reads "ANSI S1.4-1983 Type 1"). ``AU`` is
    checked against IEC 61012:1990: design goals are nominal A + nominal U
    (Table 1, plus the subclause 2.2 explicit AU values at 25/31.5/40 kHz)
    with the single Table 1 tolerance set for the filter as a separate unit,
    so both class slots carry the same margin and ``overall_class`` is 1
    (complies) or ``None``. ``G`` is not supported here (ISO 7196 defines one
    +/-1 dB instrumentation tolerance, no class structure; the CI conformance
    report pins it), nor is ``D`` (the tolerance tables of the withdrawn
    IEC 537 did not survive it; the conformance report pins the D response
    against its published transfer function and tabulated curve).

    The filter's relative response (normalized to its 1 kHz gain) is evaluated
    at the *exact* base-10 frequency behind each nominal label below
    the Nyquist frequency (IEC 61672-1 Table 3 NOTE: the design goals are
    computed at :math:`f = 1000 \cdot 10^{0.1 (n - 30)}`, e.g.
    15 848.9 Hz for
    "16 kHz"; IEC 61672-3:2013 subclause 13.3 tests the deviation at the same
    exact frequencies, and IEC 61012 Table 1 lists the same exact
    frequencies). The deviation from the design-goal weighting is checked
    against the two acceptance masks.

    A dense logarithmic sweep between the checked frequencies additionally
    enforces IEC 61672-1 subclause 5.5.7: at any frequency between two
    adjacent nominal frequencies, the deviation of the response from the
    analytic design goal (Annex E for A/C/Z, the ANSI S1.4-1983 Appendix C
    formulas for B, the A response cascaded with the IEC 61012 Table 2 poles
    for AU) must stay within the *larger* of the two adjacent limits. Without
    it a resonance or notch between the nominal frequencies would go
    unnoticed (for B and AU the sweep is applied as the analogous engineering
    check). Both the per-frequency verdicts and the sweep must pass for
    ``overall_class``. The sweep samples ``sweep_points`` grid frequencies; a
    violation narrower than the grid spacing could in principle fall between
    samples, so raise ``sweep_points`` for higher-Q suspects (the verdict
    attests the sampled grid, not a continuous proof).

    The response is taken from the designed second-order sections (evaluated
    with ``sosfreqz`` at their design rate), so it is exact and deterministic;
    it does not model the runtime resampling stages that ``high_accuracy``
    adds around them, whose anti-alias response is flat across the audio band
    checked here. The ``Z`` weighting is a flat bypass and always complies.

    When rows that carry a *finite lower* acceptance limit fall at or
    above the Nyquist frequency (e.g. the 8-16 kHz class 1 rows of a 16 kHz
    sampled system, or the 25-40 kHz AU rows of a 48 kHz one), they cannot be
    checked and ``range_limited`` is ``True``: the returned class then
    attests conformance over the checked frequencies only, not conformance
    over the standard's full frequency range.

    :param wf: The weighting filter to verify (``A``, ``B``, ``C``, ``AU``
        or ``Z``).
    :param sweep_points: Number of points of the 5.5.7 between-nominals sweep
        (>= 64).
    :return: Dict with ``overall_class`` (1, 2 or None), ``range_limited``
        (see above), ``bands``: a list of ``{"freq", "class", "deviation_db",
        "margin_class1_db", "margin_class2_db"}`` where ``freq`` is the
        nominal label, a positive margin means the limits are met with that
        much room, and ``between_nominals``: ``{"worst_freq",
        "margin_class1_db", "margin_class2_db"}`` for the sweep.
    """
    if wf.curve not in _WEIGHTING_COL and wf.curve not in ("B", "AU"):
        raise ValueError("Weighting curve must be 'A', 'B', 'C', 'AU' or 'Z'.")
    if sweep_points < 64:
        raise ValueError("'sweep_points' must be at least 64.")

    nyquist = wf.fs / 2.0
    nominal, design_all, limits1_all, limits2_all = _curve_design_and_limits(
        wf.curve
    )
    lower1_all, upper1_all = limits1_all
    lower2_all, upper2_all = limits2_all
    exact = _exact_base10(nominal)
    in_range = exact < nyquist

    # Any row beyond Nyquist cannot be demonstrated (its acceptance
    # limits and the adjacent between-nominal interval go unchecked), so the
    # verdict is then range-limited, not full-range conformance.
    dropped = ~in_range
    range_limited = bool(np.any(dropped))

    freqs_nom = nominal[in_range]
    freqs_exact = exact[in_range]
    design = design_all[in_range]
    response = _weighting_response_db(wf, freqs_exact)
    deviation = response - design

    lower1, upper1 = lower1_all[in_range], upper1_all[in_range]
    lower2, upper2 = lower2_all[in_range], upper2_all[in_range]

    bands = _weighting_band_verdicts(
        freqs_nom, deviation, (lower1, upper1), (lower2, upper2)
    )

    if not bands:
        return {
            "overall_class": None,
            "range_limited": range_limited,
            "bands": [],
            "between_nominals": None,
        }

    between = _between_nominals_sweep(
        wf, freqs_exact, (lower1, upper1), (lower2, upper2), sweep_points
    )

    classes = [band["class"] for band in bands]
    if all(c == 1 for c in classes) and between["margin_class1_db"] >= 0.0:
        overall: int | None = 1
    elif (
        all(c in (1, 2) for c in classes)
        and between["margin_class2_db"] >= 0.0
    ):
        overall = 2
    else:
        overall = None

    return {
        "overall_class": overall,
        "range_limited": range_limited,
        "bands": bands,
        "between_nominals": between,
    }
