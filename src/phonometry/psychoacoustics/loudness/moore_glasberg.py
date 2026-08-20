#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Stationary loudness per ISO 532-2:2017 (Moore-Glasberg method).

Clean-room implementation of the spectral loudness model of ISO 532-2:2017,
the third calculation method of the ISO 532 series (distinct from the Zwicker
method of ISO 532-1 and the Sottek model of ECMA-418-2).  The procedure of
Clause 7 (Figure 1) transforms a stationary sound spectrum into a loudness
value through five stages:

* the fixed outer-ear and middle-ear transfer functions that map the recorded
  spectrum to the spectrum at the cochlea (clauses 7.2, 7.3, Table 1);
* the level-dependent rounded-exponential (roex) auditory filter bank that
  turns that spectrum into an excitation pattern along the ERB-number scale
  (clause 7.4, Formulae 1-6), sampled at ERB-number ``i`` from 1.8 Cam to
  38.9 Cam in 0.1 Cam steps;
* the compressive transformation of excitation into specific loudness
  ``N'(i)`` in sone/Cam (clause 7.5, Formulae 7-9, Tables 2-4);
* the binaural inhibition model that combines the two ears (clause 8.1,
  Formulae 10-13); and
* the integration of specific loudness over the ERB-number scale to the total
  loudness ``N`` in sone, with the loudness level ``L_N`` in phon obtained by
  inverting the loudness/loudness-level relationship of Table 5 (clause 8.2).

The specific-loudness calibration constant ``C`` of Formula (7) is the value
tabulated by the standard (0.0617 sone/Cam); a 1 kHz tone at 40 dB SPL
presented binaurally in a free field yields 1.000 sone by definition of the
sone (clause 3.17), which this implementation reproduces without tuning.

The stationary method is spectrum-based.  :func:`loudness_moore_glasberg_from_spectrum`
takes the exact sinusoidal-component representation of clauses 5.2/5.4,
:func:`loudness_moore_glasberg_from_third_octave` takes the 29 one-third-octave
band levels of clause 5.5, and :func:`loudness_moore_glasberg` forms the
narrowband (FFT) line spectrum of a calibrated pressure signal and feeds it to
the exact sinusoidal-component method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ...io._resolve import apply_calibration, resolve_fs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from ...io._signal import Signal

from ..._internal.utils import _typesignal
from ..._internal.validation import require_1d_signal
from ..erb_scale import CAM_C, ERB_C1, ERB_C2, erb_bandwidth, frequency_from_cam

# ---------------------------------------------------------------------------
# Fixed transfer functions (clauses 7.2, 7.3, Table 1)
# ---------------------------------------------------------------------------
# Columns: frequency [Hz], free-field, diffuse-field and (scaled) middle-ear
# level differences in dB.  Interpolation is linear in dB versus linear
# frequency (clause 7.2.6); values are clamped outside [20, 20000] Hz.
_T1_FREQ = np.array(
    [
        20,
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        750,
        800,
        1000,
        1250,
        1500,
        1600,
        2000,
        2500,
        3000,
        3150,
        4000,
        5000,
        6000,
        6300,
        8000,
        9000,
        10000,
        11200,
        12500,
        14000,
        15000,
        16000,
        20000,
    ],
    dtype=np.float64,
)
_T1_FREE = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.3,
        0.5,
        0.9,
        1.4,
        1.6,
        1.7,
        2.5,
        2.7,
        2.6,
        2.6,
        3.2,
        5.2,
        6.6,
        12.0,
        16.8,
        15.3,
        15.2,
        14.2,
        10.7,
        7.1,
        6.4,
        1.8,
        -0.9,
        -1.6,
        1.9,
        4.9,
        2.0,
        -2.0,
        2.5,
        2.5,
    ],
    dtype=np.float64,
)
_T1_DIFFUSE = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.3,
        0.4,
        0.5,
        1.0,
        1.6,
        1.7,
        2.2,
        2.7,
        2.9,
        3.8,
        5.3,
        6.8,
        7.2,
        10.2,
        14.9,
        14.5,
        14.4,
        12.7,
        10.8,
        8.9,
        8.7,
        8.5,
        6.2,
        5.0,
        4.5,
        4.0,
        3.3,
        2.6,
        2.0,
        2.0,
    ],
    dtype=np.float64,
)
_T1_MIDDLE = np.array(
    [
        -39.6,
        -32.0,
        -25.85,
        -21.4,
        -18.5,
        -15.9,
        -14.1,
        -12.4,
        -11.0,
        -9.6,
        -8.3,
        -7.4,
        -6.2,
        -4.8,
        -3.8,
        -3.3,
        -2.9,
        -2.6,
        -2.6,
        -4.5,
        -5.4,
        -6.1,
        -8.5,
        -10.4,
        -7.3,
        -7.0,
        -6.6,
        -7.0,
        -9.2,
        -10.2,
        -12.2,
        -10.8,
        -10.1,
        -12.7,
        -15.0,
        -18.2,
        -23.8,
        -32.3,
        -45.5,
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# ERB-number scale and roex auditory filter (clause 7.4)
# ---------------------------------------------------------------------------
# Formulae (1) and (6) of clause 7.4 are the Glasberg and Moore (1990) ERB_N
# scale; they are shared with the public utilities of
# :mod:`phonometry.psychoacoustics.erb_scale`, which own the constants.
_ERB_C1 = ERB_C1
_ERB_C2 = ERB_C2
_CAM_C = CAM_C  # Formula (6): i = 21.366 * log10(C2 * fc + 1)
_ROEX_D = 0.35  # constant D of Formula (5)
_ROEX_G_MAX = 4.0  # Formula (4): drop upper-side components g > 4 (clause 7.4)
_I_MIN, _I_MAX, _I_STEP = 1.8, 38.9, 0.1  # ERB-number grid (clause 7.4)


def _erb_bandwidth(fc: np.ndarray) -> np.ndarray:
    """Equivalent rectangular bandwidth ERB_n in Hz (Formula 1)."""
    return np.asarray(erb_bandwidth(fc), dtype=np.float64)


def _fc_from_cam(i: np.ndarray) -> np.ndarray:
    """Centre frequency fc in Hz for ERB-number i (inverse of Formula 6)."""
    return np.asarray(frequency_from_cam(i), dtype=np.float64)


_I_GRID = np.round(
    np.arange(_I_MIN, _I_MAX + _I_STEP / 2.0, _I_STEP), 4
)  # 372 values, 1.8..38.9 Cam
_FC_GRID = _fc_from_cam(_I_GRID)
_ERB_GRID = _erb_bandwidth(_FC_GRID)
_PL51_1K = 4.0 * 1000.0 / _erb_bandwidth(np.array([1000.0]))[0]  # p_l(51 dB, 1 kHz)

# ---------------------------------------------------------------------------
# Reference threshold, cochlear gain, and specific-loudness parameters
# (clause 7.5, Tables 2-4)
# ---------------------------------------------------------------------------
# Table 2: centre frequency, excitation level at threshold [dB], 10 lg G [dB].
_T2_FREQ = np.array(
    [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 750, 800, 1000],
    dtype=np.float64,
)
_T2_LE_THR = np.array(
    [
        27.46,
        23.45,
        18.47,
        15.13,
        11.97,
        9.34,
        7.43,
        5.75,
        4.73,
        3.92,
        3.15,
        3.15,
        3.15,
        3.15,
        3.15,
    ],
    dtype=np.float64,
)
_T2_10LGG = np.array(
    [
        -24.31,
        -20.30,
        -15.32,
        -11.98,
        -8.82,
        -6.19,
        -4.28,
        -2.60,
        -1.58,
        -0.77,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
# Table 3: parameter alpha as a function of 10 lg G [dB].
_T3_10LGG = np.array([-25.0, -20.0, -15.0, -10.0, -5.0, 0.0], dtype=np.float64)
_T3_ALPHA = np.array(
    [0.26692, 0.25016, 0.23679, 0.22228, 0.21055, 0.20000], dtype=np.float64
)
# Table 4: parameter A as a function of 10 lg G [dB] (0.5 dB grid, -25..0).
_T4_10LGG = np.round(np.arange(-25.0, 0.0 + 0.25, 0.5), 3)
_T4_A = np.array(
    [
        7.784,
        7.667,
        7.551,
        7.435,
        7.318,
        7.210,
        7.103,
        6.996,
        6.889,
        6.782,
        6.675,
        6.596,
        6.517,
        6.438,
        6.360,
        6.281,
        6.202,
        6.124,
        6.047,
        5.975,
        5.902,
        5.823,
        5.744,
        5.665,
        5.587,
        5.510,
        5.437,
        5.364,
        5.291,
        5.218,
        5.145,
        5.086,
        5.027,
        4.972,
        4.918,
        4.863,
        4.808,
        4.754,
        4.699,
        4.644,
        4.590,
        4.542,
        4.496,
        4.451,
        4.405,
        4.359,
        4.314,
        4.268,
        4.222,
        4.177,
        4.131,
    ],
    dtype=np.float64,
)

_C_SONE = 0.0617  # calibration constant C of Formula (7) [sone/Cam]
_E_HIGH_LEVEL = 1e10  # E/E0 bound of the high-level Formula (9) (clause 7.5)
_HF_MIN_FC_HZ = 500.0  # _HF parameters apply for fc >= this (clauses 7.5.2/7.5.4)
_ALPHA_HF = 0.2  # exponent alpha for fc >= 500 Hz (clause 7.5.4)
_A_HF = 4.13  # parameter A for fc >= 500 Hz = 2 * E_THRQ/E0 (clause 7.5.4)
_E_THRQ_HF = 2.065  # E_THRQ/E0 for fc >= 500 Hz (clause 7.5.2)

# ---------------------------------------------------------------------------
# Binaural inhibition (clause 8.1, Formulae 10-13)
# ---------------------------------------------------------------------------
_INH_B = 0.08  # spread parameter B of Formulae 10/11
_INH_THETA = 1.5978  # exponent theta of Formulae 12/13
# Formula (10)/(11): the smoothing weight is exp(-(B*Di)^2) with B = 0.08 and
# Di changed in steps of 0,1 over -18..+18 Cam (clause 8.1).  Taps are index
# offsets on the 0.1-Cam grid, so Di in Cam = tap * _I_STEP.
_INH_TAPS = np.arange(-180, 181)  # 0.1 Cam steps, +/-18 Cam (Formula 10)
_INH_KERNEL = np.exp(-((_INH_B * (_INH_TAPS * _I_STEP)) ** 2))
_EPS = 1e-13  # additive constant of clause 8.1 (avoids division by zero)

# ---------------------------------------------------------------------------
# Loudness-level relationship (clause 8.2, Table 5)
# ---------------------------------------------------------------------------
_T5_PHON = np.array(
    [
        0.0,
        2.2,
        4.0,
        5.0,
        7.5,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
        55.0,
        60.0,
        65.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
        95.0,
        100.0,
        105.0,
        110.0,
        115.0,
        120.0,
    ],
    dtype=np.float64,
)
_T5_SONE = np.array(
    [
        0.001,
        0.004,
        0.008,
        0.010,
        0.019,
        0.031,
        0.073,
        0.146,
        0.26,
        0.43,
        0.67,
        1.00,
        1.46,
        2.09,
        2.96,
        4.14,
        5.77,
        8.04,
        11.2,
        15.8,
        22.7,
        32.9,
        47.7,
        69.6,
        102.0,
        151.0,
        225.0,
        337.6,
    ],
    dtype=np.float64,
)
_LOG_T5_SONE = np.log10(_T5_SONE)

_FIELDS = ("free", "diffuse", "eardrum")
_PRESENTATIONS = ("binaural", "diotic", "monaural")

# Nominal one-third-octave centre frequencies, 25 Hz..16 kHz (clause 5.5).
_THIRD_OCTAVE_FREQ = np.array(
    [
        25.0,
        31.5,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
        800.0,
        1000.0,
        1250.0,
        1600.0,
        2000.0,
        2500.0,
        3150.0,
        4000.0,
        5000.0,
        6300.0,
        8000.0,
        10000.0,
        12500.0,
        16000.0,
    ],
    dtype=np.float64,
)
_N_THIRD_OCTAVE = _THIRD_OCTAVE_FREQ.size
_THIRD_OCTAVE_RATIO = 2.0 ** (1.0 / 6.0)  # band edge factor (half a third-octave)
_FINE_SPACING_MAX_CENTRE_HZ = 125.0  # 1 Hz spacing at/below, 10 Hz above (clause 5.5)


@dataclass(frozen=True)
class MooreGlasbergLoudness:
    """Result of an ISO 532-2:2017 Moore-Glasberg loudness calculation.

    ``loudness`` is the total loudness N in sone; ``loudness_level`` is the
    loudness level L_N in phon (obtained by inverting the loudness/loudness-
    level relationship of Table 5).  ``specific`` holds the specific loudness
    N'(i) in sone/Cam sampled at the ERB-number grid ``erb_number`` (Cam),
    i.e. i from 1.8 Cam to 38.9 Cam in 0.1 Cam steps (372 values), and
    ``centre_frequencies`` the corresponding auditory-filter centre
    frequencies in Hz.  For a binaural (diotic) result the specific-loudness
    pattern is that of a single ear before binaural inhibition; the total
    ``loudness`` already includes the binaural summation.  ``field`` and
    ``presentation`` echo the listening conditions.
    """

    loudness: float
    loudness_level: float
    specific: np.ndarray
    erb_number: np.ndarray
    centre_frequencies: np.ndarray
    field: str
    presentation: str

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the specific loudness N'(i) over the ERB-number scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.  See :mod:`phonometry._plot.psychoacoustics`.
        """
        from ..._i18n import check_language
        from ..._plot.psychoacoustics import plot_moore_glasberg_loudness

        return plot_moore_glasberg_loudness(
            self, ax=ax, language=check_language(language), **kwargs
        )


# ---------------------------------------------------------------------------
# Signal chain
# ---------------------------------------------------------------------------


def _cochlea_levels(freqs: np.ndarray, levels: np.ndarray, field: str) -> np.ndarray:
    """Component levels at the cochlea (clauses 7.2, 7.3).

    Adds the outer-ear transfer (free/diffuse field, or zero for an eardrum/
    earphone-flat presentation) and the middle-ear transfer of Table 1.
    """
    middle = np.interp(freqs, _T1_FREQ, _T1_MIDDLE)
    if field == "free":
        outer = np.interp(freqs, _T1_FREQ, _T1_FREE)
    elif field == "diffuse":
        outer = np.interp(freqs, _T1_FREQ, _T1_DIFFUSE)
    else:  # "eardrum": signal already specified at the tympanic membrane
        outer = np.zeros_like(freqs)
    return np.asarray(levels + outer + middle, dtype=np.float64)


def _source_levels(freqs: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Excitation-band level X_j of each component (clause 7.4).

    X_j is the power (dB re E0) of the input components lying within one
    ERB_n-wide band centred on the component's own frequency; it sets the
    sharpness of the level-dependent lower skirt of every auditory filter the
    component drives (Formula 5).  Computed with a cumulative sum over the
    frequency-sorted spectrum.

    Approximation: clause 7.4 forms X with a rounded-exponential (roex)
    weighting W(g, fc) of slope p = 4 fc/ERB_n over g = 0..1 (lower) and
    0..4 (upper) (Formulae 2-4), not the hard rectangular +/-ERB_n/2 window
    used here.  For an isolated tonal component the single in-band term is
    weighted W(0) = 1 in either form, so the two are identical; the deviation
    is immaterial for the broadband cases validated to ~1 % across Annex B.
    """
    order = np.argsort(freqs)
    f_sorted = freqs[order]
    cumulative = np.concatenate(([0.0], np.cumsum(powers[order])))
    half = _erb_bandwidth(freqs) / 2.0
    lo = np.searchsorted(f_sorted, freqs - half, side="left")
    hi = np.searchsorted(f_sorted, freqs + half, side="right")
    band_power = cumulative[hi] - cumulative[lo]
    return np.asarray(10.0 * np.log10(np.maximum(band_power, 1e-300)), dtype=np.float64)


def _excitation_pattern(freqs: np.ndarray, levels_cochlea: np.ndarray) -> np.ndarray:
    r"""Excitation ratio E/E0 at every ERB-number filter (clause 7.4).

    Each filter is a level-dependent rounded-exponential (roex) filter
    (Formulae 2-5): the upper skirt slope
    :math:`p_\mathrm{u} = 4 f_\mathrm{c} / \mathrm{ERB}_n` is level
    independent while the lower skirt slope ``p_l`` decreases (broadens) with
    the source level X_j of the component it filters, giving the
    level-dependent upward spread of excitation.
    """
    powers = np.power(10.0, levels_cochlea / 10.0)
    x_source = _source_levels(freqs, powers)  # per component, dB re E0
    excitation = np.zeros(_FC_GRID.size, dtype=np.float64)
    for k in range(_FC_GRID.size):
        fc = _FC_GRID[k]
        erb = _ERB_GRID[k]
        p_ref = 4.0 * fc / erb  # p_u and p_l(51 dB, fc)
        g = np.abs(freqs - fc) / fc
        upper = freqs > fc
        # Lower skirt slope from the per-component source level (Formula 5).
        p_lower = p_ref - _ROEX_D * (p_ref / _PL51_1K) * (x_source - 51.0)
        np.clip(p_lower, 0.1, 1e4, out=p_lower)
        p = np.where(upper, p_ref, p_lower)
        weight = (1.0 + p * g) * np.exp(-p * g)
        # Integration ranges of Formula (4): drop upper-side components g > 4.
        weight = np.where(upper & (g > _ROEX_G_MAX), 0.0, weight)
        excitation[k] = float(np.dot(powers, weight))
    return excitation


def _specific_loudness(excitation: np.ndarray) -> np.ndarray:
    """Specific loudness N'(i) in sone/Cam from E/E0 (clause 7.5).

    Applies the compressive transformation of Formulae (7)-(9) with the
    frequency-dependent cochlear gain G, exponent alpha and parameter A of
    Tables 2-4; below the reference threshold the near-threshold expression of
    Formula (8) is used and above E/E0 = 1e10 the high-level Formula (9).
    """
    high = _FC_GRID >= _HF_MIN_FC_HZ
    g_lin = np.ones(_FC_GRID.size)
    alpha = np.full(_FC_GRID.size, _ALPHA_HF)
    a_par = np.full(_FC_GRID.size, _A_HF)
    e_thr = np.full(_FC_GRID.size, _E_THRQ_HF)
    if np.any(~high):
        fc_low = _FC_GRID[~high]
        le_thr = np.interp(fc_low, _T2_FREQ, _T2_LE_THR)
        lg_g = np.interp(fc_low, _T2_FREQ, _T2_10LGG)
        e_thr[~high] = np.power(10.0, le_thr / 10.0)
        g_lin[~high] = np.power(10.0, lg_g / 10.0)
        alpha[~high] = np.interp(lg_g, _T3_10LGG, _T3_ALPHA)
        a_par[~high] = np.interp(lg_g, _T4_10LGG, _T4_A)

    n_spec = np.zeros(_FC_GRID.size, dtype=np.float64)
    active = excitation > 0.0
    e = excitation[active]
    ge_a = g_lin[active] * e + a_par[active]
    core = np.power(ge_a, alpha[active]) - np.power(a_par[active], alpha[active])
    values = _C_SONE * core
    # Near threshold (Formula 8): E/E0 < E_THRQ/E0.
    below = e < e_thr[active]
    factor = np.power(2.0 * e / (e + e_thr[active]), 1.5)
    values = np.where(below, _C_SONE * factor * core, values)
    # High level (Formula 9): E/E0 > 1e10.
    very_high = e > _E_HIGH_LEVEL
    values = np.where(very_high, _C_SONE * np.power(e / 1.0707, 0.2), values)
    n_spec[active] = values
    return n_spec


def _smooth(pattern: np.ndarray) -> np.ndarray:
    """Broadly tuned smoothing of a specific-loudness pattern (Formulae 10/11).

    Inert on every validated Annex B case: those are all diotic (identical ears
    give an inhibition ratio of 1) or monaural (silent contralateral ear gives
    unit inhibition), so the smoothed pattern never changes the result.
    """
    smoothed = np.zeros_like(pattern)
    n = pattern.size
    for offset, weight in zip(_INH_TAPS, _INH_KERNEL):
        src_lo = max(0, offset)
        src_hi = min(n, n + offset)
        dst_lo = max(0, -offset)
        dst_hi = min(n, n - offset)
        smoothed[dst_lo:dst_hi] += weight * pattern[src_lo:src_hi]
    return smoothed


def _binaural_loudness(n_left: np.ndarray, n_right: np.ndarray) -> tuple[float, float]:
    """Total loudness in sone with binaural inhibition (clause 8.1).

    Returns (loudness_left, loudness_right); their sum is the binaural
    loudness.  For a silent ear the inhibition factor is unity, so a monaural
    presentation reduces to the single-ear loudness.
    """
    left_smooth = _smooth(n_left) + _EPS
    right_smooth = _smooth(n_right) + _EPS
    # sech(ratio) -> 0 for a silent contralateral ear; clip to avoid cosh
    # overflow (the resulting inhibition factor is then unity, as intended).
    ratio_l = np.clip(right_smooth / left_smooth, 0.0, 700.0)
    ratio_r = np.clip(left_smooth / right_smooth, 0.0, 700.0)
    inh_left = 2.0 / (1.0 + (1.0 / np.cosh(ratio_l)) ** _INH_THETA)
    inh_right = 2.0 / (1.0 + (1.0 / np.cosh(ratio_r)) ** _INH_THETA)
    loud_left = float(np.sum(n_left / inh_left) * _I_STEP)
    loud_right = float(np.sum(n_right / inh_right) * _I_STEP)
    return loud_left, loud_right


def _loudness_level(loudness: float) -> float:
    """Loudness level L_N in phon from loudness N in sone (invert Table 5).

    NOTE: Table 5 ends at 120 phon and ``np.interp`` extrapolates flat, so
    the returned loudness level saturates silently at 120 phon while the
    sone value keeps growing (a 1 kHz tone at 130 dB returns N = 760 sone,
    L_N = 120 phon). The standard marks that range as unvalidated; callers
    needing a warning should check the sone value themselves. The
    ``loudness_moore_glasberg_time`` Table 5 mapping saturates identically.
    """
    if loudness <= 0.0:
        return 0.0
    return float(np.interp(math.log10(loudness), _LOG_T5_SONE, _T5_PHON))


def _result_from_components(
    freqs: np.ndarray, levels: np.ndarray, field: str, presentation: str
) -> MooreGlasbergLoudness:
    """Run the full signal chain for one spectrum and both ears."""
    if freqs.size == 0:
        n_spec = np.zeros(_FC_GRID.size, dtype=np.float64)
        loudness = 0.0
    else:
        levels_cochlea = _cochlea_levels(freqs, levels, field)
        excitation = _excitation_pattern(freqs, levels_cochlea)
        n_spec = _specific_loudness(excitation)
        if presentation == "monaural":
            silent = np.zeros_like(n_spec)
            loud_left, _ = _binaural_loudness(n_spec, silent)
            loudness = loud_left
        else:  # binaural / diotic: identical spectrum at both ears
            loud_left, loud_right = _binaural_loudness(n_spec, n_spec)
            loudness = loud_left + loud_right
    return MooreGlasbergLoudness(
        loudness=loudness,
        loudness_level=_loudness_level(loudness),
        specific=n_spec,
        erb_number=_I_GRID.copy(),
        centre_frequencies=_FC_GRID.copy(),
        field=field,
        presentation=presentation,
    )


# ---------------------------------------------------------------------------
# Input validation and public API
# ---------------------------------------------------------------------------


def _validate_conditions(field: str, presentation: str) -> None:
    if field not in _FIELDS:
        msg = f"field must be one of {_FIELDS}, got {field!r}."
        raise ValueError(msg)
    if presentation not in _PRESENTATIONS:
        msg = f"presentation must be one of {_PRESENTATIONS}, got {presentation!r}."
        raise ValueError(msg)


def _as_components(
    components: Sequence[tuple[float, float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a sequence of (frequency, level) pairs into arrays and validate."""
    array = np.asarray(components, dtype=np.float64)
    if array.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:  # noqa: PLR2004
        msg = (
            "components must be a sequence of (frequency_Hz, level_dB) pairs, "
            f"got array of shape {array.shape}."
        )
        raise ValueError(msg)
    freqs = array[:, 0]
    levels = array[:, 1]
    if not np.all(np.isfinite(array)):
        msg = "components must contain only finite values."
        raise ValueError(msg)
    if np.any(freqs <= 0.0):
        msg = "component frequencies must be positive."
        raise ValueError(msg)
    return freqs, levels


def loudness_moore_glasberg_from_spectrum(
    components: Sequence[tuple[float, float]] | np.ndarray,
    *,
    field: Literal["free", "diffuse", "eardrum"] = "free",
    presentation: Literal["binaural", "diotic", "monaural"] = "binaural",
) -> MooreGlasbergLoudness:
    """Moore-Glasberg loudness from a sinusoidal-component spectrum.

    Implements the exact spectral input of ISO 532-2:2017 clauses 5.2/5.4:
    the sound is specified as a set of discrete sinusoidal components, each a
    ``(frequency_Hz, level_dB_SPL)`` pair (levels re 20 uPa in the stated
    sound field).  Bands of noise can be represented by the equivalent set of
    closely spaced components of clause 5.3.

    :param components: Sequence of ``(frequency_Hz, level_dB)`` pairs (or an
        ``(n, 2)`` array).  An empty spectrum yields zero loudness.
    :param field: Listening condition setting the outer-ear transfer:
        ``"free"`` (frontal free field, default), ``"diffuse"`` (diffuse
        field) or ``"eardrum"`` (levels already specified at the tympanic
        membrane, e.g. a flat earphone).
    :param presentation: ``"binaural"`` (default; ``"diotic"`` is an
        equivalent alias: the same sound at both ears) or ``"monaural"``
        (one ear only).
    :return: A :class:`MooreGlasbergLoudness`.

    A 1 kHz tone at 40 dB SPL in a free field presented binaurally yields
    1.000 sone / 40 phon by definition of the sone.
    """
    _validate_conditions(field, presentation)
    freqs, levels = _as_components(components)
    return _result_from_components(freqs, levels, field, presentation)


def _third_octave_components(
    band_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sinusoidal-component representation of one-third-octave levels
    (clause 5.5).

    Each band is treated as flat: its spectrum level is
    :math:`L_T - 10 \log_{10}(W/\text{Hz})`
    with band width :math:`W = f_T (2^{1/6} - 2^{-1/6})`, and it is replaced
    by components spaced 10 Hz apart (1 Hz for centre frequencies <= 125 Hz)
    with a level of the spectrum level plus
    :math:`10 \log_{10}(\text{spacing}/\text{Hz})`.
    """
    freqs: list[float] = []
    levels: list[float] = []
    for centre, level in zip(_THIRD_OCTAVE_FREQ, band_levels):
        width = centre * (_THIRD_OCTAVE_RATIO - 1.0 / _THIRD_OCTAVE_RATIO)
        spectrum_level = float(level) - 10.0 * math.log10(width)
        spacing = 1.0 if centre <= _FINE_SPACING_MAX_CENTRE_HZ else 10.0
        component_level = spectrum_level + 10.0 * math.log10(spacing)
        f_lo = centre / _THIRD_OCTAVE_RATIO
        f_hi = centre * _THIRD_OCTAVE_RATIO
        first = math.ceil(f_lo / spacing)
        last = math.floor(f_hi / spacing)
        for index in range(first, last + 1):
            freqs.append(index * spacing)
            levels.append(component_level)
    return np.asarray(freqs, dtype=np.float64), np.asarray(levels, dtype=np.float64)


def loudness_moore_glasberg_from_third_octave(
    band_levels: Sequence[float] | np.ndarray,
    *,
    field: Literal["free", "diffuse", "eardrum"] = "free",
    presentation: Literal["binaural", "diotic", "monaural"] = "binaural",
) -> MooreGlasbergLoudness:
    """Moore-Glasberg loudness from 29 one-third-octave band levels.

    Implements the practical spectral input of ISO 532-2:2017 clause 5.5: the
    sound is specified by the sound pressure levels in the 29 adjacent
    one-third-octave bands with nominal centre frequencies 25 Hz to 16 kHz
    (IEC 61260-1).  Each band is expanded into the equivalent set of
    sinusoidal components and the exact method is applied.

    :param band_levels: 29 band levels in dB SPL (re 20 uPa), 25 Hz..16 kHz.
    :param field: ``"free"`` (default), ``"diffuse"`` or ``"eardrum"``.
    :param presentation: ``"binaural"`` (default; alias ``"diotic"``) or
        ``"monaural"``.
    :return: A :class:`MooreGlasbergLoudness`.
    """
    _validate_conditions(field, presentation)
    levels = np.asarray(band_levels, dtype=np.float64)
    if levels.ndim != 1 or levels.size != _N_THIRD_OCTAVE:
        msg = (
            f"band_levels must contain exactly {_N_THIRD_OCTAVE} one-third-"
            f"octave band levels (25 Hz to 16 kHz), got shape {levels.shape}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(levels)):
        msg = "band_levels must contain only finite values."
        raise ValueError(msg)
    freqs, comp_levels = _third_octave_components(levels)
    return _result_from_components(freqs, comp_levels, field, presentation)


_SIGNAL_FLOOR_DB = 100.0  # discard bins > 100 dB below the spectral peak
_AUDIBLE_LO, _AUDIBLE_HI = 20.0, 20000.0  # component band retained (clause 7.2)


def _signal_components(pressure: np.ndarray, fs: float) -> np.ndarray:
    r"""Narrowband sinusoidal-component spectrum of a pressure signal.

    ISO 532-2:2017 is a spectrum-based method whose exact input (clauses
    5.2/5.4) is a set of discrete sinusoidal components ``(frequency, level)``.
    The natural representation of a recorded signal is therefore its narrowband
    (FFT) line spectrum: every FFT bin becomes one component at its own
    frequency with its calibrated sound pressure level re 20 uPa.  A pure tone
    then enters the excitation model (Formula 5, evaluated per component) as a
    single line at its frequency and level - as the standard's convention
    requires - rather than being smeared over a one-third-octave band.

    The single-sided power spectrum uses the power-preserving (Parseval)
    window normalisation :math:`\lvert X \rvert^2 / (N \sum w^2)` so that the
    summed bin powers equal the signal's mean square regardless of the
    window: a 1 kHz tone at 40 dB SPL yields 1.000 sone, the definitional
    anchor of the sone.
    Bins outside the audible range or more than ``_SIGNAL_FLOOR_DB`` below the
    spectral peak are dropped as inaudible.
    """
    n = pressure.size
    window = np.hanning(n)
    window_power = float(np.sum(window**2))
    if window_power <= 0.0:
        return np.empty((0, 2), dtype=np.float64)
    spectrum = np.fft.rfft(pressure * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Single-sided mean-square contribution of each FFT bin (Parseval).
    power = np.abs(spectrum) ** 2 / (n * window_power)
    power[1:] *= 2.0
    levels = 10.0 * np.log10(np.maximum(power, 1e-300) / (2e-5) ** 2)
    audible = (freqs >= _AUDIBLE_LO) & (freqs <= _AUDIBLE_HI)
    keep = audible & (levels > levels.max() - _SIGNAL_FLOOR_DB)
    return np.column_stack((freqs[keep], levels[keep])).astype(np.float64)


def loudness_moore_glasberg(
    x: Signal | list[float] | np.ndarray,
    fs: float | None = None,
    *,
    field: Literal["free", "diffuse", "eardrum"] = "free",
    presentation: Literal["binaural", "diotic", "monaural"] = "binaural",
) -> MooreGlasbergLoudness:
    """Moore-Glasberg loudness of a calibrated stationary pressure signal.

    Convenience wrapper around
    :func:`loudness_moore_glasberg_from_spectrum`: the signal's narrowband
    (FFT) line spectrum is formed and each bin is passed to the exact
    sinusoidal-component method of ISO 532-2:2017 (clauses 5.2/5.4).  Because
    the model computes the excitation pattern per spectral component
    (Formula 5), a pure tone enters as a single line - so a calibrated 1 kHz
    tone at 40 dB SPL yields 1.000 sone / 40 phon (the definitional anchor),
    matching :func:`loudness_moore_glasberg_from_spectrum`.  The signal must be
    calibrated so that ``x`` is the instantaneous sound pressure in pascals.

    :param x: Single-channel calibrated pressure signal in pascals. Accepts a
        :class:`phonometry.io.Signal`, which is where "calibrated" comes
        from without arithmetic: this model reads absolute levels, so an
        uncalibrated record is taken as if one digital unit were one
        pascal and the answer is wrong by however far that is from true.
    :param fs: Sampling rate in Hz (positive). Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param field: ``"free"`` (default), ``"diffuse"`` or ``"eardrum"``.
    :param presentation: ``"binaural"`` (default; alias ``"diotic"``) or
        ``"monaural"``.
    :return: A :class:`MooreGlasbergLoudness`.
    """
    _validate_conditions(field, presentation)
    fs = resolve_fs(x, fs)
    pressure = apply_calibration(
        x, require_1d_signal(_typesignal(np.asarray(x)), name="'x'")
    )
    if pressure.size == 0:
        msg = "Input signal 'x' cannot be empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(pressure)):
        msg = "Input signal 'x' must contain only finite values."
        raise ValueError(msg)
    if fs <= 0.0:
        msg = f"'fs' must be a positive sampling rate, got {fs!r}."
        raise ValueError(msg)
    components = _signal_components(pressure, float(fs))
    return loudness_moore_glasberg_from_spectrum(
        components, field=field, presentation=presentation
    )
