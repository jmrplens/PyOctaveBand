#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Time-varying loudness per ISO 532-3:2023 (Moore-Glasberg-Schlittenlacher).

Clean-room implementation of the time-varying loudness model of ISO 532-3:2023,
the extension of the stationary Moore-Glasberg method of ISO 532-2:2017 to
sounds whose loudness changes over time.  The procedure of Clause 7 turns a
calibrated pressure waveform into two time histories:

* the **short-term loudness** ``S'(t)`` in sone (clause 7.8), the momentary
  loudness of a short segment such as a spoken word or a musical note; and
* the **long-term loudness** ``S''(t)`` in sone (clause 7.9), the loudness of a
  longer segment such as a whole sentence or musical phrase.

The signal chain is:

* a running short-term spectrum computed from six parallel Hann-windowed FFTs
  whose segment durations (2, 4, 8, 16, 32 and 64 ms) trade temporal against
  spectral resolution, each contributing only its own frequency range, updated
  every 1 ms (clause 7.3);
* the level-dependent rounded-exponential excitation pattern of ISO 532-2
  sampled on the ERB-number grid from 1.75 Cam to 39 Cam in 0.25 Cam steps
  (clause 7.4, Formulae 1-6);
* the compressive transformation of excitation into instantaneous specific
  loudness ``N'(i)`` in sone/Cam (clause 7.5, Formulae 7-9, Tables 2-4, with
  :math:`C = 0.063` sone/Cam);
* an attack/release temporal smoothing of the specific loudness at every centre
  frequency to the short-term specific loudness (clause 7.6, Formulae 10-13);
* the across-frequency smoothing and binaural inhibition of ISO 532-2
  (clause 7.7, Formulae 14-17), and integration over the ERB-number scale to
  the short-term loudness of each ear and their binaural sum (clause 7.8);
* a slower attack/release temporal smoothing of the short-term loudness to the
  long-term loudness (clause 7.9, Formulae 18-21).

The loudness of a sound lasting up to about 5 s is well predicted by the
maximum of the long-term loudness (clause 7.9); this ``n_max`` and the
corresponding loudness level in phon (Table 5) are the headline results the
standard asks to report (clause 9).  A steady 1 kHz tone at 40 dB SPL presented
binaurally in a free field yields a long-term loudness of 1.000 sone (40 phon)
by definition of the sone; the additive spectral calibration of clause 7.3
(nominally +3.32 dB per component) is set so this anchor holds exactly.

**Conformance mode for the sampling rate.**  Clause 5 prescribes converting
the input to 32 kHz and clause 7.3 uses fixed 2048-point FFTs; this
implementation deliberately processes at the **native sampling rate** and
grows the FFT to ``max(2048, next_pow2(segment))`` so the 64 ms Hann window
is never truncated at 44.1/48 kHz (see
``test_low_freq_band_not_truncated_across_sample_rates``).  The spectral
calibration ``_SPECTRAL_CAL_DB`` was fixed against the 32 kHz anchor; at
other rates the Annex C.1 anchor reproduces with a 0.3 % cross-rate spread
(a 0.5 s tone reads ``n_max`` = 0.9918 / 0.9926 / 0.9900 at 32 / 44.1 /
48 kHz; a settled 1.3 s tone 1.0000 / 1.0000 / 0.9982), far inside the
standard's 2.8 phon expanded uncertainty.  These values are pinned by the
test suite; resample to 32 kHz first if letter-of-standard processing is
required.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from .moore_glasberg import (
    _ERB_C1,
    _ERB_C2,
    _ROEX_D,
    _T3_10LGG,
    _T3_ALPHA,
    _cochlea_levels,
    _erb_bandwidth,
    _fc_from_cam,
    _validate_conditions,
)

# One entry per FFT window: (segment length, Hann window, sum of window^2,
# in-band bin indices, output slice into the concatenation-order level array,
# FFT length).  The FFT length is >= the segment length so the Hann window is
# never truncated (at 44.1/48 kHz the 64 ms segment exceeds the nominal 2048).
_Plan = tuple[int, np.ndarray, float, np.ndarray, slice, int]

# ---------------------------------------------------------------------------
# ERB-number grid (clause 7.4): i from 1.75 to 39 Cam in 0.25 Cam steps
# ---------------------------------------------------------------------------
_I_MIN, _I_MAX, _I_STEP = 1.75, 39.0, 0.25
_I_GRID = np.round(np.arange(_I_MIN, _I_MAX + _I_STEP / 2.0, _I_STEP), 4)
_FC_GRID = _fc_from_cam(_I_GRID)
_ERB_GRID = _erb_bandwidth(_FC_GRID)
_P_REF = 4.0 * _FC_GRID / _ERB_GRID  # p_u and p_l(51 dB, fc) per filter
_PL51_1K = 4.0 * 1000.0 / _erb_bandwidth(np.array([1000.0]))[0]  # p_l(51 dB, 1 kHz)

# ---------------------------------------------------------------------------
# Reference threshold, cochlear gain and specific-loudness parameters
# (clause 7.5, Tables 2-4).  These constants DIFFER from ISO 532-2.
# ---------------------------------------------------------------------------
# Table 2: centre frequency, excitation level at threshold [dB], 10 lg G [dB].
_T2_FREQ = np.array(
    [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 750, 800, 1000],
    dtype=np.float64,
)
_T2_LE_THR = np.array(
    [
        28.18,
        23.90,
        19.20,
        15.68,
        12.67,
        10.09,
        8.08,
        6.30,
        5.30,
        4.50,
        3.63,
        3.63,
        3.63,
        3.63,
        3.63,
    ],
    dtype=np.float64,
)
_T2_10LGG = np.array(
    [
        -24.55,
        -20.27,
        -15.57,
        -12.05,
        -9.04,
        -6.46,
        -4.45,
        -2.67,
        -1.67,
        -0.87,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
# Table 4: parameter A as a function of 10 lg G [dB] (0.5 dB grid, -25..0).
_T4_10LGG = np.round(np.arange(-25.0, 0.0 + 0.25, 0.5), 3)
_T4_A = np.array(
    [
        8.7923,
        8.6584,
        8.5245,
        8.3906,
        8.2567,
        8.1324,
        8.0095,
        7.8866,
        7.7637,
        7.6408,
        7.5179,
        7.4268,
        7.3360,
        7.2468,
        7.1562,
        7.0661,
        6.9759,
        6.8857,
        6.7984,
        6.7153,
        6.6322,
        6.5420,
        6.4518,
        6.3616,
        6.2714,
        6.1834,
        6.1002,
        6.0169,
        5.9336,
        5.8504,
        5.7671,
        5.6998,
        5.6328,
        5.5705,
        5.5082,
        5.4459,
        5.3837,
        5.3214,
        5.2591,
        5.1969,
        5.1346,
        5.0806,
        5.0287,
        4.9768,
        4.9249,
        4.8730,
        4.8211,
        4.7692,
        4.7173,
        4.6654,
        4.6135,
    ],
    dtype=np.float64,
)

_C_SONE = 0.063  # calibration constant C of Formula (7) [sone/Cam]
_ALPHA_HF = 0.2  # exponent alpha for fc >= 500 Hz (clause 7.5.4)
_E_THRQ_HF = 2.307  # E_THRQ/E0 for fc >= 500 Hz (L_E = 3.63 dB, clause 7.5.2)
_A_HF = 2.0 * _E_THRQ_HF  # A = 2 * E_THRQ/E0 for fc >= 500 Hz (clause 7.5.4)

# Precomputed per-filter specific-loudness parameters (fixed by the grid).
_HIGH = _FC_GRID >= 500.0
_G_LIN = np.ones(_FC_GRID.size)
_ALPHA = np.full(_FC_GRID.size, _ALPHA_HF)
_A_PAR = np.full(_FC_GRID.size, _A_HF)
_E_THR = np.full(_FC_GRID.size, _E_THRQ_HF)
if np.any(~_HIGH):
    _fc_low = _FC_GRID[~_HIGH]
    _le_thr = np.interp(_fc_low, _T2_FREQ, _T2_LE_THR)
    _lg_g = np.interp(_fc_low, _T2_FREQ, _T2_10LGG)
    _E_THR[~_HIGH] = np.power(10.0, _le_thr / 10.0)
    _G_LIN[~_HIGH] = np.power(10.0, _lg_g / 10.0)
    _ALPHA[~_HIGH] = np.interp(_lg_g, _T3_10LGG, _T3_ALPHA)
    _A_PAR[~_HIGH] = np.interp(_lg_g, _T4_10LGG, _T4_A)
_A_POW_ALPHA = np.power(_A_PAR, _ALPHA)

# ---------------------------------------------------------------------------
# Temporal integration time constants (clauses 7.6 and 7.9)
# ---------------------------------------------------------------------------
_ALPHA_A = 0.045  # short-term attack constant (Formula 10/11), T_a ~ 21.7 ms
_ALPHA_R = 0.033  # short-term release constant (Formula 12/13), T_r ~ 29.8 ms
_ALPHA_AL = 0.01  # long-term attack constant (Formula 18/19), T_al ~ 99.5 ms
_ALPHA_RL = 0.00133  # long-term release constant (Formula 20/21), T_rl ~ 751 ms

# ---------------------------------------------------------------------------
# Across-frequency smoothing and binaural inhibition (clause 7.7, Formulae 14-17)
# The kernel is broadly tuned (B = 0.08) on the 0.25 Cam grid; for diotic or
# monaural inputs the two smoothed patterns are proportional so the inhibition
# factor is uniform and independent of the kernel (all validation signals are
# diotic or monaural).
# ---------------------------------------------------------------------------
_INH_B = 0.08
_INH_THETA = 1.5978
# Formula (14)/(15): Di runs in 0.25 Cam steps over +/-18 Cam; the Gaussian
# weight is exp(-(B*Di)^2) with Di the ERB-number offset in Cam and B = 0.08.
_INH_TAPS = np.arange(-72, 73)  # 0.25 Cam steps, +/-18 Cam (Formula 14)
_INH_KERNEL = np.exp(-((_INH_B * (_INH_TAPS * _I_STEP)) ** 2))
_EPS = 1e-13  # additive constant of clause 7.7 (avoids division by zero)

# ---------------------------------------------------------------------------
# Running short-term spectrum (clause 7.3): six FFTs / windows / ranges
# ---------------------------------------------------------------------------
_N_FFT = 2048
_FRAME_MS = 1.0  # T0 = 1 ms frame interval
# Segment durations [s] and the frequency range [Hz] each one contributes.
_WINDOWS: tuple[tuple[float, float, float], ...] = (
    (0.064, 20.0, 80.0),
    (0.032, 80.0, 500.0),
    (0.016, 500.0, 1250.0),
    (0.008, 1250.0, 2540.0),
    (0.004, 2540.0, 4050.0),
    (0.002, 4050.0, 15000.0),
)
# Additive spectral calibration (clause 7.3): nominally +3.32 dB per component
# to offset the Hann-window loss; with the energy-preserving (Parseval) per-bin
# normalisation used here it is set so a 1 kHz tone at 40 dB SPL binaural free
# field yields exactly 1.000 sone, exactly as the standard derives the 3.32 dB.
_SPECTRAL_CAL_DB = -0.9252
_P0 = 2e-5  # reference sound pressure [Pa]
_PRUNE_DB = 80.0  # drop components > 80 dB below the frame's loudest component

# ---------------------------------------------------------------------------
# Loudness-level relationship (clause 7.10, Table 5) - DIFFERS from ISO 532-2.
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
        0.002,
        0.004,
        0.006,
        0.014,
        0.025,
        0.066,
        0.138,
        0.252,
        0.422,
        0.664,
        1.00,
        1.46,
        2.09,
        2.95,
        4.11,
        5.71,
        7.92,
        11.0,
        15.4,
        21.7,
        31.1,
        44.7,
        64.8,
        94.3,
        138.0,
        205.0,
        306.0,
    ],
    dtype=np.float64,
)
_LOG_T5_SONE = np.log10(_T5_SONE)


@dataclass(frozen=True)
class MooreGlasbergTimeVaryingLoudness:
    """Result of an ISO 532-3:2023 time-varying loudness calculation.

    ``time`` is the frame time axis in seconds (1 ms spacing, clause 7.3).
    ``short_term_loudness`` and ``long_term_loudness`` are the binaural
    short-term ``S'(t)`` (clause 7.8) and long-term ``S''(t)`` (clause 7.9)
    loudness traces in sone; ``short_term_loudness_level`` and
    ``long_term_loudness_level`` are the corresponding loudness levels in phon
    (Table 5).  ``n_max`` is the maximum of the long-term loudness (sone) - the
    predictor of the loudness of sounds up to about 5 s (clause 7.9) - and
    ``loudness_level_max`` the phon value it maps to.  ``percentiles`` gives the
    long-term-loudness values (sone) exceeded for the stated fraction of the
    active trace (e.g. ``percentiles[5]`` is the level exceeded 5 % of the
    time); the standard itself reports only the peak long-term loudness
    (clause 9), the percentiles are provided as a convenience.  ``field`` and
    ``presentation`` echo the listening conditions.
    """

    time: np.ndarray
    short_term_loudness: np.ndarray
    long_term_loudness: np.ndarray
    short_term_loudness_level: np.ndarray
    long_term_loudness_level: np.ndarray
    n_max: float
    loudness_level_max: float
    percentiles: dict[float, float]
    field: str
    presentation: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the short-term and long-term loudness against time.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.  See :mod:`phonometry._plot.psychoacoustics`.
        """
        from ..._i18n import check_language
        from ..._plot.psychoacoustics import plot_moore_glasberg_time_loudness

        return plot_moore_glasberg_time_loudness(self, ax=ax, language=check_language(language), **kwargs)


# ---------------------------------------------------------------------------
# Per-frame signal chain
# ---------------------------------------------------------------------------


def _excitation(comp_f: np.ndarray, comp_pow: np.ndarray) -> np.ndarray:
    """Excitation ratio E/E0 at every ERB-number filter (clause 7.4).

    ``comp_f`` are the (ascending) component frequencies and ``comp_pow`` their
    powers re E0.  Each filter is the level-dependent rounded-exponential (roex)
    filter of ISO 532-2 (Formulae 2-5): the upper skirt slope is level
    independent while the lower skirt broadens with the source level X of the
    component it filters (the power within a 1-ERB_n band centred on it).
    """
    if comp_f.size == 0:
        return np.zeros(_FC_GRID.size, dtype=np.float64)
    erb_c = _ERB_C1 * (_ERB_C2 * comp_f + 1.0)
    cumulative = np.concatenate(([0.0], np.cumsum(comp_pow)))
    half = erb_c / 2.0
    lo = np.searchsorted(comp_f, comp_f - half, side="left")
    hi = np.searchsorted(comp_f, comp_f + half, side="right")
    x_source = 10.0 * np.log10(np.maximum(cumulative[hi] - cumulative[lo], 1e-300))
    g = np.abs(comp_f[None, :] - _FC_GRID[:, None]) / _FC_GRID[:, None]
    upper = comp_f[None, :] > _FC_GRID[:, None]
    p_ref = _P_REF[:, None]
    p_lower = p_ref - _ROEX_D * (p_ref / _PL51_1K) * (x_source[None, :] - 51.0)
    np.clip(p_lower, 0.1, 1e4, out=p_lower)
    p = np.where(upper, p_ref, p_lower)
    weight = (1.0 + p * g) * np.exp(-p * g)
    weight[upper & (g > 4.0)] = 0.0
    return np.asarray(weight @ comp_pow, dtype=np.float64)


def _specific_loudness(excitation: np.ndarray) -> np.ndarray:
    r"""Instantaneous specific loudness N'(i) in sone/Cam (clause 7.5).

    Applies the compressive transformation of Formulae (7)-(9) with the
    ISO 532-3 constants (:math:`C = 0.063`, Tables 2-4,
    :math:`E_{THRQ}/E_0 = 2.307` for
    fc >= 500 Hz): the near-threshold Formula (8) below :math:`E_{THRQ}/E_0`
    and the high-level Formula (9) above :math:`E/E_0 = 10^{10}`.
    """
    n_spec = np.zeros(_FC_GRID.size, dtype=np.float64)
    active = excitation > 0.0
    e = excitation[active]
    core = (
        np.power(_G_LIN[active] * e + _A_PAR[active], _ALPHA[active])
        - _A_POW_ALPHA[active]
    )
    values = _C_SONE * core
    below = e < _E_THR[active]
    factor = np.power(2.0 * e / (e + _E_THR[active]), 1.5)
    values = np.where(below, _C_SONE * factor * core, values)
    very_high = e > 1e10
    values = np.where(very_high, _C_SONE * np.power(e / 1.0707, 0.2), values)
    n_spec[active] = values
    return n_spec


def _smooth(pattern: np.ndarray) -> np.ndarray:
    """Broadly tuned across-frequency smoothing (clause 7.7, Formulae 14/15)."""
    smoothed = np.zeros_like(pattern)
    n = pattern.size
    for offset, weight in zip(_INH_TAPS, _INH_KERNEL):
        src_lo = max(0, offset)
        src_hi = min(n, n + offset)
        dst_lo = max(0, -offset)
        dst_hi = min(n, n - offset)
        smoothed[dst_lo:dst_hi] += weight * pattern[src_lo:src_hi]
    return smoothed


def _short_term_loudness(stsl_l: np.ndarray, stsl_r: np.ndarray) -> tuple[float, float]:
    """Short-term loudness of each ear with binaural inhibition (clauses 7.7/7.8).

    ``stsl_*`` are the (temporally smoothed) short-term specific loudness
    patterns of the two ears.  Returns ``(S'_left, S'_right)``; their sum is the
    binaural short-term loudness.  A silent contralateral ear yields an
    inhibition factor of unity, so a monaural presentation reduces to the
    single-ear loudness.
    """
    left_sm = _smooth(stsl_l) + _EPS
    right_sm = _smooth(stsl_r) + _EPS
    ratio_l = np.clip(right_sm / left_sm, 0.0, 700.0)
    ratio_r = np.clip(left_sm / right_sm, 0.0, 700.0)
    inh_l = 2.0 / (1.0 + (1.0 / np.cosh(ratio_l)) ** _INH_THETA)
    inh_r = 2.0 / (1.0 + (1.0 / np.cosh(ratio_r)) ** _INH_THETA)
    s_left = float(np.sum(stsl_l / inh_l) * _I_STEP)
    s_right = float(np.sum(stsl_r / inh_r) * _I_STEP)
    return s_left, s_right


def _loudness_level(loudness: np.ndarray | float) -> np.ndarray | float:
    """Loudness level L_N in phon from loudness N in sone (invert Table 5)."""
    values = np.asarray(loudness, dtype=np.float64)
    out = np.zeros_like(values)
    positive = values > 0.0
    out[positive] = np.interp(np.log10(values[positive]), _LOG_T5_SONE, _T5_PHON)
    if np.isscalar(loudness) or values.ndim == 0:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# Spectral analysis (clause 7.3)
# ---------------------------------------------------------------------------


def _spectral_plan(fs: float) -> tuple[np.ndarray, list[_Plan], np.ndarray]:
    """Precompute the six-window analysis for a sampling rate.

    Returns the ascending component frequencies, a list of
    ``(length, window, sum_w2, bin_indices, out_slice, n_fft)`` per FFT window
    (the slices index a per-frame level array in concatenation order), and the
    permutation ``perm`` that sorts that concatenation-order array into
    ascending frequency order.

    The FFT length is per window: the nominal ``_N_FFT`` (2048) when the
    segment fits, otherwise the next power of two that is >= the segment length.
    This keeps the Hann window intact - at 44.1/48 kHz the 64 ms segment is
    2822/3072 samples, longer than 2048, and a fixed 2048-point FFT would
    truncate the window tail while ``sum_w2`` still normalises the full window,
    skewing the 20-80 Hz band levels.
    """
    plans = []
    freqs_parts = []
    cursor = 0
    for duration, f_lo, f_hi in _WINDOWS:
        length = max(2, round(duration * fs))
        window = np.hanning(length)
        sum_w2 = float(np.sum(window**2))
        n_fft = max(_N_FFT, 1 << (length - 1).bit_length())
        bin_freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        idx = np.nonzero((bin_freqs >= f_lo) & (bin_freqs < f_hi))[0]
        out_slice = slice(cursor, cursor + idx.size)
        cursor += idx.size
        plans.append((length, window, sum_w2, idx, out_slice, n_fft))
        freqs_parts.append(bin_freqs[idx])
    comp_f = np.concatenate(freqs_parts) if freqs_parts else np.empty(0)
    perm = np.argsort(comp_f, kind="stable")
    return comp_f[perm], plans, perm


def _frame_levels(
    signal: np.ndarray, centre: int, plans: list[_Plan], n_components: int
) -> np.ndarray:
    """Component sound pressure levels (dB SPL) for one 1-ms frame (clause 7.3).

    Each window's Hann-windowed segment is centred on ``centre`` (zero padded
    outside the signal), transformed to a per-window FFT (>= the segment length
    so the window is never truncated) and the in-band bins kept.  The per-bin
    power uses an energy-preserving (Parseval) normalisation so a sinusoid
    recovers its true mean-square regardless of spectral leakage; the fixed
    calibration of clause 7.3 is added afterwards.
    """
    levels = np.empty(n_components, dtype=np.float64)
    n = signal.size
    for length, window, sum_w2, idx, out_slice, n_fft in plans:
        start = centre - length // 2
        lo = max(0, start)
        hi = min(n, start + length)
        segment = np.zeros(length, dtype=np.float64)
        if hi > lo:
            segment[lo - start : hi - start] = signal[lo:hi]
        spectrum = np.fft.rfft(segment * window, n=n_fft)
        power = 2.0 * np.abs(spectrum[idx]) ** 2 / (n_fft * sum_w2)
        levels[out_slice] = 10.0 * np.log10(np.maximum(power, 1e-300) / _P0**2)
    return levels + _SPECTRAL_CAL_DB


def _run_ear(
    signal: np.ndarray,
    fs: float,
    field: str,
    plans: list[_Plan],
    comp_f_sorted: np.ndarray,
    perm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous and short-term specific loudness traces for one ear.

    Returns ``(instantaneous, short_term)`` arrays of shape ``(n_frames, K)``
    with ``K`` ERB-number points; the short-term pattern is the attack/release
    running average of the instantaneous one (clause 7.6, Formulae 10-13).
    """
    step = max(1, round(_FRAME_MS * 1e-3 * fs))
    centres = np.arange(0, signal.size, step)
    n_frames = centres.size
    n_components = perm.size
    inst = np.zeros((n_frames, _FC_GRID.size), dtype=np.float64)
    stsl = np.zeros((n_frames, _FC_GRID.size), dtype=np.float64)
    state = np.zeros(_FC_GRID.size, dtype=np.float64)
    for f_idx, centre in enumerate(centres):
        levels = _frame_levels(signal, int(centre), plans, n_components)[perm]
        # Prune inaudible components for speed (well below the frame peak).
        keep = levels > (levels.max() - _PRUNE_DB)
        comp_f = comp_f_sorted[keep]
        comp_levels = levels[keep]
        levels_cochlea = _cochlea_levels(comp_f, comp_levels, field)
        excitation = _excitation(comp_f, np.power(10.0, levels_cochlea / 10.0))
        n_inst = _specific_loudness(excitation)
        inst[f_idx] = n_inst
        attack = n_inst > state
        state = np.where(
            attack,
            _ALPHA_A * n_inst + (1.0 - _ALPHA_A) * state,
            _ALPHA_R * n_inst + (1.0 - _ALPHA_R) * state,
        )
        stsl[f_idx] = state
    return inst, stsl


# ---------------------------------------------------------------------------
# Input handling and public API
# ---------------------------------------------------------------------------


def _as_two_channels(
    signal: Sequence[float] | np.ndarray, presentation: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split the input into left and (optional) right calibrated pressure signals."""
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[1] == 2 and array.shape[0] != 2:
            left, right = array[:, 0], array[:, 1]
        elif array.shape[0] == 2:
            left, right = array[0], array[1]
        else:
            raise ValueError(
                f"signal must be mono (1-D) or two-channel; got shape {array.shape}."
            )
    elif array.ndim == 1:
        left = array
        right = None  # mono: diotic (both ears) or single ear (monaural)
    else:
        raise ValueError(
            f"signal must be mono (1-D) or two-channel; got shape {array.shape}."
        )
    if left.size == 0:
        raise ValueError("Input signal cannot be empty.")
    if not np.all(np.isfinite(left)) or (
        right is not None and not np.all(np.isfinite(right))
    ):
        raise ValueError("Input signal must contain only finite values.")
    if presentation == "monaural":
        right = None
    return left, right


def _percentiles(trace: np.ndarray, fractions: Sequence[float]) -> dict[float, float]:
    """Long-term-loudness value exceeded for each stated fraction of the trace."""
    if trace.size == 0:
        return {float(p): 0.0 for p in fractions}
    return {float(p): float(np.percentile(trace, 100.0 - p)) for p in fractions}


def loudness_moore_glasberg_time(
    signal: Sequence[float] | np.ndarray,
    fs: float,
    *,
    field: Literal["free", "diffuse", "eardrum"] = "free",
    presentation: Literal["binaural", "diotic", "monaural"] = "binaural",
    percentiles: Sequence[float] = (1.0, 5.0, 10.0, 50.0, 90.0, 95.0),
) -> MooreGlasbergTimeVaryingLoudness:
    """Time-varying loudness of a calibrated pressure signal (ISO 532-3:2023).

    Implements the Moore-Glasberg-Schlittenlacher method: a running short-term
    spectrum (six parallel FFTs, clause 7.3) feeds the ISO 532-2 excitation and
    specific-loudness model (clauses 7.4, 7.5), which is integrated over time to
    the short-term loudness ``S'(t)`` (clauses 7.6-7.8) and long-term loudness
    ``S''(t)`` (clause 7.9).  The signal must be calibrated so its samples are
    the instantaneous sound pressure in pascals.

    :param signal: Calibrated pressure signal in pascals.  A 1-D array is
        treated as diotic (the same sound at both ears) for a binaural/diotic
        presentation, or as the single active ear for a monaural presentation;
        a two-channel ``(n, 2)`` array gives the left and right ear signals.
    :param fs: Sampling rate in Hz (positive).
    :param field: Listening condition setting the outer-ear transfer:
        ``"free"`` (frontal free field, default), ``"diffuse"`` (diffuse field)
        or ``"eardrum"`` (levels already at the tympanic membrane).
    :param presentation: ``"binaural"``/``"diotic"`` (default) or ``"monaural"``.
    :param percentiles: Fractions (percent) for which the exceeded long-term
        loudness is reported.
    :return: A :class:`MooreGlasbergTimeVaryingLoudness`.

    A steady 1 kHz tone at 40 dB SPL, binaural, free field, yields a peak
    long-term loudness of 1.000 sone (40 phon) by definition of the sone.
    """
    _validate_conditions(field, presentation)
    if fs <= 0.0:
        raise ValueError(f"'fs' must be a positive sampling rate, got {fs!r}.")
    left, right = _as_two_channels(signal, presentation)

    comp_f, plans, perm = _spectral_plan(float(fs))
    _, stsl_l = _run_ear(left, float(fs), field, plans, comp_f, perm)
    if presentation == "monaural":
        stsl_r = np.zeros_like(stsl_l)
    elif right is None:  # diotic: identical signal at both ears
        stsl_r = stsl_l
    else:
        _, stsl_r = _run_ear(right, float(fs), field, plans, comp_f, perm)

    n_frames = stsl_l.shape[0]
    stl = np.zeros(n_frames, dtype=np.float64)
    ltl = np.zeros(n_frames, dtype=np.float64)
    # Clause 7.9: the long-term loudness is computed per ear by the attack/
    # release averager (Formulae 18-21) and the two ears are then summed.  The
    # averager is nonlinear (attack vs release branch), so summing the two
    # short-term loudnesses before smoothing would not equal the per-ear result
    # for dichotic input.  For diotic/monaural the branches coincide and this is
    # identical to smoothing the binaural sum.
    ltl_left = 0.0
    ltl_right = 0.0
    for k in range(n_frames):
        s_left, s_right = _short_term_loudness(stsl_l[k], stsl_r[k])
        stl[k] = s_left + s_right
        if s_left > ltl_left:
            ltl_left = _ALPHA_AL * s_left + (1.0 - _ALPHA_AL) * ltl_left
        else:
            ltl_left = _ALPHA_RL * s_left + (1.0 - _ALPHA_RL) * ltl_left
        if s_right > ltl_right:
            ltl_right = _ALPHA_AL * s_right + (1.0 - _ALPHA_AL) * ltl_right
        else:
            ltl_right = _ALPHA_RL * s_right + (1.0 - _ALPHA_RL) * ltl_right
        ltl[k] = ltl_left + ltl_right

    time = np.arange(n_frames, dtype=np.float64) * (_FRAME_MS * 1e-3)
    n_max = float(ltl.max()) if n_frames else 0.0
    return MooreGlasbergTimeVaryingLoudness(
        time=time,
        short_term_loudness=stl,
        long_term_loudness=ltl,
        short_term_loudness_level=np.asarray(_loudness_level(stl)),
        long_term_loudness_level=np.asarray(_loudness_level(ltl)),
        n_max=n_max,
        loudness_level_max=float(_loudness_level(n_max)),
        percentiles=_percentiles(ltl, percentiles),
        field=field,
        presentation=presentation,
    )
