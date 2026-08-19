#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Zwicker loudness for stationary and time-varying sounds per ISO 532-1:2017.

Clean-room Python port of the normative reference implementation given in
Annex A of ISO 532-1:2017 (program "ISO_532-1", Annex A.4).  Two entry
points are provided:

* :func:`loudness_zwicker_from_spectrum` - stationary loudness from 28
  one-third-octave band levels, 25 Hz to 12.5 kHz (clause 5.3 / A.2).
* :func:`loudness_zwicker` - loudness from a calibrated time signal,
  either with the stationary method (clause 5) or the time-varying
  method (clause 6), including the one-third-octave filterbank of
  Annex A (Tables A.1/A.2), the nonlinear temporal decay and the
  temporal weighting of the total loudness.

All numeric tables live in :mod:`._zwicker_data` and reproduce Tables A.1 to
A.9 of the standard digit for digit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ...io._resolve import resolve_calibration, resolve_fs
from ...io._signal import Signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata
from scipy import signal

from ..._internal.utils import _typesignal
from ..._internal.validation import require_positive
from ._zwicker_data import (
    A0_TRANSMISSION,
    DCB_ADAPTATION,
    DDF_DIFFUSE,
    DLL_WEIGHTS,
    FILTER_DELTA_COEFFS,
    FILTER_GAINS,
    FILTER_REFERENCE_COEFFS,
    LTQ_THRESHOLD,
    RAP_RANGES,
    RNS_RANGES,
    USL_SLOPES,
    ZUP_BARK_LIMITS,
)

# Reference intensity for band levels: I_REF = (20 uPa)^2, so that
# L = 10*lg(p^2 / I_REF) equals the SPL re 20 uPa of a pressure signal
# expressed in pascals (Annex A, constant I_REF).
_I_REF = 4e-10
# Additive floor preventing lg(0) in the level calculation (Annex A).
_TINY_VALUE = 1e-12

# Internal sampling rate of the reference algorithm (clause A.2).
_FS_REF = 48000
# Sampling rate of the one-third-octave level time series: one value
# every 0.5 ms (SR_LEVEL = 2000 Hz; clause 6.2). The 2 ms / 500 Hz spacing
# applies only to the final loudness-vs-time output (_SR_LOUDNESS).
_SR_LEVEL = 2000
# Output sampling rate of the total loudness vs. time (clause 6.5).
_SR_LOUDNESS = 500

_N_BANDS = 28  # one-third-octave bands, 25 Hz .. 12.5 kHz
_N_LCB_BANDS = 11  # bands 25 Hz .. 250 Hz grouped into critical bands
_N_CORE = 21  # core loudness values (20 critical bands + trailing zero)
_N_BARK = 240  # specific loudness samples, 0.1 Bark steps to 24 Bark

# Inner iteration factors used for virtual upsampling in the nonlinear
# decay and the temporal-weighting low-passes (Annex A).
_NL_ITER = 24
_LP_ITER = 24

# Time constants of the nonlinear temporal decay (clause 6.3).
_T_SHORT = 0.005
_T_LONG = 0.015
_T_VAR = 0.075

# Python-list copies of the Table A.8/A.9 data for the sequential slope
# state machine (scalar float access is much faster than numpy indexing).
_ZUP: list[float] = ZUP_BARK_LIMITS.tolist()
_RNS: list[float] = RNS_RANGES.tolist()
_USL: list[list[float]] = USL_SLOPES.tolist()
_N_RNS = len(_RNS)  # 18 specific-loudness ranges (Table A.9)
_N_CBR = len(_USL[0])  # 8 critical-band ranges (Table A.9)


@dataclass(frozen=True)
class ZwickerLoudness:
    """Result of an ISO 532-1:2017 Zwicker loudness calculation.

    ``loudness`` is the total loudness N in sone (the stationary value, or
    the maximum of the time-varying loudness); ``loudness_level`` is the
    loudness level LN in phon obtained from ``loudness`` with the
    sone-to-phon mapping of the reference implementation.  ``specific`` holds the
    specific loudness N' in sone/Bark at 0.1-Bark steps (240 values; for
    the time-varying method it is the pattern at the instant of maximum
    loudness).  ``n5``/``n10`` are the percentile loudness values N5/N10
    and ``time``/``loudness_vs_time`` the 500 Hz loudness-vs-time trace
    (clause 6.5); these four are ``None`` for stationary results.
    ``field`` records the sound field the calculation assumed (``"free"``
    or ``"diffuse"``), one of the items clause 7 requires a loudness report
    to state; it defaults to ``None`` only for backward-compatible manual
    construction (the library constructors always set it).
    """

    loudness: float
    loudness_level: float
    specific: np.ndarray
    n5: float | None = None
    n10: float | None = None
    time: np.ndarray | None = None
    loudness_vs_time: np.ndarray | None = None
    field: str | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes | np.ndarray:
        """Plot the specific loudness N'(z) over Bark (see :mod:`phonometry._plot.psychoacoustics`).

        Adds a loudness-vs-time panel when the time-varying trace is
        present.  Requires matplotlib (``pip install phonometry[plot]``);
        returns the :class:`~matplotlib.axes.Axes` (or array thereof).
        """
        from ..._i18n import check_language
        from ..._plot.psychoacoustics import plot_zwicker_loudness

        return plot_zwicker_loudness(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 532-1 Zwicker loudness fiche to a PDF.

        Writes a one-page accredited loudness report with the items clause 7
        makes mandatory: the standard-basis line stating the method used
        (stationary, clause 5, or time-varying, clause 6) and the sound field
        (free or diffuse, when the result carries it), an optional metadata
        header block, a compact metrics table (total loudness N, or maximum
        loudness Nmax with the N5/N10 percentiles for a time-varying result,
        and the loudness level LN) beside the specific-loudness pattern (the
        result's own :meth:`plot`), for a time-varying result the loudness-
        versus-time function N(t) in sones (clause 7 f)), the boxed
        ``N = X sone (LN = Y phon)`` result, an optional verdict row and a
        footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a
            prediction fiche (body, result and disclaimer only). A supplied
            ``requirement`` is read as the maximum permitted loudness in sone.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform signature; it has no effect on
            the single-layout loudness fiche.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso532 import render_iso532_report

        return render_iso532_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


# ---------------------------------------------------------------------------
# One-third-octave filterbank (clause A.2, Tables A.1 and A.2)
# ---------------------------------------------------------------------------


def _filterbank_sos() -> np.ndarray:
    """Second-order sections of the 28-band filterbank at 48 kHz.

    Table A.1 lists the reference section; Table A.2 the per-band/per-stage
    deviations and stage gains. The working coefficients are
    (reference - deviation), with the gains folded into the numerator --
    algebraically identical to the reference code's scaling of the
    recursion input.
    Returns shape (28, 3, 6) with rows ``[b0, b1, b2, 1, a1, a2]``.
    """
    sos = np.asarray(FILTER_REFERENCE_COEFFS[np.newaxis, :, :] - FILTER_DELTA_COEFFS)
    sos[:, :, :3] *= FILTER_GAINS[:, :, np.newaxis]
    return sos


_FILTER_SOS = _filterbank_sos()


def _band_center_frequency(band: int) -> float:
    """Exact center frequency of band ``band`` (0..27) as in Annex A."""
    return float(10.0 ** ((band - 16) / 10.0) * 1000.0)


def _third_octave_levels(
    x: np.ndarray, stationary: bool, skip_samples: int = 0
) -> np.ndarray:
    """One-third-octave band levels of a 48 kHz pressure signal.

    Implements the filtering, squaring, smoothing and level calculation of
    clause A.2.  For the stationary method one level per band is returned
    (mean square over the signal from ``skip_samples`` on; Annex B.1's
    TimeSkip); for the time-varying method the squared output is smoothed
    by three cascaded first-order low-passes with tau = 2/(3*fc) (fc capped
    at 1 kHz) and sampled every 0.5 ms (SR_LEVEL = 2000 Hz).

    :param x: Sound pressure signal in Pa at 48 kHz.
    :param stationary: Select the stationary or the time-varying method.
    :param skip_samples: Leading samples excluded from the stationary mean
        square (filterbank transient; ignored for the time-varying method).
    :return: Levels in dB, shape (28, 1) or (28, num_level_steps) at 2000 Hz.
    """
    num_samples = x.size
    if stationary:
        num_dec = 1
        dec_factor = num_samples
    else:
        dec_factor = _FS_REF // _SR_LEVEL
        num_dec = num_samples // dec_factor
        # One 500 Hz output sample needs _SR_LEVEL/_SR_LOUDNESS level steps.
        min_steps = _SR_LEVEL // _SR_LOUDNESS
        if num_dec < min_steps:
            raise ValueError(
                "Input signal is too short for the time-varying method: at "
                f"least {dec_factor * min_steps} samples at 48 kHz are required."
            )

    levels = np.empty((_N_BANDS, num_dec))
    for band in range(_N_BANDS):
        y = np.asarray(signal.sosfilt(_FILTER_SOS[band], x), dtype=np.float64)
        if stationary:
            kept = y[skip_samples:]
            mean_square = float(np.mean(kept * kept))
            levels[band, 0] = 10.0 * math.log10((mean_square + _TINY_VALUE) / _I_REF)
        else:
            # Frequency-dependent smoothing time constant (clause A.2).
            tau = 2.0 / (3.0 * min(_band_center_frequency(band), 1000.0))
            a1 = math.exp(-1.0 / (_FS_REF * tau))
            b0 = 1.0 - a1
            y = y * y
            for _ in range(3):
                y = np.asarray(signal.lfilter([b0], [1.0, -a1], y))
            decimated = y[: num_dec * dec_factor : dec_factor]
            levels[band] = 10.0 * np.log10((decimated + _TINY_VALUE) / _I_REF)
    return levels


# ---------------------------------------------------------------------------
# Core loudness (clauses 5.4 / A.2, Tables A.3 to A.7)
# ---------------------------------------------------------------------------


def _corrected_intensities(levels: np.ndarray) -> np.ndarray:
    """Equal-loudness-contour correction of the lowest 11 bands (Table A.3).

    Each band level is reduced by the Table A.3 value DLL of the first
    level range RAP whose corrected limit it does not exceed, and the
    corrected level is converted to intensity.

    :param levels: Band levels, shape (28, T).
    :return: Corrected intensities of bands 25 Hz..250 Hz, shape (11, T).
    """
    low = levels[:_N_LCB_BANDS]
    idx = np.zeros(low.shape, dtype=np.intp)
    # Sequential range search of the reference code, vectorized over time:
    # advance to range r+1 only where range r was reached and exceeded.
    for rng in range(RAP_RANGES.size - 1):
        limit = RAP_RANGES[rng] - DLL_WEIGHTS[rng]  # (11,)
        idx[(idx == rng) & (low > limit[:, np.newaxis])] = rng + 1
    reduction = DLL_WEIGHTS[idx, np.arange(_N_LCB_BANDS)[:, np.newaxis]]
    return np.asarray(10.0 ** ((low + reduction) / 10.0))


def _lcb_levels(intensities: np.ndarray) -> np.ndarray:
    """Levels of the first three critical bands (clause 5.4).

    The corrected intensities of the 11 lowest one-third-octave bands are
    summed into the critical bands 25-80 Hz, 100-160 Hz and 200-250 Hz.

    :param intensities: Corrected intensities, shape (11, T).
    :return: Critical-band levels LCB in dB, shape (3, T).
    """
    lcb = np.stack(
        (
            intensities[0:6].sum(axis=0),
            intensities[6:9].sum(axis=0),
            intensities[9:11].sum(axis=0),
        )
    )
    positive = lcb > 0.0
    lcb[positive] = 10.0 * np.log10(lcb[positive])
    return lcb


def _core_loudness(levels: np.ndarray, lcb: np.ndarray, diffuse: bool) -> np.ndarray:
    """Core loudness of the 20 critical bands (clause 5.4, Tables A.4-A.7).

    Applies the a0 transmission correction (Table A.4), optionally the
    diffuse-field difference DDF (Table A.5), the bandwidth adaptation
    DCB (Table A.7) and the loudness transformation with the threshold
    levels LTQ (Table A.6).

    :param levels: Band levels, shape (28, T).
    :param lcb: Lower critical-band levels, shape (3, T).
    :param diffuse: True for the diffuse sound field.
    :return: Core loudness, shape (21, T); the last band is always zero.
    """
    le = np.concatenate((lcb, levels[_N_LCB_BANDS:]), axis=0)  # (20, T)
    le = le - A0_TRANSMISSION[:, np.newaxis]
    if diffuse:
        le = le + DDF_DIFFUSE[:, np.newaxis]
    ltq = LTQ_THRESHOLD[:, np.newaxis]
    audible = le > ltq
    le = le - DCB_ADAPTATION[:, np.newaxis]
    s = 0.25
    mp1 = 0.0635 * 10.0 ** (0.025 * ltq)
    mp2 = (1.0 - s + s * 10.0 ** (0.1 * (le - ltq))) ** 0.25 - 1.0
    core = np.zeros((_N_CORE, le.shape[1]))
    core[: _N_CORE - 1] = np.where(audible, np.maximum(mp1 * mp2, 0.0), 0.0)
    return core


def _correct_lowest_band(core: np.ndarray) -> None:
    """Threshold-in-quiet correction within the lowest critical band.

    Clause 5.4: the specific loudness of the first critical band is scaled
    by 0.4 + 0.32*N^0.2 where this factor is below one.  Modifies ``core``
    in place.
    """
    factor = 0.4 + 0.32 * core[0] ** 0.2
    core[0] = np.where(factor < 1.0, core[0] * factor, core[0])


# ---------------------------------------------------------------------------
# Nonlinear temporal decay (clause 6.3)
# ---------------------------------------------------------------------------


def _nl_coefficients(
    sample_rate: float,
) -> tuple[float, float, float, float, float, float]:
    """Coefficients of the nonlinear decay network for ``sample_rate``.

    Analytical solution of the two-capacitor analog circuit of clause 6.3
    with the time constants 5 ms / 15 ms / 75 ms, discretized at the
    (virtually upsampled) sampling rate.
    """
    delta_t = 1.0 / sample_rate
    p = (_T_VAR + _T_LONG) / (_T_VAR * _T_SHORT)
    q = 1.0 / (_T_SHORT * _T_VAR)
    lambda1 = -p / 2 + math.sqrt(p * p / 4 - q)
    lambda2 = -p / 2 - math.sqrt(p * p / 4 - q)
    den = _T_VAR * (lambda1 - lambda2)
    e1 = math.exp(lambda1 * delta_t)
    e2 = math.exp(lambda2 * delta_t)
    return (
        (e1 - e2) / den,
        ((_T_VAR * lambda2 + 1.0) * e1 - (_T_VAR * lambda1 + 1.0) * e2) / den,
        ((_T_VAR * lambda1 + 1.0) * e1 - (_T_VAR * lambda2 + 1.0) * e2) / den,
        (_T_VAR * lambda1 + 1.0) * (_T_VAR * lambda2 + 1.0) * (e1 - e2) / den,
        math.exp(-delta_t / _T_LONG),
        math.exp(-delta_t / _T_VAR),
    )


def _nl_lp_step(
    ui: float,
    uo_last: float,
    u2_last: float,
    b: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    """One step of the nonlinear decay state machine (clause 6.3).

    ``ui`` is the input, ``uo_last``/``u2_last`` the previous output and
    inner-capacitor states.  Returns the new ``(uo, u2)``.  The three
    cases distinguish decaying input (with sub-cases for the relation of
    the capacitor voltages), constant input and rising input.
    """
    if ui < uo_last:  # decaying input
        if uo_last > u2_last:
            u2 = uo_last * b[0] - u2_last * b[1]
            uo = uo_last * b[2] - u2_last * b[3]
            uo = max(uo, ui)
            u2 = min(u2, uo)
        else:
            uo = uo_last * b[4]
            uo = max(uo, ui)
            u2 = uo
    elif abs(ui - uo_last) < 1e-5:  # steady input
        uo = ui
        u2 = (u2_last - ui) * b[5] + ui if uo > u2_last else ui
    else:  # rising input: output follows immediately
        uo = ui
        u2 = (u2_last - ui) * b[5] + ui
    return uo, u2


def _nonlinear_decay(core: np.ndarray, sample_rate: float) -> None:
    """Nonlinear temporal decay of the core loudness (clause 6.3).

    Processes each critical band independently.  Between consecutive 0.5 ms
    core-loudness samples the input is linearly interpolated and the state machine is
    advanced ``_NL_ITER`` times (virtual upsampling) for precision, as in
    the reference implementation.  Modifies ``core`` in place.
    """
    b = _nl_coefficients(sample_rate * _NL_ITER)
    num_samples = core.shape[1]
    for band in range(_N_CORE):
        row: list[float] = core[band].tolist()
        uo_last = 0.0
        u2_last = 0.0
        for t in range(num_samples - 1):
            delta = (row[t + 1] - row[t]) / _NL_ITER
            ui = row[t]
            row[t], u2_last = _nl_lp_step(ui, uo_last, u2_last, b)
            uo_last = row[t]
            ui += delta
            for _ in range(1, _NL_ITER):
                uo_last, u2_last = _nl_lp_step(ui, uo_last, u2_last, b)
                ui += delta
        row[num_samples - 1], u2_last = _nl_lp_step(
            row[num_samples - 1], uo_last, u2_last, b
        )
        core[band] = row


# ---------------------------------------------------------------------------
# Specific loudness pattern and total loudness (clause 5.5, Tables A.8/A.9)
# ---------------------------------------------------------------------------


def _rns_index_of(core_l: float) -> int:
    """Specific-loudness range index of a core loudness value (Table A.9)."""
    idx_rns = 0
    while idx_rns < _N_RNS and _RNS[idx_rns] >= core_l:
        idx_rns += 1
    return idx_rns


def _next_rns_index(idx_rns: int, n2: float) -> int:
    """Advance to the specific-loudness range of the segment end ``n2``."""
    while idx_rns < _N_RNS - 1 and n2 <= _RNS[idx_rns]:
        idx_rns += 1
    return min(idx_rns, _N_RNS - 1)


def _masked_segment(
    ns: list[float],
    state: tuple[float, float, float, int],
    core_l: float,
    zup: float,
    usl: float,
    idx_rns: int,
) -> tuple[float, float, float, float, int, bool]:
    """Extend the masking slope of the previous band by one segment.

    The slope from the previous band still masks this band: extend it with
    steepness ``usl`` until it meets the core loudness, a range limit RNS or
    the band edge ZUP, return its trapezoidal ``area`` contribution and write
    its 0.1-Bark samples into ``ns``.

    :param ns: Specific-loudness samples, written in place.
    :param state: Current ``(n1, z, z1, idx_ns)`` sampling state.
    :return: ``(n2, z2, area, z, idx_ns, next_band)``.
    """
    n1, z, z1, idx_ns = state
    n2 = max(_RNS[idx_rns], core_l)
    dz = (n1 - n2) / usl
    z2 = z1 + dz
    next_band = False
    if z2 > zup:
        next_band = True
        z2 = zup
        dz = z2 - z1
        n2 = n1 - dz * usl
    area = dz * (n1 + n2) / 2.0
    zk = z
    while zk <= z2:
        ns[idx_ns] = n1 - (zk - z1) * usl
        idx_ns += 1
        zk += 0.1
    return n2, z2, area, zk, idx_ns, next_band


def _flat_segment(
    ns: list[float],
    state: tuple[float, float, float, int],
    core_l: float,
    zup: float,
) -> tuple[float, float, float, float, int]:
    """Take the unmasked core loudness flat up to the band edge ``zup``.

    Returns the rectangular ``area`` contribution and writes the 0.1-Bark
    samples into ``ns``.

    :param ns: Specific-loudness samples, written in place.
    :param state: Current ``(n1, z, z1, idx_ns)`` sampling state.
    :return: ``(n2, z2, area, z, idx_ns)``.
    """
    _, z, z1, idx_ns = state
    z2 = zup
    n2 = core_l
    area = n2 * (z2 - z1)
    zk = z
    while zk <= z2:
        ns[idx_ns] = n2
        idx_ns += 1
        zk += 0.1
    return n2, z2, area, zk, idx_ns


def _calc_slopes(core: list[float]) -> tuple[float, list[float]]:
    """Specific loudness pattern with upper slopes for one time step.

    Attaches the masking slopes towards higher frequencies to the core
    loudness values (steepness USL per specific-loudness range RNS,
    Table A.9; critical-band limits ZUP, Table A.8), integrates the total
    loudness and samples the pattern at 0.1-Bark steps.  The two segment
    kinds live in :func:`_masked_segment` and :func:`_flat_segment`.

    :param core: 21 core loudness values.
    :return: Total loudness in sone and 240 specific loudness values.
    """
    ns = [0.0] * _N_BARK
    total = 0.0
    n1 = 0.0  # specific loudness at the current lower band edge
    z = 0.1  # next 0.1-Bark sampling position
    z1 = 0.0  # Bark position of the current lower band edge
    idx_rns = 0
    idx_ns = 0

    for idx_cl in range(_N_CORE):
        core_l = core[idx_cl]
        zup = _ZUP[idx_cl] + 0.0001
        idx_cbn = min(idx_cl - 1, _N_CBR - 1)  # never used while negative
        while True:
            if n1 > core_l:
                usl = _USL[idx_rns][idx_cbn]
                n2, z2, area, z, idx_ns, next_band = _masked_segment(
                    ns, (n1, z, z1, idx_ns), core_l, zup, usl, idx_rns
                )
            else:
                # Unmasked band: find the specific-loudness range of the
                # core value, then take it flat up to the band edge.
                if n1 < core_l:
                    idx_rns = _rns_index_of(core_l)
                next_band = True
                n2, z2, area, z, idx_ns = _flat_segment(
                    ns, (n1, z, z1, idx_ns), core_l, zup
                )
            total += area
            idx_rns = _next_rns_index(idx_rns, n2)
            z1 = z2
            n1 = n2
            if next_band:
                break
        total = max(total, 0.0)
    return total, ns


def _slopes_over_time(core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`_calc_slopes` to every time step.

    :param core: Core loudness, shape (21, T).
    :return: Total loudness (T,) and specific loudness (240, T).
    """
    num_samples = core.shape[1]
    loudness = np.empty(num_samples)
    specific = np.empty((_N_BARK, num_samples))
    core_list: list[list[float]] = core.T.tolist()
    for t in range(num_samples):
        loudness[t], specific[:, t] = _calc_slopes(core_list[t])
    return loudness, specific


# ---------------------------------------------------------------------------
# Temporal weighting of the total loudness (clause 6.4)
# ---------------------------------------------------------------------------


def _lowpass_interpolated(x: np.ndarray, tau: float, sample_rate: float) -> np.ndarray:
    """First-order low-pass with linear input interpolation (clause 6.4).

    The filter runs at ``_LP_ITER`` times the sampling rate on a linearly
    interpolated input for increased precision; one output value is kept
    per input sample (taken right after the un-interpolated sample).
    """
    a1 = math.exp(-1.0 / (sample_rate * _LP_ITER * tau))
    b0 = 1.0 - a1
    data: list[float] = x.tolist()
    out = np.empty(x.size)
    y1 = 0.0
    num_samples = len(data)
    for t in range(num_samples):
        x0 = data[t]
        y1 = b0 * x0 + a1 * y1
        out[t] = y1
        if t < num_samples - 1:
            xd = (data[t + 1] - x0) / _LP_ITER
            for _ in range(1, _LP_ITER):
                x0 += xd
                y1 = b0 * x0 + a1 * y1
    return out


def _temporal_weighting(loudness: np.ndarray, sample_rate: float) -> np.ndarray:
    """Duration-dependent weighting of the total loudness (clause 6.4).

    Weighted sum of two first-order low-passes (3.5 ms and 70 ms) with
    the factors 0.47 and 0.53.
    """
    fast = _lowpass_interpolated(loudness, 3.5e-3, sample_rate)
    slow = _lowpass_interpolated(loudness, 70e-3, sample_rate)
    return np.asarray(0.47 * fast + 0.53 * slow)


# ---------------------------------------------------------------------------
# Statistics and unit conversion
# ---------------------------------------------------------------------------


def _sone_to_phon(loudness: float) -> float:
    """Loudness level LN in phon from loudness N in sone (clause 5.6).

    Provenance notes (electronic attachment vs printed standard): the
    factor above 1 sone is the exact 10/lg 2 = 33.2193... used by the
    reference program, where Formula (2) prints the rounded 33,22 (the
    difference stays below 0.005 phon even at 1000 sone). The 3.0 phon
    floor below is the reference main program's behaviour and appears in
    no printed formula; it only affects N < 2.3e-4 sone (deep silence).
    Confirmed line by line against the attachment's Annex A.4 reference
    source (its sone-to-phon conversion adds the same 0.0005 offset inside
    the 0.35-power branch and applies an explicit 3-phon floor,
    else ``10. * log(Loudness) / log(2.) + 40.``).
    """
    if loudness < 1.0:
        loudness_level: float = 40.0 * (loudness + 0.0005) ** 0.35
        return max(loudness_level, 3.0)
    return 10.0 * math.log(loudness) / math.log(2.0) + 40.0


def _percentile(values: np.ndarray, percentile: int) -> float:
    r"""Percentile loudness as computed by the reference implementation.

    NX is the loudness exceeded X % of the time: with the values sorted
    ascending and :math:`k = \lfloor (1 - X/100)\, n \rfloor`, the mean of
    the samples at positions k-1 and k (clause 6.5, Annex A main program).

    Provenance: this (k-1, k) mean comes from the electronic attachment's
    main program, not from a printed formula, and is supported by the
    Annex B.5 results (N5 within the clause 6.1 tolerance for all twelve
    technical signals, Nmax to < 0.01 %). Confirmed line by line against
    the attachment's Annex A.4 reference source (``f_calc_percentile``:
    the reference program truncates (1 - X/100)*n to an integer index and
    averages that sorted sample with its predecessor on the ascending
    sort). The workbooks' own B.5 N5 header values are not reproducible
    from their published traces with any Annex A percentile reading, so
    the headers are held to the 5-6 % tolerance only.
    """
    ordered = np.sort(values)
    n = ordered.size
    if percentile == 0:
        return float(ordered[-1])
    if percentile == 100:
        return float(ordered[0])
    k = int((1.0 - percentile / 100.0) * n)
    k = min(max(k, 1), n - 1)
    return float((ordered[k - 1] + ordered[k]) / 2.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _validate_field(field: str) -> bool:
    """Return True for the diffuse field; raise on invalid values."""
    if field not in ("free", "diffuse"):
        raise ValueError(f"'field' must be 'free' or 'diffuse', got {field!r}.")
    return field == "diffuse"


def _core_loudness_from_levels(levels: np.ndarray, diffuse: bool) -> np.ndarray:
    """Band levels (28, T) -> corrected core loudness (21, T)."""
    intensities = _corrected_intensities(levels)
    lcb = _lcb_levels(intensities)
    core = _core_loudness(levels, lcb, diffuse)
    _correct_lowest_band(core)
    return core


def loudness_zwicker_from_spectrum(
    levels: list[float] | np.ndarray,
    field: Literal["free", "diffuse"] = "free",
) -> ZwickerLoudness:
    """
    Stationary Zwicker loudness from one-third-octave band levels.

    Implements the method for stationary sounds of ISO 532-1:2017
    (clause 5) starting from the 28 one-third-octave band levels with
    center frequencies 25 Hz to 12.5 kHz (base-ten bands, IEC 61260-1).

    :param levels: 28 band levels in dB SPL (re 20 uPa), 25 Hz..12.5 kHz.
    :param field: Sound field the levels were measured in: 'free' or
        'diffuse'.
    :return: :class:`ZwickerLoudness` with ``loudness`` (N in sone),
        ``loudness_level`` (LN in phon) and ``specific`` (N' in sone/Bark,
        240 values at 0.1-Bark steps); the time-varying fields are None.
    """
    diffuse = _validate_field(field)
    band_levels = np.asarray(levels, dtype=np.float64)
    if band_levels.ndim != 1 or band_levels.size != _N_BANDS:
        raise ValueError(
            f"'levels' must contain exactly {_N_BANDS} one-third-octave band "
            f"levels (25 Hz to 12.5 kHz), got shape {band_levels.shape}."
        )
    if not np.all(np.isfinite(band_levels)):
        raise ValueError("'levels' must contain only finite values.")
    core = _core_loudness_from_levels(band_levels[:, np.newaxis], diffuse)
    loudness, specific = _slopes_over_time(core)
    total = float(loudness[0])
    return ZwickerLoudness(
        loudness=total,
        loudness_level=_sone_to_phon(total),
        specific=specific[:, 0].copy(),
        field=field,
    )


def loudness_zwicker(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    field: Literal["free", "diffuse"] = "free",
    stationary: bool = False,
    calibration_factor: float | None = None,
    time_skip: float = 0.0,
) -> ZwickerLoudness:
    r"""
    Zwicker loudness of a calibrated time signal per ISO 532-1:2017.

    The signal is resampled to the internal 48 kHz rate if needed, split
    into 28 one-third-octave bands with the Annex A filterbank (Tables
    A.1/A.2), squared and smoothed.  With ``stationary=True`` the method
    for stationary sounds (clause 5) is applied to the per-band mean
    square of the whole signal.  Otherwise the method for time-varying
    sounds (clause 6) is used: core loudness every 0.5 ms, nonlinear
    temporal decay, specific-loudness slopes, temporal weighting of the
    total loudness and the percentile values N5/N10 from the full-rate
    (2000 Hz) weighted loudness series, while the public 500 Hz
    loudness-vs-time trace is left unchanged.

    Input scaling follows the reference implementation's WAV convention:
    ``x * calibration_factor`` must be the instantaneous sound pressure in
    pascals, so that band levels are
    :math:`10 \log_{10}(p^2 / (20~\mu\text{Pa})^2)` dB SPL.
    The reference program reads 32-bit float WAV files as pressure in Pa
    directly (``calibration_factor = 1``), while 16-bit PCM samples are
    divided by 32768 (full scale = +-1) and multiplied by a calibration
    factor derived from a reference recording of known level Lref:

    .. math::

       \text{calibration factor} = \sqrt{
           10^{L_{\text{ref}}/10} \cdot 4\times10^{-10} /
           \overline{\text{ref}^2}
       }

    with ``ref`` scaled to +-1 full scale as well.

    :param x: Single-channel time signal (see scaling convention above).
        Accepts a :class:`phonometry.io.Signal`, which is where that
        convention is met without arithmetic: a calibrated record already
        carries the digital-to-pascal factor, and this model is defined on
        pressure, so an uncalibrated record is read as if one digital unit
        were one pascal and the loudness comes out wrong by however far
        that is from true.
    :param fs: Sampling rate in Hz (positive integer; resampled to 48 kHz
        with :func:`scipy.signal.resample_poly` when not 48000). Required
        for a bare array; a :class:`~phonometry.io.Signal` brings its own,
        and an explicit value that disagrees with it raises instead of
        silently winning.
    :param field: Sound field of the recording: 'free' or 'diffuse'.
    :param stationary: Use the stationary method (clause 5) instead of the
        time-varying method (clause 6).
    :param calibration_factor: Multiplier converting ``x`` to pascals.
        An explicit value wins over the one a Signal carries, since the
        caller may know of a re-calibration the file predates; otherwise
        the object supplies it, and a bare array with neither is taken to
        be in pascals already.
    :param time_skip: Leading time, in seconds, excluded from the stationary
        mean square (the reference implementation's TimeSkip). Annex B.1
        states the stationary calculation "shall start from 0,2 s" when
        validating against the official Annex B WAV files, excluding the
        filterbank transient; the default 0.0 preserves the whole-signal
        behaviour for synthetic steady signals. Validated always (negative,
        non-finite or whole-signal skips raise :class:`ValueError`) but
        applied only by the stationary method -- clause 6 has no TimeSkip.
    :return: :class:`ZwickerLoudness`.  Stationary: as in
        :func:`loudness_zwicker_from_spectrum`.  Time-varying:
        ``loudness`` is the maximum loudness Nmax, ``loudness_level`` its
        phon mapping, ``specific`` the pattern at the loudness maximum,
        ``n5``/``n10`` the percentile values and ``time`` /
        ``loudness_vs_time`` the loudness trace at 500 Hz.
    """
    diffuse = _validate_field(field)
    fs = resolve_fs(x, fs)
    if fs <= 0:
        raise ValueError(f"'fs' must be a positive sampling rate, got {fs!r}.")
    factor = resolve_calibration(x, calibration_factor)
    require_positive(factor, "calibration_factor")
    pressure = _typesignal(np.asarray(x))
    if pressure.ndim != 1:
        raise ValueError(
            "loudness_zwicker() accepts single-channel signals only, got "
            f"shape {pressure.shape}."
        )
    if pressure.size == 0:
        raise ValueError("Input signal 'x' cannot be empty.")
    if not np.all(np.isfinite(pressure)):
        raise ValueError("Input signal 'x' must contain only finite values.")
    pressure = pressure * factor

    if int(fs) != fs:
        raise ValueError(f"'fs' must be an integer sampling rate, got {fs!r}.")
    fs_int = int(fs)
    if fs_int != _FS_REF:
        gcd = math.gcd(_FS_REF, fs_int)
        up, down = _FS_REF // gcd, fs_int // gcd
        if max(up, down) > 1000:
            raise ValueError(
                f"Sampling rate {fs_int} Hz needs an impractical resampling "
                f"ratio ({up}/{down}) to reach 48000 Hz; resample the signal "
                "to a standard rate first."
            )
        pressure = np.asarray(signal.resample_poly(pressure, up, down))

    time_skip = float(time_skip)
    if not math.isfinite(time_skip) or time_skip < 0.0:
        raise ValueError("'time_skip' must be a non-negative, finite time.")
    skip_samples = round(time_skip * _FS_REF)
    if skip_samples >= pressure.size:
        raise ValueError(
            "'time_skip' must leave at least one sample of signal "
            f"({time_skip} s skips the whole {pressure.size / _FS_REF:.3f} s input)."
        )

    levels = _third_octave_levels(pressure, stationary, skip_samples)
    core = _core_loudness_from_levels(levels, diffuse)

    if stationary:
        loudness, specific = _slopes_over_time(core)
        total = float(loudness[0])
        return ZwickerLoudness(
            loudness=total,
            loudness_level=_sone_to_phon(total),
            specific=specific[:, 0].copy(),
            field=field,
        )

    _nonlinear_decay(core, _SR_LEVEL)
    loudness, specific = _slopes_over_time(core)
    loudness = _temporal_weighting(loudness, _SR_LEVEL)

    # Loudness-vs-time output at 500 Hz (clause 6.5): plain decimation of
    # the 0.5 ms (2000 Hz) series, as in the reference main program. This decimated
    # trace remains the public ``time``/``loudness_vs_time`` contract.
    dec_factor = _SR_LEVEL // _SR_LOUDNESS
    num_out = loudness.size // dec_factor
    loudness_out = loudness[: num_out * dec_factor : dec_factor].copy()

    # N5/N10 percentiles are taken on the FULL-rate 2000 Hz weighted series
    # rather than the 4x-decimated 500 Hz trace: decimation keeps only one
    # of four phases and discards 75 % of the samples, giving up to ~3 %
    # phase-dependent spread in N5 (Annex B TS 10). The full-rate percentile
    # is phase-unambiguous and lies inside that spread. Nmax stays tied to
    # the reported 500 Hz trace (so it matches the reported pattern).
    n_max = float(np.max(loudness_out, initial=0.0))
    n5 = _percentile(loudness, 5)
    n10 = _percentile(loudness, 10)
    # The reported pattern must correspond to the reported maximum: pick the
    # same decimated instant that produced n_max, mapped back to the 0.5 ms axis.
    idx_max = int(np.argmax(loudness_out)) * dec_factor
    specific_at_max = specific[:, idx_max].copy()
    time = np.arange(num_out) / _SR_LOUDNESS
    return ZwickerLoudness(
        loudness=n_max,
        loudness_level=_sone_to_phon(n_max),
        specific=specific_at_max,
        n5=n5,
        n10=n10,
        time=time,
        loudness_vs_time=loudness_out,
        field=field,
    )
