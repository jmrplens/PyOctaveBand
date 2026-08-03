#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Room acoustic parameters from impulse responses per ISO 3382-1:2009
(performance spaces) and ISO 3382-2:2008 (ordinary rooms).

The measured impulse response (acquired e.g. with the swept-sine or MLS
front end of :mod:`phonometry.room.impulse_response`, ISO 18233) is filtered into
fractional-octave bands (IEC 61260) and converted to a decay curve by
Schroeder backward integration of the squared impulse response
(ISO 3382-1:2009, 5.3.3, Equation (1)). To limit the influence of
background noise, the integration is truncated at the crossing point
between the background-noise level and a sloping line fitted to the
squared impulse response, and the missing tail is compensated assuming
an exponential decay with the fitted rate (5.3.3, Equation (3)).

From the decay curve the reverberation times are evaluated by
least-squares line fits (ISO 3382-2:2008, Clause 6 and Annex C):
EDT over 0 dB to -10 dB (ISO 3382-1:2009, A.2.2), T20 over -5 dB to
-25 dB and T30 over -5 dB to -35 dB, each extrapolated to a 60 dB decay
(T = -60/slope). The energy parameters follow ISO 3382-1:2009 Annex A:
clarity C50/C80 (Equation (A.10)), definition D50 (Equation (A.11)) and
centre time Ts (Equation (A.13)), with t = 0 at the start of the direct
sound (A.2.1).

Validity flags implement the dynamic-range criterion of ISO 3382-1:2009,
5.3.3: the background noise must lie at least the evaluation range plus
15 dB below the maximum of the (squared) impulse response - 25 dB for EDT
(equivalently, the noise floor sits at least 10 dB below the lowest
evaluation point). The +15 dB rule is derived for finite forward
integration without tail compensation (C = 0), which under-estimates T;
because this module compensates the truncated tail (5.3.3, Equation (3),
C != 0) with a residual positive bias, the T20 and T30 flags add extra
headroom (46 dB for T20, 54 dB for T30) so that a flagged-valid decay
time stays within the 5 % just-noticeable difference of ISO 3382-1:2009,
Table A.1. The curvature indicator C = 100*(T30/T20 - 1) follows
ISO 3382-2:2008, B.3; values above 10 % flag a decay curve that is far
from a straight line.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.utils import _typesignal
from ..filters.core import OctaveFilterBank

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

#: Default octave-band analysis range (ISO 3382-1:2009, 5.1: engineering
#: and precision methods cover at least 125 Hz to 4 kHz in octave bands).
_DEFAULT_BANDS = (125.0, 4000.0)

#: Onset threshold: the direct sound starts where the squared IR first
#: rises to within 20 dB of its maximum (trigger point per ISO 3382-1:2009,
#: A.3.4, which places it where the signal first rises significantly above
#: the background but is more than 20 dB below the maximum; this detector
#: takes the first sample at/above the -20 dB edge, one sample inside that
#: bound, which errs early - the safe direction, see :func:`_onset_index`).
_ONSET_DB = 20.0

#: Fraction of the (onset-trimmed) response used to estimate the
#: background-noise level from its tail.
_NOISE_TAIL_FRACTION = 0.1

#: ISO 3382-1:2009, 5.3.3: noise at least evaluation range + 15 dB below
#: the maximum of the impulse response (i.e. 10 dB below the lowest
#: evaluation point).
_NOISE_MARGIN_DB = 15.0

#: Extra dynamic-range headroom (dB) beyond the ISO 3382-1 +15 dB rule for
#: the T20/T30 validity flags. ISO 3382-1:2009, 5.3.3 requires the noise to
#: lie at least the evaluation range + 15 dB below the IR maximum, but that
#: rule is derived for finite forward integration WITHOUT tail compensation
#: (C = 0), which UNDER-estimates T. This module compensates the truncated
#: tail (Schroeder Eq. (3), C != 0), whose residual bias is POSITIVE and
#: larger than the +15 dB budget: at the bare thresholds (35 dB T20, 45 dB
#: T30) a flagged-valid decay time still carries a bias above the 5 % JND
#: (ISO 3382-1:2009 Table A.1). The flagged-valid bias only falls below the
#: JND at dyn >= 46 dB (T20) and dyn >= 54 dB (T30), i.e. +11 dB / +9 dB of
#: extra headroom, at the cost of flagging borderline measurements invalid.
_T20_TAIL_HEADROOM_DB = 11.0
_T30_TAIL_HEADROOM_DB = 9.0

#: The decay curve is only trusted down to noise floor + 10 dB.
_TRUST_MARGIN_DB = 10.0

#: Moving-average window (seconds) used to smooth the squared IR before
#: fitting the sloping line of ISO 3382-1:2009, 5.3.3, Equation (3).
_SMOOTH_SECONDS = 0.010

#: Evaluation ranges in dB below the steady-state level:
#: EDT 0 -> -10 (ISO 3382-1, A.2.2); T20 -5 -> -25 and T30 -5 -> -35
#: (ISO 3382-2:2008, Clause 6).
_EDT_RANGE = (0.0, 10.0)
_T20_RANGE = (5.0, 25.0)
_T30_RANGE = (5.0, 35.0)


@dataclass(frozen=True)
class RoomAcousticsResult:
    """Per-band room acoustic parameters from one impulse response.

    All arrays have one entry per analysis band (``frequency`` holds the
    exact band centre frequencies; it is ``None`` for a broadband
    analysis, in which case the arrays have length 1). ``edt``, ``t20``
    and ``t30`` are decay times in seconds extrapolated to 60 dB
    (ISO 3382-1:2009, A.2.2; ISO 3382-2:2008, Clause 6); ``c50``/``c80``
    are early-to-late indices in dB (Equation (A.10)), ``d50`` the
    definition ratio (Equation (A.11)) and ``ts`` the centre time in
    seconds (Equation (A.13); the Table A.1 JND is 10 ms).

    ``dynamic_range`` is the peak-to-noise-floor distance of the squared
    band impulse response in dB. ``edt_valid``, ``t20_valid`` and
    ``t30_valid`` apply the ISO 3382-1:2009, 5.3.3 criterion (noise at
    least evaluation range + 15 dB below the maximum: 25 dB for EDT), with
    T20 and T30 tightened to 46 dB and 54 dB to absorb the positive bias of
    the tail compensation (5.3.3, Eq. (3)) and keep a flagged-valid value
    within the 5 % JND (ISO 3382-1:2009, Table A.1); they are False when the
    value could not be evaluated. ``curvature`` is
    C = 100*(T30/T20 - 1) in percent (ISO 3382-2:2008, B.3); values
    above 10 % indicate an unreliable, non-straight decay.
    """

    frequency: np.ndarray | None
    edt: np.ndarray
    t20: np.ndarray
    t30: np.ndarray
    c50: np.ndarray
    c80: np.ndarray
    d50: np.ndarray
    ts: np.ndarray
    dynamic_range: np.ndarray
    edt_valid: np.ndarray
    t20_valid: np.ndarray
    t30_valid: np.ndarray
    curvature: np.ndarray

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes | np.ndarray:
        """Plot per-band decay times (EDT/T20/T30) and clarity (C50/C80).

        Invalid bands are hatched and greyed. With ``ax`` given, only the
        decay-times panel is drawn on it. Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` (or array thereof).
        """
        from .._i18n import check_language
        from .._plot.room import plot_room_acoustics

        check_language(language)
        return plot_room_acoustics(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a room acoustic parameters fiche to a PDF (ISO 3382-1/-2).

        Writes a one-page report laid out like a room-acoustics measurement
        report: the standard-basis line, an optional metadata header block
        (room, volume, source/receiver positions, climate ...), the full-width
        per-band parameter table (T20/T30/EDT and C50/C80/D50/Ts) above the
        result's own per-band decay-time plot (:meth:`plot`), the boxed
        mid-frequency reverberation time T_mid (the mean of the 500 Hz and
        1000 Hz band T30; a one-third-octave analysis averages the 500 Hz and
        1 kHz one-third-octave bands and labels them as such), an optional
        verdict row and a footer with the fixed disclaimer. ISO 3382-1/-2 are
        characterisation standards with no intrinsic pass/fail, so the verdict
        row appears only when a target mid-frequency T is supplied through
        ``metadata.requirement`` (read as the maximum acceptable T_mid). A
        broadband result (``frequency`` is ``None``) has no 500 Hz and 1000 Hz
        bands to average, so the box and the verdict fall back to the plain
        broadband T30 instead of a mid-frequency average, with no
        "500-1000 Hz" label; a band range without both mid bands boxes its
        first finite T30 band, named explicitly.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            characterisation fiche (body, result and disclaimer only). The
            room-specific fields ``room_volume``, ``source_positions`` and
            ``receiver_positions`` populate the header; ``requirement`` is read
            as the maximum mid-frequency reverberation time.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for parity with the other fiches; the room
            table already shows every computed parameter, so it has no effect.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso3382 import render_iso3382_report

        return render_iso3382_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _onset_index(p2: np.ndarray) -> int:
    """Index where the direct sound starts: first sample of the squared
    IR within ``_ONSET_DB`` of its maximum (trigger per ISO 3382-1, A.3.4).

    A.3.4 places the trigger where the signal first rises significantly
    above the background but is "more than 20 dB below the maximum"; this
    detector takes the first sample at or above the -20 dB edge, i.e. one
    sample inside that bound rather than the last sample outside it.
    Taking the *first* sample within the threshold makes the detector err
    early, which is the safe direction: a late onset that clips the direct
    sound is catastrophic for the early-to-late energy ratios (a +1 ms late
    onset can cost several dB on C50/C80 and tens of ms on Ts), whereas an
    early onset is essentially harmless. The clarity/definition/centre-time
    parameters therefore rely on a clean, impulsive direct arrival; a soft
    direct sound or pre-ringing from external processing can still push
    detection late.
    """
    peak = int(np.argmax(p2))
    threshold = p2[peak] * 10.0 ** (-_ONSET_DB / 10.0)
    above = np.nonzero(p2[: peak + 1] >= threshold)[0]
    return int(above[0]) if above.size else peak


def _noise_power(p2: np.ndarray) -> float:
    """Background-noise power estimated from the tail of the squared IR."""
    tail = max(1, round(p2.size * _NOISE_TAIL_FRACTION))
    return float(np.mean(p2[-tail:]))


def _truncation(
    p2: np.ndarray, fs: int, noise_power: float
) -> tuple[int, float, float]:
    r"""Truncation point and tail compensation (ISO 3382-1, 5.3.3, Eq. (3)).

    Fits a sloping line to the smoothed squared IR (in dB) between 5 dB
    below its peak and 10 dB above the noise level; the integration stops
    at the crossing ``t1`` of that line with the noise level, and the
    missing tail is compensated assuming an exponential decay with the
    fitted rate.

    :param p2: Squared impulse response, onset-trimmed.
    :param fs: Sample rate in Hz.
    :param noise_power: Background-noise power (same units as ``p2``).
    :return: ``(i1, tail_energy, tail_first_moment)`` where ``i1`` is the
        truncation sample, ``tail_energy`` approximates
        :math:`\int_{t_1}^{\infty} p^2 \, dt` and ``tail_first_moment``
        approximates :math:`\int_{t_1}^{\infty} t \, p^2 \, dt` (both in
        seconds units, i.e. energy = sum(p2)/fs).
    """
    n = p2.size
    no_truncation = (n, 0.0, 0.0)
    if noise_power <= 0.0:
        return no_truncation
    window = min(max(1, round(_SMOOTH_SECONDS * fs)), n)
    cumulative = np.concatenate(([0.0], np.cumsum(p2)))
    smoothed = (cumulative[window:] - cumulative[:-window]) / window
    t_smooth = (np.arange(smoothed.size) + 0.5 * window) / fs
    tiny = np.finfo(np.float64).tiny
    level = 10.0 * np.log10(np.maximum(smoothed, tiny))
    noise_db = 10.0 * np.log10(noise_power)
    mask = (level <= level.max() - 5.0) & (level >= noise_db + _TRUST_MARGIN_DB)
    if int(mask.sum()) < 2:
        return no_truncation
    slope, intercept = np.polyfit(t_smooth[mask], level[mask], 1)
    # A non-negative slope means no decay; a barely-negative slope (e.g.
    # -1e-16 dB/s from fitting near-constant noise) would make the decay
    # constant alpha underflow toward 0 and the tail terms p2_t1/alpha and
    # 1/alpha**2 overflow to inf. A slope of -1e-7 dB/s implies a T60 of
    # ~6e8 s, which is physically meaningless, so treat anything shallower
    # as no decay.
    if slope >= -1e-7:
        return no_truncation
    t1 = (noise_db - intercept) / slope
    i1 = min(max(round(t1 * fs), 2), n)
    # Exponential tail with the fitted rate: p2_fit(t) = 10^((a + b*t)/10),
    # decay constant alpha = -b*ln(10)/10 (1/s).
    alpha = -slope * np.log(10.0) / 10.0
    p2_t1 = 10.0 ** ((intercept + slope * (i1 / fs)) / 10.0)
    tail_energy = p2_t1 / alpha
    tail_moment = p2_t1 * (i1 / fs / alpha + 1.0 / alpha**2)
    return i1, float(tail_energy), float(tail_moment)


def _schroeder(
    p2: np.ndarray, fs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
    r"""Backward-integrated decay curve (ISO 3382-1, 5.3.3, Eq. (1)-(3)).

    :param p2: Squared impulse response, onset-trimmed.
    :param fs: Sample rate in Hz.
    :return: ``(time, level, cumulative, total, i1, tail_moment)``:
        decay times in seconds, decay levels in dB re the steady-state
        level (0 dB at t = 0), the running early energy
        :math:`\int_0^t p^2`, the total energy including the tail
        compensation, the truncation sample ``i1`` and the tail first
        moment :math:`\int_{t_1}^{\infty} t \, p^2 \, dt`.
    """
    noise = _noise_power(p2)
    i1, tail_energy, tail_moment = _truncation(p2, fs, noise)
    cumulative = np.cumsum(p2[:i1]) / fs
    total = float(cumulative[-1]) + tail_energy
    remaining = total - np.concatenate(([0.0], cumulative[:-1]))
    tiny = np.finfo(np.float64).tiny
    level = 10.0 * np.log10(np.maximum(remaining, tiny) / total)
    time = np.arange(i1) / fs
    return time, np.asarray(level), cumulative, total, i1, tail_moment


def _fit_decay_time(
    time: np.ndarray,
    level: np.ndarray,
    decay_range: tuple[float, float],
    trust_floor_db: float,
) -> float:
    r"""Least-squares decay time over an evaluation range (Annex C).

    Fits :math:`L = a + b t` to the decay-curve samples between
    ``-decay_range[0]`` and ``-decay_range[1]`` dB and returns
    :math:`T = -60/b` (ISO 3382-2:2008, Equations (C.1)-(C.6)). NaN when the
    range is unreachable or extends below the trusted part of the curve
    (noise floor + 10 dB).
    """
    upper, lower = -decay_range[0], -decay_range[1]
    if lower < trust_floor_db:
        return float("nan")
    mask = (level <= upper) & (level >= lower)
    if int(mask.sum()) < 2 or float(level.min()) > lower:
        return float("nan")
    slope = float(np.polyfit(time[mask], level[mask], 1)[0])
    if slope >= 0.0:
        return float("nan")
    return -60.0 / slope


def _band_parameters(x: np.ndarray, fs: int) -> tuple[float, ...]:
    """All ISO 3382 parameters for one band signal.

    :return: ``(edt, t20, t30, c50, c80, d50, ts, dynamic_range)``.
    """
    nan = float("nan")
    p2 = x.astype(np.float64) ** 2
    if not np.any(p2 > 0.0):
        return (nan,) * 7 + (0.0,)
    p2 = p2[_onset_index(p2) :]
    noise = _noise_power(p2)
    peak = float(p2.max())
    dyn = 10.0 * np.log10(peak / noise) if noise > 0.0 else float("inf")
    time, level, cumulative, total, i1, tail_moment = _schroeder(p2, fs)
    trust_floor = -(dyn - _TRUST_MARGIN_DB) if np.isfinite(dyn) else -np.inf

    edt = _fit_decay_time(time, level, _EDT_RANGE, trust_floor)
    t20 = _fit_decay_time(time, level, _T20_RANGE, trust_floor)
    t30 = _fit_decay_time(time, level, _T30_RANGE, trust_floor)

    c50 = c80 = d50 = nan
    i50 = round(0.050 * fs)
    i80 = round(0.080 * fs)
    if 0 < i50 <= i1:
        early = float(cumulative[i50 - 1])
        late = total - early
        d50 = early / total
        if late > 0.0:
            c50 = 10.0 * np.log10(early / late)
    if 0 < i80 <= i1:
        early = float(cumulative[i80 - 1])
        late = total - early
        if late > 0.0:
            c80 = 10.0 * np.log10(early / late)
    first_moment = float(np.dot(time, p2[:i1])) / fs + tail_moment
    ts = first_moment / total
    return edt, t20, t30, c50, c80, d50, ts, dyn


def _validate_ir(ir: list[float] | np.ndarray, fs: int) -> np.ndarray:
    x = _typesignal(ir)
    if x.ndim != 1:
        raise ValueError("The impulse response must be one-dimensional.")
    if fs <= 0:
        raise ValueError("Sample rate 'fs' must be positive.")
    if not np.any(x):
        raise ValueError("Impulse response 'ir' is silent.")
    return x


@dataclass(frozen=True)
class DecayCurve:
    """Schroeder backward-integrated decay curve of an impulse response.

    ``time`` holds the sample times in seconds from the direct sound and
    ``level`` the decay levels in dB (0 dB at time zero), up to the noise
    truncation point (ISO 3382-1:2009, 5.3.3). ``band`` is the
    octave/third-octave band centre in Hz, or ``None`` for a broadband decay.

    For backward compatibility with the previous ``(time, level)`` tuple
    return of :func:`decay_curve`, the dataclass is iterable and unpacks as
    ``time, level = decay_curve(...)``.
    """

    time: np.ndarray
    level: np.ndarray
    band: float | None = None

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield ``time`` then ``level`` so the result unpacks like a tuple."""
        yield self.time
        yield self.level

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the decay curve with optional straight T-fit overlays.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`. Pass ``fits=False`` to omit the
        EDT/T20/T30 fit lines.
        """
        from .._i18n import check_language
        from .._plot.room import plot_decay_curve

        check_language(language)
        return plot_decay_curve(self, ax=ax, language=language, **kwargs)


def decay_curve(
    ir: list[float] | np.ndarray,
    fs: int,
    band: float | None = None,
    fraction: int = 1,
    zero_phase: bool = False,
) -> DecayCurve:
    """
    Schroeder decay curve of an impulse response.

    Backward integration of the squared impulse response
    (ISO 3382-1:2009, 5.3.3, Equation (1)), with noise truncation at the
    crossing of the background-noise level with the fitted decay slope
    and exponential compensation of the missing tail (Equation (3)).
    Time zero is the start of the direct sound (A.2.1) and the level is
    referenced to the steady-state level (the total energy of the
    integrated impulse response, Clause 6).

    :param ir: Measured impulse response (1D), e.g. from
        :func:`phonometry.impulse_response` (ISO 18233).
    :param fs: Sample rate in Hz.
    :param band: Optional band centre frequency in Hz. When given, the
        impulse response is first filtered with the matching IEC 61260
        fractional-octave filter; when None the broadband response is
        integrated directly.
    :param fraction: Bandwidth fraction of the band filter (1 = octave,
        3 = one-third octave). Only used when ``band`` is not None.
    :param zero_phase: If True, filter the band with forward-backward
        (zero-phase) filtering, removing the octave filter's group delay
        before the backward integration. ISO 3382-2:2008 Clause 7.3 NOTE
        permits time-reversed filtering (it relaxes the B*T > 16 rule to
        B*T > 4); it roughly halves the low-frequency short-decay bias at
        125 Hz. Only used when ``band`` is not None. Default False (causal).
    :return: A :class:`DecayCurve` with ``time`` in seconds from the direct
        sound and ``level`` in dB (0 dB at time zero), up to the noise
        truncation point. It unpacks as ``time, level = decay_curve(...)``
        for backward compatibility and exposes :meth:`DecayCurve.plot`.
    """
    x = _validate_ir(ir, fs)
    if band is not None:
        if band <= 0.0:
            raise ValueError("Band centre frequency 'band' must be positive.")
        half_width = 2.0 ** (1.0 / (4.0 * fraction))
        bank = OctaveFilterBank(
            fs=fs,
            fraction=fraction,
            order=6,
            limits=[band / half_width, band * half_width],
        )
        _, freqs, signals = bank.filter(
            x, sigbands=True, detrend=False, calculate_level=False,
            zero_phase=zero_phase,
        )
        idx = int(np.argmin(np.abs(np.asarray(freqs, dtype=np.float64) - band)))
        x = signals[idx]
    p2 = x.astype(np.float64) ** 2
    if not np.any(p2 > 0.0):
        raise ValueError("The selected band has no energy.")
    p2 = p2[_onset_index(p2) :]
    time, level, _, _, _, _ = _schroeder(p2, fs)
    return DecayCurve(time=time, level=level, band=band)


def room_parameters(
    ir: list[float] | np.ndarray,
    fs: int,
    limits: tuple[float, float] | None = _DEFAULT_BANDS,
    fraction: int = 1,
    zero_phase: bool = False,
) -> RoomAcousticsResult:
    """
    Room acoustic parameters per ISO 3382-1:2009 / ISO 3382-2:2008.

    The impulse response (e.g. acquired with the ISO 18233 swept-sine or
    MLS methods of :mod:`phonometry.room.impulse_response`) is filtered into
    fractional-octave bands (IEC 61260) and each band decay curve is
    obtained by Schroeder backward integration with noise truncation and
    tail compensation (ISO 3382-1:2009, 5.3.3). Least-squares line fits
    (ISO 3382-2:2008, Annex C) yield EDT (0 dB to -10 dB, ISO 3382-1,
    A.2.2), T20 (-5 dB to -25 dB) and T30 (-5 dB to -35 dB), each
    extrapolated to 60 dB. Clarity C50/C80, definition D50 and centre
    time Ts follow ISO 3382-1:2009, Equations (A.10), (A.11) and (A.13),
    with t = 0 at the start of the direct sound.

    Values that cannot be evaluated (evaluation range unreachable, or
    reaching below the noise floor + 10 dB) are NaN. The validity flags
    apply the dynamic-range criterion of ISO 3382-1:2009, 5.3.3 (noise
    at least evaluation range + 15 dB below the maximum of the impulse
    response: 25 dB for EDT), with T20 and T30 raised to 46 dB and 54 dB
    to absorb the positive bias of the tail compensation and keep a
    flagged-valid decay time within the 5 % JND (ISO 3382-1:2009,
    Table A.1).

    :param ir: Measured impulse response (1D).
    :param fs: Sample rate in Hz.
    :param limits: ``(f_min, f_max)`` band-centre limits in Hz; default
        octave bands 125 Hz to 4 kHz (ISO 3382-1:2009, 5.1). Use
        ``(100.0, 5000.0)`` with ``fraction=3`` for the one-third-octave
        engineering/precision range. ``None`` analyses the broadband
        response as a single band (``frequency`` is then ``None``).
    :param fraction: Bandwidth fraction (1 = octave, 3 = one-third
        octave). Default 1.
    :param zero_phase: If True, use forward-backward (zero-phase) octave
        filtering, removing the filter group delay before the backward
        integration. ISO 3382-2:2008 Clause 7.3 NOTE permits time-reversed
        filtering (relaxing B*T > 16 to B*T > 4); it roughly halves the
        125 Hz short-decay T30 bias (about +4.9 % -> +2.4 % at T = 0.2 s).
        The benefit is small next to the ~10 % measurement variance but is
        free and standards-sanctioned. Default False (causal filtering).
    :return: :class:`RoomAcousticsResult` with one entry per band.
    """
    x = _validate_ir(ir, fs)
    frequency: np.ndarray | None
    if limits is None:
        frequency = None
        band_signals: list[np.ndarray] = [x]
    else:
        if len(limits) != 2:
            raise ValueError("'limits' must be a (f_min, f_max) pair or None.")
        bank = OctaveFilterBank(
            fs=fs, fraction=fraction, order=6, limits=[limits[0], limits[1]]
        )
        _, freqs, band_signals = bank.filter(
            x, sigbands=True, detrend=False, calculate_level=False,
            zero_phase=zero_phase,
        )
        frequency = np.asarray(freqs, dtype=np.float64)

    values = np.array([_band_parameters(sig, fs) for sig in band_signals])
    edt, t20, t30, c50, c80, d50, ts, dyn = (values[:, k] for k in range(8))
    with np.errstate(invalid="ignore"):
        curvature = 100.0 * (t30 / t20 - 1.0)
    return RoomAcousticsResult(
        frequency=frequency,
        edt=edt,
        t20=t20,
        t30=t30,
        c50=c50,
        c80=c80,
        d50=d50,
        ts=ts,
        dynamic_range=dyn,
        edt_valid=np.isfinite(edt) & (dyn >= _EDT_RANGE[1] + _NOISE_MARGIN_DB),
        t20_valid=np.isfinite(t20)
        & (
            dyn
            >= _T20_RANGE[1] - _T20_RANGE[0] + _NOISE_MARGIN_DB + _T20_TAIL_HEADROOM_DB
        ),
        t30_valid=np.isfinite(t30)
        & (
            dyn
            >= _T30_RANGE[1] - _T30_RANGE[0] + _NOISE_MARGIN_DB + _T30_TAIL_HEADROOM_DB
        ),
        curvature=curvature,
    )
