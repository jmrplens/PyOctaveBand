#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Room acoustic parameters from impulse responses per ISO 3382-1:2009
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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    check_engine,
    require_ranks,
    require_same_length,
)
from ..filters.core import OctaveFilterBank
from ..io._resolve import resolve_fs
from ._shared import (
    MIN_LINE_FIT_POINTS,
    TRUST_MARGIN_DB,
    noise_power,
    onset_index,
    split_bands,
    truncation,
    validate_ir,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from .._report.metadata import ReportMetadata
    from ..io._signal import Signal

#: Default octave-band analysis range (ISO 3382-1:2009, 5.1: engineering
#: and precision methods cover at least 125 Hz to 4 kHz in octave bands).
_DEFAULT_BANDS = (125.0, 4000.0)

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

    def __post_init__(self) -> None:
        """Reject a result whose parameters do not run over the same bands.

        The ISO 3382 fiche prints one row per band, taking the row count from
        ``t30`` and reading every other parameter at that row's index, with
        the band label of the row taken from ``frequency``. A parameter one
        entry longer than the rest is dropped off the end of the table; one
        entry shorter raises an index error out of the row builder itself,
        while the rows are still plain lists, naming neither the field nor
        the two lengths. Worse than either, the boxed mid-frequency
        reverberation time is read by looking the 500 Hz and 1000 Hz bands up
        in ``frequency`` and taking ``t30`` at the index found there, so a
        band axis that has slipped against the parameters quotes another
        band's decay time under the "500-1000 Hz" label, on a sheet that
        renders without complaint and whose verdict row then compares that
        number with the target.

        The validity flags travel the same axis: the plot greys and hatches
        the bars of the bands they reject, pairing each bar with its flag
        under a strict zip, so a flag array of another length stops the
        drawing part-way through in either direction, leaving the bars it had
        already reached on whatever axes the caller passed in.

        :raises ValueError: if any per-band array disagrees with the rest.
        """
        require_ranks(
            self,
            frequency=1,
            edt=1,
            t20=1,
            t30=1,
            c50=1,
            c80=1,
            d50=1,
            ts=1,
            dynamic_range=1,
            edt_valid=1,
            t20_valid=1,
            t30_valid=1,
            curvature=1,
        )
        require_same_length(
            self,
            "frequency",
            "edt",
            "t20",
            "t30",
            "c50",
            "c80",
            "d50",
            "ts",
            "dynamic_range",
            "edt_valid",
            "t20_valid",
            "t30_valid",
            "curvature",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | np.ndarray:
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
        row appears only when a target T is supplied through
        ``metadata.requirement`` (read as the maximum acceptable value of
        whichever descriptor the box carries). A broadband result
        (``frequency`` is ``None``) has no 500 Hz and 1000 Hz bands to average,
        so the box and the verdict fall back to the plain broadband T30 instead
        of a mid-frequency average, with no "500-1000 Hz" label; so does a
        banded result that does not span both mid bands, or that spans them
        with a NaN T30 in either, which box and compare the first finite T30
        band (the box names that band, the verdict line does not).

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            characterisation fiche (body, result and disclaimer only). The
            room-specific fields ``room_volume``, ``source_positions`` and
            ``receiver_positions`` populate the header; ``requirement`` is read
            as the maximum acceptable reverberation time, compared with the
            descriptor the result box shows (the mid-frequency T where both mid
            bands carry a finite T30, otherwise a single band's T30).
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
        check_engine(engine)
        from .._report.iso3382 import render_iso3382_report

        return render_iso3382_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


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
    noise = noise_power(p2)
    i1, tail_energy, tail_moment = truncation(p2, fs, noise)
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
    if int(mask.sum()) < MIN_LINE_FIT_POINTS or float(level.min()) > lower:
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
    p2 = p2[onset_index(p2) :]
    noise = noise_power(p2)
    peak = float(p2.max())
    dyn = 10.0 * np.log10(peak / noise) if noise > 0.0 else float("inf")
    time, level, cumulative, total, i1, tail_moment = _schroeder(p2, fs)
    trust_floor = -(dyn - TRUST_MARGIN_DB) if np.isfinite(dyn) else -np.inf

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

    def __post_init__(self) -> None:
        """Reject a curve whose two halves are not the same curve.

        ``time`` and ``level`` are one sampled decay read off by position:
        the plot draws the levels against the times point for point, and each
        straight-line fit cuts its evaluation range as a mask over ``level``
        and indexes ``time`` with it, so a decay time is only ever as right as
        the pairing. Nothing downstream re-establishes it, and the dataclass
        unpacks as ``time, level = decay_curve(...)``, which sends the two
        arrays on into code that never sees this class again. Left to be
        discovered there, the halves are reported by matplotlib or numpy as a
        pair of shapes, from inside whatever finally drew them, which for a
        curve computed once and plotted later is a long way from the point
        the two parted company.

        ``band`` is a single centre frequency labelling the whole curve, not
        an axis, and is left out.

        :raises ValueError: if ``time`` and ``level`` differ in length.
        """
        require_ranks(self, time=1, level=1)
        require_same_length(self, "time", "level", axis="sample")

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield ``time`` then ``level`` so the result unpacks like a tuple."""
        yield self.time
        yield self.level

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
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
    ir: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    band: float | None = None,
    fraction: int = 1,
    zero_phase: bool = False,
) -> DecayCurve:
    """Schroeder decay curve of an impulse response.

    Backward integration of the squared impulse response
    (ISO 3382-1:2009, 5.3.3, Equation (1)), with noise truncation at the
    crossing of the background-noise level with the fitted decay slope
    and exponential compensation of the missing tail (Equation (3)).
    Time zero is the start of the direct sound (A.2.1) and the level is
    referenced to the steady-state level (the total energy of the
    integrated impulse response, Clause 6).

    :param ir: Measured impulse response (1D), e.g. from
        :func:`phonometry.room.impulse_response` (ISO 18233). Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples and then cancels: the curve is normalised to 0 dB at its
        start, so a factor on the record moves neither the levels nor the
        decay times read off them.
    :param fs: Sample rate in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
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
    fs = resolve_fs(ir, fs, name="ir")
    x = validate_ir(ir, fs)
    if band is not None:
        if band <= 0.0:
            msg = "Band centre frequency 'band' must be positive."
            raise ValueError(msg)
        if not np.isfinite(fraction) or fraction <= 0:
            msg = "Bandwidth 'fraction' must be positive."
            raise ValueError(msg)
        half_width = 2.0 ** (1.0 / (4.0 * fraction))
        bank = OctaveFilterBank(
            fs=fs,
            fraction=fraction,
            order=6,
            limits=[band / half_width, band * half_width],
        )
        _, freqs, signals = bank.filter(
            x,
            sigbands=True,
            detrend=False,
            calculate_level=False,
            zero_phase=zero_phase,
        )
        if len(freqs) == 0:
            msg = (
                f"'band' ({band:g} Hz) has no filter at fs={fs} Hz; "
                "its upper edge exceeds the Nyquist frequency."
            )
            raise ValueError(msg)
        idx = int(np.argmin(np.abs(np.asarray(freqs, dtype=np.float64) - band)))
        # np.asarray, not a cast: a bank on the default calibration hands
        # back Signals, and the squaring below is array arithmetic.
        x = np.asarray(signals[idx])
    p2 = x.astype(np.float64) ** 2
    if not np.any(p2 > 0.0):
        msg = "The selected band has no energy."
        raise ValueError(msg)
    p2 = p2[onset_index(p2) :]
    time, level, _, _, _, _ = _schroeder(p2, fs)
    return DecayCurve(time=time, level=level, band=band)


def room_parameters(
    ir: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    limits: tuple[float, float] | None = _DEFAULT_BANDS,
    fraction: int = 1,
    zero_phase: bool = False,
) -> RoomAcousticsResult:
    """Room acoustic parameters per ISO 3382-1:2009 / ISO 3382-2:2008.

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

    :param ir: Measured impulse response (1D). Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples and then cancels: every parameter here is a decay time, a
        ratio of energies or a centre time, and none of them moves with a
        factor on the record.
    :param fs: Sample rate in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
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
    fs = resolve_fs(ir, fs, name="ir")
    x = validate_ir(ir, fs)
    frequency, band_signals = split_bands(
        x, fs, limits, fraction, zero_phase=zero_phase
    )

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


#: Bandwidth of a fractional-octave filter as a fraction of its mid-band
#: frequency, for the uncertainty of ISO 3382-1:2009, 7.1: "For an octave
#: filter, B = 0,71 f_c, and for one-third-octave filter, B = 0,23 f_c".
#:
#: These are the printed working values, not filter design values. The exact
#: IEC 61260 band-edge widths are 2^(1/2) - 2^(-1/2) = 0,7071 and
#: 2^(1/6) - 2^(-1/6) = 0,2316, so the printed pair is each of those rounded
#: to two figures, 0,4 % and 0,7 % away. Since sigma follows the square root
#: of B, that is at most 0,4 % on the answer.
FILTER_BANDWIDTH_FRACTION = {1: 0.71, 3: 0.23}

#: The coefficients ISO 3382-1:2009, Equations (4) and (5) print, keyed by the
#: evaluation range in dB they belong to: ``(prefactor, decay term)``.
#:
#: They are the D = 20 dB and D = 30 dB rows of the general form ISO
#: 3382-2:2008 prints as Equation (A.4), at its Table A.1 column for
#: gamma = T/T_det = 5, which ISO 3382-1 does not print: G = 88 % with
#: H = 1,90, and G = 55 % with H = 1,52.
DECAY_UNCERTAINTY_COEFFICIENTS = {20.0: (0.88, 1.90), 30.0: (0.55, 1.52)}

#: Decays per position the integrated impulse response method is worth
#: (ISO 3382-1:2009, 7.2). The theory says an infinite number, but the clause
#: puts the practical figure at ten and that is what an uncertainty is
#: computed with: the two differ by 7 % on sigma(T30) and 9 % on sigma(T20).
INTEGRATED_RESPONSE_DECAYS = 10

#: The bandwidth-time product ISO 3382-1:2009, Equation (6) asks a reliable
#: decay time to clear.
MINIMUM_BANDWIDTH_TIME_PRODUCT = 16.0

#: The multiple of the averaging detector's own reverberation time that
#: Equation (7) asks a reliable decay time to clear.
MINIMUM_DETECTOR_MULTIPLE = 2.0


def filter_bandwidth(centre: ArrayLike, fraction: int = 1) -> np.ndarray | float:
    """Bandwidth of a fractional-octave filter (ISO 3382-1:2009, 7.1).

    The ``B`` of Equations (4) to (6), as the clause prints it: 0,71 times
    the mid-band frequency for an octave filter and 0,23 times it for a
    one-third-octave one. See :data:`FILTER_BANDWIDTH_FRACTION` for how
    those two figures sit against the exact IEC 61260 band edges.

    :param centre: Mid-band frequency, in Hz.
    :param fraction: Bandwidth fraction: 1 for an octave filter, 3 for a
        one-third-octave one.
    :return: The bandwidth in Hz, in the shape of ``centre``.
    :raises ValueError: If ``fraction`` is not one the clause prints a
        coefficient for, or a centre frequency is not positive and finite.
    """
    if fraction not in FILTER_BANDWIDTH_FRACTION:
        msg = (
            f"ISO 3382-1:2009, 7.1 prints a bandwidth for the octave and "
            f"one-third-octave filters, so 'fraction' must be one of "
            f"{sorted(FILTER_BANDWIDTH_FRACTION)}; got {fraction!r}."
        )
        raise ValueError(msg)
    frequencies = np.asarray(centre, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        msg = "'centre' must be a positive, finite frequency in Hz."
        raise ValueError(msg)
    width = np.asarray(
        FILTER_BANDWIDTH_FRACTION[fraction] * frequencies, dtype=np.float64
    )
    return float(width) if width.ndim == 0 else width


def reverberation_time_standard_deviation(
    reverberation_time: ArrayLike,
    bandwidth: ArrayLike,
    *,
    evaluation_range: float = 30.0,
    decays: ArrayLike = INTEGRATED_RESPONSE_DECAYS,
    positions: ArrayLike = 1,
) -> np.ndarray | float:
    r"""Standard deviation of a measured reverberation time (7.1).

    ISO 3382-1:2009, Equations (4) and (5):

    .. math::

       \sigma(T_{20}) = 0{,}88\, T_{20} \sqrt{\frac{1 + 1{,}90/n}{N B T_{20}}},
       \qquad
       \sigma(T_{30}) = 0{,}55\, T_{30} \sqrt{\frac{1 + 1{,}52/n}{N B T_{30}}}

    The uncertainty is a property of the *excitation*, not of the room: the
    interrupted-noise method restarts a random process for every decay, and
    the clause quantifies how much of the answer that randomness owns. The
    integrated impulse response method is deterministic, and 7.2 values it at
    ten interrupted-noise decays per position rather than at the infinity the
    theory gives, which is what ``decays`` defaults to.

    Note that :math:`\sigma` grows as :math:`\sqrt{T}`, not as :math:`T`: the
    prefactor's :math:`T` and the :math:`T` under the radical leave one half
    power between them, so a long reverberation time is measured with a
    larger absolute uncertainty and a smaller relative one.

    :param reverberation_time: The measured decay time, in seconds.
    :param bandwidth: Bandwidth of the analysis filter, in Hz;
        :func:`filter_bandwidth` gives the clause's own figure for it.
    :param evaluation_range: The range the decay time was fitted over, in
        dB: 20 for :math:`T_{20}` or 30 for :math:`T_{30}`, the two the
        clause prints coefficients for.
    :param decays: Decays measured in each position, the ``n`` of the
        equations. Broadcast against the rest.
    :param positions: Independent measurement positions, the ``N``: source
        and receiver combinations, not receivers alone. Broadcast against
        the rest, so a sweep over survey sizes is one call.
    :return: The standard deviation in seconds, of the broadcast shape.
    :raises ValueError: If ``evaluation_range`` is not one the clause prints
        coefficients for, if ``decays`` or ``positions`` is below one, or if a
        reverberation time or bandwidth is not positive and finite. An
        infinite ``decays`` is taken: it is the limit 7.2 declines to use.
    """
    key = float(evaluation_range)
    if key not in DECAY_UNCERTAINTY_COEFFICIENTS:
        msg = (
            f"ISO 3382-1:2009, 7.1 prints coefficients for the 20 dB and "
            f"30 dB evaluation ranges, so 'evaluation_range' must be one of "
            f"{sorted(DECAY_UNCERTAINTY_COEFFICIENTS)}; got "
            f"{evaluation_range!r}."
        )
        raise ValueError(msg)
    counted = np.asarray(decays, dtype=np.float64)
    places = np.asarray(positions, dtype=np.float64)
    if np.any(np.isnan(counted)) or np.any(counted < 1.0) or np.any(places < 1.0):
        msg = "'decays' and 'positions' count measurements, so both must be at least 1."
        raise ValueError(msg)
    times = np.asarray(reverberation_time, dtype=np.float64)
    width = np.asarray(bandwidth, dtype=np.float64)
    finite = np.all(np.isfinite(times)) and np.all(np.isfinite(width))
    if not finite or np.any(times <= 0.0) or np.any(width <= 0.0):
        msg = "'reverberation_time' and 'bandwidth' must be positive and finite."
        raise ValueError(msg)
    prefactor, decay_term = DECAY_UNCERTAINTY_COEFFICIENTS[key]
    sigma = np.asarray(
        prefactor
        * times
        * np.sqrt((1.0 + decay_term / counted) / (places * width * times)),
        dtype=np.float64,
    )
    return float(sigma) if sigma.ndim == 0 else sigma


def minimum_reliable_reverberation_time(
    bandwidth: ArrayLike, detector_time: float = 0.0
) -> np.ndarray | float:
    r"""Shortest decay time a forward analysis can be trusted with (7.3).

    ISO 3382-1:2009, Equations (6) and (7) put two lower limits on a
    reverberation time measured by traditional forward analysis, and both
    are normative:

    .. math::

       B T > 16, \qquad T > 2\, T_\mathrm{det}

    The first is the filter's: a band of width ``B`` cannot resolve a decay
    faster than its own impulse response. The second is the averaging
    detector's, and it drops out when the analysis has no detector, which is
    the case for the backward integration of 5.3.3. This function returns the
    larger of the two, which is the limit that binds. Both relations are
    strict, so the value returned is a bound the decay time has to clear and
    not one it may equal: a room whose decay time is exactly this long is
    already outside what a forward analysis can be trusted with.

    ISO 3382-2:2008, 7.3 NOTE relaxes the first to ``B T > 4`` when the
    filtering is time-reversed, which is what
    :func:`phonometry.room.room_parameters` does with ``zero_phase=True``.

    :param bandwidth: Bandwidth of the analysis filter, in Hz;
        :func:`filter_bandwidth` gives the clause's own figure for it.
    :param detector_time: Reverberation time of the averaging detector, in
        seconds. Zero, the default, for an analysis with no detector.
    :return: The exclusive lower bound on a reliable reverberation time, in
        seconds, of the shape of ``bandwidth``.
    :raises ValueError: If a bandwidth is not positive and finite, or the
        detector time is not a finite time of zero seconds or more.
    """
    width = np.asarray(bandwidth, dtype=np.float64)
    if not np.all(np.isfinite(width)) or np.any(width <= 0.0):
        msg = "'bandwidth' must be a positive, finite bandwidth in Hz."
        raise ValueError(msg)
    if not math.isfinite(detector_time) or detector_time < 0.0:
        msg = "'detector_time' must be a finite time of zero seconds or more."
        raise ValueError(msg)
    limit = np.asarray(
        np.maximum(
            MINIMUM_BANDWIDTH_TIME_PRODUCT / width,
            MINIMUM_DETECTOR_MULTIPLE * detector_time,
        ),
        dtype=np.float64,
    )
    return float(limit) if limit.ndim == 0 else limit
