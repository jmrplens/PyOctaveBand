#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Integrated and statistical sound levels (Leq, LAeq, LN percentiles).

:func:`leq`, :func:`ln_levels`, :func:`sel` and :func:`lc_peak` accept a
:class:`phonometry.io.Signal` in place of the bare ``(x, fs)`` pair: the
object read from a measurement file already knows its sample rate and,
when calibrated, its digital-to-pascal factor, so asking the caller to
repeat either is asking for a transcription error. The bare-array
signatures are unchanged -- a plain array with an explicit ``fs`` and
``calibration_factor`` computes exactly what it always did.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._internal.peaks import inter_sample_peak
from .._internal.types import as_float_or_array
from .._internal.utils import _typesignal
from ..filters.weighting import time_weighting, weighting_filter
from ..io._signal import Signal

_REF_PRESSURE = 2e-5


def _resolve_fs(x: Signal | list[float] | np.ndarray, fs: int | None) -> int:
    """Resolve the sample rate from the argument or from the Signal itself.

    A :class:`~phonometry.io.Signal` brings its own rate; an explicit one
    that disagrees is refused rather than arbitrated (the same rule as
    :func:`phonometry.io.write`): the sample rate is a fact of the
    recording, not a preference, and silently trusting either side of a
    disagreement mis-times every filter downstream. A bare array knows
    nothing about time, so there the argument is mandatory.
    """
    if isinstance(x, Signal):
        if fs is not None and fs != x.fs:
            raise ValueError(
                f"fs={fs} conflicts with the Signal's own fs={x.fs}; "
                "pass one or the other, not a disagreement"
            )
        return x.fs
    if fs is None:
        raise ValueError("fs is required when 'x' is a bare array")
    return fs


def _resolve_calibration(
    x: Signal | list[float] | np.ndarray, calibration_factor: float | None
) -> float:
    """Resolve the digital-to-pascal factor by the documented precedence.

    An explicit argument always wins -- the caller knows more than the
    object (a re-calibration after the file was written, a deliberate
    what-if). Otherwise a calibrated :class:`~phonometry.io.Signal`
    supplies the factor it carries. Otherwise 1.0: digital units straight
    through, which is exactly what the bare-array signatures have always
    computed when no factor was given.
    """
    if calibration_factor is not None:
        return calibration_factor
    if isinstance(x, Signal) and x.calibration_factor is not None:
        return x.calibration_factor
    return 1.0


def _resolve_samples(x: Signal | list[float] | np.ndarray) -> np.ndarray:
    """The float64 samples of the input, whichever form it arrived in.

    A :class:`~phonometry.io.Signal` contributes its array view (1-D for
    one channel, ``(channels, samples)`` for several), so a mono Signal
    yields the same scalar level a mono array does; bare input passes
    through :func:`_typesignal` untouched.
    """
    return _typesignal(np.asarray(x) if isinstance(x, Signal) else x)


def _level_db(mean_square: np.ndarray, calibration_factor: float, dbfs: bool) -> np.ndarray:
    """Convert mean-square values to dB SPL (re 20 uPa) or dBFS."""
    eps = np.finfo(float).eps
    rms = np.sqrt(np.maximum(mean_square, eps))
    if dbfs:
        # dBFS is relative to digital full scale: calibration does not apply
        # (consistent with OctaveFilterBank's dbfs mode).
        return np.asarray(20 * np.log10(np.maximum(rms, eps)))
    rms = rms * calibration_factor
    return np.asarray(20 * np.log10(np.maximum(rms, eps) / _REF_PRESSURE))


def _validate_level_input(x_proc: np.ndarray, calibration_factor: float) -> None:
    """Shared validation for the public level functions."""
    if x_proc.shape[-1] == 0:
        raise ValueError("Input signal 'x' cannot be empty.")
    if calibration_factor <= 0:
        raise ValueError("'calibration_factor' must be positive.")


def leq(
    x: Signal | list[float] | np.ndarray,
    calibration_factor: float | None = None,
    dbfs: bool = False,
) -> float | np.ndarray:
    """
    Equivalent continuous sound level (Leq) over the whole signal.

    :param x: Input signal (1D or 2D [channels, samples]) in raw pressure
        units, or a :class:`phonometry.io.Signal` read from a measurement
        file.
    :param calibration_factor: Multiplier converting digital units to
        Pascals. Precedence: an explicit value always wins; ``None`` (the
        default) takes the factor a calibrated
        :class:`~phonometry.io.Signal` carries, and falls back to 1.0
        (levels in digital units) for everything else.
    :param dbfs: If True, return dBFS (0 dB = RMS 1.0) instead of dB SPL;
        calibration does not apply.
    :return: Scalar for 1D input, array of shape (channels,) for 2D input.
    """
    calibration = _resolve_calibration(x, calibration_factor)
    x_proc = _resolve_samples(x)
    _validate_level_input(x_proc, calibration)
    ms = np.mean(x_proc**2, axis=-1)
    out = _level_db(np.asarray(ms), calibration, dbfs)
    return as_float_or_array(out)


def laeq(
    x: list[float] | np.ndarray,
    fs: int,
    calibration_factor: float = 1.0,
    dbfs: bool = False,
) -> float | np.ndarray:
    """
    A-weighted equivalent continuous sound level (LAeq).

    :param x: Input signal (1D or 2D [channels, samples]), raw pressure units.
    :param fs: Sample rate in Hz.
    :param calibration_factor: Multiplier converting digital units to Pascals.
    :param dbfs: If True, return dBFS instead of dB SPL.
    :return: Scalar for 1D input, array of shape (channels,) for 2D input.
    """
    return leq(weighting_filter(x, fs, "A"), calibration_factor, dbfs)


def ln_levels(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    n: Sequence[int] = (10, 50, 90),
    mode: str = "fast",
    weighting: str | None = None,
    calibration_factor: float | None = None,
    dbfs: bool = False,
) -> dict[int, float | np.ndarray]:
    """
    Statistical percentile levels (LN) from the time-weighted level envelope.

    L10 is the level exceeded 10% of the time (90th percentile of the level
    distribution), L90 the level exceeded 90% of the time, etc.

    :param x: Input signal (1D or 2D [channels, samples]) in raw pressure
        units, or a :class:`phonometry.io.Signal` read from a measurement
        file.
    :param fs: Sample rate in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit
        value that disagrees with it raises instead of silently winning.
    :param n: Percentile exceedance values, e.g. (10, 50, 90).
    :param mode: Time weighting for the envelope: 'fast', 'slow' or 'impulse'.
    :param weighting: Optional frequency weighting, any curve accepted by
        :func:`~phonometry.filters.weighting.weighting_filter`: 'A', 'B', 'C',
        'D', 'G', 'AU' or 'Z'. None (the default) and 'Z' both leave the
        signal unweighted.
    :param calibration_factor: Multiplier converting digital units to
        Pascals. Precedence as in :func:`leq`: explicit value, then a
        calibrated Signal's own factor, then 1.0.
    :param dbfs: If True, return dBFS instead of dB SPL.
    :return: Dict mapping each N to its level (scalar for 1D input,
        array (channels,) for 2D input).
    """
    fs = _resolve_fs(x, fs)
    calibration_factor = _resolve_calibration(x, calibration_factor)
    x_proc = _resolve_samples(x)
    _validate_level_input(x_proc, calibration_factor)
    for value in n:
        if not 0 < value < 100:
            raise ValueError("Percentile values in 'n' must be between 0 and 100.")
    if weighting is not None and weighting.upper() != "Z":
        x_proc = weighting_filter(x_proc, fs, weighting)

    envelope = time_weighting(x_proc, fs, mode=mode)
    # Discard the attack transient of the exponential integrator. At 2*tau the
    # F integrator is only 1-e^-2 = 86% settled (-0.6 dB), so the leading ramp
    # is counted in the distribution and drags the low percentiles down (a
    # 0.15 dB L10-L90 spread on a 2 s steady tone). 5*tau leaves it 99.3%
    # settled, cutting that residual ~12x, and matches the ~8*tau skip that
    # _validate_reference_stability already uses in calibration.py.
    tau = {"fast": 0.125, "slow": 1.0, "impulse": 0.035}[mode.lower()]
    skip = min(int(5 * tau * fs), envelope.shape[-1] // 2)
    levels_db = _level_db(envelope[..., skip:], calibration_factor, dbfs)

    result: dict[int, float | np.ndarray] = {}
    for value in n:
        p = np.percentile(levels_db, 100 - value, axis=-1)
        result[value] = float(p) if np.ndim(p) == 0 else np.asarray(p)
    return result


def lc_peak(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    calibration_factor: float | None = None,
    dbfs: bool = False,
    oversample: int = 8,
) -> float | np.ndarray:
    """
    C-weighted peak sound level, LCpeak (IEC 61672-1:2013, subclause 5.13).

    The absolute maximum of the C-weighted signal, expressed in dB. This is
    the quantity used by occupational-noise regulations (e.g. 135/137/140
    dB(C) action limits). Verified against the reference one-cycle and
    half-cycle responses of BS EN 61672-1:2013 Table 5 in the test suite.

    The true peak of a continuous waveform generally falls *between* samples.
    A raw on-grid maximum therefore under-reads sustained high-frequency
    tones (worst near integer samples-per-cycle rates, e.g. an 8 kHz tone at
    fs = 48 kHz is 6.0 samples/cycle and under-reads by up to ~1.15 dB). The
    C-weighted signal is polyphase-oversampled by ``oversample`` before the
    maximum is taken, recovering the inter-sample peak to within about
    +/-0.5 dB of the analytic value.

    :param x: Input signal (1D or 2D [channels, samples]) in raw pressure
        units, or a :class:`phonometry.io.Signal` read from a measurement
        file.
    :param fs: Sample rate in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit
        value that disagrees with it raises instead of silently winning.
    :param calibration_factor: Multiplier converting digital units to
        Pascals. Precedence as in :func:`leq`: explicit value, then a
        calibrated Signal's own factor, then 1.0.
    :param dbfs: If True, return dBFS (0 dB = peak 1.0) instead of dB SPL.
    :param oversample: Integer oversampling factor applied before peak
        detection (default 8, the audit-validated value). Use 1 to disable
        oversampling and detect the peak on the original sample grid.
    :return: Scalar for 1D input, array of shape (channels,) for 2D input.
    """
    if not isinstance(oversample, (int, np.integer)) or oversample < 1:
        raise ValueError("oversample must be an integer >= 1.")
    fs = _resolve_fs(x, fs)
    calibration_factor = _resolve_calibration(x, calibration_factor)
    x_proc = _resolve_samples(x)
    _validate_level_input(x_proc, calibration_factor)
    weighted = weighting_filter(x_proc, fs, "C")
    peak = inter_sample_peak(weighted, int(oversample))
    out = _level_db(np.asarray(peak) ** 2, calibration_factor, dbfs)
    return as_float_or_array(out)


def sel(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    weighting: str | None = None,
    calibration_factor: float | None = None,
    dbfs: bool = False,
) -> float | np.ndarray:
    r"""
    Sound exposure level (SEL / LAE): the event level normalized to 1 second.

    :math:`\text{SEL} = L_{\mathrm{eq},T} + 10 \log_{10}(T / 1\,\text{s})`, the
    standard single-event metric
    (aircraft flyovers, train passes). With ``weighting="A"`` this is LAE as
    defined by IEC 61672-1:2013 (verified against the Table 4 toneburst
    reference responses, Equation 8, in the test suite).

    :param x: Input signal covering the whole event (1D or 2D), or a
        :class:`phonometry.io.Signal` read from a measurement file.
    :param fs: Sample rate in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit
        value that disagrees with it raises instead of silently winning.
    :param weighting: Optional frequency weighting, any curve accepted by
        :func:`~phonometry.filters.weighting.weighting_filter`: 'A', 'B', 'C',
        'D', 'G', 'AU' or 'Z'. None (the default) and 'Z' both leave the
        signal unweighted.
    :param calibration_factor: Multiplier converting digital units to
        Pascals. Precedence as in :func:`leq`: explicit value, then a
        calibrated Signal's own factor, then 1.0.
    :param dbfs: If True, reference digital full scale instead of 20 uPa.
    :return: Scalar for 1D input, array of shape (channels,) for 2D input.
    """
    fs = _resolve_fs(x, fs)
    calibration_factor = _resolve_calibration(x, calibration_factor)
    x_proc = _resolve_samples(x)
    _validate_level_input(x_proc, calibration_factor)
    if fs <= 0:
        raise ValueError("Sample rate 'fs' must be positive.")
    if weighting is not None and weighting.upper() != "Z":
        x_proc = weighting_filter(x_proc, fs, weighting)
    duration_s = x_proc.shape[-1] / fs
    base = leq(x_proc, calibration_factor, dbfs)
    out = np.asarray(base) + 10 * np.log10(duration_s)
    return as_float_or_array(out)


def sound_exposure(
    x: list[float] | np.ndarray,
    fs: int,
    duration_hours: float | None = None,
    calibration_factor: float = 1.0,
) -> float | np.ndarray:
    """
    A-weighted sound exposure E in pascal-squared hours (IEC 61252, 3.1).

    The time integral of the squared A-weighted sound pressure. By default
    the input is the whole event (E integrates over ``len(x)/fs``); pass
    ``duration_hours`` to treat the input as a representative sample of a
    longer exposure period (E = mean-square * duration). Anchors from
    BS EN 61252:1995 (3.3 NOTE 4): 3.2 Pa²h <-> LEX,8h of exactly 90 dB.

    :param x: Input signal in raw pressure units (1D or 2D).
    :param fs: Sample rate in Hz.
    :param duration_hours: Exposure period the input represents, in hours.
        Default: the recording duration itself.
    :param calibration_factor: Multiplier converting digital units to Pascals.
    :return: Exposure in Pa²·h (scalar or per-channel array).
    """
    x_proc = _typesignal(x)
    _validate_level_input(x_proc, calibration_factor)
    if duration_hours is not None and duration_hours <= 0:
        raise ValueError("'duration_hours' must be positive.")
    p_a = weighting_filter(x_proc, fs, "A") * calibration_factor
    mean_square = np.mean(p_a ** 2, axis=-1)
    hours = duration_hours if duration_hours is not None else x_proc.shape[-1] / fs / 3600.0
    out = np.asarray(mean_square * hours)
    return as_float_or_array(out)


def lex_8h(
    x: list[float] | np.ndarray,
    fs: int,
    duration_hours: float | None = None,
    calibration_factor: float = 1.0,
) -> float | np.ndarray:
    """
    Normalized 8-h average sound level, LEX,8h (IEC 61252, 3.3).

    The daily personal noise exposure level: the steady level that, sustained
    over a nominal 8 h working day, carries the same A-weighted sound
    exposure as the measured event. Identical to LEP,d (Directive 86/188/EEC)
    and LEX,8h of ISO 1999 (BS EN 61252:1995, 3.3 NOTES 5-6).

    :param x: Input signal in raw pressure units (1D or 2D).
    :param fs: Sample rate in Hz.
    :param duration_hours: Exposure period the input represents, in hours.
        Default: the recording duration itself.
    :param calibration_factor: Multiplier converting digital units to Pascals.
    :return: LEX,8h in dB (scalar or per-channel array).
    """
    exposure = np.asarray(
        sound_exposure(x, fs, duration_hours=duration_hours, calibration_factor=calibration_factor)
    )
    eps = np.finfo(float).eps
    out = 10 * np.log10(np.maximum(exposure, eps) / (8.0 * _REF_PRESSURE ** 2))
    return as_float_or_array(out)
