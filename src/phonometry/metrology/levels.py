#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Integrated and statistical sound levels (Leq, LAeq, LN percentiles).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._internal.peaks import inter_sample_peak
from .._internal.types import as_float_or_array
from .._internal.utils import _typesignal
from .parametric_filters import time_weighting, weighting_filter

_REF_PRESSURE = 2e-5


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
    x: list[float] | np.ndarray,
    calibration_factor: float = 1.0,
    dbfs: bool = False,
) -> float | np.ndarray:
    """
    Equivalent continuous sound level (Leq) over the whole signal.

    :param x: Input signal (1D or 2D [channels, samples]), raw pressure units.
    :param calibration_factor: Multiplier converting digital units to Pascals.
    :param dbfs: If True, return dBFS (0 dB = RMS 1.0) instead of dB SPL.
    :return: Scalar for 1D input, array of shape (channels,) for 2D input.
    """
    x_proc = _typesignal(x)
    _validate_level_input(x_proc, calibration_factor)
    ms = np.mean(x_proc**2, axis=-1)
    out = _level_db(np.asarray(ms), calibration_factor, dbfs)
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
    x: list[float] | np.ndarray,
    fs: int,
    n: Sequence[int] = (10, 50, 90),
    mode: str = "fast",
    weighting: str | None = None,
    calibration_factor: float = 1.0,
    dbfs: bool = False,
) -> dict[int, float | np.ndarray]:
    """
    Statistical percentile levels (LN) from the time-weighted level envelope.

    L10 is the level exceeded 10% of the time (90th percentile of the level
    distribution), L90 the level exceeded 90% of the time, etc.

    :param x: Input signal (1D or 2D [channels, samples]), raw pressure units.
    :param fs: Sample rate in Hz.
    :param n: Percentile exceedance values, e.g. (10, 50, 90).
    :param mode: Time weighting for the envelope: 'fast', 'slow' or 'impulse'.
    :param weighting: Optional frequency weighting: 'A', 'C', 'Z' or None.
    :param calibration_factor: Multiplier converting digital units to Pascals.
    :param dbfs: If True, return dBFS instead of dB SPL.
    :return: Dict mapping each N to its level (scalar for 1D input,
        array (channels,) for 2D input).
    """
    x_proc = _typesignal(x)
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
    x: list[float] | np.ndarray,
    fs: int,
    calibration_factor: float = 1.0,
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

    :param x: Input signal (1D or 2D [channels, samples]), raw pressure units.
    :param fs: Sample rate in Hz.
    :param calibration_factor: Multiplier converting digital units to Pascals.
    :param dbfs: If True, return dBFS (0 dB = peak 1.0) instead of dB SPL.
    :param oversample: Integer oversampling factor applied before peak
        detection (default 8, the audit-validated value). Use 1 to disable
        oversampling and detect the peak on the original sample grid.
    :return: Scalar for 1D input, array of shape (channels,) for 2D input.
    """
    if not isinstance(oversample, (int, np.integer)) or oversample < 1:
        raise ValueError("oversample must be an integer >= 1.")
    x_proc = _typesignal(x)
    _validate_level_input(x_proc, calibration_factor)
    weighted = weighting_filter(x_proc, fs, "C")
    peak = inter_sample_peak(weighted, int(oversample))
    out = _level_db(np.asarray(peak) ** 2, calibration_factor, dbfs)
    return as_float_or_array(out)


def sel(
    x: list[float] | np.ndarray,
    fs: int,
    weighting: str | None = None,
    calibration_factor: float = 1.0,
    dbfs: bool = False,
) -> float | np.ndarray:
    """
    Sound exposure level (SEL / LAE): the event level normalized to 1 second.

    ``SEL = Leq,T + 10*log10(T / 1 s)``, the standard single-event metric
    (aircraft flyovers, train passes). With ``weighting="A"`` this is LAE as
    defined by IEC 61672-1:2013 (verified against the Table 4 toneburst
    reference responses, Equation 8, in the test suite).

    :param x: Input signal covering the whole event (1D or 2D).
    :param fs: Sample rate in Hz.
    :param weighting: Optional frequency weighting: 'A', 'C', 'Z' or None.
    :param calibration_factor: Multiplier converting digital units to Pascals.
    :param dbfs: If True, reference digital full scale instead of 20 uPa.
    :return: Scalar for 1D input, array of shape (channels,) for 2D input.
    """
    x_proc = _typesignal(x)
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
