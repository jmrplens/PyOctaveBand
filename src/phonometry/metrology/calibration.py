#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Calibration utilities for mapping digital signals to physical SPL levels.
"""

from __future__ import annotations

import warnings

import numpy as np

from .._internal.warnings import PhonometryWarning, _warn_renamed


class CalibrationWarning(PhonometryWarning):
    """The calibration reference recording looks unreliable."""


# IEC 60942:2017 Table 2 (p. 16): short-term level fluctuation acceptance
# limits in dB by range of nominal frequencies, class 1 column. Classes LS
# and 2 are only specified for 160 Hz to 1250 Hz (0,03 / 0,15 dB); the other
# rows carry "-" for them, so this table exposes the class 1 limits only.
def _class1_fluctuation_limit(frequency: float) -> float:
    """Class 1 short-term fluctuation limit for a nominal calibrator frequency.

    IEC 60942:2017 Table 2 rows: 31.5-63 Hz -> 0.20 dB; > 63 to < 160 Hz ->
    0.10 dB; 160 Hz and above -> 0.07 dB. Frequencies outside the specified
    31,5 Hz to 16 kHz span fall back to the strictest limit (0,07 dB).
    """
    if 31.5 <= frequency <= 63.0:
        return 0.20
    if 63.0 < frequency < 160.0:
        return 0.10
    return 0.07


def sensitivity(
    ref_signal: list[float] | np.ndarray,
    target_spl: float = 94.0,
    ref_pressure: float = 2e-5,
    fs: int | None = None,
    validate: bool = True,
    max_fluctuation_db: float | None = None,
    frequency: float = 1000.0,
    narrowband: bool = False,
) -> float:
    r"""
    Calculate the calibration factor (multiplier) to convert digital units
    to Pascals based on a reference recording (e.g., 1kHz @ 94dB).

    When ``fs`` is provided (and ``validate`` is True), the recording's
    stability is checked the way IEC 60942:2017 specifies for the calibrator
    itself (5.3.3): levels are measured with time-weighting F and the
    *short-term level fluctuation* (the absolute difference between each of
    the maximum and minimum levels and the mean level) must not exceed the
    Table 2 acceptance limit for the calibrator class (class 1: 0.07 dB at
    and above 160 Hz, relaxed to 0.10 dB above 63 Hz and below 160 Hz, and to
    0.20 dB for the 31.5-63 Hz rows where the F time-weighting itself ripples;
    below Table 2's 31.5 Hz span the strict 0.07 dB applies). A larger
    fluctuation usually means a badly coupled microphone or handling noise in
    the recording, which would silently corrupt every calibrated level; a
    :class:`CalibrationWarning` is emitted.

    .. note:: IEC 60942 certifies calibrators over 60 s of operation sampled
       at least 30 times; this check applies the same criterion to whatever
       settled portion of the recording is available (>= 1 s after the 1 s
       integrator attack).

    :param ref_signal: Recording of the calibration tone.
    :param target_spl: The known SPL level of the calibrator (default 94 dB).
    :param ref_pressure: Reference pressure (default 20 microPascals).
    :param fs: Sample rate of the recording in Hz. Required for the
        stability validation; without it the check is skipped.
    :param validate: If True (default) and ``fs`` is given, warn when the
        recording's short-term level fluctuation exceeds the limit.
    :param max_fluctuation_db: Explicit fluctuation limit in dB. Default
        (None) resolves the IEC 60942:2017 Table 2 class 1 limit for
        ``frequency``.
    :param frequency: Nominal frequency of the calibration tone in Hz
        (default 1000.0), used to select the Table 2 row.
    :param narrowband: If True (requires ``fs``), estimate the tone level with
        a coherent single-frequency (Goertzel) detector locked to the tone
        near ``frequency`` instead of the full-band RMS. This rejects
        broadband hum/noise in the reference take, which otherwise inflates
        the RMS and shrinks the factor by
        :math:`-10 \lg(1 + 1/\mathrm{SNR})` (about
        -0.44 dB at 20 dB SNR), silently biasing every subsequent level.
        The default (False) keeps the exact legacy broadband-RMS behaviour;
        enable it for noisy coupler recordings.
    :return: Calibration factor (sensitivity multiplier).
    """
    signal_arr = np.asarray(ref_signal, dtype=np.float64)
    if signal_arr.size == 0:
        raise ValueError("Reference signal is empty, cannot calibrate.")
    if not np.all(np.isfinite(signal_arr)):
        raise ValueError(
            "Reference signal contains non-finite samples (NaN/inf); they "
            "would silently corrupt the calibration factor."
        )
    if narrowband:
        if fs is None:
            raise ValueError("narrowband tone estimation requires 'fs'.")
        rms_ref = _narrowband_tone_rms(signal_arr, fs, frequency)
    else:
        rms_ref = float(np.sqrt(np.mean(signal_arr ** 2)))
    if rms_ref == 0:
        raise ValueError("Reference signal is silent, cannot calibrate.")

    if validate and fs is not None:
        limit = (
            max_fluctuation_db
            if max_fluctuation_db is not None
            else _class1_fluctuation_limit(frequency)
        )
        _validate_reference_stability(signal_arr, fs, limit)

    factor = (ref_pressure * 10 ** (target_spl / 20)) / rms_ref
    return float(factor)


def calculate_sensitivity(
    ref_signal: list[float] | np.ndarray,
    target_spl: float = 94.0,
    ref_pressure: float = 2e-5,
    fs: int | None = None,
    validate: bool = True,
    max_fluctuation_db: float | None = None,
    frequency: float = 1000.0,
    narrowband: bool = False,
) -> float:
    """Deprecated alias of :func:`sensitivity`."""
    _warn_renamed("calculate_sensitivity()", "sensitivity()")
    return sensitivity(
        ref_signal,
        target_spl=target_spl,
        ref_pressure=ref_pressure,
        fs=fs,
        validate=validate,
        max_fluctuation_db=max_fluctuation_db,
        frequency=frequency,
        narrowband=narrowband,
    )


def _validate_reference_stability(
    signal_arr: np.ndarray, fs: int, max_fluctuation_db: float
) -> None:
    """Warn if the F-weighted level of the recording fluctuates too much."""
    from .parametric_filters import time_weighting

    # The integrator attack lasts ~8*tau (1 s for F); we need at least
    # another second of settled envelope to assess the fluctuation.
    if signal_arr.shape[-1] < 2 * fs:
        warnings.warn(
            "Calibration tone is shorter than 2 s: too short to validate its "
            "stability (IEC 60942 measures the generated level over 60 s). "
            "Record a longer, steady tone.",
            CalibrationWarning,
            stacklevel=3,
        )
        return
    envelope = time_weighting(signal_arr, fs, mode="fast")
    skip = int(1.0 * fs)
    steady = np.maximum(envelope[..., skip:], np.finfo(float).eps)
    levels_db = 10 * np.log10(steady)
    # IEC 60942:2017 5.3.3: |max - mean| and |min - mean| shall each not
    # exceed the limit. Computed per channel: channels may sit at different
    # (individually stable) levels, which must not read as fluctuation.
    mean = np.mean(levels_db, axis=-1, keepdims=True)
    deviation = np.max(np.abs(levels_db - mean), axis=-1)
    fluctuation = float(np.max(deviation))
    if fluctuation > max_fluctuation_db:
        warnings.warn(
            f"Calibration tone level fluctuation is {fluctuation:.2f} dB "
            f"(limit {max_fluctuation_db:.2f} dB per IEC 60942:2017 Table 2). "
            "Check the microphone coupling and trim handling noise before "
            "trusting the resulting sensitivity.",
            CalibrationWarning,
            stacklevel=3,
        )


def _narrowband_tone_rms(
    signal_arr: np.ndarray, fs: int, frequency: float
) -> float:
    r"""RMS amplitude of the calibration tone via coherent detection.

    A Hann-windowed single-frequency (Goertzel) detector locked to the tone
    near ``frequency`` recovers the tone RMS while rejecting broadband noise,
    which averages toward zero as :math:`1/\sqrt{N}` because it is
    incoherent with
    the detector. Multichannel input is reduced to the same flattened power
    the broadband path uses. The nominal frequency is refined to the local
    spectral peak (parabolic interpolation) so calibrator frequency tolerance
    does not bias the estimate.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim > 1:
        per_channel = [_narrowband_tone_rms(row, fs, frequency) for row in x]
        return float(np.sqrt(np.mean(np.square(per_channel))))
    n = x.size
    if n < 4:
        # Too short for a coherent estimate; fall back to broadband RMS.
        return float(np.sqrt(np.mean(x ** 2)))
    window = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    df = float(fs) / n
    magnitude = np.abs(np.fft.rfft(x * window))
    # Search a +/-10 % window around the nominal calibrator frequency so a
    # strong low-frequency hum cannot capture the peak estimate.
    in_range = (freqs >= 0.9 * frequency) & (freqs <= 1.1 * frequency)
    candidates = np.flatnonzero(in_range)
    if candidates.size:
        peak = int(candidates[int(np.argmax(magnitude[candidates]))])
    else:
        peak = int(np.argmax(magnitude[1:]) + 1) if magnitude.size > 1 else 0
    f0 = freqs[peak]
    if 1 <= peak < magnitude.size - 1:
        a, b, c = magnitude[peak - 1], magnitude[peak], magnitude[peak + 1]
        denom = a - 2.0 * b + c
        # Skip parabolic refinement on a degenerate flat-top peak. Guard the
        # denominator against being negligible relative to the bin magnitudes
        # involved rather than testing exact float equality (which the
        # SonarCloud quality gate flags): a genuine tone gives denom of order
        # the peak magnitude, so normal-tone behaviour is unchanged.
        if abs(denom) > 1e-12 * max(a, b, c, np.finfo(np.float64).tiny):
            delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
            f0 = (peak + delta) * df
    idx = np.arange(n)
    demod = np.exp(-2j * np.pi * f0 * idx / fs)
    amplitude = np.sum(x * window * demod) / np.sum(window)
    return float(np.sqrt(2.0) * np.abs(amplitude))
