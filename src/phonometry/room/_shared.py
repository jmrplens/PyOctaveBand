#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What the ISO 3382 impulse-response modules share before they measure.

Every ISO 3382-1:2009 measure starts from the same three steps: accept
the response and its calibration, find where the direct sound begins
(A.3.4) and split the response into bands (5.1 asks for at least the
octave bands from 125 Hz to 4 kHz). Doing them once here keeps the time
zero and the band axis of a decay time, a clarity index and a sound
strength measured from the same response identical, which is what lets
them be printed as rows of one table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._internal.utils import _typesignal
from .._internal.validation import require_finite_array
from ..filters.core import OctaveFilterBank
from ..io._resolve import apply_calibration

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..io._signal import Signal

#: Onset threshold: the direct sound starts where the squared IR first
#: rises to within 20 dB of its maximum (trigger point per ISO 3382-1:2009,
#: A.3.4, which places it where the signal first rises significantly above
#: the background but is more than 20 dB below the maximum; this detector
#: takes the first sample at/above the -20 dB edge, one sample inside that
#: bound, which errs early - the safe direction, see :func:`onset_index`).
ONSET_DB = 20.0

#: Fraction of the (onset-trimmed) response used to estimate the
#: background-noise level from its tail.
NOISE_TAIL_FRACTION = 0.1


def validate_ir(
    ir: Signal | list[float] | NDArray[np.float64], fs: int
) -> NDArray[np.float64]:
    """Accept a one-dimensional, non-silent impulse response, calibrated.

    :param ir: The impulse response, as an array or a
        :class:`phonometry.io.Signal` whose calibration is applied.
    :param fs: Sample rate in Hz.
    :return: The samples as a plain 1D float array.
    :raises ValueError: If the response is not one-dimensional, is silent,
        or ``fs`` is not positive.
    """
    x = _typesignal(ir, name="ir")
    if x.ndim != 1:
        msg = "The impulse response must be one-dimensional."
        raise ValueError(msg)
    if fs <= 0:
        msg = "Sample rate 'fs' must be positive."
        raise ValueError(msg)
    if not np.any(x):
        msg = "Impulse response 'ir' is silent."
        raise ValueError(msg)
    return apply_calibration(ir, x)


def onset_index(p2: NDArray[np.float64]) -> int:
    """Index where the direct sound starts: first sample of the squared
    IR within :data:`ONSET_DB` of its maximum (trigger per ISO 3382-1, A.3.4).

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

    :param p2: The squared impulse response.
    :return: Index of the first sample of the direct sound.
    """
    peak = int(np.argmax(p2))
    threshold = p2[peak] * 10.0 ** (-ONSET_DB / 10.0)
    above = np.nonzero(p2[: peak + 1] >= threshold)[0]
    return int(above[0]) if above.size else peak


def noise_power(p2: NDArray[np.float64]) -> float:
    """Background-noise power estimated from the tail of the squared IR.

    :param p2: The squared impulse response.
    :return: Mean power of its last :data:`NOISE_TAIL_FRACTION`.
    """
    tail = max(1, round(p2.size * NOISE_TAIL_FRACTION))
    return float(np.mean(p2[-tail:]))


def split_bands(
    x: NDArray[np.float64],
    fs: int,
    limits: tuple[float, float] | None,
    fraction: int,
    *,
    zero_phase: bool = False,
    name: str = "limits",
) -> tuple[NDArray[np.float64] | None, list[NDArray[np.float64]]]:
    """Split a response into IEC 61260 fractional-octave bands.

    :param x: The response, as a plain 1D array of samples.
    :param fs: Sample rate in Hz.
    :param limits: ``(f_min, f_max)`` band-centre limits in Hz, or ``None``
        to leave the response broadband.
    :param fraction: Bandwidth fraction (1 = octave, 3 = one-third octave).
    :param zero_phase: Filter forward and backward, removing the filter's
        group delay.
    :param name: Name of the caller's own parameter, for the error messages.
    :return: ``(frequency, bands)``, the exact band centre frequencies in Hz
        and one array of samples per band. ``frequency`` is ``None`` and
        ``bands`` holds the untouched response when ``limits`` is ``None``.
    :raises ValueError: If ``limits`` is not a pair, is not finite, or leaves
        no band below the Nyquist frequency.
    """
    if limits is None:
        return None, [x]
    if len(limits) != 2:  # noqa: PLR2004
        msg = f"'{name}' must be a (f_min, f_max) pair or None."
        raise ValueError(msg)
    require_finite_array(limits, name)
    bank = OctaveFilterBank(
        fs=fs, fraction=fraction, order=6, limits=[limits[0], limits[1]]
    )
    _, freqs, bands = bank.filter(
        x,
        sigbands=True,
        detrend=False,
        calculate_level=False,
        zero_phase=zero_phase,
    )
    # np.asarray, not a cast: a bank on the default calibration hands back
    # Signals, and every caller does array arithmetic on what comes out.
    band_signals = [np.asarray(band, dtype=np.float64) for band in bands]
    if not band_signals:
        msg = (
            f"'{name}' {tuple(limits)} leaves no band below the Nyquist "
            f"frequency at fs={fs} Hz."
        )
        raise ValueError(msg)
    return np.asarray(freqs, dtype=np.float64), band_signals


#: Moving-average window (seconds) used to smooth the squared IR before
#: fitting the sloping line of ISO 3382-1:2009, 5.3.3, Equation (3).
SMOOTH_SECONDS = 0.010

#: The decay curve is only trusted down to noise floor + 10 dB.
TRUST_MARGIN_DB = 10.0

#: Minimum number of level samples for the degree-1 least-squares line fits.
MIN_LINE_FIT_POINTS = 2

#: Threshold on the fitted decay slope in dB/s: a slope at or shallower than
#: this (implying a T60 of ~6e8 s, physically meaningless) is treated as no
#: decay, protecting the decay constant alpha from underflowing and the tail
#: terms p2_t1/alpha and 1/alpha**2 from overflowing to inf.
NO_DECAY_SLOPE_DB_PER_S = -1e-7


def truncation(
    p2: NDArray[np.float64], fs: int, noise: float
) -> tuple[int, float, float]:
    r"""Truncation point and tail compensation (ISO 3382-1, 5.3.3, Eq. (3)).

    Fits a sloping line to the smoothed squared IR (in dB) between 5 dB
    below its peak and 10 dB above the noise level; the integration stops
    at the crossing ``t1`` of that line with the noise level, and the
    missing tail is compensated assuming an exponential decay with the
    fitted rate.

    :param p2: Squared impulse response, onset-trimmed.
    :param fs: Sample rate in Hz.
    :param noise: Background-noise power (same units as ``p2``).
    :return: ``(i1, tail_energy, tail_first_moment)`` where ``i1`` is the
        truncation sample, ``tail_energy`` approximates
        :math:`\int_{t_1}^{\infty} p^2 \, dt` and ``tail_first_moment``
        approximates :math:`\int_{t_1}^{\infty} t \, p^2 \, dt` (both in
        seconds units, i.e. energy = sum(p2)/fs).
    """
    n = p2.size
    no_truncation = (n, 0.0, 0.0)
    if noise <= 0.0:
        return no_truncation
    window = min(max(1, round(SMOOTH_SECONDS * fs)), n)
    cumulative = np.concatenate(([0.0], np.cumsum(p2)))
    smoothed = (cumulative[window:] - cumulative[:-window]) / window
    t_smooth = (np.arange(smoothed.size) + 0.5 * window) / fs
    tiny = np.finfo(np.float64).tiny
    level = 10.0 * np.log10(np.maximum(smoothed, tiny))
    noise_db = 10.0 * np.log10(noise)
    mask = (level <= level.max() - 5.0) & (level >= noise_db + TRUST_MARGIN_DB)
    if int(mask.sum()) < MIN_LINE_FIT_POINTS:
        return no_truncation
    slope, intercept = np.polyfit(t_smooth[mask], level[mask], 1)
    # A non-negative slope means no decay; a barely-negative slope (e.g.
    # -1e-16 dB/s from fitting near-constant noise) would make the decay
    # constant alpha underflow toward 0 and the tail terms p2_t1/alpha and
    # 1/alpha**2 overflow to inf. A slope of -1e-7 dB/s implies a T60 of
    # ~6e8 s, which is physically meaningless, so treat anything shallower
    # as no decay.
    if slope >= NO_DECAY_SLOPE_DB_PER_S:
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
