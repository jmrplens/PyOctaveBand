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
