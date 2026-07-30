#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The ERB_N scale: auditory-filter bandwidth and the Cam frequency scale.

The cochlea behaves as a bank of overlapping band-pass **auditory filters**.
The width of the filter centred on a given frequency is summarised by its
**equivalent rectangular bandwidth** ``ERB_N``: the bandwidth of the
rectangular filter that passes the same power and has the same peak response.
Glasberg and Moore (1990), fitting notched-noise data for young listeners at
moderate levels, give it as a straight line in frequency (Moore, *An
Introduction to the Psychology of Hearing* 6th ed., p. 76):

    ERB_N = 24.7 (4.37 F + 1)   Hz,  with F in kHz

Integrating ``df / ERB_N(f)`` turns that into a frequency scale whose unit is
one auditory-filter width, the **ERB_N number**, whose unit is called the
**Cam** (after Cambridge, following a suggestion by Hartmann):

    ERB_N number = 21.4 log10(4.37 F + 1)   Cam

The Cam scale plays the same role as the Bark scale of Zwicker and Terhardt or
the mel scale of pitch: it is the axis on which masking patterns, excitation
patterns and specific loudness are naturally expressed. It differs from Bark
numerically, mostly below 500 Hz where the "old" critical-band function
flattens out but direct measurements show ``ERB_N`` continuing to shrink.

This module states the constants to the precision used by the ISO 532-2
implementation of the Moore-Glasberg loudness model
(:mod:`phonometry.psychoacoustics.loudness_moore_glasberg`), which shares
them:

    ERB_N = 24.673 (0.004368 f + 1)   Hz,  with f in Hz
    ERB_N number = 21.366 log10(0.004368 f + 1)   Cam

They are the same fit written with one more digit: the two forms agree to
better than 0.2 % over the whole audible range. The extra digits matter for
the round trip through :func:`frequency_from_cam`, and they are what reproduce
the check value Moore prints on p. 77, that 1000 Hz corresponds to 15.59 Cam
(the two-significant-digit constants give 15.62).

Reference: Glasberg, B. R. and Moore, B. C. J. (1990), "Derivation of auditory
filter shapes from notched-noise data", *Hearing Research* 47, 103-138.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .._internal.types import as_float_or_array

#: Slope-and-offset constant of ``ERB_N = C1 (C2 f + 1)``, Hz.
ERB_C1 = 24.673

#: Frequency coefficient of ``ERB_N = C1 (C2 f + 1)``, 1/Hz.
ERB_C2 = 0.004368

#: Constant of the Cam scale ``i = C_CAM log10(C2 f + 1)``, Cam.
CAM_C = 21.366


def erb_bandwidth(frequency: ArrayLike) -> np.ndarray | float:
    """Equivalent rectangular bandwidth of the auditory filter, Hz.

    ``ERB_N = 24.673 (0.004368 f + 1)`` (Glasberg and Moore 1990; Moore 6th
    ed., p. 76, printed there with the rounded constants
    ``24.7 (4.37 F + 1)`` for ``F`` in kHz).

    :param frequency: Filter centre frequency ``f``, Hz (scalar or array,
        non-negative).
    :return: The bandwidth ``ERB_N``, Hz; a float for a scalar input.
    """
    f = np.asarray(frequency, dtype=np.float64)
    if not np.all(np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError("'frequency' must be non-negative and finite.")
    return as_float_or_array(ERB_C1 * (ERB_C2 * f + 1.0))


def cam_from_frequency(frequency: ArrayLike) -> np.ndarray | float:
    """ERB_N number (Cam) of a frequency.

    ``i = 21.366 log10(0.004368 f + 1)`` (Glasberg and Moore 1990; Moore 6th
    ed., p. 76, printed there as ``21.4 log10(4.37 F + 1)`` for ``F`` in kHz).
    One Cam is one auditory-filter width.

    :param frequency: Frequency ``f``, Hz (scalar or array, non-negative).
    :return: The ERB_N number, Cam; a float for a scalar input.
    """
    f = np.asarray(frequency, dtype=np.float64)
    if not np.all(np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError("'frequency' must be non-negative and finite.")
    return as_float_or_array(CAM_C * np.log10(ERB_C2 * f + 1.0))


def frequency_from_cam(cam: ArrayLike) -> np.ndarray | float:
    """Frequency of an ERB_N number, the inverse of :func:`cam_from_frequency`.

    ``f = (10^(i / 21.366) - 1) / 0.004368``.

    :param cam: ERB_N number ``i``, Cam (scalar or array, non-negative).
    :return: The frequency ``f``, Hz; a float for a scalar input.
    """
    i = np.asarray(cam, dtype=np.float64)
    if not np.all(np.isfinite(i)) or np.any(i < 0.0):
        raise ValueError("'cam' must be non-negative and finite.")
    return as_float_or_array((np.power(10.0, i / CAM_C) - 1.0) / ERB_C2)
