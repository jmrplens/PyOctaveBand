#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Inter-sample peak recovery by polyphase oversampling (private).

The true peak of a band-limited continuous waveform generally falls between
samples, so a raw on-grid maximum under-reads it (worst for sustained tones
near integer submultiples of the sample rate). Both the C-weighted peak
level (:func:`phonometry.metrology.levels.lc_peak`, IEC 61672-1) and the
true-peak programme level (:func:`phonometry.broadcast.true_peak_level`,
ITU-R BS.1770-5 Annex 2) recover the inter-sample peak the same way:
polyphase-oversample the signal, then take the absolute maximum. This module
is the single home of that machinery.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# Above this many samples per channel the oversampled waveform is scanned in
# bounded chunks instead of being materialized whole (an hour of stereo audio
# at 48 kHz oversampled 4x would otherwise allocate several gigabytes). Each
# chunk is interpolated with an overlap margin much longer than the Kaiser
# FIR of resample_poly (half-length 10*factor input samples), and only the
# interior of each chunk contributes to the maximum, so the chunked scan is
# exactly the full computation evaluated piecewise.
_CHUNK_SAMPLES = 1 << 20
_CHUNK_MARGIN = 64


def _chunked_peak(x: np.ndarray, factor: int) -> np.ndarray:
    """Peak of the oversampled signal, scanned in bounded chunks."""
    n = x.shape[-1]
    peak = np.zeros(x.shape[:-1], dtype=np.float64)
    for start in range(0, n, _CHUNK_SAMPLES):
        stop = min(start + _CHUNK_SAMPLES, n)
        lo = max(0, start - _CHUNK_MARGIN)
        hi = min(n, stop + _CHUNK_MARGIN)
        up = signal.resample_poly(x[..., lo:hi], factor, 1, axis=-1)
        # Keep only the interior samples that belong to [start, stop): the
        # margins exist purely to feed the FIR its full support.
        i0 = (start - lo) * factor
        i1 = i0 + (stop - start) * factor
        peak = np.maximum(peak, np.max(np.abs(up[..., i0:i1]), axis=-1))
    return np.asarray(peak)


def inter_sample_peak(x: np.ndarray, factor: int) -> np.ndarray:
    """Absolute peak along the last axis after ``factor``-times oversampling.

    Zero-stuffs and low-pass interpolates via
    :func:`scipy.signal.resample_poly` (a Kaiser-windowed FIR polyphase
    interpolator, the "similar or superior" filter that ITU-R BS.1770-5
    Annex 2 admits in place of its example 48-tap filter), then returns the
    absolute maximum. ``factor=1`` skips the oversampling and reduces to the
    on-grid sample peak. Long signals are scanned in bounded chunks so the
    memory use stays independent of the programme length.

    :param x: Input signal, 1D or 2D ``[channels, samples]``.
    :param factor: Integer oversampling factor (>= 1).
    :return: The absolute peak per channel, in the input's linear units
        (0-d array for 1D input).
    """
    if factor > 1 and x.shape[-1] > 0:
        if x.shape[-1] > _CHUNK_SAMPLES:
            return _chunked_peak(x, factor)
        # Recover inter-sample peaks: the on-grid maximum misses the true
        # continuous peak between samples for sustained HF tones.
        x = signal.resample_poly(x, factor, 1, axis=-1)
    return np.asarray(np.max(np.abs(x), axis=-1))
