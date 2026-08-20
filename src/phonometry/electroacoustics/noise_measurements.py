#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Noise measurements of audio equipment (AES17-2015 6.4).

The two levels that say how quiet a device is, both measured through the
AES17 measurement chain -- the standard notch (5.2.8) where a test tone has
to be removed, the CCIR-RMS weighting (5.2.7, the ITU-R BS.468-4 curve with
the flat -5,63 dB gain that puts unity at 2 kHz) and the 20 Hz to 20 kHz
band (5.2.5):

* **Dynamic range** (6.4.1), the ratio of the maximum output level to the
  weighted noise-plus-distortion residual left when the device is driven
  with a 997 Hz sine 60 dB below full scale -- what the standard's own note
  also calls the signal-to-noise ratio, since it includes every harmonic,
  inharmonic and noise component.
* **Idle channel noise** (6.4.2), the weighted output level, relative to
  full scale, of a device driven with no signal at all.

They share the notch, the weighting and the band with the distortion
metrics of :mod:`phonometry.electroacoustics.distortion`, and differ from
them in what is being asked: not how much of the output is not the tone,
but how much output there is when there should be none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..io._resolve import resolve_fs
from ..io._signal import Signal
from .distortion import (
    _AES17_BANDWIDTH_HZ,
    _AES17_HIGHPASS_HZ,
    _DEFAULT_NOTCH_Q,
    _amplitude_spectrum,
    _fundamental_frequency,
    _notched_residual,
    _positive,
    _steady_slice,
    _validate_notch_q,
    _validate_signal,
    itu_r_468_weighting,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: AES17-2015 5.2.7: the CCIR-RMS weighting is the ITU-R BS.468-4 curve with an
#: additional flat gain of -5,63 dB, which places unity gain at 2 kHz. Table 1
#: tabulates the result (for example -5,6 dB at 1 kHz, 0 dB at 2 kHz).
_CCIR_RMS_OFFSET_DB = -5.63

#: AES17-2015 6.4.1 dynamic-range test level: a 997 Hz sine 60 dB below the
#: maximum input level.
_AES17_DYNAMIC_RANGE_LEVEL_DBFS = -60.0

#: AES17-2015 6.4.1 dynamic-range test frequency, in hertz.
_AES17_DYNAMIC_RANGE_FREQ_HZ = 997.0


def _ccir_rms_weighted_rms(
    x: NDArray[np.float64], fs: float, bandwidth: float | None
) -> float:
    """RMS of ``x`` through the AES17 CCIR-RMS weighting (5.2.7), band-limited.

    The ITU-R BS.468-4 curve plus the flat ``-5,63`` dB CCIR-RMS gain, applied
    in the frequency domain via Parseval and restricted to the AES17
    measurement band (20 Hz high-pass plus the standard low-pass at
    ``bandwidth``; ``None`` keeps the full Nyquist band).
    """
    n = x.size
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    gains = np.zeros(freqs.size)
    nonzero = freqs > 0.0
    gains[nonzero] = 10.0 ** (
        (itu_r_468_weighting(freqs[nonzero]) + _CCIR_RMS_OFFSET_DB) / 20.0
    )
    f_hi = (
        fs / 2.0
        if bandwidth is None
        else min(_positive(bandwidth, "bandwidth"), fs / 2.0)
    )
    out_of_band = (freqs < _AES17_HIGHPASS_HZ) | (freqs > f_hi)
    gains[out_of_band] = 0.0
    weights = np.full(freqs.size, 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    mean_square = float(np.sum(weights * (gains * np.abs(spec)) ** 2)) / n**2
    return float(np.sqrt(mean_square))


def _full_scale_rms(full_scale: float) -> float:
    """RMS of a full-scale sinusoid of peak amplitude ``full_scale`` (0 dBFS).

    AES17-2015 defines 0 dBFS as the RMS of a full-scale sine wave (3.13), so
    the maximum output level (6.2.6) referenced by the noise measurements is
    ``full_scale / sqrt(2)``.
    """
    return _positive(full_scale, "full_scale") / float(np.sqrt(2.0))


def dynamic_range(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    notch_q: float = _DEFAULT_NOTCH_Q,
    bandwidth: float | None = _AES17_BANDWIDTH_HZ,
    full_scale: float = 1.0,
    window: str = "hann",
) -> float:
    r"""Dynamic range of an audio device (AES17-2015 6.4.1), in dB CCIR-RMS.

    The device is driven with a 997 Hz sine 60 dB below full scale; the
    captured output has its fundamental removed by the standard notch filter
    (5.2.8), and the residual noise-plus-distortion is weighted by the
    CCIR-RMS filter (5.2.7) over the AES17 measurement band. The dynamic range
    is the ratio of the maximum output level (a full-scale sine, 6.2.6) to
    that weighted residual level:

    .. math::

       \mathrm{DR} = 20 \log_{10}\left(
       (\text{full\_scale} / \sqrt{2}) / V_\text{residual,CCIR-RMS}
       \right)

    It includes all harmonic, inharmonic and noise components and is also
    known as the signal-to-noise ratio (6.4.1 note).

    :param signal: Captured output of the device under test (1-D), scaled so
        that ``full_scale`` is the digital full-scale peak amplitude.
        Accepts a :class:`phonometry.io.Signal` for the rate, but a
        calibration factor it carries is deliberately **not** applied: this
        quantity is referenced to digital full scale, not to 20 uPa, so
        scaling the samples to pascals would move the reading by
        ``20 lg(factor)`` under a full-scale name.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param fundamental: Test frequency, in Hz; ``None`` uses the 997 Hz of the
        standard.
    :param notch_q: Effective notch quality factor (AES17 5.2.8: 1.2..3;
        default 2.0).
    :param bandwidth: AES17 measurement bandwidth, in Hz (default 20 kHz;
        ``None`` measures the full Nyquist band).
    :param full_scale: Digital full-scale peak amplitude (default 1.0).
    :param window: FFT window used only for fundamental auto-detection.
    :return: The dynamic range, in dB CCIR-RMS (a positive number).
    :raises ValueError: If the inputs are invalid or ``notch_q`` is out of
        range.
    """
    fs = resolve_fs(signal, fs, name="signal")
    sig = _validate_signal(signal, calibrate=False)
    fs_v = _positive(fs, "fs")
    _validate_notch_q(notch_q)
    reference = _full_scale_rms(full_scale)
    if fundamental is None:
        freqs, amp = _amplitude_spectrum(sig, fs_v, window)
        f0 = _fundamental_frequency(freqs, amp)
    else:
        f0 = _positive(fundamental, "fundamental")
    residual = _notched_residual(sig, fs_v, f0, float(notch_q))
    sl = _steady_slice(sig.size, fs_v, f0, float(notch_q))
    weighted = _ccir_rms_weighted_rms(np.asarray(residual[sl]), fs_v, bandwidth)
    if weighted <= 0.0:
        raise ValueError("Residual has no energy; cannot form a dynamic range.")
    return float(20.0 * np.log10(reference / weighted))


def idle_channel_noise(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    bandwidth: float | None = _AES17_BANDWIDTH_HZ,
    full_scale: float = 1.0,
) -> float:
    r"""Idle channel noise level (AES17-2015 6.4.2), in dBFS CCIR-RMS.

    The weighted output of the device when driven with no signal (a
    short-circuited analogue input or digital zero at the input). The captured
    idle output is weighted by the CCIR-RMS filter (5.2.7) over the AES17
    measurement band and reported relative to full scale:

    .. math::

       L_\text{idle} = 20 \log_{10}\left(
       V_\text{idle,CCIR-RMS} / (\text{full\_scale} / \sqrt{2})
       \right)

    :param signal: Captured idle output of the device under test (1-D), scaled
        so that ``full_scale`` is the digital full-scale peak amplitude.
        Accepts a :class:`phonometry.io.Signal` for the rate, but a
        calibration factor it carries is deliberately **not** applied: this
        quantity is referenced to digital full scale, not to 20 uPa, so
        scaling the samples to pascals would move the reading by
        ``20 lg(factor)`` under a full-scale name.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param bandwidth: AES17 measurement bandwidth, in Hz (default 20 kHz;
        ``None`` measures the full Nyquist band).
    :param full_scale: Digital full-scale peak amplitude (default 1.0).
    :return: The idle channel noise level, in dBFS CCIR-RMS (a negative
        number for any real device).
    :raises ValueError: If the inputs are invalid.
    """
    fs = resolve_fs(signal, fs, name="signal")
    sig = _validate_signal(signal, calibrate=False)
    fs_v = _positive(fs, "fs")
    reference = _full_scale_rms(full_scale)
    weighted = _ccir_rms_weighted_rms(sig, fs_v, bandwidth)
    if weighted <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(weighted / reference))
