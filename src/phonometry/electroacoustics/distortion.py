#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Harmonic distortion of electroacoustic equipment (IEC 60268-3 / AES17).

Single-tone distortion metrics of amplifiers and audio equipment, from a
captured signal:

* **Total harmonic distortion** ``THD`` (IEC 60268-3 14.12.2-3), relative to
  the fundamental (``kind='F'``, the widespread convention) or to the total
  RMS (``kind='R'``, the 14.12.3.2 quantity), and the **nth-order harmonic
  distortion** (14.12.5).
* **THD+N** and the derived **SINAD** (AES17-2015 6.3.1): the fundamental is
  removed with the standard notch filter and the residual is compared with
  the total signal, both through the AES17 measurement bandwidth (20 Hz to
  20 kHz by default).
* **Weighted THD** (14.12.11), the harmonic residual weighted by the
  IEC 60268-1 / ITU-R BS.468-4 network (A/C optional), whose nominal
  response :func:`itu_r_468_weighting` exposes on its own.

All metrics have an exact analytic oracle: a signal synthesised with known
harmonic amplitudes reproduces the closed-form ratio. The functions assume
the tones fall on (or very near) FFT bins -- use coherent sampling (an
integer number of periods) or supply a low-leakage window, as audio
analysers do.

The two-tone metrics of IEC 60268-3 14.12.7-10 live in
:mod:`phonometry.electroacoustics.intermodulation`, and the AES17-2015 6.4
noise levels in :mod:`phonometry.electroacoustics.noise_measurements`; both
are built on the spectrum, notch and weighting helpers of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..io._resolve import apply_calibration, resolve_fs
from ..io._signal import Signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

#: Default number of harmonics summed for the THD (IEC 60268-3).
_DEFAULT_N_HARMONICS = 10
#: Default standard-notch quality factor for THD+N (AES17 5.2.8: 1.2 <= Q <= 3).
_DEFAULT_NOTCH_Q = 2.0
#: AES17 5.2.8 defines Q on the *applied* (combined) response. ``filtfilt``
#: applies the notch twice, so the single-pass design must be sharper for the
#: squared magnitude to have the requested -3 dB width. For the biquad notch
#: ``|H|^2 = x^2 / (x^2 + Q^-2)`` with ``x = (f0^2 - f^2)/(f0 f)``, the -3 dB
#: width of ``|H|^4`` is ``f0 sqrt(1 + sqrt(2)) / Q_design``, so designing at
#: ``Q_design = Q sqrt(1 + sqrt(2))`` makes the effective Q equal the request
#: (without this the applied Q is 0.644x nominal: a requested 2.0 acted as
#: 1.29, and requests below ~1.87 silently fell outside the AES17 range).
#: The relation is exact for the analogue prototype; the bilinear warp
#: degrades it near Nyquist (about +5 % at f0 = fs/4.8), which stays inside
#: the AES17 [1.2, 3] window for the default Q.
_FILTFILT_NOTCH_Q_FACTOR = float(np.sqrt(1.0 + np.sqrt(2.0)))
#: AES17 measurement-bandwidth chain (5.2.5 / 6.3.1): the analyzer passband is
#: 20 Hz to the upper band-edge frequency (default 20 kHz).
_AES17_HIGHPASS_HZ = 20.0
_AES17_BANDWIDTH_HZ = 20000.0
#: Notch settling allowance, in fundamental cycles per unit Q, discarded from
#: each end of the notched residual so the ``filtfilt`` start/stop transient does
#: not leak fundamental energy into the THD+N residual.
_NOTCH_SETTLE_CYCLES = 8.0
#: Harmonic peak-search half-width, as a fraction of the fundamental. Wide
#: enough to catch a harmonic under window leakage or a slightly mistuned
#: fundamental, but well inside ±f0 so a nearby non-harmonic tone is not latched
#: onto a harmonic bin.
_HARMONIC_SEARCH_FACTOR = 0.1


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return scalar


def _validate_notch_q(notch_q: float) -> float:
    """Validate the effective notch quality factor (AES17 5.2.8: 1.2 <= Q <= 3)."""
    q = float(notch_q)
    if not 1.2 <= q <= 3.0:
        raise ValueError("'notch_q' must be within the AES17 range [1.2, 3].")
    return q


#: Minimum accepted capture length. A shorter buffer cannot resolve a
#: fundamental and its harmonics with any useful frequency resolution (a
#: 16-sample capture was previously accepted and produced meaningless ratios).
_MIN_SIGNAL_SAMPLES = 64


def _validate_signal(
    signal: Signal | NDArray[np.float64] | list[float],
    *,
    calibrate: bool = True,
) -> NDArray[np.float64]:
    """Validate the record and, unless exempted, present it in pascals.

    ``calibrate=False`` is for the quantities referenced to digital full
    scale rather than to 20 uPa: see :mod:`phonometry.io._resolve`.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 1:
        raise ValueError("'signal' must be one-dimensional.")
    if sig.size < _MIN_SIGNAL_SAMPLES:
        raise ValueError(
            f"'signal' must contain at least {_MIN_SIGNAL_SAMPLES} samples to "
            "resolve a fundamental and its harmonics."
        )
    if not np.all(np.isfinite(sig)):
        raise ValueError("'signal' must be finite.")
    return apply_calibration(signal, sig) if calibrate else sig


def _amplitude_spectrum(
    signal: NDArray[np.float64], fs: float, window: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coherent-gain-normalised amplitude spectrum (a tone on a bin reads its peak)."""
    from scipy import signal as sp_signal

    n = signal.size
    w = sp_signal.get_window(window, n, fftbins=True)
    spectrum = np.fft.rfft(signal * w)
    amp = np.abs(spectrum) * 2.0 / np.sum(w)
    freqs = np.asarray(np.fft.rfftfreq(n, d=1.0 / fs), dtype=np.float64)
    return freqs, amp


def _tone_amplitude(
    freqs: NDArray[np.float64],
    amp: NDArray[np.float64],
    frequency: float,
    search_hz: float,
) -> float:
    """Peak amplitude within ``±search_hz`` of ``frequency`` (0 if out of range)."""
    lo, hi = frequency - search_hz, frequency + search_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.max(amp[mask]))


def _fundamental_frequency(
    freqs: NDArray[np.float64], amp: NDArray[np.float64]
) -> float:
    """Frequency of the largest non-DC spectral peak."""
    idx = int(np.argmax(amp[1:])) + 1
    return float(freqs[idx])


def _harmonic_amplitudes(
    signal: NDArray[np.float64],
    fs: float,
    fundamental: float | None,
    n_harmonics: int,
    window: str,
) -> tuple[float, NDArray[np.float64]]:
    """Return ``(f0, amplitudes)`` with ``amplitudes[k]`` the (k+1)-th harmonic."""
    freqs, amp = _amplitude_spectrum(signal, fs, window)
    f0 = _fundamental_frequency(freqs, amp) if fundamental is None else float(fundamental)
    if f0 <= 0.0:
        raise ValueError("Could not determine a positive fundamental frequency.")
    search = f0 * _HARMONIC_SEARCH_FACTOR
    nyquist = fs / 2.0
    amps = []
    for k in range(1, n_harmonics + 1):
        fk = k * f0
        if fk >= nyquist:
            break
        amps.append(_tone_amplitude(freqs, amp, fk, search))
    return f0, np.asarray(amps, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Harmonic distortion (IEC 60268-3 14.12.2-3 / 14.12.5, AES17 6.3)
# --------------------------------------------------------------------------- #
def thd(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    kind: Literal["F", "R"] = "F",
    n_harmonics: int = _DEFAULT_N_HARMONICS,
    window: str = "hann",
) -> float:
    r"""Total harmonic distortion (IEC 60268-3 14.12.2-3).

    From the harmonic amplitudes :math:`a_n`:

    .. math::

       \mathrm{THD}_F = \sqrt{\sum_{n \ge 2} a_n^2} / a_1

       \mathrm{THD}_R = \sqrt{\sum_{n \ge 2} a_n^2} /
       \sqrt{\sum_{n \ge 1} a_n^2}

    relative to the fundamental (``kind='F'``) or to the total RMS
    (``kind='R'``), respectively.

    Convention note: the quantity the IEC 60268-3 14.12.3.2 formula defines
    is the R form (harmonic RMS over total RMS). The default ``kind='F'`` is
    the fundamental-referenced convention widespread in audio practice and
    datasheets; the two agree to first order for small distortion.

    :param signal: Captured signal (1-D). Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples and then cancels: this is a ratio of amplitudes drawn from
        the same record, so the factor divides out and the answer is the
        same calibrated or not. Coherent sampling (integer periods) or
        a low-leakage window gives the exact value.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param fundamental: Fundamental frequency :math:`f_1` in Hz, or ``None``
        to take the largest spectral peak.
    :param kind: ``'F'`` (relative to the fundamental, the default) or ``'R'``
        (relative to the total RMS, the 14.12.3.2 quantity).
    :param n_harmonics: Highest harmonic order summed (default 10).
    :param window: FFT window (default ``'hann'``).
    :return: Total harmonic distortion, as a ratio (0..).
    :raises ValueError: If the signal/parameters are invalid, ``kind`` is
        unknown, or no harmonic of the fundamental lies below Nyquist.
    """
    fs = resolve_fs(signal, fs, name="signal")
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    if kind not in ("F", "R"):
        raise ValueError("'kind' must be 'F' or 'R'.")
    _f0, amps = _harmonic_amplitudes(sig, fs_v, fundamental, n_harmonics, window)
    if amps.size == 0 or amps[0] <= 0.0:
        raise ValueError("No fundamental component found in the signal.")
    if amps.size < 2:
        raise ValueError(
            "No harmonic of the fundamental lies below the Nyquist frequency; "
            "the THD is undefined (raise fs or lower the fundamental)."
        )
    harmonic_rms = float(np.sqrt(np.sum(amps[1:] ** 2)))
    if kind == "F":
        return float(harmonic_rms / amps[0])
    total_rms = float(np.sqrt(np.sum(amps**2)))
    return float(harmonic_rms / total_rms)


def harmonic_distortion(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    fundamental: float,
    order: int,
    n_harmonics: int = _DEFAULT_N_HARMONICS,
    window: str = "hann",
) -> float:
    r"""nth-order harmonic distortion :math:`d_n` (IEC 60268-3 14.12.5).

    :math:`d_n = a_n / \sqrt{\sum_{k \ge 1} a_k^2}` -- the nth harmonic
    amplitude relative to the total RMS.

    :param signal: Captured signal (1-D). Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples and then cancels: this is a ratio of amplitudes drawn from
        the same record, so the factor divides out and the answer is the
        same calibrated or not.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param fundamental: Fundamental frequency :math:`f_1`, in Hz.
    :param order: Harmonic order ``n`` (>= 2).
    :param n_harmonics: Highest harmonic order used for the total RMS.
    :param window: FFT window (default ``'hann'``).
    :return: nth-order harmonic distortion, as a ratio.
    :raises ValueError: If ``order`` < 2 or the inputs are invalid.
    """
    fs = resolve_fs(signal, fs, name="signal")
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    f0 = _positive(fundamental, "fundamental")
    if order < 2:
        raise ValueError("'order' must be at least 2.")
    n = max(n_harmonics, order)
    _, amps = _harmonic_amplitudes(sig, fs_v, f0, n, window)
    if amps.size < order or amps[0] <= 0.0:
        raise ValueError("The requested harmonic order exceeds the Nyquist limit.")
    total_rms = float(np.sqrt(np.sum(amps**2)))
    return float(amps[order - 1] / total_rms)


def _notched_residual(
    signal: NDArray[np.float64], fs: float, f0: float, notch_q: float
) -> NDArray[np.float64]:
    """Signal with the fundamental removed by the AES17 standard notch filter.

    ``notch_q`` is the *effective* quality factor of the applied (zero-phase,
    twice-filtered) response per AES17 5.2.8; the single-pass ``iirnotch``
    design is sharpened by ``_FILTFILT_NOTCH_Q_FACTOR`` to realise it.
    """
    from scipy import signal as sp_signal

    if f0 >= fs / 2.0:
        raise ValueError(
            "'fundamental' must be below the Nyquist frequency (fs / 2)."
        )
    b, a = sp_signal.iirnotch(f0, notch_q * _FILTFILT_NOTCH_Q_FACTOR, fs)
    return np.asarray(sp_signal.filtfilt(b, a, signal), dtype=np.float64)


def _steady_slice(n: int, fs: float, f0: float, notch_q: float) -> slice:
    """Interior slice discarding the notch ``filtfilt`` start/stop transient.

    The zero-phase notch needs several fundamental cycles to settle; without
    trimming, its start/stop transient leaks fundamental energy into the
    residual (a spurious THD+N floor). The trim is capped at a quarter of the
    signal from each end so at least the middle half always survives.
    """
    design_q = notch_q * _FILTFILT_NOTCH_Q_FACTOR  # ring time follows the design Q
    settle = round(_NOTCH_SETTLE_CYCLES * design_q * fs / f0)
    settle = min(settle, n // 4)
    return slice(settle, n - settle) if settle > 0 else slice(None)


def _band_rms(x: NDArray[np.float64], fs: float, f_lo: float, f_hi: float) -> float:
    """RMS of ``x`` restricted to the band ``[f_lo, f_hi]`` (brick-wall).

    Computed in the frequency domain via Parseval (AES17 5.2.10 sanctions
    frequency-domain filters); a brick-wall band comfortably exceeds the
    5.2.5 low-pass template (passband ±0.1 dB, stopband >= 60 dB).
    """
    n = x.size
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    sel = np.nonzero((freqs >= f_lo) & (freqs <= f_hi))[0]
    if sel.size == 0:
        return 0.0
    weights = np.full(sel.size, 2.0)
    if sel[0] == 0:
        weights[0] = 1.0  # DC bin is not mirrored
    if n % 2 == 0 and sel[-1] == freqs.size - 1:
        weights[-1] = 1.0  # Nyquist bin is not mirrored (even n)
    mean_square = float(np.sum(weights * np.abs(spec[sel]) ** 2)) / n**2
    return float(np.sqrt(mean_square))


def _aes17_rms_pair(
    signal: NDArray[np.float64],
    residual: NDArray[np.float64],
    fs: float,
    f0: float,
    notch_q: float,
    bandwidth: float | None,
) -> tuple[float, float]:
    """Total and residual RMS over the steady slice, band-limited per AES17.

    With ``bandwidth`` set (default 20 kHz), both are measured through the
    AES17 chain: 20 Hz high-pass plus the standard low-pass (5.2.5 / 6.3.1),
    so DC offsets and out-of-band noise do not count. ``bandwidth=None``
    keeps the full-Nyquist legacy measurement.
    """
    sl = _steady_slice(signal.size, fs, f0, notch_q)
    if bandwidth is None:
        total = float(np.sqrt(np.mean(signal[sl] ** 2)))
        resid = float(np.sqrt(np.mean(residual[sl] ** 2)))
        return total, resid
    bw = _positive(bandwidth, "bandwidth")
    f_hi = min(bw, fs / 2.0)
    total = _band_rms(signal[sl], fs, _AES17_HIGHPASS_HZ, f_hi)
    resid = _band_rms(residual[sl], fs, _AES17_HIGHPASS_HZ, f_hi)
    return total, resid


def thd_plus_noise(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    notch_q: float = _DEFAULT_NOTCH_Q,
    bandwidth: float | None = _AES17_BANDWIDTH_HZ,
    window: str = "hann",
    as_db: bool = False,
) -> float:
    r"""THD+N ratio (AES17-2015 6.3.1).

    The fundamental is removed with the standard notch filter
    (:math:`1.2 \le Q \le 3`, validated on the applied zero-phase response
    per 5.2.8) and the residual RMS is compared with the total RMS:
    :math:`\mathrm{THD{+}N} = V_\text{residual} / V_\text{total}` (a
    ratio, or :math:`20 \log_{10}` of it in dB). Both voltages are measured
    through the AES17 measurement bandwidth -- a 20 Hz high-pass plus the
    standard low-pass at ``bandwidth`` (5.2.5 / 6.3.1) -- so DC offsets
    and out-of-band noise do not inflate the result.

    :param signal: Captured signal (1-D). Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples and then cancels: this is a ratio of amplitudes drawn from
        the same record, so the factor divides out and the answer is the
        same calibrated or not.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param fundamental: Fundamental frequency, or ``None`` to auto-detect.
    :param notch_q: Effective notch quality factor (AES17: 1.2..3; default 2.0).
    :param bandwidth: Upper band-edge frequency of the AES17 chain, in Hz
        (default 20 kHz, the 5.2.5 standard value; capped at Nyquist).
        ``None`` disables the chain and measures the full Nyquist band
        (20 Hz high-pass included only when the chain is active).
    :param window: FFT window used only for fundamental auto-detection.
    :param as_db: Return :math:`20 \log_{10}(\text{ratio})` in dB instead of the
        ratio.
    :return: THD+N as a ratio (default) or in dB.
    :raises ValueError: If the inputs are invalid or ``notch_q`` out of range.
    """
    fs = resolve_fs(signal, fs, name="signal")
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    _validate_notch_q(notch_q)
    if fundamental is None:
        freqs, amp = _amplitude_spectrum(sig, fs_v, window)
        f0 = _fundamental_frequency(freqs, amp)
    else:
        f0 = _positive(fundamental, "fundamental")
    residual = _notched_residual(sig, fs_v, f0, float(notch_q))
    total_rms, residual_rms = _aes17_rms_pair(
        sig, residual, fs_v, f0, float(notch_q), bandwidth
    )
    if total_rms <= 0.0:
        raise ValueError("Signal has no energy.")
    ratio = residual_rms / total_rms
    if as_db:
        return float(20.0 * np.log10(ratio)) if ratio > 0.0 else -np.inf
    return ratio


def sinad(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    notch_q: float = _DEFAULT_NOTCH_Q,
    bandwidth: float | None = _AES17_BANDWIDTH_HZ,
    window: str = "hann",
) -> float:
    r"""Signal-to-noise-and-distortion ratio SINAD, in dB.

    .. math::

       \mathrm{SINAD} = -(\text{THD+N in dB})
       = 20 \log_{10}(V_\text{total} / V_\text{residual}),

    the reciprocal, in dB, of the THD+N ratio. AES17-2015 does not itself
    define SINAD; this value is derived from the AES17 6.3.1 THD+N
    measurement (same notch, same measurement bandwidth).

    :param signal: Captured signal (1-D). Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples and then cancels: SINAD is the decibel form of a ratio of
        amplitudes drawn from the same record rather than a level, so the
        factor divides out and the answer is the same calibrated or not.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param fundamental: Fundamental frequency, or ``None`` to auto-detect.
    :param notch_q: Effective notch quality factor (AES17: 1.2..3; default 2.0).
    :param bandwidth: Upper band-edge frequency of the AES17 chain, in Hz
        (default 20 kHz); ``None`` measures the full Nyquist band.
    :param window: FFT window used only for fundamental auto-detection.
    :return: SINAD, in dB.
    :raises ValueError: If the inputs are invalid.
    """
    thdn_db = thd_plus_noise(
        signal,
        fs,
        fundamental,
        notch_q=notch_q,
        bandwidth=bandwidth,
        window=window,
        as_db=True,
    )
    return float(-thdn_db)


#: ITU-R BS.468-4 weighting network response (Table 1): (frequency in Hz,
#: response in dB re 1 kHz). The recommendation's own tolerance rule
#: interpolates between the mask frequencies linearly in dB on a logarithmic
#: frequency axis, which is the interpolation used here.
_ITU_R_468_TABLE: tuple[tuple[float, float], ...] = (
    (31.5, -29.9), (63.0, -23.9), (100.0, -19.8), (200.0, -13.8),
    (400.0, -7.8), (800.0, -1.9), (1000.0, 0.0), (2000.0, 5.6),
    (3150.0, 9.0), (4000.0, 10.5), (5000.0, 11.7), (6300.0, 12.2),
    (7100.0, 12.0), (8000.0, 11.4), (9000.0, 10.1), (10000.0, 8.1),
    (12500.0, 0.0), (14000.0, -5.3), (16000.0, -11.7), (20000.0, -22.2),
    (31500.0, -42.7),
)


def itu_r_468_weighting(frequencies: ArrayLike) -> NDArray[np.float64]:
    """ITU-R BS.468-4 weighting response, in dB re 1 kHz.

    The nominal response of the Recommendation's Table 1 (identical to the
    IEC 60268-1 Appendix A network required by IEC 60268-3 14.12.11),
    interpolated linearly in dB over log-frequency -- the Recommendation's
    own rule for values between the mask frequencies -- and extrapolated
    beyond the table with the end-segment slopes. Zero frequency (DC) maps
    to ``-inf`` dB. AES17-2015 5.2.7 tabulates the same curve with an
    additional gain of -5,63 dB (unity at 2 kHz, the "CCIR-RMS" filter).

    :param frequencies: Frequencies, in Hz (scalar or array-like, >= 0).
    :return: Response in dB re the 1 kHz value, same shape as the input.
    :raises ValueError: for negative or non-finite frequencies.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if not np.all(np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError("'frequencies' must be finite and non-negative.")
    table_f = np.array([row[0] for row in _ITU_R_468_TABLE])
    table_db = np.array([row[1] for row in _ITU_R_468_TABLE])
    log_f = np.log10(np.maximum(f, np.finfo(np.float64).tiny))
    log_table = np.log10(table_f)
    out = np.asarray(np.interp(log_f, log_table, table_db), dtype=np.float64)
    # Extrapolate with the end-segment slopes (dB per decade); the
    # low-frequency slope also sends DC to -inf.
    lo_slope = (table_db[1] - table_db[0]) / (log_table[1] - log_table[0])
    hi_slope = (table_db[-1] - table_db[-2]) / (log_table[-1] - log_table[-2])
    below = log_f < log_table[0]
    above = log_f > log_table[-1]
    out[below] = table_db[0] + lo_slope * (log_f[below] - log_table[0])
    out[above] = table_db[-1] + hi_slope * (log_f[above] - log_table[-1])
    out[~(f > 0.0)] = -np.inf
    return out


def _weighted_rms_468(x: NDArray[np.float64], fs: float) -> float:
    """RMS of ``x`` weighted by the ITU-R BS.468-4 response (Parseval)."""
    n = x.size
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    gains = np.zeros(freqs.size)
    nonzero = freqs > 0.0
    gains[nonzero] = 10.0 ** (itu_r_468_weighting(freqs[nonzero]) / 20.0)
    weights = np.full(freqs.size, 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    mean_square = float(np.sum(weights * (gains * np.abs(spec)) ** 2)) / n**2
    return float(np.sqrt(mean_square))


def weighted_thd(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    notch_q: float = _DEFAULT_NOTCH_Q,
    weighting: Literal["468", "A", "C"] = "468",
    window: str = "hann",
) -> float:
    """Weighted total harmonic distortion (IEC 60268-3 14.12.11).

    The fundamental is notched out and the residual is frequency-weighted
    before its RMS is compared with the total signal RMS, so the perceptual
    emphasis of the distortion products is accounted for. The default
    weighting is the network required by the clause -- IEC 60268-1:1985
    Appendix A, the ITU-R BS.468-4 curve (peaking +12,2 dB near 6,3 kHz) with
    its standard 0 dB at 1 kHz normalization; ``'A'`` and ``'C'`` (IEC
    61672-1) are kept as explicitly labelled alternatives, not 14.12.11
    quantities.

    Validity note (14.12.11): because of the shape of the weighting response,
    the weighted measurement is valid only for fundamental frequencies
    between 31,5 Hz and 400 Hz.

    :param signal: Captured signal (1-D). Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples and then cancels: this is a ratio of amplitudes drawn from
        the same record, so the factor divides out and the answer is the
        same calibrated or not.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param fundamental: Fundamental frequency, or ``None`` to auto-detect.
    :param notch_q: Effective notch quality factor (default 2.0).
    :param weighting: Frequency weighting applied to the residual:
        ``'468'`` (ITU-R BS.468-4 / IEC 60268-1, the 14.12.11 default),
        ``'A'`` or ``'C'``.
    :param window: FFT window used only for fundamental auto-detection.
    :return: Weighted THD, as a ratio.
    :raises ValueError: If the inputs are invalid.
    """
    fs = resolve_fs(signal, fs, name="signal")
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    if weighting not in ("468", "A", "C"):
        raise ValueError("'weighting' must be '468', 'A' or 'C'.")
    _validate_notch_q(notch_q)
    if fundamental is None:
        freqs, amp = _amplitude_spectrum(sig, fs_v, window)
        f0 = _fundamental_frequency(freqs, amp)
    else:
        f0 = _positive(fundamental, "fundamental")
    residual = _notched_residual(sig, fs_v, f0, float(notch_q))
    sl = _steady_slice(sig.size, fs_v, f0, float(notch_q))
    total_rms = float(np.sqrt(np.mean(sig[sl] ** 2)))
    if total_rms <= 0.0:
        raise ValueError("Signal has no energy.")
    if weighting == "468":
        weighted_rms = _weighted_rms_468(np.asarray(residual[sl]), fs_v)
    else:
        from ..filters.weighting import weighting_filter

        weighted = weighting_filter(residual, int(fs_v), curve=weighting)
        weighted_rms = float(np.sqrt(np.mean(weighted[sl] ** 2)))
    return float(weighted_rms / total_rms)


# --------------------------------------------------------------------------- #
# Result bundle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HarmonicDistortionResult:
    r"""Harmonic analysis of a signal (IEC 60268-3 / AES17).

    :ivar fundamental: Fundamental frequency :math:`f_1`, in Hz.
    :ivar harmonic_frequencies: Harmonic frequencies :math:`n f_1`
        present, in Hz.
    :ivar harmonic_amplitudes: Peak amplitudes :math:`a_n` of the
        harmonics.
    :ivar thd_f: Total harmonic distortion relative to the fundamental.
    :ivar thd_r: Total harmonic distortion relative to the total RMS.
    :ivar thd_plus_noise: THD+N ratio (AES17).
    :ivar sinad_db: SINAD, in dB.
    """

    fundamental: float
    harmonic_frequencies: NDArray[np.float64]
    harmonic_amplitudes: NDArray[np.float64]
    thd_f: float
    thd_r: float
    thd_plus_noise: float
    sinad_db: float

    def plot(self, ax: Axes | None = None, *, language: str = "en",
             **kwargs: Any) -> Axes:
        """Plot the magnitude spectrum with the harmonics marked.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.electroacoustics import plot_harmonic_distortion

        check_language(language)
        return plot_harmonic_distortion(self, ax=ax, language=language, **kwargs)


def harmonic_analysis(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    n_harmonics: int = _DEFAULT_N_HARMONICS,
    notch_q: float = _DEFAULT_NOTCH_Q,
    bandwidth: float | None = _AES17_BANDWIDTH_HZ,
    window: str = "hann",
) -> HarmonicDistortionResult:
    """Full harmonic analysis of a signal (THD, THD+N, SINAD).

    Bundles the fundamental, the harmonic amplitudes and the THD (both
    conventions), THD+N and SINAD into a plottable result.

    :param signal: Captured signal (1-D). Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples: the three distortion ratios come out unchanged, and so does
        ``sinad_db``, which is the decibel form of one such ratio rather
        than a level. Only ``harmonic_amplitudes`` carries the unit, and so
        lands in pascals when the record is calibrated.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param fundamental: Fundamental frequency, or ``None`` to auto-detect.
    :param n_harmonics: Highest harmonic order (default 10).
    :param notch_q: Effective notch quality factor for THD+N (default 2.0).
    :param bandwidth: AES17 measurement bandwidth for THD+N/SINAD, in Hz
        (default 20 kHz; ``None`` measures the full Nyquist band).
    :param window: FFT window (default ``'hann'``).
    :return: A :class:`HarmonicDistortionResult`.
    :raises ValueError: If the inputs are invalid.
    """
    fs = resolve_fs(signal, fs, name="signal")
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    f0, amps = _harmonic_amplitudes(sig, fs_v, fundamental, n_harmonics, window)
    if amps.size == 0 or amps[0] <= 0.0:
        raise ValueError("No fundamental component found in the signal.")
    freqs = np.array([(k + 1) * f0 for k in range(amps.size)], dtype=np.float64)
    thd_f = thd(sig, fs_v, f0, kind="F", n_harmonics=n_harmonics, window=window)
    thd_r = thd(sig, fs_v, f0, kind="R", n_harmonics=n_harmonics, window=window)
    thdn = thd_plus_noise(
        sig, fs_v, f0, notch_q=notch_q, bandwidth=bandwidth, window=window
    )
    sinad_db = sinad(
        sig, fs_v, f0, notch_q=notch_q, bandwidth=bandwidth, window=window
    )
    return HarmonicDistortionResult(
        fundamental=f0,
        harmonic_frequencies=freqs,
        harmonic_amplitudes=amps,
        thd_f=thd_f,
        thd_r=thd_r,
        thd_plus_noise=thdn,
        sinad_db=sinad_db,
    )
