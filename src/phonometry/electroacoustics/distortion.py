#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Distortion metrics for electroacoustic equipment (IEC 60268-3 / AES17).

Harmonic and intermodulation distortion of amplifiers and audio equipment, from
a captured signal:

* **Total harmonic distortion** ``THD`` (IEC 60268-3 14.12.2-3), relative to
  the fundamental (``kind='F'``, the widespread convention) or to the total
  RMS (``kind='R'``, the 14.12.3.2 quantity), and the **nth-order harmonic
  distortion** (14.12.5).
* **THD+N** and the derived **SINAD** (AES17-2015 6.3.1): the fundamental is
  removed with the standard notch filter and the residual is compared with
  the total signal, both through the AES17 measurement bandwidth (20 Hz to
  20 kHz by default).
* **Modulation distortion** ``d_m,2``/``d_m,3`` (IEC 60268-3 14.12.7) and
  **difference-frequency distortion** ``d_d,2``/``d_d,3`` (14.12.8), plus the
  **total difference-frequency distortion** (14.12.10) -- the IEC per-order
  definitions, with the SMPTE combined-RMS convention alongside.
* **Dynamic intermodulation distortion** ``DIM`` (14.12.9) from the 15 kHz sine /
  3.15 kHz square-wave test signal.
* **Weighted THD** (14.12.11), the harmonic residual weighted by the
  IEC 60268-1 / ITU-R BS.468-4 network (A/C optional).

All metrics have an exact analytic oracle: a signal synthesised with known
harmonic or intermodulation amplitudes reproduces the closed-form ratio. The
functions assume the tones fall on (or very near) FFT bins -- use coherent
sampling (an integer number of periods) or supply a low-leakage window, as audio
analysers do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

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


def _validate_signal(signal: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
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
    return sig


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
    signal: NDArray[np.float64] | list[float],
    fs: float,
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

    :param signal: Captured signal (1-D). Coherent sampling (integer periods) or
        a low-leakage window gives the exact value.
    :param fs: Sample rate, in Hz.
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
    signal: NDArray[np.float64] | list[float],
    fs: float,
    fundamental: float,
    order: int,
    *,
    n_harmonics: int = _DEFAULT_N_HARMONICS,
    window: str = "hann",
) -> float:
    r"""nth-order harmonic distortion :math:`d_n` (IEC 60268-3 14.12.5).

    :math:`d_n = a_n / \sqrt{\sum_{k \ge 1} a_k^2}` -- the nth harmonic
    amplitude relative to the total RMS.

    :param signal: Captured signal (1-D).
    :param fs: Sample rate, in Hz.
    :param fundamental: Fundamental frequency :math:`f_1`, in Hz.
    :param order: Harmonic order ``n`` (>= 2).
    :param n_harmonics: Highest harmonic order used for the total RMS.
    :param window: FFT window (default ``'hann'``).
    :return: nth-order harmonic distortion, as a ratio.
    :raises ValueError: If ``order`` < 2 or the inputs are invalid.
    """
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
    signal: NDArray[np.float64] | list[float],
    fs: float,
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

    :param signal: Captured signal (1-D).
    :param fs: Sample rate, in Hz.
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
    signal: NDArray[np.float64] | list[float],
    fs: float,
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

    :param signal: Captured signal (1-D).
    :param fs: Sample rate, in Hz.
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
    signal: NDArray[np.float64] | list[float],
    fs: float,
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

    :param signal: Captured signal (1-D).
    :param fs: Sample rate, in Hz.
    :param fundamental: Fundamental frequency, or ``None`` to auto-detect.
    :param notch_q: Effective notch quality factor (default 2.0).
    :param weighting: Frequency weighting applied to the residual:
        ``'468'`` (ITU-R BS.468-4 / IEC 60268-1, the 14.12.11 default),
        ``'A'`` or ``'C'``.
    :param window: FFT window used only for fundamental auto-detection.
    :return: Weighted THD, as a ratio.
    :raises ValueError: If the inputs are invalid.
    """
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
        from ..metrology.parametric_filters import weighting_filter

        weighted = weighting_filter(residual, int(fs_v), curve=weighting)
        weighted_rms = float(np.sqrt(np.mean(weighted[sl] ** 2)))
    return float(weighted_rms / total_rms)


# --------------------------------------------------------------------------- #
# Intermodulation distortion (IEC 60268-3 14.12.7-10)
# --------------------------------------------------------------------------- #

#: Intermodulation product search half-width, in FFT bins. Wide enough to catch
#: a product under mild window leakage, narrow enough (together with the
#: spacing-based cap below) that neighbouring products, the primary tones and
#: DC are never swallowed into a product's window.
_IMD_SEARCH_BINS = 5.0


def _imd_component(
    freqs: NDArray[np.float64],
    amp: NDArray[np.float64],
    frequency: float,
    half_width: float,
    exclude: tuple[float, ...] = (),
) -> float:
    r"""Peak amplitude within ``±half_width`` of an intermodulation product.

    Returns 0 for products outside (0, Nyquist). The window is shrunk so it
    never contains the DC bin or an excluded component (a primary tone): a
    product that cannot be separated from a primary reads 0 rather than the
    primary's amplitude (e.g. octave-spaced clean tones must not report
    :math:`d_2 = 1`).
    """
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    nyquist = float(freqs[-1])
    if frequency <= 0.0 or frequency >= nyquist:
        return 0.0
    half = min(half_width, frequency - df)  # keep the DC bin out
    for fx in exclude:
        half = min(half, abs(frequency - fx) - df)
    if half < 0.0:
        return 0.0
    return _tone_amplitude(freqs, amp, frequency, half)


@dataclass(frozen=True)
class ModulationDistortionResult:
    r"""Modulation (intermodulation) distortion (IEC 60268-3 14.12.7).

    :ivar d2: Second-order modulation distortion :math:`d_{m,2}`
        (14.12.7.2 g): the *arithmetic* sum of the sideband amplitudes at
        :math:`f_2 \pm f_1` relative to the output amplitude at ``f2``.
    :ivar d3: Third-order modulation distortion :math:`d_{m,3}`
        (14.12.7.2 h): the arithmetic sum of the sidebands at
        :math:`f_2 \pm 2 f_1` relative to the output amplitude at ``f2``.
    :ivar smpte: Combined-RMS convention of SMPTE-type analyzers (not an
        IEC 60268-3 quantity): :math:`\sqrt{\sum a_s^2} / a_{f_2}` over
        all four sidebands.
    :ivar f_low: Low modulating tone ``f1``, in Hz.
    :ivar f_high: High carrier tone ``f2``, in Hz.
    :ivar carrier_amplitude: Measured output amplitude at ``f2`` (the
        reference of the per-order ratios).
    :ivar sideband_frequencies: The four intermodulation product
        frequencies in ascending order: :math:`f_2 - 2f_1`,
        :math:`f_2 - f_1`, :math:`f_2 + f_1` and :math:`f_2 + 2f_1`,
        in Hz.
    :ivar sideband_amplitudes: Measured peak amplitudes at
        ``sideband_frequencies`` (zero for a product that falls outside
        the analysis band or cannot be separated from a primary tone).
    """

    d2: float
    d3: float
    smpte: float
    f_low: float | None = None
    f_high: float | None = None
    carrier_amplitude: float | None = None
    sideband_frequencies: NDArray[np.float64] | None = None
    sideband_amplitudes: NDArray[np.float64] | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en",
             **kwargs: Any) -> Axes:
        r"""Plot the carrier and its modulation sidebands, with d2/d3
        annotated.

        Draws the output amplitude at ``f2`` (the 0 dB reference) and the
        four intermodulation sidebands at :math:`f_2 \pm f_1` and
        :math:`f_2 \pm 2f_1` as a stem-style spectrum in dB relative to
        the carrier, the modulation counterpart of
        :meth:`HarmonicDistortionResult.plot`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the marker ``plot`` call.
        :return: The axes.
        :raises ValueError: If the result carries no sideband spectrum data
            (a result constructed by hand without the spectral fields).
        """
        from .._i18n import check_language
        from .._plot.electroacoustics import plot_modulation_distortion

        check_language(language)
        return plot_modulation_distortion(self, ax=ax, language=language, **kwargs)


def modulation_distortion(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    f_low: float,
    f_high: float,
    *,
    window: str = "hann",
) -> ModulationDistortionResult:
    r"""Modulation distortion of the nth order (IEC 60268-3 14.12.7).

    A low-frequency tone :math:`f_1` (``f_low``, large) and a
    high-frequency tone :math:`f_2` (``f_high``; small, amplitude ratio
    preferably 4:1) are applied; the nth-order distortion shows up as
    modulation sidebands at :math:`f_2 \pm (n-1) f_1`. Per 14.12.7.2
    g)-h) the per-order values use the *arithmetic* sum of the two
    sideband amplitudes, referenced to the output voltage at ``f2``:

    .. math::

       d_{m,2} = (a_{f_2+f_1} + a_{f_2-f_1}) / a_{f_2}

       d_{m,3} = (a_{f_2+2f_1} + a_{f_2-2f_1}) / a_{f_2}

    (The alternative presentation :math:`d'_{m,n} = 5 d_{m,n}` references
    the 4:1 reference output voltage
    :math:`U_{2,\mathrm{ref}} = 5 U_{2,f_2}` instead.) The combined
    root-sum-square that SMPTE-type analyzers report is returned alongside
    as ``smpte``.

    :param signal: Captured signal (1-D).
    :param fs: Sample rate, in Hz.
    :param f_low: Low modulating tone ``f1``, in Hz (e.g. 60 Hz).
    :param f_high: High carrier tone ``f2``, in Hz (e.g. 7 kHz).
    :param window: FFT window (default ``'hann'``).
    :return: A :class:`ModulationDistortionResult` with ``d2``, ``d3`` and
        the ``smpte`` combined RMS.
    :raises ValueError: If the inputs are invalid.
    """
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    fl = _positive(f_low, "f_low")
    fh = _positive(f_high, "f_high")
    freqs, amp = _amplitude_spectrum(sig, fs_v, window)
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    # Sidebands are spaced f1 apart: cap the search window well inside that.
    half = min(_IMD_SEARCH_BINS * df, fl / 4.0)
    carrier = _tone_amplitude(freqs, amp, fh, half)
    if carrier <= 0.0:
        raise ValueError("No carrier component found at 'f_high'.")
    sidebands = {
        n: tuple(
            _imd_component(freqs, amp, fh + sign * (n - 1) * fl, half, exclude=(fl, fh))
            for sign in (1.0, -1.0)
        )
        for n in (2, 3)
    }
    d2 = (sidebands[2][0] + sidebands[2][1]) / carrier
    d3 = (sidebands[3][0] + sidebands[3][1]) / carrier
    smpte = float(
        np.sqrt(sum(a**2 for pair in sidebands.values() for a in pair)) / carrier
    )
    # Sideband spectrum in ascending frequency order: f2-2f1, f2-f1, f2+f1,
    # f2+2f1 (the ``sidebands`` pairs are stored (upper, lower) per order).
    sb_freqs = np.array([fh - 2.0 * fl, fh - fl, fh + fl, fh + 2.0 * fl])
    sb_amps = np.array(
        [sidebands[3][1], sidebands[2][1], sidebands[2][0], sidebands[3][0]]
    )
    return ModulationDistortionResult(
        d2=float(d2),
        d3=float(d3),
        smpte=smpte,
        f_low=fl,
        f_high=fh,
        carrier_amplitude=carrier,
        sideband_frequencies=sb_freqs,
        sideband_amplitudes=sb_amps,
    )


def difference_frequency_distortion(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    f1: float,
    f2: float,
    *,
    order: int = 2,
    window: str = "hann",
) -> float:
    r"""Difference-frequency distortion of the nth order (IEC 60268-3
    14.12.8).

    Two equal-amplitude tones :math:`f_1 < f_2` are applied. Per 14.12.8.1
    the reference voltage is :math:`U_{2,\mathrm{ref}} = 2 U_{2,f_2}` --
    realised here as the sum of both measured tone amplitudes, identical
    for the standard equal-amplitude tones -- and

    .. math::

       d_{d,2} = a_{f_2-f_1} / (a_{f_1} + a_{f_2})

       d_{d,3} = (a_{2f_2-f_1} + a_{2f_1-f_2}) / (a_{f_1} + a_{f_2})

    with the third order an *arithmetic* sum of the two products. Products
    that fall outside (0, Nyquist) or that cannot be separated from a primary
    tone or DC read zero.

    :param signal: Captured signal (1-D).
    :param fs: Sample rate, in Hz.
    :param f1: Lower tone, in Hz.
    :param f2: Upper tone, in Hz.
    :param order: Product order (2 or 3).
    :param window: FFT window (default ``'hann'``).
    :return: nth-order difference-frequency distortion, as a ratio.
    :raises ValueError: If ``order`` is not 2 or 3 or the inputs are invalid.
    """
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    fa = _positive(f1, "f1")
    fb = _positive(f2, "f2")
    if fa >= fb:
        raise ValueError("'f1' must be lower than 'f2'.")
    if order not in (2, 3):
        raise ValueError("'order' must be 2 or 3.")
    freqs, amp = _amplitude_spectrum(sig, fs_v, window)
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    half = min(_IMD_SEARCH_BINS * df, (fb - fa) / 4.0)
    ref = _tone_amplitude(freqs, amp, fa, half) + _tone_amplitude(freqs, amp, fb, half)
    if ref <= 0.0:
        raise ValueError("No primary tones found at 'f1'/'f2'.")
    if order == 2:
        value = _imd_component(freqs, amp, fb - fa, half, exclude=(fa, fb))
    else:
        value = _imd_component(
            freqs, amp, 2 * fa - fb, half, exclude=(fa, fb)
        ) + _imd_component(freqs, amp, 2 * fb - fa, half, exclude=(fa, fb))
    return float(value / ref)


def total_difference_frequency_distortion(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    f1: float = 8000.0,
    f2: float = 11950.0,
    *,
    window: str = "hann",
) -> float:
    r"""Total difference-frequency distortion (IEC 60268-3 14.12.10).

    A specific two-tone test with :math:`f_1 = 2 f_0` and
    :math:`f_2 = 3 f_0 - \delta` (the standard values, kept as defaults,
    are :math:`f_1 = 8` kHz, :math:`f_2 = 11.95` kHz, so
    :math:`f_0 = 4` kHz and :math:`\delta = 50` Hz). Only the two in-band
    products at :math:`f_0 \mp \delta` enter -- the second-order product
    at :math:`f_2 - f_1` and the third-order product at
    :math:`2 f_1 - f_2` -- combined in RMS over the arithmetic sum of the
    two tone output amplitudes (14.12.10.2 g):

    .. math::

       d_{\mathrm{TDFD}} =
       \sqrt{a_{f_2-f_1}^2 + a_{2f_1-f_2}^2} / (a_{f_1} + a_{f_2})

    (The out-of-band product at :math:`2 f_2 - f_1` is explicitly not
    part of it.)

    :param signal: Captured signal (1-D).
    :param fs: Sample rate, in Hz.
    :param f1: Lower tone, in Hz (default 8 kHz, per 14.12.10.2 b).
    :param f2: Upper tone, in Hz (default 11.95 kHz, per 14.12.10.2 b).
    :param window: FFT window (default ``'hann'``).
    :return: Total difference-frequency distortion, as a ratio.
    :raises ValueError: If the inputs are invalid.
    """
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    fa = _positive(f1, "f1")
    fb = _positive(f2, "f2")
    if fa >= fb:
        raise ValueError("'f1' must be lower than 'f2'.")
    freqs, amp = _amplitude_spectrum(sig, fs_v, window)
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    p_lo, p_hi = fb - fa, 2 * fa - fb  # f0 - delta and f0 + delta
    # The two products sit 2*delta apart: cap the window well inside that
    # so they are never double-counted, besides the usual bin-based cap.
    spacing = abs(p_hi - p_lo)
    half = min(_IMD_SEARCH_BINS * df, (fb - fa) / 4.0)
    if spacing > 0.0:
        half = min(half, spacing / 4.0)
    ref = _tone_amplitude(freqs, amp, fa, half) + _tone_amplitude(freqs, amp, fb, half)
    if ref <= 0.0:
        raise ValueError("No primary tones found at 'f1'/'f2'.")
    a_lo = _imd_component(freqs, amp, p_lo, half, exclude=(fa, fb))
    a_hi = _imd_component(freqs, amp, p_hi, half, exclude=(fa, fb))
    return float(np.sqrt(a_lo**2 + a_hi**2) / ref)


#: Highest square-wave harmonic order that produces a DIM difference product
#: below f_sine. For the standard 15 kHz / 3.15 kHz signal the nine products of
#: IEC 60268-3 Table 2 span 0.75 kHz (k=5) to 13.35 kHz (k=9), so k must reach 9.
_DIM_MAX_ORDER = 9
#: DIM product search half-width, as a fraction of f_square. Kept well below the
#: spacing between a product and the square-wave fundamental/harmonics (750 Hz
#: for the standard signal) so a strong f_square component is never mistaken for
#: an intermodulation product.
_DIM_SEARCH_FACTOR = 0.1


def _dim_components(f_sine: float, f_square: float, nyquist: float) -> list[float]:
    r"""DIM difference products
    :math:`\lvert k \, f_\mathrm{square} - f_\mathrm{sine} \rvert`
    below ``f_sine`` (Table 2)."""
    products: set[float] = set()
    for k in range(1, _DIM_MAX_ORDER + 1):
        for sign in (-1.0, 1.0):
            f = abs(k * f_square + sign * f_sine)
            if 0.0 < f < min(f_sine, nyquist):
                products.add(round(f, 6))
    return sorted(products)


def dynamic_intermodulation_distortion(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    *,
    f_sine: float = 15000.0,
    f_square: float = 3150.0,
    window: str = "hann",
) -> float:
    r"""Dynamic intermodulation distortion DIM (IEC 60268-3 14.12.9).

    From the standard test signal -- a ``f_sine`` = 15 kHz sine plus a
    low-pass-filtered ``f_square`` = 3.15 kHz square wave in a 1:4 peak
    ratio -- the DIM is the RMS of the intermodulation products
    :math:`\lvert k \, f_\mathrm{square} \pm f_\mathrm{sine} \rvert` that
    fall below ``f_sine`` (IEC 60268-3 Table 2), relative to the 15 kHz
    sine amplitude.

    :param signal: Captured signal (1-D).
    :param fs: Sample rate, in Hz.
    :param f_sine: High sine frequency, in Hz (default 15 kHz).
    :param f_square: Square-wave fundamental, in Hz (default 3.15 kHz).
    :param window: FFT window (default ``'hann'``).
    :return: Dynamic intermodulation distortion, as a ratio.
    :raises ValueError: If the inputs are invalid.
    """
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    fsine = _positive(f_sine, "f_sine")
    fsq = _positive(f_square, "f_square")
    freqs, amp = _amplitude_spectrum(sig, fs_v, window)
    search = fsq * _DIM_SEARCH_FACTOR
    # Reference: the output amplitude at f_s, per the 14.12.9.1 definition
    # ("the ratio of the r.m.s. sum of the output voltages ... to the
    # amplitude of the output voltage at the frequency f_s"). The 14.12.9.2 f)
    # formula prints the denominator as "U2", which Table 2 and item d) define
    # as the 8,70 kHz intermodulation component -- i.e. one of the nine terms
    # of the formula's own numerator. That is an editorial defect of
    # IEC 60268-3:2013 (see docs/ERRATA.md); the Otala convention implemented
    # here follows the 14.12.9.1 definition.
    ref = _tone_amplitude(freqs, amp, fsine, search)
    if ref <= 0.0:
        raise ValueError("No 15 kHz sine component found.")
    power = 0.0
    for f in _dim_components(fsine, fsq, fs_v / 2.0):
        power += _tone_amplitude(freqs, amp, f, search) ** 2
    return float(np.sqrt(power) / ref)


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
    signal: NDArray[np.float64] | list[float],
    fs: float,
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

    :param signal: Captured signal (1-D).
    :param fs: Sample rate, in Hz.
    :param fundamental: Fundamental frequency, or ``None`` to auto-detect.
    :param n_harmonics: Highest harmonic order (default 10).
    :param notch_q: Effective notch quality factor for THD+N (default 2.0).
    :param bandwidth: AES17 measurement bandwidth for THD+N/SINAD, in Hz
        (default 20 kHz; ``None`` measures the full Nyquist band).
    :param window: FFT window (default ``'hann'``).
    :return: A :class:`HarmonicDistortionResult`.
    :raises ValueError: If the inputs are invalid.
    """
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


# --------------------------------------------------------------------------- #
# AES17-2015 noise measurements (6.4)
# --------------------------------------------------------------------------- #

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
    f_hi = fs / 2.0 if bandwidth is None else min(_positive(bandwidth, "bandwidth"), fs / 2.0)
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
    signal: NDArray[np.float64] | list[float],
    fs: float,
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
    :param fs: Sample rate, in Hz.
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
    sig = _validate_signal(signal)
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
    signal: NDArray[np.float64] | list[float],
    fs: float,
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
    :param fs: Sample rate, in Hz.
    :param bandwidth: AES17 measurement bandwidth, in Hz (default 20 kHz;
        ``None`` measures the full Nyquist band).
    :param full_scale: Digital full-scale peak amplitude (default 1.0).
    :return: The idle channel noise level, in dBFS CCIR-RMS (a negative
        number for any real device).
    :raises ValueError: If the inputs are invalid.
    """
    sig = _validate_signal(signal)
    fs_v = _positive(fs, "fs")
    reference = _full_scale_rms(full_scale)
    weighted = _ccir_rms_weighted_rms(sig, fs_v, bandwidth)
    if weighted <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(weighted / reference))
