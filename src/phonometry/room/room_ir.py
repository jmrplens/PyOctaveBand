#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Impulse-response acquisition per BS EN ISO 18233:2006.

This module provides the deterministic-excitation front end for the "new
measurement methods" of ISO 18233: generate an excitation signal, play it
through the system under test, and recover the broadband impulse response
(IR) by deconvolution. Later processing (fractional-octave filtering,
Schroeder backward integration, reverberation time) consumes this IR.

Two excitation families are implemented:

* **Swept-sine (Annex B)** -- an exponential sine sweep (ESS). The frequency
  rises exponentially with time, which mimics a pink-noise source
  (ISO 18233:2006, B.3.2) and is the recommended broadband excitation. The
  IR is recovered by linear (zero-padded, non-circular) spectral
  deconvolution (B.5, Figure B.3):
  :math:`H = Y \overline{X} / (|X|^2 + \text{reg})`. A
  Farina inverse-filter variant (``method="farina"``) convolves the
  recording with the time-reversed, amplitude-compensated sweep (B.5,
  Figure B.2). Because a low-to-high sweep places distortion products at
  negative arrival times, harmonic distortion is separated from the linear
  IR and discarded by keeping only the causal part (B.5). To *analyse*
  those discarded products instead (per-order harmonic responses and
  THD(f) from the same recording), see
  :func:`phonometry.swept_sine_distortion`.

* **Maximum-length sequence (Annex A)** -- an order-``N`` binary sequence of
  length :math:`2^N - 1` generated with a linear-feedback shift register
  (LFSR). Its circular autocorrelation is a near-perfect delta (A.1), so the
  IR of a periodically excited linear system is recovered by circular
  cross-correlation of the recorded period with the sequence
  (equivalent to the Hadamard-transform recovery of A.1).

Two further excitations from the transfer-function measurement literature
complete the family:

* **Complementary Golay pair** -- two binary sequences of length
  :math:`L = 2^n` whose periodic autocorrelations sum to an *exact* delta of
  height :math:`2L` (Golay 1961; Havelock, Kuwano & Vorlaender (eds.), Handbook
  of Signal Processing in Acoustics, Springer 2008, Part I Ch. 6 by
  N. Xiang, Eq. (2)). Exciting the system with each code in turn and
  summing the two circular cross-correlations recovers the IR with zero
  correlation noise: the deterministic residue of each single-code
  correlation cancels identically, so only uncorrelated background noise
  remains (Xiang Eq. (4)). See :func:`golay_pair` and
  :func:`golay_impulse_response`.

* **Sweep with an arbitrary magnitude spectrum** -- a swept sine synthesized
  in the frequency domain by shaping its group delay so the dwell time at
  each frequency is proportional to the desired spectral power
  (Mueller & Massarani, "Transfer-Function Measurement with Sweeps", JAES
  49(6), 2001, Secs. 4.2-4.3). The sweep keeps the near-ideal crest factor
  of a swept sine while following any prescribed emphasis (pink,
  noise-floor-matched, loudspeaker-equalizing, ...). See
  :func:`shaped_sweep_signal`; the recording is deconvolved with the
  ordinary spectral method of :func:`impulse_response`, or post-equalized
  with :func:`phonometry.regularized_inverse_filter`.

The recovered IR is broadband; ISO 18233 6.3.2 requires subsequent
fractional-octave-band weighting (IEC 61260) before computing levels or
decay curves -- that step belongs to downstream room-acoustics modules.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal

from .._internal.utils import _typesignal
from .._internal.warnings import PhonometryWarning


class ImpulseResponseWarning(PhonometryWarning):
    """Warns about suspect recovered impulse responses (e.g. MLS aliasing)."""


if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Warn from mls_impulse_response when the recovered IR still carries more than
#: this level (dB re its peak) in the last 10 % of the MLS period: an IR longer
#: than one period aliases circularly and this residual tail is the symptom.
_MLS_ALIAS_TAIL_DB = -35.0

#: A genuine (optionally faded) exponential sweep never ends in a run of
#: exact zeros; more trailing zeros than this small tolerance indicate the
#: reference was zero-padded (see the Farina guard in ``impulse_response``).
_FARINA_MAX_TRAILING_ZEROS = 8

# Primitive-polynomial feedback taps (1-indexed register positions, highest
# tap == order) yielding maximum-length sequences. Values from the standard
# LFSR tables (Xilinx XAPP052, "Efficient Shift Registers, LFSR Counters,
# and Long Pseudo-Random Sequence Generators", P. Alfke, 1996; equivalent to
# the primitive-polynomial tables used by Rife & Vanderkooy, JAES 37 (1989),
# ISO 18233 Bibliography [16]).
_MLS_TAPS: dict[int, tuple[int, ...]] = {
    2: (2, 1),
    3: (3, 2),
    4: (4, 3),
    5: (5, 3),
    6: (6, 5),
    7: (7, 6),
    8: (8, 6, 5, 4),
    9: (9, 5),
    10: (10, 7),
    11: (11, 9),
    12: (12, 6, 4, 1),
    13: (13, 4, 3, 1),
    14: (14, 5, 3, 1),
    15: (15, 14),
    16: (16, 15, 13, 4),
    17: (17, 14),
    18: (18, 11),
    19: (19, 6, 2, 1),
    20: (20, 17),
}


@dataclass(frozen=True)
class ImpulseResponseResult:
    """Recovered broadband impulse response with its acquisition metadata.

    Returned by :func:`impulse_response` and :func:`mls_impulse_response`.
    The impulse response samples live in ``ir``; ``fs`` is the sample rate in
    Hz (or ``None`` when unknown, e.g. an MLS recovery called without one) and
    ``method`` records how the IR was obtained (``"spectral"``, ``"farina"``
    or ``"mls"``).

    The object is a drop-in replacement for the raw array it used to be: it
    implements :meth:`__array__`, so ``np.asarray(result)`` yields the IR and
    the result can be passed straight to array consumers such as
    :func:`phonometry.room_parameters`, :func:`phonometry.decay_curve` and
    :func:`phonometry.sti_from_impulse_response`. Indexing, ``len(result)``
    and the ``size``/``ndim``/``shape``/``dtype`` attributes forward to ``ir``.
    """

    ir: np.ndarray
    fs: int | None
    method: str

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Return the impulse response as an array (optionally recast)."""
        return np.asarray(self.ir, dtype=dtype)

    def __len__(self) -> int:
        return int(self.ir.shape[-1])

    def __getitem__(self, key: Any) -> Any:
        return self.ir[key]

    @property
    def size(self) -> int:
        """Number of samples in the impulse response."""
        return int(self.ir.size)

    @property
    def ndim(self) -> int:
        return int(self.ir.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.ir.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.ir.dtype

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes | np.ndarray:
        """Plot the impulse response: waveform and log-magnitude decay.

        Draws the (normalised) time-domain waveform and, below it, the
        log-magnitude envelope in dB with a Schroeder energy-decay overlay.
        With ``ax`` given, only the decay panel is drawn on it. Requires
        matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` (or an array of two axes).
        """
        from .._i18n import check_language
        from .._plot.room import plot_impulse_response

        check_language(language)
        return plot_impulse_response(self, ax=ax, language=language, **kwargs)


def sweep_signal(
    fs: int,
    f1: float,
    f2: float,
    seconds: float,
    *,
    amplitude: float = 1.0,
    fade: float = 0.01,
) -> np.ndarray:
    r"""
    Generate an exponential sine sweep (ESS) with exact analytic phase.

    The instantaneous frequency rises exponentially from ``f1`` to ``f2``,
    :math:`f(t) = f_1 (f_2/f_1)^{t/T}`, so the time spent per octave is
    constant and the sweep mimics a pink-noise excitation
    (ISO 18233:2006, B.3.2). The phase is the closed-form integral of
    :math:`2 \pi f(t)` (Farina, AES 108th Conv., 2000; ISO 18233 Bibliography
    [14]), avoiding numerical phase accumulation.

    :param fs: Sampling frequency in Hz.
    :param f1: Start frequency in Hz (at or below the lowest band edge to be
        measured, ISO 18233 B.3.1). Must be > 0.
    :param f2: Stop frequency in Hz (at or above the highest band edge). Must
        satisfy :math:`f_1 < f_2 \le f_s/2`.
    :param seconds: Sweep duration in seconds. Any duration may be used; a
        longer sweep raises the effective signal-to-noise ratio (B.2, B.6).
    :param amplitude: Peak amplitude of the sweep. Default 1.0.
    :param fade: Half-Hann fade-in/out length as a fraction of the sweep
        duration, applied to suppress start/stop transients (B.3.3). Default
        0.01. Set to 0.0 to disable. Because the sweep frequency is
        logarithmic in time, the fades consume roughly ``fade*log2(f2/f1)``
        octaves at each band edge (the fade-out lands on the highest
        frequencies): with the default 0.01 the top ~29 dB of the highest
        band is unusable, so choose ``f1``/``f2`` with margin beyond the
        analysis range (ISO 18233 B.3.1) rather than relying on a smaller
        fade.
    :return: The sweep samples, length ``round(seconds*fs)``.
    """
    if f1 <= 0.0:
        raise ValueError("f1 must be positive")
    if f2 <= f1:
        raise ValueError("f2 must be greater than f1")
    if f2 > fs / 2.0:
        raise ValueError("f2 must not exceed the Nyquist frequency fs/2")
    if not 0.0 <= fade < 0.5:
        raise ValueError("fade must be in [0, 0.5)")

    n = round(seconds * fs)
    if n < 2:
        raise ValueError("seconds*fs must yield at least 2 samples")

    t = np.arange(n) / fs
    ts = n / fs
    ratio = f2 / f1
    # phi(t) = 2*pi*f1*T/ln(f2/f1) * ((f2/f1)^(t/T) - 1)
    k = 2.0 * np.pi * f1 * ts / np.log(ratio)
    phase = k * (ratio ** (t / ts) - 1.0)
    sweep: np.ndarray = amplitude * np.sin(phase)

    if fade > 0.0:
        sweep = _apply_fade(sweep, fade)
    return sweep


def _apply_fade(x: np.ndarray, fade: float) -> np.ndarray:
    """Apply symmetric half-Hann fade windows to both ends of ``x``."""
    n = x.size
    m = round(fade * n)
    if m < 1:
        return x
    ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
    window = np.ones(n)
    window[:m] = ramp
    window[-m:] = ramp[::-1]
    return x * window


def inverse_filter(
    fs: int,
    f1: float,
    f2: float,
    seconds: float,
    *,
    amplitude: float = 1.0,
    fade: float = 0.01,
) -> np.ndarray:
    """
    Build the Farina inverse filter for an exponential sine sweep.

    The inverse filter is the time-reversed sweep multiplied by an amplitude
    envelope that rises by 6 dB/octave (``prop. to the instantaneous
    frequency``), which whitens the ESS's pink (-3 dB/octave) spectrum so
    that convolving the sweep with its inverse yields an impulse
    (ISO 18233:2006, B.5, Figure B.2; Farina 2000, Bibliography [14]). The
    filter is scaled to unit in-band magnitude: it is normalised by the
    median in-band magnitude of the compressed pulse ``sweep * inverse`` so
    that the deconvolution reproduces a system's true in-band level, matching
    the spectral-division convention (rather than a unit pulse peak).

    :param fs: Sampling frequency in Hz.
    :param f1: Sweep start frequency in Hz (same value used for the sweep).
    :param f2: Sweep stop frequency in Hz.
    :param seconds: Sweep duration in seconds.
    :param amplitude: Peak amplitude used for the source sweep. Default 1.0.
    :param fade: Fade fraction used for the source sweep. Default 0.01.
    :return: The inverse-filter samples (same length as the sweep).
    """
    sweep = sweep_signal(fs, f1, f2, seconds, amplitude=amplitude, fade=fade)
    n = sweep.size
    ts = n / fs
    t = np.arange(n) / fs
    ratio = f2 / f1
    # Reversed-sweep instantaneous frequency, normalised to f2:
    # f_rev(t) = f2 * (f1/f2) ** (t/T) = f2 * ratio ** (-t/T). Multiplying the
    # reversed sweep by f_rev(t)/f2 = ratio ** (-t/T) weights the spectrum by
    # a factor proportional to frequency (+6 dB/octave), which flattens the
    # ESS's pink (-3 dB/octave) magnitude so the compressed output is an
    # impulse (Farina 2000).
    env = ratio ** (-t / ts)  # decreases from 1 (at f2) to f1/f2 (at f1)
    inv = sweep[::-1] * env
    # Normalise so that the compressed pulse (sweep * inverse) has unit
    # magnitude across the sweep band. Scaling by the in-band gain (rather
    # than the pulse peak) makes the deconvolution reproduce a system's true
    # in-band level, matching the spectral-division convention.
    comp = signal.fftconvolve(sweep, inv)
    freqs = np.fft.rfftfreq(comp.size, d=1.0 / fs)
    band = (freqs >= f1) & (freqs <= f2)
    if not np.any(band):
        raise ValueError(
            "no frequency bin falls within the sweep band [f1, f2]; the "
            "sweep is too short for this frequency resolution; increase "
            "'seconds' or widen the (f1, f2) range."
        )
    gain = float(np.median(np.abs(np.fft.rfft(comp)[band])))
    if gain > 0.0:
        inv = inv / gain
    return inv


def _farina_deconvolve(
    rec: np.ndarray,
    ref: np.ndarray,
    fs: int,
    f_range: tuple[float, float] | None,
) -> np.ndarray:
    """Farina inverse-filter deconvolution (ISO 18233:2006, Figure B.2).

    Rebuilds the analytic inverse filter from ``ref`` and convolves it with the
    recording, returning the full sequence with the causal IR rolled to index
    0 (matching the spectral-method layout). Raises ``ValueError`` if
    ``f_range`` is missing or if ``ref`` is zero-padded (the correct input for
    the spectral method, which would silently produce a wrong inverse filter).
    """
    if f_range is None:
        raise ValueError("method='farina' requires f_range=(f1, f2)")
    # A reference that was zero-padded to the recording length - the correct
    # input for method="spectral" - makes the rebuilt sweep longer than the
    # real excitation, silently producing a wrong inverse filter (the IR peak
    # is mislocated and orders of magnitude too small). Detect trailing
    # zero-padding and reject it: a genuine (optionally faded) sweep has no
    # trailing run of exact zeros.
    nonzero = np.flatnonzero(ref)
    last_nonzero = int(nonzero[-1]) if nonzero.size else -1
    n_trailing = ref.size - 1 - last_nonzero
    if n_trailing > _FARINA_MAX_TRAILING_ZEROS:
        raise ValueError(
            f"method='farina' requires the unpadded excitation sweep as "
            f"'reference', but it ends in {n_trailing} zero samples "
            "(zero-padding to the recording length is only correct for "
            "method='spectral'). Pass the exact-length sweep from "
            "sweep_signal(), or use method='spectral'."
        )
    inv = inverse_filter(fs, f_range[0], f_range[1], ref.size / fs)
    conv = signal.fftconvolve(rec, inv)
    # The linear IR peaks where the inverse aligns with the sweep, at index
    # len(reference)-1. Roll it to index 0 so the layout matches the spectral
    # method (causal at 0, negative times in the tail).
    return np.roll(conv, -(ref.size - 1))


def impulse_response(
    recorded: list[float] | np.ndarray,
    reference: list[float] | np.ndarray,
    fs: int,
    *,
    method: str = "spectral",
    f_range: tuple[float, float] | None = None,
    regularization: float = 1e-6,
    length: int | None = None,
    return_full: bool = False,
) -> ImpulseResponseResult:
    r"""
    Recover the broadband impulse response by sweep deconvolution.

    Implements the linear (non-circular) deconvolution of ISO 18233:2006,
    B.5. Both signals are zero-padded to ``len(recorded)+len(reference)-1``
    to avoid circular convolution (B.5). The causal IR occupies the start of
    the result; distortion products from a low-to-high sweep fall at negative
    arrival times (the wrapped tail) and are discarded by returning only the
    causal part (B.5).

    :param recorded: Recorded system response to the sweep.
    :param reference: The emitted sweep (excitation signal).
    :param fs: Sampling frequency in Hz (kept for API symmetry; the
        deconvolution itself is sample-rate agnostic).
    :param method: ``"spectral"`` for spectral division
        :math:`H = Y \overline{X} / (\lvert X \rvert^2 + \text{reg})`
        (Figure B.3, default) or ``"farina"``
        for convolution with the analytic inverse filter (Figure B.2). The
        Farina method requires ``f_range`` and the **exact-length, unpadded**
        excitation sweep as ``reference`` (it rebuilds the inverse filter from
        ``reference.size/fs`` as the sweep duration); a reference zero-padded
        to the recording length - the correct input for the spectral method -
        is rejected with a ``ValueError`` because it would silently produce a
        wrong inverse filter. It also assumes the reference sweep was
        generated with the default ``amplitude``/``fade`` of
        :func:`sweep_signal`; a non-unit amplitude or custom fade yields a
        scaled IR, so use the spectral method in that case.
    :param f_range: ``(f1, f2)`` of the sweep, required for ``method="farina"``
        to rebuild the inverse filter; ignored for the spectral method.
    :param regularization: Tikhonov term added to the denominator, expressed
        as a fraction of the peak spectral energy
        :math:`\max(\lvert X \rvert^2)` (spectral
        method only). Guards against amplifying noise where the sweep has
        little energy, e.g. outside its frequency range (B.5). Default 1e-6.
    :param length: Number of samples of the causal IR to return. Defaults to
        ``len(recorded)``. Ignored when ``return_full`` is True.
    :param return_full: If True, return the full deconvolution sequence
        (causal IR at index 0, negative-time distortion products in the
        tail) instead of the trimmed causal IR. Default False.
    :return: An :class:`ImpulseResponseResult` wrapping the recovered impulse
        response. It behaves like the raw IR array (``np.asarray(result)``,
        indexing, ``.size``) for every downstream consumer and adds
        :meth:`ImpulseResponseResult.plot`.
    """
    rec = _typesignal(recorded)
    ref = _typesignal(reference)
    if rec.ndim != 1 or ref.ndim != 1:
        raise ValueError("'recorded' and 'reference' must be one-dimensional.")
    if rec.size == 0 or ref.size == 0:
        raise ValueError("'recorded' and 'reference' must be non-empty.")
    out_len = length if length is not None else rec.size
    n = rec.size + ref.size - 1

    if method == "spectral":
        spec_x = np.fft.rfft(ref, n=n)
        spec_y = np.fft.rfft(rec, n=n)
        power = np.abs(spec_x) ** 2
        reg = regularization * float(power.max()) if power.size else 0.0
        h_spec = spec_y * np.conj(spec_x) / (power + reg)
        full = np.fft.irfft(h_spec, n=n)
    elif method == "farina":
        full = _farina_deconvolve(rec, ref, fs, f_range)
    else:
        raise ValueError(f"unknown method {method!r}")

    ir = full if return_full else full[:out_len]
    return ImpulseResponseResult(ir=np.asarray(ir), fs=fs, method=method)


def mls_signal(order: int) -> np.ndarray:
    """
    Generate a maximum-length sequence (MLS) of the given order.

    A Fibonacci LFSR with primitive-polynomial feedback taps produces a
    binary sequence of length ``2**order - 1`` whose circular
    autocorrelation is a near-perfect periodic delta
    (ISO 18233:2006, A.1). The binary values are mapped to ``+1``/``-1``.

    :param order: Register length ``N`` (2 to 20). The sequence length is
        ``2**order - 1``.
    :return: The bipolar MLS samples (values in ``{-1.0, +1.0}``).
    """
    if order not in _MLS_TAPS:
        raise ValueError(
            f"order must be one of {sorted(_MLS_TAPS)} (got {order})"
        )
    taps = _MLS_TAPS[order]
    length = (1 << order) - 1
    # LFSR state as a boolean register, seeded with all ones (any non-zero
    # seed produces the same sequence up to a cyclic shift).
    state = np.ones(order, dtype=np.int8)
    out = np.empty(length, dtype=np.float64)
    tap_idx = [t - 1 for t in taps]
    for i in range(length):
        bit = state[-1]
        out[i] = 1.0 - 2.0 * bit  # 0 -> +1, 1 -> -1
        feedback = 0
        for t in tap_idx:
            feedback ^= state[t]
        state[1:] = state[:-1]
        state[0] = feedback
    return out


def mls_impulse_response(
    recorded: list[float] | np.ndarray,
    mls: list[float] | np.ndarray,
    *,
    length: int | None = None,
    fs: int | None = None,
) -> ImpulseResponseResult:
    """
    Recover an impulse response from a periodic MLS excitation.

    The recording must span an integer number of MLS periods; the periods
    are averaged (raising the effective signal-to-noise ratio by 3 dB per
    doubling, ISO 18233:2006, 6.3.6) and the IR is obtained by circular
    cross-correlation of the averaged period with the sequence, normalised by
    ``2**N`` (A.1). Because the sequence is periodic, the recovery is a
    circular deconvolution: a system IR longer than one period aliases back
    into the record (A.1).

    :param recorded: Recorded response, length a multiple of ``2**N - 1``.
    :param mls: The excitation sequence returned by :func:`mls_signal`.
    :param length: Number of IR samples to return. Defaults to the sequence
        length ``2**N - 1``.
    :param fs: Optional sample rate in Hz, stored on the result so that
        :meth:`ImpulseResponseResult.plot` can label a time axis in seconds
        (the recovery itself is sample-rate agnostic). Default ``None``.
    :return: An :class:`ImpulseResponseResult` wrapping the recovered impulse
        response. It behaves like the raw IR array for every downstream
        consumer and adds :meth:`ImpulseResponseResult.plot`.

    .. note::
        An :class:`ImpulseResponseWarning` is emitted when the recovered IR retains significant
        energy at the end of the period (a circular-aliasing symptom). The
        tail-RMS heuristic is advisory: a high ambient noise floor in the
        recording raises the tail RMS on its own and can trigger a
        false positive even when the IR fits within one period, so treat the
        warning as a prompt to check the noise floor and MLS order rather than
        a definitive aliasing diagnosis.
    """
    rec = _typesignal(recorded)
    seq = _typesignal(mls)
    if rec.ndim != 1 or seq.ndim != 1:
        raise ValueError("'recorded' and 'mls' must be one-dimensional.")
    if rec.size == 0 or seq.size == 0:
        raise ValueError("'recorded' and 'mls' must be non-empty.")
    period = seq.size
    if rec.size % period != 0:
        raise ValueError(
            "recorded length must be a positive multiple of the MLS length"
        )
    # Average the recorded periods (synchronous averaging, ISO 18233 6.3.6).
    averaged = rec.reshape(-1, period).mean(axis=0)
    # Circular cross-correlation: DFT{corr} = conj(SEQ) * REC.
    spec = np.conj(np.fft.rfft(seq)) * np.fft.rfft(averaged)
    corr = np.fft.irfft(spec, n=period)
    ir = corr / (period + 1.0)  # normalise by 2**N so the delta peaks at 1

    # Circular-aliasing guard: if the system IR is longer than one MLS period
    # it folds back into the record (A.1). The symptom is undecayed energy in
    # the last part of the period; warn (do not raise) so short-order misuse is
    # visible instead of silently biasing the result.
    _warn_alias_tail(ir, period, "MLS", "Use a higher MLS order.")
    out = _periodic_output(ir, period, length)
    return ImpulseResponseResult(ir=out, fs=fs, method="mls")


def _warn_alias_tail(
    ir: np.ndarray, period: int, excitation: str, remedy: str
) -> None:
    """Warn when a circularly recovered IR keeps energy at the period end.

    A system IR longer than one excitation period aliases back into the
    record under any circular (periodic) deconvolution; the symptom is
    undecayed energy in the last 10 % of the period. Warn (do not raise) so
    short-period misuse is visible instead of silently biasing the result.
    """
    peak = float(np.max(np.abs(ir)))
    if peak <= 0.0:
        return
    tail = ir[int(0.9 * period):]
    tail_rms = float(np.sqrt(np.mean(tail ** 2)))
    if tail_rms > peak * 10.0 ** (_MLS_ALIAS_TAIL_DB / 20.0):
        warnings.warn(
            f"Recovered {excitation} impulse response still has energy near "
            f"the end of the {period}-sample period "
            f"({20 * np.log10(tail_rms / peak):.0f} dB re peak, above "
            f"{_MLS_ALIAS_TAIL_DB:.0f} dB): the impulse response is likely "
            f"longer than one period and aliases circularly. {remedy}",
            ImpulseResponseWarning,
            stacklevel=3,
        )


def _periodic_output(
    ir: np.ndarray, period: int, length: int | None
) -> np.ndarray:
    """Trim or periodically extend a one-period IR to ``length`` samples."""
    out_len = length if length is not None else period
    if out_len <= period:
        out = ir[:out_len]
    else:
        # Periodic extension for requests longer than one period.
        reps = (out_len + period - 1) // period
        out = np.tile(ir, reps)[:out_len]
    return np.asarray(out)


# Largest supported Golay order: 2**22 samples per code (~87 s at 48 kHz)
# keeps the pair comfortably in memory while covering any realistic IR.
_GOLAY_MAX_ORDER = 22

#: Band-edge taper width of the shaped sweep's magnitude, in octaves. The
#: half-cosine roll-off lives *outside* the requested band, so the target
#: magnitude is honoured across all of ``[f1, f2]`` (Mueller & Massarani
#: 2001, Sec. 4.2: band-limiting the synthesis magnitude avoids the ripple
#: of an abrupt spectral start/stop).
_SHAPED_EDGE_OCTAVES = 1.0 / 6.0


def golay_pair(order: int) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Generate a complementary Golay pair of length ``2**order``.

    Built with the append recursion of Golay (1961): starting from
    ``a1 = (+1, +1)``, ``b1 = (+1, -1)``, each step appends ``b`` to ``a``
    and ``-b`` to ``a`` (Havelock, Handbook of Signal Processing in
    Acoustics, Springer 2008, Part I Ch. 6 (N. Xiang), Eq. (1)). The pair is
    *complementary*: the sum of the two periodic autocorrelations is exactly
    :math:`2L` at zero lag and exactly zero everywhere else (Xiang Eq. (2)) --
    an algebraic identity, not an approximation, unlike the near-delta
    autocorrelation of an MLS.

    :param order: Number of recursion steps ``n`` (1 to 22). Each code has
        :math:`L = 2^{\text{order}}` samples.
    :return: The pair ``(a, b)`` as bipolar float arrays (values ``+1/-1``).
    """
    if not 1 <= order <= _GOLAY_MAX_ORDER:
        raise ValueError(
            f"order must be between 1 and {_GOLAY_MAX_ORDER} (got {order})"
        )
    a = np.array([1.0, 1.0])
    b = np.array([1.0, -1.0])
    for _ in range(order - 1):
        a, b = np.concatenate((a, b)), np.concatenate((a, -b))
    return a, b


def golay_impulse_response(
    recorded_a: list[float] | np.ndarray,
    recorded_b: list[float] | np.ndarray,
    pair: tuple[np.ndarray, np.ndarray],
    *,
    length: int | None = None,
    fs: int | None = None,
) -> ImpulseResponseResult:
    r"""
    Recover an impulse response from a complementary Golay-pair excitation.

    Each code of the pair is emitted periodically (as with an MLS, record in
    the steady state: at least one settling period before acquisition); the
    recorded periods of each code are averaged and the IR is the sum of the
    two circular cross-correlations, normalised by :math:`2L` (Havelock 2008,
    Part I Ch. 6 (N. Xiang), Eq. (4) and the measurement procedure of
    Fig. 2):

    .. math::

       h = \operatorname{IFFT}\!\left[ \overline{A}\, \operatorname{FFT}(y_a)
       + \overline{B}\, \operatorname{FFT}(y_b) \right] / (2L)

    Because the pair's autocorrelations are *exactly* complementary
    (Xiang Eq. (2)), the recovery has no correlation noise: for a noiseless
    linear time-invariant system the IR is exact to machine precision,
    whereas an MLS leaves a small deterministic residue. Uncorrelated
    background noise is only attenuated by the averaging, and the price is
    a doubled excitation time and two steady states, which makes the pair
    more exposed to time variance than a single sweep (Xiang, Sec. 2).

    :param recorded_a: Recorded response to the periodic ``a`` code; its
        length must be a positive multiple of the code length ``L``.
    :param recorded_b: Recorded response to the periodic ``b`` code; its
        length must be a positive multiple of ``L`` (the period counts of
        the two recordings may differ).
    :param pair: The complementary pair ``(a, b)`` from :func:`golay_pair`.
    :param length: Number of IR samples to return. Defaults to ``L``;
        longer requests are periodic extensions.
    :param fs: Optional sample rate in Hz, stored on the result so that
        :meth:`ImpulseResponseResult.plot` can label a time axis in seconds
        (the recovery itself is sample-rate agnostic). Default ``None``.
    :return: An :class:`ImpulseResponseResult` (``method="golay"``). It
        behaves like the raw IR array for every downstream consumer and
        adds :meth:`ImpulseResponseResult.plot`.

    .. note::
        As with any periodic (circular) recovery, a system IR longer than
        one code period aliases back into the record; an
        :class:`ImpulseResponseWarning` flags undecayed energy at the end
        of the period (see the note in :func:`mls_impulse_response` about
        the heuristic's noise-floor false positives).
    """
    code_a, code_b = (_typesignal(c) for c in pair)
    if code_a.ndim != 1 or code_b.ndim != 1:
        raise ValueError("both Golay codes must be one-dimensional.")
    period = code_a.size
    if period < 2 or code_b.size != period:
        raise ValueError(
            "'pair' must hold two equal-length codes of at least 2 samples "
            "(use golay_pair())."
        )
    responses = []
    for name, rec_in, code in (
        ("recorded_a", recorded_a, code_a),
        ("recorded_b", recorded_b, code_b),
    ):
        rec = _typesignal(rec_in)
        if rec.ndim != 1:
            raise ValueError(f"'{name}' must be one-dimensional.")
        if rec.size == 0 or rec.size % period != 0:
            raise ValueError(
                f"'{name}' length must be a positive multiple of the "
                f"{period}-sample code length"
            )
        # Synchronous averaging of the recorded periods (as with the MLS).
        averaged = rec.reshape(-1, period).mean(axis=0)
        # Circular cross-correlation: DFT{corr} = conj(CODE) * REC.
        responses.append(np.conj(np.fft.rfft(code)) * np.fft.rfft(averaged))
    # Complementary sum, normalised by 2L (Xiang Eq. (4)).
    ir = np.fft.irfft(responses[0] + responses[1], n=period) / (2.0 * period)

    _warn_alias_tail(ir, period, "Golay", "Use a higher Golay order.")
    out = _periodic_output(ir, period, length)
    return ImpulseResponseResult(ir=out, fs=fs, method="golay")


@dataclass(frozen=True)
class ShapedSweepResult:
    """A sweep synthesized to follow an arbitrary target magnitude spectrum.

    Returned by :func:`shaped_sweep_signal`. The playable samples live in
    ``signal``; the object implements :meth:`__array__`, so it can be passed
    straight to a sound-card writer or as the ``reference`` of
    :func:`impulse_response` (spectral method). The synthesis metadata --
    the frequency grid, the band-limited magnitude actually imposed on the
    spectrum and the group delay that encodes the sweep's time-frequency
    trajectory (Mueller & Massarani 2001, Secs. 4.2-4.3) -- travels with
    the result, together with the achieved crest factor.

    :ivar signal: The sweep samples (peak ``amplitude``).
    :ivar fs: Sample rate, in Hz.
    :ivar frequencies: Frequency grid of the synthesis FFT, in Hz.
    :ivar magnitude: Band-limited magnitude imposed on the synthesis
        spectrum, normalised to a peak of 1 (linear).
    :ivar group_delay: Synthesized group delay ``tau_G(f)`` on
        :attr:`frequencies`, in seconds: the time at which each frequency
        is swept through.
    :ivar f_range: ``(f1, f2)`` band covered by the sweep, in Hz.
    :ivar crest_factor_db: Peak-to-RMS ratio over the sweep's central
        (constant-envelope) interval, in dB. A time-domain swept sine has
        the ideal 3.02 dB; the frequency-domain synthesis stays close to
        it (Mueller & Massarani 2001, Sec. 4.3: normally below 4 dB).
        Their figure assumes the slight magnitude smoothing the paper
        applies before synthesis, which this module does not implement:
        typical shaped sweeps here measure about 4.2-4.3 dB.
    """

    signal: np.ndarray
    fs: float
    frequencies: np.ndarray
    magnitude: np.ndarray
    group_delay: np.ndarray
    f_range: tuple[float, float]
    crest_factor_db: float

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Return the sweep samples as an array (optionally recast)."""
        return np.asarray(self.signal, dtype=dtype)

    def __len__(self) -> int:
        return int(self.signal.shape[-1])

    def __getitem__(self, key: Any) -> Any:
        return self.signal[key]

    @property
    def size(self) -> int:
        """Number of samples in the sweep."""
        return int(self.signal.size)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | np.ndarray:
        """Plot the sweep waveform and its spectrum against the target.

        Two stacked panels: the time-domain waveform, and the sweep's Welch
        magnitude spectrum overlaid on the synthesis target (both in dB re
        their in-band maximum). With ``ax`` given, only the spectrum panel
        is drawn on it. Requires matplotlib
        (``pip install phonometry[plot]``).

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.room import plot_shaped_sweep

        check_language(language)
        return plot_shaped_sweep(self, ax=ax, language=language, **kwargs)


def _shaped_target_db(
    target: str | tuple[np.ndarray, np.ndarray],
    freqs: np.ndarray,
    f_range: tuple[float, float],
) -> np.ndarray:
    """Evaluate the target magnitude, in dB, on the synthesis grid.

    Named targets: ``"white"`` is flat; ``"pink"`` falls 3 dB per octave in
    magnitude (``-10*log10(f/f1)``), the spectrum of the classical
    logarithmic sweep (Mueller & Massarani 2001, Sec. 4). An arbitrary
    target is a ``(frequencies_hz, magnitude_db)`` pair, interpolated
    linearly in dB over log-frequency and held constant beyond its ends.
    """
    safe = np.maximum(freqs, freqs[1] if freqs.size > 1 else 1.0)
    if isinstance(target, str):
        if target == "white":
            return np.zeros_like(freqs)
        if target == "pink":
            return np.asarray(-10.0 * np.log10(safe / f_range[0]))
        raise ValueError(
            f"unknown named target {target!r}; use 'white', 'pink' or a "
            "(frequencies_hz, magnitude_db) pair"
        )
    t_freq = np.asarray(target[0], dtype=np.float64)
    t_db = np.asarray(target[1], dtype=np.float64)
    if t_freq.ndim != 1 or t_freq.size < 2 or t_db.shape != t_freq.shape:
        raise ValueError(
            "an array target must be a (frequencies_hz, magnitude_db) pair "
            "of equal-length 1-D arrays with at least 2 points"
        )
    if np.any(t_freq <= 0.0) or np.any(np.diff(t_freq) <= 0.0):
        raise ValueError("target frequencies must be positive and increasing")
    if not np.all(np.isfinite(t_db)):
        raise ValueError("target magnitudes must be finite")
    return np.asarray(np.interp(np.log10(safe), np.log10(t_freq), t_db))


def _band_edge_taper(
    freqs: np.ndarray, f_range: tuple[float, float], fs: float
) -> np.ndarray:
    """Half-cosine band-edge weights (1 inside ``f_range``, 0 far outside).

    The tapers roll off over ``_SHAPED_EDGE_OCTAVES`` octaves *outside* the
    band, so the requested range keeps the full target magnitude; the upper
    taper is clipped at Nyquist. DC is always zeroed (a real periodic
    signal cannot carry an arbitrary DC magnitude).
    """
    f1, f2 = f_range
    ratio = 2.0 ** _SHAPED_EDGE_OCTAVES
    lo_start = f1 / ratio
    hi_stop = min(f2 * ratio, fs / 2.0)
    weight = np.zeros_like(freqs)
    inside = (freqs >= f1) & (freqs <= f2)
    weight[inside] = 1.0
    below = (freqs >= lo_start) & (freqs < f1)
    if np.any(below):
        x = np.log2(freqs[below] / lo_start) / np.log2(f1 / lo_start)
        weight[below] = 0.5 - 0.5 * np.cos(np.pi * x)
    above = (freqs > f2) & (freqs <= hi_stop) if hi_stop > f2 else \
        np.zeros_like(freqs, dtype=bool)
    if np.any(above):
        x = np.log2(freqs[above] / f2) / np.log2(hi_stop / f2)
        weight[above] = 0.5 + 0.5 * np.cos(np.pi * x)
    weight[0] = 0.0
    return weight


def shaped_sweep_signal(
    fs: int,
    f1: float,
    f2: float,
    seconds: float,
    *,
    target: str | tuple[np.ndarray, np.ndarray] = "pink",
    amplitude: float = 1.0,
    start_delay: float | None = None,
    fade: float = 0.01,
) -> ShapedSweepResult:
    r"""
    Synthesize a sweep with an arbitrary target magnitude spectrum.

    Implements the frequency-domain sweep construction of
    Mueller & Massarani ("Transfer-Function Measurement with Sweeps", JAES
    49(6), 2001, Secs. 4.2-4.3): the magnitude of the synthesis spectrum is
    set to the band-limited target, and the group delay grows in proportion
    to the target's spectral power (Eqs. (11)-(12)):

    .. math::

       \tau_G(f) = \tau_G(f - df) + C\, |H(f)|^2

       C = \frac{\tau_G(f_{\text{end}}) - \tau_G(f_{\text{start}})}
       {\sum |H|^2}

    so the sweep dwells on each frequency for a time proportional to the
    energy it must radiate there and its temporal envelope stays nearly
    constant -- the crest factor stays close to a swept sine's ideal
    3.02 dB regardless of the spectral shape (Sec. 4.3). The phase is the
    integral of the group delay, corrected to land on a real spectrum at
    Nyquist (Eq. (10)), and the sweep is obtained by inverse FFT over a
    block at least double the sweep length so the pre-ringing of the
    band-limited spectrum cannot fold onto the sweep's tail (Sec. 4.2).

    Deconvolve the recording with :func:`impulse_response`
    (``method="spectral"``), passing ``np.asarray(result)`` zero-padded as
    the reference, exactly as with :func:`sweep_signal`; the sweep's
    coloration divides out, so the target emphasis only re-weights the
    measurement's noise floor (that is its purpose: SNR shaping).

    :param fs: Sampling frequency in Hz.
    :param f1: Start frequency of the sweep band in Hz. Must be > 0. The
        magnitude rolls off over 1/6 octave *below* ``f1`` (clipped at the
        first FFT bin), so the full target level holds across ``[f1, f2]``.
    :param f2: Stop frequency in Hz. Must satisfy :math:`f_1 < f_2 \le f_s/2`;
        keep some margin below Nyquist so the upper roll-off has room.
    :param seconds: Sweep duration :math:`\tau_G(f_2) - \tau_G(f_1)` in
        seconds.
        The returned signal is slightly longer (lead-in plus tail margin,
        see ``start_delay``).
    :param target: The magnitude shape: ``"pink"`` (default; 3 dB per
        octave falling, the classical room-measurement emphasis),
        ``"white"`` (flat), or a ``(frequencies_hz, magnitude_db)`` pair of
        arrays interpolated in dB over log-frequency (only the shape
        matters; any overall offset is normalised away).
    :param amplitude: Peak amplitude of the returned sweep. Default 1.0.
    :param start_delay: Group delay assigned to ``f1``, in seconds; the
        same margin is left after ``tau_G(f2)``, so the signal lasts
        ``seconds + 2*start_delay``. The sweep spreads slightly beyond its
        nominal start (Sec. 4.2: the group delay of the lowest bin "should
        not be set to zero"), so the default ``0.05*seconds`` gives the
        first half-wave room to evolve.
    :param fade: Half-Hann fade-in/out length as a fraction of the returned
        signal, applied to pin the ends to zero (Sec. 4.2). Default 0.01;
        0.0 disables.
    :return: A :class:`ShapedSweepResult` wrapping the sweep samples and
        the synthesis metadata (grid, imposed magnitude, group delay,
        crest factor).
    """
    fs_v = float(fs)
    if fs_v <= 0.0:
        raise ValueError("fs must be positive")
    if f1 <= 0.0:
        raise ValueError("f1 must be positive")
    if f2 <= f1:
        raise ValueError("f2 must be greater than f1")
    if f2 > fs_v / 2.0:
        raise ValueError("f2 must not exceed the Nyquist frequency fs/2")
    if seconds <= 0.0:
        raise ValueError("seconds must be positive")
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    if not 0.0 <= fade < 0.5:
        raise ValueError("fade must be in [0, 0.5)")
    lead = 0.05 * seconds if start_delay is None else float(start_delay)
    if lead <= 0.0:
        raise ValueError("start_delay must be positive")

    # FFT block at least double the total signal so low-frequency
    # pre-ringing folding to "negative times" stays clear of the tail
    # (Mueller & Massarani 2001, Sec. 4.2).
    total = seconds + 2.0 * lead
    n_keep = round(total * fs_v)
    if n_keep < 16:
        raise ValueError("seconds*fs must yield at least 16 samples")
    n_fft = int(2 ** np.ceil(np.log2(2 * n_keep)))
    freqs = np.asarray(np.fft.rfftfreq(n_fft, 1.0 / fs_v), dtype=np.float64)
    if not np.any((freqs >= f1) & (freqs <= f2)):
        raise ValueError(
            "no frequency bin falls within [f1, f2]; lengthen 'seconds' or "
            "widen the (f1, f2) range."
        )

    # Band-limited target magnitude (linear, normalised to peak 1).
    target_db = _shaped_target_db(target, freqs, (float(f1), float(f2)))
    magnitude = 10.0 ** (target_db / 20.0) * _band_edge_taper(
        freqs, (float(f1), float(f2)), fs_v
    )
    magnitude /= float(np.max(magnitude))

    # Group delay grows with the spectral power (Eqs. (11)-(12)); the
    # cumulative sum is the discrete form of the bin-by-bin recursion.
    power = magnitude ** 2
    tau = lead + seconds * np.cumsum(power) / float(np.sum(power))

    # Phase = -2*pi * integral of the group delay; then force the Nyquist
    # phase onto a multiple of pi with a linear-in-f correction (Eq. (10)).
    df = fs_v / n_fft
    phase = -2.0 * np.pi * df * np.cumsum(tau)
    phase -= phase[0]
    residual = phase[-1] - np.pi * np.round(phase[-1] / np.pi)
    phase -= residual * freqs / (fs_v / 2.0)

    spectrum = magnitude * np.exp(1j * phase)
    sweep = np.fft.irfft(spectrum, n=n_fft)[:n_keep]
    if fade > 0.0:
        sweep = _apply_fade(sweep, fade)
    sweep *= amplitude / float(np.max(np.abs(sweep)))

    # Crest factor over the constant-envelope interval [lead, lead+seconds].
    # A tiny 'seconds' next to a dominant 'start_delay' can round the two
    # edges onto the same sample, leaving an empty (or single-sample) core
    # that would make the statistic blow up; fall back to the whole
    # retained sweep in that degenerate case.
    core = sweep[round(lead * fs_v):round((lead + seconds) * fs_v)]
    if core.size < 2:
        core = sweep
    rms = float(np.sqrt(np.mean(core ** 2)))
    crest_db = 20.0 * np.log10(float(np.max(np.abs(core))) / rms)

    return ShapedSweepResult(
        signal=sweep,
        fs=fs_v,
        frequencies=freqs,
        magnitude=magnitude,
        group_delay=tau,
        f_range=(float(f1), float(f2)),
        crest_factor_db=crest_db,
    )
