#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Cepstral analysis: real/power/complex cepstrum, liftering and echo detection.

The **cepstrum** is the inverse Fourier transform of the logarithm of a
spectrum. Because the log turns the convolution ``x = h * u`` into the sum
``ln X = ln H + ln U`` (Havelock, Kahle & Cocchi (eds.), *Handbook of Signal
Processing in Acoustics*, Springer 2008: Milner, Ch. 27, Eqs. (22)-(23)),
components that overlap hopelessly in the spectrum separate cleanly in the
cepstral domain -- the smooth spectral envelope collapses onto the low
**quefrencies** (the time-like axis of the cepstrum, in seconds) while
periodic spectral ripple from harmonics, reflections or echoes concentrates
at the quefrency of its period. Three variants are standard:

* the **power cepstrum**, the inverse transform of the log *power* spectrum
  ``ln|X|^2`` (Milner Fig. 21) -- real, even, phase-blind. (Convention
  note: Bogert, Healy & Tukey's original 1963 power cepstrum squares once
  more, ``|IDFT(ln|X|^2)|^2``; this module follows Milner's linear
  ``IDFT(ln|X|^2)`` throughout.);
* the **real cepstrum**, the inverse transform of ``ln|X|`` -- exactly half
  the power cepstrum, and the quantity whose causal folding yields the
  minimum-phase reconstruction of :func:`phonometry.minimum_phase`
  (Bendat & Piersol, *Random Data*, 4th ed., Sec. 13.1.4; Tohyama in
  Havelock Ch. 75 manipulates minimum-phase and all-pass components the
  same way);
* the **complex cepstrum**, the inverse transform of the full complex
  logarithm ``ln|X| + j arg X`` with the phase unwrapped (Neelamani in
  Havelock Ch. 87, Eq. (14)) -- invertible, hence the engine of
  homomorphic deconvolution.

**Echoes.** A single reflection ``x(t) = s(t) + a s(t - t0)`` multiplies the
spectrum by ``1 + a e^{-j 2 pi f t0}``, whose complex logarithm expands (for
``|a| < 1``) into the exactly summable series

``ln(1 + a e^{-j theta}) = sum_{n>=1} (-1)^{n+1} (a^n / n) e^{-j n theta}``,

so the cepstrum carries a spike train at the *rahmonics* ``n t0`` with
amplitudes ``a, -a^2/2, a^3/3, ...`` (their sum is ``ln(1 + a)``): a peak at
exactly the echo delay whose height reads out the reflection coefficient.
:func:`echo_detection` automates that reading on the power cepstrum, where
the first rahmonic's height is ``a`` itself.

**Liftering** -- filtering in the quefrency domain (Milner Sec. 4.3) --
splits a log spectrum into its smooth envelope (low quefrencies kept by a
lowpass lifter) and its fine structure (highpass): the classical route to a
spectral envelope free of harmonic ripple, or to the ripple alone.

All estimators here operate on a single record with one FFT of length
``nfft`` (even, at least the record length; the record is zero-padded up
to it). The discrete cepstrum is the inverse *DFT* of the log of a
*sampled* spectrum, so it is time-aliased when the log spectrum has
features sharper than the grid resolves; zero-padding ``nfft`` is the
remedy, exactly like the ``oversample`` padding of
:func:`phonometry.minimum_phase`, whose cepstral folding core
(:func:`_fold_causal`) this module shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .spectra import _positive, _validate_signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "CepstrumResult",
    "EchoDetectionResult",
    "LifterResult",
    "cepstrum",
    "echo_detection",
    "lifter",
]

#: Relative magnitude floor (about -300 dB) applied before the logarithm so
#: exact spectral zeros stay finite, matching ``phonometry.minimum_phase``.
_MAGNITUDE_FLOOR = 1e-15

#: The cepstrum kinds :func:`cepstrum` accepts.
_KINDS = ("power", "real", "complex")

#: Natural-log to dB conversion for log-magnitude spectra: ``20/ln(10)``.
_NEPER_TO_DB = 20.0 / np.log(10.0)


def _fold_causal(cepstrum: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fold a real (even) cepstrum onto positive quefrencies.

    The minimum-phase reconstruction keeps the quefrency-zero and Nyquist
    samples, doubles the positive quefrencies and drops the negative ones
    (Bendat & Piersol Sec. 13.1.4; the causal-part manipulation of Tohyama
    in Havelock Ch. 75): the result is the complex cepstrum of the unique
    minimum-phase signal whose log magnitude produced ``cepstrum``. Shared
    verbatim with :func:`phonometry.minimum_phase`, whose output is
    bit-identical to the pre-refactor implementation.

    :param cepstrum: Real cepstrum over a full even-length quefrency axis.
    :return: The folded (causal) cepstrum, same length.
    """
    n = cepstrum.size
    folded = np.zeros(n, dtype=np.float64)
    folded[0] = cepstrum[0]
    folded[1 : n // 2] = 2.0 * cepstrum[1 : n // 2]
    folded[n // 2] = cepstrum[n // 2]
    return folded


def _validate_nfft(n: int, nfft: int | None) -> int:
    if nfft is None:
        return n if n % 2 == 0 else n + 1
    out = int(nfft)
    if out < n:
        raise ValueError(
            f"'nfft' must be at least the record length ({n} samples)."
        )
    if out % 2 != 0:
        raise ValueError("'nfft' must be even.")
    return out


def _log_magnitude(
    x: NDArray[np.float64], nfft: int
) -> NDArray[np.float64]:
    """One-sided ``ln|X|`` of the zero-padded record, floored like phase.py."""
    magnitude = np.abs(np.fft.rfft(x, nfft))
    peak = float(np.max(magnitude))
    if peak <= 0.0:  # max of |X| is non-negative: <= 0 means all-zero
        raise ValueError("'x' must not be identically zero.")
    return np.asarray(
        np.log(np.maximum(magnitude, peak * _MAGNITUDE_FLOOR)),
        dtype=np.float64,
    )


def _complex_cepstrum(
    x: NDArray[np.float64], nfft: int
) -> tuple[NDArray[np.float64], int]:
    """Complex cepstrum and the removed linear-phase component.

    The full-grid phase is unwrapped and its linear component removed --
    the integer number ``r`` of half-turns accumulated at the Nyquist bin,
    negative for a bulk delay -- so the log spectrum is continuous and
    periodic and the inverse DFT converges (Neelamani Eq. (14); the
    removal is restored by :meth:`CepstrumResult.invert`).
    """
    spectrum = np.fft.fft(x, nfft)
    magnitude = np.abs(spectrum)
    peak = float(np.max(magnitude))
    if peak <= 0.0:  # max of |X| is non-negative: <= 0 means all-zero
        raise ValueError("'x' must not be identically zero.")
    log_mag = np.log(np.maximum(magnitude, peak * _MAGNITUDE_FLOOR))
    phase = np.unwrap(np.angle(spectrum))
    half = nfft // 2
    delay = round(phase[half] / np.pi)
    phase = phase - np.pi * delay * np.arange(nfft) / half
    values = np.fft.ifft(log_mag + 1j * phase).real
    return np.asarray(values, dtype=np.float64), int(delay)


@dataclass(frozen=True)
class CepstrumResult:
    """A cepstrum over its full quefrency axis.

    The quefrency axis runs ``0 .. (nfft-1)/fs``; quefrencies above
    ``nfft/(2 fs)`` are the negative (anticausal) quefrencies of the
    periodic axis. The power and real cepstra are even about that midpoint,
    so their first half carries everything; the complex cepstrum is not
    (its causal/anticausal split is what separates minimum-phase from
    all-pass content, Havelock Ch. 75).

    :ivar quefrencies: Quefrency axis, in seconds.
    :ivar cepstrum: Cepstrum values (real-valued for the three kinds).
    :ivar kind: ``"power"``, ``"real"`` or ``"complex"``.
    :ivar fs: Sample rate of the analysed record, in Hz.
    :ivar nfft: FFT length used (even; the record is zero-padded to it).
    :ivar linear_phase_samples: Linear-phase component removed from the
        unwrapped phase before the complex cepstrum: the integer number of
        half-turns at the Nyquist bin, roughly minus the bulk delay in
        samples (0 for the other kinds, and for a minimum-phase record);
        restored by :meth:`invert`.
    """

    quefrencies: NDArray[np.float64]
    cepstrum: NDArray[np.float64]
    kind: str
    fs: float
    nfft: int
    linear_phase_samples: int

    def invert(self) -> NDArray[np.float64]:
        """Reconstruct the record from a complex cepstrum.

        Applies the forward chain in reverse -- DFT, restore the removed
        linear phase, exponentiate, inverse DFT (the homomorphic
        deconvolution round trip of Neelamani Sec. 3.3) -- and returns the
        zero-padded record, length :attr:`nfft`. Only the complex cepstrum
        keeps the phase, so only it is invertible.

        :return: The reconstructed signal, length :attr:`nfft`.
        :raises ValueError: If :attr:`kind` is not ``"complex"``.
        """
        if self.kind != "complex":
            raise ValueError(
                "Only the complex cepstrum is invertible; the power and "
                "real cepstra discard the phase."
            )
        log_spectrum = np.fft.fft(self.cepstrum)
        ramp = (
            np.pi
            * self.linear_phase_samples
            * np.arange(self.nfft)
            / (self.nfft // 2)
        )
        spectrum = np.exp(log_spectrum) * np.exp(1j * ramp)
        return np.asarray(np.fft.ifft(spectrum).real, dtype=np.float64)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the cepstrum against quefrency (positive quefrencies).

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_cepstrum

        check_language(language)
        return plot_cepstrum(self, ax=ax, language=language, **kwargs)


def cepstrum(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    kind: str = "power",
    nfft: int | None = None,
) -> CepstrumResult:
    """
    Cepstrum of a record: power, real or complex.

    * ``"power"``: inverse DFT of ``ln|X|^2`` (Milner Fig. 21 in Havelock
      Ch. 27). Even, phase-blind; an echo of reflection coefficient ``a``
      at delay ``t0`` shows rahmonics of amplitude
      ``(-1)^{n+1} a^n / n`` at quefrencies ``n t0`` -- height ``a`` at the
      delay itself.
    * ``"real"``: inverse DFT of ``ln|X|`` -- exactly half the power
      cepstrum. Folding it causally is the minimum-phase reconstruction
      (see :func:`phonometry.minimum_phase`, which shares this module's
      folding core).
    * ``"complex"``: inverse DFT of ``ln|X| + j arg X`` with the phase
      unwrapped and its linear component removed (Neelamani Eq. (14) in
      Havelock Ch. 87). Real-valued for a real record, and invertible:
      :meth:`CepstrumResult.invert` returns the signal.

    :param x: Signal, 1-D.
    :param fs: Sample rate, in Hz.
    :param kind: ``"power"`` (default), ``"real"`` or ``"complex"``.
    :param nfft: Even FFT length, at least ``x.size`` (default: the record
        length, rounded up to even). Zero-padding reduces the cepstral
        time-aliasing of sharp log-spectrum features.
    :return: A :class:`CepstrumResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x", context="a cepstrum")
    fs_v = _positive(fs, "fs")
    if kind not in _KINDS:
        raise ValueError(f"'kind' must be one of {_KINDS}, got {kind!r}.")
    n = _validate_nfft(xa.size, nfft)

    delay = 0
    if kind == "complex":
        values, delay = _complex_cepstrum(xa, n)
    else:
        scale = 2.0 if kind == "power" else 1.0
        values = np.asarray(
            np.fft.irfft(scale * _log_magnitude(xa, n), n), dtype=np.float64
        )
    quefrencies = np.arange(n, dtype=np.float64) / fs_v
    return CepstrumResult(
        quefrencies=quefrencies,
        cepstrum=values,
        kind=kind,
        fs=fs_v,
        nfft=n,
        linear_phase_samples=delay,
    )


@dataclass(frozen=True)
class LifterResult:
    """A log spectrum split by liftering (Milner Sec. 4.3).

    :ivar frequencies: Frequency axis, in Hz.
    :ivar spectrum_db: Log-magnitude spectrum of the record, in dB.
    :ivar liftered_db: Log-magnitude spectrum rebuilt from the kept
        quefrencies alone, in dB. Lowpass and highpass parts add up to
        :attr:`spectrum_db` exactly (the split is linear in the log).
    :ivar quefrencies: Quefrency axis of the underlying real cepstrum, s.
    :ivar cepstrum: The real cepstrum the lifter window was applied to.
    :ivar cutoff: Lifter cutoff quefrency, in seconds.
    :ivar mode: ``"lowpass"`` (spectral envelope) or ``"highpass"``
        (fine structure).
    :ivar fs: Sample rate of the analysed record, in Hz.
    :ivar nfft: FFT length used.
    """

    frequencies: NDArray[np.float64]
    spectrum_db: NDArray[np.float64]
    liftered_db: NDArray[np.float64]
    quefrencies: NDArray[np.float64]
    cepstrum: NDArray[np.float64]
    cutoff: float
    mode: str
    fs: float
    nfft: int

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the cepstrum with the cutoff and the two log spectra.

        With ``ax`` given, only the spectrum panel is drawn on it.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_lifter

        check_language(language)
        return plot_lifter(self, ax=ax, language=language, **kwargs)


def lifter(
    x: NDArray[np.float64] | list[float],
    fs: float,
    cutoff: float,
    *,
    mode: str = "lowpass",
    nfft: int | None = None,
) -> LifterResult:
    """
    Lifter a record's log spectrum: keep quefrencies below or above a cutoff.

    Liftering is filtering in the quefrency domain (Milner Sec. 4.3 in
    Havelock Ch. 27): the real cepstrum is windowed and transformed back
    to a log-magnitude spectrum. A **lowpass** lifter keeps quefrencies
    below ``cutoff`` (and the quefrency-zero mean level), recovering the
    smooth spectral envelope with the harmonic or echo ripple removed; a
    **highpass** lifter keeps the complement, isolating the ripple. The two
    modes are exactly complementary in dB.

    :param x: Signal, 1-D.
    :param fs: Sample rate, in Hz.
    :param cutoff: Cutoff quefrency, in seconds; must resolve to at least
        one sample and at most ``nfft/2`` samples.
    :param mode: ``"lowpass"`` (default) or ``"highpass"``.
    :param nfft: Even FFT length, at least ``x.size`` (default: the record
        length, rounded up to even).
    :return: A :class:`LifterResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x", context="liftering")
    fs_v = _positive(fs, "fs")
    if mode not in ("lowpass", "highpass"):
        raise ValueError(
            f"'mode' must be 'lowpass' or 'highpass', got {mode!r}."
        )
    n = _validate_nfft(xa.size, nfft)
    cut = round(_positive(cutoff, "cutoff") * fs_v)
    if not 1 <= cut <= n // 2:
        raise ValueError(
            "'cutoff' must resolve to between 1 sample and nfft/2 samples "
            f"({1.0 / fs_v:.3g} s to {n // 2 / fs_v:.3g} s), got "
            f"{cutoff:.3g} s."
        )

    log_mag = _log_magnitude(xa, n)
    values = np.asarray(np.fft.irfft(log_mag, n), dtype=np.float64)
    keep_low = np.zeros(n, dtype=np.float64)
    keep_low[:cut] = 1.0
    keep_low[n - cut + 1 :] = 1.0  # the mirrored negative quefrencies
    window = keep_low if mode == "lowpass" else 1.0 - keep_low
    liftered_log = np.fft.rfft(values * window).real

    return LifterResult(
        frequencies=np.fft.rfftfreq(n, 1.0 / fs_v),
        spectrum_db=np.asarray(_NEPER_TO_DB * log_mag, dtype=np.float64),
        liftered_db=np.asarray(_NEPER_TO_DB * liftered_log, dtype=np.float64),
        quefrencies=np.arange(n, dtype=np.float64) / fs_v,
        cepstrum=values,
        cutoff=cut / fs_v,
        mode=mode,
        fs=fs_v,
        nfft=n,
    )


def _parabolic_offset(
    magnitude: NDArray[np.float64], peak: int, lo: int, hi: int
) -> float:
    """Sub-sample offset of a peak by quadratic (parabolic) interpolation.

    Fits a parabola through ``magnitude`` at ``peak - 1, peak, peak + 1``
    and returns the vertex offset in ``[-0.5, 0.5]`` samples: the standard
    three-point refinement of a delay that falls between quefrency bins.
    Zero when the peak sits at an edge of the searched band or the triple
    is not concave.
    """
    if peak <= lo or peak >= hi:
        return 0.0
    y0, y1, y2 = magnitude[peak - 1 : peak + 2]
    curvature = y0 - 2.0 * y1 + y2
    if curvature >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / curvature, -0.5, 0.5))


@dataclass(frozen=True)
class EchoDetectionResult:
    """An echo delay and reflection coefficient read off the power cepstrum.

    :ivar quefrencies: Quefrency axis, in seconds.
    :ivar cepstrum: Power cepstrum searched.
    :ivar delay: The echo delay, in seconds: the quefrency of the largest
        ``|cepstrum|`` peak in the searched band, refined by quadratic
        (parabolic) interpolation of ``|cepstrum|`` through the peak sample
        and its two neighbours, so an off-sample delay is estimated to a
        fraction of a sample.
    :ivar delay_samples: The peak position in whole samples (no
        interpolation): the index at which ``reflection_coefficient``
        is read, so ``delay`` differs from ``delay_samples/fs`` by the
        interpolated sub-sample offset (at most half a sample).
    :ivar reflection_coefficient: Signed cepstrum value at the peak sample.
        For a single in-record echo ``x(t) = s(t) + a s(t - t0)`` the power
        cepstrum's first rahmonic height is exactly ``a`` -- of either sign
        -- (the ``n = 1`` term of the ``ln(1 + a e^{-j theta})`` series),
        so the value estimates the reflection coefficient directly,
        including its polarity. When the true delay falls between samples
        the rahmonic is split across neighbouring quefrency bins and the
        reported coefficient underestimates ``|a|`` (down to roughly 65 %
        of it for a delay midway between samples); the interpolated
        :attr:`delay` is unaffected.
    :ivar search_range: The ``(min, max)`` quefrency band searched, s.
    :ivar fs: Sample rate of the analysed record, in Hz.
    :ivar nfft: FFT length used.
    """

    quefrencies: NDArray[np.float64]
    cepstrum: NDArray[np.float64]
    delay: float
    delay_samples: int
    reflection_coefficient: float
    search_range: tuple[float, float]
    fs: float
    nfft: int

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the power cepstrum with the detected echo marked.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_echo_detection

        check_language(language)
        return plot_echo_detection(self, ax=ax, language=language, **kwargs)


def echo_detection(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    min_quefrency: float | None = None,
    max_quefrency: float | None = None,
    nfft: int | None = None,
) -> EchoDetectionResult:
    """
    Detect an echo as the largest power-cepstrum peak in a quefrency band.

    A reflection ``x(t) = s(t) + a s(t - t0)`` leaves a spike of height
    ``a`` -- positive or negative with the sign of the reflection -- at
    quefrency ``t0`` in the power cepstrum (module note; the seismic
    reverberation spike trains of Neelamani Sec. 3.3 are the same
    signature), regardless of the spectrum of ``s`` itself, which
    concentrates at low quefrencies. The peak is therefore picked on
    ``|cepstrum|``, so an inverting reflection (``a < 0``, e.g. a
    pressure-release boundary) is found at its true delay, and the
    *signed* cepstrum value at the peak is returned as the reflection
    coefficient. The search band starts above the low-quefrency region
    occupied by the source's spectral envelope: raise ``min_quefrency``
    if the source is very reverberant or narrowband.

    The delay is refined by quadratic interpolation of ``|cepstrum|``
    around the peak; see :class:`EchoDetectionResult` for the bin-splitting
    caveat on the coefficient at off-sample delays.

    :param x: Signal (an impulse response, or any record with an in-record
        echo), 1-D.
    :param fs: Sample rate, in Hz.
    :param min_quefrency: Lower edge of the searched band, in seconds
        (default 16 samples, clearing the immediate quefrency-zero region;
        raise it above the source's envelope quefrencies when needed).
    :param max_quefrency: Upper edge of the searched band, in seconds
        (default and maximum: half the FFT length, the end of the
        unambiguous quefrency axis).
    :param nfft: Even FFT length, at least ``x.size`` (default: the record
        length, rounded up to even).
    :return: An :class:`EchoDetectionResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    result = cepstrum(x, fs, kind="power", nfft=nfft)
    n = result.nfft
    fs_v = result.fs

    lo = 16 if min_quefrency is None else round(min_quefrency * fs_v)
    hi = n // 2 if max_quefrency is None else round(max_quefrency * fs_v)
    if not 1 <= lo < hi <= n // 2:
        raise ValueError(
            "The quefrency search band must satisfy "
            f"1 sample <= min < max <= nfft/2 samples; got [{lo}, {hi}] "
            f"samples of nfft = {n}."
        )

    # Peak-pick on the magnitude so a negative (inverting) reflection is
    # found at its true delay; the signed value at the peak is reported.
    magnitude = np.abs(result.cepstrum)
    peak = int(np.argmax(magnitude[lo : hi + 1])) + lo
    return EchoDetectionResult(
        quefrencies=result.quefrencies,
        cepstrum=result.cepstrum,
        delay=(peak + _parabolic_offset(magnitude, peak, lo, hi)) / fs_v,
        delay_samples=peak,
        reflection_coefficient=float(result.cepstrum[peak]),
        search_range=(lo / fs_v, hi / fs_v),
        fs=fs_v,
        nfft=n,
    )
