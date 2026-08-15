#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Intermodulation distortion of audio equipment (IEC 60268-3 14.12.7-10).

Where the harmonic metrics of :mod:`phonometry.electroacoustics.distortion`
drive the equipment with a single tone, these drive it with two (or with a
sine against a square wave) and read the products that a non-linearity puts
*between* the excitation frequencies -- the components a harmonic
measurement cannot see and the ear finds least forgiving, because they are
inharmonic with the programme:

* **Modulation distortion** ``d_m,2``/``d_m,3`` (14.12.7): a large low tone
  :math:`f_1` modulating a small high tone :math:`f_2`, read on the
  sidebands at :math:`f_2 \\pm (n-1) f_1`.
* **Difference-frequency distortion** ``d_d,2``/``d_d,3`` (14.12.8) from two
  equal-amplitude tones, and the **total difference-frequency distortion**
  (14.12.10) of the standard 8 kHz / 11,95 kHz pair.
* **Dynamic intermodulation distortion** ``DIM`` (14.12.9) from the 15 kHz
  sine / 3,15 kHz square-wave test signal.

The per-order definitions are the IEC ones (arithmetic sums of the product
amplitudes, referenced as each clause prescribes), with the SMPTE
combined-RMS convention reported alongside where analyzers use it. As with
the harmonic metrics, every quantity has an exact analytic oracle: a signal
synthesised with known product amplitudes reproduces the closed-form ratio,
and the tones are assumed to fall on (or very near) FFT bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .distortion import (
    _amplitude_spectrum,
    _positive,
    _tone_amplitude,
    _validate_signal,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

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

    :ivar d2: Second-order modulation distortion :math:`d_{\mathrm{m},2}`
        (14.12.7.2 g): the *arithmetic* sum of the sideband amplitudes at
        :math:`f_2 \pm f_1` relative to the output amplitude at ``f2``.
    :ivar d3: Third-order modulation distortion :math:`d_{\mathrm{m},3}`
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
        :meth:`~phonometry.electroacoustics.distortion.HarmonicDistortionResult.plot`.

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

       d_{\mathrm{m},2} = (a_{f_2+f_1} + a_{f_2-f_1}) / a_{f_2}

       d_{\mathrm{m},3} = (a_{f_2+2f_1} + a_{f_2-2f_1}) / a_{f_2}

    (The alternative presentation :math:`d'_{\mathrm{m},n} = 5 d_{\mathrm{m},n}` references
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

       d_{\mathrm{d},2} = a_{f_2-f_1} / (a_{f_1} + a_{f_2})

       d_{\mathrm{d},3} = (a_{2f_2-f_1} + a_{2f_1-f_2}) / (a_{f_1} + a_{f_2})

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
