#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Time synchronous averaging (TSA) of a periodic waveform in noise.

Time domain averaging extracts a repetitive signal of known period ``T``
from additive noise by ensemble-averaging successive length-``T`` blocks,
following P. D. McFadden, "A revised model for the extraction of periodic
waveforms by time domain averaging", *Mechanical Systems and Signal
Processing* 1(1) 1987, 83-95. Given a signal
:math:`y(t) = x(t) + e(t)` with ``x`` periodic in ``T`` and ``e``
asynchronous, the average

.. math::

   a(t) = \frac{1}{N} \sum_{n=0}^{N-1} y(t + n T)
   \tag{McFadden Eq. 5}

reinforces every component synchronous with ``T`` and suppresses the rest.

**Two models, one implementation.** McFadden distinguishes the *existing*
comb-filter model from the *revised* model. In the frequency domain the
average is the multiplication of :math:`Y(f)` by the comb filter (Eq. 8)

.. math::

   C(f) = \frac{1}{N} \, \frac{\sin(N \pi f T)}{\sin(\pi f T)}

whose magnitude
:math:`\lvert C(f) \rvert
= \lvert \sin(N \pi f T) / (N \sin(\pi f T)) \rvert` is a Dirichlet
kernel: unity at every harmonic :math:`k/T` (the teeth, Eq. 9, of unit
height regardless of ``N``) and zero at the nodes :math:`j/(N T)` with
``j`` not a
multiple of ``N``. That model assumes knowledge of ``y`` over infinite time
and produces a result that is not exactly periodic. McFadden's *revised*
model applies a rectangular window of width ``T`` in the time domain and
samples the transform in the frequency domain, so it needs only a finite
block of the signal and yields a result that is exactly periodic and can be
stored as a single period. The digital block average computed here, ``N``
consecutive periods of an integer number of samples reduced to one period,
*is* that revised model: the returned :attr:`period_waveform`, repeated,
is exactly periodic.

**Noise reduction.** Asynchronous noise of variance :math:`\sigma^2`
averaged over ``N`` periods has residual variance :math:`\sigma^2/N`:
the residual standard deviation falls as :math:`1/\sqrt{N}` and the
amplitude signal-to-noise ratio improves by :math:`\sqrt{N}` (a power
reduction of :math:`10 \log_{10} N` dB, reported as
:attr:`noise_reduction_db`).

**Choosing N (McFadden's revised-model correction).** Because a discrete
interfering tone at a *non-harmonic* order :math:`q = f T` is only
attenuated, not removed, its rejection is optimised by choosing ``N`` so
that a comb node lands exactly on it, i.e. the smallest ``N`` with
:math:`N q` an integer.
McFadden's own example, a tone at 32.05 orders, is suppressed by more
than 100 dB with :math:`N = 20` (since :math:`20 \cdot 32.05 = 641`)
yet -- evaluating Eq. 8
at that order -- by only about 14 dB with the common power-of-two choice
:math:`N = 32` (:math:`32 \cdot 32.05 = 1025.6`; the paper makes the
comparison but does
not print this figure). Thus the habit of taking a power-of-two number of
averages is not, in general, optimal.

**Non-integer samples per period.** When :math:`f_\mathrm{s} T` is not an integer
the period boundaries fall between samples. Each block is then aligned to
a common integer grid by the band-limited fractional delay of
:func:`phonometry.signals.test_signals.fractional_delay` before averaging, so
the periodic waveform is recovered within the interpolation error of that
band-limited shift. An integer :math:`f_\mathrm{s} T` needs no interpolation and
the waveform is recovered to machine precision. The averaged samples stay
on the :math:`1/f_\mathrm{s}` sampling grid throughout: output sample ``m`` is the
average of the input at the times :math:`n T + m/f_\mathrm{s}`, so the returned
time axis is :math:`m/f_\mathrm{s}` and the
:math:`M = \operatorname{round}(f_\mathrm{s} T)` samples cover one period exactly
when :math:`f_\mathrm{s} T` is an integer and to within half a sample otherwise.
No resampling onto an ``M``-point angular grid is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .spectra import _positive
from .test_signals import _validate_1d_finite, fractional_delay

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "SynchronousAverageResult",
    "comb_filter_response",
    "time_synchronous_average",
]

#: Below this a fractional sample offset counts as integer-aligned, so no
#: interpolation is applied and an integer-period record is exact.
_ALIGN_TOL = 1e-9


def comb_filter_response(
    frequencies: NDArray[np.float64] | list[float],
    period: float,
    n_averages: int,
) -> NDArray[np.float64]:
    r"""Magnitude of the N-period synchronous-averaging comb filter.

    The closed form of McFadden Eq. 8,
    :math:`\lvert C(f) \rvert
    = \lvert \sin(N \pi f T) / (N \sin(\pi f T)) \rvert`,
    a Dirichlet kernel with unit-height teeth at the
    harmonics :math:`k/T` (Eq. 9) and nodes at :math:`j/(N T)` for ``j``
    not a multiple of ``N``.

    :param frequencies: Frequencies at which to evaluate, in Hz.
    :param period: Repetition period ``T``, in seconds.
    :param n_averages: Number of averaged periods ``N`` (at least 1).
    :return: The filter magnitude at each frequency (unitless; bounded by 1,
        though floating-point cancellation immediately beside a tooth can
        return values above 1 by a few parts in 1e9).
    :raises ValueError: If the parameters are invalid.
    """
    period_v = _positive(period, "period")
    n = int(n_averages)
    if n < 1:
        raise ValueError("'n_averages' must be a positive integer.")
    freqs = np.asarray(frequencies, dtype=np.float64)
    if not np.all(np.isfinite(freqs)):
        raise ValueError("'frequencies' must be finite.")
    order = freqs * period_v
    # Finite inputs can still overflow the phase products f*T and N*pi*f*T
    # (sin(inf) is NaN); reject them so the bounded-response contract holds.
    if not np.all(np.isfinite(n * np.pi * order)):
        raise ValueError(
            "'frequencies' * 'period' (times n_averages*pi) overflows the "
            "floating-point range; the comb filter cannot be evaluated at "
            "such orders."
        )
    lower = np.sin(np.pi * order)
    upper = np.sin(n * np.pi * order)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(upper / (n * lower))
    # At the teeth (integer order) the 0/0 limit is exactly 1 (Eq. 9).
    at_tooth = np.abs(lower) < _ALIGN_TOL
    response = np.where(at_tooth, 1.0, ratio)
    return np.asarray(response, dtype=np.float64)


@dataclass(frozen=True)
class SynchronousAverageResult:
    r"""Time synchronous average of a periodic waveform in noise.

    :ivar period_waveform: The averaged periodic waveform, one period of
        :attr:`samples_per_period` samples.
    :ivar times: Time axis of :attr:`period_waveform`, in seconds: the
        sampling grid :math:`m/f_\mathrm{s}`, :math:`m = 0 \ldots M-1` (the
        averaged samples stay on the :math:`1/f_\mathrm{s}` grid; see the module
        note). The axis spans one period exactly when :math:`f_\mathrm{s} T` is an
        integer, and to within half a sample otherwise.
    :ivar residual: Input minus the periodic reconstruction, over the
        analysed span (``n_averages * samples_per_period`` samples, aligned
        to the integer period grid): what is left after the synchronous
        component is removed.
    :ivar n_averages: Number of periods averaged, ``N``.
    :ivar samples_per_period: Integer samples per period ``M`` after any
        alignment.
    :ivar period: Repetition period ``T``, in seconds.
    :ivar fs: Sample rate, in Hz.
    :ivar interpolated: Whether band-limited fractional-delay alignment was
        applied (``True`` when :math:`f_\mathrm{s} T` is not an integer).
    :ivar noise_reduction_db: Power reduction of asynchronous noise,
        :math:`10 \log_{10} N` dB (amplitude SNR gain :math:`\sqrt{N}`).
    :ivar residual_rms: Root-mean-square of :attr:`residual`.
    :ivar comb_frequencies: Frequency axis of the comb-filter response, in
        Hz (from DC over a whole number of harmonics of ``1/T``).
    :ivar comb_response: Magnitude of the comb filter (McFadden Eq. 8) on
        :attr:`comb_frequencies`.
    """

    period_waveform: NDArray[np.float64]
    times: NDArray[np.float64]
    residual: NDArray[np.float64]
    n_averages: int
    samples_per_period: int
    period: float
    fs: float
    interpolated: bool
    noise_reduction_db: float
    residual_rms: float
    comb_frequencies: NDArray[np.float64]
    comb_response: NDArray[np.float64]

    @property
    def amplitude_snr_gain(self) -> float:
        r"""Amplitude signal-to-noise improvement :math:`\sqrt{N}`."""
        return float(np.sqrt(self.n_averages))

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the averaged waveform and the comb-filter magnitude.

        With ``ax`` given, only the averaged-waveform panel is drawn on it.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_synchronous_average

        check_language(language)
        return plot_synchronous_average(self, ax=ax, language=language, **kwargs)


def _samples_per_period(fs: float, period: float) -> tuple[float, int]:
    """Exact and integer samples per period, with an integer-fit check."""
    samples = fs * period
    rounded = round(samples)
    if rounded < 2:
        raise ValueError(
            "'period' is too short for the sample rate: it must span at "
            "least 2 samples."
        )
    return samples, rounded


def _resolve_n_averages(
    n_averages: int | None, length: int, samples: float, m_int: int
) -> int:
    """Number of whole periods to average, validated against the record."""
    available = int(np.floor((length - m_int) / samples)) + 1
    if available < 1:
        raise ValueError(
            "The record is shorter than one period; nothing to average."
        )
    if n_averages is None:
        return available
    requested = int(n_averages)
    if requested < 1:
        raise ValueError("'n_averages' must be a positive integer.")
    if requested > available:
        raise ValueError(
            f"'n_averages' = {requested} exceeds the {available} whole "
            f"periods available in the record."
        )
    return requested


def _extract_period(
    x: NDArray[np.float64], start: float, m_int: int
) -> NDArray[np.float64]:
    """One period of ``m_int`` samples starting at fractional ``start``.

    The block is aligned to the integer grid by a band-limited fractional
    delay; an integer ``start`` (within :data:`_ALIGN_TOL`) is sliced
    directly, so an integer-period record stays exact.
    """
    i0 = round(start)
    frac = start - i0
    if abs(frac) < _ALIGN_TOL:
        return x[i0 : i0 + m_int]
    # fractional_delay(x, d) yields x(j - d); advancing by frac (d = -frac)
    # gives x(j + frac), so sample i0 + m carries x(i0 + frac + m).
    shifted = fractional_delay(x, -frac)
    return np.asarray(shifted[i0 : i0 + m_int], dtype=np.float64)


def _comb_grid(
    period: float, n_averages: int, n_harmonics: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Frequency axis (Hz) and comb-filter magnitude over the first teeth."""
    points = max(256, 200 * n_harmonics)
    freqs = np.linspace(0.0, n_harmonics / period, points)
    response = comb_filter_response(freqs, period, n_averages)
    return freqs, response


def time_synchronous_average(
    x: NDArray[np.float64] | list[float],
    fs: float,
    period: float,
    *,
    n_averages: int | None = None,
    n_harmonics: int = 8,
) -> SynchronousAverageResult:
    r"""Extract a periodic waveform of known period by time domain averaging.

    Ensemble-averages ``N`` successive periods of the record (McFadden
    Eq. 5) to reinforce the component synchronous with ``period`` and
    suppress asynchronous noise, whose residual standard deviation falls as
    :math:`1/\sqrt{N}`. When ``fs * period`` is an integer the periods are
    sliced
    directly and a noiseless periodic signal is recovered exactly; otherwise
    each period is aligned to a common integer grid by the band-limited
    fractional delay of :func:`~phonometry.signals.test_signals.fractional_delay`
    and recovered within that interpolation error.

    :param x: Signal, 1-D, containing the periodic component plus noise.
    :param fs: Sample rate, in Hz.
    :param period: Known repetition period ``T``, in seconds (e.g. one
        revolution of a rotating machine).
    :param n_averages: Number of whole periods to average (default: as many
        as the record holds). Choosing ``N`` so that :math:`N q` is an
        integer
        places a comb node on an interfering tone at order ``q`` and
        maximises its rejection (McFadden's revised-model result).
    :param n_harmonics: Number of harmonics of ``1/T`` spanned by the
        returned comb-filter response (default 8).
    :return: A :class:`SynchronousAverageResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_1d_finite(x, "x")
    fs_v = _positive(fs, "fs")
    period_v = _positive(period, "period")
    n_harmonics_v = int(n_harmonics)
    if n_harmonics_v < 1:
        raise ValueError("'n_harmonics' must be a positive integer.")

    samples, m_int = _samples_per_period(fs_v, period_v)
    n_avg = _resolve_n_averages(n_averages, xa.size, samples, m_int)

    interpolated = abs(samples - round(samples)) >= _ALIGN_TOL
    blocks = np.empty((n_avg, m_int), dtype=np.float64)
    for n in range(n_avg):
        blocks[n] = _extract_period(xa, n * samples, m_int)

    period_waveform = np.asarray(blocks.mean(axis=0), dtype=np.float64)
    residual = np.asarray((blocks - period_waveform).ravel(), dtype=np.float64)
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    noise_reduction_db = 10.0 * float(np.log10(n_avg))

    comb_freqs, comb_response = _comb_grid(period_v, n_avg, n_harmonics_v)
    # The averaged samples sit on the 1/fs sampling grid (each block is
    # aligned back to it before averaging), not on a T/M angular grid.
    times = np.arange(m_int, dtype=np.float64) / fs_v

    return SynchronousAverageResult(
        period_waveform=period_waveform,
        times=times,
        residual=residual,
        n_averages=n_avg,
        samples_per_period=m_int,
        period=period_v,
        fs=fs_v,
        interpolated=bool(interpolated),
        noise_reduction_db=noise_reduction_db,
        residual_rms=residual_rms,
        comb_frequencies=comb_freqs,
        comb_response=comb_response,
    )
