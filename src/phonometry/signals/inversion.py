#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Regularized spectral inversion with frequency-dependent regularization.

Inverting a measured transfer function -- to equalize a measurement
loudspeaker, flatten a microphone response or post-process an acquired
impulse response -- cannot be a plain reciprocal :math:`1/H(f)`: wherever
the
system radiates little energy (outside its passband, in deep notches) the
reciprocal explodes and the inverse filter amplifies nothing but noise.
Mueller & Massarani ("Transfer-Function Measurement with Sweeps", JAES
49(6), 2001, Secs. 3.1 and 4.5) therefore confine the inversion to the
transmission band: unity equalization in-band and a controlled, bounded
gain outside.

This module implements that behaviour with the frequency-dependent
Tikhonov regularization of Kirkeby & Nelson ("Digital Filter Design for
Inversion Problems in Sound Reproduction", JAES 47(7/8), 1999, Eq. (17)):

.. math::

   H_{\mathrm{inv}}(f)
   = \frac{\overline{H(f)}}{\lvert H(f) \rvert^2 + \epsilon(f)}

where the regularization profile :math:`\epsilon(f)` is small inside the
band :math:`[f_1, f_2]` (the filter equalizes to unity within an analytic
bound) and large outside (the out-of-band gain is capped at
:math:`1/(2\sqrt{\epsilon})`, the maximum of
:math:`x/(x^2 + \epsilon)`), with a smooth geometric cross-fade
over a transition zone bordering the band edges. A modeling delay makes
the mixed-phase inverse causal (Kirkeby & Nelson Sec. 2.4: the inverse of
a non-minimum-phase response is anticausal and must be delayed to be
realisable).

The closed forms above are the module's oracles: in-band the equalized
magnitude :math:`\lvert H \, H_{\mathrm{inv}} \rvert` deviates from unity
by exactly :math:`\epsilon/(\lvert H \rvert^2 + \epsilon)`, and
out-of-band the filter gain never exceeds the
:math:`1/(2\sqrt{\epsilon})` bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.utils import _typesignal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "InverseFilterResult",
    "regularized_inverse_filter",
]


@dataclass(frozen=True)
class InverseFilterResult:
    r"""A regularized inverse filter with its achieved equalization.

    Returned by :func:`regularized_inverse_filter`. The causal filter
    samples live in ``inverse`` (the equalized response arrives ``delay``
    samples late; :meth:`apply` compensates it). ``spectrum`` is the
    complex inverse spectrum *including* the modeling delay;
    ``regularization`` is the :math:`\epsilon(f)` profile actually used,
    and ``flatness_db`` reports how flat the equalized magnitude
    :math:`\lvert H(f) \, H_{\mathrm{inv}}(f) \rvert` actually is across
    the requested band.

    :ivar inverse: Inverse-filter samples (time domain, length ``n_fft``).
    :ivar frequencies: Frequency grid of the design, in Hz.
    :ivar spectrum: Complex inverse spectrum on :attr:`frequencies`,
        including the :math:`\exp(-j 2 \pi f \, \text{delay} / f_s)`
        modeling delay.
    :ivar response_spectrum: Complex spectrum of the measured response the
        filter was designed from, on the same grid.
    :ivar regularization: The frequency-dependent profile
        :math:`\epsilon(f)` (absolute units of
        :math:`\lvert H \rvert^2`).
    :ivar f_range: ``(f1, f2)`` band equalized to unity, in Hz.
    :ivar delay: Modeling delay of the filter, in samples.
    :ivar fs: Sample rate, in Hz.
    :ivar flatness_db: Largest deviation of the equalized magnitude
        :math:`20 \log_{10} \lvert H \, H_{\mathrm{inv}} \rvert` from 0 dB
        inside :math:`[f_1, f_2]`.
    :ivar max_gain_db: Largest filter gain *outside* the transition-padded
        band, peak-normalized as
        :math:`20 \log_{10}(\max \lvert H_{\mathrm{inv}} \rvert \cdot
        \text{peak}_h)` where ``peak_h`` is the peak of
        :math:`\lvert H \rvert` -- the achieved out-of-band boost
        the regularization allowed where the measurement carries no signal.
    """

    inverse: np.ndarray
    frequencies: np.ndarray
    spectrum: np.ndarray
    response_spectrum: np.ndarray
    regularization: np.ndarray
    f_range: tuple[float, float]
    delay: int
    fs: float
    flatness_db: float
    max_gain_db: float

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Return the inverse-filter samples as an array."""
        return np.asarray(self.inverse, dtype=dtype)

    def __len__(self) -> int:
        return int(self.inverse.shape[-1])

    @property
    def size(self) -> int:
        """Number of samples in the inverse filter."""
        return int(self.inverse.size)

    def apply(self, x: list[float] | np.ndarray) -> np.ndarray:
        """Equalize a signal with the inverse filter.

        Convolves ``x`` with the filter and removes the modeling delay, so
        the output is time-aligned with the input and has the same length.
        Feeding the response the filter was designed from returns (in-band)
        a band-limited unit impulse at sample 0.

        :param x: The signal to equalize (1-D).
        :return: The equalized, delay-compensated signal, ``len(x)``
            samples.
        """
        from scipy import signal as sp_signal

        arr = _typesignal(x)
        if arr.ndim != 1:
            raise ValueError("'x' must be one-dimensional.")
        full = sp_signal.fftconvolve(arr, self.inverse)
        return np.asarray(full[self.delay:self.delay + arr.size])

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        r"""Plot the measured, inverse and equalized magnitudes.

        One panel: :math:`\lvert H \rvert`,
        :math:`\lvert H_{\mathrm{inv}} \rvert` and the equalized product
        :math:`\lvert H \, H_{\mathrm{inv}} \rvert` in dB over
        log-frequency, with the equalized band
        shaded. Requires matplotlib (``pip install phonometry[plot]``).

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_inverse_filter

        check_language(language)
        return plot_inverse_filter(self, ax=ax, language=language, **kwargs)


def _regularization_profile(
    freqs: np.ndarray,
    f_range: tuple[float, float],
    eps_inside: float,
    eps_outside: float,
    transition_octaves: float,
) -> np.ndarray:
    r"""Frequency-dependent :math:`\epsilon(f)`: geometric cross-fade.

    :math:`\epsilon` equals ``eps_inside`` across all of
    :math:`[f_1, f_2]` and ``eps_outside`` beyond a
    ``transition_octaves``-wide zone bordering
    each edge; inside each zone the two levels are blended geometrically
    (linearly in :math:`\log \epsilon`) with a half-cosine, following the
    smooth transition-band advice of Kirkeby & Nelson (1999, Sec. 3.2).
    """
    f1, f2 = f_range
    ratio = 2.0 ** transition_octaves
    weight = np.zeros_like(freqs)  # 1 = in-band, 0 = out-of-band
    inside = (freqs >= f1) & (freqs <= f2)
    weight[inside] = 1.0
    below = (freqs >= f1 / ratio) & (freqs < f1)
    if np.any(below):
        x = np.log2(freqs[below] * ratio / f1) / transition_octaves
        weight[below] = 0.5 - 0.5 * np.cos(np.pi * x)
    above = (freqs > f2) & (freqs <= f2 * ratio)
    if np.any(above):
        x = np.log2(freqs[above] / f2) / transition_octaves
        weight[above] = 0.5 + 0.5 * np.cos(np.pi * x)
    log_eps = weight * np.log(eps_inside) + (1.0 - weight) * np.log(
        eps_outside
    )
    return np.asarray(np.exp(log_eps))


def _validated_response(response: Any) -> np.ndarray:
    """Validate the measured impulse response array."""
    h = _typesignal(np.asarray(response, dtype=np.float64))
    if h.ndim != 1:
        raise ValueError("'response' must be one-dimensional.")
    if h.size < 2:
        raise ValueError("'response' must have at least 2 samples.")
    if not np.all(np.isfinite(h)):
        raise ValueError("'response' must be finite.")
    return h


def _validated_band(
    f_range: tuple[float, float], fs: float
) -> tuple[float, float]:
    """Validate the equalization band against the sample rate."""
    f1, f2 = float(f_range[0]), float(f_range[1])
    if f1 <= 0.0:
        raise ValueError("f_range[0] must be positive")
    if f2 <= f1:
        raise ValueError("f_range[1] must be greater than f_range[0]")
    if f2 > fs / 2.0:
        raise ValueError("f_range[1] must not exceed the Nyquist frequency")
    return f1, f2


def _resolve_fs(response: Any, fs: float | None) -> float:
    """Take ``fs`` from the argument or from a result carrying one."""
    if fs is None:
        fs = getattr(response, "fs", None)
    if fs is None:
        raise ValueError(
            "'fs' is required (pass it explicitly, or pass a result object "
            "that carries a sample rate)"
        )
    fs_v = float(fs)
    if fs_v <= 0.0:
        raise ValueError("fs must be positive")
    return fs_v


def regularized_inverse_filter(
    response: list[float] | np.ndarray | Any,
    fs: float | None = None,
    *,
    f_range: tuple[float, float],
    regularization_inside: float = 1e-6,
    regularization_outside: float = 1.0,
    transition_octaves: float = 1.0 / 3.0,
    n_fft: int | None = None,
    delay: int | None = None,
) -> InverseFilterResult:
    r"""
    Design a regularized inverse filter for a measured impulse response.

    Computes the frequency-dependent Tikhonov inverse of Kirkeby & Nelson
    (JAES 47(7/8), 1999, Eq. (17)),

    .. math::

       H_{\mathrm{inv}}(f)
       = \frac{\overline{H(f)}}{\lvert H(f) \rvert^2 + \epsilon(f)}

    with :math:`\epsilon(f)` small across :math:`[f_1, f_2]` and large
    outside, so the filter equalizes the response to unity in-band while
    the out-of-band gain stays bounded by
    :math:`1/(2\sqrt{\epsilon_{\mathrm{outside}}})` -- the behaviour
    Mueller & Massarani (JAES 49(6), 2001, Secs. 3.1/4.5) obtain by
    band-passing the plain inverse. A modeling delay of ``delay`` samples
    (default half the FFT block) shifts the generally anticausal inverse of
    a mixed-phase response into a causal filter (Kirkeby & Nelson,
    Sec. 2.4).

    Both regularization levels are *relative* to the peak of
    :math:`\lvert H \rvert^2`
    (like the scalar ``regularization`` of :func:`phonometry.impulse_response`,
    which this generalises): in-band the equalized magnitude deviates from
    unity by at most ``regularization_inside * max|H|**2 / min|H|**2`` --
    the analytic residue
    :math:`\epsilon/(\lvert H \rvert^2 + \epsilon)` -- and the achieved
    figure is reported as :attr:`InverseFilterResult.flatness_db`.

    Use the result's :meth:`InverseFilterResult.apply` to equalize
    recordings (or the excitation, for pre-emphasis) and read
    :attr:`InverseFilterResult.spectrum` to apply it spectrally.

    :param response: Measured impulse response (1-D array), or an
        :class:`phonometry.ImpulseResponseResult` from the sweep/MLS/Golay
        front ends (its sample rate is used when ``fs`` is omitted).
    :param fs: Sample rate in Hz. Optional when ``response`` carries one.
    :param f_range: ``(f1, f2)`` band, in Hz, over which the response is
        equalized to unity. Choose it inside the band actually excited and
        radiated; inverting unexcited regions only amplifies noise.
    :param regularization_inside: In-band regularization, as a fraction of
        the peak spectral power :math:`\max \lvert H \rvert^2`.
        Default 1e-6.
    :param regularization_outside: Out-of-band regularization, same units.
        Default 1.0, which caps the out-of-band gain at 6 dB below the
        peak-normalised unity (:math:`1/(2\sqrt{1}) = 0.5`).
    :param transition_octaves: Width of the geometric cross-fade between
        the two regularization levels, in octaves outside each band edge.
        Default 1/3.
    :param n_fft: FFT block length of the design (also the filter length).
        Default: the next power of two of ``2*len(response)``, so the
        circular design has room for the anticausal (delayed) part.
    :param delay: Modeling delay in samples. Default ``n_fft // 2``.
    :return: An :class:`InverseFilterResult`.
    """
    h = _validated_response(response)
    fs_v = _resolve_fs(response, fs)
    f1, f2 = _validated_band(f_range, fs_v)
    if regularization_inside <= 0.0 or regularization_outside <= 0.0:
        raise ValueError("both regularization levels must be positive")
    if transition_octaves <= 0.0:
        raise ValueError("transition_octaves must be positive")

    size = n_fft if n_fft is not None else 2 ** int(np.ceil(np.log2(2 * h.size)))
    if size < h.size:
        raise ValueError("n_fft must be at least the response length")
    lag = size // 2 if delay is None else int(delay)
    if not 0 <= lag < size:
        raise ValueError("delay must be in [0, n_fft)")

    freqs = np.asarray(np.fft.rfftfreq(size, 1.0 / fs_v), dtype=np.float64)
    if not np.any((freqs >= f1) & (freqs <= f2)):
        raise ValueError(
            "no frequency bin falls within f_range; increase n_fft or widen "
            "the band."
        )
    spectrum_h = np.fft.rfft(h, n=size)
    power = np.abs(spectrum_h) ** 2
    scale = float(np.max(power))
    if scale <= 0.0:
        raise ValueError("'response' must not be identically zero.")
    eps = _regularization_profile(
        freqs,
        (f1, f2),
        regularization_inside * scale,
        regularization_outside * scale,
        transition_octaves,
    )
    inv_spectrum = np.conj(spectrum_h) / (power + eps)
    inv_spectrum = inv_spectrum * np.exp(-2j * np.pi * freqs * lag / fs_v)
    inverse = np.fft.irfft(inv_spectrum, n=size)

    equalized = np.abs(spectrum_h * inv_spectrum)
    band = (freqs >= f1) & (freqs <= f2)
    with np.errstate(divide="ignore"):
        flatness = float(np.max(np.abs(20.0 * np.log10(equalized[band]))))
    ratio = 2.0 ** transition_octaves
    outside = (freqs > 0.0) & ((freqs < f1 / ratio) | (freqs > f2 * ratio))
    gain = np.abs(inv_spectrum[outside]) if np.any(outside) else np.array([])
    peak_h = float(np.sqrt(scale))
    max_gain = (
        20.0 * np.log10(float(np.max(gain)) * peak_h)
        if gain.size
        else -np.inf
    )

    return InverseFilterResult(
        inverse=np.asarray(inverse),
        frequencies=freqs,
        spectrum=np.asarray(inv_spectrum),
        response_spectrum=np.asarray(spectrum_h),
        regularization=np.asarray(eps),
        f_range=(f1, f2),
        delay=lag,
        fs=fs_v,
        flatness_db=flatness,
        max_gain_db=max_gain,
    )
