#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Phase utilities: minimum phase, group delay and excess phase.

For a causal, stable, minimum-phase system the log-magnitude and the phase
of the frequency response form a Hilbert-transform pair (Bendat & Piersol,
*Random Data*, 4th ed., 2010, Sec. 13.1.4; Oppenheim & Schafer,
*Discrete-Time Signal Processing*, Ch. 12): the phase is fully determined
by :math:`\lvert H(f) \rvert`. This module computes that **minimum
phase** with the real cepstrum -- fold the inverse transform of
:math:`\ln \lvert H \rvert` onto positive
quefrencies and transform back -- and derives the standard decomposition
of a measured response,

.. math::

   H(f) = H_{\mathrm{min}}(f) \, H_{\mathrm{ap}}(f)
   \quad \text{with} \quad
   \phi_{\mathrm{excess}}(f) = \phi(f) - \phi_{\mathrm{min}}(f)

whose all-pass **excess phase** collects pure latency and any genuine
non-minimum-phase behaviour (reflections, non-invertible zeros): the part
of the phase no stable causal equalizer can remove. The **group delay**
:math:`\tau_\mathrm{g} = -(1/2\pi) \, d\phi/df` is estimated from the unwrapped
phase.

Sampling precautions (documented, and the reason for the ``oversample``
padding): the estimate operates on a *uniformly sampled* one-sided response
(DC to Nyquist inclusive, the layout of ``numpy.fft.rfft`` and of
:func:`phonometry.room.impulse_response` spectra). The real cepstrum of the
sampled log-magnitude is time-aliased when the grid is coarse relative to
how sharp the response is, so the magnitude is resampled onto an
``oversample`` times denser grid by exact trigonometric (zero-padded
even-sequence) interpolation before the logarithm and the phase is read
back on the original bins. Magnitude zeros (a response that vanishes at
DC or Nyquist, e.g. a band-pass) are not minimum-phase-representable:
bins below a relative floor are clipped, and the reconstruction is only
accurate away from them. On a strictly minimum-phase response (all poles
and zeros inside the unit circle) sampled on an adequate grid the
reconstruction matches the true phase to better than ``1e-12`` rad -- the
tolerance the biquad oracle pins in the tests; near-circle zeros on
coarse grids degrade it and are what ``oversample`` mitigates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .cepstrum import _fold_causal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "PhaseDecompositionResult",
    "excess_phase",
    "group_delay",
    "minimum_phase",
    "phase_decomposition",
]

#: Relative magnitude floor (about -300 dB) applied before the logarithm so
#: exact zeros (band edges of a band-pass, notch bottoms) stay finite. Bins
#: at the floor are not minimum-phase-representable; see the module note.
_MAGNITUDE_FLOOR = 1e-15

#: Fewest response bins a phase estimate makes sense on.
_MIN_BINS = 8


def _validate_response(
    response: NDArray[np.complex128] | NDArray[np.float64] | list[float],
    *,
    name: str = "response",
) -> NDArray[np.complex128]:
    arr = np.asarray(response)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be one-dimensional.")
    if arr.size < _MIN_BINS:
        raise ValueError(f"'{name}' must have at least {_MIN_BINS} frequency bins.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' must be finite.")
    out = arr.astype(np.complex128, copy=False)
    if not np.any(np.abs(out) > 0.0):
        raise ValueError(f"'{name}' must not be identically zero.")
    return out


def _validate_oversample(oversample: int) -> int:
    factor = int(oversample)
    if factor < 1:
        raise ValueError("'oversample' must be a positive integer.")
    return factor


def _trig_oversample(
    magnitude: NDArray[np.float64], factor: int
) -> NDArray[np.float64]:
    """Resample a one-sided magnitude onto a ``factor`` times denser grid.

    Exact trigonometric interpolation: the magnitude's inverse transform is
    a real even sequence, which is zero-padded in the middle (splitting the
    Nyquist-quefrency sample symmetrically) and transformed back. The
    original bins are reproduced exactly; between them the interpolant is
    the unique band-limited extension, which is what keeps the cepstral
    folding consistent on the denser grid.
    """
    if factor == 1:
        return magnitude
    n = 2 * (magnitude.size - 1)
    nfft = n * factor
    sequence = np.fft.irfft(magnitude, n)
    padded = np.zeros(nfft, dtype=np.float64)
    half = n // 2
    padded[:half] = sequence[:half]
    padded[half] = 0.5 * sequence[half]
    padded[nfft - half] = 0.5 * sequence[half]
    padded[nfft - half + 1 :] = sequence[half + 1 :]
    return np.asarray(np.fft.rfft(padded).real, dtype=np.float64)


def minimum_phase(
    response: NDArray[np.complex128] | NDArray[np.float64] | list[float],
    *,
    oversample: int = 8,
) -> NDArray[np.complex128]:
    r"""
    Minimum-phase response with the magnitude of ``response``.

    Computes the phase that the Hilbert relation between log-magnitude and
    phase assigns to :math:`\lvert H(f) \rvert` (Bendat & Piersol
    Sec. 13.1.4) via the real cepstrum: the inverse transform of
    :math:`\ln \lvert H \rvert` is folded onto positive
    quefrencies (doubling them, keeping the ends; the folding core is
    shared with :mod:`phonometry.signals.cepstrum`) and transformed
    back, so ``exp`` of the result is the unique stable, causal, causally
    invertible response with that magnitude. The input phase, if any, is
    ignored: passing a plain magnitude array works.

    :param response: One-sided response (or magnitude), uniformly sampled
        from DC to Nyquist inclusive (``rfft`` layout).
    :param oversample: Cepstral anti-aliasing factor: the magnitude is
        resampled onto a grid this many times denser (exact trigonometric
        interpolation) before the log and the cepstrum (default 8). Raise
        it if the magnitude has very sharp features -- near-circle zeros,
        deep notches -- relative to the grid.
    :return: Complex minimum-phase response on the same bins (same
        magnitude, reconstructed phase).
    :raises ValueError: If the inputs are invalid.

    .. note::
        Magnitude zeros cannot be represented by a minimum-phase system;
        they are floored at 1e-15 of the peak and the phase near them is
        not reliable (see the module docstring).
    """
    resp = _validate_response(response)
    factor = _validate_oversample(oversample)
    magnitude = np.abs(resp)
    floor = float(np.max(magnitude)) * _MAGNITUDE_FLOOR

    bins = resp.size
    nfft = 2 * (bins - 1) * factor
    mag_fine = _trig_oversample(magnitude, factor)
    log_fine = np.log(np.maximum(mag_fine, floor))
    cepstrum = np.fft.irfft(log_fine, nfft)
    folded = _fold_causal(cepstrum)
    phase = np.imag(np.fft.rfft(folded))[::factor]
    return np.asarray(magnitude * np.exp(1j * phase), dtype=np.complex128)


def group_delay(
    response: NDArray[np.complex128] | list[float],
    fs: float,
) -> NDArray[np.float64]:
    r"""
    Group delay :math:`\tau_\mathrm{g}(f) = -(1/2\pi) \, d\phi/df` of a response.

    The phase is unwrapped and differentiated with second-order central
    differences (one-sided at the grid ends). The estimate is exact for a
    linear phase and accurate to :math:`O(df^2)` otherwise; the unwrapping
    requires the response to be sampled densely enough that the phase
    advances less than :math:`\pi` per bin -- a pure delay of ``D`` samples
    needs fewer than ``response.size`` of them, i.e. the underlying
    impulse response must fit the record the grid implies.

    :param response: One-sided complex response, uniformly sampled from DC
        to Nyquist inclusive (``rfft`` layout).
    :param fs: Sample rate the response grid corresponds to, in Hz.
    :return: Group delay per bin, in seconds.
    :raises ValueError: If the inputs are invalid.
    """
    resp = _validate_response(response)
    fs_v = float(fs)
    if not np.isfinite(fs_v) or fs_v <= 0.0:
        raise ValueError("'fs' must be a positive, finite number.")
    phase = np.unwrap(np.angle(resp))
    domega = 2.0 * np.pi * (fs_v / 2.0) / (resp.size - 1)
    return np.asarray(-np.gradient(phase, domega), dtype=np.float64)


def excess_phase(
    response: NDArray[np.complex128] | list[float],
    *,
    oversample: int = 8,
) -> NDArray[np.float64]:
    r"""
    Excess phase: measured phase minus the minimum phase of
    :math:`\lvert H \rvert`.

    :math:`\phi_{\mathrm{excess}} = \operatorname{unwrap}(\arg H) -
    \phi_{\mathrm{min}}` is the phase of the all-pass
    factor in :math:`H = H_{\mathrm{min}} H_{\mathrm{ap}}`: zero for a
    minimum-phase system,
    :math:`-2 \pi f t_0` for a pure latency :math:`t_0`, and it
    additionally bends
    wherever the response has non-minimum-phase zeros. Equalizing a
    response down to its excess phase is the realizability limit of any
    stable causal inverse filter.

    :param response: One-sided complex response, uniformly sampled from DC
        to Nyquist inclusive (``rfft`` layout).
    :param oversample: Cepstral anti-aliasing factor for the minimum-phase
        reconstruction (default 8).
    :return: Excess phase per bin, in radians (continuous, 0 at DC).
    :raises ValueError: If the inputs are invalid.
    """
    resp = _validate_response(response)
    reconstructed = minimum_phase(resp, oversample=oversample)
    measured = np.unwrap(np.angle(resp))
    measured = measured - measured[0]  # reference the branch at DC
    return np.asarray(measured - np.unwrap(np.angle(reconstructed)), dtype=np.float64)


@dataclass(frozen=True)
class PhaseDecompositionResult:
    r"""Minimum-phase / all-pass decomposition of a frequency response.

    :ivar frequencies: Frequency axis, in Hz.
    :ivar magnitude: :math:`\lvert H(f) \rvert` (shared by the measured
        and minimum-phase responses).
    :ivar phase: Measured phase, unwrapped and referenced to DC, rad.
    :ivar minimum_phase: Phase reconstructed from :math:`\lvert H \rvert`
        alone, rad.
    :ivar excess_phase: ``phase - minimum_phase``: the all-pass part, rad.
    :ivar group_delay: Group delay of the measured response, s.
    :ivar excess_group_delay: Group delay of the all-pass part alone, s
        (constant :math:`t_0` for a pure latency).
    :ivar minimum_phase_response: Complex minimum-phase response
        :math:`\lvert H \rvert \exp(j \, \text{minimum\_phase})`.
    :ivar fs: Sample rate of the underlying record, in Hz.
    """

    frequencies: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    phase: NDArray[np.float64]
    minimum_phase: NDArray[np.float64]
    excess_phase: NDArray[np.float64]
    group_delay: NDArray[np.float64]
    excess_group_delay: NDArray[np.float64]
    minimum_phase_response: NDArray[np.complex128]
    fs: float

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        r"""Plot the magnitude, the phase decomposition and the group delay.

        Three stacked panels: :math:`\lvert H \rvert` in dB, the
        measured / minimum /
        excess phases, and the total and excess group delays. With ``ax``
        given, only the phase panel is drawn on it.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_phase_decomposition

        check_language(language)
        return plot_phase_decomposition(self, ax=ax, language=language, **kwargs)


def phase_decomposition(
    response: NDArray[np.complex128] | list[float],
    fs: float,
    *,
    oversample: int = 8,
) -> PhaseDecompositionResult:
    """
    Decompose a response into its minimum-phase and all-pass parts.

    Bundles :func:`minimum_phase`, :func:`excess_phase` and
    :func:`group_delay` on one frequency axis: the minimum phase carries
    everything an equalizer can invert, the excess phase is the residual
    all-pass (latency plus non-minimum-phase zeros), and the two group
    delays quantify both in seconds.

    :param response: One-sided complex response, uniformly sampled from DC
        to Nyquist inclusive (``rfft`` layout) -- e.g.
        ``numpy.fft.rfft(ir)`` of a measured impulse response.
    :param fs: Sample rate of the underlying record, in Hz.
    :param oversample: Cepstral anti-aliasing factor (default 8).
    :return: A :class:`PhaseDecompositionResult`.
    :raises ValueError: If the inputs are invalid.
    """
    resp = _validate_response(response)
    fs_v = float(fs)
    if not np.isfinite(fs_v) or fs_v <= 0.0:
        raise ValueError("'fs' must be a positive, finite number.")
    reconstructed = minimum_phase(resp, oversample=oversample)
    measured = np.unwrap(np.angle(resp))
    measured = measured - measured[0]
    phase_min = np.unwrap(np.angle(reconstructed))
    phase_exc = measured - phase_min
    tau = group_delay(resp, fs_v)
    tau_min = group_delay(reconstructed, fs_v)
    frequencies = np.linspace(0.0, fs_v / 2.0, resp.size)
    return PhaseDecompositionResult(
        frequencies=frequencies,
        magnitude=np.abs(resp),
        phase=measured,
        minimum_phase=phase_min,
        excess_phase=phase_exc,
        group_delay=tau,
        excess_group_delay=np.asarray(tau - tau_min, dtype=np.float64),
        minimum_phase_response=reconstructed,
        fs=fs_v,
    )
