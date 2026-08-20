#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Thomson multitaper spectral estimation (Percival & Walden 1993, Ch. 7).

Thomson's multitaper estimator (Thomson 1982), as developed in Percival &
Walden, *Spectral Analysis for Physical Applications* (1993, Chapter 7),
is the whole-record alternative to the Welch segment averaging of
:mod:`phonometry.signals.spectra`: ``K`` orthogonal discrete prolate
spheroidal (Slepian) tapers of time-half-bandwidth ``NW`` produce ``K``
nearly uncorrelated eigenspectra whose (adaptively weighted) average
carries about :math:`2K` chi-square degrees of freedom *without*
splitting the record - the estimator of choice for short records where
Welch would leave too few segments.

The statistical apparatus is the same as that of the Welch estimators,
one step further: the chi-square confidence interval keeps the form of
Bendat & Piersol Eq. 8.163, but its degrees of freedom are
per-frequency, because Thomson's adaptive weights (P&W Eq. 368a)
downweight leakage-prone tapers wherever the spectrum is locally weak
and that costs degrees of freedom there (P&W Eq. 370b). Calibration -
no detrending, one-sided ``'density'`` scaling integrating to the signal
power, ``'spectrum'`` scaling reading :math:`A^2/2` at a tone's peak -
matches :mod:`phonometry.signals.spectra` exactly, so the two estimators
of the same record are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..io._resolve import resolve_fs
from .spectra import (
    _positive,
    _validate_confidence,
    _validate_scaling,
    _validate_signal,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..io._signal import Signal

__all__ = [
    "MultitaperSpectralDensityResult",
    "multitaper_psd",
]

#: Relative tolerance of the adaptive-weight fixed-point iteration.
_ADAPTIVE_RTOL = 1e-10
#: Iteration cap for the adaptive weights (Percival & Walden report that
#: two passes usually suffice; the fixed point is reached long before this).
_ADAPTIVE_MAX_ITER = 100


@dataclass(frozen=True)
class MultitaperSpectralDensityResult:
    r"""Thomson multitaper spectral density (Percival & Walden Ch. 7).

    One whole-record estimate from ``K`` orthogonal Slepian (dpss) tapers:
    the ``K`` eigenspectra are nearly uncorrelated, so their weighted
    average trades the two chi-square degrees of freedom of a periodogram
    for about :math:`2K` - without segmenting the record as Welch's
    method does. The chi-square machinery mirrors
    :class:`~phonometry.signals.spectra.SpectralDensityResult`, but here
    the degrees of freedom are
    per-frequency: Thomson's adaptive weights (P&W Eq. 368a) downweight
    leakage-prone tapers wherever the spectrum is locally weak, which
    costs degrees of freedom there (P&W Eq. 370b).

    :ivar frequencies: One-sided frequency axis, in Hz.
    :ivar psd: Multitaper spectral density
        :math:`\hat{S}^{(mt)}(f)` (units²/Hz for
        ``'density'`` scaling, units² for ``'spectrum'``).
    :ivar ci_lower: Lower chi-square confidence bound,
        :math:`\nu \hat{S}/\chi^2_{\nu;\alpha/2}` with the per-frequency
        :math:`\nu` (the same interval
        form as B&P Eq. 8.163, with :math:`\nu` from P&W Eq. 370b).
    :ivar ci_upper: Upper chi-square confidence bound.
    :ivar confidence: Confidence level of the interval (e.g. ``0.95``).
    :ivar degrees_of_freedom: Per-frequency equivalent chi-square degrees
        of freedom
        :math:`\nu(f) = 2 \left( \sum_k d_k \right)^2 / \sum_k d_k^2`
        with :math:`d_k = b_k^2(f) \lambda_k`
        (P&W Eq. 370b);
        :math:`2K \left( \sum \lambda_k / K \right)^2 K /
        \sum \lambda_k^2 \approx 2K` for unity weights.
        The DC bin - and the Nyquist bin for an even record length -
        carries half (a single real Fourier component per eigenspectrum).
    :ivar random_error: Per-frequency normalized random error
        :math:`\varepsilon[\hat{S}^{(mt)}] = \sqrt{2/\nu}`
        (:math:`\approx 1/\sqrt{K}`), the multitaper counterpart of
        B&P Eq. 8.158.
    :ivar weights: Normalized combination weights
        :math:`d_k(f) / \sum_j d_j(f)`,
        shape ``(n_tapers, n_frequencies)``. Adaptive weighting makes them
        frequency dependent; they converge to :math:`\approx 1/K` where
        the spectrum
        is locally white (exactly uniform weights would be
        :math:`\lambda_k / \sum \lambda_j`).
    :ivar eigenvalues: Concentration ratios :math:`\lambda_k(N, W)` of
        the tapers -
        the fraction of each taper's spectral-window energy inside the
        design band :math:`[-W, W]` (P&W Section 7.1; near unity for
        :math:`k < 2NW`).
    :ivar time_half_bandwidth: The duration x half-bandwidth product
        ``NW`` (dimensionless; :math:`W = NW/(N \Delta t)`).
    :ivar n_tapers: Number of tapers ``K`` averaged.
    :ivar resolution_bandwidth: The resolution bandwidth :math:`2W` of
        the estimator, in Hz - the multitaper analog of the Welch
        :math:`B_\mathrm{e}`
        (P&W call :math:`2W` *the* natural resolution measure of the
        method).
    :ivar adaptive: Whether Thomson's adaptive weights were used.
    :ivar scaling: ``'density'`` or ``'spectrum'``.
    """

    frequencies: NDArray[np.float64]
    psd: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    confidence: float
    degrees_of_freedom: NDArray[np.float64]
    random_error: NDArray[np.float64]
    weights: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    time_half_bandwidth: float
    n_tapers: int
    resolution_bandwidth: float
    adaptive: bool
    scaling: str

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the multitaper density in dB with its confidence band.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_multitaper_spectral_density

        check_language(language)
        return plot_multitaper_spectral_density(
            self, ax=ax, language=language, **kwargs
        )


def _validate_multitaper_params(
    n: int, time_half_bandwidth: float, n_tapers: int | None
) -> tuple[float, int]:
    r"""Validate ``NW`` and ``K`` against the record length.

    ``K`` defaults to :math:`2NW - 1` and is capped at the Shannon number
    :math:`2NW`: beyond it the taper concentrations collapse (P&W
    Section 7.1) and the extra eigenspectra are pure leakage.
    """
    nw = _positive(time_half_bandwidth, "time_half_bandwidth")
    if nw < 1.0 or nw >= n / 2.0:
        raise ValueError(
            f"'time_half_bandwidth' must be in [1, n/2) (got {nw:g} for {n} samples)."
        )
    shannon = int(2.0 * nw)
    k = shannon - 1 if n_tapers is None else int(n_tapers)
    if not 1 <= k <= shannon:
        raise ValueError(
            "'n_tapers' must be between 1 and the Shannon number "
            f"2·NW = {shannon} (got {k})."
        )
    return nw, k


def _dpss_eigenspectra(
    x: NDArray[np.float64], fs: float, nw: float, k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Eigenspectra :math:`\hat{S}_k(f)` on the one-sided rfft grid.

    Two-sided scale. The Slepian tapers come from
    :func:`scipy.signal.windows.dpss`
    (unit-energy normalization, :math:`\sum_t h_{tk}^2 = 1`, exactly P&W
    Eq. 334b);
    the estimator built on them is P&W Eq. 333: each eigenspectrum is the
    direct spectral estimator

    .. math::

       \hat{S}_k(f) = \Delta t \left\lvert \sum_t h_{tk} x_t
       e^{-i 2 \pi f t \Delta t} \right\rvert^2

    Returns ``(eigenspectra, eigenvalues, taper_dc_gains_squared)`` where
    the last term is :math:`\left( \sum_t h_{tk} \right)^2`, needed for
    the ``'spectrum'`` scaling.
    """
    from scipy.signal import windows as sp_windows

    tapers, eigenvalues = sp_windows.dpss(x.size, nw, Kmax=k, return_ratios=True)
    tapers = np.atleast_2d(np.asarray(tapers, dtype=np.float64))
    yk = np.fft.rfft(tapers * x, axis=-1)
    sk = (np.abs(yk) ** 2) / fs
    # Near-unity concentrations can exceed 1 by a few ulp (the numerical
    # hazard P&W document around their Table 380); clip so the broad-band
    # bias term (1-lambda) of the adaptive weights can never turn negative.
    lam = np.clip(np.atleast_1d(np.asarray(eigenvalues, dtype=np.float64)), 0.0, 1.0)
    return sk, lam, np.sum(tapers, axis=-1) ** 2


def _adaptive_multitaper_weights(
    sk: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
    power_density: float,
) -> NDArray[np.float64]:
    r"""Thomson's adaptive weights :math:`d_k(f) = b_k^2(f) \lambda_k`.

    P&W Section 7.4: fixed-point iteration of P&W Eq. 368a,
    :math:`b_k(f) = S(f) / (\lambda_k S(f) +
    (1 - \lambda_k) \sigma^2 \Delta t)`, through the weighted
    estimator of Eq. 370a,
    :math:`\hat{S} = \sum_k d_k \hat{S}_k / \sum_k d_k`, seeded with the
    eigenvalue-weighted average of the two lowest-order eigenspectra
    (P&W's recipe). :math:`\sigma^2 \Delta t` - the flat density carrying
    the process
    power - is the broad-band bias scale: high-order tapers (smaller
    :math:`\lambda_k`) are downweighted wherever the local spectrum falls
    below the
    leakage that the total power could push through their sidelobes.
    """
    lam = eigenvalues[:, np.newaxis]
    k_seed = min(2, sk.shape[0])
    s = np.sum(lam[:k_seed] * sk[:k_seed], axis=0) / float(np.sum(eigenvalues[:k_seed]))
    d = eigenvalues[:, np.newaxis] * np.ones_like(sk)
    for _ in range(_ADAPTIVE_MAX_ITER):
        # The denominator is bounded below by (1-lambda_k)*sigma^2*dt,
        # positive for every taper with lambda_k < 1; the floor covers the
        # degenerate lambda_k == 1 taper meeting an all-zero bin.
        b = s / np.maximum(lam * s + (1.0 - lam) * power_density, np.finfo(float).tiny)
        d = b * b * lam
        dsum = np.sum(d, axis=0)
        s_new = np.divide(np.sum(d * sk, axis=0), dsum, out=s.copy(), where=dsum > 0.0)
        # Relative change per bin; a bin whose new estimate is exactly zero is
        # trivially converged, so it contributes 0 to the maximum rather than
        # dividing by a clamped subnormal.
        rel = np.divide(
            np.abs(s_new - s), s_new, out=np.zeros_like(s_new), where=s_new > 0.0
        )
        delta = float(np.max(rel))
        s = s_new
        if delta < _ADAPTIVE_RTOL:
            break
    # A bin where every eigenspectrum is exactly zero leaves d = 0 there;
    # fall back to the eigenvalue weights so the combination stays defined.
    return np.where(np.sum(d, axis=0) > 0.0, d, lam)


def _multitaper_dof(d: NDArray[np.float64], nyquist_bin: bool) -> NDArray[np.float64]:
    r"""Equivalent degrees of freedom
    :math:`\nu(f) = 2 \left( \sum_k d_k \right)^2 / \sum_k d_k^2`.

    P&W Eq. 370b: each eigenspectrum contributes two chi-square degrees
    of freedom (one complex Fourier component), combined with weights
    :math:`d_k`. The DC bin - and the Nyquist bin when the record length
    is even - has a single real component per eigenspectrum, so its
    :math:`\nu` is halved (same convention as the Welch estimators of
    :mod:`phonometry.signals.spectra`).
    """
    dof = 2.0 * np.sum(d, axis=0) ** 2 / np.sum(d * d, axis=0)
    dof[0] /= 2.0
    if nyquist_bin:
        dof[-1] /= 2.0
    return np.asarray(dof, dtype=np.float64)


def _chi2_interval_pointwise(
    psd: NDArray[np.float64],
    dof: NDArray[np.float64],
    confidence: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Chi-square interval with per-frequency degrees of freedom.

    Same form as B&P Eq. 8.163
    (:math:`\nu \hat{S}/\chi^2_{\nu;\alpha/2} \le S \le
    \nu \hat{S}/\chi^2_{\nu;1-\alpha/2}`)
    with the P&W Eq. 370b :math:`\nu(f)` of the multitaper estimator.
    """
    from scipy import stats as sp_stats

    alpha = 1.0 - confidence
    lower = (
        psd * dof / np.asarray(sp_stats.chi2.isf(alpha / 2.0, dof), dtype=np.float64)
    )
    upper = (
        psd
        * dof
        / np.asarray(sp_stats.chi2.isf(1.0 - alpha / 2.0, dof), dtype=np.float64)
    )
    return lower, upper


def multitaper_psd(
    x: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    time_half_bandwidth: float = 4.0,
    n_tapers: int | None = None,
    adaptive: bool = True,
    scaling: Literal["density", "spectrum"] = "density",
    confidence: float = 0.95,
) -> MultitaperSpectralDensityResult:
    r"""Thomson multitaper spectral density with chi-square interval.

    Implements the multitaper estimator of Thomson (1982) as developed in
    Percival & Walden (1993, Chapter 7): the record is multiplied by
    ``K`` orthogonal discrete prolate spheroidal (Slepian) data tapers -
    the sequences that maximize spectral concentration in the design band
    :math:`[-W, W]`, computed by :func:`scipy.signal.windows.dpss` - and
    the
    ``K`` resulting eigenspectra (P&W Eq. 333) are averaged. Because the
    tapers are orthogonal the eigenspectra are nearly uncorrelated, so the
    average has about :math:`2K` chi-square degrees of freedom and
    :math:`1/K` of
    the periodogram's variance *without* segmenting the record: the
    estimator of choice for short records, where Welch's method
    (:func:`~phonometry.signals.spectra.power_spectral_density`) would
    leave too few segments.

    With ``adaptive=True`` (default) the eigenspectra are combined with
    Thomson's frequency-dependent weights (P&W Eqs. 368a/370a, iterated to
    convergence): wherever the local spectrum is weak relative to the
    broad-band leakage each taper could carry, the leakier high-order
    tapers are downweighted, trading degrees of freedom (Eq. 370b) for
    leakage protection in high-dynamic-range spectra. The broadband
    :math:`\sigma^2` driving the weights is
    :math:`\operatorname{mean}(x^2)` with no mean removal,
    consistent with the no-detrending calibration below. For a locally
    white spectrum the weights converge to uniform and nothing is lost.
    With ``adaptive=False`` the eigenvalue-weighted average of P&W
    Eq. 369a is returned.

    Calibration matches the Welch estimators of
    :mod:`phonometry.signals.spectra` exactly: no
    detrending, ``'density'`` scaling integrates to the signal power
    (units²/Hz, one-sided) and ``'spectrum'`` scaling reads
    :math:`A^2/2` at
    the peak of a sinusoid of amplitude ``A`` (the tone calibration is
    exact for the taper set in use, computed from the taper DC gains
    :math:`\left( \sum_t h_{tk} \right)^2`; a tone's power in
    ``'density'`` scaling is spread over
    the resolution bandwidth :math:`2W`).

    :param x: Signal, 1-D (used whole; no segmentation). Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the density and its
        confidence interval come out in Pa²/Hz, or Pa² for
        ``scaling='spectrum'``.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param time_half_bandwidth: Duration x half-bandwidth product ``NW``
        (dimensionless; default 4, P&W's worked choice). The design
        half-bandwidth is :math:`W = NW f_\mathrm{s} / N` Hz; larger ``NW``
        admits more
        tapers (lower variance) at the cost of resolution :math:`2W`.
    :param n_tapers: Number of tapers ``K``; ``None`` picks
        :math:`2 NW - 1`
        (all tapers with near-unity concentration, P&W Section 7.1). At
        most the Shannon number :math:`2 NW`.
    :param adaptive: Use Thomson's adaptive weights (default) or the
        eigenvalue-weighted average.
    :param scaling: ``'density'`` (units²/Hz) or ``'spectrum'`` (units²,
        sinusoid-peak reading).
    :param confidence: Confidence level for the chi-square interval.
    :return: A :class:`MultitaperSpectralDensityResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x")
    fs_v = _positive(resolve_fs(x, fs), "fs")
    scaling_v = _validate_scaling(scaling)
    conf = _validate_confidence(confidence)
    nw, k = _validate_multitaper_params(xa.size, time_half_bandwidth, n_tapers)
    # Broadband power sigma^2 for the adaptive weights: mean(x^2) without
    # mean removal, consistent with the module-wide no-detrending
    # calibration (a DC offset counts as signal power here too).
    power = float(np.mean(xa * xa))
    if power <= 0.0:
        raise ValueError("'x' must not be identically zero.")

    sk, eigenvalues, dc_gains_sq = _dpss_eigenspectra(xa, fs_v, nw, k)
    if adaptive:
        d = _adaptive_multitaper_weights(sk, eigenvalues, power / fs_v)
    else:
        # Eigenvalue-weighted average, cited in the docstring as P&W
        # Eq. 369a; the equation number is pending verification against
        # the book (source copy not yet acquired). The formula itself is
        # Thomson's standard lambda_k-weighted eigenspectrum average.
        d = eigenvalues[:, np.newaxis] * np.ones_like(sk)
    weights = d / np.sum(d, axis=0)
    psd = np.sum(weights * sk, axis=0)

    # Fold to one-sided; interior bins carry both spectral halves.
    even_n = xa.size % 2 == 0
    last = psd.size - 1 if even_n else psd.size
    psd[1:last] *= 2.0
    if scaling_v == "spectrum":
        # Tone calibration: a bin-centred sinusoid of amplitude A yields
        # a one-sided density peak (A²/2)·Δt·Σₖwₖ·(Σₜhₜₖ)², so dividing
        # by that factor makes the peak read A²/2 exactly.
        psd /= np.sum(weights * dc_gains_sq[:, np.newaxis], axis=0) / fs_v

    dof = _multitaper_dof(d, nyquist_bin=even_n)
    lower, upper = _chi2_interval_pointwise(psd, dof, conf)
    return MultitaperSpectralDensityResult(
        frequencies=np.fft.rfftfreq(xa.size, 1.0 / fs_v),
        psd=psd,
        ci_lower=lower,
        ci_upper=upper,
        confidence=conf,
        degrees_of_freedom=dof,
        random_error=np.sqrt(2.0 / dof),
        weights=weights,
        eigenvalues=eigenvalues,
        time_half_bandwidth=nw,
        n_tapers=k,
        resolution_bandwidth=2.0 * nw * fs_v / xa.size,
        adaptive=adaptive,
        scaling=scaling_v,
    )
