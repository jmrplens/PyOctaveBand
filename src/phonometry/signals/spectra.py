#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Calibrated spectral-density estimation with statistical error analysis.

Welch-averaged auto- and cross-spectral density estimators that report,
alongside the spectrum itself, the statistical quality of the estimate,
following Bendat & Piersol, *Random Data: Analysis and Measurement
Procedures* (4th ed., 2010):

* the **number of averages**: the raw segment count and the effective number
  of independent averages :math:`n_\mathrm{d}` once the correlation between
  overlapped,
  tapered segments is accounted for (Section 11.5.2.2 and its Ref. 11,
  Welch 1967);
* the **normalized random error** of the autospectrum estimate,
  :math:`\varepsilon[\hat{G}_{xx}] = 1/\sqrt{n_\mathrm{d}}` (Eq. 8.158), and of
  the cross-spectrum magnitude and phase,
  :math:`\varepsilon[\lvert \hat{G}_{xy} \rvert]
  = 1/(\lvert \gamma_{xy} \rvert \sqrt{n_\mathrm{d}})` (Eq. 9.33) and
  :math:`\mathrm{s.d.}[\hat{\theta}_{xy}]
  = (1 - \gamma^2_{xy})^{1/2} /
  (\lvert \gamma_{xy} \rvert \sqrt{2 n_\mathrm{d}})` (Eq. 9.52);
* **chi-square confidence intervals** for the autospectrum: the sampling
  distribution is :math:`n \hat{G}_{xx}/G_{xx} \sim \chi^2_n` with
  :math:`n = 2 n_\mathrm{d}` degrees of freedom
  (Eq. 8.162), giving the interval
  :math:`n \hat{G}_{xx}/\chi^2_{n;\alpha/2} \le G_{xx} \le
  n \hat{G}_{xx}/\chi^2_{n;1-\alpha/2}` (Eq. 8.163);
* the **first-order resolution-bias error**:
  :math:`b[\hat{G}_{xx}] \approx (B_\mathrm{e}^2/24) \, G''_{xx}`
  (Eq. 8.139), which for a resonance peak of half-power bandwidth
  :math:`B_\mathrm{r}`
  becomes :math:`\varepsilon_\mathrm{b} \approx -(B_\mathrm{e}/B_\mathrm{r})^2/3` (Eq. 8.141) -
  exposed here as
  :func:`resolution_bias_error`;
* the **coherent output spectrum**
  :math:`G_{vv} = \gamma^2_{xy} G_{yy}` and the noise output
  spectrum :math:`G_{nn} = (1 - \gamma^2_{xy}) G_{yy}` of the
  single-input/single-output model
  (Eqs. 9.55-9.56), with the spectral signal-to-noise ratio
  :math:`\gamma^2/(1 - \gamma^2)` and the random error
  :math:`\varepsilon[\hat{G}_{vv}] = (2 - \gamma^2_{xy})^{1/2} /
  (\lvert \gamma_{xy} \rvert \sqrt{n_\mathrm{d}})` (Eq. 9.73).

The same Welch core (Hann taper and 50% overlap by default, ``detrend``
off so absolute calibration is preserved) also backs the H1/H2 frequency
response and coherence estimators of
:mod:`phonometry.electroacoustics.frequency_response` and the p-p intensity
probe of :mod:`phonometry.emission.intensity`.

A **fractional-octave smoothing** utility completes the module: a
constant-power rectangular kernel of 1/n-octave width in log-frequency
(the constant-percentage resolution bandwidth that Bendat & Piersol,
Section 8.5.3, recommend for resonant-response spectra), applicable to
power spectra, magnitude responses and dB curves. A flat spectrum is left
exactly unchanged.

The taper the ``window`` parameter selects is characterized, with the
figures of merit of Harris (1978), by
:func:`phonometry.signals.windows.window_metrics`; the whole-record
alternative to Welch segment averaging, Thomson's multitaper estimator,
is :func:`phonometry.signals.multitaper.multitaper_psd`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "CoherentOutputSpectrumResult",
    "CrossSpectralDensityResult",
    "SpectralDensityResult",
    "coherent_output_spectrum",
    "cross_spectral_density",
    "fractional_octave_smoothing",
    "power_spectral_density",
    "resolution_bias_error",
]

#: Default overlap fraction between Welch segments.
_DEFAULT_OVERLAP = 0.5
#: Minimum samples for a spectral estimate.
_MIN_SAMPLES = 32


# ---------------------------------------------------------------------------
# Shared Welch core (also used by electroacoustics.frequency_response and
# emission.intensity so every spectral density in the library is computed
# by exactly one code path).
# ---------------------------------------------------------------------------


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return scalar


def _validate_signal(
    x: NDArray[np.float64] | list[float],
    name: str,
    *,
    context: str = "a spectral estimate",
) -> NDArray[np.float64]:
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 1:
        raise ValueError(f"'{name}' must be one-dimensional.")
    if xa.size < _MIN_SAMPLES:
        raise ValueError(
            f"Signal too short for {context}: {xa.size} samples."
        )
    if not np.all(np.isfinite(xa)):
        raise ValueError(f"'{name}' must be finite.")
    return xa


def _validate_welch_params(
    n: int, fs: float, nperseg: int | None, overlap: float
) -> tuple[int, float]:
    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError("'overlap' must be in [0, 1).")
    seg = _default_nperseg(n, fs) if nperseg is None else int(nperseg)
    if seg < _MIN_SAMPLES or seg > n:
        raise ValueError(
            f"'nperseg' must be between {_MIN_SAMPLES} and the signal length."
        )
    return seg, float(overlap)


def _default_nperseg(n: int, fs: float) -> int:
    """Segment length for a bin spacing of at most 4 Hz (min 32 samples)."""
    nperseg = int(min(n, 2 ** int(np.ceil(np.log2(fs / 4.0)))))
    return max(nperseg, _MIN_SAMPLES)


def _noverlap_samples(nperseg: int, overlap: float) -> int:
    """Overlap fraction -> samples, capped one short of the segment."""
    return min(round(overlap * nperseg), nperseg - 1)


def _welch_autospectrum(
    x: NDArray[np.float64],
    fs: float,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
    scaling: str = "density",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One-sided Welch autospectrum ``(freqs, Gxx)``, no detrending."""
    from scipy import signal as sp_signal

    freqs, gxx = sp_signal.welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling=scaling,
    )
    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(gxx, dtype=np.float64),
    )


def _welch_cross_spectrum(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    fs: float,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
    scaling: str = "density",
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """One-sided Welch cross-spectrum ``(freqs, Gxy)``, no detrending."""
    from scipy import signal as sp_signal

    freqs, gxy = sp_signal.csd(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling=scaling,
    )
    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(gxy, dtype=np.complex128),
    )


def _welch_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    fs: float,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
    scaling: str = "density",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """One-sided ``(freqs, Gxy, Gxx, Gyy)`` from Welch-averaged segments."""
    freqs, gxy = _welch_cross_spectrum(x, y, fs, nperseg, noverlap, window, scaling)
    _, gxx = _welch_autospectrum(x, fs, nperseg, noverlap, window, scaling)
    _, gyy = _welch_autospectrum(y, fs, nperseg, noverlap, window, scaling)
    return freqs, gxy, gxx, gyy


def _segment_statistics(
    n: int, nperseg: int, noverlap: int, window: str, fs: float
) -> tuple[int, float, float]:
    r"""Averaging statistics of the Welch estimate.

    Returns ``(n_segments, n_averages, resolution_bandwidth)``:

    * ``n_segments`` - raw number of (possibly overlapped) segments averaged;
    * ``n_averages`` - effective number of *independent* averages
      :math:`n_\mathrm{d}`.
      Bendat & Piersol's error formulas assume :math:`n_\mathrm{d}` distinct
      records
      (Section 9.1.1); with overlapped, tapered segments (Section 11.5.2.2)
      the variance reduction follows their Ref. 11 (Welch 1967, Eq. 12):
      :math:`\mathrm{Var} \propto (1/k)
      \left[ 1 + 2 \sum_j (1 - j/k) \rho^2(j D) \right]`
      where :math:`\rho(s)` is the
      normalized correlation of the taper with itself shifted by the hop
      ``D``, so
      :math:`n_\mathrm{d} = k / \left( 1 + 2 \sum_j (1 - j/k)
      \rho^2(j D) \right)`. Without overlap
      this reduces to :math:`n_\mathrm{d} = k` exactly.
    * ``resolution_bandwidth`` - the effective noise bandwidth of the
      tapered segment,
      :math:`B_\mathrm{e} = f_\mathrm{s} \sum w^2 / \left( \sum w \right)^2`, the
      tapered-window analog
      of Bendat & Piersol's :math:`B_\mathrm{e} \approx 1/T` (Eq. 8.160).
    """
    from scipy import signal as sp_signal

    step = nperseg - noverlap
    k = (n - noverlap) // step
    w = np.asarray(sp_signal.get_window(window, nperseg), dtype=np.float64)
    wsum2 = float(np.sum(w * w))
    benbw = fs * wsum2 / float(np.sum(w)) ** 2

    if noverlap == 0 or k == 1:
        return k, float(k), benbw

    # Normalized taper correlation at multiples of the hop.
    max_shift = (nperseg - 1) // step
    corr = 0.0
    for j in range(1, min(k, max_shift + 1)):
        shift = j * step
        rho = float(np.dot(w[: nperseg - shift], w[shift:])) / wsum2
        corr += (1.0 - j / k) * rho * rho
    nd = k / (1.0 + 2.0 * corr)
    return k, float(nd), benbw


def _chi2_bounds(dof: float, confidence: float) -> tuple[float, float]:
    r"""Interval factors :math:`n/\chi^2_{n;\alpha/2}` and
    :math:`n/\chi^2_{n;1-\alpha/2}` (Eq. 8.163).

    B&P's :math:`\chi^2_{n;\alpha}` is the value exceeded with
    probability :math:`\alpha` (isf).
    """
    from scipy import stats as sp_stats

    alpha = 1.0 - confidence
    return (
        dof / float(sp_stats.chi2.isf(alpha / 2.0, dof)),
        dof / float(sp_stats.chi2.isf(1.0 - alpha / 2.0, dof)),
    )


def _chi2_interval(
    gxx: NDArray[np.float64],
    nd: float,
    confidence: float,
    nyquist_bin: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Chi-square confidence interval for the autospectrum (Eq. 8.163).

    Interior bins average two squared Gaussian components per segment
    (:math:`n = 2 n_\mathrm{d}`, Eq. 8.162); the DC bin - and the Nyquist bin when
    the segment length is even - has a single real Fourier component, so
    its estimate carries only :math:`n = n_\mathrm{d}` degrees of freedom and a
    wider interval.
    """
    low_i, up_i = _chi2_bounds(2.0 * nd, confidence)
    lower = low_i * gxx
    upper = up_i * gxx
    low_e, up_e = _chi2_bounds(nd, confidence)
    lower[0] = low_e * gxx[0]
    upper[0] = up_e * gxx[0]
    if nyquist_bin:
        lower[-1] = low_e * gxx[-1]
        upper[-1] = up_e * gxx[-1]
    return lower, upper


def _coherence_from_spectra(
    gxy: NDArray[np.complex128],
    gxx: NDArray[np.float64],
    gyy: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Coherence :math:`\gamma^2 = \lvert G_{xy} \rvert^2 /
    (G_{xx} G_{yy})` clipped to [0, 1]."""
    denom = gxx * gyy
    coh = np.divide(
        np.abs(gxy) ** 2, denom, out=np.zeros_like(gxx), where=denom > 0.0
    )
    return np.asarray(np.clip(coh, 0.0, 1.0), dtype=np.float64)


def _validate_confidence(confidence: float) -> float:
    conf = float(confidence)
    if not 0.0 < conf < 1.0:
        raise ValueError("'confidence' must be in (0, 1).")
    return conf


def _validate_scaling(scaling: str) -> str:
    if scaling not in ("density", "spectrum"):
        raise ValueError("'scaling' must be 'density' or 'spectrum'.")
    return scaling


# ---------------------------------------------------------------------------
# Autospectral density (Bendat & Piersol Sections 5.2, 8.5.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralDensityResult:
    r"""Welch autospectral density with its statistical error (B&P Ch. 8).

    :ivar frequencies: One-sided frequency axis, in Hz.
    :ivar psd: Autospectral density :math:`\hat{G}_{xx}(f)` (units²/Hz
        for ``'density'`` scaling, units² for ``'spectrum'``).
    :ivar ci_lower: Lower chi-square confidence bound on
        :math:`G_{xx}` (Eq. 8.163;
        the DC bin, and the Nyquist bin for an even segment length, use
        :math:`n = n_\mathrm{d}` degrees of freedom - a wider interval - because
        those bins carry a single real Fourier component).
    :ivar ci_upper: Upper chi-square confidence bound on :math:`G_{xx}`.
    :ivar confidence: Confidence level of the interval (e.g. ``0.95``).
    :ivar random_error: Normalized random error
        :math:`\varepsilon[\hat{G}_{xx}] = 1/\sqrt{n_\mathrm{d}}`
        (Eq. 8.158) of the interior bins (:math:`\sqrt{2/n_\mathrm{d}}` at
        DC/Nyquist).
    :ivar n_segments: Raw number of (possibly overlapped) segments averaged.
    :ivar n_averages: Effective number of independent averages
        :math:`n_\mathrm{d}`
        (equals ``n_segments`` without overlap; smaller with overlap).
    :ivar degrees_of_freedom: Chi-square degrees of freedom
        :math:`n = 2 n_\mathrm{d}` of
        the interior bins (Eq. 8.162; :math:`n_\mathrm{d}` at DC/Nyquist).
    :ivar resolution_bandwidth: Effective noise bandwidth :math:`B_\mathrm{e}` of
        the tapered segment, in Hz (drives the bias error of Eq. 8.139).
    :ivar window: Taper name.
    :ivar nperseg: Segment length, in samples.
    :ivar overlap: Segment overlap fraction.
    :ivar scaling: ``'density'`` or ``'spectrum'``.
    """

    frequencies: NDArray[np.float64]
    psd: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    confidence: float
    random_error: float
    n_segments: int
    n_averages: float
    degrees_of_freedom: float
    resolution_bandwidth: float
    window: str
    nperseg: int
    overlap: float
    scaling: str

    def plot(self, ax: Axes | None = None, *, language: str = "en",
             **kwargs: Any) -> Axes:
        """Plot the spectral density in dB with its confidence band.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_spectral_density

        check_language(language)
        return plot_spectral_density(self, ax=ax, language=language, **kwargs)


def power_spectral_density(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    window: str = "hann",
    nperseg: int | None = None,
    overlap: float = _DEFAULT_OVERLAP,
    scaling: Literal["density", "spectrum"] = "density",
    confidence: float = 0.95,
) -> SpectralDensityResult:
    r"""Calibrated autospectral density with chi-square confidence interval.

    Welch's method (Bendat & Piersol Section 11.5.2: tapered, overlapped
    segment averaging, no detrending so absolute calibration is preserved).
    Alongside :math:`\hat{G}_{xx}(f)` the result reports the effective
    number of
    independent averages :math:`n_\mathrm{d}`, the normalized random error
    :math:`\varepsilon = 1/\sqrt{n_\mathrm{d}}` (Eq. 8.158) and the chi-square
    confidence interval with
    :math:`2 n_\mathrm{d}` degrees of freedom (Eq. 8.163). For the first-order
    resolution-bias error at a resonance peak see
    :func:`resolution_bias_error`.

    :param x: Signal, 1-D.
    :param fs: Sample rate, in Hz.
    :param window: Segment taper (any scipy window name; default Hann,
        the B&P Section 11.5.2 recommendation for side-lobe suppression).
    :param nperseg: Welch segment length; ``None`` picks a length giving a
        bin spacing of at most 4 Hz (the resolution bandwidth
        :math:`B_\mathrm{e}` further
        depends on the taper; see :attr:`SpectralDensityResult.resolution_bandwidth`).
    :param overlap: Segment overlap fraction in [0, 1) (default 0.5, which
        with a Hann taper retrieves most of the stability lost to tapering,
        B&P Section 11.5.2.2).
    :param scaling: ``'density'`` (units²/Hz) or ``'spectrum'`` (units² per
        segment bandwidth).
    :param confidence: Confidence level for the chi-square interval.
    :return: A :class:`SpectralDensityResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x")
    fs_v = _positive(fs, "fs")
    scaling_v = _validate_scaling(scaling)
    conf = _validate_confidence(confidence)
    seg, ovl = _validate_welch_params(xa.size, fs_v, nperseg, overlap)
    nov = _noverlap_samples(seg, ovl)

    freqs, gxx = _welch_autospectrum(xa, fs_v, seg, nov, window, scaling_v)
    k, nd, benbw = _segment_statistics(xa.size, seg, nov, window, fs_v)
    # An even segment length puts the one-sided spectrum's last bin exactly
    # at Nyquist, where (like DC) only one real Fourier component exists.
    lower, upper = _chi2_interval(gxx, nd, conf, nyquist_bin=seg % 2 == 0)
    return SpectralDensityResult(
        frequencies=freqs,
        psd=gxx,
        ci_lower=lower,
        ci_upper=upper,
        confidence=conf,
        random_error=1.0 / float(np.sqrt(nd)),
        n_segments=k,
        n_averages=nd,
        degrees_of_freedom=2.0 * nd,
        resolution_bandwidth=benbw,
        window=window,
        nperseg=seg,
        overlap=ovl,
        scaling=scaling_v,
    )


def resolution_bias_error(
    resolution_bandwidth: float, half_power_bandwidth: float
) -> float:
    r"""First-order resolution-bias error at a resonance peak (Eq. 8.141).

    :math:`\varepsilon_\mathrm{b}[\hat{G}_{xx}(f_\mathrm{r})] \approx -(B_\mathrm{e}/B_\mathrm{r})^2/3` for
    a resonance of half-power bandwidth
    :math:`B_\mathrm{r}` analysed with resolution bandwidth :math:`B_\mathrm{e}`: peaks are
    underestimated (and valleys overestimated) by frequency smoothing, in
    the direction of reduced dynamic range (B&P Section 8.5.1). The
    approximation assumes :math:`B_\mathrm{e} < B_\mathrm{r}`.

    :param resolution_bandwidth: Analysis resolution bandwidth
        :math:`B_\mathrm{e}`, Hz
        (:attr:`SpectralDensityResult.resolution_bandwidth`).
    :param half_power_bandwidth: Half-power (-3 dB) bandwidth
        :math:`B_\mathrm{r}` of the
        spectral peak, in Hz.
    :return: Normalized bias error (dimensionless, negative at a peak).
    :raises ValueError: If either bandwidth is not positive.
    """
    be = _positive(resolution_bandwidth, "resolution_bandwidth")
    br = _positive(half_power_bandwidth, "half_power_bandwidth")
    return -((be / br) ** 2) / 3.0


# ---------------------------------------------------------------------------
# Cross-spectral density (Bendat & Piersol Section 9.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossSpectralDensityResult:
    r"""Welch cross-spectral density with its statistical error (B&P Ch. 9).

    The error formulas replace the unknown true coherence with the computed
    estimate, as Bendat & Piersol recommend for measured data (Section 9.2).

    :ivar frequencies: One-sided frequency axis, in Hz.
    :ivar csd: Complex cross-spectral density :math:`\hat{G}_{xy}(f)`.
    :ivar magnitude: :math:`\lvert \hat{G}_{xy}(f) \rvert`.
    :ivar phase: Cross-spectrum phase :math:`\hat{\theta}_{xy}(f)`, in
        radians (unwrapped).
    :ivar coherence: Ordinary coherence
        :math:`\hat{\gamma}^2_{xy}(f) \in [0, 1]`.
    :ivar magnitude_random_error: Normalized random error of
        :math:`\lvert \hat{G}_{xy} \rvert`,
        :math:`\varepsilon = 1/(\lvert \gamma_{xy} \rvert \sqrt{n_\mathrm{d}})`
        (Eq. 9.33).
    :ivar phase_std: Standard deviation of the phase estimate, in radians,
        :math:`\mathrm{s.d.} = (1 - \gamma^2_{xy})^{1/2} /
        (\lvert \gamma_{xy} \rvert \sqrt{2 n_\mathrm{d}})` (Eq. 9.52).
    :ivar n_segments: Raw number of segments averaged.
    :ivar n_averages: Effective number of independent averages
        :math:`n_\mathrm{d}`.
    :ivar resolution_bandwidth: Effective noise bandwidth :math:`B_\mathrm{e}`, in
        Hz.
    :ivar window: Taper name.
    :ivar nperseg: Segment length, in samples.
    :ivar overlap: Segment overlap fraction.
    :ivar scaling: ``'density'`` or ``'spectrum'``.
    """

    frequencies: NDArray[np.float64]
    csd: NDArray[np.complex128]
    magnitude: NDArray[np.float64]
    phase: NDArray[np.float64]
    coherence: NDArray[np.float64]
    magnitude_random_error: NDArray[np.float64]
    phase_std: NDArray[np.float64]
    n_segments: int
    n_averages: float
    resolution_bandwidth: float
    window: str
    nperseg: int
    overlap: float
    scaling: str

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the magnitude, phase (with ±σ band) and coherence.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_cross_spectral_density

        check_language(language)
        return plot_cross_spectral_density(self, ax=ax, language=language,
                                           **kwargs)


def cross_spectral_density(
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    fs: float,
    *,
    window: str = "hann",
    nperseg: int | None = None,
    overlap: float = _DEFAULT_OVERLAP,
    scaling: Literal["density", "spectrum"] = "density",
) -> CrossSpectralDensityResult:
    r"""Calibrated cross-spectral density with statistical error analysis.

    Welch's method on both channels; alongside
    :math:`\hat{G}_{xy}(f)` the result
    reports the ordinary coherence and the Bendat & Piersol random errors:
    :math:`\varepsilon[\lvert \hat{G}_{xy} \rvert]
    = 1/(\lvert \gamma_{xy} \rvert \sqrt{n_\mathrm{d}})` (Eq. 9.33) for the
    magnitude and
    :math:`\mathrm{s.d.}[\hat{\theta}_{xy}] = (1 - \gamma^2_{xy})^{1/2} /
    (\lvert \gamma_{xy} \rvert \sqrt{2 n_\mathrm{d}})` (Eq. 9.52) for the phase,
    with the measured coherence in place of the unknown true value.

    :param x: First signal, 1-D.
    :param y: Second signal, 1-D, same length as ``x``.
    :param fs: Sample rate, in Hz.
    :param window: Segment taper (default Hann).
    :param nperseg: Welch segment length; ``None`` picks a default.
    :param overlap: Segment overlap fraction in [0, 1) (default 0.5).
    :param scaling: ``'density'`` or ``'spectrum'``.
    :return: A :class:`CrossSpectralDensityResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x")
    ya = _validate_signal(y, "y")
    if xa.size != ya.size:
        raise ValueError("'x' and 'y' must have the same length.")
    fs_v = _positive(fs, "fs")
    scaling_v = _validate_scaling(scaling)
    seg, ovl = _validate_welch_params(xa.size, fs_v, nperseg, overlap)
    nov = _noverlap_samples(seg, ovl)

    freqs, gxy, gxx, gyy = _welch_pair(xa, ya, fs_v, seg, nov, window, scaling_v)
    k, nd, benbw = _segment_statistics(xa.size, seg, nov, window, fs_v)
    coh = _coherence_from_spectra(gxy, gxx, gyy)
    gamma = np.sqrt(coh)
    sqrt_nd = float(np.sqrt(nd))
    mag_err = np.divide(
        1.0, gamma * sqrt_nd, out=np.full_like(gamma, np.inf), where=gamma > 0.0
    )
    phase_std = np.divide(
        np.sqrt(1.0 - coh),
        gamma * float(np.sqrt(2.0 * nd)),
        out=np.full_like(gamma, np.inf),
        where=gamma > 0.0,
    )
    return CrossSpectralDensityResult(
        frequencies=freqs,
        csd=gxy,
        magnitude=np.abs(gxy),
        phase=np.unwrap(np.angle(gxy)),
        coherence=coh,
        magnitude_random_error=mag_err,
        phase_std=phase_std,
        n_segments=k,
        n_averages=nd,
        resolution_bandwidth=benbw,
        window=window,
        nperseg=seg,
        overlap=ovl,
        scaling=scaling_v,
    )


# ---------------------------------------------------------------------------
# Coherent output spectrum (Bendat & Piersol Section 9.2.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoherentOutputSpectrumResult:
    r"""Coherent output spectrum of a single-input/single-output model.

    The measured output autospectrum splits into the part linearly
    explained by the input,
    :math:`G_{vv} = \gamma^2_{xy} G_{yy}` (Eq. 9.55), and the
    uncorrelated noise remainder
    :math:`G_{nn} = (1 - \gamma^2_{xy}) G_{yy}` (Eq. 9.56), with
    :math:`G_{yy} = G_{vv} + G_{nn}`
    (Eq. 9.57). Their ratio is the spectral signal-to-noise ratio.

    :ivar frequencies: One-sided frequency axis, in Hz.
    :ivar output_psd: Measured output autospectrum
        :math:`\hat{G}_{yy}(f)`.
    :ivar coherent_psd: Coherent output spectrum
        :math:`\hat{G}_{vv} = \hat{\gamma}^2_{xy} \hat{G}_{yy}`.
    :ivar noise_psd: Noise output spectrum
        :math:`\hat{G}_{nn} = (1 - \hat{\gamma}^2_{xy}) \hat{G}_{yy}`.
    :ivar coherence: Ordinary coherence
        :math:`\hat{\gamma}^2_{xy}(f) \in [0, 1]`.
    :ivar snr: Spectral signal-to-noise ratio
        :math:`\hat{\gamma}^2/(1 - \hat{\gamma}^2)` (:math:`\infty` at
        :math:`\hat{\gamma}^2 = 1`).
    :ivar snr_db: :math:`10 \log_{10}` of :attr:`snr`, in dB.
    :ivar random_error: Normalized random error of :math:`\hat{G}_{vv}`,
        :math:`\varepsilon = (2 - \gamma^2_{xy})^{1/2} /
        (\lvert \gamma_{xy} \rvert \sqrt{n_\mathrm{d}})` (Eq. 9.73), with the
        measured coherence in place of the true value.
    :ivar snr_random_error: Normalized random error of the SNR,
        :math:`\varepsilon = \sqrt{2} /
        (\lvert \gamma_{xy} \rvert \sqrt{n_\mathrm{d}})`, first-order propagation
        of the coherence
        random error of Eq. 9.82 through
        :math:`\gamma^2/(1 - \gamma^2)`.
    :ivar coherence_bias: First-order bias of the coherence estimate,
        :math:`b[\hat{\gamma}^2] \approx (1 - \gamma^2)^2 / n_\mathrm{d}`
        (Eq. 9.75).
    :ivar n_segments: Raw number of segments averaged.
    :ivar n_averages: Effective number of independent averages
        :math:`n_\mathrm{d}`.
    :ivar resolution_bandwidth: Effective noise bandwidth :math:`B_\mathrm{e}`, in
        Hz.
    :ivar window: Taper name.
    :ivar nperseg: Segment length, in samples.
    :ivar overlap: Segment overlap fraction.
    :ivar scaling: ``'density'`` or ``'spectrum'``.
    """

    frequencies: NDArray[np.float64]
    output_psd: NDArray[np.float64]
    coherent_psd: NDArray[np.float64]
    noise_psd: NDArray[np.float64]
    coherence: NDArray[np.float64]
    snr: NDArray[np.float64]
    snr_db: NDArray[np.float64]
    random_error: NDArray[np.float64]
    snr_random_error: NDArray[np.float64]
    coherence_bias: NDArray[np.float64]
    n_segments: int
    n_averages: float
    resolution_bandwidth: float
    window: str
    nperseg: int
    overlap: float
    scaling: str

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the output/coherent/noise spectra and the spectral SNR.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_coherent_output_spectrum

        check_language(language)
        return plot_coherent_output_spectrum(self, ax=ax, language=language,
                                             **kwargs)


def coherent_output_spectrum(
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    fs: float,
    *,
    window: str = "hann",
    nperseg: int | None = None,
    overlap: float = _DEFAULT_OVERLAP,
    scaling: Literal["density", "spectrum"] = "density",
) -> CoherentOutputSpectrumResult:
    r"""Coherent output spectrum and spectral SNR (Bendat & Piersol 9.2.2).

    Splits the measured output autospectrum :math:`G_{yy}` into the
    coherent part
    :math:`G_{vv} = \gamma^2_{xy} G_{yy}` linearly explained by the
    input ``x`` and the noise
    remainder :math:`G_{nn} = (1 - \gamma^2_{xy}) G_{yy}`, and reports
    the spectral
    signal-to-noise ratio :math:`\gamma^2/(1 - \gamma^2)` together with
    the Bendat & Piersol
    random errors (Eqs. 9.73 and 9.82). For additive uncorrelated output
    noise of known level the coherence satisfies
    :math:`\gamma^2 = \mathrm{SNR}/(1 + \mathrm{SNR})`, which is the
    closed-form oracle used to verify
    the implementation.

    :param x: Input (reference) signal, 1-D.
    :param y: Output (response) signal, 1-D, same length as ``x``.
    :param fs: Sample rate, in Hz.
    :param window: Segment taper (default Hann).
    :param nperseg: Welch segment length; ``None`` picks a default.
    :param overlap: Segment overlap fraction in [0, 1) (default 0.5).
    :param scaling: ``'density'`` or ``'spectrum'``.
    :return: A :class:`CoherentOutputSpectrumResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x")
    ya = _validate_signal(y, "y")
    if xa.size != ya.size:
        raise ValueError("'x' and 'y' must have the same length.")
    fs_v = _positive(fs, "fs")
    scaling_v = _validate_scaling(scaling)
    seg, ovl = _validate_welch_params(xa.size, fs_v, nperseg, overlap)
    nov = _noverlap_samples(seg, ovl)

    freqs, gxy, gxx, gyy = _welch_pair(xa, ya, fs_v, seg, nov, window, scaling_v)
    k, nd, benbw = _segment_statistics(xa.size, seg, nov, window, fs_v)
    coh = _coherence_from_spectra(gxy, gxx, gyy)
    gvv = coh * gyy
    gnn = (1.0 - coh) * gyy
    one_minus = 1.0 - coh
    snr = np.divide(
        coh, one_minus, out=np.full_like(coh, np.inf), where=one_minus > 0.0
    )
    with np.errstate(divide="ignore"):
        snr_db = 10.0 * np.log10(snr)
    gamma = np.sqrt(coh)
    sqrt_nd = float(np.sqrt(nd))
    cop_err = np.divide(
        np.sqrt(2.0 - coh),
        gamma * sqrt_nd,
        out=np.full_like(gamma, np.inf),
        where=gamma > 0.0,
    )
    snr_err = np.divide(
        float(np.sqrt(2.0)),
        gamma * sqrt_nd,
        out=np.full_like(gamma, np.inf),
        where=gamma > 0.0,
    )
    return CoherentOutputSpectrumResult(
        frequencies=freqs,
        output_psd=gyy,
        coherent_psd=gvv,
        noise_psd=gnn,
        coherence=coh,
        snr=snr,
        snr_db=snr_db,
        random_error=cop_err,
        snr_random_error=snr_err,
        coherence_bias=one_minus**2 / nd,
        n_segments=k,
        n_averages=nd,
        resolution_bandwidth=benbw,
        window=window,
        nperseg=seg,
        overlap=ovl,
        scaling=scaling_v,
    )


# ---------------------------------------------------------------------------
# Fractional-octave smoothing (constant-percentage bandwidth, B&P 8.5.3)
# ---------------------------------------------------------------------------


def _smoothing_validate(
    f: NDArray[np.float64], v: NDArray[np.float64]
) -> None:
    """Validate the frequency axis / spectrum pair of the smoother."""
    if f.ndim != 1 or v.ndim != 1:
        raise ValueError("'frequencies' and 'values' must be one-dimensional.")
    if f.size != v.size:
        raise ValueError("'frequencies' and 'values' must have the same length.")
    if f.size < 2:
        raise ValueError("At least two frequency points are required.")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(v)):
        raise ValueError("'frequencies' and 'values' must be finite.")
    if np.any(np.diff(f) <= 0.0):
        raise ValueError("'frequencies' must be strictly increasing.")


def _smoothing_to_power(
    v: NDArray[np.float64], domain: str
) -> NDArray[np.float64]:
    """Map the input spectrum onto power-like values per ``domain``."""
    if domain not in ("power", "amplitude", "db"):
        raise ValueError("'domain' must be 'power', 'amplitude' or 'db'.")
    if domain == "db":
        return np.asarray(10.0 ** (v / 10.0), dtype=np.float64)
    if np.any(v < 0.0):
        raise ValueError(f"'values' must be non-negative in the {domain} domain.")
    return v.copy() if domain == "power" else v * v


def _smoothing_from_power(
    power: NDArray[np.float64], domain: str
) -> NDArray[np.float64]:
    """Map smoothed power back to the input ``domain``."""
    if domain == "power":
        return power
    if domain == "amplitude":
        return np.asarray(np.sqrt(power), dtype=np.float64)
    with np.errstate(divide="ignore"):
        out_db = 10.0 * np.log10(power)
    return np.asarray(out_db, dtype=np.float64)


def _smoothing_window_average(
    fp: NDArray[np.float64], pp: NDArray[np.float64], fraction: float
) -> NDArray[np.float64]:
    """Average piecewise-constant power over the 1/n-octave windows.

    Bins are bounded by arithmetic midpoints; the running integral makes
    the window average exact (and a flat spectrum exactly invariant). The
    window is clipped at the ends of the axis.
    """
    edges = np.empty(fp.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (fp[:-1] + fp[1:])
    edges[0] = max(fp[0] - 0.5 * (fp[1] - fp[0]), 0.0)
    edges[-1] = fp[-1] + 0.5 * (fp[-1] - fp[-2])
    cumulative = np.concatenate(([0.0], np.cumsum(pp * np.diff(edges))))
    half = 2.0 ** (1.0 / (2.0 * fraction))
    f_low = np.clip(fp / half, edges[0], edges[-1])
    f_high = np.clip(fp * half, edges[0], edges[-1])
    integral = np.interp(f_high, edges, cumulative) - np.interp(
        f_low, edges, cumulative
    )
    width = f_high - f_low
    return np.asarray(
        np.divide(integral, width, out=pp.copy(), where=width > 0.0),
        dtype=np.float64,
    )


def fractional_octave_smoothing(
    frequencies: NDArray[np.float64] | list[float],
    values: NDArray[np.float64] | list[float],
    fraction: float = 3.0,
    *,
    domain: Literal["power", "amplitude", "db"] = "power",
) -> NDArray[np.float64]:
    r"""Smooth a spectrum with a constant-power 1/n-octave kernel.

    Each output point is the power average of the input over a rectangular
    window of 1/``fraction`` octave centred (geometrically) on its
    frequency: :math:`[f \cdot 2^{-1/2n}, f \cdot 2^{+1/2n}]`. This is
    the
    constant-percentage resolution bandwidth that Bendat & Piersol
    (Section 8.5.3) recommend for spectra of resonant systems, and the de
    facto standard presentation of loudspeaker and room responses. The
    average is computed on power regardless of ``domain`` (amplitudes are
    squared first, dB levels converted), so smoothing conserves band power
    rather than amplitude; a flat spectrum is left exactly unchanged.

    The window is clipped at the ends of the frequency axis, and points at
    non-positive frequencies (where a log-frequency window is undefined)
    are copied unchanged.

    :param frequencies: Frequency axis, 1-D, strictly increasing.
    :param values: Spectrum sampled on ``frequencies``: power-like values
        (``'power'``), magnitudes (``'amplitude'``) or levels in dB
        (``'db'``).
    :param fraction: The ``n`` of the 1/n-octave width (default 3, one-third
        octave).
    :param domain: How ``values`` map to power (see above). The output is
        returned in the same domain.
    :return: Smoothed spectrum, same shape and domain as ``values``.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    _smoothing_validate(f, v)
    frac = _positive(fraction, "fraction")
    power = _smoothing_to_power(v, domain)

    pos = f > 0.0
    fp = f[pos]
    out_power = power.copy()
    if fp.size >= 2:
        out_power[pos] = _smoothing_window_average(fp, power[pos], frac)
    return _smoothing_from_power(out_power, domain)
