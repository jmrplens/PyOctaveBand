#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Calibrated spectral-density estimation with statistical error analysis.

Welch-averaged auto- and cross-spectral density estimators that report,
alongside the spectrum itself, the statistical quality of the estimate,
following Bendat & Piersol, *Random Data: Analysis and Measurement
Procedures* (4th ed., 2010):

* the **number of averages**: the raw segment count and the effective number
  of independent averages :math:`n_d` once the correlation between
  overlapped,
  tapered segments is accounted for (Section 11.5.2.2 and its Ref. 11,
  Welch 1967);
* the **normalized random error** of the autospectrum estimate,
  :math:`\varepsilon[\hat{G}_{xx}] = 1/\sqrt{n_d}` (Eq. 8.158), and of
  the cross-spectrum magnitude and phase,
  :math:`\varepsilon[\lvert \hat{G}_{xy} \rvert]
  = 1/(\lvert \gamma_{xy} \rvert \sqrt{n_d})` (Eq. 9.33) and
  :math:`\mathrm{s.d.}[\hat{\theta}_{xy}]
  = (1 - \gamma^2_{xy})^{1/2} /
  (\lvert \gamma_{xy} \rvert \sqrt{2 n_d})` (Eq. 9.52);
* **chi-square confidence intervals** for the autospectrum: the sampling
  distribution is :math:`n \hat{G}_{xx}/G_{xx} \sim \chi^2_n` with
  :math:`n = 2 n_d` degrees of freedom
  (Eq. 8.162), giving the interval
  :math:`n \hat{G}_{xx}/\chi^2_{n;\alpha/2} \le G_{xx} \le
  n \hat{G}_{xx}/\chi^2_{n;1-\alpha/2}` (Eq. 8.163);
* the **first-order resolution-bias error**:
  :math:`b[\hat{G}_{xx}] \approx (B_e^2/24) \, G''_{xx}`
  (Eq. 8.139), which for a resonance peak of half-power bandwidth
  :math:`B_r`
  becomes :math:`\varepsilon_b \approx -(B_e/B_r)^2/3` (Eq. 8.141) -
  exposed here as
  :func:`resolution_bias_error`;
* the **coherent output spectrum**
  :math:`G_{vv} = \gamma^2_{xy} G_{yy}` and the noise output
  spectrum :math:`G_{nn} = (1 - \gamma^2_{xy}) G_{yy}` of the
  single-input/single-output model
  (Eqs. 9.55-9.56), with the spectral signal-to-noise ratio
  :math:`\gamma^2/(1 - \gamma^2)` and the random error
  :math:`\varepsilon[\hat{G}_{vv}] = (2 - \gamma^2_{xy})^{1/2} /
  (\lvert \gamma_{xy} \rvert \sqrt{n_d})` (Eq. 9.73).

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

:func:`window_metrics` characterizes any taper the ``window`` parameter
accepts with the figures of merit of Harris (1978, *On the use of windows
for harmonic analysis with the discrete Fourier transform*): equivalent
noise bandwidth, coherent gain, scalloping loss, worst-case processing
loss, highest sidelobe level and the -3 dB main-lobe width - the numbers
that turn "which window should I use?" into a trade-off one can read.

:func:`multitaper_psd` adds Thomson's multitaper estimator (Thomson 1982;
Percival & Walden, *Spectral Analysis for Physical Applications*, 1993,
Chapter 7) as the whole-record alternative to Welch segment averaging:
``K`` orthogonal discrete prolate spheroidal (Slepian) tapers of
time-half-bandwidth ``NW`` produce ``K`` nearly uncorrelated eigenspectra
whose (adaptively weighted) average carries about :math:`2K` chi-square
degrees of freedom without splitting the record - the estimator of choice
for short records where Welch would leave too few segments.
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
    "MultitaperSpectralDensityResult",
    "SpectralDensityResult",
    "WindowMetricsResult",
    "coherent_output_spectrum",
    "cross_spectral_density",
    "fractional_octave_smoothing",
    "multitaper_psd",
    "power_spectral_density",
    "resolution_bias_error",
    "window_metrics",
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
      :math:`n_d`.
      Bendat & Piersol's error formulas assume :math:`n_d` distinct
      records
      (Section 9.1.1); with overlapped, tapered segments (Section 11.5.2.2)
      the variance reduction follows their Ref. 11 (Welch 1967, Eq. 12):
      :math:`\mathrm{Var} \propto (1/k)
      \left[ 1 + 2 \sum_j (1 - j/k) \rho^2(j D) \right]`
      where :math:`\rho(s)` is the
      normalized correlation of the taper with itself shifted by the hop
      ``D``, so
      :math:`n_d = k / \left( 1 + 2 \sum_j (1 - j/k)
      \rho^2(j D) \right)`. Without overlap
      this reduces to :math:`n_d = k` exactly.
    * ``resolution_bandwidth`` - the effective noise bandwidth of the
      tapered segment,
      :math:`B_e = f_s \sum w^2 / \left( \sum w \right)^2`, the
      tapered-window analog
      of Bendat & Piersol's :math:`B_e \approx 1/T` (Eq. 8.160).
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
    (:math:`n = 2 n_d`, Eq. 8.162); the DC bin - and the Nyquist bin when
    the segment length is even - has a single real Fourier component, so
    its estimate carries only :math:`n = n_d` degrees of freedom and a
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
        :math:`n = n_d` degrees of freedom - a wider interval - because
        those bins carry a single real Fourier component).
    :ivar ci_upper: Upper chi-square confidence bound on :math:`G_{xx}`.
    :ivar confidence: Confidence level of the interval (e.g. ``0.95``).
    :ivar random_error: Normalized random error
        :math:`\varepsilon[\hat{G}_{xx}] = 1/\sqrt{n_d}`
        (Eq. 8.158) of the interior bins (:math:`\sqrt{2/n_d}` at
        DC/Nyquist).
    :ivar n_segments: Raw number of (possibly overlapped) segments averaged.
    :ivar n_averages: Effective number of independent averages
        :math:`n_d`
        (equals ``n_segments`` without overlap; smaller with overlap).
    :ivar degrees_of_freedom: Chi-square degrees of freedom
        :math:`n = 2 n_d` of
        the interior bins (Eq. 8.162; :math:`n_d` at DC/Nyquist).
    :ivar resolution_bandwidth: Effective noise bandwidth :math:`B_e` of
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
    independent averages :math:`n_d`, the normalized random error
    :math:`\varepsilon = 1/\sqrt{n_d}` (Eq. 8.158) and the chi-square
    confidence interval with
    :math:`2 n_d` degrees of freedom (Eq. 8.163). For the first-order
    resolution-bias error at a resonance peak see
    :func:`resolution_bias_error`.

    :param x: Signal, 1-D.
    :param fs: Sample rate, in Hz.
    :param window: Segment taper (any scipy window name; default Hann,
        the B&P Section 11.5.2 recommendation for side-lobe suppression).
    :param nperseg: Welch segment length; ``None`` picks a length giving a
        bin spacing of at most 4 Hz (the resolution bandwidth
        :math:`B_e` further
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

    :math:`\varepsilon_b[\hat{G}_{xx}(f_r)] \approx -(B_e/B_r)^2/3` for
    a resonance of half-power bandwidth
    :math:`B_r` analysed with resolution bandwidth :math:`B_e`: peaks are
    underestimated (and valleys overestimated) by frequency smoothing, in
    the direction of reduced dynamic range (B&P Section 8.5.1). The
    approximation assumes :math:`B_e < B_r`.

    :param resolution_bandwidth: Analysis resolution bandwidth
        :math:`B_e`, Hz
        (:attr:`SpectralDensityResult.resolution_bandwidth`).
    :param half_power_bandwidth: Half-power (-3 dB) bandwidth
        :math:`B_r` of the
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
        :math:`\varepsilon = 1/(\lvert \gamma_{xy} \rvert \sqrt{n_d})`
        (Eq. 9.33).
    :ivar phase_std: Standard deviation of the phase estimate, in radians,
        :math:`\mathrm{s.d.} = (1 - \gamma^2_{xy})^{1/2} /
        (\lvert \gamma_{xy} \rvert \sqrt{2 n_d})` (Eq. 9.52).
    :ivar n_segments: Raw number of segments averaged.
    :ivar n_averages: Effective number of independent averages
        :math:`n_d`.
    :ivar resolution_bandwidth: Effective noise bandwidth :math:`B_e`, in
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
    = 1/(\lvert \gamma_{xy} \rvert \sqrt{n_d})` (Eq. 9.33) for the
    magnitude and
    :math:`\mathrm{s.d.}[\hat{\theta}_{xy}] = (1 - \gamma^2_{xy})^{1/2} /
    (\lvert \gamma_{xy} \rvert \sqrt{2 n_d})` (Eq. 9.52) for the phase,
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
        (\lvert \gamma_{xy} \rvert \sqrt{n_d})` (Eq. 9.73), with the
        measured coherence in place of the true value.
    :ivar snr_random_error: Normalized random error of the SNR,
        :math:`\varepsilon = \sqrt{2} /
        (\lvert \gamma_{xy} \rvert \sqrt{n_d})`, first-order propagation
        of the coherence
        random error of Eq. 9.82 through
        :math:`\gamma^2/(1 - \gamma^2)`.
    :ivar coherence_bias: First-order bias of the coherence estimate,
        :math:`b[\hat{\gamma}^2] \approx (1 - \gamma^2)^2 / n_d`
        (Eq. 9.75).
    :ivar n_segments: Raw number of segments averaged.
    :ivar n_averages: Effective number of independent averages
        :math:`n_d`.
    :ivar resolution_bandwidth: Effective noise bandwidth :math:`B_e`, in
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


# ---------------------------------------------------------------------------
# Window figures of merit (Harris 1978)
# ---------------------------------------------------------------------------

#: Oversampling factor of the window spectrum (samples per DFT bin).
_WINDOW_OVERSAMPLE = 256


@dataclass(frozen=True)
class WindowMetricsResult:
    r"""Figures of merit of a taper (Harris 1978), DFT-even sampling.

    Losses are positive dB (how much is lost), sidelobe levels negative dB
    (relative to the main lobe), bandwidths in DFT bins (multiply by
    ``fs/n`` for Hz), matching the conventions of Harris' Table 1. The
    window is sampled DFT-even (periodic), exactly as
    :func:`scipy.signal.welch` and the estimators of this module use it.

    :ivar window: The window specification as given (any name or
        ``(name, param)`` tuple :func:`scipy.signal.get_window` accepts).
    :ivar n: Window length, in samples.
    :ivar taps: The window samples ``w[m]`` (DFT-even).
    :ivar coherent_gain: Normalized DC gain :math:`\sum w / n` (1 for
        rectangular);
        the amplitude a bin-centered tone is scaled by before correction.
    :ivar enbw_bins: Equivalent noise bandwidth
        :math:`n \sum w^2 / \left( \sum w \right)^2`, in bins:
        the width of the ideal rectangular filter that would pass the same
        white-noise power (1 rectangular, 1.5 Hann, 1987/1458 Hamming).
    :ivar scalloping_loss_db: Attenuation of a tone midway between two
        bins, :math:`-20 \log_{10} \lvert W(1/2)/W(0) \rvert`, in dB
        (positive).
    :ivar worst_case_processing_loss_db: Scalloping loss plus the ENBW
        processing loss :math:`10 \log_{10}(\mathrm{ENBW})`, in dB: the
        worst-case reduction
        in output signal-to-noise ratio for a tone in white noise.
    :ivar highest_sidelobe_db: Level of the highest sidelobe relative to
        the main lobe, in dB (negative; -13.3 rectangular, -31.5 Hann).
    :ivar mainlobe_width_3db_bins: Two-sided -3 dB width of the main
        lobe, in bins.
    """

    window: str | tuple[Any, ...]
    n: int
    taps: NDArray[np.float64]
    coherent_gain: float
    enbw_bins: float
    scalloping_loss_db: float
    worst_case_processing_loss_db: float
    highest_sidelobe_db: float
    mainlobe_width_3db_bins: float

    def enbw_hz(self, fs: float) -> float:
        """Equivalent noise bandwidth in Hz for a sample rate ``fs``.

        :param fs: Sample rate, in Hz.
        :return: ``enbw_bins·fs/n``, in Hz.
        """
        return self.enbw_bins * _positive(fs, "fs") / self.n

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the window shape and its spectrum with the metrics marked.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_window_metrics

        check_language(language)
        return plot_window_metrics(self, ax=ax, language=language, **kwargs)


def _window_spectrum_db(
    w: NDArray[np.float64], oversample: int
) -> NDArray[np.float64]:
    """Magnitude of the window spectrum re its peak, in dB, oversampled.

    Zero-padding the DFT by ``oversample`` samples the underlying
    Dirichlet-kernel spectrum every ``1/oversample`` bins, so bin
    ``k/oversample`` of the result is exactly ``W(k/oversample)``.
    """
    spectrum = np.abs(np.fft.rfft(w, n=w.size * oversample))
    with np.errstate(divide="ignore"):
        level = 20.0 * np.log10(spectrum / spectrum[0])
    return np.asarray(level, dtype=np.float64)


def _mainlobe_edge(level_db: NDArray[np.float64]) -> int:
    """Index of the first local minimum (the edge of the main lobe)."""
    falling = np.diff(level_db) < 0.0
    edges = np.flatnonzero(~falling[1:] & falling[:-1]) + 1
    if edges.size == 0:  # pragma: no cover - every proper taper has a null
        raise ValueError("The window spectrum has no sidelobe structure.")
    return int(edges[0])


def _width_3db_bins(
    level_db: NDArray[np.float64], oversample: int
) -> float:
    """Two-sided -3 dB main-lobe width in bins (linear dB interpolation)."""
    below = np.flatnonzero(level_db <= -3.0103)
    i = int(below[0])
    d0, d1 = float(level_db[i - 1]), float(level_db[i])
    frac = ((-3.0103) - d0) / (d1 - d0)
    return 2.0 * (i - 1 + frac) / oversample


def window_metrics(
    window: str | tuple[Any, ...],
    n: int = 1024,
) -> WindowMetricsResult:
    """Figures of merit of a spectral-analysis taper (Harris 1978).

    Computes the numbers behind the window trade-off for any taper the
    ``window`` parameter of this module's estimators accepts: the
    equivalent noise bandwidth and coherent gain (closed forms of the
    samples, machine-exact), and the scalloping loss, worst-case
    processing loss, highest sidelobe level and -3 dB main-lobe width
    (measured on the spectrum of the sampled window, oversampled by
    zero-padding). The window is sampled DFT-even (periodic), exactly as
    the Welch estimators apply it.

    :param window: Window name or ``(name, param)`` tuple, anything
        :func:`scipy.signal.get_window` accepts (e.g. ``'hann'``,
        ``('kaiser', 8.6)``, ``('tukey', 0.5)``).
    :param n: Window length, in samples (at least 16).
    :return: A :class:`WindowMetricsResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    from scipy import signal as sp_signal

    n_v = int(n)
    if n_v < 16:
        raise ValueError("'n' must be at least 16 samples.")
    try:
        w = np.asarray(
            sp_signal.get_window(window, n_v, fftbins=True), dtype=np.float64
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unknown window specification: {window!r}") from exc
    if np.any(~np.isfinite(w)) or float(np.sum(w)) <= 0.0:
        raise ValueError(f"Degenerate window: {window!r}")

    wsum = float(np.sum(w))
    coherent_gain = wsum / n_v
    enbw = n_v * float(np.sum(w * w)) / (wsum * wsum)

    level_db = _window_spectrum_db(w, _WINDOW_OVERSAMPLE)
    # Bin k/oversample of the padded DFT is exactly W(k/oversample), so
    # index oversample/2 evaluates the spectrum at the half-bin offset.
    scalloping = -float(level_db[_WINDOW_OVERSAMPLE // 2])
    edge = _mainlobe_edge(level_db)
    highest_sidelobe = float(np.max(level_db[edge:]))
    return WindowMetricsResult(
        window=window,
        n=n_v,
        taps=w,
        coherent_gain=coherent_gain,
        enbw_bins=enbw,
        scalloping_loss_db=scalloping,
        worst_case_processing_loss_db=scalloping + 10.0 * np.log10(enbw),
        highest_sidelobe_db=highest_sidelobe,
        mainlobe_width_3db_bins=_width_3db_bins(level_db, _WINDOW_OVERSAMPLE),
    )


# ---------------------------------------------------------------------------
# Thomson multitaper spectral estimation (Percival & Walden 1993, Ch. 7)
# ---------------------------------------------------------------------------

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
    :class:`SpectralDensityResult`, but here the degrees of freedom are
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
        :math:`B_e`
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

    def plot(self, ax: Axes | None = None, *, language: str = "en",
             **kwargs: Any) -> Axes:
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
            "'time_half_bandwidth' must be in [1, n/2) "
            f"(got {nw:g} for {n} samples)."
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
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
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

    tapers, eigenvalues = sp_windows.dpss(
        x.size, nw, Kmax=k, return_ratios=True
    )
    tapers = np.atleast_2d(np.asarray(tapers, dtype=np.float64))
    yk = np.fft.rfft(tapers * x, axis=-1)
    sk = (np.abs(yk) ** 2) / fs
    # Near-unity concentrations can exceed 1 by a few ulp (the numerical
    # hazard P&W document around their Table 380); clip so the broad-band
    # bias term (1-lambda) of the adaptive weights can never turn negative.
    lam = np.clip(
        np.atleast_1d(np.asarray(eigenvalues, dtype=np.float64)), 0.0, 1.0
    )
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
    s = np.sum(lam[:k_seed] * sk[:k_seed], axis=0) / float(
        np.sum(eigenvalues[:k_seed])
    )
    d = eigenvalues[:, np.newaxis] * np.ones_like(sk)
    for _ in range(_ADAPTIVE_MAX_ITER):
        # The denominator is bounded below by (1-lambda_k)*sigma^2*dt,
        # positive for every taper with lambda_k < 1; the floor covers the
        # degenerate lambda_k == 1 taper meeting an all-zero bin.
        b = s / np.maximum(
            lam * s + (1.0 - lam) * power_density, np.finfo(float).tiny
        )
        d = b * b * lam
        dsum = np.sum(d, axis=0)
        s_new = np.divide(
            np.sum(d * sk, axis=0), dsum, out=s.copy(), where=dsum > 0.0
        )
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


def _multitaper_dof(
    d: NDArray[np.float64], nyquist_bin: bool
) -> NDArray[np.float64]:
    r"""Equivalent degrees of freedom
    :math:`\nu(f) = 2 \left( \sum_k d_k \right)^2 / \sum_k d_k^2`.

    P&W Eq. 370b: each eigenspectrum contributes two chi-square degrees
    of freedom (one complex Fourier component), combined with weights
    :math:`d_k`. The DC bin - and the Nyquist bin when the record length
    is even - has a single real component per eigenspectrum, so its
    :math:`\nu` is halved (same convention as the Welch estimators
    above).
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
    lower = psd * dof / np.asarray(
        sp_stats.chi2.isf(alpha / 2.0, dof), dtype=np.float64
    )
    upper = psd * dof / np.asarray(
        sp_stats.chi2.isf(1.0 - alpha / 2.0, dof), dtype=np.float64
    )
    return lower, upper


def multitaper_psd(
    x: NDArray[np.float64] | list[float],
    fs: float,
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
    (:func:`power_spectral_density`) would leave too few segments.

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

    Calibration matches the Welch estimators of this module exactly: no
    detrending, ``'density'`` scaling integrates to the signal power
    (units²/Hz, one-sided) and ``'spectrum'`` scaling reads
    :math:`A^2/2` at
    the peak of a sinusoid of amplitude ``A`` (the tone calibration is
    exact for the taper set in use, computed from the taper DC gains
    :math:`\left( \sum_t h_{tk} \right)^2`; a tone's power in
    ``'density'`` scaling is spread over
    the resolution bandwidth :math:`2W`).

    :param x: Signal, 1-D (used whole; no segmentation).
    :param fs: Sample rate, in Hz.
    :param time_half_bandwidth: Duration x half-bandwidth product ``NW``
        (dimensionless; default 4, P&W's worked choice). The design
        half-bandwidth is :math:`W = NW f_s / N` Hz; larger ``NW``
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
    fs_v = _positive(fs, "fs")
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
