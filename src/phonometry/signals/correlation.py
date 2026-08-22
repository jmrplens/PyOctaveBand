#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Correlation analysis and time-delay estimation.

Auto- and cross-correlation estimators with the three standard
normalizations, generalized cross-correlation (GCC) time-delay estimation
with the Knapp & Carter weightings, and sub-sample delay location for
impulse responses, following Bendat & Piersol, *Random Data: Analysis and
Measurement Procedures* (4th ed., 2010) and Knapp & Carter (1976):

* **correlation estimates** computed via FFT with zero padding so the
  circular product never wraps (B&P Section 11.4.2, Eq. 11.95), with the
  :math:`1/N` *biased*, :math:`1/(N-r)` *unbiased* (Eq. 11.96) and
  *coefficient*
  :math:`\rho_{xy}(\tau) = C_{xy}(\tau)/(\sigma_x \sigma_y)` (Eq. 5.16)
  normalizations, plus the
  large-``T`` normalized random error of the estimate for bandwidth-limited
  Gaussian data,
  :math:`\varepsilon[\hat{R}_{xy}(\tau)]
  = [1 + \rho_{xy}^{-2}(\tau)]^{1/2} / \sqrt{2BT}`
  (Eqs. 8.109/8.112), exposed as :func:`correlation_random_error`;
* **time-delay estimation**: the peak of the cross-correlation locates the
  delay of a common signal between two sensors (B&P Section 5.1.4,
  Eq. 5.21). :func:`time_delay` implements the direct correlator, the
  weighted-phase-slope estimator of the cross-spectrum (Eq. 5.101b) and the
  **generalized cross-correlation** of Knapp & Carter (1976): the averaged
  cross-spectrum is weighted by :math:`\psi(f)` before the inverse
  transform, with the Table I processors ``'roth'``
  (:math:`1/G_{xx}`), ``'scot'``
  (:math:`1/\sqrt{G_{xx} G_{yy}}`), ``'phat'``
  (:math:`1/\lvert G_{xy} \rvert`) and the maximum-likelihood
  ``'ml'`` (Hannan-Thomson) weighting
  :math:`\lvert \gamma \rvert^2 /
  (\lvert G_{xy} \rvert (1 - \lvert \gamma \rvert^2))`
  that attains the Cramér-Rao bound;
* **sub-sample peak location** by three-point parabolic interpolation,
  optionally after band-limited local upsampling of the correlation around
  its peak, and the **peak-location uncertainty** of the delay estimate,
  :math:`\sigma(\hat{\tau}_0) \approx (3/4)^{1/4} \sqrt{\varepsilon} /
  (\pi B)` (Eq. 8.129), with the 95 % interval
  :math:`\hat{\tau}_0 \pm 2\sigma` (Eq. 8.130);
* **impulse-response utilities**: the sub-sample arrival time of a single
  IR (its peak is the cross-correlation with an ideal impulse) and the
  alignment of an IR pair by the estimated delay, applied as an exact
  fractional shift in the frequency domain.

The GCC estimators run on the same Welch core (segmentation, tapering,
overlap policy) as :mod:`phonometry.signals.spectra`, so a GCC and a
cross-spectral density computed with the same segment length are mutually
consistent bin by bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._internal.validation import require_ranks, require_same_length
from ..io._resolve import resolve_fs, resolve_pair_fs
from ..io._signal import Signal
from .spectra import (
    _coherence_from_spectra,
    _noverlap_samples,
    _positive,
    _validate_signal,
    _validate_welch_params,
    _welch_pair,
)
from .test_signals import _fractional_advance

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "AlignedImpulseResponseResult",
    "CorrelationResult",
    "TimeDelayResult",
    "align_impulse_responses",
    "correlation",
    "correlation_random_error",
    "impulse_response_delay",
    "time_delay",
]

#: Half-width, in samples, of the window that the local band-limited
#: upsampling extracts around the coarse correlation peak.
_UPSAMPLE_HALF_WINDOW = 32

#: Minimum number of averaged Welch segments the 'ml' GCC weighting
#: requires: a single-segment coherence estimate is identically one, and
#: Knapp & Carter's Eq. 45b weight exists only for |γ|² < 1.
_MIN_COHERENCE_SEGMENTS = 2

#: Validation context of the impulse-response delay utilities.
_DELAY_CONTEXT = "a delay estimate"

_Normalization = Literal["biased", "unbiased", "coefficient"]
_Method = Literal["gcc", "direct", "phase"]
_Weighting = Literal["none", "roth", "scot", "phat", "ml"]
_Interpolation = Literal["parabolic", "none"]


# ---------------------------------------------------------------------------
# Correlation estimates (Bendat & Piersol Sections 5.1, 11.4)
# ---------------------------------------------------------------------------


def _linear_correlation(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Raw linear correlation sums :math:`c(r) = \sum_k x[k] y[k+r]`.

    Zero-padded FFT computation (B&P Section 11.4.2): the returned array
    covers lags :math:`r = -(N-1) \ldots N-1` in order.
    """
    from scipy import signal as sp_signal

    return np.asarray(
        sp_signal.correlate(y, x, mode="full", method="fft"), dtype=np.float64
    )


def _coefficient_correlation(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Correlation coefficient function :math:`\rho_{xy}(\tau)`.

    B&P Eq. 5.16: biased covariance estimate over the mean-removed
    records, normalized by :math:`\sigma_x \sigma_y` so
    :math:`\lvert \rho \rvert \le 1` always holds (Cauchy-Schwarz).
    """
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.std(xc)) * float(np.std(yc)) * x.size
    if denom <= 0.0:  # a constant record: no covariance to normalize
        return np.zeros(2 * x.size - 1, dtype=np.float64)
    return _linear_correlation(xc, yc) / denom


@dataclass(frozen=True)
class CorrelationResult:
    r"""Auto- or cross-correlation estimate (B&P Sections 5.1, 8.4, 11.4).

    :ivar lags: Lag axis :math:`\tau`, in seconds, symmetric about zero.
        Positive lag means the second signal is delayed relative to the
        first (:math:`\hat{R}_{xy}(\tau) \sim E[x(t) y(t+\tau)]`, B&P
        Eq. 5.19-5.20 convention).
    :ivar values: Correlation estimate on ``lags`` with the requested
        :attr:`normalization`.
    :ivar coefficient: Correlation coefficient function
        :math:`\hat{\rho}_{xy}(\tau) \in [-1, 1]` on the same lags
        (Eq. 5.16; equals :attr:`values` when
        ``normalization='coefficient'``).
    :ivar normalization: ``'biased'`` (:math:`1/N`), ``'unbiased'``
        (:math:`1/(N - \lvert r \rvert)`, Eq. 11.96) or ``'coefficient'``.
    :ivar kind: ``'autocorrelation'`` or ``'cross-correlation'``.
    :ivar fs: Sample rate, in Hz.
    :ivar n_samples: Record length ``N``, in samples.
    :ivar duration: Record length :math:`T = N/f_\mathrm{s}`, in seconds.
    """

    lags: NDArray[np.float64]
    values: NDArray[np.float64]
    coefficient: NDArray[np.float64]
    normalization: str
    kind: str
    fs: float
    n_samples: int
    duration: float

    def __post_init__(self) -> None:
        """Reject an estimate whose three curves do not share one lag axis.

        The three are read as columns of one table: :meth:`plot` draws
        :attr:`values` against :attr:`lags`, and :meth:`random_error` turns
        :attr:`coefficient` into an error per lag, position by position,
        without ever looking at :attr:`lags` again. Nothing downstream can
        catch a coefficient of another length, because what comes back is a
        well-formed error curve; it is simply indexed by a lag axis it was
        never measured on, and the reader pairs it with the delay that
        Bendat & Piersol's Eq. 8.129 uncertainty is quoted for.

        :raises ValueError: if the two estimates disagree with the lag axis.
        """
        require_ranks(self, lags=1, values=1, coefficient=1)
        require_same_length(self, "lags", "values", "coefficient", axis="lag")

    def random_error(self, signal_bandwidth: float) -> NDArray[np.float64]:
        r"""Normalized random error of the estimate at each lag.

        For bandwidth-limited Gaussian data of bandwidth ``B`` and record
        length ``T`` (B&P Eqs. 8.109 and 8.112, identical in form for the
        cross- and autocorrelation, with the measured coefficient in place
        of the true value):

        .. math::

           \varepsilon[\hat{R}_{xy}(\tau)]
           = \frac{[1 + \rho_{xy}^{-2}(\tau)]^{1/2}}{\sqrt{2BT}}

        so :math:`\varepsilon[\hat{R}_{xx}(0)] = 1/\sqrt{BT}`
        (Eq. 8.111). The large-``T`` approximation behind it assumes
        :math:`T \ge 10 \lvert \tau \rvert` and :math:`BT \ge 5`
        (Section 8.4.1). Lags where the measured coefficient is zero
        return ``inf``.

        :param signal_bandwidth: Signal bandwidth ``B``, in Hz.
        :return: Normalized random error per lag (dimensionless).
        :raises ValueError: If the bandwidth is not positive.
        """
        b = _positive(signal_bandwidth, "signal_bandwidth")
        rho2 = self.coefficient * self.coefficient
        ratio = np.divide(
            1.0,
            rho2,
            out=np.full_like(rho2, np.inf),
            where=rho2 > 0.0,
        )
        return np.asarray(
            np.sqrt(1.0 + ratio) / np.sqrt(2.0 * b * self.duration),
            dtype=np.float64,
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the correlation estimate against the lag in seconds.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_correlation

        check_language(language)
        return plot_correlation(self, ax=ax, language=language, **kwargs)


def correlation(
    x: Signal | NDArray[np.float64] | list[float],
    y: Signal | NDArray[np.float64] | list[float] | None = None,
    fs: float | None = None,
    *,
    normalization: _Normalization = "unbiased",
    max_lag: float | None = None,
) -> CorrelationResult:
    r"""Auto- or cross-correlation estimate with a chosen normalization.

    Computed via zero-padded FFT so the circular product never wraps (B&P
    Section 11.4.2). ``y=None`` gives the autocorrelation of ``x``. The
    sign convention follows B&P Eq. 5.19-5.20: with
    :math:`y(t) = \alpha x(t - \tau_0) + n(t)` the estimate peaks at
    :math:`\tau = +\tau_0`.

    Normalizations:

    * ``'biased'`` - the raw lag sums divided by ``N``; tapers toward the
      record ends and stays bounded by
      :math:`[\hat{R}_{xx}(0) \hat{R}_{yy}(0)]^{1/2}`;
    * ``'unbiased'`` - divided by :math:`N - \lvert r \rvert`
      (Eq. 11.96), an unbiased estimate of :math:`R_{xy}(\tau)` whose
      variance grows toward the ends;
    * ``'coefficient'`` - the correlation coefficient function
      :math:`\hat{\rho}_{xy}(\tau)
      = \hat{C}_{xy}(\tau)/(\sigma_x \sigma_y) \in [-1, 1]` over the
      mean-removed records (Eq. 5.16).

    :param x: First signal, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the correlation values
        come out in Pa². The normalized ``coefficient`` divides them out and
        does not move.
    :param y: Second signal, same length, or ``None`` for autocorrelation. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the correlation values
        come out in Pa². The normalized ``coefficient`` divides them out and
        does not move.
    :param fs: Sample rate, in Hz. Required when both records are bare
        arrays; either may be a :class:`~phonometry.io.Signal` and supply it,
        and two Signals recorded at different rates are refused rather than
        arbitrated.
    :param normalization: See above (default ``'unbiased'``).
    :param max_lag: Largest lag magnitude to keep, in seconds (default:
        the full ``N-1`` samples).
    :return: A :class:`CorrelationResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x", context="a correlation estimate")
    is_auto = y is None
    ya = xa if y is None else _validate_signal(y, "y", context="a correlation estimate")
    if xa.size != ya.size:
        msg = "'x' and 'y' must have the same length."
        raise ValueError(msg)
    if isinstance(x, Signal) or isinstance(y, Signal):
        fs = resolve_pair_fs(x, x if y is None else y, fs)
    elif fs is None:
        # The one place a bare pair may omit the rate: unlike every other
        # estimate here, a correlation with no rate is still meaningful --
        # its lag axis is in samples. That default predates the Signal and
        # is kept, so `correlation(a, b)` reads exactly as it always did.
        fs = 1.0
    fs_v = _positive(fs, "fs")
    n = xa.size

    m = n - 1
    if max_lag is not None:
        m = round(_positive(max_lag, "max_lag") * fs_v)
        if not 1 <= m <= n - 1:
            msg = "'max_lag' must be at least one sample and shorter than the record."
            raise ValueError(msg)

    lag_samples = np.arange(-(n - 1), n, dtype=np.float64)
    coeff = _coefficient_correlation(xa, ya)
    if normalization == "coefficient":
        values = coeff
    elif normalization == "biased":
        values = _linear_correlation(xa, ya) / n
    elif normalization == "unbiased":
        values = _linear_correlation(xa, ya) / (n - np.abs(lag_samples))
    else:
        msg = "'normalization' must be 'biased', 'unbiased' or 'coefficient'."
        raise ValueError(msg)

    keep = np.abs(lag_samples) <= m
    return CorrelationResult(
        lags=lag_samples[keep] / fs_v,
        values=np.asarray(values[keep], dtype=np.float64),
        coefficient=np.asarray(coeff[keep], dtype=np.float64),
        normalization=normalization,
        kind="autocorrelation" if is_auto else "cross-correlation",
        fs=fs_v,
        n_samples=n,
        duration=n / fs_v,
    )


def correlation_random_error(
    coefficient: float, signal_bandwidth: float, duration: float
) -> float:
    r"""Normalized random error of a correlation estimate.

    B&P Eqs. 8.109/8.112:
    :math:`\varepsilon[\hat{R}_{xy}(\tau)]
    = [1 + \rho_{xy}^{-2}(\tau)]^{1/2} / \sqrt{2BT}` for
    bandwidth-limited
    Gaussian data of bandwidth ``B`` observed for ``T`` seconds, with
    :math:`\rho_{xy}(\tau)` the correlation coefficient at the lag of
    interest. At the zero lag of an autocorrelation (:math:`\rho = 1`)
    this is :math:`1/\sqrt{BT}`
    (Eq. 8.111). Valid for :math:`T \ge 10 \lvert \tau \rvert` and
    :math:`BT \ge 5` (Section 8.4.1).

    For the two-detector time-delay problem
    :math:`x = s + m`, :math:`y = s' + n`
    (Section 8.4.2) the peak coefficient is
    :math:`\rho = S/\sqrt{(S+M)(S+N)}`, which reproduces the book's
    Example 8.5:
    :math:`B = 100` Hz, :math:`T = 5` s, :math:`M/S = N/S = 10` give
    :math:`\varepsilon \approx 0.35`.

    :param coefficient: Correlation coefficient :math:`\rho_{xy}(\tau)`
        at the lag.
    :param signal_bandwidth: Signal bandwidth ``B``, in Hz.
    :param duration: Record length ``T``, in seconds.
    :return: Normalized random error (dimensionless; ``inf`` at
        :math:`\rho = 0`).
    :raises ValueError: If the bandwidth or duration is not positive, or
        the coefficient is outside [-1, 1].
    """
    b = _positive(signal_bandwidth, "signal_bandwidth")
    t = _positive(duration, "duration")
    rho = float(coefficient)
    if not np.isfinite(rho) or abs(rho) > 1.0:
        msg = "'coefficient' must be in [-1, 1]."
        raise ValueError(msg)
    # IEEE division carries rho = 0 to the correct limit (an infinite
    # error) without a floating-point equality branch.
    with np.errstate(divide="ignore"):
        ratio = float(np.divide(1.0, rho * rho))
    return float(np.sqrt(1.0 + ratio) / np.sqrt(2.0 * b * t))


def _peak_location_std(random_error: float, bandwidth: float) -> float:
    r"""Standard deviation of the correlation-peak location (Eq. 8.129).

    :math:`\sigma(\hat{\tau}_0) \approx (3/4)^{1/4} \sqrt{\varepsilon} /
    (\pi B)` for a bandwidth-limited-white
    correlation peak (Eq. 8.120) whose magnitude estimate carries the
    normalized random error :math:`\varepsilon`; B&P Example 8.5 rounds
    the :math:`(3/4)^{1/4}` factor to 0.93.
    """
    return float((0.75**0.25) * np.sqrt(random_error) / (np.pi * bandwidth))


# ---------------------------------------------------------------------------
# Sub-sample peak refinement (shared by the TDE and IR utilities)
# ---------------------------------------------------------------------------


def _parabolic_offset(cm: float, c0: float, cp: float) -> float:
    """Vertex offset in [-1/2, 1/2] of the parabola through three points."""
    denom = cm - 2.0 * c0 + cp
    if denom >= 0.0:  # not a strict local maximum: keep the sample peak
        return 0.0
    return float(np.clip(0.5 * (cm - cp) / denom, -0.5, 0.5))


def _upsampled_peak(
    curve: NDArray[np.float64], index: int, factor: int
) -> tuple[NDArray[np.float64], int, float, float]:
    """Band-limited local upsampling of ``curve`` around ``index``.

    Extracts a window around the coarse peak (clamped to the curve, so a
    peak near a boundary keeps a full-size window rather than a shrunken
    one), resamples it by ``factor`` with the FFT method and re-locates
    the maximum of the magnitude within one coarse sample of the original
    peak (the resampling assumes a periodic window, so its edges are
    unreliable but the peak region is not).

    :return: ``(dense_window, dense_index, dense_step, refined_position)``
        where ``refined_position`` is in original-sample units.
    """
    from scipy import signal as sp_signal

    start = max(0, index - _UPSAMPLE_HALF_WINDOW)
    stop = min(curve.size, index + _UPSAMPLE_HALF_WINDOW + 1)
    window = curve[start:stop]
    dense = np.asarray(
        sp_signal.resample(window, window.size * factor), dtype=np.float64
    )
    centre = (index - start) * factor
    lo = max(centre - factor, 0)
    hi = min(centre + factor + 1, dense.size)
    j = lo + int(np.argmax(np.abs(dense[lo:hi])))
    return dense, j, 1.0 / factor, start + j / factor


def _refine_peak(
    curve: NDArray[np.float64],
    index: int,
    interpolation: str,
    upsample: int,
) -> float:
    """Sub-sample peak position of ``curve`` near the sample ``index``.

    Optional band-limited local upsampling first, then three-point
    parabolic interpolation on the (possibly densified) grid. Sub-sample
    accuracy presumes the peak is oversampled - i.e. the underlying
    signals are band-limited below the Nyquist frequency; on a full-band
    peak one sample wide the refinement degrades gracefully to the sample
    grid.
    """
    position = float(index)
    step = 1.0
    grid: NDArray[np.float64] = curve
    j = index
    if upsample > 1:
        grid, j, step, position = _upsampled_peak(curve, index, upsample)
    if interpolation == "parabolic" and 0 < j < grid.size - 1:
        sign = -1.0 if grid[j] < 0.0 else 1.0
        position += step * _parabolic_offset(
            sign * float(grid[j - 1]),
            sign * float(grid[j]),
            sign * float(grid[j + 1]),
        )
    return position


def _validate_refinement(interpolation: str, upsample: int) -> int:
    if interpolation not in ("parabolic", "none"):
        msg = "'interpolation' must be 'parabolic' or 'none'."
        raise ValueError(msg)
    factor = int(upsample)
    if factor < 1:
        msg = "'upsample' must be a positive integer."
        raise ValueError(msg)
    return factor


# ---------------------------------------------------------------------------
# Time-delay estimation (B&P 5.1.4/5.2.7, Knapp & Carter 1976)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeDelayResult:
    r"""Time-delay estimate between two records.

    :ivar delay: Estimated delay :math:`\hat{\tau}_0` of the second
        record relative to
        the first, in seconds (positive: ``y`` lags ``x``).
    :ivar delay_samples: The same delay in (fractional) samples.
    :ivar method: ``'direct'``, ``'gcc'`` or ``'phase'``.
    :ivar weighting: GCC weighting name (``None`` unless
        ``method='gcc'``).
    :ivar lags: Lag axis of :attr:`correlation`, in seconds.
    :ivar correlation: The correlation function whose peak was located:
        the correlation coefficient :math:`\hat{\rho}_{xy}(\tau)` for
        ``'direct'``, the weighted GCC :math:`\hat{R}_\psi(\tau)`
        (normalized to unit peak magnitude) for
        ``'gcc'``, and the unweighted equivalent for ``'phase'`` (whose
        estimate comes from Eq. 5.101b, not from this curve).
    :ivar peak_correlation: Plain correlation coefficient
        :math:`\hat{\rho}_{xy}` at the
        estimated delay (rounded to the nearest sample) - the quantity
        entering the B&P error formulas, whatever the method.
    :ivar delay_std: Standard deviation of the peak-location estimate,
        :math:`\sigma(\hat{\tau}_0) \approx (3/4)^{1/4}
        \sqrt{\varepsilon} / (\pi B)` (Eq. 8.129), in seconds; ``None``
        unless ``signal_bandwidth`` was given.
    :ivar delay_interval: Approximate 95 % confidence interval
        :math:`\hat{\tau}_0 \pm 2\sigma` (Eq. 8.130), in seconds;
        ``None`` without a bandwidth.
    :ivar signal_bandwidth: The bandwidth ``B`` used for the error, Hz.
    :ivar fs: Sample rate, in Hz.
    """

    delay: float
    delay_samples: float
    method: str
    weighting: str | None
    lags: NDArray[np.float64]
    correlation: NDArray[np.float64]
    peak_correlation: float
    delay_std: float | None
    delay_interval: tuple[float, float] | None
    signal_bandwidth: float | None
    fs: float

    def __post_init__(self) -> None:
        """Reject a delay estimate whose curve does not run over its lags.

        The curve is the evidence for the number: :meth:`plot` draws it
        against :attr:`lags` and rules :attr:`delay` across it, so the figure
        asserts that the marked delay is where the peak is. A curve of
        another length shifts the peak away from the estimate the searcher
        actually located, and the estimate is reported as a scalar that
        carries no record of the axis it was found on.

        :attr:`delay_interval` is left out: it holds the two ends of the
        95 % interval, a pair rather than a curve.

        :raises ValueError: if the curve disagrees with the lag axis.
        """
        require_ranks(self, lags=1, correlation=1)
        require_same_length(self, "lags", "correlation", axis="lag")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the correlation function with the estimated delay marked.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_time_delay

        check_language(language)
        return plot_time_delay(self, ax=ax, language=language, **kwargs)


def _gcc_weight(
    weighting: str,
    gxy: NDArray[np.complex128],
    gxx: NDArray[np.float64],
    gyy: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Knapp & Carter (1976) Table I weighting :math:`\psi(f)`.

    Bins where the required denominator vanishes get zero weight (the
    band carries no usable information; Knapp & Carter note the PHAT
    phase is erratic wherever :math:`\lvert G_{xy} \rvert = 0`).
    """
    mag = np.abs(gxy)
    if weighting == "none":
        return np.ones_like(gxx)
    if weighting == "roth":
        return np.divide(1.0, gxx, out=np.zeros_like(gxx), where=gxx > 0.0)
    if weighting == "scot":
        denom = np.sqrt(gxx * gyy)
        return np.divide(1.0, denom, out=np.zeros_like(denom), where=denom > 0.0)
    if weighting == "phat":
        return np.divide(1.0, mag, out=np.zeros_like(mag), where=mag > 0.0)
    if weighting == "ml":
        coherence = _coherence_from_spectra(gxy, gxx, gyy)
        denom = mag * (1.0 - coherence)
        return np.divide(coherence, denom, out=np.zeros_like(denom), where=denom > 0.0)
    msg = "'weighting' must be 'none', 'roth', 'scot', 'phat' or 'ml'."
    raise ValueError(msg)


def _delay_error(
    xa: NDArray[np.float64],
    ya: NDArray[np.float64],
    delay_samples: float,
    signal_bandwidth: float | None,
    fs: float,
) -> tuple[float, float | None, tuple[float, float] | None]:
    """Peak coefficient and the Eq. 8.129 location error of the delay."""
    r0 = round(delay_samples)
    xc = xa - float(np.mean(xa))
    yc = ya - float(np.mean(ya))
    denom = float(np.std(xc)) * float(np.std(yc)) * xa.size
    rho = 0.0
    # The phase-slope estimate is not bounded by a search window, so a
    # noisy delay can exceed the record: no overlap, zero coefficient.
    if denom > 0.0 and abs(r0) < xa.size:
        if r0 >= 0:
            overlap = float(np.dot(xc[: xa.size - r0], yc[r0:]))
        else:
            overlap = float(np.dot(xc[-r0:], yc[: ya.size + r0]))
        rho = overlap / denom
    if signal_bandwidth is None:
        return rho, None, None
    b = _positive(signal_bandwidth, "signal_bandwidth")
    eps = correlation_random_error(abs(rho), b, xa.size / fs)
    std = _peak_location_std(eps, b) if np.isfinite(eps) else float("inf")
    delay = delay_samples / fs
    return rho, std, (delay - 2.0 * std, delay + 2.0 * std)


def _phase_slope_delay(
    freqs: NDArray[np.float64], gxy: NDArray[np.complex128]
) -> float:
    r"""Weighted phase-slope delay from the cross-spectrum (Eq. 5.101b).

    .. math::

       \hat{\tau}_0 = \frac{\int 2 \pi f \,
       \lvert \hat{G}_{xy} \rvert \, \hat{\theta}_{xy} \, df}
       {\int (2 \pi f)^2 \, \lvert \hat{G}_{xy} \rvert \, df}

    with :math:`\hat{\theta}_{xy} = -\arg \hat{G}_{xy}` unwrapped (a
    delayed second record has
    :math:`\hat{G}_{xy} \propto e^{-j 2 \pi f \tau_0}`, Eq. 5.95b). The
    small-angle linearization behind it assumes the residual
    :math:`2 \pi f \tau_0 - \hat{\theta}_{xy}` stays small, i.e. a
    clean, moderate delay.

    The phase is referenced to its DC value before the fit: a pure delay
    has :math:`\theta(0) = 0`, but a polarity-inverted path
    (:math:`\alpha < 0`) carries a
    constant :math:`\pi` offset that the intercept-free least squares of
    Eq. 5.101b would otherwise fold into the slope.
    """
    theta = -np.unwrap(np.angle(gxy))
    theta -= theta[0]
    mag = np.abs(gxy)
    w = 2.0 * np.pi * freqs
    denom = float(np.sum(w * w * mag))
    if denom <= 0.0:  # sum of non-negative terms: an empty cross-spectrum
        return 0.0
    return float(np.sum(w * mag * theta) / denom)


def _direct_tde_curve(
    xa: NDArray[np.float64],
    ya: NDArray[np.float64],
    fs: float,
    max_delay: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coefficient cross-correlation restricted to the search window."""
    n = xa.size
    m = n - 1
    if max_delay is not None:
        m = round(_positive(max_delay, "max_delay") * fs)
        if not 1 <= m <= n - 1:
            msg = "'max_delay' must be at least one sample and shorter than the record."
            raise ValueError(msg)
    curve = _coefficient_correlation(xa, ya)
    lag_samples = np.arange(-(n - 1), n, dtype=np.float64)
    keep = np.abs(lag_samples) <= m
    return lag_samples[keep], np.asarray(curve[keep], dtype=np.float64)


def _gcc_curve(
    xa: NDArray[np.float64],
    ya: NDArray[np.float64],
    fs: float,
    weighting: str,
    window: str,
    nperseg: int | None,
    overlap: float,
    max_delay: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Weighted GCC over the shared Welch core, fftshifted to ±seg/2."""
    seg, ovl = _validate_welch_params(xa.size, fs, nperseg, overlap)
    nov = _noverlap_samples(seg, ovl)
    if weighting == "ml" and (xa.size - nov) // (seg - nov) < _MIN_COHERENCE_SEGMENTS:
        # Knapp & Carter's Eq. 45b weight exists only for |γ|² < 1; a
        # single-segment coherence estimate is identically one, so the
        # weight would be floating-point rounding noise.
        msg = (
            "the 'ml' weighting needs an averaged coherence estimate: "
            "use at least two Welch segments (reduce 'nperseg')."
        )
        raise ValueError(msg)
    _freqs, gxy, gxx, gyy = _welch_pair(xa, ya, fs, seg, nov, window)
    psi = _gcc_weight(weighting, gxy, gxx, gyy)
    r = np.fft.irfft(psi * gxy, n=seg)
    lag_samples = np.fft.fftshift(np.fft.fftfreq(seg)) * seg
    curve = np.asarray(np.fft.fftshift(r), dtype=np.float64)
    if max_delay is not None:
        m = round(_positive(max_delay, "max_delay") * fs)
        if not 1 <= m <= (seg - 1) // 2:
            msg = (
                "'max_delay' must be at least one sample and shorter than "
                "half the segment length (increase 'nperseg' for longer "
                "delays)."
            )
            raise ValueError(msg)
        keep = np.abs(lag_samples) <= m
        lag_samples = lag_samples[keep]
        curve = curve[keep]
    return np.asarray(lag_samples, dtype=np.float64), curve, seg


def time_delay(
    x: Signal | NDArray[np.float64] | list[float],
    y: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    method: _Method = "gcc",
    weighting: _Weighting = "phat",
    window: str = "hann",
    nperseg: int | None = None,
    overlap: float = 0.5,
    max_delay: float | None = None,
    interpolation: _Interpolation = "parabolic",
    upsample: int = 1,
    signal_bandwidth: float | None = None,
) -> TimeDelayResult:
    r"""Time delay of ``y`` relative to ``x`` (TDE).

    Three estimators of the delay :math:`\tau_0` in the two-sensor model
    :math:`y(t) = \alpha x(t - \tau_0) + n(t)` (B&P Section 5.1.4):

    * ``'direct'`` - the peak of the full-record correlation coefficient
      function (Eq. 5.21);
    * ``'gcc'`` - the peak of the generalized cross-correlation of
      Knapp & Carter (1976): the Welch-averaged cross-spectrum (shared
      core with :func:`~phonometry.signals.spectra.cross_spectral_density`)
      is weighted by :math:`\psi(f)` before the inverse transform.
      Weightings (Table I): ``'none'`` (plain correlator), ``'roth'``
      (:math:`1/G_{xx}`,
      suppresses bands where the first sensor is noisy), ``'scot'``
      (:math:`1/\sqrt{G_{xx} G_{yy}}`, prewhitens both channels),
      ``'phat'``
      (:math:`1/\lvert G_{xy} \rvert`: for uncorrelated noises the ideal
      GCC is a delta at
      the delay, but errors are accentuated wherever signal power is
      small), and ``'ml'`` (Hannan-Thomson,
      :math:`\lvert \gamma \rvert^2 / (\lvert G_{xy} \rvert
      (1 - \lvert \gamma \rvert^2))`):
      the maximum-likelihood processor that
      weights the phase by its coherence-derived reliability and attains
      the Cramér-Rao bound - provided the coherence estimate is averaged
      over at least two segments, the discrete form of Knapp & Carter's
      :math:`\lvert \gamma \rvert^2 \ne 1` existence condition (enforced
      here). The delay must
      fit within half a segment; raise ``nperseg`` for long delays.
    * ``'phase'`` - the :math:`\lvert \hat{G}_{xy} \rvert`-weighted
      least-squares slope of the
      cross-spectrum phase (Eq. 5.101b); accurate for clean, moderate
      delays where the unwrapped phase is unambiguous, and independent of
      any peak interpolation (``interpolation``/``upsample`` do not
      apply).

    The correlation-peak methods refine the sample peak to sub-sample
    resolution by three-point parabolic interpolation, optionally after
    band-limited local upsampling (``upsample > 1``); this presumes the
    signals are band-limited below Nyquist so the peak is oversampled.
    With ``signal_bandwidth`` given, the result carries the B&P
    peak-location uncertainty (Eq. 8.129) and its :math:`\pm 2\sigma`
    interval (Eq. 8.130).

    :param x: Reference record, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, though a delay is scale-free:
        the Signal is taken here for the rate it carries, and nothing in the
        result moves with the factor.
    :param y: Delayed record, 1-D, same length. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, though a delay is scale-free:
        the Signal is taken here for the rate it carries, and nothing in the
        result moves with the factor.
    :param fs: Sample rate, in Hz. Required when both records are bare
        arrays; either may be a :class:`~phonometry.io.Signal` and supply it,
        and two Signals recorded at different rates are refused rather than
        arbitrated.
    :param method: ``'gcc'`` (default), ``'direct'`` or ``'phase'``.
    :param weighting: GCC weighting (default ``'phat'``; ignored
        otherwise).
    :param window: Welch taper for ``'gcc'``/``'phase'`` (default Hann).
    :param nperseg: Welch segment length for ``'gcc'``/``'phase'``
        (``None`` picks the shared default).
    :param overlap: Welch overlap fraction for ``'gcc'``/``'phase'``.
    :param max_delay: Largest delay magnitude searched, in seconds.
    :param interpolation: ``'parabolic'`` (default) or ``'none'``.
    :param upsample: Integer local-upsampling factor (default 1: off).
    :param signal_bandwidth: Signal bandwidth ``B`` in Hz for the
        Eq. 8.129 delay uncertainty (``None``: no error reported).
    :return: A :class:`TimeDelayResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x", context="a time-delay estimate")
    ya = _validate_signal(y, "y", context="a time-delay estimate")
    fs = resolve_pair_fs(x, y, fs)
    if xa.size != ya.size:
        msg = "'x' and 'y' must have the same length."
        raise ValueError(msg)
    if float(np.std(xa)) <= 0.0 or float(np.std(ya)) <= 0.0:
        msg = (
            "'x' and 'y' must not be constant: there is no correlation peak to locate."
        )
        raise ValueError(msg)
    fs_v = _positive(fs, "fs")
    factor = _validate_refinement(interpolation, upsample)

    weight_name: str | None = None
    if method == "direct":
        lag_samples, curve = _direct_tde_curve(xa, ya, fs_v, max_delay)
        peak = int(np.argmax(np.abs(curve)))
        delay_samples = lag_samples[0] + _refine_peak(
            curve, peak, interpolation, factor
        )
    elif method == "gcc":
        weight_name = weighting
        lag_samples, curve, _seg = _gcc_curve(
            xa, ya, fs_v, weighting, window, nperseg, overlap, max_delay
        )
        peak = int(np.argmax(np.abs(curve)))
        delay_samples = lag_samples[0] + _refine_peak(
            curve, peak, interpolation, factor
        )
        peak_mag = float(np.max(np.abs(curve)))
        if peak_mag > 0.0:
            curve = curve / peak_mag
    elif method == "phase":
        seg, ovl = _validate_welch_params(xa.size, fs_v, nperseg, overlap)
        nov = _noverlap_samples(seg, ovl)
        freqs, gxy, _gxx, _gyy = _welch_pair(xa, ya, fs_v, seg, nov, window)
        delay_samples = _phase_slope_delay(freqs, gxy) * fs_v
        lag_samples, curve, _seg = _gcc_curve(
            xa, ya, fs_v, "none", window, nperseg, overlap, max_delay
        )
        peak_mag = float(np.max(np.abs(curve)))
        if peak_mag > 0.0:
            curve = curve / peak_mag
    else:
        msg = "'method' must be 'gcc', 'direct' or 'phase'."
        raise ValueError(msg)

    rho, std, interval = _delay_error(
        xa, ya, float(delay_samples), signal_bandwidth, fs_v
    )
    return TimeDelayResult(
        delay=float(delay_samples) / fs_v,
        delay_samples=float(delay_samples),
        method=method,
        weighting=weight_name,
        lags=lag_samples / fs_v,
        correlation=curve,
        peak_correlation=rho,
        delay_std=std,
        delay_interval=interval,
        signal_bandwidth=signal_bandwidth,
        fs=fs_v,
    )


# ---------------------------------------------------------------------------
# Impulse-response delay and alignment (sub-sample, over the same peak
# refinement as the TDE estimators)
# ---------------------------------------------------------------------------


def impulse_response_delay(
    ir: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    reference: Signal | NDArray[np.float64] | list[float] | None = None,
    interpolation: _Interpolation = "parabolic",
    upsample: int = 8,
) -> float:
    r"""Sub-sample delay of an impulse response, in seconds.

    Without a reference, the arrival time of the IR itself: the
    cross-correlation of an IR with an ideal unit impulse *is* the IR, so
    its peak magnitude location - refined to sub-sample resolution by
    band-limited local upsampling plus parabolic interpolation - is the
    delay relative to :math:`t = 0`. With a ``reference`` IR, the delay of
    ``ir`` relative to it, from the peak of their full-record
    cross-correlation with the same refinement (one-shot transients are
    not stationary records, so the direct correlator is used rather than
    the Welch-averaged GCC).

    Sub-sample accuracy presumes the IR is band-limited below Nyquist;
    the synthetic fractional-delay tests pin the achievable accuracy
    (about 1e-3 samples for a :math:`0.4 f_\mathrm{s}` band-limited pulse at the
    default ``upsample=8``).

    :param ir: Impulse response, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, though a delay is scale-free:
        nothing in the result moves with the factor.
    :param fs: Sample rate, in Hz. Required when both records are bare
        arrays; either may be a :class:`~phonometry.io.Signal` and supply it,
        and two Signals recorded at different rates are refused rather than
        arbitrated.
    :param reference: Optional reference IR (same length) the delay is
        measured against. Accepts a :class:`phonometry.io.Signal`, which may
        be the side that supplies ``fs``; a delay is scale-free, so its
        calibration does not move the result.
    :param interpolation: ``'parabolic'`` (default) or ``'none'``.
    :param upsample: Integer local-upsampling factor (default 8).
    :return: Delay in seconds (relative to :math:`t = 0` or to ``reference``).
    :raises ValueError: If the inputs or parameters are invalid.
    """
    ira = _validate_signal(ir, "ir", context=_DELAY_CONTEXT)
    if reference is None:
        fs = resolve_fs(ir, fs, name="ir")
    else:
        fs = resolve_pair_fs(ir, reference, fs, names=("ir", "reference"))
    fs_v = _positive(fs, "fs")
    factor = _validate_refinement(interpolation, upsample)
    if reference is not None:
        res = time_delay(
            reference,
            ira,
            fs_v,
            method="direct",
            interpolation=interpolation,
            upsample=factor,
        )
        return res.delay
    peak = int(np.argmax(np.abs(ira)))
    return _refine_peak(ira, peak, interpolation, factor) / fs_v


@dataclass(frozen=True)
class AlignedImpulseResponseResult:
    """An impulse response aligned onto a reference.

    :ivar aligned: The input IR advanced by the estimated delay (exact
        fractional shift applied as a frequency-domain phase ramp over a
        zero-padded record, so nothing wraps around).
    :ivar reference: The reference IR.
    :ivar delay: Estimated delay removed from the IR, in seconds.
    :ivar delay_samples: The same delay in (fractional) samples.
    :ivar fs: Sample rate, in Hz.
    """

    aligned: NDArray[np.float64]
    reference: NDArray[np.float64]
    delay: float
    delay_samples: float
    fs: float

    def __post_init__(self) -> None:
        """Reject a pair of impulse responses that are not comparable.

        The point of the result is that the two records can now be read
        sample against sample -- averaged into an ensemble, subtracted,
        overlaid. :meth:`plot` does exactly that, building one time axis from
        :attr:`reference` and drawing both traces on it. Two records of
        different lengths were never comparable in the first place, and
        :attr:`delay_samples`, a fractional shift stated in samples of a rate
        both are supposed to share, has no meaning between them; that half is
        loud, refused from inside matplotlib for want of a matching first
        dimension, though without naming the field that is wrong.

        A second axis on :attr:`aligned` is the half nothing catches. Its
        first axis still counts one value per sample, so the lengths agree,
        and :meth:`plot` draws one trace per column against that single time
        axis, repeating the aligned label and the one stored delay in the
        legend for a column that was never aligned. On :attr:`reference` the
        same extra axis is loud, because the time axis is built from its
        total number of elements and so comes out as long as the whole grid.

        :raises ValueError: if the two records differ in length, or either
            carries a second axis.
        """
        require_ranks(self, aligned=1, reference=1)
        require_same_length(self, "aligned", "reference", axis="sample")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the reference and the aligned impulse response.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_aligned_impulse_response

        check_language(language)
        return plot_aligned_impulse_response(self, ax=ax, language=language, **kwargs)


def align_impulse_responses(
    ir: Signal | NDArray[np.float64] | list[float],
    reference: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    interpolation: _Interpolation = "parabolic",
    upsample: int = 8,
) -> AlignedImpulseResponseResult:
    """Align an impulse response onto a reference by its estimated delay.

    Estimates the sub-sample delay of ``ir`` relative to ``reference``
    (:func:`impulse_response_delay`) and removes it with an exact
    band-limited fractional shift (frequency-domain phase ramp over a
    zero-padded record). Use it to average IR ensembles or to compare
    measurements taken at slightly different distances.

    :param ir: Impulse response to align, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the aligned pair comes out
        in Pa. The delay between them is scale-free and does not move.
    :param reference: Reference impulse response, same length. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the aligned pair comes out
        in Pa. The delay between them is scale-free and does not move.
    :param fs: Sample rate, in Hz. Required when both records are bare
        arrays; either may be a :class:`~phonometry.io.Signal` and supply it,
        and two Signals recorded at different rates are refused rather than
        arbitrated.
    :param interpolation: ``'parabolic'`` (default) or ``'none'``.
    :param upsample: Integer local-upsampling factor (default 8).
    :return: An :class:`AlignedImpulseResponseResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    ira = _validate_signal(ir, "ir", context=_DELAY_CONTEXT)
    refa = _validate_signal(reference, "reference", context=_DELAY_CONTEXT)
    if ira.size != refa.size:
        msg = "'ir' and 'reference' must have the same length."
        raise ValueError(msg)
    fs_v = _positive(
        resolve_pair_fs(ir, reference, fs, names=("ir", "reference")), "fs"
    )
    delay = impulse_response_delay(
        ira,
        fs_v,
        reference=refa,
        interpolation=interpolation,
        upsample=upsample,
    )
    aligned = _fractional_advance(ira, delay * fs_v)
    return AlignedImpulseResponseResult(
        aligned=aligned,
        reference=refa.copy(),
        delay=delay,
        delay_samples=delay * fs_v,
        fs=fs_v,
    )
