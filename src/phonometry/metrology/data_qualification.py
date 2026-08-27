#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Random-data qualification: stationarity tests and Rice crossing statistics.

Before a record is averaged into a PSD, condensed into a Leq or fed to a GUM
budget, Bendat & Piersol (*Random Data*, 4th ed., Sec. 10.3) require it to be
**qualified**: classified as stationary or not, checked for periodicities and
validated against acquisition anomalies. This module implements the two
quantitative pillars of that chapter.

**Stationarity** (B&P Sec. 10.3.1.1). The record is divided into ``N`` equal
intervals, a mean square value (or another moment) is computed for each, and
the sequence is tested for underlying trends with a *nonparametric* test --
no assumption about the sampling distribution of the values is needed. The
book's test is the **reverse arrangement test** (Sec. 4.5.2): count the pairs
:math:`i < j` with :math:`x_i > x_j`. For an ordered sequence of independent
observations that count ``A`` has mean ``N(N-1)/4`` (Eq. (4.54)) and variance
``N(2N+5)(N-1)/72`` (Eq. (4.55)), and the hypothesis of no trend is accepted
at significance ``alpha`` when ``A`` falls inside the two-sided acceptance
region of Table A.6 (for :math:`N = 20` and :math:`\alpha = 0.05`: more than
64 and at most 125 reverse arrangements, the book's Examples 4.4 and 10.3).
The classical companion **runs test** (the run distribution tabulated in the
third edition; Wald & Wolfowitz 1940) classifies each value as above or below
the sequence median and counts runs of like classification: a trend produces
too few runs, rapid alternation too many. Both tests are distribution-free
and, per Sec. 10.3.1.1, work equally well on mean values, rms values,
standard deviations or any other parameter sequence.

**Level crossings and peaks** (B&P Sec. 5.5, originally Rice). For a zero
mean value stationary Gaussian record with one-sided autospectrum ``G(f)``,
the geometric moments

.. math::

   \sigma_x^2 = \int G \, df, \qquad
   \sigma_v^2 = \int (2\pi f)^2 G \, df, \qquad
   \sigma_a^2 = \int (2\pi f)^4 G \, df

(Eqs. (5.214)-(5.216)) determine every crossing and peak statistic:

* the expected number of zero crossings per unit time (both slopes),
  :math:`N_0 = (1/\pi)(\sigma_v / \sigma_x)` (Eq. (5.195)) -- equivalently
  :math:`2\sqrt{m_2 / m_0}` with the plain frequency moments
  :math:`m_k = \int f^k G \, df`; twice the record's *apparent frequency*;
* the expected number of crossings of level ``a``,
  :math:`N_a = N_0 e^{-a^2 / (2\sigma_x^2)}` (Eq. (5.196));
* the expected number of maxima per unit time,
  :math:`M = (1/2\pi)(\sigma_a / \sigma_v) = \sqrt{m_4 / m_2}`
  (Eq. (5.211));
* the **irregularity factor** :math:`r = N_0 / (2M)`, between 0 and 1
  (Eq. (5.220)): 1 for narrow bandwidth data (one maximum per zero-crossing
  cycle), toward 0 as ever more local maxima ride on each cycle;
* the peak probability functions: Rayleigh for narrow bandwidth data
  (Eqs. (5.206)-(5.207)), the standardized Gaussian for :math:`r \to 0`
  (Eq. (5.221)) and, in between, Rice's mixture for the standardized peak
  height ``z`` (Eqs. (5.217) and (5.223) with
  :math:`\epsilon = \sqrt{1 - r^2}`):

  .. math::

     P[\text{peak} > z] = Q(z / \epsilon)
     + r\, e^{-z^2 / 2} \left[ 1 - Q(rz / \epsilon) \right]

  where ``Q`` is the standardized normal exceedance (Eq. (5.250)).

:func:`stationarity_test` and :func:`trend_test` implement the first block,
:func:`level_crossing_rate` and :func:`peak_statistics` the second, each
comparing the counts measured on the record with the closed-form
expectations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import special

from .._internal.validation import (
    require_axis_count,
    require_equal_counts,
    require_finite_fields,
    require_ranks,
    require_same_length,
)
from ..io._resolve import resolve_fs
from ..signals.spectra import _positive, _validate_signal, power_spectral_density

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..io._signal import Signal

__all__ = [
    "LevelCrossingResult",
    "PeakStatisticsResult",
    "StationarityTestResult",
    "TrendTestResult",
    "level_crossing_rate",
    "peak_statistics",
    "stationarity_test",
    "trend_test",
]

#: Trend-test methods accepted by :func:`trend_test`.
_METHODS = ("reverse_arrangements", "runs")

#: Segment statistics accepted by :func:`stationarity_test`.
_STATISTICS = ("mean_square", "rms", "mean", "variance")

#: Largest sequence length for which the reverse-arrangement p-value uses
#: the exact (Mahonian) null distribution; beyond it the normal
#: approximation of B&P Eqs. (4.54)-(4.55) takes over. Table A.6 ends here.
_EXACT_LIMIT = 100

#: Fewest observations a trend test accepts (Table A.6 starts at N = 10).
_MIN_OBSERVATIONS = 10

#: Threshold on the squared Rice bandwidth parameter
#: :math:`\epsilon^2 = 1 - r^2` below which (:math:`\epsilon \le` ~1e-15,
#: ``r`` equal to 1 to machine precision) the standardized peak-height
#: distribution of B&P Eqs. (5.217)/(5.223) is treated as exactly Rayleigh
#: (the narrow-bandwidth limit, Eqs. (5.206)/(5.222)), avoiding division
#: by :math:`\epsilon`.
_EPS_SQ_RAYLEIGH_LIMIT = 1e-30


# ---------------------------------------------------------------------------
# Reverse arrangement test (B&P Sec. 4.5.2, Table A.6)
# ---------------------------------------------------------------------------


def _reverse_arrangements(values: NDArray[np.float64]) -> int:
    r"""Count reverse arrangements ``A``: pairs :math:`i < j` with
    :math:`x_i > x_j`.

    B&P Eqs. (4.51)-(4.53). Equal values do not count (the inequality is
    strict), matching the book's definition :math:`h_{ij} = 1` iff
    :math:`x_i > x_j`.
    """
    greater = values[:, np.newaxis] > values[np.newaxis, :]
    return int(np.sum(np.triu(greater, k=1)))


def _reverse_arrangement_moments(n: int) -> tuple[float, float]:
    """Null mean and standard deviation of ``A`` (B&P Eqs. (4.54)-(4.55))."""
    mean = n * (n - 1) / 4.0
    variance = n * (2 * n + 5) * (n - 1) / 72.0
    return mean, float(np.sqrt(variance))


def _mahonian_cdf(n: int) -> NDArray[np.float64]:
    r"""Exact null CDF of the reverse-arrangement count for ``n`` values.

    For independent continuous observations the ranks form a uniformly
    random permutation and ``A`` is its inversion count, whose distribution
    (the Mahonian distribution) follows from convolving uniform steps: the
    ``i``-th value inserted into the running order creates 0 to ``i - 1``
    new inversions with equal probability, independently of the previous
    ones. :math:`\mathrm{cdf}[k] = P(A \le k)`,
    :math:`k = 0, \ldots, n(n-1)/2`.
    """
    pmf = np.ones(1)
    for i in range(2, n + 1):
        padded = np.concatenate((np.zeros(1), np.cumsum(pmf)))
        size = pmf.size + i - 1
        upper = np.minimum(np.arange(1, size + 1), pmf.size)
        lower = np.maximum(np.arange(size) - i + 1, 0)
        pmf = (padded[upper] - padded[lower]) / i
    return np.asarray(np.cumsum(pmf), dtype=np.float64)


def _reverse_arrangement_bounds(n: int, alpha: float) -> tuple[int, int]:
    r"""Two-sided acceptance bounds on ``A``: accept ``lower < A <= upper``.

    Percentage points ``A_{N;a}`` with :math:`P(A > A_{N;a}) = a` for
    :math:`a = 1 - \alpha/2` (lower) and :math:`a = \alpha/2` (upper),
    computed from the normal approximation with the B&P moments and a
    continuity correction. This reproduces every :math:`\alpha = 0.05`
    entry of B&P Table A.6 (``N`` = 10 to 100), e.g. ``(64, 125)`` for
    :math:`N = 20` -- the book's Examples 4.4 and 10.3.

    The normal approximation is kept *deliberately*, for all ``N``: the
    book's table itself follows it. No percentage-point convention on the
    exact Mahonian null distribution reproduces every tabulated entry
    (checked digit by digit: e.g. at :math:`N = 40`, :math:`a = 0.975` the
    exact tail jumps from :math:`P(A > 304) = 0.9771` to
    :math:`P(A > 305) = 0.9758`, neither equal to 0.975, and the
    nearest-tail point would be 306, yet the table prints 305 -- exactly
    where the continuity-corrected normal point 305.39 rounds to;
    :math:`N = 100`, :math:`a = 0.975` behaves the same way).
    Exact-distribution bounds would therefore *disagree* with the
    published Table A.6 that users check against. The p-value, which has
    no tabulated convention to honour, does use the exact distribution
    (:func:`_reverse_arrangement_p_value`), so verdict and p-value can
    disagree by at most one count right at an acceptance boundary.
    """
    mean, std = _reverse_arrangement_moments(n)
    spread = float(special.ndtri(1.0 - alpha / 2.0)) * std
    lower = round(mean - spread - 0.5)
    upper = round(mean + spread - 0.5)
    return int(lower), int(upper)


def _reverse_arrangement_p_value(n: int, statistic: int) -> float:
    """Exact (``n`` <= 100) or normal-approximated two-sided p-value."""
    if n <= _EXACT_LIMIT:
        cdf = _mahonian_cdf(n)
        below = float(cdf[statistic])
        above = 1.0 - (float(cdf[statistic - 1]) if statistic > 0 else 0.0)
    else:
        mean, std = _reverse_arrangement_moments(n)
        below = float(special.ndtr((statistic + 0.5 - mean) / std))
        above = float(special.ndtr(-(statistic - 0.5 - mean) / std))
    return min(1.0, 2.0 * min(below, above))


# ---------------------------------------------------------------------------
# Runs test about the median (Wald & Wolfowitz 1940)
# ---------------------------------------------------------------------------


def _log_comb(n: int, k: NDArray[np.int64]) -> NDArray[np.float64]:
    r"""``ln C(n, k)`` element-wise, ``-inf`` outside :math:`0 \le k \le n`."""
    kk = np.asarray(k, dtype=np.float64)
    valid = (kk >= 0.0) & (kk <= n)
    safe = np.where(valid, kk, 0.0)
    out = (
        special.gammaln(n + 1.0)
        - special.gammaln(safe + 1.0)
        - special.gammaln(n - safe + 1.0)
    )
    return np.asarray(np.where(valid, out, -np.inf), dtype=np.float64)


def _runs_pmf(n1: int, n2: int) -> NDArray[np.float64]:
    r"""Exact null PMF of the number of runs, indexed
    :math:`r = 0, \ldots, n_1+n_2`.

    Wald & Wolfowitz (1940): conditional on ``n1`` values of one kind and
    ``n2`` of the other, all ``C(n1+n2, n1)`` orderings are equally likely
    under independence, and the number of runs ``r`` has the classical
    combinatorial distribution (even/odd cases). Computed in log space so
    any ``n1 + n2`` is exact to double precision.
    """
    n = n1 + n2
    log_total = float(
        special.gammaln(n + 1.0) - special.gammaln(n1 + 1.0) - special.gammaln(n2 + 1.0)
    )
    pmf = np.zeros(n + 1)
    r = np.arange(2, n + 1)
    k = r // 2
    even = r % 2 == 0
    log_even = np.log(2.0) + _log_comb(n1 - 1, k - 1) + _log_comb(n2 - 1, k - 1)
    with np.errstate(divide="ignore"):
        odd_a = _log_comb(n1 - 1, k) + _log_comb(n2 - 1, k - 1)
        odd_b = _log_comb(n1 - 1, k - 1) + _log_comb(n2 - 1, k)
        log_odd = np.logaddexp(odd_a, odd_b)
    log_count = np.where(even, log_even, log_odd)
    with np.errstate(invalid="ignore"):
        pmf[2:] = np.where(np.isfinite(log_count), np.exp(log_count - log_total), 0.0)
    return pmf


def _runs_moments(n1: int, n2: int) -> tuple[float, float]:
    """Null mean and standard deviation of the number of runs."""
    n = n1 + n2
    mean = 1.0 + 2.0 * n1 * n2 / n
    variance = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    return mean, float(np.sqrt(variance))


def _runs_bounds(n1: int, n2: int, alpha: float) -> tuple[int, int]:
    r"""Two-sided acceptance bounds on ``r``: accept ``lower < r <= upper``.

    Conservative percentage points from the exact distribution: ``lower``
    is the largest count with ``P(r <= lower) <= alpha/2`` and ``upper``
    the largest count with ``P(r > upper) <= alpha/2``, so the two-sided
    size never exceeds ``alpha``. For :math:`n_1 = n_2 = 10` at
    :math:`\alpha = 0.05` this is ``(6, 15)`` -- the classical tabulated
    critical values (reject at 6 or fewer, or 16 or more, runs).
    """
    cdf = np.cumsum(_runs_pmf(n1, n2))
    half = alpha / 2.0
    below = np.flatnonzero(cdf <= half)
    lower = int(below[-1]) if below.size else 1
    above = np.flatnonzero(1.0 - cdf <= half)
    upper = int(above[0]) if above.size else n1 + n2
    return lower, upper


def _runs_p_value(n1: int, n2: int, statistic: int) -> float:
    """Exact two-sided p-value of an observed run count."""
    cdf = np.cumsum(_runs_pmf(n1, n2))
    below = float(cdf[statistic])
    above = 1.0 - float(cdf[statistic - 1])
    return min(1.0, 2.0 * min(below, above))


def _count_runs(flags: NDArray[np.bool_]) -> int:
    """Number of runs of equal values in a boolean sequence."""
    return int(np.count_nonzero(flags[1:] != flags[:-1])) + 1


# ---------------------------------------------------------------------------
# Trend and stationarity tests
# ---------------------------------------------------------------------------

#: Where each null mean is written down, by the method that counts it. Only
#: the two keys here yield a moment in :func:`_null_trend_mean`, so a message
#: is only ever built for a method this maps.
_NULL_MEAN_SOURCES = {
    "reverse_arrangements": "B&P Eq. (4.54)",
    "runs": "Wald & Wolfowitz 1940",
}


def _null_trend_mean(
    values: NDArray[np.float64], method: str, median: float | None
) -> float | None:
    r"""Null mean of the trend statistic counted on *values*, or ``None``.

    Restates what the producers store: ``n(n-1)/4`` over the whole sequence
    for reverse arrangements (B&P Eq. (4.54)), and the Wald & Wolfowitz
    ``1 + 2 n_1 n_2 / n`` for runs, counted on the values that differ from
    *median* -- the classification median, which for a
    :class:`TrendTestResult` is the one it kept and for a
    :class:`StationarityTestResult` is the median of the segment sequence
    :func:`trend_test` took itself. Re-filtering an already filtered
    sequence removes nothing, so one form serves both.

    ``None`` where no moment is determined: a ``method`` this module does
    not count, a runs result carrying no classification median, or a runs
    sequence with nothing on one side of it. There is no null mean to
    measure a hand-built result against in any of those, and inventing one
    would refuse a result the producers simply never reach.

    :param values: The sequence the statistic was counted on.
    :param method: ``"reverse_arrangements"`` or ``"runs"``.
    :param median: Classification median for ``"runs"``, else ignored.
    :return: The null mean, or ``None`` when it is not determined.
    """
    seq = np.asarray(values, dtype=np.float64)
    if method == "reverse_arrangements":
        return _reverse_arrangement_moments(seq.size)[0]
    if method != "runs" or median is None:
        return None
    kept = seq[np.abs(seq - median) > 0.0]
    n1 = int(np.count_nonzero(kept > median))
    n2 = int(kept.size - n1)
    if n1 == 0 or n2 == 0:
        return None
    return _runs_moments(n1, n2)[0]


def _check_null_mean(
    owner: str,
    sequence: str,
    values: NDArray[np.float64],
    method: str,
    median: float | None,
    mean: float,
) -> None:
    """Require ``mean`` to restate the sequence it was counted on.

    ``mean`` is not an average of the values beside it: it is the mean of
    the *statistic* under the no-trend hypothesis, fixed by the method and
    by how many observations were counted. That makes it a closed function
    of fields the result already carries, and the one the reader divides by
    -- ``(statistic - mean) / std`` is the standardized departure the whole
    test is read through, and it is computed by the caller, not by any
    renderer here, so a mean that belongs to another sequence is never
    caught downstream. A billionth is allowed so a caller who recomputed
    the moment along another floating-point path is not refused over the
    last bit.

    The two methods do not share a moment or a source, so the message cites
    the one the count came from: ``n(n-1)/4`` is B&P Eq. (4.54), and
    ``1 + 2 n_1 n_2 / n`` is Wald & Wolfowitz. Any other method leaves
    :func:`_null_trend_mean` with nothing to state and returns above.

    :param owner: Name of the result type, used in the error message.
    :param sequence: Name of the field the statistic was counted on.
    :param values: That sequence.
    :param method: The trend test the count came from.
    :param median: Classification median for ``"runs"``, else ignored.
    :param mean: The stored null mean.
    :raises ValueError: if the stored mean is not the null mean of the
        statistic counted on that sequence.
    """
    expected = _null_trend_mean(values, method, median)
    if expected is None or math.isclose(mean, expected, rel_tol=0.0, abs_tol=1e-9):
        return
    msg = (
        f"{owner}: 'mean' must be the null mean of the {method!r} statistic "
        f"counted on '{sequence}' ({_NULL_MEAN_SOURCES[method]}); got "
        f"{mean!r} where that sequence states {expected!r}."
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class TrendTestResult:
    r"""A nonparametric trend test on a sequence of parameter estimates.

    The hypothesis of *no underlying trend* is accepted at significance
    :attr:`alpha` when :attr:`statistic` lies inside the two-sided
    acceptance region ``lower < statistic <= upper`` (B&P Sec. 4.5.2 and
    Table A.6 for reverse arrangements); :attr:`p_value` is the exact
    two-sided tail probability of the observed count.

    :ivar values: The tested sequence (for ``"runs"``, after discarding
        values exactly equal to the median).
    :ivar method: ``"reverse_arrangements"`` or ``"runs"``.
    :ivar statistic: Observed count: reverse arrangements ``A`` or runs
        ``r``.
    :ivar n: Number of observations used.
    :ivar mean: Null mean of the statistic (B&P Eq. (4.54) for ``A``).
    :ivar std: Null standard deviation (B&P Eq. (4.55) for ``A``).
    :ivar bounds: Acceptance region ``(lower, upper)``: percentage points
        such that the no-trend hypothesis is accepted when
        ``lower < statistic <= upper``. For reverse arrangements these
        follow B&P Table A.6's own convention (normal approximation with
        continuity correction, which reproduces the book's tabulated
        :math:`\alpha = 0.05` entries exactly; the table is not derivable
        from the exact Mahonian distribution), so at an acceptance
        boundary the verdict and the exact :attr:`p_value` can disagree
        by one count.
    :ivar p_value: Two-sided p-value from the exact null distribution
        (normal approximation above :math:`n = 100` for reverse
        arrangements).
    :ivar trend_free: ``True`` when the statistic falls inside the
        acceptance region.
    :ivar alpha: Significance level of the region (default 0.05).
    :ivar median: For ``"runs"``, the median of the *original* sequence
        against which each value was classified (before values equal to it
        were discarded); ``None`` for ``"reverse_arrangements"``.
    """

    values: NDArray[np.float64]
    method: str
    statistic: int
    n: int
    mean: float
    std: float
    bounds: tuple[int, int]
    p_value: float
    trend_free: bool
    alpha: float
    median: float | None = None

    def __post_init__(self) -> None:
        """Reject a sequence that disagrees with the count it was tested at.

        The verdict is a statement about one sequence of :attr:`n`
        observations: the statistic was counted on that sequence, and the
        null mean, the standard deviation, the acceptance region and the
        p-value all follow from the count. The plot then draws
        :attr:`values` against a sample index counted out of the declared
        :attr:`n` rather than out of the array beside it, so a sequence of
        another length is a plot of a different record than the one the
        legend rates. Matplotlib does refuse to draw the two against each
        other, but from inside the renderer and about first dimensions,
        naming neither the field nor the result; the declared count is the
        authority here, and checking it at construction says which sequence
        moved.

        :attr:`mean` is pinned for the same reason one step further in: it
        is the null mean of the *statistic*, ``n(n-1)/4`` for reverse
        arrangements, not an average of :attr:`values`, so it is fixed by
        the count and the method and nothing else. Nothing here reads it --
        the plot states the count and the bounds -- which is exactly why it
        goes unchecked otherwise: the caller is the reader, standardizing
        the departure as ``(statistic - mean) / std``, and a mean belonging
        to another sequence turns an ordinary count into a flat rejection
        with no array anywhere the wrong length.

        :raises ValueError: if the tested sequence spans a different number
            of observations than the declared count, or the null mean is not
            the one that sequence and method state.
        """
        require_ranks(self, values=1)
        owner = type(self).__name__
        require_equal_counts(
            owner,
            {
                "n": self.n,
                "values": require_axis_count(
                    self.values, owner, "values", "observation", rank=None
                ),
            },
            "observation",
        )
        _check_null_mean(
            owner, "values", self.values, self.method, self.median, self.mean
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the tested sequence against its sample index with the verdict.

        Draws the sequence of observations against a plain sample index and
        states the outcome in the legend: the reverse-arrangement count
        ``A`` (or the run count ``r``), the acceptance region and whether
        the no-trend hypothesis is accepted. For the runs test the
        classification median is drawn as a reference line.

        :param ax: Existing :class:`~matplotlib.axes.Axes` to draw on, or
            ``None`` (default) to create a fresh figure and axes.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Extra keyword arguments forwarded to the sequence
            ``plot`` call (e.g. ``color``, ``lw``, ``marker``).
        :return: The :class:`~matplotlib.axes.Axes` the sequence was drawn
            on, so the figure can be composed further.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_trend_test

        check_language(language)
        return plot_trend_test(self, ax=ax, language=language, **kwargs)


def _trend_test_reverse(values: NDArray[np.float64], alpha: float) -> TrendTestResult:
    n = values.size
    statistic = _reverse_arrangements(values)
    mean, std = _reverse_arrangement_moments(n)
    bounds = _reverse_arrangement_bounds(n, alpha)
    return TrendTestResult(
        values=values,
        method="reverse_arrangements",
        statistic=statistic,
        n=n,
        mean=mean,
        std=std,
        bounds=bounds,
        p_value=_reverse_arrangement_p_value(n, statistic),
        trend_free=bounds[0] < statistic <= bounds[1],
        alpha=alpha,
    )


def _trend_test_runs(values: NDArray[np.float64], alpha: float) -> TrendTestResult:
    median = float(np.median(values))
    keep = np.abs(values - median) > 0.0
    kept = values[keep]
    if kept.size < _MIN_OBSERVATIONS:
        msg = (
            "The runs test needs at least "
            f"{_MIN_OBSERVATIONS} observations distinct from the median; "
            f"got {kept.size}."
        )
        raise ValueError(msg)
    above = kept > median
    n1 = int(np.count_nonzero(above))
    n2 = int(above.size - n1)
    if n1 == 0 or n2 == 0:
        msg = "The runs test needs observations on both sides of the median."
        raise ValueError(msg)
    statistic = _count_runs(above)
    mean, std = _runs_moments(n1, n2)
    bounds = _runs_bounds(n1, n2, alpha)
    return TrendTestResult(
        values=kept,
        method="runs",
        statistic=statistic,
        n=int(kept.size),
        mean=mean,
        std=std,
        bounds=bounds,
        p_value=_runs_p_value(n1, n2, statistic),
        trend_free=bounds[0] < statistic <= bounds[1],
        alpha=alpha,
        median=median,
    )


def _validate_trend_inputs(
    values: NDArray[np.float64] | list[float], method: str, alpha: float
) -> NDArray[np.float64]:
    seq = np.asarray(values, dtype=np.float64)
    if seq.ndim != 1:
        msg = "'values' must be one-dimensional."
        raise ValueError(msg)
    if seq.size < _MIN_OBSERVATIONS:
        msg = (
            f"'values' must hold at least {_MIN_OBSERVATIONS} observations "
            f"(B&P Table A.6 starts at N = 10); got {seq.size}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(seq)):
        msg = "'values' must be finite."
        raise ValueError(msg)
    if method not in _METHODS:
        msg = f"'method' must be one of {_METHODS}, got {method!r}."
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"'alpha' must be inside (0, 1), got {alpha!r}."
        raise ValueError(msg)
    return seq


def trend_test(
    values: NDArray[np.float64] | list[float],
    *,
    method: str = "reverse_arrangements",
    alpha: float = 0.05,
) -> TrendTestResult:
    r"""Nonparametric trend test on a sequence of observations or estimates.

    Tests the hypothesis that the sequence holds independent observations
    of one random variable -- no underlying trend -- without assuming any
    sampling distribution (B&P Sec. 4.5.2). Two classical statistics:

    * ``"reverse_arrangements"`` (default): the count ``A`` of pairs
      :math:`i < j` with :math:`x_i > x_j`. A downward trend inflates
      ``A``, an upward trend depresses it. The acceptance region
      reproduces B&P Table A.6 (:math:`\alpha = 0.05`, ``N`` = 10 to
      100); the p-value uses the exact Mahonian null distribution up to
      :math:`N = 100`.
    * ``"runs"``: values are classified against the sequence median and
      runs of like classification are counted (the run test of B&P's
      earlier editions; Wald & Wolfowitz 1940). Trends and slow drifts
      produce too few runs, alternation too many. Exact conditional
      distribution at any ``N``; values equal to the median are discarded.

    The reverse arrangement test is the more powerful of the two against
    monotonic trends (B&P Sec. 4.5.2); the runs test also reacts to
    non-monotonic clustering.

    :param values: Sequence of observations or parameter estimates
        (e.g. segment mean square values), at least 10.
    :param method: ``"reverse_arrangements"`` (default) or ``"runs"``.
    :param alpha: Two-sided significance level (default 0.05, the level
        tabulated by B&P).
    :return: A :class:`TrendTestResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    seq = _validate_trend_inputs(values, method, alpha)
    if method == "reverse_arrangements":
        return _trend_test_reverse(seq, alpha)
    return _trend_test_runs(seq, alpha)


@dataclass(frozen=True)
class StationarityTestResult:
    """A stationarity test on segment statistics of a single record.

    The record was divided into :attr:`n_segments` equal intervals, the
    per-segment :attr:`statistic` sequence was computed, and the sequence
    was tested for trends with :func:`trend_test` (the B&P Sec. 10.3.1.1
    procedure). The hypothesis of stationarity is accepted when
    ``bounds[0] < count <= bounds[1]``.

    :ivar segment_values: Per-segment values of the tested statistic.
    :ivar segment_times: Segment centre times, in seconds.
    :ivar statistic: The per-segment statistic: ``"mean_square"``,
        ``"rms"``, ``"mean"`` or ``"variance"``.
    :ivar method: Trend test used: ``"reverse_arrangements"`` or
        ``"runs"``.
    :ivar count: Observed test statistic (reverse arrangements or runs).
    :ivar mean: Null mean of the count.
    :ivar std: Null standard deviation of the count.
    :ivar bounds: Acceptance region ``(lower, upper)`` at :attr:`alpha`.
    :ivar p_value: Two-sided p-value of the observed count.
    :ivar stationary: ``True`` when the count falls inside the region.
    :ivar alpha: Significance level (default 0.05).
    :ivar n_segments: Number of segments the record was divided into.
    :ivar segment_duration: Duration of each segment, in seconds.
    :ivar fs: Sample rate of the record, in Hz.
    """

    segment_values: NDArray[np.float64]
    segment_times: NDArray[np.float64]
    statistic: str
    method: str
    count: int
    mean: float
    std: float
    bounds: tuple[int, int]
    p_value: float
    stationary: bool
    alpha: float
    n_segments: int
    segment_duration: float
    fs: float

    def __post_init__(self) -> None:
        """Reject a test whose per-segment sequences disagree.

        The verdict is a statement about one sequence of segment values, and
        everything the reader is shown has to be that same sequence: the plot
        draws the values against a segment index counted out of
        :attr:`n_segments` rather than out of the array beside it, and
        :attr:`segment_times` places those values on the record's time axis
        while nothing downstream re-derives it. A sequence of another length
        than the count it is drawn against is a plot of a different record
        than the one the legend rates, so the declared count is taken as the
        authority here and the two sequences have to follow it.

        :attr:`mean` follows from the same sequence: it is the null mean of
        the count, not an average of :attr:`segment_values`, and
        :func:`stationarity_test` takes it from the trend test it ran on
        exactly this array -- for ``"runs"``, on the values that differ from
        the array's own median, which is the median
        :func:`~phonometry._plot.metrology.plot_stationarity_test` draws.
        So it is recomputable from what is stored, and worth recomputing:
        the caller standardizes the count as ``(count - mean) / std`` and no
        renderer here would notice a mean belonging to another record.

        :raises ValueError: if either per-segment sequence disagrees with the
            other or with the declared number of segments, or the null mean
            is not the one those segment values and that method state.
        """
        require_ranks(self, segment_values=1, segment_times=1)
        require_same_length(self, "segment_values", "segment_times", axis="segment")
        owner = type(self).__name__
        require_equal_counts(
            owner,
            {
                "n_segments": self.n_segments,
                "segment_values": require_axis_count(
                    self.segment_values, owner, "segment_values", "segment", rank=None
                ),
            },
            "segment",
        )
        centre = (
            float(np.median(self.segment_values)) if self.method == "runs" else None
        )
        _check_null_mean(
            owner, "segment_values", self.segment_values, self.method, centre, self.mean
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the segment-statistic sequence with the test verdict.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_stationarity_test

        check_language(language)
        return plot_stationarity_test(self, ax=ax, language=language, **kwargs)


def _segment_statistic(
    segments: NDArray[np.float64], statistic: str
) -> NDArray[np.float64]:
    """Per-row statistic of the segment matrix (B&P Sec. 10.3.1.1 step 2)."""
    if statistic == "mean_square":
        return np.asarray(np.mean(segments**2, axis=1), dtype=np.float64)
    if statistic == "rms":
        return np.asarray(np.sqrt(np.mean(segments**2, axis=1)), dtype=np.float64)
    if statistic == "mean":
        return np.asarray(np.mean(segments, axis=1), dtype=np.float64)
    return np.asarray(np.var(segments, axis=1), dtype=np.float64)


def stationarity_test(
    x: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    n_segments: int = 20,
    statistic: str = "mean_square",
    method: str = "reverse_arrangements",
    alpha: float = 0.05,
) -> StationarityTestResult:
    r"""Test a record for stationarity via trends in segment statistics.

    The B&P Sec. 10.3.1.1 procedure: (1) divide the record into
    ``n_segments`` equal intervals, assumed long enough for the values to
    be independent; (2) compute a mean square value (or another moment)
    for each interval; (3) test the sequence for underlying trends with a
    nonparametric test -- no knowledge of the record's bandwidth or
    sampling distribution required. Nonstationarities that show in the
    mean square value (the usual case: gain drifts, level ramps, machinery
    running up) are detected regardless of the record's spectrum; the
    default 20 segments matches the book's worked Example 10.3, which
    rejects stationarity for a noise record with a 20 % gain ramp
    (:math:`A = 52`, below the Table A.6 lower bound of 64).

    Any trailing samples that do not fill the last segment are discarded.
    The segment count trades resolution against independence: each
    segment must remain long against the record's lowest frequencies.

    :param x: Signal, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the per-segment values come
        out in Pa² for the squared statistics (``'mean_square'``, the
        default, and ``'variance'``) and in Pa for ``'rms'`` and ``'mean'``.
        The verdict, the bounds and the p-value are scale-free.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param n_segments: Number of equal intervals (default 20, as in B&P
        Example 10.3); at least 10, at most the record length in samples.
    :param statistic: Per-segment statistic: ``"mean_square"`` (default,
        the book's choice), ``"rms"``, ``"mean"`` or ``"variance"``.
    :param method: Trend test: ``"reverse_arrangements"`` (default) or
        ``"runs"``.
    :param alpha: Two-sided significance level (default 0.05).
    :return: A :class:`StationarityTestResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x", context="a stationarity test")
    fs_v = _positive(resolve_fs(x, fs), "fs")
    if statistic not in _STATISTICS:
        msg = f"'statistic' must be one of {_STATISTICS}, got {statistic!r}."
        raise ValueError(msg)
    segments = int(n_segments)
    if not _MIN_OBSERVATIONS <= segments <= xa.size:
        msg = (
            f"'n_segments' must be between {_MIN_OBSERVATIONS} and the "
            f"record length ({xa.size} samples), got {n_segments!r}."
        )
        raise ValueError(msg)

    per_segment = xa.size // segments
    trimmed = xa[: per_segment * segments].reshape(segments, per_segment)
    values = _segment_statistic(trimmed, statistic)
    trend = trend_test(values, method=method, alpha=alpha)

    duration = per_segment / fs_v
    times = (np.arange(segments, dtype=np.float64) + 0.5) * duration
    return StationarityTestResult(
        segment_values=values,
        segment_times=times,
        statistic=statistic,
        method=method,
        count=trend.statistic,
        mean=trend.mean,
        std=trend.std,
        bounds=trend.bounds,
        p_value=trend.p_value,
        stationary=trend.trend_free,
        alpha=alpha,
        n_segments=segments,
        segment_duration=duration,
        fs=fs_v,
    )


# ---------------------------------------------------------------------------
# Rice level crossings and peaks (B&P Sec. 5.5)
# ---------------------------------------------------------------------------


def _spectral_moments(
    x: NDArray[np.float64], fs: float, nperseg: int | None
) -> tuple[float, float, float]:
    r"""Frequency moments ``m0, m2, m4`` of the one-sided Welch autospectrum.

    :math:`m_k = \int f^k G(f) \, df` -- the B&P Eqs. (5.214)-(5.216)
    variances are :math:`\sigma_x^2 = m_0`,
    :math:`\sigma_v^2 = (2\pi)^2 m_2` and
    :math:`\sigma_a^2 = (2\pi)^4 m_4`, so every ``2 pi`` cancels in the
    rate formulas: :math:`N_0 = 2\sqrt{m_2/m_0}` and
    :math:`M = \sqrt{m_4/m_2}`.
    """
    psd = power_spectral_density(x, fs, nperseg=nperseg)
    f = psd.frequencies
    m0 = float(np.trapezoid(psd.psd, f))
    m2 = float(np.trapezoid(psd.psd * f**2, f))
    m4 = float(np.trapezoid(psd.psd * f**4, f))
    return m0, m2, m4


def _count_level_crossings(
    x: NDArray[np.float64], levels: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Crossings of each level with both slopes, counted as sign changes."""
    signs = x[np.newaxis, :] > levels[:, np.newaxis]
    return np.asarray(
        np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1),
        dtype=np.int64,
    )


@dataclass(frozen=True)
class LevelCrossingResult:
    r"""Measured level-crossing rates against the Rice expectation.

    All rates count crossings with *both* slopes per unit time, following
    B&P Sec. 5.5.1; the rate of zero crossings is twice the record's
    apparent frequency. The Rice curve
    :math:`N_a = N_0 e^{-a^2 / (2\sigma^2)}` (Eq. (5.196)) holds for
    Gaussian records; systematic departures of the measured rates from
    it are themselves a useful non-Gaussianity screen (B&P Sec. 5.5.1.1).

    :ivar levels: Crossing levels ``a``, in signal units (about the
        removed record mean).
    :ivar rates: Measured crossing rates per level, in 1/s.
    :ivar rice_rates: Rice expectation :math:`N_0 e^{-a^2/(2\sigma^2)}`, 1/s.
    :ivar zero_crossing_rate: Measured zero-crossing rate ``N0``, in 1/s.
    :ivar zero_crossing_rate_rice: Expected
        :math:`N_0 = 2\sqrt{m_2/m_0}` from the record's Welch
        autospectrum moments (Eq. (5.195)), in 1/s.
    :ivar apparent_frequency: ``N0 / 2`` from the spectral moments, in Hz
        (a 60 Hz sine crosses zero 120 times per second).
    :ivar sigma: RMS value of the demeaned record, in signal units.
    :ivar duration: Record duration used for the rates, in seconds.
    :ivar fs: Sample rate, in Hz.
    """

    levels: NDArray[np.float64]
    rates: NDArray[np.float64]
    rice_rates: NDArray[np.float64]
    zero_crossing_rate: float
    zero_crossing_rate_rice: float
    apparent_frequency: float
    sigma: float
    duration: float
    fs: float

    def __post_init__(self) -> None:
        """Reject a result whose per-level quantities disagree.

        The comparison this result exists for is level by level: each measured
        rate belongs beside the Rice expectation at the same level, and the
        plot draws the expectation as a curve indexed by the order that sorts
        :attr:`levels`. An expectation array longer than the levels is
        therefore not caught anywhere: that order reaches only as many of its
        entries as there are levels, so the curve is drawn from part of one
        expectation while the markers beside it were measured against
        another, and the plot is well formed either way. One that is short
        raises about an index from inside the renderer instead, naming
        neither array.

        The scalar zero-crossing rates stand outside the group: they describe
        the single level zero, whatever levels were asked for.

        :raises ValueError: if the levels, the measured rates or the Rice
            expectations span different numbers of levels.
        """
        require_ranks(self, levels=1, rates=1, rice_rates=1)
        require_same_length(self, "levels", "rates", "rice_rates", axis="level")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot measured crossing rates against the Rice curve.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_level_crossing_rate

        check_language(language)
        return plot_level_crossing_rate(self, ax=ax, language=language, **kwargs)


def level_crossing_rate(
    x: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    levels: NDArray[np.float64] | list[float] | None = None,
    nperseg: int | None = None,
) -> LevelCrossingResult:
    r"""Level-crossing rates of a record and their Rice expectations.

    Counts the crossings of each level ``a`` (both slopes, B&P Sec. 5.5.1)
    and compares them with the closed forms for Gaussian data: the
    zero-crossing rate
    :math:`N_0 = (1/\pi)(\sigma_v/\sigma_x) = 2\sqrt{m_2/m_0}`
    (Eq. (5.195)) and the level dependence
    :math:`N_a = N_0 e^{-a^2/(2\sigma_x^2)}` (Eq. (5.196)), with the
    spectral moments taken from the record's own Welch autospectrum. For
    low-pass white noise of bandwidth ``B`` the expectation is
    :math:`N_0 = 2B/\sqrt{3}` -- an apparent frequency of ``0.58 B`` (B&P
    Example 5.12) -- and for bandwidth-limited white noise centred on
    ``fc``, :math:`N_0 = 2\sqrt{f_\mathrm{c}^2 + B^2/12}` (Example 5.13).

    The record mean is removed first (the formulas hold for zero mean
    value records); ``levels`` are then relative to that mean. Crossings
    are counted as sign changes of the sampled record, so the sample rate
    must comfortably oversample the signal bandwidth for the count not to
    miss crossings between samples.

    :param x: Signal, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, which means ``levels`` has to
        be given in the same units, so pascals rather than digital ones, and
        ``sigma`` comes out in Pa. The crossing rates themselves are
        scale-free.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param levels: Crossing levels in signal units about the mean
        (default: 13 levels evenly spaced over +-3 RMS).
    :param nperseg: Welch segment length for the spectral moments
        (default: the :func:`power_spectral_density` default).
    :return: A :class:`LevelCrossingResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x", context="level-crossing statistics")
    fs_v = _positive(resolve_fs(x, fs), "fs")
    xa = xa - float(np.mean(xa))
    sigma = float(np.sqrt(np.mean(xa**2)))
    if sigma <= 0.0:  # RMS is non-negative: <= 0 means a constant record
        msg = "'x' must not be constant."
        raise ValueError(msg)

    if levels is None:
        level_arr = np.linspace(-3.0, 3.0, 13) * sigma
    else:
        level_arr = np.asarray(levels, dtype=np.float64)
        if level_arr.ndim != 1 or level_arr.size == 0:
            msg = "'levels' must be a non-empty 1-D array."
            raise ValueError(msg)
        if not np.all(np.isfinite(level_arr)):
            msg = "'levels' must be finite."
            raise ValueError(msg)

    duration = (xa.size - 1) / fs_v
    counts = _count_level_crossings(xa, level_arr)
    zero_count = int(_count_level_crossings(xa, np.zeros(1))[0])

    m0, m2, _ = _spectral_moments(xa, fs_v, nperseg)
    n0_rice = 2.0 * float(np.sqrt(m2 / m0))
    rice = n0_rice * np.exp(-(level_arr**2) / (2.0 * m0))

    return LevelCrossingResult(
        levels=level_arr,
        rates=np.asarray(counts / duration, dtype=np.float64),
        rice_rates=np.asarray(rice, dtype=np.float64),
        zero_crossing_rate=zero_count / duration,
        zero_crossing_rate_rice=n0_rice,
        apparent_frequency=n0_rice / 2.0,
        sigma=sigma,
        duration=duration,
        fs=fs_v,
    )


def _rice_peak_exceedance(
    z: NDArray[np.float64], irregularity: float
) -> NDArray[np.float64]:
    r""":math:`P[\text{peak} > z]` for standardized peak height ``z``
    (B&P Eq. (5.223)).

    :math:`Q(z/\epsilon) + r\, e^{-z^2/2} \left[ 1 - Q(rz/\epsilon) \right]`
    with :math:`\epsilon = \sqrt{1 - r^2}`: the Rayleigh exceedance
    :math:`e^{-z^2/2}` for narrow bandwidth data (:math:`r \to 1`, B&P
    Eq. (5.206)) and the Gaussian exceedance for :math:`r \to 0`
    (Eq. (5.221)).
    """
    r = irregularity
    eps_sq = max(0.0, 1.0 - r * r)
    tail = np.exp(-(z**2) / 2.0)
    if eps_sq < _EPS_SQ_RAYLEIGH_LIMIT:  # narrow bandwidth limit: exactly Rayleigh
        return np.asarray(np.where(z > 0.0, tail, 1.0), dtype=np.float64)
    eps = float(np.sqrt(eps_sq))
    q_first = special.ndtr(-z / eps)
    phi_second = special.ndtr(r * z / eps)
    return np.asarray(q_first + r * tail * phi_second, dtype=np.float64)


def _rice_peak_density(
    z: NDArray[np.float64], irregularity: float
) -> NDArray[np.float64]:
    r"""Peak probability density ``w(z)`` (B&P Eq. (5.217)).

    :math:`\frac{\epsilon}{\sqrt{2\pi}} e^{-z^2/(2\epsilon^2)}
    + r z\, e^{-z^2/2} \left[ 1 - Q(rz/\epsilon) \right]`: the mixture
    between the standardized Gaussian (:math:`r = 0`, Eq. (5.221)) and Rayleigh
    (:math:`r = 1`, Eq. (5.222)) densities; minus the derivative of
    :func:`_rice_peak_exceedance`.
    """
    r = irregularity
    eps_sq = max(0.0, 1.0 - r * r)
    tail = np.exp(-(z**2) / 2.0)
    if eps_sq < _EPS_SQ_RAYLEIGH_LIMIT:  # narrow bandwidth limit: exactly Rayleigh
        return np.asarray(np.where(z > 0.0, z * tail, 0.0), dtype=np.float64)
    eps = float(np.sqrt(eps_sq))
    gaussian = eps / np.sqrt(2.0 * np.pi) * np.exp(-(z**2) / (2.0 * eps_sq))
    phi_second = special.ndtr(r * z / eps)
    return np.asarray(gaussian + r * z * tail * phi_second, dtype=np.float64)


def _check_measured_peak_rate(
    peak_rate: float, peaks: int, duration: float, owner: str
) -> None:
    """Require ``peak_rate`` to count the peaks stored beside it.

    The measured rate is a count over a record length, and both are in
    hand: the count is how many entries :attr:`PeakStatisticsResult.
    peak_values` holds -- every detected maximum is standardized and kept,
    none are dropped -- and the length is :attr:`~PeakStatisticsResult.
    duration`. The plot draws the empirical exceedance from that same count
    and floors the axis at ``1/n``, so a rate that disagrees with it puts a
    number in the caller's hands that the figure beside it contradicts,
    with neither one wrong on its face.

    The slack is relative as well as absolute here, unlike the decibel
    margins the other guards in the tree compare: a rate has no bounded
    scale, and a record sampled fast enough carries maxima in the millions
    per second, where a billionth of an absolute unit is under one bit.

    :param peak_rate: The stored measured rate, in 1/s.
    :param peaks: Number of detected maxima stored on the result.
    :param duration: Record length the rate was taken over, in seconds.
    :param owner: Name of the result type, used in the error message.
    :raises ValueError: if the duration is not a positive finite record
        length, or the rate is not the peak count over it.
    """
    if not math.isfinite(duration) or duration <= 0.0:
        msg = (
            f"{owner}: 'duration' must be a positive, finite record length "
            f"for 'peak_rate' to be a rate over it; got {duration!r}."
        )
        raise ValueError(msg)
    expected = peaks / duration
    if math.isclose(peak_rate, expected, rel_tol=1e-9, abs_tol=1e-9):
        return
    msg = (
        f"{owner}: 'peak_rate' must be the number of maxima in 'peak_values' "
        f"over 'duration'; got {peak_rate!r} where {peaks} peaks over "
        f"{duration!r} s state {expected!r}."
    )
    raise ValueError(msg)


def _check_rice_peak_rate(
    peak_rate_rice: float,
    zero_crossing_rate_rice: float,
    irregularity: float,
    owner: str,
) -> None:
    r"""Require ``irregularity_factor`` to restate the rates it came from.

    :math:`M = \sqrt{m_4/m_2}` is a moment of a Welch autospectrum the
    result does not store, so on its own there is nothing to measure it
    against. It is closed all the same, because the result carries the two
    scalars it was combined with: :func:`peak_statistics` writes
    :math:`r = \min(1, N_0 / 2M)` (B&P Eq. (5.220), capped where the
    Schwartz inequality is tight to the last bit), and that identity pins
    the three of them together. The cap is reproduced rather than divided
    away, so a narrow-band record whose ratio rounds past 1 still passes.

    This is the one the reader publishes. :meth:`PeakStatisticsResult.plot`
    labels its Rice curve with ``r`` and draws it from
    :meth:`~PeakStatisticsResult.peak_exceedance`, itself a function of
    ``r`` alone, so a triple that does not close paints a peak-height
    distribution under a legend the record's own rates contradict. It is
    the factor the message names: ``M`` and ``N0`` are read off the
    spectrum, and ``r`` is the one of the three the producer derives.

    Only a zero ``M`` is turned away before the ratio, and only because
    the division needs it. A non-finite ``M`` is not: an amplitude large
    enough to overflow the spectral moments has :func:`peak_statistics`
    emitting :math:`M = \infty` beside :math:`r = 0` (and, further up,
    ``NaN`` beside 1), and the identity closes over both -- capped, they
    are exactly what ``min(1, N_0/2M)`` states. Refusing them would refuse
    the producer's own output over an amplitude, which is not this guard's
    business.

    :param peak_rate_rice: Expected rate of maxima ``M``, in 1/s.
    :param zero_crossing_rate_rice: Expected zero-crossing rate ``N0``, 1/s.
    :param irregularity: The stored irregularity factor ``r``.
    :param owner: Name of the result type, used in the error message.
    :raises ValueError: if the expected maxima rate is zero or negative, or
        the three rates do not state one irregularity factor.
    """
    if peak_rate_rice <= 0.0:
        msg = (
            f"{owner}: 'peak_rate_rice' must be a positive expected rate of "
            f"maxima to state a ratio; got {peak_rate_rice!r}."
        )
        raise ValueError(msg)
    stated = min(1.0, zero_crossing_rate_rice / (2.0 * peak_rate_rice))
    if math.isclose(irregularity, stated, rel_tol=0.0, abs_tol=1e-9):
        return
    msg = (
        f"{owner}: 'irregularity_factor' must be the r that 'peak_rate_rice' "
        "and 'zero_crossing_rate_rice' state, r = min(1, N0 / 2M) "
        f"(B&P Eq. (5.220)); got r {irregularity!r} where M "
        f"{peak_rate_rice!r} beside N0 {zero_crossing_rate_rice!r} state "
        f"{stated!r}."
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class PeakStatisticsResult:
    r"""Peak (maxima) statistics of a record against the Rice expectations.

    "Positive peaks" are the record's local maxima, positive or negative
    in value (B&P Sec. 5.5.3). For Gaussian records the expected rate of
    maxima is :math:`M = (1/2\pi)(\sigma_a/\sigma_v) = \sqrt{m_4/m_2}`
    (Eq. (5.211)), and the ratio :math:`r = N_0/(2M)` -- the
    **irregularity factor**, between 0 and 1 by the Schwartz inequality
    (Eq. (5.220)) -- fixes the standardized peak-height distribution:
    Rayleigh at :math:`r = 1` (every cycle carries one maximum), Gaussian
    as :math:`r \to 0` (Sec. 5.5.4).

    :ivar peak_rate: Measured rate of local maxima, in 1/s.
    :ivar peak_rate_rice: Expected rate :math:`M = \sqrt{m_4/m_2}` from
        the Welch autospectrum moments, in 1/s.
    :ivar zero_crossing_rate_rice: Expected
        :math:`N_0 = 2\sqrt{m_2/m_0}`, 1/s.
    :ivar irregularity_factor: ``N0 / (2 M)`` from the spectral moments.
    :ivar peak_values: Standardized heights
        :math:`z = \text{peak} / \sigma` of the detected maxima.
    :ivar sigma: RMS value of the demeaned record, in signal units.
    :ivar duration: Record duration used for the rates, in seconds.
    :ivar fs: Sample rate, in Hz.
    """

    peak_rate: float
    peak_rate_rice: float
    zero_crossing_rate_rice: float
    irregularity_factor: float
    peak_values: NDArray[np.float64]
    sigma: float
    duration: float
    fs: float

    def __post_init__(self) -> None:
        """Reject standardized peak heights that are not finite and sorted.

        The plot pairs :attr:`peak_values` positionally with the empirical
        exceedance ``1 - rank/n``, and spans its Rice curves from the first
        entry to the last: both readings are statements about the sorted
        order :func:`peak_statistics` writes, not about the array as such.
        An unsorted array is accepted by every count -- nothing changes
        length -- and publishes a wrong exceedance under the ordinary title,
        so the order is pinned here, where the mistake can be named.

        Finiteness is pinned first, and not out of tidiness: the order check
        is a comparison, and every comparison against a ``NaN`` is false, so
        a ``NaN`` is not merely admitted -- it also hides the descent on
        either side of it, which is the one thing this guard exists to catch.
        Two equal infinities differ by a ``NaN`` and read as sorted for the
        same reason. Either one then reaches the curves' span, a ``linspace``
        between the first entry and the last, and the empirical exceedance is
        published against heights that are not heights.
        :func:`peak_statistics` cannot emit them -- the record is validated
        finite and ``sigma`` is positive -- so nothing measurable is refused.

        The rates beside the heights are pinned too, each against what the
        result already holds: :attr:`peak_rate` against the peaks it counts
        over :attr:`duration`, and :attr:`irregularity_factor` against the
        two expected rates it was derived from. See
        :func:`_check_measured_peak_rate` and :func:`_check_rice_peak_rate`
        for why each one closes.

        :raises ValueError: if the peak heights carry more than one axis,
            are not finite throughout, or are not in ascending order; if the
            measured rate is not the stored peaks over the stored duration;
            or if the expected rates and the irregularity factor do not
            state one another.
        """
        require_ranks(self, peak_values=1)
        require_finite_fields(self, "peak_values")
        values = np.asarray(self.peak_values)
        if values.ndim == 1 and np.any(np.diff(values) < 0.0):
            msg = (
                "PeakStatisticsResult: 'peak_values' must be sorted "
                "ascending; the empirical exceedance pairs each peak with "
                "its rank."
            )
            raise ValueError(msg)
        owner = type(self).__name__
        _check_measured_peak_rate(self.peak_rate, values.size, self.duration, owner)
        _check_rice_peak_rate(
            self.peak_rate_rice,
            self.zero_crossing_rate_rice,
            self.irregularity_factor,
            owner,
        )

    def peak_exceedance(
        self, z: NDArray[np.float64] | list[float] | float
    ) -> NDArray[np.float64]:
        r""":math:`P[\text{peak} > z]` at the record's irregularity
        factor.

        B&P Eq. (5.223): the probability that a maximum chosen at random
        exceeds ``z`` record RMS units. :math:`e^{-z^2/2}` for narrow
        bandwidth data -- :math:`e^{-8} = 0.00033` at :math:`z = 4`, B&P
        Example 5.14 -- and the Gaussian exceedance in the wide
        bandwidth limit.

        :param z: Standardized peak heights (units of the record RMS).
        :return: Exceedance probabilities, same shape.
        """
        za = np.atleast_1d(np.asarray(z, dtype=np.float64))
        return _rice_peak_exceedance(za, self.irregularity_factor)

    def peak_density(
        self, z: NDArray[np.float64] | list[float] | float
    ) -> NDArray[np.float64]:
        """Peak probability density ``w(z)`` (B&P Eq. (5.217)).

        :param z: Standardized peak heights (units of the record RMS).
        :return: Probability density values, same shape.
        """
        za = np.atleast_1d(np.asarray(z, dtype=np.float64))
        return _rice_peak_density(za, self.irregularity_factor)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the empirical peak exceedance against the Rice curves.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_peak_statistics

        check_language(language)
        return plot_peak_statistics(self, ax=ax, language=language, **kwargs)


def peak_statistics(
    x: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    nperseg: int | None = None,
) -> PeakStatisticsResult:
    r"""Peak rate, irregularity factor and peak-height distribution of a record.

    Detects the record's local maxima and compares their rate and height
    distribution with the Rice closed forms for Gaussian data (B&P
    Secs. 5.5.2 to 5.5.4): expected maxima rate :math:`M = \sqrt{m_4/m_2}`
    (Eq. (5.211)), irregularity factor :math:`r = N_0/(2M)` (Eq. (5.220))
    and the standardized peak-height distribution that interpolates
    between Rayleigh (narrow bandwidth, :math:`r = 1`) and Gaussian
    (wide bandwidth, :math:`r \to 0`) via
    :meth:`PeakStatisticsResult.peak_exceedance`. The
    irregularity factor is the bridge to fatigue and vibro-acoustic
    damage models, where it selects the cycle-counting correction.

    The record mean is removed first. The ``m4`` moment weights the
    autospectrum by :math:`f^4`, so broadband instrumentation noise far above
    the physical band inflates ``M`` and deflates ``r``: band-limit the
    record to the physically meaningful band first.

    :param x: Signal, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so ``sigma`` comes out in Pa.
        The peak heights are standardized by it, so they and the rates do not
        move.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param nperseg: Welch segment length for the spectral moments
        (default: the :func:`power_spectral_density` default).
    :return: A :class:`PeakStatisticsResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x", context="peak statistics")
    fs_v = _positive(resolve_fs(x, fs), "fs")
    xa = xa - float(np.mean(xa))
    sigma = float(np.sqrt(np.mean(xa**2)))
    if sigma <= 0.0:  # RMS is non-negative: <= 0 means a constant record
        msg = "'x' must not be constant."
        raise ValueError(msg)

    interior = xa[1:-1]
    maxima = (interior > xa[:-2]) & (interior > xa[2:])
    peaks = interior[maxima]
    duration = (xa.size - 1) / fs_v

    m0, m2, m4 = _spectral_moments(xa, fs_v, nperseg)
    n0_rice = 2.0 * float(np.sqrt(m2 / m0))
    m_rice = float(np.sqrt(m4 / m2))
    return PeakStatisticsResult(
        peak_rate=peaks.size / duration,
        peak_rate_rice=m_rice,
        zero_crossing_rate_rice=n0_rice,
        irregularity_factor=min(1.0, n0_rice / (2.0 * m_rice)),
        peak_values=np.asarray(np.sort(peaks) / sigma, dtype=np.float64),
        sigma=sigma,
        duration=duration,
        fs=fs_v,
    )
