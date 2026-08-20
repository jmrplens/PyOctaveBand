#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Measurement uncertainty by the GUM and its Monte Carlo supplement.

Implements the two propagation methods of the *Guide to the Expression of
Uncertainty in Measurement*:

* the **law of propagation of uncertainty** (ISO/IEC Guide 98-3:2008, clause 5)
  - the combined standard uncertainty of a measurement model
  :math:`y = f(x_1, \ldots, x_N)`
  from the input standard uncertainties and sensitivity coefficients, with
  optional input correlations, the effective degrees of freedom
  (Welch-Satterthwaite, Annex G.4) and the expanded uncertainty
  :math:`U = k u_\mathrm{c}` with a coverage factor from the t-distribution
  (clause 6);
* the **Monte Carlo method** (ISO/IEC Guide 98-3-1:2008, Supplement 1) - the
  numerical propagation of the input probability density functions, giving the
  estimate, its standard uncertainty and a probabilistically symmetric coverage
  interval (clause 7.7).

Input quantities are described by :class:`Quantity`; :func:`rectangular`,
:func:`triangular` and :func:`u_shaped` build Type B quantities from a
half-width (clause 4.3).

Scope notes: :func:`monte_carlo` runs a **fixed** number of trials and reports
the probabilistically symmetric interval only -- the adaptive procedure of
Supplement 1 clause 7.9 and the shortest coverage interval of 5.3.4 are not
implemented; its inputs are sampled independently (the multivariate-Gaussian
path of 6.4.8 for non-independent quantities is not implemented -- use
:func:`combine_uncertainty` with ``correlation`` for correlated budgets).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike


from .._internal.warnings import PhonometryWarning

Model = Callable[..., float]


class UncertaintyWarning(PhonometryWarning):
    """A GUM propagation fell back outside its nominal assumptions."""


#: Recognised probability density functions for the Monte Carlo method.
DISTRIBUTIONS: tuple[str, ...] = ("gaussian", "rectangular", "triangular", "u-shaped")


@dataclass(frozen=True)
class Quantity:
    r"""An input quantity of a measurement model (GUM clause 4).

    :ivar value: Best estimate :math:`x_i` of the input quantity.
    :ivar uncertainty: Standard uncertainty :math:`u(x_i)` (>= 0).
    :ivar distribution: PDF used by the Monte Carlo method: ``"gaussian"``,
        ``"rectangular"``, ``"triangular"`` or ``"u-shaped"``.
    :ivar dof: Degrees of freedom of ``uncertainty`` (``inf`` for Type B).
    :ivar name: Optional label used in the uncertainty budget and its plot.
    """

    value: float
    uncertainty: float
    distribution: str = "gaussian"
    dof: float = math.inf
    name: str = ""

    def __post_init__(self) -> None:
        if self.uncertainty < 0.0:
            raise ValueError("uncertainty must be non-negative.")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {DISTRIBUTIONS}; "
                f"got {self.distribution!r}."
            )
        if self.dof <= 0.0:
            raise ValueError("dof must be positive.")


def rectangular(value: float, half_width: float, name: str = "") -> Quantity:
    r"""Type B quantity with a rectangular PDF of half-width ``a`` (GUM 4.3.7).

    The standard uncertainty is :math:`a / \sqrt{3}`.
    """
    return Quantity(value, half_width / math.sqrt(3.0), "rectangular", name=name)


def triangular(value: float, half_width: float, name: str = "") -> Quantity:
    r"""Type B quantity with a triangular PDF of half-width ``a`` (GUM 4.3.9).

    The standard uncertainty is :math:`a / \sqrt{6}`.
    """
    return Quantity(value, half_width / math.sqrt(6.0), "triangular", name=name)


def u_shaped(value: float, half_width: float, name: str = "") -> Quantity:
    r"""Type B quantity with a U-shaped (arcsine) PDF of half-width ``a``.

    The standard uncertainty is :math:`a / \sqrt{2}`.
    """
    return Quantity(value, half_width / math.sqrt(2.0), "u-shaped", name=name)


@dataclass(frozen=True)
class UncertaintyResult:
    r"""Result of the GUM law of propagation of uncertainty (Guide 98-3).

    :ivar value: The output estimate :math:`y = f(x_1, \ldots, x_N)`.
    :ivar combined_uncertainty: Combined standard uncertainty
        :math:`u_\mathrm{c}(y)`.
    :ivar sensitivities: Sensitivity coefficients
        :math:`c_i = \partial f/\partial x_i`.
    :ivar contributions: Per-input contributions
        :math:`\lvert c_i \rvert u(x_i)` to :math:`u_\mathrm{c}(y)`.
    :ivar effective_dof: Welch-Satterthwaite effective degrees of freedom
        (Annex G.4, defined for independent inputs). For a correlated budget
        with finite input dof it is ``NaN`` (undefined: the GUM has no
        correlated form and ``expanded()`` then needs an explicit factor);
        with all-infinite input dof it is ``inf`` (normal-distribution coverage
        factor), since the GUM defines no correlated Welch-Satterthwaite form.
    :ivar names: Input labels aligned with the arrays above.
    """

    value: float
    combined_uncertainty: float
    sensitivities: np.ndarray
    contributions: np.ndarray
    effective_dof: float
    names: tuple[str, ...] = field(default=())

    def expanded(
        self, coverage: float = 0.95, *, coverage_factor_override: float | None = None
    ) -> tuple[float, float]:
        r"""Coverage factor ``k`` and expanded uncertainty :math:`U = k u_\mathrm{c}`.

        :param coverage: Coverage probability in (0, 1); ``0.95`` by default.
        :param coverage_factor_override: Explicit ``k``. Required for a
            correlated budget with finite input degrees of freedom, where the
            GUM defines no effective-dof formula (``effective_dof`` is NaN).
        :return: The pair ``(k, U)`` (GUM clause 6, Annex G).
        :raises ValueError: If the effective dof are undefined and no
            explicit coverage factor is given.
        """
        if coverage_factor_override is not None:
            k = float(coverage_factor_override)
            return k, k * self.combined_uncertainty
        if math.isnan(self.effective_dof):
            raise ValueError(
                "The effective degrees of freedom are undefined for a "
                "correlated budget with finite input dof (the GUM defines "
                "no Welch-Satterthwaite form there): pass an explicit "
                "coverage_factor_override."
            )
        k = coverage_factor(coverage, self.effective_dof)
        return k, k * self.combined_uncertainty

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the uncertainty budget (per-input contributions).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_uncertainty_budget

        check_language(language)
        return plot_uncertainty_budget(self, ax=ax, language=language, **kwargs)


@dataclass(frozen=True)
class MonteCarloResult:
    r"""Result of the Monte Carlo method (Guide 98-3-1, Supplement 1).

    :ivar value: Estimate ``y`` (the sample mean of the output).
    :ivar standard_uncertainty: :math:`u(y)` (the sample standard
        deviation).
    :ivar interval: Probabilistically symmetric coverage interval
        ``(low, high)`` (clause 7.7).
    :ivar coverage: The coverage probability of ``interval``.
    :ivar trials: Number of Monte Carlo trials.
    :ivar samples: The raw model-output sample (one value per trial), kept
        only when :func:`monte_carlo` is called with ``keep_samples=True``
        (it feeds the output-distribution histogram of :meth:`plot`).
    """

    value: float
    standard_uncertainty: float
    interval: tuple[float, float]
    coverage: float
    trials: int
    samples: np.ndarray | None = field(default=None, repr=False)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the output histogram with the coverage interval marked.

        Needs the raw output sample, so call ``monte_carlo(...,
        keep_samples=True)``. Requires matplotlib (``pip install
        phonometry[plot]``); returns the :class:`~matplotlib.axes.Axes`.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_monte_carlo

        check_language(language)
        return plot_monte_carlo(self, ax=ax, language=language, **kwargs)


def _sensitivity(
    model: Model, values: np.ndarray, uncertainties: np.ndarray
) -> np.ndarray:
    r"""Central-difference sensitivities :math:`\partial f/\partial x_i`.

    Follows GUM 5.1.3. The step is
    :math:`\max(u(x_i), \sqrt{\epsilon} \lvert x_i \rvert)` -- GUM 5.1.4
    NOTE 2 itself suggests :math:`dx_i = u(x_i)` -- with a floating-point
    guard of a few ULP of
    :math:`x_i` and an absolute floor for all-zero inputs, so a tiny
    uncertainty on a large value can no longer underflow the perturbation
    and silently zero the sensitivity (e.g. :math:`x_i = 10^9`,
    :math:`u = 10^{-6}`).
    """
    import warnings

    n = values.size
    coeffs = np.empty(n)
    sqrt_eps = math.sqrt(float(np.finfo(np.float64).eps))
    for i in range(n):
        # GUM 5.1.4 NOTE 2 suggests the step u(xi); a 64-ULP floor keeps the
        # perturbation representable for large-magnitude inputs without
        # abandoning locality (a sqrt(eps)*|xi| floor would probe the model
        # far outside the uncertainty region for large xi).
        step = float(uncertainties[i])
        if step <= 0.0:
            step = sqrt_eps
        step = max(step, 64.0 * float(np.spacing(abs(values[i]))))
        up = values.copy()
        down = values.copy()
        up[i] += step
        down[i] -= step
        f_up = float(model(*up))
        f_down = float(model(*down))
        if f_up == f_down and uncertainties[i] > 0.0:
            warnings.warn(
                f"The model output does not change when input {i + 1} is "
                f"perturbed by its evaluation step ({step:.3g}): its "
                "sensitivity evaluates to exactly zero and its uncertainty "
                "does not propagate. This is legitimate for genuinely flat "
                "directions (e.g. GUM H.1) but worth verifying.",
                UncertaintyWarning,
                stacklevel=3,
            )
        coeffs[i] = (f_up - f_down) / (2.0 * step)
    return coeffs


def _validated_correlation(correlation: ArrayLike | None, n: int) -> np.ndarray | None:
    """Check a caller-supplied correlation matrix of an ``N``-input budget.

    :param correlation: The ``N x N`` correlation matrix, or ``None``.
    :param n: Number of input quantities.
    :return: The matrix as a float array, or ``None`` when none was given.
    :raises ValueError: If the matrix is not square, symmetric, unit-diagonal
        and positive semi-definite.
    """
    if correlation is None:
        return None
    r = np.asarray(correlation, dtype=np.float64)
    if r.shape != (n, n):
        raise ValueError(f"correlation must have shape ({n}, {n}); got {r.shape}.")
    if not np.allclose(r, r.T):
        raise ValueError("correlation matrix must be symmetric.")
    if not np.allclose(np.diag(r), 1.0):
        raise ValueError("correlation matrix diagonal must be 1.0.")
    # A symmetric, unit-diagonal matrix can still be indefinite, which would
    # make the variance negative and be silently masked by the clamp that
    # follows the propagation; reject it.
    if float(np.min(np.linalg.eigvalsh(r))) < -1e-8:
        raise ValueError("correlation matrix must be positive semi-definite.")
    return r


def _effective_dof(
    dofs: np.ndarray,
    contributions: np.ndarray,
    combined: float,
    *,
    correlated: bool,
) -> float:
    """Welch-Satterthwaite effective degrees of freedom (Annex G.4).

    Formula (G.2b) is derived for independent input quantities only and the GUM
    defines no correlated form: a correlated budget with finite input dof
    therefore carries NO effective dof (NaN), and expanded() requires an
    explicit coverage factor from the caller. With all input dof infinite
    the output is treated as normal and veff stays infinite.
    """
    finite = np.isfinite(dofs)
    if correlated and np.any(finite):
        import warnings

        warnings.warn(
            "Welch-Satterthwaite (GUM G.4.1) is defined for independent "
            "inputs only and the GUM defines no correlated form: the "
            "effective degrees of freedom are undefined (NaN) for this "
            "budget, and expanded() requires an explicit coverage_factor.",
            UncertaintyWarning,
            stacklevel=3,
        )
        return math.nan
    if correlated:
        return math.inf
    if combined > 0.0 and np.any(finite & (contributions > 0.0)):
        terms = np.where(finite, contributions**4 / np.where(finite, dofs, 1.0), 0.0)
        denom = float(np.sum(terms))
        return combined**4 / denom if denom > 0.0 else math.inf
    return math.inf


def combine_uncertainty(
    model: Model,
    quantities: Sequence[Quantity],
    correlation: ArrayLike | None = None,
) -> UncertaintyResult:
    r"""Combined standard uncertainty by the GUM law of propagation (clause 5).

    :param model: The measurement function
        :math:`f(x_1, \ldots, x_N)` returning ``y``.
    :param quantities: The input :class:`Quantity` objects, in the order the
        model takes its arguments.
    :param correlation: Optional ``N x N`` correlation matrix
        :math:`r_{ij}` between
        the inputs; ``None`` treats them as uncorrelated. With a non-identity
        matrix and finite input dof the effective degrees of freedom are
        ``NaN`` (undefined; the GUM defines no correlated
        fallback -- Welch-Satterthwaite holds for independent inputs only)
        and an :class:`UncertaintyWarning` is issued when finite input dof
        would otherwise have been propagated.
    :return: An :class:`UncertaintyResult` with :math:`u_\mathrm{c}(y)`, the
        sensitivity
        coefficients, the contributions and the effective degrees of freedom.
    :raises ValueError: for no inputs or a malformed correlation matrix.
    """
    if len(quantities) == 0:
        raise ValueError("at least one input quantity is required.")
    values = np.array([q.value for q in quantities], dtype=np.float64)
    uncert = np.array([q.uncertainty for q in quantities], dtype=np.float64)
    n = values.size

    r = _validated_correlation(correlation, n)

    coeffs = _sensitivity(model, values, uncert)
    contributions = np.abs(coeffs) * uncert  # ui(y) = |ci| u(xi)

    correlated = r is not None and not np.allclose(r, np.eye(n))
    if r is None or not correlated:
        variance = float(np.sum(contributions**2))
    else:
        signed = coeffs * uncert
        variance = float(signed @ r @ signed)
    combined = math.sqrt(max(variance, 0.0))

    dofs = np.array([q.dof for q in quantities], dtype=np.float64)
    effective_dof = _effective_dof(dofs, contributions, combined, correlated=correlated)

    return UncertaintyResult(
        value=float(model(*values)),
        combined_uncertainty=combined,
        sensitivities=coeffs,
        contributions=contributions,
        effective_dof=effective_dof,
        names=tuple(q.name or f"x{i + 1}" for i, q in enumerate(quantities)),
    )


def coverage_factor(coverage: float = 0.95, dof: float = math.inf) -> float:
    r"""Coverage factor ``k`` from the t-distribution (GUM clause 6, Annex G).

    :param coverage: Coverage probability in (0, 1).
    :param dof: Effective degrees of freedom; ``inf`` gives the normal quantile.
    :return: The two-sided coverage factor
        :math:`k = t_p(\text{dof})`.
    :raises ValueError: for a coverage outside (0, 1).
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1); got {coverage}.")
    p = 0.5 * (1.0 + coverage)
    if math.isinf(dof):
        from scipy.special import ndtri

        return float(ndtri(p))
    from scipy.stats import t

    return float(t.ppf(p, dof))


def expanded_uncertainty(
    result: UncertaintyResult, coverage: float = 0.95
) -> tuple[float, float]:
    """Coverage factor and expanded uncertainty of a GUM result (clause 6).

    Convenience wrapper for :meth:`UncertaintyResult.expanded`.
    """
    return result.expanded(coverage)


def _sample(q: Quantity, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` samples of a quantity from its PDF (Supplement 1, 6.4).

    A constant (or numerically negligible) uncertainty collapses every PDF to a
    spike at ``mu``; ``rng.uniform`` and ``rng.normal`` already return ``mu`` in
    that case, but ``rng.triangular`` rejects a zero-width support, so it is
    guarded explicitly (including the case where ``mu - a`` and ``mu + a`` round
    to the same float for a tiny but non-zero ``u``).
    """
    mu, u = q.value, q.uncertainty
    if q.distribution == "gaussian":
        return rng.normal(mu, u, size)
    if q.distribution == "rectangular":
        a = u * math.sqrt(3.0)
        return rng.uniform(mu - a, mu + a, size)
    if q.distribution == "triangular":
        a = u * math.sqrt(6.0)
        left, right = mu - a, mu + a
        if left >= right:  # zero-width support (u == 0 or underflow)
            return np.full(size, mu)
        return rng.triangular(left, mu, right, size)
    # U-shaped (arcsine): a * cos(theta), theta uniform on [0, pi).
    a = u * math.sqrt(2.0)
    return mu + a * np.cos(rng.uniform(0.0, math.pi, size))


def monte_carlo(
    model: Model,
    quantities: Sequence[Quantity],
    trials: int = 1_000_000,
    coverage: float = 0.95,
    seed: int | None = None,
    keep_samples: bool = False,
) -> MonteCarloResult:
    r"""Propagate uncertainty by the Monte Carlo method (Supplement 1).

    Draws ``trials`` samples of each input from its PDF, evaluates the model and
    reports the sample mean, the sample standard deviation and the
    probabilistically symmetric coverage interval (clause 7.7).

    The inputs are sampled **independently**: the Supplement's
    multivariate-Gaussian path for non-independent quantities (6.4.8) is not
    implemented (use :func:`combine_uncertainty` with ``correlation`` for a
    correlated budget). The number of trials is fixed (the adaptive procedure
    of clause 7.9 is not implemented) and the reported interval is the
    probabilistically symmetric one (not the 5.3.4 shortest interval).

    :param model: The measurement function
        :math:`f(x_1, \ldots, x_N)` returning ``y``;
        it must accept array arguments (vectorised over the trials).
    :param quantities: The input :class:`Quantity` objects, in argument order.
    :param trials: Number of Monte Carlo trials ``M`` (at least 2; the sample
        standard deviation needs two values).
    :param coverage: Coverage probability of the reported interval.
    :param seed: Optional seed for the random generator (reproducibility).
    :param keep_samples: Retain the raw output sample on the result (one
        float per trial) so :meth:`MonteCarloResult.plot` can draw the
        output-distribution histogram.
    :return: A :class:`MonteCarloResult`.
    :raises ValueError: for no inputs, fewer than 2 trials or bad coverage.
    """
    if len(quantities) == 0:
        raise ValueError("at least one input quantity is required.")
    if trials < 2:
        raise ValueError("trials must be at least 2.")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1); got {coverage}.")

    rng = np.random.default_rng(seed)
    samples = [_sample(q, trials, rng) for q in quantities]
    output = np.asarray(model(*samples), dtype=np.float64)

    low_q = 0.5 * (1.0 - coverage)
    high_q = 0.5 * (1.0 + coverage)
    low, high = np.quantile(output, [low_q, high_q])
    return MonteCarloResult(
        value=float(np.mean(output)),
        standard_uncertainty=float(np.std(output, ddof=1)),
        interval=(float(low), float(high)),
        coverage=coverage,
        trials=trials,
        samples=output if keep_samples else None,
    )
