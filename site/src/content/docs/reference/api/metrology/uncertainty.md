---
title: "metrology.uncertainty"
description: "Measurement uncertainty by the GUM and its Monte Carlo supplement."
sidebar:
  label: "uncertainty"
---

Measurement uncertainty by the GUM and its Monte Carlo supplement.

Implements the two propagation methods of the *Guide to the Expression of
Uncertainty in Measurement*:

* the **law of propagation of uncertainty** (ISO/IEC Guide 98-3:2008, clause 5)
  - the combined standard uncertainty of a measurement model `y = f(x1..xN)`
  from the input standard uncertainties and sensitivity coefficients, with
  optional input correlations, the effective degrees of freedom
  (Welch-Satterthwaite, Annex G.4) and the expanded uncertainty
  `U = k * uc` with a coverage factor from the t-distribution (clause 6);
* the **Monte Carlo method** (ISO/IEC Guide 98-3-1:2008, Supplement 1) - the
  numerical propagation of the input probability density functions, giving the
  estimate, its standard uncertainty and a probabilistically symmetric coverage
  interval (clause 7.7).

Input quantities are described by [`Quantity`](/phonometry/reference/api/metrology/uncertainty/#quantity); [`rectangular`](/phonometry/reference/api/metrology/uncertainty/#rectangular),
[`triangular`](/phonometry/reference/api/metrology/uncertainty/#triangular) and [`u_shaped`](/phonometry/reference/api/metrology/uncertainty/#u_shaped) build Type B quantities from a
half-width (clause 4.3).

Scope notes: [`monte_carlo`](/phonometry/reference/api/metrology/uncertainty/#monte_carlo) runs a **fixed** number of trials and reports
the probabilistically symmetric interval only -- the adaptive procedure of
Supplement 1 clause 7.9 and the shortest coverage interval of 5.3.4 are not
implemented; its inputs are sampled independently (the multivariate-Gaussian
path of 6.4.8 for non-independent quantities is not implemented -- use
[`combine_uncertainty`](/phonometry/reference/api/metrology/uncertainty/#combine_uncertainty) with `correlation` for correlated budgets).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## combine_uncertainty

```python
combine_uncertainty(
    model: Model,
    quantities: Sequence[Quantity],
    correlation: ArrayLike | None = None,
) -> UncertaintyResult
```

Combined standard uncertainty by the GUM law of propagation (clause 5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `model` | The measurement function `f(x1, ..., xN)` returning `y`. |
| `quantities` | The input [`Quantity`](/phonometry/reference/api/metrology/uncertainty/#quantity) objects, in the order the model takes its arguments. |
| `correlation` | Optional `N x N` correlation matrix `r_ij` between the inputs; `None` treats them as uncorrelated. With a non-identity matrix and finite input dof the effective degrees of freedom are `NaN` (undefined; the GUM defines no correlated fallback -- Welch-Satterthwaite holds for independent inputs only) and an [`UncertaintyWarning`](/phonometry/reference/api/metrology/uncertainty/#uncertaintywarning) is issued when finite input dof would otherwise have been propagated. |

**Returns:** An [`UncertaintyResult`](/phonometry/reference/api/metrology/uncertainty/#uncertaintyresult) with `uc(y)`, the sensitivity coefficients, the contributions and the effective degrees of freedom.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for no inputs or a malformed correlation matrix. |

## monte_carlo

```python
monte_carlo(
    model: Model,
    quantities: Sequence[Quantity],
    trials: int = 1000000,
    coverage: float = 0.95,
    seed: int | None = None,
    keep_samples: bool = False,
) -> MonteCarloResult
```

Propagate uncertainty by the Monte Carlo method (Supplement 1).

Draws `trials` samples of each input from its PDF, evaluates the model and
reports the sample mean, the sample standard deviation and the
probabilistically symmetric coverage interval (clause 7.7).

The inputs are sampled **independently**: the Supplement's
multivariate-Gaussian path for non-independent quantities (6.4.8) is not
implemented (use [`combine_uncertainty`](/phonometry/reference/api/metrology/uncertainty/#combine_uncertainty) with `correlation` for a
correlated budget). The number of trials is fixed (the adaptive procedure
of clause 7.9 is not implemented) and the reported interval is the
probabilistically symmetric one (not the 5.3.4 shortest interval).

**Parameters**

| Name | Description |
| :--- | :--- |
| `model` | The measurement function `f(x1, ..., xN)` returning `y`; it must accept array arguments (vectorised over the trials). |
| `quantities` | The input [`Quantity`](/phonometry/reference/api/metrology/uncertainty/#quantity) objects, in argument order. |
| `trials` | Number of Monte Carlo trials `M` (at least 2; the sample standard deviation needs two values). |
| `coverage` | Coverage probability of the reported interval. |
| `seed` | Optional seed for the random generator (reproducibility). |
| `keep_samples` | Retain the raw output sample on the result (one float per trial) so [`MonteCarloResult.plot`](/phonometry/reference/api/metrology/uncertainty/#montecarloresultplot) can draw the output-distribution histogram. |

**Returns:** A [`MonteCarloResult`](/phonometry/reference/api/metrology/uncertainty/#montecarloresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for no inputs, fewer than 2 trials or bad coverage. |

## MonteCarloResult

```python
MonteCarloResult(
    value: float,
    standard_uncertainty: float,
    interval: tuple[float, float],
    coverage: float,
    trials: int,
    samples: np.ndarray | None = None,
)
```

Result of the Monte Carlo method (Guide 98-3-1, Supplement 1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `value` | Estimate `y` (the sample mean of the output). |
| `standard_uncertainty` | `u(y)` (the sample standard deviation). |
| `interval` | Probabilistically symmetric coverage interval `(low, high)` (clause 7.7). |
| `coverage` | The coverage probability of `interval`. |
| `trials` | Number of Monte Carlo trials. |
| `samples` | The raw model-output sample (one value per trial), kept only when [`monte_carlo`](/phonometry/reference/api/metrology/uncertainty/#monte_carlo) is called with `keep_samples=True` (it feeds the output-distribution histogram of `plot`). |

### MonteCarloResult.plot()

```python
MonteCarloResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the output histogram with the coverage interval marked.

Needs the raw output sample, so call `monte_carlo(...,
keep_samples=True)`. Requires matplotlib (`pip install
phonometry[plot]`); returns the `Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## Quantity

```python
Quantity(
    value: float,
    uncertainty: float,
    distribution: str = 'gaussian',
    dof: float = inf,
    name: str = '',
)
```

An input quantity of a measurement model (GUM clause 4).

**Attributes**

| Name | Description |
| :--- | :--- |
| `value` | Best estimate `xi` of the input quantity. |
| `uncertainty` | Standard uncertainty `u(xi)` (>= 0). |
| `distribution` | PDF used by the Monte Carlo method: `"gaussian"`, `"rectangular"`, `"triangular"` or `"u-shaped"`. |
| `dof` | Degrees of freedom of `uncertainty` (`inf` for Type B). |
| `name` | Optional label used in the uncertainty budget and its plot. |

## rectangular

```python
rectangular(value: float, half_width: float, name: str = '') -> Quantity
```

Type B quantity with a rectangular PDF of half-width `a` (GUM 4.3.7).

The standard uncertainty is `a / sqrt(3)`.

## triangular

```python
triangular(value: float, half_width: float, name: str = '') -> Quantity
```

Type B quantity with a triangular PDF of half-width `a` (GUM 4.3.9).

The standard uncertainty is `a / sqrt(6)`.

## u_shaped

```python
u_shaped(value: float, half_width: float, name: str = '') -> Quantity
```

Type B quantity with a U-shaped (arcsine) PDF of half-width `a`.

The standard uncertainty is `a / sqrt(2)`.

## UncertaintyResult

```python
UncertaintyResult(
    value: float,
    combined_uncertainty: float,
    sensitivities: np.ndarray,
    contributions: np.ndarray,
    effective_dof: float,
    names: tuple[str, ...] = (),
)
```

Result of the GUM law of propagation of uncertainty (Guide 98-3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `value` | The output estimate `y = f(x1..xN)`. |
| `combined_uncertainty` | Combined standard uncertainty `uc(y)`. |
| `sensitivities` | Sensitivity coefficients `ci = df/dxi`. |
| `contributions` | Per-input contributions `\|ci\| u(xi)` to `uc(y)`. |
| `effective_dof` | Welch-Satterthwaite effective degrees of freedom (Annex G.4, defined for independent inputs). For a correlated budget with finite input dof it is `NaN` (undefined: the GUM has no correlated form and `expanded()` then needs an explicit factor); with all-infinite input dof it is `inf` (normal-distribution coverage factor), since the GUM defines no correlated Welch-Satterthwaite form. |
| `names` | Input labels aligned with the arrays above. |

### UncertaintyResult.expanded()

```python
UncertaintyResult.expanded(
    coverage: float = 0.95,
    *,
    coverage_factor_override: float | None = None,
) -> tuple[float, float]
```

Coverage factor `k` and expanded uncertainty `U = k*uc`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `coverage` | Coverage probability in (0, 1); `0.95` by default. |
| `coverage_factor_override` | Explicit `k`. Required for a correlated budget with finite input degrees of freedom, where the GUM defines no effective-dof formula (`effective_dof` is NaN). |

**Returns:** The pair `(k, U)` (GUM clause 6, Annex G).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the effective dof are undefined and no explicit coverage factor is given. |

### UncertaintyResult.plot()

```python
UncertaintyResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the uncertainty budget (per-input contributions).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## UncertaintyWarning

A GUM propagation fell back outside its nominal assumptions.
