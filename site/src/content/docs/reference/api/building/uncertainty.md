---
title: "building.measurement.uncertainty"
description: "Measurement uncertainty in building acoustics (ISO 12999-1:2020)."
sidebar:
  label: "uncertainty"
---

Measurement uncertainty in building acoustics (ISO 12999-1:2020).

This module supplies the **measurement uncertainty** of the sound-insulation
quantities produced by the field/lab/prediction modules
([`phonometry.building.measurement.insulation`](/phonometry/reference/api/building/insulation/), [`phonometry.building.measurement.lab_insulation`](/phonometry/reference/api/building/lab-insulation/),
[`phonometry.building.prediction.simplified_model`](/phonometry/reference/api/building/simplified-model/)). ISO 12999-1 does not re-measure anything;
it tabulates *standard uncertainties* `u` derived from inter-laboratory tests
(ISO 5725) and prescribes how to expand and combine them.

**Three measurement situations (Clause 5.2)** fix which standard deviation is the
standard uncertainty `u`:

- **A**: laboratory characterisation (ISO 10140); `u` = reproducibility `σR`.
- **B**: same location, different teams; `u` = in-situ `σsitu`.
- **C**: same location, same operator/equipment repeated; `u` = repeatability `σr`.

**Tabulated standard uncertainties** (one-third-octave and single-number):

- Airborne `R`/`R'`/`Dn`/`DnT`: Table 2 (bands) and Table 3 (ratings).
- Impact `Ln`/`L'n`/`L'nT`: Table 4 (bands, situations B/C only) and Table 5
  (ratings). ISO 12999-1:2020 Table 4 has **no 500 Hz band** (the 2014 edition did).
- Reduction of impact noise by floor coverings `ΔL`/`ΔLw`: Table 6 (bands) and
  Table 7 (rating), situation A only.
- Upper 95 % limit of airborne reproducibility `σR95`: Annex D Tables D.1/D.2
  (situation A; informative). In ISO 12999-1:2014 these were extra columns of
  Tables 2/3.
- Maximum repeatability standard deviation for lab self-verification: Table 1.

**Expansion (Clause 8).** $U = k\,u$ (Formula 2) with the coverage
factor `k` of
Table 8 (a minimum of $k = 1$ is enforced). Declaring conformity with a
requirement uses the **one-sided** factor (Formulae 4/5); reporting a two-sided
interval $Y = y \pm U$ (Formula 3) uses the two-sided factor.

**Combination.** Uncorrelated quadrature $u_c = \sqrt{\sum u_i^2}$
(Formula C.2);
prediction input uncertainty (Formula A.1); model/reality combination (Formula A.2);
reduction by `m` independent measurements $u/\sqrt{m}$ (Formula A.7);
and the
uncorrelated single-number combination of Annex B (Formula B.2).

Clause/table numbers refer to ISO 12999-1:2020(E).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## band_uncertainty

```python
band_uncertainty(
    measurand: Measurand,
    situation: Situation,
    *,
    upper_limit: bool = False,
) -> BandUncertainty
```

Return the one-third-octave standard uncertainties for a measurand.

Airborne (Table 2) offers situations A/B/C; impact (Table 4) only B/C; the
reduction `ΔL` (Table 6) only A. `upper_limit=True` selects the `σR95`
upper limit for airborne, situation A (Annex D Table D.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `measurand` | `"airborne"`, `"impact"` or `"impact_reduction"`. |
| `situation` | Measurement situation `"A"`, `"B"` or `"C"` (Clause 5.2). |
| `upper_limit` | Select the `σR95` upper limit (airborne, situation A). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Unknown measurand, or a situation not tabulated for it. |

## BandUncertainty

```python
BandUncertainty(
    measurand: str,
    situation: str,
    frequencies: tuple[float, ...],
    uncertainties: tuple[float, ...],
    upper_limit: bool = False,
)
```

One-third-octave-band standard uncertainties (ISO 12999-1 Tables 2/4/6/D.1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `measurand` | `"airborne"`, `"impact"` or `"impact_reduction"`. |
| `situation` | Measurement situation `"A"`, `"B"` or `"C"` (Clause 5.2). |
| `frequencies` | Band centre frequencies, in Hz. |
| `uncertainties` | Standard uncertainty `u` per band, in dB. |
| `upper_limit` | `True` for the `σR95` upper limit (Annex D Table D.1). |

### BandUncertainty.plot()

```python
BandUncertainty.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band standard uncertainty spectrum.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### BandUncertainty.to_arrays()

```python
BandUncertainty.to_arrays() -> tuple[np.ndarray, np.ndarray]
```

Return `(frequencies, uncertainties)` as float `numpy.ndarray`.

## combine_uncertainties

```python
combine_uncertainties(*components: float) -> float
```

Combine independent standard uncertainties in quadrature (Formula C.2).

$u_c = \sqrt{\sum u_i^2}$ for uncorrelated contributions with unit
sensitivity
coefficients, also the model/reality combination of Formula (A.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `components` | Standard-uncertainty contributions, in dB (non-negative). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | No components, or a negative component. |

## coverage_factor

```python
coverage_factor(confidence: float = 0.95, one_sided: bool = False) -> float
```

Deprecated alias of [`insulation_coverage_factor`](/phonometry/reference/api/building/uncertainty/#insulation_coverage_factor).

## COVERAGE_FACTORS

*Constant* (`mappingproxy`).

```python
COVERAGE_FACTORS = {(0.68, False): 1.0, (0.8, False): 1.28, (0.9, False): 1.65, (0.95, False): 1.96, (0.99, False): 2.58, (0.999, False): 3.29, (0.84, True): 1.0, (0.9, True): 1.28, (0.95, True): 1.65, (0.975, True): 1.96, (0.995, True): 2.58, (0.9995, True): 3.29}
```

## expanded_uncertainty

```python
expanded_uncertainty(
    u: float,
    coverage: float = 0.95,
    one_sided: bool = False,
) -> float
```

Deprecated alias of [`insulation_expanded_uncertainty`](/phonometry/reference/api/building/uncertainty/#insulation_expanded_uncertainty).

## insulation_coverage_factor

```python
insulation_coverage_factor(
    confidence: float = 0.95,
    one_sided: bool = False,
) -> float
```

Return the coverage factor `k` for a confidence level (Table 8).

**Parameters**

| Name | Description |
| :--- | :--- |
| `confidence` | Confidence level as a fraction. Two-sided values are `0.68, 0.80, 0.90, 0.95, 0.99, 0.999`; one-sided values are `0.84, 0.90, 0.95, 0.975, 0.995, 0.9995`. |
| `one_sided` | Use the one-sided column (conformity checks, Formulae 4/5). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Confidence level not tabulated in Table 8. |

## insulation_expanded_uncertainty

```python
insulation_expanded_uncertainty(
    u: float,
    coverage: float = 0.95,
    one_sided: bool = False,
) -> float
```

Return the expanded uncertainty $U = k\,u$ (Formula 2, Clause 8).

The coverage factor `k` is taken from Table 8 for the requested confidence
level; a minimum of $k = 1$ is enforced (Clause 8).

**Parameters**

| Name | Description |
| :--- | :--- |
| `u` | Standard uncertainty `u`, in dB (must be non-negative). |
| `coverage` | Confidence level as a fraction (see [`insulation_coverage_factor`](/phonometry/reference/api/building/uncertainty/#insulation_coverage_factor)). |
| `one_sided` | Use the one-sided coverage factor (conformity checks). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Negative `u` or an untabulated confidence level. |

## maximum_repeatability_standard_deviation

```python
maximum_repeatability_standard_deviation() -> BandUncertainty
```

Return Table 1, the maximum repeatability standard deviation per band (Clause 5.8).

A laboratory verifies its own procedure when the repeatability standard
deviation of `nx` repeated measurements stays below these values.

## prediction_input_uncertainty

```python
prediction_input_uncertainty(
    sigma_reproducibility: float,
    sigma_product: float,
    n: int,
) -> float
```

Return the prediction input uncertainty `u_input` (Formula A.1).

$$
u_{\mathrm{input}} = \sqrt{\frac{\sigma_R^2 + \sigma_{\mathrm{product}}^2}{n} + \sigma_{\mathrm{product}}^2}
$$

combines the
reproducibility standard deviation with the product-homogeneity scatter over
`n` measurements of nominally identical specimens.

**Parameters**

| Name | Description |
| :--- | :--- |
| `sigma_reproducibility` | Reproducibility standard deviation `σR`, in dB. |
| `sigma_product` | Product-homogeneity standard deviation `σ_product`, in dB. |
| `n` | Number of measurements of the product ($n \ge 1$). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Non-positive `n` or a negative standard deviation. |

## reduce_by_independent_measurements

```python
reduce_by_independent_measurements(u: float, m: int) -> float
```

Reduce a standard uncertainty by `m` independent measurements (Formula A.7).

$u_{\mathrm{reduced}} = u / \sqrt{m}$: measurements by different
persons with different
equipment lower the in-situ uncertainty.

**Parameters**

| Name | Description |
| :--- | :--- |
| `u` | Standard uncertainty of a single measurement, in dB (non-negative). |
| `m` | Number of independent measurements ($m \ge 1$). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Non-positive `m` or negative `u`. |

## satisfies_lower_requirement

```python
satisfies_lower_requirement(
    value: float,
    expanded_uncertainty_value: float,
    requirement: float,
) -> bool
```

Test a minimum requirement with one-sided uncertainty (Formula 5).

Returns `True` when
$\text{value} - U > \text{requirement}$, e.g. an apparent sound
reduction index `R'w` provably exceeds a minimum. `U` should be computed
with the one-sided coverage factor.

## satisfies_upper_requirement

```python
satisfies_upper_requirement(
    value: float,
    expanded_uncertainty_value: float,
    requirement: float,
) -> bool
```

Test a maximum requirement with one-sided uncertainty (Formula 4).

Returns `True` when
$\text{value} + U < \text{requirement}$, e.g. a normalized impact
level `L'n,w` provably stays below a maximum. `U` should be computed with
the one-sided coverage factor.

## single_number_uncertainty

```python
single_number_uncertainty(
    quantity: str,
    situation: Situation,
    *,
    upper_limit: bool = False,
) -> float
```

Return the tabulated single-number standard uncertainty `u`, in dB.

Descriptors (case-insensitive, with aliases) cover the ISO 717 ratings:
`"r_w"` (also `rprime_w`/`dn_w`/`dnt_w`) and its spectrum-adaptation
variants `"r_w+c_50_5000"` etc. (Table 3); `"ln_w"`/`"ln_w+ci"` (Table 5);
`"delta_lw"` (Table 7). `upper_limit=True` selects the situation-A `σR95`
(Annex D Table D.2), defined for airborne descriptors only.

:::note
For the impact descriptors (`"ln_w"`/`"ln_w+ci"`, Table 5) the
situation-A value is an *estimate*: no reproducibility results are
available for impact sound insulation (Table 5, footnote a).
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `quantity` | Rating descriptor (see above). |
| `situation` | Measurement situation `"A"`, `"B"` or `"C"` (Clause 5.2). |
| `upper_limit` | Select the `σR95` upper limit (airborne, situation A). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Unknown descriptor, an untabulated situation, or an `upper_limit` request outside airborne/situation A. |

## single_number_uncertainty_uncorrelated

```python
single_number_uncertainty_uncorrelated(
    band_uncertainties: Sequence[float] | np.ndarray,
    reference_differences: Sequence[float] | np.ndarray,
) -> float
```

Uncorrelated single-number uncertainty from band uncertainties (Formula B.2).

$$
u(R_w{+}C) = \sqrt{\sum_i (w_i \, u_i)^2}
$$

with energy weights

$$
w_i = \frac{10^{(L_i - R_i)/10}}{\sum_j 10^{(L_j - R_j)/10}}
$$

derived from the
reference spectrum. This is the *no-correlation* estimate of Annex B; the
fully correlated bound (Formulae B.3-B.6) instead re-runs the ISO 717 rating
and is not reproduced here.

**Parameters**

| Name | Description |
| :--- | :--- |
| `band_uncertainties` | Per-band standard uncertainties `u_i`, in dB. |
| `reference_differences` | Per-band $L_i - R_i$ (reference-spectrum level minus measured band value), in dB. |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Mismatched lengths, empty input, or negative `u_i`. |

## uncertain_value

```python
uncertain_value(
    value: float,
    quantity: str,
    situation: Situation,
    *,
    coverage: float = 0.95,
    one_sided: bool = False,
    upper_limit: bool = False,
) -> UncertainValue
```

Attach the ISO 12999-1 expanded uncertainty to a single-number rating.

Convenience wrapper combining [`single_number_uncertainty`](/phonometry/reference/api/building/uncertainty/#single_number_uncertainty),
[`insulation_coverage_factor`](/phonometry/reference/api/building/uncertainty/#insulation_coverage_factor) and [`insulation_expanded_uncertainty`](/phonometry/reference/api/building/uncertainty/#insulation_expanded_uncertainty) into an
[`UncertainValue`](/phonometry/reference/api/building/uncertainty/#uncertainvalue) ($y \pm U$) without modifying the rating
dataclasses. For conformity checks pass `one_sided=True` and read
[`UncertainValue.lower`](/phonometry/reference/api/building/uncertainty/#uncertainvaluelower) / [`UncertainValue.upper`](/phonometry/reference/api/building/uncertainty/#uncertainvalueupper) (Formulae 4/5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `value` | Best estimate `y` (e.g. `Rw` in dB). |
| `quantity` | Rating descriptor (see [`single_number_uncertainty`](/phonometry/reference/api/building/uncertainty/#single_number_uncertainty)). |
| `situation` | Measurement situation `"A"`, `"B"` or `"C"` (Clause 5.2). |
| `coverage` | Confidence level as a fraction. |
| `one_sided` | Use the one-sided coverage factor. |
| `upper_limit` | Use the `σR95` upper limit (airborne, situation A). |

## UncertainValue

```python
UncertainValue(
    value: float,
    standard_uncertainty: float,
    coverage_factor: float,
    expanded_uncertainty: float,
    confidence: float,
    one_sided: bool,
)
```

A best estimate with its ISO 12999-1 expanded uncertainty (Clause 8).

**Attributes**

| Name | Description |
| :--- | :--- |
| `value` | Best estimate `y` (e.g. a weighted rating), in dB. |
| `standard_uncertainty` | Standard uncertainty `u`, in dB. |
| `coverage_factor` | Coverage factor `k` (Table 8). |
| `expanded_uncertainty` | $U = k\,u$, in dB. |
| `confidence` | Confidence level as a fraction (e.g. `0.95`). |
| `one_sided` | `True` for a one-sided interval (conformity checks). |

### UncertainValue.lower

*property*

Lower interval bound $y - U$ (Formula 3/5).

### UncertainValue.upper

*property*

Upper interval bound $y + U$ (Formula 3/4).
