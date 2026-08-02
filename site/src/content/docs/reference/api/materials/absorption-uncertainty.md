---
title: "materials.absorption_uncertainty"
description: "Measurement uncertainty for sound absorption (ISO 12999-2:2020)."
sidebar:
  label: "absorption_uncertainty"
---

Measurement uncertainty for sound absorption (ISO 12999-2:2020).

Companion of the sound-insulation uncertainty of [`phonometry.building.measurement.uncertainty`](/phonometry/reference/api/building/uncertainty/)
(ISO 12999-1). This part gives the standard uncertainty `u` of the quantities
produced by a reverberation-room absorption measurement and its ratings:

* the **sound absorption coefficient** `αs` and the **equivalent sound absorption
  area** `AT` measured according to ISO 354, in one-third-octave bands
  (Clause 5, Table 1);
* the **practical sound absorption coefficient** `αp` determined according to
  ISO 11654, in octave bands (Clause 6, Table 2);
* the single-number **weighted sound absorption coefficient** `αw` (ISO 11654)
  and **single-number rating** `DLα,NRD` (EN 1793-1) (Clause 7).

The standard uncertainty is estimated by the reproducibility standard deviation
`σR` (different laboratories) or the repeatability standard deviation `σr`
(same location, operator and equipment), both derived from inter-laboratory tests
to ISO 5725. Where specimen-specific uncertainty data exist they take precedence
(Clause 4).

**Band formulae (Clause 5-6).** For the absorption coefficient
$\sigma_R = m \alpha + n$ (Formula (1)/(4)); for the equivalent area
$\sigma_R = m A_T + n S$ with $S = 10~\text{m}^2$ (Formula (2));
the repeatability value is $\sigma_r = 0.6\,\sigma_R$ (Formula (3)/(5)).
`m` and `n` are the frequency-dependent
constants of Table 1 (one-third-octave, 63-5000 Hz) and Table 2 (octave,
250-4000 Hz).

**Single numbers (Clause 7).** `αw`: $\sigma_R = 0.035$
(Formula (6)), $\sigma_r = 0.020$ (Formula (7)). `DLα,NRD`:
$\sigma_R = 0.10\,D_{L\alpha,\mathrm{NRD}}$ (Formula (8)),
$\sigma_r = 0.02\,D_{L\alpha,\mathrm{NRD}}$ (Formula (9)).

**Reporting (Clause 8).** The expanded uncertainty is
$U = k\,u$ (Formula (10))
with the coverage factor `k` of Table 3 (Gaussian assumption). The reported
`U` is rounded to two decimal digits for absorption coefficients and one decimal
digit for the equivalent area and `DLα,NRD`; the code keeps the exact `U` and
exposes the rounded view through [`AbsorptionUncertaintyResult.reported_expanded_uncertainty`](/phonometry/reference/api/materials/absorption-uncertainty/#absorptionuncertaintyresultreported_expanded_uncertainty).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## absorption_coverage_factor

```python
absorption_coverage_factor(confidence: float = 0.95) -> float
```

Return the coverage factor `k` for a confidence level (Table 3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `confidence` | Confidence level as a fraction; one of `0.68, 0.80, 0.90, 0.95, 0.99, 0.999` (Table 3, Gaussian measurand). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Confidence level not tabulated in Table 3. |

## AbsorptionUncertaintyResult

```python
AbsorptionUncertaintyResult(
    quantity: str,
    condition: str,
    frequencies: np.ndarray,
    values: np.ndarray,
    standard_uncertainty: np.ndarray,
    coverage_factor: float,
    expanded_uncertainty: np.ndarray,
)
```

Standard and expanded uncertainty of an absorption quantity
(ISO 12999-2).

For the single-number quantities (`αw`, `DLα,NRD`) the frequency and value
arrays are empty and the uncertainty arrays hold a single element.

**Attributes**

| Name | Description |
| :--- | :--- |
| `quantity` | `"absorption_coefficient"`, `"equivalent_area"`, `"practical_coefficient"`, `"weighted_coefficient"` or `"single_number_rating"`. |
| `condition` | `"reproducibility"` (`σR`) or `"repeatability"` (`σr`). |
| `frequencies` | Band centre frequencies, in Hz (empty for single numbers). |
| `values` | The input quantity per band (empty for single numbers). |
| `standard_uncertainty` | Standard uncertainty `u` (`σR`/`σr`), one per band. |
| `coverage_factor` | Coverage factor `k` (Table 3). |
| `expanded_uncertainty` | Exact $U = k\,u$ (Formula (10)), one per band. |

### AbsorptionUncertaintyResult.lower

*property*

Lower interval bound `value − U` (exact `U`).

### AbsorptionUncertaintyResult.plot()

```python
AbsorptionUncertaintyResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the quantity with its `±U` uncertainty ribbon (band quantities).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### AbsorptionUncertaintyResult.reported_expanded_uncertainty

*property*

`U` rounded for reporting (Clause 8): two decimals for absorption
coefficients, one decimal for the equivalent area and `DLα,NRD`.

### AbsorptionUncertaintyResult.upper

*property*

Upper interval bound `value + U` (exact `U`).

## equivalent_area_uncertainty

```python
equivalent_area_uncertainty(
    area: float | Sequence[float] | np.ndarray,
    frequencies: float | Sequence[float] | np.ndarray,
    *,
    condition: str = 'reproducibility',
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult
```

Uncertainty of the ISO 354 equivalent sound absorption area `AT`
(Clause 5).

$\sigma_R = m A_T + n S$ with $S = 10~\text{m}^2$ (Formula (2))
per one-third-octave band; $\sigma_r = 0.6\,\sigma_R$ (Formula (3)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `area` | Equivalent sound absorption area `AT` per band, in m². |
| `frequencies` | One-third-octave midband frequencies in Hz (Table 1), aligned with `area`. |
| `condition` | `"reproducibility"` (default) or `"repeatability"`. |
| `confidence` | Confidence level for the coverage factor `k` (Table 3). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Mismatched shapes, an untabulated frequency, an unknown `condition` or an untabulated `confidence`. |

## practical_coefficient_uncertainty

```python
practical_coefficient_uncertainty(
    alpha_p: float | Sequence[float] | np.ndarray,
    frequencies: float | Sequence[float] | np.ndarray,
    *,
    condition: str = 'reproducibility',
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult
```

Uncertainty of the ISO 11654 practical absorption coefficient `αp`
(Clause 6).

$\sigma_R = m \alpha_p + n$ (Formula (4)) per octave band;
$\sigma_r = 0.6\,\sigma_R$ (Formula (5)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `alpha_p` | Practical sound absorption coefficient `αp` per band. |
| `frequencies` | Octave midband frequencies in Hz (Table 2, 250-4000), aligned with `alpha_p`. |
| `condition` | `"reproducibility"` (default) or `"repeatability"`. |
| `confidence` | Confidence level for the coverage factor `k` (Table 3). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Mismatched shapes, an untabulated frequency, an unknown `condition` or an untabulated `confidence`. |

## single_number_rating_uncertainty

```python
single_number_rating_uncertainty(
    dl_alpha: float,
    *,
    condition: str = 'reproducibility',
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult
```

Uncertainty of the EN 1793-1 single-number rating `DLα,NRD`
(Clause 7).

$\sigma_R = 0.10\,D_{L\alpha,\mathrm{NRD}}$ (Formula (8)),
$\sigma_r = 0.02\,D_{L\alpha,\mathrm{NRD}}$ (Formula (9)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `dl_alpha` | Single-number rating `DLα,NRD`, in dB (non-negative). |
| `condition` | `"reproducibility"` (default) or `"repeatability"`. |
| `confidence` | Confidence level for the coverage factor `k` (Table 3). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Negative `dl_alpha`, unknown `condition` or an untabulated `confidence`. |

## sound_absorption_coefficient_uncertainty

```python
sound_absorption_coefficient_uncertainty(
    alpha: float | Sequence[float] | np.ndarray,
    frequencies: float | Sequence[float] | np.ndarray,
    *,
    condition: str = 'reproducibility',
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult
```

Uncertainty of the ISO 354 sound absorption coefficient `αs`
(Clause 5).

$\sigma_R = m \alpha_s + n$ (Formula (1)) per one-third-octave band;
$\sigma_r = 0.6\,\sigma_R$ (Formula (3)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `alpha` | Sound absorption coefficient `αs` per band. |
| `frequencies` | One-third-octave midband frequencies in Hz (Table 1, 63-5000), aligned with `alpha`. |
| `condition` | `"reproducibility"` (default) or `"repeatability"`. |
| `confidence` | Confidence level for the coverage factor `k` (Table 3). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Mismatched shapes, an untabulated frequency, an unknown `condition` or an untabulated `confidence`. |

## weighted_coefficient_uncertainty

```python
weighted_coefficient_uncertainty(
    alpha_w: float = 0.0,
    *,
    condition: str = 'reproducibility',
    confidence: float = 0.95,
) -> AbsorptionUncertaintyResult
```

Uncertainty of the ISO 11654 weighted absorption coefficient `αw`
(Clause 7).

The standard uncertainty is a constant: $\sigma_R = 0.035$
(Formula (6)), $\sigma_r = 0.020$ (Formula (7)), independent of
the value.

**Parameters**

| Name | Description |
| :--- | :--- |
| `alpha_w` | Weighted sound absorption coefficient `αw` (carried through for the reported interval; does not affect `u`). |
| `condition` | `"reproducibility"` (default) or `"repeatability"`. |
| `confidence` | Confidence level for the coverage factor `k` (Table 3). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Unknown `condition` or an untabulated `confidence`. |
