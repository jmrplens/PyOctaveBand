---
title: "materials.absorption_rating"
description: "Single-number rating of sound absorption (ISO 11654:1997)."
sidebar:
  label: "absorption_rating"
---

Single-number rating of sound absorption (ISO 11654:1997).

From one-third-octave sound absorption coefficients `alpha_s` measured in
a reverberation room (ISO 354) this module forms the practical sound
absorption coefficient `alpha_p` per octave band (Clause 4.1), the
weighted sound absorption coefficient `alpha_w` by the reference-curve
shifting method (Clause 4.2), the shape indicators `L`, `M`, `H`
(Clause 4.3) and the sound absorption class `A` to `E` of the
informative Table B.1 (Annex B).

**Practical absorption coefficient (Clause 4.1).** For each octave band the
practical coefficient is the arithmetic mean of the three one-third-octave
coefficients it contains:

```text
alpha_p,i = (alpha_i1 + alpha_i2 + alpha_i3) / 3
```

The mean is evaluated to the second decimal and then rounded in steps of
0,05 (the NOTE of Clause 4.1 gives `0,92 -> 0,90`); rounded means above
1,00 are set to 1,00. The five rating bands are 250, 500, 1000, 2000 and
4000 Hz, fed from the fifteen one-third octaves 200 Hz to 5000 Hz.

**Weighted absorption (Clause 4.2).** The fixed reference curve of Figure 1
(`{250: 0.80, 500: 1.00, 1000: 1.00, 2000: 1.00, 4000: 0.90}`) is shifted
downwards, towards the measured `alpha_p`, in steps of 0,05 until the sum
of the unfavourable deviations -- taken only where the measured value lies
below the shifted curve, with magnitude `curve - measured` -- is not more
than 0,10. `alpha_w` is the shifted-curve value read at 500 Hz, i.e.
`1.00 - shift`.

**Shape indicators (Clause 4.3).** A shape indicator is appended in
parentheses whenever a practical coefficient exceeds the shifted reference
curve by 0,25 or more: `L` at 250 Hz, `M` at 500 Hz or 1000 Hz, `H` at
2000 Hz or 4000 Hz (e.g. `0.60(M)`).

**Absorption class (Table B.1, informative).** `alpha_w` maps to a class:
A (0,90-1,00), B (0,80-0,85), C (0,60-0,75), D (0,30-0,55), E (0,15-0,25),
and "Not classified" (0,00-0,10). Because `alpha_w` is always a multiple
of 0,05 these ranges partition the grid exactly.

The rating is defined only over the whole reference range 250 Hz to
4000 Hz (Clause 1); the 125 Hz octave that is customarily plotted is not
part of the shift and is not produced here.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## absorption_class

```python
absorption_class(alpha_w: float) -> str
```

Sound absorption class for `alpha_w` (ISO 11654 Table B.1, Annex B).

**Parameters**

| Name | Description |
| :--- | :--- |
| `alpha_w` | Weighted sound absorption coefficient (a multiple of 0,05 in `[0, 1]`). |

**Returns:** `"A"`, `"B"`, `"C"`, `"D"`, `"E"` or `"Not classified"`.

## AbsorptionRatingResult

```python
AbsorptionRatingResult(
    alpha_w: float,
    shape_indicator: str,
    absorption_class: str,
    shift: float,
    unfavourable_sum: float,
    band_centers: NDArray[np.float64],
    measured: NDArray[np.float64],
    shifted_reference: NDArray[np.float64],
    third_octave_alpha_s: NDArray[np.float64] | None = None,
    third_octave_bands: NDArray[np.float64] | None = None,
)
```

Weighted sound absorption rating (ISO 11654:1997).

**Attributes**

| Name | Description |
| :--- | :--- |
| `alpha_w` | Weighted sound absorption coefficient `alpha_w`, the shifted reference curve read at 500 Hz (Clause 4.2). A multiple of 0,05 in `[0, 1]`. |
| `shape_indicator` | Concatenated shape indicators, `L`/`M`/`H` in that order, or an empty string when none applies (Clause 4.3). |
| `absorption_class` | Sound absorption class `A`-`E` or `"Not classified"` from Table B.1 (Annex B). |
| `shift` | Downward shift applied to the reference curve, in absorption units (Clause 4.2); `alpha_w == 1.00 - shift`. |
| `unfavourable_sum` | Sum of the unfavourable deviations at the final shift (Clause 4.2); at most 0,10. |
| `band_centers` | Octave rating-band centre frequencies, in Hz (250 Hz to 4000 Hz). |
| `measured` | Practical absorption coefficients `alpha_p` used for the rating (snapped to the 0,05 grid of Clause 4.1). |
| `shifted_reference` | Reference curve of Figure 1 after the final shift. |
| `third_octave_alpha_s` | The fifteen one-third-octave sound absorption coefficients `alpha_s` (200 Hz to 5000 Hz, ISO 354) the rating was built from, as given; `None` when the rating was formed from octave practical coefficients only. Carried so the fiche can show the full one-third-octave table every accredited ISO 354 certificate reports. |
| `third_octave_bands` | The one-third-octave centre frequencies, in Hz, that pair with `third_octave_alpha_s` (the module [`THIRD_OCTAVE_BANDS`](/phonometry/reference/api/materials/absorption-rating/#third_octave_bands)); `None` when `third_octave_alpha_s` is. |

### AbsorptionRatingResult.plot()

```python
AbsorptionRatingResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the practical curve vs the shifted reference (ISO 11654).

Unfavourable deviations (measured below the shifted reference) are
shaded and `alpha_w` annotated. Requires matplotlib
(`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

### AbsorptionRatingResult.rating_label

*property*

The rating as reported in Clause 5.3, e.g. `"0.60(M)"` or
`"0.60"` when no shape indicator applies.

### AbsorptionRatingResult.report()

```python
AbsorptionRatingResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 11654 sound-absorption rating fiche to a PDF.

Writes a one-page accredited absorption report: the standard-basis
line, an optional metadata header block, an absorption table beside the
practical-versus-shifted-reference plot (the result's own `plot`),
the boxed `alpha_w` result with its absorption class, an optional
verdict row and a footer with the fixed disclaimer. When the rating was
built by [`weighted_absorption_from_third_octave`](/phonometry/reference/api/materials/absorption-rating/#weighted_absorption_from_third_octave), the table is the
full ISO 354 one-third-octave `alpha_s` table with the octave
`alpha_p` on the matching rows, as accredited certificates print it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a prediction fiche (body, result and disclaimer only). A supplied `requirement` is read as the minimum `alpha_w`. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table adds the ISO 11654 evaluation columns (practical coefficient, shifted reference, unfavourable deviation) instead of the two-column `f \| alpha_p` table. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## OCTAVE_BANDS

*Constant* (`tuple`).

```python
OCTAVE_BANDS = (250, 500, 1000, 2000, 4000)
```

## practical_absorption_coefficient

```python
practical_absorption_coefficient(
    third_octave_alpha_s: Mapping[Any, float] | Sequence[float] | ArrayLike,
) -> NDArray[np.float64]
```

Practical sound absorption coefficients `alpha_p` (ISO 11654 Clause 4.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `third_octave_alpha_s` | The fifteen one-third-octave coefficients `alpha_s` from 200 Hz to 5000 Hz (ISO 354), as a sequence ordered low to high or a mapping keyed by band centre frequency (Hz). Values may exceed 1,00 (reverberation-room measurement); the resulting octave coefficient is capped at 1,00. |

**Returns:** The five octave practical coefficients for 250, 500, 1000, 2000 and 4000 Hz, each the mean of its three one-third octaves rounded in steps of 0,05.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if any coefficient is negative or the wrong number of values is supplied. |

## REFERENCE_CURVE

*Constant* (`dict`).

```python
REFERENCE_CURVE = {250: 0.8, 500: 1.0, 1000: 1.0, 2000: 1.0, 4000: 0.9}
```

## THIRD_OCTAVE_BANDS

*Constant* (`tuple`).

```python
THIRD_OCTAVE_BANDS = (200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000)
```

## weighted_absorption

```python
weighted_absorption(
    alpha_p: Mapping[Any, float] | Sequence[float] | ArrayLike,
) -> AbsorptionRatingResult
```

Weighted sound absorption coefficient `alpha_w` (ISO 11654 Clause 4.2).

The reference curve of Figure 1 is shifted downwards in steps of 0,05,
towards the measured practical coefficients, until the sum of the
unfavourable deviations (measured below the shifted curve) is at most
0,10; `alpha_w` is the shifted curve read at 500 Hz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `alpha_p` | The five octave practical coefficients for 250, 500, 1000, 2000 and 4000 Hz (e.g. from [`practical_absorption_coefficient`](/phonometry/reference/api/materials/absorption-rating/#practical_absorption_coefficient)), as a sequence ordered low to high or a mapping keyed by band centre frequency (Hz). Inputs are snapped to the 0,05 grid of Clause 4.1. |

**Returns:** A frozen [`AbsorptionRatingResult`](/phonometry/reference/api/materials/absorption-rating/#absorptionratingresult) with `alpha_w`, the shape indicators, the applied shift, the fitted reference curve and the absorption class.

## weighted_absorption_from_third_octave

```python
weighted_absorption_from_third_octave(
    third_octave_alpha_s: Mapping[Any, float] | Sequence[float] | ArrayLike,
) -> AbsorptionRatingResult
```

Weighted absorption rating from one-third-octave `alpha_s` (ISO 11654).

Convenience wrapper over [`practical_absorption_coefficient`](/phonometry/reference/api/materials/absorption-rating/#practical_absorption_coefficient) and
[`weighted_absorption`](/phonometry/reference/api/materials/absorption-rating/#weighted_absorption): it forms the octave practical coefficients
`alpha_p` (Clause 4.1) from the fifteen one-third-octave coefficients,
rates them (Clause 4.2) and returns the same
[`AbsorptionRatingResult`](/phonometry/reference/api/materials/absorption-rating/#absorptionratingresult) as [`weighted_absorption`](/phonometry/reference/api/materials/absorption-rating/#weighted_absorption) would, but
with the input `alpha_s` and its band centres retained on the result so a
fiche can print the full one-third-octave table that every accredited
ISO 354 certificate carries.

**Parameters**

| Name | Description |
| :--- | :--- |
| `third_octave_alpha_s` | The fifteen one-third-octave coefficients `alpha_s` from 200 Hz to 5000 Hz (ISO 354), as a sequence ordered low to high or a mapping keyed by band centre frequency (Hz). Values may exceed 1,00 (reverberation-room measurement); the octave coefficient is capped at 1,00. |

**Returns:** A frozen [`AbsorptionRatingResult`](/phonometry/reference/api/materials/absorption-rating/#absorptionratingresult) identical to [`weighted_absorption`](/phonometry/reference/api/materials/absorption-rating/#weighted_absorption) on the derived `alpha_p`, additionally carrying `third_octave_alpha_s` (the input as given) and `third_octave_bands` ([`THIRD_OCTAVE_BANDS`](/phonometry/reference/api/materials/absorption-rating/#third_octave_bands)).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if any coefficient is negative or the wrong number of values is supplied. |
