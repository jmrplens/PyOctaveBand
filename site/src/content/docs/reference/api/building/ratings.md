---
title: "building.measurement.ratings"
description: "Single-number weighted ratings of sound insulation and their spectrum adaptation terms (ISO 717-1 airborne, ISO 717-2 impact)."
sidebar:
  label: "ratings"
---

Single-number weighted ratings of sound insulation and their spectrum
adaptation terms (ISO 717-1 airborne, ISO 717-2 impact).

ISO 717 rates a curve, not a room. Whatever produced the band values (a field
measurement to ISO 16283, a laboratory measurement to ISO 10140, a prediction
to ISO 12354), both parts of ISO 717 reduce them to a single number by the same
reference-curve method: one shift search, one pair of deviation bounds, one
rounding rule, run against the airborne reference curve of ISO 717-1 or the
impact reference curve of ISO 717-2, whose only structural difference is the
sign an unfavourable deviation has. That shared machinery, with the Table 3
reference curves and the Table 4 / Table B.1 spectra it reads, is the subject
of this module.

**Weighted rating (ISO 717-1).** The reference-curve method of Clause 4.4
shifts the reference curve of Table 3 in 1 dB steps towards the measured
curve until the sum of unfavourable deviations (measured below the
shifted reference) is as large as possible but not more than 32,0 dB for
the 16 one-third-octave bands (100 Hz to 3150 Hz) or 10,0 dB for the 5
octave bands (125 Hz to 2000 Hz). The weighted rating (`Rw`, `R'w`,
`Dn,w`, `DnT,w` ...) is the shifted reference read at 500 Hz. The
spectrum adaptation terms are $C = X_{\mathrm{A}1} - X_\mathrm{w}$ and
$C_\mathrm{tr} = X_{\mathrm{A}2} - X_\mathrm{w}$
with $X_{\mathrm{A}j} = -10 \log_{10} \sum 10^{(L_{ij} - X_i)/10}$ rounded to an
integer, using
the A-weighted spectra No. 1 (pink noise, `C`) and No. 2 (urban traffic,
`Ctr`) of Table 4 (Clause 4.5, Formula (1) and (2)). Input levels are
reduced to one decimal place before use (Clause 4.4, footnote 1). The
reference values, spectra and shifting rule are identical in the 2013 and
2020 editions of ISO 717-1.

**Enlarged frequency ranges (ISO 717-1 Annex B; ISO 717-2 A.2.1 NOTE).**
When measurements cover an enlarged range, additional adaptation terms are
stated with the range as a subscript: `C50-3150`, `C50-5000`,
`C100-5000` (and the `Ctr` counterparts) with the Table B.1 spectra, and
`CI,50-2500` for impact. [`weighted_rating_extended`](/phonometry/reference/api/building/ratings/#weighted_rating_extended) and
[`weighted_impact_rating_extended`](/phonometry/reference/api/building/ratings/#weighted_impact_rating_extended) compute them alongside the core
rating. Both accept `one_decimal=True` for the "1/10 dB for the expression
of uncertainty" variant of Clauses 4.4/4.5 (reference-curve shift in 0,1 dB
steps and one-decimal reductions), which ISO 12999-1:2020 Annex B requires
when stating the uncertainty of single-number values.

**Weighted impact rating (ISO 717-2).** The reference-curve method of
Clause 4.3 shifts the Table 3 impact reference curve towards the measured
curve until the sum of unfavourable deviations (here where the
**measurement exceeds** the reference, the sign opposite to airborne) is
as large as possible but not more than 32,0 dB (16 one-third-octave bands)
or 10,0 dB (5 octave bands). The rating (`Ln,w`, `L'n,w`, `L'nT,w`)
is the shifted reference read at 500 Hz, reduced by a further 5 dB for
octave bands (Clause 4.3.2). The spectrum adaptation term
$C_\mathrm{I} = L_\mathrm{n,sum} - 15 - L_\mathrm{n,w}$ uses the energetic sum `Ln,sum`
over
100 Hz to 2500 Hz (one-third octave) or 125 Hz to 2000 Hz (octave),
rounded to an integer (Clause A.2.1, Formulae (A.1) to (A.3)). The Table 3
reference values, the shifting rule and CI are identical in the 2013 and
2020 editions of ISO 717-2 (the 2020 edition only adds Annex D for the
rubber-ball heavy/soft impactor, out of scope here).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## ExtendedImpactRatingResult

```python
ExtendedImpactRatingResult(
    rating: float,
    ci: float,
    ci_50_2500: float | None,
    core: ImpactRatingResult,
    band_centers: np.ndarray | None = None,
    measured: np.ndarray | None = None,
)
```

Weighted impact rating with `CI,50-2500` (ISO 717-2:2020 A.2.1 NOTE).

Values are integers unless computed with `one_decimal=True`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `rating` | Weighted impact rating (`Ln,w`, ...) from the core 100-3150 Hz bands, in dB. |
| `ci` | Core spectrum adaptation term `CI` (100-2500 Hz), in dB. |
| `ci_50_2500` | Enlarged-range term `CI,50-2500`, in dB, or `None` when the supplied bands do not cover 50-2500 Hz. |
| `core` | The integer-mode [`ImpactRatingResult`](/phonometry/reference/api/building/ratings/#impactratingresult) of the core bands (independent of `one_decimal`). |
| `band_centers` | Band centre frequencies of the full (enlarged-range) measured curve, in Hz. Defaults to `None` for backward-compatible construction. |
| `measured` | The measured impact levels over the full enlarged range (after the one-decimal reduction of Clause 4.3), in dB. Defaults to `None`. |

### ExtendedImpactRatingResult.plot()

```python
ExtendedImpactRatingResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the enlarged-range curve vs the shifted reference (ISO 717-2).

The measured curve is drawn over the full enlarged range, the
ISO 717-2 reference curve (after the final shift) over the 16 core
bands 100-3150 Hz, with the unfavourable deviations (measurement
above the reference) shaded on the core bands and the bands outside
the core range marked as the enlarged range; the title carries the
impact rating with `CI` and, when covered, `CI,50-2500`.
Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## ExtendedWeightedRatingResult

```python
ExtendedWeightedRatingResult(
    rating: float,
    c: float,
    ctr: float,
    c_50_3150: float | None,
    c_50_5000: float | None,
    c_100_5000: float | None,
    ctr_50_3150: float | None,
    ctr_50_5000: float | None,
    ctr_100_5000: float | None,
    core: WeightedRatingResult,
    band_centers: np.ndarray | None = None,
    measured: np.ndarray | None = None,
)
```

Weighted rating with the enlarged-range adaptation terms (ISO 717-1 Annex B).

All values are integers unless the result was computed with
`one_decimal=True` (the "1/10 dB for the expression of uncertainty"
variant of Clauses 4.4/4.5), in which case they carry one decimal place.
An extended term is `None` when the supplied bands do not cover its
frequency range.

**Attributes**

| Name | Description |
| :--- | :--- |
| `rating` | Weighted rating (`Rw`, `R'w`, ...) from the core 100-3150 Hz bands, in dB. |
| `c` | Core spectrum adaptation term `C` (100-3150 Hz), in dB. |
| `ctr` | Core spectrum adaptation term `Ctr` (100-3150 Hz), in dB. |
| `c_50_3150` | `C50-3150`, in dB, or `None`. |
| `c_50_5000` | `C50-5000`, in dB, or `None`. |
| `c_100_5000` | `C100-5000`, in dB, or `None`. |
| `ctr_50_3150` | `Ctr,50-3150`, in dB, or `None`. |
| `ctr_50_5000` | `Ctr,50-5000`, in dB, or `None`. |
| `ctr_100_5000` | `Ctr,100-5000`, in dB, or `None`. |
| `core` | The integer-mode [`WeightedRatingResult`](/phonometry/reference/api/building/ratings/#weightedratingresult) of the core bands (independent of `one_decimal`), for plotting and the unfavourable-deviation sum. |
| `band_centers` | Band centre frequencies of the full (enlarged-range) measured curve, in Hz. Defaults to `None` for backward-compatible construction. |
| `measured` | The measured band quantities over the full enlarged range (after the one-decimal reduction of Clause 4.4), in dB. Defaults to `None`. |

### ExtendedWeightedRatingResult.plot()

```python
ExtendedWeightedRatingResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the enlarged-range curve vs the shifted reference (Annex B).

The measured curve is drawn over the full enlarged range, the
ISO 717-1 reference curve (after the final shift) over the 16 core
bands 100-3150 Hz, with the unfavourable deviations shaded on the
core bands and the bands outside the core range marked as the
enlarged range; the title carries `Rw (C; Ctr)` and every Annex B
adaptation term the input covered. Requires matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

## impact_improvement_adaptation_term

```python
impact_improvement_adaptation_term(
    delta_l: Sequence[float] | np.ndarray,
) -> int
```

Adaptation term `CI,Δ` of a floor covering (ISO 717-2:2020 A.2.2).

$C_{\mathrm{I},\Delta} = C_\mathrm{I,r,0} - C_\mathrm{I,r}$ (Formula (A.4)) with
$C_\mathrm{I,r,0} = -11$ dB (the
bare Table 4 reference floor) and `CI,r` the ISO 717-2 spectrum
adaptation term of the reference floor with the covering under test,
$L_\mathrm{n,r} = L_\mathrm{n,r,0} - \Delta L$ (Formula (1)). Together with
[`weighted_impact_improvement`](/phonometry/reference/api/building/ratings/#weighted_impact_improvement) it yields the single-number reduction
for a flat spectrum, $\Delta L_\mathrm{lin} = \Delta L_\mathrm{w} + C_{\mathrm{I},\Delta}$
(Formula (A.5)). ISO 16251-1
Clause 8 e) requires this term in the statement of results.

**Parameters**

| Name | Description |
| :--- | :--- |
| `delta_l` | The reduction of impact sound pressure level `ΔL` per band, in dB; 16 one-third-octave values from 100 Hz to 3150 Hz. |

**Returns:** The spectrum adaptation term `CI,Δ`, in dB (integer).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `delta_l` is not 16 one-third-octave values, or is non-finite. |

## ImpactRatingResult

```python
ImpactRatingResult(
    rating: int,
    ci: int,
    unfavourable_sum: float,
    band_centers: np.ndarray | None = None,
    measured: np.ndarray | None = None,
    shifted_reference: np.ndarray | None = None,
    quantity: Literal['impact'] = 'impact',
)
```

Single-number weighted impact rating and CI (ISO 717-2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `rating` | Weighted impact rating (`Ln,w`, `L'n,w`, `L'nT,w`), the shifted reference read at 500 Hz, in dB (Clause 4.3; octave-band ratings include the -5 dB reduction of Clause 4.3.2). Integer. |
| `ci` | Spectrum adaptation term `CI` (Clause A.2.1), in dB. Integer. |
| `unfavourable_sum` | Sum of unfavourable deviations at the final shift, in dB (Clause 4.3); at most 32,0 (16 bands) or 10,0 (5 bands). |
| `band_centers` | Band centre frequencies of the measured curve, in Hz. Defaults to `None` for backward-compatible construction. |
| `measured` | The measured impact levels used for the rating (after the one-decimal reduction of Clause 4.3.1), in dB. Defaults to `None`. |
| `shifted_reference` | Table 3 impact reference curve after the final shift, in dB. Defaults to `None`. |
| `quantity` | Always `"impact"` (ISO 717-2), selecting the impact labels of the ISO 717 Annex C report. |

### ImpactRatingResult.plot()

```python
ImpactRatingResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the measured curve vs the shifted reference (ISO 717-2).

Unfavourable deviations (measurement above the reference, the sign
opposite to airborne) are shaded and `Ln,w (CI)` annotated.
Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ImpactRatingResult.report()

```python
ImpactRatingResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
    symbol: str | None = None,
) -> str
```

Render an ISO 717-2 impact-insulation fiche to a PDF.

Writes a one-page accredited-laboratory report for impact sound: the
standard-basis line, an optional metadata header block, the band
table beside the measured-versus-shifted-reference plot (the
result's own `plot`), the boxed `Ln,w (CI)` result, an
optional verdict row and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a prediction fiche (body, result and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table uses the ISO 717 Annex C columns (frequency, measured value, shifted reference, unfavourable deviation) instead of the two-column `f \| value` table. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |
| `symbol` | The reported single-number quantity, as plain text: `"Ln,w"` (the default when `None`), `"L'n,w"` or `"L'nT,w"` per ISO 717-2 Table 1, so a field measurement is not mislabelled with the laboratory descriptor. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`, `symbol` is not a valid quantity-symbol shape, or the result was built without the per-band data (`band_centers`, `measured`, `shifted_reference`). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## weighted_impact_improvement

```python
weighted_impact_improvement(delta_l: Sequence[float] | np.ndarray) -> int
```

Weighted reduction of impact level `ΔLw` (ISO 717-2:2020 §5).

Relates a measured improvement spectrum `ΔL` to the heavyweight
reference
floor of Table 4: the reference level with the covering is
$L_\mathrm{n,r} = L_\mathrm{n,r,0} - \Delta L$ (Formula (1)) and the weighted
improvement is
$\Delta L_\mathrm{w} = L_\mathrm{n,r,0,w} - L_\mathrm{n,r,w} = 78 - L_\mathrm{n,r,w}$
(Formula (2)), where `Ln,r,w` is
the ISO 717-2 weighted rating of `Ln,r` from
[`weighted_impact_rating`](/phonometry/reference/api/building/ratings/#weighted_impact_rating).

**Parameters**

| Name | Description |
| :--- | :--- |
| `delta_l` | The reduction of impact sound pressure level `ΔL` per band, in dB; 16 one-third-octave values from 100 Hz to 3150 Hz (e.g. from a floor-covering measurement to ISO 10140-3 or ISO 16251-1). |

**Returns:** The weighted reduction `ΔLw`, in dB (rounded, per ISO 717-2).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `delta_l` is not 16 one-third-octave values, or is non-finite. |

## weighted_impact_rating

```python
weighted_impact_rating(
    values_by_band: Sequence[float] | np.ndarray,
    bands: str | None = None,
) -> ImpactRatingResult
```

Single-number weighted impact rating and CI per ISO 717-2.

Applies the reference-curve method of Clause 4.3: the Table 3 impact
reference curve is shifted in 1 dB steps towards the measured curve
until the sum of unfavourable deviations is as large as possible but
not more than 32,0 dB (16 one-third-octave bands, 100 Hz to 3150 Hz)
or 10,0 dB (5 octave bands, 125 Hz to 2000 Hz). For impact sound an
unfavourable deviation occurs where the **measurement exceeds** the
reference (the sign opposite to ISO 717-1 airborne). The rating is the
shifted reference read at 500 Hz; for octave bands it is then reduced
by 5 dB (Clause 4.3.2). The spectrum adaptation term `CI` follows
Clause A.2.1. Input values are first reduced to one decimal place
(Clause 4.3.1, footnote 1).

The shift search reuses the verified engine of [`weighted_rating`](/phonometry/reference/api/building/ratings/#weighted_rating)
on the negated curves: minimising
$\sum \max(0, \text{measured} - (\text{ref} + k))$ over `k`
equals maximising
$\sum \max(0, (-\text{ref}) + (-k) - (-\text{measured}))$, the
airborne problem, so no separate search is duplicated.

**Parameters**

| Name | Description |
| :--- | :--- |
| `values_by_band` | Measured impact levels (`Ln`, `L'n`, `L'nT`) in dB. 16 values are read as one-third-octave bands, 5 values as octave bands. |
| `bands` | `"third-octave"`, `"octave"` or `None` to infer the band set from the number of values. |

**Returns:** [`ImpactRatingResult`](/phonometry/reference/api/building/ratings/#impactratingresult) with `rating`, `ci` and `unfavourable_sum`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the number of values does not match the band set, or if any value is non-finite. |

## weighted_impact_rating_extended

```python
weighted_impact_rating_extended(
    values_by_band: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray | None = None,
    *,
    one_decimal: bool = False,
) -> ExtendedImpactRatingResult
```

Weighted impact rating with `CI,50-2500` (ISO 717-2:2020 A.2.1).

Computes the weighted impact rating from the core one-third-octave bands
100-3150 Hz (Clause 4.3) and, when the input covers 50-2500 Hz, the
enlarged-range spectrum adaptation term `CI,50-2500` of the A.2.1 NOTE:
the energetic sum runs over 50-2500 Hz instead of 100-2500 Hz in
Formula (A.1), $C_\mathrm{I} = L_\mathrm{n,sum} - 15 - L_\mathrm{n,w}$.

With `one_decimal=True` the reference-curve shift runs in 0,1 dB steps
and the sums keep one decimal place (Clauses 4.3.1/4.4; e.g. the
reference floor yields $L_\mathrm{n,r,0,w} = 77.6$ dB and
$C_\mathrm{I,r,0} = -10.3$ dB
as printed in A.2.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `values_by_band` | Measured impact levels (`Ln`, `L'n`, `L'nT`) in dB, one-third-octave bands. |
| `frequencies` | Band centre frequencies, in Hz (one per value). `None` assumes exactly the 16 core bands 100-3150 Hz. |
| `one_decimal` | Use the 0,1 dB shift and one-decimal reductions. |

**Returns:** An [`ExtendedImpactRatingResult`](/phonometry/reference/api/building/ratings/#extendedimpactratingresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the input is not one-dimensional and finite, the band counts differ, or the core bands are missing. |

## weighted_rating

```python
weighted_rating(
    values_by_band: Sequence[float] | np.ndarray,
    bands: str | None = None,
) -> WeightedRatingResult
```

Single-number weighted rating and C / Ctr per ISO 717-1.

Applies the reference-curve method of Clause 4.4: the Table 3
reference curve is shifted in 1 dB steps towards the measured curve
until the sum of unfavourable deviations is as large as possible but
not more than 32,0 dB (16 one-third-octave bands, 100 Hz to 3150 Hz)
or 10,0 dB (5 octave bands, 125 Hz to 2000 Hz). The rating is the
shifted reference read at 500 Hz. The spectrum adaptation terms
`C` and `Ctr` follow Clause 4.5 with the Table 4 spectra No. 1 and
No. 2. Input values are first reduced to one decimal place
(Clause 4.4, footnote 1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `values_by_band` | Measured band quantities (`R`, `R'`, `Dn`, `DnT` ...) in dB. 16 values are read as one-third-octave bands, 5 values as octave bands. |
| `bands` | `"third-octave"`, `"octave"` or `None` to infer the band set from the number of values. |

**Returns:** [`WeightedRatingResult`](/phonometry/reference/api/building/ratings/#weightedratingresult) with `rating`, `c`, `ctr` and `unfavourable_sum`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the number of values does not match the band set, or if any value is non-finite. |

## weighted_rating_extended

```python
weighted_rating_extended(
    values_by_band: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray | None = None,
    *,
    one_decimal: bool = False,
) -> ExtendedWeightedRatingResult
```

Weighted rating with enlarged-range adaptation terms (ISO 717-1 An. B).

Computes the weighted rating from the core one-third-octave bands
100-3150 Hz (Clause 4.4) and, for every enlarged frequency range covered
by the input, the additional spectrum adaptation terms of Annex B
(`C50-3150`, `C50-5000`, `C100-5000` and the `Ctr` counterparts)
with the Table B.1 spectra: $C_j = X_{\mathrm{A}j} - X_\mathrm{w}$ where `XAj` sums
over the
bands of the enlarged range (Clause 4.5 with Annex B).

With `one_decimal=True` the reference-curve shift runs in 0,1 dB steps
and every reduction keeps one decimal place; the variant Clauses 4.4/4.5
prescribe "for the expression of uncertainty" and ISO 12999-1:2020
Annex B requires for the uncertainty of single-number values.

**Parameters**

| Name | Description |
| :--- | :--- |
| `values_by_band` | Measured band quantities (`R`, `R'`, `Dn`, `DnT` ...) in dB, one-third-octave bands. |
| `frequencies` | Band centre frequencies, in Hz (one per value). `None` assumes exactly the 16 core bands 100-3150 Hz. The 16 core bands must always be present; extended terms are formed for each Annex B range whose bands are all present. |
| `one_decimal` | Use the 0,1 dB shift and one-decimal reductions. |

**Returns:** An [`ExtendedWeightedRatingResult`](/phonometry/reference/api/building/ratings/#extendedweightedratingresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the input is not one-dimensional and finite, the band counts differ, or the core bands are missing. |

## WeightedRatingResult

```python
WeightedRatingResult(
    rating: int,
    c: int,
    ctr: int,
    unfavourable_sum: float,
    band_centers: np.ndarray | None = None,
    measured: np.ndarray | None = None,
    shifted_reference: np.ndarray | None = None,
    quantity: Literal['airborne'] = 'airborne',
)
```

Single-number weighted rating and adaptation terms (ISO 717-1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `rating` | Weighted rating (`Rw`, `R'w`, `DnT,w` ...), the shifted reference read at 500 Hz, in dB (Clause 4.4). Integer. |
| `c` | Spectrum adaptation term `C` (spectrum No. 1), in dB (Clause 4.5). Integer. |
| `ctr` | Spectrum adaptation term `Ctr` (spectrum No. 2), in dB (Clause 4.5). Integer. |
| `unfavourable_sum` | Sum of unfavourable deviations at the final shift, in dB (Clause 4.4); at most 32,0 (16 bands) or 10,0 (5 bands). |
| `band_centers` | Band centre frequencies of the measured curve, in Hz. Defaults to `None` for backward-compatible construction. |
| `measured` | The measured band quantities used for the rating (after the one-decimal reduction of Clause 4.4), in dB. Defaults to `None`. |
| `shifted_reference` | Table 3 reference curve after the final shift, in dB. Defaults to `None`. |
| `quantity` | Always `"airborne"`: this class carries the ISO 717-1 airborne rating, and the renderers dispatch on this tag when handed the union with [`ImpactRatingResult`](/phonometry/reference/api/building/ratings/#impactratingresult), which carries `"impact"`. The field used to admit both values and promise that `"impact"` would select the impact labels; it never could, since the impact labels read `ci` off the result and this class does not have one, so the promise ended in the renderer's `AttributeError`. |

### WeightedRatingResult.plot()

```python
WeightedRatingResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the measured curve vs the shifted reference (ISO 717-1).

Unfavourable deviations (reference above measurement) are shaded and
`Rw (C; Ctr)` annotated. Requires matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

### WeightedRatingResult.report()

```python
WeightedRatingResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
    symbol: str | None = None,
) -> str
```

Render an ISO 717-1 airborne sound-insulation fiche to a PDF.

Writes a one-page accredited-laboratory report: the standard-basis
line, an optional metadata header block, the band table beside the
measured-versus-shifted-reference plot (the result's own
`plot`), the boxed `Rw (C; Ctr)` result, an optional verdict
row and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a prediction fiche (body, result and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table uses the ISO 717 Annex C columns (frequency, measured value, shifted reference, unfavourable deviation) instead of the two-column `f \| value` table. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |
| `symbol` | The reported single-number quantity, as plain text: `"Rw"` (the default when `None`), `"R'w"`, `"Dn,w"`, `"DnT,w"` ... per ISO 717-1 Tables 1-2, so a field measurement (e.g. a standardized level difference rated to `DnT,w`) is not mislabelled with the laboratory descriptor. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`, `symbol` is not a valid quantity-symbol shape, or the result was built without the per-band data (`band_centers`, `measured`, `shifted_reference`). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |
