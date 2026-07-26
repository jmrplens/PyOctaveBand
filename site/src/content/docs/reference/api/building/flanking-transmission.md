---
title: "building.flanking_transmission"
description: "Laboratory measurement of flanking sound transmission (ISO 10848:2006/2010)."
sidebar:
  label: "flanking_transmission"
---

Laboratory measurement of flanking sound transmission (ISO 10848:2006/2010).

This is the **measurement** counterpart of the flanking-transmission
*prediction* in [`phonometry.building_prediction`](/phonometry/reference/api/building/building-prediction/). EN 12354-1 predicts the
apparent in-situ performance from, among other inputs, the **vibration
reduction index** `Kij` of each junction; ISO 10848 is the standard that
*measures* that `Kij` (and the overall flanking descriptors `Dn,f` /
`Ln,f`) in a qualified test facility. The measured `Kij` is a
situation-invariant junction descriptor that feeds straight into the
[`phonometry.flanking_path`](/phonometry/reference/api/building/building-prediction/#flanking_path) model.

**Vibration reduction index (Part 1, Clause 3.9).** From the *direction
averaged* velocity level difference `D̄v,ij = ½(Dv,ij + Dv,ji)` (Formula (11))
this module forms, per one-third-octave band,
`Kij = D̄v,ij + 10 lg( lij / √(ai·aj) )` (Formula (13)) with the common-edge
junction length `lij` and the equivalent absorption lengths `ai`, `aj` of
the two elements. For lightweight, well-damped elements the equivalent
absorption length collapses to the element area (`aj = Sj / l0`, `l0 = 1 m`,
Clause 3.8 Note 3) and Formula (13) reduces to Formula (14),
`Kij = D̄v,ij + 10 lg( lij / √(Si·Sj) )`. Because it uses the direction average,
`Kij` is symmetric (`Kij = Kji`).

**Equivalent absorption length (Part 1, Formula (12)).**
`aj = (2,2 · π² · Sj) / (Ts,j · c0) · √(f_ref / f)` with the structural
reverberation time `Ts,j`, the element area `Sj`, the speed of sound in air
`c0` and the reference frequency `f_ref = 1000 Hz`. The related total loss
factor is `η = 2,2 / (f · Ts)` (Clause 7.3.1).

**Overall flanking descriptors (Part 1, Clauses 3.2/3.3).** With airborne
excitation the normalized flanking level difference is
`Dn,f = L1 − L2 − 10 lg(A/A0)` (Formula (4)); with a tapping machine on the
source-room floor the normalized flanking impact level is
`Ln,f = L2 + 10 lg(A/A0)` (Formula (5)), both with the reference absorption
area `A0 = 10 m²`. Their single-number ratings `Dn,f,w (C; Ctr)` and
`Ln,f,w (CI)` follow ISO 717-1/-2 through the verified
[`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating) / [`phonometry.weighted_impact_rating`](/phonometry/reference/api/building/insulation/#weighted_impact_rating)
engines, reused unchanged.

**Validity of Kij.** `Kij` rests on a statistical-energy-analysis
simplification (weak coupling, diffuse vibration fields). This module exposes
the standard's own applicability checks: the strong-coupling inequality
(Formula (15)), and, for heavy junctions (Part 4), the modal density,
in-band mode count and modal overlap factor (Formulas (5), (4), (6)) whose
thresholds bracket or exclude unreliable bands.

**c0 note.** ISO 10848 writes `c0` only as "the speed of sound in air" and
gives no number. This module defaults to `343 m/s` (20 °C) and exposes it as
a parameter so a facility can pin its own value.

**Frequency range (Part 1, Clause 7.5).** The mandatory one-third-octave range
is 100 Hz to 5000 Hz (18 bands). The single-number `Kij` is the arithmetic
mean over 200 Hz to 1250 Hz for one-third-octave bands, or over 125 Hz to
1000 Hz for octave bands (Annex A); the automatic mean is formed only when
the corresponding band set is present in the supplied frequencies. For heavy
junctions (Part 4, Clause 9) bands whose modal overlap factor is below 0,25
are bracketed and excluded from the single-number mean when the per-band
`modal_overlap` is supplied.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## band_mode_count

```python
band_mode_count(
    frequency: Sequence[float] | np.ndarray,
    area: float,
    critical_frequency: float,
    *,
    speed_of_sound: float = 343.0,
) -> np.ndarray
```

In-band mode count `N = B · n` (Part 4, Formula (4)).

With the one-third-octave bandwidth approximation `B = 0,23 · f` and the
modal density `n` from [`modal_density`](/phonometry/reference/api/building/flanking-transmission/#modal_density). `N ≥ 5` modes per band is
"always satisfactory".

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequency `f`, in Hz, per band. |
| `area` | Element area `S`, in m². |
| `critical_frequency` | Critical frequency `fc`, in Hz. |
| `speed_of_sound` | Speed of sound in air `c0`, in m/s. |

**Returns:** In-band mode count `N` per band (dimensionless).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any input is not positive/finite. |

## critical_frequency

```python
critical_frequency(
    longitudinal_wave_speed: float,
    thickness: float,
    *,
    speed_of_sound: float = 343.0,
) -> float
```

Thin-plate critical frequency `fc` (Part 1, Formula (20)).

`fc = c0² / (1,8 · cL · h)` for a homogeneous isotropic element. The
constant 1,8 already carries the `2π/√12` factor of the thin-plate
dispersion relation, so for a plate whose bending stiffness and mass are
mutually consistent this equals [`phonometry.coincidence_frequency`](/phonometry/reference/api/vibration/radiation-efficiency/#coincidence_frequency)
(Hopkins Eq. 2.201) to within the rounding of the constant.

:::note
The printed ISO 10848-1:2006 Formula (20) carries a spurious extra
`π` in the denominator (`1,8 cL · h · π`), which would misplace
`fc` by a factor π; the π-free form implemented here is the one
ISO 10848-1:2017 restores in its Formula (5), ISO 12354-1:2017
prints in its symbol definitions (`fc = c0²/(1,8 cL t)`) and
Hopkins Eq. 2.201 derives. See `docs/ERRATA.md`.
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `longitudinal_wave_speed` | Longitudinal wave speed `cL`, in m/s. |
| `thickness` | Element thickness `h`, in m. |
| `speed_of_sound` | Speed of sound in air `c0`, in m/s. |

**Returns:** Critical frequency `fc`, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any input is not positive/finite. |

## direction_averaged_level_difference

```python
direction_averaged_level_difference(
    dv_ij: Sequence[float] | np.ndarray,
    dv_ji: Sequence[float] | np.ndarray,
) -> np.ndarray
```

Direction-averaged velocity level difference (Formula (11)).

`D̄v,ij = ½ (Dv,ij + Dv,ji)` with `Dv,ij` measured exciting element
`i` and `Dv,ji` exciting element `j`. The average makes the derived
`Kij` symmetric.

**Parameters**

| Name | Description |
| :--- | :--- |
| `dv_ij` | `Dv,ij` (element `i` excited), in dB, per band. |
| `dv_ji` | `Dv,ji` (element `j` excited), in dB, per band. |

**Returns:** `D̄v,ij` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are empty, non-finite, or differ in length. |

## equivalent_absorption_length

```python
equivalent_absorption_length(
    area: float,
    structural_reverberation_time: float | Sequence[float] | np.ndarray,
    frequency: Sequence[float] | np.ndarray,
    *,
    speed_of_sound: float = 343.0,
) -> np.ndarray
```

Equivalent absorption length `aj` (Formula (12)).

`aj = (2,2 · π² · Sj) / (Ts,j · c0) · √(f_ref / f)` with
`f_ref = 1000 Hz`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `area` | Element surface area `Sj`, in m². |
| `structural_reverberation_time` | Structural reverberation time `Ts,j`, in s, per band (or a single value broadcast to all bands). |
| `frequency` | Band centre frequency `f`, in Hz, per band. |
| `speed_of_sound` | Speed of sound in air `c0`, in m/s (default 343 m/s). |

**Returns:** Equivalent absorption length `aj`, in m, per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any input is not positive/finite or the band counts are incompatible. |

## FlankingImpactLevelResult

```python
FlankingImpactLevelResult(
    l_n_f: np.ndarray,
    rating: ImpactRatingResult | None,
)
```

Normalized flanking impact level `Ln,f` (Formula (5)).

**Attributes**

| Name | Description |
| :--- | :--- |
| `l_n_f` | `Ln,f = L2 + 10 lg(A/A0)` per band, in dB. |
| `rating` | Single-number `Ln,f,w` with `CI` (ISO 717-2), or `None` when the band count is neither 16 nor 5. |

### FlankingImpactLevelResult.plot()

```python
FlankingImpactLevelResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `Ln,f` against the shifted ISO 717-2 reference curve.

### FlankingImpactLevelResult.report()

```python
FlankingImpactLevelResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
    part: int = 2,
) -> str
```

Render a normalized flanking impact level `Ln,f` fiche to a PDF.

Writes the one-page impact flanking-transmission report of the
ISO 10848 series: the standard-basis line naming the part governing
the tested construction, an optional metadata header, the per-band
`Ln,f` table beside the measured-versus-shifted-ISO 717-2
reference curve, the boxed single-number `Ln,f,w (CI)`, the
measurement statement, an optional requirement verdict and a footer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the left table shows the ISO 717 evaluation per band (the `Ln,f` value, the shifted reference and the unfavourable deviation) instead of the two-column form. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |
| `part` | The ISO 10848 part governing the tested construction and named in the basis line: `2` (lightweight elements, default, ISO 10848-2:2006), `3` (lightweight elements with a junction of substantial influence, ISO 10848-3:2006) or `4` (heavy, Type A elements, ISO 10848-4:2010). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown, `part` is not 2, 3 or 4, or the result has no single-number rating (the band count is neither 16 nor 5). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## FlankingLevelDifferenceResult

```python
FlankingLevelDifferenceResult(
    d_n_f: np.ndarray,
    rating: WeightedRatingResult | None,
)
```

Normalized flanking level difference `Dn,f` (airborne, Formula (4)).

**Attributes**

| Name | Description |
| :--- | :--- |
| `d_n_f` | `Dn,f = L1 − L2 − 10 lg(A/A0)` per band, in dB. |
| `rating` | Single-number `Dn,f,w` with `C`/`Ctr` (ISO 717-1), or `None` when the band count is neither 16 nor 5. |

### FlankingLevelDifferenceResult.plot()

```python
FlankingLevelDifferenceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `Dn,f` against the shifted ISO 717-1 reference curve.

### FlankingLevelDifferenceResult.report()

```python
FlankingLevelDifferenceResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
    part: int = 2,
) -> str
```

Render a normalized flanking level difference `Dn,f` fiche to a PDF.

Writes the one-page airborne flanking-transmission report of the
ISO 10848 series: the standard-basis line naming the part governing
the tested construction, an optional metadata header, the per-band
`Dn,f` table beside the measured-versus-shifted-ISO 717-1
reference curve, the boxed single-number `Dn,f,w (C; Ctr)`, the
measurement statement, an optional requirement verdict and a footer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the left table shows the ISO 717 evaluation per band (the `Dn,f` value, the shifted reference and the unfavourable deviation) instead of the two-column form. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |
| `part` | The ISO 10848 part governing the tested construction and named in the basis line: `2` (lightweight elements, default, ISO 10848-2:2006), `3` (lightweight elements with a junction of substantial influence, ISO 10848-3:2006) or `4` (heavy, Type A elements, ISO 10848-4:2010). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown, `part` is not 2, 3 or 4, or the result has no single-number rating (the band count is neither 16 nor 5). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## modal_density

```python
modal_density(
    area: float,
    critical_frequency: float,
    *,
    speed_of_sound: float = 343.0,
) -> float
```

Modal density `n = π · S · fc / c0²` (Part 4, Formula (5)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `area` | Element area `S`, in m². |
| `critical_frequency` | Critical frequency `fc`, in Hz. |
| `speed_of_sound` | Speed of sound in air `c0`, in m/s. |

**Returns:** Modal density `n`, in modes per Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any input is not positive/finite. |

## modal_overlap_factor

```python
modal_overlap_factor(
    area: float,
    critical_frequency: float,
    structural_reverberation_time: float | Sequence[float] | np.ndarray,
    *,
    speed_of_sound: float = 343.0,
) -> np.ndarray
```

Modal overlap factor `M = 2,2 · n / Ts` (Part 4, Formula (6)).

With the modal density `n` from [`modal_density`](/phonometry/reference/api/building/flanking-transmission/#modal_density). Part 4 prefers
`M ≥ 1` at 250 Hz and above, and requires bands with `M < 0,25` to be
bracketed in the report and excluded from the single-number rating
(Clause 9). This function only computes `M`; pass it to
[`vibration_reduction_index`](/phonometry/reference/api/building/flanking-transmission/#vibration_reduction_index) via `modal_overlap` to apply the
bracketing and single-number exclusion.

**Parameters**

| Name | Description |
| :--- | :--- |
| `area` | Element area `S`, in m². |
| `critical_frequency` | Critical frequency `fc`, in Hz. |
| `structural_reverberation_time` | `Ts`, in s, per band (or a single value). |
| `speed_of_sound` | Speed of sound in air `c0`, in m/s. |

**Returns:** Modal overlap factor `M` per band (dimensionless).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any input is not positive/finite. |

## normalized_flanking_impact_level

```python
normalized_flanking_impact_level(
    receive_level: Sequence[float] | np.ndarray,
    absorption_area: Sequence[float] | np.ndarray,
    *,
    reference_area: float = 10.0,
    bands: str | None = None,
) -> FlankingImpactLevelResult
```

Normalized flanking impact level `Ln,f` (Formula (5)).

`Ln,f = L2 + 10 lg(A/A0)` with the reference absorption area
`A0 = 10 m²`, from the receiving-room impact level with the tapping
machine on the source-room floor.

**Parameters**

| Name | Description |
| :--- | :--- |
| `receive_level` | Receiving-room average impact SPL `L2` per band, in dB. |
| `absorption_area` | Receiving-room equivalent absorption area `A` per band, in m². |
| `reference_area` | `A0`, in m² (default 10 m²). |
| `bands` | Band spacing passed to [`phonometry.weighted_impact_rating`](/phonometry/reference/api/building/insulation/#weighted_impact_rating); auto-detected when `None`. |

**Returns:** A [`FlankingImpactLevelResult`](/phonometry/reference/api/building/flanking-transmission/#flankingimpactlevelresult); the single-number rating is formed only for 16 (one-third-octave) or 5 (octave) bands.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On incompatible band counts or non-positive areas. |

## normalized_flanking_level_difference

```python
normalized_flanking_level_difference(
    source_level: Sequence[float] | np.ndarray,
    receive_level: Sequence[float] | np.ndarray,
    absorption_area: Sequence[float] | np.ndarray,
    *,
    reference_area: float = 10.0,
    bands: str | None = None,
) -> FlankingLevelDifferenceResult
```

Normalized flanking level difference `Dn,f` (airborne, Formula (4)).

`Dn,f = L1 − L2 − 10 lg(A/A0)` with the reference absorption area
`A0 = 10 m²`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_level` | Source-room average SPL `L1` per band, in dB. |
| `receive_level` | Receiving-room average SPL `L2` per band, in dB. |
| `absorption_area` | Receiving-room equivalent absorption area `A` per band, in m². |
| `reference_area` | `A0`, in m² (default 10 m²). |
| `bands` | Band spacing passed to [`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating); auto-detected when `None`. |

**Returns:** A [`FlankingLevelDifferenceResult`](/phonometry/reference/api/building/flanking-transmission/#flankingleveldifferenceresult); the single-number rating is formed only for 16 (one-third-octave) or 5 (octave) bands.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On incompatible band counts or non-positive areas. |

## strong_coupling_satisfied

```python
strong_coupling_satisfied(
    velocity_level_difference: Sequence[float] | np.ndarray,
    mass_i: float,
    mass_j: float,
    critical_frequency_i: float,
    critical_frequency_j: float,
) -> np.ndarray
```

Strong-coupling applicability check (Part 1, Formula (15)).

`Kij` is relevant only where
`D̄v,ij ≥ 3 − 10 lg( (mi·fcj)/(mj·fci) )`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `velocity_level_difference` | Direction-averaged `D̄v,ij` per band, in dB. |
| `mass_i` | Mass per unit area `mi` of element `i`, in kg/m². |
| `mass_j` | Mass per unit area `mj` of element `j`, in kg/m². |
| `critical_frequency_i` | Critical frequency `fci` of element `i`, in Hz. |
| `critical_frequency_j` | Critical frequency `fcj` of element `j`, in Hz. |

**Returns:** Boolean array, `True` where the inequality holds.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any scalar input is not positive/finite. |

## total_loss_factor

```python
total_loss_factor(
    frequency: Sequence[float] | np.ndarray,
    structural_reverberation_time: float | Sequence[float] | np.ndarray,
) -> np.ndarray
```

Total loss factor `η = 2,2 / (f · Ts)` (Clause 7.3.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequency `f`, in Hz, per band. |
| `structural_reverberation_time` | Structural reverberation time `Ts`, in s, per band (or a single value broadcast to all bands). |

**Returns:** Total loss factor `η` (dimensionless) per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are not positive/finite or their band counts are incompatible. |

## velocity_level_difference

```python
velocity_level_difference(
    source_level: Sequence[float] | np.ndarray,
    receive_level: Sequence[float] | np.ndarray,
) -> np.ndarray
```

Velocity level difference `Dv,ij = Lv,i − Lv,j` (Formula (8)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_level` | Average velocity level `Lv,i` of the excited element, in dB, per band. |
| `receive_level` | Average velocity level `Lv,j` of the receiving element, in dB, per band. |

**Returns:** `Dv,ij` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are empty, non-finite, or differ in length. |

## vibration_reduction_index

```python
vibration_reduction_index(
    velocity_level_difference: Sequence[float] | np.ndarray,
    junction_length: float,
    area_i: float,
    area_j: float,
    *,
    frequency: Sequence[float] | np.ndarray | None = None,
    structural_reverberation_time_i: float | Sequence[float] | np.ndarray | None = None,
    structural_reverberation_time_j: float | Sequence[float] | np.ndarray | None = None,
    speed_of_sound: float = 343.0,
    modal_overlap: Sequence[float] | np.ndarray | None = None,
) -> VibrationReductionResult
```

Vibration reduction index `Kij` (Formula (13), or simplified (14)).

`Kij = D̄v,ij + 10 lg( lij / √(ai·aj) )`. When the structural reverberation
times and the frequencies are supplied, the equivalent absorption lengths
`ai`, `aj` come from Formula (12) and the full Formula (13) is used.
Otherwise the lightweight, well-damped simplification `aj = Sj / l0`
(`l0 = 1 m`) applies and Formula (14),
`Kij = D̄v,ij + 10 lg( lij / √(Si·Sj) )`, is used.

**Parameters**

| Name | Description |
| :--- | :--- |
| `velocity_level_difference` | Direction-averaged velocity level difference `D̄v,ij` (see [`direction_averaged_level_difference`](/phonometry/reference/api/building/flanking-transmission/#direction_averaged_level_difference)), in dB, per band. |
| `junction_length` | Common-edge junction length `lij`, in m. |
| `area_i` | Area `Si` of element `i`, in m². |
| `area_j` | Area `Sj` of element `j`, in m². |
| `frequency` | Band centre frequencies, in Hz. Required for Formula (12) and for the single-number mean; optional for the simplified form. |
| `structural_reverberation_time_i` | `Ts,i` per band (or a single value), in s. Supply together with `structural_reverberation_time_j` and `frequency` to use Formula (12); omit for the simplified form. |
| `structural_reverberation_time_j` | `Ts,j` per band (or a single value), in s. |
| `speed_of_sound` | Speed of sound in air `c0`, in m/s. |
| `modal_overlap` | Modal overlap factor `M` per band for the heavier (least-overlapped) of the two elements (see [`modal_overlap_factor`](/phonometry/reference/api/building/flanking-transmission/#modal_overlap_factor)). When supplied, bands with `M < 0,25` are flagged as bracketed and excluded from the single-number `K̄ij` (ISO 10848-4:2010, Clause 9). |

**Returns:** A [`VibrationReductionResult`](/phonometry/reference/api/building/flanking-transmission/#vibrationreductionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On incompatible band counts, non-positive geometry, or if only one of the two structural reverberation times is supplied. |

## vibration_reduction_index_from_flanking

```python
vibration_reduction_index_from_flanking(
    normalized_flanking_level_difference: Sequence[float] | np.ndarray,
    reduction_index_i: Sequence[float] | np.ndarray,
    reduction_index_j: Sequence[float] | np.ndarray,
    junction_length: float,
    area_i: float,
    area_j: float,
    absorption_length_i: Sequence[float] | np.ndarray,
    absorption_length_j: Sequence[float] | np.ndarray,
    *,
    reference_area: float = 10.0,
) -> np.ndarray
```

Indirect `Kij` from the normalized flanking level difference.

ISO 10848-1:2006, Clause 4.3.1 Note 2 (unnumbered):

```text
Kij = Dn,f − (Ri + Rj)/2 − 10 lg(√(ai·aj)/lij) + 10 lg(√(Si·Sj)/A0)
```

The standard warns this holds only for resonant-only transmission; measured
`R` also includes forced transmission, so a direct measurement of `Kij`
(Formula (13)) is preferred. Provided for completeness.

**Parameters**

| Name | Description |
| :--- | :--- |
| `normalized_flanking_level_difference` | `Dn,f` per band, in dB. |
| `reduction_index_i` | Sound reduction index `Ri` per band, in dB. |
| `reduction_index_j` | Sound reduction index `Rj` per band, in dB. |
| `junction_length` | `lij`, in m. |
| `area_i` | `Si`, in m². |
| `area_j` | `Sj`, in m². |
| `absorption_length_i` | `ai` per band, in m. |
| `absorption_length_j` | `aj` per band, in m. |
| `reference_area` | `A0`, in m² (default 10 m²). |

**Returns:** `Kij` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On incompatible band counts or non-positive geometry. |

## VibrationReductionResult

```python
VibrationReductionResult(
    frequencies: np.ndarray | None,
    k_ij: np.ndarray,
    single_number: float | None,
    bracketed: np.ndarray | None = None,
)
```

Per-band vibration reduction index `Kij` (ISO 10848-1:2006).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz, or `None` when they were not supplied. |
| `k_ij` | Vibration reduction index `Kij` per band, in dB (Formula (13) or the simplified Formula (14)). |
| `single_number` | Arithmetic-mean single-number `K̄ij` over 200 Hz to 1250 Hz (one-third octave) or 125 Hz to 1000 Hz (octave) per Annex A, in dB, or `None` when the frequencies do not cover the corresponding band set. Bands bracketed for poor modal overlap (ISO 10848-4:2010 Clause 9) are excluded from the mean. |
| `bracketed` | Per-band boolean flags, `True` where the modal overlap factor is below 0,25 so the band is bracketed and excluded from the single-number rating (ISO 10848-4:2010 Clause 9), or `None` when no modal overlap was supplied. |

### VibrationReductionResult.octave_bands()

```python
VibrationReductionResult.octave_bands() -> VibrationReductionResult
```

Combine one-third-octave `Kij` into octave bands.

`Kij,oct = −10 lg[ (1/3) Σ 10^(−Kij/10) ]` over each group of three
one-third-octave bands (Part 2/3/4). Requires a band count that is a
multiple of three and, for the frequency labels, that frequencies were
supplied; supplied frequencies must group into whole octave triples
(lower/centre/upper one-third around an octave centre). The octave
single-number `K̄ij` is averaged over 125 Hz to 1000 Hz (Annex A).
An octave band is bracketed when any of its one-third-octave bands is.

**Returns:** A new [`VibrationReductionResult`](/phonometry/reference/api/building/flanking-transmission/#vibrationreductionresult) on octave centres.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the band count is not a multiple of three, or the frequencies do not open on complete octave triples. |

### VibrationReductionResult.plot()

```python
VibrationReductionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `Kij` against frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### VibrationReductionResult.report()

```python
VibrationReductionResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a vibration reduction index `Kij` junction fiche to a PDF.

Writes the one-page ISO 10848-1:2006 junction-characterization report:
the standard-basis line, an optional metadata header, the per-band
`Kij` table beside the `Kij(f)` curve, the boxed single-number mean
`Kij` (Annex A band range) with the count of averaged and bracketed
bands, the measurement statement and a footer. Bands bracketed for poor
modal overlap (ISO 10848-4:2010 Clause 9) are printed in brackets and
excluded from the single number.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, result and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the per-band table adds a column stating whether each band enters the single-number mean. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown or the result carries no band centre frequencies. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |
