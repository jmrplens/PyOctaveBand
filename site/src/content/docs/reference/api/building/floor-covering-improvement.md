---
title: "building.measurement.floor_covering_improvement"
description: "Impact-sound improvement of floor coverings on a small mock-up (ISO 16251-1:2014)."
sidebar:
  label: "floor_covering_improvement"
---

Impact-sound improvement of floor coverings on a small mock-up
(ISO 16251-1:2014).

Laboratory method for the **improvement of impact sound insulation** `ΔL` of a
soft, locally-reacting floor covering (carpet, PVC, linoleum; ISO 10140-1
Annex H category I) laid on a small concrete mock-up plate and excited by a
standard tapping machine. The two rooms of the ISO 10140 series are removed and
the floor is replaced by a softly-supported concrete plate; instead of the
receiving-room sound pressure level, the **structure-borne acceleration level**
on the underside of the plate is measured with and without the covering. For
locally-reacting coverings this acceleration-level difference equals the ISO
10140 impact sound reduction (Clause 4).

**Acceleration level (Formula (1)).**
$L_a = 10 \log_{10}[(1/T_m) \int a(t)^2/a_0^2\,dt]$ dB,
with the reference acceleration $a_0 = 10^{-6}$ m/s², i.e.
$L_a = 10 \log_{10}(\overline{a^2}/a_0^2)$.

**Background correction (Formula (2)).** Each measured level `L'` is
corrected against the background `Lb` per accelerometer position, by the
margin $L' - L_\mathrm{b}$: unchanged for $\ge 15$ dB; energy
subtraction $10 \log_{10}(10^{L'/10} - 10^{L_\mathrm{b}/10})$ for
$6 \le \mathrm{margin} < 15$ dB; and the fixed $L' - 1.3$ dB limit
for $< 6$ dB. Bands hitting the 1.3 dB limit are flagged as the *limit of
measurement* (reported as $> \Delta L$). This differs from the ISO
10140-4 correction ([`phonometry.building.background_correction`](/phonometry/reference/api/building/lab-insulation/#background_correction)) only at
exactly $\mathrm{margin} = 6$ dB.

**Improvement (Formulae (3)/(4)).** The per-position difference is
$\Delta L_{t,a} = L_{0,t,a} - L_{1,t,a}$ (0 = bare plate, 1 = specimen)
and the improvement is
their arithmetic mean over all tapping-machine (t) and accelerometer (a)
positions, $\Delta L = (1/(t \cdot a)) \sum_t \sum_a \Delta L_{t,a}$.

**Octave bands (Formula (5)).**
$\Delta L_\mathrm{oct} = -10 \log_{10}[(1/3) \sum 10^{-\Delta L_j/10}]$ dB from
the three one-third-octave values in each octave.

**Weighted improvement.** `ΔLw` is the ISO 717-2 weighted reduction of impact
sound pressure level (Clause 6.5), computed with the heavyweight reference
floor via [`phonometry.building.weighted_impact_improvement`](/phonometry/reference/api/building/ratings/#weighted_impact_improvement) on the 16
rating bands 100 Hz to 3150 Hz; a wider clause 6.3 spectrum (18 bands 100-5000
Hz, optionally extended to 50 Hz) is rated on that sub-range. The statement of
results (Clause 8 e)) also carries the spectrum adaptation term `CI,Δ` (ISO
717-2:2020 Formula (A.4)) via
[`phonometry.building.impact_improvement_adaptation_term`](/phonometry/reference/api/building/ratings/#impact_improvement_adaptation_term). 

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## acceleration_level

```python
acceleration_level(
    acceleration: float | Sequence[float] | np.ndarray,
    *,
    reference: float = 1e-06,
) -> np.ndarray
```

Vibratory acceleration level `La` (ISO 16251-1 Formula (1)).

$L_a = 10 \log_{10}(a_\mathrm{rms}^2 / a_0^2) = 20 \log_{10}(a_\mathrm{rms} / a_0)$ dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `acceleration` | RMS acceleration `a_rms` per band, in m/s² (> 0). |
| `reference` | Reference acceleration `a0`, in m/s² (default 1e-6). |

**Returns:** The acceleration level `La` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Non-finite or non-positive acceleration, or a non-positive reference. |

## background_corrected_level

```python
background_corrected_level(
    signal_and_background: float | Sequence[float] | np.ndarray,
    background: float | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]
```

Correct a measured level for background noise (ISO 16251-1 F. (2)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal_and_background` | Measured level `L'` (with background) per band, in dB. |
| `background` | Background-noise level `Lb` per band, in dB. |

**Returns:** `(corrected, limited)`: the corrected levels `L` per band, in dB, and a boolean mask of bands at the 1.3 dB limit of measurement (report as $> \Delta L$).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Mismatched shapes or non-finite values. |

## FloorCoveringImprovementResult

```python
FloorCoveringImprovementResult(
    frequencies: np.ndarray,
    improvement: np.ndarray,
    limited: np.ndarray,
    delta_lw: int | None,
    ci_delta: int | None = None,
)
```

Impact-sound improvement of a floor covering (ISO 16251-1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centre frequencies, in Hz. |
| `improvement` | Improvement of impact sound insulation `ΔL` per band, in dB. |
| `limited` | Per-band boolean mask of bands at the 1.3 dB limit of measurement (reported as $> \Delta L$); all `False` when no background correction was applied. |
| `delta_lw` | Weighted improvement `ΔLw` (ISO 717-2), in dB, or `None` when the spectrum does not contain the 16 one-third-octave rating bands 100-3150 Hz. A wider clause 6.3 spectrum (e.g. the 18 bands 100-5000 Hz, optionally extended down to 50 Hz) is rated on its 100-3150 Hz sub-range. |
| `ci_delta` | Spectrum adaptation term `CI,Δ` (ISO 717-2:2020 Formula (A.4); required in the ISO 16251-1 Clause 8 e) statement of results), in dB, or `None` when `delta_lw` is `None`. |

### FloorCoveringImprovementResult.octave_bands()

```python
FloorCoveringImprovementResult.octave_bands() -> tuple[np.ndarray, np.ndarray]
```

Return `(octave_freqs, ΔLoct)` via Formula (5) (needs 16 1/3-oct bands).

### FloorCoveringImprovementResult.plot()

```python
FloorCoveringImprovementResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the improvement spectrum `ΔL` (with `ΔLw` when available).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### FloorCoveringImprovementResult.report()

```python
FloorCoveringImprovementResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 16251-1 impact-improvement test-report fiche to PDF.

Writes a one-page accredited report (BS EN ISO 16251-1:2014, the
small-mock-up laboratory method for the reduction of transmitted impact
sound by soft floor coverings): the standard-basis line, an optional
metadata header block, a two-panel body with the per-band table
(frequency and the reduction of impact sound pressure level `ΔL`,
bands at the 1,3 dB limit of measurement prefixed `>`) beside the
`ΔL(f)` improvement curve, a boxed single-number weighted improvement
`ΔLw (CI,Δ)` (ISO 717-2, the Clause 8 e) statement of results; a
characterisation headline replaces it when the spectrum lacks the 16
rating bands 100-3150 Hz), an optional verdict row and a footer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a body-and-disclaimer fiche whose header shows only the measured frequency range. The applicable descriptive fields are `client`, `manufacturer`, `specimen` (the floor covering under test), `mounting`, `mass_per_area`, `test_room`, `test_date`, `temperature`, `pressure`, `measurement_standard`, `laboratory`, `operator`, `report_id` and `notes`. The bare reference floor is the standardised heavyweight floor of ISO 717-2:2020 Table 4, fixed by the standard. When `requirement` is set the fiche adds a verdict row (a higher weighted improvement is better, so the result passes at or above the requirement). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the per-band table gains the reference-floor-with-covering column $L_\mathrm{n,r} = L_\mathrm{n,r,0} - \Delta L$ (ISO 717-2:2020 Formula (1)), the derivation basis of `ΔLw`, when the spectrum is exactly the 16 rating bands 100-3150 Hz. |
| `language` | Fiche language: `"en"` (default, English, decimal point) or `"es"` (Spanish, decimal comma). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## impact_improvement

```python
impact_improvement(
    bare: float | Sequence[float] | np.ndarray,
    with_covering: float | Sequence[float] | np.ndarray,
    frequencies: float | Sequence[float] | np.ndarray,
    *,
    background: float | Sequence[float] | np.ndarray | None = None,
) -> FloorCoveringImprovementResult
```

Improvement of impact sound insulation `ΔL` of a floor covering.

ISO 16251-1. The acceleration levels without (`L0`) and with (`L1`)
the specimen may be
given either as one value per band (already averaged over the
tapping-machine
and accelerometer positions) or, for the raw measurement, as a
`(positions, bands)` array. The standard's order of operations is
followed:
the background correction (Formula (2)) is applied **per position**, then
the
level difference $\Delta L_{t,a} = L_0 - L_1$ (Formula (3)), then
the arithmetic mean
over positions $\Delta L = \mathrm{mean}(\Delta L_{t,a})$
(Formula (4)), necessary because Formula
(2) is non-linear, so correcting must precede averaging.

**Parameters**

| Name | Description |
| :--- | :--- |
| `bare` | Bare-plate acceleration level `L0`, in dB; `(bands,)` or `(positions, bands)`. |
| `with_covering` | Level with the specimen `L1`, in dB; same shape as `bare`. |
| `frequencies` | One-third-octave band centre frequencies, in Hz (`(bands,)`). |
| `background` | Optional background-noise level `Lb`, in dB; `(bands,)` (applied to every position) or `(positions, bands)`. When given, `L0` and `L1` are background-corrected (Formula (2)) per position before the difference, and bands where any position hits the 1.3 dB limit are flagged. |

**Returns:** [`FloorCoveringImprovementResult`](/phonometry/reference/api/building/floor-covering-improvement/#floorcoveringimprovementresult) (`ΔL` per band).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | Mismatched shapes or non-finite values. |

## improvement_octave_bands

```python
improvement_octave_bands(
    improvement: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]
```

Convert a one-third-octave improvement spectrum to octaves (F. (5)).

$\Delta L_\mathrm{oct} = -10 \log_{10}[(1/3) \sum_{j=1}^{3} 10^{-\Delta L_j/10}]$
dB over the three thirds of
each octave.

**Parameters**

| Name | Description |
| :--- | :--- |
| `improvement` | Improvement `ΔL` per one-third-octave band, in dB. |
| `frequencies` | The matching one-third-octave centre frequencies, in Hz; must contain whole octave triplets (centre / centre·2^(±1/3)). |

**Returns:** `(octave_freqs, ΔLoct)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | The thirds do not group into complete octave triplets. |
