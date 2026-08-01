---
title: "building.lab_insulation"
description: "Laboratory sound insulation of building elements (ISO 10140)."
sidebar:
  label: "lab_insulation"
---

Laboratory sound insulation of building elements (ISO 10140).

This is the **laboratory** counterpart of the field ISO 16283 family in
[`phonometry.building.insulation`](/phonometry/reference/api/building/insulation/). In a qualified test facility flanking
transmission is suppressed, so the *direct* airborne sound reduction index
`R` (not the apparent `R'`) is the primary quantity, and the receiving
room's equivalent absorption area `A` is a property of the known facility.

**Airborne sound reduction index (ISO 10140-2:2010).** From the
energy-average sound pressure levels in the source room `L1` and receiving
room `L2` this module forms, per one-third-octave band,
$R = L_1 - L_2 + 10 \log_{10}(S/A)$ (Clause 3.1, Formula (2)) with the free
test
opening area `S` and the Sabine equivalent absorption area
$A = 0.16\,V/T$ (ISO 10140-4:2010, Clause 4.6.3, Formula (5)). The
single-number weighted rating `Rw` and the adaptation terms `C` / `Ctr`
follow ISO 717-1 (Clause 5.3) through the verified
[`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating) engine, reused unchanged.

**Impact sound pressure level (ISO 10140-3:2010).** With the standard
tapping machine exciting the floor under test this module forms, from the
energy-average impact sound pressure level `Li` in the receiving room, the
normalized impact sound pressure level $L_n = L_i + 10 \log_{10}(A/A_0)$
(Clause 3.2,
Formula (1)) with $A = 0.16\,V/T$ and the reference absorption area
$A_0 = 10$ m². The single-number weighted rating `Ln,w` and the term
`CI` follow ISO 717-2 (Clause 5.3) through
[`phonometry.weighted_impact_rating`](/phonometry/reference/api/building/insulation/#weighted_impact_rating), reused unchanged.

**Background-noise correction (ISO 10140-4:2010, Clause 4.3, Formula (4)).**
The receiving-room levels must be corrected for background noise before the
insulation is formed. [`background_correction`](/phonometry/reference/api/building/lab-insulation/#background_correction) implements the correction
$L = 10 \log_{10}(10^{L_{sb}/10} - 10^{L_b/10})$ for a signal-to-background
margin
between 6 dB and 15 dB, the fixed 1.3 dB correction (limit of measurement)
for a margin of 6 dB or less, and no correction for a margin of 15 dB or
more. The 6/15 dB criteria are the laboratory analogue of the 6/10 dB
criteria of ISO 16283-1 Clause 9.2; both cap the correction at 1.3 dB.

**Frequency range (ISO 10140-4:2010, Clause 4.1).** Quantities are measured
over the mandatory one-third-octave range 100 Hz to 5000 Hz (optionally down
to 50 Hz). The single-number rating uses the core 100 Hz to 3150 Hz (16
one-third-octave bands) / 125 Hz to 2000 Hz (5 octave bands) range of
ISO 717-1/2, so the automatic rating is formed only when exactly 16 or 5
per-band values are supplied.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## background_correction

```python
background_correction(
    signal_and_background: Sequence[float] | np.ndarray,
    background: Sequence[float] | np.ndarray,
) -> np.ndarray
```

Correct receiving-room levels for background noise (ISO 10140-4:2010).

Applies the correction of Clause 4.3 per band from the combined
signal-plus-background level `Lsb` and the background level `Lb`,
using the margin `Lsb - Lb`:

- $\mathrm{margin} \ge 15$ dB: the background is negligible and
  the level is
  returned unchanged (Clause 4.3, quality requirement).
- $6\,\mathrm{dB} < \mathrm{margin} < 15\,\mathrm{dB}$: the level
  is corrected with Formula (4),
  $L = 10 \log_{10}(10^{L_{sb}/10} - 10^{L_b/10})$.
- $\mathrm{margin} \le 6$ dB: the fixed 1.3 dB correction is
  applied
  ($L = L_{sb} - 1.3$); such bands are the *limit of measurement*
  and a
  [`LabInsulationWarning`](/phonometry/reference/api/building/lab-insulation/#labinsulationwarning) is emitted (Clause 4.3). A *negative*
  margin ($L_b > L_{sb}$, i.e. background above the measured
  signal) falls
  in this branch and is likewise capped at $L_{sb} - 1.3$: the
  band is
  simply flagged as the limit of measurement rather than yielding a
  nonsensical (or `NaN`) corrected level.

This is the sound-insulation counterpart of
[`phonometry.background_noise_correction`](/phonometry/reference/api/power/sound-power/#background_noise_correction) (ISO 3744:2010): both apply
the same energy subtraction
$10 \log_{10}(10^{L_{sb}/10} - 10^{L_b/10})$, but that
routine returns the correction *offset* `K1` (to subtract from `Lsb`),
whereas this one returns the already-corrected levels `L` directly.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal_and_background` | Combined signal-plus-background levels `Lsb` per band, in dB. |
| `background` | Background-noise levels `Lb` per band, in dB. |

**Returns:** The background-corrected levels per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the shapes differ or contain non-finite values. |

## lab_airborne_insulation

```python
lab_airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    area: float,
    volume: float,
) -> LabAirborneInsulationResult
```

Laboratory airborne sound reduction index per ISO 10140-2:2010.

Computes, per frequency band, the sound reduction index
$R = L_1 - L_2 + 10 \log_{10}(S/A)$ (Clause 3.1, Formula (2)) with the
free test
opening area `S` and the Sabine equivalent absorption area
$A = 0.16\,V/T$ (ISO 10140-4:2010, Formula (5)). When exactly 16
one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz) values are
supplied, the single-number weighted rating `Rw` with `C` / `Ctr`
is also formed via [`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating) (ISO 717-1).

`l1` and `l2` may be one value per band (already energy-averaged) or a
two-dimensional `(positions, bands)` array, in which case the positions
are energy-averaged (ISO 10140-4:2010, Formula (2)). The band levels are
assumed already corrected for background noise (see
[`background_correction`](/phonometry/reference/api/building/lab-insulation/#background_correction)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `l1` | Source-room sound pressure levels, in dB. |
| `l2` | Receiving-room sound pressure levels, in dB. |
| `t2` | Receiving-room reverberation time per band, in seconds. |
| `area` | Area `S` of the free test opening, in m². |
| `volume` | Receiving-room volume `V`, in m³. |

**Returns:** [`LabAirborneInsulationResult`](/phonometry/reference/api/building/lab-insulation/#labairborneinsulationresult) with `r`, `absorption` and `rating`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the band counts of `l1`, `l2` and `t2` differ, if `area`/`volume`/`t2` are not positive, or if inputs are non-finite. |

## lab_impact_insulation

```python
lab_impact_insulation(
    li: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    volume: float,
) -> LabImpactInsulationResult
```

Laboratory impact sound pressure level per ISO 10140-3:2010.

Computes, per frequency band, the normalized impact sound pressure level
$L_n = L_i + 10 \log_{10}(A/A_0)$ (Clause 3.2, Formula (1)) with the
Sabine
equivalent absorption area $A = 0.16\,V/T$ (ISO 10140-4:2010,
Formula (5)) and the reference absorption area $A_0 = 10$ m².
When exactly
16 one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz) values are
supplied, the single-number weighted rating `Ln,w` with `CI` is also
formed via [`phonometry.weighted_impact_rating`](/phonometry/reference/api/building/insulation/#weighted_impact_rating) (ISO 717-2).

`li` may be one value per band (already energy-averaged) or a
two-dimensional `(positions, bands)` array, in which case the positions
are energy-averaged (ISO 10140-4:2010, Formula (2)). The band levels are
assumed already corrected for background noise (see
[`background_correction`](/phonometry/reference/api/building/lab-insulation/#background_correction)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `li` | Energy-average impact sound pressure levels, in dB. |
| `t2` | Receiving-room reverberation time per band, in seconds. |
| `volume` | Receiving-room volume `V`, in m³. |

**Returns:** [`LabImpactInsulationResult`](/phonometry/reference/api/building/lab-insulation/#labimpactinsulationresult) with `l_n`, `absorption` and `rating`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the band counts of `li` and `t2` differ, if `volume`/`t2` are not positive, or if inputs are non-finite. |

## LabAirborneInsulationResult

```python
LabAirborneInsulationResult(
    r: np.ndarray,
    absorption: np.ndarray,
    rating: WeightedRatingResult | None,
)
```

Per-band laboratory airborne sound insulation (ISO 10140-2:2010).

**Attributes**

| Name | Description |
| :--- | :--- |
| `r` | Sound reduction index $R = L_1 - L_2 + 10 \log_{10}(S/A)$ per band, in dB (Clause 3.1, Formula (2)). |
| `absorption` | Equivalent sound absorption area $A = 0.16\,V/T$ per band, in m² (ISO 10140-4:2010, Formula (5)). |
| `rating` | Single-number weighted rating `Rw` with `C` / `Ctr` (ISO 717-1), or `None` when the number of bands is neither 16 (one-third octave) nor 5 (octave) and no rating can be formed. |

### LabAirborneInsulationResult.plot()

```python
LabAirborneInsulationResult.plot(
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes
```

Plot `R` against the shifted ISO 717-1 reference curve.

Delegates to the weighted-rating plot (measured `R` versus the
shifted reference, unfavourable deviations shaded). Requires the
automatic rating to be available (16 or 5 bands) and matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

### LabAirborneInsulationResult.report()

```python
LabAirborneInsulationResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 10140-2 laboratory airborne-insulation report to a PDF.

Writes the one-page laboratory test report of ISO 10140-2:2010: the
standard-basis line, an optional metadata header block (client,
specimen, mounting, room volumes, climatic conditions ...), the band
table (16 one-third-octave or 5 octave bands) beside the
measured-versus-shifted-reference curve, the boxed laboratory rating
`Rw (C; Ctr)` (evaluated per ISO 717-1), the laboratory-method
statement, an optional verdict row and a footer with the identity
block and disclaimer. The report requires the single-number `rating`
to be present on the result; it is formed automatically only for
exactly 16 one-third-octave (100 Hz to 3150 Hz) or 5 octave (125 Hz to
2000 Hz) bands, and a result carrying no rating (any other band count)
is rejected.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating, statement and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table annexes the per-band equivalent sound absorption area `A` beside the reported `R`. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown, `language` is not one of the supported values, or the result carries no single-number rating (its band count is neither 16 one-third-octave nor 5 octave, so the ISO 717-1 rating the fiche needs was not formed). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded rating figure (`pip install phonometry[plot]`). |

## LabImpactInsulationResult

```python
LabImpactInsulationResult(
    l_n: np.ndarray,
    absorption: np.ndarray,
    rating: ImpactRatingResult | None,
)
```

Per-band laboratory impact sound insulation (ISO 10140-3:2010).

**Attributes**

| Name | Description |
| :--- | :--- |
| `l_n` | Normalized impact sound pressure level $L_n = L_i + 10 \log_{10}(A/A_0)$ per band, in dB (Clause 3.2, Formula (1)). |
| `absorption` | Equivalent sound absorption area $A = 0.16\,V/T$ per band, in m² (ISO 10140-4:2010, Formula (5)). |
| `rating` | Single-number weighted rating `Ln,w` with `CI` (ISO 717-2), or `None` when the number of bands is neither 16 (one-third octave) nor 5 (octave) and no rating can be formed. |

### LabImpactInsulationResult.plot()

```python
LabImpactInsulationResult.plot(
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes
```

Plot `Ln` against the shifted ISO 717-2 reference curve.

Delegates to the weighted impact-rating plot. Requires the automatic
rating to be available (16 or 5 bands) and matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

### LabImpactInsulationResult.report()

```python
LabImpactInsulationResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 10140-3 laboratory impact-insulation report to a PDF.

Writes the one-page laboratory test report of ISO 10140-3:2010: the
standard-basis line, an optional metadata header block, the band table
(16 one-third-octave or 5 octave bands) beside the
measured-versus-shifted-reference curve, the boxed laboratory rating
`Ln,w (CI)` (evaluated per ISO 717-2), the laboratory-method
statement, an optional verdict row and a footer with the identity
block and disclaimer. The report requires the single-number `rating`
to be present on the result; it is formed automatically only for
exactly 16 one-third-octave (100 Hz to 3150 Hz) or 5 octave (125 Hz to
2000 Hz) bands, and a result carrying no rating (any other band count)
is rejected.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating, statement and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table annexes the per-band equivalent sound absorption area `A` beside the reported `Ln`. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown, `language` is not one of the supported values, or the result carries no single-number rating (its band count is neither 16 one-third-octave nor 5 octave, so the ISO 717-2 rating the fiche needs was not formed). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded rating figure (`pip install phonometry[plot]`). |

## LabInsulationWarning

Warning for laboratory-insulation limit-of-measurement conditions.
