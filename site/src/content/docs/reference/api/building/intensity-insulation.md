---
title: "building.measurement.intensity_insulation"
description: "Sound insulation measured with sound intensity (ISO 15186)."
sidebar:
  label: "intensity_insulation"
---

Sound insulation measured with sound intensity (ISO 15186).

This is the sound-**intensity** counterpart of the sound-pressure methods in
[`phonometry.building.measurement.lab_insulation`](/phonometry/reference/api/building/lab-insulation/) (ISO 10140) and [`phonometry.building.measurement.insulation`](/phonometry/reference/api/building/insulation/)
(ISO 16283). Instead of an equivalent absorption area in the receiving room,
the transmitted sound power is measured directly by scanning an intensity
probe over a measurement surface enclosing the specimen. The main use is when
the traditional pressure method fails because of high flanking transmission
(ISO 15186-1:2000, Clause 1): the intensity method only captures the power
radiated by the element itself.

**Intensity sound reduction index (ISO 15186-1:2000, Clause 3.8, Formula
(7)).** From the average source-room sound pressure level `Lp1` and the
average normal sound intensity level `LIn` over the measurement surface,
in dB,

$$
R_\mathrm{I} = L_{p1} - 6 - \left[ L_{I\mathrm{n}} + 10 \log_{10}\frac{S_\mathrm{m}}{S} \right]
$$

with the measurement-surface area `Sm` and the specimen area `S`. The
constant `6` dB is the diffuse-field relationship between the sound pressure
level and the sound intensity level incident on the specimen. The same formula
yields the *apparent* index `R'I` in the field (ISO 15186-2), the only
difference being the measurement condition (flanking is not suppressed), not
the arithmetic.

**Modified intensity sound reduction index (Clause 3.10, Formula (9)).**
$R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}$ corrects `RI` so that it reproduces the
ISO 140-3 (now ISO 10140-2) pressure result, which slightly overestimates
`R` because the power radiated into the receiving room is underestimated.
The adaptation term `Kc` (Annex B) is
$10 \log_{10}(1 + S_{\mathrm{b}2} \lambda / (8 V_2))$ (Formula (B.1)) for a
well-defined receiving room of boundary area `Sb2` and volume `V2`, or
the room-independent approximation $10 \log_{10}(1 + 61.4 / f)$
(Formula (B.2)); both use the speed of sound $c = 340$ m/s so that
(B.1) with the reference room $S_{\mathrm{b}2} = 117$ m², $V_2 = 81$ m³
reduces to (B.2).

**Intensity element normalized level difference (Clause 3.9, Formula (8)).**
For small building elements, in dB,

$$
D_\mathrm{I,n,e} = L_{p1} - 6 - \left( L_{I\mathrm{n}} + 10 \log_{10}\frac{S_\mathrm{m}}{A_0} \right) + 10 \log_{10} N
$$

with the reference absorption area $A_0 = 10$ m² and the number `N`
of element units in the measurement surface. The printed Formula (8)
subtracts its $10 \log_{10} N$ term instead of adding it, which is physically
inconsistent with ISO 10140-2:2010 Formula (6) and ISO 15186-2:2010
Formula (12); the corrected per-unit form is implemented (see
`docs/ERRATA.md`).

**Surface pressure-intensity indicator (Clause 3.6 / 6.4.2, Formula (10)).**
$F_{pI} = L_p - L_{I\mathrm{n}}$ qualifies the measurement surface: it must not
exceed 10 dB for a sound-reflecting specimen (6 dB when the receiving side is
sound absorbing), and the probe's pressure-residual intensity index must
exceed $F_{pI} + 10$ dB (Clause 4.1) for the dynamic capability to be
adequate.

**Frequency range (part 1, Clause 6.6).** The part 1 and part 2 quantities are
measured over the mandatory one-third-octave range 100 Hz to 5000 Hz (18
bands), optionally extended down to 50 Hz. The part 3 quantities at the end of
this module answer over 50 Hz to 160 Hz instead (its Clause 1.1), and results
from the two ranges are meant to be combined into one 50 Hz to 5000 Hz curve.
 The single-number weighted rating uses the ISO 717-1 core range, so
the automatic rating (`RI,w`, `RI,M,w`, `DI,n,e,w`) is formed via the
verified [`phonometry.building.weighted_rating`](/phonometry/reference/api/building/ratings/#weighted_rating) engine only when exactly
16 one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz) values are
supplied.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## adaptation_term_kc

```python
adaptation_term_kc(
    freq: Sequence[float] | np.ndarray,
    *,
    boundary_area: float,
    volume: float,
) -> np.ndarray

adaptation_term_kc(freq: Sequence[float] | np.ndarray) -> np.ndarray
```

Adaptation term `Kc` per ISO 15186-1:2000, Annex B.

Returns, per one-third-octave midband frequency, the term `Kc` that
turns the intensity sound reduction index `RI` into the modified index
$R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}$ (Clause 3.10). Two forms are available:

- **Well-defined receiving room (Formula (B.1)):** when both
  `boundary_area` (`Sb2`) and `volume` (`V2`) are supplied,
  $K_\mathrm{c} = 10 \log_{10}(1 + S_{\mathrm{b}2} \lambda / (8 V_2))$ with the midband
  wavelength $\lambda = c / f$ and $c = 340$ m/s.
- **Room-independent approximation (Formula (B.2)):** when neither is
  supplied, $K_\mathrm{c} = 10 \log_{10}(1 + 61.4 / f)$, the exact reduction of
  (B.1) for the reference room $S_{\mathrm{b}2} = 117$ m²,
  $V_2 = 81$ m³.

**Parameters**

| Name | Description |
| :--- | :--- |
| `freq` | One-third-octave midband frequencies, in Hz. |
| `boundary_area` | Total boundary-surface area `Sb2` of the receiving room, in m². Supply together with `volume` for (B.1). |
| `volume` | Receiving-room volume `V2`, in m³. |

**Returns:** The adaptation term `Kc` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `freq` is not positive/finite, if only one of `boundary_area` / `volume` is supplied, or if either is not positive. |

## combine_subareas

```python
combine_subareas(
    l_in: Sequence[Sequence[float]] | np.ndarray,
    measurement_area: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, float]
```

Combine per-subarea intensity levels (ISO 15186-1, Formulas (11)-(12)).

When the measurement surface is divided into subareas `Smi` each scanned
individually, the normal sound intensity level over the whole surface is
the area-weighted energy average, in dB,

$$
L_{I\mathrm{n}} = 10 \log_{10}\!\left[ \frac{1}{S_\mathrm{m}} \sum_i S_{\mathrm{m}i}\, 10^{0.1 L_{I\mathrm{n}i}} \right]
$$

with the total measured area $S_\mathrm{m} = \sum_i |S_{\mathrm{m}i}|$
(Formula (12)).

**Negative-direction subareas (Clause 6.4.6).** When the sound intensity
of a subarea has a negative direction (net energy flowing back towards
the specimen), the standard requires a minus sign before that `Smi` in
Formula (11). Express this by passing the subarea's area as a *negative*
number: its energy is subtracted in the numerator while `Sm` keeps the
unsigned area sum.

**Parameters**

| Name | Description |
| :--- | :--- |
| `l_in` | Per-subarea intensity levels as a `(subareas, bands)` array (one row per subarea), in dB (magnitude of the intensity). |
| `measurement_area` | Subarea areas `Smi`, in m² (one per row). Negative values mark reverse-flow subareas per Clause 6.4.6; zero is invalid. |

**Returns:** A tuple `(LIn, Sm)` with the combined level per band, in dB, and the total measured area $S_\mathrm{m} = \sum \lvert S_{\mathrm{m}i} \rvert$, in m².

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the shapes are inconsistent or values non-finite, if any subarea area is zero, or if the signed energy sum of Formula (11) is not positive in some band (the reverse flows cancel or exceed the forward flow, so no level exists). |

## intensity_element_normalized_difference

```python
intensity_element_normalized_difference(
    lp1: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
    *,
    measurement_area: float,
    n: int = 1,
) -> IntensityElementNormalizedResult
```

Intensity element normalized level difference per ISO 15186-1 (Formula (8)).

Computes, per frequency band, the intensity element normalized level
difference of a single element unit, in dB,

$$
D_\mathrm{I,n,e} = L_{p1} - 6 - \left( L_{I\mathrm{n}} + 10 \log_{10}\frac{S_\mathrm{m}}{A_0} \right) + 10 \log_{10} N
$$

from the average source-room sound pressure level `Lp1`, the average
normal sound intensity level `LIn` over the measurement surface of area
`Sm` (`measurement_area`), the reference absorption area
$A_0 = 10$ m² and the number `N` of element units installed within
the surface. The weighted rating `DI,n,e,w` is computed via
[`phonometry.building.weighted_rating`](/phonometry/reference/api/building/ratings/#weighted_rating) (ISO 717-1) when exactly 16 or
5 values are supplied.

:::note
The printed Formula (8) *subtracts* its $10 \log_{10} N$ term. That
sign cannot be derived: measuring `N` identical units together
raises the transmitted power by $10 \log_{10} N$, so recovering the
per-unit `DI,n,e` requires *adding* $10 \log_{10} N$, exactly as
the pressure-based ISO 10140-2:2010 Formula (6) does with its
$10 \log_{10}(n A_0/A)$ term (and consistently with ISO 15186-2:2010
Formula (12), which is Formula (8) without an `N` term). This
function implements the corrected per-unit form and warns when
`n > 1` deviates from the print (see `docs/ERRATA.md`).
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `lp1` | Source-room sound pressure levels, in dB. |
| `l_in` | Normal sound intensity levels over the measurement surface, in dB. |
| `measurement_area` | Measurement-surface area `Sm`, in m². |
| `n` | Number `N` of small element units in the surface (Default: 1). |

**Returns:** [`IntensityElementNormalizedResult`](/phonometry/reference/api/building/intensity-insulation/#intensityelementnormalizedresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the band counts differ, if `measurement_area` is not positive, if `n` is not a positive integer, or if inputs are non-finite. |

## intensity_sound_reduction

```python
intensity_sound_reduction(
    lp1: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
    *,
    measurement_area: float,
    area: float,
    kc: Sequence[float] | np.ndarray | None = None,
) -> IntensityReductionResult
```

Intensity sound reduction index per ISO 15186-1:2000 (Formula (7)).

Computes, per frequency band, the intensity sound reduction index, in dB,

$$
R_\mathrm{I} = L_{p1} - 6 - \left[ L_{I\mathrm{n}} + 10 \log_{10}\frac{S_\mathrm{m}}{S} \right]
$$

from the average source-room sound pressure level `Lp1` and the average
normal sound intensity level `LIn` over the measurement surface of area
`Sm` (`measurement_area`), for a specimen of area `S` (`area`). The
same formula gives the apparent index `R'I` in the field (ISO 15186-2).
When an adaptation term `kc` is supplied (see
[`adaptation_term_kc`](/phonometry/reference/api/building/intensity-insulation/#adaptation_term_kc)), the modified index
$R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}$ (Formula (9)) is also
formed. Weighted ratings `RI,w` (and `RI,M,w`) are computed via
[`phonometry.building.weighted_rating`](/phonometry/reference/api/building/ratings/#weighted_rating) (ISO 717-1) when exactly 16
one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz) values are
supplied.

`lp1` and `l_in` may be one value per band (already averaged) or a
two-dimensional `(positions, bands)` array, in which case the positions
are energy-averaged. Subareas scanned separately should first be combined
with [`combine_subareas`](/phonometry/reference/api/building/intensity-insulation/#combine_subareas).

**Parameters**

| Name | Description |
| :--- | :--- |
| `lp1` | Source-room sound pressure levels, in dB. |
| `l_in` | Normal sound intensity levels over the measurement surface, in dB. |
| `measurement_area` | Measurement-surface area `Sm`, in m². |
| `area` | Specimen area `S`, in m². |
| `kc` | Adaptation term `Kc` per band (dB) for the modified index, or `None` to skip it. |

**Returns:** [`IntensityReductionResult`](/phonometry/reference/api/building/intensity-insulation/#intensityreductionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the band counts differ, if `measurement_area` / `area` are not positive, or if inputs are non-finite. |

## IntensityElementNormalizedResult

```python
IntensityElementNormalizedResult(
    d_i_n_e: np.ndarray,
    rating: WeightedRatingResult | None,
    measurement_area: float | None = None,
    n: int = 1,
)
```

Per-band intensity element normalized level difference (ISO 15186-1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `d_i_n_e` | Intensity element normalized level difference $D_\mathrm{I,n,e} = L_{p1} - 6 - (L_{I\mathrm{n}} + 10 \log_{10}(S_\mathrm{m}/A_0)) + 10 \log_{10} N$ per band, in dB (Clause 3.9, Formula (8) with the corrected sign of its $10 \log_{10} N$ term; see `docs/ERRATA.md`). |
| `rating` | Single-number weighted rating `DI,n,e,w` with `C` / `Ctr` (ISO 717-1), or `None` when the band count is neither 16 (one-third octave) nor 5 (octave). |
| `measurement_area` | Measurement-surface area `Sm`, in m², or `None` on a manually built result (Clause 8 g). |
| `n` | Number `N` of element units within the measurement surface. |

### IntensityElementNormalizedResult.plot()

```python
IntensityElementNormalizedResult.plot(
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes
```

Plot `DI,n,e` against the shifted ISO 717-1 reference curve.

Delegates to the weighted-rating plot. Requires the automatic rating
to be available (16 or 5 bands) and matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

### IntensityElementNormalizedResult.report()

```python
IntensityElementNormalizedResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
    fpi: Sequence[float] | np.ndarray | None = None,
    residual_index: Sequence[float] | np.ndarray | None = None,
) -> str
```

Render an ISO 15186-1 element-normalized insulation report to a PDF.

Writes the one-page laboratory test report of ISO 15186-1:2000
Clause 8 for the element-normalized level difference `DI,n,e` of a
small building element measured with sound intensity: the
standard-basis line, an optional metadata header block, the band table
(16 one-third-octave or 5 octave bands) beside the
measured-versus-shifted-reference curve, the boxed rating `DI,n,e,w
(C; Ctr)` (the element-normalized level difference `DI,n,e` rated
per ISO 717-1), the intensity-method statement with the
measurement-surface area `Sm` and unit count `N` when the result
carries them (Clause 8 g), an optional verdict row and a footer with
the identity block and disclaimer. The report requires the
single-number `rating` to be present on the result; it is formed
automatically only for exactly 16 one-third-octave (100 Hz to 3150 Hz)
or 5 octave (125 Hz to 2000 Hz) bands, and a result carrying no rating
(any other band count) is rejected.

The applicable [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) fields describe the
intensity measurement: `specimen` (the tested element and its
mounting and sealing, Clause 8 e), `area` (the element area),
`client`, `manufacturer`, `test_room` (the laboratory /
facility), `laboratory`, `operator`, `report_id` and
`test_date`, plus the room/climate fields shared with the other
insulation fiches. The measurement-surface shape, the measurement
distance and the scanning-versus-discrete-point acquisition method
(Clause 8 j-l) are not dedicated metadata fields; record them in
`notes` (free text) and name the measurement standard in
`measurement_standard` (`"ISO 15186-1"`). A `requirement` adds a
PASS/FAIL verdict; the element insulation passes at or above the target.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating, statement and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the left table shows the ISO 717 evaluation per band (the `DI,n,e` value, the shifted reference and the unfavourable deviation) instead of the two-column form. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |
| `fpi` | Optional per-band surface pressure-intensity indicator `FpI` (Clause 8 i requires it as a function of frequency); annexed as a table column when supplied. |
| `residual_index` | Optional per-band pressure-residual intensity index `δpI0` of the probe and analyser (Clause 8 i); annexed as a table column when supplied. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown, `language` is not one of the supported values, the result carries no single-number rating (its band count is neither 16 one-third-octave nor 5 octave, so the ISO 717-1 rating the fiche needs was not formed), or `fpi` / `residual_index` do not match the band count. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded rating figure (`pip install phonometry[plot]`). |

## IntensityReductionResult

```python
IntensityReductionResult(
    r_i: np.ndarray,
    r_i_modified: np.ndarray | None,
    rating: WeightedRatingResult | None,
    rating_modified: WeightedRatingResult | None,
    area: float | None = None,
    measurement_area: float | None = None,
)
```

Per-band intensity sound reduction index (ISO 15186-1:2000).

**Attributes**

| Name | Description |
| :--- | :--- |
| `r_i` | Intensity sound reduction index $R_\mathrm{I} = L_{p1} - 6 - [L_{I\mathrm{n}} + 10 \log_{10}(S_\mathrm{m}/S)]$ per band, in dB (Clause 3.8, Formula (7)). In the field (ISO 15186-2) this is the apparent index `R'I`. |
| `r_i_modified` | Modified index $R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}$ per band, in dB (Clause 3.10, Formula (9)), or `None` when no adaptation term was supplied. |
| `rating` | Single-number weighted rating `RI,w` with `C` / `Ctr` (ISO 717-1), or `None` when the band count is neither 16 (one-third octave) nor 5 (octave). |
| `rating_modified` | Weighted rating `RI,M,w` of the modified index, or `None` when unavailable. |
| `area` | Test-object area `S`, in m², or `None` on a manually built result. Carried so the report can state it (Clause 8 g). |
| `measurement_area` | Measurement-surface area `Sm`, in m², or `None` on a manually built result (Clause 8 g). |

### IntensityReductionResult.plot()

```python
IntensityReductionResult.plot(ax: Axes | None = None, **kwargs: Any) -> Axes
```

Plot `RI` against the shifted ISO 717-1 reference curve.

Delegates to the weighted-rating plot (measured `RI` versus the
shifted reference, unfavourable deviations shaded). Requires the
automatic rating to be available (16 or 5 bands) and matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

### IntensityReductionResult.report()

```python
IntensityReductionResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
    fpi: Sequence[float] | np.ndarray | None = None,
    residual_index: Sequence[float] | np.ndarray | None = None,
) -> str
```

Render an ISO 15186-1 intensity sound-insulation report to a PDF.

Writes the one-page laboratory test report of ISO 15186-1:2000
Clause 8 for sound insulation measured with sound intensity: the
standard-basis line, an optional metadata header block, the band table
(16 one-third-octave or 5 octave bands) beside the
measured-versus-shifted-reference curve, the boxed rating `RI,w
(C; Ctr)` (the intensity sound reduction index `RI` rated per
ISO 717-1), the intensity-method statement with the test-object and
measurement-surface areas `S` / `Sm` when the result carries them
(Clause 8 g), an optional verdict row and a footer with the identity
block and disclaimer. The report requires the single-number `rating`
to be present on the result; it is formed automatically only for
exactly 16 one-third-octave (100 Hz to 3150 Hz) or 5 octave (125 Hz
to 2000 Hz) bands, and a result carrying no rating (any other band
count) is rejected.

The applicable [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) fields describe the
intensity measurement: `specimen` (the tested element and its
mounting, sealing and mass per unit area, Clause 8 e), `client`,
`manufacturer`, `test_room` (the laboratory / facility),
`laboratory`, `operator`, `report_id` and `test_date`, plus
the room/climate fields shared with the other insulation fiches. The
measurement-surface shape, the measurement distance and the
scanning-versus-discrete-point acquisition method (Clause 8 j-l) are
not dedicated metadata fields; record them in `notes` (free text)
and name the measurement standard in `measurement_standard`
(`"ISO 15186-1"`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating, statement and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` and the Kc-modified index `RI,M` is available, the table annexes `RI,M` beside the reported `RI`. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |
| `fpi` | Optional per-band surface pressure-intensity indicator `FpI` (Clause 8 i requires it as a function of frequency); annexed as a table column when supplied. |
| `residual_index` | Optional per-band pressure-residual intensity index `δpI0` of the probe and analyser (Clause 8 i); annexed as a table column when supplied. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown, `language` is not one of the supported values, the result carries no single-number rating (its band count is neither 16 one-third-octave nor 5 octave, so the ISO 717-1 rating the fiche needs was not formed), or `fpi` / `residual_index` do not match the band count. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded rating figure (`pip install phonometry[plot]`). |

## limp_panel_reduction_index

```python
limp_panel_reduction_index(
    frequencies: Sequence[float] | np.ndarray,
    *,
    surface_mass: float,
    area: float,
    temperature: float = 23.0,
    static_pressure: float = 101300.0,
) -> np.ndarray
```

Sound reduction index of a limp panel (ISO 15186-3:2002, Annex A).

Annex A is normative and is how a laboratory qualifies itself: measure a
limp panel of area $S > 1~\text{m}^2$, calculate what it should
read, and require the two to agree within 4,0 dB from 50 Hz to 160 Hz.
This is the calculated half.

A.1 states two different things about the area, and this enforces the
second: the *qualification panel* is required to be larger than 1 m², while
Formula (A.3) is declared valid "if the area of the test specimen is at
least 1 m²". A panel of exactly 1 m² is therefore refused as a
qualification but accepted as an input, which is the boundary the code
takes.

$$
R = R_0 - 10 \lg 2\sigma_\mathrm{d} \tag{A.1}
$$

$$
R_0 = 20 \lg \frac{\pi f m}{\rho c} \tag{A.2}
$$

$$
\sigma_\mathrm{d} = \frac{1}{2} \left[ 0{,}20 + \ln\left( 2\pi \frac{f}{c} \sqrt{S} \right) \right] \tag{A.3}
$$

with the characteristic impedance and the speed of sound taken from the
climate of the test (Formulas (A.4) and (A.5)),

$$
\rho c = 427 \sqrt{\frac{273}{273 + \theta}} \cdot \frac{B}{B_0}, \qquad c = 331 + 0{,}6\,\theta
$$

The panel is limp by assumption: $R_0$ is the mass law and
$\sigma_\mathrm{d}$ the radiation efficiency of forced transmission
alone. The 160 Hz ceiling is not this annex's own: Clause 1.1 applies the
whole of this part over 50 Hz to 160 Hz, and the qualification inherits it.

Any frequency is computed, and no range is imposed. A.1 declares Formula
(A.3) valid "for the frequency range of this part of ISO 15186", so a
result outside 50 Hz to 160 Hz is the model evaluated past its stated
validity: useful for seeing where a real panel leaves it, not usable as a
qualification. Restricting the input is the caller's to do, unlike
[`low_frequency_intensity_reduction`](/phonometry/reference/api/building/intensity-insulation/#low_frequency_intensity_reduction), whose quantity is defined over
that range and nowhere else.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave mid-band frequencies, in hertz. |
| `surface_mass` | Surface mass `m` of the panel, in kg/m². |
| `area` | Panel area `S`, in m². Formula (A.3) is stated valid for at least 1 m², so a smaller one is refused rather than extrapolated. A panel used to qualify a facility has to exceed 1 m² (A.1). |
| `temperature` | Air temperature `theta`, in degrees Celsius. |
| `static_pressure` | Static pressure `B`, in pascals. |

**Returns:** The calculated sound reduction index per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-finite or non-positive frequency, surface mass, area below 1 m², or a climate the formulas cannot be evaluated in. |

## low_frequency_element_normalized_difference

```python
low_frequency_element_normalized_difference(
    lp_surface: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
    *,
    measurement_area: float,
    elements: int = 1,
    l_p: Sequence[float] | np.ndarray | None = None,
    frequencies: Sequence[float] | np.ndarray | None = None,
    absorbing_specimen_surface: bool = False,
) -> LowFrequencyElementResult
```

Element normalized level difference at low frequencies (ISO 15186-3).

Clause 3.9, Formula (8), for small building elements measured with the
surface-pressure method of this part:

$$
D_{I\mathrm{n,e}} = L_{p\mathrm{S}} - 9 - \left[ L_{I\mathrm{n}} - 10 \lg \frac{A_0}{S_\mathrm{m}} - 10 \lg N \right] \mathrm{dB}
$$

with the reference absorption area $A_0 = 10$ m². Two things
separate it from its part 1 sibling
([`intensity_element_normalized_difference`](/phonometry/reference/api/building/intensity-insulation/#intensity_element_normalized_difference)): the 9 dB of the
surface measurement in place of 6, and the sign of the $10\lg N$
term, which this part prints as the derivable one. Part 1 prints it
subtracted, which is registered in `docs/ERRATA.md`; the two parts
disagree on the page, and this is the one that agrees with the physics.

**Parameters**

| Name | Description |
| :--- | :--- |
| `lp_surface` | Sound pressure levels over the surface of the test specimen in the source room, `LpS`, in dB. One value per band, or a `(positions, bands)` array that is energy-averaged. |
| `l_in` | Normal sound intensity levels over the measurement surface in the receiving room, in dB. |
| `measurement_area` | Measurement-surface area `Sm`, in m². |
| `elements` | Number `N` of identical element units installed within the measurement surface, at least 1. |
| `l_p` | Sound pressure levels on the measurement surface in the receiving room, measured alongside `l_in` (Clause 6.4.2), in dB, or `None` when they were not. |
| `frequencies` | Mid-band frequencies, in hertz, or `None`. Clause 6.6 admits 50 Hz to 160 Hz and nothing else. |
| `absorbing_specimen_surface` | `True` when the test specimen presents a sound-absorbing surface in the receiving room, which tightens the Clause 6.4.2 limit from 10 dB to 6 dB. |

**Returns:** [`LowFrequencyElementResult`](/phonometry/reference/api/building/intensity-insulation/#lowfrequencyelementresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the band counts differ, if the measurement area is not positive, if `elements` is not a positive integer, if any level is non-finite, or if `frequencies` carries a band outside the range this part is defined for. |

## low_frequency_intensity_reduction

```python
low_frequency_intensity_reduction(
    lp_surface: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
    *,
    measurement_area: float,
    area: float,
    l_p: Sequence[float] | np.ndarray | None = None,
    frequencies: Sequence[float] | np.ndarray | None = None,
    absorbing_specimen_surface: bool = False,
) -> LowFrequencyIntensityResult
```

Intensity sound reduction index at low frequencies (ISO 15186-3:2002).

Below 100 Hz a source room has too few modes for its average pressure to
describe what reaches the specimen, so this part measures the pressure
**on the surface of the specimen** instead and the receiving side with an
intensity probe as before (Clause 3.8, Formula (7)):

$$
R_\mathrm{I} = L_{p\mathrm{S}} - 9 - \left[ L_{I\mathrm{n}} + 10 \lg \frac{S_\mathrm{m}}{S} \right] \mathrm{dB}
$$

The 9 dB is the whole difference from [`intensity_sound_reduction`](/phonometry/reference/api/building/intensity-insulation/#intensity_sound_reduction),
which subtracts 6: close to a rigid boundary a diffuse field carries twice
the mean-square pressure it carries away from one, so the surface average
sits three decibels above the room average of the same field.

Clause 7 requires the surface-pressure intensity indicator
$F_{pI} = L_p - L_{I\mathrm{n}}$ (Formula (5)) to be reported beside
the index, and Clause 6.4.2 refuses the measurement surface where it
exceeds 10 dB for a sound-reflecting test specimen, or 6 dB for a specimen
with a sound-absorbing surface in the receiving room. Those bands come
back flagged rather than dropped, because the standard's answer to them is
to improve the measurement environment, not to discard the band.

Both levels of Formula (5) are read on the measurement surface in the
**receiving** room, so the indicator needs `l_p` and not the source-room
surface levels Formula (7) is built from. Clause 6.4.2 asks for that
second measurement "if possible", so it is optional here and its absence
leaves the indicator and the qualification unanswered rather than
guessed.

Clause 6.4.2 refuses a negative measured intensity for the same reason,
which a level cannot carry: pass the signed sub-area intensities through
[`combine_subareas`](/phonometry/reference/api/building/intensity-insulation/#combine_subareas) first, which is where the sign lives.

`lp_surface` and `l_in` may be one value per band or a
`(positions, bands)` array, in which case the positions are
energy-averaged. Sub-areas scanned separately are combined first with
[`combine_subareas`](/phonometry/reference/api/building/intensity-insulation/#combine_subareas), which is Formulas (9) and (10) here.

**Parameters**

| Name | Description |
| :--- | :--- |
| `lp_surface` | Sound pressure levels over the surface of the test specimen in the source room, `LpS`, in dB. |
| `l_in` | Normal sound intensity levels over the measurement surface in the receiving room, in dB. |
| `measurement_area` | Measurement-surface area `Sm`, in m². |
| `area` | Test-object area `S`, in m². |
| `l_p` | Sound pressure levels on the measurement surface in the receiving room, measured alongside `l_in` (Clause 6.4.2), in dB, or `None` when they were not. They are what Formula (5) subtracts `l_in` from; without them no band can be qualified. |
| `frequencies` | Mid-band frequencies, in hertz, or `None`. Clause 6.6 requires at least the 50 Hz, 63 Hz and 80 Hz one-third octaves and allows 100 Hz, 125 Hz and 160 Hz; a band outside 50 Hz to 160 Hz is refused, because this method is defined for the low-frequency range alone. |
| `absorbing_specimen_surface` | `True` when the test specimen presents a sound-absorbing surface in the receiving room, which tightens the Clause 6.4.2 limit from 10 dB to 6 dB. A specimen absorbing on one side only is mounted with that side towards the source room (Clause 5.3), so this is the two-absorbing-sides case. |

**Returns:** [`LowFrequencyIntensityResult`](/phonometry/reference/api/building/intensity-insulation/#lowfrequencyintensityresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the band counts differ, if either area is not positive, if any level is non-finite, or if `frequencies` carries a band outside the range this part is defined for. |

## LowFrequencyElementResult

```python
LowFrequencyElementResult(
    d_i_n_e: np.ndarray,
    surface_pressure_intensity: np.ndarray | None,
    qualified: np.ndarray | None,
    frequencies: np.ndarray | None,
    measurement_area: float,
    elements: int,
    absorbing_specimen_surface: bool,
)
```

Per-band element normalized level difference at low frequencies.

The result of ISO 15186-3:2002, Clause 3.9, Formula (8), the small-element
counterpart of [`LowFrequencyIntensityResult`](/phonometry/reference/api/building/intensity-insulation/#lowfrequencyintensityresult). No single-number
rating accompanies it: Clause 6.6 stops at 160 Hz, six one-third octaves,
and ISO 717-1 needs sixteen.

**Attributes**

| Name | Description |
| :--- | :--- |
| `d_i_n_e` | Intensity element normalized level difference $D_{I\mathrm{n,e}} = L_{p\mathrm{S}} - 9 - [L_{I\mathrm{n}} - 10\lg(A_0/S_\mathrm{m}) - 10\lg N]$ per band, in dB. |
| `surface_pressure_intensity` | Surface-pressure intensity indicator $F_{pI}$ per band, in dB (Formula (5)), or `None` where the receiving-side pressure level was not measured alongside the intensity. |
| `qualified` | The Clause 6.4.2 verdict per band, or `None` throughout when the indicator itself is `None`. |
| `frequencies` | Mid-band frequencies, in hertz, or `None`. |
| `measurement_area` | Measurement-surface area `Sm`, in m². |
| `elements` | Number `N` of element units installed within the measurement surface. |
| `absorbing_specimen_surface` | Which of the two Clause 6.4.2 limits was applied. |

### LowFrequencyElementResult.indicator_limit

*property*

The Clause 6.4.2 limit on `FpI` that this result was judged by.

**Returns:** 6.0 dB when the specimen presents a sound-absorbing surface in the receiving room, 10.0 dB when it is sound-reflecting.

### LowFrequencyElementResult.plot()

```python
LowFrequencyElementResult.plot(
    ax: Axes | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw `DI,n,e` per band, hatching any band Clause 6.4.2 refuses.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the band bar. |

**Returns:** The axes.

## LowFrequencyIntensityResult

```python
LowFrequencyIntensityResult(
    r_i: np.ndarray,
    surface_pressure_intensity: np.ndarray | None,
    qualified: np.ndarray | None,
    frequencies: np.ndarray | None,
    area: float,
    measurement_area: float,
    absorbing_specimen_surface: bool,
)
```

Per-band intensity sound reduction index at low frequencies.

The result of ISO 15186-3:2002, Clause 3.8, Formula (7). It differs from
its part 1 sibling in where the source-room pressure is measured and so in
what has to be subtracted from it: 9 dB against the surface of the
specimen, where part 1 subtracts 6 dB from a room average.

**Attributes**

| Name | Description |
| :--- | :--- |
| `r_i` | Intensity sound reduction index $R_\mathrm{I} = L_{p\mathrm{S}} - 9 - [L_{I\mathrm{n}} + 10\lg(S_\mathrm{m}/S)]$ per band, in dB. |
| `surface_pressure_intensity` | Surface-pressure intensity indicator $F_{pI} = L_p - L_{I\mathrm{n}}$ per band, in dB (Formula (5)), which Clause 7 requires to be reported beside the index, or `None` where the receiving-side pressure level was not measured alongside the intensity. Clause 6.4.2 only asks for that measurement "if possible". |
| `qualified` | `True` in each band whose `FpI` is within the limit Clause 6.4.2 sets, `False` where the measurement surface is not qualified and the index is not a result the standard admits, and `None` throughout when the indicator itself is `None`. |
| `frequencies` | Mid-band frequencies, in hertz, or `None`. |
| `area` | Test-object area `S`, in m². |
| `measurement_area` | Measurement-surface area `Sm`, in m². |
| `absorbing_specimen_surface` | Which of the two Clause 6.4.2 limits was applied, 6 dB when the specimen presents a sound-absorbing surface in the receiving room and 10 dB when it is sound-reflecting. |

### LowFrequencyIntensityResult.indicator_limit

*property*

The Clause 6.4.2 limit on `FpI` that this result was judged by.

Reported beside the indicator, and drawn by `plot` so the figure
does not have to restate which of the two limits applies.

**Returns:** 6.0 dB when the specimen presents a sound-absorbing surface in the receiving room, 10.0 dB when it is sound-reflecting.

### LowFrequencyIntensityResult.plot()

```python
LowFrequencyIntensityResult.plot(
    ax: Axes | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the index per band, hatching any band Clause 6.4.2 refuses.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the band bar. |

**Returns:** The axes.

## surface_pressure_intensity_indicator

```python
surface_pressure_intensity_indicator(
    lp: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
) -> np.ndarray
```

Surface pressure-intensity indicator `FpI` (ISO 15186-1, Formula (10)).

Returns $F_{pI} = L_p - L_{I\mathrm{n}}$ per band from the surface- and
time-averaged
sound pressure level `Lp` and normal sound intensity level `LIn` on
the measurement surface (Clause 3.6 / 6.4.2). The measurement surface is
adequately qualified when `FpI` does not exceed 10 dB for a
sound-reflecting specimen, or 6 dB when the receiving side is sound
absorbing (Clause 6.4.2 flags $F_{pI} > 10$ dB /
$F_{pI} > 6$ dB as not
satisfactory); in addition the probe's pressure-residual intensity index
must exceed $F_{pI} + 10$ dB (Clause 4.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `lp` | Surface-averaged sound pressure levels, in dB. |
| `l_in` | Normal sound intensity levels on the surface, in dB. |

**Returns:** The indicator `FpI` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the shapes differ or contain non-finite values. |
