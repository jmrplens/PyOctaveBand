---
title: "building.facade_prediction"
description: "Fa\u00e7ade sound insulation and outdoor radiation prediction (EN 12354-3/-4:2000)."
sidebar:
  label: "facade_prediction"
---

Façade sound insulation and outdoor radiation prediction (EN 12354-3/-4:2000).

Two companion prediction models for the building envelope, both built on the
same energy summation of element transmission factors
$\tau = 10^{-R/10}$, area-weighted by $S_i/S$ (small elements /
air paths enter through their element-normalized level difference `Dn,e`
with the reference area $A_0 = 10$ m²):

**EN 12354-3, outdoor → indoor (façade sound insulation).** The apparent
sound reduction index of a façade for diffuse incidence (Formula 10):

$$
R' = -10 \log_{10}\!\left( \sum \tau_{e,i} \right)
$$

$$
\tau_{e,i} = \frac{S_i}{S}\, 10^{-R_i/10} \tag{Formula 15}
$$

$$
\tau_{e,i} = \frac{A_0}{S}\, 10^{-D_{n,e,i}/10} \tag{Formula 14}
$$

from which the loudspeaker- and traffic-referenced indices
$R_{45} = R' + 1$ (Formula 11) and $R_{tr,s} = R'$ (Formula 12),
and the primary output, the standardized level difference at 2 m
(Formula 13):

$$
D_{2m,nT} = R' + \Delta L_{fs} + 10 \log_{10}\!\left( \frac{V}{6 T_0 S} \right) \qquad T_0 = 0.5~\text{s}
$$

with the façade-shape term `ΔLfs` (Annex C; 0 dB for a flat reflecting
façade).

**EN 12354-4, indoor → outdoor (sound radiated to the outside).** The sound
power level radiated by a segment (Formulas 2-3):

$$
R' = -10 \log_{10}\!\left( \sum \frac{S_i}{S}\, 10^{-R_i/10} + \sum \frac{A_0}{S}\, 10^{-D_{n,e,i}/10} \right)
$$

$$
L_W = L_{p,\text{in}} + C_d - R' + 10 \log_{10}\!\left( \frac{S}{S_0} \right) \qquad S_0 = 1~\text{m}^2
$$

with the inside-field diffusivity term `Cd` (Annex B; -6 dB ideal diffuse,
-5 dB average industrial). An opening is modelled here as an element whose "R"
is the silencer insertion loss `D` (a bare opening is $D = 0$),
combined in the *same* energy sum as the structural elements over the segment
area `S` -- a practical extension for a mixed wall-plus-opening segment.
This is NOT the standard's Formula (4), which treats a segment made up *only*
of openings with a different area normalization (`S` = the opening area) and
sums its `LW` with the envelope segments only at the final energetic stage.
The exterior level follows from the simplified Annex E attenuation `Atot` of
a finite radiating side and $L_p = L_W - A_{tot}$.

Single-number ratings reuse EN ISO 717-1 via [`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating)
(exact for $R'_w + C_{tr}$, a good approximation for `R'w`, Part 3
NOTE 7).

Clause/formula citations refer to EN 12354-3:2000 or EN 12354-4:2000.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## facade_shape_level_difference

```python
facade_shape_level_difference(
    shape: str,
    *,
    line_of_sight: float = 0.0,
    absorption: float = 0.3,
) -> float
```

Façade-shape level difference `ΔLfs` (EN 12354-3:2000 Annex C).

Looks up Figure C.2 for the level difference caused by the exterior shape
of the façade (gallery, balcony or terrace), as a function of the height
of the line of sight from the source on the façade plane and the weighted
sound absorption coefficient `αw` (EN ISO 11654) of the underside of the
balcony/roof above. The value feeds `delta_l_fs` of
[`facade_sound_reduction`](/phonometry/reference/api/building/facade-prediction/#facade_sound_reduction) (Formula 13). Intermediate `αw` values
are interpolated linearly between the tabulated 0,3 / 0,6 / 0,9 columns,
as the annex allows; outside that range the edge column applies. The
2017 edition tabulates the same values (Tabelle C.1).

Shapes follow the Figure C.2 numbering: `"plane_facade"` (1, always
0 dB), `"gallery_2"` to `"gallery_5"` (2-5), `"balcony_6"` to
`"balcony_8"` (6-8) and `"terrace_open"` / `"terrace_closed"` (9,
open or closed fence).

**Parameters**

| Name | Description |
| :--- | :--- |
| `shape` | Façade shape key (see above). |
| `line_of_sight` | Height of the line of sight from the source at the façade plane, in m (bins: below 1,5 m; 1,5 m to 2,5 m; above 2,5 m). |
| `absorption` | Weighted absorption coefficient `αw` of the underside above the façade (default 0,3; reflecting). |

**Returns:** `ΔLfs`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For an unknown shape, a negative height/absorption, or a shape/height combination the figure marks "does not apply". |

## facade_sound_reduction

```python
facade_sound_reduction(
    elements: Sequence[FacadeElement],
    *,
    area: float,
    volume: float,
    delta_l_fs: float = 0.0,
    bands: str | None = None,
    frequencies: Sequence[float] | None = None,
) -> FacadePredictionResult
```

Predict façade airborne sound insulation `D2m,nT` (EN 12354-3:2000).

Energetically combines the element transmission factors (Formula 10) into the
apparent sound reduction index `R'`, then derives the loudspeaker/traffic
indices (Formulas 11-12) and the standardized level difference (Formula 13).

**Parameters**

| Name | Description |
| :--- | :--- |
| `elements` | Façade elements (see [`FacadeElement`](/phonometry/reference/api/building/facade-prediction/#facadeelement)); per-band arrays must share a common length (5 octave or 16 third-octave bands to get single-number ratings). |
| `area` | Total façade area `S` seen from inside, in m². |
| `volume` | Receiving-room volume `V`, in m³ (Formula 13). |
| `delta_l_fs` | Façade-shape term `ΔLfs` in dB (Annex C; 0 for a flat reflecting façade). |
| `bands` | `"octave"`, `"third-octave"` or `None` (auto) for the single number ratings, passed to [`weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating). |
| `frequencies` | Optional band centre frequencies (Hz), stored on the result for plotting; must match the element band count. |

**Returns:** A [`FacadePredictionResult`](/phonometry/reference/api/building/facade-prediction/#facadepredictionresult).

## FacadeElement

```python
FacadeElement(
    name: str,
    area: float | None = None,
    r: float | Sequence[float] | np.ndarray | None = None,
    dn_e: float | Sequence[float] | np.ndarray | None = None,
    insertion_loss: float | Sequence[float] | np.ndarray | None = None,
)
```

One façade element as a transmission path (EN 12354-3/-4).

Provide exactly one of `r` (an area element, Formula 15 / Part 4 Formula 3),
`dn_e` (a small element or air path, Formula 14) or `insertion_loss` (an
opening in Part 4, modelled as an element whose reduction is the silencer's
insertion loss `D` and combined in the same energy sum -- a practical
extension, not the standard's separate segment-of-openings Formula 4). Per-band
values may be scalars or equal-length arrays.

**Attributes**

| Name | Description |
| :--- | :--- |
| `name` | Label used in results and plots. |
| `area` | Element area `Sᵢ` in m² (required for `r` / `insertion_loss`; ignored for `dn_e` small elements, which use `A₀` instead). |
| `r` | Sound reduction index `Rᵢ` in dB. |
| `dn_e` | Element-normalized level difference `Dn,e,i` in dB. |
| `insertion_loss` | Opening silencer insertion loss `Dᵢ` in dB. |

### FacadeElement.tau()

```python
FacadeElement.tau(total_area: float, n_bands: int) -> np.ndarray
```

Transmission factor `τ` of this element for the whole façade area.

## FacadePredictionResult

```python
FacadePredictionResult(
    r_prime: np.ndarray,
    r_45: np.ndarray,
    r_tr_s: np.ndarray,
    d_2m_nt: np.ndarray,
    element_r: dict[str, np.ndarray],
    r_tr_s_w: int | None = None,
    d_2m_nt_w: int | None = None,
    c_tr: int | None = None,
    frequencies: np.ndarray | None = None,
    elements: tuple[FacadeElement, ...] | None = None,
)
```

Predicted façade airborne insulation (EN 12354-3:2000).

**Attributes**

| Name | Description |
| :--- | :--- |
| `r_prime` | Apparent sound reduction index `R'` per band, in dB (Formula 10). |
| `r_45` | $R_{45} = R' + 1$ (loudspeaker method, Formula 11), in dB. |
| `r_tr_s` | $R_{tr,s} = R'$ (traffic, Formula 12), in dB. |
| `d_2m_nt` | Standardized level difference `D2m,nT` per band, in dB (Formula 13). |
| `element_r` | Per-element partial index $R_p = -10 \log_{10} \tau$ per band, in dB. |
| `r_tr_s_w` | Single-number `Rtr,s,w` (ISO 717-1); `None` if the bands are not the ISO 717-1 octave/third-octave set. |
| `d_2m_nt_w` | Single-number `D2m,nT,w` (ISO 717-1); `None` as above. |
| `c_tr` | Spectrum adaptation term `Ctr` of `R'` (ISO 717-1). |
| `frequencies` | Band centre frequencies (Hz) for plotting; `None` labels the axis by band index. |
| `elements` | The element sequence the prediction was run with, retained so `plot_geometry` can draw the elevation; appended after the original fields and `None` for hand-built results. |

### FacadePredictionResult.plot()

```python
FacadePredictionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-element partial indices and the façade `R'` / `D2m,nT`.

### FacadePredictionResult.plot_geometry()

```python
FacadePredictionResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the composite facade elevation, element areas to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

### FacadePredictionResult.report()

```python
FacadePredictionResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a predicted façade sound insulation report to a PDF (EN 12354-3).

Writes a one-page **prediction** report for the predicted standardized
level difference of a façade `D2m,nT` estimated by the EN/ISO
12354-3:2000 model (Formula 13): a standard-basis line that states the
sheet is a prediction from element data and not a measurement, an
optional metadata header block, a two-panel body with the façade-element
table (each element's weighted partial index `Rp,w`) beside the
per-element partial-index and `R'` / `D2m,nT` plot, the boxed
predicted rating `D2m,nT,w`, the prediction statement and, when a
requirement is supplied, a PASS/FAIL verdict (the level difference
passes at or above the requirement), followed by a footer.

The applicable [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) fields describe the
predicted situation: `specimen` (the façade element set), `area`
(the exposed façade area `S`), `receiving_volume` (the receiving
room volume `V`), `test_room` (the traffic / outdoor situation),
`client`, `manufacturer`, `measurement_standard`, `laboratory`
(the calculator / laboratory), `operator`, `report_id` and
`test_date`. A summary of the façade shape (`ΔLfs`) and the model
assumptions is recorded in `notes` (free text), and `requirement`
supplies the target `D2m,nT,w`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating, statement, disclaimer). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the element table also shows each element's share of the transmitted sound energy. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown, `language` is not supported, or the result lacks the ISO 717-1 single-number ratings (build it on the 5 octave or 16 one-third-octave bands). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## outdoor_attenuation

```python
outdoor_attenuation(width: float, height: float, distance: float) -> float
```

Simplified attenuation `Atot` of a finite side (EN 12354-4 Annex E).

Reception point in front of the centre of a rectangular side
$S = \mathrm{width} \cdot \mathrm{height}$ at perpendicular
`distance` d. Uses the finite-side Formula (E.2a)
up to the largest side dimension and the point-source Formula (E.2b) beyond
it, following the Annex E Note 3 switching rule. The two branches do not
join continuously at the switch distance: for a square 10 m x 10 m side
the step is about -0,7 dB (about -0,3 dB for a 60 m x 10 m side), an
artefact of the standard's own simplification. The `+6 dB` for radiation
into the quarter-space over hard ground is built into the formula.

**Parameters**

| Name | Description |
| :--- | :--- |
| `width` | Side width `L`, in m. |
| `height` | Side height `H`, in m. |
| `distance` | Perpendicular distance `d` to the reception point, in m. |

**Returns:** `Atot` in dB (subtract from `LW` to get the exterior `Lp`).

## outdoor_level

```python
outdoor_level(
    l_w: float | Sequence[float],
    attenuation: float | Sequence[float],
) -> float
```

Exterior level from one or more radiating sides (EN 12354-4 Formula E.1).

$L_p = 10 \log_{10}( \sum 10^{L_{W,k}/10} ) - A_{tot}$ for sides sharing
a reception point,
or the per-side `LW - Atot` energetically summed. Pass matching sequences
of side power levels and their attenuations, or scalars for a single side; a
scalar broadcasts against an array (e.g. several sides, one common `Atot`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `l_w` | Radiated power level(s) `LW` (dB), scalar or per side. |
| `attenuation` | Attenuation(s) `Atot` (dB), scalar or per side. |

**Returns:** Exterior sound pressure level `Lp` in dB.

## plot_facade_elements

```python
plot_facade_elements(
    elements: Sequence[FacadeElement],
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw a composite facade elevation with element areas to scale.

Each element of [`facade_sound_reduction`](/phonometry/reference/api/building/facade-prediction/#facade_sound_reduction) is a
tile whose drawn area equals its real area (small-area elements without
an area, such as airbriks rated by `dn_e`, get a nominal 0,1 m2 tile).

**Parameters**

| Name | Description |
| :--- | :--- |
| `elements` | The [`FacadeElement`](/phonometry/reference/api/building/facade-prediction/#facadeelement) sequence. |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the first element rectangle. |

**Returns:** The axes.

## radiated_sound_power

```python
radiated_sound_power(
    elements: Sequence[FacadeElement],
    *,
    lp_in: float | Sequence[float] | np.ndarray,
    area: float,
    c_d: float = -6.0,
    r_prime_cap: float | None = None,
    octave_bands: Sequence[int] | None = None,
) -> RadiatedPowerResult
```

Predict the sound power radiated outside by a segment (EN 12354-4).

`R'` combines the element transmission factors (Formula 3); the radiated
power level is $L_W = L_{p,in} + C_d - R' + 10 \log_{10}(S/S_0)$
(Formula 2). Openings
may be included as [`FacadeElement`](/phonometry/reference/api/building/facade-prediction/#facadeelement) entries with an `insertion_loss`
(0 for a bare opening); see the module docstring for how this differs from
the standard's separate segment-of-openings Formula (4).

**Parameters**

| Name | Description |
| :--- | :--- |
| `elements` | Segment elements (see [`FacadeElement`](/phonometry/reference/api/building/facade-prediction/#facadeelement)). |
| `lp_in` | Inside sound pressure level `Lp,in` per band, in dB. |
| `area` | Segment area `S`, in m². |
| `c_d` | Inside-field diffusivity term `Cd` in dB (Annex B: -6 ideal diffuse, -5 average industrial building). |
| `r_prime_cap` | Optional practical maximum on `R'` per band, in dB. This cap is **not** part of Formula (2)/(3): it appears only as a footnote of the Annex G worked example ("R' limited to 40 dB" for field situations with unavoidable leaks). Pass `40.0` to reproduce Annex G; the default `None` computes the bare formulas. |
| `octave_bands` | Optional octave-band centre frequencies (Hz) matching the per-band data; enables the A-weighted single number. |

**Returns:** A [`RadiatedPowerResult`](/phonometry/reference/api/building/facade-prediction/#radiatedpowerresult).

## RadiatedPowerResult

```python
RadiatedPowerResult(
    l_w: np.ndarray,
    r_prime: np.ndarray,
    l_w_dba: float | None = None,
    frequencies: np.ndarray | None = None,
)
```

Predicted sound power radiated to the outside by a segment (EN 12354-4).

**Attributes**

| Name | Description |
| :--- | :--- |
| `l_w` | Radiated sound power level `LW` per band, in dB re 1 pW (Formula 2). |
| `r_prime` | Apparent sound reduction index `R'` per band, in dB (Formula 3). |
| `l_w_dba` | A-weighted `LW` in dB(A), if the bands are known octave bands; else `None`. |
| `frequencies` | Band centre frequencies (Hz) for plotting; `None` labels the axis by band index. |

### RadiatedPowerResult.plot()

```python
RadiatedPowerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the radiated sound power level `LW` per band.
