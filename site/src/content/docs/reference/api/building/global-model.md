---
title: "building.prediction.global_model"
description: "Building acoustic performance prediction (EN 12354-1/-2:2000)."
sidebar:
  label: "global_model"
---

Building acoustic performance prediction (EN 12354-1/-2:2000).

This is the **prediction** counterpart of the measurement modules
([`phonometry.building.measurement.lab_insulation`](/phonometry/reference/api/building/lab-insulation/) for laboratory `R`/`Ln` and
[`phonometry.building.measurement.insulation`](/phonometry/reference/api/building/insulation/) for field `R'`/`L'n`). EN 12354 estimates the
*in-situ* apparent performance of a building from the laboratory performance of
its elements, adding the flanking transmission that a field measurement would
capture but a laboratory measurement suppresses.

Both parts have a *detailed* per-band model and a *simplified* single-number
model. This module implements the **simplified single-number model** (Part 1
Clause 4.4, Part 2 Clause 4.3): it takes the weighted single-number ratings of
the elements (`Rw` of walls/floors, `ΔRw`/`ΔLw` of linings/coverings and
the `Kij` vibration-reduction indices of the junctions) and predicts the
apparent weighted rating (`R'w` airborne, `L'n,w` impact). The simplified
model is exact for `RA` and a good approximation for `R'w` (Part 1
Clause 4.4.1), with a reported standard deviation of about 2 dB (Clause 5).

**Airborne, Formula (26).** The apparent weighted sound reduction index is the
energetic sum of the direct path `Dd` and, for every flanking element, the
three flanking paths `Ff`, `Df` and `Fd`:

$$
R'_w = -10 \log_{10}\!\left[ 10^{-R_{Dd,w}/10} + \sum 10^{-R_{Ff,w}/10} + \sum 10^{-R_{Df,w}/10} + \sum 10^{-R_{Fd,w}/10} \right]
$$

with the direct path $R_{Dd,w} = R_{s,w} + \Delta R_{Dd,w}$ (Formula 27)
and each flanking path (Formula 28a)

$$
R_{ij,w} = \frac{R_{i,w} + R_{j,w}}{2} + \Delta R_{ij,w} + K_{ij} + 10 \log_{10}\frac{S_s}{l_0 l_f}
$$

where $l_0 = 1$ m is the reference coupling length.

**Junctions, Annex E.** The vibration reduction index `Kij` of rigid cross
(E.3) and T (E.4) junctions, junctions with flexible interlayers (E.5),
lightweight façade junctions (E.6), junctions of lightweight double-leaf walls
with homogeneous elements (E.7) or with other coupled double-leaf walls (E.8),
and corners / thickness changes (E.9) are empirical functions of the mass
ratio $M = \log_{10}(m'_{\perp,i} / m'_i)$. A minimum value `Kij,min`
follows from the Kij,min relation of Clause 4.4.2 (printed as Eq. (23)
in the BS EN 12354-1:2000 edition).

**Impact, Formula (21).** $L'_{n,w} = L_{n,w,eq} - \Delta L_w + K$ with
the bare-floor equivalent level `Ln,w,eq` (Annex B
$164 - 35 \log_{10}(m'/m'_0)$), the covering improvement `ΔLw` (ISO 717-2)
and the flanking correction `K` from Table 1.

Clause citations refer to EN 12354-1:2000 (airborne) or EN 12354-2:2000 (impact).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AirbornePredictionResult

```python
AirbornePredictionResult(
    r_prime_w: float,
    r_direct_w: float,
    paths: tuple[PathContribution, ...],
    dominant: PathContribution,
)
```

Predicted apparent airborne insulation (EN 12354-1:2000, Formula 26).

**Attributes**

| Name | Description |
| :--- | :--- |
| `r_prime_w` | Apparent weighted sound reduction index `R'w`, in dB. |
| `r_direct_w` | Direct-path weighted index `RDd,w`, in dB (Formula 27). |
| `paths` | Per-path contributions in input order (direct path first, then the flanking paths as supplied), each with its share of the energy. |
| `dominant` | The path carrying the most energy (`PathContribution`). |

### AirbornePredictionResult.plot()

```python
AirbornePredictionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-path shares of the transmitted energy.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### AirbornePredictionResult.report()

```python
AirbornePredictionResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a predicted airborne insulation report to a PDF (EN 12354-1).

Writes a one-page **prediction** report for the predicted apparent
sound reduction index `R'` between rooms estimated by the EN/ISO
12354-1:2000 simplified single-number model (Clause 4.4): a
standard-basis line that states the sheet is a prediction from element
data and not a measurement, an optional metadata header block, a
two-panel body with the transmission-path table (the direct path and
each flanking path's weighted index `Rij,w`) beside the per-path
share-of-energy plot, the boxed predicted rating `R'w`, the
prediction statement (with the model's ~2 dB standard deviation) and,
when a requirement is supplied, a PASS/FAIL verdict (the apparent index
passes at or above the requirement), followed by a footer.

The applicable [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) fields describe the
predicted situation: `specimen` (the separating element),
`area` (the separating-element area `Ss`), `source_volume` /
`receiving_volume` (the room geometry), `client`, `manufacturer`,
`test_room`, `laboratory` (the calculator / laboratory),
`operator`, `report_id` and `test_date`. A summary of the
flanking construction and the model assumptions is recorded in
`notes` (free text), and `requirement` supplies the target `R'w`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating, statement, disclaimer). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the path table also shows each path's share of the transmitted sound energy. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown or `language` is not supported. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## combine_linings

```python
combine_linings(delta_a: float, delta_b: float) -> float
```

Combine two lining improvements (EN 12354-1 Formulas 30/31).

For two linings the total improvement is the larger value plus half the
smaller: $\Delta R = \max(a, b) + \min(a, b)/2$. For a single lining
pass the other as `0`.

:::note
ISO 12354-1:2017 (Formulas (22)/(23)) adds a special case the 2000
edition implemented here does not have: when *both* linings are
negative (each lining worsens the insulation), half is taken of the
*higher* value instead,
$\Delta R = \min(a, b) + \max(a, b)/2$, so two
degrading linings degrade further (e.g. −2 and −4 dB combine to
−5 dB under the 2017 rule, −4 dB under the 2000 rule used here).
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `delta_a` | Improvement of the first lining, in dB. |
| `delta_b` | Improvement of the second lining, in dB. |

**Returns:** Combined `ΔR`/`ΔRij`, in dB.

## equivalent_impact_level

```python
equivalent_impact_level(mass_per_area: float) -> float
```

Bare-floor equivalent weighted impact level `Ln,w,eq` (Part 2, Annex B).

$L_{n,w,eq} = 164 - 35 \log_{10}(m'/m'_0)$ with $m'_0 = 1$ kg/m²,
the closed form
used in the Annex E worked example for a homogeneous concrete floor. The
Annex B relation is stated for homogeneous floors of 100 kg/m² to
600 kg/m²; outside that envelope the value is an extrapolation and a
`UserWarning` is emitted.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass_per_area` | Mass per unit area `m'` of the bare floor, in kg/m² (must be positive; the Annex B relation covers 100-600 kg/m²). |

**Returns:** `Ln,w,eq`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `mass_per_area` is not positive. |

## flanking_element

```python
flanking_element(
    *,
    label: str,
    r_flanking: float,
    r_separating: float,
    k_ff: float,
    k_fd: float,
    k_df: float,
    separating_area: float,
    coupling_length: float,
    delta_r_ff: float = 0.0,
    delta_r_fd: float = 0.0,
    delta_r_df: float = 0.0,
    flanking_area: float | None = None,
) -> tuple[FlankingPath, FlankingPath, FlankingPath]
```

Build the three flanking paths (Ff, Df, Fd) of one flanking element.

Convenience wrapper over [`flanking_path`](/phonometry/reference/api/building/global-model/#flanking_path) for the common case where a
flanking element is essentially the same on the source and receiving side
(Clause 4.4.1). Returns the `Ff`, `Df` and `Fd` paths that this element
contributes across its junction with the separating element.

**Kij,min (Clause 4.4.2).** When `flanking_area` is given, the mandatory
floor $K_{ij} \ge K_{ij,\mathrm{min}}$ is applied automatically per
path: `KFf` is clamped to $10 \log_{10}[l_f l_0 (2/S_F)]$ (both
junction elements are
the flanking element) and `KFd`/`KDf` to
$10 \log_{10}[l_f l_0 (1/S_F + 1/S_s)]$ (flanking and separating
element), via
[`junction_min_vibration_reduction`](/phonometry/reference/api/building/global-model/#junction_min_vibration_reduction). Without `flanking_area` the
per-path floors cannot be formed from the available geometry, so the raw
`k_ff`/`k_fd`/`k_df` are used unchanged; compute the floors
yourself (or call [`flanking_path`](/phonometry/reference/api/building/global-model/#flanking_path) with `kij_min`) in that case to
stay within Clause 4.4.2.

**Parameters**

| Name | Description |
| :--- | :--- |
| `label` | Base name; paths are labelled `"<label>-Ff"` etc. |
| `r_flanking` | Weighted sound reduction index of the flanking element. |
| `r_separating` | Weighted sound reduction index of the separating element. |
| `k_ff` | `KFf` vibration reduction index, in dB. |
| `k_fd` | `KFd` vibration reduction index, in dB. |
| `k_df` | `KDf` vibration reduction index, in dB. |
| `separating_area` | Separating-element area `Ss`, in m². |
| `coupling_length` | Junction coupling length `lf`, in m. |
| `delta_r_ff` | Combined lining improvement for the Ff path, in dB. |
| `delta_r_fd` | Combined lining improvement for the Fd path, in dB. |
| `delta_r_df` | Combined lining improvement for the Df path, in dB. |
| `flanking_area` | Flanking-element area $S_F = S_f$, in m². Enables the automatic `Kij,min` clamp (Clause 4.4.2); `None` skips it. |

**Returns:** The `(Ff, Df, Fd)` [`FlankingPath`](/phonometry/reference/api/building/global-model/#flankingpath) triple.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a geometry value is not positive or an input is non-finite. |

## flanking_path

```python
flanking_path(
    *,
    label: str,
    kind: Literal['Ff', 'Df', 'Fd'],
    r_source: float,
    r_receive: float,
    k_ij: float,
    separating_area: float,
    coupling_length: float,
    delta_r: float = 0.0,
    kij_min: float | None = None,
) -> FlankingPath
```

Build one flanking path `Rij,w` (EN 12354-1 Formula 28a).

$$
R_{ij,w} = \frac{R_{i,w} + R_{j,w}}{2} + \Delta R_{ij,w} + K_{ij} + 10 \log_{10}\frac{S_s}{l_0 l_f}
$$

with `r_source` and `r_receive` as $R_{i,w}$ / $R_{j,w}$,
`delta_r` as $\Delta R_{ij,w}$ and `k_ij` as $K_{ij}$.
The two element indices depend on the path: for `Ff` both are the flanking
element (`RF,w`, `Rf,w`); for `Fd` they are the flanking (source) and
separating (receive) elements; for `Df` the separating (source) and
flanking (receive) elements.

When `kij_min` is given, `k_ij` is clamped up to it
(`max(k_ij, kij_min)`) before the path is formed, enforcing the floor
$K_{ij} \ge K_{ij,\mathrm{min}}$ of Clause 4.4.2 (compute
`kij_min` with
[`junction_min_vibration_reduction`](/phonometry/reference/api/building/global-model/#junction_min_vibration_reduction)). Left as `None` the raw `k_ij`
is used unchanged.

**Parameters**

| Name | Description |
| :--- | :--- |
| `label` | Human-readable path name. |
| `kind` | `"Ff"`, `"Df"` or `"Fd"`. |
| `r_source` | Weighted sound reduction index of the source-side element. |
| `r_receive` | Weighted sound reduction index of the receive-side element. |
| `k_ij` | Vibration reduction index of this path, in dB. |
| `separating_area` | Area `Ss` of the separating element, in m². |
| `coupling_length` | Junction coupling length `lf`, in m. |
| `delta_r` | Combined lining improvement `ΔRij,w` for this path, in dB. |
| `kij_min` | Optional `Kij,min` floor (Clause 4.4.2); `k_ij` is raised to it when it lies below. `None` disables the clamp. |

**Returns:** The [`FlankingPath`](/phonometry/reference/api/building/global-model/#flankingpath).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `kind` is unknown, areas/lengths are not positive, or any value is non-finite. |

## FlankingPath

```python
FlankingPath(label: str, kind: Literal['Ff', 'Df', 'Fd'], r_ij_w: float)
```

One flanking transmission path (Ff, Df or Fd) of the simplified model.

**Attributes**

| Name | Description |
| :--- | :--- |
| `label` | Human-readable path name, e.g. `"floor-Ff"`. |
| `kind` | Path type, one of `"Ff"`, `"Df"`, `"Fd"`. |
| `r_ij_w` | Weighted flanking sound reduction index `Rij,w` of the path, in dB (EN 12354-1 Formula 28a). |

## impact_flanking_correction

```python
impact_flanking_correction(
    separating_mass: float,
    flanking_mass: float,
) -> int
```

Flanking correction `K` from Table 1 (EN 12354-2:2000).

Looks up `K` (dB) for the separating-floor mass and the mean mass of the
homogeneous flanking elements, selecting the nearest tabulated row/column
(the table is discrete; masses outside 100–900 / 100–500 kg/m² clamp to the
nearest edge).

**Parameters**

| Name | Description |
| :--- | :--- |
| `separating_mass` | Mass per unit area of the separating floor, in kg/m². |
| `flanking_mass` | Mean mass per unit area of the homogeneous flanking elements not covered by additional layers, in kg/m². |

**Returns:** The correction `K`, in dB (a non-negative integer).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a mass is not positive. |

## ImpactPredictionResult

```python
ImpactPredictionResult(
    l_prime_n_w: float,
    ln_w_eq: float,
    delta_l_w: float,
    k_correction: float,
)
```

Predicted apparent impact insulation (EN 12354-2:2000, Formula 21).

**Attributes**

| Name | Description |
| :--- | :--- |
| `l_prime_n_w` | Apparent weighted normalized impact sound pressure level `L'n,w`, in dB. |
| `ln_w_eq` | Bare-floor equivalent weighted level `Ln,w,eq`, in dB. |
| `delta_l_w` | Weighted covering improvement `ΔLw`, in dB. |
| `k_correction` | Flanking correction `K`, in dB (Table 1). |

### ImpactPredictionResult.plot()

```python
ImpactPredictionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the Formula 21 terms and the resulting `L'n,w`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ImpactPredictionResult.report()

```python
ImpactPredictionResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a predicted impact insulation report to a PDF (EN 12354-2).

Writes a one-page **prediction** report for the predicted apparent
normalized impact sound pressure level `L'n` estimated by the EN/ISO
12354-2:2000 simplified single-number model (Clause 4.3): a
standard-basis line that states the sheet is a prediction from element
data and not a measurement, an optional metadata header block, a
two-panel body with the Formula (21) term table (the bare-floor
equivalent level `Ln,w,eq`, the covering improvement `ΔLw` and the
flanking correction `K`) beside the term plot, the boxed predicted
rating `L'n,w`, the prediction statement (with the model's ~2 dB
standard deviation) and, when a requirement is supplied, a PASS/FAIL
verdict (the apparent level passes at or below the requirement, a lower
impact level being better), followed by a footer.

The applicable [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) fields describe the
predicted situation: `specimen` (the separating floor), `area` (the
floor area), `mass_per_area` (the bare floor's mass per unit area),
`receiving_volume` (the receiving-room geometry), `client`,
`manufacturer`, `test_room`, `laboratory` (the calculator /
laboratory), `operator`, `report_id` and `test_date`. A summary
of the flanking construction and the model assumptions is recorded in
`notes` (free text), and `requirement` supplies the target
`L'n,w`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating, statement, disclaimer). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform `.report()` signature; the impact fiche has a single body layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown or `language` is not supported. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## junction_min_vibration_reduction

```python
junction_min_vibration_reduction(
    coupling_length: float,
    s_i: float,
    s_j: float,
) -> float
```

Minimum vibration reduction index `Kij,min` (EN 12354-1 Clause 4.4.2).

Printed as Formula (29) in the EN 12354-1:2000 edition.

$K_{ij,\mathrm{min}} = 10 \log_{10}[l_f \, l_0 \, (1/S_i + 1/S_j)]$ with
the reference coupling
length $l_0 = 1$ m. When the tabulated `Kij` is below this value,
the minimum is used (Clause 4.4.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `coupling_length` | Common coupling length `lf` of the junction, in m. |
| `s_i` | Area of element `i`, in m². |
| `s_j` | Area of element `j`, in m². |

**Returns:** `Kij,min`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any argument is not positive. |

## junction_vibration_reduction

```python
junction_vibration_reduction(
    junction_type: JunctionType,
    path: PathKind,
    mass_ratio: float,
    *,
    frequency: float = 500.0,
    f1: float = 125.0,
) -> float
```

Vibration reduction index `Kij` of a junction (EN 12354-1 Annex E).

Empirical `Kij` for common junctions as a function of the mass ratio
$M = \log_{10}(m'_{\perp,i} / m'_i)$ (Formula E.2), where `mass_ratio`
is $m'_{\perp,i} / m'_i$, the mass per unit area of the
perpendicular element over that of the element carrying the path.
`path` selects the *through* branch (in-line elements,
`K13`), the *corner* branch ($K_{12} = K_{23}$) or, for double-leaf
separating walls, the *double-leaf* branch (`K24`, the path between the
two flanking legs across the double leaf).

Supported `junction_type` values and their formulas:

- `"rigid_cross"` (E.3): through $8.7 + 17.1 M + 5.7 M^2$;
  corner $8.7 + 5.7 M^2$.
- `"rigid_t"` (E.4): through $5.7 + 14.1 M + 5.7 M^2$;
  corner $5.7 + 5.7 M^2$.
- `"flexible_t"` (E.5, wall junction with flexible interlayers): through
  $5.7 + 14.1 M + 5.7 M^2 + 2 \Delta_1$; corner
  $5.7 + 5.7 M^2 + \Delta_1$ with
  $\Delta_1 = 10 \log_{10}(f/f_1)$ for $f > f_1$ (else 0) and
  $f_1 = 125$ Hz for the typical interlayer
  $E_1/t_1 \approx 100$ MN/m³; double-leaf
  $K_{24} = 3.7 + 14.1 M + 5.7 M^2$ clamped to
  $-4 \le K_{24} \le 0$ dB.
  (The 2000 print states the clamp as "0 ≤ K24 ≤ −4 dB", an obvious
  misprint of the bounds' order.)
- `"lightweight_facade"` (E.6): through $\max(5 + 10 M, 5)$;
  corner $10 + 10 |M|$.
- `"lightweight_double_homogeneous"` (E.7, lightweight double-leaf wall
  joined to homogeneous elements): through
  $\max(10 + 20 M - 3.3 \log_{10}(f/f_k), 10)$; corner
  $10 + 10 |M| + 3.3 \log_{10}(f/f_k)$; double-leaf
  $K_{24} = 3.0 + 14.1 M + 5.7 M^2$ with $f_k = 500$ Hz.
  The K24 path is
  carried by the homogeneous element crossing the double leaf, so its
  per-path `mass_ratio` $= m'_{\perp,i}/m'_i$ is
  leaf-over-homogeneous and the
  validity condition (homogeneous over three times heavier than a leaf)
  reads `mass_ratio` $< 1/3$. (The 2000 print states this line as
  $3.0 - 14.1 M + 5.7 M^2$ in the *figure-axis* variable
  $M = \log_{10}(m_2/m_1)$ of Figure E.9, contradicting the annex's own
  per-path definition of M; both forms are numerically identical, and
  ISO 12354-1:2017 E.3.5 prints the per-path form implemented here. See
  `docs/ERRATA.md`.)
- `"lightweight_double_coupled"` (E.8, junction of lightweight coupled
  double-leaf walls): through
  $\max(10 + 20 M - 3.3 \log_{10}(f/f_k), 10)$;
  corner $10 + 10 |M| - 3.3 \log_{10}(f/f_k)$; with
  $f_k = 500$ Hz.
- `"corner"` (E.9 A, two elements meeting at a corner): corner
  $K_{12} = \max(15 |M| - 3, -2)$ ($= K_{21}$); the only
  path.
- `"thickness_change"` (E.9 B, thickness change in an element): through
  $K_{12} = 5 M^2 - 5$ ($= K_{21}$); the only path.

**Parameters**

| Name | Description |
| :--- | :--- |
| `junction_type` | Junction geometry (see above). |
| `path` | `"through"` (K13; also the single K12 path of a thickness change), `"corner"` (K12 = K23; also the single path of a corner) or `"double_leaf"` (K24). |
| `mass_ratio` | $m'_{\perp,i} / m'_i$, the mass per unit area of the perpendicular element over that of the element carrying the path (must be positive). The same per-path convention applies to every branch, including both `double_leaf` (K24) branches. |
| `frequency` | Frequency at which `Kij` is evaluated, in Hz; only the `"flexible_t"` (through/corner) and the E.7/E.8 lightweight double-leaf junctions are frequency dependent. Defaults to 500 Hz, the value used by the simplified model (Clause 4.4.2), at which the E.7/E.8 $\log_{10}(f/f_k)$ terms vanish. |
| `f1` | Interlayer characteristic frequency for `"flexible_t"`, in Hz. |

**Returns:** `Kij`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `mass_ratio` is not positive, `frequency`/`f1` are not positive, an unknown `junction_type`/`path` is given, the requested path does not exist for the junction, or the E.7 double-leaf branch is requested outside its $m_2/m_1 > 3$ validity. |

## PathContribution

```python
PathContribution(label: str, kind: str, r_w: float, fraction: float)
```

A transmission path with its share of the total transmitted energy.

**Attributes**

| Name | Description |
| :--- | :--- |
| `label` | Path name (`"Dd"` for the direct path). |
| `kind` | `"Dd"`, `"Ff"`, `"Df"` or `"Fd"`. |
| `r_w` | Weighted sound reduction index of the path, in dB. |
| `fraction` | Fraction of the total transmitted sound energy carried by this path (0 to 1); the dominant path has the largest fraction. |

## predicted_airborne_insulation

```python
predicted_airborne_insulation(
    *,
    r_direct: float,
    flanking_paths: Sequence[FlankingPath] = (),
    delta_r_direct: float = 0.0,
) -> AirbornePredictionResult
```

Predict the apparent airborne insulation `R'w` (EN 12354-1 Formula 26).

Energetically combines the direct path
$R_{Dd,w} = R_{s,w} + \Delta R_{Dd,w}$ (Formula 27, from
`r_direct` and `delta_r_direct`) with the supplied flanking paths:

$$
R'_w = -10 \log_{10}\!\left[ 10^{-R_{Dd,w}/10} + \sum 10^{-R_{ij,w}/10} \right]
$$

With no flanking paths the result equals the direct path `RDd,w`; each
added path strictly lowers `R'w`. The result exposes every path's share of
the transmitted energy so the dominant path is visible.

**Parameters**

| Name | Description |
| :--- | :--- |
| `r_direct` | Weighted sound reduction index of the separating element `Rs,w`, in dB. |
| `flanking_paths` | Flanking paths (see [`flanking_element`](/phonometry/reference/api/building/global-model/#flanking_element)). May be empty for the direct-only case. |
| `delta_r_direct` | Combined lining improvement `ΔRDd,w` on the separating element, in dB. |

**Returns:** The [`AirbornePredictionResult`](/phonometry/reference/api/building/global-model/#airbornepredictionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any input is non-finite. |

## predicted_impact_insulation

```python
predicted_impact_insulation(
    *,
    ln_w_eq: float,
    delta_l_w: float = 0.0,
    k_correction: float = 0.0,
) -> ImpactPredictionResult
```

Predict the apparent impact insulation `L'n,w` (EN 12354-2 Formula 21).

$L'_{n,w} = L_{n,w,eq} - \Delta L_w + K$. The bare-floor equivalent
level may come from
[`equivalent_impact_level`](/phonometry/reference/api/building/global-model/#equivalent_impact_level) and the flanking correction from
[`impact_flanking_correction`](/phonometry/reference/api/building/global-model/#impact_flanking_correction).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ln_w_eq` | Bare-floor equivalent weighted level `Ln,w,eq`, in dB. |
| `delta_l_w` | Weighted covering improvement `ΔLw` (ISO 717-2), in dB. |
| `k_correction` | Flanking correction `K` (Table 1), in dB. |

**Returns:** The [`ImpactPredictionResult`](/phonometry/reference/api/building/global-model/#impactpredictionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any input is non-finite. |

## standardized_impact_level

```python
standardized_impact_level(l_prime_n_w: float, volume: float) -> float
```

Standardized apparent impact level `L'nT,w` (EN 12354-2 Formula 3).

$$
L'_{nT,w} = L'_{n,w} - 10 \log_{10}\frac{0.16\,V}{A_0 T_0} = L'_{n,w} - 10 \log_{10}(0.032\,V)
$$

with $A_0 = 10$ m² and $T_0 = 0.5$ s, the exact Formula (3)
form. The standard's own Annex E.3 worked example rounds the factor to
$10 \log_{10}(V/30)$ ($1/0.032 = 31.25 \approx 30$), 0.18 dB below
the exact form; both round to the same integer rating in E.3.

**Parameters**

| Name | Description |
| :--- | :--- |
| `l_prime_n_w` | Apparent weighted normalized impact level `L'n,w`, dB. |
| `volume` | Receiving-room volume `V`, in m³ (must be positive). |

**Returns:** `L'nT,w`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `volume` is not positive. |

## standardized_level_difference

```python
standardized_level_difference(
    r_prime_w: float,
    volume: float,
    separating_area: float,
) -> float
```

Standardized level difference `DnT,w` from `R'w` (EN 12354-1 Formula 5b).

$$
D_{nT} = R' + 10 \log_{10}\frac{0.16\,V}{T_0 S_s} = R' + 10 \log_{10}\frac{0.32\,V}{S_s}
$$

with $T_0 = 0.5$ s, the exact Formula (5b) form, applied to the
weighted single numbers of the simplified model (Clause 4.4). The Annex
H.3 worked example rounds the factor to $10 \log_{10}(V/(3 S_s))$
($1/0.32 = 3.125 \approx 3$), printing $52.2 + 1.6 = 53.8$ dB
where the exact form gives 53.6 dB;
both round to the same $D_{nT,w} = 54$ dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `r_prime_w` | Apparent weighted sound reduction index `R'w`, in dB (see [`predicted_airborne_insulation`](/phonometry/reference/api/building/global-model/#predicted_airborne_insulation)). |
| `volume` | Receiving-room volume `V`, in m³ (must be positive). |
| `separating_area` | Separating-element area `Ss`, in m² (must be positive). |

**Returns:** `DnT,w`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `volume` or `separating_area` is not positive. |
