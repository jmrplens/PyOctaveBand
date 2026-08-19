---
title: "materials.resilient.dynamic_stiffness"
description: "Dynamic stiffness of resilient materials under floating floors (EN 29052-1:1992)."
sidebar:
  label: "dynamic_stiffness"
---

Dynamic stiffness of resilient materials under floating floors (EN 29052-1:1992).

A floating floor is a heavy floating slab resting on a resilient layer; the
combination is a mass-spring system whose natural frequency governs the impact
and airborne improvement of the floor. EN 29052-1 (identical to ISO 9052-1:1989)
measures the **dynamic stiffness per unit area** `s'` of the resilient layer
from the resonance of a standard load plate on a 200 mm x 200 mm specimen.

The dynamic stiffness per unit area is the ratio of a dynamic force per area to
the resulting change in thickness (Formula 1):

$$
s' = \frac{F/S}{\Delta d} \qquad [\text{N/m}^3]
$$

The resiliently supported floor is a mass-spring resonator; its natural
frequency (Formula 2) and, in the laboratory arrangement, the measured resonant
frequency (Formula 3) are:

$$
f_0 = \frac{1}{2\pi} \sqrt{\frac{s'}{m'}} \qquad \text{(installed floor)}
$$

$$
f_\mathrm{r} = \frac{1}{2\pi} \sqrt{\frac{s'_\mathrm{t}}{m'_\mathrm{t}}} \qquad \text{(test arrangement)}
$$

so the *apparent* dynamic stiffness follows from the resonance (Formula 4):

$$
s'_\mathrm{t} = 4 \pi^2 m'_\mathrm{t} f_\mathrm{r}^2
$$

With an air-permeable resilient material the enclosed gas adds a parallel
stiffness (Formula 7), from the isothermal compression of the pore air:

$$
s'_\mathrm{a} = \frac{p_0}{d\,\epsilon}
$$

($s'_\mathrm{a} = 111/d$ MN/m3 for $p_0 = 0.1$ MPa,
$\epsilon = 0.9$ and `d` in mm, the standard's worked NOTE). The
dynamic stiffness of the installed material is then obtained by airflow
resistivity `r` (clause 8.2):

$$
s' = s'_\mathrm{t}, \qquad r \ge 100~\text{kPa}\cdot\text{s/m}^2 \tag{Formula 5}
$$

$$
s' = s'_\mathrm{t} + s'_\mathrm{a}, \qquad 10 \le r < 100~\text{kPa}\cdot\text{s/m}^2 \tag{Formula 6}
$$

For $r < 10$ kPa.s/m2, `s'a` follows Formula 7; the method only
applies when $s'_\mathrm{t} \gg s'_\mathrm{a}$, otherwise `s'` cannot be resolved.

This module is the resilient-layer characterisation feeding the floating-floor
term of the EN 12354-2 impact model
([`phonometry.building.prediction.simplified_model`](/phonometry/reference/api/building/simplified-model/)). It does **not** feed
ISO 16251-1 ([`phonometry.building.measurement.floor_covering_improvement`](/phonometry/reference/api/building/floor-covering-improvement/)), whose
scope is limited to soft, locally-reacting floor coverings; floating floors
are explicitly excluded there.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## apparent_dynamic_stiffness

```python
apparent_dynamic_stiffness(
    resonant_frequency: ArrayLike,
    total_mass_per_area: float,
) -> np.ndarray | float
```

Apparent dynamic stiffness per unit area `s't` (Formula 4).

Inverts the test resonance $f_\mathrm{r} = (1/2\pi)\sqrt{s'_\mathrm{t}/m'_\mathrm{t}}$ to
$s'_\mathrm{t} = 4 \pi^2 m'_\mathrm{t} f_\mathrm{r}^2$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `resonant_frequency` | Extrapolated resonant frequency `fr`, in hertz (scalar or array). |
| `total_mass_per_area` | Total mass per unit area used during the test `m't`, in kg/m2 (the load plate plus fittings over the 0,04 m2 specimen; the standard's plate gives $m'_\mathrm{t} = 8~\text{kg} / 0.04~\text{m}^2 = 200$ kg/m2). |

**Returns:** The apparent dynamic stiffness per unit area `s't`, in N/m3 (numerically MN/m3 when divided by 1e6).

## DynamicStiffnessResult

```python
DynamicStiffnessResult(
    apparent_stiffness: float,
    gas_stiffness: float,
    dynamic_stiffness: float,
    resonant_frequency: float,
    floor_mass_per_area: float,
    natural_frequency: float,
)
```

Dynamic stiffness of a resilient layer and the floating-floor resonance.

**Attributes**

| Name | Description |
| :--- | :--- |
| `apparent_stiffness` | Apparent dynamic stiffness `s't`, in N/m3. |
| `gas_stiffness` | Enclosed-gas dynamic stiffness `s'a`, in N/m3. |
| `dynamic_stiffness` | Installed dynamic stiffness `s'`, in N/m3. |
| `resonant_frequency` | Measured test resonant frequency `fr`, in hertz. |
| `floor_mass_per_area` | Supported-floor mass per unit area `m'`, kg/m2. |
| `natural_frequency` | Installed-floor natural frequency `f0`, in hertz. |

### DynamicStiffnessResult.plot()

```python
DynamicStiffnessResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `f0(s')` with this design point marked.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### DynamicStiffnessResult.report()

```python
DynamicStiffnessResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an EN 29052-1 dynamic-stiffness test-report fiche to a PDF.

Writes a one-page accredited dynamic-stiffness report (EN 29052-1:1992,
identical to ISO 9052-1:1989): the standard-basis line, an optional
metadata header block (client, specimen, the total mass per unit area
`m't` used during the test, the loaded specimen thickness `d`, test
facility, date, climate ...), a two-panel body with a compact metrics
table (the resonant frequency `fr`, the apparent dynamic stiffness
`s't` of Formula 4, the enclosed-gas term `s'a` of Formula 7 when it
applies, the installed dynamic stiffness `s'` of Clause 8.2 and the
supported-floor natural frequency `f0` of Formula 2) beside the
`f0(s')` design curve, a boxed apparent dynamic stiffness `s't` with
the installed `s'` and the resonance `fr` alongside, and a footer
with the fixed disclaimer. EN 29052-1 is a characterisation, so there is
no pass/fail verdict.

Clause 9 requires every dynamic stiffness per unit area to be stated in
meganewtons per cubic metre to the nearest meganewton per cubic metre,
so the stiffness values are rounded to the nearest MN/m3; the
frequencies are shown to 0,1 Hz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a body-and-disclaimer fiche. The applicable descriptive fields are `client`, `manufacturer`, `specimen`, `mass_per_area` (the total mass per unit area `m't`), `thickness` (the loaded specimen thickness `d`, in metres, shown in millimetres), `test_room`, `test_date`, `temperature`, `relative_humidity`, `measurement_standard`, `laboratory`, `operator`, `report_id` and `notes`. The `requirement` field is ignored (EN 29052-1 has no verdict). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform `.report()` signature; the dynamic-stiffness fiche has a single body layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English, decimal point) or `"es"` (Spanish, decimal comma). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab or matplotlib is not installed. The fiche always embeds the `f0(s')` design curve, so both are required (`pip install "phonometry[report,plot]"`). |

## DynamicStiffnessWarning

Advisory when the enclosed-gas term makes `s'` unresolvable (clause 8.2).

## enclosed_gas_stiffness

```python
enclosed_gas_stiffness(
    thickness: ArrayLike,
    porosity: float,
    *,
    atmospheric_pressure: float = 100000.0,
) -> np.ndarray | float
```

Enclosed-gas dynamic stiffness per unit area `s'a` (Formula 7).

The isothermal compression of the pore air adds a stiffness in parallel
with the material's structure: $s'_\mathrm{a} = p_0 / (d\,\epsilon)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `thickness` | Thickness `d` of the specimen under the static load, in **metres** (scalar or array). |
| `porosity` | Porosity `epsilon` of the specimen (0-1). |
| `atmospheric_pressure` | Atmospheric pressure `p0`, in pascals (default `STANDARD_ATMOSPHERIC_PRESSURE`, the standard's 0,1 MPa). |

**Returns:** The enclosed-gas dynamic stiffness per unit area `s'a`, in N/m3.

:::note
With the standard's $p_0 = 0.1$ MPa and $\epsilon = 0.9$
this reduces to $s'_\mathrm{a} = 111/d$ MN/m3 for `d` in millimetres
(clause 8.2 NOTE).
:::

## floating_floor_resonance

```python
floating_floor_resonance(
    resonant_frequency: float,
    total_mass_per_area: float,
    floor_mass_per_area: float,
    *,
    airflow_resistivity: float = inf,
    thickness: float | None = None,
    porosity: float | None = None,
    atmospheric_pressure: float = 100000.0,
) -> DynamicStiffnessResult
```

Full EN 29052-1 chain: measured resonance -> installed `s'` and `f0`.

Chains the apparent dynamic stiffness (Formula 4), the enclosed-gas term
(Formula 7, when `thickness` and `porosity` are given), the airflow
resistivity combination (clause 8.2) and the installed-floor natural
frequency (Formula 2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `resonant_frequency` | Measured resonant frequency `fr`, in hertz. |
| `total_mass_per_area` | Test total mass per unit area `m't`, kg/m2. |
| `floor_mass_per_area` | Supported-floor mass per unit area `m'`, kg/m2. |
| `airflow_resistivity` | Lateral airflow resistivity `r`, in kPa.s/m2 (default `inf` -> the high-resistivity case $s' = s'_\mathrm{t}$). |
| `thickness` | Specimen thickness `d` under load, in metres. Required together with `porosity` for the enclosed-gas term, which applies when $r < 100$ kPa.s/m2. That condition is on the *value* of `airflow_resistivity` rather than on a literal, so a signature cannot state it: it is checked here and raises. |
| `porosity` | Specimen porosity `epsilon`, required with `thickness` (see above). |
| `atmospheric_pressure` | Atmospheric pressure `p0`, in pascals. |

**Returns:** The [`DynamicStiffnessResult`](/phonometry/reference/api/materials/dynamic-stiffness/#dynamicstiffnessresult).

## installed_dynamic_stiffness

```python
installed_dynamic_stiffness(
    apparent_stiffness: float,
    airflow_resistivity: float,
    *,
    gas_stiffness: float = 0.0,
) -> float
```

Dynamic stiffness per unit area `s'` of the installed material (clause 8.2).

Combines the apparent stiffness with the enclosed-gas term according to the
lateral airflow resistivity `r`:

* $r \ge 100$ kPa.s/m2 -> $s' = s'_\mathrm{t}$ (Formula 5);
* $10 \le r < 100$ kPa.s/m2 -> $s' = s'_\mathrm{t} + s'_\mathrm{a}$
  (Formula 6);
* $r < 10$ kPa.s/m2 -> the standard only requires the qualitative
  criterion $s'_\mathrm{t} \gg s'_\mathrm{a}$ (clause 8.2). This implementation
  applies its own engineering threshold: `s'a` below 10 % of `s't` is treated as
  negligible and $s' = s'_\mathrm{t}$ (a [`DynamicStiffnessWarning`](/phonometry/reference/api/materials/dynamic-stiffness/#dynamicstiffnesswarning) is
  emitted; clause 8.2 requires the error caused by disregarding `s'a` to
  be stated in the test report); above it the result is `nan`, as the
  method cannot resolve `s'`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `apparent_stiffness` | Apparent dynamic stiffness `s't`, in N/m3. |
| `airflow_resistivity` | Lateral airflow resistivity `r`, in kPa.s/m2 (ISO 9053). |
| `gas_stiffness` | Enclosed-gas dynamic stiffness `s'a`, in N/m3 (see [`enclosed_gas_stiffness`](/phonometry/reference/api/materials/dynamic-stiffness/#enclosed_gas_stiffness)); needed for $r < 100$ kPa.s/m2. |

**Returns:** The installed dynamic stiffness per unit area `s'`, in N/m3 (`nan` when the method cannot resolve it).

## natural_frequency

```python
natural_frequency(
    dynamic_stiffness: ArrayLike,
    mass_per_area: float,
) -> np.ndarray | float
```

Natural frequency `f0` of the resiliently supported floor (Formula 2).

$f_0 = (1/2\pi)\sqrt{s'/m'}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `dynamic_stiffness` | Dynamic stiffness per unit area `s'`, in N/m3 (scalar or array). |
| `mass_per_area` | Mass per unit area of the supported floor `m'`, in kg/m2. |

**Returns:** The natural frequency `f0`, in hertz.

## plot_dynamic_stiffness_rig

```python
plot_dynamic_stiffness_rig(
    ax: Axes | None = None,
    *,
    specimen_side: float = 0.2,
    specimen_thickness: float = 0.02,
    load_mass: float = 8.0,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the dynamic-stiffness resonance rig to scale.

Resilient specimen on the rigid base, the standard square load plate on
top (its mass annotated), the exciter above and an accelerometer on the
plate; defaults are the standard 200 mm square specimen under the 8 kg
plate.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `specimen_side` | Specimen side length, in metres. |
| `specimen_thickness` | Specimen thickness, in metres. |
| `load_mass` | Load-plate mass, in kilograms (annotation). |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the specimen rectangle. |

**Returns:** The axes.
