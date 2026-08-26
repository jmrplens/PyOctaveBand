---
title: "materials.absorbers.impedance_tube"
description: "Two-microphone transfer-function method in the impedance tube."
sidebar:
  label: "impedance_tube"
---

Two-microphone transfer-function method in the impedance tube.

**BS EN ISO 10534-2:2001**: the complex reflection factor `r` at the sample
surface is obtained from the measured transfer function `H12` between two
microphones flush-mounted in the wall of a tube terminated by the specimen,
and from it the surface impedance and the normal-incidence absorption
coefficient (Clause 7, Eqs. (17)-(20)). Time convention
$e^{+j\omega t}$; the incident wave carries $e^{+jk_0x}$ and the
reflected wave $e^{-jk_0x}$ (Annex D, Eqs. (D.1)-(D.8)). The complex
wavenumber is $k_0 = k_0' - jk_0''$ with the attenuation constant
$k_0''$ (Clause 2.6, Annex A). Air properties from Clause 7.2,
Eqs. (5)/(7), use temperature in **kelvin**.

The tube itself is described here as well - its cross-section, the hydraulic
diameter a rectangular tube reports, the complex wavenumber, the lower-bound
wall attenuation and the plane-wave working range - because a specimen is
characterised only where the field in the tube is a plane wave, and because
the four-microphone method measures in the same tube and reuses that
arithmetic.

The other two standardised impedance-tube methods are their own modules, each
kept in its own sign convention (they are **not** interchangeable):
[`standing_wave`](/phonometry/reference/api/materials/standing-wave/) for the probe-traverse
standing-wave-ratio method of BS EN ISO 10534-1:2001, and
[`four_microphone`](/phonometry/reference/api/materials/four-microphone/) for the transmission
transfer-matrix method of ASTM E2611-19, whose air properties are given in
degrees Celsius and whose forward wave carries the opposite exponent sign.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## absorption_from_reflection

```python
absorption_from_reflection(reflection: ArrayLike) -> Real
```

Normal-incidence absorption coefficient (ISO 10534-2, Eq. (18)).

$\alpha = 1 - |r|^2$. This form is shared with ISO 10534-1 Eq. (9) and
ASTM E2611-19 Eq. (28).

**Parameters**

| Name | Description |
| :--- | :--- |
| `reflection` | Complex reflection factor `r`. |

**Returns:** Absorption coefficient `alpha` (real).

## air_density_iso

```python
air_density_iso(
    temperature: ArrayLike,
    atmospheric_pressure: ArrayLike = 101.325,
) -> Real
```

Air density (ISO 10534-2:2001, Eq. (7)).

$\rho = \rho_0 (p_\mathrm{a} T_0) / (p_0 T)$ with $\rho_0 = 1.186$
kg/m3, $T_0 = 293$ K and $p_0 = 101.325$ kPa.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Air temperature `T`, in **kelvin**. |
| `atmospheric_pressure` | Atmospheric pressure `pa`, in kilopascals (default 101,325 kPa). |

**Returns:** Air density `rho`, in kilograms per cubic metre.

## apply_mic_calibration

```python
apply_mic_calibration(
    h12_uncorrected: ArrayLike,
    calibration_factor: ArrayLike,
) -> Complex
```

Apply the microphone calibration factor (ISO 10534-2, Eq. (13)).

$H_{12} = H_{12,\text{uncorrected}} / H_\mathrm{c}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `h12_uncorrected` | Uncorrected measured transfer function. |
| `calibration_factor` | Calibration factor `Hc` from [`mic_calibration_factor`](/phonometry/reference/api/materials/impedance-tube/#mic_calibration_factor). |

**Returns:** Corrected transfer function `H12`.

## characteristic_impedance

```python
characteristic_impedance(density: float, speed_of_sound: float) -> float
```

Characteristic impedance of air `rho c` (rayls).

A convenience for both standards (ISO 10534-2 Clause 7.2; ASTM E2611-19
Clause 8.2/8.3): the real product of air density and speed of sound.

**Parameters**

| Name | Description |
| :--- | :--- |
| `density` | Air density `rho`, in kg/m3. |
| `speed_of_sound` | Speed of sound `c`, in m/s. |

**Returns:** Characteristic impedance `rho c`, in rayls.

## hydraulic_diameter

```python
hydraulic_diameter(width: float, height: float) -> float
```

Hydraulic diameter of a rectangular tube, $4A/P$ (ISO 10534-2, A.2.1.5).

For a rectangular cross-section of side lengths `w` and `h` the ratio
of four times the area to the perimeter reduces to
$d_\mathrm{h} = 2wh/(w + h)$; a square tube gives `d_h` equal to the side
length. This is the `d` the Eq. (A.18) attenuation estimate expects for
rectangular tubes (see [`tube_attenuation_constant`](/phonometry/reference/api/materials/impedance-tube/#tube_attenuation_constant)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `width` | Inner side length `w`, in metres. |
| `height` | Inner side length `h`, in metres. |

**Returns:** Hydraulic diameter $d_\mathrm{h} = 4A/P$, in metres.

## ImpedanceTubeResult

```python
ImpedanceTubeResult(
    frequency: Real,
    reflection: Complex,
    surface_impedance: Complex,
    normalized_impedance: Complex,
    absorption: Real,
    spacing: float | None = None,
    x1: float | None = None,
    diameter: float | None = None,
    shape: str | None = None,
)
```

Two-microphone impedance-tube result (ISO 10534-2:2001).

All arrays share the shape of `frequency`. `reflection` is the complex
reflection factor `r` at the sample surface (Eq. (17)),
`surface_impedance` the absolute surface impedance `Z` in rayls
(Eq. (19)), `normalized_impedance` the ratio $Z/(\rho c_0)$
(Eq. (19)) and `absorption` the normal-incidence coefficient
$\alpha = 1 - \lvert r\rvert^2$ (Eq. (18)).

The trailing fields retain the tube geometry the reduction was run with
(microphone `spacing` `s`, distance `x1` from the sample to the
farther microphone, tube `diameter` and cross-section `shape`, stored
canonically as `"circular"`/`"rectangular"` - a `"square"` input is
kept as `"rectangular"`); they default to `None` when not supplied to
[`two_microphone_impedance`](/phonometry/reference/api/materials/impedance-tube/#two_microphone_impedance).

### ImpedanceTubeResult.plot()

```python
ImpedanceTubeResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the absorption spectrum `alpha(f)` with `|r|` overlaid.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ImpedanceTubeResult.plot_geometry()

```python
ImpedanceTubeResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the two-microphone tube to scale (dimensioned side view).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its tube geometry (`spacing`/`x1`). |

### ImpedanceTubeResult.report()

```python
ImpedanceTubeResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 10534-2 impedance-tube test-report fiche to a PDF.

Writes a one-page accredited normal-incidence report (BS EN ISO
10534-2:2001, two-microphone transfer-function method): the
standard-basis line, an optional metadata header block (client,
specimen, tube diameter `d`, microphone spacing `s`, the measured
frequency range, mounting, climate ...), a two-panel body with the
per-frequency table (frequency, absorption `alpha` and the
real/imaginary parts of the normalised surface impedance
$z = Z/(\rho c_0)$) beside the `alpha(f)` curve, and a footer with
the fixed disclaimer. ISO 10534-2 is a characterisation, so there is no
pass/fail verdict and no single-number rating (the random-incidence
weighted `alpha_w` is an ISO 11654 / ISO 354 quantity, not comparable
to the normal-incidence coefficient reported here).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a body-and-disclaimer fiche whose header shows only the measured frequency range. The applicable descriptive/geometric fields are `client`, `manufacturer`, `specimen`, `tube_diameter`, `tube_shape`, `mic_spacing`, `mounting`, `test_room`, `test_date`, `temperature`, `pressure`, `measurement_standard`, `laboratory`, `operator`, `report_id` and `notes`. The `requirement` field is ignored (ISO 10534-2 has no verdict). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the value table inserts the reflection-factor magnitude `\|r\|` column. |
| `language` | Fiche language: `"en"` (default, English, decimal point) or `"es"` (Spanish, decimal comma). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## ImpedanceTubeWarning

Advisory for out-of-plane-wave-range impedance-tube frequencies.

## mic_calibration_factor

```python
mic_calibration_factor(
    h12_config1: ArrayLike,
    h12_config2: ArrayLike,
) -> Complex
```

Microphone-mismatch calibration factor `Hc` (ISO 10534-2, Eq. (10)).

$H_\mathrm{c} = \sqrt{H_{12}^{I} / H_{12}^{II}}$ from a transfer function measured on an
absorptive specimen in the standard configuration (I) and with the two
microphones physically interchanged (II) - the cabling to the analyser is
**not** swapped (Clause 7.5.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `h12_config1` | Transfer function `H12^I` in the standard configuration. |
| `h12_config2` | Transfer function `H12^II` with microphones swapped. |

**Returns:** Complex calibration factor `Hc`.

## normalized_surface_admittance

```python
normalized_surface_admittance(reflection: ArrayLike) -> Complex
```

Normalised surface admittance $G \rho c_0$ (ISO 10534-2, Eq. (20)).

$G \rho c_0 = (\rho c_0) / Z = (1 - r) / (1 + r)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reflection` | Complex reflection factor `r`. |

**Returns:** Normalised surface admittance (complex).

## normalized_surface_impedance

```python
normalized_surface_impedance(reflection: ArrayLike) -> Complex
```

Normalised surface impedance $Z/(\rho c_0)$ (ISO 10534-2, Eq. (19)).

$Z / (\rho c_0) = (1 + r) / (1 - r)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reflection` | Complex reflection factor `r`. |

**Returns:** Normalised surface impedance $Z/(\rho c_0)$ (complex).

## plane_wave_frequency_range

```python
plane_wave_frequency_range(
    spacing: float,
    speed_of_sound: float,
    *,
    diameter: float | None = None,
    shape: str = 'circular',
) -> tuple[float, float]
```

Working plane-wave frequency range `(f_l, f_u)` (ISO 10534-2, 4.2-4.5).

The upper limit is the smaller of the microphone-spacing bound
$f_\mathrm{u} s < 0.45 c_0$ (Eq. (4)) and, when the tube `diameter` is
given, the cut-on bound $f_\mathrm{u} d < 0.58 c_0$ for a circular tube
(Eq. (2)) or $f_\mathrm{u} d < 0.50 c_0$ for a rectangular tube (Eq. (3)).
The lower limit uses the Clause 4.2 guideline that the spacing exceed
5 % of the wavelength, i.e. $f_\mathrm{l} = c_0 / (20 s)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `spacing` | Microphone spacing `s`, in metres. |
| `speed_of_sound` | Speed of sound `c0`, in metres per second. |
| `diameter` | Tube diameter (circular) or maximum lateral dimension (rectangular/square) `d`, in metres; `None` applies only the spacing bound. |
| `shape` | `"circular"`, `"rectangular"` or `"square"` (a square tube is the rectangular bound with `d` the side length). |

**Returns:** Tuple `(f_l, f_u)` of the lower and upper frequency limits, in Hz.

## plot_impedance_tube_geometry

```python
plot_impedance_tube_geometry(
    ax: Axes | None = None,
    *,
    spacing: float,
    x1: float,
    diameter: float | None = None,
    shape: str | None = 'circular',
    sample_thickness: float | None = None,
    speed_of_sound: float = 343.2,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the ISO 10534-2 two-microphone tube to scale.

Side view: loudspeaker at the left (three tube diameters before the
farther microphone, the Clause 4.3 margin), the two flush microphones at
`x1` and `x1 - s` from the sample face, the sample against its rigid
backing at the right, the cross-section emblem, and the plane-wave
working range of [`plane_wave_frequency_range`](/phonometry/reference/api/materials/impedance-tube/#plane_wave_frequency_range).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `spacing` | Microphone spacing `s`, in metres. |
| `x1` | Distance from the sample face to the farther microphone, in metres. |
| `diameter` | Inner diameter (circular) or lateral dimension (rectangular/square), in metres; `None` draws a nominal bore and omits the bore dimension and the cut-on bound. |
| `shape` | `"circular"`, `"rectangular"`, `"square"` or `None`. |
| `sample_thickness` | Drawn sample thickness, in metres; `None` draws a 50 mm nominal sample. |
| `speed_of_sound` | Speed of sound for the working range, in m/s. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the tube-bore rectangle. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a spacing, distance, diameter or sample thickness that is not finite and positive, or an `x1` no greater than the spacing. |

## plot_transmission_tube_geometry

```python
plot_transmission_tube_geometry(
    ax: Axes | None = None,
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    diameter: float | None = None,
    shape: str | None = 'circular',
    speed_of_sound: float = 343.2,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the ASTM E2611 four-microphone transmission tube to scale.

Side view: loudspeaker, the two upstream microphones at `l1` and
`l1 + s1` from the front face of the specimen, the specimen spanning
its thickness, the two downstream microphones at `l2` and `l2 + s2`
(measured from the front face, the module's locked convention), and the
changeable termination of the two-load method, with the ASTM working
range of
[`plane_wave_frequency_range_astm`](/phonometry/reference/api/materials/four-microphone/#plane_wave_frequency_range_astm).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `l1` | Front face to the nearer upstream microphone, in metres. |
| `s1` | Upstream microphone spacing, in metres. |
| `l2` | Front face to the nearer downstream microphone, in metres. |
| `s2` | Downstream microphone spacing, in metres. |
| `thickness` | Specimen thickness, in metres; must be smaller than `l2` (the downstream microphones sit past the back face). |
| `diameter` | Inner diameter (circular) or largest section dimension (rectangular/square), in metres; `None` draws a nominal bore and omits the bore dimension and the cut-on bound. |
| `shape` | `"circular"`, `"rectangular"`, `"square"` or `None`. |
| `speed_of_sound` | Speed of sound for the working range, in m/s. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the tube-bore rectangle. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | naming the first length or diameter that is not finite and positive, or an `l2` no greater than the thickness. |

## reflection_factor

```python
reflection_factor(
    h12: ArrayLike,
    *,
    spacing: float,
    x1: float,
    wavenumber: ArrayLike,
) -> Complex
```

Complex reflection factor at the sample surface (ISO 10534-2, Eq. (17)).

$$
r = \frac{H_{12} - H_\mathrm{I}}{H_\mathrm{R} - H_{12}} \, e^{+2jk_0x_1}
$$

with the incident- and reflected-wave transfer functions
$H_\mathrm{I} = e^{-jk_0s}$ (Eq. (D.5)) and $H_\mathrm{R} = e^{+jk_0s}$
(Eq. (D.6)), `s` the microphone spacing and `x1` the distance from
the sample to the **farther** microphone (Clause 7.7).

**Parameters**

| Name | Description |
| :--- | :--- |
| `h12` | Measured transfer function `H12` between microphone positions 1 and 2 (Clause 7.6, Eq. (14)); complex, scalar or per band. It must already be corrected for microphone mismatch (see [`apply_mic_calibration`](/phonometry/reference/api/materials/impedance-tube/#apply_mic_calibration)). |
| `spacing` | Microphone spacing $s = x_1 - x_2$, in metres. |
| `x1` | Distance from the sample surface to the farther microphone (position 1), in metres. |
| `wavenumber` | Complex wavenumber `k0` (from [`tube_wavenumber`](/phonometry/reference/api/materials/impedance-tube/#tube_wavenumber)), scalar or per band. |

**Returns:** Complex reflection factor `r` at the reference plane.

## speed_of_sound_iso

```python
speed_of_sound_iso(temperature: ArrayLike) -> Real
```

Speed of sound in air (ISO 10534-2:2001, Eq. (5)).

$c_0 = 343.2 \sqrt{T / 293}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Air temperature `T`, in **kelvin**. |

**Returns:** Speed of sound `c0`, in metres per second.

## surface_impedance

```python
surface_impedance(
    reflection: ArrayLike,
    characteristic_impedance: float,
) -> Complex
```

Absolute surface impedance `Z` (ISO 10534-2, Eq. (19)).

$Z = \rho c_0 (1 + r) / (1 - r)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reflection` | Complex reflection factor `r`. |
| `characteristic_impedance` | Characteristic impedance of air `rho c0`, in rayls (`rho` and `c0` from the Clause 7.2 helpers). |

**Returns:** Surface impedance `Z`, in rayls (complex).

## tube_attenuation_constant

```python
tube_attenuation_constant(
    frequency: ArrayLike,
    speed_of_sound: float,
    diameter: float,
) -> Real
```

Lower-bound tube attenuation constant `k0''` (ISO 10534-2, Eq. (A.18)).

$k_0'' = 1.94\times 10^{-2} \sqrt{f} / (c_0 d)$
(nepers per metre). This ignores
porous-wall and object losses and is therefore a lower limit (Clause A.2.1.5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or per band). |
| `speed_of_sound` | Speed of sound `c0`, in metres per second. |
| `diameter` | Circular-tube diameter `d`, in metres, or the hydraulic diameter `4 * area / perimeter` for a rectangular tube (see [`hydraulic_diameter`](/phonometry/reference/api/materials/impedance-tube/#hydraulic_diameter)). |

**Returns:** Attenuation constant `k0''`, in nepers per metre.

## tube_wavenumber

```python
tube_wavenumber(
    frequency: ArrayLike,
    speed_of_sound: float,
    *,
    attenuation: ArrayLike | None = None,
) -> Complex
```

Complex wavenumber $k_0 = k_0' - jk_0''$ (ISO 10534-2, Clause 2.6).

The real part is $k_0' = 2\pi f/c_0$ (Eq. (2)); the optional attenuation
constant `k0''` enters with a **minus** sign on the imaginary part
(Clause 2.6 NOTE, Eq. (A.1)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or per band). |
| `speed_of_sound` | Speed of sound `c0`, in metres per second. |
| `attenuation` | Attenuation constant `k0''`, in nepers per metre (scalar or matching `frequency`); `None` gives the lossless real wavenumber. Obtain a lower-bound estimate from [`tube_attenuation_constant`](/phonometry/reference/api/materials/impedance-tube/#tube_attenuation_constant). |

**Returns:** Complex wavenumber `k0`, in reciprocal metres.

## two_microphone_impedance

```python
two_microphone_impedance(
    h12: ArrayLike,
    *,
    frequency: ArrayLike,
    spacing: float,
    x1: float,
    speed_of_sound: float,
    characteristic_impedance: float,
    attenuation: ArrayLike | None = None,
    diameter: float | None = None,
    shape: str = 'circular',
) -> ImpedanceTubeResult
```

Full two-microphone reduction (ISO 10534-2:2001, Clause 7).

Builds the complex wavenumber (Clause 2.6), the reflection factor
(Eq. (17)), the surface impedance (Eq. (19)) and the absorption coefficient
(Eq. (18)) from the measured transfer function `H12`. When `diameter` is
supplied, frequencies outside the plane-wave range (Eqs. (1)-(4)) raise an
[`ImpedanceTubeWarning`](/phonometry/reference/api/materials/impedance-tube/#impedancetubewarning); the results are still returned.

**Parameters**

| Name | Description |
| :--- | :--- |
| `h12` | Measured (mismatch-corrected) transfer function `H12`. |
| `frequency` | Frequency vector `f`, in hertz. |
| `spacing` | Microphone spacing `s`, in metres. |
| `x1` | Distance from the sample to the farther microphone, in metres. |
| `speed_of_sound` | Speed of sound `c0`, in m/s (see [`speed_of_sound_iso`](/phonometry/reference/api/materials/impedance-tube/#speed_of_sound_iso)). |
| `characteristic_impedance` | Characteristic impedance `rho c0`, in rayls. |
| `attenuation` | Optional tube attenuation constant `k0''`, in nepers/m (see [`tube_attenuation_constant`](/phonometry/reference/api/materials/impedance-tube/#tube_attenuation_constant)). |
| `diameter` | Optional tube diameter/lateral dimension, in metres, that activates the plane-wave range check. |
| `shape` | Tube cross-section, `"circular"`, `"rectangular"` or `"square"`. |

**Returns:** An [`ImpedanceTubeResult`](/phonometry/reference/api/materials/impedance-tube/#impedancetuberesult) (the tube geometry is retained on the result).
