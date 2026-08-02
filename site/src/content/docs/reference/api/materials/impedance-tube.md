---
title: "materials.absorbers.impedance_tube"
description: "Impedance-tube material characterisation."
sidebar:
  label: "impedance_tube"
---

Impedance-tube material characterisation.

Three complementary standardised methods are implemented, each kept in its
own sign convention (they are **not** interchangeable):

* **BS EN ISO 10534-2:2001** - two-microphone transfer-function method. The
  complex reflection factor `r` at the sample surface is obtained from the
  measured transfer function `H12` between two microphones, and from it the
  surface impedance and the normal-incidence absorption coefficient
  (Clause 7, Eqs. (17)-(20)). Time convention $e^{+j\omega t}$; the
  incident wave carries $e^{+jk_0x}$ and the reflected wave
  $e^{-jk_0x}$ (Annex D, Eqs. (D.1)-(D.8)). The complex wavenumber is
  $k_0 = k_0' - jk_0''$ with the attenuation constant $k_0''$
  (Clause 2.6, Annex A). Air properties from Clause 7.2, Eqs. (5)/(7), use
  temperature in **kelvin**.

* **BS EN ISO 10534-1:2001** - standing-wave-ratio method. The reflection
  magnitude, phase, absorption coefficient and normalised impedance follow
  from the measured standing-wave ratio and the position of the first pressure
  minimum (Clause 5, Eqs. (12)-(26)).

* **ASTM E2611-19** - four-microphone transfer-matrix method. The wave field
  is decomposed into forward/backward amplitudes on each side of the specimen
  (Eqs. (17)-(20)), the face pressures and particle velocities are formed
  (Eq. (21)) and the transfer matrix `[[T11, T12], [T21, T22]]` is solved
  from a two-load (Eq. (22)) or a symmetric one-load (Eq. (24)) measurement.
  Transmission loss (Eq. (26)), hard-backed reflection/absorption
  (Eqs. (27)/(28)) and the material wavenumber/characteristic impedance
  (Eqs. (29)/(30)) follow. Time convention $e^{+j\omega t}$ with the
  forward wave carried by $e^{-jkx}$ (Eq. (21)); air properties from
  Clause 8.2/8.3, Eqs. (4)/(5), use temperature in **degrees Celsius**.

The two standards adopt different sign ansaetze and different temperature
units on purpose; the helpers are named per standard so the two are never
mixed.

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

## air_density_astm

```python
air_density_astm(
    temperature: ArrayLike,
    atmospheric_pressure: ArrayLike = 101.325,
) -> Real
```

Air density (ASTM E2611-19, Eq. (5)).

$\rho = 1.290 \, \frac{P}{101.325} \, \frac{273.15}{273.15 + T}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Room temperature `T`, in **degrees Celsius**. |
| `atmospheric_pressure` | Atmospheric pressure `P`, in kilopascals (default 101,325 kPa). |

**Returns:** Air density `rho`, in kilograms per cubic metre.

## air_density_iso

```python
air_density_iso(
    temperature: ArrayLike,
    atmospheric_pressure: ArrayLike = 101.325,
) -> Real
```

Air density (ISO 10534-2:2001, Eq. (7)).

$\rho = \rho_0 (p_a T_0) / (p_0 T)$ with $\rho_0 = 1.186$
kg/m3, $T_0 = 293$ K and $p_0 = 101.325$ kPa.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Air temperature `T`, in **kelvin**. |
| `atmospheric_pressure` | Atmospheric pressure `pa`, in kilopascals (default 101,325 kPa). |

**Returns:** Air density `rho`, in kilograms per cubic metre.

## air_layer_transfer_matrix

```python
air_layer_transfer_matrix(
    wavenumber: ArrayLike,
    thickness: float,
    characteristic_impedance: float,
) -> TransferMatrix
```

Analytic transfer matrix of a pure air layer of thickness `d`.

$$
T = [[\cos(k d),\; j \rho c \sin(k d)],\; [j \sin(k d) / (\rho c),\; \cos(k d)]]
$$

the classical loss-free layer used to validate the ASTM E2611-19 reduction
(it is reciprocal, $\operatorname{det}(T) = 1$, and symmetric,
$T_{11} = T_{22}$).

**Parameters**

| Name | Description |
| :--- | :--- |
| `wavenumber` | Air wavenumber `k`. |
| `thickness` | Layer thickness `d`, in metres. |
| `characteristic_impedance` | Characteristic impedance `rho c`, in rayls. |

**Returns:** The air-layer [`TransferMatrix`](/phonometry/reference/api/materials/impedance-tube/#transfermatrix).

## apply_mic_calibration

```python
apply_mic_calibration(
    h12_uncorrected: ArrayLike,
    calibration_factor: ArrayLike,
) -> Complex
```

Apply the microphone calibration factor (ISO 10534-2, Eq. (13)).

$H_{12} = H_{12,\text{uncorrected}} / H_c$.

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

## face_quantities

```python
face_quantities(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    *,
    wavenumber: ArrayLike,
    thickness: float,
    characteristic_impedance: float,
) -> tuple[Complex, Complex, Complex, Complex]
```

Face pressures and particle velocities (ASTM E2611-19, Eq. (21)).

$$
p_0 = A + B, \qquad p_d = C e^{-jkd} + D e^{+jkd}
$$

$$
u_0 = \frac{A - B}{\rho c}, \qquad u_d = \frac{C e^{-jkd} - D e^{+jkd}}{\rho c}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `a` | Upstream forward amplitude `A`. |
| `b` | Upstream backward amplitude `B`. |
| `c` | Downstream forward amplitude `C`. |
| `d` | Downstream backward amplitude `D`. |
| `wavenumber` | Air wavenumber `k`. |
| `thickness` | Specimen thickness `d`, in metres. |
| `characteristic_impedance` | Characteristic impedance `rho c`, in rayls. |

**Returns:** Tuple `(p0, pd, u0, ud)` of face pressures and velocities.

## hydraulic_diameter

```python
hydraulic_diameter(width: float, height: float) -> float
```

Hydraulic diameter of a rectangular tube, $4A/P$ (ISO 10534-2, A.2.1.5).

For a rectangular cross-section of side lengths `w` and `h` the ratio
of four times the area to the perimeter reduces to
$d_h = 2wh/(w + h)$; a square tube gives `d_h` equal to the side
length. This is the `d` the Eq. (A.18) attenuation estimate expects for
rectangular tubes (see [`tube_attenuation_constant`](/phonometry/reference/api/materials/impedance-tube/#tube_attenuation_constant)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `width` | Inner side length `w`, in metres. |
| `height` | Inner side length `h`, in metres. |

**Returns:** Hydraulic diameter $d_h = 4A/P$, in metres.

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

$H_c = \sqrt{H_{12}^{I} / H_{12}^{II}}$ from a transfer function measured on an
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
$f_u s < 0.45 c_0$ (Eq. (4)) and, when the tube `diameter` is
given, the cut-on bound $f_u d < 0.58 c_0$ for a circular tube
(Eq. (2)) or $f_u d < 0.50 c_0$ for a rectangular tube (Eq. (3)).
The lower limit uses the Clause 4.2 guideline that the spacing exceed
5 % of the wavelength, i.e. $f_l = c_0 / (20 s)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `spacing` | Microphone spacing `s`, in metres. |
| `speed_of_sound` | Speed of sound `c0`, in metres per second. |
| `diameter` | Tube diameter (circular) or maximum lateral dimension (rectangular/square) `d`, in metres; `None` applies only the spacing bound. |
| `shape` | `"circular"`, `"rectangular"` or `"square"` (a square tube is the rectangular bound with `d` the side length). |

**Returns:** Tuple `(f_l, f_u)` of the lower and upper frequency limits, in Hz.

## plane_wave_frequency_range_astm

```python
plane_wave_frequency_range_astm(
    spacing: float,
    speed_of_sound: float,
    *,
    diameter: float | None = None,
    shape: str = 'circular',
) -> tuple[float, float]
```

Working plane-wave frequency range `(f_l, f_u)` (ASTM E2611-19).

The upper limit is the smaller of the microphone-spacing bound
$s \le 0.8 c / (2 f_u)$, i.e. $f_u s < 0.40 c$ (6.5.4), and,
when the tube `diameter` is given, the cut-on bound
$f_u < K c / d$ with $K = 0.586$ for a circular tube
(6.2.4.1, Eq. (2)) or $K = 0.500$ for a rectangular tube with `d`
the largest section dimension (6.2.5). The lower limit follows 6.2.3: the
spacing shall be greater than 1 % of the wavelength, i.e.
$f_l = c / (100 s)$.

With two different spacings `s1`/`s2`, call with the larger one for
the upper bound and the smaller one for the lower bound (each bound is
binding for every microphone pair).

**Parameters**

| Name | Description |
| :--- | :--- |
| `spacing` | Microphone spacing `s`, in metres. |
| `speed_of_sound` | Speed of sound `c`, in metres per second. |
| `diameter` | Tube diameter (circular) or largest section dimension (rectangular/square) `d`, in metres; `None` applies only the spacing bound. |
| `shape` | `"circular"`, `"rectangular"` or `"square"`. |

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
[`plane_wave_frequency_range_astm`](/phonometry/reference/api/materials/impedance-tube/#plane_wave_frequency_range_astm).

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
r = \frac{H_{12} - H_I}{H_R - H_{12}} \, e^{+2jk_0x_1}
$$

with the incident- and reflected-wave transfer functions
$H_I = e^{-jk_0s}$ (Eq. (D.5)) and $H_R = e^{+jk_0s}$
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

## speed_of_sound_astm

```python
speed_of_sound_astm(temperature: ArrayLike) -> Real
```

Speed of sound in air (ASTM E2611-19, Eq. (4)).

$c = 20.047 \sqrt{273.15 + T}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Room temperature `T`, in **degrees Celsius**. |

**Returns:** Speed of sound `c`, in metres per second.

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

## standing_wave_absorption

```python
standing_wave_absorption(swr: ArrayLike) -> Real
```

Absorption coefficient from the standing-wave ratio (ISO 10534-1).

Combining $\alpha = 1 - |r|^2$ (Eq. (9)) with
$|r| = (s - 1)/(s + 1)$ (Eq. (14)) gives
$\alpha = 4s/(s + 1)^2$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `swr` | Standing-wave ratio `s` (>= 1). |

**Returns:** Absorption coefficient `alpha` in `[0, 1]`.

## standing_wave_normalized_impedance

```python
standing_wave_normalized_impedance(
    swr: ArrayLike,
    first_min_distance: ArrayLike,
    wavelength: ArrayLike,
) -> Complex
```

Normalised impedance from the standing wave (ISO 10534-1, Eqs. (24)-(26)).

$z = Z/Z_0 = (1 + r)/(1 - r)$; the real/imaginary split is
Eqs. (25)/(26).

**Parameters**

| Name | Description |
| :--- | :--- |
| `swr` | Standing-wave ratio `s` (>= 1). |
| `first_min_distance` | Distance `x_min1` to the first minimum, in metres. |
| `wavelength` | Wavelength `lambda0`, in metres. |

**Returns:** Normalised surface impedance `z` (complex).

## standing_wave_ratio_from_level

```python
standing_wave_ratio_from_level(level_difference: ArrayLike) -> Real
```

Standing-wave ratio from a level difference (ISO 10534-1, Eq. (15)).

$s = 10^{\Delta L / 20}$ with
$\Delta L = L_{\max} - L_{\min}$ in decibels.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level_difference` | Level difference $\Delta L = L_{\max} - L_{\min}$, in dB. |

**Returns:** Standing-wave ratio `s` (>= 1).

## standing_wave_reflection

```python
standing_wave_reflection(
    swr: ArrayLike,
    first_min_distance: ArrayLike,
    wavelength: ArrayLike,
) -> Complex
```

Complex reflection factor from the standing wave (ISO 10534-1, Eqs. (17)-(23)).

$r = |r| e^{j\phi}$ with $|r| = (s - 1)/(s + 1)$ (Eq. (14))
and the phase at the first pressure minimum
$\phi = \pi (4 x_{\text{min},1} / \lambda_0 - 1)$ (Eq. (20)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `swr` | Standing-wave ratio `s` (>= 1). |
| `first_min_distance` | Distance `x_min1` from the reference plane to the first pressure minimum (toward the source), in metres. |
| `wavelength` | Wavelength `lambda0`, in metres (Eq. (27)). |

**Returns:** Complex reflection factor `r`.

## standing_wave_reflection_magnitude

```python
standing_wave_reflection_magnitude(swr: ArrayLike) -> Real
```

Reflection magnitude from the standing-wave ratio (ISO 10534-1, Eq. (14)).

$|r| = (s - 1) / (s + 1)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `swr` | Standing-wave ratio `s` (>= 1). |

**Returns:** Reflection magnitude `|r|` in `[0, 1]`.

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

## transfer_matrix_one_load

```python
transfer_matrix_one_load(
    load: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    wavenumber: ArrayLike,
    characteristic_impedance: float,
    frequency: ArrayLike | None = None,
    diameter: float | None = None,
    shape: str = 'circular',
) -> TransferMatrix
```

One-load transfer matrix, symmetric specimen (ASTM E2611-19, Eqs. (23)-(24)).

Valid only for a reciprocal **and** symmetric specimen
($T_{11} = T_{22}$ and $T_{11} T_{22} - T_{12} T_{21} = 1$,
Eq. (23)). A single termination suffices:

$$
\begin{aligned} \mathrm{DEN} &= p_0 u_d + p_d u_0 \\ T_{11} = T_{22} &= (p_d u_d + p_0 u_0) / \mathrm{DEN} \\ T_{12} &= (p_0^{2} - p_d^{2}) / \mathrm{DEN} \\ T_{21} &= (u_0^{2} - u_d^{2}) / \mathrm{DEN} \end{aligned}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `load` | Microphone transfer functions `(H1, H2, H3, H4)`. |
| `l1` | Upstream reference distance `l1`, in metres. |
| `s1` | Upstream microphone spacing `s1`, in metres. |
| `l2` | Downstream reference distance `l2`, in metres. |
| `s2` | Downstream microphone spacing `s2`, in metres. |
| `thickness` | Specimen thickness `d`, in metres. |
| `wavenumber` | Air wavenumber `k`. |
| `characteristic_impedance` | Characteristic impedance `rho c`. |
| `frequency` | Optional frequency vector `f`, in hertz, retained on the result so [`TransferMatrix.plot`](/phonometry/reference/api/materials/impedance-tube/#transfermatrixplot) needs no arguments. |
| `diameter` | Optional tube diameter (circular) or largest section dimension (rectangular/square), in metres, that activates the plane-wave working-range check (6.2.3-6.2.5, 6.5.4). |
| `shape` | Tube cross-section, `"circular"`, `"rectangular"` or `"square"`. |

**Returns:** The specimen [`TransferMatrix`](/phonometry/reference/api/materials/impedance-tube/#transfermatrix) (measurement context retained on the result).

## transfer_matrix_two_load

```python
transfer_matrix_two_load(
    load_a: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    load_b: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    wavenumber: ArrayLike,
    characteristic_impedance: float,
    frequency: ArrayLike | None = None,
    diameter: float | None = None,
    shape: str = 'circular',
) -> TransferMatrix
```

Two-load transfer matrix (ASTM E2611-19, Eqs. (17)-(22)).

Each load is the tuple `(H1, H2, H3, H4)` of the four microphone transfer
functions measured with a different downstream termination. The two loads
give four equations for the four unknowns (Eq. (22)):

$$
\begin{aligned} \mathrm{DEN} &= p_{da} u_{db} - p_{db} u_{da} \\ T_{11} &= (p_{0a} u_{db} - p_{0b} u_{da}) / \mathrm{DEN} \\ T_{12} &= (p_{0b} p_{da} - p_{0a} p_{db}) / \mathrm{DEN} \\ T_{21} &= (u_{0a} u_{db} - u_{0b} u_{da}) / \mathrm{DEN} \\ T_{22} &= (p_{da} u_{0b} - p_{db} u_{0a}) / \mathrm{DEN} \end{aligned}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `load_a` | Microphone transfer functions `(H1, H2, H3, H4)` for load a. |
| `load_b` | Microphone transfer functions `(H1, H2, H3, H4)` for load b. |
| `l1` | Upstream reference distance `l1`, in metres. |
| `s1` | Upstream microphone spacing `s1`, in metres. |
| `l2` | Downstream reference distance `l2`, in metres. |
| `s2` | Downstream microphone spacing `s2`, in metres. |
| `thickness` | Specimen thickness `d`, in metres. |
| `wavenumber` | Air wavenumber `k`. |
| `characteristic_impedance` | Characteristic impedance `rho c`. |
| `frequency` | Optional frequency vector `f`, in hertz, retained on the result so [`TransferMatrix.plot`](/phonometry/reference/api/materials/impedance-tube/#transfermatrixplot) needs no arguments. |
| `diameter` | Optional tube diameter (circular) or largest section dimension (rectangular/square), in metres, that activates the plane-wave working-range check (6.2.3-6.2.5, 6.5.4). |
| `shape` | Tube cross-section, `"circular"`, `"rectangular"` or `"square"`. |

**Returns:** The specimen [`TransferMatrix`](/phonometry/reference/api/materials/impedance-tube/#transfermatrix) (measurement context retained on the result).

## TransferMatrix

```python
TransferMatrix(
    t11: Complex,
    t12: Complex,
    t21: Complex,
    t22: Complex,
    l1: float | None = None,
    s1: float | None = None,
    l2: float | None = None,
    s2: float | None = None,
    thickness: float | None = None,
    diameter: float | None = None,
    shape: str | None = None,
    frequency: Real | None = None,
    air_characteristic_impedance: float | None = None,
)
```

Acoustic transfer matrix `[[T11, T12], [T21, T22]]` (ASTM E2611-19).

Relates the pressure and normal particle velocity across a specimen,
$[p; u]_{x=0} = T \, [p; u]_{x=d}$ (Eq. (16)). Each entry is complex and
may be scalar or a per-frequency array of matching shape.

The trailing fields retain the measurement context when the matrix comes
out of [`transfer_matrix_two_load`](/phonometry/reference/api/materials/impedance-tube/#transfer_matrix_two_load) / [`transfer_matrix_one_load`](/phonometry/reference/api/materials/impedance-tube/#transfer_matrix_one_load)
(tube geometry `l1`/`s1`/`l2`/`s2`, specimen `thickness`, tube
`diameter` and canonical cross-section `shape`, the `frequency`
vector when supplied to the solver, and the air
`air_characteristic_impedance` `rho c`); all default to `None` so a
hand-built matrix (for example [`air_layer_transfer_matrix`](/phonometry/reference/api/materials/impedance-tube/#air_layer_transfer_matrix)) is
unchanged.

### TransferMatrix.absorption_hard_backed()

```python
TransferMatrix.absorption_hard_backed(
    characteristic_impedance: float,
) -> Real
```

Hard-backed absorption coefficient (ASTM E2611-19, Eq. (28)).

$\alpha = 1 - \lvert R \rvert^2$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `characteristic_impedance` | Characteristic impedance `rho c`. |

**Returns:** Absorption coefficient `alpha`.

### TransferMatrix.characteristic_impedance_material()

```python
TransferMatrix.characteristic_impedance_material() -> Complex
```

Characteristic impedance of the material (ASTM E2611-19, Eq. (30)).

$Z = \sqrt{T_{12} / T_{21}}$.

**Returns:** Complex characteristic impedance `Z`, in rayls.

### TransferMatrix.determinant()

```python
TransferMatrix.determinant() -> Complex
```

Determinant $T_{11} T_{22} - T_{12} T_{21}$ (unity for a reciprocal specimen).

### TransferMatrix.material_wavenumber()

```python
TransferMatrix.material_wavenumber(thickness: float) -> Complex
```

Propagation wavenumber inside the material (ASTM E2611-19, Eq. (29)).

$k' = \arccos(T_{11}) / d$ (complex `arccos`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `thickness` | Specimen thickness `d`, in metres. |

**Returns:** Complex material wavenumber `k'`, in reciprocal metres.

### TransferMatrix.plot()

```python
TransferMatrix.plot(
    frequency: ArrayLike | None = None,
    characteristic_impedance: float | None = None,
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the transmission loss with the hard-backed absorption overlaid.

Reads the four-pole entries out as the two ASTM E2611-19 spectra a
laboratory quotes: the normal-incidence transmission loss `TLn(f)`
(Eq. (26), the primary curve, left axis) and the hard-backed
absorption coefficient `alpha(f)` (Eq. (28), a muted companion on a
0..1 right axis). The four-pole entries carry no frequency axis of
their own, so the plot needs the measurement's `frequency` vector
(matching the shape of the entries) and the air characteristic
impedance `rho c`. A matrix built by the solvers retains both
(`self.frequency` / `self.air_characteristic_impedance`), so
`plot()` takes no arguments there; only a hand-built matrix (for
example [`air_layer_transfer_matrix`](/phonometry/reference/api/materials/impedance-tube/#air_layer_transfer_matrix)) must supply them.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` of the transmission-loss curve.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz, matching the shape of the matrix entries; `None` uses the stored `frequency`. |
| `characteristic_impedance` | Characteristic impedance `rho c` of the air in the tube, in rayls; `None` uses the stored `air_characteristic_impedance`. |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Plot language: `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the transmission-loss `plot` call. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `frequency` or `characteristic_impedance` is neither supplied nor stored on the matrix. |

### TransferMatrix.plot_geometry()

```python
TransferMatrix.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the four-microphone tube to scale (dimensioned side view).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the matrix does not retain its tube geometry (`l1`/`s1`/`l2`/`s2`/`thickness`). |

### TransferMatrix.reflection_hard_backed()

```python
TransferMatrix.reflection_hard_backed(
    characteristic_impedance: float,
) -> Complex
```

Hard-backed reflection coefficient (ASTM E2611-19, Eq. (27)).

$R = (T_{11} - \rho c T_{21}) / (T_{11} + \rho c T_{21})$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `characteristic_impedance` | Characteristic impedance `rho c`. |

**Returns:** Complex reflection coefficient `R`.

### TransferMatrix.transmission_loss()

```python
TransferMatrix.transmission_loss(characteristic_impedance: float) -> Real
```

Normal-incidence transmission loss in dB (ASTM E2611-19, Eq. (26)).

With

$$
t = \frac{2 e^{jkd}} {T_{11} + T_{12}/(\rho c) + \rho c \, T_{21} + T_{22}} \tag{Eq. 25}
$$

$$
TL = 20 \log_{10} \left| \frac{1}{t} \right| = 20 \log_{10} \frac{\lvert T_{11} + T_{12}/(\rho c) + \rho c \, T_{21} + T_{22} \rvert}{2} \tag{Eq. 26}
$$

(the $e^{jkd}$ factor has unit magnitude for
a real wavenumber).

**Parameters**

| Name | Description |
| :--- | :--- |
| `characteristic_impedance` | Characteristic impedance `rho c`. |

**Returns:** Transmission loss `TLn`, in decibels.

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

## wave_decomposition

```python
wave_decomposition(
    h1: ArrayLike,
    h2: ArrayLike,
    h3: ArrayLike,
    h4: ArrayLike,
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    wavenumber: ArrayLike,
    diameter: float | None = None,
    shape: str = 'circular',
) -> tuple[Complex, Complex, Complex, Complex]
```

Decompose the wave field into `(A, B, C, D)` (ASTM E2611-19, Eqs. (17)-(20)).

The exponents are implemented exactly as printed:

$$
A = \frac{j \left( H_1 e^{-jkl_1} - H_2 e^{-jk(l_1+s_1)} \right)} {2 \sin(k s_1)}
$$

$$
B = \frac{j \left( H_2 e^{+jk(l_1+s_1)} - H_1 e^{+jkl_1} \right)} {2 \sin(k s_1)}
$$

$$
C = \frac{j \left( H_3 e^{+jk(l_2+s_2)} - H_4 e^{+jkl_2} \right)} {2 \sin(k s_2)}
$$

$$
D = \frac{j \left( H_4 e^{-jkl_2} - H_3 e^{-jk(l_2+s_2)} \right)} {2 \sin(k s_2)}
$$

`A`/`B` are the forward/backward complex amplitudes on the upstream
(source) side and `C`/`D` those on the downstream side, all referenced
to the front face $x = 0$. With the $e^{+j\omega t}$ /
forward-$e^{-jkx}$
convention these exponents correspond to the microphone whose transfer
function is `H2` sitting nearest the front face at distance `l1` (and
`H1` at $l_1 + s_1$), and to `H3` nearest the downstream side at
`l2` (and `H4` at $l_2 + s_2$), with `l1`, `l2` measured
from the front reference plane. The convention was locked down against the analytic
air-layer transfer matrix (see [`air_layer_transfer_matrix`](/phonometry/reference/api/materials/impedance-tube/#air_layer_transfer_matrix)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `h1` | Transfer function `H1,ref` (upstream, farther microphone). |
| `h2` | Transfer function `H2,ref` (upstream, nearer microphone). |
| `h3` | Transfer function `H3,ref` (downstream, nearer microphone). |
| `h4` | Transfer function `H4,ref` (downstream, farther microphone). |
| `l1` | Distance `l1` from the front reference plane, in metres. |
| `s1` | Upstream microphone spacing `s1`, in metres. |
| `l2` | Distance `l2` from the front reference plane, in metres. |
| `s2` | Downstream microphone spacing `s2`, in metres. |
| `wavenumber` | Air wavenumber `k` (real or complex), scalar or per band. |
| `diameter` | Optional tube diameter (circular) or largest section dimension (rectangular/square), in metres, that activates the plane-wave working-range check (6.2.3-6.2.5, 6.5.4). |
| `shape` | Tube cross-section, `"circular"`, `"rectangular"` or `"square"`. |

**Returns:** Tuple `(A, B, C, D)` of complex amplitudes.
