---
title: "materials.diffusers.scattering_diffusion"
description: "Directional and random-incidence diffusion coefficients in a free field."
sidebar:
  label: "scattering_diffusion"
---

Directional and random-incidence diffusion coefficients in a free field.

**ISO 17497-2:2012.** From the set of reflected sound-pressure levels `L_i`
on a semicircle or hemisphere the autocorrelation diffusion coefficient
`d_theta` is formed for equal-area receivers (Clause 8.1, Formula (5)) or
with per-receiver area weights `N_i` (Formula (6)); the area weights follow
from the solid-angle factors of Clause 8.3 (Formula (8)). Finite-panel effects
are removed by normalising to the reference flat surface (Clause 8.2,
Formula (7)), and the random-incidence coefficient is the (weighted) average of
the directional coefficients over the source positions (Clause 8.4).

One subject: the free-field polar response of a surface and the single number
Clause 8 distils from it, band by band. Part 1 of ISO 17497 is a different
measurement, made in a reverberation room, and lives in
[`phonometry.materials.diffusers.reverberation_room_scattering`](/phonometry/reference/api/materials/reverberation-room-scattering/); the two
parts share no formula, and the helpers are named per part so they are never
mixed. Neither part contains a numeric worked example.

The Part 2 diffusion coefficient is the design target of subwavelength diffuser
panels such as the metadiffusers of Jiménez, Cox, Romero-García & Groby
(2017, *Scientific Reports* 7, 5389, doi:10.1038/s41598-017-05710-5), whose
slow-sound resonant slots reach the diffusion of a Schroeder or
quadratic-residue diffuser in a fraction of the depth.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## area_factors

```python
area_factors(
    elevations: ArrayLike,
    *,
    delta_theta: float,
    delta_phi: float | None = None,
) -> Real
```

Per-receiver area weights `N_i` (ISO 17497-2, Clause 8.3, Formula (8)).

For a hemispherical measurement the solid-angle area sampled by a receiver
at elevation `theta` (with angular spacings `delta_theta`,
`delta_phi`) is:

$$
\begin{aligned} A_i &= (4 \pi / \Delta\phi) \sin^{2}(\Delta\theta / 4) && \text{for } \theta = 0^\circ \\ A_i &= 2 \sin(\theta) \sin(\Delta\theta / 2) && \text{for } \theta \ne 0^\circ, 90^\circ \\ A_i &= \sin(\Delta\theta / 2) && \text{for } \lvert \theta \rvert = 90^\circ \end{aligned}
$$

and $N_i = A_i / A_{\min}$ (Formula (8)), with `A_min` the
smallest `A_i`. All angles are handled internally in **radians**; the
$\theta = 0$ form in particular requires `delta_phi` in radians to
be dimensionally consistent with the $4 \pi$ factor.

**Parameters**

| Name | Description |
| :--- | :--- |
| `elevations` | Receiver elevation angles `theta` from the reference normal, in **degrees** (1-D), over the measurement domain $0 \le \theta \le 90$ (Figure 7). Formula (8) assumes a single receiver at $\theta = 0$ (the zenith); duplicate zenith entries would each take the full zenith area. |
| `delta_theta` | Elevation spacing between adjacent receivers, in degrees (typically 5). |
| `delta_phi` | Azimuth spacing between adjacent receivers, in degrees; defaults to `delta_theta`. Required (implicitly) for the $\theta = 0$ receiver. |

**Returns:** Per-receiver area weights `N_i` (dimensionless, min value 1).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-1-D input or non-positive spacings. |

## diffusion_spectrum

```python
diffusion_spectrum(
    frequencies: ArrayLike,
    diffusion: ArrayLike,
    *,
    normalized: ArrayLike | None = None,
) -> DiffusionSpectrum
```

Diffusion-coefficient spectrum `d(f)` (ISO 17497-2, Clause 8.5).

Pairs the per-band diffusion coefficients `d` with their band centres and
returns a plottable, reportable [`DiffusionSpectrum`](/phonometry/reference/api/materials/scattering-diffusion/#diffusionspectrum). The coefficient
is the *directional* coefficient `d_theta` (Formula (5)/(6)) when it comes
from a single source position, or the *random-incidence* coefficient `d`
when it is the per-band average of the directional coefficients over the
source positions (Clause 8.4, e.g. via [`random_incidence_diffusion`](/phonometry/reference/api/materials/scattering-diffusion/#random_incidence_diffusion)
band by band). The optional normalised coefficients `d_n` (Formula (7))
are carried through when supplied.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centres, in hertz (1-D). |
| `diffusion` | Diffusion coefficient `d` per band. |
| `normalized` | Optional normalised diffusion coefficient `d_n` per band; `None` when the reference flat surface was not measured. |

**Returns:** A [`DiffusionSpectrum`](/phonometry/reference/api/materials/scattering-diffusion/#diffusionspectrum) with `.plot()` and `.report()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the inputs differ in length, are empty or not 1-D. |

## DiffusionResult

```python
DiffusionResult(angles: Real, levels: Real, coefficient: float)
```

A measured polar response and its diffusion coefficient (ISO 17497-2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `angles` | Receiver angles of the polar response, in degrees. |
| `levels` | Reflected sound-pressure level at each angle, in decibels. |
| `coefficient` | Autocorrelation diffusion coefficient `d` (Formula (5)). |

### DiffusionResult.plot()

```python
DiffusionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the polar response with the diffusion coefficient annotated.

Requires matplotlib (`pip install phonometry[plot]`); returns the
polar `Axes` and never calls `plt.show`.

### DiffusionResult.report()

```python
DiffusionResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 17497-2 polar-response test-report fiche to a PDF.

Writes a one-page accredited free-field diffusion report for a single
source position (ISO 17497-2:2012, Clause 8.5): the standard-basis line,
an optional metadata header block, a two-panel body with the corrected
polar-response table (receiver angle and reflected sound-pressure level
`L`, rounded to 0,1 dB) beside the semicircular polar plot, a boxed
directional diffusion coefficient `d_theta` (Formula (5)/(6)) and a
footer with the fixed disclaimer. ISO 17497-2 is a characterisation, so
there is no pass/fail verdict.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a body-and-disclaimer fiche. The applicable descriptive fields are `client`, `manufacturer`, `specimen`, `mounting`, `test_room`, `test_date`, `temperature`, `relative_humidity`, `pressure`, `measurement_standard`, `laboratory`, `operator`, `report_id` and `notes`. The `requirement` field is ignored (ISO 17497-2 has no verdict). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for signature parity; the polar-response fiche has no extended table, so it renders the same body. |
| `language` | Fiche language: `"en"` (default, English, decimal point) or `"es"` (Spanish, decimal comma). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## DiffusionSpectrum

```python
DiffusionSpectrum(
    frequencies: Real,
    diffusion: Real,
    normalized: Real | None = None,
)
```

A diffusion-coefficient spectrum `d(f)` (ISO 17497-2, Clause 8.5).

Where [`DiffusionResult`](/phonometry/reference/api/materials/scattering-diffusion/#diffusionresult) holds the polar response of a single
one-third-octave band, this holds the diffusion coefficient across the
measured bands, so it can be tabulated and plotted against frequency as
Clause 8.5 requires. The per-band coefficient is a *directional* diffusion
coefficient `d_theta` (Formula (5)/(6)) when it comes from one source
position, or a *random-incidence* diffusion coefficient `d` when it is the
per-band average of the directional coefficients over the source positions
(Clause 8.4); the standard defines both as frequency-dependent quantities,
so this carries a spectrum rather than a single number.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centre frequencies, in hertz. |
| `diffusion` | Diffusion coefficient `d` per band (directional per source, or random-incidence when averaged over source positions). |
| `normalized` | Optional normalised diffusion coefficient `d_n` per band (Formula (7)), or `None` when the reference flat surface was not measured. |

### DiffusionSpectrum.plot()

```python
DiffusionSpectrum.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the diffusion coefficient `d` versus frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

### DiffusionSpectrum.report()

```python
DiffusionSpectrum.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 17497-2 diffusion-coefficient test-report fiche to a PDF.

Writes a one-page accredited free-field diffusion report
(ISO 17497-2:2012, Clause 8.5): the standard-basis line, an optional
metadata header block, a two-panel body with the per-band table
(frequency, the diffusion coefficient `d` and, when present, the
normalised `d_n`) beside the `d(f)` curve on a categorical band
axis, a boxed characterisation headline over the tested frequency range,
and a footer with the fixed disclaimer. ISO 17497-2 is a
characterisation, so there is no pass/fail verdict.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a body-and-disclaimer fiche whose header shows only the measured frequency range. The applicable descriptive fields are `client`, `manufacturer`, `specimen`, `mounting`, `test_room`, `test_date`, `temperature`, `relative_humidity`, `pressure`, `measurement_standard`, `laboratory`, `operator`, `report_id` and `notes`. The `requirement` field is ignored (ISO 17497-2 has no verdict). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` and a normalised spectrum is present, the value table adds the normalised `d_n` column. |
| `language` | Fiche language: `"en"` (default, English, decimal point) or `"es"` (Spanish, decimal comma). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## directional_diffusion

```python
directional_diffusion(
    angles: ArrayLike,
    levels: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> DiffusionResult
```

Diffusion coefficient of a polar response (ISO 17497-2, Formula (5)/(6)).

Convenience wrapper over [`directional_diffusion_coefficient`](/phonometry/reference/api/materials/scattering-diffusion/#directional_diffusion_coefficient) that
keeps the receiver angles alongside the levels and returns a plottable
[`DiffusionResult`](/phonometry/reference/api/materials/scattering-diffusion/#diffusionresult).

**Parameters**

| Name | Description |
| :--- | :--- |
| `angles` | Receiver angles of the polar response, in degrees (1-D). |
| `levels` | Reflected sound-pressure level at each angle, in decibels. |
| `weights` | Optional area weights `N_i` (Formula (8)); `None` uses the equal-area Formula (5). |

**Returns:** A [`DiffusionResult`](/phonometry/reference/api/materials/scattering-diffusion/#diffusionresult) with `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `angles` and `levels` differ in length or are shorter than two receivers. |

## directional_diffusion_coefficient

```python
directional_diffusion_coefficient(
    levels: ArrayLike,
    *,
    area_weights: ArrayLike | None = None,
) -> float
```

Directional diffusion coefficient `d_theta` (ISO 17497-2, Formulas (5)/(6)).

For a fixed source position and one-third-octave band, from the `n`
reflected sound-pressure levels `L_i` (dB). With equal-area receivers
(`area_weights is None`, Formula (5)):

$$
d_\theta = \frac{\left( \sum_i p_i \right)^{2} - \sum_i p_i^{2}} {(n - 1) \sum_i p_i^{2}}
$$

where $p_i = 10^{L_i / 10}$. When each receiver samples a different
area (Formula (6)) the per-receiver weights `N_i` (from
[`area_factors`](/phonometry/reference/api/materials/scattering-diffusion/#area_factors)) enter:

$$
d_\theta = \frac{\left( \sum_i p_i N_i \right)^{2} - \sum_i N_i p_i^{2}} {\left( \sum_i N_i - 1 \right) \sum_i N_i p_i^{2}}
$$

which reduces to Formula (5) for uniform weights. The coefficient is 0 when
only one receiver has non-zero scattered energy and 1 when all receivers
are equal.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | The $n \ge 2$ reflected sound-pressure levels `L_i`, in decibels (a level of `-inf` denotes a receiver with zero energy). |
| `area_weights` | Optional per-receiver area weights `N_i` (Formula (8)); `None` selects the equal-area Formula (5). |

**Returns:** Directional diffusion coefficient `d_theta` (a scalar).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for fewer than two receivers, a non-1-D input, a length mismatch, or non-positive total weight. |

## normalized_diffusion_coefficient

```python
normalized_diffusion_coefficient(
    d_theta: ArrayLike,
    d_theta_reference: ArrayLike,
) -> Real
```

Normalised directional diffusion coefficient (ISO 17497-2, Formula (7)).

$d_{\theta,n} = (d_\theta - d_{\theta,r}) / (1 - d_{\theta,r})$,
removing the
finite-panel diffusion of the reference flat surface `d_theta_r` (same
projected footprint as the test surface). It maps
$d_\theta = d_{\theta,r}$
to 0 and $d_\theta = 1$ to 1.

**Parameters**

| Name | Description |
| :--- | :--- |
| `d_theta` | Directional diffusion coefficient of the test surface. |
| `d_theta_reference` | Directional diffusion coefficient of the reference flat surface `d_theta_r`. |

**Returns:** Normalised directional diffusion coefficient `d_theta_n`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if any reference coefficient equals 1 (undefined ratio). |

## plot_goniometer_geometry

```python
plot_goniometer_geometry(
    ax: Axes | None = None,
    *,
    source_distance: float = 10.0,
    receiver_radius: float = 5.0,
    angular_step: float = 5.0,
    sample_width: float = 0.6,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the free-field diffusion goniometer in plan, to scale.

Receiver semicircle at its radius with one microphone per angular step,
the source on the normal at its distance and the sample at the centre;
defaults are the standard 10 m source, 5 m receiver arc and 5-degree
resolution (37 microphones).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `source_distance` | Source distance from the sample, in metres. |
| `receiver_radius` | Receiver-arc radius, in metres. |
| `angular_step` | Angular spacing of the receivers, in degrees. |
| `sample_width` | Drawn sample width, in metres. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the receiver scatter. |

**Returns:** The axes.

## random_incidence_diffusion

```python
random_incidence_diffusion(
    directional_coefficients: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> float
```

Random-incidence diffusion coefficient `d` (ISO 17497-2, Clause 8.4).

The (normalised or non-normalised) directional coefficients are averaged
over the source positions. Hemispherical measurements use **equal**
weightings (`weights is None`); two-dimensional (single-plane)
measurements use the source weighting of Clause 8.4 - weight 1 for the
0 deg source and weight 3 for each of the four +/-30 deg, +/-60 deg sources
(see [`TWO_DIMENSIONAL_SOURCE_WEIGHTS`](/phonometry/reference/api/materials/scattering-diffusion/#two_dimensional_source_weights)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `directional_coefficients` | Directional diffusion coefficients `d_theta` (or `d_theta_n`), one per source position (1-D). |
| `weights` | Optional source-position weights; `None` averages with equal weight. |

**Returns:** Random-incidence diffusion coefficient `d` (a scalar).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an empty or non-1-D input, a length mismatch, or non-positive total weight. |

## TWO_DIMENSIONAL_SOURCE_WEIGHTS

*Constant* (`tuple`).

```python
TWO_DIMENSIONAL_SOURCE_WEIGHTS = (1, 3, 3, 3, 3)
```
