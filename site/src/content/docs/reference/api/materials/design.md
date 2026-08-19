---
title: "materials.diffusers.design"
description: "Far-field polar response and diffusion coefficient predicted from a diffuser design."
sidebar:
  label: "design"
---

Far-field polar response and diffusion coefficient predicted from a diffuser design.

Where [`phonometry.materials.diffusers.scattering_diffusion`](/phonometry/reference/api/materials/scattering-diffusion/) evaluates the *measured*
directional diffusion coefficient of ISO 17497-2 from a set of reflected
sound-pressure levels, this module *predicts* those levels from the physical
design of a Schroeder phase-grating diffuser (well-depth sequence and well
width), so the diffusion coefficient can be estimated before a sample is built.

The prediction is the standard single-plane Fraunhofer (Fourier) far-field model
of a phase-grating surface, as set out in Cox and D'Antonio, *Acoustic Absorbers
and Diffusers* (3rd ed., CRC Press, 2017):

* Each well of depth `d_n` behaves, at normal incidence and with a rigid
  bottom, as a locally reacting patch of pressure reflection coefficient
  $R_n = e^{-2jkd_n}$ (Chapter 10; the phase change of a wave travelling
  down and back up the well). An arbitrary complex reflection coefficient per
  well is accepted as well, so admittance or resonator-loaded surfaces computed
  elsewhere can be fed in directly.
* The scattered pressure at reflection angle $\theta$ for a source at
  incidence $\psi$ is the sum over the wells of the periodic surface
  (Chapter 5, Equation (5.8), and Chapter 9, Equation (9.32)):
  $p(\theta) = F(\theta) \sum_n R_n e^{jkx_n(\sin\psi + \sin\theta)}$,
  with `x_n` the centre of the `n`-th well and $k = 2\pi f/c$. The
  optional prefactor $F(\theta)$ collects the single-well aperture
  directivity $\operatorname{sinc}(kw(\sin\psi + \sin\theta)/2)$ (the
  Fourier transform of one well of width `w`) and the Kirchhoff obliquity
  factor $(1 + \cos\theta)/2$ of Equation (9.32); both are absorbed
  into the constant `A` of Equation (5.8) and can be switched off to
  recover the plain Fourier form.

The polar levels $L_i = 20 \log_{10}|p(\theta_i)|$ are then passed to the
ISO 17497-2 autocorrelation diffusion coefficient of
[`directional_diffusion_coefficient`](/phonometry/reference/api/materials/scattering-diffusion/#directional_diffusion_coefficient),
and normalised against the same-footprint flat reference with
[`normalized_diffusion_coefficient`](/phonometry/reference/api/materials/scattering-diffusion/#normalized_diffusion_coefficient)
(Formula (7)), exactly as a measurement would be reduced.

For a quadratic residue diffuser the depth sequence follows from the prime
generator `N` and the design frequency `f_0` (Cox and D'Antonio,
Equations (10.2) and (10.3)):

$$
s_n = n^2 \bmod N, \qquad d_n = \frac{s_n \lambda_0}{2N}, \qquad \lambda_0 = c / f_0.
$$

The model is a far-field approximation and, like every Fourier diffuser model,
loses accuracy at low frequencies, at grazing angles and for surfaces with
strong absorption; it is meant for design estimation, not for replacing an
ISO 17497-2 measurement. A resonator-loaded "metadiffuser" slit surface is not
modelled here: only its per-well reflection coefficient would change, so such a
surface can be predicted by supplying the complex `reflection` sequence
directly.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## DEFAULT_POLAR_ANGLES

*Constant* (`tuple`).

```python
DEFAULT_POLAR_ANGLES = (-90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90)
```

## DiffuserPolarResponse

```python
DiffuserPolarResponse(
    frequency: float,
    angles: Real,
    levels: Real,
    coefficient: float,
    source_angle: float = 0.0,
    well_width: float | None = None,
    depths: Real | None = None,
    periods: int | None = None,
)
```

A predicted far-field polar response of a diffuser at one frequency.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency of the prediction, in hertz. |
| `angles` | Receiver reflection angles, in degrees. |
| `levels` | Predicted reflected sound-pressure level at each angle, in decibels, referenced to the peak of the response (peak at 0 dB). |
| `coefficient` | Directional diffusion coefficient `d_theta` of the predicted response (ISO 17497-2, Formula (5)). |
| `source_angle` | Angle of incidence `psi` of the source, in degrees. |
| `well_width` | Well width `w` of the predicted surface, in metres, always retained by the predictor (with `periods`) so `plot_geometry` can draw the well profile; appended after the original fields and `None` only for hand-built responses. |
| `depths` | Well depths `d_n` of one period, in metres, when the response was predicted from depths; `None` otherwise (explicit `reflection` surfaces have no drawable well profile). |
| `periods` | Number of repeated periods of the prediction. |

### DiffuserPolarResponse.plot()

```python
DiffuserPolarResponse.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the predicted polar response with the diffusion coefficient annotated.

Requires matplotlib (`pip install phonometry[plot]`); returns the polar
`Axes` and never calls `plt.show`.

### DiffuserPolarResponse.plot_geometry()

```python
DiffuserPolarResponse.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the well profile of the predicted surface to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the response does not retain its well geometry (hand-built, or predicted from an explicit `reflection` sequence). |

## plot_qrd_geometry

```python
plot_qrd_geometry(
    depths: ArrayLike,
    well_width: float,
    ax: Axes | None = None,
    *,
    periods: int = 1,
    fin_width: float | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the well profile of a quadratic-residue diffuser, to scale.

Wells open upward; the profile repeats `periods` times with thin fins
between wells. Pairs with
[`qrd_well_depths`](/phonometry/reference/api/materials/design/#qrd_well_depths), which supplies the depth
sequence.

**Parameters**

| Name | Description |
| :--- | :--- |
| `depths` | Well depths `d_n`, in metres (one period). |
| `well_width` | Well width `w`, in metres. |
| `ax` | Existing axes, or `None` to create a figure. |
| `periods` | Number of repeated periods (>= 1). |
| `fin_width` | Fin thickness between wells, in metres; `None` draws `w / 12`. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the base-slab rectangle. |

**Returns:** The axes.

## predict_diffuser_polar_response

```python
predict_diffuser_polar_response(
    well_width: float,
    frequency: float,
    *,
    depths: ArrayLike,
    angles: ArrayLike = ...,
    source_angle: float = ...,
    periods: int = ...,
    speed_of_sound: float = ...,
    include_aperture: bool = ...,
    include_obliquity: bool = ...,
) -> DiffuserPolarResponse

predict_diffuser_polar_response(
    well_width: float,
    frequency: float,
    *,
    reflection: ArrayLike,
    angles: ArrayLike = ...,
    source_angle: float = ...,
    periods: int = ...,
    speed_of_sound: float = ...,
    include_aperture: bool = ...,
    include_obliquity: bool = ...,
) -> DiffuserPolarResponse
```

Predict the far-field polar response of a diffuser (Cox and D'Antonio, Eq. (5.8)).

Evaluates the single-plane Fraunhofer scattered pressure of a periodic
phase-grating surface and reduces it to the ISO 17497-2 directional
diffusion coefficient. Supply the surface either as rigid-bottom well
`depths` ($R_n = e^{-2jkd_n}$) or as an explicit per-well complex
`reflection` sequence (for admittance or resonator-loaded surfaces); give
exactly one.

**Parameters**

| Name | Description |
| :--- | :--- |
| `well_width` | Well width `w`, in metres (uniform across wells). |
| `frequency` | Frequency of the prediction `f`, in hertz. |
| `depths` | Well depths `d_n` of one period, in metres (1-D, at least two wells); mutually exclusive with `reflection`. |
| `reflection` | Explicit per-well complex pressure reflection coefficient of one period (1-D, at least two wells); mutually exclusive with `depths`. |
| `angles` | Receiver reflection angles `theta`, in degrees; defaults to the ISO 17497-2 semicircle [`DEFAULT_POLAR_ANGLES`](/phonometry/reference/api/materials/design/#default_polar_angles). |
| `source_angle` | Angle of incidence `psi` of the source, in degrees (0 = normal incidence). |
| `periods` | Number of repetitions `N_p` of the single period; the grating lobes that define a Schroeder diffuser require `periods >= 2`. |
| `speed_of_sound` | Speed of sound `c`, in metres per second. |
| `include_aperture` | Include the single-well aperture directivity $\operatorname{sinc}(kw(\sin\psi + \sin\theta)/2)$ of Eq. (9.32); defaults to `True`. |
| `include_obliquity` | Include the Kirchhoff obliquity factor $(\cos\theta + \cos\psi)/2$, the oblique-source generalisation of the normal-incidence $(1 + \cos\theta)/2$ of Eq. (9.32); defaults to `True`. |

**Returns:** A [`DiffuserPolarResponse`](/phonometry/reference/api/materials/design/#diffuserpolarresponse) with the per-angle levels, the directional diffusion coefficient and `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for invalid geometry, or if not exactly one of `depths` and `reflection` is supplied. |

## predicted_diffusion_spectrum

```python
predicted_diffusion_spectrum(
    well_width: float,
    frequencies: ArrayLike,
    *,
    depths: ArrayLike,
    reflection_of: Any | None = None,
    angles: ArrayLike = (-90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90),
    source_angle: float = 0.0,
    periods: int = 1,
    speed_of_sound: float = 343.0,
    include_aperture: bool = True,
    include_obliquity: bool = True,
    normalize: bool = True,
) -> DiffusionSpectrum
```

Predicted diffusion-coefficient spectrum `d(f)` of a diffuser design.

Predicts the far-field polar response at each frequency with
[`predict_diffuser_polar_response`](/phonometry/reference/api/materials/design/#predict_diffuser_polar_response), forms the ISO 17497-2 directional
diffusion coefficient band by band, and (by default) normalises it against a
same-footprint flat reference surface (all wells of zero depth) with
[`normalized_diffusion_coefficient`](/phonometry/reference/api/materials/scattering-diffusion/#normalized_diffusion_coefficient)
(Formula (7)). The result reuses
[`DiffusionSpectrum`](/phonometry/reference/api/materials/scattering-diffusion/#diffusionspectrum), so it
plots and reports like a measured spectrum.

Only the rigid-bottom well `depths` path supports the automatic flat
reference; an explicit frequency-dependent reflection model is out of scope
for this helper (build the spectrum yourself from
[`predict_diffuser_polar_response`](/phonometry/reference/api/materials/design/#predict_diffuser_polar_response) in that case).

**Parameters**

| Name | Description |
| :--- | :--- |
| `well_width` | Well width `w`, in metres. |
| `frequencies` | Frequencies of the spectrum, in hertz (1-D). |
| `depths` | Well depths `d_n` of one period, in metres (1-D, at least two wells). |
| `reflection_of` | Reserved for future frequency-dependent reflection models; must be `None`. |
| `angles` | Receiver reflection angles `theta`, in degrees; defaults to [`DEFAULT_POLAR_ANGLES`](/phonometry/reference/api/materials/design/#default_polar_angles). |
| `source_angle` | Angle of incidence `psi`, in degrees. |
| `periods` | Number of repetitions `N_p` of the single period. |
| `speed_of_sound` | Speed of sound `c`, in metres per second. |
| `include_aperture` | Include the single-well aperture directivity. |
| `include_obliquity` | Include the Kirchhoff obliquity factor. |
| `normalize` | If `True` (default), also compute the normalised coefficient `d_n` against the flat reference. |

**Returns:** A [`DiffusionSpectrum`](/phonometry/reference/api/materials/scattering-diffusion/#diffusionspectrum) carrying the raw diffusion `d(f)` and, when `normalize` is `True`, the normalised `d_n(f)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for invalid geometry, an empty `frequencies` array, or if `reflection_of` is not `None`. |

## qrd_well_depths

```python
qrd_well_depths(
    prime: int,
    design_frequency: float,
    *,
    speed_of_sound: float = 343.0,
) -> Real
```

Well depths of a quadratic residue diffuser (Cox and D'Antonio, Eq. (10.3)).

$d_n = s_n \lambda_0 / (2N)$ with the quadratic residue sequence
`s_n` of [`quadratic_residue_sequence`](/phonometry/reference/api/materials/design/#quadratic_residue_sequence) and the design wavelength
$\lambda_0 = c / f_0$. The depths span 0 to roughly
$\lambda_0 / 2$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `prime` | Prime generator `N` (odd prime, at least 3). |
| `design_frequency` | Design frequency `f_0`, in hertz, normally the lower frequency limit of the device. |
| `speed_of_sound` | Speed of sound `c`, in metres per second; defaults to 343 m/s. |

**Returns:** Well depths `d_n`, in metres, one per well of a single period.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `prime` is not an odd prime, or if `design_frequency` or `speed_of_sound` is not positive. |

## quadratic_residue_sequence

```python
quadratic_residue_sequence(prime: int) -> Real
```

Quadratic residue sequence `s_n` (Cox and D'Antonio, Eq. (10.2)).

$s_n = n^2 \bmod N$ for $n = 0, 1, \ldots, N - 1$, the least
non-negative remainders, where `N` is the (odd) prime generator and
number of wells per period. For $N = 7$ this returns
`[0, 1, 4, 2, 2, 4, 1]`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `prime` | Prime generator `N` (odd prime, at least 3). |

**Returns:** The sequence `s_n` as an integer-valued float array of length `N`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `prime` is not an odd prime of at least 3. |
