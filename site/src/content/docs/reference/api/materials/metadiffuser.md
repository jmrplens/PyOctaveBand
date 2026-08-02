---
title: "materials.diffusers.metadiffuser"
description: "Metadiffusers: deep-subwavelength Schroeder-like sound diffusers."
sidebar:
  label: "metadiffuser"
---

Metadiffusers: deep-subwavelength Schroeder-like sound diffusers.

A metadiffuser is a rigidly backed slotted panel whose slits are each loaded
by an array of Helmholtz resonators (Jimenez, Cox, Romero-Garcia and Groby,
*Metadiffusers: Deep-subwavelength sound diffusers*, Sci. Rep. 7, 5389,
2017). The resonators slow the sound inside each slit, so the slit reaches
its quarter-wavelength condition at a fraction of the depth a plain well
would need; by giving every slit a different geometry the panel presents a
spatially dependent complex reflection coefficient `R_n(f)` along its
face. Tuning that profile to a Schroeder phase grating reproduces the
scattering of a quadratic-residue or primitive-root diffuser from a panel
1/46 to 1/20 of the design wavelength thick, and driving single slits to
critical coupling adds the perfectly absorbing `0` state that ternary
sequences require.

Each slit is modelled with the transfer-matrix chain of
[`slit_helmholtz_absorber`](/phonometry/reference/api/materials/slow-sound/#slit_helmholtz_absorber)
(visco-thermal effective parameters, resonator end corrections and slit
radiation correction); the panel is locally reacting, so the wells do not
couple internally and the far field follows from the Fraunhofer integral of
the per-well reflection sequence
([`predict_diffuser_polar_response`](/phonometry/reference/api/materials/design/#predict_diffuser_polar_response))
reduced to the ISO 17497-2 directional diffusion coefficient.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## metadiffuser_diffusion_spectrum

```python
metadiffuser_diffusion_spectrum(
    frequencies: ArrayLike,
    wells: Sequence[MetadiffuserWell | None],
    *,
    depth: float,
    period: float,
    angles: ArrayLike = (-90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90),
    source_angle: float = 0.0,
    periods: int = 1,
    resonator_geometry: str = 'slit',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
    prandtl_number: float = 0.71,
    heat_capacity_ratio: float = 1.4,
    atmospheric_pressure: float = 101325.0,
) -> DiffusionSpectrum
```

Normalized diffusion-coefficient spectrum `d_n(f)` of a metadiffuser.

Evaluates the far-field polar response at each frequency with
[`metadiffuser_polar_response`](/phonometry/reference/api/materials/metadiffuser/#metadiffuser_polar_response), forms the ISO 17497-2 directional
diffusion coefficient band by band and normalises it against a flat
rigid reference of the same footprint (all wells $R = 1$) with
[`normalized_diffusion_coefficient`](/phonometry/reference/api/materials/scattering-diffusion/#normalized_diffusion_coefficient),
exactly as the paper reports `delta_n`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies of the spectrum, in hertz (1-D). |
| `wells` | Sequence of [`MetadiffuserWell`](/phonometry/reference/api/materials/metadiffuser/#metadiffuserwell) (or `None` for a flat rigid strip) describing one period of the panel face. |
| `depth` | Panel depth `L` common to all slits, in metres. |
| `period` | Well pitch `d` along the panel face, in metres. |
| `angles` | Receiver reflection angles `theta`, in degrees. |
| `source_angle` | Angle of incidence `psi`, in degrees. |
| `periods` | Number of repetitions `N_p` of the single period. |
| `resonator_geometry` | `"slit"` (default) for the paper's two-dimensional resonators, `"square"` for square-duct necks and cavities. |
| `speed_of_sound` | Speed of sound `c0` in air, in m/s. |
| `air_density` | Air density `rho0`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |

**Returns:** A [`DiffusionSpectrum`](/phonometry/reference/api/materials/scattering-diffusion/#diffusionspectrum) carrying the raw `d(f)` and the normalised `d_n(f)`.

## metadiffuser_polar_response

```python
metadiffuser_polar_response(
    frequency: float,
    wells: Sequence[MetadiffuserWell | None],
    *,
    depth: float,
    period: float,
    angles: ArrayLike = (-90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90),
    source_angle: float = 0.0,
    periods: int = 1,
    resonator_geometry: str = 'slit',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
    prandtl_number: float = 0.71,
    heat_capacity_ratio: float = 1.4,
    atmospheric_pressure: float = 101325.0,
) -> DiffuserPolarResponse
```

Far-field polar response of a metadiffuser at one frequency.

Computes the per-well complex reflection sequence at `frequency` with
[`metadiffuser_reflection`](/phonometry/reference/api/materials/metadiffuser/#metadiffuser_reflection) (the panel is locally reacting, so the
slit chains see the incidence angle `source_angle`) and evaluates the
Fraunhofer far field and ISO 17497-2 directional diffusion coefficient
with
[`predict_diffuser_polar_response`](/phonometry/reference/api/materials/design/#predict_diffuser_polar_response).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency of the prediction `f`, in hertz. |
| `wells` | Sequence of [`MetadiffuserWell`](/phonometry/reference/api/materials/metadiffuser/#metadiffuserwell) (or `None` for a flat rigid strip) describing one period of the panel face. |
| `depth` | Panel depth `L` common to all slits, in metres. |
| `period` | Well pitch `d` along the panel face, in metres; it is the `well_width` of the far-field model. |
| `angles` | Receiver reflection angles `theta`, in degrees. |
| `source_angle` | Angle of incidence `psi` of the source, in degrees; also applied to the local slit reflection. |
| `periods` | Number of repetitions `N_p` of the single period; the grating lobes of a Schroeder-like design require `periods >= 2`. |
| `resonator_geometry` | `"slit"` (default) for the paper's two-dimensional resonators, `"square"` for square-duct necks and cavities. |
| `speed_of_sound` | Speed of sound `c0` in air, in m/s. |
| `air_density` | Air density `rho0`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |

**Returns:** A [`DiffuserPolarResponse`](/phonometry/reference/api/materials/design/#diffuserpolarresponse).

## metadiffuser_reflection

```python
metadiffuser_reflection(
    frequency: ArrayLike,
    wells: Sequence[MetadiffuserWell | None],
    *,
    depth: float,
    period: float,
    angle: float = 0.0,
    resonator_geometry: str = 'slit',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
    prandtl_number: float = 0.71,
    heat_capacity_ratio: float = 1.4,
    atmospheric_pressure: float = 101325.0,
) -> MetadiffuserResult
```

Per-well reflection spectra of a metadiffuser panel (Sci. Rep. Eq. (6)).

Runs the rigidly backed slit transfer-matrix chain of
[`slit_helmholtz_absorber`](/phonometry/reference/api/materials/slow-sound/#slit_helmholtz_absorber)
once per well: each [`MetadiffuserWell`](/phonometry/reference/api/materials/metadiffuser/#metadiffuserwell) becomes a slit of height
`h_n` and depth `L` loaded by its `M` resonators on the lattice
$a = L / M$, and `None` wells are flat rigid strips with
$R = 1$.
The panel is locally reacting, so a well's reflection does not depend on
its neighbours and the incidence `angle` enters only through the front
air impedance.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `wells` | Sequence of [`MetadiffuserWell`](/phonometry/reference/api/materials/metadiffuser/#metadiffuserwell) (or `None` for a flat rigid strip) describing one period of the panel face. |
| `depth` | Panel depth `L` common to all slits, in metres. |
| `period` | Well pitch `d` along the panel face, in metres. |
| `angle` | Polar angle of incidence `theta`, in radians. |
| `resonator_geometry` | `"slit"` (default) for the paper's two-dimensional resonators, `"square"` for square-duct necks and cavities. |
| `speed_of_sound` | Speed of sound `c0` in air, in m/s. |
| `air_density` | Air density `rho0`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |

**Returns:** A [`MetadiffuserResult`](/phonometry/reference/api/materials/metadiffuser/#metadiffuserresult) with one reflection row per well.

## MetadiffuserResult

```python
MetadiffuserResult(
    frequency: Real,
    reflection: Complex,
    absorption: Real,
    well_absorption: Real,
    wells: tuple[MetadiffuserWell | None, ...] | None = None,
    depth: float | None = None,
    period: float | None = None,
)
```

Spectra of a metadiffuser panel, one reflection row per well.

`reflection` has shape `(N, len(frequency))` with the complex
pressure reflection factor of each well (flat strips are exactly `1`),
`absorption` is the face-averaged energy absorption
$\alpha(f) = 1 - \operatorname{mean}_n \lvert R_n \rvert^2$ and
`well_absorption` the per-well
$\alpha_n = 1 - \lvert R_n \rvert^2$. The trailing fields retain the geometry the
prediction was run with (`wells`, `depth`, `period`) so
`plot_geometry` can draw the panel section; they default to
`None` for hand-built results.

### MetadiffuserResult.plot()

```python
MetadiffuserResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-well and face-averaged absorption spectra.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### MetadiffuserResult.plot_geometry()

```python
MetadiffuserResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the panel cross-section to scale (slits and resonators).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

## MetadiffuserWell

```python
MetadiffuserWell(
    slit_height: float,
    resonators: tuple[HelmholtzResonator, ...],
)
```

One slit of a metadiffuser panel.

`slit_height` is the slit opening `h_n` along the panel face and
`resonators` the Helmholtz resonators loading the slit, ordered from
the panel face towards the rigid backing; the resonator lattice step is
$a = L / M$ for a panel of depth `L` and `M` resonators. All
lengths are in metres. A `None` entry in a well sequence stands for a
flat rigid strip of the panel face ($R = 1$), the `+1` state of
ternary-sequence designs.
