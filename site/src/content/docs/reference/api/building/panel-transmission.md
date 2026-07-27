---
title: "building.panel_transmission"
description: "Predicted airborne sound reduction index of panels (Bies, Hansen & Howard 2017, Engineering Noise Control 5e, Section 7.2; Sharp 1973)."
sidebar:
  label: "panel_transmission"
---

Predicted airborne sound reduction index of panels (Bies, Hansen & Howard
2017, Engineering Noise Control 5e, Section 7.2; Sharp 1973).

Where EN 12354-1 ([`phonometry.building.building_prediction`](/phonometry/reference/api/building/building-prediction/)) takes the
element sound reduction index `R` as a *measured* input, this module
**predicts** `R(f)` from the physical properties of the construction: the mass
per unit area, bending stiffness (through the coincidence frequency) and loss
factor. The prediction feeds the same ISO 717-1 weighting
([`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating)) as the measured quantities, closing the
chain from panel physics to the single-number `Rw`.

**Mass law (Bies Eq. 7.40/7.42).** A non-stiff panel transmits by forced motion;
the transmission coefficient of an infinite limp panel gives the normal- and
field-incidence transmission loss:

```text
TL_normal = 10 lg(1 + (pi f m'' / (rho0 c0))**2)
TL_field  = TL_normal - dB(band)
```

with `m''` the mass per unit area, `rho0 c0` the characteristic impedance of
air and the field-incidence correction `dB = 5.5` dB for one-third-octave or
`4.0` dB for octave bands (Eq. 7.42). The mass law rises 6 dB per octave and
6 dB per doubling of mass.

**Single panel, Sharp's method (Bies 7.2.4.1).** Below the coincidence region
the field-incidence mass law holds; from the coincidence frequency `fc`
upwards the loss factor `eta` controls the transmission (Eq. 7.44):

```text
TL = 10 lg(1 + (pi f m'' / rho0 c0)**2) + 10 lg(2 eta f / (pi fc))
```

and between `fc/2` and `fc` the curve is a straight line on `TL` versus
`log10 f`. The coincidence dip at `fc` sits `10 lg(2 eta / pi)` below the
extrapolated mass law (Bies design-chart point B,
`TL = 20 lg(fc m'') + 10 lg eta - 44`).

**Double wall (Bies 7.2.6, Eq. 7.62-7.64).** Two leaves `m1`, `m2` separated
by a gap `d` behave as a mass-spring-mass system. Below the resonance
`f0 = (1/2 pi) sqrt(s'' (m1 + m2)/(m1 m2))` the pair follows the mass law of
the combined mass `m1 + m2`; above it the two mass laws add, boosted by the
cavity (Eq. 7.64):

```text
TL = TL_M                              , f <= f0
TL = TL_1 + TL_2 + 20 lg(2 k d)        , f0 < f < f_l   (k = 2 pi f / c0)
TL = TL_1 + TL_2 + 6                   , f >= f_l = c0 / (2 pi d)
```

The cavity stiffness `s''` is `rho0 c0**2 / d` for an empty (adiabatic) air
gap; a porous fill (a [`PorousMediumResult`](/phonometry/reference/api/materials/porous-absorber/#porousmediumresult) from
[`phonometry.materials.porous_absorber`](/phonometry/reference/api/materials/porous-absorber/)) lowers the resonance through its
softer, near-isothermal effective bulk modulus and damps the cavity so the
mid-band slope is realised without standing-wave dips.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## double_wall_transmission_loss

```python
double_wall_transmission_loss(
    frequency: ArrayLike,
    mass1: float,
    mass2: float,
    gap: float,
    *,
    loss_factor: float = 0.1,
    cavity_medium: PorousMediumResult | None = None,
    band: str = 'third',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> SoundReductionResult
```

Sound reduction index of a double wall (Bies 7.2.6, Eq. 7.64).

Piecewise Sharp model: below the mass-spring-mass resonance `f0` the pair
behaves as the mass law of the combined mass; between `f0` and the
limiting frequency `f_l = c0/(2 pi d)` the two mass laws add plus
`20 lg(2 k d)`; above `f_l` they add plus 6 dB. The curve is continuous
at `f_l` (`20 lg(2 k d) = 6` there).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (array, > 0). |
| `mass1` | Surface density of leaf 1 `m1`, in kg/m^2 (> 0). |
| `mass2` | Surface density of leaf 2 `m2`, in kg/m^2 (> 0). |
| `gap` | Cavity depth `d`, in m (> 0). |
| `loss_factor` | Leaf loss factor `eta` (> 0, Default: 0.1); reserved for the coincidence extension and reported for reference. |
| `cavity_medium` | Optional porous fill; see [`mass_spring_mass_resonance`](/phonometry/reference/api/building/panel-transmission/#mass_spring_mass_resonance). |
| `band` | Band width for the field correction (`"third"`/`"octave"`). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** A [`SoundReductionResult`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult) (model `"double-wall"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## field_incidence_correction

```python
field_incidence_correction(band: str = 'third') -> float
```

Field-incidence mass-law correction `dB` (Bies Eq. 7.42).

**Parameters**

| Name | Description |
| :--- | :--- |
| `band` | `"third"` (5.5 dB) or `"octave"` (4.0 dB). |

**Returns:** The correction subtracted from the normal-incidence mass law, dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown band width. |

## mass_law_transmission_loss

```python
mass_law_transmission_loss(
    frequency: ArrayLike,
    mass_per_area: float,
    *,
    incidence: str = 'field',
    band: str = 'third',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> np.ndarray
```

Mass-law transmission loss of a limp panel (Bies Eq. 7.40/7.42).

`TL_normal = 10 lg(1 + (pi f m'' / rho0 c0)**2)`; the field-incidence
value subtracts the band correction of [`field_incidence_correction`](/phonometry/reference/api/building/panel-transmission/#field_incidence_correction).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array, > 0). |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2 (> 0). |
| `incidence` | `"normal"` or `"field"` (Default: `"field"`). |
| `band` | Band width for the field correction (`"third"`/`"octave"`). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** The transmission loss `TL`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or unknown incidence/band. |

## mass_spring_mass_resonance

```python
mass_spring_mass_resonance(
    mass1: float,
    mass2: float,
    gap: float,
    *,
    cavity_medium: PorousMediumResult | None = None,
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> float
```

Mass-spring-mass resonance `f0` of a double wall (Bies Eq. 7.62).

`f0 = (1/2 pi) sqrt(s'' (m1 + m2)/(m1 m2))` with the cavity stiffness per
unit area `s''`. For an empty air gap `s'' = rho0 c0**2 / d` (adiabatic,
Hopkins Eq. 4.72); with a porous *cavity_medium* the fill's effective
(near-isothermal) bulk modulus at the lowest supplied frequency sets a
softer `s'' = Re(K_e) / d`, lowering `f0`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass1` | Surface density of leaf 1 `m1`, in kg/m^2 (> 0). |
| `mass2` | Surface density of leaf 2 `m2`, in kg/m^2 (> 0). |
| `gap` | Cavity depth `d`, in m (> 0). |
| `cavity_medium` | Optional porous fill (a [`PorousMediumResult`](/phonometry/reference/api/materials/porous-absorber/#porousmediumresult)) whose effective bulk modulus sets the cavity stiffness. |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** The mass-spring-mass resonance `f0`, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## plot_double_wall_geometry

```python
plot_double_wall_geometry(
    mass1: float,
    mass2: float,
    gap: float,
    ax: Axes | None = None,
    *,
    resonance_frequency: float | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the mass-spring-mass double wall to scale.

Two leaves separated by the `gap`; leaf thicknesses are drawn from the
surface densities at a nominal board density, and the mass-spring-mass
resonance is annotated when given.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass1` | Surface density of the first leaf, in kg/m2. |
| `mass2` | Surface density of the second leaf, in kg/m2. |
| `gap` | Cavity depth, in metres. |
| `ax` | Existing axes, or `None` to create a figure. |
| `resonance_frequency` | Optional `f0` to annotate, in Hz. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the leaf rectangles. |

**Returns:** The axes.

## single_panel_transmission_loss

```python
single_panel_transmission_loss(
    frequency: ArrayLike,
    mass_per_area: float,
    *,
    critical_frequency: float | None = None,
    bending_stiffness: float | None = None,
    loss_factor: float = 0.01,
    band: str = 'third',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> SoundReductionResult
```

Sound reduction index of a single panel, Sharp's method (Bies 7.2.4.1).

Field-incidence mass law up to `fc/2`, Eq. 7.44 from `fc` upwards, and a
straight line in `log10 f` across the coincidence region between them.

Provide the coincidence frequency directly through *critical_frequency*, or
let it be computed from *bending_stiffness* and *mass_per_area* through
[`coincidence_frequency`](/phonometry/reference/api/vibration/radiation-efficiency/#coincidence_frequency).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (array, > 0). |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2 (> 0). |
| `critical_frequency` | Coincidence frequency `fc`, in hertz (> 0). |
| `bending_stiffness` | Bending stiffness per unit width `B'`, in N.m, used to compute `fc` when *critical_frequency* is not given. |
| `loss_factor` | Total loss factor `eta` (> 0, Default: 0.01). |
| `band` | Band width for the field correction (`"third"`/`"octave"`). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** A [`SoundReductionResult`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult) (model `"sharp-single"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input, or if neither *critical_frequency* nor *bending_stiffness* is given. |

## SoundReductionResult

```python
SoundReductionResult(
    frequencies: np.ndarray,
    transmission_loss: np.ndarray,
    model: str,
    critical_frequency: float | None = None,
    resonance_frequency: float | None = None,
    mass1: float | None = None,
    mass2: float | None = None,
    gap: float | None = None,
)
```

Predicted airborne sound reduction index `R(f)` of a construction.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in hertz. |
| `transmission_loss` | Sound reduction index `R` per band, in dB. |
| `model` | Prediction model (e.g. `"sharp-single"`, `"double-wall"`). |
| `critical_frequency` | Coincidence frequency `fc`, in hertz, or `None` (double wall reports the mass-spring-mass resonance instead). |
| `resonance_frequency` | Mass-spring-mass resonance `f0`, in hertz, or `None` (single panel). |
| `mass1` | First-leaf surface density, in kg/m2, retained (with `mass2` and `gap`) by the double-wall constructor so `plot_geometry` can draw the section; `None` otherwise. |
| `mass2` | Second-leaf surface density, in kg/m2, or `None`. |
| `gap` | Cavity depth, in metres, or `None`. |

### SoundReductionResult.plot()

```python
SoundReductionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the predicted sound reduction index `R(f)`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### SoundReductionResult.plot_geometry()

```python
SoundReductionResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the mass-spring-mass cross-section to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

### SoundReductionResult.rating()

```python
SoundReductionResult.rating(
    bands: str | None = None,
) -> WeightedRatingResult
```

Single-number weighted rating `Rw` of the predicted `R(f)`.

Delegates to [`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating) (ISO 717-1); requires
the spectrum to be on the 16 one-third-octave bands (100 Hz to
3150 Hz) or the 5 octave bands (125 Hz to 2000 Hz).

**Parameters**

| Name | Description |
| :--- | :--- |
| `bands` | Band set forwarded to [`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating). |

**Returns:** The [`WeightedRatingResult`](/phonometry/reference/api/building/insulation/#weightedratingresult).

### SoundReductionResult.report()

```python
SoundReductionResult.report(path: str, **kwargs: Any) -> str
```

Render the ISO 717-1 Annex C rating fiche of `R(f)` to a PDF.

Convenience wrapper delegating to
[`report`](/phonometry/reference/api/building/insulation/)
on `rating`; requires the predicted spectrum to be on the 16
one-third-octave bands (100 Hz to 3150 Hz) or the 5 octave bands
(125 Hz to 2000 Hz).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `kwargs` | Forwarded to [`report`](/phonometry/reference/api/building/insulation/) (e.g. `engine`). |

**Returns:** The written `path` as a `str`.

### SoundReductionResult.transmission_coefficient

*property*

Transmission coefficient `tau = 10**(-R/10)` per band.
