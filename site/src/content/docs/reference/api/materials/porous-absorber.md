---
title: "materials.porous_absorber"
description: "Porous-material models and multilayer absorber prediction."
sidebar:
  label: "porous_absorber"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Porous-material models and multilayer absorber prediction.

Three complementary building blocks, all in the `e^{+j w t}` time
convention with the forward wave carried by `e^{-j k x}` (so a passive
medium has `Im(k) < 0`):

* **Equivalent-fluid models** for the characteristic impedance `Zc` and the
  complex wavenumber `k` of a rigid-frame porous material:

  - the one-parameter **Delany-Bazley** power law in the absorber variable
    `X = rho0 f / sigma` (Mechel, *Formulas of Acoustics* 2e, Sect. G.11
    Eqs. (1)-(2); Bies, Hansen & Howard, *Engineering Noise Control* 5e,
    Appendix D Eqs. (D.22)-(D.23) and Table D.1; Hopkins, *Sound Insulation*,
    Eqs. (1.171)-(1.174)), stated valid for `0.01 < X < 1.0` and porosity
    close to one. Table D.1 also provides coefficient sets fitted to
    polyester (Garai & Pompoli 2005) and to foams (Dunn & Davern 1986,
    Wu 1988), exposed here as presets.
  - the **Miki** modification, regressed on the same Delany-Bazley data under
    a positive-real (passivity) constraint so the model stays well behaved
    below the fit range (Miki 1990, *J. Acoust. Soc. Jpn (E)* 11(1),
    Eqs. (30)-(34), in the variable `f / sigma`).
  - the five-parameter **Johnson-Champoux-Allard (JCA)** semi-phenomenological
    model with flow resistivity, porosity, tortuosity and the viscous/thermal
    characteristic lengths (Cox & D'Antonio, *Acoustic Absorbers and
    Diffusers* 3e, Eqs. (6.19)-(6.25); Attenborough & Van Renterghem,
    *Predicting Outdoor Sound* 2e, Eqs. (5.13)-(5.14)). The returned
    equivalent-fluid density and bulk modulus are the surface-normalised
    quantities (they absorb the porosity), so `Zc = sqrt(rho_e K_e)` and
    `k = w sqrt(rho_e / K_e)` hold for every model.

* **Transfer-matrix multilayer prediction**: each fluid layer contributes
  `[[cos(kx d), j Zx sin(kx d)], [j sin(kx d)/Zx, cos(kx d)]]` with the
  in-depth wavenumber `kx = sqrt(k^2 - k0^2 sin^2 theta)` from Snell's law
  and `Zx = Zc k / kx` (Cox & D'Antonio Eqs. (2.29)-(2.32); Bies
  Eq. (D.83); equivalent to the layer-recursion of Bies Eq. (D.95) and
  Mechel Sect. D.4). Thin resonant sheets (perforated plate, microperforated
  plate, limp membrane) enter as series transfer impedances
  `[[1, z],[0, 1]]`. The stack is closed by a rigid wall, by free air or
  by an arbitrary termination impedance, giving the surface impedance, the
  oblique reflection factor and `alpha(theta)`. This same layer transfer
  matrix underlies the critically-coupled perfect-absorber designs of Jiménez,
  Groby, Pagneux & Romero-García (2017, *Applied Sciences* 7(6), 618,
  doi:10.3390/app7060618) and, for a rigidly-backed high-porosity layer,
  Jiménez, Romero-García & Groby (2018, *Acta Acustica united with Acustica*
  104(3), 396-409, doi:10.3813/AAA.919183), where the critical-coupling
  condition on the surface impedance yields total single-frequency absorption.

* **Resonant sheets and random incidence**: the perforated-plate impedance
  uses the end-corrected air-plug mass and the visco-thermal surface
  resistance (Cox & D'Antonio Eqs. (7.6)/(7.12)/(7.21), end-correction
  variants of Table 7.1); the microperforated plate follows Maa's exact
  short-tube impedance (Maa 1998, *J. Acoust. Soc. Am.* 104(5), Eq. (2),
  with the Eq. (5) end corrections; reproduced as Cox & D'Antonio
  Eqs. (7.33)-(7.35) and built on the same Bessel kernel as Mechel
  Sect. G.3); the membrane is the limp surface
  mass `j w m` (Cox & D'Antonio Eq. (7.14); Bies Eq. (D.96)). The
  random-incidence (Paris) integral follows Mechel Sect. D.5 Eqs. (9)-(10),
  with the closed form for locally reacting surfaces implemented in
  [`statistical_absorption`](/phonometry/reference/api/materials/porous-absorber/#statistical_absorption) (its maximum over passive impedances is the
  published 0.951).

## AirLayer

```python
AirLayer(thickness: float)
```

A plain air gap of `thickness` metres inside the stack.

## delany_bazley

```python
delany_bazley(
    frequency: ArrayLike,
    flow_resistivity: float,
    *,
    coefficients: str | tuple[float, ...] = 'delany_bazley',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> PorousMediumResult
```

Delany-Bazley one-parameter porous model (power laws in `X`).

`Zc = rho c (1 + C1 X^-C2 - j C3 X^-C4)` and
`k = (w/c)(1 + C5 X^-C6 - j C7 X^-C8)` with `X = rho f / sigma`
(Mechel 2e Sect. G.11 Eqs. (1)-(2); Bies 5e Eqs. (D.22)-(D.23) with the
Table D.1 coefficients; Hopkins Eqs. (1.171)-(1.173)). A
[`PorousAbsorberWarning`](/phonometry/reference/api/materials/porous-absorber/#porousabsorberwarning) is raised when any `X` leaves the stated
`0.01 < X < 1.0` validity range (Hopkins Eq. (1.174)); the values are
still returned.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `flow_resistivity` | Airflow resistivity `sigma`, in Pa s/m2. |
| `coefficients` | Preset name from [`DELANY_BAZLEY_COEFFICIENTS`](/phonometry/reference/api/materials/porous-absorber/#delany_bazley_coefficients) (`"delany_bazley"` rockwool/fibreglass default, `"garai_pompoli"` polyester, `"dunn_davern"` / `"wu"` foams) or an explicit `(C1..C8)` tuple. |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |

**Returns:** A [`PorousMediumResult`](/phonometry/reference/api/materials/porous-absorber/#porousmediumresult).

## DELANY_BAZLEY_COEFFICIENTS

*Constant* (`dict`).

```python
DELANY_BAZLEY_COEFFICIENTS = {'delany_bazley': (0.0571, 0.754, 0.087, 0.732, 0.0978, 0.7, 0.189, 0.595), 'garai_pompoli': (0.078, 0.623, 0.074, 0.66, 0.159, 0.571, 0.121, 0.53), 'dunn_davern': (0.114, 0.369, 0.0985, 0.758, 0.168, 0.715, 0.136, 0.491), 'wu': (0.212, 0.455, 0.105, 0.607, 0.163, 0.592, 0.188, 0.544)}
```

## DELANY_BAZLEY_VALIDITY

*Constant* (`tuple`).

```python
DELANY_BAZLEY_VALIDITY = (0.01, 1.0)
```

## diffuse_field_absorption

```python
diffuse_field_absorption(
    frequency: ArrayLike,
    layers: list[Layer] | tuple[Layer, ...],
    *,
    angle_limit: float = 1.5707963267948966,
    quadrature_points: int = 64,
    termination: str | complex | ArrayLike = 'rigid',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
) -> DiffuseFieldAbsorptionResult
```

Random-incidence absorption by the Paris integral (Mechel Sect. D.5).

`alpha_dif = (2 / sin^2 theta_lim) * int_0^theta_lim alpha(theta)
cos(theta) sin(theta) d(theta)` (Mechel 2e Sect. D.5 Eq. (9)), evaluated
with fixed-order Gauss-Legendre quadrature over the bulk-reacting
`alpha(theta)` of [`layered_absorber`](/phonometry/reference/api/materials/porous-absorber/#layered_absorber) (Sect. D.6 notes the bulk
integral generally must be evaluated numerically). Some references
truncate the integral at 75-87 degrees instead of 90 (Sect. D.5); set
`angle_limit` accordingly.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `layers` | Layer stack, as in [`layered_absorber`](/phonometry/reference/api/materials/porous-absorber/#layered_absorber). |
| `angle_limit` | Upper integration angle `theta_lim`, in radians (0 \< theta_lim \<= pi/2; default pi/2). |
| `quadrature_points` | Gauss-Legendre order (default 64). |
| `termination` | As in [`layered_absorber`](/phonometry/reference/api/materials/porous-absorber/#layered_absorber). |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |
| `viscosity` | Dynamic viscosity of air, in Pa s. |

**Returns:** A [`DiffuseFieldAbsorptionResult`](/phonometry/reference/api/materials/porous-absorber/#diffusefieldabsorptionresult).

## DiffuseFieldAbsorptionResult

```python
DiffuseFieldAbsorptionResult(
    frequency: Real,
    absorption: Real,
    angle_limit: float,
)
```

Random-incidence (Paris-integral) absorption of a layered absorber.

`absorption` is `alpha_dif(f)` from Mechel 2e Sect. D.5 Eq. (9):
the plane-wave `alpha(theta)` weighted by `cos(theta) sin(theta)` and
normalised by `sin^2(theta_limit)`.

### DiffuseFieldAbsorptionResult.plot()

```python
DiffuseFieldAbsorptionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the random-incidence absorption spectrum `alpha_dif(f)`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## helmholtz_resonance_frequency

```python
helmholtz_resonance_frequency(
    *,
    cavity_depth: float,
    plate_thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float | None = None,
    speed_of_sound: float = 343.0,
) -> float
```

Resonance of a perforated sheet over a shallow cavity (closed form).

`f0 = (c / 2 pi) sqrt(eps / (t' d))` with the end-corrected plug length
`t' = t + 2 delta a` (Cox & D'Antonio 3e, Eqs. (7.4)/(7.6), valid for
`k d << 1`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `cavity_depth` | Cavity depth `d`, in metres. |
| `plate_thickness` | Plate thickness `t`, in metres. |
| `hole_radius` | Hole radius `a`, in metres. |
| `open_area` | Fractional open area `eps` (0..1). |
| `end_correction` | End-correction factor `delta` per end; default [`perforation_end_correction`](/phonometry/reference/api/materials/porous-absorber/#perforation_end_correction) of `eps`. |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |

**Returns:** Resonance frequency `f0`, in hertz.

## johnson_champoux_allard

```python
johnson_champoux_allard(
    frequency: ArrayLike,
    flow_resistivity: float,
    *,
    porosity: float,
    tortuosity: float,
    viscous_length: float,
    thermal_length: float,
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
    prandtl_number: float = 0.71,
    heat_capacity_ratio: float = 1.4,
    atmospheric_pressure: float = 101325.0,
) -> PorousMediumResult
```

Johnson-Champoux-Allard five-parameter rigid-frame model.

Effective density (Cox & D'Antonio 3e, Eq. (6.19)):

`rho_e = (T rho / phi) [1 + (sigma phi / (j w rho T))
sqrt(1 + 4 j T^2 eta rho w / (sigma^2 L^2 phi^2))]`

and effective bulk modulus (Eq. (6.20)):

`K_e = (gamma P0 / phi) / (gamma - (gamma - 1) [1 +
(8 eta / (j L'^2 Pr w rho)) sqrt(1 + j rho w Pr L'^2 / (16 eta))]^-1)`

with tortuosity `T`, porosity `phi`, viscous/thermal characteristic
lengths `L` / `L'`; then `Zc = sqrt(K_e rho_e)` and
`k = w sqrt(rho_e / K_e)` (Eqs. (6.24)-(6.25)). Both quantities are
surface-normalised (the `1/phi` factors are included). The model has
the exact limits `j w rho_e -> sigma` as `w -> 0` and
`rho_e -> (T rho / phi)(1 + (1 - j) delta_v / L)` as `w -> inf`
(Johnson et al. 1987), pinned in the tests.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `flow_resistivity` | Airflow resistivity `sigma`, in Pa s/m2. |
| `porosity` | Open porosity `phi` (0 \< phi \<= 1). |
| `tortuosity` | High-frequency tortuosity `T = alpha_inf` (>= 1). |
| `viscous_length` | Viscous characteristic length `L`, in metres. |
| `thermal_length` | Thermal characteristic length `L'`, in metres (physically `L' >= L`). |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |

**Returns:** A [`PorousMediumResult`](/phonometry/reference/api/materials/porous-absorber/#porousmediumresult).

## layered_absorber

```python
layered_absorber(
    frequency: ArrayLike,
    layers: list[Layer] | tuple[Layer, ...],
    *,
    angle: float = 0.0,
    termination: str | complex | ArrayLike = 'rigid',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
) -> LayeredAbsorberResult
```

Transfer-matrix prediction of a layered absorber at one angle.

The *layers* list is ordered from the sound-incidence side towards the
*termination*. Fluid layers ([`AirLayer`](/phonometry/reference/api/materials/porous-absorber/#airlayer), [`PorousLayer`](/phonometry/reference/api/materials/porous-absorber/#porouslayer))
contribute the oblique chain matrix of Cox & D'Antonio 3e Eq. (2.29)
(equivalently the impedance recursion of Bies 5e Eq. (D.95) and the
scheme of Mechel 2e Sect. D.4); sheet layers ([`PerforatedPlateLayer`](/phonometry/reference/api/materials/porous-absorber/#perforatedplatelayer),
[`MicroperforatedPlateLayer`](/phonometry/reference/api/materials/porous-absorber/#microperforatedplatelayer), [`MembraneLayer`](/phonometry/reference/api/materials/porous-absorber/#membranelayer)) enter as
locally reacting series impedances. The chain is closed by a rigid wall
(`termination="rigid"`), by radiation into free air behind
(`termination="free"`, `Z_L = rho c / cos(theta)`) or by an arbitrary
complex impedance. The reflection factor is
`R = (Zs cos(theta) - rho c) / (Zs cos(theta) + rho c)` and
`alpha = 1 - |R|^2` (Mechel 2e Sect. D.3 Eq. (2)).

`Zs`, `R` and `alpha` are evaluated with the numerically robust
admittance recursion (algebraically identical to the chain product but
immune to the `e^{|Im(kx)| d}` overflow of the raw matrix entries for
extremely attenuating layers); the raw chain matrix is still returned in
`transfer_matrix` and may overflow in such extreme cases.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `layers` | Layer stack from the incidence side to the termination. |
| `angle` | Polar angle of incidence `theta`, in radians (`0 <= theta < pi/2 - 1e-6`; grazing incidence is excluded). |
| `termination` | `"rigid"` (default), `"free"`, or a non-zero complex impedance (scalar or per-frequency array), in Pa s/m. |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |
| `viscosity` | Dynamic viscosity of air, in Pa s (sheet layers). |

**Returns:** A [`LayeredAbsorberResult`](/phonometry/reference/api/materials/porous-absorber/#layeredabsorberresult).

## LayeredAbsorberResult

```python
LayeredAbsorberResult(
    frequency: Real,
    angle: float,
    surface_impedance: Complex,
    normalized_impedance: Complex,
    reflection: Complex,
    absorption: Real,
    transfer_matrix: Complex,
)
```

Oblique-incidence prediction of a layered absorber.

All arrays share the shape of `frequency`. `surface_impedance` is the
specific impedance `Zs = p / u_n` at the front face (may be `inf`
for a lossless-sheet stack over a rigid wall), `reflection` the complex
plane-wave reflection factor `R(theta)`, `absorption` the coefficient
`alpha(theta) = 1 - |R|^2` and `transfer_matrix` the total chain
matrix with shape `(2, 2, len(frequency))` (unimodular: every layer is
reciprocal).

### LayeredAbsorberResult.plot()

```python
LayeredAbsorberResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the absorption spectrum `alpha(f)` with `|R|` overlaid.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## membrane_impedance

```python
membrane_impedance(
    frequency: ArrayLike,
    *,
    surface_density: float,
    resistance: float = 0.0,
) -> Complex
```

Transfer impedance of a limp impervious membrane.

`z = r + j w m` - the surface-mass reactance (Cox & D'Antonio 3e,
Eq. (7.14); Bies 5e Eq. (D.96)) plus an optional empirical resistance
for the internal/fixing losses.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `surface_density` | Mass per unit area `m`, in kg/m2. |
| `resistance` | Series flow resistance `r`, in Pa s/m (default 0). |

**Returns:** Complex transfer impedance `z`, in Pa s/m.

## membrane_resonance_frequency

```python
membrane_resonance_frequency(
    *,
    surface_density: float,
    cavity_depth: float,
    isothermal: bool = False,
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> float
```

Mass-spring resonance of a membrane over a shallow cavity.

`f0 = (1 / 2 pi) sqrt(rho c^2 / (m d))` for an adiabatic air spring -
numerically the classical `f0 = 60 / sqrt(m d)` (Cox & D'Antonio 3e,
Eq. (7.9)). With `isothermal=True` the spring stiffness drops by
`gamma`, giving `~50 / sqrt(m d)` (Eq. (7.10)), the porous-filled
cavity case below about 500 Hz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `surface_density` | Membrane mass per unit area `m`, in kg/m2. |
| `cavity_depth` | Cavity depth `d`, in metres. |
| `isothermal` | Use the isothermal air-spring stiffness. |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |

**Returns:** Resonance frequency `f0`, in hertz.

## MembraneLayer

```python
MembraneLayer(surface_density: float, resistance: float = 0.0)
```

A limp impervious membrane (see [`membrane_impedance`](/phonometry/reference/api/materials/porous-absorber/#membrane_impedance)).

## microperforated_plate_impedance

```python
microperforated_plate_impedance(
    frequency: ArrayLike,
    *,
    thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float = 0.85,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
) -> Complex
```

Transfer impedance of a microperforated plate (Maa's exact model).

The specific impedance of one submillimetre hole is the exact short-tube
result (Maa 1998, Eq. (2); reproduced as Cox & D'Antonio 3e Eq. (7.33)
and the same Bessel kernel as Mechel 2e Sect. G.3):

`z1 = j w rho t [1 - (2 / (x sqrt(-j))) J1(x sqrt(-j)) / J0(x sqrt(-j))]^-1`

with the perforate constant `x = a sqrt(rho w / eta)`. Dividing by the
open area and adding Maa's Eq. (5) end corrections - the Rayleigh/Ingard
surface resistance `sqrt(2 w rho eta) / (2 eps)` and the piston
end-correction reactance `j w rho (2 delta a) / eps` (`0.85 d` total
for the default `delta = 0.85` per end) - gives the sheet transfer
impedance (Cox & D'Antonio Eq. (7.35)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `thickness` | Plate thickness `t`, in metres. |
| `hole_radius` | Hole radius `a`, in metres (submillimetre for a genuine microperforated design). |
| `open_area` | Fractional open area `eps` (0..1). |
| `end_correction` | End-correction factor `delta` per end (default 0.85, the isolated-orifice value used by Maa). |
| `air_density` | Air density `rho`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |

**Returns:** Complex transfer impedance `z`, in Pa s/m.

## MicroperforatedPlateLayer

```python
MicroperforatedPlateLayer(
    thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float = 0.85,
)
```

A microperforated plate (see [`microperforated_plate_impedance`](/phonometry/reference/api/materials/porous-absorber/#microperforated_plate_impedance)).

## miki

```python
miki(
    frequency: ArrayLike,
    flow_resistivity: float,
    *,
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> PorousMediumResult
```

Miki (1990) positive-real modification of the Delany-Bazley model.

In the variable `Y = f / sigma` (Miki 1990, Eqs. (30)-(34)):
`Zc = rho c (1 + 0.070 Y^-0.632 - j 0.107 Y^-0.632)` and, from the
propagation constant `gamma = alpha + j beta` via `k = beta - j alpha`,
`k = (w/c)(1 + 0.109 Y^-0.618 - j 0.160 Y^-0.618)`. The regression was
constrained to be positive real, so the surface impedance of a
hard-backed layer keeps a non-negative real part even below the
Delany-Bazley range; a [`PorousAbsorberWarning`](/phonometry/reference/api/materials/porous-absorber/#porousabsorberwarning) still flags
`Y` outside the fit range `0.01 < f/sigma < 1.0` (paper Sect. 4.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `flow_resistivity` | Airflow resistivity `sigma`, in Pa s/m2. |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |

**Returns:** A [`PorousMediumResult`](/phonometry/reference/api/materials/porous-absorber/#porousmediumresult).

## MIKI_VALIDITY

*Constant* (`tuple`).

```python
MIKI_VALIDITY = (0.01, 1.0)
```

## perforated_plate_impedance

```python
perforated_plate_impedance(
    frequency: ArrayLike,
    *,
    thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float | None = None,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
) -> Complex
```

Transfer impedance of a rigid perforated plate with circular holes.

Acoustic mass with both end corrections and the boundary-layer term
(Cox & D'Antonio 3e, Eq. (7.6)):

`m = (rho/eps)[t + 2 delta a + sqrt(8 nu / w)(1 + t/(2a))]`

and visco-thermal surface resistance (Eq. (7.12)):

`r = (rho/eps) sqrt(8 nu w) (1 + t/(2a))`,

giving `z = r + j w m` (the series impedance added on top of the
backing, Eq. (7.21)). Assumes hole radii well above the boundary-layer
thickness; use [`microperforated_plate_impedance`](/phonometry/reference/api/materials/porous-absorber/#microperforated_plate_impedance) for submillimetre
holes.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `thickness` | Plate thickness `t`, in metres. |
| `hole_radius` | Hole radius `a`, in metres. |
| `open_area` | Fractional open area `eps` (0..1). |
| `end_correction` | End-correction factor `delta` per end; default [`perforation_end_correction`](/phonometry/reference/api/materials/porous-absorber/#perforation_end_correction) of `eps`. |
| `air_density` | Air density `rho`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |

**Returns:** Complex transfer impedance `z`, in Pa s/m.

## PerforatedPlateLayer

```python
PerforatedPlateLayer(
    thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float | None = None,
)
```

A rigid perforated plate (see [`perforated_plate_impedance`](/phonometry/reference/api/materials/porous-absorber/#perforated_plate_impedance)).

## perforation_end_correction

```python
perforation_end_correction(open_area: float) -> float
```

End-correction factor `delta` of a circular perforation.

`delta = 0.85 (1 - 1.47 eps^1/2 + 0.47 eps^3/2)` - the Fok-function
interaction correction for circular holes (Cox & D'Antonio 3e, Table 7.1,
Nesterov row; no open-area limit). Each orifice end adds `delta a` of
air-plug length, and `delta -> 0.85` for an isolated hole.

**Parameters**

| Name | Description |
| :--- | :--- |
| `open_area` | Fractional open area `eps` of the sheet (0..1). |

**Returns:** End-correction factor `delta` (dimensionless, per end).

## PorousAbsorberWarning

Advisory for porous-model use outside the published fit range.

## PorousLayer

```python
PorousLayer(thickness: float, medium: PorousMediumResult)
```

A porous layer of `thickness` metres described by *medium*.

`medium` is a [`PorousMediumResult`](/phonometry/reference/api/materials/porous-absorber/#porousmediumresult) (from [`delany_bazley`](/phonometry/reference/api/materials/porous-absorber/#delany_bazley),
[`miki`](/phonometry/reference/api/materials/porous-absorber/#miki), [`johnson_champoux_allard`](/phonometry/reference/api/materials/porous-absorber/#johnson_champoux_allard), or built directly from
measured `Zc`/`k` data) evaluated on the same frequency vector that
is passed to [`layered_absorber`](/phonometry/reference/api/materials/porous-absorber/#layered_absorber).

## PorousMediumResult

```python
PorousMediumResult(
    frequency: Real,
    characteristic_impedance: Complex,
    wavenumber: Complex,
    effective_density: Complex,
    bulk_modulus: Complex,
    model: str,
    flow_resistivity: float,
    speed_of_sound: float,
    air_density: float,
)
```

Equivalent-fluid characterisation of a porous material.

All arrays share the shape of `frequency`. `characteristic_impedance`
is the complex characteristic impedance `Zc` in Pa s/m as seen from the
material surface, `wavenumber` the complex wavenumber `k` in rad/m
(`Im(k) < 0` for the `e^{+j w t}` convention),
`effective_density = Zc k / w` and `bulk_modulus = Zc w / k` the
surface-normalised equivalent-fluid density and bulk modulus, so that
`Zc = sqrt(rho_e K_e)` and `k = w sqrt(rho_e / K_e)` for every model.

### PorousMediumResult.normalized_impedance

*property*

Characteristic impedance normalised by `rho c` of air.

### PorousMediumResult.normalized_wavenumber

*property*

Wavenumber normalised by the free-air wavenumber `k0 = w / c`.

### PorousMediumResult.plot()

```python
PorousMediumResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the normalised `Zc` and `k` components against frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## statistical_absorption

```python
statistical_absorption(
    normalized_impedance: ArrayLike,
    *,
    angle_limit: float = 1.5707963267948966,
) -> Real
```

Closed-form Paris integral for a locally reacting plane.

With the normalised surface admittance `Z0 G = g1 + j g2 = 1/z`
(Mechel 2e Sect. D.5 Eq. (10)):

`alpha_dif = (8 g1 / sin^2 T) [1 - cos T
+ ((g1^2 - g2^2)/g2)(arctan((1 + g1)/g2) - arctan((g1 + cos T)/g2))
+ g1 ln((g1^2 + g2^2 + 2 g1 cos T + cos^2 T)/(1 + g1^2 + g2^2 + 2 g1))]`

reducing for `T = pi/2` to Eq. (4) and, for real admittance, to the
printed `g2 = 0` special case. The maximum over passive impedances is
0.951 (the published bound for locally reacting absorbers, Sect. D.5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `normalized_impedance` | Normalised surface impedance `z = Zs / (rho c)` (complex scalar or array), with `Re(z) > 0`. |
| `angle_limit` | Upper integration angle `theta_lim`, in radians (0 \< theta_lim \<= pi/2; default pi/2). |

**Returns:** Statistical absorption coefficient `alpha_dif`.
