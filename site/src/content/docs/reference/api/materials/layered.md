---
title: "materials.absorbers.layered"
description: "Declarative layer stacks and the transfer-matrix absorber solver."
sidebar:
  label: "layered"
---

Declarative layer stacks and the transfer-matrix absorber solver.

An absorber is declared as a list of layers ordered from the sound-incidence
side towards the termination and solved at one angle, in the same
$e^{+j \omega t}$ time convention as the element models of
[`porous`](/phonometry/reference/api/materials/porous/), with the forward wave carried
by $e^{-j k x}$ (so a passive medium has
$\operatorname{Im}(k) < 0$):

* **Transfer-matrix multilayer prediction**: each fluid layer contributes
  $[[\cos(k_x d), jZ_x\sin(k_x d)], [j\sin(k_x d)/Z_x, \cos(k_x d)]]$
  with the in-depth wavenumber
  $k_x = \sqrt{k^2 - k_0^2 \sin^2 \theta}$ from Snell's law and
  $Z_x = Z_\mathrm{c} k / k_x$ (Cox & D'Antonio Eqs. (2.29)-(2.32); Bies
  Eq. (D.83); equivalent to the layer-recursion of Bies Eq. (D.95) and
  Mechel Sect. D.4). Thin resonant sheets (perforated plate, microperforated
  plate, limp membrane) enter as series transfer impedances
  $[[1, z], [0, 1]]$. The stack is closed by a rigid wall, by free air
  or by an arbitrary termination impedance, giving the surface impedance,
  the oblique reflection factor and $\alpha(\theta)$. This same
  layer transfer matrix underlies the critically-coupled perfect-absorber
  designs of Jiménez,
  Groby, Pagneux & Romero-García (2017, *Applied Sciences* 7(6), 618,
  doi:10.3390/app7060618) and, for a rigidly-backed high-porosity layer,
  Jiménez, Romero-García & Groby (2018, *Acta Acustica united with Acustica*
  104(3), 396-409, doi:10.3813/AAA.919183), where the critical-coupling
  condition on the surface impedance yields total single-frequency absorption.

* **Random incidence**: the random-incidence (Paris) integral follows Mechel
  Sect. D.5 Eqs. (9)-(10), with the closed form for locally reacting surfaces
  implemented in [`statistical_absorption`](/phonometry/reference/api/materials/layered/#statistical_absorption) (its maximum over passive
  impedances is the published 0.951).

The elements a stack is built from live elsewhere: the equivalent fluid a
[`PorousLayer`](/phonometry/reference/api/materials/layered/#porouslayer) carries and the sheet impedances the plate and membrane
layers evaluate come from
[`porous`](/phonometry/reference/api/materials/porous/), and the three Biot waves a
[`PoroelasticLayer`](/phonometry/reference/api/materials/layered/#poroelasticlayer) carries come from
[`biot`](/phonometry/reference/api/materials/biot/).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AirLayer

```python
AirLayer(thickness: float)
```

A plain air gap of `thickness` metres inside the stack.

## diffuse_field_absorption

```python
diffuse_field_absorption(
    frequency: ArrayLike,
    layers: list[Layer] | tuple[Layer, ...],
    *,
    angle_limit: float = 1.5707963267948966,
    quadrature_points: int = 64,
    termination: str | complex | ArrayLike = 'rigid',
    fluid: Fluid = ...,
) -> DiffuseFieldAbsorptionResult
```

Random-incidence absorption by the Paris integral (Mechel Sect. D.5).

$$
\alpha_{\mathrm{dif}} = \frac{2}{\sin^2 \theta_{\mathrm{lim}}} \int_0^{\theta_{\mathrm{lim}}} \alpha(\theta) \cos(\theta) \sin(\theta) \, d\theta
$$

(Mechel 2e Sect. D.5 Eq. (9)), evaluated
with fixed-order Gauss-Legendre quadrature over the bulk-reacting
$\alpha(\theta)$ of [`layered_absorber`](/phonometry/reference/api/materials/layered/#layered_absorber) (Sect. D.6 notes the
bulk integral generally must be evaluated numerically). Some references
truncate the integral at 75-87 degrees instead of 90 (Sect. D.5); set
`angle_limit` accordingly.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `layers` | Layer stack, as in [`layered_absorber`](/phonometry/reference/api/materials/layered/#layered_absorber). |
| `angle_limit` | Upper integration angle `theta_lim`, in radians (0 \< theta_lim \<= pi/2; default pi/2). |
| `quadrature_points` | Gauss-Legendre order (default 64). |
| `termination` | As in [`layered_absorber`](/phonometry/reference/api/materials/layered/#layered_absorber). |
| `fluid` | The medium, a [`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid) (Default: [`PUBLISHED_AIR`](/phonometry/reference/api/materials/porous/#published_air), the air this model was published with). Pass a computed one, such as `fluids.air(temperature_c=30.0, relative_humidity_percent=70.0)`, to work in the air of the room. |

**Returns:** A [`DiffuseFieldAbsorptionResult`](/phonometry/reference/api/materials/layered/#diffusefieldabsorptionresult).

## DiffuseFieldAbsorptionResult

```python
DiffuseFieldAbsorptionResult(
    frequency: Real,
    absorption: Real,
    angle_limit: float,
)
```

Random-incidence (Paris-integral) absorption of a layered absorber.

`absorption` is $\alpha_{\mathrm{dif}}(f)$ from Mechel 2e
Sect. D.5 Eq. (9): the plane-wave $\alpha(\theta)$ weighted by
$\cos(\theta) \sin(\theta)$ and normalised by
$\sin^2(\theta_{\mathrm{limit}})$.

### DiffuseFieldAbsorptionResult.plot()

```python
DiffuseFieldAbsorptionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the random-incidence absorption spectrum
$\alpha_{\mathrm{dif}}(f)$.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## layered_absorber

```python
layered_absorber(
    frequency: ArrayLike,
    layers: list[Layer] | tuple[Layer, ...],
    *,
    angle: float = 0.0,
    termination: str | complex | ArrayLike = 'rigid',
    fluid: Fluid = ...,
) -> LayeredAbsorberResult
```

Transfer-matrix prediction of a layered absorber at one angle.

The *layers* list is ordered from the sound-incidence side towards the
*termination*. Fluid layers ([`AirLayer`](/phonometry/reference/api/materials/layered/#airlayer), [`PorousLayer`](/phonometry/reference/api/materials/layered/#porouslayer))
contribute the oblique chain matrix of Cox & D'Antonio 3e Eq. (2.29)
(equivalently the impedance recursion of Bies 5e Eq. (D.95) and the
scheme of Mechel 2e Sect. D.4); sheet layers ([`PerforatedPlateLayer`](/phonometry/reference/api/materials/layered/#perforatedplatelayer),
[`MicroperforatedPlateLayer`](/phonometry/reference/api/materials/layered/#microperforatedplatelayer), [`MembraneLayer`](/phonometry/reference/api/materials/layered/#membranelayer)) enter as
locally reacting series impedances; a [`PoroelasticLayer`](/phonometry/reference/api/materials/layered/#poroelasticlayer) carries the
three Biot waves of its elastic frame and switches the whole stack to the
six-variable global-matrix assembly of Allard & Atalla 2e Sect. 11.5, with
the coupling matrices of Sect. 11.4. The chain is closed by a rigid wall
(`termination="rigid"`), by radiation into free air behind
(`termination="free"`, $Z_L = \rho c / \cos(\theta)$) or by an
arbitrary complex impedance. The reflection factor is
$R = (Z_\mathrm{s} \cos(\theta) - \rho c) / (Z_\mathrm{s} \cos(\theta) + \rho c)$
and $\alpha = 1 - \lvert R \rvert^2$ (Mechel 2e Sect. D.3
Eq. (2)).

`Zs`, `R` and `alpha` are evaluated with the numerically robust
admittance recursion (algebraically identical to the chain product but
immune to the $e^{\lvert \operatorname{Im}(k_x) \rvert d}$
overflow of the raw matrix entries for
extremely attenuating layers); the raw chain matrix is still returned in
`transfer_matrix` and may overflow in such extreme cases.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `layers` | Layer stack from the incidence side to the termination. |
| `angle` | Polar angle of incidence `theta`, in radians ($0 \le \theta < \pi/2 - 10^{-6}$; grazing incidence is excluded). |
| `termination` | `"rigid"` (default), `"free"`, or a non-zero complex impedance (scalar or per-frequency array), in Pa s/m. |
| `fluid` | The medium, a [`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid) (Default: [`PUBLISHED_AIR`](/phonometry/reference/api/materials/porous/#published_air), the air this model was published with). Pass a computed one, such as `fluids.air(temperature_c=30.0, relative_humidity_percent=70.0)`, to work in the air of the room. |

**Returns:** A [`LayeredAbsorberResult`](/phonometry/reference/api/materials/layered/#layeredabsorberresult).

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
    layers: tuple[Layer, ...] | None = None,
)
```

Oblique-incidence prediction of a layered absorber.

All arrays share the shape of `frequency`. `surface_impedance` is the
specific impedance $Z_\mathrm{s} = p / u_n$ at the front face (may be
`inf` for a lossless-sheet stack over a rigid wall), `reflection`
the complex plane-wave reflection factor $R(\theta)$,
`absorption` the coefficient
$\alpha(\theta) = 1 - \lvert R \rvert^2$ and `transfer_matrix`
the total chain matrix with shape `(2, 2, len(frequency))`
(unimodular: every layer is reciprocal).

`layers` retains the layer sequence the stack was solved with (front
layer first) so `plot_geometry` can draw the cross-section; it is
appended after the original fields and defaults to `None` for
hand-built results.

### LayeredAbsorberResult.plot()

```python
LayeredAbsorberResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the absorption spectrum $\alpha(f)$ with
$\lvert R \rvert$ overlaid.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### LayeredAbsorberResult.plot_geometry()

```python
LayeredAbsorberResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the solved stack cross-section to scale (dimensioned).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its `layers`. |

## MembraneLayer

```python
MembraneLayer(surface_density: float, resistance: float = 0.0)
```

A limp impervious membrane (see [`membrane_impedance`](/phonometry/reference/api/materials/porous/#membrane_impedance)).

## MicroperforatedPlateLayer

```python
MicroperforatedPlateLayer(
    thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float = 0.85,
)
```

A microperforated plate (see [`microperforated_plate_impedance`](/phonometry/reference/api/materials/porous/#microperforated_plate_impedance)).

## PerforatedPlateLayer

```python
PerforatedPlateLayer(
    thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float | None = None,
)
```

A rigid perforated plate (see [`perforated_plate_impedance`](/phonometry/reference/api/materials/porous/#perforated_plate_impedance)).

## PoroelasticLayer

```python
PoroelasticLayer(
    thickness: float,
    medium: PorousMediumResult,
    porosity: float,
    tortuosity: float,
    frame_density: float,
    shear_modulus: complex,
    poisson_ratio: float = 0.0,
)
```

A porous layer whose frame is elastic (full Biot theory).

Where [`PorousLayer`](/phonometry/reference/api/materials/layered/#porouslayer) collapses the material into a single wave in an
equivalent fluid, this layer carries the three Biot waves of Allard &
Atalla 2e chapter 6 - two compressional and one shear - so the frame can
resonate. It is the only layer type that reproduces the quarter-wavelength
frame resonance of [`frame_quarter_wave_resonance`](/phonometry/reference/api/materials/biot/#frame_quarter_wave_resonance),
and the only one for which an air gap behind the layer, a bonded backing or
an oblique angle change the frame motion rather than only the pore fluid.

`medium` is the **rigid-frame** equivalent fluid of the pores (normally a
[`johnson_champoux_allard`](/phonometry/reference/api/materials/porous/#johnson_champoux_allard) result on the solver's frequency vector):
the frame inertia is added by the Biot model itself, so a limp-corrected
medium would count it twice. The remaining fields describe the frame.

Adding one of these to a stack switches [`layered_absorber`](/phonometry/reference/api/materials/layered/#layered_absorber) to the
global-matrix assembly of Allard & Atalla Sect. 11.5. Two adjacent
poroelastic layers are coupled as *bonded* frames (their Eq. (11.67)); a
sheet layer next to a poroelastic layer is coupled as a free, mechanically
decoupled screen (air on both sides, their Sect. 11.3.6).

## PorousLayer

```python
PorousLayer(thickness: float, medium: PorousMediumResult)
```

A porous layer of `thickness` metres described by *medium*.

`medium` is a [`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult) (from [`delany_bazley`](/phonometry/reference/api/materials/porous/#delany_bazley),
[`miki`](/phonometry/reference/api/materials/porous/#miki), [`johnson_champoux_allard`](/phonometry/reference/api/materials/porous/#johnson_champoux_allard), or built directly from
measured `Zc`/`k` data) evaluated on the same frequency vector that
is passed to [`layered_absorber`](/phonometry/reference/api/materials/layered/#layered_absorber).

## statistical_absorption

```python
statistical_absorption(
    normalized_impedance: ArrayLike,
    *,
    angle_limit: float = 1.5707963267948966,
) -> Real
```

Closed-form Paris integral for a locally reacting plane.

With the normalised surface admittance $Z_0 G = g_1 + j g_2 = 1/z$
(Mechel 2e Sect. D.5 Eq. (10)):

$$
\alpha_{\mathrm{dif}} = \frac{8 g_1}{\sin^2 T} \left[1 - \cos T + \frac{g_1^2 - g_2^2}{g_2} \left(\arctan\frac{1 + g_1}{g_2} - \arctan\frac{g_1 + \cos T}{g_2}\right) + g_1 \ln\frac{g_1^2 + g_2^2 + 2 g_1 \cos T + \cos^2 T} {1 + g_1^2 + g_2^2 + 2 g_1}\right]
$$

reducing for $T = \pi/2$ to Eq. (4) and, for real admittance, to
the printed $g_2 = 0$ special case. The maximum over passive
impedances is 0.951 (the published bound for locally reacting absorbers,
Sect. D.5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `normalized_impedance` | Normalised surface impedance $z = Z_\mathrm{s} / (\rho c)$ (complex scalar or array), with $\operatorname{Re}(z) > 0$. |
| `angle_limit` | Upper integration angle `theta_lim`, in radians (0 \< theta_lim \<= pi/2; default pi/2). |

**Returns:** Statistical absorption coefficient `alpha_dif`.
