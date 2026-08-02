---
title: "materials.absorbers.porous"
description: "Porous-material models and multilayer absorber prediction."
sidebar:
  label: "porous"
---

Porous-material models and multilayer absorber prediction.

Three complementary building blocks, all in the $e^{+j \omega t}$
time convention with the forward wave carried by $e^{-j k x}$ (so a
passive medium has $\operatorname{Im}(k) < 0$):

* **Equivalent-fluid models** for the characteristic impedance `Zc` and the
  complex wavenumber `k` of a rigid-frame porous material:

  - the one-parameter **Delany-Bazley** power law in the absorber variable
    $X = \rho_0 f / \sigma$ (Mechel, *Formulas of Acoustics* 2e,
    Sect. G.11 Eqs. (1)-(2); Bies, Hansen & Howard, *Engineering Noise
    Control* 5e, Appendix D Eqs. (D.22)-(D.23) and Table D.1; Hopkins,
    *Sound Insulation*, Eqs. (1.171)-(1.174)), stated valid for
    $0.01 < X < 1.0$ and porosity close to one. Table D.1 also
    provides coefficient sets fitted to polyester (Garai & Pompoli 2005)
    and to foams (Dunn & Davern 1986, Wu 1988), exposed here as presets.
  - the **Miki** modification, regressed on the same Delany-Bazley data under
    a positive-real (passivity) constraint so the model stays well behaved
    below the fit range (Miki 1990, *J. Acoust. Soc. Jpn (E)* 11(1),
    Eqs. (30)-(34), in the variable $f / \sigma$).
  - the five-parameter **Johnson-Champoux-Allard (JCA)** semi-phenomenological
    model with flow resistivity, porosity, tortuosity and the viscous/thermal
    characteristic lengths (Cox & D'Antonio, *Acoustic Absorbers and
    Diffusers* 3e, Eqs. (6.19)-(6.25); Attenborough & Van Renterghem,
    *Predicting Outdoor Sound* 2e, Eqs. (5.13)-(5.14)). The returned
    equivalent-fluid density and bulk modulus are the surface-normalised
    quantities (they absorb the porosity), so
    $Z_c = \sqrt{\rho_e K_e}$ and
    $k = \omega \sqrt{\rho_e / K_e}$ hold for every model.
  - the **limp-frame** correction of any of the three rigid-frame models
    (Allard & Atalla, *Propagation of Sound in Porous Media* 2e, Sect. 11.3.4,
    Eqs. (11.53)-(11.55), printed pp. 251-253): a light frame is dragged along
    by the pore fluid, so its inertia has to be carried by the equivalent
    fluid. Only the effective density changes; the bulk modulus is the
    rigid-frame one. See [`limp_frame`](/phonometry/reference/api/materials/porous/#limp_frame) and
    [`decoupling_frequency`](/phonometry/reference/api/materials/porous/#decoupling_frequency).

* **Transfer-matrix multilayer prediction**: each fluid layer contributes
  $[[\cos(k_x d), jZ_x\sin(k_x d)], [j\sin(k_x d)/Z_x, \cos(k_x d)]]$
  with the in-depth wavenumber
  $k_x = \sqrt{k^2 - k_0^2 \sin^2 \theta}$ from Snell's law and
  $Z_x = Z_c k / k_x$ (Cox & D'Antonio Eqs. (2.29)-(2.32); Bies
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

* **Resonant sheets and random incidence**: the perforated-plate impedance
  uses the end-corrected air-plug mass and the visco-thermal surface
  resistance (Cox & D'Antonio Eqs. (7.6)/(7.12)/(7.21), end-correction
  variants of Table 7.1); the microperforated plate follows Maa's exact
  short-tube impedance (Maa 1998, *J. Acoust. Soc. Am.* 104(5), Eq. (2),
  with the Eq. (5) end corrections; reproduced as Cox & D'Antonio
  Eqs. (7.33)-(7.35) and built on the same Bessel kernel as Mechel
  Sect. G.3); the membrane is the limp surface
  mass $j \omega m$ (Cox & D'Antonio Eq. (7.14); Bies Eq. (D.96)). The
  random-incidence (Paris) integral follows Mechel Sect. D.5 Eqs. (9)-(10),
  with the closed form for locally reacting surfaces implemented in
  [`statistical_absorption`](/phonometry/reference/api/materials/porous/#statistical_absorption) (its maximum over passive impedances is the
  published 0.951).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AirLayer

```python
AirLayer(thickness: float)
```

A plain air gap of `thickness` metres inside the stack.

## decoupling_frequency

```python
decoupling_frequency(
    flow_resistivity: float,
    *,
    porosity: float,
    frame_density: float,
) -> float
```

Zwikker-Kosten decoupling frequency `Fd` of a porous frame.

$F_d = \sigma \phi^2 / (2 \pi \rho_1)$ (Allard & Atalla 2e,
Sect. 11.3.4, printed p. 251; the same closed form as their Eq. (6.90),
printed p. 126).
Above `Fd` the visco-inertial coupling between the pore fluid and the
frame is too weak for the acoustic wave to shake the frame, so the
rigid-frame equivalent fluid of [`johnson_champoux_allard`](/phonometry/reference/api/materials/porous/#johnson_champoux_allard) applies;
below it the frame moves and the limp correction of [`limp_frame`](/phonometry/reference/api/materials/porous/#limp_frame)
matters.

**Parameters**

| Name | Description |
| :--- | :--- |
| `flow_resistivity` | Airflow resistivity `sigma`, in Pa s/m2 (> 0). |
| `porosity` | Open porosity `phi` (0 \< phi \<= 1). |
| `frame_density` | Bulk density of the frame `rho1`, in kg/m3 (> 0): the mass of solid per unit volume of material, i.e. the density of the sample as weighed, not the density of the material the fibres are made of. |

**Returns:** The decoupling frequency `Fd`, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or a porosity above 1. |

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

$Z_c = \rho c (1 + C_1 X^{-C_2} - j C_3 X^{-C_4})$ and
$k = (\omega/c)(1 + C_5 X^{-C_6} - j C_7 X^{-C_8})$ with
$X = \rho f / \sigma$
(Mechel 2e Sect. G.11 Eqs. (1)-(2); Bies 5e Eqs. (D.22)-(D.23) with the
Table D.1 coefficients; Hopkins Eqs. (1.171)-(1.173)). A
[`PorousAbsorberWarning`](/phonometry/reference/api/materials/porous/#porousabsorberwarning) is raised when any `X` leaves the stated
$0.01 < X < 1.0$ validity range (Hopkins Eq. (1.174)); the values
are still returned.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `flow_resistivity` | Airflow resistivity `sigma`, in Pa s/m2. |
| `coefficients` | Preset name from [`DELANY_BAZLEY_COEFFICIENTS`](/phonometry/reference/api/materials/porous/#delany_bazley_coefficients) (`"delany_bazley"` rockwool/fibreglass default, `"garai_pompoli"` polyester, `"dunn_davern"` / `"wu"` foams) or an explicit `(C1..C8)` tuple. |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |

**Returns:** A [`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult).

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

$$
\alpha_{\mathrm{dif}} = \frac{2}{\sin^2 \theta_{\mathrm{lim}}} \int_0^{\theta_{\mathrm{lim}}} \alpha(\theta) \cos(\theta) \sin(\theta) \, d\theta
$$

(Mechel 2e Sect. D.5 Eq. (9)), evaluated
with fixed-order Gauss-Legendre quadrature over the bulk-reacting
$\alpha(\theta)$ of [`layered_absorber`](/phonometry/reference/api/materials/porous/#layered_absorber) (Sect. D.6 notes the
bulk integral generally must be evaluated numerically). Some references
truncate the integral at 75-87 degrees instead of 90 (Sect. D.5); set
`angle_limit` accordingly.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `layers` | Layer stack, as in [`layered_absorber`](/phonometry/reference/api/materials/porous/#layered_absorber). |
| `angle_limit` | Upper integration angle `theta_lim`, in radians (0 \< theta_lim \<= pi/2; default pi/2). |
| `quadrature_points` | Gauss-Legendre order (default 64). |
| `termination` | As in [`layered_absorber`](/phonometry/reference/api/materials/porous/#layered_absorber). |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |
| `viscosity` | Dynamic viscosity of air, in Pa s. |

**Returns:** A [`DiffuseFieldAbsorptionResult`](/phonometry/reference/api/materials/porous/#diffusefieldabsorptionresult).

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

$f_0 = (c / 2 \pi) \sqrt{\varepsilon / (t' d)}$ with the
end-corrected plug length $t' = t + 2 \delta a$ (Cox & D'Antonio
3e, Eqs. (7.4)/(7.6), valid for $k d \ll 1$).

**Parameters**

| Name | Description |
| :--- | :--- |
| `cavity_depth` | Cavity depth `d`, in metres. |
| `plate_thickness` | Plate thickness `t`, in metres. |
| `hole_radius` | Hole radius `a`, in metres. |
| `open_area` | Fractional open area `eps` (0..1). |
| `end_correction` | End-correction factor `delta` per end; default [`perforation_end_correction`](/phonometry/reference/api/materials/porous/#perforation_end_correction) of `eps`. |
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

$$
\rho_e = \frac{T \rho}{\phi} \left[1 + \frac{\sigma \phi}{j \omega \rho T} \sqrt{1 + \frac{4 j T^2 \eta \rho \omega}{\sigma^2 L^2 \phi^2}} \right]
$$

and effective bulk modulus (Eq. (6.20)):

$$
K_e = \frac{\gamma P_0 / \phi}{\gamma - (\gamma - 1) \left[1 + \frac{8 \eta}{j {L'}^2 \mathrm{Pr}\, \omega \rho} \sqrt{1 + \frac{j \rho \omega \mathrm{Pr}\, {L'}^2}{16 \eta}} \right]^{-1}}
$$

with tortuosity `T`, porosity `phi`, viscous/thermal characteristic
lengths `L` / `L'`; then $Z_c = \sqrt{K_e \rho_e}$ and
$k = \omega \sqrt{\rho_e / K_e}$ (Eqs. (6.24)-(6.25)). Both
quantities are surface-normalised (the $1/\phi$ factors are
included). The model has the exact limits
$j \omega \rho_e \to \sigma$ as $\omega \to 0$ and
$\rho_e \to (T \rho / \phi)(1 + (1 - j) \delta_v / L)$ as
$\omega \to \infty$ (Johnson et al. 1987), pinned in the tests.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `flow_resistivity` | Airflow resistivity `sigma`, in Pa s/m2. |
| `porosity` | Open porosity `phi` (0 \< phi \<= 1). |
| `tortuosity` | High-frequency tortuosity $T = \alpha_\infty$ (>= 1). |
| `viscous_length` | Viscous characteristic length `L`, in metres. |
| `thermal_length` | Thermal characteristic length `L'`, in metres (physically $L' \ge L$). |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |

**Returns:** A [`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult).

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
*termination*. Fluid layers ([`AirLayer`](/phonometry/reference/api/materials/porous/#airlayer), [`PorousLayer`](/phonometry/reference/api/materials/porous/#porouslayer))
contribute the oblique chain matrix of Cox & D'Antonio 3e Eq. (2.29)
(equivalently the impedance recursion of Bies 5e Eq. (D.95) and the
scheme of Mechel 2e Sect. D.4); sheet layers ([`PerforatedPlateLayer`](/phonometry/reference/api/materials/porous/#perforatedplatelayer),
[`MicroperforatedPlateLayer`](/phonometry/reference/api/materials/porous/#microperforatedplatelayer), [`MembraneLayer`](/phonometry/reference/api/materials/porous/#membranelayer)) enter as
locally reacting series impedances; a [`PoroelasticLayer`](/phonometry/reference/api/materials/porous/#poroelasticlayer) carries the
three Biot waves of its elastic frame and switches the whole stack to the
six-variable global-matrix assembly of Allard & Atalla 2e Sect. 11.5, with
the coupling matrices of Sect. 11.4. The chain is closed by a rigid wall
(`termination="rigid"`), by radiation into free air behind
(`termination="free"`, $Z_L = \rho c / \cos(\theta)$) or by an
arbitrary complex impedance. The reflection factor is
$R = (Z_s \cos(\theta) - \rho c) / (Z_s \cos(\theta) + \rho c)$
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
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |
| `viscosity` | Dynamic viscosity of air, in Pa s (sheet layers). |

**Returns:** A [`LayeredAbsorberResult`](/phonometry/reference/api/materials/porous/#layeredabsorberresult).

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
specific impedance $Z_s = p / u_n$ at the front face (may be
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

## limp_frame

```python
limp_frame(
    medium: PorousMediumResult,
    frame_density: float,
    *,
    porosity: float = 1.0,
) -> PorousMediumResult
```

Limp-frame correction of a rigid-frame equivalent fluid (A&A 11.3.4).

A light frame (aeronautic-grade fibreglass, felts, screens) is dragged
along by the pore fluid instead of standing still, and the rigid-frame
models of [`delany_bazley`](/phonometry/reference/api/materials/porous/#delany_bazley), [`miki`](/phonometry/reference/api/materials/porous/#miki) and
[`johnson_champoux_allard`](/phonometry/reference/api/materials/porous/#johnson_champoux_allard) have no way to carry that inertia.
Neglecting the stiffness of the frame altogether in the Biot mixed
pressure-displacement formulation leaves an equivalent fluid with the same
bulk modulus and a corrected effective density (Allard & Atalla 2e,
Eqs. (11.53)-(11.55), printed pp. 252-253, after Panneton 2007):

$$
\rho_{\mathrm{limp}} = \frac{\rho_t \rho_{\mathrm{eq}} - \rho_0^2} {\rho_t + \rho_{\mathrm{eq}} - 2 \rho_0}
$$

with `rho_eq` the rigid-frame effective density of *medium*, `rho0`
the density of the pore fluid and $\rho_t = \rho_1 + \phi \rho_0$
the apparent total density of the material. What anchors this
expression is the printed
equation itself, transcribed term by term; Allard & Atalla tabulate no
computed limp density anywhere, so there are no published digits to check
against. The book also states two exact limits in prose, and both are
verified, but they are weaker than they look: neither pins the
$\rho_0^2$ and $2 \rho_0$ terms, since a sign-flipped
variant of Eq. (11.55) satisfies both of them (and even the
$1/\rho_1$ decay of the heavy-frame residual).
They corroborate the transcription rather than determine it:

* **heavy frame**: as $\rho_1 \to \infty$ the correction vanishes
  and the rigid-frame result is recovered (the book's own reading of
  Eq. (11.55));
* **low frequency**: since
  $\rho_{\mathrm{eq}} \to \sigma / (j \omega)$ as
  $\omega \to 0$ (Eq. (5.37)),
  $\rho_{\mathrm{limp}} \to \rho_t$, a finite real density, where
  the rigid-frame model diverges. The rigid frame forbids rigid-body
  motion of the sample; the limp one allows it, which is why the two
  differ mainly
  at low frequency and why the limp model is the right one for an
  unconstrained sample in an impedance tube.

The corrected medium is a drop-in
[`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult), so it can be handed to
[`PorousLayer`](/phonometry/reference/api/materials/porous/#porouslayer) inside [`layered_absorber`](/phonometry/reference/api/materials/porous/#layered_absorber) exactly like the
rigid-frame one.

Use [`decoupling_frequency`](/phonometry/reference/api/materials/porous/#decoupling_frequency) to see where the frame stops following the
fluid and [`limp_frame_applicable`](/phonometry/reference/api/materials/porous/#limp_frame_applicable) for the published bulk-modulus
rule of thumb on when the frame may be treated as limp at all.

**Parameters**

| Name | Description |
| :--- | :--- |
| `medium` | A rigid-frame [`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult) (its `effective_density` is `rho_eq` and its `bulk_modulus` is kept). |
| `frame_density` | Bulk density of the frame `rho1`, in kg/m3 (> 0). |
| `porosity` | Open porosity `phi` (0 \< phi \<= 1, Default: 1,0, the high-porosity assumption of the one-parameter models). |

**Returns:** A [`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult) with model `"limp_frame(<base model>)"`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or a porosity above 1. |

## limp_frame_applicable

```python
limp_frame_applicable(
    frame_bulk_modulus: float,
    *,
    criterion: str = 'doutres',
    fluid_bulk_modulus: float = 101325.0,
) -> bool
```

Whether the limp-frame model may be used, by published rule of thumb.

Both published criteria compare the bulk modulus of the frame *in vacuum*
`K_c` with that of the fluid in the pores `K_f` (Allard & Atalla 2e,
printed pp. 253-254): Beranek (1947) requires
$\lvert K_c/K_f \rvert < 0.05$, and the frame structural
interaction study of Doutres et al. (2007) relaxes it to
$\lvert K_c/K_f \rvert < 0.2$. With `K_f` taken as the
isothermal bulk modulus of air, $P_0 = 101.3$ kPa, the relaxed
criterion is the book's statement that
"the limp model is applicable for materials having a bulk modulus lower
than 20 kPa". Neither criterion accounts for boundary or mounting
conditions, and the book notes that a thin light foam decoupled from a
vibrating structure by an air gap behaves limply well above the limit.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frame_bulk_modulus` | Bulk modulus of the frame in vacuum `K_c`, in Pa (>= 0; pass `abs(K_c)` for a complex modulus). |
| `criterion` | Key into [`LIMP_FRAME_CRITERIA`](/phonometry/reference/api/materials/porous/#limp_frame_criteria), `"doutres"` (Default, 0,2) or `"beranek"` (0,05). |
| `fluid_bulk_modulus` | Bulk modulus of the pore fluid `K_f`, in Pa (Default: 101 325, the isothermal value for air). |

**Returns:** `True` when $\lvert K_c/K_f \rvert$ does not exceed the threshold.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a negative modulus or an unknown criterion. |

## LIMP_FRAME_CRITERIA

*Constant* (`dict`).

```python
LIMP_FRAME_CRITERIA = {'beranek': 0.05, 'doutres': 0.2}
```

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

$z = r + j \omega m$ - the surface-mass reactance (Cox & D'Antonio
3e, Eq. (7.14); Bies 5e Eq. (D.96)) plus an optional empirical
resistance for the internal/fixing losses.

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

$f_0 = (1 / 2 \pi) \sqrt{\rho c^2 / (m d)}$ for an adiabatic air
spring - numerically the classical $f_0 = 60 / \sqrt{m d}$ (Cox &
D'Antonio 3e, Eq. (7.9)). With `isothermal=True` the spring stiffness
drops by `gamma`, giving $\sim 50 / \sqrt{m d}$ (Eq. (7.10)),
the porous-filled cavity case below about 500 Hz.

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

A limp impervious membrane (see [`membrane_impedance`](/phonometry/reference/api/materials/porous/#membrane_impedance)).

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

$$
z_1 = j \omega \rho t \left[1 - \frac{2}{x \sqrt{-j}} \frac{J_1(x \sqrt{-j})}{J_0(x \sqrt{-j})}\right]^{-1}
$$

with the perforate constant $x = a \sqrt{\rho \omega / \eta}$.
Dividing by the open area and adding Maa's Eq. (5) end corrections - the
Rayleigh/Ingard surface resistance
$\sqrt{2 \omega \rho \eta} / (2 \varepsilon)$ and the piston
end-correction reactance
$j \omega \rho (2 \delta a) / \varepsilon$ ($0.85 d$ total
for the default $\delta = 0.85$ per end) - gives the sheet
transfer impedance (Cox & D'Antonio Eq. (7.35)).

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

A microperforated plate (see [`microperforated_plate_impedance`](/phonometry/reference/api/materials/porous/#microperforated_plate_impedance)).

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

In the variable $Y = f / \sigma$ (Miki 1990, Eqs. (30)-(34)):
$Z_c = \rho c (1 + 0.070 Y^{-0.632} - j 0.107 Y^{-0.632})$ and,
from the propagation constant $\gamma = \alpha + j \beta$ via
$k = \beta - j \alpha$,
$k = (\omega/c)(1 + 0.109 Y^{-0.618} - j 0.160 Y^{-0.618})$. The
regression was constrained to be positive real, so the surface impedance
of a hard-backed layer keeps a non-negative real part even below the
Delany-Bazley range; a [`PorousAbsorberWarning`](/phonometry/reference/api/materials/porous/#porousabsorberwarning) still flags
`Y` outside the fit range $0.01 < f/\sigma < 1.0$ (paper
Sect. 4.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `flow_resistivity` | Airflow resistivity `sigma`, in Pa s/m2. |
| `speed_of_sound` | Speed of sound `c` in air, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |

**Returns:** A [`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult).

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

$$
m = \frac{\rho}{\varepsilon} \left[t + 2 \delta a + \sqrt{\frac{8 \nu}{\omega}} \left(1 + \frac{t}{2a}\right)\right]
$$

and visco-thermal surface resistance (Eq. (7.12)):

$$
r = \frac{\rho}{\varepsilon} \sqrt{8 \nu \omega} \left(1 + \frac{t}{2a}\right)
$$

giving $z = r + j \omega m$ (the series impedance added on top of
the backing, Eq. (7.21)). Assumes hole radii well above the boundary-layer
thickness; use [`microperforated_plate_impedance`](/phonometry/reference/api/materials/porous/#microperforated_plate_impedance) for submillimetre
holes.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `thickness` | Plate thickness `t`, in metres. |
| `hole_radius` | Hole radius `a`, in metres. |
| `open_area` | Fractional open area `eps` (0..1). |
| `end_correction` | End-correction factor `delta` per end; default [`perforation_end_correction`](/phonometry/reference/api/materials/porous/#perforation_end_correction) of `eps`. |
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

A rigid perforated plate (see [`perforated_plate_impedance`](/phonometry/reference/api/materials/porous/#perforated_plate_impedance)).

## perforation_end_correction

```python
perforation_end_correction(open_area: float) -> float
```

End-correction factor `delta` of a circular perforation.

The Fok-function interaction correction for circular holes (Cox &
D'Antonio 3e, Table 7.1, Nesterov row; no open-area limit):

$$
\delta = 0.85 (1 - 1.47 \varepsilon^{1/2} + 0.47 \varepsilon^{3/2})
$$

Each orifice end adds $\delta a$ of air-plug length, and
$\delta \to 0.85$ for an isolated hole.

**Parameters**

| Name | Description |
| :--- | :--- |
| `open_area` | Fractional open area `eps` of the sheet (0..1). |

**Returns:** End-correction factor `delta` (dimensionless, per end).

## plot_absorber_stack

```python
plot_absorber_stack(
    layers: Sequence[Layer] | Layer,
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw a layered-absorber cross-section to scale, rigid backing at right.

Sound arrives from the left; each layer is drawn with its material fill
and its thickness dimensioned below the stack. A membrane (no physical
depth) is drawn as a thin sheet.

**Parameters**

| Name | Description |
| :--- | :--- |
| `layers` | The layer sequence of [`layered_absorber`](/phonometry/reference/api/materials/porous/#layered_absorber), front layer first, or a single layer. |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the front-layer rectangle. |

**Returns:** The axes.

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

Where [`PorousLayer`](/phonometry/reference/api/materials/porous/#porouslayer) collapses the material into a single wave in an
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

Adding one of these to a stack switches [`layered_absorber`](/phonometry/reference/api/materials/porous/#layered_absorber) to the
global-matrix assembly of Allard & Atalla Sect. 11.5. Two adjacent
poroelastic layers are coupled as *bonded* frames (their Eq. (11.67)); a
sheet layer next to a poroelastic layer is coupled as a free, mechanically
decoupled screen (air on both sides, their Sect. 11.3.6).

## PorousAbsorberWarning

Advisory for porous-model use outside the published fit range.

## PorousLayer

```python
PorousLayer(thickness: float, medium: PorousMediumResult)
```

A porous layer of `thickness` metres described by *medium*.

`medium` is a [`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult) (from [`delany_bazley`](/phonometry/reference/api/materials/porous/#delany_bazley),
[`miki`](/phonometry/reference/api/materials/porous/#miki), [`johnson_champoux_allard`](/phonometry/reference/api/materials/porous/#johnson_champoux_allard), or built directly from
measured `Zc`/`k` data) evaluated on the same frequency vector that
is passed to [`layered_absorber`](/phonometry/reference/api/materials/porous/#layered_absorber).

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
($\operatorname{Im}(k) < 0$ for the $e^{+j \omega t}$
convention), `effective_density` $= Z_c k / \omega$ and
`bulk_modulus` $= Z_c \omega / k$ the surface-normalised
equivalent-fluid density and bulk modulus, so that
$Z_c = \sqrt{\rho_e K_e}$ and
$k = \omega \sqrt{\rho_e / K_e}$ for every model.

### PorousMediumResult.normalized_impedance

*property*

Characteristic impedance normalised by $\rho c$ of air.

### PorousMediumResult.normalized_wavenumber

*property*

Wavenumber normalised by the free-air wavenumber
$k_0 = \omega / c$.

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
| `normalized_impedance` | Normalised surface impedance $z = Z_s / (\rho c)$ (complex scalar or array), with $\operatorname{Re}(z) > 0$. |
| `angle_limit` | Upper integration angle `theta_lim`, in radians (0 \< theta_lim \<= pi/2; default pi/2). |

**Returns:** Statistical absorption coefficient `alpha_dif`.
