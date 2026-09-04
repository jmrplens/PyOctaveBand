---
title: "materials.absorbers.slow_sound"
description: "Slow-sound slit panels loaded with Helmholtz resonators (perfect absorbers)."
sidebar:
  label: "slow_sound"
---

Slow-sound slit panels loaded with Helmholtz resonators (perfect absorbers).

A rigid panel perforated by a periodic array of thin closed slits, whose upper
wall is loaded by an array of Helmholtz resonators (HRs), behaves as a
deep-subwavelength, locally reacting sound absorber. The resonators slow the
sound inside the slit, pulling the slit resonance down to the deep
subwavelength regime, and the intrinsic visco-thermal losses can be tuned to
exactly balance the leakage of the structure (critical coupling), giving
perfect absorption at a chosen frequency and angle. The model follows the
transfer-matrix treatment of Jimenez, Groby, Pagneux and Romero-Garcia
(*Iridescent Perfect Absorption in Critically-Coupled Acoustic Metamaterials
Using the Transfer Matrix Method*, Appl. Sci. 2017, 7, 618) together with the
resonator model and end corrections detailed in the supplementary material of
Jimenez, Huang, Romero-Garcia, Pagneux and Groby (*Ultra-thin metamaterial for
perfect and quasi-omnidirectional sound absorption*, Appl. Phys. Lett. 2016,
109, 121902).

The building blocks, all in the `e^{+j w t}` convention used throughout
phonometry (a passive medium has $\operatorname{Im}(k) < 0$):

* **Visco-thermal effective parameters.** The slit of height `h` uses the
  narrow-channel effective density and bulk modulus (Appl. Sci. Eq. (6);
  Appl. Phys. Lett. Eqs. (A1)-(A2)):

  $\rho_\mathrm{s} = \rho_0 [1 - \tanh((h/2) G_\rho) / ((h/2) G_\rho)]^{-1}$
  and $\kappa_\mathrm{s} = \kappa_0 [1 + (\gamma - 1) \tanh((h/2) G_\kappa) / ((h/2) G_\kappa)]^{-1}$

  with $G_\rho = \sqrt{j \omega \rho_0 / \eta}$ and
  $G_\kappa = \sqrt{j \omega \mathrm{Pr} \rho_0 / \eta}$. The square
  necks and cavities use the rectangular-duct series of Stinson (1991),
  reproduced as Appl. Sci. Eqs. (7)-(8) with the transverse wavenumbers
  $\alpha_k = (2k+1) \pi / a$ and $\beta_m = (2m+1) \pi / b$.
  The duct series is printed in the opposite time convention of the source;
  it is returned conjugated here so the neck and cavity share the
  $e^{+j\omega t}$ passivity of the slit. Both models are pinned in the
  tests to their exact limits: the effective density tends to `rho0` and the
  bulk modulus to `kappa0` as the boundary layers vanish, and
  $j\omega\rho$ tends to the Poiseuille flow resistivity of the channel
  as $\omega \to 0$ ($12\eta/h^2$ for the slit,
  $28.454\,\eta/w^2$ for a square duct).

* **Helmholtz-resonator impedance.** Each resonator is a neck (length `l_n`,
  side `w_n`) over a closed cavity (length `l_c`, side `w_c`); its
  impedance follows Appl. Phys. Lett. Eq. (A23) with the neck-to-cavity
  radiation end correction of Eqs. (A24)-(A26).

* **Transfer matrix.** The panel is the chain
  `M_dl (M_s M_HR M_s)...` of half-lattice slit steps (Appl. Sci. Eq. (2)),
  resonators as point shunt scatterers (Eq. (3)) and the slit-radiation end
  correction (Eq. (3)/(A27)). The slit-radiation series impedance is printed
  in the sources as `-i w dl_slit rho0 / (phi_t S0)`; like the duct series
  it is used conjugated here (`+j w`), so it acts as the added radiation
  mass it models and lowers the slit-panel resonance. The rigidly-backed
  reflection factor is
  $R = (T_{11} \cos(\theta) - Z_0 T_{21}) / (T_{11} \cos(\theta) + Z_0 T_{21})$ with
  $Z_0 = \rho_0 c_0 / S_0$ (Eq. (4)), and
  $\alpha = 1 - \lvert R \rvert^2$. Perfect absorption (critical
  coupling) is reached when the reflection zero sits on the real-frequency
  axis, i.e. $\operatorname{Re}(Z) \cos(\theta) = Z_0$ and
  $\operatorname{Im}(Z) = 0$ with $Z = T_{11} / T_{21}$ the
  acoustic surface impedance (Eq. (9)).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## critical_coupling_design

```python
critical_coupling_design(
    target_frequency: float,
    resonator: HelmholtzResonator,
    *,
    lattice_step: float,
    period: float,
    angle: float = 0.0,
    slit_height_bounds: tuple[float, float] = (0.0002, 0.005),
    cavity_length_bounds: tuple[float, float] = (0.002, 0.2),
    end_correction: bool = True,
    slit_radiation: bool = True,
    fluid: Fluid = ...,
) -> CriticalCouplingResult
```

Solve resonator/slit geometry for perfect absorption at a frequency.

Critical coupling (perfect absorption) requires the acoustic surface
impedance $Z = T_{11} / T_{21}$ of the rigidly-backed panel to
satisfy $\operatorname{Re}(Z) \cos(\theta) = Z_0$ and
$\operatorname{Im}(Z) = 0$ at `target_frequency`
(Appl. Sci. 2017 Eq. (9)), i.e. the reflection zero lies on the
real-frequency axis. Holding the neck geometry and cavity side of
`resonator` fixed, this tunes the cavity length (which sets the
resonance frequency) and the slit height (which sets the visco-thermal
leakage balance) to meet both conditions, so `alpha ~ 1` at the design
point.

**Parameters**

| Name | Description |
| :--- | :--- |
| `target_frequency` | Design frequency `f0`, in hertz. |
| `resonator` | Base geometry; its `cavity_length` is used as the initial guess and its neck and cavity side are held fixed. |
| `lattice_step` | Resonator lattice step `a`, in metres. |
| `period` | Slit array period `d`, in metres. |
| `angle` | Design angle of incidence `theta`, in radians. |
| `slit_height_bounds` | Search bounds for the slit height, in metres. |
| `cavity_length_bounds` | Search bounds for the cavity length, in metres. |
| `end_correction` | Include the resonator radiation end corrections. |
| `slit_radiation` | Include the slit-to-free-air radiation correction. |
| `fluid` | State of the air the panel is designed for ([`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid)): its speed of sound `c0`, density `rho0`, viscosity `eta`, Prandtl number `Pr`, ratio of specific heats `gamma` and static pressure `P0`. |

**Returns:** A [`CriticalCouplingResult`](/phonometry/reference/api/materials/slow-sound/#criticalcouplingresult). A [`SlowSoundAbsorberWarning`](/phonometry/reference/api/materials/slow-sound/#slowsoundabsorberwarning) is emitted (via `warnings.warn`) if the solver does not reach perfect absorption within tolerance.

## CriticalCouplingResult

```python
CriticalCouplingResult(
    target_frequency: float,
    angle: float,
    resonator: HelmholtzResonator,
    slit_height: float,
    absorption: float,
    normalized_impedance: complex,
    converged: bool,
)
```

Outcome of a critical-coupling (perfect-absorption) design.

`resonator` and `slit_height` are the solved geometry that places the
reflection zero on the real-frequency axis at `target_frequency` and
`angle`; `absorption` is the modelled coefficient there (`~1`) and
`normalized_impedance` the achieved `Z cos(theta) / Z0` (`~1`).
`converged` flags whether the root find met its tolerance.

## helmholtz_resonator_impedance

```python
helmholtz_resonator_impedance(
    frequency: ArrayLike,
    resonator: HelmholtzResonator,
    *,
    slit_height: float | None = None,
    lattice_step: float | None = None,
    end_correction: bool = True,
    geometry: str = 'square',
    fluid: Fluid = ...,
    sum_terms: int = 40,
) -> Complex
```

Acoustic impedance of a Helmholtz resonator with visco-thermal losses.

With the default `geometry="square"` the neck and cavity are square
ducts using the effective parameters of
[`rectangular_duct_properties`](/phonometry/reference/api/materials/slow-sound/#rectangular_duct_properties); the impedance is Appl. Phys. Lett. 2016
Eq. (A23) with the neck-to-cavity radiation correction of Eq. (A24) and,
when `slit_height` and `lattice_step` are supplied, the neck-to-slit
correction of Eqs. (A25)-(A26) added to the total neck length correction:

$$
Z_{\mathrm{HR}} = -j \frac{\cos(k_\mathrm{n} l_\mathrm{n}) \cos(k_\mathrm{c} l_\mathrm{c}) - Z_\mathrm{n} k_\mathrm{n} \mathrm{dl} \cos(k_\mathrm{n} l_\mathrm{n}) \sin(k_\mathrm{c} l_\mathrm{c}) / Z_\mathrm{c} - Z_\mathrm{n} \sin(k_\mathrm{n} l_\mathrm{n}) \sin(k_\mathrm{c} l_\mathrm{c}) / Z_\mathrm{c}} {\sin(k_\mathrm{n} l_\mathrm{n}) \cos(k_\mathrm{c} l_\mathrm{c}) / Z_\mathrm{n} - k_\mathrm{n} \mathrm{dl} \sin(k_\mathrm{n} l_\mathrm{n}) \sin(k_\mathrm{c} l_\mathrm{c}) / Z_\mathrm{c} + \cos(k_\mathrm{n} l_\mathrm{n}) \sin(k_\mathrm{c} l_\mathrm{c}) / Z_\mathrm{c}}
$$

with $Z_\mathrm{n} = \sqrt{\kappa_\mathrm{n} \rho_\mathrm{n}} / w_\mathrm{n}^2$,
$k_\mathrm{n} = \omega \sqrt{\rho_\mathrm{n} / \kappa_\mathrm{n}}$ (and likewise for the
cavity), reducing to Eq. (A22) when `dl = 0`.

With `geometry="slit"` the resonator is two-dimensional (the neck and
cavity are slit-like ducts spanning the lattice step): the effective
parameters come from [`slit_effective_properties`](/phonometry/reference/api/materials/slow-sound/#slit_effective_properties) with the neck and
cavity widths, the duct sections are `w_n a` and `w_c a`, and the end
corrections are the 2-D fits of Sci. Rep. 7:5389 Eqs. (11)-(12); both
`slit_height` and `lattice_step` are then required.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `resonator` | The [`HelmholtzResonator`](/phonometry/reference/api/materials/slow-sound/#helmholtzresonator) geometry. |
| `slit_height` | Slit height `h` for the neck-to-slit correction; if `None` that correction is omitted (`"square"` only). |
| `lattice_step` | Lattice step `a` for the neck-to-slit correction. |
| `end_correction` | Include the radiation end corrections (default True). |
| `geometry` | `"square"` (default) for square-duct necks and cavities, `"slit"` for the two-dimensional resonator model. |
| `fluid` | State of the air in the neck and cavity ([`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid)); the density `rho0`, viscosity `eta`, Prandtl number `Pr`, ratio of specific heats `gamma` and static pressure `P0` are read from it. |
| `sum_terms` | Transverse modes kept per axis in the duct series. |

**Returns:** Complex acoustic impedance `Z_HR`, in Pa s/m3, shaped like `frequency`.

## HelmholtzResonator

```python
HelmholtzResonator(
    neck_length: float,
    neck_side: float,
    cavity_length: float,
    cavity_side: float,
)
```

A square-cross-section Helmholtz resonator loading a slit.

`neck_length` `l_n` and `neck_side` `w_n` describe the neck,
`cavity_length` `l_c` and `cavity_side` `w_c` the closed cavity;
all lengths are in metres.

### HelmholtzResonator.plot()

```python
HelmholtzResonator.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the resonator cross-section to scale (dimensioned).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## plot_helmholtz_resonator_geometry

```python
plot_helmholtz_resonator_geometry(
    resonator: HelmholtzResonator,
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw a square-section Helmholtz resonator cross-section, to scale.

Neck opening upward into free air, cavity below, with the four defining
dimensions (neck side and length, cavity side and length) dimensioned.

**Parameters**

| Name | Description |
| :--- | :--- |
| `resonator` | A [`HelmholtzResonator`](/phonometry/reference/api/materials/slow-sound/#helmholtzresonator). |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the cavity rectangle. |

**Returns:** The axes.

## plot_slit_absorber_geometry

```python
plot_slit_absorber_geometry(
    resonators: Sequence[HelmholtzResonator] | HelmholtzResonator,
    ax: Axes | None = None,
    *,
    slit_height: float,
    lattice_step: float,
    period: float,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw one period of the slit metamaterial absorber, to scale.

Side cut of the panel: the slit (height `h`) runs from the mouth at the
left into the panel; `N` Helmholtz resonators load it from below at the
lattice step `a` (total depth `L = N a`); the panel repeats vertically
with `period` `d`; rigid back wall at the right.

**Parameters**

| Name | Description |
| :--- | :--- |
| `resonators` | The resonator chain of [`slit_helmholtz_absorber`](/phonometry/reference/api/materials/slow-sound/#slit_helmholtz_absorber) (one per lattice step, or a single resonator reused for all steps). |
| `ax` | Existing axes, or `None` to create a figure. |
| `slit_height` | Slit height `h`, in metres. |
| `lattice_step` | Lattice step `a`, in metres. |
| `period` | Panel period `d`, in metres. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the slit rectangle. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | naming the first of the three lengths that is not finite and positive, or for an empty resonator chain. |

## rectangular_duct_properties

```python
rectangular_duct_properties(
    frequency: ArrayLike,
    *,
    side: float,
    fluid: Fluid = ...,
    sum_terms: int = 40,
) -> tuple[Complex, Complex]
```

Effective density and bulk modulus of a square duct of the given side.

The Stinson (1991) rectangular-duct series (Appl. Sci. 2017 Eqs. (7)-(8)),

$$
\rho = \frac{-\rho_0 a^2 b^2}{64 G_\rho^2 S_\rho}
$$

$$
\kappa = \frac{\kappa_0} {\gamma + 64 (\gamma - 1) G_\kappa^2 / (a^2 b^2) S_\kappa}
$$

with
$S = \sum_k \sum_m [\alpha_k^2 \beta_m^2 (\alpha_k^2 + \beta_m^2 - G^2)]^{-1}$,
$\alpha_k = (2k+1) \pi / a$, $\beta_m = (2m+1) \pi / b$,
$G_\rho^2 = j \omega \rho_0 / \eta$ and
$G_\kappa^2 = j \omega \mathrm{Pr} \rho_0 / \eta$. Here the duct
is square ($a = b$, both equal to `side`). The series is
transcribed in the source's time convention and returned conjugated so
the result is passive in the $e^{+j\omega t}$ convention
($\operatorname{Im}(k) < 0$). The normalising constant 64 is fixed
by the exact limits $\rho \to \rho_0$, $\kappa \to \kappa_0$
as the boundary layers vanish and by the Poiseuille resistivity
$28.454\,\eta/\text{side}^2$ as $\omega \to 0$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `side` | Square-duct side length, in metres. |
| `fluid` | State of the air in the duct ([`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid)); the density `rho0`, viscosity `eta`, Prandtl number `Pr`, ratio of specific heats `gamma` and static pressure `P0` are read from it. |
| `sum_terms` | Transverse modes kept per axis (default 40). |

**Returns:** `(rho, kappa)` complex arrays shaped like `frequency`.

## slit_effective_properties

```python
slit_effective_properties(
    frequency: ArrayLike,
    *,
    slit_height: float,
    fluid: Fluid = ...,
) -> tuple[Complex, Complex]
```

Effective density and bulk modulus of a narrow slit of height `h`.

$$
\rho_\mathrm{s} = \rho_0 \left[1 - \frac{\tanh(x_\rho)}{x_\rho}\right]^{-1}
$$

$$
\kappa_\mathrm{s} = \kappa_0 \left[1 + (\gamma - 1) \frac{\tanh(x_\kappa)}{x_\kappa}\right]^{-1}
$$

with $x_\rho = (h/2) \sqrt{j \omega \rho_0 / \eta}$ and
$x_\kappa = (h/2) \sqrt{j \omega \mathrm{Pr} \rho_0 / \eta}$
(Appl. Sci. 2017 Eq. (6); Appl. Phys. Lett. 2016 Eqs. (A1)-(A2)).
$\kappa_0 = \gamma P_0$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `slit_height` | Slit height `h`, in metres. |
| `fluid` | State of the air in the slit ([`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid)); the density `rho0`, viscosity `eta`, Prandtl number `Pr`, ratio of specific heats `gamma` and static pressure `P0` are read from it. |

**Returns:** `(rho_s, kappa_s)` complex arrays shaped like `frequency`.

## slit_helmholtz_absorber

```python
slit_helmholtz_absorber(
    frequency: ArrayLike,
    resonators: HelmholtzResonator | list[HelmholtzResonator] | tuple[HelmholtzResonator, ...],
    *,
    slit_height: float,
    lattice_step: float,
    period: float,
    angle: float = 0.0,
    end_correction: bool = True,
    slit_radiation: bool = True,
    resonator_geometry: str = 'square',
    fluid: Fluid = ...,
) -> SlitResonatorAbsorberResult
```

Transfer-matrix prediction of a slit panel loaded with resonators.

The panel is a periodic array (period `d` along the panel face) of thin
closed slits of height `h`, each loaded from its upper wall by the given
`resonators` spaced by the lattice step `a` (Appl. Sci. 2017,
Section 2). The total chain matrix is
$T = M_{\mathrm{dl}} (M_\mathrm{s} M_{\mathrm{HR}} M_\mathrm{s}) \cdots$ over the
`N` resonators, where each resonator sits between two half-lattice
slit steps; the rigidly-backed reflection factor is
$R = (T_{11} \cos(\theta) - Z_0 T_{21}) / (T_{11} \cos(\theta) + Z_0 T_{21})$ with
$Z_0 = \rho_0 c_0 / S_0$, $S_0 = d a$, and
$\alpha = 1 - \lvert R \rvert^2$ (Eq. (4)). The structure is
locally reacting, so the internal chain does not depend on `theta`;
only the front air impedance carries `cos(theta)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `resonators` | One [`HelmholtzResonator`](/phonometry/reference/api/materials/slow-sound/#helmholtzresonator) or a sequence of them, ordered from the panel face towards the rigid backing. |
| `slit_height` | Slit height `h`, in metres. |
| `lattice_step` | Resonator lattice step `a` along the slit, in metres; the slit depth is $L = N a$. |
| `period` | Slit array period `d` along the face, in metres ($d \ge h$). |
| `angle` | Polar angle of incidence `theta`, in radians ($0 \le \theta < \pi/2 - 10^{-6}$). |
| `end_correction` | Include the resonator radiation end corrections. |
| `slit_radiation` | Include the slit-to-free-air radiation correction. |
| `fluid` | State of the air the panel radiates into and the slit and resonators are filled with ([`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid)): its speed of sound `c0`, density `rho0`, viscosity `eta`, Prandtl number `Pr`, ratio of specific heats `gamma` and static pressure `P0`. |

**Returns:** A [`SlitResonatorAbsorberResult`](/phonometry/reference/api/materials/slow-sound/#slitresonatorabsorberresult).

## SlitResonatorAbsorberResult

```python
SlitResonatorAbsorberResult(
    frequency: Real,
    angle: float,
    surface_impedance: Complex,
    normalized_impedance: Complex,
    reflection: Complex,
    absorption: Real,
    effective_wavenumber: Complex,
    effective_impedance: Complex,
    transfer_matrix: Complex,
    resonators: tuple[HelmholtzResonator, ...] | None = None,
    slit_height: float | None = None,
    lattice_step: float | None = None,
    period: float | None = None,
)
```

Prediction of a slit panel loaded with Helmholtz resonators.

All spectra share the shape of `frequency`. `surface_impedance` is
the acoustic surface impedance $Z = T_{11} / T_{21}$ in Pa s/m3 of
the rigidly backed panel, `normalized_impedance` its ratio to
$Z_0 = \rho_0 c_0 / S_0$, `reflection` the plane-wave reflection
factor `R(theta)`, `absorption` the coefficient
$\alpha = 1 - \lvert R \rvert^2$, `effective_wavenumber` and
`effective_impedance` the retrieved `k_eff` and `Z_eff`
(Appl. Sci. 2017 Eq. (5)), and `transfer_matrix` the total 2x2 chain
matrix with shape `(2, 2, len(frequency))`.

The trailing fields retain the panel geometry the prediction was run
with (`resonators`, `slit_height`, `lattice_step`, `period`) so
`plot_geometry` can draw the cross-section; they are appended after
the original fields and default to `None` for hand-built results.

### SlitResonatorAbsorberResult.plot()

```python
SlitResonatorAbsorberResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the absorption spectrum `alpha(f)` with `|R|` overlaid.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### SlitResonatorAbsorberResult.plot_geometry()

```python
SlitResonatorAbsorberResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw one period of the panel cross-section to scale (dimensioned).

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

## SlowSoundAbsorberWarning

Advisory for slow-sound absorber use outside the modelled regime.
