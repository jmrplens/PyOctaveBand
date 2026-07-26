---
title: "materials.slow_sound_absorber"
description: "Slow-sound slit panels loaded with Helmholtz resonators (perfect absorbers)."
sidebar:
  label: "slow_sound_absorber"
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
phonometry (a passive medium has `Im(k) < 0`):

* **Visco-thermal effective parameters.** The slit of height `h` uses the
  narrow-channel effective density and bulk modulus (Appl. Sci. Eq. (6);
  Appl. Phys. Lett. Eqs. (A1)-(A2)):

  `rho_s = rho0 [1 - tanh((h/2) G_rho) / ((h/2) G_rho)]^-1` and
  `kappa_s = kappa0 [1 + (gamma - 1) tanh((h/2) G_kappa) / ((h/2) G_kappa)]^-1`

  with `G_rho = sqrt(j w rho0 / eta)` and `G_kappa = sqrt(j w Pr rho0 /
  eta)`. The square necks and cavities use the rectangular-duct series of
  Stinson (1991), reproduced as Appl. Sci. Eqs. (7)-(8) with the transverse
  wavenumbers `alpha_k = (2k+1) pi / a` and `beta_m = (2m+1) pi / b`. The
  duct series is printed in the opposite time convention of the source; it is
  returned conjugated here so the neck and cavity share the `e^{+j w t}`
  passivity of the slit. Both models are pinned in the tests to their exact
  limits: the effective density tends to `rho0` and the bulk modulus to
  `kappa0` as the boundary layers vanish, and `j w rho` tends to the
  Poiseuille flow resistivity of the channel as `w -> 0`
  (`12 eta / h^2` for the slit, `28.454 eta / w^2` for a square duct).

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
  `R = (T11 cos(theta) - Z0 T21) / (T11 cos(theta) + Z0 T21)` with
  `Z0 = rho0 c0 / S0` (Eq. (4)), and `alpha = 1 - |R|^2`. Perfect
  absorption (critical coupling) is reached when the reflection zero sits on
  the real-frequency axis, i.e. `Re(Z) cos(theta) = Z0` and `Im(Z) = 0`
  with `Z = T11 / T21` the acoustic surface impedance (Eq. (9)).

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
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
    prandtl_number: float = 0.71,
    heat_capacity_ratio: float = 1.4,
    atmospheric_pressure: float = 101325.0,
) -> CriticalCouplingResult
```

Solve resonator/slit geometry for perfect absorption at a frequency.

Critical coupling (perfect absorption) requires the acoustic surface
impedance `Z = T11 / T21` of the rigidly-backed panel to satisfy
`Re(Z) cos(theta) = Z0` and `Im(Z) = 0` at `target_frequency`
(Appl. Sci. 2017 Eq. (9)), i.e. the reflection zero lies on the
real-frequency axis. Holding the neck geometry and cavity side of
`resonator` fixed, this tunes the cavity length (which sets the resonance
frequency) and the slit height (which sets the visco-thermal leakage
balance) to meet both conditions, so `alpha ~ 1` at the design point.

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
| `speed_of_sound` | Speed of sound `c0` in air, in m/s. |
| `air_density` | Air density `rho0`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |

**Returns:** A [`CriticalCouplingResult`](/phonometry/reference/api/materials/slow-sound-absorber/#criticalcouplingresult). A [`SlowSoundAbsorberWarning`](/phonometry/reference/api/materials/slow-sound-absorber/#slowsoundabsorberwarning) is emitted (via `warnings.warn`) if the solver does not reach perfect absorption within tolerance.

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
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
    prandtl_number: float = 0.71,
    heat_capacity_ratio: float = 1.4,
    atmospheric_pressure: float = 101325.0,
    sum_terms: int = 40,
) -> Complex
```

Acoustic impedance of a Helmholtz resonator with visco-thermal losses.

The neck and cavity use the square-duct effective parameters of
[`rectangular_duct_properties`](/phonometry/reference/api/materials/slow-sound-absorber/#rectangular_duct_properties); the impedance is Appl. Phys. Lett. 2016
Eq. (A23) with the neck-to-cavity radiation correction of Eq. (A24) and,
when `slit_height` and `lattice_step` are supplied, the neck-to-slit
correction of Eqs. (A25)-(A26) added to the total neck length correction:

`Z_HR = -j [cos(k_n l_n) cos(k_c l_c)
- Z_n k_n dl cos(k_n l_n) sin(k_c l_c) / Z_c
- Z_n sin(k_n l_n) sin(k_c l_c) / Z_c]
/ [sin(k_n l_n) cos(k_c l_c) / Z_n
- k_n dl sin(k_n l_n) sin(k_c l_c) / Z_c
+ cos(k_n l_n) sin(k_c l_c) / Z_c]`

with `Z_n = sqrt(kappa_n rho_n) / w_n^2`, `k_n = w sqrt(rho_n / kappa_n)`
(and likewise for the cavity), reducing to Eq. (A22) when `dl = 0`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `resonator` | The [`HelmholtzResonator`](/phonometry/reference/api/materials/slow-sound-absorber/#helmholtzresonator) geometry. |
| `slit_height` | Slit height `h` for the neck-to-slit correction; if `None` that correction is omitted. |
| `lattice_step` | Lattice step `a` for the neck-to-slit correction. |
| `end_correction` | Include the radiation end corrections (default True). |
| `air_density` | Air density `rho0`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |
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

## rectangular_duct_properties

```python
rectangular_duct_properties(
    frequency: ArrayLike,
    *,
    side: float,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
    prandtl_number: float = 0.71,
    heat_capacity_ratio: float = 1.4,
    atmospheric_pressure: float = 101325.0,
    sum_terms: int = 40,
) -> tuple[Complex, Complex]
```

Effective density and bulk modulus of a square duct of the given side.

The Stinson (1991) rectangular-duct series (Appl. Sci. 2017 Eqs. (7)-(8)),

`rho = -rho0 a^2 b^2 / (64 G_rho^2 S_rho)` and
`kappa = kappa0 / (gamma + 64 (gamma - 1) G_kappa^2 / (a^2 b^2) S_kappa)`,

with `S = sum_k sum_m [alpha_k^2 beta_m^2 (alpha_k^2 + beta_m^2 - G^2)]^-1`,
`alpha_k = (2k+1) pi / a`, `beta_m = (2m+1) pi / b`,
`G_rho^2 = j w rho0 / eta` and `G_kappa^2 = j w Pr rho0 / eta`. Here the
duct is square (`a = b = side`). The series is transcribed in the source's
time convention and returned conjugated so the result is passive in the
`e^{+j w t}` convention (`Im(k) < 0`). The normalising constant 64 is
fixed by the exact limits `rho -> rho0`, `kappa -> kappa0` as the
boundary layers vanish and by the Poiseuille resistivity `28.454 eta /
side^2` as `w -> 0`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `side` | Square-duct side length, in metres. |
| `air_density` | Air density `rho0`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |
| `sum_terms` | Transverse modes kept per axis (default 40). |

**Returns:** `(rho, kappa)` complex arrays shaped like `frequency`.

## slit_effective_properties

```python
slit_effective_properties(
    frequency: ArrayLike,
    *,
    slit_height: float,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
    prandtl_number: float = 0.71,
    heat_capacity_ratio: float = 1.4,
    atmospheric_pressure: float = 101325.0,
) -> tuple[Complex, Complex]
```

Effective density and bulk modulus of a narrow slit of height `h`.

`rho_s = rho0 [1 - tanh(x_rho) / x_rho]^-1` and
`kappa_s = kappa0 [1 + (gamma - 1) tanh(x_kappa) / x_kappa]^-1` with
`x_rho = (h/2) sqrt(j w rho0 / eta)` and
`x_kappa = (h/2) sqrt(j w Pr rho0 / eta)` (Appl. Sci. 2017 Eq. (6);
Appl. Phys. Lett. 2016 Eqs. (A1)-(A2)). `kappa0 = gamma P0`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `slit_height` | Slit height `h`, in metres. |
| `air_density` | Air density `rho0`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |

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
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
    viscosity: float = 1.84e-05,
    prandtl_number: float = 0.71,
    heat_capacity_ratio: float = 1.4,
    atmospheric_pressure: float = 101325.0,
) -> SlitResonatorAbsorberResult
```

Transfer-matrix prediction of a slit panel loaded with resonators.

The panel is a periodic array (period `d` along the panel face) of thin
closed slits of height `h`, each loaded from its upper wall by the given
`resonators` spaced by the lattice step `a` (Appl. Sci. 2017,
Section 2). The total chain matrix is
`T = M_dl (M_s M_HR M_s) ...` over the `N` resonators, where each
resonator sits between two half-lattice slit steps; the rigidly-backed
reflection factor is `R = (T11 cos(theta) - Z0 T21) / (T11 cos(theta) +
Z0 T21)` with `Z0 = rho0 c0 / S0`, `S0 = d a`, and
`alpha = 1 - |R|^2` (Eq. (4)). The structure is locally reacting, so the
internal chain does not depend on `theta`; only the front air impedance
carries `cos(theta)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency vector `f`, in hertz. |
| `resonators` | One [`HelmholtzResonator`](/phonometry/reference/api/materials/slow-sound-absorber/#helmholtzresonator) or a sequence of them, ordered from the panel face towards the rigid backing. |
| `slit_height` | Slit height `h`, in metres. |
| `lattice_step` | Resonator lattice step `a` along the slit, in metres; the slit depth is `L = N a`. |
| `period` | Slit array period `d` along the face, in metres (`d >= h`). |
| `angle` | Polar angle of incidence `theta`, in radians (`0 <= theta < pi/2 - 1e-6`). |
| `end_correction` | Include the resonator radiation end corrections. |
| `slit_radiation` | Include the slit-to-free-air radiation correction. |
| `speed_of_sound` | Speed of sound `c0` in air, in m/s. |
| `air_density` | Air density `rho0`, in kg/m3. |
| `viscosity` | Dynamic viscosity `eta` of air, in Pa s. |
| `prandtl_number` | Prandtl number `Pr` of air. |
| `heat_capacity_ratio` | Ratio of specific heats `gamma`. |
| `atmospheric_pressure` | Static pressure `P0`, in Pa. |

**Returns:** A [`SlitResonatorAbsorberResult`](/phonometry/reference/api/materials/slow-sound-absorber/#slitresonatorabsorberresult).

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
)
```

Prediction of a slit panel loaded with Helmholtz resonators.

All spectra share the shape of `frequency`. `surface_impedance` is the
acoustic surface impedance `Z = T11 / T21` in Pa s/m3 of the rigidly
backed panel, `normalized_impedance` its ratio to `Z0 = rho0 c0 / S0`,
`reflection` the plane-wave reflection factor `R(theta)`,
`absorption` the coefficient `alpha = 1 - |R|^2`,
`effective_wavenumber` and `effective_impedance` the retrieved
`k_eff` and `Z_eff` (Appl. Sci. 2017 Eq. (5)), and `transfer_matrix`
the total 2x2 chain matrix with shape `(2, 2, len(frequency))`.

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

## SlowSoundAbsorberWarning

Advisory for slow-sound absorber use outside the modelled regime.
