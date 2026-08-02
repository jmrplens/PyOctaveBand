---
title: "materials.absorbers.biot"
description: "Biot poroelastic layers: the three waves and the 6x6 transfer matrix."
sidebar:
  label: "biot"
---

Biot poroelastic layers: the three waves and the 6x6 transfer matrix.

An equivalent fluid replaces a porous material by a single wave travelling in
the pores while the skeleton stands still (rigid frame) or is merely dragged
along by the pore fluid (limp frame, see
[`limp_frame`](/phonometry/reference/api/materials/porous/#limp_frame)). Neither can carry a
wave in the skeleton itself, so neither can produce a frame resonance. The Biot
theory can: it treats the frame as an elastic solid coupled to the pore fluid
through a potential coupling coefficient and an inertial coupling coefficient,
and it predicts **three** waves in an isotropic porous layer, two compressional
and one shear (Allard & Atalla, *Propagation of Sound in Porous Media* 2e,
chapter 6).

This module implements that theory in the $e^{+j \omega t}$ convention
of the rest of the package, exactly as printed:

* **Elastic coefficients** `P`, `Q` and `R` for the usual case of a
  frame built from a material far stiffer than the frame itself
  ($K_s \to \infty$), Eqs. (6.26)-(6.29) printed pp. 116-117:
  $R = \phi K_f$, $Q = (1 - \phi) K_f$,
  $P = 4N/3 + K_b + (1 - \phi)^2 K_f / \phi$ with the frame bulk
  modulus $K_b = 2N(1 + \nu)/(3(1 - 2\nu))$.
* **Modified densities** `rho11`, `rho12`, `rho22` (Eq. (6.56), printed
  p. 120) built from the inertial coupling
  $\rho_a = \phi \rho_0 (\alpha_\infty - 1)$ and the visco-inertial
  term $j \sigma \phi^2 G(\omega) / \omega$. The latter is *not*
  re-derived here: it is read back from the equivalent-fluid model handed in,
  through the identity $\rho_{22} = \phi^2 \rho_{\mathrm{eq}}$ stated
  on printed p. 253, so a
  Biot layer and a rigid-frame layer built from the same
  [`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult) share one
  visco-thermal description by construction.
* **The two compressional waves** as the eigenvalues of Eq. (6.65)
  (Eqs. (6.67)-(6.69), printed p. 121), the **shear wave** of Eq. (6.83), and
  the fluid-to-frame velocity ratios `mu1`, `mu2` (Eq. (6.71)) and `mu3`
  (Eq. (6.84)).
* **The closed-form surface impedance** of a hard-backed layer at normal
  incidence, Eqs. (6.107)-(6.108) printed p. 128, in
  [`biot_surface_impedance`](/phonometry/reference/api/materials/biot/#biot_surface_impedance), and the $\lambda/4$ frame resonance it
  develops, Eq. (6.110) printed p. 129, in
  [`frame_quarter_wave_resonance`](/phonometry/reference/api/materials/biot/#frame_quarter_wave_resonance).
* **The 6x6 layer matrix** of chapter 11: the field vector
  $[v_1^s, v_3^s, v_3^f, s_{33}^s, s_{13}^s, s_{33}^f]$ (Eq. (11.26))
  expressed through the wave-amplitude matrix $[\Gamma(x_3)]$ of
  Table 11.1 (printed p. 252), from which
  [`poroelastic_transfer_matrix`](/phonometry/reference/api/materials/biot/#poroelastic_transfer_matrix) returns
  $[T] = [\Gamma(-h)][\Gamma(0)]^{-1}$ (Eq. (11.34)).

The layer is used through
[`PoroelasticLayer`](/phonometry/reference/api/materials/porous/#poroelasticlayer) inside
[`layered_absorber`](/phonometry/reference/api/materials/porous/#layered_absorber), which switches
to the global-matrix assembly of Sect. 11.5 as soon as a poroelastic layer is
present, with the coupling matrices of Sect. 11.4 ($[I_{pf}]$ /
$[J_{pf}]$ Eq. (11.73), $[I_{pp}]$ Eq. (11.67)) and the hard-wall
conditions $[Y]$ of Eq. (11.81).

**On oracles.** Allard & Atalla publish no table of computed surface
impedances, so the model is anchored on closed forms and on exact limits: the
rigid-frame limit reproduces the already-anchored Johnson-Champoux-Allard
equivalent fluid, the limp limit reproduces
[`limp_frame`](/phonometry/reference/api/materials/porous/#limp_frame), and the chapter 11
assembly reproduces the chapter 6 closed form Eq. (6.107) to machine precision.
The book does print four *output* numbers for the fully specified glass wool of
Table 6.1, and all four are reproduced: the airborne branch changes from
$(\delta_1, \mu_1)$ to $(\delta_2, \mu_2)$ at 495 Hz,
$\lvert \mu_a \rvert > 40$ above 50 Hz, and `mu_b` runs from 1.0 at
50 Hz to 0.82 at 1500 Hz (all printed pp. 124-125), while the impedance peak
of a 5.6 cm layer sits at 860 Hz (printed p. 129). The third of those is
reproduced by $\operatorname{Re}(\mu_b)$: the printed sentence calls it
a modulus, but $\lvert \mu_b \rvert$ is 0.939 at 1500 Hz against the
printed 0.82, and `docs/ERRATA.md` records why. See
`tests/materials/absorbers/test_biot.py`.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## biot_surface_impedance

```python
biot_surface_impedance(waves: BiotWavesResult, thickness: float) -> Complex
```

Surface impedance of a hard-backed Biot layer at normal incidence.

The closed form of Allard & Atalla 2e, Eqs. (6.107)-(6.108) (printed
p. 128), obtained by writing the four compressional-wave amplitudes against
the three boundary conditions of a layer *glued* to an impervious rigid
wall (zero frame and fluid velocity at the wall, Eq. (6.95); continuity of
pressure and of the total normal stress at the free face, Eqs. (6.97)-(6.99);
conservation of the volume flow, Eq. (6.100)):

$$
Z = -j (Z_1^s Z_2^f \mu_2 - Z_2^s Z_1^f \mu_1) / D
$$

$$
D = (1 - \phi + \phi \mu_2) [Z_1^s - (1 - \phi) Z_1^f \mu_1] \tan(\delta_2 l) + (1 - \phi + \phi \mu_1) [(1 - \phi) Z_2^f \mu_2 - Z_2^s] \tan(\delta_1 l)
$$

with the four characteristic impedances of Eqs. (6.74)-(6.77),
$Z_i^f = (R + Q/\mu_i) \delta_i / (\phi \omega)$ and
$Z_i^s = (P + Q \mu_i) \delta_i / \omega$.

This is an independent derivation of the same physics as the chapter 11
transfer-matrix assembly reached through
[`PoroelasticLayer`](/phonometry/reference/api/materials/porous/#poroelasticlayer); the two
agree to machine precision, which is one of the anchors of this module.
Unlike the transfer matrix, it is restricted to normal incidence, to a
single layer and to a glued rigid backing.

**Parameters**

| Name | Description |
| :--- | :--- |
| `waves` | The [`BiotWavesResult`](/phonometry/reference/api/materials/biot/#biotwavesresult) of the material. |
| `thickness` | Layer thickness `l`, in metres (> 0). |

**Returns:** The complex surface impedance `Z`, in Pa s/m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive thickness. |

## biot_waves

```python
biot_waves(
    medium: PorousMediumResult,
    *,
    porosity: float,
    tortuosity: float,
    frame_density: float,
    shear_modulus: complex,
    poisson_ratio: float = 0.0,
) -> BiotWavesResult
```

The two compressional waves and the shear wave of a Biot layer.

*medium* supplies the visco-thermal description of the pore fluid, through
the two identities of Allard & Atalla printed p. 253,
$\rho_{22} = \phi^2 \rho_{\mathrm{eq}}$ and
$R = \phi^2 K_{\mathrm{eq}}$, where `rho_eq` and
`K_eq` are the surface-normalised effective density and bulk modulus that
every model in [`porous`](/phonometry/reference/api/materials/porous/) returns. Pass
the rigid-frame [`johnson_champoux_allard`](/phonometry/reference/api/materials/porous/#johnson_champoux_allard)
result: the frame motion is the Biot model's business, not the equivalent
fluid's, so the *rigid-frame* effective density is the correct input (a
limp-corrected one would count the frame inertia twice).

From it, with the inertial coupling
$\rho_a = \phi \rho_0 (\alpha_\infty - 1)$ (Eq. (6.44)):

* $\rho_{22} = \phi\rho_0 + \rho_a - j\sigma\phi^2 G(\omega)/\omega$,
  $\rho_{12} = -\rho_a + j\sigma\phi^2 G(\omega)/\omega$ and
  $\rho_{11} = \rho_1 + \rho_a - j\sigma\phi^2 G(\omega)/\omega$
  (Eq. (6.56), printed p. 120), the visco-inertial term being recovered as
  $\phi \rho_0 \alpha_\infty - \rho_{22}$;
* $K_f = \phi K_{\mathrm{eq}}$, then $R = \phi K_f$,
  $Q = (1 - \phi) K_f$ and
  $P = 4N/3 + K_b + (1 - \phi)^2 K_f / \phi$ (Eqs. (6.26)-(6.28));
* $\delta_1^2$ and $\delta_2^2$ from Eqs. (6.67)-(6.69),
  $\delta_3^2$ from
  Eq. (6.83), `mu1`, `mu2` from Eq. (6.71) and `mu3` from Eq. (6.84).

**Parameters**

| Name | Description |
| :--- | :--- |
| `medium` | Rigid-frame [`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult) for the pore fluid, evaluated on the frequency vector of interest. |
| `porosity` | Open porosity `phi` (0 \< phi \<= 1). |
| `tortuosity` | High-frequency tortuosity `a_inf` (>= 1). |
| `frame_density` | Bulk density of the frame `rho1`, in kg/m3 (> 0). |
| `shear_modulus` | Complex shear modulus `N` of the frame, in Pa ($\operatorname{Im}(N) \ge 0$; a structural loss factor `eta` gives $N = N'(1 + j \eta)$). |
| `poisson_ratio` | Poisson coefficient `nu` of the frame (Default: 0, the value Allard & Atalla use for their glass wool). |

**Returns:** A [`BiotWavesResult`](/phonometry/reference/api/materials/biot/#biotwavesresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an out-of-range porosity, tortuosity, frame density, shear modulus or Poisson coefficient. |

## BiotWavesResult

```python
BiotWavesResult(
    frequency: Real,
    porosity: float,
    tortuosity: float,
    frame_density: float,
    shear_modulus: complex,
    poisson_ratio: float,
    elastic_p: Complex,
    elastic_q: Complex,
    elastic_r: Complex,
    density_11: Complex,
    density_12: Complex,
    density_22: Complex,
    compressional_wavenumber_1: Complex,
    compressional_wavenumber_2: Complex,
    shear_wavenumber: Complex,
    velocity_ratio_1: Complex,
    velocity_ratio_2: Complex,
    velocity_ratio_3: Complex,
)
```

The three Biot waves of an isotropic air-saturated porous material.

All arrays share the shape of `frequency`. `compressional_wavenumber_1`
and `compressional_wavenumber_2` are `delta1` and `delta2` of
Eqs. (6.67)-(6.68) (the branch with $-\sqrt{\Delta}$ first, as
printed, with $\sqrt{\Delta}$ taken on the root with non-positive
real part so that
the numbering matches the book's own example), `shear_wavenumber` is
`delta3` of Eq. (6.83), all in rad/m and all taken on the root with
non-negative real part. `velocity_ratio_1`,
`velocity_ratio_2` and `velocity_ratio_3` are the ratios `mu` of the
fluid displacement over the frame displacement (Eqs. (6.71) and (6.84)).
`elastic_p`, `elastic_q` and `elastic_r` are the Biot elastic
coefficients and `density_11`, `density_12`, `density_22` the
modified densities of Eq. (6.56).

The **airborne** wave is the one whose $\lvert \mu \rvert$ is the
larger (the pore
fluid moves far more than the frame); the **frame-borne** wave is the
other. Which of `delta1` / `delta2` plays which role swaps with
frequency, so use the `airborne_*` and `frame_borne_*` properties
rather than the numbered ones.

### BiotWavesResult.airborne_is_second

*property*

Whether the airborne wave is $(\delta_2, \mu_2)$ at each
frequency.

**Neither labelling is continuous in general.** `delta1` and
`delta2` are the two branches of one square root (Eqs. (6.67) and
(6.68)), so `compressional_wavenumber_1` and
`compressional_wavenumber_2` swap wherever the discriminant crosses
the cut of `numpy.sqrt`; on the Table 6.1 glass wool that
happens at 495.99 Hz, exactly where the book puts the change of root,
and the two wavenumbers jump past each other by 24 rad/m there. This
$\lvert \mu \rvert$ sorting is the physical labelling of
Sect. 6.5.4 and it
removes that jump where the two events coincide, as they do there,
but it introduces one of its own wherever $\lvert \mu_1 \rvert$
and $\lvert \mu_2 \rvert$
cross *away* from the cut: on a sweep of 864 parameter sets, 30 left
a visible step in the sorted airborne wavenumber.

Nothing downstream depends on which root is called which. The closed
form Eq. (6.107) and the $[\Gamma]$ of Table 11.1 are both
invariant
under the permutation of the two compressional waves and under a sign
flip of either, so the surface impedance is continuous across the
crossing even where the labels are not.

### BiotWavesResult.airborne_velocity_ratio

*property*

Fluid-over-frame velocity ratio `mu_a` of the airborne wave.

### BiotWavesResult.airborne_wavenumber

*property*

Wavenumber `delta_a` of the airborne compressional wave, rad/m.

### BiotWavesResult.frame_borne_velocity_ratio

*property*

Fluid-over-frame velocity ratio `mu_b` of the frame-borne wave.

### BiotWavesResult.frame_borne_wavenumber

*property*

Wavenumber `delta_b` of the frame-borne wave, in rad/m.

### BiotWavesResult.plot()

```python
BiotWavesResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the three Biot wavenumbers against frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## frame_bulk_modulus

```python
frame_bulk_modulus(shear_modulus: complex, poisson_ratio: float) -> complex
```

Bulk modulus `Kb` of the frame in vacuum from `N` and `nu`.

$K_b = 2 N (\nu + 1) / (3 (1 - 2 \nu))$ (Allard & Atalla 2e,
Eq. (6.29), printed p. 116). `Kb` is the quantity the jacketed
"gedanken experiment" of Eq. (6.7) measures, and the one the limp-frame
rules of thumb of
[`limp_frame_applicable`](/phonometry/reference/api/materials/porous/#limp_frame_applicable) compare
with the bulk modulus of the pore fluid.

**Parameters**

| Name | Description |
| :--- | :--- |
| `shear_modulus` | Complex shear modulus `N` of the frame, in Pa ($\operatorname{Im}(N) \ge 0$ for a lossy frame in the $e^{+j \omega t}$ convention). |
| `poisson_ratio` | Poisson coefficient `nu` of the frame ($-1 < \nu < 0.5$). |

**Returns:** The complex bulk modulus `Kb` of the frame in vacuum, in Pa.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive or non-finite `N`, a negative $\operatorname{Im}(N)$ or a `nu` outside $(-1, 0.5)$. |

## frame_elastic_coefficient

```python
frame_elastic_coefficient(
    shear_modulus: complex,
    poisson_ratio: float,
) -> complex
```

Longitudinal elastic coefficient `Kc` of the frame in vacuum.

$$
K_c = \lambda + 2 \mu = K_b + 4 N / 3 = 2 (1 - \nu) N / (1 - 2 \nu)
$$

(Allard & Atalla 2e, Eqs. (1.76) and (6.111), printed pp. 12 and 130).
A compressional wave in the frame *in vacuum* travels at
$\sqrt{K_c / \rho_1}$, which is what sets the $\lambda/4$
frame resonance of [`frame_quarter_wave_resonance`](/phonometry/reference/api/materials/biot/#frame_quarter_wave_resonance).

**Parameters**

| Name | Description |
| :--- | :--- |
| `shear_modulus` | Complex shear modulus `N` of the frame, in Pa. |
| `poisson_ratio` | Poisson coefficient `nu` of the frame. |

**Returns:** The complex elastic coefficient `Kc`, in Pa.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | as [`frame_bulk_modulus`](/phonometry/reference/api/materials/biot/#frame_bulk_modulus). |

## frame_quarter_wave_resonance

```python
frame_quarter_wave_resonance(
    thickness: float,
    *,
    shear_modulus: complex,
    poisson_ratio: float,
    frame_density: float,
) -> float
```

Quarter-wavelength resonance of the frame-borne wave (closed form).

A porous layer glued to a rigid wall holds the frame still at the wall and
free at the front face, so the frame-borne compressional wave resonates
where $l \operatorname{Re}(\delta_b) = \pi/2$ (Allard & Atalla 2e,
Eq. (6.109), printed p. 129). Since `delta_b` stays close to the
frame-in-vacuum wavenumber $\omega \sqrt{\rho_1 / K_c}$ of
Eq. (6.88), the resonance sits at

$$
f_r = \frac{1}{4 l} \sqrt{\operatorname{Re}(K_c) / \rho_1} \tag{Eq. 6.110}
$$

This is the frequency at which the peak that no
equivalent-fluid model can produce appears in the surface impedance of
[`biot_surface_impedance`](/phonometry/reference/api/materials/biot/#biot_surface_impedance); Eq. (6.110) is an approximation to it
(`delta_b` is not exactly the in-vacuum wavenumber), so the peak lands a
few per cent above `fr`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `thickness` | Layer thickness `l`, in metres (> 0). |
| `shear_modulus` | Complex shear modulus `N` of the frame, in Pa. |
| `poisson_ratio` | Poisson coefficient `nu` of the frame. |
| `frame_density` | Bulk density of the frame `rho1`, in kg/m3 (> 0): the mass of solid per unit volume of material. |

**Returns:** The resonance frequency `fr`, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive thickness or frame density, or an invalid `N` / `nu`. |

## poroelastic_transfer_matrix

```python
poroelastic_transfer_matrix(
    waves: BiotWavesResult,
    thickness: float,
    *,
    transverse_wavenumber: ArrayLike = 0.0,
) -> Complex
```

The 6x6 transfer matrix $[T^p]$ of a Biot poroelastic layer.

$[T^p] = [\Gamma(-h)][\Gamma(0)]^{-1}$ (Allard & Atalla 2e,
Eq. (11.34), printed p. 249), relating the field vector
$[v_1^s, v_3^s, v_3^f, s_{33}^s, s_{13}^s, s_{33}^f]$ (Eq. (11.26))
just inside the front
face of the layer to the same vector just inside its back face,
$V(M) = [T^p] V(M')$. The wave-amplitude matrix $[\Gamma]$ of
Table 11.1 behind it is built by `_gamma`.

The layer solver of
[`layered_absorber`](/phonometry/reference/api/materials/porous/#layered_absorber) does *not*
use this matrix: it assembles the wave amplitudes directly, which avoids
inverting $[\Gamma(0)]$ and is far better conditioned for a very
soft or a
very thick frame. The matrix is exposed because it is the object chapter 11
is written around and the one an external multilayer chain expects.

**Parameters**

| Name | Description |
| :--- | :--- |
| `waves` | The [`BiotWavesResult`](/phonometry/reference/api/materials/biot/#biotwavesresult) of the material. |
| `thickness` | Layer thickness `h`, in metres (> 0). |
| `transverse_wavenumber` | In-plane wavenumber $k_t = k \sin(\theta)$, in rad/m (Default: 0, normal incidence). Scalar or one value per frequency. |

**Returns:** The transfer matrix with shape `(len(frequency), 6, 6)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive thickness. |
