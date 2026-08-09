← [Documentation index](../../README.md)

# Atmospheric refraction: ray tracing and the parabolic equation (Salomons / Attenborough)

The [ISO 9613-2 method](outdoor-propagation.md) and the
[spherical-wave ground effect](ground-barriers.md) both assume a **homogeneous**
atmosphere. In reality the sound speed changes with height, because temperature
and wind change with height, and this **refracts** sound: rays curve, and over a
few hundred metres the received level can swing by tens of decibels. This page
covers `phonometry.environment.propagation.refraction`, the
refracting-atmosphere counterpart of the ocean solvers in
[`phonometry.underwater.propagation.numerical`](../../underwater/underwater-solvers.md): a
**ray model** and a **parabolic-equation (PE)** solver, both clean-room from
Salomons, *Computational Atmospheric Acoustics* (2001) and Attenborough & Van
Renterghem, *Predicting Outdoor Sound* 2e (2021, Ch. 11).

Whether refraction matters is mostly a question of range. A representative
surface-layer gradient of ±0.1 s⁻¹ curves rays with a radius
$R_c = c_0/|g| \approx 3.4$ km, so over the first hundred metres the paths are
sensibly straight and the homogeneous models above are accurate. Beyond a few
hundred metres the geometry takes over: downwind, or under a nocturnal
temperature inversion, the downward-curved rays close over the ground and hold
the level near (or even above) the homogeneous prediction, while upwind the
same wind profile opens an acoustic shadow into which the level collapses by
20 dB or more (Attenborough & Van Renterghem 2021, Ch. 11). That is the
familiar upwind/downwind asymmetry of any steady outdoor source: the same
machine at the same distance, tens of decibels apart depending on which side
you stand. It is also why ISO 9613-2 fixes its "favourable" downwind
atmosphere by decree, and what its scalar $C_{met}$ compresses; the models on
this page compute the physics that decides both.

The sketch below puts both sides in one scene: one source, a receiver 350 m
away on each side, and the wind profile deciding which of them hears anything.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_atmospheric_refraction_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_atmospheric_refraction.svg" alt="Atmospheric refraction scene: a source 2 m above the ground with a receiver 350 m away on each side, wind arrows growing longer with height, and an inset of the effective sound-speed profiles c_eff(z) = c(z) + u(z); on the downwind side the rays curve back toward the ground and the receiver gets a direct plus a ground-bounced arrival, on the upwind side every ray bends upward and an acoustic shadow opens on the ground beyond about 220 m, swallowing the receiver standing inside it" width="92%"></picture>

![Atmospheric refraction: ray bending and the acoustic shadow](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/atmospheric_refraction.webp)

<details>
<summary>Show the code for this figure</summary>

```python
import warnings

import matplotlib.pyplot as plt
import numpy as np

from phonometry import (
    atmospheric_parabolic_equation,
    atmospheric_ray_paths,
    log_linear_sound_speed_profile,
    shadow_zone_distance,
)

c0 = 340.0
profile = log_linear_sound_speed_profile(-1.0, ground_speed=c0, max_height=60.0)
zs = 2.0
fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)

rays = atmospheric_ray_paths(profile, source_height=zs,
                             launch_angles_deg=np.linspace(-8.0, 8.0, 17),
                             max_range=600.0, n_steps=3000)
for i in range(rays.heights.shape[0]):
    axes[0].plot(rays.ranges[i], rays.heights[i], color="#1f77b4", lw=0.8, alpha=0.7)
axes[0].plot([0.0], [zs], "o", color="#d62728", label="Source")
grad = (profile.speed_at(10.0) - c0) / 10.0
x_sh = shadow_zone_distance(float(grad), zs, zs, ground_speed=c0)
axes[0].axvline(x_sh, color="#d62728", ls="--", label="Shadow-zone boundary")
axes[0].set(ylabel="Height [m]", ylim=(0, 40), title="Sound rays (upward refraction)")
axes[0].legend()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pe = atmospheric_parabolic_equation(400.0, profile, source_height=zs,
                                        flow_resistivity=200e3,
                                        max_range=600.0, max_height=40.0)
img = axes[1].imshow(pe.relative_level, cmap="RdBu_r", vmin=-30, vmax=6,
                     aspect="auto", origin="lower", interpolation="bilinear",
                     extent=(pe.ranges[0], pe.ranges[-1], pe.heights[0], pe.heights[-1]))
axes[1].axvline(x_sh, color="k", ls="--")
axes[1].set(ylabel="Height [m]", xlabel="Range [m]", ylim=(0, 40),
            title="GFPE relative sound level")
fig.colorbar(img, ax=axes[1], label="Level re free field [dB]")
plt.tight_layout()
plt.show()
```

</details>

## 1. Effective sound-speed profile

A moving (windy) atmosphere is well approximated by a non-moving one with the
**effective sound speed** $c_\text{eff}(z) = c(z) + u(z)$, the adiabatic sound
speed
plus the component of the wind in the propagation direction (Salomons Eq. 4.4).
Two profile shapes cover most surface-layer cases:

- a **linear** profile $c_\text{eff}(z) = c_0 + g\,z$
  (`linear_sound_speed_profile`), the simplest refracting atmosphere and the
  one with exact ray geometry;
- the realistic **logarithmic** surface-layer profile
  $c_\text{eff}(z) = c_0 + b\ln(1 + z/z_0)$ (`log_linear_sound_speed_profile`,
  Salomons Eq. 4.5), with $b \approx +1\ \text{m/s}$ for a typical
  downward-refracting atmosphere, $b \approx -1\ \text{m/s}$ for an
  upward-refracting one, and $z_0$ the roughness length
  (about 0.1 m for grassland).

A positive gradient (sound speed increasing upward) bends rays **down** toward
the receiver (favourable propagation); a negative gradient bends them **up** and
opens an acoustic **shadow** near the ground.

```python
from phonometry import log_linear_sound_speed_profile

profile = log_linear_sound_speed_profile(-1.0, ground_speed=340.0)  # upward
profile.speed_at(10.0)   # effective sound speed 10 m above the ground
profile.plot()           # c_eff(z) with height on the vertical axis
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/atmospheric_sound_speed_profiles_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/atmospheric_sound_speed_profiles.svg" alt="Two logarithmic effective sound-speed profiles over the first 60 metres of atmosphere, height on the vertical axis: the downward-refracting profile (b = +1 m/s) bends right of the 340 m/s ground value and the upward-refracting one (b = -1 m/s) bends left, both with their steepest change concentrated in the first few metres above the roughness length" width="62%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import log_linear_sound_speed_profile

# The two canonical surface layers of Salomons Eq. 4.5 over grassland.
down = log_linear_sound_speed_profile(+1.0, ground_speed=340.0, max_height=60.0)
up = log_linear_sound_speed_profile(-1.0, ground_speed=340.0, max_height=60.0)
ax = down.plot(color="#1f77b4")
up.plot(ax=ax, color="#d62728")
plt.show()
```

</details>

## 2. Ray model

In geometrical acoustics a sound ray obeys Snell's law
$\cos(\gamma(z))/c(z) = \text{const}$ (Salomons Eq. 4.3). `atmospheric_ray_paths` integrates
it with a fourth-order Runge-Kutta scheme, marching in range and reflecting
specularly at the ground, and returns the curved paths, the turning points, the
travel times and the number of ground reflections.

```python
import numpy as np
from phonometry import atmospheric_ray_paths, log_linear_sound_speed_profile

profile = log_linear_sound_speed_profile(-1.0, ground_speed=340.0)
rays = atmospheric_ray_paths(profile, source_height=2.0,
                             launch_angles_deg=np.linspace(-8.0, 8.0, 17),
                             max_range=600.0)
rays.plot()   # curved ray fan with the acoustic shadow near the ground
```

The page-top figure shows this upward-refracting fan opening its shadow. The
favourable case is its mirror image: under **downward** refraction ($b = +1$)
the shallow rays are bent back to the ground, bounce, and carry energy along
the surface instead of losing it upward.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/atmospheric_ray_fan_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/atmospheric_ray_fan.svg" alt="A fan of sound rays traced from a source 2 m above the ground through a downward-refracting logarithmic atmosphere: the shallow launch angles curve back down, reflect off the ground and arch onward in repeated hops out to 600 m, while the steepest rays climb out of the 40 m frame" width="88%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import atmospheric_ray_paths, log_linear_sound_speed_profile

# Downward refraction: the favourable-propagation mirror of the shadow case.
profile = log_linear_sound_speed_profile(+1.0, ground_speed=340.0)
rays = atmospheric_ray_paths(profile, source_height=2.0,
                             launch_angles_deg=np.linspace(-8.0, 8.0, 17),
                             max_range=600.0, n_steps=600)
rays.plot()   # shallow rays return to the ground and bounce on down-range
plt.show()
```

</details>

### Closed-form geometry (linear gradient)

For a **linear** profile every ray is an exact **circular arc**. Its radius of
curvature is (Salomons Sec. 4.4; Attenborough Ch. 11):

$$
R_c = \frac{1}{|g|\,\xi}, \qquad \xi = \frac{\cos\theta_0}{c(\text{launch})},
\qquad\Rightarrow\qquad
R_c = \frac{c_0}{|g|\cos\theta_0},
$$

exposed as `ray_curvature_radius`. A ray launched at angle $\theta_0$ in
downward refraction turns at the height $R_c(1 - \cos\theta_0)$. For an
**upward-refracting**
linear profile the ground-grazing ray bounds a region beyond which no direct or
once-reflected ray arrives, at the closed-form **shadow-zone distance**
(`shadow_zone_distance`):

$$
x_\text{shadow} = \sqrt{2 R_c}\left(\sqrt{h_s} + \sqrt{h_r}\right),
\qquad R_c = \frac{c_0}{|g|}.
$$

These closed forms are the exact oracle for the ray tracer: a circle fit of a
traced ray recovers $R_c$ to machine precision.

## 3. Parabolic equation (Green's Function PE)

The **parabolic equation** is the reference method for refraction and shadow
zones at long range. It replaces the wave equation with a one-way (outgoing)
equation valid within a limiting elevation angle, and marches it in range on a
range-height grid. `atmospheric_parabolic_equation` implements the **Green's
Function PE** (GFPE, Salomons Appendix H), the atmospheric member of the same
split-step Fourier family as the ocean
[`parabolic_equation`](../../underwater/underwater-solvers.md). Each range step:

1. transforms the field to the vertical-wavenumber domain (FFT);
2. applies the free-space propagator
   $\exp(i\,\Delta r\,(\sqrt{k_a^2 - k_z^2} - k_a))$ together
   with the finite-impedance ground reflection
   $R(k_z) = (k_z Z - k_0)/(k_z Z + k_0)$ (Salomons Eq. H.28);
3. transforms back and adds the surface-wave residue of the reflection pole
   at $k_z = -k_0/Z$ (the third term of Eq. H.49, present for a passive
   ground, $\operatorname{Im}(Z) > 0$);
4. applies the refraction phase screen $\exp(i\,\Delta r\,(k(z) - k_a))$
   (Eq. H.58).

The source is a **Gaussian starter** with its ground image (Eqs. G.64, G.76)
and an **absorbing layer** at the top of the grid (Sec. G.9) suppresses
top-boundary reflections. The result is the **relative sound level**
$\Delta L(z, r) = 20\log_{10}(|p| R_1)$ (dB re free field, Salomons Eq. 3.6) over
the whole range-height plane.

```python
from phonometry import atmospheric_parabolic_equation, log_linear_sound_speed_profile

profile = log_linear_sound_speed_profile(-1.0, ground_speed=340.0)  # upward
pe = atmospheric_parabolic_equation(400.0, profile, source_height=2.0,
                                    flow_resistivity=200e3,  # grassland
                                    max_range=600.0, max_height=40.0)
pe.level_at_height(2.0)   # relative level vs range at 2 m
pe.plot()                 # the range-height relative-level field
```

Cut the field at the receiver height and the whole story of this page is one
figure: with everything else identical, the sign of the gradient alone moves
the 400 Hz level at 600 m by more than 30 dB.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/atmospheric_pe_range_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/atmospheric_pe_range.svg" alt="GFPE relative sound level against range at a 2 m receiver height over grassland at 400 Hz for three atmospheres. The downward-refracting curve recovers from the ground dip back toward 0 dB re free field, the homogeneous dashed curve decays gently to about minus 27 dB at 600 m, and the upward-refracting curve plunges past minus 40 dB beyond the dotted closed-form shadow-zone boundary near 110 m" width="88%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import warnings

import matplotlib.pyplot as plt

from phonometry import (
    atmospheric_parabolic_equation,
    linear_sound_speed_profile,
    log_linear_sound_speed_profile,
    shadow_zone_distance,
)

cases = [
    (log_linear_sound_speed_profile(+1.0, ground_speed=340.0), "Downward (b = +1 m/s)"),
    (linear_sound_speed_profile(0.0, ground_speed=340.0), "Homogeneous (b = 0)"),
    (log_linear_sound_speed_profile(-1.0, ground_speed=340.0), "Upward (b = -1 m/s)"),
]
fig, ax = plt.subplots(figsize=(11, 6.2))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for profile, label in cases:
        pe = atmospheric_parabolic_equation(400.0, profile, source_height=2.0,
                                            flow_resistivity=200e3,
                                            max_range=600.0, max_height=40.0)
        ax.plot(pe.ranges, pe.level_at_height(2.0), label=label)
# The closed-form boundary of the equivalent linear upward gradient (its
# 10 m mean), the dotted line of the figure.
up = cases[2][0]
grad = float(up.speed_at(10.0) - 340.0) / 10.0
ax.axvline(shadow_zone_distance(grad, 2.0, 2.0, ground_speed=340.0),
           color="k", ls=":", label="Shadow-zone boundary")
ax.set(xlabel="Range [m]", ylabel="Level re free field [dB]", ylim=(-40, 10))
ax.legend()
plt.show()
```

</details>

The ground impedance is supplied directly (`impedance=`, a normalized complex
value in the $e^{-i\omega t}$ convention of Salomons, with
$\operatorname{Im}(Z) > 0$ for a passive ground), as a `PorousMediumResult`,
or from an effective `flow_resistivity` via
the [porous models](../../materials/absorbers/porous-absorbers.md) of the materials domain; the porous
models work in the opposite $e^{+j\omega t}$ convention, so their impedance is
conjugated internally.

The clip below runs the full wave physics through both signs of the
logarithmic profile: a steady 50 Hz source 2 m over rigid ground in a 2D FDTD
slice, with the library's ray fans traced through the same $c(z)$ profiles laid
over the fields. Downwind the fronts bend back and stream along the ground to
the 350 m receiver; upwind they lift off the surface and the ground goes
quiet beyond the ray-model shadow boundary.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_refraction_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_refraction.gif" alt="Animation: a steady 50 Hz source 2 m above rigid ground in a 2D FDTD slice, downwind and upwind panels with their logarithmic effective sound-speed profiles drawn beside them; downwind the wavefronts curve back to the ground and keep the 350 m receiver loud, upwind they lift away and an acoustic shadow opens beyond the annotated 109 m ray-model boundary, with the library ray fans overlaid and a closing spreading-compensated RMS map showing the bright ground duct against the dark shadow wedge" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_refraction.webm)

## 4. Validation

The models are anchored by independent oracles:

- **Homogeneous limit → spherical ground effect.** With a zero gradient the
  GFPE field must reproduce the exact Weyl-Van der Pol
  [`ground_effect`](ground-barriers.md) at every range. It does so to a few
  tenths of a dB over grassland on the default grid (source and receiver near
  the ground, 50 m to 1 km), and to the coherent **+6 dB** two-ray enhancement
  over a rigid ground.
- **Exact ray geometry.** For a linear profile the traced ray is a circular arc
  whose fitted radius matches the closed-form `ray_curvature_radius` to machine
  precision, and the turning height matches $R_c(1 - \cos\theta_0)$.
- **Reciprocity.** Swapping the source and receiver heights leaves the PE level.
- **Shadow zone.** Over an upward-refracting profile the PE level collapses far
  inside the closed-form shadow distance.

These checks are pinned numerically in the
[conformance report](../../CONFORMANCE.md) (section "Atmospheric refraction").

## What this guide covers

**Covered.** The ray model — Runge-Kutta integration of Snell's law in
`atmospheric_ray_paths` — with the closed-form linear-profile geometry
(`ray_curvature_radius`, `shadow_zone_distance`, Salomons Sec. 4.4), and the
Green's Function parabolic equation (`atmospheric_parabolic_equation`, Salomons
Appendices G and H): the Gaussian starter, the finite-impedance ground
reflection, the refraction phase screen and the absorbing top layer. The
homogeneous limit is checked against the exact spherical ground effect (a few
tenths of a decibel, including the +6 dB rigid-ground enhancement), the ray
geometry to machine precision, and reciprocity and the shadow-zone collapse as
tests.

**Not covered.** Both models take one effective sound-speed profile that varies
with height alone. A horizontally inhomogeneous atmosphere — a gradient that
changes along the path — is outside their range-independent formulation, which
is the airborne counterpart of the range-independent ocean solvers. Both also
assume flat ground at $z = 0$; neither accepts a terrain elevation profile.

## See also

- Theory: [Outdoor propagation](../../reference/theory/environment-transport.md#outdoor-propagation-general-method-iso-9613-2): the meteorological assumptions ISO 9613-2 makes, which is what refraction takes apart.

## References

- Salomons, E. M. (2001). *Computational Atmospheric Acoustics*. Kluwer /
  Springer. Chapter 4 (ray model, PE examples), Appendix G (CNPE), Appendix H
  (GFPE).
- Attenborough, K., & Van Renterghem, T. (2021). *Predicting Outdoor Sound*
  (2nd ed.). CRC Press. Chapter 11 (refraction and ray models).

---

← [Spherical ground and barriers](ground-barriers.md) ·
[Outdoor propagation (ISO 9613-2)](outdoor-propagation.md) ·
[Underwater propagation solvers](../../underwater/underwater-solvers.md) ·
[Documentation index](../../README.md)
