← [Documentation index](README.md)

# Metamaterial Absorbers

A porous blanket absorbs low frequencies only by growing thick: the
[porous models](porous-absorbers.md) put the useful absorption where the layer
is a fair fraction of the wavelength, which at 300 Hz means tens of
centimetres. Acoustic metamaterial absorbers break that rule with resonance
instead of bulk. A rigid panel of thin closed slits, each loaded by Helmholtz
resonators, slows the sound inside the slit until a 3 cm panel resonates where
a plain cavity of the same depth would need to be ten times deeper, and tuning
the panel's internal losses against its leakage produces **perfect
absorption**, $\alpha = 1$, at a design frequency where the panel is only
$\lambda/38$ deep; published designs of the same family reach $\lambda/88$
(Jiménez et al. 2016). This guide covers the critical-coupling condition that
makes perfect absorption possible, the slow-sound slit panel implemented in
the library, and the classical resonant absorbers re-read through the same
lens.

## 1. Critical coupling: when loss exactly balances leakage

A rigidly backed absorber panel is a one-port resonator: sound enters through
the face, rings inside, and either leaks back out or is dissipated. Its
reflection coefficient over the complex frequency plane has a pole and a zero
for each resonance. Two rates compete at the resonance: the **leakage rate**,
how fast the stored energy radiates back into the incident medium, and the
**intrinsic loss rate**, how fast the visco-thermal boundary layers of the
structure dissipate it.

- Too little loss (an almost lossless panel) and the energy mostly leaks back
  out: the reflection dips but recovers, $|R| > 0$.
- Too much loss (an over-damped panel) and the wave barely enters: the panel
  starts to look rigid again, $|R| > 0$ from the other side.
- When the two rates are **equal**, the zero of the reflection coefficient
  lands exactly on the real-frequency axis: at that frequency nothing is
  reflected, $R = 0$ and $\alpha = 1 - |R|^2 = 1$. This is the
  **critical-coupling condition** (Jiménez et al. 2017).

In impedance terms the same statement reads $Z = Z_0$: the panel's surface
impedance is purely resistive and matched to the incident medium,
$\mathrm{Re}(Z)\cos\theta = Z_0$ and $\mathrm{Im}(Z) = 0$. The condition says
nothing about thickness, which is exactly the loophole metamaterial absorbers
exploit: make the resonance deep-subwavelength with slow sound, then tune the
loss to match. The balance is delicate by design, and the library's
`critical_coupling_design` solves it numerically rather than by trial: at the
returned geometry the normalised impedance sits at $1 + 0j$ and the
reflection at zero.

```python
import numpy as np
from phonometry import (
    HelmholtzResonator, critical_coupling_design, slit_helmholtz_absorber,
)

base = HelmholtzResonator(
    neck_length=1.0e-3, neck_side=3.0e-3,
    cavity_length=30.0e-3, cavity_side=27.0e-3,
)
design = critical_coupling_design(300.0, base, lattice_step=3.0e-2, period=5.0e-2)

f = np.array([300.0])
res = slit_helmholtz_absorber(
    f, design.resonator, slit_height=design.slit_height,
    lattice_step=3.0e-2, period=5.0e-2,
)
z = res.normalized_impedance[0]
print(round(z.real, 2), round(z.imag, 2))       # 1.0 -0.0  (matched)
print(round(float(np.abs(res.reflection[0])), 3))   # 0.0   (reflection zero)
```

## 2. The slow-sound slit panel

A rigid panel of thin closed slits, each loaded on its upper wall by an array
of Helmholtz resonators, is a locally reacting, deep-subwavelength absorber
(Jiménez et al. 2017). The resonators slow the sound in the slit, pulling its
resonance far below the quarter-wavelength frequency, and the visco-thermal
losses of the sub-millimetre slit and the resonator necks make *perfect*
absorption possible: when the intrinsic loss balances the leakage, the
reflection zero lands on the real-frequency axis and $\alpha = 1$ (critical
coupling). The panel transfer matrix is the chain
$T = M_{\Delta l}\,\prod_n (M_s\,M_{HR}^{(n)}\,M_s)$; the rigidly-backed
reflection factor is
$R = (T_{11}\cos\theta - Z_0 T_{21})/(T_{11}\cos\theta + Z_0 T_{21})$ with
$Z_0 = \rho_0 c_0 / S_0$. The slit uses the narrow-channel visco-thermal
parameters and the square necks and cavities the rectangular-duct series of
Stinson (1991). `critical_coupling_design` inverts the model, tuning
the cavity length and the slit height so both matching conditions
$\mathrm{Re}(Z)\cos\theta = Z_0$ and $\mathrm{Im}(Z) = 0$ hold.

```python
import numpy as np
from phonometry import (
    HelmholtzResonator, critical_coupling_design, slit_helmholtz_absorber,
)

base = HelmholtzResonator(
    neck_length=1.0e-3, neck_side=3.0e-3,
    cavity_length=30.0e-3, cavity_side=27.0e-3,
)
design = critical_coupling_design(300.0, base, lattice_step=3.0e-2, period=5.0e-2)
print(round(design.absorption, 4))          # ~1.0 (perfect absorption)

f = np.linspace(150.0, 500.0, 700)
res = slit_helmholtz_absorber(
    f, design.resonator, slit_height=design.slit_height,
    lattice_step=3.0e-2, period=5.0e-2,
)
res.plot()   # alpha(f) with |R| overlaid; peak = 1 at 300 Hz
```

The solved geometry is worth a look before the absorption curve. The
resonator draws itself with `.plot()` and the panel result with
`.plot_geometry()`, both dimensioned and to scale.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/helmholtz_resonator_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/helmholtz_resonator_geometry.svg" alt="To-scale cross-section of the square-section Helmholtz resonator the 300 Hz critical-coupling design starts from: a neck 3 mm wide and 1 mm long opening into a rigid-walled cavity 27 mm wide and 30 mm deep, with the four defining dimensions dimensioned" width="80%"></picture>

*The four numbers that define the resonator: the neck side and length set
the moving mass and most of the loss, the cavity side and length set the
stiffness. `critical_coupling_design` keeps this cross-section and retunes
the cavity length to place the resonance.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import HelmholtzResonator

resonator = HelmholtzResonator(
    neck_length=1.0e-3, neck_side=3.0e-3,
    cavity_length=30.0e-3, cavity_side=27.0e-3,
)
resonator.plot()   # dimensioned cross-section, to scale
plt.show()
```

</details>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/slit_absorber_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/slit_absorber_geometry.svg" alt="To-scale cross-section of one period of the critically-coupled slit panel: a slit about 1 mm high running the 30 mm depth of the panel, loaded by the tuned Helmholtz resonator cavity that fills the rest of the 50 mm period, with the rigid backing behind and the incident sound arriving from the left" width="80%"></picture>

*One period of the solved design, to scale: the whole panel is 30 mm deep,
$\lambda/38$ at 300 Hz, and the sub-millimetre slit that does all the
absorbing is barely visible. That is exactly the point of the slow-sound
mechanism.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import HelmholtzResonator, critical_coupling_design, materials

base = HelmholtzResonator(
    neck_length=1.0e-3, neck_side=3.0e-3,
    cavity_length=30.0e-3, cavity_side=27.0e-3,
)
design = critical_coupling_design(
    300.0, base, lattice_step=3.0e-2, period=5.0e-2,
)

# The free function draws any resonator list; a slit_helmholtz_absorber
# result retains its geometry, so res.plot_geometry() draws the same period.
materials.plot_slit_absorber_geometry(
    [design.resonator], slit_height=design.slit_height,
    lattice_step=3.0e-2, period=5.0e-2,
)
plt.show()
```

</details>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/slow_sound_absorber_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/slow_sound_absorber.svg" alt="Absorption of a slit panel loaded by one Helmholtz resonator: the critically-coupled design reaches alpha = 1 at 300 Hz, while narrowing or widening the slit breaks the balance and lowers the peak; the panel depth is lambda/38" width="80%"></picture>

*One resonator, one slit, one loss-versus-leakage balance: the critically
coupled design reaches $\alpha = 1$ at 300 Hz in a panel only $\lambda/38$
deep; detuning the slit height drops the peak below one.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import (
    HelmholtzResonator, critical_coupling_design, slit_helmholtz_absorber,
)

a, d, f0 = 3.0e-2, 5.0e-2, 300.0
base = HelmholtzResonator(1.0e-3, 3.0e-3, 30.0e-3, 27.0e-3)
design = critical_coupling_design(f0, base, lattice_step=a, period=d)
h0 = design.slit_height

f = np.linspace(150.0, 500.0, 700)
fig, ax = plt.subplots()
for factor, label in [(1.0, "critically coupled"),
                      (0.6, "narrow slit"), (1.7, "wide slit")]:
    res = slit_helmholtz_absorber(
        f, design.resonator, slit_height=factor * h0,
        lattice_step=a, period=d,
    )
    ax.plot(f, res.absorption, label=label)
ax.set(xlabel="Frequency [Hz]", ylabel="Absorption coefficient")
ax.legend()
plt.show()
```

</details>

The clip below drives this exact cell in a virtual plane-wave tube: the
sub-millimetre slit and its resonator are meshed on the FDTD grid and filled
with the model's visco-thermal effective fluids. At the design slit height
the standing wave collapses and the tone dies inside the panel; at 1.7 times
the height the loss balance breaks and the reflection rebuilds it.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_slit_absorber_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_slit_absorber.gif" alt="Animation: a 300 Hz plane tone in a 2D FDTD tube meets the meshed critical-coupling cell, its 0.98 mm slit and Helmholtz resonator resolved on the grid and shown in a zoomed panel; at the design slit height the pressure envelope stays flat and the annotated library absorption is 1.00, while the 1.7 times wider slit stands a deep wave in front of the panel and the absorption drops to 0.34" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_slit_absorber.webm)

## 3. How slow is the sound?

The name of the mechanism is measurable. The transfer-matrix result retains
the retrieved effective wavenumber of the loaded slit, so the phase speed
inside it is one division away, and the comparison with the empty slit says
everything about why the panel can be thin:

```python
import numpy as np
from phonometry import (
    HelmholtzResonator, critical_coupling_design, slit_helmholtz_absorber,
)

base = HelmholtzResonator(
    neck_length=1.0e-3, neck_side=3.0e-3,
    cavity_length=30.0e-3, cavity_side=27.0e-3,
)
design = critical_coupling_design(300.0, base, lattice_step=3.0e-2, period=5.0e-2)
res = slit_helmholtz_absorber(
    np.array([300.0]), design.resonator, slit_height=design.slit_height,
    lattice_step=3.0e-2, period=5.0e-2,
)

# Phase speed of the slit mode at the design frequency, from k_eff.
c_eff = 2 * np.pi * 300.0 / res.effective_wavenumber[0].real
print(round(float(c_eff), 1))               # 37.0   [m/s]
print(round(float(c_eff / 343.0), 2))       # 0.11   about a ninth of c0

# Quarter-wave resonance of the 30 mm depth: plain air vs the loaded slit.
print(round(343.0 / (4 * 0.03)))            # 2858   [Hz] empty slit
print(round(float(c_eff) / (4 * 0.03)))     # 308    [Hz] loaded slit
```

Below their resonance the shunt resonators add compliance to the slit without
adding moving mass, and the phase speed falls to about a ninth of the free-air
value. The 30 mm slit that would resonate at 2.9 kHz empty resonates at
about 300 Hz loaded, which is precisely where the critical-coupling design of
section 1 places its reflection zero. Slow sound moves the resonance into the
deep-subwavelength regime; critical coupling then makes that resonance
perfectly absorbing.

## 4. Classical resonant absorbers through the same lens

Critical coupling is not exclusive to metamaterials; it is the organising
principle hiding inside the classical resonant absorbers of the
[porous and multilayer guide](porous-absorbers.md):

- **The microperforated panel is a critically coupled absorber avant la
  lettre.** Maa's peak absorption $4r/(1+r)^2$ equals one exactly when the
  normalised resistance $r = 1$, which is the impedance-matching condition of
  section 1 stated for a locally reacting sheet. An MPP over a cavity tuned
  to $r = 1$ is doing precisely what the slit panel does, with the viscous
  loss provided by submillimetre holes instead of a submillimetre slit; what
  it cannot do is slow the sound, so its cavity depth stays a quarter
  wavelength deep at resonance and the construction cannot reach the
  deep-subwavelength regime.
- **Membrane and decorated-membrane absorbers** trade the hole viscosity for
  the flexural loss of a limp sheet; the same loss-versus-leakage balance
  governs their peak, and adding masses to the membrane (the decorated
  membrane of the metamaterials literature) tunes the resonance downward the
  way the loading resonators tune the slit.
- **The design question is always the same.** Pick the resonance with the
  geometry, then match the loss to the leakage. The porous-layer models ask
  "how thick"; the resonant family asks "how lossy", and
  `critical_coupling_design` answers it for the slit panel exactly.

The price of resonance is bandwidth: a critically coupled peak is narrow, and
away from it the panel reflects like the rigid wall it almost is. Broadband
metamaterial absorbers stack detuned cells side by side (the iridescent
absorber of Jiménez et al. 2017 chains resonators of graded dimensions for
exactly this reason); in the library the spectrum of any candidate geometry comes
from `slit_helmholtz_absorber` over a frequency array, so a graded design is
a loop over cells.

## See also

- [Porous and Multilayer Absorbers](porous-absorbers.md): the equivalent-fluid
  and resonant layers of the classical family, including the Maa
  microperforated panel that section 4 re-reads.
- [Metadiffusers](metadiffusers.md): the same slit and resonator cell tuned
  for controlled reflection phase instead of perfect absorption; its ternary
  designs borrow the critically coupled slit as their `0` state.
- [Impedance Tube](impedance-tube.md): the normal-incidence measurement a
  built panel would be verified in, and the virtual FDTD tube behind the
  animation above.
- [Airflow Resistance](airflow-resistance.md): the viscous physics of the
  narrow channels, measured on the bulk material.
- API reference: [`materials.slow_sound_absorber`](https://jmrplens.github.io/phonometry/reference/api/materials/slow-sound-absorber/).

## References

- Jiménez, N., Groby, J.-P., Pagneux, V., & Romero-García, V. (2017).
  Iridescent perfect absorption in critically-coupled acoustic metamaterials
  using the transfer matrix method. *Applied Sciences*, 7(6), 618.
  [doi:10.3390/app7060618](https://doi.org/10.3390/app7060618). The slit +
  Helmholtz-resonator transfer-matrix model and the critical-coupling
  condition implemented in `slit_helmholtz_absorber`, plus the graded
  (iridescent) broadband chains of section 4.
- Jiménez, N., Huang, W., Romero-García, V., Pagneux, V., & Groby, J.-P.
  (2016). Ultra-thin metamaterial for perfect and quasi-omnidirectional sound
  absorption. *Applied Physics Letters*, 109(12), 121902.
  [doi:10.1063/1.4962328](https://doi.org/10.1063/1.4962328). The resonator
  impedance (Eq. A23) and its radiation end corrections (Eqs. A24-A27), and
  the published λ/88 perfect absorber cited in the introduction.
- Stinson, M. R. (1991). The propagation of plane sound waves in narrow and
  wide circular tubes, and generalization to uniform tubes of arbitrary
  cross-sectional shape. *Journal of the Acoustical Society of America*,
  89(2), 550-558. [doi:10.1121/1.400379](https://doi.org/10.1121/1.400379).
  The visco-thermal effective parameters of the slit and the square necks
  and cavities.
- Maa, D.-Y. (1998). Potential of microperforated panel absorber.
  *Journal of the Acoustical Society of America*, 104(5), 2861-2866.
  [doi:10.1121/1.423870](https://doi.org/10.1121/1.423870).
  The MPP impedance and design formulas whose $r = 1$ peak condition
  section 4 identifies with critical coupling.
- Cox, T. J., & D'Antonio, P. (2017). *Acoustic absorbers and diffusers:
  Theory, design and application* (3rd ed.). CRC Press.
  ISBN 978-1-4987-4099-9.
  [doi:10.1201/9781315369211](https://doi.org/10.1201/9781315369211).
  The classical resonant-absorber designs the metamaterial family extends.
- Jiménez, N., Umnova, O., & Groby, J.-P. (Eds.). (2021). *Acoustic waves in
  periodic structures, metamaterials, and porous media* (Topics in Applied
  Physics, Vol. 143). Springer.
  [doi:10.1007/978-3-030-84300-7](https://doi.org/10.1007/978-3-030-84300-7).
  The book-length treatment of critical coupling, slow sound and resonant
  metamaterial absorbers.

## Standards

No standard governs these prediction models; they are journal methods
(Jiménez et al. 2016/2017; Stinson 1991) implemented clean-room from the
cited sources. The model is pinned to its exact analytic anchors: the
critical-coupling design reaches $\alpha = 1$ at the design frequency, the
slit and square-duct effective densities reduce to the Poiseuille flow
resistivities $12\eta/h^2$ and $28.454\eta/w^2$ as $\omega \to 0$, and the
effective parameters tend to $\rho_0$, $\kappa_0$ as the boundary layers
vanish. A built panel would be measured in the
[impedance tube](impedance-tube.md) (ISO 10534-2) or the
[reverberation room](absorption-measurement.md) (ISO 354); misprints found in
the source papers during this work are recorded in the
[errata registry](ERRATA.md).
