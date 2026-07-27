← [Documentation index](README.md)

# Porous and Multilayer Absorbers

Given a porous material's **flow resistivity** (the quantity the
[flow rig](materials.md) measures) the classical equivalent-fluid models
predict its complex characteristic impedance and wavenumber, and a
**transfer-matrix** stack of layers (porous blankets, air gaps, perforated
and microperforated panels, membranes) predicts the absorption coefficient
of the whole construction before anything is built. This page covers the
three porous models (Delany–Bazley, Miki, Johnson–Champoux–Allard), the
multilayer solver, the resonant sheet layers of Maa and the random-incidence
(Paris) integral. The measurement counterparts live in
[Acoustic Materials](materials.md); rating the predicted spectrum lives in
[the same page's ISO 11654 section](materials.md).

## 1. Equivalent-fluid models of a porous material

A rigid-frame porous material behaves like an *equivalent fluid* with a
complex characteristic impedance $Z_c$ and wavenumber $k$ (time convention
$e^{+j\omega t}$, so a passive medium has $\mathrm{Im}(k) < 0$).

**Delany–Bazley** (Mechel 2e Sect. G.11; Bies 5e Appendix D, Table D.1;
Hopkins Eqs. 1.171–1.174) is the one-parameter power law in the absorber
variable $X = \rho_0 f / \sigma$:

$$
\frac{Z_c}{\rho_0 c_0} = 1 + C_1 X^{-C_2} - j\,C_3 X^{-C_4}, \qquad
\frac{k}{k_0} = 1 + C_5 X^{-C_6} - j\,C_7 X^{-C_8},
$$

with the classic rockwool/fibreglass coefficients
$(0.0571,\,0.754,\,0.087,\,0.732,\,0.0978,\,0.700,\,0.189,\,0.595)$ and a
stated fit range $0.01 < X < 1.0$ (porosity close to one). The library also
ships the Table D.1 presets fitted to polyester (`"garai_pompoli"`) and to
foams (`"dunn_davern"`, `"wu"`). Outside the fit range a
`PorousAbsorberWarning` is raised and the extrapolated values are still
returned; the classic failure is a *negative* real part of the layer
input impedance at low frequency (Mechel Sect. G.12).

**Miki** (1990) refitted the same Delany–Bazley data under a passivity
(positive-real) constraint, so the model stays physically well behaved below
the fit range; it is the usual choice when a one-parameter model must be
evaluated broadband.

**Johnson–Champoux–Allard (JCA)** is the five-parameter semi-phenomenological
model (Cox & D'Antonio 3e Eqs. 6.19–6.25): flow resistivity $\sigma$,
porosity $\phi$, tortuosity $\alpha_\infty$ and the viscous/thermal
characteristic lengths $\Lambda$, $\Lambda'$ give the effective density and
bulk modulus with the exact limits $j\omega\rho_e \to \sigma$ at DC,
$\rho_e \to (\alpha_\infty \rho_0/\phi)(1 + (1-j)\,\delta_v/\Lambda)$ at high
frequency, and the isothermal-to-adiabatic transition in $K_e$.

```python
import numpy as np
from phonometry import materials

f = np.geomspace(200.0, 4000.0, 200)
db = materials.delany_bazley(f, 20000.0)          # sigma in Pa s/m2
mk = materials.miki(f, 20000.0)
jca = materials.johnson_champoux_allard(
    f, 20000.0, porosity=0.98, tortuosity=1.0,
    viscous_length=8.7e-5, thermal_length=8.7e-5,
)
print(np.round(db.normalized_impedance[0], 3))    # (2.598-2.209j)
print(np.round(mk.normalized_impedance[0], 3))    # (2.286-1.965j)
print(np.round(jca.normalized_impedance[0], 3))   # (2.321-2.075j)

db.plot()   # normalised Zc and k components vs frequency
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/porous_medium_model_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/porous_medium_model.svg" alt="Normalised characteristic impedance and wavenumber of a porous material with a flow resistivity of 20 kPa s/m2 predicted by the Miki model on a log-log grid: the real and imaginary components all fall towards unity and zero as frequency rises" width="80%"></picture>

*The classical presentation of an equivalent-fluid model (Cox & D'Antonio
3e, Figs. 6.19–6.20): at low frequency the viscous forces dominate and the
material looks stiff and lossy (all four components large); as frequency
rises the components fall towards the free-air limits
$Z_c \to \rho_0 c_0$ and $k \to k_0$, so a thin layer only works where its
thickness is a fair fraction of the wavelength inside the material.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

f = np.geomspace(100.0, 5000.0, 260)
mk = materials.miki(f, 20000.0)          # sigma = 20 kPa s/m^2

# One line: normalised Zc and k components on a log-log grid.
mk.plot()
plt.show()

# By hand, from the result's fields:
fig, ax = plt.subplots()
ax.loglog(f, mk.normalized_impedance.real, label="Re(Zc)/rho c")
ax.loglog(f, -mk.normalized_impedance.imag, "--", label="-Im(Zc)/rho c")
ax.loglog(f, mk.normalized_wavenumber.real, label="Re(k)/k0")
ax.loglog(f, -mk.normalized_wavenumber.imag, "--", label="-Im(k)/k0")
ax.set(xlabel="Frequency [Hz]", ylabel="Normalised characteristic value")
ax.legend()
plt.show()
```

</details>

The three models agree closely over the Delany–Bazley fit range (Cox &
D'Antonio Figs. 6.19–6.21 make the same comparison); JCA extends the
prediction physically outside it. A `PorousMediumResult` built from measured
data (for example the $Z_c$, $k$ recovered by the
[ASTM E2611 transfer-matrix reduction](materials.md)) plugs into the layer
solver exactly like a modelled one.

## 2. Multilayer prediction by transfer matrices

Each fluid layer of thickness $d$ contributes the chain matrix (Cox &
D'Antonio Eq. 2.29; equivalently the impedance recursion of Bies Eq. D.95
and Mechel Sect. D.4)

$$
\begin{bmatrix} p \\ u \end{bmatrix}_{\text{front}} =
\begin{bmatrix}
\cos(k_x d) & j Z_x \sin(k_x d) \\
j \sin(k_x d)/Z_x & \cos(k_x d)
\end{bmatrix}
\begin{bmatrix} p \\ u \end{bmatrix}_{\text{back}},
$$

with the in-depth wavenumber $k_x = \sqrt{k^2 - k_0^2 \sin^2\theta}$ from
Snell's law and $Z_x = Z_c k / k_x$. Thin resonant sheets enter as series
impedances $[[1, z], [0, 1]]$. Closing the chain with a rigid wall (or free
air, or any impedance) gives the surface impedance, the reflection factor
$R(\theta)$ and $\alpha(\theta) = 1 - |R|^2$. A single hard-backed porous
layer reduces to the textbook closed form
$Z_s = -j Z_c \cot(k d)$ (Mechel Sect. D.3, Eq. 1).

```python
import numpy as np
from phonometry import materials

f = np.geomspace(200.0, 4000.0, 300)
med = materials.miki(f, 20000.0)
res = materials.layered_absorber(f, [materials.PorousLayer(0.05, med)])
i = np.argmin(np.abs(f - 1000.0))
print(round(res.absorption[i], 3))               # 0.937 at 1 kHz

res.plot()   # alpha(f) with |R| overlaid
```

The solver evaluates the physical quantities through a numerically robust
admittance recursion (immune to the $e^{|\mathrm{Im}(k_x)| d}$ overflow of
raw matrix entries for extremely attenuating layers) and still exposes the
full chain matrix (reciprocal by construction, $\det T = 1$) in
`transfer_matrix`, ready for the
[ASTM E2611 machinery](materials.md) (`TransferMatrix`).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/porous_absorber_designs_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/porous_absorber_designs.svg" alt="Predicted normal-incidence absorption of four 50 mm constructions: a porous layer absorbs broadband from mid frequency up, a microperforated panel over a cavity peaks near 700 Hz, a perforated panel over porous peaks near 500 Hz, and a membrane over a porous-filled cavity peaks near 175 Hz; dotted vertical lines mark the closed-form Helmholtz and membrane resonances" width="80%"></picture>

*Four constructions, one 50 mm budget: the porous layer works broadband but
fades at low frequency; the microperforated, perforated and membrane designs
trade bandwidth for a resonant peak placed ever lower. The dotted lines are
the shallow-cavity closed forms; the full model sits below them because the
viscous plug mass and the finite cavity depth are not negligible.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials as m

f = np.geomspace(50.0, 5000.0, 500)
med = m.miki(f, 20000.0)
med_light = m.miki(f, 10000.0)
designs = {
    "Porous 50 mm": [m.PorousLayer(0.05, med)],
    "MPP + cavity": [m.MicroperforatedPlateLayer(0.5e-3, 0.15e-3, 0.008),
                     m.AirLayer(0.048)],
    "Perforated + porous": [m.PerforatedPlateLayer(0.006, 0.0025, 0.05),
                            m.PorousLayer(0.025, med), m.AirLayer(0.019)],
    "Membrane + porous": [m.MembraneLayer(2.0), m.AirLayer(0.01),
                          m.PorousLayer(0.038, med_light)],
}
fig, ax = plt.subplots()
for label, layers in designs.items():
    ax.semilogx(f, m.layered_absorber(f, layers).absorption, label=label)
ax.set(xlabel="Frequency [Hz]", ylabel="Absorption coefficient")
ax.legend()
plt.show()
```

</details>

Behind every one of those curves there is a plain layer list, read front to
back in the order the incident wave meets it. `plot_absorber_stack` draws
that list to scale before any physics runs, and a solved result retains its
layers, so `materials.layered_absorber(f, layers).plot_geometry()` draws the
same cross-section from the result itself.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/absorber_stack_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/absorber_stack_geometry.svg" alt="To-scale cross-section of a three-layer absorber: a 1 mm microperforated plate at the front, a 30 mm air cavity and a 50 mm porous layer against the rigid backing at the right, each thickness dimensioned, with the incident sound arriving from the left" width="80%"></picture>

*The layer list as the wave meets it: microperforated plate, air cavity,
porous layer, rigid backing. Drawing the stack to scale before computing
anything catches the classic metres-versus-millimetres slip in a thickness
at a glance.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

f = np.linspace(200.0, 4000.0, 100)
layers = [
    materials.MicroperforatedPlateLayer(0.001, 0.0002, 0.01),
    materials.AirLayer(0.03),
    materials.PorousLayer(0.05, materials.miki(f, 20000.0)),
]

# The free function draws any layer list; a solved result retains its
# layers, so this draws the same cross-section:
#   materials.layered_absorber(f, layers).plot_geometry()
materials.plot_absorber_stack(layers)
plt.show()
```

</details>

## 3. Resonant sheets: perforated, microperforated, membrane

**Perforated panel.** The air plugs in the holes are the mass of a
Helmholtz resonator: $m = (\rho_0/\varepsilon)\,[t + 2\delta a +
\sqrt{8\nu/\omega}\,(1 + t/2a)]$ with the open area $\varepsilon$, the
end-correction factor $\delta$ per orifice end and the visco-thermal
resistance $r = (\rho_0/\varepsilon)\sqrt{8\nu\omega}\,(1 + t/2a)$
(Cox & D'Antonio Eqs. 7.6/7.12). The default end correction is the
Fok-function interaction fit $\delta = 0.85\,(1 - 1.47\sqrt{\varepsilon}
+ 0.47\varepsilon^{3/2})$ (Table 7.1), valid for any open area. For a
shallow cavity the resonance is
$f_0 = (c_0/2\pi)\sqrt{\varepsilon/(t'\,d)}$ (Eq. 7.4).

**Microperforated panel (MPP).** With submillimetre holes the viscous
boundary layer fills the orifice and the panel absorbs *without any porous
material*. The library implements Maa's exact short-tube impedance
(Maa 1998, Eq. 2),

$$
z_1 = j\omega\rho_0 t \left[ 1 -
\frac{2}{x\sqrt{-j}}\,\frac{J_1(x\sqrt{-j})}{J_0(x\sqrt{-j})} \right]^{-1},
\qquad x = a\sqrt{\rho_0 \omega/\eta},
$$

plus the Eq. 5 end corrections (surface resistance
$\tfrac{1}{2}\sqrt{2\omega\rho_0\eta}$ and piston reactance $0.85\,d$
total), divided by the open area. The perforate constant $x$ (proportional to the
hole radius over the viscous boundary-layer thickness) governs everything: at
the resonance $\omega_0 m = \cot(\omega_0 D/c_0)$ the peak absorption is
$4r/(1+r)^2$ and the half-absorption bandwidth is
$f_2/f_1 = \pi/\mathrm{arccot}(1+r) - 1$ (Maa Eqs. 9–21, Table I).

```python
import numpy as np
from phonometry import materials

# Maa (1998) Fig. 5: d = t = 0.2 mm, holes every 2.5 mm, cavity 6 cm.
eps = (np.pi / 4.0) * (0.2 / 2.5) ** 2
f = np.linspace(100.0, 4000.0, 2000)
res = materials.layered_absorber(
    f, [materials.MicroperforatedPlateLayer(0.2e-3, 0.1e-3, eps),
        materials.AirLayer(0.06)],
)
i = np.argmax(res.absorption)
print(f"peak alpha = {res.absorption[i]:.2f} at {f[i]:.0f} Hz")
# peak alpha = 0.96 at 677 Hz
res.plot()   # alpha(f) with |R| overlaid: the resonant MPP peak
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mpp_absorption_peak_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mpp_absorption_peak.svg" alt="Predicted absorption of Maa's microperforated panel over a 6 cm cavity: a broad resonant peak reaching alpha 0.96 near 677 Hz, with the reflection-factor magnitude as its mirror image, and a narrow secondary feature near the half-wave cavity resonance at 2.9 kHz" width="80%"></picture>

*Maa's own Fig. 5 design, no porous material anywhere: the viscous losses in
the submillimetre holes damp the panel-cavity resonance into a broad
absorption peak. The narrow feature near 2.9 kHz is the half-wave cavity
resonance, where the cavity presents a pressure node to the panel and the
absorption collapses before the next resonance restores it.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

# Maa (1998) Fig. 5: d = t = 0.2 mm, holes every 2.5 mm, cavity 6 cm.
eps = (np.pi / 4.0) * (0.2 / 2.5) ** 2
f = np.linspace(100.0, 4000.0, 1200)
res = materials.layered_absorber(
    f, [materials.MicroperforatedPlateLayer(0.2e-3, 0.1e-3, eps),
        materials.AirLayer(0.06)],
)

# One line: alpha(f) with |R| overlaid.
res.plot()
plt.show()

# By hand, from the result's fields:
fig, ax = plt.subplots()
ax.semilogx(f, res.absorption, label="Absorption alpha")
ax.semilogx(f, np.abs(res.reflection), "--", label="Reflection factor |R|")
ax.set(xlabel="Frequency [Hz]", ylabel="Coefficient")
ax.legend()
plt.show()
```

</details>

**Membrane.** A limp impervious sheet is the surface mass $z = j\omega m$
(Cox Eq. 7.14; Bies Eq. D.96); over a cavity it resonates at the classical
$f_0 \approx 60/\sqrt{m d}$ (adiabatic; $\approx 50/\sqrt{m d}$ when the
cavity is porous-filled and isothermal, Cox Eqs. 7.9/7.10). The closed forms
are exposed as `helmholtz_resonance_frequency` and
`membrane_resonance_frequency`; the full frequency response comes from the
same layer stack.

## 4. Oblique and random incidence

`layered_absorber(..., angle=theta)` evaluates the full bulk-reacting stack
at any polar angle; sheets are locally reacting (angle-independent), fluid
layers refract per Snell's law; for an MPP over a cavity this reproduces
Maa's oblique closed form (Eq. 23) exactly. The random-incidence coefficient
is the Paris integral (Mechel Sect. D.5, Eq. 9)

$$
\alpha_{dif} = \frac{2}{\sin^2\theta_{lim}} \int_0^{\theta_{lim}}
\alpha(\theta)\,\cos\theta\,\sin\theta\,\mathrm{d}\theta,
$$

evaluated by Gauss–Legendre quadrature in `diffuse_field_absorption`
(``angle_limit`` defaults to 90°; truncations at 75–87° are in use). For a
*locally reacting* surface with known normalised impedance the integral has
the closed form of Mechel Eq. 10, exposed as `statistical_absorption`; its
maximum over all passive impedances is the published **0.951** (at
$z \approx 1.57$).

```python
import numpy as np
from phonometry import materials

f = np.array([250.0, 500.0, 1000.0, 2000.0])
med = materials.miki(f, 20000.0)
layers = [materials.PorousLayer(0.05, med)]
normal = materials.layered_absorber(f, layers)
diffuse = materials.diffuse_field_absorption(f, layers)
print(np.round(normal.absorption, 2))    # [0.26 0.62 0.94 0.95]
print(np.round(diffuse.absorption, 2))   # [0.37 0.68 0.9  0.95]
diffuse.plot()   # alpha_dif(f); overlay normal.absorption to compare

print(round(float(materials.statistical_absorption(1.567 + 0j)), 3))  # 0.951
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diffuse_field_absorption_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diffuse_field_absorption.svg" alt="Random-incidence absorption of a 50 mm porous layer by the Paris integral compared with its normal-incidence coefficient: the diffuse-field curve sits clearly above the normal-incidence one below 700 Hz and the two converge towards 0.95 at high frequency" width="80%"></picture>

*Why the reverberation room reads higher than the tube: the Paris integral
weights the oblique angles, whose waves travel a longer path inside the
layer, so $\alpha_{dif}$ exceeds the normal-incidence $\alpha(0°)$ exactly
where the layer is thin against the wavelength. This is the model-side
counterpart of the tube-versus-reverberation-room discussion of
[Acoustic Materials](materials.md).*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

f = np.geomspace(125.0, 4000.0, 200)
layers = [materials.PorousLayer(0.05, materials.miki(f, 20000.0))]
normal = materials.layered_absorber(f, layers)
diffuse = materials.diffuse_field_absorption(f, layers)

# One line, then the normal-incidence overlay on the same axes:
ax = diffuse.plot()
ax.plot(f, normal.absorption, "--", label="Normal incidence alpha(0)")
ax.legend()
plt.show()
```

</details>

## 5. Slow-sound slit panels with Helmholtz resonators

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
$Z_0 = \rho_0 c_0 / S_0$. `critical_coupling_design` inverts the model, tuning
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

## Practical notes

**Fit ranges.** Delany–Bazley warns (and extrapolates) outside
$0.01 < X < 1$ and Miki outside $0.01 < f/\sigma < 1$; treat sub-range
values as qualitative. JCA needs four extra
parameters but behaves physically everywhere; with
$\Lambda = \Lambda' = \sqrt{8\alpha_\infty\eta/(\phi\sigma)}$ and
$\alpha_\infty = 1$ it tracks Delany–Bazley over the fit range.

**Local vs. bulk reaction.** The layer solver is bulk-reacting (sound
refracts and travels inside the layers). `statistical_absorption` assumes
local reaction, a good approximation for high flow resistivity, partitioned
cavities or thin resonant facings; for thick, light porous layers integrate
the bulk model with `diffuse_field_absorption` instead (Mechel Sect. D.6).

**Where the numbers were checked.** The models are pinned digit-exact to the
printed coefficient tables (Bies Table D.1, Miki Eqs. 30–34), the solver to
the closed forms above and to the `TransferMatrix` recovery of the
[impedance-tube page](materials.md), the MPP to Maa's own approximation
(stated ~6 % agreement with the exact Eq. 2), design example and Table I,
and the Paris integral to its locally reacting closed form. Five misprints
found in the sources during this work are recorded in the
[errata registry](ERRATA.md).

## References

- Mechel, F. P. (Ed.). (2008). *Formulas of acoustics* (2nd ed.). Springer.
  [doi:10.1007/978-3-540-76833-3](https://doi.org/10.1007/978-3-540-76833-3).
  Sections D.3–D.6 (layer reflection, multilayer scheme, diffuse-field
  integrals) and G.11 (empirical porous relations).
- Bies, D. A., Hansen, C. H., & Howard, C. Q. (2017). *Engineering noise
  control* (5th ed.). CRC Press.
  [doi:10.1201/9781351228152](https://doi.org/10.1201/9781351228152).
  Appendix D: porous-material properties, Table D.1 coefficient sets and
  the layered-construction recursions D.91–D.99.
- Cox, T. J., & D'Antonio, P. (2017). *Acoustic absorbers and diffusers:
  Theory, design and application* (3rd ed.). CRC Press.
  [doi:10.1201/9781315369211](https://doi.org/10.1201/9781315369211).
  Transfer-matrix modelling (Sect. 2.6), porous models (Sect. 6.5) and
  resonant-absorber design equations (Sects. 7.3/7.5).
- Attenborough, K., & Van Renterghem, T. (2021). *Predicting
  outdoor sound* (2nd ed.). CRC Press.
  [doi:10.1201/9780429470806](https://doi.org/10.1201/9780429470806).
  Chapter 5: ground-impedance models, including the JCA family.
- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  Section 1.3.2.2: the equivalent-gas model and the Delany–Bazley SI form.
- Miki, Y. (1990). Acoustical properties of porous materials —
  Modifications of Delany–Bazley models. *Journal of the Acoustical Society
  of Japan (E)*, 11(1), 19–24.
  [doi:10.1250/ast.11.19](https://doi.org/10.1250/ast.11.19).
  The positive-real regression implemented in `miki`.
- Maa, D.-Y. (1998). Potential of microperforated panel absorber.
  *Journal of the Acoustical Society of America*, 104(5), 2861–2866.
  [doi:10.1121/1.423870](https://doi.org/10.1121/1.423870).
  The exact MPP impedance (Eq. 2), end corrections, design formulas and the
  Fig. 5 example pinned in the tests.
- Johnson, D. L., Koplik, J., & Dashen, R. (1987). Theory of dynamic
  permeability and tortuosity in fluid-saturated porous media. *Journal of
  Fluid Mechanics*, 176, 379–402.
  [doi:10.1017/S0022112087000727](https://doi.org/10.1017/S0022112087000727).
  The dynamic-tortuosity model behind the JCA effective density.
- Delany, M. E., & Bazley, E. N. (1970). Acoustical properties of fibrous
  absorbent materials. *Applied Acoustics*, 3(2), 105–116.
  [doi:10.1016/0003-682X(70)90031-9](https://doi.org/10.1016/0003-682X(70)90031-9).
  The original empirical relations and their stated validity.
- Jiménez, N., Groby, J.-P., Pagneux, V., & Romero-García, V. (2017).
  Iridescent perfect absorption in critically-coupled acoustic metamaterials
  using the transfer matrix method. *Applied Sciences*, 7(6), 618.
  [doi:10.3390/app7060618](https://doi.org/10.3390/app7060618). The slit +
  Helmholtz-resonator transfer-matrix model and the critical-coupling
  condition implemented in `slit_helmholtz_absorber`.
- Jiménez, N., Huang, W., Romero-García, V., Pagneux, V., & Groby, J.-P.
  (2016). Ultra-thin metamaterial for perfect and quasi-omnidirectional sound
  absorption. *Applied Physics Letters*, 109(12), 121902.
  [doi:10.1063/1.4962328](https://doi.org/10.1063/1.4962328). The resonator
  impedance (Eq. A23) and its radiation end corrections (Eqs. A24–A27).
- Stinson, M. R. (1991). The propagation of plane sound waves in narrow and
  wide circular tubes, and generalization to uniform tubes of arbitrary
  cross-sectional shape. *Journal of the Acoustical Society of America*,
  89(2), 550–558. [doi:10.1121/1.400379](https://doi.org/10.1121/1.400379).
  The visco-thermal effective parameters of the slit and the square necks
  and cavities.

## Standards

No standard governs these prediction models; they are textbook and journal
methods (Mechel; Bies, Hansen & Howard; Cox & D'Antonio; Attenborough & Van
Renterghem; Miki 1990; Maa 1998; Johnson et al. 1987) implemented
clean-room from the cited sources. The measurement standards they connect to
(ISO 9053-1/-2 for flow resistivity, ISO 10534-1/-2 and ASTM E2611
for the impedance tube, ISO 354 / ISO 11654 for random-incidence absorption and
rating) live in [Acoustic Materials](materials.md).
