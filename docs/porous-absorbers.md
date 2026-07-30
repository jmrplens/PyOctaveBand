← [Documentation index](README.md)

# Porous and Multilayer Absorbers

Given a porous material's **flow resistivity** (the quantity the
[flow rig](airflow-resistance.md) measures) the classical equivalent-fluid models
predict its complex characteristic impedance and wavenumber, and a
**transfer-matrix** stack of layers (porous blankets, air gaps, perforated
and microperforated panels, membranes) predicts the absorption coefficient
of the whole construction before anything is built. This page covers the
three porous models (Delany–Bazley, Miki, Johnson–Champoux–Allard), the
multilayer solver, the resonant sheet layers of Maa and the random-incidence
(Paris) integral. The measurement counterparts live in the
[impedance tube](impedance-tube.md) and
[reverberation room](absorption-measurement.md) guides; rating the predicted
spectrum lives in the latter's
[ISO 11654 section](absorption-measurement.md#3-weighted-rating-and-absorption-class-iso-11654);
and the resonant metamaterial relatives of these constructions, slow-sound
slit panels that reach perfect absorption at critical coupling, have their
own guide, [Metamaterial Absorbers](metamaterial-absorbers.md).

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
[ASTM E2611 transfer-matrix reduction](impedance-tube.md)) plugs into the layer
solver exactly like a modelled one.

### Limp frames: when the skeleton moves (Allard & Atalla 11.3.4)

Every model above assumes the frame stands still. That is only true above the
**decoupling frequency** of Zwikker and Kosten,

$$
F_d = \frac{\sigma\,\phi^2}{2\pi\rho_1},
$$

with $\rho_1$ the bulk density of the frame (the density of the sample as
weighed, not of the material the fibres are made of). Below $F_d$ the
visco-inertial coupling is strong enough for the wave in the pores to drag the
frame along, and a light frame (aeronautic-grade fibreglass, felts, thin
screens) has real inertia to contribute. Neglecting the *stiffness* of the
frame but not its *mass* in the Biot mixed pressure-displacement formulation
leaves an equivalent fluid with the same bulk modulus and a corrected effective
density (Allard & Atalla 2e Eqs. 11.53-11.55, after Panneton 2007):

$$
\tilde\rho_{\text{limp}} =
\frac{\rho_t\,\tilde\rho_{\text{eq}} - \rho_0^2}
     {\rho_t + \tilde\rho_{\text{eq}} - 2\rho_0},
\qquad \rho_t = \rho_1 + \phi\rho_0,
$$

where $\tilde\rho_{\text{eq}}$ is the rigid-frame effective density of any of
the three models above and $\rho_t$ is the apparent total density of the
material. What anchors the correction is the printed equation itself,
transcribed term by term against the page; the book tabulates no computed limp
density anywhere. Two limits it states in prose are exact as well and are
checked, but they corroborate rather than pin the form: neither constrains the
$\rho_0$ terms, and a sign-flipped variant of the same equation satisfies both.
The limits are:

- **heavy frame**: as $\rho_1$ grows the correction vanishes and the
  rigid-frame result comes back;
- **low frequency**: the rigid-frame density diverges as $\sigma/(j\omega)$,
  while the limp one converges on the finite, real $\rho_t$. A rigid frame
  forbids rigid-body motion of the sample; a limp one allows it, which is why
  the limp model is the right one for an unconstrained specimen in an
  impedance tube.

`limp_frame` takes a `PorousMediumResult` and returns another one, so it drops
straight into a `PorousLayer` in the stack.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/limp_frame_effective_density_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/limp_frame_effective_density.svg" alt="Normalised effective density against frequency from 0 to 2000 hertz for a soft fibrous layer, comparing the rigid-frame and limp-frame models. The rigid-frame real part is a flat dashed line at 1.2 while its imaginary part dives off the bottom of the plot below the 127 hertz decoupling frequency marked by a dotted vertical line. The limp-frame real part starts at the apparent total density of 25.9 at zero hertz and falls smoothly onto the rigid-frame line by about 1500 hertz, while its imaginary part dips to minus 12 near 130 hertz and then merges with the rigid-frame curve" width="88%"></picture>

*The same 50 mm soft fibrous layer under both models (Allard & Atalla
Table 11.2, the input set behind their Fig. 11.2). Above the decoupling
frequency the two predictions are indistinguishable, which is the plot's own
check on the correction; below it the rigid-frame effective density runs away
and the limp one settles on $\rho_t/\rho_0 = 25.9$.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import (
    PorousLayer, decoupling_frequency, johnson_champoux_allard,
    layered_absorber, limp_frame, limp_frame_applicable,
)

# Allard & Atalla Table 11.2: soft fibrous layer, 50 mm.
f = np.linspace(1.0, 2000.0, 800)
rigid = johnson_champoux_allard(
    f, 25e3, porosity=0.98, tortuosity=1.02,
    viscous_length=90e-6, thermal_length=180e-6,
)
limp = limp_frame(rigid, frame_density=30.0, porosity=0.98)

print(round(decoupling_frequency(25e3, porosity=0.98, frame_density=30.0), 1))
# 127.4
print(round(float(limp.effective_density[0].real), 1))     # 31.2 = rho_t
print(limp_frame_applicable(20e3), limp_frame_applicable(25e3))  # True False

limp.plot()   # normalised Zc and k of the corrected medium
plt.show()

# By hand, the Fig. 11.2 view:
rho0 = rigid.air_density
fig, ax = plt.subplots()
for medium, style in ((rigid, "--"), (limp, "-")):
    ax.plot(f, medium.effective_density.real / rho0, style)
    ax.plot(f, medium.effective_density.imag / rho0, style)
ax.set(xlabel="Frequency [Hz]", ylabel="rho_e / rho_0", ylim=(-30, 30))
plt.show()
```

</details>

The correction is only worth applying where the frame really is limp. Beranek
(1947) asked for $|K_c/K_f| < 0.05$ between the bulk modulus of the frame in
vacuum and that of the fluid in the pores; the frame-structural-interaction
study of Doutres et al. (2007) relaxed that to $0.2$, which for air
($K_f \approx P_0 = 101.3$ kPa) is the rule of thumb that the frame must be
softer than about 20 kPa. `limp_frame_applicable` applies either threshold.
Neither accounts for mounting: a thin light foam decoupled from a vibrating
structure by an air gap behaves limply well above the limit, and a material
bonded to a vibrating structure should not be treated as rigid-framed at all.

The practical consequence, for the 50 mm layer above with a rigid backing, is a
lower absorption coefficient in the bottom two thirds of an octave and a
slightly higher one either side of 500 Hz:

```python
import numpy as np
from phonometry import (
    PorousLayer, johnson_champoux_allard, layered_absorber, limp_frame,
)

bands = np.array([100, 125, 160, 200, 250, 315, 400, 500, 1000], dtype=float)
rigid = johnson_champoux_allard(
    bands, 25e3, porosity=0.98, tortuosity=1.02,
    viscous_length=90e-6, thermal_length=180e-6,
)
limp = limp_frame(rigid, frame_density=30.0, porosity=0.98)
for medium in (rigid, limp):
    print(layered_absorber(bands, [PorousLayer(0.05, medium)]).absorption.round(2))
# [0.07 0.11 0.17 0.24 0.32 0.43 0.54 0.64 0.88]   rigid frame
# [0.04 0.08 0.15 0.24 0.36 0.48 0.61 0.71 0.91]   limp frame
```

**Honest note on validation.** Allard & Atalla contains exactly one table of
computed numbers in the whole book, and it is not a surface impedance: every
prediction-versus-measurement pair in the porous chapters, Fig. 11.2 included,
is a figure. No published source checked (Allard & Atalla itself, Cox &
D'Antonio, Mechel, and the round-robin and Biot literature) tabulates the
quantity either. The anchor is therefore the printed Eq. 11.55 itself,
transcribed term by term against the printed page. The two exact limits the
book states in prose (heavy frame and $\omega \to 0$) are checked as well, but
they are weaker than they look: a sign-flipped variant of the equation
satisfies both, and reproduces the $1/\rho_1$ decay of the heavy-frame residual
too, so the limits cannot tell the printed form from that variant. The
decoupling frequency evaluated on the fully specified glass wool of their
Table 6.1, where pure arithmetic gives 43.27 Hz, is independent of all this.

### Elastic frames: the full Biot layer (Allard & Atalla 6 and 11)

The limp model throws the frame stiffness away. Keep it, and the porous layer
stops being a fluid: the Biot theory treats the skeleton as an elastic solid
coupled to the pore fluid through a potential coupling coefficient $Q$ and an
inertial coupling coefficient, and predicts **three** waves in an isotropic
material instead of one. Two are compressional and one is shear (Allard &
Atalla 2e ch. 6). The stress-strain relations are

$$
\sigma^s_{ij} = \left[(P - 2N)\theta^s + Q\theta^f\right]\delta_{ij}
  + 2N e^s_{ij}, \qquad
\sigma^f_{ij} = -\phi p\,\delta_{ij} = (Q\theta^s + R\theta^f)\delta_{ij},
$$

and for the usual case of a frame whose solid grains are much stiffer than the
frame they build ($K_s \to \infty$, true of glass, rock and polymer frames) the
three elastic coefficients follow from the shear modulus $N$, the Poisson
coefficient $\nu$ and the bulk modulus $K_f$ of the fluid in the pores
(Eqs. 6.26-6.29):

$$
R = \phi K_f, \qquad Q = (1-\phi)K_f, \qquad
P = \tfrac{4}{3}N + K_b + \frac{(1-\phi)^2}{\phi}K_f, \qquad
K_b = \frac{2N(1+\nu)}{3(1-2\nu)}.
$$

With the modified densities $\tilde\rho_{11}$, $\tilde\rho_{12}$,
$\tilde\rho_{22}$ of Eq. 6.56 the two compressional wavenumbers come out as the
eigenvalues of a 2x2 problem (Eqs. 6.67-6.69) and the shear wavenumber from
Eq. 6.83. `biot_waves` returns all three, together with the ratios $\mu$ of the
fluid displacement over the frame displacement that say which medium each wave
travels in. The wave whose $|\mu|$ is large is the **airborne** wave, the one an
equivalent fluid already models; the other is the **frame-borne** wave, which
has no equivalent-fluid counterpart at all.

`biot_waves` takes the *rigid-frame* equivalent fluid of the pores, normally a
`johnson_champoux_allard` result. That is deliberate: the frame inertia is the
Biot model's own business, so handing it a `limp_frame` medium would count that
inertia twice.

The consequence a designer cares about is a resonance. A layer glued to a rigid
wall holds its frame still at the wall and free at the front face, so the
frame-borne wave resonates a quarter wavelength inside the layer, at

$$
f_r = \frac{1}{4l}\sqrt{\frac{\mathrm{Re}(K_c)}{\rho_1}}, \qquad
K_c = \frac{2(1-\nu)N}{1-2\nu}
$$

(Eqs. 6.109-6.111), where $K_c$ is the longitudinal elastic coefficient of the
frame **in vacuum** and $\rho_1$ its bulk density. `frame_quarter_wave_resonance`
evaluates it. Nothing in the rigid-frame or limp-frame models can produce that
peak.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/biot_frame_resonance_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/biot_frame_resonance.svg" alt="Normalised surface impedance against frequency from 200 to 1500 hertz for a 100 millimetre glass-wool layer glued to a rigid wall, comparing the rigid-frame equivalent fluid with the full Biot poroelastic layer. The rigid-frame real part falls smoothly from 3 to 1.5 as a dashed line and its imaginary part rises smoothly from minus 3 to minus 0.9 as a dotted line. The Biot real part instead dips to 1.5 near 450 hertz and rebounds to 2.6 near 520 hertz, while the Biot imaginary part peaks sharply at minus 0.7 near 480 hertz, both features straddling the frame quarter-wave resonance of 460 hertz marked by a dotted vertical line, and both curves rejoin the rigid-frame ones above 700 hertz" width="88%"></picture>

*The glass wool of Allard & Atalla Table 6.1, 100 mm glued to a rigid wall,
under both models. Away from the resonance the two are indistinguishable, which
is the plot's own check on the Biot layer; around it the poroelastic prediction
develops the dip-and-peak in the real part and the sharp maximum in the
imaginary part that the book measures and plots in its Fig. 6.10.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import (
    PoroelasticLayer, PorousLayer, biot_waves, frame_elastic_coefficient,
    frame_quarter_wave_resonance, johnson_champoux_allard, layered_absorber,
)

# Allard & Atalla Table 6.1: glass wool "Domisol Coffrage", 100 mm, glued.
f = np.linspace(200.0, 1500.0, 1301)
shear = 2.2e6 * (1 + 0.1j)          # 220 N/cm2, loss factor 0.1
med = johnson_champoux_allard(
    f, 40e3, porosity=0.94, tortuosity=1.06,
    viscous_length=0.56e-4, thermal_length=1.1e-4,
)

print(frame_elastic_coefficient(shear, 0.0))       # (4400000+440000j)
print(round(frame_quarter_wave_resonance(
    0.10, shear_modulus=shear, poisson_ratio=0.0, frame_density=130.0), 1))
# 459.9

waves = biot_waves(med, porosity=0.94, tortuosity=1.06,
                   frame_density=130.0, shear_modulus=shear)
print(round(float(abs(waves.airborne_velocity_ratio[800])), 1))     # 42.4
print(np.round(waves.frame_borne_velocity_ratio[-1], 3))  # (0.811+0.473j)

biot = layered_absorber(f, [PoroelasticLayer(0.10, med, 0.94, 1.06, 130.0, shear)])
rigid = layered_absorber(f, [PorousLayer(0.10, med)])

fig, ax = plt.subplots()
for res, style in ((rigid, "--"), (biot, "-")):
    ax.plot(f, res.normalized_impedance.real, style)
    ax.plot(f, res.normalized_impedance.imag, style)
ax.set(xlabel="Frequency [Hz]", ylabel="Zs / rho0 c0", ylim=(-3, 3))
plt.show()

waves.plot()   # the three Biot wavenumbers against frequency
plt.show()
```

</details>

Inside a stack, a `PoroelasticLayer` carries the same six variables the theory
needs, $[v_1^s, v_3^s, v_3^f, \sigma^s_{33}, \sigma^s_{13}, \sigma^f_{33}]$, so
`layered_absorber` switches from the two-variable chain to the global-matrix
assembly of Allard & Atalla Sect. 11.5: fluid and sheet layers keep their 2x2
matrices, each poroelastic layer enters through the six wave amplitudes of its
own $[\Gamma]$ blocks rather than through a transfer matrix, and the two are
joined by the printed coupling matrices of Sect. 11.4. Solving for the
amplitudes directly avoids inverting $[\Gamma(0)]$, which is what makes a very
soft or very thick frame tractable at all. Two adjacent poroelastic layers
are coupled as **bonded** frames (their Eq. 11.67); a sheet next to one is
coupled as a free, mechanically decoupled screen. The returned
`transfer_matrix` is filled with `nan` for such a stack, because a 2x2 chain
matrix does not exist for it; the surface impedance, reflection factor and
absorption are unaffected.

For the same 100 mm layer the resonance moves absorption by up to 0.21 at its
sharpest; on third-octave centres the largest shift is the 0.12 at 500 Hz:

```python
import numpy as np
from phonometry import (
    PoroelasticLayer, PorousLayer, johnson_champoux_allard, layered_absorber,
)

bands = np.array([250, 315, 400, 500, 630, 800, 1000], dtype=float)
med = johnson_champoux_allard(
    bands, 40e3, porosity=0.94, tortuosity=1.06,
    viscous_length=0.56e-4, thermal_length=1.1e-4,
)
shear = 2.2e6 * (1 + 0.1j)
for layer in (PorousLayer(0.10, med),
              PoroelasticLayer(0.10, med, 0.94, 1.06, 130.0, shear)):
    print(layered_absorber(bands, [layer]).absorption.round(2))
# [0.55 0.58 0.62 0.65 0.69 0.74 0.78]   rigid frame
# [0.53 0.56 0.61 0.77 0.71 0.74 0.78]   Biot poroelastic
```

The layer is worth the extra five parameters when the frame is stiff and heavy
enough to resonate in the band of interest (dense mineral wool, structural
foams), when the material is bonded to a plate that shakes it directly, or when
a measured impedance shows a feature no equivalent fluid can explain. A light,
soft blanket in free air is better served by `limp_frame`.

**Honest note on validation.** There is no published table of $Z_s(f)$ or
$\alpha(f)$ for a fully specified Biot layer, here or anywhere else checked, so
this model cannot be pinned digit by digit the way the standards-based modules
are. What it is anchored on instead, in decreasing strength: the **rigid-frame
limit**, where making the frame infinitely stiff and heavy must reproduce the
JCA equivalent fluid, whose own conformance *is* pinned on published digits, and
does so with a residual that falls exactly as the inverse of the stiffness over
eight decades, at four angles of incidence, with and without a rigid backing;
the **limp limit**, where taking the stiffness to zero reproduces `limp_frame`;
the agreement to machine precision between the **two independent derivations**
the book gives, the ch. 6 closed form Eq. 6.107 and the ch. 11 global-matrix
assembly; and the three computed numbers the book does print in prose for the
Table 6.1 glass wool, all reproduced: the airborne wave changes root at 495 Hz,
$|\mu_a| > 40$ above 50 Hz, and $\mu_b$ falls from 1.0 at 50 Hz to 0.82 at
1500 Hz. That last one is matched by $\mathrm{Re}(\mu_b)$, not by $|\mu_b|$,
even though the printed sentence says "the ratio modulus": the model gives
$\mu_b(1500) = 0.811 + 0.473j$, whose real part is 1.1 % from the printed
value while its modulus, 0.939, is 14.5 % away. That is recorded in
[Errata](/phonometry/reference/errata/), along with the parameter sweep that
fails to bring $|\mu_b|$ anywhere near 0.82. The impedance peak of the book's
thinner sample is printed as 860 Hz and comes out at 863.5 Hz under the
$\mathrm{Im}(Z_s)$ peak rule. Everything else, including the whole
oblique-incidence behaviour beyond its rigid-frame limit, rests on closed forms
and on structural identities, not on published digits.

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

Before the matrices, the physical picture: a hard-backed porous layer is an
equivalent fluid wrapped in five measurable numbers. The 50 mm layer below,
evaluated with the five-parameter JCA model those numbers belong to, absorbs
91 % of a normal-incidence 1 kHz wave; the code that follows runs the same
layer through the one-parameter Miki fit instead, which lands slightly
higher (0.937), a fair picture of the spread between the two models.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_porous_layer_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_porous_layer.svg" alt="Section of a porous absorber: a 50 mm mineral-wool layer against a rigid backing, a normal-incidence plane wave arriving from the left, a small reflected arrow carrying 9 % of the energy and the transmitted wave decaying inside the layer, a magnified circle showing the fibre frame and the air-filled pores, and the JCA parameter set annotated with flow resistivity 20 kPa s/m2, porosity 0.98, tortuosity 1.0 and characteristic lengths of 87 micrometres, giving an absorption coefficient of 0.91 at 1 kHz" width="92%"></picture>

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
[ASTM E2611 machinery](impedance-tube.md) (`TransferMatrix`).

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
[Sound Absorption Measurement and Rating](absorption-measurement.md).*

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

The classical layers above absorb with bulk or with a quarter-wave cavity.
Their resonant metamaterial relatives, rigid panels of sub-millimetre slits
loaded by Helmholtz resonators that slow the sound and reach perfect
absorption ($\alpha = 1$) at critical coupling in panels only $\lambda/38$
deep, moved to their own guide:
[Metamaterial Absorbers](metamaterial-absorbers.md) covers the
transfer-matrix model, the critical-coupling design solver and the FDTD
cross-check of the meshed cell.


## Practical notes

**Fit ranges.** Delany–Bazley warns (and extrapolates) outside
$0.01 < X < 1$ and Miki outside $0.01 < f/\sigma < 1$; treat sub-range
values as qualitative. JCA needs four extra
parameters but behaves physically everywhere; with
$\Lambda = \Lambda' = \sqrt{8\alpha_\infty\eta/(\phi\sigma)}$ and
$\alpha_\infty = 1$ it tracks Delany–Bazley over the fit range.

**Rigid or limp frame.** Every equivalent-fluid model above assumes a
motionless frame, which only holds above the decoupling frequency
$F_d = \sigma\phi^2/2\pi\rho_1$. Below it a light frame moves with the pore
fluid, and `limp_frame` adds its inertia (Allard & Atalla Eqs 11.53-11.55).
Use it for felts, screens and light fibreglass, for anything measured
unconstrained in an impedance tube, and wherever the frame in vacuum is softer
than about 20 kPa (`limp_frame_applicable`); do not use it, or the rigid model
either, for a material bonded to a vibrating structure. When the frame is stiff
and heavy enough to resonate in the band of interest, neither equivalent fluid
will do at all: use a `PoroelasticLayer` and the full Biot theory.

**Local vs. bulk reaction.** The layer solver is bulk-reacting (sound
refracts and travels inside the layers). `statistical_absorption` assumes
local reaction, a good approximation for high flow resistivity, partitioned
cavities or thin resonant facings; for thick, light porous layers integrate
the bulk model with `diffuse_field_absorption` instead (Mechel Sect. D.6).

**Where the numbers were checked.** The models are pinned digit-exact to the
printed coefficient tables (Bies Table D.1, Miki Eqs. 30–34), the solver to
the closed forms above and to the `TransferMatrix` recovery of the
[impedance-tube page](impedance-tube.md), the MPP to Maa's own approximation
(stated ~6 % agreement with the exact Eq. 2), design example and Table I,
and the Paris integral to its locally reacting closed form. Misprints
found in the sources during this work are recorded in the
[errata registry](ERRATA.md).

## References

- Mechel, F. P. (Ed.). (2008). *Formulas of acoustics* (2nd ed.). Springer.
  [doi:10.1007/978-3-540-76833-3](https://doi.org/10.1007/978-3-540-76833-3).
  Sections D.3–D.6 (layer reflection, multilayer scheme, diffuse-field
  integrals) and G.11 (empirical porous relations).
- Allard, J. F., & Atalla, N. (2009). *Propagation of Sound in Porous Media:
  Modelling Sound Absorbing Materials* (2nd ed.). Wiley.
  ISBN 978-0-470-74661-5.
  [doi:10.1002/9780470747339](https://doi.org/10.1002/9780470747339).
  Chapter 6: the Biot theory of an elastic-framed porous material, its elastic
  coefficients, its two compressional waves and its shear wave, and the surface
  impedance and quarter-wave frame resonance of a hard-backed layer
  (Eqs 6.107-6.111). Chapter 11: the transfer-matrix method, the six-variable
  poroelastic layer matrix (Table 11.1), the coupling matrices (Sect. 11.4) and
  the global assembly (Sect. 11.5). Section 11.3.4: the rigid and limp frame
  limits of the Biot theory, the decoupling frequency and the limp effective
  density (Eqs 11.53-11.55, after Panneton 2007).
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

## Standards

No standard governs these prediction models; they are textbook and journal
methods (Mechel; Bies, Hansen & Howard; Cox & D'Antonio; Attenborough & Van
Renterghem; Miki 1990; Maa 1998; Johnson et al. 1987) implemented
clean-room from the cited sources. The measurement standards they connect to
(ISO 9053-1/-2 for [flow resistivity](airflow-resistance.md),
ISO 10534-1/-2 and ASTM E2611 for the [impedance tube](impedance-tube.md),
ISO 354 / ISO 11654 for
[random-incidence absorption and rating](absorption-measurement.md)) live in
their own guides.
