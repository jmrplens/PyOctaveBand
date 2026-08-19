← [Documentation index](../../README.md)

# Bending-wave transmission at plate junctions (Cremer / Craik / Hopkins)

When a bending wave travelling on a wall or floor reaches a rigid junction with
another plate, part of its energy is reflected and part is transmitted into the
connected plates. The **wave approach** of Cremer et al. (1973), tabulated by
Craik (1981, 1996) and collected in Hopkins (2007, *Sound Insulation*,
Section 5.2.1.3), gives the transmission coefficient in closed form for the four
most common junctions of thin, homogeneous, isotropic plates: the **X**, **T**,
**L** and **in-line** junctions. Modelling the junction as a simply supported
(pinned) massless beam forces an incident bending wave to generate only
reflected and transmitted *bending* waves, with no conversion to in-plane
waves, and the resulting coefficients are **independent of frequency**. That is
what makes them convenient closed-form inputs for statistical energy analysis
(SEA) and for the EN 12354 flanking-transmission model, where they feed the
coupling loss factor $\eta_{ij}$ and the vibration reduction index $K_{ij}$.

The object itself first: a T-junction where a 140 mm concrete floor runs into
a continuous 200 mm wall. A `junction_transmission` result retains its plates,
so `res.plot_geometry()` draws the junction to scale with the incident bending
wave marked.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_plate_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_plate_geometry.svg" alt="To-scale cross-section of a T2 plate junction: a 140 mm concrete floor arrives horizontally from the left with the incident-sound arrow on it and ends against the continuous 200 mm wall drawn vertically, both plate thicknesses dimensioned" width="82%"></picture>

*Drawn to scale: the incident bending wave arrives along the 140 mm floor, and
everything the closed-form coefficients describe happens where it meets the
continuous 200 mm wall.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import vibration

# A 140 mm concrete floor meeting a 200 mm wall at a T-junction.
res = vibration.junction_transmission("T2", 0.14, 3500.0, 320.0, 0.2, 3500.0, 460.0)
res.plot_geometry()
plt.show()
```

</details>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_transmission_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_transmission.svg" alt="Transmission coefficient versus incidence angle for a rigid X-junction of a 100 mm and a 200 mm concrete plate, showing the corner coefficient tau12 and the straight-section coefficient tau13 falling from their normal-incidence values to zero at grazing incidence, with their diffuse-field angular averages marked as horizontal lines" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import vibration

# X-junction between a 100 mm and a 200 mm concrete plate (cL = 3200 m/s).
res = vibration.junction_transmission("X", 0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
res.plot()  # tau(theta) for the corner and straight paths, with averages
plt.show()
```

</details>

## 1. Wave parameters χ and ψ (Eqs 5.10, 5.11)

With plate $i$ of thickness $h_i$, quasi-longitudinal wave speed $c_{\mathrm{L},i}$ and
surface density $\rho_{\mathrm{s},i}$, the whole family of coefficients depends on just
two dimensionless ratios (Cremer et al. 1973):

$$
\chi = \frac{k_{\mathrm{B}2}}{k_{\mathrm{B}1}}
     = \left(\frac{\rho_{\mathrm{s}2} B_1}{\rho_{\mathrm{s}1} B_2}\right)^{1/4}
     = \sqrt{\frac{h_1 c_{\mathrm{L}1}}{h_2 c_{\mathrm{L}2}}}
     = \sqrt{\frac{f_{\mathrm{c}2}}{f_{\mathrm{c}1}}}
$$

$$
\psi = \frac{B_2 k_{\mathrm{B}2}^2}{B_1 k_{\mathrm{B}1}^2}
     = \frac{h_2 c_{\mathrm{L}2} \rho_{\mathrm{s}2}}{h_1 c_{\mathrm{L}1} \rho_{\mathrm{s}1}}
     = \frac{\rho_{\mathrm{s}2} f_{\mathrm{c}1}}{\rho_{\mathrm{s}1} f_{\mathrm{c}2}}
$$

$\chi$ is the ratio of the plates' bending wavenumbers (equivalently the square
root of their critical-frequency ratio) and sets the total-internal-reflection
cut-off angle $\theta_\text{co} = \arcsin\chi$; $\psi$ is the ratio of their
bending-moment mobilities. For **identical plates** both are 1.

```python
from phonometry import vibration

chi, psi = vibration.junction_wave_parameters(0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
#  -> (sqrt(0.5), 4.0)
```

## 2. Transmission around a corner and across a straight section

For an incident wave on plate 1, transmission **around the corner** (into the
perpendicular plate 2) is $\tau_{12}(\theta)$ (Eq. 5.12), and transmission
**across the straight section** (into the collinear plate 3, X- and
T-junction (1) only) is $\tau_{13}(\theta)$ (Eq. 5.13):

$$
\tau_{12}(\theta) =
\frac{0.5\,J_1 J_2 \psi \cos\theta \sqrt{\chi^2 - \sin^2\theta}}
     {(J_2\psi)^2 + \chi^2
      + J_2\psi\left(\sqrt{(1+\sin^2\theta)(\chi^2+\sin^2\theta)}
      + \sqrt{(1-\sin^2\theta)(\chi^2-\sin^2\theta)}\right)},
\qquad \chi \ge \sin\theta
$$

$$
\tau_{12}(\theta) = 0, \qquad \chi < \sin\theta
$$

(no propagating transmitted wave beyond the cut-off).

The junction constants $J_1$, $J_2$, $J_3$ select the geometry:

| Junction | $J_1$ | $J_2$ | $J_3$ |
|---|---|---|---|
| X | 1 | 1 | 1 |
| T-junction (1) | 2 | 0.5 | 0.5 |
| T-junction (2) | 2 | 2 | — |
| L | 4 | 1 | — |

The straight section is undefined for the T-junction (2) and the L-junction. In
the assumed symmetry the X-junction has plates 1 and 3 identical and plates 2
and 4 identical; T-junction (1) has plates 1 and 3 identical; T-junction (2) has
plates 2 and 4 identical.

```python
import numpy as np
from phonometry import vibration

theta = np.radians(np.linspace(0.0, 90.0, 91))
tau12 = vibration.corner_transmission_coefficient(theta, chi, psi, "X")
tau13 = vibration.straight_transmission_coefficient(theta, chi, psi, "X")
```

The clip below runs this experiment in the time domain with the library's 2D
[elastic FDTD solver](../../simulation/elastic-waves.md): a bending packet on a 10 mm steel plate reaches an
L-junction with an identical plate, and the corner splits it into the
reflected and transmitted waves this section prices at $\tau_{12}(0°) = 0.5$.
The fast in-plane precursor racing down the receiving plate is the mode
conversion the pinned-junction model deliberately leaves out.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_plate_junction_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_plate_junction.gif" alt="Animation: a 4 kHz bending-wave packet travels along a 10 mm steel plate in a 2D elastic FDTD field; on a straight control plate it runs on and nothing returns, while at an L-junction with an identical perpendicular plate it splits into a reflected packet, a transmitted bending wave descending the vertical plate and a faster in-plane precursor, with the closed-form junction transmission coefficient of 0.50 at normal incidence and the diffuse vibration reduction index of 5.2 dB annotated" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_plate_junction.webm)

## 3. Diffuse-field angular average (Eq. 5.6)

In a diffuse vibration field every angle of incidence is equally probable and
the incident intensity carries a $\cos\theta$ obliquity factor, so the average
transmission coefficient is

$$
\bar{\tau}_{ij} = \int_0^{\pi/2} \tau_{ij}(\theta)\cos\theta\,\mathrm{d}\theta
$$

(the $\cos\theta$ weight already normalises the average). For **identical
plates** the algebra collapses to exact fractions that serve as the library's
first-principles oracle:

* X-junction corner and straight: $\tau_{ij}(\theta) = \cos^2\theta / 8$, so
  $\bar{\tau}_{ij} = 1/12$;
* L-junction corner: $\tau_{ij}(\theta) = \cos^2\theta / 2$, so
  $\bar{\tau}_{ij} = 1/3$;
* in-line junction: $\tau_{12}(0°) = 1$ (a continuous plate transmits fully).

```python
from phonometry import vibration

vibration.angular_average_transmission_coefficient(1.0, 1.0, "X", section="corner")  # 1/12
vibration.angular_average_transmission_coefficient(1.0, 1.0, "L", section="corner")  # 1/3
```

The two directions obey the SEA consistency relationship (Eq. 5.7),
$\bar{\tau}_{12} = \chi\,\bar{\tau}_{21}$, so only one direction needs to be
computed.

## 4. Coupling loss factor and vibration reduction index

The angular average is the bridge to the two junction descriptors used in
SEA-based building models. The **coupling loss factor** (Eq. 2.154) for a
source plate $i$ of area $S_i$, bending-wave group velocity $c_{\mathrm{g},i}$ and
junction length $L_{ij}$ is

$$
\eta_{ij} = \frac{c_{\mathrm{g},i} L_{ij} \tau_{ij}}{2\pi^2 f S_i}
$$

and the wave-approach **vibration reduction index** (Eq. 5.116) is

$$
K_{ij} = 10\log_{10}(1/\tau_{ij}) + 5\log_{10}(f_{\mathrm{c},j}/f_\text{ref}),
\qquad f_\text{ref} = 1000\ \text{Hz}
$$

with $f_{\mathrm{c},j}$ the critical frequency of the *receiving* plate. Combined with
the Eq. 5.7 reciprocity this form is symmetric, $K_{ij} = K_{ji}$, as EN 12354
requires of the junction descriptor. For the identical 100 mm concrete
X-junction ($f_\mathrm{c} \approx 203\ \text{Hz}$),
$K_{ij} = 10\log_{10} 12 + 5\log_{10}(203/1000) \approx 7.3\ \text{dB}$.

```python
from phonometry import vibration

eta = vibration.coupling_loss_factor(1.0 / 12.0, group_velocity=200.0,
                                     junction_length=4.0, frequency=500.0, plate_area=10.0)
res = vibration.junction_transmission("X", 0.1, 3200.0, 240.0, 0.1, 3200.0, 240.0)
kij = vibration.wave_vibration_reduction_index(res.corner_average,
                                               res.critical_frequency2)  # 7.33 dB
kij = res.corner_reduction_index  # the same, precomputed on the result

res.plot()   # tau(theta) for this junction's corner and straight paths (needs matplotlib)
```

$K_{ij}$ is also what ISO 10848 measures on a built junction: excite one
element, read the velocity level difference on both, and normalize by the
junction length, so the closed-form value above has a direct experimental
counterpart.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_junction_rig_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_junction_rig.svg" alt="ISO 10848 junction measurement rigs for an L-junction and a T-junction of 140 to 200 mm concrete plates: a shaker or hammer excites element i, accelerometers on elements i and j read the velocity level difference Dv,ij, and the junction length of at least 2.3 m runs along the highlighted corner line" width="92%"></picture>

The junction descriptor is a *design* quantity: sweeping the receiving
plate's thickness shows how much a mass change at the junction buys. The
corner paths stiffen quickly with a heavier receiving plate, while the
straight (in-line) path of the X-junction rises fastest of all, since the
perpendicular plates increasingly pin the junction line:

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_kij_thickness_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_kij_thickness.svg" alt="Wave-approach vibration reduction index Kij versus the thickness ratio of two concrete plates for the X-junction corner and straight paths, the T-junction corner and the L-junction corner, with the identical-plates X-junction value of about 7.3 dB marked" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import vibration

# Concrete plates (cL = 3200 m/s, rho = 2400 kg/m3): plate 1 fixed at
# 100 mm, plate 2 swept from 50 mm to 400 mm.
h1, cl, rho = 0.1, 3200.0, 2400.0
ratios = np.linspace(0.5, 4.0, 36)
kij_corner = []
for ratio in ratios:
    h2 = h1 * float(ratio)
    res = vibration.junction_transmission("X", h1, cl, rho * h1, h2, cl, rho * h2)
    kij_corner.append(res.corner_reduction_index)

fig, ax = plt.subplots()
ax.plot(ratios, kij_corner, label="X corner")
ax.set_xlabel("Thickness ratio h2/h1")
ax.set_ylabel("Vibration reduction index Kij [dB]")
ax.set_title("Wave-approach junction Kij (Hopkins Eq. 5.116)")
ax.legend()
plt.show()
```

</details>

## 5. Worked example: feeding Kij into EN 12354

The number this page predicts is exactly what the EN 12354-1 flanking model
consumes. Take the 100 mm / 200 mm concrete X-junction from the figure at
the top: its corner path gives $K_{12} = 9.8\ \text{dB}$. Handing that to
`flanking_element` in place of a tabulated Annex E value prices the
junction's three flanking paths and their effect on the apparent rating:

```python
from phonometry import building, vibration

# The 100 mm / 200 mm concrete X-junction of the opening figure:
res = vibration.junction_transmission("X", 0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
k12 = res.corner_reduction_index                     # 9.8 dB (corner path)

# Feed it to the EN 12354-1 simplified model as this junction's Kij:
ff, df, fd = building.flanking_element(
    label="floor", r_flanking=49.0, r_separating=57.0,
    k_ff=k12, k_fd=k12, k_df=k12, separating_area=11.5, coupling_length=4.5)
pred = building.predicted_airborne_insulation(r_direct=57.0,
                                              flanking_paths=[ff, df, fd])
print(round(pred.r_prime_w, 1))                      # 55.4  (Rw 57 direct)
print(pred.dominant.label, round(pred.dominant.fraction, 2))   # Dd 0.68
```

One junction with a moderate $K_{ij}$ already trims 1.6 dB off the direct
$R_\mathrm{w} = 57\ \text{dB}$; a full building repeats this for every junction, which
is the [EN 12354 prediction guide](../../buildings/design/insulation-prediction.md).

The measured, EN 12354 counterpart of $K_{ij}$ (from the direction-averaged
velocity level difference) is the separate
[laboratory flanking-transmission](../../buildings/insulation/flanking-lab.md) `vibration_reduction_index`; this
guide is the closed-form *predicted* value from the wave approach.

## 6. Experimental SEA: coupling loss factors from measured energies

Everything above is the **predictive** route: a coupling loss factor derived
from a wave model of the junction. Real joints - welds, bolt rows, spot welds,
adhesives - are not tractable that way, and the **experimental** route inverts
the steady-state SEA power balance from *measured* subsystem energies instead
(Norton & Karczub 2003, Sections 6.3.3 and 6.3.4). For two subsystems,

$$
\Pi_1 = \omega\left[(\eta_1 + \eta_{12})\,E_1 - \eta_{21}\,E_2\right]
\quad (6.10)
$$

$$
0 = \omega\left[(\eta_2 + \eta_{21})\,E_2 - \eta_{12}\,E_1\right]
\quad (6.11)
$$

with $E_i = M_i \langle v_i^2 \rangle$ the band energy of subsystem $i$ (its
mass times the space- and time-averaged mean-square velocity). Drive
subsystem 1 only, add the SEA consistency (reciprocity) relationship
$n_1\eta_{12} = n_2\eta_{21}$ (Eq. 6.8), and both coupling loss factors follow
from the two measured energies (Eq. 6.15):

$$
\eta_{12} = \frac{\eta_2 E_2}{E_1 - E_2\,n_1/n_2}, \qquad
\eta_{21} = \eta_{12}\,\frac{n_1}{n_2}, \qquad
\Pi_\text{in} = \omega\,(\eta_1 E_1 + \eta_2 E_2)
$$

The input power collapses to the total dissipated power, as it must in the
steady state: substituting Eq. (6.11) into Eq. (6.10) cancels the two coupling
terms exactly, which is a free check on any measurement.

`power_injection_clf` performs that inversion, with the modal densities of
Norton Section 6.4.1 alongside it: `flat_plate_modal_density` (Eq. 6.25),
`bar_modal_density` (6.23), `beam_modal_density` (6.24) and
`cylindrical_shell_modal_density` (6.27 to 6.29, the Szechenyi approximations
in three regimes about the `ring_frequency` of Eq. 6.26). The flat-plate
expression $n(f) = S\sqrt{12}/(2 c_\mathrm{L} t)$ is the same quantity as EN 12354-4's
$n = \pi S f_\mathrm{c} / c_0^2$ used by the [flanking model](../../buildings/insulation/flanking-lab.md), only
parametrised by the plate itself rather than by its critical frequency.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/experimental_sea_clf_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/experimental_sea_clf.svg" alt="Two-panel figure. Left panel: coupling loss factor on a logarithmic axis against the octave bands from 125 hertz to 2 kilohertz for two aluminium plates at right angles, with the welded line junction falling from about 3 times 10 to the minus 3 to 8 times 10 to the minus 4 as one over the square root of frequency, the twelve-bolt point connection falling twice as steeply from about 1.4 times 10 to the minus 2 to 9 times 10 to the minus 4 as one over frequency, and a dashed horizontal line marking an internal loss factor of 10 to the minus 2. Right panel: a bar chart of the four loss factors of a satellite platform and cylinder in the 500 hertz octave on a logarithmic axis, the internal loss factors 4.40 times 10 to the minus 3 and 2.40 times 10 to the minus 3 clearly above the coupling loss factors 4.26 times 10 to the minus 4 and 3.91 times 10 to the minus 4, with the injected power of 1.31 watts annotated" width="92%"></picture>

*Left: two ways of joining the same two plates. Right: the loss-factor budget
inverted from a pair of measured velocities; the coupling stays an order of
magnitude below the damping, which is the condition for a two-subsystem SEA
model to be trustworthy.*

<details>
<summary>Show the code for this figure</summary>

```python
import math
import matplotlib.pyplot as plt
import numpy as np
from phonometry import vibration
from phonometry.vibration.structural.point_mobility import plate_bending_wave_speed

rho, nu, young = 2700.0, 0.33, 7.1e10                 # aluminium
cl = math.sqrt(young / (rho * (1.0 - nu**2)))         # 5432 m/s (Eq. 6.25)
bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0])

# Predicted: a 3 mm x 2.5 m x 1.2 m plate meeting a 5.5 mm x 2.0 m x 1.2 m
# plate at right angles along the 1.2 m edge, welded and then bolted.
h1, h2, area1, length = 0.003, 0.0055, 2.5 * 1.2, 1.2
tau = vibration.right_angle_transmission_coefficient(h1, h2, density1=rho, density2=rho,
                                                     wave_speed1=cl, wave_speed2=cl)
cb = plate_bending_wave_speed(bands, vibration.plate_bending_stiffness(young, h1, nu),
                              rho * h1)
welded = [float(vibration.coupling_loss_factor(tau, 2.0 * c, length, f, area1))
          for c, f in zip(cb, bands, strict=True)]
bolted = vibration.point_connection_coupling_loss_factor(
    bands, 12, thickness1=h1, thickness2=h2, surface_density1=rho * h1,
    surface_density2=rho * h2, wave_speed1=cl, wave_speed2=cl,
    plate_area1=area1)

# Measured: a 5 mm platform driven directly, a 3 mm cylinder driven only
# through the joints; 27.2 and 13.2 mm/s in the 500 Hz octave.
t_p, t_c, radius = 0.005, 0.003, 0.75
area_c = 2.0 * math.pi * radius * 2.0
area_p = 3.5 * 3.0 - math.pi * radius**2
sea = vibration.power_injection_clf(
    500.0, rho * t_p * area_p * 0.0272**2, rho * t_c * area_c * 0.0132**2,
    4.4e-3, 2.4e-3,
    vibration.flat_plate_modal_density(area_p, t_p, cl),
    float(vibration.cylindrical_shell_modal_density(500.0, area_c, t_c, radius, cl)[0]))

print(f"{float(sea.coupling_loss_factor12[0]):.2e}")   # 4.26e-04
print(f"{float(sea.coupling_loss_factor21[0]):.2e}")   # 3.91e-04
print(round(float(sea.input_power[0]), 2))             # 1.31 W
sea.plot()
plt.show()
```

</details>

Two closed forms sit between the two routes and are what a measured value is
normally compared against.
`right_angle_transmission_coefficient` gives the wave transmission coefficient
of a right-angle plate junction without any angular integration (Eqs. 6.53 to
6.55, after Bies & Hamid and Cremer et al.); feeding it to
`coupling_loss_factor` reproduces Norton Eq. (6.52) identically, because that
equation and Hopkins Eq. (2.154) are the same expression once $c_\mathrm{g} = 2 c_\mathrm{B}$.
And `point_connection_coupling_loss_factor` covers plates joined at $N$
discrete points instead of along a line (Eq. 6.56). Which one applies is
decided by the bending wavelength: use the point form when it is shorter than
the joint length, and the line form when it is longer. The two differ in slope
as well as in level, $1/f$ against $1/\sqrt{f}$, so bolting and welding are
not interchangeable across the spectrum.

When no independent value of the internal loss factors is available, the
single-drive inversion is underdetermined, and the classical **power-injection
method** drives each subsystem in turn while measuring both energies each time.
That gives four equations for $\eta_1$, $\eta_2$, $\eta_{12}$ and $\eta_{21}$
with no prior assumption at all, which `power_injection_matrix` solves band by
band. Reciprocity then becomes a *check* on the measurement rather than an
input: `modal_density_ratio` compares the measured $\eta_{21}/\eta_{12}$
against the $n_1/n_2$ computed from the geometry, and a large disagreement
means the subsystem boundaries were drawn in the wrong place.

## What this guide covers

**Covered.** The frequency-independent, rigid-junction transmission
coefficients of Cremer, Heckl & Ungar (1973), tabulated by Craik (1981/1996)
and collected in Hopkins (2007, Section 5.2.1.3), for X, T, L and in-line
junctions of thin, homogeneous, isotropic plates: the wave parameters
$\chi$/$\psi$, the corner and straight-section coefficients
$\tau_{12}(\theta)$/$\tau_{13}(\theta)$, their diffuse-field angular average,
the SEA coupling loss factor and the wave-approach vibration reduction index
$K_{ij}$, through `junction_wave_parameters`,
`corner_transmission_coefficient`, `straight_transmission_coefficient`,
`angular_average_transmission_coefficient`, `inline_transmission_coefficient`,
`coupling_loss_factor`, `wave_vibration_reduction_index` and
`junction_transmission`. Also the experimental route of Norton & Karczub
Chapter 6: the two-subsystem power balance and its in-situ inversion
(`power_injection_clf`, `power_injection_matrix`), the modal densities of bars,
beams, flat plates and thin-walled cylindrical shells with the `ring_frequency`
they are split about, and the two closed forms between the routes
(`right_angle_transmission_coefficient`,
`point_connection_coupling_loss_factor`).

**Not covered.** The predicted $K_{ij}$ is a closed-form idealisation of a
rigid, simply supported junction, not a measurement: the empirical $K_{ij}$
from a direction-averaged velocity level difference is ISO 10848's
`vibration_reduction_index`, in
[Laboratory Flanking Transmission](../../buildings/insulation/flanking-lab.md).
The straight-section coefficient $\tau_{13}$ is undefined for the T-junction
(2) and L-junction geometries, which have no collinear third plate, so only the
corner path applies there. The tabulated coefficients assume a *symmetric*
junction — opposite plates identical — and there is no closed form here for one
that is not. On the experimental side the inversion is written for **two**
subsystems only, and `PowerInjectionResult.modal_density_ratio` is a
consistency check the reader makes, not one the library automates.

## See also

- [Predicting Sound Insulation (EN 12354)](../../buildings/design/insulation-prediction.md): the
  flanking model that consumes $K_{ij}$ junction by junction.
- [Laboratory Flanking Transmission (ISO 10848)](../../buildings/insulation/flanking-lab.md): the
  measurement of $K_{ij}$ from velocity level differences, the empirical
  counterpart of this page's closed forms.
- [Structure-borne sound power of equipment (EN 15657)](../../buildings/design/structure-borne-power.md):
  the source power that these junction transmissions carry through a building.
- [Mechanical mobility and the FRF family (ISO 7626-1)](mechanical-mobility.md):
  the plate mobilities behind the wave parameters.

## References

- Cremer, L., Heckl, M., & Ungar, E. E. (1973). *Structure-borne sound* (1st
  ed.). Springer.
- Craik, R. J. M. (1996). *Sound transmission through buildings using
  statistical energy analysis*. Gower.
- Hopkins, C. (2007). *Sound insulation* (Section 5.2.1.3).
  Butterworth-Heinemann.
- Norton, M. P., & Karczub, D. G. (2003). *Fundamentals of noise and vibration
  analysis for engineers* (2nd ed., Chapter 6). Cambridge University Press.

See the [bibliography](../../reference/bibliography.md) for full entries.
