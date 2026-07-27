← [Documentation index](README.md)

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
coupling loss factor `ηij` and the vibration reduction index `Kij`.

The object itself first: a T-junction where a 140 mm concrete floor runs into
a continuous 200 mm wall. A `junction_transmission` result retains its plates,
so `res.plot_geometry()` draws the junction to scale with the incident bending
wave marked.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_plate_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_plate_geometry.svg" alt="To-scale cross-section of a T1 plate junction: a 140 mm concrete floor arrives horizontally from the left with the incident-sound arrow on it and ends against the continuous 200 mm wall drawn vertically, both plate thicknesses dimensioned" width="82%"></picture>

*Drawn to scale: the incident bending wave arrives along the 140 mm floor, and
everything the closed-form coefficients describe happens where it meets the
continuous 200 mm wall.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import junction_transmission

# A 140 mm concrete floor meeting a 200 mm wall at a T-junction.
res = junction_transmission("T2", 0.14, 3500.0, 320.0, 0.2, 3500.0, 460.0)
res.plot_geometry()
plt.show()
```

</details>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_transmission_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/junction_transmission.svg" alt="Transmission coefficient versus incidence angle for a rigid X-junction of a 100 mm and a 200 mm concrete plate, showing the corner coefficient tau12 and the straight-section coefficient tau13 falling from their normal-incidence values to zero at grazing incidence, with their diffuse-field angular averages marked as horizontal lines" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import junction_transmission

# X-junction between a 100 mm and a 200 mm concrete plate (cL = 3200 m/s).
res = junction_transmission("X", 0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
res.plot()  # tau(theta) for the corner and straight paths, with averages
plt.show()
```

</details>

## 1. Wave parameters χ and ψ (Eqs 5.10, 5.11)

With plate `i` of thickness `h_i`, quasi-longitudinal wave speed `cL_i` and
surface density `ρs_i`, the whole family of coefficients depends on just two
dimensionless ratios (Cremer et al. 1973):

```
χ = kB2 / kB1 = (ρs2 B1 / (ρs1 B2))**0.25 = sqrt(h1 cL1 / (h2 cL2)) = sqrt(fc2 / fc1)
ψ = B2 kB2**2 / (B1 kB1**2) = (h2 cL2 ρs2) / (h1 cL1 ρs1) = (ρs2 fc1) / (ρs1 fc2)
```

`χ` is the ratio of the plates' bending wavenumbers (equivalently the square
root of their critical-frequency ratio) and sets the total-internal-reflection
cut-off angle `θco = arcsin(χ)`; `ψ` is the ratio of their bending-moment
mobilities. For **identical plates** both are 1.

```python
from phonometry import junction_wave_parameters

chi, psi = junction_wave_parameters(0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
#  -> (sqrt(0.5), 4.0)
```

## 2. Transmission around a corner and across a straight section

For an incident wave on plate 1, transmission **around the corner** (into the
perpendicular plate 2) is `τ12(θ)` (Eq. 5.12), and transmission **across the
straight section** (into the collinear plate 3, X- and T-junction (1) only) is
`τ13(θ)` (Eq. 5.13):

```
              0.5 J1 J2 ψ cos θ sqrt(χ² − sin²θ)
τ12(θ) = ─────────────────────────────────────────────────── ,   χ ≥ sin θ
          (J2 ψ)² + χ² + J2 ψ ( sqrt((1+sin²θ)(χ²+sin²θ))
                              + sqrt((1−sin²θ)(χ²−sin²θ)) )

τ12(θ) = 0 ,   χ < sin θ   (no propagating transmitted wave beyond the cut-off)
```

The junction constants `J1`, `J2`, `J3` select the geometry:

| Junction | `J1` | `J2` | `J3` |
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
from phonometry import (
    corner_transmission_coefficient,
    straight_transmission_coefficient,
)

theta = np.radians(np.linspace(0.0, 90.0, 91))
tau12 = corner_transmission_coefficient(theta, chi, psi, "X")
tau13 = straight_transmission_coefficient(theta, chi, psi, "X")
```

## 3. Diffuse-field angular average (Eq. 5.6)

In a diffuse vibration field every angle of incidence is equally probable and
the incident intensity carries a `cos θ` obliquity factor, so the average
transmission coefficient is

```
τ̄ij = ∫₀^{π/2} τij(θ) cos θ dθ
```

(the `cos θ` weight already normalises the average). For **identical plates**
the algebra collapses to exact fractions that serve as the library's
first-principles oracle:

* X-junction corner and straight: `τij(θ) = cos²θ / 8`, so `τ̄ij = 1/12`;
* L-junction corner: `τij(θ) = cos²θ / 2`, so `τ̄ij = 1/3`;
* in-line junction: `τ12(0°) = 1` (a continuous plate transmits fully).

```python
from phonometry import angular_average_transmission_coefficient

angular_average_transmission_coefficient(1.0, 1.0, "X", section="corner")  # 1/12
angular_average_transmission_coefficient(1.0, 1.0, "L", section="corner")  # 1/3
```

The two directions obey the SEA consistency relationship (Eq. 5.7),
`τ̄12 = χ · τ̄21`, so only one direction needs to be computed.

## 4. Coupling loss factor and vibration reduction index

The angular average is the bridge to the two junction descriptors used in
SEA-based building models. The **coupling loss factor** (Eq. 2.154) for a source
plate `i` of area `S_i`, bending-wave group velocity `cg_i` and junction length
`L_ij` is

```
ηij = cg_i L_ij τij / (2 π² f S_i)
```

and the wave-approach **vibration reduction index** (Eq. 5.116) is

```
Kij = 10 lg(1 / τij) + 5 lg(fc_j / f_ref),   f_ref = 1000 Hz
```

with `fc_j` the critical frequency of the *receiving* plate. Combined with the
Eq. 5.7 reciprocity this form is symmetric, `Kij = Kji`, as EN 12354 requires
of the junction descriptor. For the identical 100 mm concrete X-junction
(`fc ≈ 203 Hz`), `Kij = 10 lg 12 + 5 lg(203 / 1000) ≈ 7.3 dB`.

```python
from phonometry import (coupling_loss_factor, junction_transmission,
                        wave_vibration_reduction_index)

eta = coupling_loss_factor(1.0 / 12.0, group_velocity=200.0,
                           junction_length=4.0, frequency=500.0, plate_area=10.0)
res = junction_transmission("X", 0.1, 3200.0, 240.0, 0.1, 3200.0, 240.0)
kij = wave_vibration_reduction_index(res.corner_average,
                                     res.critical_frequency2)  # 7.33 dB
kij = res.corner_reduction_index  # the same, precomputed on the result

res.plot()   # tau(theta) for this junction's corner and straight paths (needs matplotlib)
```

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
from phonometry import junction_transmission, wave_vibration_reduction_index

# Concrete plates (cL = 3200 m/s, rho = 2400 kg/m3): plate 1 fixed at
# 100 mm, plate 2 swept from 50 mm to 400 mm.
h1, cl, rho = 0.1, 3200.0, 2400.0
ratios = np.linspace(0.5, 4.0, 36)
kij_corner = []
for ratio in ratios:
    h2 = h1 * float(ratio)
    res = junction_transmission("X", h1, cl, rho * h1, h2, cl, rho * h2)
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
the top: its corner path gives `K12 = 9.8` dB. Handing that to
`flanking_element` in place of a tabulated Annex E value prices the
junction's three flanking paths and their effect on the apparent rating:

```python
from phonometry import building, junction_transmission

# The 100 mm / 200 mm concrete X-junction of the opening figure:
res = junction_transmission("X", 0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
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

One junction with a moderate `Kij` already trims 1.6 dB off the direct
`Rw = 57 dB`; a full building repeats this for every junction, which is the
[EN 12354 prediction guide](insulation-prediction.md).

The measured, EN 12354 counterpart of `Kij` (from the direction-averaged
velocity level difference) is the separate
[flanking-transmission](insulation-field.md) `vibration_reduction_index`; this
guide is the closed-form *predicted* value from the wave approach.

## See also

- [Predicting Sound Insulation (EN 12354)](insulation-prediction.md): the
  flanking model that consumes `Kij` junction by junction.
- [Laboratory Insulation Measurement](insulation-lab.md): the ISO 10848
  measurement of `Kij` from velocity level differences, the empirical
  counterpart of this page's closed forms.
- [Structure-borne sound power of equipment (EN 15657)](structure-borne-power.md):
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

See the [bibliography](references.md) for full entries.
