← [Documentation index](README.md)

# Predicting Panel Sound Insulation (mass law, coincidence, double walls)

A laboratory test measures the sound reduction index $R$ of a finished element;
this page instead **predicts** $R(f)$ from the physical properties of the
construction, so a partition can be designed before it is built. It covers the
airborne insulation of a single panel (the mass law and the coincidence dip),
the double wall (its mass-spring-mass resonance), the transmission through slits
and apertures that caps any real construction, the radiation efficiency of a
bending plate, and the point mobilities that set the vibrational power a
structure absorbs. The measured counterparts these predictions feed live in
[Predicting Sound Insulation (EN 12354)](insulation-prediction.md) and
[Field Insulation Measurement and Ratings](insulation-field.md).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/panel_insulation_concept_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/panel_insulation_concept.svg" alt="Four panels: the single-panel mass law with its coincidence dip, the double wall with the mass-spring-mass resonance and cavity gain, the plate radiation efficiency rising to unity above the critical frequency, and a composite wall whose 1 % open slit caps R at the open-area limit" width="92%"></picture>

## Single panel: the mass law and coincidence (Bies 7.2)

A limp, non-stiff panel transmits sound by being driven bodily by the incident
pressure. The **normal-incidence mass law** (Bies Eq. 7.40) and its diffuse-field
form (Eq. 7.42) are

$$
TL_0 = 10 \log_{10}\!\left[ 1 + \left(\frac{\pi f\, m''}{\rho_0 c_0}\right)^2 \right],
\qquad TL = TL_0 - \Delta_{\text{band}},
$$

with $m''$ the mass per unit area and $\Delta_{\text{band}} = 5.5$ dB (one-third
octave) or $4.0$ dB (octave). The mass law rises **6 dB per octave and 6 dB per
doubling of mass**. At the **coincidence (critical) frequency** (Bies Eq. 7.3)

$$
f_c = \frac{c_0^2}{2\pi}\sqrt{\frac{m''}{B'}} = \frac{0.55\, c_0^2}{c_L\, h},
$$

the free bending wavelength matches the trace wavelength and the panel goes
transparent: the **coincidence dip**. Sharp's method holds the field-incidence
mass law up to $f_c/2$, drops through a straight line in $\log f$, and from $f_c$
upward follows Eq. 7.44 with the loss factor $\eta$; the dip sits at Bies
design-chart point B, $TL = 20\lg(f_c m'') + 10\lg\eta - 44$.

```python
import numpy as np
from phonometry import (
    coincidence_frequency, plate_bending_stiffness,
    single_panel_transmission_loss,
)

# 6 mm float glass: E = 62 GPa, rho = 2500 kg/m3, nu = 0.24, eta = 0.024.
bands = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
mass = 2500.0 * 0.006                                 # 15 kg/m2
bp = plate_bending_stiffness(6.2e10, 0.006, 0.24)     # B' [N.m]
fc = coincidence_frequency(mass, bp)
print(round(fc))                                      # 2107 Hz (Hopkins declares ~2079)

res = single_panel_transmission_loss(bands, mass, critical_frequency=fc,
                                     loss_factor=0.024)
print(round(res.rating().rating))                     # 32  ->  Rw = 32 dB (catalogue 6 mm glass)

res.plot()   # predicted R(f) with the critical frequency marked (needs matplotlib)
```

The predicted spectrum plugs straight into the ISO 717-1 rating through
`res.rating()`, and into EN 12354 as the "predicted" element $R$.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/single_panel_rating_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/single_panel_rating.svg" alt="Predicted sound reduction index of a 6 mm float glass pane per one-third-octave band against the shifted ISO 717-1 reference curve, with the coincidence dip at about 2100 Hz marked, the unfavourable deviations shaded and the Rw rating annotated" width="80%"></picture>

*The predicted Sharp spectrum rated exactly like a measurement: the
coincidence dip at `fc ~ 2.1 kHz` collects most of the unfavourable
deviations, and the shifted reference read at 500 Hz gives the catalogue
`Rw = 32 dB` of 6 mm glass.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import (
    coincidence_frequency, plate_bending_stiffness,
    single_panel_transmission_loss,
)

# 6 mm float glass: E = 62 GPa, rho = 2500 kg/m3, nu = 0.24, eta = 0.024.
bands = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
mass = 2500.0 * 0.006
bp = plate_bending_stiffness(6.2e10, 0.006, 0.24)
fc = coincidence_frequency(mass, bp)
res = single_panel_transmission_loss(bands, mass, critical_frequency=fc,
                                     loss_factor=0.024)
w = res.rating()

# One line each — the predicted R(f), or the rated curve vs the reference:
res.plot()
w.plot()
plt.show()

# By hand, combining both on one axes:
fig, ax = plt.subplots()
ax.semilogx(bands, res.transmission_loss, "o-", label="predicted R (Sharp)")
ax.semilogx(w.band_centers, w.shifted_reference, "s--",
            label="shifted reference")
ax.fill_between(w.band_centers, w.measured, w.shifted_reference,
                where=w.measured < w.shifted_reference, interpolate=True,
                alpha=0.3, label="unfavourable deviations")
ax.axvline(fc, ls=":", color="tab:green", label=f"fc = {fc:.0f} Hz")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Sound reduction index R [dB]")
ax.set_title(f"Rw = {w.rating} dB  (C={w.c:+d}; Ctr={w.ctr:+d})")
ax.legend()
plt.show()
```

</details>

## Double wall: the mass-spring-mass resonance (Bies 7.2.6)

Two leaves separated by a cavity behave as a mass-spring-mass system. Below its
**resonance** (Bies Eq. 7.62, Hopkins Eq. 4.73)

$$
f_0 = \frac{1}{2\pi}\sqrt{\frac{s''\,(m_1 + m_2)}{m_1\, m_2}}
    = 60\sqrt{\frac{m_1 + m_2}{m_1\, m_2\, d}}\quad\text{(empty air gap)},
$$

the pair follows the mass law of the *combined* mass; above it the two mass laws
add, boosted by the cavity, until $f_l = c_0/(2\pi d)$ where the boost saturates
at 6 dB (Eq. 7.64). A porous fill lowers $f_0$.

```python
from phonometry import double_wall_transmission_loss, mass_spring_mass_resonance
from phonometry import mass_law_transmission_loss, miki

# Two 12 kg/m2 leaves, 75 mm air gap.
f0 = mass_spring_mass_resonance(12.0, 12.0, 0.075)
print(round(f0))                                          # 89 Hz

# Below f0 the double wall equals the mass law of the total mass 24 kg/m2:
dw = double_wall_transmission_loss(bands, 12.0, 12.0, 0.075)
print(round(float(dw.transmission_loss[0]), 1),
      round(float(mass_law_transmission_loss(bands[0], 24.0)), 1))   # equal

# A mineral-wool fill (a materials porous model) lowers the resonance:
fill = miki([f0], 7000.0)
print(round(mass_spring_mass_resonance(12.0, 12.0, 0.075, cavity_medium=fill)))  # < 89 Hz

dw.plot()   # double-wall R(f) with the mass-spring-mass resonance marked (needs matplotlib)
```

## Slits, holes and apertures (Hopkins 4.3.10)

A small air path is the real limit on any heavy construction. The transmission
coefficient of a straight slit (Gomperts, Hopkins Eq. 4.99) and of a circular
hole (Wilson & Soroka, Eq. 4.102) are predicted directly, with the slit's
resonances at $d + 2e = z\lambda/2$ (Eq. 4.101). They combine with the wall in
the area-weighted energy sum (Eq. 4.92)

$$
R = -10\log_{10}\!\left( \frac{1}{\sum_n S_n} \sum_n S_n\, 10^{-R_n/10} \right),
$$

so a bare opening of relative area $S_a/S$ caps the composite at $10\lg(S/S_a)$:
a 1 % opening can never do better than 20 dB, whatever the wall.

```python
from phonometry import (
    composite_transmission_loss, slit_transmission_coefficient,
    slit_resonance_frequencies,
)

# A 2 mm x 100 mm-deep slit: transmission peaks at the depth's half-wavelength
# resonances.
print(slit_resonance_frequencies(0.1, 0.002, orders=2).round().tolist())   # [~1500, ~3100]

# A wall of Rw = 50 dB with 1 % of its area open as a slit is capped:
print(round(float(composite_transmission_loss([0.99, 0.01], [50.0, 0.0])), 1))   # 20.0
```

## Radiation efficiency of a bending plate (Hopkins 2.9)

How much airborne power a vibrating plate radiates per unit mean-square velocity
is its **radiation efficiency** $\sigma$, the radiation factor $\varepsilon$
that [Sound Power from Surface Vibration](vibration-sound-power.md) (ISO 7849)
otherwise takes as a measured input. Below the critical frequency the plate
radiates weakly; above it $\sigma \to 1$ (Leppington/Maidanik, Eqs 2.227-2.230):
$\sigma = (1 - f_c/f)^{-1/2}$ for $f > f_c$.

```python
from phonometry import radiation_efficiency, sound_power_from_vibration

# The 6 mm glass pane (1.5 x 1.25 m) of the single-panel example above.
sig = radiation_efficiency(bands, 1.5, 1.25, fc)
print(sig.radiation_efficiency[bands == 2000].round(2))    # ~2.5 (peak at coincidence)

# Feed the prediction straight into ISO 7849 as the radiation factor:
lw = sound_power_from_vibration(velocity_level=80.0, area=1.875,
                                radiation_factor=sig.radiation_efficiency,
                                frequencies=bands)

sig.plot()   # sigma(f) with the coincidence peak (needs matplotlib)
```

## Point mobilities of infinite structures (Cremer Table 5.1)

The vibrational power a point force injects is $W = \tfrac12 |F|^2\,\mathrm{Re}\{Y\}$
(Cremer Eq. 5.23), so the driving-point **mobility** $Y$ (the reciprocal of the
impedance) sets how much energy a structure absorbs. An infinite thin plate is a
pure resistance $Z = 8\sqrt{B'\,m''}$ (real, frequency independent); an infinite
beam has $Y = (1-\mathrm{j})/(4 m' c_B)$ (45 degrees, falling as
$\omega^{-1/2}$). They supply the receiver mobility EN 12354-5 needs when no
measurement exists.

```python
from phonometry import infinite_plate_impedance, infinite_beam_mobility, injected_power

z_plate = infinite_plate_impedance(bp, mass)          # Z = 8 sqrt(B' m'') [N.s/m]
print(round(z_plate))                                 # real, frequency independent
w = injected_power(force=10.0, mobility=1.0 / z_plate)
print(round(float(w) * 1e3, 3), "mW")                 # W = |F|^2 / (16 sqrt(B' m''))
```

## References

- Bies, D. A., Hansen, C. H., & Howard, C. Q. (2017). *Engineering Noise
  Control* (5th ed.). CRC Press. ISBN 978-1-4987-2405-0.
  [doi:10.1201/9781351228152](https://doi.org/10.1201/9781351228152).
  Section 7.2: the mass law, coincidence frequency, Sharp's single-panel method
  and the double-wall model.
- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  Section 2.9 (plate radiation efficiency) and Section 4.3.10 (slits, holes and
  apertures, composite transmission).
- Cremer, L., Heckl, M., & Petersson, B. A. T. (2005). *Structure-Borne Sound*
  (3rd ed.). Springer. ISBN 978-3-540-22696-3.
  [doi:10.1007/b137728](https://doi.org/10.1007/b137728).
  Chapter 5, Table 5.1: the point impedances and mobilities of infinite
  structures and the injected-power relation.

## See also

- [Predicting Sound Insulation (EN 12354)](insulation-prediction.md): assembles
  predicted or measured element $R$ into the in-situ $R'_w$.
- [Sound Power from Surface Vibration (ISO 7849)](vibration-sound-power.md):
  consumes the predicted radiation efficiency as its radiation factor.
- [Mechanical mobility and the FRF family (ISO 7626-1)](mechanical-mobility.md):
  the measured counterpart of the theoretical point mobilities.
- [Porous and multilayer absorbers](porous-absorbers.md): the cavity-fill models
  a double wall consumes.
- API reference: [`building.panel_transmission`](https://jmrplens.github.io/phonometry/reference/api/building/panel-transmission/), [`building.aperture_transmission`](https://jmrplens.github.io/phonometry/reference/api/building/aperture-transmission/), [`vibration.radiation_efficiency`](https://jmrplens.github.io/phonometry/reference/api/vibration/radiation-efficiency/) and [`vibration.point_mobility`](https://jmrplens.github.io/phonometry/reference/api/vibration/point-mobility/).
