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
[Field Insulation Measurement (ISO 16283)](insulation-field.md).

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

The whole section in one sketch: a diffuse field drives a single leaf, and the
predicted $R(f)$ climbs with the mass law until the coincidence dip. A 12.5 mm
plasterboard leaf puts $f_c$ near 2.6 kHz and rates $R_w = 27$ dB.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_panel_insulation_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_panel_insulation.svg" alt="Section of a sound insulation test: a 12.5 mm plasterboard panel mounted between heavy filler walls, green arrows of diffuse incidence arriving at several angles from the source room, thinner blue transmitted arrows leaving into the receiving room, a bending wave drawn along the leaf, and an inset of the predicted sound reduction index rising 6 dB per octave to a coincidence dip at fc = 2.6 kHz, with the leaf's mass of 8.8 kg per square metre and the Rw = 27 dB rating annotated" width="92%"></picture>

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
coincidence dip at $f_c \approx 2.1$ kHz collects most of the unfavourable
deviations, and the shifted reference read at 500 Hz gives the catalogue
$R_w = 32$ dB of 6 mm glass.*

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

The clip below watches coincidence happen in a 2D
[elastic FDTD field](elastic-waves.md): a 10 mm
steel plate immersed in air, driven by a sustained plane wave at 45 degrees,
with both frequencies picked from the library's `coincidence_frequency`. At
$f_c/2 = 603$ Hz the measured transmission lands on the oblique mass law; at
$2f_c = 2413$ Hz the 45-degree trace equals $\lambda_B$ exactly and the
trace-matched plate re-radiates a 45-degree beam that grows along the plate,
pinning the transmitted level at the $f_c/2$ figure even though the mass law
demands 12 dB
more blocking at four times the frequency. The air drives the heavy steel so
weakly that the resonant bending wave needs tens of metres of illuminated
plate to build up fully, which is why a real coincidence dip, not total
transparency, is what measurements show.

What the plate lets through is about 47 dB under the incident wave, so on the
colour scale the standing wave above the plate sets, the transmitted field
would be a black band. Both panels therefore draw the air below the plate
with the display gain printed on them (x150, that is +44 dB); the gain is the
same on the two so the panels stay comparable, and the level annotations are
the measured, physical ones.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_coincidence_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_coincidence.gif" alt="Animation: two side-by-side 2D elastic FDTD fields of the same 10 mm steel plate lying in air while a sustained plane wave arrives at 45 degrees; at 603 Hz, half the coincidence frequency, the wave reflects almost totally and the faint transmitted level matches the oblique mass law, while at 2413 Hz, twice the coincidence frequency, the trace wavelength matches the free bending wavelength and a 45-degree transmitted beam grows below the plate, holding the same level as the low-frequency panel where the mass law predicted 12 dB more insulation, the air below the plate drawn on both panels with an annotated 150-fold display gain (+44 dB) so the transmitted field is legible, with the 1206 Hz coincidence frequency of the library and both measured and mass-law levels annotated" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_coincidence.webm)

### Quick estimates: the plateau method (Norton 3.9.1)

Before the physical model there is a shortcut that practitioners have drawn by
hand for decades. The **plateau method** (Norton & Karczub 2003, Section 3.9.1,
after Watters) approximates the whole field-incidence curve of a single panel
from three numbers per material, tabulated in `PLATEAU_MATERIALS` from Norton's
Table 3.1: the surface density per millimetre of thickness, the height of the
coincidence plateau in decibels, and the frequency ratio $B/A$ that sets its
width.

The construction has three parts:

1. the **field-incidence mass law**
   $TL = 10\lg(1 + (\pi f m''/\rho_0 c_0)^2) - 5\ \text{dB}$
   (Norton Eqs. 3.104 and 3.106), rising 6 dB per octave;
2. a horizontal **coincidence plateau** at the tabulated height, with point
   **A** where the mass-law line reaches it;
3. point **B** at $B/A \times f_A$, above which the estimate recovers at
   **10 dB per octave**.

It needs neither the bending stiffness nor the loss factor - the tabulated
plateau absorbs both - and it assumes a diffuse field on both sides of a panel
whose length and width are at least twenty times its thickness.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/plateau_transmission_loss_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/plateau_transmission_loss.svg" alt="Transmission loss in decibels against one-third-octave bands from 100 hertz to 10 kilohertz for a 6 mm float glass panel, comparing two curves. Both rise together at 6 decibels per octave from about 15.5 decibels at 100 hertz to 25.5 decibels at 315 hertz. Above that the physical model keeps rising to a peak near 35.5 decibels at 1 kilohertz, dips to about 28.5 decibels at the 2033 hertz critical frequency marked by a dotted vertical line, and then climbs steeply to 49 decibels at 10 kilohertz, while the plateau estimate flattens at the tabulated 27 decibel coincidence height across a shaded band from point A at 374 hertz to point B at 3742 hertz and then recovers at 10 decibels per octave to 41 decibels" width="88%"></picture>

*The two curves are the same line below point A. Above it the plateau estimate
replaces the entire coincidence region with one flat value, which is exactly
what it claims to be: an estimate, not the dip.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import plateau_transmission_loss, single_panel_transmission_loss

# 6 mm float glass: Norton Table 3.1 gives 2.47 kg/m2 per mm, a 27 dB
# coincidence plateau and B/A = 10; its critical frequency is 2033 Hz.
bands = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                  1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                  10000], dtype=float)
quick = plateau_transmission_loss(bands, material="glass", thickness_mm=6.0,
                                  field_correction=5.5)
physical = single_panel_transmission_loss(bands, 2.47 * 6.0,
                                          critical_frequency=2033.0,
                                          loss_factor=0.02)

print(round(quick.plateau_start), round(quick.plateau_end))   # 374 3742
print(quick.plateau_height)                                   # 27.0
quick.plot()          # shades the plateau between points A and B
plt.show()
```

</details>

A worked check against the book: an 8 m x 3 m solid brick wall 110 mm thick at
2.1 kg/m² per mm ($m'' = 231\ \text{kg/m}^2$) gives 35.8 dB in the 63 Hz
octave, on the
mass-law line, then the brick plateau of 37 dB across the 125 Hz and 250 Hz
octaves, then 10 dB per octave above point B - Norton's printed answer to his
problem 3.11 exactly, band for band, in the two regions the construction fixes
analytically.

```python
from phonometry import plateau_transmission_loss

octaves = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
res = plateau_transmission_loss(octaves, material="brick", thickness_mm=110.0,
                                air_density=1.21)
print(res.transmission_loss.round(1))
# [35.8 37.  37.  43.3 53.3 63.3 73.3]
```

The physical model has its own Norton-flavoured variant for the region above
coincidence: `coincidence_model="cremer"` replaces Sharp's Eq. 7.44 with
Cremer's empirical $TL = TL_0 + 10\lg(f/f_c - 1) + 10\lg\eta - 2\ \text{dB}$
(Norton Eq. 3.110), and runs the mass law right up to $f_c$ with no interpolated
bridge. Both rise at about 10 dB per octave far above coincidence; they differ
in how they leave the dip.

## Orthotropic panels: a coincidence range, not a dip (Vigran 6.5.3)

Ribbed and corrugated cladding is not isotropic. It is very stiff *along* the
corrugations and almost as limp as a flat sheet *across* them, so instead of one
coincidence frequency it has a whole **coincidence range** bounded by the two
principal bending stiffnesses (Vigran Eq. 6.107, the isotropic
$f_c = (c_0^2/2\pi)\sqrt{m''/B}$ evaluated twice):

$$
f_{c1} = \frac{c_0^2}{2\pi}\sqrt{\frac{m''}{B_1}}, \qquad
f_{c2} = \frac{c_0^2}{2\pi}\sqrt{\frac{m''}{B_2}}, \qquad B_1 > B_2 .
$$

The stiffest direction sets the *lowest* coincidence frequency, which is the
sting in the tail: corrugating a sheet to gain strength drags $f_{c1}$ down,
often to a few hundred hertz, while $f_{c2}$ runs up to 15 kHz or 30 kHz. Over
that whole span the resonant transmission dominates and $R$ flattens far below
the mass law of a flat plate of the same mass.

The bending-wave impedance now depends on the azimuth $\vartheta$ as well as the
incidence angle $\varphi$ (Heckl 1960; Hansen 1993; Vigran Eq. 6.108, which is
Bies Eq. 7.30):

$$
Z_w = \mathrm{j}\omega m''\left[1 -
\left(\frac{f}{f_{c1}}\cos^2\vartheta + \frac{f}{f_{c2}}\sin^2\vartheta\right)^2
(1 + \mathrm{j}\eta)\sin^4\varphi\right],
$$

and the diffuse-field average of $\tau = |1 + Z_w\cos\varphi/2\rho_0c_0|^{-2}$
becomes a double integral (Vigran Eq. 6.111 = Bies Eq. 7.38). Setting
$f_{c1} = f_{c2}$ recovers Cremer's isotropic impedance exactly.

`orthotropic_transmission_loss` offers both published routes:

- `method="integral"` evaluates that double integral numerically. It is the
  only route that responds to the loss factor, and it is the one that shows how
  deep the coincidence region really goes. The near-grazing angles are excluded
  by a limiting angle: pass `area` for the size-dependent limit of Bies
  Eq. 7.36 (Vigran writes it as Eq. 6.113), or leave it out for the fixed
  `limiting_angle` (78 degrees by default, Sharp's value, which is Vigran's
  $\sin^2\varphi = 0.96$).
- `method="heckl"` is Heckl's closed-form approximation for $\eta = 0$, the
  design chart of Bies Fig. 7.9(b): field-incidence mass law below $f_{c1}/2$,
  then

  $$
  \tau_F \approx \frac{\rho_0c_0}{2\pi^2 m''}\,\frac{f_{c1}}{f^2}
  \left[\ln\frac{4f}{f_{c1}}\right]^2 \quad (f_{c1} \le f < f_{c2}/2),
  \qquad
  \tau_F \approx \frac{\rho_0c_0}{2m''}\,\frac{\sqrt{f_{c1}f_{c2}}}{f^2}
  \quad (f > 2f_{c2}),
  $$

  with straight lines in $\log_{10} f$ across the two gaps. It needs no loss
  factor and no numerical work, but it requires $f_{c2} > 4f_{c1}$ for its four
  construction points to stay in order.

For the common "wavy" corrugation the equivalent stiffnesses come from
Timoshenko and Woinowsky-Krieger through `corrugated_plate_stiffness`
(Vigran Eq. 3.115), and the surface density has to grow with the **developed
length** of the profile, which is what `corrugated_plate_mass_factor` returns.
`orthotropic_plate_resonance` is the matching eigenfrequency of a simply
supported orthotropic plate (Vigran Eq. 3.113 = Bies Eq. 7.27, after Hearmon
1959); its lowest mode matters because both infinite-panel models are only
valid above roughly $1.5f_{1,1}$.

```python
from phonometry import (
    coincidence_frequency, corrugated_plate_mass_factor,
    corrugated_plate_stiffness, orthotropic_critical_frequencies,
    orthotropic_plate_resonance, plate_bending_stiffness,
)

# Vigran's worked example: a 1 m x 1 m steel sheet 1 mm thick, E = 210 GPa,
# nu = 0.3, m'' = 7.8 kg/m2, corrugated into a sinusoid 20 mm deep
# (amplitude H = 10 mm) at a 100 mm pitch.
flat_b = plate_bending_stiffness(2.1e11, 1.0e-3, 0.3)
b_x, b_z, b_xz = corrugated_plate_stiffness(
    1.0e-3, 0.010, 0.100, youngs_modulus=2.1e11, poisson_ratio=0.3,
)
mass = 7.8 * corrugated_plate_mass_factor(0.010, 0.100)
print(round(flat_b, 1), round(b_x, 1), round(b_z))     # 19.2 17.5 2202
print(round(mass, 2))                                  # 8.52 kg/m2 (+9 %)

flat = {"length_x": 1.0, "length_z": 1.0, "mass_per_area": 7.8,
        "bending_stiffness_x": flat_b, "bending_stiffness_z": flat_b,
        "bending_stiffness_xz": flat_b}
corr = {"length_x": 1.0, "length_z": 1.0, "mass_per_area": mass,
        "bending_stiffness_x": b_x, "bending_stiffness_z": b_z,
        "bending_stiffness_xz": b_xz}
print(round(orthotropic_plate_resonance(1, 1, **flat), 1),
      round(orthotropic_plate_resonance(2, 2, **flat), 1))    # 4.9 19.7
print(round(orthotropic_plate_resonance(1, 1, **corr), 1),
      round(orthotropic_plate_resonance(2, 2, **corr), 1))    # 25.5 102.1

# The same sheet, flat and corrugated, in the coincidence picture:
print(round(coincidence_frequency(7.8, flat_b)))               # 11925 Hz
print([round(f) for f in orthotropic_critical_frequencies(mass, b_x, b_z)])
# [1165, 13064]
```

The four eigenfrequencies are Vigran's own printed answers (4.9 Hz, 19.7 Hz
flat; 25.5 Hz, 102 Hz corrugated), and reproducing the corrugated pair is what
proves the mass factor belongs there: with the flat 7.8 kg/m² the same formulas
return 26.7 Hz and 106.7 Hz.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/orthotropic_transmission_loss_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/orthotropic_transmission_loss.svg" alt="Transmission loss in decibels against one-third-octave bands from 100 hertz to 16 kilohertz for a 1 mm steel sheet, flat and corrugated. Both curves rise together at 6 decibels per octave from about 11 decibels at 100 hertz to 28 decibels at 800 hertz. Above 1 kilohertz the flat sheet keeps climbing to a peak near 45 decibels at 6.3 kilohertz before its own coincidence dip, while the corrugated sheet collapses to 22 decibels at 1.6 kilohertz inside a shaded band running from 1165 hertz to 13.1 kilohertz and then recovers slowly, staying about 13 decibels below the flat sheet at 2.5 kilohertz. A dashed line shows Heckl's closed-form approximation tracking the same collapse more smoothly" width="88%"></picture>

*The trade corrugating makes, on Vigran's own geometry. Below $f_{c1}$ the two
panels are within about 2 dB of each other (the corrugated one slightly higher:
it is 9 % heavier, and the diffuse-field integral uses a 78 degree limiting
angle rather than Sharp's 5.5 dB band correction). Across the coincidence range
the corrugated sheet gives away up to 13 dB, and $R_w$ falls from 28 dB to
25 dB, for a panel that is stiffer and barely heavier.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import (
    coincidence_frequency, corrugated_plate_mass_factor,
    corrugated_plate_stiffness, orthotropic_critical_frequencies,
    orthotropic_transmission_loss, plate_bending_stiffness,
    single_panel_transmission_loss,
)

bands = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                  1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                  10000, 12500, 16000], dtype=float)
eta = 0.011
flat_b = plate_bending_stiffness(2.1e11, 1.0e-3, 0.3)
b_x, b_z, _ = corrugated_plate_stiffness(
    1.0e-3, 0.010, 0.100, youngs_modulus=2.1e11, poisson_ratio=0.3,
)
mass = 7.8 * corrugated_plate_mass_factor(0.010, 0.100)
fc1, fc2 = orthotropic_critical_frequencies(mass, b_x, b_z)

flat = single_panel_transmission_loss(
    bands, 7.8, critical_frequency=coincidence_frequency(7.8, flat_b),
    loss_factor=eta,
)
corrugated = orthotropic_transmission_loss(
    bands, mass, critical_frequency_lower=fc1, critical_frequency_upper=fc2,
    loss_factor=eta,
)
heckl = orthotropic_transmission_loss(
    bands, mass, critical_frequency_lower=fc1, critical_frequency_upper=fc2,
    method="heckl",
)

# One line: R(f) with the coincidence range shaded between fc1 and fc2.
corrugated.plot()
plt.show()

# By hand, all three on one axes:
fig, ax = plt.subplots()
ax.axvspan(fc1, fc2, color="0.85", zorder=0)
ax.semilogx(bands, flat.transmission_loss, "-o", ms=4, label="flat sheet")
ax.semilogx(bands, corrugated.transmission_loss, "-s", ms=4,
            label="corrugated, integral")
ax.semilogx(bands, heckl.transmission_loss, "--", label="Heckl approximation")
ax.set(xlabel="Frequency [Hz]", ylabel="Transmission loss TL [dB]")
ax.legend()
plt.show()
```

</details>

Two caveats Bies attaches to the Heckl branch are worth repeating, because no
smooth model predicts either. Below about $0.7f_{c1}$ the estimate
underestimates $R$ on small panels, the error growing as the panel shrinks; and
real corrugated panels almost always show a dip of up to 5 dB somewhere between
2 kHz and 4 kHz, which finite-element work traced to resonances of the panel
sections *between* the ribs rather than to any coincidence effect.

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

The classic lightweight case is worth drawing to scale: two 8.8 kg/m²
plasterboard leaves on a 100 mm empty cavity put $f_0$ at 90 Hz.
`plot_double_wall_geometry` annotates the resonance in the cavity, and a
`double_wall_transmission_loss` result that retained its geometry redraws its
own cross-section with `dw.plot_geometry()`.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/double_wall_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/double_wall_geometry.svg" alt="To-scale cross-section of a lightweight double wall: two grey plasterboard leaves of 8.8 kg/m2 each, drawn 12.6 mm thick, separated by the 100 mm cavity, the incident-sound arrow arriving from the left and the mass-spring-mass resonance f0 = 90 Hz annotated in the cavity" width="80%"></picture>

*The whole model in one section: the two thin leaves are the masses, the
100 mm air gap is the spring, and $f_0 = 90$ Hz is where the dip of every
double-wall curve sits.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import mass_spring_mass_resonance, plot_double_wall_geometry

# Two 8.8 kg/m2 plasterboard leaves on a 100 mm cavity.
f0 = mass_spring_mass_resonance(8.8, 8.8, 0.1)
plot_double_wall_geometry(8.8, 8.8, 0.1, resonance_frequency=f0)
plt.show()

# A double-wall prediction retains its geometry and redraws it:
#   dw = double_wall_transmission_loss(bands, 8.8, 8.8, 0.1)
#   dw.plot_geometry()
```

</details>

## Masonry cavity walls: the wall-tie bridge (Hopkins 4.3.5.4)

The double-wall model above treats the cavity as pure air. A real masonry
cavity wall is stitched together by **wall ties** every few courses, and those
ties do two things the air-only model cannot see: they add a mechanical spring
in parallel with the air spring, which pushes the mass-spring-mass resonance
up, and they open a structure-borne path from one leaf to the other.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/masonry_wall_ties_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/masonry_wall_ties.svg" alt="Two panels: the coupling loss factor of a wall-tie array against frequency for three tie types with the rigid-connection ceiling as a dashed line, and the predicted sound reduction index of a masonry cavity wall with and without ties showing the shaded band between the two mass-spring-mass resonances" width="92%"></picture>

*Left: a soft tie rolls the structure-borne coupling off as $1/f^3$, a stiff
one stays on the rigid ceiling for another two octaves. Right: the same ties
lift the resonance from 26 Hz to 50 Hz, extending the combined-mass branch.*

**The tie as a spring (Hopkins Eq. 4.89).** A tie is characterised by a single
number, its dynamic stiffness $s_{X\,\text{mm}}$ at a cavity width $X$,
measured on two nominally identical 100 mm concrete cubes
(Section 3.11.3.2, Eq. 3.202). $N$ ties over a plate of area $S$ add
$Nk/S$ to the cavity air stiffness:

$$
f_\text{msm} = \frac{1}{2\pi}
\sqrt{\frac{s_a + Nk/S}{\rho_{s1}\rho_{s2}/(\rho_{s1}+\rho_{s2})}}.
$$

Below $f_\text{msm}$ the two leaves act as one plate of the combined mass, so
stiff ties are doubly bad: they raise the resonance into the rating range and
bridge the cavity. `WALL_TIE_STIFFNESS` carries Hopkins' Table A4, whose 50 mm
rows come from Hopkins, Wilson & Craik (1999) and whose 100 mm row from
Hall & Hopkins (2001).

| Wall tie | Cavity width (mm) | $s_{X\,\text{mm}}$ (MN/m) |
|---|---|---|
| Butterfly tie (BS 1243:1978) | 50 | 1.7 |
| Double-triangle tie (BS 1243:1978) | 50 | 16.1 |
| Vertical-twist tie (BS 1243:1978) | 50 | 94.0 |
| Vertical-twist tie (proprietary) | 100 | 43.4 |

```python
from phonometry import (
    double_wall_transmission_loss,
    mass_spring_mass_resonance,
    wall_tie_stiffness,
    wall_tie_stiffness_per_area,
)

print(wall_tie_stiffness("butterfly"))        # (0.05 m, 1.7e6 N/m)

# Hopkins Fig. 4.35: two 140 kg/m2 leaves across an empty 75 mm cavity.
print(round(mass_spring_mass_resonance(140.0, 140.0, 0.075)))          # 26 Hz

# Add 2.5 ties per m2 of s_75mm = 2 MN/m: the resonance nearly doubles.
ties = wall_tie_stiffness_per_area(2.5, 2.0e6)
print(round(mass_spring_mass_resonance(140.0, 140.0, 0.075,
                                       tie_stiffness_per_area=ties)))  # 50 Hz

dw = double_wall_transmission_loss(bands, 140.0, 140.0, 0.075,
                                   tie_stiffness_per_area=ties)
```

**The tie as a point connection (Hopkins Eqs. 4.84 to 4.88).** Each tie
transmits structure-borne power between the leaves. With the driving-point
mobilities $Y_i$, $Y_j$ of the two leaves (infinite thin plates,
$Y = 1/(8\sqrt{B'm''})$, Eq. 2.190) and the connector mobility of a linear
spring $Y_c = \mathrm{i}\omega/k$ (Eq. 4.88), $N$ identical uncorrelated
connections give the coupling loss factor (Eq. 4.87)

$$
\eta_{ij} = \frac{N}{\omega m_i}
\frac{\mathrm{Re}\{Y_j\}}{|Y_i + Y_j + Y_c|^2}.
$$

The plate area cancels ($N/m_i = n/\rho_{s1}$ with $n$ ties per m²), so only
the tie density enters. A rigid connection (a screw, a nail, a bolt, or a tie
stiff enough never to yield) is the limit $Y_c = 0$; a resilient tie rolls the
coupling off two powers faster once $|Y_c| = \omega/k$ overtakes the plate
mobilities: $\eta_{ij}$ then falls as $1/f^3$ against the rigid ceiling's
$1/f$, so the *ratio* between them goes as $1/f^2$. That is exactly why a
butterfly tie at 1.7 MN/m and a vertical-twist tie at 94 MN/m behave so
differently.

```python
import numpy as np
from phonometry import plate_bending_stiffness, wall_tie_coupling_loss_factor

freq = np.logspace(np.log10(50.0), np.log10(5000.0), 60)
b1 = plate_bending_stiffness(2.0e10, 0.1, 0.2)   # 100 mm masonry leaves
res = wall_tie_coupling_loss_factor(
    freq, 150.0, 170.0, b1, b1, ties_per_area=2.5, tie="butterfly"
)
print(res.coupling_loss_factor[0], res.rigid_coupling_loss_factor[0])
res.plot()   # eta_ij against the rigid-connection ceiling (needs matplotlib)
```

The **inputs** of this model are printed data: Table A4, confirmed value for
value by Hopkins, Wilson & Craik (1999) Table 1, which prints the same
1.7 / 16.1 / 94.0 MN/m at a 50 mm cavity. Craik & Wilson (1995) Table 1
measures the same tie *types* at an 85 mm cavity and reports 1.1 and
4.3 MN/m for the butterfly and double-triangle ties, so it corroborates the
ordering but not the values: the dynamic stiffness is defined at a given
cavity width and changes with it. The **output** is not printed anywhere: every published sound reduction
index of a bridged masonry cavity wall is a figure, so the per-band
transmission-loss penalty of the ties has no printed numeric oracle. The
resonance shift does: Hopkins Fig. 4.35 prints 26 Hz without ties and 50 Hz
with them for the same wall.

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

The leak that undoes a heavy wall is almost invisible, and `.plot_geometry()`
makes the point by drawing it to scale: a 2 mm gap through 100 mm of
masonry.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/aperture_slit_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/aperture_slit_geometry.svg" alt="To-scale cross-section of a 2 mm slit through a 100 mm wall: the hatched wall drawn in section with the narrow horizontal air gap at mid-height, an incident-sound arrow pointing at the gap from the left, the 100 mm wall depth and 2 mm slit width dimensioned, and circular transmitted wavefronts sketched spreading from the slit exit on the right" width="80%"></picture>

*The tiny geometry behind a large leak: the gap is 50 times deeper than it
is wide, which is why it behaves as a short tube with half-wavelength depth
resonances near 1.5 and 3.1 kHz rather than as a simple open area.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import slit_transmission_coefficient

f = np.geomspace(100.0, 5000.0, 200)
result = slit_transmission_coefficient(f, 0.002, 0.1)

# One line: the wall section with the slit to scale.
result.plot_geometry()
plt.show()
```

</details>

The clip below puts the two regimes side by side in a 2D FDTD field: a plane
front meets a rigid 0.10 m wall with a 25 mm slit ($\lambda/20$ at 686 Hz)
and with a
0.50 m opening (one wavelength). The narrow slit re-radiates what it swallows
as a cylindrical wave, the Gomperts transmission of the model annotated; the
wavelength-sized opening lets the front through nearly intact and casts
sharp-edged shadows. What the slit passes is about 23 dB under the standing
wave that faces the wall, so in the instantaneous panels the half space
behind the wall rides the display gain each panel prints (x10, that is
+20 dB, for the slit and none for the wavelength-sized opening); the RMS
maps below keep one shared scale, which is where the two openings are
compared level for level.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_aperture_slit_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_aperture_slit.gif" alt="Animation: a 686 Hz plane wave in a 2D FDTD field hits a rigid wall with a 25 mm slit and, in a second panel, a 0.50 m opening; the narrow slit re-radiates a cylindrical wave into the half space behind, drawn with an annotated ten-fold display gain (+20 dB) and with the Gomperts transmission coefficient of 0.55 annotated, while the wavelength-sized opening passes a beam with sharp-edged shadows at unit gain, both RMS maps on the same colour scale" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_aperture_slit.webm)

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

<details>
<summary>Show the code for the concept figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import (
    coincidence_frequency, composite_transmission_loss,
    double_wall_transmission_loss, mass_law_transmission_loss,
    mass_spring_mass_resonance, plate_bending_stiffness,
    radiation_efficiency, single_panel_transmission_loss,
)

bands = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000], dtype=float)
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

bp = plate_bending_stiffness(6.2e10, 0.006, 0.24)
fc = coincidence_frequency(15.0, bp)
ml = mass_law_transmission_loss(bands, 15.0, incidence="field")
sp = single_panel_transmission_loss(bands, 15.0, critical_frequency=fc, loss_factor=0.024)
ax[0, 0].semilogx(bands, ml, "--", label="field-incidence mass law")
ax[0, 0].semilogx(bands, sp.transmission_loss, "-o", ms=3, label="single panel R (Sharp)")
ax[0, 0].axvline(fc, ls=":", color="r"); ax[0, 0].set_title("Single panel")

dw = double_wall_transmission_loss(bands, 12.0, 12.0, 0.075)
ax[0, 1].semilogx(bands, mass_law_transmission_loss(bands, 24.0), "--", label="single leaf")
ax[0, 1].semilogx(bands, dw.transmission_loss, "-o", ms=3, label="double wall")
ax[0, 1].axvline(mass_spring_mass_resonance(12.0, 12.0, 0.075), ls=":", color="r")
ax[0, 1].set_title("Double wall")

sig = radiation_efficiency(bands, 1.5, 1.25, fc)
ax[1, 0].loglog(bands, sig.radiation_efficiency, "-o", ms=3, label=r"$\sigma(f)$")
ax[1, 0].axhline(1.0, ls=":"); ax[1, 0].set_title("Radiation efficiency")

wall = sp.transmission_loss
comp = [float(composite_transmission_loss([0.99, 0.01], [w, 0.0])) for w in wall]
ax[1, 1].semilogx(bands, wall, "-o", ms=3, label="solid wall")
ax[1, 1].semilogx(bands, comp, "-s", ms=3, label="wall + 1 % slit")
ax[1, 1].axhline(20.0, ls=":"); ax[1, 1].set_title("Composite with aperture")

for a in ax.flat:
    a.set_xlabel("Frequency [Hz]"); a.legend(fontsize=8); a.grid(alpha=0.3)
fig.suptitle("Theoretical panel sound insulation")
fig.tight_layout(); plt.show()
```

</details>

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
- Vigran, T. E. (2008). *Building Acoustics*. Taylor & Francis.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  Sections 3.7.3.3 (orthotropic plate eigenfrequencies, Eqs 3.113-3.115 after
  Timoshenko & Woinowsky-Krieger) and 6.5.3 (orthotropic panel transmission,
  Eqs 6.107-6.113 after Heckl 1960 and Hansen 1993).

- Hopkins, C., Wilson, R., & Craik, R. J. M. (1999). Dynamic stiffness as an
  acoustic specification parameter for wall ties used in masonry cavity walls.
  *Applied Acoustics*, 58, 51-68.
  [doi:10.1016/S0003-682X(98)00068-1](https://doi.org/10.1016/S0003-682X(98)00068-1).
  The measurement behind the 50 mm rows of Hopkins' Table A4.

## See also

- [Predicting Sound Insulation (EN 12354)](insulation-prediction.md): assembles
  predicted or measured element $R$ into the in-situ $R'_w$.
- [Sound Power from Surface Vibration (ISO 7849)](vibration-sound-power.md):
  consumes the predicted radiation efficiency as its radiation factor.
- [Mechanical mobility and the FRF family (ISO 7626-1)](mechanical-mobility.md):
  the measured counterpart of the theoretical point mobilities.
- [Porous and multilayer absorbers](porous-absorbers.md): the cavity-fill models
  a double wall consumes.
- API reference: [`building.panel_transmission`](https://jmrplens.github.io/phonometry/reference/api/building/panel-transmission/), [`building.masonry_cavity_wall`](https://jmrplens.github.io/phonometry/reference/api/building/masonry-cavity-wall/), [`building.aperture_transmission`](https://jmrplens.github.io/phonometry/reference/api/building/aperture-transmission/), [`vibration.radiation_efficiency`](https://jmrplens.github.io/phonometry/reference/api/vibration/radiation-efficiency/) and [`vibration.point_mobility`](https://jmrplens.github.io/phonometry/reference/api/vibration/point-mobility/).
