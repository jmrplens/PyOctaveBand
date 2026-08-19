← [Documentation index](../../README.md)

# Silencers

A silencer earns its keep in a duct: between an engine and its tailpipe,
between a fan and the room it serves. Two working principles divide the
field. A **reactive** silencer attenuates by *reflecting* sound with
impedance discontinuities (chambers, side branches) and dominates at low
frequency, where a tone from a firing engine or a fan blade passage can be
targeted exactly; a **dissipative** silencer *absorbs* sound in a porous
lining and dominates broadband, at mid and high frequency. This guide
covers the reactive family phonometry implements with the engineering
theory of Bies, Hansen & Howard and Munjal's transfer-matrix formulation:
the four-pole method, the closed-form expansion chamber, the Helmholtz,
quarter-wave and extended-tube resonators, the independent FDTD
cross-check, and the design trade-offs against dissipative linings. The
rest of the installation, HVAC duct attenuation, flow noise and machine
enclosures, lives in [Industrial noise control](noise-control.md).

## 1. Reactive silencers (four-pole method)

A reactive silencer attenuates by *reflecting* sound with impedance
discontinuities. Each acoustic element is a 2×2 **transfer (four-pole)
matrix** relating the sound pressure $p$ and the volume velocity $Su$ at its
two ends (Bies Eq. (8.133); Munjal, *Acoustics of Ducts and Mufflers*), and a
compound silencer is the ordered matrix product of its elements. A straight
duct of length $L$ and area $S$ is (Bies Eq. (8.143), no flow)

$$
\begin{bmatrix} \cos kL & j\,\tfrac{\rho c}{S}\sin kL \\[2pt]
j\,\tfrac{S}{\rho c}\sin kL & \cos kL \end{bmatrix},
\qquad k = \omega/c,
$$

and a side branch of acoustic impedance $Z_\mathrm{b}$ is the shunt
$\left[\begin{smallmatrix} 1 & 0 \\ 1/Z_\mathrm{b} & 1 \end{smallmatrix}\right]$
(Eq. (8.144)). The **transmission loss** follows from the compound matrix $T$
with the port impedances $Z_1 = \rho c/S_\text{in}$ and
$Z_n = \rho c/S_\text{out}$ (Munjal Eq. (3.27); Bies Eq. (8.141) prints the
`T11`/`T22` impedance weights of this formula inverted and fails the
sudden-expansion limit, see the [errata registry](../../ERRATA.md))

$$
\mathrm{TL} = 10\log_{10}\!\left[\frac{Z_n}{Z_1}\,\tfrac{1}{4}\left|\,T_{11}
+ \tfrac{T_{12}}{Z_n} + Z_1\,T_{21} + \tfrac{Z_1}{Z_n}\,T_{22}\right|^2\right],
$$

which for equal inlet/outlet areas reduces to (Bies Eq. (8.148))

$$
\mathrm{TL} = 20\log_{10}\!\left(\tfrac{1}{2}\left|\,T_{11}
+ \tfrac{T_{12}}{Z_\mathrm{c}} + Z_\mathrm{c}\,T_{21} + T_{22}\right|\right),
\qquad Z_\mathrm{c} = \frac{\rho c}{S},
$$

and the **insertion loss** for a source impedance $Z_\mathrm{s}$ and a radiation
impedance $Z_\mathrm{r}$ is the extra attenuation over a direct (zero-length)
connection, so a through connection gives $\mathrm{IL} = 0$.

### Expansion chamber

A chamber of area $S_\text{exp}$ and length $L$ between pipes of area
$S_\text{duct}$ has the closed-form transmission loss (Bies Eq. (8.111)) with
area ratio $m = S_\text{exp}/S_\text{duct}$:

$$
\mathrm{TL} = 10\log_{10}\!\left[1 + \tfrac{1}{4}\left(m - \tfrac{1}{m}\right)^2
\sin^2 kL\right],
$$

peaking at $10\log_{10}[1 + \tfrac14(m-1/m)^2]$ at $kL = \pi/2, 3\pi/2, \dots$
(1.94 dB for $m = 2$, 6.55 dB for $m = 4$, 12.18 dB for $m = 8$, 18.10 dB for
$m = 16$) and dropping to 0 at $kL = n\pi$, where the chamber is a
half-wavelength long and transparent. The four-pole product reproduces this
exactly.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/silencer_expansion_chamber_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/silencer_expansion_chamber.svg" alt="Expansion-chamber transmission loss against frequency for area ratios m = 2, 4, 8 and 16, showing periodic peaks rising with m at odd multiples of the quarter-wave frequency and troughs returning to 0 dB at every half-wavelength of the chamber length" width="88%"></picture>

```python
import numpy as np
from phonometry import noise_control

freqs = np.linspace(20.0, 2000.0, 2000)
res = noise_control.expansion_chamber(freqs, length=0.3, chamber_area=0.04, pipe_area=0.01)
print(round(res.transmission_loss.max(), 2))   # 6.55 dB peak (m = 4)
res.plot()                                      # TL (and IL) vs frequency
```

The numbers passed to `expansion_chamber` describe a real device, and
`.plot_geometry()` draws it: the same 0.3 m chamber with its 4:1 area ratio,
to scale and fully dimensioned.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/expansion_chamber_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/expansion_chamber_geometry.svg" alt="To-scale cross-section of the expansion-chamber silencer of the transmission-loss example: a 300 mm long chamber of 225.7 mm equivalent diameter inserted between inlet and outlet pipes of 112.8 mm equivalent diameter, with the chamber length and both diameters dimensioned" width="88%"></picture>

*The chamber behind the curves above, to scale: the areas enter the four-pole
method only through the ratio $m$, and the drawing uses the equivalent
circular diameters $d = 2\sqrt{S/\pi}$ of the 0.04 and 0.01 m² cross-sections.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import noise_control

freqs = np.linspace(20.0, 2000.0, 2000)
res = noise_control.expansion_chamber(freqs, length=0.3, chamber_area=0.04, pipe_area=0.01)

# One line: the dimensioned cross-section of the chamber just computed.
res.plot_geometry()
plt.show()

# The same drawing without a result, from the free function:
noise_control.plot_silencer_geometry("expansion chamber", length=0.3,
                                     chamber_area=0.04, pipe_area=0.01)
plt.show()
```

</details>

The clip below runs an $m = 4$ chamber of the same 0.30 m length in a 2D FDTD
duct at its two characteristic frequencies. At $kL = \pi$ the chamber is a
half-wave resonator and the tone crosses as if it were not there; at
$kL = \pi/2$ the two area jumps reflect in phase and send the wave back up the
inlet, the 6.5 dB peak of the four-pole curve above.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_expansion_chamber_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_expansion_chamber.gif" alt="Animation: a 2D FDTD duct with a 0.30 m expansion chamber of area ratio 4 drawn as hardware between a loudspeaker and an anechoic termination, at two frequencies side by side; at 572 Hz the pressure envelope stays flat and the tone crosses the chamber unchanged with the annotated transmission loss of 0.0 dB, while at 286 Hz a standing wave fills the inlet pipe and the outlet is left with less than half the amplitude, matching the annotated 6.5 dB peak" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_expansion_chamber.webm)

### Side-branch and extended-tube resonators

A **Helmholtz resonator** (`helmholtz_resonator`) and a closed **quarter-wave
tube** (`quarter_wave_resonator`) each short the duct at their tuning
frequency, $f_0 = \tfrac{c}{2\pi}\sqrt{S_\text{neck}/(l_\mathrm{e} V)}$
(Bies Eq. (8.46)) and $f = c/4l_\mathrm{e}$ (Eq. (8.44)), giving a sharp
transmission-loss spike there. An
**extended-tube chamber** (`extended_tube_chamber`) buries quarter-wave side
branches in an expansion chamber to fill its troughs; with zero extensions it
reduces exactly to the plain chamber. Advanced layouts chain elements directly
with `duct_matrix`, `shunt_matrix`, `cascade`, `transmission_loss` and
`insertion_loss`.

```python
import numpy as np
from phonometry import noise_control

f = np.linspace(20.0, 600.0, 4000)

hr = noise_control.helmholtz_resonator(f, duct_area=0.01, neck_area=1e-4,
                                       neck_length=0.02, cavity_volume=1e-3)
print(round(float(hr.resonances[0]), 1))       # tuning frequency, Hz
hr.plot()   # TL spike at the tuning frequency (needs matplotlib)

qw = noise_control.quarter_wave_resonator(f, duct_area=0.01, length=1.516, branch_area=2e-3,
                                          speed_of_sound=343.24)
print(round(float(qw.resonances[0]), 1))        # 56.6 Hz (Bies Example 8.1)

# Each extension is a quarter-wave stub, so its own length picks the trough
# it fills: L/4 = 0.1 m shorts the duct at c/4(L/4) = c/L, the chamber's
# second trough, and L/2 = 0.2 m would take the first one at c/2L.
et = noise_control.extended_tube_chamber(f, length=0.4, chamber_area=0.04, pipe_area=0.01,
                                         inlet_extension=0.1)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/silencer_side_branch_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/silencer_side_branch.svg" alt="Transmission loss of a Helmholtz resonator and a closed quarter-wave tube on the same 10 cm2 duct: each side branch produces a sharp spike at its own tuning frequency, near 120 Hz for the Helmholtz volume and near 285 Hz for the 0.3 m tube, and is transparent elsewhere" width="88%"></picture>

*Each side branch shorts the duct at its own tuning frequency and is nearly
transparent elsewhere: the narrow spike is why resonators are matched to a
firing frequency or a fan blade-passing tone rather than used broadband.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import noise_control

f = np.linspace(20.0, 600.0, 4000)
hr = noise_control.helmholtz_resonator(f, duct_area=0.01, neck_area=1e-4,
                                       neck_length=0.02, cavity_volume=1e-3)
qw = noise_control.quarter_wave_resonator(f, duct_area=0.01, length=0.3, branch_area=2e-3)

# One line for one device: TL vs frequency with the resonance marked.
hr.plot()
plt.show()

# By hand: both side branches on the same axes.
fig, ax = plt.subplots()
ax.plot(f, hr.transmission_loss, label="Helmholtz resonator")
ax.plot(f, qw.transmission_loss, "--", label="Quarter-wave tube")
for fr in (hr.resonances[0], qw.resonances[0]):
    ax.axvline(float(fr), ls=":", color="#2ca02c")
ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Transmission loss [dB]")
ax.set_ylim(0.0, 50.0)
ax.legend()
plt.show()
```

</details>

Why is the spike only a few hertz wide? Because a resonator takes time to
work: it has to **charge**. The clip below drives the 0.30 m stub above in
a 2D FDTD duct at its 285.8 Hz tuning frequency and at 150 Hz side by
side. On tune the closed-end pressure ratchets up over about six periods
to 8.2 times the incident wave (the lossless branch at exact resonance
would reach ten); off tune it settles at 1.5 times immediately and never
charges — the charge has the percent-wide bandwidth $f/Q$, while the
lossless TL spike is hertz-wide. The clip also reports the trim procedure
once on screen: built at the drilled 300 mm, the simulated device rings at
272.9 Hz, an effective length $c/4f = 314$ mm, the junction end correction
made visible.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_side_branch_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_side_branch.gif" alt="Animation: a 2D FDTD duct carrying a 0.30 m closed quarter-wave stub, driven at the 285.8 Hz tuning frequency and at 150 Hz side by side, with the closed-end pressure of both runs traced below; on tune the pressure inside the stub ratchets up over about six periods to 8.2 times the incident wave, off tune it settles at 1.5 times immediately, and an annotation reports that the built 300 mm stub rings at 272.9 Hz, an effective length of 314 mm" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_side_branch.webm)

Both branches are small hardware, and `.plot_geometry()` shows just how
small: the resonator of the 120 Hz spike is a 1 L cavity fed by a 1 cm²
neck only 2 cm long.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/helmholtz_branch_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/helmholtz_branch_geometry.svg" alt="To-scale cross-section of the side-branch Helmholtz resonator: a duct of 112.8 mm equivalent diameter with a narrow 11.3 mm neck, 20 mm long, opening into a 1 litre cavity drawn as its equal-volume cube on top of the duct, with the neck diameter, neck length and duct diameter dimensioned" width="88%"></picture>

*The whole 120 Hz notch hangs on a 1 L box and a 2 cm neck: the cavity is
drawn as its equal-volume cube, and the tuning moves as
$\sqrt{S_\mathrm{n}/(l_\mathrm{e} V)}$, so small errors in these dimensions shift the spike off
its target.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import noise_control

f = np.linspace(20.0, 600.0, 4000)
hr = noise_control.helmholtz_resonator(f, duct_area=0.01, neck_area=1e-4,
                                       neck_length=0.02, cavity_volume=1e-3)

# One line: the side branch drawn to scale, cavity as its equal-volume cube.
hr.plot_geometry()
plt.show()
```

</details>

The quarter-wave tube needs no cavity at all: the 285 Hz spike of the figure
above comes from a plain closed tube of the right length standing on the
same duct.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/quarter_wave_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/quarter_wave_geometry.svg" alt="To-scale cross-section of the quarter-wave side branch: a closed tube 300 mm long and 50.5 mm in equivalent diameter standing on a duct of 112.8 mm equivalent diameter, with the tube length, tube diameter and duct diameter dimensioned" width="88%"></picture>

*A quarter-wave stub is just a closed tube of the right length: 0.3 m of
pipe puts the spike at $c/4l_\mathrm{e} \approx 285\ \text{Hz}$, and the 20 cm² branch
area only sets how strongly the stub loads the duct.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import noise_control

f = np.linspace(20.0, 600.0, 4000)
qw = noise_control.quarter_wave_resonator(f, duct_area=0.01, length=0.3, branch_area=2e-3)

# One line: the closed 0.3 m tube on its duct, to scale.
qw.plot_geometry()
plt.show()
```

</details>

Each device returns a `ReactiveSilencerResult` with `transmission_loss`,
`insertion_loss` (when source/radiation impedances are given), the compound
`transfer_matrix`, the tuning `resonances`, the `plane_wave_limit` of its widest
cross section and `.plot()`. That last one is the validity ceiling: above the
first higher-order cut-on the computed peaks and troughs do not survive, and the
result raises a `PlaneWaveWarning` when the analysis grid runs past it (890.8 Hz
for the 0.04 m² chamber above).

**Layouts of your own.** A two-chamber muffler, or a chamber with a stub on its
inlet pipe, is cascaded element by element. `SilencerChain` makes the same
`duct_matrix`/`shunt_matrix`/`cascade` calls and keeps what each element was
handed, so the layout can be drawn as well as computed. What the drawing may
show follows from what the elements declare: a duct is handed a length and an
area and is drawn to scale and dimensioned, while a shunt is handed an
impedance, which fixes no length, no area and no volume, so it is marked at the
station where it joins and lettered with the frequency at which it is least,
and nothing about it is dimensioned. A chain that holds no duct of positive
length has no geometry and no scale to draw at, so it raises rather than
drawing one.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/silencer_chain_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/silencer_chain_geometry.svg" alt="To-scale cross-section of a hand-built silencer chain: 100 mm and 200 mm runs of 200 mm duct opening into a 400 mm shell 600 mm long and returning to 300 mm of 200 mm duct, with each run length, both bores and the 1200 mm overall length dimensioned, and two side branches marked by leaders at the stations where they join, one lettered as a quarter-wave stub with least impedance at 125 Hz and the other as a Helmholtz resonator at 242 Hz" width="88%"></picture>

*Every measurement on the page comes from what the chain was given: the four
duct lengths and their sum are the numbers themselves, and the two bores are
the areas it was handed, restated as the equivalent circular diameter
$d = 2\sqrt{S/\pi}$. The two branches carry no dimension at
all, because an impedance is not a shape.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import noise_control
from phonometry.noise_control import helmholtz_impedance, quarter_wave_impedance

freqs = np.linspace(20.0, 500.0, 481)
s_duct = np.pi * 0.100**2               # nominal 200 mm duct
s_shell = np.pi * 0.200**2              # nominal 400 mm shell

chain = (
    noise_control.SilencerChain(freqs)
    .duct(0.10, s_duct)
    .shunt(quarter_wave_impedance(freqs, 343.0 / (4.0 * 125.0), np.pi * 0.050**2),
           label="Quarter-wave stub")
    .duct(0.20, s_duct)
    .duct(0.60, s_shell)
    .shunt(helmholtz_impedance(freqs, np.pi * 0.025**2, 0.05, 2e-3),
           label="Helmholtz resonator")
    .duct(0.30, s_duct)
)

# One line: the chain drawn exactly as it was declared.
chain.plot_geometry()
plt.show()

# The same elements evaluated, ports included.
res = chain.result(inlet_area=s_duct, outlet_area=s_duct)
```

</details>

**Test-report fiche.** `ReactiveSilencerResult.report(path)` renders a one-page
fiche in the layout of a silencer performance sheet: a metadata header, the
octave-band transmission-loss table beside the same curve plotted against
frequency, and the boxed mean transmission loss over the analysis bands together
with the peak value and the device kind, plus an optional verdict against a
declared minimum mean transmission loss. Rendering needs reportlab and, for the
embedded figure, matplotlib (`pip install "phonometry[report,plot]"`), and
`language="es"` renders a Spanish fiche.

```python
import numpy as np
from phonometry import ReportMetadata, noise_control

# A 0.5 m chamber of area ratio m = 8, at the octave-band centres.
freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
res = noise_control.expansion_chamber(freqs, length=0.5, chamber_area=0.08, pipe_area=0.01)
res.report(
    "silencer_fiche.pdf",
    metadata=ReportMetadata(
        specimen="Simple expansion-chamber muffler (m = 8, design case)",
        measurement_standard="Munjal Eq. (3.27) four-pole model",
        laboratory="Phonometry Reference Laboratory",
        requirement=6.0,            # minimum acceptable mean transmission loss
    ),
)                                   # mean and peak transmission loss (dB)
```

The example fiche is regenerated with `make reports` and kept rendered in the
repository:

[![One-page reactive-silencer fiche: a metadata header with the client, the expansion-chamber muffler of area ratio m = 8 as the noise source, the duct-system design study and the test date, the octave-band transmission-loss table running 7.5, 11.4, 9.9, 12.1, 3.2, 7.0 and 11.1 dB from 63 Hz to 4 kHz beside the same curve plotted against frequency, the boxed mean transmission loss TL = 8.9 dB with the peak transmission loss of 12.1 dB and the device named as an expansion chamber, a PASS verdict against the required minimum of 6.0 dB, and the note that the result is a plane-wave prediction from the declared geometry and not a measurement.](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/reactive_silencer_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/reactive_silencer_example.pdf)

**What that fiche is not.** The number in the box is a plane-wave prediction
from the declared geometry. The figure a supplier publishes is an **insertion
loss measured by substitution** to ISO 7235:2003: two series with everything
else unchanged, one with the test object installed and one with a substitution
duct in its place, differenced third octave by third octave as
`D_i = L_pII - L_pI`. The rig carries its own requirements — a sealed, lined
loudspeaker box driving at least 6 dB and preferably 10 dB above the background,
a modal filter attenuating the fundamental by at least 3 dB and higher-order
modes by at least 5 dB above cut-on, a substitution duct matched within 5 % in
every linear dimension, a receiving side with a reflection coefficient no
greater than 0.3, and at least three microphone positions on a line inclined to
the duct axis — and every facility has a *limiting insertion loss* set by
flanking along its own duct walls, which caps what it can report at all. A
computed transmission loss and a catalogue insertion loss are therefore not the
same quantity and must not be compared directly.


## 2. Reactive or dissipative?

Everything above works by reflection, and reflection has a shape: sharp,
periodic, frequency-selective. The complementary family, **dissipative**
silencers, replaces the impedance discontinuities with a duct section whose
walls are lined with porous material (often protected by a perforated
facing), so the grazing wave loses energy to viscous friction in the lining
instead of being sent back to the source. The behaviours differ where it
matters for selection:

- **Frequency reach.** A reactive chamber or resonator is strongest exactly
  where it is tuned and transparent elsewhere; a lined duct attenuates over
  a broad band that peaks where the lining depth is comparable with a
  quarter wavelength. At low frequency a practical lining is acoustically
  thin and does little, which is the regime where the reactive chamber
  wins; at high frequency the sound beams down the open airway and passes
  over the lining, so the attenuation of both families collapses and only
  splitter geometries (narrow airways, more lined perimeter per unit area)
  keep working.
- **Spectrum type.** A tonal source, an engine firing order or a
  blade-passing frequency, is a resonator's natural prey; broadband fan or
  flow noise wants the dissipative band. Production exhaust silencers
  routinely combine the two, packing an expansion chamber with fibre so the
  reflective troughs of the chamber are filled by absorption.
- **The medium.** A porous lining in a hot, sooty or pulsating exhaust
  clogs and degrades, one reason vehicle exhausts are predominantly
  reactive; clean HVAC air is where dissipative attenuators and lined
  plenums are the default. In either case the airflow adds its own floor:
  a silencer regenerates flow noise at its own outlet, and past a certain
  pressure drop the silencer becomes the noise source.

phonometry models the reactive family in closed form on this page. The
dissipative side enters at installation level, through the
[HVAC methods](noise-control.md): the lined-elbow insertion loss read from
tabulated ASHRAE data and the lined plenum attenuation from Wells' closed
form, neither of them a liner model. The porous physics that a first-principles
liner calculation needs, the equivalent-fluid models fed by the airflow
resistivity, is the same material theory as
[Porous and Multilayer Absorbers](../../materials/absorbers/porous-absorbers.md).

## Cross-check against the FDTD solver

That cross-check is the clip embedded in section 1, and it is worth returning
to it now with the algebra in hand. The four-pole expansion chamber is checked
against the independent 2D
[FDTD wave solver](../../simulation/fdtd-simulation.md), which shares no formula and no
assumption with the transfer-matrix product beyond the wave equation itself: a
plane-wave duct that widens into the same 0.30 m, $m = 4$ chamber and narrows
back transmits far less at the four-pole TL peak ($kL = \pi/2$, here 286 Hz)
than at the transparent trough ($kL = \pi$, 572 Hz). The amplitude ratio
measured downstream in the field is the transmission loss annotated on the
clip, 6.5 dB at 286 Hz and 0.0 dB at 572 Hz, against the 6.55 dB the closed
form gives for $m = 4$ (test `tests/noise_control/test_fdtd_crosscheck.py`).
Agreement that close rules out an algebra error on either side. The two must
eventually part company above the duct's first cut-on frequency, where
higher-order modes propagate: the two-dimensional solver keeps working there
and the plane-wave algebra does not.


## What this guide covers

**Covered.** Reactive silencers by the four-pole transfer-matrix method
(Bies §8.8-8.9, Munjal Eq. (3.27)): the closed-form `expansion_chamber`
(Eq. (8.111)), `helmholtz_resonator` and `quarter_wave_resonator` (Eqs. (8.46),
(8.44)), `extended_tube_chamber`, and the `duct_matrix` / `shunt_matrix` /
`cascade` / `transmission_loss` / `insertion_loss` blocks for arbitrary chains,
gathered by `SilencerChain` into a chain that keeps its geometry, each with its
to-scale `.plot_geometry()` and cross-checked against the independent FDTD
solver; the `plane_wave_limit` that bounds all of them; and,
as prose rather than code, the ISO 7235 substitution measurement that produces
the insertion loss a supplier publishes.

**Not covered.** Reactive elements only. Dissipative (absorptive, duct-lining)
silencers are discussed for selection but never modelled from liner properties,
and in the [HVAC methods](noise-control.md) the lined-elbow figure is a table
lookup (Bies Table 8.11) and the plenum is Wells' closed form driven by a
declared mean absorption — neither is a liner model.
Mean-flow effects — convection, temperature gradients, the
flow-dependent impedance of perforates — are outside the no-flow element
matrices used here. Nothing on this page is a measurement: no part of ISO 7235
is implemented, shell breakout and the end corrections at the area jumps are
not modelled, and the branch models are lossless unless a `resistance` is
supplied.

## See also

- [Duct-borne noise: fan to room](duct-path.md): the end-to-end fan-to-room
  calculation these silencers sit inside, and the higher-order-mode cut-on
  above which the four-pole method describes the plane-wave mode alone.
- [Industrial noise control](noise-control.md): the rest of the
  installation: HVAC duct attenuation and flow noise, plenums, end
  reflection and machine enclosures.
- [2D FDTD wave simulation](../../simulation/fdtd-simulation.md): the independent solver
  behind the expansion-chamber cross-check.
- [Porous and Multilayer Absorbers](../../materials/absorbers/porous-absorbers.md): the
  equivalent-fluid material theory behind dissipative linings.
- [Loudspeaker Characterisation (IEC 60268-5)](../electroacoustics/loudspeakers.md): the
  radiating piston, the companion radiator model of a duct's open end.
- API reference: [`noise_control.silencers`](https://jmrplens.github.io/phonometry/reference/api/noise_control/silencers/).

## References

- Bies, D. A., Hansen, C. H., & Howard, C. Q. (2017). *Engineering noise
  control* (5th ed.). CRC Press.
  [doi:10.1201/9781351228152](https://doi.org/10.1201/9781351228152). The
  muffler four-pole method, the expansion-chamber TL and the resonator
  tuning formulas (§8.8–8.9) of this guide.
- Munjal, M. L. (2014). *Acoustics of ducts and mufflers* (2nd ed.). Wiley.
  [doi:10.1002/9781118443767](https://doi.org/10.1002/9781118443767). The
  transfer-matrix formulation behind the element matrices and the
  transmission loss from the compound matrix (Eq. (3.27)), and the
  reference treatment of dissipative and combined mufflers.
- Vér, I. L., & Beranek, L. L. (2006). *Noise and vibration control
  engineering* (2nd ed.). Wiley.
  [doi:10.1002/9780470172568](https://doi.org/10.1002/9780470172568). The
  companion treatment of reactive and dissipative silencers.
