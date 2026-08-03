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

and a side branch of acoustic impedance $Z_b$ is the shunt
$\left[\begin{smallmatrix} 1 & 0 \\ 1/Z_b & 1 \end{smallmatrix}\right]$
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
+ \tfrac{T_{12}}{Z_c} + Z_c\,T_{21} + T_{22}\right|\right),
\qquad Z_c = \frac{\rho c}{S},
$$

and the **insertion loss** for a source impedance $Z_s$ and a radiation
impedance $Z_r$ is the extra attenuation over a direct (zero-length)
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
from phonometry import expansion_chamber

freqs = np.linspace(20.0, 2000.0, 2000)
res = expansion_chamber(freqs, length=0.3, chamber_area=0.04, pipe_area=0.01)
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
from phonometry import expansion_chamber, plot_silencer_geometry

freqs = np.linspace(20.0, 2000.0, 2000)
res = expansion_chamber(freqs, length=0.3, chamber_area=0.04, pipe_area=0.01)

# One line: the dimensioned cross-section of the chamber just computed.
res.plot_geometry()
plt.show()

# The same drawing without a result, from the free function:
plot_silencer_geometry("expansion chamber", length=0.3,
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
frequency, $f_0 = \tfrac{c}{2\pi}\sqrt{S_\text{neck}/(l_e V)}$
(Bies Eq. (8.46)) and $f = c/4l_e$ (Eq. (8.44)), giving a sharp
transmission-loss spike there. An
**extended-tube chamber** (`extended_tube_chamber`) buries quarter-wave side
branches in an expansion chamber to fill its troughs; with zero extensions it
reduces exactly to the plain chamber. Advanced layouts chain elements directly
with `duct_matrix`, `shunt_matrix`, `cascade`, `transmission_loss` and
`insertion_loss`.

```python
import numpy as np
from phonometry import (
    helmholtz_resonator, quarter_wave_resonator, extended_tube_chamber,
)

f = np.linspace(20.0, 600.0, 4000)

hr = helmholtz_resonator(f, duct_area=0.01, neck_area=1e-4,
                         neck_length=0.02, cavity_volume=1e-3)
print(round(float(hr.resonances[0]), 1))       # tuning frequency, Hz
hr.plot()   # TL spike at the tuning frequency (needs matplotlib)

qw = quarter_wave_resonator(f, duct_area=0.01, length=1.516, branch_area=2e-3,
                            speed_of_sound=343.24)
print(round(float(qw.resonances[0]), 1))        # 56.6 Hz (Bies Example 8.1)

# An inlet extension of L/4 fills the first expansion-chamber trough.
et = extended_tube_chamber(f, length=0.4, chamber_area=0.04, pipe_area=0.01,
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
from phonometry import helmholtz_resonator, quarter_wave_resonator

f = np.linspace(20.0, 600.0, 4000)
hr = helmholtz_resonator(f, duct_area=0.01, neck_area=1e-4,
                         neck_length=0.02, cavity_volume=1e-3)
qw = quarter_wave_resonator(f, duct_area=0.01, length=0.3, branch_area=2e-3)

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

Both branches are small hardware, and `.plot_geometry()` shows just how
small: the resonator of the 120 Hz spike is a 1 L cavity fed by a 1 cm²
neck only 2 cm long.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/helmholtz_branch_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/helmholtz_branch_geometry.svg" alt="To-scale cross-section of the side-branch Helmholtz resonator: a duct of 112.8 mm equivalent diameter with a narrow 11.3 mm neck, 20 mm long, opening into a 1 litre cavity drawn as its equal-volume cube on top of the duct, with the neck diameter, neck length and duct diameter dimensioned" width="88%"></picture>

*The whole 120 Hz notch hangs on a 1 L box and a 2 cm neck: the cavity is
drawn as its equal-volume cube, and the tuning moves as
$\sqrt{S_n/(l_e V)}$, so small errors in these dimensions shift the spike off
its target.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import helmholtz_resonator

f = np.linspace(20.0, 600.0, 4000)
hr = helmholtz_resonator(f, duct_area=0.01, neck_area=1e-4,
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
pipe puts the spike at $c/4l_e \approx 285\ \text{Hz}$, and the 20 cm² branch
area only sets how strongly the stub loads the duct.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import quarter_wave_resonator

f = np.linspace(20.0, 600.0, 4000)
qw = quarter_wave_resonator(f, duct_area=0.01, length=0.3, branch_area=2e-3)

# One line: the closed 0.3 m tube on its duct, to scale.
qw.plot_geometry()
plt.show()
```

</details>

Each device returns a `ReactiveSilencerResult` with `transmission_loss`,
`insertion_loss` (when source/radiation impedances are given), the compound
`transfer_matrix`, the tuning `resonances` and `.plot()`.


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
dissipative side enters through the installation data of the
[HVAC methods](noise-control.md): the lined-elbow insertion loss and the
lined plenum attenuation of Wells' method, both from interpolated ASHRAE
data rather than a liner model. The porous physics that a first-principles
liner calculation needs, the equivalent-fluid models fed by the airflow
resistivity, is the same material theory as
[Porous and Multilayer Absorbers](../../materials/absorbers/porous-absorbers.md).

## Cross-check against the FDTD solver

The four-pole expansion chamber is cross-checked against the independent 2D
[FDTD wave solver](../../simulation/fdtd-simulation.md): a plane-wave duct that widens into a
chamber and narrows back transmits far less at the four-pole TL peak
($kL = \pi/2$) than at the transparent trough ($kL = \pi$), and the measured
amplitude ratio reproduces the closed-form peak transmission loss to a fraction
of a decibel (test `tests/noise_control/test_fdtd_crosscheck.py`).


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
