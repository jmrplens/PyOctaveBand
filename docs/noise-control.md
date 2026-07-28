← [Documentation index](README.md)

# Industrial noise control: HVAC and enclosures

Three passive measures dominate applied noise control, and the
`noise_control` domain covers all three with the engineering theory of Bies,
Hansen & Howard, *Engineering Noise Control* (5th ed., CRC Press 2017):
**reactive silencers** in a duct (the four-pole transmission-matrix method),
the passive attenuations and regenerated noise of an **HVAC** run, and the
insertion loss of a **machine enclosure**. The radiating piston of the
[loudspeaker guide](loudspeakers.md) is the companion radiator model.

The three families in one scene: enclose the source, silence the path,
shield the receiver. Each measure carries the value its section computes,
here or in the silencer guide.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_noise_control_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_noise_control.svg" alt="Noise-control scene split into three zones: at the source a machine inside a lined enclosure rated IL = R − C = 25 dB at 500 Hz, along the path a 113 mm duct with a 0.30 m expansion chamber of area ratio 4 peaking at 6.5 dB of transmission loss at 286 Hz, a lined elbow worth 6 dB at 1 kHz and an open end reflecting 18 dB at 63 Hz, and at the receiver an operator cabin rated by the same formula at 31 dB at 1 kHz, with a person standing inside" width="92%"></picture>

The path measure, the silencer itself, has its own guide:
[Silencers](silencers.md) covers the reactive four-pole method, the
closed-form expansion chamber, the Helmholtz, quarter-wave and
extended-tube resonators, the independent FDTD cross-check and the
trade-off against dissipative linings. This page keeps the rest of the
installation: what the duct run adds and removes on its own, and what an
enclosure around the machine actually delivers.

## 1. HVAC duct attenuation and flow noise

`noise_control.hvac` gathers the Bies Chapter 8 duct methods:

- `end_reflection_loss` — the low-frequency reflection back up an open duct end
  (ASHRAE Table 8.14, interpolated over diameter and frequency; it passes
  exactly through the tabulated nodes).
- `elbow_insertion_loss` — the insertion loss per bend for square/round,
  vaned/unvaned and lined/unlined elbows keyed by `W / lambda` (ASHRAE
  Table 8.11).
- `plenum_attenuation` — the plenum-chamber transmission loss by Wells' method
  (Eq. (8.275)), whose reverberant term uses the plenum
  [room constant](room-image-sources.md).
- `flow_noise_straight_duct`, `flow_noise_bend` — the flow-generated (self)
  noise sound power of straight ducts and mitred bends (VDI 2081, Eqs. (8.251),
  (8.254)).

```python
from phonometry.noise_control import hvac

bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
er = hvac.end_reflection_loss(bands, diameter=0.30, termination="flush")
el = hvac.elbow_insertion_loss(bands, width=0.3, bend_type="square", lined=True)
er.plot()   # the band attenuation (or regenerated Lw) in one line (needs matplotlib)
tl = hvac.plenum_attenuation(0.1, 1.0, 20.0, 0.2)      # Wells' method, dB
fn = hvac.flow_noise_straight_duct(bands, flow_velocity=10.0, area=0.04)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/hvac_end_reflection_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/hvac_end_reflection.svg" alt="Duct end reflection loss per octave band for flush duct terminations of 150, 300 and 600 mm diameter: the reflection back up the duct grows steeply towards low frequency and shrinks with duct size, exceeding 17 dB at 63 Hz for the 150 mm duct and vanishing above 1 kHz" width="88%"></picture>

*The open end of a duct reflects low-frequency energy back up the run — for
free, before any silencer: the smaller the duct against the wavelength, the
larger the loss, which is why small diffuser necks tame low-frequency fan
rumble and why the correction must not be double-counted when a manufacturer's
diffuser data already includes it.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry.noise_control import hvac

bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]

# One line for one duct: the HvacSpectrumResult of the 300 mm flush end.
er = hvac.end_reflection_loss(bands, diameter=0.30, termination="flush")
er.plot()
plt.show()

# By hand: the family over duct diameters of the concept figure.
fig, ax = plt.subplots()
for diameter in (0.15, 0.30, 0.60):
    er = hvac.end_reflection_loss(bands, diameter=diameter, termination="flush")
    ax.semilogx(er.frequencies, er.values, "o-",
                label=f"D = {int(diameter * 1000)} mm")
ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("End reflection loss [dB]")
ax.legend(title="Duct diameter")
plt.show()
```

</details>

Wells' plenum formula takes only two truly geometric inputs, the
inlet-to-outlet line of sight `r` and the outlet area, plus the lined wall
area; `plot_plenum_geometry` draws exactly those, honouring `r` and its
angle off the inlet axis.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/plenum_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/plenum_geometry.svg" alt="Section of a plenum chamber: the inlet duct enters low on the left, the outlet mouth is marked on the right wall, the 1.2 m inlet-to-outlet line of sight is drawn as a dashed diagonal at 0.35 rad off the inlet axis, and the wall area of 6 square metres and outlet area of 0.09 square metres are annotated below" width="88%"></picture>

*Only `r` and its angle off the inlet axis fix the drawn box; `S_out` and
`S_w` enter Wells' method as bare areas, so any plenum sharing these four
numbers has the same predicted attenuation.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import plot_plenum_geometry

# The r, S_out and S_w that Wells' formula actually uses, drawn exactly.
plot_plenum_geometry(0.09, 1.2, 6.0, angle=0.35)
plt.show()
```

</details>

Rectangular ducts use the equivalent diameter `D = sqrt(4 S / pi)`. Bies 5th
ed. gives the duct end reflection only as the ASHRAE table (no closed form in
that edition); this module reproduces and interpolates it.

## 2. Machine enclosures

A sealed enclosure reduces the radiated noise by its panel transmission loss
`R`, minus a penalty `C` for the reverberant build-up inside the small, hard
cavity (Bies Eqs. (7.103), (7.111)):

```text
IL = R - C ,   C = 10 log10( 0.3 + S_E / R_i ) ,
```

with the external area `S_E` and the interior room constant
`R_i = S_i alpha_i / (1 - alpha_i)` (the same `room_constant` as the
steady-state room field). A hard interior wastes much of the panel `R`; lining
it drives `C` toward its floor `10 log10 0.3 = -5.2 dB`.

**The panel transmission loss `R` is supplied by the caller** — measured, or
predicted by a panel model — as a per-band array or a callable of frequency.
This module never predicts `R` itself; it combines a given `R` with the
interior absorption.

```python
import numpy as np
from phonometry import enclosure_insertion_loss

bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
panel_R = np.array([18.0, 24.0, 30.0, 36.0, 42.0, 46.0])   # measured, dB
enc = enclosure_insertion_loss(panel_R, external_area=6.0, internal_area=5.0,
                               internal_absorption=0.3, frequencies=bands)
print(np.round(enc.insertion_loss, 1))          # net IL = R - C per band
enc.plot()
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/enclosure_insertion_loss_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/enclosure_insertion_loss.svg" alt="Machine-enclosure insertion loss per octave band: the measured panel sound reduction index R as a dashed line, the flat interior correction C near 5 dB for a lined interior, and the net insertion loss IL equal to R minus C tracking about 5 dB below the panel curve" width="88%"></picture>

*What the enclosure delivers is `R − C`, not the panel `R`: even this lined
interior (mean absorption 0.3) costs about 5 dB of the panel's rating in every
band, and a hard, unlined interior would cost far more. Budget the lining
together with the panels, not as an afterthought.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import enclosure_insertion_loss

bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
panel_R = np.array([18.0, 24.0, 30.0, 36.0, 42.0, 46.0])   # measured, dB

enc = enclosure_insertion_loss(panel_R, external_area=6.0, internal_area=5.0,
                               internal_absorption=0.3, frequencies=bands)

# One line — panel R, interior correction C and the net IL = R - C:
enc.plot()
plt.show()

# By hand, from the per-band fields the result carries:
fig, ax = plt.subplots()
ax.plot(bands, enc.panel_transmission_loss, "s--", label="Panel R")
ax.plot(bands, enc.correction, "^:", label="Interior correction C")
ax.plot(bands, enc.insertion_loss, "o-", label="Insertion loss (R - C)")
ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Level [dB]")
ax.set_xscale("log")
ax.legend()
plt.show()
```

</details>

`enclosure_insertion_loss` returns an `EnclosureResult` with the panel
`panel_transmission_loss`, the interior `correction`, the net `insertion_loss`,
the interior `room_constant` and `.plot()`.

## References

- Bies, D. A., Hansen, C. H., & Howard, C. Q. (2017). *Engineering noise
  control* (5th ed.). CRC Press.
  [doi:10.1201/9781351228152](https://doi.org/10.1201/9781351228152). The
  HVAC duct methods (§8.11–8.17) and the machine-enclosure noise reduction
  (§7.4).
- Vér, I. L., & Beranek, L. L. (2006). *Noise and vibration control
  engineering* (2nd ed.). Wiley.
  [doi:10.1002/9780470172568](https://doi.org/10.1002/9780470172568). The
  companion treatment of ducts and enclosures.
