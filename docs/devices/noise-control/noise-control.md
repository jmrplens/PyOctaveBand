← [Documentation index](../../README.md)

# Industrial noise control: HVAC and enclosures

Three passive measures dominate applied noise control, and the
`noise_control` domain covers all three with the engineering theory of Bies,
Hansen & Howard, *Engineering Noise Control* (5th ed., CRC Press 2017):
**reactive silencers** in a duct (the four-pole transmission-matrix method),
the passive attenuations and regenerated noise of an **HVAC** run, and the
insertion loss of a **machine enclosure**. The radiating piston of the
[loudspeaker guide](../electroacoustics/loudspeakers.md) is the companion radiator model.

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
enclosure around the machine actually delivers. Chaining those element
models end to end, from the fan sound power to the room criterion, is
[Duct-borne noise: fan to room](duct-path.md).

## 1. HVAC duct attenuation and flow noise

`noise_control.hvac` gathers the Bies Chapter 8 duct methods:

- `end_reflection_loss` — the low-frequency reflection back up an open duct end
  (ASHRAE Table 8.14, interpolated over diameter and frequency; it passes
  exactly through the tabulated nodes).
- `elbow_insertion_loss` — the insertion loss per bend for square/round,
  vaned/unvaned and lined/unlined elbows keyed by $W/\lambda$ (ASHRAE
  Table 8.11).
- `plenum_attenuation` — the plenum-chamber transmission loss by Wells' method
  (Eq. (8.275)), whose reverberant term uses the plenum
  [room constant](../../buildings/rooms/room-image-sources.md).
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
inlet-to-outlet line of sight $r$ and the outlet area, plus the lined wall
area; `plot_plenum_geometry` draws exactly those, honouring $r$ and its
angle off the inlet axis.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/plenum_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/plenum_geometry.svg" alt="Section of a plenum chamber: the inlet duct enters low on the left, the outlet mouth is marked on the right wall, the 1.2 m inlet-to-outlet line of sight is drawn as a dashed diagonal at 0.35 rad off the inlet axis, and the wall area of 6 square metres and outlet area of 0.09 square metres are annotated below" width="88%"></picture>

*Only $r$ and its angle off the inlet axis fix the drawn box; $S_\text{out}$
and $S_w$ enter Wells' method as bare areas, so any plenum sharing these four
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

Rectangular ducts use the equivalent diameter $D = \sqrt{4S/\pi}$. Bies 5th
ed. gives the duct end reflection only as the ASHRAE table (no closed form in
that edition); this module reproduces and interpolates it.

Two conditions travel with those models. The lined-elbow values assume the
lining extends **at least three duct diameters up- and downstream of the bend**,
so `lined=True` describes a lined run that contains a bend rather than a lined
bend in a bare duct; a bend in bare duct takes the unlined column. And the
flow-noise model is for a straight, undisturbed run: the sound power carries
$50\log_{10}U$, so the velocity term alone costs 15.1 dB per doubling, and
because the spectrum shape also shifts upward the band levels of a 0.04 m² duct
rise by 15.6 dB at 63 Hz and 21.6 dB at 4 kHz between 10 and 20 m/s. Hence the
customary design velocities: roughly 7 to 10 m/s in plant-room mains, 4 to 5 m/s
in branches and 2 to 3 m/s in the last run before an occupied room.

**Test-report fiche.** `HvacSpectrumResult.report(path)` renders a one-page
duct-noise fiche: the octave-band table of the spectrum beside the same curve
plotted against frequency, the boxed single-number result — the A-weighted sound
power level when the spectrum is regenerated noise, the mean attenuation when it
is a loss — and, when a `requirement` is declared, the verdict against it.
Rendering needs reportlab and, for the embedded figure, matplotlib
(`pip install "phonometry[report,plot]"`).

```python
from phonometry import ReportMetadata
from phonometry.noise_control import hvac

octaves = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]

# The flow noise of a straight supply duct carrying air at 12 m/s.
duct = hvac.flow_noise_straight_duct(octaves, flow_velocity=12.0, area=0.04)
duct.report(
    "hvac-duct-noise.pdf",
    metadata=ReportMetadata(
        specimen="Straight supply duct, 0.04 m2 cross-section (design case)",
        test_room="Air-handling plant room (design case)",
        measurement_standard="VDI 2081-1 prediction model",
        requirement=45.0,               # maximum acceptable L_WA, dB(A)
    ),
)
```

[![One-page HVAC duct-noise fiche: a metadata header naming the straight supply duct and the air-handling plant room, an octave-band table of the flow-generated sound power level falling from 42.5 dB at 63 Hz to 21.8 dB at 4 kHz, the same spectrum plotted beside it, the boxed A-weighted sound power level LWA = 38.8 dB(A) re 1 pW with the overall unweighted LW = 47.0 dB re 1 pW, and a PASS verdict against a declared maximum of 45.0 dB(A).](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/hvac_duct_noise_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/hvac_duct_noise_example.pdf)

That PASS has very little margin: the same duct at 17 m/s returns 48.1 dB(A) and
fails, with nothing else changed.

## 2. Machine enclosures

A sealed enclosure reduces the radiated noise by its panel transmission loss
$R$, minus a penalty $C$ for the reverberant build-up inside the small, hard
cavity (Bies Eqs. (7.103), (7.111)):

$$
\mathrm{IL} = R - C,\qquad C = 10\log_{10}\!\left(0.3 + \frac{S_E}{R_i}\right),
$$

with the external area $S_E$ and the interior room constant
$R_i = S_i \alpha_i/(1-\alpha_i)$ (the same `room_constant` as the
steady-state room field). A hard interior wastes much of the panel $R$; lining
it drives $C$ toward its floor $10\log_{10}0.3 = -5.2$ dB.

**The panel transmission loss $R$ is supplied by the caller** — measured, or
predicted by a panel model — as a per-band array or a callable of frequency.
This module never predicts $R$ itself; it combines a given $R$ with the
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

*What the enclosure delivers is $R - C$, not the panel $R$: even this lined
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

**What the enclosure actually delivers.** An enclosure is a composite, so the
elements combine on an energy basis before $\mathrm{IL} = R - C$ is applied, and
the result is set by the worst of them. A bare opening of relative area
$S_a/S_E$ caps the composite at $10\log_{10}(S_E/S_a)$ whatever the panels are:
one per cent of open area caps it at 20 dB. For the sheet-steel case below
(mean panel $R$ = 32.3 dB, $S_E$ = 24 m²), a 1.28 m² door at $R$ = 15 dB takes
the mean insertion loss from 28.9 dB to 21.4 dB, and adding a 0.24 m² gap at the
door foot takes it to 15.1 dB. `composite_transmission_loss(areas,
reduction_indices)` builds that composite and `enclosure_insertion_loss` takes
its result directly as the panel $R$. Cooling openings become short lined ducts
rather than holes, and the machine must not touch the shell or share its slab,
because a rigid contact bypasses the panels entirely.

**Test-report fiche.** `EnclosureResult.report(path)` renders a one-page
enclosure fiche: the octave-band table of the supplied panel $R$, the interior
correction $C$ and the net $\mathrm{IL}$, the three curves plotted beside it,
and the boxed mean insertion loss with the mean panel $R$ and the external and
internal surface areas, plus the verdict against a declared minimum mean
insertion loss.

```python
import numpy as np
from phonometry import ReportMetadata, enclosure_insertion_loss

octaves = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
sheet_steel_R = np.array([18.0, 22.0, 28.0, 33.0, 38.0, 42.0, 45.0])

case = enclosure_insertion_loss(sheet_steel_R, external_area=24.0,
                                internal_area=30.0, internal_absorption=0.30,
                                frequencies=octaves)
case.report(
    "enclosure.pdf",
    metadata=ReportMetadata(
        specimen="Sheet-steel close-fitting machine enclosure (design case)",
        test_room="Machine hall, line 3 (design case)",
        measurement_standard="Bies & Hansen 7.4.2 prediction model",
        requirement=20.0,               # minimum acceptable mean IL, dB
    ),
)
```

[![One-page machine-enclosure fiche: a metadata header naming the sheet-steel close-fitting enclosure and the machine hall, an octave-band table of the supplied panel transmission loss R, the interior correction C and the net insertion loss IL (18.0, 3.4 and 14.6 dB at 63 Hz up to 45.0, 3.4 and 41.6 dB at 4 kHz), the R, C and IL curves plotted beside it, the boxed mean insertion loss IL = 28.9 dB with the mean panel R = 32.3 dB and the external and internal surface areas SE = 24.00 m2 and Si = 30.00 m2, and a PASS verdict against a declared minimum of 20.0 dB.](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/enclosure_insertion_loss_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/enclosure_insertion_loss_example.pdf)

**And how it would be measured.** ISO 11546-1:1995 determines the same quantity
under laboratory conditions for declaration, ISO 11546-2:1995 in situ for
acceptance: the sound power level (or the level at a specified position) is
determined with the enclosure and again without it, at identical microphone
positions and with the machine at the same operating point, and the difference
is reported band by band over at least 100 Hz to 5 kHz in third octaves.
What has to be recorded includes the **leak ratio** (open area over interior
surface area, with the openings described) and the **fill ratio** (source volume
over interior volume) — the two numbers a prediction never captures.

## What this guide covers

**Covered.** The Bies §8.11-8.17 / ASHRAE HVAC methods —
`hvac.end_reflection_loss` and `hvac.elbow_insertion_loss` (interpolated
tables), `hvac.plenum_attenuation` (Wells' closed form) and
`hvac.flow_noise_straight_duct` / `flow_noise_bend` (VDI 2081) — and the
machine-enclosure insertion loss of Bies §7.4, Eqs. (7.103) and (7.111),
through `enclosure_insertion_loss`, which combines a supplied panel
transmission loss with the interior room-constant correction, together with the
`composite_transmission_loss` of panels, doors and openings that feeds it.

**Not covered.** The reactive elements — expansion chambers, side branches,
extended tubes — live in [Silencers](silencers.md). Dissipative duct-lining
silencers are modelled from liner properties nowhere in the library; the lined
elbow and the plenum here are interpolated installation tables. Nothing on this
page is a measurement: the ISO 11546 procedure of section 2 is described so a
declared figure can be read, not implemented. Structure-borne transmission from
a machine into its enclosure or its slab is outside every model here.

## See also

- [Silencers](silencers.md): the reactive four-pole
  elements and the reactive-versus-dissipative selection.
- [Duct-Borne Noise: Fan to Room](duct-path.md): the
  end-to-end calculation that chains these element models from the fan to the
  room criterion.
- [Loudspeaker Characterisation (IEC 60268-5)](../electroacoustics/loudspeakers.md):
  the radiating piston (radiation impedance and directivity), the companion
  radiator model.
- [Sound Power](../emission/sound-power.md): the source $L_W$ that feeds a
  duct or an enclosure.
- [Room image sources and steady field](../../buildings/rooms/room-image-sources.md):
  the `room_constant` reused by the enclosure interior correction.
- [Panel sound insulation](../../buildings/design/panel-sound-insulation.md):
  the panel transmission loss this page asks you to supply, and the slit and
  circular-aperture models behind the composite of section 2.1.
- [Conformance report](https://github.com/jmrplens/phonometry/blob/main/docs/CONFORMANCE.md):
  the closed forms and worked anchors these implementations are validated
  against.
- API reference:
  [`noise_control.hvac`](https://jmrplens.github.io/phonometry/reference/api/noise_control/hvac/) and
  [`noise_control.enclosures`](https://jmrplens.github.io/phonometry/reference/api/noise_control/enclosures/).

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
