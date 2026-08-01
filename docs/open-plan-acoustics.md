← [Documentation index](README.md)

# Open-Plan Office Acoustics (ISO 3382-3)

An open-plan office is not a reverberation problem; it is a privacy
problem, and a room that flatters the closed-room parameters can still fail
it. ISO 3382-3 therefore characterises the office with four single numbers
measured along a line of workstations, walking away from a talker. This
guide covers that measurement chain: the quantities, the to-scale
measurement line and the accredited fiche. The impulse responses and levels
behind it are acquired as in
[Measuring the Room Impulse Response](room-impulse-response.md); the
per-position STI comes from the
[Speech Transmission Index guide](speech-transmission.md); the closed-room
decay parameters live in [Room Acoustics](room-acoustics.md).

## Speech privacy along a line of workstations

Open-plan acoustics are about **speech privacy**: how fast a talker's speech
fades to unintelligibility as you walk away. Levels and STI are measured
along a line of workstations (at least 4 positions, 6–10 preferred), and
four single-number quantities summarise the room. The **spatial decay
rate** of A-weighted speech is the slope of the level against
$\log_{10}(r/r_0)$, scaled to a per-doubling figure using only the 2–16 m
positions,

$$
D_{2,S} = -\log_{10}(2)\ b, \qquad L = a + b\ \log_{10}(r/r_0),\ r_0 = 1\ \text{m},
$$

with **$L_{p,A,S,4m}$** read off the same line at 4 m. The **distraction
distance** $r_D$ (STI = 0.50) and **privacy distance** $r_P$ (STI = 0.20) come
from a linear regression of STI against distance. Good offices push $r_D$
below ~5 m; poor ones leave speech distracting past 10 m.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_open_plan_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_open_plan.svg" alt="ISO 3382-3 open-plan measurement line from the source at 1 m along positions from 2 m to 16 m, feeding the four single-number quantities D2,S, Lp,A,S,4m, rD and rP" width="86%"></picture>

```python
import numpy as np
from phonometry import room

r = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0])       # distances from the talker (m)
lp = 65.0 - 7.0 * np.log2(r)                          # A-weighted speech level (dB)
sti = 0.70 - 0.03 * r                                 # STI per position

m = room.open_plan_metrics(r, lp, sti)
print(round(m.d2s, 1), round(m.lp_as_4m, 1))         # 7.0 dB, 51.0 dB
print(round(m.rd, 1), round(m.rp, 1))                # 6.7 m, 16.7 m
m.plot()   # the spatial-decay regression of the figure below
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/open_plan_decay_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/open_plan_decay.svg" alt="Open-plan spatial decay: A-weighted speech level and STI against source distance on a log axis, with the D2,S regression, the Lp,A,S,4m marker at 4 m and the rD and rP distance crossings" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import room

r = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0])       # distances from the talker (m)
lp = 65.0 - 7.0 * np.log2(r)                          # A-weighted speech level (dB)
sti = 0.70 - 0.03 * r                                 # STI per position
m = room.open_plan_metrics(r, lp, sti)

# One line: the D2,S regression rebuilt from the result fields, with the
# rD / rP crossings marked (the figure above adds the measured points and
# the STI axis on top of it):
m.plot()
plt.show()
```

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import room

r = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0])       # distances from the talker (m)
lp = 65.0 - 7.0 * np.log2(r)                          # A-weighted speech level (dB)
sti = 0.70 - 0.03 * r                                 # STI per position
m = room.open_plan_metrics(r, lp, sti)

# Spatial decay: measured Lp,A,S vs distance on a log axis, the D2,S
# regression rebuilt from the result fields, and STI with the rD / rP
# crossings on a twin axis:
b = -m.d2s / np.log10(2.0)                 # regression slope vs lg(r)
a = m.lp_as_4m - b * np.log10(4.0)         # intercept from the 4 m level
rr = np.logspace(np.log10(2.0), np.log10(16.0), 100)

fig, ax = plt.subplots()
ax.semilogx(r, lp, "o", label="Measured Lp,A,S")
ax.semilogx(rr, a + b * np.log10(rr), "--", label=f"D2,S = {m.d2s:.1f} dB")
ax.plot(4.0, m.lp_as_4m, "D", label=f"Lp,A,S,4m = {m.lp_as_4m:.0f} dB")
ax.set_xlabel("Distance from the talker r [m]")
ax.set_ylabel("A-weighted speech level [dB]")
ax.set_xlim(1.8, 20.0)

twin = ax.twinx()
twin.semilogx(r, sti, "s-", color="#2ca02c", label="STI")
twin.axvline(m.rd, ls=":", color="#2ca02c")
twin.axvline(m.rp, ls=":", color="#9467bd")
twin.annotate(f"rD = {m.rd:.1f} m", (m.rd, 0.52))
twin.annotate(f"rP = {m.rp:.1f} m", (m.rp, 0.22))
twin.set_ylabel("STI")
twin.set_ylim(0.0, 1.0)

lines, labels = ax.get_legend_handles_labels()
tl, tlab = twin.get_legend_handles_labels()
ax.legend(lines + tl, labels + tlab, loc="best")
plt.show()
```

</details>

The line itself is worth a to-scale plan. `plot_open_plan_geometry` draws the
source, the microphone line across the workstations and the two distances on
the axis, and a result that retained its positions redraws its own line with
`m.plot_geometry()`.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/open_plan_line_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/open_plan_line_geometry.svg" alt="To-scale plan of the open-plan measurement line: the red source star at the origin, six blue microphone dots from 2 m to 16 m on a dotted line through the grey workstation blocks, the dashed distraction distance rD = 6.5 m and privacy distance rP = 13 m marked across the line and the 16 m span dimensioned" width="86%"></picture>

*Every ISO 3382-3 quantity comes off this one line: six microphones from 2 m
to 16 m through the workstations, with $r_D$ and $r_P$ landing between them.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import room

# The 2-16 m line of the example, with the distraction and privacy
# distances marked on the axis.
room.plot_open_plan_geometry([2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                             rd=6.5, rp=13.0)
plt.show()

# A result retains its positions, so m.plot_geometry() draws its own
# line with the regressed rD and rP.
```

</details>

### `open_plan_metrics()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `positions_m` | 1D array | m | ≥ 4 positions, all > 0 | Source-to-receiver distances |
| `spl_a_speech` | 1D array | dB | same length | A-weighted speech level $L_{p,A,S,n}$ per position |
| `sti_values` | 1D array | — | same length | STI per position (full IEC 60268-16 method) |

Returns an `OpenPlanResult` with `d2s`, `lp_as_4m`, `rd` and `rp`; its
`.plot()` redraws the Clause 6.2 spatial-decay regression from those four
fields and marks `rd` / `rp`.
`d2s`/`lp_as_4m` are `nan` if fewer than two positions fall in 2–16 m;
`rd`/`rp` are `nan` when STI does not decrease with distance. The per-position
STI can itself be measured with the STIPA tools in the
[Speech Transmission Index guide](speech-transmission.md).

## ISO 3382-3 report (`.report()`)

`OpenPlanResult.report(path)` renders a one-page PDF fiche laid out like an
open-plan-office speech-privacy measurement report: a standard-basis line, an
optional metadata header block, a compact metrics table of the four
single-number quantities of Clause 4 ($D_{2,S}$, $L_{p,A,S,4m}$, the
distraction distance $r_D$ and the privacy distance $r_P$) stacked above the
full-width spatial-decay plot (`.plot()`, the Clause 6.2 regression on the
logarithmic distance axis with the 4 m read-off and the $r_D$ / $r_P$ crossings
marked), the boxed $D_{2,S}$ with the other quantities alongside, and a footer with the
fixed disclaimer. ISO 3382-3 **characterises** a space rather than defining an
intrinsic pass/fail, so a verdict row appears only when a target spatial decay
rate is supplied through the metadata's `requirement` field
(`ReportMetadata(requirement=...)`, read as the minimum acceptable $D_{2,S}$ in
dB, reflecting the informative quality ranges of Annex A where a larger spatial
decay is better; the room passes at or above it). It uses the same
`ReportMetadata` container as the [ISO 3382-1/-2 room-acoustics fiche](room-acoustics.md#iso-3382-report-report);
the open-plan-specific fields `area` (floor area), `source_positions` and
`receiver_positions` (the number of measurement positions) populate the header,
alongside `client`, `test_room`, `specimen`, `instrumentation`, `temperature`,
`relative_humidity`, `pressure`, `measurement_standard`, `test_date`,
`laboratory`, `operator`, `report_id` and `notes`. Passing `metadata=None`
produces a bare characterisation fiche. The fiche embeds the spatial-decay
chart, so rendering needs both reportlab and matplotlib
(`pip install "phonometry[report,plot]"`); only `engine="reportlab"` is
supported. The fiche renders in English by default; pass `language="es"` for a
Spanish fiche (translated fixed strings and a comma decimal separator).

```python
import numpy as np
from phonometry import room, ReportMetadata

r = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0])   # distances from the talker (m)
lp = 62.0 - 7.0 * np.log2(r)                           # A-weighted speech level (dB)
sti = 0.65 - 0.03 * r                                  # STI per position

result = room.open_plan_metrics(r, lp, sti)
result.report(
    "open_plan_fiche.pdf",
    metadata=ReportMetadata(
        test_room="Open-plan office B",
        specimen="Furnished, unoccupied, background noise present",
        area=420.0, source_positions=2, receiver_positions=7,
        measurement_standard="ISO 3382-3",
        temperature=22.0, relative_humidity=45.0,
        laboratory="Phonometry Reference Laboratory",
        requirement=7.0,          # adds a verdict against a target D2,S
    ),
)                                 # D2,S + Lp,A,S,4m, rD, rP and the decay curve
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![ISO 3382-3 open-plan office acoustics example report: metadata header, the metrics table of the four single-number quantities (D2,S = 7.0 dB per doubling, Lp,A,S,4m = 48.0 dB, rD = 5.0 m, rP = 15.0 m) above the spatial-decay plot on a logarithmic distance axis with the D2,S regression, the 4 m read-off and the rD and rP crossings, boxed D2,S and a PASS verdict against a 7.0 dB target](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3382_3_open_plan_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3382_3_open_plan_example.pdf)

*Open-plan office acoustics fiche (`OpenPlanResult.report`), $D_{2,S}$, $L_{p,A,S,4m}$, $r_D$, $r_P$ and the spatial-decay curve.*

## Crowd self-noise in an occupied room

ISO 3382-3 above **measures** an office. The complementary question at design
time is a **prediction**: in a room whose only noise source is its own
occupants, how loud does the background get, and does conversation still work?
Long (*Architectural Acoustics* 2e, Chapter 17, Equations (17.50) to (17.54))
sets the two competing levels at a listener:

$$
L_p(\text{signal}) = L_W + 10\log_{10}\!\left[\frac{Q}{4\pi r^2}\right]
\quad \text{(Eq. (17.50))}
$$

$$
L_p(\text{noise}) = L_W + 10\log_{10} N + 10\log_{10}\!\left[\frac{4}{N A_\text{tab}}\right]
\quad \text{(Eq. (17.51))}
$$

with $L_W$ the sound power level of one talker (about 70 dB in normal
conversation), $Q$ the talker's forward directivity (about 2), $r$ the
talker-to-listener distance and $A_\text{tab}$ the equivalent absorption area
**per occupied table**, so $N A_\text{tab}$ is the room's total absorption.

```python
from phonometry import room

# Long's hard restaurant: about 20 metric sabins, a talker at Lw = 70 dB,
# a listener 1.2 m away across the table.
print(round(room.speech_direct_level(1.2)))          # 60 dB direct field
print(round(room.crowd_noise_level(1, 20.0)))        # 63 dB with one talker
print(round(room.crowd_noise_level(20, 20.0)))       # 76 dB with 20 tables

# An alpha 0.9 ceiling over the 13.7 x 13.7 m room adds 170 metric sabins.
print(round(room.crowd_noise_level(20, 20.0 + 170.0)))   # 66 dB
```

Subtracting the two gives the **speech-to-noise ratio**, and this is the result
worth remembering:

$$
L_{SN} = 10\log_{10}\!\left[\frac{Q}{4\pi r^2}\right]
+ 10\log_{10}\!\left[\frac{A_\text{tab}}{4}\right]
\quad \text{(Eq. (17.52))}
$$

Neither $L_W$ nor $N$ survives. A busier room is not intrinsically worse,
because each new table brings both a talker and its own share of absorption.
What decides whether a restaurant works is the absorption **per table**, not
the absorption of the room. Requiring $L_{SN} > -6$ dB for adequate
cross-table communication at a separation $r_s$, and $L_{SN} < -9$ dB so a
neighbouring table $r_t$ away is not overheard, turns into a pair of design
bounds:

$$
A_\text{tab} > 6.31\,r_s^2 \quad \text{(Eq. (17.53))}, \qquad
A_\text{tab} < 3.16\,r_t^2 \quad \text{(Eq. (17.54))},
\qquad \text{both for } Q = 2
$$

> **The first constant is printed as 6.33.** The closed form above, at the same
> $Q = 2$ that yields Long's own second constant 3.16 exactly, gives 6.31, and his
> prose ("6.3 or more square metres, 68 sq ft") converts to 6.317 m². The library
> computes both bounds from the speech-to-noise ratio rather than hardcoding
> either constant, so they stay mutually consistent; the misprint is recorded in
> the [errata registry](ERRATA.md).

```python
from phonometry import room

print(round(room.absorption_per_table(1.0, -6.0), 2))   # 6.31 m2 per table at 1 m
print(round(room.absorption_per_table(2.5, -9.0), 1))   # 19.8 m2 at 2.5 m spacing

crowd = room.crowd_noise([20.0, 95.0, 190.0], distance=1.2)
crowd.plot()      # self-noise vs occupancy, against the -6 dB limit
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/restaurant_crowd_noise_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/restaurant_crowd_noise.svg" alt="Self-generated crowd noise level against the number of simultaneous talkers for three room absorption areas of 20, 95 and 190 metric sabins, with the 60 dB direct speech level at 1.2 m and the 66 dB communication limit drawn as horizontal references" width="85%"></picture>

*The same 20 talkers in three rooms. The hard room (20 m² of absorption)
crosses the communication limit before the fourth table is occupied; adding the
absorptive ceiling of Long's example (190 m² in total) keeps all 20 tables below
it. The curves are parallel because occupancy always costs $10\log_{10} N$: only
the vertical offset, which is the absorption, is a design variable.*

### `crowd_noise()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `absorption_areas` | float or 1D array | m² | > 0 | Total room absorption areas to compare |
| `talkers` | 1D array, optional | — | ≥ 1, default 1..20 | Occupancy axis $N$ |
| `distance` | float | m | > 0, default 1.2 | Talker-to-listener distance $r$ |
| `sound_power_level` | float | dB re 1 pW | default 70 | Talker $L_W$ |
| `directivity` | float | — | > 0, default 2 | Talker $Q$ |

Returns a `CrowdNoiseResult` (`talkers`, `absorption_areas`, `levels`,
`signal_level`, `communication_level`, `speech_to_noise()`) with `.plot()`. The
pieces `speech_direct_level`, `crowd_noise_level`, `speech_to_noise_ratio` and
`absorption_per_table` are callable directly.

This is a *design* model and deliberately does not model the Lombard reflex: it
cancels out of `L_SN` as long as everyone raises their voice equally, so the
model explains why the level spirals upward in a hard room rather than
predicting where it stops. It also assumes a diffuse reverberant field and a
single absorption figure per table, so it says nothing about screens, local
absorption or where in the room a given table sits.

## References

- International Organization for Standardization. (2012). *Acoustics —
  Measurement of room acoustic parameters — Part 3: Open plan offices*
  (ISO 3382-3:2012).
  [iso.org catalogue](https://www.iso.org/standard/46520.html).
  The open-plan speech-privacy quantities this page implements.
- Long, M. (2014). *Architectural acoustics* (2nd ed.). Academic Press.
  [doi:10.1016/C2012-0-03257-5](https://doi.org/10.1016/C2012-0-03257-5).
  Chapter 17 (the crowd self-noise model, Equations (17.50) to (17.54)).

## Standards

ISO 3382-3:2012 (open-plan office speech metrics: the spatial decay rate
$D_{2,S}$, the A-weighted speech level at 4 m, and the distraction and
privacy distances from STI). Validated against the standard's own quantity
definitions in the [conformance report](CONFORMANCE.md).

## See also

- [Room Acoustics](room-acoustics.md): the ISO 3382-1/2 decay parameters of
  the same room.
- [Measuring the Room Impulse Response](room-impulse-response.md): the
  acquisition behind the per-position levels.
- [Speech Transmission Index](speech-transmission.md): the STI/STIPA
  measurement that feeds the per-position `sti_values`.
- [Speech Intelligibility Index](speech-intelligibility.md): the ANSI S3.5
  view of the same intelligibility question.
- [Levels](levels.md): the A-weighted levels behind the spatial decay.
- API reference: [`room.open_plan`](https://jmrplens.github.io/phonometry/reference/api/rooms/open-plan/).
