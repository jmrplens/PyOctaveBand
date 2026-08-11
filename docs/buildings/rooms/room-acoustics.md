← [Documentation index](../../README.md)

# Room Acoustics

Room acoustics starts from one measurement: the **impulse response** (IR)
between a source and a receiver. Filter it into bands and integrate it, and
it yields reverberation time, clarity and centre time: everything
ISO 3382-1/2 asks about the sound field inside a single room. This page
turns a measured IR into those parameters: the Schroeder decay curve, the
evaluation windows, the validity flags and the accredited fiche. Acquiring
the IR itself (sweeps, MLS, where to measure) lives in
[Measuring the Room Impulse Response](room-impulse-response.md), and the
open-plan speech metrics of ISO 3382-3 in
[Open-Plan Office Acoustics](open-plan-acoustics.md). Two separate decay
measurements in a reverberation room, one with the room empty and one with
the specimen installed, yield a material's sound absorption; that method
(ISO 354) lives in
[Sound Absorption Measurement and Rating](../../materials/absorbers/absorption-measurement.md). For
sound insulation *between* spaces (the same IR measured either side of a
partition) see the companion
[Field Insulation Measurement (ISO 16283) guide](../insulation/insulation-field.md).

## Decay analysis and room parameters (ISO 3382-1/2)

The IR is filtered into octave (or one-third-octave) bands and each band is
turned into a **decay curve** by Schroeder backward integration of the
squared IR:

$$
E(t) = \int_t^{\infty} p^2(\tau)\ d\tau, \qquad
L(t) = 10 \log_{10} \frac{E(t)}{E(0)}\ \text{dB}.
$$

Integrating *backwards* removes the fluctuation that plagues a raw squared
IR and yields a smooth curve whose slope is the decay rate. Background
noise would make $E(t)$ level off, so integration is truncated where the
fitted decay line crosses the noise floor and the missing tail is
compensated assuming an exponential decay.

Reverberation times come from a least-squares line fit over an evaluation
range, extrapolated to a full 60 dB drop, $T = -60/\text{slope}$:
**EDT** over 0 to −10 dB (perceived reverberance), **T20** over −5 to −25 dB
and **T30** over −5 to −35 dB. Energy splits at an early/late boundary give
**clarity** and **definition**,

$$
C_{te} = 10 \log_{10} \frac{\int_0^{te} p^2\ dt}{\int_{te}^{\infty} p^2\ dt}\ \text{dB}, \qquad
D_{50} = \frac{\int_0^{0.05} p^2\ dt}{\int_0^{\infty} p^2\ dt},
$$

with $te = 50$ ms → C50 (speech) and $te = 80$ ms → C80 (music), plus the
**centre time** $T_s = \int t\ p^2\ dt / \int p^2\ dt$. Each parameter has a
**just-noticeable difference** (ISO 3382-1 Table A.1: EDT 5 %, C80 1 dB,
D50 0.05, Ts 10 ms) that sets how precisely it is worth reporting.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_schroeder_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_schroeder.gif" alt="Animation: the tail energy of the squared impulse response fills from the end while the backward integral advances toward t = 0, and the Schroeder decay curve emerges on a companion axis ending with the T20 and T30 regression lines" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_schroeder.webm)

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/schroeder_decay_dark.webp"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/schroeder_decay.webp" alt="Squared impulse response with its Schroeder backward-integrated decay curve, and the EDT, T20 and T30 regression windows marked" width="80%"></picture>

*The jagged squared IR (grey) integrates to the smooth Schroeder curve
(blue); the EDT, T20 and T30 windows are fitted on that curve and each
extrapolated to a 60 dB decay.*

```python
import numpy as np
from phonometry import room

fs = 48000
# Single-slope decay with T = 1 s: p^2 = exp(-13.8155 t)  (60/ln(10)/13.8155 = 1)
t = np.arange(fs) / fs
# ir: measured room impulse response; a synthetic single-slope decay stands in here.
ir = np.concatenate([np.zeros(10), np.exp(-13.8155 * t / 2.0)])

time, level = room.decay_curve(ir, fs)                    # Schroeder curve (0 dB at t = 0)

res = room.room_parameters(ir, fs, limits=None)           # broadband single band
print(round(float(res.t30[0]), 2))                   # 1.0  s
print(round(float(res.c80[0]), 2))                   # 3.05 dB
print(round(float(res.d50[0]), 3))                   # 0.499
print(round(float(res.ts[0]) * 1000, 0))             # 72 ms

# Octave bands 125 Hz - 4 kHz (ISO 3382-1 default); use fraction=3 for thirds
octaves = room.room_parameters(ir, fs)
print(octaves.frequency)                             # ~[126, 251, 501, 1000, 1995, 3981]
print(octaves.t30_valid)                             # per-band dynamic-range flags

octaves.plot()               # per-band EDT/T20/T30 + C50/C80 bars (needs matplotlib)
room.decay_curve(ir, fs).plot()   # Schroeder decay with EDT/T20/T30 fit overlays
```

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import room

fs = 48000
# Single-slope decay with T = 1 s: p^2 = exp(-13.8155 t)  (60/ln(10)/13.8155 = 1)
t = np.arange(fs) / fs
# ir: measured room impulse response; a synthetic single-slope decay stands in here.
ir = np.concatenate([np.zeros(10), np.exp(-13.8155 * t / 2.0)])
time, level = room.decay_curve(ir, fs)                    # Schroeder curve (0 dB at t = 0)

# One line — Schroeder decay with the EDT/T20/T30 straight-line fits:
decay = room.decay_curve(ir, fs)          # a DecayCurve (still unpacks as time, level)
decay.plot()
plt.show()

# By hand, the decay is just the Schroeder curve; mark the evaluation levels:
fig, ax = plt.subplots()
ax.plot(time, level, color="#1f77b4", label="Schroeder decay")
for db in (-5.0, -25.0, -35.0):      # T20 / T30 evaluation-window edges
    ax.axhline(db, ls=":", alpha=0.4)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Level re steady state [dB]")
ax.set_ylim(top=3.0)
ax.legend()
plt.show()
```

</details>

For this single-slope decay EDT, T20 and T30 all return ≈ 1.0 s, and the
energy parameters match their closed forms (C80 = 3.05 dB, D50 = 0.499,
Ts = 72 ms). A real room has a steeper early slope, so EDT < T30.

On a real, frequency-dependent decay the per-band `.plot()` is the working
summary of the whole measurement: the decay times as grouped bars per octave
(invalid bands hatched) over a second panel with C50 and C80.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/room_parameters_bands_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/room_parameters_bands.svg" alt="ISO 3382 per-band parameters of a synthetic room impulse response: grouped EDT, T20 and T30 bars per octave band falling from about 1.4 s at 125 Hz to 0.7 s at 4 kHz, over a second panel where C50 and C80 rise with frequency" width="92%"></picture>

*A room whose reverberation time falls from 1.4 s at 125 Hz to 0.7 s at
4 kHz, the typical signature of a furnished room whose absorption grows with
frequency; clarity rises as the decay shortens.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from phonometry import room

# A synthetic room IR with a frequency-dependent decay: octave-band noise
# carriers whose T60 falls from 1.4 s at 125 Hz to 0.7 s at 4 kHz.
fs = 48000
rng = np.random.default_rng(3382)
t = np.arange(int(1.6 * fs)) / fs
ir = np.zeros_like(t)
for fc, t60 in [(125.0, 1.4), (250.0, 1.25), (500.0, 1.1),
                (1000.0, 1.0), (2000.0, 0.85), (4000.0, 0.7)]:
    sos = signal.butter(4, [fc / np.sqrt(2), fc * np.sqrt(2)],
                        btype="bandpass", fs=fs, output="sos")
    carrier = signal.sosfilt(sos, rng.standard_normal(t.size))
    ir += carrier * np.exp(-3.0 * np.log(10.0) / t60 * t)

# One line: per-band EDT/T20/T30 bars + C50/C80 (needs matplotlib).
octaves = room.room_parameters(ir, fs)
octaves.plot()
plt.show()

# By hand: the T30 spectrum from the result's fields.
fig, ax = plt.subplots()
ax.bar(np.arange(octaves.t30.size), octaves.t30)
ax.set_xticks(np.arange(octaves.t30.size))
ax.set_xticklabels([f"{f:g}" for f in octaves.frequency])
ax.set_xlabel("Octave-band centre frequency [Hz]")
ax.set_ylabel("T30 [s]")
plt.show()
```

</details>

**Reading EDT, T20 and T30 against each other.** The three times
extrapolate the same 60 dB decay from different windows, so their
disagreement carries information:

- **T20 ≈ T30** (curvature below 10 %): the decay is close to a single
  straight slope over both evaluation windows, which is consistent with
  (though not proof of) a diffuse field, and either time can stand for
  "the" reverberation time of the band.
- **T30 > T20** (curvature above 10 %): the decay sags, with late energy
  decaying more slowly than early energy. The usual causes are coupled
  volumes (an open door to a corridor or stairwell, a deep balcony or a
  stage house feeding energy back) and strongly uneven absorption that
  leaves one room axis reverberant. No single number describes such a
  decay: report both windows together with the curvature, and treat the
  [statistical predictions](reverberation-prediction.md) with suspicion,
  because their diffuse-field assumption has visibly failed.
- **EDT far from T20/T30**: EDT is fitted where the direct sound and the
  first reflections still dominate, so it varies from seat to seat while
  T30 barely moves. EDT below T30 means the position receives strong early
  energy (close to the source, under a reflector): the room sounds drier
  there than its T30 suggests, because perceived reverberance follows EDT.
  EDT above T30 at one seat points to an echo or a focusing surface
  concentrating late energy there.

**How much decay the noise floor allows.** A fit window is only as good as
the decay range underneath it. The **impulse-to-noise ratio** (INR) is the
level distance between the peak of the band-filtered IR and its noise
floor; the fit window plus a safety margin must fit inside it, which is the
ISO 3382 requirement of at least 35 dB of usable decay range for T20 and
45 dB for T30. An undersized range biases the fitted time upward, toward
the flat tail the noise floor imposes on the decay curve, and the bias
grows quietly before the fit visibly fails (Hak, Wenmaekers, & van
Luxemburg, 2012). `room_parameters` reports the per-band `dynamic_range`
and tightens the acceptance limits to 46 dB (T20) and 54 dB (T30) before
flagging a value valid, so the residual truncation-and-compensation bias of
a flagged-valid time stays inside the 5 % JND. When a band fails its flag,
the order of remedies is: use T20 instead of T30 (its window needs 10 dB
less range under the ISO minima, 8 dB under the tightened flags); raise the
INR at acquisition, since doubling the sweep length or
the number of synchronous averages buys 3 dB each time; and only then fall
back to EDT, never to a fit stretched into the noise.

Below the **Schroeder frequency** $f_s \approx 2000\sqrt{T/V}$ these
decay statistics stop telling the whole story: the field is ruled by
discrete **room modes**. The simulation below drives the same rigid
5 m by 3.5 m room at its (2,1) mode and then between two modes; on
resonance a standing-wave pattern with fixed nodal lines grows until it
dominates the RMS pressure map, off resonance the room still responds,
but the forced field stays weak and never organises into that (2,1)
nodal pattern.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_room_modes_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_room_modes.gif" alt="Animation: a 2D FDTD simulation of a 5 by 3.5 metre room driven at the 84 Hz (2,1) mode and at an off-mode frequency; on resonance a standing-wave pattern with fixed nodal lines grows to dominate the RMS pressure map, off resonance the forced response stays weak and disorganised" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_room_modes.webm)

### `room_parameters()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `ir` | 1D array | any | non-silent | Measured impulse response |
| `fs` | int | Hz | > 0 | Sample rate |
| `limits` | (float, float) or `None` | Hz | default `(125.0, 4000.0)` | Band-centre limits; `None` = broadband single band |
| `fraction` | int | — | `1` (octave, default) / `3` (third) | Bandwidth fraction |
| `zero_phase` | bool | — | default `False` | Forward-backward octave filtering (ISO 3382-2 §7.3 NOTE, which relaxes $BT > 16$ to $BT > 4$); removes the filter group delay before the backward integration and roughly halves the 125 Hz short-decay T30 bias (~+4.9 % → +2.4 % at $T$ = 0.2 s). `decay_curve` accepts it too |

Returns a `RoomAcousticsResult`: `frequency` (band centres, or `None`
broadband), `edt`/`t20`/`t30` (s), `c50`/`c80` (dB), `d50`, `ts` (s),
`dynamic_range` (dB), the `edt_valid`/`t20_valid`/`t30_valid` flags (ISO
3382-1 §5.3.3: noise ≥ 25 dB below the peak for EDT, tightened to 46 dB for
T20 and 54 dB for T30 so the tail-compensation bias of a flagged-valid value
stays within the 5 % JND) and `curvature`
$C = 100\ (T_{30}/T_{20} - 1)$ % (values above 10 % flag a non-straight
decay). `decay_curve(ir, fs, band=None, fraction=1, zero_phase=False)` returns
just the `(time, level)` curve for one band or the broadband response.

ISO 3382-1 Annex A defines more parameters than this: the sound strength $G$
and the binaural/spatial parameters LF, LFC and IACC are not implemented —
phonometry computes only the reverberance and clarity family above (EDT,
T20, T30, C50, C80, D50, Ts).

### ISO 3382 report (`.report()`)

`RoomAcousticsResult.report(path)` renders a one-page PDF fiche laid out like a
room-acoustics measurement report (a performance space per ISO 3382-1:2009 or
an ordinary room per ISO 3382-2:2008, both evaluated by the integrated
impulse-response method): a standard-basis line, an optional metadata header
block, the full-width per-band parameter table ($T_{20}$, $T_{30}$, EDT,
$C_{50}$, $C_{80}$, $D_{50}$, $T_s$) above the result's own per-band decay-time
plot (`.plot()`), the boxed mid-frequency reverberation time $T_\text{mid}$
(the mean of the 500 Hz and 1000 Hz octave $T_{30}$) with the mid-frequency EDT
alongside, and a footer with the fixed disclaimer. ISO 3382-1/-2 are
characterisation standards with no intrinsic pass/fail, so a verdict row appears
only when a target mid-frequency reverberation time is supplied through the
metadata's `requirement` field (`ReportMetadata(requirement=...)`, read as the
maximum acceptable $T_\text{mid}$); a broadband result has no 500 Hz / 1000 Hz
octaves to average, so its box and verdict fall back to the plain broadband
$T_{30}$ with no "500-1000 Hz" claim. It uses the same
`ReportMetadata` container as the [ISO 11654 absorption fiche](../../materials/absorbers/absorption-measurement.md#iso-11654-report-report);
the room-specific fields `room_volume`, `source_positions` and
`receiver_positions` populate the header (ISO 3382 requires the room volume and
the number of source and microphone positions to be reported), alongside
`test_room`, `specimen`, `area`, `instrumentation`, `temperature`,
`relative_humidity`, `pressure`, `measurement_standard`, `test_date`,
`laboratory`, `operator`, `report_id` and `notes`. Passing `metadata=None`
produces a bare characterisation fiche. Rendering needs reportlab and, for the
figure the fiche embeds, matplotlib (`pip install "phonometry[report,plot]"`);
only `engine="reportlab"` is supported. The fiche renders in English by default;
pass `language="es"` for a Spanish fiche (translated fixed strings and a comma
decimal separator).

```python
from phonometry import room, ReportMetadata

result = room.room_parameters(ir, fs)   # octave bands 125 Hz - 4 kHz
result.report(
    "room_fiche.pdf",
    metadata=ReportMetadata(
        specimen="Small auditorium, unoccupied, fully furnished",
        test_room="Auditorium A",
        room_volume=2830.0, area=340.0,
        source_positions=2, receiver_positions=8,
        measurement_standard="ISO 3382-1",
        temperature=21.0, relative_humidity=45.0,
        laboratory="Phonometry Reference Laboratory",
        requirement=1.3,           # adds a verdict against a target T_mid
    ),
)                                  # T_mid + the per-band parameter table
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![ISO 3382 room acoustic parameters example report: metadata header, the octave-band parameter table (T20, T30, EDT, C50, C80, D50, Ts from 125 Hz to 4 kHz) above the per-band decay-time bar plot, boxed mid-frequency T_mid = 1.15 s and a PASS verdict against a 1.3 s target](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3382_room_acoustics_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3382_room_acoustics_example.pdf)

*Room acoustic parameters fiche (`RoomAcousticsResult.report`), $T_\text{mid}$ and the per-band table.*

## The rest of the measurement chain

Two sections that used to live on this page have guides of their own: the
acquisition of the impulse response, with the ISO 18233 sweeps and MLS and
the placement rules, is
[Measuring the Room Impulse Response](room-impulse-response.md); and the
ISO 3382-3 open-plan speech metrics, with their fiche, are
[Open-Plan Office Acoustics](open-plan-acoustics.md).

## See also

- [Measuring the Room Impulse Response](room-impulse-response.md): the
  ISO 18233 acquisition of the IR this page analyses.
- [Open-Plan Office Acoustics (ISO 3382-3)](open-plan-acoustics.md): the
  speech-privacy quantities of open-plan offices.
- [Sound Absorption Measurement and Rating](../../materials/absorbers/absorption-measurement.md): the
  ISO 354 reverberation-room absorption measurement (`absorption_area`,
  `absorption_coefficient`, `measure_sound_absorption`) that consumes the
  reverberation times `room_parameters` returns, and its ISO 11654 rating.
- [Field](../insulation/insulation-field.md), [laboratory](../insulation/insulation-lab.md) and
  [predicted](../design/insulation-prediction.md) sound insulation: field,
  laboratory and predicted sound insulation between spaces, and its measurement uncertainty.
- [Sound Power](../../devices/emission/sound-power.md): the $L_W$ methods that consume the
  ISO 354 absorption area (the ISO 3744 $K_2$ and the ISO 3741 absorption term).
- [Loudness](../../perception/psychoacoustics/loudness.md) and [Sound Quality Metrics](../../perception/psychoacoustics/sound-quality.md): loudness,
  sharpness and the other perception metrics of what the room delivers.
- [Filter Banks](../../signals/filters/filter-banks.md): the IEC 61260 fractional-octave filters
  used for band decay curves and insulation spectra.
- [Theory](../../reference/theory/rooms-buildings.md): Schroeder integration, regression windows and the
  reference-curve derivation.
- API reference: [`room.acoustics`](https://jmrplens.github.io/phonometry/reference/api/rooms/acoustics/).

## Quick answers

### How are EDT, T20 and T30 defined?

Each band of the impulse response is turned into a decay curve by Schroeder backward integration of the squared IR, and a least-squares line fitted over an evaluation range is extrapolated to a full 60 dB drop, $T = -60/\text{slope}$ (ISO 3382-1/2): EDT over 0 to −10 dB (perceived reverberance), T20 over −5 to −25 dB and T30 over −5 to −35 dB.

### How much decay range do I need for a valid T20 or T30?

The fit window plus a safety margin must fit inside the impulse-to-noise ratio, the level distance between the band-filtered IR peak and its noise floor: ISO 3382 requires at least 35 dB of usable decay range for T20 and 45 dB for T30. An undersized range biases the fitted time upward, so `room_parameters` tightens its validity flags to 46 dB and 54 dB, keeping the bias inside the 5 % JND.

## References

- Kuttruff, H. (2016). *Room acoustics* (6th ed.). CRC Press.
  [doi:10.1201/9781315372150](https://doi.org/10.1201/9781315372150).
  The reference monograph behind this page: the statistical theory of
  decaying sound fields, the Schroeder frequency and the perceptual room
  parameters of this page.
- Schroeder, M. R. (1965). New method of measuring reverberation time.
  *The Journal of the Acoustical Society of America*, 37(3), 409-412.
  [doi:10.1121/1.1909343](https://doi.org/10.1121/1.1909343).
  The backward-integration method that turns the squared impulse response
  into the smooth decay curve of this page.
- Hak, C. C. J. M., Wenmaekers, R. H. C., & van Luxemburg, L. C. J. (2012).
  Measuring room impulse responses: Impact of the decay range on derived
  room acoustic parameters. *Acta Acustica united with Acustica*, 98(6),
  907-915. [doi:10.3813/aaa.918574](https://doi.org/10.3813/aaa.918574).
  The INR analysis behind the dynamic-range discussion and the tightened
  validity flags of this page.
- International Organization for Standardization. (2009). *Acoustics —
  Measurement of room acoustic parameters — Part 1: Performance spaces*
  (ISO 3382-1:2009).
  [iso.org catalogue](https://www.iso.org/standard/40979.html).
  The parameter definitions, position requirements and just-noticeable
  differences of this page.
- International Organization for Standardization. (2008). *Acoustics —
  Measurement of room acoustic parameters — Part 2: Reverberation time in
  ordinary rooms* (ISO 3382-2:2008).
  [iso.org catalogue](https://www.iso.org/standard/36201.html).
  The accuracy grades and the dynamic-range criterion behind the decay
  analysis; its position rules are applied in the
  [room-impulse-response guide](room-impulse-response.md).

## Standards

ISO 3382-1:2009 and ISO 3382-2:2008 (reverberation time and room parameters
from the Schroeder decay). The impulse-response acquisition (ISO 18233)
lives in [Measuring the Room Impulse Response](room-impulse-response.md)
and the open-plan metrics (ISO 3382-3) in
[Open-Plan Office Acoustics](open-plan-acoustics.md). Validated against
closed-form decays and the standards' own parameter definitions in the
[conformance report](../../CONFORMANCE.md).
