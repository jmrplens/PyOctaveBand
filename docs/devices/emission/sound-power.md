← [Documentation index](../../README.md)

# Sound Power

Sound *pressure* depends on where you stand and on the room you stand in;
sound **power** does not. The sound power level $L_W$ is the total acoustic
energy per second a source radiates, referenced to $P_0 = 1\ \text{pW}$, and it is
the device-independent **emission** descriptor that goes on a datasheet,
feeds a room prediction (EN 12354) or is checked against a noise-emission
limit. phonometry implements eight standardised routes to it, split across
six guides: an enveloping *pressure* surface in the field
(ISO 3744/3746) and the precision grade in an *anechoic room* (ISO 3745),
covered in [Sound Power by Pressure Methods](sound-power-pressure.md); the
diffuse field of a *reverberation room* (ISO 3741), covered in
[Sound Power in the Reverberation Room](sound-power-reverberation.md); the
same comparison against a reference sound source taken to the room the
machine works in (ISO 3747), covered in
[Sound Power in Situ by Comparison](sound-power-in-situ.md);
*intensity* read at discrete points over a surface (ISO 9614-1), covered in
[Sound Intensity (p-p)](intensity.md); the same intensity *scanned* over that
surface (ISO 9614-2), with its precision
counterpart (ISO 9614-3), covered in
[Sound Power by Intensity Scanning](sound-power-intensity.md); and the
*surface velocity* of the machine's own casing (ISO/TS 7849-1 and -2), the one
route that needs no acoustic measurement at all, covered in
[Sound power from surface vibration](vibration-sound-power.md). This page is
the front door: how to choose among them, what the accuracy grades actually
promise, and how a measured $L_W$ becomes the ISO 4871 noise-emission
declaration a datasheet prints.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_power_two_rooms_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_power_two_rooms.gif" alt="Animation: the same source in an anechoic room and in a reverberation room produces different microphone pressures, and the free-field and diffuse-field formulas converge to the same sound power level L_W" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_power_two_rooms.webm)

## Choosing a method

All deliver the same quantity for a source that runs steadily, a per-band
$L_W$ and an A-weighted total $L_{W\mathrm{A}}$, but under different
environments, accuracy grades and practical constraints. A source that emits in
bursts has no steady power to report, and the in situ route determines its
sound energy level $L_J$ instead (clause 8.5).

| Method | Standard | Measured quantity | Environment | Accuracy grade | Use when |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Enveloping surface | **ISO 3744** (engineering) / **ISO 3746** (survey) | Sound pressure on a hemisphere or box | Essentially free field over one or more reflecting planes | Grade 2 ($\sigma_{\mathrm{R}0} \approx 1.5\ \text{dB}$) / grade 3 ($\approx 3.0\ \text{dB}$) | In situ or a large room; no special test facility available |
| Reverberation room | **ISO 3741** | Sound pressure in the diffuse field | Qualified hard-walled reverberation room | Grade 1 (precision) | Highest accuracy for steady, broadband sources in a lab |
| In situ comparison | **ISO 3747** | Sound pressure of the source and of a calibrated reference sound source at the same positions | The reverberant part of the room the machine works in ($\Delta L_f \ge 7$ dB) | Grade 2 ($\sigma_{\mathrm{R}0} \approx 1.5\ \text{dB}$) / grade 3 ($\approx 4.0\ \text{dB}$) | A machine that cannot leave its installation, in a room too reverberant for an enveloping surface |
| Intensity at discrete points | **ISO 9614-1** | Normal sound intensity held at each of $N$ points, one per segment | Almost any, tolerant of steady extraneous noise | Grade 1 or 2 per band from the Annex B criteria; grade 3 on the A-weighted total | On-site where the probe stands still at each point rather than sweeping the surface |
| Intensity scanning | **ISO 9614-2** | Normal sound intensity scanned over a surface | Almost any, tolerant of steady extraneous noise | Grade 2 / 3 (from per-band field indicators) | On-site with background noise, or one machine among many |
| Anechoic room | **ISO 3745** | Sound pressure on a fixed microphone array | Qualified anechoic or hemi-anechoic room | Grade 1 (precision) | Reference-grade emission in a free-field laboratory |
| Precision intensity scanning | **ISO 9614-3** | Scanned normal intensity, tighter criteria | Almost any, tolerant of steady extraneous noise | Grade 1 (precision) | Precision on-site, with the ISO 9614-3 field-indicator checks |
| Surface vibration | **ISO/TS 7849-1** (survey) / **-2** (engineering) | Surface-averaged velocity level and a radiation factor | Any; no acoustic measurement | Upper limit ($\varepsilon = 1$) / engineering | The machine cannot be quietened, enclosed or approached with a microphone |

The pressure methods correct the surface level for the room ($K_2$) and for
background noise ($K_1$); the reverberation method needs a *qualified* room
but reaches precision grade; intensity rejects steady background energy at
the cost of a two-microphone probe and a per-band validity check; the
surface-velocity route abandons the microphone altogether and pays for it with
a radiation factor. Each method guide walks its routes in turn.

The plate below draws seven of the eight: the in situ comparison shares the
sound pressure row's algebra with ISO 3741 and is drawn on its own page.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_power_methods_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_power_methods.svg" alt="The seven sound power routes, one row for each measured quantity, every cell carrying the same five attributes. The sound pressure row: the ISO 3744/3746 hemispherical enveloping surface over a reflecting plane, the ISO 3745 fixed microphone array in a wedge-lined anechoic room, and the ISO 3741 diffuse field of a reverberation room. The sound intensity row: the ISO 9614-1 measurement surface cut into ten segments with the probe held still at a point in each, the ISO 9614-2 serpentine scan swept over that same surface, and the tighter ISO 9614-3 precision scan. The surface velocity row carries one route across the full width, ISO/TS 7849 with an accelerometer on a radiating casing and no microphone at all" width="92%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import emission

# One steady source (octave-band LW below) determined by three routes.
freqs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
lw_true = np.array([85.0, 88.0, 90.0, 89.0, 86.0, 82.0])

# ISO 3744: SPL at 10 positions on a hemisphere, r = 2 m (10 lg(S/S0) = 14 dB).
pres = emission.sound_power_pressure(np.tile(lw_true - 14.0, (10, 1)),
                                     "hemisphere", radius=2.0,
                                     frequencies=freqs)
# ISO 9614-2: uniform normal intensity scanned over six 0.5 m2 segments.
# In this free field Lp reads the intensity level, and the probe's
# delta_pI0 = 18 dB makes the clause 10.6 b screening evaluable.
i_n = np.tile(10.0 ** (lw_true / 10.0) * 1e-12 / 3.0, (6, 1))
lp = np.tile(lw_true - 10.0 * np.log10(3.0), (6, 1))
inten = emission.sound_power_intensity(i_n, np.full(6, 0.5),
                                       pressure_levels=lp,
                                       pressure_residual_index=18.0,
                                       frequencies=freqs, band_type="octave")
# ISO 3741: comparison against a reference source of known LW = 84 dB per band.
comp = emission.sound_power_comparison(lw_true - 20.0, np.full(6, 64.0),
                                       np.full(6, 84.0), frequencies=freqs)

fig, ax = plt.subplots()
for res, style, ms, label in ((pres, "-o", 11, "pressure (ISO 3744)"),
                              (inten, "--s", 8, "intensity (ISO 9614-2)"),
                              (comp, ":^", 5, "reference source (ISO 3741)")):
    ax.semilogx(freqs, res.sound_power_level, style, markersize=ms,
                label=f"{label}: LWA = {res.sound_power_level_a:.1f} dB")
ax.set(xlabel="Frequency [Hz]", ylabel="Sound power level LW [dB]")
ax.legend()
plt.show()
```

</details>

### A decision path

The table compresses into a short sequence of questions (ISO 3740 dedicates
its Table 3 and Annex D to exactly this decision). Work through them in
order; the first match names the standard.

1. **What is the number for?** A datasheet declaration or a limit check
   normally asks for engineering grade (grade 2, the preferred grade for
   noise declarations); a reference source, a product ranking or a dispute
   calls for precision (grade 1); a first walk-through of a noisy plant
   tolerates survey grade (grade 3). Grade 1 exists only in a qualified
   laboratory room (ISO 3741, ISO 3745) or via the precision intensity
   methods, both of which are here: ISO 9614-1 at discrete points in
   [Sound Intensity (p-p)](intensity.md), ISO 9614-3 by scanning in
   [Sound Power by Intensity Scanning](sound-power-intensity.md).
2. **Can the source travel to a laboratory?** ISO 3741 wants the source
   small next to the room (volume no more than about 2 % of the room volume)
   and its noise steady; ISO 3745 wants it inside a qualified anechoic or
   hemi-anechoic room with a characteristic dimension below half the
   measurement radius, and it is the route that also yields directivity. A
   machine bolted to its foundation rules both out and leaves the in-situ
   methods.
3. **How quiet and how dry is the site?** ISO 3744 needs the background at
   least 6 dB below the source (preferably more than 15 dB) and
   $K_2 \le 4\ \text{dB}$. If only a
   3 dB margin or $K_2 \le 7\ \text{dB}$ can be met, the same microphones and formulae
   degrade gracefully to ISO 3746 at survey grade.
4. **Is the background the problem?** When neighbouring machines cannot be
   switched off, or the margin is outright negative, the pressure methods
   are out. Intensity (ISO 9614-1 at discrete points, ISO 9614-2 by scanning,
   ISO 9614-3 for grade 1 by scanning) tolerates steady extraneous noise even
   some 10 dB *above* the source, because only the net energy flux through the
   surface counts; the per-band field indicators then decide the grade
   actually achieved.
5. **Can you put a microphone there at all?** A machine that cannot be stopped,
   an environment that destroys a capsule, an enclosure with no room for a
   surface: when no acoustic measurement is possible, the surface-velocity route
   estimates the same quantity from accelerometers on the casing
   (ISO/TS 7849-1 and -2). It answers a subtly different question — it
   characterises what the *structure* radiates, and stays blind to sound
   escaping through openings, intakes and gaps — and it costs a radiation factor
   you must either assume (Part 1, an upper limit) or measure once (Part 2).

### What the accuracy grades mean

The grade is a claim about **reproducibility**: $\sigma_{\mathrm{R}0}$ is the standard
deviation you would see if different laboratories measured the same source,
each following the standard correctly. Typical A-weighted values are
$\sigma_{\mathrm{R}0} \approx 0.5\ \text{dB}$ for grade 1 (ISO 3741), 1.5 dB for grade 2 (ISO 3744, ISO 3747,
ISO 9614-2) and 3 dB or more for grade 3 (larger still when $K_2$ is
large or the spectrum is tonal). Per-band values are larger at the
spectrum edges. The `uncertainty` field of the pressure-method results
(enveloping surface and anechoic) is the expanded uncertainty
$U = 2\sigma_\text{tot}$ (95 % coverage), where
$\sigma_\text{tot} = \sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\text{omc}^2}$ also folds
in the operating/mounting instability
$\sigma_\text{omc}$ that you estimate and pass in; the grade only bounds the method's
share of the budget.

In practice: a grade-2 $L_{W\mathrm{A}}$ of 92.4 dB carries $U \approx 3\ \text{dB}$, so two grade-2
results 2 dB apart are statistically indistinguishable, and checking that
same source against a 93 dB limit is a coin flip. Choose the grade from the
decision the number has to support, not from the facility that happens to
be free.

## Declaring the noise emission (ISO 4871)

A measured sound power level is not yet a *declaration*. ISO 4871:1996 is the
standard for the noise-emission declaration a manufacturer prints in technical
documents: which quantities are stated, in which form, and how a declared value
is verified. The preferred quantity is the A-weighted sound power level
$L_{W\mathrm{A}}$, optionally accompanied by the A-weighted emission sound pressure level
$L_{p\mathrm{A}}$ at a work station.

A declaration takes one of two alternative forms (clause 4):

- the **dual-number** form (clause 3.16): the measured value $L_{W\mathrm{A}}$ and its
  uncertainty $K_{W\mathrm{A}}$ stated together but separately; and
- the **single-number** form (clause 3.15): the derived declared value
  $L_{W\mathrm{Ad}} = L_{W\mathrm{A}} + K_{W\mathrm{A}}$, an upper limit that repeated measurements are unlikely
  to exceed at the stated confidence level.

$K_{W\mathrm{A}}$ combines the measurement (reproducibility) and, for a batch, the
production spread; for a single machine $K = 1.645\,\sigma_\mathrm{R}$ (Annex A.2.2). A
`NoiseEmissionDeclaration` holds one or more per-operating-mode declarations
and renders the ISO 4871 fiche through `.report()`. The quickest route is to
`declare()` straight from a measured sound power:

```python
import numpy as np
from phonometry import emission
from phonometry import ReportMetadata

# ISO 3744: SPL at 10 positions on a hemisphere, r = 1 m (10 lg(S/S0) = 8 dB).
levels = np.tile(lw_true - 8.0, (10, 1))
result = emission.sound_power_pressure(levels, "hemisphere", radius=1.0,
                                 frequencies=freqs)

declaration = result.declare(
    uncertainty=2.0,                 # K_WA in dB (defaults to the expanded U)
    machine="Type 990, Model 11-TC",
    operating_conditions="50 Hz, 230 V, rated load",
    basic_standards="ISO 3744",
    verification_level=result.sound_power_level_a,  # L_1 for clause 6.2
)
declaration.report(
    "iso4871.pdf",
    metadata=ReportMetadata(measurement_standard="ISO 3744"),
)   # -> L_WAd = L_WA + K_WA, verified when L_1 <= L_WAd
```

Or build the declaration directly, reproducing the ISO 4871 Annex B example
(two operating modes, $L_{W\mathrm{A}} = 88$ and 95 dB with $K_{W\mathrm{A}} = 2$ dB, giving
declared $L_{W\mathrm{Ad}} = 90$ and 97 dB):

```python
mode1 = emission.OperatingModeDeclaration(
    "Operating mode 1", sound_power_level=88.0, sound_power_uncertainty=2.0,
    emission_pressure_level=78.0, emission_pressure_uncertainty=2.0,
    verification_level=89.0,          # passes: 89 <= 90
)
mode2 = emission.OperatingModeDeclaration(
    "Operating mode 2", sound_power_level=95.0, sound_power_uncertainty=2.0,
    emission_pressure_level=86.0, emission_pressure_uncertainty=2.0,
    verification_level=98.0,          # fails: 98 > 97
)
emission.NoiseEmissionDeclaration(
    (mode1, mode2), machine="Type 990, Model 11-TC",
    basic_standards=("ISO 3744", "ISO 11202"), form="dual-number",
).report("iso4871.pdf")
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![ISO 4871 noise emission declaration example report: a header with the machine identification and operating conditions, the declared dual-number table across two operating-mode columns listing the measured A-weighted sound power level L_WA, its uncertainty K_WA, the emission sound pressure level L_pA and the derived declared value L_WAd = L_WA + K_WA (90 and 97 dB), the noise-test-code and basic-standards footnote, and a clause 6.2 verification table where mode 1 passes and mode 2 fails](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso4871_declaration_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso4871_declaration_example.pdf)

*Noise emission declaration fiche (`NoiseEmissionDeclaration.report`), the
ISO 4871 Annex B dual-number table with the declared $L_{W\mathrm{Ad}} = L_{W\mathrm{A}} + K_{W\mathrm{A}}$ and
the clause 6.2 verification verdict.*


## See also

- [Sound Power by Pressure Methods (ISO 3744 / ISO 3746 / ISO 3745)](sound-power-pressure.md):
  the enveloping surface with its $K_1$/$K_2$ corrections, and the precision
  anechoic array.
- [Sound Power in the Reverberation Room (ISO 3741)](sound-power-reverberation.md):
  the precision diffuse-field method with the Waterhouse and meteorological
  corrections.
- [Sound Power in Situ by Comparison (ISO 3747)](sound-power-in-situ.md):
  the reference-source comparison where the machine works, with the sound
  energy level of an impulsive source.
- [Sound Power by Intensity Scanning (ISO 9614-2 / ISO 9614-3)](sound-power-intensity.md):
  the routes that tolerate steady background noise, qualified by their field
  indicators.
- [Sound power from surface vibration (ISO/TS 7849)](vibration-sound-power.md):
  the surface-velocity route, for a machine no microphone can approach.
- [Sound Intensity (p-p)](intensity.md): the two-microphone probe behind every
  intensity route, and the ISO 9614-1 determination at discrete points.
- [Room Acoustics](../../buildings/rooms/room-acoustics.md): the reverberation time and equivalent
  absorption area that feed the room corrections.
- [Levels](../../signals/levels/levels.md): energy averaging and the A-weighting behind $L_{W\mathrm{A}}$.
- [Theory](../../reference/theory/environment-transport.md): the Waterhouse, $K_1$/$K_2$ and $C_1$/$C_2$ derivations.
- API reference: [`emission.sound_power`](https://jmrplens.github.io/phonometry/reference/api/power/sound-power/), [`emission.sound_power_reverberation`](https://jmrplens.github.io/phonometry/reference/api/power/sound-power-reverberation/) and [`emission.sound_power_intensity`](https://jmrplens.github.io/phonometry/reference/api/power/sound-power-intensity/).

## Quick answers

### What is the difference between sound power and sound pressure?

Sound pressure depends on where you stand and on the room; sound power does
not. The sound power level $L_W$ is the total acoustic energy per second a
source radiates, referenced to $P_0 = 1\ \text{pW}$, and it is the device-independent
emission descriptor that goes on a datasheet or is checked against a
noise-emission limit; ISO 3744, ISO 3741, ISO 3747, ISO 9614-1, ISO 9614-2,
ISO 3745, ISO 9614-3 and ISO/TS 7849 all determine it.

### What do the accuracy grades in sound power measurement mean?

The grade is a claim about reproducibility: $\sigma_{\mathrm{R}0}$ is the standard deviation
you would see if different laboratories measured the same source, each
following the standard correctly. Typical A-weighted values are
$\sigma_{\mathrm{R}0} \approx 0.5\ \text{dB}$ for grade 1 (ISO 3741), 1.5 dB for grade 2 (ISO 3744, ISO 3747,
ISO 9614-2) and 3 dB or more for grade 3. A grade-2 $L_{W\mathrm{A}}$ carries
$U \approx 3\ \text{dB}$, so two grade-2 results 2 dB apart are statistically
indistinguishable.

### How do I measure sound power when background noise cannot be switched off?

Use intensity: ISO 9614-1 at discrete points, ISO 9614-2 by scanning (grade 2
or 3) or ISO 9614-3 by precision scanning (grade 1). Because sound intensity
is the net energy flux through the measurement surface, steady extraneous
noise even some 10 dB above the source is tolerated, whereas the ISO 3744
pressure method needs the background at least 6 dB below the source. The
per-band field indicators then decide the grade actually achieved.


## References

- Beranek, L. L., & Mellow, T. J. (2012). *Acoustics: Sound fields and
  transducers*. Academic Press. ISBN 978-0-12-391421-7.
  [doi:10.1016/C2011-0-05897-0](https://doi.org/10.1016/C2011-0-05897-0).
  Radiation and sound fields: the free-field and diffuse-field relations
  between pressure and power that the whole method family rests on.
- International Organization for Standardization. (2019). *Acoustics —
  Determination of sound power levels of noise sources — Guidelines for the
  use of basic standards* (ISO 3740:2019).
  [iso.org catalogue](https://www.iso.org/standard/45107.html).
  The selection guide behind "Choosing a method": grades, environments,
  source-size and background criteria for the whole family.
- International Organization for Standardization. (1996). *Acoustics —
  Declaration and verification of noise emission values of machinery and
  equipment* (ISO 4871:1996).
  [iso.org catalogue](https://www.iso.org/standard/10868.html).
  The declaration section: the dual/single-number forms,
  $L_{W\mathrm{Ad}} = L_{W\mathrm{A}} + K_{W\mathrm{A}}$ (clause 3.15) and the clause 6.2 verification.

## Standards

ISO 3740:2019, *Acoustics — Determination of sound power levels of noise
sources — Guidelines for the use of basic standards*: the Table 3 / Annex D
selection guidance condensed into the decision path of this page.
ISO 4871:1996, *Acoustics — Declaration and verification of noise emission
values of machinery and equipment*: the dual-number and single-number
declaration forms, the declared value $L_{W\mathrm{Ad}} = L_{W\mathrm{A}} + K_{W\mathrm{A}}$ and the
clause 6.2 verification. The basic determination standards (ISO 3744/3746,
ISO 3741, ISO 3745, ISO 3747, ISO 9614-1/-2/-3, ISO/TS 7849-1/-2) are covered
in their method guides.

**Not covered.** Two members of the ISO 3740 family are not implemented at
all, ISO 3743-1 and ISO 3743-2. The sound *energy* level $L_J$ of a single
event is covered for the rest, each on its own page: for ISO 3744 and ISO 3746
on the [pressure-methods](sound-power-pressure.md#4-sound-energy-level-of-a-burst-clause-83)
page, for ISO 3741 on the
[reverberation-room](sound-power-reverberation.md#3-sound-energy-level-of-a-single-event-clause-92)
page, and for ISO 3747 in
[Sound Power in Situ by Comparison](sound-power-in-situ.md). ISO 3745 defines
none. The emission sound pressure level $L_{p\mathrm{A}}$
that stands beside $L_{W\mathrm{A}}$ in a declaration is consumed here, never
determined: ISO 11201, ISO 11202 and ISO 11204 are outside the library. Of
ISO 4871, only the clause 6.2 single-machine verification is evaluated; the
batch criteria of clause 6.3 are not, and the batch statistics beyond the
single-machine $K = 1{,}645\,\sigma_\mathrm{R}$ case of Annex A.2.2 are stated rather
than derived.

