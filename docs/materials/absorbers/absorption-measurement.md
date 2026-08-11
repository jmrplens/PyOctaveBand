← [Documentation index](../../README.md)

# Sound Absorption Measurement and Rating

Of all the numbers a materials laboratory produces, the sound-absorption
coefficient is the one that travels furthest: it leaves the reverberation room
on a datasheet, enters a Sabine estimate, an EN 12354-6 absorption budget or a
public-tender requirement, and is rarely questioned again. This guide covers
that number's whole laboratory life. The **measurement** is ISO 354: a sample
of 10 m² to 12 m² on the floor of a reverberation room, two decay times, and
Sabine's formula run backwards to get the random-incidence coefficient
$\alpha_s$ per one-third-octave band. The **rating** is ISO 11654: the spectrum
collapsed into the weighted coefficient $\alpha_w$ with its letter class A to
E, the single number absorber datasheets quote. And because a coefficient
without an uncertainty is only half a result, ISO 12999-2 supplies the
**measurement uncertainty** of both. The normal-incidence counterpart measured
on small samples lives in the [impedance tube](impedance-tube.md) guide; the
closing section here explains when each of the two is the right tool.

## 1. Reverberation-room measurement (ISO 354)

The reverberation-room measurement itself is `measure_sound_absorption`. It
takes the one-third-octave reverberation time of the empty room ($T_1$) and of
the room with the specimen installed ($T_2$), the room volume $V$ and the
specimen area $S$, and returns a frozen `SoundAbsorptionMeasurement`. The
equivalent sound absorption areas follow from Sabine's equation (ISO 354:2003
Eq. (5)/(7)), $A = 55.3\,V/(c\,T) - 4\,V\,m$, with the speed of sound from
Eq. (6), $c = 331 + 0.6\,t$; the sound absorption coefficient is
$\alpha_s = (A_2 - A_1)/S$ (Eq. (8)/(9)). The coefficient may exceed 1.0 from
edge and diffraction effects (Clause 3.7 NOTE 2) and is never clamped. Air
attenuation enters only through the per-band coefficient $m$ (default 0, the
zero-attenuation reference); pass `attenuation_from_alpha` of an ISO 9613-1
value when it is needed.

ISO 354 is a characterisation: it produces the $\alpha_s$ spectrum, not a
single-number rating. The weighted coefficient $\alpha_w$ is an ISO 11654
quantity; feed the measured $\alpha_s$ to `weighted_absorption_from_third_octave`
in section 3 to obtain it.

```python
import numpy as np
from phonometry import materials

freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000], float)
t_empty = np.array([9.0, 9.0, 8.8, 8.6, 8.4, 8.2, 8.0, 7.8, 7.5, 7.2,
                    6.9, 6.6, 6.2, 5.8, 5.4, 5.0, 4.6, 4.2])
t_specimen = np.array([8.4, 8.2, 7.7, 7.2, 6.5, 5.7, 4.9, 4.2, 3.6, 3.15,
                       2.85, 2.65, 2.55, 2.5, 2.55, 2.6, 2.7, 2.85])

m = materials.measure_sound_absorption(
    freqs, t_empty, t_specimen, volume=200.0, area=10.8, temperature=20.0
)
print(m.alpha_s[7])   # alpha_s at 500 Hz: 0.328...
m.plot()              # alpha_s versus one-third-octave frequency
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_absorption_measurement_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_absorption_measurement.svg" alt="ISO 354 reverberation-room sound absorption: the alpha_s spectrum of a porous absorber sample over the one-third-octave bands from 100 Hz to 5000 Hz, rising from near zero at low frequency to a broad maximum around 0.69 near 1600 Hz and easing off above" width="80%"></picture>

*The Sabine inversion of the two decay times: the porous sample barely
changes the long low-frequency decays ($\alpha_s \to 0$), bites hardest where
its thickness approaches a quarter wavelength, and eases off at high
frequency, a typical thin-porous-absorber signature (Cox & D'Antonio 3e,
Ch. 5).*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000], float)
t_empty = np.array([9.0, 9.0, 8.8, 8.6, 8.4, 8.2, 8.0, 7.8, 7.5, 7.2,
                    6.9, 6.6, 6.2, 5.8, 5.4, 5.0, 4.6, 4.2])
t_specimen = np.array([8.4, 8.2, 7.7, 7.2, 6.5, 5.7, 4.9, 4.2, 3.6, 3.15,
                       2.85, 2.65, 2.55, 2.5, 2.55, 2.6, 2.7, 2.85])
m = materials.measure_sound_absorption(
    freqs, t_empty, t_specimen, volume=200.0, area=10.8, temperature=20.0
)

# One line: the alpha_s spectrum over the one-third-octave band axis.
m.plot()
plt.show()

# By hand, from the result's fields:
fig, ax = plt.subplots()
ax.plot(np.arange(freqs.size), m.alpha_s, "o-")
ax.set_xticks(np.arange(freqs.size))
ax.set_xticklabels([f"{f:g}" if f < 1000 else f"{f/1000:g}k" for f in freqs],
                   rotation=45)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Sound absorption coefficient alpha_s")
plt.show()
```

</details>

### ISO 354 report (`.report()`)

`SoundAbsorptionMeasurement.report(path)` renders a one-page PDF fiche laid out
like an accredited reverberation-room absorption test report (ISO 354:2003): a
standard-basis line, a metadata header block, the one-third-octave $\alpha_s$
table beside the $\alpha_s$ curve (the result's own `.plot()`), a boxed
characterisation headline and a footer with the fixed disclaimer. ISO 354 has
no pass/fail verdict and no single-number rating, so the fiche carries neither.
Setting `verbose=True` adds the reverberation times $T_1$/$T_2$ and the
equivalent absorption areas $A_1$/$A_2$ to the table.

It uses the same `ReportMetadata` container and rendering engine as the other
fiches. The specimen area $S$, room volume $V$, speed of sound $c$, temperature
and humidity are taken from the measurement result (they drove the Sabine
inversion); the descriptive `ReportMetadata` fields that apply here are
`client`, `manufacturer`, `specimen`, `mounting`, `test_room`, `test_date`,
`pressure`, `measurement_standard`, `laboratory`, `operator`, `report_id` and
`notes`. The `requirement` field is ignored (ISO 354 has no verdict). Rendering
needs reportlab and, for the figure the fiche embeds, matplotlib (`pip install
"phonometry[report,plot]"`); only `engine="reportlab"` is supported. The fiche
renders in English by default; pass `language="es"` for a Spanish fiche
(translated fixed strings and a comma decimal separator).

```python
from phonometry import materials, ReportMetadata

m = materials.measure_sound_absorption(
    freqs, t_empty, t_specimen, volume=200.0, area=10.8,
    temperature=20.0, humidity=54.0,
)
m.report(
    "alpha_s_fiche.pdf",
    metadata=ReportMetadata(
        specimen="50 mm porous absorber over a 100 mm air gap",
        mounting="Type A (against a rigid wall)",
        measurement_standard="ISO 354",
        test_room="Reverberation room R1",
        laboratory="Phonometry Reference Laboratory",
    ),
)                                  # one-third-octave alpha_s, 100 Hz to 5000 Hz
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![ISO 354 absorption example report: metadata header with sample area S, room volume V and speed of sound c, the one-third-octave alpha_s table grouped by octave beside the alpha_s curve, and the boxed characterisation headline over the tested one-third-octave range](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso354_absorption_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso354_absorption_example.pdf)

*Reverberation-room absorption fiche (`SoundAbsorptionMeasurement.report`), the $\alpha_s$ spectrum.*


## 2. Absorption areas from room decay times

The equivalent absorption area $A$ that drives $R'$, $L'_n$, the ISO 3744 $K_2$
environmental correction and the ISO 3741 absorption term is itself measured in
a reverberation room (ISO 354).
Measure the room's reverberation time **empty** ($T_1$) and again **with the
test specimen installed** ($T_2$); the specimen's absorption is the difference
of the two Sabine areas, and dividing by the covered area gives the absorption
coefficient:

$$
A = \frac{55.3\ V}{c\ T} - 4 V m, \qquad
\alpha_s = \frac{A_2 - A_1}{S}, \qquad c = 331 + 0.6\ t ,
$$

with $c$ from the room air temperature $t$ in °C (valid 15–30 °C) and $m$ the
power attenuation coefficient of air (default 0; convert an ISO 9613-1
$\alpha$ in dB/m with `attenuation_from_alpha`). Because edge and diffraction
effects can scatter more energy than the sample's flat area intercepts,
$\alpha_s$ may exceed 1.0 and is never clamped (ISO 354 Clause 3.7).

```python
import numpy as np
from phonometry import materials

# Third-octave reverberation times of a 200 m^3 room, empty (T1) and with a
# 10.8 m^2 absorber sample installed (T2).
t1 = np.array([5.0, 4.0, 3.0])
t2 = np.array([3.0, 2.5, 2.0])

a_empty = materials.absorption_area(t1, volume=200.0, temperature=20.0)
print(np.round(a_empty, 2))                    # [ 6.45  8.06 10.75] m^2

alpha = materials.absorption_coefficient(t1, t2, volume=200.0, sample_area=10.8,
                               temperature1=20.0)
print(np.round(alpha, 3))                      # [0.398 0.448 0.498]
```

$T_1$ and $T_2$ are exactly the reverberation times
[`room_parameters`](../../buildings/rooms/room-acoustics.md) returns, so an ISO 3382-2 decay
measurement of the empty and treated room flows straight into
`absorption_coefficient`.

Each of those two numbers is read off a decay, and the clip below shows how
one is read: the squared impulse response is integrated backwards from the
tail, the Schroeder curve emerges, and the T20 and T30 regressions are fitted
to a straight portion of it. That is the operation behind $T_1$, and again
behind $T_2$ — the clip shows a *single* room, not the pair, so it answers
"where does one $T$ come from" and not "what does subtracting two of them
cost". The second question is the one that governs this measurement, and
section 4 puts a number on it: because $\alpha_s$ is a difference of two
reciprocal decay times, its uncertainty is worst exactly where the two decays
are most alike, at the low-frequency end.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_schroeder_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_schroeder.gif" alt="Animation: the tail energy of a squared impulse response filling from the end while the backward integral advances toward t = 0, the Schroeder decay curve emerging on a companion axis and ending with the T20 and T30 regression lines" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_schroeder.webm)

A room volume below the 150 m³ minimum or a
sample area outside 10–12 m² raises an advisory `AbsorptionWarning`; the result
still returns. That advisory pair is the only check made: the rest of the
Clause 6 / Annex A room qualification — the room-shape rule, the ceiling on the
empty room's absorption area $A_1$, the diffusivity qualification and the
microphone and loudspeaker counts — is the laboratory's to verify, and nothing
here checks any of it.

### `absorption_area()` / `absorption_coefficient()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `t60` / `t1`, `t2` | 1D array | s | > 0 | Reverberation time(s); `t1` empty, `t2` with specimen |
| `volume` | float | m³ | > 0 | Room volume $V$ (advisory below 150 m³) |
| `sample_area` | float | m² | > 0 | Area $S$ the specimen covers (coefficient only) |
| `temperature` / `temperature1`, `temperature2` | float | °C | default `20.0`, 15–30 | Sets $c$ via Eq. (6); `temperature2` defaults to `temperature1` |
| `speed_of_sound` (`…1`, `…2`) | float, optional | m/s | > 0 | Overrides the temperature-derived $c$ |
| `m` (`m1`, `m2`) | float or 1D array | 1/m | ≥ 0, default `0` | Air power attenuation coefficient |

`absorption_area()` returns the equivalent absorption area $A$ (m²) with the
shape of `t60`; `absorption_coefficient()` returns $\alpha_s$;
`attenuation_from_alpha(alpha)` converts an ISO 9613-1 $\alpha$ (dB/m) to $m$.


## 3. Weighted rating and absorption class (ISO 11654)

The measurement of section 1 delivers the $\alpha_s$ spectrum in
one-third-octave bands. ISO 11654:1997 turns that spectrum into a
single-number rating comparable across products.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_iso11654_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_iso11654.svg" alt="ISO 11654 rating flow: measured alpha_s becomes practical alpha_p per octave band, the reference curve is shifted to best fit, alpha_w is read at 500 Hz with shape indicators, giving the absorption class A to E" width="82%"></picture>

**Practical absorption coefficient (Clause 4.1).** The one-third-octave data are
first grouped into octave bands, each the arithmetic mean of its three thirds:

$$
\alpha_{p,i} = \tfrac{1}{3}\big(\alpha_{i1} + \alpha_{i2} + \alpha_{i3}\big),
$$

evaluated to the second decimal and then rounded in steps of $0.05$ (the
Clause 4.1 NOTE fixes the rounding, e.g. $0.92 \to 0.90$); rounded means above
$1.00$ are set to $1.00$. The five rating bands are 250, 500, 1000, 2000 and
4000 Hz.

**Weighted absorption (Clause 4.2).** A fixed reference curve
$\{250{:}\,0.80,\ 500{:}\,1.00,\ 1000{:}\,1.00,\ 2000{:}\,1.00,\ 4000{:}\,0.90\}$
is shifted downwards, towards the measured $\alpha_p$, in steps of $0.05$ until
the sum of the **unfavourable** deviations (taken only where the measurement
lies below the shifted curve, with magnitude $(\text{curve} - \text{measured})$)
is no more than $0.10$. The weighted coefficient $\alpha_w$ is the shifted-curve
value read at 500 Hz.

**Shape indicators (Clause 4.3).** When a practical coefficient exceeds the
shifted curve by $0.25$ or more, a shape indicator is appended: `L` at 250 Hz,
`M` at 500 or 1000 Hz, `H` at 2000 or 4000 Hz (e.g. `0.60(M)`).

**Absorption class (Table B.1).** Finally $\alpha_w$ maps to a class: A
(0.90–1.00), B (0.80–0.85), C (0.60–0.75), D (0.30–0.55), E (0.15–0.25), or "not
classified" (0.00–0.10). Because $\alpha_w$ is always a multiple of $0.05$ these
ranges partition the grid exactly.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/absorption_rating_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/absorption_rating.svg" alt="ISO 11654 weighted sound absorption rating: the practical absorption spectrum plotted against the shifted reference curve over 250 Hz to 4000 Hz, with the unfavourable deviation at 250 Hz shaded and the weighted coefficient alpha_w read at 500 Hz" width="80%"></picture>

*The Annex A.2 worked example: the reference curve is shifted down by 0.40 until
the unfavourable deviations sum to 0.05 ($\le 0.10$), giving $\alpha_w = 0.60$;
the 500 Hz peak overshoots the shifted curve by $\ge 0.25$, adding the `M`
indicator, so the rating is $0.60(\text{M})$, class C.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import materials

# ISO 11654 Annex A.2 practical coefficients at 250/500/1000/2000/4000 Hz
result = materials.weighted_absorption([0.35, 1.00, 0.65, 0.60, 0.55])
result.plot()   # practical curve vs shifted reference, deviations shaded
plt.show()
```

</details>

```python
from phonometry import materials

# ISO 11654 Annex A.2 practical coefficients at 250/500/1000/2000/4000 Hz
alpha_p = [0.35, 1.00, 0.65, 0.60, 0.55]

result = materials.weighted_absorption(alpha_p)
print(result.rating_label)          # 0.60(M)
print(result.alpha_w)               # 0.6
print(result.absorption_class)      # C
print(round(result.unfavourable_sum, 2))  # 0.05
result.plot()   # the figure above: practical curve vs shifted reference

# A bare alpha_w also maps straight to its class (Table B.1)
print(materials.absorption_class(0.85))       # B
```

`weighted_absorption` accepts the five octave-band $\alpha_p$ values (as a
sequence or a `{frequency: value}` mapping); pass the fifteen one-third-octave
$\alpha_s$ values to `practical_absorption_coefficient` first if you are starting
from raw ISO 354 data. To keep the one-third-octave $\alpha_s$ on the result (so
the fiche can print the full table every accredited ISO 354 certificate carries),
rate it in one step with `weighted_absorption_from_third_octave(alpha_s)`, which
forms $\alpha_p$, rates it and retains the input $\alpha_s$ and its band centres
(`third_octave_alpha_s`, `third_octave_bands`). The result carries the shifted
reference curve and the per-band deviations, and its `.plot()` renders the figure
above.

```python
from phonometry import materials

# Fifteen one-third-octave alpha_s (200 Hz to 5000 Hz), as an ISO 354 report gives
alpha_s = [0.30, 0.35, 0.40, 1.00, 1.00, 1.00, 0.62, 0.66, 0.67,
           0.58, 0.60, 0.62, 0.53, 0.55, 0.57]
result = materials.weighted_absorption_from_third_octave(alpha_s)
print(result.rating_label)          # 0.60(M)
print(result.third_octave_alpha_s)  # the input alpha_s, retained for the fiche
```

### ISO 11654 report (`.report()`)

`AbsorptionRatingResult.report(path)` renders a one-page PDF fiche laid out like
an accredited absorption test report (an ISO 354 reverberation-room measurement
rated per ISO 11654): a standard-basis line, an optional metadata header block,
the octave-band $\alpha_p$ table beside the practical-versus-shifted-reference
plot (the result's own `.plot()`), the boxed $\alpha_w$ single number with its
absorption class and applied shift, an optional verdict row and a footer with
the fixed disclaimer. When the rating was built with
`weighted_absorption_from_third_octave`, the left table becomes the full ISO 354
one-third-octave $\alpha_s$ table with the octave $\alpha_p$ on the matching
rows, exactly as accredited certificates print it. It uses the same
`ReportMetadata` container (documented
under [Insulation ratings](../../buildings/insulation/insulation-ratings.md#report-metadata-reportmetadata))
and rendering engine as the ISO 717 insulation fiche; passing
`metadata=None` produces a lightweight prediction fiche, and a supplied
`requirement` is read as the minimum $\alpha_w$ for the PASS/FAIL verdict.
Setting `verbose=True` swaps the two-column table for the ISO 11654 evaluation
columns (practical coefficient, shifted reference, unfavourable deviation).
Rendering needs reportlab and, for the figure the fiche embeds, matplotlib (`pip
install "phonometry[report,plot]"`); only `engine="reportlab"` is supported. The
fiche renders in English by default; pass `language="es"` for a Spanish fiche
(translated fixed strings and a comma decimal separator), e.g.
`result.report("alpha_w_fiche_es.pdf", language="es")`.

```python
from phonometry import materials, ReportMetadata

# Rate from the fifteen one-third-octave alpha_s so the fiche prints the full
# ISO 354 table; materials.weighted_absorption([...]) also works from alpha_p.
alpha_s = [0.30, 0.35, 0.40, 1.00, 1.00, 1.00, 0.62, 0.66, 0.67,
           0.58, 0.60, 0.62, 0.53, 0.55, 0.57]
result = materials.weighted_absorption_from_third_octave(alpha_s)
result.report(
    "alpha_w_fiche.pdf",
    metadata=ReportMetadata(
        specimen="50 mm porous absorber over a 100 mm air gap",
        area=10.8, mounting="Type A (against a rigid wall)",
        measurement_standard="ISO 354",
        temperature=21.4, relative_humidity=54.0,
        laboratory="Phonometry Reference Laboratory",
        requirement=0.55,          # adds the PASS/FAIL verdict row
    ),
)                                  # alpha_w (shape) + absorption class
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![ISO 11654 absorption example report: metadata header, the full one-third-octave alpha_s table with the octave alpha_p on the matching rows beside the practical-versus-shifted-reference plot, boxed alpha_w = 0.60 (M) with absorption class C and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso11654_absorption_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso11654_absorption_example.pdf)

*Weighted absorption fiche (`AbsorptionRatingResult.report`), $\alpha_w$ with its class.*


## Tube or reverberation room?

The tube and the reverberation room both deliver a number called the
"absorption coefficient", and the two are routinely confused. They are
different physical quantities, measured under different sound fields, and they
do not match, sometimes not even closely.

The [tube](impedance-tube.md) measures a **normal-incidence** coefficient: one
plane wave, one angle, a specimen a few centimetres across, and a complex
reflection factor that keeps magnitude and phase. ISO 354 measures the **random-incidence**
coefficient $\alpha_s$: a diffuse field striking a sample of 10 m² to 12 m²
from every direction at once, recovered from the change in the room's decay
time through Sabine's formula, an energy average with no phase left in it.
Because the diffuse field finds more ways into an absorber than the single
normal-incidence wave (oblique waves travel a longer path inside the layer),
$\alpha_s$ usually comes out higher. For a locally reacting surface the two
are linked by Paris' angular average, which defines the statistical
absorption coefficient

$$
\alpha_{st} = \int_0^{\pi/2} \alpha(\theta)\,\sin 2\theta\,\mathrm{d}\theta,
$$

an integral that weights the oblique angles most heavily; evaluating it
needs the
angle-dependent $\alpha(\theta)$ from the measured surface impedance, not just
the normal-incidence coefficient itself.

**Why ISO 354 values exceed 1.** A ratio of absorbed to incident energy cannot
exceed one, yet reverberation-room reports of $\alpha_s = 1.05$ to $1.20$ for
thick porous absorbers are routine and correct by the method. Sabine's formula
converts the decay-time change into an equivalent absorption area $A$, and
$\alpha_s = A/S$ divides by the *geometric* sample area $S$. Diffraction at
the sample edges lets the specimen drain energy from a sound field wider than
its footprint (the edge effect), so the equivalent area can exceed the
geometric one. Such values are not errors, but they are not portable either:
they depend on the sample size and perimeter, which is precisely why ISO 354
fixes both. The ISO 11654 rating simply truncates: practical coefficients
above $1.00$ are set to $1.00$ (section 3). Prediction inputs get no such
silent clipping in this library: each
[reverberation-time estimator](../../buildings/rooms/reverberation-prediction.md) enforces its own
mathematical domain, so Sabine and Eyring accept ISO 354 values at or above
one as supplied (Eyring as long as the mean absorption stays below one),
while Millington-Sette rejects them (its per-surface logarithm diverges at
one) and any adjustment below one is left to the caller. The
equivalent-absorption-area budget of
[EN 12354-6](../../buildings/rooms/enclosed-space-absorption.md) likewise accepts the coefficients
as supplied.

**Which to use.** They answer different questions. The reverberation-room
value is the one that feeds diffuse-field prediction: Sabine reverberation
estimates, the equivalent absorption areas of
[EN 12354-6](../../buildings/rooms/enclosed-space-absorption.md) and the $\alpha_w$ rating and
class, all of which expect random incidence over a mounted, finite sample
(ISO 354's Annex B mounting types exist because the mounting is part of the
result). The tube value is the laboratory and development tool: it needs only
a few square centimetres of material, resolves magnitude *and* phase, and its
surface impedance pins down the parameters of porous-material models in the
Allard and Atalla tradition, with the
[airflow resistivity](airflow-resistance.md) as the first input; the fitted model then predicts the layer at any angle, thickness
or backing. What the tube number is *not* is a drop-in substitute for
$\alpha_s$: feeding normal-incidence coefficients into a Sabine or EN 12354-6
budget systematically underpredicts the installed absorption.


## 4. Sound-absorption measurement uncertainty (ISO 12999-2)

A rated absorption coefficient means little without its uncertainty. ISO
12999-2:2020 gives the standard uncertainty $u$ of the quantities produced by a
reverberation-room measurement (ISO 354) and its ratings (ISO 11654, EN 1793-1),
estimated from inter-laboratory tests to ISO 5725. It is the sound-absorption
companion of the sound-insulation uncertainty of ISO 12999-1
([Field Insulation Measurement (ISO 16283)](../../buildings/insulation/insulation-field.md)).

**One-third-octave bands (Clause 5).** For the sound-absorption coefficient the
reproducibility standard deviation is $\sigma_R = m\,\alpha_s + n$ (Formula (1)),
and for the equivalent absorption area $\sigma_R = m\,A_T + n\,S$ with
$S = 10\ \text{m}^2$ (Formula (2)), where $m$ and $n$ are the frequency-dependent
constants of Table 1 (63–5000 Hz). The repeatability value is
$\sigma_r = 0.6\,\sigma_R$ (Formula (3)).

**Practical coefficient (Clause 6).** For the ISO 11654 practical coefficient
$\sigma_R = m\,\alpha_p + n$ in octave bands with the constants of Table 2
(250–4000 Hz); again $\sigma_r = 0.6\,\sigma_R$.

**Single numbers (Clause 7).** The weighted coefficient $\alpha_w$ has a constant
standard uncertainty ($\sigma_R = 0.035$, $\sigma_r = 0.020$); the EN 1793-1
single-number rating $DL_{\alpha,\text{NRD}}$ scales with the value
($\sigma_R = 0.10\,DL_\alpha$, $\sigma_r = 0.02\,DL_\alpha$). That uncertainty
formula is all this page takes from EN 1793-1: the standard's own in-situ
measurement method for road-traffic noise-reducing devices is not implemented.

**Reporting (Clause 8).** The expanded uncertainty is $U = k\,u$ (Formula (10))
with the Table 3 coverage factor $k$ ($k = 2.0$ at 95 %, Gaussian). The reported
$U$ is rounded to two decimals for absorption coefficients and one decimal for
the equivalent area and $DL_{\alpha,\text{NRD}}$.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/absorption_uncertainty_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/absorption_uncertainty.svg" alt="ISO 12999-2 sound absorption coefficient uncertainty: the measured alpha_s spectrum over one-third-octave bands from 63 Hz to 5000 Hz with a shaded plus-or-minus U band at coverage factor k = 2, reproducing the standard's worked Table 4 example" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import materials

# ISO 12999-2 Table 4 worked example: alpha_s per one-third-octave band.
freqs = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
         630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
alpha_s = [0.33, 0.35, 0.39, 0.38, 0.37, 0.36, 0.36, 0.36, 0.43, 0.49,
           0.58, 0.63, 0.68, 0.71, 0.73, 0.75, 0.77, 0.79, 0.81, 0.81]
result = materials.sound_absorption_coefficient_uncertainty(alpha_s, freqs, confidence=0.95)
result.plot()   # alpha_s with the +/-U (k = 2) reproducibility ribbon
plt.show()
```
</details>

```python
from phonometry import materials

# Reproducibility uncertainty of alpha_s at 1000 Hz (Table 1: m=0.040, n=0.015).
r = materials.sound_absorption_coefficient_uncertainty([0.68], [1000], confidence=0.95)
print(round(float(r.standard_uncertainty[0]), 4))            # 0.0422 (sigma_R)
print(float(r.reported_expanded_uncertainty[0]))             # 0.08  (U, k=2)
r.plot()   # the figure above: alpha_s with its +/-U (k = 2) ribbon

# Single-number ratings (Clause 7 worked examples).
print(float(materials.weighted_coefficient_uncertainty(0.70).reported_expanded_uncertainty[0]))  # 0.07
print(float(materials.single_number_rating_uncertainty(8.1).reported_expanded_uncertainty[0]))   # 1.6
```


## See also

- [Impedance Tube](impedance-tube.md): the normal-incidence absorption,
  surface impedance and transmission loss of small samples (ISO 10534-1/-2,
  ASTM E2611), and why its coefficient differs from $\alpha_s$.
- [Airflow Resistance](airflow-resistance.md): the ISO 9053-1/-2 flow
  resistivity that anchors the porous-material models behind most absorbers.
- [Porous and Multilayer Absorbers](porous-absorbers.md): predicting the
  absorption of a construction from material parameters before it is built.
- [Room Acoustics](../../buildings/rooms/room-acoustics.md): the ISO 3382 decay measurements that
  supply the reverberation times $T_1$ and $T_2$.
- [Reverberation-time prediction](../../buildings/rooms/reverberation-prediction.md): the Sabine,
  Eyring and Millington-Sette estimators that consume $\alpha_s$.
- [Equivalent absorption area of furnished rooms](../../buildings/rooms/enclosed-space-absorption.md):
  the EN 12354-6 absorption budget these coefficients feed.
- [Field Insulation Measurement (ISO 16283)](../../buildings/insulation/insulation-field.md): the
  sound-insulation companion uncertainty standard, ISO 12999-1.
- API reference: [`materials.absorbers.sound_absorption`](https://jmrplens.github.io/phonometry/reference/api/materials/sound-absorption/), [`materials.absorbers.rating`](https://jmrplens.github.io/phonometry/reference/api/materials/rating/) and [`materials.absorbers.uncertainty`](https://jmrplens.github.io/phonometry/reference/api/materials/uncertainty/).

## References

- Cox, T. J., & D'Antonio, P. (2017). *Acoustic absorbers and diffusers:
  Theory, design and application* (3rd ed.). CRC Press.
  ISBN 978-1-4987-4099-9.
  [doi:10.1201/9781315369211](https://doi.org/10.1201/9781315369211).
  Absorber measurement in practice: the room method, mountings, and the
  edge-effect discussion behind the tube-or-reverberation-room section.
- International Organization for Standardization. (2003). *Acoustics —
  Measurement of sound absorption in a reverberation room* (ISO 354:2003).
  [iso.org catalogue](https://www.iso.org/standard/34545.html).
  The reverberation-room method of section 1, including the Annex B specimen
  mountings.
- International Organization for Standardization. (1997). *Acoustics — Sound
  absorbers for use in buildings — Rating of sound absorption*
  (ISO 11654:1997).
  [iso.org catalogue](https://www.iso.org/standard/19583.html).
  The weighted rating of section 3: the practical coefficient, the shifted
  reference curve, the shape indicators and the absorption class.
- International Organization for Standardization. (2020). *Acoustics —
  Determination and application of measurement uncertainties in building
  acoustics — Part 2: Sound absorption* (ISO 12999-2:2020).
  [iso.org catalogue](https://www.iso.org/standard/68749.html).
  The measurement uncertainty of section 4, validated against the standard's
  own worked examples (Tables 4/5).

## Standards

ISO 354:2003 (sound absorption in a reverberation room); ISO 11654:1997
(weighted sound-absorption rating); ISO 12999-2:2020 (sound-absorption
measurement uncertainty). Every equation is derived from the standard text;
the [conformance report](../../CONFORMANCE.md) validates the library against the
standards' own worked examples (ISO 11654 Annex A, ISO 12999-2 Tables 4/5)
and closed-form identities.
