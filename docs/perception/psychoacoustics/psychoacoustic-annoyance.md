← [Documentation index](../../README.md)

# Psychoacoustic annoyance and fluctuation strength (Fastl & Zwicker)

How *annoying* a sound is depends on more than how loud it is. Fastl & Zwicker,
*Psychoacoustics: Facts and Models*, combine four psychoacoustic sensations
(**loudness**, **sharpness**, **roughness** and **fluctuation strength**) into a
single **psychoacoustic annoyance** $PA$, a scalar that grows with loudness and
is lifted further when the sound is sharp, rough or slowly fluctuating. This page
covers the exact $PA$ model (Eqs 16.2–16.4), the **fluctuation strength** it
consumes, both the closed form for amplitude-modulated broadband noise
(Eq. 10.2) and the Osses et al. (2016) signal model, and the signal
convenience that derives all four sensations from a recording.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/psychoacoustic_annoyance_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/psychoacoustic_annoyance.svg" alt="Psychoacoustic annoyance PA against percentile loudness N5 for three sensation profiles: a neutral baseline where PA equals N5, a sharp sound and a rough-and-fluctuating sound both lifted above the baseline, with the worked example PA = 37.05 marked" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import psychoacoustics

n5 = np.linspace(4.0, 60.0, 200)
profiles = [
    ("Baseline: S = 1.75 acum, F = R = 0", 1.75, 0.0, 0.0),
    ("Sharp: S = 3.5 acum", 3.5, 0.0, 0.0),
    ("Rough + fluctuating: F = 1.2 vacil, R = 0.7 asper", 2.0, 1.2, 0.7),
]
fig, ax = plt.subplots()
for label, s, f, r in profiles:
    pa = [psychoacoustics.psychoacoustic_annoyance(v, s, f, r).annoyance for v in n5]
    ax.plot(n5, pa, label=label)

ex = psychoacoustics.psychoacoustic_annoyance(30.0, 2.0, 0.5, 0.3)
ax.plot([30.0], [ex.annoyance], "o", label=f"Worked example (PA = {ex.annoyance:.2f})")
ax.set_xlabel("Percentile loudness N5 [sone]")
ax.set_ylabel("Psychoacoustic annoyance PA")
ax.legend()
plt.show()
```

</details>

## 1. The four sensations

Psychoacoustic annoyance rests on four hearing sensations, each with its own
model and unit in the library:

- **Loudness**: the percentile loudness $N_5$ (the loudness exceeded 5 % of the
  time), in **sone**, from the ISO 532-1 Zwicker time-varying model
  (`loudness_zwicker`, see [Loudness](loudness.md)).
- **Sharpness** $S$, in **acum**: the spectral balance towards high
  frequencies, DIN 45692 (`sharpness_din`).
- **Roughness** $R$, in **asper**: the harshness of fast (~70 Hz) amplitude
  modulation, ECMA-418-2 Sottek model (`roughness_ecma`).
- **Fluctuation strength** $F$, in **vacil**: the sensation of slow (~4 Hz)
  loudness fluctuation (§3 below).

## 2. Psychoacoustic annoyance (Eqs 16.2–16.4)

The exact model (Fastl & Zwicker Eq. 16.2; origin Widmann 1992) scales $N_5$
by a factor that grows with the sharpness weighting $w_S$ and the combined
roughness/fluctuation weighting $w_{FR}$:

$$
PA = N_5\left(1 + \sqrt{w_S^2 + w_{FR}^2}\right), \qquad
w_S = (S - 1.75)\,0.25\,\log_{10}(N_5 + 10)\ \ (S > 1.75\ \mathrm{acum}), \qquad
w_{FR} = \frac{2.18}{N_5^{0.4}}\,(0.4\,F + 0.6\,R).
$$

$w_S$ is zero for $S \le 1.75\ \text{acum}$ (sharpness only adds annoyance
above that threshold); $w_{FR}$ weights roughness more heavily than
fluctuation strength
(0.6 vs 0.4). `psychoacoustic_annoyance` takes the four quantities directly and
returns the annoyance together with the two intermediate weightings:

```python
from phonometry import psychoacoustics

res = psychoacoustics.psychoacoustic_annoyance(30.0, 2.0, 0.5, 0.3)   # N5, S, F, R
print(round(res.annoyance, 4))   # 37.0478
print(round(res.w_s, 4), round(res.w_fr, 4))   # 0.1001 0.2125

res.plot()   # PA beside its wS and wFR weightings (needs matplotlib)
```

The figure above sweeps $PA$ against $N_5$ for three profiles: a neutral
baseline ($S = 1.75\ \text{acum}$, $F = R = 0$, so $PA = N_5$), a sharp sound
and a
rough-and-fluctuating sound, both lifted above the baseline, with the worked
example marked.

The model is small enough to draw whole: two weightings and one combination.
The diagram traces the worked example through it, sensation by sensation.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_psychoacoustic_annoyance_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_psychoacoustic_annoyance.svg" alt="Block diagram of the Fastl and Zwicker psychoacoustic annoyance model: the four sensations of the worked example, sharpness of 2.0 acum, percentile loudness N5 of 30 sone, fluctuation strength of 0.5 vacil and roughness of 0.3 asper, feed two weightings, the sharpness weighting wS of 0.1001 that is zero below 1.75 acum and the roughness-and-fluctuation weighting wFR of 0.2125 that weighs roughness 0.6 against 0.4, and combine into PA equal to N5 times one plus the root of wS squared plus wFR squared, giving 37.05; a closing note recalls that a neutral sound sits on the baseline PA equal to N5" width="92%"></picture>

### 2.1 From a signal (engineering estimate)

`psychoacoustic_annoyance_from_signal` is a convenience that derives all four
sensations from a calibrated pressure signal and combines them: $N_5$ from the
ISO 532-1 Zwicker time-varying loudness, $S$ from DIN 45692 sharpness, $R$
from the ECMA-418-2 Sottek roughness and $F$ from the fluctuation-strength
signal
model.

```python
from phonometry import psychoacoustics

res = psychoacoustics.psychoacoustic_annoyance_from_signal(x, fs, field="free")
print(res.annoyance, res.n5, res.sharpness, res.roughness, res.fluctuation_strength)

res.plot()   # the same PA / wS / wFR view, now from the four derived sensations
```

> **Model-mixing caveat.** This composite mixes model families: Zwicker-family
> $N_5$ and $S$, Sottek $R$ and Osses $F$. The $PA$ model was calibrated with
> Zwicker-family sensations, so treat the signal convenience as an *engineering
> estimate*. For exact, reproducible results, compute the four quantities with
> whichever models you trust and pass them to `psychoacoustic_annoyance`
> directly (the four-argument function is the exact model).

## 3. Fluctuation strength

**Fluctuation strength** $F$ (vacil) quantifies the perception of *slow*
loudness fluctuation. Like roughness it is a band-pass sensation of the
modulation frequency, but it peaks about an order of magnitude lower, at
$f_\mathrm{mod} \approx 4\ \text{Hz}$ rather than the ~70 Hz roughness peak. By
definition, a 1 kHz tone at 60 dB,
100 % amplitude-modulated at 4 Hz, produces 1 vacil. This page covers the
Fastl & Zwicker models; the normative Sottek-model fluctuation strength of
ECMA-418-2 Clause 9 lives in [Sound Quality Metrics](sound-quality.md).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fluctuation_strength_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fluctuation_strength.svg" alt="Fluctuation strength versus modulation frequency on a log axis: the closed-form AM broadband-noise curve and the AM-tone signal-model sweep both show a band-pass characteristic peaking at 4 Hz" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import psychoacoustics

# Exact closed form (Eq. 10.2), AM broadband noise at 60 dB, 100 % modulation:
fmod = np.logspace(np.log10(0.5), np.log10(32.0), 240)
f_bbn = [psychoacoustics.fluctuation_strength_am_noise(60.0, 1.0, fm) for fm in fmod]

# Osses 2016 signal model on a 1 kHz / 70 dB AM tone over the same sweep:
fs = 48000
t = np.arange(int(2.0 * fs)) / fs
carrier = np.sin(2 * np.pi * 1000 * t)
fm_tone = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
f_tone = []
for fm in fm_tone:
    am = (1.0 + np.sin(2 * np.pi * fm * t)) * carrier
    am = am / np.sqrt(np.mean(am ** 2)) * 2e-5 * 10 ** (70 / 20)
    f_tone.append(psychoacoustics.fluctuation_strength(am, float(fs)).fluctuation_strength)

fig, ax = plt.subplots()
ax.semilogx(fmod, f_bbn, label="AM broadband noise (closed form)")
ax2 = ax.twinx()
ax2.plot(fm_tone, f_tone, "s--", color="tab:green", label="AM tone (signal model)")
ax.axvline(4.0, ls="--", color="0.4")
ax.set_xlabel("Modulation frequency f_mod [Hz]")
ax.set_ylabel("Fluctuation strength F [vacil]")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right")
plt.show()
```

</details>

### 3.1 Closed form for AM broadband noise (Eq. 10.2)

For sinusoidally amplitude-modulated broadband noise, Fastl & Zwicker give a
closed form (Eq. 10.2) in modulation factor $m$, level $L$ and modulation
frequency $f_\mathrm{mod}$:

$$
F = \frac{5.8\,(1.25\,m - 0.25)\,[0.05\,(L/\mathrm{dB}) - 1]}
{(f_\mathrm{mod}/5\,\mathrm{Hz})^2 + (4\,\mathrm{Hz}/f_\mathrm{mod}) + 1.5}\ \ \mathrm{vacil}.
$$

The denominator is the 4 Hz band-pass: it bottoms out near
$f_\mathrm{mod} \approx 3.7\ \text{Hz}$ and
rises on either side. The result is clamped at 0 (the sensation vanishes below
~20 dB or $m < 0.2$). This exact form is the value to quote for AM broadband
noise.

```python
from phonometry import psychoacoustics

print(round(psychoacoustics.fluctuation_strength_am_noise(60.0, 1.0, 4.0), 4))   # 3.6943 vacil
```

### 3.2 The Osses 2016 signal model

`fluctuation_strength` implements the Osses et al. (2016) signal model: it builds
an excitation pattern over 47 auditory filters, extracts the ~4 Hz envelope
modulation per band, weights and combines it, and returns the overall $F$
(vacil)
plus the specific fluctuation strength over the Bark axis and the time-dependent
trace.

```python
from phonometry import psychoacoustics

res = psychoacoustics.fluctuation_strength(x, fs)
print(res.fluctuation_strength)   # vacil
res.plot()   # specific fluctuation strength F′(z) over the Bark axis (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fluctuation_strength_specific_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fluctuation_strength_specific.svg" alt="Specific fluctuation strength over the Bark axis for a 1 kHz tone at 70 dB SPL fully amplitude-modulated at 4 Hz: the sensation is concentrated in the critical bands around the carrier near 8 Bark and integrates to about 1.1 vacil" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import psychoacoustics

# The reference-like stimulus: a 1 kHz tone at 70 dB SPL, 100 % amplitude
# modulated at 4 Hz, where the sensation peaks.
fs = 48000
t = np.arange(int(2.0 * fs)) / fs
am = (1.0 + np.sin(2 * np.pi * 4.0 * t)) * np.sin(2 * np.pi * 1000 * t)
am = am / np.sqrt(np.mean(am ** 2)) * 2e-5 * 10 ** (70 / 20)
res = psychoacoustics.fluctuation_strength(am, float(fs))
print(round(res.fluctuation_strength, 2))    # 1.09 vacil

# One line: the specific fluctuation strength over the Bark axis.
res.plot()
plt.show()

# Or draw f'(z) by hand from the arrays the result carries:
fig, ax = plt.subplots()
ax.fill_between(res.bark_axis, res.specific, alpha=0.3)
ax.plot(res.bark_axis, res.specific)
ax.set_xlabel("Critical-band rate z [Bark]")
ax.set_ylabel("Specific fluctuation strength [vacil/Bark]")
plt.show()
```

</details>

The sensation stays where the modulated energy is, in the critical bands
around the carrier, which is exactly why the closed form of section 3.1 and
the signal model diverge for **broadband** modulated noise: there the
modulation is spread over the whole Bark axis and the band-by-band model
accumulates more of it than the single closed form allows.

> **No numeric standard.** Fluctuation strength has no ISO/IEC standard. This is
> a clean-room implementation from the Osses 2016 paper, calibrated so that the
> 1 kHz / 60 dB / $m = 1$ / 4 Hz AM tone reads 1.00 vacil by construction, and
> cross-checked against the Osses 2016 Table 1 literature values and the open
> SQAT reference (used only as a numeric oracle). Over the 70 dB AM-tone sweep
> $f_\mathrm{mod} \in \{1, 2, 4, 8, 16, 32\}\ \text{Hz}$ it gives
> `[0.40, 0.79, 1.09, 1.05, 0.17, 0.09]` vacil against the literature
> `[0.39, 0.84, 1.25, 1.30, 0.36, 0.06]`
> (Pearson $r = 0.98$, correct 4 Hz peak, within ~2.1×). FM-tone accuracy is
> explicitly not pursued; only AM stimuli are validated. The front-end also
> departs from the paper's exact framing (the paper's 2 s frames with 90 %
> overlap and absolute-threshold screening become 50 % overlap, a Hann window
> and relative floors here), so perfectly steady signals do not read exactly
> zero: a steady 1 kHz tone comes out near 0.09 vacil and steady broadband
> noise about 0.15-0.19 vacil, partly the physical level fluctuation of the
> noise itself. For **AM broadband noise** the signal model overshoots
> the absolute level (it spreads the modulated energy across bands); quote the
> closed form `fluctuation_strength_am_noise` (§3.1) for that stimulus.

## See also

- [Loudness](loudness.md): the ISO 532-1
  percentile loudness $N_5$ this model takes as its first argument, and the
  acquisition rules the recording has to satisfy before any of it applies.
- [Sound Quality Metrics](sound-quality.md):
  the DIN 45692 sharpness $S$, the ECMA-418-2 roughness $R$ and the normative
  vacil_HMS fluctuation strength of the same sensation this page models in
  vacil.
- [Advanced Loudness](advanced-loudness.md):
  the alternative loudness models, and why mixing families makes the composite
  an engineering estimate.
- [Environmental Levels](../../environment/assessment/environmental-levels.md):
  community annoyance, the other thing the word means, and the rating levels it
  is assessed with.
- Theory: [Advanced loudness models and sound quality](../../reference/theory/perception.md#advanced-loudness-models--sound-quality): the loudness and sound-quality metrics the annoyance model combines, derived one at a time.
- API reference: [`psychoacoustics.quality.annoyance`](https://jmrplens.github.io/phonometry/reference/api/psychoacoustics/annoyance/) and [`psychoacoustics.quality.fluctuation_strength`](https://jmrplens.github.io/phonometry/reference/api/psychoacoustics/fluctuation-strength/).

## References

- Fastl, H., & Zwicker, E. (2007). *Psychoacoustics: Facts and models*
  (3rd ed.). Springer.
  [doi:10.1007/978-3-540-68888-4](https://doi.org/10.1007/978-3-540-68888-4).
  The source of the psychoacoustic-annoyance model of section 2
  (Eqs 16.2–16.4, chapter 16) and of the closed-form fluctuation strength for
  AM broadband noise of §3.1 (Eq. 10.2, chapter 10).
- Osses Vecchi, A., García León, R., & Kohlrausch, A. (2016). Modelling the
  sensation of fluctuation strength. *Proceedings of Meetings on Acoustics*,
  28, 050005. [doi:10.1121/2.0000410](https://doi.org/10.1121/2.0000410).
  The fluctuation-strength signal model implemented in §3.2, including the
  Table 1 literature values used as its cross-check.
- Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B.
  (2025). *SQAT: a sound quality analysis toolbox for MATLAB* (open-source
  software). [github.com/ggrecow/SQAT](https://github.com/ggrecow/SQAT),
  [doi:10.5281/zenodo.7934709](https://doi.org/10.5281/zenodo.7934709).
  The open MATLAB reference used as the numeric oracle for the
  fluctuation-strength cross-checks on this page.

## Standards

Fastl & Zwicker (2007, 3rd ed.), *Psychoacoustics: Facts and Models*
(Springer): psychoacoustic annoyance
$PA = N_5(1 + \sqrt{w_S^2 + w_{FR}^2})$ with the
sharpness weighting $w_S$ and roughness/fluctuation weighting $w_{FR}$
(Eqs 16.2–16.4; origin Widmann 1992), and the closed form for the fluctuation
strength of amplitude-modulated broadband noise (Eq. 10.2). The fluctuation-
strength signal model follows Osses García & Kohlrausch (2016), *Modelling the
sensation of fluctuation strength* (ICA 2016), clean-room and with no numeric
standard: calibrated to 1 vacil at the reference AM tone and cross-checked
against the paper's Table 1 literature trends and the open SQAT reference oracle.
The four sensations reuse the library's ISO 532-1 loudness, DIN 45692 sharpness
and ECMA-418-2 roughness (see [Loudness](loudness.md) and
[Sound Quality Metrics](sound-quality.md)).
