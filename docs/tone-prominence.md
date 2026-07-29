← [Documentation index](README.md)

# Prominent Discrete Tones (ECMA-418-1)

Tonal components in machinery noise are far more annoying than their level
suggests. ECMA-418-1:2024 (referenced by ECMA-74 Annex D) gives two FFT-based
methods to decide whether a discrete tone is *prominent*:
`tone_to_noise_ratio()` compares the tone level with the masking noise in its
critical band (clause 11), and `prominence_ratio()` compares the critical band
centred on the tone with the two contiguous bands (clause 12). Both return a
structured verdict against the frequency-dependent prominence criteria.

## 1. Tone-to-noise ratio and prominence ratio

```python
import numpy as np
from phonometry import psychoacoustics

fs = 48000
rng = np.random.default_rng(0)
t = np.arange(fs) / fs
x = np.sin(2 * np.pi * 1000 * t) + 0.05 * rng.standard_normal(fs)  # 1 kHz tone in noise
tnr = psychoacoustics.tone_to_noise_ratio(x, fs)            # highest peak, or tone_freq=...
pr = psychoacoustics.prominence_ratio(x, fs, tone_freq=1000.0)
print(tnr.ratio_db, tnr.criterion_db, tnr.prominent)

tnr.plot()   # the tone against its prominence criterion (needs matplotlib)
```

The methods hinge on the **critical band**, the ear's analysis bandwidth,
$\Delta f_c = 25 + 75\ [1 + 1.4(f/1000)^2]^{0.69}$ Hz (162 Hz at 1 kHz): a
tone is masked only by the noise *inside* its critical band, so both methods
focus on that band rather than the whole spectrum, but they use it
differently. The tone-to-noise ratio works within the band, separating its
spectral lines into tone and noise and subtracting their levels (clause 11,
Formulae 9–11); the prominence ratio instead compares the *whole* band centred
on the tone with the mean of its two contiguous critical bands (clause 12,
Formula 23):

$$
\mathrm{TNR} = L_t - L_n, \qquad
\mathrm{PR} = 10\,\lg\!\frac{W_M}{\tfrac{1}{2}(W_L + W_U)}\ \text{dB},
$$

where $L_t$ is the tone level (the energy sum of the tonal lines above the
band-edge baseline, Formula 9), $L_n$ the level of the masking noise that
remains in the critical band, rescaled to the full critical bandwidth
(Formulae 10–11), and $W_M$, $W_L$, $W_U$ the powers in the middle, lower and
upper critical bands (below $f_t = 171.4$ Hz the truncated lower band is
rescaled to a 100 Hz bandwidth, Formula 24).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tonality_spectrum_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tonality_spectrum.svg" alt="Averaged spectrum of a tone in noise with the critical band shaded and the tone-to-noise ratio annotated against its prominence criterion" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from phonometry import psychoacoustics

fs = 48000
rng = np.random.default_rng(21)
t = np.arange(30 * fs) / fs
x = (np.sqrt(2) * 0.1 * np.sin(2 * np.pi * 1000 * t)
     + 0.05 * rng.standard_normal(t.size))
res = psychoacoustics.tone_to_noise_ratio(x, fs)

# Averaged 1 Hz Hann spectrum (the clause 11.1 front end) and the
# critical band about the detected tone (edges approximated as +/- dfc/2):
f, p = welch(x, fs, window="hann", nperseg=fs, scaling="spectrum")
dfc = 25 + 75 * (1 + 1.4 * (res.frequency / 1000) ** 2) ** 0.69
sel = (f > 700) & (f < 1400)
plt.plot(f[sel], 10 * np.log10(p[sel]))
plt.axvspan(res.frequency - dfc / 2, res.frequency + dfc / 2, alpha=0.15)
plt.title(f"TNR = {res.ratio_db:.1f} dB (criterion {res.criterion_db:.1f} dB)")
plt.xlabel("Frequency [Hz]"); plt.ylabel("Bin power [dB]")
plt.show()
```

</details>

A TNR at or above $8 + 8.33\log_{10}(1000/f_t)$ dB below 1 kHz (a flat 8 dB for
$f_t \ge 1$ kHz) classifies the tone as *prominent*; the PR criterion is
$9 + 10\log_{10}(1000/f_t)$ dB below 1 kHz and 9 dB from there up, likewise
applied with $\ge$. Low frequencies get higher thresholds because wider
relative bands mask more.

`ToneAssessment.plot()` puts the verdict in its context: it draws the
criterion of the producing method over the whole 89.1 Hz to 11.2 kHz range of
interest and marks the assessed tone at its own frequency, so the margin that
decides the verdict is visible rather than implied.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_prominence_assessment_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_prominence_assessment.svg" alt="Tone-to-noise ratio of a 250 Hz fan tone plotted against the ECMA-418-1 prominence criterion: the criterion falls from about 17 dB at 89 Hz to a flat 8 dB above 1 kHz, and the assessed tone sits at 15.1 dB, 2.1 dB above the 13.0 dB criterion at 250 Hz, so it is prominent" width="88%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import psychoacoustics

# A 250 Hz fan tone recorded in broadband machinery noise, 10 s at 48 kHz.
fs = 48000
rng = np.random.default_rng(4)
t = np.arange(10 * fs) / fs
x = (np.sqrt(2) * 0.011 * np.sin(2 * np.pi * 250.0 * t)
     + 0.03 * rng.standard_normal(t.size))
res = psychoacoustics.tone_to_noise_ratio(x, fs)
print(round(res.ratio_db, 1), round(res.criterion_db, 1), res.prominent)
# 15.1 13.0 True

# One line: the tone against the criterion curve of its own method.
res.plot()
plt.show()

# By hand, mirroring what ToneAssessment.plot() draws:
f = np.logspace(np.log10(89.1), np.log10(11200.0), 400)
criterion = np.where(f < 1000.0, 8.0 + 8.33 * np.log10(1000.0 / f), 8.0)
fig, ax = plt.subplots()
ax.semilogx(f, criterion, color="#d62728", label="prominence criterion")
ax.plot([res.frequency], [res.ratio_db], "o", label="assessed tone")
ax.plot([res.frequency] * 2, [res.criterion_db, res.ratio_db], ":", color="0.6")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Tone-to-noise ratio TNR [dB]")
ax.legend()
plt.show()
```

</details>

**When the two ratios disagree.** Near the criteria the verdicts can differ,
because each ratio is fragile in a different situation. TNR has to *split*
the critical band into tonal and noise lines first, so it degrades when that
separation is ambiguous: a tone riding a steep noise slope, or closely
spaced components whose skirts overlap. PR needs no separation, which makes
it the robust, automatable choice when **several tones share the critical
band** (they all land in $W_M$); in exchange it reads low when a
*neighbouring* band also carries a tone (the flanking bands are then not
noise) and is biased on sharply sloping spectra, where the two flanking
bands no longer estimate the masking at the tone. In practice: prefer TNR
for a clean, isolated tone, prefer PR for multi-tone complexes sharing a
band, and report both when they straddle their criteria.

## 2. Where to measure (ECMA-74) and practice

ECMA-74 (which delegates its tone assessments to ECMA-418-1) also fixes where to measure around a device (context only): phonometry implements the ECMA-418-1 assessments, not the ECMA-74 Annex D measurement procedure:

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_tonality_positions_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_tonality_positions.svg" alt="ECMA-74 emission measurement positions: seated operator microphone at 0.25 m and 1.20 m, and the four bystander positions at 1 m" width="92%"></picture>

Proximate secondary tones in the same critical band are combined per
clause 11.6; for harmonic complexes assess each component (`tone_freq=`).
Both methods work on Hann-windowed, RMS-averaged spectra and need no absolute
calibration (the ratios are level differences).

### `tone_to_noise_ratio()` / `prominence_ratio()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `x` | 1D array | any (uncalibrated OK) | ≥ `fs/resolution_hz` samples | Ratios are level differences: calibration cancels out |
| `fs` | int | Hz | > 0 | |
| `tone_freq` | float, optional | Hz | 89.1–11 200; default `None` | `None` assesses the highest peak in the range of interest |
| `resolution_hz` | float | Hz | > 0; default `1.0` | Tone band must stay within 15 % of the critical band (clause 11.2) |

Both return a `ToneAssessment(frequency, ratio_db, criterion_db, prominent)`.

## 3. Which tonality metric, and when

The library implements four tonality assessments, and they are not
interchangeable: each belongs to a different standard with its own purpose,
input and output.

| | TNR / PR | Tone audibility `ΔL` | Psychoacoustic tonality `T` | Wind-turbine tonal audibility |
| :--- | :--- | :--- | :--- | :--- |
| Standard | ECMA-418-1:2024, clauses 11 and 12 | ISO/PAS 20065:2016, adopted by ISO 1996-2:2017 Annex J | ECMA-418-2:2025, clause 6 | IEC 61400-11:2012, clause 9.5 |
| Question answered | Is this discrete tone prominent in the emission of a device? | Is the tone audible above the noise that masks it, and by how much? | How tonal does the sound feel? | Is a turbine tone audible at the reference position, wind bin by wind bin? |
| Input | One FFT spectrum, no calibration needed | Narrow-band spectra with the line spacing declared | The calibrated pressure signal, through the Sottek hearing model | Narrow-band spectra per wind-speed bin, with the reference distance |
| Output | A ratio in dB against a frequency-dependent criterion, plus a prominent / not prominent verdict | An audibility in dB, whose decisive and mean values feed the tonal adjustment `Kt` | A tonality in `tu_HMS`, a perceptual magnitude with no criterion | An audibility in dB per bin, reported with the turbine's sound power |
| Use it for | Declaring ITT equipment emission (ECMA-74 Annex D) | Environmental-noise rating: justifying, or refusing, a tonal penalty | Sound-quality work: comparing designs, not passing a limit | Type testing and acceptance of wind turbines |

Read the table as a decision: the *purpose of the report* picks the metric,
not the convenience of the input. A device emission declaration is an
ECMA-418-1 prominence verdict; a complaint about a tone from an installation
is an ISO 1996-2 rating, so it needs the audibility `ΔL` that maps to `Kt`
([tone audibility](tone-audibility.md)); a product comparison where nobody is
being fined is where the ECMA-418-2 tonality earns its place, because it is a
magnitude rather than a threshold test ([sound quality](sound-quality.md));
and a turbine is its own regime, with a measurement procedure in
IEC 61400-11 that fixes the geometry and the wind bins
([wind-turbine noise](wind-turbine-noise.md)).

Two consequences worth keeping in mind. First, the numbers do not convert:
a TNR of 12 dB is not an audibility of 12 dB, because the masking model, the
band definition and the reference differ. Second, the metrics can disagree
about the same sound, and legitimately so. A tone can be prominent by
ECMA-418-1 in the near field of the machine and inaudible at the dwelling
where the ISO 1996-2 assessment is made, while a hearing-model tonality
stays moderate throughout because it rates the whole sound and not one
spectral line. Report the metric the assessment calls for, and quote any
other only as supporting evidence.

## See also

- [Environmental levels](environmental-levels.md): the ISO 1996-1 rating levels whose tonal adjustments
  (Table A.1) these prominence verdicts justify objectively.
- [Sound Quality Metrics](sound-quality.md): the ECMA-418-2 psychoacoustic
  tonality T in tu_HMS, the hearing-model counterpart of these FFT ratios.
- [Objective audibility of tones in noise](tone-audibility.md): the ISO/PAS 20065
  audibility ΔL that the ISO 1996-2 tonal adjustment Kt is read from.
- [Wind-turbine noise](wind-turbine-noise.md): the IEC 61400-11 tonal audibility,
  the same question asked per wind-speed bin.
- [Impulsive-sound prominence](impulse-prominence.md): the NT ACOU 112
  counterpart for impulsive (rather than tonal) character.
- [Theory](theory-perception.md): the critical-band model and criteria derivation.
- API reference: [`psychoacoustics.tonality`](https://jmrplens.github.io/phonometry/reference/api/psychoacoustics/tonality/).

## References

- Ecma International. (2024). *ECMA-418-1: Psychoacoustic metrics for ITT
  equipment — Part 1: Prominent discrete tones* (3rd ed.).
  [Free PDF](https://ecma-international.org/wp-content/uploads/ECMA-418-1_3rd_edition_december_2024.pdf).
  The implemented standard, freely downloadable: the TNR (clause 11) and PR
  (clause 12) methods, the critical-band model and the prominence criteria of
  section 1.
- Ecma International. (2025). *ECMA-74: Measurement of airborne noise emitted
  by information technology and telecommunications equipment* (22nd ed.).
  [Free PDF](https://ecma-international.org/wp-content/uploads/ECMA-74_22nd_edition_december_2025.pdf).
  The parent emission standard, freely downloadable: the operator/bystander
  measurement positions of section 2, with Annex D delegating the tone
  assessments to ECMA-418-1.

## Standards

ECMA-418-1:2024 (3rd edition), *Psychoacoustic metrics for ITT
equipment — Part 1: Prominent discrete tones* — the tone-to-noise ratio
(clause 11), the prominence ratio (clause 12), the critical-band model and the
frequency-dependent prominence criteria.
