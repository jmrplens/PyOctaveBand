← [Documentation index](README.md)

# Objective audibility of tones in noise (ISO/PAS 20065)

A steady tone embedded in broadband noise stands out when it rises audibly above
the noise that would otherwise mask it, the objective precondition for the tonal
penalties applied in noise assessment. **ISO/PAS 20065:2016** is the *engineering
method* that quantifies this audibility: from a narrow-band FFT spectrum it
derives, for every prominent tone, the **audibility** $\Delta L$: how many
decibels the tone level exceeds the masking threshold of the surrounding noise.
(Whether a tone is *annoying* is a separate, downstream rating judgement.) It
is the detailed method that **ISO 1996-2:2017** defers to (the simpler Annex C
route lives in [environmental measurement](environmental-levels.md)); the mean
audibility $\Delta L$ it produces feeds the ISO 1996-2 tonal adjustment $K_t$.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_audibility_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_audibility.svg" alt="Per-tone audibility ΔL of the nine tones of the ISO/PAS 20065 Annex E combustion-engine spectrum, with the decisive tone at 137.3 Hz highlighted and the ΔL = 0 dB audibility threshold marked" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import psychoacoustics

# ISO/PAS 20065 Annex E combustion-engine spectrum 1: nine tones (fT, LT, LS)
# from a narrow-band spectrum with line spacing 2.7 Hz
fT = [118.4, 137.3, 158.8, 314.9, 433.4, 592.2, 629.8, 643.3, 1582.7]
LT = [64.56, 67.96, 68.63, 68.50, 73.17, 78.31, 75.00, 79.75, 71.07]
LS = [48.91, 49.22, 50.50, 52.85, 58.29, 59.53, 59.71, 61.98, 54.16]

res = psychoacoustics.assess_tones(fT, LT, LS, 2.7)
print(round(res.decisive_audibility, 2), res.decisive_frequency)  # 5.01 137.3
res.plot()   # per-tone audibility bars, decisive tone highlighted
plt.show()
```

</details>

## 1. The critical band about the tone

Each tone of frequency $f_T$ is evaluated inside a critical band whose width
is (Formula 2)

$$
\Delta f_c = 25.0 + 75.0\left(1.0 + 1.4\left(\tfrac{f_T}{1000}\right)^{2}\right)^{0.69}\ \mathrm{Hz}.
$$

With a geometric placement of the corner frequencies about the tone
(Formulae 3–5), $\sqrt{f_1 f_2} = f_T$ and $f_2 - f_1 = \Delta f_c$, so
$f_1 = -\Delta f_c/2 + \sqrt{\Delta f_c^2 + 4 f_T^2}/2$ and
$f_2 = f_1 + \Delta f_c$.

```python
from phonometry import psychoacoustics

print(round(psychoacoustics.critical_bandwidth_engineering(137.3), 2))   # 101.36 Hz
f1, f2 = psychoacoustics.critical_band_corners(137.3)
print(round(f1, 2), round(f2, 2))                            # 95.67 197.04
```

## 2. Audibility of a tone

The mean narrow-band level $L_S$ of the masking noise (Formula 6, an iterative
energy average of the lines in the critical band) and the tone level $L_T$
(Formula 8, the energy sum of the tonal lines) are derived from the narrow-band
spectrum; `mean_narrowband_level` and `tone_level` do this directly (see §4).
The critical-band level of the masking noise spreads $L_S$ over the critical
bandwidth (Formula 12), the masking index accounts for the ear (Formula 13) and
the audibility is their difference (Formula 14):

$$
L_G = L_S + 10\log_{10}\!\frac{\Delta f_c}{\Delta f}, \qquad
a_v = -2 - \log_{10}\!\Big[1 + \big(\tfrac{f}{502}\big)^{2.5}\Big], \qquad
\Delta L = L_T - L_G - a_v .
$$

A supplied tone is *audible* when $\Delta L > 0$. $\Delta f$ is the line
spacing (frequency resolution); the energy sums over $K > 1$ lines carry a
window correction of $10\log_{10}(\Delta f/\Delta f_e)$ (−1.76 dB for the
recommended Hanning window, $\Delta f_e = 1.5\,\Delta f$, Formula (8)), while
a single-line tone ($K = 1$) takes its level unchanged (Formula (7), no
bandwidth correction).

```python
from phonometry import psychoacoustics

# ISO/PAS 20065 Annex E, tone at 137.3 Hz (Δf = 2.7 Hz):
#   LS = 49.22 dB (Formula 6), LT = 67.96 dB (Formula 8).
print(round(psychoacoustics.tone_audibility(67.96, 49.22, 137.3, 2.7), 2))   # 5.01 dB
print(round(psychoacoustics.masking_index(137.3), 2))                        # -2.02 dB
```

The whole method is one chain from the spectrum to the penalty, and every
intermediate quantity above is a stop on it. The diagram walks the Annex E
tone through that chain, down to the $K_t$ the mean audibility earns.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_tone_audibility_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_tone_audibility.svg" alt="Flow diagram of the ISO/PAS 20065 engineering method on the Annex E combustion-engine spectrum: a narrow-band FFT spectrum with 2.7 hertz line spacing yields a tone at 137.3 hertz, its critical band of 101.36 hertz spans 95.67 to 197.04 hertz, the masking noise level of 49.22 decibels and tone level of 67.96 decibels give a masking threshold of 64.97 decibels with a masking index of minus 2.02 decibels, and the audibility is 5.01 decibels, the decisive value of the spectrum; a closing note maps the 6.98 decibel energy-mean audibility of the five spectra to a tonal adjustment Kt of 4 decibels through ISO 1996-2 Table J.1" width="92%"></picture>

## 3. Decisive and mean audibility

The **decisive** audibility of one narrow-band spectrum is the largest tone
audibility in it (clause 5.3.8). Over $J$ staggered spectra the **mean
audibility** is their energy mean (Formula 20); a spectrum in which no tone is
found contributes $\Delta L_j = -10\ \text{dB}$ (Formula 21). `assess_tones` applies the whole
chain to a spectrum's tones and reports the decisive tone.

**How the decisive band is selected.** The method does not scan a fixed set of
bands: each detected *tone* defines its own critical band (§1), the audibility
is evaluated tone by tone, and, after Step 3 has merged audible same-band
tones into `FG` groups rated at their most audible member (§5), the decisive
audibility is simply the largest $\Delta L$ left standing (Step 4). The "decisive
band" is therefore the critical band centred on whichever tone or group wins,
and it is free to move from spectrum to spectrum as the source runs through
its operating states; the energy mean of Formula 20 then lets the loudest
(most audible) spectra dominate the reported value, which is deliberate: a
tone that is clearly audible part of the time is not excused by intervals in
which it disappears.

```python
from phonometry import psychoacoustics

# Annex E combustion-engine spectrum 1: nine tones (fT, LT, LS), Δf = 2.7 Hz.
fT = [118.4, 137.3, 158.8, 314.9, 433.4, 592.2, 629.8, 643.3, 1582.7]
LT = [64.56, 67.96, 68.63, 68.50, 73.17, 78.31, 75.00, 79.75, 71.07]
LS = [48.91, 49.22, 50.50, 52.85, 58.29, 59.53, 59.71, 61.98, 54.16]
res = psychoacoustics.assess_tones(fT, LT, LS, 2.7)
print(round(res.decisive_audibility, 2), res.decisive_frequency)  # 5.01 137.3

# Mean audibility of the five measured spectra (Table E.3 decisive values):
print(round(psychoacoustics.mean_audibility([9.18, 6.04, 7.46, 2.67, 7.17]), 2))  # 6.98 dB

res.plot(view="levels")   # tone levels above their critical-band masking noise
```

The same assessment reads two ways. The audibility bars at the top of this
page answer "how far above the masking threshold is each tone"; the levels
view answers "what did the analyser see", which is the view an assessment
report has to defend:

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_audibility_levels_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_audibility_levels.svg" alt="Tone levels and critical-band masking noise of the ISO/PAS 20065 Annex E combustion-engine spectrum on a logarithmic frequency axis from about 96 Hz to 1.8 kHz: each tone is a stem from its critical-band masking level up to its tone level, and the decisive 137.3 Hz tone is highlighted with its critical band shaded" width="88%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import psychoacoustics

# The Annex E combustion-engine spectrum of the snippet above.
fT = [118.4, 137.3, 158.8, 314.9, 433.4, 592.2, 629.8, 643.3, 1582.7]
LT = [64.56, 67.96, 68.63, 68.50, 73.17, 78.31, 75.00, 79.75, 71.07]
LS = [48.91, 49.22, 50.50, 52.85, 58.29, 59.53, 59.71, 61.98, 54.16]
res = psychoacoustics.assess_tones(fT, LT, LS, 2.7)

# One line: the levels view, the same one the .report() fiche embeds.
res.plot(view="levels")
plt.show()

# The default view is the per-tone audibility instead:
res.plot()      # or res.plot(view="audibility")
plt.show()
```

</details>

Reading it left to right: each horizontal segment is the critical-band
masking-noise level $L_G$ drawn across the band it applies to, each marker is
the tone level $L_T$, and the gap between them, less the masking index, is the
audibility. A tone whose marker sits *below* its segment is masked, and no
amount of level justifies a penalty for it.

### 3.1 Extended uncertainty of the audibility

Clause 5.4 attaches a 90 % bilateral extended uncertainty $U$ to every
audibility, and clause 6 makes it mandatory whenever fewer than 12 spectra
have been averaged. `assess_tones` computes it per tone
(`res.extended_uncertainties`), `audibility_uncertainty` evaluates it straight
from the spectrum lines, and `mean_audibility_uncertainty` propagates it to
the energy-averaged audibility of a spectrum set. The Annex E example
reproduces the printed Table E.2 column (for the 137.3 Hz tone,
$U = 2.80\ \text{dB}$ against the printed 2,79).

**How to read $U$.** A 90 % *bilateral* interval leaves 5 % in each tail, so
$\Delta L - U$ is a one-sided 95 % statement: when the whole interval
$\Delta L \pm U$ sits above 0 dB the tone is audible with at least 95 %
confidence, and when the interval straddles zero the verdict is not
statistically secured; the remedy is more spectra, since $U$ shrinks with the
number averaged (which is exactly why clause 6 makes reporting it mandatory
below 12 spectra). The same logic guards the downstream penalty:
**ISO 1996-2:2017 Annex J** converts the mean audibility into the tonal
adjustment $K_t$ in 1 dB steps (Table J.1: $K_t = 0$ for $\Delta L \le 0$, up
to $K_t = 6\ \text{dB}$ for $\Delta L > 12\ \text{dB}$, or the coarser
0/3/6 dB ladder of its note), so an uncertainty that spans a table boundary
propagates straight into a 1–3 dB question mark on the rating level. Quoting
$\Delta L \pm U$ alongside $K_t$ shows whether the adjustment is robust or
hinges on one borderline spectrum.

## 4. From the narrow-band spectrum

Given the FFT lines of the critical band about a tone, `mean_narrowband_level`
runs the iterative Formula 6 procedure (energy average, dropping any line more
than 6 dB above the running $L_S$, until stable within ±0.005 dB or fewer than
five lines remain each side, Annex D) and `tone_level` sums the tonal lines
contiguous with the peak (above both $L_S + 6\ \text{dB}$ and
$L_\text{peak} - 10\ \text{dB}$). The
mean always carries the −1.76 dB Hanning bandwidth correction; the tone level
carries it only when the run spans more than one line (Formulae (7)/(8)).

```python
from phonometry import psychoacoustics

# Annex E Table E.1: the 38 lines of the 137.3 Hz critical band (Δf = 2.7 Hz).
freqs = [96.9, 99.6, 102.3, 105.0, 107.7, 110.4, 113.0, 115.7, 118.4, 121.1,
         123.8, 126.5, 129.2, 131.9, 134.6, 137.3, 140.0, 142.7, 145.3, 148.0,
         150.7, 153.4, 156.1, 158.8, 161.5, 164.2, 166.9, 169.6, 172.3, 175.0,
         177.6, 180.3, 183.0, 185.7, 188.4, 191.1, 193.8, 196.5]
levels = [49.40, 50.68, 50.09, 53.37, 44.47, 50.91, 51.41, 59.40, 64.54, 57.57,
          51.02, 50.76, 59.93, 62.94, 58.49, 65.87, 62.66, 50.25, 51.32, 52.30,
          52.58, 53.15, 67.04, 67.27, 57.40, 57.17, 52.56, 51.39, 52.49, 47.68,
          51.26, 49.03, 61.42, 59.52, 48.43, 50.84, 48.20, 55.95]

ls = psychoacoustics.mean_narrowband_level(levels, freqs, 137.3)
lt = psychoacoustics.tone_level(levels, freqs, 137.3, ls)
print(round(ls, 2), round(lt, 2))                  # 49.22 67.96
print(round(psychoacoustics.tone_audibility(lt, ls, 137.3, 2.7), 2))   # 5.01 dB
```

## 5. Whole-spectrum detection

`analyze_spectrum` runs the full front-end over a spectrum (mean narrow-band
level per line, peak detection (Clause 5.3.8 Step 1, a tone cannot sit on a
slope), tone level, the distinctness test (Clause 5.3.4: bandwidth
$\le 26\,(1 + 0.001 f_T)$ Hz and edge steepness $\ge 24\ \text{dB}$), and
audibility) and
returns the distinct, audible tones. It then applies **Step 3**: audible
tones sharing a critical band have their tone levels energy-summed
(Formula 17, shared lines counted once, via `combined_tone_level`) into a
combined "FG" entry rated at the most audible member, unless the
exactly-two-tones-below-1000-Hz exception of §5.1 keeps them separate. The
result's `group_sizes` tells individual tones (`1`) from FG entries
($N \ge 2$), and the decisive audibility (Step 4) is the maximum over all
entries.

```python
from phonometry import psychoacoustics

# Annex E Table E.1: the 38 lines of the 137.3 Hz critical band (Δf = 2.7 Hz).
freqs = [96.9, 99.6, 102.3, 105.0, 107.7, 110.4, 113.0, 115.7, 118.4, 121.1,
         123.8, 126.5, 129.2, 131.9, 134.6, 137.3, 140.0, 142.7, 145.3, 148.0,
         150.7, 153.4, 156.1, 158.8, 161.5, 164.2, 166.9, 169.6, 172.3, 175.0,
         177.6, 180.3, 183.0, 185.7, 188.4, 191.1, 193.8, 196.5]
levels = [49.40, 50.68, 50.09, 53.37, 44.47, 50.91, 51.41, 59.40, 64.54, 57.57,
          51.02, 50.76, 59.93, 62.94, 58.49, 65.87, 62.66, 50.25, 51.32, 52.30,
          52.58, 53.15, 67.04, 67.27, 57.40, 57.17, 52.56, 51.39, 52.49, 47.68,
          51.26, 49.03, 61.42, 59.52, 48.43, 50.84, 48.20, 55.95]

# Same Table E.1 spectrum as above.
res = psychoacoustics.analyze_spectrum(levels, freqs, 2.7)
singles = res.group_sizes == 1
print([round(f, 1) for f in res.tone_frequencies[singles]])  # [118.4, 137.3, 158.8]

# Step 3 already combined the three same-band tones into an FG entry:
fg = res.group_sizes > 1
print(int(res.group_sizes[fg][0]), round(float(res.tone_levels[fg][0]), 2))  # 3 72.15

# The same Formula 17 combination, called directly (LS from Table E.2):
lt_fg = psychoacoustics.combined_tone_level(levels, freqs, [118.4, 137.3, 158.8],
                               [48.91, 49.22, 50.50])
print(round(lt_fg, 2))                                # 72.15

res.plot()   # the detected entries, FG groups included, as audibility bars
```

Reproducing a *decisive* audibility exactly needs the **complete** narrow-band
spectrum: Table E.1 is truncated to the 137.3 Hz critical band, so the 158.8 Hz
tone's mean narrow-band level is under-estimated from it (the algorithm itself
matches the parent standard DIN 45681:2005-03 reference program). The peak
detection and FG combination above are verified against the Annex E worked
example (the three tone frequencies and $L_T = 72.15\ \text{dB}$).

### 5.1 Two tones below 1000 Hz

When **exactly two** tones share a critical band and both lie below 1000 Hz, the
ear can still tell them apart (so they are rated *separately* rather than
FG-combined) if their frequency difference $|f_{T1} - f_{T2}|$ (Formula 18)
exceeds

$$
f_D = 21 \cdot 10^{\,1.2\,\left|\log_{10}(f_T/212)\right|^{1.8}}\ \text{Hz}
\qquad (\text{Formula 19},\ 88\ \text{Hz} < f_T < 1000\ \text{Hz})
$$

evaluated at the more prominent tone $f_T$ (the larger audibility $\Delta L$).
The threshold bottoms out at 21 Hz at $f_T = 212\ \text{Hz}$ and grows on
either side. `two_tone_separation_frequency` gives $f_D$;
`resolve_tones_separately` applies the decision.

```python
from phonometry import psychoacoustics

psychoacoustics.two_tone_separation_frequency(212.0)             # 21.0 Hz (minimum)
psychoacoustics.resolve_tones_separately(200.0, 260.0, 3.0, 2.0) # True  → rate separately
psychoacoustics.resolve_tones_separately(118.4, 137.3, 4.0, 5.0) # False → combine (Δf < fD)
```

> **No numeric oracle.** No ISO/PAS 20065 worked example exercises this branch;
> the Annex E band groups *three* tones, so the "exactly two tones" rule never
> fires there. The formula and decision are implemented clean-room from the text
> and verified against the **DIN 45681:2005-03** Annex J reference program
> (`fD = 21 * 10 ^ (1.2 * Abs(Log(fT / 212) / Log(10)) ^ 1.8)`). Reassuringly,
> evaluated at the Annex E tones the threshold ($\approx 24\ \text{Hz}$ at
> 137.3 Hz) keeps them combined, consistent with that example's FG grouping.

## Tonal assessment report (`.report()`)

`ToneAudibilityResult.report(path)` renders a one-page PDF fiche laid out like a
tonal-assessment report of an environmental-noise laboratory, following the
ISO 1996-2:2017 Annex J engineering method: a standard-basis line, an optional
metadata header block (source/situation, client, measurement position,
instrumentation and date, with the analysis line spacing $\Delta f$ read from
the result), a full-width table of the key quantities for every detected tone
(tone frequency $f_T$, entry type, tone level $L_{pt}$, critical-band
masking-noise level $L_{pn}$, critical bandwidth $\Delta f_c$ and the
audibility $\Delta L_{ta}$) above the level-versus-frequency analysis plot
with the tones and their critical-band masking noise marked, the boxed
decisive audibility $\Delta L_{ta}$ together with the derived tonal
adjustment $K$ (Table J.1), an optional PASS/FAIL verdict row and a
prominence note, and a footer with the fixed disclaimer.

It uses the same `ReportMetadata` container
(documented under [Insulation ratings](insulation-ratings.md#report-metadata-reportmetadata))
and rendering engine as the [ISO 532-1 loudness fiche](loudness.md#iso-532-1-report-report);
a supplied `requirement` is read as the maximum acceptable decisive audibility
$\Delta L_{ta}$ in dB (a quieter tone passes). Rendering needs reportlab and, for the
figure the fiche embeds, matplotlib (`pip install "phonometry[report,plot]"`);
only `engine="reportlab"` is supported. The fiche renders in English by default;
pass `language="es"` for a Spanish fiche (translated fixed strings and a comma
decimal separator), e.g. `res.report("tone_fiche_es.pdf", language="es")`.

```python
from phonometry import psychoacoustics, ReportMetadata

# The Annex E combustion-engine spectrum (analyze_spectrum).
res = psychoacoustics.analyze_spectrum(levels, freqs, 2.7)
res.report(
    "tone_fiche.pdf",
    metadata=ReportMetadata(
        specimen="Combustion engine, steady operation",
        measurement_standard="ISO 1996-2",
        laboratory="Phonometry Reference Laboratory",
        requirement=6.0,             # maximum acceptable ΔL_ta (dB)
    ),
)                                    # decisive ΔL_ta (dB) and K (dB, Table J.1)
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository; click the preview to open the PDF.

[![ISO 1996-2 tonal audibility example report: metadata header, a per-tone table of the tone level Lpt, the critical-band masking-noise level Lpn, the critical bandwidth and the audibility, the level-versus-frequency analysis plot with the tones and their masking noise marked, the boxed decisive ΔL_ta = 9.1 dB with the tonal adjustment K = 5 dB (ISO 1996-2:2017 Table J.1) and a FAIL verdict against a 6 dB audibility limit](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso1996_tone_audibility_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso1996_tone_audibility_example.pdf)

*Tonal audibility fiche (`ToneAudibilityResult.report`), decisive $\Delta L_{ta}$ in dB with the tonal adjustment $K$.*

## References

- International Organization for Standardization. (2016). *Acoustics —
  Objective method for assessing the audibility of tones in noise —
  Engineering method* (ISO/PAS 20065:2016; withdrawn, superseded by
  [ISO/TS 20065:2022](https://www.iso.org/standard/81518.html)).
  [iso.org catalogue](https://www.iso.org/standard/66941.html).
  The implemented engineering method: every formula on this page, from the
  critical band to the mean audibility and its uncertainty, follows the 2016
  PAS edition.
- International Organization for Standardization. (2017). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 2:
  Determination of sound pressure levels* (ISO 1996-2:2017).
  [iso.org catalogue](https://www.iso.org/standard/59766.html).
  The environmental-noise standard this method serves: its Annex J adopts the
  engineering method and maps the mean audibility to the tonal adjustment
  `Kt` (Table J.1) discussed in §3.1.

## Standards

ISO/PAS 20065:2016, *Acoustics — Objective method for assessing
the audibility of tones in noise — Engineering method*: the critical bandwidth
$\Delta f_c$ (Formula 2) and its corner frequencies (Formulae 3–5), the
critical-band level $L_G$ (Formula 12), the masking index $a_v$ (Formula 13),
the audibility $\Delta L = L_T - L_G - a_v$ (Formula 14) and the energy-mean
mean audibility (Formula 20). The mean narrow-band level $L_S$ (Formula 6,
iterative Annex D) and
tone level $L_T$ (Formula 8) are computed from the critical-band spectrum, and
`analyze_spectrum` adds peak detection (Clause 5.3.8) with the distinctness
criteria (Clause 5.3.4) and the multi-tone `FG` combination (Formula 17), plus
the separate evaluation of two tones below 1000 Hz (Formulae 18/19). The
−1.76 dB Hanning bandwidth correction, the iterative masking-level procedure and
the detection/combination logic are confirmed against the parent standard
**DIN 45681:2005-03** (its Annex J reference program). Conformance is anchored on
the Annex E combustion-engine worked example (Tables E.1/E.2/E.3): $L_S$ and
$L_T$ from the spectrum, tone detection and the `FG` combined level, the
per-tone audibility, the masking index and the mean audibility of the five
spectra.
