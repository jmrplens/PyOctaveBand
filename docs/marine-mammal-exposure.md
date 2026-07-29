← [Documentation index](README.md)

# Marine-mammal noise exposure: audiograms, auditory weighting and injury criteria

A marine mammal does not hear every frequency equally, so an underwater noise
assessment cannot compare a broadband level against a single number. Regulatory
practice instead **weights** the spectrum with a filter shaped like the hearing
group's sensitivity, sums the weighted energy, accumulates it over the whole
activity and compares the result against a published onset criterion. This page
covers the three pieces the library provides for that chain: the hearing curves
themselves, the weighting functions and criteria of the current guidance, and
the end-to-end assessment of a piling campaign.

Levels are in dB re 1 µPa (sound pressure), dB re 1 µPa²·s (sound exposure) or,
for the two in-air carnivore groups, dB re 20 µPa and dB re (20 µPa)²·s. The
underwater reference conventions themselves are in
[Underwater acoustics](underwater-acoustics.md).

## 1. Hearing groups (and why the names collide)

Every current criteria set sorts marine mammals into hearing groups and gives
each one a filter. The **group codes are not portable between guidance
versions**, and the collision is a real trap:

| Southall et al. (2019) / NMFS 2024 | NMFS 2018 | Typical genera |
|---|---|---|
| LF cetaceans | LF | baleen whales |
| HF cetaceans | **MF** | sperm, beaked and most delphinid whales |
| VHF cetaceans | **HF** | porpoises, *Cephalorhynchus*, *Kogia* |
| PCW / PW | PW | phocid seals in water |
| OCW / OW | OW | otariids, in water |
| SI, PCA, OCA | — | sirenians and carnivores in air |

`hearing_groups(guidance)` lists the codes a version defines, and passing a
code from the wrong version raises rather than silently returning the wrong
filter.

```python
from phonometry import underwater

print(underwater.hearing_groups("nmfs-2024"))
# ('LF', 'HF', 'VHF', 'PW', 'OW', 'PA', 'OA')
print(underwater.hearing_groups("nmfs-2018"))
# ('LF', 'MF', 'HF', 'PW', 'OW')
```

## 2. Group audiograms

`group_audiogram` evaluates the band-pass fit of Southall et al. (2019),
Equation (1), after Finneran (2016):

$$
T(f) = T_0 + A \lg\!\left(1 + \frac{F_1}{f}\right) + (f/F_2)^{B},
$$

with $f$ in kilohertz. `normalized=False` (the default) uses the Table 2
parameters, fitted to the absolute median behavioural thresholds;
`normalized=True` uses the Table 3 refit on thresholds normalised to each
individual's best value. The article publishes **no fitted audiogram for LF
cetaceans** (no audiometric data exist and $F_1$ is never printed), so that
group is deliberately absent rather than reconstructed by guesswork.

```python
import numpy as np
from phonometry import underwater

freqs = np.logspace(2, 5.3, 400)
audiogram = underwater.group_audiogram(freqs, "VHF")
print(audiogram.best_frequency, audiogram.best_threshold)
audiogram.plot()   # threshold vs frequency (needs matplotlib)

print(underwater.AUDIOGRAM_GROUPS)
# ('HF', 'VHF', 'SI', 'PCW', 'OCW', 'PCA', 'OCA')
```

For a species rather than a group, `orca_audiogram` implements the killer-whale
curve of Wensveen & Van Roij (2007) as printed in Ainslie (2010),
Equation (11.159), a three-branch power law over 0.5 to 80 kHz. Its minimum is
39.0 dB re 1 µPa at 22.6 kHz, and 51.2 dB re 1 µPa at 50 kHz. That second
value needs the **third** branch: evaluating the second one there returns
50.5 dB instead, which is why both published points are pinned by the tests.

```python
from phonometry import underwater

print(underwater.orca_audiogram(50e3).threshold[0])   # 51.20 dB re 1 uPa
```

That value is the hearing threshold in Ainslie's orca-versus-salmon example:
with a source level of 198.2 dB re 1 µPa²·m² and a salmon target strength of
−29.0 dB re m², the hearing-limited figure of merit is
$(SL + TS - HT)/2 = 59.0$ dB re m².

## 3. Auditory weighting and exposure functions

All three current criteria sets use the same generic band-pass filter (NMFS
2018 Equation 1, Southall et al. Equation 2):

$$
W(f) = C + 10 \lg \frac{(f/f_1)^{2a}}
                       {\left[1+(f/f_1)^2\right]^{a}\left[1+(f/f_2)^2\right]^{b}} ,
$$

with $f$ in kilohertz. $C$ is fixed by putting the peak of $W$ at 0 dB, so the
companion **exposure function** $E(f) = K + C - W(f)$ has its minimum at the
weighted TTS-onset threshold $T_w = K + C$. Below $f_1$ the filter falls at
$20a$ dB/decade and above $f_2$ at $20b$ dB/decade.

Only the parameter table changes between versions, so the version is an
explicit argument and is carried on the result:

- **`"nmfs-2024"`** (the default): NOAA Fisheries *Updated Technical
  Guidance* v3.0, October 2024. It supersedes the 2018 revision, sets $b = 5$
  for every group, adopts the Southall group names and replaces "PTS onset"
  with "auditory injury (AUD INJ) onset".
- **`"nmfs-2018"`**: the 2018 revision v2.0, still cited by assessments
  already in flight.
- **`"southall-2019"`**: the peer-reviewed criteria, numerically identical to
  NMFS 2018 on the five shared groups and adds sirenians and both in-air
  carnivore groups.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/marine_mammal_weighting_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/marine_mammal_weighting.svg" alt="Auditory weighting functions of the five NMFS 2024 in-water hearing groups: low-frequency cetaceans peak near 1.4 kHz, high-frequency cetaceans near 11 kHz, very high-frequency cetaceans near 27 kHz, phocid pinnipeds near 5.6 kHz and otariid pinnipeds near 7.8 kHz, each falling steeply outside its passband" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import underwater

freqs = np.logspace(1, 5.4, 700)
fig, ax = plt.subplots()
for group in ("LF", "HF", "VHF", "PW", "OW"):
    res = underwater.auditory_weighting(freqs, group, guidance="nmfs-2024")
    crit = underwater.exposure_criteria(group, guidance="nmfs-2024", impulsive=True)
    ax.semilogx(res.frequencies, res.weighting,
                label=f"{group} (AUD INJ {crit.injury_sel:.0f} dB)")
ax.set(xlabel="Frequency [Hz]", ylabel="Weighting amplitude W(f) [dB]",
       ylim=(-75, 5))
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()
```

</details>

```python
from phonometry import underwater

res = underwater.auditory_weighting(1000.0, "LF", guidance="nmfs-2018")
print(res.weighting[0])           # -0.06 dB, the published Appendix D value
print(res.weighted_tts_onset)     # Tw = K + C

params = underwater.weighting_parameters("OW", guidance="nmfs-2024")
print(params.a, params.b, params.f1_khz, params.f2_khz, params.c_db)
```

> NMFS 2024 prints $C = 1.37$ dB for the otariid in-water group and states in
> the same table's footnote that the value should be 1.36 dB. The library
> implements 1.36 (recomputing $C$ from the peak of $W$ gives 1.3643) and keeps
> the printed digit as `c_db_as_printed`. Southall's Table 7 carries four
> peak-SPL misprints corrected by the journal's own errata, and the corrected
> values are the ones implemented. Both are recorded in
> [Errata found in published sources](ERRATA.md).

## 4. Onset criteria

`exposure_criteria` returns the published TTS and injury onset criteria of a
group. Impulsive noise carries a **dual metric**: a weighted sound exposure
level *and* an unweighted ("flat") peak sound pressure level, and the criterion
that produces the larger isopleth governs.

```python
from phonometry import underwater

crit = underwater.exposure_criteria("VHF", guidance="nmfs-2024", impulsive=True)
print(crit.injury_label)        # 'AUD INJ'
print(crit.tts_sel, crit.injury_sel)          # 144, 159 dB re 1 uPa2 s (weighted)
print(crit.tts_peak_spl, crit.injury_peak_spl)  # 196, 202 dB re 1 uPa (flat)
print(crit.source)
```

The tables are internally consistent in ways the library pins as tests: the
non-impulsive injury level is always TTS + 20 dB, the published weighted TTS
onset is always the rounded $K + C$, and in Southall's impulsive table the SEL
criteria run TTS + 15 dB and the peak criteria TTS + 6 dB.

## 5. A worked pile-driving assessment

Percussive pile driving is the canonical impulsive case. The chain runs from
the recorded strike to the verdict:

1. `strike_sel_spectrum` splits the single-strike sound exposure of the record
   into fractional-octave bands (the band energies sum back to the broadband
   `single_strike_sel` of [Underwater acoustics](underwater-acoustics.md), by
   Parseval);
2. `weighted_exposure` applies $W(f)$ band by band, sums the weighted energy,
   accumulates it over the number of strikes (the ISO 18406 $+10\lg N$) and
   compares the result with the criteria, unweighted peak SPL included.

```python
import numpy as np
from phonometry import underwater

# One recorded strike (here a synthetic 200 Hz decaying burst).
fs = 48_000
t = np.arange(int(0.2 * fs)) / fs
strike = 50.0 * np.exp(-t / 0.06) * np.sin(2 * np.pi * 200.0 * t)

spectrum = underwater.strike_sel_spectrum(strike, fs, fraction=3)
peak = underwater.peak_sound_pressure_level(strike)

res = underwater.weighted_exposure(
    spectrum.frequencies, spectrum.band_sel, "LF",
    guidance="nmfs-2024", impulsive=True, n_events=3000, peak_spl=peak,
)
print(res.unweighted_sel, res.weighted_sel, res.cumulative_sel)
print(res.sel_margin, res.peak_margin)       # positive means the criterion is exceeded
print(res.exceeds_injury, res.exceeds_tts)
res.plot()   # weighted spectrum against the criteria (needs matplotlib)
```

A 200 Hz hammer sits inside the low-frequency cetacean passband and far outside
the very high-frequency one, so the same campaign weights tens of decibels
lower for a porpoise than for a baleen whale, which is the whole point of
weighting, and the reason a single unweighted cumulative SEL is not an
assessment.

The natural companion is the sonar-equation chapter of
[Underwater sound propagation](underwater-propagation.md): the same figure of
merit machinery that gives a sonar detection range gives the range at which a
piling campaign's weighted exposure falls below a criterion.

## References

- National Marine Fisheries Service (2018). *2018 Revision to: Technical
  Guidance for Assessing the Effects of Anthropogenic Sound on Marine Mammal
  Hearing (Version 2.0)*. NOAA Technical Memorandum NMFS-OPR-59.
  [NOAA Fisheries](https://www.fisheries.noaa.gov/s3/2023-05/TECHMEMOGuidance508.pdf).
  The Table 3 weighting parameters, the Table ES3 PTS onset thresholds and the
  Appendix D worked example (W at 1 kHz for the five groups) used as the
  numeric oracle of section 3.
- National Marine Fisheries Service (2024). *2024 Update to: Technical Guidance
  for Assessing the Effects of Anthropogenic Sound on Marine Mammal Hearing
  (Version 3.0)*. NOAA Technical Memorandum NMFS-OPR-71.
  [NOAA Fisheries](https://www.fisheries.noaa.gov/s3/2024-11/Tech_Memo-Guidance_-3.0-_OCT-2024-508_OPR1.pdf).
  The current guidance and the default of `guidance`: Table 5 parameters,
  Table ES3 AUD INJ criteria and the Navy Phase 4 Table A.E-2 impulsive TTS.
- Southall, B. L., Finneran, J. J., Reichmuth, C., Nachtigall, P. E.,
  Ketten, D. R., Bowles, A. E., Ellison, W. T., Nowacek, D. P., &
  Tyack, P. L. (2019). Marine mammal noise exposure criteria: Updated
  scientific recommendations for residual hearing effects. *Aquatic Mammals*,
  45(2), 125-232.
  [doi:10.1578/AM.45.2.2019.125](https://doi.org/10.1578/AM.45.2.2019.125).
  The group audiograms of section 2 (Equation 1, Tables 2 to 4) and the
  weighting and threshold tables of sections 3 and 4 (Tables 5 to 7); the
  errata, *Aquatic Mammals* 45(5), 569-572,
  [doi:10.1578/AM.45.5.2019.569](https://doi.org/10.1578/AM.45.5.2019.569),
  corrects four peak-SPL values of Table 7.
- Finneran, J. J. (2016). *Auditory weighting functions and TTS/PTS exposure
  functions for marine mammals exposed to underwater noise*. Technical Report
  3026, SSC Pacific.
  The band-pass filter form and the audiogram equation both criteria sets
  adopt.
- Ainslie, M. A. (2010). *Principles of Sonar Performance Modelling*.
  Springer/Praxis.
  [doi:10.1007/978-3-540-87662-5](https://doi.org/10.1007/978-3-540-87662-5).
  The orca audiogram of section 2 (Equation 11.159, after Wensveen &
  Van Roij 2007) and the orca-versus-salmon worked example.
- ISO 18405:2017. *Underwater acoustics — Terminology*.
  [ISO page](https://www.iso.org/standard/62406.html).
  The symbols and references the guidance documents adopt for their criteria
  (Lp,0-pk and LE,p re 1 µPa and 1 µPa²·s).

## Standards & sources

- Auditory weighting and exposure criteria: NMFS (2024) v3.0 (default),
  NMFS (2018) v2.0 and Southall et al. (2019) with its published errata.
- Group audiograms: Southall et al. (2019), Equation (1) with Tables 2 and 3.
- Orca audiogram: Wensveen & Van Roij (2007) via Ainslie (2010) Eq. (11.159).
- Pile-driving sound exposure: ISO 18406:2017, via
  [Underwater acoustics](underwater-acoustics.md).
