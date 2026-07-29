← [Documentation index](README.md)

# Environmental Levels (ISO 1996-1/-2)

A community does not hear a single LAeq: it hears a day whose evenings and
nights matter more, a source whose tones or impulses annoy beyond their
energy, and a measurement taken over a residual background with a finite
confidence. This page is the regulatory assessment chain built on top of
the measured period levels: the whole-day descriptors $L_{den}$/$L_{dn}$
and the composite rating levels of ISO 1996-1, and the ISO 1996-2
determination procedures that make the reported number defensible: the
tonal adjustment, the residual-noise correction and the measurement
uncertainty budget.

The level-computation half of the topic, the $L_{eq}$/$L_{Aeq}$ integrals,
the percentile levels $L_N$, SEL and the noise dose that produce the period
levels this page consumes, is
[Integrated and Statistical Levels](levels.md); everything here assumes
those per-period values are already in hand.

## Environmental noise: Lden, Ldn and rating levels (ISO 1996-1)

Regulatory noise assessment weights evenings and nights more heavily.
`lden()` implements the day-evening-night level of ISO 1996-1:2016 (3.6.4:
+5 dB evening, +10 dB night, default 12/4/8 h periods, adjustable because
countries define them differently), `ldn()` the day-night variant (3.6.5),
and `composite_rating_level()` the general whole-day composite of clause 6.5
(Formulae 5-6) for arbitrary periods with source or character adjustments
(Table A.1: e.g. +5 dB regular impulsive, +12 dB highly impulsive, +3 to
+6 dB prominent tones):

```python
from phonometry import environmental

l = environmental.lden(63.2, 58.1, 51.4)                      # from LAeq per period
r = environmental.composite_rating_level([(63.2, 12, 0.0),    # day
                            (58.1, 4, 5.0),     # evening (+5)
                            (51.4, 8, 10.0)])   # night  (+10) == environmental.lden
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/lden_profile_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/lden_profile.svg" alt="Synthetic 24-hour urban LAeq profile with day, evening and night bands, the +5 and +10 dB weighted period levels and the resulting Lden" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import environmental

# Synthetic hourly LAeq of an urban road (dB), hours 00 to 23
laeq_h = np.array([48, 46, 45, 45, 46, 50, 56, 64, 66, 65, 63, 63,
                   64, 63, 63, 64, 65, 66, 65, 64, 63, 62, 61, 50], dtype=float)

def period_leq(idx):
    return 10 * np.log10(np.mean(10 ** (laeq_h[idx] / 10)))  # energy mean

ld = period_leq(np.arange(7, 19))                # day 07-19
le = period_leq(np.arange(19, 23))               # evening 19-23
ln_ = period_leq(np.r_[23, np.arange(0, 7)])     # night 23-07
l_den = environmental.lden(ld, le, ln_)
print(f"Lden = {l_den:.1f} dB")   # Lden = 64.3 dB

fig, ax = plt.subplots()
ax.axvspan(19, 23, color="C1", alpha=0.15)                                # evening
ax.axvspan(23, 24, color="C0", alpha=0.15); ax.axvspan(0, 7, color="C0", alpha=0.15)
ax.step(np.arange(25), np.r_[laeq_h, laeq_h[-1]], where="post",
        color="0.3", label="Hourly LAeq")
ax.hlines(ld, 7, 19, color="C2", linestyle="--", label="Lday (+0 dB)")
ax.hlines(le + 5, 19, 23, color="C1", linestyle="--", label="Levening + 5 dB")
ax.hlines([ln_ + 10, ln_ + 10], [23, 0], [24, 7], color="C0",
          linestyle="--", label="Lnight + 10 dB")
ax.hlines(l_den, 0, 24, color="C3", linewidth=2, label=f"Lden = {l_den:.1f} dB")
ax.set(xlabel="Hour of day", ylabel="Level [dB]", xlim=(0, 24))
ax.legend(loc="upper left", fontsize=8, ncol=2)
plt.show()
```

</details>

### `lden()` / `ldn()` / `composite_rating_level()` parameters

| Function | Key parameters | Notes |
| :--- | :--- | :--- |
| `lden(lday, levening, lnight, hours=(12, 4, 8))` | period LAeq values [dB]; `hours` must sum to 24 | +5 dB evening, +10 dB night (3.6.4) |
| `ldn(lday, lnight, hours=(15, 9))` | | +10 dB night (3.6.5) |
| `composite_rating_level(periods)` | iterable of `(level_db, hours, adjustment_db)`; hours positive, finite and summing to 24 | General Formulae (5)-(6); adjustments per Table A.1 |

Where you put the microphone changes the number: ISO 1996-2 fixes the receiver positions and their façade corrections. The diagram is measurement context; apply the corrections to your levels before analysis:

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_env_measurement_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_env_measurement.svg" alt="Environmental noise measurement positions per ISO 1996-2: free field, 2 m from the facade and flush-mounted, with their corrections" width="92%"></picture>

Combine with `laeq()` per time period to go from recordings to Lden. The
tonal adjustment itself is justified by the tonal audibility route of the next
section (fed, for the ISO/PAS 20065 method, by
[Objective audibility of tones in noise](tone-audibility.md));
the `tone_to_noise_ratio()` / `prominence_ratio()` verdicts of
[Prominent Discrete Tones](tone-prominence.md) are
complementary emission screening, not the Kt basis.

## Determining levels: tonal adjustment, residual noise and uncertainty (ISO 1996-2)

ISO 1996-2:2017 is the **determination** part: how the measured level is turned
into a rating level and reported with its uncertainty. The rating-level *summation*
and the time-of-day penalties live in ISO 1996-1 (above); ISO 1996-2 supplies the
tonal adjustment, the residual-noise correction and the uncertainty budget.

**Tonal adjustment (engineering method, Annex C).** From the energy-summed tone
level $L_{pt}$ and the masking-noise level $L_{pn}$ in the critical band around a
tone, the audibility above the masking threshold is
$\Delta L_{ta} = L_{pt} - L_{pn} + 2 + \lg[1 + (f_c/502)^{2.5}]$ dB (Formula (C.3)),
and the adjustment is $K_t = 0$ for $\Delta L_{ta} < 4$, $K_t = \Delta L_{ta} - 4$
for $4 \le \Delta L_{ta} \le 10$ and $K_t = 6$ above (Formulae (C.4)–(C.6)). The
critical bandwidth is 100 Hz up to 500 Hz and 20 % of $f_c$ above (Table C.1).
The one-third-octave **survey method** (`tonal_seeking_survey`) flags a band
exceeding both neighbours by 15/8/5 dB (low/mid/high), and
`tonal_adjustment_from_mean_audibility` maps the ISO/PAS 20065 mean audibility to
$K_t$ (Table J.1).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tonal_audibility_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tonal_audibility.svg" alt="ISO 1996-2 tonal adjustment Kt as a piecewise function of the tonal audibility: zero below 4 dB, rising linearly to 6 dB between 4 and 10 dB, and 6 dB above, with the four Annex C.5 worked examples and a mid-range tone marked" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import environmental

# ISO 1996-2:2007 Annex C.5, Example 2 (two tones near 400 Hz):
res = environmental.assess_tonal_audibility(tone_level=54.1, masking_noise_level=45.2,
                              centre_frequency=430.0)
print(res.audibility, res.adjustment)   # ΔLta ≈ 11.1 dB -> Kt = 6 dB
res.plot()
plt.show()
```
</details>

**Residual-noise correction (Clause 10.4).** `residual_sound_correction()`
applies $L = 10\lg(10^{L'/10} - 10^{L_\text{res}/10})$ (Formula (16)). With a
residual within 3 dB of the measured level no correction is allowed: the
*uncorrected* measured level $L'$ is then the reportable value, as an upper
bound of the specific sound (exposed as `reportable_upper_bound`, with
`reliable=False`). `gaussian_residual_level()` estimates the residual from
percentile levels (Annex I) and rejects inverted percentile orderings.

**Measurement uncertainty (Clause 4, Annex F).** `combined_standard_uncertainty()`
forms $u = \sqrt{\sum (c_j u_j)^2}$ (Formula (2)) and
`expanded_uncertainty()` applies $k = 2$ (95 %) or $k = 1.3$ (80 %);
`residual_correction_uncertainty()` carries the residual-correction sensitivity
(Formulae (F.7)/(F.8)) and `uncertainty_from_repeated_measurements()` the
repeated-measurement standard uncertainty: the primary energy-domain route
(Formulae (17)+(19)), with the level-domain Note 2 substitute (Formula (20))
reported alongside as `approximate_uncertainty` and a warning when the levels
spread beyond 3 dB, where the substitute grossly inflates.

```python
from phonometry import environmental

# Tonal adjustment for a prominent tone:
tonal = environmental.assess_tonal_audibility(54.1, 45.2, 430.0)  # TonalAssessmentResult
kt = tonal.adjustment                                             # 6 dB
tonal.plot()   # this audibility on the Kt curve, as in the figure above

# Subtract residual (background) noise from a measured level:
corr = environmental.residual_sound_correction(measured_level=58.0, residual_level=50.0)
corr.corrected_level, corr.reliable

# Combine an uncertainty budget and expand to 95 %:
u = environmental.combined_standard_uncertainty([0.59, 0.3, 2.0, 0.40, 0.38])  # 2.18 dB (G.2)
environmental.expanded_uncertainty(u)                            # 4.36 dB (k = 2)
```

## Quick answers

### What penalties does Lden apply to evening and night noise?

$L_{den}$, the day-evening-night level of ISO 1996-1:2016 (3.6.4), adds
+5 dB to the evening level and +10 dB to the night level before
energy-averaging the whole day, with default periods of 12, 4 and 8 hours,
adjustable because countries define them differently. The day-night variant
$L_{dn}$ (3.6.5) keeps only the +10 dB night penalty.

## See also

- [Integrated and Statistical Levels](levels.md): the Leq/LAeq, percentile
  and event levels the indicators of this page are assembled from.
- [Objective audibility of tones in noise](tone-audibility.md): the tonal
  audibility whose mean value maps to the Kt adjustment (Table J.1).
- [Prominent Discrete Tones](tone-prominence.md): the ECMA-418-1
  tone-to-noise and prominence-ratio verdicts, complementary emission
  screening for the tonal question.
- [Occupational Noise Exposure](occupational-exposure.md): the workplace
  counterpart, from task samples to the daily exposure level with its
  uncertainty budget.
- API reference: [`environmental.measurement`](https://jmrplens.github.io/phonometry/reference/api/environment/measurement/)
  and [`environmental.rating`](https://jmrplens.github.io/phonometry/reference/api/environment/rating/).

## References

- British Standards Institution. (2003). *Description and measurement of
  environmental noise — Guide to quantities and procedures* (BS 7445-1:2003).
  [BSI Knowledge](https://knowledge.bsigroup.com/products/description-and-measurement-of-environmental-noise-guide-to-quantities-and-procedures).
  The survey-practice companion of ISO 1996-1: which descriptor family fits
  which assessment question (BS 7445-2:1991 covers the land-use data
  acquisition).

## Standards

ISO 1996-1:2016, *Acoustics — Description, measurement and assessment of
environmental noise — Part 1: Basic quantities and assessment procedures*:
Lden (3.6.4), Ldn (3.6.5) and the composite whole-day rating level of
clause 6.5 (Formulae 5-6, Table A.1 adjustments). ISO 1996-2:2017,
*Acoustics — Description, measurement and assessment of environmental
noise — Part 2: Determination of sound pressure levels*: the Annex C tonal
adjustment, the Clause 10.4 residual-noise correction and the Clause 4 /
Annex F measurement uncertainty budget.
