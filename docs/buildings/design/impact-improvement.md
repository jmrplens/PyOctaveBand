← [Documentation index](../../README.md)

# Floor-Covering Impact Improvement (ISO 16251-1)

A soft floor covering does not block airborne sound; it cushions footsteps.
What it improves is the impact level of the floor underneath, and measuring
that improvement in a full ISO 10140 transmission suite is out of proportion
to a square metre of carpet. ISO 16251-1 therefore shrinks the laboratory to
a small heavyweight mock-up: a softly supported concrete plate, a standard
tapping machine and an accelerometer underneath. This guide covers the
mock-up measurement of the improvement $\Delta L$ and its weighted
$\Delta L_\mathrm{w}$ against the ISO 717-2 reference floor, the accredited fiche,
and the ISO 12354-2 engineering estimate that predicts a floating floor's
improvement from the dynamic stiffness of its resilient layer. The
full-size laboratory chain lives in
[Laboratory Insulation Measurement](../insulation/insulation-lab.md); the $s'$
measurement behind the estimate is the subject of
[Dynamic stiffness of resilient materials (EN 29052-1)](../../materials/resilient/dynamic-stiffness.md).

## The small-mock-up method

ISO 16251-1:2014 is a laboratory method for the **improvement of impact sound
insulation** $\Delta L$ of a soft, locally-reacting floor covering (carpet, PVC,
linoleum). The two ISO 10140 rooms are replaced by a small softly-supported
concrete plate; a standard tapping machine excites it and the structure-borne
**acceleration level** on the underside is measured with and without the
covering. For locally-reacting coverings that acceleration-level difference
equals the ISO 10140 impact sound reduction.

**Acceleration level (Formula (1)).** $L_a = 10\log_{10}(\langle a^2\rangle / a_0^2)$ dB,
reference $a_0 = 10^{-6}\ \text{m/s}^2$. **Background correction (Formula (2))**
follows the ISO 10140 three-branch rule (unchanged ≥ 15 dB; energy subtraction
for 6 ≤ margin < 15 dB; the 1.3 dB limit below 6 dB, flagged as $> \Delta L$).
The improvement is the position-averaged difference
$\Delta L = L_0 - L_1$ (Formulae (3)/(4)); octaves follow
$\Delta L_\mathrm{oct} = -10\log_{10}[\tfrac{1}{3}\sum 10^{-\Delta L_j/10}]$ (Formula (5)).

**Weighted improvement.** $\Delta L_\mathrm{w}$ is the ISO 717-2 weighted reduction: the
improvement is applied to the heavyweight **reference floor** $L_\mathrm{n,r,0}$
(ISO 717-2 Table 4), $L_\mathrm{n,r} = L_\mathrm{n,r,0} - \Delta L$, and
$\Delta L_\mathrm{w} = 78 - L_\mathrm{n,r,w}$, computed by `weighted_impact_improvement()`, which
reuses the verified ISO 717-2 rating engine. A clause 6.3 measurement spans 18
bands (100–5000 Hz, optionally extended to 50 Hz); the rating is formed on the
100–3150 Hz sub-range of whatever spectrum contains it. The statement of
results (clause 8 e)) also carries the spectrum adaptation term
$C_{\mathrm{I},\Delta} = C_\mathrm{I,r,0} - C_\mathrm{I,r}$ (ISO 717-2:2020 Formula (A.4)), exposed as
`ci_delta` on the result and standalone as
`impact_improvement_adaptation_term()`.

The worked example below is a real measurement: the improvement of a textile
carpet on the CSTB heavyweight mock-up, digitized from Figure 4 of Foret,
Chéné and Guigou-Carter, "A comparison of the reduction of transmitted impact
noise by floor coverings measured using ISO 140-8 and ISO/CD 16251-1" (Forum
Acusticum 2011, Aalborg). Its published ISO 16251-1 weighted improvement is
$\Delta L_\mathrm{w} = 29$ dB, reproduced exactly by the rating engine.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/floor_covering_improvement_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/floor_covering_improvement.svg" alt="ISO 16251-1 floor-covering impact sound improvement: the improvement delta-L of a real textile carpet rising with frequency across one-third-octave bands from 100 Hz to 3150 Hz, with the shaded improvement area and the weighted single-number delta-Lw annotated" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

freqs = [100, 125, 160, 200, 250, 315, 400, 500,
         630, 800, 1000, 1250, 1600, 2000, 2500, 3150]
bare = np.full(16, 78.0)                       # bare-plate acceleration level
# A real textile carpet measured on the CSTB mock-up (Foret et al. 2011, Fig. 4).
covering = bare - np.array([5, 8, 10, 14, 18, 23, 30, 31,
                            39, 49, 53, 57, 60, 67, 68, 71])
res = building.impact_improvement(bare, covering, freqs)
print(res.delta_lw)   # weighted improvement delta-Lw = 29 dB (ISO 717-2)
res.plot()
plt.show()
```

</details>

```python
from phonometry import building

# delta-Lw straight from an improvement spectrum (16 one-third-octave bands):
delta_l = [5, 8, 10, 14, 18, 23, 30, 31, 39, 49, 53, 57, 60, 67, 68, 71]
print(building.weighted_impact_improvement(delta_l))    # 29 dB (carpet)

# From the measured bare/covered acceleration levels, with a background trace:
freqs = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150]
bare_levels = [72, 73, 74, 74, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79, 80, 80]
covered_levels = [b - d for b, d in zip(bare_levels, delta_l)]
bg = [40.0] * 16
res = building.impact_improvement(bare_levels, covered_levels, freqs, background=bg)
res.improvement       # delta-L per band
res.delta_lw          # weighted single number (rated on the 100-3150 Hz sub-range)
res.ci_delta          # spectrum adaptation term CI,delta (Formula (A.4))
res.limited           # bands at the 1.3 dB limit of measurement (> delta-L)
res.octave_bands()    # (octave freqs, delta-L_oct) via Formula (5)
res.plot()            # the delta-L(f) improvement spectrum above (needs matplotlib)
```

## ISO 16251-1 impact-improvement report (`.report()`)

`FloorCoveringImprovementResult.report()` writes a one-page accredited
impact-improvement fiche: the ISO 16251-1 basis line, a metadata header, the
per-band table (frequency and $\Delta L$, bands at the 1.3 dB limit prefixed
`>`) beside the $\Delta L(f)$ improvement curve, the boxed single-number
$\Delta L_\mathrm{w}\ (C_{\mathrm{I},\Delta})$ (the ISO 16251-1 Clause 8 e) statement of
results, rated
per ISO 717-2) and a footer. The applicable `ReportMetadata` fields are
`specimen` (the floor covering under test), `client`, `manufacturer`,
`mounting`, `mass_per_area`, `test_room`, `test_date`, `temperature`,
`pressure`, `measurement_standard`, `laboratory`, `operator`, `report_id`,
`notes` and `requirement` (a higher weighted improvement is better, so the
verdict passes at or above it). The bare reference floor is the standardised
heavyweight floor of ISO 717-2:2020 Table 4, fixed by the standard.
`verbose=True` adds the reference-floor-with-covering column
$L_\mathrm{n,r} = L_\mathrm{n,r,0} - \Delta L$, the derivation basis of $\Delta L_\mathrm{w}$.

```python
from phonometry import building, ReportMetadata

freqs = [100, 125, 160, 200, 250, 315, 400, 500,
         630, 800, 1000, 1250, 1600, 2000, 2500, 3150]
delta_l = [5, 8, 10, 14, 18, 23, 30, 31, 39, 49, 53, 57, 60, 67, 68, 71]
bare = [78.0] * 16
res = building.impact_improvement(bare, [b - d for b, d in zip(bare, delta_l)], freqs)
res.report("dLw.pdf",
           metadata=ReportMetadata(
               specimen="Textile floor covering (carpet), laid loose",
               measurement_standard="ISO 16251-1",
               requirement=20.0))  # delta-Lw (CI,delta) = 29 (-13) dB
```

A rendered example fiche, regenerated with `make reports`, is kept in the
repository. Click the preview to open the PDF:

[![Floor-covering impact improvement ISO 16251-1 example report: metadata header, one-third-octave delta-L table beside the delta-L(f) improvement curve, boxed delta-Lw (CI,delta) = 29 (-13) dB weighted improvement (ISO 717-2) and a PASS verdict against the 20 dB requirement](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso16251_floor_covering_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso16251_floor_covering_example.pdf)

## From dynamic stiffness to the improvement: the ISO 12354-2 estimate

The mock-up measures a *covering* that exists; a **floating floor** (a
screed poured on a resilient layer) is designed the other way round, from
two numbers known at the drawing stage: the mass per unit area $m'$ of the
slab and the dynamic stiffness per unit area $s'$ of the resilient layer,
measured per EN 29052-1 as in the
[dynamic-stiffness guide](../../materials/resilient/dynamic-stiffness.md). Together they set the mass-spring
resonance of the system (ISO 12354-2:2017, Annex C, Formula (C.2)):

$$
f_0 = 160 \sqrt{\frac{s'}{m'}}\ \text{Hz}
\qquad (s'\ \text{in MN/m}^3,\ m'\ \text{in kg/m}^2),
$$

where 160 is the rounded $1000/2\pi$; `materials.natural_frequency` computes
the exact $\tfrac{1}{2\pi}\sqrt{s'/m'}$ form. Above the resonance the slab
decouples from the structural floor and Annex C estimates the improvement as
a straight slope per Formula (C.1), with the steeper Formula (C.3) for
constructions whose higher internal losses follow the infinite-plate theory:

$$
\Delta L = 30 \log_{10} \frac{f}{f_0}\ \text{dB (sand/cement, calcium-sulfate screeds)},
\qquad
\Delta L = 40 \log_{10} \frac{f}{f_0}\ \text{dB (asphalt, dry screeds)}.
$$

```python
import numpy as np
from phonometry import building, materials

# The ISO 12354-2:2017 Annex G worked floor: a 73.5 kg/m2 screed on a resilient
# layer with s' = 8 MN/m3.
f0 = 160.0 * np.sqrt(8.0 / 73.5)                                 # Formula (C.2)
print(round(f0, 1))                                              # 52.8  Hz
print(round(float(materials.natural_frequency(8.0e6, 73.5)), 1))  # 52.5  exact 1/(2 pi) form

# delta-L = 30 lg(f/f0) above the resonance (Formula (C.1)); build the
# one-third-octave spectrum and rate it with the ISO 717-2 engine.
freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
delta_l = 30.0 * np.log10(freqs / f0)
print(building.weighted_impact_improvement(delta_l))             # 32  dB

# The one-line estimate of Formula (C.4) lands right beside it:
print(round(13.0 * np.log10(73.5) - 14.2 * np.log10(8.0) + 20.8, 1))   # 32.2 dB
```

The closed-form weighted estimate of Formula (C.4),
$\Delta L_\mathrm{w} = 13\log_{10} m' - 14{,}2\log_{10} s' + 20{,}8$ dB, condenses the same
physics into one line: heavier slabs and softer layers rate better. Both
estimates are design aids kept deliberately on the safe side (the 30 lg
slope undercuts the 40 lg infinite-plate theory where experimental data say
real screeds fall short); once a specimen exists, the measured mock-up of
this page, or the full ISO 10140-3 floor, is the reference, and the
[EN 12354-2 prediction](insulation-prediction.md) consumes whichever $\Delta L_\mathrm{w}$ you have.

## References

- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  The transmission theory of floors and floating floors that the improvement
  quantifies.
- Foret, R., Chéné, J.-B., & Guigou-Carter, C. (2011). A comparison of the
  reduction of transmitted impact noise by floor coverings measured using
  ISO 140-8 and ISO/CD 16251-1. *Forum Acusticum 2011, Aalborg* (CSTB).
  The measured textile-carpet improvement spectrum used as the worked
  example on this page ($\Delta L_\mathrm{w} = 29$ dB); the per-band $\Delta L$ was
  digitized from its
  Figure 4.
- International Organization for Standardization. (2014). *Acoustics —
  Laboratory measurement of the reduction of transmitted impact noise by
  floor coverings on a small floor mock-up — Part 1: Heavyweight compact
  floor* (ISO 16251-1:2014).
  [iso.org catalogue](https://www.iso.org/standard/56017.html).
  The small-mock-up method this page implements.
- International Organization for Standardization. (2020). *Acoustics —
  Rating of sound insulation in buildings and of building elements — Part 2:
  Impact sound insulation* (ISO 717-2:2020).
  [iso.org catalogue](https://www.iso.org/standard/69867.html).
  The reference floor $L_\mathrm{n,r,0}$ (Table 4) and the rating engine behind
  $\Delta L_\mathrm{w}$, with the Formula (A.4) adaptation term $C_{\mathrm{I},\Delta}$.

- International Organization for Standardization. (2017). *Building acoustics
  — Estimation of acoustic performance of buildings from the performance of
  elements — Part 2: Impact sound insulation between rooms*
  (ISO 12354-2:2017).
  [iso.org catalogue](https://www.iso.org/standard/70239.html).
  The informative Annex C floating-floor estimate reproduced above (Formulae
  C.1 to C.4) and the worked floor of Annex G.

## Standards

ISO 16251-1:2014, which specifies the small-mock-up laboratory method for
the impact-sound improvement $\Delta L$ of floor coverings; ISO 717-2:2020,
which supplies the reference floor, the $\Delta L_\mathrm{w}$ rating and the
$C_{\mathrm{I},\Delta}$ adaptation term; and ISO 12354-2:2017 Annex C (informative),
whose floating-floor estimate this page reproduces from the EN 29052-1
dynamic stiffness, with the worked floor of its Annex G (context for design,
not a measurement).

**Not covered.** The mock-up facility itself — the plate dimensions, the
resilient supports, the tapping-machine positions of Clause 5 — is not checked:
the functions consume measured acceleration levels wherever they came from. The
ISO 12354-2:2017 Annex C floating-floor formulae are quoted here as design aids
and evaluated inline; the helpers that wrap them, the tapping-machine force
model behind a soft covering's improvement and the **ISO 12354-1** Annex D
rating of a wall lining belong to
[Predicting resilient-layer performance](resilient-layers.md). The superseded
full-size ISO 140-8 method survives only as the comparison axis of the
Foret et al. (2011) worked example.

## See also

- [Laboratory Insulation Measurement](../insulation/insulation-lab.md): the ISO 10140
  suite this mock-up replaces for soft coverings, and the full-size
  ISO 10140-3 improvement measurement.
- [Dynamic stiffness of resilient materials (EN 29052-1)](../../materials/resilient/dynamic-stiffness.md):
  the $s'$ measurement that feeds the floating-floor estimate.
- [Predicting Resilient-Layer Performance](resilient-layers.md): the
  prediction counterpart of this page, from the tapping-machine force model
  to floating floors and wall linings.
- [Predicting Sound Insulation (EN 12354)](insulation-prediction.md): the
  impact model whose Formula (21) consumes $\Delta L_\mathrm{w}$.
- [Insulation Ratings (ISO 717)](../insulation/insulation-ratings.md): the reference-curve
  engine behind the weighted improvement.
- API reference: [`building.measurement.floor_covering_improvement`](https://jmrplens.github.io/phonometry/reference/api/building/floor-covering-improvement/).
