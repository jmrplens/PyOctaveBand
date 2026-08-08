---
title: "Levels and weighting"
description: "From weighted signal to reported number: the IEC 61672-1 frequency weightings and the special-purpose curves, Fast/Slow/Impulse ballistics, the integrated, statistical and dose levels, and the ISO 1996 environmental indicators built on them."
---

A sound level meter does three things to a calibrated signal, in order: it
**weights it in frequency** to mimic the ear's sensitivity, it **smooths it in
time** with a standardised ballistic, and it **integrates it into a level**.
The pages of this section implement that chain stage by stage for the
displayed level: the A/C/Z curves and the Fast and Slow ballistics of
**IEC 61672-1:2013**, verified in CI against the standard's own tolerance
tables (Table 3 for the weightings, Table 4 for the tone-burst responses), plus
the legacy Impulse ballistics that IEC 61672-1 inherited from IEC 60651 and then
dropped from its requirements, kept here for older national procedures.

[Frequency Weighting (A, C, Z)](/phonometry/signals/levels/weighting/) covers the
first stage. The A-curve tracks hearing sensitivity at moderate levels and
dominates regulation; C is nearly flat and serves peaks and low-frequency
checks; Z is unweighted by definition. The rest of the family lives in
[Special Weightings (G, B, D, AU)](/phonometry/signals/levels/special-weightings/):
the G-curve of **ISO 7196** extends the idea into infrasound, where
conventional weightings are blind, the historical B and D curves serve legacy
data, and AU rejects ultrasound from an audible-sound reading per IEC 61012.

[Time Weighting](/phonometry/signals/levels/time-weighting/) covers the second stage:
the exponential Fast (125 ms) and Slow (1 s) ballistics that decide how quickly
a displayed level follows the sound, and the legacy asymmetric Impulse
ballistics (35 ms rise, 1.5 s decay) that came from IEC 60651 and is no longer
required by IEC 61672-1. phonometry implements the exact time constants,
verified against the tone-burst responses of the standard.

[Integrated and Statistical Levels](/phonometry/signals/levels/levels/) is the payoff:
the equivalent continuous level Leq and its A-weighted LAeq, the percentile
levels L10/L50/L90 that describe fluctuating noise, LCpeak and SEL, the noise
dose of IEC 61252, plus the octave spectrogram for visualising level against
time and band at once. This is the page where most practical measurements
end, and where the environmental and occupational sections pick up.

Turning those levels into a regulatory verdict is
[Environmental Levels (ISO 1996-1/-2)](/phonometry/environment/assessment/environmental-levels/):
the day-evening-night level Lden and the rating levels of **ISO 1996-1** with
their adjustments, and the ISO 1996-2 determination chain of tonal
adjustment, residual-noise correction and the measurement uncertainty budget.

National frameworks build their own index on top of that chain, and
[Spanish Noise Regulation (RD 1367/2007)](/phonometry/environment/assessment/spanish-noise-regulation/)
implements the Spanish one: the corrected level LKeq with its tonal,
low-frequency and impulsive corrections, the evaluation periods split into
noise phases, and the limit tables an activity is judged against.

## Pages in this section

- [Frequency Weighting (A, C, Z)](/phonometry/signals/levels/weighting/): the
  IEC 61672-1 A/C/Z curves, the high-frequency accuracy mode and the class
  verification.
- [Special Weightings (G, B, D, AU)](/phonometry/signals/levels/special-weightings/):
  the ISO 7196 infrasound G-weighting, the historical B and D curves and AU
  per IEC 61012.
- [Time Weighting](/phonometry/signals/levels/time-weighting/): the Fast and Slow
  exponential ballistics of IEC 61672-1, and the legacy Impulse ballistics it
  dropped.
- [Integrated and Statistical Levels](/phonometry/signals/levels/levels/): Leq and
  LAeq, percentile levels, LCpeak/SEL, noise dose and octave spectrograms.

## See also

Pages elsewhere on the site that this section leans on:

- [Environmental Levels (ISO 1996-1/-2)](/phonometry/environment/assessment/environmental-levels/):
  Lden, Ldn and rating levels, tonal adjustment, residual noise and
  uncertainty.
- [Spanish Noise Regulation (RD 1367/2007)](/phonometry/environment/assessment/spanish-noise-regulation/):
  the corrected level LKeq, the Kt/Kf/Ki corrections, the evaluation periods
  and noise phases, and the limit tables.

## What this section does not cover

These pages implement the signal processing of a sound level meter, not the
meter. The rest of IEC 61672-1 — level ranges, overload indication, the
self-generated noise floor, the directional response and the IEC 61672-3
periodic tests — is not implemented anywhere in the library, so nothing here
assigns a class to a physical instrument; [Build a sound level
meter](/phonometry/signals/sound-level-meter/) states exactly what a class
verdict from the library does and does not mean. Two curves come without a
verdict of any kind: ISO 7196 defines a single ±1 dB tolerance for G with no
class structure, and the withdrawn IEC 537 left no tolerance table behind for
D, so both curves filter a signal but neither reaches
`verify_weighting_class`, and they are pinned against their published tables in
the conformance report instead. The noise dose is the 1993
first edition of IEC 61252 only, not the 2025 revision. And a dose is a
quantity, not a verdict: the exposure strategies, the sampling plan and the
limits that decide whether a worker is over-exposed are [Occupational exposure
(ISO 9612)](/phonometry/perception/hearing/occupational-exposure/).
